"""
Open-Source Hugging Face VLM backend (Pixtral, SmolVLM, Llama-Vision, Idefics3, etc.).
Designed to run on HPC GPU clusters (PACE H100 / H200 / A100) or local CUDA environments.
"""

from typing import Any, List, Optional, Tuple, Union
import time
import os
from PIL import Image

from hindsight.vlm.base import BaseVLM
from hindsight.data.schema import QuestionRecord, VLMAnswerOutput
from hindsight.vlm.prompt import SYSTEM_PROMPT_QA, build_qa_prompt, parse_vlm_response


GEMMA4_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def _is_gemma4(name: str) -> bool:
    return "gemma-4" in name.lower() or "gemma4" in name.lower()


class HuggingFaceVLM(BaseVLM):
    """
    Open-Source VLM backend powered by Hugging Face transformers.
    Supports top-tier vision-language models for H100 / H200:
    - mistralai/Pixtral-12B-2409 (Native high-res multi-image/video, Mistral AI - France)
    - HuggingFaceTB/SmolVLM-Instruct (Fast, lightweight, open weights - Hugging Face)
    - meta-llama/Llama-3.2-11B-Vision-Instruct (Meta AI - USA)
    - HuggingFaceM4/Idefics3-8B-Llama (Hugging Face - USA/France)
    """

    def __init__(
        self,
        model_name_or_path: str = "mistralai/Pixtral-12B-2409",
        device: str = "auto",
        load_in_4bit: bool = False,
        torch_dtype: Optional[str] = "bfloat16",
        max_frames: Optional[int] = 16,
        max_new_tokens: int = 512,
        temperature: float = 0.01,
        image_token_budget: Optional[int] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if image_token_budget is not None and image_token_budget not in GEMMA4_TOKEN_BUDGETS:
            raise ValueError(f"image_token_budget must be one of {GEMMA4_TOKEN_BUDGETS}, got {image_token_budget}")
        self.image_token_budget = image_token_budget
        self._logged_prompt_len = False

        self._model = None
        self._processor = None
        self._is_loaded = False

    def _lazy_load_model(self):
        """Loads model and processor on first call."""
        if self._is_loaded:
            return

        is_gemma4 = _is_gemma4(self.model_name_or_path)

        try:
            import torch
            from transformers import AutoProcessor
            if is_gemma4:
                # Gemma 4 uses Transformers' multimodal auto class and is only
                # available in recent Transformers 5.x releases.
                from transformers import AutoModelForMultimodalLM
                auto_model_cls = AutoModelForMultimodalLM
            else:
                try:
                    # Transformers 5 replaces AutoModelForVision2Seq.
                    from transformers import AutoModelForImageTextToText
                    auto_model_cls = AutoModelForImageTextToText
                except ImportError:
                    from transformers import AutoModelForVision2Seq
                    auto_model_cls = AutoModelForVision2Seq
        except ImportError as e:
            raise ImportError(
                "PyTorch and Transformers are required to run HuggingFaceVLM. "
                "Please install a compatible PyTorch/Transformers environment. "
                "Gemma 4 requires Transformers 5.x and AutoModelForMultimodalLM."
            ) from e

        # Determine dtype
        dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" and torch.cuda.is_available() else torch.float16
        if not torch.cuda.is_available() and self.device != "mps":
            dtype = torch.float32

        # Device mapping
        device_map = "auto" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device_map == "auto" and not torch.cuda.is_available():
            device_map = "cpu"

        print(f"[HuggingFaceVLM] Loading {self.model_name_or_path} (device_map={device_map}, 4bit={self.load_in_4bit})...")

        # Select model class based on model architecture
        lower_name = self.model_name_or_path.lower()
        if is_gemma4:
            model_cls = auto_model_cls
        elif "llama-3.2" in lower_name and "vision" in lower_name:
            try:
                from transformers import MllamaForConditionalGeneration
                model_cls = MllamaForConditionalGeneration
            except ImportError:
                model_cls = auto_model_cls
        elif "pixtral" in lower_name or "llava" in lower_name:
            try:
                from transformers import LlavaForConditionalGeneration
                model_cls = LlavaForConditionalGeneration
            except ImportError:
                model_cls = auto_model_cls
        else:
            model_cls = auto_model_cls

        kwargs = {"device_map": device_map}
        kwargs["dtype" if is_gemma4 else "torch_dtype"] = dtype
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True

        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        if is_gemma4 and self.image_token_budget is not None:
            # Gemma 4: max_soft_tokens on the image processor sets tokens per image, and the processor's
            # image_seq_length must match for placeholder expansion. The per-call kwarg is passed in
            # predict() as well; the prompt length logged on the first call shows whether it took effect.
            if hasattr(self._processor, "image_processor"):
                self._processor.image_processor.max_soft_tokens = self.image_token_budget
            if hasattr(self._processor, "image_seq_length"):
                self._processor.image_seq_length = self.image_token_budget
        self._model = model_cls.from_pretrained(self.model_name_or_path, **kwargs)
        self._is_loaded = True
        print("[HuggingFaceVLM] Model loaded successfully.")

    def _extract_frames_from_video(self, video_path: str, num_frames: Optional[int]) -> List[Image.Image]:
        """Samples evenly spaced frames from a video file (all frames if num_frames is None).

        Streams the video and keeps only the chosen frames, so a multi-hour session does not have to
        fit in RAM; a K-frame curated mp4 is read whole.
        """
        try:
            import cv2
        except ImportError as e:
            raise ImportError("opencv-python(-headless) is required to read video files") from e
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or total <= 0:
            cap.release()
            raise ValueError(f"Could not read video frames from {video_path}")
        n = total if not num_frames else min(num_frames, total)
        wanted = sorted({int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)})
        frames, nxt, i = [], 0, 0
        while nxt < len(wanted) and cap.grab():      # grab() skips decoding of unused frames
            if i == wanted[nxt]:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                nxt += 1
            i += 1
        cap.release()
        if not frames:
            raise ValueError(f"Could not read video frames from {video_path}")
        return frames

    def _prepare_visual_inputs(
        self,
        video_input: Union[str, List[str], List[Image.Image], Image.Image]
    ) -> List[Image.Image]:
        """Normalizes video input into a list of PIL Images."""
        return self._prepare_labelled_inputs(video_input)[0]

    def _prepare_labelled_inputs(self, video_input) -> Tuple[List[Image.Image], List[Optional[str]]]:
        """Normalizes video input into (images, labels). List items may be image paths, PIL Images,
        or (label, image) pairs such as hindsight.evaluation.curated produces; labels are None otherwise."""
        if isinstance(video_input, str):
            if os.path.isfile(video_input):
                images = self._extract_frames_from_video(video_input, self.max_frames)
                return images, [None] * len(images)
            raise FileNotFoundError(f"Video file not found: {video_input}")
        if isinstance(video_input, Image.Image):
            return [video_input.convert("RGB")], [None]
        if not isinstance(video_input, list):
            raise ValueError(f"Unsupported visual input type: {type(video_input)}")
        images, labels = [], []
        for item in video_input:
            label = None
            if isinstance(item, tuple):
                label, item = item
            if isinstance(item, str):
                item = Image.open(item)
            if isinstance(item, Image.Image):
                images.append(item.convert("RGB"))
                labels.append(label)
        # Subsample if too many frames (max_frames=None keeps all, e.g. curated K-frame inputs)
        if self.max_frames and len(images) > self.max_frames:
            step = len(images) / self.max_frames
            indices = [int(i * step) for i in range(self.max_frames)]
            images, labels = [images[i] for i in indices], [labels[i] for i in indices]
        return images, labels

    @staticmethod
    def _image_content(frames: List[Image.Image], labels: List[Optional[str]]) -> List[dict]:
        content = []
        for img, label in zip(frames, labels):
            if label:
                content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": img})
        return content

    def predict(
        self,
        video_input: Union[str, List[str], List[Image.Image], Image.Image],
        question: QuestionRecord,
        **kwargs: Any
    ) -> VLMAnswerOutput:
        self._lazy_load_model()
        import torch

        start_time = time.time()
        frames, labels = self._prepare_labelled_inputs(video_input)
        text_only = isinstance(video_input, list) and len(video_input) == 0  # explicit no-video control
        if not frames and not text_only:
            raise ValueError("No visual frames could be extracted from input.")

        # Build prompt
        formatted_options = question.formatted_options()
        user_prompt = build_qa_prompt(question.question, formatted_options, has_frames=bool(frames),
                                      labelled=any(labels))

        # Gemma 4 supports a native system role and expects its multimodal chat
        # template to tokenize image/text messages together. Keep the existing
        # sampled-frame path so the number of video frames stays bounded.
        if _is_gemma4(self.model_name_or_path):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_QA},
                {
                    "role": "user",
                    "content": self._image_content(frames, labels) + [{"type": "text", "text": user_prompt}],
                },
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
                **({"max_soft_tokens": self.image_token_budget} if self.image_token_budget else {}),
            )
        else:
            # Build chat messages format for vision-language models
            combined_text = f"{SYSTEM_PROMPT_QA}\n\n{user_prompt}"
            content = self._image_content(frames, labels)
            content.append({"type": "text", "text": combined_text})

            messages = [{"role": "user", "content": content}]

            # Use processor chat template, with compatibility fallbacks for
            # models whose templates require a separate system role.
            try:
                prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self._processor(
                    text=[prompt_text],
                    images=frames or None,
                    padding=True,
                    return_tensors="pt"
                )
            except Exception:
                try:
                    alt_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_QA},
                        {"role": "user", "content": self._image_content(frames, labels) + [{"type": "text", "text": user_prompt}]}
                    ]
                    prompt_text = self._processor.apply_chat_template(alt_messages, tokenize=False, add_generation_prompt=True)
                    inputs = self._processor(
                        text=[prompt_text],
                        images=frames or None,
                        padding=True,
                        return_tensors="pt"
                    )
                except Exception:
                    inputs = self._processor(
                        text=[combined_text],
                        images=frames or None,
                        return_tensors="pt"
                    )

        if not self._logged_prompt_len and "input_ids" in inputs:
            print(f"[HuggingFaceVLM] first prompt: {len(frames)} frames -> {inputs['input_ids'].shape[1]} tokens "
                  f"(token budget per frame: {self.image_token_budget or 'default'})")
            self._logged_prompt_len = True

        # Move to model device
        inputs = {k: v.to(self._model.device) for k, v in inputs.items() if hasattr(v, "to")}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None
            )

        # Trim input tokens
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_ids_trimmed = generated_ids[:, input_len:]
        response_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Parse structured answer
        letter, idx, text, reasoning, abstain = parse_vlm_response(response_text, question.choices)
        latency = time.time() - start_time

        return VLMAnswerOutput(
            question_id=question.question_id,
            predicted_choice=letter,
            predicted_choice_idx=idx,
            predicted_text=text,
            reasoning=reasoning,
            abstain=abstain,
            confidence=None,  # not calibrated yet; option log-probs would go here
            raw_model_response=response_text,
            latency_seconds=latency
        )
