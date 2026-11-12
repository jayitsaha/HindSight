"""
Open-Source Hugging Face VLM backend (Pixtral, SmolVLM, Llama-Vision, Idefics3, etc.).
Designed to run on HPC GPU clusters (PACE H100 / H200 / A100) or local CUDA environments.
"""

from typing import Any, List, Optional, Union
import time
import os
from PIL import Image

from hindsight.vlm.base import BaseVLM
from hindsight.data.schema import QuestionRecord, VLMAnswerOutput
from hindsight.vlm.prompt import SYSTEM_PROMPT_QA, build_qa_prompt, parse_vlm_response


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
        max_frames: int = 16,
        max_new_tokens: int = 512,
        temperature: float = 0.01,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model = None
        self._processor = None
        self._is_loaded = False

    def _lazy_load_model(self):
        """Loads model and processor on first call."""
        if self._is_loaded:
            return

        is_gemma4 = "gemma-4" in self.model_name_or_path.lower() or "gemma4" in self.model_name_or_path.lower()

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
        self._model = model_cls.from_pretrained(self.model_name_or_path, **kwargs)
        self._is_loaded = True
        print("[HuggingFaceVLM] Model loaded successfully.")

    def _extract_frames_from_video(self, video_path: str, num_frames: int) -> List[Image.Image]:
        """Samples evenly spaced frames from a video file."""
        frames = []
        try:
            import av

            with av.open(video_path) as container:
                stream = container.streams.video[0]
                all_frames = [f.to_image() for f in container.decode(stream)]
            if not all_frames:
                raise ValueError(f"Could not read video frames from {video_path}")
            indices = [int(i * (len(all_frames) - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
            frames = [all_frames[i] for i in indices]
        except Exception as av_error:
            # Fall back to OpenCV if PyAV is absent or cannot decode this file.
            try:
                import cv2

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    cap.release()
                    raise ValueError(f"Could not read video frames from {video_path}")
                indices = [int(i * (total_frames - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                cap.release()
                if not frames:
                    raise ValueError(f"Could not read video frames from {video_path}")
            except Exception as cv_error:
                raise RuntimeError(
                    f"Could not decode video with PyAV ({av_error}) or OpenCV ({cv_error}). "
                    "Install a compatible `opencv-python` or `av` package."
                ) from cv_error
        return frames

    def _prepare_visual_inputs(
        self,
        video_input: Union[str, List[str], List[Image.Image], Image.Image]
    ) -> List[Image.Image]:
        """Normalizes video input into a list of PIL Images."""
        if isinstance(video_input, str):
            if os.path.isfile(video_input):
                return self._extract_frames_from_video(video_input, self.max_frames)
            else:
                raise FileNotFoundError(f"Video file not found: {video_input}")
        elif isinstance(video_input, list):
            images = []
            for item in video_input:
                if isinstance(item, str):
                    images.append(Image.open(item).convert("RGB"))
                elif isinstance(item, Image.Image):
                    images.append(item.convert("RGB"))
            # Subsample if too many frames
            if len(images) > self.max_frames:
                step = len(images) / self.max_frames
                indices = [int(i * step) for i in range(self.max_frames)]
                images = [images[i] for i in indices]
            return images
        elif isinstance(video_input, Image.Image):
            return [video_input.convert("RGB")]
        else:
            raise ValueError(f"Unsupported visual input type: {type(video_input)}")

    def predict(
        self,
        video_input: Union[str, List[str], List[Image.Image], Image.Image],
        question: QuestionRecord,
        **kwargs: Any
    ) -> VLMAnswerOutput:
        self._lazy_load_model()
        import torch

        start_time = time.time()
        frames = self._prepare_visual_inputs(video_input)
        text_only = isinstance(video_input, list) and len(video_input) == 0  # explicit no-video control
        if not frames and not text_only:
            raise ValueError("No visual frames could be extracted from input.")

        # Build prompt
        formatted_options = question.formatted_options()
        user_prompt = build_qa_prompt(question.question, formatted_options, has_frames=bool(frames))

        # Gemma 4 supports a native system role and expects its multimodal chat
        # template to tokenize image/text messages together. Keep the existing
        # sampled-frame path so the number of video frames stays bounded.
        if "gemma-4" in self.model_name_or_path.lower() or "gemma4" in self.model_name_or_path.lower():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_QA},
                {
                    "role": "user",
                    "content": ([{"type": "image", "image": img} for img in frames]
                                + [{"type": "text", "text": user_prompt}]),
                },
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        else:
            # Build chat messages format for vision-language models
            combined_text = f"{SYSTEM_PROMPT_QA}\n\n{user_prompt}"
            content = [{"type": "image", "image": img} for img in frames]
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
                        {"role": "user", "content": [{"type": "image", "image": img} for img in frames] + [{"type": "text", "text": user_prompt}]}
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
            confidence=0.85,
            raw_model_response=response_text,
            latency_seconds=latency
        )
