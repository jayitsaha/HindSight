"""Vision-language model answering for SuperMemory-VQA (written by Veer Chheda).

hindsight.vlm.pipeline.VLMPipeline  -> one entry point: question + frames/video -> VLMAnswerOutput
hindsight.vlm.hf_vlm                -> Hugging Face backend (Gemma 4, SmolVLM, Pixtral, ...)
hindsight.vlm.mock_vlm              -> deterministic backend for tests
hindsight.vlm.config                -> VLMConfig / DataConfig / EvalConfig dataclasses

Kept import-light on purpose: importing the package does not pull in torch or transformers.
"""
