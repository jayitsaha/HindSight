"""Dataset helpers for SuperMemory-VQA.

supermemory  annotation helpers used by frame curation (load_qa, to_sec, spans_of, evidence_rows)
schema       QuestionRecord / VLMAnswerOutput / evaluation dataclasses
loader       SuperMemoryDataLoader: QA json -> QuestionRecord (Object & Location Memory filter)
pilot        oracle pilot question selection
"""
