"""SuperMemory-VQA evaluation: run a VLM over questions and score it.

    python -m hindsight.evaluation.prepare_pilot   # pick pilot questions, cut oracle evidence frames
    python -m hindsight.evaluation.run             # evaluate one condition (oracle / blind / curated)
    python -m hindsight.evaluation.compare         # side-by-side table of several runs

Metrics (accuracy, abstention, hallucination, answerability F1) are in hindsight.evaluation.metrics.
"""
