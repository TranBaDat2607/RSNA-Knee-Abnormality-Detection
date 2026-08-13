"""Turning ground truth + LLM-generated labels into per-study training targets.

Two sources, in priority order: the ~58 studies annotated directly in ``train.csv`` (full
confidence, weighted up via ``gold_weight``), and everything else from
``data/llm_labels_full.csv`` (weighted by distance from the rubric's 0.5 midpoint, since
the LLM labeler doesn't emit an explicit confidence — see CLAUDE.md's "Model history" for
why that pipeline replaced the notebook's rule-based extractor as the training-target
source). A study with neither source contributes nothing and is dropped from training.
"""
