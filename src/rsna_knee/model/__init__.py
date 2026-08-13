"""The imaging model: a partially fine-tuned DINOv2 backbone plus a per-diagnosis
attention head over the study's slot embeddings, tracked with a weight EMA.

:func:`backbone.build_model` loads DINOv2 and returns a :class:`network.Model` wrapping it
with :class:`heads.SlotHead`. :class:`ema.Ema` is a separate, optional weight-averaging
wrapper used by the training loop, not by the model itself.
"""
