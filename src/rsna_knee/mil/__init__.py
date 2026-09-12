"""2.5-D per-finding attention-MIL models (the CoAtNet "Raptor" family).

A study is read as a bag of three-slice windows (adjacent slices stacked as RGB), each window
goes through an image backbone, and a separate attention distribution per finding pools the
windows. On the 58 gold studies this family (0.91-0.92 macro AUC per checkpoint) is far ahead
of the DINOv2 slot model in ``rsna_knee.model`` (0.84), which is why it gets its own package.

``recipes``      volume recipes (which series, how many slices, crop, resolution) and arm specs.
``volume``       DICOM series -> fixed ``uint8`` slice stack, exactly as the public checkpoints saw it.
``windows``      window-centre selection, triplet assembly, GPU-side resize/normalise.
``model``        ``MILClassifier`` (state-dict compatible with the public checkpoints) and loading.
``blend``        rank-space fusion of arms.
``infer``        multi-arm inference: one decode per study shared by every recipe, on every GPU.
``corpus``       the public pre-decoded 44 x 336 training corpus, lazily memory-mapped.
``orientation``  canonical anatomical orientation (left/right mirroring) per slot.
``teachers``     public report-label tables used as weak training targets.
``train``        corpus training loop (fp16, grad-checkpointing, single-GPU or DDP), gold-58 never trained on.
``plan``         runs training stages in isolated subprocesses (A/B pairs side by side, or DDP).
``submit``       offline submission: MIL arms + residual-gated CoAtNet, fixed-weight rank fusion, validation.
"""
