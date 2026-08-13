"""rsna_knee: modular training/inference pipeline for the RSNA Knee Abnormality Detection
competition.

This package is the productionised, module-per-concern rewrite of
``eda/rsna-knee-data-structure-eda-baseline.ipynb``. The notebook stays the place for
narrative EDA and one-off analysis; this package is what actually trains and predicts.

Layout
------
``config``            All tunable constants in one place (targets, slots, hyperparameters).
``paths``              Locating the competition data / DINOv2 weights / LLM labels, on
                       Kaggle or locally.
``logging_utils``      A single relative-time logger shared by every module.
``dicom``              DICOM ingestion: header probing, laterality resolution, slot
                       selection, physical-scale slice sampling, in-memory caching.
``labels``             Turning ``train.csv`` ground truth + ``llm_labels_full.csv`` into
                       per-study training targets and sample weights.
``model``              The SlotHead attention pooling, the DINOv2-backed ``Model``, the
                       partially-unfrozen backbone builder, and the EMA weight tracker.
``training``           Augmentation, losses, the AUC metrics, and the per-fold training
                       loop.
``inference``          Prediction (group-averaged, sigmoid) and submission-file writing.
``pipeline``            Orchestrates all of the above into the same 4-fold CV run the
                       notebook performs.
``cli``                 ``python -m rsna_knee.cli`` / the ``rsna-knee`` console script.

Nothing in this package reimplements the multilingual rule-based report extractor from
the notebook's section 2 -- that extractor is no longer what feeds the model. Training
targets come from ``train.csv``'s own annotated rows plus the LLM-generated labels in
``data/llm_labels_full.csv`` (see CLAUDE.md for how those were produced).
"""

__version__ = "0.1.0"
