"""All tunable constants for the pipeline, in one place.

Every value here mirrors a constant from the "v4" configuration cells of
``eda/rsna-knee-data-structure-eda-baseline.ipynb`` (image/cache sizing, slot definitions,
augmentation magnitudes, training hyperparameters, laterality/CV knobs). Reproducing the
notebook run exactly means constructing ``Config()`` with no overrides; every field can
also be overridden per-run (e.g. from the CLI) without editing this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------------- #

TARGETS: list[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

OA_TARGETS: set[str] = {"Medial OA", "Lateral OA", "PF OA"}

# --------------------------------------------------------------------------------- #
# Slot definitions
# --------------------------------------------------------------------------------- #
# A "slot" is one (anatomical plane x acquisition axis) combination. Six slots cover the
# protocol: three planes, crossed with fat-suppressed-fluid-sensitive vs. the rest.

# name, plane, fluid ("None" = don't condition on recovered weighting), fat-saturated
SlotDef = tuple[str, str, "bool | None", bool]

SLOTS_RECOVERED: list[SlotDef] = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

# The alternative scheme: plane x the provided fat-sat flag only, ignoring recovered
# weighting. Kept as a switch so the slot definition can be varied on its own.
SLOTS_PUBLIC: list[SlotDef] = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]


def slots_for_scheme(scheme: str) -> list[SlotDef]:
    if scheme == "public":
        return SLOTS_PUBLIC
    if scheme == "recovered":
        return SLOTS_RECOVERED
    raise ValueError(f"unknown SLOT_SCHEME {scheme!r}, expected 'recovered' or 'public'")


# --------------------------------------------------------------------------------- #
# Sequence-typing regexes (DICOM header -> fat-sat / T1 / T2 / PD)
# --------------------------------------------------------------------------------- #

import re  # noqa: E402  (grouped with the regexes that use it)

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
SEP_RX = re.compile(r"[_\-.]")
FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
T1_RX = re.compile(r"\bt1\b|\bt1w\b")
T2_RX = re.compile(r"\bt2\b|\bt2w\b")
PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

HDR_TAGS: list[str] = [
    "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
    "RepetitionTime", "EchoTime", "Laterality", "ImageLaterality",
    "ImagePositionPatient", "PixelSpacing", "Rows",
    "Columns", "RescaleSlope", "RescaleIntercept",
]


# --------------------------------------------------------------------------------- #
# Grouped hyperparameter config
# --------------------------------------------------------------------------------- #

@dataclass
class CacheConfig:
    """Physical-scale sampling and the in-memory decode cache."""

    img: int = 224                 # encoder input side; a multiple of the ViT patch size 14
    crop_mm: float = 160.0         # physical extent of the centre crop (a knee FOV is 140-180mm)
    group: int = 3                 # slices per encoder input, stacked as 3 channels
    n_group_max: int = 3           # groups per slot before the memory budget is applied
    cache_budget_gb: float = 12.0
    hdr_threads: int = 16
    pix_threads: int = 12
    slot_scheme: str = field(default_factory=lambda: os.environ.get("SLOT_SCHEME", "recovered"))

    @property
    def slots(self) -> list[SlotDef]:
        return slots_for_scheme(self.slot_scheme)

    @property
    def n_slot(self) -> int:
        return len(self.slots)


@dataclass
class AugConfig:
    """Augmentation magnitudes. No flips of either kind — see training/augment.py."""

    use_vflip: bool = False        # a knee is not vertically symmetric; kept as an off-switch
    affine: bool = True
    rot_deg: float = 8.0
    scale: float = 0.08
    shift: float = 0.05
    intensity: float = 0.10


@dataclass
class LateralityConfig:
    """Left/right resolution: DICOM tag first, patient-position sign as a fallback."""

    fallback: str = "auto"         # "auto" | "on" | "off"
    min_agreement: float = 0.85    # required tag/position agreement before "auto" enables it
    min_offset_mm: float = 5.0     # |x| below this is not a usable side cue


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_studies: int = 8         # a study is a bag of up to n_slot slot images
    lr_backbone: float = 8e-6      # the encoder is adapted, not retrained
    lr_head: float = 1e-3
    weight_decay: float = 0.02
    unfreeze_last: int = 6         # trainable transformer blocks, counted from the output end
    eval_batch: int = 12
    time_budget_s: float = 8.0 * 3600
    ema_decay: float = 0.997       # 0 disables the moving average
    rank_loss_w: float = 0.05      # 0 disables the pairwise ranking term
    rank_pos: float = 0.60         # graded-target cutoffs defining a usable ranking pair
    rank_neg: float = 0.40
    seed: int = 2026


@dataclass
class CVConfig:
    n_folds: int = 4
    max_folds: str | int = "auto"    # "auto" fits as many as the time budget allows
    fold_time_pad: float = 1.15      # safety factor on the measured per-fold time
    selection: str = "worse_of_two"  # "worse_of_two" | "derived"
    gold_weight: float = 3.0         # sample weight of an annotated study relative to an LLM one


@dataclass
class Config:
    """Top-level bundle. Construct with no arguments to reproduce the notebook's run."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    laterality: LateralityConfig = field(default_factory=LateralityConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    targets: list[str] = field(default_factory=lambda: list(TARGETS))
