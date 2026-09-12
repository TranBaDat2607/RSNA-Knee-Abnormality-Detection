"""Volume recipes and arm specifications.

A *recipe* fixes how a study becomes a slice stack: which series fill which slot, how many
slices each slot takes and from what fraction of the series, the physical crop, and the pixel
grid. A checkpoint only reproduces its validation score on the recipe it was trained on, so an
*arm* is always a (checkpoint, recipe, eval-window count) triple.

Arms sharing a recipe share one DICOM decode per study (see ``infer``), which is where most of
the per-study runtime goes (0.6-1.8 s decode vs ~1 s model on a T4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    plane: str   # "Sagittal" | "Coronal" | "Axial"
    fluid: int   # 1 prefer a fluid-sensitive series, 0 prefer a non-fluid one, -1 no preference
    n: int       # slices taken from the chosen series


@dataclass(frozen=True)
class VolumeRecipe:
    name: str
    img: int                    # stack pixel grid (windows are resized to the checkpoint's res)
    crop_mm: float              # centred physical crop before resizing
    span: tuple[float, float]   # fraction of the series the slices are spread over
    slots: tuple[Slot, ...]

    @property
    def n_slices(self) -> int:
        return sum(s.n for s in self.slots)


# Sagittal carries the cruciates and menisci, so it gets the most slices; the second sagittal
# and coronal slots prefer the non-fluid (anatomy/cartilage) sequence.
SLOTS_64 = (Slot("Sagittal", 1, 18), Slot("Sagittal", 0, 14), Slot("Coronal", 1, 12),
            Slot("Coronal", 0, 8), Slot("Axial", -1, 12))
SLOTS_44 = (Slot("Sagittal", 1, 12), Slot("Sagittal", 0, 10), Slot("Coronal", 1, 8),
            Slot("Coronal", 0, 6), Slot("Axial", -1, 8))

MAXSPAN_336 = VolumeRecipe("maxspan_336", 336, 140.0, (0.02, 0.98), SLOTS_64)
DENSE_384 = VolumeRecipe("dense_384", 384, 140.0, (0.02, 0.98), SLOTS_64)
NATIVE44_384 = VolumeRecipe("native44_384", 384, 140.0, (0.06, 0.94), SLOTS_44)
# The public pre-decoded corpus (dreaddevelopment/knee-raptor-corpus). Its span is the older narrow
# 15-85% one, not the 6-94% of the native44 checkpoint: with it, build_volume reproduces the corpus
# stacks exactly (mean |diff| 0.0 on 29 slots of 6 studies, E5c). A model trained on the corpus must
# be served with this recipe.
CORPUS44_336 = VolumeRecipe("corpus44_336", 336, 140.0, (0.15, 0.85), SLOTS_44)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    checkpoint: str       # file name, searched for under /kaggle/input
    recipe: VolumeRecipe
    k_eval: int           # evenly spaced eval windows
    weight: float = 1.0


# Public Raptor checkpoints worth running (gold-58 macro AUC measured in E1, see
# docs/experiment_ledger.md). Deliberately absent:
#  - a "reversed window order" view of v5: the attention pool is permutation-invariant, so it
#    reproduces v5 to 3e-5 while costing a full pass;
#  - v4 widedense: Spearman 0.94 with v5 and no gold-58 gain when added.
PUBLIC_RAPTOR_ARMS = (
    ArmSpec("raptor_v5_maxspan", "raptor_ft_coatnet_v5_full_swa.pt", MAXSPAN_336, 62),   # 0.920
    ArmSpec("raptor_v10_dense", "raptor_ft_coatnet_v10_full.pt", DENSE_384, 62),         # 0.917
    ArmSpec("raptor_v8_native44", "raptor_ft_coatnet_v8_full_swa.pt", NATIVE44_384, 42),  # 0.912
)
