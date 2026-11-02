"""Write data/gold_annotated.csv from Kaggle's train.csv: the studies with expert labels.

The file holds competition data (report text and labels), so it is not in git. Download train.csv from the competition
page first; scripts/llm_label_gold.py and the phase 2 analysis scripts read this output.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    train = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[train[TARGETS].notna().all(axis=1)]
    gold[["StudyInstanceUID", "Report", *TARGETS]].to_csv(DATA / "gold_annotated.csv", index=False)
    print(f"wrote {len(gold)} studies")


if __name__ == "__main__":
    main()
