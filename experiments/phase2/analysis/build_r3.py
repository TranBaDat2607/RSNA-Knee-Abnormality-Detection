"""R3 soft targets: teach4 text labels + several image models' OUT-OF-FOLD predictions.

Every image source is quantile-mapped onto teach4's per-finding distribution (so mixing changes the
ordering, not the label calibration), sources are averaged with fixed weights (renormalised where a
source lacks a study), then mixed with teach4 at ``W_TEXT``.

Gold-58 check: each source's gold-58 prediction (fold-mean for k-fold runs, OOF for the old arms)
mixed the same way, with teach4 on gold.

usage: python build_r3.py [out.csv]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']
HERE = os.path.dirname(os.path.abspath(__file__))
W_TEXT = 0.3
# source -> weight of the image part
SOURCES = {"r2_convnext_nano": 0.5, "r1_convnext_nano": 0.2, "r1_resnet34": 0.1, "dinov3": 0.1, "radimagenet": 0.1}
RUNS = {"r2_convnext_nano": "out_r2/run", "r1_convnext_nano": "out_r1/run", "r1_resnet34": "out_r1/run"}

gold = pd.read_csv('C:/Users/Admin/Desktop/RSNA-Knee-Abnormality-Detection/data/gold_annotated.csv',
                   dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T].astype(float)
teach = pd.read_csv(os.path.join(HERE, 'teach4.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]


def macro(y, p):
    a = {t: roc_auc_score((y[:, j] > 0.5).astype(int), p[:, j]) for j, t in enumerate(T)}
    return float(np.mean(list(a.values()))), a


def qmap(src: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Per column: replace each value by teach4's quantile at the value's rank."""
    out = src.copy()
    for t in T:
        r = np.sort(ref[t].dropna().to_numpy())
        q = src[t].rank(pct=True).to_numpy()
        out[t] = np.interp(q, np.linspace(0, 1, len(r)), r)
    return out


def load_sources():
    oof, gp = {}, {}
    for tag, run in RUNS.items():
        fo = sorted(glob.glob(os.path.join(HERE, run, tag, 'fold*', 'oof.csv')))
        if not fo:
            print(tag, 'missing'); continue
        oof[tag] = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fo]).set_index('StudyInstanceUID')[T]
        gids = json.load(open(os.path.join(os.path.dirname(fo[0]), 'run.json')))['gold_ids']
        g = np.mean([np.load(os.path.join(os.path.dirname(f), 'gold.npy')) for f in fo], axis=0)
        gp[tag] = pd.DataFrame(g, index=[str(x) for x in gids], columns=T).reindex(gold.index)
        print(f"{tag}: {len(fo)} folds, OOF n={len(oof[tag])}, gold-58 {macro(gold.values, gp[tag].values)[0]:.4f}")
    for tag, f in (("dinov3", "oldoof/dinov3-oof-full-corpus/dinov3_oof_audit.csv"),
                   ("radimagenet", "oldoof/radimagenet-oof-full-corpus/radimagenet_oof_audit.csv")):
        d = pd.read_csv(os.path.join(HERE, f), dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[T]
        oof[tag] = d.drop(index=gold.index, errors='ignore')
        gp[tag] = d.reindex(gold.index)
        print(f"{tag}: OOF n={len(oof[tag])}, gold-58 {macro(gold.values, gp[tag].values)[0]:.4f}")
    return oof, gp


def mix(frames: dict, ids, ref) -> pd.DataFrame:
    num = pd.DataFrame(0.0, index=ids, columns=T)
    den = pd.Series(0.0, index=ids)
    for tag, w in SOURCES.items():
        if tag not in frames:
            continue
        m = qmap(frames[tag].reindex(ids).dropna(), ref).reindex(ids)
        have = m.notna().all(axis=1)
        num[have] += w * m[have]
        den[have] += w
    img = num.div(den.replace(0, np.nan), axis=0)
    return img, den


if __name__ == '__main__':
    oof, gp = load_sources()
    # gold-58 check of the recipe
    img_g, _ = mix(gp, gold.index, teach)
    print(f"image mix gold-58 {macro(gold.values, img_g.values)[0]:.4f}")
    for wt in (0.0, 0.2, 0.3, 0.4, 0.5):
        m = wt * teach.reindex(gold.index) + (1 - wt) * img_g
        mm, per = macro(gold.values, m.values)
        print(f"  W_TEXT={wt}: gold-58 {mm:.4f}", {k: round(v, 3) for k, v in per.items()} if wt == W_TEXT else '')
    ids = teach.index.difference(gold.index)
    img, den = mix(oof, ids, teach)
    r3 = W_TEXT * teach.reindex(ids) + (1 - W_TEXT) * img
    r3 = r3.fillna(teach.reindex(ids))  # no image source at all -> text only
    print('coverage of image sources (weight sum):', den.describe().round(2).to_dict())
    if len(sys.argv) > 1:
        r3.clip(0, 1).rename_axis('StudyInstanceUID').reset_index().to_csv(sys.argv[1], index=False)
        print('wrote', sys.argv[1], len(r3))
