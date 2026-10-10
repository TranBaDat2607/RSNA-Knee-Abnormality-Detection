"""Gold-58 check of the OAI-track blend variants from leg predictions (rsna-knee-oai-gold58 + our R5t gold.npy).

usage: python gold_legs.py <gold58_out_dir> <r5t_out_dir>

nartaa's checkpoints were selected on gold-58, so every number that leans on them is optimistic; the paired
bootstrap of each variant against the A+B parent is what matters, and only as a regression guard.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']
READER_W = {"Baker's": 0.35, "Contusion": 0.3, "Medial OA": 0.25, "ACL": 0.25, "Medial Meniscus": 0.2, "MCL": 0.2,
            "Lateral Meniscus": 0.08, "Fracture": 0.08, "Lateral OA": 0.08, "PF OA": 0.08, "Effusion": 0.04, "Synovitis": 0.03}
g58, r5 = sys.argv[1], sys.argv[2]
y = pd.read_csv(os.path.join(g58, 'gold_root', 'gold_labels.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
Y = (y.values > 0.5).astype(int)


def rk(x):
    return rankdata(x, axis=0) / len(x)


def leg(name):
    return pd.read_csv(os.path.join(g58, f'gold_{name}.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[y.index, T].values


def ours_tag(pattern):
    fs = sorted(glob.glob(os.path.join(r5, pattern)))
    ps = []
    for f in fs:
        ids = [str(u) for u in json.load(open(os.path.join(os.path.dirname(f), 'run.json')))['gold_ids']]
        ps.append(pd.DataFrame(np.load(f), index=ids, columns=T).loc[y.index].values)
    return np.mean(ps, axis=0), len(fs)


def aucs(p, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    out = []
    for j in range(len(T)):
        yy = Y[idx, j]
        out.append(roc_auc_score(yy, p[idx, j]) if 0 < yy.sum() < len(yy) else np.nan)
    return np.array(out)


acc, eff, reader = leg('acc'), leg('eff'), leg('reader')
f5, n5 = ours_tag('r5t*/run/r5t_nano336/fold*/gold.npy')
ff, nf = ours_tag('r5tfull/run_s*/r5tfull_nano336/fold-1/gold.npy')
ours = (rk(f5) + rk(ff)) / 2          # rsna_knee.mil.ours.combine: fold mean per tag, then mean of tag ranks
print(f'our folds {n5}, full-data models {nf}')
wr = np.array([READER_W[t] for t in T])


def with_reader(a):
    return rk((1 - wr) * rk(a) + wr * rk(reader))


def mix(a, b, w):
    return rk((1 - w) * rk(a) + w * rk(b))


variants = {
    'acc': acc, 'eff': eff, 'reader': reader, 'ours': ours,
    'AB (goodpjw 0.950)': with_reader(acc),
    'AB + ours 0.10 [v1]': mix(with_reader(acc), ours, 0.10),
    'A2B: eff 0.25 [v2]': with_reader(mix(acc, eff, 0.25)),
    'A2B + ours 0.10': mix(with_reader(mix(acc, eff, 0.25)), ours, 0.10),
    'AB + ours 0.20': mix(with_reader(acc), ours, 0.20),
}
base = variants['AB (goodpjw 0.950)']
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(Y), len(Y)) for _ in range(2000)]
base_b = np.array([np.nanmean(aucs(base, i)) for i in boots])
print(f"{'variant':24s} macro   d vs AB  [95% CI]   per-target")
for k, p in variants.items():
    a = aucs(p)
    d = np.array([np.nanmean(aucs(p, i)) for i in boots]) - base_b
    print(f"{k:24s} {np.nanmean(a):.4f} {np.nanmean(a) - np.nanmean(aucs(base)):+.4f} [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}]",
          ' '.join(f'{v:.2f}' for v in a))
r = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method='spearman') for j in range(len(T))])
print(f"mean spearman: ours-acc {r(ours, acc):.3f} ours-reader {r(ours, reader):.3f} reader-acc {r(reader, acc):.3f} eff-acc {r(eff, acc):.3f}")
