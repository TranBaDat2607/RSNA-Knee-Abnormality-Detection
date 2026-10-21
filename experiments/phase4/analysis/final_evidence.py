"""Final-picks evidence (/kg-final, 2026-10-21): gold-58 macro, pairwise Spearman and paired bootstrap CIs of the OAI candidates.

usage: python final_evidence.py <gold dir holding the downloaded gold kernel outputs>

A12, A16, A17 are rebuilt from the gold legs exactly as in a17_gate.py. Gold-58 cannot resolve gaps of a few thousandths
(SE about 0.02); this is a table for the user's final choice, not a decision rule. The bootstrap uses a vectorised
Mann-Whitney AUC (same values as roc_auc_score) so that it runs in seconds.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, "Baker's": 0.30}
W_READER = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02, "MCL": 0.02,
            "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008, "Effusion": 0.004,
            "Synovitis": 0.003}
g = sys.argv[1]
y = pd.read_csv(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_root', 'gold_labels.csv'),
                dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
Y = (y.values > 0.5).astype(int)


def leg(*p):
    return pd.read_csv(os.path.join(g, *p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[y.index, T].values


def rk(x):
    return rankdata(x, axis=0) / len(x)


def aucs(p, idx=None):
    """Per-finding AUC (Mann-Whitney); NaN where a resample has one class."""
    i = np.arange(len(Y)) if idx is None else idx
    yy, pp = Y[i], p[i]
    r = rankdata(pp, axis=0)
    pos = yy.sum(0)
    neg = len(i) - pos
    with np.errstate(invalid='ignore', divide='ignore'):
        a = ((r * yy).sum(0) - pos * (pos + 1) / 2) / (pos * neg)
    return np.where((pos > 0) & (neg > 0), a, np.nan)


def macro(p):
    return float(np.nanmean(aucs(p)))


def rho(a, b):
    return float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(len(T))]))


def boot(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y))
        d.append(np.nanmean(aucs(a, i) - aucs(b, i)))
    return np.percentile(d, [2.5, 97.5])


n384 = leg('rsna-knee-oai-gold58', 'gold_acc.csv')
n224 = leg('rsna-knee-oai-gold58', 'gold_eff.csv')
reader = leg('rsna-knee-oai-gold58', 'gold_reader.csv')
r6f = leg('rsna-knee-r6-gold58', 'gold_r6_flip.csv')
r7 = leg('rsna-knee-oai-gold58-r7', 'gold_r7_plain.csv')
s6f = leg('rsna-knee-oai-gold58-s6fall', 'gold_s6fall.csv')
raptor = 0.65 * rk(n384) + 0.35 * rk(n224)
w = np.array([W_RAPTOR_C.get(c, 0.5) for c in T])
wr = np.array([W_READER[c] for c in T])
A16 = rk((1 - wr) * rk(w * rk(raptor) + (1 - w) * rk(r6f)) + wr * rk(reader))
C = {'A12': (2 / 3) * (0.5 * rk(raptor) + 0.5 * rk(r7)) + (1 / 3) * rk(s6f), 'A16': A16, 'A17': (2 / 3) * A16 + (1 / 3) * rk(s6f)}
print('gold-58 macro:', {k: round(macro(v), 4) for k, v in C.items()}, '(a17_gate: A12 0.9404, A16 0.9387, A17 0.9395)')
names = list(C)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        print('%s vs %s  rho %.3f  gap %+.4f  CI %s' % (a, b, rho(C[a], C[b]), macro(C[a]) - macro(C[b]), np.round(boot(C[a], C[b]), 4)))
