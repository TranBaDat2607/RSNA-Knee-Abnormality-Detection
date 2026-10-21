"""A16 / A17 gold-58 guard (pre-registered in docs/phase4_notes.md, 2026-10-19): read once.

usage: python a17_gate.py <gold dir holding the downloaded gold kernel outputs>

A16 = the public 0.9586 recipe reproduced from the gold legs: Raptor half = 0.65 rank(n384) + 0.35 rank(n224), per-finding
Raptor weight W_RAPTOR_C (else 0.5) against rank(R6 flip), then goodpjw's reader rank-blended at its per-finding weights.
A17 = 2/3 x A16 + 1/3 x rank(s6fall).  Guard: A17 >= A16 - 0.003.
Information only (final picks): A16 with uniform 0.5 weights, i.e. the effect of the LB-probed per-finding weights on gold-58.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

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


def aucs(p):
    return np.array([roc_auc_score(Y[:, j], p[:, j]) for j in range(len(T))])


def macro(p):
    return float(aucs(p).mean())


def rho(a, b):
    return float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(len(T))]))


def boot(a, b, n=4000, seed=0):
    """Paired bootstrap of macro(a) - macro(b) over studies (findings with one class in a resample are skipped)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y))
        v = [roc_auc_score(Y[i, j], a[i, j]) - roc_auc_score(Y[i, j], b[i, j]) for j in range(len(T)) if 0 < Y[i, j].sum() < len(i)]
        d.append(np.mean(v))
    return np.percentile(d, [2.5, 97.5])


n384 = leg('rsna-knee-oai-gold58', 'gold_acc.csv')
n224 = leg('rsna-knee-oai-gold58', 'gold_eff.csv')
reader = leg('rsna-knee-oai-gold58', 'gold_reader.csv')
r6f = leg('rsna-knee-r6-gold58', 'gold_r6_flip.csv')
r6p = leg('rsna-knee-r6-gold58', 'gold_r6_plain.csv')
r7 = leg('rsna-knee-oai-gold58-r7', 'gold_r7_plain.csv')
s6f = leg('rsna-knee-oai-gold58-s6fall', 'gold_s6fall.csv')
raptor = 0.65 * rk(n384) + 0.35 * rk(n224)


def a16(w_c, compact, with_reader=True):
    w = np.array([w_c.get(c, 0.5) for c in T])
    base = w * rk(raptor) + (1 - w) * rk(compact)
    if not with_reader:
        return base
    wr = np.array([W_READER[c] for c in T])
    return rk((1 - wr) * rk(base) + wr * rk(reader))


a12 = (2 / 3) * (0.5 * rk(raptor) + 0.5 * rk(r7)) + (1 / 3) * rk(s6f)
A16 = a16(W_RAPTOR_C, r6f)
A17 = (2 / 3) * A16 + (1 / 3) * rk(s6f)
print('reference: A12 %.4f (phase 3/4: 0.9404-0.9406)' % macro(a12))
print('legs: raptor %.4f  R6 flip %.4f  reader %.4f  s6fall %.4f' % (macro(raptor), macro(r6f), macro(reader), macro(s6f)))
print('A16 %.4f   A17 %.4f   A17 - A16 %+.4f  CI %s   guard (>= -0.003): %s' % (
    macro(A16), macro(A17), macro(A17) - macro(A16), np.round(boot(A17, A16), 4), 'PASS' if macro(A17) >= macro(A16) - 0.003 else 'FAIL'))
print('A17 - A12 %+.4f  CI %s' % (macro(A17) - macro(a12), np.round(boot(A17, a12), 4)))
print('rho(A16, s6fall) %.3f  rho(raptor, R6 flip) %.3f' % (rho(A16, s6f), rho(raptor, r6f)))
# information only (final picks): LB-probed per-finding weights vs uniform 0.5 on gold-58
U16 = a16({}, r6f)
print('info: A16 uniform-0.5 %.4f -> per-finding %.4f (%+.4f, CI %s); without reader %.4f' % (
    macro(U16), macro(A16), macro(A16) - macro(U16), np.round(boot(A16, U16), 4), macro(a16(W_RAPTOR_C, r6f, False))))
per = pd.DataFrame({'raptor': aucs(raptor), 'R6flip': aucs(r6f), 'w_raptor': [W_RAPTOR_C.get(c, 0.5) for c in T],
                    'A16_uniform': aucs(U16), 'A16': aucs(A16), 'A17': aucs(A17)}, index=T)
print(per.round(3).to_string())
