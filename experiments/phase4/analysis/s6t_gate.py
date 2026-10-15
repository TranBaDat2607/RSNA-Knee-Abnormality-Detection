"""S6T gate (pre-registered in docs/phase4_notes.md): read once, after training.

usage: python s6t_gate.py <gold_out dir with the three phase-3 gold kernels> <s6t gold kernel output dir>

1. leg gold-58 macro AUC >= 0.925
2. mean Spearman (12 findings) <= 0.90 against each of raptor, R7, s6fall
3. A13 gold-58 >= A12 gold-58 - 0.003 (A12 = 1/3 raptor + 1/3 R7 + 1/3 s6fall in ranks; A13 = 3/4 A12 + 1/4 rank(S6T))
Rule 4 (runtime) is checked from the D1 probe and the kernel's timing, not here.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
g, s = sys.argv[1], sys.argv[2]
y = pd.read_csv(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_root', 'gold_labels.csv'),
                dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
Y = (y.values > 0.5).astype(int)


def leg(p):
    return pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[y.index, T].values


def rk(x):
    return rankdata(x, axis=0) / len(x)


def macro(p):
    return float(np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in range(len(T))]))


def rho(a, b):
    return float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(len(T))]))


n384 = leg(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_acc.csv'))
n224 = leg(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_eff.csv'))
r7 = leg(os.path.join(g, 'rsna-knee-oai-gold58-r7', 'gold_r7_plain.csv'))
s6f = leg(os.path.join(g, 'rsna-knee-oai-gold58-s6fall', 'gold_s6fall.csv'))
raptor = 0.65 * rk(n384) + 0.35 * rk(n224)
a10 = 0.5 * rk(raptor) + 0.5 * rk(r7)
a12 = (2 / 3) * a10 + (1 / 3) * rk(s6f)
print('A12 gold-58 %.4f (phase 3: 0.9406)' % macro(a12))
seeds = sorted(f for f in os.listdir(s) if f.startswith('gold_s6t_s'))
for f in seeds + ['gold_s6t.csv']:
    p = leg(os.path.join(s, f))
    print('%-18s macro %.4f | rho raptor %.3f R7 %.3f s6fall %.3f' % (f, macro(p), rho(p, raptor), rho(p, r7), rho(p, s6f)))
t = leg(os.path.join(s, 'gold_s6t.csv'))
a13 = 0.75 * a12 + 0.25 * rk(t)
c = {'1 leg >= 0.925': macro(t) >= 0.925,
     '2 rho <= 0.90 vs raptor, R7, s6fall': max(rho(t, raptor), rho(t, r7), rho(t, s6f)) <= 0.90,
     '3 A13 >= A12 - 0.003': macro(a13) >= macro(a12) - 0.003}
print('A13 gold-58 %.4f (A12 %.4f, delta %+.4f)' % (macro(a13), macro(a12), macro(a13) - macro(a12)))
per = pd.DataFrame({'S6T': [roc_auc_score(Y[:, j], t[:, j]) for j in range(len(T))],
                    's6fall': [roc_auc_score(Y[:, j], s6f[:, j]) for j in range(len(T))],
                    'A12': [roc_auc_score(Y[:, j], a12[:, j]) for j in range(len(T))],
                    'A13': [roc_auc_score(Y[:, j], a13[:, j]) for j in range(len(T))]}, index=T).round(3)
print(per.to_string())
for k, v in c.items():
    print('rule %-40s %s' % (k, 'PASS' if v else 'FAIL'))
print('GATE', 'PASS' if all(c.values()) else 'FAIL')
