"""R6 gate (pre-registered in docs/phase4_notes.md, 2026-10-16): read once, after the r6-gold58 kernel.

usage: python r6_gate.py <gold_out dir with the three phase-3 gold kernels> <r6 gold kernel output dir>

Rule 1 (fourth leg): R6 flip gold >= 0.925, mean Spearman <= 0.90 vs each of raptor, R7, s6fall,
        A14 = 3/4 A12 + 1/4 rank(R6) >= A12 - 0.003.
Rule 2 (swap, if rule 1 fails on correlation): R6 gold >= 0.925 and A14swap (A12 with R7 -> R6) >= A12 - 0.003.
Runtime is reported from the receipts; the decision on flip vs plain uses it (see the notes).
"""
import json, os, sys
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
a12 = (2 / 3) * (0.5 * rk(raptor) + 0.5 * rk(r7)) + (1 / 3) * rk(s6f)
print('A12 gold-58 %.4f (phase 3: 0.9406); R7 plain %.4f' % (macro(a12), macro(r7)))
legs = {}
for tag in ('plain', 'flip'):
    f = os.path.join(s, 'gold_r6_%s.csv' % tag)
    if os.path.isfile(f):
        legs[tag] = leg(f)
        p = legs[tag]
        print('R6 %-5s macro %.4f | rho raptor %.3f R7 %.3f s6fall %.3f' % (tag, macro(p), rho(p, raptor), rho(p, r7), rho(p, s6f)))
    rc = os.path.join(s, 'r6_%s_receipt.json' % tag)
    if os.path.isfile(rc):
        r = json.load(open(rc))
        print('   receipt %s: seconds %s per_study_mean %s gpus %s scored %s fallback %s' % (
            tag, r.get('seconds'), r.get('per_study_mean'), r.get('gpus'), r.get('scored'), r.get('fallback_studies')))
if 'plain' in legs and 'flip' in legs:
    print('rho R6 plain vs flip %.3f' % rho(legs['plain'], legs['flip']))
t = legs.get('flip', legs.get('plain'))
a14 = 0.75 * a12 + 0.25 * rk(t)
a14s = (2 / 3) * (0.5 * rk(raptor) + 0.5 * rk(t)) + (1 / 3) * rk(s6f)
print('A14 fourth leg %.4f (delta %+.4f) | A14 swap %.4f (delta %+.4f)' % (
    macro(a14), macro(a14) - macro(a12), macro(a14s), macro(a14s) - macro(a12)))
per = pd.DataFrame({'R6': [roc_auc_score(Y[:, j], t[:, j]) for j in range(len(T))],
                    'R7': [roc_auc_score(Y[:, j], r7[:, j]) for j in range(len(T))],
                    'A12': [roc_auc_score(Y[:, j], a12[:, j]) for j in range(len(T))],
                    'A14_4th': [roc_auc_score(Y[:, j], a14[:, j]) for j in range(len(T))],
                    'A14_swap': [roc_auc_score(Y[:, j], a14s[:, j]) for j in range(len(T))]}, index=T).round(3)
print(per.to_string())
strong = macro(t) >= 0.925
decor = max(rho(t, raptor), rho(t, r7), rho(t, s6f)) <= 0.90
r1 = {'1a R6 >= 0.925': strong, '1b rho <= 0.90 vs raptor, R7, s6fall': decor,
      '1c A14 4th >= A12 - 0.003': macro(a14) >= macro(a12) - 0.003}
r2 = {'2a R6 >= 0.925': strong, '2b A14 swap >= A12 - 0.003': macro(a14s) >= macro(a12) - 0.003}
for k, v in list(r1.items()) + list(r2.items()):
    print('rule %-42s %s' % (k, 'PASS' if v else 'FAIL'))
if all(r1.values()):
    print('DECISION fourth leg (rule 1)')
elif all(r2.values()):
    print('DECISION swap R7 -> R6 (rule 2)')
else:
    print('DECISION drop R6')
