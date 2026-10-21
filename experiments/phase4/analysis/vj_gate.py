"""VJ gate (pre-registered in docs/phase4_notes.md, 2026-10-21): read once.

usage: python vj_gate.py <gold dir holding the downloaded gold kernel outputs>

VJ = nurikw3's V-JEPA 2.1 5-fold mean probability on the 58 gold studies, at the deployment precision (fp16 if max
|p_fp16 - p_fp32| <= 0.01 and no NaN, else fp32).  A16 as in a17_gate.py.  A18 = 3/4 rank(A16) + 1/4 rank(VJ).
Rules: 1. VJ >= 0.925; 2. every fold < 0.975 (leak guard); 3. mean Spearman <= 0.90 vs the Raptor half, R6 flip and
s6fall; 4. A18 >= A16 - 0.003; 5. 5.3 h + VJ stage (s/study per worker x 1322 / 2 x 1.15) <= 8.0 h.
"""
import json, os, sys
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
s6f = leg('rsna-knee-oai-gold58-s6fall', 'gold_s6fall.csv')
raptor = 0.65 * rk(n384) + 0.35 * rk(n224)
w = np.array([W_RAPTOR_C.get(c, 0.5) for c in T])
wr = np.array([W_READER[c] for c in T])
A16 = rk((1 - wr) * rk(w * rk(raptor) + (1 - w) * rk(r6f)) + wr * rk(reader))
print('reference: A16 %.4f (a17_gate 0.9387)' % macro(A16))

info = json.load(open(os.path.join(g, 'rsna-knee-vj-gold58', 'vj_gold_info.json')))
print('timing/precision', json.dumps(info))
fp16_ok = info['nan_fp16'] == 0 and info['max_abs_dp_fp16_vs_fp32'] <= 0.01
prec = 'fp16' if fp16_ok else 'fp32'
vj = leg('rsna-knee-vj-gold58', 'gold_vj.csv' if fp16_ok else 'gold_vj_fp32.csv')
print('deployment precision:', prec)
folds = [macro(leg('rsna-knee-vj-gold58', 'gold_vj_f%d.csv' % k)) for k in range(5)]

r1 = macro(vj)
rhos = {'raptor': rho(vj, raptor), 'r6f': rho(vj, r6f), 's6fall': rho(vj, s6f)}
A18 = 0.75 * rk(A16) + 0.25 * rk(vj)
d = macro(A18) - macro(A16)
sec = info['decode_s_mean'] + (info['fp16_s_mean'] if fp16_ok else info['fp32_s_mean'])
stage_h = sec * 1322 / 2 * 1.15 / 3600
print()
print('VJ per finding', dict(zip(T, np.round(aucs(vj), 3))))
print('VJ folds (fp16)', np.round(folds, 4))
print('rule 1  VJ gold-58 %.4f >= 0.925: %s' % (r1, r1 >= 0.925))
print('rule 2  max fold %.4f < 0.975: %s' % (max(folds), max(folds) < 0.975))
print('rule 3  rho %s <= 0.90: %s' % ({k: round(v, 3) for k, v in rhos.items()}, max(rhos.values()) <= 0.90))
print('rule 4  A18 %.4f - A16 %.4f = %+.4f >= -0.003: %s  (CI %s)' % (macro(A18), macro(A16), d, d >= -0.003,
                                                                     np.round(boot(A18, A16), 4)))
print('rule 5  VJ stage %.2f s/study/worker -> %.2f h; 5.3 + %.2f = %.2f h <= 8.0: %s' % (sec, stage_h, stage_h,
                                                                                        5.3 + stage_h, 5.3 + stage_h <= 8.0))
ok = r1 >= 0.925 and max(folds) < 0.975 and max(rhos.values()) <= 0.90 and d >= -0.003 and 5.3 + stage_h <= 8.0
print('GATE', 'PASS' if ok else 'FAIL')
print('A18 per finding delta', dict(zip(T, np.round(aucs(A18) - aucs(A16), 3))))
