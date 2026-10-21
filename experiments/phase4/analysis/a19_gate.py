"""A19 member rule and guard (pre-registered in docs/phase4_notes.md, 2026-10-21 15:35 UTC): read once.

usage: python a19_gate.py <gold dir holding the downloaded gold kernel outputs>

Harness: nartaa 384 from the A19 kernel must match gold_acc (mean Spearman >= 0.99).
Member rule per checkpoint: (1) scores >= 95 % of gold studies, (2) gold-58 >= 0.925, (3) gold-58 < 0.975 (leak guard),
(4) Spearman vs nartaa 384 <= 0.995. Members in the fixed order fullb -> cohort -> bs14, at most 2.
A19 = A16 with the Raptor-384 slot = equal-weight rank mean of nartaa 384 and the members.  Guard: A19 >= A16 - 0.003.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

sys.argv += [] if len(sys.argv) > 1 else ['experiments/phase2/analysis/gold_out']
g = sys.argv[1]
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, "Baker's": 0.30}
W_READER = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02, "MCL": 0.02,
            "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008, "Effusion": 0.004,
            "Synovitis": 0.003}
y = pd.read_csv(os.path.join(g, 'rsna-knee-oai-gold58', 'gold_root', 'gold_labels.csv'),
                dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
Y = (y.values > 0.5).astype(int)


def leg(*p):
    return pd.read_csv(os.path.join(g, *p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(y.index)[T].values


def rk(x):
    return rankdata(x, axis=0) / len(x)


def aucs(p, idx=None):
    i = np.arange(len(Y)) if idx is None else idx
    yy, r = Y[i], rankdata(p[i], axis=0)
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
    d = [np.nanmean(aucs(a, i) - aucs(b, i)) for i in (rng.integers(0, len(Y), len(Y)) for _ in range(n))]
    return np.percentile(d, [2.5, 97.5])


def a16(n384):
    raptor = 0.65 * rk(n384) + 0.35 * rk(n224)
    w = np.array([W_RAPTOR_C.get(c, 0.5) for c in T])
    wr = np.array([W_READER[c] for c in T])
    return rk((1 - wr) * rk(w * rk(raptor) + (1 - w) * rk(r6f)) + wr * rk(reader))


K = 'rsna-knee-a19-gold58'
info = json.load(open(os.path.join(g, K, 'a19_gold_info.json')))
print('run info', json.dumps(info))
gold_acc = leg('rsna-knee-oai-gold58', 'gold_acc.csv')
n224 = leg('rsna-knee-oai-gold58', 'gold_eff.csv')
reader = leg('rsna-knee-oai-gold58', 'gold_reader.csv')
r6f = leg('rsna-knee-r6-gold58', 'gold_r6_flip.csv')
n384 = leg(K, 'n384_sub.csv')
h = rho(n384, gold_acc)
print('harness: nartaa 384 here %.4f vs gold_acc %.4f, rho %.4f >= 0.99: %s' % (macro(n384), macro(gold_acc), h, h >= 0.99))
if h < 0.99:
    sys.exit('HARNESS FAIL: stop and diagnose before reading members')
A16 = a16(n384)
print('A16 (this kernel n384) %.4f; with gold_acc %.4f (a17_gate 0.9387)' % (macro(A16), macro(a16(gold_acc))))
members = []
for tag in ('fullb', 'cohort', 'bs14'):
    f = os.path.join(g, K, tag + '_raw.csv')
    if not os.path.isfile(f) or not info.get(tag, {}).get('sub'):
        print(tag, 'no output (rc %s): not a member' % info.get(tag, {}).get('rc'))
        continue
    raw = leg(K, tag + '_raw.csv')
    cov = float(1 - (np.isnan(raw).all(1) | (raw == 0.5).all(1)).mean())
    p = leg(K, tag + '_sub.csv')
    m, r = macro(p), rho(p, n384)
    ok = [cov >= 0.95, m >= 0.925, m < 0.975, r <= 0.995]
    print('%-6s coverage %.3f  gold %.4f  rho vs n384 %.4f  rules %s  -> %s  (%.0f s on 58 studies)' % (
        tag, cov, m, r, ok, 'MEMBER' if all(ok) and len(members) < 2 else ('pass, cap reached' if all(ok) else 'no'),
        info[tag]['seconds']))
    if all(ok) and len(members) < 2:
        members.append((tag, p))
if not members:
    print('A19 GATE FAIL: no member')
    sys.exit(0)
slot = np.mean([rk(n384)] + [rk(p) for _, p in members], axis=0)
A19 = a16(slot)
d = macro(A19) - macro(A16)
print('A19 members %s: gold %.4f vs A16 %.4f = %+.4f  CI %s  guard (>= -0.003): %s' % (
    [t for t, _ in members], macro(A19), macro(A16), d, np.round(boot(A19, A16), 4), 'PASS' if d >= -0.003 else 'FAIL'))
print('A19 GATE', 'PASS' if d >= -0.003 else 'FAIL', 'members', ','.join(t for t, _ in members))
print('per finding delta', dict(zip(T, np.round(aucs(A19) - aucs(A16), 3))))
