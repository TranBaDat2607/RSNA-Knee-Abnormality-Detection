"""Gold-58 check of the compact legs (R7 plain / flip TTA, S6) inside the A6-style blend.

usage: python gold_compact.py <oai_gold58_out_dir> <oai_gold58_r7_out_dir>

R7 never trained on gold (manifest gold_in_training 0); nartaa was selected on gold, so blends that lean on it are
optimistic. Used only for the pre-registered A8 rule and as a regression guard, never to tune weights.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']
g58, r7 = sys.argv[1], sys.argv[2]
y = pd.read_csv(os.path.join(g58, 'gold_root', 'gold_labels.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
Y = (y.values > 0.5).astype(int)


def leg(d, name):
    p = os.path.join(d, f'gold_{name}.csv')
    if not os.path.isfile(p):
        return None
    return pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[y.index, T].values


def rk(x):
    return rankdata(x, axis=0) / len(x)


def aucs(p, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    return np.array([roc_auc_score(Y[idx, j], p[idx, j]) if 0 < Y[idx, j].sum() < len(idx) else np.nan for j in range(len(T))])


def boot(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(Y), len(Y))
        d.append(np.nanmean(aucs(a, idx)) - np.nanmean(aucs(b, idx)))
    return np.percentile(d, [2.5, 97.5])


acc, eff = leg(g58, 'acc'), leg(g58, 'eff')
legs = {'nartaa384': acc, 'nartaa224': eff, 'reader': leg(g58, 'reader'),
        'r7_plain': leg(r7, 'r7_plain'), 'r7_flip': leg(r7, 'r7_flip'), 's6': leg(r7, 's6')}
legs = {k: v for k, v in legs.items() if v is not None}
rows = {k: aucs(v) for k, v in legs.items()}
raptor = 0.65 * rk(acc) + 0.35 * rk(eff)
blends = {}
for name in ('r7_plain', 'r7_flip'):
    if name in legs:
        blends['A6[' + name + ']'] = 0.5 * rk(raptor) + 0.5 * rk(legs[name])
        if 's6' in legs:
            blends['A9[' + name + '+s6 0.3]'] = 0.5 * rk(raptor) + 0.5 * rk(0.7 * rk(legs[name]) + 0.3 * rk(legs['s6']))
rows.update({k: aucs(v) for k, v in blends.items()})
tab = pd.DataFrame(rows, index=T).T
tab['macro'] = tab[T].mean(1)
pd.set_option('display.width', 250)
print(tab.round(4).to_string())
if 'r7_plain' in legs and 'r7_flip' in legs:
    rho = np.mean([spearmanr(legs['r7_plain'][:, j], legs['r7_flip'][:, j])[0] for j in range(len(T))])
    print('R7 flip vs plain: mean Spearman %.4f, macro diff %.4f CI %s' % (
        rho, tab.loc['r7_flip', 'macro'] - tab.loc['r7_plain', 'macro'], boot(legs['r7_flip'], legs['r7_plain']).round(4)))
    a, b = blends['A6[r7_flip]'], blends['A6[r7_plain]']
    print('A6 blend flip vs plain: macro diff %.4f CI %s' % (np.nanmean(aucs(a)) - np.nanmean(aucs(b)), boot(a, b).round(4)))
for k in [k for k in blends if k.startswith('A9')]:
    base = blends[k.replace('A9', 'A6').replace('+s6 0.3', '')]
    print('%s vs A6: macro diff %.4f CI %s' % (k, np.nanmean(aucs(blends[k])) - np.nanmean(aucs(base)), boot(blends[k], base).round(4)))
for f in ('r7_plain_receipt.json', 'r7_flip_receipt.json', 's6_receipt.json'):
    p = os.path.join(r7, f)
    if os.path.isfile(p):
        r = json.load(open(p))
        print(f, 'seconds', r.get('seconds'), 'per_study', r.get('per_study_mean'), 'scored', r.get('scored'))
