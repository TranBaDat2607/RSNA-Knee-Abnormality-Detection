"""ft96b decision: are the v8/v5-started siblings of ftC accepted, and does the family rank-mean beat ftC alone?

usage: python ft_family.py <gold58_labels.csv> <teach4.csv> <run_dir>...   (each run_dir has run.json + *_pred_ep2.npy)

Pre-registered (docs/phase3_notes.md): a sibling is accepted if its ep2 gold-58 >= 0.912; the family mean replaces
ftC if its gold-58 >= ftC - 0.002. The 300-study holdout (same split_seed for every run, never trained on) scored
against teach4 is reported as the second, larger signal.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']
gold = pd.read_csv(sys.argv[1], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
teach = pd.read_csv(sys.argv[2], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
rk = lambda x: rankdata(x, axis=0) / len(x)


def macro(y, p):
    yb = (y > 0.5).astype(int)
    return float(np.mean([roc_auc_score(yb[:, j], p[:, j]) for j in range(len(T)) if 0 < yb[:, j].sum() < len(yb)]))


runs = {}
for d in sys.argv[3:]:
    rj = json.load(open(os.path.join(d, 'run.json')))
    ep = max(h['ep'] for h in rj['history'] if 'gold_macro' in h)
    g = pd.DataFrame(np.load(os.path.join(d, f'gold_pred_ep{ep}.npy')), index=[str(u) for u in rj['gold_ids']], columns=T).loc[gold.index]
    h = pd.DataFrame(np.load(os.path.join(d, f'holdout_pred_ep{ep}.npy')), index=[str(u) for u in rj['holdout_ids']], columns=T)
    runs[os.path.basename(d.rstrip('/\\'))] = (g, h)
hold_ids = sorted(set.intersection(*[set(h.index) for _, h in runs.values()]))
yh = teach.loc[hold_ids].values
print(f'shared holdout studies: {len(hold_ids)}')
acc = {}
for n, (g, h) in runs.items():
    gm, hm = macro(gold.values, g.values), macro(yh, h.loc[hold_ids].values)
    ok = n == 'ftC' or gm >= 0.912
    acc[n] = ok
    print(f'{n:6s} gold-58 {gm:.4f}  holdout(teach4) {hm:.4f}  {"accepted" if ok else "REJECTED"}')
keep = [n for n in runs if acc[n]]
gm = np.mean([rk(runs[n][0].values) for n in keep], axis=0)
hm = np.mean([rk(runs[n][1].loc[hold_ids].values) for n in keep], axis=0)
base_g = macro(gold.values, runs['ftC'][0].values)
fam_g = macro(gold.values, gm)
print(f'family mean of {keep}: gold-58 {fam_g:.4f} (ftC {base_g:.4f}), holdout {macro(yh, hm):.4f} '
      f'(ftC {macro(yh, runs["ftC"][1].loc[hold_ids].values):.4f})')
print('DECISION:', 'use family mean' if len(keep) > 1 and fam_g >= base_g - 0.002 else 'keep ftC alone')
r = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method='spearman') for j in range(len(T))])
names = list(runs)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f'holdout spearman {names[i]}-{names[j]}: {r(runs[names[i]][1].loc[hold_ids].values, runs[names[j]][1].loc[hold_ids].values):.3f}')
