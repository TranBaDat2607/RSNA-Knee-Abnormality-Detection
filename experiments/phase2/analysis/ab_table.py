"""Per-finding table for fold-0 A/B runs: val vs teach4 (best epoch) and gold-58, plus paired bootstrap of val macro diff."""
import json, sys, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
teach = pd.read_csv('teach4.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
gold = pd.read_csv('C:/Users/Admin/Desktop/RSNA-Knee-Abnormality-Detection/data/gold_annotated.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T].astype(float)
runs = {}
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, 'run', '*', 'fold0', 'oof.csv')):
        tag = f.split(os.sep)[-3] if os.sep in f else f.split('/')[-3]
        tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
        o = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
        gids = json.load(open(os.path.join(os.path.dirname(f), 'run.json')))['gold_ids']
        g = pd.DataFrame(np.load(os.path.join(os.path.dirname(f), 'gold.npy')), index=[str(x) for x in gids], columns=T).reindex(gold.index)
        runs[tag] = (o, g)
ids = sorted(set.intersection(*[set(o.index) for o, _ in runs.values()]))
y = (teach.reindex(ids) > 0.5).astype(int)
def per(yv, p): return {t: roc_auc_score(yv[t], p[t]) for t in T}
rows = {}
for tag, (o, g) in runs.items():
    v = per(y, o.reindex(ids)); gg = per((gold > 0.5).astype(int), g)
    rows[tag] = {**{t: v[t] for t in T}, 'VAL': np.mean(list(v.values())), 'GOLD': np.mean(list(gg.values())), 'gold_MCL': gg['MCL'], 'gold_LatMen': gg['Lateral Meniscus']}
print(pd.DataFrame(rows).round(4).to_string())
base = 'gab_slot' if 'gab_slot' in runs else list(runs)[0]
rng = np.random.default_rng(0); n = len(ids)
for tag in runs:
    if tag == base: continue
    diffs = []
    for _ in range(300):
        b = rng.integers(0, n, n); yb = y.iloc[b]
        ok = [t for t in T if 0 < yb[t].sum() < n]
        a = np.mean([roc_auc_score(yb[t], runs[tag][0].reindex(ids).iloc[b][t]) for t in ok])
        c = np.mean([roc_auc_score(yb[t], runs[base][0].reindex(ids).iloc[b][t]) for t in ok])
        diffs.append(a - c)
    d = np.array(diffs); print(f'{tag} - {base}: val macro diff {d.mean():+.4f}  90% CI [{np.percentile(d,5):+.4f}, {np.percentile(d,95):+.4f}]  P>0 {np.mean(d>0):.2f}')
