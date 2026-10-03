"""Score a k-fold run: gold-58 (fold-mean) and OOF vs teach4 per tag, the tag blend, and build R2 targets.

usage: python analyze_run.py <run_dir> <tag1,tag2,...> [r2_out.csv] [w_label]
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']
HERE = os.path.dirname(os.path.abspath(__file__))
gold = pd.read_csv('C:/Users/Admin/Desktop/RSNA-Knee-Abnormality-Detection/data/gold_annotated.csv')
gold['StudyInstanceUID'] = gold.StudyInstanceUID.astype(str)
gold = gold.set_index('StudyInstanceUID')[T].astype(float)
teach = pd.read_csv(os.path.join(HERE, 'teach4.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]


def macro(y, p, binar=True):
    a = {t: roc_auc_score((y[:, j] > 0.5).astype(int) if binar else y[:, j], p[:, j]) for j, t in enumerate(T)}
    return np.mean(list(a.values())), a


def rk(x):
    return rankdata(x, axis=0) / len(x)


run, tags = sys.argv[1], sys.argv[2].split(',')
oofs, golds = {}, {}
for tag in tags:
    fo = sorted(glob.glob(os.path.join(run, tag, 'fold*', 'oof.csv')))
    if not fo:
        print(tag, 'no folds'); continue
    oofs[tag] = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fo]).set_index('StudyInstanceUID')[T]
    gids = json.load(open(os.path.join(os.path.dirname(fo[0]), 'run.json')))['gold_ids']
    g = np.mean([np.load(os.path.join(os.path.dirname(f), 'gold.npy')) for f in fo], axis=0)
    golds[tag] = pd.DataFrame(g, index=[str(x) for x in gids], columns=T).reindex(gold.index)
    hist = [json.load(open(os.path.join(os.path.dirname(f), 'run.json')))['history'] for f in fo]
    best = [max((h for h in hh if 'val_ref_macro' in h), key=lambda h: h['val_ref_macro']) for hh in hist]
    o = oofs[tag]
    om, _ = macro(teach.reindex(o.index).values, o.values)
    gm, gper = macro(gold.values, golds[tag].values)
    print(f"{tag}: folds {len(fo)} | best eps {[b['ep'] for b in best]} | OOF vs teach4 {om:.4f} | gold-58 {gm:.4f}")
    print('   gold per:', {k: round(v, 3) for k, v in gper.items()})
if len(golds) > 1:
    gb = np.mean([rk(golds[t].values) for t in golds], axis=0)
    ids = sorted(set.intersection(*[set(o.index) for o in oofs.values()]))
    ob = np.mean([rk(oofs[t].reindex(ids).values) for t in oofs], axis=0)
    print(f"blend: OOF vs teach4 {macro(teach.reindex(ids).values, ob)[0]:.4f} | gold-58 {macro(gold.values, gb)[0]:.4f}")
    a, b = list(oofs)
    rho = np.mean([pd.Series(oofs[a].reindex(ids)[t].values).corr(pd.Series(oofs[b].reindex(ids)[t].values), method='spearman') for t in T])
    print(f"OOF spearman between tags: {rho:.3f}")
if len(sys.argv) > 3 and oofs:
    w_label = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
    ids = sorted(set.intersection(*[set(o.index) for o in oofs.values()]))
    pm = sum(oofs[t].reindex(ids) for t in oofs) / len(oofs)
    r2 = w_label * teach.reindex(ids) + (1 - w_label) * pm
    print('R2 targets vs gold (n/a: gold not in OOF);', 'n', len(r2), 'mean', r2.mean().round(3).to_dict())
    r2.rename_axis('StudyInstanceUID').reset_index().to_csv(sys.argv[3], index=False)
