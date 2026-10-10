"""Build a diagnostic notebook that runs the public OAI-track legs on the 58 gold-labelled training studies.

    python build.py <goodpjw_nb.ipynb> <nartaa_eff_nb.ipynb>

Writes gold_acc.csv (nartaa 0.949, mirror TTA), gold_eff.csv (nartaa 0.945, 224 crop) and gold_reader.csv (goodpjw
2.5D ConvNeXt reader, raw probabilities) for a stand-in competition root whose "test" set is the 58 gold studies
(test_series/ entries are symlinks into train_series/). nartaa used gold-58 for checkpoint selection, so its gold AUC
is optimistic; the reader excluded gold from training. Main use: correlations between legs, and a regression guard.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_ROOT = "/kaggle/working/gold_root"


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


MAKE_ROOT = r'''# Stand-in competition root: "test" = the 58 studies of train.csv with expert labels.
import os, glob
import pandas as pd
COMP = next(c for c in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection')
            if os.path.exists(c + '/train.csv'))
G = %r
os.makedirs(G + '/test_series', exist_ok=True)
tr = pd.read_csv(COMP + '/train.csv', dtype={'StudyInstanceUID': str})
ss = pd.read_csv(COMP + '/sample_submission.csv', nrows=1)
labels = [c for c in ss.columns if c != 'StudyInstanceUID']
gold = tr[tr[labels].notna().all(axis=1)].reset_index(drop=True)
print('gold studies', len(gold))
gold[['StudyInstanceUID']].to_csv(G + '/test.csv', index=False)
gold[['StudyInstanceUID'] + labels].to_csv(G + '/gold_labels.csv', index=False)
sub = gold[['StudyInstanceUID']].copy()
for c in labels: sub[c] = 0.5
sub.to_csv(G + '/sample_submission.csv', index=False)
ser = pd.read_csv(COMP + '/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
ser[ser.StudyInstanceUID.isin(set(gold.StudyInstanceUID))].to_csv(G + '/test_series.csv', index=False)
for u in gold.StudyInstanceUID:
    dst = G + '/test_series/' + u
    if not os.path.exists(dst):
        os.symlink(COMP + '/train_series/' + u, dst)
shutil_ok = True
print('root ready', G, len(os.listdir(G + '/test_series')))
''' % GOLD_ROOT


def patch_nartaa(src, out_name):
    a = 'def find_test_root():\n    cands = ['
    assert a in src, "find_test_root layout changed"
    src = src.replace(a, a + f'"{GOLD_ROOT}", ')
    # the stand-in root lives under /kaggle/working, so the existence check must accept it first
    b = 'out = "/kaggle/working/submission.csv"'
    assert b in src
    return src.replace(b, f'out = "/kaggle/working/{out_name}"')


def main():
    goodpjw_nb, eff_nb = sys.argv[1], sys.argv[2]
    gcells = json.load(open(goodpjw_nb, encoding="utf8"))["cells"]
    ecells = json.load(open(eff_nb, encoding="utf8"))["cells"]
    acc_check, acc_src = "".join(gcells[6]["source"]), "".join(gcells[7]["source"])
    eff_check, eff_src = "".join(ecells[3]["source"]), "".join(ecells[4]["source"])
    acc = acc_check + "\n\n" + patch_nartaa(acc_src, "gold_acc.csv")
    eff = eff_check + "\n\n" + patch_nartaa(eff_src, "gold_eff.csv")
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Gold-58 diagnostic: OAI-track public legs\n", __doc__]},
        code(MAKE_ROOT),
        code("open('/tmp/acc_run.py','w').write(" + repr(acc) + ")\nopen('/tmp/eff_run.py','w').write(" + repr(eff) + ")\n"),
        code("import subprocess, sys\nfor s in ('/tmp/acc_run.py', '/tmp/eff_run.py'):\n"
             "    print(s, 'exit', subprocess.run([sys.executable, '-u', s]).returncode, flush=True)\n"),
    ]
    # Part B source cells (own_src/*.py writers) unchanged, then run the reader on the stand-in root
    cells += [c for c in gcells[9:13]]
    cells.append(code(r'''import glob, os, subprocess, sys
A = next(d.rstrip('/') for pat in ['/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/'] for d in sorted(glob.glob(pat))
         if os.path.exists(os.path.join(d, 'knee.py')))
T = '/kaggle/working/_own_env'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', T, '--find-links', A,
                'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'pyjpegls', 'python-gdcm'], check=False)
env = dict(os.environ, PYTHONPATH=T + os.pathsep + os.environ.get('PYTHONPATH', ''))
ck = sorted(glob.glob(A + '/cnxt_v0_fold[012].pt'))
print('reader exit', subprocess.run([sys.executable, '/kaggle/working/own_src/infer.py', %r, 'test', '/kaggle/working/gold_reader.csv'] + ck,
                                   env=env).returncode, flush=True)
''' % GOLD_ROOT))
    cells.append(code(r'''# Gold-58 macro AUC of each leg and of simple rank blends (nartaa is gold-selected: optimistic).
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
y = pd.read_csv('/kaggle/working/gold_root/gold_labels.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
L = list(y.columns)
legs = {}
for n in ('acc', 'eff', 'reader'):
    f = f'/kaggle/working/gold_{n}.csv'
    if os.path.exists(f):
        legs[n] = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[y.index, L]
def auc(p):
    r = {c: roc_auc_score((y[c] > 0.5).astype(int), p[c]) for c in L if (y[c] > 0.5).nunique() == 2}
    return float(np.mean(list(r.values()))), r
for n, p in legs.items():
    m, r = auc(p); print(n, round(m, 4), {k: round(v, 3) for k, v in r.items()})
rk = {n: p.rank(pct=True) for n, p in legs.items()}
if 'acc' in rk and 'eff' in rk:
    print('acc-eff rank corr', rk['acc'].corrwith(rk['eff']).round(3).to_dict())
if 'acc' in rk and 'reader' in rk:
    print('acc-reader rank corr', rk['acc'].corrwith(rk['reader']).round(3).to_dict())
'''))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(os.path.join(HERE, "rsna-knee-oai-gold58.ipynb"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("cells", len(cells))


if __name__ == "__main__":
    main()
