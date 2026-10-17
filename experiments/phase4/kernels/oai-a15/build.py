"""Build A15 = A12 + V1 (our OrthoFoundation ViT-L ofl_l leg) as a fourth equal leg (pre-registered in docs/phase4_notes.md).

    python build.py <A12 notebook rsna-knee-oai-957s6f.ipynb> <n_seeds 2|4>

A12's cells are unchanged. One cell is added after A12's final blend; it never fails the notebook (A12's submission.csv
stays on any problem):
- It runs only if A12's s6fall stage completed, and reuses that stage's loaded s6fall module (_mod) and inputs (_I).
- It runs a second run_b pass with the V1 seeds in the models folder: --backbone ofl_l, --init (official OrthoFoundation-L,
  sha256 checked, converted exactly as in training), --freeze-blocks 16, one shared frozen trunk for all seeds.
- Eval stride: the smallest of 1, 2, 3, 4 whose time measured on gold-58 (rsna-knee-v1-gold58-dev, s/study/worker x N / 2
  x 1.10) fits the time left before 9 h minus 20 min. It is chosen from timing only; none fits -> A12 kept.
- Final = 3/4 x A12 value + 1/4 x rank(V1), fixed before any run (all four legs weigh 1/4). Studies V1 did not score keep
  A12's value; V1 coverage < 90 % -> A12 kept.
"""
import ast, json, sys
from pathlib import Path

nb = json.load(open(sys.argv[1], encoding="utf8"))
n_seeds = int(sys.argv[2])
assert n_seeds in (2, 4)
blend_i = [i for i, c in enumerate(nb["cells"]) if "[A12] final = 2/3" in "".join(c["source"])]
stage_i = [i for i, c in enumerate(nb["cells"]) if "[A12] nartaa's s6fall" in "".join(c["source"])]
assert len(blend_i) == 1 and len(stage_i) == 1 and blend_i[0] > stage_i[0], (blend_i, stage_i)
_v1 = (Path(__file__).resolve().parents[1] / "v1-train" / "v1_train.py").read_text()
_fn = next(n for n in ast.parse(_v1).body if isinstance(n, ast.FunctionDef) and n.name == "convert_state")
convert_src = ast.get_source_segment(_v1, _fn)

a15 = '''
# [A15] V1 = our OrthoFoundation ViT-L (ofl_l) leg, teach4opus labels, gold-58 excluded: a fourth equal leg after A12's blend.
# final = 3/4 x A12 value + 1/4 x rank(V1), fixed before any run.  Never fatal: A12's submission.csv stays on any problem.
import glob, hashlib, json, os, shutil, time
import numpy as np, pandas as pd
from pathlib import Path
V1_N_SEEDS = __N_SEEDS__
V1_PER = {1: 17.9, 2: 9.8, 3: 8.2, 4: 7.6}     # s/study/worker on gold-58 (rsna-knee-v1-gold58 with the real seeds 7+8: 17.9 at stride 1; 2-4 scaled from the dev run)
V1_OFL_SHA = '385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9'
V1_CONVERT = __CONVERT__
V1_OK = False
try:
    if not S6F_OK:
        raise RuntimeError('s6fall stage incomplete: no s6fall module / inputs to reuse')
    import torch
    _a12 = pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
    shutil.copyfile('/kaggle/working/submission.csv', '/kaggle/working/_a12_submission.csv')
    _n = len(_I.test_ids)
    _left = 9 * 3600 - (time.time() - NB_T0) - 20 * 60
    _stride = next((s for s in (1, 2, 3, 4) if V1_PER[s] * _n / 2 * 1.10 <= _left), None)
    print('V1_PLAN studies %d, %.0f min left, stride %s' % (_n, _left / 60, _stride), flush=True)
    if _stride is None:
        raise RuntimeError('no eval stride fits the time left')
    exec(V1_CONVERT)
    _src = [p for p in glob.glob('/kaggle/input/**/OrthoFoudation-L.pth', recursive=True) if '/orthofoundation-l-dinov3/' in p]
    if not _src:
        raise RuntimeError('OrthoFoundation-L checkpoint not attached')
    _h = hashlib.sha256()
    with open(_src[0], 'rb') as _f:
        for _b in iter(lambda: _f.read(1 << 24), b''):
            _h.update(_b)
    if _h.hexdigest() != V1_OFL_SHA:
        raise RuntimeError('OrthoFoundation-L checkpoint differs from the official release')
    _ofl = Path('/tmp/ofl_l_timm.pt')
    torch.save(convert_state(torch.load(_src[0], map_location='cpu', weights_only=True)), _ofl)
    _v1 = {Path(p).name: Path(p) for p in sorted(glob.glob('/kaggle/input/**/rsna-knee-v1-weights/E_prior_s*_f0.pt', recursive=True))}
    if len(_v1) != V1_N_SEEDS:
        raise RuntimeError('expected %d V1 models, found %s' % (V1_N_SEEDS, sorted(_v1)))
    shutil.rmtree(_I.models)
    _I.models.mkdir(parents=True)
    for _k, _p in _v1.items():
        os.symlink(_p, _I.models / _k)
    _first = sorted(_v1)[0]
    _mod.B_MODEL = dict(_mod.B_MODEL, file=_first, source_file=_first, sha256=hashlib.sha256(_v1[_first].read_bytes()).hexdigest())
    _bcmd0, _collect0 = _mod.b_cmd, _mod.collect_part

    def _bcmd_v1(I, split, assignment, out_dir, tmp_dir, deadline):
        c = _bcmd0(I, split, assignment, out_dir, tmp_dir, deadline)
        c[c.index('--backbone') + 1] = 'ofl_l'
        c[c.index('--eval-stride') + 1] = str(_stride)
        return c + ['--init', str(_ofl), '--freeze-blocks', '16']

    def _collect_v1(pdir, part, I):
        rp, lp = pdir / 'receipt.json', pdir / 'logits.json'
        rec = json.loads(rp.read_text())
        if rec.get('backbone') != 'ofl_l' or rec.get('models') != sorted(_v1) or rec.get('spatial_models') != 0:
            raise RuntimeError('V1 receipt: backbone %s models %s spatial %s' % (rec.get('backbone'), rec.get('models'), rec.get('spatial_models')))
        rec['models'], rec['spatial_models'] = [_mod.B_MODEL['file']], 1     # s6fall's single-model checks, on the seed mean
        rp.write_text(json.dumps(rec))
        L = json.loads(lp.read_text())
        lp.write_text(json.dumps({u: [np.asarray(a, np.float64).mean(0).tolist()] for u, a in L.items()}))
        return _collect0(pdir, part, I)

    _mod.b_cmd, _mod.collect_part = _bcmd_v1, _collect_v1
    _mod.OUT = Path('/tmp/k1v1_out')
    _mod.OUT.mkdir(parents=True, exist_ok=True)
    _t = time.time()
    _P, _Zv, _info, _failed = _mod.run_b(_I, 'B', _I.test_ids, 'test', _left)
    _vp = pd.read_csv(_mod.OUT / 'B' / 'B_probs.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    _labels = [c for c in _a12.columns if c != 'StudyInstanceUID']
    _vp = _vp.reindex(_a12['StudyInstanceUID'])[_labels]
    _ok = np.isfinite(_vp.values).all(axis=1)
    if _ok.sum() < 0.9 * len(_a12):
        raise RuntimeError('V1 scored only %d of %d studies' % (_ok.sum(), len(_a12)))
    _out = _a12.copy()
    for _c in _labels:
        _r = np.full(len(_a12), np.nan)
        _r[_ok] = pd.Series(_vp[_c].values[_ok]).rank(method='average', pct=True).values
        _v = _a12[_c].values.astype(np.float64)
        _out[_c] = np.where(_ok, 0.75 * _v + 0.25 * _r, _v)
    assert np.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert _out['StudyInstanceUID'].tolist() == _a12['StudyInstanceUID'].tolist()
    _out.to_csv('/kaggle/working/submission.csv', index=False)
    V1_OK = True
    _rho = {c: round(float(pd.Series(_a12[c].values[_ok]).corr(pd.Series(_vp[c].values[_ok]), method='spearman')), 3)
            for c in _labels} if _ok.sum() >= 3 else {}
    print('A15_BLEND_DONE V1 stride %d, %.0fs, scored %d of %d, failed %d; spearman(A12, V1) %s' % (
        _stride, time.time() - _t, _ok.sum(), len(_a12), len(_failed), _rho), flush=True)
except BaseException as _e:
    print('A15_BLEND_SKIPPED (A12 submission kept):', repr(_e)[:500], flush=True)
    if Path('/kaggle/working/_a12_submission.csv').is_file():
        shutil.copyfile('/kaggle/working/_a12_submission.csv', '/kaggle/working/submission.csv')
'''.replace("__N_SEEDS__", str(n_seeds)).replace("__CONVERT__", repr(convert_src))
nb["cells"].insert(blend_i[0] + 1, {"cell_type": "code", "metadata": {}, "source": a15, "outputs": [], "execution_count": None})
nb["cells"].insert(blend_i[0] + 1, {"cell_type": "markdown", "metadata": {},
                                    "source": "## A15: V1 (OrthoFoundation ViT-L, our labels) as a fourth equal leg (time-guarded, never fatal)"})
json.dump(nb, open("rsna-knee-oai-a15.ipynb", "w", encoding="utf8"), indent=1)
meta = json.load(open(Path(sys.argv[1]).with_name("kernel-metadata.json")))
meta.update(id="tranbadat/rsna-knee-oai-a15", title="rsna-knee-oai-a15", code_file="rsna-knee-oai-a15.ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + ["tranbadat/rsna-knee-v1-weights", "pritom2357/orthofoundation-l-dinov3"]
json.dump(meta, open("kernel-metadata.json", "w"), indent=1)
print("A15 notebook written, seeds", n_seeds, "cells", len(nb["cells"]), "datasets", meta["dataset_sources"])
