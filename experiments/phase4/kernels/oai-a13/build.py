"""Build A13 = A12 + S6T as a fourth equal leg (pre-registered in docs/phase4_notes.md).

    python build.py <A12 notebook rsna-knee-oai-957s6f.ipynb> <n_seeds 1|2>

S6T rides in s6fall's own compact_infer call: `--models2 <S6T folder> --backbone2 convnext_small --w2 0`.
- With w2 = 0, compact_infer's predictions.csv is s6fall's alone (p = 1 x p1 + 0 x p2).
- `--dump-logits` writes every model's logits per study: row 0 = s6fall, rows 1.. = S6T seeds.
- Before s6fall's collect_part() runs its strict checks, the S6T rows are taken out of logits.json and the file is
  rewritten with s6fall's row only, so every original check (shape (1, 12), predictions.csv == sigmoid(logits)) is
  unchanged.
Final = 3/4 x A12 value + 1/4 x rank(S6T) (A12 = 1/3 Raptor slot + 1/3 R7 + 1/3 s6fall, so all four legs weigh 1/4).
S6T is attached only if the stage has time for s6fall + S6T + 25 min; any S6T problem, or S6T coverage < 90 %, keeps A12's
file (A12 itself keeps A10's on any s6fall problem).
"""
import json, sys

nb = json.load(open(sys.argv[1], encoding="utf8"))
n_seeds = int(sys.argv[2])
assert n_seeds in (1, 2)
stage_i = [i for i, c in enumerate(nb["cells"]) if "[A12] nartaa's s6fall" in "".join(c["source"])]
blend_i = [i for i, c in enumerate(nb["cells"]) if "[A12] final = 2/3" in "".join(c["source"])]
assert len(stage_i) == 1 and len(blend_i) == 1, (stage_i, blend_i)
stage = "".join(nb["cells"][stage_i[0]]["source"])


def sub_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


s6t_hook = '''
    # [A13] S6T (our S6-recipe retrain on teach4opus, gold-58 excluded) as models2 in s6fall's compact_infer call
    S6T_Z, S6T_ON = {}, False
    try:
        _s6t = sorted(Path('/kaggle/input').glob('**/rsna-knee-s6t-weights/E_prior_s*_f0.pt'))[:__N_SEEDS__]
        _n_test = len(_I.test_ids)
        _need = (1.7 + 0.85 * len(_s6t)) * 3600 * _n_test / 1322 + 25 * 60
        _left_now = 9 * 3600 - (time.time() - NB_T0)
        if len(_s6t) == __N_SEEDS__ and _left_now >= _need:
            _s6t_dir = Path('/tmp/s6t_models'); _s6t_dir.mkdir(exist_ok=True)
            for _p in _s6t:
                if not (_s6t_dir / _p.name).exists():
                    os.symlink(_p, _s6t_dir / _p.name)
            _orig_b_cmd, _orig_collect = _mod.b_cmd, _mod.collect_part

            def _b_cmd_s6t(I, split, assignment, out_dir, tmp_dir, deadline):
                return _orig_b_cmd(I, split, assignment, out_dir, tmp_dir, deadline) + [
                    '--models2', str(_s6t_dir), '--arm2', 'E_prior', '--backbone2', 'convnext_small', '--w2', '0']

            def _collect_s6t(pdir, part, I):
                _lp = pdir / 'logits.json'
                if _lp.is_file():
                    _full = json.loads(_lp.read_text())
                    for _u, _a in _full.items():
                        _a = np.asarray(_a, np.float64)
                        if _a.ndim == 2 and len(_a) == 1 + __N_SEEDS__ and np.isfinite(_a).all():
                            S6T_Z[_u] = _a[1:].mean(0)
                    _lp.write_text(json.dumps({_u: [_a[0]] for _u, _a in _full.items()}))
                return _orig_collect(pdir, part, I)
            _mod.b_cmd, _mod.collect_part = _b_cmd_s6t, _collect_s6t
            S6T_ON = True
        print('S6T_ATTACH', S6T_ON, 'models', [p.name for p in _s6t], 'need %.0f min, left %.0f min' % (_need / 60, _left_now / 60), flush=True)
    except BaseException as _e:
        print('S6T_ATTACH_FAILED', repr(_e)[:300], flush=True)
'''.replace("__N_SEEDS__", str(n_seeds))
stage = sub_once(stage, "    _I = _mod.prepare_inputs()\n", "    _I = _mod.prepare_inputs()\n" + s6t_hook)
stage = sub_once(stage, "import pandas as pd\n", "import numpy as np\nimport pandas as pd\n")
stage = stage.replace("[A12] nartaa's s6fall", "[A13 base = A12] nartaa's s6fall")
nb["cells"][stage_i[0]]["source"] = stage

a13_blend = '''
# [A13] final = 3/4 x A12 value + 1/4 x rank(S6T): four equal legs (Raptor slot, R7, s6fall, S6T), fixed before any run.
# Runs after A12's blend; never fatal: A12's submission.csv stays on any problem.
import numpy as np, pandas as pd, shutil
try:
    if not S6T_ON or not S6F_OK:
        raise RuntimeError('S6T not attached or s6fall stage incomplete')
    _base = pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
    shutil.copyfile('/kaggle/working/submission.csv', '/kaggle/working/_a12_submission.csv')
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _lab_order = list(_mod.LAB)
    _ok = np.array([u in S6T_Z for u in _base['StudyInstanceUID']])
    if _ok.sum() < 0.9 * len(_base):
        raise RuntimeError('S6T scored only %d of %d studies' % (_ok.sum(), len(_base)))
    _Z = np.full((len(_base), len(_lab_order)), np.nan)
    for _i, _u in enumerate(_base['StudyInstanceUID']):
        if _u in S6T_Z:
            _Z[_i] = S6T_Z[_u]
    _out = _base.copy()
    for _c in _labels:
        _j = _lab_order.index(_c)
        _r = np.full(len(_base), np.nan)
        _r[_ok] = pd.Series(_Z[_ok, _j]).rank(method='average', pct=True).values
        _v = _base[_c].values.astype(np.float64)
        _out[_c] = np.where(_ok, 0.75 * _v + 0.25 * _r, _v)
    assert np.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert _out['StudyInstanceUID'].tolist() == _base['StudyInstanceUID'].tolist()
    _out.to_csv('/kaggle/working/submission.csv', index=False)
    _rho = {c: round(float(pd.Series(_base[c].values[_ok]).corr(pd.Series(_Z[_ok, _lab_order.index(c)]), method='spearman')), 3)
            for c in _labels} if _ok.sum() >= 3 else {}
    print('A13_BLEND_DONE S6T scored %d of %d; spearman(A12, S6T) %s' % (_ok.sum(), len(_base), _rho), flush=True)
except BaseException as _e:
    print('A13_BLEND_SKIPPED (A12 submission kept):', repr(_e)[:500], flush=True)
'''
cell = {"cell_type": "code", "metadata": {}, "source": a13_blend, "outputs": [], "execution_count": None}
nb["cells"].insert(blend_i[0] + 1, cell)
nb["cells"].insert(stage_i[0] - 1 if nb["cells"][stage_i[0] - 1]["cell_type"] == "markdown" else stage_i[0],
                   {"cell_type": "markdown", "metadata": {}, "source": "## A13: S6T as a fourth equal leg (inside s6fall's pass, time-guarded, never fatal)"})
json.dump(nb, open("rsna-knee-oai-a13.ipynb", "w", encoding="utf8"), indent=1)
print("A13 notebook written, seeds", n_seeds)
