"""Build A20 = A17 with our S6T sharing the S6 slot with nartaa's s6fall (docs/phase4_notes.md, 2026-10-21 16:00 UTC).

    python build.py <rsna-knee-oai-a17.ipynb>

A17's cells are unchanged except the s6fall stage, which gets A13's hook: S6T (heliosli's S6 recipe retrained by us on
teach4opus, gold-58 excluded) rides in s6fall's own compact_infer call as `--models2 ... --w2 0`, so s6fall's predictions
and strict checks are untouched and S6T costs only its GPU forward pass. 2, 1 or 0 seeds by the time left. After A17's
blend, one cell rewrites the file as final = 2/3 x A16 + 1/3 x rank(1/2 rank(s6fall) + 1/2 rank(S6T)); a study S6T did not
score keeps A17's value; any problem or S6T coverage < 90 % keeps A17's file.
"""
import json, sys

nb = json.load(open(sys.argv[1], encoding='utf8'))
stage_i = [i for i, c in enumerate(nb['cells']) if "[A12] nartaa's s6fall" in ''.join(c['source'])]
blend_i = [i for i, c in enumerate(nb['cells']) if '[A12] final = 2/3' in ''.join(c['source'])]
assert len(stage_i) == 1 and len(blend_i) == 1 and blend_i[0] == stage_i[0] + 1 == len(nb['cells']) - 1, (stage_i, blend_i)
stage = ''.join(nb['cells'][stage_i[0]]['source'])


def sub_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


hook = '''
    # [A20] S6T (our S6-recipe retrain on teach4opus, gold-58 excluded) as models2 in s6fall's compact_infer call (A13's hook)
    S6T_Z, S6T_ON, S6T_SEEDS = {}, False, 0
    try:
        _cands = []
        for _pat in ('/kaggle/input/rsna-knee-s6t-weights/', '/kaggle/input/*/rsna-knee-s6t-weights/',
                     '/kaggle/input/*/*/rsna-knee-s6t-weights/', '/kaggle/input/*/*/*/rsna-knee-s6t-weights/'):
            _cands += [Path(p) for p in sorted(__import__('glob').glob(_pat + 'E_prior_s*_f0.pt'))]
        _cands = sorted({p.name: p for p in _cands}.values(), key=lambda p: p.name)
        _n_test = len(_I.test_ids)
        _left_now = 9 * 3600 - (time.time() - NB_T0)
        _need = lambda k: (1.7 + 0.85 * k) * 3600 * _n_test / 1322 + 25 * 60
        S6T_SEEDS = next((k for k in (2, 1) if len(_cands) >= k and _left_now >= _need(k)), 0)
        if S6T_SEEDS:
            _s6t_dir = Path('/tmp/s6t_models'); _s6t_dir.mkdir(exist_ok=True)
            for _p in _cands[:S6T_SEEDS]:
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
                        _a = __import__('numpy').asarray(_a, dtype='float64')
                        if _a.ndim == 2 and len(_a) == 1 + S6T_SEEDS and __import__('numpy').isfinite(_a).all():
                            S6T_Z[_u] = _a[1:].mean(0)
                    _lp.write_text(json.dumps({_u: [_a[0]] for _u, _a in _full.items()}))
                return _orig_collect(pdir, part, I)
            _mod.b_cmd, _mod.collect_part = _b_cmd_s6t, _collect_s6t
            S6T_ON = True
        print('S6T_ATTACH', S6T_ON, 'seeds', S6T_SEEDS, 'found', [p.name for p in _cands],
              'need(2) %.0f min, left %.0f min' % (_need(2) / 60, _left_now / 60), flush=True)
    except BaseException as _e:
        print('S6T_ATTACH_FAILED', repr(_e)[:300], flush=True)
'''
stage = sub_once(stage, '    _I = _mod.prepare_inputs()\n', '    _I = _mod.prepare_inputs()\n' + hook)
stage = sub_once(stage, "# [A12] nartaa's s6fall", "# [A20: A12's s6fall stage + S6T as models2] nartaa's s6fall")
nb['cells'][stage_i[0]]['source'] = stage

blend = '''# [A20] final = 2/3 x A16 + 1/3 x S6 slot, slot = rank(1/2 rank(s6fall) + 1/2 rank(S6T)), fixed before any run.
# Runs after A17's blend; never fatal: A17's submission.csv stays on any problem. A study S6T did not score keeps A17's value.
import numpy as np, pandas as pd, shutil
try:
    if not (S6F_OK and S6T_ON):
        raise RuntimeError('s6fall stage incomplete or S6T not attached')
    shutil.copyfile('/kaggle/working/submission.csv', '/kaggle/working/_a17_submission.csv')
    _base = pd.read_csv('/kaggle/working/_a10_submission.csv', dtype={'StudyInstanceUID': str})   # A16's file, kept by the stage
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _ids = _base['StudyInstanceUID'].tolist()
    _sf = pd.read_csv('/kaggle/working/s6fall_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    _sf = _sf.reindex(_ids)[_labels]
    _okf = np.isfinite(_sf.values).all(axis=1)
    _lab_order = list(_mod.LAB)
    _okt = np.array([u in S6T_Z for u in _ids])
    if _okf.sum() < 0.9 * len(_ids) or _okt.sum() < 0.9 * len(_ids):
        raise RuntimeError('coverage s6fall %d, S6T %d of %d' % (_okf.sum(), _okt.sum(), len(_ids)))
    _Z = np.full((len(_ids), len(_lab_order)), np.nan)
    for _i, _u in enumerate(_ids):
        if _u in S6T_Z:
            _Z[_i] = S6T_Z[_u]
    _pct = lambda v: pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values
    _both = _okf & _okt
    _out = _base.copy()
    for _c in _labels:
        _rf = np.full(len(_ids), np.nan); _rf[_okf] = _pct(_sf[_c].values[_okf])
        _rt = np.full(len(_ids), np.nan); _rt[_okt] = _pct(_Z[_okt, _lab_order.index(_c)])
        _slot = _rf.copy()
        _slot[_both] = _pct(0.5 * _rf[_both] + 0.5 * _rt[_both])
        _v = _base[_c].values.astype(np.float64)
        _out[_c] = np.where(_okf, (2.0 / 3.0) * _v + (1.0 / 3.0) * _slot, _v)
    assert np.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert _out['StudyInstanceUID'].tolist() == _ids
    _out.to_csv('/kaggle/working/submission.csv', index=False)
    _rho = {c: round(float(pd.Series(_sf[c].values[_both]).corr(pd.Series(_Z[_both, _lab_order.index(c)]), method='spearman')), 3)
            for c in _labels} if _both.sum() >= 3 else {}
    print('A20_BLEND_DONE seeds %d; s6fall %d, S6T %d, both %d of %d; spearman(s6fall, S6T) %s' % (
        S6T_SEEDS, _okf.sum(), _okt.sum(), _both.sum(), len(_ids), _rho), flush=True)
except BaseException as _e:
    print('A20_BLEND_SKIPPED (A17 submission kept):', repr(_e)[:500], flush=True)
'''
nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': blend, 'outputs': [], 'execution_count': None})
nb['cells'][1]['source'] = ("## A20: A17 with our S6T sharing the S6 slot with nartaa's s6fall\n\n"
                            "A16's cells and A12's s6fall stage are unchanged; S6T (our retrain of heliosli's S6 recipe on our "
                            "teach4opus labels) rides in s6fall's own inference call. final = 2/3 x A16 + 1/3 x rank(1/2 rank(s6fall) "
                            "+ 1/2 rank(S6T)). Time-guarded, never fatal (A17's file stays on any S6T problem).")
json.dump(nb, open('rsna-knee-oai-a20.ipynb', 'w', encoding='utf8'), indent=1)
meta = json.load(open('../oai-a17/kernel-metadata.json'))
meta.update(id='tranbadat/rsna-knee-oai-a20', title='rsna-knee-oai-a20', code_file='rsna-knee-oai-a20.ipynb')
meta['dataset_sources'] = meta['dataset_sources'] + ['tranbadat/rsna-knee-s6t-weights']
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('A20 cells', len(nb['cells']))
