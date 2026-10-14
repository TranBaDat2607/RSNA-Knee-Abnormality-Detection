"""Build A12 = A10 (heliosli 0.9573 recipe) + nartaa's s6fall as a third equal leg.

    python build.py <a10_notebook.ipynb> <rsna-knee-s6fall-solo.py>

After A10's blend cell has written submission.csv (= 0.5 Raptor slot + 0.5 R7, in ranks), nartaa's s6fall script is loaded
unchanged as a module and only its prepare_inputs() + run_b() run (B = s6fall; component A, weight 0 in the solo version, is
skipped). Final = 2/3 x A10 value + 1/3 x rank(s6fall), i.e. the Raptor slot, R7 and s6fall at 1/3 each. s6fall runs only if
at least 70 min of the 9 h remain, under a cap that leaves 25 min; any failure keeps A10's submission unchanged, and a study
s6fall did not score keeps its A10 value.
"""
import base64, json, sys


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


nb = json.load(open(sys.argv[1], encoding='utf8'))
script = open(sys.argv[2], 'rb').read()
stage = '''
# [A12] nartaa's s6fall (0.950 alone, gold-58 excluded, no OAI) as a third equal leg; never fatal: A10's file stays on any failure
import base64, hashlib, json, os, shutil, sys, time
import pandas as pd
from pathlib import Path
S6F_OK = False
try:
    _left = 9 * 3600 - (time.time() - NB_T0)
    if _left < 70 * 60:
        raise RuntimeError('only %.0f min left' % (_left / 60))
    _src = base64.b64decode(__S6F_B64__)
    print('S6F script sha256', hashlib.sha256(_src).hexdigest(), flush=True)
    shutil.copyfile('/kaggle/working/submission.csv', '/kaggle/working/_a10_submission.csv')
    os.environ['K1_TMP'] = '/tmp/k1s6f'
    _mod = type(sys)('s6fall_mod')
    _mod.__dict__['__file__'] = '/tmp/s6fall_solo.py'
    Path('/tmp/s6fall_solo.py').write_bytes(_src)
    exec(compile(_src, '/tmp/s6fall_solo.py', 'exec'), _mod.__dict__)
    _mod.OUT = Path('/tmp/k1s6f_out'); _mod.OUT.mkdir(parents=True, exist_ok=True)
    _mod.TMP.mkdir(parents=True, exist_ok=True)
    _mod._LOG = open(_mod.OUT / 'k1_log.txt', 'a')
    _mod._RECEIPT['env'] = _mod.probe_env()
    _I = _mod.prepare_inputs()
    print('S6F_PROBE', _mod.probe_b_stack(_I), flush=True)
    _budget = 9 * 3600 - (time.time() - NB_T0) - 25 * 60
    _t = time.time()
    _P, _Z, _info, _failed = _mod.run_b(_I, 'B', _I.test_ids, 'test', _budget)
    _df = pd.read_csv(_mod.OUT / 'B' / 'B_probs.csv', dtype={'StudyInstanceUID': str})
    _df[['StudyInstanceUID'] + _mod.LAB].to_csv('/kaggle/working/s6fall_sub.csv', index=False)
    S6F_OK = True
    print('S6F_STAGE_DONE seconds %.0f failed %d' % (time.time() - _t, len(_failed)), flush=True)
except BaseException as _e:
    print('S6F_SKIPPED', repr(_e)[:500], flush=True)
finally:
    if not Path('/kaggle/working/submission.csv').is_file() and Path('/kaggle/working/_a10_submission.csv').is_file():
        shutil.copyfile('/kaggle/working/_a10_submission.csv', '/kaggle/working/submission.csv')
        print('S6F restored the A10 submission', flush=True)
'''.replace('__S6F_B64__', repr(base64.b64encode(script).decode()))
blend = '''
# [A12] final = 2/3 x A10 value + 1/3 x rank(s6fall), weights fixed before any run; a study s6fall did not score keeps its A10 value
import numpy as np, pandas as pd
try:
    if not S6F_OK:
        raise RuntimeError('s6fall stage did not complete')
    _base = pd.read_csv('/kaggle/working/_a10_submission.csv', dtype={'StudyInstanceUID': str})
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _sf = pd.read_csv('/kaggle/working/s6fall_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    _sf = _sf.reindex(_base['StudyInstanceUID'])[_labels]
    _ok = np.isfinite(_sf.values).all(axis=1)
    if _ok.sum() < 0.9 * len(_base):
        raise RuntimeError('s6fall scored only %d of %d studies' % (_ok.sum(), len(_base)))
    _out = _base.copy()
    for _c in _labels:
        _r = np.full(len(_base), np.nan)
        _r[_ok] = pd.Series(_sf[_c].values[_ok]).rank(method='average', pct=True).values
        _v = _base[_c].values.astype(np.float64)
        _out[_c] = np.where(_ok, (2.0 / 3.0) * _v + (1.0 / 3.0) * _r, _v)
    assert np.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert _out['StudyInstanceUID'].tolist() == _base['StudyInstanceUID'].tolist()
    _out.to_csv('/kaggle/working/submission.csv', index=False)
    _rho = {c: round(float(pd.Series(_base[c].values[_ok]).corr(pd.Series(_sf[c].values[_ok]), method='spearman')), 3) for c in _labels} if _ok.sum() >= 3 else {}
    print('A12_BLEND_DONE s6fall scored %d of %d; spearman(A10, s6fall) %s' % (_ok.sum(), len(_base), _rho), flush=True)
except BaseException as _e:
    print('A12_BLEND_SKIPPED (A10 submission kept):', repr(_e)[:500], flush=True)
'''
nb['cells'] += [cell('## A12: nartaa s6fall as a third equal leg (time-guarded, never fatal)', 'markdown'), cell(stage), cell(blend)]
json.dump(nb, open('rsna-knee-oai-957s6f.ipynb', 'w', encoding='utf8'), indent=1)
print('cells', len(nb['cells']))
