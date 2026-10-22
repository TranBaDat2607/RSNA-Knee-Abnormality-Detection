"""Build A19 = A16 with extra Raptor-384 checkpoints averaged into its 384 slot (docs/phase4_notes.md, 2026-10-21 15:35 UTC).

    python build.py <rsna-knee-oai-a16.ipynb> <member tags, comma-separated, from a19_gate.py>

A16's cells are unchanged. After stage 2 (R6 flip) one cell per member runs A16's own Raptor-384 script with that
member's checkpoint pins (the same patch as the gold kernel), as its own process: it starts only with >= 135 min left
before 9 h and must finish 60 min before it. A merge cell then rewrites raptor_raw.csv / raptor_sub.csv as the
equal-weight rank mean of nartaa 384 and the members that scored >= 90 % of the studies (each ranked over its own scored
studies). Stage 3, Part B and everything after are A16's. Any failure keeps A16's 384 files.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'a19-gold58'))
from build import MEMBERS, patch, cell  # noqa: E402  (the gold kernel's patch: identical member scripts)

a16 = json.load(open(sys.argv[1], encoding='utf8'))
tags = [t for t in sys.argv[2].split(',') if t]
spec = {t: (n, s) for t, n, s in MEMBERS}
assert tags and all(t in spec and t != 'n384' for t in tags), tags
raptor = ''.join(a16['cells'][3]['source'])
assert raptor.startswith('%%writefile /tmp/raptor_run.py\n')
raptor = raptor.split('\n', 1)[1]
assert 'probe_dual.py' in ''.join(a16['cells'][8]['source']) and 'Stage 3' in ''.join(a16['cells'][9]['source'])

stage = '''# [A19] extra Raptor-384 member %(tag)s (%(name)s): A16's Raptor-384 script with only the checkpoint pins swapped,
# its own process; starts only with >= 135 min left, must end 60 min before 9 h. Never fatal.
import os, sys, time, subprocess
A19_MEMBERS = globals().get('A19_MEMBERS', [])
_left = 9 * 3600 - (time.time() - NB_T0)
if _left < 135 * 60:
    print('A19_MEMBER_SKIPPED %(tag)s: only %%.0f min left' %% (_left / 60), flush=True)
else:
    open('/tmp/a19_%(tag)s_run.py', 'w').write(%(src)r)
    _t = time.time()
    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/a19_%(tag)s_run.py'], timeout=_left - 60 * 60).returncode
    except subprocess.TimeoutExpired:
        _rc = 'timeout'
    _ok = _rc == 0 and all(os.path.isfile('/kaggle/working/%(tag)s_' + f) for f in ('raw.csv', 'sub.csv'))
    if _ok:
        A19_MEMBERS.append('%(tag)s')
    print('A19_MEMBER %(tag)s rc %%s ok %%s seconds %%.0f' %% (_rc, _ok, time.time() - _t), flush=True)
'''
merge = '''# [A19] Raptor-384 slot = equal-weight rank mean of nartaa 384 and the members that finished (>= 90 % coverage each),
# each ranked over its own scored studies; a study no model scored stays 0.5 (stage 3 treats it as a 384 failure).
import shutil
import numpy as np, pandas as pd
_W = '/kaggle/working/'
shutil.copyfile(_W + 'raptor_raw.csv', _W + '_a16_raptor_raw.csv')
shutil.copyfile(_W + 'raptor_sub.csv', _W + '_a16_raptor_sub.csv')
try:
    _base = pd.read_csv(_W + 'raptor_raw.csv', dtype={'StudyInstanceUID': str})
    _ids = _base['StudyInstanceUID'].tolist()
    _lab = [c for c in _base.columns if c != 'StudyInstanceUID']
    def _rank_on(df):
        v = df.set_index('StudyInstanceUID').reindex(_ids)[_lab].values.astype(np.float64)
        ok = np.isfinite(v).all(1) & ~(v == 0.5).all(1)
        r = np.full(v.shape, np.nan)
        if ok.any():
            r[ok] = pd.DataFrame(v[ok]).rank(method='average', pct=True).values
        return r, ok
    _parts = [_rank_on(_base)]
    _used = []
    for _tag in globals().get('A19_MEMBERS', []):
        _r, _ok = _rank_on(pd.read_csv(_W + _tag + '_raw.csv', dtype={'StudyInstanceUID': str}))
        print('A19 member', _tag, 'coverage %.3f' % _ok.mean(), flush=True)
        if _ok.mean() >= 0.9:
            _parts.append((_r, _ok))
            _used.append(_tag)
    if not _used:
        raise RuntimeError('no member finished with >= 90 % coverage')
    with np.errstate(invalid='ignore'):
        _m = np.nanmean(np.stack([p for p, _ in _parts]), axis=0)
    _none = ~np.any(np.stack([o for _, o in _parts]), axis=0)
    _m[_none] = 0.5
    assert np.isfinite(_m).all()
    _rho = {c: round(float(pd.Series(_m[:, j]).corr(pd.Series(_parts[0][0][:, j]), method='spearman')), 4)
            for j, c in enumerate(_lab)} if len(_ids) >= 3 else {}
    _out = pd.DataFrame(_m, columns=_lab)
    _out.insert(0, 'StudyInstanceUID', _ids)
    _out.to_csv(_W + 'raptor_raw.csv', index=False)
    _sub = _out.copy()
    _sub[_lab] = _sub[_lab].rank(method='average', pct=True)
    _sub.to_csv(_W + 'raptor_sub.csv', index=False)
    print('A19_SLOT_DONE members', _used, 'no-model studies', int(_none.sum()), 'spearman(slot, n384)', _rho, flush=True)
except BaseException as _e:
    shutil.copyfile(_W + '_a16_raptor_raw.csv', _W + 'raptor_raw.csv')
    shutil.copyfile(_W + '_a16_raptor_sub.csv', _W + 'raptor_sub.csv')
    print('A19_SLOT_SKIPPED (A16 384 files kept):', repr(_e)[:400], flush=True)
'''
new = [cell('## A19: extra Raptor-384 members (%s), averaged into the 384 slot before stage 3' % ', '.join(tags), 'markdown')]
for t in tags:
    name, sha = spec[t]
    new.append(cell(stage % {'tag': t, 'name': name, 'src': patch(raptor, t, name, sha)}))
new.append(cell(merge))
a16['cells'] = a16['cells'][:9] + new + a16['cells'][9:]
a16['cells'][1]['source'] = ("## A19: A16 (heliosli's public 0.9586 recipe) with %d more Raptor-384 checkpoint(s) in the 384 slot\n\n"
                             "A16's cells are unchanged. After stage 2, member stages (%s) run A16's own Raptor-384 script with only "
                             "the checkpoint swapped; the 384 slot becomes the equal-weight rank mean. Time-guarded, never fatal "
                             "(A16's 384 files stay on any failure)." % (len(tags), ', '.join(tags)))
json.dump(a16, open('rsna-knee-oai-a19.ipynb', 'w', encoding='utf8'), indent=1)
meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'oai-a16', 'kernel-metadata.json')))
src_ds = {'fullb': 'ephemeralbyte/rsna-knee-fullb-w', 'cohort': 'manatoshoji03/rsna-knee-coat-extra2',
          'bs14': 'manatoshoji03/rsna-knee-coat-extra'}
meta.update(id='tranbadat/rsna-knee-oai-a19', title='rsna-knee-oai-a19', code_file='rsna-knee-oai-a19.ipynb')
meta['dataset_sources'] = meta['dataset_sources'] + [src_ds[t] for t in tags]
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('A19 cells', len(a16['cells']), 'members', tags)
