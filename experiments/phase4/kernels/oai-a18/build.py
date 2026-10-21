"""Build A18 = A16 + nurikw3's V-JEPA 2.1 5-fold leg (VJ) at 1/4 (docs/phase4_notes.md, 2026-10-21).

    python build.py <rsna-knee-oai-a16.ipynb> <fp16|fp32>

A16's cells are unchanged. Two cells are appended after Part B: the VJ stage (the author's infer.py imported unchanged,
one worker process per T4, 5-fold mean probability at the precision the gate picked; starts only with >= 75 min left,
stops 25 min before the 9 h limit) and the blend: final = 3/4 x rank(A16) + 1/4 x rank(VJ). Any failure, or VJ scoring
fewer than 90 % of the studies, keeps A16's file.
"""
import json
import sys

a16 = json.load(open(sys.argv[1], encoding='utf8'))
prec = sys.argv[2]
assert prec in ('fp16', 'fp32')
worker = r'''
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ds, uid_file, out, comp, deadline, prec = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
try:
    import einops                                   # the image's einops first; infer.py puts its own copy on sys.path
except ImportError:
    pass
sys.path.append(ds)
import infer                                         # unchanged author code
uids = [u for u in open(uid_file).read().split() if u]
ser = pd.read_csv(comp / 'test_series.csv', dtype={'StudyInstanceUID': str})
by = {str(u): g for u, g in ser.groupby('StudyInstanceUID', sort=False)}
ckpts = sorted(Path(ds).glob('f*_best.pt'))
assert len(ckpts) == 5, ckpts
models = [infer.load_model(c) for c in ckpts]
done, P, t0 = [], [], time.time()
for i, u in enumerate(uids):
    if time.time() > deadline:
        print('VJ worker deadline after %d studies' % i, flush=True)
        break
    try:
        gray, present = infer.load_study_tensor(comp, u, by.get(u))
        x = torch.from_numpy(gray).unsqueeze(0).cuda()
        p = torch.from_numpy(present).unsqueeze(0).cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=(prec == 'fp16')):
            z = np.stack([m(x, p).float().squeeze(0).cpu().numpy() for m in models])
        if np.isfinite(z).all():
            done.append(u)
            P.append((1 / (1 + np.exp(-z.astype(np.float64)))).mean(0))
    except Exception as e:
        print('VJ study failed', u, repr(e)[:200], flush=True)
    if (i + 1) % 100 == 0:
        print('VJ %d/%d %.0fs' % (i + 1, len(uids), time.time() - t0), flush=True)
np.savez(out, uids=np.array(done), p=np.array(P).reshape(-1, 12))
print('VJ_WORKER_DONE %d of %d in %.0fs' % (len(done), len(uids), time.time() - t0), flush=True)
'''
stage = '''# [A18] nurikw3's V-JEPA 2.1 5-fold (public weights, ViT-B video encoder) as a fourth-family leg; never fatal: A16's file stays
import glob, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
VJ_OK = False
_W = Path('/kaggle/working')
try:
    shutil.copyfile(_W / 'submission.csv', _W / '_a16_submission.csv')
    _left = 9 * 3600 - (time.time() - NB_T0)
    if _left < 75 * 60:
        raise RuntimeError('only %%.0f min left' %% (_left / 60))
    _deadline = time.time() + _left - 25 * 60
    _c = [p for p in glob.glob('/kaggle/input/**/f0_best.pt', recursive=True) if 'vjepa21' in p]
    assert len(_c) == 1, _c
    _ds = str(Path(_c[0]).parent)
    _comp = next(c for c in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection')
                 if os.path.exists(c + '/test.csv'))
    (_W / 'vj_worker.py').write_text(%r)
    _uids = pd.read_csv(_comp + '/test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    _procs, _t = [], time.time()
    for _g in range(2):
        (_W / ('vj_uids_%%d.txt' %% _g)).write_text('\\n'.join(_uids[_g::2]))
        _log = open(_W / ('vj_worker_%%d.log' %% _g), 'w')
        _procs.append(subprocess.Popen([sys.executable, str(_W / 'vj_worker.py'), _ds, str(_W / ('vj_uids_%%d.txt' %% _g)),
                                        str(_W / ('vj_%%d.npz' %% _g)), _comp, str(_deadline), %r],
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(_g)), stdout=_log, stderr=subprocess.STDOUT))
    for _p in _procs:
        try:
            _p.wait(timeout=max(60, _deadline - time.time() + 300))
        except subprocess.TimeoutExpired:
            _p.kill()
    for _g in range(2):
        print('--- VJ worker', _g, 'rc', _procs[_g].returncode, open(_W / ('vj_worker_%%d.log' %% _g)).read()[-1500:], flush=True)
    _parts = [np.load(_W / ('vj_%%d.npz' %% _g)) for _g in range(2) if (_W / ('vj_%%d.npz' %% _g)).is_file()]
    assert _parts, 'no VJ worker output'
    _LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
            "Baker's", 'Contusion', 'Fracture']
    _vj = pd.DataFrame(np.concatenate([q['p'] for q in _parts]), columns=_LAB)
    _vj.insert(0, 'StudyInstanceUID', np.concatenate([q['uids'] for q in _parts]))
    _vj.to_csv(_W / 'vj_sub.csv', index=False)
    VJ_OK = True
    print('VJ_STAGE_DONE seconds %%.0f scored %%d of %%d' %% (time.time() - _t, len(_vj), len(_uids)), flush=True)
except BaseException as _e:
    print('VJ_SKIPPED', repr(_e)[:500], flush=True)
finally:
    if not (_W / 'submission.csv').is_file() and (_W / '_a16_submission.csv').is_file():
        shutil.copyfile(_W / '_a16_submission.csv', _W / 'submission.csv')
        print('VJ restored the A16 submission', flush=True)
''' % (worker, prec)
blend = '''# [A18] final = 3/4 x rank(A16) + 1/4 x rank(VJ), weights fixed before any run; a study VJ did not score keeps its A16 rank
import numpy as np, pandas as pd
try:
    if not VJ_OK:
        raise RuntimeError('VJ stage did not complete')
    _base = pd.read_csv('/kaggle/working/_a16_submission.csv', dtype={'StudyInstanceUID': str})
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _vj = pd.read_csv('/kaggle/working/vj_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    _vj = _vj[~_vj.index.duplicated()].reindex(_base['StudyInstanceUID'])[_labels]
    _ok = np.isfinite(_vj.values).all(axis=1)
    if _ok.sum() < 0.9 * len(_base):
        raise RuntimeError('VJ scored only %d of %d studies' % (_ok.sum(), len(_base)))
    _out = _base.copy()
    for _c in _labels:
        _rb = pd.Series(_base[_c].values.astype(np.float64)).rank(method='average', pct=True).values
        _r = np.full(len(_base), np.nan)
        _r[_ok] = pd.Series(_vj[_c].values[_ok]).rank(method='average', pct=True).values
        _out[_c] = np.where(_ok, 0.75 * _rb + 0.25 * _r, _rb)
    assert np.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert _out['StudyInstanceUID'].tolist() == _base['StudyInstanceUID'].tolist()
    _out.to_csv('/kaggle/working/submission.csv', index=False)
    _rho = {c: round(float(pd.Series(_base[c].values[_ok]).corr(pd.Series(_vj[c].values[_ok]), method='spearman')), 3) for c in _labels} if _ok.sum() >= 3 else {}
    print('A18_BLEND_DONE VJ scored %d of %d; spearman(A16, VJ) %s' % (_ok.sum(), len(_base), _rho), flush=True)
except BaseException as _e:
    print('A18_BLEND_SKIPPED (A16 submission kept):', repr(_e)[:500], flush=True)
'''


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


assert 'NB_T0' in ''.join(a16['cells'][4]['source']), 'A16 stage-1 cell no longer sets NB_T0'
a16['cells'][1]['source'] = ("## A18: A16 (heliosli's public 0.9586 recipe) + nurikw3's V-JEPA 2.1 5-fold leg at 1/4\n\n"
                             "A16's cells are unchanged; the last two cells run the VJ stage (%s) and the blend: "
                             "final = 3/4 x rank(A16) + 1/4 x rank(VJ), time-guarded, never fatal (A16's file stays on any failure)." % prec)
a16['cells'] += [cell("## A18: V-JEPA 2.1 leg ([nurikw3/rsna-knee-vjepa21-checkpoints](https://www.kaggle.com/datasets/nurikw3/rsna-knee-vjepa21-checkpoints), "
                      "V-JEPA 2 code Apache-2.0 / MIT per its LICENSE files)", 'markdown'), cell(stage), cell(blend)]
json.dump(a16, open('rsna-knee-oai-a18.ipynb', 'w', encoding='utf8'), indent=1)
meta = json.load(open('../oai-a16/kernel-metadata.json'))
meta.update(id='tranbadat/rsna-knee-oai-a18', title='rsna-knee-oai-a18', code_file='rsna-knee-oai-a18.ipynb')
meta['dataset_sources'] = meta['dataset_sources'] + ['nurikw3/rsna-knee-vjepa21-checkpoints']
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('A18 cells', len(a16['cells']), prec)
