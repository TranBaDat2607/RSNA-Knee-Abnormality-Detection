"""V1 gold-58 kernel: s6fall's own inference path (compact_infer via run_b, 2 GPUs) on our ofl_l (OrthoFoundation-L) models.

    python build.py <rsna-knee-s6fall-solo.py> <oai_gold58_notebook.ipynb> <dev|gate> <strides, e.g. 1,3> <train slug> [<train slug> ...]

As rsna-knee-s6t-gold58: nartaa's s6fall script is loaded as a module, with K1_KAGGLE_ROOT pointing at a fake root whose
competition folder is the gold-58 stand-in. Every V1 seed found in the attached training kernels goes into ONE models folder.
That is the deployment form: compact_infer runs one shared frozen trunk and one tail per seed, and averages the logits.
b_cmd gets --backbone ofl_l --init <OFL-L> --freeze-blocks 16 --eval-stride <s>. collect_part keeps the per-seed logits,
then hands s6fall's strict checks a one-row mean (= predictions.csv, sigmoid of the mean logit).

dev:  timing only (v1_gold_info.json); no probabilities are written, so the dev model's score is never produced.
gate: gold_v1.csv (all seeds, the stride given) + gold_v1_s<seed>.csv + timing.  Rules: docs/phase4_notes.md (V1).
"""
import ast, base64, json, sys
from pathlib import Path


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


script = open(sys.argv[1], 'rb').read()
g58 = json.load(open(sys.argv[2], encoding='utf8'))
mode, strides, slugs = sys.argv[3], [int(s) for s in sys.argv[4].split(',')], sys.argv[5:]
assert mode in ('dev', 'gate') and slugs and (mode == 'dev' or len(strides) == 1)
root = ''.join(g58['cells'][1]['source'])
_v1 = (Path(__file__).resolve().parents[1] / 'v1-train' / 'v1_train.py').read_text()
_fn = next(n for n in ast.parse(_v1).body if isinstance(n, ast.FunctionDef) and n.name == 'convert_state')
convert_src = ast.get_source_segment(_v1, _fn)          # the exact conversion the training run used
driver = '''
import base64, glob, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
SRC = base64.b64decode(%r)
MODE, STRIDES, SLUGS = %r, %r, %r
OFL_SHA = '385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9'
CONVERT = %r
print('s6fall script sha256', hashlib.sha256(SRC).hexdigest(), flush=True)
FAKE = Path('/kaggle/gold_k')
(FAKE / 'input').mkdir(parents=True, exist_ok=True)
(FAKE / 'working').mkdir(parents=True, exist_ok=True)


def real(slug, marker):
    c = [p for p in glob.glob('/kaggle/input/**/' + marker, recursive=True) if '/' + slug + '/' in p]
    assert c, (slug, marker)
    return Path(c[0]).parent


for slug, target in (('rsna-knee-abnormality-detection', Path('/kaggle/working/gold_root')),
                     ('rsna-knee-s6fall-weights', real('rsna-knee-s6fall-weights', 'checkpoint-manifest.json')),
                     ('rsna-orthofoundation-fold0-epoch7-20261004', real('rsna-orthofoundation-fold0-epoch7-20261004', 'anatomy.pt'))):
    link = FAKE / 'input' / slug
    if not link.exists():
        os.symlink(target, link)
os.environ['K1_KAGGLE_ROOT'] = str(FAKE)
os.environ['K1_TMP'] = '/tmp/k1gold'
mod = type(sys)('s6fall_mod')
mod.__dict__['__file__'] = '/kaggle/working/s6fall_solo.py'
Path('/kaggle/working/s6fall_solo.py').write_bytes(SRC)
exec(compile(SRC, '/kaggle/working/s6fall_solo.py', 'exec'), mod.__dict__)
mod.OUT.mkdir(parents=True, exist_ok=True)
mod.TMP.mkdir(parents=True, exist_ok=True)
mod._LOG = open(mod.OUT / 'k1_log.txt', 'a')
mod._RECEIPT['env'] = mod.probe_env()
I = mod.prepare_inputs()

# the public OFL-L encoder the V1 tails are rebuilt on: official checkpoint (sha256 checked), converted as in training
import torch
exec(CONVERT)
_src = [p for p in glob.glob('/kaggle/input/**/OrthoFoudation-L.pth', recursive=True) if '/orthofoundation-l-dinov3/' in p]
assert _src, 'OrthoFoundation-L checkpoint not attached'
h = hashlib.sha256()
with open(_src[0], 'rb') as f:
    for b in iter(lambda: f.read(1 << 24), b''):
        h.update(b)
assert h.hexdigest() == OFL_SHA, h.hexdigest()
OFL = Path('/tmp/ofl_l_timm.pt')
torch.save(convert_state(torch.load(_src[0], map_location='cpu', weights_only=True)), OFL)
print('OFL-L encoder converted', _src[0], flush=True)

ckpts = {Path(p).name: Path(p) for p in sorted(glob.glob('/kaggle/input/**/v1_s*/models/E_prior_s*_f0.pt', recursive=True))
         if any('/' + s + '/' in p for s in SLUGS)}
print('V1 models', {k: str(v) for k, v in ckpts.items()}, flush=True)
assert ckpts, 'no V1 model attached'
shutil.rmtree(I.models)
I.models.mkdir(parents=True)
for n, p in ckpts.items():
    os.symlink(p, I.models / n)
first = sorted(ckpts)[0]
mod.B_MODEL = dict(mod.B_MODEL, file=first, source_file=first, sha256=hashlib.sha256(ckpts[first].read_bytes()).hexdigest())

STRIDE, V1_REC, V1_Z = [STRIDES[0]], {}, {}
_bcmd0, _collect0 = mod.b_cmd, mod.collect_part


def _bcmd_v1(I, split, assignment, out_dir, tmp_dir, deadline):
    c = _bcmd0(I, split, assignment, out_dir, tmp_dir, deadline)
    c[c.index('--backbone') + 1] = 'ofl_l'
    c[c.index('--eval-stride') + 1] = str(STRIDE[0])
    return c + ['--init', str(OFL), '--freeze-blocks', '16']


def _collect_v1(pdir, part, I):
    rp, lp = pdir / 'receipt.json', pdir / 'logits.json'
    rec = json.loads(rp.read_text())
    assert rec.get('backbone') == 'ofl_l' and rec.get('models') == sorted(ckpts) and rec.get('spatial_models') == 0, (
        rec.get('backbone'), rec.get('models'), rec.get('spatial_models'))
    V1_REC[pdir.name] = dict(rec)
    rec['models'], rec['spatial_models'] = [mod.B_MODEL['file']], 1        # s6fall's single-model checks, on the seed mean
    rp.write_text(json.dumps(rec))
    L = json.loads(lp.read_text())
    V1_Z.update({u: np.asarray(a, np.float64) for u, a in L.items()})
    lp.write_text(json.dumps({u: [np.asarray(a, np.float64).mean(0).tolist()] for u, a in L.items()}))
    return _collect0(pdir, part, I)


mod.b_cmd, mod.collect_part = _bcmd_v1, _collect_v1
info_all = {}
for s in STRIDES:
    STRIDE[0] = s
    V1_REC.clear()
    V1_Z.clear()
    t = time.time()
    P, Z, info, failed = mod.run_b(I, 'B', I.test_ids, 'test', 3 * 3600)
    sec = time.time() - t
    recs = list(V1_REC.values())
    info_all[s] = {'seconds': round(sec, 1), 'studies': len(I.test_ids), 'failed': len(failed), 'models': sorted(ckpts),
                   'shared_trunk': [r.get('shared_trunk') for r in recs],
                   'parts': [{'studies': r.get('studies'), 'seconds': r.get('seconds'), 'stats': r.get('stats')} for r in recs]}
    print('V1 stride %%d: %%.0fs for %%d studies, failed %%d, parts %%s' %% (s, sec, len(I.test_ids), len(failed),
          [(p['studies'], p['seconds']) for p in info_all[s]['parts']]), flush=True)
    if MODE == 'gate':
        df = pd.read_csv(mod.OUT / 'B' / 'B_probs.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[mod.LAB]
        df.reset_index().to_csv('/kaggle/working/gold_v1.csv', index=False)
        for j, n in enumerate(sorted(ckpts)):
            seed = n.split('_s')[1].split('_')[0]
            zz = {u: a[j] for u, a in V1_Z.items()}
            pd.DataFrame(1 / (1 + np.exp(-np.array(list(zz.values())))), index=list(zz), columns=mod.LAB).rename_axis(
                'StudyInstanceUID').reset_index().to_csv('/kaggle/working/gold_v1_s%%s.csv' %% seed, index=False)
    shutil.rmtree(mod.OUT / 'B', ignore_errors=True)
json.dump(info_all, open('/kaggle/working/v1_gold_info.json', 'w'), indent=1, default=str)
print('GOLD_V1_DONE', MODE, STRIDES, flush=True)
''' % (base64.b64encode(script).decode(), mode, strides, slugs, convert_src)
nb = {'cells': [cell(__doc__, 'markdown'), cell(root), cell(driver)],
      'metadata': g58['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
name = 'rsna-knee-v1-gold58' if mode == 'gate' else 'rsna-knee-v1-gold58-dev'
json.dump(nb, open(name + '.ipynb', 'w', encoding='utf8'), indent=1)
meta = {"id": "tranbadat/" + name, "title": name, "code_file": name + ".ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["nartaa/rsna-knee-s6fall-weights", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
                            "pritom2357/orthofoundation-l-dinov3"],
        "kernel_sources": ["tranbadat/" + s for s in slugs], "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4"}
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('ok', name, mode, strides, slugs, len(script))
