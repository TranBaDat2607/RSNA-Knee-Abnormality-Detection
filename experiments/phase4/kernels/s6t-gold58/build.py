"""Gold-58 gate run for S6T: s6fall's own inference path (compact_infer, eval stride 1, fp16) on our S6T checkpoints.

    python build.py <rsna-knee-s6fall-solo.py> <oai_gold58_notebook.ipynb> <train_kernel_slug>

As rsna-knee-oai-gold58-s6fall: nartaa's script is loaded as a module with K1_KAGGLE_ROOT pointing at a fake root whose
competition folder is the gold-58 stand-in. prepare_inputs() checks s6fall's own files as before; then, per S6T seed, the
models folder is swapped for that seed's checkpoint and run_b() scores the 58 studies. Writes gold_s6t_s<seed>.csv
(probabilities) and gold_s6t.csv (sigmoid of the mean logit over the seeds = compact_infer's group mean).
"""
import base64, json, sys


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


script = open(sys.argv[1], 'rb').read()
g58 = json.load(open(sys.argv[2], encoding='utf8'))
train_slug = sys.argv[3]
root = ''.join(g58['cells'][1]['source'])
driver = '''
import base64, glob, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
SRC = base64.b64decode(%r)
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
ckpts = sorted(p for p in glob.glob('/kaggle/input/**/models/E_prior_s*_f0.pt', recursive=True) if %r in p)
print('S6T checkpoints', ckpts, flush=True)
assert ckpts, 'no S6T checkpoint attached'
probs, logits, info_all = {}, {}, {}
for ck in ckpts:
    seed = Path(ck).name.split('_s')[1].split('_')[0]
    shutil.rmtree(I.models)
    I.models.mkdir(parents=True)
    os.symlink(ck, I.models / Path(ck).name)
    mod.B_MODEL = dict(mod.B_MODEL, file=Path(ck).name, source_file=Path(ck).name,
                       sha256=hashlib.sha256(Path(ck).read_bytes()).hexdigest())
    t = time.time()
    P, Z, info, failed = mod.run_b(I, 'B', I.test_ids, 'test', 3 * 3600)
    print('S6T seed %%s: %%.0fs, failed %%d' %% (seed, time.time() - t, len(failed)), flush=True)
    df = pd.read_csv(mod.OUT / 'B' / 'B_probs.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[mod.LAB]
    df.reset_index().to_csv('/kaggle/working/gold_s6t_s%%s.csv' %% seed, index=False)
    probs[seed] = df
    info_all[seed] = {k: v for k, v in info.items() if k != 'runs'}
    shutil.rmtree(mod.OUT / 'B', ignore_errors=True)
ids = probs[sorted(probs)[0]].index
z = np.mean([np.log(np.clip(p.loc[ids].values, 1e-7, 1 - 1e-7) / (1 - np.clip(p.loc[ids].values, 1e-7, 1 - 1e-7))) for p in probs.values()], 0)
pd.DataFrame(1 / (1 + np.exp(-z)), index=ids, columns=mod.LAB).reset_index().to_csv('/kaggle/working/gold_s6t.csv', index=False)
json.dump(info_all, open('/kaggle/working/s6t_gold_info.json', 'w'), indent=1, default=str)
print('GOLD_S6T_DONE seeds', sorted(probs), flush=True)
''' % (base64.b64encode(script).decode(), train_slug)
nb = {'cells': [cell(__doc__, 'markdown'), cell(root), cell(driver)],
      'metadata': g58['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
json.dump(nb, open('rsna-knee-s6t-gold58.ipynb', 'w', encoding='utf8'), indent=1)
meta = {"id": "tranbadat/rsna-knee-s6t-gold58", "title": "rsna-knee-s6t-gold58", "code_file": "rsna-knee-s6t-gold58.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["nartaa/rsna-knee-s6fall-weights", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
        "kernel_sources": ["tranbadat/" + train_slug], "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4"}
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('ok', len(script))
