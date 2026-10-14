"""Build the gold-58 diagnostic for nartaa's s6fall (S6 ConvNeXt, all data, gold excluded, no OAI; 0.950 public alone).

    python build.py <rsna-knee-s6fall-solo.py> <oai_gold58_notebook.ipynb>

nartaa's script is embedded unchanged and loaded as a module with K1_KAGGLE_ROOT pointing at a fake root whose competition
folder is the gold-58 stand-in root (its own local-test hook). Only its prepare_inputs() and run_b() run: component A (the
0.949 kernel, weight 0 in the solo version) reads the real competition paths and is not needed. Writes gold_s6fall.csv.
"""
import base64, json, sys


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


script = open(sys.argv[1], 'rb').read()
g58 = json.load(open(sys.argv[2], encoding='utf8'))
root = ''.join(g58['cells'][1]['source'])
driver = '''
import base64, glob, hashlib, json, os, shutil, sys, time
from pathlib import Path
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
    print('linked', link, '->', target, flush=True)
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
print('env', mod._RECEIPT['env'], flush=True)
I = mod.prepare_inputs()
print('B probe', mod.probe_b_stack(I), flush=True)
t = time.time()
P, Z, info, failed = mod.run_b(I, 'B', I.test_ids, 'test', 4 * 3600)
print('B done seconds %%.0f failed %%d' %% (time.time() - t, len(failed)), flush=True)
probs = mod.OUT / 'B' / 'B_probs.csv'
import pandas as pd
df = pd.read_csv(probs, dtype={'StudyInstanceUID': str})
df[['StudyInstanceUID'] + mod.LAB].to_csv('/kaggle/working/gold_s6fall.csv', index=False)
json.dump({k: v for k, v in info.items() if k != 'runs'}, open('/kaggle/working/s6fall_info.json', 'w'), indent=1, default=str)
shutil.copyfile(mod.OUT / 'k1_log.txt', '/kaggle/working/s6fall_k1_log.txt')
print(df.head(), flush=True)
''' % base64.b64encode(script).decode()
nb = {'cells': [cell(__doc__, 'markdown'), cell(root), cell(driver)],
      'metadata': g58['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
json.dump(nb, open('rsna-knee-oai-gold58-s6fall.ipynb', 'w', encoding='utf8'), indent=1)
print('ok', len(script))
