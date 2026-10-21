"""VJ gold-58 kernel: nurikw3's V-JEPA 2.1 5-fold checkpoints on the 58 gold studies (rules: docs/phase4_notes.md, VJ).

    python build.py <oai_gold58_notebook.ipynb>

The author's infer.py is imported unchanged (preprocessing, load_model); one worker process per T4 runs its half of the
studies with every fold, first under fp16 autocast (the deployment candidate; T4 has no bf16) and then in fp32 (the
precision check). Per-fold logits are kept so the gate can read the per-fold leak guard. No AUC is computed here.
Outputs: gold_vj.csv (fp16 5-fold mean prob), gold_vj_fp32.csv, gold_vj_f<k>.csv (fp16 per fold), vj_gold_info.json.
"""
import json, sys


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


g58 = json.load(open(sys.argv[1], encoding='utf8'))
root = ''.join(g58['cells'][1]['source'])
worker = r'''
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
ds, uid_file, out, comp = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
try:
    import einops                                   # the image's einops first; infer.py puts its own copy on sys.path
    print('einops', einops.__file__, flush=True)
except ImportError:
    print('einops: image has none, using the dataset copy', flush=True)
sys.path.append(ds)
import infer                                         # unchanged author code (adds its own dir and vjepa2/ to sys.path)
uids = open(uid_file).read().split()
ser = pd.read_csv(comp / 'test_series.csv', dtype={'StudyInstanceUID': str})
by = {str(u): g for u, g in ser.groupby('StudyInstanceUID', sort=False)}
ckpts = sorted(Path(ds).glob('f*_best.pt'))
assert len(ckpts) == 5, ckpts
t = time.time()
models = [infer.load_model(c) for c in ckpts]
t_load = time.time() - t
print('loaded', [c.name for c in ckpts], 'in %.1fs on' % t_load, torch.cuda.get_device_name(0), flush=True)
Z16, Z32, per = [], [], []
for i, u in enumerate(uids):
    t0 = time.time()
    gray, present = infer.load_study_tensor(comp, u, by.get(u))
    t1 = time.time()
    x = torch.from_numpy(gray).unsqueeze(0).cuda()
    p = torch.from_numpy(present).unsqueeze(0).cuda()
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.float16):
            z16 = [m(x, p).float().squeeze(0).cpu().numpy() for m in models]
        torch.cuda.synchronize()
        t2 = time.time()
        z32 = [m(x, p).float().squeeze(0).cpu().numpy() for m in models]
        torch.cuda.synchronize()
        t3 = time.time()
    Z16.append(z16)
    Z32.append(z32)
    per.append((t1 - t0, t2 - t1, t3 - t2, int(present.sum())))
    if i < 3 or (i + 1) % 10 == 0:
        print('%d/%d decode %.2fs fp16 %.2fs fp32 %.2fs views %d' % ((i + 1, len(uids)) + per[-1]), flush=True)
np.savez(out, uids=np.array(uids), z16=np.array(Z16), z32=np.array(Z32), per=np.array(per), t_load=t_load)
print('WORKER_DONE', len(uids), flush=True)
'''
driver = '''
import glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
c = [p for p in glob.glob('/kaggle/input/**/f0_best.pt', recursive=True) if 'vjepa21' in p]
assert len(c) == 1, c
DS = str(Path(c[0]).parent)
print('VJ dataset', DS, sorted(os.listdir(DS)), flush=True)
W = Path('/kaggle/working')
(W / 'vj_worker.py').write_text(%r)
uids = pd.read_csv(G + '/test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
procs, t = [], time.time()
for g in range(2):
    (W / ('vj_uids_%%d.txt' %% g)).write_text('\\n'.join(uids[g::2]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    log = open(W / ('vj_worker_%%d.log' %% g), 'w')
    procs.append((subprocess.Popen([sys.executable, str(W / 'vj_worker.py'), DS, str(W / ('vj_uids_%%d.txt' %% g)),
                                    str(W / ('vj_%%d.npz' %% g)), G], env=env, stdout=log, stderr=subprocess.STDOUT), log))
rc = [p.wait() for p, _ in procs]
wall = time.time() - t
for g in range(2):
    print('--- worker', g, 'rc', rc[g]); print(open(W / ('vj_worker_%%d.log' %% g)).read()[-3000:])
assert rc == [0, 0], rc
parts = [np.load(W / ('vj_%%d.npz' %% g)) for g in range(2)]
U = np.concatenate([q['uids'] for q in parts])
Z16 = np.concatenate([q['z16'] for q in parts]).astype(np.float64)
Z32 = np.concatenate([q['z32'] for q in parts]).astype(np.float64)
PER = np.concatenate([q['per'] for q in parts])
sig = lambda z: 1 / (1 + np.exp(-z))
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
def save(p, name):
    pd.DataFrame(p, columns=LAB).assign(StudyInstanceUID=U)[['StudyInstanceUID'] + LAB].to_csv(W / name, index=False)
p16, p32 = sig(Z16).mean(1), sig(Z32).mean(1)
save(p16, 'gold_vj.csv')
save(p32, 'gold_vj_fp32.csv')
for k in range(Z16.shape[1]):
    save(sig(Z16[:, k]), 'gold_vj_f%%d.csv' %% k)
info = {'studies': int(len(U)), 'wall_s': round(wall, 1), 'load_s': [float(q['t_load']) for q in parts],
        'decode_s_mean': float(PER[:, 0].mean()), 'fp16_s_mean': float(PER[1:, 1].mean()), 'fp32_s_mean': float(PER[1:, 2].mean()),
        'views_present_mean': float(PER[:, 3].mean()), 'nan_fp16': int(np.isnan(Z16).sum()), 'nan_fp32': int(np.isnan(Z32).sum()),
        'max_abs_dp_fp16_vs_fp32': float(np.nanmax(np.abs(p16 - p32))),
        'max_abs_dp_fold_fp16_vs_fp32': float(np.nanmax(np.abs(sig(Z16) - sig(Z32))))}
json.dump(info, open(W / 'vj_gold_info.json', 'w'), indent=1)
print(json.dumps(info, indent=1))
print('GOLD_VJ_DONE', flush=True)
''' % worker
nb = {'cells': [cell(__doc__, 'markdown'), cell(root), cell(driver)],
      'metadata': g58['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
name = 'rsna-knee-vj-gold58'
json.dump(nb, open(name + '.ipynb', 'w', encoding='utf8'), indent=1)
meta = {"id": "tranbadat/" + name, "title": name, "code_file": name + ".ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["nurikw3/rsna-knee-vjepa21-checkpoints"],
        "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4"}
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('ok', name)
