"""Write rsna-knee-c96-cache.py: the CPU kernel that caches nartaa's 96-slice native-384 study volumes for training.

    python build.py

The volumes come from nartaa's own `fastread` module (variant study_lut, byte-identical to their inference
`build_study`), embedded here as base64 so the script kernel is one file. Each study is stored as
c96/<uid>.npz: JPEG (q=90) of the centre 352x352 of every non-empty slice (training crops 320 + jitter),
byte offsets and the 96-slice mask.
"""
import base64, os
HERE = os.path.dirname(os.path.abspath(__file__))
b64 = base64.b64encode(open(os.path.join(HERE, 'fastread_nartaa.py'), 'rb').read()).decode()
SCRIPT = r'''# CPU kernel: cache nartaa's d96 (96 slices x 384 px) volumes for all training studies as JPEG slices.
import base64, glob, importlib.util, json, os, sys, time
from multiprocessing import get_context
import numpy as np, pandas as pd, cv2
FASTREAD_B64 = "%s"
open('/kaggle/working/_fastread.py', 'wb').write(base64.b64decode(FASTREAD_B64))
spec = importlib.util.spec_from_file_location('_fastread', '/kaggle/working/_fastread.py'); FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
IMG, KEEP, Q = 384, 352, 90
LIMIT = int(os.environ.get('LIMIT', '0'))
COMP = next(c for c in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection') if os.path.exists(c + '/train.csv'))
OUT = '/kaggle/working/c96'; os.makedirs(OUT, exist_ok=True)
ser = pd.read_csv(COMP + '/train_series.csv'); ser['StudyInstanceUID'] = ser.StudyInstanceUID.astype(str); ser['SeriesInstanceUID'] = ser.SeriesInstanceUID.astype(str)
SER = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}
ids = pd.read_csv(COMP + '/train.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
if LIMIT: ids = ids[:LIMIT]
TS = COMP + '/train_series'
B = None
def init():
    global B
    B = FR.make_builder('study_lut', IMG, SLOTS, hdr_threads=32, px_threads=16)
def one(sid):
    t = time.time()
    try:
        vol, mask = B(sid, SER, TS)
    except Exception as e:
        return sid, 0, 0, 'FAIL %%s: %%s' %% (type(e).__name__, str(e)[:100])
    lo = (IMG - KEEP) // 2
    bufs, offs = [], [0]
    for i in range(vol.shape[0]):
        if mask[i]:
            ok, enc = cv2.imencode('.jpg', np.ascontiguousarray(vol[i, lo:lo + KEEP, lo:lo + KEEP]), [cv2.IMWRITE_JPEG_QUALITY, Q])
            b = enc.tobytes() if ok else b''
        else:
            b = b''
        bufs.append(b); offs.append(offs[-1] + len(b))
    np.savez(f'{OUT}/{sid}.npz', jpg=np.frombuffer(b''.join(bufs), np.uint8), offs=np.array(offs, np.int64), mask=mask.astype(np.uint8))
    return sid, int(mask.sum()), offs[-1], '%%.1fs' %% (time.time() - t)
t0 = time.time(); nproc = max(4, (os.cpu_count() or 4)); rows = []; tot = 0
print('studies', len(ids), 'procs', nproc, flush=True)
with get_context('fork').Pool(nproc, initializer=init) as pool:
    for i, (sid, nm, nb, msg) in enumerate(pool.imap_unordered(one, ids, chunksize=2)):
        rows.append((sid, nm, nb, msg)); tot += nb
        if msg.startswith('FAIL') or (i + 1) %% 200 == 0 or i + 1 == len(ids):
            print(f'[{i+1}/{len(ids)}] {time.time()-t0:.0f}s total {tot/1e9:.2f} GB | {sid[:20]} slices {nm} {msg}', flush=True)
pd.DataFrame(rows, columns=['StudyInstanceUID', 'slices', 'bytes', 'msg']).to_csv('/kaggle/working/c96_index.csv', index=False)
os.remove('/kaggle/working/_fastread.py')
print('DONE', len(rows), 'studies', f'{tot/1e9:.2f} GB', f'{(time.time()-t0)/60:.1f} min', 'fails', sum(r[3].startswith('FAIL') for r in rows), flush=True)
''' % b64
open(os.path.join(HERE, 'rsna-knee-c96-cache.py'), 'w').write(SCRIPT)
print('wrote', len(SCRIPT))
