"""VJ decode diagnostic (CPU, no GPU quota): why does nurikw3's infer.py read only ~1.8 of ~4.7 views per gold study?

    python build.py <oai_gold58_notebook.ipynb>

Calls the author's _load_source_slices directly (load_study_tensor swallows its exceptions) on every VIEWS5 candidate
series of the first 12 gold studies and prints the exception, the transfer syntax and the file count. No labels are read.
"""
import json, sys


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


g58 = json.load(open(sys.argv[1], encoding='utf8'))
root = ''.join(g58['cells'][1]['source'])
diag = r'''
import glob, sys, traceback, collections
from pathlib import Path
import pandas as pd, pydicom
c = [p for p in glob.glob('/kaggle/input/**/f0_best.pt', recursive=True) if 'vjepa21' in p]
DS = str(Path(c[0]).parent)
sys.path.append(DS)
import infer
print('pydicom', pydicom.__version__)
for m in ('pylibjpeg', 'gdcm', 'libjpeg', 'openjpeg', 'jpeg_ls', 'dicomsdl'):
    try:
        __import__(m); print('has', m)
    except Exception as e:
        print('no', m, type(e).__name__)
comp = Path(G)
ser = pd.read_csv(comp / 'test_series.csv', dtype={'StudyInstanceUID': str})
uids = pd.read_csv(comp / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()[:12]
errs, ts, ok = collections.Counter(), collections.Counter(), 0
for u in uids:
    g = ser[ser.StudyInstanceUID == u]
    for plane, fluid in infer.VIEWS5:
        for row in g[(g.Anatomical_Plane == plane) & (g.Fluid_Sensitive == fluid)].itertuples(index=False):
            d = comp / 'test_series' / u / str(row.SeriesInstanceUID)
            files = sorted(glob.glob(str(d / '*.dcm')))
            any_files = sorted(glob.glob(str(d / '*')))
            t = None
            if any_files:
                try:
                    t = str(pydicom.dcmread(any_files[0], stop_before_pixels=True, force=True).file_meta.TransferSyntaxUID)
                except Exception as e:
                    t = 'hdr-err ' + type(e).__name__
            ts[t] += 1
            try:
                s = infer._load_source_slices(d)
                r = 'None' if s is None else ('ok %s nonzero %s' % (s.shape, bool(s.any())))
                ok += s is not None and bool(s.any())
            except Exception as e:
                r = 'EXC ' + repr(e)[:160]
                errs[type(e).__name__ + ': ' + str(e)[:80]] += 1
            print(u[-8:], plane, fluid, 'dcm %d all %d' % (len(files), len(any_files)), t, '->', r, flush=True)
            break
print('series ok', ok, 'transfer syntaxes', dict(ts))
print('errors', dict(errs))
print('VJ_DIAG_DONE')
'''
nb = {'cells': [cell(__doc__, 'markdown'), cell(root), cell(diag)], 'metadata': g58['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
json.dump(nb, open('rsna-knee-vj-diag.ipynb', 'w', encoding='utf8'), indent=1)
meta = {"id": "tranbadat/rsna-knee-vj-diag", "title": "rsna-knee-vj-diag", "code_file": "rsna-knee-vj-diag.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "keywords": [], "dataset_sources": ["nurikw3/rsna-knee-vjepa21-checkpoints"],
        "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"}
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('ok')
