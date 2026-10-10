"""Build the OAI-track blend notebook from the public notebooks it stacks.

    python build.py <goodpjw_nb.ipynb> <nartaa_eff_nb.ipynb> [--w-eff 0.25] [--ours-w 0.10]

Parts A+B: goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-950 unchanged (nartaa 0.949 CoAtNet + 2.5D
ConvNeXt reader, per-finding weights; public 0.950).
Part A2 (if --w-eff > 0): nartaa's 0.945 efficiency checkpoint (224 crop) run as a subprocess and rank-blended into
Part A's file before Part B reads it.
Part C (if --ours-w > 0): our R5t ConvNeXt-nano leg via rsna_knee.mil.ours.
Every added part keeps the previous submission.csv on any failure.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def eff_cells(eff_nb, w_eff):
    cells = json.load(open(eff_nb, encoding="utf8"))["cells"]
    check, source = "".join(cells[3]["source"]), "".join(cells[4]["source"])
    assert 'out = "/kaggle/working/submission.csv"' in source
    source = source.replace('out = "/kaggle/working/submission.csv"', 'out = "/kaggle/working/eff_sub.csv"')
    script = check + "\n\n" + source
    write = code("# ---- Part A2: nartaa's 0.945 efficiency checkpoint (224 crop), written to eff_sub.csv ----\n"
                 "open('/tmp/eff_run.py', 'w').write(" + repr(script) + ")\n")
    run = code(r'''# Part A2 run + rank blend into Part A's submission.csv (Part B then reads the blended file).
import gc, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
W_EFF = %r
try:
    import torch; gc.collect(); torch.cuda.empty_cache()
except Exception as _e:
    print('cuda cleanup skipped', _e)
shutil.copy('/kaggle/working/submission.csv', '/kaggle/working/_a_submission.csv')
_t0 = time.time()
try:
    _rc = subprocess.run([sys.executable, '-u', '/tmp/eff_run.py'], timeout=3 * 3600).returncode
    assert _rc == 0, 'eff leg exit code %%d' %% _rc
    _a = pd.read_csv('/kaggle/working/_a_submission.csv', dtype={'StudyInstanceUID': str})
    _e = pd.read_csv('/kaggle/working/eff_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[_a.StudyInstanceUID].reset_index()
    _lab = [c for c in _a.columns if c != 'StudyInstanceUID']
    assert _e[_lab].notna().all().all() and np.isfinite(_e[_lab].values).all(), 'eff leg NaN'
    _r = lambda d: d[_lab].rank(method='average', pct=True)
    _out = _a.copy()
    _out[_lab] = ((1 - W_EFF) * _r(_a) + W_EFF * _r(_e)).rank(method='average', pct=True)
    _out.to_csv('/kaggle/working/submission.csv', index=False)
    print('Part A2: eff leg blended at', W_EFF, 'in %%.0fs' %% (time.time() - _t0),
          '| mean rank corr acc-eff', round(float(_r(_a).corrwith(_r(_e)).mean()), 4) if len(_a) > 2 else 'n<3', flush=True)
except Exception as _x:
    shutil.copy('/kaggle/working/_a_submission.csv', '/kaggle/working/submission.csv')
    print('Part A2 FAILED, Part A submission kept:', repr(_x), flush=True)
''' % w_eff)
    return [write, run]


def ours_cell(ours_w):
    return code(r'''# ---- Part C: our leg (R5t ConvNeXt-nano 336), rank blend at a fixed weight; any failure keeps the previous file ----
import gc, os, shutil, subprocess, sys
try:
    import torch; gc.collect(); torch.cuda.empty_cache()
except Exception as _e:
    print('cuda cleanup skipped', _e)
OURS_W = %r
OURS_TAGS = 'r5t_nano336,r5tfull_nano336'
_pub = '/kaggle/working/_ab_submission.csv'
shutil.copy('/kaggle/working/submission.csv', _pub)
_code = None
for _d, _dirs, _files in os.walk('/kaggle/input'):
    _dirs[:] = [x for x in _dirs if x not in ('competitions', 'train_series', 'test_series')]
    if os.path.basename(_d) == 'rsna_knee' and '__init__.py' in _files:
        _code = os.path.dirname(_d); break
print('our code root:', _code, flush=True)
_out = '/kaggle/working/_ours_submission.csv'
if _code and OURS_W > 0:
    _pp = [_code, '/kaggle/working/_own_env', os.environ.get('PYTHONPATH', '')]   # _own_env: Part B's DICOM decoders
    _env = {**os.environ, 'PYTHONPATH': os.pathsep.join(p for p in _pp if p)}
    _rc = subprocess.run([sys.executable, '-m', 'rsna_knee.mil.ours', '--public', _pub, '--out', _out,
                          '--w', str(OURS_W), '--tags', OURS_TAGS], env=_env).returncode
    print('ours leg exit code', _rc, flush=True)
    if _rc == 0 and os.path.exists(_out):
        import pandas as _pd
        _a, _b = _pd.read_csv(_pub), _pd.read_csv(_out)
        if list(_a.columns) == list(_b.columns) and len(_a) == len(_b) and _b.notna().all().all():
            shutil.copy(_out, '/kaggle/working/submission.csv')
            print('submission.csv = previous + ours blend at', OURS_W, flush=True)
''' % ours_w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("goodpjw_nb")
    p.add_argument("eff_nb")
    p.add_argument("--w-eff", type=float, default=0.0)
    p.add_argument("--ours-w", type=float, default=0.10)
    a = p.parse_args()
    nb = json.load(open(a.goodpjw_nb, encoding="utf8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    cells = nb["cells"]
    # cells[0..7] = intro + Part A (nartaa, ends by writing submission.csv); cells[8..] = Part B (reader)
    assert "Part B" in "".join(cells[8]["source"]), "goodpjw notebook layout changed"
    head = {"cell_type": "markdown", "metadata": {}, "source": [
        "# RSNA Knee: OAI track blend (private)\n\n",
        "Track **OAI**: uses public weights trained with OAI external data.\n",
        f"Parts A+B: goodpjw2008's public 0.950 notebook unchanged. Part A2: nartaa 0.945 efficiency checkpoint, w={a.w_eff}. ",
        f"Part C: our R5t leg, w={a.ours_w}. Every added part falls back to the previous file on failure.\n"]}
    new = [head] + cells[:8]
    if a.w_eff > 0:
        new += eff_cells(a.eff_nb, a.w_eff)
    new += cells[8:]
    if a.ours_w > 0:
        new.append(ours_cell(a.ours_w))
    nb["cells"] = new
    json.dump(nb, open(os.path.join(HERE, "rsna-knee-oai-blend.ipynb"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("cells", len(new), "w_eff", a.w_eff, "ours_w", a.ours_w)


if __name__ == "__main__":
    main()
