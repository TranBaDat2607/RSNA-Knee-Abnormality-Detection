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


def ft_script(acc_src, ckpt_glob, out_csv):
    """nartaa's accuracy-notebook source (320 crop, all 94 windows, anatomical mirror TTA) serving one of our
    FT96 checkpoints instead of the 0.949 one: the checkpoint is located by glob, the SHA/384 pins are relaxed."""
    sha = "assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'"
    res = 'assert int(ck.get("res", res_default)) == 384, "Expected original 384 checkpoint"'
    call = 'load_model(find_weight_file(arm["file"]), arm["arch"], arm["res"], d)'
    out = 'out = "/kaggle/working/submission.csv"'
    for x in (sha, res, call, out):
        assert x in acc_src, x
    src = (acc_src.replace(sha, "pass  # FT96 checkpoint: no SHA pin; " + sha[:20])
           .replace(res, 'assert int(ck.get("res", res_default)) in (320, 384)')
           .replace(call, 'load_model(_FT_CKPT, arm["arch"], arm["res"], d)')
           .replace(out, f'out = "/kaggle/working/{out_csv}"'))
    pats = ", ".join(repr(f"/kaggle/input/{'*/' * d}{ckpt_glob}") for d in (1, 2, 3))
    head = (f"import glob as _g\n_FT_CKPT = sorted(p for pat in ({pats}) for p in _g.glob(pat))[0]\n"
            "print('FT96 checkpoint', _FT_CKPT, flush=True)\n")
    return head + src


def leg_cells(title, script, w, out_csv, tmp, skip_after_h=0.0):
    """Write ``script`` to ``tmp``, run it, and rank-blend its ``out_csv`` into submission.csv at weight ``w``.
    ``skip_after_h`` > 0: skip the leg when the notebook has already run that long (needs the T0 cell first)."""
    write = code(f"# ---- {title} ----\nopen({tmp!r}, 'w').write(" + repr(script) + ")\n")
    run = code(r'''import gc, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
_W, _TMP, _OUT = %r, %r, %r
try:
    import torch; gc.collect(); torch.cuda.empty_cache()
except Exception as _e:
    print('cuda cleanup skipped', _e)
shutil.copy('/kaggle/working/submission.csv', '/kaggle/working/_prev_submission.csv')
_t0 = time.time()
try:
    _elapsed_h = (time.time() - globals().get('_NB_T0', time.time())) / 3600
    assert not (%r > 0 and _elapsed_h > %r), 'time guard: notebook already at %%.2f h' %% _elapsed_h
    _rc = subprocess.run([sys.executable, '-u', _TMP], timeout=3 * 3600).returncode
    assert _rc == 0, 'leg exit code %%d' %% _rc
    _a = pd.read_csv('/kaggle/working/_prev_submission.csv', dtype={'StudyInstanceUID': str})
    _e = pd.read_csv('/kaggle/working/' + _OUT, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[_a.StudyInstanceUID].reset_index()
    _lab = [c for c in _a.columns if c != 'StudyInstanceUID']
    assert _e[_lab].notna().all().all() and np.isfinite(_e[_lab].values).all(), 'leg NaN'
    _r = lambda d: d[_lab].rank(method='average', pct=True)
    _o = _a.copy()
    _o[_lab] = ((1 - _W) * _r(_a) + _W * _r(_e)).rank(method='average', pct=True)
    _o.to_csv('/kaggle/working/submission.csv', index=False)
    print('leg', _OUT, 'blended at', _W, 'in %%.0fs' %% (time.time() - _t0), flush=True)
except Exception as _x:
    shutil.copy('/kaggle/working/_prev_submission.csv', '/kaggle/working/submission.csv')
    print('leg', _OUT, 'FAILED, previous submission kept:', repr(_x), flush=True)
''' % (w, tmp, out_csv, skip_after_h, skip_after_h))
    return [write, run]


T0_CELL = code("import time\n_NB_T0 = time.time()  # notebook start, for the legs' time guards\n")


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
    p.add_argument("--ft-glob", default="", help="FT96 checkpoint glob below /kaggle/input/<x>/, e.g. ftO/ckpt_ep2.pt")
    p.add_argument("--w-ft", type=float, default=0.0)
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
        f"Part A3: FT96 {a.ft_glob} w={a.w_ft}. Part C: our R5t leg, w={a.ours_w}. Every added part falls back to the previous file on failure.\n"]}
    new = [head] + cells[:8]
    if a.w_eff > 0:
        new += eff_cells(a.eff_nb, a.w_eff)
    if a.w_ft > 0:
        acc_src = "".join(cells[7]["source"])
        new += leg_cells("Part A3: our FT96 checkpoint " + a.ft_glob, ft_script(acc_src, a.ft_glob, "ft_sub.csv"),
                         a.w_ft, "ft_sub.csv", "/tmp/ft_run.py")
    new += cells[8:]
    if a.ours_w > 0:
        new.append(ours_cell(a.ours_w))
    nb["cells"] = new
    json.dump(nb, open(os.path.join(HERE, "rsna-knee-oai-blend.ipynb"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("cells", len(new), "w_eff", a.w_eff, "ours_w", a.ours_w)


if __name__ == "__main__":
    main()
