"""Build A22 / A23 = runtime-safe A21 (docs/phase4_notes.md, 2026-10-22 02:50 UTC).

    python build.py <rsna-knee-oai-a21.ipynb> a22|a23

A21's cells with three runtime changes (predictions are A21's whenever every stage finishes):
1. the cohort member runs in its own session; it starts only with >= cap + 120 min left and its whole process group is
   killed at cap = 1.5 x RAPTOR_SECONDS + 10 min (stage 1 is the same script with an equal-size checkpoint);
2. Part B's reader subprocess ends 25 min before 9 h at the latest (Part B's except-branch keeps the stage-3 file);
3. a23 only: R6's flip TTA off (FLIP_TTA = False, R6 plain as in A12/A14), which frees ~1.75 h for the extra legs.
"""
import json, os, sys

nb = json.load(open(sys.argv[1], encoding='utf8'))
tag = sys.argv[2]
assert tag in ('a22', 'a23'), tag


def sub_once(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


def find(marker):
    i = [k for k, c in enumerate(nb['cells']) if marker in ''.join(c['source'])]
    assert len(i) == 1, (marker, i)
    return i[0]


m = find("# [A19] extra Raptor-384 member cohort")
s = ''.join(nb['cells'][m]['source'])
s = sub_once(s, "# its own process; starts only with >= 135 min left, must end 60 min before 9 h. Never fatal.",
             "# [A22] its own session; starts only with >= cap + 120 min left, the whole process group is killed at\n"
             "# cap = 1.5 x RAPTOR_SECONDS + 10 min (stage 1 = the same script, equal-size checkpoint). Never fatal.")
s = sub_once(s, "_left = 9 * 3600 - (time.time() - NB_T0)\nif _left < 135 * 60:\n",
             "import signal\n_left = 9 * 3600 - (time.time() - NB_T0)\n_cap = 1.5 * RAPTOR_SECONDS + 600\n"
             "if _left < _cap + 120 * 60:\n")
s = sub_once(s, "print('A19_MEMBER_SKIPPED cohort: only %.0f min left' % (_left / 60), flush=True)",
             "print('A22_MEMBER_SKIPPED cohort: only %.0f min left, cap %.0f min' % (_left / 60, _cap / 60), flush=True)")
s = sub_once(s, """    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/a19_cohort_run.py'], timeout=_left - 60 * 60).returncode
    except subprocess.TimeoutExpired:
        _rc = 'timeout'
""", """    _p = subprocess.Popen([sys.executable, '-u', '/tmp/a19_cohort_run.py'], start_new_session=True)
    try:
        _rc = _p.wait(timeout=_cap)
    except subprocess.TimeoutExpired:
        _rc = 'timeout'
    finally:
        try:
            os.killpg(_p.pid, signal.SIGKILL)   # the member and every process it forked (its Pool workers)
        except ProcessLookupError:
            pass
        _p.wait()
""")
s = sub_once(s, "print('A19_MEMBER cohort rc %s ok %s seconds %.0f' % (_rc, _ok, time.time() - _t), flush=True)",
             "print('A22_MEMBER cohort rc %s ok %s seconds %.0f cap %.0f' % (_rc, _ok, time.time() - _t, _cap), flush=True)")
nb['cells'][m]['source'] = s

b = find("# Part B (V2): goodpjw2008's 2.5D ConvNeXt MIL reader")
s = ''.join(nb['cells'][b]['source'])
s = sub_once(s, """                  check=True, env=_o_env)
""", """                  check=True, env=_o_env,   # [A22] ends 25 min before 9 h at the latest; the except-branch keeps stage 3's file
                  timeout=max(60, 9 * 3600 - (_o_time.time() - globals().get('NB_T0', _o_time.time())) - 25 * 60))
""")
nb['cells'][b]['source'] = s

if tag == 'a23':
    r = find("FLIP_TTA = True   # [d957] package E: v4c's flip TTA on R6")
    nb['cells'][r]['source'] = sub_once(''.join(nb['cells'][r]['source']),
                                        "FLIP_TTA = True   # [d957] package E: v4c's flip TTA on R6",
                                        "FLIP_TTA = False   # [A23] R6 plain (as A12/A14): frees ~1.75 h for the cohort, s6fall and S6T legs")

title = {'a22': "## A22: A21 (A16 + cohort in the 384 slot + s6fall and our S6T in the S6 slot), runtime-safe\n\n",
         'a23': "## A23: A22 with R6 plain (flip TTA off) to make room for the extra legs\n\n"}[tag]
nb['cells'][1]['source'] = title + (
    "A16's cells with time-capped extras: the cohort member runs in its own process group, starts only with its cap + 2 h "
    "left and is killed at 1.5 x the stage-1 time + 10 min; Part B's reader ends 25 min before 9 h at the latest; "
    "s6fall + S6T are self-guarded. final = 2/3 x A19 + 1/3 x rank(1/2 rank(s6fall) + 1/2 rank(S6T)); every failure "
    "falls back to the previous file.")
name = 'rsna-knee-oai-' + tag
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'oai-' + tag)
os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, name + '.ipynb'), 'w', encoding='utf8'), indent=1)
meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'oai-a21', 'kernel-metadata.json')))
meta.update(id='tranbadat/' + name, title=name, code_file=name + '.ipynb')
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
print(name, 'cells', len(nb['cells']))
