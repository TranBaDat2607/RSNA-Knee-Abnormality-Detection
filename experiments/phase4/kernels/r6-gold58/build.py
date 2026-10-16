"""Build the gold-58 diagnostic for heliosli's R6 compact model: R6 plain, then R6 with flip TTA.

    python build.py <A12 notebook (oai-957s6f)> <oai_gold58_notebook.ipynb>

The "test" set is the 58 gold-labelled training studies (stand-in root from rsna-knee-oai-gold58). R6 (bundle
fe055406fad6a12f, heliosli/rsna-knee-r6-frozen-fe055406-w1) reports gold_in_training 0. The R6 stage is A12's R7 stage
with the bundle id and dataset changed (as in heliosli's public 0.958 notebook). Writes gold_r6_plain.csv and
gold_r6_flip.csv (raw probabilities) plus receipts and worker logs. Rules: docs/phase4_notes.md (2026-10-16).
"""
import json
import sys


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


def src_of(nb, i):
    return ''.join(nb['cells'][i]['source'])


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


a12, g58 = (json.load(open(p, encoding='utf8')) for p in sys.argv[1:3])
root = 'import time\nNB_T0 = time.time()\n' + src_of(g58, 1)
r6_setup = src_of(a12, 6)
r6_setup = sub(r6_setup, "BUNDLE_ID = '8c7b666e27dc0e8f'", "BUNDLE_ID = 'fe055406fad6a12f'   # [r6]")
r6_setup = sub(r6_setup, "C_SLUG = 'rsna-knee-compact-r7cnxfsss'", "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'   # [r6]")
r6_setup = sub(r6_setup, "DATA = find_mount(COMPETITION, 'test.csv')", "DATA = Path('/kaggle/working/gold_root')")
assert "FLIP_TTA = False" in r6_setup
r6_run = '''
import json, os, shutil, subprocess, sys, time
deadline = T0 + DEADLINE_HOURS * 3600
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
for tag, extra in (('plain', []), ('flip', ['--flip-tta'])):
    pred = TMP / ('prediction_' + tag)
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                         '--code', str(CODE), '--bundle', str(BUNDLE), '--arm', ARM, '--output', str(pred),
                         '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                         '--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS + extra),
                         '--max-fallback-fraction', '1.0'], env=env).returncode
    for log in sorted(pred.glob('worker*/log.txt')):
        shutil.copyfile(log, WORK / ('r6_%s_%s_log.txt' % (tag, log.parent.name)))
    print('R6', tag, 'rc', rc, 'seconds %.0f' % (time.time() - t), flush=True)
    if rc == 0:
        shutil.copyfile(pred / 'submission.csv', WORK / ('gold_r6_%s.csv' % tag))
        shutil.copyfile(pred / 'submission_receipt.json', WORK / ('r6_%s_receipt.json' % tag))
'''
nb = {'cells': [cell(__doc__, 'markdown'), cell(root), cell(r6_setup), cell(r6_run)],
      'metadata': a12['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
json.dump(nb, open('rsna-knee-r6-gold58.ipynb', 'w', encoding='utf8'), indent=1)
print('cells', len(nb['cells']))
