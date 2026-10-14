"""Build the gold-58 diagnostic for the compact legs: R7 plain, R7 with flip TTA, S6.

    python build.py <a6_notebook.ipynb> <goodpjw_v12_notebook.ipynb> <oai_gold58_notebook.ipynb>

The "test" set is the 58 gold-labelled training studies (stand-in root from rsna-knee-oai-gold58). R7 reports
gold_in_training 0. Writes gold_r7_plain.csv, gold_r7_flip.csv, gold_s6.csv (raw probabilities) plus receipts.
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
    assert text.count(old) >= 1, old
    return text.replace(old, new)


a6, gp, g58 = (json.load(open(p, encoding='utf8')) for p in sys.argv[1:4])
root = 'import time\nNB_T0 = time.time()\n' + src_of(g58, 1)
r7_setup = sub(src_of(a6, 6), "DATA = find_mount(COMPETITION, 'test.csv')", "DATA = Path('/kaggle/working/gold_root')")
r7_run = '''
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
        shutil.copyfile(log, WORK / ('r7_%s_%s_log.txt' % (tag, log.parent.name)))
    print('R7', tag, 'rc', rc, 'seconds %.0f' % (time.time() - t), flush=True)
    if rc == 0:
        shutil.copyfile(pred / 'submission.csv', WORK / ('gold_r7_%s.csv' % tag))
        shutil.copyfile(pred / 'submission_receipt.json', WORK / ('r7_%s_receipt.json' % tag))
'''
s6_setup = sub(src_of(gp, 9), "DATA = find_mount(COMPETITION, 'test.csv')", "DATA = Path('/kaggle/working/gold_root')")
s6_run = sub(src_of(gp, 10), "shutil.copyfile(pred_s6 / 'submission.csv', WORK / 's6_sub.csv')",
             "shutil.copyfile(pred_s6 / 'submission.csv', WORK / 's6_sub.csv'); shutil.copyfile(pred_s6 / 'submission.csv', WORK / 'gold_s6.csv')")
md = __doc__
nb = {'cells': [cell(md, 'markdown'), cell(root), cell(r7_setup), cell(r7_run), cell(s6_setup), cell(s6_run)],
      'metadata': a6['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
json.dump(nb, open('rsna-knee-oai-gold58-r7.ipynb', 'w', encoding='utf8'), indent=1)
print('cells', len(nb['cells']))
