"""Check a dev run (3-row public test) of A19/A20/A21: required log markers, no fallback, submission schema.

usage: python dev_check.py <kernel slug> <out dir> <markers...>
Downloads the log and submission.csv, prints each marker's lines, then runs kgkit validate.
"""
import json, os, subprocess, sys

slug, out, markers = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
subprocess.run(['kaggle', 'kernels', 'output', 'tranbadat/' + slug, '-p', out, '--file-pattern',
                r'(submission\.csv$|\.log$|receipt\.json$)'], check=False, capture_output=True)
log = open(os.path.join(out, slug + '.log'), encoding='utf8', errors='replace').read()
lines = []
try:
    lines = [e.get('data', '') for e in json.loads(log)]
except Exception:
    lines = log.splitlines()
text = ''.join(lines)
ok = True
for m in markers:
    hits = [l.strip()[:300] for l in text.splitlines() if m in l]
    print(('OK   ' if hits else 'MISS ') + m, '|', hits[-1] if hits else '')
    ok &= bool(hits)
for bad in ('BLEND_FALLBACK', '_SKIPPED', 'Traceback', 'S6T_ATTACH_FAILED', 'A19_MEMBER_SKIPPED'):
    hits = [l.strip()[:300] for l in text.splitlines() if bad in l]
    if hits:
        print('WARN ', bad, '|', hits[:3])
sub = os.path.join(out, 'submission.csv')
print('submission.csv', os.path.isfile(sub))
if os.path.isfile(sub):
    env = dict(os.environ, PYTHONPATH='/home/nammovuivui/.claude/plugins/cache/claude-code-kaggle-grandmaster/kaggle-grandmaster/1.6.2')
    r = subprocess.run([sys.executable, '-m', 'kgkit', 'validate', sub, 'data/sample_submission.csv'], env=env,
                       capture_output=True, text=True, cwd='/home/nammovuivui/RSNA-Knee-Abnormality-Detection')
    print('validate:', (r.stdout + r.stderr).strip()[-400:])
print('DEV', 'PASS' if ok else 'FAIL', slug)
