# Diagnostic: our R3 ConvNeXt-nano 336 px 5-fold leg alone (rank space), no public stack.
import os, subprocess, sys, shutil, glob
code = None
for d, dirs, files in os.walk('/kaggle/input'):
    dirs[:] = [x for x in dirs if x not in ('competitions', 'train_series', 'test_series')]
    if os.path.basename(d) == 'rsna_knee' and '__init__.py' in files:
        code = os.path.dirname(d); break
print('code root', code, flush=True)
ss = (glob.glob('/kaggle/input/competitions/rsna-knee-abnormality-detection/sample_submission.csv')
      + glob.glob('/kaggle/input/rsna-knee-abnormality-detection/sample_submission.csv'))[0]
env = {**os.environ, 'PYTHONPATH': code}
rc = subprocess.run([sys.executable, '-m', 'rsna_knee.mil.ours', '--public', ss, '--out',
                     '/kaggle/working/submission.csv', '--w', '1.0', '--tags', os.environ.get('TAGS', 'r3_nano336'),
                     '--workers', '4'], env=env).returncode
print('rc', rc, flush=True)
import pandas as pd
s = pd.read_csv('/kaggle/working/submission.csv'); print(s.shape); print(s.describe().T)
