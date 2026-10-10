"""Build the CLEAN-track blend: community stack (jiweiliu 0.943, as in ours-blend-r3) + goodpjw2008's 2.5D ConvNeXt
reader at a flat 0.30 rank weight (goodpjw measured 0.944 public for exactly this pair), optional our R5t leg.

    python build.py <goodpjw_nb.ipynb> [--ours-w 0.0]

No OAI-trained weights: the stack is the pre-October community stack, the reader was trained on report labels only.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'oai-blend'))
from build import ours_cell  # noqa: E402  same Part C cell as the OAI track

p = argparse.ArgumentParser()
p.add_argument('goodpjw_nb')
p.add_argument('--ours-w', type=float, default=0.0)
p.add_argument('--reader-w', type=float, default=0.30)
a = p.parse_args()
base = json.load(open(os.path.join(HERE, '..', 'ours-blend-r3', 'rsna-knee-ours-blend.ipynb'), encoding='utf8'))
assert 'OURS_TAGS' in ''.join(base['cells'][-1]['source']), 'last cell of ours-blend-r3 is not our leg'
cells = base['cells'][:-1]
g = json.load(open(a.goodpjw_nb, encoding='utf8'))['cells']
assert 'Part B' in ''.join(g[8]['source'])
reader = [dict(c) for c in g[8:14]]
src = ''.join(reader[-1]['source'])
old_pt = [l for l in src.splitlines() if l.startswith('OWN_W_PER_TARGET = ')][0]
src = src.replace('OWN_W = 0.15\n', 'OWN_W = %r\n' % a.reader_w).replace(old_pt, 'OWN_W_PER_TARGET = {}  # flat weight: goodpjw v1-5 setting (stack + reader 0.30 = 0.944)')
assert 'OWN_W = %r' % a.reader_w in src
reader[-1] = dict(reader[-1], source=src)
for c in cells + reader:
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None
head = {'cell_type': 'markdown', 'metadata': {}, 'source': [
    '# RSNA Knee: CLEAN track blend (private)\n\n',
    'Community stack (jiweiliu 0.943) + goodpjw2008 2.5D ConvNeXt reader at %.2f (flat)' % a.reader_w,
    (' + our R5t leg at %.2f' % a.ours_w if a.ours_w > 0 else '') + '. No OAI-trained weights.\n']}
cells = [head] + cells + reader + ([ours_cell(a.ours_w)] if a.ours_w > 0 else [])
base['cells'] = cells
json.dump(base, open(os.path.join(HERE, 'rsna-knee-clean-blend.ipynb'), 'w', encoding='utf8'), indent=1, ensure_ascii=False)
print('cells', len(cells))
