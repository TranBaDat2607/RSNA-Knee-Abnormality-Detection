"""Build A21 = A19 + A20 (docs/phase4_notes.md, 2026-10-21 16:40 UTC).

    python build.py <rsna-knee-oai-a19.ipynb> <rsna-knee-oai-a17.ipynb>

The A19 notebook (A16 + cohort in the 384 slot) gets A17's s6fall stage and blend cells appended, exactly as A17 was built
from A16; then the A20 builder adds S6T to that stage and the A20 blend: final = 2/3 x A19 + 1/3 x rank(1/2 rank(s6fall)
+ 1/2 rank(S6T)).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
a19 = json.load(open(sys.argv[1], encoding='utf8'))
a17 = json.load(open(sys.argv[2], encoding='utf8'))
md, stage, blend = a17['cells'][-3:]
assert md['cell_type'] == 'markdown' and "[A12] nartaa's s6fall" in ''.join(stage['source']) and 'A12_BLEND_DONE' in ''.join(blend['source'])
assert not any("s6fall" in ''.join(c['source']) for c in a19['cells'] if c['cell_type'] == 'code')
a19['cells'] += [md, stage, blend]
tmp = os.path.join(HERE, '_a19_plus_s6fall.ipynb')
json.dump(a19, open(tmp, 'w', encoding='utf8'), indent=1)
base_meta = os.path.join(HERE, '_base_meta.json')
json.dump(json.load(open(os.path.join(HERE, '..', 'oai-a19', 'kernel-metadata.json'))), open(base_meta, 'w'), indent=1)
subprocess.run([sys.executable, os.path.join(HERE, '..', 'oai-a20', 'build.py'), tmp, 'rsna-knee-oai-a21', base_meta],
               cwd=HERE, check=True)
os.remove(tmp)
os.remove(base_meta)
nb = json.load(open(os.path.join(HERE, 'rsna-knee-oai-a21.ipynb'), encoding='utf8'))
nb['cells'][1]['source'] = ("## A21: A16 + cohort in the Raptor-384 slot (A19) + our S6T sharing the S6 slot with s6fall (A20)\n\n"
                            "A16's cells are unchanged. Extra stages, each time-guarded and never fatal: cohort (384 slot, merged "
                            "before stage 3), then nartaa's s6fall with our S6T in the same inference call. final = 2/3 x A19 + "
                            "1/3 x rank(1/2 rank(s6fall) + 1/2 rank(S6T)).")
json.dump(nb, open(os.path.join(HERE, 'rsna-knee-oai-a21.ipynb'), 'w', encoding='utf8'), indent=1)
print('A21 cells', len(nb['cells']))
