"""Build A17 = A16 + nartaa's s6fall as a third equal leg (docs/phase4_notes.md, 2026-10-19).

    python build.py <rsna-knee-oai-a16.ipynb> <rsna-knee-oai-a14.ipynb>

A14's last two code cells (A12's s6fall stage with the embedded script, and A12's blend: final = 2/3 x base + 1/3 x
rank(s6fall), time guard, < 90 % coverage or any failure keeps the base file) are appended unchanged after A16's Part B.
"""
import json
import sys

a16 = json.load(open(sys.argv[1], encoding='utf8'))
a14 = json.load(open(sys.argv[2], encoding='utf8'))
stage, blend = a14['cells'][-2], a14['cells'][-1]
assert ''.join(stage['source']).lstrip().startswith("# [A12] nartaa's s6fall"), 'unexpected A14 stage cell'
assert 'A12_BLEND_DONE' in ''.join(blend['source']), 'unexpected A14 blend cell'
a16['cells'][1]['source'] = ("## A17: A16 (heliosli's public 0.9586 recipe) + nartaa's s6fall as a third equal leg\n\n"
                             "A16's cells are unchanged; the last two cells are A12's s6fall stage and blend, unchanged: "
                             "final = 2/3 x A16 + 1/3 x rank(s6fall), time-guarded, never fatal (A16's file stays on any failure).")
a16['cells'] += [{'cell_type': 'markdown', 'metadata': {}, 'source': '## A17: nartaa s6fall as a third equal leg (A12 cells, unchanged)'},
                 stage, blend]
json.dump(a16, open('rsna-knee-oai-a17.ipynb', 'w', encoding='utf8'), indent=1)
meta = json.load(open('../oai-a16/kernel-metadata.json'))
meta.update(id='tranbadat/rsna-knee-oai-a17', title='rsna-knee-oai-a17', code_file='rsna-knee-oai-a17.ipynb')
meta['dataset_sources'] = meta['dataset_sources'] + ['nartaa/rsna-knee-s6fall-weights']
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('A17 cells', len(a16['cells']))
