"""Build A24 / A25 / A26 = A22 with only the final slot weight changed (docs/phase4_notes.md, A24/A25/A26).

    python build.py <rsna-knee-oai-a22.ipynb> a24|a25|a26

final = (1 - w) x A16-with-cohort + w x rank(1/2 rank(s6fall) + 1/2 rank(S6T)); A22's w = 1/3.
"""
import json, os, sys

nb = json.load(open(sys.argv[1], encoding='utf8'))
tag = sys.argv[2]
W = {'a24': ('1.0 / 2.0', '1/2'), 'a25': ('1.0 / 4.0', '1/4'), 'a26': ('2.0 / 5.0', '2/5')}[tag]


def sub_once(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


i = [k for k, c in enumerate(nb['cells']) if '# [A20] final = 2/3 x A16 + 1/3 x S6 slot' in ''.join(c['source'])]
assert len(i) == 1, i
s = ''.join(nb['cells'][i[0]]['source'])
s = sub_once(s, "# [A20] final = 2/3 x A16 + 1/3 x S6 slot,",
             "# [%s] final = (1 - w) x A16 + w x S6 slot, w = %s (A22: 1/3),"  % (tag.upper(), W[1]))
s = sub_once(s, "_out[_c] = np.where(_okf, (2.0 / 3.0) * _v + (1.0 / 3.0) * _slot, _v)",
             "_out[_c] = np.where(_okf, (1.0 - _WS) * _v + _WS * _slot, _v)")
s = sub_once(s, "import numpy as np, pandas as pd, shutil\n", "import numpy as np, pandas as pd, shutil\n_WS = %s   # [%s]\n" % (W[0], tag.upper()))
s = sub_once(s, "print('A20_BLEND_DONE seeds %d;", "print('A20_BLEND_DONE w %.3f seeds %d;")
s = sub_once(s, "        S6T_SEEDS, _okf.sum(),", "        _WS, S6T_SEEDS, _okf.sum(),")
nb['cells'][i[0]]['source'] = s
nb['cells'][1]['source'] = ("## %s: A22 with the final S6-slot weight w = %s (A22: 1/3)\n\n" % (tag.upper(), W[1]) +
                            "Every stage, cap and fallback is A22's; only the last blend weight differs.")
name = 'rsna-knee-oai-' + tag
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'oai-' + tag)
os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, name + '.ipynb'), 'w', encoding='utf8'), indent=1)
meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), 'kernel-metadata.json')))
meta.update(id='tranbadat/' + name, title=name, code_file=name + '.ipynb')
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
print('ok', name)
