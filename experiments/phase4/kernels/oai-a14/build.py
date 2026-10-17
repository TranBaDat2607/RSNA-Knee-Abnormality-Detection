"""Build A14 = A12 with the compact leg R7 replaced by heliosli's R6 (rule 2 in docs/phase4_notes.md, 2026-10-16).

    python build.py <A12 notebook (oai-957s6f)> <flip|plain>

Only cell 6 (the compact stage) changes: bundle id fe055406fad6a12f, dataset heliosli/rsna-knee-r6-frozen-fe055406-w1
(manifest sha256 6508ce69.., the same bytes as rostigor/rsna-knee-r6-fe055406 that goodpjw's 0.957/0.958 submissions use),
and FLIP_TTA. The cell's own asserts check the bundle id and every file's sha256. Raptor, s6fall and the blend cells are A12's.
"""
import json
import sys


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


nb = json.load(open(sys.argv[1], encoding='utf8'))
flip = {'flip': True, 'plain': False}[sys.argv[2]]
c6 = ''.join(nb['cells'][6]['source'])
c6 = sub(c6, "BUNDLE_ID = '8c7b666e27dc0e8f'", "BUNDLE_ID = 'fe055406fad6a12f'   # [A14] heliosli's R6")
c6 = sub(c6, "C_SLUG = 'rsna-knee-compact-r7cnxfsss'", "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'   # [A14] R6 bundle (manifest sha256 6508ce69..)")
c6 = sub(c6, "FLIP_TTA = False", "FLIP_TTA = %s   # [A14] %s" % (flip, sys.argv[2]))
nb['cells'][6]['source'] = c6
nb['cells'].insert(1, {'cell_type': 'markdown', 'metadata': {},
                       'source': '## A14: A12 with the compact leg R7 replaced by heliosli\'s R6 (%s)\n\nOnly the compact '
                                 'stage\'s bundle changes; Raptor 384 + 224, s6fall and both blend cells are A12\'s.' % sys.argv[2]})
json.dump(nb, open('rsna-knee-oai-a14.ipynb', 'w', encoding='utf8'), indent=1)
meta = {'id': 'tranbadat/rsna-knee-oai-a14', 'title': 'rsna-knee-oai-a14', 'code_file': 'rsna-knee-oai-a14.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'keywords': ['gpu'],
        'dataset_sources': ['heliosli/rsna-knee-r6-frozen-fe055406-w1', 'nartaa/rsna-knee-publication-swa-weights-20261007',
                            'heliosli/rsna-orthofoundation-fold0-epoch7-20261004', 'nartaa/rsna-knee-s6fall-weights'],
        'kernel_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'], 'model_sources': [],
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
        'machine_shape': 'NvidiaTeslaT4'}
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('A14', sys.argv[2], 'cells', len(nb['cells']))
