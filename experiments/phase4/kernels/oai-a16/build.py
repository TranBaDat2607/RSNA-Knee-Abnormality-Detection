"""Build A16 = heliosli's public 0.9586 notebook unchanged apart from the R6 dataset slug (docs/phase4_notes.md, 2026-10-19).

    python build.py <this-is-best-public-dataset-new-version-0-9586.ipynb, pulled 2026-10-19>

The notebook reads R6 from `heliosli/rsna-knee-r6-frozen-fe055406`, which is non-versioned (a submission is refused); the same
bundle (manifest sha256 6508ce69.., checked by the cell together with every file's sha256) is `rsna-knee-r6-frozen-fe055406-w1`.
goodpjw's notebook is not attached as a kernel source: Part B writes the reader's three sources from embedded strings
(sha256-checked, identical to A7's) and finds `knee.py` + checkpoints in `goodpjw2008/rsna-knee-2-5d-convnext-reader`.
"""
import json
import sys


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


nb = json.load(open(sys.argv[1], encoding='utf8'))
c6 = ''.join(nb['cells'][6]['source'])
assert "BUNDLE_ID = 'fe055406fad6a12f'" in c6 and 'FLIP_TTA = True' in c6
c6 = sub(c6, "C_SLUG = 'rsna-knee-r6-frozen-fe055406'", "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'   # [A16] versioned copy, same bundle")
nb['cells'][6]['source'] = c6
nb['cells'].insert(1, {'cell_type': 'markdown', 'metadata': {},
                       'source': "## A16: heliosli's public 0.9586 notebook, unchanged apart from the R6 dataset slug\n\n"
                                 "The R6 bundle is read from the versioned `heliosli/rsna-knee-r6-frozen-fe055406-w1` (same manifest "
                                 "and files, sha256-checked by the compact cell)."})
json.dump(nb, open('rsna-knee-oai-a16.ipynb', 'w', encoding='utf8'), indent=1)
meta = {'id': 'tranbadat/rsna-knee-oai-a16', 'title': 'rsna-knee-oai-a16', 'code_file': 'rsna-knee-oai-a16.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'keywords': ['gpu'],
        'dataset_sources': ['goodpjw2008/rsna-knee-2-5d-convnext-reader', 'heliosli/rsna-knee-r6-frozen-fe055406-w1',
                            'nartaa/rsna-knee-publication-swa-weights-20261007', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004'],
        'kernel_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'], 'model_sources': [],
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
        'machine_shape': 'NvidiaTeslaT4'}
json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
print('A16 cells', len(nb['cells']))
