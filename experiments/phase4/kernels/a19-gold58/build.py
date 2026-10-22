"""A19 gold-58 kernel (pre-registered in docs/phase4_notes.md, 2026-10-21 15:35 UTC): A16's own Raptor-384 script on the 58
gold studies, once for nartaa's 384 checkpoint (harness reference against gold_acc) and once for each candidate member.

    python build.py <rsna-knee-oai-a16.ipynb> <rsna-knee-oai-gold58.ipynb>

Only the checkpoint name and SHA pins, the test root (gold root first) and the two output names change; the model, windows,
mirror TTA and rank blend are nartaa's. Outputs: <tag>_sub.csv, <tag>_raw.csv per checkpoint, a19_gold_info.json (timings).
No AUC is computed here.
"""
import json, sys

GOLD_ROOT = '/kaggle/working/gold_root'
OLD_NAME = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
OLD_SHA = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
MEMBERS = [   # tag, file, sha256 (computed locally from the downloaded public files)
    ('n384', OLD_NAME, OLD_SHA),
    ('fullb', 'raptor_ft_oai384_fullb_swa.pt', '900dddb5f0d6ac2861543a4875e935ca1f7560c7ed569f8b2281fc74429cbfb1'),
    ('cohort', 'coatnet_384_full_cohort_swa.pt', '755ee7dd4f17d54f8bd7aa0caf713b3b8d65b6af6ea6660d2bda260639abafca'),
    ('bs14', 'raptor_ft_coat384_full_bs14.pt', 'df66a00b98a3b6cf9e8e4e83060b54b93ce1b6a4801f2cc3c7eeb56756dd3e5c'),
]
_FIND = "_weights = next((p for p in _candidates if p.is_file()), None)"
_FIND_NEW = ("_weights = next((p for p in _candidates if p.is_file()), None)\n"
             "if _weights is None:   # [A19] the member's own dataset mount (depth <= 3, never the competition tree)\n"
             "    import glob as _g\n"
             "    _weights = next((Path(q) for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/')\n"
             "                     for q in _g.glob(pat + _weight_name) if 'competition' not in q), None)")
_ROOT = 'cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",'
_RAW = '"/kaggle/working/raptor_raw.csv"'
_OUT = 'out = "/kaggle/working/raptor_sub.csv"'


def patch(src, tag, name, sha, gold_root=None):
    """A16's Raptor-384 script with the checkpoint pins swapped and the outputs renamed to <tag>_raw/_sub.csv."""
    for s, n in ((OLD_NAME, 2), (OLD_SHA, 2), (_FIND, 1), (_ROOT, 1), (_RAW, 1), (_OUT, 1)):
        assert src.count(s) == n, ('Raptor script changed', s[:60], src.count(s))
    src = src.replace(OLD_NAME, name).replace(OLD_SHA, sha).replace(_FIND, _FIND_NEW)
    src = src.replace(_RAW, '"/kaggle/working/%s_raw.csv"' % tag).replace(_OUT, 'out = "/kaggle/working/%s_sub.csv"' % tag)
    if gold_root:
        src = src.replace(_ROOT, 'cands = ["%s", "/kaggle/input/competitions/rsna-knee-abnormality-detection",' % gold_root)
    return src


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


if __name__ == '__main__':
    a16 = json.load(open(sys.argv[1], encoding='utf8'))
    g58 = json.load(open(sys.argv[2], encoding='utf8'))
    raptor = ''.join(a16['cells'][3]['source'])
    assert raptor.startswith('%%writefile /tmp/raptor_run.py\n')
    raptor = raptor.split('\n', 1)[1]
    root = ''.join(g58['cells'][1]['source'])
    scripts = {tag: patch(raptor, tag, name, sha, GOLD_ROOT) for tag, name, sha in MEMBERS}
    driver = '''import json, os, subprocess, sys, time
SCRIPTS = %r
info = {}
for tag, src in SCRIPTS.items():
    p = '/tmp/%%s_run.py' %% tag
    open(p, 'w').write(src)
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', p], timeout=3600).returncode
    info[tag] = {'rc': rc, 'seconds': round(time.time() - t, 1),
                 'sub': os.path.isfile('/kaggle/working/%%s_sub.csv' %% tag)}
    print('A19_GOLD', tag, info[tag], flush=True)
json.dump(info, open('/kaggle/working/a19_gold_info.json', 'w'), indent=1)
print('A19_GOLD_DONE', json.dumps(info), flush=True)
''' % scripts
    nb = {'cells': [cell(__doc__, 'markdown'), cell(root), cell(driver)], 'metadata': g58['metadata'],
          'nbformat': 4, 'nbformat_minor': 4}
    name = 'rsna-knee-a19-gold58'
    json.dump(nb, open(name + '.ipynb', 'w', encoding='utf8'), indent=1)
    meta = {"id": "tranbadat/" + name, "title": name, "code_file": name + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": ["gpu"],
            "dataset_sources": ["nartaa/rsna-knee-publication-swa-weights-20261007", "ephemeralbyte/rsna-knee-fullb-w",
                                "manatoshoji03/rsna-knee-coat-extra2", "manatoshoji03/rsna-knee-coat-extra"],
            "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
            "machine_shape": "NvidiaTeslaT4"}
    json.dump(meta, open('kernel-metadata.json', 'w'), indent=1)
    print('ok', name, list(scripts))
