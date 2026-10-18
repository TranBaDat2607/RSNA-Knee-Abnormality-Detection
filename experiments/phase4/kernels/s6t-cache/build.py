"""Write one Kaggle script kernel per S6T cache shard.

    python build.py dev              -> k_dev/   (shard 0 of 1, first 12 studies)
    python build.py shards N         -> k_s0/ .. k_s{N-1}/   (shard i of N, all studies of the shard)

Push each folder with `kaggle kernels push -p <folder>`.  GPU kernels (the SKM-TEA V-Net runs on CUDA; on CPU a study took
73 s per worker), internet off, same image as A12.  Rows are stored as JPEG q90: 13.6 MB/study, 4 shards stay under 20 GB.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "pxcache_build.py").read_text()
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


def make(tag, shard, nshard, limit, gpu=True):
    d = HERE / ("k_" + tag)
    d.mkdir(exist_ok=True)
    src, n = re.subn(r"^SHARD, NSHARD, LIMIT = .*$", "SHARD, NSHARD, LIMIT = %d, %d, %d" % (shard, nshard, limit), SRC, count=1,
                     flags=re.M)
    assert n == 1
    slug = "rsna-knee-s6t-cache-" + tag
    (d / (slug + ".py")).write_text(src)
    meta = {"id": "tranbadat/" + slug, "title": slug, "code_file": slug + ".py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": ["nartaa/rsna-knee-s6fall-weights", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            "docker_image": IMAGE}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(d)


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        make("dev", 0, 1, 12)
    else:
        n = int(sys.argv[2])
        for i in range(n):
            make("s%d" % i, i, n, 0)
