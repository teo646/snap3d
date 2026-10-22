"""SHIQ (JSHDR, third_party/SHIQ) worker: runs in the `shiq` conda env, not the pipeline's.

Standalone on purpose - it imports nothing from `pipeline`, and is written for that env's
Python 3.6 (no `from __future__ import annotations`, no dataclasses) - because the model
is only shipped as compiled Python 3.6 modules (`network/utils/*.cpython-36m*.so`) on torch
1.5 / CUDA 10.2. That torch cannot run on this repo's GPU (Blackwell, sm_120: cublas fails),
so it runs on the CPU, the way third_party/SHIQ/network/infer_cpu.py does: `torch.load` is
pointed at the CPU and `.cuda()` becomes a no-op. About 1.7 s a 3 MP frame.

`pipeline.core.highlight_removal_shiq` starts it with a JSON job: a list of
{"image": <in>, "out": <out>}. The model only reads and writes files, and names its output
`D_<input file name>` in a results folder, so each image goes through a scratch PNG named
`<n>_A.png`. The scratch copy is rebuilt from the pixel array, which drops any EXIF
orientation tag: the pixels the pipeline works in are the stored ones, as COLMAP posed them.

What it writes to `out` is the network's own output, unmodified (lossless PNG). Keeping the
original's fine detail is the caller's job - see `merge_shiq_detail`.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from PIL import Image

_load = torch.load
torch.load = lambda *a, **k: _load(*a, **dict(k, map_location=torch.device("cpu")))
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    job = json.load(open(parser.parse_args().job))

    network_dir = os.path.join(job["shiq_root"], "network")
    os.chdir(network_dir)  # the checkpoints are found relative to it
    sys.path.insert(0, ".")
    from utils.detectionRemoval import test_single_highlight_image

    scratch = tempfile.mkdtemp(prefix="shiq_")
    try:
        items = job["items"]
        for n, item in enumerate(items):
            pixels = np.asarray(Image.open(item["image"]).convert("RGB"))
            src = os.path.join(scratch, "%d_A.png" % n)
            Image.fromarray(pixels).save(src)
            test_single_highlight_image(argparse.Namespace(image_path=src, results_path=scratch, checkpoint="jshdr"))
            result = os.path.join(scratch, "D_%d_A.png" % n)
            if not os.path.exists(result):
                raise RuntimeError("SHIQ wrote no output for %s" % item["image"])
            shutil.move(result, item["out"])
            os.remove(src)
            if (n + 1) % 10 == 0 or n + 1 == len(items):
                print("  %d/%d" % (n + 1, len(items)), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
