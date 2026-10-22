"""Export real BF16 weight rows from a GGUF to a raw file for the GEMV benchmark (read-only input).
Concatenates the given tensors (all must share the inner dimension) as uint16 rows; writes a small JSON header."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pattern", default="ffn_up.weight")
    ap.add_argument("--max-tensors", type=int, default=8)
    ap.add_argument("--gguf-root", type=Path, default=Path("<llama.cpp-checkout>/gguf-py"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.gguf_root))
    from gguf import GGUFReader
    r = GGUFReader(str(a.model), "r")
    mats, names = [], []
    for t in r.tensors:
        if t.name.endswith(a.pattern) and t.tensor_type.name == "BF16":
            d = np.asarray(t.data).view(np.uint16)
            d = d.reshape(-1, int(t.shape[0]))  # GGUF shape[0] is the inner (contiguous) dimension
            mats.append(d)
            names.append(t.name)
            if len(mats) >= a.max_tensors:
                break
    w = np.concatenate(mats)
    w.tofile(a.output)
    meta = {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "tensors": names}
    a.output.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(meta["rows"], meta["cols"], len(names))


if __name__ == "__main__":
    main()
