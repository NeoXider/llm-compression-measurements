"""Build the fixed local evaluation corpus (no download).

Takes whole lines from the start of each source until ~max_bytes, concatenates in the given order,
and writes corpus.txt plus a manifest with source SHA256s and the corpus SHA256. Sources are local
texts written in Aug–Oct 2026, so they are unlikely to be in the evaluated models' training data;
the comparison is relative to the Q8_0 reference anyway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, type=Path)
    ap.add_argument("--max-bytes", type=int, default=48000, help="per source, cut at line boundary")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise SystemExit("refusing to overwrite")
    parts, manifest = [], []
    for src in args.source:
        data = src.read_bytes()
        text = data.decode("utf-8")
        taken, size = [], 0
        for line in text.splitlines(keepends=True):
            n = len(line.encode("utf-8"))
            if size + n > args.max_bytes:
                break
            taken.append(line)
            size += n
        part = "".join(taken)
        parts.append(part)
        manifest.append({"path": str(src), "source_sha256": hashlib.sha256(data).hexdigest(),
                         "source_bytes": len(data), "taken_bytes": size})
    corpus = "\n\n".join(parts).encode("utf-8")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "corpus.txt").write_bytes(corpus)
    result = {"sources": manifest, "corpus_bytes": len(corpus), "corpus_sha256": hashlib.sha256(corpus).hexdigest()}
    (args.out_dir / "manifest.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
