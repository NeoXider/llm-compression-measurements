"""Lossless compression ON TOP OF a weight baseline (BF16, Q8_0, Q4_K_M, Q3_K_M GGUF files). CPU, bit-exact.

Per tensor, codecs (frames are independent = random access per frame; every frame round-trip checked):
  native_zlib64k     zlib-6 on the native bytes in 64 KiB frames
  split_zlib64k      header plane (d/dmin/scales) and code plane separated, each zlib-6 in 64 KiB frames
  split_lzma256k     same with LZMA-6 (raw) in 256 KiB frames (skipped for BF16: too slow on CPU)
  huffman            exact size of static per-tensor Huffman codes: BF16 = exponent byte Huffman + sign
                     and mantissa raw (DFloat11-style); quantized = byte-level Huffman on the code plane
                     (nibble-level for Q4_K) + header plane raw; +table cost. Size only, not packed.
Other tensors (F32 norms etc.) are counted raw in every codec.
"""
from __future__ import annotations

import argparse
import heapq
import json
import lzma
import sys
import zlib
from pathlib import Path

import numpy as np

LZ = [{"id": lzma.FILTER_LZMA2, "preset": 6}]
# (block bytes, header slices, code slices)
LAYOUT = {"Q8_0": (34, [(0, 2)], [(2, 34)]), "Q4_K": (144, [(0, 16)], [(16, 144)]), "Q5_K": (176, [(0, 16)], [(16, 176)]),
          "Q6_K": (210, [(192, 210)], [(0, 192)]), "Q3_K": (110, [(96, 110)], [(0, 96)]), "Q2_K": (84, [(0, 16), (80, 84)], [(16, 80)])}


def huff_bits(counts):
    c = [int(x) for x in counts if x > 0]
    if len(c) <= 1:
        return sum(c)
    h = list(c)
    heapq.heapify(h)
    total = 0
    while len(h) > 1:
        a, b = heapq.heappop(h), heapq.heappop(h)
        total += a + b
        heapq.heappush(h, a + b)
    return total  # sum over symbols of count * code length


def frames(data: bytes, size, enc, dec):
    tot = 0
    for i in range(0, len(data), size):
        fr = data[i:i + size]
        e = enc(fr)
        if dec(e) != fr:
            raise SystemExit("frame round-trip mismatch")
        tot += min(len(e), len(fr)) + 8
    return tot


ZE, ZD = (lambda b: zlib.compress(b, 6)), zlib.decompress
LE, LD = (lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=LZ)), (lambda b: lzma.decompress(b, format=lzma.FORMAT_RAW, filters=LZ))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, action="append", required=True)
    ap.add_argument("--gguf-root", type=Path, default=Path("<llama.cpp-checkout>/gguf-py"))
    ap.add_argument("--no-lzma", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("refusing to overwrite")
    sys.path.insert(0, str(a.gguf_root))
    from gguf import GGUFReader
    out = {}
    for m in a.model:
        r = GGUFReader(str(m), "r")
        tot = {"base": 0, "native_zlib64k": 0, "split_zlib64k": 0, "split_lzma256k": 0, "huffman": 0}
        for t in r.tensors:
            q = t.tensor_type.name
            raw = np.asarray(t.data).view(np.uint8).reshape(-1)
            n = raw.size
            tot["base"] += n
            if q not in LAYOUT and q != "BF16":
                for k in tot:
                    if k != "base":
                        tot[k] += n
                continue
            tot["native_zlib64k"] += frames(raw.tobytes(), 65536, ZE, ZD)
            if q == "BF16":
                w = raw.view(np.uint16)
                hi, lo = (w >> 8).astype(np.uint8), (w & 255).astype(np.uint8)
                tot["split_zlib64k"] += frames(hi.tobytes(), 65536, ZE, ZD) + frames(lo.tobytes(), 65536, ZE, ZD)
                tot["split_lzma256k"] += n  # not run for BF16
                exp = ((w >> 7) & 0xFF).astype(np.int64)
                tot["huffman"] += huff_bits(np.bincount(exp, minlength=256)) / 8 + w.size * 8 / 8 + 256 * 2
                continue
            bs, hs, cs = LAYOUT[q]
            b = raw.reshape(-1, bs)
            head = np.concatenate([b[:, s:e] for s, e in hs], axis=1).tobytes()
            code = np.concatenate([b[:, s:e] for s, e in cs], axis=1)
            cb = code.tobytes()
            tot["split_zlib64k"] += frames(head, 65536, ZE, ZD) + frames(cb, 65536, ZE, ZD)
            if a.no_lzma:
                tot["split_lzma256k"] += n
            else:
                tot["split_lzma256k"] += frames(head, 262144, LE, LD) + frames(cb, 262144, LE, LD)
            if q == "Q4_K":
                nib = np.concatenate([(code & 15).ravel(), (code >> 4).ravel()]).astype(np.int64)
                hb = huff_bits(np.bincount(nib, minlength=16)) / 8 + 16
            else:
                hb = huff_bits(np.bincount(code.ravel().astype(np.int64), minlength=256)) / 8 + 256
            tot["huffman"] += hb + len(head)
        res = {k: v for k, v in tot.items()}
        res["ratio_vs_base"] = {k: v / tot["base"] for k, v in tot.items() if k != "base"}
        out[m.name] = res
        print(m.name, json.dumps({k: round(v, 4) for k, v in res["ratio_vs_base"].items()}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
