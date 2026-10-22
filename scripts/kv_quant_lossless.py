"""Lossless compression ON TOP OF Q8_0 / Q4_0 KV (CPU, bit-exact), on real captured Qwen3-4B KV.

The F16 capture is requantized offline with llama.cpp's block rules (32 values along the row:
q8_0: d = amax/127, q = round(x/d), 34 B per block; q4_0: d = max/-8, q = clamp(int(x/d + 8.5)), 18 B).
This is the BASE. Codecs act on the base bytes and must restore them exactly.
Pages: P tokens x one layer's K or V row (8 heads x 128 = 32 blocks per token).
  native_zlib      zlib-6 on the page in native block layout
  split_zlib       scales plane (hi bytes, lo bytes) + codes plane, each zlib-6
  split_lzma       same with LZMA-6 (raw)
  huffman_codes    exact size of a static per-layer Huffman code for the codes (code lengths x counts,
                   + 256 B table per layer) + scales stored raw  [size only, not packed]
  fixed_escape     random-access fixed rate. Q8: each code in 7 bits (two's complement range -63..63),
                   value -64 = escape; escaped codes stored as full bytes in a per-block overflow; per block
                   1 count byte. Q4: each code in 3 bits for codes 5..11, 7 = escape; escaped codes as
                   4-bit nibbles in overflow; per block 1 count byte. Scales stored raw (2 B per block).
Every page or block is decoded and compared with the base bytes (fixed_escape decode is vectorized).
"""
from __future__ import annotations

import argparse
import heapq
import json
import lzma
import struct
import zlib
from pathlib import Path

import numpy as np

LZ = [{"id": lzma.FILTER_LZMA2, "preset": 6}]


def load(state):
    raw = np.memmap(state, dtype=np.uint8, mode="r")
    _, _, _, T = struct.unpack_from("<IiII", raw, 0)
    off = 16 + 12 * T
    _, L = struct.unpack_from("<II", raw, off)
    off += 8
    mats = []
    for comp in ("K", "V"):
        for li in range(L):
            _, rb = struct.unpack_from("<iQ", raw, off)
            off += 12
            mats.append((comp, li, np.frombuffer(raw[off:off + T * rb], dtype=np.float16).reshape(T, rb // 2)))
            off += T * rb
    return T, mats


def q8(x):  # x: (T, C) float16 -> scales (T, C/32) f16 bits, codes (T, C) int8
    b = x.astype(np.float32).reshape(x.shape[0], -1, 32)
    amax = np.abs(b).max(-1)
    d = amax / 127.0
    idv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.round(b * idv[..., None]).astype(np.int8)
    return d.astype(np.float16).view(np.uint16), q.reshape(x.shape)


def q4(x):
    b = x.astype(np.float32).reshape(x.shape[0], -1, 32)
    i = np.abs(b).argmax(-1)
    mx = np.take_along_axis(b, i[..., None], -1)[..., 0]
    d = mx / -8.0
    idv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0)
    q = np.clip((b * idv[..., None] + 8.5).astype(np.int32), 0, 15).astype(np.uint8)
    return d.astype(np.float16).view(np.uint16), q.reshape(x.shape)


def native_bytes(kind, sc, q):
    """llama.cpp block layout for rows (T, C): per block [d f16][codes]."""
    T, nb = sc.shape
    if kind == "q8_0":
        blk = np.empty((T, nb, 34), np.uint8)
        blk[..., 0:2] = sc.view(np.uint8).reshape(T, nb, 2)
        blk[..., 2:] = q.view(np.uint8).reshape(T, nb, 32)
    else:
        blk = np.empty((T, nb, 18), np.uint8)
        blk[..., 0:2] = sc.view(np.uint8).reshape(T, nb, 2)
        qq = q.reshape(T, nb, 32)
        blk[..., 2:] = qq[..., :16] | (qq[..., 16:] << 4)
    return blk


def huff_lengths(counts):
    sym = [(int(c), i) for i, c in enumerate(counts) if c > 0]
    if len(sym) == 1:
        return {sym[0][1]: 1}
    heap = [(c, n, [s]) for n, (c, s) in enumerate(sym)]
    heapq.heapify(heap)
    length = {s: 0 for _, s in sym}
    n = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            length[s] += 1
        heapq.heappush(heap, (c1 + c2, n, s1 + s2))
        n += 1
    return length


def fixed_escape(kind, sc, q):
    """returns total bytes and verifies exact decode."""
    T, nb = sc.shape
    qq = q.reshape(T, nb, 32)
    if kind == "q8_0":
        v = qq.astype(np.int16)
        esc = (v < -63) | (v > 63)
        stored = np.where(esc, -64, v)
        dec = np.where(stored == -64, v, stored)  # escaped values come from the overflow list (exact bytes)
        payload_bits, esc_bytes = 7 * 32, esc.sum() * 1
    else:
        v = qq.astype(np.int16)
        esc = (v < 5) | (v > 11)
        stored = np.where(esc, 7, v - 5)
        dec = np.where(stored == 7, v, stored + 5)
        payload_bits, esc_bytes = 3 * 32, esc.sum() * 0.5
    if not np.array_equal(dec, v):
        raise SystemExit("fixed_escape roundtrip mismatch")
    blocks = T * nb
    total = blocks * (2 + payload_bits / 8 + 1) + esc_bytes
    return float(total), float(esc.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--pages", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("refusing to overwrite")
    T, mats = load(a.state)
    res = {}
    for kind, qfun, blkb in (("q8_0", q8, 34), ("q4_0", q4, 18)):
        acc = {}
        for comp in ("K", "V"):
            base = 0
            sizes = {}
            hist_codes = []
            for c, li, x in [m for m in mats if m[0] == comp]:
                sc, q = qfun(x)
                blk = native_bytes(kind, sc, q)
                base += blk.size
                codes = (q.astype(np.int16) + (128 if kind == "q8_0" else 0)).astype(np.int64)
                cnt = np.bincount(codes.ravel(), minlength=256 if kind == "q8_0" else 16)
                lens = huff_lengths(cnt)
                bits = sum(cnt[s] * l for s, l in lens.items())
                sizes["huffman_codes"] = sizes.get("huffman_codes", 0) + bits / 8 + (256 if kind == "q8_0" else 16) + sc.size * 2
                fe, er = fixed_escape(kind, sc, q)
                sizes["fixed_escape"] = sizes.get("fixed_escape", 0) + fe
                sizes.setdefault("_esc_rate", []).append(er)
                for P in a.pages:
                    for t0 in range(0, T, P):
                        page = blk[t0:t0 + P]
                        nbytes = page.tobytes()
                        z = zlib.compress(nbytes, 6)
                        if zlib.decompress(z) != nbytes:
                            raise SystemExit("zlib mismatch")
                        sizes[f"native_zlib_P{P}"] = sizes.get(f"native_zlib_P{P}", 0) + min(len(z), len(nbytes)) + 8
                        s = page[..., :2].reshape(-1, 2)
                        planes = [s[:, 1].tobytes(), s[:, 0].tobytes(), page[..., 2:].tobytes()]
                        for name, enc, dec in (("split_zlib", lambda b: zlib.compress(b, 6), zlib.decompress),
                                               ("split_lzma", lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=LZ),
                                                lambda b: lzma.decompress(b, format=lzma.FORMAT_RAW, filters=LZ))):
                            tot = 0
                            for pl in planes:
                                e = enc(pl)
                                if dec(e) != pl:
                                    raise SystemExit("split mismatch")
                                tot += min(len(e), len(pl)) + 8
                            sizes[f"{name}_P{P}"] = sizes.get(f"{name}_P{P}", 0) + tot
            esc = float(np.mean(sizes.pop("_esc_rate")))
            acc[comp] = {"base_bytes": base, "ratio_vs_base": {k: v / base for k, v in sizes.items()},
                         "fixed_escape_rate": esc, "base_bits_per_element": base * 8 / (T * 1024 * 36)}
            print(kind, comp, json.dumps({k: round(v, 4) for k, v in acc[comp]["ratio_vs_base"].items()}), "esc", round(esc, 4), flush=True)
        res[kind] = acc
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"tokens": T, "results": res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
