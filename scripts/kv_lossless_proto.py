"""Bit-exact CPU prototype of lossless F16 KV page compression (read-only input).

Pages = P consecutive tokens x one layer's full K or V row (8 heads x 128 dims, F16).
Codecs (all from the Python stdlib, each page compressed independently = random access per page):
  raw            no compression (reference)
  zlib_plain     zlib level 6 on the page bytes as stored
  zlib_planes    byte-plane split (all high bytes, then all low bytes), zlib level 6
  lzma_planes    byte-plane split, LZMA preset 6 (raw format, no container header)
  zlib_exp       sign+exponent (6 bits, one byte each) zlib'ed, 10-bit mantissas stored packed raw
Each page is decoded and compared byte-for-byte with the original; any mismatch aborts.
Also reports per-field order-0 entropies (sign, exponent, mantissa) so the gap to PAL3 can be explained.
CPU timings are single-thread Python+stdlib diagnostics, not GPU kernel costs.
"""
from __future__ import annotations

import argparse
import json
import lzma
import struct
import time
import zlib
from pathlib import Path

import numpy as np

LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 6}]


def H(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def planes(b: bytes) -> bytes:
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, 2)
    return a[:, 1].tobytes() + a[:, 0].tobytes()


def unplanes(b: bytes) -> bytes:
    n = len(b) // 2
    out = np.empty((n, 2), dtype=np.uint8)
    out[:, 1] = np.frombuffer(b[:n], dtype=np.uint8)
    out[:, 0] = np.frombuffer(b[n:], dtype=np.uint8)
    return out.tobytes()


def pack10(m: np.ndarray) -> bytes:
    m = m.astype(np.uint64).reshape(-1, 4)  # 4 x 10 bits = 40 bits = 5 bytes
    v = m[:, 0] | (m[:, 1] << 10) | (m[:, 2] << 20) | (m[:, 3] << 30)
    return np.stack([(v >> (8 * i)) & 0xFF for i in range(5)], axis=1).astype(np.uint8).tobytes()


def unpack10(b: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, 5).astype(np.uint64)
    v = sum(a[:, i] << (8 * i) for i in range(5))
    return np.stack([(v >> (10 * i)) & 0x3FF for i in range(4)], axis=1).reshape(-1)[:n].astype(np.uint16)


def enc_exp(b: bytes):
    w = np.frombuffer(b, dtype=np.uint16)
    se = (w >> 10).astype(np.uint8)
    return zlib.compress(se.tobytes(), 6), pack10(w & 0x3FF)


def dec_exp(parts, n):
    se = np.frombuffer(zlib.decompress(parts[0]), dtype=np.uint8).astype(np.uint16)
    return ((se << 10) | unpack10(parts[1], n)).astype(np.uint16).tobytes()


CODECS = {
    "zlib_plain": (lambda b: [zlib.compress(b, 6)], lambda p, n: zlib.decompress(p[0])),
    "zlib_planes": (lambda b: [zlib.compress(planes(b), 6)], lambda p, n: unplanes(zlib.decompress(p[0]))),
    "lzma_planes": (lambda b: [lzma.compress(planes(b), format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)],
                    lambda p, n: unplanes(lzma.decompress(p[0], format=lzma.FORMAT_RAW, filters=LZMA_FILTERS))),
    "zlib_exp": (lambda b: list(enc_exp(b)), lambda p, n: dec_exp(p, n)),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--page-tokens", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--layers", default="all", help="'all' or comma list (applied to K and V)")
    ap.add_argument("--index-bytes-per-page", type=int, default=8, help="offset+length charge per page")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite")
    raw = np.memmap(args.state, dtype=np.uint8, mode="r")
    _, _, _, T = struct.unpack_from("<IiII", raw, 0)
    off = 16 + 12 * T
    _, L = struct.unpack_from("<II", raw, off)
    off += 8
    mats = []
    for comp in ("K", "V"):
        for li in range(L):
            _, rb = struct.unpack_from("<iQ", raw, off)
            off += 12
            mats.append((comp, li, np.frombuffer(raw[off:off + T * rb], dtype=np.uint16).reshape(T, rb // 2)))
            off += T * rb
    wanted = None if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    mats = [m for m in mats if wanted is None or m[1] in wanted]
    fields = {}
    for comp in ("K", "V"):
        w = np.concatenate([m[2].ravel() for m in mats if m[0] == comp])
        fields[comp] = {"sign_bits": H(np.bincount((w >> 15).astype(np.int64), minlength=2)),
                        "exponent_bits": H(np.bincount(((w >> 10) & 31).astype(np.int64), minlength=32)),
                        "mantissa_bits": H(np.bincount((w & 1023).astype(np.int64), minlength=1024))}
    results = {}
    for P in args.page_tokens:
        for name, (enc, dec) in CODECS.items():
            tot = {"K": [0, 0, 0], "V": [0, 0, 0]}  # raw bytes, compressed bytes incl index, pages
            t_enc = t_dec = 0.0
            for comp, li, m in mats:
                for t0 in range(0, T, P):
                    page = np.ascontiguousarray(m[t0:t0 + P]).tobytes()
                    a = time.perf_counter()
                    parts = enc(page)
                    b = time.perf_counter()
                    back = dec(parts, len(page) // 2)
                    c = time.perf_counter()
                    if back != page:
                        raise SystemExit(f"ROUNDTRIP MISMATCH {name} P={P} {comp}{li} t0={t0}")
                    t_enc += b - a
                    t_dec += c - b
                    size = sum(len(x) for x in parts) + args.index_bytes_per_page * len(parts)
                    size = min(size, len(page) + args.index_bytes_per_page)  # raw fallback per page
                    tot[comp][0] += len(page)
                    tot[comp][1] += size
                    tot[comp][2] += 1
            r = {c: {"raw_bytes": v[0], "coded_bytes": v[1], "ratio": v[1] / v[0], "pages": v[2]}
                 for c, v in tot.items()}
            allraw = tot["K"][0] + tot["V"][0]
            r["all"] = {"raw_bytes": allraw, "coded_bytes": tot["K"][1] + tot["V"][1],
                        "ratio": (tot["K"][1] + tot["V"][1]) / allraw,
                        "cpu_encode_MBps": allraw / t_enc / 1e6, "cpu_decode_MBps": allraw / t_dec / 1e6}
            results[f"P{P}/{name}"] = r
            print(f"P={P:4d} {name:12s} K {r['K']['ratio']:.4f} V {r['V']['ratio']:.4f} all {r['all']['ratio']:.4f} "
                  f"enc {r['all']['cpu_encode_MBps']:.1f} MB/s dec {r['all']['cpu_decode_MBps']:.1f} MB/s", flush=True)
    out = {"state": str(args.state), "tokens": T, "layers": args.layers, "field_entropy_bits": fields,
           "roundtrip": "every page byte-exact", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(fields, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
