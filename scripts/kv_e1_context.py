"""E1: beyond order-0 for real F16 KV. TRACE-like channel-major grouping + context-model entropies (CPU).

Input: one or more llama.cpp state files (F16 KV) -- native export (cq format) or server slot-save files
(the slot file header [magic, version, n_tokens, tokens] is skipped by searching for the KV layout).
Per layer and component (K, V), with W = (T tokens x C channels) uint16 F16 words:
 A) Context entropies (bits/element, exact empirical counts):
    H0(word)                    order-0 of the 16-bit word
    Hfields                     H(sign)+H(exp)+H(mant)
    H(exp|chan)                 exponent given channel
    H(exp|prev exp, same chan)  order-1 in time
    H(exp|prev exp, chan)       order-1 in time AND channel (1024 x 32 contexts; Miller-Madow corrected)
    H(mant_hi3|exp)             top 3 mantissa bits given exponent (mantissa structure check)
    best_bits = 1 + min(exp models) + H(mantissa | exp) estimate
 B) TRACE-like codec (bit-exact): channel-major, groups of G tokens per channel, exponent delta to group max,
    planes [sign bits][delta bits (5 planes)][mantissa bits (10 planes)] packed per group, concatenated and
    compressed with zstd-19 and lz4 in 4 KiB chunks; round trip checked. Plus zstd-19 on the plain channel-major
    bytes and on byte planes for reference.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import lz4.frame
import numpy as np
import zstandard as zstd


def H(c):
    c = c[c > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def Hcond(x, ctx, nx, nctx, mm=False):
    joint = np.bincount(ctx.astype(np.int64) * nx + x.astype(np.int64), minlength=nctx * nx).reshape(nctx, nx)
    tot = joint.sum()
    h = 0.0
    for row in joint:
        n = row.sum()
        if n:
            hr = H(row)
            if mm:
                hr += (np.count_nonzero(row) - 1) / (2 * n * np.log(2))
            h += n / tot * hr
    return h


def load_layers(path: Path):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    # find KV layout: [u32 v_trans=0][u32 n_layer][i32 dtype][u64 row_bytes] and T consistent with file size
    for off0 in range(0, min(len(raw), 1 << 22), 4):
        if off0 + 20 > len(raw):
            break
        vt, nl = struct.unpack_from("<II", raw, off0)
        if vt != 0 or not (8 <= nl <= 128):
            continue
        # newer llama-server builds write 8 extra bytes after n_layer; try both layouts
        for pad in (0, 8):
            if off0 + 20 + pad > len(raw):
                continue
            dt, rb = struct.unpack_from("<iQ", raw, off0 + 8 + pad)
            if dt not in (1, 30) or rb % 256 or not (512 <= rb <= 65536):
                continue
            payload = len(raw) - (off0 + 8 + pad)
            T = (payload // (2 * nl) - 12) // rb
            if T <= 0 or (12 + T * rb) * 2 * nl != payload:
                continue
            mats, off = [], off0 + 8 + pad
            ok = True
            for comp in ("K", "V"):
                for li in range(nl):
                    d2, r2 = struct.unpack_from("<iQ", raw, off)
                    if d2 != dt or r2 != rb:
                        ok = False
                        break
                    off += 12
                    mats.append((comp, li, raw[off:off + T * rb].view(np.uint16).reshape(T, rb // 2)))
                    off += T * rb
                if not ok:
                    break
            if ok and off == len(raw):
                return T, nl, mats, ("bf16" if dt == 30 else "f16")
    raise SystemExit(f"no F16 KV layout found in {path}")


def fields(W, fmt):
    if fmt == "bf16":
        return (W >> 15), (W >> 7) & 255, W & 127, 8, 7
    return (W >> 15), (W >> 10) & 31, W & 1023, 5, 10


def trace_codec(W, G, fmt="f16", level=19):
    T, C = W.shape
    EB, MB = (8, 7) if fmt == "bf16" else (5, 10)
    Tg = T - T % G
    x = W[:Tg].T.reshape(C, Tg // G, G)  # channel-major groups
    sign = (x >> 15).astype(np.uint8)
    exp = ((x >> MB) & ((1 << EB) - 1)).astype(np.int16)
    base = exp.max(-1, keepdims=True)
    delta = (base - exp).astype(np.uint8)
    mant = (x & ((1 << MB) - 1)).astype(np.uint16)
    planes = [np.packbits(sign, axis=-1)]
    planes += [np.packbits(((delta >> b) & 1).astype(np.uint8), axis=-1) for b in range(EB)]
    planes += [np.packbits(((mant >> b) & 1).astype(np.uint8), axis=-1) for b in range(MB)]
    stream = base.astype(np.uint8).tobytes() + b"".join(p.tobytes() for p in planes)
    # verify reconstruction from planes (bit-exact)
    s2 = np.unpackbits(planes[0], axis=-1)[..., :G].astype(np.uint16)
    d2 = sum(np.unpackbits(planes[1 + b], axis=-1)[..., :G].astype(np.uint16) << b for b in range(EB))
    m2 = sum(np.unpackbits(planes[1 + EB + b], axis=-1)[..., :G].astype(np.uint16) << b for b in range(MB))
    rec = (s2 << 15) | ((base.astype(np.uint16) - d2) << MB) | m2
    if not np.array_equal(rec, x):
        raise SystemExit("TRACE-like reconstruction mismatch")
    cz = zstd.ZstdCompressor(level=level)
    dz = zstd.ZstdDecompressor()
    zs = lz = 0
    for i in range(0, len(stream), 4096):
        ch = stream[i:i + 4096]
        z = cz.compress(ch)
        if dz.decompress(z) != ch:
            raise SystemExit("zstd mismatch")
        zs += min(len(z), len(ch))
        l = lz4.frame.compress(ch)
        lz += min(len(l), len(ch))
    raw_bytes = x.size * 2
    return zs / raw_bytes, lz / raw_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, action="append", required=True)
    ap.add_argument("--groups", default="16,64,256")
    ap.add_argument("--layer-stride", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("refusing to overwrite")
    groups = [int(g) for g in a.groups.split(",")]
    out = {}
    for sp in a.state:
        T, nl, mats, fmt = load_layers(sp)
        acc = {}
        for comp, li, W in mats:
            if li % a.layer_stride:
                continue
            Tn, C = W.shape
            s_, e_, m_, EB, MB = fields(W, fmt)
            sign, exp, mant = s_.astype(np.int64), e_.astype(np.int64), m_.astype(np.int64)
            NE, NM = 1 << EB, 1 << MB
            chan = np.broadcast_to(np.arange(C), W.shape)
            r = {"H0_word": H(np.bincount(W.ravel().astype(np.int64), minlength=65536)),
                 "H_sign": H(np.bincount(sign.ravel(), minlength=2)),
                 "H_exp": H(np.bincount(exp.ravel(), minlength=NE)),
                 "H_mant": H(np.bincount(mant.ravel(), minlength=NM)),
                 "H_exp|chan": Hcond(exp.ravel(), chan.ravel(), NE, C, mm=True),
                 "H_exp|prev": Hcond(exp[1:].ravel(), exp[:-1].ravel(), NE, NE),
                 "H_exp|prev,chan": Hcond(exp[1:].ravel(), (exp[:-1] * C + chan[1:]).ravel(), NE, NE * C, mm=True),
                 "H_sign|chan": Hcond(sign.ravel(), chan.ravel(), 2, C, mm=True),
                 "H_manthi3|exp": Hcond((mant >> (MB - 3)).ravel(), exp.ravel(), 8, NE)}
            r["H_mant_lo"] = H(np.bincount((mant & ((1 << (MB - 3)) - 1)).ravel(), minlength=1 << (MB - 3)))
            best_exp = min(r["H_exp|chan"], r["H_exp|prev"], r["H_exp|prev,chan"])
            r["best_bits"] = min(r["H_sign"], r["H_sign|chan"]) + best_exp + r["H_manthi3|exp"] + r["H_mant_lo"]
            for G in groups:
                z, l = trace_codec(W, G, fmt)
                r[f"trace_zstd19_G{G}"] = z
                r[f"trace_lz4_G{G}"] = l
            cz = zstd.ZstdCompressor(level=19)
            cm = W.T.copy().tobytes()
            r["zstd19_channel_major_4k"] = sum(min(len(cz.compress(cm[i:i + 4096])), 4096) for i in range(0, len(cm), 4096)) / len(cm)
            a_ = acc.setdefault(comp, {})
            for k, v in r.items():
                a_.setdefault(k, []).append(v)
        res = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in acc.items()}
        for c in res:
            res[c]["best_ratio_vs_f16"] = res[c]["best_bits"] / 16
        out[sp.name] = {"format": fmt, "tokens": T, "layers": nl, "layer_stride": a.layer_stride, "results": res}
        print(sp.name, T, json.dumps({c: {k: round(v, 4) for k, v in d.items()} for c, d in res.items()}, indent=1), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
