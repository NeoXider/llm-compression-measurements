"""Context-model entropy of already-quantized llama.cpp GGUF weights (Q8_0, Q4_K, Q5_K, Q6_K). CPU only, read-only.

Question: how far below native size can LOSSLESS re-coding go when the coder may use context, compared with
order-0 (R16)? Every byte of a block is counted (codes + sub-block scales/mins + d/dmin).

Cost model (ideal static coding, per tensor):
  data bits  = sum over symbols of -log2 p(symbol | context), p from the tensor's own (context, symbol) histogram
  table bits = for every context that occurs in the tensor: min(A*16, nnz*(ceil(log2 A)+16)) where A = alphabet
               size and nnz = distinct symbols in that table (the min only matters for 16-bit fields, A=65536)
Sampling: rows r = 1, 1+S, 1+2S, ... of every tensor (S = --stride, default 4); the column context uses row r-1,
which is always the true previous row. Data bits are scaled by nrows / sampled_rows; table bits are not scaled
(tables are counted from the sample). Bucket edges (quantiles) are taken from the first row-chunk of each tensor.

Code schemes (codes only; the header fields are then added at their order-0 cost for the "block" ratio):
  o0            order-0
  pos           position class: Q8_0 = element index mod 4; K-quants = sub-block index in the block
  scale_raw     K-quants: raw sub-block scale (Q4_K/Q5_K 6-bit sc, Q6_K int8 sc)
  scale_b8      sub-block scale bucketed into 8 quantiles (Q8_0: block d as float; K: raw sc value)
  effscale_b8   K-quants: effective scale d*sc as float, 8 quantiles
  zp            Q4_K/Q5_K: zero point dmin*m/(d*sc) in code units, rounded to 1/2 code (2A buckets)
  prev          order-1: previous code in the same row (sentinel at row start)
  col           code at the same column in the previous row (sentinel for row 0)
  prev_b16/col_b16  Q8_0: neighbour bucketed into 16 classes (sign x 8 |q| quantiles)
  combos        scale/zp context x prev / col
  magsign       Q8_0: |q| order-0 + sign entropy over nonzero codes
  maxpos(_sb8)  Q8_0: per-block flag "exactly one |q|=127"; if set, position (32-way) + sign of that code are coded
                explicitly and the code is removed from the stream; the rest is order-0 (or d-bucket context)
Header schemes per field: o0; delta (to previous block, same row, same sub-block slot; mod alphabet);
  ctxprev (conditioned on previous block value: 16-bit fields as H(hi|prev hi)+H(lo|hi); sub-scales directly);
  ctxprevsub (sub-scales: previous sub-block of the same block); ctxsc (Q4_K/Q5_K mins conditioned on scale).
GPU-practical: static per-tensor Huffman on the codes with lengths limited to 12 bits (package-merge), length
  table A*4 bits, plus a u16 bit-offset anchor every 64 or 256 elements; header fields raw. Variant huff12ctx uses
  one Huffman table per context (Q8_0 d bucket, Q4_K/Q5_K zp, Q6_K scale_b8).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

BLOCK = {"Q8_0": (34, 32), "Q4_K": (144, 256), "Q5_K": (176, 256), "Q6_K": (210, 256)}
ALPH = {"Q8_0": 256, "Q4_K": 16, "Q5_K": 32, "Q6_K": 64}
SUBSZ = {"Q8_0": 32, "Q4_K": 32, "Q5_K": 32, "Q6_K": 16}
NATIVE_CODE_BITS = {"Q8_0": 8, "Q4_K": 4, "Q5_K": 5, "Q6_K": 6}


# ----------------------------------------------------------------------------------------------------- counting
def counts_cost(m: np.ndarray, A: int):
    """m: (nctx, A) counts -> (data_bits, table_bits)."""
    m = m.astype(np.float64)
    rs = m.sum(1)
    used = rs > 0
    m, rs = m[used], rs[used]
    nz = m > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(nz, np.log2(np.where(nz, m, 1) / rs[:, None]), 0.0)
    data = float(-(m * lg).sum())
    nnz = nz.sum(1)
    tab = np.minimum(A * 16, nnz * (math.ceil(math.log2(A)) + 16)).sum()
    return data, float(tab)


class Acc:
    def __init__(self):
        self.m = {}

    def add(self, name, x, ctx, A, nctx):
        x = np.asarray(x).ravel().astype(np.int64)
        if ctx is None:
            idx, nctx = x, 1
        else:
            idx = np.asarray(ctx).ravel().astype(np.int64) * A + x
        c = np.bincount(idx, minlength=nctx * A).reshape(nctx, A)
        if name in self.m:
            self.m[name][0] += c
        else:
            self.m[name] = [c, A]


# ----------------------------------------------------------------------------------------------- Huffman (PM)
def huff_lengths(counts: np.ndarray, L: int = 12) -> np.ndarray:
    """Optimal length-limited prefix code lengths (package-merge)."""
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.nonzero(counts)[0]
    n = len(idx)
    out = np.zeros(len(counts), dtype=np.int64)
    if n == 0:
        return out
    if n == 1:
        out[idx] = 1
        return out
    assert n <= (1 << L)
    w = counts[idx]
    order = np.argsort(w, kind="stable")
    eye = np.eye(n, dtype=np.int32)
    leaves = [(int(w[i]), eye[i]) for i in order]
    prev = []
    merged = leaves
    for _ in range(L):
        merged = sorted(leaves + prev, key=lambda t: t[0])
        prev = [(merged[k][0] + merged[k + 1][0], merged[k][1] + merged[k + 1][1]) for k in range(0, len(merged) - 1, 2)]
    lens = np.sum([v for _, v in merged[:2 * n - 2]], axis=0)
    out[idx] = lens
    assert (2.0 ** -out[idx]).sum() <= 1 + 1e-12 and out.max() <= L
    return out


# ------------------------------------------------------------------------------------------------- decoding
def f16(u):
    return u.view(np.float16).astype(np.float32)


def decode(qt: str, b: np.ndarray):
    """b: (R, nb, bs) uint8 -> codes (R, ne0) int64 symbols, sub-level dict, header field dict."""
    R, nb, _ = b.shape
    hdr = {}
    sub = {}
    if qt == "Q8_0":
        d = np.ascontiguousarray(b[..., 0:2]).view(np.uint16)[..., 0]
        q = np.ascontiguousarray(b[..., 2:34]).view(np.int8).astype(np.int64)
        codes = (q + 128).reshape(R, nb * 32)
        hdr["d"] = ("u16", d)
        sub["dval"] = np.abs(f16(d))
        return codes, sub, hdr, q
    if qt in ("Q4_K", "Q5_K"):
        d = np.ascontiguousarray(b[..., 0:2]).view(np.uint16)[..., 0]
        dmin = np.ascontiguousarray(b[..., 2:4]).view(np.uint16)[..., 0]
        s = b[..., 4:16].astype(np.int64)
        sc = np.concatenate([s[..., 0:4] & 63, (s[..., 8:12] & 15) | ((s[..., 0:4] >> 6) << 4)], -1)
        mn = np.concatenate([s[..., 4:8] & 63, (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)], -1)
        if qt == "Q4_K":
            qs = b[..., 16:144].reshape(R, nb, 4, 32).astype(np.int64)
            codes = np.stack([qs & 15, qs >> 4], axis=3)
        else:
            qh = b[..., 16:48].astype(np.int64)[:, :, None, :]
            qs = b[..., 48:176].reshape(R, nb, 4, 32).astype(np.int64)
            sh = (2 * np.arange(4))[None, None, :, None]
            codes = np.stack([(qs & 15) | (((qh >> sh) & 1) << 4), (qs >> 4) | (((qh >> (sh + 1)) & 1) << 4)], axis=3)
        codes = codes.reshape(R, nb * 256)
        hdr.update(d=("u16", d), dmin=("u16", dmin), sc=("sub", sc, 64), m=("sub", mn, 64))
        dv, dmv = f16(d), f16(dmin)
        eff = dv[..., None] * sc
        sub["sc"] = sc
        sub["eff"] = np.abs(eff)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(eff > 0, dmv[..., None] * mn / np.where(eff > 0, eff, 1), np.inf)
        A = ALPH[qt]
        sub["zp"] = np.clip(np.nan_to_num(np.round(2 * z), posinf=2 * A - 1, neginf=0), 0, 2 * A - 1).astype(np.int64)
        return codes, sub, hdr, None
    if qt == "Q6_K":
        ql = b[..., 0:128].reshape(R, nb, 2, 64).astype(np.int64)
        qh = b[..., 128:192].reshape(R, nb, 2, 32).astype(np.int64)
        lo, hi = ql[..., :32], ql[..., 32:]
        codes = np.stack([(lo & 15) | ((qh & 3) << 4), (hi & 15) | (((qh >> 2) & 3) << 4),
                          (lo >> 4) | (((qh >> 4) & 3) << 4), (hi >> 4) | (((qh >> 6) & 3) << 4)], axis=3)
        codes = codes.reshape(R, nb * 256)
        sc8 = np.ascontiguousarray(b[..., 192:208]).view(np.int8).astype(np.int64)
        d = np.ascontiguousarray(b[..., 208:210]).view(np.uint16)[..., 0]
        hdr.update(d=("u16", d), sc=("sub", sc8 + 128, 256))
        sub["sc"] = sc8 + 128
        sub["eff"] = np.abs(f16(d)[..., None] * sc8)
        return codes, sub, hdr, None
    raise ValueError(qt)


def qbucket(v, edges):
    return np.searchsorted(edges, v, side="right").astype(np.int64)


def qedges(v, k=8):
    return np.unique(np.quantile(np.asarray(v, dtype=np.float64).ravel(), np.arange(1, k) / k))


def rep(sublevel, R, sb):
    """(R, nb, S) -> (R, nb*S*sb) element-level."""
    return np.repeat(sublevel.reshape(R, -1), sb, axis=1)


# ------------------------------------------------------------------------------------------- per-chunk model
def process_chunk(qt, cur, prv, has_prev, acc, st):
    R = cur.shape[0]
    A = ALPH[qt]
    sb = SUBSZ[qt]
    codes, sub, hdr, q = decode(qt, cur)
    pcodes = decode(qt, prv)[0]
    ne0 = codes.shape[1]
    # previous code in row / previous row
    prevc = np.empty_like(codes)
    prevc[:, 0] = A
    prevc[:, 1:] = codes[:, :-1]
    colc = np.where(has_prev[:, None], pcodes, A)
    if "edges" not in st:
        if qt == "Q8_0":
            st["edges"] = {"scale": qedges(sub["dval"]), "mag": qedges(np.abs(q))}
        else:
            st["edges"] = {"scale": qedges(sub["sc"]), "eff": qedges(sub["eff"])}
    E = st["edges"]
    acc.add("c:o0", codes, None, A, 1)
    if qt == "Q8_0":
        pos = np.broadcast_to(np.arange(ne0) % 4, codes.shape)
        sbk = rep(qbucket(sub["dval"], E["scale"])[..., None], R, 32)
        nsb = len(E["scale"]) + 1

        def b16(c):
            s = c - 128
            return np.where(c == A, 16, qbucket(np.abs(s), E["mag"]) + 8 * (s < 0))
        pb, cb = b16(prevc), b16(colc)
        acc.add("c:pos", codes, pos, A, 4)
        acc.add("c:scale_b8", codes, sbk, A, nsb)
        acc.add("c:prev", codes, prevc, A, A + 1)
        acc.add("c:col", codes, colc, A, A + 1)
        acc.add("c:prev_b16", codes, pb, A, 17)
        acc.add("c:col_b16", codes, cb, A, 17)
        acc.add("c:scale_b8xprev_b16", codes, sbk * 17 + pb, A, nsb * 17)
        acc.add("c:scale_b8xcol_b16", codes, sbk * 17 + cb, A, nsb * 17)
        mag = np.abs(q)
        acc.add("c:mag", mag, None, 129, 1)
        acc.add("c:sign", (q[mag > 0] < 0), None, 2, 1)
        is127 = mag == 127
        k = is127.sum(-1)
        single = k == 1
        st.setdefault("q8stats", np.zeros(4, np.int64))
        st["q8stats"] += np.bincount(np.minimum(k, 3).ravel(), minlength=4)
        st.setdefault("q8max127", [0, 0])
        st["q8max127"][0] += int((mag.max(-1) == 127).sum())
        st["q8max127"][1] += int(k.size)
        acc.add("c:maxflag", single, None, 2, 1)
        acc.add("c:maxpos_pos", np.argmax(is127, -1)[single], None, 32, 1)
        sidx = np.nonzero(single)
        acc.add("c:maxpos_sign", q[sidx[0], sidx[1], np.argmax(is127, -1)[single]] < 0, None, 2, 1)
        keep = ~(single[..., None] & is127).reshape(R, ne0)
        acc.add("c:maxpos_rest", codes[keep], None, A, 1)
        acc.add("c:maxpos_rest_sb8", codes[keep], sbk[keep], A, nsb)
    else:
        S = 256 // sb
        sidx = np.broadcast_to(np.repeat(np.arange(S), sb)[None, :].repeat(ne0 // 256, 0).ravel(), codes.shape)
        ns = 64 if qt != "Q6_K" else 256
        scr = rep(sub["sc"], R, sb)
        sbk = rep(qbucket(sub["sc"], E["scale"]), R, sb)
        nsb = len(E["scale"]) + 1
        ebk = rep(qbucket(sub["eff"], E["eff"]), R, sb)
        neb = len(E["eff"]) + 1
        acc.add("c:pos", codes, sidx, A, S)
        acc.add("c:scale_raw", codes, scr, A, ns)
        acc.add("c:scale_b8", codes, sbk, A, nsb)
        acc.add("c:effscale_b8", codes, ebk, A, neb)
        acc.add("c:prev", codes, prevc, A, A + 1)
        acc.add("c:col", codes, colc, A, A + 1)
        acc.add("c:effscale_b8xprev", codes, ebk * (A + 1) + prevc, A, neb * (A + 1))
        acc.add("c:effscale_b8xcol", codes, ebk * (A + 1) + colc, A, neb * (A + 1))
        if qt != "Q6_K":
            zp = rep(sub["zp"], R, sb)
            nz = 2 * A
            acc.add("c:zp", codes, zp, A, nz)
            acc.add("c:zpxprev", codes, zp * (A + 1) + prevc, A, nz * (A + 1))
            acc.add("c:zpxcol", codes, zp * (A + 1) + colc, A, nz * (A + 1))
            acc.add("c:zpxeffscale_b8", codes, zp * neb + ebk, A, nz * neb)
        else:
            acc.add("c:scale_rawxprev_b", codes, scr * 9 + np.where(prevc == A, 8, prevc // 8), A, ns * 9)
    # ---- header fields
    for name, spec in hdr.items():
        if spec[0] == "u16":
            v = spec[1].astype(np.int64)  # (R, nb)
            pv = np.concatenate([np.zeros((R, 1), np.int64), v[:, :-1]], 1)
            acc.add(f"h:{name}:o0", v, None, 65536, 1)
            acc.add(f"h:{name}:delta", (v - pv) & 0xFFFF, None, 65536, 1)
            ph = np.concatenate([np.full((R, 1), 256, np.int64), v[:, :-1] >> 8], 1)
            acc.add(f"h:{name}:ctxprev_hi", v >> 8, ph, 256, 257)
            acc.add(f"h:{name}:ctxprev_lo", v & 255, v >> 8, 256, 256)
            acc.add(f"h:{name}:hilo_hi", v >> 8, None, 256, 1)
        else:
            v, Af = spec[1], spec[2]  # (R, nb, S)
            Sv = v.shape[2]
            pb_ = np.concatenate([np.full((R, 1, Sv), Af, np.int64), v[:, :-1]], 1)
            acc.add(f"h:{name}:o0", v, None, Af, 1)
            acc.add(f"h:{name}:delta", (v - np.where(pb_ == Af, 0, pb_)) % Af, None, Af, 1)
            acc.add(f"h:{name}:ctxprev", v, pb_, Af, Af + 1)
            ps = np.concatenate([np.full((R, nb_(v), 1), Af, np.int64), v[:, :, :-1]], 2)
            acc.add(f"h:{name}:ctxprevsub", v, ps, Af, Af + 1)
            acc.add(f"h:{name}:ctxsubidx", v, np.broadcast_to(np.arange(Sv), v.shape), Af, Sv)
            if name == "m":
                acc.add("h:m:ctxsc", v, hdr["sc"][1], Af, 64)
            if name == "sc" and qt == "Q6_K":
                acc.add("h:sc:ctxprevblockxprevsub_b", v, np.minimum(pb_, Af) // 16 * 17 + np.minimum(ps, Af) // 16, Af, 17 * 17)


def nb_(v):
    return v.shape[1]


CODE_SCHEMES = {
    "Q8_0": {"o0": ["c:o0"], "pos": ["c:pos"], "scale_b8": ["c:scale_b8"], "prev": ["c:prev"], "col": ["c:col"],
             "prev_b16": ["c:prev_b16"], "col_b16": ["c:col_b16"], "scale_b8xprev_b16": ["c:scale_b8xprev_b16"],
             "scale_b8xcol_b16": ["c:scale_b8xcol_b16"], "magsign": ["c:mag", "c:sign"],
             "maxpos": ["c:maxflag", "c:maxpos_pos", "c:maxpos_sign", "c:maxpos_rest"],
             "maxpos_sb8": ["c:maxflag", "c:maxpos_pos", "c:maxpos_sign", "c:maxpos_rest_sb8"]},
    "K": {"o0": ["c:o0"], "pos": ["c:pos"], "scale_raw": ["c:scale_raw"], "scale_b8": ["c:scale_b8"],
          "effscale_b8": ["c:effscale_b8"], "prev": ["c:prev"], "col": ["c:col"],
          "effscale_b8xprev": ["c:effscale_b8xprev"], "effscale_b8xcol": ["c:effscale_b8xcol"]},
    "Q45": {"zp": ["c:zp"], "zpxprev": ["c:zpxprev"], "zpxcol": ["c:zpxcol"], "zpxeffscale_b8": ["c:zpxeffscale_b8"]},
    "Q6": {"scale_rawxprev_b": ["c:scale_rawxprev_b"]},
}
HDR_SCHEMES_U16 = {"o0": ["o0"], "delta": ["delta"], "ctxprev": ["ctxprev_hi", "ctxprev_lo"]}
HDR_SCHEMES_SUB = {"o0": ["o0"], "delta": ["delta"], "ctxprev": ["ctxprev"], "ctxprevsub": ["ctxprevsub"],
                   "ctxsubidx": ["ctxsubidx"]}
HUFF_CTX = {"Q8_0": "c:scale_b8", "Q4_K": "c:zp", "Q5_K": "c:zp", "Q6_K": "c:scale_b8"}


def code_schemes(qt):
    s = dict(CODE_SCHEMES["Q8_0"] if qt == "Q8_0" else CODE_SCHEMES["K"])
    if qt in ("Q4_K", "Q5_K"):
        s.update(CODE_SCHEMES["Q45"])
    if qt == "Q6_K":
        s.update(CODE_SCHEMES["Q6"])
    return s


def hdr_fields(qt):
    return {"Q8_0": {"d": "u16"}, "Q4_K": {"d": "u16", "dmin": "u16", "sc": "sub", "m": "sub"},
            "Q5_K": {"d": "u16", "dmin": "u16", "sc": "sub", "m": "sub"}, "Q6_K": {"d": "u16", "sc": "sub"}}[qt]


# --------------------------------------------------------------------------------------------- per tensor
def analyse_tensor(qt, data2d, stride, chunk_elems):
    nrows, rowbytes = data2d.shape
    bs, QK = BLOCK[qt]
    nb = rowbytes // bs
    ne0 = nb * QK
    rows = np.arange(1 if nrows > 1 else 0, nrows, stride)
    scale = nrows / len(rows)
    acc, st = Acc(), {}
    rpc = max(1, chunk_elems // ne0)
    for i in range(0, len(rows), rpc):
        r = rows[i:i + rpc]
        cur = np.asarray(data2d[r]).reshape(len(r), nb, bs)
        has_prev = r > 0
        prv = np.asarray(data2d[np.maximum(r - 1, 0)]).reshape(len(r), nb, bs)
        process_chunk(qt, cur, prv, has_prev, acc, st)
    comp = {}
    for k, (m, A) in acc.m.items():
        dbits, tbits = counts_cost(m, A)
        comp[k] = {"data": dbits * scale, "table": tbits}
    n_el = nrows * ne0
    nblk = nrows * nb
    res = {"native_bits": nrows * rowbytes * 8, "code_native_bits": n_el * NATIVE_CODE_BITS[qt]}
    # code schemes
    cs = {}
    for name, parts in code_schemes(qt).items():
        cs[name] = {"data": sum(comp[p]["data"] for p in parts), "table": sum(comp[p]["table"] for p in parts)}
    res["code"] = cs
    # header schemes
    hs = {}
    for fld, kind in hdr_fields(qt).items():
        sch = dict(HDR_SCHEMES_U16 if kind == "u16" else HDR_SCHEMES_SUB)
        if fld == "m":
            sch["ctxsc"] = ["ctxsc"]
        if fld == "sc" and qt == "Q6_K":
            sch["ctxprevblockxprevsub_b"] = ["ctxprevblockxprevsub_b"]
        nsym = nblk if kind == "u16" else nblk * (8 if qt != "Q6_K" else 16)
        fb = 16 if kind == "u16" else (6 if qt != "Q6_K" else 8)
        hs[fld] = {"native_bits": nsym * fb, "schemes": {n: {"data": sum(comp[f"h:{fld}:{p}"]["data"] for p in ps),
                                                              "table": sum(comp[f"h:{fld}:{p}"]["table"] for p in ps)}
                                                          for n, ps in sch.items()}}
    res["hdr"] = hs
    assert res["code_native_bits"] + sum(h["native_bits"] for h in hs.values()) == res["native_bits"], qt
    # GPU-practical Huffman (codes), header fields raw
    m0 = acc.m["c:o0"][0][0]
    A = ALPH[qt]
    hb = float((m0 * huff_lengths(m0)).sum()) * scale
    mc = acc.m[HUFF_CTX[qt]][0]
    hbc, ntab = 0.0, 0
    for row in mc:
        if row.sum() > 0:
            hbc += float((row * huff_lengths(row)).sum())
            ntab += 1
    hbc *= scale
    hdr_raw = sum(h["native_bits"] for h in hs.values())
    res["huff"] = {"huff12_codes_bits": hb, "huff12ctx_codes_bits": hbc, "len_table_bits": A * 4, "ctx_tables": ntab,
                   "anchor64_bits": nrows * math.ceil(ne0 / 64) * 16, "anchor256_bits": nrows * math.ceil(ne0 / 256) * 16,
                   "hdr_raw_bits": hdr_raw}
    if qt == "Q8_0":
        res["q8"] = {"k127_hist_0_1_2_3plus": [int(x) for x in st["q8stats"]], "blocks_max127": st["q8max127"][0],
                     "blocks_sampled": st["q8max127"][1]}
    return res


# -------------------------------------------------------------------------------------------- aggregation
def summarise(tensors):
    by = {}
    for t in tensors:
        by.setdefault(t["type"], []).append(t)
    out = {}
    for qt, ts in by.items():
        nat = sum(t["native_bits"] for t in ts)
        cnat = sum(t["code_native_bits"] for t in ts)
        flds = list(ts[0]["hdr"].keys())
        hdr_o0 = sum(t["hdr"][f]["schemes"]["o0"]["data"] + t["hdr"][f]["schemes"]["o0"]["table"] for t in ts for f in flds)
        hdr_raw = sum(t["hdr"][f]["native_bits"] for t in ts for f in flds)
        code = {}
        for name in ts[0]["code"]:
            d = sum(t["code"][name]["data"] for t in ts)
            tb = sum(t["code"][name]["table"] for t in ts)
            code[name] = {"codes_ratio": (d + tb) / cnat, "codes_ratio_no_table": d / cnat, "table_bits": tb,
                          "block_ratio": (d + tb + hdr_o0) / nat}
        hdr = {}
        hdr_best = 0.0
        for f in flds:
            fn = sum(t["hdr"][f]["native_bits"] for t in ts)
            hdr[f] = {"native_bits": fn, "share_of_block": fn / nat, "schemes": {}}
            for sn in ts[0]["hdr"][f]["schemes"]:
                d = sum(t["hdr"][f]["schemes"][sn]["data"] for t in ts)
                tb = sum(t["hdr"][f]["schemes"][sn]["table"] for t in ts)
                hdr[f]["schemes"][sn] = {"ratio": (d + tb) / fn, "ratio_no_table": d / fn}
            # best per tensor for this field (a coder can pick per tensor; selector cost ignored)
            hdr_best += sum(min(s["data"] + s["table"] for s in t["hdr"][f]["schemes"].values()) for t in ts)
            hdr[f]["best_scheme_aggregate"] = min(hdr[f]["schemes"], key=lambda s: hdr[f]["schemes"][s]["ratio"])
        best_code = min(code, key=lambda n: code[n]["codes_ratio"])
        cbest_bits = code[best_code]["codes_ratio"] * cnat
        cbest_pt = sum(min(s["data"] + s["table"] for s in t["code"].values()) for t in ts)
        hf = {}
        for kind in ("huff12", "huff12ctx"):
            cb = sum(t["huff"][f"{kind}_codes_bits"] + t["huff"]["len_table_bits"] * (t["huff"]["ctx_tables"] if kind == "huff12ctx" else 1) for t in ts)
            hf[f"{kind}_codes_only_ratio"] = cb / cnat
            for a in (64, 256):
                an = sum(t["huff"][f"anchor{a}_bits"] for t in ts)
                hf[f"{kind}_anchor{a}_block_ratio"] = (cb + an + hdr_raw) / nat
        o0_no_table = (sum(t["code"]["o0"]["data"] for t in ts) + sum(t["hdr"][f]["schemes"]["o0"]["data"] for t in ts for f in flds)) / nat
        rec = {"tensors": len(ts), "native_bytes": nat / 8, "native_MiB": nat / 8 / 2**20,
               "code_share_of_block": cnat / nat, "o0_block_ratio": code["o0"]["block_ratio"],
               "o0_block_ratio_no_table_cost": o0_no_table,
               "best_code_scheme": best_code, "best_code_scheme_block_ratio": code[best_code]["block_ratio"],
               "best_combined_block_ratio": (cbest_bits + hdr_best) / nat,
               "best_combined_per_tensor_choice_block_ratio": (cbest_pt + hdr_best) / nat,
               "code_schemes": code, "header_fields": hdr, "gpu_huffman": hf}
        if qt == "Q8_0":
            k = np.sum([t["q8"]["k127_hist_0_1_2_3plus"] for t in ts], axis=0)
            m127 = sum(t["q8"]["blocks_max127"] for t in ts)
            nbk = sum(t["q8"]["blocks_sampled"] for t in ts)
            rec["q8_max_stats"] = {"blocks_sampled": int(nbk), "frac_max_abs_eq_127": m127 / nbk,
                                   "frac_blocks_by_count_of_abs127": {"0": k[0] / nbk, "1": k[1] / nbk, "2": k[2] / nbk, "3+": k[3] / nbk}}
        out[qt] = rec
    nat = sum(r["native_bytes"] for r in out.values()) * 8
    w = lambda key: sum(r[key] * r["native_bytes"] * 8 for r in out.values()) / nat  # noqa: E731
    overall = {"native_bytes": nat / 8, "native_MiB": nat / 8 / 2**20, "o0_block_ratio": w("o0_block_ratio"),
               "o0_block_ratio_no_table_cost": w("o0_block_ratio_no_table_cost"),
               "best_code_scheme_block_ratio": w("best_code_scheme_block_ratio"),
               "best_combined_block_ratio": w("best_combined_block_ratio"),
               "best_combined_per_tensor_choice_block_ratio": w("best_combined_per_tensor_choice_block_ratio")}
    for key in out[next(iter(out))]["gpu_huffman"]:
        overall["gpu_" + key] = sum(r["gpu_huffman"][key] * r["native_bytes"] * 8 for r in out.values()) / nat
    return {"overall": overall, "by_type": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, action="append", required=True)
    ap.add_argument("--gguf-root", type=Path, default=Path("<workdir>/llama.cpp-w/gguf-py"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--chunk-elems", type=int, default=1 << 22)
    ap.add_argument("--skip", default=r"^blk\.24\.", help="regex of tensor names to skip (qwen35-2b files only)")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite")
    sys.path.insert(0, str(args.gguf_root))
    from gguf import GGUFReader  # noqa: E402
    out = {"meta": {"stride": args.stride, "row_sampling": f"rows 1,1+{args.stride},...; column context uses row r-1",
                    "skip_regex_for_qwen35_2b": args.skip, "huffman_max_len": 12}, "files": {}}
    for model in args.model:
        t0 = time.time()
        reader = GGUFReader(str(model), "r")
        skip = re.compile(args.skip) if "qwen35-2b" in model.name else None
        tensors, skipped = [], []
        for t in reader.tensors:
            qt = t.tensor_type.name
            if skip is not None and skip.search(t.name):
                skipped.append(t.name)
                continue
            dims = [int(x) for x in t.shape]
            if qt not in BLOCK or len(dims) < 2 or any(x != 1 for x in dims[2:]) or dims[1] < 2:
                continue
            ne0 = int(t.shape[0])
            bs, QK = BLOCK[qt]
            rowbytes = ne0 // QK * bs
            data2d = np.asarray(t.data).view(np.uint8).reshape(-1, rowbytes)
            assert data2d.size == int(t.n_bytes)
            r = analyse_tensor(qt, data2d, args.stride, args.chunk_elems)
            r.update(name=t.name, type=qt, shape=[int(x) for x in t.shape])
            tensors.append(r)
            print(f"{model.name} {t.name} {qt} {list(t.shape)} o0={((r['code']['o0']['data'] + r['code']['o0']['table']) / r['code_native_bits']):.4f} "
                  f"t={time.time() - t0:.0f}s", flush=True)
            if args.max_tensors and len(tensors) >= args.max_tensors:
                break
        s = summarise(tensors)
        s["file_bytes"] = model.stat().st_size
        s["skipped_tensors"] = len(skipped)
        s["seconds"] = time.time() - t0
        s["tensors"] = [{k: v for k, v in t.items() if k in ("name", "type", "shape", "native_bits")} |
                        {"o0_codes_ratio": (t["code"]["o0"]["data"] + t["code"]["o0"]["table"]) / t["code_native_bits"],
                         "best_codes": min(t["code"], key=lambda n: t["code"][n]["data"] + t["code"][n]["table"])}
                        for t in tensors]
        out["files"][str(model)] = s
        print(json.dumps(s["overall"], indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
