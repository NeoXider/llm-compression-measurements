"""Final matrix (work/FINAL_MATRIX.md): regular vs compressed model vs compressed KV vs both, for every base pair.

One binary set (cq/combined build) for all rows. Phases (run separately, each holds the GPU lock in the caller):
  speed   : llama-bench decode/prefill at depths, ABBA over rows (forward then reverse), jsonl per row and round
  memory  : llama-cli with filled prompts (-c ctx), process DXGI memory via CQ_VRAM_REPORT, per row and fill
  load    : model load time (llama-cq-logits --load-only), 3 repeats
  quality : KL / top-1 vs the pair's regular row (llama-perplexity), 39 tasks (cpu_task_eval.py)
  exact   : raw logits of lossless rows vs the regular row (llama-cq-logits, batch and per-token)
Usage: python final_matrix.py PHASE --out DIR [--rows ROWS.json] [--pairs bf16,q8,q4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
BIN = Path("<workdir>/llama.cpp-combo/build-combo/bin")
Q = P / "work" / "qwen35_2b"
CQ = P / "work" / "cqw"
CORPUS = P / "artifacts" / "corpus_v1" / "corpus.txt"
PROMPTS = P / "artifacts" / "prompts_filled"

# name, pair, variant, model file, kv type, extra env, status note
ROWS = [
    ("bf16.A",        "bf16", "A regular",                 Q / "qwen35-2b-BF16.gguf",        "f16",   {}),
    ("bf16.B.ll.H",   "bf16", "B model lossless (Huffman)",  CQ / "qwen35-2b-BF16-cqH.gguf",  "f16",   {}),
    ("bf16.B.ll.W4",  "bf16", "B model lossless (W4+esc)",   CQ / "qwen35-2b-BF16-cqw4.gguf", "f16",   {}),
    ("bf16.B.ly.W3",  "bf16", "B model lossy (W3)",          CQ / "qwen35-2b-BF16-cqw3.gguf", "f16",   {}),
    ("bf16.B.ly.R",   "bf16", "B model lossy (FFN->Q8_0)",   Q / "qwen35-2b-BF16+ffnQ8.gguf", "f16",   {}),
    ("bf16.C.ll",     "bf16", "C KV lossless (f16e3)",       Q / "qwen35-2b-BF16.gguf",        "f16e3", {}),
    ("bf16.C.ly",     "bf16", "C KV lossy (f16d4)",          Q / "qwen35-2b-BF16.gguf",        "f16d4", {"LLAMA_ATTN_ROT_DISABLE": "1"}),
    ("bf16.D.ll",     "bf16", "D both lossless (H+f16e3)",   CQ / "qwen35-2b-BF16-cqH.gguf",  "f16e3", {}),
    ("bf16.D.ly",     "bf16", "D both lossy (W3+f16d4)",     CQ / "qwen35-2b-BF16-cqw3.gguf", "f16d4", {"LLAMA_ATTN_ROT_DISABLE": "1"}),
    ("q8.A",          "q8",   "A regular",                 Q / "qwen35-2b-Q8_0.gguf",         "q8_0",  {}),
    ("q8.B.ly",       "q8",   "B model lossy (FFN->Q6_K)",   Q / "qwen35-2b-Q8_0+ffnQ6K.gguf",  "q8_0",  {}),
    ("q8.C.kernel",   "q8",   "C KV same bytes, GQA kernel", Q / "qwen35-2b-Q8_0.gguf",         "q8_0",  {"CQ_FA_VEC_GQA": "1"}),
    ("q8.D.ly",       "q8",   "D model lossy + GQA kernel",  Q / "qwen35-2b-Q8_0+ffnQ6K.gguf",  "q8_0",  {"CQ_FA_VEC_GQA": "1"}),
    ("q4.A",          "q4",   "A regular",                 Q / "qwen35-2b-Q4_K_M.gguf",       "q4_0",  {}),
    ("q4.B.ly",       "q4",   "B model lossy (emb->Q4_K)",   Q / "qwen35-2b-Q4_K_M+embQ4K.gguf", "q4_0", {}),
    ("q4.C.ly",       "q4",   "C KV lossy (q4o)",            Q / "qwen35-2b-Q4_K_M.gguf",       "q4o",   {}),
    ("q4.D.ly",       "q4",   "D both lossy (emb->Q4_K+q4o)", Q / "qwen35-2b-Q4_K_M+embQ4K.gguf", "q4o",  {}),
]


def run(argv, out_file: Path, env_extra: dict, timeout: int = 7200) -> int:
    env = dict(os.environ)
    for k in ("CQ_FA_VEC_GQA", "LLAMA_ATTN_ROT_DISABLE", "CQ_KV_LAZY", "CQ_FA_SCRATCH_POOL", "CQ_VRAM_REPORT"):
        env.pop(k, None)
    env.update(env_extra)
    with out_file.open("wb") as fo:
        return subprocess.run([str(a) for a in argv], stdout=fo, stderr=subprocess.STDOUT, env=env, timeout=timeout,
                              stdin=subprocess.DEVNULL).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["speed", "memory", "load", "quality", "exact"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", default="bf16,q8,q4")
    ap.add_argument("--depths", default="4096,32768,122880")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--kl-ctx", type=int, default=2048)
    ap.add_argument("--kl-chunks", type=int, default=6)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = a.pairs.split(",")
    rows = [r for r in ROWS if r[1] in pairs]
    prog = a.out / "progress.txt"
    log = lambda s: prog.open("a", encoding="utf-8").write(f"{s} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    if a.phase == "speed":
        i = 0
        for rnd in range(a.rounds):
            order = rows if rnd % 2 == 0 else list(reversed(rows))
            for name, pair, var, model, kv, env in order:
                f = a.out / f"{i:03d}-{name}-r{rnd}.jsonl"
                rc = run([BIN / "llama-bench.exe", "-m", model, "-ngl", "99", "-fa", "1", "-ctk", kv, "-ctv", kv,
                          "-p", "512", "-n", "64", "-d", a.depths, "-r", str(a.reps), "-o", "jsonl"], f, env)
                log(f"speed {i} {name} round {rnd} exit={rc}")
                i += 1
    elif a.phase == "memory":
        fills = {4096: PROMPTS / "c4096.txt", 32768: PROMPTS / "c32768.txt", 122880: PROMPTS / "c122880.txt"}
        for name, pair, var, model, kv, env in rows:
            for fill, prompt in fills.items():
                for lazy in (0, 1):
                    f = a.out / f"mem-{name}-f{fill}-lazy{lazy}.log"
                    e = dict(env, CQ_VRAM_REPORT="1")
                    if lazy:
                        e.update(CQ_KV_LAZY="1", CQ_FA_SCRATCH_POOL="1")
                    rc = run([BIN / "llama-cli.exe", "-m", model, "-ngl", "99", "--device", "CUDA0", "-c", str(a.ctx),
                              "-b", "512", "-ub", "512", "-ctk", kv, "-ctv", kv, "-fa", "on", "-n", "16", "-s", "4242",
                              "--temp", "0", "--reasoning", "off", "-f", prompt, "--single-turn", "--simple-io",
                              "--no-display-prompt", "--no-escape", "--ignore-eos"], f, e, timeout=3600)
                    txt = f.read_text(encoding="utf-8", errors="replace")
                    vr = re.findall(r"CQ_VRAM n_tokens_total=(\d+) local_mib=([\d.]+)", txt)
                    rec = {"row": name, "fill": fill, "lazy": lazy, "exit": rc,
                           "tokens": int(vr[-1][0]) if vr else None, "local_mib": float(vr[-1][1]) if vr else None}
                    (a.out / "memory.jsonl").open("a", encoding="utf-8").write(json.dumps(rec) + "\n")
                    log(f"memory {name} {fill} lazy={lazy} exit={rc}")
    elif a.phase == "load":
        for rep in range(3):
            for name, pair, var, model, kv, env in rows:
                f = a.out / f"load-{name}-r{rep}.log"
                rc = run([BIN / "llama-cq-logits.exe", "-m", model, "--load-only"], f, env)
                ms = re.findall(r"load_ms ([\d.]+)", f.read_text(encoding="utf-8", errors="replace"))
                (a.out / "load.jsonl").open("a", encoding="utf-8").write(json.dumps(
                    {"row": name, "rep": rep, "exit": rc, "load_ms": float(ms[0]) if ms else None,
                     "file_bytes": Path(model).stat().st_size}) + "\n")
        log("load done")
    elif a.phase == "quality":
        kld_dir = Path("/path/to/home/AppData/Local/Temp/vramc_kl/final")
        kld_dir.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            prow = [r for r in rows if r[1] == pair]
            base = prow[0]
            kld = kld_dir / f"{pair}_A_c{a.kl_ctx}.kld"
            common = ["-f", CORPUS, "-c", str(a.kl_ctx), "-b", "512", "-ngl", "99", "-fa", "on", "--chunks", str(a.kl_chunks)]
            rc = run([BIN / "llama-perplexity.exe", "-m", base[3], "-ctk", base[4], "-ctv", base[4], *common,
                      "--kl-divergence-base", kld], a.out / f"kl-{base[0]}-base.log", base[5])
            log(f"kl base {pair} exit={rc}")
            for name, _, var, model, kv, env in prow[1:]:
                rc = run([BIN / "llama-perplexity.exe", "-m", model, "-ctk", kv, "-ctv", kv, *common,
                          "--kl-divergence-base", kld, "--kl-divergence"], a.out / f"kl-{name}.log", env)
                log(f"kl {name} exit={rc}")
        for name, pair, var, model, kv, env in rows:
            rc = run(["python", "-B", P / "scripts" / "cpu_task_eval.py", "--gpu", "--kv-type", kv, "--server",
                      BIN / "llama-server.exe", "--model", f"{name}={model}", "--ctx-size", "16384", "--port", "18971",
                      "--out-dir", a.out / f"tasks-{name}"], a.out / f"tasks-{name}.log", env)
            log(f"tasks {name} exit={rc}")
    elif a.phase == "exact":
        prompt = PROMPTS / "c1024.txt"
        for pair in pairs:
            prow = [r for r in rows if r[1] == pair]
            for name, _, var, model, kv, env in prow:
                if not (name.endswith(".A") or ".ll" in name):
                    continue
                for mode in ("batch", "pertoken"):
                    extra = ["--per-token"] if mode == "pertoken" else []
                    f = a.out / f"logits-{name}-{mode}.bin"
                    rc = run([BIN / "llama-cq-logits.exe", "-m", model, "-f", prompt, "-n", "64", "--kv", kv, "--ctx", "4096",
                              *extra, "-o", f], a.out / f"logits-{name}-{mode}.log", dict(env, CQ_FA_VEC_GQA="1",
                                                                                         GGML_CUDA_DISABLE_FUSION="1"))
                    log(f"exact {name} {mode} exit={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
