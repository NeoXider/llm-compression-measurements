"""Per-process GPU memory with a FILLED context, native KV F16/Q8_0/Q4_0 (extends R1).

Runs the observer-enabled llama-cli (<project-repo> experimental/wddm-cuda-luid build) with -f prompt
files sized to fill most of each context, interleaving KV types across repeats
(order per repeat: f16,q8_0,q4_0 then reversed). Parses the DXGI calling-process observer records
with the repo's validate_receipt.validate and llama.cpp's declared buffer sizes.
Timings are recorded but are diagnostic only (window contended unless the preflight was CLEAN).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path

MIB = 1 << 20


def load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("cq_wddm_validator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", type=Path, required=True)
    ap.add_argument("--validator", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--case", action="append", required=True, help="CTX=prompt_file")
    ap.add_argument("--kv", default="f16,q8_0,q4_0")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--extra", default="", help="extra llama-cli args, e.g. '--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768'")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise SystemExit("refusing to overwrite")
    args.out_dir.mkdir(parents=True)
    validator = load_validator(args.validator)
    kvs = args.kv.split(",")
    env = dict(os.environ, CQ_GPU_MEMORY_TRACE="1", CQ_GPU_MEMORY_INTERVAL_MS="5")
    for k in ("CUDA_PATH", "CUDA_PATH_V11_8"):
        env.pop(k, None)
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;" + env["PATH"]
    rows = []
    slot = 0
    for case in args.case:
        ctx, prompt = case.split("=", 1)
        prompt_path = Path(prompt)
        for rep in range(args.repeats):
            order = kvs if rep % 2 == 0 else list(reversed(kvs))
            for kv in order:
                d = args.out_dir / f"{slot:03d}-c{ctx}-{kv}-r{rep}"
                d.mkdir()
                argv = [str(args.cli), "-m", str(args.model), "-ngl", "99", "--device", "CUDA0",
                        "-c", ctx, "-b", "512", "-ub", "128", "-t", "8", "-ctk", kv, "-ctv", kv, "-fa", "on",
                        "-n", str(args.decode), "-s", "4242", "--temp", "0", "--reasoning", "off", "-lv", "5",
                        "-f", str(prompt_path), "--no-host", "--single-turn", "--simple-io",
                        "--no-display-prompt", "--perf", "--no-escape", "--ignore-eos"] + args.extra.split()
                (d / "command.json").write_text(json.dumps({"argv": argv}, indent=1), encoding="utf-8")
                t0 = time.time()
                with (d / "stdout.txt").open("wb") as so, (d / "stderr.txt").open("wb") as se:
                    proc = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL, env=env)
                    rc = proc.wait(timeout=1800)
                wall = time.time() - t0
                err = (d / "stderr.txt").read_text(encoding="utf-8", errors="replace")
                row = {"slot": slot, "ctx": int(ctx), "kv": kv, "repeat": rep, "exit": rc, "wall_s": wall,
                       "pid": proc.pid, "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()}
                try:
                    v = validator.validate(err, proc.pid)
                    s = v["summary"]
                    row.update(observer_valid=True,
                               local_after_init_mib=v["after_init"]["local"]["current_usage_bytes"] / MIB,
                               local_sampled_peak_mib=s["local_peak_current_usage_bytes"] / MIB,
                               non_local_sampled_peak_mib=s["non_local_peak_current_usage_bytes"] / MIB,
                               valid_pairs=s["valid_pairs"])
                except Exception as exc:
                    row.update(observer_valid=False, observer_error=repr(exc))
                for key, pat in {"kv_buffer_mib": r"llama_kv_cache: size =\s*([\d.]+) MiB",
                                 "compute_buffer_mib": r"CUDA0 compute buffer size =\s*([\d.]+) MiB",
                                 "model_buffer_mib": r"CUDA0 model buffer size =\s*([\d.]+) MiB"}.items():
                    vals = [float(x) for x in re.findall(pat, err)]
                    row[key] = max(vals) if vals else None
                m = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", err)
                row["prompt_tokens"] = int(m.group(2)) if m else None
                row["prompt_ms_diag"] = float(m.group(1)) if m else None
                m = re.search(r"\n[^\n]*\beval time =\s*([\d.]+) ms /\s*(\d+) runs", err)
                row["decode_ms_diag"] = float(m.group(1)) if m else None
                row["decode_tokens"] = int(m.group(2)) if m else None
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ("slot", "ctx", "kv", "exit", "observer_valid",
                                                          "local_sampled_peak_mib", "kv_buffer_mib",
                                                          "compute_buffer_mib", "prompt_tokens")}), flush=True)
                slot += 1
    (args.out_dir / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
