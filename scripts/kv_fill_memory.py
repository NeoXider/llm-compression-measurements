"""Per-process VRAM at a filled context, eager vs lazily mapped KV buffer (cq/kv-compress fork).

Runs llama-cli of the fork with a fixed large context (-c) and prompts of different lengths, env CQ_VRAM_REPORT=1
(the fork prints the process's DXGI local-segment usage after every llama_decode) and, for the lazy mode,
CQ_KV_LAZY=1. Records the last and the maximum CQ_VRAM value, llama.cpp's declared KV buffer size and the
lazy reservation. Order interleaves modes and KV types (ABBA over repeats).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=122880)
    ap.add_argument("--case", action="append", required=True, help="FILL=prompt_file")
    ap.add_argument("--kv", default="f16,q8_0,q4_0,f16d4")
    ap.add_argument("--modes", default="eager,lazy")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--extra", default="--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("refusing to overwrite")
    a.out.mkdir(parents=True)
    rows = []
    combos = [(m, kv) for kv in a.kv.split(",") for m in a.modes.split(",")]
    slot = 0
    for rep in range(a.repeats):
        order = combos if rep % 2 == 0 else list(reversed(combos))
        for case in a.case:
            fill, prompt = case.split("=", 1)
            for mode, kv in order:
                d = a.out / f"{slot:03d}-f{fill}-{kv}-{mode}-r{rep}"
                d.mkdir()
                env = dict(os.environ, CQ_VRAM_REPORT="1")
                env.pop("CQ_KV_LAZY", None)
                if mode == "lazy":
                    env["CQ_KV_LAZY"] = "1"
                argv = [a.cli, "-m", a.model, "-ngl", "99", "--device", "CUDA0", "-c", str(a.ctx), "-b", "512", "-ub", "512",
                        "-ctk", kv, "-ctv", kv, "-fa", "on", "-n", str(a.decode), "-s", "4242", "--temp", "0",
                        "--reasoning", "off", "-f", prompt, "--single-turn", "--simple-io", "--no-display-prompt",
                        "--perf", "--no-escape", "--ignore-eos"] + a.extra.split()
                t0 = time.time()
                with (d / "stdout.txt").open("wb") as so, (d / "stderr.txt").open("wb") as se:
                    rc = subprocess.run(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL, env=env, timeout=1800).returncode
                wall = time.time() - t0
                err = (d / "stderr.txt").read_text(encoding="utf-8", errors="replace")
                vr = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"CQ_VRAM n_tokens_total=(\d+) local_mib=([\d.]+)", err)]
                kvbuf = [float(x) for x in re.findall(r"KV buffer size =\s*([\d.]+) MiB", err)]
                pool = re.findall(r"f16e3_pool_used_blocks=(\d+) f16e3_pool_lost_blocks=(\d+)", err)
                lazy_res = [float(x) for x in re.findall(r"lazy KV buffer: reserved ([\d.]+) MiB", err)]
                row = {"slot": slot, "fill_target": int(fill), "kv": kv, "mode": mode, "repeat": rep, "exit": rc,
                       "wall_s": round(wall, 1), "tokens_total": vr[-1][0] if vr else None,
                       "local_mib_last": vr[-1][1] if vr else None, "local_mib_max": max(v for _, v in vr) if vr else None,
                       "kv_buffer_mib_declared": kvbuf, "lazy_reserved_mib": lazy_res,
                       "f16e3_pool_used_mib": int(pool[-1][0]) * 256 / 2**20 if pool else None,
                       "f16e3_pool_lost_blocks": int(pool[-1][1]) if pool else None}
                rows.append(row)
                (d / "row.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
                print(json.dumps(row), flush=True)
                slot += 1
    (a.out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
