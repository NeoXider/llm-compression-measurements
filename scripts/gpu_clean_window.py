"""Clean-window preflight and 1 s GPU telemetry logger (read-only nvidia-smi queries).

preflight: samples the GPU N times and writes a verdict JSON. CLEAN only if
  - utilization <= --max-util on every sample,
  - no compute processes besides an explicit allow-list (default: none),
  - temperature <= --max-temp, and no active clock-throttle reasons except idle/application-clocks.
  It never stops or changes any process; on DIRTY it lists the blocking processes so the
  operator can decide.
log: appends one CSV row per second (power, clocks, temp, util, memory.used, throttle reasons)
  until the stop file appears or --seconds elapses; run it alongside a measurement series.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

GPU_FIELDS = ["timestamp", "index", "uuid", "utilization.gpu", "utilization.memory", "memory.used",
              "memory.total", "temperature.gpu", "power.draw", "clocks.sm", "clocks.mem",
              "clocks_throttle_reasons.active", "pstate"]
BENIGN_THROTTLE = {0x0, 0x1, 0x2}  # none, GPU idle, application clocks setting


def query_gpu(index: int) -> dict:
    out = subprocess.run(["nvidia-smi", f"--id={index}", f"--query-gpu={','.join(GPU_FIELDS)}",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    values = [v.strip() for v in out.strip().split(",")]
    return dict(zip(GPU_FIELDS, values))


def compute_apps() -> list[dict]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            apps.append({"pid": int(parts[0]), "name": parts[1], "used_memory": parts[2]})
    return apps


def preflight(args) -> int:
    samples, problems = [], []
    allow = {a.lower() for a in args.allow}
    for _ in range(args.samples):
        g = query_gpu(args.gpu)
        samples.append(g)
        if float(g["utilization.gpu"]) > args.max_util:
            problems.append(f"utilization {g['utilization.gpu']}% > {args.max_util}%")
        if float(g["temperature.gpu"]) > args.max_temp:
            problems.append(f"temperature {g['temperature.gpu']}C > {args.max_temp}C")
        try:
            reasons = int(g["clocks_throttle_reasons.active"], 16)
        except ValueError:
            reasons = 0
        if reasons not in BENIGN_THROTTLE:
            problems.append(f"throttle reasons {g['clocks_throttle_reasons.active']}")
        time.sleep(1)
    # Under WDDM every desktop app holding a D3D context shows up here, so the full list is recorded
    # for manual review and only known heavy GPU workloads mark the window DIRTY by name.
    apps = compute_apps()
    blockers = [a for a in apps if Path(a["name"]).name.lower() not in allow
                and any(h in a["name"].lower() for h in args.heavy)]
    if blockers:
        problems.append(f"{len(blockers)} known heavy GPU process(es) running")
    verdict = {"verdict": "CLEAN" if not problems else "DIRTY",
               "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
               "thresholds": {"max_util": args.max_util, "max_temp": args.max_temp},
               "problems": sorted(set(problems)), "blocking_compute_processes": blockers,
               "all_gpu_context_processes": apps, "samples": samples,
               "note": "Under WDDM nvidia-smi lists graphics/desktop apps as C+G; only type C or "
                       "busy C+G apps matter. Decide manually; this tool never kills anything."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps({k: verdict[k] for k in ("verdict", "problems")}, indent=1))
    return 0 if verdict["verdict"] == "CLEAN" else 3


def log(args) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.seconds
    with args.output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GPU_FIELDS)
        writer.writeheader()
        while time.time() < deadline and not (args.stop_file and args.stop_file.exists()):
            writer.writerow(query_gpu(args.gpu))
            f.flush()
            time.sleep(1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--max-util", type=float, default=3.0)
    p.add_argument("--max-temp", type=float, default=70.0)
    p.add_argument("--allow", action="append", default=[], help="process basename allowed to hold a context")
    p.add_argument("--heavy", action="append",
                   default=["llama-server", "llama-cli", "lm studio", "lms.exe", "comfyui",
                            "ollama", "python"],
                   help="substring of a process path that marks a heavy GPU workload (python is "
                        "included because ComfyUI/H3/training run as python; review manually)")
    p.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("log")
    q.add_argument("--gpu", type=int, default=0)
    q.add_argument("--seconds", type=float, default=4 * 3600)
    q.add_argument("--stop-file", type=Path)
    q.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    return preflight(args) if args.cmd == "preflight" else log(args)


if __name__ == "__main__":
    raise SystemExit(main())
