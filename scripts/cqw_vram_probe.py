"""Run a command and record the per-process dedicated GPU memory of that process (Windows WDDM counter
"\\GPU Process Memory(pid_<PID>_*)\\Dedicated Usage", the source Task Manager uses), sampled about once per second.

Usage: python cqw_vram_probe.py --out OUT.json -- CMD ARGS...
Writes {"cmd", "exit", "wall_s", "peak_mib", "samples": [[t_s, mib], ...]} and the command's stdout/stderr to
OUT.json.log. The peak is the largest sample; short spikes between samples can be missed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

PS = r"""
$p = {pid}
while ($true) {{
  $c = Get-Counter "\GPU Process Memory(pid_$($p)_*)\Dedicated Usage" -ErrorAction SilentlyContinue
  if ($c) {{ $s = ($c.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum; Write-Output ("{{0}} {{1}}" -f [DateTimeOffset]::Now.ToUnixTimeMilliseconds(), $s) }}
  if (-not (Get-Process -Id $p -ErrorAction SilentlyContinue)) {{ break }}
  Start-Sleep -Milliseconds 200
}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    log = open(str(a.out) + ".log", "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    mon = subprocess.Popen(["powershell", "-NoProfile", "-Command", PS.format(pid=proc.pid)],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    samples = []

    def reader():
        for line in mon.stdout:
            parts = line.split()
            if len(parts) == 2:
                try:
                    samples.append([int(parts[0]) / 1000.0 - t0, float(parts[1]) / 2**20])
                except ValueError:
                    pass

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    rc = proc.wait()
    wall = time.time() - t0
    try:
        mon.wait(15)
    except subprocess.TimeoutExpired:
        mon.kill()
    th.join(5)
    log.close()
    peak = max((s[1] for s in samples), default=None)
    a.out.write_text(json.dumps({"cmd": cmd, "exit": rc, "wall_s": wall, "peak_mib": peak, "samples": samples}, indent=1),
                     encoding="utf-8")
    print(f"exit={rc} wall={wall:.1f}s peak_mib={peak}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
