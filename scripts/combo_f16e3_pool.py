"""f16e3 overflow-pool use on REAL text (llama-bench fills the cache with random tokens). Starts llama-server (combined
fork) with -ctk/-ctv f16e3 and CQ_VRAM_REPORT=1, sends a long prompt twice (cache_prompt=false, so the second request
rewrites the cache), and records the pool counters printed after every llama_decode. Run under the GPU lock.
Usage: python combo_f16e3_pool.py MODEL.gguf PROMPT.txt CTX OUT_PREFIX
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER = r"D:\Git\llama.cpp-combo\build-combo\bin\llama-server.exe"


def post(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    model, prompt_file, ctx, out = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    env = dict(os.environ, CQ_VRAM_REPORT="1")
    log = open(str(out) + ".server.log", "w", encoding="utf-8", errors="replace")
    port = 18991
    cmd = [SERVER, "-m", model, "-ngl", "99", "-c", str(ctx), "-fa", "on", "-ctk", "f16e3", "-ctv", "f16e3", "--port", str(port),
           "--host", "127.0.0.1", "-np", "1", "--no-webui", "-b", "2048", "-ub", "512"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    res = {"cmd": cmd, "requests": []}
    try:
        for _ in range(300):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        text = prompt_file.read_text(encoding="utf-8", errors="replace")
        for i in range(2):
            t0 = time.time()
            r = post(f"http://127.0.0.1:{port}/completion", {"prompt": text, "n_predict": 1, "cache_prompt": False})
            res["requests"].append({"i": i, "seconds": time.time() - t0, "tokens_evaluated": r.get("tokens_evaluated"),
                                    "timings": r.get("timings")})
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    txt = Path(str(out) + ".server.log").read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"CQ_VRAM n_tokens_total=(\d+) local_mib=([\d.]+) f16e3_pool_used_blocks=(\d+) f16e3_pool_lost_blocks=(\d+)", txt)
    res["pool_trace"] = [[int(a), float(b), int(c), int(d)] for a, b, c, d in rows]
    res["last"] = res["pool_trace"][-1] if rows else None
    Path(str(out) + ".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"requests": res["requests"], "last": res["last"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
