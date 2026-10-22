"""Capture real KV caches via llama-server slot save (GPU). For E1 (more texts/lengths) and E4 (BF16 vs F16 KV).

For each (kv type, text, length): start our llama-server with --slot-save-path, send the text as a raw
completion prompt (n_predict 1, cache_prompt), then POST /slots/0?action=save. Only our own server is stopped.
Texts are cut to ~target tokens (3.85 bytes/token estimate; exact token count is recorded from the server).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait(base, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", action="append", required=True, help="NAME=path")
    ap.add_argument("--plan", action="append", required=True, help="KV:LENGTH, e.g. f16:4096")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18981)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    texts = {}
    for spec in a.text:
        n, p = spec.split("=", 1)
        texts[n] = Path(p).read_text(encoding="utf-8", errors="replace")
    plans = {}
    for spec in a.plan:
        kv, L = spec.split(":")
        plans.setdefault(kv, []).append(int(L))
    log = []
    for kv, lengths in plans.items():
        ctx = max(lengths) + 1024
        cmd = [a.server, "-m", a.model, "-ngl", "99", "--device", "CUDA0", "-c", str(ctx), "-fa", "on", "-ctk", kv, "-ctv", kv,
               "--port", str(a.port), "--host", "127.0.0.1", "-np", "1", "--no-webui", "--slot-save-path", str(a.out)]
        lf = (a.out / f"server_{kv}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{a.port}"
            wait(base, proc)
            for name, txt in texts.items():
                for L in lengths:
                    nbytes = int(L * 3.85 * 0.95)
                    t = (txt * (nbytes // max(1, len(txt.encode())) + 2)).encode()[:nbytes].decode("utf-8", "ignore")
                    r = post(base + "/completion", {"prompt": t, "n_predict": 1, "cache_prompt": True, "temperature": 0})
                    fn = f"{name}_{kv}_{L}.bin"
                    s = post(base + "/slots/0?action=save", {"filename": fn})
                    rec = {"kv": kv, "text": name, "target": L, "file": fn, "tokens_evaluated": r.get("tokens_evaluated"),
                           "save": {k: s.get(k) for k in ("n_saved", "n_written")}}
                    log.append(rec)
                    print(json.dumps(rec), flush=True)
                    post(base + "/slots/0?action=erase", {})
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            lf.close()
    (a.out / "captures.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
