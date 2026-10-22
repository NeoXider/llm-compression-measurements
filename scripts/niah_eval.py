"""Needle-in-a-haystack on GPU via llama-server, per KV type (deterministic, greedy, thinking off).

Haystack = the fixed corpus repeated and cut to ~target tokens (3.85 bytes/token estimate); a needle
sentence with a unique 4-digit code is inserted at relative depths; the question asks for the code.
Pass = the exact code appears in the answer. One server per KV type; only our own server is stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

NEEDLE = "Important note: the access code for the archive room {name} is {code}. "
NAMES = ["ORION", "VEGA", "LYRA", "RIGEL", "ALTAIR"]
CODES = ["4817", "9052", "3364", "7190", "2583"]


def post(url, body, timeout):
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
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--lengths", default="4096,32768")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--kv", default="f16,q8_0,q4_0")
    ap.add_argument("--port", type=int, default=18951)
    ap.add_argument("--extra", default="", help="extra server args, e.g. rope scaling")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("refusing to overwrite")
    a.out.mkdir(parents=True)
    src = a.corpus.read_text(encoding="utf-8")
    lengths = [int(x) for x in a.lengths.split(",")]
    depths = [float(x) for x in a.depths.split(",")]
    results = []
    for kv in a.kv.split(","):
        ctx = max(lengths) + 2048
        cmd = [a.server, "-m", a.model, "-ngl", "99", "--device", "CUDA0", "-c", str(ctx), "-fa", "on", "-ctk", kv, "-ctv", kv,
               "--port", str(a.port), "--host", "127.0.0.1", "--jinja", "-np", "1", "--no-webui"] + a.extra.split()
        log = (a.out / f"server_{kv}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{a.port}"
            wait(base, proc)
            for L in lengths:
                hay_bytes = int(L * 3.85 * 0.92)
                hay = (src * (hay_bytes // len(src.encode()) + 2)).encode()[:hay_bytes].decode("utf-8", "ignore")
                for i, d in enumerate(depths):
                    pos = int(len(hay) * d)
                    pos = hay.rfind("\n", 0, pos) + 1 or pos
                    text = hay[:pos] + NEEDLE.format(name=NAMES[i], code=CODES[i]) + hay[pos:]
                    q = (f"{text}\n\nQuestion: what is the access code for the archive room {NAMES[i]}? "
                         "Answer with the four digits only.")
                    body = {"messages": [{"role": "user", "content": q}], "temperature": 0, "top_k": 1, "seed": 4242,
                            "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}}
                    try:
                        r = post(base + "/v1/chat/completions", body, 1800)
                        ans = r["choices"][0]["message"].get("content") or ""
                        rec = {"kv": kv, "length": L, "depth": d, "code": CODES[i], "answer": ans.strip()[:80],
                               "pass": CODES[i] in ans, "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                               "timings": r.get("timings"), "error": None}
                    except Exception as e:
                        rec = {"kv": kv, "length": L, "depth": d, "code": CODES[i], "pass": False, "error": repr(e)}
                    results.append(rec)
                    print(json.dumps({k: rec.get(k) for k in ("kv", "length", "depth", "pass", "prompt_tokens", "answer", "error")}), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
    (a.out / "niah.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
