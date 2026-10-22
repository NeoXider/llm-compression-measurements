"""CPU-only answer-quality run of the frozen 39-case suite over several GGUF configs.

For each model: start our own llama-server with the GPU hidden (CUDA_VISIBLE_DEVICES=-1, -ngl 0),
send every case once through /v1/chat/completions (greedy, seed 4242, thinking disabled,
max_tokens = case budget), score with the repo's frozen deterministic scorer
(tools/evaluate_compression_quality.score_case), then stop only that server process.
Speed fields are CPU diagnostics only and are not GPU results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_health(base: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become healthy")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="LABEL=path.gguf")
    ap.add_argument("--repo", type=Path, default=Path("<project-repo>"))
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--gpu", action="store_true", help="run on CUDA0 with all layers offloaded (default: CPU only)")
    ap.add_argument("--kv-type", default=None, help="f16|q8_0|q4_0: sets -ctk/-ctv and -fa on")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise SystemExit("refusing to overwrite")
    sys.path.insert(0, str(args.repo / "tools"))
    import evaluate_compression_quality as q  # noqa: E402

    bench_path = args.repo / "benchmarks/compression_quality_v1.json"
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True)
    env = dict(os.environ) if args.gpu else dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    summary = {"benchmark_sha256": hashlib.sha256(bench_path.read_bytes()).hexdigest(),
               "server": str(args.server), "backend": "cuda" if args.gpu else "cpu",
               "kv_type": args.kv_type, "ctx_size": args.ctx_size, "configs": {}}
    for spec in args.model:
        label, path = spec.split("=", 1)
        log = (args.out_dir / f"server_{label}.log").open("w", encoding="utf-8")
        cmd = [str(args.server), "-m", path, "-ngl", "99" if args.gpu else "0", "-c", str(args.ctx_size),
               "-t", str(args.threads), "--port", str(args.port), "--host", "127.0.0.1", "--jinja", "-np", "1",
               "--no-webui"]
        if args.gpu:
            cmd += ["--device", "CUDA0"]
        if args.kv_type:
            cmd += ["-fa", "on", "-ctk", args.kv_type, "-ctv", args.kv_type]
        summary.setdefault("commands", {})[label] = cmd
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        records = []
        try:
            base = f"http://127.0.0.1:{args.port}"
            wait_health(base, proc, 600)
            load_s = time.time() - t0
            for case in bench["cases"]:
                prompt = (case["context"] + "\n\n" if case["context"] else "") + case["prompt"]
                body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_k": 1,
                        "seed": 4242, "max_tokens": case["decode_budget_tokens"],
                        "chat_template_kwargs": {"enable_thinking": False}}
                rec = {"case_id": case["id"], "stratum": case["stratum"]}
                try:
                    r = post(base + "/v1/chat/completions", body, 1800)
                    choice = r["choices"][0]
                    rec.update(output=choice["message"].get("content"), finish_reason=choice.get("finish_reason"),
                               error=None, timings=r.get("timings"), usage=r.get("usage"))
                except Exception as exc:  # recorded, never dropped
                    rec.update(output=None, finish_reason=None, error=repr(exc))
                outcome, detail = q.score_case(case, rec)
                rec["outcome"], rec["detail"] = outcome, detail
                records.append(rec)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        counts = {}
        for r in records:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        by_stratum = {}
        for r in records:
            s = by_stratum.setdefault(r["stratum"], {"n": 0, "pass": 0})
            s["n"] += 1
            s["pass"] += r["outcome"] == "pass"
        model_path = Path(path)
        summary["configs"][label] = {"model": path, "model_bytes": model_path.stat().st_size,
                                     "load_seconds_cpu": load_s, "outcomes": counts,
                                     "pass": counts.get("pass", 0), "n": len(records), "by_stratum": by_stratum}
        (args.out_dir / f"records_{label}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
        print(label, counts, flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
