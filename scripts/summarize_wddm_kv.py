"""Re-derive the native-KV memory table from raw <project-repo> receipts (read-only).

Input: a results directory produced by tools/gpu_paired_inference.py --capture-mode memory
(e.g. <project-repo>/results/gpu_wddm_native_kv_20261011_r01).
Output: JSON with, per slot, the DXGI calling-process LOCAL/NON_LOCAL sampled peak,
llama.cpp declared buffer sizes, perf counters, and the arm/variant labels.
No GPU, no model, no writes outside --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MIB = 1024 * 1024


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite existing output")
    slots = []
    for slot_dir in sorted(p for p in args.run_dir.iterdir() if p.is_dir() and p.name[:3].isdigit()):
        receipt_path = slot_dir / "receipt.json"
        memory_path = slot_dir / "driver_memory.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        summary = memory["observer"]["summary"]
        kv = next(a for a in json.loads((slot_dir / "command.json").read_text(encoding="utf-8"))["argv"]
                  if a in ("f16", "q8_0", "q4_0"))
        allocated = [s for s in receipt["buffers"]["snapshots"] if not s["no_alloc"]][-1]["buffers_mib"]
        slots.append({
            "slot": slot_dir.name, "kv_type": kv, "arm": receipt["arm"], "group_variant": receipt["variant"],
            "exit_code": receipt["process"]["returncode"],
            "local_sampled_peak_mib": summary["local_peak_current_usage_bytes"] / MIB,
            "non_local_sampled_peak_mib": summary["non_local_peak_current_usage_bytes"] / MIB,
            "valid_sample_pairs": summary["valid_pairs"],
            "physical_peak_proven": memory.get("physical_peak_proven"),
            "declared_buffers_mib": allocated,
            "prompt_tokens": receipt["perf"]["prompt_eval"]["count"],
            "decode_tokens": receipt["perf"]["eval"]["count"],
            "decode_tok_s": receipt["perf"]["eval"]["tokens_per_second"],
            "prefill_tok_s": receipt["perf"]["prompt_eval"]["tokens_per_second"],
            "receipt_sha256": sha(receipt_path), "driver_memory_sha256": sha(memory_path),
        })
    by_kv: dict[str, dict] = {}
    for s in slots:
        b = by_kv.setdefault(s["kv_type"], {"n": 0, "local_mib": set(), "non_local_mib": set(),
                                            "kv_buffer_mib": set(), "compute_buffer_mib": set()})
        b["n"] += 1
        b["local_mib"].add(s["local_sampled_peak_mib"])
        b["non_local_mib"].add(s["non_local_sampled_peak_mib"])
        b["kv_buffer_mib"].add(s["declared_buffers_mib"].get("CUDA0:KV"))
        b["compute_buffer_mib"].add(s["declared_buffers_mib"].get("CUDA0:compute"))
    base = None
    table = {}
    for kv in ("f16", "q8_0", "q4_0"):
        b = by_kv.get(kv)
        if not b:
            continue
        local = sorted(b["local_mib"])
        row = {"slots": b["n"], "local_mib_distinct_values": local,
               "non_local_mib_distinct_values": sorted(b["non_local_mib"]),
               "kv_buffer_mib": sorted(b["kv_buffer_mib"]), "compute_buffer_mib": sorted(b["compute_buffer_mib"])}
        if kv == "f16":
            base = local[0]
        elif base is not None and len(local) == 1:
            row["delta_vs_f16_mib"] = local[0] - base
            row["delta_vs_f16_pct"] = 100.0 * (local[0] - base) / base
        table[kv] = row
    out = {"schema": "wddm-native-kv-summary-v1", "run_dir": str(args.run_dir),
           "metric": "DXGI QueryVideoMemoryInfo CurrentUsage of the calling process, LOCAL segment, "
                     "sampled every 5 ms; sampled peak is a lower bound, not a proven physical peak",
           "table": table, "slots": slots}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
