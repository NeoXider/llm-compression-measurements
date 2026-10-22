# make_more_figures.py -- extra figures (KV, weights, server GPUs, infographics).
# Run from anywhere:  python scripts/make_more_figures.py [fig_function_name ...]    (no args = all)
# Parts: mf_common.py (helpers/data), mf_formats.py, mf_kv.py, mf_weights.py, mf_server.py, mf_info.py.
# Palette: gray #8c8c8c = base, green #1b9e77 = ours lossless, orange #e66101 = ours lossy; hatch = CPU-only prototype; n/m = not measured.
import sys, os, importlib, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PARTS = ["mf_formats", "mf_kv", "mf_kv2", "mf_weights", "mf_server", "mf_info"]


def main(names):
    ok = []
    for p in PARTS:
        try:
            mod = importlib.import_module(p)
        except ModuleNotFoundError as e:
            if e.name == p:
                print("part not present:", p); continue
            raise
        for fn in sorted(n for n in dir(mod) if n.startswith("fig") and callable(getattr(mod, n)) and getattr(getattr(mod, n), "__module__", "") == p):
            if names and fn not in names and not any(fn.startswith(n) for n in names):
                continue
            try:
                getattr(mod, fn)(); ok.append(fn)
            except Exception:
                print("FAILED", fn); traceback.print_exc()
    print("done:", len(ok), "figure functions")


if __name__ == "__main__":
    main(sys.argv[1:])
