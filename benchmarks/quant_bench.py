#!/usr/bin/env python3
"""Compare every quant (f32/f16/q8_0/q6_k/q5_k/q4_k) on the fixture: slow-mode
inference time (CLI, inference-only via the info-subtraction trick) + box parity
against the f32 output (which is byte-identical to the official model). No upstream
needed. Writes benchmarks/quant_results.json."""
import json, subprocess, time, statistics
from pathlib import Path
from scripts.diff_upstream import match

ROOT = Path(__file__).resolve().parent.parent
CLI = str(ROOT / "build/examples/cli/locate-anything-cli")
FIX = str(ROOT / "tests/fixtures/parity_image.png")
PROMPT = "Locate all the instances that matches the following description: cat</c>remote."
ORDER = ["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k"]


def t(cli_args, n=2):
    ts = []
    for _ in range(n):
        t0 = time.time(); r = subprocess.run([CLI, *cli_args], capture_output=True, text=True)
        ts.append(time.time() - t0)
        assert r.returncode == 0, r.stderr[-300:]
    return statistics.median(ts)


def boxes(gguf):
    out = ROOT / "benchmarks" / "_q.json"
    dt = t(["detect", "--model", gguf, "--input", FIX, "--prompt", PROMPT,
            "--mode", "slow", "--threads", "16", "--output", str(out)])
    b = [(d["label"].strip(), [float(x) for x in d["box"]])
         for d in json.loads(out.read_text())["detections"]]
    out.unlink()
    return dt, b


def main():
    quants = {q: str(ROOT / f"models/locate-anything-{q}.gguf") for q in ORDER
              if (ROOT / f"models/locate-anything-{q}.gguf").exists()}
    load = {q: t(["info", "--model", g]) for q, g in quants.items()}
    ref = None; rows = []
    for q in ORDER:
        if q not in quants: continue
        det, b = boxes(quants[q])
        infer = max(0.05, det - load[q])
        if ref is None: ref = b
        m, un_r, un_b, miou, mcd = match(ref, b, 0.5)
        sz = Path(quants[q]).stat().st_size
        rows.append({"dtype": q, "size_gb": round(sz/1e9, 2), "infer_s": round(infer, 2),
                     "load_s": round(load[q], 2), "n_boxes": len(b),
                     "match_vs_f32": f"{len(m)}/{len(ref)}", "min_iou": round(miou, 3),
                     "max_coord_px": round(mcd, 2)})
        print(f"{q:5s} {sz/1e9:5.2f}GB  infer={infer:6.2f}s  boxes={len(b)} "
              f"match={len(m)}/{len(ref)} iou={miou:.3f} maxpx={mcd:.2f}", flush=True)
    (ROOT / "benchmarks" / "quant_results.json").write_text(json.dumps(rows, indent=2))
    # speedup vs f32
    f32 = next((r for r in rows if r["dtype"] == "f32"), None)
    if f32:
        print("\nspeedup vs f32:")
        for r in rows:
            print(f"  {r['dtype']:5s} {f32['infer_s']/r['infer_s']:.2f}x  ({r['size_gb']}GB)")
    print("\nwrote benchmarks/quant_results.json")


if __name__ == "__main__":
    main()
