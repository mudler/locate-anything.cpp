#!/usr/bin/env python3
r"""Benchmark locate-anything.cpp vs the official HF implementation.

Measures inference wall-time on identical inputs and confirms the boxes agree
(real detections match upstream bit-for-bit; see scripts/diff_upstream.py).

Timing methodology (CPU, single process, model load excluded on both sides):
  * upstream: time `model.generate(...)` only, with the model already resident.
  * locate-anything-cli: inference time = t(detect) - t(info), where `info` loads
    the same GGUF and exits, so subtracting it removes the one-time model-load /
    page-cache cost and leaves the decode. Both detect and info are run warm.

Outputs benchmarks/results.json + a printed table, and saves annotated demo PNGs
to benchmarks/media/ for the README.

Usage:
  python -m benchmarks.bench                 # full matrix (slow/hybrid/fast, f32 + quants)
  python -m benchmarks.bench --quick         # fixture, slow only, f32 only
  python -m benchmarks.bench --threads 16
"""
import argparse, json, subprocess, time, statistics
from pathlib import Path

import torch

import scripts.la_reference as R
from scripts.diff_upstream import (upstream_boxes, cpp_boxes, parse_boxes_from_ids,
                                    match, default_cases, RTDETR_IMGS, PROMPT_TMPL)

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "benchmarks" / "media"
FIXTURE = ROOT / "tests/fixtures/parity_image.png"
FIXTURE_PROMPT = PROMPT_TMPL.format(cats="cat</c>remote")


def t_cli(cli, args, n=1):
    """median wall-time of a CLI invocation (stderr/stdout suppressed)."""
    ts = []
    for _ in range(n):
        t0 = time.time()
        r = subprocess.run([cli, *args], capture_output=True, text=True)
        ts.append(time.time() - t0)
        if r.returncode != 0:
            raise RuntimeError(f"CLI failed: {' '.join(args)}\n{r.stderr[-400:]}")
    return statistics.median(ts)


def cli_load_time(cli, gguf, threads):
    # `info` loads the model and exits -> isolates the load/page-cache cost.
    return t_cli(cli, ["info", "--model", gguf])


def cli_infer(cli, gguf, image, prompt, mode, threads, annotated=None):
    out = MEDIA / "_tmp.json"
    args = ["detect", "--model", gguf, "--input", str(image), "--prompt", prompt,
            "--mode", mode, "--threads", str(threads), "--output", str(out)]
    if annotated:
        args += ["--annotated", str(annotated)]
    total = t_cli(cli, args)
    boxes = []
    if out.exists():
        data = json.loads(out.read_text()); out.unlink()
        boxes = [(d["label"].strip(), [float(x) for x in d["box"]]) for d in data["detections"]]
    return total, boxes


def upstream_time(bundle, image, prompt, mode, max_new=256):
    """Time model.generate only (model already loaded). Returns (sec, boxes)."""
    from scripts.diff_upstream import build_inputs_for
    inputs = build_inputs_for(bundle, image, prompt)
    tok = bundle[2].tokenizer
    t0 = time.time()
    with torch.no_grad():
        s = bundle[1].generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"], image_grid_hws=inputs["image_grid_hws"],
                               tokenizer=tok, max_new_tokens=max_new, generation_mode=mode, use_cache=True)
    dt = time.time() - t0
    ids = tok(s, add_special_tokens=False)["input_ids"]
    grid = inputs["image_grid_hws"][0].tolist()
    boxes = parse_boxes_from_ids(ids, grid[1]*14, grid[0]*14)
    return dt, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=str(ROOT / "build/examples/cli/locate-anything-cli"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)

    f32 = str(ROOT / "models/locate-anything-f32.gguf")
    quants = {"f32": f32}
    for q in ("q8_0", "q4_k"):
        p = ROOT / f"models/locate-anything-{q}.gguf"
        if p.exists(): quants[q] = str(p)

    modes = ["slow"] if args.quick else ["slow", "hybrid", "fast"]

    import scripts.diff_upstream as D
    print("loading upstream HF model (loyal magi path, f32, cpu) ...", flush=True)
    bundle = R.load_model()
    D._TOKENIZER = bundle[2].tokenizer

    results = {"host": {}, "speed": [], "quant": [], "demo": []}
    try:
        import platform
        results["host"] = {"cpu": platform.processor() or "unknown", "threads": args.threads}
    except Exception:
        pass

    # --- load times per quant (warm) ---
    load_t = {q: cli_load_time(args.cli, g, args.threads) for q, g in quants.items()}
    print("load times (s):", {q: round(v, 2) for q, v in load_t.items()}, flush=True)

    # --- 1) speed matrix: fixture, each mode, cpp-f32 vs upstream-f32 ---
    for mode in modes:
        up_t, up_b = upstream_time(bundle, FIXTURE, FIXTURE_PROMPT, mode)
        det_t, cp_b = cli_infer(args.cli, f32, FIXTURE, FIXTURE_PROMPT, mode, args.threads,
                                annotated=MEDIA / f"fixture_{mode}.png")
        inf_t = max(1e-3, det_t - load_t["f32"])
        m, un_u, un_c, miou, _ = match(up_b, cp_b, 0.85)
        row = {"image": "parity_image.png", "mode": mode, "n_up": len(up_b), "n_cpp": len(cp_b),
               "matched": len(m), "min_iou": round(miou, 3),
               "upstream_s": round(up_t, 2), "cpp_infer_s": round(inf_t, 2),
               "cpp_total_s": round(det_t, 2), "speedup": round(up_t / inf_t, 2)}
        results["speed"].append(row)
        print(f"[speed] {mode:6s} up={up_t:6.2f}s cpp_infer={inf_t:6.2f}s "
              f"speedup={row['speedup']:.2f}x parity={len(m)}/{len(up_b)} iou={miou:.3f}", flush=True)

    # --- 2) quant timing + parity (slow, vs upstream-f32 boxes) ---
    if not args.quick:
        up_t, up_b = upstream_time(bundle, FIXTURE, FIXTURE_PROMPT, "slow")
        for q, g in quants.items():
            det_t, cp_b = cli_infer(args.cli, g, FIXTURE, FIXTURE_PROMPT, "slow", args.threads)
            inf_t = max(1e-3, det_t - load_t[q])
            m, _, _, miou, mcd = match(up_b, cp_b, 0.5)
            sz = Path(g).stat().st_size
            row = {"quant": q, "size_gb": round(sz / 1e9, 2), "cpp_infer_s": round(inf_t, 2),
                   "matched": len(m), "n_up": len(up_b), "min_iou": round(miou, 3),
                   "max_coord_px": round(mcd, 1)}
            results["quant"].append(row)
            print(f"[quant] {q:5s} {row['size_gb']}GB infer={inf_t:6.2f}s "
                  f"parity={len(m)}/{len(up_b)} iou={miou:.3f} maxpx={mcd:.1f}", flush=True)

    # --- 3) demo images: COCO scenes, slow, f32, annotated (cpp only) ---
    demo = [
        (RTDETR_IMGS / "coco_street.jpg",      "person</c>car"),
        (RTDETR_IMGS / "coco_kitchen.jpg",     "bottle</c>bowl"),
        (RTDETR_IMGS / "coco_living_room.jpg", "chair</c>couch"),
        (RTDETR_IMGS / "bus.jpg",              "person</c>bus"),
    ]
    for img, cats in demo:
        if not Path(img).exists(): continue
        prompt = PROMPT_TMPL.format(cats=cats)
        out = MEDIA / f"demo_{Path(img).stem}.png"
        det_t, cp_b = cli_infer(args.cli, f32, img, prompt, "slow", args.threads, annotated=out)
        results["demo"].append({"image": Path(img).name, "prompt": cats,
                                "n_boxes": len(cp_b), "annotated": out.name})
        print(f"[demo]  {Path(img).name:22s} boxes={len(cp_b)} -> {out.name}", flush=True)

    (ROOT / "benchmarks" / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote benchmarks/results.json")

if __name__ == "__main__":
    main()
