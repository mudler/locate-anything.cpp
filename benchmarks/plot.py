#!/usr/bin/env python3
"""Render the benchmark plots into benchmarks/plots/ from the measured JSON files.
Dark theme matching the demo clips."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "plots"; OUT.mkdir(parents=True, exist_ok=True)
BG, FG, DIM, GRID = "#0d1117", "#d7dde5", "#6e7681", "#222b34"
TEAL, SLATE, AMBER = "#3ec8e0", "#94a3b2", "#ffcf56"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": GRID, "font.size": 12, "font.family": "DejaVu Sans"})


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)


# ---- 1) GPU speedup vs the official bf16 (greedy), per image, f16 + q8 ----
q8 = json.loads((ROOT/"benchmarks/demo/dgx_official.json").read_text())
f16 = json.loads((ROOT/"benchmarks/demo/dgx_official_f16.json").read_text())
imgs = ["coco_skater", "bus", "coco_kitchen"]; names = ["skater", "bus", "kitchen"]
f16_sp = [q8[k]["up_greedy_s"]/f16[k]["ours_s"] for k in imgs]
q8_sp  = [q8[k]["up_greedy_s"]/q8[k]["ours_s"] for k in imgs]
fig, ax = plt.subplots(figsize=(7.5, 4.2)); x = range(len(imgs)); w = 0.36
b1 = ax.bar([i-w/2 for i in x], f16_sp, w, label="ours f16 (precision-matched)", color=SLATE)
b2 = ax.bar([i+w/2 for i in x], q8_sp, w, label="ours q8_0 (deployment)", color=TEAL)
ax.axhline(1.0, color=AMBER, lw=1.5, ls="--"); ax.text(len(imgs)-0.5, 1.03, "official bf16 (greedy) = 1.0", color=AMBER, ha="right", fontsize=10)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{b.get_height():.1f}×", ha="center", color=FG, fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(names); ax.set_ylabel("speedup vs official (greedy)")
ax.set_title("GB10 GPU: ours vs official bf16 (greedy)", color=FG, fontweight="bold", loc="left")
ax.set_ylim(0, max(q8_sp)*1.25); ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=10); style(ax)
fig.tight_layout(); fig.savefig(OUT/"gpu_speedup.png", dpi=150); plt.close(fig)

# ---- 2) Quant tradeoff (CPU): size vs speedup, box-parity coded ----
qr = json.loads((ROOT/"benchmarks/quant_results.json").read_text())
OFFICIAL_F32_SLOW = 23.65  # official PyTorch f32 slow, the CPU baseline
order = ["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k"]
qr = {r["dtype"]: r for r in qr}; rows = [qr[d] for d in order if d in qr]
sizes = [r["size_gb"] for r in rows]; sp = [OFFICIAL_F32_SLOW/r["infer_s"] for r in rows]
# identical-to-f32 boxes (iou 1.0) -> teal, else amber
ident = [r.get("min_iou", 1.0) >= 0.999 for r in rows]
cols = [TEAL if i else AMBER for i in ident]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar([r["dtype"] for r in rows], sp, color=cols, width=0.6)
for b, s, sz in zip(bars, sp, sizes):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"{s:.1f}×\n{sz:.1f} GB", ha="center", color=FG, fontsize=9)
ax.set_ylabel("speedup vs official PyTorch f32 (CPU, slow)")
ax.set_title("Quantization (CPU): smaller + faster, same detections", color=FG, fontweight="bold", loc="left")
ax.set_ylim(0, max(sp)*1.2); style(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=TEAL, label="box-identical to f32"), Patch(color=AMBER, label="sub-pixel box drift")],
          facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=10)
fig.tight_layout(); fig.savefig(OUT/"quant_tradeoff.png", dpi=150); plt.close(fig)

# ---- 3) CPU decode modes vs official PyTorch f32 ----
modes = ["slow", "hybrid", "fast"]; mode_sp = [1.66, 3.09, 2.96]   # from the speed table
fig, ax = plt.subplots(figsize=(6.5, 4.0))
bars = ax.bar(modes, mode_sp, color=TEAL, width=0.55)
ax.axhline(1.0, color=AMBER, lw=1.5, ls="--")
for b, s in zip(bars, mode_sp):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"{s:.2f}×", ha="center", color=FG, fontweight="bold")
ax.set_ylabel("speedup vs official PyTorch f32"); ax.set_ylim(0, max(mode_sp)*1.2)
ax.set_title("CPU (Ryzen 9 9950X3D): decode modes vs official, same boxes", color=FG, fontweight="bold", loc="left")
style(ax); fig.tight_layout(); fig.savefig(OUT/"cpu_speedup.png", dpi=150); plt.close(fig)

print("wrote", *(p.name for p in sorted(OUT.glob("*.png"))))
