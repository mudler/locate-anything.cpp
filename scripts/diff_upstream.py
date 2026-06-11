#!/usr/bin/env python3
r"""Live differential test: upstream HF LocateAnything-3B  vs  our C++/ggml engine.

Unlike the frozen-dump parity gates (which compare the C++ output to tensors
captured *once* from two fixed images), this harness loads the upstream model
LIVE and compares it to our `locate-anything-cli` on a sweep of images / prompts /
modes that are NOT in any dump. Agreement on novel inputs is the real evidence
that the port is loyal to upstream — drift the two frozen fixtures can't catch.

How it stays loyal to upstream:
  * Model is loaded via ``scripts.la_reference.load_model`` — the SAME magi-pinned
    path (block-diffusion mask, faithful magi->sdpa CPU fallback) that produced
    every reference dump. No shortcuts.
  * Inputs are built with the structured chat-content form the chat template
    requires (``[{"type":"image"}, {"type":"text",...}]``) + the numpy->torch
    fixups, exactly like ``la_reference.build_inputs``.
  * Boxes are parsed from upstream's generated token stream with the SAME
    token-id -> coord -> pixel logic as ``dump_slow_reference.py`` (which our
    C++ ``boxes.cpp`` is gated to reproduce). Both sides denormalize against the
    preprocessed grid size (gw*14 x gh*14), so the comparison is apples-to-apples.
  * Both sides use ``max_new_tokens=256`` (the C++ engine/CLI default) so the
    hybrid degenerate tail is the same length on both.

Usage:
  # default sweep (rt-detr COCO photos + the fixture, both modes):
  python -m scripts.diff_upstream

  # one custom case:
  python -m scripts.diff_upstream --image path.jpg --prompt "cat</c>dog" --mode slow

  # a JSON case list: [{"image":..,"prompt":..,"mode":"slow"|"hybrid"}, ...]
  python -m scripts.diff_upstream --cases cases.json

  # point the sweep at a different CLI / gguf:
  python -m scripts.diff_upstream --cli build/examples/cli/locate-anything-cli \
                                  --gguf models/locate-anything-f32.gguf

Exit code 0 iff every case agrees (same box count, same labels, IoU >= --iou).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import scripts.la_reference as R

ROOT = Path(__file__).resolve().parent.parent

# Token ids (same constants as dump_slow_reference.py / the converter).
COORD_START, COORD_END, BOX_START, BOX_END = 151677, 152677, 151668, 151669
REF_START, REF_END = 151672, 151673

# Default prompt template (matches tests/fixtures/fixture_spec.json).
PROMPT_TMPL = "Locate all the instances that matches the following description: {cats}."
RTDETR_IMGS = Path("/home/mudler/_git/rt-detr.cpp/benchmarks/images")


def default_cases():
    """A sweep that is strictly broader than the two frozen dumps: 6 unseen COCO
    scenes with unseen prompts across both decode modes, plus two sanity cases
    that overlap the dumps (parity_image/cat-remote and bus/person-bus)."""
    cats = lambda *c: PROMPT_TMPL.format(cats="</c>".join(c))
    cases = [
        # --- unseen images + unseen prompts (the real differential signal) ---
        (RTDETR_IMGS / "coco_cats.jpg",        cats("cat"),                 "slow"),
        (RTDETR_IMGS / "coco_street.jpg",      cats("person", "car"),       "slow"),
        (RTDETR_IMGS / "coco_kitchen.jpg",     cats("bottle", "bowl"),      "hybrid"),
        (RTDETR_IMGS / "coco_living_room.jpg", cats("chair", "couch"),      "slow"),
        (RTDETR_IMGS / "coco_skater.jpg",      cats("person", "skateboard"), "hybrid"),
        (RTDETR_IMGS / "coco_indoor.jpg",      cats("person"),              "hybrid"),
        # --- sanity: overlaps the frozen dumps (should be exact) ---
        (RTDETR_IMGS / "bus.jpg",              cats("person", "bus"),       "slow"),
        (ROOT / "tests/fixtures/parity_image.png", cats("cat", "remote"),  "slow"),
    ]
    return [{"image": str(i), "prompt": p, "mode": m} for i, p, m in cases if Path(i).exists()]


# --------------------------------------------------------------------------- #
# Upstream side
# --------------------------------------------------------------------------- #
def build_inputs_for(model_bundle, image_path, prompt, device="cpu"):
    """Generalized la_reference.build_inputs for an arbitrary image + prompt,
    reusing an already-loaded (cfg, model, processor) bundle."""
    cfg, model, processor = model_bundle
    image = Image.open(image_path).convert("RGB")
    messages = R.build_messages(prompt)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, np.ndarray):
            inputs[k] = torch.as_tensor(v).to(device)
    return inputs


def parse_boxes_from_ids(ids, img_w, img_h):
    """Token-id -> [(label, [x1,y1,x2,y2]) ...] in grid-pixel space.
    Identical logic to dump_slow_reference.py / src/boxes.cpp (x1,y1,x2,y2)."""
    boxes, i, n, cur_label = [], 0, len(ids), ""
    while i < n:
        t = int(ids[i])
        if t == REF_START:
            j = i + 1; lab = []
            while j < n and int(ids[j]) != REF_END:
                lab.append(int(ids[j])); j += 1
            cur_label = _decode_label(lab); i = j + 1; continue
        if t == BOX_START:
            j = i + 1; coords = []
            while j < n and int(ids[j]) != BOX_END:
                v = int(ids[j])
                if COORD_START <= v <= COORD_END:
                    coords.append(v - COORD_START)
                j += 1
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                boxes.append((cur_label,
                              [x1 / 1000 * img_w, y1 / 1000 * img_h,
                               x2 / 1000 * img_w, y2 / 1000 * img_h]))
            i = j + 1; continue
        i += 1
    return boxes


_TOKENIZER = None
def _decode_label(lab):
    return _TOKENIZER.decode(lab, skip_special_tokens=False).strip() if lab else ""


def upstream_boxes(model_bundle, image_path, prompt, mode, max_new=256):
    cfg, model, processor = model_bundle
    inputs = build_inputs_for(model_bundle, image_path, prompt)
    tok = processor.tokenizer
    with torch.no_grad():
        s = model.generate(
            pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], image_grid_hws=inputs["image_grid_hws"],
            tokenizer=tok, max_new_tokens=max_new, generation_mode=mode, use_cache=True)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    grid = inputs["image_grid_hws"][0].tolist()
    img_w, img_h = grid[1] * 14, grid[0] * 14   # grid = [h_patches, w_patches]
    return parse_boxes_from_ids(ids, img_w, img_h), (img_w, img_h)


# --------------------------------------------------------------------------- #
# C++ side
# --------------------------------------------------------------------------- #
def cpp_boxes(cli, gguf, image_path, prompt, mode):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    cmd = [cli, "detect", "--model", gguf, "--input", str(image_path),
           "--prompt", prompt, "--mode", mode, "--output", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CLI failed ({r.returncode}): {r.stderr.strip()[:400]}")
    data = json.loads(Path(out).read_text())
    Path(out).unlink(missing_ok=True)
    return [(d["label"].strip(), [float(x) for x in d["box"]]) for d in data["detections"]]


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ub = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    den = ua + ub - inter
    return inter / den if den > 0 else (1.0 if inter == 0 else 0.0)


def match(up, cp, iou_thr):
    """Greedy best-IoU matching, label-exact. Returns (matched list, n_unmatched_up,
    n_unmatched_cp, min_iou_over_matches, max_coord_diff_over_matches)."""
    used = set()
    matched, ious, cdiffs = [], [], []
    for ul, ub in up:
        best, bj = -1.0, -1
        for j, (cl, cb) in enumerate(cp):
            if j in used or cl != ul:
                continue
            v = iou(ub, cb)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= iou_thr:
            used.add(bj)
            matched.append((ul, ub, cp[bj][1], best))
            ious.append(best)
            cdiffs.append(max(abs(ub[k] - cp[bj][1][k]) for k in range(4)))
    n_un_up = len(up) - len(matched)
    n_un_cp = len(cp) - len(used)
    return matched, n_un_up, n_un_cp, (min(ious) if ious else 1.0), (max(cdiffs) if cdiffs else 0.0)


def main():
    ap = argparse.ArgumentParser(description="Live upstream-vs-C++ differential test.")
    ap.add_argument("--cli", default=str(ROOT / "build/examples/cli/locate-anything-cli"))
    ap.add_argument("--gguf", default=str(ROOT / "models/locate-anything-f32.gguf"))
    ap.add_argument("--cases", help="JSON file: [{image,prompt,mode}, ...]")
    ap.add_argument("--image"); ap.add_argument("--prompt")
    ap.add_argument("--mode", default="slow", choices=["slow", "hybrid"])
    ap.add_argument("--iou", type=float, default=0.85, help="min IoU to count a match")
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    if args.image and args.prompt:
        cases = [{"image": args.image, "prompt": args.prompt, "mode": args.mode}]
    elif args.cases:
        cases = json.loads(Path(args.cases).read_text())
    else:
        cases = default_cases()

    if not Path(args.cli).exists():
        sys.exit(f"CLI not found: {args.cli} (build it: cmake --build build --target locate-anything-cli)")
    if not Path(args.gguf).exists():
        sys.exit(f"GGUF not found: {args.gguf}")

    print(f"loading upstream HF model (loyal magi path, f32, cpu) ...", flush=True)
    bundle = R.load_model()              # (cfg, model, processor)
    global _TOKENIZER
    _TOKENIZER = bundle[2].tokenizer
    print(f"running {len(cases)} differential case(s); IoU>={args.iou}, max_new={args.max_new}\n", flush=True)

    all_ok = True
    rows = []
    for idx, c in enumerate(cases):
        img, prompt, mode = c["image"], c["prompt"], c["mode"]
        name = Path(img).name
        try:
            up, (gw, gh) = upstream_boxes(bundle, img, prompt, mode, args.max_new)
            cp = cpp_boxes(args.cli, args.gguf, img, prompt, mode)
        except Exception as e:
            print(f"[{idx}] {name:24s} {mode:6s} ERROR: {e}")
            all_ok = False
            rows.append((name, mode, "ERR", 0, 0, 0.0, 0.0))
            continue

        matched, un_up, un_cp, min_iou, max_cd = match(up, cp, args.iou)
        ok = (len(up) == len(cp)) and (un_up == 0) and (un_cp == 0)
        all_ok &= ok
        tag = "OK " if ok else "DIFF"
        print(f"[{idx}] {name:24s} {mode:6s} {tag}  up={len(up)} cpp={len(cp)} "
              f"matched={len(matched)} unmatched(up/cpp)={un_up}/{un_cp} "
              f"minIoU={min_iou:.3f} maxCoordDiff={max_cd:.1f}px  grid={gw}x{gh}")
        if not ok:
            up_s = [(l, [round(x, 1) for x in b]) for l, b in up]
            cp_s = [(l, [round(x, 1) for x in b]) for l, b in cp]
            print(f"      upstream: {up_s}")
            print(f"      cpp     : {cp_s}")
        rows.append((name, mode, tag, len(up), len(cp), min_iou, max_cd))

    print("\n=== summary ===")
    nok = sum(1 for r in rows if r[2] == "OK ")
    print(f"{nok}/{len(rows)} cases agree (same count, same labels, IoU>={args.iou})")
    worst = [r for r in rows if r[2] != "OK "]
    if worst:
        print("disagreements:", [(r[0], r[1], r[2]) for r in worst])
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
