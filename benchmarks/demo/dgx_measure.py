#!/usr/bin/env python3
"""Run on dgx (GB10). Times the official PyTorch model AND locate-anything.cpp on the
SAME image, on BOTH cpu and cuda, so each tier is apples-to-apples on one box.
Writes ~/dgx_results.json: {cpu:{upstream,ours}, cuda:{upstream,ours}} inference seconds."""
import os, sys, time, json, subprocess, types
from pathlib import Path
import numpy as np
import torch

# torchvision is real (matched torch+torchvision installed). decord has no aarch64
# wheel and is VIDEO-only (never touched by image detection), so stub it + skip the
# remote-code hard-import check so the model loads for the detection path.
def _install_stubs():
    import importlib.machinery
    d = types.ModuleType("decord")
    d.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)  # so find_spec() works
    sys.modules.setdefault("decord", d)
    import transformers.dynamic_module_utils as dmu
    dmu.check_imports = lambda f: []
_install_stubs()

REPO = "/home/mudler/_git/locate-anything.cpp"
sys.path.insert(0, REPO)
import scripts.la_reference as R
import scripts.diff_upstream as D
from scripts.diff_upstream import build_inputs_for, parse_boxes_from_ids

IMG = "/home/mudler/coco_street.jpg"
PROMPT = "Locate all the instances that matches the following description: person</c>car."
CLI = f"{REPO}/build/examples/cli/locate-anything-cli"
GGUF = "/home/mudler/la-q8.gguf"
MODE = "hybrid"


def time_upstream(device):
    cfg, model, proc = R.load_model(device=device)
    inputs = build_inputs_for((cfg, model, proc), IMG, PROMPT, device=device)
    tok = proc.tokenizer
    D._TOKENIZER = tok   # parse_boxes_from_ids -> _decode_label uses this module global
    def gen():
        with torch.no_grad():
            return model.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], image_grid_hws=inputs["image_grid_hws"],
                tokenizer=tok, max_new_tokens=256, generation_mode=MODE, use_cache=True)
    if device == "cuda":
        _ = gen(); torch.cuda.synchronize()                 # warmup
    t0 = time.time(); s = gen()
    if device == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    ids = tok(s, add_special_tokens=False)["input_ids"]
    g = inputs["image_grid_hws"][0].tolist()
    boxes = _dedup(parse_boxes_from_ids(ids, g[1]*14, g[0]*14))
    del model; torch.cuda.empty_cache() if device == "cuda" else None
    return round(dt, 2), len(ids), boxes


def _dedup(boxes):
    seen, out = set(), []
    for lab, b in boxes:
        k = (lab, tuple(round(x) for x in b))
        if k in seen: continue
        seen.add(k); out.append((lab, b))
    return out


def time_cli(ladev):
    env = dict(os.environ, LA_DEVICE=ladev)
    outp = "/home/mudler/_cli.json"
    def t(args):
        t0 = time.time(); subprocess.run([CLI, *args], env=env, capture_output=True); return time.time() - t0
    li = t(["info", "--model", GGUF])
    dt = t(["detect", "--model", GGUF, "--input", IMG, "--prompt", PROMPT, "--mode", MODE,
            "--threads", "16", "--output", outp])
    b = [(d["label"].strip(), [float(x) for x in d["box"]]) for d in json.loads(Path(outp).read_text())["detections"]]
    return round(max(0.05, dt - li), 2), _dedup(b)


def _maxdiff(a, b):
    if len(a) != len(b): return 9999.0
    return max((max(abs(x-y) for x, y in zip(ba[1], bb[1])) for ba, bb in zip(a, b)), default=0.0)


def main():
    res = {}
    for dev, ladev in (("cpu", "cpu"), ("cuda", "")):
        up, ntok, up_b = time_upstream(dev)
        ours, our_b = time_cli(ladev)
        md = _maxdiff(up_b, our_b)
        res[dev] = {"upstream_s": up, "ours_s": ours, "speedup": round(up/ours, 2),
                    "tokens": ntok, "up_boxes": len(up_b), "our_boxes": len(our_b),
                    "box_maxdiff_px": round(md, 2)}
        print(f"[{dev:4s}] upstream={up:7.2f}s  ours={ours:6.2f}s  speedup={up/ours:.1f}x  "
              f"boxes up={len(up_b)}/ours={len(our_b)} maxdiff={md:.1f}px  (tokens={ntok})", flush=True)
    Path("/home/mudler/dgx_results.json").write_text(json.dumps(res, indent=2))
    print("wrote ~/dgx_results.json")


if __name__ == "__main__":
    main()
