#!/usr/bin/env python3
"""Run on dgx (GB10). Benchmarks the OFFICIAL LocateAnything-3B usage (exactly as the
HF model card: bf16, cuda, py_apply_chat_template + process_vision_info, the documented
generate call) vs locate-anything.cpp (q8_0, via the C-API), on the GPU, WARM (a warmup
pass, then median of N timed), on several images.

Three measurements per image (all warm, GPU):
  ours        : q8_0, greedy hybrid (our CLI default), via la_capi (ctypes)
  up_greedy   : official bf16, do_sample=False, hybrid, max_new=256  -> deterministic,
                used for the apples-to-apples box-parity + fair-work timing
  up_sample   : official bf16 EXACTLY as the card (do_sample=True, temp=0.7, top_p=0.9,
                repetition_penalty=1.1, max_new=2048) -> the literal out-of-box time
Writes ~/dgx_official.json.
"""
import os, sys, time, json, ctypes, statistics, types, importlib.machinery
import numpy as np, torch

# decord has no aarch64 wheel and is video-only; stub it + skip the remote-code hard
# import check so the model loads for image detection (torchvision is real/matched).
_d = types.ModuleType("decord"); _d.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
sys.modules.setdefault("decord", _d)
import transformers.dynamic_module_utils as _dmu; _dmu.check_imports = lambda f: []

from transformers import AutoModel, AutoTokenizer, AutoProcessor
sys.path.insert(0, "/home/mudler/_git/locate-anything.cpp")
from scripts.diff_upstream import parse_boxes_from_ids

MODEL = "/home/mudler/_git/locate-anything.cpp/models/LocateAnything-3B"
SO    = "/home/mudler/_git/locate-anything.cpp/build-shared/liblocate_anything.so"
GGUF  = os.environ.get("LA_GGUF", "/home/mudler/la-q8.gguf")  # ours: f16 (precision-matched to bf16) or q8
PT    = "Locate all the instances that matches the following description: {}."
IMAGES = [("coco_skater", "person</c>skateboard"), ("bus", "person</c>bus"),
          ("coco_kitchen", "bottle</c>bowl")]
NTIME = 3       # timed runs (median); sampling uses fewer
_TOK = None


def dedup(boxes):
    seen, out = set(), []
    for lab, b in boxes:
        k = (lab, tuple(round(x) for x in b))
        if k in seen: continue
        seen.add(k); out.append((lab, b))
    return out


# ---------------- ours (la_capi via ctypes; load once -> warm) ----------------
def measure_ours(images):
    os.environ["LA_DEVICE"] = ""    # auto-select GPU
    lib = ctypes.CDLL(SO)
    lib.la_capi_load.restype = ctypes.c_void_p; lib.la_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.la_capi_locate_path.restype = ctypes.c_void_p
    lib.la_capi_locate_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.la_capi_free_string.argtypes = [ctypes.c_void_p]; lib.la_capi_free.argtypes = [ctypes.c_void_p]
    ctx = lib.la_capi_load(GGUF.encode(), 16)
    assert ctx, "la_capi_load failed"
    res = {}
    for img, cats in images:
        ip = f"/home/mudler/{img}.jpg".encode(); pr = PT.format(cats).encode()
        def run():
            r = lib.la_capi_locate_path(ctx, ip, pr, 0)  # 0 = hybrid
            js = ctypes.string_at(r).decode(); lib.la_capi_free_string(r); return js
        run()  # warmup
        ts = []
        for _ in range(NTIME):
            t0 = time.time(); js = run(); ts.append(time.time() - t0)
        boxes = [(d["label"].strip(), [float(x) for x in d["box"]]) for d in json.loads(js)["detections"]]
        res[img] = {"median_s": round(statistics.median(ts), 3), "boxes": dedup(boxes)}
    lib.la_capi_free(ctx)
    return res


# ---------------- upstream (official bf16, exactly as the card) ----------------
def load_upstream():
    # The card omits attn_implementation; on this transformers it resolves to 'eager'
    # which the remote code's mask path rejects (NotImplementedError). The model's
    # intended attention is 'magi' (-> faithful sdpa block-diffusion fallback with no
    # CUDA magi kernel). scripts.la_reference.load_model pins it on both config levels;
    # reuse it at the card's bf16 + cuda. Everything else (inputs/generate) is the card.
    import scripts.la_reference as R
    cfg, model, proc = R.load_model(dtype=torch.bfloat16, device="cuda")
    return model, proc.tokenizer, proc


def up_inputs(proc, img, cats):
    from PIL import Image
    image = Image.open(f"/home/mudler/{img}.jpg").convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": PT.format(cats)}]}]
    text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(messages)
    inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    for k, v in list(inputs.items()):
        if isinstance(v, np.ndarray): inputs[k] = torch.as_tensor(v).to("cuda")
    return inputs


def gen_call(model, tok, inputs, **extra):
    with torch.no_grad():
        return model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16), input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=tok, use_cache=True, generation_mode="hybrid", **extra)


def measure_upstream(model, tok, proc, images):
    global _TOK; _TOK = tok
    import scripts.diff_upstream as D; D._TOKENIZER = tok
    res = {}
    for img, cats in images:
        inputs = up_inputs(proc, img, cats)
        g = inputs["image_grid_hws"][0].tolist(); tw, th = g[1]*14, g[0]*14
        # --- greedy (deterministic; parity + fair-work timing, max_new=256) ---
        s = gen_call(model, tok, inputs, max_new_tokens=256, do_sample=False); torch.cuda.synchronize()
        tg = []
        for _ in range(NTIME):
            t0 = time.time(); s = gen_call(model, tok, inputs, max_new_tokens=256, do_sample=False)
            torch.cuda.synchronize(); tg.append(time.time() - t0)
        gids = tok(s, add_special_tokens=False)["input_ids"]
        gboxes = dedup(parse_boxes_from_ids(gids, tw, th))
        # --- sampling EXACTLY as the card (max_new=2048, temp/top_p/rep_pen) ---
        sk = dict(max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
        gen_call(model, tok, inputs, **sk); torch.cuda.synchronize()
        tsamp = []
        for _ in range(2):
            t0 = time.time(); gen_call(model, tok, inputs, **sk); torch.cuda.synchronize()
            tsamp.append(time.time() - t0)
        res[img] = {"greedy_s": round(statistics.median(tg), 3), "greedy_boxes": gboxes,
                    "sample_s": round(statistics.median(tsamp), 3), "target_w": tw, "target_h": th}
    return res


def main():
    print("measuring ours (q8_0, GPU, warm) ...", flush=True)
    ours = measure_ours(IMAGES)
    print("loading official upstream (bf16, cuda) ...", flush=True)
    model, tok, proc = load_upstream()
    up = measure_upstream(model, tok, proc, IMAGES)
    out = {}
    for img, _ in IMAGES:
        o, u = ours[img], up[img]
        md = (max((max(abs(x-y) for x, y in zip(a[1], b[1])) for a, b in zip(o["boxes"], u["greedy_boxes"])), default=0.0)
              if len(o["boxes"]) == len(u["greedy_boxes"]) else 9999.0)
        out[img] = {"ours_s": o["median_s"], "up_greedy_s": u["greedy_s"], "up_sample_s": u["sample_s"],
                    "speedup_vs_greedy": round(u["greedy_s"]/o["median_s"], 2),
                    "speedup_vs_official_sample": round(u["sample_s"]/o["median_s"], 2),
                    "ours_boxes": len(o["boxes"]), "up_greedy_boxes": len(u["greedy_boxes"]),
                    "box_maxdiff_px": round(md, 2), "target_w": u["target_w"], "target_h": u["target_h"],
                    "boxes": o["boxes"]}
        print(f"[{img:14s}] ours={o['median_s']:.2f}s  up_greedy={u['greedy_s']:.2f}s ({out[img]['speedup_vs_greedy']}x)  "
              f"up_sample={u['sample_s']:.2f}s ({out[img]['speedup_vs_official_sample']}x)  "
              f"boxes ours={len(o['boxes'])}/up={len(u['greedy_boxes'])} maxdiff={md:.1f}px", flush=True)
    from pathlib import Path
    Path("/home/mudler/dgx_official.json").write_text(json.dumps(out, indent=2))
    print("wrote ~/dgx_official.json")


if __name__ == "__main__":
    main()
