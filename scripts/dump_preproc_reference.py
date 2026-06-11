#!/usr/bin/env python3
r"""M6a Task 1: dump the HF preprocessing + vision intermediates for a NON-448
(variable-resolution, non-square) real image into ``dumps/reference_preproc.gguf``.

This is the GOLD that every M6a task gates against:
  * Task 2 gates the C++ PIL-bicubic resize + patchify on ``resized_chw`` /
    ``pixel_values``.
  * Task 3 gates the variable-grid MoonViT on ``merged_tokens``.
  * Task 5 gates the prompt builder on ``input_ids``.
  * Task 6 gates the end-to-end engine on ``slow_boxes`` / ``box_labels``.

The 448x448 fixture used by ``dump_reference.py`` collapses to a clean 32x32 grid;
here we deliberately use ``bus.jpg`` (810x1080) so the variable-resolution path is
exercised (grid ~76x56).

We replicate the image processor's ``_preprocess`` stages MANUALLY so we can capture
``resized_chw`` -- the normalized CHW tensor AFTER rescale+to_tensor+normalize but
BEFORE patchify (this isolates the PIL-bicubic resampler for Task 2). The merged
vision tokens are captured with the SAME forward hook on ``model.vision_model`` that
``dump_reference.py`` (M0) used, so the reference stays faithful.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import gguf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from scripts.la_reference import (
    load_model,
    build_messages,
    IMAGE_CONTEXT_TOKEN_ID,
)

DUMP_DIR = REPO_ROOT / "dumps"
OUT_PATH = DUMP_DIR / "reference_preproc.gguf"

IMAGE_PATH = "/home/mudler/_git/rt-detr.cpp/benchmarks/images/bus.jpg"
PROMPT = "Locate all the instances that matches the following description: person</c>bus."

# Box-parse token ids (same as scripts/dump_slow_reference.py).
COORD_START, COORD_END = 151677, 152677
BOX_START, BOX_END = 151668, 151669
REF_START, REF_END = 151672, 151673


def _np(t):
    return t.detach().to(torch.float32).cpu().contiguous().numpy()


def main():
    torch.manual_seed(0)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    cfg, model, processor = load_model()
    model.eval()
    tok = processor.tokenizer
    ip = processor.image_processor

    img = Image.open(IMAGE_PATH).convert("RGB")
    print(f"image {IMAGE_PATH} size(w,h)={img.size}")

    # =================== Manual _preprocess (capture resized_chw) ===========
    # Mirror LocateAnythingImageProcessor._preprocess exactly:
    #   rescale (PIL-bicubic resize+pad-to-multiple) -> to_tensor (/255) ->
    #   normalize (mean/std 0.5) -> patchify.
    resized = ip.rescale(img, ip.merge_kernel_size)
    chw = ip.normalize(ip.to_tensor(resized))           # [3, target_h, target_w]
    patches, grid_hw = ip.patchify(chw)                 # [N,3,14,14], (gh, gw)
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    target_h, target_w = int(chw.shape[1]), int(chw.shape[2])

    print(f"GRID {gh}x{gw} (gh x gw)  target(h,w)=({target_h},{target_w})")
    assert gh % 2 == 0 and gw % 2 == 0, f"grid not even: {gh}x{gw}"
    assert gh < 512 and gw < 512, f"grid exceeds pos emb: {gh}x{gw}"
    assert (gh, gw) != (32, 32), "got the 448 fixture grid; expected a NON-448 grid"
    assert gh * gw == patches.shape[0], "grid/pixel_values mismatch"
    n_img = (gh // 2) * (gw // 2)
    print(f"N_img={n_img}  pixel_values={tuple(patches.shape)}")

    # =================== Build prompt inputs (same path as build_inputs) =====
    messages = build_messages(PROMPT)
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat], images=[img], return_tensors="pt")
    for k, v in list(inputs.items()):
        if isinstance(v, np.ndarray):
            inputs[k] = torch.as_tensor(v)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_hws = inputs["image_grid_hws"]
    pixel_values = inputs["pixel_values"]

    # The processor's own patchify must match our manual one.
    assert torch.allclose(pixel_values.to(torch.float32), patches.to(torch.float32)), (
        "processor pixel_values disagree with manual patchify"
    )
    pg = image_grid_hws[0].tolist()
    assert (int(pg[0]), int(pg[1])) == (gh, gw), (
        f"processor grid {pg} != manual ({gh},{gw})"
    )

    n_ctx = int((input_ids == IMAGE_CONTEXT_TOKEN_ID).sum().item())
    print(f"input_ids={tuple(input_ids.shape)}  IMG_CONTEXT count={n_ctx}")
    assert n_ctx == n_img, f"IMG_CONTEXT count {n_ctx} != n_img {n_img}"

    # =================== Vision tower -> merged_tokens (M0 hook) =============
    captured = {}
    h = model.vision_model.register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("merged_tokens", torch.cat(list(o), dim=0))
    )
    with torch.no_grad():
        pv = pixel_values.to(model.language_model.dtype)
        _ = model.extract_feature(pv, image_grid_hws)
    h.remove()

    merged = captured["merged_tokens"]
    print(f"merged_tokens={tuple(merged.shape)}")
    assert tuple(merged.shape) == (n_img, 4608), (
        f"merged_tokens shape {tuple(merged.shape)} != ({n_img},4608)"
    )

    # =================== slow generate -> boxes (Task 6 gate) ===============
    print("running slow generate (this may take several minutes on CPU)...")
    with torch.no_grad():
        gen = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            tokenizer=tok,
            max_new_tokens=256,
            generation_mode="slow",
            do_sample=False,
            use_cache=True,
        )
    ids = tok(gen, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids, skip_special_tokens=False) == gen, "id round-trip mismatch"
    ids = np.array(ids, dtype=np.int64)

    boxes, labels = [], []
    i, n, cur_label = 0, len(ids), ""
    while i < n:
        t = int(ids[i])
        if t == REF_START:
            j = i + 1
            lab = []
            while j < n and int(ids[j]) != REF_END:
                lab.append(int(ids[j]))
                j += 1
            cur_label = tok.decode(lab, skip_special_tokens=False)
            i = j + 1
            continue
        if t == BOX_START:
            j = i + 1
            coords = []
            while j < n and int(ids[j]) != BOX_END:
                v = int(ids[j])
                if COORD_START <= v <= COORD_END:
                    coords.append(v - COORD_START)
                j += 1
            if len(coords) == 4:
                x1, y1, x2, y2 = coords  # x1,y1,x2,y2
                boxes.append([
                    x1 / 1000 * target_w, y1 / 1000 * target_h,
                    x2 / 1000 * target_w, y2 / 1000 * target_h,
                ])
                labels.append(cur_label)
            i = j + 1
            continue
        i += 1
    barr = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32)
    print("boxes(px):", [[round(x, 1) for x in b] for b in boxes], "labels:", labels)

    # =================== Write gguf =========================================
    resized_chw = _np(chw)
    pv_np = _np(pixel_values)
    merged_np = _np(merged)
    ids_i32 = input_ids[0].cpu().numpy().astype(np.int32)

    for name, a in (("resized_chw", resized_chw), ("pixel_values", pv_np),
                    ("merged_tokens", merged_np)):
        assert a.size > 1 and np.isfinite(a).all() and np.abs(a).sum() > 0, \
            f"{name} looks like a stub/non-finite"

    w = gguf.GGUFWriter(str(OUT_PATH), "la_preproc")
    w.add_tensor("resized_chw", np.ascontiguousarray(resized_chw))
    w.add_tensor("pixel_values", np.ascontiguousarray(pv_np))
    w.add_tensor("merged_tokens", np.ascontiguousarray(merged_np))
    w.add_tensor("input_ids", np.ascontiguousarray(ids_i32))
    w.add_tensor("slow_boxes", np.ascontiguousarray(barr))
    w.add_uint32("la_preproc.gh", gh)
    w.add_uint32("la_preproc.gw", gw)
    w.add_uint32("la_preproc.target_h", target_h)
    w.add_uint32("la_preproc.target_w", target_w)
    w.add_uint32("la_preproc.n_img_tokens", n_img)
    w.add_string("la_preproc.image_path", IMAGE_PATH)
    w.add_string("la_preproc.prompt", PROMPT)
    w.add_array("la_preproc.box_labels", labels if labels else ["__none__"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"\nwrote {OUT_PATH}")
    print(f"  resized_chw   {resized_chw.shape} {resized_chw.dtype}")
    print(f"  pixel_values  {pv_np.shape} {pv_np.dtype}")
    print(f"  merged_tokens {merged_np.shape} {merged_np.dtype}")
    print(f"  input_ids     {ids_i32.shape} {ids_i32.dtype}")
    print(f"  slow_boxes    {barr.shape} {barr.dtype}")
    print(f"  gh={gh} gw={gw} target=({target_h},{target_w}) n_img={n_img} "
          f"n_boxes={len(boxes)}")


if __name__ == "__main__":
    main()
