#!/usr/bin/env python3
"""Generate the generation_mode='fast' (MTP-only, no AR fallback) greedy reference
into dumps/reference_fast.gguf, for gating the C++ fast-mode decode + box parser.

Fast mode differs from hybrid ONLY in the pure-decode post-processing (the forward
is identical and already gated by M5):
  * decode_bbox_avg: no abnormal-coord->0 zeroing (final_coords = first_valid_ids).
  * handle_pattern: a malformed box stays a coord_box (full 6 tokens) instead of
    becoming an error_box that switches to AR.
  * control flow: use_mtp stays True forever (the AR branch never runs).
So this dump captures the fast token stream + boxes on the same fixture as the
slow/hybrid dumps; the C++ engine in Mode::Fast must reproduce it exactly."""
from pathlib import Path
import numpy as np
import torch
import gguf
import scripts.la_reference as R

ROOT = Path(__file__).resolve().parent.parent
COORD_START, COORD_END, BOX_START, BOX_END = 151677, 152677, 151668, 151669
REF_START, REF_END, NONE_ID, EOS = 151672, 151673, 4064, 151645

def main():
    cfg, model, processor, inputs, spec = R.build_inputs()
    tok = processor.tokenizer
    with torch.no_grad():
        s = model.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"], image_grid_hws=inputs["image_grid_hws"],
                           tokenizer=tok, max_new_tokens=256, generation_mode="fast", use_cache=True)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids, skip_special_tokens=False) == s, "id round-trip mismatch"
    ids = np.array(ids, dtype=np.int64)
    grid = inputs["image_grid_hws"][0].tolist()      # [h_patches, w_patches]
    img_h, img_w = grid[0]*14, grid[1]*14            # patches * patch_size
    boxes, labels = [], []
    i, n, cur_label = 0, len(ids), ""
    while i < n:
        t = int(ids[i])
        if t == REF_START:
            j = i+1; lab = []
            while j < n and int(ids[j]) != REF_END: lab.append(int(ids[j])); j += 1
            cur_label = tok.decode(lab, skip_special_tokens=False); i = j+1; continue
        if t == BOX_START:
            j = i+1; coords = []
            while j < n and int(ids[j]) != BOX_END:
                v = int(ids[j])
                if COORD_START <= v <= COORD_END: coords.append(v - COORD_START)
                j += 1
            if len(coords) == 4:
                x1,y1,x2,y2 = coords     # x1,y1,x2,y2 order
                boxes.append([x1/1000*img_w, y1/1000*img_h, x2/1000*img_w, y2/1000*img_h])
                labels.append(cur_label)
            i = j+1; continue
        i += 1
    barr = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), np.float32)
    out = ROOT / "dumps" / "reference_fast.gguf"
    w = gguf.GGUFWriter(str(out), "la_fast")
    w.add_tensor("fast_token_ids", ids.astype(np.int32))
    w.add_tensor("fast_boxes", np.ascontiguousarray(barr))
    w.add_array("la_fast.box_labels", labels if labels else ["__none__"])
    w.add_uint32("la_fast.img_w", img_w); w.add_uint32("la_fast.img_h", img_h)
    w.add_uint32("la_fast.n_tokens", len(ids)); w.add_uint32("la_fast.n_boxes", len(boxes))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: n_tokens={len(ids)} n_boxes={len(boxes)} img={img_w}x{img_h}")
    print("first 16 ids:", ids[:16].tolist())
    print("boxes(px):", [[round(x,1) for x in b] for b in boxes], "labels:", labels)
    print("string:", s[:200])

if __name__ == "__main__":
    main()
