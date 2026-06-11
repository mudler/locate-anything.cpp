# LocateAnything-3B ggml Port — M6a (Engine completion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the C++ run the full LocateAnything pipeline on an ARBITRARY image file + text prompt — load/resize/normalize/patchify the image (any resolution), build the prompt token ids (BPE), run vision→projector→LM→decode→box-parse, and emit labeled pixel boxes — numerically matching the HF reference for a non-448 test image.

**Architecture:** Add the pieces the M0–M5 parity work assumed pre-computed: a PIL-equivalent image preprocessor (the riskiest new numeric op — Pillow bicubic a=−0.5 with mandatory antialias, distinct from the pos-emb bicubic), variable-grid threading into the MoonViT tower (replace the hardcoded 32×32), a byte-level Qwen2 BPE tokenizer (encode for prompts, decode for box labels), the prompt builder (chat template + image-token expansion), and an `la::Engine` that ties it all together. Each piece is gated tensor-for-tensor against a new non-448 reference dump.

**Tech Stack:** C++17, ggml (embedded), stb_image (vendored from rt-detr.cpp), CMake/CTest, Python (transformers/gguf) for the reference.

**Reference material (read before starting):**
- Preprocessing source of truth: `models/LocateAnything-3B/image_processing_locateanything.py` (`rescale` L46-71, `to_tensor`/`normalize` L73-77, `patchify` L79-86) + `processing_locateanything.py` (`replace_media_placeholder` L363-449). Restated inline; verify against these.
- stb + image load to mirror: `/home/mudler/_git/rt-detr.cpp/src/image_io.{cpp,hpp}` (stb impl TU, `image_load_file`), `/home/mudler/_git/rt-detr.cpp/third_party/stb/` (`stb_image.h`). (PNG save + renderer are M6b.)
- Existing repo: `la::VitEncoder` (`src/vit_encoder.{hpp,cpp}` — hardcodes 32×32 at L78/109/131, `build_patch_pos`/`block0`/`forward`; `merge_patches` already takes gh,gw), `la::Projector`, `la::LMForward` (`decode_hybrid`, `decode_greedy_resident`, `embed_and_splice`, `prefill_resident`), `la::boxes::parse_boxes` (`src/boxes.{hpp,cpp}`), `la::ModelLoader` (tokenizer KV: `locateanything.tokenizer.tokens`/`.merges`/`.token_types`), `la_parity`.
- Bicubic distinction: `src/vit_posemb.{hpp,cpp}` is torch bicubic (a=−0.75, no antialias) for POS-EMB — do NOT reuse for the image resize.

**Key constants:** patch_size 14, merge 2×2, pad multiple = 28, in_token_limit 25600, max grid side < 512 patches. Normalize = `raw/127.5 − 1`, RGB. Patch order `row*gw+col`, within-patch `[C,i,j]`. Image-token count `N = (gh/2)·(gw/2)`, `<IMG_CONTEXT>` = 151665. `image_grid_hws = (gh,gw) = (target_h/14, target_w/14)`, both even.

**Scope:** M6a ends with `la::Engine::locate(image_path, prompt) → vector<Box>` matching the reference on a non-448 image. M6b (quantize + C-API + CLI + renderer) is a separate plan.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/dump_preproc_reference.py` | Run the HF processor + model on a non-448 test image → `dumps/reference_preproc.gguf`: resized_chw, pixel_values, image_grid_hws, input_ids, merged_tokens (variable grid), boxes |
| `third_party/stb/stb_image.h` | Vendored image loader (from rt-detr.cpp) |
| `src/image_io.{hpp,cpp}` | `load_image_rgb(path)` (stb), `preprocess(image, ...) → {pixel_values, gh, gw}` (PIL-bicubic resize + normalize + patchify) |
| `src/pil_resize.{hpp,cpp}` | Pillow-equivalent bicubic (a=−0.5, two-pass, antialias) — the riskiest numeric op, isolated + gated |
| `src/tokenizer.{hpp,cpp}` | Byte-level Qwen2 BPE: `encode(text)→ids`, `decode(ids)→text`, special-token handling — loaded from the GGUF tokenizer KV |
| `src/prompt.{hpp,cpp}` | Build the full prompt token ids (chat template + image-token expansion for a given grid + query) |
| `src/vit_encoder.{hpp,cpp}` (modify) | Thread `(gh,gw)` through `build_patch_pos`/`block0`/`forward` (remove hardcoded 32×32) |
| `src/engine.{hpp,cpp}` | `la::Engine::load(gguf)`, `locate(image_path, prompt, mode) → vector<Box>` (preprocess→vision→projector→splice→decode→parse) |
| `src/lm.{hpp,cpp}` (modify) | Factor the repeated `Qwen2Hparams` init + shared resident chunk (the M5 de-dup carry item) |
| `tests/test_preproc.cpp`, `tests/test_tokenizer.cpp`, `tests/test_engine.cpp` | Gates |
| `tests/CMakeLists.txt`, `CMakeLists.txt` (modify) | Wire tests + new sources + stb |

---

## Task 1: Non-448 preprocessing + vision reference dump

Everything M6a gates against. Run the HF processor + model on a real non-square, non-448 image and dump every intermediate.

**Files:**
- Create: `scripts/dump_preproc_reference.py`

- [ ] **Step 1: Write `scripts/dump_preproc_reference.py`.**
```python
#!/usr/bin/env python3
"""Run the HF processor + model on a NON-448 test image; dump preprocessing +
vision + final boxes for gating the C++ engine on arbitrary-resolution images."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import gguf
import torchvision.transforms.functional as TF
import scripts.la_reference as R

ROOT = Path(__file__).resolve().parent.parent
TEST_IMG = "/home/mudler/_git/rt-detr.cpp/benchmarks/images/bus.jpg"   # non-square real image
PROMPT_TMPL = "Locate all the instances that matches the following description: {cats}."
CATS = "person</c>bus"
IMG_CONTEXT = 151665

def main():
    # READ scripts/la_reference.py and use its model+processor loader (the magi-faithful
    # AutoModel/AutoProcessor load). It exposes load_model()/build_inputs(); reuse whichever
    # gives (model, processor) — or inline the same AutoModel.from_pretrained(... magi ...)
    # + AutoProcessor.from_pretrained the file already does. Do NOT call build_inputs (it uses
    # the fixed 448 fixture); we need the processor applied to TEST_IMG below.
    cfg, model, processor = R.load_model()        # adapt to la_reference's actual loader name
    img = Image.open(TEST_IMG).convert("RGB")
    ip = processor.image_processor
    # Replicate _preprocess stages to dump the intermediate `resized_chw`:
    resized = ip.rescale(img)                          # PIL resize, multiple-of-28
    chw = ip.normalize(ip.to_tensor(resized))          # [3,H,W] normalized
    patches, grid_hw = ip.patchify(chw)                # [N,3,14,14], (gh,gw)
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    # Build the prompt + full processor inputs (chat template + token expansion):
    prompt = PROMPT_TMPL.format(cats=CATS)
    messages = [{"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[img], return_tensors="pt")
    input_ids = inputs["input_ids"][0].cpu().numpy().astype(np.int64)
    assert int((input_ids==IMG_CONTEXT).sum()) == (gh//2)*(gw//2)
    # Run vision tower -> merged_tokens for this grid (faithful path via model internals).
    # Reuse the M0 dump's vision-capture approach; capture merged vision tokens.
    with torch.no_grad():
        vit = model.vision_model(inputs["pixel_values"], inputs["image_grid_hws"])
        merged = model.vision_model.patch_merger(vit, inputs["image_grid_hws"], [2,2]) \
                 if hasattr(model.vision_model,"patch_merger") else None
    # (If the merged-token capture differs, hook the same module M0's dump_reference used.)
    out = ROOT/"dumps"/"reference_preproc.gguf"
    w = gguf.GGUFWriter(str(out), "la_preproc")
    w.add_tensor("resized_chw", np.ascontiguousarray(chw.float().numpy()))         # [3,H,W]
    w.add_tensor("pixel_values", np.ascontiguousarray(patches.float().numpy()))    # [N,3,14,14]
    w.add_tensor("input_ids", input_ids.astype(np.int32))
    if merged is not None:
        m = (torch.cat(merged,0) if isinstance(merged,list) else merged).float().numpy()
        w.add_tensor("merged_tokens", np.ascontiguousarray(m))                      # [N,4608]
    w.add_uint32("la_preproc.gh", gh); w.add_uint32("la_preproc.gw", gw)
    w.add_uint32("la_preproc.target_h", gh*14); w.add_uint32("la_preproc.target_w", gw*14)
    w.add_uint32("la_preproc.n_img_tokens", (gh//2)*(gw//2))
    w.add_string("la_preproc.image_path", TEST_IMG)
    w.add_string("la_preproc.prompt", prompt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: grid={gh}x{gw} N_img={(gh//2)*(gw//2)} input_ids={input_ids.shape} pix={patches.shape}")
```
NOTE: READ `scripts/la_reference.py` for the exact model/processor load API (it may only expose `build_inputs`; factor a `load_model`/`load_processor` or inline the AutoModel/AutoProcessor load with the magi-faithful settings). VERIFY the vision/merged-token capture matches how `dump_reference.py` (M0) captured `merged_tokens` — reuse that hook/path so the reference is faithful. The point of this task is a correct dump; if the merged-token capture is unclear, capture it via the same forward-hook M0 used on the vision model + projector. Also dump the final boxes (run a short `generate(generation_mode="slow")` and parse boxes like `dump_slow_reference.py`) into `slow_boxes`/labels so Task 6 can gate the end-to-end output — OR defer box-gating to Task 6 using a fresh generate there.

- [ ] **Step 2: Generate it.**
```bash
. .venv/bin/activate
python -m scripts.dump_preproc_reference
```
Expected: prints a grid like `grid=GHxGW` where GH≠32 or GW≠32 (bus.jpg is ~810×1080 → a non-square grid), `N_img=(gh/2)*(gw/2)`, and the input_ids shape. The gguf is gitignored. CONFIRM the grid is genuinely non-448 (so it exercises variable-resolution) and `gh,gw` are both even and `<512`.

- [ ] **Step 3: Commit (script only).**
```bash
git add scripts/dump_preproc_reference.py
git commit -m "M6a: non-448 preprocessing + vision reference dump"
```
Author `mudler <mudler@localai.io>`; end every M6a commit with:
```

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: Image load + PIL-bicubic resize + normalize + patchify → gate `resized_chw` + `pixel_values`

The PIL bicubic resampler (a=−0.5, mandatory antialias on downscale) is the riskiest new numeric op — isolate it in `src/pil_resize.cpp` and gate `resized_chw` directly.

**Files:**
- Create: `third_party/stb/stb_image.h` (copy from rt-detr), `src/image_io.{hpp,cpp}`, `src/pil_resize.{hpp,cpp}`
- Create: `tests/test_preproc.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: Vendor stb + the resize arithmetic + PIL bicubic.**
```bash
mkdir -p third_party/stb
cp /home/mudler/_git/rt-detr.cpp/third_party/stb/stb_image.h third_party/stb/stb_image.h
```
Write `src/pil_resize.hpp`:
```cpp
#pragma once
#include <vector>
#include <cstdint>
namespace la {
// Pillow-equivalent bicubic resize of an RGB uint8 HWC image (a=-0.5, two-pass H then V,
// support-scaled by the reduction ratio = antialias on downscale). Returns RGB uint8 HWC.
std::vector<uint8_t> pil_bicubic_resize(const std::vector<uint8_t>& src, int sw, int sh,
                                        int dw, int dh);
}
```
Implement `src/pil_resize.cpp` porting Pillow's `ImagingResample` (the horizontal+vertical passes with the cubic filter `a=-0.5`, support=2.0 scaled by `max(1, src/dst)` per axis, precomputed coefficient bounds + kernels, clamped edges, round-to-uint8). READ Pillow's Resample.c algorithm (the cubic filter, `precompute_coeffs`, `ImagingResampleHorizontal_8bpc`/`Vertical`) and reproduce it — this is the load-bearing op. (Gate it on `resized_chw` in Step 4; if off, the kernel/support/coeff-normalization is the suspect.)

- [ ] **Step 2: `src/image_io.{hpp,cpp}` — load + the resize/normalize/patchify pipeline.**
hpp:
```cpp
#pragma once
#include <vector>
#include <string>
#include <cstdint>
namespace la {
struct Image { int w=0, h=0; std::vector<uint8_t> rgb; };   // HWC RGB uint8
bool load_image_rgb(const std::string& path, Image& out);   // stb
// Compute the resize target (gh,gw) per image_processing_locateanything.rescale.
void preproc_target(int w0, int h0, int& target_w, int& target_h);  // multiple-of-28
struct Preprocessed { std::vector<float> pixel_values; int gh=0, gw=0; };  // [N*3*14*14] (c,i,j) per patch
// Full pipeline: resize (pil_bicubic) -> normalize (raw/127.5-1, RGB) -> patchify (row*gw+col, [c,i,j]).
bool preprocess(const Image& img, Preprocessed& out);
}
```
cpp: `load_image_rgb` via `stbi_load` (force 3 channels) into Image. `preproc_target`: implement rescale's arithmetic EXACTLY — (A) if `(w0/14)*(h0/14) > 25600`: `scale=sqrt(25600/((w0/14)*(h0/14)))`, `w1=int(w0*scale), h1=int(h0*scale)` (truncate) and that's the intermediate (PIL resize to w1×h1 first); (B) `target_w=ceil(w1/28)*28, target_h=ceil(h1/28)*28`; assert both `/14 < 512`. NOTE the two-step resize: if step A fires, Pillow resizes to (w1,h1) THEN to (target). If A doesn't fire, just resize to (target). Replicate BOTH resizes through pil_bicubic_resize (compose or do two passes to match Pillow exactly — a single resize to target is NOT identical when A fired). `preprocess`: pil_bicubic to target → normalize `(c,i,j)` patch-major: for patch token `t = row*gw+col`, write `out[t*588 + c*196 + i*14 + j] = rgb[((row*14+i)*target_w + (col*14+j))*3 + c]/127.5 - 1`. gh=target_h/14, gw=target_w/14.

- [ ] **Step 3: `tests/test_preproc.cpp` — gate resized_chw + pixel_values + grid.**
```cpp
#include "image_io.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
int main(){
    const char* pp=std::getenv("LA_TEST_PREPROC");
    if(!pp){std::fprintf(stderr,"skip\n");return 77;}
    // image path + expected grid from the dump KV
    std::string img_path = la_parity::load_kv_str(pp,"la_preproc.image_path");
    int ref_gh=(int)la_parity::load_kv_u32(pp,"la_preproc.gh"), ref_gw=(int)la_parity::load_kv_u32(pp,"la_preproc.gw");
    la::Image im; if(!la::load_image_rgb(img_path, im)) return 1;
    la::Preprocessed P; if(!la::preprocess(im, P)) return 1;
    int ok = (P.gh==ref_gh && P.gw==ref_gw);
    std::printf("grid got %dx%d ref %dx%d\n", P.gh,P.gw,ref_gh,ref_gw);
    // pixel_values gate
    std::vector<float> ref; std::vector<int64_t> rs; la_parity::load_baseline(pp,"pixel_values",ref,rs);
    ok &= la_parity::compare(P.pixel_values, ref, "pixel_values", 2e-2f, 2e-2f);   // PIL resampler tolerance
    return ok?0:1;
}
```
ALSO add a resized_chw gate (build the normalized [3,H,W] from pil_bicubic and compare to `resized_chw` — this isolates the resampler from patchify). Tolerance: the PIL resampler should match closely; start at 2e-2 and tighten if it comes in tight (report the actual max-diff). If it's far off, the bicubic kernel/antialias is wrong — debug on resized_chw first (it's the resampler in isolation).

- [ ] **Step 4: Wire (`la_add_test(test_preproc)`, `LA_TEST_PREPROC=${CMAKE_SOURCE_DIR}/dumps/reference_preproc.gguf`; add `src/image_io.cpp`,`src/pil_resize.cpp` to LA_SOURCES; add stb include dir), build, run.** Expected: grid matches exactly, resized_chw + pixel_values within tolerance. Commit:
```bash
git add third_party/stb src/image_io.hpp src/image_io.cpp src/pil_resize.hpp src/pil_resize.cpp tests/test_preproc.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M6a: image load + PIL-bicubic resize/normalize/patchify -> preproc parity"
```

---

## Task 3: Variable-grid threading in the MoonViT tower → gate `merged_tokens` (non-448)

Replace the hardcoded 32×32 in `vit_encoder` with the real `(gh,gw)` from preprocessing; gate the vision output on the non-448 image.

**Files:**
- Modify: `src/vit_encoder.{hpp,cpp}`
- Modify: `tests/test_preproc.cpp` (add the merged_tokens gate)

- [ ] **Step 1: Thread `(gh,gw)` through `src/vit_encoder.cpp`.** Change `build_patch_pos`/`block0`/`forward` to take `(gh,gw)` (or read from a member set per image) instead of the literal `32,32` at L78/109/131. Specifically: `n_tok = gh*gw`; `bicubic_pos_emb(pe_src, 64, 64, 1152, gh, gw)`; `build_rope_tables(gh, gw, heads, hd, theta)`; `merge_patches(vf, gh, gw, 1152)`. Update `VitEncoder::forward`/`block0`/`patch_and_pos` signatures to accept `gh,gw` (the Engine passes them from preprocessing). KEEP the existing 448 fixture tests passing (they pass gh=gw=32 explicitly).

- [ ] **Step 2: Gate `merged_tokens` for the non-448 image.** In `tests/test_preproc.cpp`, after preprocessing: load the model GGUF (`LA_TEST_GGUF`), build a `VitEncoder`, run `forward(P.pixel_values, P.gh, P.gw, ...)` → merge → compare to the dump's `merged_tokens` (shape `[(gh/2)*(gw/2), 4608]`). Gate atol 3e-2 (the M2 ViT tolerance + the preproc tolerance compound — report actual). This proves the variable-grid vision path.
```cpp
    // (in test_preproc.cpp, needs LA_TEST_GGUF)
    la::ModelLoader ml; ml.load(std::getenv("LA_TEST_GGUF"));
    la::Backend be; la::VitEncoder vit(ml, be);
    std::vector<float> vfinal; std::vector<std::vector<float>> caps;
    vit.forward(P.pixel_values, P.gh, P.gw, vfinal, {}, caps);
    std::vector<float> merged = la::merge_patches(/*token-major*/ to_token_major(vfinal, 1152, P.gh*P.gw), P.gh, P.gw, 1152);
    std::vector<float> rmt; std::vector<int64_t> ms; la_parity::load_baseline(pp,"merged_tokens",rmt,ms);
    ok &= la_parity::compare(merged, rmt, "merged_tokens_var", 3e-2f, 3e-2f);
```
(Adapt to the actual `forward` signature you choose; reuse `to_token_major` from the M2 tests if present, else inline the transpose.)

- [ ] **Step 3: Build + run + debug.** Both the 448 fixture tests (test_vit_encoder) AND the non-448 merged_tokens gate must pass. DEBUG: if the non-448 fails but 448 passes, the grid threading missed a spot (pos-emb interp target, rope grid, or merger dims) — print P.gh/P.gw and check each call site uses them. Run the full suite. Commit:
```bash
git add src/vit_encoder.hpp src/vit_encoder.cpp tests/test_preproc.cpp
git commit -m "M6a: thread variable (gh,gw) through MoonViT tower -> non-448 merged_tokens parity"
```

---

## Task 4: Byte-level Qwen2 BPE tokenizer (encode + decode) → gate round-trip + known text

The C++ has no tokenizer (prompts were pre-tokenized). Build a byte-level BPE from the GGUF tokenizer KV: `encode(text)→ids`, `decode(ids)→text`, with atomic special/added tokens.

**Files:**
- Create: `src/tokenizer.{hpp,cpp}`
- Create: `tests/test_tokenizer.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: `src/tokenizer.hpp`.**
```cpp
#pragma once
#include "model_loader.hpp"
#include <vector>
#include <string>
#include <cstdint>
namespace la {
class Tokenizer {
public:
    bool load(const ModelLoader& ml);            // reads locateanything.tokenizer.tokens/.merges/.token_types
    std::vector<int32_t> encode(const std::string& text) const;   // byte-level BPE, special tokens atomic
    std::string decode(const std::vector<int32_t>& ids) const;    // ids -> bytes -> utf8
    int32_t token_to_id(const std::string& piece) const;          // exact piece lookup (for special tokens)
private:
    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string,int32_t> piece_to_id_;
    std::unordered_map<std::string,int> merge_rank_;   // "a b" -> rank
    std::vector<int> token_types_;                     // 1=normal 4=control(added/special)
};
}
```

- [ ] **Step 2: `src/tokenizer.cpp` — byte-level BPE (the GPT-2/Qwen2 algorithm).** Implement:
  - `load`: read `locateanything.tokenizer.tokens` (id→piece string array), `.merges` (space-separated pairs → rank = index), `.token_types`. Build `piece_to_id_`, `merge_rank_`. The vocab strings use the GPT-2 byte-encoding (e.g. `Ġ`=space). Build the GPT-2 byte→unicode table (the standard 256-entry map: printable bytes map to themselves, others to U+0100+offset) and its inverse for decode.
  - `encode(text)`: (1) split off ANY added/special token (token_type==4 / control) by exact substring match FIRST — the added tokens (`<IMG_CONTEXT>`, `<box>`, `<0>`..`<1000>`, `<|im_start|>`, etc.) are atomic and must not be BPE-merged. Scan the text for the longest special-token piece at each position; between specials, BPE the plain text. (2) For a plain-text run: GPT-2 byte-encode each UTF-8 byte → the unicode-mapped chars, then apply BPE merges by lowest rank repeatedly (standard BPE merge loop over the char sequence), then map each resulting merged piece → id via `piece_to_id_`. (Qwen2 has no word pre-tokenization regex beyond the byte-level; reproduce llama.cpp's gpt2 BPE — read `/home/mudler/_git/llama.cpp/src/llama-vocab.cpp` `llm_tokenizer_bpe` for the exact merge loop + byte mapping if needed.)
  - `decode(ids)`: map each id→piece, concatenate, then GPT-2 byte-decode the unicode chars back to bytes → utf8 string. Special tokens decode to their literal piece (or empty for non-printable control — match the reference's `skip_special_tokens=False` which keeps them).

- [ ] **Step 3: `tests/test_tokenizer.cpp` — gate encode/decode.**
```cpp
#include "tokenizer.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* pp=std::getenv("LA_TEST_PREPROC");
    if(!gguf||!pp){std::fprintf(stderr,"skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Tokenizer tok; if(!tok.load(ml)) return 1;
    int ok=1;
    // round-trip on plain ascii + the box/coord special tokens
    for(const char* s : {"Locate all the instances", "person", "bus"}){
        auto ids = tok.encode(s); auto back = tok.decode(ids);
        ok &= (back==s); if(back!=s) std::printf("roundtrip fail: '%s' -> '%s'\n", s, back.c_str());
    }
    // special tokens atomic
    int img = tok.token_to_id("<IMG_CONTEXT>"); ok &= (img==151665);
    int b0  = tok.token_to_id("<0>"); ok &= (b0==151677);
    std::printf("tokenizer ok=%d (IMG_CONTEXT=%d <0>=%d)\n", ok, img, b0);
    return ok?0:1;
}
```
ALSO gate against the reference: decode the dump's `input_ids` and assert it contains the prompt text; and encode the known prompt-fragment and check the ids appear in the reference input_ids. The decisive prompt-level gate is in Task 5 (full input_ids match). Build, run to green. Commit:
```bash
git add src/tokenizer.hpp src/tokenizer.cpp tests/test_tokenizer.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M6a: byte-level Qwen2 BPE tokenizer (encode/decode) -> round-trip parity"
```

---

## Task 5: Prompt builder → gate full `input_ids` (non-448 grid)

Build the full prompt token ids for an arbitrary query + the real grid's image-token count, matching the reference processor's chat-template + token expansion.

**Files:**
- Create: `src/prompt.{hpp,cpp}`
- Modify: `tests/test_tokenizer.cpp` (add the input_ids gate)

- [ ] **Step 1: `src/prompt.{hpp,cpp}`.** Build the fully-expanded text then encode it:
```cpp
// build_prompt(tok, gh, gw, query) -> input_ids reproducing the HF processor output.
std::vector<int32_t> build_prompt(const Tokenizer& tok, int gh, int gw, const std::string& query);
```
The expanded text (verified vs processing_locateanything.py + chat_template.json):
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image 1><img>{<IMG_CONTEXT> × N}</img>QUERY<|im_end|>\n<|im_start|>assistant\n
```
where `N = (gh/2)*(gw/2)`. Build this string (the `<|im_start|>`/`<|im_end|>`/`<img>`/`</img>`/`<IMG_CONTEXT>` are atomic special tokens — the tokenizer handles them; `<image 1>` is literal text that gets BPE'd) and `return tok.encode(expanded)`. VERIFY the exact chat-template whitespace/newlines against `chat_template.json` (the `\n` placements matter for the token ids).

- [ ] **Step 2: Gate in `tests/test_tokenizer.cpp`.**
```cpp
    int gh=(int)la_parity::load_kv_u32(pp,"la_preproc.gh"), gw=(int)la_parity::load_kv_u32(pp,"la_preproc.gw");
    std::string query = la_parity::load_kv_str(pp,"la_preproc.prompt");
    std::vector<int32_t> built = la::build_prompt(tok, gh, gw, query);
    std::vector<int32_t> ref; la_parity::load_baseline_i32(pp,"input_ids",ref);
    int pok = ((int)built.size()==(int)ref.size());
    for(size_t i=0;i<ref.size() && pok;++i){ if(built[i]!=ref[i]){ std::printf("prompt mismatch@%zu got=%d ref=%d\n",i,built[i],ref[i]); pok=0; } }
    std::printf("prompt build: input_ids match=%d (%zu vs %zu)\n", pok, built.size(), ref.size());
    ok &= pok;
```
Exact id match vs the reference input_ids (incl. the correct N=`(gh/2)*(gw/2)` IMG_CONTEXT tokens). DEBUG a mismatch: print the divergence index — likely the chat-template whitespace, the `<image 1>` literal, or a BPE merge on the query. Build, run to green. Commit:
```bash
git add src/prompt.hpp src/prompt.cpp tests/test_tokenizer.cpp
git commit -m "M6a: prompt builder (chat template + image-token expansion) -> input_ids parity"
```

---

## Task 6: `la::Engine` end-to-end + lm.cpp de-dup → gate boxes (non-448)

Tie it together: image path + prompt → boxes. Gate the final labeled boxes against the reference for the non-448 image. Fold in the M5 `lm.cpp` de-dup.

**Files:**
- Create: `src/engine.{hpp,cpp}`
- Modify: `src/lm.{hpp,cpp}` (factor the repeated Qwen2Hparams init + shared resident chunk)
- Create: `tests/test_engine.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: lm.cpp de-dup (the M5 carry).** Factor the `Qwen2Hparams` init (repeated in run_layer0/forward/run_resident_causal/mtp_block_forward) into a `hparams_from_config()` helper, and unify the shared resident layer-loop where run_resident_causal/mtp_block_forward overlap (they differ only in mask/positions + the final 1-col vs 6-col slice). Keep behavior bit-identical — run test_decode + test_mtp after to confirm no regression. (Do this first so the Engine wraps a clean surface.)

- [ ] **Step 2: `src/engine.{hpp,cpp}`.**
```cpp
#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "tokenizer.hpp"
#include "boxes.hpp"
#include <memory>
#include <string>
#include <vector>
namespace la {
class Engine {
public:
    static std::unique_ptr<Engine> load(const std::string& gguf_path, int n_threads=0);
    enum class Mode { Hybrid, Slow };
    // image file + open-vocab query -> labeled pixel boxes.
    std::vector<Box> locate(const std::string& image_path, const std::string& query,
                            Mode mode = Mode::Hybrid, int max_new=256);
private:
    ModelLoader ml_; Backend be_; Tokenizer tok_;
    // VitEncoder/Projector/LMForward constructed per-call or held.
};
}
```
`locate`: `load_image_rgb` → `preprocess` (→ pixel_values, gh, gw) → `build_prompt(tok_, gh, gw, query)` → `VitEncoder::forward(pixel_values, gh, gw)` → `merge_patches` → `Projector::project` → `LMForward::decode_hybrid`(or decode_greedy_resident for Slow) with the projected vision tokens + prompt ids → `parse_boxes(generated_ids, target_w, target_h, tok_.id_to_piece)` (labels via the tokenizer decode). Return boxes (with proper labels now that the BPE detokenizer exists). NOTE: `decode_hybrid`/`decode_greedy_resident` take `prompt_ids` + `projected_host` — wire the variable-grid projected tokens (count = N = (gh/2)*(gw/2)) and the prompt ids (which have N IMG_CONTEXT slots). The splice in embed_and_splice overwrites those N slots — confirm the count matches.

- [ ] **Step 3: `tests/test_engine.cpp` — gate end-to-end boxes.** Load the engine, run `locate(image_path, prompt, Slow)` on the non-448 image, and compare boxes to the reference (the dump's `slow_boxes`/labels, or generate a fresh reference in Task 1's dump). Gate: box count matches, coords within a few px (the full pipeline compounds preproc + vision + LM tolerances — use ~5px or relative), labels match (now that decode works). Print the boxes. (If end-to-end coords drift more than expected, the per-stage gates (Tasks 2-5) localize it; the Engine just wires them.)
```cpp
    auto eng = la::Engine::load(gguf, 0);
    auto boxes = eng->locate(img_path, prompt, la::Engine::Mode::Slow);
    // compare to reference boxes (count, coords within tol, labels)
    ...
    std::printf("engine: %zu boxes\n", boxes.size());
```
Build, run. Then the FULL suite (all prior gates + the new ones) green. Commit (M6a complete):
```bash
cd build && ctest --output-on-failure; cd ..
git add src/engine.hpp src/engine.cpp src/lm.hpp src/lm.cpp tests/test_engine.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M6a: la::Engine end-to-end (image+prompt -> boxes) + lm.cpp de-dup (M6a complete)"
```

---

## Milestone exit criteria

**M6a done when:** the C++ runs the full pipeline on an arbitrary (non-448) image file + text prompt — `tests/{test_preproc,test_tokenizer,test_engine}.cpp` pass: preprocessing reproduces the HF `resized_chw`/`pixel_values`/grid, the variable-grid MoonViT reproduces `merged_tokens`, the BPE tokenizer round-trips and the prompt builder reproduces the reference `input_ids`, and `la::Engine::locate` produces labeled boxes matching the reference — all green under ctest, full suite green. The 448 fixture tests still pass (no regression from grid threading / lm.cpp de-dup).

## Roadmap (M6b — next plan)

- **M6b — Packaging.** C++ `quantize` subcommand + `scripts/quantize_gguf.py` (allowlist: quantize LM `attn_{q,k,v,o}`/`ffn_{gate,up,down}`/`output`; keep MoonViT/projector/norms/embeds f32/f16) → re-gate boxes per quant type (f16/q8_0 near-lossless, q4_k looser). Flat C-API (`include/la_capi.h` + `src/la_capi.cpp`: opaque `la_ctx`, `la_capi_load`/`locate_path`/`locate_buffer`/`free_string`/`last_error`/`abi_version` + per-detection accessors, all try/catch→last_error). CLI `locate-anything-cli detect --model --input --prompt [--annotated] [--threads]` (+ `info`, `quantize`) mirroring rt-detr's `main.cpp`/`cli.cpp`, with the box renderer (`src/visualize.cpp` from rt-detr, label→color hash) + PNG save (stb_image_write). CMake `LA_SHARED` (static-ggml into the .so via PIC) + `LA_BUILD_CLI`. Then optionally the LocalAI purego backend (separate effort).
