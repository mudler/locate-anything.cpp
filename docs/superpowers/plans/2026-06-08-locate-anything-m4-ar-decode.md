# LocateAnything-3B ggml Port — M4 (AR decode + box parse) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the slow/autoregressive greedy token stream in C++ (prefill → argmax → append → stop on EOS) and parse the coordinate tokens into labeled bounding boxes, numerically matching a freshly-generated `generation_mode="slow"` reference.

**Architecture:** First build the AR decode loop on the already-bit-exact eager `qwen2_layer_forward` by re-running the full causal prefill over the growing sequence each step (the "reprefill" path — zero new infra, correct by construction) and gate it on a short prefix. Add a pure coord→box parser gated on the full reference stream. Then port vibevoice's `ResidentKV` KV-cache for a fast full-sequence decode, proven bit-identical to the reprefill path.

**Tech Stack:** C++17, ggml (embedded), CMake/CTest, Python (transformers/gguf) for the slow reference.

**Reference material (read before starting):**
- AR decode + parse facts: `models/LocateAnything-3B/modeling_locateanything.py` (`generate`, slow path L351-517) and `generate_utils.py` (`sample_tokens` L135-168). Restated inline; verify against these.
- KV-cache port source: `/home/mudler/_git/vibevoice.cpp/src/qwen2.cpp` (`ResidentKV` L205-244, `qwen2_layer_forward_resident` L250-358), `src/qwen2.hpp` L78-137, driver `src/vibevoice_asr.cpp:519-805` (prefill+step + the **interleaved write-expand** gotcha L591-602), buffer alloc `src/backend.cpp:151-155`.
- Existing repo (M3): `la::LMForward` (`src/lm.{hpp,cpp}`: `embed_and_splice`, `forward`), `la::qwen2_layer_forward` (`src/qwen2.{hpp,cpp}`, eager bit-exact), `la::Backend` (`compute`/`forward_capture`/`add_graph_input_nd`/`add_int32_input_nd`/`capture`/`GraphInputPool`, persistent gallocr — private `impl_->backend`), `la::ModelLoader` (`ml.config()`, `ml.tensor`), `la_parity` harness.
- Token ids (config.json / design spec): box_start 151668, box_end 151669, coord_start 151677, coord_end 152677, ref_start 151672, ref_end 151673, none 4064, **eos/im_end 151645**, image 151665. Coord: `coord_norm = id − 151677` ∈ [0,1000]; `pixel = coord_norm/1000 × dim`.

**Decode facts (verified):** slow path is plain greedy — `temperature=0` → `x0 = argmax(logits[:,-1,:])`, no top_p/top_k/repetition_penalty, no in-loop box handling (box structure is emergent). Terminate on token == 151645 or at `prompt_len + max_new_tokens`. The reference must be generated with `generation_mode="slow"` (the M0 hybrid dump will NOT match).

**Coord order (resolved):** within-box order is **x1, y1, x2, y2** (README/design convention). The vendored code comments say x1,x2,y1,y2 but never actually map to pixels; the first generated box `<7><113><495><990>` under x1,y1,x2,y2 yields a plausible left-side cat box at 448px (≈ (3,50)–(222,443)), while x1,x2,y1,y2 is degenerate — so x1,y1,x2,y2 it is. x→image width, y→image height. The parser asserts `x1≤x2, y1≤y2, 0≤coords≤dim` as a sanity check.

**Scope:** M4 ends with the C++ producing the slow token stream + labeled boxes matching the slow reference. Hybrid Parallel-Box-Decoding is M5; quantize + CLI + C-API is M6.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/dump_slow_reference.py` | Generate `generation_mode="slow"` greedy stream + parsed boxes → `dumps/reference_slow.gguf` (token ids i32, box array f32, image dims) |
| `src/boxes.{hpp,cpp}` | Pure coord-token → labeled-box parser (no ggml): `parse_boxes(ids, vocab→str?, img_w, img_h)` |
| `src/lm.{hpp,cpp}` (modify) | Add `decode_greedy_reprefill(...)` (B1) and later `decode_greedy_resident(...)` |
| `src/qwen2.{hpp,cpp}` (modify) | Add `ResidentKV` + `qwen2_layer_forward_resident` (ported from vibevoice) |
| `src/backend.{hpp,cpp}` (modify) | Expose backend (`allocate_ctx_tensors`) + ordered no-readback graph roots (`add_graph_root`) |
| `tests/test_decode.cpp` | Gate AR token stream (prefix), box parse (full ref stream), resident==reprefill, full resident stream |
| `tests/CMakeLists.txt`, `CMakeLists.txt` (modify) | Wire test + new sources |

---

## Task 1: Generate the slow-mode reference (token stream + boxes)

The M0 reference is hybrid; M4 gates against slow/greedy. Produce it once.

**Files:**
- Create: `scripts/dump_slow_reference.py`

- [ ] **Step 1: Write `scripts/dump_slow_reference.py`.**
```python
#!/usr/bin/env python3
"""Generate the generation_mode='slow' greedy reference (token stream + boxes)
into dumps/reference_slow.gguf, for gating the C++ AR decode + box parser."""
import re
from pathlib import Path
import numpy as np
import torch
import gguf
import scripts.la_reference as R

ROOT = Path(__file__).resolve().parent.parent
COORD_START, BOX_START, BOX_END = 151677, 151668, 151669
REF_START, REF_END, NONE_ID, EOS = 151672, 151673, 4064, 151645

def main():
    cfg, model, processor, inputs, spec = R.build_inputs()
    tok = processor.tokenizer
    with torch.no_grad():
        s = model.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"], image_grid_hws=inputs["image_grid_hws"],
                           tokenizer=tok, max_new_tokens=256, generation_mode="slow", use_cache=True)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids, skip_special_tokens=False) == s, "id round-trip mismatch"
    ids = np.array(ids, dtype=np.int64)
    # image dims (the fixture is 448x448; read from the processed grid to be exact)
    grid = inputs["image_grid_hws"][0].tolist()           # [h_patches, w_patches]
    img_h, img_w = grid[0]*14, grid[1]*14                 # patches*patch_size
    # parse boxes (x1,y1,x2,y2 order) into pixels, label = preceding <ref>..</ref>
    boxes = []  # each: [x1,y1,x2,y2] pixels (label kept separately as a string list)
    labels = []
    i, n = 0, len(ids)
    cur_label = ""
    while i < n:
        t = int(ids[i])
        if t == REF_START:
            j = i+1; lab=[]
            while j < n and int(ids[j]) != REF_END: lab.append(int(ids[j])); j += 1
            cur_label = tok.decode(lab, skip_special_tokens=False); i = j+1; continue
        if t == BOX_START:
            # collect coord tokens until box_end
            j = i+1; coords=[]
            while j < n and int(ids[j]) != BOX_END:
                v = int(ids[j])
                if COORD_START <= v <= 152677: coords.append(v - COORD_START)
                j += 1
            if len(coords) == 4:
                x1,y1,x2,y2 = coords  # x1,y1,x2,y2 order
                boxes.append([x1/1000*img_w, y1/1000*img_h, x2/1000*img_w, y2/1000*img_h])
                labels.append(cur_label)
            i = j+1; continue
        i += 1
    barr = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), np.float32)
    out = ROOT / "dumps" / "reference_slow.gguf"
    w = gguf.GGUFWriter(str(out), "la_slow")
    w.add_tensor("slow_token_ids", ids.astype(np.int32))
    w.add_tensor("slow_boxes", np.ascontiguousarray(barr))   # [n_box,4] pixels
    w.add_array("la_slow.box_labels", labels if labels else ["__none__"])
    w.add_uint32("la_slow.img_w", img_w); w.add_uint32("la_slow.img_h", img_h)
    w.add_uint32("la_slow.n_tokens", len(ids)); w.add_uint32("la_slow.n_boxes", len(boxes))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: n_tokens={len(ids)} n_boxes={len(boxes)} img={img_w}x{img_h}")
    print("first 16 ids:", ids[:16].tolist())
    print("boxes(px):", [[round(x,1) for x in b] for b in boxes], "labels:", labels)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate it.**
```bash
. .venv/bin/activate
python scripts/dump_slow_reference.py
```
Expected: prints `n_tokens=…`, `n_boxes=…` (≥2 for cat+remote), `img=448x448`, the first ids beginning with `151672` (`<ref>`), and plausible pixel boxes (x1<x2, y1<y2, within 0..448). NOTE: slow greedy generates up to 256 tokens one at a time on CPU — this can take **10-20 minutes**. Be patient. The output gguf is gitignored. If the boxes look degenerate (y1>y2 etc.), the coord order assumption is wrong — STOP and report the raw coord tokens so the order can be reconciled.

- [ ] **Step 3: Commit (script only; reference_slow.gguf is gitignored).**
```bash
git add scripts/dump_slow_reference.py
git commit -m "M4: generate slow-mode greedy reference (token stream + boxes)"
```
Author `mudler <mudler@localai.io>`; end every M4 commit message with:
```

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: AR greedy decode (reprefill) → gate the token stream (prefix)

Implement the decode loop on the existing bit-exact eager prefill: each step re-runs `LMForward::forward`-style full causal prefill over the growing token sequence, takes the last-position argmax, appends, stops on EOS. Gate a short prefix against the slow reference (full stream is gated by the resident path in Task 5; reprefill is O(n³) so we cap it here).

**Files:**
- Modify: `src/lm.hpp`, `src/lm.cpp`
- Create: `tests/test_decode.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add `decode_greedy_reprefill` to `src/lm.hpp`.**
```cpp
    // Greedy AR decode via full reprefill each step (correctness oracle; O(n^3)).
    // prompt_ids: the 297 prompt tokens. projected_host: [2048*256] vision tokens.
    // Appends generated ids to `out_ids` until EOS (151645) or `max_new` tokens.
    bool decode_greedy_reprefill(const std::vector<int32_t>& prompt_ids,
                                 const std::vector<float>& projected_host,
                                 int max_new, std::vector<int32_t>& out_ids);
```

- [ ] **Step 2: Implement it in `src/lm.cpp`.** Reuse `forward` (it returns last-position logits). Each step: build the current sequence (prompt + generated), embed+splice (the vision tokens only occupy the original 256 image positions — generated tokens are plain text, so splice only applies to the prompt's image slots), run `forward`, argmax, append, stop.
```cpp
bool LMForward::decode_greedy_reprefill(const std::vector<int32_t>& prompt_ids,
                                        const std::vector<float>& projected_host,
                                        int max_new, std::vector<int32_t>& out_ids){
    const int EOS = 151645;
    std::vector<int32_t> seq = prompt_ids;       // grows as we decode
    out_ids.clear();
    for (int step=0; step<max_new; ++step){
        std::vector<float> logits; std::vector<std::vector<float>> caps;
        if(!forward(seq, projected_host, logits, {}, caps)) return false;  // logits at last pos
        int best=0; for(int i=1;i<(int)logits.size();++i) if(logits[i]>logits[best]) best=i;
        out_ids.push_back(best);
        if(best==EOS) break;
        seq.push_back(best);
    }
    return true;
}
```
NOTE: `forward` already embeds+splices the WHOLE `seq` and runs the full causal prefill; the splice overwrites exactly the 256 `id==151665` positions (which only exist in the prompt prefix — generated tokens are never 151665), so passing the growing `seq` is correct. `forward` slices the last token's logits, which is the next-token distribution. This is the bit-exact eager path, reused verbatim — no new graph code.

- [ ] **Step 3: Write `tests/test_decode.cpp` (token-stream prefix gate).**
```cpp
#include "lm.hpp"
#include "projector.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* base=std::getenv("LA_TEST_BASELINE");
    const char* slow=std::getenv("LA_TEST_SLOW");
    if(!gguf||!base||!slow){std::fprintf(stderr,"env unset; skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Backend be; la::LMForward lm(ml, be);
    std::vector<int32_t> ids; la_parity::load_baseline_i32(base,"input_ids",ids);   // 297 prompt
    std::vector<float> proj; std::vector<int64_t> ps;
    la_parity::load_baseline(base,"projected_tokens",proj,ps);                       // [256,2048]
    // reference slow token ids
    std::vector<int32_t> ref; la_parity::load_baseline_i32(slow,"slow_token_ids",ref);
    const int N = 24;   // short prefix (covers the first box) — reprefill is O(n^3)
    std::vector<int32_t> got;
    if(!lm.decode_greedy_reprefill(ids, proj, N, got)) return 1;
    int ok=1, m=(int)std::min((size_t)N, ref.size());
    for(int i=0;i<m;++i){ if(got[i]!=ref[i]){ std::printf("mismatch@%d got=%d ref=%d\n",i,got[i],ref[i]); ok=0; } }
    std::printf("decode prefix match=%d (checked %d tokens)\n", ok, m);
    return ok?0:1;
}
```

- [ ] **Step 4: Wire + build + run.** Add `la_add_test(test_decode)`; add `LA_TEST_SLOW=${CMAKE_SOURCE_DIR}/dumps/reference_slow.gguf` to the ENVIRONMENT string in `tests/CMakeLists.txt`'s `la_add_test` (alongside LA_TEST_GGUF/BASELINE/FIXTURES). Then:
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_decode -j 2>&1 | tail -5
cd build && ctest -R test_decode --output-on-failure; cd ..
```
Expected: `decode prefix match=1 (checked 24 tokens)`. Reprefill of 24 steps over ~300-320 tokens takes a few minutes on CPU — acceptable. DEBUG: token 0 must be 151672 (matches the M3 argmax). A mismatch at step k means the k-th greedy step diverged — since each step is the bit-exact `forward`, a divergence implies an argmax tie or an EOS-handling bug; print the top-few logits at the mismatch.

- [ ] **Step 5: Commit.**
```bash
git add src/lm.hpp src/lm.cpp tests/test_decode.cpp tests/CMakeLists.txt
git commit -m "M4: AR greedy decode (reprefill oracle) -> slow token-stream prefix parity"
```

---

## Task 3: Coordinate-token → box parser → gate boxes (full reference stream)

A pure function (no ggml): parse a token-id stream into labeled pixel boxes. Gate it on the FULL reference token ids → reference boxes (isolates the parser from the decode).

**Files:**
- Create: `src/boxes.hpp`, `src/boxes.cpp`
- Modify: `tests/test_decode.cpp` (add the box-parse gate)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write `src/boxes.hpp`.**
```cpp
#pragma once
#include <vector>
#include <string>
#include <cstdint>
namespace la {
struct Box { float x1,y1,x2,y2; std::string label; };   // pixels
// Parse the generated token-id stream into boxes. Coord order x1,y1,x2,y2.
// coord_norm = id-151677; pixel = coord_norm/1000 * (img_w for x, img_h for y).
// label = the <ref>..</ref> text immediately preceding each <box>..</box>.
// decode_label(ids) maps a run of text token-ids to a string (the C++ tokenizer
// detokenizer; for M4 pass a functor or a precomputed id->piece table).
std::vector<Box> parse_boxes(const std::vector<int32_t>& ids, int img_w, int img_h,
                             const std::vector<std::string>& id_to_piece);
}
```
NOTE on labels: full BPE detokenization of the `<ref>` text needs the byte-level decoder. For M4's gate we only need the box COORDINATES to match; the label string can be a best-effort concat of `id_to_piece[id]` (the raw vocab pieces embedded in the GGUF — already available via the tokenizer KV `locateanything.tokenizer.tokens`). If exact label strings are hard, gate ONLY the coordinates + box count and compare labels loosely (or skip label exactness, noting it). The coords are the load-bearing output.

- [ ] **Step 2: Write `src/boxes.cpp`.**
```cpp
#include "boxes.hpp"
namespace la {
static constexpr int COORD_START=151677, COORD_END=152677,
                     BOX_START=151668, BOX_END=151669, REF_START=151672, REF_END=151673;
std::vector<Box> parse_boxes(const std::vector<int32_t>& ids, int img_w, int img_h,
                             const std::vector<std::string>& id_to_piece){
    std::vector<Box> out;
    std::string cur_label;
    int i=0, n=(int)ids.size();
    while(i<n){
        int t=ids[i];
        if(t==REF_START){
            int j=i+1; cur_label.clear();
            while(j<n && ids[j]!=REF_END){
                if(ids[j]>=0 && ids[j]<(int)id_to_piece.size()) cur_label+=id_to_piece[ids[j]];
                ++j;
            }
            i=j+1; continue;
        }
        if(t==BOX_START){
            int j=i+1; std::vector<int> coords;
            while(j<n && ids[j]!=BOX_END){
                if(ids[j]>=COORD_START && ids[j]<=COORD_END) coords.push_back(ids[j]-COORD_START);
                ++j;
            }
            if(coords.size()==4){
                Box b;
                b.x1 = coords[0]/1000.f*img_w; b.y1 = coords[1]/1000.f*img_h;
                b.x2 = coords[2]/1000.f*img_w; b.y2 = coords[3]/1000.f*img_h;
                b.label = cur_label;
                out.push_back(b);
            }
            i=j+1; continue;
        }
        ++i;
    }
    return out;
}
}
```

- [ ] **Step 3: Add the box-parse gate to `tests/test_decode.cpp`.** Load the FULL reference `slow_token_ids` and the reference `slow_boxes` + img dims; parse the reference ids with `parse_boxes`; assert box count matches and each coord is within 1px of the reference (reference computed the same coord math, so this gates the PARSER MECHANICS — grammar, coord_norm, x/y→w/h — exactly). Read img_w/img_h from the slow gguf KV (`la_slow.img_w`/`img_h`).
```cpp
    // --- box parse gate (on the FULL reference stream) ---
    int img_w = (int)la_parity::load_kv_u32(slow, "la_slow.img_w");   // see note
    int img_h = (int)la_parity::load_kv_u32(slow, "la_slow.img_h");
    std::vector<std::string> pieces; la_parity::load_kv_str_array(gguf, "locateanything.tokenizer.tokens", pieces);
    std::vector<la::Box> boxes = la::parse_boxes(ref, img_w, img_h, pieces);
    std::vector<float> rboxes; std::vector<int64_t> rbs;
    la_parity::load_baseline(slow, "slow_boxes", rboxes, rbs);   // [n_box*4] flat
    int nb=(int)rboxes.size()/4;
    int bok = ((int)boxes.size()==nb);
    for(int k=0;k<nb && bok;++k){
        float* r=&rboxes[k*4];
        bok &= (std::fabs(boxes[k].x1-r[0])<1.f && std::fabs(boxes[k].y1-r[1])<1.f &&
                std::fabs(boxes[k].x2-r[2])<1.f && std::fabs(boxes[k].y2-r[3])<1.f);
        // sanity: well-formed box within image bounds
        bok &= (boxes[k].x1<=boxes[k].x2+1 && boxes[k].y1<=boxes[k].y2+1);
    }
    std::printf("box parse: got %zu ref %d match=%d\n", boxes.size(), nb, bok);
    // overall test return must reflect BOTH the decode-prefix gate AND bok
```
NOTE: `load_kv_u32` / `load_kv_str_array` may not exist in parity.hpp (it has `load_kv_str`). Add minimal readers, or read these via a tiny gguf open in the test. The tokenizer pieces array key is `locateanything.tokenizer.tokens` (from `scripts/gguf_keys.py`). If label exactness is troublesome (byte-level BPE pieces like `Ġ`), gate coords + count only and print labels for eyeballing — coords are the load-bearing check.

- [ ] **Step 4: Wire (`src/boxes.cpp` to LA_SOURCES), build, run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_decode -j 2>&1 | tail -5
cd build && ctest -R test_decode --output-on-failure; cd ..
```
Expected: `box parse: got N ref N match=1`, all boxes well-formed within bounds. If coords are off, the order (x1,y1,x2,y2) or coord_norm math is wrong. If boxes are degenerate (x1>x2), the order assumption is wrong — reconcile against the raw coord tokens.

- [ ] **Step 5: Commit.**
```bash
git add src/boxes.hpp src/boxes.cpp tests/test_decode.cpp CMakeLists.txt
git commit -m "M4: coord-token -> labeled box parser -> slow boxes parity"
```

---

## Task 4: Backend infra for a persistent KV-cache + ResidentKV struct

Add the two `la::Backend` capabilities the resident path needs and the `ResidentKV` buffer. No decode yet — just the infra + an alloc/free unit test.

**Files:**
- Modify: `src/backend.hpp`, `src/backend.cpp`
- Modify: `src/qwen2.hpp`, `src/qwen2.cpp`
- Modify: `tests/test_decode.cpp` (add a ResidentKV alloc test) or a small standalone check

- [ ] **Step 1: Expose the backend + ordered no-readback graph roots in `src/backend.hpp`.** Add to the public API:
```cpp
    // Allocate a persistent backend buffer for all tensors in `ctx` (for ResidentKV).
    ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx);
    // Register an extra graph root that compute()/forward_capture must expand (in
    // registration order, BEFORE the output) WITHOUT reading it back. For resident
    // K/V write (ggml_cpy) nodes that must land in the graph.
    void add_graph_root(ggml_tensor* t);
```

- [ ] **Step 2: Implement them in `src/backend.cpp`.**
- `allocate_ctx_tensors`: `return ggml_backend_alloc_ctx_tensors(ctx, impl_->backend);`
- `add_graph_root`: push `t` onto a new `impl_->roots` vector; clear it at the start of each compute (like pending/captures); in compute, AFTER `build` and BEFORE expanding the output, loop `for (auto* r : impl_->roots) ggml_build_forward_expand(gf, r);` (in registration order). Do NOT read roots back. Confirm against the existing compute() body — roots must be expanded before the final output expand so the K/V writes are ordered correctly (the interleaved ordering is the caller's responsibility via registration order).

- [ ] **Step 3: Add `ResidentKV` to `src/qwen2.hpp`** (ported from vibevoice qwen2.hpp:78-89).
```cpp
struct ResidentKV {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor*> k, v;   // each [head_dim, n_kv, max_seq, 1]
    int max_seq = 0, past_len = 0;
    bool init(Backend& be, int n_layers, int hd, int n_kv, int max_seq);
    void free();
    ~ResidentKV();
};
```
(forward-declare `class Backend;` in qwen2.hpp, or include backend.hpp.)

- [ ] **Step 4: Implement `ResidentKV::init/free/~` in `src/qwen2.cpp`** (port vibevoice qwen2.cpp:207-244). `init` builds a `no_alloc=true` metadata ctx, creates `k[i]`/`v[i]` as `ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, n_kv, max_seq, 1)` for each layer, then `buffer = be.allocate_ctx_tensors(ctx)`. `free` frees buffer then ctx. (Use the `Backend&` to get the right backend, per the research: the resident buffer MUST be on the same backend as the compute gallocr.)

- [ ] **Step 5: Add an alloc/free check to `tests/test_decode.cpp`** (or a quick assertion): `ResidentKV kv; assert(kv.init(be, 36, 128, 2, 600)); assert(kv.k.size()==36 && kv.k[0] && kv.k[0]->ne[2]==600); kv.free();` — confirms the buffer allocates and tensors exist. Build + run.

- [ ] **Step 6: Commit.**
```bash
git add src/backend.hpp src/backend.cpp src/qwen2.hpp src/qwen2.cpp tests/test_decode.cpp
git commit -m "M4: Backend persistent-buffer + graph-root support + ResidentKV struct"
```

---

## Task 5: Resident decode (KV-cache) → full token stream + resident==reprefill (M4 exit)

Port `qwen2_layer_forward_resident`, drive a prefill+step decode with the KV-cache, gate the FULL slow token stream, and prove the resident path is bit-identical to the reprefill oracle on a short prefix.

**Files:**
- Modify: `src/qwen2.hpp`, `src/qwen2.cpp` (`qwen2_layer_forward_resident`)
- Modify: `src/lm.hpp`, `src/lm.cpp` (`decode_greedy_resident`)
- Modify: `tests/test_decode.cpp` (full-stream + equivalence gates)

- [ ] **Step 1: Port `qwen2_layer_forward_resident` into `src/qwen2.cpp`** (from vibevoice qwen2.cpp:250-358), eager/f32 only (no flash-attn branch). Signature:
```cpp
struct Qwen2ResidentOut { ggml_tensor* y; ggml_tensor* k_write; ggml_tensor* v_write; };
Qwen2ResidentOut qwen2_layer_forward_resident(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
                                              ggml_tensor* mask, ResidentKV& kv, int layer_idx,
                                              const Qwen2LayerWeights& w, const Qwen2Hparams& hp);
```
Body (per research): rms_norm→qkv→reshape→RoPE on q,k (identical to eager); then write new k/v into `kv.k[layer_idx]`/`kv.v[layer_idx]` at offset `kv.past_len*nb[2]` via `ggml_view_4d` dest + `ggml_cpy` → `k_write`/`v_write`; read `k_full`/`v_full` as views from offset 0 with `ne[2]=kv.past_len+n_tokens`; attention exactly like eager (`ggml_mul_mat`→`mul_mat_set_prec F32`→`soft_max_ext(scores, mask, 1/sqrt(hd), 0)`→`@V`); o_proj+residual+SwiGLU. Return `{y, k_write, v_write}`. Keep mask F32.

- [ ] **Step 2: Add `decode_greedy_resident` to `src/lm.{hpp,cpp}`.** Driver (per research vibevoice_asr.cpp:519-805):
  1. `ResidentKV kv; kv.init(be_, n_layers, head_dim, n_kv_heads, prompt_len+max_new+32);`
  2. **Prefill**: embed+splice the prompt → x [H, prompt_len]; pos = [0..prompt_len-1]; mask [prompt_len, prompt_len] causal; `kv.past_len=0`; in ONE `be_.compute` lambda, loop layers via `qwen2_layer_forward_resident`, **register each `k_write`/`v_write` via `be_.add_graph_root(...)` interleaved per layer (k0,v0,k1,v1,…) BEFORE returning** the final hidden; read back the last-token hidden → set `kv.past_len = prompt_len`.
  3. **Per step**: rms_norm(output_norm) + lm.output matmul on the last hidden → argmax → append → stop on EOS; embed the new id (one row of tok_embd, host gather); run ONE token through resident layers at pos=`kv.past_len`, mask [kv.past_len+1, 1] (`j>pos? -INF:0`), registering the writes; read its hidden; `kv.past_len += 1`.
  CRITICAL (the bit-exact-vs-catastrophic gotcha): the `k_write`/`v_write` cpy nodes MUST be expanded into the graph, interleaved per layer, before the output — they alias the resident buffer but are distinct tensor objects so ggml sees no dependency edge. Use `add_graph_root` in the exact order k0,v0,k1,v1,…. Verify by the equivalence gate (Step 4).
NOTE: the per-step graph references the resident k/v tensors (pre-allocated, on the backend buffer) as leaves; the gallocr skips them (they already have a buffer) and only allocates the new-token activations. Keep everything F32.

- [ ] **Step 3: Add the full-stream gate to `tests/test_decode.cpp`.** Run `decode_greedy_resident(prompt_ids, proj, 256, got_full)`; compare the ENTIRE `got_full` against the reference `slow_token_ids` (exact id match, all tokens). This is fast (incremental), so the full ~N-token stream is checked.
```cpp
    std::vector<int32_t> got_full;
    if(!lm.decode_greedy_resident(ids, proj, 256, got_full)) return 1;
    int fok = ((int)got_full.size()==(int)ref.size());
    for(int i=0;i<(int)ref.size() && fok;++i) fok &= (got_full[i]==ref[i]);
    std::printf("resident full-stream match=%d (%zu vs %zu)\n", fok, got_full.size(), ref.size());
```

- [ ] **Step 4: Add the resident==reprefill equivalence gate.** Decode the first ~16 tokens BOTH ways and assert identical (proves the KV-cache is bit-exact vs the gated eager path — catches the interleaved-expand gotcha):
```cpp
    std::vector<int32_t> a,b2;
    lm.decode_greedy_reprefill(ids, proj, 16, a);
    lm.decode_greedy_resident(ids, proj, 16, b2);
    int eq = (a.size()>=16 && b2.size()>=16); for(int i=0;i<16 && eq;++i) eq &= (a[i]==b2[i]);
    std::printf("resident==reprefill (16 tok) = %d\n", eq);
```
Make the overall test return reflect ALL gates: decode-prefix && box-parse && full-stream && equivalence.

- [ ] **Step 5: Build + run + debug.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_decode -j 2>&1 | tail -8
cd build && ctest -R test_decode --output-on-failure; cd ..
```
Iterate to PASS. DEBUG: if `resident==reprefill` fails at token 0/1, the KV write ordering is wrong (the writes aren't expanded interleaved-before-output, so attention reads stale K/V) — fix the `add_graph_root` registration order. If full-stream diverges late but the prefix + equivalence pass, suspect the mask `full_len` bound or a position off-by-one in the step loop. Then run the FULL suite — all green.

- [ ] **Step 6: Commit (M4 complete).**
```bash
cd build && ctest --output-on-failure; cd ..
git add src/qwen2.hpp src/qwen2.cpp src/lm.hpp src/lm.cpp tests/test_decode.cpp
git commit -m "M4: resident KV-cache AR decode -> full slow stream + resident==reprefill parity (M4 complete)"
```

---

## Milestone exit criteria

**M4 done when:** `tests/test_decode.cpp` passes — AR greedy decode reproduces the slow reference token stream (prefix via reprefill, FULL via resident KV-cache), the resident path is bit-identical to the reprefill oracle, and the coord-token parser produces the reference labeled boxes (coords within 1px, well-formed within image bounds) — all green under ctest, full suite green.

## Roadmap (next plan)

- **M5 — Hybrid Parallel Box Decoding.** MTP block_size=6: `_prepare_inputs_in_mtp` (append last token + 5 `text_mask` 151676, future-slot position_ids−1), block-diffusion (bidirectional-within-block) attention mask, `decode_bbox_avg`/`is_valid_box_frame` (top-k=5 coord selection, abnormal-position zeroing), `handle_pattern` block classifier, hybrid control flow (start MTP, error_box→AR fallback, box_end→back to MTP). Gate vs the hybrid M0 token_stream (the existing dump). Then M6 (quantize f16/q8/q4 + CLI `detect --annotated` + flat C-API).
