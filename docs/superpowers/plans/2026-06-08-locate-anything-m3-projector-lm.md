# LocateAnything-3B ggml Port — M3 (Projector + Qwen2 LM forward) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project `merged_tokens [256,4608]` through the `mlp1` projector to `projected_tokens [256,2048]`, splice them into the `<IMG_CONTEXT>` (151665) slots of the embedded 297-token prompt, run a standard causal Qwen2 prefill, and reproduce `logits_step0 [152681]` (argmax 151672) — all numerically equal to the M0 reference, component by component.

**Architecture:** Reuse vibevoice.cpp's `qwen2.cpp` per-layer math (eager, no KV-cache: a single full-sequence causal prefill over 297 tokens) — its production tensor naming already matches our GGUF. The projector and the embed-and-splice are small new pieces. Gate each stage against the M0 dumps: projected_tokens, embeds_after_splice, lm_layer_00, lm_layer_35, logits_step0.

**Tech Stack:** C++17, ggml (embedded), CMake/CTest, Python (gguf/numpy/torch) for one reference-dump extension.

**Reference material (read before starting):**
- Exact M3 math: `models/LocateAnything-3B/modeling_locateanything.py` (mlp1 at ~L136-141, splice at modeling_qwen2.py ~L1179-1192) and `modeling_qwen2.py` (decoder layer). Restated inline per task; verify against these.
- Qwen2 ggml port source: `/home/mudler/_git/vibevoice.cpp/src/qwen2.cpp` (`qwen2_layer_forward`, eager path L77-203), `src/qwen2.hpp` (Qwen2Hparams/Qwen2LayerWeights), `src/rms_norm.hpp`, `src/rope.hpp`, and the production loader `load_qwen2_layer` in `src/vibevoice_tts.cpp` L27-52. Cross-check: `/home/mudler/_git/llama.cpp/src/models/qwen2.cpp` (numerically identical).
- Existing repo infra (M2): `la::Backend` (`compute`/`forward_capture`/`add_graph_input(_nd)`/`capture`/`GraphInputPool`), `la::ModelLoader` (`ml.tensor(name)`, `ml.config()`), `la::linear/layernorm/gelu_tanh` (`src/ggml_extend.hpp`), `la_parity::load_baseline/compare`, `tests/CMakeLists.txt`'s `la_add_test` (sets LA_TEST_GGUF/LA_TEST_BASELINE/LA_TEST_FIXTURES).
- GGUF names (`scripts/gguf_keys.py`): LM = `lm.tok_embd.weight`, `lm.blk.{N}.{attn_q,attn_k,attn_v}.{weight,bias}`, `lm.blk.{N}.attn_o.weight`, `lm.blk.{N}.attn_norm.weight`, `lm.blk.{N}.ffn_norm.weight`, `lm.blk.{N}.ffn_{gate,up,down}.weight`, `lm.output_norm.weight`, `lm.output.weight`. Projector = `proj.0.{weight,bias}` (LayerNorm 4608), `proj.1.{weight,bias}` (Linear 4608→2048), `proj.3.{weight,bias}` (Linear 2048→2048). KV macros (`include/la_gguf_keys.h`): `LA_KV_LM_HIDDEN`, `LA_KV_LM_N_LAYERS`, `LA_KV_LM_N_HEADS`, `LA_KV_LM_N_KV_HEADS`, `LA_KV_LM_HEAD_DIM`, `LA_KV_LM_INTERMEDIATE`, `LA_KV_LM_VOCAB`, `LA_KV_LM_ROPE_THETA`, `LA_KV_LM_RMS_EPS`, `LA_KV_TOK_IMAGE`.
- `models/locate-anything-f32.gguf` (f32 weights) + `dumps/reference.npz`/`manifest.json` exist. The reference LM ran in f32, weights are f32 → no quantization error in M3.

**Dims:** LM hidden 2048, 36 layers, 16 q heads, 2 kv heads (GQA group 8), head_dim 128, intermediate 11008, vocab 152681, rope_theta 1e6 (NEOX), rms_eps 1e-6, attention scale 1/√128, tied embeddings. Prompt = 297 tokens (256 vision + 41 text), positions 0..296, plain causal mask. Projector eps 1e-5, **erf** GELU.

**Scope:** M3 ends at `logits_step0` parity (the single causal prefill). AR decode is M4, hybrid Parallel-Box-Decoding is M5. No KV-cache here (full-sequence prefill only). M3 isolates from M2 by feeding the **reference** `merged_tokens` into the projector (so M2's 2.65e-3 residual doesn't compound); an end-to-end pixel→logits chain via the real VitEncoder is an optional smoke at the end, not the gate.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/dump_reference.py` (modify) | Also dump `input_ids` (the 297 prompt token ids, int32) |
| `scripts/reference_npz_to_gguf.py` (modify) | Carry int32 `input_ids` into `dumps/reference.gguf` |
| `src/model_loader.{hpp,cpp}` (modify) | Add LM hparams + image-token id to the config (tensor map already loads lm.*/proj.*) |
| `src/ggml_extend.hpp` (modify) | Add `rms_norm` + `gelu_erf` helpers |
| `src/projector.{hpp,cpp}` | `mlp1` projector graph (LayerNorm→Linear→gelu_erf→Linear) + loader |
| `src/qwen2.{hpp,cpp}` | Qwen2 hparams/layer-weights, `load_qwen2_layer`, `qwen2_layer_forward` (eager, no cache) — ported from vibevoice |
| `src/lm.{hpp,cpp}` | `LMForward`: embed (`ggml_get_rows`) + vision splice + 36-layer causal prefill → logits; capture support |
| `tests/test_model_loader.cpp` (modify) | Add LM hparam + tensor-shape assertions |
| `tests/test_projector.cpp` | Gate projected_tokens |
| `tests/test_lm.cpp` | Gate embeds_after_splice, lm_layer_00, lm_layer_35, logits_step0 (+ argmax 151672) |
| `tests/CMakeLists.txt` (modify) | Register the two new tests |
| `CMakeLists.txt` (modify) | Add `src/projector.cpp`, `src/qwen2.cpp`, `src/lm.cpp` to `LA_SOURCES` |

**Decomposition rationale:** `qwen2.{hpp,cpp}` is the ported decoder layer (the bulk, but a near-verbatim clone). `projector` and `lm` are small and each own one responsibility (project; embed+splice+drive). Helpers (`rms_norm`, `gelu_erf`) go in the existing header-only `ggml_extend.hpp`. Each task gates a dumped tensor before the next.

---

## Task 1: Dump the prompt `input_ids` (reference extension)

M0 dumped `embeds_after_splice` and `logits_step0` but not the 297 prompt `input_ids` themselves — M3 needs them to embed + splice + locate the 151665 positions.

**Files:**
- Modify: `scripts/dump_reference.py`
- Modify: `scripts/reference_npz_to_gguf.py`

- [ ] **Step 1: Add `input_ids` to the dump.** In `scripts/dump_reference.py`, where the other store entries are populated (the function already has `inputs` from `R.build_inputs()`), add the prompt ids as int32:
```python
    store["input_ids"] = inputs["input_ids"][0].cpu().numpy().astype(np.int64)
```
(Place it alongside the existing `store[...] = ...` assignments, before `np.savez`. int64 to match the existing token_stream convention; reference_npz_to_gguf will downcast to int32 for the GGUF.)

- [ ] **Step 2: Regenerate the dump and confirm.**
```bash
. .venv/bin/activate
python scripts/dump_reference.py
python - <<'PY'
import numpy as np
z = np.load("dumps/reference.npz")
ids = z["input_ids"]
print("input_ids shape:", ids.shape, "n_image_tokens:", int((ids==151665).sum()))
print("first 8:", ids[:8].tolist(), "last 4:", ids[-4:].tolist())
PY
```
Expected: `input_ids shape: (297,)`, `n_image_tokens: 256`. (The dump is gitignored; only the script change is committed.)

- [ ] **Step 3: Carry int32 input_ids into reference.gguf.** In `scripts/reference_npz_to_gguf.py`, the current loop skips int64 tensors. Replace the skip with explicit handling so `input_ids` is written as int32 while other int64 (token_stream) stays skipped:
```python
    for name in z.files:
        a = z[name]
        if name == "input_ids":
            w.add_tensor(name, np.ascontiguousarray(a, dtype=np.int32))
            n += 1
            continue
        if a.dtype == np.int64:        # token_stream — skip (not needed in C++)
            continue
        w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))
        n += 1
```

- [ ] **Step 4: Regenerate reference.gguf and confirm input_ids present as int32.**
```bash
. .venv/bin/activate
python scripts/reference_npz_to_gguf.py
python - <<'PY'
import gguf, numpy as np
r = gguf.GGUFReader("dumps/reference.gguf")
t = [t for t in r.tensors if t.name=="input_ids"][0]
print("input_ids dtype:", t.tensor_type, "n:", t.data.size, "n_img:", int((np.array(t.data)==151665).sum()))
PY
```
Expected: an INT32 tensor of size 297 with 256 image tokens. Also confirm `projected_tokens`, `embeds_after_splice`, `lm_layer_00`, `lm_layer_35`, `logits_step0` are still present (they were always in the npz; M2 only used the vit_* ones).

- [ ] **Step 5: Commit.**
```bash
git add scripts/dump_reference.py scripts/reference_npz_to_gguf.py
git commit -m "M3: dump prompt input_ids + carry as int32 in reference.gguf"
```
Author `mudler <mudler@localai.io>`, end every M3 commit message with a trailing:
```

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: Extend ModelLoader with LM hparams + image-token id

The loader already builds a name→tensor map over ALL gguf tensors (so `lm.*`/`proj.*` tensors already resolve via `ml.tensor(...)`). This task adds the LM scalar hparams + the image-token id to the config struct.

**Files:**
- Modify: `src/model_loader.hpp` (extend the config struct)
- Modify: `src/model_loader.cpp` (read the LM KV)
- Modify: `tests/test_model_loader.cpp` (assert LM hparams + tensor shapes)

- [ ] **Step 1: Add LM fields to the config struct in `src/model_loader.hpp`.** Add to `struct VitConfig` (it's the model config; keep the name or rename — keep `VitConfig` to avoid churn):
```cpp
    // --- language model (Qwen2) ---
    uint32_t lm_hidden = 0, lm_n_layers = 0, lm_n_heads = 0, lm_n_kv_heads = 0,
             lm_head_dim = 0, lm_intermediate = 0, lm_vocab = 0;
    float    lm_rope_theta = 1.0e6f, lm_rms_eps = 1.0e-6f;
    uint32_t image_token_id = 151665;
```

- [ ] **Step 2: Read them in `ModelLoader::load` (src/model_loader.cpp), after the vit reads:**
```cpp
    cfg_.lm_hidden       = kv_u32(gguf_, LA_KV_LM_HIDDEN);
    cfg_.lm_n_layers     = kv_u32(gguf_, LA_KV_LM_N_LAYERS);
    cfg_.lm_n_heads      = kv_u32(gguf_, LA_KV_LM_N_HEADS);
    cfg_.lm_n_kv_heads   = kv_u32(gguf_, LA_KV_LM_N_KV_HEADS);
    cfg_.lm_head_dim     = kv_u32(gguf_, LA_KV_LM_HEAD_DIM);
    cfg_.lm_intermediate = kv_u32(gguf_, LA_KV_LM_INTERMEDIATE);
    cfg_.lm_vocab        = kv_u32(gguf_, LA_KV_LM_VOCAB);
    cfg_.lm_rope_theta   = kv_f32(gguf_, LA_KV_LM_ROPE_THETA, 1.0e6f);
    cfg_.lm_rms_eps      = kv_f32(gguf_, LA_KV_LM_RMS_EPS, 1.0e-6f);
    cfg_.image_token_id  = kv_u32(gguf_, LA_KV_TOK_IMAGE, 151665);
```
(`kv_u32`/`kv_f32` already exist in this file.)

- [ ] **Step 3: Add assertions to `tests/test_model_loader.cpp`** (extend the existing `ok &=` chain, before the print):
```cpp
    ok &= (c.lm_hidden == 2048);
    ok &= (c.lm_n_layers == 36);
    ok &= (c.lm_n_heads == 16);
    ok &= (c.lm_n_kv_heads == 2);
    ok &= (c.lm_head_dim == 128);
    ok &= (c.lm_intermediate == 11008);
    ok &= (c.lm_vocab == 152681);
    ok &= (c.image_token_id == 151665);
    // LM tensor shapes (ggml reversed-ne): q_proj torch [2048,2048], k/v [256,2048]
    ggml_tensor* q = ml.tensor("lm.blk.0.attn_q.weight");
    ok &= (q && q->ne[0]==2048 && q->ne[1]==2048);
    ggml_tensor* kb = ml.tensor("lm.blk.0.attn_k.bias");
    ok &= (kb && kb->ne[0]==256);
    ggml_tensor* emb = ml.tensor("lm.tok_embd.weight");
    ok &= (emb && emb->ne[0]==2048 && emb->ne[1]==152681);
    ggml_tensor* outw = ml.tensor("lm.output.weight");
    ok &= (outw && outw->ne[0]==2048 && outw->ne[1]==152681);
    ggml_tensor* p0 = ml.tensor("proj.0.weight");
    ok &= (p0 && p0->ne[0]==4608);
    ggml_tensor* p1 = ml.tensor("proj.1.weight");
    ok &= (p1 && p1->ne[0]==4608 && p1->ne[1]==2048);
```

- [ ] **Step 4: Build + run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_model_loader -j 2>&1 | tail -3
cd build && ctest -R test_model_loader --output-on-failure; cd ..
```
Expected: PASS (`loader ok=1`). If `lm.output.weight` is absent (tied embeddings sometimes omit it), check the GGUF (`gguf` reader) — M1 mapped `language_model.lm_head.weight`→`lm.output.weight` and it exists; if it truly doesn't, change the test/loader to fall back to `lm.tok_embd.weight` and note it for Task 6.

- [ ] **Step 5: Commit.**
```bash
git add src/model_loader.hpp src/model_loader.cpp tests/test_model_loader.cpp
git commit -m "M3: load LM (Qwen2) hparams + image-token id into model config"
```

---

## Task 3: ggml_extend helpers (rms_norm, gelu_erf) + MLP projector → `projected_tokens`

**Files:**
- Modify: `src/ggml_extend.hpp` (add `rms_norm`, `gelu_erf`)
- Create: `src/projector.hpp`, `src/projector.cpp`
- Create: `tests/test_projector.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: Add helpers to `src/ggml_extend.hpp`** (after the existing `gelu_tanh`):
```cpp
// RMSNorm over ne[0] with affine weight (Qwen2; eps in fp32). No bias, no mean-subtract.
inline ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Exact erf GELU (matches torch nn.GELU() default approximate='none'): 0.5*x*(1+erf(x/sqrt2)).
inline ggml_tensor* gelu_erf(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu_erf(ctx, x);
}
```
NOTE: verify `ggml_gelu_erf` exists in `third_party/ggml/include/ggml.h` (it does in current ggml). Check whether this build routes erf-gelu through an fp16 table (grep `GGML_GELU_ERF_FP16` / the erf path in `third_party/ggml/src/ggml-cpu/`). M2 found `ggml_gelu` (tanh) is fp16-tabled; if `ggml_gelu_erf` is ALSO tabled and the projected_tokens gate (Step 5) fails by ~5e-4, that's the cause — note it and loosen the projected_tokens gate to 1e-3 (still tight enough to catch a wrong variant), OR if a precise f32 erf path exists, use it. The erf-vs-tanh distinction is the load-bearing correctness point; the table precision is secondary.

- [ ] **Step 2: Write the failing test `tests/test_projector.cpp`.**
```cpp
#include "projector.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* base=std::getenv("LA_TEST_BASELINE");
    if(!gguf||!base){std::fprintf(stderr,"skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Backend be; la::Projector proj(ml, be);
    std::vector<float> merged; std::vector<int64_t> ms;
    if(!la_parity::load_baseline(base,"merged_tokens",merged,ms)) return 1;   // [256,4608]
    std::vector<float> got;                                                    // [2048,256]
    if(!proj.project(merged, got)) return 1;
    std::vector<float> ref; std::vector<int64_t> rs;
    if(!la_parity::load_baseline(base,"projected_tokens",ref,rs)) return 1;     // [256,2048]
    // got is ggml [2048,256] raw flat = [token,hidden]; ref is [256,2048] token-major -> same flat order
    bool ok = la_parity::compare(got, ref, "projected_tokens", 1e-3f, 1e-3f);
    return ok?0:1;
}
```

- [ ] **Step 3: Write `src/projector.hpp`.**
```cpp
#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <vector>
namespace la {
class Projector {
public:
    Projector(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}
    // merged_host: [256*4608] flat token-major (token outer, channel inner).
    // out: [2048*256] flat = ggml [2048,256] (hidden fastest, token next).
    bool project(const std::vector<float>& merged_host, std::vector<float>& out);
    // Graph-builder variant: project an in-graph [4608, n] tensor -> [2048, n].
    ggml_tensor* build(ggml_context* ctx, ggml_tensor* merged) const;
private:
    ModelLoader& ml_; Backend& be_;
};
}
```

- [ ] **Step 4: Write `src/projector.cpp`.**
```cpp
#include "projector.hpp"
#include "ggml_extend.hpp"
namespace la {
ggml_tensor* Projector::build(ggml_context* ctx, ggml_tensor* merged) const {
    // merged: [4608, n_tokens]. mlp1: LayerNorm(4608,eps1e-5) -> Linear -> gelu_erf -> Linear.
    ggml_tensor* x = la::layernorm(ctx, merged, ml_.tensor("proj.0.weight"),
                                   ml_.tensor("proj.0.bias"), 1e-5f);
    x = la::linear(ctx, ml_.tensor("proj.1.weight"), x, ml_.tensor("proj.1.bias")); // [2048,n]
    x = la::gelu_erf(ctx, x);
    x = la::linear(ctx, ml_.tensor("proj.3.weight"), x, ml_.tensor("proj.3.bias")); // [2048,n]
    return x;
}
bool Projector::project(const std::vector<float>& merged_host, std::vector<float>& out){
    const int n = 256, c = 4608;
    GraphInputPool pool;
    return be_.compute([&](ggml_context* ctx)->ggml_tensor*{
        const int64_t ne[2] = {c, n};                 // [4608, 256]
        ggml_tensor* m = be_.add_graph_input_nd(ctx, pool, merged_host.data(), ne, 2);
        return build(ctx, m);
    }, out);
}
}
```
NOTE on layout: `merged_host` is the reference `merged_tokens` flat [256,4608] = token-major, channel-inner = exactly ggml ne={4608,256} (channel fastest). Feed directly. Output ggml ne=[2048,256] raw flat = token-outer/hidden-inner = matches `projected_tokens` [256,2048] reference flat. No transpose (same convention proven in M2).

- [ ] **Step 5: Wire (`la_add_test(test_projector)`; add `src/projector.cpp` to `LA_SOURCES`), build, run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_projector -j 2>&1 | tail -5
cd build && ctest -R test_projector --output-on-failure; cd ..
```
Expected: PASS. projected_tokens max-abs-diff should be small (≤1e-3). If it fails by ~5e-4 and you confirmed gelu_erf is fp16-tabled, loosen to 1e-3 and note it; if it fails by a lot, the LayerNorm (must subtract mean — `la::layernorm` does, it's `ggml_norm` not `ggml_rms_norm`) or the erf-vs-tanh choice is wrong.

- [ ] **Step 6: Commit.**
```bash
git add src/ggml_extend.hpp src/projector.hpp src/projector.cpp tests/test_projector.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M3: mlp1 projector (LayerNorm+erf-gelu) -> projected_tokens parity"
```

---

## Task 4: Embed + vision-token splice → `embeds_after_splice`

Embed the 297 input_ids via `lm.tok_embd` and overwrite the 256 positions where `id==151665` with `projected_tokens` (in increasing index order, no scaling).

**Files:**
- Create: `src/lm.hpp`, `src/lm.cpp`
- Create: `tests/test_lm.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: Write `src/lm.hpp` (the embed+splice method first; prefill added in Task 6).**
```cpp
#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "projector.hpp"
#include <vector>
namespace la {
class LMForward {
public:
    LMForward(ModelLoader& ml, Backend& be) : ml_(ml), be_(be), proj_(ml, be) {}
    // ids: 297 token ids. projected_host: [2048*256] (already-projected vision tokens,
    // ggml [2048,256] flat). Returns embeds_after_splice [2048*297] = ggml [2048,297].
    bool embed_and_splice(const std::vector<int32_t>& ids,
                          const std::vector<float>& projected_host,
                          std::vector<float>& out);
private:
    ModelLoader& ml_; Backend& be_; Projector proj_;
};
}
```

- [ ] **Step 2: Write `embed_and_splice` in `src/lm.cpp`.**
The embed is a pure row lookup and the splice is pure data movement (no math), so do both host-side — `lm.tok_embd.weight` lives on the CPU buffer (`->data` accessible), so gather rows directly; no ggml graph or int32 input plumbing needed. (Same host-reindex philosophy as M2's merger.)
```cpp
#include "lm.hpp"
#include <cstring>
namespace la {
bool LMForward::embed_and_splice(const std::vector<int32_t>& ids,
                                 const std::vector<float>& projected_host,
                                 std::vector<float>& out){
    const int seq = (int)ids.size();                   // 297
    const int H   = (int)ml_.config().lm_hidden;       // 2048
    const int img = (int)ml_.config().image_token_id;  // 151665
    // 1) embed via host row-gather from tok_embd (ne=[2048,152681], CPU buffer)
    ggml_tensor* te = ml_.tensor("lm.tok_embd.weight");
    if (!te) return false;
    const float* tew = (const float*)te->data;
    out.assign((size_t)seq*H, 0.f);                    // [token, hidden] flat
    for (int t=0;t<seq;++t)
        std::memcpy(&out[(size_t)t*H], &tew[(size_t)ids[t]*H], (size_t)H*sizeof(float));
    // 2) overwrite the 256 image positions (id==151665) in order with projected vision rows
    int vi = 0;
    for (int t=0;t<seq;++t)
        if (ids[t]==img)
            std::memcpy(&out[(size_t)t*H], &projected_host[(size_t)(vi++)*H], (size_t)H*sizeof(float));
    return vi == 256;
}
}
```
NOTE: this assumes the CPU backend (M3 is CPU-only, as M2 was) so `te->data` is host-accessible — consistent with the M2 model_loader. If a GPU backend is ever introduced, this gather (and M2's `pe->data` access) would need a device read; not a concern for M3 parity.

- [ ] **Step 3: Write `tests/test_lm.cpp` (embeds_after_splice gate first).**
```cpp
#include "lm.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
static std::vector<int32_t> load_ids(const char* base){
    std::vector<int32_t> ids; std::vector<int64_t> sh;
    la_parity::load_baseline_i32(base, "input_ids", ids, sh);   // int32 reader in parity.hpp
    return ids;
}
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* base=std::getenv("LA_TEST_BASELINE");
    if(!gguf||!base){std::fprintf(stderr,"skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Backend be; la::LMForward lm(ml, be);
    std::vector<int32_t> ids = load_ids(base);
    std::vector<float> proj; std::vector<int64_t> ps;
    if(!la_parity::load_baseline(base,"projected_tokens",proj,ps)) return 1;  // [256,2048]
    std::vector<float> spliced;
    if(!lm.embed_and_splice(ids, proj, spliced)) return 1;                    // [2048,297]
    std::vector<float> ref; std::vector<int64_t> rs;
    if(!la_parity::load_baseline(base,"embeds_after_splice",ref,rs)) return 1;// [1,297,2048]
    bool ok = la_parity::compare(spliced, ref, "embeds_after_splice", 1e-4f, 1e-4f);
    return ok?0:1;
}
```
NOTE: `la_parity::load_baseline_i32` already exists in the copied parity.hpp (the report on Task 1 of M2 listed it among the preserved helpers). If its signature differs, adapt this call. `ref` embeds_after_splice is [1,297,2048] flat = [token,hidden]; `spliced` is [2048,297] raw flat = [token,hidden] — same order, direct compare.

- [ ] **Step 4: Wire (`la_add_test(test_lm)`; add `src/lm.cpp` to `LA_SOURCES`), build, run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_lm -j 2>&1 | tail -5
cd build && ctest -R test_lm --output-on-failure; cd ..
```
Expected: PASS at 1e-4 (embed is a lookup + copy, should be ~exact; any diff is just f32 storage). If it fails, check: (a) using `projected_tokens` reference (not C++-computed) so this isolates the splice; (b) the 151665 positions and order; (c) row layout `ids[t]*H`.

- [ ] **Step 5: Commit.**
```bash
git add src/lm.hpp src/lm.cpp tests/test_lm.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M3: embed + vision-token splice -> embeds_after_splice parity"
```

---

## Task 5: Port the Qwen2 decoder layer → gate `lm_layer_00`

Port vibevoice's eager `qwen2_layer_forward` and gate ONE layer: feed the reference `embeds_after_splice` through layer 0 (with positions 0..296 + causal mask) and compare against `lm_layer_00`.

**Files:**
- Create: `src/qwen2.hpp`, `src/qwen2.cpp`
- Modify: `src/lm.{hpp,cpp}` (add a `block0` test entry) OR put a layer-0 driver in the test
- Modify: `tests/test_lm.cpp` (add the lm_layer_00 gate)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write `src/qwen2.hpp`.**
```cpp
#pragma once
#include "ggml.h"
#include "model_loader.hpp"
#include <vector>
namespace la {
struct Qwen2Hparams {
    int hidden=2048, n_heads=16, n_kv_heads=2, head_dim=128, intermediate=11008;
    float rope_theta=1.0e6f, rms_eps=1.0e-6f;
};
struct Qwen2LayerWeights {
    ggml_tensor *attn_norm, *attn_q,*attn_q_b, *attn_k,*attn_k_b, *attn_v,*attn_v_b,
                *attn_o, *ffn_norm, *ffn_gate, *ffn_up, *ffn_down;
};
Qwen2LayerWeights load_qwen2_layer(const ModelLoader& ml, int i);
// Eager, no-cache single-layer forward. x:[hidden,seq]; pos: int32 [seq]; mask: f32 [seq,seq].
ggml_tensor* qwen2_layer_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
                                 ggml_tensor* mask, const Qwen2LayerWeights& w,
                                 const Qwen2Hparams& hp);
}
```

- [ ] **Step 2: Write `src/qwen2.cpp`** (ported from `vibevoice.cpp/src/qwen2.cpp` eager path; uses `la::linear`, `la::rms_norm` from ggml_extend).
```cpp
#include "qwen2.hpp"
#include "ggml_extend.hpp"
#include <cmath>
namespace la {
Qwen2LayerWeights load_qwen2_layer(const ModelLoader& ml, int i){
    auto t=[&](const std::string& s){ return ml.tensor("lm.blk."+std::to_string(i)+"."+s); };
    return Qwen2LayerWeights{
        t("attn_norm.weight"),
        t("attn_q.weight"), t("attn_q.bias"),
        t("attn_k.weight"), t("attn_k.bias"),
        t("attn_v.weight"), t("attn_v.bias"),
        t("attn_o.weight"),
        t("ffn_norm.weight"),
        t("ffn_gate.weight"), t("ffn_up.weight"), t("ffn_down.weight")};
}
ggml_tensor* qwen2_layer_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
                                 ggml_tensor* mask, const Qwen2LayerWeights& w,
                                 const Qwen2Hparams& hp){
    const int hd=hp.head_dim, n_h=hp.n_heads, n_kv=hp.n_kv_heads;
    const int seq = (int)x->ne[1];
    ggml_tensor* xn = la::rms_norm(ctx, x, w.attn_norm, hp.rms_eps);
    ggml_tensor* q = la::linear(ctx, w.attn_q, xn, w.attn_q_b);   // [2048,seq]
    ggml_tensor* k = la::linear(ctx, w.attn_k, xn, w.attn_k_b);   // [256,seq]
    ggml_tensor* v = la::linear(ctx, w.attn_v, xn, w.attn_v_b);   // [256,seq]
    q = ggml_reshape_3d(ctx, q, hd, n_h,  seq);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, seq);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, seq);
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor* qp = ggml_permute(ctx, q, 0,2,1,3);             // [hd,seq,n_h]
    ggml_tensor* kp = ggml_permute(ctx, k, 0,2,1,3);             // [hd,seq,n_kv]
    ggml_tensor* vp = ggml_permute(ctx, v, 0,2,1,3);
    ggml_tensor* scores = ggml_mul_mat(ctx, kp, qp);            // [seq_kv,seq_q,n_h] (GQA broadcast)
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f/std::sqrt((float)hd), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vp));  // [seq_kv,hd,n_kv]
    ggml_tensor* o  = ggml_mul_mat(ctx, vt, scores);           // [hd,seq_q,n_h]
    o = ggml_permute(ctx, o, 0,2,1,3);                          // [hd,n_h,seq]
    o = ggml_cont_2d(ctx, o, hd*n_h, seq);                     // [2048,seq]
    o = ggml_mul_mat(ctx, w.attn_o, o);                        // no bias
    ggml_tensor* h = ggml_add(ctx, x, o);
    ggml_tensor* hn = la::rms_norm(ctx, h, w.ffn_norm, hp.rms_eps);
    ggml_tensor* g = ggml_mul_mat(ctx, w.ffn_gate, hn);
    ggml_tensor* u = ggml_mul_mat(ctx, w.ffn_up,   hn);
    ggml_tensor* gu = ggml_silu(ctx, g);
    gu = ggml_mul(ctx, gu, u);                                  // silu(gate)*up
    ggml_tensor* f = ggml_mul_mat(ctx, w.ffn_down, gu);
    return ggml_add(ctx, h, f);
}
}
```
NOTE: this mirrors vibevoice qwen2.cpp L77-203 (eager). Verify against it: GQA is implicit via `ggml_mul_mat` broadcast (16%2==0, no explicit repeat_kv); RoPE is `GGML_ROPE_TYPE_NEOX` (half-split) base 1e6, n_dims=head_dim=128; scale 1/sqrt(128) inside soft_max_ext with the additive causal mask; scores precision F32; SwiGLU = silu(gate)*up (if `ggml_swiglu_split` exists in this ggml, `ggml_swiglu_split(ctx,g,u)` is equivalent — the manual `ggml_silu`+`ggml_mul` is bit-identical in f32). Confirm `ggml_rope_ext`'s exact arg list against `third_party/ggml/include/ggml.h` and vibevoice's call (the (pos, freq_factors, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) order).

- [ ] **Step 3: Add a layer-0 driver + gate to `tests/test_lm.cpp`.** Add a method `LMForward::run_layer0(const std::vector<float>& embeds_host, std::vector<float>& out)` (in lm.cpp) that feeds a [2048,297] host tensor through one `qwen2_layer_forward` with pos=0..296 and a causal mask, OR build it inline in the test. The causal mask + pos are graph inputs:
```cpp
// in the test, after the embeds_after_splice gate:
std::vector<float> emb_ref; std::vector<int64_t> es;
la_parity::load_baseline(base,"embeds_after_splice",emb_ref,es);     // [1,297,2048]=[2048,297] flat
std::vector<float> l0;
if(!lm.run_layer0(emb_ref, l0)) return 1;                            // [2048,297]
std::vector<float> r0; std::vector<int64_t> r0s;
la_parity::load_baseline(base,"lm_layer_00",r0,r0s);                 // [1,297,2048]
bool ok0 = la_parity::compare(l0, r0, "lm_layer_00", 2e-2f, 2e-2f);
```
`run_layer0` builds: feed embeds as input [2048,297]; build int32 pos [0..296] and f32 causal mask [297,297] (`m[i*seq+j] = j>i ? -INF : 0`) as graph inputs (use the int32/f32 input plumbing — for pos you need an int32 input; add `Backend::add_int32_input` now, or build pos/mask as ggml tensors filled via a small host-upload helper). Run `qwen2_layer_forward(ctx, x, pos, mask, load_qwen2_layer(ml,0), hp)`; capture output. Gate 2e-2.

NOTE — int32 pos input + f32 mask input: `ggml_rope_ext` needs `pos` as an I32 graph leaf and `soft_max_ext` needs the mask as an F32 leaf. Add a minimal `Backend::add_int32_input_nd` (mirror `add_graph_input_nd` but GGML_TYPE_I32 and an int32 pool buffer). The mask uses the existing F32 `add_graph_input_nd` with ne={seq,seq}. Build both before the layer call.

- [ ] **Step 4: Build + run + debug.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_lm -j 2>&1 | tail -5
cd build && ctest -R test_lm --output-on-failure; cd ..
```
Iterate to PASS. DEBUG if lm_layer_00 fails (embeds_after_splice already passes, so input is right): diff ~O(1) → RoPE type (must be NEOX half-split, not interleaved) or GQA broadcast or q/k/v bias missing; diff ~O(0.1) diffuse → scale (1/sqrt(128)), rms eps (1e-6), or causal mask orientation (j>i masked). Print first values vs ref. Cross-check the attention idiom against vibevoice qwen2.cpp.

- [ ] **Step 5: Commit.**
```bash
git add src/qwen2.hpp src/qwen2.cpp src/lm.hpp src/lm.cpp tests/test_lm.cpp src/backend.hpp src/backend.cpp CMakeLists.txt
git commit -m "M3: port Qwen2 decoder layer (eager) -> lm_layer_00 parity"
```
(Include backend.* if you added the int32 input helper.)

---

## Task 6: Full 36-layer causal prefill + logits → `lm_layer_35` + `logits_step0` (M3 exit)

**Files:**
- Modify: `src/lm.{hpp,cpp}` (add `forward` — embed+splice+36 layers+final norm+logits, with capture)
- Modify: `tests/test_lm.cpp` (add lm_layer_35 + logits_step0 gates incl. argmax)

- [ ] **Step 1: Add `LMForward::forward` to `src/lm.cpp`.**
```cpp
// ids:297, projected_host:[2048*256]. Returns logits at the LAST position [vocab].
// capture_layers: block indices whose output to also read back.
bool LMForward::forward(const std::vector<int32_t>& ids,
                        const std::vector<float>& projected_host,
                        std::vector<float>& logits_last,
                        const std::vector<int>& capture_layers,
                        std::vector<std::vector<float>>& captured){
    const int seq=(int)ids.size(), H=(int)ml_.config().lm_hidden;
    // host embed+splice -> spliced [H*seq]
    std::vector<float> spliced;
    if(!embed_and_splice(ids, projected_host, spliced)) return false;
    Qwen2Hparams hp; // dims from config:
    const auto& c = ml_.config();
    hp.hidden=c.lm_hidden; hp.n_heads=c.lm_n_heads; hp.n_kv_heads=c.lm_n_kv_heads;
    hp.head_dim=c.lm_head_dim; hp.intermediate=c.lm_intermediate;
    hp.rope_theta=c.lm_rope_theta; hp.rms_eps=c.lm_rms_eps;
    std::vector<int32_t> pos(seq); for(int i=0;i<seq;++i) pos[i]=i;
    std::vector<float> mask((size_t)seq*seq);
    for(int i=0;i<seq;++i) for(int j=0;j<seq;++j) mask[(size_t)i*seq+j] = (j>i)? -INFINITY : 0.0f;
    GraphInputPool pool;
    captured.assign(capture_layers.size(), {});
    return be_.forward_capture([&](ggml_context* ctx)->ggml_tensor*{
        const int64_t ene[2]={H,seq};
        ggml_tensor* x = be_.add_graph_input_nd(ctx, pool, spliced.data(), ene, 2); // [H,seq]
        ggml_tensor* post = be_.add_int32_input_nd(ctx, pool, pos.data(), (int64_t[]){seq},1);
        const int64_t mne[2]={seq,seq};
        ggml_tensor* maskt = be_.add_graph_input_nd(ctx, pool, mask.data(), mne, 2);
        for(int il=0; il<(int)c.lm_n_layers; ++il){
            x = la::qwen2_layer_forward(ctx, x, post, maskt, la::load_qwen2_layer(ml_,il), hp);
            for(size_t cc=0;cc<capture_layers.size();++cc)
                if(capture_layers[cc]==il) be_.capture(x,&captured[cc]);
        }
        x = la::rms_norm(ctx, x, ml_.tensor("lm.output_norm.weight"), hp.rms_eps);
        // logits: slice last token then project, to avoid a [vocab,seq] blowup
        ggml_tensor* last = ggml_view_2d(ctx, x, H, 1, x->nb[1], (size_t)(seq-1)*x->nb[1]);
        last = ggml_cont(ctx, last);                              // [H,1]
        return ggml_mul_mat(ctx, ml_.tensor("lm.output.weight"), last); // [vocab,1]
    }, logits_last);
}
```
NOTE: declare `forward` in lm.hpp. `add_int32_input_nd` is the int32 helper from Task 5. Slicing the last token before the `lm.output` matmul keeps the output `[vocab,1]` instead of `[vocab,297]` (cheaper, and we only gate the last position). If `lm.output.weight` is absent (tied), use `lm.tok_embd.weight` (same values). Bump `kGraphSize` in backend.cpp if 36 layers overflow (~36×~30 nodes ≈ 1100 — fits 16384).

- [ ] **Step 2: Add the lm_layer_35 + logits_step0 gates to `tests/test_lm.cpp`.**
```cpp
std::vector<float> logits; std::vector<std::vector<float>> caps;
if(!lm.forward(ids, proj, logits, {35}, caps)) return 1;          // logits [152681]
std::vector<float> r35; std::vector<int64_t> r35s;
la_parity::load_baseline(base,"lm_layer_35",r35,r35s);            // [1,297,2048]
bool ok35 = la_parity::compare(caps[0], r35, "lm_layer_35", 3e-2f, 3e-2f);
std::vector<float> rl; std::vector<int64_t> rls;
la_parity::load_baseline(base,"logits_step0",rl,rls);            // [152681]
bool okl = la_parity::compare(logits, rl, "logits_step0", 1e-1f, 1e-1f);
// argmax must match (greedy decode depends on it) — reference argmax is 151672 (<ref>)
auto argmax=[](const std::vector<float>& v){ int b=0; for(int i=1;i<(int)v.size();++i) if(v[i]>v[b]) b=i; return b; };
int am_got=argmax(logits), am_ref=argmax(rl);
std::printf("argmax got=%d ref=%d (expect 151672)\n", am_got, am_ref);
bool oka = (am_got==am_ref && am_ref==151672);
```
Make the test return reflect ALL gates (projected? — that's test_projector; here: embeds_after_splice && lm_layer_00 && lm_layer_35 && logits && argmax).

- [ ] **Step 3: Build + run + debug.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_lm -j 2>&1 | tail -5
cd build && ctest -R test_lm --output-on-failure; cd ..
```
Iterate to PASS. lm_layer_00 at ~1e-3 → 36 layers should land lm_layer_35 within 3e-2 and logits within ~1e-1 (logits have large magnitude; the argmax match is the decisive correctness signal). DEBUG: if argmax is wrong but layer_00 passed, suspect accumulation OR the final norm / lm.output tensor (tied — confirm lm.output.weight == lm.tok_embd values). If lm_layer_35 drifts a lot but layer_00 is tight, print per-layer diffs at 0/18/35 (temporarily capture them) — gradual = accumulation (fine), sudden jump = a layer-specific issue.

- [ ] **Step 4: (optional) end-to-end smoke pixel→logits.** If time permits, add a non-gating print: run `VitEncoder::forward` → `merge_patches` → `Projector` → `LMForward::forward` from the reference `pixel_values`, and confirm argmax is still 151672 (this exercises the full M2+M3 chain; expect slightly looser logit match due to M2's 2.65e-3, but argmax should hold). Keep it a print/smoke, not a hard gate.

- [ ] **Step 5: Run full suite + commit (M3 complete).**
```bash
cd build && ctest --output-on-failure; cd ..
git add src/lm.hpp src/lm.cpp tests/test_lm.cpp
git commit -m "M3: full Qwen2 causal prefill -> lm_layer_35 + logits_step0 parity (M3 complete)"
```
Expected: all C++ tests (smoke, model_loader, backend, ggml_extend, vit_posemb, vit_rope, vit_encoder, projector, lm) green; python suite still green.

---

## Milestone exit criteria

**M3 done when:** `tests/test_lm.cpp` passes the full chain against the M0 baselines — projected_tokens (`test_projector`, ≤1e-3), embeds_after_splice (≤1e-4), lm_layer_00 (≤2e-2), lm_layer_35 (≤3e-2), logits_step0 (≤1e-1 AND argmax == 151672) — plus the extended loader test, all green under ctest. The projector + Qwen2 prefill reproduce the HF reference logits for the fixed fixture.

## Roadmap (next plan)

- **M4 — AR decode.** Reuse the Qwen2 layer with a KV-cache (port vibevoice's `ResidentKV` / `qwen2_layer_forward_resident`), greedy/sampled decode loop, coord-token parse (`token_id = 151677 + v`), emit `<box><c><c><c><c></box>`; gate the generated token stream (slow/AR mode) + boxes vs reference. Then M5 (hybrid Parallel Box Decoding), M6 (quantize + CLI + flat C-API). Carry the known M2 generalization item (hardcoded 32×32 grid) when wiring live preprocessing.
