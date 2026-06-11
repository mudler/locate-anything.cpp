# LocateAnything-3B → ggml port (`locate-anything.cpp`) — Design

**Date:** 2026-06-08
**Status:** Approved design, pre-implementation
**Target model:** [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B)

## 1. Goal

Replicate `nvidia/LocateAnything-3B` as a standalone C++/ggml inference engine:
one self-contained GGUF file, no Python / PyTorch / CUDA toolkit at inference
time. Ship a CLI and a flat C-API, with f16 + quantized GGUFs. LocalAI backend
integration is a **separate later effort**, explicitly out of scope here.

Two non-negotiable principles (from `porting-pytorch-models-to-ggml`):

- **Parity-first.** C++ output must be numerically equal to the HF reference,
  proven per-component against dumped reference tensors — not just "plausible
  boxes".
- **Metadata-driven.** Every dimension, hyperparameter, special-token id and the
  tokenizer/vocab live *inside* the GGUF. The loader reads them. Nothing is
  hardcoded; no external config/vocab files shipped.

## 2. What the model is

A native-resolution vision-language model for **visual grounding / open-vocabulary
detection / localization**. All detection logic lives in **token space** — there
is no regression/detection head. The model emits a token sequence containing
coordinate tokens that decode to boxes/points.

Three components + projector:

- **MoonViT** vision encoder (SigLIP-SO400M lineage, native/variable resolution),
  weight prefix `vision_model.`
- **MLP projector** (`mlp1.`): `LayerNorm(4608) → Linear(4608,2048) → GELU →
  Linear(2048,2048)`
- **Qwen2.5-3B-Instruct** language model (`language_model.`), standard Qwen2
  decoder; coordinates are vocabulary tokens, `lm_head` tied to embeddings.

Headline feature: **Parallel Box Decoding (PBD)** — block-wise Multi-Token
Prediction (`block_size = 6`) that emits a whole box in one parallel step. Runs on
identical weights to the pure-AR path; it is a speed layer, not a different model.

### 2.1 Exact dimensions (from `config.json`)

**Text (`Qwen2.5-3B`):** `hidden_size 2048`, `intermediate_size 11008`,
`num_hidden_layers 36`, `num_attention_heads 16`, `num_key_value_heads 2` (GQA,
`head_dim 128`), `vocab_size 152681`, `max_position_embeddings 32768`,
`rope_theta 1000000.0`, `rms_norm_eps 1e-6`, `hidden_act silu`,
`tie_word_embeddings true`. Q/K/V projections **have bias**; o_proj none.
MTP fields: `block_size 6`, `causal_attn false`, `null_token_id 152678`,
`switch_token_id 152679`, `text_mask_token_id 151676`.

**Vision (MoonViT):** `hidden_size 1152`, `intermediate_size 4304`,
`num_hidden_layers 27`, `num_attention_heads 16` (`head_dim 72`), `patch_size 14`,
`merge_kernel_size [2,2]`, `init_pos_emb_height/width 64`. Pre-norm LayerNorm,
packed `wqkv` (bias=True), `wo`, MLP `fc0 → GELU(tanh) → fc1` (bias=True),
`final_layernorm` after blocks.

### 2.2 Special tokens & coordinate math

- `image_token_index 151665` (`<IMG_CONTEXT>`) — replaced 1:1 by projected
  vision tokens.
- Box: `box_start 151668` (`<box>`), `box_end 151669` (`</box>`).
- Coords: `coord_start 151677` (`<0>`) … `coord_end 152677` (`<1000>`), 1001
  tokens. **`token_id = 151677 + clamp(round(coord/dim*1000), 0, 1000)`**; inverse
  `coord_norm = token_id − 151677`, `pixel = coord_norm/1000 * image_dim`.
- Ref label: `ref_start 151672` (`<ref>`), `ref_end 151673` (`</ref>`).
- `none 4064` (empty `<box>none</box>`), `null 152678`, `switch 152679`,
  `text_mask 151676`, `im_end / eos 151645`, `bos 151643`.

### 2.3 Output grammar

- Box: `<box><x1><y1><x2><y2></box>` → 6 tokens.
- Point: `<box><x><y></box>` → 4 tokens.
- Empty: `<box>none</box>` → `[151668, 4064, 151669]`.
- Image placeholder expansion: each image → `<img>` + `<IMG_CONTEXT>`×N +
  `</img>`, where **N = (h/14)·(w/14)/4** merged vision tokens.

### 2.4 Image preprocessing

`image_mean/std = [0.5,0.5,0.5]` (normalize to [-1,1]), `patch_size 14`,
`merge_kernel_size [2,2]`, `in_token_limit 25600`. Resize so
`(w/14)*(h/14) ≤ in_token_limit`, pad/resize to a multiple of `merge*patch = 28`
per dim, grid side `< 512` patches. Patchify → `(num_patches, 3, 14, 14)` with
`grid_hw = (H/14, W/14)`. Outputs `pixel_values` (concatenated patches) and
`image_grid_hws` (`[h,w]` per image).

### 2.5 Parity fixture (anchor)

Open-vocabulary detection. Fixed prompt:
`"Locate all the instances that matches the following description: {cats}."`
(categories joined with `</c>`). One fixed image + this prompt is the gold fixture
all milestones gate against. Chat template is standard Qwen ChatML.

## 3. Approach / reuse strategy

Fork sibling-repo *structure* (`CMakeLists.txt`, `src/`,
`include/<name>_capi.h` + `<name>.h`, `scripts/convert_*`, `model_loader`,
`backend`) from `parakeet.cpp` / `vibevoice.cpp`. Selectively port code:

| Component | Source | Effort |
|---|---|---|
| Qwen2.5-3B decoder + KV cache | port `vibevoice.cpp/src/qwen2.*`; **cross-check vs `llama.cpp` `build_qwen2`** | Low — adapt |
| Qwen2 BPE tokenizer + 1038 added tokens | port vibevoice tokenizer + converter | Low–Med |
| Image preprocess + box render/CLI | adapt `rt-detr.cpp` patterns | Med |
| GGUF converter, quantize, C-API shell, `backend.cpp` | sibling templates | Low |
| **MoonViT vision tower** | **new** | **High** |
| **MLP projector + vision-token injection** | **new** | Med |
| **Coord-token decode + Parallel Box Decoding** | **new** | **High** |

**Reference repos.** `llama.cpp` is the canonical Qwen2 reference: its
`build_qwen2` graph, GGUF tensor naming (`blk.N.attn_q.weight`,
`token_embd.weight`, `output_norm.weight`, …), GGUF KV conventions, and
RoPE-NEOX handling are the standard to follow for our GGUF schema and to
cross-check the vibevoice Qwen2 port against. `vibevoice.cpp` supplies a working
ggml Qwen2 layer (GQA + RoPE, Q/K/V bias, SwiGLU, persistent `ResidentKV` cache,
flash-attn) that is hparam-driven and matches LocateAnything's LM architecture.
`rt-detr.cpp` supplies image-preprocessing, COCO-style benchmarking and box
rendering/annotation patterns. `whisper.cpp` for general ggml conventions.

Decision: vendor/port `qwen2.*` into this repo (no shared lib) — siblings each
keep their own copy; matches established practice.

## 4. Architecture & data flow

```
image ─► native-res patchify (14px, multiple-of-28) ─► MoonViT (27 layers)
          ─► 2×2 patch merge (1152→4608) ─► MLP projector (4608→2048)
          ─► vision tokens ┐
text ─► Qwen2 BPE ─► embed ─┴─► splice vision tokens into <IMG_CONTEXT> slots
          ─► Qwen2.5-3B (36 layers, GQA 16/2, head_dim 128, vocab 152681)
          ─► lm_head ─► coord-token decode (AR + hybrid MTP)
          ─► <box><c><c><c><c></box> ─► pixel = (id−151677)/1000 × dim ─► boxes
```

### 4.1 MoonViT specifics (the high-risk component)

- Patch embed: `Conv2d(3→1152, kernel=stride=14)`
  (`patch_embed.proj.{weight,bias}`).
- **Learnable 2D absolute pos-emb** (`patch_embed.pos_emb.weight`, `[64,64,1152]`)
  **bicubic-interpolated** to the actual `h×w` grid, added to patch embeddings —
  *in addition to* RoPE.
- **2D RoPE** (`theta_base 10000`, `dim = head_dim = 72`, max grid 512×512),
  interleaved x/y: even pairs use x-position freq, odd pairs use y-position freq;
  freqs over `arange(0, dim, 4)`.
- 27 pre-norm layers: `norm0`/`norm1` LayerNorm, packed `wqkv` (bias),
  `wo` (bias), MLP `fc0 → GELU(tanh) → fc1` (bias). Attention **bidirectional,
  block-diagonal** per image (cu_seqlens mask). `final_layernorm` after blocks.
- **Patch merger**: group 2×2 neighbours → concat channels → `1152*4 = 4608` per
  merged token; output `(h/2 · w/2, 4608)` per image.

### 4.2 Decode (AR + hybrid MTP)

- Vision features extracted once, projected, injected at the first forward via the
  `image_token_index` slots. Batch size 1, KV-cache required.
- **AR path (slow / fallback):** standard Qwen2 causal decode, last-position
  logits, greedy or sampled (temperature, top_p, top_k, repetition_penalty;
  greedy at temperature 0). Built first as the stepping stone and as the hybrid
  fallback.
- **MTP path (fast / hybrid, `n_future_tokens = 6`):** append last token +
  `(n−1)` copies of `text_mask_token_id 151676`; one forward; read logits of the
  last 6 positions; future-slot `position_ids -= 1`; KV-cache truncated back to
  committed length each round. Box assembly (`decode_bbox_avg`/`is_valid_box_frame`):
  `p(box_start)@pos0 ≥ 0.7`; end check `box_end/null/im_end @pos5 ≥ 0.2`; empty-box
  pattern; coord positions 1–4 take top-k=5, keep highest-prob token in
  `[151677,152677]`; "abnormal" position (top-prob<0.9 + spread>60 ids) zeroed →
  AR fallback. `handle_pattern` classifies block → `im_end / empty_box /
  coord_box / point_box / error_box / ref_object`. Hybrid: start MTP, on
  `error_box` switch to AR token-by-token, on `box_end` switch back; terminate on
  `im_end 151645`.

## 5. Milestone plan (each ends at a parity gate → commit → push)

- **M0 — Scaffold + download + reference dumps.**
  Repo skeleton (CMake, dirs, third_party ggml submodule). **Download step:**
  script to fetch the ~7.66 GB / 770-tensor / 2-shard safetensors + tokenizer
  files from HF into a local cache. Python reference script runs the real HF model
  on the fixed image+prompt and dumps gold tensors: patchified pixels,
  post-pos-emb, per-ViT-layer, merged tokens, projected tokens,
  post-embed-splice, per-LM-layer, step-0 logits, full token stream, final boxes.
  Capture exact decode details (hybrid + AR).
- **M1 — Converter + GGUF.** `convert_locateanything_to_gguf.py`: all config as
  KV (vision + text hparams, every special-token id, coord base 151677,
  `block_size 6`), embed tokenizer (`vocab.json` + `merges.txt` +
  `added_tokens.json`, 1038 added), then 770 weights. GGUF tensor naming follows
  llama.cpp conventions for the LM; vision/projector get a documented scheme.
  Loader reads every dim/param — nothing hardcoded.
- **M2 — MoonViT parity.** Patchify → pos-emb (bicubic-interp) → 27 layers →
  merger, gated tensor-by-tensor vs M0 dumps. Highest-risk milestone.
- **M3 — Projector + LM forward parity.** MLP projector, vision-token splice,
  full Qwen2 forward → step-0 logits match dump.
- **M4 — AR decode parity.** AR loop emits exact token stream for the detection
  prompt; box parse matches reference boxes.
- **M5 — Hybrid Parallel Box Decoding parity.** MTP block-parallel positions,
  per-round KV truncation, frame validation, `handle_pattern`, AR fallback. Parity
  vs reference hybrid output.
- **M6 — Quantize + CLI + C-API.** f16 / q8_0 / q6_k / q5_k / q4_k (accuracy
  near-lossless at f16/q8, monotone degradation). `locate-anything-cli detect
  --annotated`. Flat C ABI (`la_load`/`la_free`/`la_locate`/`la_free_string`/
  `la_last_error`/`la_abi_version`), opaque handles, C types only.

## 6. Testing / parity strategy

Primary test = **per-component numerical equality** to the HF reference, asserted
on M0's dumped tensors (bit-exact where possible; tight tolerance for f16 /
accumulation). End-to-end box IoU vs reference is a secondary check. Match the
reference decode exactly (coord token base, top-k tie-breaking, blank/null
handling, frame thresholds). The detection prompt + fixed image is the standing
fixture across all milestones.

## 7. ggml performance (applied from M3 on)

Persistent `ggml_gallocr` (no per-call realloc), zero-copy mmap'd weights, build
with `GGML_LLAMAFILE`, fused forward graph, persistent KV (`ResidentKV` pattern).
Run vision patchify/pos-emb on-device where practical to avoid host round-trips.

## 8. Out of scope

LocalAI backend (L0–L5), non-detection task tuning beyond shared decode, training,
MagiAttention CUDA kernel (SDPA-equivalent path only), batch > 1.
