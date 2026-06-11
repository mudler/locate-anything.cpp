# LocateAnything-3B ggml Port — M5 (Hybrid Parallel Box Decoding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the hybrid Parallel Box Decoding (block-wise MTP, block_size=6) decode path in C++ — block-diffusion masked forward, softmax-gated box-frame decode, block classifier, and the MTP↔AR hybrid control flow — reproducing the reference `generation_mode="hybrid"` token stream exactly.

**Architecture:** Each MTP round appends `[last_committed_token, text_mask×5]`, runs them (plus the not-yet-cached committed tokens) through the existing resident Qwen2 layers under a block-diffusion attention mask, reads the 6 block-position logits, and decodes a whole box/ref/empty frame per round via softmax-threshold gating (not argmax). On a malformed frame it falls back to plain-AR (M4) until a `</box>`, then resumes MTP. Gated component-by-component (masked forward → box-frame decode → classifier+control-flow → end-to-end stream) against new MTP reference dumps.

**Tech Stack:** C++17, ggml (embedded), CMake/CTest, Python (transformers/gguf) for the MTP reference.

**Reference material (read before starting):**
- Exact MTP mechanics: `models/LocateAnything-3B/modeling_locateanything.py` (`generate` loop ~L464, `_prepare_inputs_in_mtp` L361-396, `_sample_token_in_mtp` L413, `_sample_token_in_ar` L430, KV-trunc L483), `generate_utils.py` (`sample_tokens` L135, `is_valid_box_frame` L246, `decode_bbox_avg` L276, `decode_ref` L364, `handle_pattern` L408), `mask_sdpa_utils.py` (`update_causal_mask_for_one_gen_window_2d` L104). Restated inline per task; verify against these — quote exactly.
- Existing repo (M4): `la::qwen2_layer_forward_resident(ctx, x, pos, mask, kv, layer, w, hp)` (accepts ARBITRARY pos/mask — already M5-ready), `la::ResidentKV`, `la::LMForward` (`embed_and_splice`, `forward`, `decode_greedy_reprefill`, `decode_greedy_resident`), `la::Backend` (`add_graph_root`, `add_int32_input_nd`, `add_graph_input_nd`), `la::boxes` parser, `la_parity` harness (incl. `load_kv_u32`, `load_kv_str_array`).
- The M0 `dumps/reference.gguf`... the M0 dump used `generation_mode="hybrid"` and `dumps/manifest.json` holds the 258-token hybrid `token_stream` + `generated_string` — the end-to-end M5 gold.

**Constants (config.json / generate_utils):** block_size/n_future_tokens = **6**; causal_attn = **False**; text_mask 151676; box_start 151668, box_end 151669; coord_start 151677, coord_end 152677; ref_start 151672, ref_end 151673; none 4064, null 152678; eos/im_end 151645; image 151665.

**LOAD-BEARING caveats (verified):**
1. **Effective `keep_k = 4`, not 5.** `_sample_token_in_mtp` passes `keep_k=5` but `sample_tokens` calls `decode_bbox_avg(keep_k=generate_kwargs.get('keep_k_avg', 4))` — and `keep_k_avg` is never set, so it's **4**. The `keep_k=5` / `is_valid_box_frame(topk=...)` args are dead. Task 1 MUST instrument the reference to confirm the value actually used.
2. **MTP committed tokens are NOT argmax.** `decode_bbox_avg` does softmax-threshold gating + coord-constrained top-k + abnormal-zeroing. Plain argmax is only the `is_box_empty` fallback and the AR path. The gates are on **probabilities** → softmax the 6 positions first.
3. **Attention = sdpa block-diffusion** (CPU fallback of magi; the path the dump exercises). Port the additive mask form.

**Scope:** M5 ends with the C++ hybrid decode reproducing the 258-token hybrid reference stream + its boxes. M6 = quantize + CLI + flat C-API.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/dump_mtp_reference.py` | Instrument hybrid `generate`: per-MTP-round `logits6 [6,vocab]`, round-0 input_ids/position_ids/mask, the effective keep_k, + a per-round decision trace (use_mtp, out_type, committed, box_avg, x0, new_tokens) → `dumps/reference_mtp.gguf` + `dumps/reference_mtp_trace.json` |
| `src/mtp.{hpp,cpp}` | Pure decode logic (no ggml): `is_valid_box_frame`, `decode_bbox_avg`, `decode_ref`, `sample_box` (the sample_tokens wrapper + fallback), `handle_pattern`, and the `HybridState` control-flow state machine |
| `src/lm.{hpp,cpp}` (modify) | `decode_hybrid(...)` driver: build MTP block input/mask/positions, run resident forward, call `mtp::` decode, manage KV/past_len + AR fallback |
| `tests/test_mtp.cpp` | Gate: masked-forward logits6, box-frame decode (on captured logits), classifier+control-flow trace replay, full hybrid stream |
| `tests/CMakeLists.txt`, `CMakeLists.txt` (modify) | Wire test + new sources + `LA_TEST_MTP` env |

**Decomposition rationale:** `mtp.{hpp,cpp}` is pure logic (softmax + thresholds + classification + a state machine) gateable against captured reference intermediates with NO correct forward. The masked forward reuses the M4 resident layer with a new mask/position builder. `decode_hybrid` integrates. Each task gates a dumped intermediate before the next.

---

## Task 1: MTP reference dump (per-round logits + decision trace)

The M0 hybrid `token_stream` (258 ids) is the end-to-end gold, but M5 needs per-round intermediates to gate the masked forward and the decode/control-flow independently. Instrument a hybrid generate run.

**Files:**
- Create: `scripts/dump_mtp_reference.py`

- [ ] **Step 1: Write `scripts/dump_mtp_reference.py`.** Run the fixture through `model.generate(generation_mode="hybrid", do_sample=False, use_cache=True)` with hooks/monkey-patches capturing, per MTP round: the 6-position logits, and the decision record. Capture by wrapping the model's `_sample_token_in_mtp`/`_sample_token_in_ar` (they're methods defined inside `generate` via closure — so instead monkey-patch `generate_utils.sample_tokens` and `generate_utils.handle_pattern`, and hook `model.language_model` forward for the logits). Concretely:
```python
#!/usr/bin/env python3
"""Instrument hybrid generate to dump per-round MTP intermediates for gating M5."""
import json
from pathlib import Path
import numpy as np
import torch
import gguf
import scripts.la_reference as R
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
TEXT_MASK = 151676

def main():
    cfg, model, processor, inputs, spec = R.build_inputs()
    tok = processor.tokenizer
    rounds = []          # per-round decision trace
    logits6 = []         # per MTP round: [6, vocab] f32
    ar_logits = []       # per AR round: [vocab] (optional)
    # Hook the LM forward to grab the last-N logits each call; we tag MTP rounds by
    # whether the round's input ended in TEXT_MASK (the generate loop builds that).
    # Simplest robust capture: wrap model.language_model.__call__.
    orig_lm = model.language_model.forward
    last = {}
    def lm_hook(*a, **kw):
        out = orig_lm(*a, **kw)
        ids = kw.get("input_ids", a[0] if a else None)
        last["logits"] = out.logits.detach()
        last["last_id"] = int(ids[0, -1].item()) if ids is not None else None
        return out
    model.language_model.forward = lm_hook
    # Wrap handle_pattern to record out_type + committed tokens per MTP round.
    import generate_utils as gu  # remote-code module is importable once the model is loaded; if not, import via the model's module path
    orig_hp = gu.handle_pattern
    def hp_hook(x0, token_ids, mode):
        r = orig_hp(x0, token_ids, mode)
        rounds.append({"kind":"mtp", "out_type": r["type"],
                       "committed": [int(t) for t in r["tokens"]],
                       "new_tokens": [int(t) for t in (x0.tolist() if hasattr(x0,'tolist') else x0)]})
        # also stash the 6-position logits captured by lm_hook for this round
        if "logits" in last and last["logits"].shape[1] >= 6:
            logits6.append(last["logits"][0, -6:, :].float().cpu().numpy())
        return r
    gu.handle_pattern = hp_hook
    with torch.no_grad():
        s = model.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"], image_grid_hws=inputs["image_grid_hws"],
                           tokenizer=tok, max_new_tokens=256, generation_mode="hybrid", use_cache=True)
    ids = np.array(tok(s, add_special_tokens=False)["input_ids"], dtype=np.int32)
    L6 = np.stack(logits6, axis=0).astype(np.float32) if logits6 else np.zeros((0,6,1),np.float32)  # [n_mtp,6,vocab]
    out = ROOT/"dumps"/"reference_mtp.gguf"
    w = gguf.GGUFWriter(str(out), "la_mtp")
    w.add_tensor("hybrid_token_ids", ids)
    w.add_tensor("mtp_logits6", np.ascontiguousarray(L6))      # [n_mtp_rounds, 6, vocab]
    w.add_uint32("la_mtp.n_tokens", len(ids)); w.add_uint32("la_mtp.n_mtp_rounds", L6.shape[0])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    json.dump({"rounds": rounds, "n_tokens": int(len(ids)),
               "first16": ids[:16].tolist(), "string": s[:300]},
              open(ROOT/"dumps"/"reference_mtp_trace.json","w"), indent=1)
    print(f"wrote {out}: n_tokens={len(ids)} n_mtp_rounds={L6.shape[0]} L6={L6.shape}")
    print("first rounds:", rounds[:5])
```
NOTE: the exact monkey-patch points must be verified against the vendored code — `handle_pattern` and `sample_tokens` live in `generate_utils.py` (imported by `modeling_locateanything.py`); patch the module object the model actually calls (it may be `model.__class__`'s module namespace, e.g. `sys.modules[model.__class__.__module__]`'s `handle_pattern`/`sample_tokens` references, or the `generate_utils` module — whichever the `generate` closure resolves). Read `modeling_locateanything.py` to see how it imports these (e.g. `from .generate_utils import handle_pattern, sample_tokens`) and patch THAT bound reference. The robust capture of `logits6` is the LM's `outputs.logits[0,-6:,:]` on MTP rounds. ALSO capture and print the effective `keep_k` by wrapping `decode_bbox_avg` and printing its `keep_k` arg on the first call (confirm it's 4 — caveat #1). If your hooks can't cleanly capture per-round logits aligned with rounds, simplify: capture EVERY lm forward's `logits[0,-6:,:]` when `last_id==TEXT_MASK` (MTP rounds end in a mask token) — that aligns 1:1 with MTP rounds.

- [ ] **Step 2: Generate it.**
```bash
. .venv/bin/activate
python -m scripts.dump_mtp_reference
```
Expected: prints `n_tokens=258` (the hybrid stream), `n_mtp_rounds=N` (some dozens), `L6=[N,6,152681]`, the first rounds' out_types (round 0 likely `ref_object` for `<ref>cat</ref>`, later rounds `coord_box`), and the confirmed `keep_k`. This runs the real model in hybrid mode (~minutes on CPU). The gguf (~N×6×152681×4 bytes ≈ tens-to-hundreds of MB) + json are gitignored. SANITY: the `string` should match the M0 hybrid `generated_string` (the same fixture+mode); `first16` should start with the prompt-continuation tokens. Report the confirmed keep_k and n_mtp_rounds.

- [ ] **Step 3: Commit (script only; dumps gitignored).**
```bash
git add scripts/dump_mtp_reference.py
git commit -m "M5: instrument hybrid generate -> per-round MTP logits + decision trace"
```
Author `mudler <mudler@localai.io>`; end every M5 commit with:
```

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: Block-diffusion masked MTP forward → gate per-round logits6

Build one MTP round's forward in C++: input = the committed-but-uncached tokens + `[last_committed, text_mask×5]`; positions per §MTP; block-diffusion mask; run through the resident layers; read the 6 block-position logits. Gate against `mtp_logits6[round]`.

**Files:**
- Modify: `src/lm.{hpp,cpp}` (add `mtp_block_forward` helper + a mask/position builder)
- Create: `tests/test_mtp.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add the MTP block-diffusion mask + position builders to `src/lm.cpp`** (host-side, fed as graph inputs). For a round with `cached_len = past_len` (KV holds 0..past_len-1), `n_recompute` newly-committed-uncached tokens, and 6 block slots: `n_new = n_recompute + 6`, `full_len = past_len + n_new`. Positions (i32, length n_new): the recompute tokens get `[past_len .. past_len+n_recompute-1]`; the 6 block slots get `[base-1, base, base+1, base+2, base+3, base+4]` where `base = past_len + n_recompute` (i.e. slot0 = base-1 = the last committed token's position, slots 1-5 = base..base+4). Mask (f32, `[full_len, n_new]`, key-major: `m[q*full_len + key]`):
  - For the first `n_recompute` query rows (recomputed committed tokens): plain causal — `m[q,key] = key > (past_len+q) ? -INF : 0`.
  - For the 6 block query rows (q = n_recompute + b, b in 0..5, absolute pos = base-1 for b=0 else base+b-1): start causal up to the row's own absolute position, THEN set the 6×6 block (keys `full_len-6 .. full_len-1`) all to 0 (bidirectional), THEN set key `full_len-7` (= the last committed token, absolute pos base-1) to -INF for ALL 6 block rows.
  Implement as: build the full causal mask first (`m[q,key] = key > abspos(q) ? -INF : 0` using each row's absolute position), then apply the two block overrides. Write a helper `build_mtp_mask(past_len, n_recompute, /*block=6*/, out_mask)` + `build_mtp_positions(...)`. VERIFY against `mask_sdpa_utils.update_causal_mask_for_one_gen_window_2d` (the `[-6:,-6:]=0` and `[-6:,-7]=-inf` overrides) — quote it in a comment.

- [ ] **Step 2: Add `mtp_block_forward` to `src/lm.{hpp,cpp}`.**
```cpp
    // Run one MTP block forward through the resident layers. `x_host` = embeddings of
    // the n_new = n_recompute+6 tokens [H*n_new]; kv.past_len must be set to the cached
    // length. Returns the 6 block-position logits [6*vocab] (after final norm + lm_head).
    // Advances kv.past_len by n_commit (the committed count) — NOT n_new (6 block slots
    // are speculative and overwritten next round). Caller passes n_commit after deciding.
    bool mtp_block_forward(ResidentKV& kv, const std::vector<float>& x_host, int n_new,
                           int n_recompute, std::vector<float>& logits6_out);
```
Implementation: build pos (i32, n_new) + mask (f32, full_len×n_new) via the Step-1 builders; in one `be_.compute` lambda, feed x [H,n_new], pos, mask; loop layers via `qwen2_layer_forward_resident(ctx, x, pos, mask, kv, il, load_qwen2_layer(ml_,il), hp)` registering `k_write`/`v_write` via `add_graph_root` interleaved per layer; final `rms_norm(output_norm)`; slice the LAST 6 columns `[H,6]` (view at `(n_new-6)*nb[1]`, cont); `mul_mat(lm.output)` → `[vocab,6]`; read back → `logits6_out` (6*vocab, row-major per position). Do NOT advance kv.past_len here (the caller advances by n_commit after the decode decides how many tokens the block commits — for the gate, the test advances by 0 or re-inits).
NOTE: for the FIRST MTP round the embeddings are the prompt prefill embeds (vision-spliced) for the recompute part — but actually round 0's recompute is the whole prompt. To gate round 0 in isolation, the test does: prefill the prompt into the KV (kv.past_len=prompt_len) via a normal resident prefill, THEN run an MTP block with n_recompute=0 (the prompt is already cached) and the 6 slots = [last_prompt_token_embed, text_mask_embed×5]. So `mtp_block_forward` with n_recompute=0, n_new=6 is the round-0 case. Use that for the gate.

- [ ] **Step 3: Write `tests/test_mtp.cpp` (round-0 logits6 gate).** Prefill the prompt into a ResidentKV (reuse the decode_greedy_resident prefill path or a minimal prefill), then run the round-0 MTP block (n_recompute=0, embeds = last-prompt-token + 5×text_mask), and compare the 6×vocab logits against `mtp_logits6[0]` from `reference_mtp.gguf`. Gate atol/rtol ~5e-2 (36 layers + the block mask; the prefill logits matched ~7e-5 in M3 so the block forward should be tight — but the block-diffusion mask is new, so allow some margin and report the actual diff). Embeddings: `text_mask` (151676) embeds via the same host row-gather as other tokens.
```cpp
// pseudocode of the gate body
la::LMForward lm(ml, be);
ResidentKV kv; kv.init(be, n_layers, head_dim, n_kv, prompt_len+64);
lm.prefill_resident(prompt_ids, projected, kv);   // fills KV 0..prompt_len-1, past_len=prompt_len
// round-0 block: slot0 = last prompt token, slots1-5 = text_mask
std::vector<int32_t> block = { prompt_ids.back(), 151676,151676,151676,151676,151676 };
std::vector<float> xblk; lm.embed_tokens_host(block, xblk);   // [H*6]
std::vector<float> got6;
lm.mtp_block_forward(kv, xblk, /*n_new*/6, /*n_recompute*/0, got6);
std::vector<float> ref6; std::vector<int64_t> rs;
la_parity::load_baseline(mtp, "mtp_logits6", ref6, rs);       // [n_rounds,6,vocab]; take first 6*vocab
bool ok = la_parity::compare(got6, std::vector<float>(ref6.begin(), ref6.begin()+got6.size()),
                             "mtp_logits6_round0", 5e-2f, 5e-2f);
```
NOTE: this needs small helpers `prefill_resident(prompt_ids, projected, kv)` (the prompt-prefill chunk extracted from `decode_greedy_resident`) and `embed_tokens_host(ids, out)` (host row-gather, also already inside embed_and_splice). Factor them out as reusable methods (refactor, don't duplicate). The `text_mask` row exists in tok_embd (it's a normal vocab id).

- [ ] **Step 4: Wire (`la_add_test(test_mtp)`; add `LA_TEST_MTP=${CMAKE_SOURCE_DIR}/dumps/reference_mtp.gguf` to the env in `la_add_test`), build, run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_mtp -j 2>&1 | tail -5
cd build && ctest -R test_mtp --output-on-failure; cd ..
```
Expected: `mtp_logits6_round0` within tolerance. DEBUG: if it's way off, the block-diffusion mask is wrong — check the `[-6:,-6:]=0` (bidirectional block) and `[-6:,-7]=-inf` (block last-committed) and the decremented slot0 position. If only slot0 is off, the position tie (slot0 = base-1) is wrong. Compare against the reference round-0 mask if you dumped it.

- [ ] **Step 5: Commit.**
```bash
git add src/lm.hpp src/lm.cpp tests/test_mtp.cpp tests/CMakeLists.txt
git commit -m "M5: block-diffusion masked MTP forward -> per-round logits6 parity"
```

---

## Task 3: Box-frame decode (`is_valid_box_frame` + `decode_bbox_avg` + `decode_ref` + fallback) → gate on captured logits

Pure logic over a 6-position softmax → the committed-block tokens. Gate it by feeding the CAPTURED reference `logits6[r]` and matching the reference round's `new_tokens`/`out_type` (isolates decode from the forward).

**Files:**
- Create: `src/mtp.hpp`, `src/mtp.cpp`
- Modify: `tests/test_mtp.cpp`, `CMakeLists.txt`

- [ ] **Step 1: Write `src/mtp.hpp`** — token-id constants + the decode API.
```cpp
#pragma once
#include <vector>
#include <cstdint>
namespace la::mtp {
struct Tokens { int box_start=151668, box_end=151669, coord_start=151677, coord_end=152677,
                ref_start=151672, ref_end=151673, none=4064, null=152678,
                text_mask=151676, im_end=151645; };
// probs6: [6][vocab] softmax of the 6 block-position logits.
// Returns the decoded block tokens (variable length): a 6-tuple box, the empty-box
// 6-tuple, a ref sequence, or the length-1 {0} fallback. keep_k defaults to 4.
std::vector<int32_t> sample_box(const std::vector<std::vector<float>>& probs6,
                                int vocab, int keep_k=4);
// classify the chosen block into committed tokens + out_type (see handle_pattern).
struct Pattern { std::string type; std::vector<int32_t> tokens; bool terminal=false; bool need_ar=false; };
Pattern handle_pattern(const std::vector<int32_t>& x0);
}
```

- [ ] **Step 2: Write `src/mtp.cpp`** — port `is_valid_box_frame` + `decode_bbox_avg` + `decode_ref` + the `sample_tokens` fallback chain, EXACTLY per `generate_utils.py`. The decoder operates on probs (softmax of logits). READ `generate_utils.py` L246-405 and reproduce verbatim:
  - `is_valid_box_frame(probs)`: `p_start=probs[0][box_start]`; if `p_start>=0.7`: empty-box if `probs[1][none]>0.2 && probs[2][box_end]>0.2 && probs[3][null]>0.1 && probs[4][null]>0.1` → "empty_box". Then `end_score = probs[5][box_end]+probs[5][null]+probs[5][im_end]`; if `end_score>=0.2` → "legal_box" else "illegal_box".
  - `decode_bbox_avg(probs, keep_k=4, mode=hybrid)`: empty_box → `[box_start, none, box_end, null, null, null]`; illegal_box → return empty (None). Else for positions 1..4: top-k (k=4) of `probs[pos]`; valid = ids in `[coord_start, coord_end]`; if any position has no valid coord → return empty (None). `first_valid` = the highest-prob valid id per position. Hybrid abnormal rule per position: `valid_counts = #valid in topk`; `valid_max/min` over valid ids; `is_abnormal = (first_valid_prob<0.9) && (valid_counts>1) && ((valid_max-valid_min)>60)` → coord=0 else first_valid_id. Return `[box_start, c1,c2,c3,c4, box_end]`.
  - `decode_ref(probs, ...)`: (READ L364-405 for exact logic) — if `probs[0][ref_start]>=0.6`, build a ref sequence picking the highest-prob NON-coord token per position until ref_end/condition; else None. Reproduce exactly.
  - `sample_box`: try `decode_bbox_avg`; if empty, try `decode_ref`; if empty, return `{0}` (the length-1 fallback).
  - `handle_pattern(x0)`: port L408-504 exactly (the im_end / empty_box / box_start→coord_box/point_box/error_box / ref_object branches with the exact committed tokens). For `error_box`, `tokens = x0[:coord_ix]`, `need_ar=true`.
NOTE: where the reference uses `box_avg[0]` vs `x0[0]` (the `is_box_empty` fallback in `_sample_token_in_mtp`): `is_box_empty = (sample_box result == {0})`; if so `new_tokens = the 6 plain-argmax ids`, else `new_tokens = sample_box result`. So `sample_box` returning `{0}` signals "use argmax". Expose that (e.g. return `{0}` and let the caller compute the 6 argmax). Put the argmax-of-6 in the caller (Task 5) or a helper here.

- [ ] **Step 3: Gate in `tests/test_mtp.cpp`** — feed CAPTURED reference logits, match reference `new_tokens`. Load `mtp_logits6` [n_rounds,6,vocab] + the trace json's `rounds` (parse the `new_tokens`/`out_type` — read the json in C++ with a tiny hand parser, or convert the needed fields into the gguf as arrays in Task 1; simplest: in Task 1 ALSO write `mtp_new_tokens` as a flat gguf array + `mtp_round_lens` so the C++ test reads them without JSON). For each captured MTP round r: softmax `logits6[r]` → probs6 → `sample_box(probs6)` (+ argmax fallback) → `new_tokens`; assert it equals the reference `new_tokens[r]`. Gate exact (these are token ids). Start by asserting round 0 (likely ref_object) and the first `coord_box` round, then all rounds.
NOTE: to avoid JSON parsing in C++, extend Task 1's gguf with: `mtp_new_tokens` (flat i32, all rounds' new_tokens concatenated), `mtp_new_lens` (i32 per round), `mtp_committed` (flat i32) + `mtp_committed_lens`, `mtp_out_types` (string array). Then the C++ test reads them via `load_baseline_i32`/`load_kv_str_array`. ADD these to dump_mtp_reference.py in this task (amend it) so the C++ gates have no JSON dependency.

- [ ] **Step 4: Build + run + debug.** Add `src/mtp.cpp` to LA_SOURCES. DEBUG: if a round's decode mismatches, print the reference vs computed new_tokens and the gating values (p_start, end_score, per-position top-k). Most likely culprits: keep_k (must be the value Task 1 confirmed — 4), the abnormal-rule constants (0.9, 60, valid_counts>1), or decode_ref's non-coord selection. Run to green.

- [ ] **Step 5: Commit.**
```bash
git add src/mtp.hpp src/mtp.cpp tests/test_mtp.cpp scripts/dump_mtp_reference.py CMakeLists.txt
git commit -m "M5: box-frame decode (decode_bbox_avg/decode_ref/handle_pattern) -> captured-logits parity"
```

---

## Task 4: Hybrid control-flow state machine → gate the per-round trace

Port the MTP↔AR control flow as a pure state machine and gate it by replaying the captured per-round `new_tokens` (MTP) / argmax (AR) through it, matching the reference `committed`/`out_type` sequence and the MTP/AR mode switches — WITHOUT a forward.

**Files:**
- Modify: `src/mtp.{hpp,cpp}` (add `HybridState`)
- Modify: `tests/test_mtp.cpp`

- [ ] **Step 1: Add `HybridState` to `src/mtp.{hpp,cpp}`.**
```cpp
struct HybridState { bool use_mtp = true; bool terminated = false; };
// MTP step: given the chosen block tokens new_tokens (from sample_box+fallback),
// classify + return committed tokens, update state (error_box -> use_mtp=false).
std::vector<int32_t> hybrid_mtp_step(HybridState& st, const std::vector<int32_t>& new_tokens);
// AR step: given the single argmax token, classify (box_end_ar -> use_mtp=true,
// coord_ar/continue -> stay AR, else im_end/terminate), return committed token(s).
std::vector<int32_t> hybrid_ar_step(HybridState& st, int32_t ar_token);
```
Implement per `modeling_locateanything.py` L464-513: `hybrid_mtp_step` calls `handle_pattern(new_tokens)`; if type `im_end` → set terminated, commit `[151645]`; `error_box` → `use_mtp=false`, commit `tokens`; else commit `tokens`. `hybrid_ar_step`: if `ar_token==box_end (151669)` → commit it, `use_mtp=true` (box_end_ar); elif coord token (151677..152677) or none (4064) → commit (coord_ar, stay AR); else → terminated, commit `[151645]` (im_end). (Quote the exact branches.)

- [ ] **Step 2: Gate in `tests/test_mtp.cpp`** — replay the reference trace. For each round in the reference trace: if it's an MTP round, feed the reference `new_tokens[r]` to `hybrid_mtp_step`; if AR, feed the reference AR token. Assert the committed tokens + the running `use_mtp`/terminated match the reference `committed`/`out_type` exactly, and that the concatenation of all committed tokens equals the full `hybrid_token_ids` (258). This proves the control-flow state machine independent of forward+decode.
NOTE: the reference trace must distinguish MTP vs AR rounds and (for AR rounds) carry the AR token. Ensure Task 1's dump records `kind` (mtp/ar) per round + the AR token for AR rounds (amend dump_mtp_reference.py if needed; the `_sample_token_in_ar` path — capture its `out_token` + `out_type`). If the fixture's hybrid run never enters AR fallback (no error_box), the AR path is untested by this fixture — note it and ensure `hybrid_ar_step` is still unit-tested on synthetic inputs (box_end_ar, coord_ar, im_end) so the AR transitions are covered even without a fixture AR round.

- [ ] **Step 3: Build + run + debug.** DEBUG: a control-flow mismatch shows as a wrong committed-token sequence or a wrong mode switch; print the round index, reference vs computed. Run to green (the concatenated committed tokens == 258-token reference stream proves the replay is faithful).

- [ ] **Step 4: Commit.**
```bash
git add src/mtp.hpp src/mtp.cpp tests/test_mtp.cpp scripts/dump_mtp_reference.py
git commit -m "M5: hybrid MTP<->AR control-flow state machine -> trace-replay parity"
```

---

## Task 5: Full hybrid decode → gate the 258-token hybrid stream (M5 exit)

Integrate the masked forward (Task 2) + box-frame decode (Task 3) + control flow (Task 4) + AR fallback (M4 resident step) + KV management into `LMForward::decode_hybrid`, and gate the full hybrid token stream.

**Files:**
- Modify: `src/lm.{hpp,cpp}` (add `decode_hybrid`)
- Modify: `tests/test_mtp.cpp`

- [ ] **Step 1: Add `decode_hybrid` to `src/lm.{hpp,cpp}`.**
```cpp
    // Hybrid Parallel Box Decoding. Returns the generated token ids (matching the
    // reference generation_mode="hybrid" stream). Uses the resident KV-cache.
    bool decode_hybrid(const std::vector<int32_t>& prompt_ids,
                       const std::vector<float>& projected_host,
                       int max_new, std::vector<int32_t>& out_ids);
```
Driver (port `generate` loop L464-513 + the KV/past_len discipline from §MTP):
1. `ResidentKV kv; kv.init(be_, n_layers, head_dim, n_kv, prompt_len+max_new+64);` `prefill_resident(prompt_ids, projected, kv)` → `kv.past_len = prompt_len`; `generated = prompt_ids` (track committed). `HybridState st; st.use_mtp=true;`. Track `cached_len = prompt_len` (KV holds 0..cached_len-1).
2. Loop until terminated or `generated.size() >= prompt_len+max_new`:
   - **MTP round** (`st.use_mtp`): `n_recompute = generated.size() - cached_len` (tokens committed in PRIOR rounds but not yet written to the KV); build the block input = embeds of `generated[cached_len:]` ++ `[last_committed, text_mask×5]`; `n_new = n_recompute + 6`; `kv.past_len = cached_len`; `mtp_block_forward(kv, xblk, n_new, n_recompute, logits6)`; softmax → probs6; `nt = sample_box(probs6)`; if `nt=={0}` → `nt = argmax6(logits6)`; `committed = hybrid_mtp_step(st, nt)`; append `committed` to `generated`; then `cached_len += n_recompute`.
     The KV bookkeeping (mirrors the reference's per-round `kv[...,:generated.shape[1],:]` truncation): each MTP round forwards `n_recompute` previously-committed-but-uncached tokens (recomputed causally → their K/V land at `[cached_len, cached_len+n_recompute)` and are kept) PLUS the 6 block slots (their K/V land at `[cached_len+n_recompute, …)` and are speculative). Advancing `cached_len` by ONLY `n_recompute` means next round sets `kv.past_len = cached_len+n_recompute` and writes there, OVERWRITING the stale block slots — so the block K/V is effectively discarded with no explicit clear. THIS round's freshly-committed box tokens are NOT yet cached; they become next round's `n_recompute`. (Verified against the reference loop order: forward → truncate-to-pre-commit-length → commit.)
   - **AR round** (`!st.use_mtp`): `n_recompute = generated.size() - cached_len`; run a 1-token-style resident step over `generated[cached_len:]` (the uncached committed tokens) — i.e. forward `n_recompute` tokens, take the LAST position's logits, argmax → `ar_token`; `committed = hybrid_ar_step(st, ar_token)`; append; `cached_len += n_recompute` (these get cached); the single new AR token becomes next round's recompute. (Reuse `decode_greedy_resident`'s step machinery — factor a `resident_forward_chunk(generated[cached_len:], cached_len, &last_logits)` helper.)
   - if `st.terminated` break.
3. `out_ids = generated[prompt_len:]` (the generated suffix).
NOTE: the KV/cached_len bookkeeping is the subtle part — mirror the reference's "truncate KV to committed length each round, recompute uncached committed + block". The resident layer writes at `kv.past_len`; setting `kv.past_len=cached_len` before each forward and advancing `cached_len` by the recompute count realizes the truncation (next round overwrites the speculative slots). `argmax6` = per-position argmax of the 6×vocab logits. Verify the masked-forward + AR-forward share the resident path; only the mask/positions differ (causal for AR, block-diffusion for MTP).

- [ ] **Step 2: Gate the full hybrid stream in `tests/test_mtp.cpp`.**
```cpp
    std::vector<int32_t> got;
    if(!lm.decode_hybrid(ids /*prompt*/, proj, 256, got)) return 1;
    std::vector<int32_t> ref; la_parity::load_baseline_i32(mtp, "hybrid_token_ids", ref);
    // ref includes the prompt? No — hybrid_token_ids is the generated stream (258). Compare got to ref.
    int hok = ((int)got.size()==(int)ref.size());
    for(int i=0;i<(int)ref.size() && hok;++i) hok &= (got[i]==ref[i]);
    std::printf("hybrid full-stream match=%d (%zu vs %zu)\n", hok, got.size(), ref.size());
```
(Confirm whether `hybrid_token_ids` in the dump is the generated-only stream or includes the prompt; the M0 `token_stream` was the generated suffix — match that. Adjust the comparison so `got` (generated suffix) aligns with the reference generated stream.) Make the overall test return reflect ALL gates (logits6 && decode && control-flow && hybrid-stream).

- [ ] **Step 3: Build + run + debug.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_mtp -j 2>&1 | tail -8
cd build && ctest -R test_mtp --output-on-failure; cd ..
```
Iterate to PASS. DEBUG (each prior gate isolates a layer): if logits6 (Task 2) + decode (Task 3) + control-flow (Task 4) all pass but the full stream diverges, the bug is in the INTEGRATION — most likely the cached_len/past_len bookkeeping (the KV truncation/recompute) or the per-round n_recompute. Print, per round, `cached_len, n_recompute, committed` vs the reference trace's per-round committed. A divergence at round r with correct logits means the KV state fed to round r was wrong (recompute boundary off). Then run the FULL suite — all green.

- [ ] **Step 4: Commit (M5 complete).**
```bash
cd build && ctest --output-on-failure; cd ..
git add src/lm.hpp src/lm.cpp tests/test_mtp.cpp
git commit -m "M5: full hybrid Parallel Box Decoding -> hybrid stream parity (M5 complete)"
```

---

## Milestone exit criteria

**M5 done when:** `tests/test_mtp.cpp` passes — the block-diffusion masked forward reproduces the reference per-round `logits6` (≤5e-2), the box-frame decode (`decode_bbox_avg`/`decode_ref`/`handle_pattern`) reproduces the reference per-round `new_tokens`/committed tokens exactly on captured logits, the hybrid control-flow state machine reproduces the reference round trace, and the full `decode_hybrid` reproduces the 258-token hybrid reference stream — all green under ctest, full suite green. The C++ now supports both decode modes (M4 slow/AR and M5 hybrid PBD).

## Roadmap (next plan)

- **M6 — Quantize + CLI + flat C-API.** Add quant types (f16/q8_0/q6_k/q5_k/q4_k) to the converter + verify accuracy degrades monotonically (re-gate boxes at each quant); `locate-anything-cli detect --annotated <img>` (preprocess → vision → projector → decode → parse → draw boxes, reusing rt-detr.cpp's renderer patterns); flat C ABI (`la_load`/`la_free`/`la_locate`/`la_free_string`/`la_last_error`/`la_abi_version`). Then optionally the LocalAI backend (separate effort). Also: thread the hardcoded 32×32 grid through from real image preprocessing (the M2 generalization item), and add a proper byte-level BPE detokenizer for box labels (M4 deferred item).
