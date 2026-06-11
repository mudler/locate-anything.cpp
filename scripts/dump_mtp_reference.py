#!/usr/bin/env python3
"""M5 Task 1: instrument a hybrid `generate` run and dump per-MTP-round
intermediates that the C++ M5 tasks gate against.

Outputs:
  dumps/reference_mtp.gguf   -- logits + token ids + per-round arrays (no JSON
                                needed by C++ tasks 2/3/4/5)
  dumps/reference_mtp_trace.json -- human-inspectable trace

How the capture works (verified against the remote-code files):
  - modeling_locateanything.py does `from .generate_utils import sample_tokens,
    handle_pattern, get_token_ids_from_config`. The hybrid generate loop calls
    `handle_pattern(new_tokens, ...)` exactly ONCE per MTP round (inside the
    nested `_sample_token_in_mtp`). AR rounds (inside `_sample_token_in_ar`) do
    NOT call handle_pattern.
  - `sample_tokens` (same module) calls `decode_bbox_avg(... keep_k=
    generate_kwargs.get('keep_k_avg', 4) ...)`, so the EFFECTIVE keep_k for the
    box-average decode is 4 (NOT the keep_k=5 passed to sample_tokens). We wrap
    decode_bbox_avg to confirm the actual value on the first call.
  - We wrap `model.language_model.forward` to stash the most recent
    `outputs.logits`. Each language_model forward == exactly one generation
    round. We attribute a round to MTP when handle_pattern fires for it, else AR
    (event ordering: a forward whose stashed logits are never consumed by a
    handle_pattern call before the NEXT forward / end-of-run is an AR round).
  - For an MTP round we grab logits[0, -n_future_tokens:, :] == the 6 block
    positions of the round that just ran (1:1 alignment with the round).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import gguf

import scripts.la_reference as R

ROOT = Path(__file__).resolve().parent.parent
N_FUTURE = 6  # block_size / n_future_tokens
TEXT_MASK_TOKEN_ID = 151676


def main():
    cfg, model, processor, inputs, spec = R.build_inputs()
    tok = processor.tokenizer

    token_ids = model.token_ids
    box_end_id = token_ids['box_end_token_id']
    coord_start_id = token_ids['coord_start_token_id']
    coord_end_id = token_ids['coord_end_token_id']
    none_id = token_ids['none_token_id']

    # ---- module bindings (DO NOT guess: resolve from the loaded objects) ----
    model_mod = sys.modules[model.__class__.__module__]          # modeling_locateanything
    gu_mod = sys.modules[model_mod.sample_tokens.__module__]      # generate_utils

    orig_handle_pattern = model_mod.handle_pattern
    orig_decode_bbox_avg = gu_mod.decode_bbox_avg
    orig_lm_forward = model.language_model.forward

    # ---- capture state ----
    rounds = []            # in-order list of {kind,out_type,committed,new_tokens,logits6}
    pending = {"box": None}    # logits from the most recent forward, not yet attributed
    keep_k_seen = {"val": None}

    def _ar_out_type(token_val):
        # mirror _sample_token_in_ar (hybrid mode) so the human trace is faithful
        if token_val == box_end_id:
            return 'box_end_ar'
        if coord_start_id <= token_val <= coord_end_id or token_val == none_id:
            return 'coord_ar'
        return 'im_end'

    def _flush_pending_as_ar():
        p = pending["box"]
        if p is None:
            return
        pending["box"] = None
        last = p["last_logits"]            # [vocab] f32
        tv = int(np.argmax(last))
        rounds.append({
            "kind": "ar",
            "out_type": _ar_out_type(tv),
            "committed": [tv],
            "new_tokens": [tv],
            "logits6": None,
        })

    def wrapped_lm_forward(*args, **kwargs):
        out = orig_lm_forward(*args, **kwargs)
        logits = out.logits  # [1, seq, vocab]
        # A NEW forward => the previous forward (if still pending) was an AR round.
        _flush_pending_as_ar()
        seq = logits.shape[1]
        nlast = min(N_FUTURE, seq)
        l6 = logits[0, -nlast:, :].detach().to(torch.float32).cpu().numpy()
        last = logits[0, -1, :].detach().to(torch.float32).cpu().numpy()
        pending["box"] = {"logits6": l6, "last_logits": last}
        return out

    def wrapped_handle_pattern(x0, tids, generation_mode='hybrid'):
        r = orig_handle_pattern(x0, tids, generation_mode)
        # This MTP round consumes the most recent forward.
        p = pending["box"]
        pending["box"] = None
        l6 = p["logits6"] if p is not None else None
        rounds.append({
            "kind": "mtp",
            "out_type": r["type"],
            "committed": list(int(t) for t in r["tokens"]),
            "new_tokens": [int(t) for t in (x0.tolist() if hasattr(x0, 'tolist') else x0)],
            "logits6": l6,
        })
        return r

    def wrapped_decode_bbox_avg(logits, probs, tids, keep_k=5, *a, **k):
        if keep_k_seen["val"] is None:
            keep_k_seen["val"] = keep_k
            print(f"[decode_bbox_avg] first-call keep_k = {keep_k}")
        return orig_decode_bbox_avg(logits, probs, tids, keep_k=keep_k, *a, **k)

    # ---- install hooks ----
    model.language_model.forward = wrapped_lm_forward
    model_mod.handle_pattern = wrapped_handle_pattern
    gu_mod.decode_bbox_avg = wrapped_decode_bbox_avg

    try:
        with torch.no_grad():
            s = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs["image_grid_hws"],
                tokenizer=tok,
                max_new_tokens=256,
                generation_mode="hybrid",
                use_cache=True,
            )
        # The final round may be an AR round whose forward is still pending.
        _flush_pending_as_ar()
    finally:
        model.language_model.forward = orig_lm_forward
        model_mod.handle_pattern = orig_handle_pattern
        gu_mod.decode_bbox_avg = orig_decode_bbox_avg

    keep_k = keep_k_seen["val"]
    assert keep_k is not None, "decode_bbox_avg was never called (no box frames?)"

    # ---- re-encode the returned string with an exact round-trip ----
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids, skip_special_tokens=False) == s, "id round-trip mismatch"
    hybrid_token_ids = np.array(ids, dtype=np.int32)
    n_tokens = int(hybrid_token_ids.shape[0])

    # ---- SANITY GUARD: concat of all rounds' committed == hybrid_token_ids ----
    committed_concat = []
    for r in rounds:
        committed_concat.extend(r["committed"])
    committed_concat = np.array(committed_concat, dtype=np.int32)
    if not (committed_concat.shape == hybrid_token_ids.shape and
            np.array_equal(committed_concat, hybrid_token_ids)):
        # dump a diff to help debugging before failing
        m = min(len(committed_concat), len(hybrid_token_ids))
        first_bad = next((i for i in range(m)
                          if committed_concat[i] != hybrid_token_ids[i]), m)
        raise AssertionError(
            "ROUND CAPTURE MISALIGNED: concatenated committed tokens != "
            f"hybrid_token_ids. len(committed)={len(committed_concat)} "
            f"len(stream)={n_tokens} first_divergence={first_bad}\n"
            f"  committed[{first_bad}:{first_bad+8}]={committed_concat[first_bad:first_bad+8].tolist()}\n"
            f"  stream   [{first_bad}:{first_bad+8}]={hybrid_token_ids[first_bad:first_bad+8].tolist()}"
        )

    # ---- assemble per-round arrays ----
    mtp_rounds = [r for r in rounds if r["kind"] == "mtp"]
    ar_rounds = [r for r in rounds if r["kind"] == "ar"]
    n_mtp_rounds = len(mtp_rounds)
    n_ar_rounds = len(ar_rounds)
    vocab = int(mtp_rounds[0]["logits6"].shape[-1])

    # logits stack [n_mtp_rounds, 6, vocab]; pad short rounds defensively (MTP
    # rounds always have >=6 positions in practice, so no padding expected).
    l6_list = []
    for r in mtp_rounds:
        a = r["logits6"]
        if a.shape[0] != N_FUTURE:
            pad = np.zeros((N_FUTURE - a.shape[0], a.shape[1]), dtype=np.float32)
            a = np.concatenate([pad, a], axis=0)
        l6_list.append(a.astype(np.float32, copy=False))
    mtp_logits6 = np.stack(l6_list, axis=0)  # [n_mtp_rounds, 6, vocab]

    mtp_new_tokens, mtp_new_lens = [], []
    mtp_committed, mtp_committed_lens = [], []
    for r in mtp_rounds:
        mtp_new_tokens.extend(r["new_tokens"]); mtp_new_lens.append(len(r["new_tokens"]))
        mtp_committed.extend(r["committed"]); mtp_committed_lens.append(len(r["committed"]))
    mtp_new_tokens = np.array(mtp_new_tokens, dtype=np.int32)
    mtp_new_lens = np.array(mtp_new_lens, dtype=np.int32)
    mtp_committed = np.array(mtp_committed, dtype=np.int32)
    mtp_committed_lens = np.array(mtp_committed_lens, dtype=np.int32)

    out_types = [r["out_type"] for r in rounds]
    round_kinds = np.array([0 if r["kind"] == "mtp" else 1 for r in rounds], dtype=np.int32)
    ar_tokens = np.array([r["committed"][0] for r in ar_rounds], dtype=np.int32)

    # gguf rejects zero-length tensors; use a [1] sentinel guarded by the
    # la_mtp.n_ar_rounds KV when there are no AR rounds.
    ar_tokens_t = ar_tokens if n_ar_rounds > 0 else np.array([-1], dtype=np.int32)

    # ---- write gguf ----
    out_gguf = ROOT / "dumps" / "reference_mtp.gguf"
    w = gguf.GGUFWriter(str(out_gguf), "la_mtp")
    w.add_tensor("mtp_logits6", np.ascontiguousarray(mtp_logits6, dtype=np.float32))
    w.add_tensor("hybrid_token_ids", np.ascontiguousarray(hybrid_token_ids, dtype=np.int32))
    w.add_tensor("mtp_new_tokens", np.ascontiguousarray(mtp_new_tokens))
    w.add_tensor("mtp_new_lens", np.ascontiguousarray(mtp_new_lens))
    w.add_tensor("mtp_committed", np.ascontiguousarray(mtp_committed))
    w.add_tensor("mtp_committed_lens", np.ascontiguousarray(mtp_committed_lens))
    w.add_tensor("mtp_round_kinds", np.ascontiguousarray(round_kinds))
    w.add_tensor("mtp_ar_tokens", np.ascontiguousarray(ar_tokens_t))
    w.add_array("la_mtp.out_types", out_types)
    w.add_uint32("la_mtp.n_tokens", n_tokens)
    w.add_uint32("la_mtp.n_rounds", len(rounds))
    w.add_uint32("la_mtp.n_mtp_rounds", n_mtp_rounds)
    w.add_uint32("la_mtp.n_ar_rounds", n_ar_rounds)
    w.add_uint32("la_mtp.keep_k", int(keep_k))
    w.add_uint32("la_mtp.vocab", vocab)
    w.add_uint32("la_mtp.block_size", N_FUTURE)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    # ---- write json trace ----
    json_rounds = []
    for i, r in enumerate(rounds):
        json_rounds.append({
            "idx": i,
            "kind": r["kind"],
            "out_type": r["out_type"],
            "committed": r["committed"],
            "new_tokens": r["new_tokens"],
        })
    trace = {
        "rounds": json_rounds,
        "n_tokens": n_tokens,
        "n_rounds": len(rounds),
        "n_mtp_rounds": n_mtp_rounds,
        "n_ar_rounds": n_ar_rounds,
        "keep_k": int(keep_k),
        "vocab": vocab,
        "block_size": N_FUTURE,
        "first16": hybrid_token_ids[:16].tolist(),
        "string": s,
    }
    out_json = ROOT / "dumps" / "reference_mtp_trace.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

    # ---- report ----
    print(f"wrote {out_gguf}")
    print(f"wrote {out_json}")
    print(f"n_tokens        = {n_tokens}")
    print(f"n_rounds        = {len(rounds)}")
    print(f"n_mtp_rounds    = {n_mtp_rounds}")
    print(f"n_ar_rounds     = {n_ar_rounds}  (entered AR fallback: {n_ar_rounds > 0})")
    print(f"keep_k (decode_bbox_avg, CONFIRMED) = {keep_k}")
    print(f"mtp_logits6 shape = {list(mtp_logits6.shape)}")
    print(f"first ~5 out_types = {out_types[:5]}")
    print(f"sum(committed_lens) MTP = {int(mtp_committed_lens.sum())}, "
          f"AR committed = {n_ar_rounds}, total = {int(mtp_committed_lens.sum()) + n_ar_rounds} "
          f"(== n_tokens: {int(mtp_committed_lens.sum()) + n_ar_rounds == n_tokens})")
    print(f"SANITY: concat(committed) == hybrid_token_ids -> OK ({len(committed_concat)} tokens)")

    # compare vs M0 manifest hybrid string if available
    man = ROOT / "dumps" / "manifest.json"
    if man.exists():
        with open(man, encoding="utf-8") as f:
            mj = json.load(f)
        gs = mj.get("generated_string")
        if gs is not None:
            print(f"matches manifest generated_string: {gs == s}")
            if gs != s:
                print("  WARNING: hybrid string differs from M0 manifest")
    print(f"string[:120] = {s[:120]}")


if __name__ == "__main__":
    main()
