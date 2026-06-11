#!/usr/bin/env python3
r"""Dump the GOLD parity reference tensors for the locate-anything.cpp port.

This runs the REAL HF LocateAnything-3B model on the fixed parity fixture (built
by ``scripts.la_reference.build_inputs``) and writes per-component intermediate
tensors to ``dumps/reference.npz`` (all float32) plus ``dumps/manifest.json``.
Every later C++ milestone (M2 vision, M3 LM forward, M4 decode) gates against
these tensors, so they must be the REAL captured activations -- there are NO
zero/placeholder stubs here.

Why this does NOT just call ``model(**inputs)`` / the HF ``generate`` signature
(both established broken / non-standard on CPU in the vendored code):
  * ``LocateAnythingForConditionalGeneration.forward`` feeds the LLM with
    ``inputs_embeds=`` only and drops ``input_ids``; the magi->sdpa block-mask
    build then dereferences a ``None`` ``input_ids``. So we never call the
    top-level forward to get logits.
  * The custom ``generate`` REQUIRES ``tokenizer=`` and RETURNS A DECODED STRING
    (special tokens kept), not a tensor. We use it for the token stream and
    recover ids by re-encoding the returned string.

Faithful prefill (mirrors what ``generate`` does internally for its 1st forward):
  1. ``vit_embeds = model.extract_feature(pixel_values, image_grid_hws)`` -- runs
     the MoonViT tower (patch_embed -> encoder blocks -> final_layernorm ->
     patch_merger), returning a list of merged [256, 4608] token tensors.
  2. ``merged = torch.cat(vit_embeds, 0)`` ; ``projected = model.mlp1(merged)``.
  3. ``model.language_model(input_ids=..., visual_features=projected,
     image_token_index=151665, attention_mask=..., use_cache=True)``. The Qwen2
     model splices ``projected`` into the ``<IMG_CONTEXT>`` (151665) positions
     (``Qwen2Model.image_processing``) and runs the 36 decoder layers. Since the
     prompt's last token is not the text-mask token, the block-diffusion mask
     path falls back to a plain causal mask -- the clean step-0 prefill.

Capture points (module attribute paths verified against the vendored code; the
M2/M3 implementers rely on this mapping):
  pixel_values        : inputs['pixel_values']            (processor output)
  vit_pos_added       : model.vision_model.patch_embed              [fwd hook out]
  vit_layer_00        : model.vision_model.encoder.blocks[0]        [fwd hook out]
  vit_layer_26        : model.vision_model.encoder.blocks[26]       [fwd hook out]
  vit_final           : model.vision_model.encoder.final_layernorm  [fwd hook out]
  merged_tokens       : model.vision_model                 [fwd hook out, list -> cat]
  projected_tokens    : model.mlp1                                  [fwd hook out]
  embeds_after_splice : model.language_model.model.layers[0]   [fwd PRE hook in[0]]
  lm_layer_00         : model.language_model.model.layers[0]       [fwd hook out[0]]
  lm_layer_35         : model.language_model.model.layers[35]      [fwd hook out[0]]
  logits_step0        : language_model prefill outputs.logits[0, -1]
  token_stream        : ids recovered from model.generate(...) decoded string
"""
import json
from pathlib import Path

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.la_reference import build_inputs, IMAGE_CONTEXT_TOKEN_ID

DUMP_DIR = REPO_ROOT / "dumps"
NPZ_PATH = DUMP_DIR / "reference.npz"
MANIFEST_PATH = DUMP_DIR / "manifest.json"


def _np(t):
    """Detach a tensor to a contiguous float32 numpy array."""
    return t.detach().to(torch.float32).cpu().contiguous().numpy()


def main():
    torch.manual_seed(0)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    cfg, model, processor, inputs, spec = build_inputs()
    model.eval()

    tokenizer = processor.tokenizer
    captured = {}
    handles = []

    def save_out(name, transform=lambda o: o):
        def hook(_module, _inp, out):
            captured[name] = transform(out)
        return hook

    vm = model.vision_model
    enc = vm.encoder
    lm_layers = model.language_model.model.layers

    # --- vision hooks ---
    handles.append(vm.patch_embed.register_forward_hook(save_out("vit_pos_added")))
    handles.append(enc.blocks[0].register_forward_hook(save_out("vit_layer_00")))
    handles.append(enc.blocks[26].register_forward_hook(save_out("vit_layer_26")))
    handles.append(
        enc.final_layernorm.register_forward_hook(save_out("vit_final"))
    )
    # vision_model returns the patch_merger output: a list of [Nmerged, 4608]
    handles.append(
        vm.register_forward_hook(
            save_out("merged_tokens", lambda o: torch.cat(list(o), dim=0))
        )
    )
    # projector
    handles.append(model.mlp1.register_forward_hook(save_out("projected_tokens")))

    # --- LM hooks ---
    # embeds_after_splice == the hidden_states fed into layer 0 (image_processing
    # has already spliced the vision tokens into the <IMG_CONTEXT> positions).
    # We also grab the ACTUAL 4D attention mask layer 0 receives so we can assert
    # below that the prompt prefill is genuinely plain-causal (see the guard after
    # the prefill forward).
    mask_capture = {}

    def pre_hook_layer0(_module, args, kwargs):
        captured["embeds_after_splice"] = args[0]
        mask_capture["attn_mask"] = kwargs.get("attention_mask")

    handles.append(
        lm_layers[0].register_forward_pre_hook(pre_hook_layer0, with_kwargs=True)
    )
    handles.append(
        lm_layers[0].register_forward_hook(
            save_out("lm_layer_00", lambda o: o[0])
        )
    )
    handles.append(
        lm_layers[35].register_forward_hook(
            save_out("lm_layer_35", lambda o: o[0])
        )
    )

    # =================== Faithful prefill ===================
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"].to(model.language_model.dtype)
    image_grid_hws = inputs["image_grid_hws"]

    with torch.no_grad():
        # (1) vision tower -> list of merged [256, 4608] tokens
        vit_embeds = model.extract_feature(pixel_values, image_grid_hws)
        merged = torch.cat(vit_embeds, dim=0)
        # (2) projector -> [256, 2048] (triggers the mlp1 hook)
        projected = model.mlp1(merged)
        # (3) LM prefill with the vision tokens spliced into <IMG_CONTEXT>
        prefill = model.language_model(
            input_ids=input_ids,
            visual_features=projected,
            image_token_index=model.config.image_token_index,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

    logits = prefill.logits  # [1, seq, vocab]
    captured["logits_step0"] = logits[0, -1, :]

    # =================== Guard: prompt prefill MUST be plain causal ===========
    # logits_step0 / lm_layer_* are the step-0 prefill activations the C++ port
    # (M3) reproduces with a STANDARD causal prefill. The HF model uses magi /
    # block-diffusion masking; for the PROMPT (no MTP mask tokens present yet) the
    # mask must collapse to plain causal over the full sequence. In
    # modeling_qwen2.Qwen2Model.forward the (CPU) sdpa branch (L1331) calls
    # _prepare_block_mask_for_inference (L1270): it builds a standard 4D causal
    # mask via _prepare_4d_causal_attention_mask and, because the prompt's last
    # token != text_mask_token_id (151676), takes the AR-mode early return (L1279)
    # -- so NO block-diffusion / x0_len (L1268) prefix masking is applied.
    # Rather than trust that branch, we inspect the ACTUAL dense additive mask fed
    # to decoder layer 0 and assert it is exactly lower-triangular causal. If a
    # future config/code change (e.g. a trailing text_mask_token_id, the magi
    # ranges path, or x0_len < seq_length block-masking) made the prefill
    # non-causal, this raises LOUDLY so the LM gold can't be silently corrupted.
    # M3: the branch that must stay causal is modeling_qwen2.py L1270-1310.
    attn_mask = mask_capture.get("attn_mask")
    seq_len = input_ids.shape[1]
    assert isinstance(attn_mask, torch.Tensor), (
        f"prefill attention_mask is not a dense tensor (got {type(attn_mask)!r}); "
        "the masking regime is NOT the plain-causal sdpa path M3 reproduces -- "
        "LM gold semantics changed. See modeling_qwen2.py _prepare_block_mask_for_inference."
    )
    assert attn_mask.shape == (1, 1, seq_len, seq_len), (
        f"unexpected prefill mask shape {tuple(attn_mask.shape)}, "
        f"expected (1, 1, {seq_len}, {seq_len})"
    )
    m = attn_mask[0, 0].to(torch.float32)
    # additive mask: ~0.0 => attention allowed, finfo.min (hugely negative) => masked.
    allowed = m > -1.0
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    assert torch.equal(allowed, causal), (
        "PROMPT prefill mask is NOT plain causal -- logits_step0/lm_layer_* would be "
        f"block-masked activations (WRONG gold). {int((allowed != causal).sum())} "
        "positions diverge from lower-triangular causal. See modeling_qwen2.py "
        "_prepare_block_mask_for_inference (~L1270) and the AR-mode early return (~L1279)."
    )
    assert torch.all(m[causal] == 0.0), "causal-allowed positions carry a nonzero bias"
    assert torch.all(m[~causal] < -1e4), "masked positions are not strongly negative"

    for h in handles:
        h.remove()

    # =================== Token stream via custom generate ===================
    # The custom generate REQUIRES tokenizer= and RETURNS A DECODED STRING.
    with torch.no_grad():
        gen_string = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs["image_grid_hws"],
            tokenizer=tokenizer,
            max_new_tokens=spec["max_new_tokens"],
            generation_mode=spec["generation_mode"],
            do_sample=False,
            use_cache=True,
        )

    # Recover ids by re-encoding (each <N>/<box>/</box>/<ref> coord/special token
    # is a single added-vocab token, so the round-trip is exact).
    token_ids = tokenizer(gen_string, add_special_tokens=False)["input_ids"]
    roundtrip = tokenizer.decode(token_ids, skip_special_tokens=False)
    assert roundtrip == gen_string, (
        "token id round-trip is NOT exact:\n"
        f"  orig: {gen_string!r}\n  back: {roundtrip!r}"
    )
    box_start = 151668
    assert box_start in token_ids, "no <box> (151668) token in recovered stream"
    assert any(151677 <= t <= 152677 for t in token_ids), (
        "no coord token (151677..152677) in recovered stream"
    )
    token_stream = np.asarray(token_ids, dtype=np.int64)
    captured["token_stream"] = token_stream

    # =================== Assemble dump ===================
    captured["pixel_values"] = inputs["pixel_values"]
    # The 297 prompt input_ids (int64, like token_stream; the gguf converter
    # downcasts to int32). Needed by M3 to embed text tokens, locate the 256
    # <IMG_CONTEXT> (151665) positions for the vision splice, and run prefill.
    captured["input_ids"] = inputs["input_ids"][0].cpu().numpy().astype(np.int64)

    npz = {}
    for k, v in captured.items():
        if k in ("token_stream", "input_ids"):
            npz[k] = v
        else:
            npz[k] = _np(v)

    # Guard: no placeholder / all-zero / non-finite tensors slipped through.
    for k, a in npz.items():
        if k in ("token_stream", "input_ids"):
            continue
        assert a.size > 1, f"{k} looks like a stub (size {a.size})"
        assert np.isfinite(a).all(), f"{k} has non-finite values"
        assert np.abs(a).sum() > 0, f"{k} is all zeros (placeholder?)"

    np.savez(NPZ_PATH, **npz)

    shapes = {k: list(npz[k].shape) for k in npz}
    manifest = {
        "image_token_index": IMAGE_CONTEXT_TOKEN_ID,  # 151665
        "coord_start_token_id": 151677,
        "coord_end_token_id": 152677,
        "box_start_token_id": box_start,  # 151668
        "box_end_token_id": 151669,
        "ref_start_token_id": 151672,
        "ref_end_token_id": 151673,
        "block_size": cfg.text_config.block_size,  # 6
        "merged_vision_tokens": int((input_ids == IMAGE_CONTEXT_TOKEN_ID).sum().item()),
        "vocab_size": cfg.text_config.vocab_size,
        "num_lm_layers": cfg.text_config.num_hidden_layers,
        "num_vit_layers": cfg.vision_config.num_hidden_layers,
        "generation_mode": spec["generation_mode"],
        "max_new_tokens": spec["max_new_tokens"],
        "generated_string": gen_string,
        "shapes": shapes,
        "token_stream": token_ids,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Wrote", NPZ_PATH)
    print("Wrote", MANIFEST_PATH)
    print("\nShapes:")
    for k in shapes:
        print(f"  {k:22s} {tuple(shapes[k])}  {npz[k].dtype}")
    print("\nmerged_vision_tokens:", manifest["merged_vision_tokens"])
    print("generated string:\n", gen_string)
    print("\ntoken_stream len:", len(token_ids))
    print("box_start present:", box_start in token_ids)
    print("coord token present:", any(151677 <= t <= 152677 for t in token_ids))


if __name__ == "__main__":
    main()
