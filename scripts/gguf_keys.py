"""GGUF key names + tensor-rename table. Imported by the converter AND tests
so the metadata schema lives in exactly one place."""
import re

ARCH = "locateanything"

# --- scalar KV keys ---
KV = {
    # text / LM
    "lm.hidden":        f"{ARCH}.lm.hidden_size",
    "lm.n_layers":      f"{ARCH}.lm.n_layers",
    "lm.n_heads":       f"{ARCH}.lm.n_heads",
    "lm.n_kv_heads":    f"{ARCH}.lm.n_kv_heads",
    "lm.head_dim":      f"{ARCH}.lm.head_dim",
    "lm.intermediate":  f"{ARCH}.lm.intermediate_size",
    "lm.vocab":         f"{ARCH}.lm.vocab_size",
    "lm.rope_theta":    f"{ARCH}.lm.rope_theta",
    "lm.rms_eps":       f"{ARCH}.lm.rms_norm_eps",
    "lm.block_size":    f"{ARCH}.lm.block_size",
    # vision / MoonViT
    "vit.hidden":       f"{ARCH}.vit.hidden_size",
    "vit.n_layers":     f"{ARCH}.vit.n_layers",
    "vit.n_heads":      f"{ARCH}.vit.n_heads",
    "vit.head_dim":     f"{ARCH}.vit.head_dim",
    "vit.intermediate": f"{ARCH}.vit.intermediate_size",
    "vit.patch":        f"{ARCH}.vit.patch_size",
    "vit.merge":        f"{ARCH}.vit.merge_kernel_size",
    "vit.pos_emb_hw":   f"{ARCH}.vit.init_pos_emb_hw",
    "vit.rope_theta":   f"{ARCH}.vit.rope_theta",
    # special tokens
    "tok.image":        f"{ARCH}.token.image",
    "tok.box_start":    f"{ARCH}.token.box_start",
    "tok.box_end":      f"{ARCH}.token.box_end",
    "tok.coord_start":  f"{ARCH}.token.coord_start",
    "tok.coord_end":    f"{ARCH}.token.coord_end",
    "tok.ref_start":    f"{ARCH}.token.ref_start",
    "tok.ref_end":      f"{ARCH}.token.ref_end",
    "tok.none":         f"{ARCH}.token.none",
    "tok.null":         f"{ARCH}.token.null",
    "tok.switch":       f"{ARCH}.token.switch",
    "tok.text_mask":    f"{ARCH}.token.text_mask",
    "tok.eos":          f"{ARCH}.token.eos",
    "tok.bos":          f"{ARCH}.token.bos",
    # image preprocessing
    "img.mean":         f"{ARCH}.image.mean",
    "img.std":          f"{ARCH}.image.std",
    "img.in_token_limit": f"{ARCH}.image.in_token_limit",
}

KV.update({
    "tok.tokens":      f"{ARCH}.tokenizer.tokens",
    "tok.token_types": f"{ARCH}.tokenizer.token_types",
    "tok.merges":      f"{ARCH}.tokenizer.merges",
    "tok.model":       f"{ARCH}.tokenizer.model",
})

# Hardcoded special-token ids from config.json. All 13 ARE present in config.json
# (8 at the top level, 5 nested in text_config) and are asserted against it at
# convert time via convert_locateanything_to_gguf.assert_special_tokens().
SPECIAL_TOKENS = {
    "tok.image": 151665, "tok.box_start": 151668, "tok.box_end": 151669,
    "tok.coord_start": 151677, "tok.coord_end": 152677,
    "tok.ref_start": 151672, "tok.ref_end": 151673,
    "tok.none": 4064, "tok.null": 152678, "tok.switch": 152679,
    "tok.text_mask": 151676, "tok.eos": 151645, "tok.bos": 151643,
}

def rename_tensor(name: str):
    """safetensors name -> gguf tensor name. Returns None to skip."""
    # LM (Qwen2) -> llama.cpp-style lm.blk.N.*
    m = re.match(r"^language_model\.model\.embed_tokens\.weight$", name)
    if m: return "lm.tok_embd.weight"
    m = re.match(r"^language_model\.model\.layers\.(\d+)\.self_attn\.([qkv])_proj\.(weight|bias)$", name)
    if m: return f"lm.blk.{m.group(1)}.attn_{m.group(2)}.{m.group(3)}"
    m = re.match(r"^language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", name)
    if m: return f"lm.blk.{m.group(1)}.attn_o.weight"
    m = re.match(r"^language_model\.model\.layers\.(\d+)\.input_layernorm\.weight$", name)
    if m: return f"lm.blk.{m.group(1)}.attn_norm.weight"
    m = re.match(r"^language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight$", name)
    if m: return f"lm.blk.{m.group(1)}.ffn_norm.weight"
    m = re.match(r"^language_model\.model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$", name)
    if m: return f"lm.blk.{m.group(1)}.ffn_{m.group(2)}.weight"
    if name == "language_model.model.norm.weight": return "lm.output_norm.weight"
    if name == "language_model.lm_head.weight":     return "lm.output.weight"
    # MLP projector
    m = re.match(r"^mlp1\.(\d+)\.(weight|bias)$", name)
    if m: return f"proj.{m.group(1)}.{m.group(2)}"
    # Vision (MoonViT)
    if name == "vision_model.patch_embed.proj.weight":  return "vit.patch_embed.weight"
    if name == "vision_model.patch_embed.proj.bias":    return "vit.patch_embed.bias"
    if name == "vision_model.patch_embed.pos_emb.weight": return "vit.pos_emb.weight"
    m = re.match(r"^vision_model\.encoder\.blocks\.(\d+)\.(norm0|norm1)\.(weight|bias)$", name)
    if m: return f"vit.blk.{m.group(1)}.{m.group(2)}.{m.group(3)}"
    m = re.match(r"^vision_model\.encoder\.blocks\.(\d+)\.(wqkv|wo)\.(weight|bias)$", name)
    if m: return f"vit.blk.{m.group(1)}.{m.group(2)}.{m.group(3)}"
    m = re.match(r"^vision_model\.encoder\.blocks\.(\d+)\.mlp\.(fc0|fc1)\.(weight|bias)$", name)
    if m: return f"vit.blk.{m.group(1)}.{m.group(2)}.{m.group(3)}"
    m = re.match(r"^vision_model\.encoder\.final_layernorm\.(weight|bias)$", name)
    if m: return f"vit.final_norm.{m.group(1)}"
    return None  # unknown -> skip (with a warning in the converter)
