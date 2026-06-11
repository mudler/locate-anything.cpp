#!/usr/bin/env python3
"""Convert nvidia/LocateAnything-3B safetensors -> one self-contained GGUF."""
import argparse, json, sys, os
from pathlib import Path
# Make the repo root importable so `import scripts.gguf_keys` works whether this
# file is run as a module (python -m / from repo root) OR invoked directly
# (python scripts/convert_locateanything_to_gguf.py). Keep scripts/ a namespace
# package (no __init__.py) so the dump scripts that rely on that keep working.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gguf
from safetensors import safe_open
import scripts.gguf_keys as K

def load_config(model_dir):
    with open(Path(model_dir) / "config.json", encoding="utf-8") as f:
        return json.load(f)

def load_preprocessor_config(model_dir):
    with open(Path(model_dir) / "preprocessor_config.json", encoding="utf-8") as f:
        return json.load(f)

# Where each SPECIAL_TOKENS id lives in config.json: 8 at the top level, 5 nested
# in text_config. All 13 are present in config.json, so all 13 are asserted.
_SPECIAL_TOKEN_SOURCES = {
    "tok.image":       ("top",  "image_token_index"),
    "tok.box_start":   ("top",  "box_start_token_id"),
    "tok.box_end":     ("top",  "box_end_token_id"),
    "tok.coord_start": ("top",  "coord_start_token_id"),
    "tok.coord_end":   ("top",  "coord_end_token_id"),
    "tok.ref_start":   ("top",  "ref_start_token_id"),
    "tok.ref_end":     ("top",  "ref_end_token_id"),
    "tok.none":        ("top",  "none_token_id"),
    "tok.null":        ("text", "null_token_id"),
    "tok.switch":      ("text", "switch_token_id"),
    "tok.text_mask":   ("text", "text_mask_token_id"),
    "tok.eos":         ("text", "eos_token_id"),
    "tok.bos":         ("text", "bos_token_id"),
}

def assert_special_tokens(cfg):
    """Assert the hardcoded SPECIAL_TOKENS ids match config.json. Raises a clear
    error on any mismatch. All 13 ids are present in config.json (top level or
    text_config), so none are unverifiable here."""
    t = cfg["text_config"]
    for key, expected in K.SPECIAL_TOKENS.items():
        scope, cfg_key = _SPECIAL_TOKEN_SOURCES[key]
        src = cfg if scope == "top" else t
        if cfg_key not in src:
            raise AssertionError(
                f"{key}: config key '{cfg_key}' missing from "
                f"config.json{'' if scope == 'top' else '.text_config'}")
        actual = src[cfg_key]
        if actual != expected:
            raise AssertionError(
                f"special token mismatch for {key}: SPECIAL_TOKENS has {expected} "
                f"but config.json{'' if scope == 'top' else '.text_config'}"
                f"['{cfg_key}'] is {actual}")

def write_hparams(w, cfg, prep):
    assert_special_tokens(cfg)
    t = cfg["text_config"]; v = cfg["vision_config"]
    w.add_uint32(K.KV["lm.hidden"],       t["hidden_size"])
    w.add_uint32(K.KV["lm.n_layers"],     t["num_hidden_layers"])
    w.add_uint32(K.KV["lm.n_heads"],      t["num_attention_heads"])
    w.add_uint32(K.KV["lm.n_kv_heads"],   t["num_key_value_heads"])
    w.add_uint32(K.KV["lm.head_dim"],     t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]))
    w.add_uint32(K.KV["lm.intermediate"], t["intermediate_size"])
    w.add_uint32(K.KV["lm.vocab"],        t["vocab_size"])
    w.add_float32(K.KV["lm.rope_theta"],  float(t["rope_theta"]))
    w.add_float32(K.KV["lm.rms_eps"],     float(t["rms_norm_eps"]))
    w.add_uint32(K.KV["lm.block_size"],   int(t["block_size"]))
    w.add_uint32(K.KV["vit.hidden"],      v["hidden_size"])
    w.add_uint32(K.KV["vit.n_layers"],    v["num_hidden_layers"])
    w.add_uint32(K.KV["vit.n_heads"],     v["num_attention_heads"])
    w.add_uint32(K.KV["vit.head_dim"],    v.get("head_dim", v["hidden_size"] // v["num_attention_heads"]))
    w.add_uint32(K.KV["vit.intermediate"], v["intermediate_size"])
    w.add_uint32(K.KV["vit.patch"],       v["patch_size"])
    w.add_array(K.KV["vit.merge"],        [int(x) for x in v["merge_kernel_size"]])
    w.add_uint32(K.KV["vit.pos_emb_hw"],  v.get("init_pos_emb_height", 64))
    w.add_float32(K.KV["vit.rope_theta"], 10000.0)
    for key, tid in K.SPECIAL_TOKENS.items():
        w.add_uint32(K.KV[key], tid)
    # Image preprocessing comes from preprocessor_config.json (keys: image_mean,
    # image_std, in_token_limit), NOT config.json. Defaults match the model card.
    w.add_array(K.KV["img.mean"], [float(x) for x in prep.get("image_mean", [0.5, 0.5, 0.5])])
    w.add_array(K.KV["img.std"],  [float(x) for x in prep.get("image_std",  [0.5, 0.5, 0.5])])
    w.add_uint32(K.KV["img.in_token_limit"], int(prep.get("in_token_limit", 25600)))

def write_tokenizer(w, model_dir):
    """Embed the Qwen2 BPE vocab + merges + added tokens (gpt2 byte-level BPE)."""
    md = Path(model_dir)
    with open(md / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)               # token -> id
    added = {}
    if (md / "added_tokens.json").exists():
        with open(md / "added_tokens.json", encoding="utf-8") as f:
            added = json.load(f)           # token -> id
    merged = dict(vocab); merged.update(added)
    id_to_tok = {i: t for t, i in merged.items()}
    n = max(id_to_tok) + 1
    tokens = [id_to_tok.get(i, f"<unused_{i}>") for i in range(n)]
    added_ids = set(added.values())
    types = [4 if i in added_ids else 1 for i in range(n)]  # 1=normal, 4=control
    merges = []
    with open(md / "merges.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0 and line.startswith("#version"):
                continue
            if line:
                merges.append(line)
    w.add_string(K.KV["tok.model"], "gpt2")
    w.add_array(K.KV["tok.tokens"], tokens)
    w.add_array(K.KV["tok.token_types"], types)
    w.add_array(K.KV["tok.merges"], merges)
    return n

def iter_safetensors(model_dir):
    """Yield (name, f32 numpy array) for every tensor across all shards.

    The weights are bf16 on disk, which numpy has no native dtype for, so we
    open with framework="pt" and do `.float().numpy()`. bf16 -> f32 is exact
    (f32 has a strict superset of bf16's exponent+mantissa range), so this is
    lossless."""
    import torch  # local import: only needed for the pt read path
    md = Path(model_dir)
    with open(md / "model.safetensors.index.json", encoding="utf-8") as f:
        idx = json.load(f)
    shards = sorted(set(idx["weight_map"].values()))
    for shard in shards:
        with safe_open(md / shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                yield name, t.float().numpy()

def write_weights(w, model_dir):
    written, skipped = 0, []
    for name, arr in iter_safetensors(model_dir):
        gname = K.rename_tensor(name)
        if gname is None:
            skipped.append(name); continue
        # M1: store raw f32 (no transpose). PyTorch [out, in] layout is exactly
        # what ggml_mul_mat expects (ne[0]=in). Quantization is M6.
        a = np.ascontiguousarray(arr, dtype=np.float32)
        w.add_tensor(gname, a)
        written += 1
    return written, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/LocateAnything-3B")
    ap.add_argument("--output", default="models/locate-anything-f32.gguf")
    args = ap.parse_args()
    cfg = load_config(args.model)
    prep = load_preprocessor_config(args.model)
    w = gguf.GGUFWriter(args.output, K.ARCH)
    write_hparams(w, cfg, prep)   # also asserts special tokens vs config.json
    n_tok = write_tokenizer(w, args.model)
    written, skipped = write_weights(w, args.model)
    # Refuse to emit an incomplete GGUF: every safetensors tensor must map to a
    # GGUF name. A silent partial file would fail far downstream in the C++
    # loader with a confusing missing-tensor error. Fail loudly here instead.
    if skipped:
        raise SystemExit(
            f"error: {len(skipped)} tensor(s) did not map to a GGUF name "
            f"(rename_tensor returned None); refusing to write an incomplete GGUF:\n  "
            + "\n  ".join(skipped[:20]))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.output}: tensors={written} skipped=0 vocab={n_tok}")

if __name__ == "__main__":
    main()
