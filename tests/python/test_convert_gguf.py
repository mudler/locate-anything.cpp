import json
from pathlib import Path
import gguf
import scripts.gguf_keys as K

ROOT = Path(__file__).resolve().parents[2]
GGUF = ROOT / "models" / "locate-anything-f32.gguf"

def _val(r, key):
    f = r.get_field(key)
    assert f is not None, f"missing KV: {key}"
    return f.parts[f.data[-1]][0] if f.data else f.parts[-1][0]

def test_gguf_has_all_hparams():
    assert GGUF.exists(), "run scripts/convert_locateanything_to_gguf.py first"
    r = gguf.GGUFReader(GGUF)
    assert int(_val(r, K.KV["lm.hidden"]))    == 2048
    assert int(_val(r, K.KV["lm.n_layers"]))  == 36
    assert int(_val(r, K.KV["lm.n_kv_heads"])) == 2
    assert int(_val(r, K.KV["lm.head_dim"]))  == 128
    assert int(_val(r, K.KV["lm.vocab"]))     == 152681
    assert int(_val(r, K.KV["vit.n_layers"])) == 27
    assert int(_val(r, K.KV["vit.hidden"]))   == 1152
    assert int(_val(r, K.KV["vit.head_dim"])) == 72

def test_gguf_special_tokens():
    r = gguf.GGUFReader(GGUF)
    assert int(_val(r, K.KV["tok.image"]))       == 151665
    assert int(_val(r, K.KV["tok.coord_start"])) == 151677
    assert int(_val(r, K.KV["tok.coord_end"]))   == 152677
    assert int(_val(r, K.KV["tok.box_start"]))   == 151668
    assert int(_val(r, K.KV["tok.box_end"]))     == 151669

def test_gguf_tensor_names_and_count():
    r = gguf.GGUFReader(GGUF)
    names = {t.name for t in r.tensors}
    assert len(names) == 770
    for expect in ["lm.tok_embd.weight", "lm.blk.0.attn_q.weight",
                   "lm.blk.0.attn_q.bias", "lm.blk.35.ffn_down.weight",
                   "lm.output_norm.weight", "proj.0.weight",
                   "vit.patch_embed.weight", "vit.blk.26.wo.weight",
                   "vit.pos_emb.weight", "vit.final_norm.weight"]:
        assert expect in names, f"missing tensor: {expect}"


def test_generated_header_matches_committed():
    """The committed include/la_gguf_keys.h must equal a fresh generation, so
    the C++ loader never compiles against KV strings that drifted from
    scripts/gguf_keys.py. Re-run scripts/gen_gguf_keys_header.py if this fails."""
    import scripts.gen_gguf_keys_header as G
    committed = G.HEADER_PATH.read_text(encoding="utf-8")
    assert committed == G.render(), \
        "include/la_gguf_keys.h is stale; re-run scripts/gen_gguf_keys_header.py"
