import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dumps" / "reference.npz"
MANIFEST = ROOT / "dumps" / "manifest.json"

REQUIRED_KEYS = [
    "pixel_values", "vit_pos_added",
    "vit_layer_00", "vit_layer_26", "vit_final",
    "merged_tokens", "projected_tokens",
    "embeds_after_splice", "lm_layer_00", "lm_layer_35",
    "logits_step0", "token_stream",
]

def test_dump_exists_with_required_keys():
    assert DUMP.exists(), "run: python scripts/dump_reference.py first"
    with np.load(DUMP) as z:
        for k in REQUIRED_KEYS:
            assert k in z.files, f"missing dump tensor: {k}"

def test_no_placeholder_zeros():
    # every captured tensor must be a real, non-trivial tensor (not a 1-elem stub, not all-zeros)
    with np.load(DUMP) as z:
        for k in REQUIRED_KEYS:
            if k == "token_stream":
                continue
            a = z[k]
            assert a.size > 1, f"{k} looks like a stub (size {a.size})"
            assert np.isfinite(a).all(), f"{k} has non-finite values"
            assert np.abs(a).sum() > 0, f"{k} is all zeros (placeholder?)"

def test_manifest_records_shapes_and_special_ids():
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    assert man["image_token_index"] == 151665
    assert man["coord_start_token_id"] == 151677
    assert man["coord_end_token_id"] == 152677
    assert man["block_size"] == 6
    assert man["merged_vision_tokens"] == 256

def test_token_stream_contains_box_and_coord():
    with np.load(DUMP) as z:
        ts = z["token_stream"].tolist()
    assert 151668 in ts, "no <box> token in generated stream"
    assert any(151677 <= t <= 152677 for t in ts), "no coord token in generated stream"

def test_dump_shapes_and_cross_tie():
    # Exact shapes + a cross-tie that catch wrong capture points the nonzero/finite
    # guard cannot (e.g. capturing pre-merge vision tokens, or the wrong LM hidden dim).
    with np.load(DUMP) as z:
        merged = z["merged_tokens"]
        projected = z["projected_tokens"]
        embeds = z["embeds_after_splice"]
        logits = z["logits_step0"]
        assert merged.shape == (256, 4608)
        assert projected.shape == (256, 2048)
        assert embeds.shape[-1] == 2048
        assert logits.ndim == 1
        merged_rows = merged.shape[0]
        projected_rows = projected.shape[0]
        logits_vocab = logits.shape[0]
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    vocab = man.get("vocab", man.get("vocab_size"))
    assert logits_vocab == vocab
    # cross-tie: merged rows == projected rows == merged_vision_tokens == #IMG_CONTEXT slots
    n = man["merged_vision_tokens"]
    assert merged_rows == n == projected_rows == 256
