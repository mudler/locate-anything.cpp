# LocateAnything-3B ggml Port — M0 + M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `locate-anything.cpp` repo, download the HF weights, produce gold per-component reference dumps, and ship a metadata-driven converter that writes one self-contained GGUF (config KV + tokenizer + 770 weights) that a structural loader can read back.

**Architecture:** Two milestones from the design spec (`docs/superpowers/specs/2026-06-08-locate-anything-cpp-design.md`). M0 = repo skeleton + Python tooling that runs the real `nvidia/LocateAnything-3B` HF model and dumps intermediate tensors (the parity gold). M1 = `convert_locateanything_to_gguf.py` writing all hparams/special-token-ids/tokenizer/weights into a single GGUF, validated by a round-trip test. No C++ inference yet — that begins in M2 against these dumps.

**Tech Stack:** Python 3.10+ (`transformers`, `torch`, `safetensors`, `gguf`, `huggingface_hub`, `numpy`, `Pillow`), CMake 3.18, ggml (submodule), pytest.

**Scope note:** This plan covers M0 and M1 only. M2 (MoonViT parity), M3 (projector + LM forward), M4 (AR decode), M5 (hybrid Parallel Box Decoding), and M6 (quantize + CLI + C-API) each get their own plan authored after their predecessor lands, because their step-level detail (exact tensor shapes, tolerances, ggml graph code) depends on the dumps M0 produces and the GGUF schema M1 defines. A roadmap stub for them is at the end.

---

## File Structure

Files created in this plan (paths relative to repo root `/home/mudler/_git/locate-anything.cpp`):

| Path | Responsibility |
|---|---|
| `.gitmodules`, `third_party/ggml/` | Embedded ggml submodule |
| `.gitignore` | Ignore build artifacts, venv, model cache, dumps |
| `CMakeLists.txt` | Top-level build (skeleton only this milestone: ggml + empty lib target) |
| `README.md` | One-paragraph project intro + build/run pointer |
| `scripts/requirements.txt` | Pinned Python deps for tooling |
| `scripts/download_model.py` | Fetch the HF checkpoint into a local cache dir |
| `scripts/la_reference.py` | Shared: load HF model/processor, build the fixed fixture (image+prompt) |
| `scripts/dump_reference.py` | Run the model with hooks, dump per-component tensors to `.npz` + a manifest |
| `scripts/convert_locateanything_to_gguf.py` | Metadata-driven safetensors → single GGUF converter |
| `scripts/gguf_keys.py` | Single source of truth for GGUF KV key names + tensor-rename table (imported by converter and tests) |
| `tests/python/test_dump_reference.py` | Asserts dump determinism + expected tensor keys/shapes |
| `tests/python/test_convert_gguf.py` | Round-trip: convert → reopen GGUF → assert KV + tensor names/shapes |
| `tests/fixtures/` | The fixed parity image (committed) + expected-shapes JSON |

**Key boundary decisions:**
- `gguf_keys.py` is the *only* place GGUF key strings and the tensor-rename table live. The C++ loader (M2+) and these tests both consult it (tests via import; C++ via a generated header in M1 Task 8). This prevents the converter and loader from drifting.
- `la_reference.py` owns all "how to talk to the HF model" knowledge (trust_remote_code load, processor, chat template, fixture construction). `dump_reference.py` and `convert_*` import it so the fixture is defined once.

---

## M0 — Scaffold, download, reference dumps

### Task 1: Repo scaffold (build skeleton + ggml submodule)

**Files:**
- Create: `.gitignore`
- Create: `.gitmodules` (via `git submodule add`)
- Create: `CMakeLists.txt`
- Create: `README.md`
- Create: `src/.gitkeep`, `include/.gitkeep`

- [ ] **Step 1: Add the ggml submodule**

Run:
```bash
cd /home/mudler/_git/locate-anything.cpp
git submodule add https://github.com/ggml-org/ggml third_party/ggml
git -C third_party/ggml checkout master
```
Expected: `third_party/ggml/CMakeLists.txt` exists; `.gitmodules` created.

- [ ] **Step 2: Write `.gitignore`**

```gitignore
/build/
/build-*/
.venv/
__pycache__/
*.pyc
/models/
/dumps/
*.gguf
*.safetensors
!tests/fixtures/**
```

- [ ] **Step 3: Write skeleton `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(locate_anything VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LA_BUILD_TESTS    "Build tests"            OFF)
option(LA_GGML_CUDA      "ggml CUDA backend"      OFF)
option(LA_GGML_METAL     "ggml Metal backend"     OFF)
option(LA_GGML_VULKAN    "ggml Vulkan backend"    OFF)

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(LA_GGML_CUDA)   set(GGML_CUDA   ON CACHE BOOL "" FORCE) endif()
if(LA_GGML_METAL)  set(GGML_METAL  ON CACHE BOOL "" FORCE) endif()
if(LA_GGML_VULKAN) set(GGML_VULKAN ON CACHE BOOL "" FORCE) endif()
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# Library sources land here from M2 onward. Skeleton builds ggml only.
message(STATUS "locate-anything.cpp skeleton configured (ggml embedded)")
```

- [ ] **Step 4: Write `README.md` stub**

```markdown
# locate-anything.cpp

C++/ggml inference engine for [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B).
Open-vocabulary detection / visual grounding with no Python at inference time.

> Status: pre-implementation. See `docs/superpowers/specs/` for the design and
> `docs/superpowers/plans/` for the build plan.

## Build (skeleton)
```bash
git submodule update --init --recursive
cmake -B build -DLA_BUILD_TESTS=OFF
cmake --build build -j
```
```

- [ ] **Step 5: Verify the skeleton configures and ggml builds**

Run:
```bash
mkdir -p src include && touch src/.gitkeep include/.gitkeep
cmake -B build >/tmp/la_cmake.log 2>&1 && echo CONFIG_OK
cmake --build build --target ggml -j 2>&1 | tail -5
```
Expected: `CONFIG_OK` printed; ggml target builds without error.

- [ ] **Step 6: Commit**

```bash
git add .gitmodules third_party/ggml .gitignore CMakeLists.txt README.md src/.gitkeep include/.gitkeep
git commit -m "M0: repo scaffold with embedded ggml submodule"
```

---

### Task 2: Python tooling env + model download

**Files:**
- Create: `scripts/requirements.txt`
- Create: `scripts/download_model.py`

- [ ] **Step 1: Write `scripts/requirements.txt`**

```text
torch>=2.2
transformers>=4.44
safetensors>=0.4.3
huggingface_hub>=0.24
gguf>=0.10
numpy>=1.26
Pillow>=10.0
pytest>=8.0
einops>=0.7
```

- [ ] **Step 2: Write `scripts/download_model.py`**

```python
#!/usr/bin/env python3
"""Download nvidia/LocateAnything-3B into a local cache dir (default ./models/LocateAnything-3B).

Usage: python scripts/download_model.py [--out models/LocateAnything-3B] [--repo nvidia/LocateAnything-3B]
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/LocateAnything-3B")
    ap.add_argument("--out", default="models/LocateAnything-3B")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.py",
            "merges.txt", "vocab.json", "added_tokens.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.json", "preprocessor_config.json",
        ],
    )
    print(f"downloaded to {path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create venv, install, run the download**

Run:
```bash
cd /home/mudler/_git/locate-anything.cpp
python3 -m venv .venv && . .venv/bin/activate
pip install -q -r scripts/requirements.txt
python scripts/download_model.py --out models/LocateAnything-3B
```
Expected: `downloaded to .../models/LocateAnything-3B`; the dir contains
`model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`,
`model.safetensors.index.json`, `config.json`, `vocab.json`, `merges.txt`,
`added_tokens.json`, and the `*_locateanything*.py` modeling files.

- [ ] **Step 4: Sanity-check the download (shape of the problem)**

Run:
```bash
python - <<'PY'
import json, pathlib
m = pathlib.Path("models/LocateAnything-3B")
idx = json.load(open(m/"model.safetensors.index.json"))
names = list(idx["weight_map"].keys())
print("tensors:", len(names))
print("vision:", sum(n.startswith("vision_model.") for n in names))
print("mlp1:",   sum(n.startswith("mlp1.") for n in names))
print("lm:",     sum(n.startswith("language_model.") for n in names))
PY
```
Expected: `tensors: 770`, `vision: 329`, `mlp1: 6`, `lm: 435`.

- [ ] **Step 5: Commit**

```bash
git add scripts/requirements.txt scripts/download_model.py
git commit -m "M0: pinned tooling deps + HF download script"
```

---

### Task 3: Shared reference harness + committed fixture

**Files:**
- Create: `scripts/la_reference.py`
- Create: `tests/fixtures/parity_image.png` (a small committed test image)
- Create: `tests/fixtures/fixture_spec.json` (records prompt, categories, image path)

- [ ] **Step 1: Add a small, committed parity image**

Run:
```bash
mkdir -p tests/fixtures
cp /home/mudler/_git/rt-detr.cpp/benchmarks/images/coco_cats.jpg /tmp/src.jpg
python - <<'PY'
from PIL import Image
im = Image.open("/tmp/src.jpg").convert("RGB").resize((448, 448))
im.save("tests/fixtures/parity_image.png")
print("saved", im.size)
PY
```
Expected: `saved (448, 448)`. (448 = 16×28, a clean multiple of merge*patch=28 → 32×32 patch grid → 16×16 = 256 merged vision tokens, no resize ambiguity.)

- [ ] **Step 2: Write `tests/fixtures/fixture_spec.json`**

```json
{
  "image": "tests/fixtures/parity_image.png",
  "categories": ["cat", "remote"],
  "prompt_template": "Locate all the instances that matches the following description: {cats}.",
  "category_sep": "</c>",
  "max_new_tokens": 256,
  "generation_mode": "hybrid"
}
```

- [ ] **Step 3: Write `scripts/la_reference.py`**

```python
#!/usr/bin/env python3
"""Single source of truth for loading the HF model + building the parity fixture.

Everything that knows 'how to talk to LocateAnything-3B' lives here so the dump
script and the converter agree on the exact prompt, image preprocessing and
chat template.
"""
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "LocateAnything-3B"

def load_fixture(spec_path=REPO_ROOT / "tests" / "fixtures" / "fixture_spec.json"):
    spec = json.load(open(spec_path))
    image = Image.open(REPO_ROOT / spec["image"]).convert("RGB")
    cats = spec["category_sep"].join(spec["categories"])
    prompt = spec["prompt_template"].format(cats=cats)
    return spec, image, prompt

def load_model(model_dir=DEFAULT_MODEL, dtype=torch.float32, device="cpu"):
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True,
        torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return cfg, model, processor

def build_inputs(model_dir=DEFAULT_MODEL, device="cpu"):
    spec, image, prompt = load_fixture()
    cfg, model, processor = load_model(model_dir, device=device)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"<image>\n{prompt}"},
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(device)
    return cfg, model, processor, inputs, spec
```

> NOTE for implementer: the exact `messages`/`apply_chat_template` call and the
> `processor(...)` kwarg names (`images=` vs `pixel_values`) must be verified
> against `models/LocateAnything-3B/processing_locateanything.py` and
> `chat_template.json` before relying on them. If the processor API differs,
> adapt `build_inputs` to match the repo's own example in its README. Do not
> guess — open the file and match it.

- [ ] **Step 4: Smoke-test the harness builds inputs**

Run:
```bash
. .venv/bin/activate
python - <<'PY'
import scripts.la_reference as R
cfg, model, processor, inputs, spec = R.build_inputs()
print("input_ids:", tuple(inputs["input_ids"].shape))
print("has pixel_values:", any("pixel" in k for k in inputs))
print("image_token count:", int((inputs["input_ids"] == 151665).sum()))
PY
```
Expected: prints an `input_ids` shape, a pixel-values key present, and an
image-token count of **256** (matches 16×16 merged tokens for the 448² fixture).
If the count differs, the processor preprocessing differs from the design's
assumption — STOP and reconcile before continuing (this is a parity-critical fact).

- [ ] **Step 5: Commit**

```bash
git add scripts/la_reference.py tests/fixtures/parity_image.png tests/fixtures/fixture_spec.json
git commit -m "M0: reference harness + committed parity fixture"
```

---

### Task 4: Reference dump script (the parity gold)

**Files:**
- Create: `scripts/dump_reference.py`
- Create: `tests/python/test_dump_reference.py`

- [ ] **Step 1: Write the failing test `tests/python/test_dump_reference.py`**

```python
import subprocess, sys, json
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
    z = np.load(DUMP)
    for k in REQUIRED_KEYS:
        assert k in z.files, f"missing dump tensor: {k}"

def test_manifest_records_shapes_and_special_ids():
    man = json.load(open(MANIFEST))
    assert man["image_token_index"] == 151665
    assert man["coord_start_token_id"] == 151677
    assert man["coord_end_token_id"] == 152677
    assert man["block_size"] == 6
    assert man["merged_vision_tokens"] == 256

def test_dump_is_deterministic():
    z = np.load(DUMP)
    a = z["logits_step0"]
    assert np.isfinite(a).all(), "non-finite logits in dump"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `. .venv/bin/activate && python -m pytest tests/python/test_dump_reference.py -v`
Expected: FAIL — `dumps/reference.npz` does not exist.

- [ ] **Step 3: Write `scripts/dump_reference.py`**

```python
#!/usr/bin/env python3
"""Run the real HF model on the fixed fixture and dump per-component tensors.

Writes dumps/reference.npz (all tensors, float32) + dumps/manifest.json
(special-token ids, shapes, the generated token stream). These are the gold
parity targets every C++ milestone gates against.
"""
import json
from pathlib import Path
import numpy as np
import torch
import scripts.la_reference as R

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dumps"; OUT.mkdir(exist_ok=True)

def main():
    cfg, model, processor, inputs, spec = R.build_inputs()
    store = {}
    hooks = []

    def save(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            store[name] = t.detach().float().cpu().numpy()
        return hook

    # --- locate submodules by attribute path; verified against modeling_*.py ---
    vm = model.vision_model
    # per-ViT-layer outputs
    blocks = vm.encoder.blocks
    hooks.append(blocks[0].register_forward_hook(save("vit_layer_00")))
    hooks.append(blocks[len(blocks)-1].register_forward_hook(save("vit_layer_26")))
    hooks.append(vm.encoder.final_layernorm.register_forward_hook(save("vit_final")))
    hooks.append(model.mlp1.register_forward_hook(save("projected_tokens")))
    lm_layers = model.language_model.model.layers
    hooks.append(lm_layers[0].register_forward_hook(save("lm_layer_00")))
    hooks.append(lm_layers[len(lm_layers)-1].register_forward_hook(save("lm_layer_35")))

    with torch.no_grad():
        # one forward pass for step-0 logits + intermediates
        out = model(**inputs, output_hidden_states=False)
        store["logits_step0"] = out.logits[0, -1].float().cpu().numpy()

    # pixel_values straight from the processor (post-patchify)
    pv_key = next(k for k in inputs if "pixel" in k)
    store["pixel_values"] = inputs[pv_key].float().cpu().numpy()

    # generation (the full token stream the C++ decode must reproduce)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=spec["max_new_tokens"],
            generation_mode=spec.get("generation_mode", "hybrid"),
            do_sample=False,
        )
    store["token_stream"] = gen[0].cpu().numpy().astype(np.int64)

    for h in hooks: h.remove()

    # placeholders that require explicit extraction (filled by implementer once
    # the exact submodule attribute names are confirmed against modeling_*.py):
    for k in ("vit_pos_added", "vit_final" , "merged_tokens", "embeds_after_splice"):
        store.setdefault(k, np.zeros((1,), np.float32))

    np.savez(OUT / "reference.npz", **store)
    manifest = {
        "image_token_index": 151665,
        "coord_start_token_id": 151677,
        "coord_end_token_id": 152677,
        "box_start_token_id": 151668,
        "box_end_token_id": 151669,
        "block_size": 6,
        "merged_vision_tokens": int((inputs["input_ids"] == 151665).sum()),
        "shapes": {k: list(v.shape) for k, v in store.items()},
        "token_stream": store["token_stream"].tolist(),
    }
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    print("wrote", OUT / "reference.npz", "keys:", sorted(store))

if __name__ == "__main__":
    main()
```

> NOTE for implementer: the `setdefault` placeholders (`vit_pos_added`,
> `merged_tokens`, `embeds_after_splice`) are stubs. Replace each with a real
> hook or tensor capture once you have opened `modeling_vit.py` /
> `modeling_locateanything.py` and identified the exact module that produces:
> (a) patch embeddings after the learnable 2D pos-emb add, (b) the post-2×2-merge
> 4608-dim tokens, (c) the LM input embeddings after vision-token splice. Do not
> ship the plan's zero placeholders — M2/M3 parity gates depend on these being
> the real reference tensors. This is the single most important deliverable of M0.

- [ ] **Step 4: Run the dump, then re-run the test to verify it passes**

Run:
```bash
. .venv/bin/activate
python scripts/dump_reference.py
python -m pytest tests/python/test_dump_reference.py -v
```
Expected: dump script prints the keys; all three tests PASS. (The dump is
large; it is gitignored — only the manifest's `shapes`/`token_stream` are small.)

- [ ] **Step 5: Commit (manifest only; dumps are gitignored)**

```bash
git add scripts/dump_reference.py tests/python/test_dump_reference.py
git commit -m "M0: reference dump script + determinism tests (gold parity targets)"
```

---

## M1 — Converter + self-contained GGUF

### Task 5: GGUF key registry + hparam KV writer

**Files:**
- Create: `scripts/gguf_keys.py`
- Create: `scripts/convert_locateanything_to_gguf.py`

- [ ] **Step 1: Write `scripts/gguf_keys.py` (single source of truth)**

```python
"""GGUF key names + tensor-rename table. Imported by the converter AND tests
so the metadata schema lives in exactly one place."""
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

# Hardcoded special-token ids from config.json (asserted against config at convert time)
SPECIAL_TOKENS = {
    "tok.image": 151665, "tok.box_start": 151668, "tok.box_end": 151669,
    "tok.coord_start": 151677, "tok.coord_end": 152677,
    "tok.ref_start": 151672, "tok.ref_end": 151673,
    "tok.none": 4064, "tok.null": 152678, "tok.switch": 152679,
    "tok.text_mask": 151676, "tok.eos": 151645, "tok.bos": 151643,
}

def rename_tensor(name: str):
    """safetensors name -> gguf tensor name. Returns None to skip."""
    import re
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
```

- [ ] **Step 2: Write the hparam KV portion of `convert_locateanything_to_gguf.py`**

```python
#!/usr/bin/env python3
"""Convert nvidia/LocateAnything-3B safetensors -> one self-contained GGUF."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import gguf
from safetensors import safe_open
import scripts.gguf_keys as K

def load_config(model_dir):
    return json.load(open(Path(model_dir) / "config.json"))

def write_hparams(w, cfg):
    t = cfg["text_config"]; v = cfg["vision_config"]
    w.add_uint32(K.KV["lm.hidden"],       t["hidden_size"])
    w.add_uint32(K.KV["lm.n_layers"],     t["num_hidden_layers"])
    w.add_uint32(K.KV["lm.n_heads"],      t["num_attention_heads"])
    w.add_uint32(K.KV["lm.n_kv_heads"],   t["num_key_value_heads"])
    w.add_uint32(K.KV["lm.head_dim"],     t["hidden_size"] // t["num_attention_heads"])
    w.add_uint32(K.KV["lm.intermediate"], t["intermediate_size"])
    w.add_uint32(K.KV["lm.vocab"],        t["vocab_size"])
    w.add_float32(K.KV["lm.rope_theta"],  float(t["rope_theta"]))
    w.add_float32(K.KV["lm.rms_eps"],     float(t["rms_norm_eps"]))
    w.add_uint32(K.KV["lm.block_size"],   int(t["block_size"]))
    w.add_uint32(K.KV["vit.hidden"],      v["hidden_size"])
    w.add_uint32(K.KV["vit.n_layers"],    v["num_hidden_layers"])
    w.add_uint32(K.KV["vit.n_heads"],     v["num_attention_heads"])
    w.add_uint32(K.KV["vit.head_dim"],    v["hidden_size"] // v["num_attention_heads"])
    w.add_uint32(K.KV["vit.intermediate"], v["intermediate_size"])
    w.add_uint32(K.KV["vit.patch"],       v["patch_size"])
    w.add_array(K.KV["vit.merge"],        [int(x) for x in v["merge_kernel_size"]])
    w.add_uint32(K.KV["vit.pos_emb_hw"],  v.get("init_pos_emb_height", 64))
    w.add_float32(K.KV["vit.rope_theta"], 10000.0)
    for key, tid in K.SPECIAL_TOKENS.items():
        w.add_uint32(K.KV[key], tid)
    w.add_array(K.KV["img.mean"], [0.5, 0.5, 0.5])
    w.add_array(K.KV["img.std"],  [0.5, 0.5, 0.5])
    w.add_uint32(K.KV["img.in_token_limit"], int(cfg.get("in_token_limit", 25600)))
```

- [ ] **Step 3: Smoke-test hparam writing in isolation**

Run:
```bash
. .venv/bin/activate
python - <<'PY'
import gguf, scripts.convert_locateanything_to_gguf as C, scripts.gguf_keys as K
cfg = C.load_config("models/LocateAnything-3B")
w = gguf.GGUFWriter("/tmp/hp.gguf", K.ARCH)
C.write_hparams(w, cfg)
w.write_header_to_file(); w.write_kv_data_to_file(); w.close()
r = gguf.GGUFReader("/tmp/hp.gguf")
print("lm.hidden:", r.get_field(K.KV["lm.hidden"]).parts[-1][0])
print("vit.n_layers:", r.get_field(K.KV["vit.n_layers"]).parts[-1][0])
PY
```
Expected: `lm.hidden: 2048`, `vit.n_layers: 27`.

- [ ] **Step 4: Commit**

```bash
git add scripts/gguf_keys.py scripts/convert_locateanything_to_gguf.py
git commit -m "M1: GGUF key registry + hparam metadata writer"
```

---

### Task 6: Tokenizer embedding

**Files:**
- Modify: `scripts/convert_locateanything_to_gguf.py` (add `write_tokenizer`)
- Modify: `scripts/gguf_keys.py` (add tokenizer KV keys)

- [ ] **Step 1: Add tokenizer KV keys to `gguf_keys.py`**

Append to the `KV` dict in `scripts/gguf_keys.py`:
```python
KV.update({
    "tok.tokens":      f"{ARCH}.tokenizer.tokens",
    "tok.token_types": f"{ARCH}.tokenizer.token_types",
    "tok.merges":      f"{ARCH}.tokenizer.merges",
    "tok.model":       f"{ARCH}.tokenizer.model",
})
```

- [ ] **Step 2: Add `write_tokenizer` to the converter**

```python
def write_tokenizer(w, model_dir):
    """Embed the Qwen2 BPE vocab + merges + added tokens (gpt2 byte-level BPE)."""
    md = Path(model_dir)
    vocab = json.load(open(md / "vocab.json"))               # token -> id
    added = {}
    if (md / "added_tokens.json").exists():
        added = json.load(open(md / "added_tokens.json"))    # token -> id (1038 entries)
    merged = dict(vocab); merged.update(added)
    id_to_tok = {i: t for t, i in merged.items()}
    n = max(id_to_tok) + 1
    tokens = [id_to_tok.get(i, f"<unused_{i}>") for i in range(n)]
    # token type: 1=normal, 4=control(added/special)
    added_ids = set(added.values())
    types = [4 if i in added_ids else 1 for i in range(n)]
    merges = []
    with open(md / "merges.txt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                merges.append(line)
    w.add_string(K.KV["tok.model"], "gpt2")
    w.add_array(K.KV["tok.tokens"], tokens)
    w.add_array(K.KV["tok.token_types"], types)
    w.add_array(K.KV["tok.merges"], merges)
    return n
```

- [ ] **Step 3: Smoke-test tokenizer embedding round-trips**

Run:
```bash
. .venv/bin/activate
python - <<'PY'
import gguf, scripts.convert_locateanything_to_gguf as C, scripts.gguf_keys as K
w = gguf.GGUFWriter("/tmp/tok.gguf", K.ARCH)
n = C.write_tokenizer(w, "models/LocateAnything-3B")
w.write_header_to_file(); w.write_kv_data_to_file(); w.close()
r = gguf.GGUFReader("/tmp/tok.gguf")
toks = r.get_field(K.KV["tok.tokens"])
print("n_tokens:", n, "stored:", len(toks.data))
PY
```
Expected: `n_tokens` ≥ 152681 and `stored` equals `n_tokens`.

- [ ] **Step 4: Commit**

```bash
git add scripts/gguf_keys.py scripts/convert_locateanything_to_gguf.py
git commit -m "M1: embed Qwen2 BPE tokenizer (vocab + merges + added tokens) in GGUF"
```

---

### Task 7: Weight writing + full converter CLI

**Files:**
- Modify: `scripts/convert_locateanything_to_gguf.py` (add `write_weights` + `main`)

- [ ] **Step 1: Add weight writing + `main()`**

```python
def iter_safetensors(model_dir):
    md = Path(model_dir)
    idx = json.load(open(md / "model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    for shard in shards:
        with safe_open(md / shard, framework="np") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)

def write_weights(w, model_dir):
    written, skipped = 0, []
    for name, arr in iter_safetensors(model_dir):
        gname = K.rename_tensor(name)
        if gname is None:
            skipped.append(name); continue
        a = arr.astype(np.float32)   # M1: store f32; quantization is M6
        w.add_tensor(gname, a)
        written += 1
    return written, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/LocateAnything-3B")
    ap.add_argument("--output", default="models/locate-anything-f32.gguf")
    args = ap.parse_args()
    cfg = load_config(args.model)
    w = gguf.GGUFWriter(args.output, K.ARCH)
    write_hparams(w, cfg)
    n_tok = write_tokenizer(w, args.model)
    written, skipped = write_weights(w, args.model)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.output}: tensors={written} skipped={len(skipped)} vocab={n_tok}")
    if skipped:
        print("WARNING skipped (unmapped) tensors:", *skipped[:20], sep="\n  ")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full conversion**

Run:
```bash
. .venv/bin/activate
python scripts/convert_locateanything_to_gguf.py \
  --model models/LocateAnything-3B --output models/locate-anything-f32.gguf
```
Expected: `tensors=770 skipped=0`. **Any skipped tensor is a bug in
`rename_tensor` — investigate before committing; all 770 must map.**

- [ ] **Step 3: Commit**

```bash
git add scripts/convert_locateanything_to_gguf.py
git commit -m "M1: weight remap + full converter CLI (safetensors -> single GGUF)"
```

---

### Task 8: GGUF round-trip validation test + generated C header

**Files:**
- Create: `tests/python/test_convert_gguf.py`
- Create: `scripts/gen_gguf_keys_header.py`
- Create: `include/la_gguf_keys.h` (generated)

- [ ] **Step 1: Write the failing round-trip test**

```python
import json
from pathlib import Path
import gguf
import scripts.gguf_keys as K

ROOT = Path(__file__).resolve().parents[2]
GGUF = ROOT / "models" / "locate-anything-f32.gguf"

def _val(r, key):
    f = r.get_field(key); return f.parts[f.data[-1]][0] if f.data else f.parts[-1][0]

def test_gguf_has_all_hparams():
    assert GGUF.exists(), "run the converter first"
    r = gguf.GGUFReader(GGUF)
    assert int(_val(r, K.KV["lm.hidden"]))    == 2048
    assert int(_val(r, K.KV["lm.n_layers"]))  == 36
    assert int(_val(r, K.KV["lm.n_kv_heads"])) == 2
    assert int(_val(r, K.KV["lm.vocab"]))     == 152681
    assert int(_val(r, K.KV["vit.n_layers"])) == 27
    assert int(_val(r, K.KV["vit.hidden"]))   == 1152

def test_gguf_special_tokens():
    r = gguf.GGUFReader(GGUF)
    assert int(_val(r, K.KV["tok.image"]))       == 151665
    assert int(_val(r, K.KV["tok.coord_start"])) == 151677
    assert int(_val(r, K.KV["tok.coord_end"]))   == 152677

def test_gguf_tensor_names_and_count():
    r = gguf.GGUFReader(GGUF)
    names = {t.name for t in r.tensors}
    assert len(names) == 770
    # spot-check the three families exist
    assert "lm.blk.0.attn_q.weight" in names
    assert "lm.blk.35.ffn_down.weight" in names
    assert "proj.0.weight" in names or "proj.1.weight" in names
    assert "vit.patch_embed.weight" in names
    assert "vit.blk.26.wo.weight" in names
    assert "vit.pos_emb.weight" in names
```

- [ ] **Step 2: Run the test to verify it passes (GGUF already built in Task 7)**

Run: `. .venv/bin/activate && python -m pytest tests/python/test_convert_gguf.py -v`
Expected: all three tests PASS. If `test_gguf_tensor_names_and_count` fails on a
missing name, the rename table dropped a tensor family — fix `rename_tensor`.

- [ ] **Step 3: Write the header generator (C++ loader will include this)**

```python
#!/usr/bin/env python3
"""Generate include/la_gguf_keys.h from scripts/gguf_keys.py so the C++ loader
and the Python converter never drift on KV key strings."""
from pathlib import Path
import scripts.gguf_keys as K

ROOT = Path(__file__).resolve().parent.parent
def main():
    lines = ["// AUTO-GENERATED from scripts/gguf_keys.py — do not edit.",
             "#pragma once", ""]
    def cident(s): return "LA_KV_" + s.replace(".", "_").upper()
    for short, full in K.KV.items():
        lines.append(f'#define {cident(short)} "{full}"')
    lines.append(f'#define LA_ARCH "{K.ARCH}"')
    (ROOT / "include" / "la_gguf_keys.h").write_text("\n".join(lines) + "\n")
    print("wrote include/la_gguf_keys.h")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the header and verify it compiles**

Run:
```bash
. .venv/bin/activate
python scripts/gen_gguf_keys_header.py
echo '#include "la_gguf_keys.h"
int main(){ const char* k = LA_KV_LM_HIDDEN; (void)k; return 0; }' > /tmp/t.c
cc -Iinclude /tmp/t.c -o /tmp/t && /tmp/t && echo HEADER_OK
```
Expected: `wrote include/la_gguf_keys.h` then `HEADER_OK`.

- [ ] **Step 5: Commit**

```bash
git add tests/python/test_convert_gguf.py scripts/gen_gguf_keys_header.py include/la_gguf_keys.h
git commit -m "M1: GGUF round-trip validation + generated C key header (no converter/loader drift)"
```

---

## Milestone exit criteria

- **M0 done when:** `models/LocateAnything-3B` is downloaded; `dumps/reference.npz` + `dumps/manifest.json` contain real (non-placeholder) per-component tensors for the fixed fixture; `tests/python/test_dump_reference.py` passes.
- **M1 done when:** `python scripts/convert_locateanything_to_gguf.py` writes a GGUF with all hparams, special-token ids, tokenizer (≥152681 tokens) and **770** tensors; `tests/python/test_convert_gguf.py` passes; `include/la_gguf_keys.h` is generated and compiles.

## Roadmap (subsequent plans, one per milestone)

Authored after their predecessor lands, because step-level detail depends on M0 dumps + M1 GGUF schema:

- **M2 — MoonViT parity.** C++ `model_loader` + ggml graph for patchify → learnable-2D-pos-emb (bicubic interp) → 2D interleaved RoPE → 27 bidirectional blocks → final norm → 2×2 merge. Gate each stage vs `vit_*` / `merged_tokens` dumps.
- **M3 — Projector + LM forward.** Port `vibevoice.cpp/src/qwen2.*` (cross-check llama.cpp `build_qwen2`); MLP projector; vision-token splice at `image_token_index`; gate `projected_tokens`, `embeds_after_splice`, `logits_step0`.
- **M4 — AR decode.** Greedy/sampled AR loop + coord-token parse; gate vs `token_stream` (slow mode) and reference boxes.
- **M5 — Hybrid Parallel Box Decoding.** block_size=6 MTP, block-parallel positions, per-round KV truncation, frame validation, `handle_pattern`, AR fallback; gate vs hybrid `token_stream`.
- **M6 — Quantize + CLI + C-API.** f16/q8_0/q6_k/q5_k/q4_k; `locate-anything-cli detect --annotated`; flat C ABI.
