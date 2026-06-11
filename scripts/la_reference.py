#!/usr/bin/env python3
r"""Single source of truth for loading the HF model + building the parity fixture.

This module is the ONE place that knows how to talk to the HF LocateAnything-3B
model. Every later milestone (reference dump, converter validation) imports it so
the fixture (image + prompt + chat template) is defined exactly once.

API notes (verified against the repo's remote-code files, NOT guessed):
- The chat template (`chat_template.json`) only emits an image marker when a
  message's `content` is a LIST of dicts; a plain `<image>` string in a string
  content is NOT expanded. So the user message MUST use the structured-content
  form `[{"type": "image"}, {"type": "text", "text": ...}]`.
- The template writes the placeholder as `<image-1>` (1-indexed). The processor's
  `replace_media_placeholder` (regex `<(image|video)-(\d+)>`) then expands that
  single token into `<img>` + N * `<IMG_CONTEXT>` + `</img>`, where
  N = (grid_h * grid_w) // (merge_h * merge_w).
- Image preprocessing: patch_size=14, merge_kernel_size=[2,2] (effective stride
  28). A 448x448 image -> 32x32 patch grid -> 16x16 = 256 merged vision tokens.
- `<IMG_CONTEXT>` has token id 151665.
- Images are passed to the processor via the `images=` kwarg as a list of PIL
  images; text via `text=` as a list of strings.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "LocateAnything-3B"

# Token id of <IMG_CONTEXT> (the per-patch vision placeholder). Verified in
# models/LocateAnything-3B/added_tokens.json.
IMAGE_CONTEXT_TOKEN_ID = 151665


def load_fixture(spec_path=REPO_ROOT / "tests" / "fixtures" / "fixture_spec.json"):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    image = Image.open(REPO_ROOT / spec["image"]).convert("RGB")
    cats = spec["category_sep"].join(spec["categories"])
    prompt = spec["prompt_template"].format(cats=cats)
    return spec, image, prompt


def load_model(model_dir=DEFAULT_MODEL, dtype=torch.float32, device="cpu"):
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # Parity: keep the model's native "magi" attention so the remote code builds
    # the faithful block-diffusion mask (causal prefix + bidirectional "mutual"
    # blocks within the MTP/generation windows; causal_attn=False). With no
    # MagiAttention CUDA kernel on CPU, Qwen2DecoderLayer's documented fallback
    # (magi -> sdpa) engages mask_sdpa_utils, which reproduces that exact mask
    # densely via Qwen2SdpaAttention. We must pin text_config to "magi" too:
    # transformers resolves sub-configs to "eager" otherwise, which the remote
    # code's mask path rejects (NotImplementedError). Forcing plain "sdpa" at
    # the top level would instead drop to standard causal attention and break
    # parity with the real model.
    cfg._attn_implementation = "magi"
    cfg.text_config._attn_implementation = "magi"
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True,
        torch_dtype=dtype, config=cfg, attn_implementation="magi",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return cfg, model, processor


def build_messages(prompt):
    """Build the chat messages in the structured-content form the chat template
    requires to emit the `<image-1>` placeholder. The system turn is added
    automatically by the chat template (first turn is not a system turn)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def build_inputs(model_dir=DEFAULT_MODEL, device="cpu"):
    spec, image, prompt = load_fixture()
    cfg, model, processor = load_model(model_dir, device=device)
    messages = build_messages(prompt)
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(device)
    # The processor leaves image_grid_hws as a numpy ndarray (it is not picked up
    # by return_tensors="pt"). The vendored MoonViT does
    # torch.zeros(..., dtype=grid_hws.dtype), which fails on a numpy dtype, so
    # convert any stray numpy arrays to torch tensors before the forward pass.
    for k, v in list(inputs.items()):
        if isinstance(v, np.ndarray):
            inputs[k] = torch.as_tensor(v).to(device)
    return cfg, model, processor, inputs, spec
