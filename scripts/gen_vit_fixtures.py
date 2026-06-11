#!/usr/bin/env python3
"""Isolated fixtures for the host-computed MoonViT ops, into dumps/vit_fixtures.gguf.
This task: posemb_interp_32x32 = F.interpolate(pos_emb[64,64,1152]->[32,32], bicubic,
align_corners=False) flattened [1024,1152]. (rope fixtures appended in Task 6.)"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
import gguf

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "LocateAnything-3B"

def load_pos_emb():
    idx = json.load(open(MODEL/"model.safetensors.index.json"))
    shard = idx["weight_map"]["vision_model.patch_embed.pos_emb.weight"]
    with safe_open(MODEL/shard, framework="pt") as f:
        return f.get_tensor("vision_model.patch_embed.pos_emb.weight").float()  # [64,64,1152]

def main():
    out = ROOT / "dumps" / "vit_fixtures.gguf"
    w = gguf.GGUFWriter(str(out), "vit_fixtures")
    pe = load_pos_emb()  # [64,64,1152]
    interp = F.interpolate(pe.permute(2,0,1).unsqueeze(0), size=(32,32),
                           mode="bicubic", align_corners=False).squeeze(0).permute(1,2,0)
    interp = interp.reshape(-1, 1152).contiguous().numpy().astype(np.float32)  # [1024,1152]
    w.add_tensor("posemb_interp_32x32", np.ascontiguousarray(interp))

    # --- Task 6: 2D RoPE fixture (mirrors Rope2DPosEmb + apply_rope exactly) ---
    torch.manual_seed(0)
    H, W = 32, 32; n = H*W; heads = 16; hd = 72
    q = torch.randn(n, heads, hd)
    k = torch.randn(n, heads, hd)
    dim_range = torch.arange(0, hd, 4).float()          # [0,4,...,68], len 18
    invfreq = 1.0 / (10000.0 ** (dim_range / hd))       # [18]
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    xpos = xs.reshape(-1).float(); ypos = ys.reshape(-1).float()   # [1024], x=column,y=row
    xfreq = torch.outer(xpos, invfreq); yfreq = torch.outer(ypos, invfreq)  # [1024,18]
    xcis = torch.polar(torch.ones_like(xfreq), xfreq)
    ycis = torch.polar(torch.ones_like(yfreq), yfreq)
    fc = torch.cat([xcis.unsqueeze(-1), ycis.unsqueeze(-1)], dim=-1).reshape(n, hd//2)  # complex [1024,36]
    def apply(x):
        xc = torch.view_as_complex(x.float().reshape(n, heads, hd//2, 2))
        return torch.view_as_real(xc * fc.unsqueeze(1)).reshape(n, heads, hd)
    qo = apply(q).contiguous().numpy().astype(np.float32)
    ko = apply(k).contiguous().numpy().astype(np.float32)
    w.add_tensor("rope_q_in",  np.ascontiguousarray(q.numpy().astype(np.float32)))   # [1024,16,72]
    w.add_tensor("rope_k_in",  np.ascontiguousarray(k.numpy().astype(np.float32)))
    w.add_tensor("rope_q_out", np.ascontiguousarray(qo))
    w.add_tensor("rope_k_out", np.ascontiguousarray(ko))

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", out, "posemb_interp_32x32", interp.shape)

if __name__ == "__main__":
    main()
