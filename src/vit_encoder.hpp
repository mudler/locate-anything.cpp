#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <vector>
namespace la {
class VitEncoder {
public:
    VitEncoder(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}
    // Runs patch_embed + bicubic pos-emb. pixel_host = [588*gh*gw] flattened
    // (per-patch 588 in (c,i,j) order, token-major). (gh,gw) is the real image
    // patch grid. Returns [1152*gh*gw] (hidden,token).
    bool patch_and_pos(const std::vector<float>& pixel_host, int gh, int gw,
                       std::vector<float>& out);
    // Runs patch+pos then ONE encoder block (layer 0). Returns [1152*gh*gw]
    // (hidden,token) raw flat == [token,hidden] row-major.
    bool block0(const std::vector<float>& pixel_host, int gh, int gw,
                std::vector<float>& out);
    // Runs patch+pos -> 27 blocks -> final_norm. Returns vit_final [1152*gh*gw] flat
    // (hidden,token) raw == [token,hidden] row-major. (gh,gw) is the real image
    // patch grid (drives pos-emb interp, 2D RoPE grid, token count). capture_layers:
    // block indices whose OUTPUT to also read back (in order) into `captured`.
    bool forward(const std::vector<float>& pixel_host, int gh, int gw,
                 std::vector<float>& vit_final_out,
                 const std::vector<int>& capture_layers,
                 std::vector<std::vector<float>>& captured);
private:
    // Builds patch_embed + bicubic pos-emb in-graph; returns [1152,gh*gw]
    // (hidden,token). pos_tok_keep holds the host pos-emb (kept alive by caller).
    ggml_tensor* build_patch_pos(ggml_context* ctx, GraphInputPool& pool,
                                 const std::vector<float>& pixel_host,
                                 int gh, int gw,
                                 std::vector<float>& pos_tok_keep);
    ModelLoader& ml_; Backend& be_;
};

// 2x2 patch merger. vf: vit_final flat in [token, hidden] row-major order
// (token outer, hidden inner), token = row*gw + col, gh=gw=32, c=1152.
// Returns merged [mh*mw, 4*c] = [256, 4608] flat (merged-token outer, channel
// inner), channel concat order TL,TR,BL,BR.
std::vector<float> merge_patches(const std::vector<float>& vf, int gh, int gw, int c);
}
