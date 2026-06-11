#include "vit_encoder.hpp"
#include "vit_posemb.hpp"
#include "vit_rope.hpp"
#include "ggml_extend.hpp"
#include <cstring>
#include <cmath>

namespace la {

// Per-layer weights for one MoonViT encoder block.
struct VitLayerWeights {
    ggml_tensor *norm0_w,*norm0_b, *wqkv_w,*wqkv_b, *wo_w,*wo_b,
                *norm1_w,*norm1_b, *fc0_w,*fc0_b, *fc1_w,*fc1_b;
};

static VitLayerWeights load_layer(const la::ModelLoader& ml, int i){
    auto t=[&](const std::string& s){ return ml.tensor("vit.blk."+std::to_string(i)+"."+s); };
    return VitLayerWeights{
        t("norm0.weight"),t("norm0.bias"), t("wqkv.weight"),t("wqkv.bias"),
        t("wo.weight"),t("wo.bias"), t("norm1.weight"),t("norm1.bias"),
        t("fc0.weight"),t("fc0.bias"), t("fc1.weight"),t("fc1.bias")};
}

// x: [hidden=1152, tokens]. Returns block output same shape.
// cosb/sinb: pre-built rope cos/sin graph inputs, shared across all blocks.
static ggml_tensor* build_block(ggml_context* ctx, ggml_tensor* x,
                                const VitLayerWeights& w,
                                ggml_tensor* cosb, ggml_tensor* sinb,
                                const la::RopeTables& rt, int heads, int hd){
    const int H=heads, D=hd; const int tok = (int)x->ne[1];
    // --- attention ---
    ggml_tensor* xn = la::layernorm(ctx, x, w.norm0_w, w.norm0_b, 1e-5f);
    ggml_tensor* qkv = la::linear(ctx, w.wqkv_w, xn, w.wqkv_b);        // [3456, tok]
    ggml_tensor* qkv4 = ggml_reshape_4d(ctx, qkv, D, H, 3, tok);       // [D,H,3,tok]
    auto slice = [&](int idx){
        return ggml_cont(ctx, ggml_view_4d(ctx, qkv4, D,H,1,tok,
                  qkv4->nb[1],qkv4->nb[2],qkv4->nb[3], idx*qkv4->nb[2])); // [D,H,1,tok]
    };
    ggml_tensor* q = ggml_reshape_3d(ctx, slice(0), D,H,tok);
    ggml_tensor* k = ggml_reshape_3d(ctx, slice(1), D,H,tok);
    ggml_tensor* v = ggml_reshape_3d(ctx, slice(2), D,H,tok);
    q = la::apply_rope(ctx, q, cosb, sinb, rt);
    k = la::apply_rope(ctx, k, cosb, sinb, rt);
    // attention: permute to [D,tok,H]; scores = mul_mat(k,q) -> [tok_k,tok_q,H]
    ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0,2,1,3));   // [D,tok,H]
    ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0,2,1,3));
    ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 0,2,1,3));
    ggml_tensor* scores = ggml_mul_mat(ctx, kp, qp);                  // [tok_k,tok_q,H]
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f/std::sqrt((float)D), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vp, 1,0,2,3)); // [tok,D,H]
    ggml_tensor* o  = ggml_mul_mat(ctx, vt, scores);                  // [D,tok_q,H]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0,2,1,3));                // [D,H,tok]
    o = ggml_reshape_2d(ctx, o, D*H, tok);                            // [1152,tok]
    o = la::linear(ctx, w.wo_w, o, w.wo_b);
    x = ggml_add(ctx, x, o);
    // --- mlp ---
    ggml_tensor* mn = la::layernorm(ctx, x, w.norm1_w, w.norm1_b, 1e-5f);
    ggml_tensor* m  = la::linear(ctx, w.fc0_w, mn, w.fc0_b);
    m = la::gelu_tanh(ctx, m);
    m = la::linear(ctx, w.fc1_w, m, w.fc1_b);
    x = ggml_add(ctx, x, m);
    return x;
}

// Builds patch_embed + bicubic pos-emb in-graph and returns the [1152,1024]
// (hidden,token) tensor. pos_tok is filled host-side and kept alive via `pool`.
ggml_tensor* VitEncoder::build_patch_pos(ggml_context* ctx, GraphInputPool& pool,
                                         const std::vector<float>& pixel_host,
                                         int gh, int gw,
                                         std::vector<float>& pos_tok_keep) {
    ggml_tensor* W   = ml_.tensor("vit.patch_embed.weight"); // ne=[14,14,3,1152]
    ggml_tensor* b   = ml_.tensor("vit.patch_embed.bias");   // ne=[1152]
    ggml_tensor* pe  = ml_.tensor("vit.pos_emb.weight");     // ne=[1152,64,64]
    if (!W || !b || !pe) return nullptr;

    const int c    = 1152;
    const int base = 64;
    const int n_tok = gh * gw;          // real image patch grid
    const int patch_dim = 14 * 14 * 3;  // 588

    std::vector<float> pe_src((size_t)ggml_nelements(pe));
    std::memcpy(pe_src.data(), pe->data, ggml_nbytes(pe));
    pos_tok_keep = bicubic_pos_emb(pe_src, base, base, c, gh, gw); // [n_tok,1152]

    ggml_tensor* W2d = ggml_reshape_2d(ctx, W, patch_dim, c);
    const int64_t ne_px[2] = { patch_dim, n_tok };
    ggml_tensor* px = be_.add_graph_input_nd(ctx, pool, pixel_host.data(), ne_px, 2);
    ggml_tensor* emb = la::linear(ctx, W2d, px, b);    // [1152,n_tok]
    const int64_t ne_pos[2] = { c, n_tok };
    ggml_tensor* pos = be_.add_graph_input_nd(ctx, pool, pos_tok_keep.data(), ne_pos, 2);
    return ggml_add(ctx, emb, pos);                    // [1152,n_tok]
}

bool VitEncoder::patch_and_pos(const std::vector<float>& pixel_host, int gh, int gw,
                               std::vector<float>& out) {
    GraphInputPool pool;
    std::vector<float> pos_keep;
    return be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        return build_patch_pos(ctx, pool, pixel_host, gh, gw, pos_keep);
    }, out);
}

bool VitEncoder::block0(const std::vector<float>& pixel_host, int gh, int gw,
                        std::vector<float>& out) {
    const auto& cfg = ml_.config();
    const int heads = (int)cfg.vit_n_heads;
    const int hd    = (int)cfg.vit_head_dim;
    la::RopeTables rt = la::build_rope_tables(gh, gw, heads, hd, cfg.vit_rope_theta);
    VitLayerWeights w = load_layer(ml_, 0);

    GraphInputPool pool;
    std::vector<float> pos_keep;
    return be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = build_patch_pos(ctx, pool, pixel_host, gh, gw, pos_keep); // [1152,n_tok]
        if (!x) return nullptr;
        ggml_tensor* cosb; ggml_tensor* sinb;
        la::build_rope_inputs(ctx, be_, pool, rt, cosb, sinb);
        return build_block(ctx, x, w, cosb, sinb, rt, heads, hd);
    }, out);
}

bool VitEncoder::forward(const std::vector<float>& pixel_host, int gh, int gw,
                         std::vector<float>& vit_final_out,
                         const std::vector<int>& capture_layers,
                         std::vector<std::vector<float>>& captured){
    const auto& cfg = ml_.config();
    const int heads = (int)cfg.vit_n_heads;
    const int hd    = (int)cfg.vit_head_dim;
    // Rope tables + per-layer weights built ONCE, reused across all blocks.
    la::RopeTables rt = la::build_rope_tables(gh, gw, heads, hd, cfg.vit_rope_theta);
    GraphInputPool pool;
    std::vector<float> pos_keep;                 // host bicubic pos-emb, kept alive
    captured.assign(capture_layers.size(), {});
    return be_.forward_capture([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = build_patch_pos(ctx, pool, pixel_host, gh, gw, pos_keep);  // [1152,n_tok]
        if (!x) return nullptr;
        // cos/sin rope inputs built ONCE before the loop and threaded into every block.
        ggml_tensor* cosb; ggml_tensor* sinb;
        la::build_rope_inputs(ctx, be_, pool, rt, cosb, sinb);
        for (int i = 0; i < (int)cfg.vit_n_layers; ++i){
            VitLayerWeights w = load_layer(ml_, i);
            x = build_block(ctx, x, w, cosb, sinb, rt, heads, hd);
            for (size_t c = 0; c < capture_layers.size(); ++c)
                if (capture_layers[c] == i) be_.capture(x, &captured[c]);
        }
        x = la::layernorm(ctx, x, ml_.tensor("vit.final_norm.weight"),
                          ml_.tensor("vit.final_norm.bias"), 1e-5f);
        return x;
    }, vit_final_out);
}

// 2x2 patch merger (pure host reindex). See declaration in vit_encoder.hpp.
std::vector<float> merge_patches(const std::vector<float>& vf, int gh, int gw, int c){
    int mh=gh/2, mw=gw/2; int C4=c*4;
    std::vector<float> out((size_t)mh*mw*C4);
    for (int a=0;a<mh;++a) for (int cc=0;cc<mw;++cc){
        int m = a*mw + cc;
        for (int b=0;b<2;++b) for (int e=0;e<2;++e){
            int row=2*a+b, col=2*cc+e, e_idx=b*2+e;
            int tok = row*gw + col;
            for (int d=0; d<c; ++d)
                out[(size_t)m*C4 + e_idx*c + d] = vf[(size_t)tok*c + d];
        }
    }
    return out;
}

}  // namespace la
