#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "model_loader.hpp"
#include "backend.hpp"   // for la::Backend
#include <vector>
namespace la {
struct Qwen2Hparams {
    int hidden=2048, n_heads=16, n_kv_heads=2, head_dim=128, intermediate=11008;
    float rope_theta=1.0e6f, rms_eps=1.0e-6f;
};
struct Qwen2LayerWeights {
    ggml_tensor *attn_norm, *attn_q,*attn_q_b, *attn_k,*attn_k_b, *attn_v,*attn_v_b,
                *attn_o, *ffn_norm, *ffn_gate, *ffn_up, *ffn_down;
};
Qwen2LayerWeights load_qwen2_layer(const ModelLoader& ml, int i);
// Eager, no-cache single-layer forward. x:[hidden,seq]; pos: int32 [seq]; mask: f32 [seq,seq].
ggml_tensor* qwen2_layer_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
                                 ggml_tensor* mask, const Qwen2LayerWeights& w,
                                 const Qwen2Hparams& hp);

// Persistent per-layer K/V cache. Tensor metadata lives in `ctx`; tensor data
// lives in `buffer` (allocated on the compute backend via Backend). Both freed
// in free(). Used for resident-cache decode (no per-step host round-trip).
struct ResidentKV {
    ggml_context*                    ctx    = nullptr;
    ggml_backend_buffer_t            buffer = nullptr;
    std::vector<ggml_tensor*>        k, v;   // each [head_dim, n_kv, max_seq, 1]
    int max_seq = 0, past_len = 0;
    ResidentKV() = default;
    // Owns ctx+buffer — non-copyable and non-movable (use in place / via pointer)
    // to avoid double-free of the persistent backend buffer.
    ResidentKV(const ResidentKV&) = delete;
    ResidentKV& operator=(const ResidentKV&) = delete;
    bool init(Backend& be, int n_layers, int hd, int n_kv, int max_seq);
    void free();
    ~ResidentKV();
};

struct Qwen2ResidentOut { ggml_tensor* y; ggml_tensor* k_write; ggml_tensor* v_write; };
// Cache-aware layer. x:[hidden,n_new]; pos: i32 [n_new] (absolute); mask: f32 [past_len+n_new, n_new].
// Writes new K/V into kv.k[layer_idx]/kv.v[layer_idx] at offset kv.past_len; attends over
// the full [0..past_len+n_new). Returns y + the two cpy nodes (caller MUST graph-expand them).
Qwen2ResidentOut qwen2_layer_forward_resident(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
        ggml_tensor* mask, ResidentKV& kv, int layer_idx,
        const Qwen2LayerWeights& w, const Qwen2Hparams& hp);
}
