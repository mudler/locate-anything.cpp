#include "qwen2.hpp"
#include "ggml_extend.hpp"
#include <cmath>
#include <string>
namespace la {
Qwen2LayerWeights load_qwen2_layer(const ModelLoader& ml, int i){
    auto t=[&](const std::string& s){ return ml.tensor("lm.blk."+std::to_string(i)+"."+s); };
    return Qwen2LayerWeights{
        t("attn_norm.weight"),
        t("attn_q.weight"), t("attn_q.bias"),
        t("attn_k.weight"), t("attn_k.bias"),
        t("attn_v.weight"), t("attn_v.bias"),
        t("attn_o.weight"),
        t("ffn_norm.weight"),
        t("ffn_gate.weight"), t("ffn_up.weight"), t("ffn_down.weight")};
}
ggml_tensor* qwen2_layer_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
                                 ggml_tensor* mask, const Qwen2LayerWeights& w,
                                 const Qwen2Hparams& hp){
    const int hd=hp.head_dim, n_h=hp.n_heads, n_kv=hp.n_kv_heads;
    const int seq=(int)x->ne[1];
    ggml_tensor* xn = la::rms_norm(ctx, x, w.attn_norm, hp.rms_eps);
    ggml_tensor* q = la::linear(ctx, w.attn_q, xn, w.attn_q_b);   // [2048,seq]
    ggml_tensor* k = la::linear(ctx, w.attn_k, xn, w.attn_k_b);   // [256,seq]
    ggml_tensor* v = la::linear(ctx, w.attn_v, xn, w.attn_v_b);   // [256,seq]
    q = ggml_reshape_3d(ctx, q, hd, n_h,  seq);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, seq);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, seq);
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor* qp = ggml_permute(ctx, q, 0,2,1,3);             // [hd,seq,n_h]
    ggml_tensor* kp = ggml_permute(ctx, k, 0,2,1,3);             // [hd,seq,n_kv]
    ggml_tensor* vp = ggml_permute(ctx, v, 0,2,1,3);
    ggml_tensor* scores = ggml_mul_mat(ctx, kp, qp);            // [seq_kv,seq_q,n_h] (GQA broadcast)
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f/std::sqrt((float)hd), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vp));  // [seq_kv,hd,n_kv]
    ggml_tensor* o  = ggml_mul_mat(ctx, vt, scores);           // [hd,seq_q,n_h]
    o = ggml_permute(ctx, o, 0,2,1,3);                          // [hd,n_h,seq]
    o = ggml_cont_2d(ctx, o, hd*n_h, seq);                     // [2048,seq]
    o = ggml_mul_mat(ctx, w.attn_o, o);                        // no bias
    ggml_tensor* h = ggml_add(ctx, x, o);
    ggml_tensor* hn = la::rms_norm(ctx, h, w.ffn_norm, hp.rms_eps);
    ggml_tensor* g = ggml_mul_mat(ctx, w.ffn_gate, hn);
    ggml_tensor* u = ggml_mul_mat(ctx, w.ffn_up,   hn);
    ggml_tensor* gu = ggml_mul(ctx, ggml_silu(ctx, g), u);     // silu(gate)*up
    ggml_tensor* f = ggml_mul_mat(ctx, w.ffn_down, gu);
    return ggml_add(ctx, h, f);
}

// Cache-aware single-layer forward. Mirrors qwen2_layer_forward exactly through
// RoPE, then writes the new K/V into the resident buffer at offset past_len and
// reads back the full [0..past_len+n_new) range for attention. Returns y plus
// the two cpy nodes; the caller MUST graph-expand k_write/v_write (interleaved
// per layer, before the output) so the writes land before later reads.
Qwen2ResidentOut qwen2_layer_forward_resident(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos,
        ggml_tensor* mask, ResidentKV& kv, int layer_idx,
        const Qwen2LayerWeights& w, const Qwen2Hparams& hp){
    const int hd=hp.head_dim, n_h=hp.n_heads, n_kv=hp.n_kv_heads;
    const int n_new=(int)x->ne[1];
    const int full_len = kv.past_len + n_new;
    // Trap a cache-overflow / bad layer index loudly rather than silently
    // corrupting the persistent K/V buffer (matters once M5 pushes longer seqs).
    GGML_ASSERT(layer_idx >= 0 && layer_idx < (int)kv.k.size());
    GGML_ASSERT(full_len <= kv.max_seq);
    ggml_tensor* xn = la::rms_norm(ctx, x, w.attn_norm, hp.rms_eps);
    ggml_tensor* q = la::linear(ctx, w.attn_q, xn, w.attn_q_b);
    ggml_tensor* k = la::linear(ctx, w.attn_k, xn, w.attn_k_b);
    ggml_tensor* v = la::linear(ctx, w.attn_v, xn, w.attn_v_b);
    q = ggml_reshape_3d(ctx, q, hd, n_h,  n_new);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, n_new);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, n_new);
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    // write new k/v into resident buffer at offset past_len (ne[2] = sequence dim)
    ggml_tensor* kr = kv.k[layer_idx]; ggml_tensor* vr = kv.v[layer_idx];
    const size_t koff = (size_t)kv.past_len * kr->nb[2];
    const size_t voff = (size_t)kv.past_len * vr->nb[2];
    ggml_tensor* k_dst = ggml_view_4d(ctx, kr, hd, n_kv, n_new, 1, kr->nb[1],kr->nb[2],kr->nb[3], koff);
    ggml_tensor* v_dst = ggml_view_4d(ctx, vr, hd, n_kv, n_new, 1, vr->nb[1],vr->nb[2],vr->nb[3], voff);
    // k/v are [hd,n_kv,n_new]; reshape to [hd,n_kv,n_new,1] to match the 4d dst
    ggml_tensor* k4 = ggml_reshape_4d(ctx, k, hd, n_kv, n_new, 1);
    ggml_tensor* v4 = ggml_reshape_4d(ctx, v, hd, n_kv, n_new, 1);
    ggml_tensor* k_write = ggml_cpy(ctx, k4, k_dst);
    ggml_tensor* v_write = ggml_cpy(ctx, v4, v_dst);
    // read back full [0..full_len)
    ggml_tensor* k_full = ggml_view_4d(ctx, kr, hd, n_kv, full_len, 1, kr->nb[1],kr->nb[2],kr->nb[3], 0);
    ggml_tensor* v_full = ggml_view_4d(ctx, vr, hd, n_kv, full_len, 1, vr->nb[1],vr->nb[2],vr->nb[3], 0);
    k_full = ggml_reshape_3d(ctx, k_full, hd, n_kv, full_len);
    v_full = ggml_reshape_3d(ctx, v_full, hd, n_kv, full_len);
    // attention (eager, GQA implicit broadcast, scale 1/sqrt(hd), mask [full_len, n_new])
    ggml_tensor* qp = ggml_permute(ctx, q,      0,2,1,3);   // [hd,n_new,n_h]
    ggml_tensor* kp = ggml_permute(ctx, k_full, 0,2,1,3);   // [hd,full,n_kv]
    ggml_tensor* vp = ggml_permute(ctx, v_full, 0,2,1,3);
    ggml_tensor* scores = ggml_mul_mat(ctx, kp, qp);        // [full, n_new, n_h]
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f/std::sqrt((float)hd), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vp));  // [full, hd, n_kv]
    ggml_tensor* o  = ggml_mul_mat(ctx, vt, scores);        // [hd, n_new, n_h]
    o = ggml_permute(ctx, o, 0,2,1,3);
    o = ggml_cont_2d(ctx, o, hd*n_h, n_new);
    o = ggml_mul_mat(ctx, w.attn_o, o);
    ggml_tensor* h = ggml_add(ctx, x, o);
    ggml_tensor* hn = la::rms_norm(ctx, h, w.ffn_norm, hp.rms_eps);
    ggml_tensor* g = ggml_mul_mat(ctx, w.ffn_gate, hn);
    ggml_tensor* u = ggml_mul_mat(ctx, w.ffn_up,   hn);
    ggml_tensor* gu = ggml_mul(ctx, ggml_silu(ctx, g), u);
    ggml_tensor* f = ggml_mul_mat(ctx, w.ffn_down, gu);
    return { ggml_add(ctx, h, f), k_write, v_write };
}

// ResidentKV ----------------------------------------------------------------
bool ResidentKV::init(Backend& be, int n_layers, int hd, int n_kv, int max_seq_in){
    free();
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * (size_t)(2*n_layers + 16);
    p.no_alloc = true;
    ctx = ggml_init(p);
    if(!ctx) return false;
    k.assign(n_layers, nullptr); v.assign(n_layers, nullptr);
    for(int i=0;i<n_layers;++i){
        k[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, n_kv, max_seq_in, 1);
        v[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, n_kv, max_seq_in, 1);
        if(!k[i]||!v[i]){ free(); return false; }
    }
    buffer = be.allocate_ctx_tensors(ctx);
    if(!buffer){ free(); return false; }
    max_seq = max_seq_in; past_len = 0;
    return true;
}
void ResidentKV::free(){
    if(buffer){ ggml_backend_buffer_free(buffer); buffer=nullptr; }
    if(ctx){ ggml_free(ctx); ctx=nullptr; }
    k.clear(); v.clear(); max_seq=0; past_len=0;
}
ResidentKV::~ResidentKV(){ free(); }
}
