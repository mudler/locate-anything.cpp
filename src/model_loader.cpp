#include "model_loader.hpp"
#include "common.hpp"
#include "la_gguf_keys.h"
#include "backend.hpp"

#include "ggml-backend.h"

namespace la {

// Weights read directly on the host via ->data (NOT graph nodes): they produce
// host-computed graph INPUTS and so must stay in CPU/host-accessible memory.
//   - "lm.tok_embd.weight": host row-gather in LMForward::embed_tokens_host.
//   - "vit.pos_emb.weight": host bicubic in VitEncoder::build_patch_pos.
static bool is_host_read_tensor(const std::string& name) {
    return name == "lm.tok_embd.weight" || name == "vit.pos_emb.weight";
}

static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_u32(g,id);
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out; int64_t id = gguf_find_key(g,k);
    if (id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}

bool ModelLoader::load(const std::string& path){
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) { LA_LOG("gguf_init_from_file failed: %s", path.c_str()); return false; }
    cfg_.vit_hidden       = kv_u32(gguf_, LA_KV_VIT_HIDDEN);
    cfg_.vit_n_layers     = kv_u32(gguf_, LA_KV_VIT_N_LAYERS);
    cfg_.vit_n_heads      = kv_u32(gguf_, LA_KV_VIT_N_HEADS);
    cfg_.vit_head_dim     = kv_u32(gguf_, LA_KV_VIT_HEAD_DIM);
    cfg_.vit_intermediate = kv_u32(gguf_, LA_KV_VIT_INTERMEDIATE);
    cfg_.vit_patch        = kv_u32(gguf_, LA_KV_VIT_PATCH);
    cfg_.vit_pos_emb_hw   = kv_u32(gguf_, LA_KV_VIT_POS_EMB_HW);
    cfg_.vit_merge        = kv_i32_arr(gguf_, LA_KV_VIT_MERGE);
    cfg_.vit_rope_theta   = kv_f32(gguf_, LA_KV_VIT_ROPE_THETA, 10000.f);
    cfg_.lm_hidden       = kv_u32(gguf_, LA_KV_LM_HIDDEN);
    cfg_.lm_n_layers     = kv_u32(gguf_, LA_KV_LM_N_LAYERS);
    cfg_.lm_n_heads      = kv_u32(gguf_, LA_KV_LM_N_HEADS);
    cfg_.lm_n_kv_heads   = kv_u32(gguf_, LA_KV_LM_N_KV_HEADS);
    cfg_.lm_head_dim     = kv_u32(gguf_, LA_KV_LM_HEAD_DIM);
    cfg_.lm_intermediate = kv_u32(gguf_, LA_KV_LM_INTERMEDIATE);
    cfg_.lm_vocab        = kv_u32(gguf_, LA_KV_LM_VOCAB);
    cfg_.lm_rope_theta   = kv_f32(gguf_, LA_KV_LM_ROPE_THETA, 1.0e6f);
    cfg_.lm_rms_eps      = kv_f32(gguf_, LA_KV_LM_RMS_EPS, 1.0e-6f);
    cfg_.image_token_id  = kv_u32(gguf_, LA_KV_TOK_IMAGE, 151665);
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for (int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if (t) tensors_[nm] = t;
    }
    return cfg_.vit_hidden>0 && cfg_.vit_n_layers>0;
}

ggml_tensor* ModelLoader::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it==tensors_.end() ? nullptr : it->second;
}

bool ModelLoader::kv_str_array(const std::string& key, std::vector<std::string>& out) const {
    if (!gguf_) return false;
    int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gguf_, id) != GGUF_TYPE_STRING) return false;
    size_t n = gguf_get_arr_n(gguf_, id);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = gguf_get_arr_str(gguf_, id, i);
    return true;
}

bool ModelLoader::kv_i32_array(const std::string& key, std::vector<int32_t>& out) const {
    if (!gguf_) return false;
    int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gguf_, id) != GGUF_TYPE_INT32) return false;
    size_t n = gguf_get_arr_n(gguf_, id);
    const int32_t* a = (const int32_t*)gguf_get_arr_data(gguf_, id);
    out.assign(a, a+n);
    return true;
}

bool ModelLoader::offload_weights(Backend& be){
    // CPU backend: no-op. Graphs keep referencing the gguf host tensors in ctx_
    // directly (zero-copy; the CPU path is byte-identical to before this change).
    if (!be.is_offloading()) return true;
    if (device_ctx_) return true;                       // idempotent
    ggml_backend_t backend = be.handle();
    if (!backend || !ctx_){ LA_LOG("offload_weights: null backend/ctx"); return false; }

    // Mirror every gguf tensor EXCEPT the two host-read tensors into a no_alloc
    // device context, allocate it on the backend, upload the bytes, and repoint
    // tensors_[name] at the device tensor. The host-read tensors stay pointing at
    // the original ctx_ (host) tensors so embed_tokens_host / build_patch_pos can
    // keep reading ->data on the host. ctx_ remains alive as the host source.
    const size_t n = tensors_.size();
    ggml_init_params dp{};
    dp.mem_size  = ggml_tensor_overhead() * (n + 8);
    dp.no_alloc  = true;
    device_ctx_ = ggml_init(dp);
    if (!device_ctx_){ LA_LOG("offload_weights: device ctx init failed"); return false; }

    std::vector<std::pair<ggml_tensor*, const void*>> ups; ups.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> newmap; newmap.reserve(n);
    size_t n_dev = 0;
    for (auto& kv : tensors_) {
        if (is_host_read_tensor(kv.first)) {
            newmap.emplace(kv.first, kv.second);        // keep host tensor as-is
            continue;
        }
        ggml_tensor* s = kv.second;
        ggml_tensor* d = ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        newmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);                   // host source bytes in ctx_
        ++n_dev;
    }
    gpu_buf_ = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if (!gpu_buf_){ LA_LOG("offload_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(newmap);   // graphs now reference the device-resident weights
    LA_LOG("offload_weights: %zu weights -> %s (2 host-read tensors kept on CPU)",
           n_dev, be.device_name().c_str());
    return true;
}

ModelLoader::~ModelLoader(){
    // Free the device buffer + its metadata ctx BEFORE the host ctx_ (which holds
    // the host-read tensors and the source bytes the device weights were copied from).
    if (gpu_buf_)    ggml_backend_buffer_free(gpu_buf_);
    if (device_ctx_) ggml_free(device_ctx_);
    if (gguf_) gguf_free(gguf_);
    if (ctx_)  ggml_free(ctx_);
}

}  // namespace la
