#pragma once
#include "ggml.h"
#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>

struct ggml_backend_buffer;

namespace la {

class Backend;  // backend.hpp

struct VitConfig {
    uint32_t vit_hidden = 0, vit_n_layers = 0, vit_n_heads = 0, vit_head_dim = 0,
             vit_intermediate = 0, vit_patch = 0, vit_pos_emb_hw = 0;
    std::vector<int32_t> vit_merge;       // [2,2]
    float    vit_rope_theta = 10000.f;

    // --- language model (Qwen2) ---
    uint32_t lm_hidden = 0, lm_n_layers = 0, lm_n_heads = 0, lm_n_kv_heads = 0,
             lm_head_dim = 0, lm_intermediate = 0, lm_vocab = 0;
    float    lm_rope_theta = 1.0e6f, lm_rms_eps = 1.0e-6f;
    uint32_t image_token_id = 151665;
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    // Owns gguf_/ctx_ — non-copyable to avoid double-free; movable.
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    ModelLoader(ModelLoader&&) = default;
    ModelLoader& operator=(ModelLoader&&) = default;

    bool load(const std::string& path);
    const VitConfig& config() const { return cfg_; }
    // Returns a tensor owned by this loader's ctx_; valid only while the loader
    // is alive. nullptr if absent.
    ggml_tensor* tensor(const std::string& name) const;
    // Read a string-array KV (e.g. tokenizer tokens/merges). false if absent/wrong type.
    bool kv_str_array(const std::string& key, std::vector<std::string>& out) const;
    // Read an int32-array KV (e.g. tokenizer token_types). false if absent/wrong type.
    bool kv_i32_array(const std::string& key, std::vector<int32_t>& out) const;

    // Move the graph-weight tensors onto the backend's device when `be` is
    // offloading (GPU). No-op (returns true) on a CPU backend: graphs keep
    // referencing the gguf host tensors directly (zero-copy, byte-identical).
    //
    // On GPU: every gguf tensor EXCEPT the two host-read tensors
    // ("lm.tok_embd.weight", "vit.pos_emb.weight") is mirrored into a no_alloc
    // device context, allocated on the backend, and uploaded; tensors_[name] is
    // repointed at the device tensor. The two excluded tensors stay in ctx_
    // (host memory) because they are read directly via ->data on the host
    // (LMForward::embed_tokens_host, VitEncoder::build_patch_pos). ctx_ is kept
    // alive (it holds the host source bytes and the two host-read tensors).
    bool offload_weights(Backend& be);
private:
    VitConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_  = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
    // GPU offload (null on the CPU path): device_ctx_ holds the device tensor
    // metadata (no_alloc), gpu_buf_ owns the device buffer with the weights.
    ggml_context* device_ctx_ = nullptr;
    ggml_backend_buffer* gpu_buf_ = nullptr;
};

}  // namespace la
