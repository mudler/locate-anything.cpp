#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "projector.hpp"
#include "qwen2.hpp"   // ResidentKV
#include <vector>
#include <cstdint>
namespace la {
class LMForward {
public:
    LMForward(ModelLoader& ml, Backend& be) : ml_(ml), be_(be), proj_(ml, be) {}
    // ids: 297 token ids. projected_host: [2048*256] already-projected vision tokens
    // (ggml [2048,256] flat = [token,hidden]). Returns embeds_after_splice [2048*297]
    // = ggml [2048,297] flat = [token,hidden].
    bool embed_and_splice(const std::vector<int32_t>& ids,
                          const std::vector<float>& projected_host,
                          std::vector<float>& out);
    // Run the REFERENCE embeds (or any [hidden*seq] host tensor) through Qwen2 layer 0.
    bool run_layer0(const std::vector<float>& embeds_host, int seq, std::vector<float>& out);
    // Full causal prefill. ids:297, projected_host:[2048*256]. Returns logits at the
    // LAST position [vocab]. capture_layers: decoder-layer indices whose output to read back.
    bool forward(const std::vector<int32_t>& ids,
                 const std::vector<float>& projected_host,
                 std::vector<float>& logits_last,
                 const std::vector<int>& capture_layers,
                 std::vector<std::vector<float>>& captured);
    // Greedy AR decode via full reprefill each step (correctness oracle; O(n^3)).
    // prompt_ids: the 297 prompt tokens. projected_host: [2048*256] vision tokens.
    // Appends generated ids to out_ids until EOS (151645) or max_new tokens.
    bool decode_greedy_reprefill(const std::vector<int32_t>& prompt_ids,
                                 const std::vector<float>& projected_host,
                                 int max_new, std::vector<int32_t>& out_ids);
    // Greedy AR decode with the resident KV-cache (fast, incremental). Same output as
    // decode_greedy_reprefill but O(n^2). Appends generated ids until EOS or max_new.
    bool decode_greedy_resident(const std::vector<int32_t>& prompt_ids,
                                const std::vector<float>& projected_host,
                                int max_new, std::vector<int32_t>& out_ids);

    // --- Reusable resident-decode helpers -------------------------------------
    // Host row-gather from lm.tok_embd: out = [H*ids.size()] flat [token,hidden].
    void embed_tokens_host(const std::vector<int32_t>& ids, std::vector<float>& out);
    // Full causal prefill into the resident cache: embed+splice the prompt, run all
    // layers resident at past_len=0, set kv.past_len=prompt_len. If logits_last!=null,
    // returns the LAST-position logits [vocab] (final norm + lm.output).
    bool prefill_resident(const std::vector<int32_t>& prompt_ids,
                          const std::vector<float>& projected_host,
                          ResidentKV& kv,
                          std::vector<float>* logits_last = nullptr);

    // --- MTP block-diffusion (one round) --------------------------------------
    // Build the block-diffusion mask [full_len*n_new] (key-major, m[q*full_len+key])
    // and positions [n_new] for an MTP round. cached_len = kv.past_len; n_recompute
    // uncached committed tokens; block=6.
    // (See mask_sdpa_utils.update_causal_mask_for_one_gen_window_2d.)
    void build_mtp_mask(int cached_len, int n_recompute, std::vector<float>& mask_out);     // [full*n_new]
    void build_mtp_positions(int cached_len, int n_recompute, std::vector<int32_t>& pos_out); // [n_new]
    // Run one MTP block forward through the resident layers; returns the 6 block-position
    // logits [6*vocab] (after final norm + lm.output), position-major (logits6_out[p*vocab+t]).
    // kv.past_len must be set by the CALLER (= cached_len); does NOT advance kv.past_len.
    bool mtp_block_forward(ResidentKV& kv, const std::vector<float>& x_host, int n_new,
                           int n_recompute, std::vector<float>& logits6_out);

    // Parallel Box Decoding (MTP block_size=6). fast=false -> generation_mode="hybrid"
    // (MTP with AR fallback on malformed boxes); fast=true -> generation_mode="fast"
    // (MTP-only, no AR fallback). Returns the generated token ids matching that reference
    // stream. If captured_logits6 != null, each MTP round appends its 6-position logits
    // [6*vocab] (position-major) for per-round debug gating.
    // early_stop=true: stop as soon as the decode enters the degenerate repeated-box
    // tail (a coord box identical to the previous one) instead of running to max_new.
    // This is the engine/CLI default (clean, faster output); the parity gates pass
    // false to reproduce the exact reference token stream.
    bool decode_hybrid(const std::vector<int32_t>& prompt_ids,
                       const std::vector<float>& projected_host,
                       int max_new, std::vector<int32_t>& out_ids,
                       std::vector<std::vector<float>>* captured_logits6 = nullptr,
                       bool fast = false, bool early_stop = false);
private:
    // Generic resident chunk with a plain absolute-position causal mask. x_host:[H*n_new],
    // written at KV offset pos0; returns LAST-position logits [vocab]. Advances kv.past_len
    // to pos0+n_new on success.
    bool run_resident_causal(const std::vector<float>& x_host, int n_new, int pos0,
                             ResidentKV& kv, std::vector<float>& logits);
    ModelLoader& ml_; Backend& be_; Projector proj_;
};
}
