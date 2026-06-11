#pragma once
#include <vector>
#include <string>
#include <cstdint>
namespace la::mtp {
// `fast` selects generation_mode='fast' (MTP-only) decode semantics vs 'hybrid' (default):
//   * decode_bbox_avg: fast skips the abnormal-coord->0 zeroing (final_coords = first_valid_ids);
//   * handle_pattern:  fast treats a malformed box as a coord_box (full 6 tokens, no AR switch).
// (generate_utils.py L354/L473.) Both share keep_k=4; the forward is mode-independent.

// probs6: [6][vocab] (softmax of the 6 block-position logits). logits6 same shape (for argmax fallback).
// Returns the chosen block tokens: a box 6-tuple, the empty-box 6-tuple, a ref sequence,
// the {0} fallback, OR (when fallback) the 6 argmax ids. keep_k=4.
std::vector<int32_t> sample_box(const std::vector<std::vector<float>>& probs6, int keep_k=4, bool fast=false);
// Full MTP block-token selection incl. the argmax fallback (= reference new_tokens/x0).
std::vector<int32_t> select_new_tokens(const std::vector<std::vector<float>>& logits6, int keep_k=4, bool fast=false);
struct Pattern { std::string type; std::vector<int32_t> tokens; bool terminal=false; bool need_ar=false; };
Pattern handle_pattern(const std::vector<int32_t>& x0, bool fast=false);

// ---- hybrid MTP<->AR control-flow state machine (modeling_locateanything.py generate loop) ----
struct HybridState { bool use_mtp = true; bool terminated = false; };
struct StepOut { std::vector<int32_t> committed; std::string out_type; };
// MTP round: classify new_tokens via handle_pattern, commit out_pattern.tokens, update state
// (im_end -> terminated; error_box -> use_mtp=false / drop to AR). In fast mode handle_pattern
// never yields error_box, so use_mtp stays true (AR never runs) -- matching the upstream loop.
StepOut hybrid_mtp_step(HybridState& st, const std::vector<int32_t>& new_tokens, bool fast=false);
// AR round (hybrid): classify the single AR token. Commits the RAW token in every case
// (matches _sample_token_in_ar: out_token = x0[0] for box_end_ar/coord_ar/im_end alike).
// box_end -> box_end_ar (use_mtp=true / back to MTP); coord/none -> coord_ar (stay AR);
// else -> im_end (terminated).
StepOut hybrid_ar_step(HybridState& st, int32_t ar_token);
}
