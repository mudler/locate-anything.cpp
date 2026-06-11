#include "mtp.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

// Pure C++ port of the box-frame decode heuristics from
// models/LocateAnything-3B/generate_utils.py (sample_tokens / is_valid_box_frame /
// decode_bbox_avg / decode_ref / handle_pattern). No ggml; thresholds operate on softmax.

namespace la::mtp {

// ---- token id constants (generate_utils.py get_token_ids_from_config defaults) ----
static constexpr int BOX_START   = 151668;
static constexpr int BOX_END     = 151669;
static constexpr int COORD_START = 151677;
static constexpr int COORD_END   = 152677;
static constexpr int REF_START   = 151672;
static constexpr int REF_END     = 151673;
static constexpr int NONE        = 4064;
static constexpr int NULL_TOK    = 152678;
static constexpr int TEXT_MASK   = 151676;   // default_mask_token_id (unused in decode, kept for ref)
static constexpr int IM_END      = 151645;

// numeric-stable softmax of one logits row -> probs row.
static std::vector<float> softmax_row(const std::vector<float>& logits) {
    std::vector<float> out(logits.size());
    float m = -INFINITY;
    for (float v : logits) m = std::max(m, v);
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        double e = std::exp((double)logits[i] - (double)m);
        out[i] = (float)e;
        sum += e;
    }
    float inv = (float)(1.0 / sum);
    for (float& v : out) v *= inv;
    return out;
}

// torch.topk(row, k) semantics: top-k by descending prob. Tie-break by ascending index
// (matches torch CPU stable ordering). Returns (ids, probs) of length k, prob-desc.
static void topk(const std::vector<float>& row, int k,
                 std::vector<int>& ids, std::vector<float>& probs) {
    int n = (int)row.size();
    k = std::min(k, n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) {
            if (row[a] != row[b]) return row[a] > row[b];
            return a < b;
        });
    ids.resize(k); probs.resize(k);
    for (int i = 0; i < k; ++i) { ids[i] = idx[i]; probs[i] = row[idx[i]]; }
}

// is_valid_box_frame (L246-273), invoked with start_thresh=0.7, end_thresh=0.2.
static std::string is_valid_box_frame(const std::vector<std::vector<float>>& probs,
                                      float start_thresh, float end_thresh) {
    float p_start = probs[0][BOX_START];
    if (p_start >= start_thresh) {
        if (probs[1][NONE]    > 0.2f &&
            probs[2][BOX_END] > 0.2f &&
            probs[3][NULL_TOK] > 0.1f &&
            probs[4][NULL_TOK] > 0.1f) {
            return "empty_box";
        }
    }
    float end_score = probs[5][BOX_END] + probs[5][NULL_TOK] + probs[5][IM_END];
    if (end_score >= end_thresh) return "legal_box";
    return "illegal_box";
}

// decode_bbox_avg (L276-361). keep_k=4. Returns box 6-tuple or empty {} for None.
// fast=false -> hybrid abnormal-coord->0 rule; fast=true -> final_coords = first_valid_ids (L354).
static std::vector<int32_t> decode_bbox_avg(const std::vector<std::vector<float>>& probs,
                                            int keep_k, bool& is_none, bool fast) {
    is_none = false;
    std::string box_type = is_valid_box_frame(probs, 0.7f, 0.2f);
    if (box_type == "empty_box") {
        return { BOX_START, NONE, BOX_END, NULL_TOK, NULL_TOK, NULL_TOK };
    }
    if (box_type == "illegal_box") { is_none = true; return {}; }

    // positions 1..4: top-k, require >=1 coord token per position.
    std::vector<int32_t> coords(4, 0);
    for (int p = 0; p < 4; ++p) {
        std::vector<int> ids; std::vector<float> pr;
        topk(probs[1 + p], keep_k, ids, pr);
        // first valid (highest-prob coord) + valid stats
        int first_valid_idx = -1;
        int valid_counts = 0;
        int valid_max = -999999, valid_min = 999999;
        for (int i = 0; i < (int)ids.size(); ++i) {
            if (ids[i] >= COORD_START && ids[i] <= COORD_END) {
                if (first_valid_idx < 0) first_valid_idx = i;
                ++valid_counts;
                valid_max = std::max(valid_max, ids[i]);
                valid_min = std::min(valid_min, ids[i]);
            }
        }
        if (first_valid_idx < 0) { is_none = true; return {}; } // not a box
        float first_valid_prob = pr[first_valid_idx];
        int   first_valid_id   = ids[first_valid_idx];
        if (fast) {
            coords[p] = first_valid_id;   // fast: no abnormal zeroing (L355)
        } else {
            // hybrid abnormal rule (L349-353)
            bool is_abnormal = (first_valid_prob < 0.9f) && (valid_counts > 1) &&
                               ((valid_max - valid_min) > 60);
            coords[p] = is_abnormal ? 0 : first_valid_id;
        }
    }
    return { BOX_START, coords[0], coords[1], coords[2], coords[3], BOX_END };
}

// decode_ref (L364-405). NOTE: called from sample_tokens with defaults -> keep_k=5,
// start_thresh=0.6 (NOT the keep_k=4 of decode_bbox_avg). Returns ref seq or empty {} for None.
static std::vector<int32_t> decode_ref(const std::vector<std::vector<float>>& probs,
                                       bool& is_none, int keep_k = 5, float start_thresh = 0.6f) {
    is_none = false;
    if (probs[0][REF_START] < start_thresh) { is_none = true; return {}; }
    int L = (int)probs.size();
    std::vector<int32_t> out;
    out.push_back(REF_START);
    // positions 1..L-1: highest-prob NON-coord token in top-k.
    for (int p = 1; p < L; ++p) {
        std::vector<int> ids; std::vector<float> pr;
        topk(probs[p], keep_k, ids, pr);
        int first_valid_idx = -1;
        for (int i = 0; i < (int)ids.size(); ++i) {
            bool is_coord = (ids[i] >= COORD_START && ids[i] <= COORD_END);
            if (!is_coord) { first_valid_idx = i; break; }
        }
        if (first_valid_idx < 0) { is_none = true; return {}; }
        out.push_back(ids[first_valid_idx]);
    }
    return out;
}

// sample_box = decode_bbox_avg -> if None, decode_ref -> if None, {0}.
std::vector<int32_t> sample_box(const std::vector<std::vector<float>>& probs6, int keep_k, bool fast) {
    bool none1 = false;
    std::vector<int32_t> box = decode_bbox_avg(probs6, keep_k, none1, fast);
    if (!none1) return box;
    bool none2 = false;
    std::vector<int32_t> ref = decode_ref(probs6, none2);
    if (!none2) return ref;
    return { 0 };
}

// select_new_tokens: softmax the 6 rows, run sample_box; if it returns the {0} fallback,
// replace with per-position argmax of the logits (= sample_tokens x0). Mirrors
// _sample_token_in_mtp's `is_box_empty ? x0 : box_avg`.
std::vector<int32_t> select_new_tokens(const std::vector<std::vector<float>>& logits6, int keep_k, bool fast) {
    std::vector<std::vector<float>> probs6(logits6.size());
    for (size_t p = 0; p < logits6.size(); ++p) probs6[p] = softmax_row(logits6[p]);

    std::vector<int32_t> box = sample_box(probs6, keep_k, fast);
    if (box.size() == 1 && box[0] == 0) {
        // fallback -> per-position argmax of logits (argmax6)
        std::vector<int32_t> argmax6(logits6.size());
        for (size_t p = 0; p < logits6.size(); ++p) {
            const auto& row = logits6[p];
            int best = 0; float bv = row[0];
            for (int t = 1; t < (int)row.size(); ++t) if (row[t] > bv) { bv = row[t]; best = t; }
            argmax6[p] = best;
        }
        return argmax6;
    }
    return box;
}

// handle_pattern (L408-504). need_ar = need_switch_to_ar, terminal = is_terminal.
// fast=true: a malformed box is a coord_box (full x0, no AR switch) instead of error_box (L473).
Pattern handle_pattern(const std::vector<int32_t>& x0, bool fast) {
    Pattern out;
    // Defensive: handle_pattern indexes x0[1..5] in the box branch. select_new_tokens
    // always yields 6 tokens, but guard against shorter input (empty -> terminal im_end).
    if (x0.empty()) {
        out.type = "im_end"; out.tokens = { IM_END };
        out.terminal = true; out.need_ar = false; return out;
    }
    if (x0[0] == NULL_TOK || x0[0] == IM_END) {
        out.type = "im_end"; out.tokens = { IM_END };
        out.terminal = true; out.need_ar = false; return out;
    }
    if (x0.size() >= 2 && x0[0] == BOX_START && x0[1] == NONE) {
        out.type = "empty_box"; out.tokens = { BOX_START, NONE, BOX_END };
        out.terminal = false; out.need_ar = false; return out;
    }
    if (x0[0] == BOX_START) {
        const int n = (int)x0.size();
        int coord_ix = 1;
        for (int i = 1; i < 5 && i < n; ++i) {
            int c = x0[i];
            if (c >= COORD_START && c <= COORD_END) ++coord_ix;
            else break;
        }
        if (coord_ix == 5 && n >= 6 && x0[5] == BOX_END) {
            out.type = "coord_box"; out.tokens = x0;
            out.terminal = false; out.need_ar = false; return out;
        } else if (coord_ix == 3 && n >= 4 && x0[3] == BOX_END) {
            out.type = "point_box";
            out.tokens.assign(x0.begin(), x0.begin() + 4);
            out.terminal = false; out.need_ar = false; return out;
        } else if (fast) {
            // fast mode: treat as coord_box, stay in MTP (full x0, no AR switch) (L473-480)
            out.type = "coord_box"; out.tokens = x0;
            out.terminal = false; out.need_ar = false; return out;
        } else {
            // hybrid mode: error_box, switch to AR (L481-488)
            out.type = "error_box";
            out.tokens.assign(x0.begin(), x0.begin() + coord_ix);
            out.terminal = false; out.need_ar = true; return out;
        }
    }
    // ref_object: truncate at first null, dedup trailing ref_end pair.
    std::vector<int32_t> toks = x0;
    for (size_t i = 0; i < toks.size(); ++i) {
        if (toks[i] == NULL_TOK) { toks.resize(i); break; }
    }
    if (toks.size() >= 2 && toks[toks.size() - 1] == REF_END &&
        toks[toks.size() - 2] == REF_END) {
        toks.pop_back();
    }
    out.type = "ref_object"; out.tokens = toks;
    out.terminal = false; out.need_ar = false; return out;
}

// hybrid MTP round: _sample_token_in_mtp + the generate-loop switch (L489-508).
// new_tokens = the chosen block (select_new_tokens / box_avg-or-x0). handle_pattern -> out_type;
// committed tokens = out_pattern.tokens. im_end terminates; error_box drops to AR (use_mtp=false).
StepOut hybrid_mtp_step(HybridState& st, const std::vector<int32_t>& new_tokens, bool fast) {
    Pattern p = handle_pattern(new_tokens, fast);
    StepOut o; o.out_type = p.type; o.committed = p.tokens;
    if (p.terminal) st.terminated = true;           // out_type == 'im_end' -> break
    else if (p.type == "error_box") st.use_mtp = false;  // hybrid: switch_to_ar (never in fast)
    return o;
}

// hybrid AR round: _sample_token_in_ar (L430-460) + the generate-loop switch.
// out_token = x0[0] (the RAW sampled token) in EVERY branch; only out_type differs.
// box_end -> box_end_ar (use_mtp=true, back to MTP); coord/none -> coord_ar (stay AR);
// else -> im_end (committed raw token, then break / terminated).
StepOut hybrid_ar_step(HybridState& st, int32_t t) {
    StepOut o;
    o.committed = { t };  // out_token = x0[0] regardless of classification
    if (t == BOX_END) {
        o.out_type = "box_end_ar"; st.use_mtp = true;
    } else if ((t >= COORD_START && t <= COORD_END) || t == NONE) {
        o.out_type = "coord_ar"; // stays AR
    } else {
        o.out_type = "im_end"; st.terminated = true;
    }
    return o;
}

} // namespace la::mtp
