// Gate for generation_mode='fast' (MTP-only, no AR fallback). Two layers:
//  (1) synthetic unit checks of the fast decode branches (always run, hard gate):
//      handle_pattern(fast) and select_new_tokens(fast) must diverge from hybrid
//      exactly where generate_utils.py L473 / L354 say they do.
//  (2) end-to-end: decode_hybrid(fast=true) on the fixture must reproduce the
//      upstream fast token stream (dumps/reference_fast.gguf) exactly. Mirrors the
//      hybrid full-stream gate in test_mtp.cpp; isolated from ViT drift by feeding
//      the reference projected_tokens (same fixture).
#include "lm.hpp"
#include "qwen2.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "mtp.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>

static constexpr int BOX_START = 151668, BOX_END = 151669;
static constexpr int COORD_START = 151677, NONE_ID = 4064, VOCAB = 152681;

// build a VOCAB-sized logits row: low baseline, with the given {id,logit} spikes.
static std::vector<float> row(std::initializer_list<std::pair<int,float>> spikes){
    std::vector<float> r(VOCAB, -20.0f);
    for(auto& s : spikes) r[s.first] = s.second;
    return r;
}

static int synthetic(){
    int ok = 1;

    // --- handle_pattern: a MALFORMED box (box_start, 1 coord, then non-coord/non-box-end) ---
    // hybrid -> error_box (tokens=x0[:coord_ix], need_ar) ; fast -> coord_box (full x0, no AR).
    std::vector<int32_t> bad = { BOX_START, 151800, NONE_ID, NONE_ID, NONE_ID, NONE_ID };
    auto ph = la::mtp::handle_pattern(bad, /*fast=*/false);
    auto pf = la::mtp::handle_pattern(bad, /*fast=*/true);
    ok &= (ph.type == "error_box" && ph.need_ar == true &&
           pf.type == "coord_box" && pf.need_ar == false && pf.tokens == bad);
    std::printf("handle_pattern malformed: hybrid=%s(ar=%d) fast=%s(ar=%d) ok=%d\n",
                ph.type.c_str(), (int)ph.need_ar, pf.type.c_str(), (int)pf.need_ar, ok);

    // --- decode_bbox_avg (via select_new_tokens): a legal box frame whose coord
    // positions trigger the hybrid abnormal rule (first_valid_prob<0.9, valid_counts>1,
    // max-min>60). hybrid zeros those coords; fast keeps the top valid coord. ---
    const int cA = COORD_START + 100;          // 151777
    const int cB = cA + 70;                     // 151847 (spread 70 > 60)
    const int filler = 50000;                   // non-coord, highest -> pulls coord prob < 0.9
    std::vector<std::vector<float>> l6(6);
    l6[0] = row({{BOX_START, 12.0f}});          // p(box_start) ~ 1 (>0.7)
    for(int p=1;p<=4;++p) l6[p] = row({{filler,2.2f},{cA,2.0f},{cB,1.8f}});
    l6[5] = row({{BOX_END, 12.0f}});            // end_score (>0.2)
    auto bh = la::mtp::select_new_tokens(l6, 4, /*fast=*/false);
    auto bf = la::mtp::select_new_tokens(l6, 4, /*fast=*/true);
    std::vector<int32_t> exp_h = { BOX_START, 0, 0, 0, 0, BOX_END };
    std::vector<int32_t> exp_f = { BOX_START, cA, cA, cA, cA, BOX_END };
    int box_ok = (bh == exp_h && bf == exp_f);
    ok &= box_ok;
    std::printf("decode_bbox_avg abnormal: hybrid[1]=%d fast[1]=%d (expect 0 vs %d) ok=%d\n",
                bh.size()>1?bh[1]:-1, bf.size()>1?bf[1]:-1, cA, box_ok);
    return ok;
}

int main(){
    int syn = synthetic();
    if(!syn){ std::printf("synthetic fast-branch checks FAILED\n"); return 1; }

    const char* gguf=std::getenv("LA_TEST_GGUF");
    const char* base=std::getenv("LA_TEST_BASELINE");
    const char* fast=std::getenv("LA_TEST_FAST");
    if(!gguf||!base||!fast){ std::fprintf(stderr,"env unset; synthetic ok; skip e2e\n"); return 77; }

    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Backend be; la::LMForward lm(ml, be);
    std::vector<int32_t> ids; la_parity::load_baseline_i32(base,"input_ids",ids);          // 297 prompt
    std::vector<float> proj; std::vector<int64_t> ps; la_parity::load_baseline(base,"projected_tokens",proj,ps);

    std::vector<int32_t> got;
    if(!lm.decode_hybrid(ids, proj, 256, got, nullptr, /*fast=*/true)) return 1;
    // fast_token_ids is the GENERATED suffix only (upstream generated[:, seq_len:]),
    // same as decode_hybrid's return -> compare directly (cf. test_mtp hybrid gate).
    std::vector<int32_t> ref; la_parity::load_baseline_i32(fast,"fast_token_ids",ref);
    int eok = ((int)got.size() == (int)ref.size());
    for(int i=0;i<(int)ref.size() && eok;++i){
        if(got[i] != ref[i]){
            std::printf("fast stream mismatch@%d got=%d ref=%d\n", i, got[i], ref[i]); eok=0;
        }
    }
    std::printf("fast full-stream match=%d (%zu vs %zu)\n", eok, got.size(), ref.size());
    return (syn && eok)?0:1;
}
