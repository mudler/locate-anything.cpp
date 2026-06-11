#include "lm.hpp"
#include "qwen2.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "mtp.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* base=std::getenv("LA_TEST_BASELINE");
    const char* mtp=std::getenv("LA_TEST_MTP");
    if(!gguf||!base||!mtp){std::fprintf(stderr,"env unset; skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Backend be; la::LMForward lm(ml, be);
    std::vector<int32_t> ids; la_parity::load_baseline_i32(base,"input_ids",ids);   // 297 prompt
    std::vector<float> proj; std::vector<int64_t> ps; la_parity::load_baseline(base,"projected_tokens",proj,ps);
    const auto& c = ml.config();
    la::ResidentKV kv; kv.init(be, (int)c.lm_n_layers, (int)c.lm_head_dim, (int)c.lm_n_kv_heads, (int)ids.size()+64);
    if(!lm.prefill_resident(ids, proj, kv)) return 1;     // kv.past_len = 297
    // round-0 block: [last prompt token, text_mask x5]
    std::vector<int32_t> block = { ids.back(), 151676,151676,151676,151676,151676 };
    std::vector<float> xblk; lm.embed_tokens_host(block, xblk);     // [H*6]
    std::vector<float> got6;
    if(!lm.mtp_block_forward(kv, xblk, /*n_new*/6, /*n_recompute*/0, got6)) return 1;  // [6*vocab]
    // reference mtp_logits6[0] = first 6*vocab of the [44,6,vocab] tensor
    std::vector<float> ref; std::vector<int64_t> rs; la_parity::load_baseline(mtp,"mtp_logits6",ref,rs);
    std::vector<float> ref0(ref.begin(), ref.begin()+got6.size());
    bool ok = la_parity::compare(got6, ref0, "mtp_logits6_round0", 5e-2f, 5e-2f);

    // box-frame decode gate: for each MTP round, softmax the captured logits6[r] and
    // check select_new_tokens reproduces the reference mtp_new_tokens[r].
    std::vector<float> L6; std::vector<int64_t> ls; la_parity::load_baseline(mtp,"mtp_logits6",L6,ls);
    std::vector<int32_t> nt_all; la_parity::load_baseline_i32(mtp,"mtp_new_tokens",nt_all);
    std::vector<int32_t> nt_lens; la_parity::load_baseline_i32(mtp,"mtp_new_lens",nt_lens);
    int vocab=(int)la_parity::load_kv_u32(mtp,"la_mtp.vocab");
    int nrounds=(int)nt_lens.size();
    int dok=1, off=0;
    for(int r=0;r<nrounds;++r){
        std::vector<std::vector<float>> logits6(6, std::vector<float>(vocab));
        for(int p=0;p<6;++p) for(int t=0;t<vocab;++t) logits6[p][t]=L6[((size_t)r*6+p)*vocab+t];
        std::vector<int32_t> got=la::mtp::select_new_tokens(logits6);
        std::vector<int32_t> ref(nt_all.begin()+off, nt_all.begin()+off+nt_lens[r]); off+=nt_lens[r];
        if(got!=ref){ std::printf("decode mismatch round %d: got",r); for(int x:got)std::printf(" %d",x);
                      std::printf(" ref"); for(int x:ref)std::printf(" %d",x); std::printf("\n"); dok=0; }
    }
    std::printf("box-frame decode: %d/%d rounds match (ok=%d)\n", nrounds, nrounds, dok);

    // --- control-flow trace replay: drive the hybrid state machine over the reference
    //     round sequence and assert committed tokens + mode switches reproduce the stream. ---
    std::vector<int32_t> kinds; la_parity::load_baseline_i32(mtp,"mtp_round_kinds",kinds);   // [55]
    std::vector<int32_t> ar_tok; la_parity::load_baseline_i32(mtp,"mtp_ar_tokens",ar_tok);   // [11]
    std::vector<int32_t> stream; la_parity::load_baseline_i32(mtp,"hybrid_token_ids",stream);// [258]
    std::vector<int32_t> cm_all; la_parity::load_baseline_i32(mtp,"mtp_committed",cm_all);
    std::vector<int32_t> cm_lens; la_parity::load_baseline_i32(mtp,"mtp_committed_lens",cm_lens);
    la::mtp::HybridState st;
    std::vector<int32_t> replay;
    int mi=0, ai=0, ntoff2=0, cmoff=0, cfok=1;
    for(size_t r=0;r<kinds.size() && !st.terminated;++r){
        // the state machine's own use_mtp must agree with the reference round kind:
        // a real divergence in the error_box/box_end_ar switch timing would break this.
        if(st.use_mtp != (kinds[r]==0)){
            std::printf("control-flow: round %zu use_mtp=%d but ref kind=%d\n",
                        r, (int)st.use_mtp, (int)kinds[r]); cfok=0; break;
        }
        la::mtp::StepOut o;
        if(kinds[r]==0){ // MTP: feed the reference new_tokens for this MTP round
            std::vector<int32_t> nt(nt_all.begin()+ntoff2, nt_all.begin()+ntoff2+nt_lens[mi]); ntoff2+=nt_lens[mi];
            o = la::mtp::hybrid_mtp_step(st, nt);
            // sanity: committed must equal the reference mtp_committed[mi]
            std::vector<int32_t> exp(cm_all.begin()+cmoff, cm_all.begin()+cmoff+cm_lens[mi]); cmoff+=cm_lens[mi];
            if(o.committed!=exp){ std::printf("control-flow: MTP round %zu committed mismatch\n", r); cfok=0; }
            ++mi;
        } else {         // AR: feed the reference AR token for this AR round
            o = la::mtp::hybrid_ar_step(st, ar_tok[ai]); ++ai;
        }
        for(int t: o.committed) replay.push_back(t);
    }
    if(cfok){
        cfok = ((int)replay.size()==(int)stream.size());
        for(size_t i=0;i<stream.size() && cfok;++i) cfok &= (replay[i]==stream[i]);
    }
    std::printf("control-flow replay: stream match=%d (%zu vs %zu)\n", cfok, replay.size(), stream.size());

    // --- synthetic AR/MTP transition unit checks ---
    int uok=1;
    { la::mtp::HybridState s; s.use_mtp=false;                 // box_end_ar: switch back to MTP
      auto o=la::mtp::hybrid_ar_step(s, 151669);
      uok &= (o.out_type=="box_end_ar" && s.use_mtp==true && !s.terminated &&
              o.committed.size()==1 && o.committed[0]==151669); }
    { la::mtp::HybridState s; s.use_mtp=false;                 // coord_ar: stay AR, commit raw token
      auto o=la::mtp::hybrid_ar_step(s, 151700);
      uok &= (o.out_type=="coord_ar" && s.use_mtp==false && !s.terminated &&
              o.committed.size()==1 && o.committed[0]==151700); }
    { la::mtp::HybridState s; s.use_mtp=false;                 // im_end: terminate, commit RAW token
      auto o=la::mtp::hybrid_ar_step(s, 9999);
      uok &= (o.out_type=="im_end" && s.terminated==true &&
              o.committed.size()==1 && o.committed[0]==9999); }
    { la::mtp::HybridState s;                                  // MTP error_box -> drop to AR
      // box_start + 1 coord + non-coord/non-box-end -> error_box (need switch_to_ar)
      std::vector<int32_t> nt={151668,151800,4064,4064,4064,4064};
      auto o=la::mtp::hybrid_mtp_step(s, nt);
      uok &= (o.out_type=="error_box" && s.use_mtp==false && !s.terminated); }
    { la::mtp::HybridState s;                                  // MTP im_end terminal
      std::vector<int32_t> nt={152678,0,0,0,0,0};
      auto o=la::mtp::hybrid_mtp_step(s, nt);
      uok &= (o.out_type=="im_end" && s.terminated==true); }
    std::printf("synthetic transition checks: ok=%d\n", uok);

    // --- full hybrid decode (M5 EXIT): drive the real forward+decode+control flow ---
    std::vector<std::vector<float>> cap6;   // per-MTP-round logits6 [6*vocab]
    std::vector<int32_t> got;
    if(!lm.decode_hybrid(ids /*prompt*/, proj, 256, got, &cap6)) return 1;
    std::vector<int32_t> href; la_parity::load_baseline_i32(mtp,"hybrid_token_ids",href);  // 258
    int hok = ((int)got.size()==(int)href.size());
    for(int i=0;i<(int)href.size() && hok;++i){
        if(got[i]!=href[i]){ std::printf("hybrid mismatch@%d got=%d ref=%d\n",i,got[i],href[i]); hok=0; break; }
    }
    std::printf("hybrid full-stream match=%d (%zu vs %zu)\n", hok, got.size(), href.size());

    // --- early-stop: hybrid with early_stop=true truncates the degenerate box tail.
    //     It must (a) produce FEWER tokens than the run-to-cap stream, and (b) keep the
    //     leading real boxes intact (its stream is a prefix of the full stream up to the
    //     truncation point). ---
    std::vector<int32_t> got_es;
    if(!lm.decode_hybrid(ids, proj, 256, got_es, nullptr, /*fast*/false, /*early_stop*/true)) return 1;
    int es_ok = (got_es.size() < got.size()) && (got_es.size() >= 6);
    for(size_t i=0;i<got_es.size() && es_ok;++i) es_ok &= (got_es[i]==got[i]);  // prefix of full
    std::printf("early-stop: %zu tokens vs %zu full, prefix+shorter ok=%d\n",
                got_es.size(), got.size(), es_ok);

    // --- per-round logits6 debug gate: exercises the n_recompute>0 mask/position path
    //     (Task 2's round-0 gate only had n_recompute==0). Compares each captured MTP
    //     round's logits6 against the reference mtp_logits6[r]. ---
    int lr_ok=1; float lr_worst=0.f; int lr_worst_round=-1;
    int ncap=std::min((int)cap6.size(), nrounds);
    for(int r=0;r<ncap;++r){
        std::vector<float> refr(L6.begin()+(size_t)r*6*vocab, L6.begin()+(size_t)(r+1)*6*vocab);
        float md=0.f;
        size_t n=std::min(cap6[r].size(), refr.size());
        for(size_t i=0;i<n;++i){ float d=std::fabs(cap6[r][i]-refr[i]); if(d>md) md=d; }
        if(md>lr_worst){ lr_worst=md; lr_worst_round=r; }
        if(md>5e-2f){ lr_ok=0; }
    }
    std::printf("per-round logits6: %d rounds, worst diff=%.4f @round %d (ok=%d)\n",
                ncap, lr_worst, lr_worst_round, lr_ok);

    return (ok && dok && cfok && uok && hok && lr_ok && es_ok)?0:1;
}
