#include "lm.hpp"
#include "projector.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "qwen2.hpp"
#include "boxes.hpp"
#include "tokenizer.hpp"
#include "parity.hpp"
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* base=std::getenv("LA_TEST_BASELINE");
    const char* slow=std::getenv("LA_TEST_SLOW");
    if(!gguf||!base||!slow){std::fprintf(stderr,"env unset; skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Backend be; la::LMForward lm(ml, be);
    { la::ResidentKV kv;
      bool kok = kv.init(be, (int)ml.config().lm_n_layers, (int)ml.config().lm_head_dim,
                         (int)ml.config().lm_n_kv_heads, 600);
      kok &= (kv.k.size()==36 && kv.v.size()==36 && kv.k[0] && kv.k[0]->ne[0]==128 &&
              kv.k[0]->ne[1]==2 && kv.k[0]->ne[2]==600 && kv.buffer!=nullptr);
      std::printf("residentkv init=%d\n", kok);
      kv.free();
      if(!kok) return 1;
    }
    std::vector<int32_t> ids; la_parity::load_baseline_i32(base,"input_ids",ids);   // 297 prompt
    std::vector<float> proj; std::vector<int64_t> ps;
    la_parity::load_baseline(base,"projected_tokens",proj,ps);                       // [256,2048]
    std::vector<int32_t> ref; la_parity::load_baseline_i32(slow,"slow_token_ids",ref);
    // The slow stream is 31 tokens; reprefill the full stream (cheap) and compare exactly.
    const int N = (int)ref.size();   // gate the full slow stream
    std::vector<int32_t> got;
    if(!lm.decode_greedy_reprefill(ids, proj, N, got)) return 1;
    int ok = ((int)got.size() == N);
    int m = (int)std::min(got.size(), ref.size());
    for(int i=0;i<m;++i){ if(got[i]!=ref[i]){ std::printf("mismatch@%d got=%d ref=%d\n",i,got[i],ref[i]); ok=0; } }
    std::printf("decode match=%d (got %zu ref %d)\n", ok, got.size(), N);

    // --- box parse gate (on the FULL reference token stream) ---
    int img_w = (int)la_parity::load_kv_u32(slow, "la_slow.img_w");
    int img_h = (int)la_parity::load_kv_u32(slow, "la_slow.img_h");
    la::Tokenizer tok; if(!tok.load(ml)) return 1;
    auto decode_label = [&](const std::vector<int32_t>& t){ return tok.decode(t); };
    std::vector<la::Box> boxes = la::parse_boxes(ref, img_w, img_h, decode_label);
    std::vector<float> rboxes; std::vector<int64_t> rbs;
    la_parity::load_baseline(slow, "slow_boxes", rboxes, rbs);   // [4*4] flat
    int nb=(int)rboxes.size()/4;
    int bok = ((int)boxes.size()==nb);
    for(int k=0;k<nb && bok;++k){
        const float* r=&rboxes[k*4];
        bok &= (std::fabs(boxes[k].x1-r[0])<1.f && std::fabs(boxes[k].y1-r[1])<1.f &&
                std::fabs(boxes[k].x2-r[2])<1.f && std::fabs(boxes[k].y2-r[3])<1.f);
        bok &= (boxes[k].x1<=boxes[k].x2+1.f && boxes[k].y1<=boxes[k].y2+1.f);  // well-formed
    }
    std::printf("box parse: got %zu ref %d match=%d\n", boxes.size(), nb, bok);
    for(size_t k=0;k<boxes.size();++k)
        std::printf("  box[%zu] label='%s' [%.1f %.1f %.1f %.1f]\n",
                    k, boxes[k].label.c_str(),
                    boxes[k].x1, boxes[k].y1, boxes[k].x2, boxes[k].y2);

    // --- multi-word label decode gate ---
    // A multi-token label must byte-decode (space, not 'Ġ'). Encode "traffic light"
    // via the tokenizer, wrap it in a synthetic ref/box stream, and assert the
    // parsed label round-trips to "traffic light".
    const int REF_START=151672, REF_END=151673, BOX_START=151668, BOX_END=151669, COORD_START=151677;
    std::vector<int32_t> lab = tok.encode("traffic light");
    std::vector<int32_t> synth;
    synth.push_back(REF_START);
    for(int id : lab) synth.push_back(id);
    synth.push_back(REF_END);
    synth.push_back(BOX_START);
    synth.push_back(COORD_START+100); synth.push_back(COORD_START+200);
    synth.push_back(COORD_START+300); synth.push_back(COORD_START+400);
    synth.push_back(BOX_END);
    std::vector<la::Box> sboxes = la::parse_boxes(synth, 1000, 1000, decode_label);
    int lok = (sboxes.size()==1 && sboxes[0].label=="traffic light");
    std::printf("multi-word label: got '%s' match=%d\n",
                sboxes.empty()?"(none)":sboxes[0].label.c_str(), lok);

    // full-stream gate via resident KV-cache
    std::vector<int32_t> got_full;
    if(!lm.decode_greedy_resident(ids, proj, 256, got_full)) return 1;
    int fok = ((int)got_full.size()==(int)ref.size());
    for(int i=0;i<(int)ref.size() && fok;++i) fok &= (got_full[i]==ref[i]);
    std::printf("resident full-stream match=%d (%zu vs %zu)\n", fok, got_full.size(), ref.size());
    // resident == reprefill on first 16 tokens
    std::vector<int32_t> a,b2;
    lm.decode_greedy_reprefill(ids, proj, 16, a);
    lm.decode_greedy_resident(ids, proj, 16, b2);
    int eq=((int)a.size()>=16 && (int)b2.size()>=16);
    for(int i=0;i<16 && eq;++i) eq &= (a[i]==b2[i]);
    std::printf("resident==reprefill(16) = %d\n", eq);

    return (ok && bok && fok && eq && lok)?0:1;
}
