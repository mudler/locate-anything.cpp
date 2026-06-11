#include "lm.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* base=std::getenv("LA_TEST_BASELINE");
    if(!gguf||!base){std::fprintf(stderr,"skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Backend be; la::LMForward lm(ml, be);
    std::vector<int32_t> ids;
    if(!la_parity::load_baseline_i32(base,"input_ids",ids)) return 1;        // [297]
    std::vector<float> proj; std::vector<int64_t> ps;
    if(!la_parity::load_baseline(base,"projected_tokens",proj,ps)) return 1;  // [256,2048]
    std::vector<float> spliced;
    if(!lm.embed_and_splice(ids, proj, spliced)) return 1;                    // [2048,297]
    std::vector<float> ref; std::vector<int64_t> rs;
    if(!la_parity::load_baseline(base,"embeds_after_splice",ref,rs)) return 1;// [1,297,2048]
    bool ok = la_parity::compare(spliced, ref, "embeds_after_splice", 1e-4f, 1e-4f);
    std::vector<float> l0;
    if(!lm.run_layer0(ref, 297, l0)) return 1;     // feed the REFERENCE embeds_after_splice
    std::vector<float> r0; std::vector<int64_t> r0s;
    if(!la_parity::load_baseline(base,"lm_layer_00",r0,r0s)) return 1;   // [1,297,2048]
    bool ok0 = la_parity::compare(l0, r0, "lm_layer_00", 2e-2f, 2e-2f);
    std::vector<float> logits; std::vector<std::vector<float>> caps;
    if(!lm.forward(ids, proj, logits, {35}, caps)) return 1;          // logits [152681]
    std::vector<float> r35; std::vector<int64_t> r35s;
    if(!la_parity::load_baseline(base,"lm_layer_35",r35,r35s)) return 1;
    bool ok35 = la_parity::compare(caps[0], r35, "lm_layer_35", 3e-2f, 3e-2f);
    std::vector<float> rl; std::vector<int64_t> rls;
    if(!la_parity::load_baseline(base,"logits_step0",rl,rls)) return 1;  // [152681]
    bool okl = la_parity::compare(logits, rl, "logits_step0", 1e-1f, 1e-1f);
    auto argmax=[](const std::vector<float>& v){ int b=0; for(int i=1;i<(int)v.size();++i) if(v[i]>v[b]) b=i; return b; };
    int am_got=argmax(logits), am_ref=argmax(rl);
    std::printf("argmax got=%d ref=%d (expect 151672)\n", am_got, am_ref);
    bool oka = (am_got==am_ref && am_ref==151672);
    return (ok && ok0 && ok35 && okl && oka)?0:1;
}
