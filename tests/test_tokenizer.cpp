#include "tokenizer.hpp"
#include "prompt.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <string>
#include <cstdio>

int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF");
    if(!gguf){std::fprintf(stderr,"skip\n");return 77;}
    la::ModelLoader ml; if(!ml.load(gguf)) return 1;
    la::Tokenizer tok; if(!tok.load(ml)) return 1;
    int ok=1;

    // special tokens atomic + correct ids
    ok &= (tok.token_to_id("<IMG_CONTEXT>")==151665);
    ok &= (tok.token_to_id("<box>")==151668);
    ok &= (tok.token_to_id("<0>")==151677);
    ok &= (tok.token_to_id("<1000>")==152677);
    ok &= (tok.token_to_id("<|im_start|>")==151644);
    if(!ok) std::printf("special id check failed\n");

    // round-trip on plain text (encode then decode == original)
    for(const char* s : {"Locate all the instances that matches", "person", "bus", "a cat and a dog."}){
        auto ids=tok.encode(s); auto back=tok.decode(ids);
        if(back!=s){ std::printf("roundtrip fail: '%s' -> '%s'\n", s, back.c_str()); ok=0; }
    }

    // special tokens encode atomically (one id each, not BPE'd)
    auto e=tok.encode("<IMG_CONTEXT><IMG_CONTEXT>");
    if(!(e.size()==2 && e[0]==151665 && e[1]==151665)){
        std::printf("atomic special fail: size=%zu\n", e.size()); ok=0;
    }
    // mixed plain + special atomicity
    auto m=tok.encode("Locate <box> person");
    {
        int cnt=0; for(int32_t id:m) if(id==151668) cnt++;
        if(cnt!=1){ std::printf("mixed special fail: <box> count=%d\n", cnt); ok=0; }
    }

    // --- reference gate: decode the real prompt input_ids, re-encode, expect identity ---
    const char* base=std::getenv("LA_TEST_BASELINE");
    if(base){
        std::vector<int32_t> ref_ids;
        if(la_parity::load_baseline_i32(base,"input_ids",ref_ids) && !ref_ids.empty()){
            std::string text = tok.decode(ref_ids);
            std::printf("ref prompt decoded %zu ids -> %zu bytes\n", ref_ids.size(), text.size());
            // sanity: the prompt should contain ChatML markers and the image token.
            int has_im = text.find("<|im_start|>")!=std::string::npos;
            int has_img = text.find("<IMG_CONTEXT>")!=std::string::npos;
            if(!has_im || !has_img){ std::printf("ref decode missing markers (im=%d img=%d)\n", has_im, has_img); ok=0; }
            auto re = tok.encode(text);
            int eq = (re.size()==ref_ids.size());
            int mism=-1;
            for(size_t k=0;k<re.size() && k<ref_ids.size();++k)
                if(re[k]!=ref_ids[k]){ eq=0; mism=(int)k; break; }
            std::printf("ref re-encode parity: match=%d (got %zu ref %zu", eq, re.size(), ref_ids.size());
            if(mism>=0) std::printf(", first mismatch @%d got=%d ref=%d", mism, re[mism], ref_ids[mism]);
            std::printf(")\n");
            ok &= eq;
        } else {
            std::printf("(no input_ids in baseline; skipping ref gate)\n");
        }
    }

    // --- prompt builder gate: build the full prompt ids vs the HF reference ---
    const char* pp=std::getenv("LA_TEST_PREPROC");
    if(pp){
        int gh=(int)la_parity::load_kv_u32(pp,"la_preproc.gh"), gw=(int)la_parity::load_kv_u32(pp,"la_preproc.gw");
        std::string query; la_parity::load_kv_str(pp,"la_preproc.prompt",query);
        std::vector<int32_t> built = la::build_prompt(tok, gh, gw, query);
        std::vector<int32_t> ref; la_parity::load_baseline_i32(pp,"input_ids",ref);
        int pok = ((int)built.size()==(int)ref.size());
        for(size_t i=0;i<ref.size() && i<built.size() && pok;++i){ if(built[i]!=ref[i]){ std::printf("prompt mismatch@%zu got=%d ref=%d (ctx ref[%zu..]=%d %d %d)\n", i,built[i],ref[i], i, (i<ref.size()?ref[i]:-1),(i+1<ref.size()?ref[i+1]:-1),(i+2<ref.size()?ref[i+2]:-1)); pok=0; } }
        std::printf("prompt build: input_ids match=%d (%zu vs %zu)\n", pok, built.size(), ref.size());
        ok &= pok;
    }

    std::printf("tokenizer ok=%d\n", ok);
    return ok?0:1;
}
