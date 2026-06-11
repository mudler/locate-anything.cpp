#include "projector.hpp"
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
    la::Backend be; la::Projector proj(ml, be);
    std::vector<float> merged; std::vector<int64_t> ms;
    if(!la_parity::load_baseline(base,"merged_tokens",merged,ms)) return 1;   // [256,4608]
    std::vector<float> got;                                                    // [2048,256]
    if(!proj.project(merged, got)) return 1;
    std::vector<float> ref; std::vector<int64_t> rs;
    if(!la_parity::load_baseline(base,"projected_tokens",ref,rs)) return 1;     // [256,2048]
    // got is ggml [2048,256] raw flat = [token,hidden]; ref is [256,2048] token-major -> same flat order
    bool ok = la_parity::compare(got, ref, "projected_tokens", 1e-3f, 1e-3f);
    return ok?0:1;
}
