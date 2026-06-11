#include "vit_posemb.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cstdio>
int main() {
    const char* gguf = std::getenv("LA_TEST_GGUF");
    const char* fix  = std::getenv("LA_TEST_FIXTURES");
    if (!gguf || !fix) { std::fprintf(stderr,"env unset; skip\n"); return 77; }
    la::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    ggml_tensor* pe = ml.tensor("vit.pos_emb.weight");
    if (!pe) return 1;
    std::vector<float> src(ggml_nelements(pe));
    std::memcpy(src.data(), pe->data, ggml_nbytes(pe));   // CPU buffer: direct
    std::vector<float> got = la::bicubic_pos_emb(src, /*base*/64, 64, /*c*/1152, /*gh*/32, /*gw*/32);
    std::vector<float> ref; std::vector<int64_t> shape;
    if (!la_parity::load_baseline(fix, "posemb_interp_32x32", ref, shape)) return 1;
    bool ok = la_parity::compare(got, ref, "posemb_interp", 1e-3f, 1e-3f);
    return ok ? 0 : 1;
}
