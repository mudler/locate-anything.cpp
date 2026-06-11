#include "vit_encoder.hpp"
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
    la::Backend be; la::VitEncoder vit(ml,be);
    std::vector<float> px; std::vector<int64_t> s;
    if(!la_parity::load_baseline(base,"pixel_values",px,s)) return 1;  // [1024,3,14,14]=[1024,588]
    std::vector<float> got; // [1152,1024]
    if(!vit.patch_and_pos(px, 32, 32, got)) return 1;
    std::vector<float> ref; std::vector<int64_t> rs;
    if(!la_parity::load_baseline(base,"vit_pos_added",ref,rs)) return 1; // [1024,1152]
    // ggml output ne=[1152,1024] stores hidden (ne[0]) fastest, so its raw flat
    // layout is already [token=1024 outer, hidden=1152 inner] == ref [1024,1152].
    // ggml output ne=[1152,1024] stores hidden (ne[0]) fastest, so its raw flat
    // layout is [token=1024 outer, hidden=1152 inner] == ref [1024,1152]; no transpose.
    bool ok = la_parity::compare(got, ref, "vit_pos_added", 1e-3f, 1e-3f);

    // --- one encoder block -> vit_layer_00 ---
    std::vector<float> l0;                                // [1152,1024] (hidden,token)
    if(!vit.block0(px, 32, 32, l0)) return 1;
    std::vector<float> l0ref; std::vector<int64_t> ls;
    if(!la_parity::load_baseline(base,"vit_layer_00",l0ref,ls)) return 1;  // [1024,1152]
    // ggml ne=[1152,1024] raw flat = token outer, hidden inner == ref [1024,1152].
    bool ok0 = la_parity::compare(l0, l0ref, "vit_layer_00", 2e-2f, 2e-2f);

    // --- full 27-block stack + final_norm -> vit_layer_26 / vit_final ---
    std::vector<float> vfinal; std::vector<std::vector<float>> caps;
    if(!vit.forward(px, 32, 32, vfinal, {26}, caps)) return 1;
    std::vector<float> r26; std::vector<int64_t> s26;
    if(!la_parity::load_baseline(base,"vit_layer_26",r26,s26)) return 1;  // [1024,1152]
    // caps[0]: block 26 output ne=[1152,1024] raw flat == ref [1024,1152]; no transpose.
    bool ok26 = la_parity::compare(caps[0], r26, "vit_layer_26", 3e-2f, 3e-2f);
    std::vector<float> rf; std::vector<int64_t> sf;
    if(!la_parity::load_baseline(base,"vit_final",rf,sf)) return 1;       // [1024,1152]
    bool okf = la_parity::compare(vfinal, rf, "vit_final", 3e-2f, 3e-2f);

    // --- 2x2 patch merger -> merged_tokens (M2 EXIT) ---
    // vfinal raw flat is [token,hidden] row-major == vf[tok*1152 + d]; no transpose.
    std::vector<float> merged = la::merge_patches(vfinal, 32, 32, 1152);  // [256,4608]
    std::vector<float> rmt; std::vector<int64_t> ms;
    if(!la_parity::load_baseline(base,"merged_tokens",rmt,ms)) return 1;  // [256,4608]
    bool okm = la_parity::compare(merged, rmt, "merged_tokens", 3e-2f, 3e-2f);

    return (ok && ok0 && ok26 && okf && okm) ? 0 : 1;
}
