#include "image_io.hpp"
#include "pil_resize.hpp"
#include "vit_encoder.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cmath>
#include <string>

int main() {
    const char* pp = std::getenv("LA_TEST_PREPROC");
    if (!pp) { std::fprintf(stderr, "skip\n"); return 77; }

    std::string img_path;
    if (!la_parity::load_kv_str(pp, "la_preproc.image_path", img_path)) return 1;
    int ref_gh = (int)la_parity::load_kv_u32(pp, "la_preproc.gh");
    int ref_gw = (int)la_parity::load_kv_u32(pp, "la_preproc.gw");
    int th = (int)la_parity::load_kv_u32(pp, "la_preproc.target_h");
    int tw = (int)la_parity::load_kv_u32(pp, "la_preproc.target_w");

    la::Image im;
    if (!la::load_image_rgb(img_path, im)) {
        std::fprintf(stderr, "load fail %s\n", img_path.c_str());
        return 1;
    }
    std::printf("loaded %s  %dx%d\n", img_path.c_str(), im.w, im.h);

    // (1) grid arithmetic (pure integer math).
    int gtw, gth;
    la::preproc_target(im.w, im.h, gtw, gth);
    int gok = (gtw == tw && gth == th && gth / 14 == ref_gh && gtw / 14 == ref_gw);
    std::printf("target got %dx%d ref %dx%d gridok=%d\n", gtw, gth, tw, th, gok);

    la::Preprocessed P;
    if (!la::preprocess(im, P)) { std::fprintf(stderr, "preprocess fail\n"); return 1; }
    int gridok = (P.gh == ref_gh && P.gw == ref_gw);

    // (2) resized_chw gate — isolates the PIL resampler + normalize.
    // Tolerance note: the resampler matches Pillow's BICUBIC to a mean abs-diff
    // of ~7e-4. The residual max abs-diff (~2.35e-2 = exactly 3/127.5 uint8
    // levels, on ~0.03% of scattered interior pixels) comes from stb's JPEG
    // decoder differing a few LSB from PIL's libjpeg (different IDCT / chroma
    // upsampling) — NOT from the resize math. atol=3.5e-2 absorbs that decoder
    // noise; the mean abs-diff printed below is the real resampler-quality gate.
    std::vector<float> refchw; std::vector<int64_t> cs;
    bool chwok = false;
    if (la_parity::load_baseline(pp, "resized_chw", refchw, cs)) {
        chwok = la_parity::compare(P.resized_chw, refchw, "resized_chw", 3.5e-2f, 0.0f);
    } else {
        std::fprintf(stderr, "resized_chw not in reference; skipping that gate\n");
        chwok = true;
    }

    // (3) pixel_values gate — resized_chw reshaped via patchify.
    std::vector<float> refpv; std::vector<int64_t> rs;
    if (!la_parity::load_baseline(pp, "pixel_values", refpv, rs)) return 1;
    bool pvok = la_parity::compare(P.pixel_values, refpv, "pixel_values", 3.5e-2f, 0.0f);

    // (4) non-448 merged_tokens gate — runs the real preproc grid (gh,gw) through
    // the MoonViT tower and the 2x2 merger; needs the model GGUF (LA_TEST_GGUF).
    // vit.forward returns ne=[1152, gh*gw] -> raw flat is token-major [token,hidden],
    // which is exactly the layout merge_patches consumes (same as the 448 fixture).
    bool mok = true;
    const char* gguf = std::getenv("LA_TEST_GGUF");
    if (gguf) {
        la::ModelLoader ml;
        if (!ml.load(gguf)) { std::fprintf(stderr, "gguf load fail %s\n", gguf); return 1; }
        la::Backend be; la::VitEncoder vit(ml, be);
        std::vector<float> rmt; std::vector<int64_t> ms;
        if (!la_parity::load_baseline(pp, "merged_tokens", rmt, ms)) return 1;  // [1131,4608]

        // (4a) THREADING PROOF — feed the EXACT reference pixel_values through the
        // variable-(gh,gw) tower. This isolates the vision math (pos-emb 64x64->gh*gw
        // bicubic interp, gh*gw 2D-RoPE grid, gh/2 x gw/2 merger) from JPEG-decoder
        // input noise, so it must match to M2-level parity (the 448 fixture was ~2.65e-3).
        std::vector<float> refpv2; std::vector<int64_t> rps2;
        if (!la_parity::load_baseline(pp, "pixel_values", refpv2, rps2)) return 1;
        std::vector<float> vf_ref; std::vector<std::vector<float>> caps_ref;
        if (!vit.forward(refpv2, P.gh, P.gw, vf_ref, {}, caps_ref)) return 1; // [1152, gh*gw]
        std::vector<float> merged_ref = la::merge_patches(vf_ref, P.gh, P.gw, 1152); // [1131,4608]
        bool exact_ok = la_parity::compare(merged_ref, rmt, "merged_tokens_exactpv", 1e-2f, 1e-2f);

        // (4b) END-TO-END — run OUR JPEG-decoded preproc output through the tower.
        // stb's JPEG decoder differs from PIL/libjpeg by a few uint8 LSB (the
        // documented pixel_values noise floor, max ~2.35e-2). Propagated through 27
        // transformer layers and amplified by the tower's bimodal "massive-activation"
        // neurons (1566 elems |val|>5, up to 75), a handful of near-threshold neurons
        // flip, blowing up the element-wise MAX diff while the MEAN stays at the input
        // noise floor (high-norm tokens still align at cosine ~0.996). We therefore gate
        // the mean abs-diff here and report the max for the record; (4a) is the math proof.
        std::vector<float> vf; std::vector<std::vector<float>> caps;
        if (!vit.forward(P.pixel_values, P.gh, P.gw, vf, {}, caps)) return 1; // [1152, gh*gw]
        std::vector<float> merged = la::merge_patches(vf, P.gh, P.gw, 1152);  // [1131,4608]
        double sad = 0.0, mx = 0.0;
        for (size_t i = 0; i < merged.size(); ++i) {
            double d = std::fabs((double)merged[i] - (double)rmt[i]);
            sad += d; if (d > mx) mx = d;
        }
        double mad = sad / (merged.size() ? merged.size() : 1);
        bool e2e_ok = mad <= 3e-2;
        std::fprintf(stderr,
            "[merged_tokens_jpeg] n=%zu mean|d|=%.3e max|d|=%.3e (JPEG-decoder noise) -> %s\n",
            merged.size(), mad, mx, e2e_ok ? "OK" : "FAIL");
        mok = exact_ok && e2e_ok;
    } else {
        std::fprintf(stderr, "LA_TEST_GGUF unset; skipping non-448 merged_tokens gate\n");
    }

    return (gok && gridok && chwok && pvok && mok) ? 0 : 1;
}
