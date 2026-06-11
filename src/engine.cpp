#include "engine.hpp"
#include "common.hpp"
#include "image_io.hpp"
#include "prompt.hpp"
#include "vit_encoder.hpp"
#include "projector.hpp"
#include "lm.hpp"
#include <thread>
namespace la {

std::unique_ptr<Engine> Engine::load(const std::string& gguf_path, int n_threads){
    std::unique_ptr<Engine> eng(new Engine());
    if(!eng->ml_.load(gguf_path)) return nullptr;
    if(!eng->tok_.load(eng->ml_)) return nullptr;
    eng->be_.set_n_threads(n_threads > 0 ? n_threads
                                         : (int)std::thread::hardware_concurrency());
    // Move graph weights onto the compute device (GPU) before the first compute.
    // No-op on a CPU backend (graphs keep using the gguf host tensors directly).
    if(!eng->ml_.offload_weights(eng->be_)){
        LA_LOG("Engine::load: weight offload failed");
        return nullptr;
    }
    return eng;
}

std::vector<Box> Engine::locate(const std::string& image_path, const std::string& query,
                                Mode mode, int max_new){
    Image img;
    if(!load_image_rgb(image_path, img)) return {};
    return locate_image(img, query, mode, max_new);
}

std::vector<Box> Engine::locate_buffer(const unsigned char* bytes, size_t len,
                                       const std::string& query, Mode mode, int max_new){
    Image img;
    if(!load_image_rgb_buffer(bytes, len, img)) return {};
    return locate_image(img, query, mode, max_new);
}

std::vector<Box> Engine::locate_image(const Image& img, const std::string& query,
                                      Mode mode, int max_new){
    std::vector<Box> empty;

    // 1) preprocess (PIL-bicubic resize -> normalize -> patchify)
    Preprocessed P;
    if(!preprocess(img, P)) return empty;

    // 2) prompt input_ids (chat template + (gh/2)*(gw/2) <IMG_CONTEXT> tokens)
    std::vector<int32_t> prompt_ids = build_prompt(tok_, P.gh, P.gw, query);
    if(prompt_ids.empty()) return empty;

    // 3) vision encoder -> vit_final [1152, gh*gw]; raw flat is already token-major
    //    ([token,hidden]) == what merge_patches expects. 2x2 merge -> [N,4608].
    VitEncoder vit(ml_, be_);
    std::vector<float> vfinal; std::vector<std::vector<float>> caps;
    if(!vit.forward(P.pixel_values, P.gh, P.gw, vfinal, {}, caps)) return empty;
    const int vit_hidden = 1152;
    std::vector<float> merged = merge_patches(vfinal, P.gh, P.gw, vit_hidden);  // [N,4608]

    // 4) projector -> vision tokens [2048, N] (N inferred from merged size)
    Projector proj(ml_, be_);
    std::vector<float> projected;
    if(!proj.project(merged, projected)) return empty;

    // 5) LM decode (Slow = AR resident oracle; Hybrid/Fast = parallel box decoding,
    //    fast=MTP-only no AR fallback). embed_and_splice overwrites the N <IMG_CONTEXT>
    //    slots with projected tokens.
    LMForward lm(ml_, be_);
    std::vector<int32_t> gen_ids;
    bool ok;
    if(mode==Mode::Slow) ok = lm.decode_greedy_resident(prompt_ids, projected, max_new, gen_ids);
    else                 ok = lm.decode_hybrid(prompt_ids, projected, max_new, gen_ids,
                                               nullptr, /*fast=*/mode==Mode::Fast,
                                               /*early_stop=*/!std::getenv("LA_NO_EARLYSTOP"));
    if(!ok) return empty;

    // 6) parse boxes. Coords denormalize against the preprocessed target size
    //    (img_w = target_w = gw*14, img_h = target_h = gh*14).
    return parse_boxes(gen_ids, P.target_w, P.target_h,
                       [&](const std::vector<int32_t>& t){ return tok_.decode(t); });
}

}
