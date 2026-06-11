#include "image_io.hpp"
#include "pil_resize.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <cstdint>

namespace la {

namespace {
constexpr int kPatch = 14;
constexpr int kMergeH = 2;  // merge_kernel_size[0]
constexpr int kMergeW = 2;  // merge_kernel_size[1]
constexpr int kInTokenLimit = 25600;
}  // namespace

bool load_image_rgb(const std::string& path, Image& out) {
    int w = 0, h = 0, c = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &c, 3 /* force RGB */);
    if (!px) return false;
    out.w = w;
    out.h = h;
    out.rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    return true;
}

bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out) {
    int w = 0, h = 0, c = 0;
    unsigned char* px = stbi_load_from_memory(bytes, (int)len, &w, &h, &c, 3 /* force RGB */);
    if (!px) return false;
    out.w = w;
    out.h = h;
    out.rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    return true;
}

bool save_image_png(const std::string& path, const Image& img) {
    if (img.w <= 0 || img.h <= 0 ||
        img.rgb.size() != (size_t)img.w * (size_t)img.h * 3) {
        return false;
    }
    return stbi_write_png(path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3) != 0;
}

// Mirrors LocateAnythingImageProcessor.rescale. PIL image.size is (W, H).
void preproc_target(int w0, int h0, int& target_w, int& target_h) {
    int w = w0, h = h0;

    // (A) downscale if the patch-grid would exceed the token limit.
    if ((long)(w / kPatch) * (h / kPatch) > kInTokenLimit) {
        double scale = std::sqrt((double)kInTokenLimit /
                                 ((double)(w / kPatch) * (double)(h / kPatch)));
        w = (int)(w0 * scale);  // truncate (int())
        h = (int)(h0 * scale);
    }

    // (B) round each dimension up to a multiple of merge_kernel * patch.
    int pad_w = kMergeW * kPatch;  // 28
    int pad_h = kMergeH * kPatch;  // 28
    target_w = (int)std::ceil((double)w / pad_w) * pad_w;
    target_h = (int)std::ceil((double)h / pad_h) * pad_h;
}

bool preprocess(const Image& img, Preprocessed& out) {
    if (img.w <= 0 || img.h <= 0 || img.rgb.empty()) return false;

    // Reproduce the (possibly two-stage) PIL-resize chain from rescale().
    int w = img.w, h = img.h;
    std::vector<uint8_t> cur = img.rgb;

    if ((long)(w / kPatch) * (h / kPatch) > kInTokenLimit) {
        double scale = std::sqrt((double)kInTokenLimit /
                                 ((double)(w / kPatch) * (double)(h / kPatch)));
        int w1 = (int)(img.w * scale);
        int h1 = (int)(img.h * scale);
        cur = pil_bicubic_resize(cur, w, h, w1, h1);
        w = w1;
        h = h1;
    }

    int target_w, target_h;
    preproc_target(img.w, img.h, target_w, target_h);
    if (target_w != w || target_h != h) {
        cur = pil_bicubic_resize(cur, w, h, target_w, target_h);
    }

    // assert target_w//14 < 512 && target_h//14 < 512
    if (target_w / kPatch >= 512 || target_h / kPatch >= 512) return false;

    const int gh = target_h / kPatch;
    const int gw = target_w / kPatch;
    out.gh = gh;
    out.gw = gw;
    out.target_h = target_h;
    out.target_w = target_w;

    // Normalize -> CHW [3, target_h, target_w]: v = raw/127.5 - 1.
    const size_t plane = (size_t)target_h * target_w;
    out.resized_chw.resize(3 * plane);
    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            const uint8_t* p = &cur[((size_t)y * target_w + x) * 3];
            for (int c = 0; c < 3; ++c) {
                out.resized_chw[(size_t)c * plane + (size_t)y * target_w + x] =
                    (float)p[c] / 127.5f - 1.0f;
            }
        }
    }

    // Patchify: patch token t = row*gw + col (row outer). Within-patch [c,i,j].
    // pv[t*588 + c*196 + i*14 + j] = normalized pixel (c, y=row*14+i, x=col*14+j).
    const int N = gh * gw;
    out.pixel_values.resize((size_t)N * 588);
    for (int row = 0; row < gh; ++row) {
        for (int col = 0; col < gw; ++col) {
            int t = row * gw + col;
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < kPatch; ++i) {
                    int y = row * kPatch + i;
                    for (int j = 0; j < kPatch; ++j) {
                        int x = col * kPatch + j;
                        out.pixel_values[(size_t)t * 588 + (size_t)c * 196 +
                                         (size_t)i * 14 + j] =
                            out.resized_chw[(size_t)c * plane + (size_t)y * target_w + x];
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace la
