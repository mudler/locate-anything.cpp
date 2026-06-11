#pragma once
#include <vector>
#include <string>
#include <cstdint>
namespace la {

struct Image { int w = 0, h = 0; std::vector<uint8_t> rgb; };  // HWC RGB uint8

// Load an image as HWC RGB uint8 (stb, forced to 3 channels). Returns false on failure.
bool load_image_rgb(const std::string& path, Image& out);

// Load an image from an in-memory encoded buffer (PNG/JPEG/...) as HWC RGB uint8. Returns false on failure.
bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out);

// Save an HWC RGB uint8 image as a PNG. Returns false on failure.
bool save_image_png(const std::string& path, const Image& img);

// Compute the LocateAnything rescale target size (the (A)/(B)/(C) arithmetic).
void preproc_target(int w0, int h0, int& target_w, int& target_h);

struct Preprocessed {
    std::vector<float> pixel_values;  // [N*588] patch-major: pv[t*588 + c*196 + i*14 + j]
    int gh = 0, gw = 0;
    // Normalized [3, target_h, target_w] CHW after PIL-resize + normalize, before
    // patchify. Exposed to gate the resampler in isolation.
    std::vector<float> resized_chw;
    int target_h = 0, target_w = 0;
};

// Full pipeline: PIL-bicubic resize -> normalize (x/127.5 - 1) -> patchify.
bool preprocess(const Image& img, Preprocessed& out);

}  // namespace la
