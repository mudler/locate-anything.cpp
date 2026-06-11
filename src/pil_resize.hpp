#pragma once
#include <vector>
#include <cstdint>
namespace la {
// Pillow-equivalent bicubic resize, RGB uint8 HWC (sw,sh) -> (dw,dh). a=-0.5, two-pass,
// antialias on downscale (support scaled by reduction ratio). Matches PIL.Image.resize(BICUBIC).
std::vector<uint8_t> pil_bicubic_resize(const std::vector<uint8_t>& src, int sw, int sh, int dw, int dh);
}
