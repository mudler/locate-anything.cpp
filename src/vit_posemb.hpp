#pragma once
#include <vector>
namespace la {
// Bicubic interpolation matching torch F.interpolate(mode="bicubic",
// align_corners=False, antialias=False), cubic coefficient a=-0.75.
// src is the base grid flattened row-major as [h*w, c] (c fastest), size base_h*base_w*c.
// Returns [gh*gw, c] flattened the same way (token = oy*gw+ox).
std::vector<float> bicubic_pos_emb(const std::vector<float>& src,
                                   int base_h, int base_w, int c, int gh, int gw);
}
