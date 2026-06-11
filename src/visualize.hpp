#pragma once
#include "image_io.hpp"
#include "boxes.hpp"
#include <vector>
namespace la {
// Draw each box (colored stroke + label text on a colored background) onto a COPY of img; return it.
Image render_boxes(const Image& img, const std::vector<Box>& boxes);
}
