#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <functional>
namespace la {
struct Box { float x1,y1,x2,y2; std::string label; };   // pixels
// Parse the generated token-id stream into boxes. Coord order x1,y1,x2,y2.
// coord_norm = id-151677; pixel = coord_norm/1000 * (img_w for x, img_h for y).
// label = the <ref>..</ref> text immediately preceding each <box>..</box>, carried
// forward across consecutive boxes until the next <ref>.
// decode_label: maps the label token-id span to text. Must byte-decode the GPT-2
// vocab pieces (e.g. la::Tokenizer::decode) so multi-word labels like
// "traffic light" round-trip correctly instead of "trafficĠlight".
std::vector<Box> parse_boxes(const std::vector<int32_t>& ids, int img_w, int img_h,
                             const std::function<std::string(const std::vector<int32_t>&)>& decode_label);
}
