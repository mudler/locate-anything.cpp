#pragma once
#include <string>
namespace la {
// Quantize the LM matmul weights of in_gguf to `type` (f16/q8_0/q6_k/q5_k/q4_k);
// keep everything else f32. Returns false on failure.
bool quantize_gguf(const std::string& in_gguf, const std::string& out_gguf, const std::string& type);
}
