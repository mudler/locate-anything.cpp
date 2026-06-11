#pragma once
#include "tokenizer.hpp"
#include <vector>
#include <string>
#include <cstdint>
namespace la {
// Build the full prompt input_ids reproducing the HF processor (chat template +
// image-token expansion). N = (gh/2)*(gw/2) <IMG_CONTEXT> tokens.
std::vector<int32_t> build_prompt(const Tokenizer& tok, int gh, int gw, const std::string& query);
}
