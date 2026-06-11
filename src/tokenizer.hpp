#pragma once
#include "model_loader.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

namespace la {

// Byte-level Qwen2 BPE tokenizer built from the GGUF tokenizer KV arrays.
// - Added/special tokens (token_type==4) are ATOMIC: matched (longest-first)
//   before any BPE so e.g. "<IMG_CONTEXT>" is always one id.
// - Plain-text runs are pre-tokenized with the Qwen2 GPT-2-style regex split,
//   byte-encoded (GPT-2 byte<->unicode map), then merged by BPE rank.
class Tokenizer {
public:
    bool load(const ModelLoader& ml);
    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    int32_t token_to_id(const std::string& piece) const;     // -1 if absent
    const std::vector<std::string>& id_to_piece() const { return id_to_piece_; }

private:
    // BPE a single byte-encoded word (string of mapped unicode chars) -> ids.
    void bpe_word(const std::string& word, std::vector<int32_t>& out) const;

    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string,int32_t> piece_to_id_;
    std::unordered_map<std::string,int> merge_rank_;          // "a b" -> rank
    std::vector<int> token_types_;
    std::vector<int32_t> special_ids_;                        // token_type==4

    // GPT-2 byte<->unicode maps.
    std::string byte_to_str_[256];                            // byte -> utf8 of mapped cpt
    std::unordered_map<uint32_t,uint8_t> cpt_to_byte_;        // mapped cpt -> byte

    // Atomic special-token matching.
    std::unordered_set<std::string> special_set_;
    std::vector<size_t> special_lens_;                        // distinct, descending
};

}  // namespace la
