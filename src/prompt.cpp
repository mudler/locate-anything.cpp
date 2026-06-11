#include "prompt.hpp"
namespace la {
// Reproduces the HF chat template (system turn auto-prepended; first turn is the
// user turn) followed by replace_media_placeholder, which expands the
// "<image-1>" placeholder into "<image 1><img>{<IMG_CONTEXT> * N}</img>".
// add_vision_id defaults to False, so the chat template emits only "<image-1>".
// message content order is [image, text], so the query follows the image block.
std::vector<int32_t> build_prompt(const Tokenizer& tok, int gh, int gw, const std::string& query){
    int N = (gh/2)*(gw/2);
    std::string s;
    s += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    s += "<|im_start|>user\n";
    s += "<image 1><img>";
    for(int i=0;i<N;++i) s += "<IMG_CONTEXT>";
    s += "</img>";
    s += query;
    s += "<|im_end|>\n<|im_start|>assistant\n";
    return tok.encode(s);
}
}
