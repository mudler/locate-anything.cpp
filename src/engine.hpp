#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "tokenizer.hpp"
#include "boxes.hpp"
#include <memory>
#include <string>
#include <vector>
namespace la {
struct Image;  // image_io.hpp
class Engine {
public:
    enum class Mode { Hybrid, Slow, Fast };
    static std::unique_ptr<Engine> load(const std::string& gguf_path, int n_threads = 0);
    std::vector<Box> locate(const std::string& image_path, const std::string& query,
                            Mode mode = Mode::Hybrid, int max_new = 256);
    std::vector<Box> locate_buffer(const unsigned char* bytes, size_t len,
                                   const std::string& query,
                                   Mode mode = Mode::Hybrid, int max_new = 256);
    Tokenizer& tokenizer() { return tok_; }
private:
    Engine() = default;
    std::vector<Box> locate_image(const Image& img, const std::string& query,
                                  Mode mode, int max_new);
    ModelLoader ml_; Backend be_; Tokenizer tok_;
};
}
