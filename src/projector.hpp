#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include <vector>
namespace la {
class Projector {
public:
    Projector(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}
    // merged_host: [N*4608] flat token-major (token outer, channel inner), N inferred
    // from merged_host.size()/4608 (256 for the 448 fixture, variable otherwise).
    // out: [2048*N] flat = ggml [2048,N] (hidden fastest, token next).
    bool project(const std::vector<float>& merged_host, std::vector<float>& out);
    // Graph-builder variant: project an in-graph [4608, n] tensor -> [2048, n].
    ggml_tensor* build(ggml_context* ctx, ggml_tensor* merged) const;
private:
    ModelLoader& ml_; Backend& be_;
};
}
