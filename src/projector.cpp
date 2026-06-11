#include "projector.hpp"
#include "ggml_extend.hpp"
namespace la {
ggml_tensor* Projector::build(ggml_context* ctx, ggml_tensor* merged) const {
    // merged: [4608, n_tokens]. mlp1: LayerNorm(4608,eps1e-5) -> Linear -> gelu_erf -> Linear.
    ggml_tensor* x = la::layernorm(ctx, merged, ml_.tensor("proj.0.weight"),
                                   ml_.tensor("proj.0.bias"), 1e-5f);
    x = la::linear(ctx, ml_.tensor("proj.1.weight"), x, ml_.tensor("proj.1.bias")); // [2048,n]
    x = la::gelu_erf(ctx, x);
    x = la::linear(ctx, ml_.tensor("proj.3.weight"), x, ml_.tensor("proj.3.bias")); // [2048,n]
    return x;
}
bool Projector::project(const std::vector<float>& merged_host, std::vector<float>& out){
    const int c = 4608;
    const int n = (int)(merged_host.size()/(size_t)c);   // N = (gh/2)*(gw/2); 256 for the 448 fixture
    GraphInputPool pool;
    return be_.compute([&](ggml_context* ctx)->ggml_tensor*{
        const int64_t ne[2] = {c, n};                 // [4608, 256]
        ggml_tensor* m = be_.add_graph_input_nd(ctx, pool, merged_host.data(), ne, 2);
        return build(ctx, m);
    }, out);
}
}
