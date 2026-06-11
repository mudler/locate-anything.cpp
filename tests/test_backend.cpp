#include "backend.hpp"
#include <vector>
#include <cstdio>
// Build a trivial graph: out = a + b for two host-fed [4] inputs, check result.
int main() {
    la::Backend backend;
    std::vector<float> a = {1,2,3,4}, b = {10,20,30,40}, out;
    la::GraphInputPool pool;
    bool ok = backend.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* ta = backend.add_graph_input(ctx, pool, a.data(), a.size());
        ggml_tensor* tb = backend.add_graph_input(ctx, pool, b.data(), b.size());
        return ggml_add(ctx, ta, tb);
    }, out);
    if (!ok || out.size()!=4) return 1;
    int good = (out[0]==11 && out[1]==22 && out[2]==33 && out[3]==44);
    std::printf("backend add: %f %f %f %f ok=%d\n", out[0],out[1],out[2],out[3], good);
    return good ? 0 : 1;
}
