#include "backend.hpp"
#include "ggml_extend.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

static float gelu_tanh_ref(float x){
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f*x*(1.f+std::tanh(k*(x+0.044715f*x*x*x)));
}

int main() {
    la::Backend be;
    la::GraphInputPool pool;
    // layernorm over a length-4 vector with weight=1,bias=0, eps=1e-5 -> standardized
    std::vector<float> x = {1,2,3,4}, w = {1,1,1,1}, b = {0,0,0,0}, out;
    be.compute([&](ggml_context* ctx)->ggml_tensor*{
        ggml_tensor* tx = be.add_graph_input(ctx,pool,x.data(),4);
        ggml_tensor* tw = be.add_graph_input(ctx,pool,w.data(),4);
        ggml_tensor* tb = be.add_graph_input(ctx,pool,b.data(),4);
        return la::layernorm(ctx, tx, tw, tb, 1e-5f);
    }, out);
    // mean=2.5, var=1.25, std=sqrt(1.25); (1-2.5)/sqrt(1.25+1e-5)
    float exp0 = (1.f-2.5f)/std::sqrt(1.25f+1e-5f);
    int ln_ok = std::fabs(out[0]-exp0) < 1e-4f;

    std::vector<float> g = {-1.f, 0.f, 0.5f, 2.f}, gout;
    be.compute([&](ggml_context* ctx)->ggml_tensor*{
        ggml_tensor* tg = be.add_graph_input(ctx,pool,g.data(),4);
        return la::gelu_tanh(ctx, tg);
    }, gout);
    int gelu_ok = 1;
    for (int i=0;i<4;i++) gelu_ok &= (std::fabs(gout[i]-gelu_tanh_ref(g[i])) < 1e-4f);
    std::printf("ln_ok=%d gelu_ok=%d\n", ln_ok, gelu_ok);
    return (ln_ok && gelu_ok) ? 0 : 1;
}
