#include "backend.hpp"
#include "vit_rope.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
int main(){
    const char* fix = std::getenv("LA_TEST_FIXTURES");
    if (!fix) { std::fprintf(stderr,"skip\n"); return 77; }
    std::vector<float> qin, qout; std::vector<int64_t> s;
    if (!la_parity::load_baseline(fix,"rope_q_in",qin,s)) return 1;     // [1024,16,72]
    if (!la_parity::load_baseline(fix,"rope_q_out",qout,s)) return 1;
    la::RopeTables rt = la::build_rope_tables(/*H*/32,/*W*/32,/*heads*/16,/*head_dim*/72,/*theta*/10000.f);
    la::Backend be; la::GraphInputPool pool;
    std::vector<float> got;
    be.compute([&](ggml_context* ctx)->ggml_tensor*{
        // q input laid out [head_dim, heads, tokens] = ne {72,16,1024}.
        // qin fixture is [1024,16,72] flat = token*16*72 + head*72 + d, i.e. d fastest,
        // head next, token outer -> EXACTLY ne {72,16,1024}. Feed directly.
        const int64_t q_ne[3] = {72,16,1024};
        ggml_tensor* q = be.add_graph_input_nd(ctx,pool,qin.data(),q_ne,3);
        return la::apply_rope(ctx, be, pool, q, rt);
    }, got);
    bool ok = la::compare_rope(got, qout, /*tokens*/1024,/*heads*/16,/*hd*/72, 1e-4f);
    std::printf("rope ok=%d\n", ok);
    return ok ? 0 : 1;
}
