#pragma once
#include "ggml.h"
#include <vector>
namespace la {
class Backend; struct GraphInputPool;
struct RopeTables {
    std::vector<float> cos;   // size tokens*n_pairs, layout [token][pair] (pair fastest)
    std::vector<float> sin;
    int tokens=0, n_pairs=0, heads=0, head_dim=0;
};
RopeTables build_rope_tables(int H, int W, int heads, int head_dim, float theta);
// Apply interleaved-pair rotation to x [head_dim, heads, tokens]; returns same shape.
ggml_tensor* apply_rope(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                        ggml_tensor* x, const RopeTables& rt);
// Build the [1,n_pairs,1,tokens] cos/sin graph-input leaves ONCE so they can be
// reused across all encoder blocks (avoids re-copying identical tables per block).
void build_rope_inputs(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                       const RopeTables& rt, ggml_tensor*& cosb, ggml_tensor*& sinb);
// Core rotation using pre-fed cos/sin inputs (see build_rope_inputs).
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* cosb, ggml_tensor* sinb, const RopeTables& rt);
// Compare ggml flat output vs fixture flat [tok,heads,hd]; tol max-abs-diff.
bool compare_rope(const std::vector<float>& got, const std::vector<float>& ref,
                  int tokens, int heads, int hd, float tol);
}
