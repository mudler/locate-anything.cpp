#include "vit_rope.hpp"
#include "backend.hpp"
#include <cmath>
#include <cstdio>
namespace la {
RopeTables build_rope_tables(int H, int W, int heads, int head_dim, float theta){
    RopeTables rt; rt.tokens=H*W; rt.n_pairs=head_dim/2; rt.heads=heads; rt.head_dim=head_dim;
    rt.cos.resize((size_t)rt.tokens*rt.n_pairs);
    rt.sin.resize((size_t)rt.tokens*rt.n_pairs);
    for (int h=0;h<H;++h) for (int w=0;w<W;++w){
        int tok = h*W + w;
        for (int k=0;k<rt.n_pairs;++k){
            int i = k/2;
            float invf = std::pow(theta, -4.f*(float)i/(float)head_dim);
            // Even pair -> column(w), odd pair -> row(h). This follows the model's
            // actual code (cat([x_cis, y_cis]) in Rope2DPosEmb); the modeling_vit.py
            // docstring claims the opposite mapping and is WRONG — do not trust it.
            float coord = (k%2==0) ? (float)w : (float)h;
            float ang = coord * invf;
            rt.cos[(size_t)tok*rt.n_pairs + k] = std::cos(ang);
            rt.sin[(size_t)tok*rt.n_pairs + k] = std::sin(ang);
        }
    }
    return rt;
}
void build_rope_inputs(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                       const RopeTables& rt, ggml_tensor*& cosb, ggml_tensor*& sinb){
    const int64_t cs_ne[4] = {1, rt.n_pairs, 1, rt.tokens};
    cosb = be.add_graph_input_nd(ctx, pool, rt.cos.data(), cs_ne, 4);
    sinb = be.add_graph_input_nd(ctx, pool, rt.sin.data(), cs_ne, 4);
}
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* cosb, ggml_tensor* sinb, const RopeTables& rt){
    const int hd=rt.head_dim, np=rt.n_pairs, heads=rt.heads, tok=rt.tokens;
    ggml_tensor* xr = ggml_reshape_4d(ctx, x, 2, np, heads, tok);          // [2,np,heads,tok]
    ggml_tensor* x0 = ggml_cont(ctx, ggml_view_4d(ctx, xr, 1,np,heads,tok,
                          xr->nb[1],xr->nb[2],xr->nb[3], 0));               // even comp
    ggml_tensor* x1 = ggml_cont(ctx, ggml_view_4d(ctx, xr, 1,np,heads,tok,
                          xr->nb[1],xr->nb[2],xr->nb[3], xr->nb[0]));       // odd comp
    ggml_tensor* rot = ggml_concat(ctx, ggml_neg(ctx, x1), x0, 0);         // [-x1, x0]
    ggml_tensor* out = ggml_add(ctx, ggml_mul(ctx, xr, cosb), ggml_mul(ctx, rot, sinb));
    return ggml_reshape_3d(ctx, ggml_cont(ctx, out), hd, heads, tok);
}
ggml_tensor* apply_rope(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                        ggml_tensor* x, const RopeTables& rt){
    ggml_tensor* cosb; ggml_tensor* sinb;
    build_rope_inputs(ctx, be, pool, rt, cosb, sinb);
    return apply_rope(ctx, x, cosb, sinb, rt);
}
bool compare_rope(const std::vector<float>& got, const std::vector<float>& ref,
                  int tokens, int heads, int hd, float tol){
    // both got and ref are flat [tok][head][d] (d fastest) here.
    (void)tokens; (void)heads; (void)hd;
    double mx=0;
    for (size_t i=0;i<ref.size();++i){ double dd=std::fabs((double)got[i]-(double)ref[i]); if(dd>mx)mx=dd; }
    std::fprintf(stderr,"[rope] max-abs-diff=%.3e tol=%.3e\n", mx, (double)tol);
    return mx < (double)tol;
}
}
