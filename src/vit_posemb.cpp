#include "vit_posemb.hpp"
#include <cmath>
namespace la {
static inline float cubic_w(float t, float a){
    t = std::fabs(t);
    if (t <= 1.f) return ((a+2.f)*t - (a+3.f))*t*t + 1.f;
    if (t <  2.f) return (((t - 5.f)*t + 8.f)*t - 4.f)*a;
    return 0.f;
}
static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
std::vector<float> bicubic_pos_emb(const std::vector<float>& src,
                                   int base_h, int base_w, int c, int gh, int gw){
    const float a = -0.75f;
    const float sh = (float)base_h/(float)gh, sw = (float)base_w/(float)gw;
    std::vector<float> out((size_t)gh*gw*c, 0.f);
    auto at = [&](int y,int x,int ch)->float{
        y=clampi(y,0,base_h-1); x=clampi(x,0,base_w-1);
        return src[((size_t)y*base_w+x)*c+ch];
    };
    for (int oy=0;oy<gh;++oy){
        float fy=(oy+0.5f)*sh-0.5f; int iy=(int)std::floor(fy); float ty=fy-iy;
        float wy[4]={cubic_w(1.f+ty,a),cubic_w(ty,a),cubic_w(1.f-ty,a),cubic_w(2.f-ty,a)};
        for (int ox=0;ox<gw;++ox){
            float fx=(ox+0.5f)*sw-0.5f; int ix=(int)std::floor(fx); float tx=fx-ix;
            float wx[4]={cubic_w(1.f+tx,a),cubic_w(tx,a),cubic_w(1.f-tx,a),cubic_w(2.f-tx,a)};
            for (int ch=0;ch<c;++ch){
                float acc=0.f;
                for (int m=0;m<4;++m){ float row=0.f;
                    for (int n=0;n<4;++n) row+=wx[n]*at(iy-1+m, ix-1+n, ch);
                    acc+=wy[m]*row; }
                out[((size_t)oy*gw+ox)*c+ch]=acc;
            }
        }
    }
    return out;
}
}
