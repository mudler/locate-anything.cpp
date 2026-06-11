#include "pil_resize.hpp"

#include <cmath>
#include <vector>
#include <cstdint>

// Faithful port of Pillow's libImaging/Resample.c for the BICUBIC filter on
// 8-bit-per-channel RGB images. Two passes (horizontal then vertical). Unlike
// torch's bicubic (a = -0.75, no antialias), Pillow uses a = -0.5 and ALWAYS
// antialiases on downscale: the filter support is scaled by the reduction
// ratio max(1, in/out) per axis. We accumulate coefficients in double (matching
// Pillow's precompute_coeffs) and round to nearest uint8 with clamping after
// each pass (Pillow's intermediate image is 8bpc, so the round-trip through
// uint8 between passes is part of the reference behavior).

namespace la {

namespace {

// Pillow's cubic_filter with a = -0.5.
inline double cubic_filter(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

constexpr double kBicubicSupport = 2.0;

struct Coeffs {
    int ksize = 0;
    std::vector<int> bounds;   // per output pixel: [xmin, xsize]
    std::vector<double> kk;    // per output pixel: ksize weights
};

// Mirrors Pillow's precompute_coeffs for a full-extent box (in0=0, in1=inSize).
Coeffs precompute_coeffs(int in_size, int out_size) {
    double scale = (double)in_size / (double)out_size;
    double filterscale = scale;
    if (filterscale < 1.0) filterscale = 1.0;

    double support = kBicubicSupport * filterscale;
    int ksize = (int)std::ceil(support) * 2 + 1;

    Coeffs c;
    c.ksize = ksize;
    c.bounds.assign((size_t)out_size * 2, 0);
    c.kk.assign((size_t)out_size * ksize, 0.0);

    for (int xx = 0; xx < out_size; ++xx) {
        double center = (xx + 0.5) * scale;  // in0 == 0
        double ww = 0.0;
        double ss = 1.0 / filterscale;

        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;

        double* k = &c.kk[(size_t)xx * ksize];
        int x = 0;
        for (; x < xmax; ++x) {
            double w = cubic_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x) {
            if (ww != 0.0) k[x] /= ww;
        }
        for (; x < ksize; ++x) {
            k[x] = 0.0;
        }
        c.bounds[(size_t)xx * 2 + 0] = xmin;
        c.bounds[(size_t)xx * 2 + 1] = xmax;
    }
    return c;
}

inline uint8_t clip8(double v) {
    // Round to nearest, clamp to [0,255]. Pillow uses fixed-point with a
    // 1<<(PRECISION-1) bias which is round-half-up; std::lround matches for
    // the non-negative magnitudes that occur after coefficient normalization.
    long r = std::lround(v);
    if (r < 0) return 0;
    if (r > 255) return 255;
    return (uint8_t)r;
}

}  // namespace

std::vector<uint8_t> pil_bicubic_resize(const std::vector<uint8_t>& src,
                                        int sw, int sh, int dw, int dh) {
    constexpr int C = 3;

    // ---- Horizontal pass: (sw, sh) -> (dw, sh) ----
    Coeffs hc = precompute_coeffs(sw, dw);
    std::vector<uint8_t> tmp((size_t)dw * sh * C);
    for (int y = 0; y < sh; ++y) {
        const uint8_t* row = &src[(size_t)y * sw * C];
        for (int xx = 0; xx < dw; ++xx) {
            int xmin = hc.bounds[(size_t)xx * 2 + 0];
            int xsize = hc.bounds[(size_t)xx * 2 + 1];
            const double* k = &hc.kk[(size_t)xx * hc.ksize];
            double s0 = 0.0, s1 = 0.0, s2 = 0.0;
            for (int i = 0; i < xsize; ++i) {
                const uint8_t* p = &row[(size_t)(xmin + i) * C];
                s0 += p[0] * k[i];
                s1 += p[1] * k[i];
                s2 += p[2] * k[i];
            }
            uint8_t* o = &tmp[((size_t)y * dw + xx) * C];
            o[0] = clip8(s0);
            o[1] = clip8(s1);
            o[2] = clip8(s2);
        }
    }

    // ---- Vertical pass: (dw, sh) -> (dw, dh) ----
    Coeffs vc = precompute_coeffs(sh, dh);
    std::vector<uint8_t> out((size_t)dw * dh * C);
    for (int yy = 0; yy < dh; ++yy) {
        int ymin = vc.bounds[(size_t)yy * 2 + 0];
        int ysize = vc.bounds[(size_t)yy * 2 + 1];
        const double* k = &vc.kk[(size_t)yy * vc.ksize];
        for (int x = 0; x < dw; ++x) {
            double s0 = 0.0, s1 = 0.0, s2 = 0.0;
            for (int i = 0; i < ysize; ++i) {
                const uint8_t* p = &tmp[((size_t)(ymin + i) * dw + x) * C];
                s0 += p[0] * k[i];
                s1 += p[1] * k[i];
                s2 += p[2] * k[i];
            }
            uint8_t* o = &out[((size_t)yy * dw + x) * C];
            o[0] = clip8(s0);
            o[1] = clip8(s1);
            o[2] = clip8(s2);
        }
    }
    return out;
}

}  // namespace la
