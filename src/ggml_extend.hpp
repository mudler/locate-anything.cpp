#pragma once
#include "ggml.h"

namespace la {

// y = W·x (+ bias). W is [in, out] in ggml ne (PyTorch [out,in] stored raw).
inline ggml_tensor* linear(ggml_context* ctx, ggml_tensor* W, ggml_tensor* x,
                           ggml_tensor* bias=nullptr) {
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

// LayerNorm over ne[0] with affine weight+bias (eps as in PyTorch nn.LayerNorm).
inline ggml_tensor* layernorm(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* n = ggml_norm(ctx, x, eps);  // zero-mean unit-var over ne[0]
    n = ggml_mul(ctx, n, w);
    n = ggml_add(ctx, n, b);
    return n;
}

// tanh-approximation GELU (matches PytorchGELUTanh / approximate="tanh").
// Built from primitives instead of ggml_gelu: this ggml is compiled with
// GGML_GELU_FP16, so ggml_gelu uses an fp16 lookup table that loses ~5e-4 of
// precision (e.g. gelu(2.0) off by ~4.8e-4). Computing it directly keeps full
// f32 precision for PyTorch parity. Formula:
//   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
inline ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x) {
    const float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    const float kCoeff       = 0.044715f;
    ggml_tensor* t = ggml_sqr(ctx, x);                 // x^2
    t = ggml_scale_bias(ctx, t, kCoeff, 1.0f);         // 1 + 0.044715*x^2
    t = ggml_mul(ctx, t, x);                           // x*(1 + 0.044715*x^2)
    t = ggml_scale(ctx, t, kSqrt2OverPi);              // sqrt(2/pi)*(x + 0.044715*x^3)
    t = ggml_tanh(ctx, t);                             // tanh(...)
    t = ggml_scale_bias(ctx, t, 0.5f, 0.5f);           // 0.5*(1 + tanh(...))
    return ggml_mul(ctx, t, x);                        // 0.5*x*(1 + tanh(...))
}

// RMSNorm over ne[0] with affine weight (Qwen2; eps in fp32). No bias, no mean-subtract.
inline ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Exact erf GELU (matches torch nn.GELU() default approximate='none'): 0.5*x*(1+erf(x/sqrt2)).
// For f32 tensors this ggml routes through ggml_vec_gelu_erf_f32, which calls erff() in full
// f32 (no fp16 lookup table), so it stays exact — verified in ggml-cpu/vec.h + ops.cpp.
inline ggml_tensor* gelu_erf(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu_erf(ctx, x);
}

}  // namespace la
