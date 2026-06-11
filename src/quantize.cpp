// M6b Task 4 — GGUF quantization (LM matmul weights only).
//
// Mirrors rt-detr.cpp examples/cli/main.cpp cmd_quantize for the
// dequant -> ggml_quantize_chunk -> owned-buffer -> gguf_add_tensor ->
// gguf_write_to_file mechanism.
//
// CRITICAL: only the LM matmul weights consumed via ggml_mul_mat are
// quantized. The C++ reads two tensors' raw ->data as f32 on the host:
//   - lm.tok_embd.weight  (host row-gather in embed_tokens_host)
//   - vit.pos_emb.weight  (host bicubic in build_patch_pos)
// Those MUST stay f32 or parity breaks. Everything that isn't an LM
// matmul weight (vit.*, proj.*, norms, biases, embeddings) is copied
// through unchanged.
#include "quantize.hpp"
#include "ggml.h"
#include "gguf.h"
#include <regex>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdint>

namespace la {
namespace {

// String -> ggml_type. Accepts upper/lowercase. Returns false on unknown.
bool type_from_string(const std::string& s_in, ggml_type& out) {
    std::string s = s_in;
    for (auto& c : s) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    if (s == "f16")  { out = GGML_TYPE_F16;  return true; }
    if (s == "q8_0") { out = GGML_TYPE_Q8_0; return true; }
    if (s == "q6_k") { out = GGML_TYPE_Q6_K; return true; }
    if (s == "q5_k") { out = GGML_TYPE_Q5_K; return true; }
    if (s == "q4_k") { out = GGML_TYPE_Q4_K; return true; }
    return false;
}

// Only LM matmul weights consumed by ggml_mul_mat. NOTHING else.
bool should_quantize(const std::string& name) {
    static const std::regex re(
        "^lm\\.blk\\.[0-9]+\\.(attn_q|attn_k|attn_v|attn_o|ffn_gate|ffn_up|ffn_down)\\.weight$");
    return std::regex_match(name, re) || name == "lm.output.weight";
}

// Dequantize src (F32 or F16) to f32. Returns false if unsupported.
bool dequantize_to_f32(const ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.assign((size_t)n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(t->data), out.data(), n);
        return true;
    }
    const ggml_type_traits* tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) return false;
    tr->to_float(t->data, out.data(), n);
    return true;
}

} // namespace

bool quantize_gguf(const std::string& in_gguf, const std::string& out_gguf, const std::string& type) {
    ggml_type target;
    if (!type_from_string(type, target)) {
        std::fprintf(stderr, "quantize: unknown type '%s' (expected f16/q8_0/q6_k/q5_k/q4_k)\n",
                     type.c_str());
        return false;
    }

    // Open input. no_alloc=false so the gguf init owns a ggml_context holding
    // tensor data we can read.
    ggml_context* meta_ctx = nullptr;
    gguf_init_params ip{ /*no_alloc=*/false, /*ctx=*/&meta_ctx };
    gguf_context* in = gguf_init_from_file(in_gguf.c_str(), ip);
    if (!in || !meta_ctx) {
        std::fprintf(stderr, "quantize: failed to open input '%s'\n", in_gguf.c_str());
        if (in) gguf_free(in);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    // Output gguf + copy ALL metadata KVs verbatim.
    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, in);

    // Scratch ctx to hold rewritten tensor descriptors (headers only).
    // NOTE: iterate via the gguf tensor index (NOT ggml_get_first_tensor over
    // meta_ctx). With no_alloc=false, ggml allocates a single backing tensor
    // named "GGUF tensor data binary blob" that holds all data, with the named
    // tensors as views into it — walking the ggml ctx would sweep up that blob.
    const int64_t n_tensors = gguf_get_n_tensors(in);
    ggml_init_params ep{};
    ep.mem_size   = ggml_tensor_overhead() * (size_t)(n_tensors + 8);
    ep.mem_buffer = nullptr;
    ep.no_alloc   = true;
    ggml_context* out_ctx = ggml_init(ep);
    if (!out_ctx) {
        std::fprintf(stderr, "quantize: ggml_init for out_ctx failed\n");
        gguf_free(out); gguf_free(in); ggml_free(meta_ctx);
        return false;
    }

    const bool is_kquant = (target == GGML_TYPE_Q4_K || target == GGML_TYPE_Q5_K ||
                            target == GGML_TYPE_Q6_K);
    ggml_quantize_init(target);
    if (is_kquant) ggml_quantize_init(GGML_TYPE_Q8_0);

    // Owned byte buffers backing each new tensor's data ptr; must outlive the
    // gguf_write_to_file call below.
    std::vector<std::vector<uint8_t>> owners;
    owners.reserve((size_t)n_tensors);

    std::vector<float> f32_buf;
    int n_quant = 0, n_kept = 0, n_fallback_q8 = 0;
    bool failed = false;

    for (int64_t ti = 0; ti < n_tensors && !failed; ++ti) {
        const char* name = gguf_get_tensor_name(in, ti);
        ggml_tensor* src = ggml_get_tensor(meta_ctx, name);
        if (!src || !src->data) {
            std::fprintf(stderr, "quantize: tensor '%s' has no data\n", name);
            failed = true; break;
        }

        // Decide target type for this tensor.
        ggml_type used = src->type; // default: copy as-is
        bool quant = false;
        if (should_quantize(name) && ggml_n_dims(src) == 2) {
            if (src->ne[0] % 256 == 0) {
                used = target;       // K-quants need 256-divisible rows; LM = 2048/11008 -> OK
                quant = true;
            } else if (is_kquant && src->ne[0] % 32 == 0) {
                used = GGML_TYPE_Q8_0; // safety fallback (won't trigger for LM dims)
                quant = true; ++n_fallback_q8;
            } else if (!is_kquant && src->ne[0] % 32 == 0) {
                used = target;        // f16/q8_0 only need 32-divisible
                quant = true;
            }
            // else: keep f32
        }

        std::vector<uint8_t> bytes;
        const int64_t ne[GGML_MAX_DIMS] = { src->ne[0], src->ne[1], src->ne[2], src->ne[3] };
        ggml_tensor* dst = nullptr;

        if (quant) {
            if (!dequantize_to_f32(src, f32_buf)) {
                std::fprintf(stderr, "quantize: cannot dequantize '%s' (type=%s)\n",
                             name, ggml_type_name(src->type));
                failed = true; break;
            }
            const int64_t ne0   = src->ne[0];
            const int64_t nrows = ggml_nelements(src) / ne0;
            const size_t  qsz   = ggml_row_size(used, ne0) * (size_t)nrows;
            bytes.resize(qsz);
            const size_t got = ggml_quantize_chunk(used, f32_buf.data(), bytes.data(),
                                                   /*start=*/0, nrows, ne0, /*imatrix=*/nullptr);
            if (got != qsz) {
                std::fprintf(stderr, "quantize: chunk size mismatch '%s' got=%zu want=%zu\n",
                             name, got, qsz);
                failed = true; break;
            }
            dst = ggml_new_tensor(out_ctx, used, ggml_n_dims(src), ne);
            ggml_set_name(dst, name);
            ++n_quant;
        } else {
            // Pure copy: same type/shape/bytes.
            const size_t nb = ggml_nbytes(src);
            bytes.assign((const uint8_t*)src->data, (const uint8_t*)src->data + nb);
            dst = ggml_new_tensor(out_ctx, src->type, ggml_n_dims(src), ne);
            ggml_set_name(dst, name);
            ++n_kept;
        }

        // Wire the data pointer the gguf writer reads from during write.
        owners.emplace_back(std::move(bytes));
        dst->data = owners.back().data();
        gguf_add_tensor(out, dst);
    }

    bool ok = !failed;
    if (ok && !gguf_write_to_file(out, out_gguf.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "quantize: gguf_write_to_file failed for '%s'\n", out_gguf.c_str());
        ok = false;
    }

    if (ok) {
        std::fprintf(stderr, "quantize: %d quantized (%s), %d kept f32",
                     n_quant, ggml_type_name(target), n_kept);
        if (n_fallback_q8) std::fprintf(stderr, ", %d q8_0 fallback", n_fallback_q8);
        std::fprintf(stderr, "\n");
    }

    ggml_free(out_ctx);
    gguf_free(out);
    gguf_free(in);
    ggml_free(meta_ctx);
    return ok;
}

} // namespace la
