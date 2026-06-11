#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace la {

namespace {
// Number of graph nodes the metadata context must hold. Ample for the MoonViT
// tower; the per-compute metadata context + graph hash-set scale with this, so
// keep it as small as the largest single graph needs.
constexpr size_t kGraphSize = 16384;

struct PendingInput {
    ggml_tensor* tensor;
    const float* host;
    size_t       nbytes;
};
// Extra graph tensors to read back after compute (besides the final output).
struct PendingCapture {
    ggml_tensor*        tensor;
    std::vector<float>* dst;
};
}  // namespace

struct Backend::Impl {
    ggml_backend_t       backend     = nullptr;  // primary device (GPU or CPU)
    ggml_backend_t       cpu_backend = nullptr;  // CPU fallback (GPU path only)
    ggml_gallocr_t       galloc      = nullptr;  // CPU / single-backend path (unchanged)
    ggml_backend_sched_t sched       = nullptr;  // GPU path: schedules over {backend, cpu_backend}
    bool                 use_sched   = false;    // true only when `backend` is a GPU device
    // Inputs registered by the build lambda for the in-flight compute. Copied
    // into the allocated tensors after the graph is allocated, then cleared.
    // Never overlaps across calls (compute is not re-entrant).
    std::vector<PendingInput>   pending;
    // Extra tensors to read back after compute (registered via capture()).
    std::vector<PendingCapture> captures;
    // Extra graph roots to expand (registered via add_graph_root()) but NOT read
    // back. Used for resident K/V write (ggml_cpy) nodes.
    std::vector<ggml_tensor*>   roots;
};

Backend::Backend() : impl_(new Impl()) {
    // Optional override via LA_DEVICE: "cpu" forces CPU; a device name selects
    // that registry device (case-insensitive); unset auto-picks a GPU/IGPU.
    const char* force = std::getenv("LA_DEVICE");
    const std::string want = force ? force : "";
    const bool force_cpu = want == "cpu" || want == "CPU";

    // Case-insensitive equality, used to match LA_DEVICE against the registry's
    // device names (upper-case like "CUDA0"/"Vulkan0").
    auto iequals = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
                return false;
        return true;
    };

    if (!force_cpu) {
        // Walk the registry. Whatever backend was compiled in
        // (CUDA/Metal/Vulkan/HIP/SYCL) registers itself here, so this single path
        // covers them all with no backend-specific includes. Integrated GPUs
        // report GGML_BACKEND_DEVICE_TYPE_IGPU and are eligible too. When
        // LA_DEVICE names a device, match by name; otherwise pick the first
        // GPU/IGPU device.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(dev);
            const char* name = ggml_backend_dev_name(dev);

            bool selected;
            if (!want.empty()) {
                selected = name && iequals(want, name);  // explicit name match
            } else {
                selected = type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                           type == GGML_BACKEND_DEVICE_TYPE_IGPU;
            }
            if (!selected) continue;

            impl_->backend = ggml_backend_dev_init(dev, nullptr);
            if (impl_->backend) {
                device_name_ = name ? name : "";
                // Route compute through ggml_backend_sched for any non-CPU device
                // so unsupported ops can fall back to CPU.
                impl_->use_sched = type != GGML_BACKEND_DEVICE_TYPE_CPU;
                offloading_      = impl_->use_sched;
                LA_LOG("la::Backend using device: %s", device_name_.c_str());
                break;
            }
        }
        if (!want.empty() && !impl_->backend)
            LA_LOG("la::Backend: LA_DEVICE=%s not found; falling back to CPU",
                   want.c_str());
    }
    if (!impl_->backend) {              // CPU fallback (or CPU-only build)
        impl_->backend = ggml_backend_cpu_init();
        device_name_ = "cpu";
        offloading_  = false;
        if (!impl_->backend) {
            LA_LOG("la::Backend: ggml_backend_cpu_init failed");
            return;
        }
    }
    // GPU path: create a CPU fallback backend so ops the device lacks a kernel
    // for are offloaded to CPU by the scheduler instead of aborting. The
    // CPU/single-backend path keeps using the persistent gallocr and is
    // untouched.
    if (impl_->use_sched) {
        impl_->cpu_backend = ggml_backend_cpu_init();
        if (!impl_->cpu_backend) {
            LA_LOG("la::Backend: CPU fallback init failed; disabling sched");
            impl_->use_sched = false;
        }
    }
    set_n_threads(n_threads_);
}

Backend::~Backend() {
    if (impl_) {
        // Free allocators/scheduler BEFORE the backends they reference.
        if (impl_->sched)       ggml_backend_sched_free(impl_->sched);
        if (impl_->galloc)      ggml_gallocr_free(impl_->galloc);
        if (impl_->cpu_backend) ggml_backend_free(impl_->cpu_backend);
        if (impl_->backend)     ggml_backend_free(impl_->backend);
    }
}

void Backend::set_n_threads(int n) {
    n_threads_ = n > 0 ? n : 1;
    // Only the CPU backend(s) take a thread count; GPU backends ignore it.
    if (impl_ && impl_->backend && ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
    if (impl_ && impl_->cpu_backend) {
        ggml_backend_cpu_set_n_threads(impl_->cpu_backend, n_threads_);
    }
}

ggml_backend_t Backend::handle() const {
    return impl_ ? impl_->backend : nullptr;
}

ggml_tensor* Backend::add_graph_input(ggml_context* ctx, GraphInputPool& pool,
                                      const float* host, size_t n) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)n);
    ggml_set_input(t);
    impl_->pending.push_back({t, pool.keep(host, n), n * sizeof(float)});
    return t;
}

ggml_tensor* Backend::add_graph_input_nd(ggml_context* ctx, GraphInputPool& pool,
                                         const float* host, const int64_t* ne,
                                         int n_dims) {
    size_t total = 1;
    for (int i = 0; i < n_dims; ++i) total *= (size_t)ne[i];
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    ggml_set_input(t);
    impl_->pending.push_back({t, pool.keep(host, total), total * sizeof(float)});
    return t;
}

ggml_tensor* Backend::add_int32_input_nd(ggml_context* ctx, GraphInputPool& pool,
                                         const int32_t* host, const int64_t* ne,
                                         int n_dims) {
    size_t total = 1;
    for (int i = 0; i < n_dims; ++i) total *= (size_t)ne[i];
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_I32, n_dims, ne);
    ggml_set_input(t);
    // The upload is a raw, type-agnostic ggml_backend_tensor_set; reuse the same
    // pending-upload list (host pointer reinterpreted as bytes).
    impl_->pending.push_back({t, (const float*)pool.keep_i32(host, total),
                              total * sizeof(int32_t)});
    return t;
}

void Backend::capture(ggml_tensor* t, std::vector<float>* dst) {
    impl_->captures.push_back({t, dst});
}

ggml_backend_buffer_t Backend::allocate_ctx_tensors(ggml_context* ctx) {
    return ggml_backend_alloc_ctx_tensors(ctx, impl_->backend);
}

void Backend::add_graph_root(ggml_tensor* t) {
    impl_->roots.push_back(t);
}

bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                      std::vector<float>& out) {
    if (!impl_ || !impl_->backend) {
        LA_LOG("Backend::compute called on an uninitialised backend");
        return false;
    }

    // Metadata-only context: holds graph + tensor structs, no tensor data
    // (no_alloc=true). Tensor data lives in the gallocr's persistent buffer.
    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * kGraphSize +
                            ggml_graph_overhead_custom(kGraphSize, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LA_LOG("Backend::compute: ggml_init failed");
        return false;
    }

    // Reset registrations for this compute, then run the build lambda. The build
    // lambda registers host inputs / captures via member calls on this Backend.
    impl_->pending.clear();
    impl_->captures.clear();
    impl_->roots.clear();
    struct ggml_tensor* output = build(ctx);
    if (!output) {
        LA_LOG("Backend::compute: build() returned null output tensor");
        impl_->pending.clear();
        impl_->captures.clear();
        impl_->roots.clear();
        ggml_free(ctx);
        return false;
    }

    // Mark the output (and any captured tensors) so the gallocr does not recycle
    // their storage before we read them back, then expand the forward graph.
    ggml_set_output(output);
    for (const PendingCapture& pc : impl_->captures) ggml_set_output(pc.tensor);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    // Expand captures FIRST so they are present in the graph even if the final
    // output's subgraph does not reach them.
    for (const PendingCapture& pc : impl_->captures)
        ggml_build_forward_expand(gf, pc.tensor);
    // Expand registered extra graph roots (e.g. resident K/V ggml_cpy writes) in
    // registration order, BEFORE the output, so their side-effect nodes land in
    // the graph. These are not read back.
    for (ggml_tensor* r : impl_->roots)
        ggml_build_forward_expand(gf, r);
    ggml_build_forward_expand(gf, output);

    // GPU devices default to the fast persistent-gallocr path (identical to a
    // single-backend run). Only route THIS graph through the scheduler (which
    // offloads unsupported ops to CPU) when the GPU backend actually lacks a
    // kernel for one of its ops. The per-graph check is a cheap O(nodes) scan,
    // far less than a sched re-plan. The CPU path never sets use_sched, so it is
    // byte-for-byte unchanged.
    bool need_sched = false;
    if (impl_->use_sched) {
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            if (!ggml_backend_supports_op(impl_->backend, ggml_graph_node(gf, i))) {
                need_sched = true;
                break;
            }
        }
    }

    bool alloc_ok = false;
    if (need_sched) {
        // GPU path: schedule across {GPU, CPU}. Unsupported ops fall back to CPU.
        if (!impl_->sched) {
            ggml_backend_t backs[2] = { impl_->backend, impl_->cpu_backend };
            impl_->sched = ggml_backend_sched_new(
                backs, /*bufts=*/nullptr, /*n_backends=*/2,
                /*graph_size=*/kGraphSize, /*parallel=*/false, /*op_offload=*/true);
            if (!impl_->sched) {
                LA_LOG("Backend::compute: ggml_backend_sched_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                impl_->roots.clear();
                ggml_free(ctx);
                return false;
            }
        }
        ggml_backend_sched_reset(impl_->sched);
        alloc_ok = ggml_backend_sched_alloc_graph(impl_->sched, gf);
        if (!alloc_ok) LA_LOG("Backend::compute: ggml_backend_sched_alloc_graph failed");
    } else {
        // Fast path: CPU, or a GPU whose backend supports every op in this graph.
        // Persistent gallocr over the active backend's buffer type, lazily created
        // and reused on every subsequent call (it only reallocates the underlying
        // buffer when the graph grows beyond the current high-water mark).
        if (!impl_->galloc) {
            impl_->galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(impl_->backend));
            if (!impl_->galloc) {
                LA_LOG("Backend::compute: ggml_gallocr_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                impl_->roots.clear();
                ggml_free(ctx);
                return false;
            }
        }
        alloc_ok = ggml_gallocr_alloc_graph(impl_->galloc, gf);
        if (!alloc_ok) LA_LOG("Backend::compute: ggml_gallocr_alloc_graph failed");
    }
    if (!alloc_ok) {
        impl_->pending.clear();
        impl_->captures.clear();
        impl_->roots.clear();
        ggml_free(ctx);
        return false;
    }

    // Inputs are allocated now (->buffer/->data set): push host data in.
    for (const PendingInput& pi : impl_->pending) {
        ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
    }
    impl_->pending.clear();

    enum ggml_status status = need_sched
        ? ggml_backend_sched_graph_compute(impl_->sched, gf)
        : ggml_backend_graph_compute(impl_->backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LA_LOG("Backend::compute: ggml_backend_graph_compute failed (status=%d)",
               (int)status);
        impl_->captures.clear();
        impl_->roots.clear();
        ggml_free(ctx);
        return false;
    }

    // Read back any captured intermediates, then the final output.
    for (const PendingCapture& pc : impl_->captures) {
        size_t cn = (size_t)ggml_nelements(pc.tensor);
        pc.dst->resize(cn);
        ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0, cn * sizeof(float));
    }
    impl_->captures.clear();
    impl_->roots.clear();

    size_t n = (size_t)ggml_nelements(output);
    out.resize(n);
    ggml_backend_tensor_get(output, out.data(), 0, n * sizeof(float));

    ggml_free(ctx);
    return true;
}

bool Backend::forward_capture(const std::function<ggml_tensor*(ggml_context*)>& build,
                              std::vector<float>& out) {
    // Same path as compute(); captures registered during build are honored.
    return compute(build, out);
}

}  // namespace la
