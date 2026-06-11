#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <functional>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>

namespace la {

// Owns host buffers for graph inputs so they outlive a compute() call.
struct GraphInputPool {
    std::vector<std::vector<float>> bufs;
    float* keep(const float* src, size_t n) {
        bufs.emplace_back(src, src + n);
        return bufs.back().data();
    }
    std::vector<std::vector<int32_t>> ibufs;
    int32_t* keep_i32(const int32_t* src, size_t n) {
        ibufs.emplace_back(src, src + n);
        return ibufs.back().data();
    }
};

class Backend {
public:
    // Selects the compute device via the registry. By default the first GPU /
    // integrated-GPU device that a compiled-in backend (CUDA/Metal/Vulkan/...)
    // registers is picked, falling back to CPU. The LA_DEVICE env var overrides:
    //   - "cpu"          forces the CPU backend (numeric baseline / CPU-only box).
    //   - a device name   selects that registry device by name (case-insensitive,
    //                     e.g. "CUDA0", "Vulkan0", "Metal").
    //   - unset           auto-pick the first GPU/IGPU device, else CPU.
    Backend();
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Name of the selected compute device ("cpu" or a registry device name).
    const std::string& device_name() const { return device_name_; }
    // True when the active backend is NOT the CPU backend (a GPU/accelerator was
    // selected). Drives weight offload: when false, the loader keeps using the
    // gguf host tensors directly (zero-copy CPU path, byte-identical to before).
    bool is_offloading() const { return offloading_; }
    // The underlying ggml backend handle (GPU or CPU). Exposed so the loader can
    // allocate + upload the offloaded weights onto the SAME backend graphs run on.
    ggml_backend_t handle() const;

    // Register a 1-D F32 input tensor; data is copied into `pool` and uploaded
    // AFTER graph allocation. Returns the graph leaf tensor.
    ggml_tensor* add_graph_input(ggml_context* ctx, GraphInputPool& pool,
                                 const float* host, size_t n);
    // Register an N-D F32 input tensor (ne[0]=fastest). prod(ne)=n total floats.
    ggml_tensor* add_graph_input_nd(ggml_context* ctx, GraphInputPool& pool,
                                    const float* host, const int64_t* ne, int n_dims);
    // Register an N-D I32 input tensor (e.g. RoPE positions). prod(ne)=n total ints.
    ggml_tensor* add_int32_input_nd(ggml_context* ctx, GraphInputPool& pool,
                                    const int32_t* host, const int64_t* ne, int n_dims);
    // Build -> alloc (persistent gallocr) -> upload inputs -> compute -> read output.
    bool compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                 std::vector<float>& out);
    // Like compute but also reads back tensors registered via capture() during build.
    bool forward_capture(const std::function<ggml_tensor*(ggml_context*)>& build,
                         std::vector<float>& out);
    void capture(ggml_tensor* t, std::vector<float>* dst);  // read this node back too
    // Allocate a persistent backend buffer for all tensors in `ctx` (for ResidentKV).
    // The buffer is on THIS backend (must match the compute gallocr's backend).
    ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx);
    // Register an extra graph root that compute()/forward_capture() will
    // ggml_build_forward_expand (in registration order, BEFORE the output) WITHOUT
    // reading it back. For resident K/V write (ggml_cpy) nodes. Cleared each compute.
    void add_graph_root(ggml_tensor* t);
    void set_n_threads(int n);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string device_name_ = "cpu";
    bool        offloading_  = false;  // true iff `backend` is a non-CPU device
    int         n_threads_   = 1;
};

}  // namespace la
