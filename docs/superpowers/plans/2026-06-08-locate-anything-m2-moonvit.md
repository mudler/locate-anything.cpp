# LocateAnything-3B ggml Port — M2 (MoonViT C++ tower) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MoonViT vision encoder in C++/ggml so that `pixel_values [1024,3,14,14] → merged_tokens [256,4608]` is numerically equal to the HF reference dumps from M0, component by component.

**Architecture:** Mirror parakeet.cpp's C++/ggml structure (`model_loader` → `backend` compute driver with persistent `gallocr` → fused forward graph → GGUF-baseline parity harness) and borrow vibevoice's `qwen2.cpp` per-layer math idioms. Two numerically sensitive ops with no ggml built-in — **bicubic pos-emb interpolation** and **MoonViT's custom 2D RoPE** — are computed host-side in C++ and fed as graph input tensors, each gated against its own isolated Python fixture before being wired into the tower.

**Tech Stack:** C++17, ggml (embedded submodule), CMake/CTest, Python (gguf/numpy/torch) for reference fixtures.

**Reference material (read before starting):**
- Exact MoonViT math: `models/LocateAnything-3B/modeling_vit.py` (the source of truth). The load-bearing facts are restated inline per task, but verify against this file.
- ggml conventions to mirror: `/home/mudler/_git/parakeet.cpp/src/{model_loader,backend,encoder,ggml_graph}.{cpp,hpp}`, `/home/mudler/_git/parakeet.cpp/src/{conformer,graph_builder}.hpp`, `/home/mudler/_git/parakeet.cpp/tests/{parity.hpp,test_encoder.cpp,CMakeLists.txt}`; per-layer math `/home/mudler/_git/vibevoice.cpp/src/qwen2.cpp`.
- GGUF tensor names: `scripts/gguf_keys.py` (`vit.patch_embed.{weight,bias}`, `vit.pos_emb.weight`, `vit.blk.{N}.{norm0,norm1,wqkv,wo}.{weight,bias}`, `vit.blk.{N}.mlp.{fc0,fc1}.{weight,bias}`, `vit.final_norm.{weight,bias}`). KV macros: `include/la_gguf_keys.h` (`LA_KV_VIT_HIDDEN`, `LA_KV_VIT_N_LAYERS`, `LA_KV_VIT_N_HEADS`, `LA_KV_VIT_HEAD_DIM`, `LA_KV_VIT_INTERMEDIATE`, `LA_KV_VIT_PATCH`, `LA_KV_VIT_MERGE`, `LA_KV_VIT_POS_EMB_HW`, `LA_KV_VIT_ROPE_THETA`).
- The f32 GGUF (`models/locate-anything-f32.gguf`) and reference dumps (`dumps/reference.npz` + `dumps/manifest.json`) exist on disk from M0/M1. The reference vision tower ran in **float32**, and ViT weights are stored **f32** in the GGUF, so M2 has no quantization error — diffs should be tight.

**Scope:** M2 ends at `merged_tokens` parity. The MLP projector (`mlp1`/`proj.*` → `projected_tokens`) and vision-token splice are M3.

**Concrete dims (single fixture image, 448×448):** grid 32×32 = 1024 patches; hidden 1152; 16 heads × head_dim 72; 27 layers; intermediate 4304; LayerNorm eps **1e-5**; attention scale **1/sqrt(72)**; GELU **tanh-approx**; RoPE θ=10000; merge 2×2 → 16×16 = 256 merged tokens of dim 4608 (=1152×4).

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/reference_npz_to_gguf.py` | Convert `dumps/reference.npz` → `dumps/reference.gguf` so the C++ parity harness (GGUF-based) can read the gold tensors by name |
| `scripts/gen_vit_fixtures.py` | Generate two isolated op fixtures into `dumps/vit_fixtures.gguf`: bicubic pos-emb interpolation, and 2D-RoPE applied to a fixed random Q/K |
| `tests/parity.hpp` | Copied from parakeet (namespace `la`): `load_baseline` (GGUF→vector) + `compare` (max-abs-diff vs atol/rtol) |
| `src/common.hpp` | `LA_LOG` macro + small shared utilities |
| `src/ggml_extend.hpp` | Header-only `la::` helpers: `linear`, `layernorm`, `gelu_tanh_mlp`, bidirectional `attention` |
| `src/model_loader.{hpp,cpp}` | Open GGUF, read `vit.*` KV via `LA_KV_VIT_*`, build name→tensor map, `realize_weights` (CPU zero-copy) |
| `src/backend.{hpp,cpp}` | ggml backend + persistent `gallocr` + `compute(build, out)` driver + `GraphInputPool` + `forward_capture` |
| `src/vit_rope.{hpp,cpp}` | Host-side MoonViT 2D-RoPE cos/sin table builder (the freq math) |
| `src/vit_posemb.{hpp,cpp}` | Host-side PyTorch-equivalent bicubic interpolation of the learned pos-emb to the actual grid |
| `src/vit_encoder.{hpp,cpp}` | `VitLayerWeights` struct, per-block builder, fused `forward_capture` graph (patch_embed → +pos_emb → 27 blocks → final_norm → merger) |
| `tests/test_model_loader.cpp` | Gate: GGUF loads, vit hparams + tensor shapes correct |
| `tests/test_vit_ops.cpp` | Gate: bicubic interp + 2D-RoPE vs `dumps/vit_fixtures.gguf` |
| `tests/test_vit_encoder.cpp` | Gate: vit_pos_added → vit_layer_00 → vit_layer_26 → vit_final → merged_tokens vs `dumps/reference.gguf` |
| `tests/CMakeLists.txt` | `la_add_test` helper, `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}`, `SKIP_RETURN_CODE 77` |
| `CMakeLists.txt` (modify) | Add `locate_anything` static lib target + `add_subdirectory(tests)` under `LA_BUILD_TESTS` |

**Decomposition rationale:** the two host-computed ops live in their own small files (`vit_rope`, `vit_posemb`) so they can be unit-tested in isolation against fixtures — these are the bit-exactness hotspots. `model_loader`/`backend` are near-verbatim clones of parakeet (low risk). `vit_encoder` is the integration point. Each task gates against a dumped tensor before the next begins.

---

## Task 1: C++ library scaffold + parity harness + reference→GGUF baselines

Stand up the build/test infrastructure and the GGUF baselines the C++ tests read. No model math yet — just prove the lib builds, ctest runs, and the reference tensors are reachable from C++.

**Files:**
- Create: `scripts/reference_npz_to_gguf.py`
- Create: `src/common.hpp`
- Create: `tests/parity.hpp` (copy from parakeet, rename namespace)
- Create: `tests/test_smoke.cpp`
- Create: `tests/CMakeLists.txt`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write `scripts/reference_npz_to_gguf.py`**

```python
#!/usr/bin/env python3
"""Convert dumps/reference.npz -> dumps/reference.gguf so the C++ parity harness
(which reads GGUF) can load the M0 gold tensors by name."""
import sys, os
from pathlib import Path
import numpy as np
import gguf

ROOT = Path(__file__).resolve().parent.parent

def main():
    npz = ROOT / "dumps" / "reference.npz"
    out = ROOT / "dumps" / "reference.gguf"
    z = np.load(npz)
    w = gguf.GGUFWriter(str(out), "la_reference")
    n = 0
    for name in z.files:
        a = z[name]
        if a.dtype == np.int64:        # token_stream — skip (not a float tensor)
            continue
        w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))
        n += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}: {n} tensors")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the baseline and confirm tensor names**

Run:
```bash
. .venv/bin/activate
python scripts/reference_npz_to_gguf.py
python - <<'PY'
import gguf
r = gguf.GGUFReader("dumps/reference.gguf")
names = {t.name for t in r.tensors}
for k in ["pixel_values","vit_pos_added","vit_layer_00","vit_layer_26","vit_final","merged_tokens"]:
    print(k, k in names)
PY
```
Expected: each of the 6 ViT-relevant names prints `True`.

- [ ] **Step 3: Copy parakeet's parity harness into `tests/parity.hpp`**

Run:
```bash
cp /home/mudler/_git/parakeet.cpp/tests/parity.hpp tests/parity.hpp
```
Then edit `tests/parity.hpp`: change the namespace from `pktest` to `la_parity` (whatever the file uses — open it and rename the single namespace). Keep `load_baseline(path, name, out_vec, out_shape)` and `compare(got, ref, label, atol, rtol)` signatures intact. If the file `#include`s parakeet-specific headers, replace with the minimal `ggml.h`/`gguf.h`/`<vector>`/`<cstdio>`/`<cmath>` it actually needs. Verify by reading the copied file that `compare` computes max-abs-diff and returns bool.

- [ ] **Step 4: Write `src/common.hpp`**

```cpp
#pragma once
#include <cstdio>
namespace la {
#define LA_LOG(...) do { std::fprintf(stderr, "[la] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
}  // namespace la
```

- [ ] **Step 5: Write `tests/test_smoke.cpp`**

```cpp
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
// Proves the test rig links ggml/gguf and can read the reference baseline.
int main() {
    const char* base = std::getenv("LA_TEST_BASELINE");
    if (!base) { std::fprintf(stderr, "LA_TEST_BASELINE unset; skip\n"); return 77; }
    std::vector<float> px; std::vector<int64_t> shape;
    if (!la_parity::load_baseline(base, "pixel_values", px, shape)) return 1;
    // pixel_values is [1024,3,14,14] = 602112 floats
    std::printf("pixel_values nelem=%zu shape0=%lld\n", px.size(), (long long)shape[0]);
    return (px.size() == 1024ull*3*14*14) ? 0 : 1;
}
```

- [ ] **Step 6: Write `tests/CMakeLists.txt`**

```cmake
function(la_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE locate_anything)
  target_include_directories(${name} PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT "LA_TEST_GGUF=${CMAKE_SOURCE_DIR}/models/locate-anything-f32.gguf;LA_TEST_BASELINE=${CMAKE_SOURCE_DIR}/dumps/reference.gguf;LA_TEST_FIXTURES=${CMAKE_SOURCE_DIR}/dumps/vit_fixtures.gguf")
endfunction()

la_add_test(test_smoke)
```

- [ ] **Step 7: Add the library target + tests to the top `CMakeLists.txt`**

Replace the trailing `message(STATUS ...)` line in `CMakeLists.txt` with:
```cmake
# ---------- liblocate_anything ----------
set(LA_SOURCES
    src/model_loader.cpp
    src/backend.cpp
    src/vit_rope.cpp
    src/vit_posemb.cpp
    src/vit_encoder.cpp
)
# Sources are created across M2 tasks; until they all exist, an empty lib is fine.
add_library(locate_anything STATIC ${LA_SOURCES})
target_include_directories(locate_anything
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(locate_anything PUBLIC ggml)

if(LA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
message(STATUS "locate-anything.cpp configured (lib + tests)")
```
NOTE: `LA_SOURCES` lists files created in later tasks. To keep Task 1 building, create empty stub `.cpp` files now: `src/model_loader.cpp`, `src/backend.cpp`, `src/vit_rope.cpp`, `src/vit_posemb.cpp`, `src/vit_encoder.cpp` each containing just `// stub — implemented in a later M2 task`. They'll be filled in by their tasks. (A static lib with empty TUs links fine.)

- [ ] **Step 8: Build and run the smoke test**

Run:
```bash
for f in model_loader backend vit_rope vit_posemb vit_encoder; do [ -f src/$f.cpp ] || echo "// stub" > src/$f.cpp; done
cmake -B build -DLA_BUILD_TESTS=ON >/tmp/la_m2_cfg.log 2>&1 && echo CONFIG_OK
cmake --build build --target test_smoke -j 2>&1 | tail -3
LA_TEST_BASELINE=dumps/reference.gguf ./build/tests/test_smoke; echo "exit=$?"
```
Expected: `CONFIG_OK`; test_smoke builds; running it prints `pixel_values nelem=602112 shape0=1024` and `exit=0`.

- [ ] **Step 9: Commit**

```bash
git add scripts/reference_npz_to_gguf.py src/common.hpp src/*.cpp tests/parity.hpp tests/test_smoke.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M2: C++ lib scaffold + parity harness + reference.gguf baseline"
```
Author `mudler <mudler@localai.io>`, end commit messages (here and every task below) with a trailing:
```

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: ModelLoader (GGUF KV + vit tensor map + CPU realize)

**Files:**
- Create: `src/model_loader.hpp` (replace the stub `.cpp`)
- Modify: `src/model_loader.cpp`
- Create: `tests/test_model_loader.cpp`
- Modify: `tests/CMakeLists.txt` (add `la_add_test(test_model_loader)`)

Mirror `/home/mudler/_git/parakeet.cpp/src/model_loader.{hpp,cpp}` closely. Read that file first; reproduce its structure, swapping the config struct + KV keys.

- [ ] **Step 1: Write the failing test `tests/test_model_loader.cpp`**

```cpp
#include "model_loader.hpp"
#include <cstdlib>
#include <cstdio>
int main() {
    const char* gguf = std::getenv("LA_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "LA_TEST_GGUF unset; skip\n"); return 77; }
    la::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }
    const auto& c = ml.config();
    int ok = 1;
    ok &= (c.vit_hidden == 1152);
    ok &= (c.vit_n_layers == 27);
    ok &= (c.vit_n_heads == 16);
    ok &= (c.vit_head_dim == 72);
    ok &= (c.vit_intermediate == 4304);
    ok &= (c.vit_patch == 14);
    ok &= (c.vit_pos_emb_hw == 64);
    // tensor presence + shape: wqkv is [1152 -> 3456], stored raw [out=3456, in=1152]
    ggml_tensor* wqkv = ml.tensor("vit.blk.0.wqkv.weight");
    ok &= (wqkv != nullptr);
    if (wqkv) { ok &= (wqkv->ne[0] == 1152 && wqkv->ne[1] == 3456); }  // ggml ne reversed
    ggml_tensor* pos = ml.tensor("vit.pos_emb.weight");
    ok &= (pos != nullptr);
    if (pos) { ok &= (pos->ne[0] == 1152 && pos->ne[1] == 64 && pos->ne[2] == 64); }
    ggml_tensor* pe = ml.tensor("vit.patch_embed.weight");
    ok &= (pe != nullptr);  // raw torch [1152,3,14,14] -> ggml ne [14,14,3,1152]
    if (pe) { ok &= (pe->ne[0]==14 && pe->ne[1]==14 && pe->ne[2]==3 && pe->ne[3]==1152); }
    std::printf("loader ok=%d\n", ok);
    return ok ? 0 : 1;
}
```

- [ ] **Step 2: Wire the test and run it (expect link/compile fail — no ModelLoader yet)**

Add `la_add_test(test_model_loader)` to `tests/CMakeLists.txt`. Run:
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_model_loader -j 2>&1 | tail -5
```
Expected: FAIL (no `model_loader.hpp` / undefined `la::ModelLoader`).

- [ ] **Step 3: Write `src/model_loader.hpp`**

```cpp
#pragma once
#include "ggml.h"
#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace la {

struct VitConfig {
    uint32_t vit_hidden = 0, vit_n_layers = 0, vit_n_heads = 0, vit_head_dim = 0,
             vit_intermediate = 0, vit_patch = 0, vit_pos_emb_hw = 0;
    std::vector<int32_t> vit_merge;       // [2,2]
    float    vit_rope_theta = 10000.f;
};

class ModelLoader {
public:
    bool load(const std::string& path);
    const VitConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const;   // nullptr if absent
    ~ModelLoader();
private:
    VitConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_  = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};

}  // namespace la
```

- [ ] **Step 4: Write `src/model_loader.cpp`**

```cpp
#include "model_loader.hpp"
#include "common.hpp"
#include "la_gguf_keys.h"

namespace la {

static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_u32(g,id);
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out; int64_t id = gguf_find_key(g,k);
    if (id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}

bool ModelLoader::load(const std::string& path){
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) { LA_LOG("gguf_init_from_file failed: %s", path.c_str()); return false; }
    cfg_.vit_hidden       = kv_u32(gguf_, LA_KV_VIT_HIDDEN);
    cfg_.vit_n_layers     = kv_u32(gguf_, LA_KV_VIT_N_LAYERS);
    cfg_.vit_n_heads      = kv_u32(gguf_, LA_KV_VIT_N_HEADS);
    cfg_.vit_head_dim     = kv_u32(gguf_, LA_KV_VIT_HEAD_DIM);
    cfg_.vit_intermediate = kv_u32(gguf_, LA_KV_VIT_INTERMEDIATE);
    cfg_.vit_patch        = kv_u32(gguf_, LA_KV_VIT_PATCH);
    cfg_.vit_pos_emb_hw   = kv_u32(gguf_, LA_KV_VIT_POS_EMB_HW);
    cfg_.vit_merge        = kv_i32_arr(gguf_, LA_KV_VIT_MERGE);
    cfg_.vit_rope_theta   = kv_f32(gguf_, LA_KV_VIT_ROPE_THETA, 10000.f);
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for (int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if (t) tensors_[nm] = t;
    }
    return cfg_.vit_hidden>0 && cfg_.vit_n_layers>0;
}

ggml_tensor* ModelLoader::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it==tensors_.end() ? nullptr : it->second;
}

ModelLoader::~ModelLoader(){
    if (gguf_) gguf_free(gguf_);
    if (ctx_)  ggml_free(ctx_);
}

}  // namespace la
```
NOTE: with `no_alloc=false`, `gguf_init_from_file` loads all tensor data into `ctx_`'s buffer (host/CPU), so the tensors are immediately usable as CPU graph leaves in M2 (CPU backend). The `realize_weights`/device-upload path from parakeet is only needed for GPU backends — defer it (YAGNI for M2 CPU parity). Verify the actual `gguf_init_params`/`gguf_get_*` signatures against `third_party/ggml/include/gguf.h` and parakeet's loader, and adapt if the embedded ggml version differs (e.g. `gguf_get_val_u32` arg types). The tensor presence/shape contract the test asserts is the real gate.

- [ ] **Step 5: Build and run the test**

Run:
```bash
cmake --build build --target test_model_loader -j 2>&1 | tail -3
cd build && ctest -R test_model_loader --output-on-failure; cd ..
```
Expected: PASS, prints `loader ok=1`. If a tensor shape assertion fails, check ggml's reversed-ne convention against the actual stored shapes (print `wqkv->ne[0..1]`).

- [ ] **Step 6: Commit**

```bash
git add src/model_loader.hpp src/model_loader.cpp tests/test_model_loader.cpp tests/CMakeLists.txt
git commit -m "M2: GGUF model loader (vit hparams + tensor map)"
```

---

## Task 3: Backend compute driver + GraphInputPool

**Files:**
- Create: `src/backend.hpp` (replace stub)
- Modify: `src/backend.cpp`
- Create: `tests/test_backend.cpp`
- Modify: `tests/CMakeLists.txt`

Mirror `/home/mudler/_git/parakeet.cpp/src/backend.{hpp,cpp}` and `graph_builder.hpp`. The essential contract: `compute(build_fn, out)` builds a metadata-only graph (`no_alloc=true`), allocates via a **persistent** `ggml_gallocr`, writes host inputs AFTER allocation, runs `ggml_backend_graph_compute`, reads the output back into `out`. Host inputs are registered during build via `add_graph_input(tensor, host_ptr, nbytes)` and their host buffers live in a `GraphInputPool` that outlives the call.

- [ ] **Step 1: Write the failing test `tests/test_backend.cpp`**

```cpp
#include "backend.hpp"
#include <vector>
#include <cstdio>
// Build a trivial graph: out = a + b for two host-fed [4] inputs, check result.
int main() {
    la::Backend backend;
    std::vector<float> a = {1,2,3,4}, b = {10,20,30,40}, out;
    la::GraphInputPool pool;
    bool ok = backend.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* ta = backend.add_graph_input(ctx, pool, a.data(), a.size());
        ggml_tensor* tb = backend.add_graph_input(ctx, pool, b.data(), b.size());
        return ggml_add(ctx, ta, tb);
    }, out);
    if (!ok || out.size()!=4) return 1;
    int good = (out[0]==11 && out[1]==22 && out[2]==33 && out[3]==44);
    std::printf("backend add: %f %f %f %f ok=%d\n", out[0],out[1],out[2],out[3], good);
    return good ? 0 : 1;
}
```
NOTE: the exact `add_graph_input` signature is your design — keep it minimal (ctx, pool, host_ptr, n_floats) returning a 1-D F32 input tensor with `ggml_set_input` set. Match whatever shape helpers you expose; the test above defines the contract you implement. If you prefer the parakeet thread-local registry idiom over passing `pool` explicitly, adapt the test to match — but keep host-input-after-alloc ordering.

- [ ] **Step 2: Wire + run (expect fail)**

Add `la_add_test(test_backend)`. Build target → FAIL (no backend.hpp).

- [ ] **Step 3: Write `src/backend.hpp`**

```cpp
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <functional>
#include <vector>
#include <memory>

namespace la {

// Owns host buffers for graph inputs so they outlive a compute() call.
struct GraphInputPool {
    std::vector<std::vector<float>> bufs;
    float* keep(const float* src, size_t n) {
        bufs.emplace_back(src, src + n);
        return bufs.back().data();
    }
};

class Backend {
public:
    Backend();
    ~Backend();
    // Register a 1-D F32 input tensor; data is copied into `pool` and uploaded
    // AFTER graph allocation. Returns the graph leaf tensor.
    ggml_tensor* add_graph_input(ggml_context* ctx, GraphInputPool& pool,
                                 const float* host, size_t n);
    // Register an N-D F32 input tensor (ne[0]=fastest). prod(ne)=n.
    ggml_tensor* add_graph_input_nd(ggml_context* ctx, GraphInputPool& pool,
                                    const float* host, const int64_t* ne, int n_dims);
    // Build → alloc (persistent gallocr) → upload inputs → compute → read output.
    bool compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                 std::vector<float>& out);
    // Like compute but also reads back the tensors registered via capture().
    bool forward_capture(const std::function<ggml_tensor*(ggml_context*)>& build,
                         std::vector<float>& out);
    void capture(ggml_tensor* t, std::vector<float>* dst);  // read this node back too
    void set_n_threads(int n);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace la
```

- [ ] **Step 4: Write `src/backend.cpp`**

Implement against `/home/mudler/_git/parakeet.cpp/src/backend.cpp` as the template (read it). Key points to reproduce:
- `Impl` holds `ggml_backend_t backend` (CPU: `ggml_backend_cpu_init()`), a persistent `ggml_gallocr_t galloc` (created lazily on first compute from `ggml_backend_get_default_buffer_type(backend)`), a list of pending `(ggml_tensor*, const float* host, size_t nbytes)` input uploads, and a list of `(ggml_tensor*, std::vector<float>*)` captures.
- `compute`: `ggml_init({overhead*kGraphSize + graph_overhead, nullptr, /*no_alloc=*/true})` → run `build(ctx)` → `ggml_set_output(out)` (and on each captured tensor) → `ggml_new_graph_custom(ctx, kGraphSize=16384, false)` → `ggml_build_forward_expand(gf, out)` (and expand each captured tensor) → `ggml_gallocr_alloc_graph(galloc, gf)` → for each pending input `ggml_backend_tensor_set(t, host, 0, nbytes)` → `ggml_backend_graph_compute(backend, gf)` → `ggml_backend_tensor_get(out, out_vec.data(), ...)` (and each capture) → clear pending/captures → `ggml_free(ctx)`.
- `add_graph_input(_nd)`: create a `ggml_new_tensor_*d(ctx, GGML_TYPE_F32, ...)`, `ggml_set_input(t)`, push `(t, pool.keep(host,n), n*sizeof(float))` onto pending, return t. (Copy into pool so the host pointer is stable across the deferred upload.)
- `forward_capture` is just `compute` with the capture list honored (same code path; capture registers extra outputs to expand+read).

CORRECTNESS: never write input data inside the build lambda (tensor->data is NULL until alloc). Always defer to the post-alloc `ggml_backend_tensor_set`. Verify backend/gallocr API names against `third_party/ggml/include/ggml-backend.h` (the embedded version) and parakeet.

- [ ] **Step 5: Build and run**

Run:
```bash
cmake --build build --target test_backend -j 2>&1 | tail -3
cd build && ctest -R test_backend --output-on-failure; cd ..
```
Expected: PASS, prints `backend add: 11 22 33 44 ok=1`.

- [ ] **Step 6: Commit**

```bash
git add src/backend.hpp src/backend.cpp tests/test_backend.cpp tests/CMakeLists.txt
git commit -m "M2: ggml backend compute driver + persistent gallocr + input pool"
```

---

## Task 4: ggml_extend helpers (linear, layernorm, gelu-tanh MLP, attention)

**Files:**
- Create: `src/ggml_extend.hpp`
- Create: `tests/test_ggml_extend.cpp`
- Modify: `tests/CMakeLists.txt`

Header-only `la::` helpers. LayerNorm (with bias) and tanh-GELU are the ones that must match the reference exactly.

- [ ] **Step 1: Write the failing test `tests/test_ggml_extend.cpp`**

```cpp
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
    // mean=2.5, var=1.25, std=sqrt(1.25)=1.1180; (1-2.5)/1.118 = -1.3416 ...
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
```

- [ ] **Step 2: Wire + run (expect fail)** — add `la_add_test(test_ggml_extend)`, build → FAIL.

- [ ] **Step 3: Write `src/ggml_extend.hpp`**

```cpp
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
inline ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu(ctx, x);   // ggml_gelu is the tanh approximation
}

}  // namespace la
```
VERIFY: confirm `ggml_gelu` in the embedded ggml is the **tanh** approximation (it historically is; `ggml_gelu_erf`/`ggml_gelu_quick` are the variants). Check `third_party/ggml/include/ggml.h`. If `ggml_gelu` is NOT tanh in this version, use whichever symbol is the tanh approx, or build it from primitives: `0.5*x*(1+tanh(0.79788456*(x+0.044715*x^3)))`. The test's `gelu_ok` assertion is the gate — it will fail if you pick the erf variant, so this is self-checking.

- [ ] **Step 4: Build and run** → expect PASS, `ln_ok=1 gelu_ok=1`. If `gelu_ok=0`, you used the erf variant — switch to the tanh one.

- [ ] **Step 5: Commit**

```bash
git add src/ggml_extend.hpp tests/test_ggml_extend.cpp tests/CMakeLists.txt
git commit -m "M2: ggml_extend helpers (linear, layernorm, tanh-gelu) with unit gates"
```

---

## Task 5: Patch embed + bicubic pos-emb (host-side) → gate `vit_pos_added`

The patch embed is a per-patch linear of the flattened 588-vector → 1152 (conv with kernel=stride=14). The learned pos-emb `[64,64,1152]` is **bicubic-interpolated** (PyTorch `align_corners=False`, `a=-0.75`) to the 32×32 grid and added. ggml has no matching bicubic op, so compute the interpolation host-side and feed the result as an input.

**Files:**
- Create: `src/vit_posemb.hpp`, modify `src/vit_posemb.cpp`
- Create: `scripts/gen_vit_fixtures.py` (pos-emb half now; rope half in Task 6)
- Create: `tests/test_vit_posemb.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write `scripts/gen_vit_fixtures.py` (pos-emb interpolation fixture)**

```python
#!/usr/bin/env python3
"""Isolated fixtures for the two host-computed MoonViT ops, into dumps/vit_fixtures.gguf:
- posemb_interp_32x32: F.interpolate(pos_emb[64,64,1152]->[32,32], bicubic, align_corners=False) flattened [1024,1152]
(rope fixtures are appended in Task 6)."""
import sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
import gguf

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "LocateAnything-3B"

def load_pos_emb():
    idx = __import__("json").load(open(MODEL/"model.safetensors.index.json"))
    shard = idx["weight_map"]["vision_model.patch_embed.pos_emb.weight"]
    with safe_open(MODEL/shard, framework="pt") as f:
        return f.get_tensor("vision_model.patch_embed.pos_emb.weight").float()  # [64,64,1152]

def main():
    out = ROOT / "dumps" / "vit_fixtures.gguf"
    w = gguf.GGUFWriter(str(out), "vit_fixtures")
    pe = load_pos_emb()  # [64,64,1152]
    interp = F.interpolate(pe.permute(2,0,1).unsqueeze(0), size=(32,32),
                           mode="bicubic", align_corners=False).squeeze(0).permute(1,2,0)
    interp = interp.reshape(-1, 1152).contiguous().numpy().astype(np.float32)  # [1024,1152]
    w.add_tensor("posemb_interp_32x32", np.ascontiguousarray(interp))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", out, "posemb_interp_32x32", interp.shape)

if __name__ == "__main__":
    main()
```
Run it: `python scripts/gen_vit_fixtures.py` → expect `posemb_interp_32x32 (1024, 1152)`.

- [ ] **Step 2: Write the failing test `tests/test_vit_posemb.cpp`**

```cpp
#include "vit_posemb.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
int main() {
    const char* gguf = std::getenv("LA_TEST_GGUF");
    const char* fix  = std::getenv("LA_TEST_FIXTURES");
    if (!gguf || !fix) { std::fprintf(stderr,"env unset; skip\n"); return 77; }
    la::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    // pos_emb.weight stored raw [64,64,1152] -> ggml ne [1152,64,64]; read its bytes
    ggml_tensor* pe = ml.tensor("vit.pos_emb.weight");
    if (!pe) return 1;
    std::vector<float> src(ggml_nelements(pe));
    // CPU buffer: data is directly accessible
    std::memcpy(src.data(), pe->data, ggml_nbytes(pe));
    // host bicubic interp 64x64 -> 32x32, base layout [h,w,c] row-major (c fastest)
    std::vector<float> got = la::bicubic_pos_emb(src, /*base*/64, 64, /*c*/1152, /*gh*/32, /*gw*/32);
    std::vector<float> ref; std::vector<int64_t> shape;
    if (!la_parity::load_baseline(fix, "posemb_interp_32x32", ref, shape)) return 1;
    bool ok = la_parity::compare(got, ref, "posemb_interp", 1e-3f, 1e-3f);
    return ok ? 0 : 1;
}
```
NOTE: confirm `pe->data` is host-accessible (CPU backend, `no_alloc=false` load → yes). The stored pos_emb is torch `[64,64,1152]` = (h, w, c) with c fastest; ggml ne = `[1152,64,64]` = (c, w, h) with c fastest — i.e. the flat byte order is identical: element `(h,w,c)` at offset `(h*64+w)*1152 + c`. `bicubic_pos_emb` consumes that flat layout.

- [ ] **Step 3: Wire + run (expect fail)** — add `la_add_test(test_vit_posemb)`, build → FAIL.

- [ ] **Step 4: Write `src/vit_posemb.hpp` + `src/vit_posemb.cpp` (PyTorch-equivalent bicubic)**

`src/vit_posemb.hpp`:
```cpp
#pragma once
#include <vector>
namespace la {
// Bicubic interpolation matching torch.nn.functional.interpolate(mode="bicubic",
// align_corners=False, antialias=False), cubic coefficient a=-0.75.
// `src` is the base grid flattened row-major as [h*w, c] (c fastest), size base_h*base_w*c.
// Returns [gh*gw, c] flattened the same way.
std::vector<float> bicubic_pos_emb(const std::vector<float>& src,
                                   int base_h, int base_w, int c,
                                   int gh, int gw);
}  // namespace la
```
`src/vit_posemb.cpp`:
```cpp
#include "vit_posemb.hpp"
#include <cmath>
#include <algorithm>
namespace la {

// PyTorch cubic convolution kernel, a = -0.75.
static inline float cubic_w(float t, float a){
    t = std::fabs(t);
    if (t <= 1.f) return ((a+2.f)*t - (a+3.f))*t*t + 1.f;
    if (t <  2.f) return (((t - 5.f)*t + 8.f)*t - 4.f)*a;
    return 0.f;
}
static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

std::vector<float> bicubic_pos_emb(const std::vector<float>& src,
                                   int base_h, int base_w, int c,
                                   int gh, int gw){
    const float a = -0.75f;
    const float sh = (float)base_h / (float)gh;   // scale = 64/32 = 2
    const float sw = (float)base_w / (float)gw;
    std::vector<float> out((size_t)gh*gw*c, 0.f);
    auto at = [&](int y, int x, int ch)->float{
        y = clampi(y, 0, base_h-1); x = clampi(x, 0, base_w-1);
        return src[((size_t)y*base_w + x)*c + ch];
    };
    for (int oy=0; oy<gh; ++oy){
        // align_corners=False source mapping
        float fy = (oy + 0.5f)*sh - 0.5f;
        int   iy = (int)std::floor(fy);
        float ty = fy - iy;
        float wy[4] = { cubic_w(1.f+ty,a), cubic_w(ty,a), cubic_w(1.f-ty,a), cubic_w(2.f-ty,a) };
        for (int ox=0; ox<gw; ++ox){
            float fx = (ox + 0.5f)*sw - 0.5f;
            int   ix = (int)std::floor(fx);
            float tx = fx - ix;
            float wx[4] = { cubic_w(1.f+tx,a), cubic_w(tx,a), cubic_w(1.f-tx,a), cubic_w(2.f-tx,a) };
            for (int ch=0; ch<c; ++ch){
                float acc = 0.f;
                for (int m=0;m<4;++m){
                    float row = 0.f;
                    for (int n=0;n<4;++n)
                        row += wx[n]*at(iy-1+m, ix-1+n, ch);
                    acc += wy[m]*row;
                }
                out[((size_t)oy*gw + ox)*c + ch] = acc;
            }
        }
    }
    return out;
}

}  // namespace la
```
NOTE: this is the standard PyTorch bicubic (separable cubic convolution, edge-clamped, `align_corners=False`). The test gates it to atol 1e-3 against torch's own output; if it fails, the likely culprits are the source-coordinate mapping (`(o+0.5)*scale-0.5`) or the cubic coefficient ordering — debug by printing a few `got`/`ref` pairs at known indices.

- [ ] **Step 5: Build + run the pos-emb test** → expect PASS (`compare` prints OK, max-abs-diff well under 1e-3).

- [ ] **Step 6: Now wire patch_embed + posemb into a partial encoder and gate `vit_pos_added`.** Add to `src/vit_encoder.{hpp,cpp}` a minimal entry that: (a) feeds `pixel_values` (flattened host [588,1024], from the reference baseline) into a `mul_mat` with the reshaped `vit.patch_embed.weight` (ggml_reshape_2d to [588,1152]) + bias, giving [1152,1024]; (b) host-computes the bicubic pos-emb [1024,1152], feeds it as an input reshaped to [1152,1024] (transpose layout: pos-emb is [token,c]=[1024,1152] → ggml wants [c,token]=[1152,1024]; feed it already transposed so ne[0]=1152), and `ggml_add`. Output `vit_pos_added` [1152,1024]. Add `tests/test_vit_encoder.cpp` with a first assertion comparing this against the `vit_pos_added` baseline (transpose the [1024,1152] reference to [1152,1024] order, or compare with matching layout — be explicit about which axis is which). Gate atol 1e-3, rtol 1e-3.

Detailed code for (a)/(b) and the pixel flattening:
```cpp
// pixel_values baseline is [1024,3,14,14] row-major (c,i,j fastest within patch).
// Conv weight [1152,3,14,14] row-major. Per-patch linear: out[p,o] = sum_{cij} W[o,cij]*px[p,cij] + bias[o].
// Feed px as ggml [588, 1024] (588 fastest = the c*14*14 flatten), W as [588,1152] via reshape.
ggml_tensor* W = ggml_reshape_2d(ctx, ml.tensor("vit.patch_embed.weight"), 588, 1152);
ggml_tensor* px = be.add_graph_input_nd(ctx, pool, pixel_host, (int64_t[]){588,1024}, 2);
ggml_tensor* x  = la::linear(ctx, W, px, ml.tensor("vit.patch_embed.bias")); // [1152,1024]
// pos-emb: host bicubic -> [1024,1152]; transpose to [1152,1024] on host before feeding.
ggml_tensor* pe = be.add_graph_input_nd(ctx, pool, pos_host_T, (int64_t[]){1152,1024}, 2);
x = ggml_add(ctx, x, pe);  // vit_pos_added
```
The reference `vit_pos_added` dump is `[1024,1152]`; when you read it back from the graph as `[1152,1024]` row-major and compare, transpose one side so the elementwise comparison aligns. Keep the convention "ggml carries [hidden, token]" throughout the encoder and only transpose at the comparison boundary in the test.

- [ ] **Step 7: Build + run `test_vit_encoder` (vit_pos_added gate only)** → PASS at 1e-3.

- [ ] **Step 8: Commit**

```bash
git add scripts/gen_vit_fixtures.py src/vit_posemb.hpp src/vit_posemb.cpp src/vit_encoder.hpp src/vit_encoder.cpp tests/test_vit_posemb.cpp tests/test_vit_encoder.cpp tests/CMakeLists.txt
git commit -m "M2: patch embed + bicubic pos-emb (host) -> vit_pos_added parity"
```

---

## Task 6: 2D RoPE (host cos/sin + interleaved rotation) → isolated fixture gate

MoonViT's RoPE is **not** a ggml built-in: head_dim 72 split into 36 **adjacent** pairs (2k, 2k+1); pair k rotates by angle `coord(k)·invfreq[k//2]` where `coord` = **column w** for even k, **row h** for odd k, and `invfreq[i]=10000^(-4i/72)`, i∈[0,18). Rotation is interleaved: `out[2k]=x[2k]cos−x[2k+1]sin; out[2k+1]=x[2k]sin+x[2k+1]cos`. Computed in f32 on Q and K only. We precompute cos/sin host-side and apply the rotation with ggml ops.

**Files:**
- Modify: `scripts/gen_vit_fixtures.py` (append rope fixture)
- Create: `src/vit_rope.hpp`, modify `src/vit_rope.cpp`
- Create: `tests/test_vit_rope.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Append the RoPE fixture to `scripts/gen_vit_fixtures.py`**

Add (before `w.close()`), generating a fixed random Q and the model's own rope output:
```python
    import importlib.util
    spec = importlib.util.spec_from_file_location("mvit", MODEL/"modeling_vit.py")
    # If modeling_vit imports are heavy, instead inline the Rope2DPosEmb math here.
    # Simpler + dependency-free: reproduce the exact rope and verify against a small torch ref.
    torch.manual_seed(0)
    H, W = 32, 32; n = H*W; heads = 16; hd = 72
    q = torch.randn(n, heads, hd)
    k = torch.randn(n, heads, hd)
    # build freqs_cis exactly as Rope2DPosEmb: even pair -> column(w), odd pair -> row(h)
    dim_range = torch.arange(0, hd, 4).float()           # [0,4,...,68], len 18
    invfreq = 1.0 / (10000.0 ** (dim_range / hd))        # [18]
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    xpos = xs.reshape(-1).float(); ypos = ys.reshape(-1).float()   # [1024]
    xfreq = torch.outer(xpos, invfreq); yfreq = torch.outer(ypos, invfreq)  # [1024,18]
    xcis = torch.polar(torch.ones_like(xfreq), xfreq)
    ycis = torch.polar(torch.ones_like(yfreq), yfreq)
    fc = torch.cat([xcis.unsqueeze(-1), ycis.unsqueeze(-1)], dim=-1).reshape(n, hd//2)  # complex [1024,36]
    def apply(x):
        xc = torch.view_as_complex(x.float().reshape(n, heads, hd//2, 2))
        out = torch.view_as_real(xc * fc.unsqueeze(1)).reshape(n, heads, hd)
        return out
    qo = apply(q).contiguous().numpy().astype(np.float32)
    ko = apply(k).contiguous().numpy().astype(np.float32)
    w.add_tensor("rope_q_in",  np.ascontiguousarray(q.numpy().astype(np.float32)))   # [1024,16,72]
    w.add_tensor("rope_k_in",  np.ascontiguousarray(k.numpy().astype(np.float32)))
    w.add_tensor("rope_q_out", np.ascontiguousarray(qo))
    w.add_tensor("rope_k_out", np.ascontiguousarray(ko))
```
Run `python scripts/gen_vit_fixtures.py` again → confirm `rope_q_in/out`, `rope_k_in/out` written (shape [1024,16,72]).

- [ ] **Step 2: Write the failing test `tests/test_vit_rope.cpp`**

```cpp
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
    if (!la_parity::load_baseline(fix,"rope_q_in",qin,s)) return 1;
    if (!la_parity::load_baseline(fix,"rope_q_out",qout,s)) return 1;
    // build cos/sin tables host-side for the 32x32 grid
    la::RopeTables rt = la::build_rope_tables(/*H*/32,/*W*/32,/*heads*/16,/*head_dim*/72,/*theta*/10000.f);
    la::Backend be; la::GraphInputPool pool;
    std::vector<float> got;
    be.compute([&](ggml_context* ctx)->ggml_tensor*{
        // q laid out [head_dim, heads, tokens] = [72,16,1024]
        ggml_tensor* q = be.add_graph_input_nd(ctx,pool,qin.data(),(int64_t[]){72,16,1024},3);
        return la::apply_rope(ctx, be, pool, q, rt);  // returns rotated, same shape
    }, got);
    // qout fixture is [1024,16,72]; got is [72,16,1024] flattened -> compare with matching index map
    bool ok = la::compare_rope(got, qout, /*tokens*/1024,/*heads*/16,/*hd*/72, 1e-4f);
    std::printf("rope ok=%d\n", ok);
    return ok ? 0 : 1;
}
```
NOTE: define `la::compare_rope` as a tiny helper (in vit_rope.hpp) that reindexes between the ggml `[hd,heads,tok]` flat order and the fixture `[tok,heads,hd]` flat order and asserts max-abs-diff < tol. Keeping the index map explicit avoids a silent layout bug.

- [ ] **Step 3: Wire + run (expect fail)** — add `la_add_test(test_vit_rope)`, build → FAIL.

- [ ] **Step 4: Write `src/vit_rope.hpp` + `src/vit_rope.cpp`**

`src/vit_rope.hpp`:
```cpp
#pragma once
#include "ggml.h"
#include <vector>
namespace la {
class Backend; struct GraphInputPool;

struct RopeTables {
    // cos/sin per (pair, token), broadcastable as ggml [1, n_pairs, 1, tokens].
    std::vector<float> cos;   // size n_pairs*tokens, layout [token][pair] (pair fastest)
    std::vector<float> sin;
    int tokens=0, n_pairs=0, heads=0, head_dim=0;
};

// Build cos/sin for an HxW grid. Pair k in [0,head_dim/2): even k uses column(w),
// odd k uses row(h); angle = coord * 10000^(-4*(k/2)/head_dim).
RopeTables build_rope_tables(int H, int W, int heads, int head_dim, float theta);

// Apply interleaved-pair rotation to x [head_dim, heads, tokens]; returns same shape.
ggml_tensor* apply_rope(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                        ggml_tensor* x, const RopeTables& rt);

// Compare ggml [hd,heads,tok] flat vs fixture [tok,heads,hd] flat.
bool compare_rope(const std::vector<float>& got_hd_heads_tok,
                  const std::vector<float>& ref_tok_heads_hd,
                  int tokens, int heads, int hd, float tol);
}  // namespace la
```
`src/vit_rope.cpp`:
```cpp
#include "vit_rope.hpp"
#include "backend.hpp"
#include <cmath>
namespace la {

RopeTables build_rope_tables(int H, int W, int heads, int head_dim, float theta){
    RopeTables rt; rt.tokens=H*W; rt.n_pairs=head_dim/2; rt.heads=heads; rt.head_dim=head_dim;
    rt.cos.resize((size_t)rt.tokens*rt.n_pairs);
    rt.sin.resize((size_t)rt.tokens*rt.n_pairs);
    for (int h=0;h<H;++h) for (int w=0;w<W;++w){
        int tok = h*W + w;
        for (int k=0;k<rt.n_pairs;++k){
            int i = k/2;                                   // invfreq index
            float invf = std::pow(theta, -4.f*(float)i/(float)head_dim);
            float coord = (k%2==0) ? (float)w : (float)h;  // even=column, odd=row
            float ang = coord * invf;
            rt.cos[(size_t)tok*rt.n_pairs + k] = std::cos(ang);
            rt.sin[(size_t)tok*rt.n_pairs + k] = std::sin(ang);
        }
    }
    return rt;
}

ggml_tensor* apply_rope(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                        ggml_tensor* x, const RopeTables& rt){
    const int hd = rt.head_dim, np = rt.n_pairs, heads = rt.heads, tok = rt.tokens;
    // x: [hd, heads, tok] -> view as pairs [2, np, heads, tok]
    ggml_tensor* xr = ggml_reshape_4d(ctx, x, 2, np, heads, tok);
    // even/odd components along ne0
    ggml_tensor* x0 = ggml_cont(ctx, ggml_view_4d(ctx, xr, 1, np, heads, tok,
                          xr->nb[1], xr->nb[2], xr->nb[3], 0));               // [1,np,heads,tok]
    ggml_tensor* x1 = ggml_cont(ctx, ggml_view_4d(ctx, xr, 1, np, heads, tok,
                          xr->nb[1], xr->nb[2], xr->nb[3], xr->nb[0]));       // offset 1 elem
    // rot = concat(-x1, x0) along ne0  -> [2,np,heads,tok]
    ggml_tensor* rot = ggml_concat(ctx, ggml_neg(ctx, x1), x0, 0);
    // cos/sin as [1, np, 1, tok] inputs, broadcast over ne0=2 and heads
    ggml_tensor* cosb = be.add_graph_input_nd(ctx, pool, rt.cos.data(),
                          (int64_t[]){1, np, 1, tok}, 4);
    ggml_tensor* sinb = be.add_graph_input_nd(ctx, pool, rt.sin.data(),
                          (int64_t[]){1, np, 1, tok}, 4);
    ggml_tensor* out = ggml_add(ctx, ggml_mul(ctx, xr, cosb), ggml_mul(ctx, rot, sinb));
    return ggml_reshape_3d(ctx, ggml_cont(ctx, out), hd, heads, tok);
}

bool compare_rope(const std::vector<float>& got, const std::vector<float>& ref,
                  int tokens, int heads, int hd, float tol){
    // ref (fixture) flat order is [tok][head][d] (d fastest).
    // got (graph output) flat order is whatever compute() returns for a [hd,heads,tok]
    // tensor: ne0=hd fastest => flat [tok][head][d] as well, i.e. got_idx == ref_idx.
    // If your compute() returns a different order, change `got_idx` here to match —
    // this index map is the load-bearing layout reconciliation; verify it by printing
    // got[0..3] vs ref[0..3] during bring-up before trusting the pass/fail.
    double mx = 0;
    for (int t=0;t<tokens;++t) for (int h=0;h<heads;++h) for (int d=0;d<hd;++d){
        size_t idx = ((size_t)t*heads + h)*hd + d;
        size_t got_idx = idx;                            // <-- adjust if layout differs
        double diff = std::fabs((double)got[got_idx] - (double)ref[idx]);
        if (diff > mx) mx = diff;
    }
    std::fprintf(stderr,"[rope] max-abs-diff=%.3e tol=%.3e\n", mx, (double)tol);
    return mx < tol;
}

}  // namespace la
```
NOTE on the cos/sin host layout: `build_rope_tables` writes `[token][pair]` (pair fastest). The ggml input is declared `[1, np, 1, tok]` so ne0=1, ne1=np (fastest after the size-1 dim), ne3=tok — the flat host order must be pair-fastest-within-token, which matches `[token*np + pair]`. Confirm the `add_graph_input_nd` flat-order contract matches (ne0 fastest); since ne0=1 here, ne1=np is the fastest real axis, so host order `[tok][pair]` is correct. **The `compare_rope` index map above is the load-bearing correctness check — when you read the graph output back, verify whether `compute()` returns it in `[hd,heads,tok]` (ne0=hd fastest) order and fix the `gg` index accordingly so it maps to the fixture's `[tok,heads,hd]`. Print a few elements during bring-up.** This layout reconciliation is the single most error-prone part of the task; do not hand-wave it.

- [ ] **Step 5: Build + run `test_vit_rope`** → iterate until PASS at 1e-4. If the max-abs-diff is large and structured (e.g. every other element wrong), the pair interleave (even/odd, x0/x1) or the cos/sin broadcast axis is off — fix the view offsets / index map.

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_vit_fixtures.py src/vit_rope.hpp src/vit_rope.cpp tests/test_vit_rope.cpp tests/CMakeLists.txt
git commit -m "M2: MoonViT 2D RoPE (host cos/sin + interleaved rotation) vs isolated fixture"
```

---

## Task 7: One encoder block → gate `vit_layer_00`

Assemble a full block: `x = x + wo(attn(rope(qkv(norm0(x))))); x = x + fc1(gelu(fc0(norm1(x))))`. Attention is full bidirectional (no mask), scale 1/sqrt(72), softmax in f32. Gate against `vit_layer_00` (the output of block 0, whose input is `vit_pos_added`).

**Files:**
- Modify: `src/vit_encoder.{hpp,cpp}` (add `VitLayerWeights`, `load_layer`, `build_block`)
- Modify: `tests/test_vit_encoder.cpp` (add the vit_layer_00 assertion)

- [ ] **Step 1: Add `VitLayerWeights` + `load_layer` to `src/vit_encoder.cpp`**

```cpp
struct VitLayerWeights {
    ggml_tensor *norm0_w,*norm0_b, *wqkv_w,*wqkv_b, *wo_w,*wo_b,
                *norm1_w,*norm1_b, *fc0_w,*fc0_b, *fc1_w,*fc1_b;
};
static VitLayerWeights load_layer(const la::ModelLoader& ml, int i){
    auto t=[&](const std::string& s){ return ml.tensor("vit.blk."+std::to_string(i)+"."+s); };
    return VitLayerWeights{
        t("norm0.weight"),t("norm0.bias"), t("wqkv.weight"),t("wqkv.bias"),
        t("wo.weight"),t("wo.bias"), t("norm1.weight"),t("norm1.bias"),
        t("mlp.fc0.weight"),t("mlp.fc0.bias"), t("mlp.fc1.weight"),t("mlp.fc1.bias")};
}
```

- [ ] **Step 2: Add `build_block` (the per-layer graph)**

```cpp
// x: [hidden=1152, tokens=1024]. rt: prebuilt rope tables for this grid.
static ggml_tensor* build_block(ggml_context* ctx, la::Backend& be, la::GraphInputPool& pool,
                                ggml_tensor* x, const VitLayerWeights& w,
                                const la::RopeTables& rt, int heads, int hd){
    const int H = heads, D = hd; const int tok = x->ne[1];
    // --- attention ---
    ggml_tensor* xn = la::layernorm(ctx, x, w.norm0_w, w.norm0_b, 1e-5f);
    ggml_tensor* qkv = la::linear(ctx, w.wqkv_w, xn, w.wqkv_b);        // [3456, tok]
    // reshape to [D, H, 3, tok]; unbind q/k/v along ne2
    ggml_tensor* qkv4 = ggml_reshape_4d(ctx, qkv, D, H, 3, tok);
    auto slice = [&](int idx){
        return ggml_cont(ctx, ggml_view_4d(ctx, qkv4, D,H,1,tok,
                  qkv4->nb[1],qkv4->nb[2],qkv4->nb[3], idx*qkv4->nb[2])); // [D,H,1,tok]
    };
    ggml_tensor* q = ggml_reshape_3d(ctx, slice(0), D,H,tok);
    ggml_tensor* k = ggml_reshape_3d(ctx, slice(1), D,H,tok);
    ggml_tensor* v = ggml_reshape_3d(ctx, slice(2), D,H,tok);
    q = la::apply_rope(ctx, be, pool, q, rt);
    k = la::apply_rope(ctx, be, pool, k, rt);
    // attention: permute to [D, tok, H]; scores = mul_mat(k,q) -> [tok,tok,H]
    ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0,2,1,3));  // [D,tok,H]
    ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0,2,1,3));
    ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 0,2,1,3));
    ggml_tensor* scores = ggml_mul_mat(ctx, kp, qp);                  // [tok,tok,H]
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f/std::sqrt((float)D), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vp, 1,0,2,3)); // [tok,D,H]
    ggml_tensor* o  = ggml_mul_mat(ctx, vt, scores);                  // [D,tok,H]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0,2,1,3));                // [D,H,tok]
    o = ggml_reshape_2d(ctx, o, D*H, tok);                            // [1152,tok]
    o = la::linear(ctx, w.wo_w, o, w.wo_b);
    x = ggml_add(ctx, x, o);
    // --- mlp ---
    ggml_tensor* mn = la::layernorm(ctx, x, w.norm1_w, w.norm1_b, 1e-5f);
    ggml_tensor* m  = la::linear(ctx, w.fc0_w, mn, w.fc0_b);
    m = la::gelu_tanh(ctx, m);
    m = la::linear(ctx, w.fc1_w, m, w.fc1_b);
    x = ggml_add(ctx, x, m);
    return x;
}
```
NOTE: verify the q/k/v unbind matches the packed layout (Q=first 1152, K=next, V=last; within each, head-major, d fastest). The `reshape_4d(D,H,3,tok)` assumes packed order `[d, head, qkv, tok]` (d fastest) — confirm against `modeling_vit.py`'s `view(*, 3, 16, 72)` (there the order is qkv outermost, then head, then d). If the GGUF stores wqkv as torch `[3456,1152]` with output rows ordered `[qkv][head][d]`, then after `mul_mat` the output ne0=3456 has that same order (d fastest within head, head within qkv), so `reshape_4d(D, H, 3, tok)` is correct (ne0=D fastest, then H, then qkv). Double-check by gating: if `vit_layer_00` fails but `vit_pos_added` passed, the q/k/v split or the attention permutes are the suspects. The permute/mul_mat pattern mirrors `vibevoice/src/qwen2.cpp:164-188` — read that for the exact ggml idiom.

- [ ] **Step 3: Add the vit_layer_00 assertion to `tests/test_vit_encoder.cpp`** — run patch_embed+posemb (Task 5) → one `build_block` with layer-0 weights, capture the output, compare against the `vit_layer_00` baseline (transpose to match [hidden,token] vs [token,hidden]). Gate atol 2e-2, rtol 2e-2 (one layer of accumulation).

- [ ] **Step 4: Build + run** → iterate to PASS. Use the layer-0 gate to debug the block. If diff is ~O(1), suspect q/k/v split or rope; if ~O(0.1) and diffuse, suspect softmax scale or layernorm eps.

- [ ] **Step 5: Commit**

```bash
git add src/vit_encoder.hpp src/vit_encoder.cpp tests/test_vit_encoder.cpp
git commit -m "M2: MoonViT encoder block (bidir attn + rope + gelu mlp) -> vit_layer_00 parity"
```

---

## Task 8: Full 27-block stack + final_norm → gate `vit_layer_26` and `vit_final`

**Files:**
- Modify: `src/vit_encoder.{hpp,cpp}` (loop all layers + final_norm; expose `forward_capture`)
- Modify: `tests/test_vit_encoder.cpp` (add vit_layer_26 + vit_final assertions)

- [ ] **Step 1: Add the full encoder forward to `src/vit_encoder.cpp`**

```cpp
// Public entry: pixel_host [588*1024] flattened, returns vit_final flat [1152*1024]
// and (optional) captures of intermediate layer outputs.
bool VitEncoder::forward(const std::vector<float>& pixel_host,
                         std::vector<float>& vit_final_out,
                         std::vector<int> capture_layers,
                         std::vector<std::vector<float>>& captured){
    la::RopeTables rt = la::build_rope_tables(32,32, cfg_.vit_n_heads, cfg_.vit_head_dim,
                                              cfg_.vit_rope_theta);
    std::vector<float> pos_host_T = make_posemb_T();   // bicubic + transpose to [1152,1024]
    captured.assign(capture_layers.size(), {});
    return be_.forward_capture([&](ggml_context* ctx)->ggml_tensor*{
        ggml_tensor* x = build_patch_and_pos(ctx, pixel_host, pos_host_T);  // [1152,1024]
        for (int i=0;i<(int)cfg_.vit_n_layers;++i){
            VitLayerWeights w = load_layer(ml_, i);
            x = build_block(ctx, be_, pool_, x, w, rt, cfg_.vit_n_heads, cfg_.vit_head_dim);
            for (size_t c=0;c<capture_layers.size();++c)
                if (capture_layers[c]==i) be_.capture(x, &captured[c]);
        }
        x = la::layernorm(ctx, x, ml_.tensor("vit.final_norm.weight"),
                          ml_.tensor("vit.final_norm.bias"), 1e-5f);
        return x;
    }, vit_final_out);
}
```
NOTE: bump `kGraphSize` in backend.cpp if 27 layers exceed the node budget (each block ~25 nodes × 27 + rope inputs ≈ ~1500 nodes — 16384 is ample, but verify no "graph too large" error). The rope tables are built once and reused across all layers (the input tensors are re-registered per `apply_rope` call — that's fine, they're small).

- [ ] **Step 2: Add vit_layer_26 + vit_final assertions to the test** — capture layer 26, compare vs `vit_layer_26`; compare final output vs `vit_final`. Gate atol 3e-2, rtol 3e-2 (27 layers of f32 accumulation; tighten if it comes in well under).

- [ ] **Step 3: Build + run** → PASS. If `vit_layer_00` passed but `vit_layer_26`/`vit_final` drift beyond tol, it's accumulation — confirm the per-layer diff grows gradually (print max-abs-diff at layers 0/13/26). A sudden jump at one layer means a layer-specific weight-name mismatch.

- [ ] **Step 4: Commit**

```bash
git add src/vit_encoder.hpp src/vit_encoder.cpp tests/test_vit_encoder.cpp
git commit -m "M2: full 27-layer MoonViT stack + final_norm -> vit_final parity"
```

---

## Task 9: Patch merger (2×2) → gate `merged_tokens` (M2 exit)

The merger groups 2×2 neighboring patches and channel-concatenates them in **TL, TR, BL, BR** order (row-major within the 2×2 block): `merged[m, e_idx*1152 + d] = vit_final[token(row=2a+b, col=2c+e), d]`, with merged index `m = a*16 + c`, `e_idx = b*2 + e`. Output `[256, 4608]`.

**Files:**
- Modify: `src/vit_encoder.{hpp,cpp}` (add `merge_patches`)
- Modify: `tests/test_vit_encoder.cpp` (add the merged_tokens gate)

- [ ] **Step 1: Implement the merger**

The cleanest correctness-first approach is a **host-side gather** of the captured `vit_final` (you already read it back as a vector), since the merger is pure reindexing with no math — do it on the host to avoid ggml permute index bugs, then (optionally) move it into the graph later if needed for the M3 pipeline. Add:
```cpp
// vit_final_hidden_token: flat [1152,1024] (ne0=1152 hidden fastest, ne1=1024 token).
// Returns merged [256,4608] flat (ne0=4608 fastest), TL,TR,BL,BR concat.
std::vector<float> merge_patches(const std::vector<float>& vf, int gh, int gw, int c){
    int mh=gh/2, mw=gw/2; int C4=c*4;
    std::vector<float> out((size_t)mh*mw*C4);
    for (int a=0;a<mh;++a) for (int cc=0;cc<mw;++cc){
        int m = a*mw + cc;
        for (int b=0;b<2;++b) for (int e=0;e<2;++e){
            int row=2*a+b, col=2*cc+e, e_idx=b*2+e;
            int tok = row*gw + col;
            for (int d=0; d<c; ++d)
                out[(size_t)m*C4 + e_idx*c + d] = vf[(size_t)tok*c + d];
        }
    }
    return out;
}
```
NOTE: `vf` here must be in `[token, hidden]` order (token-major) for the indexing `vf[tok*c + d]`. The graph output is `[hidden, token]` (ggml ne0=hidden); transpose on read, or index `vf[d*1024 + tok]` if you keep ne0=hidden. Be explicit and consistent — the test will catch a wrong order as a large diff.

- [ ] **Step 2: Add the merged_tokens gate to the test**

```cpp
std::vector<float> vfinal;  // from forward()
... vit.forward(px, vfinal, {}, captured);
std::vector<float> merged = la::merge_patches(to_token_major(vfinal,1152,1024), 32,32,1152);
std::vector<float> ref; std::vector<int64_t> s;
la_parity::load_baseline(base, "merged_tokens", ref, s);   // [256,4608]
bool ok = la_parity::compare(merged, ref, "merged_tokens", 3e-2f, 3e-2f);
return ok ? 0 : 1;
```

- [ ] **Step 3: Build + run** → PASS. If the diff is large and looks like a permutation (chunks of 1152 swapped), the concat order (TL/TR/BL/BR) or the `e_idx`/`m` mapping is wrong — re-derive from `modeling_vit.py`'s `view(16,2,16,2,1152).permute(0,2,1,3,4)`.

- [ ] **Step 4: Run the full suite + commit**

```bash
cd build && ctest --output-on-failure; cd ..
git add src/vit_encoder.hpp src/vit_encoder.cpp tests/test_vit_encoder.cpp
git commit -m "M2: 2x2 patch merger -> merged_tokens parity (M2 complete)"
```
Expected: all C++ tests (smoke, model_loader, backend, ggml_extend, vit_posemb, vit_rope, vit_encoder) PASS; the python suite still 9/9.

---

## Milestone exit criteria

**M2 done when:** `tests/test_vit_encoder.cpp` passes the full chain against the M0 baselines — `vit_pos_added` (≤1e-3), `vit_layer_00`/`vit_layer_26` (≤2e-2/3e-2), `vit_final` (≤3e-2), and `merged_tokens` (≤3e-2) — plus the isolated op gates (`posemb_interp` ≤1e-3, `rope` ≤1e-4) and the loader/backend/helper unit tests, all green under `ctest`. The MoonViT tower reproduces the HF reference from `pixel_values` to `merged_tokens`.

## Roadmap (next plan)

- **M3 — Projector + LM forward.** MLP projector (`proj.0/1/3` → `projected_tokens`), port vibevoice `qwen2.*` for the 36-layer Qwen2 decoder (cross-check llama.cpp `build_qwen2`), splice vision tokens into the `<IMG_CONTEXT>` (151665) slots, gate `embeds_after_splice` and `logits_step0` (the prompt prefill is plain causal — asserted in the M0 dump). Then M4 (AR decode), M5 (hybrid Parallel Box Decoding), M6 (quantize + CLI + C-API).
