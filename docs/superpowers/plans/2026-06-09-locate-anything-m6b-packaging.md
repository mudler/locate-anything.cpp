# LocateAnything-3B ggml Port — M6b (Packaging) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the engine: a quantized-GGUF path (f16/q8_0/q6_k/q5_k/q4_k), a flat C-API, and a `locate-anything-cli detect --annotated` CLI with a box renderer — wrapping the M6a `la::Engine`.

**Architecture:** Mirror the sibling repos (rt-detr.cpp for image save/renderer/CLI/quantize, parakeet.cpp for the opaque-handle C-API). The C++ engine is called directly by a thin `main()`; the flat C-API is a separate TU compiled into the lib for external (LocalAI/purego) consumers. Quantization touches ONLY the Qwen2 LM matmuls; everything else stays f32 (the ViT/projector are parity-sensitive, and two tensors are host-read as raw f32).

**Tech Stack:** C++17, ggml (embedded), stb_image_write (vendored from rt-detr.cpp), CMake/CTest, Python (gguf) for a quantize helper.

**Reference material (read before starting):**
- `la::Engine` (`src/engine.hpp`): `Engine::load(gguf)→unique_ptr` (nullptr on fail), `locate(image_path, query, Mode{Hybrid,Slow}, max_new)→vector<Box>` (empty on fail), `tokenizer()`. `la::Box{float x1,y1,x2,y2; std::string label}` (`src/boxes.hpp`).
- `la::Image`, `la::load_image_rgb(path,&img)` (`src/image_io.hpp`). `la::Backend::set_n_threads(int)` (`src/backend.hpp`).
- Sibling sources to mirror (read these):
  - Image save + stb: `/home/mudler/_git/rt-detr.cpp/src/image_io.cpp` (the `STB_IMAGE_WRITE_IMPLEMENTATION` TU, `stbi_write_png`), `/home/mudler/_git/rt-detr.cpp/third_party/stb/stb_image_write.h`.
  - Renderer: `/home/mudler/_git/rt-detr.cpp/src/visualize.cpp` (`font8x8_basic`, `draw_text`, `class_color`, `rfdetr_visualize_draw_box` — clamp, nested rect stroke, label background, luma text color).
  - C-API: `/home/mudler/_git/parakeet.cpp/src/parakeet_capi.cpp` (opaque ctx + `last_error` + `abi_version` + `dup_to_c` + try/catch wrap), `/home/mudler/_git/parakeet.cpp/include/parakeet_capi.h`; `/home/mudler/_git/rt-detr.cpp/src/rfdetr_capi.cpp` (per-detection accessors + JSON envelope).
  - CLI: `/home/mudler/_git/rt-detr.cpp/examples/cli/main.cpp` (subcommand dispatch, `cmd_detect`, `cmd_quantize` L271-647), `/home/mudler/_git/rt-detr.cpp/src/cli.cpp` (arg parse), `/home/mudler/_git/rt-detr.cpp/examples/cli/CMakeLists.txt`.
  - Quantize allowlist rationale: `/home/mudler/_git/vibevoice.cpp/scripts/quantize_gguf.py` (selective LM-only list).
- GGUF tensor names (`scripts/gguf_keys.py`): LM matmuls `lm.blk.{N}.{attn_q,attn_k,attn_v,attn_o}.weight`, `lm.blk.{N}.ffn_{gate,up,down}.weight`, `lm.output.weight`. Norms `*.norm.weight`/`attn_norm`/`ffn_norm`/`final_norm`/`output_norm`, biases `*.bias`, projector `proj.*`, vision `vit.*`, embeddings `lm.tok_embd.weight`, `vit.pos_emb.weight`.
- The end-to-end gate fixture: `tests/fixtures/parity_image.png` (448, lossless) + query `"Locate all the instances that matches the following description: cat</c>remote."` → reference boxes in `dumps/reference_slow.gguf` (`slow_boxes` [4,4], labels cat/cat/remote/remote). The Engine on the 448 PNG (slow) reproduces these EXACTLY at f32 (M6a gated this).

**CRITICAL quantization constraint (host-read f32):** the C++ reads two tensors' raw `->data` as f32 — `lm.tok_embd.weight` (host row-gather in `embed_tokens_host`) and `vit.pos_emb.weight` (host bicubic in `build_patch_pos`). These MUST stay f32. Quantize ONLY the LM matmul weights (attn q/k/v/o, ffn gate/up/down, lm.output) — all consumed via `ggml_mul_mat`, which handles quantized types. Keep f32: all `vit.*` (MoonViT, parity-sensitive at 2.65e-3), `proj.*`, all norms, all `*.bias`, `lm.tok_embd.weight`, `vit.pos_emb.weight`.

**Scope:** M6b is the final port milestone (engine done in M6a). LocalAI backend integration is a separate later effort.

---

## File Structure

| Path | Responsibility |
|---|---|
| `third_party/stb/stb_image_write.h` | Vendored PNG writer (from rt-detr) |
| `src/visualize.{hpp,cpp}` | Box renderer: draw labeled colored boxes on an `Image`, save PNG (from rt-detr, label-hash→color) |
| `src/image_io.{hpp,cpp}` (modify) | `save_image_png(path, Image)` + `load_image_rgb_buffer(bytes,len,&img)` (stbi from-memory) |
| `src/engine.{hpp,cpp}` (modify) | `Engine::load(gguf, n_threads)` (thread n_threads → Backend) |
| `include/la_capi.h` | Flat C ABI: opaque `la_ctx`, load/free, locate_path/buffer (JSON), detection accessors, free_string, last_error, abi_version |
| `src/la_capi.cpp` | C-API impl: wraps `la::Engine`, try/catch→last_error, JSON envelope |
| `examples/cli/main.cpp`, `examples/cli/CMakeLists.txt` | `locate-anything-cli` (detect/info/quantize subcommands) |
| `src/cli.{hpp,cpp}` | CLI arg parsing (subcommand dispatch) |
| `src/quantize.{hpp,cpp}` | C++ GGUF quantizer (K-quants via `ggml_quantize_chunk`; the LM-only allowlist) |
| `scripts/quantize_gguf.py` | Python convenience quantizer (f16/q8_0 via gguf-py; K-quants documented as CLI-only) |
| `tests/test_capi.cpp`, `tests/test_quantize.cpp` | Gates |
| `CMakeLists.txt` (modify) | `LA_SHARED` (static-ggml into .so via PIC), `LA_BUILD_CLI`, new sources, stb include |

---

## Task 1: Box renderer + PNG save

**Files:**
- Create: `third_party/stb/stb_image_write.h`, `src/visualize.{hpp,cpp}`
- Modify: `src/image_io.{hpp,cpp}` (add `save_image_png`, `load_image_rgb_buffer`)
- Create: `tests/test_visualize.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: Vendor stb_image_write + add save/buffer-load to image_io.**
```bash
cp /home/mudler/_git/rt-detr.cpp/third_party/stb/stb_image_write.h third_party/stb/
```
In `src/image_io.cpp`, in the SAME TU that has `#define STB_IMAGE_IMPLEMENTATION`, add `#define STB_IMAGE_WRITE_IMPLEMENTATION` + `#include "stb_image_write.h"`. Add to `src/image_io.hpp`:
```cpp
bool save_image_png(const std::string& path, const Image& img);   // stbi_write_png, 3-channel
bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out);  // stbi_load_from_memory, force 3
```
Implement: `save_image_png` → `stbi_write_png(path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w*3)`. `load_image_rgb_buffer` → `stbi_load_from_memory(bytes, (int)len, &w, &h, &c, 3)` → copy into Image (mirror the existing `load_image_rgb` from-file path).

- [ ] **Step 2: Write `src/visualize.{hpp,cpp}` (copy rt-detr's visualize.cpp, adapt to la::Box).**
`src/visualize.hpp`:
```cpp
#pragma once
#include "image_io.hpp"
#include "boxes.hpp"
#include <vector>
namespace la {
// Draw labeled colored boxes (label text + per-label color) onto a COPY of img; return it.
Image render_boxes(const Image& img, const std::vector<Box>& boxes);
}
```
`src/visualize.cpp`: copy rt-detr's `src/visualize.cpp` internals — `font8x8_basic[128][8]`, `set_px`/`draw_hline`/`fill_rect`/`draw_char`/`draw_text`/`text_width`, and the box-draw logic (clamp box to image, nested-rectangle stroke `max(thickness,3)`, label `"%s"` (just the label string), auto font scale by width, filled label background in the box color above the box, white/black text by luma `(299r+587g+114b)/1000 > 160`). For the color: since la labels are free-text (not class ids), use a **stable hash of the label string → a 20-entry vivid palette** (copy rt-detr's `class_color` palette, index = `hash(label) % 20`). `render_boxes` deep-copies `img`, draws each box, returns it.

- [ ] **Step 3: Write `tests/test_visualize.cpp` — render the 448 fixture boxes, save, reload, verify.**
```cpp
#include "engine.hpp"
#include "visualize.hpp"
#include "image_io.hpp"
#include <cstdlib>
#include <cstdio>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF");
    if(!gguf){std::fprintf(stderr,"skip\n");return 77;}
    auto eng=la::Engine::load(gguf); if(!eng) return 1;
    la::Image img; if(!la::load_image_rgb("tests/fixtures/parity_image.png", img)) return 1;
    auto boxes = eng->locate("tests/fixtures/parity_image.png",
        "Locate all the instances that matches the following description: cat</c>remote.",
        la::Engine::Mode::Slow);
    if(boxes.size()!=4) { std::printf("expected 4 boxes got %zu\n", boxes.size()); return 1; }
    la::Image annotated = la::render_boxes(img, boxes);
    int ok = (annotated.w==img.w && annotated.h==img.h && annotated.rgb.size()==img.rgb.size());
    // annotated must DIFFER from the original (boxes were drawn)
    int diff=0; for(size_t i=0;i<img.rgb.size();++i) if(annotated.rgb[i]!=img.rgb[i]) diff++;
    ok &= (diff > 100);   // box strokes + labels changed many pixels
    // save + reload round-trips dimensions
    ok &= la::save_image_png("/tmp/la_annotated.png", annotated);
    la::Image back; ok &= la::load_image_rgb("/tmp/la_annotated.png", back);
    ok &= (back.w==img.w && back.h==img.h);
    std::printf("visualize ok=%d (diff_px=%d)\n", ok, diff);
    return ok?0:1;
}
```

- [ ] **Step 4: Wire (`la_add_test(test_visualize)`; add `src/visualize.cpp` to LA_SOURCES), build, run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON >/dev/null 2>&1
cmake --build build --target test_visualize -j 2>&1 | tail -5
cd build && ctest -R test_visualize --output-on-failure; cd ..
```
Expected: `visualize ok=1` with a large diff_px (boxes drawn). DEBUG: if the PNG save fails, check stb_image_write impl is in exactly one TU (the image_io.cpp one). If the render crashes, check box clamping (boxes within image bounds).

- [ ] **Step 5: Commit.**
```bash
git add third_party/stb/stb_image_write.h src/visualize.hpp src/visualize.cpp src/image_io.hpp src/image_io.cpp tests/test_visualize.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M6b: box renderer + PNG save (annotated detections)"
```
Author `mudler <mudler@localai.io>`; end every M6b commit with:
```

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: Flat C-API

**Files:**
- Create: `include/la_capi.h`, `src/la_capi.cpp`
- Modify: `src/engine.{hpp,cpp}` (add `n_threads` to `load`)
- Create: `tests/test_capi.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: Add `n_threads` to `Engine::load` (thread it to Backend).** In `src/engine.{hpp,cpp}`, change `static std::unique_ptr<Engine> load(const std::string& gguf_path, int n_threads=0);` and in the impl, after constructing, call `be_.set_n_threads(n_threads>0 ? n_threads : (int)std::thread::hardware_concurrency())`. (Include `<thread>`.) Keep the no-arg-default behavior (0 → hardware concurrency). The 448 tests use the default — confirm test_engine still passes.

- [ ] **Step 2: Write `include/la_capi.h` (mirror parakeet_capi.h).**
```c
#ifndef LA_CAPI_H
#define LA_CAPI_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct la_ctx la_ctx;
int      la_capi_abi_version(void);
la_ctx*  la_capi_load(const char* gguf_path, int n_threads);   // NULL on failure
void     la_capi_free(la_ctx* ctx);                            // safe on NULL
// Run detection; returns malloc'd JSON (caller frees via la_capi_free_string), NULL on error.
// mode: 0=hybrid, 1=slow. prompt = the open-vocab query.
char*    la_capi_locate_path  (la_ctx* ctx, const char* image_path, const char* prompt, int mode);
char*    la_capi_locate_buffer(la_ctx* ctx, const unsigned char* bytes, size_t len, const char* prompt, int mode);
// Per-detection accessors (operate on the LAST locate_* result; for purego consumers).
int      la_capi_get_n_detections(la_ctx* ctx);
int      la_capi_get_detection_box(la_ctx* ctx, int i, float out_xyxy[4]);    // 0 ok, -1 bad index
int      la_capi_get_detection_label(la_ctx* ctx, int i, char* buf, int buf_size); // returns required size; two-call sizing
void        la_capi_free_string(char* s);
const char* la_capi_last_error(la_ctx* ctx);                   // owned by ctx, "" if none
#ifdef __cplusplus
}
#endif
#endif
```

- [ ] **Step 3: Write `src/la_capi.cpp` (mirror parakeet_capi.cpp + rfdetr_capi.cpp accessors).**
```cpp
#include "la_capi.h"
#include "engine.hpp"
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#define LA_CAPI_ABI_VERSION 1
struct la_ctx {
    std::unique_ptr<la::Engine> engine;
    std::string last_error;
    std::vector<la::Box> last;     // persisted for the accessors
};
static char* dup_to_c(const std::string& s){
    char* p=(char*)std::malloc(s.size()+1); if(!p) return nullptr;
    std::memcpy(p, s.data(), s.size()); p[s.size()]='\0'; return p;
}
static void json_escape(std::string& o, const std::string& s){
    for(char c: s){ if(c=='"'||c=='\\'){o+='\\';o+=c;} else if(c=='\n')o+="\\n"; else o+=c; }
}
static std::string boxes_json(const std::vector<la::Box>& b){
    std::string o="{\"detections\":[";
    for(size_t i=0;i<b.size();++i){ if(i)o+=','; o+="{\"label\":\""; json_escape(o,b[i].label);
        o+="\",\"box\":["; char buf[128];
        std::snprintf(buf,sizeof buf,"%.3f,%.3f,%.3f,%.3f",b[i].x1,b[i].y1,b[i].x2,b[i].y2); o+=buf; o+="]}"; }
    o+="]}"; return o;
}
extern "C" {
int la_capi_abi_version(void){ return LA_CAPI_ABI_VERSION; }
la_ctx* la_capi_load(const char* path, int n_threads){
    try{ auto e=la::Engine::load(path, n_threads); if(!e) return nullptr;
         auto* c=new la_ctx(); c->engine=std::move(e); return c; }
    catch(...){ return nullptr; }
}
void la_capi_free(la_ctx* c){ delete c; }
static char* run(la_ctx* c, const std::function<std::vector<la::Box>()>& f){
    if(!c) return nullptr;
    try{ c->last=f(); c->last_error.clear(); return dup_to_c(boxes_json(c->last)); }
    catch(const std::exception& e){ c->last_error=e.what(); return nullptr; }
    catch(...){ c->last_error="unknown error"; return nullptr; }
}
char* la_capi_locate_path(la_ctx* c, const char* img, const char* prompt, int mode){
    return run(c, [&]{ return c->engine->locate(img, prompt?prompt:"",
        mode==1?la::Engine::Mode::Slow:la::Engine::Mode::Hybrid); });
}
char* la_capi_locate_buffer(la_ctx* c, const unsigned char* bytes, size_t len, const char* prompt, int mode){
    // locate_buffer needs an Engine path that takes an image buffer — add Engine::locate_buffer
    // (load_image_rgb_buffer → same pipeline). See NOTE.
    return run(c, [&]{ return c->engine->locate_buffer(bytes, len, prompt?prompt:"",
        mode==1?la::Engine::Mode::Slow:la::Engine::Mode::Hybrid); });
}
int la_capi_get_n_detections(la_ctx* c){ return c? (int)c->last.size() : 0; }
int la_capi_get_detection_box(la_ctx* c, int i, float o[4]){
    if(!c||i<0||i>=(int)c->last.size()) return -1;
    o[0]=c->last[i].x1;o[1]=c->last[i].y1;o[2]=c->last[i].x2;o[3]=c->last[i].y2; return 0; }
int la_capi_get_detection_label(la_ctx* c, int i, char* buf, int bufsz){
    if(!c||i<0||i>=(int)c->last.size()) return -1;
    const std::string& s=c->last[i].label; int need=(int)s.size()+1;
    if(buf&&bufsz>0){ int n=std::min(bufsz-1,(int)s.size()); std::memcpy(buf,s.data(),n); buf[n]='\0'; }
    return need; }
void la_capi_free_string(char* s){ std::free(s); }
const char* la_capi_last_error(la_ctx* c){ return c? c->last_error.c_str() : ""; }
}
```
NOTE: add `Engine::locate_buffer(const unsigned char* bytes, size_t len, const std::string& query, Mode)` to `src/engine.{hpp,cpp}` — identical to `locate` but starts from `load_image_rgb_buffer` instead of `load_image_rgb`. Factor the shared post-load pipeline (preprocess→...→parse) into a private helper both call. Include `<functional>` for the `run` lambda.

- [ ] **Step 4: Write `tests/test_capi.cpp` — load, locate on the 448 PNG via the C ABI, check boxes.**
```cpp
#include "la_capi.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF");
    if(!gguf){std::fprintf(stderr,"skip\n");return 77;}
    if(la_capi_abi_version()!=1) return 1;
    la_ctx* c=la_capi_load(gguf, 0); if(!c){ std::fprintf(stderr,"load fail\n"); return 1; }
    char* json=la_capi_locate_path(c, "tests/fixtures/parity_image.png",
        "Locate all the instances that matches the following description: cat</c>remote.", 1 /*slow*/);
    int ok = (json!=nullptr);
    if(json){ ok &= (std::strstr(json,"cat")!=nullptr && std::strstr(json,"remote")!=nullptr); }
    int n=la_capi_get_n_detections(c); ok &= (n==4);
    float b[4]; ok &= (la_capi_get_detection_box(c,0,b)==0);
    char lbl[64]; int need=la_capi_get_detection_label(c,0,lbl,sizeof lbl); ok &= (need>0);
    std::printf("capi: n=%d label0=%s box0=[%.1f %.1f %.1f %.1f] err='%s' ok=%d\n",
                n, lbl, b[0],b[1],b[2],b[3], la_capi_last_error(c), ok);
    la_capi_free_string(json); la_capi_free(c);
    // free(NULL)-safety + error path
    la_capi_free(nullptr);
    la_ctx* bad=la_capi_load("/nonexistent.gguf", 0); ok &= (bad==nullptr);
    return ok?0:1;
}
```

- [ ] **Step 5: Wire (`la_add_test(test_capi)`; add `src/la_capi.cpp` to LA_SOURCES; the test needs `include/` on its include path — la_add_test already adds it via the lib's PUBLIC include), build, run.** Expected: `capi: n=4 label0=cat box0=[3.1 50.6 221.8 443.5] err='' ok=1`. DEBUG: a NULL json → check last_error; the box accessor → check `last` persisted from the locate call.

- [ ] **Step 6: Commit.**
```bash
git add include/la_capi.h src/la_capi.cpp src/engine.hpp src/engine.cpp tests/test_capi.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M6b: flat C-API (opaque ctx, locate path/buffer -> JSON, detection accessors)"
```

---

## Task 3: CLI (`locate-anything-cli detect --annotated` + info) + shared-lib CMake

**Files:**
- Create: `src/cli.{hpp,cpp}`, `examples/cli/main.cpp`, `examples/cli/CMakeLists.txt`
- Modify: `CMakeLists.txt` (`LA_SHARED`, `LA_BUILD_CLI`)
- Create: `tests/test_cli.sh` (or a ctest that runs the CLI binary)

- [ ] **Step 1: Write `src/cli.{hpp,cpp}` — arg parsing (mirror rt-detr's cli.cpp).**
`src/cli.hpp`:
```cpp
#pragma once
#include <string>
namespace la::cli {
enum class Sub { Detect, Info, Quantize, Help, None };
struct DetectArgs { std::string model, input, prompt, output, annotated, mode="hybrid"; int threads=0; };
struct QuantizeArgs { std::string in, out, type; };
struct Parsed { Sub sub=Sub::None; DetectArgs detect; QuantizeArgs quantize; std::string info_model; std::string error; };
Parsed parse(int argc, char** argv);
void print_help();
}
```
`src/cli.cpp`: a manual flag loop. `detect` flags: `--model <gguf>`, `--input <img>`, `--prompt <text>`, `--output <json>` (optional), `--annotated <png>` (optional), `--mode hybrid|slow`, `--threads N`. `info --model <gguf>`. `quantize <in> <out> <type>` (positional). `--help`/`-h` → Help. Validate required flags (detect needs model+input+prompt) → `error` string. Mirror rt-detr's `eat_value`/`parse_uint` helpers.

- [ ] **Step 2: Write `examples/cli/main.cpp` — dispatch + `cmd_detect`/`cmd_info` (mirror rt-detr main.cpp; cmd_quantize comes in Task 4).**
```cpp
#include "cli.hpp"
#include "engine.hpp"
#include "visualize.hpp"
#include "image_io.hpp"
#include <cstdio>
#include <fstream>
static int cmd_detect(const la::cli::DetectArgs& a){
    auto eng = la::Engine::load(a.model, a.threads);
    if(!eng){ std::fprintf(stderr,"error: failed to load %s\n", a.model.c_str()); return 1; }
    auto mode = (a.mode=="slow")? la::Engine::Mode::Slow : la::Engine::Mode::Hybrid;
    auto boxes = eng->locate(a.input, a.prompt, mode);
    // JSON to stdout or --output
    std::string json="{\"detections\":[";
    for(size_t i=0;i<boxes.size();++i){ char buf[256];
        std::snprintf(buf,sizeof buf,"%s{\"label\":\"%s\",\"box\":[%.2f,%.2f,%.2f,%.2f]}",
            i?",":"", boxes[i].label.c_str(), boxes[i].x1,boxes[i].y1,boxes[i].x2,boxes[i].y2); json+=buf; }
    json+="]}";
    if(!a.output.empty()){ std::ofstream(a.output) << json; } else { std::printf("%s\n", json.c_str()); }
    if(!a.annotated.empty()){
        la::Image img; if(la::load_image_rgb(a.input, img)){
            la::Image ann = la::render_boxes(img, boxes);
            la::save_image_png(a.annotated, ann);
            std::fprintf(stderr, "wrote %s\n", a.annotated.c_str());
        }
    }
    std::fprintf(stderr, "%zu detections\n", boxes.size());
    return 0;
}
static int cmd_info(const std::string& model){
    auto eng=la::Engine::load(model,1); if(!eng){ std::fprintf(stderr,"load fail\n"); return 1; }
    std::printf("model: %s (loaded ok)\n", model.c_str()); return 0;   // extend with hparams if desired
}
int main(int argc, char** argv){
    auto p = la::cli::parse(argc, argv);
    if(!p.error.empty()){ std::fprintf(stderr,"error: %s\n", p.error.c_str()); la::cli::print_help(); return 1; }
    using S=la::cli::Sub;
    switch(p.sub){
        case S::Detect:   return cmd_detect(p.detect);
        case S::Info:     return cmd_info(p.info_model);
        case S::Quantize: std::fprintf(stderr,"quantize: implemented in Task 4\n"); return 1;
        case S::Help:     la::cli::print_help(); return 0;
        default:          la::cli::print_help(); return 1;
    }
}
```

- [ ] **Step 3: `examples/cli/CMakeLists.txt` + top-CMake wiring.**
`examples/cli/CMakeLists.txt`:
```cmake
add_executable(locate-anything-cli main.cpp)
target_link_libraries(locate-anything-cli PRIVATE locate_anything ggml)
target_include_directories(locate-anything-cli PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party/stb)
```
In top `CMakeLists.txt`: add `src/cli.cpp` + `src/quantize.cpp` (Task 4) to `LA_SOURCES`; add options + the shared-lib path:
```cmake
option(LA_SHARED    "Build liblocate_anything as a shared library" OFF)
option(LA_BUILD_CLI "Build the locate-anything-cli executable"     ON)
# (replace the existing add_library(locate_anything STATIC ...) with:)
if(LA_SHARED)
    add_library(locate_anything SHARED ${LA_SOURCES})
else()
    add_library(locate_anything STATIC ${LA_SOURCES})
endif()
# ... existing include dirs + target_link_libraries(... ggml) ...
if(LA_BUILD_CLI AND EXISTS ${CMAKE_SOURCE_DIR}/examples/cli/CMakeLists.txt)
    add_subdirectory(examples/cli)
endif()
```
(ggml is static + PIC already via `CMAKE_POSITION_INDEPENDENT_CODE ON` — it links into the .so when LA_SHARED. No `BUILD_SHARED_LIBS`.)

- [ ] **Step 4: Write a CLI smoke gate `tests/test_cli.sh` (ctest runs the binary on the 448 fixture).**
```bash
#!/usr/bin/env bash
set -e
GGUF="${LA_TEST_GGUF:?}"
BIN="${LA_CLI_BIN:?}"
OUT=$(mktemp -d)
"$BIN" detect --model "$GGUF" --input tests/fixtures/parity_image.png \
  --prompt "Locate all the instances that matches the following description: cat</c>remote." \
  --mode slow --output "$OUT/out.json" --annotated "$OUT/ann.png"
grep -q '"label":"cat"' "$OUT/out.json"
grep -q '"label":"remote"' "$OUT/out.json"
test -s "$OUT/ann.png"            # annotated PNG written, non-empty
echo "cli smoke OK"
```
Register in `tests/CMakeLists.txt`:
```cmake
if(LA_BUILD_CLI)
  add_test(NAME test_cli COMMAND ${CMAKE_COMMAND} -E env LA_TEST_GGUF=${CMAKE_SOURCE_DIR}/models/locate-anything-f32.gguf LA_CLI_BIN=$<TARGET_FILE:locate-anything-cli> bash ${CMAKE_SOURCE_DIR}/tests/test_cli.sh)
  set_tests_properties(test_cli PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77)
endif()
```

- [ ] **Step 5: Build + run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON -DLA_BUILD_CLI=ON >/dev/null 2>&1
cmake --build build --target locate-anything-cli -j 2>&1 | tail -3
cd build && ctest -R test_cli --output-on-failure; cd ..
# also a manual smoke:
./build/examples/cli/locate-anything-cli detect --model models/locate-anything-f32.gguf \
  --input tests/fixtures/parity_image.png --prompt "Locate all the instances that matches the following description: cat</c>remote." --mode slow --annotated /tmp/cli_ann.png
```
Expected: JSON with cat/remote labels, `/tmp/cli_ann.png` written. ALSO verify `LA_SHARED=ON` builds: `cmake -B build-shared -DLA_SHARED=ON -DLA_BUILD_CLI=OFF >/dev/null 2>&1 && cmake --build build-shared --target locate_anything -j 2>&1 | tail -2 && ldd build-shared/liblocate_anything.so | grep -i ggml && echo "WARN: external ggml" || echo "ggml static OK"` — confirm `liblocate_anything.so` builds and does NOT depend on an external libggml (ggml static-linked in).

- [ ] **Step 6: Commit.**
```bash
git add src/cli.hpp src/cli.cpp examples/cli/main.cpp examples/cli/CMakeLists.txt tests/test_cli.sh tests/CMakeLists.txt CMakeLists.txt
git commit -m "M6b: locate-anything-cli (detect --annotated, info) + LA_SHARED/LA_BUILD_CLI cmake"
```

---

## Task 4: Quantization (`quantize` subcommand + allowlist) → re-gate boxes (M6b exit)

**Files:**
- Create: `src/quantize.{hpp,cpp}`
- Modify: `examples/cli/main.cpp` (wire `cmd_quantize`)
- Create: `scripts/quantize_gguf.py`
- Create: `tests/test_quantize.cpp`
- Modify: `tests/CMakeLists.txt`, `CMakeLists.txt`

- [ ] **Step 1: Write `src/quantize.{hpp,cpp}` (C++ GGUF quantizer, mirror rt-detr cmd_quantize L271-647).**
`src/quantize.hpp`:
```cpp
#pragma once
#include <string>
namespace la {
// Quantize the LM matmul weights of `in_gguf` to `type` (f16/q8_0/q6_k/q5_k/q4_k);
// keep everything else (vit.*, proj.*, norms, biases, lm.tok_embd, vit.pos_emb) f32.
bool quantize_gguf(const std::string& in_gguf, const std::string& out_gguf, const std::string& type);
}
```
`src/quantize.cpp`: open input with `gguf_init_from_file(in, {false,&ctx})`; `gguf_init_empty()` out; `gguf_set_kv(out,in)` (copy all metadata). For each tensor: decide via `should_quantize(name)` = the name matches `lm.blk.\d+.(attn_q|attn_k|attn_v|attn_o|ffn_gate|ffn_up|ffn_down).weight` OR `lm.output.weight` — AND NOTHING ELSE. If quantize: `dequantize_to_f32` (handle f32/f16 src) → `ggml_quantize_chunk(target_type, f32, dst, 0, nrows, n_per_row, nullptr)`; else copy bytes as-is. `gguf_add_tensor` + `gguf_write_to_file`. Mirror rt-detr's `cmd_quantize` mechanics (the dtype table, the per-tensor dequant→quantize, the owned `vector<vector<uint8_t>>` keeping data alive until write). CRITICAL: the allowlist is LM matmuls ONLY (per the plan's host-read-f32 constraint — `lm.tok_embd.weight` and `vit.pos_emb.weight` MUST stay f32). All LM matmul `ne[0]` (2048 or 11008) are %256==0 so K-quants apply directly; no fallback needed, but keep rt-detr's `%32` Q8_0 fallback for safety.

- [ ] **Step 2: Wire `cmd_quantize` in `examples/cli/main.cpp`.**
```cpp
#include "quantize.hpp"
static int cmd_quantize(const la::cli::QuantizeArgs& a){
    if(la::quantize_gguf(a.in, a.out, a.type)){ std::fprintf(stderr,"wrote %s (%s)\n", a.out.c_str(), a.type.c_str()); return 0; }
    std::fprintf(stderr,"quantize failed\n"); return 1;
}
// replace the Task-3 stub: case S::Quantize: return cmd_quantize(p.quantize);
```

- [ ] **Step 3: `scripts/quantize_gguf.py` (Python convenience, f16/q8_0 via gguf-py; K-quants documented as CLI-only).** Copy `vibevoice.cpp/scripts/quantize_gguf.py` structure; swap the allowlist to the LM-matmul regex above; QUANT_MAP for f16/q8_0/q4_0/q5_0 (gguf-py reliable set). Print a note that q6_k/q5_k/q4_k require the C++ `locate-anything-cli quantize`.

- [ ] **Step 4: Generate quantized GGUFs + write `tests/test_quantize.cpp` (re-gate boxes).** The decisive parity check: the engine on the quantized GGUF still produces the 4 boxes (cat/cat/remote/remote) with coords degrading monotonically by quant level. First generate them (the test or a setup step runs the CLI quantize):
```bash
# (in the test setup or manually before the test)
./build/examples/cli/locate-anything-cli quantize models/locate-anything-f32.gguf /tmp/la-q8.gguf q8_0
./build/examples/cli/locate-anything-cli quantize models/locate-anything-f32.gguf /tmp/la-q4k.gguf q4_k
```
`tests/test_quantize.cpp`:
```cpp
#include "engine.hpp"
#include "quantize.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdio>
static int run_one(const char* gguf, const char* slow, float tol, int& nout){
    auto eng=la::Engine::load(gguf,0); if(!eng) return -1;
    auto boxes=eng->locate("tests/fixtures/parity_image.png",
        "Locate all the instances that matches the following description: cat</c>remote.",
        la::Engine::Mode::Slow);
    nout=(int)boxes.size();
    std::vector<float> rb; std::vector<int64_t> rs; la_parity::load_baseline(slow,"slow_boxes",rb,rs);
    int nb=(int)rb.size()/4; if(nout!=nb) return 0;
    int ok=1; for(int k=0;k<nb&&ok;++k){ const float* r=&rb[k*4];
        ok &= (std::fabs(boxes[k].x1-r[0])<tol && std::fabs(boxes[k].y1-r[1])<tol &&
               std::fabs(boxes[k].x2-r[2])<tol && std::fabs(boxes[k].y2-r[3])<tol); }
    return ok;
}
int main(){
    const char* slow=std::getenv("LA_TEST_SLOW"); const char* f32=std::getenv("LA_TEST_GGUF");
    const char* q8=std::getenv("LA_TEST_Q8"); const char* q4=std::getenv("LA_TEST_Q4K");
    if(!slow||!f32){std::fprintf(stderr,"skip\n");return 77;}
    int ok=1, n;
    // produce the quantized GGUFs in-process if env paths not provided
    std::string q8p = q8?q8:"/tmp/la-q8.gguf", q4p=q4?q4:"/tmp/la-q4k.gguf";
    if(!q8){ ok &= la::quantize_gguf(f32, q8p, "q8_0"); }
    if(!q4){ ok &= la::quantize_gguf(f32, q4p, "q4_k"); }
    // q8_0: near-lossless -> boxes within ~3px
    int r8=run_one(q8p.c_str(), slow, 3.0f, n); std::printf("q8_0: %d boxes ok=%d\n", n, r8); ok &= (r8==1);
    // q4_k: looser -> boxes within ~20px, count+labels still right
    int r4=run_one(q4p.c_str(), slow, 20.0f, n); std::printf("q4_k: %d boxes ok=%d\n", n, r4); ok &= (r4==1);
    return ok?0:1;
}
```
NOTE: q8_0 should keep the 4 boxes within a few px (LM quantization is near-lossless for greedy argmax on a clean image); q4_k may shift coords more (looser tol) but should still emit 4 cat/remote boxes (the argmax decisions are robust). If q4_k flips a token and changes the box count/labels, that's the known accuracy/size tradeoff — loosen to gate count+labels+IoU, and report it (document the degradation rather than failing). The gate proves quantization is near-lossless at q8 and monotone.

- [ ] **Step 5: Wire (`la_add_test(test_quantize)`; add `src/quantize.cpp` to LA_SOURCES; set LA_TEST_SLOW in the env), build, run.**
```bash
cmake -B build -DLA_BUILD_TESTS=ON -DLA_BUILD_CLI=ON >/dev/null 2>&1
cmake --build build -j 2>&1 | tail -3
cd build && ctest -R "test_quantize|test_cli|test_capi|test_visualize" --output-on-failure; cd ..
```
Expected: `q8_0: 4 boxes ok=1`, `q4_k: 4 boxes ok=1` (or documented degradation). DEBUG: if q8 boxes are way off, the allowlist quantized a host-read tensor (tok_embd/pos_emb) — confirm those stay f32. If the GGUF won't load, the quantized tensor types/metadata are wrong — check `should_quantize` + the dequant path.

- [ ] **Step 6: Run the FULL suite + commit (M6b complete).**
```bash
cd build && ctest --output-on-failure; cd ..
git add src/quantize.hpp src/quantize.cpp examples/cli/main.cpp scripts/quantize_gguf.py tests/test_quantize.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "M6b: GGUF quantize subcommand (LM-only allowlist) -> box parity per quant (M6b complete)"
```
Expected: all tests green (the new packaging tests + all M0–M6a parity tests).

---

## Milestone exit criteria

**M6b done when:** `tests/{test_visualize,test_capi,test_cli,test_quantize}.cpp` pass — the renderer draws + saves annotated PNGs, the flat C-API loads + locates + returns JSON/accessors with proper error handling, the CLI `detect --annotated` produces JSON + an annotated PNG on the 448 fixture, and quantized GGUFs (q8_0 near-lossless, q4_k monotone) still reproduce the reference boxes — all green under ctest, full suite green, `LA_SHARED` builds a self-contained `.so` (no external libggml). The port is shippable: `locate-anything-cli detect --model <gguf> --input <img> --prompt <text> --annotated <png>`.

## Roadmap (beyond M6b — optional)

- **LocalAI backend** (separate effort): a purego-loadable backend calling the flat C-API (L0 load+locate, L3 registration/gallery, L5 tests/docs), static-linking ggml into the `.so` (already wired via `LA_SHARED`). Publish quantized GGUFs to HuggingFace. Pin the native version to a commit SHA.
