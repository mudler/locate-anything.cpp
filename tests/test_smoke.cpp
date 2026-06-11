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
