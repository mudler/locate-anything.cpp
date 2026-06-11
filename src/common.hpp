#pragma once
#include <cstdio>
// LA_LOG is a preprocessor macro (file-scope, not namespaced).
#define LA_LOG(...) do { std::fprintf(stderr, "[la] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
