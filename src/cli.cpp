#include "cli.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace la::cli {

namespace {

bool eat_value(int argc, char** argv, int& i, const char* flag, std::string& out, std::string& err) {
    if (i + 1 >= argc) {
        err = std::string("missing value for ") + flag;
        return false;
    }
    out = argv[++i];
    return true;
}

bool parse_int(const std::string& s, int& out, std::string& err, const char* flag) {
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE) {
        err = std::string("invalid integer for ") + flag + ": " + s;
        return false;
    }
    out = (int)v;
    return true;
}

}  // namespace

void print_help() {
    std::puts(
        "locate-anything-cli — LocateAnything inference CLI\n"
        "\n"
        "Usage:\n"
        "  locate-anything-cli detect   --model <gguf> --input <image> --prompt <text>\n"
        "                               [--output <json>] [--annotated <png>]\n"
        "                               [--mode hybrid|slow|fast] [--threads N]\n"
        "  locate-anything-cli info     --model <gguf>\n"
        "  locate-anything-cli quantize <input.gguf> <output.gguf> <type>\n"
        "  locate-anything-cli --help\n"
        "\n"
        "Options:\n"
        "  --model <gguf>      path to the model GGUF (required for detect/info)\n"
        "  --input <image>     input image path (required for detect)\n"
        "  --prompt <text>     detection prompt / query (required for detect)\n"
        "  --output <json>     write detections JSON here (default: stdout)\n"
        "  --annotated <png>   write an annotated PNG with rendered boxes\n"
        "  --mode hybrid|slow|fast  decode mode (default: hybrid; fast=MTP-only)\n"
        "  --threads N         CPU threads for ggml (0 = auto, default 0)\n");
}

Parsed parse(int argc, char** argv) {
    Parsed r;

    if (argc < 2) { r.sub = Sub::Help; return r; }

    std::string first = argv[1];
    if (first == "--help" || first == "-h" || first == "help") { r.sub = Sub::Help; return r; }

    if (first == "detect") {
        r.sub = Sub::Detect;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--model")     { if (!eat_value(argc, argv, i, "--model",     r.detect.model,     r.error)) return r; }
            else if (a == "--input")     { if (!eat_value(argc, argv, i, "--input",     r.detect.input,     r.error)) return r; }
            else if (a == "--prompt")    { if (!eat_value(argc, argv, i, "--prompt",    r.detect.prompt,    r.error)) return r; }
            else if (a == "--output")    { if (!eat_value(argc, argv, i, "--output",    r.detect.output,    r.error)) return r; }
            else if (a == "--annotated") { if (!eat_value(argc, argv, i, "--annotated", r.detect.annotated, r.error)) return r; }
            else if (a == "--mode")      { if (!eat_value(argc, argv, i, "--mode",      r.detect.mode,      r.error)) return r; }
            else if (a == "--threads") {
                std::string v; if (!eat_value(argc, argv, i, "--threads", v, r.error)) return r;
                if (!parse_int(v, r.detect.threads, r.error, "--threads")) return r;
            }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.detect.model.empty())  { r.error = "detect: --model is required";  return r; }
        if (r.detect.input.empty())  { r.error = "detect: --input is required";  return r; }
        if (r.detect.prompt.empty()) { r.error = "detect: --prompt is required"; return r; }
        if (r.detect.mode != "hybrid" && r.detect.mode != "slow" && r.detect.mode != "fast") {
            r.error = "detect: --mode must be 'hybrid', 'slow', or 'fast' (got: " + r.detect.mode + ")";
            return r;
        }
        return r;
    }

    if (first == "info") {
        r.sub = Sub::Info;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model") { if (!eat_value(argc, argv, i, "--model", r.info_model, r.error)) return r; }
            else { r.error = "unknown flag: " + a; return r; }
        }
        if (r.info_model.empty()) { r.error = "info: --model is required"; return r; }
        return r;
    }

    if (first == "quantize") {
        r.sub = Sub::Quantize;
        std::vector<std::string> pos;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--help" || a == "-h") { r.sub = Sub::Help; return r; }
            if (!a.empty() && a[0] == '-') {
                r.error = "quantize takes positional args only: <input.gguf> <output.gguf> <type>";
                return r;
            }
            pos.push_back(a);
        }
        if (pos.size() != 3) {
            r.error = "quantize: expected exactly 3 positional args: <input.gguf> <output.gguf> <type>";
            return r;
        }
        r.quantize.in   = pos[0];
        r.quantize.out  = pos[1];
        r.quantize.type = pos[2];
        return r;
    }

    r.sub = Sub::None;
    r.error = "unknown subcommand: " + first;
    return r;
}

}  // namespace la::cli
