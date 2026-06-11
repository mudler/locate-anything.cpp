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
