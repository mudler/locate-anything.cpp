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
    std::string q8p = q8?q8:"/tmp/la-q8.gguf", q4p=q4?q4:"/tmp/la-q4k.gguf";
    if(!q8){ ok &= la::quantize_gguf(f32, q8p, "q8_0"); }
    if(!q4){ ok &= la::quantize_gguf(f32, q4p, "q4_k"); }
    int r8=run_one(q8p.c_str(), slow, 3.0f, n); std::printf("q8_0: %d boxes ok=%d\n", n, r8); ok &= (r8==1);
    int r4=run_one(q4p.c_str(), slow, 20.0f, n); std::printf("q4_k: %d boxes ok=%d\n", n, r4); ok &= (r4==1);
    return ok?0:1;
}
