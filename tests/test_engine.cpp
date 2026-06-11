#include "engine.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cmath>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF"); const char* slow=std::getenv("LA_TEST_SLOW");
    if(!gguf||!slow){std::fprintf(stderr,"skip\n");return 77;}
    auto eng = la::Engine::load(gguf); if(!eng) return 1;
    // 448 lossless PNG fixture + its prompt
    std::string img = "tests/fixtures/parity_image.png";
    std::string query = "Locate all the instances that matches the following description: cat</c>remote.";
    auto boxes = eng->locate(img, query, la::Engine::Mode::Slow);   // slow = the 31-token reference
    std::vector<float> rb; std::vector<int64_t> rs; la_parity::load_baseline(slow,"slow_boxes",rb,rs); // [4,4]
    int nb = (int)rb.size()/4;
    int ok = ((int)boxes.size()==nb);
    std::printf("engine: %zu boxes (ref %d)\n", boxes.size(), nb);
    for(int k=0;k<nb && ok;++k){
        const float* r=&rb[k*4];
        ok &= (std::fabs(boxes[k].x1-r[0])<2.f && std::fabs(boxes[k].y1-r[1])<2.f &&
               std::fabs(boxes[k].x2-r[2])<2.f && std::fabs(boxes[k].y2-r[3])<2.f);
        std::printf("  box[%d] %s [%.1f %.1f %.1f %.1f] ref [%.1f %.1f %.1f %.1f]\n",
                    k, boxes[k].label.c_str(), boxes[k].x1,boxes[k].y1,boxes[k].x2,boxes[k].y2, r[0],r[1],r[2],r[3]);
    }

    // Variable-grid (non-448) box gate on bus.jpg. The reference boxes + labels live
    // in reference_preproc.gguf (slow_boxes / la_preproc.box_labels). JPEG-decode noise
    // could shift coords slightly, so allow a few-px tolerance (review confirmed exact
    // match on this image). Gates count + coords(tol) + labels.
    int bok = 1;
    const char* pp = std::getenv("LA_TEST_PREPROC");
    const char* bus = "/home/mudler/_git/rt-detr.cpp/benchmarks/images/bus.jpg";
    FILE* bf = std::fopen(bus, "rb");
    if(bf) std::fclose(bf);
    if(pp && bf){
        std::string bpath; la_parity::load_kv_str(pp, "la_preproc.image_path", bpath);
        std::vector<float> brb; std::vector<int64_t> brs;
        std::vector<std::string> blabels;
        bool have_ref = la_parity::load_baseline(pp, "slow_boxes", brb, brs) &&
                        la_parity::load_kv_str_array(pp, "la_preproc.box_labels", blabels);
        auto bb = eng->locate(bpath.empty()?std::string(bus):bpath,
            "Locate all the instances that matches the following description: person</c>bus.",
            la::Engine::Mode::Slow);
        if(have_ref){
            int bnb = (int)brb.size()/4;
            bok = ((int)bb.size()==bnb && (int)blabels.size()==bnb);
            std::printf("bus.jpg: %zu boxes (ref %d)\n", bb.size(), bnb);
            const float TOL = 5.f;
            for(int k=0;k<bnb && bok;++k){
                const float* r=&brb[k*4];
                bok &= (std::fabs(bb[k].x1-r[0])<TOL && std::fabs(bb[k].y1-r[1])<TOL &&
                        std::fabs(bb[k].x2-r[2])<TOL && std::fabs(bb[k].y2-r[3])<TOL);
                bok &= (bb[k].label==blabels[k]);
                std::printf("  box[%d] %s [%.1f %.1f %.1f %.1f] ref %s [%.1f %.1f %.1f %.1f]\n",
                            k, bb[k].label.c_str(), bb[k].x1,bb[k].y1,bb[k].x2,bb[k].y2,
                            blabels[k].c_str(), r[0],r[1],r[2],r[3]);
            }
            std::printf("bus.jpg box gate match=%d\n", bok);
        } else {
            std::printf("bus.jpg: no reference boxes in LA_TEST_PREPROC; smoke only (%zu boxes)\n", bb.size());
        }
    } else {
        std::printf("bus.jpg: skip (LA_TEST_PREPROC unset or image missing)\n");
    }
    return (ok && bok)?0:1;
}
