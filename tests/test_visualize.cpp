#include "engine.hpp"
#include "visualize.hpp"
#include "image_io.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF");
    if(!gguf){std::fprintf(stderr,"skip\n");return 77;}
    auto eng=la::Engine::load(gguf); if(!eng) return 1;
    la::Image img; if(!la::load_image_rgb("tests/fixtures/parity_image.png", img)) return 1;
    auto boxes = eng->locate("tests/fixtures/parity_image.png",
        "Locate all the instances that matches the following description: cat</c>remote.",
        la::Engine::Mode::Slow);
    if(boxes.size()!=4){ std::printf("expected 4 boxes got %zu\n", boxes.size()); return 1; }
    la::Image ann = la::render_boxes(img, boxes);
    int ok = (ann.w==img.w && ann.h==img.h && ann.rgb.size()==img.rgb.size());
    int diff=0; for(size_t i=0;i<img.rgb.size();++i) if(ann.rgb[i]!=img.rgb[i]) diff++;
    ok &= (diff > 100);                                  // boxes actually drawn
    ok &= la::save_image_png("/tmp/la_annotated.png", ann);
    la::Image back; ok &= la::load_image_rgb("/tmp/la_annotated.png", back);
    ok &= (back.w==img.w && back.h==img.h);
    // buffer-load round-trip: read the PNG bytes, load via buffer
    std::FILE* f=std::fopen("/tmp/la_annotated.png","rb"); ok &= (f!=nullptr);
    if(f){ std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET);
           std::vector<unsigned char> buf(n); ok &= (std::fread(buf.data(),1,n,f)==(size_t)n); std::fclose(f);
           la::Image bb; ok &= (la::load_image_rgb_buffer(buf.data(), buf.size(), bb) && bb.w==img.w); }
    std::printf("visualize ok=%d (diff_px=%d)\n", ok, diff);
    return ok?0:1;
}
