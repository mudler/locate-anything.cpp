#include "cli.hpp"
#include "engine.hpp"
#include "quantize.hpp"
#include "visualize.hpp"
#include "image_io.hpp"
#include <cstdio>
#include <fstream>
#include <string>
static std::string boxes_json(const std::vector<la::Box>& b){
    std::string j="{\"detections\":["; char buf[256];
    for(size_t i=0;i<b.size();++i){ std::snprintf(buf,sizeof buf,
        "%s{\"label\":\"%s\",\"box\":[%.2f,%.2f,%.2f,%.2f]}", i?",":"",
        b[i].label.c_str(), b[i].x1,b[i].y1,b[i].x2,b[i].y2); j+=buf; }
    j+="]}"; return j; }
static int cmd_detect(const la::cli::DetectArgs& a){
    auto eng=la::Engine::load(a.model, a.threads);
    if(!eng){ std::fprintf(stderr,"error: failed to load model %s\n", a.model.c_str()); return 1; }
    auto mode = (a.mode=="slow")? la::Engine::Mode::Slow
              : (a.mode=="fast")? la::Engine::Mode::Fast
              : la::Engine::Mode::Hybrid;
    auto boxes = eng->locate(a.input, a.prompt, mode);
    std::string json = boxes_json(boxes);
    if(!a.output.empty()){ std::ofstream o(a.output); o<<json; } else { std::printf("%s\n", json.c_str()); }
    if(!a.annotated.empty()){
        la::Image img;
        if(la::load_image_rgb(a.input,img)){ la::Image ann=la::render_boxes(img,boxes);
            if(la::save_image_png(a.annotated,ann)) std::fprintf(stderr,"wrote %s\n",a.annotated.c_str());
            else std::fprintf(stderr,"error: failed to write %s\n",a.annotated.c_str()); }
        else std::fprintf(stderr,"error: failed to reload %s for annotation\n",a.input.c_str());
    }
    std::fprintf(stderr,"%zu detections\n", boxes.size());
    return 0;
}
static int cmd_quantize(const la::cli::QuantizeArgs& a){
    if(la::quantize_gguf(a.in, a.out, a.type)){ std::fprintf(stderr,"wrote %s (%s)\n", a.out.c_str(), a.type.c_str()); return 0; }
    std::fprintf(stderr,"quantize failed\n"); return 1;
}
static int cmd_info(const std::string& model){
    auto eng=la::Engine::load(model,1); if(!eng){ std::fprintf(stderr,"error: load failed\n"); return 1; }
    std::printf("model: %s\nstatus: loaded ok\n", model.c_str()); return 0;
}
int main(int argc, char** argv){
    auto p=la::cli::parse(argc, argv);
    if(!p.error.empty()){ std::fprintf(stderr,"error: %s\n", p.error.c_str()); la::cli::print_help(); return 1; }
    using S=la::cli::Sub;
    switch(p.sub){
        case S::Detect:   return cmd_detect(p.detect);
        case S::Info:     return cmd_info(p.info_model);
        case S::Quantize: return cmd_quantize(p.quantize);
        case S::Help:     la::cli::print_help(); return 0;
        default:          la::cli::print_help(); return 1;
    }
}
