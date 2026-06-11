#include "boxes.hpp"
namespace la {
static constexpr int COORD_START=151677, COORD_END=152677,
                     BOX_START=151668, BOX_END=151669, REF_START=151672, REF_END=151673;
std::vector<Box> parse_boxes(const std::vector<int32_t>& ids, int img_w, int img_h,
                             const std::function<std::string(const std::vector<int32_t>&)>& decode_label){
    std::vector<Box> out;
    std::string cur_label;
    int i=0, n=(int)ids.size();
    while(i<n){
        int t=ids[i];
        if(t==REF_START){
            int j=i+1; std::vector<int32_t> label_ids;
            while(j<n && ids[j]!=REF_END){
                label_ids.push_back(ids[j]);
                ++j;
            }
            cur_label = decode_label(label_ids);   // byte-decode the vocab pieces
            i=j+1; continue;   // label persists until next REF_START
        }
        if(t==BOX_START){
            int j=i+1; std::vector<int> coords;
            while(j<n && ids[j]!=BOX_END){
                if(ids[j]>=COORD_START && ids[j]<=COORD_END) coords.push_back(ids[j]-COORD_START);
                ++j;
            }
            if(coords.size()==4){
                Box b;
                b.x1 = coords[0]/1000.f*img_w; b.y1 = coords[1]/1000.f*img_h;
                b.x2 = coords[2]/1000.f*img_w; b.y2 = coords[3]/1000.f*img_h;
                b.label = cur_label;
                out.push_back(b);
            }
            i=j+1; continue;
        }
        ++i;
    }
    return out;
}
}
