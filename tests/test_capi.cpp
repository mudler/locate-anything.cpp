#include "la_capi.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
int main(){
    const char* gguf=std::getenv("LA_TEST_GGUF");
    if(!gguf){std::fprintf(stderr,"skip\n");return 77;}
    if(la_capi_abi_version()!=1) return 1;
    la_ctx* c=la_capi_load(gguf,0); if(!c){ std::fprintf(stderr,"load fail\n"); return 1; }
    char* json=la_capi_locate_path(c,"tests/fixtures/parity_image.png",
        "Locate all the instances that matches the following description: cat</c>remote.", 1);
    int ok=(json!=nullptr);
    if(json) ok &= (std::strstr(json,"cat")&&std::strstr(json,"remote")&&std::strstr(json,"\"box\""));
    int n=la_capi_get_n_detections(c); ok &= (n==4);
    float b[4]; ok &= (la_capi_get_detection_box(c,0,b)==0);
    int need=la_capi_get_detection_label(c,0,nullptr,0); ok &= (need>0);   // two-call sizing
    char lbl[64]; la_capi_get_detection_label(c,0,lbl,sizeof lbl);
    ok &= (la_capi_get_detection_box(c,99,b)==-1);                         // bad index
    std::printf("capi: n=%d label0=%s box0=[%.1f %.1f %.1f %.1f] err='%s' ok=%d\n",
                n,lbl,b[0],b[1],b[2],b[3], la_capi_last_error(c), ok);
    la_capi_free_string(json); la_capi_free(c);
    la_capi_free(nullptr);                                                 // NULL-safe
    ok &= (la_capi_load("/nonexistent.gguf",0)==nullptr);                  // error path
    return ok?0:1;
}
