#include "la_capi.h"
#include "engine.hpp"
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#define LA_CAPI_ABI_VERSION 1
struct la_ctx { std::unique_ptr<la::Engine> engine; std::string last_error; std::vector<la::Box> last; };
static char* dup_to_c(const std::string& s){ char* p=(char*)std::malloc(s.size()+1); if(!p) return nullptr;
    std::memcpy(p,s.data(),s.size()); p[s.size()]='\0'; return p; }
static void json_escape(std::string& o, const std::string& s){
    for(char c: s){ if(c=='"'||c=='\\'){o+='\\'; o+=c;} else if(c=='\n')o+="\\n"; else if((unsigned char)c<0x20){char b[8];std::snprintf(b,sizeof b,"\\u%04x",c);o+=b;} else o+=c; } }
static std::string boxes_json(const std::vector<la::Box>& b){
    std::string o="{\"detections\":["; char buf[160];
    for(size_t i=0;i<b.size();++i){ if(i)o+=','; o+="{\"label\":\""; json_escape(o,b[i].label);
        std::snprintf(buf,sizeof buf,"\",\"box\":[%.3f,%.3f,%.3f,%.3f]}",b[i].x1,b[i].y1,b[i].x2,b[i].y2); o+=buf; }
    o+="]}"; return o; }
extern "C" {
int la_capi_abi_version(void){ return LA_CAPI_ABI_VERSION; }
la_ctx* la_capi_load(const char* path, int n_threads){
    try{ if(!path) return nullptr; auto e=la::Engine::load(path,n_threads); if(!e) return nullptr;
         auto* c=new la_ctx(); c->engine=std::move(e); return c; } catch(...){ return nullptr; } }
void la_capi_free(la_ctx* c){ delete c; }
static char* run(la_ctx* c, const std::function<std::vector<la::Box>()>& f){
    if(!c) return nullptr;
    try{ c->last=f(); c->last_error.clear(); return dup_to_c(boxes_json(c->last)); }
    catch(const std::exception& e){ c->last_error=e.what(); return nullptr; }
    catch(...){ c->last_error="unknown error"; return nullptr; } }
static la::Engine::Mode mode_of(int m){
    return m==1? la::Engine::Mode::Slow : m==2? la::Engine::Mode::Fast : la::Engine::Mode::Hybrid;
}
char* la_capi_locate_path(la_ctx* c, const char* img, const char* prompt, int mode){
    return run(c,[&]{ return c->engine->locate(img?img:"", prompt?prompt:"", mode_of(mode)); }); }
char* la_capi_locate_buffer(la_ctx* c, const unsigned char* bytes, size_t len, const char* prompt, int mode){
    return run(c,[&]{ return c->engine->locate_buffer(bytes,len, prompt?prompt:"", mode_of(mode)); }); }
int la_capi_get_n_detections(la_ctx* c){ return c? (int)c->last.size() : 0; }
int la_capi_get_detection_box(la_ctx* c, int i, float o[4]){
    if(!c||i<0||i>=(int)c->last.size()||!o) return -1;
    o[0]=c->last[i].x1;o[1]=c->last[i].y1;o[2]=c->last[i].x2;o[3]=c->last[i].y2; return 0; }
int la_capi_get_detection_label(la_ctx* c, int i, char* buf, int bufsz){
    if(!c||i<0||i>=(int)c->last.size()) return -1;
    const std::string& s=c->last[i].label; int need=(int)s.size()+1;
    if(buf&&bufsz>0){ int n=std::min(bufsz-1,(int)s.size()); std::memcpy(buf,s.data(),n); buf[n]='\0'; }
    return need; }
void la_capi_free_string(char* s){ std::free(s); }
const char* la_capi_last_error(la_ctx* c){ return c? c->last_error.c_str() : ""; }
}
