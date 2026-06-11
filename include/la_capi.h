#ifndef LA_CAPI_H
#define LA_CAPI_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct la_ctx la_ctx;
int      la_capi_abi_version(void);
la_ctx*  la_capi_load(const char* gguf_path, int n_threads);   /* NULL on failure */
void     la_capi_free(la_ctx* ctx);                            /* safe on NULL */
/* Detection; returns malloc'd JSON (free via la_capi_free_string), NULL on error.
   mode: 0=hybrid, 1=slow, 2=fast. */
char*    la_capi_locate_path  (la_ctx* ctx, const char* image_path, const char* prompt, int mode);
char*    la_capi_locate_buffer(la_ctx* ctx, const unsigned char* bytes, size_t len, const char* prompt, int mode);
/* Accessors over the LAST locate_* result (for purego consumers). */
int      la_capi_get_n_detections(la_ctx* ctx);
int      la_capi_get_detection_box(la_ctx* ctx, int i, float out_xyxy[4]);    /* 0 ok, -1 bad index */
int      la_capi_get_detection_label(la_ctx* ctx, int i, char* buf, int buf_size); /* returns required size incl NUL; two-call sizing */
void        la_capi_free_string(char* s);
const char* la_capi_last_error(la_ctx* ctx);                   /* owned by ctx, "" if none */
#ifdef __cplusplus
}
#endif
#endif
