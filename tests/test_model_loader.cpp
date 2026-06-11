#include "model_loader.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstring>
int main() {
    const char* gguf = std::getenv("LA_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "LA_TEST_GGUF unset; skip\n"); return 77; }
    la::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }
    const auto& c = ml.config();
    int ok = 1;
    ok &= (c.vit_hidden == 1152);
    ok &= (c.vit_n_layers == 27);
    ok &= (c.vit_n_heads == 16);
    ok &= (c.vit_head_dim == 72);
    ok &= (c.vit_intermediate == 4304);
    ok &= (c.vit_patch == 14);
    ok &= (c.vit_pos_emb_hw == 64);
    // tensor presence + shape: wqkv is [1152 -> 3456], stored raw torch [out=3456, in=1152]
    ggml_tensor* wqkv = ml.tensor("vit.blk.0.wqkv.weight");
    ok &= (wqkv != nullptr);
    if (wqkv) { ok &= (wqkv->ne[0] == 1152 && wqkv->ne[1] == 3456); }  // ggml ne reversed
    ggml_tensor* pos = ml.tensor("vit.pos_emb.weight");
    ok &= (pos != nullptr);
    if (pos) { ok &= (pos->ne[0] == 1152 && pos->ne[1] == 64 && pos->ne[2] == 64); }
    ggml_tensor* pe = ml.tensor("vit.patch_embed.weight");
    ok &= (pe != nullptr);  // raw torch [1152,3,14,14] -> ggml ne [14,14,3,1152]
    if (pe) { ok &= (pe->ne[0]==14 && pe->ne[1]==14 && pe->ne[2]==3 && pe->ne[3]==1152); }
    ok &= (c.lm_hidden == 2048);
    ok &= (c.lm_n_layers == 36);
    ok &= (c.lm_n_heads == 16);
    ok &= (c.lm_n_kv_heads == 2);
    ok &= (c.lm_head_dim == 128);
    ok &= (c.lm_intermediate == 11008);
    ok &= (c.lm_vocab == 152681);
    ok &= (c.image_token_id == 151665);
    // LM tensor shapes (ggml reversed-ne): q_proj torch [2048,2048], k/v_proj [256,2048]
    ggml_tensor* q = ml.tensor("lm.blk.0.attn_q.weight");
    ok &= (q && q->ne[0]==2048 && q->ne[1]==2048);
    ggml_tensor* kb = ml.tensor("lm.blk.0.attn_k.bias");
    ok &= (kb && kb->ne[0]==256);
    ggml_tensor* emb = ml.tensor("lm.tok_embd.weight");
    ok &= (emb && emb->ne[0]==2048 && emb->ne[1]==152681);
    ggml_tensor* outw = ml.tensor("lm.output.weight");
    ok &= (outw && outw->ne[0]==2048 && outw->ne[1]==152681);
    ggml_tensor* p0 = ml.tensor("proj.0.weight");
    ok &= (p0 && p0->ne[0]==4608);
    ggml_tensor* p1 = ml.tensor("proj.1.weight");
    ok &= (p1 && p1->ne[0]==4608 && p1->ne[1]==2048);
    std::printf("loader ok=%d\n", ok);
    return ok ? 0 : 1;
}
