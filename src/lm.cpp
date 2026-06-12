#include "lm.hpp"
#include "qwen2.hpp"
#include "mtp.hpp"
#include "ggml_extend.hpp"
#include <cstring>
#include <cmath>
namespace la {
// Build Qwen2 LM hyper-parameters from the loaded model config (single source of
// truth — replaces the per-method init blocks).
static Qwen2Hparams hparams_from_config(const VitConfig& c){
    Qwen2Hparams hp;
    hp.hidden=c.lm_hidden; hp.n_heads=c.lm_n_heads; hp.n_kv_heads=c.lm_n_kv_heads;
    hp.head_dim=c.lm_head_dim; hp.intermediate=c.lm_intermediate;
    hp.rope_theta=c.lm_rope_theta; hp.rms_eps=c.lm_rms_eps;
    return hp;
}
void LMForward::embed_tokens_host(const std::vector<int32_t>& ids, std::vector<float>& out){
    const int H = (int)ml_.config().lm_hidden;          // 2048
    out.assign((size_t)ids.size()*H, 0.f);              // [token, hidden] flat
    ggml_tensor* te = ml_.tensor("lm.tok_embd.weight"); // ne=[2048,152681], CPU buffer
    if (!te) return;
    const float* tew = (const float*)te->data;
    for (size_t t=0;t<ids.size();++t)
        std::memcpy(&out[t*H], &tew[(size_t)ids[t]*H], (size_t)H*sizeof(float));
}
bool LMForward::embed_and_splice(const std::vector<int32_t>& ids,
                                 const std::vector<float>& projected_host,
                                 std::vector<float>& out){
    const int seq = (int)ids.size();                   // 297
    const int H   = (int)ml_.config().lm_hidden;       // 2048
    const int img = (int)ml_.config().image_token_id;  // 151665
    if (!ml_.tensor("lm.tok_embd.weight")) return false;
    // 1) embed via host row-gather from tok_embd (ne=[2048,152681], CPU buffer)
    embed_tokens_host(ids, out);                        // [token, hidden] flat
    // 2) overwrite the image positions (id==151665) in order with projected vision rows
    int vi = 0;
    for (int t=0;t<seq;++t)
        if (ids[t]==img)
            std::memcpy(&out[(size_t)t*H], &projected_host[(size_t)(vi++)*H], (size_t)H*sizeof(float));
    // The number of <IMG_CONTEXT> slots must match the projected vision-token count N
    // (= (gh/2)*(gw/2)). For the 448 fixture N==256; generalized for variable grids.
    const int N = (int)(projected_host.size()/(size_t)H);
    return vi == N;
}
bool LMForward::run_layer0(const std::vector<float>& embeds_host, int seq, std::vector<float>& out){
    const auto& c = ml_.config(); const int H=(int)c.lm_hidden;
    Qwen2Hparams hp = hparams_from_config(c);
    std::vector<int32_t> pos(seq); for(int i=0;i<seq;++i) pos[i]=i;
    std::vector<float> mask((size_t)seq*seq);
    for(int i=0;i<seq;++i) for(int j=0;j<seq;++j) mask[(size_t)i*seq+j] = (j>i)? -INFINITY : 0.0f;
    GraphInputPool pool;
    return be_.compute([&](ggml_context* ctx)->ggml_tensor*{
        const int64_t ene[2]={H,seq};
        ggml_tensor* x = be_.add_graph_input_nd(ctx,pool,embeds_host.data(),ene,2);
        const int64_t pne[1]={seq};
        ggml_tensor* p = be_.add_int32_input_nd(ctx,pool,pos.data(),pne,1);
        const int64_t mne[2]={seq,seq};
        ggml_tensor* m = be_.add_graph_input_nd(ctx,pool,mask.data(),mne,2);
        return la::qwen2_layer_forward(ctx, x, p, m, la::load_qwen2_layer(ml_,0), hp);
    }, out);
}
bool LMForward::forward(const std::vector<int32_t>& ids,
                        const std::vector<float>& projected_host,
                        std::vector<float>& logits_last,
                        const std::vector<int>& capture_layers,
                        std::vector<std::vector<float>>& captured){
    const auto& c = ml_.config();
    const int seq=(int)ids.size(), H=(int)c.lm_hidden;
    // host embed + splice -> spliced [H*seq]
    std::vector<float> spliced;
    if(!embed_and_splice(ids, projected_host, spliced)) return false;
    Qwen2Hparams hp = hparams_from_config(c);
    std::vector<int32_t> pos(seq); for(int i=0;i<seq;++i) pos[i]=i;
    std::vector<float> mask((size_t)seq*seq);
    for(int i=0;i<seq;++i) for(int j=0;j<seq;++j) mask[(size_t)i*seq+j]=(j>i)?-INFINITY:0.0f;
    GraphInputPool pool;
    captured.assign(capture_layers.size(), {});
    return be_.forward_capture([&](ggml_context* ctx)->ggml_tensor*{
        const int64_t ene[2]={H,seq};
        ggml_tensor* x = be_.add_graph_input_nd(ctx,pool,spliced.data(),ene,2);   // [H,seq]
        const int64_t pne[1]={seq};
        ggml_tensor* p = be_.add_int32_input_nd(ctx,pool,pos.data(),pne,1);
        const int64_t mne[2]={seq,seq};
        ggml_tensor* m = be_.add_graph_input_nd(ctx,pool,mask.data(),mne,2);
        for(int il=0; il<(int)c.lm_n_layers; ++il){
            x = la::qwen2_layer_forward(ctx, x, p, m, la::load_qwen2_layer(ml_,il), hp);
            for(size_t cc=0;cc<capture_layers.size();++cc)
                if(capture_layers[cc]==il) be_.capture(x, &captured[cc]);
        }
        x = la::rms_norm(ctx, x, ml_.tensor("lm.output_norm.weight"), hp.rms_eps);
        // slice the LAST token [H,1] then project to logits [vocab,1]
        ggml_tensor* last = ggml_view_2d(ctx, x, H, 1, x->nb[1], (size_t)(seq-1)*x->nb[1]);
        last = ggml_cont(ctx, last);
        ggml_tensor* outw = ml_.tensor("lm.output.weight");
        if(!outw) outw = ml_.tensor("lm.tok_embd.weight");   // tied fallback
        return ggml_mul_mat(ctx, outw, last);                                    // [vocab,1]
    }, logits_last);
}

bool LMForward::decode_greedy_reprefill(const std::vector<int32_t>& prompt_ids,
                                        const std::vector<float>& projected_host,
                                        int max_new, std::vector<int32_t>& out_ids){
    const int EOS = 151645;
    std::vector<int32_t> seq = prompt_ids;       // grows as we decode
    out_ids.clear();
    for (int step=0; step<max_new; ++step){
        std::vector<float> logits; std::vector<std::vector<float>> caps;
        if(!forward(seq, projected_host, logits, {}, caps)) return false;  // logits at last pos
        int best=0; for(int i=1;i<(int)logits.size();++i) if(logits[i]>logits[best]) best=i;
        out_ids.push_back(best);
        if(best==EOS) break;
        seq.push_back(best);
    }
    return true;
}

// Generic resident chunk: x_host [H*n_new] written at KV offset pos0, plain absolute-
// position causal mask, all layers resident, returns LAST-position logits (final norm +
// lm_head). Advances kv.past_len to pos0+n_new on success.
bool LMForward::run_resident_causal(const std::vector<float>& x_host, int n_new, int pos0,
                                    ResidentKV& kv, std::vector<float>& logits){
    const auto& c = ml_.config();
    const int H = (int)c.lm_hidden;
    const int n_layers = (int)c.lm_n_layers;
    Qwen2Hparams hp = hparams_from_config(c);
    ggml_tensor* output_norm = ml_.tensor("lm.output_norm.weight");
    ggml_tensor* outw = ml_.tensor("lm.output.weight");
    if(!outw) outw = ml_.tensor("lm.tok_embd.weight");   // tied fallback
    if(!output_norm || !outw) return false;

    const int full = pos0 + n_new;
    std::vector<int32_t> posv(n_new);
    for(int i=0;i<n_new;++i) posv[i] = pos0 + i;
    std::vector<float> mask((size_t)full*n_new);
    for(int qi=0; qi<n_new; ++qi){
        const int q_abs = pos0 + qi;
        for(int key=0; key<full; ++key)
            mask[(size_t)qi*full + key] = (key > q_abs) ? -INFINITY : 0.0f;
    }
    kv.past_len = pos0;   // write offset for this chunk
    GraphInputPool pool;
    bool okc = be_.compute([&](ggml_context* ctx)->ggml_tensor*{
        const int64_t ene[2]={H,n_new};
        ggml_tensor* xt = be_.add_graph_input_nd(ctx,pool,x_host.data(),ene,2);
        const int64_t pne[1]={n_new};
        ggml_tensor* pt = be_.add_int32_input_nd(ctx,pool,posv.data(),pne,1);
        const int64_t mne[2]={full,n_new};
        ggml_tensor* mt = be_.add_graph_input_nd(ctx,pool,mask.data(),mne,2);
        ggml_tensor* hcur = xt;
        for(int il=0; il<n_layers; ++il){
            auto out = la::qwen2_layer_forward_resident(ctx, hcur, pt, mt, kv, il,
                                                        la::load_qwen2_layer(ml_,il), hp);
            hcur = out.y;
            // Interleaved per-layer write expansion (k_il then v_il, before the next
            // layer / output) — load-bearing: the cpy writes must be in the graph
            // before later layers' attention reads back the full K/V.
            be_.add_graph_root(out.k_write);
            be_.add_graph_root(out.v_write);
        }
        hcur = la::rms_norm(ctx, hcur, output_norm, hp.rms_eps);
        ggml_tensor* last = ggml_view_2d(ctx, hcur, H, 1, hcur->nb[1],
                                         (size_t)(n_new-1)*hcur->nb[1]);
        last = ggml_cont(ctx, last);
        return ggml_mul_mat(ctx, outw, last);   // [vocab,1]
    }, logits);
    if(okc) kv.past_len = pos0 + n_new;   // commit
    return okc;
}

bool LMForward::prefill_resident(const std::vector<int32_t>& prompt_ids,
                                 const std::vector<float>& projected_host,
                                 ResidentKV& kv, std::vector<float>* logits_last){
    std::vector<float> spliced;
    if(!embed_and_splice(prompt_ids, projected_host, spliced)) return false;
    std::vector<float> tmp;
    if(!run_resident_causal(spliced, (int)prompt_ids.size(), 0, kv, tmp)) return false;
    if(logits_last) *logits_last = std::move(tmp);
    return true;   // kv.past_len == prompt_len
}

void LMForward::build_mtp_positions(int cached_len, int n_recompute, std::vector<int32_t>& pos_out){
    const int block = 6;
    const int n_new = n_recompute + block;
    const int base  = cached_len + n_recompute;
    pos_out.resize(n_new);
    // recompute (committed-but-uncached) tokens: consecutive absolute positions.
    for(int i=0;i<n_recompute;++i) pos_out[i] = cached_len + i;
    // the 6 block slots: [base-1, base, base+1, base+2, base+3, base+4] — slot0 ties to
    // the last committed token's OWN position (reference: "last 6 position_ids -= 1").
    for(int b=0;b<block;++b) pos_out[n_recompute + b] = base + b - 1;
}

void LMForward::build_mtp_mask(int cached_len, int n_recompute, std::vector<float>& mask_out){
    const int block = 6;
    const int n_new = n_recompute + block;
    const int full  = cached_len + n_new;
    const int base  = cached_len + n_recompute;
    mask_out.assign((size_t)full*n_new, 0.0f);
    // 1) Base = causal by ABSOLUTE position. For query row q with absolute position
    //    apos(q) (recompute rows: apos=cached_len+q; block row b: apos=base-1 if b==0 else
    //    base+b-1), m[q*full+key] = (key > apos(q)) ? -INF : 0.  (key = KV slot index.)
    for(int q=0; q<n_new; ++q){
        int apos;
        if(q < n_recompute) apos = cached_len + q;            // recompute rows: plain causal
        else                apos = base + (q - n_recompute) - 1;  // block row b
        for(int key=0; key<full; ++key)
            mask_out[(size_t)q*full + key] = (key > apos) ? -INFINITY : 0.0f;
    }
    // --- Block-diffusion overrides, ported from mask_sdpa_utils.py
    //     update_causal_mask_for_one_gen_window_2d(block_size=6, causal_attn=False,
    //     use_cache=True). The last 6 query rows == the 6 block slots; the last 6 keys
    //     == the block's own KV slots [full-6 .. full-1]; col full-7 == the last
    //     committed token (the one slot0 duplicates). Reference overrides:
    //         if not causal_attn:
    //             attn_mask_2d[-block_size:, -block_size:] = 0.0     # 6x6 bidirectional
    //         if use_cache:
    //             attn_mask_2d[-block_size:, -block_size-1] = -inf   # mask prev last token
    // 2) 6x6 window fully bidirectional among the 6 block slots.
    for(int q=n_new-block; q<n_new; ++q)
        for(int key=full-block; key<full; ++key)
            mask_out[(size_t)q*full + key] = 0.0f;
    // 3) column full-7 (= last committed token, slot0's duplicate) -> -inf for all 6 block rows.
    for(int q=n_new-block; q<n_new; ++q)
        mask_out[(size_t)q*full + (full-block-1)] = -INFINITY;
}

bool LMForward::mtp_block_forward(ResidentKV& kv, const std::vector<float>& x_host, int n_new,
                                  int n_recompute, std::vector<float>& logits6_out){
    const int block = 6;
    const auto& c = ml_.config();
    const int H = (int)c.lm_hidden;
    const int n_layers = (int)c.lm_n_layers;
    Qwen2Hparams hp = hparams_from_config(c);
    ggml_tensor* output_norm = ml_.tensor("lm.output_norm.weight");
    ggml_tensor* outw = ml_.tensor("lm.output.weight");
    if(!outw) outw = ml_.tensor("lm.tok_embd.weight");   // tied fallback
    if(!output_norm || !outw) return false;

    const int cached_len = kv.past_len;          // caller set this (= cached_len)
    const int full = cached_len + n_new;
    std::vector<int32_t> posv; build_mtp_positions(cached_len, n_recompute, posv);
    std::vector<float>   mask; build_mtp_mask(cached_len, n_recompute, mask);
    // resident layer writes new K/V at kv.past_len (= cached_len); we do NOT advance it.
    GraphInputPool pool;
    return be_.compute([&](ggml_context* ctx)->ggml_tensor*{
        const int64_t ene[2]={H,n_new};
        ggml_tensor* xt = be_.add_graph_input_nd(ctx,pool,x_host.data(),ene,2);
        const int64_t pne[1]={n_new};
        ggml_tensor* pt = be_.add_int32_input_nd(ctx,pool,posv.data(),pne,1);
        const int64_t mne[2]={full,n_new};
        ggml_tensor* mt = be_.add_graph_input_nd(ctx,pool,mask.data(),mne,2);
        ggml_tensor* hcur = xt;
        for(int il=0; il<n_layers; ++il){
            auto out = la::qwen2_layer_forward_resident(ctx, hcur, pt, mt, kv, il,
                                                        la::load_qwen2_layer(ml_,il), hp);
            hcur = out.y;
            be_.add_graph_root(out.k_write);
            be_.add_graph_root(out.v_write);
        }
        hcur = la::rms_norm(ctx, hcur, output_norm, hp.rms_eps);
        // slice the LAST 6 columns ([H,6] view) -> lm_head -> [vocab,6] (position-major).
        ggml_tensor* last6 = ggml_view_2d(ctx, hcur, H, block, hcur->nb[1],
                                          (size_t)(n_new-block)*hcur->nb[1]);
        last6 = ggml_cont(ctx, last6);
        return ggml_mul_mat(ctx, outw, last6);   // [vocab,6]
    }, logits6_out);
}

bool LMForward::decode_greedy_resident(const std::vector<int32_t>& prompt_ids,
                                       const std::vector<float>& projected_host,
                                       int max_new, std::vector<int32_t>& out_ids){
    const int EOS = 151645;
    const auto& c = ml_.config();
    const int prompt_len = (int)prompt_ids.size();
    const int n_layers = (int)c.lm_n_layers;
    out_ids.clear();

    // Fresh resident cache per call so independent calls don't share state.
    ResidentKV kv;
    if(!kv.init(be_, n_layers, (int)c.lm_head_dim, (int)c.lm_n_kv_heads,
                prompt_len + max_new + 32))
        return false;

    auto argmax = [](const std::vector<float>& l)->int{
        int best=0; for(int i=1;i<(int)l.size();++i) if(l[i]>l[best]) best=i; return best;
    };

    // ---- prefill: run the whole prompt as one chunk at pos0=0 ----
    std::vector<float> logits;
    if(!prefill_resident(prompt_ids, projected_host, kv, &logits)){ kv.free(); return false; }
    int next = argmax(logits);
    out_ids.push_back(next);

    // ---- incremental decode: one token at a time ----
    for(int step=1; step<max_new && next!=EOS; ++step){
        std::vector<float> emb;
        embed_tokens_host(std::vector<int32_t>{next}, emb);
        if(!run_resident_causal(emb, 1, kv.past_len, kv, logits)){ kv.free(); return false; }
        next = argmax(logits);
        out_ids.push_back(next);
    }
    kv.free();
    return true;
}

bool LMForward::decode_hybrid(const std::vector<int32_t>& prompt_ids,
                              const std::vector<float>& projected_host,
                              int max_new, std::vector<int32_t>& out_ids,
                              std::vector<std::vector<float>>* captured_logits6,
                              bool fast, bool early_stop){
    const int TEXT_MASK = 151676;     // default_mask_token_id
    const int block = 6;
    const auto& c = ml_.config();
    const int prompt_len = (int)prompt_ids.size();
    const int n_layers = (int)c.lm_n_layers;
    out_ids.clear();

    auto argmax = [](const std::vector<float>& l)->int{
        int best=0; for(int i=1;i<(int)l.size();++i) if(l[i]>l[best]) best=i; return best;
    };

    // Fresh resident cache per call. Headroom for the 6-token block overshoot.
    ResidentKV kv;
    if(!kv.init(be_, n_layers, (int)c.lm_head_dim, (int)c.lm_n_kv_heads,
                prompt_len + max_new + 64))
        return false;

    // ---- prefill: whole prompt at pos0=0 -> kv.past_len = prompt_len ----
    if(!prefill_resident(prompt_ids, projected_host, kv)){ kv.free(); return false; }

    std::vector<int32_t> generated = prompt_ids;   // committed stream
    int cached_len = prompt_len;                   // KV holds [0, cached_len)
    mtp::HybridState st;                           // use_mtp=true, terminated=false

    // ---- generate loop (modeling_locateanything.py generate L464-513) ----
    while((int)generated.size() - prompt_len < max_new && !st.terminated){
        const int n_recompute = (int)generated.size() - cached_len;  // uncached committed
        if(st.use_mtp){
            // block input: embeds of generated[cached_len:] ++ {generated.back(), mask x5}
            std::vector<int32_t> block_ids(generated.begin()+cached_len, generated.end());
            block_ids.push_back(generated.back());
            for(int k=0;k<block-1;++k) block_ids.push_back(TEXT_MASK);
            std::vector<float> x_host; embed_tokens_host(block_ids, x_host);
            const int n_new = n_recompute + block;
            kv.past_len = cached_len;   // resident writes new K/V at cached_len
            std::vector<float> logits6;
            if(!mtp_block_forward(kv, x_host, n_new, n_recompute, logits6)){ kv.free(); return false; }
            if(captured_logits6) captured_logits6->push_back(logits6);
            // reshape [vocab,6] flat (position-major) -> [6][vocab]
            const int v = (int)logits6.size()/block;
            std::vector<std::vector<float>> l6(block, std::vector<float>(v));
            for(int p=0;p<block;++p) for(int t=0;t<v;++t) l6[p][t]=logits6[(size_t)p*v+t];
            // Principled early-stop: greedy hybrid keeps fabricating boxes past the real
            // detections (decode_bbox_avg never checks position 0). But the model's own
            // greedy choice for the next box-frame start IS im_end / null once it's done
            // — e.g. on the fixture p(im_end)=0.9996 right after the 4th box. That's the
            // exact signal the official *sampling* config stops on (it draws im_end there).
            // So stop when argmax(block position 0) is im_end/null. (early_stop=false for
            // the parity gates, which still reproduce the full reference stream.)
            const int IM_END = 151645, NULL_TOK = 152678;
            if(early_stop){
                int a0 = argmax(l6[0]);
                if(a0 == IM_END || a0 == NULL_TOK){ st.terminated = true; break; }
            }
            std::vector<int32_t> new_tokens = mtp::select_new_tokens(l6, 4, fast);
            mtp::StepOut step = mtp::hybrid_mtp_step(st, new_tokens, fast);
            for(int t: step.committed) generated.push_back(t);
            cached_len += n_recompute;  // recomputed tokens cached; 6 block slots discarded next round
        } else {
            // AR: forward the uncached committed chunk causally, take last-pos logits.
            std::vector<int32_t> chunk_ids(generated.begin()+cached_len, generated.end());
            std::vector<float> x_host; embed_tokens_host(chunk_ids, x_host);
            std::vector<float> logits;
            // run_resident_causal sets kv.past_len=pos0 then advances to pos0+n on success.
            if(!run_resident_causal(x_host, (int)chunk_ids.size(), cached_len, kv, logits)){ kv.free(); return false; }
            const int ar_token = argmax(logits);
            mtp::StepOut step = mtp::hybrid_ar_step(st, ar_token);
            for(int t: step.committed) generated.push_back(t);
            cached_len += n_recompute;  // == chunk we just cached; keeps kv.past_len in sync
        }
        if(st.terminated) break;
        // Robust loop-stop, complementing the im_end argmax check above. On some
        // images the model never emits im_end but instead LOOPS, re-emitting one
        // identical box forever (e.g. the bus scene: [802,376,812,410] x39). The
        // im_end signal stays low there. Detect the loop directly: if the last
        // completed box has all four coordinate tokens identical to the previous
        // box, drop the duplicate and stop. Two distinct objects never share a
        // pixel-identical box, so we keep the first occurrence and never drop a
        // real detection. (early_stop=false, the parity path, keeps the full stream.)
        if(early_stop){
            const int BOX_START=151668, BOX_END=151669, COORD_START=151677, COORD_END=152677;
            int pc[4]={-2,-2,-2,-2}, lc[4]={-1,-1,-1,-1}, pn=0, ln=0, lstart=-1;
            for(int i=prompt_len;i<(int)generated.size();){
                if(generated[i]==BOX_START){
                    int j=i+1, cc[4]={0,0,0,0}, n=0;
                    while(j<(int)generated.size() && generated[j]!=BOX_END){
                        if(generated[j]>=COORD_START && generated[j]<=COORD_END && n<4) cc[n++]=generated[j];
                        ++j;
                    }
                    if(n==4){ for(int k=0;k<4;++k){ pc[k]=lc[k]; lc[k]=cc[k]; } pn=ln; ln=4; lstart=i; }
                    i=j+1;
                } else ++i;
            }
            if(pn==4 && ln==4){
                // The degenerate tail loops in a few shapes: an exact repeat (bus:
                // [802,376,812,410] x39), or a "march" along one axis - one pair of
                // edges held fixed while the box slides and shrinks into slivers
                // (kitchen: y1=175,y2=210 fixed, x marching, width 38->5). Detect a
                // repeat, or a both-x / both-y edge match where the new box is a
                // sliver. The sliver guard (a real aligned object is not tiny) keeps
                // this from dropping genuine objects that merely share a shelf line.
                bool dup   = (lc[0]==pc[0] && lc[1]==pc[1] && lc[2]==pc[2] && lc[3]==pc[3]);
                bool xedge = (lc[0]==pc[0] && lc[2]==pc[2]);          // both left+right edges
                bool yedge = (lc[1]==pc[1] && lc[3]==pc[3]);          // both top+bottom edges
                bool sliver = (lc[2]-lc[0] < 20) || (lc[3]-lc[1] < 20);  // tiny in 0..1000 space
                if(dup || ((xedge || yedge) && sliver)){
                    generated.resize(lstart); st.terminated=true; break;
                }
            }
        }
    }
    out_ids.assign(generated.begin()+prompt_len, generated.end());
    kv.free();
    return true;
}
}
