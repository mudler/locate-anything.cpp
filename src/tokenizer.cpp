#include "tokenizer.hpp"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <set>

#ifdef LA_TOK_DEBUG
#define TKDBG(...) do{ std::fprintf(stderr, "[tok] " __VA_ARGS__); std::fflush(stderr);}while(0)
#else
#define TKDBG(...) do{}while(0)
#endif

namespace la {

// ---------------- UTF-8 helpers ----------------
static void cpt_to_utf8(uint32_t cp, std::string& out) {
    if (cp <= 0x7F) {
        out.push_back((char)cp);
    } else if (cp <= 0x7FF) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// Decode utf8 -> codepoints. Invalid bytes pass through as their value.
static std::vector<uint32_t> utf8_to_cpts(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp; size_t len;
        if (c < 0x80)      { cp = c;          len = 1; }
        else if ((c>>5)==0x6){ cp = c & 0x1F; len = 2; }
        else if ((c>>4)==0xE){ cp = c & 0x0F; len = 3; }
        else if ((c>>3)==0x1E){ cp = c & 0x07; len = 4; }
        else                 { cp = c;        len = 1; }  // invalid lead
        if (i + len > n) { out.push_back(c); i++; continue; }
        bool ok = true;
        for (size_t k = 1; k < len; ++k) {
            unsigned char cc = (unsigned char)s[i+k];
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(c); i++; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

// ---------------- codepoint classification (pragmatic, ASCII-correct) ----------------
static bool is_ws(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004:
        case 0x2005: case 0x2006: case 0x2007: case 0x2008: case 0x2009:
        case 0x200A: case 0x2028: case 0x2029: case 0x202F: case 0x205F:
        case 0x3000:
            return true;
        default: return false;
    }
}
static bool is_num(uint32_t cp) { return cp >= '0' && cp <= '9'; }
// Treat ASCII letters, and any non-ASCII non-space/non-digit codepoint, as a
// letter. This matches \p{L} for the Latin/CJK text seen in LocateAnything
// prompts; pure symbol codepoints are rare and not exercised here.
static bool is_letter(uint32_t cp) {
    if (cp > 0x10FFFF) return false;  // invalid / out-of-range sentinel
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    if (cp >= 0x80 && !is_ws(cp) && !is_num(cp)) return true;
    return false;
}
static uint32_t to_lower(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
}

// ---------------- Qwen2 pre-tokenization regex split ----------------
// Faithful port of llama.cpp's unicode_regex_split_custom_qwen2 for the regex:
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Returns words as substrings (utf8) of the input.
static std::vector<std::string> qwen2_split(const std::string& text) {
    std::vector<std::string> words;
    std::vector<uint32_t> cpts = utf8_to_cpts(text);
    const size_t n = cpts.size();
    const uint32_t OOR = 0xFFFFFFFF;
    auto get = [&](size_t p) -> uint32_t { return p < n ? cpts[p] : OOR; };

    size_t prev = 0;
    auto add = [&](size_t end) {
        if (end > prev) {
            std::string w;
            for (size_t p = prev; p < end; ++p) cpt_to_utf8(cpts[p], w);
            words.push_back(std::move(w));
        }
        prev = end;
    };

    for (size_t pos = 0; pos < n; /* advanced inside */) {
        uint32_t cp = get(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp == '\'' && pos + 1 < n) {
            uint32_t c1 = to_lower(get(pos+1));
            if (c1=='s' || c1=='t' || c1=='m' || c1=='d') { add(pos+2); pos = prev; continue; }
            if (pos + 2 < n) {
                uint32_t c2 = to_lower(get(pos+2));
                if ((c1=='r'&&c2=='e')||(c1=='v'&&c2=='e')||(c1=='l'&&c2=='l')) { add(pos+3); pos = prev; continue; }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cp=='\r' || cp=='\n' || is_num(cp))) {
            if (is_letter(cp) || is_letter(get(pos+1))) {
                size_t p = pos + 1;
                while (is_letter(get(p))) p++;
                add(p); pos = prev; continue;
            }
        }

        // \p{N}  (single digit)
        if (is_num(cp)) { add(pos+1); pos = prev; continue; }

        // " ?[^\s\p{L}\p{N}]+[\r\n]*"
        {
            uint32_t fcp = (cp==' ') ? get(pos+1) : cp;
            bool fcp_ok = fcp != OOR && !(is_ws(fcp) || is_letter(fcp) || is_num(fcp));
            if (fcp_ok && cp != OOR) {
                size_t p = pos + (cp==' ' ? 1 : 0);
                while (true) {
                    uint32_t q = get(p);
                    if (q == OOR || is_ws(q) || is_letter(q) || is_num(q)) break;
                    p++;
                }
                uint32_t q = get(p);
                while (q=='\r' || q=='\n') { p++; q = get(p); }
                add(p); pos = prev; continue;
            }
        }

        // count whitespace run
        size_t nws = 0, last_rn = 0;
        while (is_ws(get(pos+nws))) {
            uint32_t q = get(pos+nws);
            if (q=='\r' || q=='\n') last_rn = pos + nws + 1;
            nws++;
        }
        // \s*[\r\n]+
        if (last_rn > 0) { add(last_rn); pos = prev; continue; }
        // \s+(?!\S)
        if (nws > 1 && get(pos+nws) != OOR) { add(pos + nws - 1); pos = prev; continue; }
        // \s+
        if (nws > 0) { add(pos + nws); pos = prev; continue; }
        // no match
        add(pos + 1); pos = prev;
    }
    return words;
}

// ---------------- load ----------------
bool Tokenizer::load(const ModelLoader& ml) {
    if (!ml.kv_str_array("locateanything.tokenizer.tokens", id_to_piece_)) return false;
    TKDBG("tokens=%zu\n", id_to_piece_.size());
    std::vector<std::string> merges;
    if (!ml.kv_str_array("locateanything.tokenizer.merges", merges)) return false;
    TKDBG("merges=%zu\n", merges.size());
    ml.kv_i32_array("locateanything.tokenizer.token_types", token_types_);
    TKDBG("token_types=%zu\n", token_types_.size());

    piece_to_id_.reserve(id_to_piece_.size()*2);
    for (size_t i = 0; i < id_to_piece_.size(); ++i)
        piece_to_id_[id_to_piece_[i]] = (int32_t)i;
    TKDBG("piece_to_id built\n");

    merge_rank_.reserve(merges.size()*2);
    for (size_t r = 0; r < merges.size(); ++r)
        merge_rank_.emplace(merges[r], (int)r);
    TKDBG("merge_rank built\n");

    // GPT-2 byte<->unicode maps.
    bool used[256] = {false};
    auto assign = [&](int b, uint32_t cp){
        used[b] = true;
        std::string u; cpt_to_utf8(cp, u);
        byte_to_str_[b] = u;
        cpt_to_byte_[cp] = (uint8_t)b;
    };
    for (int b = 0x21; b <= 0x7E; ++b) assign(b, (uint32_t)b);
    for (int b = 0xA1; b <= 0xAC; ++b) assign(b, (uint32_t)b);
    for (int b = 0xAE; b <= 0xFF; ++b) assign(b, (uint32_t)b);
    int nn = 0;
    for (int b = 0; b < 256; ++b)
        if (!used[b]) assign(b, (uint32_t)(256 + nn++));

    // Special (atomic) tokens: token_type == 4.
    std::set<size_t> lens;
    for (size_t i = 0; i < id_to_piece_.size(); ++i) {
        if (i < token_types_.size() && token_types_[i] == 4) {
            special_ids_.push_back((int32_t)i);
            const std::string& p = id_to_piece_[i];
            if (!p.empty()) { special_set_.insert(p); lens.insert(p.size()); }
        }
    }
    special_lens_.assign(lens.rbegin(), lens.rend());  // descending
    TKDBG("specials=%zu lens=%zu\n", special_ids_.size(), special_lens_.size());
    return !id_to_piece_.empty();
}

// ---------------- BPE on one byte-encoded word ----------------
void Tokenizer::bpe_word(const std::string& word, std::vector<int32_t>& out) const {
    // Split byte-encoded word into its mapped-unicode chars (each is one symbol).
    std::vector<std::string> syms;
    {
        std::vector<uint32_t> cps = utf8_to_cpts(word);
        syms.reserve(cps.size());
        for (uint32_t cp : cps) { std::string s; cpt_to_utf8(cp, s); syms.push_back(std::move(s)); }
    }
    if (syms.empty()) return;

    // Greedily merge the lowest-rank adjacent pair until none remain.
    while (syms.size() > 1) {
        int best_rank = INT_MAX; size_t best_i = 0; bool found = false;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = merge_rank_.find(syms[i] + " " + syms[i+1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second; best_i = i; found = true;
            }
        }
        if (!found) break;
        syms[best_i] += syms[best_i+1];
        syms.erase(syms.begin() + best_i + 1);
    }

    // Map symbols -> ids (fallback: split unknown symbol into single chars).
    for (const std::string& s : syms) {
        auto it = piece_to_id_.find(s);
        if (it != piece_to_id_.end()) { out.push_back(it->second); continue; }
        std::vector<uint32_t> cps = utf8_to_cpts(s);
        for (uint32_t cp : cps) {
            std::string one; cpt_to_utf8(cp, one);
            auto jt = piece_to_id_.find(one);
            if (jt != piece_to_id_.end()) out.push_back(jt->second);
        }
    }
}

// ---------------- encode ----------------
std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    TKDBG("encode start len=%zu\n", text.size());
    std::vector<int32_t> ids;
    const size_t n = text.size();
    size_t i = 0, run_start = 0;

    auto flush_run = [&](size_t end) {
        if (end <= run_start) return;
        std::string run = text.substr(run_start, end - run_start);
        TKDBG("flush_run [%zu,%zu) len=%zu\n", run_start, end, run.size());
        for (const std::string& word : qwen2_split(run)) {
            // byte-encode the word, then BPE.
            std::string enc;
            for (unsigned char c : word) enc += byte_to_str_[c];
            bpe_word(enc, ids);
        }
    };

    while (i < n) {
        // Longest atomic special-token match at position i.
        bool matched = false;
        if (!special_lens_.empty()) {
            for (size_t L : special_lens_) {
                if (i + L > n) continue;
                std::string cand = text.substr(i, L);
                auto it = special_set_.find(cand);
                if (it != special_set_.end()) {
                    flush_run(i);
                    ids.push_back(piece_to_id_.at(cand));
                    i += L;
                    run_start = i;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) i++;
    }
    flush_run(n);
    return ids;
}

// ---------------- decode ----------------
std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // Concatenate pieces, then map each mapped-unicode char back to its byte.
    std::string mapped;
    for (int32_t id : ids) {
        if (id >= 0 && id < (int32_t)id_to_piece_.size()) mapped += id_to_piece_[id];
    }
    std::string out;
    for (uint32_t cp : utf8_to_cpts(mapped)) {
        auto it = cpt_to_byte_.find(cp);
        if (it != cpt_to_byte_.end()) out.push_back((char)it->second);
        else cpt_to_utf8(cp, out);  // shouldn't happen for valid pieces
    }
    return out;
}

int32_t Tokenizer::token_to_id(const std::string& piece) const {
    auto it = piece_to_id_.find(piece);
    return it == piece_to_id_.end() ? -1 : it->second;
}

}  // namespace la
