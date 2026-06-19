// tokenizer.c — BPE tokenizers:
//   pre 0: gemma4 SPM-style (▁ escaping, raw UTF-8, <0xXX> byte fallback)
//   pre 1: gpt2/qwen35 byte-level BPE (byte↔unicode map + qwen35 \p{L}\p{M}\p{N} pre-regex)
// Mirrors llama.cpp llm_tokenizer_bpe + unicode_regex_split_custom_qwen35.
#include "g4run.h"
#include "qwen_ucat.h"   // unicode category ranges + whitespace set (from llama.cpp)

// qwen35 category flag bits (subset of llama.cpp unicode_cpt_flags)
#define QF_NUMBER 0x0002
#define QF_LETTER 0x0004
#define QF_MARK   0x0010
#define QF_WS     0x0100

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEF = 4, TT_UNUSED = 5, TT_BYTE = 6 };

typedef struct { int32_t id; } map_ent;

struct g4_tok {
    g4_model_file * mf;
    // token text -> id (open addressing)
    int32_t * tmap;     uint32_t tmask;
    // merge pair -> rank
    int32_t * mmap;     uint32_t mmask;
    uint32_t * msplit;  // split point (index of separator space) per merge
    // specials sorted by length desc
    int  * specials;    int n_specials;
    int    byte_ids[256];
    int    eog[8]; int n_eog;
    // pre 1 (gpt2/qwen35 byte-level BPE)
    int      pre;            // 0 = gemma4 SPM-style; 1 = gpt2/qwen35 byte-level
    char     b2u[256][5];    // byte -> mapped-codepoint UTF-8 (gpt2 byte encoder)
    uint8_t  b2u_len[256];
    int16_t  cpt2byte[0x400];// mapped codepoint -> byte (decode), -1 if none
};

static int cpt_utf8(uint32_t cp, char * o);   // fwd (defined in the qwen BPE section)

static uint64_t fnv1a(const char * s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
    return h;
}

static int tmap_find(const g4_tok * t, const char * s, size_t n) {
    uint64_t h = fnv1a(s, n);
    uint32_t i = (uint32_t)h & t->tmask;
    while (t->tmap[i] >= 0) {
        g4_str tx = t->mf->tok_text[t->tmap[i]];
        if (tx.len == n && memcmp(tx.ptr, s, n) == 0) return t->tmap[i];
        i = (i + 1) & t->tmask;
    }
    return -1;
}

static uint64_t merge_hash(const char * l, size_t ln, const char * r, size_t rn) {
    return fnv1a(l, ln) * 31 + fnv1a(r, rn);
}

static int mmap_find(const g4_tok * t, const char * l, size_t ln, const char * r, size_t rn) {
    uint64_t h = merge_hash(l, ln, r, rn);
    uint32_t i = (uint32_t)h & t->mmask;
    while (t->mmap[i] >= 0) {
        int rank = t->mmap[i];
        g4_str m = t->mf->merges[rank];
        uint32_t sp = t->msplit[rank];
        if (sp == ln && m.len == ln + 1 + rn &&
            memcmp(m.ptr, l, ln) == 0 && memcmp(m.ptr + sp + 1, r, rn) == 0)
            return rank;
        i = (i + 1) & t->mmask;
    }
    return -1;
}

static uint32_t pow2_atleast(uint32_t n) { uint32_t p = 1; while (p < n) p <<= 1; return p; }

g4_tok * g4_tok_init(g4_model_file * mf) {
    if (!mf->tok_text || !mf->tok_type || !mf->merges) {
        fprintf(stderr, "g4: tokenizer data missing from GGUF\n"); return NULL;
    }
    g4_tok * t = calloc(1, sizeof(*t));
    t->mf = mf;
    t->pre = mf->tok_pre;

    // gpt2 byte-level encoder: byte -> mapped-codepoint UTF-8, and the inverse
    if (t->pre == 1) {
        for (int i = 0; i < 0x400; i++) t->cpt2byte[i] = -1;
        int nn = 0;
        for (int b = 0; b < 256; b++) {
            uint32_t cp;
            if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) cp = (uint32_t)b;
            else cp = 256 + (nn++);
            int l = cpt_utf8(cp, t->b2u[b]); t->b2u[b][l] = 0; t->b2u_len[b] = (uint8_t)l;
            if (cp < 0x400) t->cpt2byte[cp] = (int16_t)b;
        }
    }

    // token map
    uint32_t tcap = pow2_atleast((uint32_t)mf->tok_n * 2);
    t->tmask = tcap - 1;
    t->tmap = malloc(tcap * 4);
    memset(t->tmap, 0xff, tcap * 4);
    for (int id = 0; id < mf->tok_n; id++) {
        g4_str s = mf->tok_text[id];
        uint64_t h = fnv1a(s.ptr, s.len);
        uint32_t i = (uint32_t)h & t->tmask;
        while (t->tmap[i] >= 0) {
            // keep first occurrence (duplicate texts are pathological)
            g4_str ex = mf->tok_text[t->tmap[i]];
            if (ex.len == s.len && memcmp(ex.ptr, s.ptr, s.len) == 0) goto next_tok;
            i = (i + 1) & t->tmask;
        }
        t->tmap[i] = id;
        next_tok:;
    }

    // merges map
    uint32_t mcap = pow2_atleast((uint32_t)mf->merges_n * 2);
    t->mmask = mcap - 1;
    t->mmap = malloc(mcap * 4);
    t->msplit = malloc((size_t)mf->merges_n * 4);
    memset(t->mmap, 0xff, mcap * 4);
    for (int r = 0; r < mf->merges_n; r++) {
        g4_str m = mf->merges[r];
        // split at the single ' ' separator (parts cannot contain raw space)
        int sp = -1;
        for (uint32_t j = 0; j < m.len; j++) if (m.ptr[j] == ' ') { sp = (int)j; break; }
        if (sp <= 0 || sp >= (int)m.len - 1) {
            // malformed merge; first/last char space — find from rear as fallback
            for (int j = (int)m.len - 2; j > 0; j--) if (m.ptr[j] == ' ') { sp = j; break; }
            if (sp <= 0) { t->msplit[r] = 0; continue; }
        }
        t->msplit[r] = (uint32_t)sp;
        uint64_t h = merge_hash(m.ptr, sp, m.ptr + sp + 1, m.len - sp - 1);
        uint32_t i = (uint32_t)h & t->mmask;
        while (t->mmap[i] >= 0) i = (i + 1) & t->mmask;
        t->mmap[i] = r;
    }

    // specials (CONTROL | USER_DEFINED | UNKNOWN), sorted by text length desc
    t->specials = malloc(1024 * 4);
    for (int id = 0; id < mf->tok_n && t->n_specials < 1024; id++) {
        int ty = mf->tok_type[id];
        if (ty == TT_CONTROL || ty == TT_USER_DEF || ty == TT_UNKNOWN)
            t->specials[t->n_specials++] = id;
    }
    // insertion sort by len desc (n_specials is small)
    for (int i = 1; i < t->n_specials; i++) {
        int v = t->specials[i];
        uint32_t vl = mf->tok_text[v].len;
        int j = i - 1;
        while (j >= 0 && mf->tok_text[t->specials[j]].len < vl) { t->specials[j+1] = t->specials[j]; j--; }
        t->specials[j+1] = v;
    }

    // byte token ids
    for (int b = 0; b < 256; b++) {
        char buf[8];
        static const char * hex = "0123456789ABCDEF";
        buf[0]='<'; buf[1]='0'; buf[2]='x'; buf[3]=hex[b>>4]; buf[4]=hex[b&15]; buf[5]='>';
        t->byte_ids[b] = tmap_find(t, buf, 6);
    }

    // EOG set: <eos>, <turn|>, <|tool_response> (per llama.cpp special_eog detection)
    if (mf->eos_id >= 0) t->eog[t->n_eog++] = mf->eos_id;
    if (mf->eot_id >= 0 && mf->eot_id != mf->eos_id) t->eog[t->n_eog++] = mf->eot_id;
    {
        const char * const * extra = t->pre == 1
            ? (const char *[]){ "<|im_end|>", "<|endoftext|>", NULL }
            : (const char *[]){ "<turn|>", "<|tool_response>", NULL };
        for (int k = 0; extra[k]; k++) {
            int id = tmap_find(t, extra[k], (int)strlen(extra[k]));
            if (id < 0) continue;
            bool have = false; for (int i = 0; i < t->n_eog; i++) have |= t->eog[i] == id;
            if (!have && t->n_eog < 8) t->eog[t->n_eog++] = id;
        }
    }
    return t;
}

void g4_tok_free(g4_tok * t) {
    if (!t) return;
    free(t->tmap); free(t->mmap); free(t->msplit); free(t->specials); free(t);
}

bool g4_tok_is_eog(g4_tok * t, int id) {
    for (int i = 0; i < t->n_eog; i++) if (t->eog[i] == id) return true;
    return false;
}

int g4_tok_find(g4_tok * t, const char * text) {
    return tmap_find(t, text, strlen(text));
}

// --------------------------------------------------------------- encode -----

typedef struct { const char * text; int n; int prev, next; } sym_t;
typedef struct { int left, right, rank, size; } bigram_t;

typedef struct {
    bigram_t * a; int n, cap;
} heap_t;

static bool bg_less(const bigram_t * x, const bigram_t * y) {
    // min-heap: lower rank first; tie -> lower left index first
    return x->rank < y->rank || (x->rank == y->rank && x->left < y->left);
}
static void heap_push(heap_t * h, bigram_t b) {
    if (h->n == h->cap) { h->cap = h->cap ? h->cap*2 : 256; h->a = realloc(h->a, h->cap * sizeof(bigram_t)); }
    int i = h->n++;
    h->a[i] = b;
    while (i > 0) {
        int p = (i-1)/2;
        if (bg_less(&h->a[i], &h->a[p])) { bigram_t tmp = h->a[p]; h->a[p] = h->a[i]; h->a[i] = tmp; i = p; }
        else break;
    }
}
static bigram_t heap_pop(heap_t * h) {
    bigram_t top = h->a[0];
    h->a[0] = h->a[--h->n];
    int i = 0;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, m = i;
        if (l < h->n && bg_less(&h->a[l], &h->a[m])) m = l;
        if (r < h->n && bg_less(&h->a[r], &h->a[m])) m = r;
        if (m == i) break;
        bigram_t tmp = h->a[m]; h->a[m] = h->a[i]; h->a[i] = tmp; i = m;
    }
    return top;
}

static int utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xc0) return 1;   // continuation byte alone (defensive)
    if (c < 0xe0) return 2;
    if (c < 0xf0) return 3;
    return 4;
}

typedef struct { g4_tok * t; int * out; int n, cap; } emit_t;
static void emit(emit_t * e, int id) { if (e->n < e->cap) e->out[e->n++] = id; }

// BPE over one pre-tokenized word
static void bpe_word(g4_tok * t, const char * w, int wn, emit_t * e,
                     sym_t ** syms_io, int * syms_cap, heap_t * heap) {
    // gemma4 fix: all-newline words try whole-word lookup first
    bool all_nl = wn > 0;
    for (int i = 0; i < wn; i++) if (w[i] != '\n') { all_nl = false; break; }
    if (all_nl) {
        int id = tmap_find(t, w, wn);
        if (id >= 0) { emit(e, id); return; }
    }

    // split into UTF-8 codepoint symbols
    if (*syms_cap < wn + 1) { *syms_cap = wn + 16; *syms_io = realloc(*syms_io, *syms_cap * sizeof(sym_t)); }
    sym_t * syms = *syms_io;
    int ns = 0;
    for (int off = 0; off < wn; ) {
        int cl = utf8_len((uint8_t)w[off]);
        if (cl > wn - off) cl = wn - off;
        syms[ns].text = w + off; syms[ns].n = cl;
        syms[ns].prev = ns - 1;
        syms[ns].next = (off + cl >= wn) ? -1 : ns + 1;
        ns++; off += cl;
    }

    heap->n = 0;
    #define ADD_BIGRAM(L, R) do { \
        int _l = (L), _r = (R); \
        if (_l != -1 && _r != -1) { \
            int _rank = mmap_find(t, syms[_l].text, syms[_l].n, syms[_r].text, syms[_r].n); \
            if (_rank >= 0) { \
                bigram_t _b = { _l, _r, _rank, syms[_l].n + syms[_r].n }; \
                heap_push(heap, _b); \
            } \
        } \
    } while (0)

    for (int i = 1; i < ns; i++) ADD_BIGRAM(i-1, i);

    while (heap->n) {
        bigram_t bg = heap_pop(heap);
        sym_t * ls = &syms[bg.left];
        sym_t * rs = &syms[bg.right];
        if (ls->n == 0 || rs->n == 0 || ls->n + rs->n != bg.size) continue; // outdated
        // merge right into left
        ls->n += rs->n;
        rs->n = 0;
        ls->next = rs->next;
        if (rs->next >= 0) syms[rs->next].prev = bg.left;
        ADD_BIGRAM(ls->prev, bg.left);
        ADD_BIGRAM(bg.left, ls->next);
    }
    #undef ADD_BIGRAM

    // output
    for (int i = 0; i != -1; i = syms[i].next) {
        if (syms[i].n == 0) continue;
        int id = tmap_find(t, syms[i].text, syms[i].n);
        if (id >= 0) { emit(e, id); continue; }
        for (int j = 0; j < syms[i].n; j++) {
            int bid = t->byte_ids[(uint8_t)syms[i].text[j]];
            if (bid >= 0) emit(e, bid);
        }
    }
}

// tokenize one raw fragment: escape spaces, split on newline runs, BPE each word
static void tokenize_fragment(g4_tok * t, const char * s, int n, emit_t * e,
                              sym_t ** syms, int * syms_cap, heap_t * heap) {
    // escape: ' ' -> U+2581 (0xE2 0x96 0x81)
    char * esc = malloc((size_t)n * 3 + 1);
    int en = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] == ' ') { esc[en++] = (char)0xE2; esc[en++] = (char)0x96; esc[en++] = (char)0x81; }
        else esc[en++] = s[i];
    }
    // regex [^\n]+|[\n]+ : alternating runs
    int i = 0;
    while (i < en) {
        int j = i;
        if (esc[i] == '\n') { while (j < en && esc[j] == '\n') j++; }
        else                { while (j < en && esc[j] != '\n') j++; }
        bpe_word(t, esc + i, j - i, e, syms, syms_cap, heap);
        i = j;
    }
    free(esc);
}

// ---------------------------------------------------- gpt2/qwen35 BPE -------

// unicode category flags for a codepoint (binary search over the ranges table,
// plus the whitespace overlay) — mirrors llama.cpp unicode_cpt_flags_from_cpt.
static uint16_t qcpt_flags(uint32_t cpt) {
    int lo = 0, hi = g4_ucat_nranges - 1, ans = 0;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (g4_ucat_ranges[mid].start <= cpt) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    uint16_t f = g4_ucat_ranges[ans].flags;
    for (int i = 0; i < g4_ucat_nwhitespace; i++) if (g4_ucat_whitespace[i] == cpt) { f |= QF_WS; break; }
    return f;
}

static uint32_t utf8_cpt(const char * s, int n, int * adv) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80)            { *adv = 1; return c; }
    if (c < 0xE0 && n >= 2)  { *adv = 2; return ((uint32_t)(c & 0x1F) << 6)  | (s[1] & 0x3F); }
    if (c < 0xF0 && n >= 3)  { *adv = 3; return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    if (c < 0xF8 && n >= 4)  { *adv = 4; return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) | ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F); }
    *adv = 1; return c;
}

static int cpt_utf8(uint32_t cp, char * o) {
    if (cp < 0x80)    { o[0] = (char)cp; return 1; }
    if (cp < 0x800)   { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

// port of unicode_regex_split_custom_qwen35: split cpts[0..n) into piece lengths
static void qwen_regex_split(const uint32_t * cpts, int n, int * plen, int * np_io) {
    int np = 0, prev_end = 0;
    #define GC(p) (((p) >= 0 && (p) < n) ? (long)cpts[p] : -1L)
    #define GF(p) (((p) >= 0 && (p) < n) ? qcpt_flags(cpts[p]) : (uint16_t)0)
    #define ADD(end) do { int _e = (end); if (_e > prev_end) plen[np++] = _e - prev_end; prev_end = _e; } while (0)
    int pos = 0;
    while (pos < n) {
        long cpt = GC(pos); uint16_t fl = GF(pos);
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < n) {
            long c1 = GC(pos+1); int l1 = (c1 >= 'A' && c1 <= 'Z') ? (int)c1 + 32 : (int)c1;
            if (l1 == 's' || l1 == 't' || l1 == 'm' || l1 == 'd') { pos += 2; ADD(pos); continue; }
            if (pos + 2 < n) {
                long c2 = GC(pos+2); int l2 = (c2 >= 'A' && c2 <= 'Z') ? (int)c2 + 32 : (int)c2;
                if ((l1=='r'&&l2=='e') || (l1=='v'&&l2=='e') || (l1=='l'&&l2=='l')) { pos += 3; ADD(pos); continue; }
            }
        }
        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!(cpt == '\r' || cpt == '\n' || (fl & QF_NUMBER))) {
            if ((fl & QF_LETTER) || (fl & QF_MARK) || (GF(pos+1) & QF_MARK) || (GF(pos+1) & QF_LETTER)) {
                pos++;
                while ((GF(pos) & QF_LETTER) || (GF(pos) & QF_MARK)) pos++;
                ADD(pos); continue;
            }
        }
        // \p{N}
        if (fl & QF_NUMBER) { pos++; ADD(pos); continue; }
        // ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            uint16_t fl2 = (cpt == ' ') ? GF(pos+1) : fl;
            if (!(fl2 & (QF_WS|QF_LETTER|QF_MARK|QF_NUMBER)) && fl) {
                if (cpt == ' ') pos++;
                while (!(fl2 & (QF_WS|QF_LETTER|QF_MARK|QF_NUMBER)) && fl2) { pos++; fl2 = GF(pos); }
                long c2 = GC(pos);
                while (c2 == '\r' || c2 == '\n') { pos++; c2 = GC(pos); }
                ADD(pos); continue;
            }
        }
        // \s*[\r\n]+ | \s+(?!\S) | \s+
        {
            int nw = 0, last_rn = 0;
            while (GF(pos + nw) & QF_WS) { long c2 = GC(pos + nw); if (c2 == '\r' || c2 == '\n') last_rn = pos + nw + 1; nw++; }
            if (last_rn > 0)                       { pos = last_rn;     ADD(pos); continue; }
            if (nw > 1 && GC(pos + nw) != -1L)     { pos += nw - 1;     ADD(pos); continue; }
            if (nw > 0)                            { pos += nw;         ADD(pos); continue; }
        }
        pos++; ADD(pos);   // no match: single codepoint
    }
    #undef GC
    #undef GF
    #undef ADD
    *np_io = np;
}

// BPE-merge one byte-encoded piece and emit token ids
static void qwen_bpe_piece(g4_tok * t, const char * enc, int en, emit_t * e,
                           sym_t ** syms_io, int * syms_cap, heap_t * heap) {
    if (en <= 0) return;
    if (*syms_cap < en + 1) { *syms_cap = en + 16; *syms_io = realloc(*syms_io, *syms_cap * sizeof(sym_t)); }
    sym_t * syms = *syms_io; int ns = 0;
    for (int off = 0; off < en; ) {
        int cl = utf8_len((uint8_t)enc[off]); if (cl > en - off) cl = en - off;
        syms[ns].text = enc + off; syms[ns].n = cl;
        syms[ns].prev = ns - 1; syms[ns].next = (off + cl >= en) ? -1 : ns + 1;
        ns++; off += cl;
    }
    heap->n = 0;
    #define ADDB(L, R) do { int _l=(L), _r=(R); if (_l!=-1 && _r!=-1) { \
        int _rk = mmap_find(t, syms[_l].text, syms[_l].n, syms[_r].text, syms[_r].n); \
        if (_rk >= 0) { bigram_t _b = { _l, _r, _rk, syms[_l].n + syms[_r].n }; heap_push(heap, _b); } } } while (0)
    for (int i = 1; i < ns; i++) ADDB(i-1, i);
    while (heap->n) {
        bigram_t bg = heap_pop(heap);
        sym_t * ls = &syms[bg.left]; sym_t * rs = &syms[bg.right];
        if (ls->n == 0 || rs->n == 0 || ls->n + rs->n != bg.size) continue;
        ls->n += rs->n; rs->n = 0; ls->next = rs->next;
        if (rs->next >= 0) syms[rs->next].prev = bg.left;
        ADDB(ls->prev, bg.left);
        ADDB(bg.left, ls->next);
    }
    #undef ADDB
    for (int i = 0; i != -1; i = syms[i].next) {
        if (syms[i].n == 0) continue;
        int id = tmap_find(t, syms[i].text, syms[i].n);
        if (id >= 0) { emit(e, id); continue; }
        // fallback: each codepoint is a byte-char token (all 256 are in the vocab)
        for (int off = 0; off < syms[i].n; ) {
            int cl = utf8_len((uint8_t)syms[i].text[off]); if (cl > syms[i].n - off) cl = syms[i].n - off;
            int cid = tmap_find(t, syms[i].text + off, cl);
            if (cid >= 0) emit(e, cid);
            off += cl;
        }
    }
}

// tokenize one raw fragment: qwen35 regex split -> byte-encode -> BPE per piece
static void qwen_tokenize_fragment(g4_tok * t, const char * s, int n, emit_t * e,
                                   sym_t ** syms, int * syms_cap, heap_t * heap) {
    uint32_t * cpts = malloc((size_t)n * 4 + 4);
    int nc = 0;
    for (int off = 0; off < n; ) { int adv; cpts[nc++] = utf8_cpt(s + off, n - off, &adv); off += adv; }
    int * plen = malloc((size_t)(nc + 1) * sizeof(int)); int np = 0;
    qwen_regex_split(cpts, nc, plen, &np);
    char * enc = malloc((size_t)nc * 8 + 16);   // each cpt -> <=4 utf8 bytes -> <=2 b2u bytes each
    int ci = 0;
    for (int p = 0; p < np; p++) {
        int en = 0;
        for (int k = 0; k < plen[p]; k++) {
            char ub[4]; int ul = cpt_utf8(cpts[ci + k], ub);
            for (int b = 0; b < ul; b++) {
                uint8_t by = (uint8_t)ub[b];
                memcpy(enc + en, t->b2u[by], t->b2u_len[by]); en += t->b2u_len[by];
            }
        }
        qwen_bpe_piece(t, enc, en, e, syms, syms_cap, heap);
        ci += plen[p];
    }
    free(cpts); free(plen); free(enc);
}

int g4_tok_encode(g4_tok * t, const char * text, int * ids, int cap,
                  bool add_bos, bool parse_special) {
    emit_t e = { t, ids, 0, cap };
    if (add_bos && t->mf->bos_id >= 0) emit(&e, t->mf->bos_id);

    int n = (int)strlen(text);

    // fragment list: (start,len,token). token>=0 means special token fragment.
    // For each special (longest first) rebuild the list, splitting raw fragments
    // at every occurrence — same result as llama.cpp tokenizer_st_partition.
    typedef struct { int start, len, token; } frag_t;
    int fcap = 64, fn = 1;
    frag_t * fr  = malloc(fcap * sizeof(frag_t));
    frag_t * nf  = malloc(fcap * sizeof(frag_t));
    fr[0].start = 0; fr[0].len = n; fr[0].token = -1;

    for (int si = 0; si < t->n_specials; si++) {
        int id = t->specials[si];
        int ty = t->mf->tok_type[id];
        if (!parse_special && (ty == TT_CONTROL || ty == TT_UNKNOWN)) continue;
        g4_str sx = t->mf->tok_text[id];
        if (sx.len == 0 || (int)sx.len > n) continue;

        int nn = 0;
        #define PUSH(S, L, T) do { \
            if (nn == fcap) { fcap *= 2; fr = realloc(fr, fcap * sizeof(frag_t)); nf = realloc(nf, fcap * sizeof(frag_t)); } \
            nf[nn].start = (S); nf[nn].len = (L); nf[nn].token = (T); nn++; \
        } while (0)
        for (int fi = 0; fi < fn; fi++) {
            if (fr[fi].token >= 0 || fr[fi].len == 0) { PUSH(fr[fi].start, fr[fi].len, fr[fi].token); continue; }
            int base = fr[fi].start, len = fr[fi].len, cur = 0;
            for (int off = 0; off + (int)sx.len <= len; ) {
                if (memcmp(text + base + off, sx.ptr, sx.len) == 0) {
                    if (off > cur) PUSH(base + cur, off - cur, -1);
                    PUSH(base + off, (int)sx.len, id);
                    off += (int)sx.len;
                    cur = off;
                } else off++;
            }
            if (cur < len) PUSH(base + cur, len - cur, -1);
        }
        #undef PUSH
        frag_t * tmp = fr; fr = nf; nf = tmp;
        fn = nn;
    }

    sym_t * syms = NULL; int syms_cap = 0;
    heap_t heap = {0};
    for (int fi = 0; fi < fn; fi++) {
        if (fr[fi].token >= 0) emit(&e, fr[fi].token);
        else if (fr[fi].len > 0) {
            if (t->pre == 1) qwen_tokenize_fragment(t, text + fr[fi].start, fr[fi].len, &e, &syms, &syms_cap, &heap);
            else                  tokenize_fragment(t, text + fr[fi].start, fr[fi].len, &e, &syms, &syms_cap, &heap);
        }
    }
    free(syms); free(heap.a); free(fr); free(nf);
    return e.n;
}

// --------------------------------------------------------------- decode -----

int g4_tok_decode(g4_tok * t, int id, char * buf, bool render_special) {
    if (id < 0 || id >= t->mf->tok_n) return 0;
    int ty = t->mf->tok_type[id];
    g4_str s = t->mf->tok_text[id];

    // gpt2/qwen35: NORMAL tokens are byte-encoded (map each codepoint back to a byte)
    if (t->pre == 1) {
        if (ty == TT_NORMAL) {
            int o = 0;
            for (uint32_t i = 0; i < s.len && o < 250; ) {
                int adv; uint32_t cp = utf8_cpt(s.ptr + i, (int)(s.len - i), &adv);
                if (cp < 0x400 && t->cpt2byte[cp] >= 0) buf[o++] = (char)t->cpt2byte[cp];
                i += adv;
            }
            return o;
        }
        // control / user-defined / unknown: literal text
        if (!render_special && (ty == TT_CONTROL || ty == TT_UNKNOWN || ty == TT_UNUSED)) return 0;
        int n = s.len < 250 ? (int)s.len : 250;
        memcpy(buf, s.ptr, n);
        return n;
    }
    if (ty == TT_BYTE) {
        // <0xXX>
        if (s.len == 6 && s.ptr[0] == '<') {
            int hi = s.ptr[3] <= '9' ? s.ptr[3]-'0' : (s.ptr[3]|32)-'a'+10;
            int lo = s.ptr[4] <= '9' ? s.ptr[4]-'0' : (s.ptr[4]|32)-'a'+10;
            buf[0] = (char)((hi << 4) | lo);
            return 1;
        }
        return 0;
    }
    if (ty == TT_CONTROL || ty == TT_USER_DEF || ty == TT_UNKNOWN || ty == TT_UNUSED) {
        if (!render_special) return 0;
        int n = s.len < 250 ? (int)s.len : 250;
        memcpy(buf, s.ptr, n);
        return n;
    }
    // normal: unescape U+2581 -> ' '
    int o = 0;
    for (uint32_t i = 0; i < s.len && o < 250; ) {
        if (i + 2 < s.len && (uint8_t)s.ptr[i] == 0xE2 && (uint8_t)s.ptr[i+1] == 0x96 && (uint8_t)s.ptr[i+2] == 0x81) {
            buf[o++] = ' '; i += 3;
        } else buf[o++] = s.ptr[i++];
    }
    return o;
}
