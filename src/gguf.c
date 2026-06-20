// gguf.c — GGUF v3 parser with Win32 memory mapping
#include "g4run.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// 1 = copy-on-write view (default; needed for Vulkan host-pointer import).
// 0 = read-only view (no pagefile commit charge) — set for CPU-only runs.
int g4_mmap_cow = 1;

// ---- type tables ----
static const struct { int t; const char * name; int blck; int size; } type_tab[] = {
    { G4_F32,   "F32",   1,   4   },
    { G4_F16,   "F16",   1,   2   },
    { G4_BF16,  "BF16",  1,   2   },
    { G4_Q4_0,  "Q4_0",  32,  18  },
    { G4_Q8_0,  "Q8_0",  32,  34  },
    { G4_Q2_K,  "Q2_K",  256, 84  },
    { G4_Q3_K,  "Q3_K",  256, 110 },
    { G4_Q4_K,  "Q4_K",  256, 144 },
    { G4_Q5_K,  "Q5_K",  256, 176 },
    { G4_Q6_K,  "Q6_K",  256, 210 },
    { G4_TQ2_0, "TQ2_0", 256, 66  },
    { G4_I8,  "I8",  1, 1 }, { G4_I16, "I16", 1, 2 },
    { G4_I32, "I32", 1, 4 }, { G4_I64, "I64", 1, 8 }, { G4_F64, "F64", 1, 8 },
};

const char * g4_type_name(int t) {
    for (size_t i = 0; i < sizeof(type_tab)/sizeof(type_tab[0]); i++)
        if (type_tab[i].t == t) return type_tab[i].name;
    return "?";
}
int g4_type_blck(int t) {
    for (size_t i = 0; i < sizeof(type_tab)/sizeof(type_tab[0]); i++)
        if (type_tab[i].t == t) return type_tab[i].blck;
    return 0;
}
size_t g4_type_size(int t) {
    for (size_t i = 0; i < sizeof(type_tab)/sizeof(type_tab[0]); i++)
        if (type_tab[i].t == t) return type_tab[i].size;
    return 0;
}
size_t g4_row_size(int t, int64_t n) {
    int b = g4_type_blck(t);
    if (!b || n % b) return 0;
    return (size_t)(n / b) * g4_type_size(t);
}

// ---- cursor over the mapped header ----
typedef struct { const uint8_t * p, * end; int err; } cur_t;

static uint64_t rd_u64(cur_t * c) { if (c->p + 8 > c->end) { c->err = 1; return 0; } uint64_t v; memcpy(&v, c->p, 8); c->p += 8; return v; }
static uint32_t rd_u32(cur_t * c) { if (c->p + 4 > c->end) { c->err = 1; return 0; } uint32_t v; memcpy(&v, c->p, 4); c->p += 4; return v; }
static int32_t  rd_i32(cur_t * c) { return (int32_t)rd_u32(c); }
static float    rd_f32(cur_t * c) { uint32_t v = rd_u32(c); float f; memcpy(&f, &v, 4); return f; }
static uint8_t  rd_u8 (cur_t * c) { if (c->p + 1 > c->end) { c->err = 1; return 0; } return *c->p++; }
static g4_str   rd_str(cur_t * c) {
    g4_str s = {0,0};
    uint64_t n = rd_u64(c);
    if (c->err || c->p + n > c->end) { c->err = 1; return s; }
    s.ptr = (const char *)c->p; s.len = (uint32_t)n; c->p += n;
    return s;
}

// gguf value type ids
enum { GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64 };

static const int gv_scalar_size[] = { 1,1,2,2,4,4,4,1,0,0,8,8,8 };

// skip a value of given type; for scalars/strings returns position info
static void skip_val(cur_t * c, uint32_t vt) {
    switch (vt) {
        case GV_STR: rd_str(c); break;
        case GV_ARR: {
            uint32_t et = rd_u32(c);
            uint64_t n  = rd_u64(c);
            if (et == GV_STR) { for (uint64_t i = 0; i < n && !c->err; i++) rd_str(c); }
            else if (et == GV_ARR) { c->err = 1; }
            else {
                uint64_t sz = (uint64_t)gv_scalar_size[et] * n;
                if (c->p + sz > c->end) { c->err = 1; return; }
                c->p += sz;
            }
        } break;
        default:
            if (vt > GV_F64) { c->err = 1; return; }
            c->p += gv_scalar_size[vt];
    }
}

static bool key_is(g4_str k, const char * s) {
    size_t n = strlen(s);
    return k.len == n && memcmp(k.ptr, s, n) == 0;
}

// arch-agnostic hparam key: matches "gemma4.<suffix>" or "gemma4-assistant.<suffix>"
// (the draft head uses the gemma4-assistant.* prefix). Order-independent: works
// whether or not general.architecture has been parsed yet.
static bool gkey(g4_str k, const char * suffix) {
    char buf[96];
    int n = snprintf(buf, sizeof buf, "gemma4.%s", suffix);
    if (k.len == (uint32_t)n && memcmp(k.ptr, buf, n) == 0) return true;
    n = snprintf(buf, sizeof buf, "gemma4-assistant.%s", suffix);
    return k.len == (uint32_t)n && memcmp(k.ptr, buf, n) == 0;
}

// read scalar value as double (u8..f64, bool)
static double rd_scalar(cur_t * c, uint32_t vt) {
    switch (vt) {
        case GV_U8:   return rd_u8(c);
        case GV_I8:   return (int8_t)rd_u8(c);
        case GV_U16:  { uint16_t v; if (c->p+2>c->end){c->err=1;return 0;} memcpy(&v,c->p,2); c->p+=2; return v; }
        case GV_I16:  { int16_t  v; if (c->p+2>c->end){c->err=1;return 0;} memcpy(&v,c->p,2); c->p+=2; return v; }
        case GV_U32:  return rd_u32(c);
        case GV_I32:  return rd_i32(c);
        case GV_F32:  return rd_f32(c);
        case GV_BOOL: return rd_u8(c) != 0;
        case GV_U64:  return (double)rd_u64(c);
        case GV_I64:  return (double)(int64_t)rd_u64(c);
        case GV_F64:  { uint64_t v = rd_u64(c); double d; memcpy(&d, &v, 8); return d; }
    }
    c->err = 1; return 0;
}

int g4_gguf_open(g4_model_file * mf, const char * path, bool verbose) {
    memset(mf, 0, sizeof(*mf));
    mf->alignment = 32;
    mf->bos_id = mf->eos_id = mf->eot_id = mf->unk_id = mf->pad_id = -1;
    mf->meta_temp = -1; mf->meta_top_p = -1; mf->meta_top_k = -1;
    mf->add_bos = true;
    mf->n_ctx_train = 131072;

    // --- map file ---
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (hf == INVALID_HANDLE_VALUE) { fprintf(stderr, "g4: cannot open %s\n", path); return -1; }
    LARGE_INTEGER sz; GetFileSizeEx(hf, &sz);
    // Copy-on-write mapping: identical lazy paging for the CPU path (we never write),
    // but the pages are registerable for Vulkan host-pointer import, which rejects
    // PAGE_READONLY views on the Intel driver. COW commits the whole file against the
    // pagefile, though — under tight commit a big read pass (e.g. writing an exported
    // copy after a long finetune) can fault. g4_mmap_cow=0 selects a read-only view
    // (file-backed, zero commit charge) for the CPU-only paths.
    DWORD prot = g4_mmap_cow ? PAGE_WRITECOPY : PAGE_READONLY;
    DWORD acc  = g4_mmap_cow ? FILE_MAP_COPY  : FILE_MAP_READ;
    HANDLE hm = CreateFileMappingA(hf, NULL, prot, 0, 0, NULL);
    if (!hm) { CloseHandle(hf); fprintf(stderr, "g4: CreateFileMapping failed\n"); return -1; }
    void * base = MapViewOfFile(hm, acc, 0, 0, 0);
    if (!base) { CloseHandle(hm); CloseHandle(hf); fprintf(stderr, "g4: MapViewOfFile failed\n"); return -1; }
    mf->map_base = base; mf->map_size = (uint64_t)sz.QuadPart;
    mf->h_file = hf; mf->h_map = hm;

    cur_t c = { (const uint8_t *)base, (const uint8_t *)base + mf->map_size, 0 };

    if (rd_u32(&c) != 0x46554747u) { fprintf(stderr, "g4: not a GGUF file\n"); return -1; } // 'GGUF'
    mf->version   = rd_u32(&c);
    mf->n_tensors = rd_u64(&c);
    mf->n_kv      = rd_u64(&c);
    if (mf->version < 2 || mf->version > 3) { fprintf(stderr, "g4: unsupported GGUF version %u\n", mf->version); return -1; }

    // --- parse KVs ---
    for (uint64_t i = 0; i < mf->n_kv && !c.err; i++) {
        g4_str key = rd_str(&c);
        uint32_t vt = rd_u32(&c);
        if (c.err) break;

        // string values we keep
        if (vt == GV_STR) {
            g4_str v = rd_str(&c);
            if      (key_is(key, "general.architecture")) {
                size_t n = v.len < sizeof(mf->arch)-1 ? v.len : sizeof(mf->arch)-1;
                memcpy(mf->arch, v.ptr, n); mf->arch[n] = 0;
            }
            else if (key_is(key, "tokenizer.chat_template")) mf->chat_template = v;
            else if (key_is(key, "tokenizer.ggml.pre")) {
                if ((v.len == 6 && memcmp(v.ptr, "qwen35", 6) == 0)) mf->tok_pre = 1;
            }
            else if (key_is(key, "tokenizer.ggml.model")) {
                if (v.len == 4 && memcmp(v.ptr, "gpt2", 4) == 0) mf->tok_pre = 1;
            }
            continue;
        }

        if (vt == GV_ARR) {
            uint32_t et = rd_u32(&c);
            uint64_t n  = rd_u64(&c);
            if (key_is(key, "tokenizer.ggml.tokens") && et == GV_STR) {
                mf->tok_n = (int)n;
                mf->tok_text = (g4_str *)malloc(n * sizeof(g4_str));
                for (uint64_t j = 0; j < n && !c.err; j++) mf->tok_text[j] = rd_str(&c);
            } else if (key_is(key, "tokenizer.ggml.merges") && et == GV_STR) {
                mf->merges_n = (int)n;
                mf->merges = (g4_str *)malloc(n * sizeof(g4_str));
                for (uint64_t j = 0; j < n && !c.err; j++) mf->merges[j] = rd_str(&c);
            } else if (key_is(key, "tokenizer.ggml.token_type") && et == GV_I32) {
                mf->tok_type = (const int32_t *)c.p;
                c.p += 4 * n;
            } else if (gkey(key, "feed_forward_length") && (et == GV_U32 || et == GV_I32)) {
                for (uint64_t j = 0; j < n && j < G4_MAX_LAYERS; j++) { memcpy(&mf->n_ff[j], c.p + 4*j, 4); }
                c.p += 4 * n;
            } else if (gkey(key, "attention.sliding_window_pattern") && et == GV_BOOL) {
                for (uint64_t j = 0; j < n && j < G4_MAX_LAYERS; j++) mf->is_swa[j] = c.p[j];
                c.p += n;
            } else if (key_is(key, "qwen35.rope.dimension_sections") && (et == GV_U32 || et == GV_I32)) {
                for (uint64_t j = 0; j < n && j < 4; j++) memcpy(&mf->rope_sections[j], c.p + 4*j, 4);
                c.p += 4 * n;
            } else if (key_is(key, "qwen35.attention.recurrent_layers") && et == GV_BOOL) {
                for (uint64_t j = 0; j < n && j < G4_MAX_LAYERS; j++) mf->is_recr[j] = c.p[j];
                mf->full_attn_interval = -1;   // explicit per-layer list provided
                c.p += n;
            } else {
                // skip other arrays
                if (et == GV_STR) { for (uint64_t j = 0; j < n && !c.err; j++) rd_str(&c); }
                else c.p += (uint64_t)gv_scalar_size[et] * n;
            }
            continue;
        }

        // scalars
        uint64_t sc_off = (uint64_t)(c.p - (const uint8_t *)base);   // offset of this scalar's value bytes
        double dv = rd_scalar(&c, vt);
        if      (key_is(key, "general.alignment"))                          mf->alignment = (uint32_t)dv;
        else if (gkey(key, "block_count"))                                  mf->n_layer = (int)dv;
        else if (gkey(key, "context_length"))                               mf->n_ctx_train = (int)dv;
        else if (gkey(key, "embedding_length"))                             mf->n_embd = (int)dv;
        else if (gkey(key, "embedding_length_out"))                         mf->n_embd_out = (int)dv;
        else if (gkey(key, "nextn_predict_layers"))                         mf->n_nextn = (int)dv;
        else if (gkey(key, "feed_forward_length"))                          { for (int j = 0; j < G4_MAX_LAYERS; j++) mf->n_ff[j] = (int)dv; }
        else if (gkey(key, "attention.head_count"))                         mf->n_head = (int)dv;
        else if (gkey(key, "attention.head_count_kv"))                      mf->n_head_kv = (int)dv;
        else if (gkey(key, "rope.freq_base"))                               mf->rope_base_full = (float)dv;
        else if (gkey(key, "rope.freq_base_swa"))                           mf->rope_base_swa = (float)dv;
        else if (gkey(key, "attention.layer_norm_rms_epsilon"))             mf->rms_eps = (float)dv;
        else if (gkey(key, "attention.key_length"))                         mf->head_dim_full = (int)dv;
        else if (gkey(key, "attention.key_length_swa"))                     mf->head_dim_swa = (int)dv;
        else if (gkey(key, "final_logit_softcapping"))                      mf->final_softcap = (float)dv;
        else if (gkey(key, "attention.sliding_window"))                     mf->n_swa = (int)dv;
        else if (gkey(key, "attention.shared_kv_layers"))                   mf->n_kv_shared = (int)dv;
        else if (gkey(key, "embedding_length_per_layer_input"))             mf->n_embd_per_layer = (int)dv;
        else if (gkey(key, "rope.dimension_count"))                         mf->n_rot_full = (int)dv;
        else if (gkey(key, "rope.dimension_count_swa"))                     mf->n_rot_swa = (int)dv;
        // ---- qwen35 (hybrid gated-deltanet + attention) ----
        else if (key_is(key, "qwen35.block_count"))                         mf->n_layer = (int)dv;
        else if (key_is(key, "qwen35.context_length"))                      mf->n_ctx_train = (int)dv;
        else if (key_is(key, "qwen35.embedding_length"))                    mf->n_embd = (int)dv;
        else if (key_is(key, "qwen35.feed_forward_length"))                 { for (int j = 0; j < G4_MAX_LAYERS; j++) mf->n_ff[j] = (int)dv; mf->ff_len_off = sc_off; mf->ff_len_vt = (int)vt; }
        else if (key_is(key, "qwen35.attention.head_count"))                mf->n_head = (int)dv;
        else if (key_is(key, "qwen35.attention.head_count_kv"))             mf->n_head_kv = (int)dv;
        else if (key_is(key, "qwen35.attention.key_length"))                mf->n_embd_head_k = (int)dv;
        else if (key_is(key, "qwen35.attention.value_length"))              mf->n_embd_head_v = (int)dv;
        else if (key_is(key, "qwen35.attention.layer_norm_rms_epsilon"))    mf->rms_eps = (float)dv;
        else if (key_is(key, "qwen35.rope.freq_base"))                      mf->rope_base_full = (float)dv;
        else if (key_is(key, "qwen35.rope.dimension_count"))                mf->n_rot_full = (int)dv;
        else if (key_is(key, "qwen35.full_attention_interval"))             mf->full_attn_interval = (int)dv;
        else if (key_is(key, "qwen35.ssm.conv_kernel"))                     mf->ssm_d_conv = (int)dv;
        else if (key_is(key, "qwen35.ssm.inner_size"))                      mf->ssm_d_inner = (int)dv;
        else if (key_is(key, "qwen35.ssm.state_size"))                      mf->ssm_d_state = (int)dv;
        else if (key_is(key, "qwen35.ssm.time_step_rank"))                  mf->ssm_dt_rank = (int)dv;
        else if (key_is(key, "qwen35.ssm.group_count"))                     mf->ssm_n_group = (int)dv;
        else if (key_is(key, "tokenizer.ggml.bos_token_id"))                mf->bos_id = (int)dv;
        else if (key_is(key, "tokenizer.ggml.eos_token_id"))                mf->eos_id = (int)dv;
        else if (key_is(key, "tokenizer.ggml.eot_token_id"))                mf->eot_id = (int)dv;
        else if (key_is(key, "tokenizer.ggml.unknown_token_id"))            mf->unk_id = (int)dv;
        else if (key_is(key, "tokenizer.ggml.padding_token_id"))            mf->pad_id = (int)dv;
        else if (key_is(key, "tokenizer.ggml.add_bos_token"))               mf->add_bos = dv != 0;
        else if (key_is(key, "general.sampling.temp"))                      mf->meta_temp = (float)dv;
        else if (key_is(key, "general.sampling.top_p"))                     mf->meta_top_p = (float)dv;
        else if (key_is(key, "general.sampling.top_k"))                     mf->meta_top_k = (int)dv;
        (void)skip_val; // silence unused (all values consumed via rd_scalar)
    }
    if (c.err) { fprintf(stderr, "g4: header parse error\n"); return -1; }
    mf->kv_end = (uint64_t)(c.p - (const uint8_t *)base);   // start of tensor-info section

    // --- tensor infos ---
    mf->tensors = (g4_tensor *)calloc(mf->n_tensors, sizeof(g4_tensor));
    for (uint64_t i = 0; i < mf->n_tensors && !c.err; i++) {
        g4_tensor * t = &mf->tensors[i];
        g4_str name = rd_str(&c);
        size_t n = name.len < sizeof(t->name)-1 ? name.len : sizeof(t->name)-1;
        memcpy(t->name, name.ptr, n); t->name[n] = 0;
        t->ndim = (int)rd_u32(&c);
        if (t->ndim > 4) { c.err = 1; break; }
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        for (int d = 0; d < t->ndim; d++) t->ne[d] = (int64_t)rd_u64(&c);
        t->type   = (int)rd_u32(&c);
        t->offset = rd_u64(&c);
    }
    if (c.err) { fprintf(stderr, "g4: tensor info parse error\n"); return -1; }

    uint64_t header_end = (uint64_t)(c.p - (const uint8_t *)base);
    mf->data_start = (header_end + mf->alignment - 1) / mf->alignment * mf->alignment;

    // resolve data pointers + validate bounds
    for (uint64_t i = 0; i < mf->n_tensors; i++) {
        g4_tensor * t = &mf->tensors[i];
        int64_t nelem = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        size_t  rs = g4_row_size(t->type, t->ne[0]);
        if (!rs) { fprintf(stderr, "g4: tensor %s has unsupported type %d\n", t->name, t->type); return -1; }
        size_t bytes = rs * (size_t)(nelem / t->ne[0]);
        if (mf->data_start + t->offset + bytes > mf->map_size) {
            fprintf(stderr, "g4: tensor %s out of bounds\n", t->name); return -1;
        }
        t->data = (const uint8_t *)base + mf->data_start + t->offset;
    }

    // --- validate arch ---
    mf->is_assistant = (strcmp(mf->arch, "gemma4-assistant") == 0);
    mf->is_qwen35    = (strcmp(mf->arch, "qwen35") == 0);
    if (mf->is_qwen35) {
        if (mf->n_layer > G4_MAX_LAYERS) { fprintf(stderr, "g4: too many layers\n"); return -1; }
        if (mf->full_attn_interval == 0) mf->full_attn_interval = 4;
        // is_recr(i) = (i+1) % full_attn_interval != 0  (unless an explicit list was given)
        if (mf->full_attn_interval > 0)
            for (int i = 0; i < mf->n_layer; i++)
                mf->is_recr[i] = ((i + 1) % mf->full_attn_interval != 0);
        if (verbose) {
            fprintf(stderr, "g4: arch=qwen35 n_layer=%d n_embd=%d n_vocab=%d heads=%d/%d hd=%d ssm[conv=%d inner=%d state=%d dt=%d grp=%d] rope_sec=%d,%d,%d,%d\n",
                    mf->n_layer, mf->n_embd, mf->tok_n, mf->n_head, mf->n_head_kv, mf->n_embd_head_k,
                    mf->ssm_d_conv, mf->ssm_d_inner, mf->ssm_d_state, mf->ssm_dt_rank, mf->ssm_n_group,
                    mf->rope_sections[0], mf->rope_sections[1], mf->rope_sections[2], mf->rope_sections[3]);
        }
        mf->n_vocab = mf->tok_n;
        return 0;
    }
    if (strcmp(mf->arch, "gemma4") != 0 && !mf->is_assistant) {
        fprintf(stderr, "g4: unsupported architecture '%s' (need gemma4, gemma4-assistant, or qwen35)\n", mf->arch);
        return -1;
    }
    // llama.cpp forces add_bos=true for gemma4 regardless of metadata
    // (workaround for broken conversions, llama.cpp PR 21500)
    mf->add_bos = true;
    mf->n_layer_kv = mf->n_layer - mf->n_kv_shared;
    // The assistant (MTP draft) has n_kv_shared == n_layer (n_layer_kv == 0): all
    // its attention layers read the *target's* KV cache, producing none of their own.
    if (mf->n_layer > G4_MAX_LAYERS || (!mf->is_assistant && mf->n_layer_kv < 2)) {
        fprintf(stderr, "g4: bad layer config\n"); return -1;
    }

    if (verbose) {
        fprintf(stderr, "g4: %s: %u tensors, %u kv, data @ %llu, %.2f GB\n",
                path, (unsigned)mf->n_tensors, (unsigned)mf->n_kv,
                (unsigned long long)mf->data_start, mf->map_size / 1e9);
        fprintf(stderr, "g4: arch=%s n_layer=%d n_embd=%d n_vocab=%d heads=%d/%d hd=%d/%d swa=%d kv_layers=%d\n",
                mf->arch, mf->n_layer, mf->n_embd, mf->tok_n, mf->n_head, mf->n_head_kv,
                mf->head_dim_full, mf->head_dim_swa, mf->n_swa, mf->n_layer_kv);
    }
    mf->n_vocab = mf->tok_n;
    return 0;
}

// Serialize the current model state to a new GGUF: the metadata-KV section is
// copied verbatim from the source mapping (preserving tokenizer etc.), and the
// tensor infos + data are re-emitted from mf->tensors[] as they are *now* — so a
// tensor that was resized/repointed in place (e.g. a pruned FFN matrix) is written
// at its new shape/type. Requires the source mapping to still be open.
int g4_gguf_write(const g4_model_file * mf, const char * path) {
    if (!mf->map_base || !mf->kv_end) { fprintf(stderr, "g4: gguf_write needs the source mapping\n"); return 1; }
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "g4: cannot create %s\n", path); return 1; }
    const uint8_t * base = (const uint8_t *)mf->map_base;
    const uint32_t magic = 0x46554747u, ver = mf->version;
    fwrite(&magic, 4, 1, f); fwrite(&ver, 4, 1, f);
    fwrite(&mf->n_tensors, 8, 1, f); fwrite(&mf->n_kv, 8, 1, f);
    // Metadata KV blob: copied verbatim, except feed_forward_length is patched to the
    // actual (possibly pruned) FFN width so llama.cpp's tensor-shape check passes — it
    // derives the expected ffn_gate/up/down shape from this scalar, not from the tensors.
    int64_t new_ff = 0;
    for (uint64_t i = 0; i < mf->n_tensors; i++)
        if (strstr(mf->tensors[i].name, "ffn_gate")) { new_ff = mf->tensors[i].ne[1]; break; }
    if (mf->ff_len_off && new_ff > 0 && mf->ff_len_vt <= GV_F64 && gv_scalar_size[mf->ff_len_vt] == 4) {
        fwrite(base + 24, 1, (size_t)(mf->ff_len_off - 24), f);              // KV before the value
        uint32_t v = (uint32_t)new_ff; fwrite(&v, 4, 1, f);                  // patched feed_forward_length
        fwrite(base + mf->ff_len_off + 4, 1, (size_t)(mf->kv_end - mf->ff_len_off - 4), f);  // KV after
    } else {
        fwrite(base + 24, 1, (size_t)(mf->kv_end - 24), f);                  // verbatim (no patch needed)
    }

    uint64_t * off = malloc(mf->n_tensors * sizeof(uint64_t));
    uint64_t * sz  = malloc(mf->n_tensors * sizeof(uint64_t));
    uint64_t cur = 0;
    for (uint64_t i = 0; i < mf->n_tensors; i++) {
        const g4_tensor * t = &mf->tensors[i];
        int64_t nelem = t->ne[0]*t->ne[1]*t->ne[2]*t->ne[3];
        sz[i]  = (uint64_t)g4_row_size(t->type, t->ne[0]) * (uint64_t)(nelem / t->ne[0]);
        off[i] = cur;
        cur += (sz[i] + mf->alignment - 1) / mf->alignment * mf->alignment;
    }
    for (uint64_t i = 0; i < mf->n_tensors; i++) {              // tensor infos
        const g4_tensor * t = &mf->tensors[i];
        uint64_t nl = strlen(t->name);
        fwrite(&nl, 8, 1, f); fwrite(t->name, 1, nl, f);
        uint32_t nd = (uint32_t)t->ndim; fwrite(&nd, 4, 1, f);
        for (int d = 0; d < t->ndim; d++) { uint64_t e = (uint64_t)t->ne[d]; fwrite(&e, 8, 1, f); }
        uint32_t ty = (uint32_t)t->type; fwrite(&ty, 4, 1, f);
        fwrite(&off[i], 8, 1, f);
    }
    long pos = ftell(f), ds = (pos + mf->alignment - 1) / mf->alignment * mf->alignment;
    for (long k = pos; k < ds; k++) fputc(0, f);               // pad to data_start
    for (uint64_t i = 0; i < mf->n_tensors; i++) {             // tensor data (each aligned)
        fwrite(mf->tensors[i].data, 1, (size_t)sz[i], f);
        uint64_t pad = (sz[i] + mf->alignment - 1) / mf->alignment * mf->alignment;
        for (uint64_t k = sz[i]; k < pad; k++) fputc(0, f);
    }
    free(off); free(sz);
    fclose(f);
    return 0;
}

void g4_gguf_close(g4_model_file * mf) {
    free(mf->tensors); free(mf->tok_text); free(mf->merges);
    if (mf->map_base) UnmapViewOfFile(mf->map_base);
    if (mf->h_map)  CloseHandle((HANDLE)mf->h_map);
    if (mf->h_file) CloseHandle((HANDLE)mf->h_file);
    memset(mf, 0, sizeof(*mf));
}

const g4_tensor * g4_find_tensor(const g4_model_file * mf, const char * name) {
    for (uint64_t i = 0; i < mf->n_tensors; i++)
        if (strcmp(mf->tensors[i].name, name) == 0) return &mf->tensors[i];
    return NULL;
}

void g4_gguf_dump(const g4_model_file * mf) {
    printf("version=%u n_tensors=%llu n_kv=%llu data_start=%llu align=%u\n",
           mf->version, (unsigned long long)mf->n_tensors,
           (unsigned long long)mf->n_kv, (unsigned long long)mf->data_start, mf->alignment);
    printf("arch=%s n_layer=%d n_embd=%d n_vocab=%d n_head=%d n_head_kv=%d\n",
           mf->arch, mf->n_layer, mf->n_embd, mf->n_vocab, mf->n_head, mf->n_head_kv);
    printf("head_dim=%d/%d n_rot=%d/%d rope_base=%.0f/%.0f swa=%d shared_kv=%d eps=%g softcap=%g\n",
           mf->head_dim_full, mf->head_dim_swa, mf->n_rot_full, mf->n_rot_swa,
           mf->rope_base_full, mf->rope_base_swa, mf->n_swa, mf->n_kv_shared,
           mf->rms_eps, mf->final_softcap);
    printf("per_layer_embd=%d ctx_train=%d bos=%d eos=%d eot=%d add_bos=%d merges=%d\n",
           mf->n_embd_per_layer, mf->n_ctx_train, mf->bos_id, mf->eos_id, mf->eot_id,
           mf->add_bos, mf->merges_n);
    printf("swa_pattern=");
    for (int i = 0; i < mf->n_layer; i++) putchar(mf->is_swa[i] ? 'S' : 'G');
    printf("\nffn=");
    for (int i = 0; i < mf->n_layer; i++) printf("%d%s", mf->n_ff[i], i+1<mf->n_layer?",":"\n");
    // quant histogram
    int hist[G4_TYPE_COUNT] = {0};
    for (uint64_t i = 0; i < mf->n_tensors; i++)
        if (mf->tensors[i].type < G4_TYPE_COUNT) hist[mf->tensors[i].type]++;
    printf("quant histogram: ");
    for (int t = 0; t < G4_TYPE_COUNT; t++)
        if (hist[t]) printf("%s:%d ", g4_type_name(t), hist[t]);
    printf("\nfirst tensors:\n");
    for (uint64_t i = 0; i < mf->n_tensors && i < 24; i++) {
        const g4_tensor * t = &mf->tensors[i];
        printf("  %-40s [%lld,%lld] %s\n", t->name,
               (long long)t->ne[0], (long long)t->ne[1], g4_type_name(t->type));
    }
}

double g4_time_ms(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    return 1000.0 * now.QuadPart / freq.QuadPart;
}
