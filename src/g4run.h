// g4run — minimal fast gemma4 GGUF inference engine for Windows x64 (MSYS2 MinGW64)
// Spec: ../SPEC.md (transcribed from llama.cpp @ daf6bc9f2)
#ifndef G4RUN_H
#define G4RUN_H

#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------- config ----
#define G4_DEFAULT_CTX      8192
#define G4_DEFAULT_THREADS  8
#define G4_MAX_THREADS      32
#define G4_MAX_LAYERS       64
#define G4_BATCH_MAX        64      // prefill batch size
#define G4_SWA_WINDOW_MAX   512     // SWA window
// SWA KV ring must hold window + a full batch of new entries simultaneously
#define G4_SWA_RING         (G4_SWA_WINDOW_MAX + G4_BATCH_MAX)

// ------------------------------------------------------------- fp16/bf16 ----
typedef uint16_t g4_fp16;
typedef uint16_t g4_bf16;

extern float g4_fp16_to_fp32_table[65536];   // built by g4_init_tables()
extern g4_fp16 g4_gelu_fp16_table[65536];    // ggml-compatible gelu LUT

static inline float g4_fp16_to_fp32(g4_fp16 h) { return g4_fp16_to_fp32_table[h]; }

static inline g4_fp16 g4_fp32_to_fp16(float f) {
    // IEEE 754 round-to-nearest-even, matches F16C / ggml
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t em   = x & 0x7fffffffu;
    if (em >= 0x47800000u) {                     // overflow / inf / nan
        return (g4_fp16)(em > 0x7f800000u ? sign | 0x7e00u | ((em >> 13) & 0x3ffu ? (em >> 13) & 0x3ffu : 1u)
                                          : sign | 0x7c00u);
    }
    if (em < 0x38800000u) {                      // subnormal / zero
        // result = round(M * 2^(E-126)) in units of 2^-24, M = 24-bit mantissa
        if (em < 0x33000000u) return (g4_fp16)sign;
        uint32_t shift = 126u - (em >> 23);      // 14..24
        uint32_t mant  = (em & 0x7fffffu) | 0x800000u;
        uint32_t half  = mant >> shift;
        uint32_t rem   = mant & ((1u << shift) - 1u);
        uint32_t mid   = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (half & 1u))) half++;
        return (g4_fp16)(sign | half);
    }
    uint32_t mant = em + 0xfffu + ((em >> 13) & 1u); // round to nearest even
    return (g4_fp16)(sign | ((mant - 0x38000000u) >> 13));
}

static inline float g4_bf16_to_fp32(g4_bf16 b) {
    uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f;
}
static inline g4_bf16 g4_fp32_to_bf16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    if ((x & 0x7fffffffu) > 0x7f800000u) return (g4_bf16)((x >> 16) | 64u); // nan
    return (g4_bf16)((x + (0x7fffu + ((x >> 16) & 1u))) >> 16);
}

void g4_init_tables(void);

// ggml-compatible gelu (GGML_GELU_FP16 path): clamp at +-10, fp16 LUT roundtrip
static inline float g4_gelu(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >=  10.0f) return x;
    return g4_fp16_to_fp32(g4_gelu_fp16_table[g4_fp32_to_fp16(x)]);
}

// ----------------------------------------------------------- ggml types -----
enum g4_type {
    G4_F32   = 0,
    G4_F16   = 1,
    G4_Q4_0  = 2,
    G4_Q8_0  = 8,
    G4_Q2_K  = 10,
    G4_Q3_K  = 11,
    G4_Q4_K  = 12,
    G4_Q5_K  = 13,
    G4_Q6_K  = 14,
    G4_BF16  = 30,
    G4_TQ2_0 = 35,
    G4_I8    = 24, G4_I16 = 25, G4_I32 = 26, G4_I64 = 27, G4_F64 = 28,
    G4_TYPE_COUNT = 40
};

const char * g4_type_name(int t);
int    g4_type_blck(int t);       // elements per block
size_t g4_type_size(int t);       // bytes per block
size_t g4_row_size(int t, int64_t n);

// ------------------------------------------------------------ quant blocks --
#define QK4_0 32
#define QK8_0 32
#define QK_K  256
#define K_SCALE_SIZE 12

#pragma pack(push, 1)
typedef struct { g4_fp16 d; uint8_t qs[QK4_0/2]; } block_q4_0;            // 18
typedef struct { g4_fp16 d; int8_t  qs[QK8_0];   } block_q8_0;            // 34
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; g4_fp16 d; } block_q6_K; // 210
typedef struct { uint8_t qs[QK_K/4]; g4_fp16 d; } block_tq2_0;            // 66
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K; // 292
// K-quant super-blocks (256 weights, ggml layout from ref/ggml-common.h)
typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; g4_fp16 d; g4_fp16 dmin; } block_q2_K;       // 84
typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; g4_fp16 d; } block_q3_K;   // 110
typedef struct { g4_fp16 d; g4_fp16 dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;  // 144
typedef struct { g4_fp16 d; g4_fp16 dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K; // 176
#pragma pack(pop)

// activation quantization
void g4_quantize_q8_0(const float * x, block_q8_0 * y, int64_t n);
void g4_quantize_q8_K(const float * x, block_q8_K * y, int64_t n);
void g4_quantize_q4_0(const float * x, block_q4_0 * y, int64_t n);   // for Q4_0 KV cache
void g4_quantize_q2_K(const float * x, block_q2_K * y, int64_t k);   // JIT requant targets
void g4_quantize_q3_K(const float * x, block_q3_K * y, int64_t k);

// dequantize one row (for token embedding gather)
void g4_dequant_row(int type, const void * x, float * y, int64_t n);

// AVX-512-VNNI dot path toggle (q8_0/q4_0 kernels). On by default when compiled
// with AVX-512; g4_set_avx512(0) forces the AVX2 path (for A/B timing).
extern int g4_avx512;
void g4_set_avx512(int on);
int  g4_avx512_available(void);

// dot products: w = weight row (quantized), a = quantized activation, n = elements
float g4_dot_q4_0_q8_0 (const void * w, const void * a, int64_t n);
float g4_dot_q8_0_q8_0 (const void * w, const void * a, int64_t n);
float g4_dot_q6_K_q8_K (const void * w, const void * a, int64_t n);
float g4_dot_tq2_0_q8_K(const void * w, const void * a, int64_t n);
float g4_dot_q2_K_q8_K (const void * w, const void * a, int64_t n);
float g4_dot_q3_K_q8_K (const void * w, const void * a, int64_t n);
float g4_dot_q4_K_q8_K (const void * w, const void * a, int64_t n);
float g4_dot_q5_K_q8_K (const void * w, const void * a, int64_t n);
float g4_dot_f16       (const void * w, const g4_fp16 * a, int64_t n);   // f16 row x f16 act
float g4_dot_bf16      (const void * w, const float * a, int64_t n);     // bf16 row x f32 act

// 4-token variants: unpack weight blocks once, dot against 4 activations
// (prefill speedup; out[i] must equal the 1-token dot exactly)
void g4_dot_q4_0_q8_0_x4 (const void * w, const void * const a[4], int64_t n, float out[4]);
void g4_dot_q8_0_q8_0_x4 (const void * w, const void * const a[4], int64_t n, float out[4]);
void g4_dot_q6_K_q8_K_x4 (const void * w, const void * const a[4], int64_t n, float out[4]);
void g4_dot_tq2_0_q8_K_x4(const void * w, const void * const a[4], int64_t n, float out[4]);
void g4_dot_q2_K_q8_K_x4 (const void * w, const void * const a[4], int64_t n, float out[4]);
void g4_dot_q3_K_q8_K_x4 (const void * w, const void * const a[4], int64_t n, float out[4]);
void g4_dot_q4_K_q8_K_x4 (const void * w, const void * const a[4], int64_t n, float out[4]);
void g4_dot_q5_K_q8_K_x4 (const void * w, const void * const a[4], int64_t n, float out[4]);

// scalar reference versions (selftest oracle)
float g4_dot_q4_0_q8_0_ref (const void * w, const void * a, int64_t n);
float g4_dot_q8_0_q8_0_ref (const void * w, const void * a, int64_t n);
float g4_dot_q6_K_q8_K_ref (const void * w, const void * a, int64_t n);
float g4_dot_tq2_0_q8_K_ref(const void * w, const void * a, int64_t n);
float g4_dot_q2_K_q8_K_ref (const void * w, const void * a, int64_t n);
float g4_dot_q3_K_q8_K_ref (const void * w, const void * a, int64_t n);
float g4_dot_q4_K_q8_K_ref (const void * w, const void * a, int64_t n);
float g4_dot_q5_K_q8_K_ref (const void * w, const void * a, int64_t n);

int g4_selftest_quants(void);

// ---------------------------------------------------------------- gguf ------
typedef struct {
    const char * ptr;   // not null-terminated (points into mmap)
    uint32_t     len;
} g4_str;

typedef struct {
    char     name[80];
    int      type;
    int      ndim;
    int64_t  ne[4];
    uint64_t offset;        // relative to data section
    const void * data;      // resolved pointer into mapping
} g4_tensor;

typedef struct {
    // file mapping
    void *   map_base;
    uint64_t map_size;
    void *   h_file, * h_map;    // win32 handles

    uint32_t version;
    uint64_t n_tensors, n_kv;
    uint64_t data_start;
    uint64_t kv_end;            // byte offset where the metadata-KV section ends (for the writer)
    uint64_t ff_len_off;        // file offset of the qwen35.feed_forward_length scalar value (0 = none)
    int      ff_len_vt;         // its gguf value-type id (for byte width) — patched on prune export
    uint32_t alignment;

    g4_tensor * tensors;

    // ---- parsed hparams (gemma4) ----
    char  arch[32];
    int   n_layer, n_embd, n_vocab, n_head, n_head_kv;
    int   n_embd_per_layer;
    int   n_ff[G4_MAX_LAYERS];
    int   head_dim_full, head_dim_swa;       // K==V dims
    int   n_rot_full, n_rot_swa;
    float rope_base_full, rope_base_swa;
    float rms_eps;
    float final_softcap;
    int   n_swa;                              // sliding window
    int   n_kv_shared;                        // shared kv layers (from end)
    int   n_layer_kv;                         // = n_layer - n_kv_shared
    uint8_t is_swa[G4_MAX_LAYERS];
    int   n_ctx_train;

    // gemma4-assistant (MTP draft head) extras
    bool  is_assistant;                       // arch == gemma4-assistant
    int   n_embd_out;                         // backbone embd dim the draft projects to/from (1536)
    int   n_nextn;                            // nextn_predict_layers (draft length)

    // qwen35 (hybrid Gated-DeltaNet linear-attention + gated full-attention)
    bool  is_qwen35;
    int   n_embd_head_k, n_embd_head_v;       // attention head dim (256); n_head/n_head_kv shared
    int   ssm_d_conv, ssm_d_inner, ssm_d_state, ssm_dt_rank, ssm_n_group;
    int   full_attn_interval;                 // every Nth layer is full attention (default 4)
    int   rope_sections[4];                   // mRoPE section split
    uint8_t is_recr[G4_MAX_LAYERS];           // 1 = SSM (linear-attention) layer, 0 = full attention
    int   tok_pre;                            // tokenizer pre-type (0=gemma4, 1=qwen35/gpt2)

    // ---- tokenizer ----
    int      tok_n;                 // n tokens
    g4_str * tok_text;
    const int32_t * tok_type;       // 1=normal 2=unknown 3=control 4=user_def 6=byte
    int      merges_n;
    g4_str * merges;
    int   bos_id, eos_id, eot_id, unk_id, pad_id;
    bool  add_bos;

    // sampling defaults from metadata (-1/nan = absent)
    float meta_temp, meta_top_p;
    int   meta_top_k;

    g4_str chat_template;
} g4_model_file;

extern int g4_mmap_cow;   // 1 = COW view (Vulkan-importable); 0 = read-only (no commit charge)
int  g4_gguf_open(g4_model_file * mf, const char * path, bool verbose);
void g4_gguf_close(g4_model_file * mf);
int  g4_gguf_write(const g4_model_file * mf, const char * path);   // export current state (pruning)
const g4_tensor * g4_find_tensor(const g4_model_file * mf, const char * name);
void g4_gguf_dump(const g4_model_file * mf);

// --------------------------------------------------------------- model ------
typedef struct {
    const g4_tensor * attn_norm, * wq, * wk, * wv, * wo;
    const g4_tensor * q_norm, * k_norm, * post_attn_norm;
    const g4_tensor * ffn_norm, * gate, * up, * down, * post_ffw_norm;
    const g4_tensor * inp_gate, * pl_proj, * pl_post_norm, * out_scale;
} g4_layer;

typedef struct {
    g4_model_file * mf;

    const g4_tensor * tok_embd, * per_layer_tok_embd, * per_layer_model_proj;
    const g4_tensor * per_layer_proj_norm, * output_norm, * rope_freqs;
    g4_layer layers[G4_MAX_LAYERS];

    int n_ctx;          // context window for global layers
    int n_threads;

    // KV cache: producers only (il < n_layer_kv)
    // SWA layers: ring of G4_SWA_RING entries; global: n_ctx entries
    // Stored as kv_type_k/kv_type_v (G4_F16 default, or G4_Q8_0 / G4_Q4_0).
    uint8_t * kc[G4_MAX_LAYERS];
    uint8_t * vc[G4_MAX_LAYERS];
    int kv_type_k, kv_type_v;

    // batch scratch (token-major, B = G4_BATCH_MAX rows)
    float * x;          // [B][n_embd] hidden
    float * xb;         // [B][n_embd]
    float * q;          // [B][8*512]
    float * k;          // [B][512]
    float * v;          // [B][512]
    float * ffh;        // [B][max_ff]
    float * ffh2;       // [B][max_ff]
    float * pl;         // [B][n_layer*256]
    float * plp;        // [B][n_layer*256]
    float * plg;        // [B][256] per-layer gate
    float * attbuf;     // [n_threads][n_ctx] score scratch
    float * rc_full;    // [B][512] rope cos/sin cache
    float * rc_swa;     // [B][256]
    float * bf32;       // [B][n_embd] bf16-rounded activation
    float * logits;     // [n_vocab]
    g4_fp16 * wf16;     // [B][qdim] f16 activations
    void  * qa;         // [B][qstride] quantized activations
    size_t  qstride;    // bytes per token in qa

    int pos;            // next position to fill
} g4_ctx;

int  g4_ctx_init(g4_ctx * c, g4_model_file * mf, int n_ctx, int n_threads,
                 int kv_type_k, int kv_type_v);   // KV types: G4_F16 | G4_Q8_0 | G4_Q4_0
void g4_ctx_free(g4_ctx * c);
// run tokens[0..n) at positions pos0..pos0+n-1; returns logits of last token
// (NULL result members unread when want_logits=false)
float * g4_forward_batch(g4_ctx * c, const int * tokens, int n, int pos0, bool want_logits);
static inline float * g4_forward(g4_ctx * c, int token, int pos) {
    return g4_forward_batch(c, &token, 1, pos, true);
}
// like g4_forward_batch but emits logits AND the post-output_norm hidden ("h_nextn")
// for EVERY position. logits_out:[n][n_vocab], hnext_out:[n][n_embd]. (MTP verify)
void g4_forward_batch_heads(g4_ctx * c, const int * tokens, int n, int pos0,
                            float * logits_out, float * hnext_out);

// ----------------------------------------------------- MTP draft (assistant) -
// gemma4-assistant multi-token-prediction head: a tiny transformer that drafts
// the target's next tokens, reusing the target's KV cache. Used for speculative
// decoding (provably greedy-exact: every draft is verified by the target).
typedef struct {
    g4_model_file * mf;             // draft gguf
    g4_ctx * tgt;                   // target context (KV cache + tok_embd source)
    const g4_tensor * tok_embd;     // draft tied output head [n_embd_d, n_vocab]
    const g4_tensor * pre_proj;     // nextn.pre_projection  [2*n_embd_out, n_embd_d]
    const g4_tensor * post_proj;    // nextn.post_projection [n_embd_d, n_embd_out]
    const g4_tensor * output_norm;  // [n_embd_d]
    const g4_tensor * rope_freqs;   // [n_rot_full/2]
    g4_layer layers[G4_MAX_LAYERS];
    int n_embd_d;                   // draft hidden dim (256)
    int n_embd_out;                 // backbone hidden dim (1536)
    float * xh, * cur, * xb, * q, * ffh, * ffh2, * rc_full, * rc_swa, * logits;
} g4_mtp;

int  g4_mtp_init(g4_mtp * d, g4_model_file * mf_draft, g4_ctx * tgt);
void g4_mtp_free(g4_mtp * d);

// ----------------------------------------------------- qwen35 hybrid model ---
// Qwen3.5: 32 layers, 24 Gated-DeltaNet (linear-attn SSM) + 8 gated full-attn.
typedef struct {
    const g4_tensor * attn_norm, * attn_post_norm;
    const g4_tensor * wq, * wk, * wv, * wo, * q_norm, * k_norm;            // full-attn
    const g4_tensor * wqkv, * wqkv_gate, * ssm_conv1d, * ssm_dt, * ssm_a;  // ssm
    const g4_tensor * ssm_beta, * ssm_alpha, * ssm_norm, * ssm_out;
    const g4_tensor * ffn_gate, * ffn_up, * ffn_down;
    bool is_recr;
} qwen_layer;

typedef struct {
    g4_model_file * mf;
    g4_ctx mm;                 // scratch-only g4_ctx for matmul_b (qa/wf16/bf32)
    const g4_tensor * tok_embd, * output_norm, * output;
    qwen_layer layers[G4_MAX_LAYERS];
    int n_ctx, n_threads;
    int head_dim, n_head, n_head_kv;
    int s_v, h_v, h_k, key_dim, val_dim, conv_dim, n_ff;
    // attention KV cache (full-attn layers only): [n_ctx][n_head_kv*head_dim] f32
    float * kc[G4_MAX_LAYERS];
    float * vc[G4_MAX_LAYERS];
    // SSM state (linear-attn layers only)
    float * conv_state[G4_MAX_LAYERS];  // [conv_dim][d_conv-1]
    float * ssm_state[G4_MAX_LAYERS];   // [s_v*s_v*h_v]
    // speculative-decode checkpoints: per-token recurrent state captured during a
    // verify batch so rejected drafts can be rolled back without a replay forward.
    float * ck_conv[G4_MAX_LAYERS];     // [ck_cap][conv_dim*(d_conv-1)]
    float * ck_ssm[G4_MAX_LAYERS];      // [ck_cap][s_v*s_v*h_v]
    size_t  ck_stride_conv, ck_stride_ssm;
    int     ck_cap;                     // number of checkpoint slots allocated
    int     ck_on;                      // when set, scan/conv kernels write checkpoints
    int     ck_base;                    // slot offset for checkpoint writes (per-token drafting)
    // scratch (single token)
    float * x, * xb, * cur, * tmp, * qkv, * convo, * qfull, * kb, * vb;
    float * attno, * z, * alpha, * beta, * gdec, * ffh, * ffh2, * att, * rc, * logits;
    // FFN neuron importance capture (pruning M0): accumulate sum|silu(gate)*up| per
    // intermediate unit over a calibration corpus. NULL/0 = capture off.
    float * ffn_imp[G4_MAX_LAYERS];     // [n_ff] running sum of |activation| per neuron
    float * ffn_mean[G4_MAX_LAYERS];    // [n_ff] running sum of signed activation (for bias comp)
    long    ffn_imp_count;              // tokens accumulated
    int     ffn_capture;               // 1 = accumulate during qwen_run_layers
    // mean/bias compensation (pruning M2): per-layer additive correction applied after
    // ffn_down = sum over dropped neurons of E[a_j] * W_down[:,j]. NULL = none.
    float * ffn_bias[G4_MAX_LAYERS];    // [n_embd]
    // blockwise least-squares compensation (pruning M3): per-layer block activation
    // covariances [n_blocks][block*block] captured over the corpus when cov_block>0.
    float * ffn_cov[G4_MAX_LAYERS];
    int     cov_block;                  // block size (0 = don't capture covariances)
    // zeroth-order (MeZO) LoRA fine-tuning: a low-rank adapter on ffn_down, trained
    // forward-only by random perturbation. ffn_down output += scale * B*(A*ffh).
    float * lora_a[G4_MAX_LAYERS];      // [rank][n_ff]   (init small gaussian)
    float * lora_b[G4_MAX_LAYERS];      // [n_embd][rank] (init 0 -> initial delta = 0)
    int     lora_rank, lora_l0, lora_l1;   // 0 rank = off; adapter on layers [l0,l1)
    float   lora_scale;
} qwen_ctx;

int     g4_qwen_init(qwen_ctx * c, g4_model_file * mf, int n_ctx, int n_threads);
void    g4_qwen_free(qwen_ctx * c);
void    g4_qwen_reset(qwen_ctx * c);   // zero recurrent conv/ssm state (new conversation)
float * g4_qwen_forward(qwen_ctx * c, int token, int pos);  // logits for token at pos
// batched (prefill): tokens[0..nb) at pos0..pos0+nb-1; logits of last token (or NULL)
float * g4_qwen_forward_batch(qwen_ctx * c, const int * tokens, int nb, int pos0, bool want_logits);
// per-position logits [nb][n_vocab] without spec checkpoints (perplexity / eval)
float * g4_qwen_forward_logits(qwen_ctx * c, const int * tokens, int nb, int pos0, float * logits_out);
// FFN neuron-importance capture (pruning M0): alloc accumulators + enable / disable
int     g4_qwen_ffn_capture_begin(qwen_ctx * c);
void    g4_qwen_ffn_capture_end(qwen_ctx * c);
// FFN-width prune (pruning M1/M2): keep top `keep` fraction of neurons; ffn_down ->
// down_quant. compensate=1 adds the dropped neurons' mean contribution as a per-layer bias.
int     g4_qwen_prune_ffn(qwen_ctx * c, float keep, int down_quant, int compensate);
// zeroth-order LoRA fine-tuning: allocate the adapter; perturb adds coef*z(seed) to all
// adapter params (z regenerated from the seed, so no gradient/optimizer storage).
int     g4_qwen_lora_init(qwen_ctx * c, int rank, int l0, int l1, float scale, uint64_t seed);
void    g4_qwen_lora_perturb(qwen_ctx * c, float coef, uint64_t seed);
// fold the trained adapter into ffn_down (W_down += scale*B*A) and clear it, so the
// fine-tune persists in an exported GGUF. ffn_down types we can't re-encode (q4_K/
// q5_K/q6_K) are written as Q8_0 (near-lossless). Returns 0 on success.
int     g4_qwen_lora_merge(qwen_ctx * c);
// speculative decoding (prompt-lookup): allocate per-token checkpoint buffers for up
// to kmax+1 batch positions; forward_spec runs a verify batch writing those checkpoints
// and returns per-position logits [nb][n_vocab]; state_restore rolls the recurrent
// conv/SSM state back to the checkpoint after batch index `idx` (0-based).
int     g4_qwen_spec_enable(qwen_ctx * c, int kmax);
float * g4_qwen_forward_spec(qwen_ctx * c, const int * tokens, int nb, int pos0, float * logits_out);
void    g4_qwen_state_restore(qwen_ctx * c, int idx);
// self-speculative draft: build `draft` as a low-bit (draft_quant) in-RAM requant of
// `mf_target`'s weights, same architecture. The draft proposes tokens cheaply; the
// full-precision target verifies them. Returns 0 on success.
int     g4_qwen_make_draft(qwen_ctx * draft, g4_model_file * mf_target, int draft_quant, int n_ctx, int n_threads);
// One draft step: given input `token` and backbone hidden h_in[n_embd_out] for a
// query at position `qpos`, return the drafted next-token id and write the chained
// hidden h_out[n_embd_out]. `kv_last` = target's highest cached position.
int  g4_mtp_step(g4_mtp * d, int token, const float * h_in, int qpos, int kv_last, float * h_out);
// threads
void g4_threads_start(int n);
// Just-in-time requantization: re-quantize large matmul weights to `target`
// (G4_Q4_0 or G4_Q8_0) in RAM at load time for less weight bandwidth (faster,
// rougher decode). Repoints tensor data to heap buffers. Call after gguf open,
// before ctx init. Returns the new total requantized weight size in bytes.
size_t g4_jit_requant(g4_model_file * mf, int target, int n_threads, bool verbose);
void g4_threads_stop(void);
// parallel-for: fn(start, end, userdata, thread_idx)
typedef void (*g4_pf_fn)(int64_t r0, int64_t r1, void * ud, int tid);
void g4_parallel_for(int64_t n, g4_pf_fn fn, void * ud);

// ------------------------------------------------------------- tokenizer ----
typedef struct g4_tok g4_tok;
g4_tok * g4_tok_init(g4_model_file * mf);
void     g4_tok_free(g4_tok * t);
// encode text -> ids. parse_special: match control tokens in text.
// returns count; ids buffer must hold cap entries
int  g4_tok_encode(g4_tok * t, const char * text, int * ids, int cap,
                   bool add_bos, bool parse_special);
// decode one token id to bytes; returns len, writes to buf (>=64 bytes)
int  g4_tok_decode(g4_tok * t, int id, char * buf, bool render_special);
bool g4_tok_is_eog(g4_tok * t, int id);
int  g4_tok_find(g4_tok * t, const char * text);   // exact text -> id, -1 if absent

// HTTP server — ctx is g4_ctx* (is_qwen=0) or qwen_ctx* (is_qwen=1).
// repeat_penalty<0 = per-arch default; no_repeat_ngram/ban_str supply anti-loop defaults.
int g4_server_run(void * ctx, int is_qwen, g4_tok * tok, g4_model_file * mf,
                  const char * host, int port, const char * model_name,
                  float repeat_penalty, int no_repeat_ngram, const char * ban_str);

// -------------------------------------------------------------- sampler -----
typedef struct {
    float temp;          // 0 = greedy
    int   top_k;
    float top_p;
    float repeat_penalty;
    int   repeat_last_n;
    float freq_penalty;
    float presence_penalty;
    int   no_repeat_ngram;  // 0/1 = off; otherwise block any repeat of an n-gram of this length
    uint64_t seed;
    // state
    uint64_t rng[4];
    int  * recent;       // ring of recent tokens
    int    recent_cap, recent_n, recent_head;
    int  * cnt;          // per-vocab counts for freq penalty (lazy alloc)
    int  * ban_ids;      // suppressed token ids (logit -> -inf); NULL = none
    int    n_ban, ban_cap;
    int    n_vocab;
} g4_sampler;

void g4_sampler_init(g4_sampler * s, int n_vocab);
void g4_sampler_reset(g4_sampler * s);   // clear recent-token history/penalty state
void g4_sampler_ban(g4_sampler * s, int id);   // permanently suppress a token id
void g4_sampler_free(g4_sampler * s);
void g4_sampler_accept(g4_sampler * s, int id);
int  g4_sample(g4_sampler * s, float * logits);

// ---------------------------------------------------------------- misc ------
double g4_time_ms(void);

#ifdef __cplusplus
}
#endif
#endif // G4RUN_H
