// model.c — gemma4 batched forward pass + KV cache + thread pool
// Graph order and numerics per ../SPEC.md (llama.cpp @ daf6bc9f2 parity).
// Batch design: weights are streamed from DRAM once per batch — the token loop
// is INSIDE the row loop, so a weight row stays L1-hot across batch tokens.
#include "g4run.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#if defined(__AVX2__)
#include <immintrin.h>

// f32 SIMD primitives for the qwen35 scan/attention inner loops (n a multiple of 8 in the
// hot paths: sv=128, hd=256). FMA + tree reduction — fast but reorders the sum vs scalar.
static inline float vdot(const float * a, const float * b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo)); lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void vmadd(float * dst, const float * src, float sc, int n) {  // dst += sc*src
    const __m256 vs = _mm256_set1_ps(sc);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(vs, _mm256_loadu_ps(src+i), _mm256_loadu_ps(dst+i)));
    for (; i < n; i++) dst[i] += sc * src[i];
}
static inline void vscale(float * dst, float sc, int n) {                     // dst *= sc
    const __m256 vs = _mm256_set1_ps(sc);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(dst+i, _mm256_mul_ps(vs, _mm256_loadu_ps(dst+i)));
    for (; i < n; i++) dst[i] *= sc;
}
#else
static inline float vdot(const float * a, const float * b, int n) { float s=0; for (int i=0;i<n;i++) s+=a[i]*b[i]; return s; }
static inline void vmadd(float * dst, const float * src, float sc, int n) { for (int i=0;i<n;i++) dst[i]+=sc*src[i]; }
static inline void vscale(float * dst, float sc, int n) { for (int i=0;i<n;i++) dst[i]*=sc; }
#endif

// ------------------------------------------------------------ thread pool ---

typedef struct {
    volatile LONG  seq;
    volatile LONG  done;
    g4_pf_fn       fn;
    void *         ud;
    int64_t        total;
    int            nth;
} pool_t;

static pool_t  pool;
static HANDLE  pool_threads[G4_MAX_THREADS];
static int     pool_n = 0;
static volatile LONG pool_quit = 0;
static volatile LONG pool_sleepers = 0;

static void pool_chunk(int idx) {
    int64_t per = (pool.total + pool.nth - 1) / pool.nth;
    int64_t r0 = per * idx;
    int64_t r1 = r0 + per;
    if (r1 > pool.total) r1 = pool.total;
    if (r0 < r1) pool.fn(r0, r1, pool.ud, idx);
}

// Hybrid wait: spin briefly so back-to-back matvec launches stay low-latency,
// then park in the kernel (WaitOnAddress) so an idle process uses ~0% CPU
// (e.g. chat mode sitting at the input prompt).
#define POOL_SPINS_BEFORE_SLEEP 20000

static DWORD WINAPI pool_worker(LPVOID arg) {
    int idx = (int)(intptr_t)arg;
    LONG seen = 0;
    for (;;) {
        int spins = 0;
        while (pool.seq == seen && !pool_quit) {
            _mm_pause();
            if (++spins > POOL_SPINS_BEFORE_SLEEP) {
                LONG expected = seen;
                InterlockedIncrement(&pool_sleepers);
                // no lost-wakeup race: WaitOnAddress returns immediately
                // if pool.seq no longer equals `expected`
                WaitOnAddress((volatile VOID *)&pool.seq, &expected, sizeof(LONG), INFINITE);
                InterlockedDecrement(&pool_sleepers);
                spins = 0;
            }
        }
        if (pool_quit) return 0;
        seen = pool.seq;
        pool_chunk(idx);
        InterlockedIncrement(&pool.done);
    }
}

static int pool_running = 0;

void g4_threads_start(int n) {
    if (pool_running) return;        // idempotent: may be started early (e.g. for JIT requant)
    if (n < 1) n = 1;
    if (n > G4_MAX_THREADS) n = G4_MAX_THREADS;
    pool.nth = n;
    pool_n = n - 1;
    for (int i = 0; i < pool_n; i++)
        pool_threads[i] = CreateThread(NULL, 0, pool_worker, (LPVOID)(intptr_t)(i + 1), 0, NULL);
    pool_running = 1;
}

void g4_threads_stop(void) {
    if (!pool_running) return;
    pool_quit = 1;
    WakeByAddressAll((PVOID)&pool.seq);   // rouse parked workers so they see quit
    for (int i = 0; i < pool_n; i++) { WaitForSingleObject(pool_threads[i], 1000); CloseHandle(pool_threads[i]); }
    pool_n = 0; pool_quit = 0; pool_running = 0;
}

void g4_parallel_for(int64_t total, g4_pf_fn fn, void * ud) {
    if (pool_n == 0 || total < 4) { fn(0, total, ud, 0); return; }
    pool.fn = fn; pool.ud = ud; pool.total = total;
    pool.done = 0;
    InterlockedIncrement(&pool.seq);
    if (pool_sleepers) WakeByAddressAll((PVOID)&pool.seq);
    pool_chunk(0);
    while (pool.done != pool_n) _mm_pause();
}

// ------------------------------------------------- activation preparation ---

// activation class for a weight type: what the activation must be converted to
enum { AC_Q8_0, AC_Q8_K, AC_F16, AC_BF16, AC_F32 };

static int act_class(int wtype) {
    switch (wtype) {
        case G4_Q4_0: case G4_Q8_0: return AC_Q8_0;
        case G4_Q6_K: case G4_TQ2_0:
        case G4_Q2_K: case G4_Q3_K: case G4_Q4_K: case G4_Q5_K: return AC_Q8_K;
        case G4_F16:  return AC_F16;
        case G4_BF16: return AC_BF16;
        default:      return AC_F32;
    }
}

// convert/quantize nb token rows of x[nb][ncols]; returns ptr + stride
static const void * prep_act(g4_ctx * c, int cls, const float * x, int64_t ncols,
                             int nb, size_t * stride) {
    switch (cls) {
        case AC_Q8_0: {
            size_t st = (ncols/QK8_0) * sizeof(block_q8_0);
            st = (st + 63) & ~(size_t)63;
            for (int b = 0; b < nb; b++)
                g4_quantize_q8_0(x + (size_t)b*ncols, (block_q8_0 *)((uint8_t *)c->qa + b*st), ncols);
            *stride = st;
            return c->qa;
        }
        case AC_Q8_K: {
            size_t st = (ncols/QK_K) * sizeof(block_q8_K);
            st = (st + 63) & ~(size_t)63;
            for (int b = 0; b < nb; b++)
                g4_quantize_q8_K(x + (size_t)b*ncols, (block_q8_K *)((uint8_t *)c->qa + b*st), ncols);
            *stride = st;
            return c->qa;
        }
        case AC_F16: {
            for (int b = 0; b < nb; b++) {
                const float * src = x + (size_t)b*ncols;
                g4_fp16 * dst = c->wf16 + (size_t)b*ncols;
                for (int64_t i = 0; i < ncols; i++) dst[i] = g4_fp32_to_fp16(src[i]);
            }
            *stride = ncols * sizeof(g4_fp16);
            return c->wf16;
        }
        case AC_BF16: {
            // ggml rounds the activation to bf16 (vec_dot_type), then the
            // no-AVX512BF16 fallback widens both sides to f32. Mirror that.
            for (int b = 0; b < nb; b++) {
                const float * src = x + (size_t)b*ncols;
                float * dst = c->bf32 + (size_t)b*ncols;
                for (int64_t i = 0; i < ncols; i++) dst[i] = g4_bf16_to_fp32(g4_fp32_to_bf16(src[i]));
            }
            *stride = ncols * sizeof(float);
            return c->bf32;
        }
        default:
            *stride = ncols * sizeof(float);
            return x;
    }
}

// --------------------------------------------------------------- matmul -----

typedef struct {
    int type;
    const uint8_t * W;
    size_t row_bytes;
    const uint8_t * act;
    size_t act_stride;
    float * Y;              // token-major [nb][nrows]
    int64_t ncols, nrows;
    int nb;
} mm_ud;

static void mm_kernel(int64_t r0, int64_t r1, void * p, int tid) {
    (void)tid;
    mm_ud * u = p;
    const int nb = u->nb;
    const int64_t nrows = u->nrows;
    // x4 path: weight block unpack shared across 4 tokens (bit-identical to x1)
    #define ROWLOOP4(DOTFN, DOTFN4, ACTT) \
        for (int64_t r = r0; r < r1; r++) { \
            const void * row = u->W + r*u->row_bytes; \
            int b = 0; \
            for (; b + 4 <= nb; b += 4) { \
                const void * ap[4] = { u->act + (size_t)(b+0)*u->act_stride, u->act + (size_t)(b+1)*u->act_stride, \
                                       u->act + (size_t)(b+2)*u->act_stride, u->act + (size_t)(b+3)*u->act_stride }; \
                float o4[4]; \
                DOTFN4(row, ap, u->ncols, o4); \
                u->Y[(size_t)(b+0)*nrows + r] = o4[0]; u->Y[(size_t)(b+1)*nrows + r] = o4[1]; \
                u->Y[(size_t)(b+2)*nrows + r] = o4[2]; u->Y[(size_t)(b+3)*nrows + r] = o4[3]; \
            } \
            for (; b < nb; b++) \
                u->Y[(size_t)b*nrows + r] = DOTFN(row, (ACTT)(u->act + (size_t)b*u->act_stride), u->ncols); \
        }
    #define ROWLOOP(DOTFN, ACTT) \
        for (int64_t r = r0; r < r1; r++) { \
            const void * row = u->W + r*u->row_bytes; \
            for (int b = 0; b < nb; b++) \
                u->Y[(size_t)b*nrows + r] = DOTFN(row, (ACTT)(u->act + (size_t)b*u->act_stride), u->ncols); \
        }
    switch (u->type) {
        case G4_Q4_0:  ROWLOOP4(g4_dot_q4_0_q8_0,  g4_dot_q4_0_q8_0_x4,  const void *) break;
        case G4_Q8_0:  ROWLOOP4(g4_dot_q8_0_q8_0,  g4_dot_q8_0_q8_0_x4,  const void *) break;
        case G4_Q6_K:  ROWLOOP4(g4_dot_q6_K_q8_K,  g4_dot_q6_K_q8_K_x4,  const void *) break;
        case G4_TQ2_0: ROWLOOP4(g4_dot_tq2_0_q8_K, g4_dot_tq2_0_q8_K_x4, const void *) break;
        case G4_Q2_K:  ROWLOOP4(g4_dot_q2_K_q8_K,  g4_dot_q2_K_q8_K_x4,  const void *) break;
        case G4_Q3_K:  ROWLOOP4(g4_dot_q3_K_q8_K,  g4_dot_q3_K_q8_K_x4,  const void *) break;
        case G4_Q4_K:  ROWLOOP4(g4_dot_q4_K_q8_K,  g4_dot_q4_K_q8_K_x4,  const void *) break;
        case G4_Q5_K:  ROWLOOP4(g4_dot_q5_K_q8_K,  g4_dot_q5_K_q8_K_x4,  const void *) break;
        case G4_F16:   ROWLOOP(g4_dot_f16,        const g4_fp16 *) break;
        case G4_BF16:  ROWLOOP(g4_dot_bf16,       const float *) break;
        case G4_F32: {
            for (int64_t r = r0; r < r1; r++) {
                const float * row = (const float *)(u->W + r*u->row_bytes);
                for (int b = 0; b < nb; b++) {
                    const float * xx = (const float *)(u->act + (size_t)b*u->act_stride);
                    float s = 0;
                    for (int64_t i = 0; i < u->ncols; i++) s += row[i] * xx[i];
                    u->Y[(size_t)b*nrows + r] = s;
                }
            }
        } break;
    }
    #undef ROWLOOP
    #undef ROWLOOP4
}

// Y[b][r] = dot(W[r,:], act[b]); pre = prepared activations (or NULL to prepare here)
static void matmul_b(g4_ctx * c, int wtype, const void * W, const float * x_f32,
                     const void * pre, size_t pre_stride,
                     float * Y, int64_t ncols, int64_t nrows, int nb) {
    mm_ud u;
    u.type = wtype;
    u.W = (const uint8_t *)W;
    u.row_bytes = g4_row_size(wtype, ncols);
    if (!pre) pre = prep_act(c, act_class(wtype), x_f32, ncols, nb, &pre_stride);
    u.act = (const uint8_t *)pre;
    u.act_stride = pre_stride;
    u.Y = Y;
    u.ncols = ncols; u.nrows = nrows; u.nb = nb;
    g4_parallel_for(nrows, mm_kernel, &u);
}

// ---- just-in-time requantization -------------------------------------------
// Re-quantize the large matmul weights to a smaller type at load time to cut
// weight bandwidth (faster, bandwidth-bound decode) at some quality cost. Each
// tensor carries its own type, so the matmul kernels pick up the new format
// transparently. Only block-aligned 2-D weights that actually shrink are touched;
// norms / conv / 1-D params (read as raw f32) are left alone.
typedef struct { const g4_tensor * t; int target; size_t old_row, new_row; uint8_t * dst; } requant_ud;

static void requant_kernel(int64_t r0, int64_t r1, void * p, int tid) {
    (void)tid;
    requant_ud * u = p;
    const int64_t cols = u->t->ne[0];
    float * tmp = malloc((size_t)cols * sizeof(float));
    if (!tmp) return;
    const uint8_t * src = (const uint8_t *)u->t->data;
    for (int64_t r = r0; r < r1; r++) {
        g4_dequant_row(u->t->type, src + (size_t)r * u->old_row, tmp, cols);
        uint8_t * drow = u->dst + (size_t)r * u->new_row;
        switch (u->target) {
            case G4_Q8_0: g4_quantize_q8_0(tmp, (block_q8_0 *)drow, cols); break;
            case G4_Q2_K: g4_quantize_q2_K(tmp, (block_q2_K *)drow, cols); break;
            case G4_Q3_K: g4_quantize_q3_K(tmp, (block_q3_K *)drow, cols); break;
            default:      g4_quantize_q4_0(tmp, (block_q4_0 *)drow, cols); break;
        }
    }
    free(tmp);
}

size_t g4_jit_requant(g4_model_file * mf, int target, int n_threads, bool verbose) {
    g4_threads_start(n_threads);            // idempotent; ctx init reuses this pool
    double t0 = g4_time_ms();
    size_t old_tot = 0, new_tot = 0; int n_req = 0;
    for (uint64_t i = 0; i < mf->n_tensors; i++) {
        g4_tensor * t = &mf->tensors[i];
        if (t->ndim < 2) continue;          // skip norms / 1-D params (read as raw f32)
        // Keep embeddings at source precision: they are read per-token (lookup, not a
        // streamed matmul) and are quality-sensitive — rough quantization of gemma's
        // per-layer embeddings degenerates the model. This also matches standard quant
        // recipes (Q4_K_M etc. keep token_embd/output higher than the block weights).
        if (strstr(t->name, "embd") || strstr(t->name, "per_layer")) continue;
        const int64_t cols = t->ne[0];
        const int blk = (target == G4_Q4_0 || target == G4_Q8_0) ? 32 : QK_K;  // K-quants are 256-wide
        if (cols % blk != 0) continue;      // unaligned columns (e.g. ssm_conv1d cols=4) -> skip
        const size_t old_row = g4_row_size(t->type, cols);
        const size_t new_row = g4_row_size(target, cols);
        if (new_row >= old_row) continue;   // only requant if it actually shrinks
        int64_t nrows = t->ne[1];
        for (int d = 2; d < t->ndim; d++) nrows *= t->ne[d];
        uint8_t * dst = malloc((size_t)nrows * new_row);
        if (!dst) { fprintf(stderr, "g4: jit requant: alloc failed for %s\n", t->name); continue; }
        requant_ud u = { t, target, old_row, new_row, dst };
        g4_parallel_for(nrows, requant_kernel, &u);
        if (verbose) fprintf(stderr, "g4:   requant %-28s %s %lldx%lld\n", t->name,
                             g4_type_name(t->type), (long long)nrows, (long long)cols);
        t->data = dst;                      // abandon the mmap pages (heap buffer freed at exit)
        t->type = target;
        old_tot += (size_t)nrows * old_row; new_tot += (size_t)nrows * new_row; n_req++;
    }
    fprintf(stderr, "g4: JIT requant -> %s: %d tensors, %.2f GB -> %.2f GB in %.0f ms\n",
            g4_type_name(target), n_req, old_tot / 1e9, new_tot / 1e9, g4_time_ms() - t0);
    return new_tot;
}

// ----------------------------------------------------------------- norms ----

// ggml_rms_norm: double sum of squares, then y = x*scale*w
static void rms_norm(const float * x, const float * w, float * y, int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * x[i];
    const float mean  = (float)(sum / n);
    const float scale = 1.0f / sqrtf(mean + eps);
    if (w) for (int i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
    else   for (int i = 0; i < n; i++) y[i] = x[i] * scale;
}

// --------------------------------------------------------------- rope -------

static void rope_cache(float * cs, int n_dims, int pos, float freq_base, const float * ff) {
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float theta = (float)pos;
    for (int ic = 0; ic < n_dims/2; ic++) {
        const float t = ff ? theta / ff[ic] : theta;
        cs[2*ic + 0] = cosf(t);
        cs[2*ic + 1] = sinf(t);
        theta *= theta_scale;
    }
}
static void rope_apply(float * x, const float * cs, int n_dims) {
    const int half = n_dims / 2;
    for (int ic = 0; ic < half; ic++) {
        const float co = cs[2*ic + 0], si = cs[2*ic + 1];
        const float x0 = x[ic], x1 = x[ic + half];
        x[ic]        = x0*co - x1*si;
        x[ic + half] = x0*si + x1*co;
    }
}

// ------------------------------------------------------------- ctx init -----

static const g4_tensor * need(g4_model_file * mf, const char * fmt, int il) {
    char name[96];
    snprintf(name, sizeof(name), fmt, il);
    const g4_tensor * t = g4_find_tensor(mf, name);
    if (!t) { fprintf(stderr, "g4: missing tensor %s\n", name); exit(1); }
    return t;
}
static const g4_tensor * opt(g4_model_file * mf, const char * fmt, int il) {
    char name[96];
    snprintf(name, sizeof(name), fmt, il);
    return g4_find_tensor(mf, name);
}

int g4_ctx_init(g4_ctx * c, g4_model_file * mf, int n_ctx, int n_threads,
                int kv_type_k, int kv_type_v) {
    memset(c, 0, sizeof(*c));
    c->mf = mf;
    c->n_ctx = n_ctx;
    c->n_threads = n_threads;
    c->kv_type_k = kv_type_k;
    c->kv_type_v = kv_type_v;

    c->tok_embd             = need(mf, "token_embd.weight", 0);
    c->per_layer_tok_embd   = need(mf, "per_layer_token_embd.weight", 0);
    c->per_layer_model_proj = need(mf, "per_layer_model_proj.weight", 0);
    c->per_layer_proj_norm  = need(mf, "per_layer_proj_norm.weight", 0);
    c->output_norm          = need(mf, "output_norm.weight", 0);
    c->rope_freqs           = need(mf, "rope_freqs.weight", 0);

    for (int il = 0; il < mf->n_layer; il++) {
        g4_layer * L = &c->layers[il];
        L->attn_norm      = need(mf, "blk.%d.attn_norm.weight", il);
        L->wq             = need(mf, "blk.%d.attn_q.weight", il);
        L->wo             = need(mf, "blk.%d.attn_output.weight", il);
        L->q_norm         = need(mf, "blk.%d.attn_q_norm.weight", il);
        L->post_attn_norm = need(mf, "blk.%d.post_attention_norm.weight", il);
        L->ffn_norm       = need(mf, "blk.%d.ffn_norm.weight", il);
        L->gate           = need(mf, "blk.%d.ffn_gate.weight", il);
        L->up             = need(mf, "blk.%d.ffn_up.weight", il);
        L->down           = need(mf, "blk.%d.ffn_down.weight", il);
        L->post_ffw_norm  = need(mf, "blk.%d.post_ffw_norm.weight", il);
        L->inp_gate       = need(mf, "blk.%d.inp_gate.weight", il);
        L->pl_proj        = need(mf, "blk.%d.proj.weight", il);
        L->pl_post_norm   = need(mf, "blk.%d.post_norm.weight", il);
        L->out_scale      = opt (mf, "blk.%d.layer_output_scale.weight", il);
        if (il < mf->n_layer_kv) {
            L->wk     = need(mf, "blk.%d.attn_k.weight", il);
            L->wv     = opt (mf, "blk.%d.attn_v.weight", il);
            L->k_norm = need(mf, "blk.%d.attn_k_norm.weight", il);
            // shared-prep assumption: q/k/v same activation class
            if (act_class(L->wk->type) != act_class(L->wq->type) ||
                (L->wv && act_class(L->wv->type) != act_class(L->wq->type))) {
                fprintf(stderr, "g4: blk %d q/k/v activation class mismatch\n", il);
                return 1;
            }
        } else {
            L->wk = NULL; L->wv = NULL; L->k_norm = NULL;
        }
        if (act_class(L->gate->type) != act_class(L->up->type)) {
            fprintf(stderr, "g4: blk %d gate/up activation class mismatch\n", il);
            return 1;
        }
        int hd = mf->is_swa[il] ? mf->head_dim_swa : mf->head_dim_full;
        if (L->wq->ne[1] != (int64_t)mf->n_head * hd) {
            fprintf(stderr, "g4: blk %d wq shape mismatch\n", il);
            return 1;
        }
    }

    for (int il = 0; il < mf->n_layer_kv; il++) {
        int hd = mf->is_swa[il] ? mf->head_dim_swa : mf->head_dim_full;
        int slots = mf->is_swa[il] ? G4_SWA_RING : n_ctx;
        c->kc[il] = calloc((size_t)slots, g4_row_size(kv_type_k, hd));
        c->vc[il] = calloc((size_t)slots, g4_row_size(kv_type_v, hd));
        if (!c->kc[il] || !c->vc[il]) { fprintf(stderr, "g4: KV alloc failed\n"); return 1; }
    }

    int max_ff = 0;
    for (int il = 0; il < mf->n_layer; il++) if (mf->n_ff[il] > max_ff) max_ff = mf->n_ff[il];
    const int n_embd = mf->n_embd;
    const int qdim = max_ff > n_embd ? max_ff : n_embd;
    const int n_pl = mf->n_layer * mf->n_embd_per_layer;
    const int B = G4_BATCH_MAX;

    c->x      = malloc((size_t)B * n_embd * 4);
    c->xb     = malloc((size_t)B * n_embd * 4);
    c->q      = malloc((size_t)B * mf->n_head * mf->head_dim_full * 4);
    c->k      = malloc((size_t)B * mf->head_dim_full * 4);
    c->v      = malloc((size_t)B * mf->head_dim_full * 4);
    c->ffh    = malloc((size_t)B * max_ff * 4);
    c->ffh2   = malloc((size_t)B * max_ff * 4);
    c->pl     = malloc((size_t)B * n_pl * 4);
    c->plp    = malloc((size_t)B * n_pl * 4);
    c->plg    = malloc((size_t)B * mf->n_embd_per_layer * 4);
    c->attbuf = malloc((size_t)n_threads * n_ctx * 4);
    c->rc_full = malloc((size_t)B * mf->n_rot_full * 4);
    c->rc_swa  = malloc((size_t)B * mf->n_rot_swa * 4);
    c->bf32   = malloc((size_t)B * n_embd * 4);
    c->logits = malloc((size_t)mf->n_vocab * 4);
    c->wf16   = malloc((size_t)B * qdim * 2);
    c->qstride = (((size_t)qdim/QK_K + 1) * sizeof(block_q8_K) + 63) & ~(size_t)63;
    c->qa     = malloc((size_t)B * c->qstride);
    if (!c->logits || !c->qa || !c->attbuf) { fprintf(stderr, "g4: scratch alloc failed\n"); return 1; }

    g4_threads_start(n_threads);
    return 0;
}

void g4_ctx_free(g4_ctx * c) {
    g4_threads_stop();
    for (int il = 0; il < G4_MAX_LAYERS; il++) { free(c->kc[il]); free(c->vc[il]); }
    free(c->x); free(c->xb); free(c->q); free(c->k); free(c->v);
    free(c->ffh); free(c->ffh2); free(c->pl); free(c->plp); free(c->plg);
    free(c->attbuf); free(c->rc_full); free(c->rc_swa); free(c->bf32);
    free(c->logits); free(c->wf16); free(c->qa);
    memset(c, 0, sizeof(*c));
}

// ------------------------------------------------------------ attention -----

typedef struct {
    g4_ctx * c;
    const uint8_t * kc, * vc;
    int ktype, vtype;       // KV cache element type (G4_F16 | G4_Q8_0 | G4_Q4_0)
    size_t krow_bytes, vrow_bytes;
    const float * q;        // [nb][n_head*hd]
    float * out;            // [nb][n_head*hd]
    int hd, n_head, nb;
    int pos0;
    int slots;
    bool swa;
    int n_swa;
    int kv_last;            // highest key position to attend (-1 = self = pos0+b).
                            // The MTP draft sets this to the target's last cached
                            // position so its query (at a not-yet-cached position)
                            // attends only valid target KV.
} attn_ud;

static void attn_kernel(int64_t i0, int64_t i1, void * p, int tid) {
    attn_ud * u = p;
    const int hd = u->hd;
    float * att = u->c->attbuf + (size_t)tid * u->c->n_ctx;

    for (int64_t it = i0; it < i1; it++) {
        const int b = (int)(it / u->n_head);
        const int h = (int)(it % u->n_head);
        const int pos = u->pos0 + b;
        int j0 = 0;
        if (u->swa && pos - (u->n_swa - 1) > 0) j0 = pos - (u->n_swa - 1);
        const int last = (u->kv_last >= 0 && u->kv_last < pos) ? u->kv_last : pos;
        const int nv = last - j0 + 1;

        const float * qh = u->q + (size_t)b * u->n_head * hd + (size_t)h * hd;
        float * o = u->out + (size_t)b * u->n_head * hd + (size_t)h * hd;
        if (nv <= 0) { for (int i = 0; i < hd; i++) o[i] = 0.0f; continue; }

        // quantize the query into the K cache's activation form (f16 row, or q8_0
        // blocks for a q8_0/q4_0 cache — both use q8_0 activations in ggml).
        const int ktype = u->ktype;
        g4_fp16 q16[512];
        block_q8_0 q8[16];   // hd/32 <= 16 blocks
        if (ktype == G4_F16) { for (int i = 0; i < hd; i++) q16[i] = g4_fp32_to_fp16(qh[i]); }
        else                   g4_quantize_q8_0(qh, q8, hd);

        float maxs = -INFINITY;
        for (int j = 0; j < nv; j++) {
            const uint8_t * krow = u->kc + (size_t)((j0 + j) % u->slots) * u->krow_bytes;
            float s;                                // no scale (f_attention_scale = 1)
            switch (ktype) {
                case G4_Q8_0: s = g4_dot_q8_0_q8_0(krow, q8, hd); break;
                case G4_Q4_0: s = g4_dot_q4_0_q8_0(krow, q8, hd); break;
                default:      s = g4_dot_f16(krow, q16, hd); break;
            }
            att[j] = s;
            if (s > maxs) maxs = s;
        }
        double sum = 0.0;
        for (int j = 0; j < nv; j++) { float e = expf(att[j] - maxs); att[j] = e; sum += e; }
        const float inv = (float)(1.0 / sum);

        for (int i = 0; i < hd; i++) o[i] = 0.0f;
        if (u->vtype == G4_F16) {
#if defined(__AVX2__) && defined(__F16C__)
            for (int j = 0; j < nv; j++) {
                const float w = g4_fp16_to_fp32(g4_fp32_to_fp16(att[j] * inv));
                const g4_fp16 * vrow = (const g4_fp16 *)(u->vc + (size_t)((j0 + j) % u->slots) * u->vrow_bytes);
                const __m256 wv = _mm256_set1_ps(w);
                for (int i = 0; i < hd; i += 8) {
                    const __m256 vf = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vrow + i)));
                    _mm256_storeu_ps(o + i, _mm256_fmadd_ps(wv, vf, _mm256_loadu_ps(o + i)));
                }
            }
#else
            for (int j = 0; j < nv; j++) {
                const float w = g4_fp16_to_fp32(g4_fp32_to_fp16(att[j] * inv));
                const g4_fp16 * vrow = (const g4_fp16 *)(u->vc + (size_t)((j0 + j) % u->slots) * u->vrow_bytes);
                for (int i = 0; i < hd; i++) o[i] += w * g4_fp16_to_fp32(vrow[i]);
            }
#endif
        } else {
            float vtmp[512];
            for (int j = 0; j < nv; j++) {
                const float w = att[j] * inv;
                const uint8_t * vrow = u->vc + (size_t)((j0 + j) % u->slots) * u->vrow_bytes;
                g4_dequant_row(u->vtype, vrow, vtmp, hd);
                for (int i = 0; i < hd; i++) o[i] += w * vtmp[i];
            }
        }
    }
}

// store one KV row (hd f32 values) into the cache slot, quantizing to `type`
static void kv_store(int type, void * dst, const float * src, int hd) {
    switch (type) {
        case G4_Q8_0: g4_quantize_q8_0(src, dst, hd); break;
        case G4_Q4_0: g4_quantize_q4_0(src, dst, hd); break;
        default: { g4_fp16 * d = dst; for (int i = 0; i < hd; i++) d[i] = g4_fp32_to_fp16(src[i]); } break;
    }
}

// --------------------------------------------------------------- forward ----

static void forward_layers(g4_ctx * c, const int * tokens, int nb, int pos0) {
    g4_model_file * mf = c->mf;
    const int n_embd = mf->n_embd;
    const int n_head = mf->n_head;
    const int n_pl_e = mf->n_embd_per_layer;
    const int n_pl   = mf->n_layer * n_pl_e;
    const float eps  = mf->rms_eps;

    if (nb > G4_BATCH_MAX) { fprintf(stderr, "g4: batch too large\n"); exit(1); }

    // ---- embeddings + per-layer token rows ----
    {
        const float es = sqrtf((float)n_embd);
        const float ts = sqrtf((float)n_pl_e);
        const size_t erb = g4_row_size(c->tok_embd->type, n_embd);
        const size_t prb = g4_row_size(c->per_layer_tok_embd->type, n_pl);
        for (int b = 0; b < nb; b++) {
            float * xb_ = c->x + (size_t)b*n_embd;
            g4_dequant_row(c->tok_embd->type, (const uint8_t *)c->tok_embd->data + (size_t)tokens[b]*erb, xb_, n_embd);
            for (int i = 0; i < n_embd; i++) xb_[i] *= es;
            float * plb = c->pl + (size_t)b*n_pl;
            g4_dequant_row(c->per_layer_tok_embd->type, (const uint8_t *)c->per_layer_tok_embd->data + (size_t)tokens[b]*prb, plb, n_pl);
            for (int i = 0; i < n_pl; i++) plb[i] *= ts;
        }
    }

    // ---- per-layer input projection ----
    {
        matmul_b(c, c->per_layer_model_proj->type, c->per_layer_model_proj->data,
                 c->x, NULL, 0, c->plp, n_embd, n_pl, nb);
        const float ps  = 1.0f / sqrtf((float)n_embd);
        const float is2 = 1.0f / sqrtf(2.0f);
        const float * pw = (const float *)c->per_layer_proj_norm->data;
        for (int b = 0; b < nb; b++) {
            float * plpb = c->plp + (size_t)b*n_pl;
            float * plb  = c->pl  + (size_t)b*n_pl;
            for (int i = 0; i < n_pl; i++) plpb[i] *= ps;
            for (int l = 0; l < mf->n_layer; l++) {
                float * pp = plpb + (size_t)l*n_pl_e;
                float * pt = plb  + (size_t)l*n_pl_e;
                rms_norm(pp, pw, pp, n_pl_e, eps);
                for (int i = 0; i < n_pl_e; i++) pt[i] = (pp[i] + pt[i]) * is2;
            }
        }
    }

    // ---- rope caches ----
    for (int b = 0; b < nb; b++) {
        rope_cache(c->rc_swa  + (size_t)b*mf->n_rot_swa,  mf->n_rot_swa,  pos0 + b, mf->rope_base_swa,  NULL);
        rope_cache(c->rc_full + (size_t)b*mf->n_rot_full, mf->n_rot_full, pos0 + b, mf->rope_base_full,
                   (const float *)c->rope_freqs->data);
    }

    // ---- layers ----
    for (int il = 0; il < mf->n_layer; il++) {
        const g4_layer * L = &c->layers[il];
        const bool swa = mf->is_swa[il] != 0;
        const int  hd  = swa ? mf->head_dim_swa : mf->head_dim_full;
        const int  qrows = n_head * hd;
        const float * rc = swa ? c->rc_swa : c->rc_full;
        const int rcs = swa ? mf->n_rot_swa : mf->n_rot_full;

        // attn input norm
        for (int b = 0; b < nb; b++)
            rms_norm(c->x + (size_t)b*n_embd, (const float *)L->attn_norm->data, c->xb + (size_t)b*n_embd, n_embd, eps);

        // Q,K,V share one activation preparation
        size_t ast;
        const void * act = prep_act(c, act_class(L->wq->type), c->xb, n_embd, nb, &ast);
        matmul_b(c, L->wq->type, L->wq->data, NULL, act, ast, c->q, n_embd, qrows, nb);

        const float * qnw = (const float *)L->q_norm->data;
        for (int b = 0; b < nb; b++) {
            for (int h = 0; h < n_head; h++) {
                float * qh = c->q + (size_t)b*qrows + (size_t)h*hd;
                rms_norm(qh, qnw, qh, hd, eps);
                rope_apply(qh, rc + (size_t)b*rcs, hd);
            }
        }

        if (il < mf->n_layer_kv) {
            matmul_b(c, L->wk->type, L->wk->data, NULL, act, ast, c->k, n_embd, hd, nb);
            if (L->wv) matmul_b(c, L->wv->type, L->wv->data, NULL, act, ast, c->v, n_embd, hd, nb);
            const int slots = swa ? G4_SWA_RING : c->n_ctx;
            const size_t krb = g4_row_size(c->kv_type_k, hd), vrb = g4_row_size(c->kv_type_v, hd);
            for (int b = 0; b < nb; b++) {
                float * kb = c->k + (size_t)b*hd;
                float * vb = c->v + (size_t)b*hd;
                if (!L->wv) memcpy(vb, kb, hd*4);   // V = K fallback (raw projection, pre-norm/rope)
                rms_norm(kb, (const float *)L->k_norm->data, kb, hd, eps);
                rope_apply(kb, rc + (size_t)b*rcs, hd);
                rms_norm(vb, NULL, vb, hd, eps);
                kv_store(c->kv_type_k, c->kc[il] + (size_t)((pos0 + b) % slots) * krb, kb, hd);
                kv_store(c->kv_type_v, c->vc[il] + (size_t)((pos0 + b) % slots) * vrb, vb, hd);
            }
        }

        // attention (own cache or reused from layer 13/14)
        {
            const int cache_il = il < mf->n_layer_kv ? il : (swa ? mf->n_layer_kv - 2 : mf->n_layer_kv - 1);
            attn_ud au;
            au.c = c;
            au.kc = c->kc[cache_il]; au.vc = c->vc[cache_il];
            au.ktype = c->kv_type_k; au.vtype = c->kv_type_v;
            au.krow_bytes = g4_row_size(c->kv_type_k, hd);
            au.vrow_bytes = g4_row_size(c->kv_type_v, hd);
            au.q = c->q;
            au.out = c->ffh;
            au.hd = hd; au.n_head = n_head; au.nb = nb;
            au.pos0 = pos0;
            au.slots = swa ? G4_SWA_RING : c->n_ctx;
            au.swa = swa; au.n_swa = mf->n_swa;
            au.kv_last = -1;
            g4_parallel_for((int64_t)nb * n_head, attn_kernel, &au);
        }

        // attn out projection + residual
        matmul_b(c, L->wo->type, L->wo->data, c->ffh, NULL, 0, c->xb, qrows, n_embd, nb);
        for (int b = 0; b < nb; b++) {
            float * xx = c->x + (size_t)b*n_embd;
            float * xb_ = c->xb + (size_t)b*n_embd;
            rms_norm(xb_, (const float *)L->post_attn_norm->data, xb_, n_embd, eps);
            for (int i = 0; i < n_embd; i++) xx[i] += xb_[i];
        }

        // FFN
        const int n_ff = mf->n_ff[il];
        for (int b = 0; b < nb; b++)
            rms_norm(c->x + (size_t)b*n_embd, (const float *)L->ffn_norm->data, c->xb + (size_t)b*n_embd, n_embd, eps);
        act = prep_act(c, act_class(L->gate->type), c->xb, n_embd, nb, &ast);
        matmul_b(c, L->gate->type, L->gate->data, NULL, act, ast, c->ffh,  n_embd, n_ff, nb);
        matmul_b(c, L->up->type,   L->up->data,   NULL, act, ast, c->ffh2, n_embd, n_ff, nb);
        for (int b = 0; b < nb; b++) {
            float * g = c->ffh + (size_t)b*n_ff;
            const float * uu = c->ffh2 + (size_t)b*n_ff;
            for (int i = 0; i < n_ff; i++) g[i] = g4_gelu(g[i]) * uu[i];
        }
        matmul_b(c, L->down->type, L->down->data, c->ffh, NULL, 0, c->xb, n_ff, n_embd, nb);
        for (int b = 0; b < nb; b++) {
            float * xx = c->x + (size_t)b*n_embd;
            float * xb_ = c->xb + (size_t)b*n_embd;
            rms_norm(xb_, (const float *)L->post_ffw_norm->data, xb_, n_embd, eps);
            for (int i = 0; i < n_embd; i++) xx[i] += xb_[i];
        }

        // per-layer embedding injection
        matmul_b(c, L->inp_gate->type, L->inp_gate->data, c->x, NULL, 0, c->plg, n_embd, n_pl_e, nb);
        for (int b = 0; b < nb; b++) {
            float * g = c->plg + (size_t)b*n_pl_e;
            const float * pli = c->pl + (size_t)b*n_pl + (size_t)il*n_pl_e;
            for (int i = 0; i < n_pl_e; i++) g[i] = g4_gelu(g[i]) * pli[i];
        }
        matmul_b(c, L->pl_proj->type, L->pl_proj->data, c->plg, NULL, 0, c->xb, n_pl_e, n_embd, nb);
        const float os = L->out_scale ? ((const float *)L->out_scale->data)[0] : 1.0f;
        for (int b = 0; b < nb; b++) {
            float * xx = c->x + (size_t)b*n_embd;
            float * xb_ = c->xb + (size_t)b*n_embd;
            rms_norm(xb_, (const float *)L->pl_post_norm->data, xb_, n_embd, eps);
            for (int i = 0; i < n_embd; i++) xx[i] = (xx[i] + xb_[i]) * os;
        }
    }

}

// public: run layers + last-token head (the common prefill/decode path)
float * g4_forward_batch(g4_ctx * c, const int * tokens, int nb, int pos0, bool want_logits) {
    g4_model_file * mf = c->mf;
    const int n_embd = mf->n_embd;
    const float eps  = mf->rms_eps;
    forward_layers(c, tokens, nb, pos0);
    c->pos = pos0 + nb;
    if (!want_logits) return NULL;

    // ---- head (last token only) ----
    float * xl = c->x + (size_t)(nb - 1)*n_embd;
    rms_norm(xl, (const float *)c->output_norm->data, c->xb, n_embd, eps);
    matmul_b(c, c->tok_embd->type, c->tok_embd->data, c->xb, NULL, 0, c->logits, n_embd, mf->n_vocab, 1);
    if (mf->final_softcap > 0) {
        const float cap = mf->final_softcap;
        for (int i = 0; i < mf->n_vocab; i++) c->logits[i] = cap * tanhf(c->logits[i] / cap);
    }
    return c->logits;
}

// public: run layers + per-position head + per-position h_nextn (MTP verify).
// logits_out: [nb][n_vocab]; hnext_out: [nb][n_embd] = post-output_norm hidden
// (the LM-head input feature, which the draft head consumes as its `h` input).
void g4_forward_batch_heads(g4_ctx * c, const int * tokens, int nb, int pos0,
                            float * logits_out, float * hnext_out) {
    g4_model_file * mf = c->mf;
    const int n_embd = mf->n_embd;
    const float eps  = mf->rms_eps;
    forward_layers(c, tokens, nb, pos0);
    c->pos = pos0 + nb;
    for (int b = 0; b < nb; b++)
        rms_norm(c->x + (size_t)b*n_embd, (const float *)c->output_norm->data,
                 hnext_out + (size_t)b*n_embd, n_embd, eps);
    matmul_b(c, c->tok_embd->type, c->tok_embd->data, hnext_out, NULL, 0,
             logits_out, n_embd, mf->n_vocab, nb);
    if (mf->final_softcap > 0) {
        const float cap = mf->final_softcap;
        const int64_t nv = mf->n_vocab;
        for (int b = 0; b < nb; b++)
            for (int64_t i = 0; i < nv; i++)
                logits_out[(size_t)b*nv + i] = cap * tanhf(logits_out[(size_t)b*nv + i] / cap);
    }
}

// ----------------------------------------------------- MTP draft (assistant) -
// The gemma4-assistant head: a 4-layer, 256-d transformer that drafts the
// target's next tokens. It owns no KV cache — its attention reads the target's
// (the `share` callback in llama.cpp maps each draft layer to the target's
// shared SWA/global cache; here that is target layers n_layer_kv-2 / n_layer_kv-1).

int g4_mtp_init(g4_mtp * d, g4_model_file * dm, g4_ctx * tgt) {
    memset(d, 0, sizeof(*d));
    d->mf = dm;
    d->tgt = tgt;
    d->n_embd_d   = dm->n_embd;            // 256
    d->n_embd_out = dm->n_embd_out;        // 1536 (backbone dim)
    if (d->n_embd_out != tgt->mf->n_embd) {
        fprintf(stderr, "g4: MTP backbone dim %d != target n_embd %d\n", d->n_embd_out, tgt->mf->n_embd);
        return 1;
    }
    d->tok_embd    = need(dm, "token_embd.weight", 0);
    d->pre_proj    = need(dm, "nextn.pre_projection.weight", 0);
    d->post_proj   = need(dm, "nextn.post_projection.weight", 0);
    d->output_norm = need(dm, "output_norm.weight", 0);
    d->rope_freqs  = need(dm, "rope_freqs.weight", 0);
    for (int il = 0; il < dm->n_layer; il++) {
        g4_layer * L = &d->layers[il];
        L->attn_norm      = need(dm, "blk.%d.attn_norm.weight", il);
        L->wq             = need(dm, "blk.%d.attn_q.weight", il);
        L->wo             = need(dm, "blk.%d.attn_output.weight", il);
        L->q_norm         = need(dm, "blk.%d.attn_q_norm.weight", il);
        L->post_attn_norm = need(dm, "blk.%d.post_attention_norm.weight", il);
        L->ffn_norm       = need(dm, "blk.%d.ffn_norm.weight", il);
        L->gate           = need(dm, "blk.%d.ffn_gate.weight", il);
        L->up             = need(dm, "blk.%d.ffn_up.weight", il);
        L->down           = need(dm, "blk.%d.ffn_down.weight", il);
        L->post_ffw_norm  = need(dm, "blk.%d.post_ffw_norm.weight", il);
        L->out_scale      = opt (dm, "blk.%d.layer_output_scale.weight", il);
    }
    const int ne_out = d->n_embd_out, ne_d = d->n_embd_d;
    const int qmax = dm->n_head * dm->head_dim_full;     // 4*512
    int max_ff = 0;
    for (int il = 0; il < dm->n_layer; il++) if (dm->n_ff[il] > max_ff) max_ff = dm->n_ff[il];
    const int ffmax = max_ff > qmax ? max_ff : qmax;
    d->xh     = malloc((size_t)2*ne_out*4);
    d->cur    = malloc((size_t)ne_d*4);
    d->xb     = malloc((size_t)ne_d*4);
    d->q      = malloc((size_t)qmax*4);
    d->ffh    = malloc((size_t)ffmax*4);
    d->ffh2   = malloc((size_t)max_ff*4);
    d->rc_full= malloc((size_t)dm->n_rot_full*4);
    d->rc_swa = malloc((size_t)dm->n_rot_swa*4);
    d->logits = malloc((size_t)dm->n_vocab*4);
    if (!d->xh || !d->cur || !d->xb || !d->q || !d->ffh || !d->ffh2 ||
        !d->rc_full || !d->rc_swa || !d->logits) {
        fprintf(stderr, "g4: MTP scratch alloc failed\n"); return 1;
    }
    return 0;
}

void g4_mtp_free(g4_mtp * d) {
    free(d->xh); free(d->cur); free(d->xb); free(d->q); free(d->ffh);
    free(d->ffh2); free(d->rc_full); free(d->rc_swa); free(d->logits);
    memset(d, 0, sizeof(*d));
}

int g4_mtp_step(g4_mtp * d, int token, const float * h_in, int qpos, int kv_last, float * h_out) {
    g4_ctx * c = d->tgt;
    g4_model_file * tm = c->mf;        // target model file (KV layout, n_swa, n_layer_kv)
    g4_model_file * dm = d->mf;        // draft model file
    const int ne_out = d->n_embd_out;  // 1536
    const int ne_d   = d->n_embd_d;    // 256
    const float eps  = dm->rms_eps;
    const int n_head = dm->n_head;     // 4

    // 1. input feature = concat( sqrt(n_embd)·target_embd(token) , h_in )  -> 2*n_embd_out
    const size_t erb = g4_row_size(c->tok_embd->type, ne_out);
    g4_dequant_row(c->tok_embd->type, (const uint8_t *)c->tok_embd->data + (size_t)token*erb, d->xh, ne_out);
    const float es = sqrtf((float)ne_out);
    for (int i = 0; i < ne_out; i++) d->xh[i] *= es;
    memcpy(d->xh + ne_out, h_in, (size_t)ne_out * 4);

    // 2. pre_projection -> draft hidden (256)
    matmul_b(c, d->pre_proj->type, d->pre_proj->data, d->xh, NULL, 0, d->cur, (int64_t)2*ne_out, ne_d, 1);

    // rope caches at the query position (draft's own rope_freqs for global layers)
    rope_cache(d->rc_swa,  dm->n_rot_swa,  qpos, dm->rope_base_swa,  NULL);
    rope_cache(d->rc_full, dm->n_rot_full, qpos, dm->rope_base_full, (const float *)d->rope_freqs->data);

    // 3. transformer layers (attention reuses the target KV cache)
    for (int il = 0; il < dm->n_layer; il++) {
        const g4_layer * L = &d->layers[il];
        const bool swa = dm->is_swa[il] != 0;
        const int  hd  = swa ? dm->head_dim_swa : dm->head_dim_full;
        const int  qrows = n_head * hd;
        const float * rc = swa ? d->rc_swa : d->rc_full;

        rms_norm(d->cur, (const float *)L->attn_norm->data, d->xb, ne_d, eps);
        matmul_b(c, L->wq->type, L->wq->data, d->xb, NULL, 0, d->q, ne_d, qrows, 1);
        const float * qnw = (const float *)L->q_norm->data;
        for (int h = 0; h < n_head; h++) {
            float * qh = d->q + (size_t)h*hd;
            rms_norm(qh, qnw, qh, hd, eps);
            rope_apply(qh, rc, hd);
        }
        const int cache_il = swa ? tm->n_layer_kv - 2 : tm->n_layer_kv - 1;
        attn_ud au;
        au.c = c;
        au.kc = c->kc[cache_il]; au.vc = c->vc[cache_il];
        au.ktype = c->kv_type_k; au.vtype = c->kv_type_v;
        au.krow_bytes = g4_row_size(c->kv_type_k, hd);
        au.vrow_bytes = g4_row_size(c->kv_type_v, hd);
        au.q = d->q; au.out = d->ffh;
        au.hd = hd; au.n_head = n_head; au.nb = 1;
        au.pos0 = qpos;
        au.slots = swa ? G4_SWA_RING : c->n_ctx;
        au.swa = swa; au.n_swa = tm->n_swa;
        au.kv_last = kv_last;
        g4_parallel_for(n_head, attn_kernel, &au);

        matmul_b(c, L->wo->type, L->wo->data, d->ffh, NULL, 0, d->xb, qrows, ne_d, 1);
        rms_norm(d->xb, (const float *)L->post_attn_norm->data, d->xb, ne_d, eps);
        for (int i = 0; i < ne_d; i++) d->cur[i] += d->xb[i];

        const int n_ff = dm->n_ff[il];
        rms_norm(d->cur, (const float *)L->ffn_norm->data, d->xb, ne_d, eps);
        matmul_b(c, L->gate->type, L->gate->data, d->xb, NULL, 0, d->ffh,  ne_d, n_ff, 1);
        matmul_b(c, L->up->type,   L->up->data,   d->xb, NULL, 0, d->ffh2, ne_d, n_ff, 1);
        for (int i = 0; i < n_ff; i++) d->ffh[i] = g4_gelu(d->ffh[i]) * d->ffh2[i];
        matmul_b(c, L->down->type, L->down->data, d->ffh, NULL, 0, d->xb, n_ff, ne_d, 1);
        rms_norm(d->xb, (const float *)L->post_ffw_norm->data, d->xb, ne_d, eps);
        for (int i = 0; i < ne_d; i++) d->cur[i] += d->xb[i];

        const float os = L->out_scale ? ((const float *)L->out_scale->data)[0] : 1.0f;
        for (int i = 0; i < ne_d; i++) d->cur[i] *= os;
    }

    // 4. output norm
    rms_norm(d->cur, (const float *)d->output_norm->data, d->cur, ne_d, eps);

    // 5a. chained backbone hidden for the next draft step
    matmul_b(c, d->post_proj->type, d->post_proj->data, d->cur, NULL, 0, h_out, ne_d, ne_out, 1);

    // 5b. draft logits (tied head; no final softcap on the draft) -> argmax
    matmul_b(c, d->tok_embd->type, d->tok_embd->data, d->cur, NULL, 0, d->logits, ne_d, tm->n_vocab, 1);
    int best = 0; float bv = d->logits[0];
    for (int i = 1; i < tm->n_vocab; i++) if (d->logits[i] > bv) { bv = d->logits[i]; best = i; }
    return best;
}

// ===================================================== qwen35 hybrid model ===
// Gated-DeltaNet (linear-attention SSM) + gated full-attention. Ported from
// llama.cpp src/models/qwen35.cpp + delta-net-base.cpp + the CPU kernels
// ggml_compute_forward_gated_delta_net_f32 / ggml_ssm_conv (token-at-a-time).

static inline float qsigmoid (float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float qsilu    (float x) { return x  / (1.0f + expf(-x)); }
static inline float qsoftplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// L2 normalize n floats in place: x /= max(||x||, eps)  (ggml_l2_norm)
static void l2_norm(float * x, int n, float eps) {
    double s = 0.0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    const float sc = 1.0f / fmaxf(sqrtf((float)s), eps);
    for (int i = 0; i < n; i++) x[i] *= sc;
}

// matmul helper for the qwen scratch ctx (token-major, NB tokens)
#define QMM(c, W, X, Y, NC, NR, NB) \
    matmul_b(&(c)->mm, (W)->type, (W)->data, (X), NULL, 0, (Y), (NC), (NR), (NB))

int g4_qwen_init(qwen_ctx * c, g4_model_file * mf, int n_ctx, int n_threads) {
    memset(c, 0, sizeof(*c));
    c->mf = mf;
    c->n_ctx = n_ctx;
    c->n_threads = n_threads;
    c->head_dim   = mf->n_embd_head_k;          // 256
    c->n_head     = mf->n_head;                 // 16
    c->n_head_kv  = mf->n_head_kv;              // 4
    c->s_v        = mf->ssm_d_state;            // 128 (= head_k_dim = head_v_dim)
    c->h_v        = mf->ssm_dt_rank;            // 32  (num v heads)
    c->h_k        = mf->ssm_n_group;            // 16  (num k heads)
    c->key_dim    = c->s_v * c->h_k;            // 2048
    c->val_dim    = mf->ssm_d_inner;            // 4096
    c->conv_dim   = c->key_dim * 2 + c->val_dim;// 8192
    c->n_ff       = mf->n_ff[0];                // 9216
    const int n_embd = mf->n_embd;
    const int dc = mf->ssm_d_conv;              // 4

    c->tok_embd    = need(mf, "token_embd.weight", 0);
    c->output_norm = need(mf, "output_norm.weight", 0);
    c->output      = g4_find_tensor(mf, "output.weight");
    if (!c->output) c->output = c->tok_embd;    // tied
    for (int il = 0; il < mf->n_layer; il++) {
        qwen_layer * L = &c->layers[il];
        L->is_recr        = mf->is_recr[il] != 0;
        L->attn_norm      = need(mf, "blk.%d.attn_norm.weight", il);
        L->attn_post_norm = need(mf, "blk.%d.post_attention_norm.weight", il);
        L->ffn_gate       = need(mf, "blk.%d.ffn_gate.weight", il);
        L->ffn_up         = need(mf, "blk.%d.ffn_up.weight", il);
        L->ffn_down       = need(mf, "blk.%d.ffn_down.weight", il);
        if (L->is_recr) {
            L->wqkv       = need(mf, "blk.%d.attn_qkv.weight", il);
            L->wqkv_gate  = need(mf, "blk.%d.attn_gate.weight", il);
            L->ssm_conv1d = need(mf, "blk.%d.ssm_conv1d.weight", il);
            L->ssm_dt     = need(mf, "blk.%d.ssm_dt.bias", il);
            L->ssm_a      = need(mf, "blk.%d.ssm_a", il);
            L->ssm_beta   = need(mf, "blk.%d.ssm_beta.weight", il);
            L->ssm_alpha  = need(mf, "blk.%d.ssm_alpha.weight", il);
            L->ssm_norm   = need(mf, "blk.%d.ssm_norm.weight", il);
            L->ssm_out    = need(mf, "blk.%d.ssm_out.weight", il);
            c->conv_state[il] = calloc((size_t)c->conv_dim * (dc - 1), sizeof(float));
            c->ssm_state[il]  = calloc((size_t)c->s_v * c->s_v * c->h_v, sizeof(float));
            if (!c->conv_state[il] || !c->ssm_state[il]) { fprintf(stderr, "g4: qwen ssm state alloc\n"); return 1; }
        } else {
            L->wq     = need(mf, "blk.%d.attn_q.weight", il);
            L->wk     = need(mf, "blk.%d.attn_k.weight", il);
            L->wv     = need(mf, "blk.%d.attn_v.weight", il);
            L->wo     = need(mf, "blk.%d.attn_output.weight", il);
            L->q_norm = need(mf, "blk.%d.attn_q_norm.weight", il);
            L->k_norm = need(mf, "blk.%d.attn_k_norm.weight", il);
            c->kc[il] = calloc((size_t)n_ctx * c->n_head_kv * c->head_dim, sizeof(float));
            c->vc[il] = calloc((size_t)n_ctx * c->n_head_kv * c->head_dim, sizeof(float));
            if (!c->kc[il] || !c->vc[il]) { fprintf(stderr, "g4: qwen kv alloc\n"); return 1; }
        }
    }

    // scratch (token-major, B rows for batched prefill)
    const int B = G4_BATCH_MAX;
    const int attno_row = c->head_dim * c->n_head > c->val_dim ? c->head_dim * c->n_head : c->val_dim;
    int big = c->n_ff > c->conv_dim ? c->n_ff : c->conv_dim;     // 9216
    c->x     = malloc((size_t)B * n_embd * 4);
    c->xb    = malloc((size_t)B * n_embd * 4);
    c->cur   = malloc((size_t)B * n_embd * 4);
    c->tmp   = malloc((size_t)B * n_embd * 4);
    c->qkv   = malloc((size_t)B * c->conv_dim * 4);
    c->convo = malloc((size_t)B * c->conv_dim * 4);
    c->qfull = malloc((size_t)B * c->head_dim * c->n_head * 2 * 4);
    c->kb    = malloc((size_t)B * c->n_head_kv * c->head_dim * 4);
    c->vb    = malloc((size_t)B * c->n_head_kv * c->head_dim * 4);
    c->attno = malloc((size_t)B * attno_row * 4);
    c->z     = malloc((size_t)B * c->val_dim * 4);
    c->alpha = malloc((size_t)B * c->h_v * 4);
    c->beta  = malloc((size_t)B * c->h_v * 4);
    c->gdec  = malloc((size_t)B * c->h_v * 4);
    c->ffh   = malloc((size_t)B * c->n_ff * 4);
    c->ffh2  = malloc((size_t)B * c->n_ff * 4);
    c->att   = malloc((size_t)n_threads * n_ctx * 4);    // per-thread attention scratch
    c->rc    = malloc((size_t)B * mf->n_rot_full * 4);
    c->logits= malloc((size_t)mf->n_vocab * 4);

    // matmul scratch ctx (only qa/wf16/bf32/n_threads/mf are read by matmul_b)
    c->mm.mf = mf;
    c->mm.n_threads = n_threads;
    c->mm.qstride = (((size_t)big / QK_K + 2) * sizeof(block_q8_K) + 63) & ~(size_t)63;
    c->mm.qa   = malloc((size_t)B * c->mm.qstride);
    c->mm.wf16 = malloc((size_t)B * big * 2);
    c->mm.bf32 = malloc((size_t)B * big * 4);
    if (!c->logits || !c->mm.qa) { fprintf(stderr, "g4: qwen scratch alloc\n"); return 1; }

    g4_threads_start(n_threads);
    return 0;
}

// zero the recurrent state (conv + ssm) for a fresh conversation. The attention
// KV cache is positional and overwritten on re-prefill, so it needs no reset.
void g4_qwen_reset(qwen_ctx * c) {
    const int dc = c->mf->ssm_d_conv;
    for (int il = 0; il < c->mf->n_layer; il++) {
        if (c->conv_state[il]) memset(c->conv_state[il], 0, (size_t)c->conv_dim * (dc - 1) * sizeof(float));
        if (c->ssm_state[il])  memset(c->ssm_state[il],  0, (size_t)c->s_v * c->s_v * c->h_v * sizeof(float));
    }
}

void g4_qwen_free(qwen_ctx * c) {
    g4_threads_stop();
    for (int il = 0; il < G4_MAX_LAYERS; il++) {
        free(c->kc[il]); free(c->vc[il]); free(c->conv_state[il]); free(c->ssm_state[il]);
        free(c->ck_conv[il]); free(c->ck_ssm[il]);
    }
    free(c->x); free(c->xb); free(c->cur); free(c->tmp); free(c->qkv); free(c->convo);
    free(c->qfull); free(c->kb); free(c->vb); free(c->attno); free(c->z);
    free(c->alpha); free(c->beta); free(c->gdec); free(c->ffh); free(c->ffh2);
    free(c->att); free(c->rc); free(c->logits);
    free(c->mm.qa); free(c->mm.wf16); free(c->mm.bf32);
    memset(c, 0, sizeof(*c));
}

// ---- parallel kernels (batched: token-major scratch, nb tokens) ----
typedef struct { qwen_ctx * c; int il, pos0, nb; } qattn_ud;
typedef struct { qwen_ctx * c; int il, nb; } qssm_ud;

// gated attention per (token,head): scores+softmax over KV[0..pos0+b], V-weight, gate
static void qattn_kernel(int64_t it0, int64_t it1, void * p, int tid) {
    qattn_ud * u = p; qwen_ctx * c = u->c; const int il = u->il;
    const int hd = c->head_dim, nh = c->n_head, nkv = c->n_head_kv, grp = nh / nkv;
    float * att = c->att + (size_t)tid * c->n_ctx;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int64_t it = it0; it < it1; it++) {
        const int b = (int)(it / nh), h = (int)(it % nh), kvh = h / grp, pos = u->pos0 + b;
        const float * q_h = c->qfull + (size_t)b * hd * nh * 2 + (size_t)h * hd * 2;
        const float * g_h = q_h + hd;
        float * o = c->attno + (size_t)b * hd * nh + (size_t)h * hd;
        float maxs = -INFINITY;
        for (int j = 0; j <= pos; j++) {
            const float * k_j = c->kc[il] + ((size_t)j * nkv + kvh) * hd;
            float s = vdot(q_h, k_j, hd) * scale;
            att[j] = s; if (s > maxs) maxs = s;
        }
        double sum = 0; for (int j = 0; j <= pos; j++) { float e = expf(att[j] - maxs); att[j] = e; sum += e; }
        const float inv = (float)(1.0 / sum);
        for (int i = 0; i < hd; i++) o[i] = 0.0f;
        for (int j = 0; j <= pos; j++) {
            const float * v_j = c->vc[il] + ((size_t)j * nkv + kvh) * hd;
            vmadd(o, v_j, att[j] * inv, hd);
        }
        for (int i = 0; i < hd; i++) o[i] *= qsigmoid(g_h[i]);   // output gate
    }
}

// causal depthwise conv per channel across the whole batch (state carries per channel)
static void qconv_kernel(int64_t c0, int64_t c1, void * p, int tid) {
    (void)tid; qssm_ud * u = p; qwen_ctx * c = u->c;
    const int dc = c->mf->ssm_d_conv, cd = c->conv_dim, nb = u->nb;
    const float * cw = (const float *)c->layers[u->il].ssm_conv1d->data;
    float * cs = c->conv_state[u->il];
    const int ck = c->ck_on;
    for (int64_t ch = c0; ch < c1; ch++) {
        const float * w = cw + (size_t)ch * dc;
        float * st = cs + (size_t)ch * (dc - 1);
        for (int b = 0; b < nb; b++) {
            const float in = c->qkv[(size_t)b * cd + ch];
            float acc = 0; for (int i = 0; i < dc - 1; i++) acc += st[i] * w[i];
            acc += in * w[dc - 1];
            c->convo[(size_t)b * cd + ch] = qsilu(acc);
            for (int i = 0; i < dc - 2; i++) st[i] = st[i + 1];
            st[dc - 2] = in;
            if (ck)   // checkpoint this channel's conv window after ingesting token b
                memcpy(c->ck_conv[u->il] + (size_t)b * c->ck_stride_conv + (size_t)ch * (dc - 1),
                       st, (size_t)(dc - 1) * sizeof(float));
        }
    }
}

// gated-delta recurrence per v-head across the whole batch (state carries per head)
static void qscan_kernel(int64_t h0, int64_t h1, void * p, int tid) {
    (void)tid; qssm_ud * u = p; qwen_ctx * c = u->c;
    const int sv = c->s_v, hk = c->h_k, hv = c->h_v, key_dim = c->key_dim, cd = c->conv_dim, vd = c->val_dim, nb = u->nb;
    const float oscale = 1.0f / sqrtf((float)sv);
    for (int64_t h = h0; h < h1; h++) {
        const int khh = (int)(h % hk);
        float * S = c->ssm_state[u->il] + (size_t)h * sv * sv;
        for (int b = 0; b < nb; b++) {
            const float * conv = c->convo + (size_t)b * cd;
            const float * q_h = conv + (size_t)khh * sv;
            const float * k_h = conv + key_dim + (size_t)khh * sv;
            const float * v_h = conv + 2 * key_dim + (size_t)h * sv;
            const float dec = expf(c->gdec[(size_t)b * hv + h]), bet = c->beta[(size_t)b * hv + h];
            vscale(S, dec, sv * sv);
            for (int j = 0; j < sv; j++) {
                float * Sr = S + (size_t)j * sv;
                const float dl = (v_h[j] - vdot(Sr, k_h, sv)) * bet;
                vmadd(Sr, k_h, dl, sv);
            }
            float * o = c->attno + (size_t)b * vd + (size_t)h * sv;
            for (int j = 0; j < sv; j++)
                o[j] = vdot(S + (size_t)j * sv, q_h, sv) * oscale;
            if (c->ck_on)   // checkpoint this head's recurrent state after ingesting token b
                memcpy(c->ck_ssm[u->il] + (size_t)b * c->ck_stride_ssm + (size_t)h * sv * sv,
                       S, (size_t)sv * sv * sizeof(float));
        }
    }
}

// gated full-attention layer (batched; reads c->cur[nb], writes c->cur[nb])
static void qwen_attn_batch(qwen_ctx * c, int il, int pos0, int nb) {
    g4_model_file * mf = c->mf;
    const qwen_layer * L = &c->layers[il];
    const int hd = c->head_dim, nh = c->n_head, nkv = c->n_head_kv, n_embd = mf->n_embd, n_rot = mf->n_rot_full;
    const float eps = mf->rms_eps;

    // q/k/v share the input c->cur — quantize the activation once and reuse it across
    // all three matmuls when they share an activation class (saves 2 of 3 prep passes).
    if (act_class(L->wq->type) == act_class(L->wk->type) && act_class(L->wq->type) == act_class(L->wv->type)) {
        size_t ast; const void * pa = prep_act(&c->mm, act_class(L->wq->type), c->cur, n_embd, nb, &ast);
        matmul_b(&c->mm, L->wq->type, L->wq->data, c->cur, pa, ast, c->qfull, n_embd, hd * nh * 2, nb);
        matmul_b(&c->mm, L->wk->type, L->wk->data, c->cur, pa, ast, c->kb,    n_embd, nkv * hd, nb);
        matmul_b(&c->mm, L->wv->type, L->wv->data, c->cur, pa, ast, c->vb,    n_embd, nkv * hd, nb);
    } else {
        QMM(c, L->wq, c->cur, c->qfull, n_embd, hd * nh * 2, nb);
        QMM(c, L->wk, c->cur, c->kb,    n_embd, nkv * hd, nb);
        QMM(c, L->wv, c->cur, c->vb,    n_embd, nkv * hd, nb);
    }

    for (int b = 0; b < nb; b++) {
        float * rc = c->rc + (size_t)b * n_rot;
        rope_cache(rc, n_rot, pos0 + b, mf->rope_base_full, NULL);
        float * qf = c->qfull + (size_t)b * hd * nh * 2;
        for (int h = 0; h < nh; h++) {
            float * q_h = qf + (size_t)h * hd * 2;
            rms_norm(q_h, (const float *)L->q_norm->data, q_h, hd, eps);
            rope_apply(q_h, rc, n_rot);
        }
        float * kb = c->kb + (size_t)b * nkv * hd;
        for (int hk = 0; hk < nkv; hk++) {
            float * k_h = kb + (size_t)hk * hd;
            rms_norm(k_h, (const float *)L->k_norm->data, k_h, hd, eps);
            rope_apply(k_h, rc, n_rot);
        }
        memcpy(c->kc[il] + (size_t)(pos0 + b) * nkv * hd, kb, (size_t)nkv * hd * 4);
        memcpy(c->vc[il] + (size_t)(pos0 + b) * nkv * hd, c->vb + (size_t)b * nkv * hd, (size_t)nkv * hd * 4);
    }
    qattn_ud u = { c, il, pos0, nb };
    g4_parallel_for((int64_t)nb * nh, qattn_kernel, &u);
    QMM(c, L->wo, c->attno, c->cur, hd * nh, n_embd, nb);
}

// gated delta-net (linear attention) layer (batched; reads c->cur[nb], writes c->cur[nb])
static void qwen_ssm_batch(qwen_ctx * c, int il, int nb) {
    g4_model_file * mf = c->mf;
    const qwen_layer * L = &c->layers[il];
    const int n_embd = mf->n_embd, sv = c->s_v, hv = c->h_v, hk = c->h_k;
    const int conv_dim = c->conv_dim, key_dim = c->key_dim, vd = c->val_dim;
    const float eps = mf->rms_eps;

    QMM(c, L->wqkv,      c->cur, c->qkv,   n_embd, conv_dim, nb);
    QMM(c, L->wqkv_gate, c->cur, c->z,     n_embd, vd, nb);
    QMM(c, L->ssm_alpha, c->cur, c->alpha, n_embd, hv, nb);
    QMM(c, L->ssm_beta,  c->cur, c->beta,  n_embd, hv, nb);

    const float * dt = (const float *)L->ssm_dt->data;
    const float * aa = (const float *)L->ssm_a->data;
    for (int b = 0; b < nb; b++)
        for (int h = 0; h < hv; h++) {
            c->gdec[(size_t)b*hv+h] = qsoftplus(c->alpha[(size_t)b*hv+h] + dt[h]) * aa[h];
            c->beta[(size_t)b*hv+h] = qsigmoid(c->beta[(size_t)b*hv+h]);
        }

    qssm_ud u = { c, il, nb };
    g4_parallel_for(conv_dim, qconv_kernel, &u);                  // conv + SiLU

    for (int b = 0; b < nb; b++) {                               // L2-norm q/k per head
        float * conv = c->convo + (size_t)b * conv_dim;
        for (int h = 0; h < hk; h++) { l2_norm(conv + (size_t)h*sv, sv, eps); l2_norm(conv + key_dim + (size_t)h*sv, sv, eps); }
    }

    g4_parallel_for(hv, qscan_kernel, &u);                       // recurrent scan

    for (int b = 0; b < nb; b++) {                               // gated RMS norm
        float * a = c->attno + (size_t)b * vd;
        const float * z = c->z + (size_t)b * vd;
        for (int h = 0; h < hv; h++) {
            rms_norm(a + (size_t)h*sv, (const float *)L->ssm_norm->data, a + (size_t)h*sv, sv, eps);
            for (int i = 0; i < sv; i++) a[(size_t)h*sv+i] *= qsilu(z[(size_t)h*sv+i]);
        }
    }
    QMM(c, L->ssm_out, c->attno, c->cur, vd, n_embd, nb);
}

// batched forward: tokens[0..nb) at positions pos0..pos0+nb-1; logits of last token
// embed tokens then run all layers; final hidden left in c->x (token-major [nb][n_embd])
static void qwen_run_layers(qwen_ctx * c, const int * tokens, int nb, int pos0) {
    g4_model_file * mf = c->mf;
    const int n_embd = mf->n_embd, n_ff = c->n_ff;
    const float eps = mf->rms_eps;
    const size_t erb = g4_row_size(c->tok_embd->type, n_embd);

    for (int b = 0; b < nb; b++)
        g4_dequant_row(c->tok_embd->type, (const uint8_t *)c->tok_embd->data + (size_t)tokens[b]*erb,
                       c->x + (size_t)b*n_embd, n_embd);

    for (int il = 0; il < mf->n_layer; il++) {
        const qwen_layer * L = &c->layers[il];
        memcpy(c->tmp, c->x, (size_t)nb * n_embd * 4);
        for (int b = 0; b < nb; b++) rms_norm(c->x + (size_t)b*n_embd, (const float *)L->attn_norm->data, c->cur + (size_t)b*n_embd, n_embd, eps);
        if (L->is_recr) qwen_ssm_batch(c, il, nb);
        else            qwen_attn_batch(c, il, pos0, nb);
        for (int64_t i = 0; i < (int64_t)nb*n_embd; i++) c->x[i] = c->tmp[i] + c->cur[i];

        memcpy(c->tmp, c->x, (size_t)nb * n_embd * 4);
        for (int b = 0; b < nb; b++) rms_norm(c->x + (size_t)b*n_embd, (const float *)L->attn_post_norm->data, c->cur + (size_t)b*n_embd, n_embd, eps);
        // ffn_gate and ffn_up share c->cur — prepare the activation once and reuse.
        if (act_class(L->ffn_gate->type) == act_class(L->ffn_up->type)) {
            size_t ast; const void * pa = prep_act(&c->mm, act_class(L->ffn_gate->type), c->cur, n_embd, nb, &ast);
            matmul_b(&c->mm, L->ffn_gate->type, L->ffn_gate->data, c->cur, pa, ast, c->ffh,  n_embd, n_ff, nb);
            matmul_b(&c->mm, L->ffn_up->type,   L->ffn_up->data,   c->cur, pa, ast, c->ffh2, n_embd, n_ff, nb);
        } else {
            QMM(c, L->ffn_gate, c->cur, c->ffh,  n_embd, n_ff, nb);
            QMM(c, L->ffn_up,   c->cur, c->ffh2, n_embd, n_ff, nb);
        }
        for (int64_t i = 0; i < (int64_t)nb*n_ff; i++) c->ffh[i] = qsilu(c->ffh[i]) * c->ffh2[i];
        QMM(c, L->ffn_down, c->ffh, c->cur, n_ff, n_embd, nb);
        for (int64_t i = 0; i < (int64_t)nb*n_embd; i++) c->x[i] = c->tmp[i] + c->cur[i];
    }
}

float * g4_qwen_forward_batch(qwen_ctx * c, const int * tokens, int nb, int pos0, bool want_logits) {
    g4_model_file * mf = c->mf;
    const int n_embd = mf->n_embd;
    const float eps = mf->rms_eps;
    qwen_run_layers(c, tokens, nb, pos0);
    if (!want_logits) return NULL;
    float * xl = c->x + (size_t)(nb - 1)*n_embd;
    rms_norm(xl, (const float *)c->output_norm->data, c->cur, n_embd, eps);
    QMM(c, c->output, c->cur, c->logits, n_embd, mf->n_vocab, 1);
    return c->logits;
}

float * g4_qwen_forward(qwen_ctx * c, int token, int pos) {
    return g4_qwen_forward_batch(c, &token, 1, pos, true);
}

// ---- speculative decoding (prompt-lookup) support -------------------------
// Allocate per-token recurrent-state checkpoints for up to kmax+1 batch positions.
int g4_qwen_spec_enable(qwen_ctx * c, int kmax) {
    const int slots = kmax + 1;
    c->ck_stride_conv = (size_t)c->conv_dim * (c->mf->ssm_d_conv - 1);
    c->ck_stride_ssm  = (size_t)c->s_v * c->s_v * c->h_v;
    for (int il = 0; il < c->mf->n_layer; il++) {
        if (!c->layers[il].is_recr) continue;
        c->ck_conv[il] = malloc((size_t)slots * c->ck_stride_conv * sizeof(float));
        c->ck_ssm[il]  = malloc((size_t)slots * c->ck_stride_ssm  * sizeof(float));
        if (!c->ck_conv[il] || !c->ck_ssm[il]) { fprintf(stderr, "g4: qwen spec ckpt alloc failed\n"); return 1; }
    }
    c->ck_cap = slots;
    return 0;
}

// Verify forward: writes per-token recurrent checkpoints (so rejected drafts can be
// rolled back) and returns per-position logits, token-major [nb][n_vocab].
float * g4_qwen_forward_spec(qwen_ctx * c, const int * tokens, int nb, int pos0, float * logits_out) {
    g4_model_file * mf = c->mf;
    const int n_embd = mf->n_embd;
    const float eps = mf->rms_eps;
    c->ck_on = 1;
    qwen_run_layers(c, tokens, nb, pos0);
    c->ck_on = 0;
    for (int b = 0; b < nb; b++)
        rms_norm(c->x + (size_t)b*n_embd, (const float *)c->output_norm->data,
                 c->cur + (size_t)b*n_embd, n_embd, eps);
    QMM(c, c->output, c->cur, logits_out, n_embd, mf->n_vocab, nb);
    return logits_out;
}

// Roll recurrent conv/SSM state back to the checkpoint captured after batch index idx.
void g4_qwen_state_restore(qwen_ctx * c, int idx) {
    for (int il = 0; il < c->mf->n_layer; il++) {
        if (!c->layers[il].is_recr) continue;
        memcpy(c->conv_state[il], c->ck_conv[il] + (size_t)idx * c->ck_stride_conv, c->ck_stride_conv * sizeof(float));
        memcpy(c->ssm_state[il],  c->ck_ssm[il]  + (size_t)idx * c->ck_stride_ssm,  c->ck_stride_ssm  * sizeof(float));
    }
}
