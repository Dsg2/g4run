// quants.c — quant block kernels, ported from llama.cpp ggml (ref/ in repo)
// Scalar versions mirror ggml generic kernels exactly; AVX2 paths mirror x86 kernels.
#include "g4run.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// 512-bit VNNI path for q8_0/q4_0, toggled at runtime via --avx512 / --no-avx512.
// DEFAULT OFF: measured ~43% SLOWER than AVX2 on Tiger Lake (i5-1135G7) — the
// interleaved per-block layout forces cross-lane insert/extract to pack two blocks
// into a 512-bit register, and that overhead plus AVX-512 downclocking outweighs
// the wider MAC. (ggml gets its AVX-512 win from *repacked* weight layouts, not
// wider per-block loops.) Kept as an opt-in so it can be tried on other CPUs.
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#define G4_HAS_AVX512 1
#else
#define G4_HAS_AVX512 0
#endif
int g4_avx512 = 0;   // opt-in (see above); AVX2 is the faster default here
void g4_set_avx512(int on) { g4_avx512 = on && G4_HAS_AVX512; }
int  g4_avx512_available(void) { return G4_HAS_AVX512; }

float   g4_fp16_to_fp32_table[65536];
g4_fp16 g4_gelu_fp16_table[65536];

static float fp16_to_fp32_compute(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (!mant) f = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float r; memcpy(&r, &f, 4); return r;
}

// exact tanh-approx gelu (ggml_gelu_f32)
static float gelu_exact(float x) {
    return 0.5f*x*(1.0f + tanhf(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x)));
}

void g4_init_tables(void) {
    for (uint32_t i = 0; i < 65536; i++) {
        float f = fp16_to_fp32_compute((uint16_t)i);
        g4_fp16_to_fp32_table[i] = f;
        g4_gelu_fp16_table[i] = g4_fp32_to_fp16(gelu_exact(f));
    }
}

// ggml nearest_int: round-half-away-from-zero via magic constant... actually
// the 12582912.0f trick rounds to nearest (ties to even, matches FPU default)
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// ------------------------------------------------------ activation quant ----

void g4_quantize_q8_0(const float * x, block_q8_0 * y, int64_t n) {
    const int64_t nb = n / QK8_0;
#if defined(__AVX2__)
    // mirrors ggml AVX2 quantize_row_q8_0 (round-to-nearest-even via _mm256_round_ps)
    for (int64_t i = 0; i < nb; i++) {
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        const float d  = maxScalar / 127.f;
        y[i].d = g4_fp32_to_fp16(d);
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);
        v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
    }
#else
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) { float ax = fabsf(x[j]); if (ax > amax) amax = ax; }
        const float d  = amax / 127.f;
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = g4_fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; j++) y[i].qs[j] = (int8_t)nearest_int(x[j]*id);
        x += QK8_0;
    }
#endif
}

// q4_0 quantization (ggml quantize_row_q4_0_ref) — used for the Q4_0 KV cache.
// Pairing matches dq_q4_0: qs[j] low nibble = elem j, high nibble = elem j+QK4_0/2.
void g4_quantize_q4_0(const float * x, block_q4_0 * y, int64_t n) {
    const int64_t nb = n / QK4_0;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            const float v = x[i*QK4_0 + j];
            if (amax < fabsf(v)) { amax = fabsf(v); max = v; }
        }
        const float d  = max / -8.0f;
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = g4_fp32_to_fp16(d);
        for (int j = 0; j < QK4_0/2; j++) {
            const float x0 = x[i*QK4_0 + j]          * id;
            const float x1 = x[i*QK4_0 + QK4_0/2 + j]* id;
            uint8_t xi0 = (uint8_t)(x0 + 8.5f); if (xi0 > 15) xi0 = 15;
            uint8_t xi1 = (uint8_t)(x1 + 8.5f); if (xi1 > 15) xi1 = 15;
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

void g4_quantize_q8_K(const float * x, block_q8_K * y, int64_t n) {
    // ggml uses the reference (scalar) implementation even on x86
    const int64_t nb = n / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        float max = 0, amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = (int8_t)(v < 127 ? v : 127);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j*16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

// -------------------------------------------- K-quant quantizers (f32->Qx) --
// Ports of ggml's quantize_row_q2_K_ref / quantize_row_q3_K_ref (+ the
// make_qkx2_quants / make_q3_quants scale solvers) so JIT requant can target
// Q2_K and Q3_K. The non-imatrix reference path (uniform / x^2 weights).
#define G4_QMAX(a,b) ((a) > (b) ? (a) : (b))
#define G4_QMIN(a,b) ((a) < (b) ? (a) : (b))
#define G4_GROUP_MAX_EPS 1e-15f

// asymmetric scale+min search (Q2_K sub-blocks)
static float g4_make_qkx2(int n, int nmax, const float * x, const float * weights,
                          uint8_t * L, float * the_min, uint8_t * Laux,
                          float rmin, float rdelta, int nstep, int use_mad) {
    float min = x[0], max = x[0];
    float sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i]; sum_w += w; sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) { for (int i = 0; i < n; ++i) L[i] = 0; *the_min = -min; return 0.f; }
    float iscale = nmax/(max - min), scale = 1/iscale, best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = G4_QMAX(0, G4_QMIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) { *the_min = -min; return scale; }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = G4_QMAX(0, G4_QMIN(nmax, nearest_int(iscale*(x[i] - min))));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l; sum_l2 += w*l*l; sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error; scale = this_scale; min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// symmetric scale search with RMSE refinement (Q3_K sub-blocks)
static float g4_make_q3(int n, int nmax, const float * x, int8_t * L) {
    float max = 0, amax = 0;
    for (int i = 0; i < n; ++i) { float ax = fabsf(x[i]); if (ax > amax) { amax = ax; max = x[i]; } }
    if (amax < G4_GROUP_MAX_EPS) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = -nmax / max;
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = G4_QMAX(-nmax, G4_QMIN(nmax-1, nearest_int(iscale * x[i])));
        L[i] = l;
        float w = x[i]*x[i];
        sumlx += w*x[i]*l; suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = x[i]*x[i];
            float slx = sumlx - w*x[i]*L[i];
            if (slx > 0) {
                float sl2 = suml2 - w*L[i]*L[i];
                int new_l = G4_QMAX(-nmax, G4_QMIN(nmax-1, nearest_int(x[i] * sl2 / slx)));
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l; sl2 += w*new_l*new_l;
                    if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2; ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    for (int i = 0; i < n; ++i) L[i] += nmax;
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

void g4_quantize_q2_K(const float * x, block_q2_K * y, int64_t k) {
    const int64_t nb = k / QK_K;
    uint8_t L[QK_K], Laux[16];
    float weights[16], mins[QK_K/16], scales[QK_K/16];
    const float q4scale = 15.f;
    for (int64_t i = 0; i < nb; i++) {
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = g4_make_qkx2(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, 1);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j]   > max_min)   max_min   = mins[j];
        }
        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = nearest_int(iscale*scales[j]);
            y[i].d = g4_fp32_to_fp16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = g4_fp32_to_fp16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) { int l = nearest_int(iscale*mins[j]); y[i].scales[j] |= (l << 4); }
            y[i].dmin = g4_fp32_to_fp16(max_min/q4scale);
        } else {
            y[i].dmin = g4_fp32_to_fp16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = g4_fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = g4_fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii)
                L[16*j + ii] = G4_QMAX(0, G4_QMIN(3, nearest_int((x[16*j + ii] + dm)/d)));
        }
        for (int j = 0; j < QK_K; j += 128)
            for (int l = 0; l < 32; ++l)
                y[i].qs[j/4 + l] = L[j+l] | (L[j+l+32] << 2) | (L[j+l+64] << 4) | (L[j+l+96] << 6);
        x += QK_K;
    }
}

void g4_quantize_q3_K(const float * x, block_q3_K * y, int64_t k) {
    const int64_t nb = k / QK_K;
    int8_t L[QK_K];
    float scales[QK_K/16];
    for (int64_t i = 0; i < nb; i++) {
        float max_scale = 0, amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = g4_make_q3(16, 4, x + 16*j, L + 16*j);
            float scale = fabsf(scales[j]);
            if (scale > amax) { amax = scale; max_scale = scales[j]; }
        }
        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = G4_QMAX(-32, G4_QMIN(31, l)) + 32;
                if (j < 8) y[i].scales[j] = l & 0xF;
                else       y[i].scales[j-8] |= ((l & 0xF) << 4);
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = g4_fp32_to_fp16(1/iscale);
        } else {
            y[i].d = g4_fp32_to_fp16(0.f);
        }
        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = g4_fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii)
                L[16*j + ii] = G4_QMAX(-4, G4_QMIN(3, nearest_int(x[16*j + ii]/d))) + 4;
        }
        memset(y[i].hmask, 0, QK_K/8);
        int m = 0; uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) { y[i].hmask[m] |= hm; L[j] -= 4; }
            if (++m == QK_K/8) { m = 0; hm <<= 1; }
        }
        for (int j = 0; j < QK_K; j += 128)
            for (int l = 0; l < 32; ++l)
                y[i].qs[j/4 + l] = L[j+l] | (L[j+l+32] << 2) | (L[j+l+64] << 4) | (L[j+l+96] << 6);
        x += QK_K;
    }
}

// ----------------------------------------------------------- dequant row ----

static void dq_q4_0(const block_q4_0 * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n/QK4_0; i++) {
        const float d = g4_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK4_0/2; ++j) {
            y[i*QK4_0 + j]          = ((int)(x[i].qs[j] & 0x0F) - 8) * d;
            y[i*QK4_0 + j + QK4_0/2] = ((int)(x[i].qs[j] >> 4) - 8) * d;
        }
    }
}
static void dq_q8_0(const block_q8_0 * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n/QK8_0; i++) {
        const float d = g4_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i*QK8_0 + j] = x[i].qs[j] * d;
    }
}
static void dq_q6_K(const block_q6_K * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n/QK_K; i++) {
        const float d = g4_fp16_to_fp32(x[i].d);
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sc = x[i].scales;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64; qh += 32; sc += 8;
        }
    }
}
static void dq_tq2_0(const block_tq2_0 * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n/QK_K; i++) {
        const float d = g4_fp16_to_fp32(x[i].d);
        for (size_t j = 0; j < QK_K/4; j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float)(q - 1) * d;
                }
            }
        }
    }
}

// K-quant scale/min unpacking (ggml get_scale_min_k4): 6-bit scale+min from 12 bytes
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void dq_q2_K(const block_q2_K * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n/QK_K; i++) {
        const float d   = g4_fp16_to_fp32(x[i].d);
        const float min = g4_fp16_to_fp32(x[i].dmin);
        const uint8_t * q = x[i].qs;
        int is = 0; float dl, ml;
        for (int nn = 0; nn < QK_K; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void dq_q3_K(const block_q3_K * x, float * y, int64_t n) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t * scales = (const int8_t *)aux;
    for (int64_t i = 0; i < n/QK_K; i++) {
        const float d_all = g4_fp16_to_fp32(x[i].d);
        const uint8_t * q = x[i].qs;
        const uint8_t * hm = x[i].hmask;
        uint8_t m = 1;
        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0; float dl;
        for (int nn = 0; nn < QK_K; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l+0] >> shift) & 3) - ((hm[l+0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                shift += 2; m <<= 1;
            }
            q += 32;
        }
    }
}

static void dq_q4_K(const block_q4_K * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n/QK_K; i++) {
        const uint8_t * q = x[i].qs;
        const float d   = g4_fp16_to_fp32(x[i].d);
        const float min = g4_fp16_to_fp32(x[i].dmin);
        int is = 0; uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void dq_q5_K(const block_q5_K * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n/QK_K; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const float d   = g4_fp16_to_fp32(x[i].d);
        const float min = g4_fp16_to_fp32(x[i].dmin);
        int is = 0; uint8_t sc, m; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >>  4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}

void g4_dequant_row(int type, const void * x, float * y, int64_t n) {
    switch (type) {
        case G4_F32:  memcpy(y, x, n*4); break;
        case G4_F16:  { const g4_fp16 * h = x; for (int64_t i = 0; i < n; i++) y[i] = g4_fp16_to_fp32(h[i]); } break;
        case G4_BF16: { const g4_bf16 * h = x; for (int64_t i = 0; i < n; i++) y[i] = g4_bf16_to_fp32(h[i]); } break;
        case G4_Q4_0:  dq_q4_0(x, y, n); break;
        case G4_Q8_0:  dq_q8_0(x, y, n); break;
        case G4_Q6_K:  dq_q6_K(x, y, n); break;
        case G4_TQ2_0: dq_tq2_0(x, y, n); break;
        case G4_Q2_K:  dq_q2_K(x, y, n); break;
        case G4_Q3_K:  dq_q3_K(x, y, n); break;
        case G4_Q4_K:  dq_q4_K(x, y, n); break;
        case G4_Q5_K:  dq_q5_K(x, y, n); break;
        default: fprintf(stderr, "g4: dequant type %d unsupported\n", type); exit(1);
    }
}

// -------------------------------------------------------- scalar ref dots ---

float g4_dot_q4_0_q8_0_ref(const void * vx, const void * vy, int64_t n) {
    const block_q4_0 * x = vx; const block_q8_0 * y = vy;
    const int64_t nb = n / QK8_0;
    float sumf = 0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < QK8_0/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + QK8_0/2];
        }
        sumf += (sumi0 + sumi1) * g4_fp16_to_fp32(x[ib].d) * g4_fp16_to_fp32(y[ib].d);
    }
    return sumf;
}

float g4_dot_q8_0_q8_0_ref(const void * vx, const void * vy, int64_t n) {
    const block_q8_0 * x = vx; const block_q8_0 * y = vy;
    const int64_t nb = n / QK8_0;
    float sumf = 0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) sumi += x[ib].qs[j] * y[ib].qs[j];
        sumf += sumi * (g4_fp16_to_fp32(x[ib].d) * g4_fp16_to_fp32(y[ib].d));
    }
    return sumf;
}

float g4_dot_tq2_0_q8_K_ref(const void * vx, const void * vy, int64_t n) {
    const block_tq2_0 * x = vx; const block_q8_K * y = vy;
    const int64_t nb = n / QK_K;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        int32_t sumi = 0;
        for (size_t j = 0; j < QK_K/4; j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }
        sumf += (float)sumi * (y[i].d * g4_fp16_to_fp32(x[i].d));
    }
    return sumf;
}

float g4_dot_q6_K_q8_K_ref(const void * vx, const void * vy, int64_t n) {
    const block_q6_K * x = vx; const block_q8_K * y = vy;
    const int64_t nb = n / QK_K;
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    float sumf = 0;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t * q4 = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t * a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128; q4 += 64; qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = g4_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

float g4_dot_q2_K_q8_K_ref(const void * vx, const void * vy, int64_t n) {
    const block_q2_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;
        int summs = 0;
        for (int j = 0; j < 16; ++j) summs += y[i].bsums[j] * (sc[j] >> 4);
        const float dall = y[i].d * g4_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * g4_fp16_to_fp32(x[i].dmin);
        int isum = 0, is = 0, d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2; q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    return sumf;
}

float g4_dot_q3_K_q8_K_ref(const void * vx, const void * vy, int64_t n) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const block_q3_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    uint32_t auxs[4];
    const int8_t * scales = (const int8_t *)auxs;
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q3 = x[i].qs;
        const uint8_t * hm = x[i].hmask;
        const  int8_t * q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t * a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;
        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = g4_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

float g4_dot_q4_K_q8_K_ref(const void * vx, const void * vy, int64_t n) {
    const block_q4_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t *)&utmp[0];
    const uint8_t * mins   = (const uint8_t *)&utmp[2];
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q4 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t * a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = g4_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = g4_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

float g4_dot_q5_K_q8_K_ref(const void * vx, const void * vy, int64_t n) {
    const block_q5_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t *)&utmp[0];
    const uint8_t * mins   = (const uint8_t *)&utmp[2];
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, sizeof(sums));
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q4 = x[i].qs;
        const uint8_t * hm = x[i].qh;
        const  int8_t * q8 = y[i].qs;
        memset(aux32, 0, sizeof(aux32));
        int8_t * a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = g4_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = g4_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

// ------------------------------------------------------------ AVX2 dots -----
#if defined(__AVX2__)

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// dot of unsigned x with signed y, 32 bytes, result as i32x8
static inline __m256i mul_sum_us8_pairs_epi32(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_dpbusd_epi32(zero, ax, sy);
#else
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, dot);
#endif
}

static inline __m256i mul_sum_i8_pairs_epi32(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_epi32(ax, sy);
}

#if G4_HAS_AVX512
// signed*signed int8 dot, 64 lanes -> 16 i32 (lanes 0-7 = low 256, 8-15 = high 256).
// AVX-512 has no _mm512_sign_epi8, so emulate it: ax=|bx|, sy = by with the sign of bx,
// via the sign-bit mask of bx + a masked negate. Then VNNI dpbusd(ax, sy) = sum(bx*by).
static inline __m512i g4_dp_i8x64(const __m512i bx, const __m512i by) {
    const __mmask64 sx = _mm512_movepi8_mask(bx);
    const __m512i   ax = _mm512_abs_epi8(bx);
    const __m512i   sy = _mm512_mask_sub_epi8(by, sx, _mm512_setzero_si512(), by);
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), ax, sy);
}
static inline __m256i g4_unpack_q4_0(const uint8_t * qs) {   // 16 bytes -> 32 int8 (-8..7)
    const __m128i lo = _mm_loadu_si128((const __m128i *)qs);
    __m256i bx = _mm256_set_m128i(_mm_srli_epi16(lo, 4), lo);
    bx = _mm256_and_si256(bx, _mm256_set1_epi8(0xF));
    return _mm256_sub_epi8(bx, _mm256_set1_epi8(8));
}
// scale the two 8-lane halves of a dp result by their per-block d and accumulate
#define G4_ACC2(ACC0, ACC1, D, DA, DB) do { \
    ACC0 = _mm256_fmadd_ps(_mm256_set1_ps(DA), _mm256_cvtepi32_ps(_mm512_castsi512_si256(D)), ACC0); \
    ACC1 = _mm256_fmadd_ps(_mm256_set1_ps(DB), _mm256_cvtepi32_ps(_mm512_extracti64x4_epi64(D, 1)), ACC1); \
} while (0)

static float dot_q8_0_q8_0_avx512(const block_q8_0 * x, const block_q8_0 * y, int64_t nb) {
    __m256 acc = _mm256_setzero_ps();
    int64_t ib = 0;
    for (; ib + 2 <= nb; ib += 2) {
        const __m512i bx = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)x[ib].qs)),
                                              _mm256_loadu_si256((const __m256i *)x[ib+1].qs), 1);
        const __m512i by = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)y[ib].qs)),
                                              _mm256_loadu_si256((const __m256i *)y[ib+1].qs), 1);
        const __m512i d = g4_dp_i8x64(bx, by);
        G4_ACC2(acc, acc, d, g4_fp16_to_fp32(x[ib].d)*g4_fp16_to_fp32(y[ib].d),
                            g4_fp16_to_fp32(x[ib+1].d)*g4_fp16_to_fp32(y[ib+1].d));
    }
    float sum = hsum_float_8(acc);
    for (; ib < nb; ++ib) {
        const __m256i i32 = mul_sum_i8_pairs_epi32(_mm256_loadu_si256((const __m256i *)x[ib].qs),
                                                   _mm256_loadu_si256((const __m256i *)y[ib].qs));
        sum += hsum_float_8(_mm256_mul_ps(_mm256_set1_ps(g4_fp16_to_fp32(x[ib].d)*g4_fp16_to_fp32(y[ib].d)),
                                          _mm256_cvtepi32_ps(i32)));
    }
    return sum;
}
static float dot_q4_0_q8_0_avx512(const block_q4_0 * x, const block_q8_0 * y, int64_t nb) {
    __m256 acc = _mm256_setzero_ps();
    int64_t ib = 0;
    for (; ib + 2 <= nb; ib += 2) {
        const __m512i bx = _mm512_inserti64x4(_mm512_castsi256_si512(g4_unpack_q4_0(x[ib].qs)),
                                              g4_unpack_q4_0(x[ib+1].qs), 1);
        const __m512i by = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)y[ib].qs)),
                                              _mm256_loadu_si256((const __m256i *)y[ib+1].qs), 1);
        const __m512i d = g4_dp_i8x64(bx, by);
        G4_ACC2(acc, acc, d, g4_fp16_to_fp32(x[ib].d)*g4_fp16_to_fp32(y[ib].d),
                            g4_fp16_to_fp32(x[ib+1].d)*g4_fp16_to_fp32(y[ib+1].d));
    }
    float sum = hsum_float_8(acc);
    for (; ib < nb; ++ib) {
        const __m256i i32 = mul_sum_i8_pairs_epi32(g4_unpack_q4_0(x[ib].qs), _mm256_loadu_si256((const __m256i *)y[ib].qs));
        sum += hsum_float_8(_mm256_mul_ps(_mm256_set1_ps(g4_fp16_to_fp32(x[ib].d)*g4_fp16_to_fp32(y[ib].d)),
                                          _mm256_cvtepi32_ps(i32)));
    }
    return sum;
}
// x4: one weight block vs 4 q8_0 activations; process activation pairs (0,1)/(2,3)
// with the weight broadcast into both 512-bit halves.
static void dot_x4_avx512(const __m256i wb, float dx, const block_q8_0 * const a[4], int64_t ib,
                          __m256 * c0, __m256 * c1, __m256 * c2, __m256 * c3) {
    const __m512i bx = _mm512_broadcast_i64x4(wb);
    const __m512i d01 = g4_dp_i8x64(bx, _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a[0][ib].qs)),
        _mm256_loadu_si256((const __m256i *)a[1][ib].qs), 1));
    G4_ACC2(*c0, *c1, d01, dx*g4_fp16_to_fp32(a[0][ib].d), dx*g4_fp16_to_fp32(a[1][ib].d));
    const __m512i d23 = g4_dp_i8x64(bx, _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a[2][ib].qs)),
        _mm256_loadu_si256((const __m256i *)a[3][ib].qs), 1));
    G4_ACC2(*c2, *c3, d23, dx*g4_fp16_to_fp32(a[2][ib].d), dx*g4_fp16_to_fp32(a[3][ib].d));
}
static void dot_q8_0_q8_0_x4_avx512(const block_q8_0 * x, const block_q8_0 * const a[4], int64_t nb, float out[4]) {
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    for (int64_t ib = 0; ib < nb; ++ib)
        dot_x4_avx512(_mm256_loadu_si256((const __m256i *)x[ib].qs), g4_fp16_to_fp32(x[ib].d), a, ib, &c0, &c1, &c2, &c3);
    out[0]=hsum_float_8(c0); out[1]=hsum_float_8(c1); out[2]=hsum_float_8(c2); out[3]=hsum_float_8(c3);
}
static void dot_q4_0_q8_0_x4_avx512(const block_q4_0 * x, const block_q8_0 * const a[4], int64_t nb, float out[4]) {
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    for (int64_t ib = 0; ib < nb; ++ib)
        dot_x4_avx512(g4_unpack_q4_0(x[ib].qs), g4_fp16_to_fp32(x[ib].d), a, ib, &c0, &c1, &c2, &c3);
    out[0]=hsum_float_8(c0); out[1]=hsum_float_8(c1); out[2]=hsum_float_8(c2); out[3]=hsum_float_8(c3);
}
#endif // G4_HAS_AVX512

// K-quant scale broadcast shuffles (ggml cpu-quants-x86.c)
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

float g4_dot_q4_0_q8_0(const void * vx, const void * vy, int64_t n) {
    const block_q4_0 * x = vx; const block_q8_0 * y = vy;
    const int64_t nb = n / QK8_0;
#if G4_HAS_AVX512
    if (g4_avx512) return dot_q4_0_q8_0_avx512(x, y, nb);
#endif
    __m256 acc = _mm256_setzero_ps();
    for (int64_t ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(g4_fp16_to_fp32(x[ib].d) * g4_fp16_to_fp32(y[ib].d));
        const __m128i lo = _mm_loadu_si128((const __m128i *)x[ib].qs);
        __m256i bx = _mm256_set_m128i(_mm_srli_epi16(lo, 4), lo);
        bx = _mm256_and_si256(bx, _mm256_set1_epi8(0xF));
        bx = _mm256_sub_epi8(bx, _mm256_set1_epi8(8));
        const __m256i by = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256i i32 = mul_sum_i8_pairs_epi32(bx, by);
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(i32), acc);
    }
    return hsum_float_8(acc);
}

float g4_dot_q8_0_q8_0(const void * vx, const void * vy, int64_t n) {
    const block_q8_0 * x = vx; const block_q8_0 * y = vy;
    const int64_t nb = n / QK8_0;
#if G4_HAS_AVX512
    if (g4_avx512) return dot_q8_0_q8_0_avx512(x, y, nb);
#endif
    __m256 acc = _mm256_setzero_ps();
    for (int64_t ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(g4_fp16_to_fp32(x[ib].d) * g4_fp16_to_fp32(y[ib].d));
        const __m256i bx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        const __m256i by = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256i i32 = mul_sum_i8_pairs_epi32(bx, by);
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(i32), acc);
    }
    return hsum_float_8(acc);
}

float g4_dot_tq2_0_q8_K(const void * vx, const void * vy, int64_t n) {
    // trits stored as 0/1/2 (unsigned); dot = dpbusd(trits, q8) - sum(q8) via bsums
    const block_tq2_0 * x = vx; const block_q8_K * y = vy;
    const int64_t nb = n / QK_K;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        __m256i isum = _mm256_setzero_si256();
        for (int j = 0; j < QK_K/4; j += 32) {
            const __m256i qbits = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
            for (int l = 0; l < 4; ++l) {
                const __m256i tr = _mm256_and_si256(_mm256_srli_epi16(qbits, 2*l), _mm256_set1_epi8(3));
                const __m256i q8 = _mm256_loadu_si256((const __m256i *)(y[i].qs + j*4 + l*32));
                isum = _mm256_add_epi32(isum, mul_sum_us8_pairs_epi32(tr, q8));
            }
        }
        // subtract sum of q8 (because trits are offset by +1)
        int32_t bsum = 0;
        for (int j = 0; j < QK_K/16; ++j) bsum += y[i].bsums[j];
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(isum), _mm256_extracti128_si256(isum, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        const int32_t sumi = _mm_cvtsi128_si32(s);
        sumf += (float)(sumi - bsum) * (y[i].d * g4_fp16_to_fp32(x[i].d));
    }
    return sumf;
}

float g4_dot_q6_K_q8_K(const void * vx, const void * vy, int64_t n) {
    const block_q6_K * x = vx; const block_q8_K * y = vy;
    const int64_t nb = n / QK_K;
    const __m256i m4  = _mm256_set1_epi8(0xF);
    const __m256i m2  = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
        const float d = y[i].d * g4_fp16_to_fp32(x[i].d);
        const uint8_t * q4 = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * q8 = y[i].qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh); qh += 32;

            const __m256i q4h0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            // per-16 scales: scales[j*8 + k] applies to 16 values
            const int8_t * sc = x[i].scales + j*8;
            const __m256i sc01 = _mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]));
            const __m256i sc23 = _mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]));
            const __m256i sc45 = _mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]));
            const __m256i sc67 = _mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]));

            p16_0 = _mm256_madd_epi16(sc01, p16_0);
            p16_1 = _mm256_madd_epi16(sc23, p16_1);
            p16_2 = _mm256_madd_epi16(sc45, p16_2);
            p16_3 = _mm256_madd_epi16(sc67, p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8(acc);
}

// AVX2 K-quant dots — mirror ggml cpu-quants-x86.c so accumulation order (and
// hence rounding) matches llama.cpp's CPU path for token-exact parity.
float g4_dot_q2_K_q8_K(const void * vx, const void * vy, int64_t n) {
    const block_q2_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d    =  y[i].d * g4_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * g4_fp16_to_fp32(x[i].dmin);
        const uint8_t * q2 = x[i].qs;
        const int8_t  * q8 = y[i].qs;
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i *)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8   = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y[i].bsums));
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = { _mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales) };
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i *)q2); q2 += 32;
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);
            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);
            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);
            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8(acc);
}

float g4_dot_q3_K_q8_K(const void * vx, const void * vy, int64_t n) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const block_q3_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    uint32_t aux[3];
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * g4_fp16_to_fp32(x[i].d);
        const uint8_t * q3 = x[i].qs;
        const int8_t  * q8 = y[i].qs;
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = { _mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales) };
        const __m256i hbits = _mm256_loadu_si256((const __m256i *)x[i].hmask);
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0, is = 0;
        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i *)q3); q3 += 32;
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2); ++bit;
            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2); ++bit;
            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2); ++bit;
            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2); ++bit;
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);
            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);
            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);
            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);
            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi  = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
        }
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8(acc);
}

float g4_dot_q4_K_q8_K(const void * vx, const void * vy, int64_t n) {
    const block_q4_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d    =  y[i].d * g4_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * g4_fp16_to_fp32(x[i].dmin);
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const uint8_t * q4 = x[i].qs;
        const int8_t  * q8 = y[i].qs;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);
        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            const __m256i q8l = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);
            const __m256i q8h = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    return hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

float g4_dot_q5_K_q8_K(const void * vx, const void * vy, int64_t n) {
    const block_q5_K * x = vx; const block_q8_K * y = vy;
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q5 = x[i].qs;
        const int8_t  * q8 = y[i].qs;
        const float d    =  y[i].d * g4_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * g4_fp16_to_fp32(x[i].dmin);
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);
        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);
        const __m256i hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
        __m256i hmask = mone;
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;
        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));
            const __m256i q5bits = _mm256_loadu_si256((const __m256i *)q5); q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8(acc) + summs;
}

float g4_dot_f16(const void * vw, const g4_fp16 * a, int64_t n) {
    const g4_fp16 * w = vw;
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 wf = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i)));
        const __m256 af = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i)));
        acc = _mm256_fmadd_ps(wf, af, acc);
    }
    float s = hsum_float_8(acc);
    for (; i < n; i++) s += g4_fp16_to_fp32(w[i]) * g4_fp16_to_fp32(a[i]);
    return s;
}

float g4_dot_bf16(const void * vw, const float * a, int64_t n) {
    const g4_bf16 * w = vw;
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // bf16 -> f32: zero-extend to 32 and shift left 16
        const __m128i raw = _mm_loadu_si128((const __m128i *)(w + i));
        const __m256i ext = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
        const __m256 wf = _mm256_castsi256_ps(ext);
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(a + i), acc);
    }
    float s = hsum_float_8(acc);
    for (; i < n; i++) s += g4_bf16_to_fp32(w[i]) * a[i];
    return s;
}

// ---------------------------------------------------------- x4 batch dots ---
// Same per-token accumulation order as the 1-token kernels (bit-identical
// results); the weight unpack work is shared across 4 activations.

void g4_dot_q4_0_q8_0_x4(const void * vx, const void * const va[4], int64_t n, float out[4]) {
    const block_q4_0 * x = vx;
    const block_q8_0 * a0 = va[0], * a1 = va[1], * a2 = va[2], * a3 = va[3];
    const int64_t nb = n / QK8_0;
#if G4_HAS_AVX512
    if (g4_avx512) { dot_q4_0_q8_0_x4_avx512(x, (const block_q8_0 * const *)va, nb, out); return; }
#endif
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float dx = g4_fp16_to_fp32(x[ib].d);
        const __m128i lo = _mm_loadu_si128((const __m128i *)x[ib].qs);
        __m256i bx = _mm256_set_m128i(_mm_srli_epi16(lo, 4), lo);
        bx = _mm256_and_si256(bx, _mm256_set1_epi8(0xF));
        bx = _mm256_sub_epi8(bx, _mm256_set1_epi8(8));
        const __m256i ax = _mm256_sign_epi8(bx, bx);
        #define ONE(K, A) do { \
            const __m256i by = _mm256_loadu_si256((const __m256i *)A[ib].qs); \
            const __m256i sy = _mm256_sign_epi8(by, bx); \
            const __m256i i32 = mul_sum_us8_pairs_epi32(ax, sy); \
            c##K = _mm256_fmadd_ps(_mm256_set1_ps(dx * g4_fp16_to_fp32(A[ib].d)), _mm256_cvtepi32_ps(i32), c##K); \
        } while (0)
        ONE(0, a0); ONE(1, a1); ONE(2, a2); ONE(3, a3);
        #undef ONE
    }
    out[0] = hsum_float_8(c0); out[1] = hsum_float_8(c1);
    out[2] = hsum_float_8(c2); out[3] = hsum_float_8(c3);
}

void g4_dot_q8_0_q8_0_x4(const void * vx, const void * const va[4], int64_t n, float out[4]) {
    const block_q8_0 * x = vx;
    const block_q8_0 * a0 = va[0], * a1 = va[1], * a2 = va[2], * a3 = va[3];
    const int64_t nb = n / QK8_0;
#if G4_HAS_AVX512
    if (g4_avx512) { dot_q8_0_q8_0_x4_avx512(x, (const block_q8_0 * const *)va, nb, out); return; }
#endif
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float dx = g4_fp16_to_fp32(x[ib].d);
        const __m256i bx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(bx, bx);
        #define ONE(K, A) do { \
            const __m256i by = _mm256_loadu_si256((const __m256i *)A[ib].qs); \
            const __m256i sy = _mm256_sign_epi8(by, bx); \
            const __m256i i32 = mul_sum_us8_pairs_epi32(ax, sy); \
            c##K = _mm256_fmadd_ps(_mm256_set1_ps(dx * g4_fp16_to_fp32(A[ib].d)), _mm256_cvtepi32_ps(i32), c##K); \
        } while (0)
        ONE(0, a0); ONE(1, a1); ONE(2, a2); ONE(3, a3);
        #undef ONE
    }
    out[0] = hsum_float_8(c0); out[1] = hsum_float_8(c1);
    out[2] = hsum_float_8(c2); out[3] = hsum_float_8(c3);
}

void g4_dot_tq2_0_q8_K_x4(const void * vx, const void * const va[4], int64_t n, float out[4]) {
    const block_tq2_0 * x = vx;
    const block_q8_K * a[4] = { va[0], va[1], va[2], va[3] };
    const int64_t nb = n / QK_K;
    float s[4] = {0, 0, 0, 0};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i is[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(),
                          _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < QK_K/4; j += 32) {
            const __m256i qbits = _mm256_loadu_si256((const __m256i *)(x[i].qs + j));
            for (int l = 0; l < 4; ++l) {
                const __m256i tr = _mm256_and_si256(_mm256_srli_epi16(qbits, 2*l), _mm256_set1_epi8(3));
                for (int t = 0; t < 4; t++) {
                    const __m256i q8 = _mm256_loadu_si256((const __m256i *)(a[t][i].qs + j*4 + l*32));
                    is[t] = _mm256_add_epi32(is[t], mul_sum_us8_pairs_epi32(tr, q8));
                }
            }
        }
        const float dx = g4_fp16_to_fp32(x[i].d);
        for (int t = 0; t < 4; t++) {
            int32_t bsum = 0;
            for (int j = 0; j < QK_K/16; ++j) bsum += a[t][i].bsums[j];
            __m128i v = _mm_add_epi32(_mm256_castsi256_si128(is[t]), _mm256_extracti128_si256(is[t], 1));
            v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
            v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
            s[t] += (float)(_mm_cvtsi128_si32(v) - bsum) * (a[t][i].d * dx);
        }
    }
    for (int t = 0; t < 4; t++) out[t] = s[t];
}

void g4_dot_q6_K_q8_K_x4(const void * vx, const void * const va[4], int64_t n, float out[4]) {
    const block_q6_K * x = vx;
    const block_q8_K * a[4] = { va[0], va[1], va[2], va[3] };
    const int64_t nb = n / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                      _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int64_t i = 0; i < nb; ++i) {
        const float dx = g4_fp16_to_fp32(x[i].d);
        const uint8_t * q4 = x[i].ql;
        const uint8_t * qh = x[i].qh;
        __m256i sumi[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(),
                            _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh); qh += 32;

            const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4),
                                 _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4));
            const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4),
                                 _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4));
            const __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4),
                                 _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4));
            const __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4),
                                 _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4));

            const int8_t * sc = x[i].scales + j*8;
            const __m256i sc01 = _mm256_set_m128i(_mm_set1_epi16(sc[1]), _mm_set1_epi16(sc[0]));
            const __m256i sc23 = _mm256_set_m128i(_mm_set1_epi16(sc[3]), _mm_set1_epi16(sc[2]));
            const __m256i sc45 = _mm256_set_m128i(_mm_set1_epi16(sc[5]), _mm_set1_epi16(sc[4]));
            const __m256i sc67 = _mm256_set_m128i(_mm_set1_epi16(sc[7]), _mm_set1_epi16(sc[6]));

            for (int t = 0; t < 4; t++) {
                const int8_t * q8 = a[t][i].qs + j*128;
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 64));
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)(q8 + 96));

                __m256i p0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q6_0, q8_0), _mm256_maddubs_epi16(m32s, q8_0));
                __m256i p1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q6_1, q8_1), _mm256_maddubs_epi16(m32s, q8_1));
                __m256i p2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q6_2, q8_2), _mm256_maddubs_epi16(m32s, q8_2));
                __m256i p3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q6_3, q8_3), _mm256_maddubs_epi16(m32s, q8_3));

                p0 = _mm256_madd_epi16(sc01, p0);
                p1 = _mm256_madd_epi16(sc23, p1);
                p2 = _mm256_madd_epi16(sc45, p2);
                p3 = _mm256_madd_epi16(sc67, p3);

                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p0, p1));
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p2, p3));
            }
        }
        for (int t = 0; t < 4; t++)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(a[t][i].d * dx), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < 4; t++) out[t] = hsum_float_8(acc[t]);
}

#else // scalar fallbacks

float g4_dot_q4_0_q8_0(const void * vx, const void * vy, int64_t n) { return g4_dot_q4_0_q8_0_ref(vx, vy, n); }
float g4_dot_q8_0_q8_0(const void * vx, const void * vy, int64_t n) { return g4_dot_q8_0_q8_0_ref(vx, vy, n); }
float g4_dot_tq2_0_q8_K(const void * vx, const void * vy, int64_t n) { return g4_dot_tq2_0_q8_K_ref(vx, vy, n); }
float g4_dot_q6_K_q8_K(const void * vx, const void * vy, int64_t n) { return g4_dot_q6_K_q8_K_ref(vx, vy, n); }
float g4_dot_q2_K_q8_K(const void * vx, const void * vy, int64_t n) { return g4_dot_q2_K_q8_K_ref(vx, vy, n); }
float g4_dot_q3_K_q8_K(const void * vx, const void * vy, int64_t n) { return g4_dot_q3_K_q8_K_ref(vx, vy, n); }
float g4_dot_q4_K_q8_K(const void * vx, const void * vy, int64_t n) { return g4_dot_q4_K_q8_K_ref(vx, vy, n); }
float g4_dot_q5_K_q8_K(const void * vx, const void * vy, int64_t n) { return g4_dot_q5_K_q8_K_ref(vx, vy, n); }
float g4_dot_f16(const void * vw, const g4_fp16 * a, int64_t n) {
    const g4_fp16 * w = vw; float s = 0;
    for (int64_t i = 0; i < n; i++) s += g4_fp16_to_fp32(w[i]) * g4_fp16_to_fp32(a[i]);
    return s;
}
float g4_dot_bf16(const void * vw, const float * a, int64_t n) {
    const g4_bf16 * w = vw; float s = 0;
    for (int64_t i = 0; i < n; i++) s += g4_bf16_to_fp32(w[i]) * a[i];
    return s;
}

void g4_dot_q4_0_q8_0_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_q4_0_q8_0(w, a[t], n);
}
void g4_dot_q8_0_q8_0_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_q8_0_q8_0(w, a[t], n);
}
void g4_dot_q6_K_q8_K_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_q6_K_q8_K(w, a[t], n);
}
void g4_dot_tq2_0_q8_K_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_tq2_0_q8_K(w, a[t], n);
}

#endif

// ---- K-quant x4 batch dots: share weight unpack across 4 activations via x1 ----
void g4_dot_q2_K_q8_K_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_q2_K_q8_K(w, a[t], n);
}
void g4_dot_q3_K_q8_K_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_q3_K_q8_K(w, a[t], n);
}
void g4_dot_q4_K_q8_K_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_q4_K_q8_K(w, a[t], n);
}
void g4_dot_q5_K_q8_K_x4(const void * w, const void * const a[4], int64_t n, float out[4]) {
    for (int t = 0; t < 4; t++) out[t] = g4_dot_q5_K_q8_K(w, a[t], n);
}

// --------------------------------------------------------------- selftest ---

static uint64_t st_rng = 0x123456789abcdefULL;
static float st_rand(void) {
    st_rng = st_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((int32_t)(st_rng >> 33)) / (float)(1u << 30);
}

int g4_selftest_quants(void) {
    int fail = 0;
    enum { N = 1536 };
    float x[N];
    static block_q8_0 a8[N/QK8_0];
    static block_q8_K aK[N/QK_K];

    for (int i = 0; i < N; i++) x[i] = st_rand();
    g4_quantize_q8_0(x, a8, N);
    g4_quantize_q8_K(x, aK, N);

    // verify q8_K bsums
    for (int b = 0; b < N/QK_K; b++) {
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int k = 0; k < 16; k++) s += aK[b].qs[g*16+k];
            if (s != aK[b].bsums[g]) { printf("FAIL q8_K bsums\n"); fail++; }
        }
    }

    // build synthetic weight rows and compare fast vs ref dots
    {
        static block_q4_0 w[N/QK4_0];
        for (size_t i = 0; i < sizeof(w); i++) ((uint8_t *)w)[i] = (uint8_t)(st_rng >> ((i%8)*8)), st_rng = st_rng*2862933555777941757ULL + 3037000493ULL;
        for (int b = 0; b < N/QK4_0; b++) w[b].d = g4_fp32_to_fp16(0.01f + 0.001f*b);
        float r = g4_dot_q4_0_q8_0_ref(w, a8, N), f = g4_dot_q4_0_q8_0(w, a8, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL q4_0 dot ref=%g fast=%g\n", r, f); fail++; }
    }
    {
        static block_q8_0 w[N/QK8_0];
        for (size_t i = 0; i < sizeof(w); i++) ((uint8_t *)w)[i] = (uint8_t)(st_rng >> ((i%8)*8)), st_rng = st_rng*2862933555777941757ULL + 3037000493ULL;
        for (int b = 0; b < N/QK8_0; b++) w[b].d = g4_fp32_to_fp16(0.01f);
        float r = g4_dot_q8_0_q8_0_ref(w, a8, N), f = g4_dot_q8_0_q8_0(w, a8, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL q8_0 dot ref=%g fast=%g\n", r, f); fail++; }
    }
    {
        static block_tq2_0 w[N/QK_K];
        for (int b = 0; b < N/QK_K; b++) {
            for (int j = 0; j < QK_K/4; j++) {
                uint8_t v = 0;
                for (int l = 0; l < 4; l++) { st_rng = st_rng*2862933555777941757ULL + 3037000493ULL; v |= (uint8_t)((st_rng >> 13) % 3) << (l*2); }
                w[b].qs[j] = v;
            }
            w[b].d = g4_fp32_to_fp16(0.02f);
        }
        float r = g4_dot_tq2_0_q8_K_ref(w, aK, N), f = g4_dot_tq2_0_q8_K(w, aK, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL tq2_0 dot ref=%g fast=%g\n", r, f); fail++; }
        // and dequant*x consistency
        static float wf[N]; g4_dequant_row(G4_TQ2_0, w, wf, N);
        double s = 0; for (int i = 0; i < N; i++) s += (double)wf[i]*((float)aK[i/QK_K].d*aK[i/QK_K].qs[i%QK_K]);
        if (fabs(s - r) > 1e-2 * (1 + fabs(s))) { printf("FAIL tq2_0 dequant consistency %g vs %g\n", s, r); fail++; }
    }
    {
        static block_q6_K w[N/QK_K];
        for (size_t i = 0; i < sizeof(w); i++) ((uint8_t *)w)[i] = (uint8_t)(st_rng >> ((i%8)*8)), st_rng = st_rng*2862933555777941757ULL + 3037000493ULL;
        for (int b = 0; b < N/QK_K; b++) {
            w[b].d = g4_fp32_to_fp16(0.002f);
            for (int j = 0; j < 16; j++) w[b].scales[j] = (int8_t)(j - 8);
        }
        float r = g4_dot_q6_K_q8_K_ref(w, aK, N), f = g4_dot_q6_K_q8_K(w, aK, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL q6_K dot ref=%g fast=%g\n", r, f); fail++; }
        // dequant consistency
        static float wf[N]; g4_dequant_row(G4_Q6_K, w, wf, N);
        double s = 0; for (int i = 0; i < N; i++) s += (double)wf[i]*((float)aK[i/QK_K].d*aK[i/QK_K].qs[i%QK_K]);
        if (fabs(s - r) > 1e-2 * (1 + fabs(s))) { printf("FAIL q6_K dequant consistency %g vs %g\n", s, r); fail++; }
    }
    // K-quants: validate dequant and vec_dot agree on the same random bytes.
    // d/dmin are forced to small positive fp16 values to avoid inf/nan payloads.
    {
        static block_q2_K w[N/QK_K];
        for (size_t i = 0; i < sizeof(w); i++) { ((uint8_t *)w)[i] = (uint8_t)(st_rng >> 19); st_rng = st_rng*2862933555777941757ULL + 3037000493ULL; }
        for (int b = 0; b < N/QK_K; b++) { w[b].d = g4_fp32_to_fp16(0.012f); w[b].dmin = g4_fp32_to_fp16(0.004f); }
        float r = g4_dot_q2_K_q8_K_ref(w, aK, N), f = g4_dot_q2_K_q8_K(w, aK, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL q2_K dot ref=%g fast=%g\n", r, f); fail++; }
        static float wf[N]; g4_dequant_row(G4_Q2_K, w, wf, N);
        double s = 0; for (int i = 0; i < N; i++) s += (double)wf[i]*((float)aK[i/QK_K].d*aK[i/QK_K].qs[i%QK_K]);
        if (fabs(s - r) > 1e-2 * (1 + fabs(s))) { printf("FAIL q2_K dequant consistency %g vs %g\n", s, r); fail++; }
    }
    {
        static block_q3_K w[N/QK_K];
        for (size_t i = 0; i < sizeof(w); i++) { ((uint8_t *)w)[i] = (uint8_t)(st_rng >> 19); st_rng = st_rng*2862933555777941757ULL + 3037000493ULL; }
        for (int b = 0; b < N/QK_K; b++) w[b].d = g4_fp32_to_fp16(0.004f);
        float r = g4_dot_q3_K_q8_K_ref(w, aK, N), f = g4_dot_q3_K_q8_K(w, aK, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL q3_K dot ref=%g fast=%g\n", r, f); fail++; }
        static float wf[N]; g4_dequant_row(G4_Q3_K, w, wf, N);
        double s = 0; for (int i = 0; i < N; i++) s += (double)wf[i]*((float)aK[i/QK_K].d*aK[i/QK_K].qs[i%QK_K]);
        if (fabs(s - r) > 1e-2 * (1 + fabs(s))) { printf("FAIL q3_K dequant consistency %g vs %g\n", s, r); fail++; }
    }
    {
        static block_q4_K w[N/QK_K];
        for (size_t i = 0; i < sizeof(w); i++) { ((uint8_t *)w)[i] = (uint8_t)(st_rng >> 19); st_rng = st_rng*2862933555777941757ULL + 3037000493ULL; }
        for (int b = 0; b < N/QK_K; b++) { w[b].d = g4_fp32_to_fp16(0.02f); w[b].dmin = g4_fp32_to_fp16(0.006f); }
        float r = g4_dot_q4_K_q8_K_ref(w, aK, N), f = g4_dot_q4_K_q8_K(w, aK, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL q4_K dot ref=%g fast=%g\n", r, f); fail++; }
        static float wf[N]; g4_dequant_row(G4_Q4_K, w, wf, N);
        double s = 0; for (int i = 0; i < N; i++) s += (double)wf[i]*((float)aK[i/QK_K].d*aK[i/QK_K].qs[i%QK_K]);
        if (fabs(s - r) > 1e-2 * (1 + fabs(s))) { printf("FAIL q4_K dequant consistency %g vs %g\n", s, r); fail++; }
    }
    {
        static block_q5_K w[N/QK_K];
        for (size_t i = 0; i < sizeof(w); i++) { ((uint8_t *)w)[i] = (uint8_t)(st_rng >> 19); st_rng = st_rng*2862933555777941757ULL + 3037000493ULL; }
        for (int b = 0; b < N/QK_K; b++) { w[b].d = g4_fp32_to_fp16(0.02f); w[b].dmin = g4_fp32_to_fp16(0.006f); }
        float r = g4_dot_q5_K_q8_K_ref(w, aK, N), f = g4_dot_q5_K_q8_K(w, aK, N);
        if (fabsf(r - f) > 1e-3f * (1 + fabsf(r))) { printf("FAIL q5_K dot ref=%g fast=%g\n", r, f); fail++; }
        static float wf[N]; g4_dequant_row(G4_Q5_K, w, wf, N);
        double s = 0; for (int i = 0; i < N; i++) s += (double)wf[i]*((float)aK[i/QK_K].d*aK[i/QK_K].qs[i%QK_K]);
        if (fabs(s - r) > 1e-2 * (1 + fabs(s))) { printf("FAIL q5_K dequant consistency %g vs %g\n", s, r); fail++; }
    }
    {
        // f16 dot
        static g4_fp16 w16[N], a16[N];
        for (int i = 0; i < N; i++) { w16[i] = g4_fp32_to_fp16(st_rand()); a16[i] = g4_fp32_to_fp16(x[i]); }
        double s = 0; for (int i = 0; i < N; i++) s += (double)g4_fp16_to_fp32(w16[i])*g4_fp16_to_fp32(a16[i]);
        float f = g4_dot_f16(w16, a16, N);
        if (fabs(s - f) > 1e-3 * (1 + fabs(s))) { printf("FAIL f16 dot %g vs %g\n", s, f); fail++; }
    }
    {
        // x4 kernels must be bit-identical to 4 separate x1 calls
        static block_q4_0 w4[N/QK4_0];
        static block_q6_K w6[N/QK_K];
        static block_tq2_0 wt[N/QK_K];
        static block_q8_0 w8[N/QK8_0];
        static block_q8_0 a8x[4][N/QK8_0];
        static block_q8_K aKx[4][N/QK_K];
        static float xs[4][N];
        for (int t = 0; t < 4; t++) {
            for (int i = 0; i < N; i++) xs[t][i] = st_rand() - 1.0f;
            g4_quantize_q8_0(xs[t], a8x[t], N);
            g4_quantize_q8_K(xs[t], aKx[t], N);
        }
        for (size_t i = 0; i < sizeof(w4); i++) ((uint8_t *)w4)[i] = (uint8_t)(st_rng >> 17), st_rng = st_rng*2862933555777941757ULL + 3037000493ULL;
        for (size_t i = 0; i < sizeof(w6); i++) ((uint8_t *)w6)[i] = (uint8_t)(st_rng >> 17), st_rng = st_rng*2862933555777941757ULL + 3037000493ULL;
        for (size_t i = 0; i < sizeof(wt); i++) ((uint8_t *)wt)[i] = (uint8_t)(st_rng >> 17), st_rng = st_rng*2862933555777941757ULL + 3037000493ULL;
        for (size_t i = 0; i < sizeof(w8); i++) ((uint8_t *)w8)[i] = (uint8_t)(st_rng >> 17), st_rng = st_rng*2862933555777941757ULL + 3037000493ULL;
        for (int b = 0; b < N/QK4_0; b++) { w4[b].d = g4_fp32_to_fp16(0.01f); w8[b].d = g4_fp32_to_fp16(0.02f); }
        for (int b = 0; b < N/QK_K; b++) {
            w6[b].d = g4_fp32_to_fp16(0.002f); wt[b].d = g4_fp32_to_fp16(0.02f);
            for (int j = 0; j < 16; j++) w6[b].scales[j] = (int8_t)((j*5)%23 - 11);
        }
        // fix trit encoding: ensure each 2-bit field is 0..2
        for (int b = 0; b < N/QK_K; b++)
            for (int j = 0; j < QK_K/4; j++) {
                uint8_t v = wt[b].qs[j], o = 0;
                for (int l = 0; l < 4; l++) { uint8_t f = (v >> (2*l)) & 3; if (f == 3) f = 2; o |= f << (2*l); }
                wt[b].qs[j] = o;
            }
        const void * aps8[4] = { a8x[0], a8x[1], a8x[2], a8x[3] };
        const void * apsK[4] = { aKx[0], aKx[1], aKx[2], aKx[3] };
        float o4[4], exp4[4];
        g4_dot_q4_0_q8_0_x4(w4, aps8, N, o4);
        for (int t = 0; t < 4; t++) exp4[t] = g4_dot_q4_0_q8_0(w4, aps8[t], N);
        for (int t = 0; t < 4; t++) if (o4[t] != exp4[t]) { printf("FAIL q4_0 x4 t=%d %g vs %g\n", t, o4[t], exp4[t]); fail++; }
        g4_dot_q8_0_q8_0_x4(w8, aps8, N, o4);
        for (int t = 0; t < 4; t++) exp4[t] = g4_dot_q8_0_q8_0(w8, aps8[t], N);
        for (int t = 0; t < 4; t++) if (o4[t] != exp4[t]) { printf("FAIL q8_0 x4 t=%d\n", t); fail++; }
        g4_dot_q6_K_q8_K_x4(w6, apsK, N, o4);
        for (int t = 0; t < 4; t++) exp4[t] = g4_dot_q6_K_q8_K(w6, apsK[t], N);
        for (int t = 0; t < 4; t++) if (o4[t] != exp4[t]) { printf("FAIL q6_K x4 t=%d %g vs %g\n", t, o4[t], exp4[t]); fail++; }
        g4_dot_tq2_0_q8_K_x4(wt, apsK, N, o4);
        for (int t = 0; t < 4; t++) exp4[t] = g4_dot_tq2_0_q8_K(wt, apsK[t], N);
        for (int t = 0; t < 4; t++) if (o4[t] != exp4[t]) { printf("FAIL tq2_0 x4 t=%d %g vs %g\n", t, o4[t], exp4[t]); fail++; }
    }
    {
        // fp16 conversion round-trip sanity
        for (uint32_t i = 0; i < 65536; i++) {
            float f = g4_fp16_to_fp32_table[i];
            if (f == f && fabsf(f) < 65504.0f) {  // skip nan/inf
                g4_fp16 back = g4_fp32_to_fp16(f);
                if (back != i && !(f == 0.0f && (back & 0x7fff) == 0)) {
                    printf("FAIL fp16 roundtrip %u -> %g -> %u\n", i, f, back); fail++;
                    if (fail > 5) return fail;
                }
            }
        }
    }
    {
        // K-quant quantizer round-trip: quantize a smooth signal, dequantize, bound the
        // relative RMS error (catches a broken layout; Q3_K is tighter than Q2_K).
        float src[QK_K], deq[QK_K];
        double ref = 0;
        for (int i = 0; i < QK_K; i++) { src[i] = sinf(i*0.1f) + 0.3f*sinf(i*0.73f); ref += src[i]*src[i]; }
        { block_q2_K b; g4_quantize_q2_K(src, &b, QK_K); g4_dequant_row(G4_Q2_K, &b, deq, QK_K);
          double e = 0; for (int i = 0; i < QK_K; i++) { double d = deq[i]-src[i]; e += d*d; }
          double rel = sqrt(e/ref);
          if (!(rel < 0.25)) { printf("FAIL q2_K roundtrip rel-rms %.3f\n", rel); fail++; } }
        { block_q3_K b; g4_quantize_q3_K(src, &b, QK_K); g4_dequant_row(G4_Q3_K, &b, deq, QK_K);
          double e = 0; for (int i = 0; i < QK_K; i++) { double d = deq[i]-src[i]; e += d*d; }
          double rel = sqrt(e/ref);
          if (!(rel < 0.12)) { printf("FAIL q3_K roundtrip rel-rms %.3f\n", rel); fail++; } }
    }
    {
        // q8_0/q4_0 dots: AVX-512 and AVX2 paths must both match the scalar reference
        static block_q8_0 w8[N/QK8_0], a8[N/QK8_0];
        static block_q4_0 w4q[N/QK4_0];
        static float xa[N];
        for (int i = 0; i < N; i++) xa[i] = st_rand() - 1.0f; g4_quantize_q8_0(xa, a8, N);
        for (int i = 0; i < N; i++) xa[i] = st_rand() - 1.0f; g4_quantize_q8_0(xa, w8, N); g4_quantize_q4_0(xa, w4q, N);
        const float r8 = g4_dot_q8_0_q8_0_ref(w8, a8, N);
        const float r4 = g4_dot_q4_0_q8_0_ref(w4q, a8, N);
        const int save = g4_avx512;
        for (int mode = 0; mode <= G4_HAS_AVX512; mode++) {   // 0 = AVX2, 1 = AVX-512 (if built in)
            g4_set_avx512(mode);
            float f8 = g4_dot_q8_0_q8_0(w8, a8, N), f4 = g4_dot_q4_0_q8_0(w4q, a8, N);
            if (fabs(f8 - r8) > 1e-2 * (1 + fabs(r8))) { printf("FAIL q8_0 dot (avx512=%d) %g vs ref %g\n", mode, f8, r8); fail++; }
            if (fabs(f4 - r4) > 1e-2 * (1 + fabs(r4))) { printf("FAIL q4_0 dot (avx512=%d) %g vs ref %g\n", mode, f4, r4); fail++; }
        }
        g4_set_avx512(save);
    }
    printf(fail ? "selftest: %d FAILURES\n" : "selftest: all quant tests passed\n", fail);
    return fail;
}
