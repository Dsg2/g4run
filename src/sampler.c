// sampler.c — sampling chain: penalties -> top-k -> top-p -> temperature -> dist
// (same stage order as llama.cpp common sampler defaults)
#include "g4run.h"

void g4_sampler_init(g4_sampler * s, int n_vocab) {
    memset(s, 0, sizeof(*s));
    s->temp = 1.0f; s->top_k = 64; s->top_p = 0.95f;
    s->repeat_penalty = 1.0f; s->repeat_last_n = 128;
    s->n_vocab = n_vocab;
    s->recent_cap = 4096;
    s->recent = malloc(s->recent_cap * 4);
    s->cnt = calloc(n_vocab, 4);
    s->seed = 0xdeadbeefULL;
}

void g4_sampler_reset(g4_sampler * s) {
    s->recent_n = 0;
    s->recent_head = 0;
    if (s->cnt) memset(s->cnt, 0, (size_t)s->n_vocab * 4);
}

void g4_sampler_free(g4_sampler * s) {
    free(s->recent); free(s->cnt); free(s->ban_ids);
    memset(s, 0, sizeof(*s));
}

void g4_sampler_ban(g4_sampler * s, int id) {
    if (id < 0 || id >= s->n_vocab) return;
    for (int i = 0; i < s->n_ban; i++) if (s->ban_ids[i] == id) return;  // dedup
    if (s->n_ban == s->ban_cap) {
        s->ban_cap = s->ban_cap ? s->ban_cap * 2 : 16;
        s->ban_ids = realloc(s->ban_ids, (size_t)s->ban_cap * sizeof(int));
    }
    s->ban_ids[s->n_ban++] = id;
}

void g4_sampler_accept(g4_sampler * s, int id) {
    if (s->recent_n == s->recent_cap) {
        int old = s->recent[s->recent_head];
        if (s->cnt[old] > 0) s->cnt[old]--;
        s->recent[s->recent_head] = id;
        s->recent_head = (s->recent_head + 1) % s->recent_cap;
    } else {
        s->recent[(s->recent_head + s->recent_n) % s->recent_cap] = id;
        s->recent_n++;
    }
    s->cnt[id]++;
}

// xoshiro-style rng
static uint64_t rng_next(g4_sampler * s) {
    if (!s->rng[0] && !s->rng[1] && !s->rng[2] && !s->rng[3]) {
        // splitmix init from seed
        uint64_t z = s->seed + 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < 4; i++) {
            z += 0x9e3779b97f4a7c15ULL;
            uint64_t v = z;
            v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
            v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
            s->rng[i] = v ^ (v >> 31);
        }
    }
    uint64_t * st = s->rng;
    const uint64_t result = st[0] + st[3];
    const uint64_t t = st[1] << 17;
    st[2] ^= st[0]; st[3] ^= st[1]; st[1] ^= st[2]; st[0] ^= st[3];
    st[2] ^= t;
    st[3] = (st[3] << 45) | (st[3] >> 19);
    return result;
}

typedef struct { float logit; int id; } cand_t;

int g4_sample(g4_sampler * s, float * logits) {
    const int nv = s->n_vocab;

    // ---- hard suppression: banned tokens never sampled ----
    for (int j = 0; j < s->n_ban; j++) logits[s->ban_ids[j]] = -INFINITY;

    // ---- no-repeat-ngram: forbid any token that would repeat an n-gram seen in
    //      the recent window (breaks verbatim degeneration loops; n>=2) ----
    if (s->no_repeat_ngram >= 2 && s->recent_n >= s->no_repeat_ngram) {
        const int N = s->no_repeat_ngram;
        int suf[64];
        int m = N - 1 < 64 ? N - 1 : 64;
        for (int i = 0; i < m; i++) {
            int idx = (s->recent_head + s->recent_n - m + i) % s->recent_cap;
            suf[i] = s->recent[idx];
        }
        for (int p = 0; p + m < s->recent_n; p++) {     // p+m is the token after the matched prefix
            int match = 1;
            for (int j = 0; j < m; j++) {
                int idx = (s->recent_head + p + j) % s->recent_cap;
                if (s->recent[idx] != suf[j]) { match = 0; break; }
            }
            if (match) {
                int idx = (s->recent_head + p + m) % s->recent_cap;
                logits[s->recent[idx]] = -INFINITY;
            }
        }
    }

    // ---- repeat / freq / presence penalties over the recent window ----
    if ((s->repeat_penalty != 1.0f || s->freq_penalty != 0 || s->presence_penalty != 0)
        && s->repeat_last_n > 0 && s->recent_n > 0) {
        int win = s->repeat_last_n < s->recent_n ? s->repeat_last_n : s->recent_n;
        // count tokens inside the window only
        for (int i = 0; i < win; i++) {
            int idx = (s->recent_head + s->recent_n - 1 - i) % s->recent_cap;
            int id = s->recent[idx];
            if (s->cnt[id] == 0) continue;            // marker handling below
        }
        // simple per-window counting (win <= 4096)
        static int local_cnt_ids[4096];
        int n_pen = 0;
        for (int i = 0; i < win; i++) {
            int idx = (s->recent_head + s->recent_n - 1 - i) % s->recent_cap;
            int id = s->recent[idx];
            // linear dedup is fine for small windows
            int seen = 0;
            for (int j = 0; j < n_pen; j++) if (local_cnt_ids[j] == id) { seen = 1; break; }
            if (!seen) local_cnt_ids[n_pen++] = id;
        }
        for (int j = 0; j < n_pen; j++) {
            int id = local_cnt_ids[j];
            int count = 0;
            for (int i = 0; i < win; i++) {
                int idx = (s->recent_head + s->recent_n - 1 - i) % s->recent_cap;
                if (s->recent[idx] == id) count++;
            }
            float l = logits[id];
            if (s->repeat_penalty != 1.0f)
                l = l <= 0 ? l * s->repeat_penalty : l / s->repeat_penalty;
            l -= (float)count * s->freq_penalty + (count > 0 ? s->presence_penalty : 0.0f);
            logits[id] = l;
        }
    }

    // ---- greedy ----
    if (s->temp <= 0.0f) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < nv; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    // ---- top-k selection (partial, via bounded min-heap) ----
    int k = s->top_k > 0 && s->top_k < 512 ? s->top_k : 512;
    if (k > nv) k = nv;
    static cand_t heap[512];
    int hn = 0;
    for (int i = 0; i < nv; i++) {
        float l = logits[i];
        if (hn < k) {
            heap[hn].logit = l; heap[hn].id = i;
            int j = hn++;
            while (j > 0) { int p = (j-1)/2; if (heap[j].logit < heap[p].logit) { cand_t t = heap[p]; heap[p] = heap[j]; heap[j] = t; j = p; } else break; }
        } else if (l > heap[0].logit) {
            heap[0].logit = l; heap[0].id = i;
            int j = 0;
            for (;;) {
                int a = 2*j+1, b = 2*j+2, m = j;
                if (a < k && heap[a].logit < heap[m].logit) m = a;
                if (b < k && heap[b].logit < heap[m].logit) m = b;
                if (m == j) break;
                cand_t t = heap[m]; heap[m] = heap[j]; heap[j] = t; j = m;
            }
        }
    }
    // sort candidates desc by logit (k small)
    cand_t cands[512];
    memcpy(cands, heap, hn * sizeof(cand_t));
    for (int i = 1; i < hn; i++) {
        cand_t v = cands[i]; int j = i - 1;
        while (j >= 0 && cands[j].logit < v.logit) { cands[j+1] = cands[j]; j--; }
        cands[j+1] = v;
    }

    // ---- top-p over softmax of candidates ----
    int n = hn;
    {
        float maxl = cands[0].logit;
        double sum = 0;
        static float probs[512];
        for (int i = 0; i < n; i++) { probs[i] = expf(cands[i].logit - maxl); sum += probs[i]; }
        if (s->top_p < 1.0f && s->top_p > 0.0f) {
            double cum = 0;
            int keep = n;
            for (int i = 0; i < n; i++) {
                cum += probs[i] / sum;
                if (cum >= s->top_p) { keep = i + 1; break; }
            }
            n = keep;
        }
    }

    // ---- temperature + final softmax + sample ----
    {
        float maxl = cands[0].logit / s->temp;
        double sum = 0;
        static double probs[512];
        for (int i = 0; i < n; i++) { probs[i] = exp((double)cands[i].logit / s->temp - maxl); sum += probs[i]; }
        double r = (double)(rng_next(s) >> 11) / 9007199254740992.0 * sum;  // [0, sum)
        double cum = 0;
        for (int i = 0; i < n; i++) {
            cum += probs[i];
            if (r < cum) return cands[i].id;
        }
        return cands[n-1].id;
    }
}
