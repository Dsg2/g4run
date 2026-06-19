// main.c — g4run CLI
#include "g4run.h"
#include <time.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Count physical cores (not logical/SMT). Inference here is memory-bandwidth
// bound, so running one thread per physical core is faster than oversubscribing
// the hyperthreads — the extra threads just contend for the same cache/memory
// ports. Returns 0 if detection fails (caller falls back to the static default).
static int g4_physical_cores(void) {
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (!len) return 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION * buf = malloc(len);
    if (!buf) return 0;
    int cores = 0;
    if (GetLogicalProcessorInformation(buf, &len))
        for (DWORD i = 0; i < len / sizeof(*buf); i++)
            if (buf[i].Relationship == RelationProcessorCore) cores++;
    free(buf);
    return cores;
}

static void usage(void) {
    printf("g4run — gemma4 GGUF inference (CPU)\n"
           "usage: g4run -m model.gguf [options]\n"
           "  -p STR          prompt (one-shot completion through chat template)\n"
           "  -i              interactive chat\n"
           "  --raw           no chat template (-p used verbatim)\n"
           "  -n N            max tokens to generate (default 256; -1 = until EOG)\n"
           "  -c N            context size (default %d)\n"
           "  -t N            threads (default: physical core count)\n"
           "  --temp F        temperature (default from model, 0 = greedy)\n"
           "  --top-k N       top-k (default from model)\n"
           "  --top-p F       top-p (default from model)\n"
           "  --repeat-penalty F  (default 1.0; qwen35 defaults to 1.1 anti-loop)\n"
           "  --repeat-last-n N   (default 128)\n"
           "  --no-repeat-ngram N block any verbatim N-gram repeat (anti-loop; 0=off)\n"
           "  --ban \"A,B\"        suppress trigger tokens, e.g. --ban \"Wait,Hmm\" (anti-loop)\n"
           "  --avx512        opt into AVX-512 q8_0/q4_0 dots (default AVX2; 512 was\n"
           "                  slower on Tiger Lake — see README; --no-avx512 forces AVX2)\n"
           "  --seed N        RNG seed (default time)\n"
           "  --sys STR       system prompt\n"
           "  --think         enable the thinking channel (model reasons in a\n"
           "                  <thought> block before answering; chat mode strips\n"
           "                  thoughts from the context after each turn)\n"
           "  --mtp PATH      gemma4-assistant draft head .gguf for speculative\n"
           "                  decoding (greedy; output identical to plain decode)\n"
           "  --draft N       draft tokens per round (default = nextn_predict_layers)\n"
           "  --spec N        qwen35 prompt-lookup speculative decoding, draft N tokens\n"
           "                  (greedy; output identical to plain greedy; one-shot -p)\n"
           "  --jit-quant T   requantize weights in RAM at load: q8_0|q4_0|q3_k|q2_k\n"
           "                  (smaller = faster but rougher; no smaller file needed)\n"
           "  --cache-type-k T  KV cache K type: f16 (default) | q8_0 | q4_0\n"
           "  --cache-type-v T  KV cache V type: f16 (default) | q8_0 | q4_0\n"
           "  --server        run as an HTTP/JSON server (sockets)\n"
           "  --host STR      server bind address (default 127.0.0.1)\n"
           "  --port N        server port (default 8080)\n"
           "  --no-bos        don't add BOS (override metadata)\n"
           "  --bos           force add BOS\n"
           "  --dump          dump GGUF metadata and exit\n"
           "  --selftest      run kernel selftests and exit\n"
           "  --tokenize STR  tokenize STR, print ids, exit\n"
           "  --parse-special parse special tokens in prompt text\n"
           "  --verbose       timings and diagnostics\n",
           G4_DEFAULT_CTX);
}

// parse a KV cache type name (matches llama.cpp --cache-type-k/v values)
static int kv_type_parse(const char * s) {
    if (!strcmp(s, "f16"))  return G4_F16;
    if (!strcmp(s, "q8_0")) return G4_Q8_0;
    if (!strcmp(s, "q4_0")) return G4_Q4_0;
    fprintf(stderr, "g4: unknown cache type '%s' (f16|q8_0|q4_0)\n", s);
    exit(1);
}

// ---- Ctrl+C handling: first press interrupts generation, second exits ----
static volatile LONG g_interrupt = 0;
static BOOL WINAPI ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT) {
        if (InterlockedIncrement(&g_interrupt) >= 2) return FALSE; // default: terminate
        return TRUE;
    }
    return FALSE;
}

// thought-channel display: dim ANSI on a terminal, plain markers when redirected
#include <io.h>
static bool out_is_tty(void) {
    static int tty = -1;
    if (tty < 0) {
        tty = _isatty(_fileno(stdout));
        if (tty) {  // enable VT processing for ANSI dim
            HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
            DWORD mode = 0;
            if (GetConsoleMode(h, &mode)) SetConsoleMode(h, mode | 0x0004 /*ENABLE_VIRTUAL_TERMINAL_PROCESSING*/);
        }
    }
    return tty > 0;
}
static void thought_begin(void) {
    if (out_is_tty()) printf("\x1b[2m<thought>\n"); else printf("<thought>\n");
    fflush(stdout);
}
static void thought_end(void) {
    if (out_is_tty()) printf("\n</thought>\x1b[0m\n"); else printf("\n</thought>\n");
    fflush(stdout);
}

// UTF-8-safe streaming emitter: buffers partial multi-byte sequences across tokens
typedef struct { char buf[16]; int n; } u8stream;
static void u8_emit(u8stream * s, const char * bytes, int len) {
    for (int i = 0; i < len; i++) {
        s->buf[s->n++] = bytes[i];
        // determine length of sequence at buf start
        uint8_t c0 = (uint8_t)s->buf[0];
        int need = c0 < 0x80 ? 1 : c0 < 0xC0 ? 1 : c0 < 0xE0 ? 2 : c0 < 0xF0 ? 3 : 4;
        if (s->n >= need) {
            fwrite(s->buf, 1, need, stdout);
            memmove(s->buf, s->buf + need, s->n - need);
            s->n -= need;
        }
    }
    fflush(stdout);
}

// first system block of the conversation: <|think|> goes at its very top
static int build_sysblk(char * dst, size_t cap, const char * sys, bool think) {
    if (think && sys) return snprintf(dst, cap, "<|turn>system\n<|think|>\n%s<turn|>\n", sys);
    if (think)        return snprintf(dst, cap, "<|turn>system\n<|think|>\n<turn|>\n");
    if (sys)          return snprintf(dst, cap, "<|turn>system\n%s<turn|>\n", sys);
    dst[0] = 0;
    return 0;
}

typedef struct { char * user; char * ans; } chat_turn_t;

static int chat_loop(g4_ctx * ctx, g4_tok * tok, g4_sampler * smp, g4_model_file * mf,
                     const char * sys_prompt, bool add_bos, int n_ctx, bool verbose,
                     bool think) {
    SetConsoleCtrlHandler(ctrl_handler, TRUE);
    fprintf(stderr, "g4run chat — %s%s\nCtrl+C stops generation; /think toggles thinking; /clear resets context;\nempty line or /exit quits.\n\n",
            mf->arch, think ? " (thinking)" : "");
    static int ids[65536];
    static int gen_ids[32768];
    char line[16384], turn[20000];
    int pos = 0;
    bool first = true;
    const int ch_open  = g4_tok_find(tok, "<|channel>");
    const int ch_close = g4_tok_find(tok, "<channel|>");
    chat_turn_t * hist = NULL;
    int hist_n = 0, hist_cap = 0;

    for (;;) {
        fprintf(stderr, pos ? "\n> " : "> ");
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t ll = strlen(line);
        while (ll && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = 0;
        if (!ll || !strcmp(line, "/exit")) break;
        if (!strcmp(line, "/clear")) {
            // restart the context: position 0, fresh template (BOS again), fresh
            // penalty history. KV needs no wipe — every slot is rewritten for its
            // position before any new token can attend to it.
            pos = 0;
            first = true;
            g4_sampler_reset(smp);
            for (int i = 0; i < hist_n; i++) { free(hist[i].user); free(hist[i].ans); }
            hist_n = 0;
            fprintf(stderr, "[context cleared]\n");
            continue;
        }
        if (!strcmp(line, "/think")) {
            think = !think;
            if (first || hist_n == 0) {
                fprintf(stderr, "[thinking %s]\n", think ? "enabled" : "disabled");
                continue;
            }
            // the <|think|> token lives in the FIRST system turn, so flipping it
            // mid-conversation means re-rendering the whole (thought-stripped)
            // transcript and re-prefilling
            size_t need = 1024 + (sys_prompt ? strlen(sys_prompt) : 0);
            for (int i = 0; i < hist_n; i++) need += strlen(hist[i].user) + strlen(hist[i].ans) + 64;
            char * full = malloc(need);
            char * p = full;
            p += build_sysblk(p, need, sys_prompt, think);
            for (int i = 0; i < hist_n; i++)
                p += sprintf(p, "<|turn>user\n%s<turn|>\n<|turn>model\n%s%s",
                             hist[i].user, hist[i].ans, i + 1 < hist_n ? "<turn|>\n" : "");
            int n = g4_tok_encode(tok, full, ids, 65536, add_bos, true);
            free(full);
            if (n + 16 >= n_ctx) {
                think = !think;
                fprintf(stderr, "[history too long to rebuild — /clear first]\n");
                continue;
            }
            double tr0 = g4_time_ms();
            for (int i = 0; i < n; i += G4_BATCH_MAX) {
                int nb = n - i < G4_BATCH_MAX ? n - i : G4_BATCH_MAX;
                g4_forward_batch(ctx, ids + i, nb, i, false);
            }
            g4_sampler_reset(smp);
            for (int i = 0; i < n; i++) g4_sampler_accept(smp, ids[i]);
            pos = n;
            fprintf(stderr, "[thinking %s — context rebuilt: %d tok in %.1f s]\n",
                    think ? "enabled" : "disabled", n, (g4_time_ms() - tr0) / 1000.0);
            continue;
        }
        g_interrupt = 0;

        if (first) {
            char sysblk[8400];
            build_sysblk(sysblk, sizeof(sysblk), sys_prompt, think);
            snprintf(turn, sizeof(turn), "%s<|turn>user\n%s<turn|>\n<|turn>model\n", sysblk, line);
        } else {  // close previous model turn (its <turn|> was sampled but never fed)
            snprintf(turn, sizeof(turn), "<turn|>\n<|turn>user\n%s<turn|>\n<|turn>model\n", line);
        }

        int n = g4_tok_encode(tok, turn, ids, 65536, first && add_bos, true);
        first = false;

        if (pos + n + 16 >= n_ctx) { fprintf(stderr, "[context full]\n"); break; }

        double t0 = g4_time_ms();
        float * logits = NULL;
        for (int i = 0; i < n; i += G4_BATCH_MAX) {
            int nb = n - i < G4_BATCH_MAX ? n - i : G4_BATCH_MAX;
            logits = g4_forward_batch(ctx, ids + i, nb, pos + i, i + nb >= n);
        }
        for (int i = 0; i < n; i++) g4_sampler_accept(smp, ids[i]);
        const int turn_start = pos;   // model turn begins here (for thought rewind)
        pos += n;
        double t1 = g4_time_ms();

        u8stream us = {{0}, 0};
        int generated = 0, hdr_skip = 0;
        bool in_thought = false, saw_thought = false;
        while (pos < n_ctx - 1 && !g_interrupt && generated < 32768) {
            int id = g4_sample(smp, logits);
            if (g4_tok_is_eog(tok, id)) break;
            gen_ids[generated] = id;
            if (id == ch_open)       { in_thought = true; saw_thought = true; hdr_skip = 2; thought_begin(); }
            else if (id == ch_close) { in_thought = false; thought_end(); }
            else {
                char buf[256];
                int len = g4_tok_decode(tok, id, buf, false);
                // hide the channel header ("thought" "\n") right after <|channel>
                if (hdr_skip > 0 && ((len == 7 && !memcmp(buf, "thought", 7)) || (len == 1 && buf[0] == '\n'))) hdr_skip--;
                else { hdr_skip = 0; u8_emit(&us, buf, len); }
            }
            g4_sampler_accept(smp, id);
            logits = g4_forward(ctx, id, pos);
            pos++;
            generated++;
        }
        if (in_thought) thought_end();   // interrupted mid-thought
        double t2 = g4_time_ms();

        // canonical template strips thought blocks from history (strip_thinking):
        // rewind the model turn and re-feed only the post-</thought> answer
        int ans_start = 0;
        if (saw_thought) {
            ans_start = generated;
            for (int i = generated - 1; i >= 0; i--)
                if (gen_ids[i] == ch_close) { ans_start = i + 1; break; }
            int n_ans = generated - ans_start;
            pos = turn_start + n;
            for (int i = 0; i < n_ans; i += G4_BATCH_MAX) {
                int nb = n_ans - i < G4_BATCH_MAX ? n_ans - i : G4_BATCH_MAX;
                g4_forward_batch(ctx, gen_ids + ans_start + i, nb, pos + i, false);
            }
            pos += n_ans;
            if (verbose) fprintf(stderr, "\n[thought stripped: ctx rewound by %d tok]", generated - n_ans);
        }

        // transcript (user line + thought-stripped answer) for /think rebuilds
        {
            size_t acap = 256, alen = 0;
            char * ans = malloc(acap);
            for (int i = ans_start; i < generated; i++) {
                char buf[256];
                int len = g4_tok_decode(tok, gen_ids[i], buf, false);
                if (alen + len + 1 > acap) { acap = (acap + len) * 2; ans = realloc(ans, acap); }
                memcpy(ans + alen, buf, len);
                alen += len;
            }
            ans[alen] = 0;
            if (hist_n == hist_cap) { hist_cap = hist_cap ? hist_cap * 2 : 16; hist = realloc(hist, hist_cap * sizeof(*hist)); }
            hist[hist_n].user = _strdup(line);
            hist[hist_n].ans = ans;
            hist_n++;
        }

        if (verbose || 1)
            fprintf(stderr, "\n[%d+%d tok | pp %.1f t/s | tg %.1f t/s | ctx %d/%d]\n",
                    n, generated, n*1000.0/(t1-t0), generated*1000.0/(t2-t1), pos, n_ctx);
    }
    for (int i = 0; i < hist_n; i++) { free(hist[i].user); free(hist[i].ans); }
    free(hist);
    return 0;
}

// ---- speculative decoding with the gemma4-assistant MTP draft head ----
// Greedy + provably exact: the target verifies every drafted token, so output is
// byte-identical to plain greedy decode — the draft only changes speed.
static int run_mtp(g4_ctx * ctx, g4_tok * tok, g4_model_file * mf, g4_mtp * mtp,
                   const int * ids, int n_in, int n_ctx, int n_gen, int K, bool verbose) {
    const int     n_embd = mf->n_embd;
    const int64_t nv     = mf->n_vocab;
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    float * logits_buf = malloc((size_t)(K+1) * nv * 4);
    float * hnext_buf  = malloc((size_t)(K+1) * n_embd * 4);
    float * h_cur = malloc((size_t)n_embd * 4);
    float * dh    = malloc((size_t)n_embd * 4);
    float * dh2   = malloc((size_t)n_embd * 4);
    int   * draft = malloc((size_t)K * sizeof(int));
    int   * batch = malloc((size_t)(K+1) * sizeof(int));
    if (!logits_buf || !hnext_buf || !h_cur || !dh || !dh2 || !draft || !batch) {
        fprintf(stderr, "g4: mtp buffer alloc failed\n"); return 1;
    }
    #define ARGMAX(P) ({ const float * _p = (P); int _b = 0; float _v = _p[0]; \
        for (int64_t _i = 1; _i < nv; _i++) if (_p[_i] > _v) { _v = _p[_i]; _b = (int)_i; } _b; })

    // ---- prefill: all-but-last cheaply, then the last token through the heads
    //      path (nb=1) to capture logits + h_nextn at the final prompt position ----
    double tp0 = g4_time_ms();
    for (int i = 0; i + 1 < n_in; i += G4_BATCH_MAX) {
        int nb = (n_in - 1) - i < G4_BATCH_MAX ? (n_in - 1) - i : G4_BATCH_MAX;
        g4_forward_batch(ctx, ids + i, nb, i, false);
    }
    g4_forward_batch_heads(ctx, ids + (n_in - 1), 1, n_in - 1, logits_buf, hnext_buf);
    double tp1 = g4_time_ms();
    int pos = n_in;
    int next_tok = ARGMAX(logits_buf);
    memcpy(h_cur, hnext_buf, (size_t)n_embd * 4);

    // ---- speculative loop ----
    u8stream us = {{0}, 0};
    int generated = 0;
    long n_drafted = 0, n_accepted = 0, n_rounds = 0;
    double tg0 = g4_time_ms();
    for (;;) {
        if (g4_tok_is_eog(tok, next_tok)) break;
        { char buf[256]; int len = g4_tok_decode(tok, next_tok, buf, false); u8_emit(&us, buf, len); }
        generated++;
        if ((n_gen >= 0 && generated >= n_gen) || g_interrupt || pos >= n_ctx - 1) break;

        int Keff = K;
        if (pos + Keff + 1 >= n_ctx) Keff = n_ctx - 1 - pos;   // stay within context
        if (Keff < 1) break;

        // draft Keff tokens from (next_tok, h_cur); each query attends target KV[0..pos-1]
        memcpy(dh, h_cur, (size_t)n_embd * 4);
        int dtok = next_tok;
        for (int i = 0; i < Keff; i++) {
            draft[i] = g4_mtp_step(mtp, dtok, dh, pos + i, pos - 1, dh2);
            float * t = dh; dh = dh2; dh2 = t;   // chained hidden
            dtok = draft[i];
        }
        n_drafted += Keff; n_rounds++;

        // verify: target runs [next_tok, draft...] at positions pos..pos+Keff in one batch
        batch[0] = next_tok;
        for (int i = 0; i < Keff; i++) batch[1 + i] = draft[i];
        g4_forward_batch_heads(ctx, batch, Keff + 1, pos, logits_buf, hnext_buf);

        // accept the longest prefix where the target's greedy choice agrees
        int n_acc = 0;
        while (n_acc < Keff && ARGMAX(logits_buf + (size_t)n_acc * nv) == draft[n_acc]) n_acc++;
        n_accepted += n_acc;

        if (verbose && n_rounds <= 8) {
            char b1[256], b2[256];
            fprintf(stderr, "[round %ld] after '%.*s' (%d):", n_rounds,
                    g4_tok_decode(tok, next_tok, b1, false), b1, next_tok);
            for (int i = 0; i < Keff; i++) {
                int tr = ARGMAX(logits_buf + (size_t)i * nv);
                int dl = g4_tok_decode(tok, draft[i], b1, false);
                int tl = g4_tok_decode(tok, tr, b2, false);
                fprintf(stderr, " d='%.*s'/t='%.*s'%s", dl, b1, tl, b2, draft[i]==tr?"=":"X");
                if (draft[i] != tr) break;
            }
            fprintf(stderr, "  (acc %d/%d)\n", n_acc, Keff);
        }

        bool stop = false;
        for (int i = 0; i < n_acc; i++) {
            if (g4_tok_is_eog(tok, draft[i])) { stop = true; break; }
            char buf[256]; int len = g4_tok_decode(tok, draft[i], buf, false); u8_emit(&us, buf, len);
            generated++;
            if (n_gen >= 0 && generated >= n_gen) { stop = true; break; }
        }
        // the bonus token = target's correct choice at the first unverified position
        int bonus = ARGMAX(logits_buf + (size_t)n_acc * nv);
        memcpy(h_cur, hnext_buf + (size_t)n_acc * n_embd, (size_t)n_embd * 4);
        pos += n_acc + 1;                  // KV now confirmed through position pos-1
        next_tok = bonus;
        if (stop || g_interrupt) break;
    }
    double tg1 = g4_time_ms();
    printf("\n");
    fprintf(stderr, "\ng4: prefill %d tok in %.0f ms (%.2f t/s) | gen %d tok in %.0f ms (%.2f t/s)\n",
            n_in, tp1 - tp0, n_in * 1000.0 / (tp1 - tp0),
            generated, tg1 - tg0, generated * 1000.0 / (tg1 - tg0));
    fprintf(stderr, "g4: mtp draft=%d | %ld rounds, %ld/%ld drafts accepted (%.1f%%), %.2f tokens/round\n",
            K, n_rounds, n_accepted, n_drafted,
            n_drafted ? 100.0 * n_accepted / n_drafted : 0.0,
            n_rounds ? (double)generated / n_rounds : 0.0);
    #undef ARGMAX
    free(logits_buf); free(hnext_buf); free(h_cur); free(dh); free(dh2); free(draft); free(batch);
    return 0;
}

// ---- qwen35 (hybrid Gated-DeltaNet + attention) generate path ----
static int run_qwen(qwen_ctx * qc, g4_tok * tok, g4_sampler * smp,
                    const char * prompt, bool raw, const char * sys_prompt,
                    int n_gen, int n_ctx, bool parse_special, bool verbose) {
    static int ids[65536];
    int n_in;
    if (raw) {
        n_in = g4_tok_encode(tok, prompt ? prompt : "", ids, 65536, false, parse_special);
    } else {
        // qwen chat template (<|im_start|> role \n content <|im_end|> ...)
        char buf[65536]; char * p = buf; size_t rem = sizeof(buf); int w;
        if (sys_prompt) { w = snprintf(p, rem, "<|im_start|>system\n%s<|im_end|>\n", sys_prompt); p += w; rem -= w; }
        snprintf(p, rem, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt ? prompt : "");
        n_in = g4_tok_encode(tok, buf, ids, 65536, false, true);
    }
    if (n_in < 1) { fprintf(stderr, "g4: empty prompt\n"); return 1; }
    if (verbose) {
        fprintf(stderr, "g4: prompt %d tokens:", n_in);
        for (int i = 0; i < n_in && i < 32; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "%s\n", n_in > 32 ? " ..." : "");
    }

    SetConsoleCtrlHandler(ctrl_handler, TRUE);
    double tp0 = g4_time_ms();
    float * logits = NULL;
    for (int i = 0; i < n_in; i += G4_BATCH_MAX) {           // batched prefill
        int b = n_in - i < G4_BATCH_MAX ? n_in - i : G4_BATCH_MAX;
        logits = g4_qwen_forward_batch(qc, ids + i, b, i, i + b >= n_in);
        for (int j = 0; j < b; j++) g4_sampler_accept(smp, ids[i + j]);
    }
    double tp1 = g4_time_ms();

    u8stream us = {{0}, 0};
    int pos = n_in, generated = 0;
    double tg0 = g4_time_ms();
    while ((n_gen < 0 || generated < n_gen) && !g_interrupt && pos < n_ctx - 1) {
        int id = g4_sample(smp, logits);
        if (g4_tok_is_eog(tok, id)) break;
        char b[256]; int l = g4_tok_decode(tok, id, b, false); u8_emit(&us, b, l);
        g4_sampler_accept(smp, id);
        logits = g4_qwen_forward(qc, id, pos);
        pos++; generated++;
    }
    double tg1 = g4_time_ms();
    printf("\n");
    fprintf(stderr, "\ng4: prefill %d tok in %.0f ms (%.2f t/s) | gen %d tok in %.0f ms (%.2f t/s)\n",
            n_in, tp1 - tp0, n_in * 1000.0 / (tp1 - tp0),
            generated, tg1 - tg0, generated * 1000.0 / (tg1 - tg0));
    return 0;
}

// qwen35 prompt-lookup speculative decoding (greedy; output identical to plain
// greedy). Drafts the next tokens by matching the recent suffix against earlier
// context, verifies up to K drafts in one batched (single weight-stream) forward,
// and rolls the recurrent SSM/conv state back via per-token checkpoints when a
// draft is rejected. A clear win on repetitive output (code/RAG/long-context),
// ~neutral on novel prose (no n-gram matches -> falls back to one token/forward).
static int run_qwen_spec(qwen_ctx * qc, g4_tok * tok,
                         const char * prompt, bool raw, const char * sys_prompt,
                         int n_gen, int n_ctx, int K, bool parse_special, bool verbose) {
    const int64_t nv = qc->mf->n_vocab;
    static int ids[65536];
    int n_in;
    if (raw) {
        n_in = g4_tok_encode(tok, prompt ? prompt : "", ids, 65536, false, parse_special);
    } else {
        char buf[65536]; char * p = buf; size_t rem = sizeof(buf); int w;
        if (sys_prompt) { w = snprintf(p, rem, "<|im_start|>system\n%s<|im_end|>\n", sys_prompt); p += w; rem -= w; }
        snprintf(p, rem, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt ? prompt : "");
        n_in = g4_tok_encode(tok, buf, ids, 65536, false, true);
    }
    if (n_in < 1) { fprintf(stderr, "g4: empty prompt\n"); return 1; }
    if (g4_qwen_spec_enable(qc, K)) return 1;

    static int hist[1 << 20];               // all tokens (prompt + generated) for n-gram lookup
    int H = 0;
    for (int i = 0; i < n_in; i++) hist[H++] = ids[i];

    float * vlog  = malloc((size_t)(K + 1) * nv * 4);   // per-position verify logits
    int   * batch = malloc((size_t)(K + 1) * sizeof(int));
    int   * draft = malloc((size_t)K * sizeof(int));
    if (!vlog || !batch || !draft) { fprintf(stderr, "g4: qwen spec alloc failed\n"); return 1; }
    #define ARGMAX(P) ({ const float * _p = (P); int _b = 0; float _v = _p[0]; \
        for (int64_t _i = 1; _i < nv; _i++) if (_p[_i] > _v) { _v = _p[_i]; _b = (int)_i; } _b; })

    SetConsoleCtrlHandler(ctrl_handler, TRUE);
    double tp0 = g4_time_ms();
    for (int i = 0; i + 1 < n_in; i += G4_BATCH_MAX) {                 // prefill all but last
        int b = (n_in - 1) - i < G4_BATCH_MAX ? (n_in - 1) - i : G4_BATCH_MAX;
        g4_qwen_forward_batch(qc, ids + i, b, i, false);
    }
    float * logits = g4_qwen_forward_batch(qc, ids + (n_in - 1), 1, n_in - 1, true);
    double tp1 = g4_time_ms();

    int pos = n_in, next_tok = ARGMAX(logits);
    u8stream us = {{0}, 0};
    int generated = 0;
    long n_drafted = 0, n_accepted = 0, n_rounds = 0;
    double accept_ema = 0.5;   // EMA of per-draft accept rate (neutral start)
    int    probe = 0;          // probe counter for re-detecting repetition when backed off
    double tg0 = g4_time_ms();
    for (;;) {
        if (g4_tok_is_eog(tok, next_tok)) break;
        { char buf[256]; int len = g4_tok_decode(tok, next_tok, buf, false); u8_emit(&us, buf, len); }
        hist[H++] = next_tok; generated++;
        if ((n_gen >= 0 && generated >= n_gen) || g_interrupt || pos >= n_ctx - 1) break;

        int room = K;
        if (pos + room + 1 >= n_ctx) room = n_ctx - 1 - pos;
        if (room < 1) break;
        // Adaptive draft length: the batched verify gets compute-bound as it grows, so
        // drafting is only worth it when recent drafts are accepted. Shrink toward 0
        // (plain decode) when acceptance drops; probe occasionally to re-detect repetition.
        int kdyn = (int)(accept_ema * K + 0.5);
        if (kdyn < 1 && ((++probe % 12) == 0)) kdyn = 1;   // cheap probe to re-detect repetition
        if (kdyn > room) kdyn = room;

        // prompt-lookup draft: most-recent match of the last ng tokens (ng = 3 then 2)
        int nd = 0;
        for (int ng = 3; ng >= 2 && nd == 0 && kdyn >= 1; ng--) {
            if (H < ng + 1) continue;
            for (int j = H - ng - 1; j >= 0; j--) {
                bool hit = true;
                for (int t = 0; t < ng; t++) if (hist[j + t] != hist[H - ng + t]) { hit = false; break; }
                if (!hit) continue;
                for (int t = 0; t < kdyn && j + ng + t < H; t++) draft[nd++] = hist[j + ng + t];
                break;
            }
        }

        n_rounds++;
        if (nd == 0) {                  // no draft -> plain single-token step (zero spec overhead)
            float * lg = g4_qwen_forward(qc, next_tok, pos);
            next_tok = ARGMAX(lg);
            pos += 1;
            continue;
        }

        // verify [next_tok, draft...] in one batch; checkpoints let us roll back rejects
        batch[0] = next_tok;
        for (int i = 0; i < nd; i++) batch[1 + i] = draft[i];
        g4_qwen_forward_spec(qc, batch, nd + 1, pos, vlog);
        n_drafted += nd;

        int n_acc = 0;
        while (n_acc < nd && ARGMAX(vlog + (size_t)n_acc * nv) == draft[n_acc]) n_acc++;
        n_accepted += n_acc;
        accept_ema = 0.75 * accept_ema + 0.25 * ((double)n_acc / nd);

        bool stop = false;
        for (int i = 0; i < n_acc; i++) {
            if (g4_tok_is_eog(tok, draft[i])) { stop = true; break; }
            char buf[256]; int len = g4_tok_decode(tok, draft[i], buf, false); u8_emit(&us, buf, len);
            hist[H++] = draft[i]; generated++;
            if (n_gen >= 0 && generated >= n_gen) { stop = true; break; }
        }
        if (n_acc < nd) g4_qwen_state_restore(qc, n_acc);     // keep state if all drafts accepted
        next_tok = ARGMAX(vlog + (size_t)n_acc * nv);         // bonus: correct token at first unverified pos
        pos += n_acc + 1;
        if (stop || g_interrupt) break;
    }
    double tg1 = g4_time_ms();
    printf("\n");
    fprintf(stderr, "\ng4: prefill %d tok in %.0f ms (%.2f t/s) | gen %d tok in %.0f ms (%.2f t/s)\n",
            n_in, tp1 - tp0, n_in * 1000.0 / (tp1 - tp0),
            generated, tg1 - tg0, generated * 1000.0 / (tg1 - tg0));
    fprintf(stderr, "g4: prompt-lookup K=%d | %ld rounds, %ld/%ld drafts accepted (%.1f%%), %.2f tokens/round\n",
            K, n_rounds, n_accepted, n_drafted,
            n_drafted ? 100.0 * n_accepted / n_drafted : 0.0,
            n_rounds ? (double)generated / n_rounds : 0.0);
    (void)verbose;
    #undef ARGMAX
    free(vlog); free(batch); free(draft);
    return 0;
}

// qwen35 multi-turn interactive chat (recurrent state + KV carry across turns)
// Copy src->dst with every <think>...</think> reasoning span removed (plus the
// whitespace immediately following </think>). An unterminated <think> (e.g. the
// turn was interrupted mid-reasoning) drops everything from the tag onward. This
// is how Qwen's chat template keeps prior-turn reasoning out of later context.
// Operates on an explicit length so embedded NULs from byte-fallback tokens are
// handled. Returns bytes written; dst is NUL-terminated.
static size_t strip_think(const char * src, size_t slen, char * dst, size_t dstcap) {
    size_t di = 0;
    const char * p = src, * end = src + slen;
    while (p < end && di + 1 < dstcap) {
        const char * t = NULL;
        for (const char * q = p; q + 7 <= end; q++)
            if (!memcmp(q, "<think>", 7)) { t = q; break; }
        if (!t) {                                   // no more reasoning blocks: copy the rest
            size_t rem = (size_t)(end - p);
            if (di + rem >= dstcap) rem = dstcap - 1 - di;
            memcpy(dst + di, p, rem); di += rem; break;
        }
        size_t pre = (size_t)(t - p);               // copy text before <think>
        if (di + pre >= dstcap) pre = dstcap - 1 - di;
        memcpy(dst + di, p, pre); di += pre;
        const char * c = NULL;
        for (const char * q = t + 7; q + 8 <= end; q++)
            if (!memcmp(q, "</think>", 8)) { c = q; break; }
        if (!c) break;                              // unterminated: drop tag through end
        p = c + 8;
        while (p < end && (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t')) p++;
    }
    dst[di] = 0;
    return di;
}

static int qwen_chat_loop(qwen_ctx * qc, g4_tok * tok, g4_sampler * smp,
                          const char * sys_prompt, int n_ctx, bool verbose) {
    (void)verbose;
    SetConsoleCtrlHandler(ctrl_handler, TRUE);
    fprintf(stderr, "g4run chat — qwen35\nCtrl+C stops generation; /clear resets context; empty line or /exit quits.\n"
                    "(prior-turn <think> reasoning is stripped from context each turn)\n\n");
    static int  ids[65536];
    static char hist[1 << 20];   // cleaned transcript: rendered template with past reasoning removed
    static char resp[1 << 18];   // raw assistant output for the current turn (with reasoning)
    static char clean[1 << 18];  // current turn with <think> stripped, appended to the transcript
    char line[16384];
    size_t hlen = 0; bool first = true;
    for (;;) {
        fprintf(stderr, hlen ? "\n> " : "> ");
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t ll = strlen(line);
        while (ll && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = 0;
        if (!ll || !strcmp(line, "/exit")) break;
        if (!strcmp(line, "/clear")) { hlen = 0; first = true; g4_qwen_reset(qc); g4_sampler_reset(smp); fprintf(stderr, "[context cleared]\n"); continue; }
        g_interrupt = 0;

        // append the user turn (+ system on the first) and the assistant opener to the transcript
        if (first && sys_prompt)
            hlen += snprintf(hist + hlen, sizeof(hist) - hlen, "<|im_start|>system\n%s<|im_end|>\n", sys_prompt);
        hlen += snprintf(hist + hlen, sizeof(hist) - hlen,
                         "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);
        first = false;
        if (hlen + 64 >= sizeof(hist)) { fprintf(stderr, "[transcript buffer full]\n"); break; }

        // Re-encode and re-prefill the whole cleaned transcript from a fresh state. Because the
        // recurrent SSM/conv state is a running summary it can't be rewound to excise reasoning,
        // so we reset it and rebuild context from the stripped transcript each turn.
        int n = g4_tok_encode(tok, hist, ids, 65536, false, true);
        if (n + 16 >= n_ctx) { fprintf(stderr, "[context full]\n"); break; }
        g4_qwen_reset(qc);
        g4_sampler_reset(smp);

        double t0 = g4_time_ms();
        int pos = 0; float * logits = NULL;
        for (int i = 0; i < n; i += G4_BATCH_MAX) {
            int b = n - i < G4_BATCH_MAX ? n - i : G4_BATCH_MAX;
            logits = g4_qwen_forward_batch(qc, ids + i, b, i, i + b >= n);
            for (int j = 0; j < b; j++) g4_sampler_accept(smp, ids[i + j]);
        }
        pos = n;
        double t1 = g4_time_ms();

        u8stream us = {{0}, 0};
        size_t rlen = 0; int generated = 0;
        while (pos < n_ctx - 1 && !g_interrupt && generated < 32768) {
            int id = g4_sample(smp, logits);
            if (g4_tok_is_eog(tok, id)) break;
            char buf[256]; int len = g4_tok_decode(tok, id, buf, false);
            u8_emit(&us, buf, len);                                   // stream raw (reasoning visible live)
            if (rlen + (size_t)len < sizeof(resp)) { memcpy(resp + rlen, buf, len); rlen += len; }
            g4_sampler_accept(smp, id);
            logits = g4_qwen_forward(qc, id, pos);
            pos++; generated++;
        }
        double t2 = g4_time_ms();

        // store only the reasoning-stripped answer, then close the assistant turn
        size_t clen = strip_think(resp, rlen, clean, sizeof(clean));
        hlen += snprintf(hist + hlen, sizeof(hist) - hlen, "%.*s<|im_end|>\n", (int)clen, clean);
        if (hlen >= sizeof(hist)) hlen = sizeof(hist) - 1;

        fprintf(stderr, "\n[%d+%d tok | pp %.1f t/s | tg %.1f t/s | ctx %d/%d]\n",
                n, generated, n*1000.0/(t1-t0), generated*1000.0/(t2-t1), pos, n_ctx);
    }
    return 0;
}

// Validated numeric arg parse: error out (instead of silently yielding 0) on
// garbage like `--spec foo`, or a missing value that swallowed the next flag.
static long ival(const char * flag, const char * s) {
    char * e; long v = strtol(s, &e, 10);
    if (e == s || *e) { fprintf(stderr, "g4: %s expects an integer, got '%s'\n", flag, s); exit(1); }
    return v;
}
static double fval(const char * flag, const char * s) {
    char * e; double v = strtod(s, &e);
    if (e == s || *e) { fprintf(stderr, "g4: %s expects a number, got '%s'\n", flag, s); exit(1); }
    return v;
}

static const char * base_name(const char * p) {
    const char * a = strrchr(p, '\\'), * b = strrchr(p, '/');
    const char * n = a > b ? a : b;
    return n ? n + 1 : p;
}

// Apply anti-repetition controls to a local sampler: no-repeat-ngram + any --ban
// trigger strings (each tokenized in bare and space-prefixed form, so e.g. "Wait"
// is suppressed at line start and mid-text). Returns the number of banned tokens.
static int apply_anti_loop(g4_sampler * smp, g4_tok * tok, int no_repeat_ngram, const char * ban_str) {
    smp->no_repeat_ngram = no_repeat_ngram;
    if (!ban_str || !*ban_str || !strcmp(ban_str, "none")) return 0;
    char buf[1024]; snprintf(buf, sizeof(buf), "%s", ban_str);
    int before = smp->n_ban;
    for (char * t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        while (*t == ' ') t++;
        if (!*t) continue;
        char var[128]; int ids[16];
        for (int v = 0; v < 2; v++) {
            snprintf(var, sizeof(var), v ? " %s" : "%s", t);
            int n = g4_tok_encode(tok, var, ids, 16, false, false);
            for (int j = 0; j < n; j++) g4_sampler_ban(smp, ids[j]);
        }
    }
    return smp->n_ban - before;
}

// Concise startup banner (shown by default) so the resolved config and any active
// flags are visible without --verbose.
static void log_config(const g4_model_file * mf, const char * model_path, int n_threads,
                       int n_ctx, int spec_k, int jit_type, const char * mtp_path,
                       int kv_k, int kv_v, int no_repeat_ngram, const char * ban_str,
                       const char * mode) {
    fprintf(stderr, "g4: %s | arch=%s | %d layers | ctx %d | %d threads | %s | simd=%s\n",
            base_name(model_path), mf->arch, mf->n_layer, n_ctx, n_threads, mode,
            g4_avx512 ? "avx512-vnni" : "avx2");
    char feat[256]; size_t fl = 0; feat[0] = 0;
    #define FEAT(...) fl += snprintf(feat + fl, sizeof(feat) - fl, __VA_ARGS__)
    if (spec_k)   FEAT(" spec=%d", spec_k);
    if (jit_type) FEAT(" jit-quant=%s", g4_type_name(jit_type));
    if (mtp_path) FEAT(" mtp");
    if (kv_k != G4_F16 || kv_v != G4_F16) FEAT(" kv=%s/%s", g4_type_name(kv_k), g4_type_name(kv_v));
    if (no_repeat_ngram >= 2) FEAT(" no_repeat_ngram=%d", no_repeat_ngram);
    if (ban_str && *ban_str && strcmp(ban_str, "none")) FEAT(" ban='%s'", ban_str);
    #undef FEAT
    if (feat[0]) fprintf(stderr, "g4: features:%s\n", feat);
}

int main(int argc, char ** argv) {
    g4_init_tables();
    SetConsoleOutputCP(CP_UTF8);

    const char * model_path = NULL;
    const char * prompt = NULL;
    const char * sys_prompt = NULL;
    const char * tokenize_str = NULL;
    bool interactive = false, raw = false, do_dump = false, do_selftest = false;
    bool verbose = false, parse_special = false, think = false, server = false;
    const char * host = "127.0.0.1";
    const char * mtp_path = NULL;        // gemma4-assistant draft head for spec decoding
    int  draft_len = 0;                  // tokens drafted per round (0 = use nextn_predict_layers)
    int  spec_k = 0;                     // qwen35 prompt-lookup draft length (0 = off)
    int  jit_type = 0;                   // JIT requant target (0 = off, else G4_Q4_0/G4_Q8_0)
    int  kv_type_k = G4_F16, kv_type_v = G4_F16;
    int port = 8080;
    int  n_gen = 256, n_ctx = G4_DEFAULT_CTX, n_threads = 0;  // 0 = auto (physical cores)
    int  bos_override = -1; // -1 metadata, 0 off, 1 on
    float temp = -2, top_p = -2, repeat_penalty = -1.0f;  // rp<0 = auto (per-arch default)
    int top_k = -2, repeat_last_n = 128;
    int  no_repeat_ngram = 0;            // 0 = off; block verbatim n-gram repeats
    const char * ban_str = NULL;         // comma-separated tokens to suppress (anti-loop)
    uint64_t seed = (uint64_t)time(NULL);

    for (int i = 1; i < argc; i++) {
        #define NEXT() (i+1 < argc ? argv[++i] : (usage(), exit(1), (char*)0))
        if      (!strcmp(argv[i], "-m"))         model_path = NEXT();
        else if (!strcmp(argv[i], "-p"))         prompt = NEXT();
        else if (!strcmp(argv[i], "-i"))         interactive = true;
        else if (!strcmp(argv[i], "--raw"))      raw = true;
        else if (!strcmp(argv[i], "-n"))         n_gen = (int)ival("-n", NEXT());
        else if (!strcmp(argv[i], "-c"))         n_ctx = (int)ival("-c", NEXT());
        else if (!strcmp(argv[i], "-t"))         n_threads = (int)ival("-t", NEXT());
        else if (!strcmp(argv[i], "--temp"))     temp = (float)fval("--temp", NEXT());
        else if (!strcmp(argv[i], "--top-k"))    top_k = (int)ival("--top-k", NEXT());
        else if (!strcmp(argv[i], "--top-p"))    top_p = (float)fval("--top-p", NEXT());
        else if (!strcmp(argv[i], "--repeat-penalty")) repeat_penalty = (float)fval("--repeat-penalty", NEXT());
        else if (!strcmp(argv[i], "--repeat-last-n"))  repeat_last_n = (int)ival("--repeat-last-n", NEXT());
        else if (!strcmp(argv[i], "--no-repeat-ngram")) no_repeat_ngram = (int)ival("--no-repeat-ngram", NEXT());
        else if (!strcmp(argv[i], "--ban"))      ban_str = NEXT();
        else if (!strcmp(argv[i], "--seed"))     seed = (uint64_t)_strtoui64(NEXT(), NULL, 10);
        else if (!strcmp(argv[i], "--sys"))      sys_prompt = NEXT();
        else if (!strcmp(argv[i], "--think"))    think = true;
        else if (!strcmp(argv[i], "--server"))   server = true;
        else if (!strcmp(argv[i], "--host"))     host = NEXT();
        else if (!strcmp(argv[i], "--port"))     port = (int)ival("--port", NEXT());
        else if (!strcmp(argv[i], "--mtp"))      mtp_path = NEXT();
        else if (!strcmp(argv[i], "--draft"))    draft_len = (int)ival("--draft", NEXT());
        else if (!strcmp(argv[i], "--spec"))     spec_k = (int)ival("--spec", NEXT());
        else if (!strcmp(argv[i], "--jit-quant")) {
            const char * jt = NEXT();
            if      (!strcmp(jt, "q4_0")) jit_type = G4_Q4_0;
            else if (!strcmp(jt, "q8_0")) jit_type = G4_Q8_0;
            else if (!strcmp(jt, "q2_k")) jit_type = G4_Q2_K;
            else if (!strcmp(jt, "q3_k")) jit_type = G4_Q3_K;
            else { fprintf(stderr, "g4: --jit-quant expects q4_0 | q8_0 | q3_k | q2_k\n"); return 1; }
        }
        else if (!strcmp(argv[i], "--cache-type-k") || !strcmp(argv[i], "-ctk")) kv_type_k = kv_type_parse(NEXT());
        else if (!strcmp(argv[i], "--cache-type-v") || !strcmp(argv[i], "-ctv")) kv_type_v = kv_type_parse(NEXT());
        else if (!strcmp(argv[i], "--no-avx512")) g4_set_avx512(0);
        else if (!strcmp(argv[i], "--avx512"))    g4_set_avx512(1);
        else if (!strcmp(argv[i], "--no-bos"))   bos_override = 0;
        else if (!strcmp(argv[i], "--bos"))      bos_override = 1;
        else if (!strcmp(argv[i], "--dump"))     do_dump = true;
        else if (!strcmp(argv[i], "--selftest")) do_selftest = true;
        else if (!strcmp(argv[i], "--list-gpu")) { extern int g4_vk_list(void); return g4_vk_list(); }
        else if (!strcmp(argv[i], "--gpu-test")) { extern int g4_vk_test(const char *); return g4_vk_test(NEXT()); }
        else if (!strcmp(argv[i], "--tokenize")) tokenize_str = NEXT();
        else if (!strcmp(argv[i], "--parse-special")) parse_special = true;
        else if (!strcmp(argv[i], "--verbose"))  verbose = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(); return 1; }
        #undef NEXT
    }

    if (do_selftest) {
        int fail = g4_selftest_quants();
        return fail ? 1 : 0;
    }
    if (!model_path) { usage(); return 1; }

    // Default thread count = physical cores. Decode is memory-bandwidth bound, so
    // oversubscribing the SMT siblings (e.g. 8 threads on a 4-core CPU) is slower,
    // not faster. Measured on i5-1135G7: ~40% faster decode, ~2x faster prefill at
    // 4 threads vs 8. An explicit -t overrides this.
    if (n_threads <= 0) {
        int pc = g4_physical_cores();
        n_threads = pc > 0 ? pc : G4_DEFAULT_THREADS;
    }
    if (verbose) fprintf(stderr, "g4: using %d threads\n", n_threads);

    double t0 = g4_time_ms();
    g4_model_file mf;
    if (g4_gguf_open(&mf, model_path, verbose)) return 1;
    if (verbose) fprintf(stderr, "g4: header parsed in %.1f ms\n", g4_time_ms() - t0);

    if (do_dump) {
        g4_gguf_dump(&mf);
        g4_gguf_close(&mf);
        return 0;
    }

    if (jit_type) g4_jit_requant(&mf, jit_type, n_threads, verbose);   // requant in RAM before ctx init

    // qwen35 is loop-prone in long <think> blocks; default to a mild repeat penalty.
    if (repeat_penalty < 0) repeat_penalty = mf.is_qwen35 ? 1.1f : 1.0f;

    bool add_bos = bos_override == -1 ? mf.add_bos : bos_override == 1;

    if (tokenize_str) {
        g4_tok * tok = g4_tok_init(&mf);
        if (!tok) return 1;
        char * text = (char *)tokenize_str;
        char * filebuf = NULL;
        if (tokenize_str[0] == '@') {       // @path = read text from file
            FILE * f = fopen(tokenize_str + 1, "rb");
            if (!f) { fprintf(stderr, "cannot open %s\n", tokenize_str + 1); return 1; }
            fseek(f, 0, SEEK_END); long fl = ftell(f); fseek(f, 0, SEEK_SET);
            filebuf = malloc(fl + 1);
            fread(filebuf, 1, fl, f); filebuf[fl] = 0; fclose(f);
            text = filebuf;
        }
        static int ids[65536];
        int n = g4_tok_encode(tok, text, ids, 65536, add_bos, parse_special);
        // llama-tokenize --ids compatible output
        printf("[");
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i+1 < n ? ", " : "");
        printf("]\n");
        if (verbose) {
            for (int i = 0; i < n; i++) {
                char buf[256];
                int len = g4_tok_decode(tok, ids[i], buf, true);
                printf("%6d -> '%.*s'\n", ids[i], len, buf);
            }
        }
        free(filebuf);
        g4_tok_free(tok);
        g4_gguf_close(&mf);
        return 0;
    }

    // ---- full inference path ----
    // -p @file reads the prompt from a file (byte-exact, for parity testing)
    char * prompt_filebuf = NULL;
    if (prompt && prompt[0] == '@') {
        FILE * f = fopen(prompt + 1, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", prompt + 1); return 1; }
        fseek(f, 0, SEEK_END); long fl = ftell(f); fseek(f, 0, SEEK_SET);
        prompt_filebuf = malloc(fl + 1);
        fread(prompt_filebuf, 1, fl, f); prompt_filebuf[fl] = 0; fclose(f);
        prompt = prompt_filebuf;
    }

    g4_tok * tok = g4_tok_init(&mf);
    if (!tok) return 1;

    const char * mode = server ? "server" : interactive ? "interactive"
                      : mtp_path ? "mtp spec-decode" : spec_k ? "prompt-lookup spec-decode" : "one-shot";
    log_config(&mf, model_path, n_threads, n_ctx, spec_k, jit_type, mtp_path,
               kv_type_k, kv_type_v, no_repeat_ngram, ban_str, mode);

    // qwen35 hybrid model: separate context + generate path (no gemma KV cache)
    if (mf.is_qwen35) {
        qwen_ctx qc;
        if (g4_qwen_init(&qc, &mf, n_ctx, n_threads)) return 1;
        int rc = 0;
        if (server) {
            rc = g4_server_run(&qc, 1, tok, &mf, host, port, base_name(model_path),
                               repeat_penalty, no_repeat_ngram, ban_str);
        } else {
            g4_sampler smp;
            g4_sampler_init(&smp, mf.n_vocab);
            smp.temp  = temp  > -2 ? temp  : 0.7f;
            smp.top_k = top_k > -2 ? top_k : 20;
            smp.top_p = top_p > -2 ? top_p : 0.8f;
            smp.repeat_penalty = repeat_penalty;
            smp.repeat_last_n  = repeat_last_n;
            smp.seed = seed;
            apply_anti_loop(&smp, tok, no_repeat_ngram, ban_str);
            fprintf(stderr, "g4: sampling temp=%.2f top_k=%d top_p=%.2f repeat_penalty=%.2f no_repeat_ngram=%d banned=%d tok\n",
                    smp.temp, smp.top_k, smp.top_p, smp.repeat_penalty, smp.no_repeat_ngram, smp.n_ban);
            if (interactive)     rc = qwen_chat_loop(&qc, tok, &smp, sys_prompt, n_ctx, verbose);
            else if (spec_k > 0) rc = run_qwen_spec(&qc, tok, prompt, raw, sys_prompt, n_gen, n_ctx, spec_k, parse_special, verbose);
            else                 rc = run_qwen(&qc, tok, &smp, prompt, raw, sys_prompt, n_gen, n_ctx, parse_special, verbose);
            g4_sampler_free(&smp);
        }
        g4_qwen_free(&qc);
        g4_tok_free(tok);
        g4_gguf_close(&mf);
        return rc;
    }

    g4_ctx ctx;
    if (g4_ctx_init(&ctx, &mf, n_ctx, n_threads, kv_type_k, kv_type_v)) return 1;
    if (verbose && (kv_type_k != G4_F16 || kv_type_v != G4_F16))
        fprintf(stderr, "g4: KV cache K=%s V=%s\n", g4_type_name(kv_type_k), g4_type_name(kv_type_v));

    if (server) {
        int rc = g4_server_run(&ctx, 0, tok, &mf, host, port, base_name(model_path),
                               repeat_penalty, no_repeat_ngram, ban_str);
        g4_ctx_free(&ctx);
        g4_tok_free(tok);
        g4_gguf_close(&mf);
        return rc;
    }

    g4_sampler smp;
    g4_sampler_init(&smp, mf.n_vocab);
    smp.temp  = temp  > -2 ? temp  : (mf.meta_temp  >= 0 ? mf.meta_temp  : 1.0f);
    smp.top_k = top_k > -2 ? top_k : (mf.meta_top_k >= 0 ? mf.meta_top_k : 64);
    smp.top_p = top_p > -2 ? top_p : (mf.meta_top_p >= 0 ? mf.meta_top_p : 0.95f);
    smp.repeat_penalty = repeat_penalty;
    smp.repeat_last_n  = repeat_last_n;
    smp.seed = seed;
    apply_anti_loop(&smp, tok, no_repeat_ngram, ban_str);

    fprintf(stderr, "g4: sampling temp=%.2f top_k=%d top_p=%.2f repeat_penalty=%.2f no_repeat_ngram=%d banned=%d tok\n",
            smp.temp, smp.top_k, smp.top_p, smp.repeat_penalty, smp.no_repeat_ngram, smp.n_ban);

    static int ids[65536];
    int n_in = 0;

    char chatbuf[65536];
    if (!raw && !interactive) {
        // gemma4 chat template (see SPEC.md); --think injects <|think|> at the
        // top of the first system turn, enabling the thought channel
        char * p = chatbuf; size_t rem = sizeof(chatbuf);
        int w;
        if (think)           { w = snprintf(p, rem, "<|turn>system\n<|think|>\n%s<turn|>\n", sys_prompt ? sys_prompt : ""); p += w; rem -= w; }
        else if (sys_prompt) { w = snprintf(p, rem, "<|turn>system\n%s<turn|>\n", sys_prompt); p += w; rem -= w; }
        w = snprintf(p, rem, "<|turn>user\n%s<turn|>\n<|turn>model\n", prompt ? prompt : "");
        n_in = g4_tok_encode(tok, chatbuf, ids, 65536, add_bos, true);
    } else if (!interactive) {
        n_in = g4_tok_encode(tok, prompt ? prompt : "", ids, 65536, add_bos, parse_special);
    }

    if (interactive) {
        return chat_loop(&ctx, tok, &smp, &mf, sys_prompt, add_bos, n_ctx, verbose, think);
    }

    if (verbose) {
        fprintf(stderr, "g4: prompt %d tokens:", n_in);
        for (int i = 0; i < n_in && i < 32; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "%s\n", n_in > 32 ? " ..." : "");
    }

    // ---- speculative decoding path (gemma4-assistant MTP draft head) ----
    if (mtp_path) {
        g4_model_file mf_draft;
        if (g4_gguf_open(&mf_draft, mtp_path, verbose)) return 1;
        if (!mf_draft.is_assistant) {
            fprintf(stderr, "g4: --mtp model '%s' is not a gemma4-assistant head\n", mtp_path);
            return 1;
        }
        g4_mtp mtp;
        if (g4_mtp_init(&mtp, &mf_draft, &ctx)) return 1;
        // The head is trained for n_nextn-token prediction, but past the first
        // draft each chained step loses accuracy (it can't attend the unconfirmed
        // tokens), so a short draft amortises the per-round overhead best here.
        int K = draft_len > 0 ? draft_len : 2;
        if (K > 16) K = 16;
        if (verbose) fprintf(stderr, "g4: MTP head loaded (%d layers, draft=%d)\n", mf_draft.n_layer, K);
        int rc = run_mtp(&ctx, tok, &mf, &mtp, ids, n_in, n_ctx, n_gen, K, verbose);
        g4_mtp_free(&mtp);
        g4_gguf_close(&mf_draft);
        g4_sampler_free(&smp);
        g4_ctx_free(&ctx);
        g4_tok_free(tok);
        g4_gguf_close(&mf);
        return rc;
    }

    // prefill (batched; logits only needed for the final token)
    double tp0 = g4_time_ms();
    float * logits = NULL;
    for (int i = 0; i < n_in; i += G4_BATCH_MAX) {
        int n = n_in - i < G4_BATCH_MAX ? n_in - i : G4_BATCH_MAX;
        bool last = i + n >= n_in;
        logits = g4_forward_batch(&ctx, ids + i, n, i, last);
    }
    for (int i = 0; i < n_in; i++) g4_sampler_accept(&smp, ids[i]);
    double tp1 = g4_time_ms();

    // generate
    SetConsoleCtrlHandler(ctrl_handler, TRUE);
    u8stream us = {{0}, 0};
    const int ch_open  = g4_tok_find(tok, "<|channel>");
    const int ch_close = g4_tok_find(tok, "<channel|>");
    bool in_thought = false;
    int hdr_skip = 0;
    int pos = n_in;
    int generated = 0;
    double tg0 = g4_time_ms();
    while ((n_gen < 0 || generated < n_gen) && !g_interrupt) {
        if (verbose) {
            // top-2 logit gap: distinguishes near-tie argmax flips from real bugs
            int b1 = 0, b2 = -1;
            for (int i = 1; i < mf.n_vocab; i++) {
                if (logits[i] > logits[b1]) { b2 = b1; b1 = i; }
                else if (b2 < 0 || logits[i] > logits[b2]) b2 = i;
            }
            fprintf(stderr, "[step %d: top1=%d (%.4f) top2=%d (%.4f) gap=%.4f]\n",
                    generated, b1, logits[b1], b2, logits[b2], logits[b1]-logits[b2]);
        }
        int id = g4_sample(&smp, logits);
        if (g4_tok_is_eog(tok, id)) break;
        if (id == ch_open)       { in_thought = true; hdr_skip = 2; thought_begin(); }
        else if (id == ch_close) { in_thought = false; thought_end(); }
        else {
            char buf[256];
            int len = g4_tok_decode(tok, id, buf, false);
            if (hdr_skip > 0 && ((len == 7 && !memcmp(buf, "thought", 7)) || (len == 1 && buf[0] == '\n'))) hdr_skip--;
            else { hdr_skip = 0; u8_emit(&us, buf, len); }
        }
        g4_sampler_accept(&smp, id);
        if (pos >= n_ctx) { fprintf(stderr, "\n[context full]\n"); break; }
        logits = g4_forward(&ctx, id, pos);
        pos++;
        generated++;
    }
    double tg1 = g4_time_ms();
    if (in_thought) thought_end();
    printf("\n");
    fprintf(stderr, "\ng4: prefill %d tok in %.0f ms (%.2f t/s) | gen %d tok in %.0f ms (%.2f t/s)\n",
            n_in, tp1-tp0, n_in*1000.0/(tp1-tp0),
            generated, tg1-tg0, generated*1000.0/(tg1-tg0));

    g4_sampler_free(&smp);
    g4_ctx_free(&ctx);
    g4_tok_free(tok);
    g4_gguf_close(&mf);
    return 0;
}
