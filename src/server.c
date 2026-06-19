// server.c — minimal HTTP/JSON inference server over Winsock (TCP sockets).
// Endpoints:
//   GET  /health                 -> {"status":"ok","model":"..."}
//   POST /completion             -> llama.cpp-style raw completion
//   POST /v1/chat/completions    -> OpenAI-compatible chat (applies gemma4 template)
// Both POST endpoints accept "stream": true for Server-Sent-Events streaming.
// Requests are served one at a time (single shared context); each starts fresh.
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdarg.h>
#include <time.h>
#include "g4run.h"

#pragma comment(lib, "ws2_32.lib")

// ----------------------------------------------------------- dyn string -----
typedef struct { char * b; size_t n, cap; } dstr;
static void ds_grow(dstr * d, size_t add) {
    if (d->n + add + 1 > d->cap) {
        d->cap = (d->n + add + 1) * 2;
        d->b = realloc(d->b, d->cap);
    }
}
static void ds_put(dstr * d, const char * s, size_t n) { ds_grow(d, n); memcpy(d->b + d->n, s, n); d->n += n; d->b[d->n] = 0; }
static void ds_puts(dstr * d, const char * s) { ds_put(d, s, strlen(s)); }
static void ds_putc(dstr * d, char c) { ds_grow(d, 1); d->b[d->n++] = c; d->b[d->n] = 0; }
static void ds_printf(dstr * d, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    char tmp[512];
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n >= 0 && n < (int)sizeof(tmp)) ds_put(d, tmp, n);
}
static void ds_json_str(dstr * d, const char * s, int n) {  // emit JSON-escaped, no quotes
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  ds_puts(d, "\\\""); break;
            case '\\': ds_puts(d, "\\\\"); break;
            case '\n': ds_puts(d, "\\n"); break;
            case '\r': ds_puts(d, "\\r"); break;
            case '\t': ds_puts(d, "\\t"); break;
            default:
                if (c < 0x20) ds_printf(d, "\\u%04x", c);
                else ds_putc(d, (char)c);
        }
    }
}

// ------------------------------------------------------------ JSON parse ----
enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ };
typedef struct {
    int type;
    const char * key; int klen;
    const char * str; int slen;   // raw (escaped) string slice
    double num; int bval;
    int child, sib;
} jnode;
typedef struct { const char * p, * end; jnode * n; int cap, cnt, err; } jparse;

static int  jalloc(jparse * J) {
    if (J->cnt >= J->cap) { J->err = 1; return -1; }
    int i = J->cnt++; memset(&J->n[i], 0, sizeof(jnode)); J->n[i].child = J->n[i].sib = -1; return i;
}
static void jws(jparse * J) { while (J->p < J->end && (*J->p==' '||*J->p=='\t'||*J->p=='\n'||*J->p=='\r')) J->p++; }
static int  jvalue(jparse * J);
static void jstr_raw(jparse * J, const char ** s, int * sl) {
    *s = NULL; *sl = 0;
    if (J->p >= J->end || *J->p != '"') { J->err = 1; return; }
    J->p++; const char * st = J->p;
    while (J->p < J->end && *J->p != '"') { if (*J->p == '\\' && J->p+1 < J->end) J->p++; J->p++; }
    *s = st; *sl = (int)(J->p - st);
    if (J->p < J->end) J->p++;
}
static int jvalue(jparse * J) {
    jws(J);
    if (J->p >= J->end) { J->err = 1; return -1; }
    char c = *J->p;
    if (c == '{') {
        int o = jalloc(J); if (o < 0) return -1; J->n[o].type = JOBJ;
        J->p++; jws(J);
        if (J->p < J->end && *J->p == '}') { J->p++; return o; }
        int last = -1;
        for (;;) {
            jws(J);
            const char * k; int kl; jstr_raw(J, &k, &kl);
            jws(J); if (J->p >= J->end || *J->p != ':') { J->err = 1; return o; } J->p++;
            int v = jvalue(J); if (v < 0) return o;
            J->n[v].key = k; J->n[v].klen = kl;
            if (last < 0) J->n[o].child = v; else J->n[last].sib = v; last = v;
            jws(J);
            if (J->p < J->end && *J->p == ',') { J->p++; continue; }
            if (J->p < J->end && *J->p == '}') { J->p++; break; }
            J->err = 1; break;
        }
        return o;
    }
    if (c == '[') {
        int a = jalloc(J); if (a < 0) return -1; J->n[a].type = JARR;
        J->p++; jws(J);
        if (J->p < J->end && *J->p == ']') { J->p++; return a; }
        int last = -1;
        for (;;) {
            int v = jvalue(J); if (v < 0) return a;
            if (last < 0) J->n[a].child = v; else J->n[last].sib = v; last = v;
            jws(J);
            if (J->p < J->end && *J->p == ',') { J->p++; continue; }
            if (J->p < J->end && *J->p == ']') { J->p++; break; }
            J->err = 1; break;
        }
        return a;
    }
    if (c == '"') {
        int s = jalloc(J); if (s < 0) return -1; J->n[s].type = JSTR;
        jstr_raw(J, &J->n[s].str, &J->n[s].slen); return s;
    }
    if (c == 't' || c == 'f') {
        int b = jalloc(J); if (b < 0) return -1; J->n[b].type = JBOOL; J->n[b].bval = (c == 't');
        while (J->p < J->end && *J->p >= 'a' && *J->p <= 'z') J->p++; return b;
    }
    if (c == 'n') { int z = jalloc(J); if (z<0) return -1; J->n[z].type = JNULL; while (J->p<J->end && *J->p>='a'&&*J->p<='z') J->p++; return z; }
    // number
    { int nn = jalloc(J); if (nn < 0) return -1; J->n[nn].type = JNUM;
      char * e; J->n[nn].num = strtod(J->p, &e); if (e == J->p) { J->err = 1; } J->p = e; return nn; }
}
static int jget(jparse * J, int obj, const char * key) {
    if (obj < 0 || J->n[obj].type != JOBJ) return -1;
    int kl = (int)strlen(key);
    for (int c = J->n[obj].child; c >= 0; c = J->n[c].sib)
        if (J->n[c].klen == kl && !memcmp(J->n[c].key, key, kl)) return c;
    return -1;
}
static double jnum(jparse * J, int obj, const char * key, double def) {
    int i = jget(J, obj, key);
    if (i < 0) return def;
    if (J->n[i].type == JNUM)  return J->n[i].num;
    if (J->n[i].type == JBOOL) return J->n[i].bval;
    return def;
}
static int utf8_enc(int cp, char * o) {
    if (cp < 0x80) { o[0]=(char)cp; return 1; }
    if (cp < 0x800) { o[0]=(char)(0xC0|(cp>>6)); o[1]=(char)(0x80|(cp&0x3F)); return 2; }
    o[0]=(char)(0xE0|(cp>>12)); o[1]=(char)(0x80|((cp>>6)&0x3F)); o[2]=(char)(0x80|(cp&0x3F)); return 3;
}
static int hex4(const char * s) {
    int v = 0; for (int i = 0; i < 4; i++) { char c=s[i]; v<<=4;
        if (c>='0'&&c<='9') v|=c-'0'; else if (c>='a'&&c<='f') v|=c-'a'+10; else if (c>='A'&&c<='F') v|=c-'A'+10; }
    return v;
}
// unescape a raw JSON string slice into out (NUL-terminated); returns length
static int junesc(const char * s, int sl, char * out, int cap) {
    int o = 0;
    for (int i = 0; i < sl && o < cap-4; i++) {
        char c = s[i];
        if (c == '\\' && i+1 < sl) {
            char e = s[++i];
            switch (e) {
                case 'n': out[o++]='\n'; break; case 't': out[o++]='\t'; break;
                case 'r': out[o++]='\r'; break; case 'b': out[o++]='\b'; break;
                case 'f': out[o++]='\f'; break; case '/': out[o++]='/'; break;
                case '"': out[o++]='"'; break; case '\\': out[o++]='\\'; break;
                case 'u': if (i+4 < sl) { o += utf8_enc(hex4(s+i+1), out+o); i += 4; } break;
                default: out[o++] = e;
            }
        } else out[o++] = c;
    }
    out[o] = 0; return o;
}

// ------------------------------------------------------------- net I/O ------
static volatile SOCKET g_listen = INVALID_SOCKET;
static volatile int    g_stop = 0;

static int send_all(SOCKET s, const char * b, int n) {
    int off = 0;
    while (off < n) { int k = send(s, b + off, n - off, 0); if (k <= 0) return -1; off += k; }
    return 0;
}
static void send_simple(SOCKET s, int code, const char * status, const char * ctype, const char * body, int blen) {
    dstr h = {0};
    ds_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
                  "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", code, status, ctype, blen);
    send_all(s, h.b, (int)h.n);
    if (blen) send_all(s, body, blen);
    free(h.b);
}

// ------------------------------------------------- generation (shared) ------
typedef struct {
    g4_ctx * ctx;        // gemma context (NULL for qwen)
    qwen_ctx * qc;       // qwen35 context (NULL for gemma)
    int is_qwen;
    g4_tok * tok; g4_model_file * mf;
    const char * model_name;
    int n_ctx; bool add_bos;
    // defaults from model metadata
    float d_temp, d_top_p; int d_top_k;
    // anti-loop defaults (resolved at startup; requests may override the first two)
    float d_repeat_penalty; int d_no_repeat_ngram;
    int ban_ids[64], n_ban;
} server_ctx;

// forward dispatch: gemma g4_ctx vs qwen35 qwen_ctx
static float * sv_fwd_batch(server_ctx * S, int * ids, int nb, int pos, bool wl) {
    return S->is_qwen ? g4_qwen_forward_batch(S->qc, ids, nb, pos, wl) : g4_forward_batch(S->ctx, ids, nb, pos, wl);
}
static float * sv_fwd1(server_ctx * S, int id, int pos) {
    return S->is_qwen ? g4_qwen_forward(S->qc, id, pos) : g4_forward(S->ctx, id, pos);
}

// is_thought=1 -> reasoning channel, 0 -> visible content
typedef void (*emit_fn)(const char * piece, int n, int is_thought, void * ud);

// "think"/"thinking" enable check, tolerant of bool / number / object
// ({"type":"enabled"}) / string ("enabled"/"disabled"/"none") forms.
static int think_enabled(jparse * J, int root) {
    int i = jget(J, root, "think");
    if (i < 0) i = jget(J, root, "thinking");
    if (i < 0) return 0;
    int t = J->n[i].type;
    if (t == JBOOL) return J->n[i].bval;
    if (t == JNUM)  return J->n[i].num != 0;
    if (t == JOBJ) {
        int ty = jget(J, i, "type");
        if (ty >= 0 && J->n[ty].type == JSTR)
            return !(J->n[ty].slen == 8 && !memcmp(J->n[ty].str, "disabled", 8));
        return 1;   // object present without an explicit type -> enabled
    }
    if (t == JSTR) return !(J->n[i].slen == 8 && !memcmp(J->n[i].str, "disabled", 8)) &&
                          !(J->n[i].slen == 4 && !memcmp(J->n[i].str, "none", 4));
    return 0;
}

// run prompt ids -> generation; visible content and reasoning are emitted on
// separate channels (is_thought flag). Returns tokens predicted.
// stop_reason: 0=eog/stop, 1=length
static int sv_generate(server_ctx * S, int * ids, int n_in, g4_sampler * smp,
                       int max_tokens, emit_fn cb, void * ud, int * stop_reason) {
    if (S->is_qwen) g4_qwen_reset(S->qc);   // fresh recurrent state per request
    float * logits = NULL;
    for (int i = 0; i < n_in; i += G4_BATCH_MAX) {
        int nb = n_in - i < G4_BATCH_MAX ? n_in - i : G4_BATCH_MAX;
        logits = sv_fwd_batch(S, ids + i, nb, i, i + nb >= n_in);
    }
    for (int i = 0; i < n_in; i++) g4_sampler_accept(smp, ids[i]);

    const int ch_open  = g4_tok_find(S->tok, "<|channel>");
    const int ch_close = g4_tok_find(S->tok, "<channel|>");
    int pos = n_in, generated = 0;
    bool in_thought = false;
    int  hdr_skip = 0;          // drop the "thought\n" channel label after <|channel>
    *stop_reason = 0;
    // separate UTF-8 reassembly per channel so chunks never split a codepoint
    char pend[2][8]; int pn[2] = {0, 0};

    while (pos < S->n_ctx - 1) {
        if (max_tokens >= 0 && generated >= max_tokens) { *stop_reason = 1; break; }
        int id = g4_sample(smp, logits);
        if (g4_tok_is_eog(S->tok, id)) break;
        if (id == ch_open)       { in_thought = true;  hdr_skip = 2; }
        else if (id == ch_close) { in_thought = false; }
        else {
            char buf[256];
            int len = g4_tok_decode(S->tok, id, buf, false);
            // skip the channel header tokens ("thought" then "\n")
            if (in_thought && hdr_skip > 0 &&
                ((len == 7 && !memcmp(buf, "thought", 7)) || (len == 1 && buf[0] == '\n'))) {
                hdr_skip--;
            } else {
                hdr_skip = 0;
                int ch = in_thought ? 1 : 0;
                for (int i = 0; i < len; i++) {
                    pend[ch][pn[ch]++] = buf[i];
                    unsigned char c0 = (unsigned char)pend[ch][0];
                    int need = c0 < 0x80 ? 1 : c0 < 0xE0 ? 2 : c0 < 0xF0 ? 3 : 4;
                    if (pn[ch] >= need) { cb(pend[ch], need, ch, ud); memmove(pend[ch], pend[ch]+need, pn[ch]-need); pn[ch] -= need; }
                }
            }
        }
        g4_sampler_accept(smp, id);
        logits = sv_fwd1(S, id, pos);
        pos++; generated++;
    }
    if (pn[0]) cb(pend[0], pn[0], 0, ud);
    if (pn[1]) cb(pend[1], pn[1], 1, ud);
    return generated;
}

static void set_sampler(server_ctx * S, g4_sampler * smp, jparse * J, int root) {
    g4_sampler_reset(smp);
    smp->temp  = (float)jnum(J, root, "temperature", S->d_temp);
    smp->top_k = (int)  jnum(J, root, "top_k", S->d_top_k);
    smp->top_p = (float)jnum(J, root, "top_p", S->d_top_p);
    smp->repeat_penalty = (float)jnum(J, root, "repeat_penalty", S->d_repeat_penalty);
    smp->repeat_last_n  = (int)  jnum(J, root, "repeat_last_n", 128);
    smp->freq_penalty   = (float)jnum(J, root, "frequency_penalty", 0.0);
    smp->presence_penalty = (float)jnum(J, root, "presence_penalty", 0.0);
    smp->no_repeat_ngram = (int) jnum(J, root, "no_repeat_ngram", S->d_no_repeat_ngram);
    for (int j = 0; j < S->n_ban; j++) g4_sampler_ban(smp, S->ban_ids[j]);
    double seed = jnum(J, root, "seed", -1);
    smp->seed = seed < 0 ? (uint64_t)time(NULL) ^ ((uint64_t)GetTickCount() << 20) : (uint64_t)seed;
    memset(smp->rng, 0, sizeof(smp->rng));
}

// render OpenAI messages[] into a gemma4 prompt string
static void render_messages(server_ctx * S, jparse * J, int msgs, bool think, dstr * out) {
    // extract first system message content
    char sysbuf[8192]; sysbuf[0] = 0;
    int have_sys = 0;
    if (msgs >= 0 && J->n[msgs].type == JARR) {
        for (int m = J->n[msgs].child; m >= 0; m = J->n[m].sib) {
            int ri = jget(J, m, "role"), ci = jget(J, m, "content");
            if (ri < 0 || ci < 0) continue;
            if (J->n[ri].slen == 6 && !memcmp(J->n[ri].str, "system", 6)) {
                junesc(J->n[ci].str, J->n[ci].slen, sysbuf, sizeof(sysbuf));
                have_sys = 1; break;
            }
        }
    }
    if (think)       { ds_puts(out, "<|turn>system\n<|think|>\n"); if (have_sys) ds_puts(out, sysbuf); ds_puts(out, "<turn|>\n"); }
    else if (have_sys) { ds_puts(out, "<|turn>system\n"); ds_puts(out, sysbuf); ds_puts(out, "<turn|>\n"); }

    if (msgs >= 0 && J->n[msgs].type == JARR) {
        char cbuf[16384];
        for (int m = J->n[msgs].child; m >= 0; m = J->n[m].sib) {
            int ri = jget(J, m, "role"), ci = jget(J, m, "content");
            if (ri < 0 || ci < 0) continue;
            const char * r = J->n[ri].str; int rl = J->n[ri].slen;
            if (rl == 6 && !memcmp(r, "system", 6)) continue;
            junesc(J->n[ci].str, J->n[ci].slen, cbuf, sizeof(cbuf));
            if (rl == 9 && !memcmp(r, "assistant", 9)) ds_printf(out, "<|turn>model\n%s<turn|>\n", cbuf);
            else if (rl == 5 && !memcmp(r, "model", 5)) ds_printf(out, "<|turn>model\n%s<turn|>\n", cbuf);
            else ds_printf(out, "<|turn>user\n%s<turn|>\n", cbuf);  // user (default)
        }
    }
    ds_puts(out, "<|turn>model\n");
}

// render OpenAI messages[] into a qwen35 (<|im_start|>) prompt string
static void render_messages_qwen(jparse * J, int msgs, dstr * out) {
    char cbuf[16384];
    if (msgs >= 0 && J->n[msgs].type == JARR) {
        for (int m = J->n[msgs].child; m >= 0; m = J->n[m].sib) {
            int ri = jget(J, m, "role"), ci = jget(J, m, "content");
            if (ri < 0 || ci < 0) continue;
            const char * r = J->n[ri].str; int rl = J->n[ri].slen;
            junesc(J->n[ci].str, J->n[ci].slen, cbuf, sizeof(cbuf));
            if      (rl == 6 && !memcmp(r, "system", 6))    ds_printf(out, "<|im_start|>system\n%s<|im_end|>\n", cbuf);
            else if (rl == 9 && !memcmp(r, "assistant", 9)) ds_printf(out, "<|im_start|>assistant\n%s<|im_end|>\n", cbuf);
            else                                            ds_printf(out, "<|im_start|>user\n%s<|im_end|>\n", cbuf);
        }
    }
    ds_puts(out, "<|im_start|>assistant\n");
}

// ---------------------------------------------------- streaming emit ud -----
typedef struct { SOCKET s; int chat; int ok; const char * id; } stream_ud;
static void emit_stream(const char * piece, int n, int is_thought, void * ud) {
    stream_ud * u = ud;
    if (!u->ok) return;
    dstr d = {0};
    ds_puts(&d, "data: ");
    if (u->chat) {
        const char * field = is_thought ? "reasoning_content" : "content";
        ds_printf(&d, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"%s\":\"", u->id, field);
        ds_json_str(&d, piece, n);
        ds_puts(&d, "\"}}]}");
    } else {
        ds_puts(&d, is_thought ? "{\"reasoning_content\":\"" : "{\"content\":\"");
        ds_json_str(&d, piece, n);
        ds_puts(&d, "\",\"stop\":false}");
    }
    ds_puts(&d, "\n\n");
    if (send_all(u->s, d.b, (int)d.n) < 0) u->ok = 0;
    free(d.b);
}
// non-streaming: accumulate content and reasoning into separate buffers
typedef struct { dstr content, reasoning; } accum_ud;
static void emit_accum(const char * piece, int n, int is_thought, void * ud) {
    accum_ud * a = ud;
    ds_put(is_thought ? &a->reasoning : &a->content, piece, n);
}

static void stream_headers(SOCKET s) {
    const char * h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     "Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    send_all(s, h, (int)strlen(h));
}

// ------------------------------------------------------ request handlers ----
static int g_req_counter = 0;
static void make_id(char * out) { snprintf(out, 40, "chatcmpl-%08x%04x", (unsigned)time(NULL), (unsigned)(g_req_counter++ & 0xffff)); }

static void handle_completion(server_ctx * S, SOCKET sock, jparse * J, int root, int chat) {
    static int ids[65536];
    char idbuf[40]; make_id(idbuf);

    bool think = think_enabled(J, root);
    bool stream = jnum(J, root, "stream", 0) != 0;
    int max_tokens = (int)jnum(J, root, chat ? "max_tokens" : "n_predict", 256);

    // build prompt
    dstr prompt = {0};
    bool parse_special;
    if (chat) {
        if (S->is_qwen) render_messages_qwen(J, jget(J, root, "messages"), &prompt);
        else            render_messages(S, J, jget(J, root, "messages"), think, &prompt);
        parse_special = true;
    } else {
        int pi = jget(J, root, "prompt");
        if (pi >= 0 && J->n[pi].type == JSTR) {
            ds_grow(&prompt, J->n[pi].slen + 1);
            prompt.n = junesc(J->n[pi].str, J->n[pi].slen, prompt.b, (int)prompt.cap);
        }
        parse_special = jnum(J, root, "parse_special", 0) != 0;
    }

    int n_in = g4_tok_encode(S->tok, prompt.b ? prompt.b : "", ids, 65536, S->add_bos, parse_special);
    free(prompt.b);
    if (n_in + 8 >= S->n_ctx) { send_simple(sock, 400, "Bad Request", "application/json",
        "{\"error\":\"prompt too long for context\"}", 39); return; }

    g4_sampler smp; g4_sampler_init(&smp, S->mf->n_vocab);
    set_sampler(S, &smp, J, root);
    if (max_tokens > S->n_ctx - n_in - 1) max_tokens = S->n_ctx - n_in - 1;

    int stop_reason = 0, n_pred = 0;
    if (stream) {
        stream_headers(sock);
        stream_ud su = { sock, chat, 1, idbuf };
        n_pred = sv_generate(S, ids, n_in, &smp, max_tokens, emit_stream, &su, &stop_reason);
        // final event
        dstr f = {0};
        const char * fr = stop_reason ? "length" : "stop";
        if (chat) ds_printf(&f, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n", idbuf, fr, n_in, n_pred, n_in + n_pred);
        else      ds_printf(&f, "data: {\"content\":\"\",\"stop\":true,\"tokens_predicted\":%d,\"tokens_evaluated\":%d}\n\n", n_pred, n_in);
        ds_puts(&f, "data: [DONE]\n\n");
        if (su.ok) send_all(sock, f.b, (int)f.n);
        free(f.b);
    } else {
        accum_ud body = {0};
        n_pred = sv_generate(S, ids, n_in, &smp, max_tokens, emit_accum, &body, &stop_reason);
        dstr out = {0};
        const char * fr = stop_reason ? "length" : "stop";
        if (chat) {
            ds_printf(&out, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%lld,\"model\":\"", idbuf, (long long)time(NULL));
            ds_json_str(&out, S->model_name, (int)strlen(S->model_name));
            ds_puts(&out, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"");
            ds_json_str(&out, body.content.b ? body.content.b : "", (int)body.content.n);
            ds_puts(&out, "\",\"reasoning_content\":\"");
            ds_json_str(&out, body.reasoning.b ? body.reasoning.b : "", (int)body.reasoning.n);
            ds_printf(&out, "\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                      fr, n_in, n_pred, n_in + n_pred);
        } else {
            ds_puts(&out, "{\"content\":\"");
            ds_json_str(&out, body.content.b ? body.content.b : "", (int)body.content.n);
            ds_puts(&out, "\",\"reasoning_content\":\"");
            ds_json_str(&out, body.reasoning.b ? body.reasoning.b : "", (int)body.reasoning.n);
            ds_printf(&out, "\",\"stop\":true,\"model\":\"");
            ds_json_str(&out, S->model_name, (int)strlen(S->model_name));
            ds_printf(&out, "\",\"tokens_predicted\":%d,\"tokens_evaluated\":%d}", n_pred, n_in);
        }
        send_simple(sock, 200, "OK", "application/json", out.b, (int)out.n);
        free(body.content.b); free(body.reasoning.b); free(out.b);
    }
    g4_sampler_free(&smp);
    fprintf(stderr, "[server] %s: %d prompt + %d gen tok%s\n",
            chat ? "chat" : "completion", n_in, n_pred, stream ? " (stream)" : "");
}

// read a full HTTP request (headers + body by Content-Length). returns 0 on ok.
static int read_request(SOCKET s, dstr * req, int * body_off) {
    char tmp[8192];
    for (;;) {
        char * he = req->b ? strstr(req->b, "\r\n\r\n") : NULL;   // header terminator
        if (he) {
            int hdr_end = (int)(he - req->b) + 4;
            int clen = 0;
            char * cl = strstr(req->b, "Content-Length:");
            if (!cl) cl = strstr(req->b, "content-length:");
            if (cl) clen = atoi(cl + 15);
            if ((int)req->n >= hdr_end + clen) { *body_off = hdr_end; return 0; }
        }
        int k = recv(s, tmp, sizeof(tmp), 0);
        if (k <= 0) return -1;
        ds_put(req, tmp, k);
        if (req->n > 64u*1024*1024) return -1;   // 64 MB guard
    }
}

static void handle_conn(server_ctx * S, SOCKET sock) {
    int one = 1; setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(one));
    dstr req = {0}; int body_off = 0;
    if (read_request(sock, &req, &body_off)) { free(req.b); return; }

    // method + path from request line
    char method[8] = {0}, path[256] = {0};
    sscanf(req.b, "%7s %255s", method, path);

    if (!strcmp(method, "OPTIONS")) {  // CORS preflight
        const char * h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(sock, h, (int)strlen(h)); free(req.b); return;
    }
    if (!strcmp(method, "GET") && (!strcmp(path, "/health") || !strcmp(path, "/"))) {
        dstr o = {0}; ds_puts(&o, "{\"status\":\"ok\",\"model\":\""); ds_json_str(&o, S->model_name, (int)strlen(S->model_name)); ds_puts(&o, "\"}");
        send_simple(sock, 200, "OK", "application/json", o.b, (int)o.n); free(o.b); free(req.b); return;
    }

    int is_chat = !strcmp(path, "/v1/chat/completions");
    int is_comp = !strcmp(path, "/completion") || !strcmp(path, "/completions");
    if (!strcmp(method, "POST") && (is_chat || is_comp)) {
        static jnode nodes[16384];
        jparse J = { req.b + body_off, req.b + req.n, nodes, 16384, 0, 0 };
        int root = jvalue(&J);
        if (J.err || root < 0) { send_simple(sock, 400, "Bad Request", "application/json", "{\"error\":\"invalid JSON\"}", 24); free(req.b); return; }
        handle_completion(S, sock, &J, root, is_chat);
        free(req.b); return;
    }
    send_simple(sock, 404, "Not Found", "application/json", "{\"error\":\"unknown endpoint\"}", 28);
    free(req.b);
}

static BOOL WINAPI srv_ctrl(DWORD t) {
    if (t == CTRL_C_EVENT || t == CTRL_CLOSE_EVENT) {
        g_stop = 1;
        if (g_listen != INVALID_SOCKET) { closesocket(g_listen); g_listen = INVALID_SOCKET; }
        return TRUE;
    }
    return FALSE;
}

int g4_server_run(void * vctx, int is_qwen, g4_tok * tok, g4_model_file * mf,
                  const char * host, int port, const char * model_name,
                  float repeat_penalty, int no_repeat_ngram, const char * ban_str) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa)) { fprintf(stderr, "server: WSAStartup failed\n"); return 1; }

    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == INVALID_SOCKET) { fprintf(stderr, "server: socket() failed\n"); return 1; }
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (char *)&one, sizeof(one));

    struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(ls, (struct sockaddr *)&addr, sizeof(addr))) { fprintf(stderr, "server: bind %s:%d failed (%d)\n", host, port, WSAGetLastError()); return 1; }
    if (listen(ls, 16)) { fprintf(stderr, "server: listen failed\n"); return 1; }
    g_listen = ls;
    SetConsoleCtrlHandler(srv_ctrl, TRUE);

    g4_ctx   * gctx = is_qwen ? NULL : (g4_ctx *)vctx;
    qwen_ctx * qctx = is_qwen ? (qwen_ctx *)vctx : NULL;
    int n_ctx = is_qwen ? qctx->n_ctx : gctx->n_ctx;
    server_ctx S = {
        gctx, qctx, is_qwen, tok, mf, model_name, n_ctx, mf->add_bos,
        mf->meta_temp >= 0 ? mf->meta_temp : (is_qwen ? 0.7f : 1.0f),
        mf->meta_top_p >= 0 ? mf->meta_top_p : (is_qwen ? 0.8f : 0.95f),
        mf->meta_top_k >= 0 ? mf->meta_top_k : (is_qwen ? 20 : 64),
        repeat_penalty >= 0 ? repeat_penalty : (is_qwen ? 1.1f : 1.0f),  // mild anti-loop default for qwen
        no_repeat_ngram, {0}, 0,
    };
    // tokenize --ban triggers once (both bare and space-prefixed forms) into ban ids
    if (ban_str && *ban_str && strcmp(ban_str, "none")) {
        char buf[1024]; snprintf(buf, sizeof(buf), "%s", ban_str);
        for (char * t = strtok(buf, ","); t && S.n_ban < 60; t = strtok(NULL, ",")) {
            while (*t == ' ') t++;
            if (!*t) continue;
            char var[128]; int ids[16];
            for (int v = 0; v < 2; v++) {
                snprintf(var, sizeof(var), v ? " %s" : "%s", t);
                int n = g4_tok_encode(tok, var, ids, 16, false, false);
                for (int j = 0; j < n && S.n_ban < 64; j++) S.ban_ids[S.n_ban++] = ids[j];
            }
        }
    }

    fprintf(stderr, "g4run server listening on http://%s:%d\n", host, port);
    if (S.d_repeat_penalty != 1.0f || S.d_no_repeat_ngram >= 2 || S.n_ban)
        fprintf(stderr, "  anti-loop: repeat_penalty=%.2f no_repeat_ngram=%d banned=%d tok\n",
                S.d_repeat_penalty, S.d_no_repeat_ngram, S.n_ban);
    fprintf(stderr, "  GET  /health\n  POST /completion            {\"prompt\":\"...\",\"n_predict\":N,\"stream\":bool}\n");
    fprintf(stderr, "  POST /v1/chat/completions   {\"messages\":[...],\"max_tokens\":N,\"stream\":bool,\"think\":bool}\n");
    fprintf(stderr, "Ctrl+C to stop.\n\n");

    while (!g_stop) {
        SOCKET c = accept(ls, NULL, NULL);
        if (c == INVALID_SOCKET) { if (g_stop) break; continue; }
        handle_conn(&S, c);
        closesocket(c);
    }
    if (g_listen != INVALID_SOCKET) closesocket(g_listen);
    WSACleanup();
    fprintf(stderr, "server stopped.\n");
    return 0;
}
