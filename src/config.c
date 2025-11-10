//load the JSON so that I can tune hyperparams
//without having to recompileconfig loader
//no checks needed since I assume config json is valid.

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

struct Config {
    char train_path[512];
    char val_path[512];
    int seq_len;
    int batch_size;
    int vocab_size;
    int d_model;
    int ffn_mult;
    double lr;
    double weight_decay;
    double clip_grad_norm;
    int seed;
    int top_k;
};

int  config_from_file(const char* path, struct Config* out);
void config_log(const struct Config* c);

static int read_text_file(const char* path, char** out_buf, size_t* out_len) {
    *out_buf = NULL; *out_len = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return -1; }
    rewind(f);
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return -1; }
    size_t rd = (n == 0) ? 0 : fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (rd != (size_t)n) { free(buf); return -1; }
    buf[n] = '\0';
    *out_buf = buf; *out_len = (size_t)n;
    return 0;
}

static const char* skip_ws(const char* p) {
    while (*p && (unsigned char)*p <= 32) ++p;
    return p;
}

static const char* find_key(const char* json, const char* key) {
    size_t klen = strlen(key);
    const char* p = json;
    while ((p = strstr(p, key)) != NULL) {
        if (p > json && p[-1] == '"' && p[klen] == '"') {
            const char* q = skip_ws(p + klen + 1);
            if (*q == ':') return q + 1;
        }
        p += klen;
    }
    return NULL;
}

static void parse_json_string(const char* p, char* out, size_t out_cap) {
    p = skip_ws(p);
    if (*p == '"') ++p;
    size_t o = 0;
    while (*p && *p != '"') {
        char c = *p++;
        if (c == '\\') {
            char esc = *p ? *p++ : '\0';
            if (esc == 'n') c = '\n';
            else if (esc == 't') c = '\t';
            else if (esc == 'r') c = '\r';
            else c = esc;
        }
        if (o + 1 < out_cap) out[o++] = c;
    }
    if (o < out_cap) out[o] = '\0'; else out[out_cap-1] = '\0';
}

static void parse_json_int(const char* p, int* out) {
    p = skip_ws(p);
    *out = (int)strtol(p, NULL, 10);
}

static void parse_json_double(const char* p, double* out) {
    p = skip_ws(p);
    *out = strtod(p, NULL);
}


int config_from_file(const char* path, struct Config* out) {
    if (!path || !out) return -1;
    char* buf = NULL; size_t len = 0;
    if (read_text_file(path, &buf, &len) != 0) return -1;

    //assume all keys exist and JSON is valid
    parse_json_string(find_key(buf,"train_path"), out->train_path, sizeof(out->train_path));
    parse_json_string(find_key(buf,"val_path"), out->val_path, sizeof(out->val_path));
    parse_json_int(find_key(buf,"seq_len"), &out->seq_len);
    parse_json_int(find_key(buf,"batch_size"), &out->batch_size);
    parse_json_int(find_key(buf,"vocab_size"), &out->vocab_size);
    parse_json_int(find_key(buf,"d_model"), &out->d_model);
    parse_json_int(find_key(buf,"ffn_mult"), &out->ffn_mult);
    parse_json_double(find_key(buf,"lr"), &out->lr);
    parse_json_double(find_key(buf,"weight_decay"), &out->weight_decay);
    parse_json_double(find_key(buf,"clip_grad_norm"), &out->clip_grad_norm);
    parse_json_int(find_key(buf,"seed"), &out->seed);
    parse_json_int(find_key(buf,"top_k"), &out->top_k);

    free(buf);
    return 0;
}

void config_log(const struct Config* c) {
    if (!c) return;
    fprintf(stderr, "train_path: %s\n", c->train_path);
    fprintf(stderr, "val_path: %s\n", c->val_path);
    fprintf(stderr, "seq_len: %d\n", c->seq_len);
    fprintf(stderr, "batch_size: %d\n", c->batch_size);
    fprintf(stderr, "vocab_size: %d\n", c->vocab_size);
    fprintf(stderr, "d_model: %d\n", c->d_model);
    fprintf(stderr, "ffn_mult: %d\n", c->ffn_mult);
    fprintf(stderr, "lr: %.8f\n", c->lr);
    fprintf(stderr, "weight_decay: %.8f\n", c->weight_decay);
    fprintf(stderr, "clip_grad_norm: %.8f\n", c->clip_grad_norm);
    fprintf(stderr, "seed: %d\n", c->seed);
    fprintf(stderr, "top_k: %d\n", c->top_k);
}

