//load the JSON so that I can tune hyperparams
//without having to recompileconfig loader
//no checks needed since I assume config json is valid.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

struct Config{
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

int config_from_file(const char* path, struct Config* out);
void config_log(const struct Config* c);

//helpers
static char* read_all(const char* path){
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char* buf = (char*)malloc((size_t)n + 1);
    fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

static const char* skip_ws(const char* p){
    while (*p && (unsigned char)*p <= 32) ++p;
    return p;
}

static const char* find_key(const char* json, const char* key){
    size_t klen = strlen(key);
    const char* p = json;
    while ((p = strstr(p, key)) != NULL) {
        if (p > json && p[-1] == '"' && p[klen] == '"'){
            const char* q = skip_ws(p + klen + 1);
            if (*q == ':') return q + 1;
        }
        p += klen;
    }
    return NULL; //not expected (assume valid)
}

static void parse_json_string(const char* p, char* out, size_t cap){
    p = skip_ws(p);
    if (*p == '"') ++p;
    size_t o = 0;
    while (*p && *p != '"') {
        char c = *p++;
        if (c == '\\') {
            char e = *p ? *p++ : '\0';
            if (e == 'n') c = '\n';
            else if (e == 't') c = '\t';
            else if (e == 'r') c = '\r';
            else c = e;
        }
        if (o + 1 < cap) out[o++] = c;
    }
    out[(o < cap) ? o : cap - 1] = '\0';
}

static void parse_json_int(const char* p, int* out){
    p = skip_ws(p);
    *out = (int)strtol(p, NULL, 10);
}

static void parse_json_double(const char* p, double* out){
    p = skip_ws(p);
    *out = strtod(p, NULL);
}


int config_from_file(const char* path, struct Config* out){
    char* buf = read_all(path);
    parse_json_string(find_key(buf, "train_path"), out->train_path, sizeof(out->train_path));
    parse_json_string(find_key(buf, "val_path"), out->val_path,   sizeof(out->val_path));
    parse_json_int(find_key(buf, "seq_len"), &out->seq_len);
    parse_json_int(find_key(buf, "batch_size"), &out->batch_size);
    parse_json_int(find_key(buf, "vocab_size"), &out->vocab_size);
    parse_json_int(find_key(buf, "d_model"), &out->d_model);
    parse_json_int(find_key(buf, "ffn_mult"), &out->ffn_mult);
    parse_json_double(find_key(buf, "lr"), &out->lr);
    parse_json_double(find_key(buf, "weight_decay"), &out->weight_decay);
    parse_json_double(find_key(buf, "clip_grad_norm"), &out->clip_grad_norm);
    parse_json_int(find_key(buf, "seed"), &out->seed);
    parse_json_int(find_key(buf, "top_k"), &out->top_k);

    free(buf);
    return 0;
}

void config_log(const struct Config* c){
    printf("train_path: %s\n", c->train_path);
    printf("val_path: %s\n", c->val_path);
    printf("seq_len: %d\n", c->seq_len);
    printf("batch_size: %d\n", c->batch_size);
    printf("vocab_size: %d\n", c->vocab_size);
    printf("d_model: %d\n", c->d_model);
    printf("ffn_mult: %d\n", c->ffn_mult);
    printf("lr: %.8f\n", c->lr);
    printf("weight_decay: %.8f\n", c->weight_decay);
    printf("clip_grad_norm: %.8f\n", c->clip_grad_norm);
    printf("seed: %d\n", c->seed);
    printf("top_k: %d\n", c->top_k);
}

