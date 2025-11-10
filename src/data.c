//dataset loader
//and sequential batcher

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct DataSet{
    uint8_t* data;// all tokens
    size_t n;// number of tokens
    size_t cursor;// current read position
    char path[512];
};

int  dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len, uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);


int dataset_load(const char* path, struct DataSet* ds) {
    if (!path || !ds) return -1;
    memset(ds, 0, sizeof(*ds));

    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    rewind(f);

    ds->n = (size_t)sz;
    ds->data = (uint8_t*)malloc(ds->n ? ds->n : 1);
    if (!ds->data) { fclose(f); return -1; }

    size_t rd = ds->n ? fread(ds->data, 1, ds->n, f) : 0;
    fclose(f);
    if (rd != ds->n) { free(ds->data); memset(ds, 0, sizeof(*ds)); return -1; }

    ds->cursor = 0;
    size_t L = strlen(path); if (L >= sizeof(ds->path)) L = sizeof(ds->path) - 1;
    memcpy(ds->path, path, L); ds->path[L] = '\0';
    return 0;
}

void dataset_free(struct DataSet* ds) {
    if (!ds) return;
    free(ds->data);
    memset(ds, 0, sizeof(*ds));
}

void dataset_reset(struct DataSet* ds, size_t pos) {
    if (!ds) return;
    if (ds->n <= 1) { ds->cursor = 0; return; }
    ds->cursor = pos % (ds->n - 1); //keep one-token lookahead for y
}

//produces
//x[b,t] = data[i], y[b,t] = data[i+1] 
//and wrapping around
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len, uint8_t* x, uint8_t* y) {
    if (!ds || ds->n <= 1) return;

    const size_t B = (size_t)batch_size;
    const size_t T = (size_t)seq_len;
    const size_t usable = ds->n - 1; //we always look one ahead for y

    for (size_t b = 0; b < B; ++b) {
        size_t start = (ds->cursor + b * T) % usable;
        uint8_t* xb = x + b * T;
        uint8_t* yb = y + b * T;
        for (size_t t = 0; t < T; ++t) {
            size_t i = (start + t) % usable;
            xb[t] = ds->data[i];
            //wrap for next token without an extra mod
            size_t j = i + 1;
            yb[t] = ds->data[(j == usable) ? 0 : j];
        }
    }

    ds->cursor = (ds->cursor + B * T) % usable;
}

void dataset_log(const struct DataSet* ds, const char* tag) {
    if (!ds) return;
    fprintf(stderr, "[data] %s: %s | tokens=%zu | cursor=%zu\n",
            tag ? tag : "ds",
            ds->path[0] ? ds->path : "(unnamed)",
            ds->n, ds->cursor);
}

