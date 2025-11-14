//dataset loader
//sequential batcher
//still byte-level
//TODO:keep DataSet ptrs
//ram warns? why
//ftell remove later
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct DataSet {
    uint8_t *data;
    size_t n;//total tokens
    size_t cursor;  //current pos
    char path[512]; //saved so we can print it
};

int  dataset_load(const char *path, struct DataSet *ds);
void dataset_free(struct DataSet *ds);
void dataset_reset(struct DataSet *ds, size_t pos);
void dataset_next_batch(struct DataSet *ds, int B, int T, uint8_t *x, uint8_t *y);
void dataset_log(const struct DataSet *ds, const char *tag);


//load whole file into RAM (byte tokens)
//blocking
int dataset_load(const char *path, struct DataSet *ds){
    memset(ds, 0, sizeof(*ds));

    FILE *f = fopen(path, "rb");
    if(!f){
        fprintf(stderr, "dataset_load: failed to open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    if(sz <= 0){
        fprintf(stderr, "dataset_load: empty file? %s\n", path);
        fclose(f);
        return -1;
    }

    ds->n = (size_t)sz;
    ds->data = (uint8_t*)malloc(ds->n);
    if(!ds->data){
        fclose(f);
        return -1;
    }

    fread(ds->data, 1, ds->n, f);
    fclose(f);

    //init cursor
    ds->cursor = 0;

    //save path
    size_t L = strlen(path);
    if(L >= sizeof(ds->path)) L = sizeof(ds->path) - 1;
    memcpy(ds->path, path, L);
    ds->path[L] = 0;

    return 0;
}
void dataset_free(struct DataSet *ds){
    free(ds->data);
    memset(ds, 0, sizeof(*ds));
}



//move cursor (wrap around)
//small mod logic
void dataset_reset(struct DataSet *ds, size_t pos){
    if(ds->n == 0){
        ds->cursor = 0;
        return;
    }
    ds->cursor = pos % ds->n;
}


//next batch:
//x[b,t] = token i
//y[b,t] = token (i+1)
//wraps around file
void dataset_next_batch(struct DataSet *ds, int B, int T, uint8_t *x, uint8_t *y){
    if(ds->n == 0) return;

    const size_t N = ds->n;
    const size_t BS = (size_t)B;
    const size_t TS = (size_t)T;

    for(size_t b=0; b<BS; b++){
        size_t start = (ds->cursor + b*TS) % N;
        uint8_t *xb = x + b*TS;
        uint8_t *yb = y + b*TS;

        for(size_t t=0; t<TS; t++){
            size_t i = (start + t) % N;
            xb[t] = ds->data[i];
            // next token (predict next byte)
            yb[t] = ds->data[(i+1) % N];
        }
    }

    //advance global cursor (wrap)
    ds->cursor = (ds->cursor + BS*TS) % N;
}


//tiny print helper
void dataset_log(const struct DataSet *ds, const char *tag){
    printf("[data] %s: %s | tokens=%zu | cursor=%zu\n",
        tag ? tag : "ds",
        ds->path[0] ? ds->path : "(unnamed)",
        ds->n,
        ds->cursor);
}

