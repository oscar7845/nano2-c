//data loader (bytes)
//simple sequential batcher
//TODO: 
// 
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct DataSet {
    uint8_t *data;
    size_t n;       // how many bytes
    size_t cursor;  // where next batch starts
    char path[512];
};

//fwd decls
int dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int B, int T, uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);


//load whole file into RAM (just bytes)
int dataset_load(const char* path, struct DataSet* ds){
    memset(ds, 0, sizeof(*ds));

    FILE *f = fopen(path, "rb");
    if(!f){
        printf("dataset_load: can't open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    ds->n = (size_t)sz;
    //alloc at least 1 byte so free() is always safe-ish
    ds->data = (uint8_t*)malloc(ds->n ? ds->n : 1);
    if(!ds->data){
        fclose(f);
        printf("dataset_load: malloc failed\n");
        return -1;
    }

    fread(ds->data, 1, ds->n, f);
    fclose(f);

    ds->cursor = 0;

    //store path
    size_t L = strlen(path);
    if(L >= sizeof(ds->path)) L = sizeof(ds->path) - 1;
    memcpy(ds->path, path, L);
    ds->path[L] = 0;

    return 0;
}


//free memory + clear struct
void dataset_free(struct DataSet* ds){
    if(!ds) return;
    free(ds->data);
    memset(ds, 0, sizeof(*ds));
}


//just jump cursor somewhere (wrap if needed)
//small diff: mod n directly (old one did n-1)
void dataset_reset(struct DataSet* ds, size_t pos){
    if(ds->n == 0){
        ds->cursor = 0;
        return;
    }
    ds->cursor = pos % ds->n;
}


//make batch:
// x[b,t] = data[i]
// y[b,t] = data[i+1] (wrap around)
// basically next-token prediction
void dataset_next_batch(struct DataSet* ds, int B, int T, uint8_t* x, uint8_t* y){
    if(ds->n == 0) return;

    size_t N = ds->n;
    size_t BT = (size_t)B * (size_t)T; //not used but helps mental check

    for(size_t b=0; b<(size_t)B; b++){
        //each row starts at cursor + b*T
        size_t start = (ds->cursor + b*(size_t)T) % N;

        uint8_t *xb = x + b*(size_t)T;
        uint8_t *yb = y + b*(size_t)T;

        for(size_t t=0; t<(size_t)T; t++){
            size_t i = (start + t) % N;
            xb[t] = ds->data[i];
            //next-token with wrap
            yb[t] = ds->data[(i+1) % N];
        }
    }

    //advance cursor for next call
    ds->cursor = (ds->cursor + (size_t)B*(size_t)T) % N;
}


//tiny logger
void dataset_log(const struct DataSet* ds, const char* tag){
    printf("[data] %s: %s | n=%zu | cursor=%zu\n",
           tag?tag:"ds",
           ds->path[0] ? ds->path : "(no name)",
           ds->n,
           ds->cursor);
}

