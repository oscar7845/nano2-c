//byte/tok dataset loader
//and sequential batcher
//TODO: remove debugs
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

//dataset holder:
//data= raw bytes/toks
//n= total number of tokens
//cursor= where next batch will start reading
//path= name of file (for logs)
struct DataSet{
    uint8_t* data;
    size_t n;
    size_t cursor;
    char path[512];
};

int dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len, uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);

//load entire file into mem
int dataset_load(const char* path, struct DataSet* ds){
    memset(ds, 0, sizeof(*ds));

    FILE* f=fopen(path, "rb");
    fseek(f,0,SEEK_END); 
    long sz=ftell(f);//how many bytes total?
    rewind(f);

    ds->n=(size_t)sz;
    ds->data=(uint8_t*)malloc(ds->n ? ds->n : 1);
    fread(ds->data,1,ds->n,f);
    fclose(f);

    ds->cursor=0;//start reading at 0
    //store path string
    size_t L=strlen(path); if (L >= sizeof(ds->path)) L = sizeof(ds->path) - 1;
    memcpy(ds->path, path, L); ds->path[L] = '\0';
    return 0;
}

void dataset_free(struct DataSet* ds){
    free(ds->data);
    memset(ds, 0, sizeof(*ds));
}
//set cursor
void dataset_reset(struct DataSet* ds, size_t pos){
    ds->cursor=(ds->n > 0) ? (pos % ds->n) : 0;
}

//make next training batch:
//x[b,t]= data[i]
//y[b,t]= data[i+1] (next tok)
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len, uint8_t* x, uint8_t* y){
    if (ds->n == 0) return;
    const size_t B=(size_t)batch_size, T=(size_t)seq_len, N=ds->n;
    //for each item in batch
    for (size_t b=0; b<B; ++b) {
        size_t start=(ds->cursor + b * T) % N;
        uint8_t* xb= x+b*T;
        uint8_t* yb= y+b*T;
        for (size_t t=0; t<T; ++t) {
            size_t i=(start + t) % N;
            xb[t]= ds->data[i];
            yb[t]= ds->data[(i + 1) % N];
        }
    }
    ds->cursor= (ds->cursor + B * T) % N;
}



//logs
void dataset_log(const struct DataSet* ds, const char* tag){
    printf("[data] %s: %s | tokens=%zu | cursor=%zu\n",
           tag ? tag : "ds",
           ds->path[0] ? ds->path : "(unnamed)",
           ds->n, ds->cursor);
}

