//add simple host/device alloc wrappers
//no RNG yet
//TODO:
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>

//host/device memory helpers (partial)
void* nano2_malloc_host(size_t bytes){
    if(bytes==0) return NULL;
    void *p = malloc(bytes);
    //printf("malloc_host %zu\n",bytes);
    return p;
}

void nano2_free_host(void *p){
    free(p);
}

void* nano2_malloc_device(size_t bytes){
    void *p=NULL;
    if(bytes) cudaMalloc(&p,bytes);
    return p;
}

void nano2_free_device(void *p){
    if(p) cudaFree(p);
}

void nano2_copy_host_to_device(void *dst, const void *src, size_t bytes){
    if(bytes) cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice);
}

void nano2_copy_device_to_host(void *dst, const void *src, size_t bytes){
    if(bytes) cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost);
}


//basic tensor struct (same as old)
struct Tensor{
    int rows;
    int cols;
    float *data;
};

struct Tensor* tensor_create(int r,int c){
    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->rows=r; t->cols=c;
    t->data = nano2_malloc_host(sizeof(float)*r*c);
    for(int i=0;i<r*c;i++) t->data[i]=0;
    printf("tensor_create %dx%d\n",r,c);
    return t;
}

void tensor_fill(struct Tensor *t,float v){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i]=v;
}

void tensor_fill_random(struct Tensor *t){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i] = (float)(rand()%100)/40.0f;
}

void tensor_show(struct Tensor *t){
    int n=t->rows*t->cols;
    if(n>16) n=16;
    printf("tensor_show %dx%d:\n",t->rows,t->cols);
    for(int i=0;i<n;i++) printf("%f ",t->data[i]);
    printf("\n");
}

void tensor_free(struct Tensor *t){
    nano2_free_host(t->data);
    free(t);
}
