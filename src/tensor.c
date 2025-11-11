#include <stdio.h>
#include <stdlib.h>
#include <time.h>   // for rand

//TEST tensor struct 
// still need comps
// add debug
// TODO: add GPU version later
//
struct Tensor{
    int n;
    float *data;
};

struct Tensor* tensor_create(int n){
    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->n = n;
    t->data = malloc(sizeof(float) * n);
    for(int i = 0; i < n; i++) t->data[i] = 0;

    //printf("DEBUG create raw ptr=%p\n", t->data); // maybe later
    printf("tensor_create: n=%d\n", n);
    return t;
}

void tensor_fill(struct Tensor *t, float v){
    for(int i = 0; i < t->n; i++)
        t->data[i] = v;

    printf("tensor_fill: v=%f\n", v);
}

// new: quick random fill test
void tensor_fill_random(struct Tensor *t){
    // srand only once maybe → TODO: move to init
    for(int i = 0; i < t->n; i++)
        t->data[i] = (float)(rand() % 100) / 25.0f;  // rough random scaling

    printf("tensor_fill_random done\n");
    //printf("DEBUG rand sample: %f\n", t->data[0]);
}

void tensor_show(struct Tensor *t){
    printf("tensor_show n=%d: ", t->n);
    for(int i = 0; i < t->n && i < 10; i++)
        printf("%f ", t->data[i]);
    printf("\n");
}

void tensor_free(struct Tensor *t)
{
    free(t->data);
    free(t);
    printf("tensor_free\n");
}

