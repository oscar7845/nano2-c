#include <stdio.h>
#include <stdlib.h>

//TEST tensor struct 
// still need comps
// add debug
// TODO:
//
struct Tensor{
    int n;
    float *data;
};

struct Tensor* tensor_create(int n){
    struct Tensor *t =malloc(sizeof(struct Tensor));
    t->n =n;
    t->data= malloc(sizeof(float) * n);
    for(int i =0; i<n; i++) t->data[i] = 0;

    printf("tensor_create: n=%d\n", n);
    return t;
}

void tensor_fill(struct Tensor *t, float v){
    for(int i = 0; i < t->n; i++)
        t->data[i] =v;

    printf("tensor_fill: v=%f\n", v);
}

void tensor_show(struct Tensor *t){
    printf("tensor_show n=%d: ", t->n);
    for(int i= 0; i<t->n && i < 10; i++)
        printf("%f ", t->data[i]);
    printf("\n");
}

void tensor_free(struct Tensor *t)
{
    free(t->data);
    free(t);
    printf("tensor_free\n");
}

