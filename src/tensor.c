#include <stdio.h>
#include <stdlib.h>

//TENSOR WIP v1
// add scalar ops + check dims
// later maybe transpose
// device copy todo

struct Tensor {
    int rows;
    int cols;
    float *data;
};

struct Tensor* tensor_create(int rows, int cols){
    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->rows = rows; t->cols = cols;
    t->data = malloc(sizeof(float) * rows * cols);
    for(int i=0;i<rows*cols;i++) t->data[i] = 0;
    printf("tensor_create %dx%d\n", rows, cols);
    return t;
}

void tensor_fill(struct Tensor *t, float v){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i]=v;
    //printf("tensor_fill: %f\n",v);
}

void tensor_fill_random(struct Tensor *t){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i]=(float)(rand()%100)/50.0f;
    printf("tensor_fill_random\n");
}

void tensor_add(struct Tensor *a, struct Tensor *b, struct Tensor *out){
    if(a->rows!=b->rows || a->cols!=b->cols){
        printf("tensor_add shape mismatch\n");
        return;
    }
    int n=a->rows*a->cols;
    for(int i=0;i<n;i++) out->data[i]=a->data[i]+b->data[i];
    //printf("tensor_add done\n");
}

void tensor_scale(struct Tensor *a,float s){
    int n=a->rows*a->cols;
    for(int i=0;i<n;i++) a->data[i]*=s;
}

void tensor_show(struct Tensor *t){
    printf("tensor_show %dx%d:\n", t->rows,t->cols);
    int n=t->rows*t->cols; if(n>16) n=16;
    for(int i=0;i<n;i++) printf("%f ",t->data[i]);
    printf("\n");
}

void tensor_free(struct Tensor *t){
    free(t->data); free(t);
    printf("tensor_free\n");
}

