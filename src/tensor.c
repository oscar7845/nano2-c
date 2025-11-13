#include <stdio.h>
#include <stdlib.h>

//TENSOR WIP v3
// now matmul + debug flag
// still basic cpu only
// TODO: device, views, etc

static int TENSOR_DEBUG = 1;

struct Tensor{
    int rows;
    int cols;
    float *data;
};

struct Tensor* tensor_create(int rows,int cols){
    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->rows=rows; t->cols=cols;
    t->data=malloc(sizeof(float)*rows*cols);
    for(int i=0;i<rows*cols;i++) t->data[i]=0;
    if(TENSOR_DEBUG) printf("tensor_create %dx%d\n",rows,cols);
    return t;
}

void tensor_fill(struct Tensor *t,float v){
    for(int i=0;i<t->rows*t->cols;i++) t->data[i]=v;
    if(TENSOR_DEBUG) printf("tensor_fill %f\n",v);
}

void tensor_fill_rand(struct Tensor *t){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i]=(float)(rand()%100)/60.0f;
    if(TENSOR_DEBUG) printf("tensor_fill_rand done\n");
}

// slow naive matmul
void tensor_matmul(struct Tensor *A,struct Tensor *B,struct Tensor *C){
    if(A->cols!=B->rows){ printf("matmul bad dims\n"); return; }
    for(int i=0;i<A->rows;i++){
        for(int j=0;j<B->cols;j++){
            float s=0;
            for(int k=0;k<A->cols;k++){
                s += A->data[i*A->cols+k]*B->data[k*B->cols+j];
            }
            C->data[i*C->cols+j]=s;
        }
    }
    //printf("matmul done\n");
}

void tensor_show(struct Tensor *t){
    printf("tensor_show %dx%d:\n",t->rows,t->cols);
    int n=t->rows*t->cols; if(n>16)n=16;
    for(int i=0;i<n;i++) printf("%.2f ",t->data[i]);
    printf("\n");
}

void tensor_free(struct Tensor *t){
    free(t->data); free(t);
    if(TENSOR_DEBUG) printf("tensor_free\n");
}

