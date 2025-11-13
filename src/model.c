#include <stdio.h>
#include <stdlib.h>

//
//tensor stuff
// TODO: fill random,check w1 b1, w2 b2
struct Tensor{ 
    int rows; int cols; float *data; 
};

struct Tensor* tensor_create(int r, int c){
    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->rows = r; t->cols = c;
    t->data = malloc(sizeof(float) * r * c);
    for(int i=0;i<r*c;i++) t->data[i] = 0;
    printf("tensor_create %dx%d\n", r, c);
    return t;
}

void tensor_fill(struct Tensor *t, float v){
    int n = t->rows * t->cols;
    for(int i=0;i<n;i++) t->data[i]=v;
    //printf("tensor_fill %f\n",v);
}

void tensor_fill_random(struct Tensor *t){
    int n = t->rows * t->cols;
    for(int i=0;i<n;i++){
        t->data[i] = (float)(rand()%100) / 40.0f;
    }
    printf("tensor_fill_random\n");
}

void tensor_show(struct Tensor *t){
    printf("tensor_show %dx%d:\n", t->rows, t->cols);
    int n = t->rows*t->cols;
    if(n > 20) n = 20;
    for(int i=0;i<n;i++) printf("%f ", t->data[i]);
    printf("\n");
}

void tensor_matmul(struct Tensor *A, struct Tensor *B, struct Tensor *C){
    //basic shapes
    if(A->cols != B->rows){
        printf("matmul mismatch %dx%d * %dx%d\n",
               A->rows,A->cols,B->rows,B->cols);
        return;
    }
    if(C->rows != A->rows || C->cols != B->cols){
        printf("matmul out shape bad\n");
        return;
    }

    for(int i=0;i<C->rows*C->cols;i++) C->data[i]=0;

    for(int i=0;i<A->rows;i++){
        for(int j=0;j<B->cols;j++){
            float s=0;
            for(int k=0;k<A->cols;k++){
                s += A->data[i*A->cols+k] *
                     B->data[k*B->cols+j];
            }
            C->data[i*C->cols+j] = s;
        }
    }
    //printf("matmul done\n");
}

void tensor_free(struct Tensor *t){
    free(t->data); free(t);
    printf("tensor_free\n");
}

//small ReLU
static void tensor_relu_inplace(struct Tensor *t){
    int n = t->rows * t->cols;
    for(int i=0;i<n;i++){
        float x = t->data[i];
        t->data[i] = (x > 0) ? x : 0;
    }
}

struct Model{
    struct Tensor *W1;
    struct Tensor *b1;
    struct Tensor *W2;
    struct Tensor *b2;
    int d_model;
    int hidden;
};

extern void train_forward_gpu(
    const float *h_x,
    const float *h_W1,
    const float *h_b1,
    const float *h_W2,
    const float *h_b2,
    float *h_out,
    int batch,
    int d_model
);

struct Model* model_new(int d_model){
    int hidden = d_model * 4;
    struct Model *m = malloc(sizeof(struct Model));
    m->d_model=d_model; m->hidden=hidden;

    m->W1 = tensor_create(d_model, hidden);
    m->b1 = tensor_create(1, hidden);
    m->W2 = tensor_create(hidden, d_model);
    m->b2 = tensor_create(1, d_model);

    tensor_fill_random(m->W1);
    tensor_fill(m->b1, 0.01f);
    tensor_fill_random(m->W2);
    tensor_fill(m->b2, 0.01f);

    printf("model_new d_model=%d hidden=%d\n", d_model, hidden);
    return m;
}

void model_forward(struct Model *m,
                   struct Tensor *x_in,
                   struct Tensor *x_tmp1,
                   struct Tensor *x_out)
{
    tensor_matmul(x_in, m->W1, x_tmp1);

    for(int b=0;b<x_tmp1->rows;b++){
        for(int j=0;j<x_tmp1->cols;j++){
            x_tmp1->data[b*x_tmp1->cols+j] += m->b1->data[j];
        }
    }

    tensor_relu_inplace(x_tmp1);

    tensor_matmul(x_tmp1, m->W2, x_out);

    for(int b=0;b<x_out->rows;b++){
        for(int j=0;j<x_out->cols;j++){
            x_out->data[b*x_out->cols+j] += m->b2->data[j];
        }
    }
}

void model_free(struct Model *m){
    tensor_free(m->W1);
    tensor_free(m->b1);
    tensor_free(m->W2);
    tensor_free(m->b2);
    free(m);
    printf("model_free\n");
}

