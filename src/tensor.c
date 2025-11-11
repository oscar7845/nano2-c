#include <stdio.h>
#include <stdlib.h>

//TENSOR WIP
// now has rows, cols (2D)
// TODO: device version later
// maybe add stride stuff?

struct Tensor {
    int rows;
    int cols;
    float *data;
};

struct Tensor* tensor_create(int rows, int cols){
    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->rows = rows;
    t->cols = cols;
    t->data = malloc(sizeof(float) * rows * cols);
    for(int i = 0; i < rows*cols; i++)
        t->data[i] = 0.0f;

    printf("tensor_create: %dx%d\n", rows, cols);
    return t;
}

void tensor_fill(struct Tensor *t, float v){
    int n = t->rows * t->cols;
    for(int i = 0; i < n; i++)
        t->data[i] = v;
    printf("tensor_fill: %f\n", v);
}

void tensor_fill_random(struct Tensor *t){
    int n = t->rows * t->cols;
    for(int i = 0; i < n; i++)
        t->data[i] = (float)(rand() % 100) / 30.0f; // rough
    printf("tensor_fill_random\n");
}

void tensor_show(struct Tensor *t){
    printf("tensor_show %dx%d:\n", t->rows, t->cols);
    int max = (t->rows * t->cols);
    if(max > 16) max = 16; // preview only
    for(int i=0;i<max;i++){
        printf("%f ", t->data[i]);
    }
    printf("\n");
}

// simple matmul (very not optimized)
// C = A * B  ; assume shapes match
// TODO: add errors
void tensor_matmul(struct Tensor *A, struct Tensor *B, struct Tensor *C){
    // A: (rA x cA), B: (rB x cB), C: (rA x cB)
    // assume cA == rB
    for(int i=0;i< C->rows * C->cols; i++) C->data[i] = 0;

    for(int i = 0; i < A->rows; i++){
        for(int j = 0; j < B->cols; j++){
            float sum = 0;
            for(int k = 0; k < A->cols; k++){
                sum += A->data[i*A->cols + k] * B->data[k*B->cols + j];
            }
            C->data[i*C->cols + j] = sum;
        }
    }

    //printf("DEBUG matmul done\n");
}

void tensor_free(struct Tensor *t){
    free(t->data);
    free(t);
    printf("tensor_free\n");
}

