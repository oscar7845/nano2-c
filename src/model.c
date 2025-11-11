#include <stdio.h>
#include <stdlib.h>

//TODO: move to header later
struct Tensor { int rows; int cols; float *data; };
struct Tensor* tensor_create(int r, int c);
void tensor_fill(struct Tensor *t, float v);
void tensor_fill_random(struct Tensor *t);
void tensor_show(struct Tensor *t);
void tensor_matmul(struct Tensor *A, struct Tensor *B, struct Tensor *C);
void tensor_free(struct Tensor *t);

struct Model{
    // pretend one linear layer: y = xW + b
    struct Tensor *W;   // (d_model x d_model) or (in_dim x out_dim)
    struct Tensor *b;   // (1 x d_model)
};

struct Model* model_new(int d_model){
    struct Model *m = malloc(sizeof(struct Model));

    m->W = tensor_create(d_model, d_model);
    m->b = tensor_create(1, d_model);

    // random W, small b
    tensor_fill_random(m->W);
    tensor_fill(m->b, 0.01f);

    printf("model_new d_model=%d\n", d_model);
    return m;
}

// x_in: (batch x d_model)
// x_out: (batch x d_model)
void model_forward(struct Model *m, struct Tensor *x_in, struct Tensor *x_out){
    // x_out = x_in * W
    tensor_matmul(x_in, m->W, x_out);

    // add bias
    for(int b = 0; b < x_out->rows; b++){
        for(int j = 0; j < x_out->cols; j++){
            x_out->data[b*x_out->cols + j] += m->b->data[j];
        }
    }

    printf("model_forward done\n");
    //tensor_show(x_out); // enable if needed
}

void model_free(struct Model *m){
    tensor_free(m->W);
    tensor_free(m->b);
    free(m);
    printf("model_free\n");
}

