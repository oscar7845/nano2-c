#include <stdio.h>
#include <stdlib.h>

//tensor
//stuff
struct Tensor{ int rows; int cols; float *data; };
struct Tensor* tensor_create(int r, int c);
void tensor_fill(struct Tensor *t, float v);
void tensor_fill_random(struct Tensor *t);
void tensor_show(struct Tensor *t);
void tensor_matmul(struct Tensor *A, struct Tensor *B, struct Tensor *C);
void tensor_free(struct Tensor *t);

//small ReLU
//TODO: GPU version later
static void tensor_relu_inplace(struct Tensor *t){
    int n = t->rows * t->cols;
    for(int i = 0; i < n; i++){
        float x = t->data[i];
        t->data[i] = (x > 0) ? x : 0;
    }
    //printf("relu applied\n"); // leave commented
}

struct Model{
    struct Tensor *W1;// (d_model x hidden)
    struct Tensor *b1; // (1 x hidden)
    struct Tensor *W2; // (hidden x d_model)
    struct Tensor *b2; // (1 x d_model)

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
    //hidden dimension maybe = 4 * d_model (standard MLP)
    int hidden = d_model * 4;

    struct Model *m = malloc(sizeof(struct Model));
    m->d_model = d_model;
    m->hidden = hidden;

    m->W1 = tensor_create(d_model, hidden);
    m->b1 = tensor_create(1, hidden);

    m->W2 = tensor_create(hidden, d_model);
    m->b2 = tensor_create(1, d_model);

    tensor_fill_random(m->W1);
    tensor_fill(m->b1, 0.01f);

    tensor_fill_random(m->W2);
    tensor_fill(m->b2, 0.01f);

    printf("model_new: d_model=%d hidden=%d\n", d_model, hidden);
    return m;
}

//fwd pass:
//x_in  : (batch x d_model)
//x_tmp1: (batch x hidden)  allocated outside or scratch?
//x_out : (batch x d_model)
void model_forward(struct Model *m, struct Tensor *x_in,
                   struct Tensor *x_tmp1,
                   struct Tensor *x_out)
{
    //x_tmp1 = x_in * W1 + b1
    tensor_matmul(x_in, m->W1, x_tmp1);

    for(int b = 0; b < x_tmp1->rows; b++){
        for(int j = 0; j < x_tmp1->cols; j++){
            x_tmp1->data[b*x_tmp1->cols + j] += m->b1->data[j];
        }
    }

    tensor_relu_inplace(x_tmp1);

    //x_out = x_tmp1 * W2 + b2
    tensor_matmul(x_tmp1, m->W2, x_out);

    for(int b = 0; b < x_out->rows; b++){
        for(int j = 0; j < x_out->cols; j++){
            x_out->data[b*x_out->cols + j] += m->b2->data[j];
        }
    }

   //GPU try path (temporary)
   //(just batch=1 currently)
   //leave commented until ready to test
   /*
   train_forward_gpu(x_in->data,
                  m->W1->data,
                  m->b1->data,
                  m->W2->data,
                  m->b2->data,
                  x_out->data,
                  x_in->rows,
                  m->d_model);

    // early return to skip CPU
    return;
    */

    //printf("DEBUG forward done\n");
}



void model_free(struct Model *m){
    tensor_free(m->W1);
    tensor_free(m->b1);
    tensor_free(m->W2);
    tensor_free(m->b2);
    free(m);
    printf("model_free\n");
}

