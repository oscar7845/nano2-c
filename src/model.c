#include <stdio.h>
#include <stdlib.h>

//tensor
//stuff (very WIP)
struct Tensor{
    int rows;
    int cols;
    float *data;
};

struct Tensor* tensor_create(int r, int c);
void tensor_fill(struct Tensor *t, float v);
void tensor_fill_random(struct Tensor *t);
void tensor_show(struct Tensor *t);
void tensor_matmul(struct Tensor *A, struct Tensor *B, struct Tensor *C);
void tensor_free(struct Tensor *t);

//small ReLU
//TODO: GPU version later
static void tensor_relu_inplace(struct Tensor *t)
{
    int n = t->rows * t->cols;
    for(int i = 0; i < n; i++){
        float x = t->data[i];
        t->data[i] = (x > 0) ? x : 0;
    }
    //printf("relu applied\n");
}

//optional GPU fwd (not used yet)
// just declared here so linker doesn't yell
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

//MODEL struct
// basically 2-layer MLP block
struct Model{
    struct Tensor *W1; // (d_model x hidden)
    struct Tensor *b1; // (1 x hidden)
    struct Tensor *W2; // (hidden x d_model)
    struct Tensor *b2; // (1 x d_model)

    int d_model;
    int hidden;
};


//=============================
// model_new
//=============================
struct Model* model_new(int d_model)
{
    int hidden = d_model * 4;  // standard MLP up-projection

    struct Model *m = malloc(sizeof(struct Model));
    m->d_model = d_model;
    m->hidden  = hidden;

    m->W1 = tensor_create(d_model, hidden);
    m->b1 = tensor_create(1,        hidden);

    m->W2 = tensor_create(hidden,   d_model);
    m->b2 = tensor_create(1,        d_model);

    tensor_fill_random(m->W1);
    tensor_fill(m->b1, 0.01f);

    tensor_fill_random(m->W2);
    tensor_fill(m->b2, 0.01f);

    printf("model_new: d_model=%d hidden=%d\n", d_model, hidden);
    return m;
}


//===================================================
// v3: adds activation flag, tiny timing stub,
//      matmul debug hook.
// NOT optimized, still "student code"
//===================================================
static int MODEL_DEBUG = 1;
static int TENSOR_ACT  = 0;   //0=ReLU, 1=sigmoid maybe later

//tiny timing stubs (no real timing yet)
static void timer_start(){ /* TODO maybe rdtsc? */ }
static void timer_end(const char *tag){
    if(MODEL_DEBUG) printf("timer_end: %s\n", tag);
}


//=============================
// model_forward
//=============================
void model_forward(struct Model *m,
                   struct Tensor *x_in,
                   struct Tensor *x_tmp1,
                   struct Tensor *x_out)
{
    //---- layer1 matmul ----
    timer_start();
    tensor_matmul(x_in, m->W1, x_tmp1);
    timer_end("mm1");

    //add bias1
    for(int i = 0; i < x_tmp1->rows; i++){
        for(int j = 0; j < x_tmp1->cols; j++){
            x_tmp1->data[i*x_tmp1->cols + j] += m->b1->data[j];
        }
    }

    //activation
    if(TENSOR_ACT == 0){
        tensor_relu_inplace(x_tmp1);
    } else {
        //TODO sigmoid later
        //printf("sigmoid todo\n");
    }

    //---- layer2 matmul ----
    timer_start();
    tensor_matmul(x_tmp1, m->W2, x_out);
    timer_end("mm2");

    //add bias2
    for(int i = 0; i < x_out->rows; i++){
        for(int j = 0; j < x_out->cols; j++){
            x_out->data[i*x_out->cols + j] += m->b2->data[j];
        }
    }

    // GPU try path (disabled for now)
    /*
    train_forward_gpu(x_in->data,
                      m->W1->data,
                      m->b1->data,
                      m->W2->data,
                      m->b2->data,
                      x_out->data,
                      x_in->rows,
                      m->d_model);
    return;
    */

    //if(MODEL_DEBUG) printf("DEBUG forward done\n");
}


//=============================
// free
//=============================
void model_free(struct Model *m)
{
    tensor_free(m->W1);
    tensor_free(m->b1);
    tensor_free(m->W2);
    tensor_free(m->b2);
    free(m);
    printf("model_free\n");
}


