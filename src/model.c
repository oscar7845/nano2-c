#include <stdio.h>
#include <stdlib.h>

//TODO: maybe add more tensors layer wise later
// test with rand vals path
// remove w1 hardcoded shape -> pass config.d_model eventually

struct Tensor{ 
	int n; float *data; 
};
struct Tensor* tensor_create(int n);
void tensor_fill(struct Tensor *t, float v);
void tensor_fill_random(struct Tensor *t);
void tensor_show(struct Tensor *t);
void tensor_free(struct Tensor *t);

struct Model{
    struct Tensor *w1;
    struct Tensor *w2;
};

struct Model* model_new(int dim) 
{
    struct Model *m = malloc(sizeof(struct Model));
    m->w1 = tensor_create(dim);
    m->w2 = tensor_create(dim);

    //TEST: random w1 instead of fixed
    //tensor_fill(m->w1, 1.0f);
    tensor_fill_random(m->w1);

    //still keep w2 fixed to see difference visually
    tensor_fill(m->w2, 2.5f);

    printf("model_new: dim=%d\n", dim);
    return m;
}

void model_forward(struct Model *m){
    printf("model_forward:\n");
    tensor_show(m->w1);
    tensor_show(m->w2);

    // simple next step: pretend forward = elementwise add
    // NOTE: no dimension check (TODO)
    for(int i = 0; i < m->w1->n; i++){
        m->w1->data[i] = m->w1->data[i] + m->w2->data[i];
        //if(i < 5) printf("DEBUG add[%d]: %f\n", i, m->w1->data[i]); // leave commented
    }

    printf("after add:\n");
    tensor_show(m->w1);
}

void model_free(struct Model *m){
    tensor_free(m->w1);
    tensor_free(m->w2);
    free(m);
    printf("model_free\n");
}

