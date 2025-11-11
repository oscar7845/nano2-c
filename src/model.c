#include <stdio.h>
#include <stdlib.h>

//TODO: fill tensor
// test with rand vals
// remove w1 hardcoded

struct Tensor{ 
	int n; float *data; 
};
struct Tensor* tensor_create(int n);
void tensor_fill(struct Tensor *t, float v);
void tensor_show(struct Tensor *t);
void tensor_free(struct Tensor *t);

struct Model{
    struct Tensor *w1;
    struct Tensor *w2;
};

struct Model* model_new(int dim) 
{
    struct Model *m=malloc(sizeof(struct Model));
    m->w1= tensor_create(dim);
    m->w2= tensor_create(dim);

    //dummy vals; need to change
    tensor_fill(m->w1,1.0f);
    tensor_fill(m->w2, 2.5f);

    printf("model_new: dim=%d\n", dim);
    return m;
}

void model_forward(struct Model *m){
    printf("model_forward:\n");
    tensor_show(m->w1);
    tensor_show(m->w2);
}



void model_free(struct Model *m){
    tensor_free(m->w1);
    tensor_free(m->w2);
    free(m);
    printf("model_free\n");
}

