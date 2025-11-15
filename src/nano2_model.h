#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//forward-declare Config so we don't duplicate its definition here
struct Config;

//parameter & buffer shapes (model.c)
struct Params{
  float *E; // [V, D]
  float *ln1_g, *ln1_b; // [D], [D]
  float *Wq, *Wk, *Wv, *Wo; // [D, D]
  float *ln2_g, *ln2_b;// [D], [D]
  float *W1, *b1; // [D, F], [F]
  float *W2, *b2;// [F, D], [D]
};

struct Opt{
  float *m; //[n_params]
  float *v; //[n_params]
};

struct Buffers{
  float *x; //(B,T,D)
  float *x_ln1; //(B,T,D)
  float *q,*k,*v; //(B,T,D)
  float *scores; //(B,T,T)
  float *probs; //(B,T,T)
  float *attn_out;//(B,T,D)
  float *x_res1; //(B,T,D)
  float *x_ln2; //(B,T,D)
  float *ff1; //(B,T,F)
  float *ff2; //(B,T,D)
  float *x_res2; //(B,T,D)
  float *logits; //(B,T,V)
  float *rowmax; //(B,T)
};

struct Model{
  int B,T,D,V,F;
  struct Params p;
  struct Opt opt;
  float *flat_params;
  float *flat_grads;
  size_t n_params;
  float *pos_sin; //[T, D/2]
  float *pos_cos; //[T, D/2]
  struct Buffers buf;
};

//API
void  model_init(struct Model* M, const struct Config* c);
void  model_free(struct Model* M);
void  model_log_summary(const struct Model* M, const struct Config* c);

//forward-only loss
float nano2_forward_loss(struct Model* M,
                         const uint8_t* h_tokens_x, // [B*T] host
                         const uint8_t* h_tokens_y);// [B*T] host

#ifdef __cplusplus
} // extern "C"
#endif

