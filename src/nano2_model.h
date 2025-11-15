#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Config; // don't want full include here

struct Params{
  float *E;
  float *ln1_g,*ln1_b;
  float *Wq,*Wk,*Wv,*Wo;
  float *ln2_g,*ln2_b;
  float *W1,*b1;
  float *W2,*b2;
};

struct Opt{
  float *m;
  float *v;
};

struct Buffers{
  float *x;
  float *x_ln1;
  float *q,*k,*v;
  float *scores;
  float *probs;
  float *attn_out;
  float *x_res1;
  float *x_ln2;
  float *ff1;
  float *ff2;
  float *x_res2;
  float *logits;
  float *rowmax;
};

struct Model{
  int B,T,D,V,F;
  struct Params p;
  struct Opt opt;
  float *flat_params;
  float *flat_grads;
  size_t n_params;
  float *pos_sin;
  float *pos_cos;
  struct Buffers buf;
};

void  model_init (struct Model *M, const struct Config *c);
void  model_free (struct Model *M);
void  model_log_summary(const struct Model *M, const struct Config *c);

// small fwd-loss helper (tokens on host)
float nano2_forward_loss(struct Model *M,
                         const uint8_t *h_x,
                         const uint8_t *h_y);

#ifdef __cplusplus
}
#endif
