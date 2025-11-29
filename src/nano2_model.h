#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//forward-declare Config
struct Config{
  char train_path[512];
  char val_path[512];
  int seq_len;
  int batch_size;
  int vocab_size;
  int d_model;
  int ffn_mult;
  double lr;
  double weight_decay;
  double clip_grad_norm;
  int seed;
  int top_k;
};

//parameter & buffer shapes must match model.c
struct Params {
  float *E;                 // [V, D]
  float *ln1_g, *ln1_b;     // [D], [D]
  float *Wq, *Wk, *Wv, *Wo; // [D, D]
  float *ln2_g, *ln2_b;     // [D], [D]
  float *W1, *b1;           // [D, F], [F]
  float *W2, *b2;           // [F, D], [D]
};

struct Opt {
  float *m; // [n_params]
  float *v; // [n_params]
};

struct Buffers {
  float *x;       // (B,T,D)
  float *x_ln1;   // (B,T,D)
  float *q,*k,*v; // (B,T,D)
  float *scores;  // (B,T,T)
  float *probs;   // (B,T,T)
  float *attn_out;// (B,T,D)
  float *x_res1;  // (B,T,D)
  float *x_ln2;   // (B,T,D)
  float *ff1;     // (B,T,F)
  float *ff2;     // (B,T,D)
  float *x_res2;  // (B,T,D)
  float *logits;  // (B,T,V)
  float *rowmax;  // (B,T)
  //saved stats/activations for backwards
  float *ln1_mean, *ln1_invstd; // [B*T]
  float *ln2_mean, *ln2_invstd; // [B*T]
  float *ff1_pre;               // [B*T, F] (pre-GELU)
};

struct Model {
  int B, T, D, V, F;
  struct Params p;
  struct Opt opt;
  float *flat_params;
  float *flat_grads;
  size_t n_params;
  float *pos_sin; // [T, D/2]
  float *pos_cos; // [T, D/2]
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

//one training step (forward + backward + Allreduce + AdamW)
float nano2_train_step(struct Model* M,
                       const uint8_t* h_tokens_x,
                       const uint8_t* h_tokens_y,
                       const struct Config* cfg,
                       int world_size, int rank);

#ifdef __cplusplus
} // extern "C"
#endif

