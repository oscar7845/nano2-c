#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

//i reuse same struct layout as config.c / data.c.

struct Config {
    char train_path[512];
    char val_path[512];
    int  seq_len;
    int  batch_size;
    int  vocab_size;
    int  d_model;
    int  ffn_mult;
    double lr;
    double weight_decay;
    double clip_grad_norm;
    int  seed;
    int  top_k;
};

int  config_from_file(const char* path, struct Config* out);
void config_log(const struct Config* c);

struct DataSet {
    uint8_t* data;
    size_t   n;
    size_t   cursor;
    char     path[512];
};
int  dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len,
                        uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);

//helpers
static void get_config_path(int argc, char** argv, char* out, size_t cap){
    const char* def = "./configs/nano2.json";
    size_t n = strlen(def);
    if (n >= cap) n = cap - 1;
    memcpy(out, def, n);
    out[n] = '\0';

    for (int i = 1; i < argc; ++i){
        const char* a = argv[i];
        if (strncmp(a, "--config=", 9) == 0){
            strncpy(out, a + 9, cap - 1);
            out[cap-1] = '\0';
        } else if (strcmp(a, "--config") == 0 && i + 1 < argc){
            strncpy(out, argv[i+1], cap - 1);
            out[cap-1] = '\0';
            ++i;
        }
    }
}

// number of forward iterations: --fw-iters=N (default 1)
static int get_fw_iters(int argc, char** argv){
    int iters = 1;
    for (int i = 1; i < argc; ++i){
        if (strncmp(argv[i], "--fw-iters=", 11) == 0){
            iters = atoi(argv[i] + 11);
        } else if (strcmp(argv[i], "--fw-iters") == 0 && i + 1 < argc){
            iters = atoi(argv[i+1]);
            ++i;
        }
    }
    if (iters < 1) iters = 1;
    return iters;
}

//cpu model
struct CPUParams {
    float* E;                 // [V, D]
    float* ln1_g; float* ln1_b; // [D]
    float* Wq; float* Wk; float* Wv; float* Wo; // [D, D]
    float* ln2_g; float* ln2_b;                 // [D]
    float* W1; float* b1;    // [D, F], [F]
    float* W2; float* b2;    // [F, D], [D]
};

struct CPUBuffer {
    float* x;        // (B,T,D)
    float* x_ln1;    // (B,T,D)
    float* q;        // (B,T,D)
    float* k;        // (B,T,D)
    float* v;        // (B,T,D)
    float* scores;   // (B,T,T)
    float* probs;    // (B,T,T)
    float* attn_out; // (B,T,D)
    float* x_res1;   // (B,T,D)
    float* x_ln2;    // (B,T,D)
    float* ff1;      // (B,T,F)
    float* ff1_pre;  // (B,T,F)
    float* ff2;      // (B,T,D)
    float* x_res2;   // (B,T,D)
    float* logits;   // (B,T,V)
};

struct CPUModel {
    int B, T, D, V, F;
    struct CPUParams p;
    struct CPUBuffer buf;
    float* pos_sin;  // [T, D/2]
    float* pos_cos;  // [T, D/2]
    size_t n_params;
};

//rng and inits
struct CPU_RNG {
    uint64_t state;
};

static void rng_seed(struct CPU_RNG* r, uint32_t seed){
    if (!seed) seed = 1u;
    r->state = seed;
}

static float rng_uniform(struct CPU_RNG* r){
    // xorshift64*
    uint64_t x = r->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->state = x;
    uint32_t hi = (uint32_t)(x >> 32);
    return hi / 4294967296.0f; // [0,1)
}

static float rng_normal(struct CPU_RNG* r){
    // Box-Muller
    float u1 = rng_uniform(r);
    float u2 = rng_uniform(r);
    if (u1 < 1e-7f) u1 = 1e-7f;
    float mag = sqrtf(-2.0f * logf(u1));
    float z0  = mag * cosf(2.0f * 3.1415926535f * u2);
    return z0;
}

static void fill_gaussian(float* dst, size_t n, float std, struct CPU_RNG* rng){
    for (size_t i = 0; i < n; ++i){
        dst[i] = std * rng_normal(rng);
    }
}

static void fill_zeros(float* dst, size_t n){
    for (size_t i = 0; i < n; ++i) dst[i] = 0.0f;
}

static void fill_ones(float* dst, size_t n){
    for (size_t i = 0; i < n; ++i) dst[i] = 1.0f;
}

static void make_sincos_tables(int T, int D, float* sin_out, float* cos_out){
    const int H = D / 2;
    for (int t = 0; t < T; ++t){
        for (int i = 0; i < H; ++i){
            float inv_freq = powf(10000.0f, -2.0f * (float)i / (float)D);
            float ang = (float)t * inv_freq;
            sin_out[(size_t)t * H + i] = sinf(ang);
            cos_out[(size_t)t * H + i] = cosf(ang);
        }
    }
}

static size_t count_params(const struct Config* c){
    int D = c->d_model;
    int V = c->vocab_size;
    int F = c->ffn_mult * D;
    size_t n = 0;
    n += (size_t)V * D;    // E
    n += 2 * (size_t)D;    // ln1_g, ln1_b
    n += 4 * (size_t)D * D;// Wq,Wk,Wv,Wo
    n += 2 * (size_t)D;    // ln2_g, ln2_b
    n += (size_t)D * F + F;  // W1,b1
    n += (size_t)F * D + D;  // W2,b2
    return n;
}

static void cpu_model_init(struct CPUModel* M, const struct Config* cfg){
    memset(M, 0, sizeof(*M));
    M->B = cfg->batch_size;
    M->T = cfg->seq_len;
    M->D = cfg->d_model;
    M->V = cfg->vocab_size;
    M->F = cfg->ffn_mult * cfg->d_model;
    M->n_params = count_params(cfg);

    const int B = M->B, T = M->T, D = M->D, V = M->V, F = M->F;
    const size_t VD = (size_t)V * D;
    const size_t DD = (size_t)D * D;
    const size_t DF = (size_t)D * F;
    const size_t FD = (size_t)F * D;

    // params
    M->p.E     = (float*)malloc(VD * sizeof(float));
    M->p.ln1_g = (float*)malloc(D * sizeof(float));
    M->p.ln1_b = (float*)malloc(D * sizeof(float));
    M->p.Wq    = (float*)malloc(DD * sizeof(float));
    M->p.Wk    = (float*)malloc(DD * sizeof(float));
    M->p.Wv    = (float*)malloc(DD * sizeof(float));
    M->p.Wo    = (float*)malloc(DD * sizeof(float));
    M->p.ln2_g = (float*)malloc(D * sizeof(float));
    M->p.ln2_b = (float*)malloc(D * sizeof(float));
    M->p.W1    = (float*)malloc(DF * sizeof(float));
    M->p.b1    = (float*)malloc(F  * sizeof(float));
    M->p.W2    = (float*)malloc(FD * sizeof(float));
    M->p.b2    = (float*)malloc(D  * sizeof(float));

    if (!M->p.E || !M->p.ln1_g || !M->p.ln1_b || !M->p.Wq || !M->p.Wk ||
        !M->p.Wv || !M->p.Wo || !M->p.ln2_g || !M->p.ln2_b ||
        !M->p.W1 || !M->p.b1 || !M->p.W2 || !M->p.b2){
        fprintf(stderr, "[cpu] param malloc failed\n");
        exit(1);
    }

    struct CPU_RNG rng;
    rng_seed(&rng, (uint32_t)cfg->seed);

    // init params (roughly mirror GPU scheme)
    fill_gaussian(M->p.E, VD, 0.02f, &rng);

    fill_ones (M->p.ln1_g, D);
    fill_zeros(M->p.ln1_b, D);
    fill_ones (M->p.ln2_g, D);
    fill_zeros(M->p.ln2_b, D);

    float std_D = 1.0f / sqrtf((float)D);
    float std_F = 1.0f / sqrtf((float)F);

    fill_gaussian(M->p.Wq, DD, std_D, &rng);
    fill_gaussian(M->p.Wk, DD, std_D, &rng);
    fill_gaussian(M->p.Wv, DD, std_D, &rng);
    fill_gaussian(M->p.Wo, DD, std_D, &rng);
    fill_gaussian(M->p.W1, DF, std_D, &rng); // in_dim=D
    fill_gaussian(M->p.W2, FD, std_F, &rng); // in_dim=F

    fill_zeros(M->p.b1, F);
    fill_zeros(M->p.b2, D);

    // positional tables
    const int H = D / 2;
    M->pos_sin = (float*)malloc((size_t)T * H * sizeof(float));
    M->pos_cos = (float*)malloc((size_t)T * H * sizeof(float));
    if (!M->pos_sin || !M->pos_cos){
        fprintf(stderr, "[cpu] pos table malloc failed\n");
        exit(1);
    }
    make_sincos_tables(T, D, M->pos_sin, M->pos_cos);

    // buffers
    const size_t BT  = (size_t)B * T;
    const size_t BTD = BT * D;
    const size_t BTF = BT * F;
    const size_t BTT = (size_t)B * T * T;
    const size_t BTV = BT * V;

    M->buf.x        = (float*)malloc(BTD * sizeof(float));
    M->buf.x_ln1    = (float*)malloc(BTD * sizeof(float));
    M->buf.q        = (float*)malloc(BTD * sizeof(float));
    M->buf.k        = (float*)malloc(BTD * sizeof(float));
    M->buf.v        = (float*)malloc(BTD * sizeof(float));
    M->buf.scores   = (float*)malloc(BTT * sizeof(float));
    M->buf.probs    = (float*)malloc(BTT * sizeof(float));
    M->buf.attn_out = (float*)malloc(BTD * sizeof(float));
    M->buf.x_res1   = (float*)malloc(BTD * sizeof(float));
    M->buf.x_ln2    = (float*)malloc(BTD * sizeof(float));
    M->buf.ff1      = (float*)malloc(BTF * sizeof(float));
    M->buf.ff1_pre  = (float*)malloc(BTF * sizeof(float));
    M->buf.ff2      = (float*)malloc(BTD * sizeof(float));
    M->buf.x_res2   = (float*)malloc(BTD * sizeof(float));
    M->buf.logits   = (float*)malloc(BTV * sizeof(float));

    if (!M->buf.x || !M->buf.x_ln1 || !M->buf.q || !M->buf.k || !M->buf.v ||
        !M->buf.scores || !M->buf.probs || !M->buf.attn_out || !M->buf.x_res1 ||
        !M->buf.x_ln2 || !M->buf.ff1 || !M->buf.ff1_pre || !M->buf.ff2 ||
        !M->buf.x_res2 || !M->buf.logits){
        fprintf(stderr, "[cpu] buffer malloc failed\n");
        exit(1);
    }
}

static void cpu_model_free(struct CPUModel* M){
    if (!M) return;
    // params
    free(M->p.E);
    free(M->p.ln1_g); free(M->p.ln1_b);
    free(M->p.Wq); free(M->p.Wk); free(M->p.Wv); free(M->p.Wo);
    free(M->p.ln2_g); free(M->p.ln2_b);
    free(M->p.W1); free(M->p.b1);
    free(M->p.W2); free(M->p.b2);
    // pos
    free(M->pos_sin); free(M->pos_cos);
    // buffers
    free(M->buf.x);
    free(M->buf.x_ln1);
    free(M->buf.q); free(M->buf.k); free(M->buf.v);
    free(M->buf.scores); free(M->buf.probs);
    free(M->buf.attn_out);
    free(M->buf.x_res1);
    free(M->buf.x_ln2);
    free(M->buf.ff1); free(M->buf.ff1_pre);
    free(M->buf.ff2);
    free(M->buf.x_res2);
    free(M->buf.logits);
    memset(M, 0, sizeof(*M));
}

static void cpu_model_log_summary(const struct CPUModel* M, const struct Config* cfg){
    (void)cfg;
    if (!M){
        printf("CPU Model: (null)\n");
        return;
    }
    double nparams = (double)M->n_params;
    double bytes   = nparams * sizeof(float);
    double mib     = bytes / (1024.0 * 1024.0);

    printf("cpu model summary:\n");
    printf("  dims: B=%d T=%d D=%d V=%d F=%d\n",
           M->B, M->T, M->D, M->V, M->F);
    printf("  parameters: %.0f (%.2f MiB as fp32)\n", nparams, mib);
    printf("  tensors (conceptual):\n");
    printf("    E           : [%d, %d]\n", M->V, M->D);
    printf("    ln1_g/ln1_b : [%d] / [%d]\n", M->D, M->D);
    printf("    Wq/Wk/Wv/Wo : 4 x [%d, %d]\n", M->D, M->D);
    printf("    ln2_g/ln2_b : [%d] / [%d]\n", M->D, M->D);
    printf("    W1/b1       : [%d, %d] / [%d]\n", M->D, M->F, M->F);
    printf("    W2/b2       : [%d, %d] / [%d]\n", M->F, M->D, M->D);
    printf("  pos tables    : sin/cos [%d, %d]\n", M->T, M->D/2);
}

// ---------------------- CPU kernels ----------------------

static void embed_add_pos_cpu(const uint8_t* tokens,
                              const float*   E,
                              const float*   pos_sin,
                              const float*   pos_cos,
                              float*         out,
                              int B, int T, int D)
{
    const int H = D / 2;
    for (int b = 0; b < B; ++b){
        for (int t = 0; t < T; ++t){
            int row = b * T + t;
            int tok = (int)tokens[row];
            size_t ebase = (size_t)tok * D;
            size_t obase = (size_t)row * D;
            size_t pbase = (size_t)t * H;
            for (int d = 0; d < D; ++d){
                float v = E[ebase + d];
                int i = d >> 1;
                v += ((d & 1) == 0) ? pos_sin[pbase + i]
                                    : pos_cos[pbase + i];
                out[obase + d] = v;
            }
        }
    }
}

static void layernorm_forward_cpu(const float* x, float* y,
                                  const float* gamma, const float* beta,
                                  int N, int D, float eps)
{
    for (int row = 0; row < N; ++row){
        size_t base = (size_t)row * D;
        float mean = 0.0f;
        for (int j = 0; j < D; ++j) mean += x[base + j];
        mean /= (float)D;

        float var = 0.0f;
        for (int j = 0; j < D; ++j){
            float d = x[base + j] - mean;
            var += d * d;
        }
        var /= (float)D;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int j = 0; j < D; ++j){
            float v = (x[base + j] - mean) * inv_std;
            y[base + j] = v * gamma[j] + beta[j];
        }
    }
}

static void attention_forward_cpu(const float* x_ln,
                                  int B, int T, int D,
                                  const float* Wq,
                                  const float* Wk,
                                  const float* Wv,
                                  const float* Wo,
                                  float* q, float* k, float* v,
                                  float* scores, float* probs,
                                  float* ctx, float* out)
{
    const int BT = B * T;
    // Q,K,V projections
    for (int b = 0; b < B; ++b){
        for (int t = 0; t < T; ++t){
            int row = b * T + t;
            size_t xbase = (size_t)row * D;
            size_t qbase = (size_t)row * D;
            size_t kbase = (size_t)row * D;
            size_t vbase = (size_t)row * D;

            for (int d_out = 0; d_out < D; ++d_out){
                float sumQ = 0.0f, sumK = 0.0f, sumV = 0.0f;
                for (int kdim = 0; kdim < D; ++kdim){
                    float xv = x_ln[xbase + kdim];
                    size_t widx = (size_t)kdim * D + d_out;
                    sumQ += xv * Wq[widx];
                    sumK += xv * Wk[widx];
                    sumV += xv * Wv[widx];
                }
                q[qbase + d_out] = sumQ;
                k[kbase + d_out] = sumK;
                v[vbase + d_out] = sumV;
            }
        }
    }

    const float scale = 1.0f / sqrtf((float)D);

    // Attention scores + softmax (causal)
    for (int b = 0; b < B; ++b){
        size_t sb = (size_t)b * T * T;
        for (int t_q = 0; t_q < T; ++t_q){
            int row_q = b * T + t_q;
            float maxv = -1e30f;

            // scores
            for (int t_k = 0; t_k < T; ++t_k){
                int row_k = b * T + t_k;
                float dot = 0.0f;
                size_t qbase = (size_t)row_q * D;
                size_t kbase = (size_t)row_k * D;
                for (int d = 0; d < D; ++d){
                    dot += q[qbase + d] * k[kbase + d];
                }
                float s = dot * scale;
                if (t_k > t_q) s = -1e30f; // causal mask
                size_t idx = sb + (size_t)t_q * T + t_k;
                scores[idx] = s;
                if (s > maxv) maxv = s;
            }

            // softmax
            double sum = 0.0;
            for (int t_k = 0; t_k < T; ++t_k){
                size_t idx = sb + (size_t)t_q * T + t_k;
                float e = expf(scores[idx] - maxv);
                probs[idx] = e;
                sum += e;
            }
            float inv_sum = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
            for (int t_k = 0; t_k < T; ++t_k){
                size_t idx = sb + (size_t)t_q * T + t_k;
                probs[idx] *= inv_sum;
            }

            // ctx = probs * V
            size_t ctxbase = (size_t)row_q * D;
            for (int d = 0; d < D; ++d){
                float acc = 0.0f;
                for (int t_k = 0; t_k < T; ++t_k){
                    int row_k = b * T + t_k;
                    size_t idx = sb + (size_t)t_q * T + t_k;
                    size_t vbase = (size_t)row_k * D;
                    acc += probs[idx] * v[vbase + d];
                }
                ctx[ctxbase + d] = acc;
            }
        }
    }

    // Output projection: out = ctx @ Wo
    for (int row = 0; row < BT; ++row){
        size_t cbase = (size_t)row * D;
        size_t obase = (size_t)row * D;
        for (int d_out = 0; d_out < D; ++d_out){
            float sum = 0.0f;
            for (int kdim = 0; kdim < D; ++kdim){
                sum += ctx[cbase + kdim] * Wo[(size_t)kdim * D + d_out];
            }
            out[obase + d_out] = sum;
        }
    }
}

static void gelu_forward_cpu(const float* x, float* y, int n){
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    for (int i = 0; i < n; ++i){
        float v = x[i];
        float v3 = v * v * v;
        float inner = k0 * (v + k1 * v3);
        float t = tanhf(inner);
        y[i] = 0.5f * v * (1.0f + t);
    }
}

static float xent_forward_mean_cpu(const float* logits,
                                   const uint8_t* targets,
                                   int rows, int cols)
{
    double total = 0.0;
    for (int i = 0; i < rows; ++i){
        const float* row = logits + (size_t)i * cols;
        float maxv = row[0];
        for (int j = 1; j < cols; ++j)
            if (row[j] > maxv) maxv = row[j];

        double sum = 0.0;
        for (int j = 0; j < cols; ++j)
            sum += expf(row[j] - maxv);

        float logsum = maxv + logf((float)sum);
        int t = (int)targets[i];
        if (t < 0 || t >= cols) t = 0; // safety
        float nll = logsum - row[t];
        total += (double)nll;
    }
    return (float)(total / (double)rows);
}

// ---------------------- Complete CPU forward ----------------------

static float nano2_forward_loss_cpu(struct CPUModel* M,
                                    const uint8_t* tokens_x,
                                    const uint8_t* tokens_y)
{
    const int B  = M->B;
    const int T  = M->T;
    const int D  = M->D;
    const int V  = M->V;
    const int F  = M->F;
    const int BT = B * T;

    // 1) Embedding + positions → x
    embed_add_pos_cpu(tokens_x,
                      M->p.E,
                      M->pos_sin,
                      M->pos_cos,
                      M->buf.x,
                      B, T, D);

    // 2) LN1 + attention + residual
    layernorm_forward_cpu(M->buf.x,
                          M->buf.x_ln1,
                          M->p.ln1_g, M->p.ln1_b,
                          BT, D, 1e-5f);

    attention_forward_cpu(M->buf.x_ln1,
                          B, T, D,
                          M->p.Wq, M->p.Wk, M->p.Wv, M->p.Wo,
                          M->buf.q, M->buf.k, M->buf.v,
                          M->buf.scores, M->buf.probs,
                          M->buf.x_res1,     // reuse as ctx
                          M->buf.attn_out);  // attention output

    // x_res1 = x + attn_out
    const size_t BTD = (size_t)BT * D;
    for (size_t i = 0; i < BTD; ++i){
        M->buf.x_res1[i] = M->buf.x[i] + M->buf.attn_out[i];
    }

    // 3) FFN block: LN2 → W1+b1 → GELU → W2+b2 → residual
    layernorm_forward_cpu(M->buf.x_res1,
                          M->buf.x_ln2,
                          M->p.ln2_g, M->p.ln2_b,
                          BT, D, 1e-5f);

    // ff1_pre = x_ln2 @ W1 + b1  (shape [BT,F])
    for (int row = 0; row < BT; ++row){
        size_t xb = (size_t)row * D;
        size_t fb = (size_t)row * F;
        for (int f = 0; f < F; ++f){
            float sum = 0.0f;
            for (int d = 0; d < D; ++d){
                sum += M->buf.x_ln2[xb + d] * M->p.W1[(size_t)d * F + f];
            }
            M->buf.ff1_pre[fb + f] = sum + M->p.b1[f];
        }
    }

    // GELU(ff1_pre) → ff1
    gelu_forward_cpu(M->buf.ff1_pre, M->buf.ff1, BT * F);

    // ff2 = ff1 @ W2 + b2 (shape [BT,D])
    for (int row = 0; row < BT; ++row){
        size_t fb = (size_t)row * F;
        size_t db = (size_t)row * D;
        for (int d = 0; d < D; ++d){
            float sum = 0.0f;
            for (int f = 0; f < F; ++f){
                sum += M->buf.ff1[fb + f] * M->p.W2[(size_t)f * D + d];
            }
            M->buf.ff2[db + d] = sum + M->p.b2[d];
        }
    }

    // x_res2 = x_res1 + ff2
    for (size_t i = 0; i < BTD; ++i){
        M->buf.x_res2[i] = M->buf.x_res1[i] + M->buf.ff2[i];
    }

    // 4) LM head: logits = x_res2 @ E^T  (shape [BT,V])
    for (int row = 0; row < BT; ++row){
        size_t xb = (size_t)row * D;
        size_t lb = (size_t)row * V;
        for (int v = 0; v < V; ++v){
            float sum = 0.0f;
            size_t eb = (size_t)v * D;
            for (int d = 0; d < D; ++d){
                sum += M->buf.x_res2[xb + d] * M->p.E[eb + d];
            }
            M->buf.logits[lb + v] = sum;
        }
    }

    // 5) Loss
    float mean_loss = xent_forward_mean_cpu(M->buf.logits,
                                            tokens_y,
                                            BT, V);
    return mean_loss;
}

// ---------------------- timing helper ----------------------

static double wall_time_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

// ---------------------- main (serial) ----------------------

int main(int argc, char** argv){
    const int world = 1;
    const int rank  = 0;

    (void)rank; // unused but kept for symmetry

    printf("nano2-cpu-fw-serial: world=%d (no MPI)\n", world);

    char config_path[512];
    get_config_path(argc, argv, config_path, sizeof(config_path));
    int fw_iters = get_fw_iters(argc, argv);

    struct Config cfg;
    if (config_from_file(config_path, &cfg) != 0){
        fprintf(stderr, "failed to load config from %s\n", config_path);
        return 1;
    }

    printf("config: %s\n", config_path);
    config_log(&cfg);
    printf("[cpu-fw-serial] forward iters: %d\n", fw_iters);

    // Load datasets
    struct DataSet train_ds, val_ds;
    if (dataset_load(cfg.train_path, &train_ds) != 0){
        fprintf(stderr, "failed to load train dataset '%s'\n", cfg.train_path);
        return 1;
    }
    if (dataset_load(cfg.val_path, &val_ds) != 0){
        fprintf(stderr, "failed to load val dataset '%s'\n", cfg.val_path);
        dataset_free(&train_ds);
        return 1;
    }

    dataset_log(&train_ds, "train");
    dataset_log(&val_ds,   "val");

    // Init CPU model
    struct CPUModel M;
    cpu_model_init(&M, &cfg);
    cpu_model_log_summary(&M, &cfg);

    const int B  = cfg.batch_size;
    const int T  = cfg.seq_len;
    const int BT = B * T;

    uint8_t* x = (uint8_t*)malloc((size_t)BT);
    uint8_t* y = (uint8_t*)malloc((size_t)BT);
    if (!x || !y){
        fprintf(stderr, "[cpu-fw-serial] host batch malloc failed\n");
        free(x);
        free(y);
        cpu_model_free(&M);
        dataset_free(&train_ds);
        dataset_free(&val_ds);
        return 1;
    }

    dataset_next_batch(&train_ds, B, T, x, y);

    // Preview
    int preview = (T < 16) ? T : 16;
    printf("batch preview x[0,0:%d): ", preview);
    for (int t = 0; t < preview; ++t) printf("%u ", (unsigned)x[t]);
    printf("\n");
    printf("batch preview y[0,0:%d): ", preview);
    for (int t = 0; t < preview; ++t) printf("%u ", (unsigned)y[t]);
    printf("\n");

    // --- timing ---
    double t0 = wall_time_ms();

    float loss = 0.0f;
    for (int i = 0; i < fw_iters; ++i){
        dataset_next_batch(&train_ds, B, T, x, y);
        loss = nano2_forward_loss_cpu(&M, x, y);
    }

    double t1 = wall_time_ms();
    double ms = t1 - t0;

    double toks       = (double)BT * (double)fw_iters * (double)world;
    double toks_per_s = toks / (ms * 1e-3);

    printf("cpu forward mean loss (last iter): %.6f (expect ~ ln(256)=5.545 random)\n", loss);
    printf("iters: %d | world size: %d\n", fw_iters, world);
    printf("total wall time (serial): %.3f ms\n", ms);
    printf("time/iter: %.3f ms\n", ms / (double)fw_iters);
    printf("tokens/iter: %d (B=%d * T=%d)\n", BT, B, T);
    printf("tokens/sec (serial CPU fw): %.0f\n", toks_per_s);

    free(x);
    free(y);
    cpu_model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);

    return 0;
}

