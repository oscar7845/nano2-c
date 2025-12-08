#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

//must match your config.c definition
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

// number of forward+backward iterations: --fw-bw-iters=N (default 1)
static int get_fw_bw_iters(int argc, char** argv){
    int iters = 1;
    for (int i = 1; i < argc; ++i){
        if (strncmp(argv[i], "--fw-bw-iters=", 14) == 0){
            iters = atoi(argv[i] + 14);
        } else if (strcmp(argv[i], "--fw-bw-iters") == 0 && i + 1 < argc){
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

struct CPUGrads {
    float* dE;
    float* dln1_g; float* dln1_b;
    float* dWq; float* dWk; float* dWv; float* dWo;
    float* dln2_g; float* dln2_b;
    float* dW1; float* db1;
    float* dW2; float* db2;
};

struct CPUBuffer {
    float* x;        // (B,T,D)
    float* x_ln1;    // (B,T,D)
    float* q;        // (B,T,D)
    float* k;        // (B,T,D)
    float* v;        // (B,T,D)
    float* scores;   // (B,T,T)
    float* probs;    // (B,T,T)
    float* ctx;      // (B,T,D)
    float* attn_out; // (B,T,D)
    float* x_res1;   // (B,T,D)
    float* x_ln2;    // (B,T,D)
    float* ff1_pre;  // (B,T,F)
    float* ff1;      // (B,T,F)
    float* ff2;      // (B,T,D)
    float* x_res2;   // (B,T,D)
    float* logits;   // (B,T,V)
};

struct CPUGradsBuf {
    float* dx;        // grad wrt x
    float* dx_ln1;
    float* dq; float* dk; float* dv;
    float* dscores; float* dprobs;
    float* dctx;
    float* dattn_out;
    float* dx_res1;
    float* dx_ln2;
    float* dff1_pre;
    float* dff1;
    float* dff2;
    float* dx_res2;
    float* dlogits;
};

struct CPUModel {
    int B, T, D, V, F;
    struct CPUParams   p;
    struct CPUGrads    g;
    struct CPUBuffer   buf;
    struct CPUGradsBuf gbuf;
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
    uint64_t x = r->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->state = x;
    uint32_t hi = (uint32_t)(x >> 32);
    return hi / 4294967296.0f; // [0,1)
}

static float rng_normal(struct CPU_RNG* r){
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
    memset(dst, 0, n * sizeof(float));
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

static void* safe_malloc(size_t nbytes, const char* msg){
    void* p = malloc(nbytes);
    if (!p){
        fprintf(stderr, "[cpu] malloc failed for %s (%zu bytes)\n", msg, nbytes);
        exit(1);
    }
    return p;
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
    M->p.E     = (float*)safe_malloc(VD * sizeof(float), "E");
    M->p.ln1_g = (float*)safe_malloc(D * sizeof(float), "ln1_g");
    M->p.ln1_b = (float*)safe_malloc(D * sizeof(float), "ln1_b");
    M->p.Wq    = (float*)safe_malloc(DD * sizeof(float), "Wq");
    M->p.Wk    = (float*)safe_malloc(DD * sizeof(float), "Wk");
    M->p.Wv    = (float*)safe_malloc(DD * sizeof(float), "Wv");
    M->p.Wo    = (float*)safe_malloc(DD * sizeof(float), "Wo");
    M->p.ln2_g = (float*)safe_malloc(D * sizeof(float), "ln2_g");
    M->p.ln2_b = (float*)safe_malloc(D * sizeof(float), "ln2_b");
    M->p.W1    = (float*)safe_malloc(DF * sizeof(float), "W1");
    M->p.b1    = (float*)safe_malloc(F  * sizeof(float), "b1");
    M->p.W2    = (float*)safe_malloc(FD * sizeof(float), "W2");
    M->p.b2    = (float*)safe_malloc(D  * sizeof(float), "b2");

    // grad params
    M->g.dE     = (float*)safe_malloc(VD * sizeof(float), "dE");
    M->g.dln1_g = (float*)safe_malloc(D * sizeof(float), "dln1_g");
    M->g.dln1_b = (float*)safe_malloc(D * sizeof(float), "dln1_b");
    M->g.dWq    = (float*)safe_malloc(DD * sizeof(float), "dWq");
    M->g.dWk    = (float*)safe_malloc(DD * sizeof(float), "dWk");
    M->g.dWv    = (float*)safe_malloc(DD * sizeof(float), "dWv");
    M->g.dWo    = (float*)safe_malloc(DD * sizeof(float), "dWo");
    M->g.dln2_g = (float*)safe_malloc(D * sizeof(float), "dln2_g");
    M->g.dln2_b = (float*)safe_malloc(D * sizeof(float), "dln2_b");
    M->g.dW1    = (float*)safe_malloc(DF * sizeof(float), "dW1");
    M->g.db1    = (float*)safe_malloc(F  * sizeof(float), "db1");
    M->g.dW2    = (float*)safe_malloc(FD * sizeof(float), "dW2");
    M->g.db2    = (float*)safe_malloc(D  * sizeof(float), "db2");

    // positional tables
    const int H = D / 2;
    M->pos_sin = (float*)safe_malloc((size_t)T * H * sizeof(float), "pos_sin");
    M->pos_cos = (float*)safe_malloc((size_t)T * H * sizeof(float), "pos_cos");
    make_sincos_tables(T, D, M->pos_sin, M->pos_cos);

    const size_t BT  = (size_t)B * T;
    const size_t BTD = BT * D;
    const size_t BTF = BT * F;
    const size_t BTT = (size_t)B * T * T;
    const size_t BTV = BT * V;

    // forward buffers
    M->buf.x        = (float*)safe_malloc(BTD * sizeof(float), "x");
    M->buf.x_ln1    = (float*)safe_malloc(BTD * sizeof(float), "x_ln1");
    M->buf.q        = (float*)safe_malloc(BTD * sizeof(float), "q");
    M->buf.k        = (float*)safe_malloc(BTD * sizeof(float), "k");
    M->buf.v        = (float*)safe_malloc(BTD * sizeof(float), "v");
    M->buf.scores   = (float*)safe_malloc(BTT * sizeof(float), "scores");
    M->buf.probs    = (float*)safe_malloc(BTT * sizeof(float), "probs");
    M->buf.ctx      = (float*)safe_malloc(BTD * sizeof(float), "ctx");
    M->buf.attn_out = (float*)safe_malloc(BTD * sizeof(float), "attn_out");
    M->buf.x_res1   = (float*)safe_malloc(BTD * sizeof(float), "x_res1");
    M->buf.x_ln2    = (float*)safe_malloc(BTD * sizeof(float), "x_ln2");
    M->buf.ff1_pre  = (float*)safe_malloc(BTF * sizeof(float), "ff1_pre");
    M->buf.ff1      = (float*)safe_malloc(BTF * sizeof(float), "ff1");
    M->buf.ff2      = (float*)safe_malloc(BTD * sizeof(float), "ff2");
    M->buf.x_res2   = (float*)safe_malloc(BTD * sizeof(float), "x_res2");
    M->buf.logits   = (float*)safe_malloc(BTV * sizeof(float), "logits");

    // grad buffers
    M->gbuf.dx        = (float*)safe_malloc(BTD * sizeof(float), "dx");
    M->gbuf.dx_ln1    = (float*)safe_malloc(BTD * sizeof(float), "dx_ln1");
    M->gbuf.dq        = (float*)safe_malloc(BTD * sizeof(float), "dq");
    M->gbuf.dk        = (float*)safe_malloc(BTD * sizeof(float), "dk");
    M->gbuf.dv        = (float*)safe_malloc(BTD * sizeof(float), "dv");
    M->gbuf.dscores   = (float*)safe_malloc(BTT * sizeof(float), "dscores");
    M->gbuf.dprobs    = (float*)safe_malloc(BTT * sizeof(float), "dprobs");
    M->gbuf.dctx      = (float*)safe_malloc(BTD * sizeof(float), "dctx");
    M->gbuf.dattn_out = (float*)safe_malloc(BTD * sizeof(float), "dattn_out");
    M->gbuf.dx_res1   = (float*)safe_malloc(BTD * sizeof(float), "dx_res1");
    M->gbuf.dx_ln2    = (float*)safe_malloc(BTD * sizeof(float), "dx_ln2");
    M->gbuf.dff1_pre  = (float*)safe_malloc(BTF * sizeof(float), "dff1_pre");
    M->gbuf.dff1      = (float*)safe_malloc(BTF * sizeof(float), "dff1");
    M->gbuf.dff2      = (float*)safe_malloc(BTD * sizeof(float), "dff2");
    M->gbuf.dx_res2   = (float*)safe_malloc(BTD * sizeof(float), "dx_res2");
    M->gbuf.dlogits   = (float*)safe_malloc(BTV * sizeof(float), "dlogits");

    // init params
    struct CPU_RNG rng;
    rng_seed(&rng, (uint32_t)cfg->seed);

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
    // grad params
    free(M->g.dE);
    free(M->g.dln1_g); free(M->g.dln1_b);
    free(M->g.dWq); free(M->g.dWk); free(M->g.dWv); free(M->g.dWo);
    free(M->g.dln2_g); free(M->g.dln2_b);
    free(M->g.dW1); free(M->g.db1);
    free(M->g.dW2); free(M->g.db2);
    // pos
    free(M->pos_sin); free(M->pos_cos);
    // buffers
    free(M->buf.x);
    free(M->buf.x_ln1);
    free(M->buf.q); free(M->buf.k); free(M->buf.v);
    free(M->buf.scores); free(M->buf.probs);
    free(M->buf.ctx);
    free(M->buf.attn_out);
    free(M->buf.x_res1);
    free(M->buf.x_ln2);
    free(M->buf.ff1_pre); free(M->buf.ff1);
    free(M->buf.ff2);
    free(M->buf.x_res2);
    free(M->buf.logits);
    // grad buffers
    free(M->gbuf.dx);
    free(M->gbuf.dx_ln1);
    free(M->gbuf.dq); free(M->gbuf.dk); free(M->gbuf.dv);
    free(M->gbuf.dscores); free(M->gbuf.dprobs);
    free(M->gbuf.dctx);
    free(M->gbuf.dattn_out);
    free(M->gbuf.dx_res1);
    free(M->gbuf.dx_ln2);
    free(M->gbuf.dff1_pre); free(M->gbuf.dff1);
    free(M->gbuf.dff2);
    free(M->gbuf.dx_res2);
    free(M->gbuf.dlogits);

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

    printf("cpu fw+bw model summary:\n");
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

// CPU kernels (forward)
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
    for (int row = 0; row < BT; ++row){
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

    const float scale = 1.0f / sqrtf((float)D);

    // scores + softmax + ctx
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
                if (t_k > t_q) s = -1e30f; // causal
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

            // ctx
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

    // out = ctx @ Wo
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
        if (t < 0 || t >= cols) t = 0;
        float nll = logsum - row[t];
        total += (double)nll;
    }
    return (float)(total / (double)rows);
}

// CPU kernels (backward) 
static void xent_backward_mean_cpu(const float* logits,
                                   const uint8_t* targets,
                                   int rows, int cols,
                                   float* dlogits)
{
    for (int i = 0; i < rows; ++i){
        const float* row = logits + (size_t)i * cols;
        float* drow      = dlogits + (size_t)i * cols;

        float maxv = row[0];
        for (int j = 1; j < cols; ++j)
            if (row[j] > maxv) maxv = row[j];

        double sum = 0.0;
        for (int j = 0; j < cols; ++j)
            sum += expf(row[j] - maxv);

        float inv_sum = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;

        int t = (int)targets[i];
        if (t < 0 || t >= cols) t = 0;

        for (int j = 0; j < cols; ++j){
            float p = expf(row[j] - maxv) * inv_sum;
            float grad = p - ((j == t) ? 1.0f : 0.0f);
            drow[j] = grad / (float)rows;
        }
    }
}

static void layernorm_backward_cpu(
    const float* x,         // [N,D]
    const float* dy,        // [N,D]
    const float* gamma,     // [D]
    int N, int D, float eps,
    float* dx,              // [N,D]
    float* dgamma,          // [D], accumulated
    float* dbeta)           // [D], accumulated
{
    memset(dx,     0, (size_t)N * D * sizeof(float));
    memset(dgamma, 0, (size_t)D * sizeof(float));
    memset(dbeta,  0, (size_t)D * sizeof(float));

    float* xhat  = (float*)safe_malloc((size_t)D * sizeof(float), "ln_xhat");
    float* dxhat = (float*)safe_malloc((size_t)D * sizeof(float), "ln_dxhat");

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
            xhat[j] = (x[base + j] - mean) * inv_std;
        }

        // dgamma, dbeta
        for (int j = 0; j < D; ++j){
            float g = dy[base + j];
            dbeta[j]  += g;
            dgamma[j] += g * xhat[j];
        }

        // dxhat
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int j = 0; j < D; ++j){
            float g = dy[base + j] * gamma[j];
            dxhat[j] = g;
            sum1 += g;
            sum2 += g * xhat[j];
        }

        const float invD = 1.0f / (float)D;
        for (int j = 0; j < D; ++j){
            float term = (float)D * dxhat[j] - sum1 - xhat[j] * sum2;
            dx[base + j] = invD * inv_std * term;
        }
    }

    free(xhat);
    free(dxhat);
}

static void gelu_backward_cpu(const float* x, const float* dy, float* dx, int n){
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    for (int i = 0; i < n; ++i){
        float v = x[i];
        float v2 = v * v;
        float v3 = v2 * v;
        float inner = k0 * (v + k1 * v3);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float d_inner = k0 * (1.0f + 3.0f * k1 * v2);
        float dy_dx = 0.5f * (1.0f + t) + 0.5f * v * sech2 * d_inner;
        dx[i] = dy[i] * dy_dx;
    }
}

static void attention_backward_cpu(
    int B, int T, int D,
    const float* x_ln,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    const float* q, const float* k, const float* v,
    const float* scores, const float* probs,
    const float* ctx,
    const float* d_out,     // [BT,D]
    float* dq, float* dk, float* dv,       // [BT,D]
    float* d_scores, float* d_probs,       // [B,T,T]
    float* d_ctx,                          // [BT,D]
    float* d_x_ln,                         // [BT,D]
    float* dWq, float* dWk, float* dWv, float* dWo)
{
    const int BT = B * T;
    const float scale = 1.0f / sqrtf((float)D);

    memset(dq, 0, (size_t)BT * D * sizeof(float));
    memset(dk, 0, (size_t)BT * D * sizeof(float));
    memset(dv, 0, (size_t)BT * D * sizeof(float));
    memset(d_scores, 0, (size_t)B * T * T * sizeof(float));
    memset(d_probs,  0, (size_t)B * T * T * sizeof(float));
    memset(d_ctx,    0, (size_t)BT * D * sizeof(float));
    memset(d_x_ln,   0, (size_t)BT * D * sizeof(float));
    memset(dWq,      0, (size_t)D * D * sizeof(float));
    memset(dWk,      0, (size_t)D * D * sizeof(float));
    memset(dWv,      0, (size_t)D * D * sizeof(float));
    memset(dWo,      0, (size_t)D * D * sizeof(float));

    // 1) out = ctx @ Wo
    for (int r = 0; r < BT; ++r){
        size_t cbase = (size_t)r * D;
        size_t obase = (size_t)r * D;
        for (int j = 0; j < D; ++j){
            float go = d_out[obase + j];
            for (int d0 = 0; d0 < D; ++d0){
                d_ctx[cbase + d0] += go * Wo[(size_t)d0 * D + j];
                dWo[(size_t)d0 * D + j] += ctx[cbase + d0] * go;
            }
        }
    }

    // 2) ctx = probs * v  (per batch)
    for (int b = 0; b < B; ++b){
        size_t sb = (size_t)b * T * T;
        for (int t_q = 0; t_q < T; ++t_q){
            int row_q = b * T + t_q;
            size_t ctxbase = (size_t)row_q * D;

            for (int t_k = 0; t_k < T; ++t_k){
                int row_k = b * T + t_k;
                size_t idx = sb + (size_t)t_q * T + t_k;
                size_t vbase = (size_t)row_k * D;

                float dP_val = 0.0f;
                for (int d0 = 0; d0 < D; ++d0){
                    float dctx = d_ctx[ctxbase + d0];
                    dP_val += dctx * v[vbase + d0];
                    dv[vbase + d0] += dctx * probs[idx];
                }
                d_probs[idx] = dP_val;
            }
        }
    }

    // 3) softmax backward: probs = softmax(scores)
    for (int b = 0; b < B; ++b){
        size_t sb = (size_t)b * T * T;
        for (int t_q = 0; t_q < T; ++t_q){
            float sum_g_p = 0.0f;
            for (int t_k = 0; t_k < T; ++t_k){
                size_t idx = sb + (size_t)t_q * T + t_k;
                float g = d_probs[idx];
                float p = probs[idx];
                sum_g_p += g * p;
            }
            for (int t_k = 0; t_k < T; ++t_k){
                size_t idx = sb + (size_t)t_q * T + t_k;
                float g = d_probs[idx];
                float p = probs[idx];
                float ds = p * (g - sum_g_p);
                if (t_k > t_q) ds = 0.0f; // keep causal mask strict
                d_scores[idx] = ds;
            }
        }
    }

    // 4) scores from q,k: scores = scale * q·k^T
    for (int b = 0; b < B; ++b){
        size_t sb = (size_t)b * T * T;
        for (int t_q = 0; t_q < T; ++t_q){
            int row_q = b * T + t_q;
            size_t qbase = (size_t)row_q * D;
            for (int t_k = 0; t_k <= t_q; ++t_k){
                int row_k = b * T + t_k;
                size_t kbase = (size_t)row_k * D;
                size_t idx = sb + (size_t)t_q * T + t_k;
                float ds = d_scores[idx] * scale;
                for (int d0 = 0; d0 < D; ++d0){
                    dq[qbase + d0] += ds * k[kbase + d0];
                    dk[kbase + d0] += ds * q[qbase + d0];
                }
            }
        }
    }

    // 5) q,k,v from x_ln via Wq,Wk,Wv
    // reuse pattern: dW = X^T * dY; dX += dY * W^T
    const int BT_int = BT;

    // Q
    for (int di = 0; di < D; ++di){
        for (int dj = 0; dj < D; ++dj){
            double acc = 0.0;
            for (int r = 0; r < BT_int; ++r){
                acc += (double)x_ln[(size_t)r * D + di] *
                       (double)dq[(size_t)r * D + dj];
            }
            dWq[(size_t)di * D + dj] = (float)acc;
        }
    }
    for (int r = 0; r < BT_int; ++r){
        size_t base = (size_t)r * D;
        for (int di = 0; di < D; ++di){
            double acc = 0.0;
            for (int dj = 0; dj < D; ++dj){
                acc += (double)dq[base + dj] * (double)Wq[(size_t)di * D + dj];
            }
            d_x_ln[base + di] += (float)acc;
        }
    }

    // K
    for (int di = 0; di < D; ++di){
        for (int dj = 0; dj < D; ++dj){
            double acc = 0.0;
            for (int r = 0; r < BT_int; ++r){
                acc += (double)x_ln[(size_t)r * D + di] *
                       (double)dk[(size_t)r * D + dj];
            }
            dWk[(size_t)di * D + dj] = (float)acc;
        }
    }
    for (int r = 0; r < BT_int; ++r){
        size_t base = (size_t)r * D;
        for (int di = 0; di < D; ++di){
            double acc = 0.0;
            for (int dj = 0; dj < D; ++dj){
                acc += (double)dk[base + dj] * (double)Wk[(size_t)di * D + dj];
            }
            d_x_ln[base + di] += (float)acc;
        }
    }

    // V
    for (int di = 0; di < D; ++di){
        for (int dj = 0; dj < D; ++dj){
            double acc = 0.0;
            for (int r = 0; r < BT_int; ++r){
                acc += (double)x_ln[(size_t)r * D + di] *
                       (double)dv[(size_t)r * D + dj];
            }
            dWv[(size_t)di * D + dj] = (float)acc;
        }
    }
    for (int r = 0; r < BT_int; ++r){
        size_t base = (size_t)r * D;
        for (int di = 0; di < D; ++di){
            double acc = 0.0;
            for (int dj = 0; dj < D; ++dj){
                acc += (double)dv[base + dj] * (double)Wv[(size_t)di * D + dj];
            }
            d_x_ln[base + di] += (float)acc;
        }
    }
}

//Complete forward + backward
static float nano2_forward_cpu(struct CPUModel* M,
                               const uint8_t* tokens_x,
                               const uint8_t* tokens_y)
{
    const int B  = M->B;
    const int T  = M->T;
    const int D  = M->D;
    const int V  = M->V;
    const int F  = M->F;
    const int BT = B * T;
    const size_t BTD = (size_t)BT * D;

    // 1) Embedding + positions → x
    embed_add_pos_cpu(tokens_x,
                      M->p.E,
                      M->pos_sin,
                      M->pos_cos,
                      M->buf.x,
                      B, T, D);

    // 2) LN1 + attention + residual1
    layernorm_forward_cpu(M->buf.x,
                          M->buf.x_ln1,
                          M->p.ln1_g, M->p.ln1_b,
                          BT, D, 1e-5f);

    attention_forward_cpu(M->buf.x_ln1,
                          B, T, D,
                          M->p.Wq, M->p.Wk, M->p.Wv, M->p.Wo,
                          M->buf.q, M->buf.k, M->buf.v,
                          M->buf.scores, M->buf.probs,
                          M->buf.ctx,
                          M->buf.attn_out);

    for (size_t i = 0; i < BTD; ++i){
        M->buf.x_res1[i] = M->buf.x[i] + M->buf.attn_out[i];
    }

    // 3) LN2 + FFN + residual2
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
            for (int d0 = 0; d0 < D; ++d0){
                sum += M->buf.x_ln2[xb + d0] * M->p.W1[(size_t)d0 * F + f];
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
        for (int d0 = 0; d0 < D; ++d0){
            float sum = 0.0f;
            for (int f = 0; f < F; ++f){
                sum += M->buf.ff1[fb + f] * M->p.W2[(size_t)f * D + d0];
            }
            M->buf.ff2[db + d0] = sum + M->p.b2[d0];
        }
    }

    for (size_t i = 0; i < BTD; ++i){
        M->buf.x_res2[i] = M->buf.x_res1[i] + M->buf.ff2[i];
    }

    // 4) LM head: logits = x_res2 @ E^T
    const int BT_int = BT;
    for (int row = 0; row < BT_int; ++row){
        size_t xb = (size_t)row * D;
        size_t lb = (size_t)row * V;
        for (int v = 0; v < V; ++v){
            float sum = 0.0f;
            size_t eb = (size_t)v * D;
            for (int d0 = 0; d0 < D; ++d0){
                sum += M->buf.x_res2[xb + d0] * M->p.E[eb + d0];
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

static float nano2_forward_backward_cpu(struct CPUModel* M,
                                        const uint8_t* tokens_x,
                                        const uint8_t* tokens_y)
{
    const int B  = M->B;
    const int T  = M->T;
    const int D  = M->D;
    const int V  = M->V;
    const int F  = M->F;
    const int BT = B * T;
    const size_t BTD = (size_t)BT * D;
    const size_t BTF = (size_t)BT * F;
    const size_t BTT = (size_t)B * T * T;
    const size_t BTV = (size_t)BT * V;

    // Forward
    float loss = nano2_forward_cpu(M, tokens_x, tokens_y);

    // Zero parameter grads
    size_t VD = (size_t)V * D;
    size_t DD = (size_t)D * D;
    size_t DF = (size_t)D * F;
    size_t FD = (size_t)F * D;
    memset(M->g.dE,      0, VD * sizeof(float));
    memset(M->g.dln1_g,  0, D  * sizeof(float));
    memset(M->g.dln1_b,  0, D  * sizeof(float));
    memset(M->g.dWq,     0, DD * sizeof(float));
    memset(M->g.dWk,     0, DD * sizeof(float));
    memset(M->g.dWv,     0, DD * sizeof(float));
    memset(M->g.dWo,     0, DD * sizeof(float));
    memset(M->g.dln2_g,  0, D  * sizeof(float));
    memset(M->g.dln2_b,  0, D  * sizeof(float));
    memset(M->g.dW1,     0, DF * sizeof(float));
    memset(M->g.db1,     0, F  * sizeof(float));
    memset(M->g.dW2,     0, FD * sizeof(float));
    memset(M->g.db2,     0, D  * sizeof(float));

    // Zero grad buffers
    memset(M->gbuf.dx,        0, BTD * sizeof(float));
    memset(M->gbuf.dx_ln1,    0, BTD * sizeof(float));
    memset(M->gbuf.dq,        0, BTD * sizeof(float));
    memset(M->gbuf.dk,        0, BTD * sizeof(float));
    memset(M->gbuf.dv,        0, BTD * sizeof(float));
    memset(M->gbuf.dscores,   0, BTT * sizeof(float));
    memset(M->gbuf.dprobs,    0, BTT * sizeof(float));
    memset(M->gbuf.dctx,      0, BTD * sizeof(float));
    memset(M->gbuf.dattn_out, 0, BTD * sizeof(float));
    memset(M->gbuf.dx_res1,   0, BTD * sizeof(float));
    memset(M->gbuf.dx_ln2,    0, BTD * sizeof(float));
    memset(M->gbuf.dff1_pre,  0, BTF * sizeof(float));
    memset(M->gbuf.dff1,      0, BTF * sizeof(float));
    memset(M->gbuf.dff2,      0, BTD * sizeof(float));
    memset(M->gbuf.dx_res2,   0, BTD * sizeof(float));
    memset(M->gbuf.dlogits,   0, BTV * sizeof(float));

    float* dE      = M->g.dE;
    float* dln1_g  = M->g.dln1_g;
    float* dln1_b  = M->g.dln1_b;
    float* dWq     = M->g.dWq;
    float* dWk     = M->g.dWk;
    float* dWv     = M->g.dWv;
    float* dWo     = M->g.dWo;
    float* dln2_g  = M->g.dln2_g;
    float* dln2_b  = M->g.dln2_b;
    float* dW1     = M->g.dW1;
    float* db1     = M->g.db1;
    float* dW2     = M->g.dW2;
    float* db2     = M->g.db2;

    float* dx        = M->gbuf.dx;
    float* dx_ln1    = M->gbuf.dx_ln1;
    float* dq        = M->gbuf.dq;
    float* dk        = M->gbuf.dk;
    float* dv        = M->gbuf.dv;
    float* dscores   = M->gbuf.dscores;
    float* dprobs    = M->gbuf.dprobs;
    float* dctx      = M->gbuf.dctx;
    float* dattn_out = M->gbuf.dattn_out;
    float* dx_res1   = M->gbuf.dx_res1;
    float* dx_ln2    = M->gbuf.dx_ln2;
    float* dff1_pre  = M->gbuf.dff1_pre;
    float* dff1      = M->gbuf.dff1;
    float* dff2      = M->gbuf.dff2;
    float* dx_res2   = M->gbuf.dx_res2;
    float* dlogits   = M->gbuf.dlogits;

    const float* x      = M->buf.x;
    const float* x_ln1  = M->buf.x_ln1;
    const float* q      = M->buf.q;
    const float* k      = M->buf.k;
    const float* v      = M->buf.v;
    const float* scores = M->buf.scores;
    const float* probs  = M->buf.probs;
    const float* ctx    = M->buf.ctx;
    const float* attn_out = M->buf.attn_out;
    const float* x_res1 = M->buf.x_res1;
    const float* x_ln2  = M->buf.x_ln2;
    const float* ff1_pre = M->buf.ff1_pre;
    const float* ff1    = M->buf.ff1;
    const float* ff2    = M->buf.ff2;
    const float* x_res2 = M->buf.x_res2;
    const float* logits = M->buf.logits;

    // -------- Backward --------

    // 1) Loss & softmax backward: logits -> dlogits
    xent_backward_mean_cpu(logits, tokens_y, BT, V, dlogits);

    // 2) LM head: logits = x_res2 @ E^T
    memset(dx_res2, 0, BTD * sizeof(float));
    for (int r = 0; r < BT; ++r){
        size_t lb = (size_t)r * V;
        size_t xb = (size_t)r * D;
        for (int vj = 0; vj < V; ++vj){
            float g = dlogits[lb + vj];
            size_t eb = (size_t)vj * D;
            for (int d0 = 0; d0 < D; ++d0){
                dx_res2[xb + d0] += g * M->p.E[eb + d0];
                dE[eb + d0]      += g * x_res2[xb + d0];
            }
        }
    }

    // 3) Residual2: x_res2 = x_res1 + ff2
    for (size_t i = 0; i < BTD; ++i){
        dx_res1[i] += dx_res2[i];
        dff2[i]    += dx_res2[i];
    }

    // 4) FF2: ff2 = ff1 @ W2 + b2
    // db2
    for (int d0 = 0; d0 < D; ++d0){
        double acc = 0.0;
        for (int r = 0; r < BT; ++r){
            acc += (double)dff2[(size_t)r * D + d0];
        }
        db2[d0] = (float)acc;
    }
    // dW2
    for (int f = 0; f < F; ++f){
        for (int d0 = 0; d0 < D; ++d0){
            double acc = 0.0;
            for (int r = 0; r < BT; ++r){
                acc += (double)ff1[(size_t)r * F + f] *
                       (double)dff2[(size_t)r * D + d0];
            }
            dW2[(size_t)f * D + d0] = (float)acc;
        }
    }
    // dff1
    memset(dff1, 0, BTF * sizeof(float));
    for (int r = 0; r < BT; ++r){
        size_t fb = (size_t)r * F;
        size_t db = (size_t)r * D;
        for (int f = 0; f < F; ++f){
            double acc = 0.0;
            for (int d0 = 0; d0 < D; ++d0){
                acc += (double)dff2[db + d0] * (double)M->p.W2[(size_t)f * D + d0];
            }
            dff1[fb + f] = (float)acc;
        }
    }

    // 5) GELU backward: ff1 = GELU(ff1_pre)
    gelu_backward_cpu(ff1_pre, dff1, dff1_pre, BT * F);

    // 6) FF1: ff1_pre = x_ln2 @ W1 + b1
    // db1
    for (int f = 0; f < F; ++f){
        double acc = 0.0;
        for (int r = 0; r < BT; ++r){
            acc += (double)dff1_pre[(size_t)r * F + f];
        }
        db1[f] = (float)acc;
    }
    // dW1
    for (int di = 0; di < D; ++di){
        for (int f = 0; f < F; ++f){
            double acc = 0.0;
            for (int r = 0; r < BT; ++r){
                acc += (double)x_ln2[(size_t)r * D + di] *
                       (double)dff1_pre[(size_t)r * F + f];
            }
            dW1[(size_t)di * F + f] = (float)acc;
        }
    }
    // dx_ln2
    memset(dx_ln2, 0, BTD * sizeof(float));
    for (int r = 0; r < BT; ++r){
        size_t xb = (size_t)r * D;
        size_t fb = (size_t)r * F;
        for (int di = 0; di < D; ++di){
            double acc = 0.0;
            for (int f = 0; f < F; ++f){
                acc += (double)dff1_pre[fb + f] * (double)M->p.W1[(size_t)di * F + f];
            }
            dx_ln2[xb + di] = (float)acc;
        }
    }

    // 7) LN2 backward: x_ln2 = LN(x_res1; ln2)
    float* dx_ln2_input = (float*)safe_malloc(BTD * sizeof(float), "dx_ln2_input");
    layernorm_backward_cpu(x_res1, dx_ln2, M->p.ln2_g,
                           BT, D, 1e-5f,
                           dx_ln2_input, dln2_g, dln2_b);
    for (size_t i = 0; i < BTD; ++i){
        dx_res1[i] += dx_ln2_input[i];
    }
    free(dx_ln2_input);

    // 8) Residual1: x_res1 = x + attn_out
    for (size_t i = 0; i < BTD; ++i){
        dattn_out[i] = dx_res1[i];
        dx[i]        += dx_res1[i];
    }

    // 9) Attention backward
    attention_backward_cpu(
        B, T, D,
        x_ln1,
        M->p.Wq, M->p.Wk, M->p.Wv, M->p.Wo,
        q, k, v,
        scores, probs,
        ctx,
        dattn_out,
        dq, dk, dv,
        dscores, dprobs,
        dctx,
        dx_ln1,
        dWq, dWk, dWv, dWo);

    // 10) LN1 backward: x_ln1 = LN(x; ln1)
    float* dx_ln1_input = (float*)safe_malloc(BTD * sizeof(float), "dx_ln1_input");
    layernorm_backward_cpu(x, dx_ln1, M->p.ln1_g,
                           BT, D, 1e-5f,
                           dx_ln1_input, dln1_g, dln1_b);
    for (size_t i = 0; i < BTD; ++i){
        dx[i] += dx_ln1_input[i];
    }
    free(dx_ln1_input);

    // 11) Embedding backward: x = E[token] + pos
    // gradient wrt E accumulates; pos tables are fixed (no grad)
    for (int b = 0; b < B; ++b){
        for (int t = 0; t < T; ++t){
            int row = b * T + t;
            int tok = (int)tokens_x[row];
            size_t xb = (size_t)row * D;
            size_t eb = (size_t)tok * D;
            for (int d0 = 0; d0 < D; ++d0){
                dE[eb + d0] += dx[xb + d0];
            }
        }
    }

    return loss;
}

// ---------------------- main ----------------------

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (rank == 0){
        printf("nano2-cpu-fw+bw: world=%d\n", world);
    }

    char config_path[512];
    get_config_path(argc, argv, config_path, sizeof(config_path));
    int iters = get_fw_bw_iters(argc, argv);

    struct Config cfg;
    config_from_file(config_path, &cfg);
    if (rank == 0){
        printf("config: %s\n", config_path);
        config_log(&cfg);
        printf("[cpu-fw+bw] forward+backward iters: %d\n", iters);
    }

    // Load datasets
    struct DataSet train_ds, val_ds;
    dataset_load(cfg.train_path, &train_ds);
    dataset_load(cfg.val_path,   &val_ds);
    if (rank == 0){
        dataset_log(&train_ds, "train");
        dataset_log(&val_ds,   "val");
    }

    // Init CPU model
    struct CPUModel M;
    cpu_model_init(&M, &cfg);
    if (rank == 0) cpu_model_log_summary(&M, &cfg);

    const int B  = cfg.batch_size;
    const int T  = cfg.seq_len;
    const int BT = B * T;

    uint8_t* x = (uint8_t*)malloc((size_t)BT);
    uint8_t* y = (uint8_t*)malloc((size_t)BT);
    if (!x || !y){
        fprintf(stderr, "[cpu-fw+bw] host batch malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    dataset_next_batch(&train_ds, B, T, x, y);

    // Preview
    if (rank == 0){
        int preview = (T < 16) ? T : 16;
        printf("batch preview x[0,0:%d): ", preview);
        for (int t0 = 0; t0 < preview; ++t0) printf("%u ", (unsigned)x[t0]);
        printf("\n");
        printf("batch preview y[0,0:%d): ", preview);
        for (int t0 = 0; t0 < preview; ++t0) printf("%u ", (unsigned)y[t0]);
        printf("\n");
    }

    // --- timing ---
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    float loss = 0.0f;
    for (int i = 0; i < iters; ++i){
        dataset_next_batch(&train_ds, B, T, x, y);
        loss = nano2_forward_backward_cpu(&M, x, y);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double local_ms = (t1 - t0) * 1000.0;
    double max_ms   = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0){
        double toks       = (double)BT * (double)iters * (double)world;
        double toks_per_s = toks / (max_ms * 1e-3);

        printf("cpu forward+backward mean loss (last iter): %.6f (random-init => near ln(256)=5.545)\n", loss);
        printf("iters: %d | world size: %d\n", iters, world);
        printf("total wall time (max across ranks): %.3f ms\n", max_ms);
        printf("time/iter: %.3f ms\n", max_ms / (double)iters);
        printf("tokens/iter: %d (B=%d * T=%d)\n", BT, B, T);
        printf("global tokens/sec (all ranks, CPU fw+bw): %.0f\n", toks_per_s);
    }

    free(x);
    free(y);
    cpu_model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}

