//bring up executable: 
//MPI and CUDA device selection, load config and data,
//run a single forward pass and print mean loss and timing
//add inference mode with --infer (top-k sampling)
//TODO: rm debug check
#include "nano2_model.h"
#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "chat.h"

//foeward declarations (chat.h)
int run_chat(const char* ckpt_file, const char* ckpt_dir, int use_best,
             struct Config cfg_file, int max_new, int top_k, float temp,
             unsigned int seed);
void get_chat_flags(int argc, char** argv, int* chat,
                    char* ckpt_file, size_t ckpt_file_cap,
                    char* ckpt_dir,  size_t ckpt_dir_cap,
                    int* use_best, int* max_new, float* temperature,
                    int* top_k, unsigned int* seed);

int config_from_file(const char* path, struct Config* out);
void config_log(const struct Config* c);

//(data.c)
struct DataSet {
    uint8_t* data;
    size_t n;
    size_t cursor;
    char path[512];
};
int dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len, uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);

//model + forward/train
struct Model; // opaque here
void model_init(struct Model* M, const struct Config* c);
void model_log_summary(const struct Model* M, const struct Config* c);
void model_free(struct Model* M);
float nano2_forward_loss(struct Model* M, const uint8_t* h_x, const uint8_t* h_y);
float nano2_train_step(struct Model* M, const uint8_t* h_x, const uint8_t* h_y, const struct Config* cfg, int world_size, int rank);

//CLI helpers
static void get_config_path(int argc, char** argv, char* out, size_t cap){
    const char* def = "./configs/nano2.json";
    size_t n = strlen(def); if (n >= cap) n = cap - 1;
    memcpy(out, def, n); out[n] = '\0';
    for (int i = 1; i < argc; ++i){
        const char* a = argv[i];
        if (strncmp(a, "--config=", 9) == 0){
            strncpy(out, a + 9, cap - 1); out[cap-1] = '\0';
        } else if (strcmp(a, "--config") == 0 && i + 1 < argc){
            strncpy(out, argv[i+1], cap - 1); out[cap-1] = '\0'; ++i;
        }
    }
}

static float lr_cosine(float base_lr, int step, int warmup, int total_steps){
    if (step < warmup){
        return base_lr * (float)(step + 1) / (float)warmup;
    }
    int s = step - warmup;
    int d = (total_steps > warmup) ? (total_steps - warmup) : 1;
    float t = (float)s / (float)d;
    return base_lr * 0.5f * (1.0f + cosf(3.1415926535f * t));
}

static void get_train_flags(int argc, char** argv, int* steps, int* eval_interval, int* eval_batches, int* warmup_steps, int* patience, char* ckpt_dir, size_t cap, int* resume){
    *steps=200000; *eval_interval=1000; *eval_batches=50; *warmup_steps=1000; *patience=10; *resume=0;
    const char* def_dir="ckpts"; strncpy(ckpt_dir,def_dir,cap-1); ckpt_dir[cap-1]='\0';
    for(int i=1;i<argc;++i){
        if(strncmp(argv[i],"--steps=",8)==0)*steps=atoi(argv[i]+8);
        else if(strncmp(argv[i],"--eval-interval=",16)==0)*eval_interval=atoi(argv[i]+16);
        else if(strncmp(argv[i],"--eval-batches=",15)==0)*eval_batches=atoi(argv[i]+15);
        else if(strncmp(argv[i],"--warmup-steps=",15)==0)*warmup_steps=atoi(argv[i]+15);
        else if(strncmp(argv[i],"--patience=",11)==0)*patience=atoi(argv[i]+11);
        else if(strncmp(argv[i],"--ckpt-dir=",11)==0){strncpy(ckpt_dir,argv[i]+11,cap-1);ckpt_dir[cap-1]='\0';}
	else if(strcmp(argv[i],"--resume")==0)*resume=1;
    }
}

//inference helpers
//signature/order matches how main() calls it (dir, file, prompt, ...).
static void get_infer_flags(int argc, char** argv,
                            int* chat,
                            char* ckpt_dir,  size_t ckpt_dir_cap,
                            char* ckpt_file, size_t ckpt_file_cap,
                            char* prompt,    size_t prompt_cap,
                            int* max_new, float* temperature, int* top_k, unsigned int* seed, int* use_best) {
    *chat=0; *use_best=1; *max_new=128; *temperature=0.9f; *top_k=50; *seed=0;
    ckpt_file[0]='\0';
    strncpy(ckpt_dir,"ckpts",ckpt_dir_cap-1); ckpt_dir[ckpt_dir_cap-1]='\0';
    if (prompt_cap) prompt[0]='\0';
    for(int i=1;i<argc;++i){
	if(strcmp(argv[i],"--infer")==0 || strcmp(argv[i],"--chat")==0) *chat=1;
        else if(strncmp(argv[i],"--ckpt-file=",12)==0){ strncpy(ckpt_file,argv[i]+12,ckpt_file_cap-1); ckpt_file[ckpt_file_cap-1]='\0'; }
        else if(strncmp(argv[i],"--ckpt-dir=",11)==0){ strncpy(ckpt_dir,argv[i]+11,ckpt_dir_cap-1); ckpt_dir[ckpt_dir_cap-1]='\0'; }
        else if(strcmp(argv[i],"--latest")==0) *use_best=0;
        else if(strcmp(argv[i],"--best")==0) *use_best=1;
        else if(strncmp(argv[i],"--max-new=",10)==0) *max_new=atoi(argv[i]+10);
        else if(strncmp(argv[i],"--temperature=",14)==0) *temperature=(float)atof(argv[i]+14);
        else if(strncmp(argv[i],"--top-k=",8)==0) *top_k=atoi(argv[i]+8);
        else if(strncmp(argv[i],"--seed=",7)==0) *seed=(unsigned int)strtoul(argv[i]+7,NULL,10);
        else if(strncmp(argv[i],"--prompt=",9)==0){ strncpy(prompt, argv[i]+9, prompt_cap-1); prompt[prompt_cap-1]='\0'; }
    }
}

//simple version of get_chat_flags
void get_chat_flags(int argc, char** argv,
                    int* chat,
                    char* ckpt_file, size_t ckpt_file_cap,
                    char* ckpt_dir,  size_t ckpt_dir_cap,
                    int* use_best, int* max_new, float* temperature,
                    int* top_k, unsigned int* seed)
{
    *chat = 0;
    *use_best = 1;
    *max_new = 128;
    *temperature = 0.9f;
    *top_k = 50;
    *seed = 0;
    ckpt_file[0] = '\0';
    strncpy(ckpt_dir, "ckpts", ckpt_dir_cap - 1);
    ckpt_dir[ckpt_dir_cap - 1] = '\0';

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--chat") == 0) *chat = 1;
        else if (strncmp(argv[i], "--ckpt-file=", 12) == 0)
            strncpy(ckpt_file, argv[i] + 12, ckpt_file_cap - 1);
        else if (strncmp(argv[i], "--ckpt-dir=", 11) == 0)
            strncpy(ckpt_dir, argv[i] + 11, ckpt_dir_cap - 1);
        else if (strcmp(argv[i], "--latest") == 0) *use_best = 0;
        else if (strcmp(argv[i], "--best") == 0) *use_best = 1;
        else if (strncmp(argv[i], "--max-new=", 10) == 0) *max_new = atoi(argv[i] + 10);
        else if (strncmp(argv[i], "--temperature=", 14) == 0) *temperature = atof(argv[i] + 14);
        else if (strncmp(argv[i], "--top-k=", 8) == 0) *top_k = atoi(argv[i] + 8);
        else if (strncmp(argv[i], "--seed=", 7) == 0) *seed = strtoul(argv[i] + 7, NULL, 10);
    }
}


static int load_params_from_file(const char* path, struct Model* M){
    FILE* f = fopen(path, "rb");
    if (!f){ perror("[infer] open params"); return -1; }
    float* hbuf = (float*)malloc(M->n_params * sizeof(float));
    if (!hbuf){ fprintf(stderr, "[infer] host malloc failed\n"); fclose(f); return -1; }
    size_t n = fread(hbuf, sizeof(float), M->n_params, f);
    fclose(f);
    if (n != M->n_params){ fprintf(stderr, "[infer] params size mismatch in %s\n", path); free(hbuf); return -1; }
    cudaError_t e = cudaMemcpy(M->flat_params, hbuf, M->n_params*sizeof(float), cudaMemcpyHostToDevice);
    free(hbuf);
    if (e != cudaSuccess){ fprintf(stderr, "[infer] H2D params failed: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;
}

static inline double rng_uniform01(uint64_t* state){
    *state = (*state * 6364136223846793005ULL) + 1ULL;
    //53-bit mantissa to double in [0,1)
    return (double)((*state >> 11) & ((1ULL<<53)-1)) * (1.0/9007199254740992.0);
}

static int sample_topk(const float* logits, int V, int top_k, float temp, uint64_t* rng){
    if (top_k <= 0 || top_k > V) top_k = V;
    int idxs[256]; float vals[256]; // V=256 for your setup
    for (int i=0;i<top_k;++i){ idxs[i] = -1; vals[i] = -1e30f; }
    //running top-k
    for (int j=0;j<V;++j){
        float v = logits[j] / (temp>0.f?temp:1.f);
        int m = -1; float mv = 1e30f;
        for (int i=0;i<top_k;++i){ if (vals[i] < mv){ mv = vals[i]; m = i; } }
        if (v > mv){ vals[m]=v; idxs[m]=j; }
    }
    //argmax if temp==0
    if (temp <= 0.f){
        int best = 0; for (int i=1;i<top_k;++i) if (vals[i] > vals[best]) best=i;
        return idxs[best];
    }
    //softmax over top-k and sample
    float maxv = vals[0]; for (int i=1;i<top_k;++i) if (vals[i]>maxv) maxv=vals[i];
    double sum = 0.0; double ps[256];
    for (int i=0;i<top_k;++i){ ps[i] = exp((double)(vals[i]-maxv)); sum += ps[i]; }
    double u = rng_uniform01(rng) * sum;
    for (int i=0;i<top_k;++i){ if (u <= ps[i]) return idxs[i]; u -= ps[i]; }
    return idxs[top_k-1];
}

static void run_inference(struct Config cfg_file,
                          const char* ckpt_dir,
                          const char* ckpt_file_opt,
                          int use_best,
                          const char* prompt_str,
                          int max_new, float temp, int top_k, unsigned seed,
                          int rank, int world)
{
    (void)world; // silence unused parameter warning

    //build an inference config: B=1, keep T from file
    struct Config cfg = cfg_file;
    cfg.batch_size = 1;

    // init model
    struct Model M; model_init(&M, &cfg);
    if (rank==0){
        printf("[infer] Using seq_len=%d, d_model=%d, vocab=%d, top_k=%d, temp=%.3f\n",
               cfg.seq_len, cfg.d_model, cfg.vocab_size, top_k, temp);
        model_log_summary(&M, &cfg);
    }

    //load checkpoint params
    char params_path[1024];
    if (ckpt_file_opt && ckpt_file_opt[0]){
        strncpy(params_path, ckpt_file_opt, sizeof(params_path)-1); params_path[sizeof(params_path)-1]='\0';
    } else {
        snprintf(params_path, sizeof(params_path), "%s/%s.params.bin", ckpt_dir,
                 use_best ? "best" : "latest");
    }
    if (rank==0) printf("[infer] loading params: %s\n", params_path);
    if (load_params_from_file(params_path, &M) != 0){
        if (rank==0) fprintf(stderr, "[infer] failed to load params; aborting.\n");
        model_free(&M);
        return;
    }

    const int T = cfg.seq_len;
    const int V = cfg.vocab_size;
    uint8_t* x = (uint8_t*)malloc((size_t)T);
    uint8_t* y = (uint8_t*)malloc((size_t)T); // dummy for forward_loss
    memset(x, 0, (size_t)T);
    memset(y, 0, (size_t)T);

    //encode prompt as raw bytes (dataset is byte-level)
    size_t L = strlen(prompt_str);
    if (L > (size_t)T){
        // left-truncate to fit context window
        prompt_str += (L - (size_t)T);
        L = (size_t)T;
    }
    for (size_t i=0;i<L;++i) x[i] = (uint8_t)prompt_str[i];

    //generation loop
    uint64_t rng = ((uint64_t)seed << 1) | 1ULL;
    if (rank==0){
        fwrite(prompt_str, 1, L, stdout); fflush(stdout);
    }
    int cur = (int)L;
    for (int step = 0; step < max_new; ++step){
        //compute full T; only read row (cur-1).
        //if cur==0 (empty prompt), we read row 0 after seeding x[0].
        if (cur == 0){ x[0] = (uint8_t)' '; cur = 1; }

        nano2_forward_loss(&M, x, y); // fills M.buf.logits [T,V]
        cudaDeviceSynchronize();

        //read logits for position (cur-1)
        int row = cur - 1;
        if (row < 0) { row = 0; }
        if (row >= T) { row = T-1; }
        float hlogits[256];
        cudaMemcpy(hlogits, M.buf.logits + (size_t)row * (size_t)V, (size_t)V*sizeof(float), cudaMemcpyDeviceToHost);

        //sample next token
        int tok = sample_topk(hlogits, V, top_k, temp, &rng);
        if (cur < T) x[cur] = (uint8_t)tok; // place next token if still within window
        //stream the character
        if (rank==0){ fputc((tok >= 32 && tok <= 126) ? (char)tok : ' ', stdout); fflush(stdout); }

        //advance context; stay within window T by discarding oldest if needed
        if (cur+1 <= T){
            ++cur;
        } else {
            // shift left by 1 to make room
            memmove(x, x+1, (size_t)(T-1));
            x[T-1] = (uint8_t)tok;
            cur = T;
        }
    }
    if (rank==0) fputc('\n', stdout);

    free(x); free(y);
    model_free(&M);
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank=0, world=1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int local_rank=0, local_size=1; MPI_Comm_rank(local, &local_rank); MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);

    //CUDA device selection (one GPU per local rank)
    int dev_count=0; cudaGetDeviceCount(&dev_count);
    int dev = (dev_count > 0) ? (local_rank % dev_count) : 0;
    cudaSetDevice(dev);

    if (rank==0){
        printf("nano2: world=%d local_rank=%d/%d device=%d\n", world, local_rank, local_size, dev);
    }

    //parse config
    char config_path[512]; get_config_path(argc, argv, config_path, sizeof(config_path));
    struct Config cfg_file; config_from_file(config_path, &cfg_file);
    if (rank==0){
        printf("config: %s\n", config_path); config_log(&cfg_file);
    }

    //chat mode
    int chat = 0;
    int chat_use_best = 1;
    int chat_max_new = 128;
    int chat_top_k = 50;
    float chat_temp = 0.9f;
    unsigned int chat_seed = 0;
    char chat_ckpt_file[512];
    char chat_ckpt_dir[512];
    get_chat_flags(argc, argv,
                   &chat,
                   chat_ckpt_file, sizeof(chat_ckpt_file),
                   chat_ckpt_dir,  sizeof(chat_ckpt_dir),
                   &chat_use_best, &chat_max_new, &chat_temp, &chat_top_k, &chat_seed);
    if (chat){
        int rc = run_chat(chat_ckpt_file, chat_ckpt_dir, chat_use_best, cfg_file, chat_max_new, chat_top_k, chat_temp, chat_seed);
        MPI_Finalize(); return rc;
    }

    //inference
    int infer=0, max_new=200, top_k=50, use_best=1; float temp=1.0f; unsigned seed=42;
    char ckpt_dir[512], ckpt_file[1024], prompt[2048];
    get_infer_flags(argc, argv, &infer, ckpt_dir, sizeof(ckpt_dir), ckpt_file, sizeof(ckpt_file),
                    prompt, sizeof(prompt), &max_new, &temp, &top_k, &seed, &use_best);
    if (infer){
        if (rank==0){
            printf("[infer] prompt=\"%s\" | max_new=%d | top_k=%d | temp=%.3f | seed=%u\n",
                   prompt, max_new, top_k, temp, seed);
        }
        run_inference(cfg_file, ckpt_dir, ckpt_file[0]?ckpt_file:NULL, use_best,
                      prompt, max_new, temp, top_k, seed, rank, world);
        MPI_Finalize();
        return 0;
    }

    //Training
    //load datasets
    struct DataSet train_ds, val_ds; dataset_load(cfg_file.train_path, &train_ds); dataset_load(cfg_file.val_path, &val_ds);
    if (rank==0){ dataset_log(&train_ds, "train"); dataset_log(&val_ds, "val"); }

    //initialize model (training uses file batch size)
    struct Model M; model_init(&M, &cfg_file);
    if (rank==0) model_log_summary(&M, &cfg_file);

    //make one batch from train set
    const int B = cfg_file.batch_size; const int T = cfg_file.seq_len; const int BT = B * T;
    uint8_t* x = (uint8_t*)malloc((size_t)BT);
    uint8_t* y = (uint8_t*)malloc((size_t)BT);
    dataset_next_batch(&train_ds, B, T, x, y);

    //preview
    if (rank==0){
        int preview = (T<16) ? T : 16;
        printf("batch preview x[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) { printf("%u ", (unsigned)x[t]); }
        printf("\n");
        printf("batch preview y[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) { printf("%u ", (unsigned)y[t]); }
        printf("\n");
    }

    //GPU memory snapshot
    size_t mem_free=0, mem_total=0; cudaMemGetInfo(&mem_free, &mem_total);
    if(rank==0){
        double used_mib = (double)(mem_total - mem_free)/(1024.0*1024.0);
        double total_mib = (double)mem_total/(1024.0*1024.0);
        printf("[gpu] memory used: %.2f MiB / %.2f MiB\n", used_mib, total_mib);
    }

    //run forward once, measure time
    cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    float loss = nano2_forward_loss(&M, x, y);
    cudaDeviceSynchronize();
    cudaEventRecord(ev1, 0); cudaEventSynchronize(ev1);
    float ms=0.0f; cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    if (rank==0){
        double toks = (double)BT;
        double toks_per_s = toks / (ms * 1e-3);
        printf("forward mean loss: %.6f (expect ~ ln(256)=5.545)\n", loss);
        printf("step time: %.3f ms | tokens/step: %d | tokens/sec: %.0f\n", ms, BT, toks_per_s);
    }

    //training with eval, lr schedule, checkpointing
    int steps, eval_interval, eval_batches, warmup_steps, patience, resume;
    char ckpt_dir_tr[512];
    get_train_flags(argc, argv, &steps, &eval_interval, &eval_batches, &warmup_steps, &patience, ckpt_dir_tr, sizeof(ckpt_dir_tr), &resume);

    float base_lr = cfg_file.lr;
    float best_val = 1e30f;
    int start_step = 0;
    int bad = 0;

    if (resume){
      float last_v = 0.f;
      if (load_checkpoint_latest(ckpt_dir_tr, &M, &cfg_file, &start_step, &last_v) == 0){
        if (rank==0) printf("[ckpt] resumed from step %d (val_loss=%.6f, lr=%.6g)\n", start_step, last_v, cfg_file.lr);
        best_val = last_v > 0 ? last_v : best_val;
      } else if (rank==0){
        printf("[ckpt] no latest checkpoint in %s, starting fresh\n", ckpt_dir_tr);
      }
    }

    //scratch batch reused
    for (int s = start_step; s < steps; ++s){
      cfg_file.lr = lr_cosine(base_lr, s, warmup_steps, steps);
      dataset_next_batch(&train_ds, B, T, x, y);
      float tr_loss = nano2_train_step(&M, x, y, &cfg_file, world, rank);
      if (rank == 0){
        printf("step %d | lr=%.6g | train loss: %.6f\n", s, cfg_file.lr, tr_loss);
      }

      int do_eval = ((s+1) % eval_interval == 0) || (s+1 == steps);
      if (rank == 0 && do_eval){
        size_t save = val_ds.cursor; dataset_reset(&val_ds, 0);
        double acc = 0.0; for (int i=0;i<eval_batches;++i){ dataset_next_batch(&val_ds, B, T, x, y); acc += (double)nano2_forward_loss(&M, x, y); }
        dataset_reset(&val_ds, save);
        float vloss = (float)(acc / (double)eval_batches);
        float ppl = expf(vloss);
        int is_best = (vloss < best_val);
        if (is_best){ best_val = vloss; bad = 0; } else { ++bad; }

        printf("[val] step %d | loss: %.6f | ppl: %.3f %s\n",
               s+1, vloss, ppl, is_best ? "(best)" : "");

        if (save_checkpoint(ckpt_dir_tr, &M, &cfg_file, s+1, vloss, is_best) != 0){
            fprintf(stderr, "[ckpt] save failed at step %d\n", s+1);
        }
        if (bad >= patience){
            printf("[early-stop] no improvement for %d evals; stopping at step %d\n", patience, s+1);
            break;
        }
      }
    }

    free(x); free(y);
    model_free(&M);
    dataset_free(&train_ds); dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}

