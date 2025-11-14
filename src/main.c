//load cfg/data, init model, try calling tiny gelu+attn stubs
//(still no real training or forward)
//
//TODO:
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <mpi.h>
#include <cuda_runtime_api.h>

//config + dataset + tensor helpers (same as before)
struct Config{
  char train_path[512];
  char val_path[512];
  int seq_len, batch_size, vocab_size, d_model, ffn_mult;
  double lr, weight_decay, clip_grad_norm;
  int seed, top_k;
};
int config_from_file(const char*, struct Config*);
void config_log(const struct Config*);
struct DataSet{
  uint8_t* data; size_t n; size_t cursor; char path[512];
};
int dataset_load(const char*, struct DataSet*);
void dataset_free(struct DataSet*);
void dataset_next_batch(struct DataSet*, int,int,uint8_t*,uint8_t*);
void dataset_log(const struct DataSet*, const char*);

void* nano2_malloc_host(size_t); void nano2_free_host(void*);
void* nano2_malloc_device(size_t); void nano2_free_device(void*);
void  nano2_copy_host_to_device(void*,const void*,size_t);

//model
struct Model;
void model_init(struct Model*, const struct Config*);
void model_free(struct Model*);
void model_log_summary(const struct Model*, const struct Config*);

//cuda ops (first simple versions)
extern void nano2_gelu_forward(const float* x, float* y, int n, int approx);
extern void nano2_attention_forward(const float* x_ln,
                                    int B,int T,int D,
                                    const float* Wq,const float* Wk,const float* Wv,const float* Wo,
                                    float* q,float* k,float* v,
                                    float* scores,float* probs,
                                    float* ctx,float* out);

//config path
static void get_cfg(int ac,char**av,char*out,size_t cap){
  const char*def="./configs/nano2.json";
  strncpy(out,def,cap-1); out[cap-1]=0;
  for(int i=1;i<ac;i++){
    if(!strncmp(av[i],"--config=",9)){ strncpy(out,av[i]+9,cap-1); out[cap-1]=0; }
  }
}

int main(int ac,char**av){

  //mpi + gpu
  MPI_Init(&ac,&av);
  int r=0,w=1; MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&w);
  MPI_Comm loc; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc);
  int lr=0,ls=1; MPI_Comm_rank(loc,&lr); MPI_Comm_size(loc,&ls); MPI_Comm_free(&loc);

  int dc=0; cudaGetDeviceCount(&dc);
  int dev = (dc? lr % dc : 0);
  cudaSetDevice(dev);

  if(r==0) printf("main v1: world=%d local=%d/%d gpu=%d\n",w,lr,ls,dev);

  //config
  char path[512]; get_cfg(ac,av,path,sizeof(path));
  struct Config cfg; config_from_file(path,&cfg);
  if(r==0){ printf("config: %s\n",path); config_log(&cfg); }

  //datasets
  struct DataSet tr,va;
  dataset_load(cfg.train_path,&tr);
  dataset_load(cfg.val_path,&va);
  if(r==0){ dataset_log(&tr,"train"); dataset_log(&va,"val"); }

  //mini batch preview
  int B=cfg.batch_size, T=cfg.seq_len, D=cfg.d_model;
  uint8_t*bx=malloc((size_t)B*T);
  uint8_t*by=malloc((size_t)B*T);
  dataset_next_batch(&tr,B,T,bx,by);
  if(r==0){
    printf("batch preview x[0]: ");
    for(int i=0;i<(T<16?T:16);i++) printf("%u ",bx[i]);
    printf("\n");
  }
  free(bx); free(by);

  //model
  struct Model M;
  if(r==0) printf("model init...\n");
  model_init(&M,&cfg);
  if(r==0) model_log_summary(&M,&cfg);

  //simple gpu test: allocate device buffer & run gelu forward once
  size_t bytes = (size_t)B*T*D*sizeof(float);
  float* htmp = (float*)nano2_malloc_host(bytes);
  for(size_t i=0;i<(size_t)B*T*D;i++) htmp[i] = (float)((i%91)*0.01f);

  nano2_copy_host_to_device(M.buf.x, htmp, bytes);

  //call gelu on M.buf.x -> M.buf.x_ln1 (just as dummy)
  nano2_gelu_forward(M.buf.x, M.buf.x_ln1, (int)(B*T*D), 1);

  if(r==0) printf("ran gelu test kernel\n");

  nano2_free_host(htmp);

  //attention sanity test (still dumb)
  //not real LN or block structure, just call it on garbage
  nano2_attention_forward(M.buf.x_ln1, B,T,D,
                          M.p.Wq, M.p.Wk, M.p.Wv, M.p.Wo,
                          M.buf.q,M.buf.k,M.buf.v,
                          M.buf.scores,M.buf.probs,
                          M.buf.attn_out, M.buf.x);

  if(r==0) printf("ran attention test kernel\n");

  //cleanup
  model_free(&M);
  dataset_free(&tr);
  dataset_free(&va);
  MPI_Finalize();
  return 0;
}
