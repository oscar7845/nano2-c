//grouped gpu ops
//call gelu -> softmax-attention -> proj
//still just sanity smoke tests
//TODO:
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <mpi.h>
#include <cuda_runtime_api.h>

//same structs + decls as v1

extern void nano2_gelu_forward(const float*,float*,int,int);
extern void nano2_attention_forward(const float*,int,int,int,
                                    const float*,const float*,const float*,const float*,
                                    float*,float*,float*,float*,float*,float*,float*);

static void get_cfg(int ac,char**av,char*out,size_t cap){
  const char*def="./configs/nano2.json";
  strncpy(out,def,cap-1); out[cap-1]=0;
}

int main(int ac,char**av){

  MPI_Init(&ac,&av);
  int r=0; MPI_Comm_rank(MPI_COMM_WORLD,&r);

  MPI_Comm loc; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc);
  int lr=0; MPI_Comm_rank(loc,&lr); MPI_Comm_free(&loc);

  int dc=0; cudaGetDeviceCount(&dc);
  cudaSetDevice(dc? lr%dc : 0);

  if(r==0) printf("main v2\n");

  char path[512]; get_cfg(ac,av,path,sizeof(path));
  struct Config cfg; config_from_file(path,&cfg);

  struct DataSet tr,va;
  dataset_load(cfg.train_path,&tr);
  dataset_load(cfg.val_path,&va);

  int B=cfg.batch_size, T=cfg.seq_len, D=cfg.d_model;

  //model
  struct Model M;
  model_init(&M,&cfg);

  //setup host random junk
  size_t bytes = (size_t)B*T*D*sizeof(float);
  float* h = nano2_malloc_host(bytes);
  for(size_t i=0;i<(size_t)B*T*D;i++) h[i]=(float)((i%47)*0.02f);
  nano2_copy_host_to_device(M.buf.x, h, bytes);
  nano2_free_host(h);

  //fw
  // LN missing, but we just pretend M.buf.x_ln1 is LN(x)
  nano2_gelu_forward(M.buf.x, M.buf.x_ln1, B*T*D, 1);

  // run attention on this LN output
  nano2_attention_forward(M.buf.x_ln1, B,T,D,
                          M.p.Wq,M.p.Wk,M.p.Wv,M.p.Wo,
                          M.buf.q,M.buf.k,M.buf.v,
                          M.buf.scores,M.buf.probs,
                          M.buf.attn_out, M.buf.x_res1);

  if(r==0) printf("dummy fwd v2 ok\n");

  model_free(&M);
  dataset_free(&tr); dataset_free(&va);
  MPI_Finalize();
  return 0;
}

