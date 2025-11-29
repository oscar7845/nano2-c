//TODO: append byte
//sample top k only
//
#include "chat.h"
#include "checkpoint.h"
#include "nano2_model.h"

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

float nano2_forward_loss(struct Model* M,const uint8_t* h_x,const uint8_t* h_y);

static void* xrealloc(void* p,size_t n){
    void* q=realloc(p,n);
    if(!q){ fprintf(stderr,"OOM realloc(%zu)\n",n); exit(1); }
    return q;
}

static void append_bytes(uint8_t** buf,size_t* len,size_t* cap,const uint8_t* src,size_t n){
    if(*len+n > *cap){
        size_t newcap=*cap?*cap:4096;
        while(newcap < *len+n) newcap*=2;
        *buf=(uint8_t*)xrealloc(*buf,newcap);
        *cap=newcap;
    }
    memcpy(*buf+*len,src,n);
    *len+=n;
}

static int load_params_into_model(const char* path,struct Model* M){
    FILE* f=fopen(path,"rb");
    if(!f){ perror("[chat open params]"); return -1; }

    float* hbuf=(float*)malloc(M->n_params*sizeof(float));
    if(!hbuf){ fprintf(stderr,"malloc fail\n"); fclose(f); return -1; }

    size_t got=fread(hbuf,sizeof(float),M->n_params,f);
    fclose(f);
    if(got!=M->n_params){ free(hbuf); return -1; }

    cudaError_t e=cudaMemcpy(M->flat_params,hbuf,M->n_params*sizeof(float),cudaMemcpyHostToDevice);
    free(hbuf);
    if(e!=cudaSuccess) return -1;
    return 0;
}

static int sample_topk(const float* logits,int V,int top_k,float temperature,unsigned int* rng){
    if(top_k<=0 || top_k>V) top_k=V;
    float scratch[1024];
    for(int i=0;i<V;++i) scratch[i]=logits[i];

    int idx[1024];
    for(int i=0;i<top_k;++i){
        int bi=0;
        for(int j=1;j<V;++j) if(scratch[j] > scratch[bi]) bi=j;
        idx[i]=bi;
        scratch[bi]=-INFINITY;
    }

    float maxlog=-INFINITY;
    for(int i=0;i<top_k;++i){
        float z=logits[idx[i]];
        if(z>maxlog) maxlog=z;
    }

    float invT=1.0f/fmaxf(1e-8f,temperature);
    float sum=0.f;
    float probs[1024];
    for(int i=0;i<top_k;++i){
        float e=expf((logits[idx[i]]-maxlog)*invT);
        probs[i]=e;
        sum+=e;
    }

    if(sum<=0.f) return idx[0];
    for(int i=0;i<top_k;++i) probs[i]/=sum;

    float r=frand01(rng);
    float c=0.f;
    for(int i=0;i<top_k;++i){
        c+=probs[i];
        if(r<=c) return idx[i];
    }
    return idx[top_k-1];
}

int run_chat(const char* ckpt_file,const char* ckpt_dir,int use_best,struct Config base_cfg,int max_new_tokens,int top_k,float temperature,unsigned int seed){
    struct Config cfg=base_cfg;
    cfg.batch_size=1;

    struct Model M;
    model_init(&M,&cfg);

    int loaded=-1;
    if(ckpt_file && ckpt_file[0]){
        loaded=load_params_into_model(ckpt_file,&M);
    } else if(ckpt_dir && ckpt_dir[0]){
        if(use_best){
            char p[1024];
            snprintf(p,sizeof(p),"%s/best.params.bin",ckpt_dir);
            loaded=load_params_into_model(p,&M);
        } else {
            float last=0.f; int step=0;
            loaded=load_checkpoint_latest(ckpt_dir,&M,&cfg,&step,&last);
            if(loaded==0){
                fprintf(stdout,"[ckpt] loaded latest step=%d\n",step);
            }
        }
    }
    if(loaded!=0){
        fprintf(stderr,"load fail\n");
        model_free(&M);
        return 1;
    }

    uint8_t* conv=NULL;
    size_t clen=0,ccap=0;
    append_cstr(&conv,&clen,&ccap,"You are Assistant.\n");

    setvbuf(stdout,NULL,_IONBF,0);

    char line[8192];
    unsigned int rng=seed?seed:(unsigned int)time(NULL);

    const int T=M.T;
    const int V=M.V;
    uint8_t* x=(uint8_t*)malloc(T);
    uint8_t* y=(uint8_t*)malloc(T);
    float* last_logits=(float*)malloc(V*sizeof(float));
}

static void fetch_last_logits_row(struct Model* M,int row,float* host_logits_out){
    size_t off=(size_t)row*(size_t)M->V;
    cudaMemcpy(host_logits_out,M->buf.logits+off,M->V*sizeof(float),cudaMemcpyDeviceToHost);
}

