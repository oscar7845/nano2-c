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

//helpers
static void* xrealloc(void* p,size_t n){
    void* q=realloc(p,n);
    if(!q){ fprintf(stderr,"OOM realloc(%zu)\n",n); exit(1); }
    return q;
}

static void append_bytes(uint8_t** buf,size_t* len,size_t* cap,const uint8_t* src,size_t n){
    if(*len+n>*cap){
        size_t newcap=*cap?*cap:4096;
        while(newcap<*len+n) newcap*=2;
        *buf=(uint8_t*)xrealloc(*buf,newcap);
        *cap=newcap;
    }
    memcpy(*buf+*len,src,n);
    *len+=n;
}

static void append_cstr(uint8_t** buf,size_t* len,size_t* cap,const char* s){
    append_bytes(buf,len,cap,(const uint8_t*)s,strlen(s));
}

static int readable_ascii(uint8_t b){
    return (b=='\n') || (b=='\t') || (b>=32 && b<127);
}

static float frand01(unsigned int* state){
    unsigned int x=*state;
    x^=x<<13; x^=x>>17; x^=x<<5;
    *state=x;
    return ((x>>8)+1)/16777217.0f;
}

static int load_params_into_model(const char* path,struct Model* M){
    FILE* f=fopen(path,"rb");
    if(!f){ perror("[chat] open params"); return -1; }
    float* hbuf=(float*)malloc((size_t)M->n_params*sizeof(float));
    if(!hbuf){ fprintf(stderr,"[chat] host malloc failed\n"); fclose(f); return -1; }

    size_t got=fread(hbuf,sizeof(float),(size_t)M->n_params,f);
    fclose(f);
    if(got!=(size_t)M->n_params){
        fprintf(stderr,"[chat] ckpt size mismatch: expected %zu floats, got %zu\n",
                (size_t)M->n_params,got);
        free(hbuf);
        return -1;
    }
    cudaError_t e=cudaMemcpy(M->flat_params,hbuf,
                             (size_t)M->n_params*sizeof(float),
                             cudaMemcpyHostToDevice);
    free(hbuf);
    if(e!=cudaSuccess){
        fprintf(stderr,"[chat] H2D params failed: %s\n",cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

//sampling (cpu)
static int sample_topk(const float* logits,int V,int top_k,float temperature,unsigned int* rng){
    if(top_k<=0 || top_k>V) top_k=V;

    float scratch[1024];
    if(V>(int)(sizeof(scratch)/sizeof(scratch[0]))){
        fprintf(stderr,"V too large for scratch (%d)\n",V);
        exit(1);
    }
    for(int i=0;i<V;++i) scratch[i]=logits[i];

    int idx[1024];
    for(int i=0;i<top_k;++i){
        int bi=0;
        for(int j=1;j<V;++j){
            if(scratch[j]>scratch[bi]) bi=j;
        }
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
        float z=(logits[idx[i]]-maxlog)*invT;
        float e=expf(z);
        probs[i]=e;
        sum+=e;
    }
    if(sum<=0.f){
        return idx[0];
    }
    for(int i=0;i<top_k;++i) probs[i]/=sum;

    float r=frand01(rng);
    float c=0.f;
    for(int i=0;i<top_k;++i){
        c+=probs[i];
        if(r<=c) return idx[i];
    }
    return idx[top_k-1];
}

//logits fetch
static void fetch_last_logits_row(struct Model* M,int row,float* host_logits_out){
    size_t off=(size_t)row*(size_t)M->V;
    cudaMemcpy(host_logits_out,M->buf.logits+off,(size_t)M->V*sizeof(float),cudaMemcpyDeviceToHost);
}
