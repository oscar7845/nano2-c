#include "checkpoint.h"
#include "nano2_model.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <unistd.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

//TODO:rm prints

//helpers
static int ensure_dir(const char* dir){
    struct stat st;
    if (stat(dir,&st)==0 && (st.st_mode & S_IFDIR)) return 0;
    if (MKDIR(dir)==0) return 0;
    if (errno==EEXIST) return 0;
    fprintf(stderr,"[ckpt] failed to create dir %s: %s\n",dir,strerror(errno));
    return -1;
}

static int write_atomic_bin(const char* path,const void* ptr,size_t count,size_t elem){
    char tmp[1024]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE* f=fopen(tmp,"wb"); if (!f){ perror("[ckpt fopen tmp]"); return -1; }
    size_t n=fwrite(ptr,elem,count,f);
    if (n!=count){ perror("[ckpt fwrite]"); fclose(f); remove(tmp); return -1; }
    fflush(f);
#if !defined(_WIN32)
    int fd=fileno(f); if (fd>=0) fsync(fd);
#endif
    fclose(f);
    if (rename(tmp,path)!=0){ perror("[ckpt rename]"); remove(tmp); return -1; }
    return 0;
}

static int write_text(const char* path,const char* s){
    char tmp[1024]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE* f=fopen(tmp,"wb"); if (!f){ perror("[ckpt fopen tmp txt]"); return -1; }
    size_t n=fwrite(s,1,strlen(s),f);
    if (n!=strlen(s)){ perror("[ckpt fwrite txt]"); fclose(f); remove(tmp); return -1; }
    fflush(f);
#if !defined(_WIN32)
    int fd=fileno(f); if (fd>=0) fsync(fd);
#endif
    fclose(f);
    if (rename(tmp,path)!=0){ perror("[ckpt rename txt]"); remove(tmp); return -1; }
    return 0;
}

static int file_exists(const char* path){
    FILE* f=fopen(path,"rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

//json key finder (no full parsing)
static int json_find_int(const char* buf,const char* key,int* out){
    const char* p=strstr(buf,key); if (!p) return -1;
    p=strchr(p,':'); if (!p) return -1;
    *out=(int)strtol(p+1,NULL,10);
    return 0;
}
static int json_find_float(const char* buf,const char* key,float* out){
    const char* p=strstr(buf,key); if (!p) return -1;
    p=strchr(p,':'); if (!p) return -1;
    *out=strtof(p+1,NULL);
    return 0;
}

//host staging buffers for D2H/H2D
static int d2h_copy(float* hbuf,const float* dptr,size_t n){
    cudaError_t e=cudaMemcpy(hbuf,dptr,n*sizeof(float),cudaMemcpyDeviceToHost);
    if (e!=cudaSuccess){ fprintf(stderr,"[ckpt] cudaMemcpy D2H failed: %s\n",cudaGetErrorString(e)); return -1; }
    return 0;
}
static int h2d_copy(float* dptr,const float* hbuf,size_t n){
    cudaError_t e=cudaMemcpy(dptr,hbuf,n*sizeof(float),cudaMemcpyHostToDevice);
    if (e!=cudaSuccess){ fprintf(stderr,"[ckpt] cudaMemcpy H2D failed: %s\n",cudaGetErrorString(e)); return -1; }
    return 0;
}

//public API
int save_checkpoint(const char* dir,
                    const struct Model* M,
                    const struct Config* cfg,
                    int step,
                    float val_loss,
                    int is_best)
{
    if (ensure_dir(dir)!=0) return -1;

    cudaDeviceSynchronize();

    float* h_params=(float*)malloc(M->n_params*sizeof(float));
    float* h_m=(float*)malloc(M->n_params*sizeof(float));
    float* h_v=(float*)malloc(M->n_params*sizeof(float));
    if (!h_params || !h_m || !h_v){
        fprintf(stderr,"[ckpt] host malloc failed\n");
        free(h_params); free(h_m); free(h_v);
        return -1;
    }

    if (d2h_copy(h_params,M->flat_params,M->n_params)!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    if (d2h_copy(h_m,M->opt.m,M->n_params)!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    if (d2h_copy(h_v,M->opt.v,M->n_params)!=0){ free(h_params); free(h_m); free(h_v); return -1; }

    char p_params[1024],p_m[1024],p_v[1024],p_meta[1024];
    char l_params[1024],l_m[1024],l_v[1024],l_meta[1024];
    char b_params[1024],b_m[1024],b_v[1024],b_meta[1024];

    snprintf(p_params,sizeof(p_params),"%s/step_%06d.params.bin",dir,step);
    snprintf(p_m,sizeof(p_m),"%s/step_%06d.opt_m.bin",dir,step);
    snprintf(p_v,sizeof(p_v),"%s/step_%06d.opt_v.bin",dir,step);
    snprintf(p_meta,sizeof(p_meta),"%s/step_%06d.meta.json",dir,step);

    snprintf(l_params,sizeof(l_params),"%s/latest.params.bin",dir);
    snprintf(l_m,sizeof(l_m),"%s/latest.opt_m.bin",dir);
    snprintf(l_v,sizeof(l_v),"%s/latest.opt_v.bin",dir);
    snprintf(l_meta,sizeof(l_meta),"%s/latest.meta.json",dir);

    snprintf(b_params,sizeof(b_params),"%s/best.params.bin",dir);
    snprintf(b_m,sizeof(b_m),"%s/best.opt_m.bin",dir);
    snprintf(b_v,sizeof(b_v),"%s/best.opt_v.bin",dir);
    snprintf(b_meta,sizeof(b_meta),"%s/best.meta.json",dir);

    if (write_atomic_bin(p_params,h_params,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    if (write_atomic_bin(p_m,h_m,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    if (write_atomic_bin(p_v,h_v,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }

    if (write_atomic_bin(l_params,h_params,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    if (write_atomic_bin(l_m,h_m,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    if (write_atomic_bin(l_v,h_v,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }

    char meta[2048];
    snprintf(meta,sizeof(meta),
        "{\n"
        "  \"step\": %d,\n"
        "  \"val_loss\": %.6f,\n"
        "  \"n_params\": %zu,\n"
        "  \"lr\": %.8f,\n"
        "  \"seq_len\": %d,\n"
        "  \"batch_size\": %d,\n"
        "  \"d_model\": %d,\n"
        "  \"ffn_mult\": %d,\n"
        "  \"vocab_size\": %d\n"
        "}\n",
        step,val_loss,M->n_params,cfg->lr,M->T,M->B,M->D,M->F/M->D,M->V);

    if (write_text(p_meta,meta)!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    if (write_text(l_meta,meta)!=0){ free(h_params); free(h_m); free(h_v); return -1; }

    if (is_best){
        if (write_atomic_bin(b_params,h_params,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }
        if (write_atomic_bin(b_m,h_m,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }
        if (write_atomic_bin(b_v,h_v,M->n_params,sizeof(float))!=0){ free(h_params); free(h_m); free(h_v); return -1; }
        if (write_text(b_meta,meta)!=0){ free(h_params); free(h_m); free(h_v); return -1; }
    }

    free(h_params); free(h_m); free(h_v);
    return 0;
}

int load_checkpoint_latest(const char* dir,
                           struct Model* M,
                           struct Config* cfg,
                           int* out_step,
                           float* out_val_loss)
{
    char l_params[1024],l_m[1024],l_v[1024],l_meta[1024];
    snprintf(l_params,sizeof(l_params),"%s/latest.params.bin",dir);
    snprintf(l_m,sizeof(l_m),"%s/latest.opt_m.bin",dir);
    snprintf(l_v,sizeof(l_v),"%s/latest.opt_v.bin",dir);
    snprintf(l_meta,sizeof(l_meta),"%s/latest.meta.json",dir);

    if (!file_exists(l_params)) return -1;

    float* h_params=(float*)malloc(M->n_params*sizeof(float));
    float* h_m=(float*)malloc(M->n_params*sizeof(float));
    float* h_v=(float*)malloc(M->n_params*sizeof(float));
    if (!h_params || !h_m || !h_v){
        fprintf(stderr,"[ckpt] host malloc failed (load)\n");
        free(h_params); free(h_m); free(h_v);
        return -1;
    }

    FILE* f=fopen(l_params,"rb");
    if (!f){ perror("[ckpt open latest params]"); free(h_params); free(h_m); free(h_v); return -1; }
    size_t n=fread(h_params,sizeof(float),M->n_params,f);
    fclose(f);
    if (n!=M->n_params){ fprintf(stderr,"[ckpt] params size mismatch\n"); free(h_params); free(h_m); free(h_v); return -1; }

    if (h2d_copy(M->flat_params,h_params,M->n_params)!=0){ free(h_params); free(h_m); free(h_v); return -1; }

    if (file_exists(l_m)){
        FILE* fm=fopen(l_m,"rb");
        if (fm){
            fread(h_m,sizeof(float),M->n_params,fm);
            fclose(fm);
            if (h2d_copy(M->opt.m,h_m,M->n_params)!=0){ free(h_params); free(h_m); free(h_v); return -1; }
        }
    }
    if (file_exists(l_v)){
        FILE* fv=fopen(l_v,"rb");
        if (fv){
            fread(h_v,sizeof(float),M->n_params,fv);
            fclose(fv);
            if (h2d_copy(M->opt.v,h_v,M->n_params)!=0){ free(h_params); free(h_m); free(h_v); return -1; }
        }
    }

    if (file_exists(l_meta)){
        FILE* mt=fopen(l_meta,"rb");
        if (mt){
            char buf[2048]={0};
            fread(buf,1,sizeof(buf)-1,mt);
            fclose(mt);
            int step; float vloss; float lr=cfg->lr;
            if (json_find_int(buf,"\"step\"",&step)==0 && out_step) *out_step=step;
            if (json_find_float(buf,"\"val_loss\"",&vloss)==0 && out_val_loss) *out_val_loss=vloss;
            if (json_find_float(buf,"\"lr\"",&lr)==0) cfg->lr=lr;
        }
    }

    free(h_params); free(h_m); free(h_v);
    return 0;
}

