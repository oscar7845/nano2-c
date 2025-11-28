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

static int ensure_dir(const char* dir){
    struct stat st;
    if (stat(dir,&st)==0 && (st.st_mode & S_IFDIR)) return 0;
    if (MKDIR(dir)==0) return 0;
    if (errno==EEXIST) return 0;
    fprintf(stderr,"[ckpt] failed to create dir %s: %s\n",dir,strerror(errno));
    return -1;
}

static int write_atomic_bin(const char* path,const void* ptr,size_t count,size_t elem){
    char tmp[1024];
    snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE* f=fopen(tmp,"wb");
    if (!f){ perror("[ckpt fopen tmp]"); return -1; }

    size_t n=fwrite(ptr,elem,count,f);
    if (n!=count){
        perror("[ckpt fwrite]");
        fclose(f);
        remove(tmp);
        return -1;
    }

    fflush(f);
#if !defined(_WIN32)
    int fd=fileno(f);
    if (fd>=0) fsync(fd);
#endif

    fclose(f);
    if (rename(tmp,path)!=0){
        perror("[ckpt rename]");
        remove(tmp);
        return -1;
    }
    return 0;
}

static int write_text(const char* path,const char* s){
    char tmp[1024];
    snprintf(tmp,sizeof(tmp),"%s.tmp",path);

    FILE* f=fopen(tmp,"wb");
    if (!f){ perror("[ckpt fopen tmp txt]"); return -1; }

    size_t n=fwrite(s,1,strlen(s),f);
    if (n!=strlen(s)){
        perror("[ckpt fwrite txt]");
        fclose(f);
        remove(tmp);
        return -1;
    }

    fflush(f);
#if !defined(_WIN32)
    int fd=fileno(f);
    if (fd>=0) fsync(fd);
#endif

    fclose(f);
    if (rename(tmp,path)!=0){
        perror("[ckpt rename txt]");
        remove(tmp);
        return -1;
    }
    return 0;
}

static int json_find_int(const char* buf,const char* key,int* out){
    const char* p=strstr(buf,key);
    if (!p) return -1;
    p=strchr(p,':');
    if (!p) return -1;
    *out=(int)strtol(p+1,NULL,10);
    return 0;
}

static int json_find_float(const char* buf,const char* key,float* out){
    const char* p=strstr(buf,key);
    if (!p) return -1;
    p=strchr(p,':');
    if (!p) return -1;
    *out=strtof(p+1,NULL);
    return 0;
}

static int file_exists(const char* path){
    FILE* f=fopen(path,"rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

// save_checkpoint
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

    if (d2h_copy(h_params,M->flat_params,M->n_params)!=0)
        { free(h_params); free(h_m); free(h_v); return -1; }

    if (d2h_copy(h_m,M->opt.m,M->n_params)!=0)
        { free(h_params); free(h_m); free(h_v); return -1; }

    if (d2h_copy(h_v,M->opt.v,M->n_params)!=0)
        { free(h_params); free(h_m); free(h_v); return -1; }

    // filenames
    char p_params[1024],p_m[1024],p_v[1024],p_meta[1024];
    snprintf(p_params,sizeof(p_params),"%s/step_%06d.params.bin",dir,step);
    snprintf(p_m,sizeof(p_m),"%s/step_%06d.opt_m.bin",dir,step);
    snprintf(p_v,sizeof(p_v),"%s/step_%06d.opt_v.bin",dir,step);
    snprintf(p_meta,sizeof(p_meta),"%s/step_%06d.meta.json",dir,step);

    // write
    write_atomic_bin(p_params,h_params,M->n_params,sizeof(float));
    write_atomic_bin(p_m,h_m,M->n_params,sizeof(float));
    write_atomic_bin(p_v,h_v,M->n_params,sizeof(float));

}
