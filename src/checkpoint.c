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
    if (stat(dir, &st) == 0 && (st.st_mode & S_IFDIR)) return 0;
    if (MKDIR(dir) == 0) return 0;
    if (errno == EEXIST) return 0;
    fprintf(stderr, "[ckpt] failed to create dir %s: %s\n", dir, strerror(errno));
    return -1;
}

static int write_atomic_bin(const char* path, const void* ptr, size_t count, size_t elem){
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f){ perror("[ckpt fopen tmp]"); return -1; }
    size_t n=fwrite(ptr, elem, count, f);   // tightened a bit
    if (n != count){
        perror("[ckpt fwrite]");
        fclose(f);
        remove(tmp);
        return -1;
    }
    fflush(f);
#if !defined(_WIN32)
    int fd = fileno(f);
    if (fd >= 0) fsync(fd);
#endif
    fclose(f);
    if (rename(tmp, path) != 0){
        perror("[ckpt rename]");
        remove(tmp);
        return -1;
    }
    return 0;
}

static int write_text(const char* path, const char* s){
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f=fopen(tmp, "wb");
    if (!f){ perror("[ckpt fopen tmp txt]"); return -1; }
    size_t n=fwrite(s, 1, strlen(s), f);
    if (n != strlen(s)){
        perror("[ckpt fwrite txt]");
        fclose(f);
        remove(tmp);
        return -1;
    }
    fflush(f);
#if !defined(_WIN32)
    int fd=fileno(f);
    if (fd >= 0) fsync(fd);
#endif
    fclose(f);
    if (rename(tmp, path) != 0){
        perror("[ckpt rename txt]");
        remove(tmp);
        return -1;
    }
    return 0;
}

static int file_exists(const char* path){
    FILE* f=fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int json_find_int(const char* buf, const char* key, int* out){
    const char* p=strstr(buf, key);
    if (!p) return -1;
    p=strchr(p, ':');
    if (!p) return -1;
    *out=(int)strtol(p+1, NULL, 10);
    return 0;
}

static int json_find_float(const char* buf, const char* key, float* out){
    const char* p=strstr(buf, key);
    if (!p) return -1;
    p=strchr(p, ':');
    if (!p) return -1;
    *out=strtof(p+1, NULL);
    return 0;
}

static int d2h_copy(float* hbuf, const float* dptr, size_t n){
    cudaError_t e=cudaMemcpy(hbuf, dptr, n*sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess){
        fprintf(stderr, "[ckpt] cudaMemcpy D2H failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

static int h2d_copy(float* dptr, const float* hbuf, size_t n){
    cudaError_t e=cudaMemcpy(dptr, hbuf, n*sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess){
        fprintf(stderr, "[ckpt] cudaMemcpy H2D failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}
