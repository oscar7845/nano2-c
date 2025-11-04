#pragma once
#include <stdio.h>
#include <stdarg.h>

static inline void log_ranked(int rank, const char* level, const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    fprintf(stderr, "[%s][rank=%d] ", level, rank);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

#define LOGI(rank, ...) do { log_ranked((rank), "INFO", __VA_ARGS__); } while (0)
#define LOGW(rank, ...) do { log_ranked((rank), "WARN", __VA_ARGS__); } while (0)
#define LOGE(rank, ...) do { log_ranked((rank), "ERR",  __VA_ARGS__); } while (0)
