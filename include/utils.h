#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void nano2_print_env(int mpi_rank);
int nano2_ensure_shared_path(const char* path);
void nano2_cuda_selftest(void);

#ifdef __cplusplus
}
#endif
