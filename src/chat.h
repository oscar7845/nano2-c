#pragma once
#include "nano2_model.h"

#ifdef __cplusplus
extern "C" {
#endif

int run_chat(const char* ckpt_file,
             const char* ckpt_dir,
             int use_best,
             struct Config base_cfg,
             int max_new_tokens,
             int top_k,
             float temperature,
             unsigned int seed);

#ifdef __cplusplus
}
#endif

