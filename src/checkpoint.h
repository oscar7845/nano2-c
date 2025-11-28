#pragma once
#include "nano2_model.h"

#ifdef __cplusplus
extern "C" {
#endif

//save params + AdamW state (+ metadata) into dir
//also writes/updates "latest.*" and "best.*" copies
//returns 0 on success, nonzero on error
int save_checkpoint(const char* dir,
                    const struct Model* M,
                    const struct Config* cfg,
                    int step,
                    float val_loss,
                    int is_best);

//load from "<dir>/latest.*" if it exists 
//Returns 0 on success
//if opt state files are missing, leaves M->opt.{m,v} unchanged
int load_checkpoint_latest(const char* dir,
                           struct Model* M,
                           struct Config* cfg,
                           int* out_step,
                           float* out_val_loss);

#ifdef __cplusplus
}
#endif

