#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
	int vocab_size;
	int d_model;
	int ffn_mult;
	int seq_len;
	int batch_size;
	float lr;
	float weight_decay;
	float clip_grad_norm;
	int seed;
	int top_k;
} nano2_config_t;
