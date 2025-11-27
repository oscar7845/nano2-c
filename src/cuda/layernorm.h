#pragma once
extern "C" void nano2_layernorm_forward(const float* x, float* y,
                                        const float* gamma, const float* beta,
                                        int N, int D, float eps);

