#pragma once
#include <stdint.h>

extern "C" void nano2_embed_add_pos(
    const uint8_t* x,
    const float*   E,
    const float*   pos_sin,
    const float*   pos_cos,
    float*         out,
    int B, int T, int D
);

