#ifndef __RELU_H__
#define __RELU_H__

#include "../include/matrix/matrix_intrinsic.h"
#include "tensor.h"
#include <stddef.h>

/*
    dst = src > base ? src : base
*/
static inline int relu(Tensor *dst, Tensor *src, float16_t base)
{
    float16_t *psrc = (float16_t *)src->data;
    float16_t *pdst = (float16_t *)dst->data;
    const int M = src->shape[2];
    const int N = src->shape[1] * src->shape[0];

    printf("psrc: %p\n", psrc);
    printf("pdst: %p\n", pdst);

    SET_MBA0_FP16();
    int tile_m = 0, tile_n = 0;
    for (int m = 0; m < M; m += tile_m)
    {
        tile_m = msettilem(M - m);
        for (int n = 0; n < N; n += tile_n)
        {
            tile_n = msettilen(N - n);
            mfloat16_t base_m;
            //   base_m = mfsub_mm(base_m, base_m);
            base_m = mfmv_a_f(base_m, base, 0);
            base_m = mbcce_m(base_m);

            mfloat16_t acc = mlc_m(psrc + m * N + n, N * sizeof(float16_t));
            mfloat16_t md = mfmax_mm(acc, base_m);
            msc_m(md, pdst + m * N + n, N * sizeof(float16_t));
        }
    }

    return 0;
}

#endif