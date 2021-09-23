#ifndef __RELU_H__
#define __RELU_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

/*
    dst = src > base ? src : base
*/
static inline int relu(Tensor *dst, Tensor *src, float16_t base)
{
    float16_t *psrc = (float16_t *)src->data;
    float16_t *pdst = (float16_t *)dst->data;

    int vlmax = VLENB * 8 / src->elemsize;

    for(int i = 0; i < src->size; i += vlmax) {
        int vl = min(vlmax, src->size - i);
        vfloat16m8_t _src = vle16_v_f16m8(psrc, vl);
        psrc += vl;
        vfloat16m8_t _dst = vfmax_vf_f16m8(_src, base, vl);
        vse16_v_f16m8(pdst, _dst, vl);
        pdst += vl;
    }
     
    return 0;
}

#endif