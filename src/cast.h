#ifndef __CAST_H__
#define __CAST_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

static inline int cast_f32_to_f16(void *dst, void *src, int size)
{
    float32_t *psrc = (float32_t *)src;
    float16_t *pdst = (float16_t *)dst;

    int vl;
    for(int i = 0; i < size; i += vl) {
        vl = vsetvl_e32m8(size - i);
        vfloat32m8_t _src = vle32_v_f32m8(psrc, vl);
        psrc += vl;
        vfloat16m4_t _dst = vfncvt_f_f_w_f16m4(_src, vl);
        vse16_v_f16m4(pdst, _dst, vl);
        pdst += vl;
    }
     
    return 0;
}

static inline int cast_f16_to_f32(void *dst, void *src, int size)
{
    float16_t *psrc = (float16_t *)src;
    float32_t *pdst = (float32_t *)dst;

    int vl;
    for(int i = 0; i < size; i += vl) {
        vl = vsetvl_e16m4(size - i);
        vfloat16m4_t _src = vle16_v_f16m4(psrc, vl);
        psrc += vl;
        vfloat32m8_t _dst = vfwcvt_f_f_v_f32m8(_src, vl);
        vse32_v_f32m8(pdst, _dst, vl);
        pdst += vl;
    }
     
    return 0;
}

#endif