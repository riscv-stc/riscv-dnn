#ifndef __SRC_ADD_H__
#define __SRC_ADD_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

//#define USE_FP32

static inline int add(void *dst, void *src1, void *src2, int size)
{
    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

    int vl;
    for(int i = 0; i < size; i += vl) {
        vl = vsetvl_e16m8(size - i);
#ifndef USE_FP32
        vfloat16m8_t _src1 = vle16_v_f16m8(psrc1, vl);
        psrc1 += vl;
        vfloat16m8_t _src2 = vle16_v_f16m8(psrc2, vl);
        psrc2 += vl;
        vfloat16m8_t _dst = vfadd_vv_f16m8(_src1, _src2, vl);
        vse16_v_f16m8(pdst, _dst, vl);
        pdst += vl;
#else
        vfloat16m4_t _src1 = vle16_v_f16m4(psrc1, vl);
        psrc1 += vl;
        vfloat16m4_t _src2 = vle16_v_f16m4(psrc2, vl);
        psrc2 += vl;
        vfloat32m8_t _dst = vfwadd_vv_f32m8(_src1, _src2, vl);
        vse16_v_f16m4(pdst, vfncvt_f_f_w_f16m4(_dst, vl), vl);
        pdst += vl;
#endif
    }
     
    return 0;
}

static inline int addw(void *dst, void *src1, void *src2, int size)
{
    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float32_t *pdst = (float32_t *)dst;

    int vl;
    for(int i = 0; i < size; i += vl) {
        vl = vsetvl_e16m4(size - i);
        vfloat16m4_t _src1 = vle16_v_f16m4(psrc1, vl);
        psrc1 += vl;
        vfloat16m4_t _src2 = vle16_v_f16m4(psrc2, vl);
        psrc2 += vl;
        vfloat32m8_t _dst = vfwadd_vv_f32m8(_src1, _src2, vl);
        vse32_v_f32m8(pdst, _dst, vl);
        pdst += vl;

    }
     
    return 0;
}

#endif //__SRC_ADD_H__
