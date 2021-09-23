#ifndef __SRC_ADD_H__
#define __SRC_ADD_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

//#define USE_FP32

static inline int add(Tensor *dst, Tensor *src1, Tensor *src2)
{
    assert(src1->size == src2->size && src2->size == dst->size);
    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;
#ifndef USE_FP32
    int vlmax = VLENB * 8 / src1->elemsize;
#else
    int vlmax = VLENB * 4 / src1->elemsize;
#endif

    for(int i = 0; i < src1->size; i += vlmax) {
        int vl = min(vlmax, src1->size - i);
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

#endif //__SRC_ADD_H__
