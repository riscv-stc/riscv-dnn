#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#include "tensor.h"
#include <stddef.h>
#include "../include/matrix/matrix_intrinsic.h"
//#define FP16_ACC16 1

static inline int rename_v2(Tensor *dst, Tensor *src1)
{
    int h = src1->h;
    int w = src1->w;

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    int mtype = e16; // sew = e16
    asm volatile("msettype x1, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    asm volatile("msettilem x1, %[rs1]"
                    :
                    : [rs1]"r"(8));
    asm volatile("msettilek x1, %[rs1]"
                    :
                    : [rs1]"r"(8));

    asm volatile("vsetvli a1, a2, e16,m1,ta,ma"
                    :
                    : [a1]"r"(8));

    asm volatile("mle16.tr.r.k tr0, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(psrc1), [rs2]"r"(w*2));
    
    asm volatile("mle16.tr.r.k tr0, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(psrc1), [rs2]"r"(w*2));

    asm volatile("mle16.tr.r.k tr1, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(psrc1), [rs2]"r"(w*2));
    
    asm volatile("mle16.tr.c.k tr0, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(psrc1), [rs2]"r"(w*2));

    asm volatile("vle16.v v1, (%[rs1])"
                :
                :[rs1]"r"(psrc1));
    
    asm volatile("mwmv.v.tr.c.k v1, tr0, x0");

    asm volatile("mclracc acc0");

    asm volatile("mfwopa.vv acc0, tr0, tr1");

    asm volatile("mse16.tr.c.k tr0, (%[rs1]), %[rs2]"
                : 
                : [rs1]"r"(pdst), [rs2]"r"(w*2));

    asm volatile("mse16.tr.c.k tr0, (%[rs1]), %[rs2]"
                : 
                : [rs1]"r"(pdst), [rs2]"r"(w*2));
                
    return 0;
}

#endif // __SRC_MATMUL_H__
