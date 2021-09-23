#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#include "tensor.h"
#include <stddef.h>
#include "../include/matrix/matrix_intrinsic.h"
//#define FP16_ACC16 1

static inline int mle(Tensor *dst, Tensor *src1)
{
    int h = src1->h;
    int w = src1->w;

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    int mtype = e16 | (7 << 5); // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    for(int i = 0; i < h; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h-i));
        for (int j = 0; j < w; j += tile_k) {
            asm volatile("msettilek %[rd], %[rs1]"
                         : [rd]"=r"(tile_k)
                         : [rs1]"r"(w-j));
            asm volatile("mle16.tr.r.k tr0, (%[rs1]), %[rs2]"
                         :
                         :[rs1]"r"(psrc1+i*w+j), [rs2]"r"(w*2));
            asm volatile("mse16.tr.r.k tr0, (%[rs1]), %[rs2]"
                         : 
                         : [rs1]"r"(pdst+i*w+j), [rs2]"r"(w*2));
        }
                
    }
    return 0;
}

static inline int multiply(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1= src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    int hout = dst->h;
    int wout = dst->w;

    int8_t *psrc1 = (int8_t *)src1->data;
    int8_t *psrc2 = (int8_t *)src2->data;
    int8_t *pdst = (int8_t *)dst->data;

    const int dataSize = sizeof(int8_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    int mtype = e8 | (7 << 5); // sew = e8
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tile_m)
                : [rs1]"r"(h1));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tile_n)
                : [rs1]"r"(w2));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tile_k)
                : [rs1]"r"(w1));
    asm volatile("mclracc acc0");
    asm volatile("mle8.tr.r.k tr0, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(psrc1), [rs2]"r"(w1));
    asm volatile("mle8.tr.r.n tr1, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(psrc2), [rs2]"r"(w2));
    asm volatile("mopa.vv acc0, tr0, tr1");
    asm volatile("mse8.xa.r.m acc0, (%[rs1]), %[rs2]"
                : 
                : [rs1]"r"(pdst), [rs2]"r"(w2));
                
    return 0;
}

#endif // __SRC_MATMUL_H__
