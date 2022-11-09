#ifndef __SRC_MATMUL_ADD_MATRIX_H__
#define __SRC_MATMUL_ADD_MATRIX_H__

#include "tensor.h"
#include <stddef.h>

//#define FP16_ACC16 1

static inline int matmul_add_matrix(Tensor *dst, Tensor *src1, Tensor  *src2)
{
    int m = src1->shape[0];
    int k = src1->shape[1];
    int n = src2->shape[1];

    int stride_s1 = src1->stride / sizeof(float16_t);
    int stride_s2 = src2->stride / sizeof(float16_t);
    int stride_d  = dst->stride / sizeof(float16_t);

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    int mtype = 1 | (1<<3) ; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    for(int i = 0; i < m; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(m-i));
        for (int j = 0; j < n; j += tile_n) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tile_n)
                        : [rs1]"r"(n-j));
            asm volatile("mwemulc.mi acc0, acc1, 0");
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*stride_s1+kk), [rs2]"r"(stride_s1*dataSize));
                
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*stride_s2+j), [rs2]"r"(stride_s2*dataSize));
                asm volatile("mfwma.mm acc0, tr0, tr1");
            }

            asm volatile("mfncvtc.f.fw.m acc1, acc0");
            asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                    :
                    : [rs1]"r"(pdst+i*stride_d+j), [rs2]"r"(stride_d*dataSize));
            asm volatile("mfaddc.mm acc0, acc1");
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*stride_d+j), [rs2]"r"(stride_d*dataSize));
        }
        

    }
    return 0;
}


static inline int matmul_add(Tensor *dst, Tensor *src1, Tensor *src2) {
    return matmul_add_matrix(dst, src1, src2);
}

#endif // __SRC_MATMUL_ADD_MATRIX_H__
