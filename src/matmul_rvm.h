#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "../include/matrix/matrix_intrinsic.h"

//#define FP16_ACC16 1

static inline int matmul_matrix(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int m = src1->h;
    int k = src1->w;

    int h2 = src2->h;
    int n = src2->w;

    assert(k == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == m && wout == n);

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    int mtype = e16 | (1<<3) ; // sew = e16, mlmul = 128
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
                            :[rs1]"r"(psrc1+i*k+kk), [rs2]"r"(k*dataSize));
                
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*n+j), [rs2]"r"(n*dataSize));
                asm volatile("mfwma.mm acc0, tr0, tr1");
            }

            asm volatile("mfncvtc.f.fw.m acc1, acc0");
            asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*n+j), [rs2]"r"(n*dataSize));
        }
        

    }
    return 0;
}


static inline int matmul(Tensor *dst, Tensor *src1, Tensor *src2) {
    return matmul_matrix(dst, src1, src2);
}

#endif // __SRC_MATMUL_H__
