#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#define OUTPUT_UINT32

#ifdef OUTPUT_UINT8
typedef uint8_t  outtype;
#elif defined(OUTPUT_UINT16)
typedef uint16_t  outtype;
#else
typedef uint32_t  outtype;
#endif

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "../include/matrix/matrix_intrinsic.h"

//#define FP16_ACC16 1

static inline int matmul_matrix(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int m = src1->shape[0];
    int k = src1->shape[1];

    int h2 = src2->shape[0];
    int n = src2->shape[1];

    assert(k == h2);

    int hout = dst->shape[0];
    int wout = dst->shape[1];

    assert(hout == m && wout == n);

    uint8_t *psrc1 = (uint8_t *)src1->data;
    uint8_t *psrc2 = (uint8_t *)src2->data;
    outtype *pdst = (outtype *)dst->data;



    const int src_dataSize = sizeof(uint8_t);
    const int dst_dataSize = sizeof(outtype);

    int tile_m = 0, tile_n = 0, tile_k = 0;
#ifndef  OUTPUT_UINT8
    int mtype = e8 | (1<<3) ; // sew = e16, mlmul = 128
#else
     int mtype = e8; // sew = e16, mlmul = 128
#endif
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
#ifdef OUTPUT_UINT8
            asm volatile("msubc.mm acc0, acc0");
#elif defined(OUTPUT_UINT16)
            asm volatile("mwsubc.mm acc0, acc0");
#else
            asm volatile("mqsubc.mm acc0, acc0");
#endif      
            // asm volatile("msubc.mm acc0, acc0");
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mlae8.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*k+kk), [rs2]"r"(k*src_dataSize));
                
                asm volatile("mlbe8.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*n+j), [rs2]"r"(n*src_dataSize));
#ifdef OUTPUT_UINT8
                asm volatile("mma.mm acc0, tr0, tr1");
#elif defined(OUTPUT_UINT16)
                asm volatile("mwma.mm acc0, tr0, tr1");
#else
                asm volatile("mqma.mm acc0, tr0, tr1");
#endif 
            }

            
#ifdef OUTPUT_UINT8    
            asm volatile("msce8.m acc0, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*n+j), [rs2]"r"(n*dst_dataSize));
#elif defined(OUTPUT_UINT16)
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*n+j), [rs2]"r"(n*dst_dataSize));
#else
            asm volatile("msce32.m acc0, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*n+j), [rs2]"r"(n*dst_dataSize));
#endif
        }
        

    }
    return 0;
}


static inline int matmul(Tensor *dst, Tensor *src1, Tensor *src2) {
    return matmul_matrix(dst, src1, src2);
}

#endif // __SRC_MATMUL_H__
