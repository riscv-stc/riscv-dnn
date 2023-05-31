#ifndef __SRC_MEMCPY_RVM_H__
#define __SRC_MEMCPY_RVM_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

//#define FP16_ACC16 1

static inline int memcpy_rvm(void *dst, void *src, int m, int k)
{
    float16_t *psrc = (float16_t *)src;
    float16_t *pdst = (float16_t *)dst;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_k = 0;
    int mtype = e16 | (1<<3) ; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    for(int i = 0; i < m; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(m-i));
        for (int j = 0; j < k; j += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-j));
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc+i*k+j), [rs2]"r"(k*dataSize));
                asm volatile("msae16.m tr0, (%[rs1]), %[rs2]"
                            : 
                            : [rs1]"r"(pdst+i*k+j), [rs2]"r"(k*dataSize));
        }
        

    }
    return 0;
}

static inline int memclr_rvm(void *dst, int len)
{
    float16_t *pdst = (float16_t *)dst;

    const int dataSize = sizeof(float16_t);
    
    int m = len / 64;
    int tile_m = 0, tile_n = 0;
    int mtype = e16 | (1<<3) ; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tile_m)
                : [rs1]"r"(64));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tile_n)
                : [rs1]"r"(64));
    asm volatile("msubc.mm acc0, acc0");
    for(int i = 0; i < m; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(m-i));
        asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*64), [rs2]"r"(64*dataSize));
    }
    return 0;
}


#endif // __SRC_MEMCPY_RVM_H__
