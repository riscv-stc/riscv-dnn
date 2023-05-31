#ifndef __SRC_MATMUL_ADD_MATRIX_H__
#define __SRC_MATMUL_ADD_MATRIX_H__

#include "tensor.h"
#include "mme.h"
#include <stddef.h>

//#define FP16_ACC16 1
static inline int matmul_add_matrix_batch16(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
{
    int m = ss->m;
    int k = ss->k;
    int n = ss->n;

    int stride_s1 = ss->stride_src1 / sizeof(float16_t);
    int stride_s2 = ss->stride_src2 / sizeof(float16_t);
    int stride_d  = ss->stride_dst / sizeof(float16_t);

    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

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
            float16_t *_pdst = pdst+i*stride_d+j;
            asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc2, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc3, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc4, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc5, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc6, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc7, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc8, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc9, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc10, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc11, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc12, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc13, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc14, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc15, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*stride_s2+j), [rs2]"r"(stride_s2*dataSize));
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc0, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc1, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc2, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc3, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc4, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc5, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc6, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc7, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc8, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc9, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc10, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc11, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc12, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc13, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc14, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc15, tr0, tr1");
            }
            _pdst = pdst+i*stride_d+j;
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc2, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc3, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc4, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc5, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc6, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc7, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc8, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc9, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc10, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc11, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc12, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc13, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc14, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc15, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
        }
        

    }
    return 0;
}


static inline int matmul_add_matrix_batch8(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
{
    int m = ss->m;
    int k = ss->k;
    int n = ss->n;

    int stride_s1 = ss->stride_src1 / sizeof(float16_t);
    int stride_s2 = ss->stride_src2 / sizeof(float16_t);
    int stride_d  = ss->stride_dst / sizeof(float16_t);

    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

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
            float16_t *_pdst = pdst+i*stride_d+j;
            asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc2, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc3, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc4, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc5, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc6, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc7, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*stride_s2+j), [rs2]"r"(stride_s2*dataSize));
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc0, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc1, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc2, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc3, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc4, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc5, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc6, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc7, tr0, tr1");
            }
            _pdst = pdst+i*stride_d+j;
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc2, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc3, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc4, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc5, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc6, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc7, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
        }
        

    }
    return 0;
}

static inline int matmul_add_matrix_batch4(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
{
    int m = ss->m;
    int k = ss->k;
    int n = ss->n;

    int stride_s1 = ss->stride_src1 / sizeof(float16_t);
    int stride_s2 = ss->stride_src2 / sizeof(float16_t);
    int stride_d  = ss->stride_dst / sizeof(float16_t);

    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

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
            float16_t *_pdst = pdst+i*stride_d+j;
            asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc2, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc3, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*stride_s2+j), [rs2]"r"(stride_s2*dataSize));
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc0, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc1, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc2, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc3, tr0, tr1");
            }
            _pdst = pdst+i*stride_d+j;
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc2, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc3, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
        }
        

    }
    return 0;
}

static inline int matmul_add_matrix_batch2(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
{
    int m = ss->m;
    int k = ss->k;
    int n = ss->n;

    int stride_s1 = ss->stride_src1 / sizeof(float16_t);
    int stride_s2 = ss->stride_src2 / sizeof(float16_t);
    int stride_d  = ss->stride_dst / sizeof(float16_t);

    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

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
            float16_t *_pdst = pdst+i*stride_d+j;
            asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*stride_s2+j), [rs2]"r"(stride_s2*dataSize));
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc0, tr0, tr1");
                _psrc1 += srcSize;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc1, tr0, tr1");
            }
            _pdst = pdst+i*stride_d+j;
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            _pdst += dstSize;
            asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
        }
        

    }
    return 0;
}

static inline int matmul_add_matrix(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
{
    int m = ss->m;
    int k = ss->k;
    int n = ss->n;

    int stride_s1 = ss->stride_src1 / sizeof(float16_t);
    int stride_s2 = ss->stride_src2 / sizeof(float16_t);
    int stride_d  = ss->stride_dst / sizeof(float16_t);

    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

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
            float16_t *_pdst = pdst+i*stride_d+j;
            asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*stride_s2+j), [rs2]"r"(stride_s2*dataSize));
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1), [rs2]"r"(stride_s1*dataSize));
                asm volatile("mfma.mm acc0, tr0, tr1");
            }
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(_pdst), [rs2]"r"(stride_d*dataSize));
        }
        

    }
    return 0;
}

static inline int matmul_add(void *dst, void *src1, void *src2, ConfigMatmul *ss, int batch, int srcSize, int dstSize) {
    switch(batch) {
    case 16:
        matmul_add_matrix_batch16(dst, src1, src2, ss, srcSize, dstSize);
        break;
    case 8:
        matmul_add_matrix_batch8(dst, src1, src2, ss, srcSize, dstSize);
        break;
    case 4:
        matmul_add_matrix_batch4(dst, src1, src2, ss, srcSize, dstSize);
        break;
    case 2:
        matmul_add_matrix_batch2(dst, src1, src2, ss, srcSize, dstSize);
        break;
    default:
        matmul_add_matrix(dst, src1, src2, ss, srcSize, dstSize);


    }
}

#endif // __SRC_MATMUL_ADD_MATRIX_H__
