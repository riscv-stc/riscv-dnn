#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#include "tensor.h"
#include <stddef.h>
#include "mme.h"
#include <riscv_vector.h>
#include <riscv_matrix.h>
#include "../include/matrix/matrix_intrinsic.h"

//#define FP16_ACC16 1

static inline int matmul_rvm(void *dst, void *src1, void *src2, int m , int k, int n)
{
    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    msettype(E16, M1, BA);

    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            mfloat16m1_t acc0;
            acc0 = mfsub_mm(acc0, acc0);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                mfloat16m1_t tr0 = mlae16_m1(psrc1+i*k+kk, k*dataSize);
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*n+j, n*dataSize);
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
            msce16_m(acc0, pdst+i*n+j, n*dataSize);
        }
        

    }
    return 0;
}

static inline int matmul_rvm_tranpose(void *dst, void *src1, void *src2, ConfigMatmul *ss)
{
    int m = ss->m;
    int k = ss->k;
    int n = ss->n;

    int stride_s1 = ss->stride_src1;
    int stride_s2 = ss->stride_src2;
    int stride_d  = ss->stride_dst;

    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    msettype(E16, M1, BA);

    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            mfloat16m1_t acc0;
            acc0 = mfsub_mm(acc0, acc0);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                mfloat16m1_t tr0 = mlae16_m1(psrc1+i*stride_s1/dataSize+kk, stride_s1);
                mfloat16m1_t tr1 = mlbte16_m1(psrc2+j*stride_s2/dataSize + kk, stride_s2);                
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
            msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
        }
    
    }
    return 0;
}

static inline int matmul_rvm_batch16(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
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
    msettype(E16, M1, BA);
    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            mfloat16m1_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8,
                acc9, acc10, acc11, acc12, acc13, acc14, acc15;
            acc0 = mfsub_mm(acc0, acc0);
            acc1 = mfsub_mm(acc1, acc1);
            acc2 = mfsub_mm(acc2, acc2);
            acc3 = mfsub_mm(acc3, acc3);
            acc4 = mfsub_mm(acc4, acc4);
            acc5 = mfsub_mm(acc5, acc5);
            acc6 = mfsub_mm(acc6, acc6);
            acc7 = mfsub_mm(acc7, acc7);
            acc8 = mfsub_mm(acc8, acc8);
            acc9 = mfsub_mm(acc9, acc9);
            acc10 = mfsub_mm(acc10, acc10);
            acc11 = mfsub_mm(acc11, acc11);
            acc12 = mfsub_mm(acc12, acc12);
            acc13 = mfsub_mm(acc13, acc13);
            acc14 = mfsub_mm(acc14, acc14);
            acc15 = mfsub_mm(acc15, acc15);

            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);        
                mfloat16m1_t tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc0 = mfma_mm(acc0, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc1 = mfma_mm(acc1, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc2 = mfma_mm(acc2, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc3 = mfma_mm(acc3, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc4 = mfma_mm(acc4, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc5 = mfma_mm(acc5, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc6 = mfma_mm(acc6, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc7 = mfma_mm(acc7, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc8 = mfma_mm(acc8, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc9 = mfma_mm(acc9, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc10 = mfma_mm(acc10, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc11 = mfma_mm(acc11, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc12 = mfma_mm(acc12, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc13 = mfma_mm(acc13, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc14 = mfma_mm(acc14, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc15 = mfma_mm(acc15, tr0, tr1);
            }

            float16_t *_pdst = pdst+i*stride_d+j;
            msce16_m(acc0, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc1, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc2, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc3, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc4, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc5, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc6, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc7, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc8, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc9, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc10, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc11, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc12, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc13, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc14, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc15, _pdst, stride_d*dataSize);
        }

    }

    return 0;
}

static inline int matmul_rvm_batch8(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
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
    msettype(E16, M1, BA);

    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            mfloat16m1_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            acc0 = mfsub_mm(acc0, acc0);
            acc1 = mfsub_mm(acc1, acc1);
            acc2 = mfsub_mm(acc2, acc2);
            acc3 = mfsub_mm(acc3, acc3);
            acc4 = mfsub_mm(acc4, acc4);
            acc5 = mfsub_mm(acc5, acc5);
            acc6 = mfsub_mm(acc6, acc6);
            acc7 = mfsub_mm(acc7, acc7);

            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);        
                mfloat16m1_t tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc0 = mfma_mm(acc0, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc1 = mfma_mm(acc1, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc2 = mfma_mm(acc2, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc3 = mfma_mm(acc3, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc4 = mfma_mm(acc4, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc5 = mfma_mm(acc5, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc6 = mfma_mm(acc6, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc7 = mfma_mm(acc7, tr0, tr1);
            }

            float16_t *_pdst = pdst+i*stride_d+j;
            msce16_m(acc0, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc1, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc2, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc3, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc4, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc5, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc6, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc7, _pdst, stride_d*dataSize);
        }

    }

    return 0;
}

static inline int matmul_rvm_batch4(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
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
    msettype(E16, M1, BA);

    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            mfloat16m1_t acc0, acc1, acc2, acc3;
            acc0 = mfsub_mm(acc0, acc0);
            acc1 = mfsub_mm(acc1, acc1);
            acc2 = mfsub_mm(acc2, acc2);
            acc3 = mfsub_mm(acc3, acc3);

            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);
                mfloat16m1_t tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);        
                acc0 = mfma_mm(acc0, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);        
                acc1 = mfma_mm(acc1, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);        
                acc2 = mfma_mm(acc2, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);        
                acc3 = mfma_mm(acc3, tr0, tr1);
            }

            float16_t *_pdst = pdst+i*stride_d+j;
            msce16_m(acc0, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc1, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc2, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc3, _pdst, stride_d*dataSize);
        }
        
    }
    return 0;
}

static inline int matmul_rvm_batch2(void *dst, void *src1, void *src2, ConfigMatmul *ss, int srcSize, int dstSize)
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
    msettype(E16, M1, BA);

    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            mfloat16m1_t acc0, acc1;
            acc0 = mfsub_mm(acc0, acc0);
            acc1 = mfsub_mm(acc1, acc1);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);        
                mfloat16m1_t tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc0 = mfma_mm(acc0, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc1 = mfma_mm(acc1, tr0, tr1);
            }

            float16_t *_pdst = pdst+i*stride_d+j;
            msce16_m(acc0, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc1, _pdst, stride_d*dataSize);
        }
    }
    return 0;
}


static inline int matmul_rvm_batch(void *dst, void *src1, void *src2, ConfigMatmul *ss, int batch, int srcSize, int dstSize) {
    switch (batch)
    {
    case 16:
        matmul_rvm_batch16(dst, src1, src2, ss, srcSize, dstSize);
        break;
    case 8:
        matmul_rvm_batch8(dst, src1, src2, ss, srcSize, dstSize);
        break;
    case 4:
        matmul_rvm_batch4(dst, src1, src2, ss, srcSize, dstSize);
        break;
    case 2:
        matmul_rvm_batch2(dst, src1, src2, ss, srcSize, dstSize);
        break;
    
    default:
        matmul_rvm(dst, src1, src2, ss->m, ss->k, ss->n);
        break;
    }
}

static inline int matmul(void *dst, void *src1, void *src2, int m, int k, int n) {
    return matmul_rvm(dst, src1, src2, m, k, n);
}

#endif // __SRC_MATMUL_H__
