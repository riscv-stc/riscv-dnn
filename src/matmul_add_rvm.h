#ifndef __SRC_MATMUL_ADD_MATRIX_H__
#define __SRC_MATMUL_ADD_MATRIX_H__

#include "tensor.h"
#include "mme.h"
#include <stddef.h>
#include <riscv_matrix.h>

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
    msettype(E16, M1, BA);
    
    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            float16_t *_pdst = pdst+i*stride_d+j;
            mfloat16m1_t acc0 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc1 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc2 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc3 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc4 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc5 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc6 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc7 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc8 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc9 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc10 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc11 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc12 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc13 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc14 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc15 = mlce16_m1(_pdst, stride_d*dataSize);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_m = msettilem(k-kk);
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
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
            _pdst = pdst+i*stride_d+j;
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
    msettype(E16, M1, BA);
    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            float16_t *_pdst = pdst+i*stride_d+j;
            mfloat16m1_t acc0 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc1 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc2 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc3 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc4 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc5 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc6 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc7 = mlce16_m1(_pdst, stride_d*dataSize);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
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
            _pdst = pdst+i*stride_d+j;
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
    msettype(E16, M1, BA);
    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            float16_t *_pdst = pdst+i*stride_d+j;
            mfloat16m1_t acc0 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc1 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc2 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc3 = mlce16_m1(_pdst, stride_d*dataSize);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
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
            _pdst = pdst+i*stride_d+j;
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
    msettype(E16, M1, BA);
    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            float16_t *_pdst = pdst+i*stride_d+j;
            mfloat16m1_t acc0 = mlce16_m1(_pdst, stride_d*dataSize);
            _pdst += dstSize;
            mfloat16m1_t acc1 = mlce16_m1(_pdst, stride_d*dataSize);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                mfloat16m1_t tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc0 = mfma_mm(acc0, tr0, tr1);
                _psrc1 += srcSize;
                tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc1 = mfma_mm(acc1, tr0, tr1);
            }
            _pdst = pdst+i*stride_d+j;
            msce16_m(acc0, _pdst, stride_d*dataSize);
            _pdst += dstSize;
            msce16_m(acc1, _pdst, stride_d*dataSize);
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
    msettype(E16, M1, BA);
    for(int i = 0; i < m; i += tile_m) {
        tile_m = msettilem(m-i);
        for (int j = 0; j < n; j += tile_n) {
            tile_n = msettilen(n-j);
            float16_t *_pdst = pdst+i*stride_d+j;
            mfloat16m1_t acc0 = mlce16_m1(_pdst, stride_d*dataSize);
            for (int kk = 0; kk < k; kk += tile_k) {
                tile_k = msettilek(k-kk);
                mfloat16m1_t tr1 = mlbe16_m1(psrc2+kk*stride_s2+j, stride_s2*dataSize);
                float16_t *_psrc1 = psrc1+i*stride_s1+kk;
                mfloat16m1_t tr0 = mlae16_m1(_psrc1, stride_s1*dataSize);
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
            msce16_m(acc0, _pdst, stride_d*dataSize);
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
