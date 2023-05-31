#ifndef __SRC_MATMUL_NCORES_H__
#define __SRC_MATMUL_NCORES_H__

#include "tensor.h"
#include <stddef.h>
#include "matmul.h"
#include "matmul_add.h"
#include "add.h"
#include "util.h"
#include "mme.h"
//#define FP16_ACC16 1

static inline int matmul_n_ncores(void *dst, void *src1, void *src2, ConfigMatmul *ss, int ncores, int batch)
{
    int m = ss->m;
    int n = ss->n;
    int k = ss->k;

    int stride_s1 = ss->stride_src1;
    int stride_s2 = ss->stride_src2;
    int stride_d = ss->stride_dst;

    int dataSize = sizeof(float16_t);


    assert(k%ncores==0 && n%ncores==0);

    char *psrc1 = (char *)src1;
    char *psrc2 = (char *)src2;
    char *pdst = (char *)dst;

    int pid = read_csr(mhartid);
    
    int part_k = k / ncores;
    int part_n = n / ncores;
    
    // clear dst
    // for (int i = 0; i < m; i++) {
    //     for (int j = pid * part_n * dataSize; j < pid * part_n * dataSize + part_n * dataSize; j++) {
    //         *(pdst + i * stride_d + j) = 0;
    //     }
    // }


    for (int i = 0; i < ncores; ++i) {
        int kidx = (pid+i)%ncores;
        
        char *_src1 = psrc1 + kidx * part_k * dataSize;
        
        int nidx = pid;

        char *_src2 = psrc2 + kidx * part_k * stride_s2 + nidx * part_n * dataSize;

        char *_dst = pdst + pid * part_n * dataSize;

        config_matmul(_ss, m, part_k, part_n, stride_s1, stride_s2, stride_d)

        matmul_add(_dst, _src1, _src2, &_ss, batch, m*k, m*n);

    }

    return 0;
}


static inline int matmul_m_ncores(void *dst, void *src1, void *src2, ConfigMatmul *ss, int ncores, int batch)
{
    int m = ss->m;
    int n = ss->n;
    int k = ss->k;

    int stride_s1 = ss->stride_src1;
    int stride_s2 = ss->stride_src2;
    int stride_d = ss->stride_dst;

    int dataSize = sizeof(float16_t);

    assert(m%ncores==0 && k%ncores==0 && n%ncores==0);

    char *psrc1 = (char *)src1;
    char *psrc2 = (char *)src2;
    char *pdst = (char *)dst;

    int pid = read_csr(mhartid);
    
    int part_m = m / ncores;
    int part_k = k / ncores;
    int part_n = n / ncores;

    // clear dst
    // for (int i = pid * part_m; i < pid * part_m + part_m; i++) {
    //     for (int j = 0; j < n * dataSize; j++) {
    //         *(pdst + i * stride_d + j) = 0;
    //     }
    // }


    for (int i = 0; i < ncores; ++i) {
        for (int j = 0; j < ncores; ++j) {
            int midx = pid;
            int kidx = (pid+j)%ncores;
            
            char *_src1 = psrc1 + midx * part_m * stride_s1 + kidx * part_k * dataSize;
            
            int nidx = i;

            char *_src2 = psrc2 + kidx * part_k * stride_s2 + nidx * part_n * dataSize;

            char *_dst = pdst + midx * part_m * stride_d + nidx * part_n * dataSize;

            config_matmul(_ss, part_m, part_k, part_n, stride_s1, stride_s2, stride_d)

            matmul_add(_dst, _src1, _src2, &_ss, batch, m*k, m*n);
        }
    }

    return 0;
}

static inline int matmul_transpose_m_ncores(void *dst, void *src1, void *src2, ConfigMatmul *ss, int ncores)
{
    int m = ss->m;
    int n = ss->n;
    int k = ss->k;

    int stride_s1 = ss->stride_src1;
    int stride_s2 = ss->stride_src2;
    int stride_d = ss->stride_dst;

    int dataSize = sizeof(float16_t);

    assert(m%ncores==0 && k%ncores==0 && n%ncores==0);

    char *psrc1 = (char *)src1;
    char *psrc2 = (char *)src2;
    char *pdst = (char *)dst;

    int pid = read_csr(mhartid);
    
    int part_m = m / ncores;
    int part_k = k / ncores;
    int part_n = n / ncores;


    for (int i = 0; i < ncores; ++i) {
        int midx = pid;
            
        char *_src1 = psrc1 + midx * part_m * stride_s1;
            
        int nidx = i;

        char *_src2 = psrc2 + nidx * part_n * stride_s2;

        char *_dst = pdst + midx * part_m * stride_d + nidx * part_n * dataSize;

        config_matmul(_ss, part_m, k, part_n, stride_s1, stride_s2, stride_d)

        matmul_rvm_tranpose(_dst, _src1, _src2, &_ss);
    }

    return 0;
}



static inline int matmul_m_ncores_2(void *dst, void *src1, void *src2, ConfigMatmul *ss, int ncores, int batch)
{
    int m = ss->m;
    int n = ss->n;
    int k = ss->k;

    int stride_s1 = ss->stride_src1;
    int stride_s2 = ss->stride_src2;
    int stride_d = ss->stride_dst;

    int dataSize = sizeof(float16_t);

    assert(m%ncores==0 && k%ncores==0 && n%ncores==0);

    char *psrc1 = (char *)src1;
    char *psrc2 = (char *)src2;
    char *pdst = (char *)dst;

    int pid = read_csr(mhartid);
    
    int part_m = m / ncores;
    int part_k = k / ncores;
    int part_n = n / ncores;


    for (int i = 0; i < ncores; ++i) {
        int midx = pid;
            
        char *_src1 = psrc1 + midx * part_m * stride_s1;
            
        int nidx = i;

        char *_src2 = psrc2 + nidx * part_n * dataSize;

        char *_dst = pdst + midx * part_m * stride_d + nidx * part_n * dataSize;

        config_matmul(_ss, part_m, k, part_n, stride_s1, stride_s2, stride_d)

        matmul_rvm_batch(_dst, _src1, _src2, &_ss, batch, m*k, m*n);
    }

    return 0;
}

static inline int matmul_k_ncores(void *dst, void *src1, void *src2, ConfigMatmul *ss, int ncores, int batch)
{
    int m = ss->m;
    int n = ss->n;
    int k = ss->k;

    int stride_s1 = ss->stride_src1;
    int stride_s2 = ss->stride_src2;
    int stride_d = ss->stride_dst;

    int dataSize = sizeof(float16_t);


    assert(k%ncores==0 && n%ncores==0);

    char *psrc1 = (char *)src1;
    char *psrc2 = (char *)src2;
    char *pdst = (char *)dst;

    int pid = read_csr(mhartid);
    
    int part_k = k / ncores;
    int part_n = n / ncores;

    // clear dst
    // memset(pdst, 0, m * n * dataSize);
    
    barrier(ncores);


    for (int i = 0; i < ncores; ++i) {
        int kidx = pid;
        
        char *_src1 = psrc1 + kidx * part_k * dataSize;
        
        int nidx = (2*ncores + (pid - 1) - i) % ncores;

        char *_src2 = psrc2 + kidx * part_k * stride_s2 + nidx * part_n * dataSize;

        char *_dst = pdst + nidx * part_n * dataSize;

        tensor_new_2d_with_stride(_dstMat, m, part_n, dataSize, _dst, stride_d);
        
        config_matmul(_ss, m, part_k, part_n, stride_s1, stride_s2, stride_d)

        matmul_add(_dst, _src1, _src2, &_ss, batch, m*k, m*n);

        barrier(ncores);

    }

    return 0;
}



static inline int matmul_ncores(void *dst, void *src1, void *src2, ConfigMatmul *ss, int ncores, int batch)
{
    matmul_m_ncores_2(dst, src1, src2, ss, ncores, batch);
}


#endif // __SRC_MATMUL_H__
