#ifndef __MATMUL_NCORES_H__
#define __MATMUL_NCORES_H__

#include "add.h"
#include "matmul.h"
#include "matmul_add.h"
#include "tensor.h"
#include "util.h"
#include <stddef.h>

// #define FP16_ACC16 1

static inline int matmul_n_ncores(Tensor *dst, Tensor *src1, Tensor *src2,
                                  int ncores) {
  int m = src1->shape[0];
  int k = src1->shape[1];
  int n = src2->shape[1];

  int stride_s1 = src1->stride;
  int stride_s2 = src2->stride;
  int stride_d = dst->stride;

  int dataSize = sizeof(float16_t);

  assert(k % ncores == 0 && n % ncores == 0);

  char *psrc1 = (char *)src1->data;
  char *psrc2 = (char *)src2->data;
  char *pdst = (char *)dst->data;

  int pid = read_csr(mhartid);

  int part_k = k / ncores;
  int part_n = n / ncores;

  // clear dst
  // for (int i = 0; i < m; i++) {
  //     for (int j = pid * part_n * dataSize; j < pid * part_n * dataSize +
  //     part_n * dataSize; j++) {
  //         *(pdst + i * stride_d + j) = 0;
  //     }
  // }

  for (int i = 0; i < ncores; ++i) {
    int kidx = (pid + i) % ncores;

    char *_src1 = psrc1 + kidx * part_k * dataSize;

    tensor_new_2d_with_stride(_src1Mat, m, part_k, dataSize, _src1, stride_s1);

    int nidx = pid;

    char *_src2 = psrc2 + kidx * part_k * stride_s2 + nidx * part_n * dataSize;

    tensor_new_2d_with_stride(_src2Mat, part_k, part_n, dataSize, _src2,
                              stride_s2);

    char *_dst = pdst + pid * part_n * dataSize;

    tensor_new_2d_with_stride(_dstMat, m, part_n, dataSize, _dst, stride_d);

    matmul_add(&_dstMat, &_src1Mat, &_src2Mat);
  }

  return 0;
}

static inline int matmul_m_ncores(Tensor *dst, Tensor *src1, Tensor *src2,
                                  int ncores) {
  int m = src1->shape[0];
  int k = src1->shape[1];
  int n = src2->shape[1];

  int stride_s1 = src1->stride;
  int stride_s2 = src2->stride;
  int stride_d = dst->stride;

  int dataSize = sizeof(float16_t);

  assert(m % ncores == 0 && k % ncores == 0 && n % ncores == 0);

  char *psrc1 = (char *)src1->data;
  char *psrc2 = (char *)src2->data;
  char *pdst = (char *)dst->data;

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
      int kidx = (pid + j) % ncores;

      char *_src1 =
          psrc1 + midx * part_m * stride_s1 + kidx * part_k * dataSize;

      tensor_new_2d_with_stride(_src1Mat, part_m, part_k, dataSize, _src1,
                                stride_s1)

          int nidx = i;

      char *_src2 =
          psrc2 + kidx * part_k * stride_s2 + nidx * part_n * dataSize;

      tensor_new_2d_with_stride(_src2Mat, part_k, part_n, dataSize, _src2,
                                stride_s2)

          char *_dst =
              pdst + midx * part_m * stride_d + nidx * part_n * dataSize;

      tensor_new_2d_with_stride(_dstMat, part_m, part_n, dataSize, _dst,
                                stride_d);

      matmul_add(&_dstMat, &_src1Mat, &_src2Mat);
    }
  }

  return 0;
}

static inline int matmul_k_ncores(Tensor *dst, Tensor *src1, Tensor *src2,
                                  int ncores) {
  int m = src1->shape[0];
  int k = src1->shape[1];
  int n = src2->shape[1];

  int stride_s1 = src1->stride;
  int stride_s2 = src2->stride;
  int stride_d = dst->stride;

  int dataSize = sizeof(float16_t);

  assert(k % ncores == 0 && n % ncores == 0);

  char *psrc1 = (char *)src1->data;
  char *psrc2 = (char *)src2->data;
  char *pdst = (char *)dst->data;

  int pid = read_csr(mhartid);

  int part_k = k / ncores;
  int part_n = n / ncores;

  // clear dst
  // memset(pdst, 0, m * n * dataSize);

  barrier(ncores);

  for (int i = 0; i < ncores; ++i) {
    int kidx = pid;

    char *_src1 = psrc1 + kidx * part_k * dataSize;

    tensor_new_2d_with_stride(_src1Mat, m, part_k, dataSize, _src1, stride_s1)

        int nidx = (2 * ncores + (pid - 1) - i) % ncores;

    char *_src2 = psrc2 + kidx * part_k * stride_s2 + nidx * part_n * dataSize;

    tensor_new_2d_with_stride(_src2Mat, part_k, part_n, dataSize, _src2,
                              stride_s2)

        char *_dst = pdst + nidx * part_n * dataSize;

    tensor_new_2d_with_stride(_dstMat, m, part_n, dataSize, _dst, stride_d);

    matmul_add(&_dstMat, &_src1Mat, &_src2Mat);

    barrier(ncores);
  }

  return 0;
}

static inline int matmul_ncores(Tensor *dst, Tensor *src1, Tensor *src2,
                                int ncores) {
  matmul_n_ncores(dst, src1, src2, ncores);
}

#endif // __MATMUL_H__
