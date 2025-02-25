#ifndef __ADD_H__
#define __ADD_H__

#include "../include/matrix/matrix_intrinsic.h"
#include "tensor.h"
#include <stddef.h>

static inline int add(Tensor *dst, Tensor *src1, Tensor *src2)
{
  float16_t *psrc1 = (float16_t *)src1->data;
  float16_t *psrc2 = (float16_t *)src2->data;
  float16_t *pdst = (float16_t *)dst->data;
  const int M = src1->shape[3] * src1->shape[2];
  const int N = src1->shape[1] * src1->shape[0];

  SET_MBA0_FP16();
  int tile_m = 0, tile_n = 0;
  for (int m = 0; m < M; m += tile_m)
  {
    tile_m = msettilem(M - m);
    for (int n = 0; n < N; n += tile_n)
    {
      tile_n = msettilen(N - n);
      mfloat16_t acc0 = mlc_m(psrc1 + m * N + n, N * sizeof(float16_t));
      mfloat16_t acc1 = mlc_m(psrc2 + m * N + n, N * sizeof(float16_t));
      mfloat16_t md = mfadd_mm(acc0, acc1);
      msc_m(md, pdst + m * N + n, N * sizeof(float16_t));
    }
  }
  printf("psrc1: %p\n", psrc1);
  printf("psrc2: %p\n", psrc2);
  printf("pdst: %p\n", pdst);

  return 0;
}

#endif //__ADD_H__
