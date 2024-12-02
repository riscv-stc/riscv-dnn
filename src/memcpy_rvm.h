#ifndef __MEMCPY_RVM_H__
#define __MEMCPY_RVM_H__

#include "tensor.h"
#include <riscv_matrix.h>
#include <riscv_vector.h>
#include <stddef.h>
// #define FP16_ACC16 1

static inline int memcpy_rvm(void *dst, void *src, int m, int k) {
  float16_t *psrc = (float16_t *)src;
  float16_t *pdst = (float16_t *)dst;

  const int dataSize = sizeof(float16_t);

  int tile_m = 0, tile_k = 0;
  msettypei(0x1);
  msettypehi(0x1);

  for (int i = 0; i < m; i += tile_m) {
    tile_m = msettilem(m - i);
    for (int j = 0; j < k; j += tile_k) {
      tile_k = msettilek(k - j);
      mfloat16_t tr0 = mla_m(psrc + i * k + j, k * dataSize);
      msa_m(tr0, pdst + i * k + j, k * dataSize);
    }
  }
  return 0;
}

static inline int memclr_rvm(void *dst, int m, int k) {
  float16_t *pdst = (float16_t *)dst;

  const int dataSize = sizeof(float16_t);

  int tile_m = 0, tile_k = 0;
  msettypei(0x1);
  msettypehi(0x1);

  for (int i = 0; i < m; i += tile_m) {
    tile_m = msettilem(m - i);
    for (int j = 0; j < k; j += tile_k) {
      tile_k = msettilek(k - j);
      mfloat16_t zero;
      zero = mfsub_mm(zero, zero);
      msa_m(zero, pdst + i * k + j, k * dataSize);
    }
  }
  return 0;
}

#endif // __MEMCPY_RVM_H__
