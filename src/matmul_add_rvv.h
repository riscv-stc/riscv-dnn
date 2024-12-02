#ifndef __MATMUL_ADD_RVV_H__
#define __MATMUL_ADD_RVV_H__

#include "tensor.h"
#include <stddef.h>

#define FP16_ACC16 1

static inline int matmul_add(Tensor *dst, Tensor *src1, Tensor *src2) {
  int m = src1->shape[0];
  int k = src1->shape[1];
  int n = src2->shape[1];

  int stride_s1 = src1->stride / sizeof(float16_t);
  int stride_s2 = src2->stride / sizeof(float16_t);
  int stride_d = dst->stride / sizeof(float16_t);

  float16_t *psrc1 = (float16_t *)src1->data;
  float16_t *psrc2 = (float16_t *)src2->data;
  float16_t *pdst = (float16_t *)dst->data;

  int vl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j += vl) {
#ifndef FP16_ACC16
      vl = vsetvl_e16m4(n - j);

      vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);

      int offset_dst = i * stride_d + j;
      float16_t *_psrc1_off = psrc1 + i * stride_s1;
      float16_t *_psrc2_off = psrc2 + j;
      for (int kk = 0; kk < k; kk++) {
        float16_t _src1 = *_psrc1_off;
        vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
        _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
        _psrc1_off++;
        _psrc2_off += stride_s2;
      }
      vfloat16m4_t _sum16 = vfncvt_f_f_w_f16m4(_sum, vl);
      vfloat16m4_t _dst_orign = vle16_v_f16m4(pdst + offset_dst, vl);
      _sum16 = vfadd_vv_f16m4(_sum16, _dst_orign, vl);
      vse16_v_f16m4(pdst + offset_dst, _sum16, vl);
#else
      vl = vsetvl_e16m8(n - j);

      vfloat16m8_t _sum = vfmv_v_f_f16m8((float16_t)0.f, vl);

      int offset_dst = i * stride_d + j;
      float16_t *_psrc1_off = psrc1 + i * stride_s1;
      float16_t *_psrc2_off = psrc2 + j;
      for (int kk = 0; kk < k; kk++) {
        float16_t _src1 = *_psrc1_off;
        vfloat16m8_t _src2 = vle16_v_f16m8(_psrc2_off, vl);
        _sum = vfmacc_vf_f16m8(_sum, _src1, _src2, vl);
        _psrc1_off++;
        _psrc2_off += stride_s1;
      }
      vfloat16m8_t _dst_orign = vle16_v_f16m8(pdst + offset_dst, vl);
      _sum = vfadd_vv_f16m8(_sum, _dst_orign, vl);
      vse16_v_f16m8(pdst + offset_dst, _sum, vl);
#endif
    }
  }

  return 0;
}

#endif // __MATMUL_ADD_RVV_H__
