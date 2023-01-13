#ifndef __POOLING_BN_RELU_H__
#define __POOLING_BN_RELU_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "mme.h"

static inline int maxpool_bn_relu(Tensor *dst, Tensor *src, Tensor *alpha, Tensor *beta, Config *ss)
{
    int kh = ss->kh;
    int kw = ss->kw;
    
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int hin = src->shape[0];
    int win = src->shape[1];
    int cin = src->shape[2];

    int hout = dst->shape[0];
    int wout = dst->shape[1];
    int cout = dst->shape[2];

    assert(cout == cin);

    int vl;

    float16_t *psrc = (float16_t *)src->data;
    float16_t *palpha = (float16_t *)alpha->data;
    float16_t *pbeta = (float16_t *)beta->data;
    float16_t *pdst = (float16_t *)dst->data;

    for (int i = 0; i < hout; i++) {
        int sh0 = i * stride_h;
        for (int j = 0; j < wout; j++) {
          int sw0 = j * stride_w;
          for (int kc = 0; kc < cin; kc += vl) { // complete vlmax point one time
            vl = vsetvl_e16m1(cin - kc);
            vfloat16m1_t _max = vfmv_v_f_f16m1(0.f, vl);
            int numValid = 0;
            for (int m = 0; m < kh; m++) {
              int sy = sh0 + m;
              if (sy < pad_t || sy >= pad_t + hin) {
                continue;
              }
              for (int n = 0; n < kw; n++) {
                int sx = sw0 + n;
                if (sx < pad_l || sx >= pad_l + win) {
                  continue;
                }
                unsigned srcOffset = (sy - pad_t) * win * cin + (sx - pad_l) * cin + kc;
                vfloat16m1_t _data = vle16_v_f16m1(psrc + srcOffset, vl);
                if (numValid == 0) {
                  _max = _data;
                } else {
                  _max = vfmax_vv_f16m1(_max, _data, vl);
                }
                numValid++;
              }
            }

            // bn
            vfloat16m1_t _alpha = vle16_v_f16m1(palpha + kc, vl);
            vfloat16m1_t _beta = vle16_v_f16m1(pbeta + kc, vl);
            vfloat16m1_t _bnout = vfmacc_vv_f16m1(_beta, _max, _alpha, vl);
            // relu
            float16_t base = 0;
            vfloat16m1_t _dst = vfmax_vf_f16m1(_bnout, base, vl);
            unsigned dstOffset = i * wout * cin + j * cin + kc;
            vse16_v_f16m1(pdst + dstOffset, _dst, vl);
          }
        }
      }
    return 0;
}

#endif