#ifndef __POOLING_H__
#define __POOLING_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "mme.h"

static inline int avgpool(Tensor *dst, Tensor *src, Config *ss)
{
    int kh = ss->kh;
    int kw = ss->kw;
    
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int hin = src->h;
    int win = src->w;
    int cin = src->cin;

    int hout = dst->h;
    int wout = dst->w;
    int cout = dst->cin;

    assert(cout == cin);

    int vlmax = VLENB / 2;

    float16_t *psrc = (float16_t *)src->data;
    float16_t *pdst = (float16_t *)dst->data;

    for (int i = 0; i < hout; i++) {
        int sh0 = i * stride_h;
        int hStart = sh0 > pad_t? 0 : pad_t - sh0;
        int hEnd = sh0 + kh > pad_t + hin? pad_t + hin - sh0 : kh;
        int hValid = hEnd - hStart;
        for (int j = 0; j < wout; j++) {
          int sw0 = j * stride_w;
          int wStart = sw0 > pad_l? 0 : pad_l - j;
          int wEnd = sw0 + kw > pad_l + win? pad_l + win - sw0 : kw;
          int srcOffset =(sh0 + hStart - pad_t) * win * cin + (sw0 + wStart- pad_l) * cin;
          float numValidRecip = 1.0 / (hValid * (wEnd - wStart));
          for (int kc = 0; kc < cin; kc += vlmax) { // complete vlmax point one time
            int vl = min(vlmax, cin - kc);
            vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);
            int srcOffset0 = srcOffset + kc;
            for (int m = hStart; m < hEnd; m++) {
              int srcOffset1 = srcOffset0;
              for (int n = wStart; n < wEnd; n++) {
                vfloat16m1_t _data = vle16_v_f16m1(psrc + srcOffset1, vl);
                _sum = vfadd_vv_f16m1(_sum, _data, vl);
                srcOffset1 += cin;
              }
            srcOffset0 += win * cin;
            }
            vfloat16m1_t _avg = vfmul_vf_f16m1(_sum, numValidRecip, vl);
            unsigned dstOffset = i * wout * cin + j * cin + kc;
            vse16_v_f16m1(pdst + dstOffset, _avg, vl);
          }
        }
      }

    return 0;
}

static inline int maxpool(Tensor *dst, Tensor *src, Config *ss)
{
    int kh = ss->kh;
    int kw = ss->kw;
    
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int hin = src->h;
    int win = src->w;
    int cin = src->cin;

    int hout = dst->h;
    int wout = dst->w;
    int cout = dst->cin;

    assert(cout == cin);

    int vlmax = VLENB / 2;

    float16_t *psrc = (float16_t *)src->data;
    float16_t *pdst = (float16_t *)dst->data;

    for (int i = 0; i < hout; i++) {
        int sh0 = i * stride_h;
        for (int j = 0; j < wout; j++) {
          int sw0 = j * stride_w;
          for (int kc = 0; kc < cin; kc += vlmax) { // complete vlmax point one time
            int vl = min(vlmax, cin - kc);
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
            unsigned dstOffset = i * wout * cin + j * cin + kc;
            vse16_v_f16m1(pdst + dstOffset, _max, vl);
          }
        }
      }
    return 0;
}

#endif