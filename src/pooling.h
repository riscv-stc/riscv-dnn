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

    int hin = src->shape[0];
    int win = src->shape[1];
    int cin = src->shape[2];

    int hout = dst->shape[0];
    int wout = dst->shape[1];
    int cout = dst->shape[2];

    assert(cout == cin);

    int vl;

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
          for (int kc = 0; kc < cin; kc += vl) { // complete vlmax point one time
            vl = vsetvl_e16m8(cin - kc);
            vfloat16m8_t _sum = vfmv_v_f_f16m8(0.f, vl);
            int srcOffset0 = srcOffset + kc;
            for (int m = hStart; m < hEnd; m++) {
              int srcOffset1 = srcOffset0;
              for (int n = wStart; n < wEnd; n++) {
                vfloat16m8_t _data = vle16_v_f16m8(psrc + srcOffset1, vl);
                _sum = vfadd_vv_f16m8(_sum, _data, vl);
                srcOffset1 += cin;
              }
            srcOffset0 += win * cin;
            }
            vfloat16m8_t _avg = vfmul_vf_f16m8(_sum, numValidRecip, vl);
            unsigned dstOffset = i * wout * cin + j * cin + kc;
            vse16_v_f16m8(pdst + dstOffset, _avg, vl);
          }
        }
      }

    return 0;
}


static inline int avgpool_mean(void *dst, void *src, Config *ss)
{
    int kh = ss->kh;
    int kw = ss->kw;
    
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int stride_src = ss->stride_src;

    assert(cout == cin);

    int vl;

    float16_t *psrc = (float16_t *)src;
    float16_t *pdst = (float16_t *)dst;

    float16_t numValidRecip = 1.0 / (hin * win);
    vl = vsetvl_e16m8(2048);
    for (int kc = 0; kc < cin; kc+=vl) {
      asm volatile("vfmv.v.f v0, %[frs1]"
                    :
                    : [frs1]"f"((float16_t)0));
      for (int i = 0; i < hin; i++) {
        for (int j = 0; j < win; j++) {
          asm volatile("vle16.v v8, (%[rs1])"
                      :
                      : [rs1]"r"(psrc + i*win*stride_src/2 + j*stride_src/2 + kc));
          asm volatile("vfadd.vv v0, v0, v8");
        }
      }
      asm volatile("vfmul.vf v16, v0, %[frs1]"
                  :
                  : [frs1]"f"(numValidRecip));
      asm volatile("vse16.v v16, (%[rs1])"
                  :
                  : [rs1]"r"(pdst + kc));
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

    int hin = src->shape[0];
    int win = src->shape[1];
    int cin = src->shape[2];

    int hout = dst->shape[0];
    int wout = dst->shape[1];
    int cout = dst->shape[2];

    assert(cout == cin);

    int vl;

    float16_t *psrc = (float16_t *)src->data;
    float16_t *pdst = (float16_t *)dst->data;

    for (int i = 0; i < hout; i++) {
        int sh0 = i * stride_h;
        for (int j = 0; j < wout; j++) {
          int sw0 = j * stride_w;
          for (int kc = 0; kc < cin; kc += vl) { // complete vlmax point one time
            vl = vsetvl_e16m8(cin - kc);
            vfloat16m8_t _max = vfmv_v_f_f16m8(0.f, vl);
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
                vfloat16m8_t _data = vle16_v_f16m8(psrc + srcOffset, vl);
                if (numValid == 0) {
                  _max = _data;
                } else {
                  _max = vfmax_vv_f16m8(_max, _data, vl);
                }
                numValid++;
              }
            }
            unsigned dstOffset = i * wout * cin + j * cin + kc;
            vse16_v_f16m8(pdst + dstOffset, _max, vl);
          }
        }
      }
    return 0;
}

#endif