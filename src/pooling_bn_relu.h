#ifndef __POOLING_BN_RELU_H__
#define __POOLING_BN_RELU_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "mme.h"


/*
  padding = 0
  wout % 4 ==0
*/
static inline int maxpool_bn_relu(Tensor *dst, Tensor *src, Tensor *alpha, Tensor *beta, Config *ss)
{
    int kh = ss->kh;
    int kw = ss->kw;
    
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

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
    vl = vsetvl_e16m1(cin);

    asm volatile("vle16.v v28, (%[rs1])"
                :
                : [rs1]"r"(palpha));
    asm volatile("vle16.v v29, (%[rs1])"
                :
                : [rs1]"r"(pbeta));
    asm volatile("vmv.v.x v30, %[rs1]"
                :
                : [rs1]"r"(0x0));
    for (int i = 0; i < hout; i++) {
        int sh0 = i * stride_h;
        for (int j = 0; j < wout; j+=4) {
          int sw0 = j * stride_w;
          asm volatile("vmv.v.x v1, %[rs1]"
                      :
                      : [rs1]"r"(0xfbff));
          asm volatile("vmv.v.x v2, %[rs1]"
                      :
                      : [rs1]"r"(0xfbff));
          asm volatile("vmv.v.x v3, %[rs1]"
                      :
                      : [rs1]"r"(0xfbff));
          asm volatile("vmv.v.x v4, %[rs1]"
                      :
                      : [rs1]"r"(0xfbff));
          for (int m = 0; m < kh; m++) {
            int sy = sh0 + m;
            for (int n = 0; n < kw ; n++) {
              float16_t *_psrc = psrc + sy * win * cin + (sw0 + n) * cin;
              asm volatile("vle16.v v5, (%[rs1])"
                          :
                          : [rs1]"r"(_psrc));
              _psrc+=cin*stride_w;
              asm volatile("vle16.v v6, (%[rs1])"
                          :
                          : [rs1]"r"(_psrc));
              _psrc+=cin*stride_w;
              asm volatile("vle16.v v7, (%[rs1])"
                          :
                          : [rs1]"r"(_psrc));
              _psrc+=cin*stride_w;
              asm volatile("vle16.v v8, (%[rs1])"
                          :
                          : [rs1]"r"(_psrc));
              
              asm volatile("vfmax.vv v1, v1, v5");
              asm volatile("vfmax.vv v2, v2, v6");
              asm volatile("vfmax.vv v3, v3, v7");
              asm volatile("vfmax.vv v4, v4, v8");
            }
          }

          // bn
          asm volatile("vfmul.vv v9,  v1,  v28");
          asm volatile("vfmul.vv v10, v2,  v28");
          asm volatile("vfmul.vv v11, v3,  v28");
          asm volatile("vfmul.vv v12, v4,  v28");

          asm volatile("vfadd.vv v13, v9,  v29");
          asm volatile("vfadd.vv v14, v10, v29");
          asm volatile("vfadd.vv v15, v11, v29");
          asm volatile("vfadd.vv v16, v12, v29");
          // relu
          asm volatile("vfmax.vv v17, v13, v30");
          asm volatile("vfmax.vv v18, v14, v30");
          asm volatile("vfmax.vv v19, v15, v30");
          asm volatile("vfmax.vv v20, v16, v30");
          float16_t *_pdst = pdst + i * wout * cin + j * cin;
          asm volatile("vse16.v v17, (%[rs1])"
                      :
                      : [rs1]"r"(_pdst));
          _pdst+=cin;
          asm volatile("vse16.v v18, (%[rs1])"
                      :
                      : [rs1]"r"(_pdst));
          _pdst+=cin;
          asm volatile("vse16.v v19, (%[rs1])"
                      :
                      : [rs1]"r"(_pdst));
          _pdst+=cin;
          asm volatile("vse16.v v20, (%[rs1])"
                      :
                      : [rs1]"r"(_pdst));
        }
    }

    // for (int i = 0; i < hout; i++) {
    //     int sh0 = i * stride_h;
    //     for (int j = 0; j < wout; j++) {
    //       int sw0 = j * stride_w;
    //       vfloat16m1_t _max = vfmv_v_f_f16m1(0.f, vl);
    //       for (int m = 0; m < (kh - (i == (hout - pad_b))); m++) {
    //         int sy = sh0 + m;
    //         for (int n = 0; n < (kw - (j == (wout - pad_l))); n++) {
    //           int sx = sw0 + n;
    //           unsigned srcOffset = sy * win * cin + sx * cin;
    //           vfloat16m1_t _data = vle16_v_f16m1(psrc + srcOffset, vl);
    //           _max = vfmax_vv_f16m1(_max, _data, vl);
    //         }
    //       }

    //       // bn
    //       vfloat16m1_t _alpha = vle16_v_f16m1(palpha, vl);
    //       vfloat16m1_t _beta = vle16_v_f16m1(pbeta, vl);
    //       vfloat16m1_t _bnout = vfmacc_vv_f16m1(_beta, _max, _alpha, vl);
    //       // relu
    //       float16_t base = 0;
    //       vfloat16m1_t _dst = vfmax_vf_f16m1(_bnout, base, vl);
    //       unsigned dstOffset = i * wout * cin + j * cin;
    //       vse16_v_f16m1(pdst + dstOffset, _dst, vl);
    //     }
    // }

    return 0;
}

#endif