#ifndef __POOLING_BN_RELU_H__
#define __POOLING_BN_RELU_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "mme.h"


/*
  padding = 0,1,0,1
  wout % 4 ==0
*/
static inline int maxpool_bn_relu(void *dst, void *src, void *alpha, void *beta, Config *ss)
{
    int kh = ss->kh;
    int kw = ss->kw;
    
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_r = ss->right;
    int pad_b = ss->bottom; 

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vl;

    float16_t *psrc = (float16_t *)src;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    float16_t *pdst = (float16_t *)dst;
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
        int last_valid = (i==(hout-1) && pad_b==1)? 1 : 0;
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
          for (int m = 0; m < kh-last_valid; m++) {
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
              
              
              asm volatile("vfmax.vv v1, v1, v5");
              asm volatile("vfmax.vv v2, v2, v6");
              asm volatile("vfmax.vv v3, v3, v7");
              if (!(pad_r==1 && n==(kw-1) && j==(wout-4))) {
                _psrc+=cin*stride_w;
                asm volatile("vle16.v v8, (%[rs1])"
                            :
                            : [rs1]"r"(_psrc));
                asm volatile("vfmax.vv v4, v4, v8");
              }
              
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

    return 0;
}

#endif