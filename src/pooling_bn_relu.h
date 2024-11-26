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
    vfloat16m1_t v28 = vle16_v_f16m1(palpha, vl);
    vfloat16m1_t v29 = vle16_v_f16m1(pbeta, vl);
    vfloat16m1_t v30 = vle16_v_f16m1(0.f, vl);

    for (int i = 0; i < hout; i++) {
        int sh0 = i * stride_h;
        int last_valid = (i==(hout-1) && pad_b==1)? 1 : 0;
        for (int j = 0; j < wout; j+=4) {
          int sw0 = j * stride_w;
          
          vfloat16m1_t v1 = vfmv_v_f_f16m1(-65500.0, vl);
          vfloat16m1_t v2 = vfmv_v_f_f16m1(-65500.0, vl);
          vfloat16m1_t v3 = vfmv_v_f_f16m1(-65500.0, vl);
          vfloat16m1_t v4 = vfmv_v_f_f16m1(-65500.0, vl);
          for (int m = 0; m < kh-last_valid; m++) {
            int sy = sh0 + m;
            for (int n = 0; n < kw ; n++) {
              float16_t *_psrc = psrc + sy * win * cin + (sw0 + n) * cin;
              vfloat16m1_t v5 = vle16_v_f16m1(_psrc, vl);
              _psrc+=cin*stride_w;
              vfloat16m1_t v6 = vle16_v_f16m1(_psrc, vl);
              _psrc+=cin*stride_w;
              vfloat16m1_t v7 = vle16_v_f16m1(_psrc, vl);
              
              v1 = vfmax_vv_f16m1(v1, v5, vl);
              v2 = vfmax_vv_f16m1(v2, v6, vl);
              v3 = vfmax_vv_f16m1(v3, v7, vl);
              if (!(pad_r==1 && n==(kw-1) && j==(wout-4))) {
                _psrc+=cin*stride_w;
                vfloat16m1_t v8 = vle16_v_f16m1(_psrc, vl);
                v4 = vfmax_vv_f16m1(v4, v8);
              }
              
            }
          }

          // bn
          vfloat16m1_t v9  = vfmul_vv_f16m1(v1, v28, vl);
          vfloat16m1_t v10 = vfmul_vv_f16m1(v2, v28, vl);
          vfloat16m1_t v11 = vfmul_vv_f16m1(v3, v28, vl);
          vfloat16m1_t v12 = vfmul_vv_f16m1(v4, v28, vl);
          vfloat16m1_t v13 = vfadd_vv_f16m1(v9, v29, vl);
          vfloat16m1_t v14 = vfadd_vv_f16m1(v10, v29, vl);
          vfloat16m1_t v15 = vfadd_vv_f16m1(v11, v29, vl);
          vfloat16m1_t v16 = vfadd_vv_f16m1(v12, v29, vl);
          // relu
          vfloat16m1_t v17 = vfmax_vv_f16m1(v13, v30, vl);
          vfloat16m1_t v18 = vfmax_vv_f16m1(v14, v30, vl);
          vfloat16m1_t v19 = vfmax_vv_f16m1(v15, v30, vl);
          vfloat16m1_t v20 = vfmax_vv_f16m1(v16, v30, vl);
          float16_t *_pdst = pdst + i * wout * cin + j * cin;
          vse16_v_f16m1(_pdst, v17, vl);
          _pdst+=cin;
          vse16_v_f16m1(_pdst, v18, vl);
          _pdst+=cin;
          vse16_v_f16m1(_pdst, v19, vl);
          _pdst+=cin;
          vse16_v_f16m1(_pdst, v20, vl);
        }
    }

    return 0;
}

#endif