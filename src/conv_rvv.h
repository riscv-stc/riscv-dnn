#ifndef __CONV_H__
#define __CONV_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "mme.h"

static inline int conv(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vlmax = VLENB * 4 / 2;

    float16_t *psrc = (float16_t *)src->data;
    // padding the input data
    if (pad_l + pad_r + pad_t + pad_b) {
        float16_t *psrcPad = (float16_t *)srcPad->data;
        psrcPad += pad_t * (win + pad_l + pad_r) * cin + pad_l * cin;  
        for (int i = 0; i < hin; i++) {
            for (int j = 0; j < win * cin; j += vlmax) {
                unsigned vl = min(vlmax, win * cin - j);
                vfloat16m4_t _data = vle16_v_f16m4(psrc, vl);
                psrc += vl;
                vse16_v_f16m4(psrcPad, _data, vl);
                psrcPad += vl;
            }
            psrcPad += (pad_r + pad_l) * cin;
        }
        psrc = (float16_t *)srcPad->data;
    }

    hin = hin + pad_t + pad_b;
    win = win + pad_l + pad_r;

    float16_t *pweight = (float16_t *)weight->data;
    float16_t *pdst = (float16_t *)dst->data;

    
    for (int i = 0; i < hout; i++) {
      for (int j = 0; j < wout; j++) { 
        for (int k = 0; k < cout; k += vlmax) { // complete  vlmax one time
          unsigned vl_out = min(vlmax, cout - k);
          int offset_dst = i * wout * cout + j * cout + k;
          vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl_out);

          int offset_src1 = i * stride_h * win * cin + j * stride_w * cin;
          for (int m = 0; m < kh; m++) {
            int offset_src = offset_src1;
            for (int n = 0; n < kw; n++) {
              int offset_weight = m * kw * cout * cin + n * cout * cin + k;
              float16_t *_psrc_off = psrc + offset_src;
              float16_t *_psrc_weight = pweight + offset_weight;
              for (int l = 0; l < cin; l++) {
                float16_t _src = *_psrc_off;
                _psrc_off++;
                vfloat16m4_t _weight = vle16_v_f16m4(_psrc_weight, vl_out);
                _psrc_weight += cout;
                _sum = vfwmacc_vf_f32m8(_sum, _src, _weight, vl_out);
              } // l
            offset_src += dilation_w * cin;
            } // n
          offset_src1 += dilation_h * win * cin;
          } // m

          vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m4(_sum, vl_out), vl_out);
 
        } // k
      } // j
    } // i

    return 0;
}

#endif