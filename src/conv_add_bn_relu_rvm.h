#ifndef __CONV_ADD_BN_RELU_H__
#define __CONV_ADD_BN_RELU_H__

#include "tensor.h"
#include <stddef.h>

#include "matmul.h"
#include "mme.h"

static inline int conv_add_bn_relu_rvm(Tensor *dst, Tensor *addout, Tensor *src,
                                       Tensor *weight, Tensor *addsrc,
                                       Tensor *alpha, Tensor *beta,
                                       Config *ss) {
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

  int dataSize = sizeof(float16_t);
  float16_t *psrc1 = (float16_t *)src->data;
  float16_t *psrc2 = (float16_t *)weight->data;
  float16_t *paddout = (float16_t *)addout->data;
  float16_t *pdst = (float16_t *)dst->data;
  float16_t *paddsrc = (float16_t *)addsrc->data;
  float16_t *palpha = (float16_t *)alpha->data;
  float16_t *pbeta = (float16_t *)beta->data;

  int stride_s1 = src->stride;
  int stride_s2 = weight->stride;
  int stride_d = dst->stride;
  int stride_addsrc = addsrc->stride;
  int stride_addout = addout->stride;

  int moutsh = hout << 16 | wout;
  int minsh = hin << 16 | win;
  int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
  int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

  int m = hout * wout;
  int k = kh * kw * cin;
  int n = cout;

  int tilem, tilen, tilek;

  msetinsh(minsh, mpad);
  msetoutsh(moutsh, mstdi);
  msetpadval(0);
  // conv
  for (int i = 0; i < m; i += tilem) {
    SET_MBA0_FP16();
    tilem = msettilem(m - i);

    int hout_pos = i / wout;
    int wout_pos = i - hout_pos * wout;

    for (int j = 0; j < n; j += tilen) {
      SET_MBA0_FP16();
      tilen = msettilen(n - j);
      mfloat16_t add =
          mlc_m(paddsrc + i * stride_addsrc / dataSize + j, stride_addsrc);
      mfloat16_t mbeta = mlc_m(pbeta + j, stride_d);
      mfloat16_t malpha = mlc_m(palpha + j, stride_d);
      mfloat16_t zero;
      zero = mfsub_mm(zero, zero);
      SET_MBA0_FP32();
      mfloat32_t acc0;
      acc0 = mfsub_f_mm(acc0, acc0);

      for (int skh = 0; skh < kh; skh++) {
        int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
        for (int skw = 0; skw < kw; skw++) {
          int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
          msetsk(hin_pos << 16 | (win_pos & 0xFFFF),
                 (skw * dilation_w) << 16 | wout_pos);
          float16_t *_prsc1 = psrc1 + hin_pos * win * stride_s1 / dataSize +
                              win_pos * stride_s1 / dataSize;
          float16_t *_psrc2 = psrc2 + skh * kw * cin * stride_s2 / dataSize +
                              skw * cin * stride_s2 / dataSize + j;
          for (int skc = 0; skc < cin; skc += tilek) {
            SET_MBA0_FP16();
            tilek = msettilek(cin - skc);
            mfloat16_t tr0 = mlufa_m(_prsc1 + skc, stride_s1);
            mfloat16_t tr1 =
                mlb_m(_psrc2 + skc * stride_s2 / dataSize, stride_s2);
            SET_MBA0_FP16_FP32();
            acc0 = mfwma_mm(acc0, tr0, tr1);
          }
        }
      }

      SET_MBA0_FP32_FP16();
      mfloat16_t acc1 = mfncvt_f_fw_m(acc0);

      SET_MBA0_FP16();
      // add
      acc1 = mfadd_mm(acc1, add);
      msc_m(acc1, paddout + i * stride_addout / dataSize + j, stride_addout);

      // batchnormal

      mbeta = mbccr_m(mbeta);
      malpha = mbccr_m(malpha);
      acc1 = mfmul_mm(acc1, malpha);
      acc1 = mfadd_mm(acc1, mbeta);

      // relu
      acc1 = mfmax_mm(acc1, zero);
      msc_m(acc1, pdst + i * stride_d / dataSize + j, stride_d);
    }
  }

  return 0;
}

#endif //__CONV_ADD_BN_RELU_H__
