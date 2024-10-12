#ifndef __CONV_IM2COL_ADD_H__
#define __CONV_IM2COL_ADD_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"
#include <riscv_matrix.h>

static inline int conv_im2col_add(void *dst, void *src, void *weight, Config *ss)
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


    int dataSize = sizeof(float16_t);
    float16_t *psrc1 = (float16_t *)src;
    float16_t *psrc2 = (float16_t *)weight;
    float16_t *pdst = (float16_t *)dst;

    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    msettypei(0x1);
    msettypehi(0x1);

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    msetinsh(minsh, mpad);
    msetoutsh(moutsh, mstdi);

    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16_t acc0;
        acc0 = mfsub_mm(acc0, acc0);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                tilek = msettilek(cin-skc);
                mfloat16_t tr0 = mlufa_m(_prsc1+skc, stride_s1);
                mfloat16_t tr1 = mlb_m(_psrc2+skc*stride_s2/dataSize, stride_s2);
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
          }
        }
        mfloat16_t acc1 = mlce16_m1(pdst+i*stride_d/dataSize+j, stride_d);
        acc0 = mfadd_mm(acc1, acc0);
        msc_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
      }
    }

    
    return 0;
}


static inline int im2col_add(void *dst, void *src, void *weight, Config *ss, int skh, int skw)
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


    int dataSize = sizeof(float16_t);
    float16_t *psrc1 = (float16_t *)src;
    float16_t *psrc2 = (float16_t *)weight;
    float16_t *pdst = (float16_t *)dst;

    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    msettypei(0x1);
    msettypehi(0x1);

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = cin;
    int n = cout;

    int tilem, tilen, tilek;

    msetinsh(minsh, mpad);
    msetoutsh(moutsh, mstdi);

    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16_t acc0;
        acc0 = mfsub_mm(acc0, acc0);

        int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
        int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
        msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
        float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
        for (int skc = 0; skc < cin;  skc+=tilek) {
            tilek = msettilek(cin-skc);
            mfloat16_t tr0 = mlufa_m(_prsc1+skc, stride_s1);
            mfloat16_t tr1 = mlb_m(psrc2+skc*stride_s2/dataSize+j, stride_s2);
            acc0 = mfma_mm(acc0, tr0, tr1);
        }
        mfloat16_t acc1 = mlce16_m1(pdst+i*stride_d/dataSize+j, stride_d);
        acc0 = mfadd_mm(acc1, acc0);
        msc_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
      }
    }

    
    return 0;
}





#endif
