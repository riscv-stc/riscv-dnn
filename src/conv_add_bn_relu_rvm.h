#ifndef __CONV_ADD_BN_RELU_H__
#define __CONV_ADD_BN_RELU_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"

#ifdef __SPIKE__
static inline int conv_add_bn_relu_rvm(void *dst, void *addout, void *src, void *addsrc, void *weight,  void *alpha, void *beta, Config *ss)
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
    float16_t *paddout = (float16_t *)addout;
    float16_t *pdst = (float16_t *)dst;
    float16_t *paddsrc = (float16_t *)addsrc;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    
    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;
    int stride_addout = ss->stride_addout;

    msettype(E16, M1, BA);

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
    // conv
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16m1_t acc0;
        acc0 = mfsub_mm(acc0, acc0);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                tilek = msettilek(cin-skc);
                mfloat16m1_t tr0 = mlufae16_m(_psrc1+skc, stride_s1);
                mfloat16m1_t tr1 = mlbe16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
          }
        }

        // add
        mfloat16m1_t acc1 = mlce16_m1(paddsrc+i*stride_addsrc/dataSize+j, stride_addsrc);
        acc0 = mfadd_mm(acc0, acc1);
        msce16_m(acc0, paddout+i*stride_addout/dataSize+j, stride_addout);
        
        // batchnormal
        mfloat16m1_t mbeta = mlce16_m1(palpha + j, stride_addout);
        mfloat16m1_t malpha = mlce16_m1(pbeta + j, stride_addout);
        mbeta = mbccr_m(mbeta);
        malpha = mbccr_m(malpha);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);

        // relu
        mfloat16m1_t zero;
        zero = mfsub_mm(zero, zero);
        acc0 = mfmax_mm(acc0, zero);
        msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
      }
    }

    
    return 0;
}

static inline int conv_add_bn_relu_last(void *dst, void *src, void *addsrc, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *paddsrc = (float16_t *)addsrc;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    
    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;

    msettype(E16, M1, BA);

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

    // conv
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m - i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16m1_t acc0;
        acc0 = mfsub_mm(acc0, acc0);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                tilek = msettilek(cin-skc);
                mfloat16m1_t tr0 = mlufae16_m(_psrc1+skc, stride_s1);
                mfloat16m1_t tr1 = mlbe16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);                
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
          }
        }

        // add
        mfloat16m1_t acc1 = mlce16_m1(paddsrc+i*stride_addsrc/dataSize+j, stride_addsrc);
        acc0 = mfadd_mm(acc0, acc1);
        
        // batchnormal
        mfloat16m1_t mbeta = mlce16_m1(pbeta + j, stride_d);
        mfloat16m1_t malpha = mlce16_m1(palpha + j, stride_d);
        mbeta = mbccr_m(mbeta);
        malpha = mbccr_m(malpha);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);
                
        // relu
        mfloat16m1_t zero;
        zero = mfsub_mm(zero, zero);
        acc0 = mfmax_mm(acc0, zero);
        msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
      }
    }

    
    return 0;
}


#else 
static inline int conv_add_bn_relu_rvm(void *dst, void *addout, void *src, void *addsrc, void *weight,  void *alpha, void *beta, Config *ss)
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
    float16_t *paddout = (float16_t *)addout;
    float16_t *pdst = (float16_t *)dst;
    float16_t *paddsrc = (float16_t *)addsrc;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    
    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;
    int stride_addout = ss->stride_addout;

    msettype(E16, M1, BA);

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

    // conv
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16m1_t acc0;
        acc0 = mfsub_mm(acc0, acc0);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                tilek = msettilek(cin-skc);
                mfloat16m1_t tr0 = mlce16_m1(_psrc1+skc, stride_s1);
                mfloat16m1_t tr1 = mlce16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
          }
        }

        // add
        mfloat16m1_t acc1 = mlce16_m1(paddsrc+i*stride_addsrc/dataSize+j, stride_addsrc);
        acc0 = mfadd_mm(acc0, acc1);
        msce16_m(acc0, paddout+i*stride_addout/dataSize+j, stride_addout);
        
        // batchnormal
        mfloat16m1_t mbeta = mlce16_m1(pbeta+j, stride_d);
        mfloat16m1_t malpha = mlce16_m1(palpha+j, stride_d);
        mbeta = mbccr_m(mbeta);
        malpha = mbccr_m(malpha);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);

        // relu
        mfloat16m1_t zero;
        zero = mfsub_mm(zero, zero);
        acc0 = mfmax_mm(acc0, zero);
        msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
      }
    }

    
    return 0;
}

static inline int conv_add_bn_relu_last(void *dst, void *src, void *addsrc, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *paddsrc = (float16_t *)addsrc;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    
    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;

    msettype(E16, M1, BA);

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

    // conv
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16m1_t acc0;
        acc0 = mfsub_mm(acc0, acc0);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                tilek = msettilek(cin-skc);
                mfloat16m1_t tr0 = mlufae16_m(_psrc1+skc, stride_s1);
                mfloat16m1_t tr1 = mlbe16_m(_psrc2+skc*stride_s2/dataSize, stride_s2);
                acc0 = mfmfa_mm(acc0, tr0, tr1);
            }
          }
        }

        // add
        mfloat16m1_t acc1 = mlce16_m1(paddsrc+i*stride_addsrc/dataSize+j, stride_addsrc);
        acc0 = mfadd_mm(acc0, acc1);
        
        // batchnormal
        mfloat16m1_t mbeta = mlce16_m1(pbeta+j, stride_d);
        mfloat16m1_t malpha = mlce16_m1(palpha+j, stride_d);
        mbeta = mbccr_m(mbeta);
        malpha = mbccr_m(malpha);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);

        // relu
        mfloat16m1_t zero;
        zero = mfsub_mm(zero, zero);
        acc0 = vfmax_mm(acc0, zero);
        int vl = vsetvl_e16m1(tilen);
        msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
      }
    }

    
    return 0;
}

#endif

static inline int conv_add_bn_relu_rvm_batch8(void *dst, void *addout, void *src, void *addsrc, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *paddout = (float16_t *)addout;
    float16_t *pdst = (float16_t *)dst;
    float16_t *paddsrc = (float16_t *)addsrc;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    
    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;
    int stride_addout = ss->stride_addout;

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;
    int addsrcSize = hout * wout * stride_addsrc / dataSize;
    int addoutSize = hout * wout * stride_addout / dataSize;

    msettype(E16, M1, BA);

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

    // conv
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        float16_t *_paddsrc = paddsrc + i*stride_addsrc/dataSize+j;
        mfloat16m1_t acc0 = mlce16_m(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc1 = mlce16_m(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc2 = mlce16_m(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc3 = mlce16_m(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc4 = mlce16_m(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc5 = mlce16_m(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc6 = mlce16_m(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc7 = mlce16_m(_paddsrc, stride_addsrc);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
              tilek = msettilek(cin-skc);
              mfloat16m1_t tr1 = mlbe16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);
              // batch 0
              float16_t *_psrc1 = _psrc11;
              mfloat16m1_t tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc0 = mfma_mm(acc0, tr0, tr1);
              // batch 1
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc1 = mfma_mm(acc1, tr0, tr1);
              // batch 2
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc2 = mfma_mm(acc2, tr0, tr1);
              // batch 3
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc3 = mfma_mm(acc3, tr0, tr1);
              // batch 4
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc4 = mfma_mm(acc4, tr0, tr1);
              // batch 5
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc5 = mfma_mm(acc5, tr0, tr1);
              // batch 6
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc6 = mfma_mm(acc6, tr0, tr1);
              // batch 7
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc7 = mfma_mm(acc7, tr0, tr1);
            }
          }
        }

        mfloat16m1_t zero;
        zero = mfsub_mm(zero, zero);
        mfloat16m1_t mbeta = mlce16_m1(pbeta + j, stride_d);
        mfloat16m1_t malpha = mlce16_m1(palpha + j, stride_d);
        mbeta = mbccr_m(mbeta);
        malpha = mbccr_m(malpha);

        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
        float16_t *_paddout = paddout + i*stride_addout/dataSize+j;
        // batch 0
        msce16_m(acc0, _paddout, stride_addout);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);
        acc0 = mfmax_mm(acc0, zero);
        msce16_m(acc0, _pdst, stride_d);
        // batch 1
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc1, _paddout, stride_addout);
        acc1 = mfmul_mm(acc1, malpha);
        acc1 = mfadd_mm(acc1, mbeta);
        acc1 = mfmax_mm(acc1, zero);
        msce16_m(acc1, _pdst, stride_d);
        // batch 2
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc2, _paddout, stride_addout);
        acc2 = mfmul_mm(acc2, malpha);
        acc2 = mfadd_mm(acc2, mbeta);
        acc2 = mfmax_mm(acc2, zero);
        msce16_m(acc2, _pdst, stride_d);
        // batch 3
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc3, _paddout, stride_addout);
        acc3 = mfmul_mm(acc3, malpha);
        acc3 = mfadd_mm(acc3, mbeta);
        acc3 = mfmax_mm(acc3, zero);
        msce16_m(acc3, _pdst, stride_d);
        // batch 4
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc4, _paddout, stride_addout);
        acc4 = mfmul_mm(acc4, malpha);
        acc4 = mfadd_mm(acc4, mbeta);
        acc4 = mfmax_mm(acc4, zero);
        msce16_m(acc4, _pdst, stride_d);
        // batch 5
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc5, _paddout, stride_addout);
        acc5 = mfmul_mm(acc5, malpha);
        acc5 = mfadd_mm(acc5, mbeta);
        acc5 = mfmax_mm(acc5, zero);
        msce16_m(acc5, _pdst, stride_d);
        // batch 6
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc6, _paddout, stride_addout);
        acc6 = mfmul_mm(acc6, malpha);
        acc6 = mfadd_mm(acc6, mbeta);
        acc6 = mfmax_mm(acc6, zero);
        msce16_m(acc6, _pdst, stride_d);
        // batch 7
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc7, _paddout, stride_addout);
        acc7 = mfmul_mm(acc7, malpha);
        acc7 = mfadd_mm(acc7, mbeta);
        acc7 = mfmax_mm(acc7, zero);
        msce16_m(acc7, _pdst, stride_d);
      }
    }

    
    return 0;
}

static inline int conv_add_bn_relu_rvm_batch4(void *dst, void *addout, void *src, void *addsrc, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *paddout = (float16_t *)addout;
    float16_t *pdst = (float16_t *)dst;
    float16_t *paddsrc = (float16_t *)addsrc;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    
    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;
    int stride_addout = ss->stride_addout;

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;
    int addsrcSize = hout * wout * stride_addsrc / dataSize;
    int addoutSize = hout * wout * stride_addout / dataSize;

    msettype(E16, M1, BA);

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

    // conv
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        float16_t *_paddsrc = paddsrc + i*stride_addsrc/dataSize+j;
        mfloat16m1_t acc0 = mlce16_m1(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc1 = mlce16_m1(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc2 = mlce16_m1(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc3 = mlce16_m1(_paddsrc, stride_addsrc);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);

            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
              tilek = msettilek(cin-skc);
              mfloat16m1_t tr1 = mlbe16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);
              // batch 0
              float16_t *_psrc1 = _psrc11;
              mfloat16m1_t tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc0 = mfma_mm(acc0, tr0, tr1);
              // batch 1
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc1 = mfma_mm(acc1, tr0, tr1);
              // batch 2
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc2 = mfma_mm(acc2, tr0, tr1);
              // batch 3
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc3 = mfma_mm(acc3, tr0, tr1);
            }
          }
        }

        mfloat16m1_t zero;
        zero = mfsub_mm(zero, zero);
        mfloat16m1_t mbeta = mlce16_m1(pbeta+j, stride_d);
        mfloat16m1_t malpha = mlce16_m1(palpha+j, stride_d);
        mbeta = mbccr_m(mbeta);
        malpha = mbccr_m(malpha);
        // batch 0
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
        float16_t *_paddout = paddout + i*stride_addout/dataSize+j;
        msce16_m(acc0, _paddout, stride_addout);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);
        acc0 = mfmax_mm(acc0, zero);
        // batch 1
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc1, _paddout, stride_addout);
        acc1 = mfmul_mm(acc1, malpha);
        acc1 = mfadd_mm(acc1, mbeta);
        acc1 = mfmax_mm(acc1, zero);
        // batch 2
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc2, _paddout, stride_addout);
        acc2 = mfmul_mm(acc2, malpha);
        acc2 = mfadd_mm(acc2, mbeta);
        acc2 = mfmax_mm(acc2, zero);
        // batch 3
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc3, _paddout, stride_addout);
        acc3 = mfmul_mm(acc3, malpha);
        acc3 = mfadd_mm(acc3, mbeta);
        acc3 = mfmax_mm(acc3, zero);
      }
    }

    
    return 0;
}

static inline int conv_add_bn_relu_rvm_batch2(void *dst, void *addout, void *src, void *addsrc, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *paddout = (float16_t *)addout;
    float16_t *pdst = (float16_t *)dst;
    float16_t *paddsrc = (float16_t *)addsrc;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;
    
    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;
    int stride_addout = ss->stride_addout;

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;
    int addsrcSize = hout * wout * stride_addsrc / dataSize;
    int addoutSize = hout * wout * stride_addout / dataSize;

    msettype(E16, M1, BA);

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

    // conv
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        float16_t *_paddsrc = paddsrc + i*stride_addsrc/dataSize+j;
        mfloat16m1_t acc0 = mlce16_m1(_paddsrc, stride_addsrc);
        _paddsrc += addsrcSize;
        mfloat16m1_t acc1 = mlce16_m1(_paddsrc, stride_addsrc);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
              tilek = msettilek(cin-skc);
              mfloat16m1_t tr1 = mlbe16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);
              // batch 0
              float16_t *_psrc1 = _psrc11;
              mfloat16m1_t tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc0 = mfma_mm(acc0, tr0, tr1);
              // batch 1
              _psrc1 += inSize;
              tr0 = mlufae16_m(_psrc1+skc, stride_s1);
              acc1 = mfma_mm(acc1, tr0, tr1);
            }
          }
        }
        mfloat16m1_t zero;
        zero = mfsub_mm(zero, zero);
        mfloat16m1_t mbeta = mlce16_m1(pbeta+j, stride_d);
        mfloat16m1_t malpha = mlce16_m1(palpha+j, stride_d);
        mbeta = mbccr_m(mbeta);
        malpha = mbccr_m(malpha);

        // batch 0
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
        float16_t *_paddout = paddout + i*stride_addout/dataSize+j;
        msce16_m(acc0, _paddout, stride_addout);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);
        acc0 = mfmax_mm(acc0, zero);
        // batch 1
        _pdst += outSize;
        _paddout += addoutSize;
        msce16_m(acc0, _paddout, stride_addout);
        acc0 = mfmul_mm(acc0, malpha);
        acc0 = mfadd_mm(acc0, mbeta);
        acc0 = mfmax_mm(acc0, zero);
      }
    }

    
    return 0;
}

#endif //__CONV_ADD_BN_RELU_H__