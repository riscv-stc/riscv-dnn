#ifndef __CONV_IM2COL_H__
#define __CONV_IM2COL_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"

static inline int conv_im2col(void *dst, void *src, void *weight, Config *ss)
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

    msettype(E16, M1, BA);

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    msetoutsh(moutsh, mstdi);
    msetinsh(minsh, mpad);

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
            float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                tilek = msettilek(cin-skc);
                mfloat16m1_t tr0 = mlufae16_m(_prsc1+skc, stride_s1);
                mfloat16m1_t tr1 = mlbe16_m(_psrc2+skc*stride_s2/dataSize, stride_s2);
                acc0 = mfma_mm(acc0, tr0, tr1);
            }
          }
        }
        msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
      }
    }

    
    return 0;
}
//batch
static inline int conv_rvm_batch8(void *dst, void *src, void *weight, Config *ss)
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

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;

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
    // tilen = msettilen(n);
    // tilek = msettilek(cin);

    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16m1_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
        acc0 = mfsub_mm(acc0, acc0);
        acc1 = mfsub_mm(acc1, acc1);
        acc2 = mfsub_mm(acc2, acc2);
        acc3 = mfsub_mm(acc3, acc3);
        acc4 = mfsub_mm(acc4, acc4);
        acc5 = mfsub_mm(acc5, acc5);
        acc6 = mfsub_mm(acc6, acc6);
        acc7 = mfsub_mm(acc7, acc7);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2 + skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
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
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
        msce16_m(acc0, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc1, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc2, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc3, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc4, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc5, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc6, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc7, _pdst, stride_d);
      }
    }

    
    return 0;
}

static inline int conv_rvm_batch4(void *dst, void *src, void *weight, Config *ss)
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

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;

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
    // tilen = msettilen(n);
    // tilek =  msettilek(cin);
    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);
        mfloat16m1_t acc0, acc1, acc2, acc3;
        acc0 = mfsub_mm(acc0, acc0);
        acc1 = mfsub_mm(acc1, acc1);        
        acc2 = mfsub_mm(acc2, acc2);        
        acc3 = mfsub_mm(acc3, acc3);        

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2 + skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
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
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
        msce16_m(acc0, _pdst, stride_d);
        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc1");
        msce16_m(acc1, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc2, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc3, _pdst, stride_d);
      }
    }

    
    return 0;
}

static inline int conv_rvm_batch2(void *dst, void *src, void *weight, Config *ss)
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

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;

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
    // tilen = msettilen(n);
    // tilek = msettilek(cin);

    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        tilen = msettilen(n-j);        
        mfloat16m1_t acc0, acc1;
        acc0 = mfsub_mm(acc0, acc0);
        acc1 = mfsub_mm(acc1, acc1);

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2 + skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
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
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
        msce16_m(acc0, _pdst, stride_d);
        _pdst += outSize;
        msce16_m(acc1, _pdst, stride_d);
      }
    }

    
    return 0;
}

// m=56*56 k=3*3*64 n=64
static inline int conv_rvm_n(void *dst, void *src, void *weight, Config *ss)
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
    tilem = msettilem(m);
    tilek = msettilek(cin);
    tilen = msettilen(n);

    for (int j = 0; j < n; j+=tilen) {
      for (int skh = 0; skh < kh; skh++) {
        for (int skw = 0; skw < kw; skw++) {
          float16_t *_psrc2 = psrc2+(skh*kw*cin+skw*cin)*stride_s2/dataSize+j;
          for (int skc = 0; skc < cin;  skc+=tilek) {
            mfloat16m1_t tr1 = mlbe16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);
            for (int i = 0; i < m; i+=tilem) {
              int hout_pos = i / wout;
              int wout_pos = i - hout_pos * wout;
              int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
              int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
              msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
              float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
              mfloat16m1_t acc0 = mlce16_m1(pdst+i*stride_d/dataSize+j, stride_d);
              mfloat16m1_t tr0 = mlufae16_m(_prsc1+skc, stride_s1);
              acc0 = mfma_mm(acc0, tr0, tr1);                
              msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);          
            }
          }
        }
        
      }
    }

    return 0;
}

static inline int conv_rvm_k(void *dst, void *src, void *weight, Config *ss)
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

    msettype(E16, M1, BA);

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    msetoutsh(moutsh, mstdi);
    msetinsh(minsh, mpad);
    tilen = msettilen(n);
    tilek = msettilek(cin);
    tilem = msettiem(m);
    int vl;
    for (int skh = 0; skh < kh; skh++) {
      for (int skw = 0; skw < kw; skw++) {
        for (int skc = 0; skc < cin;  skc+=tilek) {

          for (int j = 0; j < n; j+=tilen) {
            
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            mfloat16m1_t tr1 = mlbe16_m1(_psrc2+skc*stride_s2/dataSize, stride_s2);
            for (int i = 0; i < m; i+=tilem) {
              // asm volatile("mwsubc.mm acc0, acc0");
              
              int hout_pos = i / wout;
              int wout_pos = i - hout_pos * wout;
              int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
              int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
              msetsk(hin_pos << 16 | (win_pos & 0xffff), (skc * dilation_w) << 16 | wout_pos);
              float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
              mfloat16m1_t acc0 = mlce16_m1(pdst+i*stride_d/dataSize+j, stride_d);              
              mfloat16m1_t tr0 = mlufae16_m(_prsc1+skc, stride_s1);
              acc0 = mfma_mm(acc0, tr0, tr1);
              msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
            }
          }
        }
        
      }
    }

    
    return 0;
}


// m < 64
static inline int conv_rvm_m(void *dst, void *src, void *weight, Config *ss)
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

    msettype(E16, M1, BA);

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;
    nsetinsh(minsh, mpad);
    msetoutsh(moutsh, mstdi);

    for (int i = 0; i < m; i+=tilem) {
      tilem = msettilem(m-i);

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;

      for (int skh = 0; skh < kh; skh++) {
        int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
        for (int skw = 0; skw < kw; skw++) {
          
          int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
          msetsk(hin_pos <<  16 | (win_pos & 0xffff), (skw * dilation_w) << 16 | wout_pos);
          float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
          
          for (int skc = 0; skc < cin;  skc+=tilek) {
            float16_t *_psrc2 = psrc2+(skh*kw*cin+skw*cin+skc)*stride_s2/dataSize;
            tilek = msettilk(cin-skc);
            mfloat16m1_t tr0 = mlufae16_m(_prsc1+skc, stride_s1);            

            for (int j = 0; j < n; j+=tilen) {
                tilen = msettilen(n-j);
                // asm volatile("msubc.mm acc0, acc0");
                mfloat16m1_t acc0 = mlae16_m1(pdst+i*stride_d/dataSize+j, stride_d);
                mfloat16m1_t tr1 = mlbe16_m1(_psrc2+j, stride_s2);                
                acc0 = mfma_mm(acc0, tr0, tr1);
                msce16_m(acc0, pdst+i*stride_d/dataSize+j, stride_d);
            }
          }
        }
        
      }
    }

    
    return 0;
}

static inline int conv(void *dst, void *src, void *weight, Config *ss)
{
  conv_im2col(dst, src, weight, ss);
}

#endif
