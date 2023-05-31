#ifndef __CONV_BN_RELU_H__
#define __CONV_BN_RELU_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"

#ifdef __SPIKE__
static inline int conv_bn_relu_rvm(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tilen)
                        : [rs1]"r"(n-j));
        asm volatile("mwsubc.mm acc0, acc0");

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
            float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tilek)
                            : [rs1]"r"(cin-skc));
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
                asm volatile("mfwma.mm acc0, tr0, tr1");
                // asm volatile("mfma.mm acc1, tr0, tr1");
            }
          }
        }

        asm volatile("mfncvtc.f.fw.m acc1, acc0");

        // batchnormal
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        asm volatile("mfmacccr.mv acc1, v8, v16");
        
        
        // relu
        for (int k = 0; k < tilem; k+=8) {
          int lmul = min(8, tilem-k);
          vl = vsetvl_e16m8(tilen*lmul);
          asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }
      }
    }

    
    return 0;
}

#else
static inline int conv_bn_relu_rvm(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tilen)
                        : [rs1]"r"(n-j));
        // asm volatile("mwsubc.mm acc0, acc0");

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
            float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tilek)
                            : [rs1]"r"(cin-skc));
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
                // asm volatile("mfwma.mm acc0, tr0, tr1");
                asm volatile("mfma.mm acc1, tr0, tr1");
            }
          }
        }

        // asm volatile("mfncvtc.f.fw.m acc1, acc0");

        // batchnormal
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        asm volatile("mfmacccr.mv acc1, v8, v16");
        
        
        // relu
        for (int k = 0; k < tilem; k+=8) {
          int lmul = min(8, tilem-k);
          vl = vsetvl_e16m8(tilen*lmul);
          asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }
      }
    }

    
    return 0;
}

#endif

#define RELU \
  vl = vsetvl_e16m8(tilen*8);\
  for (k = 0; k < (tilem-8); k+=8) { \
    asm volatile("mmvcr.v.m v24, acc8, %[rs2]"  \
                  :                             \       
                  : [rs2]"r"(k));               \
    asm volatile("vfmax.vf v0, v24, %[frs2]"    \
                  :                             \
                  : [frs2]"f"((float16_t)0.f)); \
    asm volatile("msce16.v v0, (%[rs1]), %[rs2]" \
                  :                              \
                  : [rs1]"r"(_pdst+k*stride_d/dataSize), [rs2]"r"(stride_d)); \
  } \
  vl = vsetvl_e16m8((tilem-k)*tilen); \ 
  asm volatile("mmvcr.v.m v24, acc8, %[rs2]" \
                : \
                : [rs2]"r"(k)); \
  asm volatile("vfmax.vf v0, v24, %[frs2]" \
                : \
                : [frs2]"f"((float16_t)0.f)); \
  asm volatile("msce16.v v0, (%[rs1]), %[rs2]" \
                : \
                : [rs1]"r"(_pdst+k*stride_d/dataSize), [rs2]"r"(stride_d)); \


static inline int conv_bn_relu_rvm_batch8(void  *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
#ifdef __SPIKE__
        asm volatile("mwsubc.mm acc0, acc0");
        asm volatile("mwsubc.mm acc1, acc1");
        asm volatile("mwsubc.mm acc2, acc2");
        asm volatile("mwsubc.mm acc3, acc3");
        asm volatile("mwsubc.mm acc4, acc4");
        asm volatile("mwsubc.mm acc5, acc5");
        asm volatile("mwsubc.mm acc6, acc6");
        asm volatile("mwsubc.mm acc7, acc7");
#endif
        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2 +skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
                // batch 0
                float16_t *_psrc1 = _psrc11;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc0, tr0, tr1");
                // batch 1
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc1, tr0, tr1");
                // batch 2
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc2, tr0, tr1");
                // batch 3
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc3, tr0, tr1");
                // batch 4
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc4, tr0, tr1");
                // batch 5
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc5, tr0, tr1");
                // batch 6
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc6, tr0, tr1");
                // batch 7
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc7, tr0, tr1");
            }
          }
        }

        // batchnormal
        int k;
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
#ifdef __SPIKE__
        asm volatile("mfncvtc.f.fw.m acc8, acc0");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc1");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc2");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc3");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc4");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc5");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc6");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc7");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU
#else
        asm volatile("mfaddcr.mv acc8, acc0, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc1, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc2, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc3, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc4, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc5, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc6, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc7, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU
#endif
      }
    }

    
    return 0;
}

static inline int conv_bn_relu_rvm_batch4(void  *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
#ifdef __SPIKE__
        asm volatile("mwsubc.mm acc0, acc0");
        asm volatile("mwsubc.mm acc1, acc1");
        asm volatile("mwsubc.mm acc2, acc2");
        asm volatile("mwsubc.mm acc3, acc3");
#endif
        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2 +skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
                // batch 0
                float16_t *_psrc1 = _psrc11;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc0, tr0, tr1");
                // batch 1
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc1, tr0, tr1");
                // batch 2
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc2, tr0, tr1");
                // batch 3
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc3, tr0, tr1");
            }
          }
        }

        // batchnormal
        int k;
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
#ifdef __SPIKE__
        asm volatile("mfncvtc.f.fw.m acc8, acc0");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc1");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc2");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc3");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

#else
        asm volatile("mfaddcr.mv acc8, acc0, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc1, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc2, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc3, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

#endif
      }
    }

    
    return 0;
}

static inline int conv_bn_relu_rvm_batch2(void  *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int inSize = hin * win * stride_s1 / dataSize;
    int outSize = hout * wout * stride_d / dataSize;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      
      for (int j = 0; j < n; j+=tilen) {
#ifdef __SPIKE__
        asm volatile("mwsubc.mm acc0, acc0");
        asm volatile("mwsubc.mm acc1, acc1");
#endif
        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
            asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
            float16_t *_psrc11 = psrc1 + hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2 +skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
                // batch 0
                float16_t *_psrc1 = _psrc11;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc0, tr0, tr1");
                // batch 1
                _psrc1 += inSize;
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc1+skc), [rs2]"r"(stride_s1));
                asm volatile("mfwma.mm acc1, tr0, tr1");
            }
          }
        }

        // batchnormal
        int k;
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        
        float16_t *_pdst = pdst + i*stride_d/dataSize+j;
#ifdef __SPIKE__
        asm volatile("mfncvtc.f.fw.m acc8, acc0");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfncvtc.f.fw.m acc8, acc1");
        asm volatile("mfmacccr.mv acc8, v8, v16");
        RELU
#else
        asm volatile("mfaddcr.mv acc8, acc0, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

        _pdst += outSize;
        asm volatile("mfaddcr.mv acc8, acc1, v8");
        asm volatile("mfemulcr.mv acc8, acc8, v16");
        RELU

#endif
      }
    }

    
    return 0;
}


// hout * wout > 64*3
static inline int conv_bn_relu_rvm_n(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tilem)
                : [rs1]"r"(m));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    // conv
    for (int j = 0; j < n; j+=tilen) {
      int vl = vsetvl_e16m1(tilen);
      asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
      asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
      int is_end = 0;
      for (int skh = 0; skh < kh; skh++) {
        is_end |= skh == (kh - 1);
        for (int skw = 0; skw < kw; skw++) {
          is_end |= skw == (kw - 1);
          float16_t *_psrc2 = psrc2+(skh*kw*cin+skw*cin)*stride_s2/dataSize+j;
          for (int skc = 0; skc < cin;  skc+=tilek) {
            is_end |= (skc+tilek) >= cin;
            asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                          :
                          :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
            for (int i = 0; i < m; i+=tilem) {
              // asm volatile("msubc.mm acc0, acc0");
              
              int hout_pos = i / wout;
              int wout_pos = i - hout_pos * wout;
              int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
              int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
              asm volatile("msetsk x0, %[rs1], %[rs2]"
                      : 
                      : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
              float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
              
              asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                          : 
                          : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
              asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                          :
                          : [rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
              asm volatile("mfma.mm acc0, tr0, tr1");

              asm volatile("mfaddc.mm acc1, acc0");

              if (is_end) {
                asm volatile("mfmacccr.mv acc1, v8, v16");
                for (int k = 0; k < tilem; k+=8) {
                  int lmul = min(8, tilem-k);
                  vl = vsetvl_e16m8(tilen*lmul);
                  asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                                :
                                : [rs2]"r"(k));
                  asm volatile("vfmax.vf v0, v24, %[frs2]"
                                :
                                : [frs2]"f"((float16_t)0.f));
                  asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                                :
                                : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
                }
              } else {
                asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
              }
            }
          }
        }
      }

    }

    return 0;
}

static inline int conv_bn_relu_rvm_k(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    int vl;
    // conv
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tilem)
                : [rs1]"r"(m));

    for (int skh = 0; skh < kh; skh++) {
      for (int skw = 0; skw < kw; skw++) {
        for (int skc = 0; skc < cin;  skc+=tilek) {
          int flag = (skh == kh -1) && (skw == kw -1) && (skc + tilek == cin);
          int vl = vsetvl_e16m1(64);
          for (int j = 0; j < n; j+=tilen) {
            float16_t *_psrc2 = psrc2+skh*kw*cin*stride_s2/dataSize+skw*cin*stride_s2/dataSize+j;
            asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                          :
                          :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
            
            for (int i = 0; i < m; i+=tilem) {
              // asm volatile("msubc.mm acc0, acc0");
              int hout_pos = i / wout;
              int wout_pos = i - hout_pos * wout;
              int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
              int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
              asm volatile("msetsk x0, %[rs1], %[rs2]"
                      : 
                      : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
              float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
              
              asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                          : 
                          : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
              asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                          :
                          : [rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
              asm volatile("mfma.mm acc0, tr0, tr1");

              asm volatile("mfaddc.mm acc1, acc0");

              if (flag) {
                asm volatile("mfmacccr.mv acc1, v8, v16");
                asm volatile("vle16.v v8, (%[rs1])"
                            : 
                            : [rs1]"r"(palpha + j));
                asm volatile("vle16.v v16, (%[rs1])"
                            : 
                            : [rs1]"r"(pbeta + j));
                vl = vsetvl_e16m8(512);
                for (int k = 0; k < tilem; k+=8) {
                  asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                                :
                                : [rs2]"r"(k));
                  asm volatile("vfmax.vf v0, v24, %[frs2]"
                                :
                                : [frs2]"f"((float16_t)0.f));
                  asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                                :
                                : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
                }
              } else {
                asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                                : 
                                : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
              }
            }
          }
        }
      }

    }

    return 0;
}

static inline int conv_bn_relu_rvm_m(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    int vl = vsetvl_e16m1(64);
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;

      for (int skh = 0; skh < kh; skh++) {
        int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
        for (int skw = 0; skw < kw; skw++) {
          
          int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
          asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
          float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
          
          for (int skc = 0; skc < cin;  skc+=tilek) {
            float16_t *_psrc2 = psrc2+(skh*kw*cin+skw*cin+skc)*stride_s2/dataSize;
            asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tilek)
                            : [rs1]"r"(cin-skc));
            asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
            

            for (int j = 0; j < n; j+=tilen) {
                asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tilen)
                        : [rs1]"r"(n-j));
                // asm volatile("msubc.mm acc0, acc0");
                asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                            : 
                            : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
                
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+j), [rs2]"r"(stride_s2));
                asm volatile("mfma.mm acc0, tr0, tr1");
                asm volatile("mfaddc.mm acc1, acc0");
                if (skh == kh -1 && skw == kw -1 && skc + tilek == cin) {
                  asm volatile("vle16.v v8, (%[rs1])"
                              : 
                              : [rs1]"r"(palpha + j));
                  asm volatile("vle16.v v16, (%[rs1])"
                              : 
                              : [rs1]"r"(pbeta + j));
                  asm volatile("mfmacccr.mv acc1, v8, v16");
  
                  // relu
                  for (int k = 0; k < tilem; k+=8) {
                    int lmul = min(8, tilem-k);
                    vl = vsetvl_e16m8(tilen*lmul);
                    asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                                  :
                                  : [rs2]"r"(k));
                    asm volatile("vfmax.vf v0, v24, %[frs2]"
                                  :
                                  : [frs2]"f"((float16_t)0.f));
                    asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                                  :
                                  : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
                  }
                } else {
                  asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                              : 
                              : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
                }
            }
          }
        }
        
      }
    }

    
    return 0;
}

// m=56*56 k=64 n=256,64, 
// pading=0, stride=1, dilation=1, kh=kw=1, cin=64
// hin=win=hout=wout=56, cin=64, cout=256,64
static inline int conv_bn_relu_rvm_n_k64(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
{
    int dataSize = sizeof(float16_t);
    float16_t *psrc1 = (float16_t *)src;
    float16_t *psrc2 = (float16_t *)weight;
    float16_t *pdst = (float16_t *)dst;
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;

    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = 56 << 16 | 56;
    int minsh = 56 << 16 | 56;
    int mpad = 0;
    int mstdi = 0x01010101;

    int m = 56 * 56;
    int k = 64;
    int n = ss->cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));

    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tilem)
                : [rs1]"r"(m));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(64));
    int vl;
#if 1
    for (int j = 0; j < n; j+=tilen) {
      float16_t *_psrc2 = psrc2+j;
      vl = vsetvl_e16m1(64);
      asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                    :
                    :[rs1]"r"(_psrc2), [rs2]"r"(stride_s2));
      asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
      asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
      vl = vsetvl_e16m8(tilen*8);
      for (int i = 0; i < m; i+=tilem) {
        // asm volatile("msubc.mm acc0, acc0");
        int hout_pos = i / 56;
        int wout_pos = i - hout_pos * 56;
        int hin_pos = hout_pos;
        int win_pos = wout_pos;
        asm volatile("msetsk x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"(wout_pos));
        float16_t *_prsc1 = psrc1+(hin_pos*56+win_pos)*stride_s1/dataSize;

        asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                    :
                    : [rs1]"r"(_prsc1), [rs2]"r"(stride_s1));

        asm volatile("mfma.mm acc0, tr0, tr1");

        asm volatile("mfmacccr.mv acc0, v8, v16");
        
        for (int k = 0; k < tilem; k+=8) {
          asm volatile("mmvcr.v.m v24, acc0, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }
      }
    }
#else 
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / 56;
      int wout_pos = i - hout_pos * 56;
      int hin_pos = hout_pos;
      int win_pos = wout_pos;
      asm volatile("msetsk x0, %[rs1], %[rs2]"
                    : 
                    : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"(wout_pos));
      float16_t *_prsc1 = psrc1+(hin_pos*56+win_pos)*stride_s1/dataSize;
      asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                    :
                    :[rs1]"r"(_prsc1), [rs2]"r"(stride_s1));
      for (int j = 0; j < n; j+=tilen) {
        asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                        :
                        :[rs1]"r"(psrc2+j), [rs2]"r"(stride_s2));
        asm volatile("mfma.mm acc1, tr0, tr1");
        asm volatile("mfmacccr.mv acc0, v8, v16");
        vl = vsetvl_e16m1(64);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        vl = vsetvl_e16m8(64*8);
        for (int k = 0; k < tilem; k+=8) {
          asm volatile("mmvcr.v.m v24, acc0, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }
      }
    }
#endif 
    return 0;
}

// 1*1, padding =0, stride =1, dilation =1
static inline int conv_bn_relu_rvm_1x1_remain(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
{
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = 0;
    int mstdi = 0x01010101;

    int m = hout * wout;
    int k = cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tilem)
                : [rs1]"r"(m));
    // conv
    int i=0;
    for (i = 0; i < (m-64); i+=tilem) {
      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      int hin_pos = hout_pos;
      int win_pos = wout_pos;
      asm volatile("msetsk x0, %[rs1], %[rs2]"
                    : 
                    : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"(wout_pos));
      float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
      for (int j = 0; j < n; j+=tilen) {
        // asm volatile("msubc.mm acc1, acc1");
        float16_t *_psrc2 = psrc2+j;
        for (int skc = 0; skc < cin;  skc+=tilek) {
          asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
          
          asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
          asm volatile("mfma.mm acc1, tr0, tr1");
        }

        // batchnormal
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        asm volatile("mfmacccr.mv acc1, v8, v16");
        
        
        // relu
        vl = vsetvl_e16m8(512);
        for (int k = 0; k < tilem; k+=8) {
          asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }

      }
    }

    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tilem)
                : [rs1]"r"(m-i));
    int hout_pos = i / wout;
    int wout_pos = i - hout_pos * wout;
    int hin_pos = hout_pos;
    int win_pos = wout_pos;
    asm volatile("msetsk x0, %[rs1], %[rs2]"
                  : 
                  : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"(wout_pos));
    float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
    for (int j = 0; j < n; j+=tilen) {
      // asm volatile("msubc.mm acc1, acc1");
      float16_t *_psrc2 = psrc2+j;
      for (int skc = 0; skc < cin;  skc+=tilek) {
        asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                    :
                    :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
        
        asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                    :
                    :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
        asm volatile("mfma.mm acc1, tr0, tr1");
      }

      // batchnormal
      int vl = vsetvl_e16m1(tilen);
      asm volatile("vle16.v v8, (%[rs1])"
                  : 
                  : [rs1]"r"(palpha + j));
      asm volatile("vle16.v v16, (%[rs1])"
                  : 
                  : [rs1]"r"(pbeta + j));
      asm volatile("mfmacccr.mv acc1, v8, v16");
      
      
      // relu
      vl = vsetvl_e16m8(tilem*tilen);
      asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                      :
                      : [rs2]"r"(0));
      asm volatile("vfmax.vf v0, v24, %[frs2]"
                      :
                      : [frs2]"f"((float16_t)0.f));
      asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                      :
                      : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));

    }
    
    return 0;
}

static inline int conv_bn_relu_rvm_1x1(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
{
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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = 0;
    int mstdi = 0x01010101;

    int m = hout * wout;
    int k = cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(tilem)
                : [rs1]"r"(m-i));
      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;
      int hin_pos = hout_pos;
      int win_pos = wout_pos;
      asm volatile("msetsk x0, %[rs1], %[rs2]"
                    : 
                    : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"(wout_pos));
      float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
      for (int j = 0; j < n; j+=tilen) {
        // asm volatile("msubc.mm acc0, acc0");
        float16_t *_psrc2 = psrc2+j;
        for (int skc = 0; skc < cin;  skc+=tilek) {
          asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
          
          asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
          asm volatile("mfma.mm acc1, tr0, tr1");
        }

        // batchnormal
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        asm volatile("mfmacccr.mv acc1, v8, v16");
        
        
        // relu
        vl = vsetvl_e16m8(512);
        for (int k = 0; k < tilem; k+=8) {
          asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }
      }
    }

    
    return 0;
}


// 3*3, padding =1, dilation =1
static inline int conv_bn_relu_rvm_3x3_remain(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = 0x01010101;
    int mstdi = 0x01010000| stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m));
    // conv
    int i=0;
    for (i = 0; i < (m-64); i+=tilem) {
      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;

      for (int j = 0; j < n; j+=tilen) {
        // asm volatile("msubc.mm acc0, acc0");

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h- 1 + skh;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - 1 + skw ;
            asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw) << 16 | wout_pos));
            float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+(skh*kw*cin+skw*cin)*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
                asm volatile("mfma.mm acc1, tr0, tr1");
            }
          }
        }

        // batchnormal
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        asm volatile("mfmacccr.mv acc1, v8, v16");
        
        
        // relu
        vl = vsetvl_e16m8(512);
        for (int k = 0; k < (tilem); k+=8) {
          asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }

      }
    }

    //remain
    asm volatile("msettilem %[rd], %[rs1]"
                  : [rd]"=r"(tilem)
                  : [rs1]"r"(m-i));

    int hout_pos = i / wout;
    int wout_pos = i - hout_pos * wout;

    for (int j = 0; j < n; j+=tilen) {
      // asm volatile("msubc.mm acc0, acc0");

      for (int skh = 0; skh < kh; skh++) {
        int hin_pos = hout_pos * stride_h- 1 + skh;
        for (int skw = 0; skw < kw; skw++) {
          int win_pos = wout_pos * stride_w - 1 + skw ;
          asm volatile("msetsk x0, %[rs1], %[rs2]"
                      : 
                      : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw) << 16 | wout_pos));
          float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
          float16_t *_psrc2 = psrc2+(skh*kw*cin+skw*cin)*stride_s2/dataSize+j;
          for (int skc = 0; skc < cin;  skc+=tilek) {
              asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                          :
                          :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
              
              asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                          :
                          :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
              asm volatile("mfma.mm acc1, tr0, tr1");
          }
        }
      }

      // batchnormal
      int vl = vsetvl_e16m1(tilen);
      asm volatile("vle16.v v8, (%[rs1])"
                  : 
                  : [rs1]"r"(palpha + j));
      asm volatile("vle16.v v16, (%[rs1])"
                  : 
                  : [rs1]"r"(pbeta + j));
      asm volatile("mfmacccr.mv acc1, v8, v16");
      
      
      // relu
      vl = vsetvl_e16m8(tilem*tilen);
      asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                    :
                    : [rs2]"r"(0));
      asm volatile("vfmax.vf v0, v24, %[frs2]"
                    :
                    : [frs2]"f"((float16_t)0.f));
      asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                    :
                    : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
    }
    return 0;
}

// 3*3, padding =1, dilation =1
static inline int conv_bn_relu_rvm_3x3(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

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
    float16_t *palpha = (float16_t *)alpha;
    float16_t *pbeta = (float16_t *)beta;


    int stride_s1 = ss->stride_src;
    int stride_s2 = ss->stride_ker;
    int stride_d = ss->stride_dst;

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = 0x01010101;
    int mstdi = 0x01010000| stride_h << 8 | stride_w;

    int m = hout * wout;
    int k = kh *kw * cin;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(tilen)
                : [rs1]"r"(n));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(cin));
    
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                  : [rd]"=r"(tilem)
                  : [rs1]"r"(m-i));
      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;

      for (int j = 0; j < n; j+=tilen) {
        // asm volatile("msubc.mm acc0, acc0");

        for (int skh = 0; skh < kh; skh++) {
          int hin_pos = hout_pos * stride_h- 1 + skh;
          for (int skw = 0; skw < kw; skw++) {
            int win_pos = wout_pos * stride_w - 1 + skw ;
            asm volatile("msetsk x0, %[rs1], %[rs2]"
                        : 
                        : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw) << 16 | wout_pos));
            float16_t *_prsc1 = psrc1+(hin_pos*win+win_pos)*stride_s1/dataSize;
            float16_t *_psrc2 = psrc2+(skh*kw*cin+skw*cin)*stride_s2/dataSize+j;
            for (int skc = 0; skc < cin;  skc+=tilek) {
                asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
                asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
                asm volatile("mfma.mm acc1, tr0, tr1");
            }
          }
        }

        // batchnormal
        int vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        asm volatile("mfmacccr.mv acc1, v8, v16");
        
        
        // relu
        vl = vsetvl_e16m8(512);
        for (int k = 0; k < tilem; k+=8) {
          asm volatile("mmvcr.v.m v24, acc1, %[rs2]"
                        :
                        : [rs2]"r"(k));
          asm volatile("vfmax.vf v0, v24, %[frs2]"
                        :
                        : [frs2]"f"((float16_t)0.f));
          asm volatile("msce16.v v0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+(i+k)*stride_d/dataSize+j), [rs2]"r"(stride_d));
        }
        

      }
    }

    
    return 0;
}

#endif //__CONV_BN_RELU_H__