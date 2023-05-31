#ifndef __CONV_IM2COL_ADD_H__
#define __CONV_IM2COL_ADD_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"

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
            }
          }
        }
        asm volatile("mfncvtc.f.fw.m acc1, acc0");
        asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
        asm volatile("mfaddc.mm acc1, acc0");
        asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
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

    int mtype = 1;
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));

    int moutsh = hout << 16 | wout;
    int minsh = hin << 16 | win;
    int mpad = pad_t << 24 | pad_b << 16 | pad_l << 8 | pad_r;
    int mstdi = dilation_h << 24 | dilation_w << 16 | stride_h << 8 | stride_w;

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

        int hin_pos = hout_pos * stride_h - pad_t + skh * dilation_h;
        int win_pos = wout_pos * stride_w - pad_l + skw * dilation_w;
        asm volatile("msetsk x0, %[rs1], %[rs2]"
                      : 
                      : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"((skw * dilation_w) << 16 | wout_pos));
        float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
        for (int skc = 0; skc < cin;  skc+=tilek) {
            asm volatile("msettilek %[rd], %[rs1]"
                        : [rd]"=r"(tilek)
                        : [rs1]"r"(cin-skc));
            asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                        :
                        :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
            asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                        :
                        :[rs1]"r"(psrc2+skc*stride_s2/dataSize+j), [rs2]"r"(stride_s2));
            asm volatile("mfwma.mm acc0, tr0, tr1");
        }
        asm volatile("mfncvtc.f.fw.m acc1, acc0");
        asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
        asm volatile("mfaddc.mm acc1, acc0");
        asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                    : 
                    : [rs1]"r"(pdst+i*stride_d/dataSize+j), [rs2]"r"(stride_d));
      }
    }

    
    return 0;
}





#endif
