#ifndef __CONV_BN_RELU_H__
#define __CONV_BN_RELU_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"

static inline int conv_bn_relu_rvm(Tensor *dst, Tensor *src, Tensor *weight, Tensor *alpha, Tensor *beta, Config *ss)
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
    float16_t *psrc1 = (float16_t *)src->data;
    float16_t *psrc2 = (float16_t *)weight->data;
    float16_t *pdst = (float16_t *)dst->data;
    float16_t *palpha = (float16_t *)alpha->data;
    float16_t *pbeta = (float16_t *)beta->data;


    int stride_s1 = src->stride;
    int stride_s2 = weight->stride;
    int stride_d = dst->stride;

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

#endif //__CONV_BN_RELU_H__