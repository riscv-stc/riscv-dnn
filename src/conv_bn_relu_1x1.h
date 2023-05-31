#ifndef __CONV_BN_RELU_1X1_H__
#define __CONV_BN_RELU_1X1_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"

static inline int conv_bn_relu_1x1(Tensor *dst, Tensor *src, Tensor *weight, Tensor *alpha, Tensor *beta, Config *ss)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int kh = 1;
    int kw = 1;

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
    int mpad = 0;
    int mstdi = 1 << 24 | 1 << 16 | stride_h << 8 | stride_w;

    int m = hout * wout;
    int n = cout;

    int tilem, tilen, tilek;

    asm volatile("msetoutsh x0, %[rs1], %[rs2]"
                : 
                : [rs1]"r"(moutsh), [rs2]"r"(mstdi));
    asm volatile("msetinsh x0, %[rs1], %[rs2]"
                :
                : [rs1]"r"(minsh), [rs2]"r"(mpad));
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(tilek)
                : [rs1]"r"(64));
    
    int vl = vsetvl_e16m1(64);
    // conv
    for (int i = 0; i < m; i+=tilem) {
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(m-i));

      int hout_pos = i / wout;
      int wout_pos = i - hout_pos * wout;

      int hin_pos = hout_pos * stride_h;
      int win_pos = wout_pos * stride_w;
      
      asm volatile("msetsk x0, %[rs1], %[rs2]"
                  : 
                  : [rs1]"r"(hin_pos <<  16 | (win_pos & 0xffff)), [rs2]"r"(wout_pos));
      float16_t *_prsc1 = psrc1+hin_pos*win*stride_s1/dataSize+win_pos*stride_s1/dataSize;
      for (int j = 0; j < n; j+=tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tilen)
                        : [rs1]"r"(n-j));
        // asm volatile("mwsubc.mm acc0, acc0");

        for (int skc = 0; skc < cin;  skc+=tilek) {
              asm volatile("mlufae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(_prsc1+skc), [rs2]"r"(stride_s1));
                
              asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+skc*stride_s2/dataSize), [rs2]"r"(stride_s2));
              asm volatile("mfma.mm acc1, tr0, tr1");
        }

        // asm volatile("mfncvtc.f.fw.m acc1, acc0");

        // batchnormal
        
        asm volatile("vle16.v v8, (%[rs1])"
                    : 
                    : [rs1]"r"(palpha + j));
        asm volatile("vle16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(pbeta + j));
        // asm volatile("mfmacccr.mv acc1, v8, v16");
        asm volatile("mfemulcr.mv acc0, acc1, v8");
        asm volatile("mfaddcr.mv acc0, acc0, v16");
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