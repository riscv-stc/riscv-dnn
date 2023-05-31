#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

//#define FP16_ACC16 1

static inline int matmul(void *dst, void *src1, void *src2, int m , int k, int n)
{
    int h1 = m;
    int w1 = k;

    int h2 = k;
    int w2 = n;

    int hout = m;
    int wout = n;

    float16_t *psrc1 = (float16_t *)src1;
    float16_t *psrc2 = (float16_t *)src2;
    float16_t *pdst = (float16_t *)dst;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
#ifndef FP16_ACC16
            vl = vsetvl_e16m4(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vfloat32m8_t _sum;
            vfloat16m4_t _zeros;
            asm volatile("vfwsub.vv %[vd], %[vs1], %[vs1]"
                            :[vd]"=vr"(_sum)
                            :[vs1]"vr"(_zeros));

            int offset_dst = i * w2 + j;
            float16_t *_psrc1_off = psrc1 + i * w1;
            float16_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                float16_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vfloat16m4_t _src2;
                asm volatile("vle16.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_src2)
                            :[rs1]"r"(_psrc2_off));
                asm volatile("vfwmacc.vf %[vd], %[rs1], %[vs2]"
                            :[vd]"+vr"(_sum)
                            :[rs1]"f"(_src1), [vs2]"vr"(_src2));
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vfloat16m4_t _sum16;
            asm volatile("vfncvt.f.f.w %[vd], %[vs2]"
                            :[vd]"=vr"(_sum16)
                            :[vs2]"vr"(_sum));
            asm volatile("vse16.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_sum16)
                            :[rs1]"r"(pdst+offset_dst));
#else
            vl = vsetvl_e16m8(w2 - j);

            // vfloat16m8_t _sum = vfmv_v_f_f16m8((float16_t)0.f, vl);
            vfloat16m8_t _sum;
            asm volatile("vfmv.v.f %[vd], %[rs1]"
                            :[vd]"=vr"(_sum)
                            :[rs1]"f"((float16_t)0.f));

            int offset_dst = i * w2 + j;
            float16_t *_psrc1_off = psrc1 + i * w1;
            float16_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                float16_t _src1 = *_psrc1_off;
                // vfloat16m8_t _src2 = vle16_v_f16m8(_psrc2_off, vl);
                //_sum = vfmacc_vf_f16m8(_sum, _src1, _src2, vl);
                vfloat16m8_t _src2;
                asm volatile("vle16.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_src2)
                            :[rs1]"r"(_psrc2_off));
                // _sum = vfmacc_vf_f16m8(_sum, _src1, _src2, vl);
                asm volatile("vfmacc.vf %[vd], %[rs1], %[vs2]"
                            :[vd]"+vr"(_sum)
                            :[rs1]"f"(_src1), [vs2]"vr"(_src2));
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m8(pdst+offset_dst, _sum, vl);
            asm volatile("vse16.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_sum)
                            :[rs1]"r"(pdst+offset_dst));
#endif
        }
    }

    return 0;
}

#endif // __SRC_MATMUL_H__
