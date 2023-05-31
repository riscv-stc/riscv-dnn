#ifndef __SRC_MATMUL_ADD_RVV_H__
#define __SRC_MATMUL_ADD_RVV_H__

#include "tensor.h"
#include <stddef.h>

//#define FP16_ACC16 1

static inline int matmul_add(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int m = src1->shape[0];
    int k = src1->shape[1];
    int n = src2->shape[1];

    int stride_s1 = src1->stride / sizeof(float16_t);
    int stride_s2 = src2->stride / sizeof(float16_t);
    int stride_d  = dst->stride / sizeof(float16_t);

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;

    int vl;
    for(int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += vl) {
#ifndef FP16_ACC16
            vl = vsetvl_e16m4(n - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vfloat32m8_t _sum;
            vfloat16m4_t _zeros;
            asm volatile("vfwsub.vv %[vd], %[vs1], %[vs1]"
                            :[vd]"=vr"(_sum)
                            :[vs1]"vr"(_zeros));

            int offset_dst = i * stride_d + j;
            float16_t *_psrc1_off = psrc1 + i * stride_s1;
            float16_t *_psrc2_off = psrc2 + j;
            for (int kk = 0; kk < k; kk++) {
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
                _psrc2_off += stride_s2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vfloat16m4_t _sum16;
            vfloat16m4_t _dst_orign;
            asm volatile("vfncvt.f.f.w %[vd], %[vs2]"
                        :[vd]"=vr"(_sum16)
                        :[vs2]"vr"(_sum));
            asm volatile("vle16.v %[vd], (%[rs1])"
                        :[vd]"=vr"(_dst_orign)
                        :[rs1]"r"(pdst+offset_dst));
            asm volatile("vfadd.vv %[vd], %[vs1], %[vs2]"
                        :[vd]"=vr"(_sum16)
                        :[vs1]"vr"(_sum16), [vs2]"vr"(_dst_orign));
            asm volatile("vse16.v %[vd], (%[rs1])"
                        :[vd]"=vr"(_sum16)
                        :[rs1]"r"(pdst+offset_dst));
#else
            vl = vsetvl_e16m8(n - j);

            // vfloat16m8_t _sum = vfmv_v_f_f16m8((float16_t)0.f, vl);
            vfloat16m8_t _sum, _dst_orign;
            asm volatile("vfmv.v.f %[vd], %[rs1]"
                            :[vd]"=vr"(_sum)
                            :[rs1]"f"((float16_t)0.f));

            int offset_dst = i * stride_d + j;
            float16_t *_psrc1_off = psrc1 + i * stride_s1;
            float16_t *_psrc2_off = psrc2 + j;
            for (int kk = 0; kk < k; kk++) {
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
                _psrc2_off += stride_s1;
            }
            // vse16_v_f16m8(pdst+offset_dst, _sum, vl);
            asm volatile("vle16.v %[vd], (%[rs1])"
                        :[vd]"=vr"(_dst_orign)
                        :[rs1]"r"(pdst+offset_dst));
            asm volatile("vfadd.vv %[vd], %[vs1], %[vs2]"
                        :[vd]"=vr"(_sum)
                        :[vs1]"vr"(_sum), [vs2]"vr"(_dst_orign));
            asm volatile("vse16.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_sum)
                            :[rs1]"r"(pdst+offset_dst));
#endif
        }
    }

    return 0;
}

#endif // __SRC_MATMUL_ADD_RVV_H__
