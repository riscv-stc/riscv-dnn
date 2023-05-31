#ifndef __RELU_H__
#define __RELU_H__

#include "tensor.h"
#include "mme.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "exp.h"

static inline int layernorm_f32_all(void *dst, void *src, void *gamma, void *beta, Config *ss)
{   
    int hw = ss->hin;
    int c = ss->win;

    int datasize = sizeof(float16_t);

    int stride_src = ss->stride_src;
    int stride_dst = ss->stride_dst;
    
    float16_t *psrc = (float16_t *)src;
    float16_t *pdst = (float16_t *)dst;
    float16_t *pgamma = (float16_t *)gamma;
    float16_t *pbeta = (float16_t *)beta;

    float redsum[hw];
    float varsum[hw];

    const float epsilon=1e-5;

    int vl;
    int tilem, tilek, tilen;

    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(1));
    // -mean
    for (int i = 0; i < hw; i += tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tilen)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e32m2(tilen);
        asm volatile("vmv.v.x v0, x0");
        for (int j = 0; j < c; j += tilem) {
            asm volatile("msettilem %[rd], %[rs1]"
                        : [rd]"=r"(tilem)
                        : [rs1]"r"(c-j));
            asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(1));
            asm volatile("mlcte16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfwcvtc.fw.f.m acc1, acc0");
            asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(2));
            for (int k = 0; k < tilem; k++) {
                asm volatile("mmvcr.v.m v8, acc1, %[rs2]"
                            :
                            : [rs2]"r"(k));
                asm volatile("vfadd.vv v0 ,v8, v0");
            }
        }
        asm volatile("vfdiv.vf v0, v0, %[fs2]"
                    :
                    : [fs2]"f"((float)c));
        asm volatile("vfrsub.vf v0, v0, %[fs2]"
                    :
                    : [fs2]"f"(0.f));
        asm volatile("vse32.v v0, (%[rs1])"
                    : 
                    : [rs1]"r"(redsum+i));
    }

    // fang cha
    for (int i = 0; i < hw; i += tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tilen)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e32m2(tilen);
        asm volatile("vle32.v v0, (%[rs1])"
                    :
                    : [rs1]"r"(redsum+i));
        asm volatile("vmv.v.x v2, x0");
        for (int j = 0; j < c; j += tilem) {
            asm volatile("msettilem %[rd], %[rs1]"
                        : [rd]"=r"(tilem)
                        : [rs1]"r"(c-j));
            asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(1));
            asm volatile("mlcte16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfwcvtc.fw.f.m acc1, acc0");
            asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(2));
            asm volatile("mfaddcr.mv acc1, acc1, v0");
            for (int k = 0; k < tilem; k++) {
                asm volatile("mmvcr.v.m v4, acc1, %[rs2]"
                            :
                            : [rs2]"r"(k));
                asm volatile("vfmul.vv v8, v4, v4");
                asm volatile("vfadd.vv v2, v2, v8");
            }
        }
        asm volatile("vfdiv.vf v10, v2, %[fs2]"
                    :
                    : [fs2]"f"((float)c));
        asm volatile("vfadd.vf v12, v10, %[fs2]"
                    :
                    : [fs2]"f"(epsilon));
        asm volatile("vfsqrt.v v14, v12");
        asm volatile("vfrec7.v v16, v14");
        asm volatile("vse32.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(varsum+i));
    }

    for (int i = 0; i < hw; i+=tilem) { 
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e32m2(tilem);
        asm volatile("vle32.v v0, (%[rs1])"
                    :
                    : [rs1]"r"(redsum+i));
        asm volatile("vle32.v v2, (%[rs1])"
                    :
                    : [rs1]"r"(varsum+i));
        
        for (int j = 0; j < c; j+=tilen) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tilen)
                        : [rs1]"r"(c-j));
            vl = vsetvl_e16m1(tilen);
            asm volatile("vle16.v v8, (%[rs1])"
                        :
                        : [rs1]"r"(pgamma+j));
            asm volatile("vfwcvt.f.f.v v4, v8");
            asm volatile("vle16.v v8, (%[rs1])"
                        :
                        : [rs1]"r"(pbeta+j));
            asm volatile("vfwcvt.f.f.v v6, v8");
            asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(1));
            asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfwcvtc.fw.f.m acc0, acc1");
            asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(2));
            vl = vsetvl_e32m2(tilen);
            asm volatile("mfaddcc.mv acc1, acc0, v0"); // x + (-u)
            asm volatile("mfemulcc.mv acc0, acc1, v2"); // 
            asm volatile("mfemulcr.mv acc1, acc0, v4");
            asm volatile("mfaddcr.mv acc0, acc1, v6");
            asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(1));
            asm volatile("mfncvtc.f.fw.m acc1, acc0");
            asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+i*stride_dst/datasize+j), [rs2]"r"(stride_dst));
        }

    }


    return 0;
}


static inline int layernorm(void *dst, void *src, void *gamma, void *beta, Config *ss)
{   
    int hw = ss->hin;
    int c = ss->win;

    int datasize = sizeof(float16_t);

    int stride_src = ss->stride_src;
    int stride_dst = ss->stride_dst;
    
    float16_t *psrc = (float16_t *)src;
    float16_t *pdst = (float16_t *)dst;
    float16_t *pgamma = (float16_t *)gamma;
    float16_t *pbeta = (float16_t *)beta;

    float16_t redsum[hw];
    float16_t varsum[hw];

    const float epsilon=1e-5;

    int vl;
    int tilem, tilek, tilen;

    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(1));
    // -mean
    for (int i = 0; i < hw; i += tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tilen)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e32m2(tilen);
        asm volatile("vmv.v.x v0, x0");
        for (int j = 0; j < c; j += tilem) {
            asm volatile("msettilem %[rd], %[rs1]"
                        : [rd]"=r"(tilem)
                        : [rs1]"r"(c-j));
            asm volatile("mlcte16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfwcvtc.fw.f.m acc1, acc0");
            for (int k = 0; k < tilem; k++) {
                asm volatile("mwmvcr.v.m v8, acc1, %[rs2]"
                            :
                            : [rs2]"r"(k));
                asm volatile("vfadd.vv v0 ,v8, v0");
            }
        }
        asm volatile("vfdiv.vf v0, v0, %[fs2]"
                    :
                    : [fs2]"f"((float)c));
        asm volatile("vfrsub.vf v0, v0, %[fs2]"
                    :
                    : [fs2]"f"(0.f));
        vl = vsetvl_e16m1(tilen);
        asm volatile("vfncvt.f.f.w v2, v0");
        asm volatile("vse16.v v2, (%[rs1])"
                    : 
                    : [rs1]"r"(redsum+i));
    }

    // fang cha
    for (int i = 0; i < hw; i += tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tilen)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v0, (%[rs1])"
                    :
                    : [rs1]"r"(redsum+i));
        vl = vsetvl_e32m2(tilen);
        asm volatile("vmv.v.x v2, x0");
        for (int j = 0; j < c; j += tilem) {
            asm volatile("msettilem %[rd], %[rs1]"
                        : [rd]"=r"(tilem)
                        : [rs1]"r"(c-j));
            asm volatile("mlcte16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfwcvtc.fw.f.m acc1, acc0");
            asm volatile("mfwaddcr.mv acc1, acc1, v0");
            for (int k = 0; k < tilem; k++) {
                asm volatile("mwmvcr.v.m v4, acc1, %[rs2]"
                            :
                            : [rs2]"r"(k));
                asm volatile("vfmul.vv v8, v4, v4");
                asm volatile("vfadd.vv v2, v2, v8");
            }
        }
        asm volatile("vfdiv.vf v10, v2, %[fs2]"
                    :
                    : [fs2]"f"((float)c));
        asm volatile("vfadd.vf v12, v10, %[fs2]"
                    :
                    : [fs2]"f"(epsilon));
        asm volatile("vfsqrt.v v14, v12");
        asm volatile("vfrec7.v v16, v14");
        vl = vsetvl_e16m1(tilen);
        asm volatile("vfncvt.f.f.w v2, v16");
        asm volatile("vse16.v v2, (%[rs1])"
                    : 
                    : [rs1]"r"(varsum+i));
    }

    for (int i = 0; i < hw; i+=tilem) { 
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e16m1(tilem);
        asm volatile("vle16.v v0, (%[rs1])"
                    :
                    : [rs1]"r"(redsum+i));
        asm volatile("vle16.v v2, (%[rs1])"
                    :
                    : [rs1]"r"(varsum+i));
        
        for (int j = 0; j < c; j+=tilen) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tilen)
                        : [rs1]"r"(c-j));
            vl = vsetvl_e16m1(tilen);
            asm volatile("vle16.v v4, (%[rs1])"
                        :
                        : [rs1]"r"(pgamma+j));
            asm volatile("vle16.v v6, (%[rs1])"
                        :
                        : [rs1]"r"(pbeta+j));
            
            asm volatile("mlce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfwcvtc.fw.f.m acc0, acc1");
            asm volatile("mfwaddcc.mv acc1, acc0, v0"); // x + (-u)
            asm volatile("mfwemulcc.mv acc0, acc1, v2"); // 
            asm volatile("mfwemulcr.mv acc1, acc0, v4");
            asm volatile("mfwaddcr.mv acc0, acc1, v6");
            asm volatile("mfncvtc.f.fw.m acc1, acc0");
            asm volatile("msce16.m acc1, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+i*stride_dst/datasize+j), [rs2]"r"(stride_dst));
        }

    }


    return 0;
}


static inline int layernorm_fp16(void *dst, void *src, void *gamma, void *beta, Config *ss)
{   
    int hw = ss->hin;
    int c = ss->win;

    int datasize = sizeof(float16_t);

    int stride_src = ss->stride_src;
    int stride_dst = ss->stride_dst;
    
    float16_t *psrc = (float16_t *)src;
    float16_t *pdst = (float16_t *)dst;
    float16_t *pgamma = (float16_t *)gamma;
    float16_t *pbeta = (float16_t *)beta;

    float16_t redsum[hw];
    float16_t varsum[hw];

    float16_t epsilon=1e-5;

    int vl;
    int tilem, tilek, tilen;

    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(1));
    // -mean
    
    for (int i = 0; i < hw; i += tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tilen)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e16m1(tilen);
        asm volatile("vmv.v.x v0, x0");
        for (int j = 0; j < c; j += tilem) {
            asm volatile("msettilem %[rd], %[rs1]"
                        : [rd]"=r"(tilem)
                        : [rs1]"r"(c-j));
            asm volatile("mlcte16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            for (int k = 0; k < tilem; k++) {
                asm volatile("mmvcr.v.m v8, acc0, %[rs2]"
                            :
                            : [rs2]"r"(k));
                asm volatile("vfadd.vv v0 ,v8, v0");
            }
        }
        asm volatile("vfdiv.vf v0, v0, %[fs2]"
                    :
                    : [fs2]"f"((float16_t)c));
        asm volatile("vfrsub.vf v0, v0, %[fs2]"
                    :
                    : [fs2]"f"((float16_t)0.f));

        asm volatile("vse16.v v0, (%[rs1])"
                    : 
                    : [rs1]"r"(redsum+i));
    }

    // fang cha
    for (int i = 0; i < hw; i += tilen) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tilen)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e16m1(tilen);
        asm volatile("vle16.v v0, (%[rs1])"
                    :
                    : [rs1]"r"(redsum+i));
        asm volatile("vmv.v.x v2, x0");
        for (int j = 0; j < c; j += tilem) {
            asm volatile("msettilem %[rd], %[rs1]"
                        : [rd]"=r"(tilem)
                        : [rs1]"r"(c-j));
            asm volatile("mlcte16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfaddcr.mv acc1, acc0, v0");
            for (int k = 0; k < tilem; k++) {
                asm volatile("mmvcr.v.m v4, acc1, %[rs2]"
                            :
                            : [rs2]"r"(k));
                asm volatile("vfmul.vv v8, v4, v4");
                asm volatile("vfadd.vv v2, v2, v8");
            }
        }
        asm volatile("vfdiv.vf v10, v2, %[fs2]"
                    :
                    : [fs2]"f"((float16_t)c));
        asm volatile("vfadd.vf v12, v10, %[fs2]"
                    :
                    : [fs2]"f"(epsilon));
        asm volatile("vfsqrt.v v14, v12");
        asm volatile("vfrec7.v v16, v14");
        asm volatile("vse16.v v16, (%[rs1])"
                    : 
                    : [rs1]"r"(varsum+i));
    }

    for (int i = 0; i < hw; i+=tilem) { 
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tilem)
                    : [rs1]"r"(hw-i));
        vl = vsetvl_e16m1(tilem);
        asm volatile("vle16.v v0, (%[rs1])"
                    :
                    : [rs1]"r"(redsum+i));
        asm volatile("vle16.v v2, (%[rs1])"
                    :
                    : [rs1]"r"(varsum+i));
        
        for (int j = 0; j < c; j+=tilen) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tilen)
                        : [rs1]"r"(c-j));
            vl = vsetvl_e16m1(tilen);
            asm volatile("vle16.v v4, (%[rs1])"
                        :
                        : [rs1]"r"(pgamma+j));
            asm volatile("vle16.v v6, (%[rs1])"
                        :
                        : [rs1]"r"(pbeta+j));
            
            asm volatile("mlce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(psrc+i*stride_src/datasize+j), [rs2]"r"(stride_src));
            asm volatile("mfaddcc.mv acc1, acc0, v0"); // x + (-u)
            asm volatile("mfemulcc.mv acc0, acc1, v2"); // 
            asm volatile("mfemulcr.mv acc1, acc0, v4");
            asm volatile("mfaddcr.mv acc0, acc1, v6");
            asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                        :
                        : [rs1]"r"(pdst+i*stride_dst/datasize+j), [rs2]"r"(stride_dst));
        }

    }


    return 0;
}

#endif