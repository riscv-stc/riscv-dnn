#ifndef __BATCHNORM_H__
#define __BATCHNORM_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

/*
    dst = γ * (src - mean) / sqrt(variance * variance + epsilon) + β
    src: [h, w, c]
    γ: [c]
    β: [c]
*/
static int batchnorm2(Tensor *dst, Tensor *src, Tensor *mean, Tensor *variance, Tensor *gam, Tensor *beta, float16_t epsilon)
{
    int h = src->h;
    int w = src->w;
    int c = src->cin;

    assert(c == gam->w && c == beta->w);

    float16_t *psrc = (float16_t *)src->data;
    float16_t *pmean = (float16_t *)mean->data;
    float16_t *pvar = (float16_t *)variance->data;
    float16_t *pgam = (float16_t *)gam->data;
    float16_t *pbeta = (float16_t *)beta->data;
    float16_t *pdst = (float16_t *)dst->data;

#ifdef DEBUG
    printf("%f, %f, %f, %f, %f, %f\n", 
        (float)*psrc, (float)*pmean, (float)*pvar, (float)*pgam, (float)*pbeta, (float)epsilon);
#endif

    int vlmax = VLENB / 2;

    for(int i = 0; i < c; i += vlmax) {
        int vl = min(vlmax, c - i);
        vfloat16m1_t _mean = vle16_v_f16m1(pmean + i, vl);
        vfloat16m1_t _var = vle16_v_f16m1(pvar + i, vl);
        _var = vfadd_vf_f16m1(_var, epsilon, vl);
        vfloat16m1_t _var_recip = vfrsqrt7_v_f16m1(_var, vl);
        vfloat16m1_t _gam = vle16_v_f16m1(pgam + i, vl);
        vfloat16m1_t _beta = vle16_v_f16m1(pbeta + i, vl);
        _mean = vfmul_vv_f16m1(_gam, _mean, vl);
        _mean = vfmul_vv_f16m1(_mean, _var_recip, vl);
        _beta = vfsub_vv_f16m1(_beta, _mean, vl);
        for(int j = 0; j < h * w; j ++) {
            vfloat16m1_t _src = vle16_v_f16m1(psrc + j * c + i, vl);
            vfloat16m1_t _res = vfmul_vv_f16m1(_src, _gam, vl);
            _res = vfmul_vv_f16m1(_res, _var_recip, vl);
            _res = vfadd_vv_f16m1(_res, _beta, vl);
            vse16_v_f16m1(pdst + j * c + i, _res, vl);
        }
    }
     
    return 0;
}



/*
    dst = α * src  + β
    α =  γ / sqrt(variance + epsilon)
    β = -γ / sqrt(variance + epsilon) * mean + beta
    src: [h, w, c]
    γ: [c]
    β: [c]
*/
static inline int batchnorm(Tensor *dst, Tensor *src, Tensor *alpha, Tensor *beta)
{
    int h = src->h;
    int w = src->w;
    int c = src->cin;

    // assert(c == alpha->w && c == beta->w);

    float16_t *psrc = (float16_t *)src->data;
    float16_t *palpha = (float16_t *)alpha->data;
    float16_t *pbeta = (float16_t *)beta->data;
    float16_t *pdst = (float16_t *)dst->data;

    int vlmax = VLENB * 4 / 2;

    for(int i = 0; i < c; i += vlmax) {
        int vl = min(vlmax, c - i);
        vfloat16m4_t _alpha = vle16_v_f16m4(palpha + i, vl);
        vfloat16m4_t _beta_f16 = vle16_v_f16m4(pbeta + i, vl);
        vfloat32m8_t _beta_f32 = vfwcvt_f_f_v_f32m8(_beta_f16, vl);
        for(int j = 0; j < h * w; j ++) {
            vfloat16m4_t _src = vle16_v_f16m4(psrc + j * c + i, vl);
            vfloat32m8_t _res = vfwmacc_vv_f32m8(_beta_f32, _src, _alpha, vl);
            vse16_v_f16m4(pdst + j * c + i, vfncvt_f_f_w_f16m4(_res, vl), vl);
        }
    }
     
    return 0;
}


#endif