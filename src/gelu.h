#ifndef __RELU_H__
#define __RELU_H__

#include "tensor.h"
#include "mme.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "exp.h"
/*

    GELU(x) = x * (1 - 1 / (1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))))
*/

static inline int gelu(void *dst, void *src, Config *ss)
{   
    int h = ss->hin;
    int w = ss->win;

    int stride_src = ss->stride_src / sizeof(float16_t);
    int stride_dst = ss->stride_dst / sizeof(float16_t);
    
    float16_t *psrc = (float16_t *)src;
    float16_t *pdst = (float16_t *)dst;

    const float const1 = 0.044715;
    const float const2 = 1.59576912; //2 * sqrt(2 / pi)
    const float const3 = 1.0;

    int vl;

    for (int i = 0; i < h; i++) {
        float16_t *_psrc = psrc + i * stride_src;
        float16_t *_pdst = pdst + i * stride_dst;
        for (int j = 0; j < w; j += vl) {
            vl = vsetvl_e16m4(w - j);
            vfloat16m4_t _data16 = vle16_v_f16m4(_psrc, vl);
            vfloat32m8_t _data = vfwcvt_f_f_v_f32m8(_data16, vl);

            vfloat32m8_t _power = vfmul_vv_f32m8(_data, _data, vl); // x*x

            _power = vfmul_vf_f32m8(_power, const1, vl); // 0.0044715*x*x

            _power = vfadd_vf_f32m8(_power, const3, vl); // 1+0.0044715*x*x

            _power = vfmul_vv_f32m8(_data, _power, vl); // x*(1+0.0044715*x*x)

            _power = vfmul_vf_f32m8(_power, const2, vl); // 2*sqrt(2/pi)*x*(1+0.0044715*x*x)

            _power = vfexp_f32m8(_power, vl); // exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x)

            _power = vfadd_vf_f32m8(_power, const3, vl); // 1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x)

            _power = vfrec7_v_f32m8(_power, vl); // 1/(1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))

            _power = vfrsub_vf_f32m8(_power, const3, vl); // 1-1/(1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))

            _power = vfmul_vv_f32m8(_data, _power, vl); // x*(1-1/(1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))(1+0.0044715*x*x))

            vse16_v_f16m4(_pdst, vfncvt_f_f_w_f16m4(_power, vl), vl);
            
            _psrc += vl;
            _pdst += vl;
        }
    }
    return 0;
}

#endif