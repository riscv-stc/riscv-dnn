#ifndef __GELU_H__
#define __GELU_H__

#include "exp.h"
#include "mme.h"
#include "tensor.h"
#include <riscv_vector.h>
#include <stddef.h>
/*

    GELU(x) = x * (1 - 1 / (1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))))
*/

static inline int gelu(Tensor *dst, Tensor *src) {
  float16_t *_psrc = (float16_t *)src->data;
  float16_t *_pdst = (float16_t *)dst->data;

  const float const1 = 0.044715;
  const float const2 = 1.59576912; // 2 * sqrt(2 / pi)
  const float const3 = 1.0;

  int vl;

  for (int i = 0; i < src->size; i += vl) {
    vl = vsetvl_e16m4(src->size - i);
    vfloat16m4_t _data16 = vle16_v_f16m4(_psrc, vl);
    vfloat32m8_t _data = vfwcvt_f_f_v_f32m8(_data16, vl);

    vfloat32m8_t _power = vfmul_vv_f32m8(_data, _data, vl); // x*x

    _power = vfmul_vf_f32m8(_power, const1, vl); // 0.0044715*x*x

    _power = vfadd_vf_f32m8(_power, const3, vl); // 1+0.0044715*x*x

    _power = vfmul_vv_f32m8(_data, _power, vl); // x*(1+0.0044715*x*x)

    _power =
        vfmul_vf_f32m8(_power, const2, vl); // 2*sqrt(2/pi)*x*(1+0.0044715*x*x)

    _power = vfexp_f32m8(_power, vl); // exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x)

    _power = vfadd_vf_f32m8(_power, const3,
                            vl); // 1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x)

    _power = vfrec7_v_f32m8(_power,
                            vl); // 1/(1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))

    _power = vfrsub_vf_f32m8(
        _power, const3, vl); // 1-1/(1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))

    _power = vfmul_vv_f32m8(
        _data, _power,
        vl); // x*(1-1/(1+exp(2*sqrt(2/pi)*x*(1+0.0044715*x*x))(1+0.0044715*x*x))

    vse16_v_f16m4(_pdst, vfncvt_f_f_w_f16m4(_power, vl), vl);

    _psrc += vl;
    _pdst += vl;
  }
  return 0;
}

#endif // __GELU_H__