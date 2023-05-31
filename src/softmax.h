#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "exp.h"


uint32_t neg_inf_32 = 0xFF800000;

int softmax(void *dst, void *src, int size)
{

    float32_t *psrc = (float32_t *)src;
    float32_t *pdst = (float32_t *)dst;

    int vl;

    float32_t neg_inf = *(float32_t*)&neg_inf_32;
    vfloat32m1_t _maxium = vfmv_v_f_f32m1(neg_inf, 1); 
    for (int i = 0; i < size; i += vl) {
        vl = vsetvl_e32m8(size - i);
        vfloat32m8_t _src = vle32_v_f32m8(psrc + i, vl);
        _maxium = vfredmax_vs_f32m8_f32m1(_maxium, _src, _maxium, vl);
    }
    float32_t pmax = vfmv_f_s_f32m1_f32(_maxium);

    vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, 1);
    
    for (int i = 0; i < size; i += vl) {
        vl = vsetvl_e32m8(size - i);
        vfloat32m8_t _src = vle32_v_f32m8(psrc + i, vl);
        vfloat32m8_t _diff = vfsub_vf_f32m8(_src, pmax, vl); // src-max
        vfloat32m8_t _exp = vfexp_f32m8(_diff, vl);
        vse32_v_f32m8(pdst + i, _exp, vl);
        _sum = vfredosum_vs_f32m8_f32m1(_sum, _exp, _sum, vl);
    }
    float32_t psum = vfmv_f_s_f32m1_f32(_sum);
    asm("fence");

    for (int i = 0; i < size; i += vl) {
        vl = vsetvl_e32m8(size - i);
        vfloat32m8_t _src = vle32_v_f32m8(pdst + i, vl);
        vfloat32m8_t _prob = vfdiv_vf_f32m8(_src, psum, vl);
        vse32_v_f32m8(pdst + i, _prob, vl);
    }

    return 0;
}

