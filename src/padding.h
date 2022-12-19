#ifndef __PADDING_H__
#define __PADDING_H__

#include "mme.h"
#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

static inline int padding(Tensor *dst, Tensor *src, Config *ss)
{
    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int wout = win + pad_l + pad_r;

    int vl;
    float16_t *psrc = (float16_t *)src->data;
    float16_t *pdst = (float16_t *)dst->data;
    pdst = pdst + pad_t * wout * cin + pad_l * cin;
    int tmp = cin * win;

    for (int i = 0; i < hin; i++) {
        for (int k = 0; k < tmp; k += vl) {
            vl = vsetvl_e16m1(tmp - k);
            vfloat16m1_t _data = vle16_v_f16m1(psrc, vl);
            vse16_v_f16m1(pdst, _data, vl);
            psrc += vl;
            pdst += vl;
        }
    pdst += (pad_r + pad_l) * cin;
    }

    return 0;
}

#endif