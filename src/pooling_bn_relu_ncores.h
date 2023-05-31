#ifndef __POOLING_BN_RELU_NCORES_H__
#define __POOLING_BN_RELU_NCORES_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "mme.h"
#include "pooling_bn_relu.h"


/*
  padding = 0
  wout % 4 ==0
*/
static inline int maxpool_bn_relu_ncores_hout(void *dst, void *src, void *alpha, void *beta, Config *ss, int ncores, int pid)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int dataSize = sizeof(float16_t);
    char *psrc = (char *)src;
    char *pdst = (char *)dst;

    int part_hout = hout / ncores;

    int part_hin = (part_hout - 1) * stride_h + kh;
    
    int hout_idx = pid % ncores;
    int hin_idx = hout_idx;

    char *_src = psrc + hin_idx * (part_hout * stride_h) * win * ss->stride_src;

    int _pad_t = 0;
    int _pad_b = 0;

    if (hin_idx == 0) {
      part_hin -= pad_t;
      _pad_t = pad_t;
    } else {
      _src -= pad_t * win * ss->stride_src;
    }
    if (hin_idx == (ncores - 1)) {
      part_hin -= pad_b;
      _pad_b = pad_b;
    }

    char *_dst = pdst + hout_idx * part_hout * wout * ss->stride_dst;

    config_pool(_sst, part_hin, win, cin, cin,
                      kh, kw, stride_h, stride_w,
                      _pad_t, _pad_b, pad_l, pad_r,
                      ss->stride_src, ss->stride_dst);

    maxpool_bn_relu(_dst, _src, alpha, beta, &_sst);
    
    return 0;
}

#endif