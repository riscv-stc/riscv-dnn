#ifndef __CONV_ADD_BN_RELU_NOCRES_H__
#define __CONV_ADD_BN_RELU_NOCRES_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"
#include "conv_add_bn_relu_rvm.h"
#include "conv_im2col.h"
#include "util.h"

static inline int conv_add_bn_relu_ncores_hout(void *dst, void *addout, void *src, void *addsrc, void *weight, void *alpha, void *beta, Config *ss, int ncores, int pid)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int dataSize = sizeof(float16_t);
    char *psrc = (char *)src;
    char *pweight = (char *)weight;
    char *paddout = (char *)addout;
    char *pdst = (char *)dst;
    char *palpha = (char *)alpha;
    char *pbeta = (char *)beta;
    char *paddsrc = (char *)addsrc;

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;
    int stride_addout = ss->stride_addout;

    assert(hout%ncores==0 && cout%ncores==0);

    int part_hout = hout / ncores;

    int part_hin = (part_hout - 1) * stride_h + 1 + dilation_h * (kh - 1);
      
    int hout_idx = pid % ncores;
    int hin_idx = hout_idx;

    int _pad_t = 0;
    int _pad_b = 0;

    char *_src = psrc + hin_idx * (part_hout * stride_h) * win * stride_src;

    if (hin_idx == 0) {
      part_hin -= pad_t;
      _pad_t = pad_t;
    } else {
      _src -= pad_t * win * stride_src;
    }
    if (hin_idx == (ncores - 1)) {
      part_hin -= pad_b;
      _pad_b = pad_b;
    }

    char *_dst = pdst + hout_idx * part_hout * wout * stride_dst;

    config_conv_add(_sst, part_hin, win, cin, cout,
                          kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                          _pad_t, _pad_b, pad_l, pad_r,
                          stride_src, stride_weight, stride_dst, stride_addsrc, stride_addout);

    char *_addsrc = paddsrc + hout_idx * part_hout * wout * stride_addsrc;
    char *_addout = paddout + hout_idx * part_hout * wout * stride_addout;

    conv_add_bn_relu_rvm(_dst, _addout, _src, _addsrc, weight, alpha, beta, &_sst);
    
    return 0;
}


static inline int conv_add_bn_relu_last_ncores_hout(void *dst, void *src, void *addsrc, void *weight, void *alpha, void *beta, Config *ss, int ncores, int pid)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int dataSize = sizeof(float16_t);
    char *psrc = (char *)src;
    char *pweight = (char *)weight;
    char *pdst = (char *)dst;
    char *palpha = (char *)alpha;
    char *pbeta = (char *)beta;
    char *paddsrc = (char *)addsrc;

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;
    int stride_addsrc = ss->stride_addsrc;

    assert(hout%ncores==0 && cout%ncores==0);

    int part_hout = hout / ncores;

    int part_hin = (part_hout - 1) * stride_h + 1 + dilation_h * (kh - 1);
      
    int hout_idx = pid % ncores;
    int hin_idx = hout_idx;

    int _pad_t = 0;
    int _pad_b = 0;

    char *_src = psrc + hin_idx * (part_hout * stride_h) * win * stride_src;

    if (hin_idx == 0) {
      part_hin -= pad_t;
      _pad_t = pad_t;
    } else {
      _src -= pad_t * win * stride_src;
    }
    if (hin_idx == (ncores - 1)) {
      part_hin -= pad_b;
      _pad_b = pad_b;
    }

    char *_dst = pdst + hout_idx * part_hout * wout * stride_dst;

    config_conv_add(_sst, part_hin, win, cin, cout,
                          kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                          _pad_t, _pad_b, pad_l, pad_r,
                          stride_src, stride_weight, stride_dst, stride_addsrc, 0);

    char *_addsrc = paddsrc + hout_idx * part_hout * wout * stride_addsrc;

    conv_add_bn_relu_last(_dst, _src, _addsrc, weight, alpha, beta, &_sst);
    
    return 0;
}

#endif
