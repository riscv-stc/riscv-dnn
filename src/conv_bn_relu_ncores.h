#ifndef __CONV_BN_RELU_NOCRES_H__
#define __CONV_BN_RELU_NOCRES_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"
#include "conv_bn_relu_rvm.h"
#include "conv_im2col.h"
#include "util.h"

static inline int conv_bn_relu_ncores_hout_cout(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss, int ncores, int pid)
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

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;

    assert(hout%ncores==0 && cout%ncores==0);

    int part_cout = cout / ncores;
    int part_hout = hout / ncores;

    for (int i = 0; i < ncores; ++i) {
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

      int cout_idx = i;

      char *_weight = pweight + cout_idx * part_cout * dataSize;

      char *_dst = pdst + hout_idx * part_hout * wout * stride_dst  + cout_idx * part_cout * dataSize;

      config_conv(_sst, part_hin, win, cin, part_cout,
                        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                        _pad_t, _pad_b, pad_l, pad_r,
                        stride_src, stride_weight, stride_dst);

      char *_palhpa = palpha + cout_idx * part_cout * dataSize;
      char *_pbeta = pbeta + cout_idx * part_cout * dataSize;

      conv_bn_relu_rvm(_dst, _src, _weight, _palhpa, _pbeta, &_sst);
    }
    
    return 0;
}

static inline int conv_bn_relu_ncores_hout(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss, int ncores, int pid)
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
    char *pdst = (char *)dst;

    int stride_src = ss->stride_src;
    int stride_dst = ss->stride_dst;

    assert(hout%ncores==0);

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

    config_conv(_sst, part_hin, win, cin, cout,
                      kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                      _pad_t, _pad_b, pad_l, pad_r,
                      stride_src, ss->stride_ker, stride_dst);

    conv_bn_relu_rvm(_dst, _src, weight, alpha, beta, &_sst);
    
    
    return 0;
}

static inline int conv_bn_relu_ncores_cout(void *dst, void *src, void *weight, void *alpha, void *beta, Config *ss, int ncores, int pid)
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

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;

    assert(cout%ncores==0);

    int part_cout = cout / ncores;

  
    int cout_idx = pid % ncores;

    char *_weight = pweight + cout_idx * part_cout * dataSize;

    char *_dst = pdst + cout_idx * part_cout * dataSize;

    config_conv(_sst, hin, win, cin, part_cout,
                      kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                      pad_t, pad_b, pad_l, pad_r,
                      stride_src, stride_weight, stride_dst);

    char *_palhpa = palpha + cout_idx * part_cout * dataSize;
    char *_pbeta = pbeta + cout_idx * part_cout * dataSize;

    conv_bn_relu_rvm(_dst, src, _weight, _palhpa, _pbeta, &_sst);
    
    return 0;
}


#endif
