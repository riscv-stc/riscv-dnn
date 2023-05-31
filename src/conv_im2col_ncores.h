#ifndef __CONV_IM2COL_NOCRES_H__
#define __CONV_IM2COL_NOCRES_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"
#include "conv_im2col_add.h"
#include "conv_im2col.h"
#include "util.h"

static inline int conv_ncores_cout_cin(void *dst, void *src, void *weight, Config *ss, int ncores, int pid)
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

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;

    assert(cin%ncores==0 && cout%ncores==0);

    int part_cout = cout / ncores;
    int part_cin  = cin / ncores;

    for (int i = 0; i < kh; i++) {
      for (int j = 0; j <  kw; j++) {
        for (int n = 0; n < ncores; n++) {
          int cin_idx =  i * kw * ncores + j * ncores + n;
          
          char *_src =  psrc + n * part_cin * dataSize;

          int cout_idx = pid % ncores;

          char *_weight = pweight + cout_idx * part_cout * dataSize + cin_idx * part_cin * stride_weight;

          char *_dst = pdst + cout_idx * part_cout * dataSize;

          config_conv(sst, hin, win, part_cin, part_cout,
                           kh, kw, stride_h, stride_w, dilation_h, dilation_w, 
                           pad_t, pad_b, pad_l, pad_r,
                           stride_src, stride_weight, stride_dst);

          im2col_add(_dst, _src, _weight, &sst, i, j);
        }
      }
    }
    
    return 0;
}

static inline int conv_ncores_hout(void *dst, void *src, void *weight, Config *ss, int ncores, int pid)
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

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;

    assert(hout%ncores==0 && cout%ncores==0);

    int part_cout = cout / ncores;
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
                      stride_src, stride_weight, stride_dst);

    conv_im2col(_dst, _src, weight, &_sst);
    
    return 0;
}

static inline int conv_ncores_hout_small_cin(void *dst, void *src, void *weight, Config *ss, int ncores, int pid)
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

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;

    assert(hout%ncores==0 && cout%ncores==0);

    int part_cout = cout / ncores;
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
                    stride_src, stride_weight, stride_dst);

    conv_im2col_small_cin(_dst, _src, weight, &_sst);
    
    return 0;
}


static inline int conv_ncores_hout_cout(void *dst, void *src, void *weight, Config *ss, int ncores, int pid)
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

      conv_im2col(_dst, _src, _weight, &_sst);
    }
    
    return 0;
}

static inline int conv_ncores_cin_cout(void *dst, void *src, void *weight, Config *ss, int ncores, int pid)
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

    int stride_src = ss->stride_src;
    int stride_weight = ss->stride_ker;
    int stride_dst = ss->stride_dst;

    assert(cin%ncores==0 && cout%ncores==0);

    int part_cout = cout / ncores;
    int part_cin  = cin / ncores;

    for (int n = 0; n < ncores; n++) {
      int cin_idx =  pid % ncores;
      int cout_idx = ncores - 1 - n % ncores;

      char *_src =  psrc + cin_idx * part_cin * dataSize;

      tensor_new_3d_with_stride(_srcMat, hin, win, part_cin, dataSize, _src, stride_src);

      for (int i = 0; i < kh; i++) {
        for (int j = 0; j <  kw; j++) {
      
          char *_weight = pweight + cout_idx * part_cout * dataSize + (i * kw * ncores + j * ncores + pid) * part_cin * stride_weight;

          char *_dst = pdst + cout_idx * part_cout * dataSize;

          config_conv(sst, hin, win, part_cin, part_cout,
                           kh, kw, stride_h, stride_w, dilation_h, dilation_w,
                           pad_t, pad_b, pad_l, pad_r, 
                           stride_src, stride_weight, stride_dst);

          im2col_add(_dst, _src, _weight, &sst, i, j);
        }
      }
      barrier(ncores);

    }
    
    return 0;
}


#endif
