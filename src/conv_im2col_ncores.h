#ifndef __CONV_IM2COL_NOCRES_H__
#define __CONV_IM2COL_NOCRES_H__

#include "tensor.h"
#include <stddef.h>

#include "mme.h"
#include "matmul.h"
#include "conv_im2col_add.h"
#include "conv_im2col.h"
#include "util.h"

static inline int conv_ncores_cout(Tensor *dst, Tensor *src, Tensor *weight, Config *ss, int ncores)
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
    char *psrc = (char *)src->data;
    char *pweight = (char *)weight->data;
    char *pdst = (char *)dst->data;

    int stride_src = src->stride;
    int stride_weight = weight->stride;
    int stride_dst = dst->stride;

    assert(cin%ncores==0 && cout%ncores==0);
    int pid = read_csr(mhartid);

    int part_cout = cout / ncores;
    int part_cin  = cin / ncores;

    for (int i = 0; i < kh; i++) {
      for (int j = 0; j <  kw; j++) {
        for (int n = 0; n < ncores; n++) {
          int cin_idx =  i * kw * ncores + j * ncores + n;
          
          char *_src =  psrc + n * part_cin * dataSize;

          tensor_new_3d_with_stride(_srcMat, hin, win, part_cin, dataSize, _src, stride_src);

          int cout_idx = pid;

          char *_weight = pweight + cout_idx * part_cout * dataSize + cin_idx * part_cin * stride_weight;

          tensor_new_4d_with_stride(_weightMat, kh, kw, part_cin, part_cout, dataSize, _weight, stride_weight);

          char *_dst = pdst + cout_idx * part_cout * dataSize;

          tensor_new_3d_with_stride(_dstMat, hout, wout, part_cout, dataSize, _dst, stride_dst);

          config_conv(sst, hin, win, part_cin, part_cout, pad_t, pad_b, pad_l, pad_r, kh, kw, stride_h, stride_w, dilation_h, dilation_w);

          im2col_add(&_dstMat, &_srcMat, &_weightMat, &sst, i, j);
        }
      }
    }
    
    return 0;
}


static inline int conv_ncores_hout(Tensor *dst, Tensor *src, Tensor *weight, Config *ss, int ncores)
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
    char *psrc = (char *)src->data;
    char *pweight = (char *)weight->data;
    char *pdst = (char *)dst->data;

    int stride_src = src->stride;
    int stride_weight = weight->stride;
    int stride_dst = dst->stride;

    assert(hout%ncores==0 && cout%ncores==0);
    int pid = read_csr(mhartid);

    int part_cout = cout / ncores;
    int part_hout = hout / ncores;

    for (int i = 0; i < ncores; ++i) {
      int part_hin = (part_hout - 1) * stride_h + 1 + dilation_h * (kh - 1);
      
      int hout_idx = pid; 
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
  
      tensor_new_3d_with_stride(_srcMat, part_hin, win, cin, dataSize, _src, stride_src);

      int cout_idx = i;

      char *_weight = pweight + cout_idx * part_cout * dataSize;

      tensor_new_4d_with_stride(_weightMat, kh, kw, cin, part_cout, dataSize, _weight, stride_weight);

      char *_dst = pdst + hout_idx * part_hout * wout * stride_dst  + cout_idx * part_cout * dataSize;

      tensor_new_3d_with_stride(_dstMat, part_hout, wout, part_cout, dataSize, _dst, stride_dst);

      config_conv(_sst, part_hin, win, cin, part_cout, _pad_t, _pad_b, pad_l, pad_r, kh, kw, stride_h, stride_w, dilation_h, dilation_w);

      conv_im2col(&_dstMat, &_srcMat, &_weightMat, &_sst);
    }
    
    return 0;
}




static inline int conv_im2col_ncores(Tensor *dst, Tensor *src, Tensor *weight, Config *ss, int ncores)
{
  conv_ncores_cout(dst, src, weight, ss, ncores);
}

#endif
