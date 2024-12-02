#ifndef __CONV_ADD_BN_RELU_NOCRES_H__
#define __CONV_ADD_BN_RELU_NOCRES_H__

#include "../include/matrix/matrix_intrinsic.h"
#include "conv_add_bn_relu_rvm.h"
#include "conv_im2col.h"
#include "matmul.h"
#include "mme.h"
#include "tensor.h"
#include "util.h"
#include <stddef.h>

static inline int conv_add_bn_relu_ncores_hout(Tensor *dst, Tensor *addout,
                                               Tensor *src, Tensor *weight,
                                               Tensor *addsrc, Tensor *alpha,
                                               Tensor *beta, Config *ss,
                                               int ncores, int pid) {
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
  char *paddout = (char *)addout->data;
  char *pdst = (char *)dst->data;
  char *palpha = (char *)alpha->data;
  char *pbeta = (char *)beta->data;
  char *paddsrc = (char *)addsrc->data;

  int stride_src = src->stride;
  int stride_weight = weight->stride;
  int stride_dst = dst->stride;
  int stride_addsrc = addsrc->stride;
  int stride_addout = addout->stride;

  assert(hout % ncores == 0 && cout % ncores == 0);

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

    tensor_new_3d_with_stride(_srcMat, part_hin, win, cin, dataSize, _src,
                              stride_src);

    int cout_idx = i;

    char *_weight = pweight + cout_idx * part_cout * dataSize;

    tensor_new_4d_with_stride(_weightMat, kh, kw, cin, part_cout, dataSize,
                              _weight, stride_weight);

    char *_dst = pdst + hout_idx * part_hout * wout * stride_dst +
                 cout_idx * part_cout * dataSize;

    tensor_new_3d_with_stride(_dstMat, part_hout, wout, part_cout, dataSize,
                              _dst, stride_dst);

    config_conv(_sst, part_hin, win, cin, part_cout, _pad_t, _pad_b, pad_l,
                pad_r, kh, kw, stride_h, stride_w, dilation_h, dilation_w);

    char *_palhpa = palpha + cout_idx * part_cout * dataSize;
    char *_pbeta = pbeta + cout_idx * part_cout * dataSize;
    tensor_new_1d(_alphaMat, part_cout, dataSize, _palhpa);
    tensor_new_1d(_betaMat, part_cout, dataSize, _pbeta);

    char *_addsrc = paddsrc + hout_idx * part_hout * wout * stride_addsrc +
                    cout_idx * part_cout * dataSize;
    char *_addout = paddout + hout_idx * part_hout * wout * stride_addout +
                    cout_idx * part_cout * dataSize;
    tensor_new_3d_with_stride(_addsrcMat, part_hout, wout, part_cout, dataSize,
                              _addsrc, stride_addsrc);
    tensor_new_3d_with_stride(_addoutMat, part_hout, wout, part_cout, dataSize,
                              _addout, stride_addout);

    conv_add_bn_relu_rvm(&_dstMat, &_addoutMat, &_srcMat, &_weightMat,
                         &_addsrcMat, &_alphaMat, &_betaMat, &_sst);
  }

  return 0;
}

#endif
