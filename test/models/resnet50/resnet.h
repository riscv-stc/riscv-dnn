#ifndef __RESNET_H__
#define __RESNET_H__


#include <string.h>
#include <stdlib.h>
    
#include "add.h"
#include "batchnorm.h"
#include "cast.h"
#include "conv.h"
#include "matmul.h"
#include "padding.h"
#include "pooling.h"
#include "relu.h"
#include "softmax.h"

#include "resnet50_parameters.h"

// #define conv_base(conv_out, conv_in, pweight, sst) \
//     tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout,sizeof(float16_t), pweight); \
//     tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, sizeof(float16_t), conv_pad_data); \
//     conv(&conv_out, &conv_in, &conv_kernel_f16, &conv_in_pad, &sst);

// #define conv_bn_relu(relu_out, conv_in, pweight, palpha, pbeta, sst) \
//     tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, sizeof(float16_t), pweight); \
//     tensor_new_3d(conv_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), conv_data); \
//     tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, sizeof(float16_t), conv_pad_data); \
//     conv(&conv_out, &conv_in, &conv_kernel_f16, &conv_in_pad, &sst); \
//     tensor_new_1d(alpha, sst.cout, sizeof(float16_t), palpha); \
//     tensor_new_1d(beta, sst.cout, sizeof(float16_t), pbeta); \
//     tensor_new_3d(batchnorm_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), batchnorm_data); \
//     batchnorm(&batchnorm_out, &conv_out, &alpha, &beta); \
//     relu(&relu_out, &batchnorm_out, (float16_t)0);

// #define conv_add_bn_relu(relu_out, add_out, conv_in, add_in, pweight, palpha, pbeta, sst) \
//     tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, sizeof(float16_t), pweight); \
//     tensor_new_3d(conv_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), conv_data); \
//     tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, sizeof(float16_t), conv_pad_data); \
//     conv(&conv_out, &conv_in, &conv_kernel_f16, &conv_in_pad, &sst); \
//     add(&add_out, &conv_out, add_in); \
//     tensor_new_1d(alpha, sst.cout, sizeof(float16_t), palpha); \
//     tensor_new_1d(beta, sst.cout, sizeof(float16_t), pbeta); \
//     tensor_new_3d(batchnorm_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), batchnorm_data); \
//     batchnorm(&batchnorm_out, &add_out, &alpha, &beta); \
//     relu(&relu_out, &batchnorm_out, (float16_t)0); \

int resnet50_base(void *indata);

int conv_bn_relu(Tensor *relu_out, Tensor *conv_in, void *pweight, void *palpha, void *pbeta, Config sst);
int conv_base(Tensor *conv_out, Tensor *conv_in, void *pweight, Config sst);
int conv_add_bn_relu(Tensor *relu_out, Tensor *add_out, Tensor *conv_in, Tensor *add_in, void *pweight, void *palpha, void *pbeta, Config sst);




uint8_t conv2d_data[112*112*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t max_pooling2d_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t batch_normalization_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t conv2d_1_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_1_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_2_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_3_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_4_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_5_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_1_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_6_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_7_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_8_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_2_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_9_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

uint8_t conv2d_11_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_10_data[56*56*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_11_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_3_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_12_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_13_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_14_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_4_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_15_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_16_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_17_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_5_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_18_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_19_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_20_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_6_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_21_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

uint8_t conv2d_24_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_22_data[28*28*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_23_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_7_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_24_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_25_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_26_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_8_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_27_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_28_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_29_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_9_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_30_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_31_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_32_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_10_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_33_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_34_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_35_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_11_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_36_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_37_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_38_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_12_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_39_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

uint8_t conv2d_43_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_40_data[14*14*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_41_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_13_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_42_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_43_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_44_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_14_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_45_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_46_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_47_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t add_15_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t Relu_48_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

uint8_t Mean_data[1*1*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t MatMul_data[1*1001*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
uint8_t softmax_tensor_fp16_data[1*1001*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

uint8_t conv_tmp_data[56*56*256*sizeof(float16_t)];
uint8_t conv_pad_tmp_data[70*70*256*3*3*sizeof(float16_t)];
uint8_t batchnorm_tmp_data[56*56*256*sizeof(float16_t)];
uint8_t relu_tmp_data[56*56*256*sizeof(float16_t)];
uint8_t dense_tmp_data[2048*1001*sizeof(float16_t)];
uint8_t bias_tmp_data[1001*sizeof(float16_t)];
uint8_t bias_add_fp16_tmp_data[1001*sizeof(float16_t)];
uint8_t bias_add_fp32_tmp_data[1001*sizeof(float32_t)];
uint8_t softmax_tensor_fp32_data[1001*sizeof(float32_t)];

#endif
