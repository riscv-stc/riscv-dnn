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

#include "conv_im2col.h"
#include "conv_bn_relu_rvm.h"
#include "conv_add_bn_relu_rvm.h"
#include "pooling_bn_relu.h"

#include "memcpy_rvm.h"

#include "resnet50_parameters.h"


#define DATASIZE 2

#define BATCH2 2
#define BATCH3 4


int resnet50_base(void *indata, void *wdata, int num);

#define STAGE_392KB    401408

#define STAGE_2240KB   2293760  // 64*56*(256+64)*2
#define STAGE_1134KB   1161216 //36*28*(512+64)*2

#define STAGE_1011KB5  1035776  // 32*28*(512+64)*2
#define STAGE_560KB    573440   // 32*28*(256+64)*2
#define STAGE_140KB    143360   // 16*14*(256+64)*2

#define STAGE_476KB   487424  // 16*14*(1024*64)*2
#define STAGE_231KB   236544  // 8*7*(2048+64)*2
#define STAGE_252KB   258048  // 16*14*(512+64)*2
#define STAGE_63KB    64512   // 8*7*(512+64)*2
#define STAGE_2KB     2048    //


uint8_t buffer[8730624] __attribute__((__section__(".scdata.output"))); //

// uint8_t conv2d_data[112*112*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t max_pooling2d_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t batch_normalization_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t conv2d_1_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_1_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_2_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_3_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_4_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_5_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_1_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_6_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_7_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_8_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_2_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_9_data[56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv2d_11_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_10_data[56*56*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_11_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_3_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_12_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_13_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_14_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_4_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_15_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_16_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_17_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_5_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_18_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_19_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_20_data[28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_6_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_21_data[28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv2d_24_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_22_data[28*28*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_23_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_7_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_24_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_25_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_26_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_8_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_27_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_28_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_29_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_9_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_30_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_31_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_32_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_10_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_33_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_34_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_35_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_11_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_36_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_37_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_38_data[14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_12_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_39_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv2d_43_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_40_data[14*14*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_41_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_13_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_42_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_43_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_44_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_14_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_45_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_46_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_47_data[7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_15_data[14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_48_data[7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t Mean_data[1*1*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t MatMul_data[1*1001*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t softmax_tensor_fp16_data[1*1001*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv_tmp_data[56*56*256*sizeof(float16_t)];
// uint8_t conv_pad_tmp_data[70*70*256*3*3*sizeof(float16_t)];
// uint8_t batchnorm_tmp_data[56*56*256*sizeof(float16_t)];
// uint8_t relu_tmp_data[56*56*256*sizeof(float16_t)];
// uint8_t dense_tmp_data[2048*1001*sizeof(float16_t)];
// uint8_t bias_tmp_data[1001*sizeof(float16_t)];
// uint8_t bias_add_fp16_tmp_data[1001*sizeof(float16_t)];
// uint8_t bias_add_fp32_tmp_data[1001*sizeof(float32_t)] __attribute__((aligned(64)));
// uint8_t softmax_tensor_fp32_data[1001*sizeof(float32_t)] __attribute__((aligned(64)));

#endif