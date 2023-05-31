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

#include "conv_im2col_ncores.h"
#include "conv_bn_relu_ncores.h"
#include "conv_add_bn_relu_ncores.h"
#include "pooling_bn_relu_ncores.h"

#include "conv_im2col.h"
#include "conv_bn_relu_rvm.h"
#include "conv_add_bn_relu_rvm.h"
#include "pooling.h"

#include "resnet50_parameters.h"


#define DATASIZE 2

#define CACHELINE 128


int resnet50_ncores(uint8_t *indata, int ncores, int pid);

int conv_bn_relu(Tensor *relu_out, Tensor *conv_in, void *pweight, void *palpha, void *pbeta, Config sst, int wstride);
int conv_base(Tensor *conv_out, Tensor *conv_in, void *pweight, Config sst, int wstride);
int conv_add_bn_relu(Tensor *relu_out, Tensor *add_out, Tensor *conv_in, Tensor *add_in, void *pweight, void *palpha, void *pbeta, Config sst, int wstride);


#define STAGE_392KB  401408
#define STAGE_1960KB 2007040

#define STAGE_882KB  930168
#define STAGE_1176KB 1204224
#define STAGE_294KB  301056

#define STAGE_416KB5 426496
#define STAGE_490KB  501760
#define STAGE_122KB5 125440 

#define STAGE_202KB  206976
#define STAGE_220KB5 225792
#define STAGE_55KB   56448


uint8_t buffer[11640836] __attribute__((__section__(".scdata.output"))); // 11368KB, 11.102MB


// uint8_t conv2d_data[112*112*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t max_pooling2d_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t batch_normalization_data[56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_data[16][56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv2d_1_data[16][56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_1_data[16][56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_2_data[16][56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_data[16][56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_3_data[16][56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_4_data[16][56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_5_data[16][56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_1_data[16][56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_6_data[16][56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_7_data[16][56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_8_data[16][56*56*64*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_2_data[16][56*56*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_9_data[16][56*56*320*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv2d_11_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_10_data[16][56*56*192*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_11_data[4][28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_3_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_12_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_13_data[16][28*28*192*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_14_data[4][28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_4_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_15_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_16_data[4][28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_17_data[4][28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_5_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_18_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_19_data[4][28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_20_data[4][28*28*128*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_6_data[4][28*28*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_21_data[16][28*28*576*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv2d_24_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_22_data[8][28*28*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_23_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_7_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_24_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_25_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_26_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_8_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_27_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_28_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_29_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_9_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_30_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_31_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_32_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_10_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_33_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_34_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_35_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_11_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_36_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_37_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_38_data[8][14*14*256*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_12_data[8][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_39_data[16][14*14*1152*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t conv2d_43_data[16][7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_40_data[16][14*14*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_41_data[16][7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_13_data[16][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_42_data[16][7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_43_data[16][7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_44_data[16][7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_14_data[16][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_45_data[16][7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_46_data[16][7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_47_data[16][7*7*512*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t add_15_data[16][14*14*1024*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t Relu_48_data[16][7*7*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

// uint8_t Mean_data[16][1*1*2048*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t MatMul_data[16][1*1001*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));
// uint8_t softmax_tensor_fp16_data[16][1*1001*sizeof(float16_t)] __attribute__((__section__(".scdata.output")));


// uint8_t bias_tmp_data[16][1001*sizeof(float16_t)];
// uint8_t bias_add_fp16_tmp_data[16][1001*sizeof(float16_t)];
// uint8_t bias_add_fp32_tmp_data[16][1001*sizeof(float32_t)] __attribute__((aligned(64)));
// uint8_t softmax_tensor_fp32_data[16][1001*sizeof(float32_t)] __attribute__((aligned(64)));

#endif
