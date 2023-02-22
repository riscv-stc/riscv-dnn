#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/conv_add_bn_relu_ncores.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");
INCBIN(weightData, "weight.bin", ".scdata.params");
INCBIN(addsrcData, "addsrc.bin", ".scdata.params");
INCBIN(alphaData, "alpha.bin", ".scdata.params");
INCBIN(betaData, "beta.bin", ".scdata.params");

uint8_t dstData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

uint8_t addoutData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));


int main(int argc, char **argv)
{
    printf("Begin\n");

    config_conv(sst, HIN, WIN, CIN, COUT, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, KH, KW, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);

    tensor_new_3d(srcMat, HIN, WIN, CIN, sizeof(float16_t), srcData);
    tensor_new_4d(weightMat, KH, KW, CIN, COUT, sizeof(float16_t), weightData);
    tensor_new_3d(addsrcMat, HOUT, WOUT, COUT, sizeof(float16_t), addsrcData);
    tensor_new_1d(alphaMat, COUT, sizeof(float16_t), &alphaData);
    tensor_new_1d(betaMat, COUT, sizeof(float16_t), &betaData);
    tensor_new_3d(dstMat, HOUT, WOUT, COUT, sizeof(float16_t), &dstData);
    tensor_new_3d(addoutMat, HOUT, WOUT, COUT, sizeof(float16_t), &addoutData);

    int pid = read_csr(mhartid);

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        conv_add_bn_relu_ncores(&dstMat, &addoutMat, &srcMat, &weightMat, &addsrcMat, &alphaMat, &betaMat, &sst, CORENUMS, pid);
    }

    PERF_END();

    return 0;
}
