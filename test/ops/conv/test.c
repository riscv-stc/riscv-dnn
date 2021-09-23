#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/conv.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");
INCBIN(weightData, "weight.bin", ".scdata.params");

uint8_t padData[PAD_SIZE * sizeof(float16_t)];

uint8_t dstData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));


int main(int argc, char **argv)
{
    printf("Begin\n");

    config_conv(sst, HIN, WIN, CIN, COUT, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, KH, KW, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);

    tensor_new_2d(srcMat, HIN, WIN * CIN, sizeof(float16_t), srcData);
    tensor_new_2d(weightMat, KH * KW * CIN, COUT, sizeof(float16_t), weightData);
    tensor_new_2d(dstMat, HOUT * WOUT, COUT, sizeof(float16_t), &dstData);

    tensor_new_2d(srcPad, HOUT * WOUT, KH * KW * CIN, sizeof(float16_t), padData);

    PERF_BEGIN();

    conv(&dstMat, &srcMat, &weightMat, &srcPad, &sst);

    PERF_END();

    if (DEBUG_PRINT) {
        printf("Out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                sst.hout, sst.wout, sst.cout);
    }

    printf("End\n");

    return 0;
}
