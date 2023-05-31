#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/conv_add_bn_relu_rvm.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");
INCBIN(addsrcData, "addsrc.bin", ".scdata.params");
INCBIN(alphaData, "alpha.bin", ".scdata.params");
INCBIN(betaData, "beta.bin", ".scdata.params");

uint8_t dstData[OUT_SIZE * sizeof(float16_t) * 8] __attribute__((__section__(".scdata.output")));

uint8_t addoutData[OUT_SIZE * sizeof(float16_t) * 8] __attribute__((__section__(".scdata.params")));
#ifdef __SPIKE__
INCBIN(weightData, "weight.bin", ".scdata.params");
#else
uint8_t weightData[KH*KW*CIN*(COUT+CACHELINE) * sizeof(float16_t) * 8] __attribute__((__section__(".scdata.params")));
#endif


int main(int argc, char **argv)
{
    printf("Begin\n");

    config_conv_add(sst, HIN, WIN, CIN, COUT,
                    KH, KW, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
                    PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT,              
                    CIN*2+CACHELINE, COUT*2+CACHELINE, COUT*2+CACHELINE, COUT*2+CACHELINE, COUT*2+CACHELINE);

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        conv_add_bn_relu_rvm_batch8(dstData, addoutData, srcData, addsrcData, weightData+i*KH*KW*CIN*(COUT+CACHELINE/2)*2, alphaData, betaData, &sst);
    }

    PERF_END();
    asm("fence.i");
    if (DEBUG_PRINT) {
        printf("Out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                sst.hout, sst.wout, sst.cout);
    }

    printf("End\n");

    return 0;
}
