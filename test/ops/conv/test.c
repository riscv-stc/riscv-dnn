#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/conv.h"
#include "../../../src/conv_im2col.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");
INCBIN(weightData, "weight.bin", ".scdata.params");

uint8_t padData[PAD_SIZE * sizeof(float16_t)] __attribute__((aligned(128)));

uint8_t dstData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));


int main(int argc, char **argv)
{
    printf("Begin\n");
    printf("cahceline=%d\n", CACHELINE);

    config_conv(sst, HIN, WIN, CIN, COUT,
                KH, KW, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
                PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT,
                CIN*2+CACHELINE, COUT*2+CACHELINE, COUT*2+CACHELINE);

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        conv_rvm_n_k64(dstData, srcData, weightData+i*KH*KW*CIN*(COUT+CACHELINE/2)*2, &sst);
        // conv_im2col(dstData, srcData, weightData+i*KH*KW*CIN*(COUT+64)*2, &sst);
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
