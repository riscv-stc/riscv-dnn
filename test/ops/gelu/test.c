#include <stdio.h>
#include <stdlib.h>

#include "../../../src/gelu.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");

uint8_t dstData[H*W * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w) = (%d, %d)\n",
                    H, W);
    }

    config_2d(ss, H, W, W*sizeof(float16_t), W * sizeof(float16_t));

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        gelu(dstData, srcData, &ss);
    }

    PERF_END();
    
    printf("End\n");

    return 0;
}
