#include <stdio.h>
#include <stdlib.h>

#include "../../../src/relu.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");
INCBIN(baseData, "base.bin", ".scdata.params");

uint8_t dstData[SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    const int h = H;
    const int w = W;
    const int c = C;
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, c) = (%d, %d, %d)\n",
                    h, w, c);
    }

    tensor_new_3d(srcMat, h, w, c, sizeof(float16_t), srcData);
    tensor_new_3d(dstMat, h, w, c, sizeof(float16_t), &dstData);

    float16_t base = *(float16_t *)baseData;

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        relu(&dstMat, &srcMat, base);
    }

    PERF_END();
    
    printf("End\n");

    return 0;
}