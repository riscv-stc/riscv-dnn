#include <stdio.h>
#include <stdlib.h>

#include "../../../src/softmax.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");

uint8_t dstData[SIZE * sizeof(float32_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    int h = H;
    int w = W;
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w) = (%d, %d)\n",
                    h, w);
    }

    tensor_new_2d(srcMat, H, W, sizeof(float32_t), srcData);
    tensor_new_2d(dstMat, H, W, sizeof(float32_t), &dstData);

    PERF_BEGIN();

    softmax(&dstMat, &srcMat);

    PERF_END();        

    printf("End\n");

    return 0;
}