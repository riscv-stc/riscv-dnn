#include <stdio.h>
#include <stdlib.h>

#include "../../../src/exp.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");

uint8_t dstData[SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    int h = H;
    int w = W;
    int cin = CIN;
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, cin) = (%d, %d, %d)\n",
                    h, w, cin);
    }

    tensor_new_3d(srcMat, H, W, CIN, sizeof(float16_t), srcData);
    tensor_new_3d(dstMat, H, W, CIN, sizeof(float16_t), &dstData);

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        exp(&dstMat, &srcMat);
    }

    PERF_END();

    printf("End\n");

    return 0;
}