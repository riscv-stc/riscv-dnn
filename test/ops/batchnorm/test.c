#include <stdio.h>
#include <stdlib.h>

#include "../../../src/batchnorm.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");
INCBIN(gamData, "gam.bin", ".scdata.params");
INCBIN(betaData, "beta.bin", ".scdata.params");

uint8_t dstData[SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    const int h = H;
    const int w = W;
    const int c = C;
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, c, cout) = (%d, %d, %d)\n",
                    h, w, c);
    }

    tensor_new_3d(srcMat, h, w, c, sizeof(float16_t), srcData);
    tensor_new_1d(gamMat, c, sizeof(float16_t), gamData);
    tensor_new_1d(betaMat,  c, sizeof(float16_t), betaData);
    tensor_new_3d(dstMat, h, w, c, sizeof(float16_t), &dstData);

    PERF_BEGIN();

    batchnorm(&dstMat, &srcMat, &gamMat, &betaMat);

    PERF_END();
    
    printf("End\n");

    return 0;
}