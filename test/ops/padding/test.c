#include <stdio.h>
#include <stdlib.h>

#include "../../../src/padding.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");

uint8_t dstData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));


int main(int argc, char **argv)
{
    printf("Begin\n");

    config_padding(sst, HIN, WIN, CIN, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT);

    tensor_new_3d(srcMat, HIN, WIN, CIN, sizeof(float16_t), srcData);
    tensor_new_3d(dstMat, HOUT, WOUT, COUT, sizeof(float16_t), &dstData);

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        padding(&dstMat, &srcMat, &sst);
    }

    PERF_END();

    if (DEBUG_PRINT) {
        printf("Out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                sst.hout, sst.wout, sst.cout);
    }

    printf("End\n");

    return 0;
}