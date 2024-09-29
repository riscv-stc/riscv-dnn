#include <stdio.h>
#include <stdlib.h>

#include "../../../src/matmul_uint.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(src1Data, "src1.bin", ".scdata.params");
INCBIN(src2Data, "src2.bin", ".scdata.params");

uint8_t dstData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    const int m = M;
    const int k = K;
    const int n = N;

    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(m, k, n) = (%d, %d, %d)\n",
                    m, k, n);
    }

    tensor_new_2d(src1Mat, m, k, sizeof(uint8_t), src1Data);
    tensor_new_2d(src2Mat, k, n, sizeof(uint8_t), src2Data);
    tensor_new_2d(dstMat, m, n, sizeof(uint32_t), &dstData);

    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        matmul(&dstMat, &src1Mat, &src2Mat);
    }

    PERF_END();
    
    printf("End\n");

    return 0;
}
