#include <stdio.h>
#include <stdlib.h>

#include "../../../src/matmul_add.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(src1Data, "src1.bin", ".scdata.params");
INCBIN(src2Data, "src2.bin", ".scdata.params");
INCBIN(src3Data, "src3.bin", ".scdata.params");

uint8_t dstData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    const int m = M;
    const int k = K;
    const int n = N;

    //enableCount();

    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(m, k, n) = (%d, %d, %d)\n",
                    m, k, n);
    }

    tensor_new_2d(src1Mat, m, k, sizeof(float16_t), src1Data);
    tensor_new_2d(src2Mat, k, n, sizeof(float16_t), src2Data);
    tensor_new_2d(src3Mat, m, n, sizeof(float16_t), src3Data);
    tensor_new_2d(dstMat, m, n, sizeof(float16_t), &dstData);


    for (int i = 0; i < 1; i++) {
        matmul_add(&src3Mat, &src1Mat, &src2Mat);
    }

    memcpy(dstData, src3Data, OUT_SIZE * sizeof(float16_t));

    
    printf("End\n");

    return 0;
}
