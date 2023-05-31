#include <stdio.h>
#include <stdlib.h>

#include "../../../src/matmul_ncores.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(src1Data, "src1.bin", ".scdata.params");
INCBIN(src2Data, "src2.bin", ".scdata.params");

uint8_t dstData[OUT_SIZE * BATCH * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    // printf("Begin\n");
    //enableCount();
    int dataSize = sizeof(float16_t);
    
    config_matmul(ss, M, K, N, K*dataSize, N*dataSize, N*dataSize);

    for (int i = 0; i < 1; i++) {
        matmul_ncores(dstData, src1Data, src2Data, &ss, CORENUMS, BATCH);
        barrier(CORENUMS);
    }


    return 0;
}
