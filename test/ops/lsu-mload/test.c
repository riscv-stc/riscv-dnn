#include <stdio.h>
#include <stdlib.h>

#include "../../../src/tensor.h"
#include "../../../include/matrix/matrix_intrinsic.h"
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

    tensor_new_2d(src1Mat, m, k, sizeof(float16_t), src1Data);
    tensor_new_2d(src2Mat, k, n, sizeof(float16_t), src2Data);
    tensor_new_2d(dstMat, m, n, sizeof(float16_t), &dstData);

    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(e16));
    asm volatile("msettilem x0, %[rs1]"
                 :
                 : [rs1]"r"(64));
    asm volatile("msettilen x0, %[rs1]"
                 :
                 : [rs1]"r"(64));
    asm volatile("msettilek x0, %[rs1]"
                 :
                 : [rs1]"r"(64));
    
    PERF_BEGIN();

    for (int i = 0; i < NLOOPS; i++) {
        asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(src1Data), [rs2]"r"(128));
        asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                :
                :[rs1]"r"(src2Data), [rs2]"r"(128));
    }

    PERF_END();

    asm volatile("msae16.m tr0, (%[rs1]), %[rs2]"
                : 
                : [rs1]"r"(dstData), [rs2]"r"(128));
    
    printf("End\n");

    return 0;
}
