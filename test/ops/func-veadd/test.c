#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(src1Data, "src1.bin", ".scdata.params");
INCBIN(src2Data, "src2.bin", ".scdata.params");

uint8_t dstData[SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    printf("Begin\n");

    const int h = H;
    const int w = W;
    const int cin = CIN;
    const int cout = COUT;
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, cin, cout) = (%d, %d, %d, %d)\n",
                    h, w, cin, cout);
    }

    tensor_new_4d(src1Mat, h, w, cin, cout, sizeof(float16_t), src1Data);
    tensor_new_4d(src2Mat, h, w, cin, cout, sizeof(float16_t), src2Data);
    tensor_new_4d(dstMat, h, w, cin, cout, sizeof(float16_t), &dstData);

    int vl = vsetvl_e16m1(VLENB/2);
    asm volatile("vle16.v v1, (%[rs1])"
                  :
                  :[rs1]"r"(src1Data));

    asm volatile("vle16.v v2, (%[rs1])"
                  :
                  :[rs1]"r"(src2Data));
    
    PERF_BEGIN();
    for(int i = 0; i < NLOOPS; i++) {
        asm("vfadd.vv v3, v1, v2");
        asm("vfadd.vv v4, v1, v2");
        asm("vfadd.vv v5, v1, v2");
        asm("vfadd.vv v6, v1, v2");
        asm("vfadd.vv v7, v1, v2");
        asm("vfadd.vv v8, v1, v2");
        asm("vfadd.vv v9, v1, v2");
        asm("vfadd.vv v10, v1, v2");
        asm("vfadd.vv v11, v1, v2");
        asm("vfadd.vv v12, v1, v2");
    }
    PERF_END();

    asm volatile("vse16.v v3, (%[rs1])"
                  :
                  :[rs1]"r"(dstData));
    
    printf("End\n");

    return 0;
}