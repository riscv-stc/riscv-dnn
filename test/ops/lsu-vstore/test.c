#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(src1Data, "src1.bin", ".scdata.params");

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
    tensor_new_4d(dstMat, h, w, cin, cout, sizeof(float16_t), &dstData);

    int vl = vsetvl_e16m1(VLENB/2);

    asm volatile("vle16.v v1, (%[rs1])"
                  :
                  :[rs1]"r"(src1Data));
    
    PERF_BEGIN();
    for(int i = 0; i < NLOOPS; i++) {
        asm volatile("vse16.v v1, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+i*64));
         asm volatile("vse16.v v2, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+1)*64));
        asm volatile("vse16.v v3, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+2)*64));
        asm volatile("vse16.v v4, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+3)*64));
        asm volatile("vse16.v v5, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+4)*64));
        asm volatile("vse16.v v6, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+5)*64));
        asm volatile("vse16.v v7, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+6)*64));
        asm volatile("vse16.v v8, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+7)*64));
        asm volatile("vse16.v v9, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+8)*64));
        asm volatile("vse16.v v10, (%[rs1])"
                  :
                  :[rs1]"r"(dstData+(i+9)*64));
    }
    PERF_END();

    
    
    printf("End\n");

    return 0;
}