#include <stdio.h>
#include <stdlib.h>

#include "../../../src/add.h"
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

    PERF_BEGIN();

    add(&dstMat, &src1Mat, &src2Mat);

    PERF_END();
    
    printf("End\n");

    return 0;
}