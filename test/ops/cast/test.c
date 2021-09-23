#include <stdio.h>
#include <stdlib.h>

#include "../../../src/cast.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");

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

    tensor_new_4d(srcMat, h, w, cin, cout, sizeof(float32_t), srcData);
    tensor_new_4d(dstMat, h, w, cin, cout, sizeof(float16_t), &dstData);

    PERF_BEGIN();

    cast_f32_to_f16(&dstMat, &srcMat);

    PERF_END();
    
    printf("End\n");

    return 0;
}