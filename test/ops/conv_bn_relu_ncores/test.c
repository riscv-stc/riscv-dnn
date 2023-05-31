#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../src/conv_bn_relu_ncores.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(srcData, "src.bin", ".scdata.params");
INCBIN(weightData, "weight.bin", ".scdata.params");
INCBIN(alphaData, "alpha.bin", ".scdata.params");
INCBIN(betaData, "beta.bin", ".scdata.params");

uint8_t dstData[OUT_SIZE * sizeof(float16_t)] __attribute__((__section__(".scdata.output")));


int main(int argc, char **argv)
{
    printf("Begin\n");

    config_conv(sst, HIN, WIN, CIN, COUT,
                KH, KW, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
                PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT,
                CIN*2, COUT*2, COUT*2);


    int pid = read_csr(mhartid);

    PERF_BEGIN();

    for (int i = 0; i < 1; i++) {
        conv_bn_relu_ncores_hout(dstData, srcData, weightData, alphaData, betaData, &sst, CORENUMS, pid);
        barrier(CORENUMS);
    }

    PERF_END();

    return 0;
}
