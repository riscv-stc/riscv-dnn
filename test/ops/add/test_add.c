#include "../../src/add.h"
#include "../../include/incbin.h"

int main(int argc, char **argv)
{
    if (DEBUG_PRINT) {
        printf("Begin\n");
    }
    assert(argc == 5);
    int h = atoi(argv[1]);
    int w = atoi(argv[2]);
    int cin = atoi(argv[3]);
    int cout = atoi(argv[4]);
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, cin, cout) = (%d, %d, %d, %d)\n",
                    h, w, cin, cout);
    }

    int src1Size = h * w * cin * cout;
    int src2Size = src1Size;
    int outSize = src1Size;

    
    INCBIN(src1Data, "src1.bin", '.data');
    INCBIN(src2Data, "src2.bin", '.data');

    tensor_new_4d(src1Mat, h, w, cin, cout, sizeof(float16_t), src1Data);
    tensor_new_4d(src2Mat, h, w, cin, cout, sizeof(float16_t), src2Data);
    tensor_new_4d(dstMat, h, w, cin, cout, sizeof(float16_t));

    add(&dstMat, &src1Mat, &src2Mat);
    
    // if (DEBUG_PRINT) {
    //     float16_t *dst = (float16_t *)dstMat->data;
    //     FILE *dstFile = fopen("dst.bin", "w");
    //     fwrite(dst, sizeof(float16_t), outSize, dstFile);
    //     fclose(dstFile);
    //     printf("End\n");
    // }

    return 0;
}