#include "../../src/relu.h"



int main(int argc, char **argv)
{
    printf("Begin\n");
    struct Shape sst;
    assert(argc == 4);
    int h = atoi(argv[1]);
    int w = atoi(argv[2]);
    int c = atoi(argv[3]);
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, c) = (%d, %d, %d)\n",
                    h, w, c);
    }

    int srcSize = h * w *c;
    int outSize = srcSize;

    float16_t *srcData = new float16_t[srcSize];
    std::ifstream srcFile("src.bin", std::ios::binary);
    srcFile.read((char *)srcData, sizeof(float16_t) * srcSize);
    srcFile.close();

    float16_t *baseData = new float16_t[1];
    std::ifstream baseFile("base.bin", std::ios::binary);
    baseFile.read((char *)baseData, sizeof(float16_t));
    baseFile.close();

    Mat srcMat = Mat(h, w, c, sizeof(float16_t), srcData);
    float16_t base = *baseData;
    Mat dstMat = Mat(h, w, c, sizeof(float16_t));

    relu(dstMat, srcMat, base);

    float16_t *dst = (float16_t *)dstMat.data;
    FILE *dstFile = fopen("dst.bin", "w");
    fwrite(dst, sizeof(float16_t), outSize, dstFile);
    fclose(dstFile);
    printf("End\n");

    return 0;
}