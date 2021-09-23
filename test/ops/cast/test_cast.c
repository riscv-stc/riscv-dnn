#include "../../src/cast.h"



int main(int argc, char **argv)
{
    if (DEBUG_PRINT) {
        printf("Begin\n");
    }
    struct Shape sst;
    assert(argc == 4 || argc == 5);
    int h = atoi(argv[1]);
    int w = atoi(argv[2]);
    int cin = atoi(argv[3]);
    int cout = 1;
    if (argc == 5) {
        cout = atoi(argv[4]);
    }
    
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, cin, cout) = (%d, %d, %d, %d)\n",
                    h, w, cin, cout);
    }

    int srcSize = h * w * cin * cout;
    int outSize = srcSize;

    float32_t *srcData = new float32_t[srcSize];
    if (DEBUG_PRINT) {
        std::ifstream srcFile("src.bin", std::ios::binary);
        srcFile.read((char *)srcData, sizeof(float32_t) * srcSize);
        srcFile.close();
    }

    Mat srcMat = Mat(h, w, cin, cout, sizeof(float32_t), srcData);
    Mat dstMat = Mat(h, w, cin, cout, sizeof(float16_t));

    cast_f32_to_f16(dstMat, srcMat);
    if (DEBUG_PRINT) {
        float16_t *dst = (float16_t *)dstMat.data;
        FILE *dstFile = fopen("dst.bin", "w");
        fwrite(dst, sizeof(float16_t), outSize, dstFile);
        fclose(dstFile);
        printf("End\n");
    }

    return 0;
}