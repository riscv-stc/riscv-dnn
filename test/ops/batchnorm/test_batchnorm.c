#include "../../src/batchnorm.h"



int main(int argc, char **argv)
{
    if (DEBUG_PRINT) {
        printf("Begin\n");
    }
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
    int gamSize = c;
    int betaSize = gamSize;
    int meanSize = gamSize;
    int varSize = gamSize;
    int outSize = srcSize;
    float16_t *srcData = new float16_t[srcSize];
    if (DEBUG_PRINT) {
        std::ifstream srcFile("src.bin", std::ios::binary);
        srcFile.read((char *)srcData, sizeof(float16_t) * srcSize);
        srcFile.close();
    }

    float16_t *gamData = new float16_t[gamSize];
    if (DEBUG_PRINT) {
        std::ifstream gamFile("gam.bin", std::ios::binary);
        gamFile.read((char *)gamData, sizeof(float16_t) * gamSize);
        gamFile.close();
    }

    float16_t *betaData = new float16_t[betaSize];
    if (DEBUG_PRINT) {
        std::ifstream betaFile("beta.bin", std::ios::binary);
        betaFile.read((char *)betaData, sizeof(float16_t) * betaSize);
        betaFile.close();
    }


    Mat srcMat = Mat(h, w, c, sizeof(float16_t), srcData);
    Mat gamMat = Mat(c, sizeof(float16_t), gamData);
    Mat betaMat = Mat(c, sizeof(float16_t), betaData);
    Mat dstMat = Mat(h, w, c, sizeof(float16_t));

    // batchnorm(dstMat, srcMat, meanMat, varMat, gamMat, betaMat, *epsData);
    batchnorm(dstMat, srcMat, gamMat, betaMat);

    if (DEBUG_PRINT) {
        float16_t *dst = (float16_t *)dstMat.data;
        FILE *dstFile = fopen("dst.bin", "w");
        fwrite(dst, sizeof(float16_t), outSize, dstFile);
        fclose(dstFile);
        printf("End\n");
    }

    return 0;
}