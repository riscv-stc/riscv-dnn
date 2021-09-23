#include "../../src/padding.h"



int main(int argc, char **argv)
{
    printf("Begin\n");
    struct Shape sst;
    assert(argc == 8);

    sst.hin = atoi(argv[1]);
    sst.win = atoi(argv[2]);
    sst.cin = atoi(argv[3]);
  
    sst.top = atoi(argv[4]);
    sst.bottom = atoi(argv[5]);
    sst.left = atoi(argv[6]);
    sst.right = atoi(argv[7]);

    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, cin) = (%d, %d, %d)\n",
                    sst.hin, sst.win, sst.cin);
        printf("\t(pt, pb, pl, pr) = (%d, %d, %d, %d)\n", sst.top, sst.bottom, sst.left, sst.right);
    }

    sst.hout = sst.hin + sst.top + sst.bottom;
    sst.wout = sst.win + sst.left + sst.right;
    int srcSize = sst.hin * sst.win * sst.cin;
    int outSize = sst.hout * sst.wout * sst.cin;
    float16_t *srcData = new float16_t[srcSize];
    std::ifstream srcFile("src.bin", std::ios::binary);
    srcFile.read((char *)srcData, sizeof(float16_t) * srcSize);
    srcFile.close();

    Mat srcMat = Mat(sst.hin, sst.win, sst.cin, sizeof(float16_t), srcData);
    Mat dstMat = Mat(sst.hout, sst.wout, sst.cin, sizeof(float16_t));

    padding(dstMat, srcMat, &sst);

    if (DEBUG_PRINT) {
        printf("Out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                sst.hout, sst.wout, sst.cin);
    }

    float16_t *dst = (float16_t *)dstMat.data;
    FILE *dstFile = fopen("dst.bin", "w");
    fwrite(dst, sizeof(float16_t), outSize, dstFile);
    fclose(dstFile);
    printf("End\n");

    return 0;
}