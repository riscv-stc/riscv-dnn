#include "../../src/pooling.h"



int main(int argc, char **argv)
{
    printf("Begin\n");
    struct Shape sst;
    assert(argc >= 6);
    sst.kh = atoi(argv[4]);
    sst.kw = atoi(argv[5]);
    sst.hin = atoi(argv[1]);
    sst.win = atoi(argv[2]);
    sst.cin = atoi(argv[3]);
    if (argc == 8 || argc == 12) {
        sst.stride_h = atoi(argv[6]);
        sst.stride_w = atoi(argv[7]);
    } 
    if (argc == 10 || argc == 12) {
        sst.top = atoi(argv[8]);
        sst.bottom = atoi(argv[9]);
        sst.left = atoi(argv[10]);
        sst.right = atoi(argv[11]);
    }
    if (DEBUG_PRINT) {
        printf("In Shape:\n\t(h, w, cin, kh, kw) = (%d, %d, %d, %d, %d)\n",
                    sst.hin, sst.win, sst.cin, sst.kh, sst.kw);
        printf("\t(stride_h, stride_w) = (%d, %d)\n", sst.stride_h, sst.stride_w);
        printf("\t(pt, pb, pl, pr) = (%d, %d, %d, %d)\n", sst.top, sst.bottom, sst.left, sst.right);
    }

    sst.hout = (sst.hin + sst.top + sst.bottom - sst.kh) / sst.stride_h + 1;
    sst.wout = (sst.win + sst.left + sst.right - sst.kw) / sst.stride_w + 1;
    int srcSize = sst.hin * sst.win * sst.cin;
    int outSize = sst.hout * sst.wout * sst.cin;
    float16_t *srcData = new float16_t[srcSize];
    if (DEBUG_PRINT) {
        std::ifstream srcFile("src.bin", std::ios::binary);
        srcFile.read((char *)srcData, sizeof(float16_t) * srcSize);
        srcFile.close();
    }

    Mat srcMat = Mat(sst.hin, sst.win, sst.cin, sizeof(float16_t), srcData);
    Mat dstMat = Mat(sst.hout, sst.wout, sst.cin, sizeof(float16_t));

    avgpool(dstMat, srcMat, &sst);

    if (DEBUG_PRINT) {
        printf("Out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                sst.hout, sst.wout, sst.cin);
    }
    if (DEBUG_PRINT) {
        float16_t *dst = (float16_t *)dstMat.data;
        FILE *dstFile = fopen("dst.bin", "w");
        fwrite(dst, sizeof(float16_t), outSize, dstFile);
        fclose(dstFile);
        printf("End\n");
    }

    return 0;
}