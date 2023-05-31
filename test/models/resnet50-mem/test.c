#include <stdio.h>

#include "resnet.h"
#include "../../../src/hpm.h"
#include "../../../src/encoding.h"

#include "../../../src/perf.h"

#include "params.h"

// #define PICTURE_SIZE  301056 // 224 * 224 * 3 * DATASIZE
#define PICTURE_SIZE  317400 // 230 * 230 * 3 * DATASIZE

uint64_t stage_start[6], stage_end[6];

int main()
{   
    printf("begin %d, loops=%d\n", BATCH, NLOOPS);
    uint64_t cycles=0;

    PERF_BEGIN();
    asm("csrwi frm, 0");
    for(int i = 0; i < BATCH; i++) {
        for (int j = 0; j < NLOOPS; j++) {
            memcpy_rvm(merge_data, add_data, 56*56, 320);
            memcpy_rvm(conv_data, imagenet_pic_data_data, 56*56, 160);
            cycles = read_csr_safe(cycle);
            resnet50_base(imagenet_pic_data_data + i * PICTURE_SIZE, i);
            cycles = read_csr_safe(cycle) - cycles;
        }
    }
    PERF_END();

    printf("End\n");

    printf("ALL Cycles: %ld\n", cycles);
    for (int i = 0; i < 6; i++) {
        printf("    stage %d: %d\n ", i, stage_end[i] - stage_start[i]);
    }

    for (int i = 0; i < 6; i++) {
        printf("%d\n ", stage_end[i] - stage_start[i]);
    }
    printf("%ld\n", cycles);

    return 0;
}

int resnet50_base(void *indata, int num)
{

    // input(1, 224, 224, 3)
/*
 * stage 0 
*/
    stage_start[0] = read_csr_safe(cycle);
    // stage 0: padding && cast weight && conv
    config_conv(stage0_conv, 230, 230, 3, 64, 7, 7, 2, 2, 1, 1, 0, 0, 0, 0,  6, 128, 128);
    conv_im2col_small_cin(conv_data[0], indata, weight_data+woffset[0], &stage0_conv);
    if (DEBUG_PRINT) {
        printf("stage0_conv_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage0_conv.hout, stage0_conv.wout, stage0_conv.cout);
    }
    

    // stage0: maxpool, SAME
    config_pool(stage0_maxpool, 112, 112, 64, 64, 3, 3, 2, 2, 0, 1, 0, 1,  128, 128); // Relu
    maxpool_bn_relu(conv_data[1], conv_data[0], weight_data+aoffset[0], weight_data+boffset[0], &stage0_maxpool);
    if (DEBUG_PRINT) {
        printf("stage0_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout);
    }
    stage_end[0] = read_csr_safe(cycle);

/*
 * stage 1 
*/
    // s1u1
    stage_start[1] = read_csr_safe(cycle);
    config_conv(stage1_conv1, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640);
    conv_im2col(add_data, conv_data[1], weight_data+woffset[1], &stage1_conv1);

    config_conv(stage1_conv2, 56, 56, 64, 64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 128, 128); // Relu_1
    conv_bn_relu_rvm(conv_data[0], conv_data[1], weight_data+woffset[2], weight_data+aoffset[1], weight_data+boffset[1], &stage1_conv2);
    
    config_conv(stage1_conv3, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  128, 128, 128); // Relu_2
    conv_bn_relu_rvm(conv_data[1], conv_data[0], weight_data+woffset[3], weight_data+aoffset[2], weight_data+boffset[2], &stage1_conv3);

    config_conv_add(stage1_conv4, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); //Relu_3
    conv_add_bn_relu_rvm(merge_data, add_data, conv_data[1], add_data, weight_data+woffset[4], weight_data+aoffset[3], weight_data+boffset[3], &stage1_conv4); //add
    if (DEBUG_PRINT) {
        printf("stage1_relu3_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout);
    }

    config_conv(stage1_conv5, 56, 56, 256, 64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 128, 128); // Relu_4
    conv_bn_relu_rvm(conv_data[0], merge_data, weight_data+woffset[5], weight_data+aoffset[4], weight_data+boffset[4], &stage1_conv5);
    
    config_conv(stage1_conv6, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 128, 128, 128);  // Relu_5
    conv_bn_relu_rvm(conv_data[1], conv_data[0], weight_data+woffset[6], weight_data+aoffset[5], weight_data+boffset[5], &stage1_conv6);

    config_conv_add(stage1_conv7, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); // Relu_6
    conv_add_bn_relu_rvm(merge_data, add_data, conv_data[1], add_data, weight_data+woffset[7], weight_data+aoffset[6], weight_data+boffset[6], &stage1_conv7); // add_1
    if (DEBUG_PRINT) {
        printf("stage1_6_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout);
    }

    config_conv(stage1_conv8, 56, 56, 256, 64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 128, 128); // Relu_7
    conv_bn_relu_rvm(conv_data[0], merge_data, weight_data+woffset[8], weight_data+aoffset[7], weight_data+boffset[7], &stage1_conv8);

    config_conv(stage1_conv9, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  128, 128, 128); // Relu_8
    conv_bn_relu_rvm(conv_data[1], conv_data[0], weight_data+woffset[9], weight_data+aoffset[8], weight_data+boffset[8], &stage1_conv9);
 
    config_conv_add(stage1_conv10, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); // Relu_9
    conv_add_bn_relu_rvm(merge_data, add_data, conv_data[1], add_data, weight_data+woffset[10], weight_data+aoffset[9], weight_data+boffset[9], &stage1_conv10); //add_2
    if (DEBUG_PRINT) {
        printf("stage1_9_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout);
    }
    stage_end[1] = read_csr_safe(cycle);

/*
 * stage 2
*/
    // s2u1
    stage_start[2] = read_csr_safe(cycle);
    config_conv(stage2_conv11, 56, 56, 256, 512, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0,  640, 1152, 1152); 
    conv_im2col(conv_data[0], merge_data, weight_data+woffset[11], &stage2_conv11);

    config_conv(stage2_conv12, 56, 56, 256, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 384, 384); // Relu_10
    conv_bn_relu_rvm(add_data, merge_data, weight_data+woffset[12], weight_data+aoffset[10], weight_data+boffset[10], &stage2_conv12);

    config_conv(stage2_conv13, 56, 56, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 384, 384, 384); // Relu_11
    conv_bn_relu_rvm(merge_data[0], add_data, weight_data+woffset[13], weight_data+aoffset[11], weight_data+boffset[11], &stage2_conv13);

    config_conv_add(stage2_conv14, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 384, 1152, 1152, 1152, 1152); // Relu_12
    conv_add_bn_relu_rvm(merge_data[1], conv_data[0], merge_data[0], conv_data[0], weight_data+woffset[14], weight_data+aoffset[12], weight_data+boffset[12], &stage2_conv14); //add_3
    if (DEBUG_PRINT) {
        printf("stage2_12_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout);
    }

    config_conv(stage2_conv15, 28, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_13
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[15], weight_data+aoffset[13], weight_data+boffset[13], &stage2_conv15);

    config_conv(stage2_conv16, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 384, 384, 384); // Relu_14
    conv_bn_relu_rvm(merge_data[2], merge_data[0], weight_data+woffset[16], weight_data+aoffset[14], weight_data+boffset[14], &stage2_conv16);

    config_conv_add(stage2_conv17, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   384, 1152, 1152, 1152, 1152); // Relu_15
    conv_add_bn_relu_rvm(merge_data[0], conv_data[0], merge_data[2], conv_data[0], weight_data+woffset[17], weight_data+aoffset[15], weight_data+boffset[15], &stage2_conv17); //add_4
    if (DEBUG_PRINT) {
        printf("stage2_15_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout);
    }

    config_conv(stage2_conv18, 28, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_16
    conv_bn_relu_rvm(merge_data[2], merge_data[0], weight_data+woffset[18], weight_data+aoffset[16], weight_data+boffset[16], &stage2_conv18);

    config_conv(stage2_conv19, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  384, 384, 384); // Relu_17
    conv_bn_relu_rvm(merge_data[0], merge_data[2], weight_data+woffset[19], weight_data+aoffset[17], weight_data+boffset[17], &stage2_conv19);

    config_conv_add(stage2_conv20, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  384, 1152, 1152, 1152, 1152); // Relu_18
    conv_add_bn_relu_rvm(merge_data[2], conv_data[0], merge_data[0], conv_data[0], weight_data+woffset[20], weight_data+aoffset[18], weight_data+boffset[18], &stage2_conv20);//add_5
    if (DEBUG_PRINT) {
        printf("stage2_18_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout);
    }

    config_conv(stage2_conv21, 28, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_19
    conv_bn_relu_rvm(merge_data[0], merge_data[2], weight_data+woffset[21], weight_data+aoffset[19], weight_data+boffset[19], &stage2_conv21);

    config_conv(stage2_conv22, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  384, 384, 384); // Relu_20
    conv_bn_relu_rvm(merge_data[2], merge_data[0], weight_data+woffset[22], weight_data+aoffset[20], weight_data+boffset[20], &stage2_conv22);

    config_conv_add(stage2_conv23, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  384, 1152, 1152, 1152, 1152); // Relu_21
    conv_add_bn_relu_rvm(merge_data[0], conv_data[0], merge_data[2], conv_data[0], weight_data+woffset[23], weight_data+aoffset[21], weight_data+boffset[21], &stage2_conv23);//add_6
    if (DEBUG_PRINT) {
        printf("stage2_21_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout);
    }
    stage_end[2] = read_csr_safe(cycle);
/*
 * stage 3
*/
    // s3u1
    stage_start[3] = read_csr_safe(cycle);
    config_conv(stage3_conv24, 28, 28, 512, 1024, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0,  1152, 2176, 2176);
    conv_im2col(merge_data[2], merge_data[0], weight_data+woffset[24], &stage3_conv24);
  
    config_conv(stage3_conv25, 28, 28, 512, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 640, 640); // Relu_22
    conv_bn_relu_rvm(merge_data[3], merge_data[0], weight_data+woffset[25], weight_data+aoffset[22], weight_data+boffset[22], &stage3_conv25);

    config_conv(stage3_conv26, 28, 28, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_23
    conv_bn_relu_rvm(merge_data[0], merge_data[3], weight_data+woffset[26], weight_data+aoffset[23], weight_data+boffset[23], &stage3_conv26);

    config_conv_add(stage3_conv27, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   640, 2176, 2176, 2176, 2176); // Relu_24
    conv_add_bn_relu_rvm(merge_data[1], merge_data[2], merge_data[0], merge_data[2], weight_data+woffset[27], weight_data+aoffset[24], weight_data+boffset[24], &stage3_conv27);//add_7
    if (DEBUG_PRINT) {
        printf("stage3_24_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout);
    }

    config_conv(stage3_conv28, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_25
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[28], weight_data+aoffset[25], weight_data+boffset[25], &stage3_conv28);

    config_conv(stage3_conv29, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_26
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[29], weight_data+aoffset[26], weight_data+boffset[26], &stage3_conv29);

    config_conv_add(stage3_conv30, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_27
    conv_add_bn_relu_rvm(merge_data[0], merge_data[2], merge_data[1], merge_data[2], weight_data+woffset[30], weight_data+aoffset[27], weight_data+boffset[27], &stage3_conv30); // add_8
    if (DEBUG_PRINT) {
        printf("stage3_27_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout);
    }

    config_conv(stage3_conv31, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_28
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[31], weight_data+aoffset[28], weight_data+boffset[28], &stage3_conv31);

    config_conv(stage3_conv32, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_29
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[32], weight_data+aoffset[29], weight_data+boffset[29], &stage3_conv32);

    config_conv_add(stage3_conv33, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_30
    conv_add_bn_relu_rvm(merge_data[1], merge_data[2], merge_data[0], merge_data[2], weight_data+woffset[33], weight_data+aoffset[30], weight_data+boffset[30], &stage3_conv33); //add_9
    if (DEBUG_PRINT) {
        printf("stage3_30_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout);
    }

    config_conv(stage3_conv34, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_31
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[34], weight_data+aoffset[31], weight_data+boffset[31], &stage3_conv34);

    config_conv(stage3_conv35, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_32
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[35], weight_data+aoffset[32], weight_data+boffset[32], &stage3_conv35);

    config_conv_add(stage3_conv36, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_33
    conv_add_bn_relu_rvm(merge_data[0], merge_data[2], merge_data[1], merge_data[2], weight_data+woffset[36], weight_data+aoffset[33], weight_data+boffset[33], &stage3_conv36); //add_10
    if (DEBUG_PRINT) {
        printf("stage3_33_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout);
    }

    config_conv(stage3_conv37, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_34
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[37], weight_data+aoffset[34], weight_data+boffset[34], &stage3_conv37);

    config_conv(stage3_conv38, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_35
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[38], weight_data+aoffset[35], weight_data+boffset[35], &stage3_conv38);

    config_conv_add(stage3_conv39, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_36
    conv_add_bn_relu_rvm(merge_data[1], merge_data[2], merge_data[0], merge_data[2], weight_data+woffset[39], weight_data+aoffset[36], weight_data+boffset[36], &stage3_conv39); //add_11
    if (DEBUG_PRINT) {
        printf("stage3_36_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout);
    }

    config_conv(stage3_conv40, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_37
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[40], weight_data+aoffset[37], weight_data+boffset[37], &stage3_conv40);
 
    config_conv(stage3_conv41, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_38
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[41], weight_data+aoffset[38], weight_data+boffset[38], &stage3_conv41);

    config_conv_add(stage3_conv42, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_39
    conv_add_bn_relu_rvm(merge_data[0], merge_data[2], merge_data[1], merge_data[2], weight_data+woffset[42], weight_data+aoffset[39], weight_data+boffset[39], &stage3_conv42); // add_12
    if (DEBUG_PRINT) {
        printf("stage3_39_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout);
    }
    stage_end[3] = read_csr_safe(cycle);
    
/*
 * stage 4
*/

    stage_start[4] = read_csr_safe(cycle);
    config_conv(stage4_conv43, 14, 14, 1024, 2048, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 2176, 4224, 4224);
    conv_im2col(merge_data[2], merge_data[0], weight_data+woffset[43], &stage4_conv43);

    config_conv(stage4_conv44, 14, 14, 1024, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2176, 1152, 1152); // Relu_40
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[44], weight_data+aoffset[40], weight_data+boffset[40], &stage4_conv44);

    config_conv(stage4_conv45, 14, 14, 512, 512, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1152, 1152, 1152); // Relu_41
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[45], weight_data+aoffset[41], weight_data+boffset[41], &stage4_conv45);

    config_conv_add(stage4_conv46, 7, 7, 512, 2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   1152, 4224, 4224, 4224, 4224); // Relu_42
    conv_add_bn_relu_rvm(merge_data[1], merge_data[2], merge_data[0], merge_data[2], weight_data+woffset[46], weight_data+aoffset[42], weight_data+boffset[42], &stage4_conv46); //add_13
    if (DEBUG_PRINT) {
        printf("stage4_42_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout);
    }

    config_conv(stage4_conv47, 7, 7, 2048, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4224, 1152, 1152); // Relu_43
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[47], weight_data+aoffset[43], weight_data+boffset[43], &stage4_conv47);

    config_conv(stage4_conv48, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  1152, 1152, 1152); // Relu_44
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[48], weight_data+aoffset[44], weight_data+boffset[44], &stage4_conv48);

    config_conv_add(stage4_conv49, 7, 7, 512, 2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   1152, 4224, 4224, 4224, 4224); // Relu_45
    conv_add_bn_relu_rvm(merge_data[0], merge_data[2], merge_data[1], merge_data[2], weight_data+woffset[49], weight_data+aoffset[45], weight_data+boffset[45], &stage4_conv49); //add_14
    if (DEBUG_PRINT) {
        printf("stage4_45_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout);
    }

    config_conv(stage4_conv50, 7, 7, 2048, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4224, 1152, 1152); // Relu_46
    conv_bn_relu_rvm(merge_data[1], merge_data[0], weight_data+woffset[50], weight_data+aoffset[46], weight_data+boffset[46], &stage4_conv50);

    config_conv(stage4_conv51, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,   1152, 1152, 1152); // Relu_47
    conv_bn_relu_rvm(merge_data[0], merge_data[1], weight_data+woffset[51], weight_data+aoffset[47], weight_data+boffset[47], &stage4_conv51);

    config_conv_add(stage4_conv52, 7, 7, 512, 2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   1152, 4224, 4224, 4224, 4224); // Relu_48
    conv_add_bn_relu_rvm(merge_data[1], merge_data[2], merge_data[0], merge_data[2], weight_data+woffset[52], weight_data+aoffset[48], weight_data+boffset[48], &stage4_conv52); //add_15
    if (DEBUG_PRINT) {
        printf("stage4_48_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv52.hout, stage4_conv52.wout, stage4_conv52.cout);
    }
    stage_end[4] = read_csr_safe(cycle);
    
/**
 * end
 * 
 */
    stage_start[5] = read_csr_safe(cycle);
    config_pool(stage5_avgpool, 7, 7, 2048, 2048, 7, 7, 1, 1, 0, 0, 0, 0,  4224, 4224);
    avgpool_mean(merge_data[0], merge_data[1], &stage5_avgpool);

    matmul_rvm(merge_data[2], merge_data[0], weight_data+woffset[53], 1, 2048, 1001);

    addw(merge_data[0], merge_data[2], weight_data+biasoffset, 1*1001);

    softmax(merge_data[2], merge_data[0], 1*1001);

    cast_f32_to_f16(merge_data[0], merge_data[2], 1*1001);

#ifdef __SPIKE__
    memcpy(softmax_tensor_fp16_data, merge_data[0], 1001*2);
#endif

    stage_end[5] = read_csr_safe(cycle);

    return 0;
}
