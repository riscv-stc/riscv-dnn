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

    memcpy_rvm(buffer+4365312, buffer, 4264, 1024);

    PERF_BEGIN();
    asm("csrwi frm, 0");

    for (int j = 0; j < NLOOPS; j++) {
        cycles = read_csr_safe(cycle); 
        resnet50_base(imagenet_pic_data_data, weight_data, 8);
        cycles = read_csr_safe(cycle) - cycles;
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

int resnet50_base(void *indata, void *wdata, int num)
{

    // input(1, 224, 224, 3)
/*
 * stage 0 
*/
    config_conv(stage0_conv, 230, 230, 3, 64, 7, 7, 2, 2, 1, 1, 0, 0, 0, 0,  6, 128, 128);
    config_pool(stage0_maxpool, 112, 112, 64, 64, 3, 3, 2, 2, 0, 1, 0, 1,  128, 128); // Relu

    config_conv(stage1_conv1,      56, 56,  64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640);
    config_conv(stage1_conv2,      56, 56,  64,  64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 128, 128); // Relu_1
    config_conv(stage1_conv3,      56, 56,  64,  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  128, 128, 128); // Relu_2
    config_conv_add(stage1_conv4,  56, 56,  64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); //Relu_3
    config_conv(stage1_conv5,      56, 56, 256,  64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 128, 128); // Relu_4
    config_conv(stage1_conv6,      56, 56,  64,  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  128, 128, 128);  // Relu_5
    config_conv_add(stage1_conv7,  56, 56,  64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); // Relu_6
    config_conv(stage1_conv8,      56, 56, 256,  64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 128, 128); // Relu_7
    config_conv(stage1_conv9,      56, 56,  64,  64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  128, 128, 128); // Relu_8
    config_conv_add(stage1_conv10, 56, 56,  64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); // Relu_9

    config_conv(stage2_conv11,     56*BATCH2+8*BATCH2, 56, 256, 512, 1, 1, 2, 2, 1, 1, 0,    0, 0, 0,  640, 1152, 1152);
    config_conv(stage2_conv12,     56*BATCH2+8*BATCH2, 56, 256, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 384, 384); // Relu_10
    config_conv(stage2_conv13,     56*BATCH2+8*BATCH2, 56, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 384, 384, 384); // Relu_11
    config_conv_add(stage2_conv14, 28*BATCH2+4*BATCH2, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 384, 1152, 1152, 1152, 1152); // Relu_12
    config_conv(stage2_conv15,     28*BATCH2+4*BATCH2, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_13
    config_conv(stage2_conv16,     28*BATCH2+4*BATCH2, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 384, 384, 384); // Relu_14
    config_conv_add(stage2_conv17, 28*BATCH2+4*BATCH2, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   384, 1152, 1152, 1152, 1152); // Relu_15
    config_conv(stage2_conv18,     28*BATCH2+4*BATCH2, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_16
    config_conv(stage2_conv19,     28*BATCH2+4*BATCH2, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  384, 384, 384); // Relu_17
    config_conv_add(stage2_conv20, 28*BATCH2+4*BATCH2, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  384, 1152, 1152, 1152, 1152); // Relu_18
    config_conv(stage2_conv21,     28*BATCH2+4*BATCH2, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_19
    config_conv(stage2_conv22,     28*BATCH2+4*BATCH2, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  384, 384, 384); // Relu_20
    config_conv_add(stage2_conv23, 28*BATCH2+4*BATCH2, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  384, 1152, 1152, 1152, 1152); // Relu_21

    config_conv(stage3_conv24,     28*BATCH3+4*BATCH3, 28,  512, 1024, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0,  1152, 2176, 2176);
    config_conv(stage3_conv25,     28*BATCH3+4*BATCH3, 28,  512,  256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 640, 640); // Relu_22
    config_conv(stage3_conv26,     28*BATCH3+4*BATCH3, 28,  256,  256, 3, 3, 2, 2, 1, 1, 1, 0, 1, 1,  640,  640, 640); // Relu_23
    config_conv_add(stage3_conv27, 14*BATCH3+2*BATCH3, 14,  256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640,  2176, 2176, 2176, 2176); // Relu_24
    config_conv(stage3_conv28,     14*BATCH3+2*BATCH3, 14, 1024,  256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_25
    config_conv(stage3_conv29,     14*BATCH3+2*BATCH3, 14,  256,  256, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1,  640,  640, 640); // Relu_26
    config_conv_add(stage3_conv30, 14*BATCH3+2*BATCH3, 14,  256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640,  2176, 2176, 2176, 2176); // Relu_27
    config_conv(stage3_conv31,     14*BATCH3+2*BATCH3, 14, 1024,  256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_28
    config_conv(stage3_conv32,     14*BATCH3+2*BATCH3, 14,  256,  256, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1,  640,  640, 640); // Relu_29
    config_conv_add(stage3_conv33, 14*BATCH3+2*BATCH3, 14,  256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640,  2176, 2176, 2176, 2176); // Relu_30
    config_conv(stage3_conv34,     14*BATCH3+2*BATCH3, 14, 1024,  256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_31
    config_conv(stage3_conv35,     14*BATCH3+2*BATCH3, 14,  256,  256, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1,  640,  640, 640); // Relu_32
    config_conv_add(stage3_conv36, 14*BATCH3+2*BATCH3, 14,  256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640,  2176, 2176, 2176, 2176); // Relu_33
    config_conv(stage3_conv37,     14*BATCH3+2*BATCH3, 14, 1024,  256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_34
    config_conv(stage3_conv38,     14*BATCH3+2*BATCH3, 14,  256,  256, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1,  640,  640, 640); // Relu_35
    config_conv_add(stage3_conv39, 14*BATCH3+2*BATCH3, 14,  256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640,  2176, 2176, 2176, 2176); // Relu_36
    config_conv(stage3_conv40,     14*BATCH3+2*BATCH3, 14, 1024,  256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_37
    config_conv(stage3_conv41,     14*BATCH3+2*BATCH3, 14,  256,  256, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1,  640,  640, 640); // Relu_38
    config_conv_add(stage3_conv42, 14*BATCH3+2*BATCH3, 14,  256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640,  2176, 2176, 2176, 2176); // Relu_39
    
    // stage 0: padding && cast weight && conv
    for (int i = 0; i < 2; i++) {

        for (int j = 0; j < 2; j++) {

            for (int k = 0; k < 2; k++) {
                stage_start[0] = read_csr_safe(cycle);
                conv_im2col_small_cin(buffer+k*STAGE_2240KB, indata+(i*BATCH3+j*BATCH2+k)*PICTURE_SIZE, wdata+woffset[0], &stage0_conv);
                maxpool_bn_relu(buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, buffer+k*STAGE_2240KB, wdata+aoffset[0], wdata+boffset[0], &stage0_maxpool);
                stage_end[0] = read_csr_safe(cycle);

                stage_start[1] = read_csr_safe(cycle);
                conv_rvm_n_k64(buffer+k*STAGE_2240KB, buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, wdata+woffset[1], &stage1_conv1);
                conv_bn_relu_rvm_n_k64(buffer+STAGE_2240KB*BATCH2, buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, wdata+woffset[2], wdata+aoffset[1], wdata+boffset[1], &stage1_conv2);
                conv_bn_relu_rvm_3x3(buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, buffer+STAGE_2240KB*BATCH2, wdata+woffset[3], wdata+aoffset[2], wdata+boffset[2], &stage1_conv3);
                conv_add_bn_relu_rvm_n_k64(buffer+STAGE_2240KB*BATCH2, buffer+k*STAGE_2240KB, buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, buffer+k*STAGE_2240KB, wdata+woffset[4], wdata+aoffset[3], wdata+boffset[3], &stage1_conv4); //add

                conv_bn_relu_rvm_1x1(buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, buffer+STAGE_2240KB*BATCH2, wdata+woffset[5], wdata+aoffset[4], wdata+boffset[4], &stage1_conv5);
                conv_bn_relu_rvm_3x3(buffer+STAGE_2240KB*BATCH2+STAGE_2240KB+STAGE_392KB,  buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, wdata+woffset[6], wdata+aoffset[5], wdata+boffset[5], &stage1_conv6);
                conv_add_bn_relu_rvm_n_k64(buffer+STAGE_2240KB*BATCH2, buffer+k*STAGE_2240KB, buffer+STAGE_2240KB*BATCH2+STAGE_2240KB+STAGE_392KB, buffer+k*STAGE_2240KB, wdata+woffset[7], wdata+aoffset[6], wdata+boffset[6], &stage1_conv7); // add_1

                conv_bn_relu_rvm_1x1(buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, buffer+STAGE_2240KB*BATCH2, wdata+woffset[8], wdata+aoffset[7], wdata+boffset[7], &stage1_conv8);
                conv_bn_relu_rvm_3x3(buffer+STAGE_2240KB*BATCH2+STAGE_2240KB+STAGE_392KB, buffer+STAGE_2240KB*BATCH2+STAGE_2240KB, wdata+woffset[9], wdata+aoffset[8], wdata+boffset[8], &stage1_conv9);
                conv_add_bn_relu_rvm_n_k64(buffer+k*STAGE_2240KB, buffer+STAGE_2240KB*BATCH2, buffer+STAGE_2240KB*BATCH2+STAGE_2240KB+STAGE_392KB, buffer+k*STAGE_2240KB, wdata+woffset[10], wdata+aoffset[9], wdata+boffset[9], &stage1_conv10); //add_2
                stage_end[1] = read_csr_safe(cycle);
            }
    
/*
 * stage 2
*/
    // s2u1
            
            stage_start[2] = read_csr_safe(cycle);

            conv_im2col_1x1_2(buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, buffer, wdata+woffset[11], &stage2_conv11);
            conv_bn_relu_rvm_1x1(buffer, buffer, wdata+woffset[12], wdata+aoffset[10], wdata+boffset[10], &stage2_conv12);
            conv_bn_relu_rvm_3x3(buffer+STAGE_2240KB, buffer, wdata+woffset[13], wdata+aoffset[11], wdata+boffset[11], &stage2_conv13);
            conv_add_bn_relu_rvm_1x1(buffer, buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, buffer+STAGE_2240KB, buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, wdata+woffset[14], wdata+aoffset[12], wdata+boffset[12], &stage2_conv14); //add_3


            conv_bn_relu_rvm_1x1(buffer+STAGE_2240KB, buffer, wdata+woffset[15], wdata+aoffset[13], wdata+boffset[13], &stage2_conv15);
            conv_bn_relu_rvm_3x3(buffer, buffer+STAGE_2240KB, wdata+woffset[16], wdata+aoffset[14], wdata+boffset[14], &stage2_conv16);
            conv_add_bn_relu_rvm_1x1(buffer+STAGE_2240KB, buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, buffer, buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, wdata+woffset[17], wdata+aoffset[15], wdata+boffset[15], &stage2_conv17); //add_4

            conv_bn_relu_rvm_1x1(buffer, buffer+STAGE_2240KB, wdata+woffset[18], wdata+aoffset[16], wdata+boffset[16], &stage2_conv18);
            conv_bn_relu_rvm_3x3(buffer+STAGE_2240KB, buffer, wdata+woffset[19], wdata+aoffset[17], wdata+boffset[17], &stage2_conv19);
            conv_add_bn_relu_rvm_1x1(buffer, buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, buffer+STAGE_2240KB, buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, wdata+woffset[20], wdata+aoffset[18], wdata+boffset[18], &stage2_conv20);//add_5

            conv_bn_relu_rvm_1x1(buffer+STAGE_2240KB, buffer, wdata+woffset[21], wdata+aoffset[19], wdata+boffset[19], &stage2_conv21);
            conv_bn_relu_rvm_3x3(buffer, buffer+STAGE_2240KB, wdata+woffset[22], wdata+aoffset[20], wdata+boffset[20], &stage2_conv22);
            conv_add_bn_relu_rvm_1x1(buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, buffer+STAGE_2240KB, buffer, buffer+STAGE_2240KB*BATCH2+j*STAGE_1011KB5*BATCH2, wdata+woffset[23], wdata+aoffset[21], wdata+boffset[21], &stage2_conv23);//add_6
            stage_end[2] = read_csr_safe(cycle);
        }
    

/*
 * stage 3
*/
    // s3u1
        
        stage_start[3] = read_csr_safe(cycle);
    
        conv_rvm_k(buffer+i*STAGE_476KB*BATCH3, buffer+STAGE_2240KB*BATCH2, wdata+woffset[24], &stage3_conv24);
    
        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+STAGE_2240KB*BATCH2, wdata+woffset[25], wdata+aoffset[22], wdata+boffset[22], &stage3_conv25);
        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_560KB*BATCH3, buffer+STAGE_476KB*BATCH, wdata+woffset[26], wdata+aoffset[23], wdata+boffset[23], &stage3_conv26);
        conv_add_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+i*STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH+STAGE_560KB*BATCH3, buffer+i*STAGE_476KB*BATCH3, wdata+woffset[27], wdata+aoffset[24], wdata+boffset[24], &stage3_conv27);//add_7

        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH, wdata+woffset[28], wdata+aoffset[25], wdata+boffset[25], &stage3_conv28);
        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, wdata+woffset[29], wdata+aoffset[26], wdata+boffset[26], &stage3_conv29);
        conv_add_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+i*STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH, buffer+i*STAGE_476KB*BATCH3, wdata+woffset[30], wdata+aoffset[27], wdata+boffset[27], &stage3_conv30); // add_8

        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, wdata+woffset[31], wdata+aoffset[28], wdata+boffset[28], &stage3_conv31);
        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH, wdata+woffset[32], wdata+aoffset[29], wdata+boffset[29], &stage3_conv32);
        conv_add_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+i*STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+i*STAGE_476KB*BATCH3, wdata+woffset[33], wdata+aoffset[30], wdata+boffset[30], &stage3_conv33); //add_9

        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH, wdata+woffset[34], wdata+aoffset[31], wdata+boffset[31], &stage3_conv34);
        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, wdata+woffset[35], wdata+aoffset[32], wdata+boffset[32], &stage3_conv35);
        conv_add_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+i*STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH, buffer+i*STAGE_476KB*BATCH3, wdata+woffset[36], wdata+aoffset[33], wdata+boffset[33], &stage3_conv36); //add_10

        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, wdata+woffset[37], wdata+aoffset[34], wdata+boffset[34], &stage3_conv37);
        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH, wdata+woffset[38], wdata+aoffset[35], wdata+boffset[35], &stage3_conv38);
        conv_add_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+i*STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+i*STAGE_476KB*BATCH3, wdata+woffset[39], wdata+aoffset[36], wdata+boffset[36], &stage3_conv39); //add_11


        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH, wdata+woffset[40], wdata+aoffset[37], wdata+boffset[37], &stage3_conv40);
        conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3, wdata+woffset[41], wdata+aoffset[38], wdata+boffset[38], &stage3_conv41);
        conv_add_bn_relu_rvm_k(buffer+i*STAGE_476KB*BATCH3, buffer+STAGE_476KB*BATCH+STAGE_476KB*BATCH3,buffer+STAGE_476KB*BATCH, buffer+i*STAGE_476KB*BATCH3, wdata+woffset[42], wdata+aoffset[39], wdata+boffset[39], &stage3_conv42); // add_12
        stage_end[3] = read_csr_safe(cycle);
    }
    

/*
 * stage 4
*/
    config_conv(stage4_conv43,     14*BATCH+2*BATCH, 14, 1024, 2048, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 2176, 4224, 4224);
    config_conv(stage4_conv44,     14*BATCH+2*BATCH, 14, 1024,  512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2176, 1152, 1152); // Relu_40
    config_conv(stage4_conv45,     14*BATCH+2*BATCH, 14,  512,  512, 3, 3, 2, 2, 1, 1, 1, 0, 1, 1, 1152, 1152, 1152); // Relu_41
    config_conv_add(stage4_conv46, 7*BATCH+BATCH,     7,  512, 2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 4224, 4224, 4224, 4224); // Relu_42
    config_conv(stage4_conv47,     7*BATCH+BATCH,     7, 2048,  512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4224, 1152, 1152); // Relu_43
    config_conv(stage4_conv48,     7*BATCH+BATCH,     7,  512,  512, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1, 1152, 1152, 1152); // Relu_44, clr berfor compute
    config_conv_add(stage4_conv49, 7*BATCH+BATCH,     7,  512, 2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 4224, 4224, 4224, 4224); // Relu_45
    config_conv(stage4_conv50,     7*BATCH+BATCH,     7, 2048,  512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4224, 1152, 1152); // Relu_46
    config_conv(stage4_conv51,     7*BATCH+BATCH,     7,  512,  512, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1, 1152, 1152, 1152); // Relu_47, clr berfor compute
    config_conv_add(stage4_conv52, 7*BATCH+BATCH,     7,  512, 2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 4224, 4224, 4224, 4224); // Relu_48

    stage_start[4] = read_csr_safe(cycle);
    conv_rvm_k(buffer+STAGE_476KB*BATCH, buffer, wdata+woffset[43], &stage4_conv43);
    conv_bn_relu_rvm_k(buffer+STAGE_476KB*BATCH+STAGE_231KB*BATCH, buffer, wdata+woffset[44], wdata+aoffset[40], wdata+boffset[40], &stage4_conv44);
    conv_bn_relu_rvm_k(buffer, buffer+STAGE_476KB*BATCH+STAGE_231KB*BATCH, wdata+woffset[45], wdata+aoffset[41], wdata+boffset[41], &stage4_conv45);
    conv_add_bn_relu_rvm_k(buffer+STAGE_63KB*BATCH, buffer+STAGE_476KB*BATCH, buffer, buffer+STAGE_476KB*BATCH, wdata+woffset[46], wdata+aoffset[42], wdata+boffset[42], &stage4_conv46); //add_13


    conv_bn_relu_rvm_k(buffer, buffer+STAGE_63KB*BATCH, wdata+woffset[47], wdata+aoffset[43], wdata+boffset[43], &stage4_conv47);
    conv_bn_relu_rvm_k(buffer+STAGE_231KB*BATCH, buffer, wdata+woffset[48], wdata+aoffset[44], wdata+boffset[44], &stage4_conv48);
    conv_add_bn_relu_rvm_k(buffer, buffer+STAGE_476KB*BATCH, buffer+STAGE_231KB*BATCH, buffer+STAGE_476KB*BATCH, wdata+woffset[49], wdata+aoffset[45], wdata+boffset[45], &stage4_conv49); //add_14


    conv_bn_relu_rvm_k(buffer, buffer+STAGE_231KB*BATCH, wdata+woffset[50], wdata+aoffset[46], wdata+boffset[46], &stage4_conv50);
    conv_bn_relu_rvm_k(buffer+STAGE_231KB*BATCH, buffer, wdata+woffset[51], wdata+aoffset[47], wdata+boffset[47], &stage4_conv51);
    conv_add_bn_relu_rvm_k(buffer, buffer+STAGE_476KB*BATCH, buffer+STAGE_231KB*BATCH, buffer+STAGE_476KB*BATCH, wdata+woffset[52], wdata+aoffset[48], wdata+boffset[48], &stage4_conv52); //add_15

    stage_end[4] = read_csr_safe(cycle);
    
/**
 * end
 * 
 */
    stage_start[5] = read_csr_safe(cycle);
    config_pool(stage5_avgpool, 7*BATCH+BATCH, 7, 2048, 2048, 7, 7, 8, 1, 0, 0, 0, 0,  4224, 4224);
    avgpool_mean(buffer+STAGE_231KB*BATCH, buffer, &stage5_avgpool);

    matmul_rvm(buffer+STAGE_2KB*BATCH, buffer+STAGE_231KB*BATCH, wdata+woffset[53], BATCH, 2048, 1001);

    addw(buffer, buffer+STAGE_2KB*BATCH, wdata+biasoffset, BATCH*1001);

    softmax(buffer+STAGE_2KB*BATCH, buffer, 1*1001);

    cast_f32_to_f16(buffer, buffer+STAGE_2KB*BATCH, 1*1001);

// #ifdef __SPIKE__
//     memcpy(softmax_tensor_fp16_data, merge_data[0], 1001*2);
// #endif

    stage_end[5] = read_csr_safe(cycle);

    return 0;
}
