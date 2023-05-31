#include <stdio.h>

#include "resnet.h"
#include "../../../src/hpm.h"
#include "../../../src/encoding.h"
#include "../../../src/perf.h"

#include "params.h"

uint64_t stage_start[6], stage_end[6];

// #define PICTURE_SIZE  301056 // 224 * 224 * 3 * DATASIZE
#define PICTURE_SIZE  317400 // 230 * 230 * 3 * DATASIZE

int main()
{   
    // const int num_pictures = N;
    uint64_t cycles=0;
    asm("csrwi frm, 0");
    PERF_BEGIN();
    for (int j = 0; j < NLOOPS; j++) {
        cycles = read_csr_safe(cycle);
        resnet50_ncores(imagenet_pic_data_data, 16, CORENUMS);
        cycles = read_csr_safe(cycle) -cycles;
    }
    PERF_END();

    barrier(CORENUMS);

    return 0;
}

int resnet50_ncores(uint8_t *indata, int pics, int ncores)
{

    // input(1, 224, 224, 3)
/*
 * stage 0 8cores for 1 picture
*/
    int pid = read_csr(mhartid);
    stage_start[0] = read_csr_safe(cycle);
    // stage 0: padding && cast weight && conv
    config_conv(stage0_conv, 230, 230, 3, 64, 7, 7, 2, 2, 1, 1, 0, 0, 0, 0,  6, 128, 128);
    config_pool(stage0_maxpool, 112, 112, 64, 64, 3, 3, 2, 2, 0, 1, 0, 1,  128, 128); // Relu

    config_conv(stage1_conv1, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640);
    config_conv(stage1_conv2, 56, 56, 64, 64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 128, 128); // Relu_1
    config_conv(stage1_conv3, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  128, 128, 128); // Relu_2
    config_conv_add(stage1_conv4, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); //Relu_3
    config_conv(stage1_conv5, 56, 56, 256, 64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 128, 128); // Relu_4
    config_conv(stage1_conv6, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 128, 128, 128);  // Relu_5
    config_conv_add(stage1_conv7, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); // Relu_6
    config_conv(stage1_conv8, 56, 56, 256, 64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 128, 128); // Relu_7
    config_conv(stage1_conv9, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  128, 128, 128); // Relu_8
    config_conv_add(stage1_conv10, 56, 56, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  128, 640, 640, 640, 640); // Relu_9
    config_conv(stage2_conv11, 56, 56, 256, 512, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0,  640, 1152, 1152);
    config_conv(stage2_conv12, 56, 56, 256, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 384, 384); // Relu_10
    config_conv(stage2_conv13, 56, 56, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 384, 384, 384); // Relu_11
    config_conv_add(stage2_conv14, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 384, 1152, 1152, 1152, 1152); // Relu_12
    config_conv(stage2_conv15, 28, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_13
    config_conv(stage2_conv16, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 384, 384, 384); // Relu_14
    config_conv_add(stage2_conv17, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   384, 1152, 1152, 1152, 1152); // Relu_15
    config_conv(stage2_conv18, 28, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_16
    config_conv(stage2_conv19, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  384, 384, 384); // Relu_17
    config_conv_add(stage2_conv20, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  384, 1152, 1152, 1152, 1152); // Relu_18
    config_conv(stage2_conv21, 28, 28, 512, 128, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1152, 384, 384); // Relu_19
    config_conv(stage2_conv22, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  384, 384, 384); // Relu_20
    config_conv_add(stage2_conv23, 28, 28, 128, 512, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  384, 1152, 1152, 1152, 1152); // Relu_21

    int ncores1 = 4;
    int ncores3 = 1;
    for (int step=0; step < 2; step++) {
        for (int i = 0; i < 4; i++) {
            conv_ncores_hout_small_cin(buffer+2*STAGE_392KB+pid/ncores1*STAGE_392KB, indata+(step*8 + i*2 + pid/ncores1)*PICTURE_SIZE, 
                                    weight_data+woffset[0], &stage0_conv,
                                    ncores1, pid);

            barrier(ncores);

            maxpool_bn_relu_ncores_hout(buffer+(pid/ncores1)*STAGE_392KB, buffer+2*STAGE_392KB+pid/ncores1*STAGE_392KB,
                                        weight_data+aoffset[0], weight_data+boffset[0], &stage0_maxpool,
                                        ncores1, pid);
/*
 * stage 1  4cores for 1 picture
 *  392  392  1960 1960 1960 1960
 *  1960 1960 392  392  1960 1960
 * |----|----|----|----|----|----|
 *     conv     conv       add
*/
    // s1u1

    
            conv_ncores_hout(buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB, buffer+(pid/ncores1)*STAGE_392KB,
                            weight_data+woffset[1], &stage1_conv1,
                            ncores1, pid);

            conv_bn_relu_ncores_hout(buffer+2*STAGE_1960KB+(pid/ncores1)*STAGE_392KB, buffer+(pid/ncores1)*STAGE_392KB,
                                    weight_data+woffset[2], weight_data+aoffset[1], weight_data+boffset[1], &stage1_conv2,
                                    ncores1, pid);
        
            barrier(ncores);
        
            conv_bn_relu_ncores_hout(buffer+(pid/ncores1)*STAGE_392KB, buffer+2*STAGE_1960KB+(pid/ncores1)*STAGE_392KB,
                                    weight_data+woffset[3], weight_data+aoffset[2], weight_data+boffset[2], &stage1_conv3,
                                    ncores1, pid);
        
            conv_add_bn_relu_ncores_hout(buffer+2*STAGE_392KB+(pid/ncores1)*STAGE_1960KB, buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                                        buffer+(pid/ncores1)*STAGE_392KB, buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                                        weight_data+woffset[4], weight_data+aoffset[3], weight_data+boffset[3], &stage1_conv4,
                                        ncores1, pid);

            conv_bn_relu_ncores_hout(buffer+(pid/ncores1)*STAGE_392KB, buffer+2*STAGE_392KB+(pid/ncores1)*STAGE_1960KB,
                                    weight_data+woffset[5], weight_data+aoffset[4], weight_data+boffset[4], &stage1_conv5,
                                    ncores1, pid);
        
            barrier(ncores);

            conv_bn_relu_ncores_hout(buffer+2*STAGE_1960KB+(pid/ncores1)*STAGE_392KB, buffer+(pid/ncores1)*STAGE_392KB,
                                    weight_data+woffset[6], weight_data+aoffset[5], weight_data+boffset[5], &stage1_conv6,
                                    ncores1, pid);

            conv_add_bn_relu_ncores_hout(buffer+(pid/ncores1)*STAGE_1960KB, buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                                        buffer+2*STAGE_1960KB+(pid/ncores1)*STAGE_392KB, buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                                        weight_data+woffset[7], weight_data+aoffset[6], weight_data+boffset[6], &stage1_conv7,
                                        ncores1, pid);
            
            conv_bn_relu_ncores_hout(buffer+2*STAGE_1960KB+(pid/ncores1)*STAGE_392KB, buffer+(pid/ncores1)*STAGE_1960KB,
                                    weight_data+woffset[8], weight_data+aoffset[7], weight_data+boffset[7], &stage1_conv8,
                                    ncores1, pid);

            barrier(ncores);

            conv_bn_relu_ncores_hout(buffer+(pid/ncores1)*STAGE_392KB, buffer+2*STAGE_1960KB+(pid/ncores1)*STAGE_392KB,
                                weight_data+woffset[9], weight_data+aoffset[8], weight_data+boffset[8], &stage1_conv9,
                                ncores1, pid);
    
            conv_add_bn_relu_last_ncores_hout(buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                                            buffer+(pid/ncores1)*STAGE_392KB, buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                                            weight_data+woffset[10], weight_data+aoffset[9], weight_data+boffset[9], &stage1_conv10,
                                            ncores1, pid);
            barrier(ncores);
/*
 * stage 2 2cores for 1 picture
*   882  882  882  882  294  294
 *  882  882  294  294  882  882
 * |----|----|----|----|----|----|
 *     add     conv       conv
*/
    // s2u1
    
            conv_ncores_hout(buffer+(pid/ncores1)*STAGE_882KB, buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                            weight_data+woffset[11], &stage2_conv11,
                            ncores1, pid);

            
            conv_bn_relu_ncores_hout(buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_1176KB, buffer+2*STAGE_392KB+2*STAGE_1960KB+(pid/ncores1)*STAGE_1960KB,
                                    weight_data+woffset[12], weight_data+aoffset[10], weight_data+boffset[10], &stage2_conv12,
                                    ncores1, pid);

            barrier(ncores);

            
            conv_bn_relu_ncores_hout(buffer+2*STAGE_882KB+2*STAGE_1176KB+(pid/ncores1)*STAGE_294KB, buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_1176KB,
                                    weight_data+woffset[13], weight_data+aoffset[11], weight_data+boffset[11], &stage2_conv13,
                                    ncores1, pid);

            
            conv_add_bn_relu_ncores_hout(buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_882KB, buffer+(pid/ncores1)*STAGE_882KB,
                                        buffer+2*STAGE_882KB+2*STAGE_1176KB+(pid/ncores1)*STAGE_294KB, buffer+(pid/ncores1)*STAGE_882KB,
                                        weight_data+woffset[14], weight_data+aoffset[12], weight_data+boffset[12], &stage2_conv14,
                                        ncores1, pid);
        
            conv_bn_relu_ncores_hout(buffer+2*STAGE_882KB+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_882KB,
                                    weight_data+woffset[15], weight_data+aoffset[13], weight_data+boffset[13], &stage2_conv15,
                                    ncores1, pid);

            barrier(ncores);

            
            conv_bn_relu_ncores_hout(buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+2*STAGE_882KB+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB,
                                    weight_data+woffset[16], weight_data+aoffset[14], weight_data+boffset[14], &stage2_conv16,
                                    ncores1, pid);

            
            conv_add_bn_relu_ncores_hout(buffer+2*STAGE_882KB+2*STAGE_294KB+(pid/ncores1)*STAGE_882KB, buffer+(pid/ncores1)*STAGE_882KB,
                                        buffer+(pid/ncores1)*STAGE_294KB, buffer+(pid/ncores1)*STAGE_882KB,
                                        weight_data+woffset[17], weight_data+aoffset[15], weight_data+boffset[15], &stage2_conv17,
                                        ncores1, pid);

            
            
            conv_bn_relu_ncores_hout(buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+2*STAGE_882KB+2*STAGE_294KB+(pid/ncores1)*STAGE_882KB,
                                    weight_data+woffset[18], weight_data+aoffset[16], weight_data+boffset[16], &stage2_conv18,
                                    ncores1, pid);

            barrier(ncores);

            
            
            conv_bn_relu_ncores_hout(buffer+2*STAGE_882KB+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB,
                                    weight_data+woffset[19], weight_data+aoffset[17], weight_data+boffset[17], &stage2_conv19,
                                    ncores1, pid);
            
            conv_add_bn_relu_ncores_hout(buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_882KB, buffer+6*STAGE_882KB+(pid/ncores1)*STAGE_882KB,
                                        buffer+2*STAGE_882KB+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+(pid/ncores1)*STAGE_882KB,
                                        weight_data+woffset[20], weight_data+aoffset[18], weight_data+boffset[18], &stage2_conv20,
                                        ncores1, pid);

            conv_bn_relu_ncores_hout(buffer+2*STAGE_882KB+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+2*STAGE_882KB+(pid/ncores1)*STAGE_882KB,
                                    weight_data+woffset[21], weight_data+aoffset[19], weight_data+boffset[19], &stage2_conv21,
                                    ncores1, pid);

            barrier(ncores);

            conv_bn_relu_ncores_hout(buffer+4*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+2*STAGE_882KB+2*STAGE_882KB+(pid/ncores1)*STAGE_294KB,
                                    weight_data+woffset[22], weight_data+aoffset[20], weight_data+boffset[20], &stage2_conv22,
                                    ncores1, pid);

            
            conv_add_bn_relu_last_ncores_hout(buffer+(pid/ncores+i*2)*STAGE_882KB,
                                            buffer+8*STAGE_882KB+(pid/ncores1)*STAGE_294KB, buffer+6*STAGE_882KB+(pid/ncores1)*STAGE_882KB,
                                            weight_data+woffset[23], weight_data+aoffset[21], weight_data+boffset[21], &stage2_conv23,
                                            ncores1, pid);

        }
        barrier(ncores);

/*
 * stage 3, 1 core for 1 picture
*/
    // s3u1 


        config_conv(stage3_conv24, 28, 28, 512, 1024, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0,  1152, 2176, 2176);
        config_conv(stage3_conv25, 28, 28, 512, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 640, 640); // Relu_22
        config_conv(stage3_conv26, 28, 28, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_23
        config_conv_add(stage3_conv27, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,   640, 2176, 2176, 2176, 2176); // Relu_24
        config_conv(stage3_conv28, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_25
        config_conv(stage3_conv29, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_26
        config_conv_add(stage3_conv30, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_27
        config_conv(stage3_conv31, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_28
        config_conv(stage3_conv32, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_29
        config_conv_add(stage3_conv33, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_30
        config_conv(stage3_conv34, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_31
        config_conv(stage3_conv35, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_32
        config_conv_add(stage3_conv36, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_33
        config_conv(stage3_conv37, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_34
        config_conv(stage3_conv38, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_35
        config_conv_add(stage3_conv39, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_36
        config_conv(stage3_conv40, 14, 14, 1024, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  2176, 640, 640); // Relu_37
        config_conv(stage3_conv41, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,  640, 640, 640); // Relu_38
        config_conv_add(stage3_conv42, 14, 14, 256, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  640, 2176, 2176, 2176, 2176); // Relu_39

        conv_im2col(buffer+8*STAGE_882KB+pid*STAGE_416KB5, buffer+pid*STAGE_882KB,
                    weight_data+woffset[24], &stage3_conv24);
        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+pid*STAGE_882KB,
                        weight_data+woffset[25], weight_data+aoffset[22], weight_data+boffset[22], &stage3_conv25);

        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_490KB, buffer+pid*STAGE_882KB,
                        weight_data+woffset[26], weight_data+aoffset[23], weight_data+boffset[23], &stage3_conv26);

        
        conv_add_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                            buffer+pid*STAGE_882KB+STAGE_490KB, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                            weight_data+woffset[27], weight_data+aoffset[24], weight_data+boffset[24], &stage3_conv27);//add_7

        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+pid*STAGE_882KB,
                        weight_data+woffset[28], weight_data+aoffset[25], weight_data+boffset[25], &stage3_conv28);

        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+pid*STAGE_882KB+STAGE_416KB5,
                        weight_data+woffset[29], weight_data+aoffset[26], weight_data+boffset[26], &stage3_conv29);

        
        conv_add_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        buffer+pid*STAGE_882KB, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        weight_data+woffset[30], weight_data+aoffset[27], weight_data+boffset[27], &stage3_conv30); // add_8

        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+pid*STAGE_882KB+STAGE_416KB5,
                        weight_data+woffset[31], weight_data+aoffset[28], weight_data+boffset[28], &stage3_conv31);

        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+pid*STAGE_882KB,
                        weight_data+woffset[32], weight_data+aoffset[29], weight_data+boffset[29], &stage3_conv32);

        
        conv_add_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        weight_data+woffset[33], weight_data+aoffset[30], weight_data+boffset[30], &stage3_conv33); //add_9

    
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+pid*STAGE_882KB,
                        weight_data+woffset[34], weight_data+aoffset[31], weight_data+boffset[31], &stage3_conv34);

        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+pid*STAGE_882KB+STAGE_416KB5,
                        weight_data+woffset[35], weight_data+aoffset[32], weight_data+boffset[32], &stage3_conv35);

        
        conv_add_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        buffer+pid*STAGE_882KB, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        weight_data+woffset[36], weight_data+aoffset[33], weight_data+boffset[33], &stage3_conv36); //add_10

        conv_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+pid*STAGE_882KB+STAGE_416KB5,
                        weight_data+woffset[37], weight_data+aoffset[34], weight_data+boffset[34], &stage3_conv37);

        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+pid*STAGE_882KB,
                        weight_data+woffset[38], weight_data+aoffset[35], weight_data+boffset[35], &stage3_conv38);

        
        conv_add_bn_relu_rvm(buffer+pid*STAGE_882KB, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                        weight_data+woffset[39], weight_data+aoffset[36], weight_data+boffset[36], &stage3_conv39); //add_11
        
        conv_bn_relu_rvm(buffer+pid*STAGE_882KB+STAGE_416KB5, buffer+pid*STAGE_882KB,
                        weight_data+woffset[40], weight_data+aoffset[37], weight_data+boffset[37], &stage3_conv40);

    
        conv_bn_relu_rvm(buffer+8*STAGE_882KB+8*STAGE_416KB5+pid*STAGE_122KB5, buffer+pid*STAGE_882KB+STAGE_416KB5,
                        weight_data+woffset[41], weight_data+aoffset[38], weight_data+boffset[38], &stage3_conv41);

        
        conv_add_bn_relu_last(buffer+(step*8+pid)*STAGE_416KB5,
                            buffer+8*STAGE_882KB+8*STAGE_416KB5+pid*STAGE_122KB5, buffer+8*STAGE_882KB+pid*STAGE_416KB5,
                            weight_data+woffset[42], weight_data+aoffset[39], weight_data+boffset[39], &stage3_conv42); // add_12

        barrier(ncores);
    }
    

/*
 * stage 4 1core for 2pictures at the same time
*/

    config_conv(stage4_conv43,     28, 14, 1024, 2048, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 2176, 4224, 4224);
    config_conv(stage4_conv44,     28, 14, 1024, 512,  1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2176, 1152, 1152); // Relu_40
    config_conv(stage4_conv45,     14, 14, 512,  512,  3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1152, 1152, 1152); // Relu_41
    config_conv_add(stage4_conv46, 14, 7,  512,  2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 4224, 4224, 4224, 4224); // Relu_42
    config_conv(stage4_conv47,     14, 7,  2048, 512,  1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4224, 1152, 1152); // Relu_43
    config_conv(stage4_conv48,     7,  7,  512,  512,  3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1152, 1152, 1152); // Relu_44
    config_conv_add(stage4_conv49, 14, 7,  512,  2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 4224, 4224, 4224, 4224); // Relu_45
    config_conv(stage4_conv50,     14, 7,  2048, 512,  1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4224, 1152, 1152); // Relu_46
    config_conv(stage4_conv51,     7,  7,  512,  512,  3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1152, 1152, 1152); // Relu_47
    config_conv_add(stage4_conv52, 14, 7,  512,  2048, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1152, 4224, 4224, 4224, 4224); // Relu_48

    conv_im2col(buffer+16*STAGE_416KB5+(pid*2)*STAGE_202KB, buffer+(pid*2)*STAGE_416KB5,
                weight_data+woffset[43], &stage4_conv43);
        
    conv_bn_relu_rvm(buffer+(pid*2)*STAGE_220KB5, buffer+(pid*2)*STAGE_416KB5,
                    weight_data+woffset[44], weight_data+aoffset[40], weight_data+boffset[40], &stage4_conv44);

    conv_bn_relu_rvm(buffer+16*STAGE_220KB5+(pid*2)*STAGE_55KB, buffer+(pid*2)*STAGE_220KB5,
                    weight_data+woffset[45], weight_data+aoffset[41], weight_data+boffset[41], &stage4_conv45);
    conv_bn_relu_rvm(buffer+16*STAGE_220KB5+(pid*2+1)*STAGE_55KB, buffer+(pid*2+1)*STAGE_220KB5,
                    weight_data+woffset[45], weight_data+aoffset[41], weight_data+boffset[41], &stage4_conv45);
    

    conv_add_bn_relu_rvm(buffer+(pid*2)*STAGE_220KB5, buffer+16*STAGE_416KB5+(pid*2)*STAGE_202KB,
                        buffer+16*STAGE_220KB5+(pid*2)*STAGE_55KB, buffer+16*STAGE_416KB5+(pid*2)*STAGE_202KB,
                    weight_data+woffset[46], weight_data+aoffset[42], weight_data+boffset[42], &stage4_conv46); //add_13

    conv_bn_relu_rvm(buffer+16*STAGE_220KB5+(pid*2)*STAGE_55KB, buffer+(pid*2)*STAGE_220KB5,
                    weight_data+woffset[47], weight_data+aoffset[43], weight_data+boffset[43], &stage4_conv47);

    conv_bn_relu_rvm(buffer+(pid*2)*STAGE_55KB, buffer+16*STAGE_220KB5+(pid*2)*STAGE_55KB,
                    weight_data+woffset[48], weight_data+aoffset[44], weight_data+boffset[44], &stage4_conv48);
    conv_bn_relu_rvm(buffer+(pid*2+1)*STAGE_55KB, buffer+16*STAGE_220KB5+(pid*2+1)*STAGE_55KB,
                    weight_data+woffset[48], weight_data+aoffset[44], weight_data+boffset[44], &stage4_conv48);

        
    conv_add_bn_relu_rvm(buffer+16*STAGE_55KB+(pid*2)*STAGE_202KB, buffer+16*STAGE_416KB5+(pid*2)*STAGE_202KB,
                        buffer+(pid*2)*STAGE_55KB, buffer+16*STAGE_416KB5+(pid*2)*STAGE_202KB,
                        weight_data+woffset[49], weight_data+aoffset[45], weight_data+boffset[45], &stage4_conv49); //add_14
        
    conv_bn_relu_rvm(buffer+(pid*2)*STAGE_55KB, buffer+16*STAGE_55KB+(pid*2)+STAGE_202KB,
                    weight_data+woffset[50], weight_data+aoffset[46], weight_data+boffset[46], &stage4_conv50);

    conv_bn_relu_rvm(buffer+16*STAGE_202KB+(pid*2)*STAGE_55KB, buffer+(pid*2)*STAGE_55KB,
                    weight_data+woffset[51], weight_data+aoffset[47], weight_data+boffset[47], &stage4_conv51);
    conv_bn_relu_rvm(buffer+16*STAGE_202KB+(pid*2+1)*STAGE_55KB, buffer+(pid*2+1)*STAGE_55KB,
                    weight_data+woffset[51], weight_data+aoffset[47], weight_data+boffset[47], &stage4_conv51);

        
    conv_add_bn_relu_last(buffer+(pid*2)*STAGE_202KB,
                    buffer+16*STAGE_202KB+(pid*2)*STAGE_55KB, buffer+16*STAGE_416KB5+(pid*2)*STAGE_202KB,
                    weight_data+woffset[51], weight_data+aoffset[47], weight_data+boffset[47], &stage4_conv51);


/**
 * post
 * 
 */
    stage_start[5] = read_csr_safe(cycle);
    config_pool(stage5_avgpool, 7, 7, 2048, 2048, 7, 7, 1, 1, 0, 0, 0, 0,  4224, 4224);
    avgpool_mean(buffer+16*STAGE_202KB+(pid*2)*4096, buffer+(pid*2)*STAGE_202KB, &stage5_avgpool);
    avgpool_mean(buffer+16*STAGE_202KB+(pid*2+1)*4096, buffer+(pid*2+1)*STAGE_202KB, &stage5_avgpool);

    matmul_rvm(buffer+(pid*2)*4004, buffer+16*STAGE_202KB+(pid*2)*4096, weight_data+woffset[53], 2, 2048, 1001);

    addw(buffer+16*4004+(pid*2)*4004, buffer+(pid*2)*4004, weight_data+biasoffset, 1*1001);
    addw(buffer+16*4004+(pid*2+1)*4004, buffer+(pid*2+1)*4004, weight_data+biasoffset, 1*1001);

    softmax(buffer+(pid*2)*4004, buffer+16*4004+(pid*2)*4004, 1*1001);
    softmax(buffer+(pid*2+1)*4004, buffer+16*4004+(pid*2+1)*4004, 1*1001);

    cast_f32_to_f16(buffer+(pid*2)*4004, buffer+(pid*2)*4004, 1*1001);

    stage_end[5] = read_csr_safe(cycle);

    barrier(ncores);
    return 0;
}