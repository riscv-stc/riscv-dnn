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
    printf("begin %d\n", BATCH);
    uint64_t cycles=0;
    PERF_BEGIN();
    asm("csrwi frm, 0");
    for(int i = 0; i < BATCH; i++) {
        for (int j = 0; j < NLOOPS; j++) {
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
    config_conv(stage0_conv, 230, 230, 3, 64, 0, 0, 0, 0, 7, 7, 2, 2, 1, 1);
    tensor_new_3d(stage0_in, 230, 230, 3, DATASIZE, indata);
    tensor_new_3d(stage0_conv_out, stage0_conv.hout, stage0_conv.wout, stage0_conv.cout, DATASIZE, conv_data);
    tensor_new_4d_with_stride(conv_kernel_f16, stage0_conv.kh, stage0_conv.kw, stage0_conv.cin, stage0_conv.cout, DATASIZE, conv2d_kernel_data, 0);
    conv_im2col_small_cin(&stage0_conv_out, &stage0_in, &conv_kernel_f16, &stage0_conv);
    if (DEBUG_PRINT) {
        printf("stage0_conv_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage0_conv.hout, stage0_conv.wout, stage0_conv.cout);
    }
    

    // stage0: maxpool, SAME
    config_pool(stage0_maxpool, stage0_conv.hout+1, stage0_conv.wout+1, stage0_conv.cout, stage0_conv.cout, 0, 0, 0, 0, 3, 3, 2, 2)
    tensor_new_1d(stage0_alpha, stage0_maxpool.cout, DATASIZE, batch_normalization_new_alpha_data);
    tensor_new_1d(stage0_beta, stage0_maxpool.cout, DATASIZE, batch_normalization_new_beta_data);
    tensor_new_3d(stage0_relu_out, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, DATASIZE, merge_data);

    maxpool_bn_relu(&stage0_relu_out, &stage0_conv_out, &stage0_alpha, &stage0_beta, &stage0_maxpool);
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
    config_conv(stage1_conv1, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage1_conv1_out, stage1_conv1.hout, stage1_conv1.wout, stage1_conv1.cout, DATASIZE, add_data, stage1_conv1.cout * DATASIZE + CACHELINE);
    conv_base(&stage1_conv1_out, &stage0_relu_out, conv2d_1_kernel_data, stage1_conv1, stage1_conv1.cout * DATASIZE + CACHELINE);
    
    config_conv(stage1_conv2, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, 64, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_1_relu_out, stage1_conv2.hout, stage1_conv2.wout, stage1_conv2.cout, DATASIZE, conv_data);
    conv_bn_relu(&stage1_1_relu_out, &stage0_relu_out, conv2d_2_kernel_data, batch_normalization_1_new_alpha_data, batch_normalization_1_new_beta_data, stage1_conv2, 0);

    config_conv(stage1_conv3, stage1_conv2.hout, stage1_conv2.wout, stage1_conv2.cout, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage1_2_relu_out, stage1_conv3.hout, stage1_conv3.wout, stage1_conv3.cout, DATASIZE, merge_data);
    conv_bn_relu(&stage1_2_relu_out, &stage1_1_relu_out, conv2d_3_kernel_data, batch_normalization_2_new_alpha_data, batch_normalization_2_new_beta_data, stage1_conv3, 0);

    config_conv(stage1_conv4, stage1_conv3.hout, stage1_conv3.wout, stage1_conv3.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage1_3_relu_out, stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout, DATASIZE, conv_data, stage1_conv4.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage1_add_out, stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout, DATASIZE, add_data, stage1_conv4.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage1_3_relu_out, &stage1_add_out, &stage1_2_relu_out, &stage1_conv1_out, conv2d_4_kernel_data, batch_normalization_3_new_alpha_data, batch_normalization_3_new_beta_data, stage1_conv4, stage1_conv4.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage1_relu3_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout);
    }

    config_conv(stage1_conv5, stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout, 64, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_4_relu_out, stage1_conv5.hout, stage1_conv5.wout, stage1_conv5.cout, DATASIZE, merge_data);
    conv_bn_relu(&stage1_4_relu_out, &stage1_3_relu_out, conv2d_5_kernel_data, batch_normalization_4_new_alpha_data, batch_normalization_4_new_beta_data, stage1_conv5, 0);
    
    config_conv(stage1_conv6, stage1_conv5.hout, stage1_conv5.wout, stage1_conv5.cout, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage1_5_relu_out, stage1_conv6.hout, stage1_conv6.wout, stage1_conv6.cout, DATASIZE, conv_data);
    conv_bn_relu(&stage1_5_relu_out, &stage1_4_relu_out, conv2d_6_kernel_data, batch_normalization_5_new_alpha_data, batch_normalization_5_new_beta_data, stage1_conv6, 0);

    config_conv(stage1_conv7, stage1_conv6.hout, stage1_conv6.wout, stage1_conv6.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage1_add1_out, stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout, DATASIZE, add_data, stage1_conv7.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage1_6_relu_out, stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout, DATASIZE, merge_data, stage1_conv7.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage1_6_relu_out, &stage1_add1_out, &stage1_5_relu_out, &stage1_add_out, conv2d_7_kernel_data, batch_normalization_6_new_alpha_data, batch_normalization_6_new_beta_data, stage1_conv7, stage1_conv7.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage1_6_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout);
    }

    config_conv(stage1_conv8, stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout, 64, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_7_relu_out, stage1_conv8.hout, stage1_conv8.wout, stage1_conv8.cout, DATASIZE, conv_data);
    conv_bn_relu(&stage1_7_relu_out, &stage1_6_relu_out, conv2d_8_kernel_data, batch_normalization_7_new_alpha_data, batch_normalization_7_new_beta_data, stage1_conv8, 0);

    config_conv(stage1_conv9, stage1_conv8.hout, stage1_conv8.wout, stage1_conv8.cout, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage1_8_relu_out, stage1_conv9.hout, stage1_conv9.wout, stage1_conv9.cout, DATASIZE, merge_data);
    conv_bn_relu(&stage1_8_relu_out, &stage1_7_relu_out, conv2d_9_kernel_data, batch_normalization_8_new_alpha_data, batch_normalization_8_new_beta_data, stage1_conv9, 0);
 
    config_conv(stage1_conv10, stage1_conv9.hout, stage1_conv9.wout, stage1_conv9.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage1_add2_out, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, DATASIZE, add_data, stage1_conv10.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage1_9_relu_out, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, DATASIZE, conv_data, stage1_conv10.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage1_9_relu_out, &stage1_add2_out, &stage1_8_relu_out, &stage1_add1_out, conv2d_10_kernel_data, batch_normalization_9_new_alpha_data, batch_normalization_9_new_beta_data, stage1_conv10, stage1_conv10.cout * DATASIZE + CACHELINE);
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
    config_conv(stage2_conv11, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, 512, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1);
    tensor_new_3d_with_stride(stage2_conv11_out, stage2_conv11.hout, stage2_conv11.wout, stage2_conv11.cout, DATASIZE, add_data, stage2_conv11.cout * DATASIZE + CACHELINE);
    conv_base(&stage2_conv11_out, &stage1_9_relu_out, conv2d_11_kernel_data, stage2_conv11, stage2_conv11.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv12, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_10_relu_out, stage2_conv12.hout, stage2_conv12.wout, stage2_conv12.cout, DATASIZE, merge_data, stage2_conv12.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_10_relu_out, &stage1_9_relu_out, conv2d_12_kernel_data, batch_normalization_10_new_alpha_data, batch_normalization_10_new_beta_data, stage2_conv12, stage2_conv12.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv13, stage2_conv12.hout, stage2_conv12.wout, stage2_conv12.cout, 128, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1);
    tensor_new_3d_with_stride(stage2_11_relu_out, stage2_conv13.hout, stage2_conv13.wout, stage2_conv13.cout, DATASIZE, conv_data, stage2_conv13.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_11_relu_out, &stage2_10_relu_out, conv2d_13_kernel_data, batch_normalization_11_new_alpha_data, batch_normalization_11_new_beta_data, stage2_conv13, stage2_conv13.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv14, stage2_conv13.hout, stage2_conv13.wout, stage2_conv13.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_add3_out, stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout, DATASIZE, add_data, stage2_conv14.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage2_12_relu_out, stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout, DATASIZE, merge_data, stage2_conv14.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage2_12_relu_out, &stage2_add3_out, &stage2_11_relu_out, &stage2_conv11_out, conv2d_14_kernel_data, batch_normalization_12_new_alpha_data, batch_normalization_12_new_beta_data, stage2_conv14, stage2_conv14.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage2_12_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout);
    }

    config_conv(stage2_conv15, stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_13_relu_out, stage2_conv15.hout, stage2_conv15.wout, stage2_conv15.cout, DATASIZE, conv_data, stage2_conv15.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_13_relu_out, &stage2_12_relu_out, conv2d_15_kernel_data, batch_normalization_13_new_alpha_data, batch_normalization_13_new_beta_data, stage2_conv15, stage2_conv15.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv16, stage2_conv15.hout, stage2_conv15.wout, stage2_conv15.cout, 128, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_14_relu_out, stage2_conv16.hout, stage2_conv16.wout, stage2_conv16.cout, DATASIZE, merge_data, stage2_conv16.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_14_relu_out, &stage2_13_relu_out, conv2d_16_kernel_data, batch_normalization_14_new_alpha_data, batch_normalization_14_new_beta_data, stage2_conv16, stage2_conv16.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv17, stage2_conv16.hout, stage2_conv16.wout, stage2_conv16.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_add4_out, stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout, DATASIZE, add_data, stage2_conv17.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage2_15_relu_out, stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout, DATASIZE, conv_data, stage2_conv17.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage2_15_relu_out, &stage2_add4_out, &stage2_14_relu_out, &stage2_add3_out, conv2d_17_kernel_data, batch_normalization_15_new_alpha_data, batch_normalization_15_new_beta_data, stage2_conv17, stage2_conv17.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage2_15_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout);
    }

    config_conv(stage2_conv18, stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_16_relu_out, stage2_conv18.hout, stage2_conv18.wout, stage2_conv18.cout, DATASIZE, merge_data, stage2_conv18.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_16_relu_out, &stage2_15_relu_out, conv2d_18_kernel_data, batch_normalization_16_new_alpha_data, batch_normalization_16_new_beta_data, stage2_conv18, stage2_conv18.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv19, stage2_conv18.hout, stage2_conv18.wout, stage2_conv18.cout, 128, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_17_relu_out, stage2_conv19.hout, stage2_conv19.wout, stage2_conv19.cout, DATASIZE, conv_data, stage2_conv19.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_17_relu_out, &stage2_16_relu_out, conv2d_19_kernel_data, batch_normalization_17_new_alpha_data, batch_normalization_17_new_beta_data, stage2_conv19, stage2_conv19.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv20, stage2_conv19.hout, stage2_conv19.wout, stage2_conv19.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_add5_out, stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout, DATASIZE, add_data, stage2_conv20.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage2_18_relu_out, stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout, DATASIZE, merge_data, stage2_conv20.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage2_18_relu_out, &stage2_add5_out, &stage2_17_relu_out, &stage2_add4_out, conv2d_20_kernel_data, batch_normalization_18_new_alpha_data, batch_normalization_18_new_beta_data, stage2_conv20, stage2_conv20.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage2_18_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout);
    }

    config_conv(stage2_conv21, stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_19_relu_out, stage2_conv21.hout, stage2_conv21.wout, stage2_conv21.cout, DATASIZE, conv_data, stage2_conv21.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_19_relu_out, &stage2_18_relu_out, conv2d_21_kernel_data, batch_normalization_19_new_alpha_data, batch_normalization_19_new_beta_data, stage2_conv21, stage2_conv21.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv22, stage2_conv21.hout, stage2_conv21.wout, stage2_conv21.cout, 128, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_20_relu_out, stage2_conv22.hout, stage2_conv22.wout, stage2_conv22.cout, DATASIZE, merge_data, stage2_conv22.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage2_20_relu_out, &stage2_19_relu_out, conv2d_22_kernel_data, batch_normalization_20_new_alpha_data, batch_normalization_20_new_beta_data, stage2_conv22, stage2_conv22.cout * DATASIZE + CACHELINE);

    config_conv(stage2_conv23, stage2_conv22.hout, stage2_conv22.wout, stage2_conv22.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage2_add6_out, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, DATASIZE, add_data, stage2_conv23.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage2_21_relu_out, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, DATASIZE, conv_data, stage2_conv23.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage2_21_relu_out, &stage2_add6_out, &stage2_20_relu_out, &stage2_add5_out, conv2d_23_kernel_data, batch_normalization_21_new_alpha_data, batch_normalization_21_new_beta_data, stage2_conv23, stage2_conv23.cout * DATASIZE + CACHELINE);
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
    config_conv(stage3_conv24, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, 1024, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1);
    tensor_new_3d_with_stride(stage3_conv24_out, stage3_conv24.hout, stage3_conv24.wout, stage3_conv24.cout, DATASIZE, add_data, stage3_conv24.cout * DATASIZE + CACHELINE);
    conv_base(&stage3_conv24_out, &stage2_21_relu_out, conv2d_24_kernel_data, stage3_conv24, stage3_conv24.cout * DATASIZE + CACHELINE);
  
    config_conv(stage3_conv25, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_22_relu_out, stage3_conv25.hout, stage3_conv25.wout, stage3_conv25.cout, DATASIZE, merge_data, stage3_conv25.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_22_relu_out, &stage2_21_relu_out, conv2d_25_kernel_data, batch_normalization_22_new_alpha_data, batch_normalization_22_new_beta_data, stage3_conv25, stage3_conv25.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv26, stage3_conv25.hout, stage3_conv25.wout, stage3_conv25.cout, 256, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1);
    tensor_new_3d_with_stride(stage3_23_relu_out, stage3_conv26.hout, stage3_conv26.wout, stage3_conv26.cout, DATASIZE, conv_data, stage3_conv26.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_23_relu_out, &stage3_22_relu_out, conv2d_26_kernel_data, batch_normalization_23_new_alpha_data, batch_normalization_23_new_beta_data, stage3_conv26, stage3_conv26.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv27, stage3_conv26.hout, stage3_conv26.wout, stage3_conv26.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_add7_out, stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout, DATASIZE, add_data, stage3_conv27.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage3_24_relu_out, stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout, DATASIZE, merge_data, stage3_conv27.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage3_24_relu_out, &stage3_add7_out, &stage3_23_relu_out, &stage3_conv24_out, conv2d_27_kernel_data, batch_normalization_24_new_alpha_data, batch_normalization_24_new_beta_data, stage3_conv27, stage3_conv27.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage3_24_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout);
    }

    config_conv(stage3_conv28, stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_25_relu_out, stage3_conv28.hout, stage3_conv28.wout, stage3_conv28.cout, DATASIZE, conv_data, stage3_conv28.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_25_relu_out, &stage3_24_relu_out, conv2d_28_kernel_data, batch_normalization_25_new_alpha_data, batch_normalization_25_new_beta_data, stage3_conv28, stage3_conv28.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv29, stage3_conv28.hout, stage3_conv28.wout, stage3_conv28.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_26_relu_out, stage3_conv29.hout, stage3_conv29.wout, stage3_conv29.cout, DATASIZE, merge_data, stage3_conv29.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_26_relu_out, &stage3_25_relu_out, conv2d_29_kernel_data, batch_normalization_26_new_alpha_data, batch_normalization_26_new_beta_data, stage3_conv29, stage3_conv29.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv30, stage3_conv29.hout, stage3_conv29.wout, stage3_conv29.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_add8_out, stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout, DATASIZE, add_data, stage3_conv30.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage3_27_relu_out, stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout, DATASIZE, conv_data, stage3_conv30.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage3_27_relu_out, &stage3_add8_out, &stage3_26_relu_out, &stage3_add7_out, conv2d_30_kernel_data, batch_normalization_27_new_alpha_data, batch_normalization_27_new_beta_data, stage3_conv30, stage3_conv30.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage3_27_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout);
    }

    config_conv(stage3_conv31, stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_28_relu_out, stage3_conv31.hout, stage3_conv31.wout, stage3_conv31.cout, DATASIZE, merge_data, stage3_conv31.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_28_relu_out, &stage3_27_relu_out, conv2d_31_kernel_data, batch_normalization_28_new_alpha_data, batch_normalization_28_new_beta_data, stage3_conv31, stage3_conv31.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv32, stage3_conv31.hout, stage3_conv31.wout, stage3_conv31.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_29_relu_out, stage3_conv32.hout, stage3_conv32.wout, stage3_conv32.cout, DATASIZE, conv_data, stage3_conv32.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_29_relu_out, &stage3_28_relu_out, conv2d_32_kernel_data, batch_normalization_29_new_alpha_data, batch_normalization_29_new_beta_data, stage3_conv32, stage3_conv32.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv33, stage3_conv32.hout, stage3_conv32.wout, stage3_conv32.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_add9_out, stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout, DATASIZE, add_data, stage3_conv33.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage3_30_relu_out, stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout, DATASIZE, merge_data, stage3_conv33.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage3_30_relu_out, &stage3_add9_out, &stage3_29_relu_out, &stage3_add8_out, conv2d_33_kernel_data, batch_normalization_30_new_alpha_data, batch_normalization_30_new_beta_data, stage3_conv33, stage3_conv33.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage3_30_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout);
    }

    config_conv(stage3_conv34, stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_31_relu_out, stage3_conv34.hout, stage3_conv34.wout, stage3_conv34.cout, DATASIZE, conv_data, stage3_conv34.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_31_relu_out, &stage3_30_relu_out, conv2d_34_kernel_data, batch_normalization_31_new_alpha_data, batch_normalization_31_new_beta_data, stage3_conv34, stage3_conv34.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv35, stage3_conv34.hout, stage3_conv34.wout, stage3_conv34.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_32_relu_out, stage3_conv35.hout, stage3_conv35.wout, stage3_conv35.cout, DATASIZE, merge_data, stage3_conv35.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_32_relu_out, &stage3_31_relu_out, conv2d_35_kernel_data, batch_normalization_32_new_alpha_data, batch_normalization_32_new_beta_data, stage3_conv35, stage3_conv35.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv36, stage3_conv35.hout, stage3_conv35.wout, stage3_conv35.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_add10_out, stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout, DATASIZE, add_data, stage3_conv36.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage3_33_relu_out, stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout, DATASIZE, conv_data, stage3_conv36.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage3_33_relu_out, &stage3_add10_out, &stage3_32_relu_out, &stage3_add9_out, conv2d_36_kernel_data, batch_normalization_33_new_alpha_data, batch_normalization_33_new_beta_data, stage3_conv36, stage3_conv36.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage3_33_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout);
    }

    config_conv(stage3_conv37, stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_34_relu_out, stage3_conv37.hout, stage3_conv37.wout, stage3_conv37.cout, DATASIZE, merge_data, stage3_conv37.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_34_relu_out, &stage3_33_relu_out, conv2d_37_kernel_data, batch_normalization_34_new_alpha_data, batch_normalization_34_new_beta_data, stage3_conv37, stage3_conv37.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv38, stage3_conv37.hout, stage3_conv37.wout, stage3_conv37.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_35_relu_out, stage3_conv38.hout, stage3_conv38.wout, stage3_conv38.cout, DATASIZE, conv_data, stage3_conv38.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_35_relu_out, &stage3_34_relu_out, conv2d_38_kernel_data, batch_normalization_35_new_alpha_data, batch_normalization_35_new_beta_data, stage3_conv38, stage3_conv38.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv39, stage3_conv38.hout, stage3_conv38.wout, stage3_conv38.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_add11_out, stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout, DATASIZE, add_data, stage3_conv39.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage3_36_relu_out, stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout, DATASIZE, merge_data, stage3_conv39.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage3_36_relu_out, &stage3_add11_out, &stage3_35_relu_out, &stage3_add10_out, conv2d_39_kernel_data, batch_normalization_36_new_alpha_data, batch_normalization_36_new_beta_data, stage3_conv39, stage3_conv39.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage3_36_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout);
    }

    config_conv(stage3_conv40, stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_37_relu_out, stage3_conv40.hout, stage3_conv40.wout, stage3_conv40.cout, DATASIZE, conv_data, stage3_conv40.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_37_relu_out, &stage3_36_relu_out, conv2d_40_kernel_data, batch_normalization_37_new_alpha_data, batch_normalization_37_new_beta_data, stage3_conv40, stage3_conv40.cout * DATASIZE + CACHELINE);
 
    config_conv(stage3_conv41, stage3_conv40.hout, stage3_conv40.wout, stage3_conv40.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_38_relu_out, stage3_conv41.hout, stage3_conv41.wout, stage3_conv41.cout, DATASIZE, merge_data, stage3_conv41.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage3_38_relu_out, &stage3_37_relu_out, conv2d_41_kernel_data, batch_normalization_38_new_alpha_data, batch_normalization_38_new_beta_data, stage3_conv41, stage3_conv41.cout * DATASIZE + CACHELINE);

    config_conv(stage3_conv42, stage3_conv41.hout, stage3_conv41.wout, stage3_conv41.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage3_add12_out, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, DATASIZE, add_data, stage3_conv42.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage3_39_relu_out, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, DATASIZE, conv_data, stage3_conv42.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage3_39_relu_out, &stage3_add12_out, &stage3_38_relu_out, &stage3_add11_out, conv2d_42_kernel_data, batch_normalization_39_new_alpha_data, batch_normalization_39_new_beta_data, stage3_conv42, stage3_conv42.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage3_39_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout);
    }
    stage_end[3] = read_csr_safe(cycle);
    
/*
 * stage 4
*/

    stage_start[4] = read_csr_safe(cycle);
    config_conv(stage4_conv43, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, 2048, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1);
    tensor_new_3d_with_stride(stage4_conv43_out, stage4_conv43.hout, stage4_conv43.wout, stage4_conv43.cout, DATASIZE, add_data, stage4_conv43.cout * DATASIZE + CACHELINE);
    conv_base(&stage4_conv43_out, &stage3_39_relu_out, conv2d_43_kernel_data, stage4_conv43, stage4_conv43.cout * DATASIZE + CACHELINE);


    

    config_conv(stage4_conv44, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_40_relu_out, stage4_conv44.hout, stage4_conv44.wout, stage4_conv44.cout, DATASIZE, merge_data, stage4_conv44.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage4_40_relu_out, &stage3_39_relu_out, conv2d_44_kernel_data, batch_normalization_40_new_alpha_data, batch_normalization_40_new_beta_data, stage4_conv44, stage4_conv44.cout * DATASIZE + CACHELINE);

    config_conv(stage4_conv45, stage4_conv44.hout, stage4_conv44.wout, stage4_conv44.cout, 512, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1);
    tensor_new_3d_with_stride(stage4_41_relu_out, stage4_conv45.hout, stage4_conv45.wout, stage4_conv45.cout, DATASIZE, conv_data, stage4_conv45.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage4_41_relu_out, &stage4_40_relu_out, conv2d_45_kernel_data, batch_normalization_41_new_alpha_data, batch_normalization_41_new_beta_data, stage4_conv45, stage4_conv45.cout * DATASIZE + CACHELINE);

    config_conv(stage4_conv46, stage4_conv45.hout, stage4_conv45.wout, stage4_conv45.cout, 2048, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_add13_out, stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout, DATASIZE, add_data, stage4_conv46.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage4_42_relu_out, stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout, DATASIZE, merge_data, stage4_conv46.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage4_42_relu_out,& stage4_add13_out, &stage4_41_relu_out, &stage4_conv43_out, conv2d_46_kernel_data, batch_normalization_42_new_alpha_data, batch_normalization_42_new_beta_data, stage4_conv46, stage4_conv46.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage4_42_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout);
    }

    config_conv(stage4_conv47, stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_43_relu_out, stage4_conv47.hout, stage4_conv47.wout, stage4_conv47.cout, DATASIZE, conv_data, stage4_conv47.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage4_43_relu_out, &stage4_42_relu_out, conv2d_47_kernel_data, batch_normalization_43_new_alpha_data, batch_normalization_43_new_beta_data, stage4_conv47, stage4_conv47.cout * DATASIZE + CACHELINE);

    config_conv(stage4_conv48, stage4_conv47.hout, stage4_conv47.wout, stage4_conv47.cout, 512, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_44_relu_out, stage4_conv48.hout, stage4_conv48.wout, stage4_conv48.cout, DATASIZE, merge_data, stage4_conv48.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage4_44_relu_out, &stage4_43_relu_out, conv2d_48_kernel_data, batch_normalization_44_new_alpha_data, batch_normalization_44_new_beta_data, stage4_conv48, stage4_conv48.cout * DATASIZE + CACHELINE);

    config_conv(stage4_conv49, stage4_conv48.hout, stage4_conv48.wout, stage4_conv48.cout, 2048, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_add14_out, stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout, DATASIZE, add_data, stage4_conv49.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage4_45_relu_out, stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout, DATASIZE, conv_data, stage4_conv49.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage4_45_relu_out, &stage4_add14_out, &stage4_44_relu_out, &stage4_add13_out, conv2d_49_kernel_data, batch_normalization_45_new_alpha_data, batch_normalization_45_new_beta_data, stage4_conv49, stage4_conv49.cout * DATASIZE + CACHELINE);
    if (DEBUG_PRINT) {
        printf("stage4_45_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout);
    }

    config_conv(stage4_conv50, stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_46_relu_out, stage4_conv50.hout, stage4_conv50.wout, stage4_conv50.cout, DATASIZE, merge_data, stage4_conv50.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage4_46_relu_out, &stage4_45_relu_out, conv2d_50_kernel_data, batch_normalization_46_new_alpha_data, batch_normalization_46_new_beta_data, stage4_conv50, stage4_conv50.cout * DATASIZE + CACHELINE);

    config_conv(stage4_conv51, stage4_conv50.hout, stage4_conv50.wout, stage4_conv50.cout, 512, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_47_relu_out, stage4_conv51.hout, stage4_conv51.wout, stage4_conv51.cout, DATASIZE, conv_data, stage4_conv51.cout * DATASIZE + CACHELINE);
    conv_bn_relu(&stage4_47_relu_out, &stage4_46_relu_out, conv2d_51_kernel_data, batch_normalization_47_new_alpha_data, batch_normalization_47_new_beta_data, stage4_conv51, stage4_conv51.cout * DATASIZE + CACHELINE);

    config_conv(stage4_conv52, stage4_conv51.hout, stage4_conv51.wout, stage4_conv51.cout, 2048, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d_with_stride(stage4_add15_out, stage4_conv52.hout, stage4_conv52.wout, stage4_conv52.cout, DATASIZE, add_data, stage4_conv52.cout * DATASIZE + CACHELINE);
    tensor_new_3d_with_stride(stage4_48_relu_out, stage4_conv52.hout, stage4_conv52.wout, stage4_conv52.cout, DATASIZE, merge_data, stage4_conv52.cout * DATASIZE + CACHELINE);
    conv_add_bn_relu(&stage4_48_relu_out, &stage4_add15_out, &stage4_47_relu_out, &stage4_add14_out, conv2d_52_kernel_data, batch_normalization_48_new_alpha_data, batch_normalization_48_new_beta_data, stage4_conv52, stage4_conv52.cout * DATASIZE + CACHELINE);
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
    config_pool(stage5_avgpool, stage4_conv52.hout, stage4_conv52.wout, stage4_conv52.cout, stage4_conv52.cout, 0, 0, 0, 0, stage4_conv52.hout, stage4_conv52.wout, 1, 1);
    tensor_new_3d(stage5_avgpool_out, stage5_avgpool.hout, stage5_avgpool.wout, stage5_avgpool.cout, DATASIZE, conv_data);
    avgpool_mean(&stage5_avgpool_out, &stage4_48_relu_out, &stage5_avgpool);
    if (DEBUG_PRINT) {
        printf("stage5_avgpool_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage5_avgpool_out.shape[0], stage5_avgpool_out.shape[1], stage5_avgpool_out.shape[2]);
    }

    tensor_new_2d(stage5_avgpool_1d, 1, stage5_avgpool.hout * stage5_avgpool.wout * stage5_avgpool.cout, DATASIZE, conv_data);
    tensor_new_2d(stage5_dense_kernel_f16, 2048, 1001, DATASIZE, dense_kernel_data);
    tensor_new_2d(stage5_matmul_out, 1, 1001, DATASIZE, add_data);
    matmul(&stage5_matmul_out, &stage5_avgpool_1d, &stage5_dense_kernel_f16);
    if (DEBUG_PRINT) {
        printf("stage5_matmul_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage5_matmul_out.shape[0], stage5_matmul_out.shape[1], stage5_matmul_out.shape[2]);
    }

    tensor_new_2d(stage5_bias_f32, 1, 1001, sizeof(float32_t), dense_bias_data);
    tensor_new_2d(stage5_bias_f16, 1, 1001, DATASIZE, merge_data);
    cast_f32_to_f16(&stage5_bias_f16, &stage5_bias_f32);
    tensor_new_2d(stage5_bias_add_out, 1, 1001, DATASIZE, conv_data);
    add(&stage5_bias_add_out, &stage5_matmul_out, &stage5_bias_f16);
    if (DEBUG_PRINT) {
        printf("stage5_bias_add_out shape: \n\t(hout, wout, cout) = (%d, %d)\n",
                stage5_bias_add_out.shape[0], stage5_bias_add_out.shape[1]);
    }
    

    tensor_new_2d(stage5_bias_add_out_f32, 1, 1001, sizeof(float32_t), merge_data);
    cast_f16_to_f32(&stage5_bias_add_out_f32, &stage5_bias_add_out);
    tensor_new_2d(stage5_softmax_out_f32, 1, 1001, sizeof(float32_t), add_data);
    softmax(&stage5_softmax_out_f32, &stage5_bias_add_out_f32);
    if (DEBUG_PRINT) {
        printf("stage5_softmax_out_f32 shape: \n\t(hout, wout) = (%d, %d)\n",
                stage5_softmax_out_f32.shape[0], stage5_softmax_out_f32.shape[1]);
    }

    tensor_new_2d(stage5_softmax_out_f16, 1, 1001, DATASIZE, softmax_tensor_fp16_data);
    cast_f32_to_f16(&stage5_softmax_out_f16, &stage5_softmax_out_f32);
    stage_end[5] = read_csr_safe(cycle);
    // if (DEBUG_PRINT) {
    //     unsigned short *p = (unsigned short *)softmax_tensor_fp16_data;
    //     printf("result %d ", num+1);
    //     for (int i = 0; i < 1001; i++) {
    //         printf("%d,", *(p+i));
    //     }
    //     printf("\n");
    // }
    return 0;
}


int conv_bn_relu(Tensor *relu_out, Tensor *conv_in, void *pweight, void *palpha, void *pbeta, Config sst, int wstride)
{
#ifndef __RVM__
    tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, DATASIZE, pweight);
    tensor_new_3d(conv_out, sst.hout, sst.wout, sst.cout, DATASIZE, conv_tmp_data);
    memset(conv_pad_tmp_data, 0, (sst.hin+sst.top+sst.bottom)*(sst.win+sst.left+sst.right)*sst.cin*DATASIZE);// conv_pad_tmp_data must be clear
    tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, DATASIZE, conv_pad_tmp_data);
    conv(&conv_out, conv_in, &conv_kernel_f16, &conv_in_pad, &sst);
    tensor_new_1d(alpha, sst.cout, DATASIZE, palpha);
    tensor_new_1d(beta, sst.cout, DATASIZE, pbeta);
    tensor_new_3d(batchnorm_out, sst.hout, sst.wout, sst.cout, DATASIZE, batchnorm_tmp_data);
    batchnorm(&batchnorm_out, &conv_out, &alpha, &beta);
    relu(relu_out, &batchnorm_out, (float16_t)0);
#else
    tensor_new_4d_with_stride(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, DATASIZE, pweight, wstride);
    tensor_new_1d(alpha, sst.cout, DATASIZE, palpha);
    tensor_new_1d(beta, sst.cout, DATASIZE, pbeta);
    conv_bn_relu_rvm(relu_out, conv_in, &conv_kernel_f16, &alpha, &beta, &sst);
#endif
    return 0;
}

int conv_base(Tensor *conv_out, Tensor *conv_in, void *pweight, Config sst, int wstride)
{

#ifndef __RVM__
    tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout,DATASIZE, pweight);
    memset(conv_pad_tmp_data, 0, (sst.hin+sst.top+sst.bottom)*(sst.win+sst.left+sst.right)*sst.cin*DATASIZE);// conv_pad_tmp_data must be clear
    tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, DATASIZE, conv_pad_tmp_data);
    conv(conv_out, conv_in, &conv_kernel_f16, &conv_in_pad, &sst);
#else
    tensor_new_4d_with_stride(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout,DATASIZE, pweight, wstride);
    conv(conv_out, conv_in, &conv_kernel_f16, NULL, &sst);
#endif
    return 0;

}

int conv_add_bn_relu(Tensor *relu_out, Tensor *add_out, Tensor *conv_in, Tensor *add_in, void *pweight, void *palpha, void *pbeta, Config sst, int wstride)
{
#ifndef __RVM__
    tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, DATASIZE, pweight);
    tensor_new_3d(conv_out, sst.hout, sst.wout, sst.cout, DATASIZE, conv_tmp_data);
    memset(conv_pad_tmp_data, 0, (sst.hin+sst.top+sst.bottom)*(sst.win+sst.left+sst.right)*sst.cin*DATASIZE);// conv_pad_tmp_data must be clear
    tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, DATASIZE, conv_pad_tmp_data);
    conv(&conv_out, conv_in, &conv_kernel_f16, &conv_in_pad, &sst);
    add(add_out, &conv_out, add_in);
    tensor_new_1d(alpha, sst.cout, DATASIZE, palpha);
    tensor_new_1d(beta, sst.cout, DATASIZE, pbeta);
    tensor_new_3d(batchnorm_out, sst.hout, sst.wout, sst.cout, DATASIZE, batchnorm_tmp_data);
    batchnorm(&batchnorm_out, add_out, &alpha, &beta);
    relu(relu_out, &batchnorm_out, (float16_t)0);
#else
    tensor_new_4d_with_stride(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, DATASIZE, pweight, wstride);
    tensor_new_1d(alpha, sst.cout, DATASIZE, palpha);
    tensor_new_1d(beta, sst.cout, DATASIZE, pbeta);
    conv_add_bn_relu_rvm(relu_out, add_out, conv_in, &conv_kernel_f16, add_in, &alpha, &beta, &sst);
#endif
    return 0;
}
