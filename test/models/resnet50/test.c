#include <stdio.h>

#include "resnet.h"

int main()
{   
    printf("begin %d\n", N);
    const int num_pictures = N;
    const int picture_size = 224 * 224 * 3 * sizeof(float16_t);
    for(int i = num_pictures; i < num_pictures+1; i++) {
        resnet50_base(imagenet_pic_data_data + i * picture_size);
    }

    return 0;
}

int resnet50_base(void *indata)
{

    // input(1, 224, 224, 3)
/*
 * stage 0 
*/
    asm("csrwi frm, 0"); 
    // stage 0: padding && cast weight && conv
    config_conv(stage0_conv, 224, 224, 3, 64, 3, 3, 3, 3, 7, 7, 2, 2, 1, 1);
    tensor_new_3d(stage0_in, 224, 224, 3, sizeof(float16_t), 0);
    stage0_in.data = indata;
    tensor_new_3d(stage0_conv_out, stage0_conv.hout, stage0_conv.wout, stage0_conv.cout, sizeof(float16_t), conv2d_data);
    conv_base(&stage0_conv_out, &stage0_in, conv2d_kernel_data, stage0_conv);
    if (DEBUG_PRINT) {
        printf("stage0_conv_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage0_conv.hout, stage0_conv.wout, stage0_conv.cout);
    }
    

    // stage0: maxpool, SAME
    config_pool(stage0_maxpool, stage0_conv.hout, stage0_conv.wout, stage0_conv.cout, stage0_conv.cout, 0, 1, 0, 1, 3, 3, 2, 2)
    tensor_new_3d(stage0_maxpool_out, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, sizeof(float16_t), max_pooling2d_data);
    maxpool(&stage0_maxpool_out, &stage0_conv_out, &stage0_maxpool);

    tensor_new_1d(stage0_alpha, stage0_maxpool.cout, sizeof(float16_t), batch_normalization_new_alpha_data);
    tensor_new_1d(stage0_beta, stage0_maxpool.cout, sizeof(float16_t), batch_normalization_new_beta_data);
    tensor_new_3d(stage0_batchnorm_out, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, sizeof(float16_t), batch_normalization_data);
    batchnorm(&stage0_batchnorm_out, &stage0_maxpool_out, &stage0_alpha, &stage0_beta);

    tensor_new_3d(stage0_relu_out, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, sizeof(float16_t), Relu_data);
    relu(&stage0_relu_out, &stage0_batchnorm_out, (float16_t)0);

    if (DEBUG_PRINT) {
        printf("stage0_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout);
    }

/*
 * stage 1 
*/
    // s1u1   
    config_conv(stage1_conv1, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_conv1_out, stage1_conv1.hout, stage1_conv1.wout, stage1_conv1.cout, sizeof(float16_t), conv2d_1_data);
    conv_base(&stage1_conv1_out, &stage0_relu_out, conv2d_1_kernel_data, stage1_conv1);
    
    config_conv(stage1_conv2, stage0_maxpool.hout, stage0_maxpool.wout, stage0_maxpool.cout, 64, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_1_relu_out, stage1_conv2.hout, stage1_conv2.wout, stage1_conv2.cout, sizeof(float16_t), Relu_1_data);
    conv_bn_relu(&stage1_1_relu_out, &stage0_relu_out, conv2d_2_kernel_data, batch_normalization_1_new_alpha_data, batch_normalization_1_new_beta_data, stage1_conv2);

    config_conv(stage1_conv3, stage1_conv2.hout, stage1_conv2.wout, stage1_conv2.cout, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage1_2_relu_out, stage1_conv3.hout, stage1_conv3.wout, stage1_conv3.cout, sizeof(float16_t), Relu_2_data);
    conv_bn_relu(&stage1_2_relu_out, &stage1_1_relu_out, conv2d_3_kernel_data, batch_normalization_2_new_alpha_data, batch_normalization_2_new_beta_data, stage1_conv3);

    config_conv(stage1_conv4, stage1_conv3.hout, stage1_conv3.wout, stage1_conv3.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_3_relu_out, stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout, sizeof(float16_t), Relu_3_data);
    tensor_new_3d(stage1_add_out, stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout, sizeof(float16_t), add_data);
    conv_add_bn_relu(&stage1_3_relu_out, &stage1_add_out, &stage1_2_relu_out, &stage1_conv1_out, conv2d_4_kernel_data, batch_normalization_3_new_alpha_data, batch_normalization_3_new_beta_data, stage1_conv4);
    if (DEBUG_PRINT) {
        printf("stage1_relu3_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout);
    }

    config_conv(stage1_conv5, stage1_conv4.hout, stage1_conv4.wout, stage1_conv4.cout, 64, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_4_relu_out, stage1_conv5.hout, stage1_conv5.wout, stage1_conv5.cout, sizeof(float16_t), Relu_4_data);
    conv_bn_relu(&stage1_4_relu_out, &stage1_3_relu_out, conv2d_5_kernel_data, batch_normalization_4_new_alpha_data, batch_normalization_4_new_beta_data, stage1_conv5);
    
    config_conv(stage1_conv6, stage1_conv5.hout, stage1_conv5.wout, stage1_conv5.cout, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage1_5_relu_out, stage1_conv6.hout, stage1_conv6.wout, stage1_conv6.cout, sizeof(float16_t), Relu_5_data);
    conv_bn_relu(&stage1_5_relu_out, &stage1_4_relu_out, conv2d_6_kernel_data, batch_normalization_5_new_alpha_data, batch_normalization_5_new_beta_data, stage1_conv6);

    config_conv(stage1_conv7, stage1_conv6.hout, stage1_conv6.wout, stage1_conv6.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_add1_out, stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout, sizeof(float16_t), add_1_data);
    tensor_new_3d(stage1_6_relu_out, stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout, sizeof(float16_t), Relu_6_data);
    conv_add_bn_relu(&stage1_6_relu_out, &stage1_add1_out, &stage1_5_relu_out, &stage1_add_out, conv2d_7_kernel_data, batch_normalization_6_new_alpha_data, batch_normalization_6_new_beta_data, stage1_conv7);
    if (DEBUG_PRINT) {
        printf("stage1_6_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout);
    }

    config_conv(stage1_conv8, stage1_conv7.hout, stage1_conv7.wout, stage1_conv7.cout, 64, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_7_relu_out, stage1_conv8.hout, stage1_conv8.wout, stage1_conv8.cout, sizeof(float16_t), Relu_7_data);
    conv_bn_relu(&stage1_7_relu_out, &stage1_6_relu_out, conv2d_8_kernel_data, batch_normalization_7_new_alpha_data, batch_normalization_7_new_beta_data, stage1_conv8);

    config_conv(stage1_conv9, stage1_conv8.hout, stage1_conv8.wout, stage1_conv8.cout, 64, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage1_8_relu_out, stage1_conv9.hout, stage1_conv9.wout, stage1_conv9.cout, sizeof(float16_t), Relu_8_data);
    conv_bn_relu(&stage1_8_relu_out, &stage1_7_relu_out, conv2d_9_kernel_data, batch_normalization_8_new_alpha_data, batch_normalization_8_new_beta_data, stage1_conv9);
 
    config_conv(stage1_conv10, stage1_conv9.hout, stage1_conv9.wout, stage1_conv9.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage1_add2_out, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, sizeof(float16_t), add_2_data);
    tensor_new_3d(stage1_9_relu_out, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, sizeof(float16_t), Relu_9_data);
    conv_add_bn_relu(&stage1_9_relu_out, &stage1_add2_out, &stage1_8_relu_out, &stage1_add1_out, conv2d_10_kernel_data, batch_normalization_9_new_alpha_data, batch_normalization_9_new_beta_data, stage1_conv10);
    if (DEBUG_PRINT) {
        printf("stage1_9_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout);
    }

/*
 * stage 2
*/
    // s2u1
    config_conv(stage2_conv11, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, 512, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1);
    tensor_new_3d(stage2_conv11_out, stage2_conv11.hout, stage2_conv11.wout, stage2_conv11.cout, sizeof(float16_t), conv2d_11_data);
    conv_base(&stage2_conv11_out, &stage1_9_relu_out, conv2d_11_kernel_data, stage2_conv11);

    config_conv(stage2_conv12, stage1_conv10.hout, stage1_conv10.wout, stage1_conv10.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_10_relu_out, stage2_conv12.hout, stage2_conv12.wout, stage2_conv12.cout, sizeof(float16_t), Relu_10_data);
    conv_bn_relu(&stage2_10_relu_out, &stage1_9_relu_out, conv2d_12_kernel_data, batch_normalization_10_new_alpha_data, batch_normalization_10_new_beta_data, stage2_conv12);

    config_conv(stage2_conv13, stage2_conv12.hout, stage2_conv12.wout, stage2_conv12.cout, 128, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1);
    tensor_new_3d(stage2_11_relu_out, stage2_conv13.hout, stage2_conv13.wout, stage2_conv13.cout, sizeof(float16_t), Relu_11_data);
    conv_bn_relu(&stage2_11_relu_out, &stage2_10_relu_out, conv2d_13_kernel_data, batch_normalization_11_new_alpha_data, batch_normalization_11_new_beta_data, stage2_conv13);

    config_conv(stage2_conv14, stage2_conv13.hout, stage2_conv13.wout, stage2_conv13.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_add3_out, stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout, sizeof(float16_t), add_3_data);
    tensor_new_3d(stage2_12_relu_out, stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout, sizeof(float16_t), Relu_12_data);
    conv_add_bn_relu(&stage2_12_relu_out, &stage2_add3_out, &stage2_11_relu_out, &stage2_conv11_out, conv2d_14_kernel_data, batch_normalization_12_new_alpha_data, batch_normalization_12_new_beta_data, stage2_conv14);
    if (DEBUG_PRINT) {
        printf("stage2_12_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv13.hout, stage2_conv13.wout, stage2_conv13.cout);
    }

    config_conv(stage2_conv15, stage2_conv14.hout, stage2_conv14.wout, stage2_conv14.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_13_relu_out, stage2_conv15.hout, stage2_conv15.wout, stage2_conv15.cout, sizeof(float16_t), Relu_13_data);
    conv_bn_relu(&stage2_13_relu_out, &stage2_12_relu_out, conv2d_15_kernel_data, batch_normalization_13_new_alpha_data, batch_normalization_13_new_beta_data, stage2_conv15);

    config_conv(stage2_conv16, stage2_conv15.hout, stage2_conv15.wout, stage2_conv15.cout, 128, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage2_14_relu_out, stage2_conv16.hout, stage2_conv16.wout, stage2_conv16.cout, sizeof(float16_t), Relu_14_data);
    conv_bn_relu(&stage2_14_relu_out, &stage2_13_relu_out, conv2d_16_kernel_data, batch_normalization_14_new_alpha_data, batch_normalization_14_new_beta_data, stage2_conv16);

    config_conv(stage2_conv17, stage2_conv16.hout, stage2_conv16.wout, stage2_conv16.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_add4_out, stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout, sizeof(float16_t), add_4_data);
    tensor_new_3d(stage2_15_relu_out, stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout, sizeof(float16_t), Relu_15_data);
    conv_add_bn_relu(&stage2_15_relu_out, &stage2_add4_out, &stage2_14_relu_out, &stage2_add3_out, conv2d_17_kernel_data, batch_normalization_15_new_alpha_data, batch_normalization_15_new_beta_data, stage2_conv17);
    if (DEBUG_PRINT) {
        printf("stage2_15_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv16.hout, stage2_conv16.wout, stage2_conv16.cout);
    }

    config_conv(stage2_conv18, stage2_conv17.hout, stage2_conv17.wout, stage2_conv17.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_16_relu_out, stage2_conv18.hout, stage2_conv18.wout, stage2_conv18.cout, sizeof(float16_t), Relu_16_data);
    conv_bn_relu(&stage2_16_relu_out, &stage2_15_relu_out, conv2d_18_kernel_data, batch_normalization_16_new_alpha_data, batch_normalization_16_new_beta_data, stage2_conv18);

    config_conv(stage2_conv19, stage2_conv18.hout, stage2_conv18.wout, stage2_conv18.cout, 128, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage2_17_relu_out, stage2_conv19.hout, stage2_conv19.wout, stage2_conv19.cout, sizeof(float16_t), Relu_17_data);
    conv_bn_relu(&stage2_17_relu_out, &stage2_16_relu_out, conv2d_19_kernel_data, batch_normalization_17_new_alpha_data, batch_normalization_17_new_beta_data, stage2_conv19);

    config_conv(stage2_conv20, stage2_conv19.hout, stage2_conv19.wout, stage2_conv19.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_add5_out, stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout, sizeof(float16_t), add_5_data);
    tensor_new_3d(stage2_18_relu_out, stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout, sizeof(float16_t), Relu_18_data);
    conv_add_bn_relu(&stage2_18_relu_out, &stage2_add5_out, &stage2_17_relu_out, &stage2_add4_out, conv2d_20_kernel_data, batch_normalization_18_new_alpha_data, batch_normalization_18_new_beta_data, stage2_conv20);
    if (DEBUG_PRINT) {
        printf("stage2_18_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv19.hout, stage2_conv19.wout, stage2_conv19.cout);
    }

    config_conv(stage2_conv21, stage2_conv20.hout, stage2_conv20.wout, stage2_conv20.cout, 128, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_19_relu_out, stage2_conv21.hout, stage2_conv21.wout, stage2_conv21.cout, sizeof(float16_t), Relu_19_data);
    conv_bn_relu(&stage2_19_relu_out, &stage2_18_relu_out, conv2d_21_kernel_data, batch_normalization_19_new_alpha_data, batch_normalization_19_new_beta_data, stage2_conv21);

    config_conv(stage2_conv22, stage2_conv21.hout, stage2_conv21.wout, stage2_conv21.cout, 128, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage2_20_relu_out, stage2_conv22.hout, stage2_conv22.wout, stage2_conv22.cout, sizeof(float16_t), Relu_20_data);
    conv_bn_relu(&stage2_20_relu_out, &stage2_19_relu_out, conv2d_22_kernel_data, batch_normalization_20_new_alpha_data, batch_normalization_20_new_beta_data, stage2_conv22);

    config_conv(stage2_conv23, stage2_conv22.hout, stage2_conv22.wout, stage2_conv22.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage2_add6_out, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, sizeof(float16_t), add_6_data);
    tensor_new_3d(stage2_21_relu_out, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, sizeof(float16_t), Relu_21_data);
    conv_add_bn_relu(&stage2_21_relu_out, &stage2_add6_out, &stage2_20_relu_out, &stage2_add5_out, conv2d_23_kernel_data, batch_normalization_21_new_alpha_data, batch_normalization_21_new_beta_data, stage2_conv23);
    if (DEBUG_PRINT) {
        printf("stage2_21_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage2_conv22.hout, stage2_conv22.wout, stage2_conv22.cout);
    }
/*
 * stage 3
*/
    // s3u1
    config_conv(stage3_conv24, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, 1024, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1);
    tensor_new_3d(stage3_conv24_out, stage3_conv24.hout, stage3_conv24.wout, stage3_conv24.cout, sizeof(float16_t), conv2d_24_data);
    conv_base(&stage3_conv24_out, &stage2_21_relu_out, conv2d_24_kernel_data, stage3_conv24);
  
    config_conv(stage3_conv25, stage2_conv23.hout, stage2_conv23.wout, stage2_conv23.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_22_relu_out, stage3_conv25.hout, stage3_conv25.wout, stage3_conv25.cout, sizeof(float16_t), Relu_22_data);
    conv_bn_relu(&stage3_22_relu_out, &stage2_21_relu_out, conv2d_25_kernel_data, batch_normalization_22_new_alpha_data, batch_normalization_22_new_beta_data, stage3_conv25);

    config_conv(stage3_conv26, stage3_conv25.hout, stage3_conv25.wout, stage3_conv25.cout, 256, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1);
    tensor_new_3d(stage3_23_relu_out, stage3_conv26.hout, stage3_conv26.wout, stage3_conv26.cout, sizeof(float16_t), Relu_23_data);
    conv_bn_relu(&stage3_23_relu_out, &stage3_22_relu_out, conv2d_26_kernel_data, batch_normalization_23_new_alpha_data, batch_normalization_23_new_beta_data, stage3_conv26);

    config_conv(stage3_conv27, stage3_conv26.hout, stage3_conv26.wout, stage3_conv26.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_add7_out, stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout, sizeof(float16_t), add_7_data);
    tensor_new_3d(stage3_24_relu_out, stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout, sizeof(float16_t), Relu_24_data);
    conv_add_bn_relu(&stage3_24_relu_out, &stage3_add7_out, &stage3_23_relu_out, &stage3_conv24_out, conv2d_27_kernel_data, batch_normalization_24_new_alpha_data, batch_normalization_24_new_beta_data, stage3_conv27);
    if (DEBUG_PRINT) {
        printf("stage3_24_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv26.hout, stage3_conv26.wout, stage3_conv26.cout);
    }

    config_conv(stage3_conv28, stage3_conv27.hout, stage3_conv27.wout, stage3_conv27.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_25_relu_out, stage3_conv28.hout, stage3_conv28.wout, stage3_conv28.cout, sizeof(float16_t), Relu_25_data);
    conv_bn_relu(&stage3_25_relu_out, &stage3_24_relu_out, conv2d_28_kernel_data, batch_normalization_25_new_alpha_data, batch_normalization_25_new_beta_data, stage3_conv28);

    config_conv(stage3_conv29, stage3_conv28.hout, stage3_conv28.wout, stage3_conv28.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage3_26_relu_out, stage3_conv29.hout, stage3_conv29.wout, stage3_conv29.cout, sizeof(float16_t), Relu_26_data);
    conv_bn_relu(&stage3_26_relu_out, &stage3_25_relu_out, conv2d_29_kernel_data, batch_normalization_26_new_alpha_data, batch_normalization_26_new_beta_data, stage3_conv29);

    config_conv(stage3_conv30, stage3_conv29.hout, stage3_conv29.wout, stage3_conv29.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_add8_out, stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout, sizeof(float16_t), add_8_data);
    tensor_new_3d(stage3_27_relu_out, stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout, sizeof(float16_t), Relu_27_data);
    conv_add_bn_relu(&stage3_27_relu_out, &stage3_add8_out, &stage3_26_relu_out, &stage3_add7_out, conv2d_30_kernel_data, batch_normalization_27_new_alpha_data, batch_normalization_27_new_beta_data, stage3_conv30);
    if (DEBUG_PRINT) {
        printf("stage3_27_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv29.hout, stage3_conv29.wout, stage3_conv29.cout);
    }

    config_conv(stage3_conv31, stage3_conv30.hout, stage3_conv30.wout, stage3_conv30.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_28_relu_out, stage3_conv31.hout, stage3_conv31.wout, stage3_conv31.cout, sizeof(float16_t), Relu_28_data);
    conv_bn_relu(&stage3_28_relu_out, &stage3_27_relu_out, conv2d_31_kernel_data, batch_normalization_28_new_alpha_data, batch_normalization_28_new_beta_data, stage3_conv31);

    config_conv(stage3_conv32, stage3_conv31.hout, stage3_conv31.wout, stage3_conv31.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage3_29_relu_out, stage3_conv32.hout, stage3_conv32.wout, stage3_conv32.cout, sizeof(float16_t), Relu_29_data);
    conv_bn_relu(&stage3_29_relu_out, &stage3_28_relu_out, conv2d_32_kernel_data, batch_normalization_29_new_alpha_data, batch_normalization_29_new_beta_data, stage3_conv32);

    config_conv(stage3_conv33, stage3_conv32.hout, stage3_conv32.wout, stage3_conv32.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_add9_out, stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout, sizeof(float16_t), add_9_data);
    tensor_new_3d(stage3_30_relu_out, stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout, sizeof(float16_t), Relu_30_data);
    conv_add_bn_relu(&stage3_30_relu_out, &stage3_add9_out, &stage3_29_relu_out, &stage3_add8_out, conv2d_33_kernel_data, batch_normalization_30_new_alpha_data, batch_normalization_30_new_beta_data, stage3_conv33);
    if (DEBUG_PRINT) {
        printf("stage3_30_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv32.hout, stage3_conv32.wout, stage3_conv32.cout);
    }

    config_conv(stage3_conv34, stage3_conv33.hout, stage3_conv33.wout, stage3_conv33.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_31_relu_out, stage3_conv34.hout, stage3_conv34.wout, stage3_conv34.cout, sizeof(float16_t), Relu_31_data);
    conv_bn_relu(&stage3_31_relu_out, &stage3_30_relu_out, conv2d_34_kernel_data, batch_normalization_31_new_alpha_data, batch_normalization_31_new_beta_data, stage3_conv34);

    config_conv(stage3_conv35, stage3_conv34.hout, stage3_conv34.wout, stage3_conv34.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage3_32_relu_out, stage3_conv35.hout, stage3_conv35.wout, stage3_conv35.cout, sizeof(float16_t), Relu_32_data);
    conv_bn_relu(&stage3_32_relu_out, &stage3_31_relu_out, conv2d_35_kernel_data, batch_normalization_32_new_alpha_data, batch_normalization_32_new_beta_data, stage3_conv35);

    config_conv(stage3_conv36, stage3_conv35.hout, stage3_conv35.wout, stage3_conv35.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_add10_out, stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout, sizeof(float16_t), add_10_data);
    tensor_new_3d(stage3_33_relu_out, stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout, sizeof(float16_t), Relu_33_data);
    conv_add_bn_relu(&stage3_33_relu_out, &stage3_add10_out, &stage3_32_relu_out, &stage3_add9_out, conv2d_36_kernel_data, batch_normalization_33_new_alpha_data, batch_normalization_33_new_beta_data, stage3_conv36);
    if (DEBUG_PRINT) {
        printf("stage3_33_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv35.hout, stage3_conv35.wout, stage3_conv35.cout);
    }

    config_conv(stage3_conv37, stage3_conv36.hout, stage3_conv36.wout, stage3_conv36.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_34_relu_out, stage3_conv37.hout, stage3_conv37.wout, stage3_conv37.cout, sizeof(float16_t), Relu_34_data);
    conv_bn_relu(&stage3_34_relu_out, &stage3_33_relu_out, conv2d_37_kernel_data, batch_normalization_34_new_alpha_data, batch_normalization_34_new_beta_data, stage3_conv37);

    config_conv(stage3_conv38, stage3_conv37.hout, stage3_conv37.wout, stage3_conv37.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage3_35_relu_out, stage3_conv38.hout, stage3_conv38.wout, stage3_conv38.cout, sizeof(float16_t), Relu_35_data);
    conv_bn_relu(&stage3_35_relu_out, &stage3_34_relu_out, conv2d_38_kernel_data, batch_normalization_35_new_alpha_data, batch_normalization_35_new_beta_data, stage3_conv38);

    config_conv(stage3_conv39, stage3_conv38.hout, stage3_conv38.wout, stage3_conv38.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_add11_out, stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout, sizeof(float16_t), add_11_data);
    tensor_new_3d(stage3_36_relu_out, stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout, sizeof(float16_t), Relu_36_data);
    conv_add_bn_relu(&stage3_36_relu_out, &stage3_add11_out, &stage3_35_relu_out, &stage3_add10_out, conv2d_39_kernel_data, batch_normalization_36_new_alpha_data, batch_normalization_36_new_beta_data, stage3_conv39);
    if (DEBUG_PRINT) {
        printf("stage3_36_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv38.hout, stage3_conv38.wout, stage3_conv38.cout);
    }

    config_conv(stage3_conv40, stage3_conv39.hout, stage3_conv39.wout, stage3_conv39.cout, 256, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_37_relu_out, stage3_conv40.hout, stage3_conv40.wout, stage3_conv40.cout, sizeof(float16_t), Relu_37_data);
    conv_bn_relu(&stage3_37_relu_out, &stage3_36_relu_out, conv2d_40_kernel_data, batch_normalization_37_new_alpha_data, batch_normalization_37_new_beta_data, stage3_conv40);
 
    config_conv(stage3_conv41, stage3_conv40.hout, stage3_conv40.wout, stage3_conv40.cout, 256, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage3_38_relu_out, stage3_conv41.hout, stage3_conv41.wout, stage3_conv41.cout, sizeof(float16_t), Relu_38_data);
    conv_bn_relu(&stage3_38_relu_out, &stage3_37_relu_out, conv2d_41_kernel_data, batch_normalization_38_new_alpha_data, batch_normalization_38_new_beta_data, stage3_conv41);

    config_conv(stage3_conv42, stage3_conv41.hout, stage3_conv41.wout, stage3_conv41.cout, 1024, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage3_add12_out, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, sizeof(float16_t), add_12_data);
    tensor_new_3d(stage3_39_relu_out, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, sizeof(float16_t), Relu_39_data);
    conv_add_bn_relu(&stage3_39_relu_out, &stage3_add12_out, &stage3_38_relu_out, &stage3_add11_out, conv2d_42_kernel_data, batch_normalization_39_new_alpha_data, batch_normalization_39_new_beta_data, stage3_conv42);
    if (DEBUG_PRINT) {
        printf("stage3_39_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage3_conv41.hout, stage3_conv41.wout, stage3_conv41.cout);
    }

/*
 * stage 4
*/
    config_conv(stage4_conv43, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, 2048, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1);
    tensor_new_3d(stage4_conv43_out, stage4_conv43.hout, stage4_conv43.wout, stage4_conv43.cout, sizeof(float16_t), conv2d_43_data);
    conv_base(&stage4_conv43_out, &stage3_39_relu_out, conv2d_43_kernel_data, stage4_conv43);

    config_conv(stage4_conv44, stage3_conv42.hout, stage3_conv42.wout, stage3_conv42.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage4_40_relu_out, stage4_conv44.hout, stage4_conv44.wout, stage4_conv44.cout, sizeof(float16_t), Relu_40_data);
    conv_bn_relu(&stage4_40_relu_out, &stage3_39_relu_out, conv2d_44_kernel_data, batch_normalization_40_new_alpha_data, batch_normalization_40_new_beta_data, stage4_conv44);

    config_conv(stage4_conv45, stage4_conv44.hout, stage4_conv44.wout, stage4_conv44.cout, 512, 1, 1, 1, 1, 3, 3, 2, 2, 1, 1);
    tensor_new_3d(stage4_41_relu_out, stage4_conv45.hout, stage4_conv45.wout, stage4_conv45.cout, sizeof(float16_t), Relu_41_data);
    conv_bn_relu(&stage4_41_relu_out, &stage4_40_relu_out, conv2d_45_kernel_data, batch_normalization_41_new_alpha_data, batch_normalization_41_new_beta_data, stage4_conv45);

    config_conv(stage4_conv46, stage4_conv45.hout, stage4_conv45.wout, stage4_conv45.cout, 2048, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage4_add13_out, stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout, sizeof(float16_t), add_13_data);
    tensor_new_3d(stage4_42_relu_out, stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout, sizeof(float16_t), Relu_42_data);
    conv_add_bn_relu(&stage4_42_relu_out,& stage4_add13_out, &stage4_41_relu_out, &stage4_conv43_out, conv2d_46_kernel_data, batch_normalization_42_new_alpha_data, batch_normalization_42_new_beta_data, stage4_conv46);
    if (DEBUG_PRINT) {
        printf("stage4_42_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv45.hout, stage4_conv45.wout, stage4_conv45.cout);
    }

    config_conv(stage4_conv47, stage4_conv46.hout, stage4_conv46.wout, stage4_conv46.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage4_43_relu_out, stage4_conv47.hout, stage4_conv47.wout, stage4_conv47.cout, sizeof(float16_t), Relu_43_data);
    conv_bn_relu(&stage4_43_relu_out, &stage4_42_relu_out, conv2d_47_kernel_data, batch_normalization_43_new_alpha_data, batch_normalization_43_new_beta_data, stage4_conv47);

    config_conv(stage4_conv48, stage4_conv47.hout, stage4_conv47.wout, stage4_conv47.cout, 512, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage4_44_relu_out, stage4_conv48.hout, stage4_conv48.wout, stage4_conv48.cout, sizeof(float16_t), Relu_44_data);
    conv_bn_relu(&stage4_44_relu_out, &stage4_43_relu_out, conv2d_48_kernel_data, batch_normalization_44_new_alpha_data, batch_normalization_44_new_beta_data, stage4_conv48);

    config_conv(stage4_conv49, stage4_conv48.hout, stage4_conv48.wout, stage4_conv48.cout, 2048, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage4_add14_out, stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout, sizeof(float16_t), add_14_data);
    tensor_new_3d(stage4_45_relu_out, stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout, sizeof(float16_t), Relu_45_data);
    conv_add_bn_relu(&stage4_45_relu_out, &stage4_add14_out, &stage4_44_relu_out, &stage4_add13_out, conv2d_49_kernel_data, batch_normalization_45_new_alpha_data, batch_normalization_45_new_beta_data, stage4_conv49);
    if (DEBUG_PRINT) {
        printf("stage4_45_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv48.hout, stage4_conv48.wout, stage4_conv48.cout);
    }

    config_conv(stage4_conv50, stage4_conv49.hout, stage4_conv49.wout, stage4_conv49.cout, 512, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage4_46_relu_out, stage4_conv50.hout, stage4_conv50.wout, stage4_conv50.cout, sizeof(float16_t), Relu_46_data);
    conv_bn_relu(&stage4_46_relu_out, &stage4_45_relu_out, conv2d_50_kernel_data, batch_normalization_46_new_alpha_data, batch_normalization_46_new_beta_data, stage4_conv50);

    config_conv(stage4_conv51, stage4_conv50.hout, stage4_conv50.wout, stage4_conv50.cout, 512, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
    tensor_new_3d(stage4_47_relu_out, stage4_conv51.hout, stage4_conv51.wout, stage4_conv51.cout, sizeof(float16_t), Relu_47_data);
    conv_bn_relu(&stage4_47_relu_out, &stage4_46_relu_out, conv2d_51_kernel_data, batch_normalization_47_new_alpha_data, batch_normalization_47_new_beta_data, stage4_conv51);

    config_conv(stage4_conv52, stage4_conv51.hout, stage4_conv51.wout, stage4_conv51.cout, 2048, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    tensor_new_3d(stage4_add15_out, stage4_conv52.hout, stage4_conv52.wout, stage4_conv52.cout, sizeof(float16_t), add_15_data);
    tensor_new_3d(stage4_48_relu_out, stage4_conv52.hout, stage4_conv52.wout, stage4_conv52.cout, sizeof(float16_t), Relu_48_data);
    conv_add_bn_relu(&stage4_48_relu_out, &stage4_add15_out, &stage4_47_relu_out, &stage4_add14_out, conv2d_52_kernel_data, batch_normalization_48_new_alpha_data, batch_normalization_48_new_beta_data, stage4_conv52);
    if (DEBUG_PRINT) {
        printf("stage4_48_relu_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage4_conv51.hout, stage4_conv51.wout, stage4_conv51.cout);
    }

/**
 * end
 * 
 */
    config_pool(stage5_avgpool, stage4_conv52.hout, stage4_conv52.wout, stage4_conv52.cout, stage4_conv52.cout, 0, 0, 0, 0, stage4_conv52.hout, stage4_conv52.wout, 1, 1);
    tensor_new_3d(stage5_avgpool_out, stage5_avgpool.hout, stage5_avgpool.wout, stage5_avgpool.cout, sizeof(float16_t), Mean_data);
    avgpool(&stage5_avgpool_out, &stage4_48_relu_out, &stage5_avgpool);
    if (DEBUG_PRINT) {
        printf("stage5_avgpool_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage5_avgpool_out.h, stage5_avgpool_out.w, stage5_avgpool_out.cin);
    }

    tensor_new_2d(stage5_avgpool_1d, 1, stage5_avgpool.hout * stage5_avgpool.wout * stage5_avgpool.cout, sizeof(float16_t), Mean_data);
    tensor_new_2d(stage5_dense_kernel_f16, 2048, 1001, sizeof(float16_t), dense_kernel_data);
    tensor_new_2d(stage5_matmul_out, 1, 1001, sizeof(float16_t), MatMul_data);
    matmul(&stage5_matmul_out, &stage5_avgpool_1d, &stage5_dense_kernel_f16);
    if (DEBUG_PRINT) {
        printf("stage5_matmul_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage5_matmul_out.h, stage5_matmul_out.w, stage5_matmul_out.cin);
    }

    tensor_new_2d(stage5_bias_f32, 1, 1001, sizeof(float32_t), dense_bias_data);
    tensor_new_2d(stage5_bias_f16, 1, 1001, sizeof(float16_t), bias_tmp_data);
    cast_f32_to_f16(&stage5_bias_f16, &stage5_bias_f32);
    tensor_new_2d(stage5_bias_add_out, 1, 1001, sizeof(float16_t), bias_add_fp16_tmp_data);
    add(&stage5_bias_add_out, &stage5_matmul_out, &stage5_bias_f16);
    if (DEBUG_PRINT) {
        printf("stage5_bias_add_out shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage5_bias_add_out.h, stage5_bias_add_out.w, stage5_bias_add_out.cin);
    }

    tensor_new_2d(stage5_bias_add_out_f32, 1, 1001, sizeof(float32_t), bias_add_fp32_tmp_data);
    cast_f16_to_f32(&stage5_bias_add_out_f32, &stage5_bias_add_out);

    tensor_new_2d(stage5_softmax_out_f32, 1, 1001, sizeof(float32_t), softmax_tensor_fp32_data);
    softmax(&stage5_softmax_out_f32, &stage5_bias_add_out_f32);
    if (DEBUG_PRINT) {
        printf("stage5_softmax_out_f32 shape: \n\t(hout, wout, cout) = (%d, %d, %d)\n",
                stage5_softmax_out_f32.h, stage5_softmax_out_f32.w, stage5_softmax_out_f32.cin);
    }
    
    tensor_new_2d(stage5_softmax_out_f16, 1, 1001, sizeof(float16_t), softmax_tensor_fp16_data);
    cast_f32_to_f16(&stage5_softmax_out_f16, &stage5_softmax_out_f32);

    unsigned short *p = (unsigned short *)softmax_tensor_fp16_data;
    printf("result %d ", N);
    for (int i = 0; i < 1001; i++) {
        printf("%d,", *(p+i));
    }
    printf("\n");
    

    printf("End\n");
    return 0;
}


int conv_bn_relu(Tensor *relu_out, Tensor *conv_in, void *pweight, void *palpha, void *pbeta, Config sst)
{
    tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, sizeof(float16_t), pweight);
    tensor_new_3d(conv_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), conv_tmp_data);
#ifndef __RVM__
    memset(conv_pad_tmp_data, 0, (sst.hin+sst.top+sst.bottom)*(sst.win+sst.left+sst.right)*sst.cin*sizeof(float16_t));// conv_pad_tmp_data must be clear
    tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, sizeof(float16_t), conv_pad_tmp_data);
#else
    tensor_new_2d(conv_in_pad, sst.hout*sst.wout, sst.kh*sst.kw*sst.cin, sizeof(float16_t), conv_pad_tmp_data);
#endif
    conv(&conv_out, conv_in, &conv_kernel_f16, &conv_in_pad, &sst);

    tensor_new_1d(alpha, sst.cout, sizeof(float16_t), palpha);
    tensor_new_1d(beta, sst.cout, sizeof(float16_t), pbeta);
    tensor_new_3d(batchnorm_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), batchnorm_tmp_data);
    batchnorm(&batchnorm_out, &conv_out, &alpha, &beta);

    relu(relu_out, &batchnorm_out, (float16_t)0);
    return 0;
}

int conv_base(Tensor *conv_out, Tensor *conv_in, void *pweight, Config sst)
{

    tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout,sizeof(float16_t), pweight);
#ifndef __RVM__
    memset(conv_pad_tmp_data, 0, (sst.hin+sst.top+sst.bottom)*(sst.win+sst.left+sst.right)*sst.cin*sizeof(float16_t));// conv_pad_tmp_data must be clear
    tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, sizeof(float16_t), conv_pad_tmp_data);
#else
    tensor_new_2d(conv_in_pad, sst.hout*sst.wout, sst.kh*sst.kw*sst.cin, sizeof(float16_t), conv_pad_tmp_data);
#endif
    conv(conv_out, conv_in, &conv_kernel_f16, &conv_in_pad, &sst);

    return 0;

}

int conv_add_bn_relu(Tensor *relu_out, Tensor *add_out, Tensor *conv_in, Tensor *add_in, void *pweight, void *palpha, void *pbeta, Config sst)
{
    tensor_new_4d(conv_kernel_f16, sst.kh, sst.kw, sst.cin, sst.cout, sizeof(float16_t), pweight);
    
    tensor_new_3d(conv_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), conv_tmp_data);
#ifndef __RVM__
    memset(conv_pad_tmp_data, 0, (sst.hin+sst.top+sst.bottom)*(sst.win+sst.left+sst.right)*sst.cin*sizeof(float16_t));// conv_pad_tmp_data must be clear
    tensor_new_3d(conv_in_pad, sst.hin+sst.top+sst.bottom, sst.win+sst.left+sst.right, sst.cin, sizeof(float16_t), conv_pad_tmp_data);
#else
    tensor_new_2d(conv_in_pad, sst.hout*sst.wout, sst.kh*sst.kw*sst.cin, sizeof(float16_t), conv_pad_tmp_data);
#endif
    conv(&conv_out, conv_in, &conv_kernel_f16, &conv_in_pad, &sst);

    add(add_out, &conv_out, add_in);

    tensor_new_1d(alpha, sst.cout, sizeof(float16_t), palpha);
    tensor_new_1d(beta, sst.cout, sizeof(float16_t), pbeta);
    tensor_new_3d(batchnorm_out, sst.hout, sst.wout, sst.cout, sizeof(float16_t), batchnorm_tmp_data);
    batchnorm(&batchnorm_out, add_out, &alpha, &beta);

    relu(relu_out, &batchnorm_out, (float16_t)0);

    return 0;
}
