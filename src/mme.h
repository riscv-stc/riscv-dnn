#ifndef __MME_H__
#define __MME_H__

#include <stdint.h>
#include <riscv_vector.h>

typedef struct
{
    int hin;
    int win;
    int cin;
    
    int hout;
    int wout;
    int cout;

    int top;
    int bottom;
    int left;
    int right;

    int stride_h;
    int stride_w;

    int dilation_h;
    int dilation_w;

    int kh;
    int kw;

    int stride_src;
    int stride_ker;
    int stride_dst;

    int stride_addsrc;
    int stride_addout;
} Config;


#define config_conv_add(name, hin, win, cin, cout, \
            kh, kw, stride_h, stride_w, dilation_h, dilation_w, top, bottom, left, right, \
            stride_src, stride_ker, stride_dst, stride_addsrc, stride_addout) \
    Config name = { \
        hin, win, cin, \
        (hin + top + bottom - dilation_h * (kh - 1) - 1) / stride_h + 1, \
        (win + left + right - dilation_w * (kw - 1) - 1) / stride_w + 1, \
        cout, \
        top, bottom, left, right, stride_h, stride_w, dilation_h, dilation_w, kh, kw, \
        stride_src, stride_ker, stride_dst, stride_addsrc, stride_addout \
    };

#define config_conv(name, hin, win, cin, cout, \
            kh, kw, stride_h, stride_w, dilation_h, dilation_w, top, bottom, left, right, \
            stride_src, stride_ker, stride_dst) \
        config_conv_add(name, hin, win, cin, cout, \
            kh, kw, stride_h, stride_w, dilation_h, dilation_w, top, bottom, left, right, \
            stride_src, stride_ker, stride_dst, 0, 0)

#define config_pool(name, hin, win, cin, cout, \
            kh, kw, stride_h, stride_w, top, bottom, left, right, \
            stride_src, stride_dst) \
        config_conv_add(name, hin, win, cin, cout, \
            kh, kw, stride_h, stride_w, 1, 1, top, bottom, left, right, \
            stride_src, 0, stride_dst, 0, 0)
    // Config name = { \
    //     hin, win, cin, \
    //     (hin + top + bottom - kh) / stride_h + 1, \
    //     (win + left + right - kw) / stride_w + 1, \
    //     cout, \
    //     top, bottom, left, right, stride_h, stride_w, 1, 1, kh, kw, \
    //     stride_src, 0, stride_dst \
    // };
typedef struct
{
    int m;
    int k;
    int n;

    int stride_src1;
    int stride_src2;
    int stride_dst;
} ConfigMatmul;

#define config_matmul(name, m, k, n, stride_src1, stride_src2, stride_dst) \
            ConfigMatmul name = {m, k, n, stride_src1, stride_src2, stride_dst};

#ifndef __clang__
typedef __float16_t float16_t;
#else
typedef _Float16 float16_t;
#endif
typedef float float32_t;
// typedef _Float64 float64_t;

// #define VLEN 1024
// #define VLENB 128

#endif
