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
} Config;


#define config_conv(name, hin, win, cin, cout, \
            top, bottom, left, right, kh, kw, stride_h, stride_w, dilation_h, dilation_w) \
    Config name = { \
        hin, win, cin, \
        (hin + top + bottom - dilation_h * (kh - 1) - 1) / stride_h + 1, \
        (win + left + right - dilation_w * (kw - 1) - 1) / stride_w + 1, \
        cout, \
        top, bottom, left, right, stride_h, stride_w, dilation_h, dilation_w, kh, kw \
    };

#define config_pool(name, hin, win, cin, cout, \
            top, bottom, left, right, kh, kw, stride_h, stride_w) \
    Config name = { \
        hin, win, cin, \
        (hin + top + bottom - kh) / stride_h + 1, \
        (win + left + right - kw) / stride_w + 1, \
        cout, \
        top, bottom, left, right, stride_h, stride_w, 1, 1, kh, kw \
    };

#define config_padding(name, hin, win, cin, top, bottom, left, right) \
    static Config name = { \
        hin, win, cin, \
        hin + top + bottom, \
        win + left + right, \
        cin, \
        top, bottom, left, right, 1, 1, 1, 1, 1, 1 \
    };

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
