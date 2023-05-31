#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <riscv_vector.h>
#include "encoding.h"

#ifdef __SPIKE__
#define DEBUG_PRINT 1
#define CACHELINE 0
#else
#define DEBUG_PRINT 0
#define CACHELINE 128
#endif

#ifndef __clang__
typedef __float16_t float16_t;
#else
typedef _Float16 float16_t;
#endif
typedef float float32_t;
typedef double float64_t;

#define stringify1(x) #x
#define stringify(x) stringify1(x)
#define assert(x) do { \
  if (x) break; \
  printf("Assertion failed: " stringify(x) "\n"); \
  exit(3); \
} while(0)

#define VLEN  (read_csr(vlenb) * 8)
#define VLENB  (read_csr(vlenb))

#define min(a, b)  (a < b? a: b)
#define max(a, b)  (a > b? a: b)

typedef struct {
    int shape[4];
    int dims;
    int elemsize;
    int size;
    void *data;
    int stride;
} Tensor;

#define tensor_new_1d_with_stride(name, _w, _elemsize, _data, stride) \
    Tensor name = { {0, _w, 0, 0}, 1, _elemsize, _w, _data, stride==0?_w*_elemsize: stride};

#define tensor_new_2d_with_stride(name, _h, _w, _elemsize, _data, stride) \
    Tensor name = { {_h, _w, 0, 0}, 2, _elemsize, _h*_w, _data, stride==0?_w*_elemsize: stride};

#define tensor_new_3d_with_stride(name, _h, _w, _cin, _elemsize, _data, stride) \
    Tensor name = { {_h, _w, _cin, 0}, 3, _elemsize, _h*_w*_cin, _data, stride==0?_cin*_elemsize: stride};

#define tensor_new_4d_with_stride(name, _h, _w, _cin, _cout, _elemsize, _data, stride) \
    Tensor name = { {_h, _w, _cin, _cout}, 4, _elemsize, _h*_w*_cin*_cout, _data, stride==0?_cout*_elemsize: stride};

#define tensor_new_1d_with_data(name, _w, _elemsize) \
    static uint8_t name ## _data[ _w * _elemsize] __attribute__((aligned(64))); \
    static Tensor name = {[0, _w, 0, 0], 1, _elemsize, _w, name ## _data, _w *_elemsize };

#define tensor_new_2d_with_data(name, _h, _w, _elemsize) \
    static uint8_t name ## _data[ _h * _w * _elemsize] __attribute__((aligned(64))); \
    static Tensor name = { [_h, _w, 0, 0], 2, _elemsize, _h*_w, name ## _data,  _w * _elemsize};

#define tensor_new_3d_with_data(name, _h, _w, _cin, _elemsize) \
    static uint8_t name ## _data[ _h * _w * _cin *  _elemsize] __attribute__((aligned(64))); \
    static Tensor name = { [_h, _w, _cin, 0], 3, _elemsize, _h*_w*_cin, name ## _data, _cin* _elemsize };

#define tensor_new_4d_with_data(name, _h, _w, _cin, _cout, _elemsize) \
    static uint8_t name ## _data[ _h * _w * _cin * _cout * _elemsize] __attribute__((aligned(64))); \
    static Tensor name = {  [_h, _w, _cin, _cout], 4, _elemsize, _h*_w*_cin*_cout, name ## _data, _cout * _elemsize };


#define tensor_new_1d(name, _w, _elemsize, _data) \
    tensor_new_1d_with_stride(name, _w, _elemsize, _data, _w*_elemsize)

#define tensor_new_2d(name, _h, _w, _elemsize, _data) \
    tensor_new_2d_with_stride(name, _h, _w, _elemsize, _data, _w*_elemsize) \

#define tensor_new_3d(name, _h, _w, _cin, _elemsize, _data) \
    tensor_new_3d_with_stride(name, _h, _w, _cin, _elemsize, _data, _cin * _elemsize) \

#define tensor_new_4d(name, _h, _w, _cin, _cout, _elemsize, _data) \
    tensor_new_4d_with_stride(name, _h, _w, _cin, _cout, _elemsize, _data, _cout * _elemsize) \

#endif
