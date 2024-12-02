#ifndef __MATRIX_INTRINSIC_H__
#define __MATRIX_INTRINSIC_H__

#include <riscv_matrix.h>
#include <riscv_vector.h>

// Only Matrix-0.5
#define CONFIGURATION_MTYPE(hi, li)                                            \
  do {                                                                         \
    msettypei((li));                                                           \
    msettypehi((hi));                                                          \
  } while (0);

// SET DTYPE=FP16 SEW=16
#define SET_MBA0_FP16() CONFIGURATION_MTYPE(0x1, 0x1)
// SET DTYPE=FP32 SEW=32
#define SET_MBA0_FP32() CONFIGURATION_MTYPE(0x4, 0x2)
// SET DTYPE=FP16,F32 SEW=16
#define SET_MBA0_FP16_FP32() CONFIGURATION_MTYPE(0x5, 0x1)
// SET DTYPE=FP16,FP32 SEW=32
#define SET_MBA0_FP32_FP16() CONFIGURATION_MTYPE(0x5, 0x2)

#endif // __MATRIX_INTRINSIC_H__