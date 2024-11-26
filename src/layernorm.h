#ifndef __LAYERNORM_H__
#define __LAYERNORM_H__

#include "mme.h"
#include "tensor.h"
#include <riscv_matrix.h>
#include <stddef.h>

/*
 *   dst [H, W, C]
 *   src [H, W, C]
 *   gamma [1, 1, C]
 *   beta [1, 1, C]
 */
void layernorm(void *dst, void *src, void *gamma, void *beta, Config *ss) {
  const int H = ss->hin, W = ss->win, C = ss->cin;
  const int M = H * W;
  const int N = C;
  float16_t *psrc = (float16_t *)src;
  float16_t *pdst = (float16_t *)dst;
  float16_t *pgamma = (float16_t *)gamma;
  float16_t *pbeta = (float16_t *)beta;
  float16_t mean_buffer[H * W];
  float16_t var_buffer[H * W];

  int tilem, tilen;
  msettypei(0x1);
  msettypehi(0x1);
  msettileni(1);
  // mean [H, W, 1]
  for (int m = 0; m < M; m += tilem) {
    tilem = msettilem(M - m);
    mfloat16_t sum;
    sum = mfsub_mm(sum, sum);
    mfloat16_t den;
    den = mfmv_s_f(den, (float)C, 0);
    den = mbcce_m(den);

    for (int n = 0; n < N; ++n) {
      mfloat16_t tr0 = mlc_m(psrc + m * N + n, N * sizeof(float16_t));
      sum = mfadd_mm(sum, tr0);
    }
    sum = mfdiv_mm(sum, den);
    msc_m(sum, mean_buffer + m, sizeof(float16_t));
  }
  // variance [H, W, 1]
  for (int m = 0; m < M; m += tilem) {
    tilem = msettilem(M - m);
    mfloat16_t var, sum;
    var = mfsub_mm(var, var);
    sum = mfsub_mm(sum, sum);
    mfloat16_t den;
    den = mfmv_s_f(den, (float)C, 0);
    den = mbcce_m(den);
    mfloat16_t mean = mlc_m(mean_buffer + m, sizeof(float16_t));

    for (int n = 0; n < N; ++n) {
      mfloat16_t tr1 = mlc_m(psrc + m * N + n, N * sizeof(float16_t));
      var = mfsub_mm(tr1, mean);
      var = mfmul_mm(var, var);
      sum = mfadd_mm(sum, var);
    }
    var = mfdiv_mm(sum, den);
    msc_m(var, var_buffer + m, sizeof(float16_t));
  }
  // norm [H, W, C]
  for (int m = 0; m < M; m += tilem) {
    tilem = msettilem(M - m);
    mfloat16_t eps;
    eps = mfmv_s_f(eps, 1e-6, 0);
    eps = mbcce_m(eps);
    mfloat16_t var = mlc_m(var_buffer + m, sizeof(float16_t));
    mfloat16_t den = mfsqrt_m(mfadd_mm(var, eps));
    mfloat16_t mean = mlc_m(mean_buffer + m, sizeof(float16_t));

    for (int n = 0; n < N; ++n) {
      mfloat16_t tr1 = mlc_m(psrc + m * N + n, N * sizeof(float16_t));
      tr1 = mfsub_mm(tr1, mean);
      tr1 = mfdiv_mm(tr1, den);
      msc_m(tr1, pdst + m * N + n, N * sizeof(float16_t));
    }
  }
  // affine [H, W, C]
  msettilemi(1);
  for (int n = 0; n < N; n += tilen) {
    tilen = msettilen(N - n);
    mfloat16_t bias = mlc_m(pbeta + n, sizeof(float16_t));
    mfloat16_t weight = mlc_m(pgamma + n, sizeof(float16_t));

    for (int m = 0; m < M; ++m) {
      mfloat16_t tr1 = mlc_m(pdst + m * N + n, N * sizeof(float16_t));
      tr1 = mfmul_mm(tr1, weight);
      tr1 = mfadd_mm(tr1, bias);
      msc_m(tr1, pdst + m * N + n, N * sizeof(float16_t));
    }
  }
}

#endif