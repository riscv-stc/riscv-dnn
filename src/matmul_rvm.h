#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "../include/matrix/matrix_intrinsic.h"

//#define FP16_ACC16 1

static inline int matmul_rvm_v3(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e16; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    for(int i = 0; i < h1; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
        for (int j = 0; j < w2; j += tile_n) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tile_n)
                        : [rs1]"r"(w2-j));
            asm volatile("mclracc acc0");
            for (int k = 0; k < w1; k += tile_k1) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k1)
                            : [rs1]"r"(w1-k));
                asm volatile("mle16.tr.r.a tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
                
                asm volatile("mle16.tr.r.b tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));
                asm volatile("mfwopa.mm acc0, tr0, tr1");
            }

            asm volatile("mfncvt.f.f.w acc0, acc0, %[rs2]"
                        :
                        : [rs2]"r"(1)); //set tile slice index

            // asm volatile("mse16.xa.r.c acc0, (%[rs1]), %[rs2]"
            //             :
            //             : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));

            for (int k = 0; k < tile_m; k ++) {
                vfloat16m1_t _sum16;
                asm volatile("mmv.v.xa.r.n %[vd], acc0, %[rs2]"
                            :[vd]"=vr"(_sum16)
                            :[rs2]"r"(k));
                vsetvl_e16m1(tile_n);
                asm volatile("vse16.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_sum16)
                            :[rs1]"r"(pdst+i*w2+j + k * w2));
            }

        }
    }
    return 0;
}

static inline int matmul_rvm_v3_tranA(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int k = src1->h;
    int m = src1->w;

    int k2 = src2->h;
    int n = src2->w;

    assert(k == k2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == m && wout == n);

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    asm volatile("msettypei x0, e16,true,false,maccd");
    for(int i = 0; i < m; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(m-i));
        for (int j = 0; j < n; j += tile_n) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tile_n)
                        : [rs1]"r"(n-j));
            asm volatile("mclracc acc0");
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mle16.tr.c.k tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+kk*m+i), [rs2]"r"(m*dataSize));
                
                asm volatile("mle16.tr.r.n tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+kk*n+j), [rs2]"r"(n*dataSize));
                asm volatile("mfwopa.vv acc0, tr0, tr1");
            }

            asm volatile("mfncvt.f.f.w acc0, acc0, %[rs2]"
                        :
                        : [rs2]"r"(1)); //set tile slice index
            asm volatile("mse16.xa.r.m acc0, (%[rs1]), %[rs2]"
                        : 
                        : [rs1]"r"(pdst+i*n+j), [rs2]"r"(n*dataSize));
        }
    }
    return 0;
}


static inline int matmul_rvm_v3_tranB(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int m = src1->h;
    int k = src1->w;

    int n = src2->h;
    int k2 = src2->w;

    assert(k == k2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == m && wout == n);

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k = 0;
    asm volatile("msettypei x0, e16,false,true,maccd");
    for(int i = 0; i < m; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(m-i));
        for (int j = 0; j < n; j += tile_n) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tile_n)
                        : [rs1]"r"(n-j));
            asm volatile("mclracc acc0");
            for (int kk = 0; kk < k; kk += tile_k) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k)
                            : [rs1]"r"(k-kk));
                asm volatile("mle16.tr.r.k tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*m+kk), [rs2]"r"(k*dataSize));
                
                asm volatile("mle16.tr.c.n tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+j*n+kk), [rs2]"r"(k*dataSize));
                asm volatile("mfwopa.vv acc0, tr0, tr1");
            }

            asm volatile("mfncvt.f.f.w acc0, acc0, %[rs2]"
                        :
                        : [rs2]"r"(1)); //set tile slice index
            asm volatile("mse16.xa.r.m acc0, (%[rs1]), %[rs2]"
                        : 
                        : [rs1]"r"(pdst+i*n+j), [rs2]"r"(n*dataSize));
        }
    }
    return 0;
}

static inline int matmul_rvm_v3_softpipe(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    float16_t *psrc1 = (float16_t *)src1->data;
    float16_t *psrc2 = (float16_t *)src2->data;
    float16_t *pdst = (float16_t *)dst->data;

    const int dataSize = sizeof(float16_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e16; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    for(int i = 0; i < h1; i += tile_m) {
        asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
        for (int j = 0; j < w2; j += tile_n) {
            asm volatile("msettilen %[rd], %[rs1]"
                        : [rd]"=r"(tile_n)
                        : [rs1]"r"(w2-j));
            asm volatile("mclracc acc0");
            // for (int k = 0; k < w1; k += tile_k1) {
                asm volatile("msettilek %[rd], %[rs1]"
                            : [rd]"=r"(tile_k1)
                            : [rs1]"r"(64));
                asm volatile("mle16.tr.r.k tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1), [rs2]"r"(w1*dataSize));
                
                asm volatile("mle16.tr.r.n tr1, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+j), [rs2]"r"(w2*dataSize));
                asm volatile("mfwopa.vv acc0, tr0, tr1");

                asm volatile("mle16.tr.r.k tr2, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+64), [rs2]"r"(w1*dataSize));
                
                asm volatile("mle16.tr.r.n tr3, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc2+64*w2+j), [rs2]"r"(w2*dataSize));
                asm volatile("mfwopa.vv acc0, tr2, tr3");
                
            // }

            asm volatile("mfncvt.f.f.w acc1, acc0, %[rs2]"
                        :
                        : [rs2]"r"(1)); //set tile slice index
            asm volatile("mse16.xa.r.m acc1, (%[rs1]), %[rs2]"
                        : 
                        : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
        }
    }
    return 0;
}

// static inline int matmul_int(Tensor *dst, Tensor *src1, Tensor *src2)
// {
//     int h1 = src1->h;
//     int w1 = src1->w;

//     int h2 = src2->h;
//     int w2 = src2->w;

//     assert(w1 == h2);

//     int hout = dst->h;
//     int wout = dst->w;

//     assert(hout == h1 && wout == w2);

//     int8_t *psrc1 = (int8_t *)src1->data;
//     int8_t *psrc2 = (int8_t *)src2->data;
//     int8_t *pdst = (int8_t *)dst->data;

//     const int dataSize = sizeof(int8_t);

//     int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
//     int mtype = e8; // sew = e16, mlmul = 128
//     asm volatile("msettype x0, %[rs1]"
//                 : 
//                 : [rs1]"r"(mtype));
//     asm volatile("msettilem %[rd], %[rs1]"
//                     : [rd]"=r"(tile_m)
//                     : [rs1]"r"(16));
//     asm volatile("msettilen %[rd], %[rs1]"
//                     : [rd]"=r"(tile_n)
//                     : [rs1]"r"(16));
//     asm volatile("mclracc acc0");
    
//     asm volatile("msettilek %[rd], %[rs1]"
//                     : [rd]"=r"(tile_k1)
//                     : [rs1]"r"(16));
//                 asm volatile("mle16.tr.r.k tr0, (%[rs1]), %[rs2]"
//                             :
//                             :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
                
//                 asm volatile("mle16.tr.r.n tr1, (%[rs1]), %[rs2]"
//                             :
//                             :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));
//                 asm volatile("mfwopa.vv acc0, tr0, tr1");

//                 // asm volatile("msettilek %[rd], %[rs1]"
//                 //             : [rd]"=r"(tile_k2)
//                 //             : [rs1]"r"(w1-k-tile_k1));
//                 // asm volatile("mle16.tr.r.k tr2, (%[rs1]), %[rs2]"
//                 //             :
//                 //             :[rs1]"r"(psrc1+i*w1+k+tile_k1), [rs2]"r"(w1*dataSize));
                
//                 // asm volatile("mle16.tr.r.n tr3, (%[rs1]), %[rs2]"
//                 //             :
//                 //             :[rs1]"r"(psrc2+(k+tile_k1)*w2+j), [rs2]"r"(w2*dataSize));
//                 // asm volatile("mfwopa.vv acc0, tr2, tr3");
                
//             }

//             asm volatile("mfncvt.f.f.w acc0, acc0, %[rs2]"
//                         :
//                         : [rs2]"r"(1)); //set tile slice index
//             asm volatile("mse16.xa.r.m acc0, (%[rs1]), %[rs2]"
//                         : 
//                         : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
//         }
//     }
//     return 0;
// }

// static inline int matmul_rvm_v3(Tensor *dst, Tensor *src1, Tensor *src2)
// {
//     int h1 = src1->h;
//     int w1 = src1->w;

//     int h2 = src2->h;
//     int w2 = src2->w;

//     assert(w1 == h2);

//     int hout = dst->h;
//     int wout = dst->w;

//     assert(hout == h1 && wout == w2);

//     float16_t *psrc1 = (float16_t *)src1->data;
//     float16_t *psrc2 = (float16_t *)src2->data;
//     float16_t *pdst = (float16_t *)dst->data;

//     const int dataSize = sizeof(float16_t);

//     int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
//     int mtype = e16; // sew = e16, mlmul = 128
//     vfloat16m1_t ma, mb, mch;
//     vfloat32m1_t mcf;
//     msettype(e16);
//     for(int i = 0; i < h1; i += tile_m) {
//         tile_m = msettilem(h1-i);
//         for (int j = 0; j < w2; j += tile_n) {
//             tile_n = msettilen(w2-j);
//             asm volatile("mclracc acc0");
//             for (int k = 0; k < w1; k += tile_k1) {
//                 tile_k1 = msettilek(w1-k);
//                 ma = mle16_tr_r_k_mhm1(psrc1+i*w1+k, w1*dataSize);
//                 mb = mle16_tr_r_n_mhm1(psrc2+k*w2+j, w2*dataSize);
                
//                 mcf = mfwopa_vv_mhm1(ma, mb);
//             }

//             mch = mfncvt_f_f_w_mhm1(mcf, 0);
//             mse16_xa_r_m_mhm1(mch, pdst+i*w2+j, w2*dataSize);
//         }
//     }
//     return 0;
// }

static inline int matmul(Tensor *dst, Tensor *src1, Tensor *src2) {
    return matmul_rvm_v3(dst, src1, src2);
}

#endif // __SRC_MATMUL_H__
