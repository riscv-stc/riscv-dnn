#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

// 2^n 
const float32_t der_1[] = {2.9103830456733704e-11, 5.820766091346741e-11, 1.1641532182693481e-10, 2.3283064365386963e-10,
                       4.656612873077393e-10, 9.313225746154785e-10, 1.862645149230957e-09, 3.725290298461914e-09,
                       7.450580596923828e-09, 1.4901161193847656e-08,
                       2.9802322387695312e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 2.384185791015625e-07,
                       4.76837158203125e-07, 9.5367431640625e-07, 1.9073486328125e-06, 3.814697265625e-06, 
                       7.62939453125e-06, 1.52587890625e-05, 3.0517578125e-05, 6.103515625e-05, 0.0001220703125,
                       0.000244140625, 0.00048828125, 0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625,
                       0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
                       1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0};
// 2^n / 2
const float32_t der_2[] = {1.4551915228366852e-11, 2.9103830456733704e-11, 5.820766091346741e-11, 1.1641532182693481e-10,
                       2.3283064365386963e-10, 4.656612873077393e-10, 9.313225746154785e-10, 1.862645149230957e-09,
                       3.725290298461914e-09, 7.450580596923828e-09,
                       1.4901161193847656e-08, 2.9802322387695312e-08, 5.960464477539063e-08, 1.1920928955078125e-07,
                       2.384185791015625e-07, 4.76837158203125e-07, 9.5367431640625e-07, 1.9073486328125e-06,
                       3.814697265625e-06, 7.62939453125e-06, 1.52587890625e-05, 3.0517578125e-05, 6.103515625e-05,
                       0.0001220703125, 0.000244140625, 0.00048828125, 0.0009765625, 0.001953125, 0.00390625, 0.0078125,
                       0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
                       512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0};
// 2^n / 6
const float32_t der_6[] = {4.850638409455617e-12, 9.701276818911234e-12, 1.9402553637822468e-11, 3.8805107275644936e-11,
                        7.761021455128987e-11, 1.5522042910257974e-10, 3.104408582051595e-10, 6.20881716410319e-10,
                        1.241763432820638e-09, 2.483526865641276e-09,
                       4.967053731282552e-09, 9.934107462565104e-09, 1.9868214925130207e-08, 3.9736429850260414e-08,
                       7.947285970052083e-08, 1.5894571940104166e-07, 3.178914388020833e-07, 6.357828776041666e-07,
                       1.2715657552083333e-06, 2.5431315104166665e-06, 5.086263020833333e-06, 1.0172526041666666e-05,
                       2.0345052083333332e-05, 4.0690104166666664e-05, 8.138020833333333e-05, 0.00016276041666666666,
                       0.0003255208333333333, 0.0006510416666666666, 0.0013020833333333333, 0.0026041666666666665,
                       0.005208333333333333, 0.010416666666666666, 0.020833333333333332, 0.041666666666666664,
                       0.08333333333333333, 0.16666666666666666, 0.3333333333333333, 0.6666666666666666,
                       1.3333333333333333, 2.6666666666666665, 5.333333333333333, 10.666666666666666, 21.333333333333332,
                       42.666666666666664, 85.33333333333333, 170.66666666666666, 341.3333333333333, 682.6666666666666,
                       1365.3333333333333, 2730.6666666666665, 5461.333333333333};

const float32_t ln2_h =  0.6931471805599453;
const float32_t ln2_recip_h = 1.4426950408889634;

const int offset = 35;
const float32_t offset_fp32 = 35.f;
const float32_t esize_fp32 = 4.f;

static inline int exp(Tensor *dst, Tensor *src)
{
    assert(dst->size == src->size);
    
    float16_t *psrc = (float16_t *)src->data;
    float16_t *pdst = (float16_t *)dst->data;

    int vlmax = VLENB  / src->elemsize;
    static int32_t tmp[VLENB*sizeof(int32_t)];

    for (int i = 0; i < src->size; i+=vlmax) {
        int vl = min(src->size - i, vlmax);
        vfloat16m1_t _src = vle16_v_f16m1(psrc + i, vl);
        vfloat32m2_t _src_f32 = vfwcvt_f_f_v_f32m2(_src, vl);

        vfloat32m2_t _n = vfmul_vf_f32m2(_src_f32, ln2_recip_h, vl); // _n = x/ln2_h
        vint32m2_t _n_int32 = vfcvt_x_f_v_i32m2(_n, vl); //_n_int32 = int(x/ln2_h)
        vfloat32m2_t _n_h = vfcvt_f_x_v_f32m2(_n_int32, vl); //_n_h: float16_t
        vfloat32m2_t _dx = vfnmsac_vf_f32m2(_src_f32, ln2_h, _n_h, vl); // _dx = x - _n_h*ln2

        // _n_int32 + 25 作为索引取index值
        vse32_v_i32m2(tmp, vadd_vx_i32m2(_n_int32, offset, vl), vl);
        vuint32m2_t _n_uint32_25 = vle32_v_u32m2((uint32_t *)tmp, vl);
        _n_uint32_25 = vmul_vx_u32m2(_n_uint32_25, 4, vl);
        vfloat32m2_t _der_6 = vloxei32_v_f32m2(der_6, _n_uint32_25, vl);
        vfloat32m2_t _der_2 = vloxei32_v_f32m2(der_2, _n_uint32_25, vl);
        vfloat32m2_t _der_1 = vloxei32_v_f32m2(der_1, _n_uint32_25, vl);
        vfloat32m2_t _res = vfmacc_vv_f32m2(_der_2, _der_6, _dx, vl);
        _res = vfmacc_vv_f32m2(_der_1, _res, _dx, vl);
        _res = vfmacc_vv_f32m2(_der_1, _res, _dx, vl);

        vse16_v_f16m1(pdst + i, vfncvt_f_f_w_f16m1(_res, vl), vl);
    }
    return 0;
}

vfloat16m1_t vfexp_f16m1(vfloat16m1_t vs1, int vl)
{
    assert(vl <= (VLENB * 8 / 2));

    static int32_t tmp[VLENB* 8 / 2*sizeof(int32_t)];

    vfloat32m2_t _src_f32 = vfwcvt_f_f_v_f32m2(vs1, vl);

    vfloat32m2_t _n = vfmul_vf_f32m2(_src_f32, ln2_recip_h, vl); // _n = x/ln2_h
    vint32m2_t _n_int32 = vfcvt_x_f_v_i32m2(_n, vl); //_n_int32 = int(x/ln2_h)
    vfloat32m2_t _n_h = vfcvt_f_x_v_f32m2(_n_int32, vl); //_n_h: float16_t
    vfloat32m2_t _dx = vfnmsac_vf_f32m2(_src_f32, ln2_h, _n_h, vl); // _dx = x - _n_h*ln2

    // _n_int32 + 25 作为索引取index值
    vse32_v_i32m2(tmp, vadd_vx_i32m2(_n_int32, offset, vl), vl);
    vuint32m2_t _n_uint32_25 = vle32_v_u32m2((uint32_t *)tmp, vl);
    _n_uint32_25 = vmul_vx_u32m2(_n_uint32_25, 4, vl);
    vfloat32m2_t _der_6 = vloxei32_v_f32m2(der_6, _n_uint32_25, vl);
    vfloat32m2_t _der_2 = vloxei32_v_f32m2(der_2, _n_uint32_25, vl);
    vfloat32m2_t _der_1 = vloxei32_v_f32m2(der_1, _n_uint32_25, vl);
    vfloat32m2_t _res = vfmacc_vv_f32m2(_der_2, _der_6, _dx, vl);
    _res = vfmacc_vv_f32m2(_der_1, _res, _dx, vl);
    _res = vfmacc_vv_f32m2(_der_1, _res, _dx, vl);

    return vfncvt_f_f_w_f16m1(_res, vl);
}

vfloat32m1_t vfexp_f32m1(vfloat32m1_t vs1, int vl)
{
    // assert(vl <= (VLENB * 8 / 4));

    vfloat32m1_t _n = vfmul_vf_f32m1(vs1, ln2_recip_h, vl); // _n = x/ln2_h
    vint32m1_t _n_int32 = vfcvt_x_f_v_i32m1(_n, vl); //_n_int32 = int(x/ln2_h)
    vfloat32m1_t _n_h = vfcvt_f_x_v_f32m1(_n_int32, vl); //_n_h: float16_t
    vfloat32m1_t _dx = vfnmsac_vf_f32m1(vs1, ln2_h, _n_h, vl); // _dx = x - _n_h*ln2

    _n_h = vfadd_vf_f32m1( _n_h, offset_fp32, vl );    
    vuint32m1_t _n_uint32_25 = vfcvt_xu_f_v_u32m1( _n_h, vl );
    _n_uint32_25 = vmul_vx_u32m1(_n_uint32_25, 4, vl);

    vfloat32m1_t _der_6 = vloxei32_v_f32m1(der_6, _n_uint32_25, vl);
    vfloat32m1_t _der_2 = vloxei32_v_f32m1(der_2, _n_uint32_25, vl);
    vfloat32m1_t _der_1 = vloxei32_v_f32m1(der_1, _n_uint32_25, vl);

    vfloat32m1_t _res = vfmacc_vv_f32m1(_der_2, _der_6, _dx, vl);
    _res = vfmacc_vv_f32m1(_der_1, _res, _dx, vl);
    _res = vfmacc_vv_f32m1(_der_1, _res, _dx, vl);

    return _res;
}

vfloat32m8_t vfexp_f32m8(vfloat32m8_t vs1, int vl)
{
    // assert(vl <= (VLENB * 8 / 4));

    vfloat32m8_t _n = vfmul_vf_f32m8(vs1, ln2_recip_h, vl); // _n = x/ln2_h
    vint32m8_t _n_int32 = vfcvt_x_f_v_i32m8(_n, vl); //_n_int32 = int(x/ln2_h)
    vfloat32m8_t _n_h = vfcvt_f_x_v_f32m8(_n_int32, vl); //_n_h: float16_t
    vfloat32m8_t _dx = vfnmsac_vf_f32m8(vs1, ln2_h, _n_h, vl); // _dx = x - _n_h*ln2

    _n_h = vfadd_vf_f32m8( _n_h, offset_fp32, vl );    
    vuint32m8_t _n_uint32_25 = vfcvt_xu_f_v_u32m8( _n_h, vl );
    _n_uint32_25 = vmul_vx_u32m8(_n_uint32_25, 4, vl);

    vfloat32m8_t _der_6 = vloxei32_v_f32m8(der_6, _n_uint32_25, vl);
    vfloat32m8_t _der_2 = vloxei32_v_f32m8(der_2, _n_uint32_25, vl);
    vfloat32m8_t _der_1 = vloxei32_v_f32m8(der_1, _n_uint32_25, vl); 

    vfloat32m8_t _res = vfmacc_vv_f32m8(_der_2, _der_6, _dx, vl);
    _res = vfmacc_vv_f32m8(_der_1, _res, _dx, vl);
    _res = vfmacc_vv_f32m8(_der_1, _res, _dx, vl);

    return _res;
}
