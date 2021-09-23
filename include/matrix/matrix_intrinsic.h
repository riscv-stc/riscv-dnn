#include <riscv_vector.h>

#define MWIDTH 128
#define e8  0
#define e16 1
#define e32 2
#define e64 3

inline int msettype(int mtype) {
    int rd;
    asm volatile("msettype %[rd], %[rs1]"
                : [rd]"=r"(rd)
                : [rs1]"r"(mtype));
    return rd;
}

inline int msettilem(int size) {
    int rd;
    asm volatile("msettilem %[rd], %[rs1]"
                : [rd]"=r"(rd)
                : [rs1]"r"(size));
    return rd;
}

inline int msettilek(int size) {
    int rd;
    asm volatile("msettilek %[rd], %[rs1]"
                : [rd]"=r"(rd)
                : [rs1]"r"(size));
    return rd;
}

inline int msettilen(int size) {
    int rd;
    asm volatile("msettilen %[rd], %[rs1]"
                : [rd]"=r"(rd)
                : [rs1]"r"(size));
    return rd;
}

inline vfloat16m1_t mle16_ma(float16_t *src, int stride) {
    vfloat16m1_t vd;
    asm volatile("mle16.m.a %[vd], (%[rs1]), %[rs2]"
                : [vd]"=vr"(vd)
                : [rs1]"r"(src), [rs2]"r"(stride));
    return vd;
}

inline  vfloat16m1_t mle16_mb(float16_t *src, int stride) {
    vfloat16m1_t vd;
    asm volatile("mle16.m.b %[vd], (%[rs1]), %[rs2]"
                : [vd]"=vr"(vd)
                : [rs1]"r"(src), [rs2]"r"(stride));
    return vd;
}

inline vfloat16m1_t mle16_mc(float16_t *src, int stride) {
    vfloat16m1_t vd;
    asm volatile("mle16.m.c %[vd], (%[rs1]), %[rs2]"
                : [vd]"=vr"(vd)
                : [rs1]"r"(src), [rs2]"r"(stride));
    return vd;
}

inline vfloat32m2_t mfwmul_mm(vfloat32m2_t acc, vfloat16m1_t a, vfloat16m1_t b) {
    asm volatile("mfwmul.mm %[vd], %[vs1], %[vs2]"
                : [vd]"=vr"(acc)
                : [vs1]"vr"(a), [vs2]"vr"(b));
    return acc;
}

inline void mse16_mc(vfloat16m1_t dst, float16_t *src, int stride) {
    asm volatile("mse16.m.c %[vd], (%[rs1]), %[rs2]"
                : [vd]"=vr"(dst)
                : [rs1]"r"(src), [rs2]"r"(stride));
}

inline vfloat16m1_t m_vfncvt_f_f_w(vfloat32m2_t src) {
    vfloat16m1_t vd;
    asm volatile("vfncvt.f.f.w %[vd], %[vs2]"
                :[vd]"=vr"(vd)
                :[vs2]"vr"(src));
    return vd;
}

inline vfloat32m2_t m_vfmv_v_f(float32_t src) {
    vfloat32m2_t vd;
    asm volatile("vfmv.v.f %[vd], %[rs1]"
                : [vd]"=vr"(vd)
                : [rs1]"f"(src));
    return vd;
}