#ifndef __PARAMS_H__
#define __PARAMS_H__

#ifndef HIN
#define HIN 7
#endif

#ifndef WIN
#define WIN 7
#endif

#ifndef CIN
#define CIN 64
#endif

#ifndef COUT
#define COUT 64
#endif

#ifndef KH
#define KH 3
#endif

#ifndef KW
#define KW 3
#endif

#ifndef STRIDE_H
#define STRIDE_H 1
#endif

#ifndef STRIDE_W
#define STRIDE_W 1
#endif

#ifndef DILATION_H
#define DILATION_H 1
#endif

#ifndef DILATION_W
#define DILATION_W 1
#endif

#ifndef PAD_TOP
#define PAD_TOP 0
#endif

#ifndef PAD_BOTTOM
#define PAD_BOTTOM 0
#endif

#ifndef PAD_LEFT
#define PAD_LEFT 0
#endif

#ifndef PAD_RIGHT
#define PAD_RIGHT 0
#endif

#define HOUT ( (HIN + PAD_TOP + PAD_BOTTOM - DILATION_H * (KH - 1) - 1) / STRIDE_H + 1)
#define WOUT ( (WIN + PAD_LEFT + PAD_RIGHT - DILATION_W * (KW - 1) - 1) / STRIDE_W + 1)

#define PAD_SIZE ((HOUT * WOUT) * KH * KW * CIN)

#define IN_SIZE  (HIN * WIN * CIN)
#define OUT_SIZE (HOUT * WOUT * COUT)

#ifndef NLOOPS
#define NLOOPS 1
#endif

#endif
