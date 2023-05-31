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

#define HOUT (HIN + PAD_TOP + PAD_BOTTOM)
#define WOUT (WIN + PAD_LEFT + PAD_RIGHT)

#define COUT CIN

#define OUT_SIZE (HOUT * WOUT * COUT)

#ifndef NLOOPS
#define NLOOPS 1
#endif

#endif
