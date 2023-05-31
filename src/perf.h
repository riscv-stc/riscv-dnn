#ifndef _PERF_H
#define _PERF_H

#include "hpm.h"

#ifdef PERF

#ifdef __GEM5__
#define PERF_BEGIN()  enableCount();startCount()
#define PERF_END()    stopCount();
#else
#define PERF_BEGIN()  topDownCntSet()
#define PERF_END()    topDownCntGet()
#endif

#else

#define PERF_BEGIN()
#define PERF_END()

#endif

#endif // _PERF_H