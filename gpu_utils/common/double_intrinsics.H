#pragma once

#include "cuda_host_dev.H"


#ifdef __NVIDIA_BACKEND__

#define dADD(x,y) __dadd_rd(x,y)
#define dDIV(x,y) __ddiv_rd(x,y)
#define dMUL(x,y) __dmul_rd(x,y)
#define dSUB(x,y) __dsub_rd(x,y)

#else

#define dADD(x,y) (x+y)
#define dDIV(x,y) (x/y)
#define dMUL(x,y) (x*y)
#define dSUB(x,y) (x-y)

#endif