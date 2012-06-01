/*
 *  arrayvops.h
 *  XGraph
 *
 *  Created by René J.V. Bertin on 20111107.
 *  Copyright 2011 RJVB. All rights reserved.
 *
 */

#ifndef _ARRAYVOPS_H

#ifdef __cplusplus
extern "C" {
#endif

extern double ArrayCumAddScalar(double *xa, double b, double *sums, int N);
extern double ArrayCumAddArray(double *xa, double *xb, double *sums, int N);
extern void _convolve( double *Data, size_t NN, double *Mask, double *Output, int Start, int End, int Nm );

// moved to sse_mathfun.h
//#ifdef USE_SSE4
//	static inline double ssceil(double a)
//	{ v2df va = _mm_ceil_pd( _MM_SETR_PD(a,0) );
//		return *((double*)&va);
//	}
//
//	static inline double ssfloor(double a)
//	{ v2df va = _mm_floor_pd( _MM_SETR_PD(a,0) );
//		return *((double*)&va);
//	}
//#else
//// #	define ssceil(a)	ceil((a))
//// #	define ssfloor(a)	floor((a))
//	static inline double ssceil(double a)
//	{
//		return ceil(a);
//	}
//	static inline double ssfloor(double a)
//	{
//		return floor(a);
//	}
//#endif

#ifdef __cplusplus
}
#endif

#define _ARRAYVOPS_H
#endif
