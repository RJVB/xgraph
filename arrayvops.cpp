/*
 *  arrayvops.cpp
 *  XGraph
 *
 *  Created by René J.V. Bertin on 20111107.
 *  Copyright 2011 RJVB. All rights reserved.
 *
 */

#include "config.h"
IDENTIFY("macstl based array operations");

#include "stddef.h"
#include "stdint.h"
#include "stdlib.h"

#ifdef __SSE2__
#	define USE_SSE2
#endif
#ifdef __SSE3__
#	define USE_SSE2
#	define USE_SSE3
#endif
#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE2
#	define USE_SSE4
#endif

#include <valarray>
#include <macstl/valarray.h>
#include "sse_mathfun.h"

#include "arrayvops.h"

double ArrayCumAddScalar(double *xa, double b, double *sums, int N)
{ double sum;
	if( xa && sums && N > 0 ){
	  stdext::refarray<double> vxa(xa,N), vsums(sums,N);
		sum = (vsums = vxa + b).sum();
	}
	else{
		sum = 0.0;
	}
	return sum;
}

double ArrayCumAddArray(double *xa, double *xb, double *sums, int N)
{ double sum;
	if( xa && xb && sums && N > 0 ){
	  stdext::refarray<double> vxa(xa,N), vxb(xb,N), vsums(sums,N);
		sum = (vsums = vxa + vxb).sum();
	}
	else{
		sum = 0.0;
	}
	return sum;
}

void ArrayCumAddIArraySSE(double *rsum, int *xa, int *xb, int *sums, int N)
{
	if( xa && xb && sums && N > 0 ){
	  v4si *va = (v4si*) xa, *vb = (v4si*) xb, *vs = (v4si*) sums, vsum = _MM_SETZERO_SI128();
	  int i, N_4;
		N_4 = N-4 + 1;
		for( i = 0 ; i < N_4 ; va++, vb++, vs++ ){
			*vs = _mm_add_epi32( *va, *vb );
			vsum = _mm_add_epi32( vsum, *vs );
			i += 4;
		}
		_mm_empty();
		*rsum = (double)VELEM(int,vsum,0) + (double)VELEM(int,vsum,1) + (double)VELEM(int,vsum,2) + (double)VELEM(int,vsum,3);
		for( ; i < N; i++ ){
			*rsum += (sums[i] = xa[i] + xb[i]);
		}
	}
	else{
		*rsum = 0.0;
	}
}

void ArrayCumAddArraySSE(double *rsum, double *xa, double *xb, double *sums, int N)
{
	if( xa && xb && sums && N > 0 ){
	  v2df *va = (v2df*) xa, *vb = (v2df*) xb, *vs = (v2df*) sums, vsum = _MM_SETZERO_PD();
	  int i, N_1;
		N_1 = N-1;
		for( i = 0 ; i < N_1 ; va++, vb++, vs++ ){
			*vs = _mm_add_pd( *va, *vb );
			vsum = _mm_add_pd( vsum, *vs );
			i += 2;
		}
		*rsum = VELEM(double,vsum,0) + VELEM(double,vsum,1);
		if( i == N_1 ){
			*rsum += (sums[i] = xa[i] + xb[i]);
		}
	}
	else{
		*rsum = 0.0;
	}
}

void ArrayCumAddScalarSSE(double *rsum, double *xa, double xb, double *sums, int N)
{
	if( xa && xb && sums && N > 0 ){
	  v2df *va = (v2df*) xa, vb = _MM_SET1_PD(xb), *vs = (v2df*) sums, vsum = _MM_SETZERO_PD();
	  int i, N_1;
		N_1 = N-1;
		for( i = 0 ; i < N_1 ; va++, vs++ ){
			*vs = _mm_add_pd( *va, vb );
			vsum = _mm_add_pd( vsum, *vs );
			i += 2;
		}
		*rsum = VELEM(double,vsum,0) + VELEM(double,vsum,1);
		if( i == N_1 ){
			*rsum += (sums[i] = xa[i] + xb);
		}
	}
	else{
		*rsum = 0.0;
	}
}

void ArrayCumSubIArraySSE(double *rsum, int *xa, int *xb, int *sums, int N)
{
	if( xa && xb && sums && N > 0 ){
	  v4si *va = (v4si*) xa, *vb = (v4si*) xb, *vs = (v4si*) sums, vsum = _MM_SETZERO_SI128();
	  int i, N_4;
		N_4 = N-4+1;
		for( i = 0 ; i < N_4 ; va++, vb++, vs++ ){
			*vs = _mm_sub_epi32( *va, *vb );
			vsum = _mm_add_epi32( vsum, *vs );
			i += 4;
		}
		_mm_empty();
		*rsum = (double)VELEM(int,vsum,0) + (double)VELEM(int,vsum,1) + (double)VELEM(int,vsum,2) + (double)VELEM(int,vsum,3);
		for( ; i < N; i++ ){
			*rsum += (sums[i] = xa[i] - xb[i]);
		}
	}
	else{
		*rsum = 0.0;
	}
}

void ArrayCumSubArraySSE(double *rsum, double *xa, double *xb, double *sums, int N)
{
	if( xa && xb && sums && N > 0 ){
	  v2df *va = (v2df*) xa, *vb = (v2df*) xb, *vs = (v2df*) sums, vsum = _MM_SETZERO_PD();
	  int i, N_1;
		N_1 = N-1;
		for( i = 0 ; i < N_1 ; va++, vb++, vs++ ){
			*vs = _mm_sub_pd( *va, *vb );
			vsum = _mm_add_pd( vsum, *vs );
			i += 2;
		}
		*rsum = VELEM(double,vsum,0) + VELEM(double,vsum,1);
		if( i == N_1 ){
			*rsum += (sums[i] = xa[i] - xb[i]);
		}
	}
	else{
		*rsum = 0.0;
	}
}

void ArrayCumMulArraySSE(double *rsum, double *xa, double *xb, double *sums, int N)
{
	if( xa && xb && sums && N > 0 ){
	  v2df *va = (v2df*) xa, *vb = (v2df*) xb, *vs = (v2df*) sums, vsum = _MM_SETZERO_PD();
	  int i, N_1;
		N_1 = N-1;
		for( i = 0 ; i < N_1 ; va++, vb++, vs++ ){
			*vs = _mm_mul_pd( *va, *vb );
			vsum = _mm_add_pd( vsum, *vs );
			i += 2;
		}
		*rsum = VELEM(double,vsum,0) + VELEM(double,vsum,1);
		if( i == N_1 ){
			*rsum += (sums[i] = xa[i] * xb[i]);
		}
	}
	else{
		*rsum = 0.0;
	}
}

void ArrayCumMulScalarSSE(double *rsum, double *xa, double xb, double *sums, int N)
{
	if( xa && xb && sums && N > 0 ){
	  v2df *va = (v2df*) xa, vb = _MM_SET1_PD(xb), *vs = (v2df*) sums, vsum = _MM_SETZERO_PD();
	  int i, N_1;
		N_1 = N-1;
		for( i = 0 ; i < N_1 ; va++, vs++ ){
			*vs = _mm_mul_pd( *va, vb );
			vsum = _mm_add_pd( vsum, *vs );
			i += 2;
		}
		*rsum = VELEM(double,vsum,0) + VELEM(double,vsum,1);
		if( i == N_1 ){
			*rsum += (sums[i] = xa[i] * xb);
		}
	}
	else{
		*rsum = 0.0;
	}
}

void _convolve( double *Data, size_t NN, double *Mask, double *Output, int Start, int End, int Nm )
{ int nm= Nm/ 2, i, j;
  size_t end = NN - nm;
  // a refarray is like a valarray except that it uses the memory we already have:
  stdext::refarray<double> vmask(Mask,Nm), vdata(Data, NN);
	for( i= Start; i< End; i++ ){
		if( i < nm ){
		  int k;
		  stdext::valarray<double> vd = vdata[stdext::slice(0, Nm,1)].shift(i-nm);
			j = 0;
			do{
				k = i+ j- nm;
				vd[j] = Data[0];
				j++;
			} while( k < 0 && j < Nm );
			Output[i]= (vmask * vd).sum();
		}
		else if( i > end ){
		  int k;
		  stdext::valarray<double> vd = vdata[stdext::slice(NN-Nm, Nm,1)].shift(end-i);
			j = End + nm - i;
			do{
				k = i+ j- nm;
				vd[j] = Data[End-1];
				j++;
			} while( j < Nm );
			Output[i]= (vmask * vd).sum();
		}
		else{
			Output[i]= (vmask * vdata[stdext::slice(i-nm, Nm,1)]).sum();
		}
#if 0
		else{

			for( j= 0; j< Nm; j++ ){
				k = i+ j- nm;
				if( k< 0 ){
					vd[j] = Data[0];
				}
				else if( k< End ){
					vd[j] = Data[k];
				}
				else{
					vd[j] = Data[End-1];
				}
			}
			Output[i]= (vmask * vd).sum();
		}
#endif
	}
}
