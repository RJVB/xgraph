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
