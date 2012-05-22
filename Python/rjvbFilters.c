/*
 * Python interface to filtring routines
 \ (c) 2005-2010 R.J.V. Bertin
 */
 
/*
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#ifdef __CYGWIN__
#	undef _WINDOWS
#	undef WIN32
#	undef MS_WINDOWS
#	undef _MSC_VER
#endif
#if defined(_WINDOWS) || defined(WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)
#	define MS_WINDOWS
#	define _USE_MATH_DEFINES
#endif

#include <Python.h>

#if defined(__GNUC__) && !defined(_GNU_SOURCE)
#	define _GNU_SOURCE
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "NaN.h"

#include <errno.h>

#include "rjvbFilters.h"

#ifndef False
#	define False	0
#endif
#ifndef True
#	define True	1
#endif

#ifndef StdErr
#	define StdErr	stderr
#endif

#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
	(var)=(low);\
}else if((var)>(high)){\
	(var)=(high);}
#endif
#define CLIP_EXPR(var,expr,low,high)	{ double l, h; if(((var)=(expr))<(l=(low))){\
	(var)=l;\
}else if((var)>(h=(high))){\
	(var)=h;}}

extern PyObject *FMError;

unsigned long fourconv3_nan_handling( double *data, size_t NN, int nan_handling )
{ unsigned long nNaN=0;
	if( nan_handling ){
	  size_t i, j, sL= 0, eL, sR= NN, eR;
	  /* sL: 0 or index to previous, non-NaN element before this block
	   \ eL: NN or index of first non-NaN element of a 'block'
	   \ sR: NN or index of last non-NaN element in a block.
	   \ eR: NN or index of next, non-NaN element after this block.
	   */
		for( i= 0; i< NN; ){
			if( isNaN(data[i]) ){
			  /* retrieve previous, non-NaN element: */
				sL= (i>0)? i-1 : 0;
				  /* Find next, non-NaN element: */
				j= i+1;
				while( j< NN && isNaN(data[j]) ){
					j++;
				}
				eL= j;
			}
			else{
				sL= i;
				eL= i;
			}
			if( eL< NN ){
/* 						j= i;	*/
				j= eL+1;
				  /* See if there is another NaN: */
				while( j< NN && !isNaN(data[j]) ){
					j++;
				}
				if( j< NN && isNaN(data[j]) ){
					sR= (j>0)? j-1 : 0; /* MAX(j-1,0); */
					j+= 1;
					while( j< NN && isNaN(data[j]) ){
						j++;
					}
					eR= j;
				}
				else{
					sR= NN;
					eR= NN;
				}
			}
			if( sL== 0 && sR>= NN ){
			  /* Nothing special to be done: there are no NaNs here. */
			}
			else{
				if( sL != eL ){
					if( !isNaN(data[sL]) ){
						switch( nan_handling ){
							case 2:{
							    double slope= (data[eL] - data[sL]) / (eL - sL);
								  /* we have a non-NaN preceding value: fill the gap of NaN(s) with a linear
								   \ gradient.
								   */
#ifdef DEBUG
								fprintf( StdErr, "'left' NaN hole from %lu-%lu; filling with gradient {", sL+1, eL-1 );
#endif
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= data[sL] + slope* (j-sL);
									nNaN++;
#ifdef DEBUG
									fprintf( StdErr, "%g,", data[j] );
#endif
								}
#ifdef DEBUG
								fprintf( StdErr, "}\n" );
#endif
								break;
							}
							default:
							case 1:{
							  size_t halfway= sL + (eL-sL)/2;
#ifdef DEBUG
								fprintf( StdErr, "'left' NaN hole from %lu-%lu; filling with step\n", sL+1, eL-1 );
#endif
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= (j<halfway)? data[sL] : data[eL];
									nNaN++;
								}
								break;
							}
						}
					}
					else{
						  /* Best guess we can do is to fill the gap with the first non-NaN value we have at hand */
						  /* 20050108: must pad from sL and not from sL+1 ! */
#ifdef DEBUG
						fprintf( StdErr, "'left' NaN hole from %lu-%lu; padding with %g (%d)\n", sL+0, eL-1, data[eL], __LINE__ );
						fprintf( StdErr, "\td[%d]=%s d[%d]=%s d[%d]=%s d[%d]=%s\n",
							sL, ad2str(data[sL],NULL,NULL), sL+1, ad2str(data[sL+1],NULL,NULL),
							eL-1, ad2str(data[eL-1],NULL,NULL),
							eL, ad2str(data[eL],NULL,NULL), eL+1, ad2str(data[eL+1],NULL,NULL)
						);
#endif
						if( sL==0 && eL==1 ){
							data[0]= data[eL];
							nNaN++;
						}
						else{
							for( j= sL+0; j< eL && j< NN; j++ ){
								data[j]= data[eL];
								nNaN++;
							}
						}
					}
				}
				if( sR != eR ){
					if( eR< NN && !isNaN(data[eR]) ){
						switch( nan_handling ){
							case 2:{
							  double slope= (data[eR] - data[sR]) / (eR - sR);
#ifdef DEBUG
								fprintf( StdErr, "'right' NaN hole from %lu-%lu; filling with gradient {", sR+1, eR-1 );
#endif
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= data[sR] + slope* (j-sR);
									nNaN++;
#ifdef DEBUG
									fprintf( StdErr, "%g,", data[j] );
#endif
								}
#ifdef DEBUG
								fprintf( StdErr, "}\n" );
#endif
								break;
							}
							case 1:
							default:{
							  size_t halfway= sR + (eR-sR)/2;
#ifdef DEBUG
								fprintf( StdErr, "'right' NaN hole from %lu-%lu; filling with step\n", sR+1, eR-1 );
#endif
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= (j<halfway)? data[sR] : data[eR];
									nNaN++;
								}
								break;
							}
						}
					}
					else{
						  /* Best guess we can do is to fill the gap with the first non-NaN value we have at hand */
#ifdef DEBUG
						fprintf( StdErr, "'right' NaN hole from %lu-%lu; padding with %g\n", sR+1, eR-1, data[sR] );
#endif
						for( j= sR+1; j< eR && j< NN; j++ ){
							data[j]= data[sR];
							nNaN++;
						}
					}
				}
			}
			i= sR+1;
		}
	}
	return( nNaN );
}

static void _convolve( double *Data, double *Mask, double *Output, int Start, int End, int Nm )
{ int nm= Nm/ 2, i, j, k;
	for( i= Start; i< End; i++ ){
	 double accum= 0;
		for( j= 0; j< Nm; j++ ){
		  double v;
			k= i+ j- nm;
			if( k< 0 ){
				v= Data[0];
			}
			else if( k< End ){
				v= Data[k];
			}
			else{
				v= Data[End-1];
			}
			accum+= Mask[j]* v;
		}
		Output[i]= accum;
	}
}

double *convolve( double *Data, int N, double *Mask, int Nm, int nan_handling )
{ int padding, NN, i;
  double *output;
	  // we will padd the input data with half the mask's width, in order to avoid boundary artefacts
	padding = Nm/2 + 1;
	NN = N + 2 * padding;
	if( (output = (double*) PyMem_New(double, NN)) ){
	  double *data= (double*) malloc( NN * sizeof(double));
		if( data ){
			  // make a copy of the input data, with the required amount of padding
			  // with the initial and last observed values:
			for( i= 0; i < padding; i++ ){
				data[i]= Data[0];
			}
			memcpy( &data[i], Data, N*sizeof(double) );
			i += N;
			for( ; i < NN; i++ ){
				data[i]= Data[N-1];
			}
			/* If requested, treat the data (source, input) array for NaNs. Gaps with
			 \ NaNs are filled with a linear gradient between the surrounding values (nan_handling==2)
			 \ or with a 'half-way step' between these values (nan_handling==1) if possible, otherwise,
			 \ simple padding with the first or last non-NaN value is done.
			 \ These estimates are removed after the convolution.
			 */
			fourconv3_nan_handling( data, NN, nan_handling );
			_convolve( data, Mask, output, 0, NN, Nm );
			memmove( output, &output[padding], N*sizeof(double) );
			  // replace the original NaN values where they ought to go:
			if( nan_handling ){
				for( i= 0; i < N; i++ ){
					if( isNaN(Data[i]) ){
						output[i]= Data[i];
					}
				}
			}
			free(data);
		}
		else{
			PyErr_NoMemory();
		}
	}
	else{
		PyErr_NoMemory();
	}
	return( output );
}

#ifdef MINDOUBLE
#	define TINY MINDOUBLE
#else
#	define TINY DBL_MIN
#endif

int ludcmp(savgol_flp **a, int n, int *indx, savgol_flp *d)
/* 
 \ Given a matrix a[1..n][1..n], this routine replaces it by the LU decomposition of a rowwise
 \ permutation of itself. a and n are input. a is output, arranged as in equation (2.3.14) above;
 \ indx[1..n] is an output vector that records the row permutation effected by the partial
 \ pivoting; d is output as if 1 depending on whether the number of row interchanges was even
 \ or odd, respectively. This routine is used in combination with lubksb to solve linear equations
 \ or invert a matrix.
 */
{ int i, imax= 0, j, k;
  savgol_flp big, dum, sum, temp;
	  /* vv stores the implicit scaling of each row. 	*/
  savgol_flp *vv = (savgol_flp*) malloc( (n+1)*sizeof(savgol_flp) );

	if( !vv ){
		return(0);
	}
	*d= 1.0;
	for( i= 1; i<= n; i++ ){
		big= 0.0;
		for( j= 1; j<= n; j++ ){
			if( (temp= fabs(a[i][j])) > big){
				big= temp;
			}
		}
		if( big == 0.0){
			PyErr_SetString( FMError, "Singular matrix in routine ludcmp" );
			errno= EINVAL;
			xfree(vv);
			return(0);
		}
		vv[i]= 1.0/big;
	}
	for( j= 1; j<= n; j++ ){
		for( i= 1; i< j; i++ ){
			sum= a[i][j];
			for( k= 1; k< i; k++ ){
				sum -= a[i][k]*a[k][j];
			}
			a[i][j]= sum;
		}
		big= 0.0;
		for( i= j; i<= n; i++ ){
			sum= a[i][j];
			for( k= 1; k< j; k++){
				sum -= a[i][k]*a[k][j];
			}
			a[i][j]= sum;
			if( (dum= vv[i]*fabs(sum)) >= big ){
				big= dum;
				imax= i;
			}
		}
		if( j != imax ){
			for( k= 1; k<= n; k++ ){
				dum= a[imax][k];
				a[imax][k]= a[j][k];
				a[j][k]= dum;
			}
			*d = -(*d);
			vv[imax]= vv[j];
		}
		indx[j]= imax;
		if( a[j][j] == 0.0){
			a[j][j]= TINY;
		}
		if( j != n ){
			dum= 1.0/(a[j][j]);
			for( i= j+1; i<= n; i++) a[i][j] *= dum;
		}
	}
	xfree(vv);
	return(1);
}

void lubksb(savgol_flp **a, int n, int *indx, savgol_flp *b)
/* 
 \ Solves the set of n linear equations A . X = B. Here a[1..n][1..n] is input, not as the matrix
 \ A but rather as its LU decomposition, determined by the routine ludcmp. indx[1..n] is input
 \ as the permutation vector returned by ludcmp. b[1..n] is input as the right-hand side vector
 \ B, and returns with the solution vector X. a, n, andindx are not modified by this routine
 \ and can be left in place for successive calls with dirent right-hand sides b. This routine takes
 \ into account the possibility that b will begin with many zero elements, so it is efficient for use
 \ in matrix inversion.
 */
{ int i, ii= 0, ip, j;
  savgol_flp sum;
	for( i= 1; i<= n; i++ ){
	  /* When ii is set to a positive value, it will become the
	   \ index of the first nonvanishing element of b. We now
	   \ do the forward substitution. The
	   \ only new wrinkle is to unscramble the permutation as we go.
	   */
		ip= indx[i];
		sum= b[ip];
		b[ip]= b[i];
		if( ii){
			for( j= ii; j<= i-1; j++){
				sum -= a[i][j]*b[j];
			}
		}
		else if( sum){
		  /* A nonzero element was encountered, so from now on we
		   \ will have to do the sums in the loop above.
		   */
			ii= i;
		}
		b[i]= sum;
	}
	for( i= n; i>= 1; i-- ){
	  /* Now we do the backsubstitution.	*/
		sum= b[i];
		for( j= i+1; j<= n; j++){
			sum -= a[i][j]*b[j];
		}
		  /* Store a component of the solution vector X.	*/
		b[i]= sum/ a[i][i];
	}
}


void xfree_sgf_matrix( savgol_flp **a, int h, int v )
{ int i;
	if( a ){
		for( i= 0; i<= v; i++ ){
			xfree( a[i] );
		}
		xfree( a );
	}
}

savgol_flp **calloc_dmatrix( int h, int v)
{ int i;
  savgol_flp **m;

	  /* 20010901: RJVB: allocate 1 element more per row/column. Adaptation of NR
	   \ code dealing with matrices is tricky business...
	   */
	if( !(m = (savgol_flp **) calloc((unsigned) v+1,sizeof(savgol_flp*))) ){
		PyErr_SetString( FMError, "allocation failure 1 in calloc_dmatrix" );
		return( NULL );
	}
	for( i = 0; i <= v; i++ ){
		if( !(m[i] = (savgol_flp *) calloc((unsigned) h+ 1, sizeof(savgol_flp))) ){
			PyErr_SetString( FMError, "allocation failure 2 in calloc_dmatrix" );
			for( --i; i>= 0; i-- ){
				xfree( m[i] );
			}
			xfree(m);
			return(NULL);
		}
	}
	return( m );
}

/* savgol():
 \ Returns in c[1..np], in wrap-around order (N.B.!) consistent with the argument respns in
 \ routine convlv, a set of Savitzky-Golay filter coefficients. nl is the number of leftward (past)
 \ data points used, while nr is the number of rightward (future) data points, making the total
 \ number of data points used nl +nr +1. ld is the order of the derivative desired (e.g., ld = 0
 \ for smoothed function). m is the order of the smoothing polynomial, also equal to the highest
 \ conserved moment; usual values are m = 2or m = 4.
 */
int savgol(savgol_flp *c, int np, int nl, int nr, int ld, int m)
{ void lubksb(savgol_flp **a, int n, int *indx, savgol_flp *b);
  int ludcmp(savgol_flp **a, int n, int *indx, savgol_flp *d);
  int imj, ipj, j, k, kk, mm;
  savgol_flp d, fac, sum,**a;
  int *indx = (int*) malloc( (m+2) * sizeof(int) );
  savgol_flp *b = (savgol_flp*) malloc( (m+2) * sizeof(savgol_flp) );

	if( !indx || !b ){
		PyErr_NoMemory();
		return(0);
	}
	if( np < nl+nr+1 || nl < 0 || nr < 0 || ld > m || nl+nr < m){
		fprintf( StdErr, "bad args in savgol\n");
		errno= EINVAL;
		xfree(indx); xfree(b);
		return(0);
	}
	if( !(a= calloc_dmatrix(m+2, m+2)) || !indx || !b ){
		xfree(indx); xfree(b);
		return(0);
	}
	for( ipj= 0; ipj<= (m << 1); ipj++ ){
	  /* Set up the normal equations of the desired least-squares fit.	*/
		sum= (ipj)? 0.0 : 1.0;
		for( k= 1; k<= nr; k++){
			sum += (savgol_flp) pow((double) k, (double) ipj);
		}
		for( k= 1; k<= nl; k++){
			sum += (savgol_flp) pow((double)-k, (double) ipj);
		}
		mm= IMIN(ipj, 2*m-ipj);
		for( imj = -mm; imj<= mm; imj+= 2){
#ifdef DEBUG
		  int a1= 1+(ipj+imj)/ 2, a2= 1+(ipj-imj)/ 2;
			if( a1< m+2 && a2< m+2 ){
				a[a1][a2]= sum;
			}
			else{
				fprintf( stderr, "Range error in savgol.%d: a1=%d or a2=%d >= m+2=%d\n",
					__LINE__, a1, a2, m
				);
			}
#else
			a[1+(ipj+imj)/ 2][1+(ipj-imj)/ 2]= sum;
#endif
		}
	}
	if( !ludcmp(a, m+1, (int*) indx, &d) ){
		PyErr_SetString( FMError, "failure in ludcmp()" );
		xfree_sgf_matrix(a, m+2, m+2);
		xfree(indx); xfree(b);
		return(0);
	}
	for( j= 1; j<= m+1; j++){
		b[j]= 0.0;
	}
	b[ld+1]= 1.0;
	lubksb(a, m+1, indx, b);
	for( kk= 1; kk<= np; kk++){
	  /* Zero the output array (it may be bigger than number of coefficients).	*/
		c[kk]= 0.0;
	}
	for( k = -nl; k<= nr; k++ ){
	  /* Each Savitzky-Golay coefficient is the dot product of powers of
	    \an integer with the inverse matrix row.
	    */
		sum= b[1];
		fac= 1.0;
		for( mm= 1; mm<= m; mm++){
			sum += b[mm+1]*(fac *= k);
		}
		kk= ((np-k) % np)+1;	/* Store in wrap-around order.	*/
#if DEBUG
		if( kk> np ){
			fprintf( stderr, "Range error in savgol.%d: kk=%d > np=%d\n",
				__LINE__, kk, np
			);
		}
		else
#endif
		c[kk]= sum;
	}
	xfree_sgf_matrix(a, m+2, m+2);
	xfree(indx); xfree(b);
	return(1);
}
