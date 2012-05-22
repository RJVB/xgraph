#include "config.h"
IDENTIFY( "Fourier/Convolution ascanf library module using FFTW 3" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif

/* #if defined(__APPLE_CC__) || defined(__MACH__)	*/
  /* For the time being... */
/* #	undef HAVE_FFTW	*/
/* #endif	*/

#if !defined(CHKLCA_MAX) && defined(DEBUG)
#	define CHKLCA_MAX	1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#ifdef __cplusplus
#	include <valarray>
#	include <macstl/valarray.h>
#endif
#include "arrayvops.h"

  /* Get the dynamic module definitions:	*/
#include "dymod.h"

// extern FILE *StdErr;

#include "copyright.h"

  /* Include a whole bunch of headerfiles. Not all of them are strictly necessary, but if
   \ we want to have fdecl.h to know all functions we possibly might want to call, this
   \ list is needed.
   */
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "NaN.h"

#include "fdecl.h"

#undef SQR
static double sqrval;
#define SQR(v,type)	(type)(((sqrval=(v)))? sqrval*sqrval : 0)

  /* get the ascanf definitions:	*/
#include "ascanf.h"
  /* If we want to be able to access the "expression" field in the callback argument, we need compiled_ascanf.h .
   \ If we don't include it, we will just get 'theExpr= NULL' ... (this applies to when -DDEBUG)
   */
#include "compiled_ascanf.h"
#include "ascanfc-table.h"

#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;
	int (*ascanf_Arrays2Regular_ptr)(void *malloc, void *free);
	double *ascanf_elapsed_values_ptr;

#	define ascanf_Arrays2Regular (*ascanf_Arrays2Regular_ptr)
#	define ascanf_elapsed_values	(ascanf_elapsed_values_ptr)

#include <float.h>

#if defined(FFTW_SINGLE)
	  /* If you don't need the precision, use the single precision Savitzky-Golay routines. On a PIII, they're about
	   \ twice faster -- without even using mmx/sse instructions. Beware of the FFTW routines, though: they're still
	   \ bugged.
	   */
	typedef float savgol_flp;
#else
	typedef double savgol_flp;
#endif

/* savgol functions: */
/* discr_fourtr():
 \ Replaces data[1..2*nn] by its discrete Fourier transform, if isign is input as 1; or replaces
 \ data[1..2*nn] by nn times its inverse discrete Fourier transform, if isign is input as -1.
 \ data is a complex array of length nn or, equivalently, a real array of length 2*nn. nn MUST
 \ be an integer power of 2 (this is not checked for!).
 */
void discr_fourtr(savgol_flp *data, unsigned long nn, int isign)
{ unsigned long n, mmax, m, j, istep, i;
  savgol_flp wtemp, wr, wpr, wpi, wi, theta;
  savgol_flp tempr, tempi;
    /* is = isign* 2	*/
  int is= (isign>=0)? 2 : -2;

	n= nn << 1;
	j= 1;
	for( i= 1; i< n; i+= 2 ){
	  /* This is the bit-reversal section of the routine.	*/
		if(j > i ){
			SWAP(data[j], data[i], savgol_flp); /* Exchange the two complex numbers.	*/
			SWAP(data[j+1], data[i+1], savgol_flp);
		}
		m= n >> 1;
		while( m >= 2 && j > m ){
			j -= m;
			m >>= 1;
		}
		j += m;
	}
	  /* Here begins the Danielson-Lanczos section of the routine.	*/
	mmax= 2;
	while( n > mmax ){
	  /* Outer loop executed log 2 nn times.	*/
		istep= mmax << 1;
		  /* RJB: is = isign* 2 : this allows to use the system macro M_PI without
		   \ further calculations (instead of 6.28318 etc.)
		   */
		theta= is*(M_PI/ mmax);	/* Initialize the trigonometric recurrence.	*/
		wtemp= sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi= sin(theta);
		wr= 1.0;
		wi= 0.0;
		for( m= 1; m< mmax; m+= 2 ){
		  /* Here are the two nested inner loops.	*/
			for( i= m; i<= n; i+= istep ){
				j= i+mmax;
				  /* This is the Danielson-Lanczos formula:	*/
				tempr= wr*data[j]-wi*data[j+1];
				tempi= wr*data[j+1]+wi*data[j];
				data[j]= data[i]-tempr;
				data[j+1]= data[i+1]-tempi;
				data[i] += tempr;
				data[i+1] += tempi;
			}
			wr= (wtemp= wr)*wpr-wi*wpi+wr;	/* Trigonometric recurrence.	*/
			wi= wi*wpr+wtemp*wpi+wi;
		}
		mmax= istep;
	}
}


/* twofft():
 \ Given two real input arrays data1[1..n] and data2[1..n], this routine calls discr_fourtr and
 \ returns two complex output arrays, fft1[1..2n] and fft2[1..2n], each of complex length
 \ n (i.e., real length 2*n), which contain the discrete Fourier transforms of the respective data
 \ arrays. n MUST be an integer power of 2.
 */
void twofft(savgol_flp *data1, savgol_flp *data2, savgol_flp *fft1, savgol_flp *fft2, unsigned long n)
{ unsigned long nn3, nn2, jj, j;
  savgol_flp rep, rem, aip, aim;
	nn3= 1+(nn2= 2+n+n);
	for( j= 1, jj= 2; j<= n; j++, jj+= 2 ){
	  /* Pack the two real arrays into one complex array.	*/
		fft1[jj-1]= data1[j];
		fft1[jj]= data2[j];
	}
	discr_fourtr(fft1, n, 1);	/* Transform the complex array.	*/
	fft2[1]= fft1[2];
	fft1[2]= fft2[2]= 0.0;
	for( j= 3; j<= n+1; j+= 2 ){
		rep= 0.5*(fft1[j]+fft1[nn2-j]);	/* Use symmetries to separate the two transforms.	*/
		rem= 0.5*(fft1[j]-fft1[nn2-j]);
		aip= 0.5*(fft1[j+1]+fft1[nn3-j]);
		aim= 0.5*(fft1[j+1]-fft1[nn3-j]);
		fft1[j]= rep;					/* Ship them out in two complex arrays.	*/
		fft1[j+1]= aim;
		fft1[nn2-j]= rep;
		fft1[nn3-j] = -aim;
		fft2[j]= aip;
		fft2[j+1] = -rem;
		fft2[nn2-j]= aip;
		fft2[nn3-j]= rem;
	}
}


/* realft():
 \ Calculates the Fourier transform of a set of n real-valued data points. Replaces this data (which
 \ is stored in array data[1..n]) by the positive frequency half of its complex Fourier transform.
 \ The real-valued first and last components of the complex transform are returned as elements
 \ data[1] and data[2], respectively. n must be a power of 2. This routine also calculates the
 \ inverse transform of a complex data array if it is the transform of real data. (Result in this case
 \ must be multiplied by 2/ n.)
 */
void realft(savgol_flp *data, unsigned long n, int isign)
{ unsigned long i, i1, i2, i3, i4, np3;
  savgol_flp c1= 0.5, c2, h1r, h1i, h2r, h2i;
  savgol_flp wr, wi, wpr, wpi, wtemp, theta;

	theta= M_PI/((savgol_flp) (n>> 1));	/* Initialize the recurrence.	*/
	if( isign>= 0 ){
		c2 = -0.5;
		discr_fourtr(data, n>> 1, 1);	/* The forward transform is here.	*/
	}
	else{
		c2= 0.5;	/* Otherwise set up for an inverse transform.	*/
		theta = -theta;
	}
	wtemp= sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi= sin(theta);
	wr= 1.0+wpr;
	wi= wpi;
	np3= n+3;
	for( i= 2; i<= (n>> 2); i++ ){
	  /* Case i= 1 done separately below.	*/
		i4= 1+ (i3= np3-(i2= 1+(i1= i+i-1)));
		h1r= c1*(data[i1]+data[i3]);	/* The two separate transforms are separated out of data.	*/
		h1i= c1*(data[i2]-data[i4]);
		h2r= -c2*(data[i2]+data[i4]);
		h2i= c2*(data[i1]-data[i3]);
		  /* Here they are recombined to form the true transform of the original real data.	*/
		data[i1]= h1r+wr*h2r-wi*h2i;
		data[i2]= h1i+wr*h2i+wi*h2r;
		data[i3]= h1r-wr*h2r+wi*h2i;
		data[i4]= -h1i+wr*h2i+wi*h2r;
		wr= (wtemp= wr)*wpr-wi*wpi+wr;	/* The recurrence.	*/
		wi= wi*wpr+wtemp*wpi+wi;
	}
	if( isign>= 0 ){
		  /* Squeeze the first and last data together to get them all within the original array.	*/
		data[1]= (h1r= data[1])+data[2];
		data[2]= h1r-data[2];
	}
	else{
	  /* This is the inverse transform for the case isign=-1.	*/
		data[1]= c1*((h1r= data[1])+data[2]);
		data[2]= c1*(h1r-data[2]);
		discr_fourtr(data, n>> 1,-1);
	}
}

/*
 \ convlv():
 \ Convolves or deconvolves a real data set data[1..n] (including any user-supplied zero padding)
 \ with a response function respns[1..n]. The response function must be stored in wrap-around
 \ order in the first m elements of respns, wheremis an odd integer  n. Wrap-around order
 \ means that the first half of the array respns contains the impulse response function at positive
 \ times, while the second half of the array contains the impulse response function at negative times,
 \ counting down from the highest element respns[m]. On input isign is >=0 for convolution,
 \ <0 for deconvolution. The answer is returned in the first n components of ans. However,
 \ ans must be supplied in the calling program with dimensions [1..2*n], for consistency with
 \ twofft. n MUST be an integer power of two.
 */
int convlv(savgol_flp *data, unsigned long n, savgol_flp *respns, unsigned long m, int isign, savgol_flp *ans, Boolean preproc_respns )
{ unsigned long i, no2;
  savgol_flp dum, mag2;
    /* 20040122: dynamic allocation like with alloca() *crashes* on OS X 10.2.8/gcc 3.3 when n==32768, hence
     \ (n<<1)+1==65537 and fft_len==524296 ??!! This reeks like an alignment problem, as it does not happen
	\ when using xgalloca() (through _XGALLOCA which always calls that function); xgalloca() uses calloc and malloc
	\ internally.
	*/
  double *fft;

	if( !(fft = (double*) malloc( ((n<<1)+1) * sizeof(double) )) ){
		return(0);
	}
	if( preproc_respns ){
	  /* Whatever this is supposed to do, it seems it shouldn't do it
	   \ when we're called with respns initialised by savgol().
	   */
		for( i= 1; i<= (m-1)/2; i++){
		  /* Put respns in array of length n.	*/
			respns[n+1-i]= respns[m+1-i];
		}
		for( i=(m+3)/2; i<= n-(m-1)/2; i++){
		  /* Pad with zeros.	*/
			respns[i]= 0.0;
		}
	}
	twofft( (double*) data, respns, fft, ans, n);	/* FFT both at once.	*/
	no2= n>> 1;
	for( i= 2; i<= n+2; i+= 2 ){
		if( isign >= 0 ){
			ans[i-1]= (fft[i-1]*(dum= ans[i-1])-fft[i]*ans[i])/no2;	/*  Multiply FFTs to convolve.	*/
			ans[i]= (fft[i]*dum+fft[i-1]*ans[i])/no2;
		}
		else{
			if( (mag2=SQR(ans[i-1],savgol_flp)+SQR(ans[i],savgol_flp)) == 0.0){
				fprintf( StdErr, "Deconvolving at response zero in convlv");
				errno= EINVAL;
				GCA();
				return(0);
			}
			ans[i-1]= (fft[i-1]*(dum= ans[i-1])+fft[i]*ans[i])/ mag2/ no2;	/* Divide FFTs to deconvolve.	*/
			ans[i]= (fft[i]*dum-fft[i-1]*ans[i])/ mag2/ no2;
		}
	}
	ans[2]= ans[n+1];	/* Pack last element with first for realft.	*/
	realft(ans, n,-1);	/* Inverse transform back to time domain.	*/
	xfree(fft);
	return(1);
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
  ALLOCA(vv, savgol_flp, n+1, vv_len);

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
			fprintf( StdErr, "Singular matrix in routine ludcmp (row %d)\n", i);
			errno= EINVAL;
			GCA();
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
/* 				SWAP( a[imax][k], a[j][k], savgol_flp );	*/
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
	GCA();
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
		fprintf( StdErr, "allocation failure 1 in calloc_dmatrix(%d,%d) (%s)\n",
			h, v, serror()
		);
		return( NULL );
	}
	for( i = 0; i <= v; i++ ){
		if( !(m[i] = (savgol_flp *) calloc((unsigned) h+ 1, sizeof(savgol_flp))) ){
			fprintf( StdErr, "allocation failure 2 in row %d in calloc_dmatrix(%d,%d) (%s)\n",
				i, h, v,
				serror()
			);
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
  ALLOCA( indx, int, m+2, indx_len);
  ALLOCA( b, savgol_flp, m+2, b_len);

	if( np < nl+nr+1 || nl < 0 || nr < 0 || ld > m || nl+nr < m){
		fprintf( StdErr, "bad args in savgol\n");
		errno= EINVAL;
		GCA();
		return(0);
	}
	if( !(a= calloc_dmatrix(m+2, m+2)) || !indx || !b ){
		GCA();
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
				fprintf( StdErr, "Range error in savgol.%d: a1=%d or a2=%d >= m+2=%d\n",
					__LINE__, a1, a2, m
				);
			}
#else
			a[1+(ipj+imj)/ 2][1+(ipj-imj)/ 2]= sum;
#endif
		}
	}
	if( !ludcmp(a, m+1, (int*) indx, &d) ){
		fprintf( StdErr, "savgol: failure in ludcmp (%s)\n", serror() );
		GCA();
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
			fprintf( StdErr, "Range error in savgol.%d: kk=%d > np=%d\n",
				__LINE__, kk, np
			);
		}
		else
#endif
		c[kk]= sum;
	}
	xfree_sgf_matrix(a, m+2, m+2);
	GCA();
	return(1);
}


static DataSet *SavGolSet= NULL;
static int SavGolHSize, SavGolDOrder= 0, SavGolSOrder= 4, SavGolN;
static savgol_flp *SavGolCoeffs= NULL;

typedef struct SGResults{
	savgol_flp *smoothed;
} SGResults;
static SGResults *SavGolResult= NULL;
static unsigned long SGRcols= 0, SGRN= 0;
static int SGfw= 0;

static int alloc_SGResults( SGResults **SG, int cols, unsigned long N )
{ int i;
  SGResults *sg;
	if( !SG ){
		return(0);
	}
	sg= *SG;
	if( sg && SGRcols> cols ){
		for( i= cols; i< SGRcols; i++ ){
			xfree( sg[i].smoothed );
		}
	}
	if( (sg= (SGResults*) XGrealloc( sg, cols* sizeof(SGResults))) ){
		if( cols> SGRcols ){
			for( i= SGRcols; i< cols; i++ ){
				sg[i].smoothed= NULL;
			}
		}
		for( i= 0; i< cols; i++ ){
			if( !( sg[i].smoothed= (savgol_flp*) XGrealloc( sg[i].smoothed, N* sizeof(savgol_flp))) ){
				fprintf( StdErr, "alloc_SGResults: allocation failure for set-column %d (%s)\n", i, serror() );
				for( --i; i>= 0; i-- ){
					xfree( sg[i].smoothed );
				}
				xfree( sg );
				*SG= NULL;
				SGRcols= 0;
				SGRN= 0;
				return(0);
			}
		}
	}
	else{
		*SG= NULL;
		return(0);
	}
	*SG= sg;
	  /* Alllmost reentrant... ;-)	*/
	SGRcols= cols;
	SGRN= N;
	return( cols* N );
}

static unsigned long free_SGResults( SGResults **SG )
{ unsigned long i, cols= SGRcols, N= SGRN;
  SGResults *sg;
	if( !SG || !*SG ){
		return(0);
	}
	sg= *SG;
	for( i= 0; i< cols; i++ ){
		xfree( sg[i].smoothed );
	}
	xfree( sg );
	*SG= NULL;
	SGRcols= 0;
	SGRN= 0;
	return( N* cols );
}

int ascanf_SavGolCoeffs ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *CF= NULL;
	set_NaN(*result);
	if( args && ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
		CF= parse_ascanf_address(args[0], _ascanf_array, "ascanf_SavGolayCoeffs", (int) ascanf_verbose, NULL );
		if( CF && CF->array ){
		  unsigned long N;
		  savgol_flp *coeffs=NULL;
		  int i, fw, fo, deriv;

			CLIP_EXPR( fw, (int) args[1], 0, (MAXINT-1)/2 );
			if( ascanf_arguments> 2 ){
				  /* Put arbitrary upper limit on the order of the smoothing polynomial	*/
				  /* 990902: outcommented the -1	*/
				CLIP_EXPR( fo, (int) args[2], 0, 2* fw /* - 1 */ );
			}
			else{
				CLIP_EXPR( fo, 4, 0, 2* fw );
			}
			if( ascanf_arguments> 3 ){
				CLIP_EXPR( deriv, (int) args[3], -fo, fo );
			}
			else{
				deriv= 0;
			}
			N= fw*2+3;
			errno= 0;
			if( !(coeffs= (savgol_flp*) XGrealloc(coeffs, (N+ 1)* sizeof(savgol_flp) )) ){
				fprintf( StdErr, "SavGolayCoeffs: error getting necessary memory (%s)\n", serror() );
				if( ascanf_window ){
					xtb_error_box( ascanf_window, "SavGolayCoeffs: error getting necessary memory", "Failure" );
				}
				ascanf_arg_error= 1;
				ascanf_emsg= "(memory problem)";
				return(0);
			}
			else{
				if( CF->N!= N ){
					Resize_ascanf_Array( CF, N, result );
				}
				if( !(fw== 0 && fo== 0 && deriv==0) ){
					savgol( &(coeffs)[-1], N, fw, fw, deriv, fo );
					  /* Unwrap the coefficients into the target memory:	*/
					CF->array[N/2]= coeffs[0];
					for( i= 1; i<= N/2; i++ ){
						CF->array[N/2-i]= coeffs[i];
						CF->array[N/2+i]= coeffs[N-i];
					}
				}
				else{
					memset( CF->array, 0, N* sizeof(double) );
				}
				CF->value= CF->array[ (CF->last_index= 0) ];
				if( CF->accessHandler ){
					AccessHandler( CF, "SavGolayCoeffs", level, ASCB_COMPILED, AH_EXPR, NULL  );
				}
				xfree( coeffs );
				*result= (CF->own_address)? CF->own_address : take_ascanf_address(CF);
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr,
						" (determined convolution mask for Savitzky-Golay filter with width %d, smoothing polynomial order %d, order-of-derivative %d",
						fw, fo, deriv
					);
					fprintf( StdErr, " (used internal N=%lu)", N );
					fputs( ")== ", StdErr );
				}
			}
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_SavGolInit ( ASCB_ARGLIST )
{ ASCB_FRAME
  int idx;
	*result= 1;
	if( args && ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
		idx= (int) args[0];
		if( idx>= 0 && idx< setNumber && AllSets[idx].numPoints> 0 ){
		  unsigned long N, pwr2;
		  savgol_flp *coeffs=NULL, *input= NULL, pad_low, pad_high;
		  int fw, pad= 0;
			SavGolSet= &AllSets[idx];
			SavGolN= SavGolSet->numPoints;
			CLIP_EXPR( SavGolHSize, (int) args[1], 0, SavGolN/2-1 );
			if( ascanf_arguments> 2 ){
				  /* Put arbitrary upper limit on the order of the smoothing polynomial	*/
				  /* 990902: outcommented the -1	*/
				CLIP_EXPR( SavGolSOrder, (int) args[2], 0, 2* SavGolHSize /* - 1 */ );
			}
			else{
				CLIP_EXPR( SavGolSOrder, 4, 0, 2* SavGolHSize );
			}
			if( ascanf_arguments> 3 ){
/* 				CLIP_EXPR( SavGolDOrder, (int) args[3], 0, SavGolSOrder );	*/
				CLIP_EXPR( SavGolDOrder, (int) args[3], -SavGolSOrder, SavGolSOrder );
			}
			else{
				SavGolDOrder= 0;
			}
			fw= 2* SavGolHSize+ 1;
			  /* We *must* allocate a number of elements that is a power of 2..
			   \ We include twice the filterwidth, to pad around the data
			   */
			if( ascanf_arguments > 4 && args[4] ){
			  savgol_flp p;
				if( args[4]> 0 && (p= floor(args[4]))< MAXINT ){
					pad= p;
				}
				else{
					pad= SavGolHSize;
				}
			}
			else{
				pad= 0;
			}
			pwr2= (unsigned long) ceil( log(SavGolN+ 2* pad)/log(2) );
			  /* 991021: does it have to be <= ??	*/
			if( (N= 1 << pwr2)< SavGolN+ 2* pad ){
				while( N< SavGolN+ 2* pad ){
					N*= 2;
				}
			}
			errno= 0;
			if( !(
					  /* We only get SavGolHSize*2+1 coefficients, but the same array is going to be
					   \ used by convlv(..,respns), which must be the same size as the input data-array.
					   */
					(SavGolCoeffs= (savgol_flp*) XGrealloc(SavGolCoeffs, (N+ 1)* sizeof(savgol_flp) )) &&
					(coeffs= (savgol_flp*) XGrealloc(coeffs, (N+ 1)* sizeof(savgol_flp) )) &&
					(input= (savgol_flp*) XGrealloc(input, (N+ 1)* sizeof(savgol_flp) )) &&
					(alloc_SGResults( &SavGolResult, SavGolSet->ncols, 2*N+ 2 ))
				)
			){
				free_SGResults( &SavGolResult );
				xfree( SavGolCoeffs );
				xfree( coeffs );
				xfree( input );
				fprintf( StdErr, "SavGolayInit: error getting necessary memory (%s)\n", serror() );
				if( ascanf_window ){
					xtb_error_box( ascanf_window, "SavGolayInit: error getting necessary memory", "Failure" );
				}
				*result= 0;
				ascanf_arg_error= 1;
				ascanf_emsg= "(memory problem)";
				return(0);
			}
			else{
			  int i, j;
				if( (SavGolHSize== 0 && SavGolSOrder== 0 && SavGolDOrder==0) ||
					savgol( SavGolCoeffs, N, SavGolHSize, SavGolHSize, SavGolDOrder, SavGolSOrder )
				){
				  Boolean pl_set= False, ph_set= False;

					  /* This was inside the ncols loop?? */
					memcpy( coeffs, SavGolCoeffs, (N+1)* sizeof(savgol_flp) );

					for( i= 0; i< SavGolSet->ncols; i++ ){

						if( pad ){
							if( ascanf_arguments> 5 && !NaN(args[5]) ){
								pad_low= args[5];
								pl_set= True;
							}
							else{
								pad_low= SavGolSet->columns[i][0];
							}
							if( ascanf_arguments> 6 && !NaN(args[6]) ){
								pad_high= args[6];
								ph_set= True;
							}
							else{
								pad_high= SavGolSet->columns[i][SavGolN-1];
							}
						}
						for( j= 1; j<= pad; j++ ){
						  /* Leading-pad over the filterwidth with the initial value (could be halfwidth..)	*/
							input[j]= pad_low;
						}
#if defined(FFTW_SINGLE)
						{ savgol_flp *in= &input[1+pad];
						  double *value= SavGolSet->columns[i];
							for( j= 0; j< SavGolN; j++ ){
								*in++= *value++;
							}
						}
#else
						memcpy( &input[1+pad], SavGolSet->columns[i], SavGolN* sizeof(savgol_flp) );
#endif
						for( j= pad+SavGolN+1; j<= N; j++ ){
						  /* Trailing-pad over the filterwidth with the final value:	*/
							input[j]= pad_high;
						}
						  /* Pass pointer to smoothed[-1] as target area, so that NR's unit offset array
						   \ maps to a normal, C, 0 offset array. Since we don't address smoothed[-1]
						   \ (no read nor write), this should not pose problems..
						   */
						if( (SavGolHSize== 0 && SavGolSOrder== 0 && SavGolDOrder==0) ){
							memcpy( SavGolResult[i].smoothed, &input[1], N* sizeof(savgol_flp) );
						}
						else if( !(convlv( input, N, coeffs, fw, 1, &(SavGolResult[i].smoothed[-1]), False )) ){
							fprintf( StdErr, "SavGolayInit(): some problem filtering column %d (%s)\n", i, serror() );
							*result= 0;
							free_SGResults( &SavGolResult );
							xfree( SavGolCoeffs );
							xfree( coeffs );
							return(0);
						}
					}
					  /* Get rid of the extra memory allocated:	the maximal difference increases linearly
					   \ with SavGolN.
					   */
					alloc_SGResults( &SavGolResult, SavGolSet->ncols, SavGolN+ pad );
					SGfw= pad;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr,
							" (filtered set %d filter-width %d, smoothing polynomial order %d, order-of-derivative %d",
							idx, fw, SavGolSOrder, SavGolDOrder
						);
						if( pad ){
							fprintf( StdErr, ", padded over %d values, left with %s, right with %s",
								pad,
								(pl_set)? ad2str(pad_low, NULL, NULL) : "<column[0]>",
								(ph_set)? ad2str( pad_high, NULL, NULL) : "<column[N-1]>"
							);
						}
						fprintf( StdErr, " (used internal N=%lu)", N );
						fputs( ")== ", StdErr );
					}

					if( ascanf_arguments> 7 ){
					  ascanf_Function *CF;
					  int N= SavGolHSize*2+3;
						if( !(CF= parse_ascanf_address(args[7], _ascanf_array, "ascanf_SavGolayInit", (int) ascanf_verbose, NULL )) || CF->iarray ){
							ascanf_emsg= " (invalid coeffs array argument (7)) ";
						}
						else if( CF->N!= N ){
							Resize_ascanf_Array( CF, N, result );
						}
						if( CF && CF->array ){
							if( !(SavGolHSize== 0 && SavGolSOrder== 0 && SavGolDOrder==0) ){
								savgol( &(SavGolCoeffs)[-1], N, SavGolHSize, SavGolHSize, SavGolDOrder, SavGolSOrder );
								  /* Unwrap the coefficients into the target memory:	*/
								CF->array[N/2]= SavGolCoeffs[0];
								for( i= 1; i<= N/2; i++ ){
									CF->array[N/2-i]= SavGolCoeffs[i];
									CF->array[N/2+i]= SavGolCoeffs[N-i];
								}
							}
							else{
								memset( CF->array, 0, N* sizeof(double) );
							}
							CF->value= CF->array[ (CF->last_index= 0) ];
							if( CF->accessHandler ){
								AccessHandler( CF, "SavGolayInit", level, ASCB_COMPILED, AH_EXPR, NULL  );
							}
						}
					}

					  /* Don't need these anymore:	*/
					xfree( SavGolCoeffs );
					xfree( coeffs );
					xfree( input );
				}
				else{
					fprintf( StdErr, "SavGolayInit: error determining the Savitzky-Golay coefficients (%s)\n", serror() );
					*result= 0;
					free_SGResults( &SavGolResult );
					xfree( SavGolCoeffs );
					xfree( coeffs );
					xfree( input );
					return(0);
				}
			}
		}
	}
	else{
		if( !args || ascanf_arguments< 2 ){
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
	}
	GCA();
	return(1);
}

#define SGCOL_OK(col)	(SavGolResult && (col>=0 && col< SGRcols))

int ascanf_SavGolX ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  savgol_flp idx;
	if( args && ascanf_arguments> 0 && !ascanf_SyntaxCheck && SGCOL_OK(SavGolSet->xcol) ){
		CLIP_EXPR( idx, args[0], 0, SavGolN-1);
		*result= SavGolResult[SavGolSet->xcol].smoothed[(int)idx+SGfw];
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SavGolY ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  savgol_flp idx;
	if( args && ascanf_arguments> 0 && !ascanf_SyntaxCheck && SGCOL_OK(SavGolSet->ycol) ){
		CLIP_EXPR( idx, args[0], 0, SavGolN-1);
		*result= SavGolResult[SavGolSet->ycol].smoothed[(int)idx+SGfw];
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SavGolE ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  savgol_flp idx;
	if( args && ascanf_arguments> 0 && !ascanf_SyntaxCheck && SGCOL_OK(SavGolSet->ecol) ){
		CLIP_EXPR( idx, args[0], 0, SavGolN-1);
		*result= SavGolResult[SavGolSet->ecol].smoothed[(int)idx+SGfw];
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SavGolColNr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  savgol_flp idx;
  int col;
	if( args && ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
		if( args[0]< 0 || args[0]>= SGRcols ){
			ascanf_emsg= "(column out of range)";
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
		col= (int) args[0];
		if( !SGCOL_OK(col) ){
			ascanf_emsg= "(invalid column)";
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
		else{
			CLIP_EXPR( idx, args[1], 0, SavGolN-1);
			*result= SavGolResult[col].smoothed[(int)idx+SGfw];
		}
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SavGolColumn2Array ( ASCB_ARGLIST )
{ ASCB_FRAME
  savgol_flp col, *column;
  int start= 0, end= -1, offset= 0, i, j, idx;
  long N;
  ascanf_Function *targ, *visible= NULL;
	*result= 0;
	if( !args || ascanf_arguments< 2 || !SavGolSet ){
		ascanf_emsg= " (syntax error or no SavGol data) ";
		ascanf_arg_error= 1;
		return(0);
	}
	idx= SavGolSet->set_nr;
	if( !(targ= parse_ascanf_address(args[0], 0, "ascanf_SavGolColumn2Array", (int) ascanf_verbose, NULL )) || targ->type!= _ascanf_array ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( (col= args[1])< 0 || col>= SGRcols ){
		ascanf_emsg= " (column number out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( ascanf_arguments> 2 ){
/* 		CLIP(args[2], 0, SavGolN-1 );	*/
/* 		start= (int) args[2];	*/
		if( !(visible= parse_ascanf_address(args[2], 0, "ascanf_SavGolColumn2Array", (ascanf_verbose)? -1 : 0, NULL ))
			|| visible->type!= _ascanf_array
		){
			CLIP_EXPR( start, (int) args[2], 0, SavGolN-1 );
		}
	}
	if( !visible ){
		if( ascanf_arguments> 3 ){
			CLIP(args[3], -1, SavGolN-1 );
			end= (int) args[3];
		}
		if( end== -1 ){
			end= SavGolN-1;
		}
		if( ascanf_arguments> 4 ){
			CLIP(args[4], 0, MAXINT );
			offset= (int) args[4];
		}
	}
	else{
		offset= 0;
		start= 0;
		end= SavGolN-1;
		if( ascanf_arguments> 3 && pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (<end> and subsequent arguments ignored)== " );
		}
	}
	if( !ascanf_SyntaxCheck && !SGCOL_OK(col) ){
		ascanf_emsg= " (no SavGol data) ";
		ascanf_arg_error= 1;
		return(0);
	}
	column= &(SavGolResult[(int)col].smoothed[SGfw]);
	if( ascanf_SyntaxCheck ){
		return(1);
	}
	if( visible ){
		N= 0;
		if( ActiveWin && ActiveWin!= StubWindow_ptr ){
			for( i= 0; i<= end; i++ ){
				N+= ActiveWin->pointVisible[(int) idx][i];
			}
			if( ActiveWin->numVisible[(int) idx]!= N ){
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (SavGolayColumn2Array[%d]: %d points visible according to numVisible, pointVisible says %d)== ",
						(int) idx, ActiveWin->numVisible[(int) idx], N
					);
				}
			}
		}
		else if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (Warning: no window active, so 0 points are taken to be visible)== " );
		}
		if( N ){
		  signed char *pointVisible= (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin->pointVisible[(int) idx] : NULL;
			Resize_ascanf_Array( visible, N, NULL);
			if( visible->iarray ){
				for( j= 0, i= 0; i<= end && j< N; i++ ){
					if( pointVisible[i] ){
						visible->value= visible->iarray[j++]= i;
					}
				}
			}
			else{
				for( j= 0, i= 0; i<= end && j< N; i++ ){
					if( pointVisible[i] ){
						visible->value= visible->array[j++]= i;
					}
				}
			}
			visible->last_index= N-1;
			if( visible->accessHandler ){
				AccessHandler( visible, "SavGolayColumn2Array", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
		}
	}
	else{
		N= (offset+ end- start+ 1);
	}
	if( !targ->N || targ->N< N ){
		Resize_ascanf_Array( targ, MAX(N,1), NULL );
	}
	if( targ->iarray ){
		if( visible ){
			if( N ){
				if( visible->iarray ){
					start= visible->iarray[0];
					end= visible->iarray[N-1];
					for( i= 0; i< N; i++ ){
						targ->iarray[i]= (int) column[ (int) visible->iarray[i] ];
					}
				}
				else{
					start= (int) visible->array[0];
					end= (int) visible->array[N-1];
					for( i= 0; i< N; i++ ){
						targ->iarray[i]= (int) column[ (int) visible->array[i] ];
					}
				}
				targ->last_index= N-1;
			}
			j= N;
		}
		else{
			for( j= 0, i= start; i<= end; j++, i++ ){
				targ->value= targ->iarray[(targ->last_index=offset+j)]= (int) column[i];
			}
			targ->value= targ->iarray[(targ->last_index=offset+j-1)];
		}
	}
	else{
		if( visible ){
			if( N ){
				if( visible->iarray ){
					start= visible->iarray[0];
					end= visible->iarray[N-1];
					for( i= 0; i< N; i++ ){
						targ->array[i]= column[ (int) visible->iarray[i] ];
					}
				}
				else{
					start= (int) visible->array[0];
					end= (int) visible->array[N-1];
					for( i= 0; i< N; i++ ){
						targ->array[i]= column[ (int) visible->array[i] ];
					}
				}
				targ->last_index= N-1;
			}
			j= N;
		}
		else{
			for( j= 0, i= start; i<= end; j++, i++ ){
				targ->value= targ->array[(targ->last_index=offset+j)]= column[i];
			}
			targ->value= targ->array[(targ->last_index=offset+j-1)];
		}
	}
	if( targ->accessHandler ){
		AccessHandler( targ, "SavGolay2Array", level, ASCB_COMPILED, AH_EXPR, NULL  );
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr,
			" (copied %d(%d-%d) %selements from SavGol[w=%d,o=%d,d=%d] filtered set#%d column %d to %s[%d] (%d elements))== ",
			j, start, end, (visible)? "visible " : "",
			SavGolHSize, SavGolSOrder, SavGolDOrder,
			SavGolSet->set_nr, (int) col, targ->name, offset, targ->N
		);
	}
	*result= j;
	return(1);
}

int ascanf_SavGolFinished ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
	*result= SavGolN;

	free_SGResults( &SavGolResult );
	xfree( SavGolCoeffs );
	SavGolN= 0;
	SavGolSet= NULL;
	return( 1 );
}

/* end savgol functions */

/* (r)fftw functions */
#if defined(HAVE_FFTW) && HAVE_FFTW

#include <fftw3.h>

#else

/* If <complex.h> is included, use the C99 complex type.  Otherwise
   define a type bit-compatible with C99 complex */
#	ifdef _Complex_I
#	  define FFTW_DEFINE_COMPLEX(R, C) typedef R _Complex C
#	else
#	  define FFTW_DEFINE_COMPLEX(R, C) typedef R C[2]
#	endif

   FFTW_DEFINE_COMPLEX(double, fftw_complex);
   FFTW_DEFINE_COMPLEX(float, fftwf_complex);

#endif

int FFTW_Initialised= False;
char *FFTW_wisdom= NULL;
unsigned long wisdom_length= 0;

static double *fftw_planner_level, *fftw_nthreads, *fftw_interrupt_aborts;
static int nthreads_preset= False;

#ifdef FFTW_SINGLE
#	define fftw_complex	fftwf_complex
#	define fftw_real	float
#	define fftw_plan	fftwf_plan
#else
#	define fftw_real	double
#endif

#if defined(FFTW_DYNAMIC)
#	if defined(HAVE_FFTW) && HAVE_FFTW
  /* 20010725: the FFTW libraries should be loaded at runtime, when needed. This is usefull
   \ if one wants to be able to compile xgraph on a machine with these libraries installed, but
   \ (also) run it on machines that don't.
   */

  /* First, define functionpointers for the routines that we (want to) use.
   \ NB: at the moment of implementation, these are all functions, and not macros.
   \ If ever a future fftw edition defines these as macros, this will have to adapt,
   \ of course!!
   */
FNPTR( fftw_export_wisdom_to_file_ptr, void, (FILE *output_file));
FNPTR( fftw_import_wisdom_from_file_ptr, int, (FILE *input_file));
FNPTR( fftw_export_wisdom_to_string_ptr, char*, (void));
FNPTR( fftw_forget_wisdom_ptr, void, (void));
FNPTR( fftw_malloc_ptr, void*, (size_t size));
FNPTR( fftw_free_ptr, void, (void*));

#ifdef FFTW_THREADED
	FNPTR( fftw_init_threads_ptr, int, (void) );
	FNPTR( fftw_cleanup_threads_ptr, void, (void) );
	FNPTR( fftw_plan_with_nthreads_ptr, void, (int nthreads) );
#endif

#ifdef FFTW_SINGLE
	FNPTR( fftw_plan_dft_r2c_1d_ptr, fftw_plan, (int n, fftw_real *in, fftw_complex *out, int flags));
	FNPTR( fftw_plan_dft_r2c_2d_ptr, fftw_plan, (int nx, int ny, fftw_real *in, fftw_complex *out, int flags));
	FNPTR( fftw_plan_dft_c2r_1d_ptr, fftw_plan, (int n, fftw_complex *in, fftw_real *out, int flags));
	FNPTR( fftw_plan_dft_c2r_2d_ptr, fftw_plan, (int nx, int ny, fftw_complex *in, fftw_real *out, int flags));
#else
	FNPTR( fftw_plan_dft_r2c_1d_ptr, fftw_plan, (int n, fftw_real *in, fftw_complex *out, int flags));
	FNPTR( fftw_plan_dft_r2c_2d_ptr, fftw_plan, (int nx, int ny, fftw_real *in, fftw_complex *out, int flags));
	FNPTR( fftw_plan_dft_c2r_1d_ptr, fftw_plan, (int n, fftw_complex *in, fftw_real *out, int flags));
	FNPTR( fftw_plan_dft_c2r_2d_ptr, fftw_plan, (int nx, int ny, fftw_complex *in, fftw_real *out, int flags));
#endif
FNPTR( fftw_destroy_plan_ptr, void, (fftw_plan plan));
FNPTR( fftw_execute_ptr, void, (fftw_plan p));

  /* Now, define a bunch of defines such that we can access these pointers transparently from the
   \ code below: these are the FFTW_DYNAMIC definitions.
   */
#	define XG_fftw_import_wisdom_from_file	(*fftw_import_wisdom_from_file_ptr)
#	define XG_fftw_export_wisdom_to_string	(*fftw_export_wisdom_to_string_ptr)
#	define XG_fftw_export_wisdom_to_file	(*fftw_export_wisdom_to_file_ptr)
#	define XG_fftw_forget_wisdom	(*fftw_forget_wisdom_ptr)
#	define XG_fftw_malloc	(*fftw_malloc_ptr)
#	define XG_fftw_free	(*fftw_free_ptr)
#	define XG_fftw_plan_dft_r2c_1d	(*fftw_plan_dft_r2c_1d_ptr)
#	define XG_fftw_plan_dft_r2c_2d	(*fftw_plan_dft_r2c_2d_ptr)
#	define XG_fftw_plan_dft_c2r_1d	(*fftw_plan_dft_c2r_1d_ptr)
#	define XG_fftw_plan_dft_c2r_2d	(*fftw_plan_dft_c2r_2d_ptr)
#	define XG_fftw_execute	(*fftw_execute_ptr)
#	define XG_fftw_destroy_plan	(*fftw_destroy_plan_ptr)
#ifdef FFTW_THREADED
#	define XG_fftw_init_threads	(*fftw_init_threads_ptr)
#	define XG_fftw_cleanup_threads	(*fftw_cleanup_threads_ptr)
#	define XG_fftw_plan_with_nthreads	(*fftw_plan_with_nthreads_ptr)
#endif

#define fftw_xfree(x)	if(x){ XG_fftw_free((x)); (x)=NULL; }

/* 20050404: incredible... with gcc, the macros below worked *without* the & operator, effectively
 \ casting an fftw_complex to an fftw_real*....
 */
#ifndef c_re
#	define c_re(c)	(((fftw_real*)&(c))[0])
#endif
#ifndef c_im
#	define c_im(c)	(((fftw_real*)&(c))[1])
#endif

  /* We also need a handle to refer to the fftw3 shared library: */
void *lib_fftw3= NULL;
#ifdef FFTW_THREADED
	void *lib_mtfftw3= NULL;
#endif

  /* Include dymod.h. This header contains the stuff needed for dealing with shared libraries. */
#include "dymod.h"

#	else

  /* This is the straightforward, !FFTW_DYNAMIC condition. In this case, we just leave it
   \ to the linker/loader to initialise all those functionpointers we have to initialise
   \ in the DYNAMIC case. This means of course that we have to link with the libraries at
   \ build time. And, if they're shared libraries, that they have to be present when we run
   \ the programme. It will depend on the system whether the application will be runnable
   \ without - as long as no fftw's are attempted, some systems will allow execution.
   */

#	ifdef FFTW_SINGLE
#		define XG_fftw_import_wisdom_from_file	fftwf_import_wisdom_from_file
#		define XG_fftw_export_wisdom_to_string	fftwf_export_wisdom_to_string
#		define XG_fftw_export_wisdom_to_file	fftwf_export_wisdom_to_file
#		define XG_fftw_forget_wisdom	fftwf_forget_wisdom
#		define XG_fftw_malloc	fftwf_malloc
#		define XG_fftw_free	fftwf_free
#		define XG_fftw_plan_dft_r2c_1d	fftwf_plan_dft_r2c_1d
#		define XG_fftw_plan_dft_r2c_2d	fftwf_plan_dft_r2c_2d
#		define XG_fftw_plan_dft_c2r_1d	fftwf_plan_dft_c2r_1d
#		define XG_fftw_plan_dft_c2r_2d	fftwf_plan_dft_c2r_2d
#		define XG_fftw_execute	fftwf_execute
#		define XG_fftw_destroy_plan	fftwf_destroy_plan
#	else
#		define XG_fftw_import_wisdom_from_file	fftw_import_wisdom_from_file
#		define XG_fftw_export_wisdom_to_string	fftw_export_wisdom_to_string
#		define XG_fftw_export_wisdom_to_file	fftw_export_wisdom_to_file
#		define XG_fftw_forget_wisdom	fftw_forget_wisdom
#		define XG_fftw_free	fftw_free
#		define XG_fftw_plan_dft_r2c_1d	fftw_plan_dft_r2c_1d
#		define XG_fftw_plan_dft_r2c_2d	fftw_plan_dft_r2c_2d
#		define XG_fftw_plan_dft_c2r_1d	fftw_plan_dft_c2r_1d
#		define XG_fftw_plan_dft_c2r_2d	fftw_plan_dft_c2r_2d
#		define XG_fftw_execute	fftw_execute
#		define XG_fftw_destroy_plan	fftw_destroy_plan
#	endif
#	endif
#endif

  /* Since I'm not in the mood to manually change all (r)fftw function invocations
   \ for the XG_.... version we defined above, just do the following, reverse
   \ macro definitions... This works in gcc. And it is possible since the first thing
   \ we have to do while initialising FFTW is loading the library and initialising the
   \ pointers, in the FFTW_DYNAMIC case...
   */

#define fftw_import_wisdom_from_file	XG_fftw_import_wisdom_from_file
#define fftw_export_wisdom_to_string	XG_fftw_export_wisdom_to_string
#define fftw_export_wisdom_to_file	XG_fftw_export_wisdom_to_file
#define fftw_forget_wisdom	XG_fftw_forget_wisdom
#define fftw_free	XG_fftw_free
#define ffftw_one	XG_ffftw_one
#define fftw_plan_dft_r2c_1d	XG_fftw_plan_dft_r2c_1d
#define fftw_plan_dft_r2c_2d	XG_fftw_plan_dft_r2c_2d
#define fftw_plan_dft_c2r_1d	XG_fftw_plan_dft_c2r_1d
#define fftw_plan_dft_c2r_2d	XG_fftw_plan_dft_c2r_2d
#define fftw_execute	XG_fftw_execute
#define fftw_destroy_plan	XG_fftw_destroy_plan
#ifdef FFTW_THREADED
#	define fftw_init_threads	XG_fftw_init_threads
#	define fftw_cleanup_threads	XG_fftw_cleanup_threads
#	define fftw_plan_with_nthreads	XG_fftw_plan_with_nthreads
#else
#	define fftw_init_threads()	/**/
#	define fftw_cleanup_threads()	/**/
#	define fftw_plan_with_nthreads(n)	/**/
#endif

#if !defined(HAVE_FFTW) || HAVE_FFTW==0
#	undef fftw_malloc
#	undef fftw_free
#	define fftw_malloc	malloc
#	define fftw_free	free
#	define XG_fftw_malloc	fftw_malloc
#	define XG_fftw_free	fftw_free
#	define fftw_xfree(x)	if(x){ XG_fftw_free((x)); (x)=NULL; }
#elif defined(FFTW_SINGLE) && defined(DEBUG) && defined(NO_FFTW_ALLOC)
#	undef XG_fftw_free
#	undef XG_fftw_malloc
#	undef fftw_free
#	undef fftw_malloc
#	define XG_fftw_free	free
#	define XG_fftw_malloc	malloc
#	define fftw_free	free
#	define fftw_malloc	malloc
#endif

#ifdef FFTW_THREADED
int nthreads_callback ( ASCB_ARGLIST )
{
	if( FFTW_Initialised ){
		  /* Constrain the number of threads: shouldn't be less than 1 in any case. Upper limit.... what can I say?! */
		CLIP( *fftw_nthreads, 1, 999 );
		fftw_plan_with_nthreads( (int) *fftw_nthreads );
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(using %d threads for FFTW from now on) ", (int) *fftw_nthreads );
		}
	}
	else{
		nthreads_preset= True;
	}
}
#endif

static char wise_file[512];

#if defined(__APPLE_CC__) || defined(__MACH__)
#	include <mach/mach.h>
#	include <mach/mach_host.h>
#	include <mach/host_info.h>
#	include <mach/machine.h>
#endif

int ascanf_InitFFTW ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  char *c;
  FILE *fp;
  char *w;
#define LNAMELEN	1024
#ifdef __CYGWIN__

#	ifdef FFTW_SINGLE
  char libname[LNAMELEN]= "cygfftw3f";
#		ifdef FFTW_THREADED
  char mtlibname[LNAMELEN]= "cygfftw3f_threads";
#		endif
#	else
  char libname[LNAMELEN]= "cygfftw3";
#		ifdef FFTW_THREADED
  char mtlibname[LNAMELEN]= "cygfftw3_threads";
#		endif
#	endif

#else

#	ifdef FFTW_SINGLE
  char libname[LNAMELEN]= "libfftw3f";
#		ifdef FFTW_THREADED
  char mtlibname[LNAMELEN]= "libfftw3f_threads";
#		endif
#	else
  char libname[LNAMELEN]= "libfftw3";
#		ifdef FFTW_THREADED
  char mtlibname[LNAMELEN]= "libfftw3_threads";
#		endif
#	endif

#endif
#if defined(__APPLE_CC__) || defined(__MACH__)
  char *libext= ".dylib";
#elif defined(__CYGWIN__)
  char *libext= "-3.dll";
#else
  char *libext= ".so";
#endif
  static char called= 0;


	if( FFTW_Initialised || ascanf_SyntaxCheck ){
		*result= 1;
		return(1);
	}
#if defined(__APPLE_CC__) || defined(__MACH__)
	{ host_basic_info_data_t hostInfo;
	  mach_msg_type_number_t infoCount;

		infoCount = HOST_BASIC_INFO_COUNT;
		host_info( mach_host_self(), HOST_BASIC_INFO, (host_info_t)&hostInfo, &infoCount );
		if( infoCount ){
			if( !nthreads_preset && !called ){
				  /* 20050202: no real interest in doing multithreading. Planning is really slower. */
				*fftw_nthreads= 1 /* hostInfo.max_cpus */;
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (%d of %d cpus, using %g threads)",
					hostInfo.avail_cpus, hostInfo.max_cpus, *fftw_nthreads );
			}
		}
	}
#endif

#if defined(HAVE_FFTW) && defined(FFTW_DYNAMIC) && HAVE_FFTW

	{ char *ln;
#ifdef FFTW_THREADED
	  char *mtln;
#	ifdef FFTW_SINGLE
		mtln = GetEnv("XG_LIBFFTW3F_THREADS");
#	else
		mtln = GetEnv("XG_LIBFFTW3_THREADS");
#	endif
		if( mtln ){
			strncpy( mtlibname, mtln, LNAMELEN-1 );
		}
		else{
	strcat( mtlibname, libext );
		}
#endif
#ifdef FFTW_SINGLE
		ln = GetEnv("XG_LIBFFTW3F");
#else
		ln = GetEnv("XG_LIBFFTW3");
#endif
		if( ln ){
			strncpy( libname, ln, LNAMELEN-1 );
		}
		else{
			strcat( libname, libext );
		}
	}

	  /* load fftw here, and initialise function pointers on success 	*/
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, " (loading %s... ", libname );
#ifdef FFTW_THREADED
		fprintf( StdErr, "and %s... ", mtlibname );
#endif
		fflush( StdErr );
	}

#ifdef USE_LTDL
	lib_fftw3= lt_dlopen( libname );
#	ifdef FFTW_THREADED
	lib_mtfftw3= lt_dlopen( mtlibname );
#	endif
#else
	lib_fftw3= dlopen( libname, RTLD_NOW|RTLD_GLOBAL);
#	ifdef FFTW_THREADED
	lib_mtfftw3= dlopen( mtlibname, RTLD_NOW|RTLD_GLOBAL );
#	endif
#endif

	if( lib_fftw3
#ifdef FFTW_THREADED
		&& lib_mtfftw3
#endif
	){
	  int err= 0;
#ifdef FFTW_SINGLE
		LOADFUNCTION( lib_fftw3, fftw_import_wisdom_from_file_ptr, "fftwf_import_wisdom_from_file" );
		LOADFUNCTION( lib_fftw3, fftw_export_wisdom_to_string_ptr, "fftwf_export_wisdom_to_string" );
		LOADFUNCTION( lib_fftw3, fftw_export_wisdom_to_file_ptr, "fftwf_export_wisdom_to_file" );
		LOADFUNCTION( lib_fftw3, fftw_forget_wisdom_ptr, "fftwf_forget_wisdom" );
		LOADFUNCTION( lib_fftw3, fftw_malloc_ptr, "fftwf_malloc" );
		LOADFUNCTION( lib_fftw3, fftw_free_ptr, "fftwf_free" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_r2c_1d_ptr, "fftwf_plan_dft_r2c_1d" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_r2c_2d_ptr, "fftwf_plan_dft_r2c_2d" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_c2r_1d_ptr, "fftwf_plan_dft_c2r_1d" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_c2r_2d_ptr, "fftwf_plan_dft_c2r_2d" );
		LOADFUNCTION( lib_fftw3, fftw_execute_ptr, "fftwf_execute" );
		LOADFUNCTION( lib_fftw3, fftw_destroy_plan_ptr, "fftwf_destroy_plan" );
#ifdef FFTW_THREADED
		LOADFUNCTION( lib_mtfftw3, fftw_init_threads_ptr, "fftwf_init_threads" );
		LOADFUNCTION( lib_mtfftw3, fftw_cleanup_threads_ptr, "fftwf_cleanup_threads" );
		LOADFUNCTION( lib_mtfftw3, fftw_plan_with_nthreads_ptr, "fftwf_plan_with_nthreads" );
#endif
#else
		LOADFUNCTION( lib_fftw3, fftw_import_wisdom_from_file_ptr, "fftw_import_wisdom_from_file" );
		LOADFUNCTION( lib_fftw3, fftw_export_wisdom_to_string_ptr, "fftw_export_wisdom_to_string" );
		LOADFUNCTION( lib_fftw3, fftw_export_wisdom_to_file_ptr, "fftw_export_wisdom_to_file" );
		LOADFUNCTION( lib_fftw3, fftw_forget_wisdom_ptr, "fftw_forget_wisdom" );
		LOADFUNCTION( lib_fftw3, fftw_malloc_ptr, "fftw_malloc" );
		LOADFUNCTION( lib_fftw3, fftw_free_ptr, "fftw_free" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_r2c_1d_ptr, "fftw_plan_dft_r2c_1d" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_r2c_2d_ptr, "fftw_plan_dft_r2c_2d" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_c2r_1d_ptr, "fftw_plan_dft_c2r_1d" );
		LOADFUNCTION( lib_fftw3, fftw_plan_dft_c2r_2d_ptr, "fftw_plan_dft_c2r_2d" );
		LOADFUNCTION( lib_fftw3, fftw_execute_ptr, "fftw_execute" );
		LOADFUNCTION( lib_fftw3, fftw_destroy_plan_ptr, "fftw_destroy_plan" );
#ifdef FFTW_THREADED
		LOADFUNCTION( lib_mtfftw3, fftw_init_threads_ptr, "fftw_init_threads" );
		LOADFUNCTION( lib_mtfftw3, fftw_cleanup_threads_ptr, "fftw_cleanup_threads" );
		LOADFUNCTION( lib_mtfftw3, fftw_plan_with_nthreads_ptr, "fftw_plan_with_nthreads" );
#endif
#endif
		if( err ){
			if( lib_fftw3 ){
				dlclose( lib_fftw3 );
			}
#ifdef FFTW_THREADED
			if( lib_mtfftw3 ){
				dlclose( lib_mtfftw3 );
			}
#endif
			fftw_import_wisdom_from_file_ptr= NULL;
			fftw_export_wisdom_to_string_ptr= NULL;
			fftw_export_wisdom_to_file_ptr= NULL;
			fftw_forget_wisdom_ptr= NULL;
			fftw_plan_dft_r2c_1d_ptr= NULL;
			fftw_plan_dft_r2c_2d_ptr= NULL;
			fftw_plan_dft_c2r_1d_ptr= NULL;
			fftw_plan_dft_c2r_2d_ptr= NULL;
			fftw_execute_ptr= NULL;
			fftw_destroy_plan_ptr= NULL;
			*result= 0;
			return(0);
		}
	}
	else{
#ifdef USE_LTDL
		fprintf( StdErr, "Error: can't load %s (%s): no support for FFT!\n", libname, lt_dlerror() );
#else
		fprintf( StdErr, "Error: can't load %s (%s): no support for FFT!\n", libname, dlerror() );
#endif
		*result= 0;
		return(0);
	}

#ifdef FFTW_THREADED
	fftw_init_threads();
	CLIP( *fftw_nthreads, 1, 999 );
	fftw_plan_with_nthreads( (int) *fftw_nthreads );
#endif

	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, "done) " );
		fflush( StdErr );
	}
#endif // FFTW_DYNAMIC

	wise_file[511]= '\0';
	if( (c= getenv( "XG_WISDOM")) ){
		strncpy( wise_file, c, 511 );
	}
	else{
#ifdef FFTW_SINGLE
		sprintf( wise_file, "%s/fft3f_wisdom", PrefsDir );
#else
		sprintf( wise_file, "%s/fft3_wisdom", PrefsDir );
#endif
	}
#ifdef FFTW_THREADED
	strcat( wise_file, "_threaded" );
#endif
	if( (fp= fopen( wise_file, "r" )) ){
#if defined(__APPLE_CC__) || defined(__MACH__)
		errno= 0;
		flockfile(fp);
#endif
#if defined(HAVE_FFTW) && HAVE_FFTW
		if( !fftw_import_wisdom_from_file(fp) ){
			fprintf( StdErr, "ascanf_InitRFFTW(): read error on wisdom file %s: %s\n", wise_file, serror() );
			*result= 0;
			fclose(fp);
			return(0);
		}
#else
		fprintf( StdErr, "ascanf_InitRFFTW(): you need to compile with -DHAVE_FFTW; fftw routines will not compute\n" );
#endif
#if defined(__APPLE_CC__) || defined(__MACH__)
		funlockfile(fp);
#endif
		fclose(fp);
#if defined(HAVE_FFTW) && HAVE_FFTW
		if( FFTW_wisdom ){
			fftw_free( FFTW_wisdom );
		}
		w= fftw_export_wisdom_to_string();
		FFTW_wisdom= w;
		wisdom_length= strlen(w);
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(imported wisdom from %s)== ", wise_file );
		}
#endif
	}
	else if( /* c || */ errno!= ENOENT ){
		fprintf( StdErr, "ascanf_InitRFFTW(): read error on wisdom file %s: %s\n", wise_file, serror() );
		*result= 0;
		return(0);
	}
	*result= 1;
	FFTW_Initialised= True;
	called= 1;
	return(1);
}

int ascanf_CloseFFTW ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  FILE *fp;
  char *w= NULL;
  int complete;
  unsigned long wlen;

	if( !FFTW_Initialised || ascanf_SyntaxCheck ){
		*result= 0;
		return(1);
	}
#if defined(HAVE_FFTW) && HAVE_FFTW
	if( ascanf_arguments && args[0] ){
		complete= 1;
	}
	else{
		complete= 0;
	}
	*result= 1;
	if( (w= fftw_export_wisdom_to_string()) && (wlen= strlen(w)) && FFTW_wisdom ){
/* 		if( !strcmp( w, FFTW_wisdom) )	*/
		if( wlen == wisdom_length )
		{
			fftw_free(w);
			w= NULL;
		}
	}
	if( w ){
		if( (fp= fopen( wise_file, "w" )) ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "(stored increased (L%d->L%d) wisdom in %s)== ",
					wisdom_length, wlen,
					wise_file );
			}
#if defined(__APPLE_CC__) || defined(__MACH__)
			errno= 0;
			flockfile(fp);
#endif
			fftw_export_wisdom_to_file(fp);
#if defined(__APPLE_CC__) || defined(__MACH__)
			funlockfile(fp);
#endif
			if( complete ){
				fflush(fp);
				fsync(fileno(fp));
			}
			fclose(fp);
		}
		else{
			fprintf( StdErr, "ascanf_CloseRFFTW(): write error on wisdom file %s: %s\n", wise_file, serror() );
			*result= 0;
			return(0);
		}
		if( !complete ){
			fftw_free(FFTW_wisdom);
			FFTW_wisdom= w;
			wisdom_length= wlen;
		}
	}
	if( complete ){
		fftw_free(FFTW_wisdom);
		FFTW_wisdom= NULL;
#ifndef __CYGWIN__
		// crashes?!
		fftw_forget_wisdom();
		fftw_cleanup_threads();
#endif
		FFTW_Initialised= False;
	}
#else
	fprintf( StdErr, "ascanf_CloseRFFTW(): no FFTW, no wisdom to store...\n" );
	*result= 0;
#endif
	return(1);
}

#if defined(HAVE_FFTW) && HAVE_FFTW
#if defined(DEBUG) && defined(sgi)
#	include <fft.h>
#endif
#endif

#define USE_RFFTWND

static unsigned long nflops= 0;

#ifdef FFTW_SINGLE
unsigned long fourconv3f_nan_handling( fftw_real *data, size_t NN, int nan_handling )
{ unsigned long nNaN= 0;
	if( nan_handling ){
	  size_t i, j, sL= 0, eL, sR= NN, eR;
	  /* sL: 0 or index to previous, non-NaN element before this block
	   \ eL: NN or index of first non-NaN element of a 'block'
	   \ sR: NN or index of last non-NaN element in a block.
	   \ eR: NN or index of next, non-NaN element after this block.
	   */
		for( i= 0; i< NN; ){
			if( isnanf(data[i]) ){
			  /* retrieve previous, non-NaN element: */
				sL= (i>0)? i-1 : 0;
				  /* Find next, non-NaN element: */
				j= i+1;
				while( j< NN && isnanf(data[j]) ){
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
				while( j< NN && !isnanf(data[j]) ){
					j++;
				}
				if( j< NN && isnanf(data[j]) ){
					sR= (j>0)? j-1 : 0; /* MAX(j-1,0); */
					j+= 1;
					while( j< NN && isnanf(data[j]) ){
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
					if( !isnanf(data[sL]) ){
						switch( nan_handling ){
							case 2:{
							    double slope= (data[eL] - data[sL]) / (eL - sL);
								  /* we have a non-NaN preceding value: fill the gap of NaN(s) with a linear
								   \ gradient.
								   */
#ifdef DEBUG
								fprintf( StdErr, "'left' NaN hole from %lu-%lu; filling with gradient {", sL+1, eL-1 );
#endif
								nflops+= 2;
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= data[sL] + slope* (j-sL);
									nflops+= 3;
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
								nflops+= 3;
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= (j<halfway)? data[sL] : data[eL];
									nflops+= 1;
									nNaN++;
								}
								break;
							}
						}
					}
					else{
						  /* Best guess we can do is to fill the gap with the first non-NaN value we have at hand */
#ifdef DEBUG
						fprintf( StdErr, "'left' NaN hole from %lu-%lu; padding with %g (%d)\n", sL+0, eL-1, data[eL], __LINE__ );
#endif
						if( sL==0 && eL==1 ){
							data[0]= data[eL];
							nflops+= 1;
							nNaN++;
						}
						else{
							for( j= sL+0; j< eL && j< NN; j++ ){
								data[j]= data[eL];
								nflops+= 1;
								nNaN++;
							}
						}
					}
				}
				if( sR != eR ){
					if( eR< NN && !isnanf(data[eR]) ){
						switch( nan_handling ){
							case 2:{
							  double slope= (data[eR] - data[sR]) / (eR - sR);
#ifdef DEBUG
								fprintf( StdErr, "'right' NaN hole from %lu-%lu; filling with gradient {", sR+1, eR-1 );
#endif
								nflops+= 3;
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= data[sR] + slope* (j-sR);
									nflops+= 3;
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
								nflops+= 3;
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= (j<halfway)? data[sR] : data[eR];
									nflops+= 1;
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
							nflops+= 1;
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

#endif

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
								nflops+= 3;
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= data[sL] + slope* (j-sL);
									nflops+= 3;
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
								nflops+= 2;
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= (j<halfway)? data[sL] : data[eL];
									nflops+= 1;
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
							nflops+= 1;
							nNaN++;
						}
						else{
							for( j= sL+0; j< eL && j< NN; j++ ){
								data[j]= data[eL];
								nflops+= 1;
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
								nflops+= 3;
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= data[sR] + slope* (j-sR);
									nflops+= 3;
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
								nflops+= 3;
								for( j= sR+1; j< eR && j< NN; j++ ){
									data[j]= (j<halfway)? data[sR] : data[eR];
									nflops+= 1;
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
							nflops+= 1;
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

static unsigned long last_data_size= 0;

Time_Struct plan_timer, exec_timer, power_timer;

static unsigned long nanHandled;

#include <signal.h>
#include <setjmp.h>

sigjmp_buf fc_jmp;
void fcsig_h( int sig )
{
	if( *fftw_interrupt_aborts ){
		ascanf_interrupt= True;
		ascanf_escape= True;
	}
	siglongjmp( fc_jmp, 1 );
}

#if defined(HAVE_FFTW) && HAVE_FFTW

fftw_real *do_fftw( unsigned long N, fftw_real *input, fftw_real *output, fftw_real *power, int nan_handling )
{ fftw_real *in= input;
  fftw_complex *out= (fftw_complex*) output;
  fftw_plan p= NULL;
  unsigned long i;
  int plevel;
  fftw_real *data;
  sig_t prevINT;

	if( !FFTW_Initialised ){
		Elapsed_Since(&power_timer, False);
		return(NULL);
	}

	data= (fftw_real*) XG_fftw_malloc(N* sizeof(fftw_real));
	if( !data ){
		fprintf( StdErr, "\n# do_fftw() couldn't get %d value fftw_real buffer: aborting operation (%s)\n",
			N, serror()
		);
		Elapsed_Since(&power_timer, False);
		return(NULL);
	}

	last_data_size= N;

	prevINT= signal( SIGINT, fcsig_h );

	if( sigsetjmp(fc_jmp,1)== 0 ){
		CLIP( *fftw_planner_level, -1, 2 );
		switch( (int) *fftw_planner_level ){
			case -1:
				plevel= FFTW_ESTIMATE;
				break;
			case 0:
				plevel= FFTW_MEASURE;
				break;
			case 1:
				plevel= FFTW_PATIENT;
				break;
			case 2:
				plevel= FFTW_EXHAUSTIVE;
				break;
		}

		  /* plan_timer is expected to be initialised here */
		if( FFTW_Initialised ){
			p= fftw_plan_dft_r2c_1d( N, data, out, plevel );
		}
		else{
#ifdef FFTW_DYNAMIC
			ascanf_emsg= " (fftw not initialised) ";
			ascanf_arg_error= 1;
			signal(SIGINT, prevINT );
			return(0);
#endif
			p= fftw_plan_dft_r2c_1d( N, data, out, FFTW_ESTIMATE );
		}
		Elapsed_Since(&plan_timer, False);

		Elapsed_Since(&exec_timer, True);

		memcpy( data, in, N* sizeof(fftw_real) );
#ifdef FFTW_SINGLE
		nanHandled= fourconv3f_nan_handling( data, N, nan_handling );
#else
		nanHandled= fourconv3_nan_handling( data, N, nan_handling );
#	ifdef DEBUG
		if( nan_handling ){
			for( i= 0; i< N; i++ ){
				if( NaNorInf(data[i]) ){
					fprintf( StdErr, "#!! data[%d] == %s\n", i, ad2str( data[i], NULL, NULL ) );
				}
			}
			fprintf( StdErr, "}\n" );
		}
#	endif
#endif
		fftw_execute( p );
		Elapsed_Since(&exec_timer, False);
		Elapsed_Since(&power_timer, True);
		if( power ){
		  double scale= (double)N * (double)N;
			for( i= 0; i<= N/ 2; i++ ){
				power[i]= (c_re(out[i])*c_re(out[i]) + c_im(out[i])*c_im(out[i]))/ scale;
			}
		}
		Elapsed_Since(&power_timer, False);
	}
	else{
		fprintf( StdErr, "(fftw operation aborted by SIGINT) " ); fflush(StdErr);
		signal(SIGINT, prevINT );
		output= NULL;
	}

	if( p ){
		fftw_destroy_plan(p);
	}

	fftw_xfree( data );

	signal(SIGINT, prevINT );

	return( output );
}
#endif

int ascanf_fftw ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *input, *output, *power;
#if defined(FFTW_SINGLE)
  fftw_real *finput= NULL, *fpower= NULL;
  fftw_complex *foutput= NULL;
#endif
  int nan_handling;
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	else{
	  unsigned long N= 0, NN= 0;
	  static ascanf_Function Output= { "$fftw-static-output", NULL, 0, _ascanf_array, NULL };
		if( !(input= parse_ascanf_address(args[0], _ascanf_array, "ascanf_fftw", (int) ascanf_verbose, NULL )) || input->iarray ){
			ascanf_emsg= " (invalid input array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
		  ascanf_Function *indata= (input->sourceArray)? input->sourceArray : input;
			N= input->N;
			if( indata->malloc!= XG_fftw_malloc || indata->free!= XG_fftw_free ){
			  int start, end;
				if( indata!= input ){
					start= input->array - indata->array;
					end= start+ input->N- 1;
				}
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= (void*) XG_fftw_malloc;
				ascanf_array_free= (void*) XG_fftw_free;
				Resize_ascanf_Array( indata, indata->N, result );
				if( indata!= input ){
					input->array= &indata->array[start];
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (updated subset %s[%d..%d])== ", indata->name, start, end );
					}
				}
			}
			NN= 2* (N/ 2+ 1);
		}
		if( ASCANF_TRUE(args[1]) ){
			if( !(output= parse_ascanf_address(args[1], _ascanf_array, "ascanf_fftw", (int) ascanf_verbose, NULL )) || output->iarray ){
				ascanf_emsg= " (invalid output array argument (2)) ";
				ascanf_arg_error= 1;
			}
		}
		else{
			output= &Output;
			output->N= 1;
			output->function= ascanf_Variable;
			output->array= NULL;
			output->iarray= NULL;
			output->sourceArray= NULL;
			output->usage= NULL;
			  /* Dangerous: reset to NULL as soon as Resize_ascanf_Array has been called! */
			output->car= output->cdr= output;
		}
		if( output && !ascanf_arg_error
			&& (output->N!= NN || output->malloc!= XG_fftw_malloc || output->free!= XG_fftw_free)
		){
			Resize_ascanf_Array_force= True;
			ascanf_array_malloc= (void*) XG_fftw_malloc;
			ascanf_array_free= (void*) XG_fftw_free;
			  /* The output array is an array of fftw_complex, i.e. N/2+1 pairs of doubles. Since fftw_malloc
			   \ is going to be used, allocating a double array or allocating an fftw_complex array should
			   \ not make any difference.
			   */
			Resize_ascanf_Array( output, NN, result );
			if( output== &Output ){
				output->car= output->cdr= NULL;
			}
		}
		if( ascanf_arguments> 2 && ASCANF_TRUE(args[2]) ){
			if( !(power= parse_ascanf_address(args[2], _ascanf_array, "ascanf_fftw", (int) ascanf_verbose, NULL )) || power->iarray ){
				ascanf_emsg= " (invalid powerspectrum array argument (3)) ";
				ascanf_arg_error= 1;
			}
			else if( power->N!= NN ){
			  /* We only need N/2+1, but we take N for the convenience of the user	*/
				if( Resize_ascanf_Array( power, NN, result ) ){
					memset( power->array, 0, sizeof(double)* NN );
				}
			}
		}
		else{
			power= NULL;
		}
		if( ascanf_arguments> 3 && ASCANF_TRUE(args[3]) ){
			nan_handling= (int) args[3];
		}
		else{
			nan_handling= False;
		}
		if( ascanf_arg_error || !N || !output->array || (power && !power->array)
			|| ascanf_SyntaxCheck
		){
			return(0);
		}

#if defined(HAVE_FFTW) && HAVE_FFTW

#ifdef DEBUG
		fprintf( StdErr, "(real data length %d, so complex data will be %d fftw_real values) ", N, NN );
		fflush(StdErr);
#endif

		nflops= 0;
		Elapsed_Since(&plan_timer, True);
#if defined(FFTW_SINGLE)
		fpower= NULL;
		if( FFTW_Initialised ){
			if( !(
				(finput= (fftw_real*) XG_fftw_malloc(input->N* sizeof(fftw_real)))
				&& (foutput= (fftw_complex*) XG_fftw_malloc(output->N* sizeof(fftw_complex)/2))
				&& (!power || (fpower= (fftw_real*) XG_fftw_malloc(power->N* sizeof(fftw_real))))
			) ){
				fftw_xfree(finput);
				fftw_xfree(foutput);
				fftw_xfree(fpower);
				ascanf_emsg= " (intermediary buffer fftw_malloc allocation failure)== ";
				ascanf_arg_error= 1;
				goto ascanf_fftw_return;
			}
		}
		else{
			if( !(
				(finput= (fftw_real*) malloc(input->N* sizeof(fftw_real)))
				&& (foutput= (fftw_complex*) malloc(output->N* sizeof(fftw_complex)/2))
				&& (!power || (fpower= (fftw_real*) malloc(power->N* sizeof(fftw_real))))
			) ){
				xfree(finput);
				xfree(foutput);
				xfree(fpower);
				ascanf_emsg= " (intermediary buffer allocation failure)== ";
				ascanf_arg_error= 1;
				goto ascanf_fftw_return;
			}
		}
		{
		  unsigned long i;
			for( i= 0; i< input->N; i++ ){
				finput[i]= input->array[i];
			}
			do_fftw( N, finput, (fftw_real*) foutput, fpower, nan_handling );
			for( i= 0; i< output->N; i++ ){
				output->array[i]= ((fftw_real*)foutput)[i];
			}
			if( power ){
				for( i= 0; i< power->N; i++ ){
					power->array[i]= fpower[i];
				}
			}
			if( FFTW_Initialised ){
				fftw_xfree(finput);
				fftw_xfree(foutput);
				fftw_xfree(fpower);
			}
			else{
				xfree(finput);
				xfree(foutput);
				xfree(fpower);
			}
		}
#else
		do_fftw( N, input->array, output->array, (power)? power->array : NULL, nan_handling );
#endif
		output->value= output->array[ (input->last_index= 0) ];
		if( output->accessHandler ){
			AccessHandler( output, "rfftw", level, ASCB_COMPILED, AH_EXPR, NULL );
		}

		  /* phase could also be calculated: phase= arctan( imag/real )	*/
		if( power ){
			power->last_index= 0;
			power->value= power->array[0];
			if( power->accessHandler ){
				AccessHandler( power, "rfftw", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}

#endif

		  /* the approximate number of flops in the plan's execution: 	*/
	     ascanf_elapsed_values[3]= 2.5 * N * (log((double) N) / log(2.0));
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(%g ops, %gs", ascanf_elapsed_values[3], exec_timer.HRTot_T );
		}
		  /* estimate the number of flops including the planning, assuming similar execution speed: */
		{ double flops= ascanf_elapsed_values[3]/exec_timer.HRTot_T;
			ascanf_elapsed_values[3]= flops* (plan_timer.HRTot_T + exec_timer.HRTot_T);
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "; %gs planning => %g ops", plan_timer.HRTot_T, ascanf_elapsed_values[3] );
			}
		}
		if( power ){
			nflops+= (N/2+1)* 3;
		}
		ascanf_elapsed_values[3]+= nflops;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "; %lu other ops => %g total ops) ", nflops, ascanf_elapsed_values[3] );
		}

		  /* 20050806: there is little interest in returning NN as before: this can be obtained from the result array(s).
		   \ However, it is of interest to know how many "non NaN values" there were in the input array:
		   */
		*result= N - nanHandled;
#if defined(FFTW_SINGLE)
ascanf_fftw_return:;
#endif
		if( output== &Output ){
			xfree(output->array);
			output->N= 0;
		}
		return(!ascanf_arg_error);
	}
}

int ascanf_inv_fftw ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *input, *output;
#ifdef FFTW_SINGLE
  fftw_real *foutput= NULL;
#endif
  int i;
#if defined(HAVE_FFTW) && HAVE_FFTW
  fftw_plan p= NULL;
#endif
  sig_t prevINT;

	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0, NN;
		if( !(input= parse_ascanf_address(args[0], _ascanf_array, "ascanf_inv_fftw", (int) ascanf_verbose, NULL )) || input->iarray ){
			ascanf_emsg= " (invalid input array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= input->N;
			NN= N/ 2;
			if( NN % 2 == 0 ){
				NN= N- 1;
			}
			else{
				NN= (NN-1) * 2;
			}
			  /* the above estimate of the number of elements in the original array is just wrong:
			   \ 2(N/2+1) is not a unique mapping.....
			   */
			NN= 0;
		}
		if( !(output= parse_ascanf_address(args[1], _ascanf_array, "ascanf_inv_fftw", (int) ascanf_verbose, NULL )) || output->iarray ){
			ascanf_emsg= " (invalid output array argument (2)) ";
			ascanf_arg_error= 1;
		}
		if( input && output ){
			if( 2*(output->N/2+1)!= N ){
				if( 2*(last_data_size/2+1)== N ){
					  /* 20050203: warn the user that he ought to have passed a properly sized output array. However,
					   \ the error we can make with our assumption to take the last known value is 1 (one) sample only.
					   \ This should be irrelevant if the number of samples is large enough.
					   */
					if( pragma_unlikely(ascanf_verbose) || last_data_size< 1000 ){
						fprintf( StdErr,
							"\n### inv_rfftw: output array size %d does not match size of complex input array size %d;\n"
							  "###           using matching last known data size %d!!\n",
							output->N, input->N, last_data_size
						);
					}
					NN= last_data_size;
				}
				else{
					ascanf_emsg= " (can't determine required size of the real, re-transformed data) ";
					ascanf_arg_error= 1;
				}
			}
			else{
				NN= output->N;
			}
			if( output->N!= NN || output->malloc!= XG_fftw_malloc || output->free!= XG_fftw_free ){
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= (void*) XG_fftw_malloc;
				ascanf_array_free= (void*) XG_fftw_free;
				Resize_ascanf_Array( output, N, result );
				if( output->array ){
					memset( output->array, 0, sizeof(double)* N );
				}
			}
		}
		if( ascanf_arg_error || !N || !NN || !output->array || ascanf_SyntaxCheck ){
			return(0);
		}
		else{
		  fftw_complex *data= NULL;
		  int plevel;
		  size_t N2= N/2 + 1;

#ifdef DEBUG
			fprintf( StdErr, "(complex data length %d, so real data was/will be %d) ", N, NN );
			fflush(StdErr);
#endif
			nflops= 0;
			Elapsed_Since(&plan_timer, True);
			if( FFTW_Initialised ){
				if(
#ifdef FFTW_SINGLE
					!(data= (fftw_complex*) XG_fftw_malloc( N2* sizeof(fftw_complex) ))
					  /* 20050201: eureka (and shame on me): the long-standing bug causing a crash when
					   \ using single-precision inv.fftw was ... because ... I checked for !(data=...) && !(foutput=...).
					   \ Meaning that when data was allocated correctly, foutput never was.... You can shoot me now.
					   */
					|| !(foutput= (fftw_real*) XG_fftw_malloc( output->N* sizeof(fftw_real)))
#else
					!(data= (fftw_complex*) XG_fftw_malloc( N2* sizeof(fftw_complex) ))
#endif
				){
					fftw_xfree(data);
#ifdef FFTW_SINGLE
					fftw_xfree(foutput);
#endif
					ascanf_emsg= " (fftw_malloc allocation error) ";
					ascanf_arg_error= 1;
					return(0);
				}
			}
			else{
				if( !(data= (fftw_complex*) calloc(N2, sizeof(fftw_complex)))
#ifdef FFTW_SINGLE
					|| !(foutput= malloc( output->N* sizeof(fftw_real)))
#endif
				){
					xfree(data);
#ifdef FFTW_SINGLE
					xfree(foutput);
#endif
					ascanf_emsg= " (malloc allocation error) ";
					ascanf_arg_error= 1;
					return(0);
				}
			}

			prevINT= signal( SIGINT, fcsig_h );

			if( sigsetjmp(fc_jmp,1)== 0 ){

				CLIP( *fftw_planner_level, -1, 2 );
#if defined(HAVE_FFTW) && HAVE_FFTW
				switch( (int) *fftw_planner_level ){
					case -1:
						plevel= FFTW_ESTIMATE;
						break;
					case 0:
						plevel= FFTW_MEASURE;
						break;
					case 1:
						plevel= FFTW_PATIENT;
						break;
					case 2:
						plevel= FFTW_EXHAUSTIVE;
						break;
				}

				if( FFTW_Initialised ){
#ifdef FFTW_SINGLE
					p= fftw_plan_dft_c2r_1d( NN, data, foutput, plevel );
#else
					p= fftw_plan_dft_c2r_1d( NN, data, output->array, plevel );
#endif
				}
				else{
#ifdef FFTW_DYNAMIC
					ascanf_emsg= " (fftw not initialised) ";
					ascanf_arg_error= 1;
					xfree(data);
#ifdef FFTW_SINGLE
					xfree(foutput);
#endif
					signal(SIGINT, prevINT);
					return(0);
#else
#ifdef FFTW_SINGLE
					p= fftw_plan_dft_c2r_1d( NN, data, foutput, FFTW_ESTIMATE );
#else
					p= fftw_plan_dft_c2r_1d( NN, data, output->array, FFTW_ESTIMATE );
#endif
#endif
				}
				Elapsed_Since(&plan_timer, False);

				  /* We have to make a local copy of the inputdata if we want to
				   \ preserve input->array: the reverse RFFTW transform overwrites
				   \ its input data.
				   */
				Elapsed_Since(&exec_timer, True);
				if( ascanf_arguments> 2 && args[2] ){
				  double scale= 1.0/ NN;
				  fftw_real *ddata= (fftw_real*) data;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (normalising input data) " ); fflush(StdErr);
					}
					for( i= 0; i< N; i++ ){
						ddata[i]= input->array[i]* scale;
						nflops+= 2;
					}
				}
				else{
#ifdef FFTW_SINGLE
				  fftw_real *ddata= (fftw_real*) data;
					for( i= 0; i< N; i++ ){
						ddata[i]= input->array[i];
						nflops+= 1;
					}
#else
					memcpy( data, input->array, N* sizeof(fftw_real) );
#endif
				}

				fftw_execute( p );

				if( N!= NN ){
					Resize_ascanf_Array( output, NN, result );
				}

#ifdef FFTW_SINGLE
				for( i= 0; i< output->N; i++ ){
					output->array[i]= foutput[i];
					nflops+= 1;
				}
#endif
				output->value= output->array[ (output->last_index= 0) ];
				if( output->accessHandler ){
					AccessHandler( output, "inv_rfftw", level, ASCB_COMPILED, AH_EXPR, NULL );
				}

#endif
				Elapsed_Since(&exec_timer, False);

			}
			else{
				fprintf( StdErr, "(inv_rfftw operation aborted by SIGINT) " ); fflush(StdErr);
				N= 0;
			}

			if( FFTW_Initialised ){
#if defined(HAVE_FFTW) && HAVE_FFTW
				if( p ){
					fftw_destroy_plan(p);
				}
#endif
				XG_fftw_free(data);
#ifdef FFTW_SINGLE
				fftw_xfree(foutput);
#endif
			}
			else{
				xfree(data);
#ifdef FFTW_SINGLE
				xfree(foutput);
#endif
			}
			GCA();
			signal(SIGINT, prevINT);
		}

	     ascanf_elapsed_values[3]= 2.5 * N * (log((double) N) / log(2.0));
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(%g ops, %gs", ascanf_elapsed_values[3], exec_timer.HRTot_T );
		}
		  /* estimate the number of flops including the planning, assuming similar execution speed: */
		{ double flops= ascanf_elapsed_values[3]/exec_timer.HRTot_T;
			ascanf_elapsed_values[3]= flops* (plan_timer.HRTot_T + exec_timer.HRTot_T);
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "; %gs planning => %g ops", plan_timer.HRTot_T, ascanf_elapsed_values[3] );
			}
		}
		ascanf_elapsed_values[3]+= nflops;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "; %lu other ops => %g total ops) ", nflops, ascanf_elapsed_values[3] );
		}

		*result= N;
		return(1);
	}
	return(0);
}

#ifndef FFTW_SINGLE
/* convolve_fft[&Data,&Mask,&Output,direction[,&Data_spectrum[,&Mask_spectrum]]]	*/
int ascanf_convolve_fft ( ASCB_ARGLIST )
{ ASCB_FRAME
  unsigned long i, direction= 1;
  ascanf_Function *Data= NULL, *Mask= NULL, *Output= NULL;
  ascanf_Function *Data_sp= NULL, *Mask_sp= NULL;
#ifdef DEBUG
  static int show_spec= 0;
#endif
#if defined(HAVE_FFTW) && HAVE_FFTW
  fftw_complex *mask= NULL, *mask_spec= NULL;
  fftw_plan mp, rp, ip;
#else
  char *mask= NULL, *mask_spec= NULL;
  typedef char fftw_complex;
#endif
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0, NN, Nm= 0;
		if( !(Data= parse_ascanf_address(args[0], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Data->iarray ){
			ascanf_emsg= " (invalid Data array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= Data->N;
			if( Data->malloc!= XG_fftw_malloc || Data->free!= XG_fftw_free ){
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= (void*) XG_fftw_malloc;
				ascanf_array_free= (void*) XG_fftw_free;
				Resize_ascanf_Array( Data, N, result );
			}
			NN= 2* (N/2+ 1);
		}
		if( (Mask= parse_ascanf_address(args[1], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) ){
			if( Mask->iarray ){
				ascanf_emsg= " (invalid Mask array argument (2)) ";
				ascanf_arg_error= 1;
			}
			else{
				Nm= Mask->N;
			}
		}
		if( args[2] ){
			if( !(Output= parse_ascanf_address(args[2], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Output->iarray ){
				ascanf_emsg= " (invalid Output array argument (3)) ";
				ascanf_arg_error= 1;
			}
			else if( Output->N!= NN || Output->malloc!= XG_fftw_malloc || Output->free!= XG_fftw_free ){
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= (void*) XG_fftw_malloc;
				ascanf_array_free= (void*) XG_fftw_free;
				Resize_ascanf_Array( Output, NN, result );
			}
		}
		else{
			Output= Data;
		}
		direction= (args[3])? 1 : 0;

		if( ascanf_arguments> 4 ){
			if( !(Data_sp= parse_ascanf_address(args[4], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Data_sp->iarray ){
				ascanf_emsg= " (invalid data_spec array argument (5)) ";
				ascanf_arg_error= 1;
			}
			else if( Data_sp->N!= NN ){
				  /* No need to use special (de)allocators */
				Resize_ascanf_Array( Data_sp, NN, result );
			}
		}
		if( ascanf_arguments> 5 ){
			if( !(Mask_sp= parse_ascanf_address(args[5], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Mask_sp->iarray ){
				ascanf_emsg= " (invalid mask_spec array argument (6)) ";
				ascanf_arg_error= 1;
			}
			else if( Mask_sp->N!= NN ){
				if( !Mask ){
				  /* This means we re-use the mask-spectrum returned by a previous call
				   \ (or another initialisation). We must not touch it here, thus we
				   \ warn and return an error when the spectrum passed does not have the
				   \ right size.
				   */
					fprintf( StdErr, " (mask_spec array must have %d size to match Data[%d]) ",
						NN, N
					);
					ascanf_emsg= " (mask_spec array has wrong size) ";
					ascanf_arg_error= 1;
				}
				else{
					Resize_ascanf_Array( Mask_sp, NN, result );
				}
			}
		}
		if( !Mask && !Mask_sp ){
			ascanf_emsg= " (Must specify either valid Mask or mask_spec!) ";
			ascanf_arg_error= 1;
		}

		if( !ascanf_arg_error ){
			if( FFTW_Initialised ){
				mask= (fftw_complex*) XG_fftw_malloc( (NN/2)* sizeof(fftw_complex) );
				mask_spec= (fftw_complex*) XG_fftw_malloc( (NN/2)* sizeof(fftw_complex) );
			}
			else{
				mask= (fftw_complex*) malloc( (NN/2)* sizeof(fftw_complex) );
				mask_spec= (fftw_complex*) malloc( (NN/2)* sizeof(fftw_complex) );
			}
			if( !mask || !mask_spec ){
				fprintf( StdErr, " (can't get mask memory (%s)) ", serror() );
				ascanf_arg_error= 1;
			}
			else{
				memset( mask, 0, (NN/2)* sizeof(fftw_complex) );
				memset( mask_spec, 0, (NN/2)* sizeof(fftw_complex) );
			}
		}

		if( ascanf_arg_error || !N || (Mask && !Nm) || !Output->array || !mask || !mask_spec || ascanf_SyntaxCheck ){
			if( FFTW_Initialised ){
				fftw_xfree( mask );
				fftw_xfree( mask_spec );
			}
			else{
				xfree( mask );
				xfree( mask_spec );
			}
			return(0);
		}

#if defined(HAVE_FFTW) && HAVE_FFTW

		{  double *output, *data;
			output= (double*) malloc( Output->N * sizeof(double) );
			data= (double*) malloc( N * sizeof(double) );
			if( output && data ){
			  int plevel;
				memcpy( output, Output->array, Output->N* sizeof(double) );
				  /* FFTW3's planners want pointers to the to-be-used memory, and *overwrite* it. */
				memcpy( data, Data->array, N* sizeof(double));

				CLIP( *fftw_planner_level, -1, 2 );
				switch( (int) *fftw_planner_level ){
					case -1:
						plevel= FFTW_ESTIMATE;
						break;
					case 0:
						plevel= FFTW_MEASURE;
						break;
					case 1:
						plevel= FFTW_PATIENT;
						break;
					case 2:
						plevel= FFTW_EXHAUSTIVE;
						break;
				}
				if( FFTW_Initialised ){
					if( Mask ){
						mp= fftw_plan_dft_r2c_1d( N, (double*) mask, mask_spec, plevel );
					}
	// 				rp= fftw_plan_dft_r2c_1d( N, Data->array, (fftw_complex*) Output->array, plevel );
	// 				ip= fftw_plan_dft_c2r_1d( N, mask, Output->array, plevel );
					rp= fftw_plan_dft_r2c_1d( N, Data->array, (fftw_complex*) output, plevel );
					ip= fftw_plan_dft_c2r_1d( N, mask, output, plevel );
				}
				else{
#ifdef FFTW_DYNAMIC
					ascanf_emsg= " (fftw not initialised) ";
					ascanf_arg_error= 1;
					xfree( mask );
					xfree( mask_spec );
					return(0);
#else
					if( Mask ){
						mp= fftw_plan_dft_r2c_1d( N, (double*) mask, mask_spec, FFTW_ESTIMATE );
					}
	// 				rp= fftw_plan_dft_r2c_1d( N, Data->array, (fftw_complex*) Output->array, FFTW_ESTIMATE );
	// 				ip= fftw_plan_dft_r2c_1d( N, mask, Output->array, FFTW_ESTIMATE );
					rp= fftw_plan_dft_r2c_1d( N, Data->array, (fftw_complex*) output, FFTW_ESTIMATE );
					ip= fftw_plan_dft_r2c_1d( N, mask, output, FFTW_ESTIMATE );
#endif
				}
				memcpy( Data->array, data, N* sizeof(double) );
	// 			memcpy( Output->array, output, Output->N* sizeof(double) );
			}
			else{
				fprintf( StdErr, " (can't get working memory (%s)) ", serror() );
				ascanf_arg_error= 1;
				xfree(data);
				xfree(output);
				goto bail;
			}

			if( Mask ){
			  long j;
			  double *m= (double*) mask;
				  /* Put the real mask in the centre of the array to be FFTed
				   \ -- disregarding any padding at the end of that array!
				   */
				j= Nm-1;
				for( i= 0; i< N/2- Nm/2; i++ ){
					m[i]= Mask->array[j];
				}
				  /* 990914: put in the mask reversed	*/
				for( j= Nm-1; j>= 0; i++, j-- ){
					m[i]= Mask->array[j];
				}
				j= 0;
				for( ; i< N; i++ ){
					m[i]= Mask->array[j];
				}
				fftw_execute( mp );
			}
			else{
				memcpy( mask_spec, Mask_sp->array, (NN/2)* sizeof(fftw_complex) );
			}

			fftw_execute( rp );

			if( Data_sp && Data_sp->array ){
				memcpy( Data_sp->array, output, (NN)* sizeof(double) );
				Data_sp->value= Data_sp->array[ (Data_sp->last_index= 0) ];
				if( Data_sp->accessHandler ){
					AccessHandler( Data_sp, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
				}
			}
			if( Mask && Mask_sp && Mask_sp->array ){
				memcpy( Mask_sp->array, mask_spec, (NN)* sizeof(double) );
				Mask_sp->value= Mask_sp->array[ (Mask_sp->last_index= 0) ];
				if( Mask_sp->accessHandler ){
					AccessHandler( Mask_sp, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
				}
			}

			{ fftw_complex *dat= (fftw_complex*) output;
			  int sign= 1;
				  /* multiply odd elements with -1 to get "the centre in the centre"
				   \ (and not at the 2 edges)
				   */
				if( direction ){
					for( i= 0; i<= N/2; i++ ){
						c_re(mask[i])= sign* (c_re(dat[i])* c_re(mask_spec[i])- c_im(dat[i])* c_im(mask_spec[i]))/ N;
						c_im(mask[i])= sign* (c_re(dat[i])* c_im(mask_spec[i])+ c_im(dat[i])* c_re(mask_spec[i]))/ N;
						sign*= -1;
					}
				}
				else{
				  double b2;
					  /* vi subst pattern for c_re()/c_im(): s/\([^ ^I\[(]*\)\[i\]\.\([ri][em]\)/c_\2(\1[i])/gc */
					for( i= 0; i<= N/2; i++ ){
						b2= N*( c_re(mask_spec[i])* c_re(mask_spec[i]) + c_im(mask_spec[i])* c_im(mask_spec[i]) );
						if( b2 ){
							c_re(mask[i])= sign* (c_re(dat[i])* c_re(mask_spec[i])+ c_im(dat[i])* c_im(mask_spec[i]))/ b2;
							c_im(mask[i])= sign* (-c_re(dat[i])* c_im(mask_spec[i])+ c_im(dat[i])* c_re(mask_spec[i]))/ b2;
						}
						else{
						  /* Define deconvolution with NULL mask as NOOP:	*/
							  /* fftw_complex can be a structure, or it can be a double[2], in which
							   \ case a direct assignment won't work. memmove() will handle all possible
							   \ cases (including overlapping memory).
							   */
							memmove( &mask[i], &dat[i], sizeof(fftw_complex) );
						}
						sign*= -1;
					}
				}
			}
#	ifdef DEBUG
			memcpy( output, mask, NN* sizeof(double) );
			if( !show_spec ){
				fftw_execute( ip );
			}
#	else
			fftw_execute( ip );
#	endif
			fftw_destroy_plan(ip);
			fftw_destroy_plan(rp);
			fftw_destroy_plan(mp);

			memcpy( Output->array, output, Output->N* sizeof(double) );
		}
#endif
bail:
		if( FFTW_Initialised ){
			fftw_xfree( mask );
			fftw_xfree( mask_spec );
		}
		else{
			xfree( mask );
			xfree( mask_spec );
		}

		GCA();

		Output->value= Output->array[ (Output->last_index= 0) ];
		if( Output->accessHandler ){
			AccessHandler( Output, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
		}

		*result= N;
		return(1);
	}
	return(0);
}
#else
/* convolve_fft[&Data,&Mask,&Output,direction[,&Data_spectrum[,&Mask_spectrum]]]	*/
int ascanf_convolve_fft ( ASCB_ARGLIST )
{
	fprintf( StdErr, "convolve_fft[] is not yet implemented in single precision mode; use double precision (fourconv3.so)\n" );
	ascanf_arg_error= 1;
	return(0);
}
#endif

#ifndef _ARRAYVOPS_H
void _convolve( double *Data, double *Mask, double *Output, int Start, int End, int Nm )
{ int nm= Nm/ 2, i, j, k;
	for( i= Start; i< End; i++ ){
	 double accum = 0;
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
/* 			if( k>= 0 && k< N ){	*/
/* 				accum+= Mask[j]* Data[k];	*/
/* 			}	*/
			accum += Mask[j]* v;
		}
		Output[i]= accum;
	}
}
#endif

/* convolve[&Data,&Mask,&Output,direction,nan_handling]	*/
int ascanf_convolve ( ASCB_ARGLIST )
{ ASCB_FRAME
  size_t direction= 1, nan_handling, padding;
  size_t NN;
  ascanf_Function *Data= NULL, *Mask= NULL, *Output= NULL;
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
	}
	else{
	  size_t i, N= 0, Nm= 0;
	  double *output= NULL;
		direction= (args[3])? 1 : 0;
		if( !(Data= parse_ascanf_address(args[0], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Data->iarray ){
			ascanf_emsg= " (invalid Data array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= Data->N;
		}
		if( !(Mask= parse_ascanf_address(args[1], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Mask->iarray ){
			ascanf_emsg= " (invalid Mask array argument (2)) ";
			ascanf_arg_error= 1;
		}
		else{
			Nm= Mask->N;
		}
		  /* 20041225: always do padding...! */
		padding= Nm/2 + 1;
		NN= N+2*padding;
		if( args[2]!= 0 ){
			if( !(Output= parse_ascanf_address(args[2], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL ))
				|| Output->iarray
			){
				ascanf_emsg= " (invalid Output array argument (3)) ";
				ascanf_arg_error= 1;
			}
			else if( Output->N!= ((direction>=0)? NN : N) ){
				Resize_ascanf_Array( Output, ((direction>=0)? NN : N), result );
				output= Output->array;
			}
		}
		else{
			output= (double*) malloc( sizeof(double)* ((direction>=0)? NN : N) );
		}
		if( ascanf_arguments> 4 && ASCANF_TRUE(args[4]) ){
			nan_handling= (int) args[4];
		}
		else{
			nan_handling= False;
		}

		if( ascanf_arg_error || !N || !Nm || !output || ascanf_SyntaxCheck ){
			return(0);
		}

		GCA();

		if( direction>= 0 ){
		  double *data= (double*) malloc( NN*sizeof(double));
			if( data ){
				for( i= 0; i< padding; i++ ){
					data[i]= Data->array[0];
				}
				memcpy( &data[i], Data->array, N*sizeof(double) );
				i+= N;
				for( ; i< NN; i++ ){
					data[i]= Data->array[N-1];
				}
				  /* If requested, treat the data (source, input) array for NaNs. Gaps with
				   \ NaNs are filled with a linear gradient between the surrounding values (nan_handling==2)
				   \ or with a 'half-way step' between these values (nan_handling==1) if possible, otherwise,
				   \ simple padding with the first or last non-NaN value is done.
				   \ These estimates are removed after the convolution.
				   */
				nanHandled= fourconv3_nan_handling( data, NN, nan_handling );
#ifdef _ARRAYVOPS_H
				_convolve( data, NN, Mask->array, output, 0, NN, Nm );
#else
				_convolve( data, Mask->array, output, 0, NN, Nm );
#endif
				memmove( output, &output[padding], N*sizeof(double) );
				if( Output ){
					Resize_ascanf_Array( Output, N, result );
				}
				if( nan_handling ){
					for( i= 0; i< N; i++ ){
						if( isNaN(Data->array[i]) ){
							output[i]= Data->array[i];
						}
					}
				}
				xfree(data);
			}
			else{
				fprintf( StdErr, " (can't get working memory (%s)) ", serror() );
				ascanf_arg_error= 1;
			}
		}
		else{
		  int aa= ascanf_arguments;
			if( ascanf_arguments> 4 ){
				ascanf_arguments= 4;
			}
			ascanf_convolve_fft( ASCB_ARGUMENTS );
			ascanf_arguments= aa;
		}
		if( !Output ){
			memmove( Data->array, output, N*sizeof(double) );
			xfree( output );
			Output= Data;
		}
		Output->value= Output->array[ (Output->last_index= 0) ];
		if( Output->accessHandler ){
			AccessHandler( Output, "convolve", level, ASCB_COMPILED, AH_EXPR, NULL );
		}

		*result= N - nanHandled;
		return(1);
	}
	return(0);
}


static ascanf_Function fourconv_Function[] = {
	{ "SavGolayCoeffs", ascanf_SavGolCoeffs, 4, NOT_EOF_OR_RETURN,
		"SavGolayCoeffs[&coeffs_ret,<halfwidth>[,<pol_order>=4[,<deriv_order>=0]]]: determine the coefficients\n"
		" for a Savitzky-Golay convolution filter. This returns the same coefficients as SavGolayInit[] would, without\n"
		" additional action (mainly allocations) on a data set. The coefficients are returned in <coeffs_ret> which should\n"
		" be an array of floats (a pointer to this array is also returned).\n"
		" To call this from Python:\n"
		" ascanf.Eval('DCL[SGCoeffs,1,0]')\n"
		" SGCoeffs= ascanf.ImportVariable('&SavGolayCoeffs')\n"
		" coeffs= SGCoeffs( ascanf.ImportVariable('&SGCoeffs'), FW, FO, deriv )\n"
	},
	{ "SavGolayInit", ascanf_SavGolInit, 8, NOT_EOF_OR_RETURN, "SavGolayInit[<set_nr>,<halfwidth>[,<pol_order>=4[,<deriv_order>=0[,pad=0[,padlow[,padhigh[,coeff_p]]]]]]: initialise a Savitzky-Golay filter\n"
		" for all columns in set <set_nr>, with filterwidth 2* <halfwidth>, smoothing polynomial order <pol_order>,\n"
		" and order-of-derivative <deriv_order> (0 for smoothed function)\n"
		" set <pad>=-1 to add low/high padding over <halfwidth> extra values, set to positive value to add <pad> values\n"
		" <padlow>: if given, value for padding at low end, else (or if NaN) pad with the value at $Counter=0\n"
		" <padhigh>: if given, value for padding at high end, else (or if NaN) pad with the value at $Counter=$numPoints-1\n"
		" <coeff_p>: if given, store the coefficients (2*halfwidth+3 elements) in this array\n"
	},
	{ "SavGolayX", ascanf_SavGolX, 1, NOT_EOF_OR_RETURN, "SavGolayX[<pnt_nr>]: retrieve the \"savgolayed\" X value number <pnt_nr>"},
	{ "SavGolayY", ascanf_SavGolY, 1, NOT_EOF_OR_RETURN, "SavGolayY[<pnt_nr>]: retrieve the \"savgolayed\" Y value number <pnt_nr>"},
	{ "SavGolayE", ascanf_SavGolE, 1, NOT_EOF_OR_RETURN, "SavGolayE[<pnt_nr>]: retrieve the \"savgolayed\" error/orientation value number <pnt_nr>"},
	{ "SavGolay", ascanf_SavGolColNr, 2, NOT_EOF_OR_RETURN, "SavGolay[<col>,<pnt_nr>]: retrieve the \"savgolayed\" value number <pnt_nr> for column <col>"},
	{ "SavGolay2Array", ascanf_SavGolColumn2Array, 5, NOT_EOF_OR_RETURN,
		"SavGolay2Array[dest_p,col[,start[,end_incl[,offset]]]]: copy SavGolay column <col> into <dest_p> which\n"
		" must be a pointer to an array. <start> and <end_incl> specify (inclusive) source start and end of copying (end_incl==-1\n"
		" to copy until last); <offset> specifies starting point in <dest_p> which will be expanded as necessary\n"
		" <start> may also be an array. In that case, it is interpreted as <&visible>: this means that only points will\n"
		" be returned that are visible in the currently active window (no active window => no visible points!). The <visible>\n"
		" array will then contain the indices of those points (NB: that is *currently visible*; no a priori relation with the\n"
		" filtered values exists!!).\n"
	},
	{ "SavGolayFinished", ascanf_SavGolFinished, 1, NOT_EOF_OR_RETURN, "SavGolayFinished: de-allocate Savitzky-Golay filtering resources"},

#if defined(HAVE_FFTW) && HAVE_FFTW
	{ "InitFFTW", ascanf_InitFFTW, 1, NOT_EOF_OR_RETURN, "InitFFTW: optionally initialise FFTW:\n"
#ifdef FFTW_THREADED
		" load wisdom from ~/.Preferences/.xgraph/fft3[f]_wisdom_threaded or ${XG_WISDOM}_threaded\n"
#else
		" load wisdom from ~/.Preferences/.xgraph/fft3[f]_wisdom or $XG_WISDOM\n"
#endif
		" By calling InitFFTW, FFTW routines will use $fftw-planner-level, otherwise FFTW_ESTIMATE\n"
		" (see FFTW documentation)\n"
	},
	{ "CloseFFTW", ascanf_CloseFFTW, 1, NOT_EOF_OR_RETURN, "CloseFFTW or CloseFFTW[1]: optionally terminate FFTW:\n"
#ifdef FFTW_THREADED
		" store increased wisdom to ~/.Preferences/.xgraph/fft3[f]_wisdom_threaded or ${XG_WISDOM}_threaded\n"
#else
		" store increased wisdom to ~/.Preferences/.xgraph/fft3[f]_wisdom or $XG_WISDOM\n"
#endif
		" Passing True causes all wisdom to be forgotten afterwards\n"
	},
	{ "$fftw-interrupt-aborts", NULL, 2, _ascanf_variable,
		"$fftw-interrupt-aborts: if set, a SIGINT (^C) sets flags that cause the current ascanf expression\n"
		" to be aborted. This can provoke unwanted side-effects while drawing, so it is not set by default\n"
		" (only the ongoing Fourier transform will be interrupted).\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$fftw-nthreads", NULL, 2, _ascanf_variable,
		"$fftw-nthreads: determines the number of threads used for doing transforms.\n"
#ifndef FFTW_THREADED
		" !!! thread support not available, variable ignored !!!\n"
#endif
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$fftw-planner-level", NULL, 2, _ascanf_variable,
		"$fftw-planner-level: determines the thorougness of the planner routines when using wisdom:\n"
		" -1: FFTW_ESTIMATE: don't do specific planning, but make a guess at a sub-optimal plan.\n"
		" 0: FFTW_MEASURE (comparable to FFTW2)\n"
		" 1: FFTW_PATIENT: planner takes more time, subsequent execution might be a lot faster\n"
		" 2: FFTW_EXHAUSTIVE: should be the summum.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "rfftw", ascanf_fftw, 4, NOT_EOF_OR_RETURN,
		"rfftw[<input_p>,<output_p>[,<power_p>[,nan_handling]]]: do an FFT (using RFFTW) of the array pointed to by <input_p>\n"
		" storing the transform in <output_p> and optionally the power spectrum in <power_p>\n"
		" All must be double arrays; if necessary, <output_p> and <power_p> are resized to input_p->N,\n"
		" even though <power_p> will have only input_p->N/2+1 relevant elements.\n"
		" (if not of interest, <output> and <power_p> may be 0)\n"
		" The <nan_handling> argument indicates what to do with gaps of NaN value(s) in the input:\n"
		" \t1: if possible, pad with the values surrounding the gap (step halfway)\n"
		" \t2: if possible, intrapolate linearly between the surrounding values\n"
		" \t(if not possible, simply pad with the first or last non-NaN value). Note that doing an inverse\n"
		" transform on the result will 'show' how the gaps were filled, so keep a copy of the original data\n"
		" if gaps (of NaNs) are to remain gaps (so you can restore them).\n"
	},
	{ "inv_rfftw", ascanf_inv_fftw, 3, NOT_EOF_OR_RETURN,
		"inv_rfftw[<input_p>,<output_p>[,<normalise>]]: do an inverse FFT (using RFFTW) of the array pointed to by <input_p>\n"
		" storing the transform in <output_p>, optionally normalising <input_p> first (division by input_p->N).\n"
		" All must be double arrays; <input_p> should be the result of a call to rfftw[]. <output_p> should have\n"
		" the same dimension as the <input_p> array to rfftw[]. If it does not meet that requirement, a check is made\n"
		" if <input_p> matches the size of the last rfftw transformation. If so, that size is used, otherwise, the call fails.\n"
	},
	{ "convolve_fft", ascanf_convolve_fft, 6, NOT_EOF_OR_RETURN,
		"convolve_fft[&Data,&Mask,&Output,direction[,&data_spec[,&mask_spec]]]: (de)convolve the array pointed to by <&Data> by <&Mask>\n"
		" storing the transform in <Output>.\n"
		" <direction> is >=0 for convolution, <0 for deconvolution\n"
		" If specified, <data_spec> will receive FFT(<Data>), and <mask_spec> FFT(<Mask>)\n"
		" <Mask> may be 0 if <mask_spec> contains a valid mask spectrum\n"
		" All must be double arrays; <Output>, <data_spec> and/or <mask_spec> are resized to Data->N if necessary.\n"
	},
#else
	{ "InitFFTW", ascanf_InitFFTW, 1, NOT_EOF_OR_RETURN, "InitFFTW: not operational without FFTW libraries linked."},
	{ "CloseFFTW", ascanf_CloseFFTW, 1, NOT_EOF_OR_RETURN, "CloseFFTW: not operational without FFTW libraries linked."},
	{ "$fftw-planner-level", NULL, 2, _ascanf_variable,
		"$fftw-planner-level: ignored without FFTW libraries linked."
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "rfftw", ascanf_fftw, 4, NOT_EOF_OR_RETURN, "rfftw[<input_p>,<output_p>[,<power_p>[,nan_handling]]]: not operational" },
	{ "inv_rfftw", ascanf_inv_fftw, 3, NOT_EOF_OR_RETURN, "inv_rfftw[<input_p>,<output_p>[,<normalise>]]: not operational" },
	{ "convolve_fft", ascanf_convolve_fft, 6, NOT_EOF_OR_RETURN, "convolve_fft[&Data,&Mask,&Output,direction[,&data_spec[,&mask_spec]]]: not operational" },
#endif
	{ "convolve", ascanf_convolve, 5, NOT_EOF_OR_RETURN,
		"convolve[&Data,&Mask,&Output,direction[,nan_handling]]: (de)convolve the array pointed to by <&Data> by <&Mask>\n"
		" storing the transform in <Output>.\n"
		" <direction> is >=0 for convolution, <0 for deconvolution\n"
		" All must be double arrays; <Output> is resized to Data->N if necessary.\n"
		" The <nan_handling> argument indicates what to do with gaps of NaN value(s) in the input:\n"
		" \t1: if possible, pad with the values surrounding the gap (step halfway)\n"
		" \t2: if possible, intrapolate linearly between the surrounding values\n"
		" \t(if not possible, simply pad with the first or last non-NaN value).\n"
		" The estimated values are replaced by the original NaNs after convolution.\n"
		" This routine uses direct convolution.\n"
		" <Output> may be 0, in which case the operation is performed \"in place\" (= using a temp. buffer)\n"
	},

};
static int fourconv_Functions= sizeof(fourconv_Function)/sizeof(ascanf_Function);

#ifdef FFTW_THREADED
static ascanf_Function internal_AHandler= { "$fftw-nthreads-access-handler", nthreads_callback, 0, _ascanf_function,
	"An internal accesshandler variable",
};
#endif

static void af_initialise( DyModLists *theDyMod, char *label )
{ ascanf_Function *af= fourconv_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< fourconv_Functions; i++, af++ ){
		if( !called ){
			if( af->name ){
				af->name= XGstrdup( af->name );
			}
			else{
				sprintf( buf, "Function-%d", i );
				af->name= XGstrdup( buf );
			}
			if( af->usage ){
				af->usage= XGstrdup( af->usage );
			}
			ascanf_CheckFunction(af);
			if( af->function!= ascanf_Variable ){
				set_NaN(af->value);
			}
			if( label ){
				af->label= XGstrdup( label );
			}
			if( strcmp( af->name, "$fftw-planner-level")== 0 ){
				fftw_planner_level= &(af->value);
			}
			if( strcmp( af->name, "$fftw-interrupt-aborts")== 0 ){
				fftw_interrupt_aborts= &(af->value);
			}
			if( strcmp( af->name, "$fftw-nthreads")== 0 ){
#ifdef FFTW_THREADED
				af->accessHandler= &internal_AHandler;
#endif
				fftw_nthreads= &(af->value);
			}
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= theDyMod;
	}
	called+= 1;
}

#ifdef __cplusplus
extern "C" {
#endif

static int initialised= False;

DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;
  char *libnamef= "DM-fourconv3f";
  char *libname= "DM-fourconv3";

	if( !DMBase ){
	  DyModLists *current;
		DMBaseMem.sizeof_DyMod_Interface= sizeof(DyMod_Interface);
		if( !initialise(&DMBaseMem) ){
			fprintf( stderr, "Error attaching to xgraph's main (programme) module\n" );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
		DMBase= &DMBaseMem;
		if( !DyMod_API_Check(DMBase) ){
			fprintf( stderr, "DyMod API version mismatch: either this module or XGraph is newer than the other...\n" );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}
		current= DyModList;
		while( current ){
			if( strcmp( libname, current->libname)== 0
				|| strcmp( libnamef, current->libname)== 0
			){
				fprintf( StdErr, "LoadDyMod(%s): request ignored because module \"%s\" has already been loaded.\n"
					"\tUnload that module (%s from %s) first, and try again.\n",
					theDyMod->name, current->libname, current->name, current->path
				);
				return(DM_Error);
			}
			current= current->cdr;
		}

		XGRAPH_FUNCTION(ascanf_Arrays2Regular_ptr, "ascanf_Arrays2Regular");
		XGRAPH_VARIABLE( ascanf_elapsed_values_ptr, "ascanf_elapsed_values" );
#if 0
  /* not serving any purpose other than fiddling with the Accelerate framework: */
		{ double a[10]= {1,2,3,4,5,6,7,8,9,-1}, val= 2;
			vsmulD( a, 1, &val, a, 1, sizeof(a)/sizeof(double) );
		}
#endif
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( fourconv_Function, fourconv_Functions, "fourconv::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
#ifdef FFTW_SINGLE
	theDyMod->libname= XGstrdup( libnamef );
#else
	theDyMod->libname= XGstrdup( libname );
#endif
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" various Fourier and Convolution functions.\n"
#ifdef FFTW_SINGLE
		" Uses libfftw3f in single precision mode (possibly supporting SIMD on P3,AMD and PPC).\n"
		" Provides the Savitzky-Golay filter.\n"
#else
		" Uses libfftw3. Provides the Savitzky-Golay filter.\n"
#endif
	);
	return( DM_Ascanf );
}

/* The close handler. We can be called with the force flag set to True or False. True means force
 \ the unload, e.g. when exitting the programme. In that case, we are supposed not to care about
 \ whether or not there are ascanf entries still in use. In the alternative case, we *are* supposed
 \ to care, and thus we should heed remove_ascanf_function()'s return value. And not take any
 \ action when it indicates variables are in use (or any other error). Return DM_Unloaded when the
 \ module was de-initialised, DM_Error otherwise (in that case, the module will remain opened).
 */
int closeDyMod( DyModLists *target, int force )
{ static int called= 0;
  int i;
  DyModTypes ret= DM_Error;
  FILE *SE= (initialised)? StdErr : stderr;
	fprintf( SE, "%s::closeDyMod(%d): Closing %s loaded from %s, call %d", __FILE__,
		force, target->name, target->path, ++called
	);
	if( target->loaded4 ){
		fprintf( SE, "; auto-loaded because of \"%s\"", target->loaded4 );
	}
	if( initialised ){
	  int r;
		if( (r= ascanf_Arrays2Regular( (void*) XG_fftw_malloc, (void*) XG_fftw_free )) ){
			fprintf( SE, " (%d special arrays converted to regular) ", r );
		}
		r= remove_ascanf_functions( fourconv_Function, fourconv_Functions, force );
		if( force || r== fourconv_Functions ){
			for( i= 0; i< fourconv_Functions; i++ ){
				fourconv_Function[i].dymod= NULL;
			}
			initialised= False;
			xfree( target->libname );
			xfree( target->buildstring );
			xfree( target->description );
			ret= target->type= DM_Unloaded;
			if( r<= 0 || ascanf_emsg ){
				fprintf( SE, " -- warning: variables are in use (remove_ascanf_functions() returns %d,\"%s\")",
					r, (ascanf_emsg)? ascanf_emsg : "??"
				);
				Unloaded_Used_Modules+= 1;
				if( force ){
					ret= target->type= DM_FUnloaded;
				}
			}
			fputc( '\n', SE );
		}
		else{
			fprintf( SE, " -- refused: variables are in use (remove_ascanf_functions() returns %d out of %d)\n",
				r, fourconv_Functions
			);
		}
	}
	return(ret);
}

#ifdef __cplusplus
}
#endif

/* todo:
 \ re-verify re-conversion of 'special' arrays
 \ add auto-load interface through fftw3-* functions?
 */
