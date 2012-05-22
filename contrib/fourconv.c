#include "config.h"
IDENTIFY( "Fourier/Convolution ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif

#if defined(__APPLE_CC__) || defined(__MACH__)
  /* For the time being... */
#	undef HAVE_FFTW
#endif

#include <stdio.h>
#include <stdlib.h>

  /* Get the dynamic module definitions:	*/
#include "dymod.h"

extern FILE *StdErr;

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
#	define ascanf_Arrays2Regular (*ascanf_Arrays2Regular_ptr)

#include <float.h>

/* savgol functions: */
/* discr_fourtr():
 \ Replaces data[1..2*nn] by its discrete Fourier transform, if isign is input as 1; or replaces
 \ data[1..2*nn] by nn times its inverse discrete Fourier transform, if isign is input as -1.
 \ data is a complex array of length nn or, equivalently, a real array of length 2*nn. nn MUST
 \ be an integer power of 2 (this is not checked for!).
 */
void discr_fourtr(double *data, unsigned long nn, int isign)
{ unsigned long n, mmax, m, j, istep, i;
  double wtemp, wr, wpr, wpi, wi, theta;
  double tempr, tempi;
    /* is = isign* 2	*/
  int is= (isign>=0)? 2 : -2;

	n= nn << 1;
	j= 1;
	for( i= 1; i< n; i+= 2 ){
	  /* This is the bit-reversal section of the routine.	*/
		if(j > i ){
			SWAP(data[j], data[i], double); /* Exchange the two complex numbers.	*/
			SWAP(data[j+1], data[i+1], double);
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
void twofft(double *data1, double *data2, double *fft1, double *fft2, unsigned long n)
{ unsigned long nn3, nn2, jj, j;
  double rep, rem, aip, aim;
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
void realft(double *data, unsigned long n, int isign)
{ unsigned long i, i1, i2, i3, i4, np3;
  double c1= 0.5, c2, h1r, h1i, h2r, h2i;
  double wr, wi, wpr, wpi, wtemp, theta;

	theta= M_PI/((double) (n>> 1));	/* Initialize the recurrence.	*/
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
int convlv(double *data, unsigned long n, double *respns, unsigned long m, int isign, double *ans, Boolean preproc_respns )
{ unsigned long i, no2;
  double dum, mag2;
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
	twofft(data, respns, fft, ans, n);	/* FFT both at once.	*/
	no2= n>> 1;
	for( i= 2; i<= n+2; i+= 2 ){
		if( isign >= 0 ){
			ans[i-1]= (fft[i-1]*(dum= ans[i-1])-fft[i]*ans[i])/no2;	/*  Multiply FFTs to convolve.	*/
			ans[i]= (fft[i]*dum+fft[i-1]*ans[i])/no2;
		}
		else{
			if( (mag2=SQR(ans[i-1],double)+SQR(ans[i],double)) == 0.0){
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

int ludcmp(double **a, int n, int *indx, double *d)
/* 
 \ Given a matrix a[1..n][1..n], this routine replaces it by the LU decomposition of a rowwise
 \ permutation of itself. a and n are input. a is output, arranged as in equation (2.3.14) above;
 \ indx[1..n] is an output vector that records the row permutation effected by the partial
 \ pivoting; d is output as if 1 depending on whether the number of row interchanges was even
 \ or odd, respectively. This routine is used in combination with lubksb to solve linear equations
 \ or invert a matrix.
 */
{ int i, imax= 0, j, k;
  double big, dum, sum, temp;
	  /* vv stores the implicit scaling of each row. 	*/
  ALLOCA(vv, double, n+1, vv_len);

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
/* 				SWAP( a[imax][k], a[j][k], double );	*/
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

void lubksb(double **a, int n, int *indx, double *b)
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
  double sum;
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


void xfree_dmatrix( double **a, int h, int v )
{ int i;
	if( a ){
		for( i= 0; i<= v; i++ ){
			xfree( a[i] );
		}
		xfree( a );
	}
}

double **calloc_dmatrix( int h, int v)
{ int i;
  double **m;

	  /* 20010901: RJVB: allocate 1 element more per row/column. Adaptation of NR
	   \ code dealing with matrices is tricky business...
	   */
	if( !(m = (double **) calloc((unsigned) v+1,sizeof(double*))) ){
		fprintf( StdErr, "allocation failure 1 in calloc_dmatrix(%d,%d) (%s)\n",
			h, v, serror()
		);
		return( NULL );
	}
	for( i = 0; i <= v; i++ ){
		if( !(m[i] = (double *) calloc((unsigned) h+ 1, sizeof(double))) ){
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
int savgol(double *c, int np, int nl, int nr, int ld, int m)
{ void lubksb(double **a, int n, int *indx, double *b);
  int ludcmp(double **a, int n, int *indx, double *d);
  int imj, ipj, j, k, kk, mm;
  double d, fac, sum,**a;
  ALLOCA( indx, int, m+2, indx_len);
  ALLOCA( b, double, m+2, b_len);

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
			sum += pow((double) k, (double) ipj);
		}
		for( k= 1; k<= nl; k++){
			sum += pow((double)-k, (double) ipj);
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
	xfree_dmatrix(a, m+2, m+2);
	GCA();
	return(1);
}


static DataSet *SavGolSet= NULL;
static int SavGolHSize, SavGolDOrder= 0, SavGolSOrder= 4, SavGolN;
static double *SavGolCoeffs= NULL;

typedef struct SGResults{
	double *smoothed;
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
			if( !( sg[i].smoothed= (double*) XGrealloc( sg[i].smoothed, N* sizeof(double))) ){
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
		  double *coeffs=NULL;
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
			if( !(coeffs= (double*) XGrealloc(coeffs, (N+ 1)* sizeof(double) )) ){
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
/* 990902: what was this doing here?	*/
/* 		if( ascanf_arguments> 3 ){	*/
/* 			CLIP_EXPR( SavGolDOrder, (int) args[3], 0, SavGolSOrder );	*/
/* 		}	*/
/* 		else{	*/
/* 			SavGolDOrder= 0;	*/
/* 		}	*/
		if( idx>= 0 && idx< setNumber && AllSets[idx].numPoints> 0 ){
		  unsigned long N, pwr2;
		  double *coeffs=NULL, *input= NULL, pad_low, pad_high;
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
			  double p;
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
					(SavGolCoeffs= (double*) XGrealloc(SavGolCoeffs, (N+ 1)* sizeof(double) )) &&
					(coeffs= (double*) XGrealloc(coeffs, (N+ 1)* sizeof(double) )) &&
					(input= (double*) XGrealloc(input, (N+ 1)* sizeof(double) )) &&
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
					for( i= 0; i< SavGolSet->ncols; i++ ){

						memcpy( coeffs, SavGolCoeffs, (N+1)* sizeof(double) );

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
						memcpy( &input[1+pad], SavGolSet->columns[i], SavGolN* sizeof(double) );
						for( j= pad+SavGolN+1; j<= N; j++ ){
						  /* Trailing-pad over the filterwidth with the final value:	*/
							input[j]= pad_high;
						}
						  /* Pass pointer to smoothed[-1] as target area, so that NR's unit offset array
						   \ maps to a normal, C, 0 offset array. Since we don't address smoothed[-1]
						   \ (no read nor write), this should not pose problems..
						   */
						if( (SavGolHSize== 0 && SavGolSOrder== 0 && SavGolDOrder==0) ){
							memcpy( SavGolResult[i].smoothed, &input[1], N* sizeof(double) );
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
  double idx;
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
  double idx;
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
  double idx;
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
  double idx;
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
  double col, *column;
  int start= 0, end= -1, offset= 0, i, j;
  ascanf_Function *targ;
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
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
		CLIP(args[2], 0, SavGolN-1 );
		start= (int) args[2];
	}
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
	if( !ascanf_SyntaxCheck && !SGCOL_OK(col) ){
		ascanf_emsg= " (no SavGol data) ";
		ascanf_arg_error= 1;
		return(0);
	}
	column= &(SavGolResult[(int)col].smoothed[SGfw]);
	if( targ->N< (offset+ end- start+ 1) && !ascanf_SyntaxCheck ){
	  int n= offset+ end- start+ 1;
/* 		if( targ->iarray || targ->name[0]== '%' ){	*/
/* 			targ->iarray= (int*) realloc( targ->iarray, n* sizeof(int) );	*/
/* 			for( i= targ->N; i< n; i++ ){	*/
/* 				targ->iarray[i]= 0;	*/
/* 			}	*/
/* 		}	*/
/* 		else{	*/
/* 			targ->array= (double*) realloc( targ->array, n* sizeof(double) );	*/
/* 			for( i= targ->N; i< n; i++ ){	*/
/* 				targ->array[i]= 0;	*/
/* 			}	*/
/* 		}	*/
/* 		targ->N= n;	*/
		Resize_ascanf_Array( targ, n, NULL );
	}
	if( ascanf_SyntaxCheck ){
		return(1);
	}
	if( targ->iarray ){
		for( j= 0, i= start; i<= end; j++, i++ ){
			targ->value= targ->iarray[(targ->last_index=offset+j)]= (int) column[i];
		}
		targ->value= targ->iarray[(targ->last_index=offset+j-1)];
	}
	else{
		for( j= 0, i= start; i<= end; j++, i++ ){
			targ->value= targ->array[(targ->last_index=offset+j)]= column[i];
		}
		targ->value= targ->array[(targ->last_index=offset+j-1)];
	}
	if( targ->accessHandler ){
		AccessHandler( targ, "SavGolay2Array", level, ASCB_COMPILED, AH_EXPR, NULL  );
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr,
			" (copied %d(%d-%d) elements from SavGol[w=%d,o=%d,d=%d] filtered set#%d column %d to %s[%d] (%d elements))== ",
			j, start, end,
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

#include <fftw.h>
#include <rfftw.h>

#endif

int FFTW_Initialised= False;
char *FFTW_wisdom= NULL;

#if defined(FFTW_DYNAMIC) && defined(HAVE_FFTW) && HAVE_FFTW
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
FNPTR( fftw_import_wisdom_from_file_ptr, fftw_status, (FILE *input_file));
FNPTR( fftw_export_wisdom_to_string_ptr, char*, (void));
FNPTR( fftw_forget_wisdom_ptr, void, (void));
FNPTR( fftw_malloc_ptr, void*, (size_t size));
FNPTR( fftw_free_ptr, void, (void*));
FNPTR( rfftw_create_plan_ptr, rfftw_plan, (int n, fftw_direction dir, int flags));
FNPTR( rfftw_destroy_plan_ptr, void, (rfftw_plan plan));
FNPTR( rfftw2d_create_plan_ptr, rfftwnd_plan, (int nx, int ny, fftw_direction dir, int flags));
FNPTR( rfftwnd_destroy_plan_ptr, void, (rfftwnd_plan plan));
FNPTR( rfftwnd_one_real_to_complex_ptr, void, (rfftwnd_plan p, fftw_real *in, fftw_complex *out));
FNPTR( rfftwnd_one_complex_to_real_ptr, void, (rfftwnd_plan p, fftw_complex *in, fftw_real *out));
FNPTR( rfftw_one_ptr, void, (rfftw_plan plan, fftw_real *in, fftw_real *out));

  /* Now, define a bunch of defines such that we can access these pointers transparently from the
   \ code below: these are the FFTW_DYNAMIC definitions.
   */
#	define XG_fftw_import_wisdom_from_file	(*fftw_import_wisdom_from_file_ptr)
#	define XG_fftw_export_wisdom_to_string	(*fftw_export_wisdom_to_string_ptr)
#	define XG_fftw_export_wisdom_to_file	(*fftw_export_wisdom_to_file_ptr)
#	define XG_fftw_forget_wisdom	(*fftw_forget_wisdom_ptr)
#	define XG_fftw_malloc	(*fftw_malloc_ptr)
#	define XG_fftw_free	(*fftw_free_ptr)
#	define XG_rfftw_create_plan	(*rfftw_create_plan_ptr)
#	define XG_rfftw_one	(*rfftw_one_ptr)
#	define XG_rfftw_destroy_plan	(*rfftw_destroy_plan_ptr)
#	define XG_rfftw2d_create_plan	(*rfftw2d_create_plan_ptr)
#	define XG_rfftwnd_one_real_to_complex	(*rfftwnd_one_real_to_complex_ptr)
#	define XG_rfftwnd_one_complex_to_real	(*rfftwnd_one_complex_to_real_ptr)
#	define XG_rfftwnd_destroy_plan	(*rfftwnd_destroy_plan_ptr)

#define fftw_xfree(x)	if(x){ XG_fftw_free((x)); (x)=NULL; }

  /* We also need 2 handles to refer to the 2 shared libraries that fftw is comprised of: */
void *lib_fftw= NULL, *lib_rfftw= NULL;

  /* Include dymod.h. This header contains the stuff needed for dealing with shared libraries. */
// #include "dymod.h"

#else

  /* This is the straightforward, !FFTW_DYNAMIC condition. In this case, we just leave it
   \ to the linker/loader to initialise all those functionpointers we have to initialise
   \ in the DYNAMIC case. This means of course that we have to link with the libraries at
   \ build time. And, if they're shared libraries, that they have to be present when we run
   \ the programme. It will depend on the system whether the application will be runnable
   \ without - as long as no fftw's are attempted, some systems will allow execution.
   */

#	define XG_fftw_import_wisdom_from_file	fftw_import_wisdom_from_file
#	define XG_fftw_export_wisdom_to_string	fftw_export_wisdom_to_string
#	define XG_fftw_export_wisdom_to_file	fftw_export_wisdom_to_file
#	define XG_fftw_forget_wisdom	fftw_forget_wisdom
#	define XG_fftw_free	fftw_free
#	define XG_rfftw_create_plan	rfftw_create_plan
#	define XG_rfftw_one	rfftw_one
#	define XG_rfftw_destroy_plan	rfftw_destroy_plan
#	define XG_rfftw2d_create_plan	rfftw2d_create_plan
#	define XG_rfftwnd_one_real_to_complex	rfftwnd_one_real_to_complex
#	define XG_rfftwnd_one_complex_to_real	rfftwnd_one_complex_to_real
#	define XG_rfftwnd_destroy_plan	rfftwnd_destroy_plan
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
#define rfftw_create_plan	XG_rfftw_create_plan
#define ffftw_one	XG_ffftw_one
#define rfftw_destroy_plan	XG_rfftw_destroy_plan
#define rfftw2d_create_plan	XG_rfftw2d_create_plan
#define rfftwnd_one_real_to_complex	XG_rfftwnd_one_real_to_complex
#define rfftwnd_one_complex_to_real	XG_rfftwnd_one_complex_to_real
#define rfftwnd_destroy_plan	XG_rfftwnd_destroy_plan

#if !defined(HAVE_FFTW) || HAVE_FFTW==0
#	undef fftw_malloc
#	undef fftw_free
#	define fftw_malloc	malloc
#	define fftw_free	free
#	define XG_fftw_malloc	fftw_malloc
#	define XG_fftw_free	fftw_free
#	define fftw_xfree(x)	if(x){ XG_fftw_free((x)); (x)=NULL; }
#endif

int ascanf_InitFFTW ( ASCB_ARGLIST ) 
{ ASCB_FRAME_RESULT
  char *home= getenv( "HOME" ), *c, wise_file[512];
  FILE *fp;
  char *w;

	if( FFTW_Initialised || ascanf_SyntaxCheck ){
		*result= 1;
		return(1);
	}
#if defined(HAVE_FFTW) && defined(FFTW_DYNAMIC) && HAVE_FFTW
	  /* load fftw/rfftw here, and initialise function pointers on success 	*/
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, " (loading libfftw.so and librfftw.so... " );
		fflush( StdErr );
	}
#ifdef USE_LTDL
	lib_fftw= lt_dlopenext( "libfftw" );
	lib_rfftw= lt_dlopenext( "librfftw" );
#else
	lib_fftw= dlopen( "libfftw.so", RTLD_NOW|RTLD_GLOBAL);
	lib_rfftw= dlopen( "librfftw.so", RTLD_NOW|RTLD_GLOBAL);
#endif
	if( lib_fftw && lib_rfftw ){
	  int err= 0;
		LOADFUNCTION( lib_fftw, fftw_import_wisdom_from_file_ptr, "fftw_import_wisdom_from_file" );
		LOADFUNCTION( lib_fftw, fftw_export_wisdom_to_string_ptr, "fftw_export_wisdom_to_string" );
		LOADFUNCTION( lib_fftw, fftw_export_wisdom_to_file_ptr, "fftw_export_wisdom_to_file" );
		LOADFUNCTION( lib_fftw, fftw_forget_wisdom_ptr, "fftw_forget_wisdom" );
		LOADFUNCTION( lib_fftw, fftw_malloc_ptr, "fftw_malloc" );
		LOADFUNCTION( lib_fftw, fftw_free_ptr, "fftw_free" );
		LOADFUNCTION( lib_rfftw, rfftw_create_plan_ptr, "rfftw_create_plan" );
		LOADFUNCTION( lib_rfftw, rfftw_one_ptr, "rfftw_one" );
		LOADFUNCTION( lib_rfftw, rfftw_destroy_plan_ptr, "rfftw_destroy_plan" );
		LOADFUNCTION( lib_rfftw, rfftw2d_create_plan_ptr, "rfftw2d_create_plan" );
		LOADFUNCTION( lib_rfftw, rfftwnd_one_real_to_complex_ptr, "rfftwnd_one_real_to_complex" );
		LOADFUNCTION( lib_rfftw, rfftwnd_one_complex_to_real_ptr, "rfftwnd_one_complex_to_real" );
		LOADFUNCTION( lib_rfftw, rfftwnd_destroy_plan_ptr, "rfftwnd_destroy_plan" );
		if( err ){
			if( lib_rfftw ){
				dlclose( lib_rfftw );
			}
			if( lib_fftw ){
				dlclose( lib_fftw );
			}
			fftw_import_wisdom_from_file_ptr= NULL;
			fftw_export_wisdom_to_string_ptr= NULL;
			fftw_export_wisdom_to_file_ptr= NULL;
			fftw_forget_wisdom_ptr= NULL;
			rfftw_create_plan_ptr= NULL;
			rfftw_one_ptr= NULL;
			rfftw_destroy_plan_ptr= NULL;
			rfftw2d_create_plan_ptr= NULL;
			rfftwnd_one_real_to_complex_ptr= NULL;
			rfftwnd_one_complex_to_real_ptr= NULL;
			rfftwnd_destroy_plan_ptr= NULL;
			*result= 0;
			return(0);
		}
	}
	else{
#ifdef USE_LTDL
		fprintf( StdErr, "Error: can't load libfftw.so and/or librfftw.so (%s): no support for FFT!\n", lt_dlerror() );
#else
		fprintf( StdErr, "Error: can't load libfftw.so and/or librfftw.so (%s): no support for FFT!\n", dlerror() );
#endif
		*result= 0;
		return(0);
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, "done) " );
		fflush( StdErr );
	}
#endif
	wise_file[511]= '\0';
	if( (c= getenv( "XG_WISDOM")) ){
		strncpy( wise_file, c, 511 );
	}
	else{
/* 		sprintf( wise_file, "%s/.Preferences/.xgraph/fft_wisdom", home );	*/
		sprintf( wise_file, "%s/fft_wisdom", PrefsDir );
	}
	if( (fp= fopen( wise_file, "r" )) ){
#if defined(HAVE_FFTW) && HAVE_FFTW
		if( fftw_import_wisdom_from_file(fp)!= FFTW_SUCCESS ){
			fprintf( StdErr, "ascanf_InitRFFTW(): read error on wisdom file %s: %s\n", wise_file, serror() );
			*result= 0;
			fclose(fp);
			return(0);
		}
#else
		fprintf( StdErr, "ascanf_InitRFFTW(): you need to compile with -DHAVE_FFTW; fftw routines will not compute\n" );
#endif
		fclose(fp);
#if defined(HAVE_FFTW) && HAVE_FFTW
		if( FFTW_wisdom ){
			xfree( FFTW_wisdom );
		}
		w= fftw_export_wisdom_to_string();
		FFTW_wisdom= strdup(w);
		fftw_free(w);
#endif
	}
	else if( /* c || */ errno!= ENOENT ){
		fprintf( StdErr, "ascanf_InitRFFTW(): read error on wisdom file %s: %s\n", wise_file, serror() );
		*result= 0;
		return(0);
	}
	*result= 1;
	FFTW_Initialised= True;
	return(1);
}

int ascanf_CloseFFTW ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  char *home= getenv( "HOME" ), *c, wise_file[512];
  FILE *fp;
  char *w= NULL;
  int complete;

	if( !FFTW_Initialised || ascanf_SyntaxCheck ){
		*result= 0;
		return(1);
	}
	wise_file[511]= '\0';
	if( (c= getenv( "XG_WISDOM")) ){
		strncpy( wise_file, c, 511 );
	}
	else{
/* 		sprintf( wise_file, "%s/.Preferences/.xgraph/fft_wisdom", home );	*/
		sprintf( wise_file, "%s/fft_wisdom", PrefsDir );
	}
#if defined(HAVE_FFTW) && HAVE_FFTW
	if( ascanf_arguments && args[0] ){
		complete= 1;
	}
	else{
		complete= 0;
	}
	*result= 1;
	if( (w= fftw_export_wisdom_to_string()) && FFTW_wisdom ){
		if( !strcmp( w, FFTW_wisdom) ){
			fftw_free(w);
			if( complete ){
				fftw_forget_wisdom();
				xfree( FFTW_wisdom );
				FFTW_Initialised= False;
			}
			return(1);
		}
	}
	if( (fp= fopen( wise_file, "w" )) ){
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(stored increased wisdom in %s)== ", wise_file );
		}
		fftw_export_wisdom_to_file(fp);
		fclose(fp);
	}
	else /* if( c || errno!= ENOENT ) */{
		fprintf( StdErr, "ascanf_CloseRFFTW(): write error on wisdom file %s: %s\n", wise_file, serror() );
		*result= 0;
		return(0);
	}
	xfree( FFTW_wisdom );
	if( complete ){
		fftw_forget_wisdom();
		FFTW_Initialised= False;
	}
	else{
		FFTW_wisdom= strdup(w);
	}
#else
	fprintf( StdErr, "ascanf_CloseRFFTW(): no FFTW, no wisdom to store...\n" );
	*result= 0;
#endif
#if defined(HAVE_FFTW) && defined(FFTW_DYNAMIC) && defined(NEVER) && HAVE_FFTW
	if( complete ){
		  /* unload fftw/rfftw here, and null function pointers 	*/
		dlclose( lib_rfftw );
		dlclose( lib_fftw );
		fftw_import_wisdom_from_file_ptr= NULL;
		fftw_export_wisdom_to_string_ptr= NULL;
		fftw_export_wisdom_to_file_ptr= NULL;
		fftw_forget_wisdom_ptr= NULL;
		rfftw_create_plan_ptr= NULL;
		rfftw_one_ptr= NULL;
		rfftw_destroy_plan_ptr= NULL;
		rfftw2d_create_plan_ptr= NULL;
		rfftwnd_one_real_to_complex_ptr= NULL;
		rfftwnd_one_complex_to_real_ptr= NULL;
		rfftwnd_destroy_plan_ptr= NULL;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (unloaded libfftw.so and librfftw.so) " );
		}
	}
#endif
	return(1);
}

#if defined(HAVE_FFTW) && HAVE_FFTW
#if defined(DEBUG) && defined(sgi)
#	include <fft.h>
#endif
#endif

#define USE_RFFTWND

#define USE_WISDOM

#ifndef USE_RFFTWND

#define __ascanf_rfftw ascanf_rfftw
#define __ascanf_inv_rfftw ascanf_inv_rfftw
#define __ascanf_convolve_fft ascanf_convolve_fft

int __ascanf_rfftw ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  unsigned long i;
  ascanf_Function *input, *output, *power;
#if defined(HAVE_FFTW) && HAVE_FFTW
  rfftw_plan p;
#endif
#if DEBUG==2
  double *sgcoeff= NULL;
#endif
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0;
		if( !(input= parse_ascanf_address(args[0], _ascanf_array, "ascanf_rfftw", (int) ascanf_verbose, NULL )) || input->iarray ){
			ascanf_emsg= " (invalid input array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= input->N;
#if DEBUG==2
			Resize_ascanf_Array( input, N+2, result );
#endif
		}
		if( !(output= parse_ascanf_address(args[1], _ascanf_array, "ascanf_rfftw", (int) ascanf_verbose, NULL )) || output->iarray ){
			ascanf_emsg= " (invalid output array argument (2)) ";
			ascanf_arg_error= 1;
		}
		else if( output->N!= N ){
			Resize_ascanf_Array( output, N, result );
		}
		if( ascanf_arguments> 2 ){
			if( !(power= parse_ascanf_address(args[2], _ascanf_array, "ascanf_rfftw", (int) ascanf_verbose, NULL )) || power->iarray ){
				ascanf_emsg= " (invalid powerspectrum array argument (3)) ";
				ascanf_arg_error= 1;
			}
			else if( power->N!= N ){
			  /* We only need N/2+1, but we take N for the convenience of the user	*/
				if( Resize_ascanf_Array( power, N, result ) ){
					memset( power->array, 0, sizeof(double)* N );
				}
			}
		}
		else{
			power= NULL;
		}
		if( ascanf_arg_error || !N || !output->array || (power && !power->array) || ascanf_SyntaxCheck ){
			return(0);
		}

#if defined(HAVE_FFTW) && HAVE_FFTW
#ifdef USE_WISDOM
		if( FFTW_Initialised ){
			p= rfftw_create_plan( N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE|FFTW_USE_WISDOM );
		}
		else{
#ifdef FFTW_DYNAMIC
			ascanf_emsg= " (fftw not initialised) ";
			ascanf_arg_error= 1;
			return(0):
#else
			p= rfftw_create_plan( N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_USE_WISDOM );
#endif
		}
#else
		if( FFTW_Initialised ){
			p= rfftw_create_plan( N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE );
		}
		else{
#ifdef FFTW_DYNAMIC
			ascanf_emsg= " (fftw not initialised) ";
			ascanf_arg_error= 1;
			return(0):
#else
			p= rfftw_create_plan( N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE );
#endif
		}
#endif

		rfftw_one( p, input->array, output->array );

		  /* phase can also be calculated: phase= arctan( imag/real )	*/
		if( power ){
		  double *pa= power->array, *out= output->array;
			pa[0]= out[0]*out[0];
			for( i= 1; i< (N+1)/ 2; i++ ){
				pa[i]= out[i]*out[i] + out[N-i]*out[N-i];
			}
			if( N % 2== 0 ){
				pa[N/2]= out[N/2] * out[N/2];
			}
		}

		rfftw_destroy_plan(p);
#endif
#if DEBUG==2
		sgcoeff= (double*) dfft1di( N, NULL );
		dfft1du( -1, N, input->array, 1, sgcoeff );
		xfree( sgcoeff );
#endif

		*result= N;
		return(1);
	}
}

int __ascanf_inv_rfftw ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *input, *output;
#if defined(HAVE_FFTW) && HAVE_FFTW
  unsigned long i;
  rfftw_plan p;
#endif
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0;
		if( !(input= parse_ascanf_address(args[0], _ascanf_array, "ascanf_inv_rfftw", (int) ascanf_verbose, NULL )) || input->iarray ){
			ascanf_emsg= " (invalid input array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= input->N;
		}
		if( !(output= parse_ascanf_address(args[1], _ascanf_array, "ascanf_inv_rfftw", (int) ascanf_verbose, NULL )) || output->iarray ){
			ascanf_emsg= " (invalid output array argument (2)) ";
			ascanf_arg_error= 1;
		}
		else if( output->N!= N ){
			Resize_ascanf_Array( output, N, result );
			if( output->array ){
				memset( output->array, 0, sizeof(double)* N );
			}
		}
		if( ascanf_arg_error || !N || !output->array || ascanf_SyntaxCheck ){
			return(0);
		}
		else{
		  ALLOCA( data, double, N, data_len);

			  /* We have to make a local copy of the inputdata if we want to
			   \ preserve input->array: the reverse RFFTW transform overwrites
			   \ its input data.
			   */
			if( ascanf_arguments> 2 && args[2] ){
			  double scale= 1.0/ N;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (normalising input data) " );
				}
				for( i= 0; i< N; i++ ){
					data[i]= input->array[i]* scale;
				}
			}
			else{
				memcpy( data, input->array, N* sizeof(double) );
			}

#if defined(HAVE_FFTW) && HAVE_FFTW
#ifdef USE_WISDOM
			if( FFTW_Initialised ){
				p= rfftw_create_plan( N, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE|FFTW_USE_WISDOM );
			}
			else{
#ifdef FFTW_DYNAMIC
				ascanf_emsg= " (fftw not initialised) ";
				ascanf_arg_error= 1;
				return(0):
#else
				p= rfftw_create_plan( N, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE|FFTW_USE_WISDOM );
#endif
			}
#else
			if( FFTW_Initialised ){
				p= rfftw_create_plan( N, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE );
			}
			else{
#ifdef FFTW_DYNAMIC
				ascanf_emsg= " (fftw not initialised) ";
				ascanf_arg_error= 1;
				return(0):
#else
				p= rfftw_create_plan( N, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE );
#endif
			}
#endif

			rfftw_one( p, data, output->array );

			rfftw_destroy_plan(p);
#endif

			GCA();
		}

		*result= N;
		return(1);
	}
}

/* convolve_fft[&Data,&Mask,&Output,direction]	*/
int __ascanf_convolve_fft ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  unsigned int i, direction= 1;
  ascanf_Function *Data= NULL, *Mask= NULL, *Output= NULL;
#ifdef DEBUG
  ascanf_Function *Data_sp= NULL, *Mask_sp= NULL;
  static int show_spec= 0;
#endif
  double *mask= NULL, *mask_spec= NULL;
#if defined(HAVE_FFTW) && HAVE_FFTW
  rfftw_plan rp, ip;
#endif
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
	}
	else{
	  unsigned int N= 0, Nm= 0;
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
		if( !(Output= parse_ascanf_address(args[2], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Output->iarray ){
			ascanf_emsg= " (invalid Output array argument (3)) ";
			ascanf_arg_error= 1;
		}
		else if( Output->N!= N ){
			Resize_ascanf_Array( Output, N, result );
		}
		direction= (args[3])? 1 : 0;

#ifdef DEBUG
		if( ascanf_arguments> 4 ){
			if( !(Data_sp= parse_ascanf_address(args[4], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Data_sp->iarray ){
				ascanf_emsg= " (invalid Data_sp array argument (5)) ";
				ascanf_arg_error= 1;
			}
			else if( Data_sp->N!= N+1 ){
				Resize_ascanf_Array( Data_sp, N+1, result );
			}
		}
		if( ascanf_arguments> 5 ){
			if( !(Mask_sp= parse_ascanf_address(args[5], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Mask_sp->iarray ){
				ascanf_emsg= " (invalid Mask_sp array argument (6)) ";
				ascanf_arg_error= 1;
			}
			else if( Mask_sp->N!= N+1 ){
				Resize_ascanf_Array( Mask_sp, N+1, result );
			}
		}
#endif

		if( !(mask= (double*) calloc( N, sizeof(double) )) ||
			!(mask_spec= (double*) calloc( N, sizeof(double) ))
		){
			fprintf( StdErr, " (can't get mask memory (%s)) ", serror() );
			ascanf_arg_error= 1;
		}

		if( ascanf_arg_error || !N || !Nm || !Output->array || !mask || !mask_spec || ascanf_SyntaxCheck ){
			xfree( mask );
			xfree( mask_spec );
			return(0);
		}

#if defined(HAVE_FFTW) && HAVE_FFTW

		{ int j;
			for( i= 0; i< N/2- Nm/2; i++ ){
				mask[i]= Mask->array[0];
			}
			for( j= 0; j< Nm; i++, j++ ){
				mask[i]= Mask->array[j];
			}
			j-= 1;
			for( ; i< N; i++ ){
				mask[i]= Mask->array[j];
			}
		}

		if( FFTW_Initialised ){
			rp= rfftw_create_plan( N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE|FFTW_USE_WISDOM );
			ip= rfftw_create_plan( N, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE|FFTW_USE_WISDOM );
		}
		else{
#ifdef FFTW_DYNAMIC
			ascanf_emsg= " (fftw not initialised) ";
			ascanf_arg_error= 1;
			return(0):
#else
			rp= rfftw_create_plan( N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_USE_WISDOM );
			ip= rfftw_create_plan( N, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE|FFTW_USE_WISDOM );
#endif
		}

		rfftw_one( rp, Data->array, Output->array );
		rfftw_one( rp, mask, mask_spec );
#	ifdef DEBUG
		if( Data_sp && Data_sp->array ){
			memcpy( Data_sp->array, Output->array, (N+0)* sizeof(double) );
		}
		if( Mask_sp && Mask_sp->array ){
			memcpy( Mask_sp->array, mask_spec, (N+0)* sizeof(double) );
		}
#	endif

		  /* Do the convolution. (Deconvolution is for another time...)
		   \ For some reason, rfftw() gives a "+1,-1" spectrum, i.e. for
		   \ a Kronecker delta, a set of +N and -N. I don't really understand
		   \ that at the moment (dusty textbook I guess), but it is clear that
		   \ convolving with a Kron.delta with another one should be a no-operation.
		   \ But the complex multiplication I gleaned from the fftw Tutorial in
		   \ fact performs a rectification: for a kron.delt X kron.delt, this
		   \ effectively results in a kron.delt power spectrum. Needless to say
		   \ that this doesn't yield the correct convolution result (the same kron.delt).
		   \ However, if we take the absolute value of one of the factors (e.g. the 
		   \ mask_spec), we get the correct results in some cases; in other
		   \ cases we'd need to ABS the other factor....
		   \ I need to figure this one out a bit more.... :(
		   \ Ahem.. a kron.delt with N=768 and at x=384 has a purely real spectrum 
		   \ according to rfftw()...
		   */
#define __ABS__	/* ABS	*/
		mask[0]= __ABS__(mask_spec[0])* Output->array[0]/ N;
		mask[N-1]= __ABS__(mask_spec[N-1])* Output->array[N-1]/ N;
		if( N% 2 == 0 ){
			mask[N/2]= __ABS__(mask_spec[N/2])* Output->array[N/2]/ N;
		}
		for( i= 1; i< (N+1)/2; i++ ){
		  double r= (Output->array[i]), im= (Output->array[N-i]),
			  rmm= __ABS__(mask_spec[i]), imm= __ABS__(mask_spec[N-i]);
			mask[i]= (r* rmm- im* imm)/ N;
			mask[N-i]= (r* imm+ im* rmm)/ N;
		}
#	ifdef DEBUG
		memcpy( Output->array, mask, N* sizeof(double) );
		if( !show_spec ){
			rfftw_one( ip, mask, Output->array );
		}
#	else
		rfftw_one( ip, mask, Output->array );
#	endif
		rfftw_destroy_plan(ip);
		rfftw_destroy_plan(rp);
#endif
		xfree( mask );
		xfree( mask_spec );

		GCA();

		*result= N;
		return(1);
	}
}

/* unfinished:	*/
int ascanf_WrapArray ( ASCB_ARGLIST ) 
{ ASCB_FRAME_SHORT
  ascanf_Function *af;
  int i, N;
	*result= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( !(af= parse_ascanf_address(args[0], _ascanf_array, "ascanf_BitReverseArray", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
		return(0);
	}
	N= af->N;
	if( af->iarray ){
	}
	else if( af->array ){
	  ALLOCA( a, double, N, alen);
		for( i= 1; i<= N/2; i+= 2 ){
			a[i]= af->array[i];
			a[i+1]= af->array[N-i-1];
		}
		memcpy( af->array, a, N* sizeof(double) );
	}
	GCA();
	*result= take_ascanf_address( af );
	return(1);
}

#else

#if defined(HAVE_FFTW) && HAVE_FFTW
double *do_rfftw( unsigned long N, double *input, double *output, double *power )
{ fftw_real *in= (fftw_real*) input;
  fftw_complex *out= (fftw_complex*) output;
  rfftwnd_plan p;
  unsigned long i;
#ifdef USE_WISDOM
	if( FFTW_Initialised ){
		p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE|FFTW_USE_WISDOM );
	}
	else{
#ifdef FFTW_DYNAMIC
			ascanf_emsg= " (fftw not initialised) ";
			ascanf_arg_error= 1;
			return(0);
#else
		p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_USE_WISDOM );
#endif
	}
#else
	if( FFTW_Initialised ){
		p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE );
	}
	else{
		p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE );
	}
#endif
	rfftwnd_one_real_to_complex( p, in, out );
	if( power ){
	  double scale= (double)N * (double)N;
		for( i= 0; i<= N/ 2; i++ ){
			power[i]= (out[i].re*out[i].re + out[i].im*out[i].im)/ scale;
		}
	}
	rfftwnd_destroy_plan(p);
	return( output );
}
#endif

static unsigned long last_data_size= 0;

int ascanf_rfftw ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *input, *output, *power;
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	else{
	  unsigned long N= 0, NN= 0;
		if( !(input= parse_ascanf_address(args[0], _ascanf_array, "ascanf_rfftw", (int) ascanf_verbose, NULL )) || input->iarray ){
			ascanf_emsg= " (invalid input array argument (1)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= input->N;
			if( input->malloc!= XG_fftw_malloc || input->free!= XG_fftw_free ){
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= XG_fftw_malloc;
				ascanf_array_free= XG_fftw_free;
				Resize_ascanf_Array( input, N, result );
			}
			NN= 2* (N/ 2+ 1);
		}
		if( !(output= parse_ascanf_address(args[1], _ascanf_array, "ascanf_rfftw", (int) ascanf_verbose, NULL )) || output->iarray ){
			ascanf_emsg= " (invalid output array argument (2)) ";
			ascanf_arg_error= 1;
		}
		else if( output->N!= NN || output->malloc!= XG_fftw_malloc || output->free!= XG_fftw_free ){
			Resize_ascanf_Array_force= True;
			ascanf_array_malloc= XG_fftw_malloc;
			ascanf_array_free= XG_fftw_free;
			Resize_ascanf_Array( output, NN, result );
		}
		if( ascanf_arguments> 2 ){
			if( !(power= parse_ascanf_address(args[2], _ascanf_array, "ascanf_rfftw", (int) ascanf_verbose, NULL )) || power->iarray ){
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
		if( ascanf_arg_error || !N || !output->array || (power && !power->array) || ascanf_SyntaxCheck ){
			return(0);
		}

#if defined(HAVE_FFTW) && HAVE_FFTW

		last_data_size= N;

		do_rfftw( N, input->array, output->array, (power)? power->array : NULL );
		output->value= output->array[ (input->last_index= 0) ];
		if( output->accessHandler ){
			AccessHandler( output, "rfftw", level, ASCB_COMPILED, AH_EXPR, NULL );
		}

		  /* phase can also be calculated: phase= arctan( imag/real )	*/
		if( power ){
			power->last_index= 0;
			power->value= power->array[0];
			if( power->accessHandler ){
				AccessHandler( power, "rfftw", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}

#endif

		*result= NN;
		return(1);
	}
}

int ascanf_inv_rfftw ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *input, *output;
  int i;
#if defined(HAVE_FFTW) && HAVE_FFTW
  rfftwnd_plan p;
#endif
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0, NN;
		if( !(input= parse_ascanf_address(args[0], _ascanf_array, "ascanf_inv_rfftw", (int) ascanf_verbose, NULL )) || input->iarray ){
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
			  /* wrong anyway...: */
			NN= 0;
		}
		if( !(output= parse_ascanf_address(args[1], _ascanf_array, "ascanf_inv_rfftw", (int) ascanf_verbose, NULL )) || output->iarray ){
			ascanf_emsg= " (invalid output array argument (2)) ";
			ascanf_arg_error= 1;
		}
		if( output ){
			if( 2*(output->N/2+1)!= input->N ){
				if( 2*(last_data_size/2+1)== input->N ){
					fprintf( StdErr,
						"\n### inv_rfftw: output array size %d does not match size of complex input array size %d;\n"
						  "###            using matching last known data size %d!!\n",
						output->N, input->N, last_data_size
					);
					NN= last_data_size;
				}
				else{
					ascanf_emsg= " (can't determine required size of the real, re-transformed data) ";
					ascanf_arg_error= 1;
				}
			}
			if( output->N!= NN || output->malloc!= XG_fftw_malloc || output->free!= XG_fftw_free ){
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= XG_fftw_malloc;
				ascanf_array_free= XG_fftw_free;
				Resize_ascanf_Array( output, NN, result );
				if( output->array ){
					memset( output->array, 0, sizeof(double)* NN );
				}
			}
		}
		if( ascanf_arg_error || !N || !NN || !output->array || ascanf_SyntaxCheck ){
			return(0);
		}
		else{
		  double *data;

			if( !(data= XG_fftw_malloc( N* sizeof(double) )) ){
				ascanf_emsg= " (fftw_malloc allocation error) ";
				ascanf_arg_error= 1;
				return(0);
			}

			  /* We have to make a local copy of the inputdata if we want to
			   \ preserve input->array: the reverse RFFTW transform overwrites
			   \ its input data.
			   */
			if( ascanf_arguments> 2 && args[2] ){
			  double scale= 1.0/ NN;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (normalising input data) " );
				}
				for( i= 0; i< N; i++ ){
					data[i]= input->array[i]* scale;
				}
			}
			else{
				memcpy( data, input->array, N* sizeof(double) );
			}

#if defined(HAVE_FFTW) && HAVE_FFTW
#ifdef USE_WISDOM
			if( FFTW_Initialised ){
				p= rfftw2d_create_plan( 1, NN, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE|FFTW_USE_WISDOM );
			}
			else{
#ifdef FFTW_DYNAMIC
				ascanf_emsg= " (fftw not initialised) ";
				ascanf_arg_error= 1;
				XG_fftw_free(data);
				return(0);
#else
				p= rfftw2d_create_plan( 1, NN, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE|FFTW_USE_WISDOM );
#endif
			}
#else
			if( FFTW_Initialised ){
				p= rfftw2d_create_plan( 1, NN, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE );
			}
			else{
#ifdef FFTW_DYNAMIC
			ascanf_emsg= " (fftw not initialised) ";
			ascanf_arg_error= 1;
			XG_fftw_free(data);
			return(0):
#else
				p= rfftw2d_create_plan( 1, NN, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE );
#endif
			}
#endif

			rfftwnd_one_complex_to_real( p, (fftw_complex*) data, output->array );
			output->value= output->array[ (output->last_index= 0) ];
			if( output->accessHandler ){
				AccessHandler( output, "inv_rfftw", level, ASCB_COMPILED, AH_EXPR, NULL );
			}

			rfftwnd_destroy_plan(p);
#endif

			XG_fftw_free(data);
			GCA();
		}

		*result= N;
		return(1);
	}
	return(0);
}

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
  rfftwnd_plan rp, ip;
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
				ascanf_array_malloc= XG_fftw_malloc;
				ascanf_array_free= XG_fftw_free;
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
		if( !(Output= parse_ascanf_address(args[2], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Output->iarray ){
			ascanf_emsg= " (invalid Output array argument (3)) ";
			ascanf_arg_error= 1;
		}
		else if( Output->N!= NN || Output->malloc!= XG_fftw_malloc || Output->free!= XG_fftw_free ){
			if( FFTW_Initialised ){
				Resize_ascanf_Array_force= True;
				ascanf_array_malloc= XG_fftw_malloc;
				ascanf_array_free= XG_fftw_free;
			}
			Resize_ascanf_Array( Output, NN, result );
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

		if( FFTW_Initialised ){
			rp= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE|FFTW_USE_WISDOM );
			ip= rfftw2d_create_plan( 1, N, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE|FFTW_USE_WISDOM );
		}
		else{
#ifdef FFTW_DYNAMIC
			ascanf_emsg= " (fftw not initialised) ";
			ascanf_arg_error= 1;
			xfree( mask );
			xfree( mask_spec );
			return(0);
#else
			rp= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_USE_WISDOM );
			ip= rfftw2d_create_plan( 1, N, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE|FFTW_USE_WISDOM );
#endif
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
			rfftwnd_one_real_to_complex( rp, (double*) mask, mask_spec );
		}
		else{
			memcpy( mask_spec, Mask_sp->array, (NN/2)* sizeof(fftw_complex) );
		}

		rfftwnd_one_real_to_complex( rp, Data->array, (fftw_complex*) Output->array );

		if( Data_sp && Data_sp->array ){
			memcpy( Data_sp->array, Output->array, (NN)* sizeof(double) );
			Data_sp->value= Data_sp->array[ (Data_sp->last_index= 0) ];
			if( Data_sp->accessHandler ){
				AccessHandler( Data_sp, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}
		if( Mask && Mask_sp && Mask_sp->array ){
			memcpy( Mask_sp->array, mask_spec, (NN)* sizeof(double) );
			Mask_sp->value= Mask_sp->array[ (Mask_sp->last_index= 0) ];
			if( Mask_sp->accessHandler ){
				AccessHandler( Data_sp, "convolve_fft", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}

		{ fftw_complex *dat= (fftw_complex*) Output->array;
		  int sign= 1;
			  /* multiply odd elements with -1 to get "the centre in the centre"
			   \ (and not at the 2 edges)
			   */
			if( direction ){
				for( i= 0; i<= N/2; i++ ){
					mask[i].re= sign* (dat[i].re* mask_spec[i].re- dat[i].im* mask_spec[i].im)/ N;
					mask[i].im= sign* (dat[i].re* mask_spec[i].im+ dat[i].im* mask_spec[i].re)/ N;
					sign*= -1;
				}
			}
			else{
			  double b2;
				for( i= 0; i<= N/2; i++ ){
					b2= N*( mask_spec[i].re* mask_spec[i].re + mask_spec[i].im* mask_spec[i].im );
					if( b2 ){
						mask[i].re= sign* (dat[i].re* mask_spec[i].re+ dat[i].im* mask_spec[i].im)/ b2;
						mask[i].im= sign* (-dat[i].re* mask_spec[i].im+ dat[i].im* mask_spec[i].re)/ b2;
					}
					else{
					  /* Define deconvolution with NULL mask as NOOP:	*/
						mask[i]= dat[i];
					}
					sign*= -1;
				}
			}
		}
#	ifdef DEBUG
		memcpy( Output->array, mask, NN* sizeof(double) );
		if( !show_spec ){
			rfftwnd_one_complex_to_real( ip, mask, Output->array );
		}
#	else
		rfftwnd_one_complex_to_real( ip, mask, Output->array );
#	endif
		rfftwnd_destroy_plan(ip);
		rfftwnd_destroy_plan(rp);
#endif
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

#endif

double _convolve( double *Data, double *Mask, double *Output, int Start, int End, int Nm )
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
/* 			if( k>= 0 && k< N ){	*/
/* 				accum+= Mask[j]* Data[k];	*/
/* 			}	*/
			accum+= Mask[j]* v;
		}
		Output[i]= accum;
	}
}

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
	  size_t i, j, k, nm, N= 0, Nm= 0;
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
		if( !(Output= parse_ascanf_address(args[2], _ascanf_array, "ascanf_convolve", (int) ascanf_verbose, NULL )) || Output->iarray ){
			ascanf_emsg= " (invalid Output array argument (3)) ";
			ascanf_arg_error= 1;
		}
		else if( Output->N!= (direction)? NN : N ){
			Resize_ascanf_Array( Output, ((direction)? NN : N), result );
		}
		if( ascanf_arguments> 4 && ASCANF_TRUE(args[4]) ){
			nan_handling= (int) args[4];
		}
		else{
			nan_handling= False;
		}

		if( ascanf_arg_error || !N || !Nm || !Output->array || ascanf_SyntaxCheck ){
			return(0);
		}

		GCA();

		if( direction ){
		  ALLOCA( data, double, NN, data_len);
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
			if( nan_handling ){
			  size_t sL= 0, eL, sR= NN, eR;
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
										}
										break;
									}
								}
							}
							else{
								  /* Best guess we can do is to fill the gap with the first non-NaN value we have at hand */
#ifdef DEBUG
								fprintf( StdErr, "'left' NaN hole from %lu-%lu; padding with %g\n", sL+1, eL-1, data[eL] );
#endif
								for( j= sL+1; j< eL && j< NN; j++ ){
									data[j]= data[eL];
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
								}
							}
						}
					}
					i= sR+1;
				}
			}
			_convolve( data, Mask->array, Output->array, 0, NN, Nm );
			memmove( Output->array, &Output->array[padding], N*sizeof(double) );
			Resize_ascanf_Array( Output, N, result );
			if( nan_handling ){
				for( i= 0; i< N; i++ ){
					if( isNaN(Data->array[i]) ){
						Output->array[i]= Data->array[i];
					}
				}
			}
			GCA();
		}
		else{
			return( ascanf_convolve_fft( ASCB_ARGUMENTS ));
			  /* The below is a rather failed attempt to a direct deconvolution...
			for( i= 0; i< N; i++ ){
			  double w= 0;
				accum= Data->array[i];
				for( j= 0; j< Nm; j++ ){
					k= i+ j- nm;
					if( k!= j ){
						if( k>= 0 && k< N ){
							if( Mask->array[j] ){
								accum-= Data->array[k]/ Mask->array[j];
							}
						}
					}
					else{
						w= Mask->array[j];
					}
				}
				Output->array[i]= (w)? accum/w : accum;
			}
			   */
		}
		Output->value= Output->array[ (Output->last_index= 0) ];
		if( Output->accessHandler ){
			AccessHandler( Output, "convolve", level, ASCB_COMPILED, AH_EXPR, NULL );
		}

		*result= N;
		return(1);
	}
	return(0);
}


/* end (r)fftw functions */

static ascanf_Function fourconv_Function[] = {
	{ "SavGolayCoeffs", ascanf_SavGolCoeffs, 4, NOT_EOF_OR_RETURN,
		"SavGolayCoeffs[&coeffs_ret,<halfwidth>[,<pol_order>=4[,<deriv_order>=0]]]: determine the coefficients\n"
		" for a Savitzky-Golay convolution filter. This returns the same coefficients as SavGolayInit[] would, without\n"
		" additional action (mainly allocations) on a data set. The coefficients are returned in <coeffs_ret> which should\n"
		" be an array of floats (a pointer to this array is also returned).\n"
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
	},
	{ "SavGolayFinished", ascanf_SavGolFinished, 1, NOT_EOF_OR_RETURN, "SavGolayFinished: de-allocate Savitzky-Golay filtering resources"},

#if defined(HAVE_FFTW) && HAVE_FFTW
	{ "InitFFTW", ascanf_InitFFTW, 1, NOT_EOF_OR_RETURN, "InitFFTW: optionally initialise FFTW:\n"
		" load wisdom from ~/.Preferences/.xgraph/wisdom or $XG_WISDOM\n"
		" By calling InitFFTW, RFFTW routines will use FFTW_MEASURE|FFTW_USE_WISDOM, otherwise FFTW_ESTIMATE|FFTW_USE_WISDOM\n"
		" (see FFTW documentation)\n"
	},
	{ "CloseFFTW", ascanf_CloseFFTW, 1, NOT_EOF_OR_RETURN, "CloseFFTW or CloseFFTW[1]: optionally terminate FFTW:\n"
		" store increased wisdom to ~/.Preferences/.xgraph/wisdom or $XG_WISDOM\n"
		" Passing True causes all wisdom to be forgotten afterwards\n"
	},
	{ "$fftw-planner-level", NULL, 2, _ascanf_variable,
		"$fftw-planner-level: ignored for FFTW 2.x, as there is only one measuring planner level:\n"
		" 0: FFTW_MEASURE: planning takes some time, subsequent execution is considerably faster.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "rfftw", ascanf_rfftw, 3, NOT_EOF_OR_RETURN,
		"rfftw[<input_p>,<output_p>[,<power_p>]]: do an FFT (using RFFTW) of the array pointed to by <input_p>\n"
		" storing the transform in <output_p> and optionally the power spectrum in <power_p>\n"
		" All must be double arrays; if necessary, <output_p> and <power_p> are resized to input_p->N,\n"
		" even though <power_p> will have only input_p->N/2+1 relevant elements.\n"
	},
	{ "inv_rfftw", ascanf_inv_rfftw, 3, NOT_EOF_OR_RETURN,
		"inv_rfftw[<input_p>,<output_p>[,<normalise>]]: do an inverse FFT (using RFFTW) of the array pointed to by <input_p>\n"
		" storing the transform in <output_p>, optionally normalising <input_p> first (division by input_p->N).\n"
		" All must be double arrays; <output_p> is resized to input_p->N if necessary.\n"
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
	{ "rfftw", ascanf_rfftw, 3, NOT_EOF_OR_RETURN, "rfftw[<input_p>,<output_p>[,<power_p>]]: not operational" },
	{ "inv_rfftw", ascanf_inv_rfftw, 3, NOT_EOF_OR_RETURN, "inv_rfftw[<input_p>,<output_p>[,<normalise>]]: not operational" },
	{ "convolve_fft", ascanf_convolve_fft, 6, NOT_EOF_OR_RETURN, "convolve_fft[&Data,&Mask,&Output,direction[,&data_spec[,&mask_spec]]]: not operational" },
#endif
	{ "convolve", ascanf_convolve, 5, NOT_EOF_OR_RETURN,
		"convolve[&Data,&Mask,&Output,direction[,nan_handling]]: (de)convolve the array pointed to by <&Data> by <&Mask>\n"
		" storing the transform in <Output>.\n"
		" <direction> is >=0 for convolution, <0 for deconvolution\n"
		" All must be double arrays; <Output> is resized to Data->N if necessary.\n"
		" This routine uses direct convolution\n"
	},

};
static int fourconv_Functions= sizeof(fourconv_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
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
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= new;
	}
	called+= 1;
}

static int initialised= False;

DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;

	if( !DMBase ){
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
		XGRAPH_FUNCTION(ascanf_Arrays2Regular_ptr, "ascanf_Arrays2Regular");
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, new->name, new->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( new, new->name );
		add_ascanf_functions( fourconv_Function, fourconv_Functions, "fourconv::initDyMod()" );
		initialised= True;
	}
	new->libHook= NULL;
	new->libname= XGstrdup( "DM-fourconv" );
	new->buildstring= XGstrdup(XG_IDENTIFY());
	new->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" various Fourier and Convolution functions.\n"
		" Uses libfftw. Provides the Savitzky-Golay filter.\n"
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
		if( (r= ascanf_Arrays2Regular( XG_fftw_malloc, XG_fftw_free )) ){
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
