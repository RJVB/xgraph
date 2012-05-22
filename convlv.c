#include <stdio.h>
#include "xgALLOCA.h"

extern FILE *StdErr;

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
void convlv(double data[], unsigned long n, double respns[], unsigned long m, int isign, double ans[])
{ void realft(double data[], unsigned long n, int isign);
  void twofft(double data1[], double data2[], double fft1[], double fft2[], unsigned long n);
  unsigned long i, no2;
  double dum, mag2;
  ALLOCA( fft, double, (n<<1)+1, fft_len);

	for( i= 1; i<= (m-1)/2; i++){
	  /* Put respns in array of length n.	*/
		respns[n+1-i]= respns[m+1-i];
	}
	for( i=(m+3)/2; i<= n-(m-1)/2; i++){
	  /* Pad with zeros.	*/
		respns[i]= 0.0;
	}
	twofft(data, respns, fft, ans, n);	/* FFT both at once.	*/
	no2= n>> 1;
	for( i= 2; i<= n+2; i+= 2 ){
		if( isign >= 0 ){
			ans[i-1]= (fft[i-1]*(dum= ans[i-1])-fft[i]*ans[i])/no2;	/*  Multiply FFTs to convolve.	*/
			ans[i]= (fft[i]*dum+fft[i-1]*ans[i])/no2;
		}
		else if( isign == -1 ){
			if( (mag2=SQR(ans[i-1],double)+SQR(ans[i],double)) == 0.0){
				fprintf( StdErr, "Deconvolving at response zero in convlv");
			}
			ans[i-1]= (fft[i-1]*(dum= ans[i-1])+fft[i]*ans[i])/ mag2/ no2;	/* Divide FFTs to deconvolve.	*/
			ans[i]= (fft[i]*dum-fft[i-1]*ans[i])/ mag2/ no2;
		}
	}
	ans[2]= ans[n+1];	/* Pack last element with first for realft.	*/
	realft(ans, n,-1);	/* Inverse transform back to time domain.	*/
	GCA();
}
