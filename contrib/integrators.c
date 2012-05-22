/* integrators.c (C) 2002 R.J.V. Bertin.
 \ Ascanf routines for performing integration.
 \ Version Tue Nov 19 01:15:02 CET 2002
 */

#include "config.h"
IDENTIFY( "Integration ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

  /* Get the dynamic module definitions:	*/
#include "dymod.h"

extern FILE *StdErr;

#include "copyright.h"

  /* Include a whole bunch of headerfiles. Not all of them are strictly necessary, but if
   \ we want to have fdecl.h to know all functions we possibly might want to call, this
   \ list is needed.
   \ (Or, we might use fdecl_stubs.h)
   */
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "SS.h"

#include "NaN.h"

#include "xgraph.h"
#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
#include "ascanfc-table.h"
#include "compiled_ascanf.h"

#define SIGN(x)		(((x)<0)?-1:1)
#define SIGN2(a,b)	((b) >= 0.0 ? fabs(a) : -fabs(a))
#ifndef MIN
#	define MIN(a, b)	((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#	define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc).
   */
#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;
	void (*Polynom_Interpolate_ptr)(double *xa, double *ya, int n, double x, double *y, double *dy);
	int (*fascanf_ptr)( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], struct Compiled_Form **form );
	int (*Destroy_Form_ptr)( struct Compiled_Form **form );

#	define Polynom_Interpolate (*Polynom_Interpolate_ptr)
#	define fascanf	(*fascanf_ptr)
#	define Destroy_Form	(*Destroy_Form_ptr)


double (*SplineColNr_ptr)( int col, double x );
#define SplineColNr (*SplineColNr_ptr)
double (*SplineValue_ptr)( double x, double *orgX, double *orgY, int porgYN, double *coeffs, int upwlint );
#define SplineValue (*SplineValue_ptr)
double (*splint_ptr)( double *xa, double *ya, double *y2a, int n, double x, double *y );
#define splint	(*splint_ptr)

/* ---------------------- "Discrete" integrators ---------------------------- */

/* This routine computes the nth stage of refinement of an extended trapezoidal rule. func is input
 \ as a pointer to the function to be integrated between limits a and b, also input. When called with
 \ n=1, the routine returns the crudest estimate of the integral.
 \ Subsequent calls with n=2,3,...
 \ (in that sequential order) will improve the accuracy by adding 2n-2 additional interior points.
 */
static double trapezoid_stage( double (*func)(double), double a, double b, int n )
{ double x,tnm,sum,del;
  static double s;
  int it,j;
	if( n == 1 ){
		return( s=0.5*(b-a)*( (*func)(a)+ (*func)(b)) );
	}
	else{
		for( it= 1, j= 1; j< n-1; j++ ){
			it <<= 1;
		}
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for( sum= 0.0, j= 1; j<= it; j++, x+=del ){
			sum += (*func)(x);
		}
		s=0.5*(s+(b-a)*sum/tnm);
		return(s);
	}
}

/* Returns the integral of the function func from a to b. The argument precision can be set to the
 \ desired fractional accuracy and maxloops so that 2 to the power JMAX-1 is the maximum allowed
 \ number of steps. Integration is performed by the trapezoidal rule.
 */
double integrate_trapezoid_precision( double (*func)(double), double a, double b, double precision, int maxloops )
{ int j;
 double s,olds;
	  /* Any number that is unlikely to be the average of the function at its endpoints will do here. */
	olds = -1.0e30;
	if( precision<= 0 ){
		precision= 1.0e-5;
	}
	if( maxloops<= 0 ){
		maxloops= 20;
	}
	for( j= 1; j<= maxloops; j++ ){
		s=trapezoid_stage(func,a,b,j);
		if( j > 5 ){
			if( fabs(s-olds) < precision *fabs(olds) || (s == 0.0 && olds == 0.0) ){
				return(s);
			}
		}
		olds=s;
	}
	fprintf( StdErr, "integrate_trapezoid_precision(): no convergence in %d loops\n", j );
	  /* return the calculated value anyway. */
	return(s);
}

/* Returns the integral of the function func from a to b. The argument precision can be set to the
 \ desired fractional accuracy and JMAX so that 2 to the power JMAX-1 is the maximum allowed
 \ number of steps. Integration is performed by Simpson's rule.
 */
double integrate_simpson_precision( double (*func)(double), double a, double b, double precision, int maxloops )
{ int j;
  double s,st,ost,os;
	ost = os = -1.0e30;
	if( precision<= 0 ){
		precision= 1.0e-6;
	}
	if( maxloops<= 0 ){
		maxloops= 20;
	}
	for( j= 1; j<= maxloops; j++ ){
		st=trapezoid_stage(func,a,b,j);
		s=(4.0*st-ost)/3.0;
		if( j > 5 ){
			if( fabs(s-os) < precision*fabs(os) || (s == 0.0 && os == 0.0) ){
				return(s);
			}
		}
		os=s;
		ost=st;
	}
	fprintf( StdErr, "integrate_simpson_precision(): no convergence in %d loops\n", j );
	  /* return the calculated value anyway. */
	return(s);
}

/* Returns the integral of the function func from a to b. Integration is performed by Romberg's
 \ method of order 2K, where, e.g., K==extrapol=2 is Simpson's rule.
 */
double integrate_romberg_precision( double (*func)(double), double a, double b, double precision, int maxloops, int extrapol )
{ double ss,dss;
  int j;
	if( precision<= 0 ){
		precision= 1.0e-6;
	}
	if( maxloops<= 0 ){
		maxloops= 20;
	}
	if( extrapol<= 0 ){
		extrapol= 5;
	}
	{ ALLOCA( s, double, maxloops+1, s_len );
	  ALLOCA( h, double, maxloops+2, h_len );
		h[1]=1.0;
		for( j= 1; j<= maxloops; j++ ){
			s[j]=trapezoid_stage(func,a,b,j);
			if( j >= extrapol ){
				Polynom_Interpolate(&h[j-extrapol],&s[j-extrapol],extrapol,0.0,&ss,&dss);
				if( fabs(dss) <= precision*fabs(ss) ){
					return(ss);
				}
			}
			h[j+1]=0.25*h[j];
		}
		fprintf( StdErr, "integrate_romberg_precision(): no convergence in %d loops\n", j );
		  /* return the calculated value anyway. */
		GCA();
		return(ss);
	}
}

/* ---------------------- Runge-Kutta integrators ---------------------------- */

/* Given the values y[1..n] and their derivatives dydx[1..n] at <x>, use the
 \ fourth-order Runge-Kutta method to advance the solution over interval <h> and return the
 \ updated variables in yout[1..n], which may be equal to y.
 \ The routine derivs(x,y,dydx) should return derivatives dydx at x.
 */
static void RungeKutta4_fix(double *y, double *dydx, int n, double x, double h, double *yout,
	void (*derivs)(double, double*, double*,void* data), void* deriv_data )
{ int i, n1= n+1;
  double xh, hh, h6;
  ALLOCA(dy3, double, n1, dym_len);
  ALLOCA(dy2, double, n1, dyt_len);
  ALLOCA(yi, double, n1, yt_len);

	hh=h*0.5;
	h6=h/6.0;
	xh=x+hh;
	for( i=1; i<=n; i++ ){
		yi[i]= y[i] + hh*dydx[i];
	}
	(*derivs)(xh,yi,dy2, deriv_data);
	for( i=1; i<=n; i++ ){
		yi[i]=y[i] + hh*dy2[i];
	}
	(*derivs)(xh,yi,dy3, deriv_data);
	for( i=1; i<=n; i++ ){
		yi[i]=y[i]+h*dy3[i];
		dy3[i] += dy2[i];
	}
	(*derivs)(x+h,yi,dy2, deriv_data);
	for( i=1; i<=n; i++ ){
		yout[i]= y[i] + h6*(dydx[i] + dy2[i] + 2.0*dy3[i]);
	}
	GCA();
}

/* Starting from initial values Y0[1..Nvar] at <xStart> use fourth-order Runge-Kutta
 \ to advance Nstep equal steps to <xEnd>. The routine derivs(x,v,dvdx)
 \ evaluates derivatives. Results are stored in the global variables return_Y[1..Nvar][1..Nstep+1]
 \ and return_X[1..Nstep+1].
 */
void RungeKutta4( double *Y0, int Nvar, double xStart, double xEnd, int Nstep,
	void (*derivs)(double, double*, double*, void* data), void *deriv_data, double *return_X, double **return_Y)
{ int i,k, Nvar1= Nvar+1;
  double x,h;
  ALLOCA(vals,double,Nvar1, v_len);
  ALLOCA(newvals, double, Nvar1, vout_len);
  ALLOCA(dv, double, Nvar1, dv_len);

	h=(xEnd-xStart)/Nstep;
	if( xStart+h == xStart ){
		fprintf( StdErr, "RungeKutta4(): Step size (%s-%s)/%d == %s too small\n",
			d2str(xEnd,0,0), d2str(xStart,0,0), Nstep, d2str(h,0,0)
		);
		return;
	}
	for( i=1; i<= Nvar; i++ ){
		vals[i]=Y0[i];
		return_Y[i][1]=vals[i];
	}
	return_X[1]=xStart;
	x=xStart;
	for( k=1; k<= Nstep; k++ ){
		(*derivs)(x,vals,dv, deriv_data);
		RungeKutta4_fix(vals,dv,Nvar,x,h,newvals,derivs, deriv_data);
		x += h;
		return_X[k+1]=x;
		for( i= 1; i<= Nvar; i++ ){
			vals[i]=newvals[i];
			return_Y[i][k+1]=vals[i];
		}
	}
	GCA();
}

#define SAFETY	0.9
#define PGROW	-0.2
#define PSRHINK	-0.25
#define ERRCON	1.889568e-4 /* (5/SAFETY) raised to the power (1/PGROW) */

/* Given values for n variables y[1..n] and their derivatives dydx[1..n] x, use
 \ the fifth-order Cash-Karp Runge-Kutta method to advance the solution over an interval h
 \ and return the updated variables as yout[1..n]. Also return an estimate of the local
 \ truncation error in yout using the embedded fourth-order method. The user supplies the routine
 \ derivs(x,y,dydx), which returns derivatives dydx at x.
 */
static void CashKarp_RungeKutta(double *y, double *dydx, int n, double x, double h, double *yout,
	double *yerr, void (*derivs)(double, double*, double*, void *data), void *deriv_data)
{ int i, n1= n+1;
  static double a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,b21=0.2,
	  b31=3.0/40.0,b32=9.0/40.0,b41=0.3,b42 = -0.9,b43=1.2,
	  b51 = -11.0/54.0, b52=2.5,b53 = -70.0/27.0,b54=35.0/27.0,
	  b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
	  b64=44275.0/110592.0,b65=253.0/4096.0,c1=37.0/378.0,
	  c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
	  dc5 = -277.00/14336.0;
  double dc1=c1-2825.0/27648.0,dc3=c3-18575.0/48384.0,
	  dc4=c4-13525.0/55296.0,dc6=c6-0.25;
  ALLOCA(ak2, double, n1, ak2_len);
  ALLOCA(ak3, double, n1, ak3_len);
  ALLOCA(ak4, double, n1, ak4_len);
  ALLOCA(ak5, double, n1, ak5_len);
  ALLOCA(ak6, double, n1, ak6_len);
  ALLOCA(ytemp, double, n1, ytemp_len);

	for( i= 1; i<= n; i++ ){
		ytemp[i]=y[i]+b21*h*dydx[i];
	}
	(*derivs)(x+a2*h,ytemp,ak2, deriv_data);
	for( i= 1; i<= n; i++){
		ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
	}
	(*derivs)(x+a3*h,ytemp,ak3, deriv_data);
	for( i= 1; i<= n; i++){
		ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
	}
	(*derivs)(x+a4*h,ytemp,ak4, deriv_data);
	for( i= 1; i<= n; i++){
		ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
	}
	(*derivs)(x+a5*h,ytemp,ak5, deriv_data);
	for( i= 1; i<= n; i++){
		ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
	}
	(*derivs)(x+a6*h,ytemp,ak6, deriv_data);
	for( i= 1; i<= n; i++ ){
		yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
	}
	for( i= 1; i<= n; i++){
		yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);
	}
	GCA();
}

/* Fifth-order Runge-Kutta step with monitoring of local truncation error to ensure accuracy and
 \ adjust stepsize. Input are the dependent variable vector y[1..n] and its derivative dydx[1..n]
 \ at the starting value of the independent variable x. Also input are the stepsize to be attempted
 \ hIntended, the required accuracy Accuracy, and the vector yscal[1..n] against which the error is
 \ scaled. On output, y and x are replaced by their new values, hAttained is the stepsize that was
 \ actually accomplished, and hNext is the estimated next stepsize. derivs is the user-supplied
 \ routine that computes the right-hand side derivatives.
 */
static void RungeKutta5_adap(double *y, double *dydx, int n, double *x, double hIntended, double Accuracy,
	double *yscal, double *hAttained, double *hNext,
	void (*derivs)(double, double*, double*, void *data), void* deriv_data)
{ int i, under= 0, n1= n+1;
  double errmax,h,htemp,xnew, hI;
  ALLOCA(yerr, double, n1, yerr_len);
  ALLOCA(ytemp, double, n1, ytemp_len);
	h= hI= hIntended;
	if( !hIntended ){
		GCA();
		return;
	}
	for( ; under< 25 && !ascanf_escape; ){
		CashKarp_RungeKutta(y,dydx,n,*x,h,ytemp,yerr,derivs,deriv_data);
		errmax=0.0;
		for( i=1; i<= n; i++ ){
			errmax= MAX(errmax,fabs(yerr[i]/yscal[i]));
		}
		errmax /= Accuracy;
		if( errmax <= 1.0 ){
			break;
		}
		htemp=SAFETY*h*pow(errmax,PSRHINK);
		h= ( (h >= 0.0)? MAX(htemp,0.1*h) : MIN(htemp,0.1*h) );
		xnew=(*x)+h;
		if( xnew == *x ){
			  /* RJVB: attempt a larger step... */
			under+= 1;
			hI*= 10 * under;
			if( hI< 1 ){
				fprintf( StdErr, "RungeKutta5_adap(): stepsize underflow #%d at x=%s+step=%s == %s\n",
					under, d2str(*x,0,0), d2str(h,0,0), d2str(xnew,0,0)
				);
				h= hI;
				xnew+= h;
			}
			ascanf_check_event( "RungeKutta5_adap()" );
		}
		else{
			under= 0;
		}
	}
	if( under ){
		fprintf( StdErr, "RungeKutta5_adap(): stepsize underflow at x=%s+step=%s == %s\n",
			d2str(*x,0,0), d2str(h,0,0), d2str(xnew,0,0)
		);
	}
	if( errmax > ERRCON ){
		*hNext=SAFETY*h*pow(errmax,PGROW);
	}
	else{
		*hNext=5.0*h;
	}
	*x += (*hAttained=h);
	memcpy( &y[1], &ytemp[1], n* sizeof(double) );
	GCA();
}

/* Runge-Kutta driver with adaptive stepsize control. Integrate starting values yStart[1..nVar]
 \ from x1 to x2 with accuracy Accuracy. h1 should
 \ be set as a guessed first stepsize, hMin as the minimum allowed stepsize (can be zero). On
 \ output nOK and nBAD are the number of good and bad (but retried and fixed) steps taken, and
 \ yStart is replaced by values at the end of the integration interval. derivs is the user-supplied
 \ routine for calculating the right-hand side derivative. maxsteps is the max. number of steps to be taken.
 \ The remainder are for intermediate results. If kmax!=0
 \ results are stored at approximate intervals <interval> in the arrays return_X[1..N], return_Y[1..nVar]
 \ [1..N], where N is output by RungeKutta4_adaptive.
 \ memory allocations return_X[1..kmax] and return_Y[1..nVar][1..kmax] for the arrays, should be in
 \ the calling program.
 */
void RungeKutta4_adaptive(double *yStart, int nVar, double x1, double x2, double Accuracy, double h1,
	double hMin, int *nOK, int *nBAD, void (*derivs)(double, double*, double*, void *data), void *deriv_data,
	int maxsteps, int kmax, int *N,
	double *return_X, double **return_Y, double interval
)
{ int nstp,i, kount, nVar1= nVar+1;
  double xsav,x,hNext,hAttained,h;
  ALLOCA(yscal, double, nVar1, yscal_len);
  ALLOCA(y, double, nVar1, y_len);
  ALLOCA(dydx, double, nVar1, dydx_len);

	x=x1;
	if( h1 ){
		h1= 1e-3;
	}
	h=SIGN2(h1,x2-x1);
	*nOK = (*nBAD) = kount = 0;
	for( i= 1; i<= nVar; i++ ){
		y[i]=yStart[i];
	}
	if( kmax > 0 ){
		xsav=x-interval*2.0;
	}
	for( nstp= 1; nstp<= maxsteps && !ascanf_escape; nstp++ ){
		(*derivs)(x,y,dydx,deriv_data);
		for( i= 1; i<= nVar; i++){
			yscal[i]=fabs(y[i])+fabs(dydx[i]*h)+DBL_EPSILON;
		}
		if( kmax > 0 && kount < kmax-1 && fabs(x-xsav) > fabs(interval) ){
			return_X[++kount]=x;
			for( i= 1; i<= nVar; i++ ){
				return_Y[i][kount]=y[i];
			}
			xsav=x;
		}
		if( (x+h-x2)*(x+h-x1) > 0.0 ){
			h=x2-x;
		}
		RungeKutta5_adap(y,dydx,nVar,&x,h,Accuracy,yscal,&hAttained,&hNext,derivs, deriv_data);
		if( hAttained == h ){
			++(*nOK);
		}
		else{
			++(*nBAD);
		}
		if( (x-x2)*(x2-x1) >= 0.0 ){
			for( i= 1; i<= nVar; i++ ){
				yStart[i]=y[i];
			}
			if( kmax ){
				return_X[++kount]=x;
				for( i= 1; i<= nVar; i++ ){
					return_Y[i][kount]=y[i];
				}
			}
			if( N ){
				*N= kount;
			}
			GCA();
			return;
		}
		if( fabs(hNext) <= hMin ){
			fprintf( StdErr, "RungeKutta4_adaptive(): step %s smaller than minimal step %s\n",
				d2str(hNext,0,0), d2str(hMin,0,0)
			);
			hNext= hMin;
			ascanf_check_event( "RungeKutta4_adaptive()" );
		}
		h=hNext;
	}
	fprintf( StdErr, "RungeKutta4_adaptive(): Too many steps (%d) -- aborted\n", nstp );
	if( N ){
		*N= kount;
	}
	GCA();
}

/* --------------------------------------------------------------------------------------------------- */


/* ----------------------------- ascanf callback handlers etc. --------------------------------------- */

/* Spline-Adaptive-RungeKutta[ column, Y0, xLow, xHigh, Accuracy, initStep, minStep, maxSteps[, return_samples, &return_X, &return_Y]]	*/

static int SARK_column= -1;
static void SARK_deriv_col( double x, double *dum, double *derivative )
{
	derivative[1]= SplineColNr( SARK_column, x );
/* 	fprintf( StdErr, "deriv(%d,%s,%s)=%s\n", SARK_column, d2str(x,0,0), d2str(dum[1],0,0), d2str(derivative[1],0,0) );	*/
}

static double *aX= NULL, *aY= NULL, *CScoeffs= NULL;
static int CSN= -1;
static void SARK_deriv_spline( double x, double *dum, double *derivative )
{
	splint( &aX[-1], &aY[-1], &CScoeffs[-1], CSN, x, &derivative[1] );
}

int ascanf_Spline_Adaptive_RungeKutta( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double Y0[2], xLow, xHigh, Accuracy, initStep, minStep;
  int maxSteps, return_samples, argset=0;
  ascanf_Function *afRX= NULL, *afRY= NULL, *afX= NULL, *afY= NULL, *afCoeffs= NULL;
	ascanf_arg_error= False;
	if( args && ascanf_arguments>= 8 ){
		if( (afX= parse_ascanf_address( args[0], _ascanf_array, "ascanf_Spline_Adaptive_RungeKutta",
				(int) ascanf_verbose, NULL ))
		){
		    if( afX->iarray ){
			    ascanf_emsg= " (X array should contain doubles) ";
			    ascanf_arg_error= 1;
		    }
		    else{
				if( !(afY= parse_ascanf_address( args[1], _ascanf_array, "ascanf_Spline_Adaptive_RungeKutta",
						(int) ascanf_verbose, NULL ))
					|| afY->iarray || afY->N != afX->N
				){
					ascanf_emsg= " (2nd argument must be a double array with the same length as the X array) ";
					ascanf_arg_error= 1;
				}
				if( !(afCoeffs= parse_ascanf_address( args[2], _ascanf_array, "ascanf_Spline_Adaptive_RungeKutta",
						(int) ascanf_verbose, NULL ))
					|| afCoeffs->iarray || afCoeffs->N != afX->N
				){
					ascanf_emsg= " (3rd argument must be a double array with the same length as the X array) ";
					ascanf_arg_error= 1;
				}
		    }

		    if( ascanf_arg_error ){
			    return( !ascanf_arg_error );
		    }
		    aX = afX->array;
		    aY = afY->array;
		    CScoeffs = afCoeffs->array;
		    CSN = afX->N;
		    argset= 2;
		}
		else if( args[0]>= 0 && args[0]<= MAXINT ){
			SARK_column= (int) args[0];
			SplineColNr( SARK_column, 0 );
			if( ascanf_arg_error ){
				fprintf( StdErr, "ascanf_Spline_Adaptive_RungeKutta(): invalid column %s: %s\n",
					SARK_column, ascanf_emsg
				);
			}
		}
		Y0[1]= args[1+argset];
		xLow= args[2+argset];
		xHigh= args[3+argset];
		Accuracy= args[4+argset];
		initStep= args[5+argset];
		minStep= args[6+argset];
		maxSteps= args[7+argset];
		if( ascanf_arguments> 8+argset ){
			if( ascanf_arguments< 11+argset ){
				ascanf_emsg= " (should specify all of return_samples, &return_X, &return_Y) ";
				ascanf_arg_error= True;
			}
			else{
				if( args[8+argset]< 0 || args[8+argset]> MAXINT ){
					return_samples= 0;
				}
				else{
					return_samples= (int) args[8+argset];
				}
				afRX= parse_ascanf_address( args[9+argset], _ascanf_array, "ascanf_Spline_Adaptive_RungeKutta", (int) ascanf_verbose, NULL );
				afRY= parse_ascanf_address( args[10+argset], _ascanf_array, "ascanf_Spline_Adaptive_RungeKutta", (int) ascanf_verbose, NULL );
				if( !(return_samples && afRX && afRY) || afRX->iarray || afRY->iarray ){
					fprintf( StdErr,
						"ascanf_Spline_Adaptive_RungeKutta(): either of return_samples, return_X and/or return_Y "
						"is/are NULL/invalid: ignoring all.\n"
					);
					return_samples= 0;
					afRX= afRY= NULL;
				}
				if( afRX && !ascanf_SyntaxCheck ){
				  double dum;
					Resize_ascanf_Array( afRX, return_samples, &dum );
					Resize_ascanf_Array( afRY, return_samples, &dum );
				}
				if( !afRX->array || !afRY->array ){
					fprintf( StdErr,
						"ascanf_Spline_Adaptive_RungeKutta(): couldn't resize either return_X and/or return_Y (%s)",
						serror()
					);
					return_samples= 0;
					afRX= afRY= NULL;
				}
			}
		}
	}
	else{
		ascanf_arg_error= True;
	}
	if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
	  int nOK= 0, nBAD= 0, N= 0;
	  double *return_X, *return_Y[2], rX[2], rY[2];
		if( afRX && afRY ){
			return_X= &(afRX->array[-1]);
			return_Y[1]= &(afRY->array[-1]);
		}
		else{
			return_X= &(rX[-1]);
			return_Y[1]= &(rY[-1]);
			return_samples= 0;
		}
		return_Y[0]= NULL;
		Y0[0]= 0;
		switch( argset ){
			case 0:
				RungeKutta4_adaptive( Y0, 1, xLow, xHigh, Accuracy, initStep,
					minStep, &nOK, &nBAD, SARK_deriv_col, NULL,
					maxSteps, return_samples, &N,
					return_X, return_Y, (xHigh-xLow)/return_samples
				);
				break;
			case 2:
				RungeKutta4_adaptive( Y0, 1, xLow, xHigh, Accuracy, initStep,
					minStep, &nOK, &nBAD, SARK_deriv_spline, NULL,
					maxSteps, return_samples, &N,
					return_X, return_Y, (xHigh-xLow)/return_samples
				);
				aX = aY = CScoeffs = NULL;
				CSN = -1;
				break;
		}
		*result= Y0[1];
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (integrated %s%s in %d good and %d bad-but-corrected steps [init. %s, min. %s], obtained %d samples of %d) ",
				(argset==0)? "column " : "",
				(afY)? afY->name : ad2str( SARK_column, d3str_format, NULL), nOK, nBAD,
				ad2str( initStep, d3str_format, 0 ), ad2str( minStep, d3str_format, 0 ),
				N, return_samples
			);
		}
		else if( N> return_samples ){
			fprintf( StdErr, "ascanf_Spline_Adaptive_RungeKutta(): more samples (%d) returned than requested (%d)?!\n",
				N, return_samples
			);
		}
		if( afRX ){
		  double dum;
			Resize_ascanf_Array( afRX, N, &dum );
			Resize_ascanf_Array( afRY, N, &dum );
			if( !afRX->array || !afRY->array ){
				fprintf( StdErr,
				    "ascanf_Spline_Adaptive_RungeKutta(): couldn't resize either return_X and/or return_Y (%s)",
				    serror()
			    );
			}
		}
	}
	return(!ascanf_arg_error);
}

typedef struct SARK_data{
	double *orgX, *orgY;
	double *coeffs;
	int porgYN;
} SARK_data;

static void SARK_deriv2( double x, double *dum, double *derivative, SARK_data *data )
{
	derivative[1]= SplineValue( x, &(data->orgX[-1]), &(data->orgY[-1]), data->porgYN, &(data->coeffs[-1]), 0 );
}

static int EulerArrays ( ASCB_ARGLIST, int nan_resets, char *caller )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  ascanf_Function *sum= NULL, *vals, *t;
	  double ival= 0;
		if( ASCANF_TRUE(args[0]) ){
			sum= parse_ascanf_address( args[0], _ascanf_array, caller, (int) ascanf_verbose, NULL );
		}
		vals= parse_ascanf_address( args[1], _ascanf_array, caller, (int) ascanf_verbose, NULL );
		t= parse_ascanf_address( args[2], _ascanf_array, caller, (int) ascanf_verbose, NULL );
		if( ascanf_arguments> 3 ){
			ival= args[3];
		}
		if( vals && t && vals->N == t->N ){
		  int ok= 0, nnstart= 0, i;
		  double av_dt= 0, prevT, T;
			if( sum ){
				Resize_ascanf_Array( sum, vals->N, result );
				if( sum->iarray ){
					memset( sum->iarray, (int) ival, sizeof(sum->iarray[0]) * sum->N );
				}
				else{
					for( i= 0; i< sum->N; i++ ){
						set_NaN( sum->array[i] );
					}
				}
			}
			while( !ok && nnstart< vals->N ){
			  double v= ASCANF_ARRAY_ELEM(vals,nnstart);
				T= ASCANF_ARRAY_ELEM(t,nnstart);
				if( NaN(v) || NaN(T) ){
					nnstart+= 1;
				}
				else{
					ok= 1;
				}
			}
			if( ASCANF_ARRAY_ELEM(vals,nnstart) ){
				for( i = nnstart+1, prevT = ASCANF_ARRAY_ELEM(t,nnstart) ; i < t->N ; i++ ){
					T = ASCANF_ARRAY_ELEM(t,i);
					if( !NaN(T) ){
						av_dt += T - prevT;
						prevT = T;
					}
				}
				av_dt /= t->N - nnstart - 1;
				  /* we try to do something useful with a non-zero vals[0], even if t[0]==0
				   \ we'll assume that the sampled process has been going on for a while at the same (average) sampling rate.
				   */
/* 				prevT= -av_dt;	*/
				  /* 20070126: T - prevT should never be negative... */
				prevT = ASCANF_ARRAY_ELEM(t,nnstart) - av_dt;
				i = nnstart;
			}
			else{
				prevT= ASCANF_ARRAY_ELEM(t,nnstart+1);
				i= nnstart+1;
			}
			if( ascanf_verbose ){
				fprintf( StdErr, " (y=%s, x=%s d(%s)=%g, first included element: %d, initial value= %g+%g=%g)== ",
					vals->name, t->name, t->name, av_dt, i, ival, ASCANF_ARRAY_ELEM(vals,i), ival+ASCANF_ARRAY_ELEM(vals,i)
				);
			}
			if( sum && i ){
				ASCANF_ARRAY_ELEM_SET(sum,i-1,ival);
			}
			{ double n_ival;
				set_NaN(n_ival);
				for( *result= ival; i< vals->N; i++ ){
				  double v= ASCANF_ARRAY_ELEM(vals,i);
					T= ASCANF_ARRAY_ELEM(t,i);
					if( NaN(v) || NaN(T) ){
						if( nan_resets ){
							n_ival= ival;
							set_NaN(*result);
						}
					}
					else{
						if( nan_resets && !NaN(n_ival) ){
							*result= n_ival;
							set_NaN(n_ival);
						}
						else{
							*result+= ASCANF_ARRAY_ELEM(vals,i) * (T - prevT);
						}
						prevT= T;
					}
					if( sum ){
						ASCANF_ARRAY_ELEM_SET(sum,i,*result);
					}
				}
			}
		}
		else{
			ascanf_emsg= " (arrays must be of equal size) ";
			ascanf_arg_error= 1;
		}
		return( !ascanf_arg_error );
	}
}

int ascanf_EulerArrays ( ASCB_ARGLIST )
{
	return( EulerArrays( ASCB_ARGUMENTS, False, "ascanf_EulerArrays" ) );
}

int ascanf_EulerArrays_NaNResets ( ASCB_ARGLIST )
{
	return( EulerArrays( ASCB_ARGUMENTS, True, "ascanf_EulerArrays_NaNResets" ) );
}


/* ----------------------------- library interface --------------------------------------- */


static ascanf_Function integrators_Function[] = {
	{ "Spline-Adaptive-RungeKutta", ascanf_Spline_Adaptive_RungeKutta, 13, NOT_EOF_OR_RETURN,
		"Spline-Adaptive-RungeKutta[column,initial-value,xLow,xHigh,Accuracy,initStep,minStep,maxSteps[,return-samples,&return-X,&return-Y]]\n"
		"Spline-Adaptive-RungeKutta[&X,&Y,&SplineCoeffs,initial-value,xLow,xHigh,Accuracy,initStep,minStep,maxSteps[,return-samples,&return-X,&return-Y]]\n"
		" The first form attempts to integrate column <column> of the currently splined dataset over its independent (spline) variable,\n"
		" the second form takes the integrand data directly from &X, &Y and &SplineCoeffs.\n"
		" This is done using a Runge-Kutta4 integrator with adaptive stepsize, over the interval xLow to xHigh.\n"
		" Initial guess for the interval is initStep; it can get as small as minStep. No more than maxSteps steps\n"
		" are taken. The integral is returned. When <return-samples> and the array pointers return-X and return-Y\n"
		" are all given, the routine will in addition fill the arrays with at most <return-samples> integrated values\n"
		" (return-Y) at the 'sampled' independent values (return-X), with an interval of approx. (xHigh-xLow)/return-samples.\n"
		" For the first form, SplineInit[] must have been called, and the specified column must have been included!\n"
		" For the second form, X, Y and SplineCoeffs can be prepared e.g. by Spline-Resample[]\n"
	},
	{ "EulerSum", ascanf_EulerArrays, 4, NOT_EOF_OR_RETURN,
		"EulerSum[&cumsum, &vals, &t[, initial-value]]: calculate the cumulative sum of vals[i] * delta(t[i]).\n"
		" If <cumsum> is given (not 0 or NaN) and an array, the running sum is stored there.\n"
		" NB: it is assumed that the process was already being sampled before t[0], so a non-zero vals[0]\n"
		" will be added to the cumsum as vals[0] * average-delta-t! \n"
	},
	{ "EulerSum-NaNResets", ascanf_EulerArrays_NaNResets, 4, NOT_EOF_OR_RETURN,
		"EulerSum-NaNResets[&cumsum, &vals, &t[, initial-value]]: calculate the cumulative sum of vals[i] * delta(t[i]).\n"
		" The difference with EulerSum[] is that the integrator is reset to the (default) initial-value whenever\n"
		" vals[i] and/or t[i] is a NaN. This is done in such a way that for such vals[i] and/or t[i], cumsum[i] is set\n"
		" to NaN, while cumsum[j]=initial-value with the smallest j>i for which vals[j]!=NaN && t[j]!=NaN.\n"
	},
};

static int integrators_Functions= sizeof(integrators_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= integrators_Function;
  static char called= 0;
  int i, warn= True;
  char buf[64];

	for( i= 0; i< integrators_Functions; i++, af++ ){
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
			if( Check_Doubles_Ascanf( af, label, warn ) ){
				warn= False;
			}
		}
		af->dymod= new;
	}
	called+= 1;
}

static int initialised= False;
static Compiled_Form *SplineColNr_hook= NULL;

DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;

	if( !DMBase ){
	  DyModLists *splines=NULL;
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

		  /* 20040924: Make sure a library is loaded containing the Spline routines:
		   \ that will give access to the SplineColNr 'internal' routine, too.
		   \ 20051214: this needs to be done on all platforms!
		   */
		Auto_LoadDyMod_LastPtr(NULL, -1, "SplineInit", &splines );

		XGRAPH_FUNCTION(Polynom_Interpolate_ptr, "Polynom_Interpolate");
		XGRAPH_FUNCTION( fascanf_ptr, "fascanf" );
		XGRAPH_FUNCTION( Destroy_Form_ptr, "Destroy_Form" );
		if( splines ){
			DYMOD_FUNCTION(splines, SplineColNr_ptr, "SplineColNr");
			DYMOD_FUNCTION(splines, SplineValue_ptr, "SplineValue");
			DYMOD_FUNCTION(splines, splint_ptr, "splint");
		}
		else{
			fprintf( StdErr, "Error loading a splines dymod, can't obtain address for SplineColNr &c\n" );
			return( DM_Error );
		}

		  /* now "fix" the function we want, so no attempt will be made to auto-unload the (auto-loaded) library containing it: */
		{ int n= 1;
		  char expr[256];
		  double dummy;
			snprintf( expr, sizeof(expr), "IDict[ verbose[ Spline[0%c0]%c \"" __FILE__ "\"] ]", ascanf_separator, ascanf_separator );
			fascanf( &n, expr, &dummy, NULL, NULL, NULL, &SplineColNr_hook );
			if( !SplineColNr_hook ){
				fprintf( StdErr,
					"%s::initDyMod(): failed to 'lock' the needed SplineColNr function: proceed with toes crossed!\n",
					__FILE__
				);
			}
		}
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );

	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( integrators_Function, integrators_Functions, "integrators::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-integrators" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A library with routines for integrating over points in a dataset.\n"
		" Currently only provides an adaptive RungeKutta4 integrator that integrates\n"
		" Spline[<column>,indep-x] with respect to indep-x.\n"
	);

	called= True;

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
	  int r= remove_ascanf_functions( integrators_Function, integrators_Functions, force );
		if( force || r== integrators_Functions ){
			for( i= 0; i< integrators_Functions; i++ ){
				integrators_Function[i].dymod= NULL;
			}
			initialised= False;
			xfree( target->libname );
			xfree( target->buildstring );
			xfree( target->description );
			ret= target->type= DM_Unloaded;
			if( SplineColNr_hook ){
				Destroy_Form( &SplineColNr_hook );
			}
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
			fprintf( SE, " -- refused: variables are in use (remove_ascanf_functions() removed %d out of %d)\n",
				r, integrators_Functions
			);
		}
	}
	return(ret);
}

