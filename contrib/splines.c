#include "config.h"
IDENTIFY( "Splines ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

  /* get the ascanf definitions:	*/
#include "ascanf.h"
  /* If we want to be able to access the "expression" field in the callback argument, we need compiled_ascanf.h .
   \ If we don't include it, we will just get 'theExpr= NULL' ... (this applies to when -DDEBUG)
   */
#include "compiled_ascanf.h"
#include "ascanfc-table.h"

#include <float.h>

#define DYMOD_MAIN
#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;
	StAtIc char* (*Find_LabelsListLabel_ptr)( LabelsList *llist, int column );
	StAtIc double (*CurveLen_ptr)( LocalWin *wi, int idx, int pnr, int update, int Signed, double *lengths );
	StAtIc double (*ErrorLen_ptr)( LocalWin *wi, int idx, int pnr, int update );
#	define Find_LabelsListLabel	(*Find_LabelsListLabel_ptr)
#	define CurveLen	(*CurveLen_ptr)
#	define ErrorLen	(*ErrorLen_ptr)

void eliminate_NaNs( double *x, double *y, int n )
  /* 20050523: for large n (1000000 or possibly even smaller), we can run into crashes. Avoid using alloca() c.s. */
{ /* _XGALLOCA(yy, double, n+1, yylen); */
  int i, N= 0;
  double *yy= NULL;
	  /* 20050515: we're working with Pascal-style arrays, so we need to go to <=n !! */
	for( i=1; i<= n; i++ ){
	  int j, J;
	  /* NaN handling. We're going to suppose that there are none in the x,
	   \ as the preparing routines ought to have eliminated those.
	   */
		if( NaN(y[i]) ){
		  /* If the point currently under scrutiny is a NaN itself, do a linear
		   \ intrapolation between the two non-NaN values we will find:
		   */
			if( !yy ){
				if( !(yy= (double*) malloc( (n+1) * sizeof(double) )) ){
					fprintf( StdErr, " (allocation problem: no NaN elimination! %s) ", serror() );
					return;
				}
				  /* rather than doing copying of all non-NaN elements, make yy a copy of y at once,
				   \ and then only touch the appropriate (NaN) elements.
				   */
				memcpy( &yy[1], &y[1], n*sizeof(double) );
			}
			for( j=i+1; j<= n && NaN(y[j]); j++ );
			for( J=i-1; J>= 1 && NaN(y[J]); J-- );
			if( J> 0 && j<= n ){
				yy[i]= y[J]+ (x[i] - x[J])* ( (y[j]-y[J])/ (x[j]-x[J]) );
				N+= 1;
			}
			else{
				yy[i]= y[i];
			}
		}
/* 		else{	*/
/* 			yy[i]= y[i];	*/
/* 		}	*/
	}
	if( N ){
		  /* 20050515: copy the appropriate portion: */
		memcpy( &y[1], &yy[1], n*sizeof(double) );
		xfree(yy);
	}
}

void spline( double *x, double *y, int n, double yp1, double ypn, double *y2)
/*
 \ Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e., y i = f(xi), with
 \ x1<x2< ...<xN , and given values yp1 and ypn for the first derivative of the interpolating
 \ function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
 \ the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
 \ ypn are equal to 1e30 or larger, the routine is signaled to set the corresponding boundary
 \ condition for a natural spline, with zero second derivative on that boundary.
 \ NOTE: arrays are 1 based!!
 */
{ int i,k;
  double p,qn,sig,un;
  double *u= NULL;

	if( n >= 2 ){
		u= (double*) calloc( n+1, sizeof(double) );
		if( !u ){
			return;
		}
	}
	else{
		return;
	}

	eliminate_NaNs( x, y, n );
	if( yp1 > 0.99e30 ){
	  /* The lower boundary condition is set either to be "natural"
	   \ or else to have a specified first derivative.
	   */
		y2[1]= u[1]= 0.0;
	}
	else{
	  int j;
		for( j=2; j< n && ( NaN(x[j]) || NaN(y[j]) ); j++ );
		y2[1]= -0.5;
		u[1]= (3.0/(x[j]-x[1]))*((y[j]-y[1])/(x[j]-x[1])-yp1);
	}
	for( i=2; i< n; i++ ){
	  int j, J;
	  double xii, yii, yJ, yj;
	  /* This is the decomposition loop of the tridiagonal al-
	   \ gorithm. y2 and u are used for tem-
	   \ porary storage of the decomposed
	   \ factors.
	   */
		j= i+1, J= i-1;
		  /* Update y[i] here! This is not the original data in SplineSet anyway, so we can. */
		xii= x[i], yii= y[i];
		yj= y[j];
		yJ= y[J];
		sig= (xii-x[J])/(x[j]-x[J]);
		p= sig*y2[J]+2.0;
		y2[i]= (sig-1.0)/p;
		u[i]= (yj-yii)/(x[j]-xii) - (yii-yJ)/(xii-x[J]);
		u[i]= (6.0*u[i]/(x[j]-x[J])-sig*u[J])/p;
	}
	if( ypn > 0.99e30 ){
	  /* The upper boundary condition is set either to be "natural"	*/
		qn= un= 0.0;
	}
	else{
	  int J;
	  /* or else to have a specified first derivative.	*/
		for( J=n-1; J> 1 && ( NaN(x[J]) || NaN(y[J]) ); J-- );
		qn= 0.5;
		un=(3.0/(x[n]-x[J]))*(ypn-(y[n]-y[J])/(x[n]-x[J]));
	}
	y2[n]= (un-qn*u[n-1])/(qn*y2[n-1]+1.0);
	  /* This is the backsubstitution loop of the tridiagonal algorithm.	*/
	for( k= n-1; k>= 1; k--){
		y2[k]= y2[k]*y2[k+1]+u[k];
	}
	xfree( u );
	GCA();
}

/* RJVB: as spline() above, but now do a piecewise linear interpolation: */
void pwlint_coeffs( double *x, double *y, int n, double *coeff)
{ int i;
	eliminate_NaNs( x, y, n );
	for( i= 1; i< n; i++ ){
	  int ii= i, j= i+1;
		if( j<= n ){
			coeff[i]= (y[j] - y[ii]) / (x[j] - x[ii]);
		}
		else{
			set_NaN( coeff[i] );
		}
	}
}

void pwlint(double *xa, double *ya, double *coeff, int n, double x, double *y)
{ int klo,khi,k;
  static int pklo = 2, pklh = 2;
	if( !coeff || n == 1 ){
		set_NaN(*y);
	}
	else{
		if( pklo < 1 || pklo > n ){
			pklo = 2;
		}
		if( pklh < 1 || pklh > n ){
			pklh = 2;
		}
		if( xa[pklo] <= x && xa[pklh] > x ){
			klo = pklo, khi = pklh;
		}
		else if( n > 2 && xa[pklo+1] <= x && xa[pklh+1] > x ){
			klo = pklo+1, khi = pklh+1;
			pklo = klo, pklh = khi;
		}
		else if( xa[pklo-1] <= x && xa[pklh-1] > x ){
			klo = pklo-1, khi = pklh-1;
			pklo = klo, pklh = khi;
		}
		else{
			klo= 1;
			khi= n;
			while( khi-klo > 1 ){
				k= (khi+klo) >> 1;
				if( xa[k] > x ){
					khi= k;
				}
				else{
					klo= k;
				}
			}
			pklo = klo, pklh = khi;
		}
#if 0
		while( klo>=0 && (NaN(xa[klo]) || NaN(ya[klo])) ){
			klo-= 1;
		}
		while( khi<=n && (NaN(xa[khi]) || NaN(ya[khi])) ){
			khi+= 1;
		}
#endif
		  /* klo and khi are now non-NaN values bracketing the input value of x.	*/
		{ double xlo= xa[klo], h= xa[khi]-xlo;
			if( h == 0.0 ){
				fprintf( StdErr, "pwlint was passed a value (%s) not in the X values array\n", ad2str(x, d3str_format,NULL) );
				set_NaN(*y);
			}
			else{
				  /*  The xa's must be distinct.	*/
				*y= (x- xlo)* coeff[klo]+ ya[klo];
			}
		}
	}
}

static int PWLInt= False;

/*
 \ It is important to understand that the program spline is called only once to
 \ process an entire tabulated function in arrays xi and yi . Once this has been done,
 \ values of the interpolated function for any value of x are obtained by calls (as many
 \ as desired) to a separate routine splint (for "spline interpolation"):
 */
void splint(double *xa, double *ya, double *y2a, int n, double x, double *y)
 /*
  \ Given the arrays xa[1..n] and ya[1..n], which tabulate a function (with the xai's in order),
  \ and given the array y2a[1..n], which is the output from spline above, and given a value of
  \ x, this routine returns a cubic-spline interpolated value y.
  */
{ int klo,khi, kl, kh, k;
  double h,b,a;

	if( PWLInt< 0 ){
		pwlint(xa, ya, y2a, n, x, y);
	}
	else{
	  static int pklo = 2, pklh = 2, delta = 1;

		if( !y2a || n == 1 ){
			set_NaN(*y);
			return;
		}

		if( xa[pklo] <= x && xa[pklh] > x ){
			klo = pklo, khi = pklh;
		}
		else if( pklo+delta <= n && xa[pklo+delta] <= x && pklh+delta <= n && xa[pklh+delta] > x ){
			klo = pklo+delta, khi = pklh+delta;
			pklo = klo, pklh = khi;
		}
		else if( pklo-delta > 0 && xa[pklo-delta] <= x && pklh-delta > 0 && xa[pklh-delta] > x ){
			klo = pklo-delta, khi = pklh-delta;
			pklo = klo, pklh = khi;
		}
		else{
			  /*
			   \ We will find the right place in the table by means of
			   \ bisection. This is optimal if sequential calls to this
			   \ routine are at random values of x. If sequential calls
			   \ are in order, and closely spaced, one would do better
			   \ tostoreprevious values ofklo and khi and test if
			   \ they remain appropriate on the next call.
			   \ 20111104: hence the new code above testing pklo and pklh.
			   \ For large arrays, this can be around 4x faster than
			   \ indiscriminate use of bisection. (I've only tested this
			   \ for sequential access through Spline-Resample, though!)
			   */
			klo= 1;
			khi= n;
			while( khi-klo > 1 ){
				k= (khi+klo) >> 1;
				if( xa[k] > x ){
					khi= k;
				}
				else{
					klo= k;
				}
			}
			  /* klo and khi now bracket the input value of x.	*/
			kl= klo, kh= khi;
			{ int d = abs(klo-pklo);
				if( d && d != delta ){
					if( d < (n >>1) ){
						delta = d;
					}
				}
			}
			pklo = klo, pklh = khi;
		}
#if 0
		while( klo>=0 && (NaN(xa[klo]) || NaN(ya[klo])) ){
			klo-= 1;
		}
		while( khi<=n && (NaN(xa[khi]) || NaN(ya[khi])) ){
			khi+= 1;
		}
		  /* RJVB: We use the bracketing input values that are not NaNs. But the coefficients
		   \ have been determined with an y array in which all NaNs had been replaced by linearly
		   \ interpolated values between the bracketing non-NaN's. Thus, we can safely use the
		   \ calculated coefficients, and don't need to go looking for those far away.
		   \ This almost works correctly.
		   */
#else
		  /* spline() will have filtered out all NaNs by the linear interpolation described above. */
#endif
		{ double xhi, xlo;
			h= (xhi= xa[khi])- (xlo= xa[klo]);
			if( h == 0.0 ){
				fprintf( StdErr, "splint was passed a value (%s) not in the X values array\n", ad2str(x, d3str_format,NULL) );
				set_NaN(*y);
			}
			else{
				  /* 	The xa's must be distinct.	*/
				a= (xhi-x)/h;
				b= (x-xlo)/h;
				  /* Cubic spline polynomial is now evaluated.	*/
				*y= a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[kl]+(b*b*b-b)*y2a[kh])*(h*h)/6.0;
			}
		}
	}
}

typedef struct SplineColumns{
	double *orgvals;
	double *spline_coefs, *pwlint_coefs;
} SplineColumns;

static double *spline_xtab= NULL;
/* static double *x_spline_coefs= NULL, *y_spline_coefs= NULL, *e_spline_coefs= NULL;	*/
/* static double *xvals= NULL, *yvals= NULL, *evals= NULL;	*/
static SplineColumns *spline_col= NULL;
static int SplineN, SplineIndepNaNs= 0, SplineIndepCol, spline_colSize= 0, SplineAllCols, SplineIntv= 0;
static DataSet *SplineSet;
static SimpleStats SSIndep;

static double *ascanf_PWLInt;

typedef struct SplineDat{
	double indep;
	double x, y, e;
} SplineDat;

typedef struct SplinePoint{
	double indep;
	double *columns;
} SplinePoint;

int SplineValCompare( double a, double b)
{
	  /* sort NaNs towards the end of the list */
	if( NaN(a) && NaN(b) ){
		return(0);
	}
	else if( a< b || NaN(b) ){
		return( -1 );
	}
	else if( a > b || NaN(a) ){
		return( 1 );
	}
	else{
		return( 0 );
	}
}

int SplineDataCompare( SplineDat *a, SplineDat *b)
{
	return( SplineValCompare( a->indep, b->indep ) );
}

int SplineAllDataCompare( SplinePoint *a, SplinePoint *b)
{
	return( SplineValCompare( a->indep, b->indep ) );
}

static void xfree_SplinePoints( SplinePoint *sp, int N )
{ int i;
	if( !sp ){
		return;
	}
	for( i= 0; i< N; i++ ){
		xfree( sp[i].columns );
	}
	xfree( sp );
}

static SplinePoint *alloc_SplinePoints()
{ int i;
  SplinePoint *sp= NULL;
	if( (sp= (SplinePoint*) calloc( SplineN, sizeof(SplinePoint))) ){
		for( i= 0; i< SplineN; i++ ){
			if( !(sp[i].columns= (double*) calloc( SplineSet->ncols, sizeof(double))) ){
				xfree_SplinePoints( sp, i );
				return( NULL );
			}
		}
	}
	return( sp );
}

static void xfree_SplineColumns( SplineColumns *sc )
{ int i;
	if( !sc ){
		return;
	}
	for( i= 0; i< spline_colSize; i++ ){
		xfree( sc[i].orgvals );
		xfree( sc[i].spline_coefs );
		xfree( sc[i].pwlint_coefs );
	}
	xfree( sc );
	return;
}

#define SPCOL_OK(col)	(\
	spline_col && col>=0 && (SplineAllCols || \
		((col==SplineSet->xcol || col==SplineSet->ycol || col==SplineSet->ecol || col==SplineSet->lcol) && \
		spline_col[col].orgvals && spline_col[col].spline_coefs && (PWLInt<=0 || spline_col[col].pwlint_coefs))\
	) \
)

static SplineColumns *alloc_SplineColumns( int N )
{ int i;
  SplineColumns *sc;
	if( PWLInt<= 0 && spline_col ){
		for( i= 0; i< spline_colSize; i++ ){
			xfree( spline_col[i].pwlint_coefs );
		}
	}
	if( (sc= (SplineColumns*) XGrealloc( spline_col, SplineSet->ncols* sizeof(SplineColumns))) ){
		  /* Make sure that the any newly allocated cells don't point to random memory:	*/
		for( i= spline_colSize; i< SplineSet->ncols; i++ ){
			sc[i].orgvals= NULL;
			sc[i].spline_coefs= NULL;
			sc[i].pwlint_coefs= NULL;
		}
		spline_colSize= SplineSet->ncols;
		for( i= 0; i< spline_colSize; i++ ){
			if( (SplineAllCols || (i==SplineSet->xcol || i==SplineSet->ycol || i==SplineSet->ecol)) ){
				if( !(sc[i].orgvals= (double*) XGrealloc( sc[i].orgvals, N* sizeof(double))) ||
					!(sc[i].spline_coefs= (double*) XGrealloc( sc[i].spline_coefs, N* sizeof(double))) ||
					(PWLInt> 0 && !(sc[i].pwlint_coefs= (double*) XGrealloc( sc[i].pwlint_coefs, N* sizeof(double))))
				){
					xfree_SplineColumns( sc );
					return( NULL );
				}
			}
			else{
				sc[i].orgvals= NULL;
				sc[i].spline_coefs= NULL;
				sc[i].pwlint_coefs= NULL;
			}
		}
	}
	return( sc );
}

void SplineCopyData(int duplicate_handling )
{ int i= 0, j= 0, col, I;
  SplinePoint *Sdat= NULL;
  int SN= SplineN, ok= 0;
  int same= 0;
  double *Ai= NULL, *si= NULL;
  SimpleStats SSi= EmptySimpleStats, SSx= EmptySimpleStats, SSy= EmptySimpleStats, SSe= EmptySimpleStats;
	if(
		!(Sdat= alloc_SplinePoints()) ||
		!(Ai= (double*)calloc( SplineSet->ncols, sizeof(double))) ||
		!(si= (double*)calloc( SplineSet->ncols, sizeof(double)))
	){
		xfree_SplinePoints( Sdat, SN );
		xfree( Ai );
		xfree( si );
		return;
	}
	for( i= 0; i< SplineN; i++ ){
		Sdat[i].indep= spline_xtab[i+1];
		for( col= 0; col< spline_colSize; col++ ){
			Sdat[i].columns[col]= SplineSet->columns[col][i];
		}
	}
	qsort( Sdat, SplineN, sizeof(SplinePoint), (void*) SplineAllDataCompare );
	for( j= SplineN-1; j>= 0 && NaN(Sdat[j].indep); j-- ){
		SplineIndepNaNs+= 1;
	}
	if( SplineIndepNaNs ){
		if( debugFlag ){
			fprintf( StdErr, "SplineCopyData(): %d trailing NaN independent values (snipped)\n",
				SplineIndepNaNs
			);
		}
		SplineN-= SplineIndepNaNs;
	}
	for( I= 1, i= 0, j= 1; i< SplineN-1; I++, i++, j++ ){
		if( Sdat[i].indep!= Sdat[j].indep ){
			ok= 1;
			spline_xtab[I]= Sdat[i].indep;
			for( col= 0; col< SplineSet->ncols; col++ ){
				if( SPCOL_OK(col) ){
					spline_col[col].orgvals[I]= Sdat[i].columns[col];
				}
			}
		}
		else{
		  ALLOCA(N, int, SplineSet->ncols, Nlen );
			spline_xtab[I]= Sdat[i].indep;
			for( col= 0; col< SplineSet->ncols; col++ ){
				si[col]= Sdat[i].columns[col];
				if( !NaN(Sdat[i].columns[col]) ){
					Ai[col]= Sdat[i].columns[col];
					N[col]= 1;
				}
				else{
					Ai[col]= 0;
					N[col]= 0;
				}
			}
			ok= 0;
			do{
				same+= 1;
				i+=1; j+= 1;
				for( col= 0; col< SplineSet->ncols; col++ ){
					if( !NaN( Sdat[i].columns[col] ) ){
						Ai[col]+= Sdat[i].columns[col];
						N[col]+= 1;
					}
				}
			} while( j< SplineN && Sdat[i].indep== Sdat[j].indep /* && sdat[i].indep!= sdat[j].indep */);
			switch( duplicate_handling ){
				case 0:
				default:
					for( col= 0; col< spline_colSize; col++ ){
						if( SPCOL_OK(col) ){
							spline_col[col].orgvals[I]= Ai[col]/ (double)N[col];
						}
					}
					break;
				case 1:
					for( col= 0; col< spline_colSize; col++ ){
						if( SPCOL_OK(col) ){
							spline_col[col].orgvals[I]= si[col];
						}
					}
					break;
				case 2:
					for( col= 0; col< spline_colSize; col++ ){
						if( SPCOL_OK(col) ){
							spline_col[col].orgvals[I]= Sdat[i].columns[col];
						}
					}
					break;
			}
			GCA();
		}
	}
	if( ok ){
		spline_xtab[I]= Sdat[i].indep;
		for( col= 0; col< spline_colSize; col++ ){
			if( SPCOL_OK(col) ){
				spline_col[col].orgvals[I]= Sdat[i].columns[col];
			}
		}
	}
	if( same ){
		if( debugFlag ){
			fprintf( StdErr, "SplineCopyData(): %d identical independent values (averaged over their x,y,e,... vals)\n",
				same
			);
		}
		SplineN-= same;
	}
	if( pragma_unlikely(ascanf_verbose> 1) && SPCOL_OK(SplineSet->xcol) && SPCOL_OK(SplineSet->ycol) && SPCOL_OK(SplineSet->ecol) ){
		SS_Reset_(SSi);
		SS_Reset_(SSx);
		SS_Reset_(SSy);
		SS_Reset_(SSe);
		for( i= 1; i<= SplineN; i++ ){
			  /* ought to filter out the NaNs here too */
			SS_Add_Data_( SSi, 1, spline_xtab[i], 1.0 );
			SS_Add_Data_( SSx, 1, spline_col[SplineSet->xcol].orgvals[i], 1.0);
			SS_Add_Data_( SSy, 1, spline_col[SplineSet->ycol].orgvals[i], 1.0);
			SS_Add_Data_( SSe, 1, spline_col[SplineSet->ecol].orgvals[i], 1.0);
		}
		fprintf( StdErr, "SplineCopyData: indep=\"%s\"\n\tx=\"%s\"\n\ty=\"%s\"\n\te=\"%s\"\n",
			SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSi ),
			SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSx ),
			SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSy ),
			SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSe )
		);
	}
	xfree_SplinePoints( Sdat, SN );
	xfree( Ai ); xfree( si );
	GCA();
}

int ascanf_SplineInit ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, dup_handling= 0, col;
  ascanf_Function *s1= NULL;
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( args[0]>= 0 && args[0]< setNumber ){
		idx= (int) args[0];

	}
	else{
		ascanf_emsg= " (set_nr argument 1 out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( args[1]> -MAXINT && args[1]<= MAXINT ){
		SplineIndepCol= (int) args[1];
		if( ascanf_SyntaxCheck && SplineIndepCol>= AllSets[idx].ncols ){
			fprintf( StdErr, " (Warning: indep_col_nr %d>= Set[%d]->ncols=%d!) ",
				SplineIndepCol, idx, AllSets[idx].ncols
			);
			fflush( StdErr );
		}
	}
	else{
		ascanf_emsg= " (indep_col_nr argument 2 out of range) ";
		return(0);
	}
	if( ascanf_arguments> 2 ){
		dup_handling= (args[2])? 1 : 0;
	}
	if( ascanf_arguments> 3 ){
		SplineAllCols= (args[3])? 1: 0;
	}
	else{
		SplineAllCols= 1;
	}
	if( ascanf_arguments> 4 ){
		if( !(s1= parse_ascanf_address(args[4], 0, "ascanf_SplineInit", ascanf_verbose, NULL )) ||
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( debugFlag ){
				fprintf( StdErr, " (warning: 4th argument is not a valid string, using as <interval>)== " );
			}
			CLIP( args[4], -MAXINT, MAXINT );
			SplineIntv= (int) args[4];
			if( ascanf_arguments> 5 ){
				if( !(s1= parse_ascanf_address(args[5], 0, "ascanf_SplineInit", ascanf_verbose, NULL )) ||
					(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
				){
					fprintf( StdErr, " (warning: 5th argument is not a valid string)== " );
				}
			}
		}
	}
	if( !ascanf_SyntaxCheck ){
	  SimpleStats SSx= EmptySimpleStats, SSy= EmptySimpleStats, SSe= EmptySimpleStats;
		if( idx>= 0 && idx< setNumber && AllSets[idx].numPoints> 0 && SplineIndepCol< AllSets[idx].ncols ){
		  int N;
			SplineSet= &AllSets[idx];
			SplineN= SplineSet->numPoints;
			N= SplineN+ 1;
			errno= 0;
			if( ASCANF_TRUE(*ascanf_PWLInt) ){
				PWLInt= (int) *ascanf_PWLInt;
			}
			else{
				PWLInt= 0;
			}
/* fprintf(stderr, "%s:%d\n", __FILE__, __LINE__); fflush(stderr);	*/
			if( !((spline_col= (SplineColumns*) alloc_SplineColumns( N)) &&
					(spline_xtab= (double*) realloc( spline_xtab, N* sizeof(double) ))
				)
			){
				xfree( spline_xtab );
				xfree_SplineColumns( spline_col );
				fprintf( StdErr, "SplineInit: error getting working memory (%s)\n", serror() );
				if( ascanf_window ){
					xtb_error_box( ascanf_window, "SplineInit: error getting memory", "Failure" );
				}
				*result= 0;
			}
			else{
			  int i;
			  char *xmsg= "", *clabel= NULL, *smsg= "";
				for( i= 0; i<= SplineN; i++ ){
					for( col= 0; col< spline_colSize; col++ ){
						if( SPCOL_OK(col) ){
							spline_col[col].spline_coefs[i]= 0.0;
							if( PWLInt> 0 ){
								spline_col[col].pwlint_coefs[i]= 0.0;
							}
						}
					}
				}
				  /* We actually fit 3 splines; one for x, one for y, and one for error/orientation.
				   \ The x-values used for the 3 fits is:
				   \ *) the point index (select -1)
				   \ *) curve length at each point (select -2)
				   \ *) error length at each point (select -3)
				   \ *) the sum of the 2 previous (select -4)
				   \ *) select>= 0: data column <select>
				   */
				switch( SplineIndepCol ){
					case -1:
						  /* The following line is just to be sure..!	*/
						spline_xtab[0]= i- 1;
						for( i= 1; i<= SplineN; i++ ){
							spline_xtab[i]= i-1;
							for( col= 0; col< spline_colSize; col++ ){
								if( SPCOL_OK(col) ){
									spline_col[col].orgvals[i]= SplineSet->columns[col][i-1];
								}
							}
						}
						xmsg= "point index";
						break;
					case -2:
						if( !ActiveWin || ActiveWin== StubWindow_ptr ){
							fprintf( StdErr, "SplineInit: error: no window active (yet)!\n" );
						}
						else{
							CurveLen(ActiveWin, idx, SplineN, True, False, NULL );
							spline_xtab[0]= CurveLen(ActiveWin, idx, 0, False, False, NULL )-1;
							for( i= 1; i<= SplineN; i++ ){
								spline_xtab[i]= CurveLen(ActiveWin, idx, i-1, False, False, NULL );
							}
							SplineCopyData(dup_handling);
						}
						xmsg= "curve length[index]";
						break;
					case -3:
						if( !ActiveWin || ActiveWin== StubWindow_ptr ){
							fprintf( StdErr, "SplineInit: error: no window active (yet)!\n" );
						}
						else{
							ErrorLen(ActiveWin, idx, SplineN, True );
							spline_xtab[0]= ErrorLen( ActiveWin, idx, 0, False )- 1;
							for( i= 1; i<= SplineN; i++ ){
								spline_xtab[i]= ErrorLen( ActiveWin, idx, i-1, False );
							}
							SplineCopyData(dup_handling);
						}
						xmsg= "error length[index]";
						break;
					case -4:
						if( !ActiveWin || ActiveWin== StubWindow_ptr ){
							fprintf( StdErr, "SplineInit: error: no window active (yet)!\n" );
						}
						else{
							CurveLen(ActiveWin, idx, SplineN, True, False, NULL );
							ErrorLen(ActiveWin, idx, SplineN, True );
							spline_xtab[0]= CurveLen( ActiveWin, idx, 0, False, False, NULL) + ErrorLen( ActiveWin, idx, 0, False )- 1;
							for( i= 1; i<= SplineN; i++ ){
								spline_xtab[i]= CurveLen( ActiveWin, idx, i-1, False, False, NULL) + ErrorLen( ActiveWin, idx, i-1, False );
							}
							SplineCopyData(dup_handling);
						}
						xmsg= "curve length[index] + error length[index]";
						break;
					default:
						if( SplineIndepCol< 0 || SplineIndepCol>= AllSets[idx].ncols ){
							fprintf( StdErr, " (Error: indep_col_nr %d not in <0, Set[%d]->ncols=%d>!) ",
								SplineIndepCol, idx, AllSets[idx].ncols
							);
							fflush( StdErr );
							ascanf_emsg= " (indep_col_nr argument 2 out of range) ";
							return(0);
						}
						else{
							spline_xtab[0]= SplineSet->columns[SplineIndepCol][0]- 1;
							for( i= 1; i<= SplineN; i++ ){
								spline_xtab[i]= SplineSet->columns[SplineIndepCol][i-1];
							}
							SplineCopyData(dup_handling);
							xmsg= "specified datacolumn";
							if( !(clabel= Find_LabelsListLabel( SplineSet->ColumnLabels, SplineIndepCol )) ){
								clabel= Find_LabelsListLabel( ActiveWin->ColumnLabels, SplineIndepCol );
							}
						}
						break;
				}
				if( SplineIntv && abs(SplineIntv)!= 1 ){
				  int intv= abs(SplineIntv), j;
					N= 1;
					if( SplineIntv< 0 && (SplineSet->numPoints % intv)>= (intv/2)-1 ){
						intv= (int) (0.5 + (double)SplineSet->numPoints / (SplineSet->numPoints/((double)intv)+1.0));
						if( pragma_unlikely(ascanf_verbose) && intv!= abs(SplineIntv) ){
							fprintf( StdErr, " (using interval %d instead of |%d|) ", intv, SplineIntv );
						}
					}
					for( j= 0, i= 1; i<= SplineN; i++, j++ ){
						if( (j % intv == 0) || i== SplineN ){
							spline_xtab[N]= spline_xtab[i];
							for( col= 0; col< SplineSet->ncols; col++ ){
								if( SPCOL_OK(col) ){
									spline_col[col].orgvals[N]= spline_col[col].orgvals[i];
								}
							}
/* 								fprintf( StdErr, "%d,%d %s,%s\n", N,i,	*/
/* 									d2str( spline_xtab[N], 0,0),	*/
/* 									SPCOL_OK(SplineSet->ycol)? d2str(spline_col[SplineSet->ycol].orgvals[N],0,0) : "?"	*/
/* 								);	*/
							N+= 1;
						}
					}
					SplineN= N-1;
					errno= 0;
					if( !((spline_col= (SplineColumns*) alloc_SplineColumns( N)) &&
							(spline_xtab= (double*) realloc( spline_xtab, N* sizeof(double) ))
						)
					){
						xfree( spline_xtab );
						xfree_SplineColumns( spline_col );
						fprintf( StdErr, "SplineInit: error resizing working memory (%s)\n", serror() );
						if( ascanf_window ){
							xtb_error_box( ascanf_window, "SplineInit: error resizing memory", "Failure" );
						}
						*result= 0;
						return(1);
					}
				}
				  /* Calculate spline coefficients. For the 2nd derivative in the first and last point
				   \ (the latter at SplineN, since Numerical Recipes functions want the first element at
				   \ index 1), we pass the 1st derivative at the 2nd and one-but-last point.
				   */
				for( col= 0; col< spline_colSize; col++ ){
					if( SPCOL_OK(col) ){
						if( PWLInt< 0 ){
							smsg= "piecew. linear ";
							pwlint_coeffs( spline_xtab, spline_col[col].orgvals, SplineN,
								  /* 20050511: if doing a pw-linear "splining", store the coefficients in spline_coeffs!! */
								spline_col[col].spline_coefs
							);
						}
						else{
						  int first=1, last= SplineN, tried= 0;
							while( ( isNaN(spline_col[col].orgvals[first]) || isNaN(spline_col[col].orgvals[first+1])
								|| isNaN(spline_xtab[first]) || isNaN(spline_xtab[first+1]) )
								&& first< SplineN
							){
								first+= 1;
								tried+= 1;
							}
							while( ( isNaN(spline_col[col].orgvals[last]) || isNaN(spline_col[col].orgvals[last-1])
								|| isNaN(spline_xtab[last]) || isNaN(spline_xtab[last-1]) )
								&& last> first+1
							){
								last-= 1;
								tried+= 1;
							}
							if( tried && pragma_unlikely(ascanf_verbose) ){
								fprintf( StdErr, " (col #%d using non-NaN entries %d..%d (%d checks)) ",
									col, first-1, last-1, tried
								);
							}
							smsg= "cubic ";
							spline( spline_xtab, &(spline_col[col].orgvals[first-1]), last-first+1,
								(spline_col[col].orgvals[first+1]-spline_col[col].orgvals[first]) /
									(spline_xtab[first+1]-spline_xtab[first]),
								(spline_col[col].orgvals[last]-spline_col[col].orgvals[last-1])/
									(spline_xtab[last]-spline_xtab[last-1]),
								&(spline_col[col].spline_coefs[first-1])
							);
							for( tried= 1; tried< first; tried++ ){
								set_NaN( spline_col[col].spline_coefs[tried] );
							}
							for( tried= last+1; tried<= SplineN; tried++ ){
								set_NaN( spline_col[col].spline_coefs[tried] );
							}
						}
						if( PWLInt> 0 ){
							smsg= "piecew. linear & cubic ";
							pwlint_coeffs( spline_xtab, spline_col[col].orgvals, SplineN,
								spline_col[col].pwlint_coefs
							);
						}
					}
				}
				*result= 1;
				SS_Reset_(SSIndep);
				for( i= 1; i<= SplineN; i++ ){
					SS_Add_Data_( SSIndep, 1, spline_xtab[i], 1.0);
				}
				if( pragma_unlikely(ascanf_verbose) && SPCOL_OK(SplineSet->xcol) && SPCOL_OK(SplineSet->ycol) && SPCOL_OK(SplineSet->ecol) ){
					SS_Reset_(SSx);
					SS_Reset_(SSy);
					SS_Reset_(SSe);
					for( i= 1; i<= SplineN; i++ ){
						SS_Add_Data_( SSx, 1, spline_col[SplineSet->xcol].spline_coefs[i], 1.0);
						SS_Add_Data_( SSy, 1, spline_col[SplineSet->ycol].spline_coefs[i], 1.0);
						SS_Add_Data_( SSe, 1, spline_col[SplineSet->ecol].spline_coefs[i], 1.0);
					}
/* 					fprintf( StdErr, "SplineInit coeffs: x=\"%s\"\n\ty=\"%s\"\n\te=\"%s\"\n",	*/
/* 						SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSx ),	*/
/* 						SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSy ),	*/
/* 						SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSe )	*/
/* 					);	*/
				}
				if( pragma_unlikely(ascanf_verbose) || s1 ){
				  int len;
				  FILE *fp= (ascanf_verbose)? StdErr : NullDevice;
					len= fprintf( fp,
						" (initialised %sspline coefficients for set %d, independent var \"%s\" (%d:%s) [%d:%s,%s], interval=%d ",
						smsg, idx, xmsg, SplineIndepCol, (clabel)? clabel : "<unlabelled>", SplineN,
/* 						ad2str( spline_xtab[1], d3str_format, NULL), ad2str( spline_xtab[SplineN], d3str_format, NULL)	*/
						ad2str( SSIndep.min, d3str_format, NULL), ad2str( SSIndep.max, d3str_format, NULL),
						(SplineIntv==0)? 1 : abs(SplineIntv)
					);
					if( ascanf_verbose> 1 ){
						len+= fprintf( fp, "coeffs: x=\"%s\"\n\ty=\"%s\"\n\te=\"%s\"",
							SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSx ),
							SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSy ),
							SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSe )
						);
					}
					len+= fprintf( fp, ")== " );
					if( s1 ){
					  _ALLOCA( msg, char, 2* len, msg_len );
						xfree( s1->usage );
						if( msg ){
							sprintf( msg,
								" initialised %d %sspline coefficients for set %d%s, independent var \"%s\" (%d:%s) [%d:%s,%s, interval=%d] ",
								SplineN, smsg, idx, (SplineAllCols)? " (all columns)" : " (only x,y,e)",
								xmsg, SplineIndepCol, (clabel)? clabel : "<unlabelled>", SplineN,
								ad2str( SSIndep.min, d3str_format, NULL), ad2str( SSIndep.max, d3str_format, NULL),
								(SplineIntv==0)? 1 : abs(SplineIntv)
							);
							if( ascanf_verbose> 1 ){
								sprintf( msg, "%scoeffs: x=\"%s\"\n\ty=\"%s\"\n\te=\"%s\"", msg,
									SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSx ),
									SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSy ),
									SS_sprint_full( NULL, d3str_format, " #xb1 ", 0, &SSe )
								);
							}
							strcat( msg, "\n" );
							StringCheck( msg, msg_len, __FILE__, __LINE__ );
							s1->usage= strdup(msg);
						}
						else{
							fprintf( StdErr, " (can't allocate %d bytes to store message string: %s)== ", 2*len, serror() );
						}
						GCA();
					}
				}
			}
		}
		else{
			*result= 0;
		}
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SplineX ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->xcol) ){
		spx= args[0];
		splint( spline_xtab, spline_col[SplineSet->xcol].orgvals, spline_col[SplineSet->xcol].spline_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SplineY ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->ycol) ){
		spx= args[0];
		splint( spline_xtab, spline_col[SplineSet->ycol].orgvals, spline_col[SplineSet->ycol].spline_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SplineE ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->ecol) ){
		spx= args[0];
		splint( spline_xtab, spline_col[SplineSet->ecol].orgvals, spline_col[SplineSet->ecol].spline_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_SplineL ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->lcol) ){
		spx= args[0];
		splint( spline_xtab, spline_col[SplineSet->lcol].orgvals, spline_col[SplineSet->lcol].spline_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

/* A test command for splines fitted to curve_len:
   verbose[pnr[curve_len[0,DCL[N,4]]],SplineX[pnr],SplineY[pnr],SplineE[pnr],Spline[1,pnr],Spline[2,pnr],Spline[3,pnr],DataVal[0,3,N],Spline[0,pnr],DataVal[0,0,N]]
 */

double SplineColNr( int col, double x )
{ double result;
	if( spline_xtab ){
		if( col< 0 || col>= spline_colSize ){
			ascanf_emsg= "(column out of range)";
			ascanf_arg_error= 1;
			result= 0;
		}
		if( !SPCOL_OK(col) ){
			ascanf_emsg= "(unsplined column)";
			ascanf_arg_error= 1;
			result= 0;
		}
		else{
			splint( spline_xtab, spline_col[col].orgvals, spline_col[col].spline_coefs, SplineN, x, &result );
		}
	}
	else{
		ascanf_emsg= "(spline not initialised)";
		ascanf_arg_error= True;
		result= 0;
	}
	return( result );
}

int ascanf_SplineColNr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
  int col;
	if( args && ascanf_arguments> 1 && spline_xtab && !ascanf_SyntaxCheck ){
		if( args[0]< 0 || args[0]>= spline_colSize ){
			ascanf_emsg= "(column out of range)";
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
		col= (int) args[0];
		if( !SPCOL_OK(col) ){
			ascanf_emsg= "(unsplined column)";
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
		else{
			spx= args[1];
			splint( spline_xtab, spline_col[col].orgvals, spline_col[col].spline_coefs, SplineN, spx, result );
		}
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_GetSplineData ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *IndepVals= NULL, *DepVals= NULL, *Coeffs= NULL;
  int col= 0;
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( !SplineSet || !spline_xtab ){
		ascanf_emsg= " (No splined set available) ";
		ascanf_arg_error= 1;
	}
	if( args[0]>= 0 && args[0]< SplineSet->ncols ){
		col= (int) args[0];
	}
	else{
		ascanf_emsg= " (col_nr argument 1 out of range) ";
		ascanf_arg_error= 1;
	}
	if( !(IndepVals= parse_ascanf_address(args[1], _ascanf_array, "ascanf_GetSplineData", (int) ascanf_verbose, NULL )) || !IndepVals->array ){
		ascanf_emsg= " (invalid IndepVals argument (1)) ";
		ascanf_arg_error= 1;
	}
	if( !(DepVals= parse_ascanf_address(args[2], _ascanf_array, "ascanf_GetSplineData", (int) ascanf_verbose, NULL )) || !DepVals->array ){
		ascanf_emsg= " (invalid DepVals argument (2)) ";
		ascanf_arg_error= 1;
	}
	if( !(Coeffs= parse_ascanf_address(args[3], _ascanf_array, "ascanf_GetSplineData", (int) ascanf_verbose, NULL )) || !Coeffs->array ){
		ascanf_emsg= " (invalid Coeffs argument (3)) ";
		ascanf_arg_error= 1;
	}
	if( ascanf_arg_error ){
		return(0);
	}
	else if( ascanf_SyntaxCheck ){
		return(1);
	}
	if( IndepVals->N!= SplineN+2 ){
		Resize_ascanf_Array( IndepVals, SplineN+2, result );
	}
	if( IndepVals->N!= SplineN+2 ){
		fprintf( StdErr, " (failure expanding IndepVals array (%s)) ",
			serror()
		);
		fflush( StdErr );
		return(0);
	}

	IndepVals->array[0]= SSIndep.min;
	IndepVals->array[1]= SSIndep.max;
	memcpy( &(IndepVals->array)[2], &spline_xtab[1], SplineN* sizeof(double) );
	IndepVals->value= IndepVals->array[(IndepVals->last_index= 0)];
	if( IndepVals->accessHandler ){
		AccessHandler( IndepVals, "GetSplineData", level, ASCB_COMPILED, AH_EXPR, NULL);
	}

	if( SPCOL_OK(col) ){
		if( DepVals->N!= SplineN ){
			Resize_ascanf_Array( DepVals, SplineN, result );
		}
		if( Coeffs->N!= SplineN ){
			Resize_ascanf_Array( Coeffs, SplineN, result );
		}
		if( DepVals->N!= SplineN || Coeffs->N!= SplineN ){
			fprintf( StdErr, " (failure expanding DepVals and/or Coeffs array (%s)) ",
				serror()
			);
			fflush( StdErr );
			return(0);
		}
		memcpy( DepVals->array, &spline_col[col].orgvals[1], SplineN* sizeof(double) );
		DepVals->value= DepVals->array[(DepVals->last_index= 0)];
		if( DepVals->accessHandler ){
			AccessHandler( DepVals, "GetSplineData", level, ASCB_COMPILED, AH_EXPR, NULL);
		}
		memcpy( Coeffs->array, &spline_col[col].spline_coefs[1], SplineN* sizeof(double) );
		Coeffs->value= Coeffs->array[(Coeffs->last_index= 0)];
		if( Coeffs->accessHandler ){
			AccessHandler( Coeffs, "GetSplineData", level, ASCB_COMPILED, AH_EXPR, NULL );
		}
		*result= SplineN;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr,
				" (GetSplineData: stored splined Set #%d's column %d: 2-range & %d indep_vals in %s, org_vals in %s and coeffs in %s)== ",
				SplineSet->set_nr, col, SplineN, IndepVals->name, DepVals->name, Coeffs->name
			);
			fflush( StdErr );
		}
	}
	else{
		fprintf( StdErr, " (non-splined column %d requested: only IndepVals array updated!) ",
			col
		);
		fflush( StdErr );
	}
	return(1);
}

int ascanf_PWLIntX ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->xcol) ){
		spx= args[0];
		pwlint( spline_xtab, spline_col[SplineSet->xcol].orgvals, spline_col[SplineSet->xcol].pwlint_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_PWLIntY ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->ycol) ){
		spx= args[0];
		pwlint( spline_xtab, spline_col[SplineSet->ycol].orgvals, spline_col[SplineSet->ycol].pwlint_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_PWLIntE ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->ecol) ){
		spx= args[0];
		pwlint( spline_xtab, spline_col[SplineSet->ecol].orgvals, spline_col[SplineSet->ecol].pwlint_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_PWLIntL ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
	if( args && ascanf_arguments> 0 && spline_xtab && !ascanf_SyntaxCheck && SPCOL_OK(SplineSet->lcol) ){
		spx= args[0];
		pwlint( spline_xtab, spline_col[SplineSet->lcol].orgvals, spline_col[SplineSet->lcol].pwlint_coefs, SplineN, spx, result );
	}
	else{
		*result= 0;
	}
	return( 1 );
}

double PWLIntColNr( int col, double x )
{ double result;
	if( spline_xtab ){
		if( col< 0 || col>= spline_colSize ){
			ascanf_emsg= "(column out of range)";
			ascanf_arg_error= 1;
			result= 0;
		}
		if( !SPCOL_OK(col) ){
			ascanf_emsg= "(unsplined column)";
			ascanf_arg_error= 1;
			result= 0;
		}
		else{
			pwlint( spline_xtab, spline_col[col].orgvals, spline_col[col].pwlint_coefs, SplineN, x, &result );
		}
	}
	else{
		ascanf_emsg= "(spline not initialised)";
		ascanf_arg_error= True;
		result= 0;
	}
	return( result );
}

int ascanf_PWLIntColNr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double spx;
  int col;
	if( args && ascanf_arguments> 1 && spline_xtab && !ascanf_SyntaxCheck ){
		if( args[0]< 0 || args[0]>= spline_colSize ){
			ascanf_emsg= "(column out of range)";
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
		col= (int) args[0];
		if( !SPCOL_OK(col) ){
			ascanf_emsg= "(unsplined column)";
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
		else{
			spx= args[1];
			pwlint( spline_xtab, spline_col[col].orgvals, spline_col[col].pwlint_coefs, SplineN, spx, result );
		}
	}
	else{
		*result= 0;
	}
	return( 1 );
}

/* pseudo code snippet to reuse spline/dataset code for splining arbitray arrays:
	sxt= spline_xtab;
	  * make local copy?? *
	spline_xtab= &<independent_data>[-1]
	SP= SplineSet
	SplineSet= lsp;
	memset(SplineSet,0,...)
	SplineN= length[data];
	SplineSet->ncols= 1;
	SplineSet->columns= double *colums[1]; [0]= &data[-1]
	s_c= spline_col; s_cS= spline_colSize;
	spline_colSize= 0;
	spline_col= alloc_SplineColumns(SplineN+1)
	SplineCopyData(dup_handling)
					if( SPCOL_OK(0) ){
						  * retrieve independent and dependent array *
						spline( spline_xtab, spline_col[0].orgvals, SplineN,
							(spline_col[0].orgvals[2]-spline_col[0].orgvals[1])/(spline_xtab[2]-spline_xtab[1]),
							(spline_col[0].orgvals[SplineN]-spline_col[0].orgvals[SplineN-1])/
								(spline_xtab[SplineN]-spline_xtab[SplineN-1]),
							spline_col[0].spline_coefs
						);
					}
					 * put coefficients in receiving array *
					 * evaluale spline at requested position *
 */


int ascanf_SplineFromData ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *IndepVals= NULL, *DepVals= NULL, *Coeffs= NULL;
  double spx;
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return(0);
	}
	spx= args[0];
	if( !(IndepVals= parse_ascanf_address(args[1], _ascanf_array, "ascanf_SplineFromData", (int) ascanf_verbose, NULL )) || !IndepVals->array ){
		ascanf_emsg= " (invalid IndepVals argument (1)) ";
		ascanf_arg_error= 1;
	}
	if( !(DepVals= parse_ascanf_address(args[2], _ascanf_array, "ascanf_SplineFromData", (int) ascanf_verbose, NULL )) || !DepVals->array ){
		ascanf_emsg= " (invalid DepVals argument (2)) ";
		ascanf_arg_error= 1;
	}
	if( !(Coeffs= parse_ascanf_address(args[3], _ascanf_array, "ascanf_SplineFromData", (int) ascanf_verbose, NULL )) || !Coeffs->array ){
		ascanf_emsg= " (invalid Coeffs argument (3)) ";
		ascanf_arg_error= 1;
	}
	if( ascanf_arg_error ){
		return(0);
	}
	else if( ascanf_SyntaxCheck ){
		return(1);
	}
	if( !(IndepVals->N-2== DepVals->N && DepVals->N== Coeffs->N) ){
		ascanf_emsg= " (all arrays must be similar size!) ";
		ascanf_arg_error= 1;
		return(0);
	}

	splint( &(IndepVals->array[1]), &(DepVals->array[-1]), &(Coeffs->array[-1]), Coeffs->N, spx, result );

	return(1);
}

int ascanf_SplineFinished ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
	*result= SplineN;

	xfree( spline_xtab );
	xfree_SplineColumns( spline_col );
	spline_col= NULL;
	SplineN= 0;
	SplineSet= NULL;
	SS_Reset_(SSIndep);
	return( 1 );
}

int __ascanf_SplineResample ( double *args, double *result, int *level, int use_PWLInt, char *callback, char *fname, ASCB_ARGLIST )
{ ascanf_Function *OrgX= NULL, *OrgY= NULL, *ResampledX= NULL, *ResampledY= NULL, *retOX= NULL, *retOY= NULL, *retCoeffs= NULL;
  double *orgX=NULL, *orgY=NULL;
  int i, orgXN=0, orgYN= 0, resampledXN=0;
  int pad= False, free_orgX= False, free_orgY= False;

	if( ascanf_arguments> 5 && !use_PWLInt ){
		pad= ASCANF_TRUE(args[5]);
	}
	if( !(OrgX= parse_ascanf_address(args[0], _ascanf_array, callback, (int) ascanf_verbose, NULL )) ){
		orgX= NULL;
	}
	else{
		if( OrgX->array && !pad ){
			orgX= OrgX->array;
		}
		orgXN= OrgX->N;
	}
	if( !(OrgY= parse_ascanf_address(args[1], _ascanf_array, callback, (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid orgY argument (1)) ";
		ascanf_arg_error= 1;
	}
	else if( !ascanf_SyntaxCheck ){
		if( OrgX && OrgX->N != OrgY->N ){
			ascanf_emsg= " (mismatching orgX and orgY arrays) ";
			ascanf_arg_error= 1;
		}
	}
	orgYN= OrgY->N;
	if( (ResampledX= parse_ascanf_address(args[2], _ascanf_array, callback, (int) ascanf_verbose, NULL )) ){
		resampledXN= ResampledX->N;
	}
	if( !(ResampledY= parse_ascanf_address(args[3], _ascanf_array, callback, (int) ascanf_verbose, NULL ))
		|| !ResampledY->array
	){
		ascanf_emsg= " (invalid resampledY argument (3)) ";
		ascanf_arg_error= 1;
	}
	else if( ResampledY== OrgY ){
		ascanf_emsg= " (the original array and the resampled array have to be distinct!) ";
/* 		ascanf_arg_error= 1;	*/
	}
	if( ascanf_arguments> 6 && !use_PWLInt && ASCANF_TRUE(args[6]) ){
		if( !(retOX= parse_ascanf_address(args[6], _ascanf_array, callback, (int) ascanf_verbose, NULL ))
		    || !retOX->array
		){
			ascanf_emsg= " (invalid retOX argument (6)) ";
			retOX= NULL;
		}
		if( ascanf_arguments> 7 && ASCANF_TRUE(args[7]) ){
			if( !(retOY= parse_ascanf_address(args[7], _ascanf_array, callback, (int) ascanf_verbose, NULL ))
			    || !retOY->array
			    ){
				ascanf_emsg= " (invalid retOY argument (7)) ";
				retOY= NULL;
			}
		}
		if( ascanf_arguments> 8 && ASCANF_TRUE(args[8]) ){
			if( !(retCoeffs= parse_ascanf_address(args[8], _ascanf_array, callback, (int) ascanf_verbose, NULL ))
			    || !retCoeffs->array
		    ){
				ascanf_emsg= " (invalid retCoeffs argument (8)) ";
				retCoeffs= NULL;
			}
		}
	}
	if( !retOX || !retOY || !retCoeffs ){
		retOX = retOY = retCoeffs = NULL;
	}
	if( ascanf_arg_error ){
		return(0);
	}
	else if( ascanf_SyntaxCheck ){
		return(1);
	}
	if( !orgX ){
		if( OrgX ){
			if( !(orgX= (double*) calloc( ((pad)? orgXN+2 : orgXN), sizeof(double) )) ){
				ascanf_emsg= " (error allocating internal orgX array) ";
				ascanf_arg_error= 1;
			}
			else{
			  int j;
				if( pad ){
					orgX[0]= 2* ASCANF_ARRAY_ELEM(OrgX,0) - ASCANF_ARRAY_ELEM(OrgX,1);
					j= 1;
				}
				else{
					j= 0;
				}
				for( i= 0; i< orgXN; i++, j++ ){
					orgX[j]= ASCANF_ARRAY_ELEM(OrgX,i);
				}
				if( pad ){
					orgX[j]= 2* ASCANF_ARRAY_ELEM(OrgX,i-1) - ASCANF_ARRAY_ELEM(OrgX,i-2);
					if( pad && ascanf_verbose> 1 ){
						fprintf( StdErr, " OrgX[%d+2]=(%g", orgXN, orgX[0] );
						for( i= 1; i< orgXN+2; i++ ){
							fprintf( StdErr, ",%g", orgX[i] );
						}
						fprintf( StdErr, ") " );
						fflush( StdErr );
					}
				}
				free_orgX= True;
			}
		}
		else{
			  /* allocate as many independent values as there are dependent values: */
			if( !(orgX= (double*) calloc( ((pad)? orgYN+2 : orgYN), sizeof(double) )) ){
				ascanf_emsg= " (error allocating internal orgX array) ";
				ascanf_arg_error= 1;
			}
			else{
			  double offset= (args[0]< 0)? 1 : 0, val=fabs(args[0]);
				orgXN= (pad)? orgYN+2 : orgYN;
				for( i= 0; i< orgXN; i++ ){
					orgX[i]= offset + i * val;
				}
				free_orgX= True;
				if( pad ){
					orgXN-= 2;
				}
			}
		}
	}
	if( OrgY->array && !pad ){
		orgY= OrgY->array;
	}
	else if( !ascanf_arg_error ){
		if( !(orgY= (double*) calloc( ((pad)? orgYN+2 : orgYN), sizeof(double) )) ){
			ascanf_emsg= " (error allocating internal orgY array) ";
			ascanf_arg_error= 1;
		}
		else{
		  int j;
			if( pad ){
				orgY[0]= 2* ASCANF_ARRAY_ELEM(OrgY,0) - ASCANF_ARRAY_ELEM(OrgY,1);
				j= 1;
			}
			else{
				j= 0;
			}
			for( i= 0; i< orgYN; i++, j++ ){
				orgY[j]= ASCANF_ARRAY_ELEM(OrgY,i);
			}
			if( pad ){
				orgY[j]= 2* ASCANF_ARRAY_ELEM(OrgY,i-1) - ASCANF_ARRAY_ELEM(OrgY,i-2);
			}
			if( pad && ascanf_verbose> 1 ){
				fprintf( StdErr, " OrgY[%d+2]=(%g", orgYN, orgY[0] );
				for( i= 1; i< orgYN+2; i++ ){
					fprintf( StdErr, ",%g", orgY[i] );
				}
				fprintf( StdErr, ") " );
				fflush( StdErr );
			}
			free_orgY= True;
		}
	}
	if( !ResampledX ){
		resampledXN= fabs(args[2]);
	}
	if( ResampledY->N!= resampledXN && !ascanf_arg_error ){
		Resize_ascanf_Array( ResampledY, resampledXN, result );
		if( ResampledY->N != resampledXN ){
			ascanf_emsg= " (mismatching resampledX and resampledY arrays (allocation error)) ";
			ascanf_arg_error= 1;
		}
	}
	set_NaN(*result);
	if( !ascanf_arg_error ){
	  double *coeffs;
	  	if( (coeffs = calloc( ((pad)? orgXN+2 : orgXN), sizeof(double) )) ){
			if( use_PWLInt ){
				pwlint_coeffs( &orgX[-1], &orgY[-1], (pad)? orgYN+2 : orgYN, &coeffs[-1] );
			}
			else{
			  int first= 0, Last= (pad)? orgYN+1 : orgYN-1, tried= 0, last= Last;
			  double xp;
				while( ( isNaN(orgY[first]) || isNaN(orgY[first+1])
					|| isNaN(orgX[first]) || isNaN(orgX[first+1]) )
					&& first< last
				){
					first+= 1;
					tried+= 1;
				}
				while( ( isNaN(orgY[last]) || isNaN(orgY[last-1])
					|| isNaN(orgX[last]) || isNaN(orgX[last-1]) )
					&& last> first+2
				){
					last-= 1;
					tried+= 1;
				}
				  // 20100530:
				if( first+1 != last ){
					xp = (orgY[first+1]-orgY[last]) / (orgX[first+1]-orgX[last]);
				}
				else{
					xp = (orgY[first]-orgY[last]) / (orgX[first]-orgX[last]);
				}
#if 0
				fprintf( StdErr, "spline(&%p[%d][-1], &%p[%d][%d], %d-%d+1=%d, %g, %g, &%p[%d][%d])\n",
					orgX, orgXN, orgY, orgYN, first-1, last, first, last-first+1,
						xp,
						(orgY[last]-orgY[last-1]) / (orgX[last]-orgX[last-1]),
						coeffs, orgXN, first-1
				);
#endif
				if( first >= Last || first == last ){
				  /* 20090922: sometimes, one just doesn't find a usable range of usable values... */
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (unsupported non-NaN element range %d-%d of %d) ", first, last, Last );
					}
				}
				else{
					if( tried ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (using non-NaN entries %d..%d) ", first, last );
						}
	/* 					memset( coeffs, 0, orgXN*sizeof(double) );	*/
					}
					spline( &orgX[-1], &(orgY[first-1]), last-first+1,
						xp,
						(orgY[last]-orgY[last-1]) / (orgX[last]-orgX[last-1]),
						&coeffs[first-1]
					);
				}
				for( tried= 0; tried< first; tried++ ){
					set_NaN( coeffs[tried] );
				}
				for( tried= last+1; tried< orgYN; tried++ ){
					set_NaN( coeffs[tried] );
				}
				if( retOX ){
				  unsigned int retN = last - first + 1;
					Resize_ascanf_Array( retOX, retN, result );
					Resize_ascanf_Array( retOY, retN, result );
					Resize_ascanf_Array( retCoeffs, retN, result );
					if( retOX->N != retN || retOY->N != retN || retCoeffs->N != retN ){
						ascanf_emsg= " (retOX/retOY/retCoeffs allocation error) ";
						ascanf_arg_error= 1;
						Resize_ascanf_Array( retOX, 1, result );
						Resize_ascanf_Array( retOY, 1, result );
						Resize_ascanf_Array( retCoeffs, 1, result );
					}
					else{
						for( i= 0, tried= first; tried <= last; tried++, i++ ){
							ASCANF_ARRAY_ELEM_SET( retOX, i, orgX[tried] );
							ASCANF_ARRAY_ELEM_SET( retOY, i, orgY[tried] );
							ASCANF_ARRAY_ELEM_SET( retCoeffs, i, coeffs[tried] );
						}
					}
				}
			}
			if( ResampledX ){
			  int porgYN= (pad)? orgYN+2 : orgYN;
				if( use_PWLInt ){
					for( i= 0; i< resampledXN; i++ ){
						pwlint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN,
							(double) ASCANF_ARRAY_ELEM(ResampledX,i), &(ResampledY->array[i]) );
#if DEBUG == 2
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "spline(%d==%g) = %g\n", i, ASCANF_ARRAY_ELEM(ResampledX,i), ResampledY->array[i] );
						}
#endif
					}
				}
				else{
					for( i= 0; i< resampledXN; i++ ){
						splint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN,
							(double) ASCANF_ARRAY_ELEM(ResampledX,i), &(ResampledY->array[i]) );
#if DEBUG == 2
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "spline(%d==%g) = %g\n", i, ASCANF_ARRAY_ELEM(ResampledX,i), ResampledY->array[i] );
						}
#endif
					}
				}
			}
			else{
			  int porgYN= (pad)? orgYN+2 : orgYN;
			  double x, offset, scale;
				if( pad ){
					offset= orgX[1];
					scale= (orgX[orgXN] - orgX[1]) / (resampledXN-1);
				}
				else{
					offset= orgX[0];
					scale= (orgX[orgXN-1] - orgX[0]) / (resampledXN-1);
				}
				if( use_PWLInt ){
					for( i= 0; i< resampledXN; i++ ){
						x= i* scale + offset;
						pwlint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN, x, &(ResampledY->array[i]) );
#if DEBUG == 2
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "spline(%d=%g) = %g\n", i, x, ResampledY->array[i] );
						}
#endif
					}
				}
				else{
					for( i= 0; i< resampledXN; i++ ){
						x= i* scale + offset;
						splint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN, x, &(ResampledY->array[i]) );
#if DEBUG == 2
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "spline(%d=%g) = %g\n", i, x, ResampledY->array[i] );
						}
#endif
					}
				}
			}
			*result= ResampledY->N;
			xfree(coeffs);
		}
		else{
			ascanf_emsg= " (error allocating internal coeffs array) ";
			ascanf_arg_error= 1;
		}
	}
	ResampledY->last_index= 0;
	if( ResampledY->accessHandler ){
		AccessHandler( ResampledY, fname, level, ASCB_COMPILED, AH_EXPR, NULL);
	}
	if( free_orgX ){
		xfree(orgX);
	}
	if( free_orgY ){
		xfree(orgY);
	}
	return( !ascanf_arg_error );
}

double SplineValue( double x, double *orgX, double *orgY, int porgYN, double *coeffs, int upwlint )
{ double y;
	if( upwlint ){
		pwlint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN, x, &y );
	}
	else{
		splint( &orgX[-1], &orgY[-1], &coeffs[-1], porgYN, x, &y );
	}
	return(y);
}

int ascanf_SplineResample ( ASCB_ARGLIST )
{ ASCB_FRAME
	set_NaN( *result );
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return(0);
	}
	ascanf_arg_error= 0;
	return( __ascanf_SplineResample( args, result, level, 0, "ascanf_SplineResample", "Spline-Resample", __ascb_frame ) );
}

int ascanf_PWLIntResample ( ASCB_ARGLIST )
{ ASCB_FRAME
	set_NaN( *result );
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return(0);
	}
	ascanf_arg_error= 0;
	return( __ascanf_SplineResample( args, result, level, 1, "ascanf_PWLIntResample", "PWLInt-Resample", __ascb_frame ) );
}

static ascanf_Function splines_Function[] = {
	{ "$SplineInit-does-PWLInt-too", NULL, 2, _ascanf_variable,
		"$SplineInit-does-PWLInt-too: whether the cubic spline SplineInit[] routine also calculates the coefficients\n"
		" for piecewise linear interpolation, such that the PWLInt routines can be used.\n"
		" When -1, the Spline routines return the PWLInt values instead.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "SplineInit", ascanf_SplineInit, 6, NOT_EOF_OR_RETURN, "SplineInit[<set_nr>,<indep_col_nr>[,<dups>=0[,<all>=1[,[interval,]`return_msg]]]]: initialise spline coefficients "
		"for all columns in set <set_nr> (<all=1>, default), or for just X, Y and error/orientation (<all=0>)\n"
		" Use indep_col_nr==-1 for point index, -2 for curve_len, -3 for error_len\n"
		" Handling of duplicate indep. vars: (default) dups==0 for averaging, 1 for using the 1st, 2 for using the last\n"
		" The <interval> argument specifies at what interval the original data is to be sampled;\n"
		"     0 or 1 means that every sample is to be used. The 1st and last point are always included;\n"
		" a negative value means that an attempt will be made to distribute any error in the last segment\n"
		" evenly over all segments.\n"
		" The optional `return_msg contains the message(s) generated\n"
	},
	{ "SplineX", ascanf_SplineX, 1, NOT_EOF_OR_RETURN, "SplineX[<iVal>]: return the \"splined\" X value at independent value <iVal>"},
	{ "SplineY", ascanf_SplineY, 1, NOT_EOF_OR_RETURN, "SplineY[<iVal>]: return the \"splined\" Y value at independent value <iVal>"},
	{ "SplineE", ascanf_SplineE, 1, NOT_EOF_OR_RETURN, "SplineE[<iVal>]: return the \"splined\" error/orientation value at independent value <iVal>"},
	{ "SplineL", ascanf_SplineL, 1, NOT_EOF_OR_RETURN, "SplineL[<iVal>]: return the \"splined\" vectorLength value at independent value <iVal>"},
	{ "Spline", ascanf_SplineColNr, 2, NOT_EOF_OR_RETURN, "Spline[<col>,<iVal>]: return the \"splined\" value at independent value <iVal> for column <col>"},
	{ "GetSplineData", ascanf_GetSplineData, 4, NOT_EOF_OR_RETURN,
		"GetSplineData[<col>,&IndepVals,&DepVals,&Coeffs]: return the spline-parameters for column <col>\n"
		" IndepVals, DepVals and Coeffs must be double arrays that will be resized as necessary\n"
		" IndepVals[0] and [1] will contain the range of independent values (FYI); the rest the independent values\n"
	},
	{ "SplineFromData", ascanf_SplineFromData, 4, NOT_EOF_OR_RETURN,
		"SplineFromData[<val>,&IndepVals,&DepVals,&Coeffs]: evaluate the spline given by {IndepVals,DepVals,Coeffs} at <val>\n"
		" DepVals and Coeffs must be double arrays of identical size;\n"
		" IndepVals must be 2 bigger, with the 1st 2 elements the range (ignored)\n"
	},
	{ "PWLIntX", ascanf_PWLIntX, 1, NOT_EOF_OR_RETURN,
		"PWLIntX[<pnt_nr>]: return the \"piecewise linearly interpolated\" X value number <pnt_nr>"},
	{ "PWLIntY", ascanf_PWLIntY, 1, NOT_EOF_OR_RETURN,
		"PWLIntY[<pnt_nr>]: return the \"piecewise linearly interpolated\" Y value number <pnt_nr>"},
	{ "PWLIntE", ascanf_PWLIntE, 1, NOT_EOF_OR_RETURN,
		"PWLIntE[<pnt_nr>]: return the \"piecewise linearly interpolated\" error/orientation value number <pnt_nr>"},
	{ "PWLIntL", ascanf_PWLIntL, 1, NOT_EOF_OR_RETURN,
		"PWLIntL[<pnt_nr>]: return the \"piecewise linearly interpolated\" vectorLength value number <pnt_nr>"},
	{ "PWLInt", ascanf_PWLIntColNr, 2, NOT_EOF_OR_RETURN,
		"PWLInt[<col>,<pnt_nr>]: return the \"piecewise linearly interpolated\" value number <pnt_nr> for column <col>"},
	{ "SplineFinished", ascanf_SplineFinished, 1, NOT_EOF_OR_RETURN, "SplineFinished: de-allocate spline resources"},
	{ "Spline-Resample", ascanf_SplineResample, 9, NOT_EOF_OR_RETURN,
		"Spline-Resample[orgX, &orgY, resampledX, &resampledY[, dealloc[,pad[,&retOX,&retOY,&coeffs]]]]: resample the data in &orgY using a cubic spline.\n"
		" orgY and resampledY (which will hold the resampled values) must point to double arrays.\n"
		" orgX describes the independent (X) values to be used in the calculation of the spline coefficients, and can be:\n"
		"      a pointer to an array of X values, supposed to be monotonically in/decreasing\n"
		"      a scalar value >0 to use a series of 0,orgX,2*orgX,...\n"
		"      a scalar value <0 to use a series of 1,1+orgX,1+2*orgX,...\n"
		" resampledX specifies the values at which the spline must be evaluated (i.e. at which to resample), and can be:\n"
		"      a pointer to an array of new X values\n"
		"      a scalar value >0 to indicate the desired number of equally spaced values (pace 1) between orgX[0] and orgX[N-1]\n"
#if 1
		" The dealloc flag is not currently implemented\n"
#else
		" The optional dealloc flag indicates whether (or not, the default) the internal state must be preserved until a\n"
		" next call. This would allow to resample multiple orgY with respect to identical orgX and resampledX. Internal\n"
		" state is of course changed when a different orgX is given (a different scalar, or a different pointer which might\n"
		" refer to identical values!)\n"
#endif
		" The optional <pad> argument activates a simple form of padding, which, given orgX={0,1,3} orgY={0,1,3} adds 2\n"
		" external points such that resampling is done against {-1,0,1,3,5},{-1,0,1,3,5}. This reduces boundary artefacts\n"
		" to some extent.\n"
		" [retOX,retOY,coeffs]: arraypointers that will contain, respectively, the orginal X and Y data (padded, NaNs replaced)\n"
		" and the cubic spline coefficients.\n"
	},
	{ "PWLInt-Resample", ascanf_PWLIntResample, 5, NOT_EOF_OR_RETURN,
		"PWLInt-Resample[orgX, &orgY, resampledX, &resampledY[, dealloc]]: as Spline-Resample, but uses piecewise linear interpolation.\n"
	},
};
static int splines_Functions= sizeof(splines_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= splines_Function;
  static char called= 0;
  int i;
  char buf[64];

	ascanf_PWLInt= &splines_Function[0].value;

	for( i= 0; i< splines_Functions; i++, af++ ){
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
		XGRAPH_FUNCTION(Find_LabelsListLabel_ptr, "Find_LabelsListLabel");
		XGRAPH_FUNCTION(CurveLen_ptr, "CurveLen");
		XGRAPH_FUNCTION(ErrorLen_ptr, "ErrorLen");
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( splines_Function, splines_Functions, "splines::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-splines" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" functions for \"splining\" datasets.\n"
	);
	return( DM_Ascanf );
}

void initsplines()
{
	wrong_dymod_loaded( "initsplines()", "Python", "splines.so" );
}

void R_init_splines()
{
	wrong_dymod_loaded( "R_init_splines()", "R", "splines.so" );
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
	  int r= remove_ascanf_functions( splines_Function, splines_Functions, force );
		if( force || r== splines_Functions ){
			for( i= 0; i< splines_Functions; i++ ){
				splines_Function[i].dymod= NULL;
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
				r, splines_Functions
			);
		}
	}
	return(ret);
}
