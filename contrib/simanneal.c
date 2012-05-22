#include "config.h"
IDENTIFY( "Minimisation by continuous simulated annealing and Simplex downhill methods, ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif

#include <stdio.h>
#include <stdlib.h>

extern FILE *StdErr;

  /* Get the dynamic module definitions:	*/
#include "dymod.h"

/* #include "Macros.h"	*/

/* #include "xgALLOCA.h"	*/

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

#include <errno.h>
#if !defined(linux) && !defined(__MACH__) && !defined(__CYGWIN__)
	extern char *sys_errlist[];
	extern int sys_nerr;
#endif

#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
#include "compiled_ascanf.h"
#include "ascanfc-table.h"

#include "Python/PythonInterface.h"

#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;

#include "simanneal.h"


#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1) /NTAB)
#define RNMX (1.0-DBL_EPSILON)

static long ran1_iy= 0;
static long ran1_iv[NTAB];
static long ran1_seed= -1234;

void sran1( long seed )
{ int j;
  long k;
	ran1_seed= seed;
	if( ran1_seed <=  0 || !ran1_iy ){
		if( -(ran1_seed) < 1 ){
			ran1_seed= 1;
		}
		else{
/* 			ran1_seed= -(ran1_seed) ;	*/
			ran1_seed= ABS(ran1_seed) ;
		}
		for( j= NTAB+7; j>= 0; j-- ){
			k= (ran1_seed) /IQ;
			ran1_seed= IA*(ran1_seed-k*IQ) -IR*k;
			if( ran1_seed < 0 ){
				ran1_seed += IM;
			}
			if( j < NTAB ){
				ran1_iv[j] = ran1_seed;
			}
		}
		ran1_iy= ran1_iv[0];
	}
}

#ifdef __GNUC__
inline
#endif
double ran1()
/* Minimal" random number generator of Park and Miller with Bays-Durham shuffle and added
 \ safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint
 \ values) . Call with seed a negative integer to initialize; thereafter, do not alter seed between
 \ successive deviates in a sequence. RNMX should approximate the largest floating value that is
 \ less than 1.
 */
{ int j;
  long k;
  float temp;

	if( !ran1_iy ){
		sran1( ran1_seed ) ;
	}
	k= (ran1_seed) /IQ;
	ran1_seed= IA*(ran1_seed-k*IQ) -IR*k;
	if( ran1_seed < 0 ){
		ran1_seed += IM;
	}
	j= ran1_iy/NDIV;
	ran1_iy= ran1_iv[j];
	ran1_iv[j] = ran1_seed;
	if( (temp= AM* ran1_iy) > RNMX ){
		return RNMX;
	}
	else{
		return temp;
	}
}

#define GET_PSUM	for( n= 1; n<= ndim; n++ ){ \
	  double sum; \
		for( sum= 0.0, m= 1; m<= mpts; m++){ \
			sum += p[m][n]; \
		} \
		psum[n]= sum; \
	}


  /* Extrapolates by a factor fac through the face of the simplex across from the high point, tries
   \ it, and replaces the high point if the new point is better.
   */
static double csa_sub(double **p, double y[], double psum[], int ndim, double pb[],
	double *yb, double (*funk) (double []) , int ihi, double *yhi, double fac, double ttemp)
{ int j;
  double fac1, fac2, yflu, ytry;
  ALLOCA( ptry, double, ndim+1, ptry_len );

	fac1= (1.0-fac) / ndim;
	fac2= fac1-fac;
	for( j= 1; j<= ndim; j++){
		ptry[j]= psum[j] * fac1 - p[ihi][j] * fac2;
	}
	ytry= (*funk)(ptry);
	if( ytry <= *yb ){
		for( j= 1; j<= ndim; j++){
			pb[j]= ptry[j];
		}
		*yb= ytry;
	}
	yflu= (ttemp)? ytry - ttemp * log( ran1() ) : ytry;
	if( yflu < *yhi ){
		y[ihi]= ytry;
		*yhi= yflu;
		for( j= 1; j<= ndim; j++ ){
			psum[j] += ptry[j]-p[ihi][j];
			p[ihi][j]= ptry[j];
		}
	}
	GCA();
	return( yflu );
}


  /* Multidimensional minimization of the function funk(x) where x[1..ndim] is a vector in
   \ ndim dimensions, by simulated annealing combined with the downhill simplex method of Nelder
   \ and Mead. The input matrix p[1..ndim+1][1..ndim] has ndim+1 rows, each an ndimdimensional
   \ vector which is a vertex of the starting simplex. Also input are the following: the
   \ vector y[1..ndim+1], whose components must be pre-initialized to the values of funk evaluated
   \ at the ndim+1 vertices (rows) of p; ftol, the fractional convergence tolerance to be
   \ achieved in the function value for an early return; iter, and T. The routine makes iter
   \ function evaluations at an annealing temperature T, then returns. You should then de-
   \ crease T according to your annealing schedule, reset iter, and call the routine again
   \ (leaving other arguments unaltered between calls) . If iter is returned with a positive value,
   \ then early convergence and return occurred. If you initialize yb to a very large value on the first
   \ call, then yb and pb[1..ndim] will subsequently return the best function value and point ever
   \ encountered (even if it is no longer a point in the simplex) .
   */
void continuous_simanneal(double **p, double y[], int ndim, double pb[], double *yb, double ftol,
	double (*funk) (double []) , int *iter, double T)
{ int i, ihi, ilo, j, m, n, mpts= ndim+1;
  double rtol, swap, yhi, ylo, ynhi, yt, ytry;
  ALLOCA( psum, double, ndim+1, psum_len);
  double tt;

/* 	sran1(-rand());	*/

	tt = -T;
	GET_PSUM;
	for( ; ; ){
		ilo= 1;
		ihi= 2;
		ynhi= ylo= (tt)? y[1]+tt*log(ran1() ) : y[1];
		yhi= (tt)? y[2]+tt*log(ran1() ) : y[2];
		if( ylo > yhi ){
			ihi= 1;
			ilo= 2;
			ynhi= yhi;
			yhi= ylo;
			ylo= ynhi;
		}
		for( i= 3; i<= mpts; i++ ){
			yt= (tt)? y[i]+tt* log( ran1() ) : y[i];
			if( yt <= ylo ){
				ilo= i;
				ylo= yt;
			}
			if( yt > yhi ){
				ynhi= yhi;
				ihi= i;
				yhi= yt;
			}
			else if( yt > ynhi ){
				ynhi= yt;
			}
		}
		rtol= 2.0 * fabs(yhi-ylo) / ( fabs(yhi) +fabs(ylo) );
		if( rtol < ftol || *iter < 0 ){
			swap= y[1];
			y[1]= y[ilo];
			y[ilo]= swap;
			for( n= 1; n<= ndim; n++ ){
				swap= p[1][n];
				p[1][n]= p[ilo][n];
				p[ilo][n]= swap;
			}
			break;
		}
		*iter -= 2;
		ytry= csa_sub(p, y, psum, ndim, pb, yb, funk, ihi, &yhi, -1.0, tt );
		if( ytry <=  ylo ){
			ytry= csa_sub(p, y, psum, ndim, pb, yb, funk, ihi, &yhi, 2.0, tt) ;
		}
		else if( ytry >= ynhi ){
		  double ysave= yhi;
			ytry= csa_sub(p, y, psum, ndim, pb, yb, funk, ihi, &yhi, 0.5, tt );
			if( ytry >= ysave ){
			for( i= 1; i<= mpts; i++ ){
				if( i != ilo ){
					for( j= 1; j<= ndim; j++ ){
						psum[j]= 0.5*(p[i][j]+p[ilo][j]) ;
						p[i][j]= psum[j];
					}
					y[i]= (*funk) (psum) ;
				}
			}
			*iter -= ndim;
			GET_PSUM;
			}
		}
		else{
			++(*iter);
		}
	}
	GCA();
}

static double prev_ran_PM;
/* routine for the "ran-PM[low,high]" ascanf syntax	*/
int ascanf_ran_PM ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		*result= prev_ran_PM= ran1();
		return( 1 );
	}
	else{
	  double cond;
	  int n= 0;
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments> 2 ){
			if( (cond= ABS(args[2]))< 1 ){
				if( !ascanf_SyntaxCheck ){
					do{
						*result= ran1();
						ascanf_check_event( "ascanf_ran_PM" );
						n+= 1;
					} while( fabs(*result- prev_ran_PM)<= cond && !ascanf_escape );
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (%d discarded polls, diff.=%s) ", n,
							ad2str( *result- prev_ran_PM, d3str_format, NULL )
						);
					}
					prev_ran_PM= *result;
					*result= *result* (args[1]- args[0]) + args[0];
				}
			}
			else{
				if( args[2]< 0 ){
					cond*= ran1();
				}
				if( !ascanf_SyntaxCheck ){
					while( cond> 0 ){
						ran1();
						cond-= 1;
						n+= 1;
					}
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (%d discarded polls) ", n );
					}
				}
				*result= (prev_ran_PM= ran1()) * (args[1]- args[0]) + args[0];
			}
		}
		else{
			*result= (prev_ran_PM= ran1()) * (args[1]- args[0]) + args[0];
		}
		return( 1 );
	}
}

int ascanf_sran_PM ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( args ){
		*result= (long) args[0];
		sran1( (long) args[0] );
	}
	return(1);
}

static ascanf_Function *af_minimise= NULL, *af_min= NULL, *af_max= NULL;
static int *afm_level= 0, afm_nPar= 0;
static Compiled_Form *afm_form= NULL;
static unsigned int af_min_calls= 0;
static double **afm_P= NULL, **afm_P_Full= NULL, *afm_Pval= NULL, *afm_bestP= NULL;
static double bestPval;

#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif

#ifdef __GNUC__
inline
#endif
static double _ascanf_minimise_full( double *args )
{ int n= 1;
  static int L= 0;
  double *lArgList= af_ArgList->array, value;
  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList;
	switch( af_minimise->type ){
		case _ascanf_procedure:
			SET_AF_ARGLIST( args, afm_nPar );
			ascanf_update_ArgList= False;
			set_NaN(value);
			if( !afm_level ){
				afm_level= &L;
			}
			evaluate_procedure( &n, af_minimise, &value, afm_level );
			SET_AF_ARGLIST( lArgList, lArgc );
			break;
		case _ascanf_python_object:
			if( dm_python ){
				if( (*dm_python->ascanf_PythonCall)( af_minimise, afm_nPar, args, &value) ){
					af_minimise->value= value;
					af_minimise->assigns+= 1;
					if( af_minimise->accessHandler ){
						AccessHandler( af_minimise, af_minimise->name, 0, NULL, NULL, &value );
					}
				}
			}
			break;
	}
	ascanf_update_ArgList= auA;
	af_min_calls+= 1;
	if( NaNorInf(value) ){
		af_minimise->value= value= DBL_MAX;
	}
	else{
		af_minimise->value= value;
	}
	if( n== 1 ){
		af_minimise->assigns+= 1;
		if( af_minimise->accessHandler ){
			AccessHandler( af_minimise, af_minimise->name, afm_level, afm_form, "<SimplexAnneal internal>", NULL );
		}
	}
	return( value );
}

#ifdef __GNUC__
inline
#endif
static double ascanf_minimise_full( double *args )
{  /* interface function that can be passed to continuous_simanneal: */
	return( _ascanf_minimise_full( &args[1] ) );
}

  /* interface function that can be passed to continuous_simanneal: */
static double ascanf_minimise( double *args )
{ int i;
	for( i= 0; i< afm_nPar; i++ ){
		CLIP( args[i+1], af_min->array[i], af_max->array[i] );
	}
	return( ascanf_minimise_full(args) );
}

static double sima_initialise_best( int nPar, double *P, double *best, double (*minimise)(double *) )
{ int i;
	for( i= 1; i<= nPar; i++ ){
		P[i]= best[i];
	}
	return( (*minimise)( P ) );
}

typedef struct Simplex{
	int vertex, parameter;
	double value;
} Simplex;

static double *initSimplexType;

static void sima_initial_simplex( int nPar, double **P, double *bestP, double *Pval, double (*minimise)(double *),
	double *initial, double *min, double *max, int verbose
)
{ int i, j, idx= 0;
  ALLOCA( indeks, int, (nPar+2), ilen );
  ALLOCA( simplex, double*, (nPar+2), slen );
  ALLOCA( simbuf, double, (nPar+2)*(nPar+1), sblen );
	  /* Initialise the initial simplex. This is a tricky affaire, since it seems to determine the
	   \ space that continuous_simanneal() will search for a minimum. Thus, we take as the 1st
	   \ and 2nd elements the initial guess and the raw configuration. The remaining elements are
	   \ initialised such that the probably full search space is spanned: that's why we need min and max.
	   \ double P[nPar+1][nPar], double bestP[nPar], Pval[nPar+1], initial[nPar], min[nPar], max[nPar]
	   */
	  /* P1 of the simplex is the initially-guessed shiftrot'ed state: 	*/
	indeks[idx]= 0;
	for( j= 0; j< nPar+2; j++ ){
		simplex[j]= &simbuf[ j*(nPar+1) ];
	}
	for( j= 1, i= 1; i<= nPar; i++ ){
		simplex[j][i]= P[j][i]= initial[i];
		idx+= 1;
	}
	if( nPar> 1 ){
		if( *initSimplexType== 0 ){
		  ALLOCA( cum, double, (nPar+1), clen );
		  ALLOCA( delta, double, (nPar+1), dlen );
			for( i= 1; i<= nPar; i++ ){
				delta[i]= (max[i]-min[i])/ (nPar-1.0);
				cum[i]= min[i];
			}

			for( j= 2; j<= nPar+1; j++ ){
				indeks[j-1]= j-1;
				for( i= 1; i<= nPar; i++ ){
					simplex[j][i]= P[j][i]= cum[i];
					cum[i]+= delta[i];
					idx+= 1;
				}
			}
			GCA();
		}
		else if( *initSimplexType== 2 ){
			for( j= 2; j<= nPar+1; j++ ){
				indeks[j-1]= j-1;
				for( i= 1; i<= nPar; i++ ){
					simplex[j][i]= P[j][i]= (ran1()* 3.5/2.0- 1.75/2.0)* fabs(max[i]-min[i]) + initial[i];
					idx+= 1;
				}
			}
		}
		else{
			for( j= 2; j<= nPar+1; j++ ){
				indeks[j-1]= j-1;
				for( i= 1; i<= nPar; i++ ){
					simplex[j][i]= P[j][i]= ran1()* (max[i]-min[i]) + min[i];
					idx+= 1;
				}
			}
		}
	}
	if( idx> (nPar+1)* (nPar+0) ){
		fprintf( StdErr, " (%s::sima_initial_simplex:%d: memory overflow; crash is to be expected\n",
			__FILE__, __LINE__
		);
		idx= (nPar+1)*(nPar+0);
	}
	if( nPar> 1 && *initSimplexType== 0 ){
		for( i= 1; i<= nPar; i++ ){
			  /* Permute the Simplex by shuffling the indeks array that accesses P[j];
			   \ shuffles P[j][i] with P[J][i].
			   */
			q_Permute( (void*) indeks, nPar+1, sizeof(int), False );
			for( j= 1; j< nPar+2; j++ ){
			  int J= indeks[j-1]+1;
				if( ascanf_verbose> 2 || verbose> 2 ){
					fprintf( StdErr, "%d->%d : P[%d][%d]%g <- [%d][%d]%g\n",
						j, J,
						j, i, simplex[j][i],
						J, i, simplex[J][i]
					);
				}
				P[j][i]= simplex[J][i];
			}
		}
	}
	  /* now calculate the function values at the simplex nodes, and determine the maximum bestPval: */
	if( minimise ){
		for( j= 1; j<= nPar+1; j++ ){
			Pval[j]= (*minimise)( P[j] );
			bestPval= MAX( bestPval, Pval[j] );
		}
	}
	if( ascanf_verbose> 1 || verbose> 1 ){
		fprintf( StdErr, " initial simplex: " );
		for( j= 1; j<= nPar+1; j++ ){
			fprintf( StdErr, "{%g", P[j][1] );
			for( i= 2; i<= nPar; i++ ){
				fprintf( StdErr, ",%g", P[j][i] );
			}
			fputs( "}", StdErr );
		}
		fputs( "\n", StdErr );
		if( minimise ){
			fprintf( StdErr, " initial simplex values: {%g", Pval[1] );
			for( i= 2; i<= nPar+1; i++ ){
				fprintf( StdErr, ",%g", Pval[i] );
			}
			fputs( "}\n", StdErr );
		}
		if( bestP ){
			fprintf( StdErr, " initial best parameters: {%g", bestP[1] );
			for( i= 2; i<= nPar; i++ ){
				fprintf( StdErr, ",%g", bestP[i] );
			}
			fprintf( StdErr, "}; max=%g\n", bestPval );
		}
	}
	GCA();
}

static int Busy= 0;

/* Minimise-SimplexAnneal[&proc, nPar, &iter, ftol, anneal, &initial-P[nPar], &min-P[nPar], &max-P[nPar][, &best-P]] */
int ascanf_SimplexAnneal( ASCB_ARGLIST )
{ ASCB_FRAME
  double ftol, anneal;
  ascanf_Function *af_iter, *af_initial, *af_best= NULL;
  int nPar, iter= 0, reinit= False, verbose= 0;
	ascanf_arg_error= 0;
	if( Busy ){
		ascanf_arg_error= True;
		ascanf_emsg= " (called recursively!) ";
	}
	if( ascanf_arguments>= 8 && !ascanf_arg_error ){
		if( !(af_minimise=
			parse_ascanf_address( args[0], 0, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !(af_minimise->type== _ascanf_procedure || af_minimise->type== _ascanf_python_object)
		){
			ascanf_emsg= " (1st argument must be a pointer to a procedure or Python object) ";
			ascanf_arg_error= True;
			af_minimise= NULL;
		}
		if( args[1]<= 0 || args[1]>= MAXINT ){
			ascanf_emsg= " (2nd argument (nPar) must be a positive integer) ";
			ascanf_arg_error= !ascanf_SyntaxCheck;
		}
		else{
			nPar= (int) args[1];
		}
		if( !(af_iter=
			parse_ascanf_address( args[2], _ascanf_variable, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| af_iter->value<= 0 || af_iter->value>= MAXINT
		){
			ascanf_emsg= " (3rd argument must be a pointer to a positive integer) ";
			ascanf_arg_error= !ascanf_SyntaxCheck || !af_iter;
		}
		else{
			iter= (int) af_iter->value;
		}
		if( (ftol= args[3])<= 0 ){
			ascanf_emsg= " (4th argument must be positive and probably <1) ";
			ascanf_arg_error= True;
		}
		if( ASCANF_TRUE(args[4]) ){
			anneal= args[4];
		}
		else{
			anneal= 0;
		}
		if( !(af_initial=
			parse_ascanf_address( args[5], _ascanf_array, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !af_initial->array || af_initial->N< nPar
		){
			ascanf_emsg= " (6th argument must be a pointer to a double array[nPar]) ";
			ascanf_arg_error= True;
		}
		if( !(af_min=
			parse_ascanf_address( args[6], _ascanf_array, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !af_min->array || af_min->N< nPar
		){
			ascanf_emsg= " (7th argument must be a pointer to a double array[nPar]) ";
			ascanf_arg_error= True;
		}
		if( !(af_max=
			parse_ascanf_address( args[7], _ascanf_array, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !af_max->array || af_max->N< nPar
		){
			ascanf_emsg= " (8th argument must be a pointer to a double array[nPar]) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 8 && ASCANF_TRUE(args[8]) && (!(af_best=
			parse_ascanf_address( args[8], _ascanf_array, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !af_best->array)
		){
			ascanf_emsg= " (9th argument must be a pointer to a double array) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 9 && ASCANF_TRUE(args[9]) ){
			verbose= (int) args[9];
		}
		if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
		  int i, r;

			Busy= True;

			if( !Resize_ascanf_Array( af_initial, nPar, NULL) ||
				!Resize_ascanf_Array( af_min, nPar, NULL) ||
				!Resize_ascanf_Array( af_max, nPar, NULL) ||
				(af_best && !Resize_ascanf_Array( af_best, nPar, NULL))
			){
				ascanf_emsg= " (memory allocation problem) ";
				ascanf_arg_error= True;
				goto bail_out;
			}

			  /* simple check whether we got the memory. On a number of systems, ALLOCA (through alloca) won't
			   \ fail (= we'll die at once when it does). On others, we'll just suppose that P will be NULL too
			   \ when one of the preceding allocations failed...
			   */
			if( !afm_P || nPar!= afm_nPar ){
			  /* initialise the simplex memory structure: */
				reinit= True;
					if( verbose> 1 ){
						fprintf( StdErr, "Allocating simplex structure memory for %d dimensions\n", nPar );
					}
				if( !afm_P ){
					afm_nPar= nPar;
				}
				if( (afm_P= (double**) XGrealloc( afm_P, (nPar+2)* sizeof(double*))) ){
					for( i= afm_nPar+2; i< nPar+2; i++ ){
						afm_P[i]= NULL;
					}
				}
				if( afm_P && (afm_Pval= (double*) XGrealloc( afm_Pval, (nPar+2)* sizeof(double)))
					&& (afm_bestP= (double*) XGrealloc( afm_bestP, (nPar+1)* sizeof(double)))
				){
					afm_nPar= nPar;
					afm_P[0]= NULL;
					for( i= 1; i< nPar+2; i++ ){
						if( !(afm_P[i]= (double*) XGrealloc( afm_P[i], (nPar+1)* sizeof(double))) ){
							for( ; i> 0; i-- ){
								xfree(afm_P[i]);
							}
							ascanf_emsg= " (memory allocation problem) ";
							goto bail_out;
						}
					}
				}
				else{
					ascanf_emsg= " (memory allocation problem) ";
					goto bail_out;
				}
			}

			if( af_best ){
				afm_bestP[0]= 0;
				for( i= 1; i<= nPar; i++ ){
					afm_bestP[i]= af_best->array[i-1];
				}
			}
			else{
				for( i= 0; i<= nPar; i++ ){
					afm_bestP[i]= 0;
				}
			}

			afm_form= ASCB_COMPILED;
			afm_level= level;

			{ int av= ascanf_verbose;
				if( ascanf_verbose ){
					  /* avoid verbose output from the to-be-minimised procedure unless $verbose==2 */
					ascanf_verbose-= 1;
				}
				if( reinit ){
					sima_initial_simplex( afm_nPar, afm_P, afm_bestP, afm_Pval, ascanf_minimise, 
						&af_initial->array[-1], &af_min->array[-1], &af_max->array[-1], verbose
					);
					  /* Initialise the best function value ever with something large. This means a safe margin larger than
					   \ the figural distance at the "raw" configuration. The minimising function will in fact use this value
					   \ as a reference: if the initial value is lower than the real (global) minimum, no action occurs!
					   */
					if( bestPval<= DBL_MAX/10 ){
						bestPval*= 10;
					}
				}

				if( av || verbose ){
					fprintf( StdErr, "\nstarting with best value %s; %d iterations, %s:T=%g\n",
						d2str( bestPval,0,0),
						iter, (anneal)? "simulated annealing" : "Nelder & Mead simplex", anneal
					);
				}

				af_min_calls= 0;
				continuous_simanneal( afm_P, afm_Pval, nPar, afm_bestP, &bestPval, ftol, ascanf_minimise, &iter, anneal );
				ascanf_verbose= av;
			}

			if( ascanf_verbose || verbose ){
				fprintf( StdErr, " After %d iterations,ftol=%s: min@(%s",
					iter, ad2str(ftol, d3str_format,0), ad2str( afm_bestP[1], d3str_format, 0)
				);
				for( i= 2; i<= nPar; i++ ){
					fprintf( StdErr, ",%s", ad2str( afm_bestP[i], d3str_format, 0) );
				}
				fprintf( StdErr, ")=%s\n %s called %u times\n",
					ad2str( bestPval, d3str_format, NULL), af_minimise->name, af_min_calls
				);
				if( !af_best ){
					fprintf( StdErr, " ATTN: best values returned in initial values array %s!\n", af_initial->name );
				}
			}

minimised:;
			af_iter->value= iter;
			if( !af_best ){
				af_best= af_initial;
			}
			for( i= 0; i< nPar; i++ ){
				af_best->array[i]= afm_bestP[i+1];
			}
			*result= bestPval;
			af_min= NULL;
			af_max= NULL;

			Busy= False;
		}
	}
	else{
bail_out:;
		set_NaN(*result);
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

/* Minimise-SimplexAnneal-Full[&proc, nPar, &iter, ftol, anneal, &simplex[(nPar+1)*nPar], &best-P,min-Guess] */
int ascanf_SimplexAnneal_Full( ASCB_ARGLIST )
{ ASCB_FRAME
  double ftol, anneal;
  ascanf_Function *af_iter, *af_simplex, *af_best= NULL;
  int nPar, iter= 0, reinit= False, verbose= 0;
  static int _nPar= -1;
	ascanf_arg_error= 0;
	if( Busy ){
		ascanf_arg_error= True;
		ascanf_emsg= " (called recursively!) ";
	}
	if( ascanf_arguments>= 8 && !ascanf_arg_error ){
		if( !(af_minimise=
			parse_ascanf_address( args[0], _ascanf_procedure, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
		){
			ascanf_emsg= " (1st argument must be a pointer to a procedure) ";
			ascanf_arg_error= True;
		}
		if( args[1]<= 0 || args[1]>= MAXINT ){
			ascanf_emsg= " (2nd argument (nPar) must be a positive integer) ";
			ascanf_arg_error= !ascanf_SyntaxCheck;
		}
		else{
			nPar= (int) args[1];
		}
		if( !(af_iter=
			parse_ascanf_address( args[2], _ascanf_variable, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| af_iter->value<= 0 || af_iter->value>= MAXINT
		){
			ascanf_emsg= " (3rd argument must be a pointer to a positive integer) ";
			ascanf_arg_error= !ascanf_SyntaxCheck || !af_iter;
		}
		else{
			iter= (int) af_iter->value;
		}
		if( (ftol= args[3])<= 0 ){
			ascanf_emsg= " (4th argument must be positive and probably <1) ";
			ascanf_arg_error= True;
		}
		if( ASCANF_TRUE(args[4]) ){
			anneal= args[4];
		}
		else{
			anneal= 0;
		}
		if( !(af_simplex=
			parse_ascanf_address( args[5], _ascanf_array, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !af_simplex->array || (af_simplex->N!= (nPar+1)*nPar && !ascanf_SyntaxCheck)
		){
			ascanf_emsg= " (6th argument must be a pointer to a double array[(nPar+1)*nPar]) ";
			ascanf_arg_error= True;
		}
		if( !(af_best=
			parse_ascanf_address( args[6], _ascanf_array, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !af_best->array || (af_best->N< nPar && !ascanf_SyntaxCheck)
		){
			ascanf_emsg= " (7th argument must be a pointer to a double array[nPar]) ";
			ascanf_arg_error= True;
		}
		bestPval= args[7];
		if( ascanf_arguments> 8 && ASCANF_TRUE(args[8]) ){
			verbose= (int) args[8];
		}
		if( !ascanf_arg_error && !ascanf_SyntaxCheck && !NaNorInf(bestPval) ){
		  int i, r;

			Busy= True;

			if( !Resize_ascanf_Array( af_best, nPar, NULL) ){
				ascanf_emsg= " (memory allocation problem) ";
				goto bail_out;
			}

			afm_form= ASCB_COMPILED;
			afm_level= level;

			{ int av= ascanf_verbose;
				if( ascanf_verbose ){
					  /* avoid verbose output from the to-be-minimised procedure unless $verbose==2 */
					ascanf_verbose-= 1;
				}
				  /* simple check whether we got the memory. On a number of systems, ALLOCA (through alloca) won't
				   \ fail (= we'll die at once when it does). On others, we'll just suppose that P will be NULL too
				   \ when one of the preceding allocations failed...
				   */
				if( !afm_P_Full || nPar!= afm_nPar || _nPar!= nPar ){
				  /* initialise the simplex memory structure: */
					reinit= True;
					if( !afm_P_Full ){
						afm_nPar= nPar;
					}
					if( (afm_P_Full= (double**) XGrealloc( afm_P_Full, (nPar+2)* sizeof(double*)))
						&& (afm_Pval= (double*) XGrealloc( afm_Pval, (nPar+2)* sizeof(double)))
						&& (afm_bestP= (double*) XGrealloc( afm_bestP, (nPar+1)* sizeof(double)))
					){
						afm_nPar= nPar;
						_nPar= nPar;
						afm_P_Full[0]= NULL;
						for( i= 1; i< nPar+2; i++ ){
							  /* retrieve pointers to the appropriate places in the simplex array, making
							   \ sure each 1st element ends up at location 1 in the target array!
							   */
							afm_P_Full[i]= &( af_simplex->array[ (i-1)*nPar - 1 ] );
							afm_Pval[i]= ascanf_minimise_full( afm_P_Full[i] );
						}
					}
					else{
						ascanf_emsg= " (memory allocation problem) ";
						goto bail_out;
					}
				}

				if( af_best ){
					afm_bestP[0]= 0;
					for( i= 1; i<= nPar; i++ ){
						afm_bestP[i]= af_best->array[i-1];
					}
				}

				if( ascanf_verbose || verbose ){
					if( ascanf_verbose> 1 || verbose> 1 ){
					  int j;
						fprintf( StdErr, " Initial simplex: " );
						for( j= 1; j<= nPar+1; j++ ){
							fprintf( StdErr, "{%g", afm_P_Full[j][1] );
							for( i= 2; i<= nPar; i++ ){
								fprintf( StdErr, ",%g", afm_P_Full[j][i] );
							}
							fputs( "}", StdErr );
						}
						fputs( "\n", StdErr );
						fprintf( StdErr, " Initial simplex values: {%g", afm_Pval[1] );
						for( i= 2; i<= nPar+1; i++ ){
							fprintf( StdErr, ",%g", afm_Pval[i] );
						}
						fputs( "}\n", StdErr );
						fprintf( StdErr, " Initial best parameters: {%g", afm_bestP[1] );
						for( i= 2; i<= nPar; i++ ){
							fprintf( StdErr, ",%g", afm_bestP[i] );
						}
						fprintf( StdErr, "}\n" );
					}
					fprintf( StdErr, "\nstarting with best value %s; %d iterations, %s:T=%g\n",
						d2str( bestPval,0,0),
						iter, (anneal)? "simulated annealing" : "Nelder & Mead simplex", anneal
					);
				}

				af_min_calls= 0;
				continuous_simanneal( afm_P_Full, afm_Pval, nPar, afm_bestP, &bestPval, ftol, ascanf_minimise_full, &iter, anneal );
				ascanf_verbose= av;
			}

			if( ascanf_verbose || verbose ){
				fprintf( StdErr, " After %d iterations,ftol=%s: min@(%s",
					iter, ad2str(ftol, d3str_format,0), ad2str( afm_bestP[1], d3str_format, 0)
				);
				for( i= 2; i<= nPar; i++ ){
					fprintf( StdErr, ",%s", ad2str( afm_bestP[i], d3str_format, 0) );
				}
				fprintf( StdErr, ")=%s\n %s called %u times\n",
					ad2str( bestPval, d3str_format, NULL), af_minimise->name, af_min_calls
				);
			}

minimised:;
			af_iter->value= iter;
			for( i= 0; i< nPar; i++ ){
				af_best->array[i]= afm_bestP[i+1];
			}
			*result= bestPval;

			Busy= False;
		}
		else{
			set_NaN(*result);
		}
	}
	else{
bail_out:;
		set_NaN(*result);
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

int ascanf_initSimplex( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af_initial, *af_simplex;
  int nPar, verbose= 0;
	ascanf_arg_error= 0;
	if( Busy ){
		ascanf_arg_error= True;
		ascanf_emsg= " (called recursively!) ";
	}
	if( ascanf_arguments>= 5 && !ascanf_arg_error ){
		if( args[0]<= 0 || args[0]>= MAXINT ){
			ascanf_emsg= " (1st argument (nPar) must be a positive integer) ";
			ascanf_arg_error= !ascanf_SyntaxCheck;
		}
		else{
			nPar= (int) args[0];
		}
		if( !(af_initial=
			parse_ascanf_address( args[1], _ascanf_array, "ascanf_initSimplex", (int) ascanf_verbose, NULL ))
			|| !af_initial->array || af_initial->N< nPar
		){
			ascanf_emsg= " (2nd argument must be a pointer to a double array[nPar]) ";
			ascanf_arg_error= True;
		}
		if( !(af_min=
			parse_ascanf_address( args[2], _ascanf_array, "ascanf_initSimplex", (int) ascanf_verbose, NULL ))
			|| !af_min->array || af_min->N< nPar
		){
			ascanf_emsg= " (3rd argument must be a pointer to a double array[nPar]) ";
			ascanf_arg_error= True;
		}
		if( !(af_max=
			parse_ascanf_address( args[3], _ascanf_array, "ascanf_initSimplex", (int) ascanf_verbose, NULL ))
			|| !af_max->array || af_max->N< nPar
		){
			ascanf_emsg= " (4th argument must be a pointer to a double array[nPar]) ";
			ascanf_arg_error= True;
		}
		if( !(af_simplex=
			parse_ascanf_address( args[4], _ascanf_array, "ascanf_SimplexAnneal", (int) ascanf_verbose, NULL ))
			|| !af_simplex->array || (af_simplex->N!= (nPar+1)*nPar && !ascanf_SyntaxCheck)
		){
			ascanf_emsg= " (5th argument must be a pointer to a double array[(nPar+1)*nPar]) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 5 && ASCANF_TRUE(args[5]) ){
			verbose= (int) args[5];
		}
		if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
		  int i, r;

			Busy= True;

			if( !Resize_ascanf_Array( af_initial, nPar, NULL) ||
				!Resize_ascanf_Array( af_min, nPar, NULL) ||
				!Resize_ascanf_Array( af_max, nPar, NULL)
			){
				ascanf_emsg= " (memory allocation problem) ";
				ascanf_arg_error= True;
				goto bail_out;
			}

			afm_form= ASCB_COMPILED;
			afm_level= level;

			{ int av= ascanf_verbose;
				if( ascanf_verbose ){
					  /* avoid verbose output from the to-be-minimised procedure unless $verbose==2 */
					ascanf_verbose-= 1;
				}
				  /* simple check whether we got the memory. On a number of systems, ALLOCA (through alloca) won't
				   \ fail (= we'll die at once when it does). On others, we'll just suppose that P will be NULL too
				   \ when one of the preceding allocations failed...
				   */
				if( !afm_P_Full || nPar!= afm_nPar ){
				  /* initialise the simplex memory structure: */
					if( !afm_P_Full ){
						afm_nPar= nPar;
					}
					  /* We don't really use afm_Pval and afm_bestP in this function, but to ensure
					   \ that they always match afm_P_Full in dimensions, we (re)allocate them here too.
					   */
					if( (afm_P_Full= (double**) XGrealloc( afm_P_Full, (nPar+2)* sizeof(double*)))
						&& (afm_Pval= (double*) XGrealloc( afm_Pval, (nPar+2)* sizeof(double)))
						&& (afm_bestP= (double*) XGrealloc( afm_bestP, (nPar+1)* sizeof(double)))
					){
						afm_nPar= nPar;
						afm_P_Full[0]= NULL;
						for( i= 1; i< nPar+2; i++ ){
							  /* retrieve pointers to the appropriate places in the simplex array, making
							   \ sure each 1st element ends up at location 1 in the target array!
							   */
							afm_P_Full[i]= &( af_simplex->array[ (i-1)*nPar - 1 ] );
						}
					}
					else{
						ascanf_emsg= " (memory allocation problem) ";
						goto bail_out;
					}
				}
				sima_initial_simplex( afm_nPar, afm_P_Full, NULL, NULL, NULL, 
					&af_initial->array[-1], &af_min->array[-1], &af_max->array[-1], verbose
				);
				ascanf_verbose= av;
			}

			Busy= False;
		}
		*result= afm_nPar;
		af_min= NULL;
		af_max= NULL;
	}
	else{
bail_out:;
		set_NaN(*result);
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

/* SimplexDataMinimise[ &function, &parameters, &Xparameter, &Yparameter, loss, &xvals, &yvals[, &zvals[, &N_return[,verbose]]] ]	*/
int ascanf_SimplexDataMinimise( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af_parameters, *af_XPar, *af_YPar= NULL, *af_Xvals, *af_Yvals, *af_Zvals= NULL, *af_N= NULL;
  int verbose= 0, loss= 0, afm_N= afm_nPar, *afm_L= afm_level;
  ascanf_Function *afm= af_minimise;
	ascanf_arg_error= 0;
	if( ascanf_arguments>= 7 && !ascanf_arg_error ){
		  /* re-use af_minimise; not that we ought to get called from "within af_minimise"!!! */
		if( !(af_minimise=
			parse_ascanf_address( args[0], _ascanf_procedure, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
		){
			ascanf_emsg= " (1st argument must be a pointer to a procedure) ";
			ascanf_arg_error= True;
		}
		if( !(af_parameters=
			parse_ascanf_address( args[1], _ascanf_array, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
			|| !af_parameters->array || (af_parameters->N!= afm_nPar && !ascanf_SyntaxCheck)
		){
			if( af_parameters && af_parameters->N!= afm_nPar ){
				fprintf( StdErr, " (warning: %s has %d not %d dimension!)== ",
					af_parameters->name, af_parameters->N, afm_nPar
				);
				fflush( StdErr );
				afm_nPar= af_parameters->N;
			}
			else{
				ascanf_emsg= " (2nd argument must be a pointer to a double array) ";
				ascanf_arg_error= True;
			}
		}
		if( !(af_XPar=
			parse_ascanf_address( args[2], _ascanf_variable, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
		){
			ascanf_emsg= " (3rd argument must be a pointer to a scalar variable) ";
			ascanf_arg_error= True;
		}
		if( args[3] && !(af_YPar=
			parse_ascanf_address( args[3], _ascanf_variable, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
		){
			ascanf_emsg= " (4th argument must be a pointer to a scalar variable or 0) ";
		}
		  /* not yet supported: */
/* 		CLIP_EXPR( loss, (int) args[4], 0, MAXINT );	*/
		if( !(af_Xvals=
			parse_ascanf_address( args[5], _ascanf_array, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
			|| !af_Xvals->array
		){
			ascanf_emsg= " (6th argument must be a pointer to a double array) ";
			ascanf_arg_error= True;
		}
		if( !(af_Yvals=
			parse_ascanf_address( args[6], _ascanf_array, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
			|| !af_Yvals->array || (af_Yvals->N!= af_Xvals->N && !ascanf_SyntaxCheck)
		){
			ascanf_emsg= " (7th argument must be a pointer to a double array of the same dimension as XVals) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 7 && args[7] ){
			if( !(af_Zvals=
				parse_ascanf_address( args[7], _ascanf_array, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
				|| !af_Zvals->array || (af_Zvals->N!= af_Xvals->N && !ascanf_SyntaxCheck)
			){
				ascanf_emsg= " (8th argument must be a pointer to a double array of the same dimension as XVals) ";
				ascanf_arg_error= True;
			}
		}
		if( ascanf_arguments> 8 && args[8] ){
			if( !(af_N=
				parse_ascanf_address( args[8], _ascanf_variable, "ascanf_SimplexDataMinimise", (int) ascanf_verbose, NULL ))
			){
				ascanf_emsg= " (9th argument must be a pointer to a scalar variable or 0) ";
			}
		}
		if( ascanf_arguments> 9 && ASCANF_TRUE(args[9]) ){
			verbose= (int) args[9];
		}
		if( (af_YPar==NULL) ^ (af_Zvals==NULL) ){
			fprintf( StdErr, " (warning: Y-parameter==%s while Z-values==%s: ignoring Z-values!)== ",
				ad2str( args[4], d3str_format,0), ad2str( args[7], d3str_format,0 )
			);
			fflush( StdErr );
			if( af_YPar ){
				af_YPar->value= 0;
			}
			af_Zvals= NULL;
		}
		  /* should be redundant: ! */
		if( af_Zvals && !af_YPar ){
			ascanf_emsg= " (internal error: af_Zvals && !af_YPar)== ";
			ascanf_arg_error= 1;
		}
		if( !afm_level && !ascanf_SyntaxCheck ){
			fprintf( StdErr, " (Warning: SimplexDataMinimise called outside valid context!)== " );
			fflush( StdErr );
			afm_level= level;
		}
		if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
		  unsigned int i;
		  double predicted, R= 0, N= 0;
		  int afmc= af_min_calls;

			if( af_min && af_max ){
			  int j;
				for( j= 0; j< afm_nPar; j++ ){
					CLIP( af_parameters->array[j], af_min->array[j], af_max->array[j] );
				}
			}
			for( i= 0; i< af_Xvals->N; i++ ){
				af_XPar->value= af_Xvals->array[i];
				if( af_Zvals ){
				  /* 2D function minimisation; */
					af_YPar->value= af_Yvals->array[i];
				}
				predicted= _ascanf_minimise_full( af_parameters->array );
/* 				switch( loss )... :	*/
				if( af_Zvals ){
					R+= ( af_Zvals->array[i]- predicted ) * ( af_Zvals->array[i]- predicted );
				}
				else{
					R+= ( af_Yvals->array[i]- predicted ) * ( af_Yvals->array[i]- predicted );
				}
				N+= 1;
			}
			  /* af_min_calls is supposed to keep track of the number of times continuous_simanneal() called
			   \ the minimsation function. If inside it we call it again ourselves, that shouldn't be counted...
			   \ So, since we are supposed to be called through _ascanf_minimise_full() (which increments
			   \ af_min_calls), return leaving it at the value we found it...
			   */
			af_min_calls= afmc;
			*result= R/N;
			if( af_N ){
				af_N->value= N;
			}
			if( verbose ){
				fprintf( StdErr, " (loss #%d= %s/%s)== ",
					loss,
					ad2str( R, d3str_format,0), ad2str( N, d3str_format,0)
				);
				if( !ascanf_verbose ){
					fprintf( StdErr, "%s\n", ad2str( *result, d3str_format,0) );
				}
				fflush( StdErr );
			}
		}
	}
	else{
bail_out:;
		set_NaN(*result);
		if( af_N ){
			af_N->value= 0;
		}
		ascanf_arg_error= True;
	}
	  /* reset af_minimise, otherwise we won't be called again! */
	af_minimise= afm;
	afm_nPar= afm_N;
	afm_level= afm_L;
	return( !ascanf_arg_error );
}

static void afm_cleanup()
{ int i;
	if( Busy ){
		ascanf_arg_error= True;
		ascanf_emsg= " (won't cleanup while busy!) ";
	}
	else{
		if( afm_P ){
			for( i= 1; i< afm_nPar+2; i++ ){
				xfree(afm_P[i]);
			}
		}
		xfree(afm_P);
		  /* afm_P_Full contains pointers to memory that's not ours: DON'T FREE IT HERE! */
		xfree(afm_P_Full);
		xfree(afm_Pval);
		xfree(afm_bestP);
		afm_nPar= 0;
	}
}

int ascanf_SimplexAnneal_Finished( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	ascanf_arg_error= 0;
	afm_cleanup();
	return(!ascanf_arg_error);
}

static ascanf_Function simanneal_Function[] = {
	{ "sran-PM", ascanf_sran_PM, 1, NOT_EOF,
		"sran-PM[seed]: seed the ran-PM random generator:\n"
		" ran-PM is a \"minimal\" random number generator of Park and Miller with Bays-Durham shuffle and added\n"
		" safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint\n"
		" values) . Call with seed a negative integer to initialize; thereafter, do not alter seed between\n"
		" successive deviates in a sequence.\n"
	},
	{ "ran-PM", ascanf_ran_PM, 3, NOT_EOF,
		"ran-PM or ran-PM[low,high[,cond]]: the optional <cond> argument can be:\n"
		" <1: impose this as the minimal fractional difference of the <low>,<high> range\n"
		"     between the generated and the previous value\n"
		" >=1: perform this many polls of the generator before returning a value\n"
		" <=-1: perform between 0 and this many polls of the generator before returning a value\n"
		" See sran-PM[].\n"
	},
	{ "Minimise-SimplexAnneal-Full", ascanf_SimplexAnneal_Full, 9, NOT_EOF_OR_RETURN,
		"Minimise-SimplexAnneal-Full[&proc,nPar,&iter,ftol,anneal,&simplex,&best-P,min-Guess[,verbose]]:\n"
		" Find the minimum of procedure <proc> over its <nPar> parameters P that are passed as arguments to\n"
		" <proc>. A maximum of <iter> (a pos. integer) iterations is made; the actual number is returned in <iter>.\n"
		" The function returns the minimum found, and the best parameters for which this value occurs in the double\n"
		" array <best-P>. If <anneal> is True, it gives the simulated annealing temperature, otherwise the\n"
		" Nelder & Mead simplex method is used. The starting simplex (which is updated at each call) is given in\n"
		" <simplex>, that must point to an array to be indexed as @[&simplex,0,0, nPar+1, nPar]: nPar+1 combinations\n"
		" of the nPar parameters to be varied. The initial guess for the minimum is to be passed in <min-Guess>:\n"
		" the found minimum will never be larger than this value. When using simulated annealing, decrease <anneal>\n"
		" reset <iter> and pass the previous return value in <min-Guess>.\n"
		" If <proc> returns a NaN or Inf, this value is replaced by the system's DBL_MAX to make sure that the\n"
		" the underlying minimisation algorithm can continue correctly. This is done by both this function and\n"
		" its convenience version Minimise-SimplexAnneal[].\n"
		" Modifications by <proc> to the currently 'tested' parameters are taken into account: it is thus possible\n"
		" e.g. to limit the scanned range.\n"
		" These routines can not be called recursively.\n"
		" <proc> can be either an ascanf procedure, or a Python function.\n"
		,0, 1,
	},
	{ "Minimise-SimplexAnneal", ascanf_SimplexAnneal, 10, NOT_EOF_OR_RETURN,
		"Minimise-SimplexAnneal[&proc,nPar,&iter,ftol,anneal,&initial-P,&min-P,&max-P[,&best-P[,verbose]]]:\n"
		" Find the minimum of procedure <proc> over its <nPar> parameters P that are passed as arguments to\n"
		" <proc>. The initial values are specified in the double array <initial-P>, the ranges in\n"
		" the double arrays <min-P> and <max-P>. A maximum of <iter> (a pos. integer) iterations is made\n"
		" the actual number is returned in <iter>. The function returns the minimum found, and the best\n"
		" parameters for which this value occurs in the double array <best-P> if it is given, otherwise in\n"
		" <initial-P>. All arrays must have at least nPar elements, with the exception of best-P; excess\n"
		" elements are trimmed. If <anneal> is True, it gives the simulated annealing temperature, otherwise the\n"
		" Nelder & Mead simplex method is used. This routine is a convenience version of the \"full\" version.\n"
		,0, 1,
	},
	{ "Initialise-Simplex", ascanf_initSimplex, 6, NOT_EOF_OR_RETURN,
		"Initialise-Simplex[nPar, &initial-P, &min-P, &max-P, &simplex[,verbose]]: initialise <simplex> for\n"
		" use e.g. in Minimise-SimplexAnneal-Full[] using <initial-P>, <min-P> and <max-P> as described for\n"
		" Minimise-SimplexAnneal[].\n"
	},
	{ "$Simplex-Initialisation", NULL, 2, _ascanf_variable,
		"$Simplex-Initialisation: the way simplexes are initialised by Minimise-SimplexAnneal[] and Initialise-Simplex[]:\n"
		" 0: one element has the initial par. values, the others are initialised at regular intervals between the defined\n"
		"    min/max boundaries. The simplex is then shuffled randomly.\n"
		" 1: one element has the initial par. values, the others are initialised with random values drawn from the\n"
		"    defined min/max intervals.\n"
		" 2: one element has the initial par. values, the others are initialised as random values in initial +- stdv\n"
		"    where <stdv> is calculated as abs(max-min)/2.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "Minimise-SimplexAnneal-Finished", ascanf_SimplexAnneal_Finished, 0, NOT_EOF_OR_RETURN,
		"Minimise-SimplexAnneal-Finished: deallocates simplex memory structures\n"
	},
	{ "SimplexDataMinimise", ascanf_SimplexDataMinimise, 10, NOT_EOF_OR_RETURN,
		"SimplexDataMinimise[&function,&parameters,&Xparameter,&Yparameter,loss,&xvals,&yvals[,&zvals[,&N_return[,verbose]]]]:\n"
		" convenience routine to speed up a bit certain fitting/optimisation operations. The to-be-optimised <function> with\n"
		" <parameters> (an array) is applied to the data in <xvals>, <yvals> and <zvals> (all arrays of identical dimension)\n"
		" and a loss function-value is calculated (and returned) based on the values predicted by <function> and the observed\n"
		" values. The individual data values are passed to <function> in <Xparameter> and <Yparameter>, hence NOT as an argument\n"
		" to that function. When Yparameter and zvals are both valid pointers (not 0), function is supposed to be 2-D, and the\n"
		" observed values are obtained from zvals. Otherwise, function is supposed to be 1-D, and the observed values are \n"
		" obtained in yvals (if zvals=0 and Yparameter is a valid pointer, that variable will be set to 0). The <loss> parameter\n"
		" specifies the loss function to calculate; currently only a least-squares ( (observed-predicted)**2/N ) is provided.\n"
		" If <N_return> is a valid pointer, the number of observations (N) is returned in it.\n"
		, 0, 1,
	},
};
static int simanneal_Functions= sizeof(simanneal_Function)/sizeof(ascanf_Function);

static int initialised= False;

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= simanneal_Function;
  static char called= 0;
  int i, warn= True;
  char buf[64];

	initSimplexType= &simanneal_Function[5].value;

	for( i= 0; i< simanneal_Functions; i++, af++ ){
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

DyModTypes simanneal_initDyMod( INIT_DYMOD_ARGUMENTS )
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
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( simanneal_Function, simanneal_Functions, "simanneal::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-simplex-&-simulated-annealing" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		"A library containing a routine for minimisation via simplex & simulated annealing techniques.\n"
	);

/* 	sran1( - (long) time(NULL) );	*/

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
int simanneal_closeDyMod( DyModLists *target, int force )
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
	Busy= False;
	if( initialised ){
	  int r= remove_ascanf_functions( simanneal_Function, simanneal_Functions, force );
		if( force || r== simanneal_Functions ){
			afm_cleanup();
			for( i= 0; i< simanneal_Functions; i++ ){
				simanneal_Function[i].dymod= NULL;
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
				r, simanneal_Functions
			);
		}
	}
	return(ret);
}

