#include "config.h"
IDENTIFY( "Stats ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
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

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc).
   */

#include <float.h>

#define DYMOD_MAIN
#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;

	void (*register_DoubleWithIndex_ptr)( double value, long idx );
	long (*get_IndexForDouble_ptr)( double value );
	void (*delete_IndexForDouble_ptr)( double value );
	double (*lm_fit_ptr)(
		double *x, double *y, double *stdv, unsigned long N,
		double *slope, double *icept,
		double *Pslope, double *Picept, double *goodness
	);
	double (*rlm_fit_ptr)(
		double *x, double *y, unsigned long N,
		double *slope, double *icept
	);

#	define register_DoubleWithIndex	(*register_DoubleWithIndex_ptr)
#	define get_IndexForDouble	(*get_IndexForDouble_ptr)
#	define delete_IndexForDouble	(*delete_IndexForDouble_ptr)
#	define lm_fit				(*lm_fit_ptr)
#	define rlm_fit				(*rlm_fit_ptr)


typedef struct HistogramEntry{
	double value;
	unsigned long N;
} HistogramEntry;

static int qs_ints( int *a, int *b )
{
	return( *a - *b );
}

static int qs_doubles( double *a, double *b )
{
	if( *a < *b ){
		return(-1);
	}
	else if( *a > *b ){
		return(1);
	}
	else{
		return(0);
	}
}

int ascanf_MakeHistogram ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *YVal, *XVal= NULL, *histogram, *classify= NULL;
  int normalise= False;
	if( !args || ascanf_arguments< 4 ){
		ascanf_emsg= " (need at least 3 arguments)";
		ascanf_arg_error= 1;
	}
	else{
	  HistogramEntry *Histogram;
	  long N= 0;
	  int resize_XVal;
		if( !(YVal= parse_ascanf_address( args[0], _ascanf_array, "ascanf_MakeHistogram", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (1st argument must be an array)== ";
			ascanf_arg_error= 1;
		}
		else if( !(XVal= parse_ascanf_address( args[2], _ascanf_array, "ascanf_MakeHistogram", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (3rd argument must be an array)== ";
			ascanf_arg_error= 1;
		}
		else if( !(histogram= parse_ascanf_address( args[3], _ascanf_array, "ascanf_MakeHistogram", (int) ascanf_verbose, NULL ))
			|| histogram->iarray
		){
			ascanf_emsg= " (4th argument must be a double array)== ";
			ascanf_arg_error= 1;
		}
		resize_XVal= (ASCANF_TRUE(args[1]))? False : True;
		if( ascanf_arguments> 4 && ASCANF_TRUE(args[4]) ){
			normalise= (int) args[4];
		}
		else{
			normalise= 0;
		}
#if 0
		  /* For more elegance, one could allow the passing of a classification method, which would transform
		   \ the YValues into (e.g.) the desired bins (e.g. [1.5,2.5] -> 2).
		   \ This would make the routine quite a bit more complex, and the functionality can be obtained by
		   \ a separate call to e.g. Apply2Array. The drawback of that is that it would need to be done on a copy
		   \ of the data.
		   */
		if( ascanf_arguments> 5 ){
			classify= parse_ascanf_address( args[5], 0, "ascanf_MakeHistogram", (int) ascanf_verbose, NULL );
		}
		if( classify ){
			switch( classify->type ){
				case _ascanf_procedure:
				case NOT_EOF:
				case NOT_EOF_OR_RETURN:
				case _ascanf_classifyction:
					if( pragma_unlikely(ascanf_verbose) && strncmp(classify->name, "\\l\\expr-",8)== 0 ){
						fprintf( StdErr, "\n#%s%d\\l\\\tlambda[", (__ascb_frame->compiled)? "C" : "", *__ascb_frame->level );
						Print_Form( StdErr, &classify->procedure, 0, True, NULL, "#\\l\\\t", NULL, False );
						fputs( "]\n", StdErr );
					}
					break;
				default:
					ascanf_emsg= " (unsupported method (6th argument) in Make-Histogram) ";
					classify= NULL;
					break;
			}
		}
#endif
		if( !ascanf_arg_error ){
			set_NaN(*result);
			if( XVal->N== 1 || resize_XVal ){
				Resize_ascanf_Array( XVal, YVal->N, NULL );
				if( XVal->iarray && YVal->iarray ){
				  /* both int arrays: no non-numeric values are possible, so copy fast: */
					memcpy( XVal->iarray, YVal->iarray, sizeof(int) * YVal->N );
				}
				else{
				  int i, j= 0;
				  double y;
					  /* at least one double array. Conversion is necessary (= per-element copying),
					   \ and we can just as well filter out possible non-numeric values:
					   */
					for( i= 0; i< YVal->N; i++ ){
						y= ASCANF_ARRAY_ELEM(YVal,i);
						if( !NaN(y) ){
							ASCANF_ARRAY_ELEM_SET( XVal, j, y );
							j+= 1;
						}
					}
					if( j!= XVal->N ){
						Resize_ascanf_Array( XVal, j, NULL );
					}
				}
				resize_XVal= True;
			}
			else{
			  /* NB: we don't verify for the presence of non-numeric values in a specified XVal array!
			   \ You want 'em, you get 'em...
			   */
				resize_XVal= False;
			}
			  /* A histogram is usually sorted, but again we only do this when we're discovering the X values
			   \ ourselves. User wants (= gives) an unsorted XVal, he'll get one back...
			   */
			if( resize_XVal ){
				if( XVal->iarray ){
					qsort( XVal->iarray, XVal->N, sizeof(XVal->iarray[0]), (void*) qs_ints );
				}
				else{
					qsort( XVal->array, XVal->N, sizeof(XVal->array[0]), (void*) qs_doubles );
				}
			}
			if( (Histogram= (HistogramEntry*) calloc( XVal->N, sizeof(HistogramEntry) )) ){
			  int i= 0;
			  double x, px;
				  /* construct the histogram table, pass 1: the X values. */
				set_NaN(px);
#if 0
				do{
					px= ASCANF_ARRAY_ELEM(XVal,i);
					if( !NaN(px) ){
						Histogram[i].value= px;
						register_DoubleWithIndex( px, i );
						ok= True;
					}
					else{
						ok= False;
					}
					i+= 1;
				} while( !ok );
				N= 1;
#else
				N= 0;
#endif
				  /* i is initialised here */
				for( ; i< XVal->N; i++ ){
					x= ASCANF_ARRAY_ELEM(XVal,i);
					  /* If we built XVal ourselves, there should be no NaNs. We just check as a protection against
					   \ user-specified NaNs. OTOH, he wants 'em, he should get 'em....
					   */
					if( !NaN(x) && x != px ){
						Histogram[N].value= x;
						  /* For fast access in pass 2, associate the X value's index (entry into Histogram) with
						   \ the value:
						   */
						register_DoubleWithIndex( x, N );
						px= x;
						N+= 1;
					}
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (%ld unique numeric X values", N );
				}
				  /* Get/resize the appropriately size memory buffers: */
				if( (Histogram= (HistogramEntry*) realloc( Histogram, N * sizeof(HistogramEntry) ))
					&& (!resize_XVal || Resize_ascanf_Array(XVal, N, NULL))
					&& (histogram==YVal || Resize_ascanf_Array(histogram, N, NULL))
				){
				  double y;
				  long idx, msIdx;
				  double M= 0, maxScore= -1;
					for( i= 0; i< YVal->N; i++ ){
						y= ASCANF_ARRAY_ELEM(YVal,i);
						  /* Only consider values in YVal that are numeric, and that exist in Histogram:
						   \ the value/index association allows to do the exclusion and entry lookup in one step:
						   */
						if( !NaN(y) && (idx= get_IndexForDouble(y))>= 0 ){
							Histogram[idx].N+= 1;
							M+= 1;
							if( normalise==2 && Histogram[idx].N> maxScore ){
								maxScore= Histogram[idx].N;
								msIdx= idx;
							}
						}
					}
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "; %g values scored", M );
						if( normalise== 2 ){
							fprintf( StdErr, "; max. score %g at %ld==%g", maxScore, msIdx, Histogram[msIdx].value );
						}
						fputs( ")== ", StdErr );
					}
					if( histogram== YVal && histogram->N!= N ){
						Resize_ascanf_Array(histogram, N, NULL);
					}
					if( histogram ){
						  /* Now store the values in the user/output array(s): */
						for( i= 0; i< N; i++ ){
							if( resize_XVal ){
								ASCANF_ARRAY_ELEM_SET(XVal,i, Histogram[i].value);
							}
							switch( normalise ){
								default:
								case 0:
									histogram->array[i]= Histogram[i].N;
									break;
								case 1:
									histogram->array[i]= ((double) Histogram[i].N / M);
									break;
								case 2:
									histogram->array[i]= ((double) Histogram[i].N / maxScore);
									break;
							}
							  /* don't forget to unbuild the associative table. This could be more elegant:
							   \ using a dedicated table that we can destroy with a single command.
							   */
							delete_IndexForDouble( Histogram[i].value );
						}
						histogram->last_index= -1;
						if( histogram->accessHandler ){
							AccessHandler( histogram, "ascanf_MakeHistogram", level, ASCB_COMPILED, AH_EXPR, NULL );
						}
						*result= M;
					}
					else{
						fprintf( StdErr, " (Can't resize orignal data==histogram array to %d: %s)== ", N, serror() );
					}
					xfree(Histogram);
					if( resize_XVal && XVal->accessHandler ){
						XVal->last_index= -1;
						AccessHandler( XVal, "ascanf_MakeHistogram", level, ASCB_COMPILED, AH_EXPR, NULL );
					}
				}
				else{
					fprintf( StdErr, " (Can't resize internal memory from %d to %d: %s)== ", XVal->N, N, serror() );
				}
			}
			else{
				fprintf( StdErr, " (Can't allocate memory for #%d internal buffer: %s)== ", XVal->N, serror() );
			}
		}
	}
	return( !ascanf_arg_error );
}

unsigned long linfit_nan_handling( double *xx, double *yy, double *ee, ascanf_Function *XVal, ascanf_Function *YVal, ascanf_Function *EVal )
{ unsigned long i, N= 0;
  int xOK, yOK, eOK;
  double xVal, yVal, eVal;
	for( i= 0; i< YVal->N; i++ ){
		yVal= ASCANF_ARRAY_ELEM(YVal,i);
		yOK= (isNaN(yVal))? 0 : 1;
		if( XVal ){
			xVal= ASCANF_ARRAY_ELEM(XVal,i);
			xOK= (isNaN(xVal))? 0 : 1;
		}
		else{
			xVal= i;
			xOK= 1;
		}
		if( EVal ){
			eVal= ASCANF_ARRAY_ELEM(EVal,i);
			eOK= (isNaN(eVal))? 0 : 1;
		}
		else{
			eOK= 1;
		}
		if( xOK && yOK && eOK ){
			xx[N] = xVal;
			yy[N] = yVal;
			if( ee ){
				ee[N] = eVal;
			}
#if 0
/* Something in my standard combination of full optimisation flags causes gcc-4.0.0 (Apple 4061) to miscompile
\ something in my NaN macros.
*/
			if( isnan(yVal) ){
			  IEEEfp *IE= I3Ed(yVal);
				fprintf(StdErr, " %d=%d(%s,%s)", N, i,
					d2str(xVal,0,0), d2str(yVal,0,0)
				);
				fprintf( StdErr, "l:h=0x%lx:%lx s:e:f1:f2=0x%lx:%lx:%lx:%lx (%d)",
					IE->l.high, IE->l.low,
					IE->s.s, IE->s.e, IE->s.f1, IE->s.f2,
					(IE->s.e==NAN_VAL)
				);
			}
#endif
			N+= 1;
		}
	}
	return(N);
}

int ascanf_lmFit ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *YVal=NULL, *XVal= NULL, *EVal= NULL, *Slope, *ICept, *probS= NULL, *probI= NULL, *GOF= NULL;
	set_NaN(*result);
	if( !args || ascanf_arguments< 5 ){
		ascanf_emsg= " (need at least 5 arguments)";
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0;
		ascanf_arg_error= 0;
		if( args[0] && !(XVal= parse_ascanf_address( args[0], _ascanf_array, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (1st argument must be an array or 0)== ";
			ascanf_arg_error= 1;
		}
		if( !(YVal= parse_ascanf_address( args[1], _ascanf_array, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ||
			(XVal && XVal->N< YVal->N)
		){
			ascanf_emsg= " (2nd argument must be an array with not more elements than the 1st (X) array)== ";
			ascanf_arg_error= 1;
		}
		if( args[2] && !(EVal= parse_ascanf_address( args[2], _ascanf_array, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (3rd argument must be an array or 0)== ";
			ascanf_arg_error= 1;
		}
		if( !(Slope= parse_ascanf_address( args[3], _ascanf_variable, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (4th argument must be a scalar)== ";
			ascanf_arg_error= 1;
		}
		if( !(ICept= parse_ascanf_address( args[4], _ascanf_variable, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (5th argument must be a scalar)== ";
			ascanf_arg_error= 1;
		}
		if( ascanf_arguments> 5 && args[5] ){
			if( !(probS= parse_ascanf_address( args[5], _ascanf_variable, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ){
				ascanf_emsg= " (6th argument must be a scalar)== ";
				ascanf_arg_error= 1;
			}
		}
		if( ascanf_arguments> 6 && args[6] ){
			if( !(probI= parse_ascanf_address( args[6], _ascanf_variable, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ){
				ascanf_emsg= " (7th argument must be a scalar)== ";
				ascanf_arg_error= 1;
			}
		}
		if( ascanf_arguments> 7 && args[7] ){
			if( !(GOF= parse_ascanf_address( args[7], _ascanf_variable, "ascanf_lmFit", (int) ascanf_verbose, NULL )) ){
				ascanf_emsg= " (8th argument must be a scalar)== ";
				ascanf_arg_error= 1;
			}
		}
		if( !ascanf_arg_error && YVal && Slope && ICept && !ascanf_SyntaxCheck ){
		  unsigned long i;
		  double *ee= NULL, Pslope, Picept, goodness;
		  ALLOCA( xx, double, YVal->N, xx_len );
		  ALLOCA( yy, double, YVal->N, yy_len );
			if( EVal ){
				ee= (double*) calloc( YVal->N, sizeof(double) );
			}
			  /* First, copy all input values into local buffers, excluding the NaNs.
			   \ If X was missing, use the counter (of ALL values, not just of the valid sample!!)
			   \ NB: a single NaN in each of the co-ordinates invalidates the sample!
			   */
			N= linfit_nan_handling( xx, yy, ee, XVal, YVal, EVal );
#if 0
			fprintf( StdErr, " i=%lu, YVal->N=%lu, N=%lu\n", i, YVal->N, N );
#endif

			*result= lm_fit( xx, yy, ee, N, &Slope->value, &ICept->value, &Pslope, &Picept, &goodness );

			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (lm fit, N=%lu ", N );
				if( ee ){
					fprintf( StdErr, "%s+-%s", YVal->name, EVal->name );
				}
				else{
					fprintf( StdErr, "%s", YVal->name );
				}
				fprintf( StdErr, " ~ %s * %s + %s; uncert.Slope=%s, uncert.ICept=%s, GoF=%s, chi2=%s) ",
					ad2str( Slope->value, d3str_format, 0),
					(XVal)? XVal->name : "$Counter",
					ad2str( ICept->value, d3str_format, 0),
					ad2str( Pslope, d3str_format,0), ad2str( Picept, d3str_format, 0),
					ad2str( goodness, d3str_format,0), ad2str( *result, d3str_format, 0)
				);
			}

			xfree(ee);

			if( Slope->accessHandler ){
				AccessHandler( Slope, "ascanf_lmFit", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			if( ICept->accessHandler ){
				AccessHandler( ICept, "ascanf_lmFit", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			if( probS ){
				probS->value= Pslope;
				if( probS->accessHandler ){
					AccessHandler( probS, "ascanf_lmFit", level, ASCB_COMPILED, AH_EXPR, NULL );
				}
			}
			if( probI ){
				probI->value= Picept;
				if( probI->accessHandler ){
					AccessHandler( probI, "ascanf_lmFit", level, ASCB_COMPILED, AH_EXPR, NULL );
				}
			}
			if( GOF ){
				GOF->value= goodness;
				if( GOF->accessHandler ){
					AccessHandler( GOF, "ascanf_lmFit", level, ASCB_COMPILED, AH_EXPR, NULL );
				}
			}
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_rlmFit ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *YVal=NULL, *XVal= NULL, *Slope, *ICept;
	set_NaN(*result);
	if( !args || ascanf_arguments< 4 ){
		ascanf_emsg= " (need at least 4 arguments)";
		ascanf_arg_error= 1;
	}
	else{
	  unsigned long N= 0;
		ascanf_arg_error= 0;
		if( args[0] && !(XVal= parse_ascanf_address( args[0], _ascanf_array, "ascanf_rlmFit", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (1st argument must be an array or 0)== ";
			ascanf_arg_error= 1;
		}
		if( !(YVal= parse_ascanf_address( args[1], _ascanf_array, "ascanf_rlmFit", (int) ascanf_verbose, NULL )) ||
			(XVal && XVal->N< YVal->N)
		){
			ascanf_emsg= " (2nd argument must be an array with not more elements than the 1st (X) array)== ";
			ascanf_arg_error= 1;
		}
		if( !(Slope= parse_ascanf_address( args[2], _ascanf_variable, "ascanf_rlmFit", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (3rd argument must be a scalar)== ";
			ascanf_arg_error= 1;
		}
		if( !(ICept= parse_ascanf_address( args[3], _ascanf_variable, "ascanf_rlmFit", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (4th argument must be a scalar)== ";
			ascanf_arg_error= 1;
		}
		if( !ascanf_arg_error && YVal && Slope && ICept && !ascanf_SyntaxCheck ){
		  unsigned long i;
		  ALLOCA( xx, double, YVal->N, xx_len );
		  ALLOCA( yy, double, YVal->N, yy_len );
			  /* First, copy all input values into local buffers, excluding the NaNs.
			   \ If X was missing, use the counter (of valid values!)
			   \ NB: a single NaN in each of the co-ordinates invalidates the sample!
			   */
			N= linfit_nan_handling( xx, yy, NULL, XVal, YVal, NULL );

			*result= rlm_fit( xx, yy, N, &Slope->value, &ICept->value );

			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (rlm fit, N=%lu ", N );
				fprintf( StdErr, "%s ~ %s * %s + %s; mAbsDev=%s) ",
					YVal->name, ad2str( Slope->value, d3str_format, 0),
					(XVal)? XVal->name : "$Counter",
					ad2str( ICept->value, d3str_format, 0),
					ad2str( *result, d3str_format, 0)
				);
			}

			if( Slope->accessHandler ){
				AccessHandler( Slope, "ascanf_rlmFit", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			if( ICept->accessHandler ){
				AccessHandler( ICept, "ascanf_rlmFit", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}
	}
	return( !ascanf_arg_error );
}

static ascanf_Function stats_Function[] = {
	{ "Make-Histogram", ascanf_MakeHistogram, 5, NOT_EOF_OR_RETURN,
		"Make-Histogram[&YVal, useX, &XVal, &histogram[,normalise]]: makes a histogram of the values in YVal according to a\n"
		" classification as specified by XVal. XVal can be\n"
		" * an array with the values of interest, if useX is True\n"
		" * any array of length 1 or useX=False to indicate that all unique, numeric values in YVal should be considered.\n"
		" Upon return, XVal is sorted and initialised if useX=True, and <histogram> contains the scores.\n"
		" If <normalise> is\n"
		" * 1 the sum over <histogram> will be 1\n"
		" * 2 the 'bin' with the maximum score will be normalised to 1\n"
		" Upon success, the function returns the sum over the unnormalised histogram.\n"
	},
	{ "lmFit", ascanf_lmFit, 8, NOT_EOF_OR_RETURN,
		"lmFit[&X|0, &Y, &E|0, &Slope, &ICept [, probS, probI, GoF] ]: do a Chi2 linear fit to the data defined by (X,Y,E):\n"
		" &Y gives the dependent data and the number of samples; if &X is given, it must have not less elements than Y; if\n"
		" X==0, the fit uses the Y index as independent variable. &E can be 0 or define the standard deviations in Y.\n"
		" The fit's slope and intercept are returned in Slope and ICept; the other optional arguments can point to scalars\n"
		" that will hold the uncertainties in the slope and intercept, and the goodness-of-fit, respectively. NaN values are\n"
		" excluded from the fit. The routine returns the fit's Chi2 estimate.\n"
	},
	{ "rlmFit", ascanf_rlmFit, 4, NOT_EOF_OR_RETURN,
		"rlmFit[&X|0, &Y, &Slope, &ICept ]: do a robust linear fit by least absolute deviations to the data defined by (X,Y):\n"
		" &Y gives the dependent data and the number of samples; if &X is given, it must have not less elements than Y; if\n"
		" X==0, the fit uses the Y index as independent variable. The fit's slope and intercept are returned in Slope and ICept;\n"
		" NaN values are excluded from the fit. The routine returns the fit's mean absolute deviation.\n"
	},
};
static int stats_Functions= sizeof(stats_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= stats_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< stats_Functions; i++, af++ ){
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
		XGRAPH_FUNCTION(register_DoubleWithIndex_ptr, "register_DoubleWithIndex");
		XGRAPH_FUNCTION(get_IndexForDouble_ptr, "get_IndexForDouble");
		XGRAPH_FUNCTION(delete_IndexForDouble_ptr, "delete_IndexForDouble");
		XGRAPH_FUNCTION(lm_fit_ptr, "lm_fit");
		XGRAPH_FUNCTION(rlm_fit_ptr, "rlm_fit");
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){

		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( stats_Function, stats_Functions, "stats::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-stats" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" various statistics functions.\n"
	);

	return( DM_Ascanf );
}

void initstats()
{
	wrong_dymod_loaded( "initstats()", "Python", "stats.so" );
}

void R_init_stats()
{
	wrong_dymod_loaded( "R_init_stats()", "R", "stats.so" );
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
	if( initialised ){
	  int r;
		fprintf( SE, "%s::closeDyMod(%d): Closing %s loaded from %s, call %d", __FILE__,
			force, target->name, target->path, ++called
		);
		if( target->loaded4 ){
			fprintf( SE, "; auto-loaded because of \"%s\"", target->loaded4 );
		}
		r= remove_ascanf_functions( stats_Function, stats_Functions, force );
		if( force || r== stats_Functions ){
			for( i= 0; i< stats_Functions; i++ ){
				stats_Function[i].dymod= NULL;
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
				r, stats_Functions
			);
		}
	}
	return(ret);
}
