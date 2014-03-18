#ifndef XG_DYMOD_SUPPORT
#	define XG_DYMOD_SUPPORT
#endif

#include "config.h"
IDENTIFY( "ascanf interface to XGraph; Stats" );

/* 20020527: all ascanf stats stuff in its own module. This may one day become a DyMod! */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <float.h>
#include <math.h>

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

#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
#	define USE_SSE2
#	include <xmmintrin.h>
#	include <emmintrin.h>
#	include "AppleVecLib.h"
#endif

#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
#include "compiled_ascanf.h"
extern char ascanf_errmesg[];

  /* Definitions, externals, etc:	*/

#include "XXseg.h"
extern XSegment *Xsegs, *lowYsegs, *highYsegs;		/* Point space for X */
extern XXSegment *XXsegs;
#define LYsegs	lowYsegs
#define HYsegs	highYsegs
extern XSegment_error *XsegsE;
extern long XsegsSize, XXsegsSize, YsegsSize, XsegsESize;

extern int use_ps_LStyle, ps_LStyle;

#define __DLINE__	(double)__LINE__

extern int maxitems;

extern char *xgraph_NameBuf;
extern int xgraph_NameBufLen;

extern LocalWin *ActiveWin, StubWindow;

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#	include "arrayvops.h"
#elif defined(__SSE2__) || defined(__SSE3__)
#	include "arrayvops.h"
#endif

#ifndef ABS
#	define ABS(x)		(((x)<0)?-(x):(x))
#endif
#ifndef MIN
#	define MIN(a, b)	((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#	define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif
#define SIGN(x)		(((x)<0)?-1:1)
#define TRANX(xval) \
(((double) ((xval) - wi->XOrgX)) * wi->XUnitsPerPixel + wi->UsrOrgX)



extern char *ParsedColourName;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

extern int print_immediate, Animating, AddPoint_discard, Determine_tr_curve_len, Determine_AvSlope;

extern Window thePrintWindow, theSettingsWindow;
extern LocalWin *thePrintWin_Info;

extern Window ascanf_window;

extern double *ascanf_setNumber, *ascanf_Counter, *ascanf_numPoints;

extern char *SS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleStats *a);
extern char *SS_sprint( char *buffer, char *format, char *sep, double min_err, SimpleStats *a);

char **ascanf_SS_names= NULL;
SimpleStats *ascanf_SS= NULL;
char **ascanf_SAS_names= NULL;
SimpleAngleStats *ascanf_SAS= NULL;

NormDist *ascanf_normdists;

int Ascanf_AllocSSMem(int elems)
{ int i;
	if( !(ascanf_normdists= (NormDist*) XGrealloc( ascanf_normdists, elems* sizeof(NormDist))) ){
		fprintf( StdErr, "Ascanf_AllocSSMem(): Can't get memory for ascanf_normdists[%d] array (%s)\n", elems, serror() );
		elems= 0;
		return(0);
	}

	if( !(ascanf_SS= (SimpleStats*) XGrealloc( ascanf_SS, elems* sizeof(SimpleStats))) ){
		fprintf( StdErr, "Ascanf_AllocSSMem(): Can't get memory for ascanf_SS[%d] array (%s)\n", elems, serror() );
		elems= 0;
		return(0);
	}

	if( !(ascanf_SS_names= (char**) XGrealloc( ascanf_SS_names, elems* sizeof(char*))) ){
		fprintf( StdErr, "Ascanf_AllocSSMem(): Can't get memory for ascanf_SS_names[%d] array (%s)\n", elems, serror() );
	}

	if( !(ascanf_SAS= (SimpleAngleStats*) XGrealloc( ascanf_SAS, elems* sizeof(SimpleAngleStats))) ){
		fprintf( StdErr, "Ascanf_AllocSSMem(): Can't get memory for ascanf_SAS[%d] array (%s)\n", elems, serror() );
		elems= 0;
		return(0);
	}

	if( !(ascanf_SAS_names= (char**) XGrealloc( ascanf_SAS_names, elems* sizeof(char*))) ){
		fprintf( StdErr, "Ascanf_AllocSSMem(): Can't get memory for ascanf_SAS_names[%d] array (%s)\n", elems, serror() );
	}
	if( elems> ASCANF_MAX_ARGS ){
		fprintf( StdErr, "Ascanf argumentlist length increased from %d to %d: resetting newly acquired statbins! ",
			ASCANF_MAX_ARGS, elems
		);
		if( !pragma_unlikely(ascanf_verbose) ){
			fputc( '\n', StdErr );
		}
		for( i= ASCANF_MAX_ARGS; i< elems; i++ ){
			ascanf_normdists[i].av= ascanf_normdists[i].stdv=
				ascanf_normdists[i].gset= ascanf_normdists[i].iset= 0;
			ascanf_SS[i].Nvalues= 0;
			ascanf_SS[i].sample= NULL;
			ascanf_SS[i].exact= 0;
			SS_Reset_( ascanf_SS[i] );
			ascanf_SAS[i].Nvalues= 0;
			ascanf_SAS[i].sample= NULL;
			ascanf_SAS[i].exact= 0;
			SAS_Reset_( ascanf_SAS[i] );
			if( ascanf_SS_names ){
				ascanf_SS_names[i]= NULL;
			}
			if( ascanf_SAS_names ){
				ascanf_SAS_names[i]= NULL;
			}
		}
	}
	return(1);
}



int Check_ascanf_SS()
{ int r= 1;
	if( !ascanf_SS && !ascanf_SyntaxCheck ){
		if( !ASCANF_MAX_ARGS ){
			ASCANF_MAX_ARGS= AMAXARGSDEFAULT;
		}
		if( !(ascanf_SS= (SimpleStats*) calloc( ASCANF_MAX_ARGS, sizeof(SimpleStats))) ){
			fprintf( StdErr, "Can't get memory for ascanf_SS[%d] array (%s)\n", ASCANF_MAX_ARGS, serror() );
			ascanf_arg_error= 1;
			r= 0;
		}
	}
	return(r);
}

/* 20020613: lower-level routine for ascanf_SS_set_item() that receives the target bin as an argument. Callable
 \ from _ascanf_simplestats handling code.
 */
int ascanf_SS_set_bin( SimpleStats *ss, int i, char *name, double *args, double *result, int what, ascanf_Function *af_ss )
{ int I= i+1, first= i;
	if( !ascanf_SyntaxCheck ){
		if( i<0 && !af_ss ){
			I= ASCANF_MAX_ARGS;
			first= i= 0;
		}
		if( af_ss ){
			if( af_ss->N> 1 ){
				if( af_ss->last_index>= 0 && af_ss->last_index<= af_ss->N ){
					if( af_ss->last_index== af_ss->N ){
						first= 0, I= af_ss->N;
					}
					else{
						first= af_ss->last_index;
						I= first+1;
					}
				}
				else{
					sprintf( ascanf_errmesg, "(implicit $SS_StatsBin index %d out-of-range [%d,%d])",
						af_ss->last_index, 0, af_ss->N
					);
					ascanf_emsg= ascanf_errmesg;
					ascanf_arg_error= 1;
					if( !ascanf_SyntaxCheck ){
						return(0);
					}
				}
			}
			else{
				first= 0, I= 1;
			}
		}
		if( ascanf_arguments>= 2 ){
			switch( what ){
				case 0:
				default:{
				  ascanf_Function *afd, *afw;
				  double *data, *weight;
				  int *idata= NULL, *iweight= NULL, N= 1, W= 0, j;
				  /* Add value	*/
					if( ascanf_arguments== 2 ){
						args[2]= 1.0;
					}
					if( (afd= parse_ascanf_address( args[1], _ascanf_array, "ascanf_SS_set_item", (ascanf_verbose)?-1:0, NULL )) ){
						N= afd->N;
						if( afd->iarray ){
							idata= afd->iarray;
						}
						else{
							data= afd->array;
						}
					}
					else{
						data= &args[1];
					}
					if( (afw= parse_ascanf_address( args[2], _ascanf_array, "ascanf_SS_set_item", (ascanf_verbose)?-1:0, NULL )) ){
						W= afw->N;
						if( afw->iarray ){
							iweight= afw->iarray;
						}
						else{
							weight= afw->array;
						}
					}
					else{
						weight= &args[2];
					}
					*result= 0;
					for( j= 0; j< N; j++ ){
					  double d, ww= (double)((iweight)? *iweight : *weight), dw;
						if( idata ){
							d= *idata++;
						}
						else{
							d= *data++;
						}
						dw= d*ww;

						if( Check_Ignore_NaN( *SS_Ignore_NaN, dw) ){
							if( af_ss ){
								for( i= first; i< I ; i++ ){
									SS_Add_Data_(ss[i], 1, d, ww );
								}
							}
							else{
								for( i= first; i< I ; i++ ){
									SS_Add_Data_(ascanf_SS[i], 1, d, ww );
								}
							}
							*result+= d;
						}
						if( j< W ){
							if( iweight ){
								iweight++;
							}
							else{
								weight++;
							}
						}
					}
					*result/= j;
					break;
				}
				case 1:{
				  int exact= (int) args[1];
				  /* Set exact field	*/
					if( af_ss ){
						for( i= first; i< I ; i++ ){
							ss[i].exact= exact;
							SS_Reset_( ss[i] );
						}
					}
					else{
						for( ; i< I ; i++ ){
							if( ascanf_SS[i].exact!= exact ){
								ascanf_SS[i].exact= exact;
								SS_Reset_( ascanf_SS[i] );
							}
						}
					}
					*result= exact;
					break;
				}
				case 2:{
				  long size= args[1];
					if( size< 0 ){
						size= 0;
					}
					for( i= first; i< I ; i++ ){
					  SimpleStats *a= (af_ss)? &ss[i] : &ascanf_SS[i];
						if( size ){
							a->exact= 1;
							if( a->Nvalues!= size ){
								a->Nvalues= size;
								if( !(a->sample= (SS_Sample*) realloc( a->sample, a->Nvalues * sizeof(SS_Sample))) ){
									fprintf( StdErr, "SS_SampleSize(%s[%d],%s): can't get memory for %ld samples (%s)\n",
										(name)? name : ad2str( args[0], d3str_format,0), i, ad2str( args[1], d3str_format,0 ), size, serror()
									);
									  /* Don't reset a->exact: who knows if when storing a sample we do get the memory!	*/
									a->Nvalues= 0;
									a->curV= 0;
									xfree( a->sample );
								}
								SS_Reset( a );
							}
						}
						else{
							a->Nvalues= 0;
							xfree( a->sample );
							a->exact= 0;
							SS_Reset( a );
						}
					}
					*result= size;
					break;
				}
				case 3:{
					for( i= first; i< I ; i++ ){
					  SimpleStats *a= (af_ss)? &ss[i] : &ascanf_SS[i];
						if( a->count> 0 ){
							if( ascanf_arguments> 3 ){
								*result= SS_Pop_Data(a, (long) args[1], (long) args[2], ASCANF_TRUE(args[3]) );
							}
							else{
								*result= SS_Pop_Data(a, (long) args[1], (long) args[2], 0 );
							}
						}
					}
					break;
				}
			}
		}
		else{
			if( af_ss ){
				if( af_ss->N> 1 ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " reset \"%s\"[%d-%d]\n", af_ss->name, first, I-1 );
					}
					for( i= first; i< I ; i++ ){
						SS_Reset_(ss[i]);
					}
				}
				else{
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " reset \"%s\"\n", af_ss->name );
					}
					SS_Reset(ss);
				}
			}
			else{
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " reset slot %d through %d\n", i, I-1 );
				}
				for( ; i< I ; i++ ){
					SS_Reset_(ascanf_SS[i]);
				}
			}
			switch( what ){
				default:
					*result= 0.0;
					break;
				case 1:
					*result= (ss)? ss->exact : -1;
					break;
				case 2:
					*result= (ss)? ss->Nvalues : -1;
					break;
			}
		}
	}
	return( 1 );
}

int ascanf_SS_set_item( double *args, double *result, int what )
{ ascanf_Function *ass= NULL;
  SimpleStats *ss= NULL;
  int i= 0, ret;
	if( !Check_ascanf_SS() ){
		return(0);
	}
	if( !args || ascanf_arguments== 0 ){
		ascanf_arg_error= 0;
		ascanf_emsg= NULL;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (slots) " );
		}
		*result= (double) ASCANF_MAX_ARGS;
		return(1);
	}
	ascanf_arg_error= False;
	if( (ass= parse_ascanf_address( args[0], _ascanf_simplestats, "ascanf_SS_set_item", (ascanf_verbose)? -1 : 0, NULL )) ){
		ss= ass->SS;
		i= 0;
	}
	else{
		if( (args[0]= ssfloor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
			i= (int) args[0];
			if( i>= 0 ){
				ss= &ascanf_SS[i];
			}
		}
		else{
			ascanf_emsg= " (invalid SS bin specification) ";
			ascanf_arg_error= True;
		}
	}
	if( ss || i== -1 ){
		  /* ret is not currently used */
		ret= ascanf_SS_set_bin( ss, i, NULL, args, result, what, ass );
	}
	else if( !ascanf_SyntaxCheck ){
		fprintf( StdErr, " (%s: invalid SS bin specification) ", ad2str(args[0], d3str_format,0) );
		ascanf_emsg= "(range error)";
		ascanf_arg_error= 1;
		*result= 0;
	}
	if( ass ){
		ass->value= *result;
	}
	return( 1 );
}

extern char *parse_codes(char *c);

int ascanf_SS_get( double *args, double *result, int what )
{ SimpleStats *ss= NULL;
  ascanf_Function *ass= NULL;
  int i;
	if( !Check_ascanf_SS() ){
		return(0);
	}
	if( !args || ascanf_arguments== 0 ){
		ascanf_arg_error= 0;
		ascanf_emsg= NULL;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (slots) " );
		}
		*result= (double) ASCANF_MAX_ARGS;
		return(1);
	}
	ascanf_arg_error= False;
	if( (ass= parse_ascanf_address( args[0], _ascanf_simplestats, "ascanf_SS_Get", 0, NULL )) ){
		if( ass->N> 1 ){
			if( ass->last_index>= 0 && ass->last_index< ass->N ){
				ss= &ass->SS[ass->last_index];
				i= ass->last_index;
			}
			else{
				sprintf( ascanf_errmesg, "(implicit $SS_StatsBin index %d out-of-range [%d,%d])",
					ass->last_index, 0, ass->N
				);
				ascanf_emsg= ascanf_errmesg;
				ascanf_arg_error= 1;
				ss= NULL;
			}
		}
		else{
			ss= ass->SS;
			i= 0;
		}
	}
	else{
		if( (args[0]= ssfloor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
			i= (int) args[0];
			if( i>= 0 ){
				ss= &ascanf_SS[i];
			}
		}
		else{
			ascanf_emsg= " (invalid SS bin specification) ";
			ascanf_arg_error= True;
		}
	}
	if( ss || i== -1 ){
		if( ascanf_arguments>= 1 ){
			if( i== -1 ){
				ascanf_arg_error= 1;
				*result= 0.0;
			}
			else if( ascanf_SyntaxCheck ){
				*result= 0;
			}
			else{
				if( pragma_unlikely(ascanf_verbose) ){
				  char buf[1024];
					SS_sprint_full( buf, d3str_format, " #xb1 ", 0, ss );
					if( ass ){
						if( ass->N> 1 ){
							fprintf( StdErr, "%s[%d]== %s==", ass->name, i, buf );
						}
						else{
							fprintf( StdErr, "%s== %s==", ass->name, buf );
						}
					}
					else if( ascanf_SS_names && ascanf_SS_names[i] ){
						fprintf( StdErr, "%s== %s==", ascanf_SS_names[i], buf );
					}
					else{
						fprintf( StdErr, "%s==", buf );
					}
				}
				switch( what ){
					case 0:
						*result= SS_Mean( ss);
						break;
					case 1:
						*result= SS_St_Dev( ss);
						break;
					case 2:
						*result= (double) ss->count;
						break;
					case 10:
						*result= (ss->count)? (double) ss->sum : *SS_Empty_Value;
						break;
					case 3:
						*result= (ss->count)? (double) ss->weight_sum : *SS_Empty_Value;
						break;
					case 4:
						*result= (ss->count)? (double) ss->min : *SS_Empty_Value;
						break;
					case 5:
						*result= (ss->count)? (double) ss->pos_min : *SS_Empty_Value;
						break;
					case 6:
						*result= (ss->count)? (double) ss->max : *SS_Empty_Value;
						break;
					case 7:
						*result= SS_St_Err( ss);
						break;
					case 8:
						*result= SS_ADev( ss);
						break;
					case 9:
						*result= SS_Median( ss);
						break;
					case 11:
					case 12:
						if( ascanf_arguments< 2 ){
							ascanf_arg_error= True;
						}
						else if( args[1]< 0 || !(*SS_exact || ss->exact) || !ss->sample ||
								args[1]>= ss->count || args[1]>= ss->curV
						){
							ascanf_emsg= " (sample number out of range) ";
							ascanf_arg_error= True;
						}
						else{
						  int id= (int) args[1];
							*result= (what== 11)? ss->sample[id].value : ss->sample[id].weight;
						}
						break;
					case 13:
						if( ascanf_arguments< 2 ){
							ascanf_arg_error= True;
						}
						else if( args[1]< 0 || args[1]> 1 || !(*SS_exact || ss->exact) || !ss->sample ){
							ascanf_emsg= " (probability out of range [0,1] or bin not in exact mode) ";
							ascanf_arg_error= True;
						}
						else{
							*result= SS_Quantile( ss, args[1] );
						}
						break;
					case 14:
						if( !(*SS_exact || ss->exact) || !ss->sample ){
							ascanf_emsg= " (bin not in exact mode) ";
							ascanf_arg_error= True;
						}
						else{
						  int i;
						  double neg_max= 0;
							if( ss->curV ){
								for( i= 0; i< ss->curV; i++ ){
								  double v= ss->sample[i].value;
									if( v< 0 && (v> neg_max || !neg_max) ){
										neg_max= v;
									}
								}
								*result= neg_max;
							}
							else{
								*result= *SS_Empty_Value;
							}
						}
						break;
					case 15:
						*result= (ss->count)? (double) ss->sum_sqr : *SS_Empty_Value;
						break;
					case 16:
						*result= (ss->count)? (double) ss->sum_cub : *SS_Empty_Value;
						break;
					case 17:
						*result= (ss->count)? SS_Skew(ss) : *SS_Empty_Value;
						break;
					case 18:
						ss->meaned= ss->stdved= False;
						SS_St_Dev( ss);
						*result= sqrt( (ss->stdv * ss->stdv * (ss->count-1.0))/ss->count ) / fabs(ss->mean);
						break;
				}
			}
		}
		else{
			ascanf_arg_error= 1;
			*result= 0.0;
		}
	}
	else if( !ascanf_SyntaxCheck ){
		fprintf( StdErr, " (%s: invalid SS bin specification) ", ad2str(args[0], d3str_format,0) );
		ascanf_emsg= "(range error)";
		ascanf_arg_error= 1;
		*result= 0;
	}
	if( ass ){
		ass->value= *result;
	}
	return( 1 );
}

int ascanf_SS_exact ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_set_item( args, result, 1 ) );
}

int ascanf_SS_SampleSize ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_set_item( args, result, 2 ) );
}

int ascanf_SS_set ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_set_item( args, result, 0 ) );
}

int ascanf_SS_set2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int ret= ascanf_SS_set_item( args, result, 0 );
	if( ascanf_arguments && ret ){
		ret= ascanf_SS_get( args, result, 0 );
	}
	return( ret );
}

int ascanf_SS_pop ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_set_item( args, result, 3 ) );
}

int ascanf_SS_Mean ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 0) );
}

int ascanf_SS_Median ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 9) );
}

int ascanf_SS_Quantile ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 13) );
}

int ascanf_SS_St_Dev ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 1) );
}

int ascanf_SS_ADev ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 8) );
}

int ascanf_SS_St_Err ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 7) );
}

int ascanf_SS_Count ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 2) );
}

int ascanf_SS_Sum ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 10) );
}

int ascanf_SS_SumSqr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 15) );
}

int ascanf_SS_SumCub ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 16) );
}

int ascanf_SS_Skew ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 17) );
}

int ascanf_SS_CV ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 18) );
}

int ascanf_SS_WeightSum ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 3) );
}

int ascanf_SS_min ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 4) );
}

int ascanf_SS_pos_min ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 5) );
}

int ascanf_SS_neg_max ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 14) );
}

int ascanf_SS_max ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 6) );
}

int ascanf_SS_Sample ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 11) );
}

int ascanf_SS_SampleWeight ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SS_get( args, result, 12) );
}

extern double *SS_TValue, *SS_FValue;

/* 20031123:
 \ Verifies whether args0 (and optionally args1) is/are statsbin variable(s), and initialises
 \ as1/s1 (and as2/s2) accordingly. Accounts for statsbin arrays.
 \ Should be used in ascanf_SS_set_bin, too....
 \ Returns 0 upon failure, or otherwise the number of succesfully initialised statsbins:
 \ 1: s1 is valid; 2: s2 is valid TOO.
 \ Two valid arguments are expected when args1, as2 and s2 are all non-NULL.
 */
int Check_SS_StatsBins( double *args0, ascanf_Function **as1, SimpleStats **s1,
	double *args1, ascanf_Function **as2, SimpleStats **s2,
	char *caller
)
{ int r= 0;
	if( args0 && as1 && s1 ){
		if( args1 && as2 && s2 ){
			if( ((*as1)= parse_ascanf_address( *args0, _ascanf_simplestats, caller, (ascanf_verbose)? -1 : 0, NULL )) &&
				(
					((*as2)= parse_ascanf_address( *args1, _ascanf_simplestats, caller, (ascanf_verbose)? -1 : 0, NULL ))
					|| ((*as2)= parse_ascanf_address( *args1, _ascanf_array, caller, (ascanf_verbose)? -1 : 0, NULL ))
				)
			){
			  int idx0= -1, idx1= -1;
				if( (*as2)->type== _ascanf_array ){
					idx0= ASCANF_ARRAY_ELEM( (*as2), 0 );
					idx1= ASCANF_ARRAY_ELEM( (*as2), 1 );
				}
				else{
					if( (*as1)->N> 1 ){
						idx0= (*as1)->last_index;
					}
					if( (*as2)->N> 1 ){
						idx1= (*as2)->last_index;
					}
				}
				if( (*as1)->N> 1 ){
					if( idx0>= 0 && idx0< (*as1)->N ){
						*s1= &((*as1)->SS[idx0]);
						r= 1;
					}
				}
				else{
					*s1= (*as1)->SS;
					r= 1;
				}
				if( (*as2)->type== _ascanf_array ){
					  /* NB!! idx1 indexes into as1, here! */
					if( idx1>= 0 && idx1< (*as1)->N ){
						*s2= &((*as1)->SS[idx1]);
						r+= 1;
					}
				}
				else{
					if( (*as2)->N> 1 ){
						if( idx1>= 0 && idx1< (*as2)->N ){
							*s2= &((*as2)->SS[idx1]);
							r+= 1;
						}
					}
					else{
						*s2= (*as2)->SS;
						r+= 1;
					}
				}
			}
		}
		else{
			if( ((*as1)= parse_ascanf_address( *args0, _ascanf_simplestats, caller, (ascanf_verbose)? -1 : 0, NULL )) ){
				if( (*as1)->N> 1 ){
				  int idx0= (*as1)->last_index;
					if( idx0>= 0 && idx0< (*as1)->N ){
						*s1= &((*as1)->SS[idx0]);
						r= 1;
					}
				}
				else{
					*s1= (*as1)->SS;
					r= 1;
				}
			}
		}
	}
	return(r);
}

int ascanf_SS_FTest ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 2 ){
	  ascanf_Function *as1, *as2;
	  SimpleStats *s1, *s2;
		ascanf_arg_error= False;
		if( Check_SS_StatsBins( &args[0], &as1, &s1, &args[1], &as2, &s2, "ascanf_SS_FTest" )== 2 )
		  /* 20031123: cleanup */
/* 		if( (as1= parse_ascanf_address( args[0], _ascanf_simplestats, "ascanf_SS_FTest", (ascanf_verbose)? -1 : 0, NULL )) &&	*/
/* 			(as2= parse_ascanf_address( args[1], _ascanf_simplestats, "ascanf_SS_FTest", (ascanf_verbose)? -1 : 0, NULL ))	*/
/* 		)	*/
		{
/* 			s1= as1->SS, s2= as2->SS;	*/
			*result= SS_FTest( (s1), (s2), SS_FValue );
		}
		else{
		  int i1= (int) args[0], i2= (int) args[1];
			if( i1>= 0 && i1< ASCANF_MAX_ARGS && i2>= 0 && i2< ASCANF_MAX_ARGS ){
				*result= SS_FTest( (s1= &ascanf_SS[i1]), (s2= &ascanf_SS[i2]), SS_FValue );
			}
			else{
				ascanf_emsg= " (invalid argument(s) to SS_FTest) ";
				ascanf_arg_error= True;
			}
		}
		if( pragma_unlikely(ascanf_verbose) && !ascanf_arg_error ){
			fprintf( StdErr, "(%s [%d]) vs. (%s [%d]): t==%s prob== ",
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s1 ), s1->count,
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s2 ), s2->count,
				ad2str( *SS_FValue, d3str_format, NULL)
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_SS_TTest ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 2 ){
	  ascanf_Function *as1, *as2;
	  SimpleStats *s1, *s2;
		ascanf_arg_error= False;
		if( Check_SS_StatsBins( &args[0], &as1, &s1, &args[1], &as2, &s2, "ascanf_SS_TTest" )== 2 )
		  /* 20031123: cleanup */
/* 		if( (as1= parse_ascanf_address( args[0], _ascanf_simplestats, "ascanf_SS_TTest", (ascanf_verbose)? -1 : 0, NULL )) &&	*/
/* 			(as2= parse_ascanf_address( args[1], _ascanf_simplestats, "ascanf_SS_TTest", (ascanf_verbose)? -1 : 0, NULL ))	*/
/* 		)	*/
		{
/* 			s1= as1->SS, s2= as2->SS;	*/
			*result= SS_TTest( (s1), (s2), SS_TValue );
		}
		else{
		  int i1= (int) args[0], i2= (int) args[1];
			if( i1>= 0 && i1< ASCANF_MAX_ARGS && i2>= 0 && i2< ASCANF_MAX_ARGS ){
				*result= SS_TTest( (s1= &ascanf_SS[i1]), (s2= &ascanf_SS[i2]), SS_TValue );
			}
			else{
				ascanf_emsg= " (invalid argument(s) to SS_TTest) ";
				ascanf_arg_error= True;
			}
		}
		if( pragma_unlikely(ascanf_verbose) && !ascanf_arg_error ){
			fprintf( StdErr, "(%s [%d]) vs. (%s [%d]): t==%s prob== ",
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s1 ), s1->count,
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s2 ), s2->count,
				ad2str( *SS_TValue, d3str_format, NULL)
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_SS_TTest_uneq ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 2 ){
	  ascanf_Function *as1, *as2;
	  SimpleStats *s1, *s2;
		ascanf_arg_error= False;
		if( Check_SS_StatsBins( &args[0], &as1, &s1, &args[1], &as2, &s2, "ascanf_SS_TTest_uneq" )== 2 )
		  /* 20031123: cleanup */
/* 		if( (as1= parse_ascanf_address( args[0], _ascanf_simplestats, "ascanf_SS_TTest_uneq", (ascanf_verbose)? -1 : 0, NULL )) &&	*/
/* 			(as2= parse_ascanf_address( args[1], _ascanf_simplestats, "ascanf_SS_TTest_uneq", (ascanf_verbose)? -1 : 0, NULL ))	*/
/* 		)	*/
		{
/* 			s1= as1->SS, s2= as2->SS;	*/
			*result= SS_TTest_uneq( (s1), (s2), SS_TValue );
		}
		else{
		  int i1= (int) args[0], i2= (int) args[1];
			if( i1>= 0 && i1< ASCANF_MAX_ARGS && i2>= 0 && i2< ASCANF_MAX_ARGS ){
				*result= SS_TTest_uneq( (s1= &ascanf_SS[i1]), (s2= &ascanf_SS[i2]), SS_TValue );
			}
			else{
				ascanf_emsg= " (invalid argument(s) to SS_TTest_uneq) ";
				ascanf_arg_error= True;
			}
		}
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(%s [%d]) vs. (%s [%d]): t==%s prob== ",
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s1 ), s1->count,
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s2 ), s2->count,
				ad2str( *SS_TValue, d3str_format, NULL)
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_SS_TTest_paired ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 2 ){
	  ascanf_Function *as1, *as2;
	  SimpleStats *s1, *s2;
		ascanf_arg_error= False;
		if( Check_SS_StatsBins( &args[0], &as1, &s1, &args[1], &as2, &s2, "ascanf_SS_TTest_paired" )== 2 )
		  /* 20031123: cleanup */
/* 		if( (as1= parse_ascanf_address( args[0], _ascanf_simplestats, "ascanf_SS_TTest_paired", (ascanf_verbose)? -1 : 0, NULL )) &&	*/
/* 			(as2= parse_ascanf_address( args[1], _ascanf_simplestats, "ascanf_SS_TTest_paired", (ascanf_verbose)? -1 : 0, NULL ))	*/
/* 		)	*/
		{
/* 			s1= as1->SS, s2= as2->SS;	*/
			*result= SS_TTest_paired( (s1), (s2), SS_TValue );
		}
		else{
		  int i1= (int) args[0], i2= (int) args[1];
			if( i1>= 0 && i1< ASCANF_MAX_ARGS && i2>= 0 && i2< ASCANF_MAX_ARGS ){
				*result= SS_TTest_paired( (s1= &ascanf_SS[i1]), (s2= &ascanf_SS[i2]), SS_TValue );
			}
			else{
				ascanf_emsg= " (invalid argument(s) to SS_TTest_paired) ";
				ascanf_arg_error= True;
			}
		}
		if( pragma_unlikely(ascanf_verbose) && !ascanf_arg_error ){
			fprintf( StdErr, "(%s [%d]) vs. (%s [%d]): t==%s prob== ",
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s1 ), s1->count,
				SS_sprint( NULL, d3str_format, " #xb1 ", 0, s2 ), s2->count,
				ad2str( *SS_TValue, d3str_format, NULL)
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_SS_TTest_correct ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 3 ){
	  ascanf_Function *as1, *as2;
	  SimpleStats *s1, *s2;
	  double prob;
	  char *op, *type;
		ascanf_arg_error= False;
		if( Check_SS_StatsBins( &args[0], &as1, &s1, &args[1], &as2, &s2, "ascanf_SS_TTest_correct" )== 2 )
		  /* 20031123: cleanup */
/* 		if( (as1= parse_ascanf_address( args[0], _ascanf_simplestats, "ascanf_SS_TTest_correct", (ascanf_verbose)? -1 : 0, NULL )) &&	*/
/* 			(as2= parse_ascanf_address( args[1], _ascanf_simplestats, "ascanf_SS_TTest_correct", (ascanf_verbose)? -1 : 0, NULL ))	*/
/* 		)	*/
		{
/* 			s1= as1->SS, s2= as2->SS;	*/
		}
		else{
		  int i1= (int) args[0], i2= (int) args[1];
			if( i1>= 0 && i1< ASCANF_MAX_ARGS && i2>= 0 && i2< ASCANF_MAX_ARGS ){
				s1= &ascanf_SS[i1], s2= &ascanf_SS[i2];
			}
			else{
				ascanf_emsg= " (invalid argument(s) to SS_TTest_correct) ";
				ascanf_arg_error= True;
			}
		}
		if( !ascanf_arg_error ){
			prob= SS_FTest( s1, s2, SS_FValue );
			if( prob< args[2] ){
				*result= SS_TTest_uneq( s1, s2, SS_TValue );
				type= "unequal";
				op= "<";
			}
			else{
				*result= SS_TTest( s1, s2, SS_TValue );
				type= "equal";
				op= ">=";
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "(%s [%d]) vs. ( %s [%d]): f==%s, f-prob==%s%s%s => t(%s)==%s prob(%s)== ",
					SS_sprint( NULL, d3str_format, " #xb1 ", 0, s1 ), s1->count,
					SS_sprint( NULL, d3str_format, " #xb1 ", 0, s2 ), s2->count,
					ad2str( *SS_FValue, d3str_format, NULL), ad2str( prob, d3str_format, NULL), op,
					ad2str( args[2], d3str_format, NULL),
					type, ad2str( *SS_TValue, d3str_format, NULL), type
				);
			}
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_FTest ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 4 ){
		*result= FTest( args[0]* args[0], (int) args[1], args[2]* args[2], (int) args[3], SS_FValue );
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (%s [%d]) vs. (%s [%d]): f==%s prob== ",
				ad2str( args[0], d3str_format, NULL), (int) args[1],
				ad2str( args[2], d3str_format, NULL), (int) args[3],
				ad2str( *SS_FValue, d3str_format, NULL)
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_TTest ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 6 ){
		*result= TTest( args[0], args[1]* args[1], (int) args[2], args[3], args[4]* args[4], (int) args[5], SS_TValue );
		if( pragma_unlikely(ascanf_verbose) ){
		  char pc[]= "(%s #xb1 %s [%d]) vs. (%s #xb1 %s [%d]): t==%s prob== ";
			fprintf( StdErr, parse_codes(pc),
				ad2str( args[0], d3str_format, NULL), ad2str( args[1], d3str_format, NULL), (int) args[2],
				ad2str( args[3], d3str_format, NULL), ad2str( args[4], d3str_format, NULL), (int) args[5],
				ad2str( *SS_TValue, d3str_format, NULL)
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_TTest_uneq ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 6 ){
		*result= TTest_uneq( args[0], args[1]* args[1], (int) args[2], args[3], args[4]* args[4], (int) args[5], SS_TValue );
		if( pragma_unlikely(ascanf_verbose) ){
		  char pc[]= "(%s #xb1 %s [%d]) vs. (%s #xb1 %s [%d]): t==%s prob== ";
			fprintf( StdErr, parse_codes(pc),
				ad2str( args[0], d3str_format, NULL), ad2str( args[1], d3str_format, NULL), (int) args[2],
				ad2str( args[3], d3str_format, NULL), ad2str( args[4], d3str_format, NULL), (int) args[5],
				ad2str( *SS_TValue, d3str_format, NULL)
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int ascanf_TTest_correct ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *A, *B;
  double mean1, stdv1, mean2, stdv2, Fprob;
  int n1, n2;

	ascanf_arg_error= False;
	if( (A= parse_ascanf_address( args[0], 0, "ascanf_TTest_correct", (ascanf_verbose)? -1 : 0, NULL )) &&
		(B= parse_ascanf_address( args[1], 0, "ascanf_TTest_correct", (ascanf_verbose)? -1 : 0, NULL )) &&
		ascanf_arguments>= 3
	){
	  int i;
	  SimpleStats ss;
		ss.sample= NULL;
		ss.exact= (*SS_exact)? True : False;
		SS_Reset_(ss);
		for(i= 0; i< A->N; i++ ){
		  double a= ((A->iarray)? A->iarray[i] : A->array[i]);
			if( Check_Ignore_NaN( *SS_Ignore_NaN, a) ){
				SS_Add_Data_(ss, 1, a, 1.0 );
			}
		}
		mean1= SS_Mean_(ss);
		stdv1= SS_St_Dev_(ss);
		n1= A->N;
		SS_Reset_(ss);
		for(i= 0; i< B->N; i++ ){
		  double b= ((B->iarray)? B->iarray[i] : B->array[i]);
			if( Check_Ignore_NaN( *SS_Ignore_NaN, b) ){
				SS_Add_Data_(ss, 1, b, 1.0 );
			}
		}
		mean2= SS_Mean_(ss);
		stdv2= SS_St_Dev_(ss);
		n2= B->N;
		xfree( ss.sample );
		Fprob= args[2];
	}
	else if( ascanf_arguments>= 7 ){
		mean1= args[0], stdv1= args[1], n1= (int) args[2];
		mean2= args[3], stdv2= args[4], n2= (int) args[5];
		Fprob= args[6];
	}
	else{
		ascanf_arg_error= True;
	}
	if( !ascanf_arg_error ){
	  double prob;
	  char *op, *type;
		prob= FTest( stdv1* stdv1, n1, stdv2* stdv2, n2, SS_FValue );
		if( prob< Fprob ){
			*result= TTest_uneq( mean1, stdv1* stdv1, n1, mean2, stdv2* stdv2, n2, SS_TValue );
			type= "unequal";
			op= "<";
		}
		else{
			*result= TTest( mean1, stdv1* stdv1, n1, mean2, stdv2* stdv2, n2, SS_TValue );
			type= "equal";
			op= ">=";
		}
		if( pragma_unlikely(ascanf_verbose) ){
		  char pc[]= "(%s #xb1 %s [%d]) vs. (%s #xb1 %s [%d]): f==%s, f-prob==%s%s%s => t(%s)==%s prob(%s)== ";
			fprintf( StdErr, parse_codes(pc),
				ad2str( mean1, d3str_format, NULL), ad2str( stdv1, d3str_format, NULL), n1,
				ad2str( mean2, d3str_format, NULL), ad2str( stdv2, d3str_format, NULL), n2,
				ad2str( *SS_FValue, d3str_format, NULL), ad2str( prob, d3str_format, NULL), op,
				ad2str( Fprob, d3str_format, NULL),
				type, ad2str( *SS_TValue, d3str_format, NULL), type
			);
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 2;
		return(1);
	}
}

int _ascanf_SS_StatsBin( ascanf_Function *af, int exact )
{ int r= 1, N= 1, i;
	if( af->type!= _ascanf_simplestats && af->SS ){
		xfree( af->SS->sample );
		xfree( af->SS );
	}
	if( af->SAS ){
		xfree( af->SAS->sample );
		xfree( af->SAS );
	}
	if( af->type== _ascanf_array ){
		xfree( af->iarray );
		xfree( af->array );
		N= af->N= abs(af->N);
	}
	else{
		af->N= 1;
	}
	if( !af->SS ){
		if( (af->SS= (SimpleStats*) calloc( N, sizeof(SimpleStats))) ){
			for( i= 0; i< N; i++ ){
				SS_Reset_( af->SS[i] );
			}
		}
		else{
			ascanf_emsg= " (can't (re)allocate $SS statsbin memory!)== ";
			ascanf_arg_error= 1;
			r= 0;
		}
	}
	if( af->SS ){
		if( af->type!= _ascanf_simplestats ){
			af->type= _ascanf_simplestats;
			take_ascanf_address(af);
		}
		for( i= 0; i< N; i++ ){
			af->SS[i].exact= exact;
		}
	}
	return(r);
}

int ascanf_SS_StatsBin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 2 ){
	  ascanf_Function *af;
	  int exact;
	  int n= 0, r= 1;
		ascanf_arg_error= False;
		if( !(af= parse_ascanf_address( args[0], 0, "ascanf_SS_StatsBin", ascanf_verbose, NULL )) ||
			(af->type!= _ascanf_variable && af->type!= _ascanf_array &&
				af->type!= _ascanf_simplestats && af->type!= _ascanf_simpleanglestats ) ||
			(af->name[0]== '$' && !af->dollar_variable)
		){
			ascanf_emsg= " (SS_StatsBin[]: first argument must be a pointer to a variable!) ";
			ascanf_arg_error= True;
		}
		exact= ASCANF_TRUE(args[1]);
		if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
			if( (r= _ascanf_SS_StatsBin( af, exact )) ){
			  int i;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (new) " );
				}
				for( i= 2; i< ascanf_arguments; i++ ){
					if( Check_Ignore_NaN( *SS_Ignore_NaN, args[i]) ){
						SS_Add_Data( af->SS, 1, args[i], 1.0 );
						n+= 1;
					}
				}
			}
			else{
				n= -1;
			}
		}
		*result= n;
		return(r);
	}
	else{
		ascanf_arg_error= 1;
		*result= -1;
		return(0);
	}
}


extern char *SAS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a);

int Check_ascanf_SAS()
{ int r= 1;
	if( !ascanf_SAS && !ascanf_SyntaxCheck ){
		if( !ASCANF_MAX_ARGS ){
			ASCANF_MAX_ARGS= AMAXARGSDEFAULT;
		}
		if( !(ascanf_SAS= (SimpleAngleStats*) calloc( ASCANF_MAX_ARGS, sizeof(SimpleAngleStats))) ){
			fprintf( StdErr, "Can't get memory for ascanf_SAS[%d] array (%s)\n", ASCANF_MAX_ARGS, serror() );
			ascanf_arg_error= 1;
			r= 0;
		}
	}
	return(r);
}

int ascanf_SAS_set_bin( SimpleAngleStats *sas, int i, char *name, double *args, double *result, int what, ascanf_Function *af_sas )
{ int I= i+1;

	if( !ascanf_SyntaxCheck ){
		if( ascanf_arguments>= 2 ){
			if( i== -1 && !af_sas ){
				I= ASCANF_MAX_ARGS;
				i= 0;
			}
			switch( what ){
				case 0:
#ifdef NO_SAS_ARRAY_ADD
				default:{
				  double dw;
					if( ascanf_arguments== 2 ){
						args[2]= 1.0;
					}
					dw= args[1] * args[2];
					if( Check_Ignore_NaN( *SAS_Ignore_NaN, dw) ){
						if( af_sas ){
							if( ascanf_arguments> 3 && args[3] ){
								sas->Gonio_Base= args[3];
								sas->Units_per_Radian= sas->Gonio_Base/ (2*M_PI);
							}
							else if( !sas->Gonio_Base ){
								sas->Gonio_Base= M_2PI;
								sas->Units_per_Radian= sas->Gonio_Base/ (2*M_PI);
							}
							if( ascanf_arguments> 4 ){
								sas->Gonio_Offset= args[4];
							}
							SAS_Add_Data(sas, 1, args[1], args[2],
								(ascanf_arguments> 5)? ASCANF_TRUE(args[5]) : ASCANF_TRUE(*SAS_converts_angle) );
						}
						else{
							for( ; i< I ; i++ ){
								if( ascanf_arguments> 3 && args[3] ){
									ascanf_SAS[i].Gonio_Base= args[3];
									ascanf_SAS[i].Units_per_Radian= ascanf_SAS[i].Gonio_Base/ (2*M_PI);
								}
								else if( !ascanf_SAS[i].Gonio_Base ){
									ascanf_SAS[i].Gonio_Base= M_2PI;
									ascanf_SAS[i].Units_per_Radian= ascanf_SAS[i].Gonio_Base/ (2*M_PI);
								}
								if( ascanf_arguments> 4 ){
									ascanf_SAS[i].Gonio_Offset= args[4];
								}
								SAS_Add_Data_(ascanf_SAS[i], 1, args[1], args[2],
									(int)((ascanf_arguments> 5)? args[5] : *SAS_converts_angle)
								);
							}
						}
					}
					*result= args[1];
					break;
				}
#else
				default:{
				  ascanf_Function *afd, *afw;
				  double *data, *weight;
				  int *idata= NULL, *iweight= NULL, N= 1, W= 0, j, first= i;
				  int convert= (int)((ascanf_arguments> 5)? ASCANF_TRUE(args[5]) : ASCANF_TRUE(*SAS_converts_angle));
				  /* Add value	*/
					if( ascanf_arguments== 2 ){
						args[2]= 1.0;
					}
					if( (afd= parse_ascanf_address( args[1], _ascanf_array, "ascanf_SAS_set_item", (ascanf_verbose)?-1:0, NULL )) ){
						N= afd->N;
						if( afd->iarray ){
							idata= afd->iarray;
						}
						else{
							data= afd->array;
						}
					}
					else{
						data= &args[1];
					}
					if( (afw= parse_ascanf_address( args[2], _ascanf_array, "ascanf_SAS_set_item", (ascanf_verbose)?-1:0, NULL )) ){
						W= afw->N;
						if( afw->iarray ){
							iweight= afw->iarray;
						}
						else{
							weight= afw->array;
						}
					}
					else{
						weight= &args[2];
					}
					*result= 0;
					for( j= 0; j< N; j++ ){
					  double d, ww= ((iweight)? (double)*iweight : *weight), dw;
						if( idata ){
							d= *idata++;
						}
						else{
							d= *data++;
						}
						dw= d*ww;

						if( Check_Ignore_NaN( *SAS_Ignore_NaN, dw ) ){
							if( af_sas ){
								if( ascanf_arguments> 3 && args[3] ){
									sas->Gonio_Base= args[3];
									sas->Units_per_Radian= sas->Gonio_Base/ (2*M_PI);
								}
								else if( !sas->Gonio_Base ){
									sas->Gonio_Base= M_2PI;
									sas->Units_per_Radian= sas->Gonio_Base/ (2*M_PI);
								}
								if( ascanf_arguments> 4 ){
									sas->Gonio_Offset= args[4];
								}
								SAS_Add_Data(sas, 1L, d, ww, convert );
							}
							else{
								for( i= first; i< I ; i++ ){
									if( ascanf_arguments> 3 && args[3] ){
										ascanf_SAS[i].Gonio_Base= args[3];
										ascanf_SAS[i].Units_per_Radian= ascanf_SAS[i].Gonio_Base/ (2*M_PI);
									}
									else if( !ascanf_SAS[i].Gonio_Base ){
										ascanf_SAS[i].Gonio_Base= M_2PI;
										ascanf_SAS[i].Units_per_Radian= ascanf_SAS[i].Gonio_Base/ (2*M_PI);
									}
									if( ascanf_arguments> 4 ){
										ascanf_SAS[i].Gonio_Offset= args[4];
									}
									SAS_Add_Data_(ascanf_SAS[i], 1L, d, ww, convert );
								}
							}
							*result+= d;
						}
						if( j< W ){
							if( iweight ){
								iweight++;
							}
							else{
								weight++;
							}
						}
					}
					*result/= j;
					break;
				}
#endif
				case 1:{
				  int exact= (int) args[1];
				  /* Set exact field	*/
					if( af_sas ){
						if( sas->exact!= exact ){
							sas->exact= exact;
							SAS_Reset( sas );
						}
					}
					else{
						for( ; i< I ; i++ ){
							if( ascanf_SAS[i].exact!= exact ){
								ascanf_SAS[i].exact= exact;
								SAS_Reset_( ascanf_SAS[i] );
							}
						}
					}
					*result= exact;
					break;
				}
				case 2:{
				  int size= (int) args[1];
					if( size< 0 ){
						size= 0;
					}
					for( ; i< I ; i++ ){
					  SimpleAngleStats *a= (af_sas)? sas : &ascanf_SAS[i];
						if( size ){
							a->exact= 1;
							if( a->Nvalues!= size ){
								a->Nvalues= size;
								if( !(a->sample= (SS_Sample*) realloc( a->sample, a->Nvalues * sizeof(SS_Sample))) ){
									fprintf( StdErr, "SAS_SampleSize(%s[%d],%s): can't get memory for %ld samples (%s)\n",
										(name)? name : ad2str(args[0], d3str_format,0), i, ad2str(args[1], d3str_format,0), size, serror()
									);
									  /* Don't reset a->exact: who knows if when storing a sample we do get the memory!	*/
									a->Nvalues= 0;
									a->curV= 0;
									xfree( a->sample );
								}
								SAS_Reset( a );
							}
						}
						else{
							a->Nvalues= 0;
							xfree( a->sample );
							a->exact= 0;
							SAS_Reset( a );
						}
					}
					*result= size;
					break;
				}
				case 3:{
					if( af_sas ){
						sas->Gonio_Base= (args[1])? args[1] : M_2PI;
						sas->Units_per_Radian= sas->Gonio_Base/ (2*M_PI);
					}
					else{
						for( ; i< I ; i++ ){
							ascanf_SAS[i].Gonio_Base= args[1];
							ascanf_SAS[i].Units_per_Radian= ascanf_SAS[i].Gonio_Base/ (2*M_PI);
						}
					}
					*result= (args[1])? args[1] : M_2PI;
					break;
				}
				case 4:{
					if( af_sas ){
						sas->Gonio_Offset= args[1];
					}
					else{
						for( ; i< I ; i++ ){
							ascanf_SAS[i].Gonio_Offset= args[1];
						}
					}
					*result= args[1];
					break;
				}
			}
		}
		else{
			if( af_sas ){
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " reset \"%s\"\n", af_sas->name );
				}
				SAS_Reset(sas);
			}
			else{
				if( i== -1 ){
					I= ASCANF_MAX_ARGS;
					i= 0;
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " reset slot %d through %d\n", i, I-1 );
				}
				for( ; i< I ; i++ ){
					SAS_Reset_(ascanf_SAS[i]);
				}
			}
			switch( what ){
				default:
					*result= 0.0;
					break;
				case 1:
					*result= (af_sas)? sas->exact : -1;
					break;
				case 2:
					*result= (af_sas)? sas->Nvalues : -1;
					break;
				case 3:
					*result= (af_sas)? sas->Gonio_Base : -1;
					break;
				case 4:
					*result= (af_sas)? sas->Gonio_Offset : -1;
					break;
			}
		}
	}
	return( 1 );
}

int ascanf_SAS_set_item( double *args, double *result, int what )
{ ascanf_Function *asas= NULL;
  SimpleAngleStats *sas= NULL;
  int i= 0, ret= 0;

	if( !Check_ascanf_SAS() ){
		return(0);
	}
	if( !args || ascanf_arguments== 0 ){
		ascanf_arg_error= 0;
		ascanf_emsg= NULL;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (slots) " );
		}
		*result= (double) ASCANF_MAX_ARGS;
		return(1);
	}
	ascanf_arg_error= False;
	if( (asas= parse_ascanf_address( args[0], _ascanf_simpleanglestats, "ascanf_SAS_set_item", (ascanf_verbose)? -1 : 0, NULL )) ){
		sas= asas->SAS;
		i= 0;
	}
	else{
		if( (args[0]= ssfloor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
			i= (int) args[0];
			if( i>= 0 ){
				sas= &ascanf_SAS[i];
			}
		}
		else{
			ascanf_emsg= " (invalid SAS bin specification) ";
			ascanf_arg_error= True;
		}
	}
	if( sas || i== -1 ){
		ret= ascanf_SAS_set_bin( sas, i, NULL, args, result, what, asas );
	}
	else if( !ascanf_SyntaxCheck ){
		fprintf( StdErr, " (%s: invalid SAS bin specification) ", ad2str(args[0], d3str_format,0) );
		ascanf_emsg= "(range error)";
		ascanf_arg_error= 1;
		*result= 0;
		if( asas ){
			asas->value= *result;
		}
	}
	return( 1 );
}

int ascanf_SAS_exact ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_set_item( args, result, 1 ) );
}

int ascanf_SAS_SampleSize ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_set_item( args, result, 2 ) );
}

int ascanf_SAS_GonioBase ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_set_item( args, result, 3 ) );
}

int ascanf_SAS_GonioOffset ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_set_item( args, result, 4 ) );
}

int ascanf_SAS_set ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_set_item( args, result, 0 ) );
}

int ascanf_SAS_set2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int ret= ascanf_SAS_set_item( args, result, 0 );
	if( ascanf_arguments && ret ){
		ret= ascanf_SAS_get( args, result, 0 );
	}
	return( ret );
}

int ascanf_SAS_get( double *args, double *result, int what )
{ SimpleAngleStats *sas= NULL;
  ascanf_Function *asas= NULL;
  int i;
	if( !Check_ascanf_SAS() ){
		return(0);
	}
	if( !args || ascanf_arguments== 0 ){
		ascanf_arg_error= 0;
		ascanf_emsg= NULL;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (slots) " );
		}
		*result= (double) ASCANF_MAX_ARGS;
		return(1);
	}
	if( (asas= parse_ascanf_address( args[0], _ascanf_simpleanglestats, "ascanf_SAS_Get", 0, NULL )) ){
		sas= asas->SAS;
		i= 0;
	}
	else{
		if( (args[0]= ssfloor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
			i= (int) args[0];
			if( i>= 0 ){
				sas= &ascanf_SAS[i];
			}
		}
		else{
			ascanf_emsg= " (invalid SAS bin specification) ";
			ascanf_arg_error= True;
		}
	}
	if( sas || i== -1 ){
		if( ascanf_arguments>= 1 ){
			if( i== -1 ){
				ascanf_arg_error= 1;
				*result= 0.0;
			}
			else if( ascanf_SyntaxCheck ){
				*result= 0.0;
			}
			else{
			  extern double *SAS_converts_result;
			  double Scr= *SAS_converts_result;
			  int count= sas->pos_count + sas->neg_count;
				if( ascanf_arguments>= 2 ){
					*SAS_converts_result= args[1];
				}
				if( pragma_unlikely(ascanf_verbose) ){
				  char buf[1024];
					SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, sas );
					if( asas ){
						fprintf( StdErr, "%s== %s==", asas->name, buf );
					}
					else if( ascanf_SAS_names && ascanf_SAS_names[i] ){
						fprintf( StdErr, "%s== %s==", ascanf_SAS_names[i], buf );
					}
					else{
						fprintf( StdErr, "%s==", buf );
					}
				}
				switch( what ){
					case 0:
						*result= SAS_Mean( sas);
						break;
					case 1:
						*result= SAS_St_Dev( sas);
						break;
					case 2:
						*result= (count)? (double) sas->pos_count + sas->neg_count : *SS_Empty_Value;
						break;
					case 4:
						*result= (count)? (double) sas->min : *SS_Empty_Value;
						break;
					case 6:
						*result= (count)? (double) sas->max : *SS_Empty_Value;
						break;
					case 7:
					case 8:
						if( ascanf_arguments< 2 ){
							ascanf_arg_error= True;
						}
						else if( args[1]< 0 || !(*SAS_exact || sas->exact) || !sas->sample ||
								args[1]>= (sas->pos_count + sas->neg_count) || args[1]>= sas->curV
						){
							ascanf_emsg= " (sample number out of range) ";
							ascanf_arg_error= True;
						}
						else{
						  int id= (int) args[1];
							*result= (what== 7)? sas->sample[id].value : sas->sample[id].weight;
						}
						break;
					case 9:{
					  double count= sas->pos_count + sas->neg_count;
						sas->meaned= sas->stdved= False;
						SAS_St_Dev(sas);
						*result= sqrt( (sas->stdv * sas->stdv * (count-1.0))/count ) / fabs(sas->mean);
						break;
					}
				}
				*SAS_converts_result= Scr;
			}
		}
		else{
			ascanf_arg_error= 1;
			*result= 0.0;
		}
	}
	else if( !ascanf_SyntaxCheck ){
		fprintf( StdErr, " (%s: invalid SAS bin specification) ", ad2str(args[0], d3str_format,0) );
		ascanf_emsg= "(range error)";
		ascanf_arg_error= 1;
		*result= 0;
	}
	if( asas ){
		asas->value= *result;
	}
	return( 1 );
}

int ascanf_SAS_Mean ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 0) );
}

int ascanf_SAS_St_Dev ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 1) );
}

int ascanf_SAS_Count ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 2) );
}

int ascanf_SAS_min ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 4) );
}

int ascanf_SAS_max ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 6) );
}

int ascanf_SAS_Sample ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 7) );
}

int ascanf_SAS_SampleWeight ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 8) );
}

int ascanf_SAS_CV ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_SAS_get( args, result, 9) );
}

int _ascanf_SAS_StatsBin( ascanf_Function *af, int exact )
{ int r= 1;
	if( af->SS ){
		xfree( af->SS->sample );
		xfree( af->SS );
	}
	if( af->type!= _ascanf_simpleanglestats && af->SAS ){
		xfree( af->SAS->sample );
		xfree( af->SAS );
	}
	if( !af->SAS ){
		if( (af->SAS= (SimpleAngleStats*) calloc( 1, sizeof(SimpleAngleStats))) ){
			SAS_Reset( af->SAS );
		}
		else{
			ascanf_emsg= " (can't (re)allocate $SAS statsbin memory!)== ";
			ascanf_arg_error= 1;
			r= 0;
		}
	}
	if( af->SAS ){
		if( af->type!= _ascanf_simpleanglestats ){
			af->type= _ascanf_simpleanglestats;
			take_ascanf_address(af);
		}
		af->SAS->exact= exact;
	}
	return(r);
}

int ascanf_SAS_StatsBin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 2 ){
	  ascanf_Function *af;
	  int exact;
	  int n= 0, r= 1;
		ascanf_arg_error= False;
		if( !(af= parse_ascanf_address( args[0], 0, "ascanf_SAS_StatsBin", ascanf_verbose, NULL )) ||
			(af->type!= _ascanf_variable && af->type!= _ascanf_simpleanglestats && af->type!= _ascanf_simplestats)
		){
			ascanf_emsg= " (SAS_StatsBin[]: first argument must be a pointer to a variable!) ";
			ascanf_arg_error= True;
		}
		exact= ASCANF_TRUE(args[1]);
		if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
			if( (r= _ascanf_SAS_StatsBin(af, exact)) ){
			  int i;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (new) " );
				}
				if( ascanf_arguments> 2 ){
					af->SAS->Gonio_Base= args[2];
					af->SAS->Units_per_Radian= af->SAS->Gonio_Base/ (2*M_PI);
				}
				if( ascanf_arguments> 3 ){
					af->SAS->Gonio_Offset= args[3];
				}
				for( i= 5; i< ascanf_arguments; i++ ){
					if( Check_Ignore_NaN( *SAS_Ignore_NaN, args[i]) ){
						SAS_Add_Data( af->SAS, 1, args[i], 1.0, ASCANF_TRUE(args[4]) );
						n+= 1;
					}
				}
			}
			else{
				n= -1;
			}
		}
		*result= n;
		return(r);
	}
	else{
		ascanf_arg_error= 1;
		*result= -1;
		return(0);
	}
}

/* Routines associated with Chi-Square calculations: */

extern unsigned long MAXIT;

/* Return the incomplete gamma function P(a,x) determined by its series representation. When
 \ specified, return \Gamma(a) in gammaA
 */
#ifdef __clang__
static
#endif
#ifdef __GNUC__
inline
#endif
double gamma_series( double a, double x, double *gammaA )
{ int i;
  double sum, del, ap, gln= lgamma(a);
	if( gammaA ){
		*gammaA= gln;
	}
	if( x> 0.0 ){
		ap= a;
		del= sum= 1.0/ a;
		for( i= 0; i< MAXIT; i++ ){
			ap+= 1;
			del*= x/ap;
			sum+= del;
			if( fabs(del)< fabs(sum)* DBL_EPSILON ){
				return( sum* exp( -x+ a* log(x)- gln ) );
			}
		}
	}
	else{
		if( x< 0 ){
			fprintf( StdErr, " (gamma_series() called with negative x==%g argument) ", x );
		}
	}
	return(0);
}

/* Return the incomplete gamma function P(a,x) determined by its continued-fraction representation. When
 \ specified, return \Gamma(a) in gammaA
 */
#ifdef __clang__
static
#endif
#ifdef __GNUC__
inline
#endif
double gamma_contfrac( double a, double x, double *gammaA )
{ int i;
  double an, b, c, d, del, h, gln;

	gln= lgamma(a);
	if( gammaA ){
		*gammaA= gln;
	}
	b= x+ 1- a;
	c= 1/ DBL_MIN;
	h= d= 1/b;
	for( i= 1; i<= MAXIT; i++ ){
		an= -i* (i-a);
		b+= 2;
		d= an* d+ b;
		if( d> -DBL_MIN && d< DBL_MIN ){
			d= DBL_MIN;
		}
		c= b+ an/c;
		if( c> -DBL_MIN && c< DBL_MIN ){
			c= DBL_MIN;
		}
		d= 1/d;
		del= d*c;
		h*= del;
		if( fabs(del-1)< DBL_EPSILON ){
			break;
		}
	}
	if( i> MAXIT ){
		fprintf( StdErr, " (argument a==%g too large in gamma_contfrac() for current MAXIT==%lu) ", a, MAXIT );
	}
	return( h* exp( -x + a* log(x) - gln) );
}

/* Return the incomplete gamma function P(a,x): */
#ifdef __clang__
static
#endif
#ifdef __GNUC__
inline
#endif
double incomplete_gammaFunction( double a, double x )
{
	if( x< 0 || a<= 0 ){
		fprintf( StdErr, " (incomplete_gammaFunction(%g,%g): invalid argument(s)) ", a, x );
	}
	if( x< (a+1) ){
		return( gamma_series( a, x, NULL ) );
	}
	else{
		return( 1- gamma_contfrac( a, x, NULL ) );
	}
}

/* The complement 1-P(a,x): */
double incomplete_gammaFunction_complement( double a, double x );
#if defined(__GNUC__)
inline
#endif
double incomplete_gammaFunction_complement( double a, double x )
{
	return( 1- incomplete_gammaFunction(a,x) );
}


int ascanf_IncomplGamma ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments>= 2 ){
		*result= incomplete_gammaFunction( args[0], args[1] );
		return(1);
	}
	else{
		ascanf_arg_error= True;
		return(0);
	}
}

double ChiSquare( double *observed, double *expected, int N, int constraints, int *dF, double *prob )
{ int df, i;
  double dum, sq= 0;
	df= N- constraints;
	for( i= 0; i< N; i++ ){
		if( pragma_unlikely(ascanf_verbose) && expected[i]<= 0 ){
			fprintf( StdErr, " (invalid 'expected' value %s in ChiSquare()) ", ad2str( expected[i], d3str_format, 0) );
		}
		dum= observed[i]- expected[i];
		sq+= dum* dum/ expected[i];
	}
	*prob= incomplete_gammaFunction_complement( 0.5* df, 0.5* sq );
	*dF= df;
	return( sq );
}

/* "ChiSquare-Prob[&observed, &expected, constraints, &chsq-return[, df-return]]" */
int ascanf_ChiSquare ( ASCB_ARGLIST )
{ ASCB_FRAME
	*result= -1;
	if( !Check_ascanf_SS() ){
		return(0);
	}
	if( ascanf_arguments>= 4 ){
	  ascanf_Function *af_obs, *af_exps, *af_chsq= NULL, *af_df= NULL;
	  int N= 0, N1, N2, constraints, free_obs, free_exps;
	  double *obs= NULL, *exps= NULL;
	  SimpleStats *ss= NULL;
		if( (af_obs= parse_ascanf_address( args[0], 0, "ascanf_ChiSquare", (ascanf_verbose)? -1 : 0, NULL )) ||
			(args[0]>= 0 && args[0]< ASCANF_MAX_ARGS && SS_valid_exact((ss= &ascanf_SS[(int)args[0]])))
		){
			if( ss || (af_obs->type== _ascanf_simplestats && af_obs->SS && af_obs->SS->exact &&
				af_obs->SS->sample && af_obs->SS->count)
			){
				if( !ss ){
					ss= af_obs->SS;
				}
				if( (obs= (double*) calloc( (N= ss->count), sizeof(double) )) ){
				  int i;
					for( i= 0; i< N; i++ ){
						obs[i]= ss->sample[i].value* ss->sample[i].weight;
					}
					free_obs= True;
				}
			}
			else if( af_obs && af_obs->type== _ascanf_array && af_obs->N ){
				N= af_obs->N;
				if( af_obs->array ){
					obs= af_obs->array;
					free_obs= False;
				}
				else{
					if( (obs= (double*) calloc( N, sizeof(double) )) ){
					  int i;
						for( i= 0; i< N; i++ ){
							obs[i]= af_obs->iarray[i];
						}
						free_obs= True;
					}
				}
			}
			if( !obs ){
				fprintf( StdErr, " (invalid observed array %s (%s)) ", ad2str( args[0], d3str_format,0), serror() );
				ascanf_arg_error= True;
			}
		}
		N1= N;
		  /* HERE: copy/paste support for ascanf_SS from above!! */
		if( (af_exps= parse_ascanf_address( args[1], 0, "ascanf_ChiSquare", (ascanf_verbose)? -1 : 0, NULL )) ||
			(args[1]>= 0 && args[1]< ASCANF_MAX_ARGS && SS_valid_exact((ss= &ascanf_SS[(int)args[1]])))
		){
			if( ss || (af_exps->type== _ascanf_simplestats && af_exps->SS && af_exps->SS->exact &&
				af_exps->SS->sample && af_exps->SS->count)
			){
				if( !ss ){
					ss= af_exps->SS;
				}
				if( (obs= (double*) calloc( (N= ss->count), sizeof(double) )) ){
				  int i;
					for( i= 0; i< N; i++ ){
						obs[i]= ss->sample[i].value* ss->sample[i].weight;
					}
					free_exps= True;
				}
			}
			else if( af_exps && af_exps->type== _ascanf_array && af_exps->N ){
				if( (N2= af_exps->N)< N ){
					N= af_exps->N;
				}
				if( af_exps->array ){
					exps= af_exps->array;
					free_exps= False;
				}
				else{
					if( (exps= (double*) calloc( N, sizeof(double) )) ){
					  int i;
						for( i= 0; i< N; i++ ){
							exps[i]= af_exps->iarray[i];
						}
						free_exps= True;
					}
				}
			}
			if( !exps ){
				fprintf( StdErr, " (invalid expected array %s (%s)) ", ad2str( args[1], d3str_format,0), serror() );
				ascanf_arg_error= True;
			}
		}
		constraints= (int) args[2];
		if( args[3] && !(af_chsq= parse_ascanf_address( args[3], _ascanf_variable, "ascanf_ChiSquare", (ascanf_verbose)? -1 : 0, NULL)) ){
			fprintf( StdErr, " (3rd argument must point to a scalar that will return the ChiSquare value) " );
		}
		if( ascanf_arguments> 4 && args[4] &&
			!(af_df= parse_ascanf_address( args[4], _ascanf_variable, "ascanf_ChiSquare", (ascanf_verbose)? -1 : 0, NULL))
		){
				fprintf( StdErr, " (4th argument must point to a scalar that will return the df value) " );
		}
		if( obs && exps && !ascanf_arg_error && !ascanf_SyntaxCheck ){
		  int df;
		  double sq;
			sq= ChiSquare( obs, exps, N, constraints, &df, result );
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (observed{%d} vs. expected{%d}: df=%d chi-square=%s, prob=%s) ",
					N1, N2, df,
					ad2str( sq, d3str_format,0), ad2str( *result, d3str_format, 0)
				);
			}
			if( af_chsq ){
				af_chsq->value= sq;
			}
			if( af_df ){
				af_df->value= df;
			}
			if( af_chsq && af_chsq->accessHandler ){
				AccessHandler( af_chsq, "ChiSquare-Prob", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			if( af_df && af_df->accessHandler ){
				AccessHandler( af_df, "ChiSquare-Prob", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}
		else if( !ascanf_SyntaxCheck ){
			ascanf_arg_error= True;
			ascanf_emsg= " (invalid argument(s)!) ";
			return(0);
		}
		if( free_obs ){
			xfree( obs );
		}
		if( free_exps ){
			xfree( exps );
		}
		return(1);
	}
	else{
		ascanf_arg_error= True;
		return(0);
	}
}

double ChiSquare2( double *observed1, double *observed2, int N, int constraints, int *dF, double *prob )
{ int i, df;
  double dum, sq= 0, S1= 0, S2= 0, a1= 1, a2= 1;
	for( i= 0; i< N; i++ ){
		S1+= observed1[i];
		S2+= observed2[i];
	}
	if( S1!= S2 ){
		if(!constraints ){
			constraints= 1;
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (ChiSquare2(): samplelengths not equal (%g!=%g); imposing constraints[1]!) ",
					S1, S2
				);
			}
		}
		a1= sqrt(S2/S1);
		a2= sqrt(S1/S2);
	}
	df= N- constraints;
	for( i= 0; i< N; i++ ){
		if( observed1[i]== 0 && observed2[i]== 0 ){
			df-= 1;
		}
		else{
			dum= a1* observed1[i]- a2* observed2[i];
			sq+= dum* dum/ (observed1[i] + observed2[i]);
		}
	}
	*prob= incomplete_gammaFunction_complement( 0.5* df, 0.5* sq );
	*dF= df;
	return( sq );
}

/* "ChiSquare2-Prob[&observed1, &observed2, constraints, &chsq-return[, df-return]]" */
int ascanf_ChiSquare2 ( ASCB_ARGLIST )
{ ASCB_FRAME
	*result= -1;
	if( !Check_ascanf_SS() ){
		return(0);
	}
	if( ascanf_arguments>= 4 ){
	  ascanf_Function *af_obs1, *af_obs2, *af_chsq= NULL, *af_df= NULL;
	  int N= 0, N1, N2, constraints, free_obs1, free_obs2;
	  double *obs1= NULL, *obs2= NULL;
	  SimpleStats *ss= NULL;
		if( (af_obs1= parse_ascanf_address( args[0], 0, "ascanf_ChiSquare2", (ascanf_verbose)? -1 : 0, NULL )) ||
			(args[0]>= 0 && args[0]< ASCANF_MAX_ARGS && SS_valid_exact((ss= &ascanf_SS[(int)args[0]])))
		){
			if( ss || (af_obs1->type== _ascanf_simplestats && af_obs1->SS && af_obs1->SS->exact &&
				af_obs1->SS->sample && af_obs1->SS->count)
			){
				if( !ss ){
					ss= af_obs1->SS;
				}
				if( (obs1= (double*) calloc( (N= ss->count), sizeof(double) )) ){
				  int i;
					for( i= 0; i< N; i++ ){
						obs1[i]= ss->sample[i].value* ss->sample[i].weight;
					}
					free_obs1= True;
				}
			}
			else if( af_obs1 && af_obs1->type== _ascanf_array && af_obs1->N ){
				N= af_obs1->N;
				if( af_obs1->array ){
					obs1= af_obs1->array;
					free_obs1= False;
				}
				else{
					if( (obs1= (double*) calloc( N, sizeof(double) )) ){
					  int i;
						for( i= 0; i< N; i++ ){
							obs1[i]= af_obs1->iarray[i];
						}
						free_obs1= True;
					}
				}
			}
			if( !obs1 ){
				fprintf( StdErr, " (invalid obs1erved array %s (%s)) ", ad2str( args[0], d3str_format,0), serror() );
				ascanf_arg_error= True;
			}
		}
		N1= N;
		if( (af_obs2= parse_ascanf_address( args[1], 0, "ascanf_ChiSquare2", (ascanf_verbose)? -1 : 0, NULL )) ||
			(args[1]>= 0 && args[1]< ASCANF_MAX_ARGS && SS_valid_exact((ss= &ascanf_SS[(int)args[1]])))
		){
			if( ss || (af_obs2->type== _ascanf_simplestats && af_obs2->SS && af_obs2->SS->exact &&
				af_obs2->SS->sample && af_obs2->SS->count)
			){
				if( !ss ){
					ss= af_obs2->SS;
				}
				if( (obs2= (double*) calloc( (N2= ss->count), sizeof(double) )) ){
				  int i;
					for( i= 0; i< N; i++ ){
						obs2[i]= ss->sample[i].value* ss->sample[i].weight;
					}
					free_obs2= True;
				}
			}
			else if( af_obs2 && af_obs2->type== _ascanf_array && af_obs2->N ){
				if( (N2= af_obs2->N)< N ){
					N= af_obs2->N;
				}
				if( af_obs2->array ){
					obs2= af_obs2->array;
					free_obs2= False;
				}
				else{
					if( (obs2= (double*) calloc( N, sizeof(double) )) ){
					  int i;
						for( i= 0; i< N; i++ ){
							obs2[i]= af_obs2->iarray[i];
						}
						free_obs2= True;
					}
				}
			}
			if( !obs2 ){
				fprintf( StdErr, " (invalid expected array %s (%s)) ", ad2str( args[1], d3str_format,0), serror() );
				ascanf_arg_error= True;
			}
		}
		constraints= (int) args[2];
		if( args[3] &&
			!(af_chsq= parse_ascanf_address( args[3], _ascanf_variable, "ascanf_ChiSquare2", (ascanf_verbose)? -1 : 0, NULL))
		){
			fprintf( StdErr, " (3rd argument must point to a scalar that will return the ChiSquare value) " );
		}
		if( ascanf_arguments> 4 && args[4] &&
			!(af_df= parse_ascanf_address( args[4], _ascanf_variable, "ascanf_ChiSquare2", (ascanf_verbose)? -1 : 0, NULL))
		){
			fprintf( StdErr, " (4th argument must point to a scalar that will return the df value) " );
		}
		if( obs1 && obs2 && !ascanf_arg_error && !ascanf_SyntaxCheck ){
		  int df;
		  double sq= ChiSquare2( obs1, obs2, N, constraints, &df, result );
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (observed1{%d} vs. observed2{%d}: df=%d chi-square=%s, prob=%s) ",
					N1, N2, df,
					ad2str( sq, d3str_format,0), ad2str( *result, d3str_format, 0)
				);
			}
			if( af_chsq ){
				af_chsq->value= sq;
			}
			if( af_df ){
				af_df->value= df;
			}
			if( af_chsq && af_chsq->accessHandler ){
				AccessHandler( af_chsq, "ChiSquare2-Prob", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			if( af_df && af_df->accessHandler ){
				AccessHandler( af_df, "ChiSquare2-Prob", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
		}
		else if( !ascanf_SyntaxCheck ){
			ascanf_arg_error= True;
			ascanf_emsg= " (invalid argument(s)!) ";
			return(0);
		}
		if( free_obs1 ){
			xfree( obs1 );
		}
		if( free_obs2 ){
			xfree( obs2 );
		}
		return(1);
	}
	else{
		ascanf_arg_error= True;
		return(0);
	}
}

int ascanf_SS_setArray ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int bin, start= 0, end;
  ascanf_Function *Data= NULL, *Weight= NULL;
  double w= 1;
	*result= 0;
	if( !ascanf_SS && !ascanf_SyntaxCheck ){
		if( !ASCANF_MAX_ARGS ){
			ASCANF_MAX_ARGS= AMAXARGSDEFAULT;
		}
		if( !(ascanf_SS= (SimpleStats*) calloc( ASCANF_MAX_ARGS, sizeof(SimpleStats))) ){
			fprintf( StdErr, "Can't get memory for ascanf_SS[%d] array (%s)\n", ASCANF_MAX_ARGS, serror() );
			ascanf_arg_error= 1;
			return(0);
		}
	}
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
	  int i, N;
	  extern int ascanf_SS_set_item();
		if( ascanf_arguments== 1 ){
			return( ascanf_SS_set_item( args,result, 0 ) );
		}
		if( (args[0]= ssfloor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
			bin= (int) args[0];
		}
		if( !(Data= parse_ascanf_address(args[1], _ascanf_array, "ascanf_SS_setArray", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid src_p array argument (2)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= Data->N;
		}
		if( ascanf_arguments> 2 && args[2] ){
			if( !(Weight= parse_ascanf_address(args[2], 0, "ascanf_SS_setArray", (int) ascanf_verbose, NULL )) ||
				!(Weight->type== _ascanf_array || Weight->type== _ascanf_variable)
			){
				ascanf_emsg= " (invalid weight_p array or variable argument (3)) ";
				ascanf_arg_error= 1;
			}
			if( Weight && Weight->type== _ascanf_variable ){
				w= Weight->value;
				Weight= NULL;
			}
		}
		if( ascanf_arguments> 3 ){
			CLIP_EXPR( start, (int) args[3], 0, N-1 );
		}
		if( ascanf_arguments> 4 ){
			if( args[4]< 0 ){
				end= N;
			}
			else{
				CLIP_EXPR( end, (int) args[4], start, N );
			}
		}
		else{
			end= N;
		}
		if( !ascanf_arg_error && N && !ascanf_SyntaxCheck ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr,
					" (adding %s[%d] weight=%s, from %d upto %d) ",
					Data->name, Data->N,
					(Weight)? ad2str( take_ascanf_address(Weight), d3str_format, 0 ) : "<none>",
					start, end
				);
			}
			if( Data->iarray ){
				for( i= start; i< end; i++ ){
					if( Weight ){
						if( i< Weight->N ){
							w= (Weight->iarray)? Weight->iarray[i] : Weight->array[i];
						}
					}
					if( bin< 0 ){
					  double largs[3];
						largs[0]= bin;
						largs[1]= Data->iarray[i];
						largs[2]= w;
						  /* Unfortunately, either I have to copy a bunch of code, or I have to
						   \ duplicate some actions. I chose the former in this case.
						   */
						ascanf_SS_set_item( largs, result, 0 );
					}
					else{
					  double dw= Data->iarray[i]*w;
						if( Check_Ignore_NaN( *SS_Ignore_NaN, dw) ){
							SS_Add_Data_( ascanf_SS[bin], 1, Data->iarray[i], w );
						}
					}
				}
			}
			else{
				for( i= start; i< end; i++ ){
					if( Weight ){
						if( i< Weight->N ){
							w= (Weight->iarray)? Weight->iarray[i] : Weight->array[i];
						}
					}
					if( bin< 0 ){
					  double largs[3];
						largs[0]= bin;
						largs[1]= Data->array[i];
						largs[2]= w;
						  /* Unfortunately, either I have to copy a bunch of code, or I have to
						   \ duplicate some actions. I chose the former in this case.
						   */
						ascanf_SS_set_item( largs, result, 0 );
					}
					else{
					  double dw= Data->array[i]*w;
						if( Check_Ignore_NaN( *SS_Ignore_NaN, dw) ){
							SS_Add_Data_( ascanf_SS[bin], 1, Data->array[i], w );
						}
					}
				}
			}
			*result= Data->value;
		}
	}
	return(1);
}

int ascanf_SAS_setArray ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int bin, start= 0, end;
  ascanf_Function *Data= NULL, *Weight= NULL;
  double w= 1, radix= M_2PI, offset= 0, convert;
	*result= 0;
	if( !ascanf_SAS && !ascanf_SyntaxCheck ){
		if( !ASCANF_MAX_ARGS ){
			ASCANF_MAX_ARGS= AMAXARGSDEFAULT;
		}
		if( !(ascanf_SAS= (SimpleAngleStats*) calloc( ASCANF_MAX_ARGS, sizeof(SimpleAngleStats))) ){
			fprintf( StdErr, "Can't get memory for ascanf_SAS[%d] array (%s)\n", ASCANF_MAX_ARGS, serror() );
			ascanf_arg_error= 1;
			return(0);
		}
	}
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
	  int i, N;
	  extern int ascanf_SAS_set_item();
		if( ascanf_arguments== 1 ){
			return( ascanf_SAS_set_item( args,result, 0 ) );
		}
		if( (args[0]= ssfloor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
			bin= (int) args[0];
		}
		if( !(Data= parse_ascanf_address(args[1], _ascanf_array, "ascanf_SAS_setArray", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid src_p array argument (2)) ";
			ascanf_arg_error= 1;
		}
		else{
			N= Data->N;
		}
		if( ascanf_arguments> 2 && args[2] ){
			if( !(Weight= parse_ascanf_address(args[2], 0, "ascanf_SAS_setArray", (int) ascanf_verbose, NULL )) ||
				!(Weight->type== _ascanf_array || Weight->type== _ascanf_variable)
			){
				ascanf_emsg= " (invalid weight_p array or variable argument (3)) ";
				ascanf_arg_error= 1;
			}
		}
		if( ascanf_arguments> 3 ){
			CLIP_EXPR( start, (int) args[3], 0, N-1 );
		}
		if( ascanf_arguments> 4 ){
			if( args[4]< 0 ){
				end= N;
			}
			else{
				CLIP_EXPR( end, (int) args[4], start, N );
			}
		}
		else{
			end= N;
		}
		if( ascanf_arguments> 5 && args[5] ){
			radix= args[5];
		}
		else{
			radix= M_2PI;
		}
		if( ascanf_arguments> 6 ){
			offset= args[6];
		}
		else{
			offset= 0;
		}
		if( ascanf_arguments> 7 ){
			convert= (args[7])? True : False;
		}
		else{
			convert= *SAS_converts_angle;
		}
		if( Weight && Weight->type== _ascanf_variable ){
			w= Weight->value;
			Weight= NULL;
		}
		if( !ascanf_arg_error && N && !ascanf_SyntaxCheck ){
			if( Data->iarray ){
				for( i= start; i< end; i++ ){
					if( Weight ){
						if( i< Weight->N ){
							w= (Weight->iarray)? Weight->iarray[i] : Weight->array[i];
						}
					}
					if( bin< 0 ){
					  double largs[6];
						largs[0]= bin;
						largs[1]= Data->iarray[i];
						largs[2]= w;
						largs[3]= radix;
						largs[4]= offset;
						largs[5]= convert;
						  /* Unfortunately, either I have to copy a bunch of code, or I have to
						   \ duplicate some actions. I chose the former in this case.
						   */
						ascanf_SAS_set_item( largs, result, 0 );
					}
					else{
					  double dw= Data->iarray[i]*w;
						if( Check_Ignore_NaN( *SAS_Ignore_NaN, dw) ){
							ascanf_SAS[bin].Gonio_Base= radix;
							ascanf_SAS[bin].Gonio_Offset= offset;
							SAS_Add_Data_( ascanf_SAS[bin], 1, Data->iarray[i], w, (int) convert );
						}
					}
				}
			}
			else{
				for( i= start; i< end; i++ ){
					if( Weight ){
						if( i< Weight->N ){
							w= (Weight->iarray)? Weight->iarray[i] : Weight->array[i];
						}
					}
					if( bin< 0 ){
					  double largs[6];
						largs[0]= bin;
						largs[1]= Data->array[i];
						largs[2]= w;
						largs[3]= radix;
						largs[4]= offset;
						largs[5]= convert;
						  /* Unfortunately, either I have to copy a bunch of code, or I have to
						   \ duplicate some actions. I chose the former in this case.
						   */
						ascanf_SAS_set_item( largs, result, 0 );
					}
					else{
					  double dw= Data->array[i]*w;
						if( Check_Ignore_NaN( *SAS_Ignore_NaN, dw) ){
							ascanf_SAS[bin].Gonio_Base= radix;
							ascanf_SAS[bin].Gonio_Offset= offset;
							SAS_Add_Data_( ascanf_SAS[bin], 1, Data->array[i], w, (int) convert );
						}
					}
				}
			}
			*result= Data->value;
		}
	}
	return(1);
}

