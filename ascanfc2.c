/* ascanfc2.c: simple scanner for double arrays
vim:ts=4:sw=4:
 \ that doubles as a function interpreter.
 \ 2nd installment/incarnation.
 \ Includes a "compiler".
 * (C) R.J.V. Bertin 1999-..
 * :ts=4
 * :sw=4
 */

#include "config.h"
IDENTIFY( "ascanf supplementary module" );

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>


#include "cpu.h"
#include "copyright.h"

#include "math.h"
#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
#	define USE_SSE2
#	include <xmmintrin.h>
#	include <emmintrin.h>
#	include "AppleVecLib.h"
#endif


#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "NaN.h"

#include "fdecl.h"

#include "ascanf.h"
#include "compiled_ascanf.h"

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#	include "arrayvops.h"
#elif defined(__SSE2__) || defined(__SSE3__)
#	include "arrayvops.h"
#endif

extern void _ascanf_array_access( Compiled_Form *lform, ascanf_Function *larray,
	int lfirst, int lidx, int lN, double *larg, double *lresult, int *llevel
);


#include <signal.h>

#include "ux11/ux11.h"

#if !defined(__MACH__) && !defined(__CYGWIN__)
#	include <values.h>
#endif

#ifndef SWAP
#	define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}
#endif
#ifndef IMIN
static int mi1,mi2;
#	define IMIN(m,n)	(mi1=(m),mi2=(n),(mi1<mi2)? mi1 : mi2)
#endif
#ifndef MAX
#	define MAX(a,b)                (((a)>(b))?(a):(b))
#endif

#include "SS.h"

extern FILE *NullDevice;

#ifdef sgi
#	include <sys/time.h>
#else
#	include <time.h>
#endif

extern char *matherr_mark();
extern int matherr_verbose;
#define MATHERR_MARK()	matherr_mark(__FILE__ ":" STRING(__LINE__))

/* extern char *d2str( double, const char*, char *), *ad2str( double, const char *, char *);	*/
extern char d3str_format[16];

extern FILE *StdErr;
extern int debugFlag, debugLevel, line_count;

#include "xfree.h"

extern Pixmap dotMap ;
extern Window ascanf_window;
extern LocalWin *ActiveWin, *StubWindow_ptr;
extern Display *disp;

extern char *ascanf_type_name[_ascanf_types];

extern int ascanf_SyntaxCheck;
extern double *Allocate_Internal;
extern int ascanf_PopupWarn;

#define FUNNAME(fun)	((fun)?((fun)->name)?(fun)->name:"??":"NULL")
#define FORMNAME(f)		((f)? ((f)->name)?(f)->name : FUNNAME((f)->fun) : "NULL")

extern char ascanf_separator;

extern int Ascanf_Max_Args;

/* A lot of the following ..ascanf_.. variables should be lumped into a structure
 \ associated with a LocalWin (plus a global set). This should allow really
 \ reentrant code, where concurrent processings in different windows won't interfere...
 */
extern int reset_ascanf_currentself_value, reset_ascanf_index_value;
extern double *ascanf_self_value, *ascanf_current_value, *ascanf_index_value;
extern double *ascanf_memory, ascanf_progn_return;
extern int ascanf_arguments, ascanf_arg_error, ascanf_comment, ascanf_popup;
extern char ascanf_errmesg[512];
extern char *ascanf_emsg;

extern double *ascanf_counter, *ascanf_Counter;

extern char **ascanf_SS_names;
extern SimpleStats *ascanf_SS;
extern char **ascanf_SAS_names;
extern SimpleAngleStats *ascanf_SAS;

extern NormDist *ascanf_normdists;

extern double *ascanf_data_buf;
extern int *ascanf_column_buf;

extern double *ascanf_loop_counter;
extern int ascanf_escape;

extern int ascanf_check_now, ascanf_check_int, ascanf_propagate_events;

DEFUN(ascanf_if, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_if2, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_progn,  ( ASCB_ARGLIST ) , int);
DEFUN(ascanf_verbose_fnc,  ( ASCB_ARGLIST ) , int);
DEFUN(ascanf_matherr_fnc,  ( ASCB_ARGLIST ) , int);
DEFUN(ascanf_comment_fnc,  ( ASCB_ARGLIST ) , int);
DEFUN(ascanf_popup_fnc,  ( ASCB_ARGLIST ) , int);
DEFUN(fascanf, ( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], struct Compiled_Form **form), int);
DEFUN(__fascanf, ( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], struct Compiled_Form **form, int *level, int *nargs), int);
DEFUN( ascanf_whiledo,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_dowhile,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_for_to,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_print,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_Eprint,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_Dprint,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_Variable,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_Procedure,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_DeclareVariable,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_DeclareProcedure,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_DeleteVariable,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_DefinedVariable,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_compile,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_noEval,  ( ASCB_ARGLIST ) , int );
DEFUN( ascanf_SHelp,  ( ASCB_ARGLIST ) , int );

extern char *ascanf_var_search;

extern void *vars_pmenu;
extern ascanf_type current_function_type, current_DCL_type;
extern int ascanf_verbose;

extern char *SS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleStats *a);
extern char *SS_sprint( char *buffer, char *format, char *sep, double min_err, SimpleStats *a);

extern char *parse_codes(char *c);

extern double *SS_TValue, *SS_FValue;

extern char *SAS_sprint_full( char *buffer, char *format, char *sep, double min_err, SimpleAngleStats *a);

extern double **ascanf_mxy_buf;
extern int ascanf_mxy_X, ascanf_mxy_Y;

extern int ascanf_exit;

#include "DataSet.h"
extern DataSet *AllSets;
extern int MaxSets, maxitems;

extern double *ascanf_setNumber;
extern int setNumber;

extern double ascanf_log_zero_x, ascanf_log_zero_y;

extern int ascanf_xmin ( ASCB_ARGLIST ) ;
extern int ascanf_xmax ( ASCB_ARGLIST ) ;
extern int ascanf_ymin ( ASCB_ARGLIST ) ;
extern int ascanf_ymax ( ASCB_ARGLIST ) ;
extern int ascanf_errmin ( ASCB_ARGLIST ) ;
extern int ascanf_errmax ( ASCB_ARGLIST ) ;
extern int ascanf_tr_xmin ( ASCB_ARGLIST ) ;
extern int ascanf_tr_xmax ( ASCB_ARGLIST ) ;
extern int ascanf_tr_ymin ( ASCB_ARGLIST ) ;
extern int ascanf_tr_ymax ( ASCB_ARGLIST ) ;
extern int ascanf_tr_errmin ( ASCB_ARGLIST ) ;
extern int ascanf_tr_errmax ( ASCB_ARGLIST ) ;
extern int ascanf_curve_len ( ASCB_ARGLIST ) ;
extern int ascanf_error_len ( ASCB_ARGLIST ) ;
extern int ascanf_tr_curve_len ( ASCB_ARGLIST ) ;

extern int ascanf_xcol ( ASCB_ARGLIST ) ;

extern int ascanf_ycol ( ASCB_ARGLIST ) ;

extern int ascanf_ecol ( ASCB_ARGLIST ) ;

extern double drand48();
#define drand()	drand48()

extern IEEEfp zero_div_zero;

extern int ascanf_SplitHere;

#include "Elapsed.h"
extern double Tot_Start, Tot_Time, Used_Time;

#ifndef degrees
#	define degrees(a)			((a)*57.295779512)
#endif
#ifndef radians
#	define radians(a)			((a)/57.295779512)
#endif

#include <float.h>

extern char *TBARprogress_header;


extern double *param_scratch;
extern int param_scratch_len, param_scratch_inUse;

ascanf_Function *Create_Internal_ascanfString( char *string, int *level )
{ char *lbuf;
  int N= 1;
  ascanf_type type;
  ascanf_Function *allocated= NULL;
  double result= 0;
	if( string ){
		if( (lbuf = (char*) calloc( strlen(string) + 256 + 5, sizeof(char) )) ){
			sprintf( lbuf, "`\"%s\"", string );
			Create_AutoVariable( lbuf, &result, 0, &N, NULL, &type, level, &allocated, (ascanf_verbose)? True : False );
			if( allocated ){
				  /* We must now make sure that the just created internal variable behaves as a user variable: */
				allocated->user_internal= True;
				allocated->internal= True;
			}
			xfree(lbuf);
		}
		else{
			fprintf( StdErr, "Create_Internal_ascanfString(%s) couldn't allocate working memory (%s)\n", string, serror() );
		}
	}
	return( allocated );
}

ascanf_Function *Create_ascanfString( char *string, int *level )
{ char *lbuf;
  int N= 1;
  ascanf_type type;
  ascanf_Function *allocated= NULL;
  double result= 0;
	if( string ){
		if( (lbuf = (char*) calloc( strlen(string) + 256 + 5, sizeof(char) )) ){
			sprintf( lbuf, "`\"%s\"", string );
			Create_AutoVariable( lbuf, &result, 0, &N, NULL, &type, level, &allocated, (ascanf_verbose)? True : False );
			xfree(lbuf);
		}
		else{
			fprintf( StdErr, "Create_ascanfString(%s) couldn't allocate working memory (%s)\n", string, serror() );
		}
	}
	return( allocated );
}

ascanf_Function *init_Static_StringPointer( ascanf_Function *af, char *AFname )
{
	if( af && AFname ){
		if( af->name ){
			xfree(af->usage);
			memset( af, 0, sizeof(ascanf_Function) );
		}
		else{
			af->usage= NULL;
		}
		af->type= _ascanf_variable;
		af->name= AFname;
		af->is_address= af->take_address= True;
		af->is_usage= af->take_usage= True;
		af->internal= True;
	}
	return af;
}

// 20090413: add a Create_ascanfString() function that doesn't allocate on the IDict

/* savgol functions were here */

double *ascanf_getDataColumn( int set_nr, int col_nr, int *N )
{ double *col=NULL;
	if( pragma_likely(set_nr>=0 && set_nr< setNumber) ){
	  DataSet *this_set= &AllSets[set_nr];
		if( pragma_likely(this_set->columns && col_nr>=0 && col_nr< this_set->ncols) ){
			col= this_set->columns[col_nr];
			if( N ){
				*N= this_set->numPoints;
			}
		}
		else{
			ascanf_emsg= " (attempt to access invalid/non-existing DataSet column) ";
			ascanf_arg_error= 1;
		}
	}
	else{
		ascanf_emsg= " (attempt to access invalid/non-existing DataSet) ";
		ascanf_arg_error= 1;
	}
	return(col);
}

int unregister_LinkedArray( ascanf_Function *af )
{ int r= -1, snr;
	if( af->linkedArray.set_nr < 0 ){
	  // we can be called after a variable was marked as belonging to a deleted set...
		snr= -(af->linkedArray.set_nr) - 1;
	}
	else{
		snr= af->linkedArray.set_nr;
	}
	if( snr>= 0 && snr< setNumber && AllSets ){
		r= remove_LinkedArray_from_List( &(AllSets[snr].LinkedArrays), af );
	}
	return(r);
}

int Check_Sets_LinkedArrays( DataSet *set )
{ int n= 0;
	if( set->LinkedArrays ){
	  ascanf_Function *af;
	  void *iter= NULL;
		while( (af= walk_LinkedArray_List( &set->LinkedArrays, &iter )) ){
			if( af->type== _ascanf_array ){
				Check_linkedArray(af);
				n+= 1;
			}
			else{
				unregister_LinkedArray(af);
			}
		}
	}
	return(n);
}

int ascanf_LinkArray2DataColumn( ASCB_ARGLIST )
{ ASCB_FRAME
  double idx, col;
  DataSet *set;
  ascanf_Function *targ;
	*result= 0;
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( !(targ= parse_ascanf_address(args[0], 0, "ascanf_LinkArray2DataColumn", (int) ascanf_verbose, NULL )) || targ->type!= _ascanf_array ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( (idx= args[1])< 0 || idx>= setNumber || AllSets[(int)idx].numPoints<= 0 ){
		ascanf_emsg= " (setnumber out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	set= &AllSets[(int) idx];
	if( (col= args[2])< 0 || col>= set->ncols ){
		ascanf_emsg= " (column number out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	xfree(targ->iarray);
	if( targ->array!= targ->linkedArray.dataColumn ){
		xfree(targ->array);
	}
	else{
		if( targ->linkedArray.set_nr==(short) idx && targ->linkedArray.col_nr==(short)col ){
			  // just in case...
			targ->array= targ->linkedArray.dataColumn= set->columns[(int) col];
			*result= targ->last_index;
			return(1);
		}
		else{
			unregister_LinkedArray(targ);
		}
	}
	targ->array= targ->linkedArray.dataColumn= set->columns[(int) col];
	targ->linkedArray.set_nr= (short) idx;
	targ->linkedArray.col_nr= (short) col;
	targ->N= set->numPoints;
	targ->last_index= 0;
	targ->value= targ->array[targ->last_index];
	if( targ->accessHandler ){
		AccessHandler( targ, "LinkArray2DataColumn", level, ASCB_COMPILED, AH_EXPR, NULL   );
	}

	register_LinkedArray_in_List( &set->LinkedArrays, targ );

	*result= targ->last_index;
	return(1);
}

int ascanf_DataColumn2Array ( ASCB_ARGLIST )
{ ASCB_FRAME
  double idx, col, *column;
  long start= 0, end= -1, offset= 0, i, j, N;
  DataSet *set;
  ascanf_Function *targ, *visible= NULL;
  double pad_low, pad_high;
  int pad= 0, pl_set= 0, ph_set= 0, use_set_visible= True;
	*result= 0;
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( !(targ= parse_ascanf_address(args[0], 0, "ascanf_DataColumn2Array", (int) ascanf_verbose, NULL )) || targ->type!= _ascanf_array ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( (idx= args[1])< 0 || idx>= setNumber || AllSets[(int)idx].numPoints<= 0 ){
		ascanf_emsg= " (setnumber out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	set= &AllSets[(int) idx];
	if( (col= args[2])< 0 || col>= set->ncols ){
		ascanf_emsg= " (column number out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( targ->linkedArray.dataColumn && targ->linkedArray.set_nr== (short) idx && targ->linkedArray.col_nr== (short) col ){
		ascanf_emsg= " (won't copy a column onto itself) ";
		ascanf_arg_error= 1;
		return(0);
	}
	column= set->columns[(int) col];
	if( ascanf_arguments> 3 ){
		if( !(visible= parse_ascanf_address(args[3], 0, "ascanf_DataColumn2Array", (ascanf_verbose)? -1 : 0, NULL ))
			|| visible->type!= _ascanf_array
		){
/* 			CLIP(args[3], 0, set->numPoints-1 );	*/
/* 			start= (int) args[3];	*/
			CLIP_EXPR( start, (int) args[3], 0, set->numPoints-1 );
		}
	}
	if( !visible ){
		if( start< 0 ){
			start= 0;
		}
		if( ascanf_arguments> 4 ){
/* 			CLIP(args[4], -1, set->numPoints-1 );	*/
/* 			end= (int) args[4];	*/
			CLIP_EXPR( end, (int) args[4], -1, set->numPoints-1 );
		}
		if( end< 0 ){
			end= set->numPoints-1;
		}
		if( ascanf_arguments> 5 ){
/* 			CLIP(args[5], 0, MAXINT );	*/
/* 			offset= (int) args[5];	*/
			CLIP_EXPR( offset, (int) args[5], 0, MAXINT );
		}
		if( offset< 0 ){
			offset= 0;
		}
		if( ascanf_arguments > 6 && args[6]> 0 ){
		  double p;
			if( args[6]> 0 && (p= ssfloor(args[6]))< MAXINT ){
				pad= p;
			}
		}
		if( pad ){
			if( ascanf_arguments> 7 && !NaN(args[7]) ){
				pad_low= args[7];
				pl_set= 1;
			}
			else{
				pad_low= column[start];
				pl_set= 1;
			}
			if( ascanf_arguments> 8 && !NaN(args[8]) ){
				pad_high= args[8];
				ph_set= 1;
			}
			else{
				pad_high= column[end];
				ph_set= 1;
			}
		}
	}
	else{
		offset= 0;
		start= 0;
		end= set->numPoints- 1;
		if( ascanf_arguments> 4 && ascanf_verbose ){
			fprintf( StdErr, " (<offset> and subsequent arguments ignored)== " );
		}
		if( ascanf_arguments> 4 ){
			use_set_visible= ASCANF_TRUE(args[4]);
		}
	}
	if( ascanf_SyntaxCheck ){
		return(1);
	}
	if( visible ){
		N= 0;
		if( use_set_visible ){
			if( ActiveWin && ActiveWin!= StubWindow_ptr ){
				for( i= 0; i<= end; i++ ){
					N+= ActiveWin->pointVisible[(int) idx][i];
				}
				if( ActiveWin->numVisible[(int) idx]!= N ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (DataColumn2Array[%d]: %d points visible according to numVisible, pointVisible says %d)== ",
							(int) idx, ActiveWin->numVisible[(int) idx], N
						);
					}
				}
			}
			else if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (Warning: no window active, so 0 points are taken to be visible)== " );
			}
		}
		else{
			N= visible->N;
			for( i= 0; i< visible->N; i++ ){
			  double vidx= ASCANF_ARRAY_ELEM(visible,i);
				if( vidx< 0 || vidx>= set->numPoints ){
					N-= 1;
				}
			}
			if( pragma_unlikely(ascanf_verbose) && N!= visible->N ){
				fprintf( StdErr, " (Warning: %s references %d invalid datapoints) ", visible->N - N );
			}
		}
	}
	else{
		N= offset+ end- start+ 1+ 2*pad;
	}

	if( targ->linkedArray.dataColumn ){
		N= MIN(N, targ->N);
	}

	{ int n= MAX( N, 1 );
		if( !targ->linkedArray.dataColumn ){
			Resize_ascanf_Array( targ, n, NULL );
		}
		if( visible && use_set_visible ){
			Resize_ascanf_Array( visible, n, NULL );
		}
	}
	if( visible && N && use_set_visible ){
	  signed char *pointVisible= (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin->pointVisible[(int) idx] : NULL;
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
		visible->value= ASCANF_ARRAY_ELEM(visible, visible->last_index);
		if( visible->accessHandler ){
			AccessHandler( visible, "DataColumn2Array", level, ASCB_COMPILED, AH_EXPR, NULL   );
		}
	}
	if( targ->iarray ){
		if( visible ){
			if( N ){
			  int j, vidx;
				start= (int) ASCANF_ARRAY_ELEM(visible,0);
				end= (int) ASCANF_ARRAY_ELEM(visible,N-1);
				if( !use_set_visible ){
					CLIP( start, 0, set->numPoints-1 );
					CLIP( end, 0, set->numPoints-1 );
				}
				if( visible->iarray ){
					for( j= i= 0; i< N; i++ ){
						vidx= visible->iarray[i];
						if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
							targ->iarray[j]= (int) column[vidx];
							j++;
						}
					}
				}
				else{
					for( j= i= 0; i< N; i++ ){
						vidx= (int) visible->array[i];
						if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
							targ->iarray[j]= (int) column[vidx];
							j++;
						}
					}
				}
				targ->last_index= N-1;
			}
			j= N;
		}
		else{
			targ->last_index= offset;
			for( j= 0; j< pad; j++, targ->last_index++ ){
				targ->value= targ->iarray[targ->last_index]= (int) pad_low;
			}
			for( j= 0, i= start; i<= end; j++, i++, targ->last_index++ ){
				targ->value= targ->iarray[targ->last_index]= (int) column[i];
			}
			for( j= 0; j< pad; j++, targ->last_index++ ){
				targ->value= targ->iarray[targ->last_index]= (int) pad_high;
			}
			targ->last_index-= 1;
		}
		targ->value= targ->iarray[targ->last_index];
	}
	else{
		if( visible ){
			if( N ){
			  int j, vidx;
				start= (int) ASCANF_ARRAY_ELEM(visible,0);
				end= (int) ASCANF_ARRAY_ELEM(visible,N-1);
				if( !use_set_visible ){
					CLIP( start, 0, set->numPoints-1 );
					CLIP( end, 0, set->numPoints-1 );
				}
				if( visible->iarray ){
					for( j= i= 0; i< N; i++ ){
						vidx= visible->iarray[i];
						if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
							targ->array[j]= column[vidx];
							j++;
						}
					}
				}
				else{
					for( j= i= 0; i< N; i++ ){
						vidx= (int) visible->array[i];
						if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
							targ->array[j]= column[vidx];
							j++;
						}
					}
				}
				targ->last_index= N-1;
			}
			j= N;
		}
		else{
			targ->last_index= offset;
			for( j= 0; j< pad; j++, targ->last_index++ ){
				targ->value= targ->array[targ->last_index]= pad_low;
			}
			for( j= 0, i= start; i<= end; j++, i++, targ->last_index++ ){
				targ->value= targ->array[targ->last_index]= column[i];
			}
			for( j= 0; j< pad; j++, targ->last_index++ ){
				targ->value= targ->array[targ->last_index]= pad_high;
			}
			targ->last_index-= 1;
		}
		targ->value= targ->array[targ->last_index];
	}
	if( targ->accessHandler ){
		AccessHandler( targ, "DataColumn2Array", level, ASCB_COMPILED, AH_EXPR, NULL   );
	}
	if( pragma_unlikely(ascanf_verbose) ){
	  LabelsList *Labels= (set->ColumnLabels)? set->ColumnLabels : ((ActiveWin)? ActiveWin->ColumnLabels : NULL);
		fprintf( StdErr, " (copied %d(%d-%d) %selements from set#%d column %d to %s[%d]",
			j, start, end, (visible)? "visible " : "", set->set_nr, (int) col, targ->name, offset
		);
		if( Labels ){
		  char *c= Find_LabelsListLabel( Labels, (int)col );
			if( c ){
				fprintf( StdErr, "(%s)", c );
			}
		}
		if( pad ){
			fprintf( StdErr, " padded with %dx%s (low) and %dx%s (high)",
				pad, ad2str(pad_low, d3str_format,0), pad, ad2str( pad_high, d3str_format,0)
			);
		}
		fprintf( StdErr, " (%d elements))== ", targ->N );
	}
	*result= targ->last_index- offset;
	return(1);
}

int ascanf_Array2DataColumn ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double idx, col, *column;
  long start= 0, end= -1, offset= 0, i, j;
  DataSet *set;
  ascanf_Function *targ;
	*result= 0;
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( (idx= args[0])< 0 || idx> setNumber || AllSets[(int)idx].numPoints< 0 ){
		ascanf_emsg= " (setnumber out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	set= &AllSets[(int) idx];
	if( (col= args[1])< 0 || col>= set->ncols ){
		ascanf_emsg= " (column number out of range) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( !(targ= parse_ascanf_address(args[2], 0, "ascanf_Array2DataColumn", (int) ascanf_verbose, NULL )) || targ->type!= _ascanf_array ){
		ascanf_emsg= " (invalid array argument (3)) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( targ->linkedArray.dataColumn && targ->linkedArray.set_nr== (short) idx && targ->linkedArray.col_nr== (short) col ){
		ascanf_emsg= " (won't copy a column onto itself) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( ascanf_arguments> 3 ){
/* 		CLIP(args[3], 0, targ->N-1 );	*/
/* 		start= (int) args[3];	*/
		CLIP_EXPR(start, (int) args[3], 0, targ->N-1 );
	}
	if( ascanf_arguments> 4 ){
/* 		CLIP(args[4], -1, targ->N );	*/
/* 		end= (int) args[4];	*/
		CLIP_EXPR(end, (int) args[4], -1, targ->N-1 );
	}
	if( end== -1 ){
		end= targ->N- 1;
	}
	if( ascanf_arguments> 5 ){
/* 		CLIP(args[5], 0, MAXINT );	*/
/* 		offset= (int) args[5];	*/
		CLIP_EXPR(offset, (int) args[5], 0, MAXINT );
	}
	if( set->numPoints< (offset+ end- start+ 1) && !ascanf_SyntaxCheck ){
	  int n= offset+ end- start+ 1;
	  int old_n= set->numPoints;
		set->numPoints= n;
		realloc_points( set, n, False );
		if( ActiveWin && n> maxitems ){
			maxitems= n;
			realloc_Xsegments();
		}
		for( j= 0; j< set->ncols; j++ ){
			for( i= old_n; i< n && set->columns[j]; i++ ){
				set->columns[j][i]= 0;
			}
		}
	}
	column= set->columns[(int) col];
	if( ascanf_SyntaxCheck ){
		return(1);
	}
	if( targ->iarray ){
		for( j= 0, i= start; i<= end; j++, i++ ){
			column[offset+j]= targ->value= (double) targ->iarray[(targ->last_index=i)];
		}
/* 		targ->value= targ->iarray[(targ->last_index=i-1)];	*/
	}
	else{
		for( j= 0, i= start; i<= end; j++, i++ ){
			column[offset+j]= targ->value= targ->array[(targ->last_index=i)];
		}
/* 		targ->value= targ->array[(targ->last_index=i-1)];	*/
	}
	if( pragma_unlikely(ascanf_verbose) ){
	  LabelsList *Labels= (set->ColumnLabels)? set->ColumnLabels : ((ActiveWin)? ActiveWin->ColumnLabels : NULL);
		fprintf( StdErr, " (copied %d elements from %s[%d-%d] (%d elements) to set#%d column #%d[%d]",
			j, targ->name, start, end, targ->N, set->set_nr, (int) col, offset
		);
		if( Labels ){
		  char *c= Find_LabelsListLabel( Labels, (int)col );
			if( c ){
				fprintf( StdErr, "(%s)", c );
			}
		}
		fputs( ")== ", StdErr );
	}
	*result= j;
	return(1);
}

int ascanf_Set2Arrays ( ASCB_ARGLIST )
{ ASCB_FRAME
  double idx, *column;
  DataSet *set;
  ascanf_Function *array[5], *targ, *visible= NULL;
  long start= 0, end= -1, offset= 0, i, j, col, N;
  double pad_low, pad_high;
  int pad= 0, pl_set= 0, ph_set= 0, cooked, use_set_visible= True;
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return(0);
	}
	memset( array, 0, sizeof(array) );
	if( (idx= args[0])< 0 || idx>= setNumber ){
		ascanf_emsg= " (setnumber out of range) ";
		ascanf_arg_error= 1;
	}
	set= &AllSets[(int) idx];
	cooked= ! ASCANF_TRUE(args[1]);
	if( !(array[0]= parse_ascanf_address(args[2], _ascanf_array, "ascanf_Set2Arrays", (int) ascanf_verbose, NULL ))
		|| array[0]->linkedArray.dataColumn
	){
		ascanf_emsg= " (invalid X array argument) ";
		ascanf_arg_error= 1;
	}
	if( !(array[1]= parse_ascanf_address(args[3], _ascanf_array, "ascanf_Set2Arrays", (int) ascanf_verbose, NULL ))
		|| array[1]->linkedArray.dataColumn
	){
		ascanf_emsg= " (invalid Y array argument) ";
		ascanf_arg_error= 1;
	}
	if( ascanf_arguments> 4 && ASCANF_TRUE(args[4]) ){
		if( !(array[2]= parse_ascanf_address(args[4], _ascanf_array, "ascanf_Set2Arrays", (int) ascanf_verbose, NULL ))
			|| array[2]->linkedArray.dataColumn
		){
			ascanf_emsg= " (invalid Error array argument ignored) ";
			array[2]= NULL;
		}
	}
	if( ascanf_arguments> 5 && ASCANF_TRUE(args[5]) ){
		if( !(array[3]= parse_ascanf_address(args[5], _ascanf_array, "ascanf_Set2Arrays", (int) ascanf_verbose, NULL ))
			|| array[3]->linkedArray.dataColumn
		){
			ascanf_emsg= " (invalid Length array argument ignored) ";
			array[3]= NULL;
		}
	}
#ifdef ADVANCED_STATS
	if( ascanf_arguments> 6 && ASCANF_TRUE(args[6]) ){
		if( !(array[4]= parse_ascanf_address(args[6], _ascanf_array, "ascanf_Set2Arrays", (int) ascanf_verbose, NULL ))
			|| array[4]->linkedArray.dataColumn
		){
			ascanf_emsg= " (invalid N array argument ignored) ";
			array[4]= NULL;
		}
	}
#else
	fprintf( StdErr, " (no N column/data available: recompile with -DADVANCED_STATS!) " );
	fflush( StdErr );
#endif

	if( ascanf_arguments> 7 ){
		if( !(visible= parse_ascanf_address(args[7], 0, "ascanf_Set2Arrays", (ascanf_verbose)? -1 : 0, NULL ))
			|| visible->type!= _ascanf_array
		){
/* 			CLIP(args[6], 0, set->numPoints );	*/
/* 			start= (int) args[7];	*/
			CLIP_EXPR(start, (int) args[7], 0, set->numPoints-1 );
		}
	}
	if( !visible ){
		if( start< 0 ){
			start= 0;
		}
		if( ascanf_arguments> 8 ){
	/* 		CLIP(args[8], -1, set->numPoints );	*/
	/* 		end= (int) args[8];	*/
			CLIP_EXPR(end, (int) args[8], -1, set->numPoints-1 );
		}
		if( end< 0 ){
			end= set->numPoints-1;
		}
		if( ascanf_arguments> 9 ){
	/* 		CLIP(args[9], 0, MAXINT );	*/
	/* 		offset= (int) args[9];	*/
			CLIP_EXPR(offset, (int) args[9], 0, MAXINT );
		}
		if( offset< 0 ){
			offset= 0;
		}
		if( ascanf_arguments > 10 && args[10]> 0 ){
		  double p;
			if( args[10]> 0 && (p= ssfloor(args[10]))< MAXINT ){
				pad= p;
			}
		}
		if( pad ){
			if( ascanf_arguments> 11 && !NaN(args[11]) ){
				pad_low= args[11];
				pl_set= 1;
			}
			else{
				pl_set= -1;
			}
			if( ascanf_arguments> 12 && !NaN(args[12]) ){
				pad_high= args[12];
				ph_set= 1;
			}
			else{
				ph_set= -1;
			}
		}
	}
	else{
		start= 0;
		end= set->numPoints- 1;
		if( ascanf_arguments> 8 && pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (<offset> and subsequent arguments ignored)== " );
		}
		if( ascanf_arguments> 8 ){
			use_set_visible= ASCANF_TRUE(args[8]);
		}
	}

	if( ascanf_arg_error ){
		return(0);
	}
	else if( ascanf_SyntaxCheck ){
		return(1);
	}

	if( visible ){
		N= 0;
		if( use_set_visible ){
			if( ActiveWin && ActiveWin!= StubWindow_ptr ){
				for( i= 0; i<= end; i++ ){
					N+= ActiveWin->pointVisible[(int) idx][i];
				}
				if( ActiveWin->numVisible[(int) idx]!= N ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (Set2Arrays[%d]: %d points visible according to numVisible, pointVisible says %d)== ",
							(int) idx, ActiveWin->numVisible[(int) idx], N
						);
					}
				}
			}
			else if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (Warning: no window active, so 0 points are taken to be visible)== " );
			}
			{ int n= MAX( N, 1 );
				Resize_ascanf_Array( visible, n, NULL );
			}
		}
		else{
			N= visible->N;
			for( i= 0; i< visible->N; i++ ){
			  double vidx= ASCANF_ARRAY_ELEM(visible,i);
				if( vidx< 0 || vidx>= set->numPoints ){
					N-= 1;
				}
			}
		}
	}
	else{
		N= offset+ end- start+ 1+ 2*pad;
	}

	if( visible && N && use_set_visible ){
	  signed char *pointVisible= (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin->pointVisible[(int) idx] : NULL;
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
		visible->value= ASCANF_ARRAY_ELEM(visible, visible->last_index);
		if( visible->accessHandler ){
			AccessHandler( visible, "Set2Arrays", level, ASCB_COMPILED, AH_EXPR, NULL   );
		}
	}

	for( col= 0; col< 5; col++ ){
#if ADVANCED_STATS==1
	  double *Ncolumn= NULL;
#endif
		targ= array[col];
		switch( col ){
			case 0:
				column= (cooked)? set->xvec : set->columns[ set->xcol ];
				break;
			case 1:
				column= (cooked)? set->yvec : set->columns[ set->ycol ];
				break;
			case 2:
				if( set->ecol>= 0 ){
					column= (cooked)? set->errvec : set->columns[set->ecol];
				}
				else{
					column= NULL;
				}
				break;
			case 3:
				if( set->lcol>= 0 ){
					column= (cooked)? set->lvec : set->columns[set->lcol];
				}
				else{
					column= NULL;
				}
				break;
			case 4:
#if ADVANCED_STATS==1
				if( (Ncolumn= (double*) calloc(set->numPoints, sizeof(double))) ){
					for( i= 0; i< set->numPoints; i++ ){
						Ncolumn[i]= NVAL(set,i);
					}
				}
				else{
					fprintf( StdErr, " (allocation failure for temp. N column (%s)) ", serror() );
					fflush( StdErr );
				}
				column= Ncolumn
#elif ADVANCED_STATS==2
				column= (set->Ncol>= 0)? set->columns[set->Ncol] : NULL;
#endif
				break;
			default:
				fprintf( StdErr, "%s:%d: unforeseen switch value %d\n",
					__FILE__, __LINE__, col
				);
				column= NULL;
				break;
		}
		if( targ && column ){
			if( pl_set== -1 ){
				pad_low= column[start];
			}
			if( ph_set== -1 ){
				pad_high= column[end];
			}
			{ int n= MAX( N, 1 );
				Resize_ascanf_Array( targ, n, NULL );
			}
			if( targ->iarray ){
				if( visible ){
					if( N ){
					  int j, vidx;
						start= (int) ASCANF_ARRAY_ELEM(visible,0);
						end= (int) ASCANF_ARRAY_ELEM(visible,N-1);
						if( !use_set_visible ){
							CLIP( start, 0, set->numPoints-1 );
							CLIP( end, 0, set->numPoints-1 );
						}
						if( visible->iarray ){
							for( j= i= 0; i< N; i++ ){
								vidx= visible->iarray[i];
								if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
									targ->iarray[j]= (int) column[vidx];
									j++;
								}
							}
						}
						else{
							for( j= i= 0; i< N; j++ ){
								vidx= (int) visible->array[i];
								if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
									targ->iarray[j]= (int) column[vidx];
									j++;
								}
							}
						}
						targ->last_index= N-1;
					}
					j= N;
				}
				else{
					targ->last_index= offset;
					for( j= 0; j< pad; j++, targ->last_index++ ){
						targ->value= targ->iarray[targ->last_index]= (int) pad_low;
					}
					for( j= 0, i= start; i<= end; j++, i++, targ->last_index++ ){
						targ->value= targ->iarray[targ->last_index]= (int) column[i];
					}
					for( j= 0; j< pad; j++, targ->last_index++ ){
						targ->value= targ->iarray[targ->last_index]= (int) pad_high;
					}
					targ->last_index-= 1;
				}
				targ->value= targ->iarray[targ->last_index];
			}
			else{
				if( visible ){
					if( N ){
					  int j, vidx;
						start= (int) ASCANF_ARRAY_ELEM(visible,0);
						end= (int) ASCANF_ARRAY_ELEM(visible,N-1);
						if( !use_set_visible ){
							CLIP( start, 0, set->numPoints-1 );
							CLIP( end, 0, set->numPoints-1 );
						}
						if( visible->iarray ){
							for( j= i= 0; i< N; i++ ){
								vidx= visible->iarray[i];
								if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
									targ->array[j]= column[vidx];
									j++;
								}
							}
						}
						else{
							for( j= i= 0; i< N; i++ ){
								vidx= (int) visible->array[i];
								if( use_set_visible || (vidx>=0 && vidx<set->numPoints) ){
									targ->array[j]= column[vidx];
									j++;
								}
							}
						}
						targ->last_index= N-1;
					}
					j= N;
				}
				else{
					targ->last_index= offset;
					for( j= 0; j< pad; j++, targ->last_index++ ){
						targ->value= targ->array[targ->last_index]= pad_low;
					}
					for( j= 0, i= start; i<= end; j++, i++, targ->last_index++ ){
						targ->value= targ->array[targ->last_index]= column[i];
					}
					for( j= 0; j< pad; j++, targ->last_index++ ){
						targ->value= targ->array[targ->last_index]= pad_high;
					}
					targ->last_index-= 1;
				}
				targ->value= targ->array[targ->last_index];
			}
			if( targ->accessHandler ){
				AccessHandler( targ, "Set2Arrays", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (copied %d(%d-%d) %selements from set#%d column %d to %s[%d]",
					j, start, end, (visible)? "visible " : "", set->set_nr, (int) col, targ->name, offset
				);
				if( pad ){
					fprintf( StdErr, " padded with %dx%s (low) and %dx%s (high)",
						pad, ad2str(pad_low, d3str_format,0), pad, ad2str( pad_high, d3str_format,0)
					);
				}
				fprintf( StdErr, " (%d elements))== ", targ->N );
			}
			*result+= targ->last_index- offset;
		}
	}
	return(1);
}

void* (*ascanf_array_malloc)(size_t size)= NULL;
void (*ascanf_array_free)(void *memory)= NULL;
int Resize_ascanf_Array_force= False;

/* Resize an _ascanf_array to the size requested. Returns 0 without any
 \ action taken when N<= 0.
 */
int Resize_ascanf_Array( ascanf_Function *af, int N, double *result )
{ int redo= False, i, was_scalar= False;
  void (*af_free)(void *mem)= NULL;
  int *iarray= NULL;
  double *array= NULL;
  double newSize;

	if( result ){
		*result= 0;
	}
	if( ascanf_SyntaxCheck || !af ){
		return(1);
	}
	if( af->type!= _ascanf_array ){
		if( af->type== _ascanf_variable ){
			was_scalar= True;
			af->type= _ascanf_array;
			af->iarray= NULL, af->array= NULL;
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (converting scalar \"%s\" to length %d array!) ",
					af->name, N
				);
				fflush(StdErr);
			}
		}
		else if( af->type== _ascanf_novariable && af->N< 0 ){
			fprintf( StdErr, " (warning: restoring deleted array \"%s\"!!) ", af->name );
			fflush(StdErr);
			af->type= _ascanf_array;
			af->N= 1;
			if( af->name[0]== '%' ){
				af->iarray= (int*) malloc( 1 * sizeof(int) );
			}
			else{
				af->array= (double*) malloc( 1 * sizeof(double) );
			}
		}
		else{
			return(0);
		}
	}
	// 20100502: don't raise an error on otherwise rejected array types if we won't do anything anyway:
	if( af->N != N || Resize_ascanf_Array_force ){
		if( af->sourceArray ){
			  /* 20050310: for the moment, just refuse to act on this sort of array. In a future release,
			   \ we might actually resize the source array by an appropriate/equivalent amount, if possible.
			   */
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (attempt to resize copy/subset array \"%s\" (source: \"%s\"))== ",
					af->name, af->sourceArray->name
				);
			}
			if( result ){
				*result= 0;
			}
			return(0);
		}
		else if( af->linkedArray.dataColumn ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (attempt to resize array \"%s\" linked to DataSet#%d[%d])== ",
					af->name, af->linkedArray.set_nr, af->linkedArray.col_nr
				);
			}
			Check_linkedArray(af);
			if( result ){
				*result= 0;
			}
			return(0);
		}
	}
	newSize= (af->iarray)? ((double)sizeof(int))*N : ((double)sizeof(double))*N;
	if( N< 0 || ((size_t)newSize)!=newSize || !af->car ){
		if( !af->car ){
			ascanf_emsg= " (attempt to resize static array) ";
			ascanf_arg_error= True;
		}
		if( ((size_t)newSize) != newSize ){
			fprintf( StdErr, " ((internal?) attempt to resize array \"%s\" to unsupported size %d (%lu!=%g))== ",
				af->name, N, ((size_t)newSize), newSize
			);
		}
		if( pragma_unlikely(ascanf_verbose) ){
			if( N<= 0 ){
				fprintf( StdErr, " ((internal?) attempt to resize array \"%s\" to size %d)== ", af->name, N );
			}
			if( !af->car ){
				fprintf( StdErr, " ((internal?) attempt to resize static array \"%s\" to size %d)== ", af->name, N );
			}
		}
		if( result ){
			*result= 0;
		}
		return(0);
	}
	if( ascanf_array_malloc || ascanf_array_free || af->free ){
		af_free= af->free;
		iarray= af->iarray;
		array= af->array;
		if( ascanf_array_malloc ){
			if( ascanf_array_malloc== (void*) -1 ){
				af->malloc= NULL;
				af->array= NULL;
				af->iarray= NULL;
			}
			else{
				af->malloc= ascanf_array_malloc;
			}
		}
		if( ascanf_array_free ){
			af->free= (ascanf_array_free== (void*) -1)? NULL : ascanf_array_free;
		}
		ascanf_array_malloc= NULL;
		ascanf_array_free= NULL;
	}
	do{
	  // 20090414: protection against N=0 arrays:
	  int M= (N>0)? N : 1;
		if( af->N!= M || Resize_ascanf_Array_force ){
			if( af->array && N< af->N ){
				for( i= N; i< af->N; i++ ){
				  int take_usage;
				  ascanf_Function *ef= parse_ascanf_address(af->array[i], 0, "Resize_ascanf_Array", (int) ascanf_verbose, &take_usage );
					if( ef && take_usage && ef->user_internal && ef->internal && ef->links> 0 ){
						ef->links-= 1;
						af->array[i]= 0;
					}
				}
			}
			if( af->malloc ){
				if( af->iarray || af->name[0]== '%' ){
					af->iarray= (*af->malloc)( M* sizeof(int) );
				}
				else{
					af->array= (*af->malloc)( M* sizeof(double) );
				}
			}
			else{
				if( af->iarray || af->name[0]== '%' ){
					af->iarray= (int*) XGrealloc( af->iarray, M* sizeof(int) );
				}
				else{
					af->array= (double*) XGrealloc( af->array, M* sizeof(double) );
				}
			}
		}
		if( pragma_unlikely(!af->array && !af->iarray) ){
			if( ascanf_verbose ){
				fprintf( StdErr, " ((re)allocation error for %d elements: %s) ",
					M, serror()
				);
			}
			else{
				fprintf( StdErr, " ((re)allocation error for %s:%d elements: %s) ",
					af->name, M, serror()
				);
			}
			ascanf_emsg= " (allocation error) ";
			ascanf_arg_error= 1;
			af->N= 0;
			N= 1;
			redo= ! redo;
		}
		else{
			redo= False;
		}
	} while( redo );
	if( af->iarray ){
		if( iarray ){
			memmove( af->iarray, iarray, (MIN(N,af->N))* sizeof(int) );
		}
		for( i= af->N; i< N; i++ ){
			af->iarray[i]= 0;
		}
		if( was_scalar ){
			af->iarray[0]= (int) af->value;
		}
	}
	else if( af->array ){
		if( array ){
			memmove( af->array, array, (MIN(N,af->N))* sizeof(double) );
		}
		for( i= af->N; i< N; i++ ){
			af->array[i]= 0;
		}
		if( was_scalar ){
			af->array[0]= af->value;
		}
	}
	af->N= N;
	if( af->last_index>= af->N ){
		af->last_index= af->N-1;
		af->value= ASCANF_ARRAY_ELEM(af, af->last_index);
	}
	if( af_free ){
		if( iarray && iarray!= af->iarray ){
			(af_free)(iarray);
		}
		if( array && array!= af->array ){
			(af_free)(array);
		}
	}
	else{
		if( iarray && iarray!= af->iarray ){
			xfree(iarray);
		}
		if( array && array!= af->array ){
			xfree(array);
		}
	}
	if( ascanf_arg_error ){
		if( result ){
			*result= 0;
		}
		N= 0;
	}
	else{
		if( result ){
			*result= N;
		}
	}
	Resize_ascanf_Array_force= False;
	return( (result)? (int) *result : N );
}

int ascanf_SetArraySize ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af;
  int N;
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	 /* 20050805: fixed stupid bug on next line: should always have been a || .... */
	if( !(af= parse_ascanf_address(args[0], 0, "ascanf_SetArraySize", (int) ascanf_verbose, NULL )) ||
		!(af->type== _ascanf_array || af->type== _ascanf_variable || (af->type==_ascanf_novariable && af->N<=0) )
	){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( args[1]< -1 || args[1]> MAXINT ){
		ascanf_emsg= " (size out of bounds) ";
		ascanf_arg_error= 1;
		return(0);
	}
	if( args[1]< 0 ){
		N= MAX( 1, af->N );
	}
	else{
		N= (int) MAX( 1, args[1] );
	}
	  /* 20000503: don't resize while compiling?!	*/
	if( !ascanf_SyntaxCheck || af->sourceArray ){
		if( af->sourceArray ){
			af->sourceArray= NULL;
			af->array= NULL;
			af->iarray= NULL;
			af->N= 0;
			xfree(af->usage);
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (array %s unlinked from original %s!)== ", af->name, af->sourceArray->name );
			}
		}
		Resize_ascanf_Array( af, N, result );
		if( ascanf_arguments> 2 ){
		  int i, aa= 2;
			for( i= 0; i< af->N; i++ ){
				if( af->iarray ){
				  double x;
					CLIP_EXPR( x, args[aa], -MAXINT, MAXINT );
					af->value= af->iarray[i]= x;
				}
				else{
					af->value= af->array[i]= args[aa];
				}
				if( aa< ascanf_arguments- 1){
					aa+= 1;
				}
			}
			af->last_index= af->N-1;
			af->value= ASCANF_ARRAY_ELEM(af,af->last_index);
			if( af->accessHandler ){
				AccessHandler( af, "SetArraySize", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
		}
	}
	return(1);
}

extern ascanf_Function vars_ascanf_Functions[], vars_internal_Functions[];
extern int ascanf_Functions, internal_Functions;

int ascanf_Arrays2Regular(void *malloc, void *free)
{ int i, n= 0;
  ascanf_Function *af;
	for( i= 0; i< ascanf_Functions; i++ ){
		af= &vars_ascanf_Functions[i];
		do{
			if( af->type== _ascanf_array ){
				if( (malloc && af->malloc== malloc) || (free && af->free== free) ){
				  double dum= 0;
					ascanf_array_malloc= (void*) -1;
					ascanf_array_free= (void*) -1;
					Resize_ascanf_Array_force= True;
					Resize_ascanf_Array( af, af->N, &dum );
					n+= 1;
				}
			}
			af= af->cdr;
		} while( af );
	}
	for( i= 0; i< internal_Functions; i++ ){
		af= &vars_internal_Functions[i];
		do{
			if( af->type== _ascanf_array ){
				if( (malloc && af->malloc== malloc) || (free && af->free== free) ){
				  double dum= 0;
					ascanf_array_malloc= (void*) -1;
					ascanf_array_free= (void*) -1;
					Resize_ascanf_Array_force= True;
					Resize_ascanf_Array( af, af->N, &dum );
					n+= 1;
				}
			}
			af= af->cdr;
		} while( af );
	}
	return(n);
}

int AutoCrop_ascanf_Array( ascanf_Function *af, ascanf_Function *mask, int *level, void *form, char *expr, char *caller )
{ unsigned int i, N= af->N, R= 0;
  double dum;
	if( !af ){
		return(0);
	}
	if( mask ){
		Resize_ascanf_Array( mask, af->N, &dum );
		if( mask->iarray ){
			for( i= 0; i< mask->N; i++ ){
				mask->iarray[i]= 1;
			}
		}
		else{
			for( i= 0; i< mask->N; i++ ){
				mask->array[i]= 1;
			}
		}
	}
	if( af->iarray ){
	  int v0= af->iarray[0];
		if( af->N> 1 && v0== af->iarray[1] ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (auto-cropping \"%s\"", FUNNAME(af) );
			}
			  /* to decrease N correctly, we have to start at i==0 even though a[i]==v0 ... */
			for( i= 0; i< af->N && af->iarray[i]== v0; i++ ){
				N-= 1;
				if( mask ){
					ASCANF_ARRAY_ELEM_SET(mask,i,0);
					mask->last_index= i;
				}
			}
			if( i> 0 ){
				R= i;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, ": cropped upto [%d]==%s != [%d]==%s",
						i, ad2str( af->iarray[i], d3str_format, NULL),
						0, ad2str( af->iarray[0], d3str_format, NULL)
					);
				}
				memmove( af->iarray, &af->iarray[i], N* sizeof(af->iarray[0]) );
			}
		}
		{ int M= N, j= af->N-1;
			i= N-1;
			if( (v0= af->iarray[i]) == af->iarray[i-1] ){
				for( ; i>= 0 && af->iarray[i]== v0; i--, j-- ){
					N-= 1;
					if( mask ){
						ASCANF_ARRAY_ELEM_SET(mask,j,0);
						mask->last_index= j;
					}
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, ": cropped from [%d]==%s != [%d]==%s",
						i, ad2str( af->iarray[i], d3str_format, NULL),
						M-1, ad2str( af->iarray[M-1], d3str_format, NULL)
					);
				}
			}
			if( !N ){
				N= 1;
			}
		}
	}
	else{
	  double v0= af->array[0];
		if( af->N> 1 && v0== af->array[1] ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (auto-cropping \"%s\"", FUNNAME(af) );
			}
			  /* to decrease N correctly, we have to start at i==0 even though a[i]==v0 ... */
			for( i= 0; i< af->N && af->array[i]== v0; i++ ){
				N-= 1;
				if( mask ){
					ASCANF_ARRAY_ELEM_SET(mask,i,0);
					mask->last_index= i;
				}
			}
			if( i> 0 ){
				R= i;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, ": cropped upto [%d]==%s != [%d]==%s",
						i, ad2str( af->array[i], d3str_format, NULL),
						0, ad2str( af->array[0], d3str_format, NULL)
					);
				}
				memmove( af->array, &af->array[i], N* sizeof(af->array[0]) );
			}
		}
		{ int M= N, j= af->N-1;
			i= N-1;
			if( (v0= af->array[i]) == af->array[i-1] ){
				for( ; i>= 0 && af->array[i]== v0; i--, j-- ){
					N-= 1;
					if( mask ){
						ASCANF_ARRAY_ELEM_SET(mask,j,0);
						mask->last_index= j;
					}
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, ": cropped from [%d]==%s != [%d]==%s",
						i, ad2str( af->array[i], d3str_format, NULL),
						M-1, ad2str( af->array[M-1], d3str_format, NULL)
					);
				}
			}
			if( !N ){
				N= 1;
			}
		}
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fputs( ") ", StdErr );
	}
	if( mask ){
		mask->value= ASCANF_ARRAY_ELEM(mask, mask->last_index);
		if( mask->accessHandler ){
			AccessHandler( mask, caller, level, form, expr, NULL  );
		}
	}
	if( N!= af->N ){
		Resize_ascanf_Array( af, N, &dum );
		af->last_index= -1;
		af->value= N;
		if( af->accessHandler ){
			AccessHandler( af, caller, level, form, expr, NULL  );
		}
	}
	return(R);
}

int ascanf_AutoCropArray0 ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af;
  int idx= 0, n= 0;
	*result= 0;
	for( idx= 0; idx< ascanf_arguments; idx++ ){
		if( !(af= parse_ascanf_address(args[idx], _ascanf_array, "ascanf_AutoCropArray", (int) ascanf_verbose, NULL )) ){
			fprintf( StdErr, " (ignoring non-array argument %d=%s) ", idx, ad2str( args[idx], d3str_format, 0) );
		}
		else{
			n= AutoCrop_ascanf_Array( af, NULL, level, ASCB_COMPILED, AH_EXPR, "AutoCropArray" );
		}
	}
	*result= n;
	return(1);
}

int ascanf_AutoCropArray ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af, *mask;
  int idx= 0, n= 0;
	*result= 0;
	for( idx= 0; idx< ascanf_arguments; idx+= 2 ){
		if( !(
				(af= parse_ascanf_address(args[idx], _ascanf_array, "ascanf_AutoCropArray", (int) ascanf_verbose, NULL ))
			)
		){
			fprintf( StdErr, " (ignoring non-array argument %d=%s [mask %d=%s]) ",
				idx, ad2str( args[idx], d3str_format, 0),
				idx+1, ad2str( args[idx+1], d3str_format, 0)
			);
		}
		else{
			if( (mask= parse_ascanf_address(args[idx+1], _ascanf_array, "ascanf_AutoCropArray", (int) ascanf_verbose, NULL )) ){
				n+= AutoCrop_ascanf_Array( af, mask, level, ASCB_COMPILED, AH_EXPR, "AutoCropArray" );
			}
			else{
				n= AutoCrop_ascanf_Array( af, NULL, level, ASCB_COMPILED, AH_EXPR, "AutoCropArray" );
			}
		}
	}
	*result= n;
	return(1);
}

int ascanf_FixAutoArray ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af;
  int idx= 0;
	*result= 0;
	for( idx= 0; idx< ascanf_arguments; idx++ ){
		if( !(af= parse_ascanf_address(args[idx], _ascanf_array, "ascanf_FixAutoArray", (int) ascanf_verbose, NULL )) ){
			fprintf( StdErr, " (ignoring non-array argument %d=%s) ", idx, ad2str( args[idx], d3str_format, 0) );
		}
		else{
			if( af->procedure ){
				  /* 20020826: destroy the updater-procedure, but don't automatically deleted internal variables that
				   \ have their linkcount reduced to 0 because of this!
				   \ (And we should set that count to 1 just to prevent others from doing the same -- adios GC!!)
				   */
				_Destroy_Form( &af->procedure, False );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (%s (arg.%d) won't change anymore) ", ad2str( args[idx], d3str_format, 0), idx );
				}
			}
		}
	}
	if( ascanf_arguments== 1 ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return(1);
}

  /* Array quicksort routines. The comparison of the 2 elements is done
   \ by a user-provided procedure pointed to by qs_compar. Since we cannot
   \ yet pass parameters to procedures, the 2 elements have to be stored
   \ in 2 global variables, pointed to by qs_ptr1 and qs_ptr2. We thus
   \ need 2 internal routines, QSort_CompaInts and QSort_ComparDoubles that
   \ store the 2 elements to be compared in the provided variables, call
   \ the provided comparison procedure, and return the determined value
   \ to the qsort() routine.
   */

ascanf_Function *qs_compar, *qs_ptr1, *qs_ptr2;
double qs_comparisons;
Compiled_Form *qsc_form= NULL;
#ifdef ASCANF_ALTERNATE
ascanf_Callback_Frame *qsc_frame= NULL;
#endif

int QSort_ComparInts( const void *a, const void *b )
{ int n= 1;
  double comp= 0;
  double *lArgList= af_ArgList->array, args[2];
  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList;
  int level= 1;
	args[0]= *((int*) a);
	if( qs_ptr1 ){
		qs_ptr1->value= args[0];
		qs_ptr1->assigns+= 1;
		if( qs_ptr1->accessHandler ){
			AccessHandler( qs_ptr1, "QSort_ComparInts", &level, qsc_form, "<QSort_ComparInts>", NULL );
		}
	}
	args[1]= *((int*) b);
	if( qs_ptr2 ){
		qs_ptr2->value= args[1];
		qs_ptr2->assigns+= 1;
		if( qs_ptr2->accessHandler ){
			AccessHandler( qs_ptr2, "QSort_ComparInts", &level, qsc_form, "<QSort_ComparInts>", NULL  );
		}
	}
	switch( qs_compar->type ){
		case _ascanf_procedure:
			SET_AF_ARGLIST( args, 2 );
			ascanf_update_ArgList= False;
			evaluate_procedure( &n, qs_compar, &comp, &level );
			SET_AF_ARGLIST( lArgList, lArgc );
			ascanf_update_ArgList= auA;
			break;
		case NOT_EOF:
		case NOT_EOF_OR_RETURN:
		case _ascanf_function:{
		  double *fargs= qsc_frame->args;
			qsc_frame->args= args;
			qsc_frame->self= qs_compar;
			(*qs_compar->function)( qsc_frame );
			comp= *(qsc_frame->result);
			n= 1;
			qsc_frame->args= fargs;
			break;
		}
	}
	if( n== 1 ){
		qs_compar->value= comp;
		qs_compar->assigns+= 1;
		if( qs_compar->accessHandler ){
			AccessHandler( qs_compar, qs_compar->name, &level, qsc_form, "<QSort_ComparInts>", NULL  );
		}
		qs_comparisons+= 1;
	}
	else{
		comp= 0;
	}
	if( comp< 0 ){
		return(-1);
	}
	else if( comp> 0 ){
		return(1);
	}
	else{
		return(0);
	}
}

int QSort_ComparDoubles( const void *a, const void *b )
{ int n= 1;
  double comp= 0;
  double *lArgList= af_ArgList->array, args[2];
  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList;
  int level= 1;
	args[0]= *((double*) a);
	if( qs_ptr1 ){
		qs_ptr1->value= args[0];
		qs_ptr1->assigns+= 1;
		if( qs_ptr1->accessHandler ){
			AccessHandler( qs_ptr1, "QSort_ComparDoubles", &level, qsc_form, "<QSort_ComparDoubles>", NULL );
		}
	}
	args[1]= *((double*) b);
	if( qs_ptr2 ){
		qs_ptr2->value= args[1];
		qs_ptr2->assigns+= 1;
		if( qs_ptr2->accessHandler ){
			AccessHandler( qs_ptr2, "QSort_ComparDoubles", &level, qsc_form, "<QSort_ComparDoubles>", NULL  );
		}
	}
	switch( qs_compar->type ){
		case _ascanf_procedure:
			SET_AF_ARGLIST( args, 2 );
			ascanf_update_ArgList= False;
			evaluate_procedure( &n, qs_compar, &comp, &level );
			SET_AF_ARGLIST( lArgList, lArgc );
			ascanf_update_ArgList= auA;
			break;
		case NOT_EOF:
		case NOT_EOF_OR_RETURN:
		case _ascanf_function:{
		  double *fargs= qsc_frame->args;
			qsc_frame->args= args;
			qsc_frame->self= qs_compar;
			(*qs_compar->function)( qsc_frame );
			comp= *(qsc_frame->result);
			n= 1;
			qsc_frame->args= fargs;
			break;
		}
	}
	if( n== 1 ){
		qs_compar->value= comp;
		qs_compar->assigns+= 1;
		if( qs_compar->accessHandler ){
			AccessHandler( qs_compar, qs_compar->name, &level, qsc_form, "<QSort_ComparDoubles>", NULL  );
		}
		qs_comparisons+= 1;
	}
	else{
		comp= 0;
	}
	if( comp< 0 ){
		return(-1);
	}
	else if( comp> 0 ){
		return(1);
	}
	else{
		return(0);
	}
}

int ascanf_QSortArray ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *array;
  Compiled_Form *qparpar= NULL;
#ifdef ASCANF_ALTERNATE
  ascanf_Function *self= __ascb_frame->self;
#endif
	*result= 0;
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	if( !(array= parse_ascanf_address(args[0], _ascanf_array, "ascanf_QSortArray", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
	}
	if( !(qs_compar= parse_ascanf_address(args[1], 0, "ascanf_QSortArray", (int) ascanf_verbose, NULL ))
		&& !(
			qs_compar->type== _ascanf_procedure ||
				(qs_compar->function && (qs_compar->type== NOT_EOF || qs_compar->type== NOT_EOF_OR_RETURN
						|| qs_compar->type== _ascanf_function))
		)
	){
		ascanf_emsg= " (invalid compare-procedure/function argument (2)) ";
		ascanf_arg_error= 1;
	}
	if( ascanf_arguments> 2 ){
		if( !(qs_ptr1= parse_ascanf_address(args[2], _ascanf_variable, "ascanf_QSortArray", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid compare-value a argument (3)) ";
			ascanf_arg_error= 1;
		}
	}
	else{
		qs_ptr1= NULL;
	}
	if( ascanf_arguments> 3 ){
		if( !(qs_ptr2= parse_ascanf_address(args[3], _ascanf_variable, "ascanf_QSortArray", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid compare-value b argument (4)) ";
			ascanf_arg_error= 1;
		}
	}
	else{
		qs_ptr2= NULL;
	}
	if( !ascanf_SyntaxCheck && array->N && !ascanf_arg_error ){
#ifdef ASCANF_ALTERNATE
	  int aarg= ascanf_arguments;
#endif
		if( (qsc_form= ASCB_COMPILED) ){
			if( qs_compar->procedure && qs_compar->procedure->parent && __ascb_frame->compiled && __ascb_frame->compiled->parent ){
				qparpar= qs_compar->procedure->parent->parent;
				qs_compar->procedure->parent->parent= __ascb_frame->compiled->parent->parent;
			}
		}
#ifdef ASCANF_ALTERNATE
		qsc_frame= __ascb_frame;
		*(qsc_frame->level)+= 1;
#endif
		qs_comparisons= 0;
		if( array->iarray ){
			qsort( array->iarray, array->N, sizeof(int), QSort_ComparInts );
		}
		else{
			qsort( array->array, array->N, sizeof(double), QSort_ComparDoubles );
		}
		array->last_index= array->N-1;
		array->value= ASCANF_ARRAY_ELEM(array,array->last_index);
		if( array->accessHandler ){
			AccessHandler( array, "QSortArray", level, ASCB_COMPILED, AH_EXPR, NULL  );
		}
		*result= qs_comparisons;
		if( qparpar ){
			qs_compar->procedure->parent->parent= qparpar;
		}
#ifdef ASCANF_ALTERNATE
		*(qsc_frame->level)-= 1;
		qsc_frame->self= self;
		ascanf_arguments= aarg;
#endif
	}
	return( !ascanf_arg_error );
}

int ascanf_UnwrapAngles ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af;
  int i, N, w= 0;
  double radix, radix_2;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( !(af= parse_ascanf_address(args[0], _ascanf_array, "ascanf_UnwrapAngles", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
		*result= -1;
		return(0);
	}
	if( ascanf_arguments== 1 ){
		radix= M_2PI;
	}
	else{
		radix= (args[1])? args[1] : M_2PI;
	}
	radix_2= radix/ 2;
	N= af->N;
	if( ascanf_SyntaxCheck ){
		*result= 0;
		return(1);
	}
	if( af->iarray ){
	  int aa= 0, da= 0, ca, pa;
		aa= pa= af->iarray[0];
		for( i= 1; i< af->N; i++ ){
			if( (da= (ca= af->iarray[i])- pa) ){
				if( fabs(da)< radix_2 ){
					aa+= da;
				}
				else{
					aa+= conv_angle_(da, radix);
					w+= 1;
				}
			}
			af->value= af->iarray[i]= aa;
			pa= ca;
		}
	}
	else{
	  double aa= 0, da= 0, ca, pa;
		aa= pa= af->array[0];
		for( i= 1; i< af->N; i++ ){
			if( (da= (ca= af->array[i])- pa)!= 0 ){
				if( fabs(da)< radix_2 ){
					aa+= da;
				}
				else{
					aa+= conv_angle_(da, radix);
					w+= 1;
				}
			}
			if( NaNorInf(da) ){
				set_NaN(af->value);
				set_NaN(af->array[i]);
			}
			else{
				af->value= af->array[i]= aa;
			}
			pa= ca;
		}
	}
	af->last_index= af->N-1;
	af->value= ASCANF_ARRAY_ELEM(af,af->last_index);
	if( af->accessHandler ){
		AccessHandler( af, "UnwrapAngles", level, ASCB_COMPILED, AH_EXPR, NULL );
	}
	*result= w;
	return(1);
}

/* (r)fftw and ascanf_convolve functions were here */

int ascanf_AccessHandler ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *Var= NULL, *Targ= NULL;
	*result= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	else{
		if( !(Var= parse_ascanf_address(args[0], 0, "ascanf_AccessHandler", (int) ascanf_verbose, NULL )) ||
			(Var->name[0]== '$' && Var->dollar_variable== 0)
		){
			ascanf_emsg= " (invalid variable object argument (1): incorrect type, unexisting or static ($ var)) ";
			ascanf_arg_error= 1;
			return(0);
		}
		else{
		  double par[2];
			*result= take_ascanf_address( Var->accessHandler );
			if( Var->accessHandler ){
				Var->accessHandler->links-= 1;
			}
			if( ascanf_arguments> 2 ){
				par[0]= args[2];
			}
			else{
				set_NaN( par[0] );
			}
			if( ascanf_arguments> 3 ){
				par[1]= args[3];
			}
			else{
				set_NaN( par[1] );
			}
			if( ascanf_arguments> 5 ){
				CLIP_EXPR_CAST( int, Var->aH_flags[2], double, args[4], -MAXINT, MAXINT );
				Var->aH_par[2]= args[5];
			}
			else{
				Var->aH_flags[2]= False;
				Var->aH_par[2]= 0;
			}
			if( ascanf_arguments> 6 ){
				Var->aH_flags[0]= (args[6])? True : False;
			}
			else{
				Var->aH_flags[0]= False;
			}
			if( ascanf_arguments> 7 ){
				Var->aH_flags[1]= (args[7])? True : False;
			}
			else{
				Var->aH_flags[1]= False;
			}
			if( ascanf_arguments< 2 ||
				!(Targ= parse_ascanf_address(args[1], 0, "ascanf_AccessHandler", (int) ascanf_verbose, NULL ))
			){
				Var->accessHandler= NULL;
			}
			else{
				Var->accessHandler= Targ;
				Targ->links+= 1;
				  /* 991004: using NaN to get at the default makes it impossible to pass NaN's to
				   \ the handler!
				   */
				switch( Targ->type ){
					case _ascanf_variable:
						Var->aH_par[0]= /* (NaN(par[0]))? 1 : */ par[0];
						set_NaN(Var->aH_par[1]);
						break;
					case _ascanf_array:
						Var->aH_par[0]= /* (NaN(par[0]))? -1 : */ par[0];
						Var->aH_par[1]= /* (NaN(par[1]))? 1 : */ par[1];
						break;
					case _ascanf_procedure:
						Var->aH_par[0]= /* (NaN(par[0]))? 1 : */ par[0];
						Var->aH_par[1]= /* (NaN(par[1]))? 1 : */ par[1];
						break;
					case NOT_EOF:
					case NOT_EOF_OR_RETURN:
					case _ascanf_function:
						Var->aH_par[0]= /* (NaN(par[0]))? 1 : */ par[0];
						Var->aH_par[1]= /* (NaN(par[1]))? 0 : */ par[1];
						break;
				}
			}
			Var->old_value= Var->value;
			if( pragma_unlikely(ascanf_verbose) ){
				if( Targ ){
					fprintf( StdErr, " (AccessHandler[%s,%s], par=%s,%s,%d:%s, dump=%d,change=%d old_value=%s)== ",
						Var->name, Targ->name,
						ad2str( Var->aH_par[0], d3str_format, 0),
						ad2str( Var->aH_par[1], d3str_format, 0),
						Var->aH_flags[2],
						ad2str( Var->aH_par[2], d3str_format, 0),
						Var->aH_flags[0], Var->aH_flags[1],
						ad2str( Var->old_value, d3str_format, 0)
					);
				}
				else{
					fprintf( StdErr, " (no AccessHandler for %s)== ", Var->name );
				}
			}
/* 			*result= Var->value;	*/
		}
	}
	return(1);
}

ascanf_Function *qs_compar_val= NULL;

typedef struct QS_Compar_Set_Val{
	int pnt_nr;
	double value;
} QS_Compar_Set_Val;

void QSort_ComparSetValue( int set_nr, QS_Compar_Set_Val *val )
{ int n= 1;
  double *lArgList= af_ArgList->array, args[2];
  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList;
  int level= 1;
	args[0]= set_nr;
	args[1]= val->pnt_nr;
	SET_AF_ARGLIST( args, 2 );
	ascanf_update_ArgList= False;
	evaluate_procedure( &n, qs_compar_val, &(val->value), &level );
	SET_AF_ARGLIST( lArgList, lArgc );
	ascanf_update_ArgList= auA;
	if( n== 1 ){
		qs_compar_val->value= val->value;
		qs_compar_val->assigns+= 1;
		if( qs_compar_val->accessHandler ){
			AccessHandler( qs_compar_val, qs_compar_val->name, &level, qsc_form, "<QSort_ComparSetValue>", NULL  );
		}
		qs_comparisons+= 1;
	}
	else{
		set_NaN( val->value );
	}
}

int QSort_ComparSetValues( const void *a, const void *b )
{
	qs_comparisons+= 1;
	return( ((QS_Compar_Set_Val*) a)->value - ((QS_Compar_Set_Val*) b)->value );
}

static int qs_compar_setNr= 0;

int ascanf_QSortSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  Compiled_Form *qparpar= NULL;
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	else{
	  DataSet *this_set;
	  int N;
		if( args[0]< 0 || args[0]>= setNumber ){
			ascanf_emsg= " (invalid set number) ";
			ascanf_arg_error= 1;
			return(0);
		}
		else{
			this_set= &AllSets[ (qs_compar_setNr= (int) args[0]) ];
			N= this_set->numPoints;
		}
		if( !(qs_compar_val= parse_ascanf_address(args[1], _ascanf_procedure, "ascanf_QSortSet", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid comparison value argument (1)) ";
			ascanf_arg_error= 1;
			return(0);
		}
		if( !ascanf_SyntaxCheck && N ){
		  int i, col, ns= 0;
		  QS_Compar_Set_Val *valtab;
		  extern double **XGrealloc_2d_doubles( double **cur_columns, int ncols, int nlines, int cur_ncols, int cur_nlines, char *caller );
		  extern void XGfree_2d_doubles( double ***columns, int ncols, int nlines );
		  double **columns= NULL;
			if( qs_compar_val->procedure->parent && __ascb_frame->compiled && __ascb_frame->compiled->parent ){
				qparpar= qs_compar_val->procedure->parent->parent;
				qs_compar_val->procedure->parent->parent= __ascb_frame->compiled->parent->parent;
			}
			columns= XGrealloc_2d_doubles( columns, this_set->ncols, N, 0, 0, "ascanf_QSortSet" );
			valtab = (QS_Compar_Set_Val*) malloc( N * sizeof(QS_Compar_Set_Val) );
			if( valtab && columns ){
				  /* First, initialise an array associating point number with the value to sort on.
				   \ That value is to be determined by the qs_compar_val procedure.
				   */
				for( i= 0; i< N; i++ ){
					valtab[i].pnt_nr= i;
					QSort_ComparSetValue( qs_compar_setNr, &valtab[i] );
				}
				  /* Now, do the qsort on that internal table:	*/
				qs_comparisons= 0;
				qsort( valtab, N, sizeof(QS_Compar_Set_Val), QSort_ComparSetValues );
				  /* valtab is now sorted in order of ascending sorting value; we return the number
				   \ of operations necessary for this:
				   */
				*result= qs_comparisons;
				  /* I don't see any other option if I don't want to self-implement a special
				   \ qsort algorithm: make a temporary copy of the current dataset (just
				   \ the 2D data array).
				   */
				for( i= 0; i< N; i++ ){
					if( i!= valtab[i].pnt_nr ){
						for( col= 0; col< this_set->ncols; col++ ){
							columns[col][i]= this_set->columns[col][i];
						}
					}
				}
				  /* Now copy, from that temp. dataset, the values (all columns) where they ought
				   \ to go according to the sorting value:
				   */
				for( i= 0; i< N; i++ ){
					if( i!= valtab[i].pnt_nr ){
						for( col= 0; col< this_set->ncols; col++ ){
							this_set->columns[col][i]= columns[col][ valtab[i].pnt_nr ];
						}
						ns+= 1;
					}
				}
				XGfree_2d_doubles( &columns, this_set->ncols, N );
				xfree(valtab);
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (%d points exchanged) ", ns );
				}
			}
			else{
				ascanf_emsg= " (can't get memory) ";
				ascanf_arg_error= 1;
			}
			GCA();
			if( qparpar ){
				qs_compar_val->procedure->parent->parent= qparpar;
			}
		}
	}
	return(1);
}

int QSort_ComparSetValue2( int *a, int *b )
{ int n= 1;
  double *lArgList= af_ArgList->array, args[3], value;
  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList;
  int level= 1;
	args[0]= qs_compar_setNr;
	args[1]= *a;
	args[2]= *b;
	SET_AF_ARGLIST( args, 3 );
	ascanf_update_ArgList= False;
	evaluate_procedure( &n, qs_compar_val, &value, &level );
	SET_AF_ARGLIST( lArgList, lArgc );
	ascanf_update_ArgList= auA;
	if( n== 1 ){
		qs_compar_val->value= value;
		qs_compar_val->assigns+= 1;
		if( qs_compar_val->accessHandler ){
			AccessHandler( qs_compar_val, qs_compar_val->name, &level, qsc_form, "<QSort_ComparSetValue2>", NULL  );
		}
		qs_comparisons+= 1;
		return( value );
	}
	else{
		return(0);
	}
}

int ascanf_QSortSet2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int qs_compar_setNr= 0;
  Compiled_Form *qparpar= NULL;
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	else{
	  DataSet *this_set;
	  int N;
		if( args[0]< 0 || args[0]>= setNumber ){
			ascanf_emsg= " (invalid set number) ";
			ascanf_arg_error= 1;
			return(0);
		}
		else{
			this_set= &AllSets[ (qs_compar_setNr= (int) args[0]) ];
			N= this_set->numPoints;
		}
		if( !(qs_compar_val= parse_ascanf_address(args[1], _ascanf_procedure, "ascanf_QSortSet", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid comparison value argument (1)) ";
			ascanf_arg_error= 1;
			return(0);
		}
		if( !ascanf_SyntaxCheck && N ){
		  int i, col, ns= 0;
		  int *valtab;
		  extern double **XGrealloc_2d_doubles( double **cur_columns, int ncols, int nlines, int cur_ncols, int cur_nlines, char *caller );
		  extern void XGfree_2d_doubles( double ***columns, int ncols, int nlines );
		  double **columns= NULL;
			if( (qsc_form= ASCB_COMPILED) ){
				if( qs_compar_val->procedure->parent && __ascb_frame->compiled && __ascb_frame->compiled->parent ){
					qparpar= qs_compar->procedure->parent->parent;
					qs_compar_val->procedure->parent->parent= __ascb_frame->compiled->parent->parent;
				}
			}
			columns= XGrealloc_2d_doubles( columns, this_set->ncols, N, 0, 0, "ascanf_QSortSet" );
			valtab = (int*) malloc( N * sizeof(int) );
			if( valtab && columns ){
				  /* First, initialise an array associating point number with the index value to sort on.
				   \ The sorting value is to be determined by the qs_compar_val procedure.
				   */
				for( i= 0; i< N; i++ ){
					valtab[i]= i;
				}
				  /* Now, do the qsort on that internal table:	*/
				qs_comparisons= 0;
				qsort( valtab, N, sizeof(int), (void*) QSort_ComparSetValue2 );
				  /* valtab is now sorted to represent the desired order of the set; we return the number
				   \ of operations necessary for this:
				   */
				*result= qs_comparisons;
				  /* I don't see any other option if I don't want to self-implement a special
				   \ qsort algorithm: make a temporary copy of the current dataset (just
				   \ the 2D data array).
				   */
				for( i= 0; i< N; i++ ){
					if( i!= valtab[i] ){
						for( col= 0; col< this_set->ncols; col++ ){
							columns[col][i]= this_set->columns[col][i];
						}
					}
				}
				  /* Now copy, from that temp. dataset, the values (all columns) where they ought
				   \ to go according to the sorting value:
				   */
				for( i= 0; i< N; i++ ){
					if( i!= valtab[i] ){
						for( col= 0; col< this_set->ncols; col++ ){
							this_set->columns[col][i]= columns[col][ valtab[i] ];
						}
						ns+= 1;
					}
				}
				XGfree_2d_doubles( &columns, this_set->ncols, N );
				xfree( valtab );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (%d points exchanged) ", ns );
				}
			}
			else{
				ascanf_emsg= " (can't get memory) ";
				ascanf_arg_error= 1;
			}
			GCA();
			if( qparpar ){
				qs_compar_val->procedure->parent->parent= qparpar;
			}
		}
	}
	return(1);
}

int ascanf_Find_Point ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *SNR, *PNR, *X= NULL, *Y= NULL;
  double x, y, xprec, yprec;
	*result= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return(0);
	}
	else{
		x= args[0];
		y= args[1];
		SNR= parse_ascanf_address(args[2], _ascanf_variable, "ascanf_Find_Point", False, NULL );
		PNR= parse_ascanf_address(args[3], _ascanf_variable, "ascanf_Find_Point", False, NULL );
		if( !SNR || !PNR ){
			ascanf_emsg= " (missing set and/or pointnr. pointer(s)) ";
			ascanf_arg_error= 1;
			return(0);
		}
		if( ascanf_arguments> 3 ){
			X= parse_ascanf_address(args[4], _ascanf_variable, "ascanf_Find_Point", False, NULL );
		}
		if( ascanf_arguments> 4 ){
			Y= parse_ascanf_address(args[5], _ascanf_variable, "ascanf_Find_Point", False, NULL );
		}
		if( ActiveWin && ActiveWin!= StubWindow_ptr && !ascanf_SyntaxCheck ){
		  DataSet *this_set= NULL;
		  double xx= x, yy= y;
		  int av= ascanf_verbose;
			ascanf_verbose= 0;
			if( CheckProcessUpdate( ActiveWin, True, True, False )> 0 ){
				RedrawNow( ActiveWin );
			}
			ascanf_verbose= av;
			set_Find_Point_precision( ActiveWin->win_geo.R_XUnitsPerPixel/2.0, ActiveWin->win_geo.R_YUnitsPerPixel/2.0, &xprec, &yprec );
			*result= Find_Point( ActiveWin, &x, &y, &this_set, False, NULL, False, False, !NaNorInf(x), !NaNorInf(y) );
			set_Find_Point_precision( xprec, yprec, NULL, NULL );
			if( this_set ){
				SNR->value= this_set->set_nr;
			}
			if( *result>= 0 ){
				PNR->value= *result;
				*result= 1;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (ActiveWin 0x%lx, nearest to (%s,%s) is set #%d, pnt #%d = (%s,%s)) ",
						ActiveWin, d2str(xx, 0,0), d2str( yy, 0,0),
						this_set->set_nr, (int) *result,
						d2str( x, 0,0), d2str( y, 0,0)
					);
				}
			}
			else{
				*result= 0;
			}
			if( X ){
				X->value= x;
			}
			if( Y ){
				Y->value= y;
			}
		}
	}
	return(1);
}

#ifdef linux
/* For linux, we need to define _REGEX_RE_COMP in order to get the declarations we want.... */
#	define _REGEX_RE_COMP
#endif
#include <regex.h>

int ascanf_fopen ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *fp, *fname=NULL, *mode=NULL;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(fp= parse_ascanf_address(args[0], 0, "ascanf_fopen", (int) ascanf_verbose, NULL )) ||
			(fp->type== _ascanf_function)
		){
			fprintf( StdErr, " (error: 1st argument is not a valid filepointer or filename)== " );
			fp= NULL;
		}
		if( !(fname= parse_ascanf_address(args[1], 0, "ascanf_fopen", (int) ascanf_verbose, NULL )) ||
			(fname->type== _ascanf_procedure || fname->type== _ascanf_function || !fname->usage)
		){
			fprintf( StdErr, " (error: 2nd argument is not a valid filename or mode string)== " );
			fname= NULL;
		}
		if( ascanf_arguments> 2 ){
			if( !(mode= parse_ascanf_address(args[2], 0, "ascanf_fopen", (int) ascanf_verbose, NULL )) ||
				(mode->type== _ascanf_procedure || mode->type== _ascanf_function || !mode->usage)
			){
				fprintf( StdErr, " (error: 3rd argument is not a valid mode string)== " );
				mode= NULL;
			}
		}
		if( !mode ){
		  /* revert to initial calling convention: a filename in the first var (also returned fp)
		   \ and the mode string in the 2nd var:
		   */
			if( fp && !fp->usage ){
				fprintf( StdErr, " (error: 1st argument is not a valid filename)==" );
				fp= NULL;
			}
			mode= fname;
			  /* fname and fp are the same variable: */
			fname= fp;
		}

		if( fp && fname && mode && fname->usage && mode->usage ){
			if( fp->fp && fp->fp!= NullDevice && !ascanf_SyntaxCheck ){
				if( fp->fp_is_pipe ){
					delete_FILEsDescriptor(fp->fp);
					pclose( fp->fp );
					fp->fp= NULL;
				}
/* 				else{	*/
/* 					fclose( fp->fp );	*/
/* 				}	*/
			}
			if( ascanf_SyntaxCheck ){
				  /* Don't open files while compiling!	*/
				fp->fp= NullDevice;
				fp->fp_is_pipe= False;
			}
			else{
			  char *fp_mode= NULL;
				errno= 0;
				if( fname->usage[0]== '|' ){
					if( (fp->fp= popen( &fname->usage[1], mode->usage)) ){
						*result= fp->value= args[0];
						fp->fp_is_pipe= True;
						fp_mode= concat( "p", mode->usage, NULL );
						register_FILEsDescriptor(fp->fp);
					}
				}
				else{
					if( fp->fp ){
						  /* 20020826: if this variable had an open file associated with it, use
						   \ freopen() to change the target. This should ensure that the pointer
						   \ itself remains unchanged, which is handy to prevent having stale
						   \ pointers around (ex: $Dprint-file was set to &fp earlier...)
						   */
						delete_FILEsDescriptor(fp->fp);
						if( (fp->fp= register_FILEsDescriptor( freopen( fname->usage, mode->usage, fp->fp )) ) ){
							*result= fp->value= args[0];
							fp->fp_is_pipe= False;
						}
					}
					else{
						if( (fp->fp= register_FILEsDescriptor( fopen( fname->usage, mode->usage)) ) ){
							*result= fp->value= args[0];
							fp->fp_is_pipe= False;
						}
					}
					fp_mode= XGstrdup( mode->usage );
				}
				if( !fp->fp ){
					fprintf( StdErr, " (could not open \"%s\": %s) ", fname->usage, serror() );
					ascanf_arg_error= 1;
					*result= 0;
					xfree(fp_mode);
				}
				else{
					if( errno ){
						fprintf( StdErr, " (error opening \"%s\": %s) ", fname->usage, serror() );
					}
					if( fp!= fname ){
						xfree(fp->usage);
						fp->usage= strdup(fname->usage);
					}
					if( strcmp( fp->fp_mode, fp_mode ) ){
						xfree( fp->fp_mode );
						fp->fp_mode= fp_mode;
					}
					else{
						xfree(fp_mode);
					}
				}
			}
		}
		else{
			*result= 0;
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_fclose ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *fname;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(fname= parse_ascanf_address(args[0], 0, "ascanf_fclose", (int) ascanf_verbose, NULL )) ||
			(fname->type== _ascanf_function || !fname->usage)
		){
			fprintf( StdErr, " (error: 1st argument is not a valid filename)== " );
			fname= NULL;
		}
		if( fname && fname!= ascanf_XGOutput && fname->usage ){
			if( fname->fp ){
				if( fname->fp!= NullDevice ){
				  extern ascanf_Function *ascanf_Dprint_fp;
				  extern FILE *Dprint_fp;
					delete_FILEsDescriptor(fname->fp);
					if( (fname->fp_is_pipe && (*result= pclose(fname->fp))) || (*result= fclose( fname->fp )) ){
						fprintf( StdErr, " (error closing \"%s\": %s) ",
							fname->usage, serror()
						);
					}
					if( ascanf_Dprint_fp->fp== fname->fp || Dprint_fp== fname->fp ){
						ascanf_Dprint_fp->value= 0;
						AccessHandler( ascanf_Dprint_fp, "ascanf_fclose", ASCB_LEVEL, ASCB_COMPILED, NULL, NULL );
					}
				}
				fname->fp= NULL;
				fname->fp_is_pipe= False;
				xfree(fname->fp_mode);
			}
			else{
				*result= -1;
				fprintf( StdErr, " (error: \"%s\" was not open) ", fname->usage );
				ascanf_arg_error= 1;
			}
		}
		else{
			*result= -1;
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_fflush ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *fname;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
	  int r= 0;
		if( (r+= fflush( stdout )) ){
			fprintf( StdErr, " (error flushing \"stdout\": %s) ",
				serror()
			);
		}
		if( (r+= fflush( stderr )) ){
			fprintf( StdErr, " (error flushing \"stderr\": %s) ",
				serror()
			);
		}
		*result= r;
	}
	else{
	  FILE *fp;
	  char *name;
		if( !(fname= parse_ascanf_address(args[0], 0, "ascanf_fflush", (int) ascanf_verbose, NULL )) ){
			switch( (int) args[0] ){
				case 0:
					fp= stdin;
					name= "stdin";
					break;
				case 1:
					fp= stdout;
					name= "stdout";
					break;
				case 2:
					fp= StdErr;
					name= "stderr";
					break;
			}
		}
		else if( fname->type!= _ascanf_function && fname->usage ){
			fp= fname->fp;
			name= fname->usage;
		}
		if( fp ){
			if( (*result= fflush( fp )) ){
				fprintf( StdErr, " (error flushing \"%s\": %s) ",
					name, serror()
				);
			}
		}
		else{
			*result= -1;
			ascanf_arg_error= 1;
			if( fname ){
				fprintf( StdErr, " (error: \"%s\" is not open) ", name );
			}
			else{
				fprintf( StdErr, " (error: 1st argument is not a valid filename/number)== " );
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_unlink ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *fname=NULL;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(fname= parse_ascanf_address(args[0], 0, "ascanf_fopen", (int) ascanf_verbose, NULL )) ||
			(fname->type== _ascanf_procedure || fname->type== _ascanf_function || !fname->usage)
		){
			fprintf( StdErr, " (error: 1st argument is not a valid filename or mode string)== " );
			fname= NULL;
		}

		if( fname && fname->usage && !ascanf_SyntaxCheck ){
			*result= unlink( fname->usage );
			if( *result ){
				fprintf( StdErr, " (error unlinking \"%s\": %s) ", fname->usage, serror() );
			}
		}
		else{
			*result= 0;
		}
	}
	return(!ascanf_arg_error);
}

/* routine for the "scanf[x,y,...]" ascanf syntax	*/
int ascanf_scanf ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *Source, *form, *af, *Offset;
  FILE *fp= NULL;
  int offset= 0, targ_start= 2, take_usage= 0, i, arg, argc, Source_len= 0;
  char *format;
  double **arglist;
	*result= 0;
	ascanf_arg_error= 0;
	if( args && ascanf_arguments>= 3 ){
		if( !(Source= parse_ascanf_address(args[0], 0, "ascanf_scanf", False, &take_usage)) ){
			switch( (int) args[0] ){
				case 0:
					fp= stdin;
					break;
				case 1:
					fp= stdout;
					break;
				default:
					fp= StdErr;
					break;
			}
			if( pragma_unlikely(ascanf_verbose) || ascanf_SyntaxCheck ){
				if( args[0]!= 0 && args[0]!= 1 && args[0]!= 2 ){
					fprintf( StdErr, " (caution: filedescriptor %s accepted for StdErr for the time being) ",
						ad2str( args[0],d3str_format,0)
					);
				}
				else if( ascanf_verbose ){
					fprintf( StdErr, " (reading from std file) " );
				}
			}
		}
		else if( Source->fp ){
			fp= Source->fp;
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (reading from file %s) ", Source->usage );
			}
			take_usage= False;
			Source= NULL;
		}
		else if( !take_usage ){
			ascanf_emsg= " (1st argument must be a stringpointer, or 1 or 2 to select between stdout/StdErr) ";
			ascanf_arg_error= 1;
			return(0);
		}
		else if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (reading from string %s) ", Source->name );
		}
		if( !(form= parse_ascanf_address(args[1], _ascanf_variable, "ascanf_scanf", False, &take_usage)) || !take_usage ){
			ascanf_emsg= " (2nd argument must be a stringpointer) ";
			ascanf_arg_error= 1;
			return(0);
		}
		format= (form && form->usage)? form->usage : NULL;
		if( !format ){
			ascanf_emsg= " (error: empty format specifier) ";
			ascanf_arg_error= 1;
			return(0);
		}
		if( ascanf_arguments> 3 ){
			if( (Offset= parse_ascanf_address(args[2], 0, "ascanf_scanf", False, &take_usage)) ){
				if( !take_usage ){
					Offset= NULL;
				}
				else if( !Offset->usage ){
					ascanf_emsg= " (error: empty offset string specifier) ";
					ascanf_arg_error= 1;
					return(0);
				}
				else{
					targ_start+= 1;
				}
			}
			else{
				CLIP_EXPR( offset, (int) args[2], 0, MAXINT );
				targ_start+= 1;
			}
		}
		else{
			Offset= NULL;
		}
		arglist= (double**) calloc( ascanf_arguments- targ_start, sizeof(double*) );
		arg= 0;
		argc= ascanf_arguments- targ_start;
		for( i= targ_start; i< ascanf_arguments && arglist && !ascanf_arg_error; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_variable, "ascanf_scanf", False, &take_usage)) && !take_usage ){
				  /* A "scalar pointer". No need to check the type of this one.	*/
				arglist[arg]= &af->value;
				arg+= 1;
			}
			else{
				ascanf_emsg= " (pointer to unsupported type) ";
				ascanf_arg_error= 1;
			}
		}
		{
		  int n= 0;
		  char *c= format;
			while( (c= strstr(c, "%lf")) ){
				if( c== format || c[-1]!= '%' ){
					n+= 1;
				}
				c+= 3;
			}
			if( arg!= n || ((ascanf_verbose || ascanf_SyntaxCheck) && !*Allocate_Internal<= 0) ){
				fprintf( StdErr,
					" (%d valid ascanf arguments from %d passed, %sformat string has %d valid %%lf fields) ",
					arg, argc, (arg!= n)? "ERROR: " : "", n
				);
				if( arg!= n ){
#if defined(ASCANF_FRAME_EXPR)
					PF_print_string( StdErr, "\n", "\n", __ascb_frame->expr, False );
#else
					if( __ascb_frame->compiled && __ascb_frame->compiled->expr ){
						PF_print_string( StdErr, "\n", "\n", __ascb_frame->compiled->expr, False );
					}
#endif
					ascanf_emsg= " (ascanf[] number of format fields does not match the number of arguments passed) ";
					ascanf_arg_error= True;
				}
				fflush( StdErr );
			}
		}
		if( !ascanf_SyntaxCheck && !ascanf_arg_error ){
			if( Source ){
				if( Source->usage ){
				  char *source;
					Source_len= strlen(Source->usage);
					if( offset ){
						if( offset>= Source_len ){
							source= NULL;
						}
						else{
							source= &Source->usage[offset];
						}
					}
					else if( Offset ){
						if( (source= strstr( Source->usage, Offset->usage )) ){
							  /* We should point to after the specified pattern	*/
							source+= strlen(Offset->usage);
							if( source- Source->usage>= Source_len ){
								source= NULL;
							}
						}
					}
					else{
						source= Source->usage;
					}
					if( source ){
#ifdef HAVE_VSCANF
#	ifndef __x86_64__
						*result= vsscanf( source, format, arglist );
#	else
						*result= asscanf( source, format, arglist );
#	endif
#else
						ascanf_emsg= " (required vsscanf(2) routine not available on this system?!) ";
						ascanf_arg_error= 1;
						*result= -1;
#endif
					}
				}
			}
			else{
				if( offset ){
				  int n= 0;
					while( n< offset && !feof(fp) && !ferror(fp) ){
						fgetc( fp );
					}
				}
				else if( Offset ){
				  int offlen= strlen(Offset->usage);
				  char *lbuf;
				  char *d;
				  int ok= False;
					if( (lbuf = (char*) malloc( offlen * sizeof(char) )) ){
						do{
							if( (d= fgets( lbuf, offlen, fp)) ){
								ok= !strncmp( lbuf, Offset->usage, offlen);
							}
						} while( d && !feof(fp) && !ferror(fp) && !ok );
						xfree(lbuf);
					}
					  /* No need to get to the end of the pattern: the filepointer is already positioned there!	*/
				}
				if( !feof(fp) && !ferror(fp) ){
#ifdef HAVE_VSCANF
					*result= vfscanf( fp, format, arglist );
#else
					ascanf_emsg= " (required vfscanf(2) routine not available on this system?!) ";
					ascanf_arg_error= 1;
					*result= -1;
#endif
				}
				if( feof(fp) || ferror(fp) ){
					fprintf( StdErr, " (error reading from file: %s) ", serror() );
					if( ascanf_verbose ){
						fputs( "== ", StdErr );
					}
				}
				GCA();
			}
		}
		xfree( arglist );
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(0);
	}
}

int ascanf_getenv ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *s1;
  char *name, *value;
  int eval= True;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_getenv", (int) ascanf_verbose, NULL )) ||
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || (!s1->usage && ascanf_verbose) )
		){
			name= ad2str( args[0], d3str_format, 0 );
		}
		else{
			name= s1->usage;
		}
		if( ascanf_arguments> 1 && ASCANF_FALSE(args[1]) ){
			eval= False;
		}
		if( name ){
			if( (value= getenv(name)) ){
			  double v;
			  int n= 1;
				if( eval ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (%s=\"%s\"==", name, value );
					}
					if( __fascanf( &n, value, &v, NULL, NULL, NULL, NULL, level, NULL)> 0 ){
						*result= v;
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "%s[%d])== ", ad2str(v, d3str_format, NULL), n );
						}
					}
					else{
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "<parsing failed:returning string>)== " );
						}
						eval= False;
					}
				}
				if( !eval ){
				  ascanf_Function *allocated;
					allocated= Create_Internal_ascanfString( value, level );
					if( allocated ){
						  /* We must now make sure that the just created internal variable behaves as a user variable: */
						allocated->user_internal= True;
						allocated->internal= True;
#if 0
						allocated->is_address= allocated->take_address= True;
						allocated->is_usage= allocated->take_usage= True;
#endif
						*result= take_ascanf_address( allocated );
					}
					else{
						fprintf( StdErr, " (error: could not duplicate env.var %s=\"%s\": %s)== ", name, value, serror() );
						ascanf_arg_error= True;
						set_NaN(*result);
					}
				}
			}
			else{
				if( ascanf_arguments > 2 ){
					*result = args[2];
				}
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_setenv ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *s1, *s2= NULL;
  char *name, *value= NULL;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_getenv", (int) ascanf_verbose, NULL )) ||
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || (!s1->usage && ascanf_verbose) )
		){
			name= ad2str( args[0], d3str_format, 0 );
		}
		else{
			name= s1->usage;
		}
		if( ascanf_arguments> 1 && args[1] ){
			if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_getenv", (int) ascanf_verbose, NULL )) ||
				(s2->type== _ascanf_procedure || s2->type== _ascanf_function || (!s2->usage && ascanf_verbose) )
			){
				value= ad2str( args[1], d3str_format, 0 );
			}
			else{
				value= s2->usage;
			}
		}
		if( name ){
			if( (value= setenv(name, value)) ){
			  ascanf_Function *allocated;
				allocated= Create_Internal_ascanfString( value, level );
				if( allocated ){
					  /* We must now make sure that the just created internal variable behaves as a user variable: */
					allocated->user_internal= True;
					allocated->internal= True;
					*result= take_ascanf_address( allocated );
				}
				else{
					fprintf( StdErr, " (error: could not duplicate new/updated env.var %s=\"%s\": %s)== ", name, value, serror() );
					ascanf_arg_error= True;
					set_NaN(*result);
				}
			}
		}
	}
	return(!ascanf_arg_error);
}

int ascanf_CustomFont ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *rcf, *xfn, *axfn, *psfn;
  int pssize, psreencode= 1, take_usage;
  struct CustomFont *cf;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || (ascanf_arguments< 4 && ascanf_arguments!= 1) ){
		ascanf_arg_error= 1;
	}
	else{
		if( (rcf= parse_ascanf_address(args[0], _ascanf_variable, "ascanf_CustomFont", (int) ascanf_verbose, NULL )) ){
			cf= rcf->cfont;
		}
		else{
			ascanf_emsg= " (1st argument must be a variable pointer) ";
		}
		if( ascanf_arguments== 1 ){
			if( rcf && cf ){
				Free_CustomFont( cf );
				xfree( cf );
				rcf->cfont= NULL;
				*result= 0;
			}
			return(1);
		}
		if( (xfn= parse_ascanf_address(args[1], 0, "ascanf_CustomFont", (int) ascanf_verbose, &take_usage )) && !take_usage ){
			xfn= NULL;
		}
		else{
			ascanf_emsg= " (2nd argument must be a stringpointer) ";
		}
		if( (psfn= parse_ascanf_address(args[2], 0, "ascanf_CustomFont", (int) ascanf_verbose, &take_usage )) && !take_usage ){
			psfn= NULL;
		}
		else{
			ascanf_emsg= " (3rd argument must be a stringpointer) ";
		}
		CLIP_EXPR_CAST( int, pssize, double, args[3], 0, MAXINT );
		if( ascanf_arguments> 4 ){
			if( (axfn= parse_ascanf_address(args[4], 0, "ascanf_CustomFont", (int) ascanf_verbose, NULL )) && !take_usage ){
				axfn= NULL;
			}
			else{
				psreencode= (args[4]==0)? False : True;
			}
		}
		if( ascanf_arguments> 5 ){
			if( (axfn= parse_ascanf_address(args[5], 0, "ascanf_CustomFont", (int) ascanf_verbose, NULL )) && !take_usage ){
				axfn= NULL;
			}
		}
		if( rcf && xfn && xfn->usage && psfn && psfn->usage ){
		  extern struct CustomFont *Init_CustomFont( char*, char*, char*, double, int);
			if( cf ){
				Free_CustomFont( cf );
				xfree( cf );
			}
			if( (cf= Init_CustomFont( xfn->usage, (axfn)? axfn->usage : NULL, psfn->usage, pssize, psreencode )) ){
				rcf->cfont= cf;
				*result= take_ascanf_address( rcf );
			}
		}
		else{
			ascanf_arg_error= True;
		}
	}
	return(!ascanf_arg_error);
}


int ascanf_Apply2Array ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
#ifdef ASCANF_ALTERNATE
  ascanf_Function *source, *target= NULL, *fun;
  double loop_address;
  int r, arg= 2, pass_element= 1;
  static int Stride= 1, reset_stride= False;
	*result= -1;
	if( !args ){
		ascanf_arg_error= 1;
		r= 0;
	}
	else if( ascanf_arguments== 1 || ascanf_arguments== 2){
		if( (Stride= (int) Entier(args[0]))== 0 ){
			Stride= 1;
		}
		if( ascanf_arguments== 2 && ASCANF_TRUE(args[1]) ){
			reset_stride= True;
		}
		else{
			reset_stride= False;
		}
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (next %s will use stride==%d) ", (reset_stride)? "call (only)" : "calls", Stride );
		}
	}
	else if( ascanf_arguments>= 3 ){
	  extern ascanf_Function *af_loop_counter;
		ascanf_arg_error= False;
		if( !(source= parse_ascanf_address( args[0], _ascanf_array, "ascanf_Apply2Array", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (missing source array!) ";
			ascanf_arg_error= True;
		}
		if( args[1] ){
			if( args[1]< 0 ){
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (Apply2Array[%s,...] storing results in source array!) ", ad2str( args[0], d3str_format, 0 ) );
					fflush( StdErr );
				}
				target= source;
			}
			else{
				if( !(target= parse_ascanf_address( args[1], _ascanf_array, "ascanf_Apply2Array", (int) ascanf_verbose, NULL )) ){
					ascanf_emsg= " (invalid target array!) ";
					ascanf_arg_error= True;
				}
				else if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
					if( target->N!= source->N ){
						Resize_ascanf_Array( target, source->N, result );
					}
				}
			}
		}
		if( !(fun= parse_ascanf_address( args[arg], 0, "ascanf_Apply2Array", (int) ascanf_verbose, NULL )) ){
			pass_element= (ASCANF_TRUE(args[arg]))? 1 : 0;
			arg+= 1;
			fun= parse_ascanf_address( args[arg], 0, "ascanf_Apply2Array", (int) ascanf_verbose, NULL );
		}
		if( fun ){
			loop_address= take_ascanf_address(af_loop_counter);
			switch( fun->type ){
				case _ascanf_procedure:
				case NOT_EOF:
				case NOT_EOF_OR_RETURN:
				case _ascanf_function:
					if( pragma_unlikely(ascanf_verbose) && strncmp(fun->name, "\\l\\expr-",8)== 0 ){
						fprintf( StdErr, "\n#%s%d\\l\\\tlambda[", (__ascb_frame->compiled)? "C" : "", *__ascb_frame->level );
						Print_Form( StdErr, &fun->procedure, 0, True, NULL, "#\\l\\\t", NULL, False );
						fputs( "]\n", StdErr );
					}
					break;
				case _ascanf_array:
					if( ascanf_arguments> 4 ){
						fprintf( StdErr, " (warning: method is an array; arguments beyond the 4th are ignored) " );
					}
					if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
						if( fun->N!= source->N ){
							fprintf( StdErr, " (warning: method is an array of size not equal to the source array (%d!=%d): resizing) ",
								fun->N, source->N
							);
							Resize_ascanf_Array( fun, source->N, result );
						}
					}
					break;
				case _ascanf_variable:
					if( fun->own_address == loop_address || &(fun->value)== ascanf_loop_counter ){
						break;
					}
					/* else fall through to reject */
				default:
					ascanf_emsg= " (unsupported method (3rd argument) in Apply2Array) ";
					fun= NULL;
					break;
			}
		}
		  /* 20050412: don't do the real work when compiling?! */
		if( !ascanf_arg_error && fun && !ascanf_SyntaxCheck ){
		  int argc= ascanf_arguments;
		  double value;
		  double *lArgList= af_ArgList->array;
		  double alc= *ascanf_loop_counter;
		  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList, *level= __ascb_frame->level, extra_args, has_lcntr= False;
#ifdef ASCB_FRAME_EXPRESSION
		  char *expr= __ascb_frame->expr;
#else
		  char lexpr[128]= "Apply2Array[<source-array>,<target-array>,<procedure>,...]";
		  char *expr= (AH_EXPR)? AH_EXPR : lexpr;
#endif
		  double *largs;
#ifdef DEBUG
		  largs= (double*) calloc( argc, sizeof(double) );
#else
		  largs= (double*) malloc( argc * sizeof(double) );
#endif
		  int stride= Stride;

			if( !largs ){
				ascanf_emsg = " (memory error) ";
				ascanf_arg_error = 1;
				return 0;
			}

			  /* 20031023: from here on, <arg> will index the 1st 'extra argument' (to be
			   \ passed to the method.)
			   */
			arg+= 1;
			if( ascanf_arguments ){
			  int i, j;
				extra_args= 0;
				loop_address= take_ascanf_address(af_loop_counter);
				for( j= pass_element, i= arg; i< ascanf_arguments; i++, j++ ){
					if( args[i]== loop_address ){
						largs[j]= *ascanf_loop_counter;
						has_lcntr+= 1;
					}
					else{
						largs[j]= args[i];
					}
					extra_args++;
				}
			}
			  /* There remain ascanf_arguments-(arg+1) arguments, but we also pass the current element, so
			   \ decrement ascanf_arguments by 2:
			ascanf_arguments-= 2;
			   */
			  /* 20031023: ascanf_arguments is the number of extra arguments, plus 1 if we pass
			   \ the element's value as 1st argument.
			   */
			ascanf_arguments= (pass_element)? extra_args+1 : extra_args;
		{ int i, ii, strd= ABS(stride);
			i= (stride>0)? 0 : source->N-1;
			  /* use a seperate counter which can run from 0,<source->N, incremented with |stride|, to simplify the test
			   \ in the while. This might be a tad faster.
			   */
			while( ((stride>0)? i< source->N : i>=0) && !ascanf_arg_error )
/* 			for( ii= 0; ii< source->N && !ascanf_arg_error; ii+= strd )	*/
			{
				  /* an up-to-date ascanf_loop_counter ($loop) is only available in procedures! */
				*ascanf_loop_counter= i;
				if( pass_element ){
					largs[0]= (source->iarray)? source->iarray[i] : source->array[i];
				}
				  /* check if &$loop was given somewhere, and update the value: */
				if( has_lcntr ){
				  int j, k;
					for( j= pass_element, k= arg; j< ascanf_arguments; j++, k++ ){
						if( args[k]== loop_address ){
							largs[j]= *ascanf_loop_counter;
							has_lcntr+= 1;
						}
					}
				}
				switch( fun->type ){
					case NOT_EOF:
					case NOT_EOF_OR_RETURN:
					case _ascanf_function:
						if( pragma_likely(fun->function) ){
							__ascb_frame->args= (ascanf_arguments)? largs : NULL;
							r= (*fun->function)( __ascb_frame );
							__ascb_frame->args= args;
							value= *result;
						}
						break;
					case _ascanf_procedure:{
					  int ok= False;
						if( ascanf_arguments ){
							SET_AF_ARGLIST( largs, ascanf_arguments );
						}
						else{
							SET_AF_ARGLIST( ascanf_ArgList, 0 );
						}
						ascanf_update_ArgList= False;
						fun->procedure->level+= 1;
						if( pragma_likely(
							call_compiled_ascanf_function( 0, &expr, &value, &ok, "ascanf_Apply2Array", &fun->procedure, level)
						) ){
							fun->value= fun->procedure->value= value;
							ok= True;
							if( pragma_unlikely(ascanf_verbose) ){
								Print_ProcedureCall( StdErr, fun, level );
								fprintf( StdErr, "== %s\n", ad2str(value, d3str_format, NULL) );
							}
						}
						fun->procedure->level-= 1;
						if( ok && fun->accessHandler ){
							AccessHandler( fun, fun->name, level, ASCB_COMPILED, NULL, NULL );
						}
						SET_AF_ARGLIST( lArgList, lArgc );
						ascanf_update_ArgList= auA;
						GCA();
						break;
					}
					  /* 20020614: beats me... on my PIII, gcc2.95.3, the _ascanf_array: case has to be last. Whatever
					   \ else I try, performance drops by a factor of up to 2!!
					   \ This already happens when specifying -O (instead of no optim.)
					   \ Using the same compiler, this thing does not happen on an SGI O2/R5000...
					   */
					case _ascanf_array:{
						switch( ascanf_arguments ){
							case 0:
							case 1:
								value= (fun->iarray)? fun->iarray[i] : fun->array[i];
								fun->last_index= i;
								break;
							default:
								if( pragma_unlikely(ascanf_verbose) ){
									fprintf( StdErr, "%s[%d,%s]== ", fun->name, (int) largs[0], ad2str(largs[1], d3str_format,0) );
								}
								_ascanf_array_access( __ascb_frame->compiled, fun, 0, (int) largs[0], 2, largs, &value, level );
								break;
						}
						break;
					}
					case _ascanf_variable:
						  /* 20050506: should only happen when &$loop was specified as the to-be-applied function! */
						value= fun->value;
						break;
					default:
						  /* shouldn't get here! */
						ascanf_emsg= " (unsupported type in ascanf_Apply2Array) ";
						ascanf_arg_error= 1;
						break;
				}
				if( pragma_likely( target && !ascanf_arg_error ) ){
					if( target->iarray ){
						target->value= target->iarray[i]= (int) value;
					}
					else{
						target->value= target->array[i]= value;
					}
					target->last_index= i;
  /* 20050506: not sure why I chose to set target->value in the following manner: */
/* 					target->value= ASCANF_ARRAY_ELEM(target,target->last_index);	*/
					if( target->accessHandler ){
						AccessHandler( target, "Apply2Array", level, ASCB_COMPILED, AH_EXPR, NULL );
					}
				}

				i+= stride;

			}
		}
			if( fun->type== _ascanf_function ){
				*result= ascanf_progn_return;
			}
			else{
				*result= value;
			}
			xfree( largs );
			ascanf_arguments= argc;
			*ascanf_loop_counter= alc;
		}
		if( reset_stride ){
			Stride= 1;
			reset_stride= False;
		}
	}
	else{
		ascanf_arg_error= 1;
		r= 0;
	}
	return(r);
#else
	fprintf( StdErr, "ascanf_Apply2Array(): not implemented for this model of callback parameter passing -- #define ASCANF_ALTERNATE!!\n" );
	ascanf_emsg= " (functionality not available) ";
	ascanf_arg_error= 1;
	return(0);
#endif
}

int ascanf_name ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *s1, *af;
  static ascanf_Function AF= {NULL};
  static char *AFname= "name-Static-StringPointer";
	af= &AF;
	if( AF.name ){
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
	}
	else{
		af->usage= NULL;
	}
	af->name= AFname;
	af->type= _ascanf_variable;
	af->is_address= af->take_address= True;
	af->internal= True;
	af->is_usage= af->take_usage= True;

	set_NaN(*result);
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_name", (int) ascanf_verbose, NULL )) ){
			fprintf( StdErr, " (warning: argument is not a valid pointer)== " );
		}
		xfree( af->usage );
		if( s1 && s1->name ){
			af->usage= strdup( s1->name );
		}
		else{
			af->usage= strdup( ad2str( args[0], d3str_format, 0 ) );
		}
		*result= take_ascanf_address(af);
	}
	return(!ascanf_arg_error);
}

#include <sys/param.h>

char *getFileName( char *path )
{ static char fn[MAXPATHLEN];
	memset( fn, 0, sizeof(fn) );
	if( readlink( path, fn, sizeof(fn) )>= 0 ){
		fn[sizeof(fn)-1]= '\0';
		return(fn);
	}
	else{
		fprintf( StdErr, "getFileName(): can't obtain the target of link (?!) \"%s\" (%s)\n", path, serror() );
		return(NULL);
	}
}

char *get_fdName( int fd )
{ char *path, pid[64], fno[64], *c;
	snprintf( pid, sizeof(pid), "%u", (unsigned int) getpid() );
	snprintf( fno, sizeof(fno), "%u", (unsigned int) fd );
	if( (path= concat( "/proc/", pid, "/fd/", fno, NULL )) ){
		c= getFileName(path);
		xfree(path);
		return(c);
	}
	else{
		fprintf( StdErr, "%s::get_fdName(%s): can't construct /proc/%s/fd/%s pathname (%s)\n", __FILE__,
			pid, fno, serror()
		);
		c= NULL;
	}
	return(c);
}

#include <sys/stat.h>

int ascanf_fname ( ASCB_ARGLIST )
{
#if defined(linux) || defined(__CYGWIN__)
  ASCB_FRAME_SHORT
  int fd= 0;
  ascanf_Function *targ, *af;
  static ascanf_Function AF= {NULL};
  static char *AFname= "fname-Static-StringPointer";
	af= &AF;
	if( AF.name ){
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
	}
	else{
		af->usage= NULL;
	}
	af->type= _ascanf_variable;
	af->name= AFname;
	af->is_address= af->take_address= True;
	af->internal= True;
	af->is_usage= af->take_usage= True;

	set_NaN(*result);
	ascanf_arg_error= 0;
	if( args && ascanf_arguments>= 1 ){
		if( !(targ= parse_ascanf_address(args[0], 0, "ascanf_fname", False, NULL)) ){
			fd= (int) args[0];
			if( fd== 2 ){
				fd= fileno(StdErr);
			}
		}
		else if( targ->fp ){
			fd= fileno(targ->fp);
		}
		else{
			ascanf_emsg= " (1st argument must be a filepointer, or a file descriptor ([0,1,...])) ";
			ascanf_arg_error= 1;
			if( !ascanf_SyntaxCheck ){
				return(0);
			}
		}
	}
	{ char *c= get_fdName( fd );
		if( c ){
			af->usage= strdup( c );
			*result= take_ascanf_address(af);
		}
	}
#else
	fprintf( StdErr, "%s::ascanf_fname(): functionality not available on this platform :(\n", __FILE__ );
#endif
	return(1);
}

int ascanf_fsize ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int take_usage;
  char *fn= NULL, active= 0;
  ascanf_Function *targ= NULL;
  double arg0;

	set_NaN(*result);
	ascanf_arg_error= 0;
	if( args && ascanf_arguments>= 1 ){
		arg0= args[0];
		while( active<= 1 && !targ ){
			active+= 1;
			if( (targ= parse_ascanf_address(args[0], 0, "ascanf_fname", False, &take_usage)) ){
				if( take_usage ){
					fn= targ->usage;
				}
				else{
					if( targ->fp ){
						if( active== 1 ){
							fn= targ->usage;
							arg0= ascanf_fname(ASCB_ARGUMENTS);
						}
						targ= NULL;
					}
					else if( !fn ){
						ascanf_emsg= " (argument must be a stringpointer or a filepointer) ";
						ascanf_arg_error= 1;
					}
				}
			}
		}
		if( fn ){
		  struct stat st;
			if( stat( fn, &st) ){
				fn= NULL;
			}
			else{
				if( !S_ISREG(st.st_mode) && !S_ISLNK(st.st_mode) && !S_ISFIFO(st.st_mode) ){
					fn= NULL;
				}
			}
		}
		if( fn ){
		  int lines= 0;
		  char *command= NULL;
		  FILE *fp;
			if( targ && targ->fp ){
				fflush(targ->fp);
			}
			if( ascanf_arguments>= 2 ){
				lines= (ASCANF_TRUE(args[1]))? 1 : 0;
			}
			if( lines ){
				command= concat( "sync ; wc -l < ", fn, NULL );
			}
			else{
				command= concat( "sync ; wc -c < ", fn, NULL );
			}
			if( (fp= popen(command,"r")) ){
				fscanf( fp, "%lf", result );
				pclose(fp);
			}
			xfree( command );
		}
	}
	else{
		ascanf_arg_error= 1;
	}
	return(1);
}

/* Given arrays xa[1..n] and ya[1..n], and given a value x, this routine returns a value y, and
 \ an error estimate dy. If P(x) is the polynomial of degree N-1 such that P(xa[i]) = ya[i]; i =
 \ 1,...,n, then the returned value y = P(x).
 */
void Polynom_Interpolate(double *xa, double *ya, int n, double x, double *y, double *dy)
{ int i,m,ns=1, n1= n+1;
  double den,dif,dift,ho,hp,w;
  double *c, *d;

	// do only a single allocation, and point d to halfway into c:
	if( (c = (double*) malloc( 2 * n1 * sizeof(double) )) ){
		d = &c[n1];
		dif=fabs(x-xa[1]);
		for( i= 1; i<= n; i++ ){
			if( (dift=fabs(x-xa[i])) < dif ){
				ns=i;
				dif=dift;
			}
			c[i]=ya[i];
			d[i]=ya[i];
		}
		*y=ya[ns--];
		for( m= 1; m< n; m++ ){
			for( i= 1; i<= n-m; i++ ){
				ho=xa[i]-x;
				hp=xa[i+m]-x;
				w=c[i+1]-d[i];
				if( (den=ho-hp) == 0.0 ){
					fprintf( StdErr, "Polynom_Interpolate(): error: xa[%d]=%s ==xa[%d]=%s\n",
						i, d2str(xa[i],0,0), i+m, d2str(xa[i+m],0,0)
					);
				}
				den=w/den;
				d[i]=hp*den;
				c[i]=ho*den;
			}
			*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
		}
		xfree(c);
	}
}


int ascanf_ArrayFind ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *af, *af_low, *af_high;
  double x, tolerance;
  int tolerance_set= False;
	*result= 0;
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
	}
	if( !(af= parse_ascanf_address(args[0], _ascanf_array, "ascanf_ArrayFind", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid array argument (1)) ";
		ascanf_arg_error= 1;
	}
	x= args[1];
	if( !(af_low= parse_ascanf_address(args[2], _ascanf_variable, "ascanf_ArrayFind", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid low element argument (3)) ";
		ascanf_arg_error= 1;
	}
	if( !(af_high= parse_ascanf_address(args[3], _ascanf_variable, "ascanf_ArrayFind", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (invalid high element argument (3)) ";
		ascanf_arg_error= 1;
	}
	if( ascanf_arguments> 4 && ASCANF_TRUE(args[4]) ){
		  /* No use to setting tolerance to 0.... that is the same as not setting tolerance at all. */
		tolerance= args[4]/2;
		tolerance_set= True;
	}
	  /* 20000503: don't resize while compiling?!	*/
	if( !ascanf_SyntaxCheck && !ascanf_arg_error ){
	  unsigned int k, klo= 1, khi= af->N;
		if( af->iarray ){
		 int *iarray= &af->iarray[-1];
			while( khi-klo> 1 ){
				k= (khi+klo) >> 1;
				if( iarray[k] > x ){
					khi= k;
				}
				else{
					klo= k;
				}
			}
		}
		else{
		 double *array= &af->array[-1];
			while( khi-klo> 1 ){
				k= (khi+klo) >> 1;
				if( array[k] > x ){
					khi= k;
				}
				else{
					klo= k;
				}
			}
		}
		klo-= 1;
		khi-= 1;
		{ double yl, yh;
		  int ok1, ok2;
			if( (yl=ASCANF_ARRAY_ELEM(af,klo))<= x
				|| (tolerance_set && ( yl>=(x-tolerance) || yl<=(x+tolerance) ) )
			){
				ok1= True;
			}
			else{
				ok1= False;
			}
			if( (yh=ASCANF_ARRAY_ELEM(af,khi))>= x
				|| (tolerance_set && ( yh>=(x-tolerance) || yh<=(x+tolerance) ) )
			){
				ok2= True;
			}
			else{
				ok2= False;
			}
			if( ok1 && ok2 ){
				af_low->value= klo;
				af_high->value= khi;
				if( fabs(x-yl) <= fabs(yh-x) ){
					*result= -1;
				}
				else{
					*result= 1;
				}
				if( af_low->accessHandler ){
					AccessHandler( af_low, "ArrayFind", level, ASCB_COMPILED, AH_EXPR, NULL   );
				}
				if( af_high->accessHandler ){
					AccessHandler( af_high, "ArrayFind", level, ASCB_COMPILED, AH_EXPR, NULL   );
				}
			}
			else{
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (value %s not between %s[%d]==%s (d=%s) and %s[%d]==%s (d=%s)) ",
						ad2str(x, d3str_format, NULL),
						af->name, klo, ad2str( yl, d3str_format, NULL ), ad2str( yl-x, d3str_format, NULL ),
						af->name, khi, ad2str( yh, d3str_format, NULL ), ad2str( yh-x, d3str_format, NULL )
					);
				}
				*result= 0;
			}
		}
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_registerVarNames ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( ascanf_arguments >= 1 && args ){
		*result = register_VariableNames( (int) ASCANF_TRUE(args[0]) );
	}
	else{
	  int ret = register_VariableNames(True);
		  // we've got the previous setting; restore that setting (we just changed it):
		register_VariableNames(ret);
		*result = ret;
	}
	return( !ascanf_arg_error );
}

#include "dymod.h"

int ascanf_LoadDyMod ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *s1= NULL, *s2= NULL;
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, "ascanf_LoadDyMod", (int) ascanf_verbose, NULL )) ||
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
				s1= NULL;
			}
		}
		if( ascanf_arguments> 1 ){
			if( !(s2= parse_ascanf_address(args[1], 0, "ascanf_LoadDyMod", (int) ascanf_verbose, NULL )) ||
				(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
			){
				if( s2 ){
					fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
						s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
					);
					s2= NULL;
				}
			}
		}
		if( s1 && s1->usage ){
		  int flags= RTLD_LAZY, auto_unload= False, no_dump= False, autolist= False;
		  DyModAutoLoadTables new;
		  extern DyModAutoLoadTables *AutoLoadTable;
		  extern int AutoLoads;
		  char *c;
			if( s2 && s2->usage ){
				if( (c= strcasestr( s2->usage, "auto-load"))== 0 && isspace(c[9]) ){
					autolist= True;
				}
				if( (c= strcasestr( s2->usage, "export"))== 0 && isspace(c[6]) ){
					flags|= RTLD_GLOBAL;
				}
				if( (c= strcasestr( s2->usage, "auto" ))== 0 && isspace(c[4]) ){
					auto_unload= True;
				}
				if( (c= strcasestr( s2->usage, "nodump" ))== 0 && isspace(c[4]) ){
					no_dump= True;
				}
			}
			if( s1->usage[0]!= '\n' ){
				if( *s1->usage ){
					if( s1->usage[ strlen(s1->usage)-1 ]== '\n' ){
						s1->usage[ strlen(s1->usage)-1 ]= '\0';
					}
					if( autolist ){
					  char *c;
					  int str= False;
						if( (c= ascanf_index(s1->usage, ascanf_separator, &str)) ){
							*c= '\0';
							memset( &new, 0, sizeof(new) );
							new.functionName= s1->usage;
							new.DyModName= &c[1];
							new.flags= flags;
							AutoLoadTable= Add_LoadDyMod( AutoLoadTable, &AutoLoads, &new, 1 );
							*result= 1;
						}
						else{
							fprintf( StdErr, " (invalid auto-load specification!) " );
						}
					}
					else{
						if( !LoadDyMod( s1->usage, flags, no_dump, auto_unload ) ){
							ascanf_exit= True;
							ascanf_arg_error= 1;
						}
						else{
							*result= 1;
						}
					}
				}
			}
		}
		else{
			ascanf_arg_error= 1;
		}
	}
	return( !ascanf_arg_error );
}

static int ascanf_UnReloadDyMod ( ASCB_ARGLIST, int reload, char *caller )
{ ASCB_FRAME_SHORT
  ascanf_Function *s1= NULL, *s2= NULL;
  int force= 0, all= 0;
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
		if( !(s1= parse_ascanf_address(args[0], 0, caller, (int) ascanf_verbose, NULL )) ||
			(s1->type== _ascanf_procedure || s1->type== _ascanf_function || !s1->usage)
		){
			if( s1 ){
				fprintf( StdErr, " (1st argument [%s=%s,\"%s\"] is not a valid string)== ",
					s1->name, ad2str( s1->value, d3str_format, 0), (s1->usage)? s1->usage : "<NULL>"
				);
				s1= NULL;
			}
		}
		if( ascanf_arguments> 1 ){
			if( !(s2= parse_ascanf_address(args[1], 0, caller, (int) ascanf_verbose, NULL )) ||
				(s2->type== _ascanf_procedure || s2->type== _ascanf_function || !s2->usage)
			){
				if( s2 ){
					fprintf( StdErr, " (warning: 2nd argument [%s=%s,\"%s\"] is not a valid string)== ",
						s2->name, ad2str( s2->value, d3str_format, 0), (s2->usage)? s2->usage : "<NULL>"
					);
					s2= NULL;
				}
			}
		}
		if( ascanf_arguments> 2 ){
			force= ASCANF_TRUE(args[2]);
		}
		if( s1 && s1->usage ){
		  int flags= RTLD_LAZY, auto_unload= False, no_dump= False;
		  extern DyModAutoLoadTables *AutoLoadTable;
		  extern int AutoLoads;
		  char *c;

			if( s2 ){
				if( reload ){
					if( (c= strcasestr( s2->usage, "export" )) && isspace( c[6]) ){
						flags|= RTLD_GLOBAL;
					}
					if( (c= strcasestr( s2->usage, "auto" )) && isspace( c[4]) ){
						auto_unload= True;
					}
					if( (c= strcasestr( s2->usage, "nodump" )) && isspace( c[6]) ){
						no_dump= True;
					}
				}
				if( (c= strcasestr( s2->usage, "all")) && isspace( c[3]) ){
					if( !reload ){
						all= True;
					}
				}
			}
			if( all ){
			  int n;
			  int N= DyModsLoaded, i= 0;
				while( DyModList && i<= N /* DyModsLoaded */ ){
					UnloadDyMod( DyModList->name, &n, force );
					i+= 1;
				}
				if( DyModsLoaded ){
					fprintf( StdErr, "Warning: %d modules are still loaded!\n", DyModsLoaded );
					*result= -1;
				}
				else{
					*result= 1;
				}
			}
			if( s1->usage[0]!= '\n' ){
				s1->usage= parse_codes(s1->usage);
				if( !all && *s1->usage ){
				  int n, c;
					if( s1->usage[ strlen(s1->usage)-1 ]== '\n' ){
						s1->usage[ strlen(s1->usage)-1 ]= '\0';
					}
					n= UnloadDyMod( s1->usage, &c, force );
					if( n== c ){
						if( reload ){
							  /* NB: we *could* reset Unloaded_Used_Modules here to the value it had
							   \ before the above call to UnloadDyMod(). However, as we're not
							   \ sure that the reload will have put all symbols back at they're original
							   \ addresses, we won't, since that would make a coredump in CleanUp
							   \ possible.
							   */
							if( !LoadDyMod( s1->usage, flags, no_dump, auto_unload ) ){
								ascanf_exit= True;
								ascanf_arg_error= 1;
							}
							else{
								*result= 1;
							}
						}
						else{
							*result= 1;
						}
					}
				}
			}
		}
		else{
			ascanf_arg_error= 1;
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_UnloadDyMod ( ASCB_ARGLIST )
{
	return( ascanf_UnReloadDyMod( ASCB_ARGUMENTS, 0, "ascanf_UnloadDyMod" ) );
}

int ascanf_ReloadDyMod ( ASCB_ARGLIST )
{
	return( ascanf_UnReloadDyMod( ASCB_ARGUMENTS, 1, "ascanf_ReloadDyMod" ) );
}

int ascanf_usleep ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  Time_Struct timer;
	if( ascanf_arguments>= 1 && args[0]> 0 ){
		Elapsed_Since(&timer, True);
		usleep( (unsigned long) args[0] );
		Elapsed_Since(&timer, False);
	}
	else{
		ascanf_arg_error= 1;
		return( 0 );
	}
	*result= timer.HRTot_T;
	return( 1 );
}

