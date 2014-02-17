/* ascanfc.c: simple scanner for double arrays
vim:ts=4:sw=4:
 \ that doubles as a function interpreter.
 \ Includes a "compiler".
 * (C) R.J.V. Bertin 1990,1991,..,1994,1995,1996-1999,2000
 * :ts=4
 * :sw=4
 \ 20000814: made a (tentative?) move towards better reentrantness. The
 \ level indicators were until now distributed: several recursively called
 \ routines had their own static instance. Now, fascanf() and compiled_fascanf()
 \ are front-ends only, that have local level variables that are passed on to
 \ subsequent recursive calls. These level indicators are incremented/decremented
 \ only in ascanf_function() and compiled_ascanf_function(). The function callbacks
 \ that do the actual work receive a pointer to this indicator also: the arguments
 \ passed are now stored in a macro defined in ascanf.h. Each additional variable
 \ of course decreases the speed of calculation somewhat.
 \ 20000818: An new mechanism has been implemented, selectable via ASCANF_ALTERNATE
 \ (in ascanf.h, preferrably). This stores all arguments (pointers) to be passed to
 \ the callbacks in a frame variable, to which a pointer is passed as the only
 \ argument. Some timing tests of this are done in tim-asc-parm.c - the results
 \ vary quite a bit between platforms and compilers. With gcc2.9.5.2 under Linux-PIII,
 \ the ASCANF_ALTERNATE option is good in practice.
 */

#define _ASCANFC_C

#ifdef linux
#	define _GNU_SOURCE
#endif

#include "config.h"
IDENTIFY( "ascanf main module" );

#include <stdio.h>
#include <stdlib.h>
#ifdef _UNIX_C_
#	include <unistd.h>
#endif
#include <ctype.h>
#include <float.h>
#include <math.h>

#include "ux11/ux11.h"
#include "xtb/xtb.h"

#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
#	define USE_SSE2
#	include <xmmintrin.h>
#	include <emmintrin.h>
#	include "AppleVecLib.h"
#endif

#include "dymod.h"

#include "cpu.h"

#include "copyright.h"

extern double drand48();
#define drand()	drand48()

#include "ascanf.h"
#include "compiled_ascanf.h"

#include "Python/PythonInterface.h"
extern DM_Python_Interface *dm_python;

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#	include "arrayvops.h"
#elif defined(__SSE2__) || defined(__SSE3__)
#	define USE_SSE_AUTO
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#	include "arrayvops.h"
#endif

/* 20051112 */
#define unused_pragma_unlikely(x)	(x)
#define unused_pragma_likely(x)		(x)

char ascanf_separator= ',';

int ascanf_verbose, AllowSimpleArrayOps, AlwaysUpdateAutoArrays, AllowProcedureLocals, AllowArrayExpansion, systemtimers= 0;
extern double *ascanf_UseConstantsLists, *ascanf_AllowSomeCompilingInitialisations;
/* int ascanf_unique_automatic_variables= 1;	*/
extern double *ascanf_verbose_value, *AllowSimpleArrayOps_value, *ascanf_ExitOnError, *AlwaysUpdateAutoArrays_value,
	*AllowProcedureLocals_value, *Find_Point_exhaustive_value, *AllowArrayExpansion_value;
double af_verbose_old_value= 0;
extern ascanf_Function *af_verbose, *ascanf_d3str_format, *af_AllowSimpleArrayOps, *af_AlwaysUpdateAutoArrays,
	*af_AllowProcedureLocals, *af_Find_Point_exhaustive, *af_AllowArrayExpansion;
char *TBARprogress_header= NULL;
extern ascanf_Function *ascanf_d2str_NaNCode;
extern ascanf_Function *ascanf_Dprint_fp;
extern int PrintNaNCode, Find_Point_use_precision;

extern char *XGstrdup();

extern ascanf_Function vars_ascanf_Functions[], vars_internal_Functions[];
ascanf_Function **vars_local_Functions= NULL;
extern int ascanf_Functions, internal_Functions;
int *local_Functions;

extern int RemoteConnection, SetIconName;

#include <signal.h>

#undef _IDENTIFY
#include "Macros.h"

#include "Sinc.h"

#include "SS.h"

#include "xgout.h"

#include "Elapsed.h"

Time_Struct *vtimer= NULL;

#include "DataSet.h"

extern DataSet *AllSets;
extern int MaxSets, maxitems;

extern double *ascanf_setNumber, *ascanf_numPoints;
extern ascanf_Function *af_setNumber;
extern int setNumber;

#if defined(sgi) || defined(linux)
#	include <sys/time.h>
#else
#	include <time.h>
#endif

extern char *matherr_mark();
extern int matherr_verbose;
#define MATHERR_MARK()	matherr_mark(__FILE__ ":" STRING(__LINE__))

extern char d3str_format[16];
int ascanf_use_greek_inf= False;

extern int StartUp;

/* 20020404: include fdecl.h to get all (?!) the function declarations; this is possible
 \ after including fdecl_stubs.h that removes the need for xgraph.h (which we can't include here).
 */
#include "fdecl_stubs.h"
#include "fdecl.h"
#undef Process

/* Now, after including fdecl.h, we can safely include the ascanf callback function list: */
#include "ascanfc-table.h"

extern FILE *StdErr, *NullDevice;
extern int debugFlag, debugLevel, line_count;

extern int ascanf_SS_set_bin( SimpleStats *ss, int i, char *name, double *args, double *result, int what, ascanf_Function *af_ss );
extern int ascanf_SAS_set_bin( SimpleAngleStats *sas, int i, char *name, double *args, double *result, int what, ascanf_Function *af_ss );

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
void _ascanf_xfree(void *x, char *file, int lineno, ascanf_Function *af )
{
	if( x ){
#ifdef DEBUG
	if( debugLevel ){
	  int i;
	  char *c= x;
		fprintf(StdErr, "%s,line %d: Freeing 0x%lx ", file, lineno, x);
		if( c ){
			fprintf( StdErr, " { " );
			fflush( StdErr );
			for( i= 0; i< sizeof(int); i++ ){
				fprintf( StdErr, "%x ", c[i] );
				fflush( StdErr );
			}
			fprintf( StdErr, "} \"" );
			for( i= 0; i< sizeof(int); i++ ){
				fprintf( StdErr, "%c", c[i] );
				fflush( StdErr );
			}
			fputs( "\"", StdErr );
		}
		fputs( "\n", StdErr );
	}
#endif
		if( af && af->free ){
			(*af->free)(x);
		}
		else{
			free(x);
		}
// 		x=NULL;
	}
}

#undef xfree
#define xfree(x) {_ascanf_xfree((void*)(x),__FILE__,__LINE__,NULL);(x)=NULL;}

#ifdef DEBUG

int _astrcmp(const char *a, const char *b )
{
	return( strcmp(a, b) );
}

int _astrncmp(const char *a, const char *b, size_t n )
{
	return( strncmp(a, b, n) );
}

#define strcmp(a,b)	_astrcmp(a,b)
#define strncmp(a,b,n)	_astrncmp(a,b,n)

#endif

static double Then, Tot_Then, sThen;
double Tot_Start= 0.0, Tot_Time= 0.0, Used_Time= 0.0;

#if defined(sgi) && defined(FFTW_CYCLES_PER_SEC)
#	include <sys/syssgi.h>
	fftw_time timer_wraps_at;
#endif

void Set_Timer()
{	struct tms tms;
	Tot_Then= (double)times( &tms);
	Then= (double)(tms.tms_utime+ tms.tms_cutime);
	sThen= (double)(tms.tms_stime+ tms.tms_cstime);
	Tot_Time= 0.0;
	Used_Time= 0.0;
	if( !Tot_Start){
		Tot_Start= Tot_Then;
	}
#if defined(sgi) && defined(FFTW_CYCLES_PER_SEC)
	{ fftw_time res;
	  unsigned long bits;
		clock_getres( FFTW_SGI_CLOCK, &res );
		bits= syssgi(SGI_CYCLECNTR_SIZE);
		switch( bits ){
			case 32:{
			  double wrap_time;
				wrap_time= (res.tv_sec + res.tv_nsec* 1e-9)* ((unsigned long)0xffffffff);
				timer_wraps_at.tv_sec= ssfloor(wrap_time);
				timer_wraps_at.tv_nsec= (unsigned long) 1e9* (wrap_time- timer_wraps_at.tv_sec);
				break;
			}
#if _MIPS_SZLONG == 64
			case 64:{
			  /* This is untested code, a dumb 64bit extension of the 32bit case!! */
			  long double wrap_time;
				wrap_time= (res.tv_sec + res.tv_nsec* 1e-9)* ((unsigned long long)0xffffffffffffffff);
				timer_wraps_at.tv_sec= ssfloor(wrap_time);
				timer_wraps_at.tv_nsec= (unsigned long long) 1e9* (wrap_time- timer_wraps_at.tv_sec);
				break;
			}
#endif
			default:
				fprintf( stderr, "SetTimer(): Unknown/unsupported number of clock bits syssgi(SGI_CYCLECNTR_SIZE)==%lu\n",
					bits
				);
				memset( &timer_wraps_at, 0, sizeof(fftw_time) );
				break;
		}
	}
#endif
}

static struct tms ET_tms;
static double ET_Now, ET_Tot_Now;
static double sET_Now;
double Elapsed_Time()	/* return number of seconds since last call */
{	struct tms *tms= &ET_tms;
	double Elapsed;

	ET_Tot_Now= (double) times( tms);
	ET_Now= (double)(tms->tms_utime+ tms->tms_cutime);
	sET_Now= (double)(tms->tms_stime+ tms->tms_cstime);
	Elapsed= (ET_Now- Then)/((double)TICKS_PER_SECOND);
		Then= ET_Now;
		sThen= sET_Now;
	Tot_Time= (ET_Tot_Now- Tot_Then)/((double)TICKS_PER_SECOND);
		Tot_Then= ET_Tot_Now;
	return( (Used_Time= Elapsed) );
}

/* #ifdef linux	*/
int elapsed_verbose= 0;
/* #endif	*/

static struct tms ES_tms;
double Delta_Tot_T;

double Elapsed_Since( Time_Struct *then, int update )	/* return number of seconds since last call */
{	struct tms *tms= &ES_tms;
	double ES_Now, ES_Tot_Now, Elapsed;
	double sES_Now;

	ES_Tot_Now= (double) times( tms);
	ES_Now= (double)(tms->tms_utime+ tms->tms_cutime);
	sES_Now= (double)(tms->tms_stime+ tms->tms_cstime);
	then->Tot_Time= Tot_Time= (ES_Tot_Now- then->Tot_TimeStamp)/((double)TICKS_PER_SECOND);
	then->Time= Elapsed= (ES_Now- then->TimeStamp)/((double)TICKS_PER_SECOND);
	then->sTime= (sES_Now- then->sTimeStamp)/((double)TICKS_PER_SECOND);
#ifdef linux
	if( elapsed_verbose ){
		fprintf( StdErr,
			"Tot=(%g-%g)/(TT=%g)=%g Usr=((%ld+%ld=%g)-%g)/TT=%g Sys=((%ld+%ld=%g)-%g)/TT=%g == ",
				ES_Tot_Now, then->Tot_TimeStamp, (double) TICKS_PER_SECOND, Tot_Time,
				tms->tms_utime, tms->tms_cutime, ES_Now, then->TimeStamp, then->Time,
				tms->tms_stime, tms->tms_cstime, sES_Now, then->sTimeStamp, then->sTime
		);
		elapsed_verbose= False;
	}
#endif
#ifdef FFTW_CYCLES_PER_SEC
	{ fftw_timers t;
		fftw_get_time(&t.t2);
		t.t1= then->prev_tv;
		Delta_Tot_T= fftw_time_diff_to_sec( &t );
		then->prev_tv= t.t2;
		t.t1= then->Tot_tv;
		then->HRTot_T= fftw_time_diff_to_sec( &t );
		if( update || then->do_reset ){
			then->Tot_tv= t.t2;
			then->TimeStamp= ES_Now;
			then->sTimeStamp= sES_Now;
			then->Tot_TimeStamp= ES_Tot_Now;
			then->do_reset= False;
		}
	}
#else
	  /* The gettimeofday() part:	*/
	{ struct timezone tzp;
	  struct timeval ES_tv;

		gettimeofday( &ES_tv, &tzp );
		Delta_Tot_T= (ES_tv.tv_sec - then->prev_tv.tv_sec) + (ES_tv.tv_usec- then->prev_tv.tv_usec)* 1e-6;
		then->prev_tv= ES_tv;
			  /* 20020418: I don't see why the normal and the high-resolution parts should behave differently
			   \ as a function of <update> or <then->do_reset>!
			   */
			then->HRTot_T= (ES_tv.tv_sec - then->Tot_tv.tv_sec) + (ES_tv.tv_usec- then->Tot_tv.tv_usec)* 1e-6;
		if( update || then->do_reset ){
			then->Tot_tv= ES_tv;
			then->TimeStamp= ES_Now;
			then->sTimeStamp= sES_Now;
			then->Tot_TimeStamp= ES_Tot_Now;
			then->do_reset= False;
		}
	}
#endif
	return( (Used_Time= Elapsed) );
}

/* Version only using the HR timer */
double Elapsed_Since_HR( Time_Struct *then, int update )
{
#ifdef FFTW_CYCLES_PER_SEC
	{ fftw_timers t;
		fftw_get_time(&t.t2);
		t.t1= then->prev_tv;
		Delta_Tot_T= fftw_time_diff_to_sec( &t );
		then->prev_tv= t.t2;
		t.t1= then->Tot_tv;
		then->HRTot_T= fftw_time_diff_to_sec( &t );
		if( update || then->do_reset ){
			then->Tot_tv= t.t2;
			then->do_reset= False;
		}
	}
#else
	  /* The gettimeofday() part:	*/
	{ struct timezone tzp;
	  struct timeval ES_tv;

		gettimeofday( &ES_tv, &tzp );
		Delta_Tot_T= (ES_tv.tv_sec - then->prev_tv.tv_sec) + (ES_tv.tv_usec- then->prev_tv.tv_usec)/ 1e6;
		then->prev_tv= ES_tv;
		then->HRTot_T= (ES_tv.tv_sec - then->Tot_tv.tv_sec) + (ES_tv.tv_usec- then->Tot_tv.tv_usec)/ 1e6;
		if( update || then->do_reset ){
			then->Tot_tv= ES_tv;
			then->do_reset= False;
		}
	}
#endif
	return( Delta_Tot_T );
}


#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
int ASCANF_TRUE_(double x)
{
	return( ASCANF_TRUE(x) );
}

/* #if defined(ASCANF_ALTERNATE) && ( defined(ASCANF_ARG_DEBUG) || defined(DEBUG) )	*/

  /* A routine that will return frame->expr (if non-null), and print it on StdErr. The stub argument
   \ is only there to receive a pointer to the local variable (theExpr below) in which the expression is
   \ to be stored. This prevents gcc from complaining about theExpr being unused (it still is of course...
   \ but gcc is too braindead to understand that :))
   */
static char _ascb_default_expr[]= "";
char *_callback_expr( ascanf_Callback_Frame *__ascb_frame, char *fn, int lnr, char **stub )
{ char *expr;
	if( fn && (expr= rindex( fn, '/')) ){
	  /* Take the basename of the filename */
		fn= expr+1;
	}
	if( __ascb_frame ){
		if( AH_EXPR ){
			fprintf( StdErr, "###%c%d %s (%s:%d) ### ",
				(__ascb_frame->compiled)? 'C' : '#', *(__ascb_frame->level), AH_EXPR, fn, lnr
			);
			expr= AH_EXPR;
		}
		else{
			fprintf( StdErr, "###%c%d %s (%s:%d) ### ",
				(__ascb_frame->compiled)? 'C' : '#', *(__ascb_frame->level),
				(__ascb_frame->self && __ascb_frame->self->name)? __ascb_frame->self->name : "<Missing Expression!>", fn, lnr
			);
			expr= _ascb_default_expr;
		}
		fflush( StdErr );
	}
	else{
		fprintf( StdErr, "\n###??! callback without valid callback argument frame!!! (%s:%d)\n\n", fn, lnr );
		expr= _ascb_default_expr;
	}
	return(expr);
}

/* #endif	*/

/* IDENTIFY( "ascanf (compiled) routines");	*/

extern Pixmap dotMap ;
Window ascanf_window;
extern Display *disp;
extern Window init_window;

#define HAVEWIN	(ascanf_window || init_window)
#define USEWINDOW	((ascanf_window)? ascanf_window : init_window)

int Ascanf_Max_Args= 0;

#if defined(ASCANF_ALTERNATE) && defined(DEBUG)
	int ascanf_frame_has_expr= 1;
#else
	int ascanf_frame_has_expr= 0;
#endif

/* A lot of the following ..ascanf_.. variables should be lumped into a structure
 \ associated with a LocalWin (plus a global set). This should allow really
 \ reentrant code, where concurrent processings in different windows won't interfere...
 */
int reset_ascanf_currentself_value= True, reset_ascanf_index_value= True;
extern double *ascanf_self_value, *ascanf_current_value, *ascanf_index_value;
double *ascanf_memory= NULL, ascanf_progn_return;
extern double *ascanf_popup_verbose, *ascanf_compile_verbose, *ascanf_Variable_init;
static xtb_frame *popup_menu= NULL;
int ascanf_arguments, ascanf_arg_error, ascanf_comment= 0, ascanf_popup= 0;
char ascanf_errmesg[512];
char *ascanf_emsg= NULL;

extern double *ascanf_counter, *ascanf_Counter;
extern ascanf_Function *af_Counter;

extern double *ascanf_data0, *ascanf_data1, *ascanf_data2, *ascanf_data3;
extern ascanf_Function *af_data0, *af_data1, *af_data2, *af_data3;

extern char **ascanf_SS_names;
extern SimpleStats *ascanf_SS;
extern char **ascanf_SAS_names;
extern SimpleAngleStats *ascanf_SAS;

extern NormDist *ascanf_normdists;

static double ADB[ASCANF_DATA_COLUMNS];
double *ascanf_data_buf= ADB;
int *ascanf_column_buf;

static int ascanf_loop= 0, *ascanf_loop_ptr= &ascanf_loop, *ascanf_in_loop= NULL, *ascanf_loop_incr= NULL;
extern double *ascanf_loop_counter;
#ifdef DEBUG
	double *ascanf_forto_MAX= NULL;
#endif

int ascanf_escape= False;
int ascanf_interrupt= False;
double *ascanf_interrupt_value, *ascanf_escape_value;
ascanf_Function *af_interrupt, *af_escape;


long d2long( double x)
{ double X= (double) MAXLONG;
  long ret;
	if( x <= X && x>= -X -1L ){
		ret = ( (long) x);
	}
	else if( x< 0){
		ret = ( LONG_MIN);
	}
	else if( x> 0){
		ret = ( MAXLONG);
	}
	return ret;
}

short d2short(double x)
{ double X= (double)MAXSHORT;
  short ret;
	if( x <= X && x>= -X -1L ){
		ret = ( (long) x);
	}
	else if( x< 0){
		ret = ( SHRT_MIN);
	}
	else if( x> 0){
		ret = ( MAXSHORT);
	}
	return ret;
}

int d2int( double x)
{ double X= (double)MAXLONG;
  int ret;
	if( x <= X && x>= -X -1L ){
		ret = ( (long) x);
	}
	else if( x< 0){
#ifdef INT_MIN
		ret = ( INT_MIN);
#else
		ret = ( LONG_MIN);
#endif
	}
	else if( x> 0){
#ifdef MAXINT
		ret = MAXINT;
#else
		ret = ( MAXLONG);
#endif
	}
	return ret;
}

/* find the first occurence of the character 'match' in the
 \ string arg_buf, skipping over balanced pairs of <brace-left>
 \ and <brace_right>. If match equals ' ' (a space), any whitespace
 \ is assumed to match.
 */
#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
char *find_balanced( char *arg_buf, const char match, const char brace_left, const char brace_right, int others, int *instring )
{ int brace_level= 0, _instring= 0, str_level= 0, cbrace_level= 0;
  char *d= arg_buf;
	while( d && *d && !( ((match==' ' || match== '\t' )? isspace((unsigned char)*d) : *d== match) &&
			  /* 20020602: extended in-string checking: */
			(brace_level + str_level + cbrace_level)== 0 && (!instring || !*instring) && !_instring)
	){
	  int other_open= False, other_close= False;
		if( others ){
			switch( *d ){
				case '"':
					  /* _instring is a local in-string flag that supposes we are not called in the
					   \ middle of a string; it should probably be synchronised with the instring
					   \ pointer argument (=> check (instring!=&_instring || !*instring))
					   */
					if( _instring ){
						_instring= False;
						other_close= True;
						str_level-= 1;
					}
					else{
						_instring= True;
						other_open= True;
						str_level+= 1;
					}
					if( instring ){
						*instring= !*instring;
					}
					break;
				case '{':
					  /* 20020602: extended in-string checking: */
					if( !_instring ){
						other_close= True;
						cbrace_level+= 1;
					}
					break;
				case '}':
					  /* 20020602: extended in-string checking: */
					if( !_instring ){
						other_close= True;
						cbrace_level-= 1;
					}
					break;
			}
		}
		  /* 20020602: extended in-string checking: */
		if( !_instring ){
			if( *d== brace_left ){
				brace_level++;
			}
			else if( *d== brace_right ){
				brace_level--;
			}
		}
		d++;
	}
	return( (((match==' ' || match== '\t')? isspace((unsigned char)*d) : *d== match))? d : NULL );
}

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
char *find_nextquote( char *arg_buf )
{ char *prev= arg_buf;
	if( !arg_buf || !*arg_buf ){
		return( NULL );
	}
	do{
		if( *arg_buf== '"' && *prev!= '\\' ){
			return( arg_buf );
		}
		else{
			prev= arg_buf;
			arg_buf++;
		}
	} while( *arg_buf );
	return( NULL );
}

#ifndef linux
#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
static char *astrdup( const char *c )
{  int len;
   char *d;
	if( c ){
		len= strlen(c)+ 1;
		if( (d= malloc( len* sizeof(char) )) ){
			  /* memcpy is generally faster than strcpy! */
			memcpy( d, c, len );
		}
	}
	else{
		d= NULL;
	}
	return( d );
}
#define strdup(c)	astrdup(c)
#endif

/* 20020515: strdup a string, parsing it such that \" is replaced by "
 \ Note that this is a quick and dirty implementation that does the
 \ copying twice!
 */
#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
char *strdup_unquote_string( char *string )
{ char *c= NULL;
	if( string ){
		c= strdup(string);
		if( c && index( string, '"') ){
		  char *d= c;
			while( *string ){
				switch( *string ){
					case '\\':
						if( string[1]== '"' ){
							string++;
						}
						  /* no break; */
					default:
						*d++= *string++;
						break;
				}
			}
			*d= '\0';
		}
	}
	return(c);
}

#ifdef __clang__
static
#endif
#if defined(__GNUC__)
__inline__
#endif
double pow_(double a, double b)
{
	MATHERR_MARK();
	if( a< 0 ){
		return( -pow( -a, b) );
	}
	else{
		return( pow( a, b) );
	}
}

char *ascanf_type_name[_ascanf_types]= { "NOT_EOF", "NOT_EOF_OR_RETURN", "value", "function", "variable", "array", "procedure",
	"SimpleStats", "SimpleAngleStats", "PyObject",
	"deleted-variable",
};

char *AscanfTypeName( int type )
{
	if( type< 0 || type>= _ascanf_types ){
		return( "<invalid type>" );
	}
	else{
		return( ascanf_type_name[type] );
	}
}

DEFUN( ascanf_Variable, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_Procedure, ( ASCB_ARGLIST ), int );
#define COMP_FASCANF_INTERNAL	_compiled_fascanf
static int COMP_FASCANF_INTERNAL( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **Form, int *level );

static long ascanf_Variable_id= 0L;
static Compiled_Form *Evaluating_Form= NULL;

#define FASCANF_INTERNAL	_fascanf
static int FASCANF_INTERNAL( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **form, int *level, int *nargs );
char *fascanf_unparsed_remaining= NULL;

/* 20030330: an interface to _fascanf() that will cache fascanf_unparsed_remaining: */
int __fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **form, int *level, int *nargs );
#if defined(__GNUC__)
__inline__
#endif
int __fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **form, int *level, int *nargs )
{ char *fur= fascanf_unparsed_remaining;
  int ret= _fascanf( n, s, a, ch, data, column, form, level, nargs );
	fascanf_unparsed_remaining= fur;
	return(ret);
}

  /* True when an expression is being compiled. Some functions don't do anything in this case
   \ (actually, ascanf_function doesn't (shouldn't) call them in this case). Functions that result
   \ in some change of the internal state should check this variable, and refrain from such actions
   \ when it is true.
   \ 981207: ascanf_function no longer calls the internal functions anymore when compiling or when
   \ ascanf_SyntaxCheck is True. This was already the case for the 1st call (the empty parameter-list case),
   \ but now the 2 others (parameter list, and no parameter list) follow the principle. The Defined?[] semi-function
   \ *is* "evaluated" (doesn't involve a call of a C function).
   */
int ascanf_SyntaxCheck= 0;
int ascanf_PopupWarn= 0;
char *ascanf_CompilingExpression= NULL;

#define FUNNAME(fun)	((fun)?((fun)->name)?(fun)->name:"??":"NULL")
#define FORMNAME(f)		((f)? ((f)->expr)?(f)->expr : FUNNAME((f)->fun) : "NULL")
#define FORMFNAME(f)		((f)? FUNNAME((f)->fun) : "NULL")

typedef struct Add_Form_Flags{
	int store;
	int sign;
	int negate;
	int take_address, take_usage, last_value, empty_arglist;
} Add_Form_Flags;

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
int FormArguments( Compiled_Form *form, ascanf_Function *Function )
{ int n;
	if( Function ){
		if( Function->Nargs>= 0 && form->special_fun!= not_a_function ){
			n= Function->Nargs;
		}
		else{
		  /* A function that can take any number of arguments. We only need to allocate as many as are
		   \ actually passed, and hope that the callback doesn't access arguments that it didn't get.
		   \ _ascanf_variable, _ascanf_array and _ascanf_procedure etc. variables are also treated here.
		   */
			n= form->argc;
		}
	}
	else{
		n= 0;
	}
	return(n);
}

/* Append a Compiled_Form to the growing tree. The result is left in 'form', a pointer
 \ which always points to the first node in the expressiontree.
 */
Compiled_Form *Add_Form( Compiled_Form **form, ascanf_type type, Add_Form_Flags *flags, double value, char *name, ascanf_Function *fun, Compiled_Form *args )
{ Add_Form_Flags defflags= { 0, 1, 0 };

	if( !flags ){
		flags= &defflags;
	}
	if( form ){
	  Compiled_Form *f, *top= (*form)? (*form)->top : NULL;
		if( (f= (Compiled_Form*) calloc(1, sizeof(Compiled_Form))) ){
		  int verbose= (debugFlag && debugLevel);
			f->type= type;
			f->value= value;
			if( flags->store && !fun->store ){
				if( ascanf_verbose ){
					fputs( " #ac## ignoring `*' store opcode! ### ", StdErr );
				}
				flags->store= False;
			}
			f->empty_arglist= flags->empty_arglist;
			f->store= flags->store;
			f->sign= flags->sign;
			f->negate= flags->negate;
			f->take_address= flags->take_address;
			f->take_usage= flags->take_usage;
			if( f->take_address || f->take_usage ){
				f->pointer= parse_ascanf_address(f->value, 0, "Add_Form", (int) ascanf_verbose, NULL );
			}
			f->last_value= flags->last_value;
			f->expr= (name)? strdup(name) : NULL;
			f->top= top;
			f->parent= (*form);
			if( (f->fun= fun) ){
					if( !fun->special_fun ){
					  /* correct the ascanf_Function structure if necessary. This may be the case for
					   \ functions loaded via Dynamic Modules... And for runtime-declared variables and
					   \ procedures.
					   */
						if( fun->function== ascanf_Variable || fun->function== ascanf_Procedure ){
							fun->special_fun= not_a_function;
						}
					}
				fun->links+= 1;
#ifdef XG_DYMOD_SUPPORT
				if( fun->dymod ){
				  /* 20020518: if top will be set to f if not yet set, so pass either top OR f
				   \ to register the DyMod dependency, otherwise we'll miss one expression node
				   \ and that suffices to cause a crash!!
				   */
 				  Compiled_Form *depf= (top)? top : f;
 					Add_DyMod_Dependency( fun->dymod, depf, depf->fun, fun, f );
// 					if( top ){
// 						Add_DyMod_Dependency( fun->dymod, top, top->fun, fun, f );
// 					}
// 					Add_DyMod_Dependency( fun->dymod, f, f->fun, fun, f );
				}
#endif
				f->special_fun= fun->special_fun;
			}
			else{
				f->special_fun= direct_fun;
			}
			f->args= args;
			f->argc= 0;
			f->level= 0;
			while( args ){
				if( top ){
					args->top= top;
				}
#ifdef DEBUG
				if( !args->parent ){
					args->parent= f;
				}
				else{
				    /* 20020322: all non-first items in the arglist will have a non-NULL parent: the previous argument! */
					args->parent= f;
				}
#else
				args->parent= f;
#endif
				f->argc+= 1;
				args= args->cdr;
			}
			f->alloc_argc= FormArguments(f, f->fun);
			if( f->args
#if ASCANF_FORM_ARGVALS == 1
#else
				&& (type== _ascanf_procedure || (type== _ascanf_array && *ascanf_UseConstantsLists))
#endif
			){
				if( !(f->argvals= (double*) calloc( f->alloc_argc, sizeof(double) )) ){
					fprintf( StdErr, "Add_Form(): can't allocate memory for the expression's %d arguments' values (%s)\n",
						f->alloc_argc, serror()
					);
					verbose= True;
				}
			}
			f->last_cdr= f;
			set_NaN(f->last_eval_time);
			if( verbose ){
				fprintf( StdErr, "Add_Form(\"%s\",\"%s\",%d*%g,\"%s\",\"%s\",\"%s\")\n",
					FORMNAME(*form), ascanf_type_name[type], flags->sign, value,
					(name)? name : "NULL", FUNNAME(fun), FORMNAME(args)
				);
				fflush( StdErr );
			}
		}
		else{
		  char buf[1024];
			sprintf( buf, "Add_Form(\"%s\",\"%s\",%d*%g,\"%s\",\"%s\",\"%s\"): can't get new Compiled_Form (%s)\n",
				FORMNAME(*form), ascanf_type_name[type], flags->sign, value,
				(name)? name : "NULL", FUNNAME(fun), FORMNAME(args), serror()
			);
			fputs( buf, StdErr );
			fflush( StdErr );
			if( HAVEWIN ){
				xtb_error_box( USEWINDOW, buf, "Compilation error" );
			}
		}
		if( *form ){
			if( f ){
				(*form)->last_cdr->cdr= f;
				(*form)->last_cdr= f;
			}
		}
		else if( f ){
			*form= f;
			(*form)->top= f;
		}
		return( *form );
	}
	return( NULL );
}

int Correct_Form_Top( Compiled_Form **Form, Compiled_Form *top, ascanf_Function *af )
{ Compiled_Form *form;
  int n= 0;
  extern int Correct_DyMod_Dependencies( struct Compiled_Form *top, struct Compiled_Form *old );
	if( Form ){
		form= *Form;
		if( !top ){
			top= form;
		}
		if( top ){
			top->af= af;
		}
		while( form ){
			if( form->top!= top ){
				if( form->top && form->top->DyMod_Dependency ){
					Correct_DyMod_Dependencies( top, form->top );
				}
				if( af ){
					  /* Each 1st entry in a form's argument list (args) will have form->af==af */
					form->top->af= af;
				}
				form->top= top;
				n+= 1;
			}
			if( form->args ){
				n+= Correct_Form_Top( &(form->args), top, af );
			}
			form= form->cdr;
		}
	}
	return( n );
}

static void Decrement_Form_FunctionLinks( Compiled_Form *form )
{
	while( form ){
		if( form->fun ){
			form->fun->links-= 1;
		}
		Decrement_Form_FunctionLinks( form->args );
		form= form->cdr;
	}
}

int _Destroy_Form( Compiled_Form **form, int del0link )
{  Compiled_Form *cdr;
   long i= 0;
   static int level= 0;
	if( form ){
		if( *ascanf_compile_verbose> 1 ){
			if( *form && !level ){
				fprintf( StdErr, "# Destroying compiled expression:\n# " );
				Print_Form( StdErr, form, 0, True, "# ", NULL, "\n", True );
			}
		}
		if( *form && (*form)->vars_local_Functions && (*form)->local_Functions> 0 ){
		  ascanf_Function *af;
			while( (af= (*form)->vars_local_Functions) && (*form)->local_Functions> 0 ){
				(*form)->vars_local_Functions= af->cdr;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (deleting local %s \"%s\")",
						AscanfTypeName(af->type), (af->name)? af->name : "??"
					);
				}
				Delete_Variable( af );
				xfree(af);
				(*form)->local_Functions-= 1;
			}
		}
		if( !level ){
			  /* First, decrement the link-count of all referenced functions, variables, etc.
			   \ This is necessary because module dependencies have to be handled in a sort
			   \ of feed-forward fashion. I.e, it is not safe/wise to delete them after deleting
			   \ a variable, and before doing that, its non-zero link-count will prevent auto-unloading.
			   \ We can safely do this link decrementing here since no matter what happens, the current
			   \ form is going to be destroyed.
			   */
			Decrement_Form_FunctionLinks( *form );
		}
		level+= 1;
		while( *form ){
			if( (*form)->DyMod_Dependency ){
				if( *ascanf_compile_verbose> 1 && (*form)->fun ){
					fprintf( StdErr, "# Deleting DYMOD dependencies for %s\n", FUNNAME((*form)->fun) );
				}
				  /* This will also mark unneeded auto-unloadable modules for unloading: */
				Delete_DyMod_Dependencies( (*form), NULL, "Destroy_Form()" );
			}
//#if ASCANF_FORM_ARGVALS == 1
//			if( (*form)->args ){
				xfree( (*form)->argvals );
//			}
//#endif
			i+= Destroy_Form( &((*form)->args) );
			cdr= (*form)->cdr;
			if( (*form)->fun ){
				if( (*form)->fun->links<= 0 ){
				  /* This code is why module auto-unloading has to be delayed: a SIGSEGV would result
				   \ if (*form)->fun was in a (just) unloaded module.
				   */
					if( del0link ){
						(*form)->fun->links= 0;
						if( (*form)->fun->internal && !(*form)->fun->user_internal ){
							if( *ascanf_compile_verbose> 1 ){
							  char *c;
								fprintf( StdErr, "# Deleting auto-variable " );
								c=FUNNAME((*form)->fun);
								print_string2( StdErr, "", "\n", c, (*c=='"')? False : True );
							};
							Delete_Internal_Variable( NULL, (*form)->fun );
						}
					}
					else{
						  /* 20020826: this is a kludge to prevent (future) others from deleting
						   \ internal variables that apparently shouldn't be deleted... Adios
						   \ (future) GC...
						   */
						(*form)->fun->links= 1;
					}
				}
			}
			(*form)->type= _ascanf_novariable;
			(*form)->af= NULL;
			(*form)->top= NULL;
			(*form)->parent= NULL;
			(*form)->cdr= NULL;
			xfree( (*form)->expr );
			xfree( *form );
			*form= NULL;
			*form= cdr;
			i++;
		}
		*form= NULL;
		if( !(level-= 1) ){
		  /* Now finally unload the module(s) that are marked no longer needed: */
			Auto_UnloadDyMods();
		}
	}
	return( i );
}

int Destroy_Form( Compiled_Form **form )
{
	return( _Destroy_Form(form, True) );
}

typedef union AscanfAddresses{
	struct {
		ascanf_Function_type type;
		address32 address;
	} handle;
	double value;
} AscanfAddresses;

#ifndef USE_AA_REGISTER
	int PAS_already_protected= False;
	int PAS_Lazy_Protection= True;
#	ifdef DEBUG
	static int PAS_type;
	static double PAS_value;
	static char *PAS_caller;
#	endif

#	include <setjmp.h>
	static jmp_buf pa_jmp, pa_jmp_top;
	static int pa_sig;
#	undef abort
#	define VOLATILE	volatile
#else
	int PAS_Lazy_Protection= False;
#	define VOLATILE /* */
#endif
static int parsing_address;

void segv_handler( int sig )
{
#ifndef USE_AA_REGISTER
	if( sig== SIGSEGV || sig== SIGBUS ){
		if( parsing_address ){
			pa_sig= sig;
			  /* Allow parse_ascanf_address() to finish elegantly. We can't just ignore the current signal, however,
			   \ so we have to do a longjmp to after the place where we caused the signal.
			   \ We will be restored before parse_ascanf_address() exits.
			   */
			signal( sig, segv_handler );
			if( PAS_Lazy_Protection== 2 ){
#ifdef DEBUG
			  AscanfAddresses aa;
				aa.value= PAS_value;
				fprintf( StdErr, "parse_ascanf_address(%s[%s],0x%x,\"%s\"): invalid address: 0x%lx\n",
					d2str(PAS_value,d3str_format, NULL), d2str(PAS_value, "%dhex", NULL), PAS_type, PAS_caller,
					aa.handle.address
				);
				fprintf( StdErr,
					"                        : \"Ascanf_Lazy_Address_Protection\"==2: "
						"we abort the toplevel expression currently evaluating!\n"
				);
#endif
				fprintf( StdErr, "Received signal %d while trying to parse an (invalid?) address:"
					" aborting toplevel expression\n", sig );
				fprintf( stdout, "### Received signal %d while trying to parse an (invalid?) address:"
					" aborting toplevel expression\n", sig );
				siglongjmp(pa_jmp_top,1);
			}
			else{
				fprintf( StdErr, "Received signal %d while trying to parse an (invalid?) address\n", sig );
				fprintf( stdout, "### Received signal %d while trying to parse an (invalid?) address\n", sig );
				siglongjmp(pa_jmp, 1);
			}
		}
		else{
			fprintf( StdErr, "segv_handler(%d): aborting.\n", sig );
#ifndef __GNUC__
			abort();
#else
			  /* Try continuing with the default handlers. abort() is not very useful as a backtrace
			   \ on the coredump will originate in this function...
			   */
			signal( sig, SIG_DFL );
#endif
		}
	}

	else
#endif

	{
		if( parsing_address ){
			fprintf( StdErr, "Received signal %d while trying to parse an (invalid?) address\n", sig );
			fprintf( stdout, "### Received signal %d while trying to parse an (invalid?) address\n", sig );
		}
		else{
			fprintf( StdErr, "segv_handler(%d): aborting.\n", sig );
		}
#ifndef __GNUC__
		abort();
#else
		signal( sig, SIG_DFL );
#endif

	}
}

// 20080711: updating of linked arrays is done when the linked-to set is resized/deleted
#undef ALWAYS_CHECK_LINKEDARRAYS

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
void Check_linkedArray(ascanf_Function *af)
{ int N= af->N;
	if( pragma_likely(af->linkedArray.dataColumn=
			ascanf_getDataColumn( af->linkedArray.set_nr, af->linkedArray.col_nr, &N ))
	){
		af->array= af->linkedArray.dataColumn;
		af->N= N;
	}
	else{
		if( pragma_unlikely(ascanf_verbose) && af->array ){
			fprintf( StdErr, " (array \"%s\" linked to invalid/non-existing DataSet#%d[%d])== ",
				af->name, af->linkedArray.set_nr, af->linkedArray.col_nr
			);
		}
		unregister_LinkedArray(af);
		af->array= NULL;
		  // prevent reallocation errors, set N to 0 (which is the correct count...)
		af->N= 0;
		if( af->linkedArray.set_nr>= 0 ){
			af->linkedArray.set_nr= -af->linkedArray.set_nr - 1;
		}
		if( af->linkedArray.col_nr>= 0 ){
			af->linkedArray.col_nr= -af->linkedArray.col_nr - 1;
		}
		  // prevent 0-element arrays:
		Resize_ascanf_Array( af, 1, NULL );
	}
}

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
pragma_malloc ascanf_Function *parse_ascanf_address( double a, int this_type, char *caller, int verbose, int *take_usage )
{ VOLATILE AscanfAddresses aa;
  VOLATILE ascanf_Function *aaaf;
#ifdef DEBUG
  int warn= False;
#endif
	aa.value= a;
	if( aa.handle.address ){
	  VOLATILE int usg= 0;
	  VOLATILE ascanf_Function_type type= aa.handle.type- 0x12345600;
	  int pa= parsing_address;
		parsing_address= True;
		if( type & 0x10000000 ){
			if( take_usage ){
				*take_usage= 1;
			}
			usg= 1;
			type-= 0x10000000;
		}
		else if( take_usage ){
			*take_usage= 0;
		}
#if DEBUG == 2
		if( ((unsigned long) aa.handle.address) % 8 != 0 ){
			fprintf( StdErr, "Warning: parse_ascanf_address(%.22lf): aa.handle.address==0x%lx not aligned to 8\n",
				a, aa.handle.address
			);
			warn= True;
		}
#endif
		if( (this_type== 0 && (type== _ascanf_variable || type== _ascanf_function ||
				type== _ascanf_simplestats || type== _ascanf_simpleanglestats || type== _ascanf_python_object ||
				type== NOT_EOF || type== NOT_EOF_OR_RETURN || type== _ascanf_array || type== _ascanf_procedure)
			) ||
			type== this_type
		){
		  ascanf_Function_type atype;
#ifndef USE_AA_REGISTER
#ifdef DEBUG
			if( PAS_Lazy_Protection== 2 ){
				PAS_value= a;
				PAS_type= this_type;
				PAS_caller= caller;
			}
#endif
			if( PAS_already_protected || sigsetjmp(pa_jmp,1)== 0 ){
				PAS_already_protected= PAS_Lazy_Protection;
				atype= (aaaf=aa.handle.address)->type;
#else
		  int parsing_address_error= False;
				if( pragma_likely( (aaaf= verify_ascanf_Address( (address32) aa.handle.address,type)) ) ){
					atype= aaaf->type;
				}
				else{
					parsing_address_error= True;
				}
#endif
				if( atype!= type || parsing_address_error ){
					if( pragma_unlikely( verbose || debugFlag ) ){
						fprintf( StdErr, "parse_ascanf_address(%s[%s],0x%x,\"%s\"): %s: 0x%lx->type?=0x%lx != 0x%lx\n",
							d2str(a,d3str_format, NULL), d2str(a, "%dhex", NULL), this_type,caller,
							(parsing_address_error)? "garbled pointer" : "unexpected pointer type",
							aa.handle.address, (long) atype, (long) type
						);
					}
					aa.handle.address = 0, aaaf= NULL;
				}
#ifndef USE_AA_REGISTER
			}
			else{
				if( pragma_unlikely( verbose || debugFlag ) ){
					fprintf( StdErr, "parse_ascanf_address(%s[%s],0x%x,\"%s\"): invalid address: 0x%lx "
							"(Ascanf_Lazy_Address_Protection==%d)\n",
						d2str(a,d3str_format, NULL), d2str(a, "%dhex", NULL), this_type,caller,
						aaaf, PAS_Lazy_Protection
					);
				}
/* #ifdef DEBUG	*/
				else{
					fprintf( StdErr, "Received signal %d while trying to parse an (invalid?) address\n", pa_sig );
				}
/* #endif	*/
				aa.handle.address= aaaf= NULL;
				parsing_address= False;
			}
#endif
			if( aaaf ){
#ifdef ALWAYS_CHECK_LINKEDARRAYS
				if( aaaf->linkedArray.dataColumn ){
					Check_linkedArray(aaaf);
				}
#endif
				if( pragma_unlikely(verbose) ){
					fprintf( StdErr, " (%s=={0x%lx:0x%lx}==%c%s",
						d2str(a,"%dhex", NULL), aa.handle.address, aa.handle.type,
						(usg)? '`' : '&',
						aaaf->name
					);
					if( aaaf->type== _ascanf_array ){
						fprintf( StdErr, "[%d:%d]", aaaf->last_index, aaaf->N );
						if( aaaf->sourceArray ){
							  /* 20050310: dump size verification: we ought to store the subset start... */
							if( !aaaf->sourceArray->name || !(aaaf->sourceArray->array || aaaf->sourceArray->iarray)
								|| aaaf->N> aaaf->sourceArray->N
							){
								fprintf( StdErr, "<invalid subset array: rejected!!>" );
								fflush(StdErr);
								aa.handle.address = 0, aaaf= NULL;
								return(NULL);
							}
							else{
								fprintf( StdErr, "<subset of %s[%d]>", aaaf->sourceArray->name, aaaf->sourceArray->N );
							}

						}
					}
					if( usg || aaaf->fp ){
						fprintf( StdErr, ":\"%s\"", (aaaf->usage)? aaaf->usage : "" );
					}
					if( aaaf->fp ){
						fprintf( StdErr, "<open file>" );
					}
					if( aaaf->cfont ){
						fprintf( StdErr, "<CustomFont>" );
					}
					fputs( ")== ", StdErr );
				}
				if( AlwaysUpdateAutoArrays ){
					if( aaaf->type== _ascanf_array && aaaf->procedure && (this_type== 0 || this_type== _ascanf_array) &&
						aaaf->procedure->list_of_constants>= 0
					){
					  int n= aaaf->N;
					  int level= 0;
						if( pragma_unlikely(verbose>1) ){
							fprintf( StdErr, " (updating automatic array \"%s\", expression %s) ->\n",
								aaaf->name, aaaf->procedure->expr
							);
						}
						aaaf->procedure->level+= 1;
						_compiled_fascanf( &n, aaaf->procedure->expr, aaaf->array, NULL, NULL, NULL, &aaaf->procedure, &level );
						aaaf->procedure->level-= 1;
						fputs( "#\t== ", StdErr );
					}
				}
			}
		}
		else{
			if( pragma_unlikely( (verbose> 0 /* || debugFlag */) && !(ascanf_popup && strcmp(caller, "simple_array_op")==0) )  ){
#ifdef USE_AA_REGISTER
				  // RJVB 20080924: only print a warning in case of a type mismatch and an otherwise valid ascanf_address:
				if( pragma_unlikely( verify_ascanf_Address( (address32) aa.handle.address,type) ) )
#endif
				{
					fprintf( StdErr, "parse_ascanf_address(%s,0x%lx,\"%s\"): invalid type 0x%lx \"%s\"; \"%s\" expected\n",
						d2str(a,"%dhex", NULL), this_type, caller, type,
						AscanfTypeName(type), (this_type==0)? "any type" : AscanfTypeName(this_type)
					);
				}
			}
			aa.handle.address = 0, aaaf= NULL;
		}
		parsing_address= pa;
	}
	else{
		aaaf= NULL;
	}
#ifdef DEBUG
	if( aaaf && warn ){
		fprintf( StdErr, "\tfound symbol 0x%lx==\"%s\"\n", aaaf,
			(aaaf->name)? aaaf->name : "(null name)"
		);
	}
#endif
	return( aaaf );
}

#ifdef SAFE_FOR_64BIT

	  // This function's code is a bit of a mess ... the point is to find an acceptably safe approach to map
	  // a 64bit pointer to a unique 32bit value... See tim-asc-parm.c for more attempts.
	static inline address32  AAF_REPRESENTATION( ascanf_Function *af )
	{ address32 repr= 0;
// 	  char *ptr;
// 	  int i, step= sizeof(void*)/4, shift= 24;

#if 0
	  long long lptr= 0x1122334455667788;
		lptr= lptr << 4;
		ptr= (char*) &lptr ;
		for( i= 0; i< sizeof(long long); i+= 2, shift-= 8 ){
			repr |= ((ptr[i] & 0x000000ff) << shift);
		}
#	ifdef DEBUG
		fprintf( StdErr, "%d repr(%llx)=0x%lx\n", sizeof(lptr), lptr, repr );
#	endif
		repr= 0, shift= 24;
#endif

// 		if( step> 1 ){
// 			af= (ascanf_Function*) ( ((unsigned long long) af) << 4 );
// 		}
// 		ptr= (char*) &af;
// 		for( i= 0; i< sizeof(void*); i+= step, shift-= 8 ){
// 			repr |= ((ptr[i] & 0x000000ff) << shift);
// 		}

		if( sizeof(void*)> 4 ){
		  address32 *af32= (address32*) &af;
		  union addr64conv {
			  ascanf_Function *addr64;
			  uint8 byte[8];
			  struct{
				  uint16 prefix1;
				  uint8 prefix2;
				  address32 addr32;
				  uint8 suffix;
			  } conv;
		  } addr64;
			addr64.addr64 = af;
#	ifdef DDEBUG
			if( !aaf->own_address ){
				fprintf( StdErr, "0x%lx + 0x%lx ",
					af32[0], af32[1]
				);
			}
#	endif
			repr = (address32)( ((uint64)af32[0] + (uint64)af32[1]) / 2 );
//			repr= (af32[0] << 1) | (af32[1] >> 1);
//			repr = addr64.conv.addr32;
//			repr = (((address32)addr64.byte[0]) << 24) | (((address32)addr64.byte[2]) << 16)
//				| (((address32)addr64.byte[5]) << 8) | ((address32)addr64.byte[7]);
		}
		else{
		  address32 *af32= (address32*) &af;
			repr = af32[0];
		}

#	ifdef DDEBUG
		if( !aaf->own_address ){
			if( sizeof(af)== 4 ){
				fprintf( StdErr, "repr(%p)=0x%lx\n", af, repr );
			}
			else{
				fprintf( StdErr, "repr(0x%llx)=0x%lx\n", af, repr );
			}
		}
#	endif
		return(repr);
	}
#else
#	define AAF_REPRESENTATION(af)	(address32)(af)
#endif

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
double take_ascanf_address( ascanf_Function *af )
{ AscanfAddresses aa;
	if( af ){
	  long mask= 0x12345600;
		if( af->take_usage ){
			mask+= 0x10000000;
		}
		aa.handle.type= mask + af->type;
		aa.handle.address= AAF_REPRESENTATION(af);
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "#\t\t%c%s={0x%lx:0x%lx|0x%lx}==%s\n",
				(af->take_usage)? '`' : '&',
				af->name, af, mask, af->type, d2str(aa.value,"%dhex",NULL)
			);
		}
		if( af->own_address!= aa.value ){
			af->own_address= aa.value;
#ifdef USE_AA_REGISTER
#	ifdef SAFE_FOR_64BIT
			register_ascanf_Address(af, aa.handle.address);
#	else
			register_ascanf_Address(af);
#	endif
#endif
		}
		return( aa.value );
	}
	else{
		return(0);
	}
}

#define D2STR_BUFLEN	DBL_MAX_10_EXP+12

/* 20010706: a routine that handles the conversion of a double to a string.
 \ If the double is in fact a pointer to an ascanf (string) variable,
 \ an attempt is made to nicely print the name or contents of that variable,
 \ with a proper prefix (& or `) and quotes. If <buf>==NULL, a stub call
 \ to d2str() will obtain a d2str() buffer, and store the results in that arena
 \ (this functionality is thus not very time-efficient!). Otherwise, the result
 \ is stored in the provided buffer.
 \ If the result would exceed that size, the %dhex format is used instead.
 \ This routine can be used outside of ascanfc.c and ascanfc2.c; it is indeed
 \ suggested to use it everywhere where ascanf variables are to be printed.
 \ 20040602: target string is passed as a handle, **Buf. If Buf==NULL, a d2str
 \ buffer is used. Otherwise, if the resulting text fits within *Buf (which must
 \ therefore a valid string; its length serves as the criterion), the passed arena
 \ is used; if not, newly allocated memory is used and returned in *Buf. Thus, after
 \ use, *Buf can always be deallocated unless it is equal to the passed pointer in case
 \ that is a static string. If not equal to the passed pointer, two deallocations may
 \ be necessary. This is like the SubstituteOpcodes function.
 */

char *ad2str( double val, const char *format, char **Buf )
{  int ugi= use_greek_inf, take_usage= 0, dealloc= False, auaa= AlwaysUpdateAutoArrays;
   char *ret, *text= NULL, *buf= NULL;
   ascanf_Function *af;
	use_greek_inf= ascanf_use_greek_inf;
	AlwaysUpdateAutoArrays= False;
	if( Buf ){
		buf= *Buf;
	}
	if( val && (af= parse_ascanf_address( val, 0, "ad2str", False, &take_usage ))){
/* 		format= "%dhex";	*/
		if( take_usage && !af->user_internal ){
			if( af->internal ){
				sprint_string2( &text, "`\"", "\"", af->usage, True );
				if( !Buf && strlen(text)>= D2STR_BUFLEN ){
					xfree(text);
					format= "%dhex";
				}
				else{
					dealloc= True;
				}
			}
			else{
				text= concat( "`", af->name, NULL );
				if( !Buf && strlen(text)>= D2STR_BUFLEN ){
					xfree(text);
					format= "%dhex";
				}
				else{
					dealloc= True;
				}
			}
		}
		else{
			text= concat( "&", af->name, NULL );
			if( !Buf && strlen(text)>= D2STR_BUFLEN ){
				xfree(text);
				format= "%dhex";
			}
			else{
				dealloc= True;
			}
		}
	}
	else if( !format ){
		format= d3str_format;
	}
	if( text ){
		if( !Buf ){
			  /* just get a d2str buffer... 	*/
			ret= d2str( 0, 0,0);
			strcpy( ret, text );
		}
		else{
			if( strlen(text) > strlen(*Buf) ){
				*Buf= text;
				dealloc= False;
			}
			else{
				strcpy( *Buf, text );
			}
			ret= *Buf;
		}
	}
	else{
		if( Buf ){
			if( strlen(*Buf)< D2STR_BUFLEN ){
				ret= *Buf= strdup(d2str(val,format,NULL));
			}
			else{
				ret= d2str(val,format, *Buf);
			}
		}
		else{
			ret= d2str(val,format,NULL);
		}
	}
	if( dealloc ){
		xfree( text );
	}
	use_greek_inf= ugi;
	AlwaysUpdateAutoArrays= auaa;
	return( ret );
}

double HandlerParameters[8];

extern int evaluate_procedure( int *n, ascanf_Function *proc, double *args, int *level );

/* 20031012: added a result argument */
double AccessHandler( ascanf_Function *af, char *caller, int *level, Compiled_Form *form, char *expr, double *result )
{
	if( af && af->accessHandler ){
	  ascanf_Function *aH= af->accessHandler;
	  int idx, uaA= ascanf_update_ArgList;
	  double aH_par[2];
#ifdef ASCANF_ALTERNATE
	  ascanf_Callback_Frame frame;
#endif
		if( af->aH_flags[1] && af->value== af->old_value ){
			return(0);
		}
		if( af->aH_flags[2] ){
			switch( af->aH_flags[2] ){
				default:
				case 1:
					if( af->value!= af->aH_par[2] ){
						return(0);
					}
					break;
				case 2:
					if( af->value< af->aH_par[2] ){
						return(0);
					}
					break;
				case -2:
					if( af->value> af->aH_par[2] ){
						return(0);
					}
					break;
			}
		}

		  /* Don't change the $ArgList variable, but also make sure it
		   \ will not be changed within this scope.
		   */
		ascanf_update_ArgList= False;
		  /* Ensure bluntly that we will never get closed loops...	*/
		af->accessHandler= NULL;
		HandlerParameters[0]= af->aH_par[0];
		HandlerParameters[1]= af->aH_par[1];
		{ int avb= ascanf_verbose;
			ascanf_verbose= 0;
			HandlerParameters[2]= take_ascanf_address( af );
			HandlerParameters[6]= take_ascanf_address( aH );
			ascanf_verbose= avb;
		}
		HandlerParameters[3]= af->value;
		  /* 20000512: old_value is only changed/updated in this routine. And only at this
		   \ point (before the update) does it contain the variable's old value.
		   */
		HandlerParameters[4]= af->old_value;
		HandlerParameters[5]= (af->type== _ascanf_array)? af->last_index : -1;
		HandlerParameters[7]= *level;
		if( af->aH_flags[0] ){
		  char hdr[128];
			fprintf( StdErr, "\nAccessHandler[%s]<-%s: %s=%s, level=%d, value old,new=%s,%s",
				af->name, caller, aH->name, ad2str( aH->value, d3str_format, 0), *level,
				ad2str( af->old_value, d3str_format,0), ad2str( af->value, d3str_format,0)
			);
			if( af->aH_flags[1] ){
				fputs( " (changed)", StdErr );
			}
			if( af->aH_flags[2] ){
			  char *c;
				switch( af->aH_flags[2] ){
					default:
					case 1:
						c= "equals";
						break;
					case 2:
						c= "is greater than";
						break;
					case 3:
						c= "is less than";
						break;
				}
				fprintf( StdErr, "; new value %s ref.value %s", c, ad2str( af->aH_par[2], d3str_format,0) );
			}
			fputc( '\n', StdErr );
			if( form ){
				if( form->top!= form ){
					fprintf( StdErr, "\twhile evaluating:\n" );
					sprintf( hdr, "#C%d: \t    ", (*level)+1 );
					fprintf( StdErr, "#C%d: \t%s[\n%s", (*level), form->fun->name, hdr );
					  /* Here we must print <form>, since *s only contains the toplevel expression	*/
					Print_Form( StdErr, &(form->args), 1, True, hdr, NULL, "\n", True );
					fprintf( StdErr, "#C%d: \t] in toplevel expression\n", (*level) );
				}
				strcpy( hdr, "#C0: \t    " );
				if( form->top ){
					fputs( "#C0:  \t", StdErr );
					Print_Form( StdErr, &(form->top), 1, True, hdr, NULL, "\n", True );
					fputs( "#C0:  \tContext: ", StdErr );
				}
				else{
					fprintf( StdErr, "%s<UNKNOWN>", hdr );
				}
			}
			else if( expr ){
				fprintf( StdErr, "\twhile evaluating expression:" );
				PF_print_string( StdErr, "\n ", "\n", expr, False );
			}
			else{
				fprintf( StdErr, "\t<context unknown>\n" );
			}
			if( TBARprogress_header ){
				fprintf( StdErr, "\tTBAR: %s\n", TBARprogress_header );
			}
		}
		switch( aH->type ){
			case _ascanf_simplestats:
			case _ascanf_simpleanglestats:
				/* For the time being! Information could be given about the previous and actual number of observations. */
			case _ascanf_python_object:
			case _ascanf_variable:
				aH->value= (NaN(af->aH_par[0]))? 1 : af->aH_par[0];
				aH->assigns+= 1;
				aH->last_index= af->last_index;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (AccessHandler[%s]<-%s: %s=%g)== ",
						af->name, caller, aH->name, aH->value
					);
				}
				af->old_value= af->value;
				if( aH->accessHandler ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (nested AccessHandler!) " );
					}
					AccessHandler( aH, aH->name, level, form, expr, result );
				}
				break;
			case _ascanf_array:
				aH_par[0]= (NaN(af->aH_par[0]))? -1 : af->aH_par[0];
				aH_par[1]= (NaN(af->aH_par[1]))? 1 : af->aH_par[1];
				if( aH_par[0]< 0 ){
					idx= MAX( 0, aH->last_index );
				}
				if( aH_par[0]>= aH->N ){
					idx= aH->N- 1;
				}
				else{
					idx= (int) aH_par[0];
				}
				if( aH->iarray ){
					aH->value= aH->iarray[idx]= aH_par[1];
				}
				else{
					aH->value= aH->array[idx]= aH_par[1];
				}
				aH->last_index= idx;
				aH->assigns+= 1;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (AccessHandler[%s]<-%s: %s[%d]=%g)== ",
						af->name, caller, aH->name, idx, aH->value
					);
				}
				af->old_value= af->value;
				if( aH->accessHandler ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (nested AccessHandler!) " );
					}
					AccessHandler( aH, aH->name, level, form, expr, result );
				}
				break;
			case _ascanf_procedure:{
			  int n= 1;
			  double arg[1];
			  Compiled_Form *par;
				if( form && aH->procedure->parent ){
					  /* reparent the procedure's form so that its grandparent is the same
					   \ as the parent of the current form. This is as it should be - and
					   \ that is slightly different than when a procedure is called regularly!!
					   */
					par= aH->procedure->parent->parent;
					aH->procedure->parent->parent= form->parent;
				}
				else{
					par= NULL;
				}
				aH_par[0]= (NaN(af->aH_par[0]))? 1 : af->aH_par[0];
				aH_par[1]= (NaN(af->aH_par[1]))? 1 : af->aH_par[1];
				arg[0]= aH_par[1];
				evaluate_procedure( &n, aH, &arg[0], level );
				if( n== 1 ){
					aH->value= arg[0];
					aH->assigns+= 1;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (AccessHandler[%s]<-%s: %s=%g)== ",
							af->name, caller, aH->name, aH->value
						);
					}
				}
				af->old_value= af->value;
				if( par ){
					aH->procedure->parent->parent= par;
				}
				break;
			}
			case NOT_EOF:
			case NOT_EOF_OR_RETURN:
			case _ascanf_function:{
			  double arg[3], res= 0;
			  int n, aa= ascanf_arguments;
				aH_par[0]= (NaN(af->aH_par[0]))? 1 : af->aH_par[0];
				aH_par[1]= (NaN(af->aH_par[1]))? 0 : af->aH_par[1];
				arg[0]= aH_par[0];
				arg[1]= aH_par[1];
				// 20080706: also pass a pointer to the modified variable
				if( !af->own_address ){
					take_ascanf_address(af);
				}
				arg[2]= af->own_address;
				ascanf_arguments= 3;
#ifdef ASCANF_ALTERNATE
				frame.args= arg;
				frame.result= &res;
				frame.level= level;
				frame.compiled= form;
				frame.self= af;
#if defined(ASCANF_ARG_DEBUG)
				frame.expr= NULL;
#endif
				n= (*aH->function)( &frame );
#else
				n= (*aH->function)( arg, &res, level );
#endif
				aH->value= res;
				if( n && !ascanf_arg_error ){
					aH->value= res;
					aH->assigns+= 1;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (AccessHandler[%s]<-%s: %s[%g,%g]=%g)== ",
							af->name, caller, aH->name, aH_par[0], aH_par[1], aH->value
						);
					}
				}
				ascanf_arguments= aa;
				af->old_value= af->value;
				if( aH->accessHandler ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (nested AccessHandler!) " );
					}
					AccessHandler( aH, aH->name, level, form, expr, result );
				}
				break;
			}
			default:
				fprintf( StdErr, " (AccessHandler[%s->%s]<-%s: removed pointer to obsolete/invalid handler)== ",
					af->name, aH->name, caller
				);
				aH= NULL;
				break;
		}

		if( af== ascanf_d3str_format ){
		  extern int d3str_format_changed;
			if( af->usage ){
				if( strcmp(d3str_format, af->usage) ){
					strncpy( d3str_format, af->usage, sizeof(d3str_format) );
					d3str_format_changed+= 1;
				}
			}
			else{
				  /* 20030328: ==0 is missing from the test below??!! */
				if( strcmp( d3str_format, "%20g") ){
					sprintf( d3str_format, "%%.%dg", DBL_DIG+1 );
					d3str_format_changed+= 1;
				}
				af->usage= strdup(d3str_format);
			}
			ascanf_d3str_format->is_usage= True;
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (double printing format set to \"%s\")== ", d3str_format );
			}
		}
		else if( af== af_verbose ){
			if( !NaN(*ascanf_verbose_value) ){
				ascanf_verbose= (int) *ascanf_verbose_value;
			}
			else{
				ascanf_verbose= 0;
			}
		}
		else if( af== af_AllowSimpleArrayOps ){
			if( !NaN(*AllowSimpleArrayOps_value) ){
				AllowSimpleArrayOps= (int) *AllowSimpleArrayOps_value;
			}
			else{
				AllowSimpleArrayOps= 0;
			}
		}
		else if( af== af_AllowArrayExpansion ){
			if( !NaN(*AllowArrayExpansion_value) ){
				AllowArrayExpansion= (int) *AllowArrayExpansion_value;
			}
			else{
				AllowArrayExpansion= 0;
			}
		}
		else if( af== af_AlwaysUpdateAutoArrays ){
			if( !NaN(*AlwaysUpdateAutoArrays_value) ){
				AlwaysUpdateAutoArrays= (int) *AlwaysUpdateAutoArrays_value;
			}
			else{
				AlwaysUpdateAutoArrays= 0;
			}
		}
		else if( af== af_AllowProcedureLocals ){
			if( !NaN(*AllowProcedureLocals_value) ){
				AllowProcedureLocals= (int) *AllowProcedureLocals_value;
			}
			else{
				AllowProcedureLocals= 0;
			}
		}
		else if( af== af_Find_Point_exhaustive ){
			if( !NaN(*Find_Point_exhaustive_value) ){
				Find_Point_use_precision= ! (int) *Find_Point_exhaustive_value;
			}
			else{
				Find_Point_use_precision= 0;
			}
		}
		else if( af== ascanf_d2str_NaNCode ){
			PrintNaNCode= (ASCANF_TRUE(ascanf_d2str_NaNCode->value))? True : False;
		}
		else if( af== ascanf_Dprint_fp ){
		  ascanf_Function *targ;
		  extern FILE *Dprint_fp;
			if( !(targ= parse_ascanf_address(af->value, 0, "AccessHandler($Dprint-fp)", False, NULL)) ){
				switch( (int) af->value ){
					case 1:
						Dprint_fp= stdout;
						break;
					case 2:
						Dprint_fp= StdErr;
						break;
					default:
						Dprint_fp= NULL;
						break;
				}

			}
			else if( targ->fp ){
				Dprint_fp= targ->fp;
			}
			else{
				Dprint_fp= NULL;
			}
			af->fp= register_FILEsDescriptor(Dprint_fp);
			if( ascanf_verbose ){
				fprintf( StdErr, "%s==%s => file 0x%lx\n", af->name, ad2str( af->value, d3str_format, 0), Dprint_fp );
			}
		}
		else if( af== af_setNumber ){
		  extern ascanf_Function *af_numPoints;
			CLIP_EXPR( *ascanf_setNumber, ssfloor(*ascanf_setNumber), 0, setNumber-1 );
			af->value= *ascanf_setNumber;
			af_numPoints->value= *ascanf_numPoints= AllSets[ (int)*ascanf_setNumber ].numPoints;
		}
		else if( af== af_Counter ){
		  int idx= (int) *ascanf_setNumber;
			if( idx>= 0 && idx< setNumber ){
			  int pnt_nr;
			  DataSet *this_set= &AllSets[idx];
				CLIP_EXPR( *ascanf_Counter, ssfloor(*ascanf_Counter), 0, MAX(0,this_set->numPoints) );
				pnt_nr= (int) (af_Counter->value= *ascanf_Counter);
				if( this_set->numPoints> 0 ){
					if( !ascanf_data_buf ){
						ascanf_data_buf= ADB;
					}
					ascanf_data_buf[0]= af_data0->value= *ascanf_data0= XVAL(this_set,pnt_nr);
					ascanf_data_buf[1]= af_data1->value= *ascanf_data1= YVAL(this_set,pnt_nr);
					ascanf_data_buf[2]= af_data2->value= *ascanf_data2= EVAL(this_set,pnt_nr);
					if( this_set->lcol>= 0 ){
						ascanf_data_buf[3]= af_data3->value= *ascanf_data3= VVAL(this_set,pnt_nr);
					}
				}
			}
		}
		else if( af== af_interrupt ){
			if( !NaN(*ascanf_interrupt_value) ){
				ascanf_interrupt= (int) *ascanf_interrupt_value;
				if( Evaluating_Form && ascanf_interrupt && (ascanf_verbose || debugFlag || scriptVerbose) ){
					fprintf( StdErr, "ascanf::%s: interrupt requested.\n", caller );
					Print_Form( StdErr, &Evaluating_Form, 0, True, NULL, "#\t", NULL, False );
				}
			}
			else{
				ascanf_interrupt= 0;
			}
		}
		else if( af== af_escape ){
			if( !NaN(*ascanf_escape_value) ){
				ascanf_escape= (int) *ascanf_escape_value;
				if( Evaluating_Form && ascanf_escape && (ascanf_verbose || debugFlag || scriptVerbose) ){
					fprintf( StdErr, "ascanf::%s: escape requested.\n", caller );
					Print_Form( StdErr, &Evaluating_Form, 0, True, NULL, "#\t", NULL, False );
				}
			}
			else{
				ascanf_escape= 0;
			}
		}

		if( result ){
		  /* 20031012: if the result argument was specified, set it to af->value (to ensure that e.g.
		   \ foo[expr] returns the true value that foo would return upon a future invocation, and not
		   \ the value returned by expr if ever that value was altered by the acceshandler. (See the
		   \ af_setNumber case above.
		   */
			*result= af->value;
		}

		  /* Restore the access handler:	*/
		af->accessHandler= aH;
		ascanf_update_ArgList= uaA;
		return( aH->value );
	}
	return(0);
}

/* SimpleList: one in which no element has more than 2 argumentlists	*/
Boolean SimpleList( Compiled_Form *form )
{ Boolean simple= True;
	while( form && simple ){
		if( form->argc> 2 ){
			simple= False;
		}
		else if( form->args ){
			simple= SimpleList( form->args );
		}
		form= form->cdr;
	}
	return( simple );
}

#define PREFIXES	"!*-&`?"

#define XF_PREFIX(form) if( form->sign< 0 ){\
		strcat( buf, "-" );\
	}\
	else if( form->negate ){\
		strcat( buf, "!" );\
	}\
	if( form->store ){\
		strcat( buf, "*" );\
	}\
	if( form->take_address ){\
		strcat( buf, (form->take_usage)? "`" : "&" );\
	}

#define XF_PREFIX_SUB(form) if( form->sign< 0 ){\
		strcat( buf, "-" );\
	}\
	else if( form->negate ){\
		strcat( buf, "!" );\
	}\
	if( form->take_address ){\
		strcat( buf, (form->take_usage)? "`" : "&" );\
	}

static char *AF_prefix( ascanf_Function *form)
{ static char buf[16];
	buf[0]= '\0';
	XF_PREFIX(form);
	return( buf );
}

static char *AF_prefix_sub( ascanf_Function *form)
{ static char buf[16];
	buf[0]= '\0';
	XF_PREFIX_SUB(form);
	return( buf );
}

static char *CF_prefix( Compiled_Form *form)
{ static char _buf[2][16];
  static int nr= 0;
  char *buf= _buf[nr];
	buf[0]= '\0';
	XF_PREFIX(form);
	if( form->last_value ){
		strcat( buf, "?" );
	}
	nr= (nr + 1) % 2;
	return( buf );
}

static char *CF_prefix_sub( Compiled_Form *form)
{ static char buf[16];
	buf[0]= '\0';
	XF_PREFIX_SUB(form);
	if( form->last_value ){
		strcat( buf, "?" );
	}
	return( buf );
}

int PF_Sprint_string( Sinc *fp, char *header, char *trailer, char *String, int instring )
{ int len= 0;
  int hlen= (header)? strlen(header) : 0;
  char *string= (String)? String : "", *c= string;
	Sputs( header, fp);
	len+= hlen;
	while( *c ){
		if( *c== '"' && (c== string || c[-1]!= '\\') ){
			instring= !instring;
		}
		if( isprint(*c) ){
			Sputc( *c, fp );
			len+= 1;
		}
		else if( instring ){ switch( *c ){
			case '\n':
				Sputs( "#xn", fp );
				len+= 3;
				break;
			case '\r':
				Sputs( "#xr", fp );
				len+= 3;
				break;
			case '\t':
				Sputs( "#xt", fp );
				len+= 3;
				break;
			default:{
			  char buf[128];
				len+= sprintf( buf, "#x%02x", 0x00ff & *c );
				Sputs( buf, fp );
				break;
			}
		} }
		else{
			switch( *c ){
				case '\n':
				case '\r':
				case '\t':
					Sputc( *c, fp );
					len+= 1;
					break;
				default:{
				  char buf[128];
					len+= sprintf( buf, "#x%02x", 0x00ff & *c );
					Sputs( buf, fp );
					break;
				}
			}
		}
		c++;
	}
	len+= Sputs( trailer, fp );
	Sflush( fp );
	return( len );
}

int PF_print_string( FILE *fp, char *header, char *trailer, char *String, int instring )
{ Sinc sinc;
	return( PF_Sprint_string( Sinc_file(&sinc, fp, 0,0), header, trailer, String, instring ) );
}

int PF_sprint_string( char **target, char *header, char *trailer, char *String, int instring )
{ Sinc sinc;
  int r;
	sinc.sinc.string= NULL;
	Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
	Sflush( &sinc );
	r= PF_Sprint_string( &sinc, header, trailer, String, instring );
	*target= sinc.sinc.string;
	return(r);
}

/* Print a Compiled_Form. When pp==True, make an attempt at prettyprinting the thing. Undoubtedly still
 \ buggy - I was so foolish as to try to implement an algorithm to my own taste...
 \ It is also undoubtedly way to complex - I have this impression a lot of the local/static variables
 \ can be deleted by some rewriting of the code. Doesn't have a high priority to me.
 */
int Print_Form( FILE *fp, Compiled_Form **Form, int print_results, int pp, char *whdr, char *wpref, char *wrapstr, int ff )
{  static long l= 0;
   static long tlen= 0;
   static int wlevel, level= 0;
   int i, justwrapped= 0, wl0= -1;
   static int unwrapped= 0, output= 0, subwrapped;
   Compiled_Form *form;
	if( Form ){
	   static char *bold="", *nobold="", *underline="", *nounderline="";
	   static int lnbold=0, lnnobold=0, lnunderline=0, lnnounderline=0;
	   char asep[3];
		// 20101113: honour the ascanf_separator setting!!
		asep[0] = ascanf_separator, asep[1] = ' ', asep[2] = '\0';
#ifdef _UNIX_C_
		if( level== 0 ){
		 char *TERM=cgetenv("TERM");
			if( TERM && fp && isatty(fileno(fp)) ){
				if( strncaseeq( TERM, "xterm", 5) || strncaseeq( TERM, "vt1", 3) ||
					strncaseeq( TERM, "vt2", 3) || strncaseeq( TERM, "cygwin", 6)
				){
					bold= "\033[1m";
					nobold= "\033[m";
					underline= "\033[4m";
					nounderline= nobold;
				}
				else if( strncaseeq( TERM, "hp", 2) ){
					bold= "\033&dB";
					nobold= "\033&d@";
					underline= "\033&dD";
					nounderline= nobold;
				}
			}
			else{
				bold="";
				nobold="";
				underline="";
				nounderline="";
			}
			lnbold= strlen(bold);
			lnnobold= strlen(nobold);
			lnunderline= strlen(underline);
			lnnounderline= strlen(nounderline);
			wlevel= 1;
			unwrapped= output= subwrapped= 0;
			l= tlen= 0;
			if( pp ){
				if( !wpref ){
					wpref= "    ";
				}
				if( !whdr ){
					whdr= wpref;
				}
				if( !wrapstr ){
					wrapstr= "\\n\n";
				}
			}
		}
#endif
		form= *Form;
		level++;
		wl0= wlevel;
		while( form ){
		  char *fn, *fname= FUNNAME(form->fun), *val= ad2str( form->value, d3str_format, NULL);
		  int wrapped= 0;
		  Boolean sL;
			if( pp ){
			  int len= 8+ (fname)? strlen(fname) : strlen(val);
				if( l+ len>= 100 ){
					fflush(fp);
					l+= fprintf( fp, wrapstr );
					tlen+= l;
					l= 0;
					for( i= 0; i< wlevel; i++ ){
						l+= fprintf( fp, (i)? wpref : whdr );
					}
					justwrapped= 1;
					subwrapped+= 1;
					output= 0;
				}
				sL= SimpleList( form->args );
			}
			if( form->fun ){
				  /* 20010205: Methinks that fname does not depend on the prefix given...
					fn= (form->sign<0 || form->negate || form->store || form->last_value)? &fname[1] : fname;
				   */
				fn= fname;
				if( pp && (strncmp( fn, "ifelse", 6)==0 || strncmp( fn, "whiledo", 7)== 0 ||
					strncmp( fn, "dowhile", 7)== 0 || strncmp( fn, "for-to", 6)== 0 ||
					strncmp( fn, "switch", 6)== 0)
				){
					wl0= wlevel;
/* 					wlevel+= 1;	*/
					if( output && !justwrapped && wlevel!= level+1 && !sL ){
						fflush(fp);
						l+= fprintf( fp, wrapstr );
						tlen+= l;
						l= 0;
						for( i= 0; i< wlevel; i++ ){
							l+= fprintf( fp, (i)? wpref : whdr );
						};
						wlevel+= 1;
						unwrapped-= 1;
						wrapped= 1;
						output= 0;
					}
					else{
						wlevel+= 1;
						wrapped= -1;
					}
					subwrapped= 0;
				}
				l+= fprintf( fp, "%s%s", bold, CF_prefix(form) );
				if( form->fun && form->fun->function==ascanf_nDindex && strcmp(fname,"nDindex")==0 ){
					l+= PF_print_string( fp, "", nobold, "@", False );
				}
				else{
					l+= PF_print_string( fp, "", nobold, fname, False );
				}
				l-= lnbold - lnnobold;
				output= 1;
			}
			else{
				fn= NULL;
				l+= fprintf( fp, "%s%s%s", bold, val, nobold )- lnbold - lnnobold;
				output= 1;
			}
			if( print_results && form->expr ){
				l+= fprintf( fp, "(%s)", form->expr );
				output= 1;
			}
			if( form->args ){
			  int wl, wll= 0;
				if( print_results ){
					l+= fprintf( fp, "#%d[", form->argc );
					if( form->args->list_of_constants ){
						l+= fprintf( fp, "<C>");
					}
					output= 1;
				}
				else{
					fputc( '[', fp);
					output= 1;
					l+= 1;
					if( whdr && whdr[0]== '#' && form->args->list_of_constants ){
						l+= fprintf( fp, "<C>");
					}
				}
				wl= wlevel;
				if( fn ){
					if( pp ){
					  Boolean ww= 0;
						if( (strncmp( fn, "progn", 5)==0 || strncmp( fn, "verbose", 7)== 0 ||
							strncmp( fn, "Eprint", 6)== 0) || strncmp( fn, "Dprint", 6)== 0 ||
							strncmp( fn, "compile", 7)== 0
						){
							if( output && !justwrapped ){
								if( !sL ){
									fflush(fp);
									l+= fprintf( fp, wrapstr );
									tlen+= l;
									l= 0;
									wlevel+= 1;
									for( i= 0; i< wlevel; i++ ){
										l+= fprintf( fp, (i)? wpref : whdr );
									}
									unwrapped-= 1;
									wrapped= 1;
									output= 0;
									subwrapped+= 1;
									ww= True;
								}
								else{
									wlevel+= 1;
									wll= 1;
								}
							}
							else{
								if( justwrapped ){
									wlevel+= 1;
								}
								else{
									wrapped= -1;
								}
							}
						}
						if( !ww && form->args->args ){
							fputc( ' ', fp );
							l+= 1;
						}
					}
				}
				Print_Form( fp, &(form->args), print_results, pp, whdr, wpref, wrapstr, ff );
				if( pp ){
				  Boolean ww= False;
					if( wlevel> wl && output ){
						fflush(fp);
						if( wlevel> wl+ wll ){
							l+= fprintf( fp, wrapstr );
							tlen+= l;
							l= 0;
							if( wrapped!= -1 ){
								wlevel-= 1;
							}
							for( i= 0; i< wlevel; i++ ){
								l+= fprintf( fp, (i)? wpref : whdr );
							}
							unwrapped+= 1;
							output= 0;
							wrapped= 0;
							sL= False;
							ww= True;
						}
/* 						subwrapped+= 1;	*/
					}
					wlevel= wl;
					if( ((level> 1 && !unwrapped && fn) || wrapped) && (output || (wl0>= 0 && wlevel> wl0)) ){
					  Boolean unwrap= False;
						wlevel= wl0;
						if(strncmp( fn, "ifelse", 6)==0 || strncmp( fn, "whiledo", 7)== 0 ||
							strncmp( fn, "dowhile", 7)== 0 || strncmp( fn, "for-to", 6)== 0
						){
/* 							wlevel-= 1;	*/
							if( (!sL && wlevel!= wl0) || subwrapped ){
								unwrap= True;
								subwrapped-= 1;
							}
							else{
								unwrap= False;
								if( wrapped== 1 ){
									wlevel-= 2;
								}
								wrapped= 0;
							}
						}
						else if( strncmp( fn, "progn", 5)==0 || strncmp( fn, "verbose", 7)== 0 ||
							strncmp( fn, "Eprint", 6)== 0 || strncmp( fn, "Dprint", 6)== 0
						){
							if( !sL ){
								unwrap= True;
							}
							else{
								unwrap= False;
								if( wrapped== 1 ){
									wlevel-= 1;
								}
								wrapped= 0;
							}
						}
						if( unwrap || wrapped ){
							fflush(fp);
							l+= fprintf( fp, wrapstr );
							tlen+= l;
							l= 0;
							if( wrapped!= -1 ){
								wlevel-= 1;
							}
							for( i= 0; i< wlevel; i++ ){
								l+= fprintf( fp, (i)? wpref : whdr );
							}
							unwrapped+= 1;
							wrapped= 0;
							output= 0;
							ww= True;
						}
					}
					if( !ww && form->args->args ){
						fputc( ' ', fp );
						l+= 1;
					}
				}
				l+= fprintf( fp, "]");
				output= 1;
				if( print_results ){
					l+= fprintf( fp, "%s==%s%s", underline, ad2str( form->value, d3str_format, NULL), nounderline )-
						lnunderline- lnnounderline;
				}
			}
			else if( form->fun && print_results ){
				l+= fprintf( fp, "%s==%s%s", underline, ad2str( form->value, d3str_format, NULL), nounderline )-
					lnunderline- lnnounderline;
				output= 1;
			}
			form= form->cdr;
			if( form ){
				fputs( asep, fp);
				output= 1;
				l+= 2;
			}
			justwrapped= 0;
		}
		if( wl0>= 0 && wlevel!= wl0 ){
			if( wlevel< wl0 ){
				wlevel= wl0;
			}
			else{
				wlevel= wl0;
				fflush(fp);
				l+= fprintf( fp, wrapstr );
				tlen+= l;
				l= 0;
				for( i= 0; i< wlevel; i++ ){
					l+= fprintf( fp, (i)? wpref : whdr );
				}
				unwrapped+= 1;
				output= 0;
			}
		}
		level--;
		if( level== 0 && ff ){
			if( pp ){
				l+= fprintf( fp, wrapstr );
			}
			else{
				fputc( '\n', fp);
				l+= 1;
			}
		}
		fflush( fp );
		return( (level)? l : l+tlen );
	}
	level= 0;
	return( (level)? l : l+tlen );
}

int Check_Form_Dependencies( Compiled_Form **Form )
{ static int level= -1;
  Compiled_Form *form;
  int n= 0;
	if( Form && (form= *Form) ){
		level+= 1;
		while( form ){
			if( form->fun ){
				switch( form->fun->type ){
					case _ascanf_novariable:
						if( form->type== _ascanf_array && form->fun->N< 0 ){
						  char *command= NULL, Nbuf[128], *usage= form->fun->usage;
						  int n= 1;
						  double r= 0;
							sprintf( Nbuf, ",%d,", -form->fun->N );
							command= concat2( command, "DCL[", form->fun->name, Nbuf, "0]", NULL );
							if( pragma_unlikely(debugFlag) ){
								fprintf( StdErr, "Redefining deleted array %s with %s\n", form->fun->name, command );
							}
							form->fun->usage= NULL;
							__fascanf( &n, command, &r, NULL, NULL, NULL, NULL, &level, NULL );
							form->fun->usage= usage;
						}
						if( form->type== _ascanf_procedure ){
							if( pragma_unlikely(debugFlag) ){
								fprintf( StdErr,
									"Warning: expression depends on deleted procedure %s that cannot be automatically redefined!\n",
									form->fun->name
								);
							}
						}
						else{
							n+= 1;
						}
						break;
					case _ascanf_procedure:{
					  int m= Check_Form_Dependencies( &(form->fun->procedure) );
						if( m && !form->fun->dialog_procedure ){
						  /* need to recompile this procedure!	*/
						  ascanf_Function *af= form->fun, *aH= af->accessHandler;
						  char *c= strdup( af->procedure->expr );
						  int n;
						  double r= 0;
							if( pragma_unlikely(debugFlag) ){
								fprintf( StdErr, "Procedure %s depends on %d deleted variables - recompiling...\n",
									af->name, m
								);
							}
							compile_procedure_code( &n, c, &r, af, &level );
							af->accessHandler= aH;
							xfree(c);
							n+= m;
						}
						break;
					}
				}
			}
			if( form->args ){
				n+= Check_Form_Dependencies( &form->args );
			}
			form= form->cdr;
		}
		level-= 1;
	}
	return(n);
}

/* this index() function skips over a (nested) matched
 * set of square braces.
 */
char *ascanf_index( char *s, char c, int *instring )
{
	if( !( s= find_balanced( s, c, '[', ']', True, instring )) ){
		return(NULL);
	}
	if( (c== ' ' || c== '\t')? isspace((unsigned char)*s) : *s== c ){
		return( s);
	}
	else
		return( NULL);
}

/* routine that parses input in the form "[ arg ]" .
 * arg may be of the form "arg1,arg2", or of the form
 * "fun[ arg ]" or "fun". fascanf() is used to parse arg.
 * After parsing, (*function)() is called with the arguments
 * in an array, or NULL if there were no arguments. Its second argument
 * is the return-value. The return-code can be 1 (success), 0 (syntax error)
 * or EOF to skip the remainder of the line.
 */

int ascanf_check_now= 0, ascanf_check_int= 1000, ascanf_propagate_events= 1;

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
int ascanf_check_event(char *caller)
{ int ret= 0;
  extern void *WindowList;
	if( ascanf_check_int ){
		if( ascanf_check_now == 0 ){
		  int python_interrupt = False;
			if( dm_python ){
				if( dm_python->Python_CheckSignals() == -1 ){
					ascanf_escape= ascanf_interrupt= True;
					*ascanf_escape_value= *ascanf_interrupt_value= 1;
					python_interrupt = 1;
				}
			}
			if( ascanf_window ){
				if( XEventsQueued( disp, QueuedAfterFlush) )
/* 				if( XEventsQueued( disp, QueuedAlready) )	*/
				{
				  XEvent theEvent;
				  Boolean done= False;
					XNextEvent( disp, &theEvent);
					if( theEvent.xany.window== ascanf_window && theEvent.type== KeyPress ){
					  char keys[16], *pyMsg[] = { "", " from Python code", " and sent to Python code" };
					  int i, nbytes = XLookupString(&theEvent.xkey, keys, 16,
								   (KeySym *) 0, (XComposeStatus *) 0
							   );
						for( i = 0;  i < nbytes;  i++ ){
							if( keys[i] == 0x18 || keys[i]== ' ' ){
							  /* ^X	*/
								ascanf_escape= ascanf_interrupt= True;
								*ascanf_escape_value= *ascanf_interrupt_value= 1;
								systemtimers= 0;
								if( !python_interrupt && dm_python && *dm_python->pythonActive ){
									dm_python->Python_SetInterrupt();
									python_interrupt = 2;
								}
								fprintf( StdErr, "ascanf::%s: interrupt requested%s.\n", caller, pyMsg[python_interrupt] );
								if( Evaluating_Form && (ascanf_verbose || debugFlag || scriptVerbose) ){
									Print_Form( StdErr, &Evaluating_Form, 0, True, NULL, "#\t", NULL, False );
								}
								done= True;
							}
						}
						ret= 1;
					}
					if( !done ){
						if( ascanf_propagate_events ){
						  Window aw= ascanf_window;
						  void *AW= ActiveWin;
							if( _Handle_An_Event( &theEvent, 0, 0, caller ) || ascanf_window!= aw || ActiveWin!= AW ){
								*ascanf_escape_value= ascanf_escape= True;
								systemtimers= 0;
								fprintf( StdErr, "ascanf::%s: handled %s event requires interrupt of ongoing evaluation.\n",
									caller, event_name(theEvent.type)
								);
								if( Evaluating_Form ){
									Print_Form( StdErr, &Evaluating_Form, 0, True, NULL, "#\t", NULL, False );
								}
							}
						}
						else{
						  /* Put back the event. Using XPutBackEvent(), we'd immediately stumble over it again. So
						   \ send the event, which (presumably) appends the event to the tail of the queue...
						   \ I'm not sure about the 0 event_mask, but it appears to work.
						   */
							XSendEvent( disp, theEvent.xany.window, 0, 0, &theEvent);
						}
					}
				}
				if( !WindowList ){
				  /* This most likely means we're exitting.. all windows are gone...	*/
					*ascanf_escape_value= ascanf_escape= True;
					systemtimers= 0;
				}
			}
		}
		ascanf_check_now= (ascanf_check_now+1) % ascanf_check_int;
	}
	return(ret);
}

extern double *ascanf_switch_case;
DEFUN(ascanf_switch0, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_switch, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_if, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_if2, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_progn, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_IDict_fnc, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_verbose_fnc, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_noverbose_fnc, ( ASCB_ARGLIST ), int);
int ascanf_noverbose= False;
DEFUN(ascanf_global_fnc, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_matherr_fnc, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_comment_fnc, ( ASCB_ARGLIST ), int);
DEFUN(ascanf_popup_fnc, ( ASCB_ARGLIST ), int);
DEFUN(fascanf, ( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **form), int);
DEFUN( ascanf_whiledo, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_dowhile, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_for_to, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_for_toMAX, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_print, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_Eprint, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_Dprint, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_DeclareVariable, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_DeclareProcedure, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_EditProcedure, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_DeleteVariable, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_DefinedVariable, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_compile, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_noEval, ( ASCB_ARGLIST ), int );
DEFUN( ascanf_SHelp, ( ASCB_ARGLIST ), int );
DEFUN( DBG_SHelp, ( char *string, int internal ), double );

char *ascanf_var_search= NULL, *ascanf_usage_search= NULL, *ascanf_label_search= NULL;

char *ascanf_ProcedureCode( ascanf_Function *af )
{
	if( af->type== _ascanf_procedure && af->procedure ){
		return( af->procedure->expr );
	}
	else{
		return( NULL );
	}
}

extern xtb_frame *vars_pmenu;
ascanf_type current_function_type, current_DCL_type;
ascanf_Function *current_DCL_item;

  /* Routine for expanding ascanf memory. Care should be taken not to expand memory which is in use
   \ in the current scope, such as the param_scratch buffer..! Such memory should use its own length
   \ variable, and be expanded the next time before using it.. Therefore fascanf() and compiled_fascanf()
   \ set param_scratch_inUse to True during their scope/duration, so that clean_param_scratch() (which might
   \ be invocated during an unexpectedly generated redraw) won't re-allocate when it is in use.
   */
int Ascanf_AllocMem( int elems )
{ int i;
  extern int Ascanf_AllocSSMem(int elems);
	if( elems> 0 ){
		if( !(ascanf_memory= (double*) XGrealloc( ascanf_memory, elems* sizeof(double))) ){
			fprintf( StdErr, "Ascanf_AllocMem(): Can't get memory for ascanf_memory[%d] array (%s)\n", elems, serror() );
			elems= 0;
			return(0);
		}

		if( !Ascanf_AllocSSMem(elems) ){
			elems= 0;
			return(0);
		}

		if( elems> ASCANF_MAX_ARGS ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " resetting newly acquired memory! ");
			}
			for( i= ASCANF_MAX_ARGS; i< elems; i++ ){
				ascanf_memory[i]= 0;
			}
		}
		ASCANF_MAX_ARGS= elems;
	}
	else if( elems< 0 ){
		return( (ascanf_memory || ascanf_SS || ascanf_SAS) );
	}
	return( elems );
}

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
long ascanf_hash( char *name, unsigned int *hash_len)
{  long hash= 0L;
   unsigned int len= 0;
   char *first= name;
   Boolean full;
	  /* 20000926: allow spaces in names that start with a double-quote or a curly-brace-open	*/
	if( *first== '"' || *first== '{' ){
		full= True;
	}
	else{
		full= False;
	}
	if( full ){
		while( *name ){
			hash+= hash<<3L ^ *name++;
			len+= 1;
		}
	}
	else{
		while( *name && *name!= '[' && *name!= ascanf_separator && !isspace((unsigned char)*name) ){
			hash+= hash<<3L ^ *name++;
			len+= 1;
		}
	}
	if( hash_len ){
		*hash_len= len;
	}
	return( hash);
}

extern ascanf_Function *af_ArgList;
double ascanf_PointerPos[2], ascanf_ArgList[1], af_ArgList_address= 0, ascanf_elapsed_values[4];
int ascanf_update_ArgList= True;
extern double *Allocate_Internal;
int pAllocate_Internal= False;
ascanf_Function *Allocated_Internal= NULL;
extern ascanf_Function *ascanf_VarLabel;

static int EditProcedure( ascanf_Function *af, /* double *arg, */ int *level, char *expr, char *errbuf );

int Delete_Variable( ascanf_Function *af )
{
	if( af ){
		af->type= _ascanf_novariable;
		xfree( af->usage );
		af->PythonEvalType= 0;
		if( !af->sourceArray ){
			  // 20090414: centralise de-allocation of un-needed former elements:
			Resize_ascanf_Array( af, 0, NULL );
			if( af->array!= af->linkedArray.dataColumn ){
				ascanf_xfree( af, af->array );
			}
			else{
				unregister_LinkedArray(af);
				af->array= NULL;
			}
			af->linkedArray.dataColumn= NULL;
			ascanf_xfree( af, af->iarray );
		}
		else{
			af->array= NULL, af->iarray= NULL;
			af->sourceArray= NULL;
		}
		if( af->procedure ){
			Destroy_Form( &af->procedure );
			af->procedure= NULL;
		}
		if( af->accessHandler ){
			af->accessHandler= NULL;
		}
		xfree( af->label );
		if( af->fp ){
			delete_FILEsDescriptor(af->fp);
			if( af->fp_is_pipe ){
				pclose( af->fp );
			}
			else{
				fclose(af->fp);
			}
			af->fp= NULL;
		}
		if( af->cfont ){
			Free_CustomFont(af->cfont);
			xfree( af->cfont );
		}
		if( af->SS ){
			xfree( af->SS->sample );
			xfree( af->SS );
		}
		if( af->SAS ){
			xfree( af->SAS->sample );
			xfree( af->SAS );
		}
		if( af->type== _ascanf_python_object && af->PyObject && dm_python ){
			if( dm_python->Python_DECREF ){
				(*dm_python->Python_DECREF)( af->PyObject );
				af->PyObject= NULL;
				xfree(af->PyObject_Name);
			}
		}
		if( af->PyAOself.selfaf ){
			*af->PyAOself.selfaf = NULL;
		}
		af->PyAOself.self = NULL;
		af->N= -1* ABS(af->N);
		af->free= NULL;
		af->malloc= NULL;
		xtb_popup_delete( &vars_pmenu );
		if( ascanf_progn_return== af->own_address ){
			ascanf_progn_return= 0;
		}
#ifdef USE_AA_REGISTER
		delete_ascanf_Address(AAF_REPRESENTATION(af));
#else
		delete_VariableName(af->name);
#endif
		af->own_address= 0;
		return(1);
	}
	return(0);
}

static int vlabel_match( ascanf_Function *af, char *label )
{ int r= 0;
  char *vlabel= (af)? af->label : NULL;
	if( !af ){
		return(0);
	}
	if( label && *label && vlabel && strncmp( label, "RE^", 3)== 0 && label[strlen(label)-1]== '$' ){
		if( re_comp( &label[2] ) ){
			fprintf( StdErr, "vlabel_match: can't compile regular expr. \"%s\" (%s)\n",
				&label[2], serror()
			);
			r= 0;
		}
		else{
			r= re_exec( vlabel );
			if( r && pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (%s{%s} matches \"%s\") ", af->name, vlabel, label );
			}
		}
	}
	else{
		r= !strcmp( vlabel, label );
	}
	return(r);
}

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
void ShowArrayFeedback( FILE *fp, ascanf_Function *Function, int m, int first, int n )
{ int i= first+ 2, j;
	if( Function->iarray ){
		fprintf( fp, "{%s%d",
			(m)? "..," : "", Function->iarray[m]
		);
		if( n> 1 ){
			  /* NB: i is initialised here 1 higher than in the value-setting code because array[m]
			   \ is printed out outside the loop (would otherwise have incremented i by 1....)
			   */
			for( j= m+1; j< Function->N && i< n-first; j++, i++ ){
				fprintf( fp, ",%d", Function->iarray[j]);
			}
			if( j< Function->N ){
				fputs( ",..", fp );
			}
		}
		else{
			fputs( ",..", fp );
		}
	}
	else{
		fprintf( fp, "{%s%s",
			(m)? "..," : "",
			ad2str( Function->array[m], d3str_format, NULL)
		);
		if( n> 1 ){
			for( j= m+1; j< Function->N && i< n-first; j++, i++ ){
				fprintf( fp, ",%s",
					ad2str( Function->array[j], d3str_format, NULL)
				);
			}
			if( j< Function->N ){
				fputs( ",..", fp );
			}
		}
		else{
			fputs( ",..", fp );
		}
	}
	fputs( "}== ", fp );
}

int doSyntaxCheck= False, AutoCreating= False;

Time_Struct AscanfCompileTimer;
SimpleStats SS_AscanfCompileTime;

extern DyModAutoLoadTables _AutoLoadTable[];
DyModAutoLoadTables *AutoLoadTable= NULL;
extern int AutoLoads;

int contained( double value, ascanf_Function *array )
{ int i;
  int r= False;
	if( array->iarray ){
	  int *ii;
		for( i= 0, ii= array->iarray; i< array->N && !r; i++, ii++ ){
			r= (*ii==value);
		}
	}
	else if( array->array ){
	  double *dd;
		for( i= 0, dd= array->array; i< array->N && !r; i++, dd++ ){
			r= (*dd==value);
		}
	}
	return(r);
}

#define fortoMAXok(loop,incr,arg)	(( (((incr)== 1 && (loop)< (arg)[1]) || ((incr)==-1 && (loop)>(arg)[1])) && !NaN((arg)[1]) )? True : False)

/* Evaluate an ascanf() function. 's' is parsed to look for arguments for 'Function'.
 \ When s=="fun[arguments]", "arguments is passed to fascanf() for evaluation into
 \ an array of ASCANF_MAX_ARGS elements. If 'arglist' is not NULL, no final evaluation is done;
 \ the argument-tree to 'Function' is left in 'arglist'.
 \ 20001115: introduced local variable AMARGS=ASCANF_MAX_ARGS. This should prevent crashes when
 \ ASCANF_MAX_ARGS increases within a lower frame, and higher level frames try to access the memory
 \ they think they have but in fact don't. This could happen e.g. in loops.
 \ 20001116: disabled this again by redefining AMARGS as a macro to ASCANF_MAX_ARGS, to gain some speed.
 */
int ascanf_function( ascanf_Function *Function, int Index, char **s, int ind, double *A, char *caller, Compiled_Form **arglist, int *level )
{
#define AMARGS	ASCANF_MAX_ARGS
#ifndef AMARGS
	int AMARGS= ASCANF_MAX_ARGS;
#endif
#ifdef ASCANF_ALTERNATE
  ascanf_Callback_Frame frame;
#endif
//  __ALLOCA( arg_buf, char, ASCANF_FUNCTION_BUF, arg_buf_len);
//  __ALLOCA( errbuf, char, ASCANF_FUNCTION_BUF+ 256, errbuf_len);
  char *arg_buf, *errbuf;
  int errbuf_len;
  char *c, *d, *args, *s_restore= NULL, s_restore_val= 0;
  int i= 0, n= Function_args(Function), brace_level= 0, ok= 0, ok2= 0, verb=(ascanf_verbose)? 1 : 0, comm= ascanf_comment,
	popp= ascanf_popup, mverb= matherr_verbose, avb= (int) ascanf_verbose, AlInt= *Allocate_Internal,
	anvb= ascanf_noverbose, apl= AllowProcedureLocals;
  Time_Struct *timer= NULL;
  DEFMETHOD( function, ( ASCB_ARGLIST ), int)= Function->function;
  char *name= Function->name;
/*   static int level= 0;	*/
/*   int _ascanf_loop= *ascanf_loop_ptr, *awlp= ascanf_loop_ptr;	*/
  int _ascanf_loop= 0, *awlp= ascanf_loop_ptr, *ailp= ascanf_in_loop, *ali= ascanf_loop_incr;
  double odref= *ascanf_switch_case, arg1= -1;
  int alc= *ascanf_loop_counter, _ascanf_loop_counter= 0, sign= 1, negate= 0;
  static FILE *cfp= NULL, *rcfp= NULL;
    /* 991004: I doubt cSE and pSE must be statics..!	*/
  FILE *cSE= StdErr, *pSE= StdErr;
  int cugi= ascanf_use_greek_inf, pugi= ascanf_use_greek_inf;
  static char *tnam= NULL;
  static ascanf_Function *last_1st_arg= NULL;
  static int last_1st_arg_level= 0;
  static int last_1st_arg_tarray_index= -1;
  ascanf_Function *l1a= last_1st_arg;
  int l1al= last_1st_arg_level;
  char *pref= (arglist)? "#ac:" : ((ascanf_SyntaxCheck)? "#sc:" : "#");
  double *ArgList= af_ArgList->array;
  int Argc= af_ArgList->N, aaa= ascanf_arguments, dSC= doSyntaxCheck, aSC= ascanf_SyntaxCheck, cwarned= 0;
  ascanf_Function **vlF= vars_local_Functions;
  int *lF= local_Functions;

#define POPUP_ERROR(win,msg,title)	xtb_error_box(win,msg, (arglist)? "Compiler " title : title )

	if( ascanf_escape ){
		if( !(*level) ){
			*ascanf_escape_value= ascanf_escape= False;
		}
		else{
			return(-1);
		}
	}

	arg_buf = (char*) malloc( ASCANF_FUNCTION_BUF * sizeof(char) );
	errbuf_len = ASCANF_FUNCTION_BUF + 256;
	errbuf = (char*) malloc( errbuf_len * sizeof(char) );
	if( !arg_buf || !errbuf ){
		goto bail;
	}
	c = args = arg_buf;

	(*level)++;

/* 	if( (*level)== 1 ){	*/
/* 		last_1st_arg= NULL;	*/
/* 		last_1st_arg_level= 0;	*/
/* 	} else	*/
	  /* 20010503: only change when Function->store	*/
	if( Index== 0 ){
		if( /* Index== 0 && */ Function->store ){
			last_1st_arg= Function;
			last_1st_arg_level= -1;
			last_1st_arg_tarray_index= -1;
		}
		  /* 20010504: a function with the storage operator sets the flags such that
		   \ the following test is true. We can then store the information needed in
		   \ the calling level in the same flags.
		   */
		else if( last_1st_arg_level== -1 && last_1st_arg->store ){
			last_1st_arg= Function;
			last_1st_arg_level= *level;
		}
	}

	ascanf_arg_error= 0;
	ascanf_emsg= NULL;

	if( function== ascanf_systemtime || function== ascanf_systemtime2 ){
		if( (timer= (Time_Struct*) calloc(1, sizeof(Time_Struct))) ){
			Elapsed_Since( timer, True );
			set_NaN( ascanf_elapsed_values[3] );
			if( ActiveWin && Function->name && !arglist ){
				TitleMessage( ActiveWin, Function->name );
			}
		}
	}
	else if( function== ascanf_matherr_fnc ){
		matherr_verbose= 1;
	}
	else if( function== ascanf_global_fnc ){
		if( AllowProcedureLocals && ascanf_verbose ){
			fprintf( StdErr, "\n%s%d \tlocal variable declaration switched off\n", pref, *level );
			fprintf( StdErr, "%s%d \t%s %s\n", pref, (*level), *s, (TBARprogress_header)? TBARprogress_header : "" );
		}
		*AllowProcedureLocals_value= AllowProcedureLocals= 0;
		vars_local_Functions= NULL;
		local_Functions= NULL;
	}
	else if( function== ascanf_verbose_fnc ){
		if( !(*Allocate_Internal> 0 && arglist ) && !ascanf_noverbose ){
			*ascanf_verbose_value= ascanf_verbose= 1;
			matherr_verbose= -1;
			fprintf( StdErr, "#%s%d: \t%s %s\n", (arglist)? "ac" : "", (*level), *s, (TBARprogress_header)? TBARprogress_header : "" );
			if( systemtimers && !vtimer && !arglist && (vtimer= (Time_Struct*) calloc(1, sizeof(Time_Struct))) ){
				Elapsed_Since( vtimer, True );
			}
		}
	}
	else if( function== ascanf_noverbose_fnc ){
/* 		if( !(*Allocate_Internal> 0 && arglist ) )	*/
		{
			*ascanf_verbose_value= ascanf_verbose= 0;
			matherr_verbose= 0;
			ascanf_noverbose= True;
		}
	}
	else if( function== ascanf_IDict_fnc ){
		pAllocate_Internal= *Allocate_Internal;
		*Allocate_Internal= True;
		if( ascanf_verbose ){
			fprintf( StdErr, "\n%s%d \tinternal dictionary access switched on\n", pref, *level );
			fprintf( StdErr, "%s%d \t%s %s\n", pref, (*level), *s, (TBARprogress_header)? TBARprogress_header : "" );
		}
	}
	else if( (function== ascanf_comment_fnc || function== ascanf_popup_fnc) && !cfp && !(arglist || ascanf_SyntaxCheck) ){
		tnam= XGtempnam( getenv("TMPDIR"), "af_fnc");
		if( tnam ){
			cfp= fopen( (tnam= XGtempnam( getenv("TMPDIR"), "af_fnc")), "wb");
		}
		else{
			cfp = NULL;
		}
		if( cfp ){
			if( function== ascanf_comment_fnc ){
				ascanf_comment= (*level);
/* 				cSE= StdErr;	*/
				StdErr= register_FILEsDescriptor(cfp);
				ascanf_use_greek_inf= True;
			}
			else{
				ascanf_popup= (*level);
				if( *ascanf_popup_verbose && !ascanf_noverbose ){
					*ascanf_verbose_value= ascanf_verbose= 1;
				}
/* 				pSE= StdErr;	*/
				ascanf_use_greek_inf= True;
				StdErr= register_FILEsDescriptor(cfp);
			}
			if( !ascanf_noverbose ){
				fprintf( cfp, "#%s%d: \t%s %s\n", (arglist)? "ac" : "", (*level), *s, (TBARprogress_header)? TBARprogress_header : "" );
			}
			if( pragma_unlikely(debugFlag) ){
				fprintf( StdErr, "ascanf_function(): opened temp file \"%s\" as buffer\n",
					tnam
				);
			}
			rcfp= fopen( tnam, "r");
			unlink( tnam );
		}
		else if( pragma_unlikely(debugFlag) ){
			fprintf( StdErr, "ascanf_function(): can't open temp file \"%s\" as buffer (%s)\n",
				(tnam)? tnam : "<NULL!!>", serror()
			);
		}
		if( tnam ){
			xfree(tnam);
		}
	}

	if( Function->SyntaxCheck ){
	  /* 20020530: this may be overkill, but it a routine requires a thorough syntax check (of its
	   \ arguments), not only must its own callback be called, but also the callbacks of its arguments!
	   \ Therefore, we must set (and never unset) a global variable that will retain its "trueness"
	   \ until compiling the current frame has finished. This variable is not used for anything else,
	   \ so we can safely set it even when not compiling.
	   */
		doSyntaxCheck= True;
		  /* 20030409: set ascanf_SyntaxCheck, as it *may* not be set when we actually call
		   \ the callback (e.g. when we arrive here via a Create_AutoVariable call).
		   \ 20030411: bad idea. This will cause that these functions can never be evaluated
		   \ without the compiler... Thence, only set ascanf_SyntaxCheck when creating autovariables...
		   */
		ascanf_SyntaxCheck= ascanf_SyntaxCheck | AutoCreating;
	}

	if( *(d= &((*s)[ind]) )== '[' ){
#define LABEL_OPCODE	" # label={"
	  char *lab= strstr( *s, LABEL_OPCODE ), *end;
		d++;
		  /* 20020602: use find_balanced to use the end of the input pattern!
		   \ This is better since the input pattern may contain any number of the closing
		   \ braces inside one or more strings.
		   */
		end= find_balanced(d, ']', '[', ']', True, NULL);
		if( end ){
			while( *d && d!= end ){
			  /* copy until the end of this input pattern	*/
				if( i< ASCANF_FUNCTION_BUF-1 )
					*c++ = *d;
				d++;
				i++;
			}
		}
		else{
		  /* fall back. Shouldn't be necessary! */
			while( *d && !(*d== ']' && brace_level== 0) ){
			  /* find the end of this input pattern	*/
				if( i< ASCANF_FUNCTION_BUF-1 )
					*c++ = *d;
				if( *d== '[')
					brace_level++;
				else if( *d== ']')
					brace_level--;
				d++;
				i++;
			}
		}
		*c= '\0';
		if( *d== ']' ){
		  /* there was an argument list: parse it	*/
			while( isspace((unsigned char)*args) )
				args++;
			if( !strlen(args) ){
			  /* empty list...	*/
				ascanf_arguments= 0;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "%s%d \t%s== ", pref, (*level), name );
					fflush( StdErr );
				}
				if( pragma_unlikely(ascanf_comment && cfp) ){
					fprintf( cfp, "%s%d \t%s== ", pref, (*level), name );
				}
				if( (arglist || ascanf_SyntaxCheck) && !doSyntaxCheck ){
					ok= 1;
					(*ascanf_current_value)= 0.0;
				}
				else{
					  /* 20020419 */
					if( ascanf_update_ArgList && Function && Function->accessHandler ){
						if( ascanf_verbose && (arglist || ascanf_SyntaxCheck) ){
							fprintf( StdErr, "##%s%d \t(%s/access handler) setting $ args array from %d to %d elements ##",
								pref, (*level), Function->name, af_ArgList->N, 0
							);
						}
						SET_AF_ARGLIST( ascanf_ArgList, 0 );
					}
					if( Function->type== _ascanf_procedure ){
					  int nn= n, uaA= ascanf_update_ArgList;
					  double AA= 0;
						n= 1;
						ascanf_update_ArgList= False;
						evaluate_procedure( &n, Function, &AA, level );
						ascanf_update_ArgList= uaA;
						if( n== 1 ){
							*A= AA;
							Function->assigns+= 1;
						}
						n= nn;
					}
					  /* 20080916 */
					else if( Function->type== _ascanf_python_object && dm_python ){
					  double AA;
						if( (*dm_python->ascanf_PythonCall)( Function, 0, NULL, &AA) ){
							*A= Function->value= AA;
							Function->assigns+= 1;
							if( Function->accessHandler ){
								AccessHandler( Function, Function->name, level, NULL, *s, A );
							}
							if( negate ){
								*A= (*A && !NaN(*A))? 0 : 1;
							}
							else{
								*A *= sign;
							}
							(*ascanf_current_value)= *A;
							ok= 1;
						}
					}
					if( last_1st_arg== Function && last_1st_arg_level== *level && Index== 0 ){
						if( Function->type== _ascanf_array ){
							fprintf( StdErr, "##%s%d \terror: request to store in a non-referenced array (%s): ignored! ##",
								pref, (*level), *s
							);
							last_1st_arg= NULL;
							if( arglist ){
								Function->store= False;
							}
						}
					}
#ifdef ASCANF_ALTERNATE
					frame.args= NULL;
					frame.result= A;
					frame.level= level;
					frame.compiled= NULL;
					frame.self= Function;
#if defined(ASCANF_ARG_DEBUG)
					frame.expr= *s;
#endif
					ok= (*function)( &frame );
#else
					ok= (*function)( NULL, A, level );
#endif
					(*ascanf_current_value)= Function->value= *A;
					if( Function->accessHandler ){
						AccessHandler( Function, Function->name, level, NULL, *s, A );
					}
				}
				Function->reads+= 1;
				ok2= 1;
				ascanf_check_event( "ascanf_function" );
			}
			else{
			  double *arg;
				if( i<= ASCANF_FUNCTION_BUF-1 ){
				    /* 20020412: allocate the argument buffer only when really needed: */
					arg = (double*) malloc( AMARGS * sizeof(double) );
				}
				else{
					arg = NULL;
				}
				if( arg ){
				  int j, in_loop_break= 0, ilb_level= -1, loop_incr= 1;
				  char *larg[3]= { NULL, NULL, NULL};
				  Boolean var_Defined= False;
					for( arg[0]= 0.0, j= 1; j< Function_args(Function); j++){
						  /* 20010630: I think that initialising to 0 instead of 1 is a far better idea... */
						arg[j]= 0.0;
					}
					if( (arglist || ascanf_SyntaxCheck) &&
						(function== ascanf_if || function== ascanf_if2 || function== ascanf_for_to || function== ascanf_for_toMAX ||
						function== ascanf_switch0 || function== ascanf_switch)
					){
						larg[1]= ascanf_index( arg_buf, ascanf_separator, NULL);
						if( !larg[1] ){
							if( ascanf_CompilingExpression ){
								fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
								cwarned+= 1;
							}
							sprintf( errbuf, "#%s%d \t\"%s[%s]\" error: functioncall with only 1 parameter\n",
								(arglist)? "ac" : "", (*level), Function->name, arg_buf
							);
							fputs( errbuf, StdErr ); fflush( StdErr );
							if( HAVEWIN ){
								POPUP_ERROR( USEWINDOW, errbuf, "Error" );
							}
							ascanf_arg_error= 1;
						}
						else{
							larg[2]= (larg[1][1]== ascanf_separator)? &larg[1][1] : ascanf_index( &(larg[1][1]), ascanf_separator, NULL);
							if( larg[1][1]== ascanf_separator || !larg[2] ){
								if( ascanf_CompilingExpression ){
									fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
									cwarned+= 1;
								}
								sprintf( errbuf, "#%s%d \t\"%s[%s]\" warning: function call with only 2 parameters\n",
									(arglist)? "ac" : "", (*level), Function->name, arg_buf
								);
								fputs( errbuf, StdErr ); fflush( StdErr );
								if( HAVEWIN && ascanf_verbose ){
									POPUP_ERROR( USEWINDOW, errbuf, "Warning" );
								}
							}
						}
						larg[1]= larg[2]= NULL;
					}
					if( !arglist ){
						if( function== ascanf_if ){
						  int N, NN= n-2;
							larg[0]= arg_buf;
							larg[1]= ascanf_index( larg[0], ascanf_separator, NULL);
							larg[2]= (larg[1])? (larg[1][1]== ascanf_separator)? &larg[1][1] :
										ascanf_index( &(larg[1][1]), ascanf_separator, NULL) : NULL;
							if( larg[0] ){
								N= 1;
								n= 1;
								_fascanf( &n, larg[0], &arg[0], NULL, NULL, NULL, NULL, level, NULL );
								if( arg[0] && !NaN(arg[0]) ){
								  /* try to evaluate a second argument	*/
									if( larg[1] && larg[1][1]!= ascanf_separator ){
										n= 1;
										_fascanf( &n, &(larg[1][1]), &arg[1], NULL, NULL, NULL, NULL, level, NULL);
										if( n ){
											N= 2;
										}
									}
								}
								else{
									set_NaN( arg[1] );
									if( larg[2] ){
									  /* try to evaluate from the third argument.	*/
	/* 									n= 1;	*/
										n= NN;
										_fascanf( &n, &(larg[2][1]), &arg[2], NULL, NULL, NULL, NULL, level, NULL);
										if( n ){
	/* 										N= 3;	*/
											N= n+ 2;
										}
									}
									else{
									  /* default third argument	*/
										N= 3;
										arg[2]= 0.0;
									}
								}
								n= N;
							}
						}
						else if( function== ascanf_switch || function==ascanf_switch0 ){
						  int N;
						  ascanf_Function atest, aref;
						  /* switch[value, {a,b,c}, exprs] should evaluate exprs if value is contained
						   \ in {a,b,c}. Iow, if the 1st argument is *not* an array pointer, and the (a) case
						   \ argument is, check if the 1st arg. is contained in that array. This will
						   \ allow multiple switch values to share the same code, as in C, and will avoid
						   \ the need to put <True> as the test value and explicit tests as the switch values
						   \ (as switch[1, contained[value,a,b,c],exprs]).
						   */
							larg[0]= arg_buf;
							larg[1]= ascanf_index( larg[0], ascanf_separator, NULL);
							larg[2]= (larg[1])? (larg[1][1]== ascanf_separator)? &larg[1][1] :
										ascanf_index( &(larg[1][1]), ascanf_separator, NULL) : NULL;
							if( larg[0] && larg[1] ){
							  int ref= 1, ok= 1;
								N= 2;
								n= 2;
								*ascanf_switch_case= (ref- 1)/ 2;
								  /* 20050111: attn: we evaluate the first PAIR of arguments, test and switch (arg[ref])!! */
								_fascanf( &n, larg[0], &arg[0], NULL, NULL, NULL, NULL, level, NULL );
								atest.value= arg[0];
								atest.type= _ascanf_value;
								aref.value= arg[1];
								aref.type= _ascanf_value;
								if( function==ascanf_switch ){
								  ascanf_Function *af= parse_ascanf_address(arg[0], _ascanf_array, "ascanf_function", 0, NULL );
									if( !af ){
										if( (af= parse_ascanf_address(arg[1], _ascanf_array, "ascanf_function", 0, NULL )) ){
											aref= *af;
										}
									}
									else{
									  /* test variable is an array pointer: store that fact in atest.type (no
									   \ need to modify atest.value)
									  */
										atest.type= _ascanf_array;
									}
								}

#define afSWITCHCCOMPARE(a) ((function==ascanf_switch0 || atest.type==_ascanf_array || aref.type!=_ascanf_array)? \
								((a)[0]==(a)[ref]) : \
								contained((a)[0],&aref) )

								while( ok && !afSWITCHCCOMPARE(arg) ){
									set_NaN(arg[ref+1]);
									if( larg[2] ){
										larg[1]= (larg[2][1]== ascanf_separator)? &larg[2][1] :
													ascanf_index( &(larg[2][1]), ascanf_separator, NULL);
										larg[2]= (larg[1])? (larg[1][1]== ascanf_separator)? &larg[1][1] :
													ascanf_index( &(larg[1][1]), ascanf_separator, NULL) : NULL;
										if( larg[2] ){
											ref+= 2;
											*ascanf_switch_case= (ref- 1)/ 2;
											N+= 2;
											n= 1;
											_fascanf( &n, &larg[1][1], &arg[ref], NULL, NULL, NULL, NULL, level, NULL );
											aref.value= arg[ref];
											aref.type= _ascanf_value;
											if( function==ascanf_switch && atest.type!=_ascanf_array ){
											  ascanf_Function *af;
												if( (af= parse_ascanf_address(arg[ref], _ascanf_array, "ascanf_function", 0, NULL )) ){
													aref= *af;
												}
											}
											ok= 1;
										}
										else{
											ok= 0;
											  /* 20020826: */
											*ascanf_switch_case+= 1;
										}
									}
									else{
										ok= 0;
									}
								}
								if( ok && larg[2] && afSWITCHCCOMPARE(arg) ){
									N+= 1;
									n= 1;
									  /* The number of the appropriate case: 	*/
									*A= (ref-1)/ 2;
									_fascanf( &n, &larg[2][1], &arg[ref+1], NULL, NULL, NULL, NULL, level, NULL );
								}
								else if( larg[1] && !larg[2] ){
								  /* Default argument..	*/
									N+= 2;
									n= 1;
									  /* The number of the appropriate case: 	*/
									*A= -1;
									_fascanf( &n, &larg[1][1], &arg[N-1], NULL, NULL, NULL, NULL, level, NULL );
								}
								n= N;
							}
						}
					}
					ascanf_loop_ptr= &_ascanf_loop;
					if( arglist ){
						*ascanf_loop_ptr= 0;
					}
					do{
					  int nargs=-1, Nargs= n;
						if( *ascanf_loop_ptr> 0 ){
						  /* We can come back here while evaluating the arguments of
						   \ a while function. We don't want to loop those forever...
						   */
							if( pragma_unlikely(ascanf_verbose) ){
								fputs( " (loop)\n", StdErr );
							}
							*ascanf_loop_ptr= -1;
						}
						if( function== ascanf_dowhile && !arglist ){
							  /* Set the exported loopcounter (var $loop) to the local value only
							   \ when we're in a looping frame (and not below, or above..)
							   */
							*ascanf_loop_counter= _ascanf_loop_counter;
							ascanf_in_loop= &in_loop_break;
							ilb_level= *level;
							goto ascanf_function_parse_args;
						}
						  /* ascanf_whiledo is the C while() construct. If the first
						   \ element, the test, evals to false, the rest of the arguments
						   \ are not evaluated. ascanf_dowhile tests the last argument,
						   \ which is much easier to implement.
						   */
						else if( function== ascanf_whiledo && !arglist ){
						  int N= n-1;
							larg[0]= arg_buf;
							larg[1]= ascanf_index( larg[0], ascanf_separator, NULL);
							n= 1;
							*ascanf_loop_counter= _ascanf_loop_counter;
							ascanf_in_loop= &in_loop_break;
							ilb_level= *level;
							_fascanf( &n, larg[0], &arg[0], NULL, NULL, NULL, NULL, level, NULL);
							if( arg[0] && !NaN(arg[0]) ){
							  /* test argument evaluated true, now evaluate rest of the args	*/
								if( larg[1] ){
									Nargs= n= N;
									_fascanf( &n, &larg[1][1], &arg[1], NULL, NULL, NULL, NULL, level, &nargs );
									 /* the actual number of arguments:	*/
									n+= 1;
								}
								else{
									n= 1;
								}
							}
							else{
							  /* test false: we skip the rest	*/
								n= 1;
							}
						}
						  /* the C for() construct. The first argument is an initialiser, which is
						   \ initialised only the first time around. The second is the continuation
						   \ test. The rest of the arguments is evaluated only when the second is true.
						   */
						else if( (function== ascanf_for_to || function== ascanf_for_toMAX) && !arglist ){
						  int N= n-2;
						  int first_ok= 1;
							larg[0]= arg_buf;
							larg[1]= ascanf_index( larg[0], ascanf_separator, NULL);
							if( larg[1] ){
								larg[2]= ascanf_index( &(larg[1][1]), ascanf_separator, NULL);
							}
							else{
								ascanf_arg_error= 1;
							}
							n= 1;
							*ascanf_loop_counter= _ascanf_loop_counter;
							ascanf_in_loop= &in_loop_break;
							ilb_level= *level;
							if( *ascanf_loop_ptr== 0 && larg[1] ){
								_fascanf( &n, larg[0], &arg[0], NULL, NULL, NULL, NULL, level, NULL );
								first_ok= n;
								if( NaN(arg[0]) ){
									first_ok= 0;
									*ascanf_loop_counter= (_ascanf_loop_counter= 0);
								}
								else{
									*ascanf_loop_counter= (_ascanf_loop_counter= (int) arg[0]);
								}
								if( pragma_unlikely(ascanf_verbose) ){
									fprintf( StdErr, "#%s %d\n",
										(first_ok)? " for_to(init)" : " for_to(invalid init)", (*level)
									);
								}
							}
							if( first_ok && larg[1] ){
							  Boolean ok;
								n= 1;
								if( function== ascanf_for_to ){
									_fascanf( &n, &larg[1][1], &arg[1], NULL, NULL, NULL, NULL, level, NULL );
									if( pragma_unlikely(ascanf_verbose) ){
										fprintf( StdErr, "#%s %d\n",
											(arg[1])? " for_to(test,true)" : " for_to(test,false)", (*level)
										);
									}
									ok= (arg[1] && !NaN(arg[1]) )? True : False;
								}
								else if( function== ascanf_for_toMAX ){
									if( *ascanf_loop_ptr== 0 ){
										_fascanf( &n, &larg[1][1], &arg[1], NULL, NULL, NULL, NULL, level, NULL );
										ascanf_loop_incr= &loop_incr;
										if( arg[1]< arg[0] ){
											*ascanf_loop_incr= -1;
										}
									}
									if( pragma_unlikely(ascanf_verbose) ){
										fprintf( StdErr, "#%s %d\n",
											fortoMAXok(*ascanf_loop_counter,loop_incr,arg)? " for_toMAX(test,true)" : " for_toMAX(test,false)", (*level)
										);
									}
/* 									ok= (*ascanf_loop_counter< arg[1] && !NaN(arg[1]) )? True : False;	*/
									ok= fortoMAXok(*ascanf_loop_counter,loop_incr,arg);
								}
								else{
									ok= False;
								}
								if( ok && larg[2] ){
								  /* test argument evaluated true, now evaluate rest of the args	*/
									Nargs= n= N;
									_fascanf( &n, &larg[2][1], &arg[2], NULL, NULL, NULL, NULL, level, &nargs );
									 /* the actual number of arguments:	*/
									n+= 2;
								}
								else{
								  /* test false: we skip the rest	*/
									n= 1;
								}
							}
							else{
							  /* test false: we skip the rest	*/
								*ascanf_escape_value= ascanf_escape= 1;
								n= 1;
							}
						}
						else if( function== ascanf_DeclareVariable || function== ascanf_DeleteVariable ||
							function== ascanf_DefinedVariable || function== ascanf_DeclareProcedure ||
							function== ascanf_EditProcedure
						){
						  char *c= arg_buf, C;
						  int hit= 0, is_address= 0, is_usage= 0;
						  ascanf_Function *af, *del_af= NULL;

							  /* 20000223: don't allow leading whitespace in a name!	*/
							while( *c && isspace( (unsigned char)*c ) ){
								c++;
							}
							larg[0]= c;

							{ int ii= 0;
								while( index( PREFIXES, arg_buf[ii]) && hit< 5 ){
									switch( arg_buf[ii] ){
										case '-':
											larg[0]= &arg_buf[ii+1];
											sign*= -1;
											if( negate && pragma_unlikely(ascanf_verbose) ){
												if( ascanf_CompilingExpression ){
													fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
													cwarned+= 1;
												}
												sprintf( errbuf,
													"ascanf_function() #%s%d: \"%s\" (%c) turns off previous negation (!)\n",
													(arglist)? "ac" : "", (*level), arg_buf, arg_buf[ii]
												);
												fputs( errbuf, StdErr ); fflush( StdErr );
												if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
													POPUP_ERROR( ascanf_window, errbuf, "warning" );
												}
											}
											negate= 0;
											hit+= 1;
											break;
										case '!':
											larg[0]= &arg_buf[ii+1];
											if( sign< 0 && pragma_unlikely(ascanf_verbose) ){
												if( ascanf_CompilingExpression ){
													fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
													cwarned+= 1;
												}
												sprintf( errbuf,
													"ascanf_function() #%s%d: \"%s\" (%c) turns off previous negative (-)\n",
													(arglist)? "ac" : "", (*level), arg_buf, arg_buf[ii]
												);
												fputs( errbuf, StdErr ); fflush( StdErr );
												if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
													POPUP_ERROR( ascanf_window, errbuf, "warning" );
												}
											}
											sign= 1;
											negate= !negate;
											hit+= 1;
											break;
										case '`':
											is_usage= 1;
										case '&':
											larg[0]= &arg_buf[ii+1];
											is_address= 1;
											hit+= 1;
											break;
										case '?':
										case '*':
											/* make no sense here	*/
											if( ascanf_CompilingExpression ){
												fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
												cwarned+= 1;
											}
											sprintf( errbuf,
												"ascanf_function() #%s%d: \"%s\" ignoring special character (%c)\n",
												(arglist)? "ac" : "", (*level), arg_buf, arg_buf[ii]
											);
											fputs( errbuf, StdErr ); fflush( StdErr );
											if( ascanf_window && (/* arglist || ascanf_SyntaxCheck || */ ascanf_verbose || ascanf_PopupWarn) ){
												POPUP_ERROR( ascanf_window, errbuf, "warning" );
											}
											hit+= 1;
											break;
									}
									ii+= 1;
								}
							}
							  /* find the end of the label	*/
							c= find_balanced( c, ascanf_separator, '[', ']', True, NULL  );
							  /* 990413
							   \ Here we can implement (some) support for local variable. c can point
							   \ to a string "name[expr(s)]"; clearly those expr(s) can be a list of
							   \ variables that are to be local to a procedure. To this end, the part
							   \ between sq.brackets must be extracted, and parsed: passing a string
							   \ "aa,bb,cc" to fascanf to compile will create variables aa, bb and cc.
							   \ These should be stored in a local scope, for which the ascanf_Function
							   \ field <local_scope> exists. This parsing should thus be done *after*
							   \ creation of the procedure, but before compilation of its code, and with
							   \ a flag set to indicate that variables will go *not* in vars_ascanf_Functions[0].cdr !!
							   \ During compilation of the procedure code, any remaining undefined variables should
							   \ be stored in the global pool. During subsequent evaluations of procedure code,
							   \ a global variable can be set to point to local scopes, after having saved its
							   \ previous value. Initialisation of local variables has to be done before:
							   \ they can be either scalars or arrays, and initialisation should copy values
							   \ according to the receiving variable's dimensions. During dumping of a
							   \ procedure definition, a format "DEPROC[name[DCL[n1,v1],DCL[n2,d2,v0..vn],..],expr]"
							   \ can be used to completely restore the actual context upon re-definition.
							   \ NOTE: In this scheme, local variables are STATIC: nested procedure calls
							   \ will affect the stack of the calling scope!!
							   \ 990415: Not necessarily! The procedure code is evaluated either in ascanf_function,
							   \ or in compiled_ascanf_function. At those places we can save the procedure's current
							   \ stack in a local (C-context..) buffer. This buffer can be an array when we use our
							   \ knowledge of the number of local variables; saving only has to be done when we actually
							   \ are about to make a nested call - saves time for single non-nested calls (most cases?).
							   \ If we modify the ALLOCA macro to do nothing for a zero-element request, we can even
							   \ transparently use gcc's stack-allocation feature.
							   \ 20020530: just to note that another mechanism exists for quite a while yet, using an
							   \ array ($ or $Args).
							   */
							  /* *c=='\0' when no 2nd argument is given.	*/
							if( !c || !*c || *c== ascanf_separator ){
							  Boolean reusing= False;
							  int deref= False;
							  ascanf_Function *viF= (vars_local_Functions)? *vars_local_Functions : vars_internal_Functions;
							  int ifN= (vars_local_Functions)? 1 : internal_Functions;
							  ascanf_Function *funList= (*Allocate_Internal)? viF : vars_ascanf_Functions;
							  char *open_bracket= index( larg[0], '[' );
								  /* 20020505: do not truncate the argument string at the 1st ascanf_separator, to make
								   \ it possible to have commas (the default ascanf_separator) in variable label strings.
								   \ (and to delete all variables having that label...!)
								   \ Alternatively, we could to this only when a special opcode is given (see the tests below)
								   \ to Delete[]. But since that function accepts only 1 argument anyway, I don't think it is
								   \ necessary to be that subtle.
								   */
								if( c && function!= ascanf_DeleteVariable ){
									C= *c;
									*c= '\0';
								}
								else{
									C= 0;
								}

								  /* 20001102: do a simple minded check for matched brackets in the variable name.	*/
								if( open_bracket && larg[0][ strlen(larg[0])-1 ]== ']' ){
								  /* Found one. We no longer want such variables, because we may want to pass foo[n]
								   \ as the first argument to a DCL call. This currently only makes sense while compiling,
								   \ because it will allow later to access that n'th element of foo as a pointer.
								   */
									*open_bracket= '\0';
								}
								else{
									open_bracket= NULL;
								}

								if( function== ascanf_DeleteVariable ){
								  int dall= 0, dallP= 0, dunu= 0, dunuV= 0, intr= 0, labelled= 0, ulabelled= 0, lS;
								  ascanf_Function *fL;
									if( (dall= (strcmp( larg[0], "$AllDefined" )== 0)) ||
										(dallP= (strcmp( larg[0], "$AllProcedures")== 0)) ||
										(dunuV= (strcmp( larg[0], "$UnUsedVars")== 0)) ||
										(dunu= (strcmp( larg[0], "$UnUsed")== 0)) ||
										(intr= (strcmp( larg[0], "$InternalUnUsed")== 0)) ||
										(labelled= (strncmp( larg[0], "$Label=",7)== 0)) ||
										(ulabelled= (strncmp( larg[0], "$UnUsedLabel=",13)== 0))
									){
									  int sdel= 0, adel=0, pdel= 0;
									  char *label;
										if( intr ){
											dunu= True;
											lS= ifN;
											fL= viF;
										}
										else{
											lS= ascanf_Functions;
											fL= vars_ascanf_Functions;
										}
										if( labelled ){
											label= &larg[0][7];
										}
										else if( ulabelled ){
											label= &larg[0][13];
										}
										else{
											label= NULL;
										}
										  /* 20020324: */
										if( (label || dall) && *Allocate_Internal ){
											lS= ifN;
											fL= viF;
										}
										if( arglist ){
											fprintf( StdErr, " (warning: \"%s\" makes no sense in compiled expression!)== ",
												larg[0]
											);
											fflush( StdErr );
										}
										for( j= 0; j< lS; j++ ){
											af= &fL[j];
											while( af ){
											  ascanf_Function *cdr;
												if( (af->function== ascanf_Variable || af->function== ascanf_Procedure) &&
													af->type!= _ascanf_novariable &&
													(!af->name || af->name[0]!= '$') &&
													(dall ||
														(dallP && af->type== _ascanf_procedure) ||
														(dunuV && af->type!= _ascanf_procedure && af->reads<= 1 &&
															af->assigns<= 2 && !af->links) ||
														(dunu && af->reads<= 1 && af->assigns<= 2 && !af->links) ||
														(labelled &&
															((af->label && vlabel_match(af, label)) ||
															 (*label== '\0' && (!af->label || af->label[0]== '\0'))) ) ||
														(ulabelled && af->reads<= 1 && af->assigns<= 2 && !af->links &&
															((af->label && vlabel_match(af,label)) ||
															 (*label== '\0' && (!af->label || af->label[0]== '\0'))) )
													)
												){
													if( pragma_unlikely(ascanf_verbose) || arglist ){
													  char *pname= NULL;
														PF_sprint_string( &pname, "", "", af->name, False );
														if( dunu || dunuV ){
															fprintf( StdErr, " (%sunused? %s A=%d R=%d L=%d) ",
																(intr)? "internal " : "",
																pname, af->assigns, af->reads, af->links
															);
														}
														else if( af->links> 0 ){
															fprintf( StdErr, " (%s%s is in use %d times) ",
																(intr)? "internal " : "",
																pname, af->links
															);
														}
														xfree( pname );
													}
													switch( af->type ){
														case _ascanf_simplestats:
														case _ascanf_simpleanglestats:
														case _ascanf_python_object:
														case _ascanf_variable:
															sdel+= 1;
															break;
														case _ascanf_array:
															adel+= 1;
															break;
														case _ascanf_procedure:
															pdel+= 1;
															break;
													}
													arg[0]= af->value;
													  /* store the cdr in a local variable before calling Delete_Variable().
													   \ This is currently not necessary, but it is typically the thing one
													   \ tends to forget when doing some modification somewhere :)
													   */
													cdr= af->cdr;
													Delete_Variable( af );
													af= cdr;
												}
												else{
													af= af->cdr;
												}
											}
										}
										if( sdel || adel || pdel ){
											if( pragma_unlikely(ascanf_verbose) || arglist ){
											  char *qualifier= "";
											  char *l= "";
												if( dall ){
													qualifier= " all";
												}
												else if( dunuV ){
													qualifier= " unused scalars/arrays";
												}
												else if( dunu ){
													qualifier= " all unused";
												}
												else if( intr ){
													qualifier= " unused internal";
												}
												else if( labelled ){
													qualifier= " label=";
													l= label;
												}
												else if( ulabelled ){
													qualifier= " unused, label=";
													l= label;
												}
												fprintf( StdErr,
													" (deleted%s%s: %d scalars; %d arrays; %d procedures) ",
													qualifier, l,
													sdel, adel, pdel
												);
												fflush( StdErr );
											}
											xtb_popup_delete( &vars_pmenu );
										}
										// break from loop
										break;
									}
								}

								  /* Try to find a matching entry
								   \ 20010730: !! we must scan at least ascanf_Functions, but if *Allocate_Internal,
								   \ we should also scan internalFunctions iff the specified name does not exist as
								   \ a valid, non-deleted entry in ascanf_Functions. Before, this was not done, which
								   \ could lead to doubly defined variables (with the internal one shadowed).
								   */
DCL_search_entry:;
								{ int try_internals= (*Allocate_Internal || function== ascanf_DefinedVariable)? 1 : 0;
								  int llistSize= ascanf_Functions, loops= 0, maxloops= (try_internals)? 2 : 1;
								  ascanf_Function *lfunList= vars_ascanf_Functions;
									do{
										for( j= 0, hit= 0; !hit && j< llistSize; j++ ){
											af= &lfunList[j];
											if( !(hit= ! strcmp(larg[0], af->name)) ){
												while( af && strcmp( larg[0], af->name ) ){
													  /* 981207: allow for deleted variables to be reused for variables
													   \ with another name also.
													   */
													if( !del_af &&
														(af->type== _ascanf_novariable && function!= ascanf_DeleteVariable &&
														 function!= ascanf_EditProcedure &&
														 (af->function== ascanf_Variable || af->function== ascanf_Procedure))
													){
														  /* 991002: maintain a separation between variables/arrays and procedures!	*/
														if( (af->function== ascanf_Procedure && function== ascanf_DeclareProcedure) ||
															(af->function== ascanf_Variable && function!= ascanf_DeclareProcedure)
														){
															del_af= af;
														}
													}
													af= af->cdr;
												}
												if( af ){
													hit= 1;
												}
											}
										}
										if( try_internals ){
											  /* Must make sure that we won't reuse some variable in the user dictionnary
											   \ 20021101: do that by comparing names, and not just for del_af being set!
											   \ 20021110: nay, just unset del_af before checking the internal list...!
											   */
											del_af= NULL;
											if( af && hit && strcmp( af->name, larg[0]) ){
												hit= 0;
												af= NULL;
											}
											if( !hit ){
												  /* Nothing found - scan internal dict for a match or a reusable item. */
												llistSize= ifN;
												lfunList= viF;
											}
										}
										loops+= 1;
									} while( !(hit || af || del_af) && loops< maxloops );
								}

								if( !hit && !af ){
								  /* 20031029: see if this variable is available in an auto-loadable DyMod: */
									if( Auto_LoadDyMod( AutoLoadTable, AutoLoads, larg[0] ) ){
										  /* gotcha! now we have to jump back, and find the matching variable.
										   \ We have to, as there's more to a DCL[] statement than just definition
										   \ of a variable.
										   */
										goto DCL_search_entry;
									}
								}

								  /* 20031009: for DCL?[] */
								if( hit && af &&
									(af->type== _ascanf_variable || af->type== _ascanf_array
										|| af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats
										|| af->type== _ascanf_python_object )
									&& strncmp(Function->name, "DCL?", 4)== 0
								){
									if( pragma_unlikely(ascanf_verbose) ){
										fprintf( StdErr, "%s%d \t(existing %s variable %s==%s matches \"%s\": command ignored) ",
											pref, (*level), AscanfTypeName(af->type),
											af->name, ad2str(af->value, d3str_format, NULL),
											larg[0]
										);
									}
									n= (C==ascanf_separator)? 2 : 1;
									ok= 1;
									// break from loop
									break;
								}

								if( open_bracket ){
									if( !hit || (af && af->type== _ascanf_novariable) ){
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf,
											"ascanf_function() #%s%d (%s): \"%s\": a variable name can no longer contain a [] pair\n",
											(arglist)? "ac" : "", (*level), (TBARprogress_header)? TBARprogress_header : "", larg[0]
										);
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( ascanf_window && (/* arglist || ascanf_SyntaxCheck ||*/ ascanf_verbose || ascanf_PopupWarn) ){
											POPUP_ERROR( ascanf_window, errbuf, "warning" );
										}
										hit= 1;
										af= NULL;
									}
									else if( !arglist ){
									  /* 20001102: We're not compiling. Very sorry, but at the moment I write this,
									   \ it is time to go to sleep, so I won't put in the code that would replace
									   \ <af> with whatever larg[0] points to, if to anything. It should really be
									   \ that simple - like in compiled_ascanf_function (the 20001102 comment).
									   \ The error warning will remain anyway, in case larg[0] doesn't point to anything
									   \ sensible.
									   */
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf,
											"ascanf_function() #%s%d (%s): \"%s\": [] pairs in the 1st argument to DCL are only allowed while compiling.\n",
											(arglist)? "ac" : "", (*level), (TBARprogress_header)? TBARprogress_header : "", larg[0]
										);
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
											POPUP_ERROR( ascanf_window, errbuf, "warning" );
										}
										hit= 1;
										af= NULL;
									}
									else{
									  /* We found the variable that is to be dereferenced. Restore the bracket now, to be sure
									   \ it is there when the parser needs it.
									   */
										*open_bracket= '[';
										if( af && af->type== _ascanf_variable ){
											fprintf( StdErr, " (warning: [] bracket pair that probably makes no sense in \"%s\" in DCL call)== ",
												larg[0]
											);
										}
										fprintf( StdErr,
											" (warning: dereferenced variable as 1st argument to DCL[]: "
											"further arguments are compiled but not interpreted as initialisations to \"%s\")== ",
											larg[0]
										);
										deref= True;
										  /* We retain open_bracket's current value as it can serve as an indicator that the referenced
										   \ variable is itself dereferenced.
										   */
									}
								}
								if( function== ascanf_DeclareProcedure && C!= ascanf_separator ){
									hit= 1;
									af= NULL;
									if( ascanf_CompilingExpression ){
										fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
										cwarned+= 1;
									}
									sprintf( errbuf,
										"ascanf_function() #%s%d (%s): missing procedure code \"%s\" (%s)\n",
										(arglist)? "ac" : "", (*level), (TBARprogress_header)? TBARprogress_header : "", larg[0], serror()
									);
									fputs( errbuf, StdErr ); fflush( StdErr );
									if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
										POPUP_ERROR( ascanf_window, errbuf, "warning" );
									}
								}
								else if( function== ascanf_EditProcedure ){
									errbuf[0]= '\0';
									if( !hit || !af ){
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf,
											"ascanf_function() #%s%d (%s): missing procedure name \"%s\" (%s)\n",
											(arglist)? "ac" : "", (*level), (TBARprogress_header)? TBARprogress_header : "", larg[0], serror()
										);
									}
									else if( af->type!= _ascanf_procedure || C== ascanf_separator ){
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf,
											"ascanf_function() #%s%d (%s): %s: %s is not a procedure, or followed by another expression\n",
											(arglist)? "ac" : "", (*level), (TBARprogress_header)? TBARprogress_header : "", larg[0],
											af->name
										);
									}
									else if( !arglist ){
										n= EditProcedure( af, /* arg, */ level, larg[0], errbuf );
										if( n<= 0 ){
											n= 1;
										}
									}
									if( errbuf[0] ){
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
											POPUP_ERROR( USEWINDOW, errbuf, "warning" );
										}
									}
								}
								if( !hit && (function== ascanf_DeclareVariable || function== ascanf_DeclareProcedure) ){
								  /* Allocate a new entry.	*/
									if( del_af ){
										af= del_af;
										goto reuse_oldvar;
									}
									else{
										af= (ascanf_Function*) calloc( 1, sizeof(ascanf_Function) );
										af->usage= NULL;
									}
									af->type= (function== ascanf_DeclareProcedure)? _ascanf_procedure : _ascanf_variable;
								}
								else if( hit && af && function!= ascanf_DefinedVariable ){
reuse_oldvar:;
									if( (af->type== _ascanf_novariable && function!= ascanf_DeleteVariable && function!= ascanf_EditProcedure )){
										af->type= (function== ascanf_DeclareProcedure)? _ascanf_procedure : _ascanf_variable;
										if( pragma_unlikely(ascanf_verbose) || verb ){
											fprintf( StdErr, "#%s%d \t\"%s\" reusing previously deleted variable \"%s\"\n",
												(arglist)? "ac" : "", (*level), larg[0], (af->name)? af->name : "??"
											);
										}
										xfree( af->name );
										xfree( af->usage );
										af->function= NULL;
										af->PythonEvalType= 0;
										reusing= True;
									}
									else if( (!(af->type== _ascanf_variable || af->type== _ascanf_array ||
												af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats
												|| af->type== _ascanf_python_object
										) &&
										function== ascanf_DeclareVariable) ||
										(af->type!= _ascanf_procedure && function== ascanf_DeclareProcedure)
									){
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf, "#%s%d \t\"%s\" already in use for function or procedure\n",
											(arglist)? "ac" : "", (*level), larg[0]
										);
										if( af->type== _ascanf_procedure ){
											strcat( errbuf, " Delete[] first to reuse\n" );
											af= NULL;
										}
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
											POPUP_ERROR( USEWINDOW, errbuf, "warning" );
										}
										ascanf_emsg= "(in use)";
										ascanf_arg_error= 1;
									}
								}
								if( af &&
									(af->type== _ascanf_variable || af->type== _ascanf_array || af->type== _ascanf_procedure ||
										af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats
										|| af->type== _ascanf_python_object
									) &&
									(function== ascanf_DeclareVariable || function== ascanf_DeleteVariable ||
										function== ascanf_DeclareProcedure || function== ascanf_EditProcedure
									)
								){
									if( *Allocate_Internal ){
										Allocated_Internal= af;
										af->internal= True;
										af->user_internal= (*Allocate_Internal> 0)? True : False;
									}
									else{
										af->internal= False;
									}
									if( af->function== NULL ){
									  /* Initialise new entry	*/
									  int verb= (index( larg[0], '[' ) && larg[0][0]!= '"')? True : False;
										af->name= strdup(larg[0]);
										af->name_length= strlen(larg[0]);
										af->hash= ascanf_hash(larg[0], NULL);
										af->N= -1* ABS(af->N);
										if( !af->internal || af->user_internal ){
											if( ascanf_VarLabel->usage ){
												af->label= strdup(ascanf_VarLabel->usage);
											}
											else if( lab ){
											  int bl= 0;
											  char *c;
												lab+= strlen(LABEL_OPCODE);
												c= lab;
												while( c && *c && !(*c== '}' && bl==0) ){
													switch(*c){
														case '{':
															bl+= 1;
															break;
														case '}':
															bl-= 1;
															break;
													}
													c++;
												}
												if( *c== '}' ){
													*c= '\0';
													af->label= strdup(lab);
													*c= '}';
												}
											}
										}
										if( (function== ascanf_DeclareProcedure) ){
											af->function= ascanf_Procedure;
											  /* 990119:
											   \ Procedures might make it into real functions someday. For this,
											   \ a call of the type DEFUN[name,code[,args...]] would do. The (local)
											   \ args could be linked to the function's cdr field. The practical problem
											   \ is that such a call would either have to ensure that the local args
											   \ really end up in function's argument list, and not in the global variable
											   \ list. And that afterwards, they will be searched in there (but that can
											   \ be handled in the trivial manner: at evaluation/compile time searching all
											   \ cdr lists in existance that moment.
											   */
											af->Nargs= AMAXARGS;
										}
										else{
											af->function= ascanf_Variable;
											if( C!= ascanf_separator ){
											  char *imsg;
											  /* no args; give it a unique, increasing value.	*/
												if( *ascanf_Variable_init> 0 ){
													af->value= (double) ascanf_Variable_id++;
													imsg= "(id)";
												}
												else if( *ascanf_Variable_init< 0 ){
													af->value= drand();
													imsg= "(random)";
												}
												else if( NaN(*ascanf_Variable_init) ){
													set_NaN( af->value );
													imsg= "";
												}
												else{
													af->value= 0;
													imsg= "";
												}
												arg[0]= af->value;
												af->assigns+= 1;
												n= 1;
												if( (pragma_unlikely(ascanf_verbose>1) || verb) && *Allocate_Internal<= 0 && pAllocate_Internal<= 0 ){
													fprintf( StdErr, "#%s%d \t\"%s\" new label, value=%g%s (hash=%d)\n",
														(arglist)? "ac" : "", (*level), larg[0], arg[0], imsg, af->hash
													);
												}
											}
											else if( (pragma_unlikely(ascanf_verbose>1) || verb) && *Allocate_Internal<= 0 && pAllocate_Internal<= 0 ){
												fprintf( StdErr, "#%s%d \t\"%s\" new variable (hash=%d)\n",
													(arglist)? "ac" : "", (*level), larg[0], af->hash
												);
											}
											af->Nargs= 2;
										}
										if( C== ascanf_separator ){
										  /* 980914: set n..	*/
											n= 2;
										}
										  /* no reason user couldn't define his own $vars..	*/
										af->dollar_variable= 1;
										if( !reusing ){
											if( function== ascanf_DeclareVariable || *Allocate_Internal ){
											  ascanf_Function *hook= funList[0].cdr;
												  /* Variables are "hooked" onto the 1st function entry (DCL).
												   \ Sorted by decreasing hash value
												   */
												if( !hook ){
												  /* 1st ever entry	*/
													af->cdr= funList[0].cdr;
													af->car= &funList[0];
													funList[0].cdr= af;
												}
												else{
												  ascanf_Function *last= NULL;
													while( hook && hook->hash>= af->hash ){
														last= hook;
														hook= hook->cdr;
													}
													if( last ){
														last->cdr= af;
														af->car= last;
														af->cdr= hook;
													}
													else{
													  /* this means the 1st entry (funList[0].cdr) has
													   \ a hash smaller than the new entry's.
													   */
														af->cdr= funList[0].cdr;
														af->car= &funList[0];
														funList[0].cdr= af;
													}
												}
											}
											else if( function== ascanf_DeclareProcedure ){
											  static ascanf_Function *last_defined= NULL;
												  /* Procedures are "hooked" onto the LAST item
												   \ (1st defined..) under the DEPROC function entry.
												   */
												af->cdr= NULL;
												if( !last_defined ){
													Function->cdr= af;
													af->car= Function;
												}
												else{
													last_defined->cdr= af;
													af->car= last_defined;
												}
												last_defined= af;
											}
											if( vars_local_Functions && viF== *vars_local_Functions ){
												  /* local_Functions is a bit different: NOT an index
												   \ as vars_local_Functions is not an array...
												   */
												*local_Functions+= 1;
												if( arglist && !ascanf_noverbose ){
													fprintf( StdErr,
														"#%s%d \t\"%s\" creating local variable \"%s\"\n",
														(arglist)? "ac" : "", (*level), larg[0], af->name
													);
												}
											}
										}
										xtb_popup_delete( &vars_pmenu );
										  /* 20060424: */
// 										register_VariableName(af);
										  // 20100622:
										take_ascanf_address(af);
									}
									if( function== ascanf_DeleteVariable ){
										if( arglist ){
											if( ascanf_verbose ){
												fprintf( StdErr, " (compiling - %s not deleted now)==",
													af->name
												);
											}
											n= 1;
											_fascanf( &n, arg_buf, arg, NULL, NULL, NULL, arglist, level, NULL );
										}
										else{
											if( af->name && af->name[0]!= '$' ){
											  char *pname= NULL;
												PF_sprint_string( &pname, "", "", af->name, False );
												if( ascanf_verbose && af->links> 0 ){
													fprintf( StdErr, " (%s is in use %d times) ", pname, af->links );
												}
												arg[0]= af->value;
												Delete_Variable( af );
												if( ascanf_verbose ){
													fprintf( StdErr, " (deleted %s A=%d R=%d L=%d)== ",
														pname, af->assigns,
														af->reads, af->links
													);
												}
												xfree( pname );
											}
											else{
												ascanf_emsg= "(not deleteable)";
												ascanf_arg_error= 1;
											}
										}
										n= 1;
									}
									else if( function== ascanf_EditProcedure ){
										if( arglist ){
											if( ascanf_verbose ){
												fprintf( StdErr, " (compiling - %s not editing now)==",
													af->name
												);
											}
											n= 1;
											_fascanf( &n, arg_buf, arg, NULL, NULL, NULL, arglist, level, NULL );
										}
									}
									else if( af->function== ascanf_Variable &&
										(!(af->name[0]== '$' && af->dollar_variable== 0) || deref)
									){
									  int m, M;
										if( c ){
											*c= C;
										}
										current_DCL_type= af->type;
										current_DCL_item= af;
											  /* whether compiling or not, check if a new variable is
											   \ going to be an array, and allocate its memory in that
											   \ case.
											   */
										if( C== ascanf_separator ){
										  int nn= n;
										  char *ds, *de= NULL;
										  /* argument(s) to evaluate	*/
											larg[1]= &c[1];
											larg[2]= NULL;
											if( (ds= find_nextquote( larg[1])) ){
											  char *s= &ds[-1];
												while( s> larg[1] && isspace((unsigned char) *s) && *s!= ascanf_separator ){
													s--;
												}
												  /* 20040512: let's suppose that DCL[kk,"foo"] will make kk a string, as
												   \ DCL[kk, "foo"] does!! Thus, we should not accept larg[1][0]=='"'...
												   */
												if( *s== ascanf_separator
													&& ds!= larg[1]
													&& (de= find_nextquote( &ds[1]))
												){
													*de= '\0';
													if( af->usage ){
#if NO20040525
														if( strcmp( af->usage, &ds[1] ) ){
														 /* discard a previous, different string.
														  \ This will give a false alarm if the new string contains
														  \ escaped quotes, but that leaves enough valid hits.
														  */
															xfree( af->usage );
														}
#else
														  /* see 20040525 comment below: */
														xfree( af->usage );
#endif
													}
													af->PythonEvalType= 0;
													if( larg[1][0]== '"' ){
													  /* Only the description argument, no value specified.
													   \ Copy the string immediately, and truncate the input to
													   \ prevent subsequent parsing (and creation of a variable
													   \ named after the description.
													   \ 20040511: won't happen anymore.
													   */
														if( !af->usage ){
															af->usage= strdup_unquote_string( &ds[1] );
														}
														  /* 20040525: moved following line outside previous if block;
														   \ in fact, we don't want that repeating a declaration-with-descr.
														   \ results in changing (e.g.) a scalar into an array with a string
														   \ element!
														   */
														*de= '"';
														s_restore= s;
														s_restore_val= *s;
														*s= '\0';
														if( larg[1][0]== '"' ){
															larg[1]= NULL;
															n= nn= 1;
															goto var_arg_parsed;
														}
													}
													else{
													  /* Store the description argument for later processing,
													   \ iff there is no string present yet (prevents unnecessary
													   \ copying/allocating). Always truncate the input (at *s) to
													   \ prevent subsequent parsing (and creation of a variable
													   \ named after the description.
													   \ 20040512: creation of a new variable is no longer a problem (auto-string-var),
													   \ but we keep the code, deactivated.
													   */
														s_restore= s;
														s_restore_val= *s;
														*s= '\0';
														if( !af->usage ){
															larg[2]= &ds[1];
															  /* 20040512: */
															*de= '"';
														}
														else{
															*de= '"';
														}
													}
												}
											}
											n= AMARGS-1;
											  /* Parse the remaining input, with as many arguments as are
											   \ still available (n). Any description has been cut off
											   \ at this stage.
											   */
											if( s_restore ){
											  /* 20040512: */
												*s_restore= s_restore_val;
											}
											_fascanf( &n, larg[1], &arg[1], NULL, NULL, NULL, NULL, level, NULL);
											if( s_restore ){
											  /* 20040512: */
												*s_restore= '\0';
											}
											arg1= arg[1];
											if( n> 1 && larg[2] && larg[2][-1]== '"' && larg[2][-2]!= '`' ){
											  /* 20040514: this probably matches the situations where we declare
											   \ a variable, initialise it and specify its description. Before the
											   \ recent modifications, n would be 1 less. Make it be. Idem for *de.
											   */
												n-= 1;
												if( de ){
													*de= '\0';
												}
												else{
													fprintf( StdErr,
														"%s::%d(ascanf_function): probably, we ought not be here!\n",
														__FILE__, __LINE__
													);
												}
											}
											if( n== 1 && af->type!= _ascanf_array ){
											  /* handled below */
												if( arglist ){
												  /* "reset" n to process all arguments;
												   \ if not compiling (arglist==NULL), don't because
												   \ we test n==1 again.
												   */
													n= nn;
												}
												if( !af->usage && larg[2] ){
													af->usage= strdup_unquote_string( larg[2] );
												}
											}
											else if( n>= 1 ){
											  int j;
												M= m= (int) arg[1];
												if( M== -2 ){
													  /* 20020530: protect against the new support for -2 index */
													m= af->last_index;
												}
												if( m< 0 && !deref ){
													if( af->N<= 0 ){
														ascanf_xfree( af, af->iarray );
														ascanf_xfree( af, af->array );
														if( pragma_unlikely(ascanf_verbose) ){
															fprintf( StdErr,
																"#%s%d \t\"%s\"[%d] allocating %d elems\n",
																(arglist)? "ac" : "", (*level), larg[0], m, n-1
															);
														}
														m= n- 1;
													}
													else{
														m= af->N;
													}
												}
												  /* 20000514: DCL[array,0,0] resulted in crash because 0 is not > af->N...
												   \ I added the (!af->N && !()) statement.
												   \ 20001102: expansion should only be done when there is more than one argument!
												   \ 20001103: and when the referenced variable is not being dereferenced (open_bracket==0)!
												   */
												if( af->N< 0 || m> af->N ||
													(!af->N && !( (af->name[0]=='%' && af->iarray) || af->array))
												){
												  // 20080717: af->N will be negative for deleted Linked-to arrays ...
												  // but oldN should never go negative!
												  int oldN= MAX(0,af->N);
													if( (n> 1 && !open_bracket) || (af->name[0]=='%' && !af->iarray) || !af->array ){
													  /* This is going to be an array!!	*/
														af->N = m;
#ifdef DEBUG
														if( af->N >= AMARGS-2 ){
															sprintf( errbuf,
																"#%s%d \t\"%s\" NElems=%d > %d can cause problems\n",
																(arglist)? "ac" : "", (*level), larg[0], af->N, AMARGS-2
															);
															if( pragma_unlikely(ascanf_verbose) ){
																fputs( errbuf, StdErr );
															}
														}
#endif
														if( af->N<= 0 ){
															if( ascanf_CompilingExpression ){
																fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
																cwarned+= 1;
															}
															sprintf( errbuf,
																"#%s%d \t\"%s\" allocating %d elems instead of %d\n",
																(arglist)? "ac" : "", (*level), larg[0], 1, af->N
															);
															if( ascanf_window && M!= -2 ){
																POPUP_ERROR( ascanf_window, errbuf, "Error" );
															}
															else{
																fputs( errbuf, StdErr );
															}
															af->N= 1;
														}
														if( af->N> 0 ){
															if( !(af->array || af->iarray) ){
																if( af->name[0]== '%' ){
																	af->iarray= (int*) calloc( af->N, sizeof(int));
																}
																else{
																	af->array= (double*) calloc( af->N, sizeof(double));
																}
															}
															else if( !af->sourceArray ){
															  int i;
																fprintf( StdErr,
																	"#%s%d \t\"%s\" expanding from %d to %d\n",
																	(arglist)? "ac" : "", (*level), larg[0], oldN, af->N
																);
																if( af->name[0]== '%' ){
																	af->iarray= (int*) XGrealloc( af->iarray, af->N* sizeof(int) );
																	for( i= oldN; i< af->N; i++ ){
																		af->iarray[i]= 0;
																	}
																}
																else{
																	af->array= (double*) XGrealloc( af->array, af->N* sizeof(double) );
																	for( i= oldN; i< af->N; i++ ){
																		af->array[i]= 0;
																	}
																}
															}
															else{
																fprintf( StdErr,
																	"#%s%d \t\"%s\" subsetted array: NOT expanding from %d to %d\n",
																	(arglist)? "ac" : "", (*level), larg[0], oldN, af->N
																);
															}
														}
													}
													if( !(af->array || af->iarray) ){
														if( ascanf_CompilingExpression ){
															fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
															cwarned+= 1;
														}
														sprintf( errbuf,
															"#%s%d \t\"%s\" can't allocate %d elems (%s)\n",
															(arglist)? "ac" : "", (*level), larg[0], af->N, serror()
														);
														if( HAVEWIN ){
															POPUP_ERROR( USEWINDOW, errbuf, "Error" );
														}
														else{
															fputs( errbuf, StdErr );
														}
														ascanf_emsg= "(allocation error)";
														ascanf_arg_error= 1;
													}
													else{
														current_DCL_type= af->type= _ascanf_array;
														current_DCL_item= af;
														if( n-1> af->N ){
															if( ascanf_CompilingExpression ){
																fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
																cwarned+= 1;
															}
															sprintf( errbuf,
																"#%s%d \t\"%s{%s}\" %d elems requested, %d specified - excess ignored\n",
																(arglist)? "ac" : "", (*level), af->name, larg[0], af->N, n-1
															);
															if( ascanf_window && ascanf_verbose ){
																POPUP_ERROR( ascanf_window, errbuf, "Warning" );
															}
															else{
																fputs( errbuf, StdErr );
															}
														}
														  /* Make sure that when we're not compiling, the new array gets
														   \ properly initialised - do this by setting arg[1] (the size when
														   \ declaring, the offset when re-initialising!) to 0.
														   \ 20001102: this has been deactivated for some time already!
														arg[1]= 0;
														   */
														m= 0;
														if( arglist ){
															  /* We store the values passed, but only when the destination
															   \ variable is not being dereferenced!
															   */
															if( !open_bracket ){
																if( af->iarray ){
																	for( j= 0; j< af->N && j< n- 1; j++ ){
																		af->iarray[j]= (int) arg[j+2];
																	}
																}
																else{
																	for( j= 0; j< af->N && j< n- 1; j++ ){
																		af->array[j]= arg[j+2];
																	}
																}
																if( !ascanf_noverbose ){
																	fprintf( StdErr,
																		"#%s%d \t\"%s\" array initialised.\n",
																		(arglist)? "ac" : "", (*level), larg[0]
																	);
																}
															}
															n+= 1;
															af->Nargs= af->N+ 1;
														}
														else{
															af->Nargs= af->N+ 1;
														}
													}
												}
												else if( arglist && af->type== _ascanf_array ){
												  /* need to do this...	*/
													n+= 1;
												}
												if( !af->usage && larg[2] ){
													af->usage= strdup_unquote_string( larg[2] );
												}
											}
											else{
												ascanf_arg_error= 1;
												n= nn;
											}
var_arg_parsed:;
										}
										af->sign= sign;
										af->negate= negate;
										if( is_usage && is_address ){
											if( af->type!= _ascanf_variable && af->type!= _ascanf_array &&
												af->type!= _ascanf_procedure
											){
												if( ascanf_CompilingExpression ){
													fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
													cwarned+= 1;
												}
												sprintf( errbuf,
													"ascanf_function() #%s%d: \"%s\" is-string qualifier (`) only allowed for variables and arrays.\n",
													(arglist)? "ac" : "", (*level), arg_buf
												);
												fputs( errbuf, StdErr ); fflush( StdErr );
												if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
													POPUP_ERROR( ascanf_window, errbuf, "warning" );
												}
												is_address= 0;
											}
											else{
												af->is_usage= True;
											}
										}
										if( is_address ){
											  /* 991004: var OR array	*/
											if( af->type== _ascanf_variable || af->type== _ascanf_array ||
												af->type== _ascanf_procedure
											){
												af->is_address= True;
											}
											else{
												if( ascanf_CompilingExpression ){
													fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
													cwarned+= 1;
												}
												sprintf( errbuf,
													"ascanf_function() #%s%d: \"%s\" ignoring pointer qualifier (&)\n",
													(arglist)? "ac" : "", (*level), arg_buf
												);
												fputs( errbuf, StdErr ); fflush( StdErr );
												if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
													POPUP_ERROR( ascanf_window, errbuf, "warning" );
												}
											}
										}
										if( arglist ){
											  /* The variable exists. Do the compilation.	*/
											_fascanf( &n, arg_buf, arg, NULL, NULL, NULL, arglist, level, NULL);
										}
										if( !arglist || *ascanf_AllowSomeCompilingInitialisations ){
											if( C== ascanf_separator &&
												(!arglist || (*arglist)->cdr)
											){
											  Compiled_Form *carg= (arglist)? (*arglist)->cdr : NULL;
											  /* argument(s) to evaluate	*/
											  /* 20000502: arguments can also have been evaluated while compiling (if arglist)
											   \ In this case, the initialisation is probably only valid for constants
											   \ (and other variables, as long as they're constant expressions). Because
											   \ n has been set by the above (compiling) call to fascanf, it includes
											   \ the variable in its count: it must be decreased by 1.
											   */
												if( arglist ){
													n-= 1;
													if( *ascanf_UseConstantsLists && !carg->list_of_constants && pragma_unlikely(ascanf_verbose) ){
														if( ascanf_CompilingExpression ){
															fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
															cwarned+= 1;
														}
														sprintf( errbuf,
															"#%s%d \t\"%s\"==%s,..: compiling initialisation with a list of non-constants"
															" may give wrong values\n",
															(arglist)? "ac" : "", (*level), larg[0], ad2str( (carg)? carg->value : 0, NULL, NULL)
														);
														fputs( errbuf, StdErr ); fflush( StdErr );
														if( ascanf_window && ascanf_PopupWarn> 1 ){
															POPUP_ERROR( ascanf_window, errbuf, "warning" );
														}
													}
												}
												if( af->type!= _ascanf_array ){
													if( n== 1 ){
														if( carg ){
															af->value= carg->value;
														}
														else{
															af->value= arg[1];
															/* arg[0]= arg[1];	*/
														}
														af->assigns+= 1;
														if( af->accessHandler ){
															AccessHandler( af, Function->name, level, NULL, *s, NULL );
														}
														n= 2;
													}
												}
												else{ if( n>= 1 ){
												  int j;
													if( M== -2 ){
														  /* 20020530: new support for -2 index to refer to last_index */
														m= af->last_index;
													}
													if( af->N< 0 ){
													  /* This is already an array!!	*/
													}
													else if( m>= -1 && m< af->N )
access_array_element1:
													{
													  int i= 0;
														current_DCL_type= _ascanf_array;
														current_DCL_item= af;
														if( m>= 0 ){
															if( carg ){
																carg= carg->cdr;
															}
															if( af->iarray ){
																if( !open_bracket ){ if( carg ){
																	for( j= m; j< af->N && i< n-1 && carg; i++, j++ ){
																		af->iarray[j]= (int) carg->value;
																		carg= carg->cdr;
																	}
																}
																else if( !arglist ){
																	for( j= m; j< af->N && i< n-1; i++, j++ ){
																		af->iarray[j]= (int) arg[i+2];
																	}
																} }
																af->value= af->iarray[m];
															}
															else{
																if( !open_bracket ){ if( carg ){
																	for( j= m; j< af->N && i< n-1 && carg; i++, j++ ){
																		af->array[j]= carg->value;
																		carg= carg->cdr;
																	}
																}
																else if( !arglist ){
																	for( j= m; j< af->N && i< n-1; i++, j++ ){
																		af->array[j]= arg[i+2];
																	}
																} }
																af->value= af->array[m];
															}
															af->assigns+= 1;
															if( af->accessHandler && j!= m ){
																af->last_index= m;
																AccessHandler( af, Function->name, level, NULL, *s, NULL );
															}
														}
														else{
															if( i< n-1 ){
																sprintf( ascanf_errmesg, "(assign to index -1 (array-size))" );
																ascanf_emsg= ascanf_errmesg;
																ascanf_arg_error= 1;
															}
															af->value= af->N;
															af->reads+= 1;
														}
														af->last_index= m;
														n+= 1;
													}
													else if( !AllowArrayExpansion && m== af->N ){
													  /* shadow m:	*/
													  int m= 0, i= 0;
														current_DCL_type= _ascanf_array;
														current_DCL_item= af;
														if( carg ){
															carg= carg->cdr;
														}
														if( af->iarray ){
															if( carg ){
																for( j= m; j< af->N && i< n-1 && carg; j++ ){
																	af->iarray[j]= (int) carg->value;
																	  /* 990721: use all specified args, repeating only the last	*/
																	if( i< n-2 ){
																		i++;
																		carg= carg->cdr;
																	}
																}
															}
															else if( !arglist ){
																for( j= m; j< af->N && i< n-1; j++ ){
																	af->iarray[j]= (int) arg[i+2];
																	  /* 990721: use all specified args, repeating only the last	*/
																	if( i< n-2 ){
																		i++;
																	}
																}
															}
															af->value= af->iarray[m];
														}
														else{
															if( carg ){
																for( j= m; j< af->N && i< n-1 && carg; j++ ){
																	af->array[j]= carg->value;
																	  /* 990721: use all specified args, repeating only the last	*/
																	if( i< n-2 ){
																		i++;
																		carg= carg->cdr;
																	}
																}
															}
															else if( !arglist ){
																for( j= m; j< af->N && i< n-1; j++ ){
																	af->array[j]= arg[i+2];
																	  /* 990721: use all specified args, repeating only the last	*/
																	if( i< n-2 ){
																		i++;
																	}
																}
															}
															af->value= af->array[m];
														}
														af->assigns+= 1;
														af->last_index= m;
														if( af->accessHandler ){
															AccessHandler( af, Function->name, level, NULL, *s, NULL );
														}
														n+= 1;
													}
													else{
														if( Inf(arg[1])> 0 ){
															m= af->N-1;
															goto access_array_element1;
														}
														else if( Inf(arg[1])< 0 ){
															m= 0;
															goto access_array_element1;
														}
														else if( AllowArrayExpansion && Resize_ascanf_Array( af, m+1, NULL ) ){
															goto access_array_element1;
														}
														sprintf( ascanf_errmesg, "(index out-of-range [%d,%d> (line %d))",
															0, af->N, __LINE__
														);
														ascanf_emsg= ascanf_errmesg;
														ascanf_arg_error= 1;
														if( af->N< 0 && !arglist ){
															fprintf( StdErr,
																"### Array \"%s\" was deleted. Aborting operations.\n",
																af->name
															);
															ascanf_escape= ascanf_interrupt= True;
															*ascanf_escape_value= *ascanf_interrupt_value= 1;
														}
													}
												}
												else{
													ascanf_arg_error= 1;
												} }
												  /* Restore the original n:	*/
												if( arglist ){
													n+= 1;
												}
											}
											else{
												arg[0]= af->value;
												n= 1;
											}
										}
									}
									else if( af->function== ascanf_Procedure &&
										!(af->name[0]== '$' && af->dollar_variable== 0)
									){
									  Boolean evaluate= True;
										if( c ){
											*c= C;
										}
										current_DCL_type= af->type;
										current_DCL_item= af;
											  /* whether compiling or not, check if a new variable is
											   \ going to be a procedure, and compile its code.
											   */
										if( strcmp( Function->name, "ASKPROC")== 0 ){
											af->dialog_procedure= True;
										}
										else{
											af->dialog_procedure= False;
										}
										if( C== ascanf_separator ){
										  int nn= n;
										  char *de= NULL;
											larg[1]= &c[1];
											  /* 990426: An argument following the code is interpreted as a
											   \ description. It is stripped of its leading comma, and stored
											   \ into the usage field. This should be expanded to cover
											   \ all ascanf variable types. For ascanf_arrays, that would necessitate
											   \ the updating of a pointer to the last parsed argument, since
											   \ in this case multiple arguments are necessary (not just allowed).
											   */
											larg[2]= ascanf_index( &(larg[1][1]), ascanf_separator, NULL);
											if( larg[2] ){
											  char *c= &larg[2][1];
												while( isspace((unsigned char) *c) ){
													c++;
												}
												if( *c== '"' && (de= find_nextquote( &c[1])) ){
													larg[2][0]= '\0';
													larg[2]= &c[1];
													*de= '\0';
												}
												else{
													larg[2]= NULL;
												}
											}
											  /* 20010208: the following clearing of the usage field used to be within
											   \ the if(){} block just below.
											   */
											if( !( larg[2] && af->usage && strcmp( af->usage, larg[2])== 0 ) ){
												xfree( af->usage );
											}
											if( !af->procedure || !af->procedure->expr || strcmp( af->procedure->expr, larg[1]) ){
											  /* DEPROC doesn't evaluate <code> the first time (i.e. when declaring the procedure).
											   \ It does the 2nd time, even though it still re-compiles the code!
											   \ Only in this place do we free a possible usage string, since it shouldn't
											   \ be necessary to re-specify it every time!
											   \ ASKPROC behaves exactly similar, execpt for that it sets the dialog_procedure field.
											   */
												evaluate= False;
											}
											{ ascanf_Function *AI= Allocated_Internal;
											  int AlInt= *Allocate_Internal;
												  /* 20020416: the call to compile_procedure_code() can very well update
												   \ Allocated_Internal in a way that we do not want!
												   */
												compile_procedure_code( &n, larg[1], &arg[1], af, level );
												if( AI ){
													  /* Restore if Allocated_Internal was defined before. */
													Allocated_Internal= AI;
													*Allocate_Internal= AlInt;
												}
											}
											if( n!= 1 ){
												ascanf_arg_error= 1;
											}
											else{
												if( larg[2] && !af->usage ){
													af->usage= strdup_unquote_string( larg[2] );
												}
											}
											n= nn;
										}
										if( strcmp( Function->name, "DEPROC*")== 0 ){
										  /* DEPROC* does evaluate at once	*/
											evaluate= True;
										}
										else if( strcmp( Function->name, "DEPROC-noEval")== 0 ){
										  /* DEPROC-noEval never evaluates	*/
											evaluate= False;
										}
										af->sign= sign;
										af->negate= negate;
										/* if( !arglist )	*/
										{
											if( C== ascanf_separator && evaluate ){
											  /* evaluate the (newly) compiled code:	*/
											  double AA= arg[1];
												n= 1;
												evaluate_procedure( &n, af, &AA, level );
												if( n== 1 ){
													af->value= AA;
													af->assigns+= 1;
													if( af->accessHandler ){
														AccessHandler( af, Function->name, level, NULL, *s, &AA );
													}
													  /* 20000510: preserve arg[1] until after the potential
													   \ call to the accesshandler, but set it now:
													   \ 20031012: nb: af->value and AA can have been changed!
													   */
													arg[1]= AA;
													n= 2;
												}
												else{
													ascanf_arg_error= 1;
												}
											}
											else{
												arg[0]= af->value;
												n= 1;
											}
										}
										/* else	*/ if( arglist ){
											  /* The variable exists. Do the compilation.	*/
											_fascanf( &n, arg_buf, arg, NULL, NULL, NULL, arglist, level, &nargs );
										}
									}
									else{
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf, "#%s%d \t\"%s\" already in use for function or undeleteable variable\n",
											(arglist)? "ac" : "", (*level), larg[0]
										);
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
											POPUP_ERROR( USEWINDOW, errbuf, "warning" );
										}
										ascanf_emsg= "(in use)";
										ascanf_arg_error= 1;
									}
								}
								else if( function== ascanf_DefinedVariable ){
									n= 1;
									if( af && af->type!= _ascanf_novariable
/* 										&& (af->type== _ascanf_variable || af->type== _ascanf_array ||	*/
/* 										af->type== _ascanf_procedure)	*/
									){
										arg[0]= af->value;
										var_Defined= (af->internal)? -1 : 1;
									}
									else{
										set_NaN( arg[0] );
									}
								}
								else{
									if( function== ascanf_DeclareVariable ){
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf,
											"ascanf_function() #%s%d: can't alloc new variable/label \"%s\" (%s)\n",
											(arglist)? "ac" : "", (*level), larg[0], serror()
										);
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
											POPUP_ERROR( USEWINDOW, errbuf, "warning" );
										}
										ascanf_emsg= "(allocation error)";
										ascanf_arg_error= 1;
									}
									else if( function== ascanf_DeclareProcedure ){
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf,
											"ascanf_function() #%s%d: can't alloc new procedure \"%s\" (%s)\n",
											(arglist)? "ac" : "", (*level), larg[0], serror()
										);
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
											POPUP_ERROR( USEWINDOW, errbuf, "warning" );
										}
										ascanf_emsg= "(allocation error)";
										ascanf_arg_error= 1;
									}
									else{
										if( ascanf_CompilingExpression ){
											fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
											cwarned+= 1;
										}
										sprintf( errbuf,
										"ascanf_function() #%s%d: can't delete unexisting variable/label \"%s\" (%s)\n",
											(arglist)? "ac" : "", (*level), larg[0], serror()
										);
										fputs( errbuf, StdErr ); fflush( StdErr );
										if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
											POPUP_ERROR( ascanf_window, errbuf, "warning" );
										}
										arg[0]= 0;
										n= 1;
									}
								}
							}
							else{
								if( ascanf_CompilingExpression ){
									fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
									cwarned+= 1;
								}
								sprintf( errbuf, "ascanf_function() #%s%d: illegal Declare/Delete/Defined[] syntax \"%s\"\n",
									(arglist)? "ac" : "", (*level), arg_buf
								);
								fputs( errbuf, StdErr ); fflush( StdErr );
								if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
									POPUP_ERROR( USEWINDOW, errbuf, "warning" );
								}
								ascanf_emsg= "(syntax error)";
								ascanf_arg_error= 1;
							}
						}
						else if( function== ascanf_SHelp ){
						  char *c= arg_buf, C;
						  int hit= 0;
							larg[0]= arg_buf;
							{ int ii= 0;
								while( isspace( arg_buf[ii]) && arg_buf[ii] ){
									ii+= 1;
								}
								while( arg_buf[ii] && index( PREFIXES, arg_buf[ii]) && hit< 5 ){
									switch( arg_buf[ii] ){
										case '-':
/* 											larg[0]= &arg_buf[ii+1];	*/
											sign*= -1;
											hit+= 1;
											break;
										case '?':
										case '`':
										case '!':
										case '&':
										case '*':
											/* make no sense here	*/
											hit+= 1;
											break;
									}
									ii+= 1;
								}
								larg[0]= &arg_buf[ii];
							}
							if( !larg[0][0]){
								larg[0]= NULL;
								c= NULL;
								ascanf_emsg= "(syntax error)";
								ascanf_arg_error= 1;
							}
							else{
								  /* find the end of the label	*/
								c= find_balanced( c, ascanf_separator, '[', ']', True, NULL  );
							}
							  /* *c=='\0' when no 2nd argument is given.	*/
							if( !c || !*c || *c== ascanf_separator ){
								if( c ){
									C= *c;
									*c= '\0';
								}
								else{
									C= 0;
								}

								arg[0]= DBG_SHelp( larg[0], (int) *Allocate_Internal );
								ascanf_arguments= n= 1;
								if( c ){
									*c= C;
								}
							}
						}
						else if( function== ascanf_compile || function== ascanf_noEval ){
						  Compiled_Form *form= NULL;
							_fascanf( &n, arg_buf, arg, NULL, NULL, NULL, &form, level, &nargs);
							if( n && !ascanf_arg_error ){
								if( pragma_unlikely(ascanf_verbose) ){
								  char hdr[164];
									sprintf( hdr, "#%s%d #C%s:     ",
										(form)? "ac" : "", (*level)+1, (function== ascanf_noEval)? " [no eval] " : ""
									);
/* 									fputs( hdr, StdErr );	*/
/* 									Print_Form( StdErr, &form, 0, True, hdr, NULL, "\n", True );	*/
									fprintf( StdErr, "#%s%d #C%s: %s[\n%s",
										(form)? "ac" : "", (*level), (function== ascanf_noEval)? " [no eval] " : "", Function->name, hdr
									);
									  /* Here we must print <form>, since *s only contains the toplevel expression	*/
									Print_Form( StdErr, &(form->args), 0, True, hdr, NULL, "\n", True );
									fprintf( StdErr, "#%s%d #C%s: ]\n",
										(form)? "ac" : "", (*level), (function== ascanf_noEval)? " [no eval] " : ""
									);
								}
								if( function!= ascanf_noEval ){
									_compiled_fascanf( &n, FORMNAME(form), arg, NULL, NULL, NULL, &form, level );
								}
								Destroy_Form( &form );
							}
						}
						else{
ascanf_function_parse_args:;
							if( !(larg[0] || larg[1] || larg[2]) ){
							  Compiled_Form *pr_arglistval= (arglist)? *arglist : NULL;
								_fascanf( &n, arg_buf, arg, NULL, NULL, NULL, arglist, level, &nargs );
								if( arglist ){
									if( *arglist == pr_arglistval ){
										if( *arglist== NULL ){
											fprintf( StdErr, "%s%d \t%s called with empty argument list while compiling\n",
												pref, (*level), Function->name
											);
											if( (*arglist= (Compiled_Form*) calloc( 1, sizeof(Compiled_Form))) ){
												(*arglist)->type= _ascanf_value;
												set_NaN((*arglist)->value);
												  /* empty_arglist is set to -1 to indicate that *this* arglist is empty.
												   \ check_for_ascanf_function will only then store a True value for this
												   \ flag in the compiled tree. If we just use True everywhere, the flag
												   \ will propagate upwards, which is not what we want.
												   */
												(*arglist)->empty_arglist= -1;
											}
											else{
											  /* RJVB 20061025: this is dangerous, but for the moment I see no other way to signal
											   \ that an empty argument list (something like [,]) was passed.
											   */
												*arglist= (void*) -1;
											}
										}
									}
									else if( *arglist && Function->store ){
										if( (*arglist)->type== _ascanf_array && !(*arglist)->args ){
											fprintf( StdErr, "%s%d \terror: request to store in a non-referenced array (%s): ignored! ##",
												pref, (*level), *s
											);
											Function->store= False;
										}
									}
								}
								else if( last_1st_arg== Function && last_1st_arg_level== *level && Index== 0 ){
									if( Function->type== _ascanf_array ){
										if( n ){
											if( last_1st_arg_tarray_index== -1 ){
												last_1st_arg_tarray_index= (int) arg[0];
											}
										}
										else{
											  /* This means that there was an empty argument list?! */
											fprintf( StdErr, "##%s%d \terror: request to store in a non-referenced array (%s): ignored! ##",
												pref, (*level), *s
											);
											last_1st_arg= NULL;
										}
									}
								}
							}
						}
						if( nargs> Nargs ){
							fprintf( StdErr, "%s%d \t%s%s[%s]: too many arguments (%d>%d)\n",
								pref, (*level), AF_prefix(Function), name, arg_buf,
								nargs, Nargs
							);
						}
						ascanf_arguments= n;
						if( ascanf_update_ArgList && (Function && Function->accessHandler) ){
							if( pragma_unlikely( ascanf_verbose && (arglist || ascanf_SyntaxCheck) ) ){
								fprintf( StdErr, "##%s%d \t(%s/access handler) setting $ args array from %d to %d elements ##",
									pref, (*level), Function->name, af_ArgList->N, n
								);
							}
							SET_AF_ARGLIST( arg, n );
						}

						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "%s%d \t%s%s[%s", pref, (*level), AF_prefix(Function), name, ad2str( arg[0], d3str_format, NULL) );
							for( j= 1; j< n; j++){
								fprintf( StdErr, ",%s", ad2str( arg[j], d3str_format, NULL) );
							}
							fprintf( StdErr, "]== ");
							fflush( StdErr );
						}
						if( ascanf_comment && cfp ){
							fprintf( cfp, "%s%d \t%s%s[%s", pref, (*level), AF_prefix(Function), name, ad2str( arg[0], d3str_format, NULL) );
							for( j= 1; j< n; j++){
								fprintf( cfp, ",%s", ad2str( arg[j], d3str_format, NULL) );
							}
							fprintf( cfp, "]== ");
						}

						current_function_type= Function->type;
						if( !arglist || doSyntaxCheck ){
							if( function== ascanf_Variable ||
								(Function->type== _ascanf_array && function== ascanf_DeclareVariable) ||
								function== ascanf_Procedure
							){
								if( Function->type== _ascanf_variable ){
									ok= (n>0);
									if( pragma_unlikely( ascanf_verbose && doSyntaxCheck ) ){
										fprintf( StdErr,
											"(warning: %s changed from %s to %s because of extended syntax checking!)== ",
											Function->name, ad2str(Function->value,NULL,NULL), ad2str(arg[0],NULL,NULL)
										);
									}
									*A= Function->value= arg[0];
									Function->assigns+= 1;
									if( Function->accessHandler ){
										AccessHandler( Function, Function->name, level, NULL, *s, A );
									}
									if( negate ){
										*A= (*A && !NaN(*A))? 0 : 1;
									}
									else{
										*A *= sign;
									}
									(*ascanf_current_value)= *A;
								}
								else if( Function->type== _ascanf_array ){
								  int first= (int) (function== ascanf_DeclareVariable)? 1 : 0;
								  int m= (int) arg[first];
									ok= (n> first);
									if( ok ){
										if( m== -2 ){
											CLIP_EXPR( m, Function->last_index, 0, Function->N-1 );
										}
										if( m>= -1 && m< Function->N )
access_array_element2:
										{
										  int i= first+1, j;
											if( m>= 0 ){
												if( Function->iarray ){
													for( j= m; j< Function->N && i< n-first; i++, j++ ){
														Function->iarray[j]= (int) arg[i];
													}
												}
												else{
													for( j= m; j< Function->N && i< n-first; i++, j++ ){
														Function->array[j]= arg[i];
													}
												}
												if( pragma_unlikely(ascanf_verbose) ){
													ShowArrayFeedback( StdErr, Function, m, first, n );
												}
												*A= Function->value= (Function->iarray)? Function->iarray[m] : Function->array[m];
												Function->assigns+= 1;
												if( Function->accessHandler && j!= m ){
													Function->last_index= m;
													AccessHandler( Function, Function->name, level, NULL, *s, A );
												}
											}
											else{
												if( i< n-first ){
													sprintf( ascanf_errmesg, "(assign to index -1 (array-size))" );
													ascanf_emsg= ascanf_errmesg;
													ascanf_arg_error= 1;
												}
												*A= Function->value= Function->N;
												Function->reads+= 1;
											}
											Function->last_index= m;
											Function->assigns+= 1;
											if( negate ){
												*A= (*A && !NaN(*A))? 0 : 1;
											}
											else{
												*A *= sign;
											}
											(*ascanf_current_value)= *A;
										}
										else if( !AllowArrayExpansion && m== Function->N ){
										  /* shadow m..	*/
										  int m= 0, i= first+1, j;
											if( Function->iarray ){
												for( j= m; j< Function->N && i< n-first; j++ ){
													  /* 990721: use all specified args, repeating only the last	*/
													Function->iarray[j]= (int) arg[i];
													if( i< n-first-1 ){
														i++;
													}
												}
											}
											else{
												for( j= m; j< Function->N && i< n-first; j++ ){
													Function->array[j]= arg[i];
													if( i< n-first-1 ){
														i++;
													}
												}
											}
											if( pragma_unlikely(ascanf_verbose) ){
												ShowArrayFeedback( StdErr, Function, m, first, n );
											}
											*A= Function->value= (Function->iarray)? Function->iarray[m] : Function->array[m];
										}
										else{
											  // 20090428: probably not a good idea to execute the code below the 1st
											  // time Create_AutoVariabel calls us via check_for_ascanf_function() !
											if( !arglist && !AutoCreating ){
												if( Inf(arg[first])> 0 ){
													m= Function->N-1;
													goto access_array_element2;
												}
												else if( Inf(arg[first])< 0 ){
													m= 0;
													goto access_array_element2;
												}
												else if( AllowArrayExpansion && Resize_ascanf_Array( Function, m+1, NULL ) ){
													goto access_array_element2;
												}
												sprintf( ascanf_errmesg, "(index %s out-of-range [%d,%d> (line %d))",
													ad2str(arg[first], d3str_format, NULL), 0, Function->N, __LINE__
												);
												ascanf_arg_error= 1;
												if( Function->N< 0 ){
													fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n",
														Function->name );
													ascanf_escape= ascanf_interrupt= True;
													*ascanf_escape_value= *ascanf_interrupt_value= 1;
												}
											}
											else{
												sprintf( ascanf_errmesg,
													"(warning: index %s currently out-of-range [%d,%d> (line %d))",
													ad2str(arg[first], d3str_format, NULL), 0, Function->N, __LINE__
												);
											}
											ascanf_emsg= ascanf_errmesg;
											  /* 20050415: */
											set_NaN(Function->value);
											*A= Function->value;
										}
										{ /* 20050415: moved from the end of the if (not the else) above) */
											Function->assigns+= 1;
											Function->last_index= m;
											Function->assigns+= 1;
											if( Function->accessHandler ){
												AccessHandler( Function, Function->name, level, NULL, *s, A );
											}
											if( negate ){
												*A= (*A && !NaN(*A))? 0 : 1;
											}
											else{
												*A *= sign;
											}
											(*ascanf_current_value)= *A;
										}
									}
									else{
										ascanf_arg_error= 1;
									}
								}
								  /* 20000510: this block is new, an attempt to support arguments evaluation for procs	*/
								else if( Function->type== _ascanf_procedure ){
								  int nn= n, uaA= ascanf_update_ArgList;
								  double AA= arg[0];
									if( pragma_unlikely( ascanf_verbose && (arglist || ascanf_SyntaxCheck) ) ){
										fprintf( StdErr, "##%s%d \t(%s) setting $ args array from %d to %d elements ##",
											pref, (*level), Function->name, af_ArgList->N, n
										);
									}
									SET_AF_ARGLIST( arg, n );
									ascanf_update_ArgList= False;
									evaluate_procedure( &n, Function, &AA, level );
									ascanf_update_ArgList= uaA;
									if( n== 1 ){
#if 0
										if( ascanf_verbose && ascanf_SyntaxCheck ){
											fprintf( StdErr,
												"(warning: %s changed from %s to %s because of extended syntax checking!)== ",
												Function->name, ad2str(Function->value,NULL,NULL), ad2str(AA,NULL,NULL)
											);
										}
#endif
										*A= Function->value= AA;
										Function->assigns+= 1;
										if( Function->accessHandler ){
											AccessHandler( Function, Function->name, level, NULL, *s, A );
										}
										if( negate ){
											*A= (*A && !NaN(*A))? 0 : 1;
										}
										else{
											*A *= sign;
										}
										(*ascanf_current_value)= *A;
										ok= 1;
									}
									n= nn;
								}
								else if( Function->type== _ascanf_python_object && dm_python ){
								  double AA;
									if( (*dm_python->ascanf_PythonCall)( Function, n, arg, &AA) ){
										*A= Function->value= AA;
										Function->assigns+= 1;
										if( Function->accessHandler ){
											AccessHandler( Function, Function->name, level, NULL, *s, A );
										}
										if( negate ){
											*A= (*A && !NaN(*A))? 0 : 1;
										}
										else{
											*A *= sign;
										}
										(*ascanf_current_value)= *A;
										ok= 1;
									}
								}
								else if( Function->type== _ascanf_simplestats ){
									ok= (n>0);
									if( !arglist ){
									  int ac= ascanf_arguments;
									  double args[3];
										ascanf_arguments= 2;
										args[1]= arg[0];
										args[2]= 1;
										ascanf_SS_set_bin( Function->SS, 0, Function->name, args, A, 0, Function );
										ascanf_arguments= ac;
										*A= Function->value= Function->SS->last_item;
									}
									else{
										if( pragma_unlikely( ascanf_verbose && doSyntaxCheck ) ){
											fprintf( StdErr,
												"(warning: %s changed from %s to %s because of extended syntax checking!)== ",
												Function->name, ad2str(Function->value,NULL,NULL), ad2str(arg[0],NULL,NULL)
											);
										}
										*A= Function->value= arg[0];
									}
									Function->assigns+= 1;
									if( Function->accessHandler ){
										AccessHandler( Function, Function->name, level, NULL, *s, A );
									}
									if( negate ){
										*A= (*A && !NaN(*A))? 0 : 1;
									}
									else{
										*A *= sign;
									}
									(*ascanf_current_value)= *A;
									if( pragma_unlikely(ascanf_verbose) ){
										if( Function->N> 1 ){
											fprintf( StdErr, " (SS_Add[@[&%s,%d],1,%s,1])== ",
												Function->name, Function->last_index,
												ad2str( Function->value, d3str_format, 0)
											);
										}
										else{
											fprintf( StdErr, " (SS_Add[&%s,1,%s,1])== ", Function->name,
												ad2str( Function->value, d3str_format, 0)
											);
										}
									}
								}
								else if( Function->type== _ascanf_simpleanglestats ){
									ok= (n>0);
/* 									if( !arglist ){	*/
/* 										SAS_Add_Data( Function->SAS, 1, arg[0], 1.0,	*/
/* 											(n>1)? ASCANF_TRUE(arg[1]) : ASCANF_TRUE(*SAS_converts_angle) );	*/
/* 									}	*/
									if( !arglist ){
									  int ac= ascanf_arguments;
									  double args[6];
										ascanf_arguments= 6;
										args[1]= arg[0];
										args[2]= 1;
										args[3]= Function->SAS->Gonio_Base;
										args[4]= Function->SAS->Gonio_Offset;
										if( n> 1 ){
											args[5]= arg[1];
										}
										ascanf_SAS_set_bin( Function->SAS, 0, Function->name, args, A, 0, Function );
										ascanf_arguments= ac;
										*A= Function->value= Function->SAS->last_item;
									}
									else{
										if( pragma_unlikely( ascanf_verbose && doSyntaxCheck) ){
											fprintf( StdErr,
												"(warning: %s changed from %s to %s because of extended syntax checking!)== ",
												Function->name, ad2str(Function->value,NULL,NULL), ad2str(arg[0],NULL,NULL)
											);
										}
										*A= Function->value= arg[0];
									}
									Function->assigns+= 1;
									if( Function->accessHandler ){
										AccessHandler( Function, Function->name, level, NULL, *s, A );
									}
									if( negate ){
										*A= (*A && !NaN(*A))? 0 : 1;
									}
									else{
										*A *= sign;
									}
									(*ascanf_current_value)= *A;
									if( pragma_unlikely(ascanf_verbose && Function->SAS) ){
										fprintf( StdErr, " (SAS_Add[&%s,1,%s,1,%s,%s,%s=%d])== ", Function->name,
											ad2str( Function->value, d3str_format, 0),
											ad2str( Function->SAS->Gonio_Base, d3str_format, 0),
											ad2str( Function->SAS->Gonio_Offset, d3str_format, 0),
											(n>1)? "arg" : "$SAS_converts_angle",
											ASCANF_TRUE_((n>1)? arg[1] : *SAS_converts_angle)
										);
									}
								}
								else if( pragma_unlikely( Function->type== _ascanf_novariable && ascanf_verbose ) ){
									fprintf( StdErr, "DELETED ");
									ok2= False;
								}
							}
							else{
							  int set_fvalue;
								if( function== ascanf_DefinedVariable ){
									ok= 1;
									*A= (double) var_Defined;
									set_fvalue= False;
								}
								else if( (arglist || ascanf_SyntaxCheck) && !doSyntaxCheck ){
									ok= 1;
									*A= 0.0;
									set_fvalue= False;
								}
								else{
#ifdef ASCANF_ALTERNATE
									frame.args= arg;
									frame.result= A;
									frame.level= level;
									frame.compiled= NULL;
									frame.self= Function;
#if defined(ASCANF_ARG_DEBUG)
									frame.expr= *s;
#endif
									ok= (*function)( &frame );
#else
									ok= (*function)( arg, A, level );
#endif
									set_fvalue= True;
								}
								Function->reads+= 1;
								if( set_fvalue ){
									Function->value= *A;
									if( Function->accessHandler ){
										AccessHandler( Function, Function->name, level, NULL, *s, A );
									}
								}
								if( negate ){
									*A= (*A && !NaN(*A))? 0 : 1;
								}
								else{
									*A *= sign;
								}
								(*ascanf_current_value)= *A;
							}
						}
						else{
							ok= 1;
							(*ascanf_current_value)= (double) var_Defined;
						}
						ascanf_check_event( "ascanf_function" );
						_ascanf_loop_counter+= loop_incr;
					} while( *ascanf_loop_ptr> 0 && !ascanf_escape && !arglist );
					ascanf_loop_ptr= awlp;
					ascanf_loop_incr= ali;
					*ascanf_loop_counter= alc;
					*ascanf_switch_case= odref;
					if( ascanf_in_loop && ilb_level== *level ){
						if( *ascanf_in_loop && ascanf_escape ){
							*ascanf_escape_value= ascanf_escape= False;
						}
						ascanf_in_loop= ailp;
					}
					ok2= 1;
					xfree(arg);
				}
				else{
					if( ascanf_CompilingExpression ){
						fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
						cwarned+= 1;
					}
					if( i<= ASCANF_FUNCTION_BUF-1 ){
						snprintf( errbuf, errbuf_len, "%s(\"%s\"): %s arguments list too long (%s)\n",
							caller, *s, Function->name, arg_buf
						);
					}
					else{
						snprintf( errbuf, errbuf_len, "%s(\"%s\"): %s: failure allocation %d arguments (%s)\n",
							caller, *s, Function->name, AMARGS, serror()
						);
					}
					fputs( errbuf, StdErr ); fflush( StdErr );
					if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
						POPUP_ERROR( USEWINDOW, errbuf, "warning" );
					}
				}
			}
			  /* 20040512: */
			if( s_restore ){
				*s_restore= s_restore_val;
			}
			  /* 991014: point s to the first character after the closing brace
			   \ Used to be *s=d !
			   \ Hmmm.. actually seems better to use a local variable when calling check_for_ascanf_function
			   \ (in fascanf()), and storing the end-of-parsed part only upon successful parsing (in s; see
			   \ fascanf()).
			   \ 20020422: NB: s points to the closing brace (cf. note 991014)!!
			   */
			*s= &d[0];
		}
		else{
			if( ascanf_CompilingExpression ){
				fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
				cwarned+= 1;
			}
			sprintf( errbuf, "%s(%s)(\"%s\"): missing ']' (l=%d)\n",
				caller, (TBARprogress_header)? TBARprogress_header : "", *s, strlen(*s)
			);
			fputs( errbuf, StdErr ); fflush( StdErr );
			if( HAVEWIN && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
				POPUP_ERROR( USEWINDOW, errbuf, "warning" );
			}
			ascanf_arg_error= 1;
		}
	}
	else{
	  /* there was no argument list	*/
		*args= '\0';
		ascanf_arguments= 0;
		if( ascanf_update_ArgList && Function && Function->accessHandler ){
			if( pragma_unlikely( ascanf_verbose && (arglist || ascanf_SyntaxCheck) ) ){
				fprintf( StdErr, "##%s%d \t(%s/access handler) setting $ args array from %d to %d elements ##",
					pref, (*level), Function->name, af_ArgList->N, n
				);
			}
			SET_AF_ARGLIST( ascanf_ArgList, 0 );
		}

		if( pragma_unlikely( ascanf_verbose ) ){
			fprintf( StdErr, "%s%d \t%s%s== ", pref, (*level), AF_prefix(Function), name );
			fflush( StdErr );
		}
		if( pragma_unlikely( ascanf_comment && cfp ) ){
			fprintf( cfp, "%s%d \t%s%s== ", pref, (*level), AF_prefix(Function), name );
		}

		if( (arglist || ascanf_SyntaxCheck) && !doSyntaxCheck ){
			ok= 1;
			(*ascanf_current_value)= 0.0;
		}
		else{
			if( function== ascanf_Variable || function== ascanf_Procedure ){
				if( Function->type== _ascanf_variable || Function->type== _ascanf_array ||
					Function->type== _ascanf_simplestats || Function->type== _ascanf_simpleanglestats
				){
					*A= Function->value;
					Function->reads+= 1;
					ok= 1;
					if( last_1st_arg== Function && last_1st_arg_level== *level && Index== 0 ){
						if( Function->type== _ascanf_array ){
							fprintf( StdErr, "##%s%d \terror: request to store in a non-referenced array (%s): ignored! ##",
								pref, (*level), *s
							);
							last_1st_arg= NULL;
							if( arglist ){
								Function->store= False;
							}
						}
					}
				}
				else if( Function->type== _ascanf_procedure ){
				  int nn= n, uaA= ascanf_update_ArgList;
				    /* 20020412: I don't know why I initialised AA to arg[0] -- that value was
					 \ undefined!
					  double AA= arg[0];
					 */
				  double AA= 0;
					n= 1;
					ascanf_update_ArgList= False;
					evaluate_procedure( &n, Function, &AA, level );
					ascanf_update_ArgList= uaA;
					if( n== 1 ){
						*A= Function->value= AA;
						Function->assigns+= 1;
						if( Function->accessHandler ){
							AccessHandler( Function, Function->name, level, NULL, *s, A );
						}
						ok= 1;
					}
					n= nn;
				}
				  /* 20080916 */
				else if( Function->type== _ascanf_python_object && dm_python ){
				  double AA;
					if( (*dm_python->ascanf_PythonCall)( Function, 0, NULL, &AA) ){
						*A= Function->value= AA;
						Function->assigns+= 1;
						if( Function->accessHandler ){
							AccessHandler( Function, Function->name, level, NULL, *s, A );
						}
						if( negate ){
							*A= (*A && !NaN(*A))? 0 : 1;
						}
						else{
							*A *= sign;
						}
						(*ascanf_current_value)= *A;
						ok= 1;
					}
				}
				else if( Function->type== _ascanf_novariable ){
					fprintf( StdErr, "DELETED ");
					ok2= False;
				}
			}
			else{
#ifdef ASCANF_ALTERNATE
				frame.args= NULL;
				frame.result= A;
				frame.level= level;
				frame.compiled= NULL;
				frame.self= Function;
#if defined(ASCANF_ARG_DEBUG)
				frame.expr= *s;
#endif
				ok= (*function)( &frame );
#else
				ok= (*function)( NULL, A, level );
#endif
				Function->value= *A;
				if( Function->accessHandler ){
					AccessHandler( Function, Function->name, level, NULL, *s, A );
				}
			}
			Function->reads+= 1;
			(*ascanf_current_value)= *A;
		}
		ok2= 1;
		ascanf_check_event( "ascanf_function" );
		if( Function->name ){
			*s+= strlen(Function->name);
		}
	}

	if( ok2){
		if( Function->store ){
			  /* 20010503: this was a single if() check:	*/
			if( last_1st_arg && last_1st_arg_level== ((*level)+1) &&
				(last_1st_arg->type== _ascanf_array || last_1st_arg->type== _ascanf_variable)
			){
			  ascanf_Function *af= last_1st_arg;
				if( !arglist ){
					af->value= *A;
					if( af->accessHandler ){
						AccessHandler( af, Function->name, level, NULL, *s, A );
					}
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "(stored in %s", af->name );
					}
					if( af->type== _ascanf_array && (af->iarray || af->array) &&
						last_1st_arg_tarray_index>= 0 && last_1st_arg_tarray_index< af->N
					){
						if( af->iarray ){
							af->iarray[last_1st_arg_tarray_index]= (int) *A;
						}
						else{
							af->array[last_1st_arg_tarray_index]= *A;
						}
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "[%d]", last_1st_arg_tarray_index );
						}
					}
				}
				else{
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "(to be stored in %s", af->name );
					}
					if( af->type== _ascanf_array && (af->iarray || af->array) && af->last_index>= 0 && af->last_index< af->N ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "[%d?]", last_1st_arg_tarray_index );
						}
					}
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fputs( ")== ", StdErr);
				}
			}
			  /* 20010504: whatever we did with the information, restore it as it was when we entered this level: */
			last_1st_arg= l1a;
			last_1st_arg_level= l1al;
		}
#ifdef OUTCOMMENTED
		  /* 20010503: this was always tested when the above single if() failed:	*/
		else if( (*level)== last_1st_arg_level- 1 ){
		  /* didn't pass the test. For whatever reason, restore the memory to the previous value
		   \ 1 frame above where we last set it.. This way, an expression like
		   \ verbose[*add[br,add[br,4]]] gives the desired result.
		   */
			last_1st_arg= l1a;
			last_1st_arg_level= l1al;
		}
#endif
		if( pragma_unlikely(ascanf_verbose) ){
		  ascanf_Function *varray= NULL;
		  Boolean cdr= False;
			if( Function->type== _ascanf_array ){
				varray= Function;
			}
			else if( Function->cdr && (Function->cdr->type== _ascanf_array && Function->cdr->N== arg1 && ascanf_arguments> 1) ){
				varray= Function->cdr;
				cdr= True;
			}
			if( varray ){
			  int j;
				if( varray->iarray ){
					fprintf( StdErr, "%s{%d", (cdr)? varray->name : "", varray->iarray[0]);
					for( j= 1; j< varray->N && (varray->last_index==-1 || j< 48); j++ ){
						fprintf( StdErr, ",%d", varray->iarray[j]);
					}
					if( varray->N> j ){
						fputs( ",...", StdErr );
						for( j= varray->N-3; j< varray->N; j++ ){
							fprintf( StdErr, ",%d", varray->iarray[j] );
						}
					}
				}
				else{
					fprintf( StdErr, "%s{%s",
						(cdr)? varray->name : "",
						ad2str( varray->array[0], d3str_format, NULL)
					);
					for( j= 1; j< varray->N && (varray->last_index== -1 || j< 48); j++ ){
						fprintf( StdErr, ",%s",
							ad2str( varray->array[j], d3str_format, NULL)
						);
					}
					if( varray->N> j ){
						fputs( ",...", StdErr );
						for( j= varray->N-3; j< varray->N; j++ ){
							fprintf( StdErr, ",%s",
								ad2str( varray->array[j], NULL, NULL)
							);
						}
					}
				}
				if( varray->value!= *A ){
					fprintf( StdErr, "}[%d]==%s == ", varray->last_index, ad2str( varray->value, d3str_format, NULL) );
				}
				else{
					fprintf( StdErr, "}[%d]== ", varray->last_index );
				}
			}
			if( ((varray= Function) && varray->cfont) || ((varray= Function->cdr) && varray->cfont) ){
				fprintf( StdErr, "  (cfont: %s\"%s\"/%s[%g])== ",
					(varray->cfont->is_alt_XFont)? "(alt.) " : "",
					varray->cfont->XFont.name, varray->cfont->PSFont, varray->cfont->PSPointSize
				);
			}
			if( Function->type== _ascanf_simplestats ){
			  char buf[256];
				if( Function->N> 1 ){
					if( Function->last_index>= 0 && Function->last_index< Function->N ){
						fprintf( StdErr, "[%d]", Function->last_index );
						SS_sprint_full( buf, d3str_format, " #xb1 ", 0, &Function->SS[Function->last_index] );
					}
					else if( Function->last_index== -1 ){
						sprintf( buf, "%d", Function->N );
					}
				}
				else{
					SS_sprint_full( buf, d3str_format, " #xb1 ", 0, Function->SS );
				}
				fprintf( StdErr, " (%s)== ", buf );
			}
			else if( Function->type== _ascanf_simpleanglestats ){
			  char buf[256];
				if( Function->N> 1 ){
					if( Function->last_index>= 0 && Function->last_index< Function->N ){
						fprintf( StdErr, "[%d]", Function->last_index );
						SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, &Function->SAS[Function->last_index] );
					}
					else if( Function->last_index== -1 ){
						sprintf( buf, "%d", Function->N );
					}
				}
				else{
					SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, Function->SAS );
				}
				fprintf( StdErr, " (%s)== ", buf );
			}
			fprintf( StdErr, "%s%s",
				(function== ascanf_systemtime || function== ascanf_systemtime2)? "<delayed!>" : ad2str( *A, d3str_format, NULL),
				((*level)== 1)? "\t  ," : "\t->"
			);
			if( ascanf_arg_error ){
				if( ascanf_emsg ){
					fprintf( StdErr, " %s", ascanf_emsg );
				}
				else{
					fprintf( StdErr, " (needs %d arguments, has %d)", Function_args(Function), ascanf_arguments );
				}
			}
			fputc( '\n', StdErr );
			fflush( StdErr);
		}
		if( ascanf_comment && cfp ){
		  ascanf_Function *varray= NULL;
		  Boolean cdr= False;
			if( Function->type== _ascanf_array ){
				varray= Function;
			}
			  /* Apparently this works... 	*/
			else if( Function->cdr && (Function->cdr->type== _ascanf_array && Function->cdr->N== arg1 && ascanf_arguments> 1) ){
				varray= Function->cdr;
				cdr= True;
			}
			if( varray ){
			  int j;
				if( varray->iarray ){
					fprintf( cfp, "%s{%d", (cdr)? varray->name : "", varray->iarray[0]);
					for( j= 1; j< varray->N; j++ ){
						fprintf( cfp, ",%d", varray->iarray[j]);
					}
				}
				else{
					fprintf( cfp, "%s{%s",
						(cdr)? varray->name : "",
						ad2str( varray->array[0], d3str_format, NULL)
					);
					for( j= 1; j< varray->N; j++ ){
						fprintf( cfp, ",%s",
							ad2str( varray->array[j], d3str_format, NULL)
						);
					}
				}
				if( varray->value!= *A ){
					fprintf( cfp, "}[%d]==%s == ", varray->last_index, ad2str( varray->value, d3str_format, NULL) );
				}
				else{
					fprintf( cfp, "}[%d]== ", varray->last_index );
				}
			}
			if( Function->type== _ascanf_simplestats ){
			  char buf[256];
				if( Function->N> 1 ){
					if( Function->last_index>= 0 && Function->last_index< Function->N ){
						fprintf( StdErr, "[%d]", Function->last_index );
						SS_sprint_full( buf, d3str_format, " #xb1 ", 0, &Function->SS[Function->last_index] );
					}
					else if( Function->last_index== -1 ){
						sprintf( buf, "%d", Function->N );
					}
				}
				else{
					SS_sprint_full( buf, d3str_format, " #xb1 ", 0, Function->SS );
				}
				fprintf( StdErr, " (%s)== ", buf );
			}
			else if( Function->type== _ascanf_simpleanglestats ){
			  char buf[256];
				if( Function->N> 1 ){
					if( Function->last_index>= 0 && Function->last_index< Function->N ){
						fprintf( StdErr, "[%d]", Function->last_index );
						SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, &Function->SAS[Function->last_index] );
					}
					else if( Function->last_index== -1 ){
						sprintf( buf, "%d", Function->N );
					}
				}
				else{
					SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, Function->SAS );
				}
				fprintf( StdErr, " (%s)== ", buf );
			}
			fprintf( cfp, "%s",
				(function== ascanf_systemtime || function== ascanf_systemtime2)? "<delayed!>" : ad2str( *A, d3str_format, NULL)
			);
			if( ascanf_arg_error ){
				if( ascanf_emsg ){
					fprintf( cfp, " %s", ascanf_emsg );
				}
				else{
					fprintf( cfp, " (needs %d arguments, has %d)", Function_args(Function), ascanf_arguments );
				}
			}
			if( vtimer && !arglist ){
				Elapsed_Since( vtimer, True );
				fprintf( StdErr, " <%g:%gs>", vtimer->Time + vtimer->sTime, vtimer->HRTot_T );
			}
			fprintf( cfp, "%s\n", ((*level)== 1)? "\t  ," : "\t->" );
			fflush( cfp );
		}
	}
	if( (!ok2 || doSyntaxCheck) && ascanf_arg_error ){
		if( !cwarned && ascanf_CompilingExpression ){
			fprintf( StdErr, "#ac%d \t%s :\n", (*level), ascanf_CompilingExpression );
		}
		if( ascanf_emsg ){
			sprintf( errbuf, "%s== %s %s\n", name, ad2str( *A, d3str_format, NULL), ascanf_emsg );
		}
		else{
			sprintf( errbuf, "%s== %s needs %d arguments, has %d\n",
				name, ad2str( *A, d3str_format, NULL), Function_args(Function), ascanf_arguments
			);
		}
		fputs( errbuf, StdErr ); fflush( StdErr );
		if( ascanf_window && (arglist || ascanf_SyntaxCheck || ascanf_PopupWarn) ){
			POPUP_ERROR( ascanf_window, errbuf, "warning" );
		}
	}
	if( function== ascanf_verbose_fnc || function== ascanf_noverbose_fnc || function== ascanf_matherr_fnc ){
		*ascanf_verbose_value= ascanf_verbose= verb;
		matherr_verbose= mverb;
		ascanf_noverbose= anvb;
		if( vtimer && !arglist && function== ascanf_verbose_fnc ){
			xfree(vtimer);
		}
	}
	else if( function== ascanf_global_fnc ){
		*AllowProcedureLocals_value= AllowProcedureLocals= apl;
		vars_local_Functions= vlF;
		local_Functions= lF;
	}
	else if( function== ascanf_IDict_fnc ){
		*Allocate_Internal= AlInt;
		if( ascanf_verbose ){
			fprintf( StdErr, "\n%s%d \tinternal dictionary access switched off\n", pref, *level );
		}
	}
	else if( ((function== ascanf_comment_fnc && ascanf_comment== (*level)) ||
			(function== ascanf_popup_fnc && ascanf_popup== (*level))) && cfp &&
			  /* 990127: if the compiler's output is unwanted, this test should
			   \ be moved to within the rcfp read-loop!
			   */
			!(arglist || ascanf_SyntaxCheck)
	){
	  extern char* add_comment(), *comment_buf;
	  char buf[1024], *c, *cb= comment_buf;
	  extern int comment_size, NoComment;
	  int cs= comment_size, nc= NoComment;
	  Sinc List;
		if( ascanf_popup== (*level) ){
/* 			comment_buf= NULL;	*/
/* 			comment_size= 0;	*/
			List.sinc.string= NULL;
			Sinc_string_behaviour( &List, NULL, 0,0, SString_Dynamic );
			Sflush( &List );
			NoComment= False;
			StdErr= register_FILEsDescriptor(pSE);
		}
		fflush(cfp);
		if( rcfp ){
			while( (c= fgets(buf, 1023, rcfp)) && !feof(rcfp) ){
				if( function== ascanf_popup_fnc ){
					Add_SincList( &List, c, False );
				}
				else{
					add_comment( c );
				}
			}
		}
		else if( pragma_unlikely(debugFlag) ){
			fprintf( StdErr, "ascanf_function(): couldn't open temp file \"%s\" for reading (%s)\n",
				tnam, serror()
			);
		}
		if( ascanf_comment== (*level) ){
			if( !ascanf_popup ){
				fclose( cfp);
				cfp= NULL;
				if( rcfp ){
					fclose( rcfp);
					rcfp= NULL;
				}
			}
				StdErr= register_FILEsDescriptor(cSE);
			ascanf_comment= comm;
			ascanf_use_greek_inf= cugi;
		}
		if( ascanf_popup== (*level) ){
		  int id;
		  char *sel= NULL;
			if( 1 || ascanf_window ){
				if( popup_menu ){
					xtb_popup_delete( &popup_menu );
				}
				id= xtb_popup_menu( ascanf_window, List.sinc.string, "Scope's output", &sel, &popup_menu);
				if( sel ){
					while( *sel && isspace( (unsigned char) *sel) ){
						sel++;
					}
				}
				if( sel && *sel ){
					if( pragma_unlikely(debugFlag) ){
						POPUP_ERROR( ascanf_window, sel, "Copied to clipboard:" );
					}
					else{
						Boing(10);
					}
					XStoreBuffer( disp, sel, strlen(sel), 0);
					  // RJVB 20081217
					xfree(sel);
				}
			}
			else{
				fputs( List.sinc.string, StdErr );
			}
			xfree( List.sinc.string );
			if( !ascanf_comment && cfp ){
				fclose( cfp);
				cfp= NULL;
				if( rcfp ){
					fclose( rcfp);
					rcfp= NULL;
				}
			}
			ascanf_popup= popp;
			ascanf_use_greek_inf= pugi;
			*ascanf_verbose_value= ascanf_verbose= avb;
			comment_buf= cb;
			comment_size= cs;
			NoComment= nc;
		}
/* 		unlink( tnam );	*/
		xfree( tnam );
	}
	else if( (function== ascanf_systemtime || function== ascanf_systemtime2) && timer ){
		Elapsed_Since( timer, True );
		Function->value= (*ascanf_current_value)= *A= timer->Time;
		ascanf_elapsed_values[0]= Delta_Tot_T;
		ascanf_elapsed_values[1]= timer->Time;
		ascanf_elapsed_values[2]= timer->sTime;
		if( Function->accessHandler ){
			AccessHandler( Function, Function->name, level, NULL, *s, A );
		}
		fprintf( StdErr, "## %s time: user %ss, system %ss, total %ss (%s%% CPU)",
			(arglist)? "Compilation" : "Evaluation",
			ad2str( timer->Time, d3str_format, NULL),
			ad2str( timer->sTime, d3str_format, NULL),
			ad2str( timer->HRTot_T, d3str_format, NULL),
			ad2str( 100.0* (timer->Time+timer->sTime)/Delta_Tot_T, d3str_format,0)
		);
		if( ascanf_elapsed_values[3] && !NaNorInf(ascanf_elapsed_values[3]) ){
		  char *prefix= "";
		  double flops= ascanf_elapsed_values[3]/ Delta_Tot_T;
			if( flops> 1e6 ){
				flops*= 1e-6;
				prefix= "M";
			}
			fprintf( StdErr, "; %s \"%sflops\"", ad2str( flops, d3str_format, 0), prefix );
			set_NaN(ascanf_elapsed_values[3]);
		}
		fputs( "\n", StdErr );
		xfree( timer );
		TitleMessage( ActiveWin, NULL );
	}
	SET_AF_ARGLIST( ArgList, Argc );
	ascanf_arguments= aaa;
	ascanf_in_loop= ailp;
	doSyntaxCheck= dSC;
	ascanf_SyntaxCheck= aSC;
	(*level)--;
	GCA();
bail:
	xfree(arg_buf);
	xfree(errbuf);
	return( ok);
#undef POPUP_ERROR
}

/*
int set_ascanf_memory( double d)
{  int i;
	for( i= 0; i< ASCANF_MAX_ARGS; i++ ){
		ascanf_memory[i]= d;
	}
	return( i );
}
 */

typedef enum ObjectTypes{
	_int=0, _double
} ObjectTypes;

typedef struct DoubleOrInt{
	union {
		int *i;
		double *d;
	} p;
	ObjectTypes type;
} DoubleOrInt;

#define get_DOI(doi,n)	(((doi)->type==_int)? (doi)->p.i[n] : (doi)->p.d[n])
#define set_DOI(doi,n,v)	if((doi)->type==_int){ (doi)->p.i[n]=(int)(v);}else{(doi)->p.d[n]=(double)(v);}

int ascanf_array_manip( double *args, double *result, DoubleOrInt *array, int size)
{
	if( !args || ascanf_arguments== 0 ){
		ascanf_arg_error= 0;
		ascanf_emsg= NULL;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (slots) " );
		}
		*result= (double) size;
		return(1);
	}
	if( (args[0]= ssfloor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
	  int i= (int) args[0], N= i+1;
		if( ascanf_arguments>= 2 ){
			if( i== -1 ){
				N= ASCANF_MAX_ARGS;
				i= 0;
			}
			if( !ascanf_SyntaxCheck ){
				for( ; i< N; i++ ){
					set_DOI(array,i, args[1]);
				}
			}
			*result= args[1];
		}
		else{
			if( i== -1 ){
				ascanf_arg_error= 1;
				*result= 0;
			}
			else{
				*result= get_DOI( array, i);
			}
		}
	}
	else{
		ascanf_emsg= "(range error)";
		ascanf_arg_error= 1;
		*result= 0;
	}
	return( 1 );
}

#define CHK_ASCANF_MEMORY	\
	if( !ascanf_memory && !ascanf_SyntaxCheck ){ \
		if( !ASCANF_MAX_ARGS ){ \
			ASCANF_MAX_ARGS= AMAXARGSDEFAULT; \
		} \
		if( !(ascanf_memory= (double*) calloc( ASCANF_MAX_ARGS, sizeof(double))) ){ \
			fprintf( StdErr, "Can't get memory for ascanf_memory[%d] array (%s)\n", ASCANF_MAX_ARGS, serror() ); \
			ascanf_arg_error= 1; \
			return(0); \
		} \
	}

int ascanf_mem ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  DoubleOrInt doi;
	CHK_ASCANF_MEMORY;
	doi.p.d= ascanf_memory;
	doi.type= _double;
	return( ascanf_array_manip( args, result, &doi, ASCANF_MAX_ARGS ) );
}

int ascanf_data ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  DoubleOrInt doi;
	doi.p.d= ascanf_data_buf;
	doi.type= _double;
	return( ascanf_array_manip( args, result, &doi, ASCANF_DATA_COLUMNS ) );
}

int ascanf_column ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  DoubleOrInt doi;
	doi.p.i= ascanf_column_buf;
	doi.type= _int;
	return( ascanf_array_manip( args, result, &doi, ASCANF_DATA_COLUMNS ) );
}

double **ascanf_mxy_buf;
int ascanf_mxy_X= 0, ascanf_mxy_Y= 0;

void free_ascanf_mxy_buf()
{ int i;
	if( ascanf_mxy_X>= 0 && ascanf_mxy_Y>= 0 && ascanf_mxy_buf ){
		for( i= 0; i< ascanf_mxy_X; i++ ){
			xfree( ascanf_mxy_buf[i] );
		}
		xfree( ascanf_mxy_buf );
		ascanf_mxy_buf= NULL;
		ascanf_mxy_X= 0;
		ascanf_mxy_Y= 0;
	}
}

int ascanf_setmxy ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ((args[0]= ssfloor( args[0] ))>= 0 ) &&
		((args[1]= ssfloor( args[1] ))>= 0 ) &&
		args[0]!= ascanf_mxy_X && args[1]!= ascanf_mxy_Y &&
		!ascanf_SyntaxCheck
	){
	  int i, ok;
		free_ascanf_mxy_buf();
		ascanf_mxy_X= (int) args[0];
		ascanf_mxy_Y= (int) args[1];
		if( ascanf_mxy_X && (ascanf_mxy_buf= (double**) calloc( ascanf_mxy_X, sizeof(double*))) ){
			for( ok= 1, i= 0; i< ascanf_mxy_X; i++ ){
				if( !(ascanf_mxy_buf[i]= (double*) calloc( ascanf_mxy_Y, sizeof(double))) ){
					ok= 0;
				}
			}
			*result= (double) ascanf_mxy_X * ascanf_mxy_Y * sizeof(double);
		}
		else{
			ok= 0;
		}
		if( !ok && ascanf_mxy_X && ascanf_mxy_Y ){
			ascanf_emsg= "(allocation error)";
			ascanf_arg_error= 1;
			free_ascanf_mxy_buf();
			*result= 0;
			return( 0 );
		}
		xtb_popup_delete( &vars_pmenu );
	}
	else{
		*result= (double) ascanf_mxy_X * ascanf_mxy_Y * sizeof(double);
	}
	return( 1 );
}

int _ascanf_mxy( double *args, double *result, Boolean cum)
{
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ((args[0]= ssfloor( args[0] ))>= -1 && args[0]< ascanf_mxy_X) &&
		((args[1]= ssfloor( args[1] ))>= -1 && args[1]< ascanf_mxy_Y) &&
		ascanf_mxy_buf
	){
	 int x= (int) args[0], y= (int) args[1], X= x+1, Y= y+1, y0;
		if( ascanf_arguments>= 3 ){
			if( x== -1 ){
			  /* set x-range	*/
				X= ascanf_mxy_X;
				x= 0;
			}
			if( y== -1 ){
			  /* set y-range	*/
				Y= ascanf_mxy_Y;
				y= 0;
			}
			y0= y;
			if( cum ){
				for( ; x< X && !ascanf_SyntaxCheck; x++ ){
					for( y= y0; y< Y; y++ ){
						*result= (ascanf_mxy_buf[x][y]+= args[2]);
					}
				}
			}
			else{
				for( ; x< X && !ascanf_SyntaxCheck; x++ ){
					for( y= y0; y< Y; y++ ){
						ascanf_mxy_buf[x][y]= args[2];
					}
				}
				*result= args[2];
			}
		}
		else{
			if( x== -1 || y== -1 ){
				ascanf_arg_error= 1;
				*result= 0;
			}
			else{
				*result= ascanf_mxy_buf[x][y];
			}
		}
	}
	else{
		ascanf_emsg= "(range error)";
		ascanf_arg_error= 1;
		*result= 0;
	}
	return( 1 );
}

int ascanf_mxy ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( _ascanf_mxy( args, result, False ) );
}

int ascanf_mxy_cum ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( _ascanf_mxy( args, result, True ) );
}

double ***ascanf_mxyz_buf;
int ascanf_mxyz_X= 0, ascanf_mxyz_Y= 0, ascanf_mxyz_Z= 0;

void free_ascanf_mxyz_buf()
{ int i, j;
	if( ascanf_mxyz_X>= 0 && ascanf_mxyz_Y>= 0 && ascanf_mxyz_Z && ascanf_mxyz_buf ){
		for( i= 0; i< ascanf_mxyz_X; i++ ){
			if( ascanf_mxyz_buf[i] ){
				for( j= 0; j< ascanf_mxyz_Y; j++ ){
					xfree( ascanf_mxyz_buf[i][j] );
				}
			}
		}
		xfree( ascanf_mxyz_buf );
		ascanf_mxyz_buf= NULL;
		ascanf_mxyz_X= 0;
		ascanf_mxyz_Y= 0;
		ascanf_mxyz_Z= 0;
	}
}

int ascanf_setmxyz ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ((args[0]= ssfloor( args[0] ))>= 0) && args[0]<= MAXINT &&
		((args[1]= ssfloor( args[1] ))>= 0) && args[1]<= MAXINT &&
		((args[2]= ssfloor( args[2] ))>= 0) && args[2]<= MAXINT &&
		args[0]!= ascanf_mxyz_X && args[1]!= ascanf_mxyz_Y && args[2]!= ascanf_mxyz_Z &&
		!ascanf_SyntaxCheck
	){
	  int i, j, ok= 1;
		free_ascanf_mxyz_buf();
		xtb_popup_delete( &vars_pmenu );
		ascanf_mxyz_X= (int) args[0];
		ascanf_mxyz_Y= (int) args[1];
		ascanf_mxyz_Z= (int) args[2];
		if( ascanf_mxyz_X && ascanf_mxyz_Y && ascanf_mxyz_Z ){
			if( (ascanf_mxyz_buf= (double***) calloc( ascanf_mxyz_X, sizeof(double**))) ){
				for( ok= 1, i= 0; i< ascanf_mxyz_X && ok; i++ ){
					if( !(ascanf_mxyz_buf[i]= (double**) calloc( ascanf_mxyz_Y, sizeof(double*))) ){
						ok= 0;
					}
					else{
						for( j= 0; j< ascanf_mxyz_Y && ok; j++ ){
							if( !(ascanf_mxyz_buf[i][j]= (double*) calloc( ascanf_mxyz_Z, sizeof(double))) ){
								ok= 0;
							}
						}
					}
				}
				*result= (double) ascanf_mxyz_X * ascanf_mxyz_Y * ascanf_mxyz_Z * sizeof(double);
			}
			else{
				ok= 0;
			}
		}
		else if( args[0] || args[1] || args[2] ){
			ascanf_emsg= " (invalid range) ";
			ascanf_arg_error= 1;
			*result= 0;
			return(0);
		}
		if( !ok ){
			ascanf_emsg= "(allocation error)";
			ascanf_arg_error= 1;
			free_ascanf_mxyz_buf();
			*result= 0;
			return( 0 );
		}
	}
	else{
		*result= (double) ascanf_mxyz_X * ascanf_mxyz_Y * ascanf_mxyz_Z * sizeof(double);
	}
	return( 1 );
}

int _ascanf_mxyz( double *args, double *result, Boolean cum)
{
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ((args[0]= ssfloor( args[0] ))>= -1 && args[0]< ascanf_mxyz_X) &&
		((args[1]= ssfloor( args[1] ))>= -1 && args[1]< ascanf_mxyz_Y) &&
		((args[2]= ssfloor( args[2] ))>= -1 && args[2]< ascanf_mxyz_Z) &&
		ascanf_mxyz_buf
	){
	 int x= (int) args[0], y= (int) args[1], z= (int) args[2], X= x+1, Y= y+1, Z= z+1, y0, z0;
		if( ascanf_arguments>= 4 ){
			if( x== -1 ){
			  /* set x-range	*/
				X= ascanf_mxyz_X;
				x= 0;
			}
			if( y== -1 ){
			  /* set y-range	*/
				Y= ascanf_mxyz_Y;
				y= 0;
			}
			if( z== -1 ){
			  /* set z-range	*/
				Z= ascanf_mxyz_Z;
				z= 0;
			}
			y0= y;
			z0= z;
			if( cum ){
				for( ; x< X && !ascanf_SyntaxCheck; x++ ){
					for( y= y0; y< Y; y++ ){
						for( z= z0; z< Z; z++ ){
							*result= (ascanf_mxyz_buf[x][y][z]+= args[3]);
						}
					}
				}
			}
			else{
				for( ; x< X && !ascanf_SyntaxCheck; x++ ){
					for( y= y0; y< Y; y++ ){
						for( z= z0; z< Z; z++ ){
							ascanf_mxyz_buf[x][y][z]= args[3];
						}
					}
				}
				*result= args[3];
			}
		}
		else{
			if( x== -1 || y== -1 || z== -1 ){
				ascanf_arg_error= 1;
				*result= 0;
			}
			else{
				*result= ascanf_mxyz_buf[x][y][z];
			}
		}
	}
	else{
		ascanf_emsg= "(range error)";
		ascanf_arg_error= 1;
		*result= 0;
	}
	return( 1 );
}

int ascanf_mxyz ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( _ascanf_mxyz( args, result, False ) );
}

int ascanf_mxyz_cum ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( _ascanf_mxyz( args, result, True ) );
}

int ascanf_2Dindex ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double x, y, Nx, Ny;
	*result= -1;
	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return(0);
	}
	Nx= ssfloor( args[2] );
	Ny= ssfloor( args[3] );
	if( Nx<= 0 || Ny<= 0 || Nx> MAXINT || Ny> MAXINT ){
		ascanf_arg_error= 1;
		ascanf_emsg= "(range error)";
		return(0);
	}
	x= ssfloor( args[0] );
	y= ssfloor( args[1] );
	if( x< 0 || y< 0 || x>= Nx || y>= Ny ){
		ascanf_arg_error= 1;
		ascanf_emsg= "(index error)";
		return(0);
	}
	*result= y* Nx+ x;
	return(1);
}

int ascanf_nDindex ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL;
  double idx, x, X, Y, fact= 1, newval=0;
  int i, n, n0= 0, N= ascanf_arguments, mdim_array= False, newval_set= False, take_usage;
  static char emsg[128];
  static int default_val= False, reset_default_val= False;
/* 	*result= -1;	*/
	  /* 20050606: return a NaN on error. */
	set_NaN( *result );
	if( args &&
		(af= parse_ascanf_address( args[0], 0, "ascanf_nDindex", (ascanf_verbose)? -1 : 0, &take_usage ))
	){
		if( take_usage ){
		  char *string= af->usage;
		  unsigned long long hash= 0;
			if( string ){
				while( *string ){
					hash+= hash<<3L ^ *string++;
				}
				*result= hash;
			}
			else{
				*result= 0;
			}
			return(1);
		}
		else if( af->type== _ascanf_variable || (af->type== _ascanf_array && af->N) ){
			N-= 1;
			n0= 1;
		}
		else if( af->N && af->type== _ascanf_simplestats ){
			if( (N-= 1)== 2 ){
				newval= args[N];
				newval_set= (default_val)? False : True;
				N-=1;
			}
			else if( N> 2 ){
				ascanf_arg_error= !ascanf_SyntaxCheck;
				ascanf_emsg= " (too many arguments with statsbin variable!) ";
				return(!ascanf_arg_error);
			}
			n0= 1;
		}
		else{
			ascanf_emsg= " (invalid pointer type in first argument: must be a scalar or array) ";
			ascanf_arg_error= !ascanf_SyntaxCheck;
			return(!ascanf_arg_error);
		}
		if( af->type== _ascanf_array && N> 1 ){
			if( ODD(N) || N== 2 ){
				newval= args[N];
				newval_set= (default_val)? False : True;
				N-= 1;
			}
			if( N> 1 && (N< 4 || ODD(N)) ){
				ascanf_emsg= " (there must be exactly 1 or at least 4 indexing arguments) ";
				ascanf_arg_error= !ascanf_SyntaxCheck;
				return(!ascanf_arg_error);
			}
			else{
				mdim_array= True;
			}
		}
	}
	else if( args && ascanf_arguments<= 2 ){
		default_val= ASCANF_TRUE(args[0]);
		if( ascanf_arguments== 2 ){
			reset_default_val= ASCANF_TRUE(args[1]);
		}
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (next %s will return a default value for invalid elements (instead of changing a value)) ",
				(reset_default_val)? "call (only)" : "calls"
			);
		}
	}
	else if( !args || N< 4 || ODD(N) ){
		ascanf_emsg= " (there must be an even number of at least 4 arguments) ";
		ascanf_arg_error= !ascanf_SyntaxCheck;
		if( !ascanf_SyntaxCheck ){
			return(0);
		}
	}
	n= N/2+ n0;
	idx= ssfloor(args[n0]);
	if( ascanf_SyntaxCheck ){
		return(1);
	}
	  /* The following checks make sense only at runtime, when all arguments have their intended (?!) values: */
	if( mdim_array || !af ){
	  int j;
		X= ssfloor(args[n]);
		n0+= 1;
		for( i= 1, j= n0; j< n; i++, j++ ){
			if( (Y= X)< 0 || Y> MAXINT ){
				if( default_val && af ){
					idx= af->N+1;
				}
				else{
					ascanf_arg_error= 1;
					sprintf( emsg, " (index error, arg %d: invalid size %s) ", i+n-1, ad2str(Y, d3str_format,0) );
					ascanf_emsg= emsg;
					return(0);
				}
			}
			  /* 20051112: this pragma_unlikely is doubtful: */
			else if( unused_pragma_unlikely( (x= ssfloor(args[j]))< 0 || x>= (X= ssfloor(args[i+n])) ) ){
				if( default_val && af ){
					idx= af->N+1;
				}
				else{
					ascanf_arg_error= 1;
					sprintf( emsg, " (index error, arg %d: (x=ssfloor(arg %d)=%s) <0 or x>=(ssfloor(arg %d+%d)=%s)) ",
						i, j, ad2str(x, d3str_format,0), i, n, ad2str(X, d3str_format,0)
					);
#if defined(ASCANF_ALTERNATE)
					if( __ascb_frame->compiled ){
						Print_Form( StdErr, &__ascb_frame->compiled, 1, True, NULL, "#\t", NULL, False );
					}
					else
#endif
					if( AH_EXPR ){
						fprintf( StdErr, " (%s) ", AH_EXPR );
					}
					ascanf_emsg= emsg;
					return(0);
				}
			}
			else{
				  /* fact contains the size (product) of the lower dimensions: */
				fact*= Y;
				idx+= x* fact;
			}
		}
	}
	if( af ){
		switch( af->type ){
		case _ascanf_variable:
			if( idx> 0 && pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (nDindex: a variable has only 1 value; element index %d ignored!) ", (int) idx );
			}
			if( newval_set ){
				*result= af->value= newval;
			}
			else{
				*result= af->value;
			}
			break;
		case _ascanf_array:
		case _ascanf_simplestats:
			i= (int) idx;
#ifdef ALWAYS_CHECK_LINKEDARRAYS
			if( af->linkedArray.dataColumn ){
				Check_linkedArray(af);
			}
#endif
			if( i== -2 ){
				CLIP_EXPR( i, af->last_index, 0, af->N-1 );
			}
			  /* 20061017: forbid access to element af->N unless setting a new value */
			if( unused_pragma_unlikely( i>= af->N || i< -1 || ((newval_set || default_val) && i==af->N) ) ){
				if( default_val ){
					*result= newval;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (invalid element requested, returning default value %s) ",
							ad2str(newval, d3str_format, NULL)
						);
					}
				}
				else{
					switch( af->type ){
						case _ascanf_array:
							if( Inf(idx)> 0 ){
								i= af->N-1;
								goto nDindex_access_element;
							}
							else if( Inf(idx)< 0 ){
								i= 0;
								goto nDindex_access_element;
							}
							else if( AllowArrayExpansion && Resize_ascanf_Array( af, i+1, NULL ) ){
								goto nDindex_access_element;
							}
							fprintf( StdErr, " (nDindex: array \"%s\" has %d elements; invalid access to element %d) ",
								af->name, af->N, i );
							ascanf_emsg= " (array index out-of-range) ";
							if( af->N< 0 ){
								fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", af->name );
								ascanf_escape= ascanf_interrupt= True;
								*ascanf_escape_value= *ascanf_interrupt_value= 1;
							}
							break;
						case _ascanf_simplestats:
							fprintf( StdErr, " (nDindex: $SS_StatsBin \"%s\" has %d elements; invalid access to element %d) ",
								af->name, af->N, i );
							ascanf_emsg= " ($SS_StatsBin index out-of-bounds) ";
							break;
					}
					ascanf_arg_error= 1;
					return(0);
				}
			}
			else{
nDindex_access_element:;
				if( pragma_unlikely(ascanf_verbose) ){
					if( newval_set ){
						if( i== -1 ){
							fprintf( StdErr, " (nDindex: illegal assignment to %s[-1])== ", af->name );
							ascanf_arg_error= 1;
							ascanf_emsg= " (assign to element -1) ";
							newval_set= 0;
						}
						else if( af->type== _ascanf_array ){
							fprintf( StdErr, "%s[%d,%s]== ", af->name, (int) i, ad2str(newval, d3str_format,0) );
							if( af->procedure && !AlwaysUpdateAutoArrays ){
								fprintf( StdErr, " (NOT updating automatic array \"%s\", expression %s) ",
									af->name, af->procedure->expr
								);
							}
						}
					}
					else{
						fprintf( StdErr, "%s[%d]== ", af->name, (int) i );
					}
				}
				if( newval_set ){
				  int j, N;
					if( i== af->N ){
						j= 0, N= af->N;
					}
					else{
						j= i, N= i+1;
					}
					switch( af->type ){
						case _ascanf_simplestats:{
						  int ac= ascanf_arguments;
						  double largs[3];
/* 							for( ; j< N; j++ ){	*/
/* 								SS_Add_Data_( af->SS[j], 1, newval, 1.0);	*/
/* 							}	*/
							ascanf_arguments= 2;
							largs[0]= args[0];
							largs[1]= newval;
							largs[2]= 1;
							  /* ! */
							af->last_index= i;
							ascanf_SS_set_bin( af->SS, 0, af->name, largs, result, 0, af );
							ascanf_arguments= ac;
							*result= args[0];
							break;
						}
						case _ascanf_array:
							for( ; j< N; j++ ){
								if( af->iarray ){
									*result= (af->iarray[j]= ssfloor(newval));
								}
								else{
									*result= af->array[j]= newval;
								}
							}
							break;
					}
				}
				else{
					switch( af->type ){
						case _ascanf_simplestats:
							  /* we only do NOT return args[0]==&af when i==-1: @[&ssa,-1] should return N */
							*result= (i== -1)? af->N : args[0];
							break;
						case _ascanf_array:
#ifdef ALWAYS_CHECK_LINKEDARRAYS
							if( af->linkedArray.dataColumn ){
								Check_linkedArray(af);
							}
#endif
							if( af->procedure && !AlwaysUpdateAutoArrays && af->procedure->list_of_constants>= 0 ){
							  int n= af->N, level= -1;
								if( pragma_unlikely(ascanf_verbose>1) ){
									fprintf( StdErr, " (updating automatic array \"%s\", expression %s) ",
										af->name, af->procedure->expr
									);
								}
								af->procedure->level+= 1;
								_compiled_fascanf( &n, af->procedure->expr, af->array, NULL, NULL, NULL, &af->procedure, &level );
								af->procedure->level-= 1;
							}
							*result= (i== -1)? af->N : ((af->iarray)? af->iarray[i] : af->array[i]);
							break;
					}
				}
				af->last_index= i;
			}
			break;
		}
		if( reset_default_val ){
			default_val= False;
			reset_default_val= False;
		}
	}
	else{
		*result= idx;
	}
	return(1);
}

int ascanf_restart ( ASCB_ARGLIST )
{
	raise( SIGUSR2 );
	return 1;
}

int ascanf_exit= 0;
int ascanf_Exit ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_exit= 1;
		*result= ascanf_progn_return;
		return( 1 );
	}
	else{
		if( !ascanf_SyntaxCheck ){
			ascanf_exit= (args[0])? SIGN(args[0]) : 0;
		}
		*result= ascanf_progn_return;
		return( 1 );
	}
}

int ascanf_Discard ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int AddPoint_discard;
	if( args){
		if( !ascanf_SyntaxCheck ){
			AddPoint_discard= (args[0])? SIGN(args[0]) : 0;
		}
		*result= ascanf_progn_return;
		return( 1 );
	}
	return !ascanf_arg_error;
}

int ascanf_Discard2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( args && ascanf_arguments>= 3 ){
	  DataSet *this_set= NULL;
	  int pnr= -1;
		if( args[0]>= 0 && args[0]< setNumber ){
			this_set= &AllSets[ (int) args[0] ];
			if( args[1]>= 0 && args[1]< this_set->numPoints ){
				pnr= (int) args[1];
			}
			else{
				ascanf_emsg= "(invalid point number)";
				ascanf_arg_error= 1;
			}
		}
		else{
			ascanf_emsg= "(invalid setnumber)";
			ascanf_arg_error= 1;
		}
		if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
			DiscardPoint( ActiveWin, this_set, pnr, (args[2])? SIGN(args[2]) : 0 );
		}
		*result= ascanf_progn_return;
		return( 1 );
	}
	else{
		ascanf_arg_error= True;
		*result= 0;
		return(0);
	}
}

int ascanf_Discard3 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int AddPoint_discard;
  int aa= 1;
	if( args && ascanf_arguments>= aa+3 ){
		if( *ascanf_counter>= (int) args[aa+1] && *ascanf_counter<= (int) args[aa+2] ){
			if( !ascanf_SyntaxCheck ){
				AddPoint_discard= (args[aa])? SIGN(args[aa]) : 0;
			}
		}
		aa+= 3;
		while( ascanf_arguments>= aa+3 ){
			if( *ascanf_counter>= (int) args[aa+1] && *ascanf_counter<= (int) args[aa+2] ){
				if( !ascanf_SyntaxCheck ){
					AddPoint_discard= (args[aa])? SIGN(args[aa]) : 0;
				}
			}
			aa+= 3;
		}
		if( ascanf_arguments> aa- 3 && ascanf_arguments< aa ){
			fprintf( StdErr, " (Discard: incomplete argument list)" );
			fflush( StdErr );
		}
		*result= args[0];
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
	}
	return !ascanf_arg_error;
}

int ascanf_Boing ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double p= 0;
	if( ascanf_arguments> 0 ){
		CLIP_EXPR( p, args[0], -100, 100 );
	}
	Boing( (int) p );
	*result= p;
	return( 1 );
}

extern int ascanf_CursorCross ( ASCB_ARGLIST );

int ascanf_CheckEvent ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments> 0 ){
		ascanf_check_int= (int) args[0];
	}
	ascanf_check_now= 0;
	*result= (double) ascanf_check_event( "ascanf_CheckEvent" );
	return( 1 );
}

int ascanf_MaxArguments ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments> 0 && args[0]> 0 && args[0]< MAXINT && !ascanf_SyntaxCheck ){
	  double v= ascanf_verbose;
		if( args[0]< ASCANF_MAX_ARGS && ASCANF_MAX_ARGS == AMAXARGSDEFAULT ){
			ascanf_verbose= True;
			fprintf( StdErr, " Warning: setting from %d to %d below default value %d! ",
				ASCANF_MAX_ARGS, (int) args[0], AMAXARGSDEFAULT
			);
		}
		if( Ascanf_AllocMem(-1) ){
			Ascanf_AllocMem( (int) args[0] );
		}
		else{
			ASCANF_MAX_ARGS= (int) args[0];
		}
		*result= (double) ASCANF_MAX_ARGS;
		ascanf_verbose= v;
	}
	else{
		if( !ASCANF_MAX_ARGS ){
			*result= (double) AMAXARGSDEFAULT;
		}
		else{
			*result= ASCANF_MAX_ARGS;
		}
	}
	return( 1 );
}

#define STRINGCHECK(s,max)	StringCheck(s,max,__FILE__,__LINE__)

int ascanf_SetNumber ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL;
  int take_usage= 0;
	if( args ){
		if( ascanf_arguments> 0 &&
			(af= parse_ascanf_address( args[0], _ascanf_variable, "ascanf_SetNumber", (ascanf_verbose)? -1 : 0, &take_usage )) &&
			take_usage
		){
			;
		}
		else{
			*result= (double) setNumber;
			return(1);
		}
	}
	if( !ascanf_SyntaxCheck && (pragma_unlikely(ascanf_verbose) || af) ){
	  int idx= (int) *ascanf_setNumber, maxidx= (StartUp)? setNumber+1 : setNumber;
	  FILE *fp= (ascanf_verbose)? StdErr : NullDevice;
		if( AllSets && idx>= 0 && idx< maxidx && AllSets[idx].setName ){
		  int n;
			n= fprintf( fp, " (\"%s\"", AllSets[idx].setName );
			if( AllSets[idx].titleText ){
				n+= fprintf( fp, ",\"%s\"", AllSets[idx].titleText );
			}
			n+= fprintf( fp, ")== " );
			if( af){
				xfree( af->usage );
				if( (af->usage= calloc( n- 4, sizeof(char))) ){
					sprintf( af->usage, "\"%s\"", AllSets[idx].setName );
					if( AllSets[idx].titleText ){
						sprintf( af->usage, "%s,\"%s\"", af->usage, AllSets[idx].titleText );
					}
					strcat( af->usage, ")" );
					STRINGCHECK( af->usage, n-4 );
				}
			}
		}
	}
	*result= *ascanf_setNumber;
	return( 1 );
}

#if 0
int ascanf_SetTitle ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL;
  int take_usage= 0;
  static ascanf_Function AF= {NULL};
  static char *AFname= "SetTitle-Static-StringPointer";
  int idx= (int) *ascanf_setNumber;
	af= &AF;
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
	ascanf_arg_error= False;
	if( args ){
		if( args[0]>= 0 && args[0]< setNumber ){
			idx= (int) args[0];
		}
		else if( !ascanf_SyntaxCheck ){
			ascanf_emsg= " (setnumber out of range) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 1 ){
			af= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_SetTitle", (int) ascanf_verbose, &take_usage );
		}
	}
	*result= 0;
	if( af && !ascanf_arg_error && !ascanf_SyntaxCheck ){
		if( !ascanf_SyntaxCheck ){
			if( AllSets && idx>= 0 && idx< setNumber && AllSets[idx].titleText ){
				xfree( af->usage );
				af->usage= strdup( AllSets[idx].titleText);
				*result= take_ascanf_address( af );
			}
		}
		else{
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= take_ascanf_address( af );
		}
	}
	return( !ascanf_arg_error );
}
#else
int ascanf_SetTitle ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL, *naf= NULL;
  int take_usage= 0;
  static ascanf_Function AF= {NULL};
  static char *AFname= "SetTitle-Static-StringPointer";
  int idx= (int) *ascanf_setNumber, parse= False;
	af= &AF;
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
	ascanf_arg_error= False;
	if( args ){
		if( args[0]>= 0 && args[0]< setNumber ){
			idx= (int) args[0];
		}
		else if( !ascanf_SyntaxCheck ){
			ascanf_emsg= " (setnumber out of range) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 1 ){
			if( !(af= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_SetTitle", (int) ascanf_verbose, &take_usage )) ){
				af= &AF;
			}
		}
		if( ascanf_arguments> 2 ){
			if( (naf= parse_ascanf_address( args[2], _ascanf_variable, "ascanf_SetTitle", (int) ascanf_verbose, &take_usage )) &&
				!take_usage
			){
				ascanf_emsg= " (new SetTitle argument must be a stringpointer) ";
				ascanf_arg_error= True;
				naf= NULL;
			}
		}
		if( ascanf_arguments> 3 ){
			parse= (args[3])? True :False;
		}
	}
	*result= 0;
	if( af && !ascanf_arg_error && !ascanf_SyntaxCheck ){
		if( !ascanf_SyntaxCheck ){
			if( AllSets && idx>= 0 && idx< setNumber && naf ){
				xfree( AllSets[idx].titleText );
				AllSets[idx].titleText= (naf->usage)? strdup( naf->usage ) : strdup("");
			}
			if( AllSets && idx>= 0 && idx< setNumber && AllSets[idx].titleText ){
				xfree( af->usage );
				if( parse ){
				  char *ntt, *parsed_end= NULL;
					if( (ntt= ParseTitlestringOpcodes( ActiveWin, idx, AllSets[idx].titleText, &parsed_end )) ){
						af->usage= ntt;
					}
					else{
						af->usage= strdup( AllSets[idx].titleText);
					}
				}
				else{
					af->usage= strdup( AllSets[idx].titleText);
				}
				*result= take_ascanf_address( af );
			}
		}
		else{
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= take_ascanf_address( af );
		}
	}
	return( !ascanf_arg_error );
}
#endif

int ascanf_SetName ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL, *naf= NULL;
  int take_usage= 0;
  static ascanf_Function AF= {NULL};
  static char *AFname= "SetName-Static-StringPointer";
  int idx= (int) *ascanf_setNumber, parse= False;
	af= &AF;
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
	ascanf_arg_error= False;
	if( args ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			idx= (int) args[0];
		}
		else{
			ascanf_emsg= " (setnumber out of range) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 1 ){
			if( !(af= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_SetName", (int) ascanf_verbose, &take_usage )) ){
				af= &AF;
			}
		}
		if( ascanf_arguments> 2 ){
			if( (naf= parse_ascanf_address( args[2], _ascanf_variable, "ascanf_SetName", (int) ascanf_verbose, &take_usage )) &&
				!take_usage
			){
				ascanf_emsg= " (new SetName argument must be a stringpointer) ";
				ascanf_arg_error= True;
				naf= NULL;
			}
		}
		if( ascanf_arguments> 3 ){
			parse= (args[3])? True :False;
		}
	}
	*result= 0;
	if( af && !ascanf_arg_error ){
		if( !ascanf_SyntaxCheck ){
			if( AllSets && idx>= 0 && idx< setNumber && naf ){
				xfree( AllSets[idx].setName );
				AllSets[idx].setName= (naf->usage)? strdup( naf->usage ) : strdup("");
			}
			if( AllSets && idx>= 0 && idx< setNumber && AllSets[idx].setName ){
				xfree( af->usage );
				if( parse ){
				  char *ntt, *parsed_end= NULL;
					if( (ntt= ParseTitlestringOpcodes( ActiveWin, idx, AllSets[idx].setName, &parsed_end )) ){
						af->usage= ntt;
					}
					else{
						af->usage= strdup( AllSets[idx].setName);
					}
				}
				else{
					af->usage= strdup( AllSets[idx].setName);
				}
				*result= take_ascanf_address( af );
			}
		}
		else{
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= take_ascanf_address( af );
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_SetInfo ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL;
  int take_usage= 0;
  static ascanf_Function AF= {NULL};
  static char *AFname= "SetInfo-Static-StringPointer";
  int idx= (int) *ascanf_setNumber;
	ascanf_arg_error= False;
	if( args ){
		if( args[0]>= 0 && args[0]< setNumber ){
			idx= (int) args[0];
		}
		else if( !ascanf_SyntaxCheck ){
			ascanf_emsg= " (setnumber out of range) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 1 && args[1] ){
			af= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_SetInfo", (int) ascanf_verbose, &take_usage );
		}
	}
	if( !af ){
		af= &AF;
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
	*result= 0;
	if( af && !ascanf_SyntaxCheck && !ascanf_arg_error ){
		if( !ascanf_SyntaxCheck ){
			if( AllSets && idx>= 0 && idx< setNumber ){
			  ascanf_Function *naf;
				if( AllSets[idx].set_info ){
					xfree( af->usage );
					af->usage= strdup( AllSets[idx].set_info);
					*result= take_ascanf_address( af );
				}
				if( ascanf_arguments> 2 &&
					(naf= parse_ascanf_address( args[2], _ascanf_variable, "ascanf_SetInfo", (int) ascanf_verbose, &take_usage )) &&
					naf->usage
				){
					xfree( AllSets[idx].set_info );
					AllSets[idx].set_info= strdup( naf->usage );
				}
			}
		}
		else{
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= take_ascanf_address( af );
		}
	}
	return( !ascanf_arg_error );
}

double ascanf_log_zero_x= 0, ascanf_log_zero_y= 0;
int ascanf_LZx ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	*result= ascanf_log_zero_x;
	return( 1 );
}

int ascanf_LZy ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	*result= ascanf_log_zero_y;
	return( 1 );
}

extern int ascanf_xmin ( ASCB_ARGLIST );
extern int ascanf_xmax ( ASCB_ARGLIST );
extern int ascanf_ymin ( ASCB_ARGLIST );
extern int ascanf_ymax ( ASCB_ARGLIST );
extern int ascanf_errmin ( ASCB_ARGLIST );
extern int ascanf_errmax ( ASCB_ARGLIST );
extern int ascanf_tr_xmin ( ASCB_ARGLIST );
extern int ascanf_tr_xmax ( ASCB_ARGLIST );
extern int ascanf_tr_ymin ( ASCB_ARGLIST );
extern int ascanf_tr_ymax ( ASCB_ARGLIST );
extern int ascanf_tr_errmin ( ASCB_ARGLIST );
extern int ascanf_tr_errmax ( ASCB_ARGLIST );
extern int ascanf_curve_len ( ASCB_ARGLIST );
extern int ascanf_error_len ( ASCB_ARGLIST );
extern int ascanf_tr_curve_len ( ASCB_ARGLIST );
extern int ascanf_NumPoints ( ASCB_ARGLIST );
extern int ascanf_titleText ( ASCB_ARGLIST );
extern int ascanf_SetOverlap ( ASCB_ARGLIST );

int ascanf_ncols ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, nval= -1;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
		if( ascanf_arguments> 1 ){
			CLIP_EXPR( nval, (int) args[1], 3, (int) args[1] );
		}
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( idx>= 0 && idx< setNumber ){
		if( nval> 0 && !ascanf_SyntaxCheck && nval!= AllSets[idx].ncols ){
		  extern double **realloc_columns();
			AllSets[idx].columns= realloc_columns( &AllSets[idx], nval );
			Check_Columns( &AllSets[idx] );
		}
		*result= (double) AllSets[idx].ncols;
	}
	else if( idx== -1 ){
	  extern int MaxCols;
		*result= MaxCols;
	}
	else{
		*result= 0;
	}
	return( 1 );
}

extern int ascanf_xcol ( ASCB_ARGLIST );

extern int ascanf_ycol ( ASCB_ARGLIST );

extern int ascanf_ecol ( ASCB_ARGLIST );
extern int ascanf_lcol ( ASCB_ARGLIST );
extern int ascanf_Ncol ( ASCB_ARGLIST );

int ascanf_val ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

   int n, set;
	if( !args || ascanf_arguments== 0 ){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else{
		if( ascanf_arguments>= 2 ){
			CLIP_EXPR(set, (int) *ascanf_setNumber+(int)args[1], 0, setNumber );
		}
		else{
			set= (int) *ascanf_setNumber;
		}
		n= (int) args[0];
		if( *ascanf_counter>= AllSets[set].numPoints ){
			*result= 0;
		}
		else{
			switch( n ){
				case 0:
					*result= AllSets[set].columns[AllSets[set].xcol][(int)*ascanf_counter];
					break;
				case 1:
					*result= AllSets[set].columns[AllSets[set].ycol][(int)*ascanf_counter];
					break;
				case 2:
					*result= AllSets[set].columns[AllSets[set].ecol][(int)*ascanf_counter];
					break;
				case 3:
					if( AllSets[set].lcol>= 0 ){
						*result= AllSets[set].columns[AllSets[set].lcol][(int)*ascanf_counter];
						break;
					}
				default:
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(1);
					break;
			}
		}
		return( 1 );
	}
}

int ascanf_DataVal ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( setNumber ){
	  double set, col, idx;
		CLIP_EXPR(set, args[0], 0, setNumber- 1 );
		CLIP_EXPR(col, args[1], 0, AllSets[(int)set].ncols- 1 );
		CLIP_EXPR(idx, args[2], 0, AllSets[(int)set].numPoints- 1 );
		if( ascanf_arguments> 3 ){
			*result= (AllSets[(int)set].columns[(int)col][(int)idx]= args[3]);
		}
		else{
			*result= AllSets[(int)set].columns[(int)col][(int)idx];
		}
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_DataChanged ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, set;
	if( ascanf_arguments> 0 && args[0]>=0  && args[0]< setNumber ){
		set= (int) args[0];
	}
	else{
		set= (int) *ascanf_setNumber;
	}
	if( ascanf_arguments> 1 && args[1]>=0  && args[1]< AllSets[set].numPoints ){
		set= (int) args[1];
	}
	else{
		set= (int) *ascanf_counter;
	}
	if( idx ){
		if( set>= 0 && set< setNumber && idx< AllSets[set].numPoints ){
		  double *xdata= (AllSets[set].columns[AllSets[set].xcol]);
		  double *ydata= (AllSets[set].columns[AllSets[set].ycol]);
		  double *edata= (AllSets[set].columns[AllSets[set].ecol]);
			if( xdata[idx]!= xdata[idx-1] || ydata[idx]!= ydata[idx-1] || edata[idx]!= edata[idx-1] ){
				*result= 1;
			}
			else{
				if( AllSets[set].lcol>= 0 ){
				  double *vdata= (AllSets[set].columns[AllSets[set].lcol]);
					if( vdata[idx]!= vdata[idx-1] ){
						*result= 1;
					}
					else{
						*result= 0;
					}
				}
				else{
					*result= 0;
				}
			}
		}
		else{
			*result= 0;
		}
	}
	else{
		*result= 1;
	}
	return( 1 );
}

int ascanf_ColumnSame ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( setNumber ){
	  double set, col, nn, val, *column;
	  int i, pnt_nr= (int) *ascanf_counter, n;
	  DataSet *this_set;
		CLIP_EXPR(set, args[0], 0, setNumber- 1 );
		this_set= &AllSets[(int) set];
		CLIP_EXPR(col, args[1], 0, this_set->ncols- 1 );
		nn= args[2];
		if( ascanf_arguments> 3 ){
		 double x;
			CLIP_EXPR(x, args[3], 0, this_set->numPoints- 1 );
			pnt_nr= x;
		}
		if( pnt_nr+ nn< 0 ){
			*result= -1;
			n= -pnt_nr;
		}
		else if( pnt_nr+ nn>= (double) this_set->numPoints ){
			*result= -1;
			n= this_set->numPoints- 1- pnt_nr;
		}
		else{
			*result= 1;
			n= (int) nn;
		}
		column= this_set->columns[(int)col];
		if( ascanf_arguments> 4 ){
			val= args[4];
			if( column[pnt_nr]!= val ){
				*result= 0;
				return(1);
			}
		}
		else{
			val= column[pnt_nr];
		}
		if( n>= 0 ){
			pnt_nr+= 1;
			for( i= 0; i< n && pnt_nr< this_set->numPoints; i++, pnt_nr++ ){
				if( column[pnt_nr]!= val ){
					*result= 0;
				}
			}
		}
		else{
			pnt_nr-= 1;
			for( i= -n; i> 0 && pnt_nr>= 0; i--, pnt_nr-- ){
				if( column[pnt_nr]!= val ){
					*result= 0;
				}
			}
		}
	}
	else{
		*result= 0;
	}
	return( 1 );
}

int ascanf_NumObs ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else{
		if( args[0]>= 0 && args[0]< MaxSets ){
		  int set= (int) args[0];
#ifdef ADVANCED_STATS
			if( ascanf_arguments>= 2 ){
				if( args[1]>= 0 && args[1]< AllSets[set].numPoints ){
#	if ADVANCED_STATS == 1
					*result= (double) AllSets[set].N[(int) args[1]];
#	elif ADVANCED_STATS == 2
					*result= NVAL( &AllSets[set], (int) args[1]);
#	endif
				}
				else{
					*result= 0.0;
				}
				return(1);
			}
#endif
			*result= (double) AllSets[set].NumObs;
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
}

int ascanf_xval ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
	  DataSet *this_set;
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< (this_set= &AllSets[(int)args[0]])->numPoints
		){
			*result= this_set->columns[this_set->xcol][(int) args[1] ];
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_yval ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
	  DataSet *this_set;
#ifdef DEBUG
		if( pragma_unlikely(debugFlag) ){
			fprintf( StdErr, "yval[%s,%s]=",
				ad2str( args[0], d3str_format, NULL), ad2str( args[1], d3str_format, NULL)
			);
			fflush( StdErr );
		}
#endif
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< (this_set= &AllSets[(int)args[0]])->numPoints
		){
			*result= this_set->columns[this_set->ycol][(int) args[1] ];
		}
		else{
			*result= 0;
		}
#ifdef DEBUG
		if( pragma_unlikely(debugFlag) ){
			fprintf( StdErr, " %s\n", ad2str( *result, d3str_format, NULL));
			fflush( StdErr );
		}
#endif
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_error ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
	  DataSet *this_set;
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< (this_set= &AllSets[(int)args[0]])->numPoints
		){
			*result= this_set->columns[this_set->ecol][(int) args[1] ];
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_lval ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
	  DataSet *this_set;
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< (this_set= &AllSets[(int)args[0]])->numPoints &&
			this_set->lcol>= 0
		){
			*result= this_set->columns[this_set->lcol][(int) args[1] ];
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_Nval ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< AllSets[(int)args[0]].numPoints
		){
		  DataSet *this_set= &AllSets[(int) args[0]];
			*result= NVAL( this_set, (int) args[1] );
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_xvec ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< AllSets[(int)args[0]].numPoints
		){
			*result= AllSets[(int)args[0]].xvec[(int) args[1] ];
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_yvec ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
#ifdef DEBUG
		if( pragma_unlikely(debugFlag) ){
			fprintf( StdErr, "yval[%s,%s]=",
				ad2str( args[0], d3str_format, NULL), ad2str( args[1], d3str_format, NULL)
			);
			fflush( StdErr );
		}
#endif
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< AllSets[(int)args[0]].numPoints
		){
			*result= AllSets[(int)args[0]].yvec[(int) args[1] ];
		}
		else{
			*result= 0;
		}
#ifdef DEBUG
		if( pragma_unlikely(debugFlag) ){
			fprintf( StdErr, " %s\n", ad2str( *result, d3str_format, NULL));
			fflush( StdErr );
		}
#endif
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_errvec ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< AllSets[(int)args[0]].numPoints
		){
			*result= AllSets[(int)args[0]].errvec[(int) args[1] ];
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_lvec ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else if( ascanf_arguments>= 2 ){
		if( args[0]>= 0 && args[0]< MaxSets &&
			args[1]>= 0 && args[1]< AllSets[(int)args[0]].numPoints
		){
			*result= AllSets[(int)args[0]].lvec[(int) args[1] ];
		}
		else{
			*result= 0;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= 1;
		return( 1 );
	}
}

int ascanf_Count ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ascanf_arguments ){
		*result= *ascanf_Counter;
	}
	else{
		*result= *ascanf_counter;
	}
	return( 1);
}

int ascanf_Index ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	*result= *ascanf_index_value;
	return( 1);
}

int ascanf_self ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	*result= (*ascanf_self_value);
	return( 1 );
}

int ascanf_current ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	*result= (*ascanf_current_value);
	return( 1 );
}

static double prev_ran= 0;

/* routine for the "ran[low,high]" ascanf syntax	*/
int ascanf_random ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 2 ){
		  /* 20031023, 20031130: 1 argument equals 0 arguments... */
		*result= prev_ran= drand();
		return( 1 );
	}
	else{
	  double cond;
	  int n= 0;
		if( ascanf_arguments> 2 && (cond= ABS(args[2]))< 1 ){
			if( !ascanf_SyntaxCheck ){
				do{
					*result= drand();
					ascanf_check_event( "ascanf_random" );
					n+= 1;
				} while( fabs(*result- prev_ran)<= cond && !ascanf_escape );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (%d discarded polls, diff.=%s) ", n,
						ad2str( *result- prev_ran, d3str_format, NULL )
					);
				}
				prev_ran= *result;
				*result= *result* (args[1]- args[0]) + args[0];
			}
		}
		else{
			if( ascanf_arguments> 2 ){
				if( args[2]< 0 ){
					cond*= drand();
				}
				if( !ascanf_SyntaxCheck ){
					while( cond> 0 ){
						drand();
						cond-= 1;
						n+= 1;
					}
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (%d discarded polls) ", n );
					}
				}
			}
			*result= (prev_ran= drand()) * (args[1]- args[0]) + args[0];
		}
		return( 1 );
	}
}

#ifdef linux

#include <fcntl.h>

static int rndFD= -1;

double kernel_rand()
{ unsigned int rndval;
	read( rndFD, &rndval, sizeof(int) );
	return( ((double)rndval/(double) UINT_MAX) );
}

/* routine for the "kran[low,high]" ascanf syntax	*/
int ascanf_krandom ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 2 ){
		*result= prev_ran= kernel_rand();
		return( 1 );
	}
	else{
	  double cond;
	  int n= 0;
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments> 2 && (cond= ABS(args[2]))< 1 ){
			if( !ascanf_SyntaxCheck ){
				do{
					*result= kernel_rand();
					ascanf_check_event( "ascanf_krandom" );
					n+= 1;
				} while( fabs(*result- prev_ran)<= cond && !ascanf_escape );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (%d discarded polls, diff.=%s) ", n,
						ad2str( *result- prev_ran, d3str_format, NULL )
					);
				}
				prev_ran= *result;
				*result= *result* (args[1]- args[0]) + args[0];
			}
		}
		else{
			if( ascanf_arguments> 2 ){
				if( args[2]< 0 ){
					cond*= kernel_rand();
				}
				if( !ascanf_SyntaxCheck ){
					while( cond> 0 ){
						kernel_rand();
						cond-= 1;
						n+= 1;
					}
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (%d discarded polls) ", n );
					}
				}
			}
			*result= (prev_ran= kernel_rand()) * (args[1]- args[0]) + args[0];
		}
		return( 1 );
	}
}

#else

int ascanf_krandom ( ASCB_ARGLIST )
{
	return( ascanf_random(ASCB_ARGUMENTS) );
}

#endif

static char randomised= 0;

/* routine for the "randomise[low,high]" ascanf syntax	*/
int ascanf_randomise ( ASCB_ARGLIST )
{ long seed;
  int i, n;

	  /* randomise..	*/
	seed= (long)time(NULL);
	srand( (unsigned int) seed);
	srand48( seed);
	n= abs(rand()) % 500;
	for( i= 0; i< n; i++){
		drand();
		rand();
	}
	seed= (unsigned long) (drand()* ULONG_MAX);
	randomised= 1;
	prev_ran= drand();
	return( ascanf_random( ASCB_ARGUMENTS ));
}

IEEEfp zero_div_zero;

#define ARRVAL(a,i)	((a->iarray)? a->iarray[i] : a->array[i])
#define ARRVALb(n,a,i)	((a)?((a->iarray)? a->iarray[i] : a->array[i]):args[n])

int simple_array_op( double *args, int argc, int op, double *result, int *level, Compiled_Form *form )
{ ascanf_Function *dest= NULL, *a= NULL, *b= NULL;
  extern int Resize_ascanf_Array( ascanf_Function *af, int N, double *result );
  double res= 0, sum;
  int i, bN, Nops= 8, NNops= 1, barg= 2, return_sum;
  char ops[]= "=+-*/-|^^", *nops= " =";
  Boolean relat, absol;
	switch( AllowSimpleArrayOps ){
		default:
			sum= 0;
			set_NaN(*result);
			return_sum= True;
			break;
		case 2:
			*result= args[0];
			return_sum= False;
			break;
	}
	if( argc== 1 && (op== 1 || op== 3) &&
		(a= parse_ascanf_address( args[0], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL))
	){
		if( ascanf_arg_error ){
			ascanf_arg_error= 0;
		}
	}
	else if( (op> 0 || op==-1) &&
		(argc== 2
			&& (a= parse_ascanf_address( args[0], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL))
			&& !(b= parse_ascanf_address( args[1], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL))
		)
	){
/* 20030328: accept a numerical (non-array) args[1] such that arrays can be 'operated' with a constant value: */
		if( ascanf_arg_error ){
			ascanf_arg_error= 0;
		}
		barg= 1;
		dest= a;
	}
	else if( op> 0 &&
		(argc== 3
			&& (dest= parse_ascanf_address( args[0], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL))
			&& (a= parse_ascanf_address( args[1], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL))
			&& !(b= parse_ascanf_address( args[2], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL))
		)
	){
/* 20030328: accept a numerical (non-array) args[2], such that arrays can be 'operated' with a constant value: */
		if( ascanf_arg_error ){
			ascanf_arg_error= 0;
		}
	}
	else if( (argc< 2 || (op> 0 && argc< 3)) ||
		!(dest= parse_ascanf_address( args[0], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL)) ||
		!(a= parse_ascanf_address( args[1], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL))
		|| (op> 0 && !(b= parse_ascanf_address( args[2], _ascanf_array, "simple_array_op", (ascanf_verbose)? -1 : 0, NULL)))
	){
		if( pragma_unlikely( debugFlag && ascanf_verbose ) ){
			if( op> 0 ){
				fprintf( StdErr, " (invalid array in operation %s= %s %c %s) ",
					(dest)? dest->name : "?", (a)? a->name : "?", (op< Nops)? ops[op] : '?',
					(b)? b->name : "?"
				);
			}
			else{
				fprintf( StdErr, " (invalid array in operation %s %c %s) ",
					(dest)? dest->name : "?", (-op< NNops)? nops[-op] : '?', (a)? a->name : "?"
				);
			}
			fflush( StdErr );
		}
		return(1);
	}
	if( !b ){
		bN= a->N;
	}
	else{
		bN= b->N;
	}
	if( op== 5 ){
		relat= (ascanf_arguments> 3 && args[3])? True : False;
		absol= (ascanf_arguments> 4 && args[4])? True : False;
	}
	if( pragma_unlikely(ascanf_verbose) ){
		if( argc== 1 ){
			fprintf( StdErr, " (array operation %s[i] %c %s[i+1], N=%d) ",
				a->name, (op< Nops)? ops[op] : '?', a->name, a->N
			);
		}
		else{
			if( op> 0 ){
				fprintf( StdErr, " (array operation %s= %s %c %s, N=%d",
					dest->name, a->name, (op< Nops)? ops[op] : '?', ad2str( args[barg], d3str_format, 0),
					MIN(a->N, bN)
				);
				if( op== 5 ){
					fprintf( StdErr, "%s%s",
						(relat)? " relative" : "",
						(absol)? " absolute" : ""
					);
				}
				fputs( ") ", StdErr );
			}
			else if( op== -1 ){
				fprintf( StdErr, " (array operation %s= %s == %s, N=%d) ",
					dest->name, a->name, ad2str( args[barg], d3str_format, 0),
					MIN(a->N, bN)
				);
			}
			else{
				fprintf( StdErr, " (array operation %s %c %s, N=%d) ",
					dest->name, (-op< NNops)? nops[-op] : '?', a->name,
					MIN(dest->N, a->N)
				);
			}
		}
		fflush( StdErr );
	}
	if( op> 0 && dest && dest!= a ){
		if( !Resize_ascanf_Array( dest, MIN(a->N, bN), &res) ){
			return(2);
		}
	}
	switch( op ){
	  int N;
		case -1:
			return_sum= False;
			*result= 1;
			if( b ){
				if( (N= dest->N)!= a->N ){
					*result= 0;
				}
				for( i= 0; i< N; i++ ){
					if( ARRVAL(dest,i)!= ARRVAL(a,i) ){
						*result= 0;
					}
				}
			}
			else{
				for( i= 0; i< dest->N; i++ ){
					if( ARRVAL(dest,i)!= args[barg] ){
						*result= 0;
					}
				}
			}
			if( !*result ){
				return(0);
			}
			break;
		case 0:{
		  int start= 0, end= a->N, aN= a->N, j, nan_handling, dup;
			if( argc>= 5 && args[3]>= 0 && !isNaN(args[3]) && args[4]>= 0 && !isNaN(args[4]) ){
				CLIP_EXPR( start, (int) args[3], 0, aN-1 );
				CLIP_EXPR( end, (int) args[4], 0, aN-1 );
				end+= 1;
				aN= end - start;
			}
			if( argc>= 6 ){
				nan_handling= (int) args[6];
				if( nan_handling== -1 && dest->iarray ){
				  /* No NaNs in integer arrays: */
					nan_handling= 0;
				}
			}
			else{
				nan_handling= 0;
			}
			if( argc> 2 && ASCANF_TRUE(args[2]) ){
				N= aN;
				dup= True;
			}
			else{
				if( nan_handling== -1 ){
					N= 0;
					for( i= 0; i< dest->N; i++ ){
						if( !isNaN( dest->array[i] ) ){
							N+= 1;
						}
					}
					N= MIN( N, aN );
				}
				else{
					N= MIN( dest->N, aN );
				}
				dup= False;
			}
			if( dup ){
				if( !Resize_ascanf_Array( dest, N, &res) ){
					return(2);
				}
			}
			switch( nan_handling ){
				case 0:
				default:
					if( dest->iarray ){
						for( i= 0, j= start; i< N; i++, j++ ){
							sum+= (dest->iarray[i]= (int) ARRVAL(a,j));
						}
					}
					else{
						for( i= 0, j= start; i< N; i++, j++ ){
							sum+= (dest->array[i]= ARRVAL(a,j));
						}
					}
					break;
				case 1:{
				  double v;
					if( dest->iarray ){
						for( i= 0, j= start; i< N && j< aN; j++ ){
							v= ARRVAL(a,j);
							if( !isNaN(v) ){
								sum+= (dest->iarray[i]= (int) v);
								i++;
							}
						}
					}
					else{
						for( i= 0, j= start; i< N && j< aN; j++ ){
							v= ARRVAL(a,j);
							if( !isNaN(v) ){
								sum+= (dest->array[i]= v);
								i++;
							}
						}
					}
					Resize_ascanf_Array( dest, i, &res);
					break;
				}
				case -1:{
				  double v;
					for( i= 0, j= start; i< N && j< aN; j++ ){
						v= ARRVAL(a,j);
						while( isNaN(dest->array[i]) && i< dest->N ){
							i+= 1;
						}
						sum+= (dest->array[i]= v);
						i++;
					}
					break;
				}
			}
			break;
		}
		case 1:{
			if( argc== 1 ){
				return_sum= False;
				N= a->N;
				if( a->iarray ){
					*result= a->iarray[0];
					for( i= 1; i< N; i++ ){
						*result+= a->iarray[i];
					}
				}
				else{
#ifdef USE_SSE2
					*result = CumSum(a->array, N);
					i = N;
#else
					*result= a->array[0];
					for( i= 1; i< N; i++ ){
						*result+= a->array[i];
					}
#endif
				}
			}
			else{
				N= dest->N;
				if( dest->iarray ){
				  int *iarray= dest->iarray;
					if( b ){
#if defined(_ARRAYVOPS_H)
						ArrayCumAddIArraySSE( &sum, a->iarray, b->iarray, iarray, N );
						i = N;
#else
						for( i= 0; i< N; i++ ){
							sum+= (*iarray++= (int) (ARRVAL(a,i) + ARRVAL(b,i)));
						}
#endif
					}
					else{
						for( i= 0; i< N; i++ ){
							sum+= (*iarray++= (int) (ARRVAL(a,i) + args[barg]));
						}
					}
				}
				else{
				  double *array= dest->array;
					if( b ){
#ifdef _ARRAYVOPS_H0
						// slower!!
						sum += ArrayCumAddArray( a->array, b->array, array, N );
						i = N;
#elif defined(_ARRAYVOPS_H)
						ArrayCumAddArraySSE( &sum, a->array, b->array, array, N );
						i = N;
#else
						for( i= 0; i< N; i++ ){
							sum+= (*array++= ARRVAL(a,i) + ARRVAL(b,i));
						}
#endif
					}
					else{
#ifdef _ARRAYVOPS_H0
						// slower!!
						sum += ArrayCumAddScalar( a->array, args[barg], array, N );
#elif defined(_ARRAYVOPS_H)
						ArrayCumAddScalarSSE( &sum, a->array, args[barg], array, N );
						i = N;
#else
						for( i= 0; i< N; i++ ){
							sum+= (*array++= ARRVAL(a,i) + args[barg]);
						}
#endif
					}
				}
			}
			break;
		}
		case 2:{
			N= dest->N;
			if( dest->iarray ){
			  int *iarray= dest->iarray;
				if( b ){
#if defined(_ARRAYVOPS_H)
					ArrayCumSubIArraySSE( &sum, a->iarray, b->iarray, iarray, N );
					i = N;
#else
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) (ARRVAL(a,i) - ARRVAL(b,i)));
					}
#endif
				}
				else{
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) (ARRVAL(a,i) - args[barg]));
					}
				}
			}
			else{
			  double *array= dest->array;
				if( b ){
#if defined(_ARRAYVOPS_H)
					ArrayCumSubArraySSE( &sum, a->array, b->array, array, N );
					i = N;
#else
					for( i= 0; i< N; i++ ){
						sum+= (*array++= ARRVAL(a,i) - ARRVAL(b,i));
					}
#endif
				}
				else{
#if defined(_ARRAYVOPS_H)
					ArrayCumAddScalarSSE( &sum, a->array, -args[barg], array, N );
					i = N;
#else
					for( i= 0; i< N; i++ ){
						sum+= (*array++= ARRVAL(a,i) - args[barg]);
					}
#endif
				}
			}
			break;
		}
		case 3:{
			if( argc== 1 ){
				return_sum= False;
				N= a->N;
				if( a->iarray ){
					*result= a->iarray[0];
					for( i= 1; i< N; i++ ){
						*result*= a->iarray[i];
					}
				}
				else{
#ifdef USE_SSE2
					*result = CumMul(a->array, N);
					i = N;
#else
					*result= a->array[0];
					for( i= 1; i< N; i++ ){
						*result*= a->array[i];
					}
#endif
				}
			}
			else{
				N= dest->N;
				if( dest->iarray ){
				  int *iarray= dest->iarray;
					if( b ){
						for( i= 0; i< N; i++ ){
							sum+= (*iarray++= (int) (ARRVAL(a,i) * ARRVAL(b,i)));
						}
					}
					else{
						for( i= 0; i< N; i++ ){
							sum+= (*iarray++= (int) (ARRVAL(a,i) * args[barg]));
						}
					}
				}
				else{
				  double *array= dest->array;
					if( b ){
#if defined(_ARRAYVOPS_H)
						ArrayCumMulArraySSE( &sum, a->array, b->array, array, N );
						i = N;
#else
						for( i= 0; i< N; i++ ){
							sum+= (*array++= ARRVAL(a,i) * ARRVAL(b,i));
						}
#endif
					}
					else{
#if defined(_ARRAYVOPS_H)
						ArrayCumMulScalarSSE( &sum, a->array, args[barg], array, N );
						i = N;
#else
						for( i= 0; i< N; i++ ){
							sum+= (*array++= ARRVAL(a,i) * args[barg]);
						}
#endif
					}
				}
			}
			break;
		}
		case 4:{
		  double u, v;
			N= dest->N;
			if( dest->iarray ){
			  int *iarray= dest->iarray;
				for( i= 0; i< N; i++ ){
					u= ARRVAL(a,i), v= ARRVALb(barg,b,i);
					if( !v ){
						if( u ){
							*iarray= SIGN(u)* MAXINT;
						}
						else{
							CLIP_EXPR( *iarray, zero_div_zero.d, -MAXINT, MAXINT);
						}
					}
					else{
						*iarray= (int) u/ v;
					}
					sum+= *iarray++;
				}
			}
			else{
			  double *array= dest->array;
				for( i= 0; i< N; i++ ){
					u= ARRVAL(a,i), v= ARRVALb(barg,b,i);
					if( !v ){
						if( u ){
							set_Inf( (*array), u );
						}
						else{
							*array= zero_div_zero.d;
						}
					}
					else{
						*array= u/ v;
					}
					sum+= *array++;
				}
			}
			break;
		}
		case 5:
			N= dest->N;
			if( dest->iarray ){
			  int *iarray= dest->iarray;
				for( i= 0; i< N; i++ ){
					if( relat ){
					  double r;
						CLIP_EXPR( r, ((ARRVAL(a,i) - ARRVALb(barg,b,i))/ARRVALb(barg,b,i)), -MAXINT+1, MAXINT);
						*iarray= (int) r;
					}
					else{
						*iarray= (int) (ARRVAL(a,i) - ARRVALb(barg,b,i));
					}
					if( absol && *iarray< 0 ){
						(*iarray) *= -1;
					}
					sum+= *iarray++;
				}
			}
			else{
			  double *array= dest->array;
				for( i= 0; i< N; i++ ){
					if( relat ){
						*array= ((ARRVAL(a,i) - ARRVALb(barg,b,i))/ARRVALb(barg,b,i));
					}
					else{
						*array= ARRVAL(a,i) - ARRVALb(barg,b,i);
					}
					if( absol && *array< 0 ){
						(*array) *= -1;
					}
					sum+= *array++;
				}
			}
			break;
		case 6:
			N= dest->N;
			if( dest->iarray ){
			  int *iarray= dest->iarray;
				if( b ){
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) sqrt( pow(ARRVAL(a,i),2) + pow(ARRVAL(b,i),2)) );
					}
				}
				else{
				  double barg2 = pow(args[barg],2);
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) sqrt( pow(ARRVAL(a,i),2) + barg2) );
					}
				}
			}
			else{
			  double *array= dest->array;
				if( b ){
#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
					{ int N_1 = N - 1;
					  __m128d va, vb, vlen;
						for( i = 0 ; i < N ; i += 2 ){
							if( pragma_unlikely(i == N_1) ){
								if( pragma_unlikely(a->iarray) ){
									va = _MM_SET1_PD( (double) a->iarray[i] );
								}
								else{
									va = _MM_SET1_PD( a->array[i] );
								}
								if( pragma_unlikely(b->iarray) ){
									vb = _MM_SET1_PD( (double) b->iarray[i] );
								}
								else{
									vb = _MM_SET1_PD( b->array[i] );
								}
								vlen = _mm_sqrt_pd( _mm_add_pd(
									_mm_mul_pd(va,va), _mm_mul_pd(vb,vb) ) );
								sum += (*array++ = ((double*)&vlen)[0]);
							}
							else{
								if( pragma_unlikely(a->iarray) ){
									va = (__m128d){ (double) a->iarray[i], (double) a->iarray[i+1] };
								}
								else{
									va = (__m128d){ a->array[i], a->array[i+1] };
								}
								if( pragma_unlikely(b->iarray) ){
									vb = (__m128d){ (double) b->iarray[i], (double) b->iarray[i+1] };
								}
								else{
									vb = (__m128d){ b->array[i], b->array[i+1] };
								}
								vlen = _mm_sqrt_pd( _mm_add_pd(
									_mm_mul_pd(va,va), _mm_mul_pd(vb,vb) ) );
								sum += (*array++ = ((double*)&vlen)[0]);
								sum += (*array++ = ((double*)&vlen)[1]);
							}
						}
					}
#else
					for( i= 0; i< N; i++ ){
						sum+= (*array++= sqrt( pow(ARRVAL(a,i),2) + pow(ARRVAL(b,i),2) ) );
					}
#endif
				}
				else{
				  double barg2 = pow(args[barg],2);
					for( i= 0; i< N; i++ ){
						sum+= (*array++= sqrt( pow(ARRVAL(a,i),2) + barg2 ) );
					}
				}
			}
			break;
		case 7:
			MATHERR_MARK();
			N= dest->N;
			if( dest->iarray ){
			  int *iarray= dest->iarray;
				if( b ){
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) pow(ARRVAL(a,i), ARRVAL(b,i)) );
					}
				}
				else{
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) pow(ARRVAL(a,i), args[barg]) );
					}
				}
			}
			else{
			  double *array= dest->array;
				if( b ){
#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__)) && defined(__VECLIB__0)
// slower!!
					{ int N_1 = N - 1, nn;
					  __m128d va, vb, vlen;
						for( i = 0 ; i < N ; i += 2 ){
							if( pragma_unlikely(i == N_1) ){
								nn = 1;
								if( pragma_unlikely(a->iarray) ){
									va = _MM_SET1_PD( a->iarray[i] );
								}
								else{
									va = _MM_SET1_PD( a->array[i] );
								}
								if( pragma_unlikely(b->iarray) ){
									vb = _MM_SET1_PD( b->iarray[i] );
								}
								else{
									vb = _MM_SET1_PD( b->array[i] );
								}
								vvpow( (double*) &vlen, (const double*) &vb, (const double*) &va, &nn );
								sum += (*array++ = ((double*)&vlen)[0]);
							}
							else{
								nn = 2;
								if( pragma_unlikely(a->iarray) ){
									va = (__m128d){ (double) a->iarray[i], (double) a->iarray[i+1] };
								}
								else{
									va = (__m128d){ a->array[i], a->array[i+1] };
								}
								if( pragma_unlikely(b->iarray) ){
									vb = (__m128d){ (double) b->iarray[i], (double) b->iarray[i+1] };
								}
								else{
									vb = (__m128d){ b->array[i], b->array[i+1] };
								}
								vvpow( (double*) &vlen, (const double*) &vb, (const double*) &va, &nn );
								sum += (*array++ = ((double*)&vlen)[0]);
								sum += (*array++ = ((double*)&vlen)[1]);
							}
						}
					}
#else
					for( i= 0; i< N; i++ ){
						sum+= (*array++= pow(ARRVAL(a,i), ARRVAL(b,i)) );
					}
#endif
				}
				else{
#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__)) && defined(__VECLIB__0)
// slower!
					{ int N_1 = N - 1, nn;
					  __m128d va, vb, vlen;
//						vb = _mm_set_pd( args[barg], args[barg] );
						vb = (__m128d){ args[barg], args[barg] };
						for( i = 0 ; i < N ; i += 2 ){
							if( pragma_unlikely(i == N_1) ){
								nn = 1;
								if( pragma_unlikely(a->iarray) ){
									va = _MM_SET1_PD( a->iarray[i] );
								}
								else{
									va = _MM_SET1_PD( a->array[i] );
								}
								vvpow( (double*) &vlen, (const double*) &vb, (const double*) &va, &nn );
								sum += (*array++ = ((double*)&vlen)[0]);
							}
							else{
								nn = 2;
								if( pragma_unlikely(a->iarray) ){
									va = (__m128d){ (double) a->iarray[i], (double) a->iarray[i+1] };
								}
								else{
									va = (__m128d){ a->array[i], a->array[i+1] };
								}
								vvpow( (double*) &vlen, (const double*) &vb, (const double*) &va, &nn );
								sum += (*array++ = ((double*)&vlen)[0]);
								sum += (*array++ = ((double*)&vlen)[1]);
							}
						}
					}
#else
					for( i= 0; i< N; i++ ){
						sum+= (*array++= pow(ARRVAL(a,i), args[barg]) );
					}
#endif
				}
			}
			break;
		case 8:
			MATHERR_MARK();
			N= dest->N;
			if( dest->iarray ){
			  int *iarray= dest->iarray;
				if( b ){
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) pow_(ARRVAL(a,i), ARRVAL(b,i)) );
					}
				}
				else{
					for( i= 0; i< N; i++ ){
						sum+= (*iarray++= (int) pow_(ARRVAL(a,i), args[barg]) );
					}
				}
			}
			else{
			  double *array= dest->array;
				if( b ){
					for( i= 0; i< N; i++ ){
						sum+= (*array++= pow_(ARRVAL(a,i), ARRVAL(b,i)) );
					}
				}
				else{
					for( i= 0; i< N; i++ ){
						sum+= (*array++= pow_(ARRVAL(a,i), args[barg]) );
					}
				}
			}
			break;
		default:
			fprintf( StdErr, " (invalid opcode %d in simple_array_op) ", op );
			return(1);
			break;
	}
	if( dest ){
		i-= 1;
		dest->last_index= i;
		dest->value= ARRVAL(dest,i);
		if( dest->accessHandler ){
			AccessHandler( dest, "simple_array_op", level, form, NULL, NULL );
		}
	}
	if( return_sum ){
		*result= sum;
	}
	return(0);
}
#undef ARRVALb

/* routine for the "add[x,y]" ascanf syntax	*/
int ascanf_add ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i, a= 1;
		if( AllowSimpleArrayOps ){
			a= simple_array_op( args, ascanf_arguments, 1, result, level, ASCB_COMPILED );
		}
		if( ascanf_arguments> 1 && a ){
			ascanf_arg_error= (ascanf_arguments< 2 );
			*result= args[0];
			for( i= 1; i< ascanf_arguments; i++ ){
				*result+= args[i];
			}
		}
		return( 1 );
	}
}

/* routine for the "sub[x,y]" ascanf syntax	*/
int ascanf_sub ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i, a= 1;
		if( AllowSimpleArrayOps ){
			a= simple_array_op( args, ascanf_arguments, 2, result, level, ASCB_COMPILED );
		}
		if( a ){
			ascanf_arg_error= (ascanf_arguments< 2 );
			*result= args[0];
			for( i= 1; i< ascanf_arguments; i++ ){
				*result-= args[i];
			}
		}
		return( 1 );
	}
}

/* routine for the "relsub[x,y,relative,absolute]" ascanf syntax	*/
int ascanf_relsub ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  Boolean relat, absol;
	  int a= 1;
		if( AllowSimpleArrayOps ){
			a= simple_array_op( args, ascanf_arguments, 5, result, level , ASCB_COMPILED );
		}
		if( a ){
			ascanf_arg_error= (ascanf_arguments< 2 );
			relat= (ascanf_arguments> 2 && args[2])? True : False;
			absol= (ascanf_arguments> 3 && args[3])? True : False;
			*result= args[0] - args[1];
			if( relat ){
				*result/= args[1];
			}
			if( absol && *result< 0 ){
				*result*= -1;
			}
		}
		return( 1 );
	}
}

/* routine for the "mul[x,y]" ascanf syntax	*/
int ascanf_mul ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i, a= 1;
		if( AllowSimpleArrayOps ){
			a= simple_array_op( args, ascanf_arguments, 3, result, level, ASCB_COMPILED );
		}
		if( ascanf_arguments> 1 && a ){
			ascanf_arg_error= (ascanf_arguments< 2 );
			*result= args[0];
			for( i= 1; i< ascanf_arguments; i++ ){
				*result*= args[i];
			}
		}
		return( 1 );
	}
}

/* routine for the "zero_div_zero[x,y]" ascanf syntax	*/
int ascanf_zero_div_zero ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args ){
		*result= zero_div_zero.d;
	}
	else{
		*result= (zero_div_zero.d= args[0]);
	}
	return(1);
}

/* routine for the "div[x,y]" ascanf syntax	*/
int ascanf_div ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i, a= 1;
		if( AllowSimpleArrayOps ){
			a= simple_array_op( args, ascanf_arguments, 4, result, level, ASCB_COMPILED );
		}
		if( a ){
			ascanf_arg_error= (ascanf_arguments< 2 );
			if( !args[1]){
				if( args[0] ){
					set_Inf( *result, args[0] );
				}
				else{
					*result= zero_div_zero.d;
				}
			}
			else{
				*result= args[0];
				for( i= 1; i< ascanf_arguments; i++ ){
					if( !args[i]){
						if( *result ){
							set_Inf( *result, *result );
						}
						else{
							*result= zero_div_zero.d;
						}
					}
					else{
						*result/= args[i];
					}
				}
			}
		}
		return( 1 );
	}
}

double dfac( x, dx)
double x, dx;
{  double X= x-dx, result= x;
	if( dx<= 0.0){
		set_Inf( result, x);
	}
	else{
		while( X> 0){
			result*= X;
			X-= dx;
		}
	}
	return(result);
}

/* routine for the "fac[x,y]" ascanf syntax	*/
int ascanf_fac ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= dfac( args[0], args[1]);
		return( 1 );
	}
}

int ascanf_MAX ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  ascanf_Function *af;
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_array, "ascanf_MAX", False, NULL)) ){
			  int j;
			  double v;
				for( j= 0; j< af->N; j++ ){
					v= (af->iarray)? af->iarray[j] : af->array[j];
					if( i== 0 && j== 0 ){
						*result= v;
					}
					else if( v> *result ){
						*result= v;
					}
				}
				ascanf_arg_error= 0;
			}
			else{
				if( i== 0 ){
					*result= args[i];
				}
				else if( args[i]> *result ){
					*result= args[i];
				}
			}
		}
		return( 1 );
	}
}

int ascanf_MIN ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  ascanf_Function *af;
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_array, "ascanf_MIN", False, NULL)) ){
			  int j;
			  double v;
				for( j= 0; j< af->N; j++ ){
					v= (af->iarray)? af->iarray[j] : af->array[j];
					if( i== 0 && j== 0 ){
						*result= v;
					}
					else if( v< *result ){
						*result= v;
					}
				}
				ascanf_arg_error= 0;
			}
			else{
				if( i== 0 ){
					*result= args[i];
				}
				else if( args[i]< *result ){
					*result= args[i];
				}
			}
		}
		return( 1 );
	}
}

int ascanf_ABSMAX ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  ascanf_Function *af;
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_array, "ascanf_ABSMAX", False, NULL)) ){
			  int j;
			  double v;
				for( j= 0; j< af->N; j++ ){
					v= (af->iarray)? af->iarray[j] : af->array[j];
					if( i== 0 && j== 0 ){
						*result= v;
					}
					else if( fabs(v)> fabs(*result) ){
						*result= v;
					}
				}
				ascanf_arg_error= 0;
			}
			else{
				if( i== 0 ){
					*result= args[i];
				}
				else if( fabs(args[i])> fabs(*result) ){
					*result= args[i];
				}
			}
		}
		return( 1 );
	}
}

int ascanf_ABSMIN ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  ascanf_Function *af;
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_array, "ascanf_ABSMIN", False, NULL)) ){
			  int j;
			  double v;
				for( j= 0; j< af->N; j++ ){
					v= (af->iarray)? af->iarray[j] : af->array[j];
					if( i== 0 && j== 0 ){
						*result= v;
					}
					else if( fabs(v)< fabs(*result) ){
						*result= v;
					}
				}
				ascanf_arg_error= 0;
			}
			else{
				if( i== 0 ){
					*result= args[i];
				}
				else if( fabs(args[i])< fabs(*result) ){
					*result= args[i];
				}
			}
		}
		return( 1 );
	}
}

double SubVisAngle( double r, double R, double thres )
{ double ret= 0;
	if( R && R>= r ){
		if( (ret= fabs(r/ R))>= thres ){
			ret= asin(ret);
		}
	}
	else{
		ret= M_2PI;
	}
	return( ret );
}

double phival2( x, y)
double x, y;
{
	if( x> 0.0)
		return( (atan(y/x)));
	else if( x< 0.0){
		if( y>= 0.0)
			return( M_PI+ (atan(y/x)));
		else
			return( (atan(y/x))- M_PI);
	}
	else{
		if( y> 0.0)
			return( M_PI_2 );
		else if( y< 0.0)
			return( -M_PI_2 );
		else
			return( 0.0);
	}
}

double SubVisAngle2( double r, double R )
{
	return( fabs(phival2(R,r)) );
}

int ascanf_angsize1 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		if( ascanf_arguments< 4 ){
			args[3]= 0.1;
		}
		*result= args[2] * (SubVisAngle( args[0], args[1], args[3] )/ M_2PI);
		return( 1 );
	}
}

/* return phi in <-base/2,base/2]	*/
double conv_angle_( double phi, double base)
{
	double x, hbase= base/2;
	struct exception exc;

	if( NaNorInf(phi) ){
		exc.name= "fmod";
		exc.type= DOMAIN;
		exc.arg1= phi;
		exc.arg2= 0.0;
		exc.retval= 0.0;
		if( !matherr( &exc) )
			errno= EDOM;
		return(exc.retval);
	}
	x= fmod( phi, base);
	if( x> -hbase)
		return( (x<= hbase)? x : x- base);
	else
		return( base+ x);
}

/* routine for the "cnv_angle1[x" ascanf syntax	*/
int ascanf_cnv_angle1 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 1 );
		if( ascanf_arguments< 2 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= conv_angle_( args[0], args[1] );
		return( 1 );
	}
}

/* return an angle between <-base,base> */
double mod_angle_( phi, base)
double phi, base;
{
	double x;
	struct exception exc;

	if( NaNorInf(phi) ){
		exc.name= "fmod";
		exc.type= DOMAIN;
		exc.arg1= phi;
		exc.arg2= 0.0;
		exc.retval= 0.0;
		if( !matherr( &exc) )
			errno= EDOM;
		return(exc.retval);
	}
	x= fmod( phi, base);
	return( (x>= 0.0)? x : base+ x);
}

/* routine for the "cnv_angle2[x" ascanf syntax	*/
int ascanf_cnv_angle2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 1 );
		if( ascanf_arguments< 2 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= mod_angle_( args[0], args[1] );
		return( 1 );
	}
}

int ascanf_angsize2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		if( ascanf_arguments< 4 ){
		  /* dummy	*/
			args[3]= 0.1;
		}
		*result= args[2] * (SubVisAngle2( args[0], args[1] )/ M_2PI);
		return( 1 );
	}
}

/* routine for the "asin[x" ascanf syntax	*/
int ascanf_asin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 1 );
		if( ascanf_arguments< 2 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= args[1]* ( asin( args[0] ) / M_2PI );
		return( 1 );
	}
}

/* routine for the "acos[x" ascanf syntax	*/
int ascanf_acos ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 1 );
		if( ascanf_arguments< 2 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= args[1]* ( acos( args[0] ) / M_2PI );
		return( 1 );
	}
}

/* routine for the "atan[x" ascanf syntax	*/
int ascanf_atan ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 1 );
		if( ascanf_arguments< 2 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= args[1]* ( atan( args[0] ) / M_2PI );
		return( 1 );
	}
}

/* routine for the "atan2[x,y]" ascanf syntax	*/
int ascanf_atan2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		*result= args[2]* ( atan2( args[1], args[0] ) / M_2PI );
		return( 1 );
	}
}

/* routine for the "atan3[x,y]" ascanf syntax	*/
int ascanf_atan3 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double atan3();
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		*result= args[2]* ( atan3( args[0], args[1] ) / M_2PI );
		return( 1 );
	}
}

/* return arg(x,y) in radians [0,2PI]	*/
double arg( double x, double y)
{
	if( x> 0.0){
		if( y>= 0.0)
			return( (atan(y/x)));
		else
			return( M_2PI+ (atan(y/x)));
	}
	else if( x< 0.0){
		return( M_PI+ (atan(y/x)));
	}
	else{
		if( y> 0.0){
		  /* 90 degrees	*/
			return( M_PI_2 );
		}
		else if( y< 0.0){
		  /* 270 degrees	*/
			return( M_PI_2 * 3 );
		}
		else{
			return( 0.0);
		}
	}
}

/* routine for the "arg[x,y]" ascanf syntax	*/
int ascanf_arg ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		if( ascanf_arguments< 4 ){
			args[3]= 0;
		}
		*result= (args[2] * arg( args[0], args[1] ) / M_2PI) - args[3];
		return( 1 );
	}
}

#if 0
int ascanf_mod_accum ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	 int i= 0;
	 double delta;
		ascanf_arg_error= (ascanf_arguments< 2 );
		CHK_ASCANF_MEMORY;
		CLIP_EXPR( i, (int) args[1], 0, ASCANF_MAX_ARGS-1);
/* 			ascanf_memory[i]= *result= mem* ascanf_memory[i]+ args[0];	*/
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_PI;
		}
		*result= (ascanf_memory[i]+= delta);
		return( 1 );
	}
}
#endif

/* Sort 3 doubles according to their absolute values	*/
double fdsort3( double x[3])
{  double *a= &x[0], *b= &x[1], *c= &x[2], t;
   double fx[3], *fa= &fx[0], *fb= &fx[1], *fc= &fx[2];
	*fa= fabs( *a );
	*fb= fabs( *b );
	*fc= fabs( *c );
	if( *fa > *fb){
		t= *fb; *fb= *fa; *fa= t;
		t= *b; *b= *a; *a= t;
	}
	if( *fa > *fc){
		t= *fc; *fc= *fa; *fa= t;
		t= *c; *c= *a; *a= t;
	}
	if( *fb > *fc){
		t= *fc; *fc= *fa; *fa= t;
		t= *c; *c= *b; *b= t;
	}
	return( *a);
}

int ascanf_angdiff ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  double x[3];
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		x[0]= args[0]- args[1];
		x[1]= x[0]+ args[2];
		x[2]= x[0]- args[2];
		*result= fdsort3( x);
		return( 1 );
	}
}

/* routine for the "len[x,y]" ascanf syntax	*/
int ascanf_len ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		switch( ascanf_arguments ){
			case 2:
				*result= sqrt( args[0]*args[0] + args[1]*args[1] );
				break;
			case 3:{ int a= 1;
				if( AllowSimpleArrayOps ){
					a= simple_array_op( args, ascanf_arguments, 6, result, level, ASCB_COMPILED );
				}
				if( a ){
					*result= sqrt( args[0]*args[0] + args[1]*args[1] + args[2]*args[2] );
				}
				break;
			}
			default:{
			  int i;
			  double sumsq= 0.0;
				for( i= 0; i< ascanf_arguments; i++ ){
					sumsq+= args[i]* args[i];
				}
				*result= sqrt( sumsq );
				break;
			}
		}
		return( 1 );
	}
}

/* routine for the "log[x,y]" ascanf syntax	*/
int ascanf_log ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments== 1 || args[1]== 1 ){
			*result= log( args[0]);
		}
		else{
			if( args[1]== 10.0 ){
				*result= log10( args[0] );
			}
			else{
				*result= log(args[0]) / log( args[1] );
			}
		}
		return( 1 );
	}
}

/* routine for the "pow[x,y]" ascanf syntax	*/
int ascanf_pow ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int a= 1;
		if( AllowSimpleArrayOps ){
			a= simple_array_op( args, ascanf_arguments, 7, result, level, ASCB_COMPILED );
		}
		if( a ){
			ascanf_arg_error= (ascanf_arguments< 2 );
			MATHERR_MARK();
			if( args[1]== 0.5 ){
				*result= sqrt( args[0] );
			}
			else if( args[1]== 1.0/3.0 ){
				*result= cbrt( args[0] );
			}
			else{
				*result= pow( args[0], args[1] );
			}
		}
		return( 1 );
	}
}

/* routine for the "pow*[x,y]" ascanf syntax	*/
int ascanf_pow_ ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
/* 	  double sign, val;	*/
	  int a= 1;
		ascanf_arg_error= (ascanf_arguments< 2 );
/* 		if( args[0]< 0 ){	*/
/* 			sign= -1;	*/
/* 			val= args[0]* -1;	*/
/* 		}	*/
/* 		else{	*/
/* 			sign= 1;	*/
/* 			val= args[0];	*/
/* 		}	*/
/* 		MATHERR_MARK();	*/
/* 		*result= sign* pow( val, args[1] );	*/
		if( AllowSimpleArrayOps ){
			a= simple_array_op( args, ascanf_arguments, 8, result, level, ASCB_COMPILED );
		}
		if( a ){
			*result= pow_(args[0], args[1]);
		}
		return( 1 );
	}
}

int ascanf_powXFlag ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double powXFlag;
	if( ascanf_arguments> 1 ){
		powXFlag= args[0];
	}
	*result= powXFlag;
	return( 1 );
}

int ascanf_powYFlag ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double powYFlag;
	if( ascanf_arguments> 1 ){
		powYFlag= args[0];
	}
	*result= powYFlag;
	return( 1 );
}

int ascanf_barBase ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double barBase;
  extern int barBase_set;
	if( ascanf_arguments>= 2 ){
		if( NaNorInf(args[1]) ){
			barBase_set= False;
		}
		else{
			barBase= args[1];
			barBase_set= 1;
		}
	}
	if( barBase_set ){
		*result= barBase;
	}
	else{
		set_NaN(*result);
	}
	return( 1 );
}

/* routine for the "pi[mul_fact]" ascanf syntax	*/
int ascanf_pi ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		*result= M_PI;
		return( 1 );
	}
	else{
		if( ascanf_arguments== 1 ){
			*result= M_PI * args[0];
		}
		else{
			ascanf_mul( ASCB_ARGUMENTS );
			*result *= M_PI;
		}
		return( 1 );
	}
}

int ascanf_SplitHere= False;
/* routine for the "split[]" ascanf syntax	*/
int ascanf_split ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int split_set;
  int r= 1;
	if( ActiveWin ){
		if( args ){
			ascanf_SplitHere= (args[0])? 1 : 0;
			*result= ascanf_progn_return;
		}
		else{
			ascanf_SplitHere= 0;
			ascanf_arg_error= 1;
			r= (0);
		}
	}
	else{
	  /* assume we're reading, so we can directly split sets	*/
		*result= (*ascanf_self_value);
		if( !args){
			split_set= 1;
		}
		else{
			split_set= (args[0])? -1 : 1;
		}
	}
	return(r);
}

int ascanf_elapsed ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  static Time_Struct timer;
   static char called= 0;

#ifdef linux
	if( pragma_unlikely(ascanf_verbose) ){
		elapsed_verbose= 1;
	}
#endif
	Elapsed_Since( &timer, True );
	if( !called){
		called= 1;
		*result= 0.0;
		return(1);
	}
	ascanf_elapsed_values[0]= Delta_Tot_T;
	ascanf_elapsed_values[1]= timer.Time;
	ascanf_elapsed_values[2]= timer.sTime;
	if( !ascanf_arguments ){
		*result= Delta_Tot_T;
	}
	else{
		*result= timer.Time;
	}
	return( 1 );
}

int ascanf_time ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  static Time_Struct timer_0;
/*    Time_Struct timer;	*/
   static char called= 0;
	if( !called){
	  /* initialise the real timer	*/
		called= 1;
		Set_Timer();
		Elapsed_Since( &timer_0, True );
		*result= 0.0;
		return(1);
	}
	else{
	  /* make a copy of the timer, and determine the time
	   \ since initialisation
	   */
/* 		timer= timer_0;	*/
		Elapsed_Since( &timer_0, False );
	}
	ascanf_elapsed_values[0]= timer_0.HRTot_T;
	ascanf_elapsed_values[1]= timer_0.Time;
	ascanf_elapsed_values[2]= timer_0.sTime;
	if( !ascanf_arguments ){
/* 		*result= timer.Tot_Time;	*/
		*result= timer_0.HRTot_T;
	}
	else{
/* 		*result= timer.Time;	*/
		*result= timer_0.Time;
	}
	return( 1 );
}

int ascanf_gettimeofday ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  struct timeval tv;
  struct timezone tz;
	gettimeofday(&tv,&tz);
	*result= tv.tv_sec + tv.tv_usec/1000000.0;
	return(1);
}

#ifdef FFTW_CYCLES_PER_SEC

int ascanf_HRTimer ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  fftw_time t;
	fftw_get_time(&t);
	*result= fftw_time_to_sec(&t);
	return(1);
}
#endif

int ascanf_YearDay ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  struct tm atm;
  time_t t;
	time(&t);
	atm= *localtime(&t);
	if( ascanf_arguments> 0 && !NaN(args[0]) ){
		atm.tm_mday= (int) args[0];
	}
	if( ascanf_arguments> 1 && !NaN(args[1]) ){
		atm.tm_mon= (int) (args[1]>=0)? args[1]- 1 : args[1]+ 1;
	}
	if( ascanf_arguments> 2 && !NaN(args[2]) ){
		atm.tm_year= (int) (args[2]>=0)? args[2]- 1900 : atm.tm_year+ (int) args[2];
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, " %02d/%02d/%04d = ", atm.tm_mday, atm.tm_mon+1, atm.tm_year+ 1900 );
	}
	if( mktime(&atm)== -1 ){
		set_NaN(*result);
	}
	else{
		*result= atm.tm_yday+ 1;
	}
	if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, "%02d/%02d/%04d, day== ", atm.tm_mday, atm.tm_mon+1, atm.tm_year+ 1900 );
		fflush( StdErr );
	}
	return(1);
}

static int ascanf_continue= 0;
static void ascanf_continue_( int sig )
{
	ascanf_continue= 1;
	signal( SIGALRM, ascanf_continue_ );
}

int ascanf_sleep_once ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  Time_Struct timer;
	if( ascanf_arguments>= 1 && args[0]> 0 ){
/* 	  double sl= args[0]*1e6;	*/
/* 		if( sl<= ULONG_MAX ){	*/
/* 			Elapsed_Since(&timer, True);	*/
/* 			usleep( (unsigned long) sl );	*/
/* 			Elapsed_Since(&timer, False);	*/
/* 		}	*/
/* 		else	*/
		{
		  struct itimerval rtt, ortt;
			rtt.it_value.tv_sec= (unsigned long) ssfloor(args[0]);
			rtt.it_value.tv_usec= (unsigned long) ( (args[0]- rtt.it_value.tv_sec)* 1000000 );
			rtt.it_interval.tv_sec= 0;
			rtt.it_interval.tv_usec= 0;
			signal( SIGALRM, ascanf_continue_ );
			ascanf_continue= 0;
			Elapsed_Since( &timer, True );
			setitimer( ITIMER_REAL, &rtt, &ortt );
			pause();
			Elapsed_Since( &timer, True );
			  /* restore the previous setting of the timer.	*/
			setitimer( ITIMER_REAL, &ortt, &rtt );
		}
	}
	else{
		ascanf_arg_error= 1;
		return( 0 );
	}
	ascanf_continue= 0;
	*result= timer.HRTot_T;
	return( 1 );
}

int ascanf_set_interval ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  static double initial= -1, interval= -1;
  static Time_Struct timer;
	if( ascanf_arguments>= 1 && args[0]> 0 ){
	  struct itimerval rtt, ortt;
		if( ascanf_arguments< 2 || args[1]<= 0 ){
			args[1]= args[0];
		}
		if( args[0]!= initial || args[1]!= interval ){
			rtt.it_value.tv_sec= (unsigned long) ssfloor(args[0]);
			rtt.it_value.tv_usec= (unsigned long) ( (args[0]- rtt.it_value.tv_sec)* 1000000 );
			rtt.it_interval.tv_sec= (unsigned long) ssfloor(args[1]);
			rtt.it_interval.tv_usec= (unsigned long) ( (args[1]- rtt.it_interval.tv_sec)* 1000000 );
			signal( SIGALRM, ascanf_continue_ );
			ascanf_continue= 0;
			initial= args[0];
			interval= args[1];
			Elapsed_Since( &timer, True );
			setitimer( ITIMER_REAL, &rtt, &ortt );
		}
		if( !ascanf_continue ){
		  /* wait for the signal	*/
			pause();
			  /* Make sure that we will pause if no interval has expired since the last call!	*/
			ascanf_continue= 0;
			Elapsed_Since( &timer, True );
		}
	}
	else{
		ascanf_arg_error= 1;
		return( 0 );
	}
	*result= timer.HRTot_T;
	return( 1 );
}

#ifndef degrees
#	define degrees(a)			((a)*57.295779512)
#endif
#ifndef radians
#	define radians(a)			((a)/57.295779512)
#endif

int ascanf_degrees ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		*result= degrees(M_2PI);
		return( 1 );
	}
	else{
		*result= degrees( args[0]);
		return( 1 );
	}
}

int ascanf_radians ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		*result= radians(360.0);
		return( 1 );
	}
	else{
		*result= radians( args[0]);
		return( 1 );
	}
}

int ascanf_sin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments== 1 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= sin( M_2PI * args[0]/ args[1] );
		return( 1 );
	}
}

int ascanf_cos ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments== 1 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= cos( M_2PI * args[0]/ args[1] );
		return( 1 );
	}
}

int ascanf_tan ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments== 1 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		*result= tan( M_2PI * args[0]/ args[1] );
		return( 1 );
	}
}

// #undef NATIVE_SINCOS

int ascanf_sincos ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args || ascanf_arguments< 4 ){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  ascanf_Function *v= NULL, *s, *c, *t= NULL;
		if( ascanf_arguments== 1 || args[1]== 0.0 ){
			args[1]= M_2PI;
		}
		s= parse_ascanf_address( args[2], 0, "ascanf_sincos", (int) ascanf_verbose, NULL );
		c= parse_ascanf_address( args[3], 0, "ascanf_sincos", (int) ascanf_verbose, NULL );
		if( ascanf_arguments> 4 && args[4] ){
			t= parse_ascanf_address( args[4], 0, "ascanf_sincos", (int) ascanf_verbose, NULL );
		}
		if( s && c ){
			if( s->type== _ascanf_array && s->array && c->type== _ascanf_array && c->array
				&& (v= parse_ascanf_address( args[0], _ascanf_array, "ascanf_sincos", (int) ascanf_verbose, NULL ))
			){
			  int i;
			  double arg;
				Resize_ascanf_Array( s, v->N, result );
				Resize_ascanf_Array( c, v->N, result );
				if( t && (t->type!= _ascanf_array || !t->array) ){
					t= NULL;
				}
#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__)) && NATIVE_SINCOS
				{ int nn, N_1 = v->N-1;
				  __m128d va, vbase;
				  double *v2 = (double*) &va;
				  int scale;
					if( (scale = (args[1] != M_2PI)) ){
						vbase = _MM_SET1_PD((M_2PI/args[1]));
					}
					for( i = 0 ; i < v->N ; i += 2 ){
						if( pragma_unlikely(i == N_1) ){
							nn = 1;
							if( scale ){
								va = _mm_mul_pd( _MM_SET1_PD( v->array[i] ), vbase );
							}
							else{
								va = _MM_SET1_PD( v->array[i] );
							}
						}
						else{
							nn = 2;
							if( scale ){
								va = _mm_mul_pd( (__m128d){ v->array[i], v->array[i+1] }, vbase );
							}
							else{
								va = (__m128d){ v->array[i], v->array[i+1] };
							}
						}
						vvsincos( &s->array[i], &c->array[i], v2, &nn );
						if( t ){
							vvtan( &t->array[i], v2, &nn );
						}
					}
				}
#else
				for( i= 0; i< v->N; i++ ){
#if NATIVE_SINCOS
					arg = M_2PI* v->array[i]/args[1];
					sincos( arg, &s->array[i], &c->array[i] );
#else
					s->array[i]= sin( (arg= (M_2PI * v->array[i]/ args[1])) );
					c->array[i]= cos( arg );
#endif
					if( t ){
						t->array[i]= tan( arg );
					}
				}
#endif
				if( s->accessHandler ){
					s->value= s->array[0];
					AccessHandler( s, s->name, level, ASCB_COMPILED, AH_EXPR, NULL );
				}
				if( c->accessHandler ){
					c->value= c->array[0];
					AccessHandler( c, c->name, level, ASCB_COMPILED, AH_EXPR, NULL );
				}
				if( t && t->accessHandler ){
					t->value= t->array[0];
					AccessHandler( t, t->name, level, ASCB_COMPILED, AH_EXPR, NULL );
				}
				*result= s->array[0];
			}
			  /* 20050902: let's see what can be done when we just accept any type?! */
			else /* if( s->type== _ascanf_value && c->type== _ascanf_value ) */ {
			  double arg;
#if NATIVE_SINCOS
				arg = M_2PI * args[0] / args[1];
				sincos( arg, &(s->value), &(c->value) );
				*result= s->value;
#else
				*result= s->value= sin( (arg= (M_2PI * args[0]/ args[1])) );
				c->value= cos( arg );
#endif
				if( s->accessHandler ){
					AccessHandler( s, s->name, level, ASCB_COMPILED, AH_EXPR, NULL );
				}
				if( c->accessHandler ){
					AccessHandler( c, c->name, level, ASCB_COMPILED, AH_EXPR, NULL );
				}
				if( t && t->type== _ascanf_value ){
					t->value= tan( arg );
					if( t->accessHandler ){
						AccessHandler( t, t->name, level, ASCB_COMPILED, AH_EXPR, NULL );
					}
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(sin=%s cos=%s)== ",
						ad2str( s->value, d3str_format, NULL ), ad2str( c->value, d3str_format, NULL )
					);
				}
			}
/* 			else{	*/
/* 				ascanf_emsg= " (invalid arguments: scalar/array mix) ";	*/
/* 				ascanf_arg_error= True;	*/
/* 			}	*/
		}
		return( !ascanf_arg_error );
	}
}

int ascanf_exp ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		*result= exp(1.0);
		return( 1 );
	}
	else{
		MATHERR_MARK();
		*result= exp( args[0]);
		return( 1 );
	}
}

int ascanf_dcmp ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments== 2 ){
			args[2]= -1.0;
		}
		*result= dcmp( args[0], args[1], args[2] );
		return( 1 );
	}
}

/* uniform_rand(av,stdv): return a random value taken
 * from a uniform distribution. This is calculated as
 \ av + rand_in[-1.75,1.75]*stdv
 \ which gives approx. the right results.
 */
double uniform_rand( double av, double stdv )
{
	if( stdv ){
		return( av + stdv* (drand()* 3.5- 1.75) );
	}
	else{
		return( av );
	}
}

/* normal_rand(av,stdv): return a random value taken from
 \ a normal distribution
 */
double normal_rand( int i, double av, double stdv )
{ double fac,r,v1,v2;

	if( !ascanf_normdists && !ascanf_SyntaxCheck ){
		if( !ASCANF_MAX_ARGS ){
			ASCANF_MAX_ARGS= AMAXARGSDEFAULT;
		}
		if( !(ascanf_normdists= (NormDist*) calloc( ASCANF_MAX_ARGS, sizeof(NormDist))) ){
			fprintf( StdErr, "Can't get memory for ascanf_normdists[%d] array (%s)\n", ASCANF_MAX_ARGS, serror() );
			ascanf_arg_error= 1;
			return(0);
		}
	}
	if( i>= 0 && i< ASCANF_MAX_ARGS && !ascanf_SyntaxCheck ){
		if( av!= ascanf_normdists[i].av || stdv!= ascanf_normdists[i].stdv ){
			ascanf_normdists[i].iset= 0;
			ascanf_normdists[i].av= av;
			ascanf_normdists[i].stdv= stdv;
		}
		if( ascanf_normdists[i].iset == 0) {
			do {
				v1= 2.0* drand()- 1.0;
				v2= 2.0* drand()- 1.0;
				r=v1*v1+v2*v2;
			} while (r >= 1.0);
			fac= ascanf_normdists[i].stdv * sqrt(-2.0*log(r)/r);
			ascanf_normdists[i].gset= v1*fac;
			ascanf_normdists[i].iset= 1;
			return( ascanf_normdists[i].av + v2*fac );
		} else {
			ascanf_normdists[i].iset=0;
			return ascanf_normdists[i].av + ascanf_normdists[i].gset;
		}
	}
	else{
		return( av );
	}
}

/* abnormal_rand(av,stdv): return a random value taken from
 \ an abnormal distribution
 */
double abnormal_rand( double av, double stdv )
{  double r= drand()- 0.5;
	return( av + stdv * ((r< 0)? -17.87 : 17.87) *
				( exp( 0.5* r* r) - 1.0 )
	);
}

extern double Entier();

int ascanf_misc_fun( double *args, double *result, int code, int argc )
{
	if( !args || ascanf_arguments< argc ){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  double arg1= (ascanf_arguments>1)? args[1] : 0, arg2;
	  long ia0, ia1;
	  int a0= (args[0]!= 0.0 && !NaN(args[0])), a1= (arg1!= 0.0 && !NaN(arg1));
	  Boolean r0= False, r1= False;
		if( ascanf_arguments<= 2 ){
			arg2= 0;
		}
		else{
			arg2= args[2];
		}
		{ double d= (NaNorInf(args[0]))? MAXLONG*2.0 : ssfloor( fabs(args[0]) );
			if( d<= MAXLONG ){
				ia0= (long) args[0];
			}
			else{
				r0= True;
			}
			d= (NaNorInf(arg1))? MAXLONG*2.0 : ssfloor( fabs(arg1) );
			if( d<= MAXLONG ){
				ia1= (long) arg1;
			}
			else{
				r1= True;
			}
		}
		switch( code ){
			case 0:
				*result= (args[0] && !NaN(args[0]) )? arg1 : arg2;
				break;
			case -2:
				*result= (double) ( args[0] != arg1 );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(diff=%s)== ", ad2str( args[0]-args[1], d3str_format, 0) );
				}
				break;
			case -1:
				*result= (double) ( args[0] == arg1 );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(diff=%s)== ", ad2str( args[0]-args[1], d3str_format, 0) );
				}
				break;
			case 1:
				*result= (double) ( args[0] > arg1 );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(diff=%s)== ", ad2str( args[0]-args[1], d3str_format, 0) );
				}
				break;
			case 2:
				*result= (double) ( args[0] < arg1 );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(diff=%s)== ", ad2str( args[0]-args[1], d3str_format, 0) );
				}
				break;
			case 3:
				*result= (double) ( args[0] >= arg1 );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(diff=%s)== ", ad2str( args[0]-args[1], d3str_format, 0) );
				}
				break;
			case 4:
				*result= (double) ( args[0] <= arg1 );
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(diff=%s)== ", ad2str( args[0]-args[1], d3str_format, 0) );
				}
				break;
			case 5:
				*result= (double) ( a0 && a1 );
				break;
			case 6:
				*result= (double) ( a0 || a1 );
				break;
			case 7:
				  /* Boolean XOR doesn't exist (not even as ^^)
				   \ However, if a0 and a1 are Booleans, and
				   \ either 0 or 1, than the bitwise operation
				   \ does exactly the same thing
				   */
				*result= (double) ( a0 ^ a1 );
				break;
			case 8:
				*result= (double) ! a0;
				break;
			case 9:
				*result= fabs( args[0] );
				break;
			case 10:
				*result= (ascanf_progn_return= args[0]);
				break;
			case 11:
				*result= MIN( args[0], arg1 );
				break;
			case 12:
				*result= MAX( args[0], arg1 );
				break;
			case 13:
				*result= ssfloor( args[0] );
				break;
			case 14:
				*result= ssceil( args[0] );
				break;
			case 15:
				*result= uniform_rand( args[0], arg1 );
				break;
			case 16:
				*result= abnormal_rand( args[0], arg1 );
				break;
			case 17:
				*result= erf( args[0] );
				break;
			case 18:
				*result= erfc( args[0] );
				break;
			case 19:{
			  int i= (int) args[0];
				if( i>= 0 && i< ASCANF_MAX_ARGS ){
					*result= normal_rand( i, arg1, arg2 );
					break;
				}
				else{
					ascanf_arg_error= 1;
					ascanf_emsg= "(range error)";
					return(0);
				}
			}
			case 20:
				if( args[0]< arg1 ){
					*result=  arg1;
				}
				else if( args[0]> arg2 ){
					*result= arg2;
				}
				else{
					*result= args[0];
				}
				break;
			case 21:
				*result= Entier(args[0]);
				break;
			case 22:
				if( r0 || r1 ){
/* 					fprintf( StdErr, " (range error) " );	*/
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				*result= (double) ( ia0 & ia1 );
				break;
			case 23:
				if( r0 || r1 ){
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				*result= (double) ( ia0 | ia1 );
				break;
			case 24:
				if( r0 || r1 ){
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				*result= (double) ~ (ia0);
				break;
			case 25:
				if( r0 || r1 ){
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				*result= (double) ( ia0 << ia1 );
				break;
			case 26:
				if( r0 || r1 ){
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				*result= (double) ( ia0 >> ia1 );
				break;
			case 27:
				if( r0 || r1 ){
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				*result= (double) ( ia0 ^ ia1 );
				break;
			case 28:{
			  Boolean ok;
				ok= ((args[3])? (args[0]>= arg1) : (args[0]> arg1));
				if( ok ){
					ok= ((args[4])? (args[0]<= arg2) : (args[0]< arg2));
				}
				*result= (double) ok;
				break;
			}
			case 29:{
			  long x;
				if( r0 ){
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				x= ia0/ 2;
				*result= (double) ( x* 2== ia0 );
				break;
			}
			case 30:
				*result= SIGN( args[0] );
				break;
			{ int i;
				case 31:
					for( i= 0, *result= 0; i< ascanf_arguments && !*result; i++ ){
						*result= (NaN( args[i] ))? 1 : 0;
					}
					break;
				case 32:
					for( i= 0, *result= 0; i< ascanf_arguments && !*result; i++ ){
						*result= Inf( args[i] );
					}
					break;
				case 33:
					for( i= 0, *result= 0; i< ascanf_arguments && !*result; i++ ){
						*result= (NaNorInf( args[i] ))? 1 : 0;
					}
					break;
			}
			case 34:{
			  long long la0;
			  int iia1;
				if( (args[0] < LLONG_MIN || args[0] > LLONG_MAX) || (args[1] < 0 || args[1] >= 64) ){
					ascanf_emsg= "(range error)";
					ascanf_arg_error= 1;
					return(0);
				}
				la0 = (long long) args[0];
				iia1 = (int) args[1];
				*result= (double) ( (la0 >> iia1) & 0x01 );
				break;
			}
		}
		return( 1 );
	}
}

int ascanf_switch0 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int ref= 1;
  double val;
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		*result= 0;
		return(0);
	}
	val= args[0];
	if( *result== -1 ){
		*result= args[ascanf_arguments-1];
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (default case {%d}(%g))== ", (int) *ascanf_switch_case, *result );
		}
	}
	else{
		while( ref< ascanf_arguments && val!= args[ref] ){
			ref+= 2;
		}
		if( ref< ascanf_arguments && val== args[ref] ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (case #%d{%d}(%g))== ", (ref-1)/2, (int) *ascanf_switch_case, *result );
			}
			*result= args[ref+1];
		}
		else{
			*result= 0;
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (case not found, no default)== " );
			}
		}
	}
	return(1);
}

#define SWITCHCCOMPARE(a) ((atest.type==_ascanf_array || aref.type!=_ascanf_array)? \
								((a)[0]==(a)[ref]) : \
								contained((a)[0],&aref) )

/* 20050111 */
int ascanf_switch ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int ref= 1;
  ascanf_Function atest, aref;
  double val;
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		*result= 0;
		return(0);
	}
	val= args[0];
	atest.value= args[0];
	atest.type= _ascanf_value;
	aref.value= args[1];
	aref.type= _ascanf_value;
	{ ascanf_Function *af= parse_ascanf_address(args[0], _ascanf_array, "ascanf_switch", 0, NULL );
		if( !af ){
			if( (af= parse_ascanf_address(args[1], _ascanf_array, "ascanf_switch", 0, NULL )) ){
				aref= *af;
			}
		}
		else{
		  /* test variable is an array pointer: store that fact in atest.type (no
		   \ need to modify atest.value)
		  */
			atest.type= _ascanf_array;
		}
	}
	if( *result== -1 ){
		*result= args[ascanf_arguments-1];
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, " (default case {%d}(%g))== ", (int) *ascanf_switch_case, *result );
		}
	}
	else{
	  int hit;
		while( ref< ascanf_arguments && !(hit= SWITCHCCOMPARE(args)) ){
			ref+= 2;
			aref.value= args[ref];
			aref.type= _ascanf_value;
			if( atest.type!=_ascanf_array ){
			  ascanf_Function *af;
				if( (af= parse_ascanf_address(args[ref], _ascanf_array, "ascanf_switch", 0, NULL )) ){
					aref= *af;
				}
			}
		}
		if( ref< ascanf_arguments && hit ){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (case #%d{%d}(%g))== ", (ref-1)/2, (int) *ascanf_switch_case, *result );
			}
			*result= args[ref+1];
		}
		else{
			*result= 0;
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (case not found, no default)== " );
			}
		}
	}
	return(1);
}

int ascanf_if ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 0, 1) );
}

int ascanf_if2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 0, 1) );
}

int ascanf_land ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 22, 2) );
}

int ascanf_lor ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 23, 2) );
}

int ascanf_lnot ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 24, 1) );
}

int ascanf_lxor ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 27, 2) );
}

int ascanf_shl ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 25, 2) );
}

int ascanf_shr ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 26, 2) );
}

#include <float.h>

Boolean deq( double b, double a, double prec)
{ double a_low, a_high;
  Boolean R;

	if( prec>= 0.0 ){
		a_low= (1.0-prec)* a;
		a_high= (1.0+prec)* a;
		if( b< a_low ){
			R= ( False );
		}
		else if( b> a_high ){
			R= ( False );
		}
		else{
			R= ( True );
		}
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(diff=%s,prec=%s)== ", ad2str( a-b, d3str_format, 0), ad2str(prec,d3str_format,0) );
		}
	}
	else{
	  /* prec==-1 is meant to check if b equals
	   \ a allowing for the machine imprecision
	   \ DBL_EPSILON. Let's hope that a_low and
	   \ a_high are calculated correctly!
	   */
#if 0
		prec*= -1.0;
		b-= a;
		if( b< - prec * DBL_EPSILON || b> prec * DBL_EPSILON ){
			R= ( False );
		}
		else{
			R= ( True );
		}
#else
		prec= DBL_EPSILON;
		b-= a;
		if( b< - DBL_EPSILON || b> DBL_EPSILON ){
			R= ( False );
		}
		else{
			R= ( True );
		}
#endif
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(diff=%s,prec=%s)== ", ad2str( b, d3str_format, 0), ad2str(prec,d3str_format,0) );
		}
	}
	return( R );
}

int ascanf_cp_array ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( simple_array_op( args, ascanf_arguments, 0, result, level , ASCB_COMPILED ) ){
		ascanf_arg_error= 1;
	}
	return(1);
}

int ascanf_eq ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !AllowSimpleArrayOps || simple_array_op( args, ascanf_arguments, -1, result, level , ASCB_COMPILED ) ){
		if( ascanf_arguments== 3 ){
			*result= (double) deq( args[0], args[1], args[2] );
			return( 1 );
		}
		else{
			return( ascanf_misc_fun( args, result, -1, 2) );
		}
	}
	return(1);
}

int ascanf_neq ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, -2, 2) );
}

int ascanf_gt ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 1, 2) );
}

int ascanf_lt ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 2, 2) );
}

int ascanf_ge ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 3 ){
		if( !(*result= (double) deq( args[0], args[1], args[2] )) ){
			return( ascanf_misc_fun( args, result, 1, 2 ) );
		}
		return(1);
	}
	else{
		return( ascanf_misc_fun( args, result, 3, 2) );
	}
}

int ascanf_le ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 3 ){
		if( !(*result= (double) deq( args[0], args[1], args[2] )) ){
			return( ascanf_misc_fun( args, result, 2, 2 ) );
		}
		return(1);
	}
	else{
		return( ascanf_misc_fun( args, result, 4, 2) );
	}
}

int ascanf_ELEM ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 28, 5) );
}

int ascanf_and ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i= 1;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		*result= 0;
		return( 0 );
	}
	*result= (NaN(args[0]))? 0 : args[0];
	while( i< ascanf_arguments && *result ){
		*result= *result && ((NaN(args[i]))? 0 : args[i]);
		i++;
	}
	return( 1 );
/* 	return( ascanf_misc_fun( args, result, 5, 2) );	*/
}

int ascanf_or ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i= 1;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		*result= 0;
		return( 0 );
	}
	*result= (NaN(args[0]))? 0 : args[0];
	while( i< ascanf_arguments && !*result ){
		*result= *result || ( (NaN(args[i]))? 0 : args[i] );
		i++;
	}
	return( 1 );
/* 	return( ascanf_misc_fun( args, result, 6, 2) );	*/
}

int ascanf_within ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i= 1, hit= 0, first_ptr;
  ascanf_Function *af;
  double val;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		*result= 0;
		return( 0 );
	}
	val= args[0];
	if( parse_ascanf_address(args[0], 0, "ascanf_within", False, NULL) ){
		first_ptr= True;
	}
	else{
		first_ptr= False;
	}
	while( i< ascanf_arguments && !hit ){
		if( !first_ptr && (af= parse_ascanf_address(args[i], _ascanf_array, "ascanf_within", False, NULL)) ){
		  int j= 0;
			while( j< af->N && !hit ){
				if( af->iarray ){
					hit= (val== af->iarray[j]);
				}
				else{
					hit= (val== af->array[j]);
				}
				j++;
			}
		}
		else{
			hit= (val== args[i]);
		}
		i++;
	}
	*result= hit;
	return( 1 );
}

int within( double a, double b, double margin )
{
	if( margin< 0 ){
		margin= 2*DBL_EPSILON;
	}
	if( a>= b-margin && a<= b+margin ){
		return(True);
	}
	else{
		return(False);
	}
}

int ascanf_contained ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int hit= 0;
  ascanf_Function *first_ptr;
  ascanf_Function *af;
  double val;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		*result= 0;
		return( 0 );
	}
	val= args[0];
	first_ptr= parse_ascanf_address(args[0], 0, "ascanf_contained", False, NULL);
	if( ascanf_arguments<= 3 && (af= parse_ascanf_address(args[1], _ascanf_array, "ascanf_contained", False, NULL)) ){
	  Boolean fuzzy;
	  double precision;
		if( ascanf_arguments== 3 ){
			fuzzy= True;
			precision= args[2];
		}
		else{
			fuzzy= False;
		}
		if( first_ptr ){
		  int i= 0;
			val= (first_ptr->iarray)? first_ptr->iarray[i] : first_ptr->array[i];
			while( i< first_ptr->N && !hit ){
			  int j= 0;
				while( j< af->N && !hit ){
					if( af->iarray ){
						hit= (fuzzy)? within(val,af->iarray[j],precision) : (val== af->iarray[j]);
					}
					else{
						hit= (fuzzy)? within(val,af->array[j],precision) : (val== af->array[j]);
					}
					j++;
				}
				i++;
			}
		}
		else{
		  int j= 0;
			while( j< af->N && !hit ){
				if( af->iarray ){
					hit= (fuzzy)? within(val,af->iarray[j],precision) : (val== af->iarray[j]);
				}
				else{
					hit= (fuzzy)? within(val,af->array[j],precision) : (val== af->array[j]);
				}
				j++;
			}
		}
	}
	else if( ascanf_arguments>= 3 ){
		if( first_ptr ){
		  int i= 0, lincl, hincl;
		  double cval;
			if( ascanf_arguments> 3 ){
				lincl= (ASCANF_TRUE(args[3]))? True : False;
			}
			else{
				lincl= True;
			}
			if( ascanf_arguments> 4 ){
				hincl= (ASCANF_TRUE(args[4]))? True : False;
			}
			else{
				hincl= True;
			}
			while( i< first_ptr->N && !hit ){
				cval= (first_ptr->iarray)? first_ptr->iarray[i] : first_ptr->array[i];
				hit= (lincl)? (val>= cval) : (val> cval);
				if( hit ){
					hit= (hincl)? (val<= cval) : (val< cval);
				}
			}
		}
		else{
			if( ascanf_arguments> 3 ){
				if( ASCANF_TRUE(args[3]) ){
					hit= (val>= args[1]);
				}
				else{
					hit= (val> args[1]);
				}
			}
			else{
				hit= (val>= args[1]);
			}
			if( hit ){
				if( ascanf_arguments> 4 ){
					if( ASCANF_TRUE(args[4]) ){
						hit= (val<= args[2]);
					}
					else{
						hit= (val< args[2]);
					}
				}
				else{
					hit= (val<= args[2]);
				}
			}
		}
	}
	else{
		ascanf_emsg= " (currently unsupported number/mix of arguments)== ";
		hit= 0;
		ascanf_arg_error= True;
	}
	*result= hit;
	return( !ascanf_arg_error );
}

int ascanf_xor ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 7, 2) );
}

int ascanf_not ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 8, 1) );
}

int ascanf_bitset ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 34, 2) );
}

int ascanf_abs ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 9, 1) );
}

int ascanf_sign ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 30, 1) );
}

int ascanf_isNaN ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 31, 1) );
}

int ascanf_isInf ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 32, 1) );
}

int ascanf_isNaNorInf ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 33, 1) );
}

int ascanf_return ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 10, 1) );
}

int ascanf_floor ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 13, 1) );
}

int ascanf_ceil ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 14, 1) );
}

int ascanf_uniform ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 15, 2) );
}

int ascanf_abnormal ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 16, 2) );
}

int ascanf_erf ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 17, 1) );
}

int ascanf_erfc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 18, 1) );
}

int ascanf_normal ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 19, 2) );
}

int ascanf_clip ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 20, 3) );
}

int ascanf_Ent ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 21, 1) );
}

int ascanf_Even ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( ascanf_misc_fun( args, result, 29, 1) );
}

int ascanf_RoundOff ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else{
	  double round= (ascanf_arguments> 2)? args[2] : 0.5;
		if( args[0]>= 0 ){
			if( ascanf_arguments>= 2 && args[1] ){
			  double prec= pow( 10.0, args[1]);
/* 				verbose[div[floor[add[mul[1.23456789,1000000],0.5]],1000000]]	*/
				*result= ssfloor( args[0]* prec+ round)/ prec;
			}
			else{
				*result= ssfloor( args[0]+ round);
			}
		}
		else{
			if( ascanf_arguments>= 2 && args[1] ){
			  double prec= pow( 10.0, args[1]);
				*result= ssceil( args[0]* prec- round)/ prec;
			}
			else{
				*result= ssceil( args[0]- round);
			}
		}
		return( 1 );
	}
}

int ascanf_gRoundOff ( ASCB_ARGLIST )
{ ASCB_FRAME

	if( !args){
		ascanf_arg_error= 1;
		return( 1 );
	}
	else{
	  char *buf;
	  int n= 1;
		buf= ad2str( args[0], d3str_format, NULL );
		__fascanf( &n, buf, result, NULL, NULL, NULL, NULL, level, NULL );
	}
	return(1);
}

int ascanf_nop ( ASCB_ARGLIST )
{
	return( !ascanf_arg_error );
}

int ascanf_progn ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		*result= ascanf_progn_return;
		return( 1 );
	}
}

int ascanf_break_loop ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_in_loop ){
		if( !ascanf_SyntaxCheck ){
			if( args ){
				*ascanf_in_loop= (args[0])? True : False;
			}
			else{
				*ascanf_in_loop= True;
			}
			if( *ascanf_in_loop ){
				*ascanf_escape_value= ascanf_escape= True;
			}
		}
		*result= 1;
	}
	else{
		ascanf_emsg= " (not in a loop construct)== ";
		ascanf_arg_error= 1;
		*result= 0;
	}
	return(1);
}

int ascanf_for_to ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		*ascanf_loop_ptr= 0;
		return( 0 );
	}
	else if( ascanf_arguments< 2 ){
		*ascanf_loop_ptr= 0;
	}
	else{
		*result= ascanf_progn_return;
		*ascanf_loop_ptr= (args[1] && !NaN(args[1]))? 1 : 0;
	}
	return( 1 );
}

int ascanf_for_toMAX ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		*ascanf_loop_ptr= 0;
		return( 0 );
	}
	else if( ascanf_arguments< 2 ){
		*ascanf_loop_ptr= 0;
	}
	else{
		*result= ascanf_progn_return;
/* 		*ascanf_loop_ptr= (*ascanf_loop_counter< args[1] && !NaN(args[1]))? 1 : 0;	*/
		*ascanf_loop_ptr= fortoMAXok(*ascanf_loop_counter, *ascanf_loop_incr, args );
	}
	return( 1 );
}

/* RJB 20000421: gcc2.95.2 breaks on this with -fschedule-insns (-fschedule-insns2 is
 \ "allowed", but I guess that one does nothing with -fno-schedule-insns...) :
	/home/bertin/work/Archive/xgraph/ascanfc.c: In function `ascanf_whiledo':
	/home/bertin/work/Archive/xgraph/ascanfc.c:10929: fixed or forbidden register 0 (ax) was spilled for class AREG.
	/home/bertin/work/Archive/xgraph/ascanfc.c:10929: This may be due to a compiler bug or to impossible asm
	/home/bertin/work/Archive/xgraph/ascanfc.c:10929: statements or clauses.
	/home/bertin/work/Archive/xgraph/ascanfc.c:10929: This is the instruction:
	(insn:QI 65 58 66 (parallel[
				(set (cc0)
					(compare:CCFPEQ (reg:DF 9 %st(1))
						(reg:DF 8 %st(0))))
				(clobber (scratch:HI))
			] ) 28 {*cmpsf_cc_1-1} (insn_list 60 (insn_list 63 (nil)))
		(expr_list:REG_DEAD (reg:DF 9 %st(1))
			(expr_list:REG_DEAD (reg:DF 8 %st(0))
				(expr_list:REG_UNUSED (scratch:HI)
					(nil)))))
	Command exited with non-zero status 1
 */

/* 20020530: allowed ascanf_arguments==1 in ascanf_whiledo() and ascanf_dowhile()! */
int ascanf_whiledo ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 1 /* || (args[0] && ascanf_arguments< 2) */ ){
	  /* if !args[0], this routine is called with ascanf_arguments==1,
	   \ in which case it should set ascanf_loop to true
	   \ (this is to avoid jumping or deep nesting in ascanf_function
	   */
		ascanf_arg_error= 1;
		*ascanf_loop_ptr= 0;
		return( 0 );
	}
	else{
		*result= ascanf_progn_return;
		*ascanf_loop_ptr= (args[0]!= 0 && !NaN(args[0]) );
		return( 1 );
	}
}

int ascanf_dowhile ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 1 /* 2 */ ){
		ascanf_arg_error= 1;
		*ascanf_loop_ptr= 0;
		return( 0 );
	}
	else{
		*result= ascanf_progn_return;
		*ascanf_loop_ptr= (args[ascanf_arguments-1]!= 0 && !NaN(args[ascanf_arguments-1]) );
		return( 1 );
	}
}

/* routine for the "print[x,y,...]" ascanf syntax	*/
int ascanf_print ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i, take_usage;
	  ascanf_Function *af;
		if( ascanf_arguments< 2 ){
			*result= args[0];
		}
		else{
			*result= ascanf_progn_return;
		}
		if( (af= parse_ascanf_address(args[0], _ascanf_variable, "ascanf_print", False, &take_usage)) && take_usage ){
			fprintf( stdout, "print[%s", (af->usage)? af->usage : "<NULL>" );
		}
		else{
			fprintf( stdout, "print[%s", ad2str( args[0], d3str_format, NULL) );
		}
		for( i= 1; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_variable, "ascanf_print", False, &take_usage)) && take_usage ){
				fprintf( stdout, ",%s", (af->usage)? af->usage : "<NULL>" );
			}
			else{
				fprintf( stdout, ",%s", ad2str( args[i], d3str_format, NULL) );
			}
		}
		fprintf( stdout, "]== %s\n", ad2str( *result, d3str_format, NULL) );
		return( 1 );
	}
}

/* routine for the "Eprint[x,y,...]" ascanf syntax	*/
int ascanf_Eprint ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i, take_usage;
	  int ugi= use_greek_inf;
	  ascanf_Function *af;
		if( ascanf_arguments< 2 ){
			*result= args[0];
		}
		else{
			*result= ascanf_progn_return;
		}
		if( ascanf_SyntaxCheck ){
			return(1);
		}
		use_greek_inf= 0;
		if( (af= parse_ascanf_address(args[0], _ascanf_variable, "ascanf_Eprint", False, &take_usage)) && take_usage ){
			fprintf( StdErr, "print[%s", (af->usage)? af->usage : "<NULL>" );
		}
		else{
			fprintf( StdErr, "print[%s", ad2str( args[0], d3str_format, NULL) );
		}
		for( i= 1; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_variable, "ascanf_Eprint", False, &take_usage)) && take_usage ){
				fprintf( StdErr, ",%s", (af->usage)? af->usage : "<NULL>" );
			}
			else{
				fprintf( StdErr, ",%s", ad2str( args[i], d3str_format, NULL) );
			}
		}
		if( pragma_unlikely(!ascanf_verbose) ){
			fprintf( StdErr, "]== %s\n", ad2str( *result, d3str_format, NULL) );
		}
		else{
			fprintf( StdErr, "]== " );
		}
		use_greek_inf= ugi;
		fflush( StdErr );
		return( 1 );
	}
}

char *TBARprogress_header2= NULL;

/* routine for the "TBARprint[x,y,...]" ascanf syntax	*/
int ascanf_TBARprint ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i, len= 0, take_usage;
	  ascanf_Function *af;
	  static char *tbuf= NULL;
	  static int tbuf_len= 0;
		if( ascanf_SyntaxCheck ){
			return( 1 );
		}
		if( !tbuf_len ){
			tbuf= calloc( (tbuf_len= 256), sizeof(char) );
		}
		if( tbuf ){
			xfree( TBARprogress_header2 );
			if( (af= parse_ascanf_address(args[0], _ascanf_variable, "ascanf_TBARprint", False, &take_usage)) && take_usage ){
				len= sprintf( tbuf, (ascanf_arguments> 1)? "[%s" : "%s", (af->usage)? af->usage : "<NULL>" );
				if( ascanf_arguments== 1 ){
					if( strlen(af->usage) ){
						TBARprogress_header2= strdup(af->usage);
					}
				}
			}
			else{
				len= sprintf( tbuf, (ascanf_arguments> 1)? "[%s" : "%s", ad2str( args[0], d3str_format, NULL) );
			}
			for( i= 1; i< ascanf_arguments; i++ ){
				if( len> tbuf_len- 64 ){
					if( !(tbuf= realloc( tbuf, (tbuf_len+= 128)* sizeof(char) )) ){
						  /* lazy :))	*/
						goto TBAR_err;
					}
				}
				if( (af= parse_ascanf_address(args[i], _ascanf_variable, "ascanf_TBARprint", False, &take_usage)) && take_usage ){
					len= sprintf( tbuf, "%s,%s", tbuf, (af->usage)? af->usage : "<NULL>" );
				}
				else{
					len= sprintf( tbuf, "%s,%s", tbuf, ad2str( args[i], d3str_format, NULL) );
				}
				StringCheck( tbuf, tbuf_len, __FILE__, __LINE__ );
			}
			if( len> tbuf_len- 64 ){
				if( !(tbuf= realloc( tbuf, (tbuf_len+= 128)* sizeof(char) )) ){
					  /* lazy :))	*/
					goto TBAR_err;
				}
			}
			if( ascanf_arguments> 1 || !(af && take_usage ) ){
				len= sprintf( tbuf, "%s]==%s", tbuf, ad2str( *result, d3str_format, NULL) );
			}
			StringCheck( tbuf, tbuf_len, __FILE__, __LINE__ );
			if( ActiveWin ){
				TitleMessage( ActiveWin, tbuf );
			}
			else if( HAVEWIN ){
				XStoreName( disp, USEWINDOW, tbuf );
				if( SetIconName ){
					XSetIconName( disp, USEWINDOW, tbuf );
				}
				if( !RemoteConnection || init_window ){
					XFlush( disp );
				}
			}
			else{
				fprintf( stderr, "\033]0;%s\007", tbuf );
				fflush( stderr );
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "\"%s\" == ", tbuf );
			}
			if( ascanf_arguments< 2 ){
				*result= args[0];
			}
			else{
			  ascanf_Function *af= NULL;
			  static ascanf_Function AF= {NULL};
			  static char *AFname= "TBARprint-Static-StringPointer";
				af= &AF;
				if( af->name ){
				  double oa= af->own_address;
					xfree(af->usage);
					memset( af, 0, sizeof(ascanf_Function) );
					af->own_address= oa;
				}
				else{
					af->usage= NULL;
					af->type= _ascanf_variable;
					af->is_address= af->take_address= True;
					af->is_usage= af->take_usage= True;
					af->internal= True;
					af->name= AFname;
					take_ascanf_address(af);
				}
				af->type= _ascanf_variable;
				af->is_address= af->take_address= True;
				af->is_usage= af->take_usage= True;
				af->internal= True;
				af->name= AFname;
				xfree( af->usage );
				af->usage= XGstrdup( tbuf );
				*result= af->own_address;
			}
		}
		else{
TBAR_err:;
			fprintf( StdErr, "TBAR_print[%s,..]: can't get memory (%s)\n",
				ad2str( args[0], d3str_format, NULL), serror()
			);
			tbuf_len= 0;
		}
		return( 1 );
	}
}

/* routine for the "TBARprogress[..]" ascanf syntax	*/
int ascanf_TBARprogress ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int len= 0;
	  char *h= TBARprogress_header, *h2= TBARprogress_header2, *sep;
	  static char *tbuf= NULL;
	  static int n= -1, tbuf_len= 0;
	  static double pc, pf, ps;
	  double current, final, step, perc;
		current= args[0];
		final= args[1];
		if( ascanf_arguments> 2 ){
			CLIP_EXPR( step, args[2], 0.1, 100 );
		}
		else{
			step= 10;
		}
		perc= current* 100/ final;
		if( n== -1 || pf!= final || current< pc || step!= ps ){
			n= 0;
		}
		if( ascanf_SyntaxCheck ){
			return( 1 );
		}
		if( !TBARprogress_header ){
			h= "";
			sep= (h2)? ": " : "";
		}
		else{
			sep= ": ";
		}
		len= 130+ strlen(h)+ ((h2)? strlen(h2) : 0)+ 2* strlen(sep);
		if( len> tbuf_len ){
			tbuf= XGrealloc( tbuf, (tbuf_len= len)* sizeof(char) );
		}
		if( tbuf ){
			if( perc>= n* step ){
				n+= 1;
				len= sprintf( tbuf, "%s%s%s%s%s%% (%s of %s)",
					h, sep, (h2)? h2 : "", (h2)? sep : "",
					ad2str( perc, d3str_format, NULL),
					ad2str( current, d3str_format, NULL),
					ad2str( final, d3str_format, NULL)
				);
				StringCheck( tbuf, tbuf_len, __FILE__, __LINE__ );
				if( HAVEWIN ){
					XStoreName( disp, USEWINDOW, tbuf );
					if( SetIconName ){
						XSetIconName( disp, USEWINDOW, tbuf );
					}
					if( !RemoteConnection || init_window ){
						XFlush( disp );
					}
				}
				else{
					fprintf( stderr, "\033]0;%s\007", tbuf );
					fflush( stderr );
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "\"%s\" == ", tbuf );
				}
			}
			*result= perc;
		}
		else{
			fprintf( StdErr, "TBAR_process[%s,%s..]: can't get memory (%s)\n",
				ad2str( args[0], d3str_format, NULL),
				ad2str( args[1], d3str_format, NULL), serror()
			);
			tbuf_len= 0;
		}
		pc= current;
		pf= final;
		ps= step;
		return( 1 );
	}
}

/* routine for the "Dprint[x,y,...]" ascanf syntax	*/
FILE *Dprint_fp= NULL;

int _ascanf_Dprint( double *args, double *result, Boolean terpri )
{ int i, ac= (Dprint_fp== StdErr)? False : ascanf_comment;
  static unsigned int column= 0;
	if( ascanf_arguments< 2 && args ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	if( ascanf_SyntaxCheck || !Dprint_fp ){
		return(1);
	}
	if( column ){
		fputc( '\t', Dprint_fp);
		if( ac ){
			fputc( '\t', StdErr );
		}
	}
	if( ascanf_arguments && args ){
	 int ugi= use_greek_inf, take_usage;
	 ascanf_Function *af;
		use_greek_inf= 0;
		if( (af= parse_ascanf_address(args[0], _ascanf_variable, "ascanf_Dprint", False, &take_usage)) && take_usage ){
			fprintf( Dprint_fp, "%s", (af->usage)? af->usage : "<NULL>" );
		}
		else{
			fprintf( Dprint_fp, "%s", ad2str( args[0], d3str_format, NULL) );
		}
		if( ac ){
			if( af && af->take_usage ){
				fprintf( StdErr, "%s", (af->usage)? af->usage : "<NULL>" );
			}
			else{
				fprintf( StdErr, "%s", ad2str( args[0], d3str_format, NULL) );
			}
		}
		column+= 1;
		for( i= 1; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address(args[i], _ascanf_variable, "ascanf_Dprint", False, &take_usage)) && take_usage ){
				fprintf( Dprint_fp, "\t%s", (af->usage)? af->usage : "<NULL>" );
			}
			else{
				fprintf( Dprint_fp, "\t%s", ad2str( args[i], d3str_format, NULL) );
			}
			if( ac ){
				if( af && af->take_usage ){
					fprintf( StdErr, "\t%s", (af->usage)? af->usage : "<NULL>" );
				}
				else{
					fprintf( StdErr, "\t%s", ad2str( args[i], d3str_format, NULL) );
				}
				column+= 1;
			}
		}
		use_greek_inf= ugi;
	}
	if( terpri ){
		fputc( '\n', Dprint_fp);
		if( ac ){
			fputc( '\n', StdErr );
		}
		column= 0;
	}
	fflush( Dprint_fp );
	return( 1 );
}

/* routine for the "Dprint[x,y,...]" ascanf syntax	*/
int ascanf_Dprint ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( _ascanf_Dprint( args, result, True ) );
}

/* routine for the "Doutput[x,y,...]" ascanf syntax	*/
int ascanf_Doutput ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	return( _ascanf_Dprint( args, result, False ) );
}

static int printf_1entry( ascanf_Function *af, double arg, char **arglist, char *free_this, int argc )
{  char *c;
   int ok= 1, take_usage= 0;
	if( !af ){
		af= parse_ascanf_address( arg, 0, "printf_1entry", False, &take_usage);
	}
	if( af ){
		switch( af->type ){
			case _ascanf_variable:
				if( take_usage ){
					if( af->usage ){
						arglist[argc]= af->usage;
						free_this[argc]= 0;
					}
					else{
						if( (arglist[argc]= strdup("<NULL>")) ){
							free_this[argc]= 1;
						}
						else{
							ok= 0;
						}
					}
				}
				else{
					arglist[argc]= af->name;
					free_this[argc]= 0;
				}
				break;
			default:
				arglist[argc]= af->name;
				free_this[argc]= 0;
				break;
		}
	}
	else if( (c= strdup( ad2str( arg, d3str_format, NULL) )) ){
		arglist[argc]= c;
		free_this[argc]= 1;
	}
	else{
		ok= 0;
	}
	return(ok);
}

double printf_ValuesPrinted[2];
extern ascanf_Function *ascanf_ValuesPrinted;

// find the next % token that is the potential start of a printf/scanf format opcode
const char *next_fmt( const char *format )
{ const char *c = format, *f = NULL;
	if( format ){
		while( c && *c ){
			if( (c = strchr(c, '%')) ){
				if( c[1] != '%' ){
					f = c;
					c = NULL;
				}
				else{
					c++;
				}
			}
		}
	}
	return f;
}

// given the format string and the array arglist of narg string pointers, generate the formatted
// output in fp or (if fp==NULL) buf in a way inspired by printf. The function will replace
// opcodes of the form %? by the matching element from arglist. Recognised are
// %@ %s %g %d %u %f %e %p %x and %c and their (simple) derivatives.
int aprintf( FILE *fp, char *buf, int buflen, const char *format, char **arglist, int narg )
{ int len = 0;
	if( format && (fp || buf) ){
		if( !strchr( format, '%' ) ){
			// simple case: no format opcodes
			if( fp ){
				len = fprintf( fp, format );
			}
			else{
				len = snprintf( buf, buflen, format );
			}
		}
		else{
		  const char *nf, *cf = format;
		  int arg = 0, fn, n;
			if( !fp ){
				// empty the output string.
				buf[0] = '\0';
			}
			while( cf && *cf ){
				// find the next opcode
				nf = next_fmt(cf);
				if( nf ){
					// the length of the part of the format string from the current
					// position until the start of the opcode
					fn = nf - cf;
				}
				else{
					// no opcode: we print the remainder of the format string
					fn = strlen(cf);
				}
				if( fp ){
					// use fwrite to output exactly fn characters from cf
					len += fwrite( cf, sizeof(char), fn, fp );
					if( nf && arg< narg && arglist[arg] ){
						// we found an opcode and there's an argument:
						len += fprintf( fp, arglist[arg] );
					}
				}
				else{
					// check the amount to be stored against the remaining space,
					if( len + fn > buflen ){
						// need to truncate
						n = buflen - len;
					}
					else{
						n = fn;
					}
					// append the allowable characters from cf to the output buffer
					strncat( buf, cf, n );
					len += n;
					if( nf && arg< narg && arglist[arg] ){
						// we found an opcode and there's an argument; check its length
						// and make sure we don't append too much to the output buffer:
						n = strlen(arglist[arg]);
						if( len + n > buflen ){
							n = buflen - len;
						}
						strncat( buf, arglist[arg], n );
						len += n;
					}
				}
				if( nf ){
					// if we had found an opcode there is possibly more to output:
					// increment the argument counter and spool the format string cursor
					// cf to the 1st character after the opcode:
					arg += 1;
					cf = &nf[1];
					while( *cf && !strchr( "@sgdufepxc", *cf ) ){
						// we're skipping everything after the % character until we encounter
						// one of the recognised format specifiers.
						cf++;
					}
					// we will want to continue *after* the format opcode:
					cf++;
				}
				else{
					// we just finished our task: get us out of the loop.
					cf = NULL;
				}
			}
		}
	}
	return len;
}

/* routine for the "printf[x,y,...]" ascanf syntax	*/
int ascanf_printf ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *targ, *fptarg = NULL, *form, *af;
  FILE *fp= NULL;
  int take_usage= 0, i, arg, targ_len= 0;
  char *format, **arglist, *c, *free_this;
	*result= 0;
	if( args && ascanf_arguments>= 1 ){
		if( !(targ= parse_ascanf_address(args[0], 0, "ascanf_printf", False, &take_usage)) ){
			if( args[0]< 0 ){
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(negative filedescriptor %s: returning without output) ", ad2str(args[0],d3str_format,0) );
				}
				if( !ascanf_SyntaxCheck ){
					return(1);
				}
			}
			else{
				fp= (args[0]== 1)? stdout : StdErr;
				if( pragma_unlikely( ascanf_verbose || ascanf_SyntaxCheck ) ){
					if( args[0]!= 1 && args[0]!= 2 && !(args[0]== 0 && ascanf_SyntaxCheck) ){
						fprintf( StdErr, " (caution: filedescriptor %s accepted for StdErr for the time being) ",
							d2str( args[0],0,0)
						);
					}
				}
			}
		}
		else if( targ->fp
			   // 20120414: add an extra level of indirection to allow for printf[&fp, ...] where
			   // DCL[fp, fopen[name,mode]] OR (from python) ascanf.ExportVariable('fp', fileObject)
			   || ((fptarg = parse_ascanf_address(targ->value, 0, "ascanf_printf", False, NULL)) && fptarg->fp)
		){
			if( !fptarg ){
				fptarg = targ;
				fp= targ->fp;
			}
			else{
				fp = fptarg->fp;
			}
			take_usage= False;
			targ= NULL;
		}
		else if( !take_usage ){
			ascanf_emsg= " (1st argument must be a stringpointer, or 1 or 2 to select between stdout/StdErr) ";
			if( !ascanf_SyntaxCheck ){
				ascanf_arg_error= 1;
				return(0);
			}
		}
		if( ascanf_arguments== 1 ){
			if( take_usage ){
				xfree( targ->usage );
				if( targ->accessHandler ){
					AccessHandler( targ, targ->name, level, ASCB_COMPILED, AH_EXPR, NULL );
				}
				return(1);
			}
			else{
				ascanf_emsg= " (2nd argument may be missing to free the string pointed to by the 1st argument) ";
				ascanf_arg_error= 1;
				return(0);
			}
		}
		if( !(form= parse_ascanf_address(args[1], _ascanf_variable, "ascanf_printf", False, &take_usage)) || !take_usage ){
			if( args[1] ){
				ascanf_emsg= " (2nd argument must be a stringpointer) ";
				if( !ascanf_SyntaxCheck ){
					ascanf_arg_error= 1;
					return(0);
				}
			}
		}
		else if( pragma_unlikely(ascanf_verbose) ){
			fputs( "#\tfile ", StdErr );
			if( fptarg && fptarg->fp  ){
				print_string2( StdErr, "\"", "\"", (fptarg->usage)? fptarg->usage : "<??>", True );
				fputs( ":", StdErr );
			}
			if( fp ){
				fprintf( StdErr, "#%d ", fileno(fp) );
			}
			fprintf( StdErr, "format=" );
			print_string2( StdErr, "\"", "\"\n", (form && form->usage)? form->usage : "<NULL>", True );
		}
		format= (form && form->usage)? form->usage : NULL;
		if( ascanf_arguments== 2 ){
			if( !targ ){
				if( !format ){
					format= "<NULL>";
				}
				if( !ascanf_SyntaxCheck ){
					fputs( format, fp );
				}
				*result= strlen( format );
			}
			else if( !format ){
				if( !ascanf_SyntaxCheck ){
					xfree( targ->usage );
				}
				*result= 0;
			}
			else if( !ascanf_SyntaxCheck && (c= strdup(format)) ){
				xfree( targ->usage );
				*result= strlen( (targ->usage= c) );
				if( targ->accessHandler ){
					AccessHandler( targ, targ->name, level, ASCB_COMPILED, AH_EXPR, NULL );
				}
			}
		}
		else{
		    /* 20010504: take a very safe margin, in case user forgets some arguments (but not the format fields): */
		  int margin= 2, argc= ascanf_arguments- 2;
		  int narg= margin* argc;
			if( !format ){
				format= "<NULL>";
			}
			arglist= (char**) calloc( narg, sizeof(char*) );
			free_this= (char*) calloc( narg, sizeof(char) );
			arg= 0;
			  /* 20040602: added checks arg<narg */
			for( i= 2; i< ascanf_arguments && arglist && free_this && !ascanf_arg_error && arg< narg; i++ ){
				if( !(af= parse_ascanf_address(args[i], 0, "ascanf_printf", False, &take_usage)) ||
					(af->type== _ascanf_variable && !take_usage)
				){
				  /* This must be a double... or a pointer to a variable	*/
					if( !printf_1entry( af, args[i], arglist, free_this, arg ) ){
						ascanf_emsg= " (allocation error in ascanf_printf, line " STRING(__LINE__) " )";
						  /* Too bad about the allocated strings that arglist points to...	*/
						xfree(arglist);
						xfree(free_this);
						return(0);
					}
					arg+= 1;
				}
				  /* 20010205:	*/
				else if( take_usage ){
					  /* A "stringpointer". No need to check the type of this one.	*/
					  /* 20010614: unless it happens to be the same argument as the sink/target when that
					   \ is a string...
					   */
					if( targ && af->usage== targ->usage && !targ->fp ){
						arglist[arg]= strdup( (af->usage)? af->usage : "<NULL>" );
						free_this[arg]= 1;
					}
					else{
						arglist[arg]= (af->usage)? af->usage : "<NULL>";
						free_this[arg]= 0;
					}
					arg+= 1;
				}
				else{
					switch( af->type ){
						case _ascanf_variable:
							  /* Must be a "stringpointer". No need to make a copy of this one.	*/
							arglist[arg]= (af->usage)? af->usage : "<NULL>";
							free_this[arg]= 0;
							arg+= 1;
							break;
						case _ascanf_array:
						  /* Here things get complicated. What I want, is to be able to print
						   \ out all elements of the array. It is the user's responsability to
						   \ provide the right number of format (%) fields in the format string. It
						   \ is our responsability to expand the arglist.
						   */
							narg= margin*(argc- 1)+ af->N;
							if( (arglist= (char**) XGrealloc( arglist, narg* sizeof(char*) )) &&
								(free_this= (char*) XGrealloc( free_this, narg* sizeof(char) ))
							){
							  int j;
								argc+= af->N- 1;
								for( j= 0; j< af->N && arg< narg; j++ ){
									if( af->iarray ){
										arglist[arg]= strdup( ad2str( (double) af->iarray[j], d3str_format, NULL) );
										free_this[arg]= 1;
									}
									else{
										printf_1entry( NULL, af->array[j], arglist, free_this, arg );
									}
									if( !arglist[arg] ){
										ascanf_emsg= " (allocation error in ascanf_printf, line " STRING(__LINE__) " )";
										  /* Too bad about the allocated strings that arglist points to...	*/
										xfree(arglist);
										xfree(free_this);
										return(0);
									}
									arg+= 1;
								}
							}
							else{
								ascanf_emsg= " (allocation error in ascanf_printf, line " STRING(__LINE__) " )";
								xfree( arglist );
								xfree( free_this );
								return(0);
							}
							break;
						case _ascanf_procedure:
/* 							arglist[arg]= (af->procedure->expr)? af->procedure->expr : "<NULL>";	*/
							arglist[arg]= FUNNAME(af);
							free_this[arg]= 0;
							arg+= 1;
							break;
						default:
/* 							ascanf_emsg= " (pointer to unsupported type) ";	*/
/* 							ascanf_arg_error= 1;	*/
							if( !printf_1entry( af, args[i], arglist, free_this, arg ) ){
								ascanf_emsg= " (allocation error in ascanf_printf, line " STRING(__LINE__) " )";
								  /* Too bad about the allocated strings that arglist points to...	*/
								xfree(arglist);
								xfree(free_this);
								return(0);
							}
							arg+= 1;
							break;
					}
				}
				if( pragma_unlikely(ascanf_verbose) ){
					if( af ){
						fprintf( StdErr, "#\titem %d \"%s\"=", arg, af->name );
					}
					else{
						fprintf( StdErr, "#\titem %d=", arg );
					}
					print_string2( StdErr, "\"", "\"\n", arglist[arg-1], True );
				}
			}
			for( ; arg< narg; arg++ ){
				arglist[arg]= (char*) "<!MISSING!>";
				free_this[arg]= '\0';
			}
			arglist[narg-1] = NULL;
			  /* 20020314: I added a 'compiled' field to the callback frame variable. Check the arguments
			   \ when evaluating a non-compiled expression. This is necessary because the compiler would
			   \ allocate undefined variables that are passed as arguments, whereas this won't happen
			   \ for a non-compiled expression, which would then happily crash the programme when vfprintf()
			   \ tries to access arguments that weren't passed. The check-for-enough arguments will now
			   \ expand the arglist array passed to vfprintf() to accomodate for superfluous printing fields in
			   \ the format string.
			   \ 20040223: do the check when Allocate_Internal>0 && ascanf_verbose...
			   \ 20040604: also do the check when somebody is interested in the $ValuesPrinted array.
			   */
#ifdef ASCANF_ALTERNATE
			if( ascanf_ValuesPrinted->links ||
				((ascanf_verbose || ascanf_SyntaxCheck || !__ascb_frame->compiled) && (*Allocate_Internal<= 0 || ascanf_verbose))
			)
#else
			if( ascanf_ValuesPrinted->links ||
				((ascanf_verbose || ascanf_SyntaxCheck) && (*Allocate_Internal<= 0 || ascanf_verbose))
			)
#endif
			{ int n= 0;
			  char *c= format;
				while( (c= strstr(c, "%s")) ){
					if( c== format || c[-1]!= '%' ){
						n+= 1;
					}
					c+= 2;
				}
				  /* 20040124: don't confirm valid input when not compiling */
				  /* 20050413: even more reduced confirmation. */
				if( argc!= n || (ascanf_SyntaxCheck && ascanf_verbose) ){
					if( ascanf_SyntaxCheck ){
#if defined(ASCANF_ALTERNATE) && defined(DEBUG)
						PF_print_string( StdErr, "\n", "\n", AH_EXPR, False );
						if( __ascb_frame->compiled ){
							Print_Form( StdErr, &__ascb_frame->compiled, 1, True, NULL, "#\t", NULL, False );
						}
#endif
						fprintf( StdErr,
							" (%d total print arguments from %d passed, %sformat string has %d valid %%s fields) ",
							argc, ascanf_arguments- 2, (argc!= n)? "ERROR: " : "", n
						);
					}
					else{
#if defined(ASCANF_ALTERNATE) && defined(DEBUG)
						PF_print_string( StdErr, "\n", "\n", AH_EXPR, False );
#endif
						ascanf_emsg= " (printf[] number of format fields does not match the number of arguments passed) ";
						ascanf_arg_error= True;
					}
				}
				if( n> argc ){
				  /* 20020314: (try to) protect against superfluous fields by expanding the arglist array */
					narg= n;
					if( (arglist= (char**) XGrealloc( arglist, narg* sizeof(char*) )) &&
						(free_this= (char*) XGrealloc( free_this, narg* sizeof(char) ))
					){
						for( ; arg< narg; arg++ ){
							arglist[arg]= (char*) "<!MISSING!>";
							free_this[arg]= '\0';
						}
						arglist[narg-1] = NULL;
					}
					else{
						xfree( arglist );
						xfree( free_this );
						fprintf( StdErr, " (!! can't reallocate buffers to accomodate additional fields (%s)) ", serror() );
						fflush( StdErr );
						return(0);
					}
				}
				fflush( StdErr );
				printf_ValuesPrinted[0]= argc;
				printf_ValuesPrinted[1]= n;
			}
			if( targ && !ascanf_SyntaxCheck ){
			  FILE *nfp= NullDevice;
// 20110326: 64 bits mode on Mac OS X doesn't allow the kludge to pass an array of stringpointers instead
// of a va_list argument so we cannot use vfprintf to determine the exact length of the output. Which isn't too much of
// an issue as we no longer use vfprintf/vsnprintf to generate the actual output...
#ifndef __x86_64__
				if( 0 && nfp ){
					  /* vfprintf() returns the printed lenght MINUS the closing nullbyte.	*/
					targ_len= vfprintf( nfp, format, arglist )+ 1;
				}
 				else
#endif
				{ char *c;
					targ_len = strlen(format) + 1;
					// worse-case calculation of the required string length: the length
					// of the format string, plus the combined length of all arglist entries
					// that correspond to a single % token in the format string.
					arg = 0;
					c = format;
					while( (c = strchr(c, '%')) && arg < narg ){
						targ_len += (arglist[arg])? strlen(arglist[arg]) : 1;
						arg += 1;
						c++;
					}
				}
				if( targ_len> 0 && (c= (char*) calloc( targ_len, sizeof(char) )) ){
					xfree( targ->usage );
					targ->usage= c;
				}
				else{
					ascanf_emsg= " (can't allocate target storage, or invalid print length) ";
					return(0);
				}
			}
			  /* 20040304: don't enter the following block if ascanf_noverbose... 	*/
			if( (ascanf_verbose || ascanf_SyntaxCheck) && !ascanf_noverbose && !*Allocate_Internal> 0 ){
				if( targ ){
					fprintf( StdErr, ", target/print length will be %d", targ_len- 1 );
				}
				fputs( ")", StdErr );
				if( ascanf_verbose ){
					fputs( "== ", StdErr );
				}
				else if( ascanf_SyntaxCheck ){
					fputc( '\n', StdErr );
				}
				fflush( StdErr );
			}
			if( pragma_likely( !ascanf_SyntaxCheck ) ){
				if( targ ){
#if 0
/* #ifdef linux	*/
					*result= vsnprintf( targ->usage, targ_len, format, arglist );
/* #else	*/
/* 					*result= vsprintf( targ->usage, format, arglist );	*/
/* #endif	*/
#else
					*result = aprintf( NULL, targ->usage, targ_len, format, arglist, narg );
#endif
					targ->value= *result;
					targ->assigns+= 1;
					if( targ->accessHandler ){
						AccessHandler( targ, targ->name, level, ASCB_COMPILED, AH_EXPR, NULL );
					}
				}
				else{
#if 0
					*result= vfprintf( fp, format, arglist );
#else
					*result = aprintf( fp, NULL, 0, format, arglist, narg );
#endif
					fflush( fp );
				}
			}
			for( i= 0; i< argc; i++ ){
				if( free_this[i] ){
					xfree( arglist[i] );
				}
			}
			xfree( arglist );
			xfree( free_this );
		}
		return(1);
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(0);
	}
}

/* routine for the "matherr[x,y,...]" ascanf syntax	*/
int ascanf_matherr_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 1 ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return( 1 );
}

double DBG_SHelp( char *string, int internal )
{ double result;
  int AI= (int) *Allocate_Internal;
	ascanf_var_search= string;
	ascanf_usage_search= string;
	ascanf_label_search= string;
	*Allocate_Internal= internal;
	if( ActiveWin ){
		result= help_fnc( ActiveWin, True, True );
		  /* Just to be sure..	*/
		xtb_popup_delete( &vars_pmenu );
	}
	else{
		result= show_ascanf_functions( StdErr, "\t", True, 1 );
	}
	ascanf_var_search= NULL;
	ascanf_usage_search= NULL;
	ascanf_label_search= NULL;
	*Allocate_Internal= AI;
	return( result );
}

/* This routine is a handler for readline() */
char *grl_MatchVarNames( char *string, int state )
{ char *result= NULL;
  int hit= 0;
	if( !state ){
	  static char *alter= NULL;
		if( pragma_unlikely(debugFlag) ){
			fprintf( StdErr, "\ngrl_MatchVarNames(\"%s\")\n", string );
		}
		ascanf_var_search= string;
		ascanf_usage_search= NULL;
		ascanf_label_search= NULL;
		if( ActiveWin ){
			xfree(help_fnc_selected);
			hit= help_fnc( ActiveWin, True, False );
			if( help_fnc_selected ){
				result= help_fnc_selected;
				help_fnc_selected = NULL;
			}
			  /* Just to be sure..	*/
			xtb_popup_delete( &vars_pmenu );
		}
		else{
			hit= show_ascanf_functions( StdErr, "\t", True, 1 );
		}
		ascanf_var_search= NULL;
		ascanf_usage_search= NULL;
		ascanf_label_search= NULL;
		if( !hit && !alter ){
			alter= concat( "$", string, NULL );
			result= grl_MatchVarNames(alter, state);
			if( result && result[0]== '$' ){
			  char *next= &result[1], *r= result;
				  /* remove the $ that would otherwise appear twice... */
				while( *next ){
					*r++ = *next++;
				}
				*r= '\0';
			}
			xfree(alter);
		}
	}
	return( result );
}

/* routine for the "SHelp[name]" ascanf syntax	*/
int ascanf_SHelp ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments< 1 || !args ){
		ascanf_arg_error= 1;
		return(0);
	}
	else if( ascanf_arguments== 1 ){
	  /* We return the 1st and only argument	*/
		*result= args[0];
	}
	else{
	  /* or the argument selected by return[]	*/
		*result= ascanf_progn_return;
	}
	return( 1 );
}

/* routine for the "compile[x,y,...]" ascanf syntax	*/
int ascanf_compile ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments< 1 || !args ){
		ascanf_arg_error= 1;
		return(0);
	}
	else if( ascanf_arguments== 1 ){
	  /* We return the 1st and only argument	*/
		*result= args[0];
	}
	else{
	  /* or the argument selected by return[]	*/
		*result= ascanf_progn_return;
	}
	return( 1 );
}

/* routine for the "compile-noEval[x,y,...]" ascanf syntax	*/
int ascanf_noEval ( ASCB_ARGLIST )
{
	return( ascanf_compile( ASCB_ARGUMENTS ));
}

/* routine for the "verbose[x,y,...]" ascanf syntax	*/
int ascanf_verbose_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 1 ){
	  /* We return the 1st and only argument	*/
		*result= args[0];
	}
	else{
	  /* or the argument selected by return[]	*/
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "return[%s]== ", ad2str( ascanf_progn_return, d3str_format, NULL) );
		}
		*result= ascanf_progn_return;
	}
	return( 1 );
}

int ascanf_noverbose_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 1 ){
	  /* We return the 1st and only argument	*/
		*result= args[0];
	}
	else{
	  /* or the argument selected by return[]	*/
		*result= ascanf_progn_return;
	}
	return( 1 );
}

int ascanf_IDict_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 1 ){
	  /* We return the 1st and only argument	*/
		*result= args[0];
	}
	else{
	  /* or the argument selected by return[]	*/
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "return[%s]== ", ad2str( ascanf_progn_return, d3str_format, NULL) );
		}
		*result= ascanf_progn_return;
	}
	return( 1 );
}

int ascanf_global_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 1 ){
	  /* We return the 1st and only argument	*/
		*result= args[0];
	}
	else{
	  /* or the argument selected by return[]	*/
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "return[%s]== ", ad2str( ascanf_progn_return, d3str_format, NULL) );
		}
		*result= ascanf_progn_return;
	}
	return( 1 );
}

// 20101025: it may not be necessary to allocate (yet again) an array for the arguments, if our caller already did that.
int ascanf_call_method( ascanf_Function *af, int argc, double *args, double *result, int *retval, ascanf_Callback_Frame *__ascb_frame, int alloc_largs )
{ int aargc= ascanf_arguments, r;
  int aerror= 0;

	ascanf_arguments= argc;
	switch( af->type ){
		case _ascanf_procedure:{
		  int ok= False;
		  double *lArgList= af_ArgList->array;
		  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList, *level= __ascb_frame->level;
#ifdef ASCB_FRAME_EXPRESSION
		  char *expr= __ascb_frame->expr;
#else
		  char lexpr[128]= "call[<procedure>,...]";
		  char *expr= (AH_EXPR)? AH_EXPR : lexpr;
#endif
		  double *largs;
			if( alloc_largs ){
#if DEBUG == 2
				largs= (double*) calloc( argc+1, sizeof(double) );
#else
				largs= (double*) malloc( (argc+1) * sizeof(double) );
#endif
			}
			else{
				largs = args;
			}
			if( ascanf_arguments ){
				if( alloc_largs ){
					memcpy( largs, args, ascanf_arguments* sizeof(double) );
				}
				SET_AF_ARGLIST( largs, ascanf_arguments );
			}
			else{
				SET_AF_ARGLIST( ascanf_ArgList, 0 );
			}
			ascanf_update_ArgList= False;
			if( pragma_unlikely( ascanf_verbose && strncmp(af->name, "\\l\\expr-",8)== 0 ) ){
				fprintf( StdErr, "\n#%s%d\\l\\\tlambda[", (__ascb_frame->compiled)? "C" : "", *level );
				Print_Form( StdErr, &af->procedure, 0, True, NULL, "#\\l\\\t", NULL, False );
				fputs( "]\n", StdErr );
			}
			af->procedure->level+= 1;
			if( call_compiled_ascanf_function( 0, &expr, result, &ok, "ascanf_call", &af->procedure, level) ){
				af->value= af->procedure->value= *result;
				ok= True;
				if( pragma_unlikely(ascanf_verbose) ){
					Print_ProcedureCall( StdErr, af, level );
					fprintf( StdErr, "== %s\n", ad2str(*result, d3str_format, NULL) );
				}
				r= 1;
			}
			else{
				r= 0;
			}
			af->procedure->level-= 1;
			if( ok && af->accessHandler ){
				AccessHandler( af, af->name, level, __ascb_frame->compiled, NULL, NULL );
			}
			SET_AF_ARGLIST( lArgList, lArgc );
			ascanf_update_ArgList= auA;
			GCA();
			if( alloc_largs ){
				xfree( largs );
			}
			break;
		}
		case NOT_EOF:
		case NOT_EOF_OR_RETURN:
		case _ascanf_function:
			if( af->function ){
//				__ascb_frame->args= args;
				r= (*af->function)( __ascb_frame );
//				__ascb_frame->args= args;
			}
			break;
		default:
			ascanf_emsg= " (unsupported type in ascanf_call) ";
			aerror= 1;
			r= 0;
			break;
	}
	ascanf_arguments= aargc;
	if( retval ){
		*retval= r;
	}
	return(aerror);
}

int ascanf_call ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
#ifdef ASCANF_ALTERNATE
  ascanf_Function *af;
  int r, alloc_largs;
	if( args && ascanf_arguments ){
		if( (af= parse_ascanf_address( args[0], 0, "ascanf_call", (int) ascanf_verbose, NULL )) ){
		  double *largs;
		  int largc= ascanf_arguments-1;
			switch( af->type ){
				default:
					largs= &args[1];
					alloc_largs = 1;
					break;
				case NOT_EOF:
				case NOT_EOF_OR_RETURN:
				case _ascanf_python_object:
				case _ascanf_function:
					if( largc< af->Nargs ){
						if( (largs= (double*) malloc( af->Nargs* sizeof(double))) ){
						  int i;
							memcpy( largs, &args[1], largc*sizeof(double) );
							for( i= largc; i< af->Nargs; i++ ){
								largs[i]= 0;
							}
							alloc_largs = 0;
						}
						else{
							fprintf( StdErr, " (call: can't get %d local argument buffer (%s)) ", af->Nargs, serror() );
							largs= &args[1];
							alloc_largs = 1;
						}
					}
					else{
						largs= &args[1];
						alloc_largs = 1;
					}
					break;
			}
			ascanf_arg_error= ascanf_call_method( af, largc, largs, result, &r,
				__ascb_frame, alloc_largs
			);
			if( largs!= &args[1] ){
				xfree(largs);
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		r= 0;
	}
	return(r);
#else
	fprintf( StdErr, "ascanf_call(): not implemented for this model of callback parameter passing -- #define ASCANF_ALTERNATE!!\n" );
	ascanf_emsg= " (functionality not available) ";
	ascanf_arg_error= 1;
	return(0);
#endif
}

/* routine for the "comment[x,y,...]" ascanf syntax	*/
int ascanf_comment_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 1 ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return( 1 );
}

/* routine for the "popup[x,y,...]" ascanf syntax	*/
int ascanf_popup_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments== 1 ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return( 1 );
}

/* routine for the "last_popup[x,y,...]" ascanf syntax	*/
int ascanf_last_popup_fnc ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( popup_menu ){
	  char *sel= NULL;
		xtb_popup_menu( ascanf_window, "last popup's output", "Stored scope's output", &sel, &popup_menu);
		if( sel ){
			while( *sel && isspace( (unsigned char) *sel) ){
				sel++;
			}
		}
		if( sel && *sel ){
			if( pragma_unlikely(debugFlag) ){
				xtb_error_box( ascanf_window, sel, "Copied to clipboard:" );
			}
			else{
				Boing(10);
			}
			XStoreBuffer( disp, sel, strlen(sel), 0);
			  // RJVB 20081217
			xfree(sel);
		}
	}
	if( ascanf_arguments== 1 ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return( 1 );
}

/* routine for the "no.system.time[x,y,...]" ascanf syntax	*/
int ascanf_systemtime_silent ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	set_NaN(*result);
	return( 1 );
}

/* routine for the "system.time[x,y,...]" ascanf syntax	*/
int ascanf_systemtime ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	set_NaN(*result);
	return( 1 );
}

/* routine for the "system.time.p[x,y,...]" ascanf syntax	*/
int ascanf_systemtime2 ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	set_NaN(*result);
	return( 1 );
}

/* routine for the "fmod[x,y]" ascanf syntax	*/
int ascanf_fmod ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= (args[1])? fmod( args[0], args[1] ) : args[0];
		return( 1 );
	}
}

/* routine for the "fmod2[x,y]" ascanf syntax	*/
int ascanf_fmod2 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  double x_y= args[0]/args[1];
		ascanf_arg_error= (ascanf_arguments< 2 );
/*   		*result= args[0] - args[1] * SIGN(x_y) * ssfloor( ABS(x_y) );	*/
 		if( x_y< 0 ){
 			*result= args[0] + args[1] * ssfloor( -x_y );
 		}
 		else{
 			*result= args[0] - args[1] * ssfloor( x_y );
 		}
		return( 1 );
	}
}

double modulo_x( double x, double d_s)
{ double res;

  res = d_s;

  while( res< -0.5*x || res>0.5*x ){
     if( res> 0.5*x ){
		res -= x;
	}
     if( res< -0.5*x ){
		res += x;
	}
  }
  return( res );
}

/* routine for the "fmod3[x,y]" ascanf syntax	*/
int ascanf_fmod3 ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= modulo_x( args[0] , args[1] );
		return( 1 );
	}
}

/* routine for the "XSync[0/1]" ascanf syntax	*/
int ascanf_XSync ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int Synchro_State;
  extern void *X_Synchro();
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( args[0] ){
			Synchro_State= 0;
			X_Synchro(NULL);
		}
		else{
			Synchro_State= 1;
			X_Synchro(NULL);
		}
		XSync( disp, False );
		*result= ascanf_progn_return;
		return( 1 );
	}
}

/* routine for the "compress[x,C[,F]]" ascanf syntax	*/
int ascanf_compress ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 2){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments< 3 ){
			*result= args[0]/ ( ABS(args[0]) + args[1] );
		}
		else{
		  double F= args[2];
			*result= pow(args[0],F)/ ( pow( ABS(args[0]),F) + pow( args[1],F) );
		}
		return( 1 );
	}
}

/* routine for the "lowpass[x,tau,mem_index]" ascanf syntax	*/
int ascanf_lowpass ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double param_dt;
	if( !args || ascanf_arguments< 3){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i= (int) args[2];
	  double mem= (args[1])? exp( - param_dt / args[1] ) : 0.0;
		CHK_ASCANF_MEMORY;
		if( !ascanf_SyntaxCheck ){
			ascanf_memory[i]= *result= mem* ascanf_memory[i]+ args[0];
		}
		return( 1 );
	}
}

/* routine for the "nlowpass[x,tau,mem_index]" ascanf syntax	*/
int ascanf_nlowpass ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double param_dt;
	if( !args || ascanf_arguments< 3){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i= (int) args[2];
	  double dt= (ascanf_arguments> 3)? args[3] : param_dt;
	  double mem= (args[1])? exp( - dt / args[1] ) : 0.0;
		CHK_ASCANF_MEMORY;
		if( !ascanf_SyntaxCheck ){
			ascanf_memory[i]= *result= mem* ascanf_memory[i]+ (1-mem)* args[0];
		}
		return( 1 );
	}
}

/* routine for the "nlowpass*[x,tau,fac,mem_index]" ascanf syntax	*/
int ascanf_nlowpassB ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double param_dt;
	if( !args || ascanf_arguments< 4){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i= (int) args[3];
	  double dt= (ascanf_arguments> 4)? args[4] : param_dt;
	  double input= args[0], fac= args[2], mem= (args[1])? exp( - dt / args[1] ) : 0.0;
		CHK_ASCANF_MEMORY;
		if( !ascanf_SyntaxCheck ){
			if( input< ascanf_memory[i] ){
				mem*= fac;
			}
			else{
				mem/= fac;
			}
			ascanf_memory[i]= *result= mem* ascanf_memory[i]+ (1-mem)* input;
		}
		return( 1 );
	}
}

/* routine for the "nlowpass**[x,tau,fac,mem_index]" ascanf syntax	*/
int ascanf_nlowpassC ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double param_dt;
	if( !args || ascanf_arguments< 4){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i= (int) args[3];
	  double dt= (ascanf_arguments> 4)? args[4] : param_dt;
	  double input= args[0], fac= args[2], mem, tau= args[1];
		CHK_ASCANF_MEMORY;
		if( !ascanf_SyntaxCheck ){
			if( input< ascanf_memory[i] ){
				tau*= fac;
			}
			else{
				tau/= fac;
			}
			mem= (tau)? exp( - dt / tau ) : 0.0;
			ascanf_memory[i]= *result= mem* ascanf_memory[i]+ (1-mem)* input;
		}
		return( 1 );
	}
}

/* routine for the "shunt[x,y,C,tau,mem_index]" ascanf syntax */
int ascanf_shunt ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double param_dt;
	if( !args || ascanf_arguments< 5){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i= (int) args[4];
	  double dt= (ascanf_arguments> 5)? args[5] : param_dt;
	  double mem= exp( - dt / args[3] );
		CHK_ASCANF_MEMORY;
		if( !ascanf_SyntaxCheck ){
			ascanf_memory[i]= *result=
				ascanf_memory[i]*( 1+ (mem- 1)* (args[2]+ args[1]))+ (1- mem)* args[0];
		}
		return( 1 );
	}
}

/* 20020322: routines to get at the calling parent and grandparent function. Not yet (fully)
 \ operational for getting at user-defined procedures (must implement a stub Compiled_Form
 \ trick for that).
 */
int ascanf_ParentsName ( ASCB_ARGLIST )
{ static ascanf_Function AF= {NULL};
  static char *AFname= "ParentsName-Static-StringPointer";
  ascanf_Function *af;
	af= &AF;
	if( af->name ){
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
	}
	else{
		af->usage= NULL;
	}
	af->name= AFname;
	af->type= _ascanf_variable;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	af->internal= True;
#if defined(ASCANF_ALTERNATE)
	{ Compiled_Form *form;
		if( (form= __ascb_frame->compiled) ){
			if( form->parent ){
				af->usage= XGstrdup( FUNNAME(form->parent->fun) );
			}
			else{
				af->usage= strdup( "<toplevel>" );
			}
		}
		else{
			af->usage= strdup( "<uncompiled!>" );
		}
	}
#else
	af->usage= strdup( "<unsupported" );
#endif
	*(__ascb_frame->result)= take_ascanf_address(af);
	return(1);
}

int ascanf_GrandParentsName ( ASCB_ARGLIST )
{ static ascanf_Function AF= {NULL};
  static char *AFname= "GrandParentsName-Static-StringPointer";
  ascanf_Function *af;
	af= &AF;
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
#if defined(ASCANF_ALTERNATE)
	{ Compiled_Form *form;
		if( (form= __ascb_frame->compiled) ){
			if( form->parent ){
				form= form->parent;
				if( form->parent ){
					af->usage= XGstrdup( FUNNAME(form->parent->fun) );
				}
				else{
					af->usage= strdup( "<toplevel>" );
				}
			}
			else{
				af->usage= strdup( "<parent is toplevel>" );
			}
		}
		else{
			af->usage= strdup( "<uncompiled!>" );
		}
	}
#else
	af->usage= strdup( "<unsupported" );
#endif
	*(__ascb_frame->result)= take_ascanf_address(af);
	return(1);
}

int ascanf_procedureName ( ASCB_ARGLIST )
{ static ascanf_Function AF[10]= { {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL} };
  static char *AFname= "procedureName-Static-StringPointer";
  ascanf_Function *af= NULL, *paf= NULL;
  int level= 0, found= 0;
  static unsigned char bufno= 0;
	if( ascanf_arguments ){
		if( !(paf= parse_ascanf_address(__ascb_frame->args[0], _ascanf_procedure, "ascanf_procedureName",
				(int) ascanf_verbose, NULL ))
		){
			CLIP_EXPR_CAST( int, level, double, __ascb_frame->args[0], 0, MAXINT );
		}
	}
	af= &AF[bufno % 10];
	bufno+= 1;
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
	if( paf ){
		af->usage= strdup(paf->name);
	}
	else{
#if defined(ASCANF_ALTERNATE)
	  Compiled_Form *form;
		if( (form= __ascb_frame->compiled) ){
			while( form && (form->type!= _ascanf_procedure || found!= level) && form!= form->parent ){
				if( form->type== _ascanf_procedure ){
					found+= 1;
				}
				form= form->parent;
			}
			if( form && form->fun && form->fun->type== _ascanf_procedure ){
				af->usage= XGstrdup( FUNNAME(form->fun) );
			}
			else{
				af->usage= strdup( "<toplevel>" );
			}
		}
		else{
			af->usage= strdup( "<uncompiled!>" );
		}
#else
		af->usage= strdup( "<unsupported" );
#endif
	}
	*(__ascb_frame->result)= take_ascanf_address(af);
	return(1);
}

int ascanf_procedureCode ( ASCB_ARGLIST )
{ static ascanf_Function AF[10]= { {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL}, {NULL} };
  static char *AFname= "procedureCode-Static-StringPointer";
  ascanf_Function *af, *paf;
  int level= 0, found= 0, ok= False;
  static unsigned char bufno= 0;
	if( ascanf_arguments ){
		if( !(paf= parse_ascanf_address(__ascb_frame->args[0], _ascanf_procedure, "ascanf_procedureCode",
				(int) ascanf_verbose, NULL ))
		){
			CLIP_EXPR_CAST( int, level, double, __ascb_frame->args[0], 0, MAXINT );
		}
	}
	af= &AF[bufno % 10];
	bufno+= 1;
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
	if( paf && paf->procedure && paf->procedure->expr ){
		af->usage= strdup(paf->procedure->expr);
		ok= True;
	}
	else{
#if defined(ASCANF_ALTERNATE)
	  Compiled_Form *form;
		if( (form= __ascb_frame->compiled) ){
			while( form && (form->type!= _ascanf_procedure || found!= level) && form!= form->parent ){
				if( form->type== _ascanf_procedure ){
					found+= 1;
				}
				form= form->parent;
			}
			if( form && form->fun && form->fun->type== _ascanf_procedure && form->fun->procedure && form->fun->procedure->expr ){
				af->usage= XGstrdup( form->fun->procedure->expr );
				ok= True;
			}
		}
#endif
	}
	*(__ascb_frame->result)= (ok)? take_ascanf_address(af) : 0;
	return(1);
}

int ascanf_Variable ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( args ){
		*result= args[0];
		return( 1 );
	}
	else{
	  /* this routine should only be called when a variable/label has (an) argument(s)	*/
		ascanf_arg_error= 1;
		return( 0 );
	}
}

int ascanf_Procedure ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	*result= ascanf_progn_return;
	return( 1 );
}

int ascanf_DeclareVariable ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 1){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments>= 2 && current_DCL_type== _ascanf_array ){
			if( ascanf_arguments== 2 ){
			  double idx= args[1];
				if( idx== -2 ){
					  /* 20020530 */
					CLIP_EXPR( *result, current_DCL_item->last_index, 0, current_DCL_item->N-1 );
				}
				else if( idx== -1 ){
					*result= current_DCL_item->N;
				}
				else if( idx>= 0 && idx< current_DCL_item->N ){
DCL_access_array_element:;
					*result= (current_DCL_item->iarray)? current_DCL_item->iarray[ (int) idx ] :
						current_DCL_item->array[ (int) idx ];
				}
				else{
					if( Inf(idx)> 0 ){
						idx= current_DCL_item->N-1;
						goto DCL_access_array_element;
					}
					else if( Inf(idx)< 0 ){
						idx= 0;
						goto DCL_access_array_element;
					}
					else if( AllowArrayExpansion && Resize_ascanf_Array( current_DCL_item, idx+1, NULL ) ){
						goto DCL_access_array_element;
					}
					fprintf( StdErr, "ascanf_DeclareVariable(DCL[%s,%g]",
						current_DCL_item->name, idx
					);
#if defined(ASCANF_ALTERNATE) && defined(DEBUG)
					fprintf( StdErr, "\"%s\"", AH_EXPR );
#endif
					fprintf( StdErr, "): internal error: index out-of-range [0,%d> (line %d)!\n", current_DCL_item->N, __LINE__ );
					ascanf_arg_error= 1;
					ascanf_emsg= " (index out-of-range (" STRING(__LINE__) ") ) ";
					if( current_DCL_item->N< 0 ){
						fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", current_DCL_item->name );
						ascanf_escape= ascanf_interrupt= True;
						*ascanf_escape_value= *ascanf_interrupt_value= 1;
					}
					return(0);
				}
			}
			else{
				*result= args[2];
			}
		}
		else if( ascanf_arguments> 1 ){
			*result= args[1];
		}
		else{
			*result= args[0];
		}
		return( 1 );
	}
}

int ascanf_DeclareProcedure ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 1){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments> 1 ){
			*result= args[1];
		}
		else{
			*result= args[0];
		}
		return( 1 );
	}
}

int Delete_Internal_Variable( char *name, ascanf_Function *entry )
{ int listSize= internal_Functions, j, r= 0, match;
  ascanf_Function *paf, *af, *funList= vars_internal_Functions;
	if( !name && !entry ){
		return(0);
	}
	if( entry && !entry->internal ){
		return(0);
	}
	for( j= 0; j< listSize; j++ ){
		af= &funList[j];
		paf= NULL;
		while( af ){
		  ascanf_Function *cdr;
			match= (name)? strcmp( af->name, name)== 0 : af== entry;
			if( (af->function== ascanf_Variable || af->function== ascanf_Procedure) &&
				af->type!= _ascanf_novariable &&
				!af->links &&
				match
			){
				cdr= af->cdr;
				Delete_Variable( af );
#if !defined(ASCANF_ARG_DEBUG) && !defined(DEBUG)
					if( pragma_unlikely(ascanf_verbose== 2) )
#endif
					{
						fprintf( StdErr, "# deleted internal %s (A=%d R=%d L=%d)\n",
							af->name, af->assigns,
							af->reads, af->links
						);
					}
				delete_VariableName(af->name);
				xfree( af->name );
				  /* 20020524: a miracle that I didn't get crashes before because of NULL af->name fields...! */
				af->name= strdup("");
				if( ascanf_progn_return== af->own_address ){
					ascanf_progn_return= 0;
				}
#ifdef USE_AA_REGISTER
				delete_ascanf_Address(AAF_REPRESENTATION(af));
#endif
				af->own_address= 0;
				r+= 1;
				if( paf ){
					paf->cdr= cdr;
				}
				  /* NB: the following statement supposes that Delete_Variable doesn't de-allocate af!! */
				paf= af;
				af= cdr;
			}
			else{
				paf= af;
				af= af->cdr;
			}
		}
	}
	return(r);
}

int ascanf_EditProcedure ( ASCB_ARGLIST )
{
	return( ascanf_DeclareProcedure( ASCB_ARGUMENTS ));
}

int ascanf_DeleteVariable ( ASCB_ARGLIST )
{
	return( ascanf_DeclareVariable( ASCB_ARGUMENTS ));
}

int ascanf_DefinedVariable ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	*result= 1;
	return( 1 );
}

int ascanf_setdebug ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int *dbF_cache;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	debugFlag= (int) args[0];
	if( dbF_cache ){
		*dbF_cache= debugFlag;
	}
	if( ascanf_arguments> 1 && !NaN(args[1]) ){
		debugLevel= (int) args[1];
	}
	*result= ascanf_progn_return;
	return(1);
}

int ascanf_ClearReadBuffer ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  extern char event_read_buf[];
  extern double *ascanf_ReadBufVal;
  extern int event_read_buf_cleared;
	event_read_buf[0]= '\0';
	event_read_buf_cleared= True;
	*result= *ascanf_ReadBufVal;
	set_NaN( *ascanf_ReadBufVal );
	return(1);
}

int ascanf_PopReadBuffer ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  extern char event_read_buf[];
  extern double *ascanf_ReadBufVal;
  extern int event_read_buf_cleared;
	event_read_buf[strlen(event_read_buf)-1]= '\0';
	event_read_buf_cleared= True;
	*result= *ascanf_ReadBufVal;
	if( sscanf( event_read_buf, "%lf", ascanf_ReadBufVal )< 1 ){
		set_NaN(*ascanf_ReadBufVal);
	}
	return(1);
}

int ascanf_CheckPointers ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int i;
	if( args && ascanf_arguments>= 1 ){
#ifndef USE_AA_REGISTER
	  int Pap= PAS_already_protected;
#endif
		*result= 0;
		for( i= 0; i< ascanf_arguments; i++ ){
#ifndef USE_AA_REGISTER
			PAS_already_protected= False;
#endif
#if 0
			  /* hack to test the address-parsing-protection (Ascanf_Lazy_Address_Protection) mechanism: */
			{ AscanfAddresses aa;
				aa.value= args[i];
				  /* This should provoke problems: */
				aa.handle.address= (i)? 10 : aa.handle.address+1;
				if( parse_ascanf_address( aa.value, 0, "ascanf_CheckPointers", (int) ascanf_verbose, NULL ) ){
					*result+= 1;
				}
			}
#else
			if( parse_ascanf_address( args[i], 0, "ascanf_CheckPointers", (int) ascanf_verbose, NULL ) ){
				*result+= 1;
			}
#endif
		}
#ifndef USE_AA_REGISTER
		PAS_already_protected= Pap;
#endif
		return(1);
	}
	else{
		*result= 0;
		ascanf_arg_error= 1;
		return(0);
	}
}

extern double IntensityRGBValues[7];

extern double LastActionDetails[5];

ascanf_Function internal_AHandler= { "$internal-access-handler", ascanf_Variable, 2, _ascanf_variable,
	"An internal accesshandler variable",
	0, 0, 0, 0, 0, 0.0,
};

ascanf_Function vars_internal_Functions[3]= {
	{ "DCL?", ascanf_DeclareVariable, AMAXARGS, NOT_EOF_OR_RETURN,
		"DCL?[name] or DCL?[name,expr]: declare (and initialise) <name> if not already existent.\n"
		" Also a hook for internally defined variables."
	},
	{ "$ascanf.escape", ascanf_Variable, 2, _ascanf_variable,
		"$ascanf.escape: internal interrupt variable"
	},
	{ "$ascanf.interrupt", ascanf_Variable, 2, _ascanf_variable,
		"$ascanf.interrupt: prioritary internal interrupt variable"
	},
};
int internal_Functions= sizeof(vars_internal_Functions)/sizeof(ascanf_Function);

/* 20020428: moved the builtin-functions table to ascanfc-table.c . The Makefile generates a corresponding headerfile
 \ that contains all the necessary (= known) callback routines (identified through their use of the ASCB_ARGLIST macro!!)
 */

int check_for_constants_list( Compiled_Form **form )
{
	if( form && *form ){
	  Compiled_Form *f= *form;
	  int ok= 1, N= 0, nn= 0;
		while( f && ok ){
			if( f->fun && f->fun->type== _ascanf_novariable ){
				if( !f->take_address ){
					ok= 0;
				}
			}
			else switch( f->type ){
				default:
				  /* 20020328: I think that addresses should pass the test?!
				   \ 20020410: last_value ('?') ditto
				   */
					if( !(f->take_address || f->last_value) ){
						ok= 0;
					}
					break;
				case _ascanf_value:
					nn+= 1;
					break;
				case _ascanf_simplestats:
				case _ascanf_simpleanglestats:
					if( f->args ){
						if( f->args->args ){
							ok= 0;
						}
						else if( f->args->take_address ){
						  /* 20020612: this should allow us to pass a pointer to an array (to add all the elements) */
							ok= 0;
						}
						else{
							ok= f->args->list_of_constants;
						}
					}
					break;
				case _ascanf_variable:
					if( f->args ){
						ok= (f->args->args)? 0 : f->args->list_of_constants;
					}
					break;
				case _ascanf_python_object:
// 20080916: never accept a python object as a constant
// 					if( f->args || f->empty_arglist )
					{
						ok= 0;
					}
					break;
				case _ascanf_array:{
					if( (f->args && f->alloc_argc>2) || (f->fun->internal && !f->fun->user_internal && f->fun->procedure) ){
					  /* We could accept a value or variable equal to the last_index ... */
					  /* 20001001: internal arrays are to be evaluated at each reference (if they have code associated!)	*/
						ok= 0;
					}
					break;
				}
				if( f->type!= _ascanf_value && f->take_address && (f->take_usage || (f->fun && f->fun->is_usage)) ){
					nn+= 1;
				}
			}
			N+= 1;
			f= f->cdr;
		}
		if( ok ){
			f= *form;
			ok= (nn== N)? -1 : 1;
			while( f ){
				f->list_of_constants= ok;
				f= f->cdr;
			}
		}
		return( (*form)->list_of_constants );
	}
	return(0);
}

void correct_constants_args_list( Compiled_Form *form, int *level, char *file, int line )
{
/* #ifndef DEBUG	*/
/* 	if( ascanf_verbose || debugFlag )	*/
/* #endif	*/
	{ char hdr[128];
		sprintf( hdr, "#C%d:\t", *level );
		fprintf( StdErr, "%snot a valid constants list (corrected) (%s:%d):\n%s", hdr, file, line, hdr );
		Print_Form( StdErr, &(form), 0, True, hdr, NULL, "\n", True );
	}
	form->list_of_constants= 0;
}

int check_for_ascanf_function( int Index, char **s, double *result, int *ok, char *caller,
	Compiled_Form **form, ascanf_Function **ret, int *level
)
{  int i, autoload, autoloads= 2, len, alias_len= 0, hash_len= 0, listSize= ascanf_Functions, nextlistSize= internal_Functions;
   ascanf_Function *af, *nextList= vars_internal_Functions, *funList= vars_ascanf_Functions;
   long hash;
   Compiled_Form *arglist= NULL;
   int shifted= 0;
   Add_Form_Flags flags;
   char *name= *s, *Name, *alias= NULL;
   char *expr= *s;
   Boolean found= False;

	if( ret ){
		*ret= NULL;
	}
	if( reset_ascanf_index_value ){
		*ascanf_index_value= (double) Index;
	}
	if( reset_ascanf_currentself_value ){
		(*ascanf_self_value)= *result;
		(*ascanf_current_value)= *result;
	}
	  /* 20000224: remove leading whitespaces	*/
	while( isspace( (unsigned char) *name) ){
		name++;
	}
	memset( &flags, 0, sizeof(flags) );
	flags.sign= 1;
	while( index( PREFIXES, name[0]) && shifted< 5 && *name && name[1]!= '[' ){
		switch( name[0] ){
		  char errbuf[256];
			case '-':
				(name)++;
				flags.sign*= -1;
				if( flags.negate && pragma_unlikely(ascanf_verbose) ){
					sprintf( errbuf,
						"check_for_ascanf_function() #%d: (-) turns off previous negation (!)\n",
						(*level)
					);
					fputs( errbuf, StdErr ); fflush( StdErr );
					if( ascanf_window && (ascanf_SyntaxCheck || ascanf_PopupWarn) ){
						xtb_error_box( ascanf_window, errbuf, "warning" );
					}
				}
				flags.negate= 0;
				flags.take_address= 0;
				flags.take_usage= 0;
				shifted+= 1;
				break;
			case '!':
				(name)++;
				flags.negate= !flags.negate;
				if( flags.sign< 0 && pragma_unlikely(ascanf_verbose) ){
					sprintf( errbuf,
						"check_for_ascanf_function() #%d: (!) turns off previous negative (-)\n",
						(*level)
					);
					fputs( errbuf, StdErr ); fflush( StdErr );
					if( ascanf_window && (ascanf_SyntaxCheck || ascanf_PopupWarn) ){
						xtb_error_box( ascanf_window, errbuf, "warning" );
					}
				}
				flags.sign= 1;
				flags.take_address= 0;
				flags.take_usage= 0;
				shifted+= 1;
				break;
			case '`':
				flags.take_usage= 1;
			case '&':
				(name)++;
				flags.take_address= 1;
				flags.sign= 1;
				flags.negate= 0;
				shifted+= 1;
				break;
			case '?':
				(name)++;
				flags.last_value= 1;
				flags.take_address= 0;
				flags.take_usage= 0;
				flags.store= 0;
				shifted+= 1;
				break;
			case '*':
				(name)++;
				flags.store= 1;
				shifted+= 1;
				break;
		}
	}
	if( !*name ){
		return(0);
	}
	Name= name;
	  /* 20020510: alias substitution(s) */
/* 	if( name[0]== '@' && name[1]== '[' ){	*/
/* 		alias= concat( "nDindex", &name[1], NULL );	*/
/* 		alias_len= 1;	*/
/* 		name= alias;	*/
/* 	}	*/
	hash= ascanf_hash( name, &hash_len );


	if( vars_local_Functions && *vars_local_Functions && (*vars_local_Functions)->cdr ){
		nextList= funList;
		nextlistSize= listSize;
		funList= *vars_local_Functions;
		  /* vars_local_Functions points to a single instance which (here) has at least 1 list element. */
		listSize= 1;
	}
	for( autoload= 0; autoload< autoloads && !found; autoload++ ){
		if( autoload== autoloads-1 ){
			  /* 20031006: do autoloading after failure to find a match. This is necessary
			   \ to allow for pre/overloading by e.g. a different version of the standard libs.
			   */
			Auto_LoadDyMod( AutoLoadTable, AutoLoads, name );
			nextList= vars_internal_Functions;
			funList= vars_ascanf_Functions;
			listSize= ascanf_Functions;
			nextlistSize= internal_Functions;
		}
		i= 0;
		while( i< listSize ){
			for( i= 0; i< listSize && funList; i++){
				found= False;
				af= &funList[i];
				while( af && af->name && af->function ){
					if( af->name_length ){
						len= af->name_length;
					}
					else{
						len= af->name_length= strlen( parse_codes(af->name) );
						parse_codes( af->usage );
					}
					if( !af->hash ){
						af->hash= ascanf_hash( af->name, NULL );
					}
					  /* 20020510: moved the test for hash_len==8 into the top if(): */
					if( af== af_ArgList && hash_len== 8 ){
						if( strncmp( name, "$ArgList", 8 )== 0 ){
							/* if( hash_len== 8 ) */{
								len= hash_len;
								found= True;
							}
						}
					}
					else if( af->function==ascanf_nDindex && hash_len== 1 ){
						if( name[0]== '@' ){
							len= hash_len;
							found= True;
						}
					}
					if( hash== af->hash || found ){
						  /* 20000512: I think strlen(name) should constrain the comparison..
						   \ Thus, compare over hash_len instead of over len.
						   \ 20010605: new: skip over deleted variables!
						   */
						if(
							!(af->type== _ascanf_novariable && (af->function== ascanf_Variable || af->function== ascanf_Procedure )) &&
							(!strncmp( name, af->name, hash_len ) || found)
						){
						  int ss= af->store, adr= af->take_address;
							found= True;
							if( name== alias ){
								  /* Restore the <name> that we received in the argument and that was replaced
								   \ by a local buffer with an alias:
								   */
								name= Name;
								  /* restore the length of the pattern that was replaced by the alias: */
								len= alias_len;
								xfree( alias );
							}
							if( shifted ){
								(*s)= &((*s)[shifted]);
							}
							af->store= flags.store;
#ifdef ALWAYS_CHECK_LINKEDARRAYS
							if( af->linkedArray.dataColumn ){
								Check_linkedArray(af);
							}
#endif
							if( (af->function== ascanf_Variable || af->function== ascanf_Procedure) &&
								(af->type== _ascanf_variable || af->type== _ascanf_array || af->type== _ascanf_procedure ) &&
								((af->internal && !af->user_internal) && (af->is_usage || af->is_address) )
							){
								flags.take_usage= af->is_usage;
								flags.take_address= True;
								flags.sign= 1;
								flags.negate= 0;
								  /* update the elements of this automatic array:	*/
								if( af->procedure && af->type== _ascanf_array && !form &&
									!AlwaysUpdateAutoArrays && af->procedure->list_of_constants>= 0
								){
								  int n= af->N;
									if( pragma_unlikely(ascanf_verbose>1) ){
										fprintf( StdErr, "#%s%d: updating automatic array \"%s\", expression %s\n",
											(form)? "ac:" : "", (*level), af->name, af->procedure->expr
										);
									}
									af->procedure->level+= 1;
									_compiled_fascanf( &n, af->procedure->expr, af->array, NULL, NULL, NULL, &af->procedure, level );
									af->procedure->level-= 1;
								}
							}
							if( (af->function== ascanf_Variable || af->function== ascanf_Procedure) &&
								(af->type== _ascanf_variable || af->type== _ascanf_array || af->type== _ascanf_procedure ) &&
								(flags.take_usage /* || af->is_usage */)
							){
								af->take_usage= True;
							}
							else{
								if( flags.take_usage ){
									if( arglist || ascanf_SyntaxCheck ){
									  char errbuf[256];
										sprintf( errbuf,
											"ascanf_function() #%d: is-string qualifier (`) only allowed for variables and arrays.",
											(*level)
										);
										fprintf( StdErr, "%s: %s\n", errbuf, expr );
										if( ascanf_window && ascanf_PopupWarn ){
											xtb_error_box( ascanf_window, errbuf, expr );
										}
									}
									flags.take_address= 0;
								}
								af->take_usage= flags.take_usage= 0;
							}
							af->take_address= flags.take_address;
							if( form ){
								if( flags.last_value || af->function== ascanf_Variable || af->function== ascanf_Procedure ){
									if( flags.last_value ||
										af->type== _ascanf_variable || af->type== _ascanf_array || af->type== _ascanf_procedure ||
										af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats
										|| af->type== _ascanf_python_object
									){
										*ok= 1;
										if( flags.last_value ){
											*ok= 1;
											*result= af->value;
										}
										else if( !flags.take_address ){
											*ok= ascanf_function( af, Index, s, len, result, caller, &arglist, level );
											if( ((void*)arglist)==((void*)-1) ){
												flags.empty_arglist= True;
												arglist= NULL;
											}
											else if( arglist && arglist->empty_arglist<0 ){
												flags.empty_arglist= True;
												xfree(arglist);
											}
										}
										else{
											  /* 20050125: also set af->value to the pointer. This is not perfectly consistent
											   \ with normal runtime behaviour, but we are compiling and NOT evaluating. So
											   \ af->value doesn't represent anything accurate anyway.
											   */
											*ok= (*result= af->value= take_ascanf_address(af))? 1 : 0;
											if( pragma_unlikely( ascanf_verbose> 1 ) ){
												fprintf( StdErr, " %c\"%s\": address taken (or not), evaluation of expression omitted (%d)!\n",
													(af->take_usage)? '`' : '&',
													name, __LINE__
												);
											}
										}
#ifdef DEBUG
										Add_Form( form, af->type, &flags, af->value, expr, af, arglist );
#else
										Add_Form( form, af->type, &flags, af->value, NULL, af, arglist );
#endif
										if( flags.negate ){
											*result= (*result && !NaN(*result))? 0 : 1;
										}
										else{
											*result*= flags.sign;
										}
									}
									else{
										*ok= 0;
										fprintf( StdErr, "\"%s\" matches deleted variable/procedure \"%s\"!\n",
											name, af->name
										);
									}
								}
								else{
									if( !flags.take_address ){
										*ok= ascanf_function( af, Index, s, len, result, caller, &arglist, level);
										if( ((void*)arglist)==((void*)-1) ){
											flags.empty_arglist= True;
											arglist= NULL;
										}
										else if( arglist && arglist->empty_arglist<0 ){
											flags.empty_arglist= True;
											xfree(arglist);
										}
									}
									else{
										  /* 20050125: see above. */
										*ok= (*result= af->value= take_ascanf_address(af))? 1 : 0;
										if( pragma_unlikely( ascanf_verbose> 1 ) ){
											fprintf( StdErr, " %c\"%s\": address taken (or not), evaluation of expression omitted (%d)!\n",
												(af->take_usage)? '`' : '&',
												name, __LINE__
											);
										}
									}
#ifdef DEBUG
									Add_Form( form, _ascanf_function, &flags, *result, expr, af, arglist );
#else
									Add_Form( form, _ascanf_function, &flags, *result, NULL, af, arglist );
#endif
									if( flags.negate ){
										*result= (*result && !NaN(*result))? 0 : 1;
									}
									else{
										*result*= flags.sign;
									}
								}
								if( *form && (*form)->last_cdr ){
									(*form)->last_cdr->ok= *ok;
								}
							}
							else{
								if( flags.last_value ){
									*ok= 1;
									*result= af->value;
								}
								else if( flags.take_address ){
									*ok= (*result= take_ascanf_address(af))? 1 : 0;
									if( pragma_unlikely( ascanf_verbose> 1 ) ){
										fprintf( StdErr, " %c\"%s\": address taken (or not), evaluation of expression omitted (%d)!\n",
											(af->take_usage)? '`' : '&',
											name, __LINE__
										);
									}
								}
								else if( af->function== ascanf_Variable || af->function== ascanf_Procedure ){
									if( af->type== _ascanf_variable || af->type== _ascanf_array || af->type== _ascanf_procedure ||
										af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats
										|| af->type== _ascanf_python_object
									){
									  /* It may seem weird to call ascanf_function for a variable, but there can
									   \ be an assignment to it!
									   */
										*ok= ascanf_function( af, Index, s, len, result, caller, NULL, level );
									}
									else if( pragma_unlikely(ascanf_verbose) ){
										*ok= 0;
										fprintf( StdErr, "\"%s\" matches deleted variable/procedure \"%s\"!\n",
											name, af->name
										);
									}
								}
								else{
									*ok= ascanf_function( af, Index, s, len, result, caller, NULL, level );
								}
								if( flags.negate ){
									*result= (*result && !NaN(*result))? 0 : 1;
								}
								else{
									*result*= flags.sign;
								}
							}
							af->store= ss;
							af->take_address= adr;
							if( !reset_ascanf_index_value && !(*level) ){
								(*ascanf_index_value)+= 1;
							}
							if( ret ){
								*ret= af;
							}

							if( af->function!= ascanf_DeclareVariable && af->function!= ascanf_DeclareProcedure &&
								af->function!= ascanf_DeleteVariable
							){
							  /* 20010205: only the above functions should be able to pass 0 through us;
							   \ all other should not (except when ok==EOF?), to avoid attempts to create
							   \ an automatic variable with the same name...
							   */
								if( ! *ok ){
									*ok= (EOF== -1)? -2 : -1;
								}
							}
							switch(af->type){
								case NOT_EOF:
									return( (*ok!= EOF) );
									break;
								case _ascanf_python_object:
								case NOT_EOF_OR_RETURN:
									return( (*ok== EOF)? 0 : *ok );
									break;
								default:
									return( *ok);
									break;
							}
						}
						else if( pragma_unlikely(ascanf_verbose) ){
							if( !strncmp( name, af->name, len ) ){
								fprintf( StdErr, "(\"%s\"==\"%s\" over %d chars)", name, af->name, len );
							}
						}
					}
#ifdef DEBUG_EXTRA
					else{
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "!%s(%lx,%lx)", af->name, hash, af->name );
							fflush( StdErr );
						}
					}
#endif
					af= af->cdr;
				}
				if( i== listSize- 1 && nextList ){
					i= -1;
					listSize= nextlistSize;
					funList= nextList;
					if( nextList== vars_ascanf_Functions ){
						nextList= vars_internal_Functions;
						nextlistSize= internal_Functions;
					}
					else{
						nextList= NULL;
					}
				}
			}
		}
	}
	if( !reset_ascanf_index_value && !(*level) ){
		(*ascanf_index_value)+= 1;
	}
	return(0);
}

ascanf_Function *find_ascanf_function( char *name, double *result, int *ok, char *caller )
{ int Ok;
  ascanf_Function *ret= NULL;
  int level= -1;
	if( !ok ){
		ok= &Ok;
	}
	check_for_ascanf_function( 0, &name, result, ok, caller,
		NULL, &ret, &level
	);
	return( ret );
}

static void saf_header( FILE *fp, char *prefix )
{ char standout[2]= {0,0};
	if( prefix[0]!= '\t' ){
		standout[0]= 0x01;
	}
	fprintf( fp, "%s%s*** ascanf functions (functionname:\t usage\t [max-args]\t (name-length)) or\n", standout, prefix);
	fprintf( fp, "%s%s                     (variablename:\t name=value\t A=assigns \t R=reads\t [max-args]\t (name-length)) or\n", standout, prefix);
	fprintf( fp, "%s%s                     (   arrayname:\t name[len]={values}[last_index]= last_value\t [max-args]\t (name-length) [index=-1: query length]):\n", standout, prefix);
	fprintf( fp, "%s%s                     (   arrayvalues and procedurecode shown only in search mode (SHelp[name])):\n", standout, prefix);
}

char *strcasestr( const char *a,  const char *b)
{ unsigned int len= strlen(b), lena= strlen(a);
  int nomatch= 0, n= len;

	while( ( nomatch= (strncasecmp(a, b, len)) ) && n< lena ){
		a++;
		n+= 1;
	}
	return( (nomatch)? NULL : (char*) a );
}

int show_ascanf_functions( FILE *fp, char *prefix, int do_bold, int lines )
{  int i, l, searching= (ascanf_var_search || ascanf_usage_search || ascanf_label_search);
   int listSize= (*Allocate_Internal)? internal_Functions : ascanf_Functions;
   ascanf_Function *funList= (*Allocate_Internal)? vars_internal_Functions : vars_ascanf_Functions;
   ascanf_Function *af;
   char *bold="", *nobold="", *underline="", *nounderline="", *TERM=cgetenv("TERM");
   int hit= 0, avs_len= (ascanf_var_search)? strlen(ascanf_var_search) : 0;
#ifdef _UNIX_C_
	if( TERM && do_bold ){
		if( strncaseeq( TERM, "xterm", 5) || strncaseeq( TERM, "vt1", 3) ||
			strncaseeq( TERM, "vt2", 3) || strncaseeq( TERM, "cygwin", 6)
		){
			bold= "\033[1m";
			nobold= "\033[m";
			underline= "\033[4m";
			nounderline= nobold;
		}
		else if( strncaseeq( TERM, "hp", 2) ){
			bold= "\033&dB";
			nobold= "\033&d@";
			underline= "\033&dD";
			nounderline= nobold;
		}
	}
#endif
	if( !fp){
		return(0);
	}
	if( !searching ){
		saf_header( fp, prefix );
	}
	for( l= 0, i= 0; i< listSize && funList; i++){
		af= &funList[i];
		while( af){
		  Boolean found= False;
			if( ascanf_var_search ){
				if( !(found= (strncasecmp( af->name, ascanf_var_search, avs_len)== 0)) && af->usage ){
					found= (strncasecmp( af->usage, ascanf_var_search, avs_len)== 0);
				}
			}
			if( !found && ascanf_usage_search && af->usage ){
				found= (strcasestr( af->usage, ascanf_usage_search)!= NULL);
			}
			if( !found && ascanf_label_search && af->label ){
				found= (strcasestr( af->label, ascanf_label_search)!= NULL);
			}
			if( !searching || found ){
				if( searching && !hit ){
					saf_header( fp, prefix );
				}
				af->name_length= strlen( af->name );
				af->hash= ascanf_hash( af->name, NULL );
				if( !do_bold ){
					if( prefix[0]!= '\t' && !searching ){
					  /* When making a complete list, list some important items in standout	*/
						if( strcmp( af->name, "DCL")== 0 || strcmp( af->name, "DEPROC")== 0 || af!= &funList[i] ){
							fputc( 0x01, fp );
						}
					}
					fprintf( fp, " %d[%s]:", i, af->name );
				}
				if( strcmp( af->name, "$VariableLabel")== 0 ){
					fprintf( fp, "%s%s%s", prefix, bold, af->name);
					if( af->usage ){
					  char *c= &af->usage[ strlen(af->usage)-1 ];
						if( *c== '\n' ){
							*c= '\0';
						}
						else{
							c= NULL;
						}
						fprintf( fp, "=\"%s\"", af->usage);
						if( c ){
							*c= '\n';
						}
					}
					fprintf( fp, " \"This variable can be used to label newly created variables with a string\n"
							" such that they can be deleted with Delete[$Label=string].\n"
							" NB: this information is not exported!"
					);
				}
				else if( af->usage ){
				  char *c= &af->usage[ strlen(af->usage)-1 ];
					if( *c== '\n' ){
						*c= '\0';
					}
					else{
						c= NULL;
					}
					fprintf( fp, "%s%s", prefix, bold );
					if( strncmp( af->name, af->usage, strlen(af->name) ) ){
						fprintf( fp, "[%s]: ", af->name );
					}
					fprintf( fp, "\"%s\" ", af->usage );
					if( c ){
						*c= '\n';
					}
				}
				else{
					fprintf( fp, "%s%s%s", prefix, bold, af->name);
				}
				if( strcmp( af->name, "SETMXYZ")== 0 || strcmp( af->name, "MXYZ")== 0 ){
					fprintf( fp, " (%d x %d x %d) ", ascanf_mxyz_X, ascanf_mxyz_Y, ascanf_mxyz_Z );
				}
				if( strcmp( af->name, "SETMXY")== 0 || strcmp( af->name, "MXY")== 0 ){
					fprintf( fp, " (%d x %d) ", ascanf_mxy_X, ascanf_mxy_Y );
				}
				if( af->function== ascanf_Variable ){
					if( af->type== _ascanf_variable ){
					  ascanf_Function *pf;
						  /* 20020330: outcommented; appended space to usage printing above. */
/* 						if( af->usage ){	*/
/* 							fputc( ' ', fp );	*/
/* 						}	*/
						if( af->is_address ){
						  int auaa= AlwaysUpdateAutoArrays;
							AlwaysUpdateAutoArrays= False;
							pf= parse_ascanf_address( af->value, 0, "show_ascanf_variables", (int) ascanf_verbose, NULL );
							AlwaysUpdateAutoArrays= auaa;
						}
						else{
							pf= NULL;
						}
						fprintf( fp, "==%s", (pf)? pf->name : ad2str( af->value, d3str_format, NULL));
					}
					else if( af->type== _ascanf_array ){
					  int j;
#ifdef ALWAYS_CHECK_LINKEDARRAYS
						if( af->linkedArray.dataColumn ){
							Check_linkedArray(af);
						}
#endif
						if( !searching ){
							fprintf( fp, "[%d]= {values hidden", af->N );
						}
						else{
							if( af->iarray ){
								fprintf( fp, "[%d]= {%d", af->N, af->iarray[0] );
								for( j= 1; j< af->N && af->iarray && (af->last_index== -1 || j< 48) ; j++ ){
									fprintf( fp, ",%d", af->iarray[j] );
								}
								if( af->N> j ){
									fprintf( fp, ",..." );
									for( j= af->N- 3; j< af->N; j++ ){
										fprintf( fp, ",%d", af->iarray[j] );
									}
								}
							}
							else{
								fprintf( fp, "[%d]= {%s", af->N, ad2str( af->array[0], d3str_format, NULL) );
								for( j= 1; j< af->N && af->array && (af->last_index== -1 || j< 48) ; j++ ){
									fprintf( fp, ",%s", ad2str( af->array[j], d3str_format, NULL));
								}
								if( af->N> j ){
									fprintf( fp, ",..." );
									for( j= af->N- 3; j< af->N; j++ ){
										fprintf( fp, ",%s", ad2str( af->array[j], d3str_format, NULL));
									}
								}
							}
						}
						fprintf( fp, "}[%d]== %s", af->last_index, ad2str( af->value, d3str_format, NULL) );
					}
					else if( af->type== _ascanf_simplestats ){
					  char buf[256];
						if( af->N> 1 ){
							if( af->last_index>= 0 && af->last_index< af->N ){
								fprintf( fp, "[%d]", af->last_index );
								SS_sprint_full( buf, d3str_format, " #xb1 ", 0, &af->SS[af->last_index] );
							}
							else if( af->last_index== -1 ){
								sprintf( buf, "%d", af->N );
							}
						}
						else{
							SS_sprint_full( buf, d3str_format, " #xb1 ", 0, af->SS );
						}
						fprintf( fp, "==%s == %s", buf, ad2str( af->value, d3str_format, NULL) );
					}
					else if( af->type== _ascanf_simpleanglestats ){
					  char buf[256];
						if( af->N> 1 ){
							if( af->last_index>= 0 && af->last_index< af->N ){
								fprintf( fp, "[%d]", af->last_index );
								SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, &af->SAS[af->last_index] );
							}
							else if( af->last_index== -1 ){
								sprintf( buf, "%d", af->N );
							}
						}
						else{
							SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, af->SAS );
						}
						fprintf( fp, "==%s == %s", buf, ad2str( af->value, d3str_format, NULL) );
					}
					else if( af->type== _ascanf_novariable ){
						fprintf( fp, "=%s DELETED",
							ad2str( af->value, d3str_format, NULL)
						);
					}
					if( af->type!= _ascanf_novariable ){
						fprintf( fp, " A=%d R=%d L=%d", af->assigns, af->reads, af->links);
					}
				}
				else if( af->function== ascanf_Procedure ){
					if( af->type== _ascanf_procedure ){
					  char *c= af->procedure->expr;
						if( af->usage ){
							fputc( ' ', fp );
						}
						if( !searching ){
							fprintf( fp, "{code hidden" );
						}
						else{
							fputc( '{', fp );
							while( c && *c ){
/*
								if( *c== '\n' ){
									fputs( "\\\\n", fp );
								}
								else
 */
								if( *c== '\t' ){
									fputc( ' ', fp );
								}
								else{
									fputc( *c, fp );
								}
								c++;
							}
						}
						fprintf( fp, "}=%s %sA=%d R=%d L=%d",
							ad2str( af->value, d3str_format, NULL),
							(af->dialog_procedure)? "(dialog) " : "",
							af->assigns, af->reads, af->links
						);
					}
					else{
						fprintf( fp, "=%s DELETED",
							ad2str( af->value, d3str_format, NULL)
						);
					}
				}
				else{
					fprintf( fp, " C=%d L=%d", af->reads, af->links );
				}
				// 20120413: why was this 'margs=%d' ?!
				fprintf( fp, "%s\t [%sNargs=%d%s]\t (%d)", nobold,
					underline, af->Nargs, nounderline, af->name_length
				);
				if( af->type==_ascanf_array && af->linkedArray.dataColumn ){
					if( !do_bold ){
						fputc( '\n', fp );
					}
					fprintf( fp, "  # array pointing to set #%d, column %d", af->linkedArray.set_nr, af->linkedArray.col_nr );
				}
				if( af->dymod && af->dymod->name && af->dymod->path ){
					if( !do_bold ){
						fputc( '\n', fp );
					}
					fprintf( fp, "  # Module={\"%s\",%s}", af->dymod->name, af->dymod->path );
				}
				if( af->fp ){
					if( !do_bold ){
						fputc( '\n', fp );
					}
					fprintf( fp, "  # open file fd=%d", fileno(af->fp) );
				}
				if( af->type== _ascanf_python_object && af->PyObject ){
					if( !do_bold ){
						fputc( '\n', fp );
					}
					fprintf( fp, "  # PyObject %p \"%s\"", af->PyObject, (af->PyObject_Name)? af->PyObject_Name : "?" );
				}
				if( af->cfont ){
					if( !do_bold ){
						fputc( '\n', fp );
					}
					fprintf( fp, "  # cfont: %s\"%s\"/%s[%g]",
						(af->cfont->is_alt_XFont)? "(alt.) " : "",
						af->cfont->XFont.name, af->cfont->PSFont, af->cfont->PSPointSize
					);
				}
				if( af->label ){
					if( !do_bold ){
						fputc( '\n', fp );
					}
					fprintf( fp, "  # label={%s}", af->label );
				}
				if( af->accessHandler ){
				  ascanf_Function *aH= af->accessHandler;
				  double *aH_par= af->aH_par;
					if( !do_bold ){
						fputc( '\n', fp );
					}
					switch( aH->type ){
						case _ascanf_variable:
							fprintf( fp, "  # AccessHandler: var %s=%g",
								aH->name, aH_par[0]
							);
							break;
						case _ascanf_array:
							fprintf( fp, "  # AccessHandler: array %s[%g]=%g",
								aH->name, aH_par[0], aH_par[1]
							);
							break;
						case _ascanf_procedure:{
							fprintf( fp, "  # AccessHandler: call proc %s", aH->name);
							break;
						}
						case NOT_EOF:
						case NOT_EOF_OR_RETURN:
						case _ascanf_python_object:
						case _ascanf_function:{
							fprintf( fp, "  # AccessHandler: call %s[%g,%g]",
								aH->name, aH_par[0], aH_par[1]
							);
							break;
						}
						default:
							fprintf( fp, "  # AccessHandler %s: obsolete/invalid, removed", aH->name);
							af->accessHandler= NULL;
							break;
					}
				}
				if( l++ % lines== 0 ){
					fputc( '\n', fp );
				}
				else{
					fprintf( fp, "%s || ", prefix );
				}
				hit+= 1;
			}
			af= af->cdr;
		}
	}
	fprintf( fp, "%s%s", nobold, nounderline);
	if( !hit ){
		if( ascanf_var_search ){
			fprintf( fp, "\"%s\": no such function, variable or procedure!\n", ascanf_var_search );
		}
		else if( searching ){
			fprintf( fp, "\"%s\": no match!\n", ascanf_usage_search );
		}
	}
	fflush(fp);
	return(hit);
}

int add_ascanf_functions( ascanf_Function *array, int n, char *caller)
{  int i, ni /*, pi= n-1 */;
   ascanf_Function *af= &vars_ascanf_Functions[ascanf_Functions-1];
	if( !array ){
		fprintf( StdErr, "ascanfc::add_ascanf_functions(): called with NULL array by \"%s\"; n==%d\n", caller, n );
		return(0);
	}
	else if( n<= 0 ){
		return(0);
	}
	for( i= n-2; i>= 0; i--){
		if( array[i].name ){
			if( index( array[i].name, '[') || index( array[i].name, ']') ){
				fprintf( StdErr, "!!\n!! Warning: entry %d name \"%s\" contains invalid characters!\n", i, array[i].name );
			}
			  /* find next element with non-null name: */
			ni= i+1;
			while( !array[ni].name && ni< n-2 ){
				ni+= 1;
			}
			if( array[ni].name ){
				array[i].cdr= &array[ni];
				array[ni].car= &array[i];
				if( i== n-2 ){
					array[ni].cdr= NULL;
				}
/* 				if( i && pi!= ni ){	*/
/* 					array[i].car= &array[pi];	*/
/* 					pi= i;	*/
/* 				}	*/
			}
		}
		else{
			fprintf( StdErr, "add_ascanf_functions(\"%s\"): element %d has no name: entry ignored!\n",
				caller, i
			);
			array[i].car= array[i].cdr= NULL;
		}
// 		register_VariableName( &array[i] );
		take_ascanf_address( &array[i] );
	}
#ifdef CAR_NEW_FUNCTIONS
	array[n-1].cdr= af->cdr;
	af->car= &array[n-1];
	af->cdr= array;
	array->car= af;
#else
	  /* Find the end of the current list: 	*/
	while( af && af->cdr ){
			if( !af->special_fun ){
			  /* correct the ascanf_Function structure if necessary. This may be the case for
			   \ afctions loaded via Dynamic Modules...
			   */
				if( af->function== ascanf_Variable || af->function== ascanf_Procedure ){
					af->special_fun= not_a_function;
				}
			}
		af= af->cdr;
	}
	array->car= af;
	af->cdr= array;
#endif
	return( n);
}

int add_ascanf_functions_with_autoload( ascanf_Function *array, int n, char* libname, char *caller)
{ int i, m;
	if( (m= add_ascanf_functions( array, n, caller ))== n ){
	  DyModAutoLoadTables *new;
		if( (new = (DyModAutoLoadTables*) calloc( n, sizeof(DyModAutoLoadTables) )) ){
			for( i= 0; i< n; i++ ){
				new[i].functionName= array[i].name;
				new[i].DyModName= libname;
				if( strcasestr( libname, "Python" ) ){
					new[i].flags= RTLD_NOW|RTLD_GLOBAL;
				}
				else{
					new[i].flags= RTLD_LAZY|RTLD_GLOBAL;
				}
			}
			if( !(AutoLoadTable= Add_LoadDyMod( AutoLoadTable, &AutoLoads, new, n )) ){
				m= 0;
			}
			xfree(new);
		}
		else{
			fprintf( StdErr,
				"add_ascanf_functions_with_autoload(\"%s\"): failure adding %d functions from %s to the AutoLoad table (%s)\n",
				caller, n, libname
			);
			m = 0;
		}
	}
	return(m);
}

/* Remove the previously attached ascanf_Function *array. When the <force> argument
 \ is 0, don't perform any checks; otherwise, verify if all elements to be removed have a 0 link count
 \ (= they're not used).
 */
int remove_ascanf_functions( ascanf_Function *array, int n, int force )
{  int i, N= 0;
   ascanf_Function *af;
   // even if we don't currently use the registry, we should not let removed variables linger in it!
   int use_VariableNamesRegistry = register_VariableNames(1);
	ascanf_emsg= NULL;
	for( i= 0; i< n; i++){
		af= &vars_ascanf_Functions[ascanf_Functions-1];
		{ int nolinks= (af->links==0 && !af->PyAOself.self)? True : False;
			while( af->cdr && nolinks ){
				if( af->cdr== &array[i] ){
				  ascanf_Function *fl= af->cdr;
				  unsigned long nn= (fl- array);
				  int ok= ((nn< n && fl->links==0 && !af->PyAOself.self) || nn>= n)? True : False;
					while( fl && ok ){
						if( (fl= fl->cdr) ){
							nn= fl- array;
							if( (nn< n && (fl->links || af->PyAOself.self)) ){
								ok= False;
							}
						}
					}
					if( !ok ){
						nolinks= False;
					}
					af= af->cdr;
				}
				else{
					af= af->cdr;
				}
			}
			if( !force ){
				if( !nolinks ){
					N = -1;
					goto bail;
				}
			}
			else{
				if( !nolinks ){
				  static char emsg[128];
					sprintf( emsg, "Warning: removing %d entries of which at least one is in use!", n );
					if( !StringCheck( emsg, sizeof(emsg), __FILE__, __LINE__ ) ){
						ascanf_emsg= emsg;
					}
				}
			}
		}
		af= &vars_ascanf_Functions[ascanf_Functions-1];
		{ double rval= drand();
		  unsigned long loop= 0;
		  ascanf_Function *caf= af;
#if 0
#	ifdef USE_AA_REGISTER
			delete_ascanf_Address(AAF_REPRESENTATION(af));
#	endif
			delete_VariableName(af->name);
#else
// 20101021: it's array[i] that should be removed, not (necessarily) af!!
#	ifdef USE_AA_REGISTER
			delete_ascanf_Address(AAF_REPRESENTATION((&array[i])));
#	endif
			delete_VariableName(array[i].name);
#endif
			while( af->cdr ){
				if( af!= caf ){
					if( af->aH_par[0]== rval && debugFlag ){
						fprintf( StdErr, "[removing #%d\"%s\": \"%s\" already checked (loop %d?!)]",
							i, array[i].name, af->name, loop
						);
						fflush( StdErr );
					}
					af->aH_par[0]= rval;
				}
				if( af->cdr== &array[i] ){
					if( array[i].PyAOself.selfaf ){
						*array[i].PyAOself.selfaf = NULL;
					}
					array[i].PyAOself.self = NULL;
					af->cdr= af->cdr->cdr;
					if( af->cdr ){
						af->cdr->car= af;
					}
					N+= 1;
				}
				else{
					af= af->cdr;
				}
				loop+= 1;
			}
		}
	}
bail:
	// reset the registry usage state, and return:
	register_VariableNames( use_VariableNamesRegistry );
	return( N);
}

int Copy_preExisting_Variable_and_Delete( ascanf_Function *af, char *label )
{ int ret = 0;
	if( af->type == _ascanf_variable ){
	  ascanf_Function *defined = get_VariableWithName(af->name, True);
		if( defined ){
			af->value = defined->value;
			if( ascanf_verbose || debugFlag || scriptVerbose ){
				fprintf( StdErr, "%s: setting %s=%s from pre-existing %s\n",
					label, af->name, ad2str( af->value, d3str_format, 0), defined->name
				);
			}
			if( defined->internal ){
				Delete_Internal_Variable( NULL, defined );
			}
			else{
				Delete_Variable(defined);
			}
			ret = 1;
		}
	}
	return( ret );
}

extern double *param_scratch;
extern int param_scratch_len, param_scratch_inUse;

int ascanf_AutoCreate= True, ascanf_AutoVarCreate= True, ascanf_AutoStringCreate= True;
int ascanf_AutoVarWouldCreate_msg= True;

int Create_AutoVariable( char *s, double *A, int idx, int *N, Compiled_Form **form, ascanf_type *type, int *level, ascanf_Function **allocated_return, int verbose )
{ char *Buf, *buf= NULL, *b= s, *nname, *last= &s[ strlen(s)-1 ], last_sp= 0, sep[2];
  int r= 0, instring, AC= AutoCreating, SC= ascanf_SyntaxCheck;
  static char active= 0;
    /* 20020430: we can call ourselves recursively. To be sure that the calling invocation sees the Allocated_Internal
	 \ that it expects, we will have to save and restore that (global) variable's current value!!
	 */
  ascanf_Function *pAI= Allocated_Internal;
  char *fur= fascanf_unparsed_remaining;
  int CrVar= False, CrString= False;

	if( !ascanf_AutoCreate ){
		return(0);
	}

	AutoCreating= True;
	ascanf_SyntaxCheck*= -1;

	Allocated_Internal= NULL;
	instring= (*b== '"')? True : False;
	sep[0]= ascanf_separator;
	sep[1]= '\0';
	while( *b && (instring || (*b!= '[' && *b!= '{')) ){
		b++;
		if( *b== '"'){
			instring= !instring;
		}
	}
	if( isspace(*last) ){
		while( isspace(*last) ){
			last--;
		}
		  /* 20010604: cut off all trailing whitespace, but restore it afterwards! Thus,
		   \ we need to not change <last> after this point...
		   */
		if( pragma_unlikely(ascanf_verbose>1) ){
			fprintf( StdErr, "#%s%d Create_AutoVariable(%s): removed trailing whitespace\n", (form)? "ac" : "", (*level), s );
		}
		last_sp= last[1];
		last[1]= '\0';
	}
	if( *b== '[' && b!= s ){ if( ascanf_AutoVarCreate ){
	  char *l= s, *expr= NULL;
	  /* A parameter list behind a label becomes the 2nd argument to a
	   \ call to DCL[]
	   */
		CrVar= True;
		*b= '\0';
		nname= XGstrdup(s);
		*b= ascanf_separator;
		if( l[0]== '&' || l[0]== '`' ){
			 /* remove superfluous address-off operator */
			l++;
		}
		if( strncmp( l, "\\l\\", 3)== 0 ){
			expr= &l[3];
		}
		else if( strncmp( l, "lambda", 6)==0 ){
			expr= &l[6];
		}
		else if( strncmp( l, "SUBPROC", 7)==0 ){
			expr= &l[7];
		}
		if( expr && expr[0]== ascanf_separator ){
		  char *b, name[64];
		  static unsigned long serial= 1;
		  double AA= *A;
		  int jj= *N, AlInt= *Allocate_Internal;
			if( serial== 0 ){
				  /* very unlikely to ever occur, but warn in case of a wraparound of the serial number counter! */
				fprintf( StdErr,
					"Warning: automatic procedure (lambda expression) serial wraparound in Create_AutoVariable(): \n"
					"risc of name collision.\n"
				);
			}
			sprintf( name, "\\l\\expr-%d-%lu", *level, serial++ );
			b= buf= concat( "DEPROC-noEval[", name, expr, NULL );
			pAllocate_Internal= *Allocate_Internal;
			*Allocate_Internal= -1;
			if( ascanf_verbose> 1 && AlInt<= 0 ){
				fprintf( StdErr, "#%s%d lambda expression \"%s\" => creating temporary procedure with \"%s\"\n",
					(form)? "ac" : "", (*level), expr, (buf)? buf : serror()
				);
			}
			if( buf && !check_for_ascanf_function( idx, &buf, &AA, &jj, "fascanf", NULL, NULL, level ) ){
				fprintf( StdErr,
					"#%s%d lambda expression \"%s\" => could not create temporary procedure with \"%s\" (%s)\n",
					(form)? "ac" : "", (*level), expr, buf, serror()
				);
			}
			else{
				*Allocate_Internal= AlInt;
				Allocated_Internal->user_internal= False;
				Allocated_Internal->is_usage= False;
				Allocated_Internal->take_usage= False;
				Allocated_Internal->is_address= True;
				Allocated_Internal->take_address= True;
				*A= Allocated_Internal->value= take_ascanf_address( Allocated_Internal );
				if( !form ){
					  /* It should be possible to implement some form of garbage collection by checking the sign
					   \ of the internal field: autovars allocated for a non-compiled expression have a -1 sign.
					   \ This is not currently implemented.
					   */
					Allocated_Internal->internal= -1;
				}
			}
			xfree( b );
		}
		else{
			buf= concat( "DCL[", s, NULL );
		}
	} }
	else{
		if( ascanf_AutoStringCreate && (s[0]== '"' || (s[0]== '`' && s[1]== '"')) && *last== '"' ){
		  double AA= *A;
		  int jj= *N, AlInt= *Allocate_Internal;
		  char *b, *val= (s[0]== '"')? s : &s[1], serial[32];
#if ASCANF_AUTOVARS_UNIQUE /* == 2 */
		  static unsigned long serialnr= 1;
				if( serialnr== 0 ){
					  /* very unlikely to ever occur, but warn in case of a wraparound of the serial number counter! */
					fprintf( StdErr,
						"Warning: automatic string serial wraparound in Create_AutoVariable(): risk of name collision.\n"
					);
				}
				sprintf( serial, "%lu", serialnr++ );
#else
				serial[0]= '\0';
#endif
			CrString= True;
/* 			b= buf= concat( "DCL[", val, serial, sep, "0", sep, val, "]", NULL );	*/
			  /* 20040304: we have to be careful creating autostrings using a DCL[] statement. There should be no
			   \  possibility that the stringcontents get parsed (and it turns out this *can* happen in some cases,
			   \ causing crashes). Thus, we create a variable
			   \ 1) using a hopefully unique, temporary name (obtained with hash64()) and serial
			   \	(NB: outcommented ASCANF_AUTOVARS_UNIQUE==2 above!)
			   \ 2) that will "receive" its usage string "manually", i.e. below, not through the parser.
			   */
			{ char nbuf[512];
			  long long h64;
			  extern long long hash64();
				sprintf( nbuf, "\"Xx%s%llx\"", serial, (h64= hash64(val,NULL)) );
 				b= buf= concat( "DCL[", nbuf, sep, "0", "]", NULL );
			}
			pAllocate_Internal= *Allocate_Internal;
			*Allocate_Internal= -1;
			if( ascanf_verbose> 1 && AlInt<= 0 ){
				fprintf( StdErr, "#%s%d string parameter \"%s\" => creating temporary string variable with \"%s\"\n",
					(form)? "ac" : "", (*level), s, (buf)? buf : serror()
				);
			}
			else{
				 /* 20020508: the feedback that prints \"string\"== string is not really useful! */
				verbose= False;
			}
			if( buf && !check_for_ascanf_function( idx, &buf, &AA, &jj, "fascanf", NULL, NULL, level ) ){
				fprintf( StdErr,
					"#%s%d string parameter \"%s\" => could not create temporary string variable with \"%s\" (%s)\n",
					(form)? "ac" : "", (*level), s, buf, serror()
				);
			}
			else{
				*Allocate_Internal= AlInt;
					  /* 20040304: now install the correct name and the correct usage string: */
					  /* 20070629: remove/replace the entry from the symbol table... */
					delete_VariableName( Allocated_Internal->name );
					xfree( Allocated_Internal->name );
					Allocated_Internal->name= strdup(val);
					Allocated_Internal->hash= ascanf_hash( Allocated_Internal->name, NULL );
					register_VariableName( Allocated_Internal );
					xfree( Allocated_Internal->usage );
					  /* make another copy, of the contents (stripping the lead quote) */
					Allocated_Internal->usage= strdup_unquote_string( &val[1] );
					  /* Remove the trailing quote (meaning we allocated 1 byte too much... */
					Allocated_Internal->usage[ strlen(Allocated_Internal->usage)-1 ]= '\0';
				Allocated_Internal->user_internal= False;
				Allocated_Internal->is_usage= True;
				Allocated_Internal->take_usage= True;
				Allocated_Internal->value= take_ascanf_address( Allocated_Internal );
				Destroy_Form( &Allocated_Internal->procedure );
				if( !form ){
					Allocated_Internal->internal= -1;
				}
			}
			xfree( b );
		}
		else if( (s[0]== '{' || (s[0]== '&' && s[1]== '{')) && *last== '}' ){
		  double AA= *A;
		  int jj= *N, AlInt= *Allocate_Internal;
		  char *b, *S= (s[0]== '{')? s : &s[1];
		  char *name, serial[32];
		  Compiled_Form *aform= NULL, **_aform;
		  int instring= False, pw= ascanf_PopupWarn;
#ifdef ASCANF_AUTOVARS_UNIQUE
		  static unsigned long serialnr= 1;
				if( serialnr== 0 ){
					fprintf( StdErr,
						"Warning: automatic array serial wraparound in Create_AutoVariable(): risk of name collision.\n"
					);
				}
				sprintf( serial, "%lu", serialnr++ );
#else
				serial[0]= '\0';
#endif
			CrVar= True;
			name= concat( &S[1], serial, NULL );
			for( b= name, jj= 1; &S[jj]< last; jj++ ){
				if( S[jj]== '"' ){
					instring= !instring;
				}
				if( instring || !isspace(S[jj]) ){
					*b++= S[jj];
				}
			}
			*b= '\0';
/* 			b= buf= concat( "DCL[ {", name, "},-1,", name, "]", NULL );	*/
			b= buf= concat( "DCL[ {", name, "}", sep, "-1", sep, name, "]", NULL );
			jj= *N;
			pAllocate_Internal= *Allocate_Internal;
			*Allocate_Internal= -1;
			if( ascanf_verbose>1 && AlInt<= 0 ){
				fprintf( StdErr, "#%s%d array parameter \"%s\" => creating/updating temporary variable with \"%s\"\n",
					(form)? "ac:" : "", (*level), s, (buf)? buf : serror()
				);
			}
			else{
				 /* 20020524: the feedback that prints \"string\"== string is not really useful! */
				verbose= False;
			}
			  /* The expression-hook for later updating of this array internal variable	*/
			_aform= &aform;
			ascanf_PopupWarn= False;
			if( buf && !check_for_ascanf_function( idx, &buf, &AA, &jj, "fascanf", NULL, NULL, level ) ){
				fprintf( StdErr,
					"#%s%d array parameter \"%s\" => could not create temporary variable with \"%s\" (%s)\n",
					(form)? "ac" : "", (*level), s, buf, serror()
				);
			}
			else{
				*Allocate_Internal= AlInt;
				Allocated_Internal->user_internal= False;
				Allocated_Internal->is_address= True;
				Allocated_Internal->is_usage= False;
				Allocated_Internal->take_address= True;
				Allocated_Internal->take_usage= False;
				Allocated_Internal->value= take_ascanf_address( Allocated_Internal );
				if( !Allocated_Internal->procedure ||
					(Allocated_Internal->procedure->expr && strcmp( Allocated_Internal->procedure->expr, name))
				){
				  char *nm= name;
				  int n= Allocated_Internal->N;
					  /* If the newly (?!) allocated variable already has procedure code, it is not newly allocated.
					   \ Hence, we must be updating one that has not yet been deleted. For this to work correctly, we
					   \ must destroy any procedure in an internal var. when creating one that does not need any.
					   \ Anyhow, we compile the list-expression, and store it to later update the array's elements
					   \ when it is accessed.
					   */
					if( ascanf_verbose>1 && AlInt<= 0 ){
						fprintf( StdErr, "#%s%d: compiling array expression-list for updating at subsequent invocations:\n",
							(form)? "ac:" : "", (*level)
						);
					}
					Destroy_Form( &Allocated_Internal->procedure );
					if( __fascanf( &n, nm, Allocated_Internal->array, NULL, NULL, NULL, _aform, level, NULL ) && aform ){
						  /* aform->expr now contains the expression forming the array's first element. This
						   \ we substitute by the total expression (it is not used anyway...)
						   */
						if( ascanf_verbose>1 && AlInt<= 0 ){
							fprintf( StdErr, "#%s%d: array references a %s\n",
								(form)? "ac:" : "", (*level),
								(aform->list_of_constants==2)? "constant expression list (no unnecessary updating)" :
									"variable expression list (or containing variables) (will be updated)"
							);
						}
						if( aform->list_of_constants!= 2 ){
							xfree( aform->expr );
							aform->expr= strdup( name );
							Allocated_Internal->procedure= aform;
							Correct_Form_Top( &(Allocated_Internal->procedure), NULL, Allocated_Internal );
						}
						else{
							  /* 20020602 */
							Destroy_Form( &aform );
						}
					}
				}
				else{
					Destroy_Form( &aform );
				}
				if( !form ){
					Allocated_Internal->internal= -1;
				}
			}
			ascanf_PopupWarn= pw;
			xfree( b );
			xfree( name );
		}
		else{
			CrVar= True;
			nname= XGstrdup(s);
			buf= concat( "DCL[", s, "]", NULL );
		}
	}
	if( *b== ascanf_separator ){
		*b= '[';
	}
	  /* 20040512: */
	if( CrVar || CrString ){
		if( Allocated_Internal ){
		  ascanf_Function *nv;
	/* 	  char *expr= concat( (Allocated_Internal->take_usage)? "`" : "&", Allocated_Internal->name, NULL ), *c= expr;	*/
		  char *expr= concat( Allocated_Internal->name, NULL ), *c= expr;
			if( check_for_ascanf_function( idx, &expr, A, N, "fascanf", form, &nv, level ) ){
				r+= 1;
				*type= _ascanf_function;
				if( nv && *Allocate_Internal<= 0 && verbose && !ascanf_noverbose ){
					fprintf( StdErr, "#%s%d\t: %s== %s\n",
						(form)? "ac" : "", (*level),
						nv->name, (Allocated_Internal->take_usage)? (nv->usage)? nv->usage : "<empty string>" :
							d2str(nv->value, "%lf", NULL)
					);
				}
			}
			xfree( c );
		}
		else if( !form ){
			if( ascanf_AutoVarWouldCreate_msg && !active ){
				fprintf( StdErr, "#%d unknown symbol \"%s\" => compiler would create variable with \"%s\"\n",
					(*level), nname, (buf)? buf : serror()
				);
			}
		}
		else if( ascanf_arg_error ){
			fprintf( StdErr, "#%s%d unknown symbol \"%s\" => correct errors before autocreation with \"%s\"\n",
				(form)? "ac" : "", (*level), nname, (buf)? buf : serror()
			);
		}
		else{
			if( !ascanf_noverbose ){
				fprintf( StdErr, "#%s%d unknown symbol \"%s\" => creating variable with \"%s\"\n",
					(form)? "ac" : "", (*level), s, (buf)? buf : serror()
				);
			}
			if( buf ){
			  double AA= *A;
			  int jj= *N;
			  ascanf_Function *nv;
				Buf= buf;
				active= True;
				if( check_for_ascanf_function( idx, &buf, &AA, &jj, "fascanf", NULL, NULL, level ) ){
					  /* maybe active should be reset here (and change name) */
					if( check_for_ascanf_function( idx, &s, A, N, "fascanf", form, &nv, level ) ){
						r+= 1;
						*type= _ascanf_function;
						if( nv && verbose && !ascanf_noverbose ){
							fprintf( StdErr, "#%d\t: %s== %s\n",
								(*level),
								nv->name, ad2str( nv->value, d3str_format, NULL)
							);
						}
					}
				}
				active= False;
				xfree( Buf );
			}
		}
	}
	if( last_sp ){
		last[1]= last_sp;
	}
	if( allocated_return ){
		*allocated_return= Allocated_Internal;
	}
	Allocated_Internal= pAI;
	fascanf_unparsed_remaining= fur;
	AutoCreating= AC;
	ascanf_SyntaxCheck= SC;
	return( r );
}

/* ascanf(n, s, a) (ArrayScanf) functions; read a maximum of <n>
 * values from buffer 's' into array 'a'. Multiple values must be
 * separated by a comma; whitespace is ignored. <n> is updated to
 * contain the number of values actually read; this is also returned
 * unless an error occurs, in which case EOF is returned.
 * NOTE: cascanf(), dascanf() and lascanf() correctly handle mixed decimal and
 * hex values ; hex values must be preceded by '0x'
 \ 9502xx: If the 'form' argument is non-NULL, a compiled tree representing 's' is
 \ attached to it. No actual evaluation is done. Use compiled_fascanf()
 \ to evaluate it.
 \ 20001005: the low-level parser that was duplicated for elements 0,..,n-2 and the last (n-1)
 \ element is finally put in a separate function. The is_last argument is currently
 \ not used.
 */

void lSET_CHANGED_FLAG(char *ch, int i, double a, double b, int scanf_ret){
	if(ch){
		if( scanf_ret== EOF)
			ch[i]= VAR_UNCHANGED;
		else if( a== b)
			ch[i]= VAR_REASSIGN;
		else if( a!= b)
			ch[i]= VAR_CHANGED;
	}
}

int _fascanf_parser( double *a, char **s, char *ch, Compiled_Form **form, int *level, int r, int i, int *j, int is_last )
{ char *cs= *s, *ss= *s, *ns= *s, *name= NULL, *s_end;
  int sign= 1, negate= False;
  double A;
  ascanf_type type;
	type= _ascanf_value;
/* 	sign= 1;	*/
	switch( (*s)[0] ){
		case '+':
			sign= 1;
			ss++;
			break;
		case '-':
			sign= -1;
			ss++;
			break;
		case '!':
			negate= True;
			ss++;
			ns++;
			break;
	}
	s_end= &ss[strlen(ss)];
	A= *a;
	if( !strncmp( ss, "NaN", 3) ){
		if( negate ){
			A= 1;
		}
		else{
			set_NaN( A );
		}
		if( sign== -1 ){
			I3Ed(A)->s.s= 1;
			name= "-NaN";
		}
		else{
			name= "NaN";
		}
		r+= (*j= 1);
	}
	else if( !strncasecmp( ss, "Inf", 3) || !strncmp( ss, "\\#xa5\\", 6) || !strncmp( ss, GREEK_INF, GREEK_INF_LEN) ){
		if( negate ){
			A= 0;
		}
		else{
			set_Inf( A, sign);
		}
		if( sign== -1 ){
			name= (use_greek_inf)? GREEK_MIN_INF : "-Inf";
		}
		else{
			name= (use_greek_inf)? GREEK_INF : "Inf";
		}
		r+= (*j= 1);
	}
	else if( check_for_ascanf_function( i, &cs, &A, j, "fascanf", form, NULL, level) ){
		r+= 1;
		type= _ascanf_function;
		  /* point s to where ascanf_function left it (skips parsed text):	*/
		*s= MIN(cs,s_end);
		cs= NULL;
	}
	else if( !strncmp( ss, "0x", 2) ){
	  IEEEfp ie;
	  int n;
		(*s)+= 2;
		  /* !!! It is very important here that the IEEEfp structure is correctly defined! */
		if( sizeof(long) == 4 ){
			n = sscanf( ss, "%lx:%lx", &ie.l.high, &ie.l.low );
		}
		else{
			n = sscanf( ss, "%x:%x", &ie.l.high, &ie.l.low );
		}
		if( n == 2){
			*j= 1;
			r+= 1;
			A= (double) sign* ie.d;
		}
		else{
			if( sizeof(long) == 4 ){
				n = sscanf( ss, "%lx", &ie.l.low );
			}
			else{
				n = sscanf( ss, "%x", &ie.l.low );
			}
			if( n != EOF ){
				*j= 1;
				r+= 1;
				A= (double) sign* ie.l.low;
			}
		}
		if( negate ){
			A= (A && !NaN(A))? 0 : 1;
		}
	}
	else if( index( "0123456789+-.", (*s)[0]) ){
	  char *colon;
		if( ascanf_separator!= ':' && (colon= index( (*s), ':' )) && isdigit(colon[-1]) ){
		  int H, M, Si, Sn, Sd;
		  double S;
			if( (*j= sscanf( (*s), "%d:%d:%d.%d/%d", &H, &M, &Si, &Sn, &Sd ))== 5 ){
				r+= 1;
				if( Sd ){
					A= Si + ((double)Sn)/((double)Sd) + M* 60.0 + H* 3600.0;
				}
				else{
					A= Si + M* 60.0 + H* 3600.0;
				}
			}
			else if( (*j= sscanf( (*s), "%d:%d:%lf", &H, &M, &S ))== 3 ){
				r+= 1;
				A= S+ M* 60.0 + H* 3600.0;
			}
			if( negate ){
				A= (A && !NaN(A))? 0 : 1;
			}
		}
		else if( index( (*s), '/') ){
		  double B, P;
			if( (*j= sscanf( (*s), FLOFMT" ' "FLOFMT" / "FLOFMT, &P, &A, &B ))!= 3){
				*j= sscanf( (*s), FLOFMT" / "FLOFMT, &A, &B );
			}
			if( *j== 2 || *j== 3 ){
				r+= 1;
				if( B ){
					A= A/ B;
					if( *j== 3 ){
						if( P< 0 ){
							A= P- A;
						}
						else{
							A+= P;
						}
					}
				}
				else{
					if( A ){
						set_Inf( A, A );
					}
					else{
						  /* 20020612 */
						A= zero_div_zero.d;
					}
				}
			}
			if( negate ){
				A= (A && !NaN(A))? 0 : 1;
			}
		}
		else if( index( (*s), '*') ){
		  double B;
			if( (*j= sscanf( (*s), FLOFMT" * "FLOFMT, &A, &B )) ){
				r+= 1;
			}
			if( *j== 2 ){
				A*= B;
			}
			if( negate ){
				A= (A && !NaN(A))? 0 : 1;
			}
		}
		else if( index( (*s), '+') ){
		  double B;
			if( (*j= sscanf( (*s), FLOFMT" + "FLOFMT, &A, &B )) ){
				r+= 1;
			}
			if( *j== 2 ){
				A+= B;
			}
			if( negate ){
				A= (A && !NaN(A))? 0 : 1;
			}
		}
		else if( index( (*s), '-') ){
		  double B;
			if( (*j= sscanf( (*s), FLOFMT" - "FLOFMT, &A, &B )) ){
				r+= 1;
			}
			if( *j== 2 ){
				A-= B;
			}
			if( negate ){
				A= (A && !NaN(A))? 0 : 1;
			}
		}
		else{
			goto read_flpoint;
		}
	}
	else{
read_flpoint:;
		if( (*j= sscanf( ns, FLOFMT, &A ))!= EOF){
			r+= *j;
			if( negate ){
				A= (A && !NaN(A))? 0 : 1;
			}
		}
		if( *j!= 1 ){
		  /* See if we can make a valid, new, and possibly initialised label/variable out
		   \ of this unknown thing..
		   */
			r+= Create_AutoVariable( (*s), &A, i, j, form, &type, level, NULL, True );
		}
	}
	if( cs ){
		*s+= strlen(*s);
	}
	lSET_CHANGED_FLAG( ch, i, A, *a, *j);
	*a= A;
	if( type== _ascanf_value ){
		Add_Form( form, type, NULL, A, name, NULL, NULL );
	}
	if( ascanf_arg_error && (*ascanf_ExitOnError> 0 || (*ascanf_ExitOnError< 0 && ascanf_SyntaxCheck)) ){
		fprintf( StdErr, "fascanf(\"%s\"): raising read-terminate because of error(s) encountered\n",
			(*s)
		);
		ascanf_exit= 1;
		*j= EOF;
	}
	return(r);
}

/* read multiple floating point values	*/
static int _fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **form, int *level, int *nargs )
{	int i, r= 0, j= 1;
	char *S, *q;
	char *name;
	int _comp_= ascanf_SyntaxCheck;
	char *_cexpr_= ascanf_CompilingExpression;
	double DATA[4];
	int avp= ascanf_verbose, anvp= ascanf_noverbose;
	int Nset= 0;
	int raiv= reset_ascanf_index_value, racv= reset_ascanf_currentself_value;
	int psiU= param_scratch_inUse, using_ps= 0;
/* 	double *ArgList= af_ArgList->array;	*/
/* 	int Argc= af_ArgList->N;	*/

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}

	if( form ){
		Elapsed_Since(&AscanfCompileTimer, True);
	}

	S= s;
	if( nargs ){
		*nargs= -1;
	}

	if( a== param_scratch ){
		if( *n> param_scratch_len ){
		  /* When using clean_param_scratch() before passing param_scratch as scratch memory,
		   \ this test should never be necessary!
		   */
			*n= param_scratch_len;
		}
		param_scratch_inUse= True;
		using_ps= True;
	}

	if( data ){
		ascanf_data_buf= data;
		DATA[0]= *ascanf_data0;
		DATA[1]= *ascanf_data1;
		DATA[2]= *ascanf_data2;
		DATA[3]= *ascanf_data3;
		Nset= 1;
		*ascanf_data0= data[0];
		*ascanf_data1= data[1];
		*ascanf_data2= data[2];
		*ascanf_data3= data[3];
	}
	if( column ){
		ascanf_column_buf= column;
	}
/* 	if( ascanf_update_ArgList ){	*/
/* 		SET_AF_ARGLIST( a, *n );	*/
/* 	}	*/

	  /* ascanf_verbose can be changed through the variable $verbose. But if
	   \ we have no way to update the cache variable avp in that case, such a
	   \ change will never have a lasting effect. Thus, there must be an
	   \ accesshandler that will set old_value to the new value after "handling"
	   \ the change. NB: most of the time, old_value==value!! To make sure that
	   \ we can rely on its value as an indicator that $verbose was accessed,
	   \ we set it to NaN (a nonsense value for $verbose..).
	   \ 20020328: $verbose can definitely be NaN: that's the same as False...
	   \ But it shouldn't take on the value &$verbose.... (take_ascanf_address(&af_verbose)).
	   */
	if( !af_verbose->accessHandler ){
		af_verbose->accessHandler= &internal_AHandler;
	}
	ascanf_d3str_format->accessHandler= &internal_AHandler;

	af_verbose->old_value= af_verbose_old_value;
/* 	set_NaN( af_verbose->old_value );	*/

	if( !(*level)){
		reset_ascanf_index_value= True;
		  /* 991014: reset error flag on toplevel...	*/
		ascanf_arg_error= 0;
	}
	else{
		reset_ascanf_currentself_value= False;
	}
	  /* 991014: also switch off if *not* compiling. It turns out that we *can* make strange
	   \ jumps due to event handling (a redraw generated during the posting of an errorbox, e.g....)
	   */
	if( form ){
		ascanf_SyntaxCheck= 1;
		ascanf_CompilingExpression= strdup(s);
		if( *ascanf_compile_verbose> 0 ){
			*ascanf_verbose_value= ascanf_verbose= 1;
			if( !avp ){
				fprintf( StdErr, "#ac:%d: compile[%s] %s\n", (*level), s, (TBARprogress_header)? TBARprogress_header : "" );
			}
		}
		else if( *ascanf_compile_verbose< 0 ){
			ascanf_noverbose= 1;
			*ascanf_verbose_value= ascanf_verbose= 0;
		}
	}
	else{
		ascanf_CompilingExpression= NULL;
		  /* 20040620: attempt to *not* unset the SyntaxCheck flag in certain cases, like when creating an autovar.
		   \ within an expression being compiled...
		   */
		if( !(*level) || ascanf_SyntaxCheck> 0 ){
			ascanf_SyntaxCheck= 0;
		}
		  /* 20040620: */
		else if( ascanf_SyntaxCheck< 0 ){
			ascanf_SyntaxCheck*= -1;
		}
	}
/* 	(*level)++;	*/
	for( i= 0, q= ascanf_index( s, ascanf_separator, NULL); i< *n && j!= EOF && q && *s; a++, i++ ){
	  int oq= *q;
		*q= '\0';
		for( ; isspace((unsigned char)*s) && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		name= NULL;
		if( *s ){
			r= _fascanf_parser( a, &s, ch, form, level, r, i, &j, False );
		}
		*q= oq;
		s= q+ 1;
		  /* 20010613: do an additional check here for redundant whitespace  */
		for( ; isspace((unsigned char)*s) && *s; s++);
		q= ascanf_index( s, ascanf_separator, NULL);
	}
	if( ascanf_SyntaxCheck && q && i>= *n ){
	  int ii= i, jj= 0;
	  char *qq= q, *ss= s;
		  /* jj!=j but jj=0, thus never EOF. This ensures that we can get warnings about redundant arguments	*/
		for( ; jj!= EOF && qq && *ss; ii++ ){
			ss= qq+ 1;
			qq= ascanf_index( ss, ascanf_separator, NULL);
		}
		ii+= 1;
		if( *n== ASCANF_MAX_ARGS ){
			fprintf( StdErr, "IMPORTANT: MORE arguments (%d) than currently ALLOWED (%d) - use MaxArguments[]\n\"%s\"\n",
				ii, *n, S
			);
		}
		else{
			fprintf( StdErr, "Warning: too many arguments (%d>%d)\n", ii, *n );
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "\"%s\"\n", S );
			}
		}
		if( nargs ){
			*nargs= ii;
		}
	}
	for( ; isspace((unsigned char)*s) && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s && !ascanf_exit ){
		r= _fascanf_parser( a, &s, ch, form, level, r, i, &j, False );
		  /* Do the increment of the destination array outside the parser routine!	*/
		a++;
	}

/* 	(*level)--;	*/

	if( form ){
		if( pragma_unlikely(ascanf_verbose) && !avp ){
		  char hdr[16];
			sprintf( hdr, "#ac:%d: ", (*level) );
			fputs( hdr, StdErr );
			Print_Form( StdErr, form, 0, True, hdr, NULL, "\n", True );
		}
		if( *ascanf_UseConstantsLists ){
		  int c= check_for_constants_list( form );
			if( pragma_unlikely(ascanf_verbose) && c && (*level)<= 1 ){
				fprintf( StdErr, "#ac:%d: argumentlist consists of only constants and variables\n", (*level) );
			}
		}
	}

	if( using_ps ){
		param_scratch_inUse= psiU;
	}

	  /* If old_value is not a NaN at this point, set avp (and thus ascanf_verbose)
	   \ to its value, which *must* be the value to which $verbose was set by the user.
	   \ And reset old_value to a NaN immediately, to prevent that this value propagates
	   \ to places where we don't want it. I think this implements understandable behaviour
	   \ for $verbose...
	   */
/* 	if( !NaN( af_verbose->old_value ) )	*/
	if( af_verbose->old_value!= af_verbose_old_value )
	{
		avp= af_verbose->old_value;
/* 		set_NaN(af_verbose->old_value);	*/
		af_verbose->old_value= af_verbose_old_value;
	}

	xfree( ascanf_CompilingExpression );
	ascanf_CompilingExpression= _cexpr_;
	ascanf_SyntaxCheck= _comp_;
	*ascanf_verbose_value= ascanf_verbose= avp;
	ascanf_noverbose= anvp;
	reset_ascanf_currentself_value= racv;
	reset_ascanf_index_value= raiv;

/* 	SET_AF_ARGLIST( ArgList, Argc );	*/

	if( Nset ){
		*ascanf_data0= DATA[0];
		*ascanf_data1= DATA[1];
		*ascanf_data2= DATA[2];
		*ascanf_data3= DATA[3];
		Nset= 0;
	}
	if( nargs && *nargs< 0 ){
		*nargs= r;
	}

	if( *s== ']' ){
		fascanf_unparsed_remaining= s+1;
	}
	else{
		fascanf_unparsed_remaining= s;
	}

	if( form ){
		Elapsed_Since( &AscanfCompileTimer, False );
		SS_Add_Data_( SS_AscanfCompileTime, 1, AscanfCompileTimer.HRTot_T, 1.0 );
	}

	if( r< *n){
		*n= r;					/* not enough read	*/
		if( form && *form ){
			(*form)->ok= EOF;
		}
		return( EOF);				/* so return EOF	*/
	}
	if( form && *form ){
		(*form)->ok= r;
	}
	return( r);
}

void Convert_French_Numerals( char *rbuf )
{ char *c = &rbuf[1];
  int i, n = strlen(rbuf)-1;
	for( i = 1 ; i < n ; i++, c++ ){
		if( *c == ',' && (isdigit(c[-1]) || isdigit(c[1])) ){
			*c = '.';
		}
	}
}

int fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **form )
{ int level= 0;
#ifndef USE_AA_REGISTER
  int _comp_= ascanf_SyntaxCheck;
  int avp= ascanf_verbose, acp= ascanf_comment, app= ascanf_popup;
  int AI= *Allocate_Internal;
	PAS_already_protected= False;
	if( PAS_Lazy_Protection!=2 || sigsetjmp(pa_jmp_top,1)==0 )
#endif
	{
		if( s && ascanf_separator != ',' ){
			Convert_French_Numerals( s );
		}
		return( _fascanf( n, s, a, ch, data, column, form, &level, NULL ) );
	}
#ifndef USE_AA_REGISTER
	else{
		*n= -1;
		ascanf_SyntaxCheck= _comp_;
		*ascanf_verbose_value= ascanf_verbose= avp, ascanf_comment= acp, ascanf_popup= app;
		*Allocate_Internal= AI;
		return(EOF);
	}
#endif
}

#define ASCANF_ARRAY_ACCESS(lform,larray,lfirst,lidx,lN,larg,lresult,llevel) { \
	int i= (lfirst)+1, j; \
		if( (lidx)== -2 ){ \
		  /* 20020530: protect against the new support for -2 index */ \
			(lidx)= (larray)->last_index; \
		} \
		if( unused_pragma_likely( (lidx)>= 0 && (lidx)< (larray)->N ) ){ \
			if( (larray)->iarray ){ \
				for( j= (lidx); j< (larray)->N && i< (lN)-(lfirst); i++, j++ ){ \
					(larray)->iarray[j]= (int) (larg)[i]; \
				} \
				if( pragma_unlikely(ascanf_verbose) ){ \
					ShowArrayFeedback( StdErr, (larray), (lidx), (lfirst), (lN) ); \
				} \
				*(lresult)= (larray)->value= (larray)->iarray[(lidx)]; \
			} \
			else{ \
				for( j= (lidx); j< (larray)->N && i< (lN)-(lfirst); i++, j++ ){ \
					(larray)->array[j]= (larg)[i]; \
				} \
				if( pragma_unlikely(ascanf_verbose) ){ \
					ShowArrayFeedback( StdErr, (larray), (lidx), (lfirst), (lN) ); \
				} \
				*(lresult)= (larray)->value= (larray)->array[(lidx)]; \
			} \
			(larray)->assigns+= 1; \
				if( (larray)->accessHandler && j!= (lidx) ){ \
					(larray)->last_index= (lidx); \
						AccessHandler( (larray), (larray)->name, (llevel), (lform), NULL, NULL ); \
				} \
		} \
		else if( (lidx)== (larray)->N || ((lidx)< 0 && i< (lN)-(lfirst)) ){ \
			(lidx)= 0; \
				if( (larray)->iarray ){ \
					for( j= (lidx); j< (larray)->N && i< (lN)-(lfirst); j++ ){ \
						(larray)->iarray[j]= (int) (larg)[i]; \
							/* 990721: use all specified args, repeating only the last	*/ \
							if( i< (lN)-(lfirst)-1 ){ \
								i++; \
							} \
					} \
					if( pragma_unlikely(ascanf_verbose) ){ \
						ShowArrayFeedback( StdErr, (larray), (lidx), (lfirst), (lN) ); \
					} \
					*(lresult)= (larray)->value= (larray)->iarray[(lidx)]; \
				} \
				else{ \
					for( j= (lidx); j< (larray)->N && i< (lN)-(lfirst); j++ ){ \
						(larray)->array[j]= (larg)[i]; \
							if( i< (lN)-(lfirst)-1 ){ \
								i++; \
							} \
					} \
					if( pragma_unlikely(ascanf_verbose) ){ \
						ShowArrayFeedback( StdErr, (larray), (lidx), (lfirst), (lN) ); \
					} \
					*(lresult)= (larray)->value= (larray)->array[(lidx)]; \
				} \
			(larray)->assigns+= 1; \
				if( (larray)->accessHandler ){ \
					(larray)->last_index= (lidx); \
						AccessHandler( (larray), (larray)->name, (llevel), (lform), NULL, NULL ); \
				} \
		} \
		else{ \
			if( i< (lN)-(lfirst) ){ \
				sprintf( ascanf_errmesg, "(assign to index %d outside [-2,%d>)", (lidx), (larray)->N ); \
					ascanf_emsg= ascanf_errmesg; \
					ascanf_arg_error= 1; \
			} \
			*(lresult)= (larray)->value= (larray)->N; \
				(larray)->reads+= 1; \
		} \
	(larray)->last_index= (lidx); \
}

/* Lowlevel routine that handles array writes for various others. <lform> should point to the Compiled_Form frame
 \ describing the operation (if available). <larray> is the target array. <lfirst> gives the 1st element of the argument
 \ array <larg>, the one containing the (original) array index, that is copied (or rather, specified) in <lidx>.
 \ <lN> is the number of elements in <larg>. The result is stored in *<lresult>.
 \ The number of values to store is thus <lN>-<lfirst>-1, starting at index <lidx> (supposing <lidx>!= <larray>->N)
 */
#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
void _ascanf_array_access( Compiled_Form *lform, ascanf_Function *larray,
	int lfirst, int lidx, int lN, double *larg, double *lresult, int *llevel
)
{
	ASCANF_ARRAY_ACCESS( lform, larray, lfirst, lidx, lN, larg, lresult, llevel );
}

/* extern int ecl_get_2ndargs( Compiled_Form *fa, Compiled_Form *faa, int *level );	*/
extern int ecl_get_args( Compiled_Form *form, Compiled_Form *arg1, int *level );

int ecl_get_2ndargs( Compiled_Form *fa, Compiled_Form *faa, int *level )
{ ascanf_type ega_type= (faa->last_value)? _ascanf_variable : faa->type;
	if( faa->args ){
	  int llevel= *level+1;
		if( !faa->args->list_of_constants || ecl_get_args(faa, faa->args,&llevel) ){
			return(1);
		}
	}
	if( faa->fun ){
		if( faa->take_address ){
			if( faa->take_usage || 0 ){
				faa->fun->take_usage= True;
			}
			else{
				faa->fun->take_usage= False;
			}
			faa->value= take_ascanf_address( faa->fun );
		}
		else{
			switch( ega_type ){
				case _ascanf_simplestats:
				case _ascanf_simpleanglestats:
				case _ascanf_python_object:
				case _ascanf_variable:
					if( faa->negate ){
						faa->value= (faa->fun->value && !NaN(faa->fun->value))? 0 : 1;
					}
					else{
						faa->value= faa->sign* faa->fun->value;
					}
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " #Ceg2a%d(%s%s) ,%s%s==", (*level)+1,
							CF_prefix(fa), fa->fun->name,
							CF_prefix(faa), faa->fun->name
						);
						if( faa->list_of_constants ){
							fputs( "<C>", StdErr );
						}
						fprintf( StdErr, "%s",
							ad2str( faa->fun->value, d3str_format, NULL)
						);
					}
#endif
					break;
				case _ascanf_array:
					if( unused_pragma_likely( faa->fun->last_index< faa->fun->N ) ){
						if( faa->fun->iarray ){
							faa->fun->value= (faa->fun->last_index==-1)? faa->fun->N :
								faa->fun->iarray[faa->fun->last_index];
							if( faa->negate ){
								faa->value=
									(faa->fun->value && !NaN(faa->fun->value))? 0 : 1;
							}
							else{
								faa->value= faa->sign* faa->fun->value;
							}
						}
						else{
							faa->fun->value= (faa->fun->last_index==-1)? faa->fun->N :
								faa->fun->array[faa->fun->last_index];
							if( faa->negate ){
								faa->value=
									(faa->fun->value && !NaN(faa->fun->value))? 0 : 1;
							}
							else{
								faa->value= faa->sign* faa->fun->value;
							}
						}
					}
					else{
						ascanf_emsg= " (array index out-of-range (" STRING(__LINE__) ") ) ";
						ascanf_arg_error= True;
						if( faa->fun->N< 0 ){
							fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", faa->fun->name );
							ascanf_escape= ascanf_interrupt= True;
							*ascanf_escape_value= *ascanf_interrupt_value= 1;
						}
					}
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " #Ceg2a%d(%s%s) ,%s%s[%d]==", (*level)+1,
							CF_prefix(fa), fa->fun->name,
							CF_prefix(faa), faa->fun->name, faa->fun->last_index
						);
						if( faa->list_of_constants ){
							fputs( "<C>", StdErr );
						}
						fprintf( StdErr, "%s",
							ad2str( faa->fun->value, d3str_format, NULL)
						);
					}
#endif
					break;
				default:
					return(1);
					break;
			}
		}
	}
#ifdef VERBOSE_CONSTANTS_EVALUATION
	else if( pragma_unlikely(ascanf_verbose) ){
		fprintf( StdErr, " #Ceg2a%d(%s%s) ,", (*level)+1, CF_prefix(fa), fa->fun->name );
		if( faa->list_of_constants ){
			fputs( "<C>", StdErr );
		}
		fprintf( StdErr, "%s", ad2str( faa->value, d3str_format, NULL) );
	}
#endif
	return(0);
}

#ifdef VERBOSE_CONSTANTS_EVALUATION

#define EVALUATE_SS_CONSTANTS_LISTS(lform,fa,eval_args) { \
	if( fa->args ){  \
	  Compiled_Form *faa= fa->args;  \
		if( eval_args && faa ){ \
			if( !faa->list_of_constants || (faa->take_address && faa->type== _ascanf_array) || ecl_get_args(fa, faa,level) ){ \
				goto evaluate_arguments; \
			} \
		} \
		if( fa->type== _ascanf_simplestats ){ \
			if( fa->fun->N>1 ){ \
				if( fa->fun->last_index>= 0 && fa->fun->last_index< fa->fun->N ){ \
					if( !ascanf_SyntaxCheck ){\
						SS_Add_Data( &(fa->fun->SS[fa->fun->last_index]), 1, faa->value, 1.0 ); \
					} \
				} \
				else{ \
					ascanf_emsg= " ($SS_StatsBin index out-of-bounds) "; \
					ascanf_arg_error= 1; \
				} \
			} \
			else{ \
				if( !ascanf_SyntaxCheck ){\
					SS_Add_Data( fa->fun->SS, 1, faa->value, 1.0 ); \
				} \
			} \
			if( pragma_unlikely(ascanf_verbose) ){ \
				if( fa->fun->N> 1 ){ \
					fprintf( StdErr, " (SS_Add[@[&%s,%d],1,%s,1])== ", fa->fun->name, fa->fun->last_index, \
						ad2str( faa->value, d3str_format, 0) \
					); \
				} \
				else{ \
					fprintf( StdErr, " (SS_Add[&%s,1,%s,1])== ", fa->fun->name, \
						ad2str( faa->value, d3str_format, 0) \
					); \
				} \
			} \
		} \
		else{ \
		  Compiled_Form *faa2= faa->cdr; \
			if( faa2 ){ \
				if( !faa2->list_of_constants || ecl_get_args(fa, faa2,level) ){ \
					goto evaluate_arguments; \
				} \
				if( !ascanf_SyntaxCheck ){ \
					SAS_Add_Data( fa->fun->SAS, 1, faa->value, 1.0, ASCANF_TRUE(faa2->value) ); \
				} \
			} \
			else if( !ascanf_SyntaxCheck ){ \
				SAS_Add_Data( fa->fun->SAS, 1, faa->value, 1.0, ASCANF_TRUE(*SAS_converts_angle) ); \
			} \
			if( pragma_unlikely(ascanf_verbose) && fa->fun->SAS ){ \
				fprintf( StdErr, " (SAS_Add[&%s,1,%s,1,%s,%s,%s=%d])== ", fa->fun->name, \
					ad2str( faa->value, d3str_format, 0), \
					ad2str( fa->fun->SAS->Gonio_Base, d3str_format, 0), \
					ad2str( fa->fun->SAS->Gonio_Offset, d3str_format, 0), \
					(faa2)? "arg" : "$SAS_converts_angle", \
					ASCANF_TRUE_((faa2)? faa2->value : *SAS_converts_angle) \
				); \
			} \
		} \
		fa->fun->value= faa->value;  \
		fa->fun->assigns+= 1;  \
		if( fa->fun->accessHandler ){  \
			AccessHandler( fa->fun, fa->fun->name, level, (lform), NULL, NULL );  \
		}  \
	}  \
	break;  \
}

#else

#define EVALUATE_SS_CONSTANTS_LISTS(lform,fa,eval_args) { \
	if( fa->args ){  \
	  Compiled_Form *faa= fa->args;  \
		if( eval_args && faa ){ \
			if( !faa->list_of_constants || ecl_get_args(fa, faa,level) ){ \
				goto evaluate_arguments; \
			} \
		} \
		if( fa->type== _ascanf_simplestats ){ \
			if( fa->fun->N>1 ){ \
				if( unused_pragma_likely( fa->fun->last_index>= 0 && fa->fun->last_index< fa->fun->N ) ){ \
					if( !ascanf_SyntaxCheck ){\
						SS_Add_Data( &(fa->fun->SS[fa->fun->last_index]), 1, faa->value, 1.0 ); \
					} \
				} \
				else{ \
					ascanf_emsg= " ($SS_StatsBin index out-of-bounds) "; \
					ascanf_arg_error= 1; \
				} \
			} \
			else{ \
				if( !ascanf_SyntaxCheck ){\
					SS_Add_Data( fa->fun->SS, 1, faa->value, 1.0 ); \
				} \
			} \
		} \
		else{ \
		  Compiled_Form *faa2= faa->cdr; \
			if( faa2 ){ \
				if( !faa2->list_of_constants || ecl_get_args(fa, faa2,level) ){ \
					goto evaluate_arguments; \
				} \
				if( !ascanf_SyntaxCheck ){ \
					SAS_Add_Data( fa->fun->SAS, 1, faa->value, 1.0, ASCANF_TRUE(faa2->value) ); \
				} \
			} \
			else if( !ascanf_SyntaxCheck ){ \
				SAS_Add_Data( fa->fun->SAS, 1, faa->value, 1.0, ASCANF_TRUE(*SAS_converts_angle) ); \
			} \
		} \
		fa->fun->value= faa->value;  \
		fa->fun->assigns+= 1;  \
		if( fa->fun->accessHandler ){  \
			AccessHandler( fa->fun, fa->fun->name, level, (lform), NULL, NULL );  \
		}  \
	}  \
	break;  \
}

#endif


int ecl_get_args( Compiled_Form *fa, Compiled_Form *faa, int *level )
{
  ascanf_type ega_type= (faa->last_value)? _ascanf_variable : faa->type;
	if( faa->args ){
	  int llevel= *level+1, ret;
		if( !faa->args->list_of_constants || (ret= ecl_get_args(faa, faa->args,&llevel)) ){
		 /* Not sure we need to call correct_constants_args_list() here: */
			  /* 20050415: */
			if( unused_pragma_unlikely( !(ret== -1 && faa->fun->type== _ascanf_array) ) ){
				correct_constants_args_list( faa, level, __FILE__, __LINE__ );
			}
			return(1);
		}
	}
	if( faa->fun ){
		if( faa->take_address ){
			if( faa->take_usage || 0 ){
				faa->fun->take_usage= True;
			}
			else{
				faa->fun->take_usage= False;
			}
			faa->value= take_ascanf_address( faa->fun );
		}
		else{
			switch( ega_type ){
				case _ascanf_simplestats:
				case _ascanf_simpleanglestats:
				case _ascanf_python_object:
				case _ascanf_variable:
					if( faa->negate ){
						faa->value= (faa->fun->value && !NaN(faa->fun->value))? 0 : 1;
					}
					else{
						faa->value= faa->sign* faa->fun->value;
					}
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "#Cega%d: \t%s%s[%s%s==", (*level)+1,
							CF_prefix(fa), fa->fun->name,
							CF_prefix(faa), faa->fun->name
						);
						if( faa->list_of_constants ){
							fputs( "<C>", StdErr );
						}
						fprintf( StdErr, "%s]\t->\n",
							ad2str( faa->fun->value, d3str_format, NULL)
						);
					}
#endif
					break;
				case _ascanf_array:
					if( unused_pragma_likely( faa->fun->last_index< faa->fun->N ) ){
						if( faa->fun->iarray ){
							faa->fun->value= (faa->fun->last_index==-1)? faa->fun->N :
								faa->fun->iarray[faa->fun->last_index];
							if( faa->negate ){
								faa->value=
									(faa->fun->value && !NaN(faa->fun->value))? 0 : 1;
							}
							else{
								faa->value= faa->sign* faa->fun->value;
							}
						}
						else{
							faa->fun->value= (faa->fun->last_index==-1)? faa->fun->N :
								faa->fun->array[faa->fun->last_index];
							if( faa->negate ){
								faa->value=
									(faa->fun->value && !NaN(faa->fun->value))? 0 : 1;
							}
							else{
								faa->value= faa->sign* faa->fun->value;
							}
						}
					}
					else{
						ascanf_emsg= " (array index out-of-range (" STRING(__LINE__) ") ) ";
						ascanf_arg_error= True;
						if( faa->fun->N< 0 ){
							fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", faa->fun->name );
							ascanf_escape= ascanf_interrupt= True;
							*ascanf_escape_value= *ascanf_interrupt_value= 1;
						}
					}
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "#Cega%d: \t%s%s[%s%s[%d]==", (*level)+1,
							CF_prefix(fa), fa->fun->name,
							CF_prefix(faa), faa->fun->name, faa->fun->last_index
						);
						if( faa->list_of_constants ){
							fputs( "<C>", StdErr );
						}
						fprintf( StdErr, "%s]\t->\n",
							ad2str( faa->fun->value, d3str_format, NULL)
						);
					}
#endif
					break;
				default:
					correct_constants_args_list( faa, level, __FILE__, __LINE__ );
					return(1);
					break;
			}
		}
	}
	switch( fa->type ){
		case _ascanf_simplestats:
		case _ascanf_simpleanglestats:
		  /* data is added to simplestats variables at the higher level; here we set only their readout value
		   \ (this is different than how arrays are handled!
		   */
		case _ascanf_variable:
		case _ascanf_python_object:
			fa->fun->value= faa->value;
			break;
		case _ascanf_array:{
		  int idx;
		  double didx= faa->value;
			  /* 20050131: !(a<b) should be faster than a>=b for fl.p on PPC?? Use idx ... */
			if( unused_pragma_likely( didx>= -2 && didx< (fa->fun->N + ((AllowArrayExpansion)?0:1)) ) ){
				idx= (int) faa->value;
ega_access_array_element:;
				if( faa->cdr ){
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "#Cega%d: \t%s%s[%d", (*level)+1,
							CF_prefix(fa), fa->fun->name, (int) idx
						);
					}
#endif
					(*level)+= 1;
					  /* 20051112: I hope pragma_unlikely doesn't influence the evaluation of its argument! */
					if( unused_pragma_unlikely( ecl_get_2ndargs(fa, faa->cdr, level) ) ){
						correct_constants_args_list( faa, level, __FILE__, __LINE__ );
						return(1);
					}
			if( fa->level ){
				fprintf( StdErr, "## ecl_get_args modifying fa->argvals[1] with fa->level == %d\n", fa->level );
			}
					fa->argvals[1]= faa->cdr->value;
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fputs( "]== \n", StdErr );
					}
#endif
					  /* 20020530 */
					_ascanf_array_access( fa, fa->fun, 0, (int) idx,
						fa->alloc_argc, fa->argvals, &(fa->fun->value), level );
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "==%s\t->\n", ad2str( fa->fun->value, d3str_format, NULL) );
					}
#endif
					(*level)-= 1;
				}
				else{
					if( pragma_unlikely(ascanf_verbose) ){
						  /* It is not necessary to print anything at this level; the higher level will print do that */
						if( (int) idx== fa->fun->N ){
							  /* We can do this extra check here without penalty */
							goto ega_array_error;
						}
					}
					  /* 20030403: DON'T set last_index to -2!!! */
					if( idx!= -2 ){
						fa->fun->last_index= (int) idx;
					}
				}
			}
			else{
				if( Inf(faa->value)> 0 ){
					idx= fa->fun->N-1;
					goto ega_access_array_element;
				}
				else if( Inf(faa->value)< 0 ){
					idx= 0;
					goto ega_access_array_element;
				}
				else if( AllowArrayExpansion && Resize_ascanf_Array( fa->fun, (int) faa->value+1, NULL ) ){
					idx= (int) faa->value;
					goto ega_access_array_element;
				}
				if( pragma_unlikely(ascanf_verbose) ){
ega_array_error:;
					fprintf( StdErr, "#Cega%d: array %s index %s out-of-range (-1..%d) (line %d)\n",
						(*level)+1, fa->fun->name, ad2str( faa->value, d3str_format, NULL), fa->fun->N,
						__LINE__
					);
				}
				if( fa->fun->N< 0 ){
					fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", fa->fun->name );
					ascanf_escape= ascanf_interrupt= True;
					*ascanf_escape_value= *ascanf_interrupt_value= 1;
					ascanf_arg_error= 1;
				}
				  /* 20050415: accessing error: the result is a NaN! */
				set_NaN(fa->fun->value);
				return(-1);
			}
			break;
		}
	}
	return(0);
/* evaluate_arguments:;	*/
     /* Not sure we need to call correct_constants_args_list() here: */
/* 	correct_constants_args_list( faa, level, __FILE__, __LINE__ );	*/
/* 	return(1);	*/
}


/* 20020519: handle the case when fa->fun==NULL!! (see Form_Remove...()). */
int evaluate_constants_args_list( Compiled_Form *lform, double *arg, int n, int *level )
{
  Compiled_Form *fa= (lform)->args;
  int i= 0;
	while( fa ){
	  ascanf_type ecal_type= (fa->last_value)? _ascanf_variable : fa->type;
		if( fa->take_address ){
			if( fa->take_usage || 0 ){
				fa->fun->take_usage= True;
			}
			else{
				fa->fun->take_usage= False;
			}
			fa->value= take_ascanf_address( fa->fun );
			arg[i]= fa->value;
		}
		else switch( ecal_type ){
			default:
				return(-1);
				break;
			case _ascanf_novariable:
				fa->value= 0;
				arg[i]= fa->value;
				fprintf( StdErr, "#Cecal%d: value of deleted %s\"%s\"=0 (was %s)\n",
					(*level)+1, (fa->fun->internal)? "internal " : "", fa->fun->name, ad2str( fa->fun->value, d3str_format, NULL)
				);
				break;
			case _ascanf_value:
				if( fa->negate ){
					arg[i]= (fa->value && !NaN(fa->value))? 0 : 1;
				}
				else{
					arg[i]= fa->sign* fa->value;
				}
				break;
			case _ascanf_variable:
			case _ascanf_python_object:
				if( fa->args ){
				  Compiled_Form *faa= fa->args;
					if( !faa->list_of_constants || ecl_get_args(fa, faa, level) ){
						return(-1);
					}
					fa->fun->value= faa->value;
					fa->fun->assigns+= 1;
					if( fa->fun->accessHandler ){
						AccessHandler( fa->fun, fa->fun->name, level, (lform), NULL, NULL );
					}
				}
				if( fa->negate ){
					fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
				}
				else{
					fa->value= fa->sign* fa->fun->value;
				}
				arg[i]= fa->value;
				break;
			case _ascanf_array:
				if( fa->args ){
				  Compiled_Form *faa= fa->args;
				  int ret;
					if( !faa->list_of_constants || (ret= ecl_get_args(fa, faa, level)) ){
						  /* 20050415: */
						if( ret== -1 ){
							arg[i]= fa->fun->value;
							return(-2);
						}
						else{
							return( -1 );
						}
					}
				}
				if( unused_pragma_likely( fa->fun->last_index< fa->fun->N ) ){
					if( fa->fun->iarray ){
						fa->fun->value= (fa->fun->last_index==-1)? fa->fun->N :
							fa->fun->iarray[fa->fun->last_index];
					}
					else{
						fa->fun->value= (fa->fun->last_index==-1)? fa->fun->N :
							fa->fun->array[fa->fun->last_index];
					}
					if( fa->negate ){
						fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
					}
					else{
						fa->value= fa->sign* fa->fun->value;
					}
					arg[i]= fa->value;
				}
				else{
					ascanf_emsg= " (array index out-of-range (" STRING(__LINE__) ") ) ";
					ascanf_arg_error= True;
					if( fa->fun->N< 0 ){
						fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", fa->fun->name );
						ascanf_escape= ascanf_interrupt= True;
						*ascanf_escape_value= *ascanf_interrupt_value= 1;
					}
				}
				break;
			case _ascanf_simplestats:
				EVALUATE_SS_CONSTANTS_LISTS(lform,fa,True);
				if( fa->negate ){
					fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
				}
				else{
					fa->value= fa->sign* fa->fun->value;
				}
				arg[i]= fa->value;
				break;
			case _ascanf_simpleanglestats:
				EVALUATE_SS_CONSTANTS_LISTS(lform,fa,True);
				if( fa->negate ){
					fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
				}
				else{
					fa->value= fa->sign* fa->fun->value;
				}
				arg[i]= fa->value;
				break;
		}
#ifdef VERBOSE_CONSTANTS_EVALUATION
		if( pragma_unlikely(ascanf_verbose) && fa->fun ){
			fprintf( StdErr, "#Cecal%d: \t%s%s",
				(*level)+1, CF_prefix(fa), FUNNAME(fa->fun)
			);
			if( ecal_type== _ascanf_array ){
				fprintf( StdErr, "[%d]", fa->fun->last_index );
			}
			fprintf( StdErr, "==%s\n", ad2str( fa->value, d3str_format, NULL) );
		}
#endif
		fa= fa->cdr; {
			i+= 1;
		}
	}
	n= i;
	return(n);
evaluate_arguments:;
	return(-1);
}

int evaluate_constants_list( Compiled_Form *lform, double *arg, int n, int *level, int follow_list )
{ Compiled_Form *fa= (lform);
  int i= 0;
	while( fa && i< n ){
	  ascanf_type ecl_type= (fa->last_value)? _ascanf_variable : fa->type;
		if( fa->take_address ){
			if( fa->take_usage || 0 ){
				fa->fun->take_usage= True;
			}
			else{
				fa->fun->take_usage= False;
			}
			fa->value= take_ascanf_address( fa->fun );
			arg[i]= fa->value;
		}
		else switch( ecl_type ){
			default:
				return(-1);
				break;
			case _ascanf_novariable:
				fa->value= 0;
				arg[i]= 0;
				fprintf( StdErr, "#Cecl%d: value of deleted %s\"%s\"=0 (was %s)\n",
					(*level)+1, (fa->fun->internal)? "internal " : "", fa->fun->name, ad2str( fa->fun->value, d3str_format, NULL)
				);
				break;
			case _ascanf_value:
				if( fa->negate ){
					arg[i]= (fa->value && !NaN(fa->value))? 0 : 1;
				}
				else{
					arg[i]= fa->sign* fa->value;
				}
				break;
			case _ascanf_variable:
			case _ascanf_python_object:
				if( fa->args ){
				  Compiled_Form *faa= fa->args;
					if( !faa->list_of_constants || ecl_get_args(fa, faa, level) ){
						return(-1);
					}
					fa->fun->value= faa->value;
					fa->fun->assigns+= 1;
					if( fa->fun->accessHandler ){
						AccessHandler( fa->fun, fa->fun->name, level, (lform), NULL, NULL );
					}
				}
				if( fa->negate ){
					fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
				}
				else{
					fa->value= fa->sign* fa->fun->value;
				}
				arg[i]= fa->value;
				break;
			case _ascanf_array:
				if( fa->args ){
				  Compiled_Form *faa= fa->args;
				  int ret;
					if( !faa->list_of_constants || (ret= ecl_get_args(fa, faa,level)) ){
						return( (ret<0)? -2 : -1);
					}
					  /* 20040225: I wonder if we shouldn't accept faa->value==fa->fun->N here... */
					if( unused_pragma_likely( faa->value>= -1 && faa->value< fa->fun->N ) ){
						fa->fun->last_index= (int) faa->value;
					}
					else{
						if( Inf(faa->value)> 0 ){
							fa->fun->last_index= fa->fun->N-1;
						}
						else if( Inf(faa->value)< 0 ){
							fa->fun->last_index= 0;
						}
						else if( AllowArrayExpansion && Resize_ascanf_Array( fa->fun, (int) faa->value+1, NULL ) ){
							fa->fun->last_index= (int) faa->value;
						}
						else{
						  /* 20040225: following 2 lines don't resolve this issue? */
/* 							ascanf_arg_error= True;	*/
/* 							correct_constants_args_list( lform, level, __FILE__, __LINE__ );	*/
							if( pragma_unlikely(ascanf_verbose) && faa->value!= -2 ){
								fprintf( StdErr, "#Cecl%d: array %s index %s out-of-range (-1..%d) (line %d)\n",
									(*level)+1, fa->fun->name, ad2str( faa->value, d3str_format, NULL), fa->fun->N,
									__LINE__
								);
							}
							if( fa->fun->N< 0 ){
								fprintf( StdErr, "### This array was deleted. Aborting operations.\n" );
								ascanf_escape= ascanf_interrupt= True;
								*ascanf_escape_value= *ascanf_interrupt_value= 1;
								ascanf_arg_error= 1;
							}
						}
					}
				}
				if( unused_pragma_likely( fa->fun->last_index< fa->fun->N ) ){
					if( fa->fun->iarray ){
						fa->fun->value= (fa->fun->last_index==-1)? fa->fun->N :
							fa->fun->iarray[fa->fun->last_index];
					}
					else{
						fa->fun->value= (fa->fun->last_index==-1)? fa->fun->N :
							fa->fun->array[fa->fun->last_index];
					}
					if( fa->negate ){
						fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
					}
					else{
						fa->value= fa->sign* fa->fun->value;
					}
					arg[i]= fa->value;
				}
				else{
					ascanf_emsg= " (array index out-of-range ( " STRING(__LINE__) ")!! ) ";
					ascanf_arg_error= True;
					if( fa->fun->N< 0 ){
						fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", fa->fun->name );
						ascanf_escape= ascanf_interrupt= True;
						*ascanf_escape_value= *ascanf_interrupt_value= 1;
					}
				}
				break;
			case _ascanf_simplestats:
				EVALUATE_SS_CONSTANTS_LISTS(lform,fa,True);
				if( fa->negate ){
					fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
				}
				else{
					fa->value= fa->sign* fa->fun->value;
				}
				arg[i]= fa->value;
				break;
			case _ascanf_simpleanglestats:
				EVALUATE_SS_CONSTANTS_LISTS(lform,fa,True);
				if( fa->negate ){
					fa->value= (fa->fun->value && !NaN(fa->fun->value))? 0 : 1;
				}
				else{
					fa->value= fa->sign* fa->fun->value;
				}
				arg[i]= fa->value;
				break;
		}
#ifdef VERBOSE_CONSTANTS_EVALUATION
		if( pragma_unlikely(ascanf_verbose) && fa->fun ){
			fprintf( StdErr, "#Cecl%d: \t%s%s",
				(*level)+1, CF_prefix(fa), FUNNAME(fa->fun)
			);
			if( ecl_type== _ascanf_array ){
				fprintf( StdErr, "[%d]", fa->fun->last_index );
			}
			fprintf( StdErr, "==%s\n", ad2str( fa->value, d3str_format, NULL) );
		}
#endif
		fa= (follow_list)? fa->cdr : NULL;
		i+= 1;
	}
	n= i;
	return(n);
evaluate_arguments:;
	return(-1);
}

#define EVALUATE_ARGUMENTS(ln,texpr,larg,ch,data,column,lform,llevel) {\
  int r; \
	if( (*(lform))->list_of_constants && (r= evaluate_constants_list((*(lform)), larg, *(ln), llevel, True))!= -1 ){ \
		*(ln)= r; \
	} \
	else{ \
		if( unused_pragma_unlikely( (*(lform))->list_of_constants ) ){ \
			correct_constants_args_list( (*(lform)), llevel, __FILE__, __LINE__ ); \
		} \
		_compiled_fascanf(ln, texpr, larg, ch, data, column, lform, llevel ); \
	} \
}

/* Evaluates the compiled argument-list to 'Function'	*/
int compiled_ascanf_function( Compiled_Form *form, int Index, char **s, int ind, double *A, char *caller, int *level )
{
#ifdef ASCANF_ALTERNATE
  ascanf_Callback_Frame frame;
#endif
  ascanf_Function *Function= form->fun;
  int ok= 0, ok2= 0, verb=(ascanf_verbose)? 1 : 0, comm= ascanf_comment,
	popp= ascanf_popup, mverb= matherr_verbose, avb= (int) ascanf_verbose, AlInt= *Allocate_Internal,
	anvb= ascanf_noverbose, apl= AllowProcedureLocals;
  Time_Struct *timer= NULL;
  DEFMETHOD( function, ( ASCB_ARGLIST ), int)= Function->function;
  char *name= Function->name;
/*   static int level= 0;	*/
/*   int _ascanf_loop= *ascanf_loop_ptr, *awlp= ascanf_loop_ptr;	*/
  int _ascanf_loop= 0, *awlp= ascanf_loop_ptr, *ali= ascanf_loop_incr,
  	alc= *ascanf_loop_counter, _ascanf_loop_counter= 0, *ailp= ascanf_in_loop,
  	sign= 1, negate= 0;
  double odref= *ascanf_switch_case;
#ifdef DEBUG
  double *ftM= ascanf_forto_MAX, _ftM= -1;
#endif
  static FILE *cfp= NULL, *rcfp= NULL;
  FILE *pSE= StdErr, *cSE= StdErr;
  int cugi= ascanf_use_greek_inf, pugi= ascanf_use_greek_inf;
  static char *tnam= NULL;
  static char lstring[]= "<no textual expression>";
  double *ArgList= af_ArgList->array;
  int Argc= af_ArgList->N, aaa= ascanf_arguments;
#ifdef DEBUG
  int flevel= form->level;
#endif
  ascanf_Function **vlF= vars_local_Functions;
  int *lF= local_Functions;

	if( ascanf_escape ){
		if( !(*level) ){
			*ascanf_escape_value= ascanf_escape= False;
		}
		else{
			return(-1);
		}
	}
	(*level)++;
	ascanf_arg_error= 0;
	ascanf_emsg= NULL;

	  /* *s will typically contain the whole (toplevel) expression. It is not
	   \ kept updated to show the current expression. This has to be retrieved
	   \ by printing <form>, if necessary.
	   */
	if( ! *s ){
		*s= lstring;
	}

	switch( form->special_fun ){
		case matherr_fun:
			matherr_verbose= 1;
			break;
		case global_fun:
			if( AllowProcedureLocals && pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "\n#C%d: \tlocal variable declaration switched off in compiled mode\n", *level );
			}
			*AllowProcedureLocals_value= AllowProcedureLocals= 0;
			vars_local_Functions= NULL;
			local_Functions= NULL;
			break;
		case verbose_fun: if( !ascanf_noverbose ){
		  char hdr[128];
			*ascanf_verbose_value= ascanf_verbose= 1;
			matherr_verbose= -1;
			sprintf( hdr, "#C%d: \t    ", (*level)+1 );
			fprintf( StdErr, "#C%d: \t%s[\n%s", (*level), Function->name, hdr );
			  /* Here we must print <form>, since *s only contains the toplevel expression	*/
			Print_Form( StdErr, &(form->args), 0, True, hdr, NULL, "\n", True );
			fprintf( StdErr, "#C%d: \t] %s\n", (*level), (TBARprogress_header)? TBARprogress_header : "" );
			if( systemtimers && !vtimer && (vtimer= (Time_Struct*) calloc(1, sizeof(Time_Struct))) ){
				Elapsed_Since( vtimer, True );
			}
			break;
		}
		case no_verbose_fun:
			*ascanf_verbose_value= ascanf_verbose= 0;
			matherr_verbose= 0;
			ascanf_noverbose= True;
			break;
		case IDict_fun:
			*Allocate_Internal= True;
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "\n#C%d: \tinternal dictionary access switched on in compiled mode\n", *level );
			}
			break;
		case comment_fun:
		case popup_fun:
			if( !cfp ){
				tnam= XGtempnam( getenv("TMPDIR"), "af_fnc");
				if( tnam ){
					cfp= fopen( (tnam= XGtempnam( getenv("TMPDIR"), "af_fnc")), "wb");
				}
				else{
					cfp = NULL;
				}
				if( cfp ){
				  char hdr[128];
					if( function== ascanf_comment_fnc ){
						ascanf_comment= (*level);
/*						cSE= StdErr;	*/
						StdErr= register_FILEsDescriptor(cfp);
						ascanf_use_greek_inf= True;
					}
					else{
						ascanf_popup= (*level);
						if( *ascanf_popup_verbose && !ascanf_noverbose ){
							*ascanf_verbose_value= ascanf_verbose= 1;
						}
/*						pSE= StdErr;	*/
						StdErr= register_FILEsDescriptor(cfp);
						ascanf_use_greek_inf= True;
					}
					if( !ascanf_noverbose ){
						sprintf( hdr, "#C%d: \t    ", (*level)+1 );
						fprintf( cfp, "#C%d: \t%s[\n%s", (*level), Function->name, hdr );
						  /* Here we must print <form>, since *s only contains the toplevel expression	*/
						Print_Form( cfp, &(form->args), 0, True, hdr, NULL, "\n", True );
						fprintf( cfp, "#C%d: \t] %s\n", (*level), (TBARprogress_header)? TBARprogress_header : "" );
					}
					if( pragma_unlikely(debugFlag) ){
						fprintf( StdErr, "compiled_ascanf_function(): opened temp file \"%s\" as buffer\n",
							tnam
						);
					}
					rcfp= fopen( tnam, "r");
					unlink( tnam );
				}
				else if( pragma_unlikely(debugFlag) ){
					fprintf( StdErr, "compiled_ascanf_function(): can't open temp file \"%s\" as buffer (%s)\n",
						(tnam)? tnam : "<NULL!!>", serror()
					);
				}
				if( tnam ){
					xfree(tnam);
				}
			}
			break;
		case systemtime_fun:
		case systemtime_fun2:
			if( (timer= (Time_Struct*) calloc(1, sizeof(Time_Struct))) ){
				if( ActiveWin && form->last_eval_time> 2.5 ){
					if( form->special_fun== systemtime_fun2 ){
						if( form->parent ){
							TitleMessage( ActiveWin, FUNNAME(form->parent->fun) );
						}
					}
					else if( form->args ){
						TitleMessage( ActiveWin, FUNNAME(form->args->fun) );
					}
				}
				Elapsed_Since( timer, True );
				set_NaN( ascanf_elapsed_values[3] );
				systemtimers += 1;
			}
			break;
	}
	if( form->args ){
	    /* 20020409: moved the ALLOCA() of arg into the block where it is needed, and out-commented
		 \ the allocation of arg_buf that wasn't needed anymore. Also only ALLOCAte as many
		 \ slots as needed (plus 1, otherwise crashes occur?!)
		 */
	  int n= form->alloc_argc;

#ifdef ASCANF_FORM_ARGVALS
	  double *arg;
	  short dealloc;
#elif DEBUG
	  double *arg;
#else
	  ALLOCA( arg, double, n, arg_len);
#endif
	  int j, callback= True, call_callback= True, in_loop_break= 0, ilb_level= -1;
	  Compiled_Form *larg[3]= { NULL, NULL, NULL}, *EF= Evaluating_Form;
#if DEBUG
	  char msgbuf[256]= "";
#endif
#ifdef ASCANF_FORM_ARGVALS
		if( form->level ){
			  /* 20020816: if we're here with non-zero recursion, we shouldn't use the argvals array. This
			   \ would mess up the arguments of the calling expression form.
			   */
			if( !(arg= (double*) calloc( n, sizeof(double) )) ){
				n= 0;
				dealloc= False;
			}
			else{
				  /* RJB 20081208: only deallocate arg when we allocated it. Testing for arg==form->argvals
				   \ will cause a double free if the user closed the window during an evaluation, for instance
				   \ (form->argvals will be NULL, arg will point to the original address).
				   */
				dealloc= True;
			}
		}
		else{
			arg= form->argvals;
			memset( arg, 0, n* sizeof(double) );
			dealloc= False;
		}
#elif DEBUG
		arg= (double*) calloc( n, sizeof(double) );
#else
//		for( arg[0]= 0.0, j= 1; j< n; j++){
//			  /* 20010630: I think that initialising to 0 instead of 1 is a far better idea... */
//			arg[j]= 0.0;
//		}
		memset( arg, 0, n * sizeof(double) );
#endif
		Evaluating_Form= form;
		form->level+= 1;
		if( form->direct_eval ){
			  /* We can directly go to where the arguments are evaluated. No need to test for all
			   \ those special function cases...
			   */
			ascanf_loop_ptr= &_ascanf_loop;
			goto compiled_ascanf_function_parse_args;
		}
		else{
			  /* 20020320: Set direct_eval to True. If we're a special case, it will be unset again...
			   \ I could perform this optimisation in the compiling phase in ascanf_function(). I prefer
			   \ to do it here since it *is* possible that different sets of special-cases tests are performed
			   \ in ascanf_function() and in compiled_ascanf_function().
			   */
			form->direct_eval= True;
		}
		switch( form->special_fun ){
			case ifelse_fun:{
			  int N, NN= n-2;
				form->direct_eval= False;
				larg[0]= form->args;
				larg[1]= larg[0]->cdr;
				larg[2]= (larg[1])? larg[1]->cdr : NULL;
				N= 1;
				n= 1;
				EVALUATE_ARGUMENTS( &n, larg[0]->expr, &arg[0], NULL, NULL, NULL, &larg[0], level );
				if( arg[0] && !NaN(arg[0]) ){
				  /* try to evaluate a second argument	*/
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "#C%d: \tifelse[True]: evaluating 'if' expression\n", *level );
#if DEBUG
						snprintf( msgbuf, sizeof(msgbuf)/sizeof(char), "('if' expression)== " );
#endif
					}
					if( larg[1] ){
						n= 1;
						EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[1], NULL, NULL, NULL, &larg[1], level );
						if( n ){
							N= 2;
						}
#if DEBUG != 2
						*A= arg[1];
						call_callback= False;
#endif
					}
					if( larg[2] && larg[2]->fun ){
						set_NaN( larg[2]->value );
					}
				}
				else if( larg[2] ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "#C%d: \tifelse[False]: evaluating 'else' expression\n", *level );
#if DEBUG
						snprintf( msgbuf, sizeof(msgbuf)/sizeof(char), "('else' expression)== " );
#endif
					}
					set_NaN( arg[1] );
					if( larg[1] && larg[1]->fun ){
						larg[1]->value= arg[1];
					}
					if( larg[2] ){
					  /* try to evaluate the third argument.	*/
						n= NN;
						EVALUATE_ARGUMENTS( &n, FORMNAME(larg[2]), &arg[2], NULL, NULL, NULL, &larg[2], level );
						if( n ){
							N= n+2;
						}
					}
					else{
					  /* default third argument	*/
						N= 3;
						arg[2]= 0.0;
					}
#if DEBUG != 2
					*A= arg[2];
					call_callback= False;
#endif
				}
				n= N;
				break;
			}
			case switch_fun0:{
			  int N, ok= 1, ref= 1;
				form->direct_eval= False;
				larg[0]= form->args;
				larg[1]= larg[0]->cdr;
				larg[2]= (larg[1])? larg[1]->cdr : NULL;
				N= 2;
				n= 2;
				*ascanf_switch_case= (ref- 1)/ 2;
				EVALUATE_ARGUMENTS( &n, larg[0]->expr, &arg[0], NULL, NULL, NULL, &larg[0], level );
				while( arg[0]!= arg[ref] && ok ){
					set_NaN(arg[ref+1]);
					if( larg[2] ){
						if( larg[2]->fun ){
							set_NaN(larg[2]->value);
						}
						larg[1]= larg[2]->cdr;
						larg[2]= (larg[1])? larg[1]->cdr : NULL;
						if( larg[2] ){
							ref+= 2;
							*ascanf_switch_case= (ref- 1)/ 2;
							N+= 2;
							n= 1;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[ref], NULL, NULL, NULL, &larg[1], level );
							ok= 1;
						}
						else{
							ok= 0;
							  /* 20020826: */
							*ascanf_switch_case+= 1;
						}
					}
					else{
						ok= 0;
					}
				}
				if( arg[0]== arg[ref] && larg[2] && ok ){
					N+= 1;
					n= 1;
					  /* The number of the appropriate case: 	*/
					*A= (ref- 1)/ 2;
					EVALUATE_ARGUMENTS( &n, FORMNAME(larg[2]), &arg[ref+1], NULL, NULL, NULL, &larg[2], level );
#if DEBUG != 2
					*A= arg[ref+1];
#if DEBUG
					if( pragma_unlikely(ascanf_verbose) ){
						snprintf( msgbuf, sizeof(msgbuf)/sizeof(char),
							"(case #%d{%d}(%g))== ", (ref-1)/2, (int) *ascanf_switch_case, *A );
					}
#endif
					call_callback= False;
#endif
				}
				else if( larg[1] && !larg[2] ){
					N+= 2;
					  /* The number of the appropriate case: 	*/
					*A= -1;
					EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[N-1], NULL, NULL, NULL, &larg[1], level );
#if DEBUG != 2
					*A= arg[N-1];
#if DEBUG
					if( pragma_unlikely(ascanf_verbose) ){
						snprintf( msgbuf, sizeof(msgbuf)/sizeof(char),
							"(default case #%d{%d}(%g))== ", (N-2)/2, (int) *ascanf_switch_case, *A );
					}
#endif
					call_callback= False;
#endif
				}
				n= N;
				break;
			}
			  /* 20050111 */
			case switch_fun:{
			  int N, ok= 1, hit, ref= 1;
			  ascanf_Function atest, aref;
				form->direct_eval= False;
				larg[0]= form->args;
				larg[1]= larg[0]->cdr;
				larg[2]= (larg[1])? larg[1]->cdr : NULL;
				N= 2;
				n= 2;
				*ascanf_switch_case= (ref- 1)/ 2;
				EVALUATE_ARGUMENTS( &n, larg[0]->expr, &arg[0], NULL, NULL, NULL, &larg[0], level );
				atest.value= arg[0];
				atest.type= _ascanf_value;
				aref.value= arg[1];
				aref.type= _ascanf_value;
				{ ascanf_Function *af= parse_ascanf_address(arg[0], _ascanf_array, "compiled_ascanf_function", 0, NULL );
					if( !af ){
						if( (af= parse_ascanf_address(arg[1], _ascanf_array, "compiled_ascanf_function", 0, NULL )) ){
							aref= *af;
						}
					}
					else{
					  /* test variable is an array pointer: store that fact in atest.type (no
					   \ need to modify atest.value)
					  */
						atest.type= _ascanf_array;
					}
				}
				while( ok && !(hit=SWITCHCCOMPARE(arg)) ){
					set_NaN(arg[ref+1]);
					if( larg[2] ){
						if( larg[2]->fun ){
							set_NaN(larg[2]->value);
						}
						larg[1]= larg[2]->cdr;
						larg[2]= (larg[1])? larg[1]->cdr : NULL;
						if( larg[2] ){
						  ascanf_Function *af;
							ref+= 2;
							*ascanf_switch_case= (ref- 1)/ 2;
							N+= 2;
							n= 1;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[ref], NULL, NULL, NULL, &larg[1], level );
							if( atest.type!=_ascanf_array &&
								(af= parse_ascanf_address(arg[ref], _ascanf_array, "ascanf_function", 0, NULL ))
							){
									aref= *af;
							}
							else{
								aref.value= arg[ref];
								aref.type= _ascanf_value;
							}
							ok= 1;
						}
						else{
							ok= 0;
							  /* 20020826: */
							*ascanf_switch_case+= 1;
						}
					}
					else{
						ok= 0;
					}
				}
				if( larg[2] && ok && hit ){
					N+= 1;
					n= 1;
					  /* The number of the appropriate case: 	*/
					*A= (ref- 1)/ 2;
					EVALUATE_ARGUMENTS( &n, FORMNAME(larg[2]), &arg[ref+1], NULL, NULL, NULL, &larg[2], level );
#if DEBUG != 2
					*A= arg[ref+1];
#if DEBUG
					if( pragma_unlikely(ascanf_verbose) ){
						snprintf( msgbuf, sizeof(msgbuf)/sizeof(char),
							"(case #%d{%d}(%g))== ", (ref-1)/2, (int) *ascanf_switch_case, *A );
					}
#endif
					call_callback= False;
#endif
				}
				else if( larg[1] && !larg[2] ){
					N+= 2;
					  /* The number of the appropriate case: 	*/
					*A= -1;
					EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[N-1], NULL, NULL, NULL, &larg[1], level );
#if DEBUG != 2
					*A= arg[N-1];
#if DEBUG
					if( pragma_unlikely(ascanf_verbose) ){
						snprintf( msgbuf, sizeof(msgbuf)/sizeof(char),
							"(default case #%d{%d}(%g))== ", (N-2)/2, (int) *ascanf_switch_case, *A );
					}
#endif
					call_callback= False;
#endif
				}
				n= N;
				break;
			}
		}
		if( !ascanf_escape ){ int loop_incr= 1; do{
			ascanf_loop_ptr= &_ascanf_loop;
			if( *ascanf_loop_ptr> 0 ){
			  /* We can come back here while evaluating the arguments of
			   \ a while function. We don't want to loop those forever...
			   */
				if( pragma_unlikely(ascanf_verbose) ){
					fputs( " (loop)\n", StdErr );
				}
				*ascanf_loop_ptr= -1;
			}
			switch( form->special_fun ){
				case dowhile_fun:{
					form->direct_eval= False;
					*ascanf_loop_counter= _ascanf_loop_counter;
					ascanf_in_loop= &in_loop_break;
					ilb_level= *level;
					goto compiled_ascanf_function_parse_args;
					break;
				}
				case whiledo_fun:{
				  /* ascanf_whiledo is the C while() construct. If the first
				   \ element, the test, evals to false, the rest of the arguments
				   \ are not evaluated. ascanf_dowhile tests the last argument,
				   \ which is much easier to implement.
				   */
				  int N= n-1;
					form->direct_eval= False;
					larg[0]= form->args;
					larg[1]= larg[0]->cdr;
					n= 1;
					*ascanf_loop_counter= _ascanf_loop_counter;
					ascanf_in_loop= &in_loop_break;
					ilb_level= *level;
					EVALUATE_ARGUMENTS( &n, FORMNAME(larg[0]), &arg[0], NULL, NULL, NULL, &larg[0], level );
					if( arg[0] && !NaN(arg[0]) ){
					  /* test argument evaluated true, now evaluate rest of the args	*/
						if( larg[1] ){
							n= N;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[1], NULL, NULL, NULL, &larg[1], level );
							 /* the actual number of arguments:	*/
							n+= 1;
						}
						else{
							n= 1;
						}
#if DEBUG != 2
						*A= ascanf_progn_return;
						*ascanf_loop_ptr= 1;
						call_callback= False;
#endif
					}
					else{
					  /* test false: we skip the rest	*/
						n= 1;
#if DEBUG != 2
						*A= ascanf_progn_return;
						*ascanf_loop_ptr= 0;
						call_callback= False;
#endif
					}
					break;
				}
				case DCL_fun:{
				  Compiled_Form *af;
				  ascanf_Function *taf;
					form->direct_eval= False;
					larg[0]= form->args;
					af= form->args;
					if( af->argc && af->type== _ascanf_array && af->fun->array ){
					  /* 20001102: An array of doubles with a single argument may reference an element that
					   \ is a pointer to an array. Check that. If not, discard the syntax.
					   */
						if( !af->list_of_constants || evaluate_constants_args_list( af, arg, 1, level )== -1 ){
						  int n= 1;
/* 							if( af->list_of_constants ){	*/
/* 								correct_constants_args_list( af, level, __FILE__, __LINE__ );	*/
/* 							}	*/
							_compiled_fascanf( &n, af->expr, arg, NULL, NULL, NULL, &af, level );
						}
						taf= parse_ascanf_address(af->value, 0, "DCL", (int) ascanf_verbose, NULL );
						if( taf && (taf->type== _ascanf_variable || taf->type== _ascanf_array) ){
							current_DCL_item= taf;
							current_DCL_type= taf->type;
						}
						else{
							ascanf_emsg= " (dereferenced array as first argument to DCL does not contain a valid pointer) ";
							ascanf_arg_error= 1;
							current_DCL_type= _ascanf_novariable;
							callback= False;
						}
					}
					else{
						current_DCL_item= (taf= af->fun);
						current_DCL_type= af->type;
					}
					if( current_DCL_type== _ascanf_variable || current_DCL_type== _ascanf_array
/* 							|| current_DCL_type== _ascanf_simplestats || current_DCL_type== _ascanf_simpleanglestats	*/
							|| current_DCL_type== _ascanf_python_object
					){
						  /* an existing variable. Only change its value when an expression is
						   \ given!
						   */
						arg[0]= taf->value;
						sign= af->sign;
						negate= af->negate;
						if( (larg[1]= af->cdr) ){
							n= n-1;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[1], NULL, NULL, NULL, &larg[1], level );
							if( n ){
								if( current_DCL_type== _ascanf_variable ){
									af->value= arg[1];
									taf->value= arg[1];
									taf->assigns+= 1;
									if( taf->accessHandler ){
										AccessHandler( taf, Function->name, level, form, NULL, NULL );
									}
									n= 2;
								}
								else{
								  int m= (int) arg[1], expanded= 0;
									  /* 981116: shouldn't be difficult to allow expansion in compiled expressions..!	*/
									  /* 20010716: check for car!= NULL, which indicates a user-variable. taf with car==NULL
									   \ are likely to be hardcoded arrays with static memory that SHOULD NOT BE freed.
									   */
									if( unused_pragma_unlikely( n> 1 && m> taf->N && taf->car ) ){
									  int oldN= MAX(0,taf->N);
										taf->N= m;
										if( !(taf->array || taf->iarray) ){
										  /* This should never happen...	*/
											if( taf->name[0]== '%' ){
												taf->iarray= (int*) calloc( taf->N, sizeof(int));
											}
											else{
												taf->array= (double*) calloc( taf->N, sizeof(double));
											}
										}
										else if( !taf->sourceArray ){
										  int i;
											fprintf( StdErr,
												"#C%d \t\"%s\" expanding from %d to %d\n",
												(*level), taf->name, oldN, taf->N
											);
											if( taf->name[0]== '%' ){
												if( !(taf->iarray= (int*) XGrealloc( taf->iarray, taf->N* sizeof(int) ))
												){
													fprintf( StdErr, "PANIC - can't get new memory (%s) - deleting variable\n",
														serror()
													);
													goto compiled_DeleteVariable;
												}
												for( i= oldN; i< taf->N; i++ ){
													taf->iarray[i]= 0;
												}
												expanded= 1;
											}
											else{
												if( !(taf->array= (double*) XGrealloc( taf->array, taf->N* sizeof(double) ))
												){
													fprintf( StdErr, "PANIC - can't get new memory (%s) - deleting variable\n",
														serror()
													);
													goto compiled_DeleteVariable;
												}
												for( i= oldN; i< taf->N; i++ ){
													taf->array[i]= 0;
												}
												expanded= 1;
											}
										}
										else{
											fprintf( StdErr,
												"#C%d \t\"%s\" subsetted array: NOT expanding from %d to %d\n",
												(*level), taf->name, oldN, taf->N
											);
										}
									}
									if( unused_pragma_likely( m>= -1 && m<= taf->N ) )
access_array_element3:
									{
									  int i= 0, j;
										if( n== 1 && m== taf->N ){
											goto compiled_array_bounds_error;
										}
										if( m>= 0 && m< taf->N ){
											if( taf->iarray ){
												for( j= m; j< taf->N && i< n-1; i++, j++ ){
													taf->iarray[j]= (int) arg[i+2];
												}
												af->value= taf->iarray[m];
												taf->value= taf->iarray[m];
											}
											else{
												for( j= m; j< taf->N && i< n-1; i++, j++ ){
													taf->array[j]= arg[i+2];
												}
												af->value= taf->array[m];
												taf->value= taf->array[m];
											}
											if( taf->accessHandler ){
												taf->last_index= m;
												AccessHandler( taf, Function->name, level, form, NULL, NULL );
											}
										}
										else if( (!AllowArrayExpansion && m== taf->N) || (m< 0 && i< n-1) ){
											if( taf->iarray ){
												for( j= 0; j< taf->N && i< n-1; j++ ){
													taf->iarray[j]= (int) arg[i+2];
													  /* 990721: use all specified args, repeating only the last	*/
													if( i< n-2 ){
														i++;
													}
												}
												af->value= taf->iarray[0];
												taf->value= taf->iarray[0];
											}
											else{
												for( j= 0; j< taf->N && i< n-1; j++ ){
													taf->array[j]= arg[i+2];
													if( i< n-2 ){
														i++;
													}
												}
												af->value= taf->array[0];
												taf->value= taf->array[0];
											}
											m= 0;
											if( taf->accessHandler ){
												taf->last_index= m;
												AccessHandler( taf, Function->name, level, form, NULL, NULL );
											}
										}
										else{
											if( i< n-1 ){
												sprintf( ascanf_errmesg, "(assign to index -1 (array-size))" );
												ascanf_emsg= ascanf_errmesg;
												ascanf_arg_error= 1;
											}
											af->value= taf->N;
											taf->value= taf->N;
											taf->reads+= 1;
										}
										taf->last_index= m;
									}
									else if( !expanded ){
										if( Inf(arg[1])> 0 ){
											m= taf->N-1;
											goto access_array_element3;
										}
										else if( Inf(arg[1])< 0 ){
											m= 0;
											goto access_array_element3;
										}
										else if( AllowArrayExpansion && Resize_ascanf_Array( taf, m+1, NULL ) ){
											goto access_array_element3;
										}
compiled_array_bounds_error:;
										sprintf( ascanf_errmesg, "(index out-of-range [%d,%d> (line %d))",
											0, taf->N, __LINE__
										);
										ascanf_emsg= ascanf_errmesg;
										ascanf_arg_error= 1;
										callback= False;
										if( taf->N< 0 ){
											fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", taf->name );
											ascanf_escape= ascanf_interrupt= True;
											*ascanf_escape_value= *ascanf_interrupt_value= 1;
										}
									}
									n+= 1;
								}
								larg[0]->value= af->value;
								larg[0]->fun->assigns+= 1;
							}
						}
						else{
							n= 1;
							taf->reads+= 1;
							larg[0]->fun->reads+= 1;
							larg[0]->value= af->value;
						}
					}
					else if( pragma_unlikely(ascanf_verbose) &&
						(current_DCL_type== _ascanf_simplestats || current_DCL_type== _ascanf_simpleanglestats)
					){
						fprintf( StdErr, "(compiled DCL[] expression on $SS and $SAS stats variables have no effect)" );
					}
					break;
				}
				case DEPROC_fun:{
				  Compiled_Form *af;
					form->direct_eval= False;
					larg[0]= form->args;
					af= form->args;
					current_DCL_type= af->type;
					current_DCL_item= af->fun;
					if( current_DCL_type== _ascanf_procedure && Function->dollar_variable ){
						  /* an existing procedure. All procedures are declared and defined and compiled and etc. during
						   \ compilation, so the same expression DEPROC[name,expr] will, when evaluated in its compiled
						   \ form, just evaluate <expr>.
						   */
						arg[0]= af->fun->value;
						sign= af->sign;
						negate= af->negate;
						if( (larg[1]= af->fun->procedure) ){
							n= 1;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[1], NULL, NULL, NULL, &larg[1], level );
							if( n ){
								if( current_DCL_type== _ascanf_procedure ){
									af->value= arg[1];
									af->fun->value= arg[1];
									if( af->fun->accessHandler ){
										AccessHandler( af->fun, Function->name, level, form, NULL, NULL );
									}
									n= 2;
								}
								larg[0]->value= af->value;
								if( pragma_unlikely(ascanf_verbose) ){
/* 									fprintf( StdErr, "#C%d \t%s== %s\n", *level, af->fun->name, ad2str(arg[1], d3str_format, NULL) );	*/
									Print_ProcedureCall( StdErr, af->fun, level );
									fprintf( StdErr, "== %s\n", ad2str(arg[1], d3str_format, NULL) );
								}
							}
							larg[0]->fun->reads+= 1;
							af->fun->reads+= 1;
							larg[0]->fun->assigns+= 1;
							af->fun->assigns+= 1;
						}
						else{
							n= 1;
							larg[0]->fun->reads+= 1;
							af->fun->reads+= 1;
						}
					}
					break;
				}
				case Delete_fun:{
compiled_DeleteVariable:;
					form->direct_eval= False;
					larg[0]= form->args;
					if( larg[0]->type== _ascanf_variable || larg[0]->type== _ascanf_array || larg[0]->type== _ascanf_procedure ||
						larg[0]->type== _ascanf_simplestats || larg[0]->type== _ascanf_simpleanglestats
						|| larg[0]->type== _ascanf_python_object
					){
						if( larg[0]->fun->name[0]!= '$' ){
							larg[0]->type= _ascanf_novariable;
							arg[0]= larg[0]->fun->value;
							if( pragma_unlikely(ascanf_verbose) ){
								fprintf( StdErr, " (deleted %s A=%d R=%d L=%d)== ",
									larg[0]->fun->name, larg[0]->fun->assigns,
									larg[0]->fun->reads, larg[0]->fun->links
								);
							}
							Delete_Variable( larg[0]->fun );
						}
						else{
							ascanf_emsg= "(not deleteable)";
							ascanf_arg_error= 1;
						}
						n= 1;
					}
					else{
						arg[0]= 0;
						n= 1;
					}
					break;
				}
				case EditPROC_fun:{
					form->direct_eval= False;
					larg[0]= form->args;
					if( larg[0]->type== _ascanf_procedure ){
						n= EditProcedure( larg[0]->fun, /* arg, */ level, larg[0]->expr, NULL );
						if( n<= 0 ){
							n= 1;
						}
					}
					else{
						arg[0]= 0;
						n= 1;
					}
					break;
				}
				case for_to_fun:{
				  /* the C for() construct. The first argument is an initialiser, which is
				   \ initialised only the first time around. The second is the continuation
				   \ test. The rest of the arguments is evaluated only when the second is true.
				   */
				  int N= n-2;
				  int first_ok= 1;
					form->direct_eval= False;
					larg[0]= form->args;
					larg[1]= larg[0]->cdr;
					larg[2]= (larg[1])? larg[1]->cdr : NULL;
					n= 1;
					*ascanf_loop_counter= _ascanf_loop_counter;
					ascanf_in_loop= &in_loop_break;
					ilb_level= *level;
					if( *ascanf_loop_ptr== 0 && larg[1] ){
						EVALUATE_ARGUMENTS( &n, FORMNAME(larg[0]), &arg[0], NULL, NULL, NULL, &larg[0], level );
						if( n && NaN(arg[0]) ){
							first_ok= 0;
							(_ascanf_loop_counter= 0);
						}
						else{
							first_ok= n;
							(_ascanf_loop_counter= (int) arg[0]);
						}
						*ascanf_loop_counter= _ascanf_loop_counter;
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "#%s %d\n",
								(first_ok)? " for_to(init)" : " for_to(invalid init)", (*level)
							);
						}
					}
					if( first_ok && larg[1] ){
						n= 1;
						EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[1], NULL, NULL, NULL, &larg[1], level );
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "#%s %d\n",
								(arg[1])? " for_to(test,true)" : " for_to(test,false)", (*level)
							);
						}
						if( (arg[1] && !NaN(arg[1])) && larg[2] ){
						  /* test argument evaluated true, now evaluate rest of the args	*/
							n= N;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[2]), &arg[2], NULL, NULL, NULL, &larg[2], level );
							 /* the actual number of arguments:	*/
							n+= 2;
						}
						else{
						  /* test false: we skip the rest	*/
							n= 2;
						}
					}
					else{
					  /* test false: we skip the rest	*/
						n= 1;
					}
					break;
				}
				case for_toMAX_fun:{
				  int N= n-2;
				  int first_ok= 1;
					form->direct_eval= False;
					larg[0]= form->args;
					larg[1]= larg[0]->cdr;
					larg[2]= (larg[1])? larg[1]->cdr : NULL;
					n= 1;
					*ascanf_loop_counter= _ascanf_loop_counter;
					ascanf_in_loop= &in_loop_break;
					ilb_level= *level;
					if( *ascanf_loop_ptr== 0 && larg[1] ){
						EVALUATE_ARGUMENTS( &n, FORMNAME(larg[0]), &arg[0], NULL, NULL, NULL, &larg[0], level );
						if( n && NaN(arg[0]) ){
							first_ok= 0;
							(_ascanf_loop_counter= 0);
						}
						else{
							first_ok= n;
							(_ascanf_loop_counter= (int) arg[0]);
						}
						*ascanf_loop_counter= _ascanf_loop_counter;
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "#%s==%s %d\n",
								(first_ok)? " for_toMAX(init)" : " for_toMAX(invalid init)",
								ad2str( arg[0], d3str_format, 0),
								(*level)
							);
						}
					}
					if( first_ok && larg[1] ){
						if( *ascanf_loop_ptr== 0 ){
							n= 1;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[1]), &arg[1], NULL, NULL, NULL, &larg[1], level );
							if( arg[1]< arg[0] ){
								loop_incr= -1;
							}
							ascanf_loop_incr= &loop_incr;
#ifdef DEBUG
							if( n ){
								ascanf_forto_MAX= &arg[1];
								_ftM= arg[1];
							}
							else{
								ascanf_forto_MAX= &_ftM;
							}
#endif
						}
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "#%s:%g|%s argc=%d(%d) level=%d\n",
								fortoMAXok(*ascanf_loop_counter, loop_incr,arg)? " for_toMAX(test,true)" : " for_toMAX(test,false)",
								*ascanf_loop_counter, ad2str( arg[1], d3str_format, NULL),
								N, form->alloc_argc, (*level)
							);
						}
/* 						if( (*ascanf_loop_counter< arg[1] && !NaN(arg[1])) && larg[2] )	*/
						if( fortoMAXok(*ascanf_loop_counter, loop_incr,arg) && larg[2] )
						{
						  /* test argument evaluated true, now evaluate rest of the args	*/
							n= N;
							EVALUATE_ARGUMENTS( &n, FORMNAME(larg[2]), &arg[2], NULL, NULL, NULL, &larg[2], level );
							 /* the actual number of arguments:	*/
							n+= 2;
#if DEBUG != 2
							*A= ascanf_progn_return;
							*ascanf_loop_ptr= 1;
							call_callback= False;
#endif
						}
						else{
						  /* test false: we skip the rest	*/
							n= 2;
#if DEBUG != 2
							*A= ascanf_progn_return;
							*ascanf_loop_ptr= 0;
							call_callback= False;
#endif
						}
					}
					else{
					  /* test false: we skip the rest	*/
						n= 1;
					}
					break;
				}
				case AND_fun:{
				  int i= 1;
				  Compiled_Form *expr= form->args;
					form->direct_eval= False;
					n= 1;
					EVALUATE_ARGUMENTS( &n, expr->expr, &arg[0], NULL, NULL, NULL, &expr, level );
					expr= expr->cdr;
					while( i< form->argc && expr && arg[i-1] && !NaN(arg[i-1]) ){
						n= 1;
						EVALUATE_ARGUMENTS( &n, expr->expr, &arg[i], NULL, NULL, NULL, &expr, level );
						i+= 1;
						expr= expr->cdr;
					}
					n= i;
					break;
				}
				case OR_fun:{
				  int i= 1;
				  Compiled_Form *expr= form->args;
					form->direct_eval= False;
					n= 1;
					EVALUATE_ARGUMENTS( &n, expr->expr, &arg[0], NULL, NULL, NULL, &expr, level );
					expr= expr->cdr;
					while( i< form->argc && expr && (!arg[i-1] || NaN(arg[i-1])) ){
						n= 1;
						EVALUATE_ARGUMENTS( &n, expr->expr, &arg[i], NULL, NULL, NULL, &expr, level );
						i+= 1;
						expr= expr->cdr;
					}
					n= i;
					break;
				}
				default:{
compiled_ascanf_function_parse_args:;
#ifdef DEBUG
					if( form->args ){
						current_function_type= form->args->type;
					}
					else{
						current_function_type= form->type;
					}
#else
					current_function_type= (form->args)? form->args->type : form->type;
#endif
					if( !(larg[0] || larg[1] || larg[2]) ){
					  int nn= n;
						if( form->args->list_of_constants && (nn= evaluate_constants_args_list(form, arg, n, level))!= -1 ){
							n= (nn==-2 && form->args->type==_ascanf_array)? 1 : nn;
						}
						else{
/* evaluate_arguments:;	*/
/* 							if( form->args->list_of_constants ){	*/
/* 								correct_constants_args_list( form->args, level, __FILE__, __LINE__ );	*/
/* 							}	*/
							_compiled_fascanf( &n, ((form->expr)? form->expr : *s), arg, NULL, NULL, NULL, &form->args, level );
							if( n< form->argc ){
								if( pragma_unlikely(ascanf_verbose) ){
									fprintf( StdErr, "#C%d \t%s%s: evaluated only %d elements of a %d element argument list\n",
										(*level), CF_prefix(form), name, n, form->argc
									);
								}
								n= form->argc;
							}
						}
					}
					break;
				}
			}

			if( ascanf_escape ){
				goto next_compiled_loop;
			}

			ascanf_arguments= n;
			if( ascanf_update_ArgList && Function->accessHandler ){
				SET_AF_ARGLIST( arg, n );
			}

			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "#C%d \t%s%s[", (*level), CF_prefix(form), name );
				if( form->args->list_of_constants ){
					fputs( "<C>", StdErr );
				}
				fprintf( StdErr, "%s", ad2str( arg[0], d3str_format, NULL) );
				for( j= 1; j< n; j++){
					fprintf( StdErr, ",%s", ad2str( arg[j], d3str_format, NULL) );
				}
				for( ;j< form->argc; j++ ){
					fprintf( StdErr, ",?" );
				}
				fprintf( StdErr, "]== ");
#if DEBUG
				if( *msgbuf ){
					fputs( msgbuf, StdErr );
				}
#endif
				fflush( StdErr );
			}
			if( ascanf_comment && cfp ){
				fprintf( cfp,    "#C%d \t%s%s[%s", (*level), CF_prefix(form), name, ad2str( arg[0], d3str_format, NULL) );
				for( j= 1; j< n; j++){
					fprintf( cfp, ",%s", ad2str( arg[j], d3str_format, NULL) );
				}
				for( ;j< form->argc; j++ ){
					fprintf( cfp, ",?" );
				}
				fprintf( cfp, "]== ");
#if DEBUG
				if( msgbuf ){
					fputs( msgbuf, cfp );
				}
#endif
			}
			if( function== ascanf_Variable || (Function->type== _ascanf_array && function== ascanf_DeclareVariable) ){
				switch( Function->type ){
					case _ascanf_variable:
					case _ascanf_python_object:
						ok= (n> 0);
						Function->value= arg[0];
						*A= arg[0];
						if( Function->accessHandler ){
							AccessHandler( Function, Function->name, level, form, NULL, A );
						}
						Function->assigns+= 1;
						if( negate ){
							(*ascanf_current_value)= (*A= (*A && !NaN(*A))? 0 : 1);
						}
						else{
							(*ascanf_current_value)= (*A*= sign);
						}
						form->value= *ascanf_current_value;
						break;
					case _ascanf_array:{
					  int first= (function== ascanf_DeclareVariable)? 1 : 0;
					  int m= (int) arg[first];
						ok= (n> first);
						if( ok ){
							if( unused_pragma_likely( m>= -2 && m< (Function->N + ((AllowArrayExpansion)?0:1)) ) ){
access_array_element4:;
								ASCANF_ARRAY_ACCESS( form, Function, first, m, n, arg, A, level );
							}
							else{
								if( Inf(arg[first])> 0 ){
									m= Function->N-1;
									goto access_array_element4;
								}
								else if( Inf(arg[first])< 0 ){
									m= 0;
									goto access_array_element4;
								}
								else if( AllowArrayExpansion && Resize_ascanf_Array( Function, m+1, NULL ) ){
									goto access_array_element4;
								}
								sprintf( ascanf_errmesg, "(index out-of-range [%d,%d> (line %d))",
									0, Function->N, __LINE__
								);
								ascanf_emsg= ascanf_errmesg;
								ascanf_arg_error= 1;
								set_NaN(*A);
								if( Function->N< 0 ){
									fprintf( StdErr, "### Array \"%s\" was deleted. Aborting operations.\n", Function->name );
									ascanf_escape= ascanf_interrupt= True;
									*ascanf_escape_value= *ascanf_interrupt_value= 1;
								}
							}
							{ /* 20050415: moved here from the if (not else) statement above */
								if( negate ){
									(*ascanf_current_value)= (*A= (*A && !NaN(*A))? 0 : 1);
								}
								else{
									(*ascanf_current_value)= (*A *= sign);
								}
								form->value= *ascanf_current_value;
							}
						}
						else{
							ascanf_arg_error= 1;
						}
						break;
					}
					case _ascanf_simplestats:{
						ok= (n> 0);
						if( !ascanf_SyntaxCheck ){
							if( Function->N> 1 || (form->args && form->args->take_address && form->args->type== _ascanf_array) ){
							  int ac= ascanf_arguments;
							  double args[3];
								ascanf_arguments= 2;
								args[1]= arg[0];
								args[2]= 1;
								ascanf_SS_set_bin( Function->SS, 0, Function->name, args, A, 0, Function );
								ascanf_arguments= ac;
							}
							else{
								SS_Add_Data( Function->SS, 1, arg[0], 1.0 );
							}
							Function->value= Function->SS->last_item;
						}
						else{
							Function->value= arg[0];
						}
						*A= Function->value;
						Function->assigns+= 1;
						if( Function->accessHandler ){
							AccessHandler( Function, Function->name, level, form, NULL, A );
						}
						if( negate ){
							(*ascanf_current_value)= (*A= (*A && !NaN(*A))? 0 : 1);
						}
						else{
							(*ascanf_current_value)= (*A*= sign);
						}
						form->value= *ascanf_current_value;
						if( pragma_unlikely(ascanf_verbose) ){
							if( Function->N> 1 ){
								fprintf( StdErr, " (SS_Add[@[&%s,%d],1,%s,1])== ", Function->name, Function->last_index,
									ad2str( Function->value, d3str_format, 0)
								);
							}
							else{
								fprintf( StdErr, " (SS_Add[&%s,1,%s,1])== ", Function->name,
									ad2str( Function->value, d3str_format, 0)
								);
							}
						}
						break;
					}
					case _ascanf_simpleanglestats:{
						ok= (n> 0);
						if( !ascanf_SyntaxCheck ){
							if( form->args && form->args->take_address && form->args->type== _ascanf_array ){
							  int ac= ascanf_arguments;
							  double args[6];
								ascanf_arguments= 6;
								args[1]= arg[0];
								args[2]= 1;
								args[3]= Function->SAS->Gonio_Base;
								args[4]= Function->SAS->Gonio_Offset;
								if( n> 1 ){
									args[5]= arg[1];
								}
								ascanf_SAS_set_bin( Function->SAS, 0, Function->name, args, A, 0, Function );
								ascanf_arguments= ac;
							}
							else{
								SAS_Add_Data( Function->SAS, 1, arg[0], 1.0,
									(n>1)? ASCANF_TRUE(arg[1]) : ASCANF_TRUE(*SAS_converts_angle) );
							}
							Function->value= Function->SAS->last_item;
						}
						else{
							Function->value= arg[0];
						}
						*A= Function->value;
						Function->assigns+= 1;
						if( Function->accessHandler ){
							AccessHandler( Function, Function->name, level, form, NULL, A );
						}
						if( negate ){
							(*ascanf_current_value)= (*A= (*A && !NaN(*A))? 0 : 1);
						}
						else{
							(*ascanf_current_value)= (*A*= sign);
						}
						form->value= *ascanf_current_value;
						if( pragma_unlikely(ascanf_verbose) && Function->SAS ){
							fprintf( StdErr, " (SAS_Add[&%s,1,%s,1,%s,%s,%s=%d])== ", Function->name,
								ad2str( Function->value, d3str_format, 0),
								ad2str( Function->SAS->Gonio_Base, d3str_format, 0),
								ad2str( Function->SAS->Gonio_Offset, d3str_format, 0),
								(n>1)? "arg" : "$SAS_converts_angle",
								ASCANF_TRUE_((n>1)? arg[1] : *SAS_converts_angle)
							);
						}
						break;
						break;
					}
					case _ascanf_novariable:
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "DELETED " );
							ok2= False;
						}
						break;
				}
			}
			else if( callback ){
				current_function_type= form->args->type;
#ifdef ASCANF_ALTERNATE
				frame.args= arg;
				frame.result= A;
				frame.level= level;
				frame.compiled= form;
				frame.self= Function;
#if defined(ASCANF_ARG_DEBUG)
				frame.expr= form->expr;
#endif
				ok= (call_callback)? (*function)( &frame ) : True;
#else
				ok= (call_callback)? (*function)( arg, A, level ) : True;
#endif
				if( Function->accessHandler ){
					AccessHandler( Function, Function->name, level, form, NULL, A );
				}
				if( negate ){
					(*ascanf_current_value)= Function->value= (*A= (*A && !NaN(*A))? 0 : 1);
				}
				else{
					(*ascanf_current_value)= Function->value= (*A *= sign);
				}
				form->value= *ascanf_current_value;
				Function->reads+= 1;
			}
next_compiled_loop:;
			ascanf_check_event( "compiled_ascanf_function" );
			_ascanf_loop_counter+= loop_incr;
		} while( *ascanf_loop_ptr> 0 && !ascanf_escape ); }
		if( ascanf_interrupt ){
		  char hdr[128];
			fprintf( StdErr, "Processing was interrupted:\n" );
			if( form->top!= form ){
				sprintf( hdr, "#C%d: \t    ", (*level)+1 );
				fprintf( StdErr, "#C%d: \t%s[\n%s", (*level), Function->name, hdr );
				  /* Here we must print <form>, since *s only contains the toplevel expression	*/
				Print_Form( StdErr, &(form->args), 1, True, hdr, NULL, "\n", True );
				fprintf( StdErr, "#C%d: \t] in toplevel expression\n", (*level) );
			}
			strcpy( hdr, "#C0: \t    " );
			if( form->top ){
				fputs( "#C0:  \t", StdErr );
				Print_Form( StdErr, &(form->top), 0, True, hdr, NULL, "\n", True );
				fputs( "#C0:  \tContext: ", StdErr );
			}
			else{
				fprintf( StdErr, "%s<UNKNOWN>", hdr );
			}
			fprintf( StdErr, " %s\n", (TBARprogress_header)? TBARprogress_header : "" );
			ascanf_interrupt= False;
		}
		ascanf_loop_ptr= awlp;
		ascanf_loop_incr= ali;
		*ascanf_loop_counter= alc;
		*ascanf_switch_case= odref;
#ifdef DEBUG
		ascanf_forto_MAX= ftM;
#endif
		if( ascanf_in_loop && ilb_level== *level ){
			if( *ascanf_in_loop && ascanf_escape ){
				*ascanf_escape_value= ascanf_escape= False;
			}
			ascanf_in_loop= ailp;
		}
		ok2= 1;
#ifdef ASCANF_FORM_ARGVALS
		form->level-= 1;
/* 		if( form->level ){	*/
/* 			xfree( arg );	*/
/* 		}	*/
		if( dealloc ){
			xfree( arg );
		}
#elif DEBUG
		xfree( arg );
#endif
		Evaluating_Form= EF;
	}
	else{
	  Compiled_Form *EF= Evaluating_Form;
	  /* there was no argument list	*/
		Evaluating_Form= form;
		ascanf_arguments= 0;
		if( ascanf_update_ArgList && Function->accessHandler ){
			SET_AF_ARGLIST( ascanf_ArgList, 0 );
		}

		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "#C%d \t%s%s== ", (*level), CF_prefix(form), name );
			fflush( StdErr );
		}
		if( ascanf_comment && cfp ){
			fprintf( cfp,    "#C%d \t%s%s== ", (*level), CF_prefix(form), name );
		}
		if( function== ascanf_Variable ){
		  /* In principle, we do never get here.	*/
			if( Function->type== _ascanf_variable || Function->type== _ascanf_array ||
				Function->type== _ascanf_simplestats || Function->type== _ascanf_simpleanglestats
				|| Function->type== _ascanf_python_object
			){
				ok= 1;
				*A= Function->value;
			}
		}
		else{
#ifdef ASCANF_ALTERNATE
			frame.args= NULL;
			frame.result= A;
			frame.level= level;
			frame.compiled= form;
			frame.self= Function;
#if defined(ASCANF_ARG_DEBUG)
			frame.expr= form->expr;
#endif
			ok= (*function)( &frame );
#else
			ok= (*function)( NULL, A, level );
#endif
			Function->value= *A;
			if( Function->accessHandler ){
				AccessHandler( Function, Function->name, level, form, NULL, A );
			}
		}
		Function->reads+= 1;
		(*ascanf_current_value)= *A;
		form->value= *A;
		ok2= 1;
		ascanf_check_event( "compiled_ascanf_function");
		Evaluating_Form= EF;
	}
	if( ok2){
		if( form->store && form->args && (form->args->type== _ascanf_array || form->args->type== _ascanf_variable) ){
		  ascanf_Function *af= form->args->fun;
		  int target= (form->args->args)? (int) form->args->args->value : -1;
			if( af->type!= _ascanf_novariable ){
				af->value= *A;
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(stored in %s", af->name );
				}
				if( af->type== _ascanf_array && (af->iarray || af->array) && unused_pragma_likely(target>= 0 && target< af->N) ){
					if( af->iarray ){
						af->iarray[target]= (int) *A;
					}
					else{
						af->array[target]= *A;
					}
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "[%d])", target );
					}
				}
				else if( pragma_unlikely(ascanf_verbose) ){
					fputs( ")== ", StdErr);
				}
				if( af->accessHandler ){
					AccessHandler( af, Function->name, level, form, NULL, NULL );
				}
			}
			else if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "(nothing stored in DELETED variable)" );
			}
		}
		if( pragma_unlikely(ascanf_verbose) ){
			if( form->args && form->args->fun ){
				if( form->args->fun->type== _ascanf_array ){
				  int j;
					fprintf( StdErr, "(1st arg " );
					if( form->args->fun->iarray ){
						fprintf( StdErr, "%s%s{%d",
							CF_prefix_sub(form->args),
							form->args->fun->name, form->args->fun->iarray[0]
						);
						if( !form->args->take_address ){
							for( j= 1; j< form->args->fun->N && (form->args->fun->last_index== -1 || j< 48); j++ ){
								fprintf( StdErr, ",%d", form->args->fun->iarray[j]);
							}
							if( form->args->fun->N> j ){
								fputs( ",...", StdErr );
								for( j= form->args->fun->N-3; j< form->args->fun->N; j++ ){
									fprintf( StdErr, ",%d", form->args->fun->iarray[j] );
								}
							}
						}
						else{
							fprintf( StdErr, ",..." );
						}
					}
					else{
						fprintf( StdErr, "%s%s{%s",
							CF_prefix_sub(form->args),
							form->args->fun->name,
							ad2str( form->args->fun->array[0], d3str_format, NULL)
						);
						if( !form->args->take_address ){
							for( j= 1; j< form->args->fun->N && (form->args->fun->last_index== -1 || j< 48); j++ ){
								fprintf( StdErr, ",%s",
									ad2str( form->args->fun->array[j], d3str_format, NULL)
								);
							}
							if( form->args->fun->N> j ){
								fputs( ",...", StdErr );
								for( j= form->args->fun->N-3; j< form->args->fun->N; j++ ){
									fprintf( StdErr, ",%s",
										ad2str( form->args->fun->array[j], NULL, NULL)
									);
								}
							}
						}
						else{
							fprintf( StdErr, ",..." );
						}
					}
					if( form->args->fun->value!= *A ){
						fprintf( StdErr, "}[%d:%d]==%s) == ",
							form->args->fun->last_index, form->args->fun->N, ad2str( form->args->fun->value, d3str_format, NULL) );
					}
					else{
						fprintf( StdErr, "}[%d:%d]) == ", form->args->fun->last_index, form->args->fun->N );
					}
				}
				else if( form->args->fun->type== _ascanf_simplestats ){
				  char buf[256];
					if( form->args->fun->N> 1 ){
						if( form->args->fun->last_index>= 0 && form->args->fun->last_index< form->args->fun->N ){
							fprintf( StdErr, "[%d]", form->args->fun->last_index );
							SS_sprint_full( buf, d3str_format, " #xb1 ", 0, &form->args->fun->SS[form->args->fun->last_index] );
						}
						else if( form->args->fun->last_index== -1 ){
							sprintf( buf, "%d", form->args->fun->N );
						}
					}
					else{
						SS_sprint_full( buf, d3str_format, " #xb1 ", 0, form->args->fun->SS );
					}
					fprintf( StdErr, " (1st arg %s:%s)== ", form->args->fun->name, buf );
				}
				else if( form->args->fun->type== _ascanf_simpleanglestats ){
				  char buf[256];
					SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, form->args->fun->SAS );
					fprintf( StdErr, " (1st arg %s:%s)== ", form->args->fun->name, buf );
				}
				if( form->args->fun->cfont ){
					fprintf( StdErr, "  (1st arg cfont %s: %s\"%s\"/%s[%g])== ",
						form->args->fun->name,
						(form->args->fun->cfont->is_alt_XFont)? "(alt.) " : "",
						form->args->fun->cfont->XFont.name, form->args->fun->cfont->PSFont, form->args->fun->cfont->PSPointSize
					);
				}
			}
			fprintf( StdErr, "%s",
				(form->special_fun== systemtime_fun || form->special_fun== systemtime_fun2)? "<delayed!>"
					: ad2str( *A, d3str_format, NULL)
			);
			if( ascanf_arg_error ){
				if( ascanf_emsg ){
					fprintf( StdErr, " %s", ascanf_emsg );
				}
				else{
					fprintf( StdErr, " (needs %d arguments, has %d/%d)", Function_args(Function), ascanf_arguments, form->argc );
				}
			}
			if( vtimer ){
				Elapsed_Since( vtimer, True );
				fprintf( StdErr, " <%g:%gs>", vtimer->Time + vtimer->sTime, vtimer->HRTot_T );
			}
			fprintf( StdErr, "%s\n", ((*level)== 1)? "\t  ," : "\t->" );
			fflush( StdErr);
		}
		if( ascanf_comment && cfp ){
			if( form->args && form->args->fun ){
				if( form->args->fun->type== _ascanf_array ){
				  int j;
					if( form->args->fun->iarray ){
						fprintf( cfp, "%s%s{%d",
							CF_prefix_sub(form->args),
							form->args->fun->name, form->args->fun->iarray[0]
						);
						for( j= 1; j< form->args->fun->N; j++ ){
							fprintf( cfp, ",%d", form->args->fun->iarray[j]);
						}
					}
					else{
						fprintf( cfp, "%s%s{%s",
							CF_prefix_sub(form->args),
							form->args->fun->name,
							ad2str( form->args->fun->array[0], d3str_format, NULL)
						);
						for( j= 1; j< form->args->fun->N; j++ ){
							fprintf( cfp, ",%s",
								ad2str( form->args->fun->array[j], d3str_format, NULL)
							);
						}
					}
					if( form->args->fun->value!= *A ){
						fprintf( cfp, "}[%d]==%s == ", form->args->fun->last_index, ad2str( form->args->fun->value, d3str_format, NULL) );
					}
					else{
						fprintf( cfp, "}[%d]== ", form->args->fun->last_index );
					}
				}
				else if( form->args->fun->type== _ascanf_simplestats ){
				  char buf[256];
					if( form->args->fun->N> 1 ){
						if( form->args->fun->last_index>= 0 && form->args->fun->last_index< form->args->fun->N ){
							fprintf( StdErr, "[%d]", form->args->fun->last_index );
							SS_sprint_full( buf, d3str_format, " #xb1 ", 0, &form->args->fun->SS[form->args->fun->last_index] );
						}
						else if( form->args->fun->last_index== -1 ){
							sprintf( buf, "%d", form->args->fun->N );
						}
					}
					else{
						SS_sprint_full( buf, d3str_format, " #xb1 ", 0, form->args->fun->SS );
					}
					fprintf( StdErr, " (%s)== ", buf );
				}
				else if( form->args->fun->type== _ascanf_simpleanglestats ){
				  char buf[256];
					SAS_sprint_full( buf, d3str_format, " #xb1 ", 0, form->args->fun->SAS );
					fprintf( StdErr, " (%s)== ", buf );
				}
				if( form->args->fun->cfont ){
					fprintf( StdErr, "  (cfont: %s\"%s\"/%s[%g])== ",
						(form->args->fun->cfont->is_alt_XFont)? "(alt.) " : "",
						form->args->fun->cfont->XFont.name, form->args->fun->cfont->PSFont, form->args->fun->cfont->PSPointSize
					);
				}
			}
			fprintf( cfp, "%s%s",
				(form->special_fun== systemtime_fun || form->special_fun== systemtime_fun2)? "<delayed!>"
					: ad2str( *A, d3str_format, NULL),
				((*level)== 1)? "\t  ," : "\t->"
			);
			if( ascanf_arg_error ){
				if( ascanf_emsg ){
					fprintf( cfp, " %s", ascanf_emsg );
				}
				else{
					fprintf( cfp, " (needs %d arguments, has %d/%d)", Function_args(Function), ascanf_arguments, form->argc );
				}
			}
			fputc( '\n', cfp );
			fflush( cfp );
		}
	}
	else if( ascanf_arg_error ){
		if( ascanf_emsg ){
			fprintf( StdErr, "%s== %s %s\n", name, ad2str( *A, d3str_format, NULL), ascanf_emsg );
		}
		else{
			fprintf( StdErr, "%s== %s needs %d arguments, has %d/%d\n",
				name, ad2str( *A, d3str_format, NULL), Function_args(Function), ascanf_arguments, form->argc
			);
		}
		fflush( StdErr );
	}
	switch( form->special_fun ){
		case verbose_fun:
			if( vtimer ){
				xfree(vtimer);
			}
		case no_verbose_fun:
		case matherr_fun:
			*ascanf_verbose_value= ascanf_verbose= verb;
			matherr_verbose= mverb;
			ascanf_noverbose= anvb;
			break;
		case global_fun:
			*AllowProcedureLocals_value= AllowProcedureLocals= apl;
			vars_local_Functions= vlF;
			local_Functions= lF;
			break;
		case IDict_fun:
			*Allocate_Internal= AlInt;
			if( ascanf_verbose ){
				fprintf( StdErr, "\n#C%d: \tinternal dictionary access switched off in compiled mode\n", *level );
			}
			break;
		case comment_fun:
		case popup_fun:
			if( ((function== ascanf_comment_fnc && ascanf_comment== (*level)) ||
					(function== ascanf_popup_fnc && ascanf_popup== (*level))) && cfp
			){
			  extern char* add_comment();
			  char buf[1024], *c;
			  Sinc List;
				if( ascanf_popup== (*level) ){
					List.sinc.string= NULL;
					Sinc_string_behaviour( &List, NULL, 0,0, SString_Dynamic );
					Sflush( &List );
					StdErr= register_FILEsDescriptor(pSE);
				}
				fflush( cfp );
				if( rcfp ){
					while( (c= fgets(buf, 1023, rcfp)) && !feof(rcfp) ){
						if( function== ascanf_popup_fnc ){
							Add_SincList( &List, c, False );
						}
						else{
							add_comment( c );
						}
					}
				}
				else if( pragma_unlikely(debugFlag) ){
					fprintf( StdErr, "compiled_ascanf_function(): couldn't open temp file \"%s\" for reading (%s)\n",
						tnam, serror()
					);
				}
				if( ascanf_comment== (*level) ){
					if( !ascanf_popup ){
						fclose( cfp);
						cfp= NULL;
						if( rcfp ){
							fclose( rcfp);
							rcfp= NULL;
						}
					}
						StdErr= register_FILEsDescriptor(cSE);
					ascanf_comment= comm;
					ascanf_use_greek_inf= cugi;
				}
				if( ascanf_popup== (*level) ){
				  int id;
				  char *sel= NULL;
					if( HAVEWIN ){
						if( popup_menu ){
							xtb_popup_delete( &popup_menu );
						}
						id= xtb_popup_menu( USEWINDOW, List.sinc.string, "Scope's output", &sel, &popup_menu);
						if( sel ){
							while( *sel && isspace( (unsigned char) *sel) ){
								sel++;
							}
						}
						if( sel && *sel ){
							if( pragma_unlikely(debugFlag) ){
								xtb_error_box( USEWINDOW, sel, "Copied to clipboard:" );
							}
							else{
								Boing(10);
							}
							XStoreBuffer( disp, sel, strlen(sel), 0);
							  // RJVB 20081217
							xfree(sel);
						}
					}
					else{
						fputs( List.sinc.string, StdErr );
					}
					xfree( List.sinc.string );
					if( !ascanf_comment && cfp ){
						fclose( cfp);
						cfp= NULL;
						if( rcfp ){
							fclose( rcfp);
							rcfp= NULL;
						}
					}
					ascanf_popup= popp;
					ascanf_use_greek_inf= pugi;
					*ascanf_verbose_value= ascanf_verbose= avb;
				}
		/* 		unlink( tnam );	*/
				xfree( tnam );
			}
			break;
		case systemtime_fun2:
		case systemtime_fun: if( timer ){
		  char *fname= NULL;
			Elapsed_Since( timer, True );
			form->value= (*ascanf_current_value)= Function->value= *A= timer->Time;
			ascanf_elapsed_values[0]= Delta_Tot_T;
			ascanf_elapsed_values[1]= timer->Time;
			ascanf_elapsed_values[2]= timer->sTime;
			if( Function->accessHandler ){
				AccessHandler( Function, Function->name, level, NULL, *s, A );
			}
			if( form->special_fun== systemtime_fun2 ){
				if( form->parent ){
					fname= FUNNAME(form->parent->fun);
				}
			}
			else if( form->args ){
				fname= FUNNAME(form->args->fun);
			}
			if( fname ){
				fprintf( StdErr, "#C# Evaluation time '%s': user %ss, system %ss, total %ss (%s%% CPU)",
					fname,
					ad2str( timer->Time, d3str_format, NULL),
					ad2str( timer->sTime, d3str_format, NULL),
					ad2str( timer->HRTot_T, d3str_format, NULL),
					ad2str( 100.0* (timer->Time+timer->sTime)/Delta_Tot_T, d3str_format,NULL)
				);
			}
			else{
				fprintf( StdErr, "#C# Evaluation time: user %ss, system %ss, total %ss (%s%% CPU)",
					ad2str( timer->Time, d3str_format, NULL),
					ad2str( timer->sTime, d3str_format, NULL),
					ad2str( timer->HRTot_T, d3str_format, NULL),
					ad2str( 100.0* (timer->Time+timer->sTime)/Delta_Tot_T, d3str_format, NULL)
				);
			}
			if( ascanf_elapsed_values[3] && !NaNorInf(ascanf_elapsed_values[3]) ){
			  char *prefix= "";
			  double flops= ascanf_elapsed_values[3]/ Delta_Tot_T;
				if( flops> 1e6 ){
					flops*= 1e-6;
					prefix= "M";
				}
				fprintf( StdErr, "; %s \"%sflops\"", ad2str( flops, d3str_format, 0), prefix );
				set_NaN(ascanf_elapsed_values[3]);
			}
			fputs( "\n", StdErr );
			xfree(timer);
			if( ActiveWin && form->last_eval_time> 2.5 ){
				TitleMessage( ActiveWin, NULL );
			}
			form->last_eval_time= Delta_Tot_T;
			if( systemtimers > 0 ){
				systemtimers -= 1;
			}
			break;
		}
	}
	SET_AF_ARGLIST( ArgList, Argc );
	ascanf_arguments= aaa;
	ascanf_in_loop= ailp;

#ifdef DEBUG
	if( form->level!= flevel ){
		fprintf( StdErr, "Warning: entered form 0x%lx with level %d, exit with level %d; \"%s\"\n",
			form, flevel, form->level, (form->expr)? form->expr : ((s)? *s : "<unknown>")
		);
	}
#endif

	(*level)--;
	GCA();
	return( ok);
}

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
int call_compiled_ascanf_function( int Index, char **s, double *result, int *ok, char *caller, Compiled_Form **Form, int *level )
{  int len= 0;
   Compiled_Form *form;

	if( !Form || !*Form ){
		return( check_for_ascanf_function( Index, s, result, ok, caller, NULL, NULL, level ) );
	}
	form= *Form;

	if( reset_ascanf_index_value ){
		(*ascanf_index_value)= (double) Index;
	}
	if( reset_ascanf_currentself_value ){
		(*ascanf_self_value)= *result;
		(*ascanf_current_value)= *result;
	}
	if( form->fun ){
#ifdef DEBUG
	  ascanf_Function *ffun= form->fun;
#endif
		*ok= compiled_ascanf_function( form, Index, s, len, result, caller, level );
		if( form->negate ){
			*result= (*result && !NaN(*result))? 0 : 1;
		}
		else{
			*result*= form->sign;
		}
#ifdef DEBUG
		if( form->fun != ffun ){
			fprintf( StdErr, "call_compiled_ascanf_function(): form calling \"%s\" mutated its callback!!\n",
				ffun->name
			);
			fflush(StdErr);
			form->fun= ffun;
		}
#endif
		switch( form->fun->type){
			case NOT_EOF:
				return( (*ok!= EOF) );
				break;
			case _ascanf_python_object:
			case NOT_EOF_OR_RETURN:
				return( (*ok== EOF)? 0 : *ok );
				break;
			default:
				return( *ok);
				break;
		}
	}
	else{
		*ok= 1;
		*result= form->value;
		return( 1 );
	}
	return(0);
}

/* read multiple floating point values, uses 'Form' instead of 's'	*/
static int _compiled_fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **Form, int *level )
{	int i= 0, r= 0, j= 1;
	double A;
	Compiled_Form *form;
	double DATA[4];
	int Nset= 0;
	int raiv= reset_ascanf_index_value, racv= reset_ascanf_currentself_value;
	int psiU= param_scratch_inUse, using_ps= 0;
/* 	double *ArgList= af_ArgList->array;	*/
/* 	int Argc= af_ArgList->N;	*/

	if( ascanf_escape ){
		if( !(*level) ){
			*ascanf_escape_value= ascanf_escape= False;
		}
		else{
			return(-1);
		}
	}
	if( a== param_scratch ){
		if( *n> param_scratch_len ){
		  /* When using clean_param_scratch() before passing param_scratch as scratch memory,
		   \ this test should never be necessary!
		   */
			*n= param_scratch_len;
		}
		param_scratch_inUse= True;
		using_ps= True;
	}

	form= *Form;

	if( data ){
		ascanf_data_buf= data;
		DATA[0]= *ascanf_data0;
		DATA[1]= *ascanf_data1;
		DATA[2]= *ascanf_data2;
		DATA[3]= *ascanf_data3;
		Nset= 1;
		*ascanf_data0= data[0];
		*ascanf_data1= data[1];
		*ascanf_data2= data[2];
		*ascanf_data3= data[3];
	}
/* 	if( ascanf_update_ArgList ){	*/
/* 		SET_AF_ARGLIST( a, *n );	*/
/* 	}	*/
	if( column ){
		ascanf_column_buf= column;
	}
	if( !(*level)){
		reset_ascanf_index_value= True;
		  /* 991014: reset error flag on toplevel...	*/
		ascanf_arg_error= 0;
	}
	else{
		reset_ascanf_currentself_value= False;
	}
/* 	(*level)++;	*/
	while( form && i< *n && j!= EOF && !ascanf_escape ){
		RESET_CHANGED_FLAG( ch, i);
		A= *a;
		if( ! form->fun ){
			if( form->negate ){
				A= (form->value && !NaN(form->value))? 0 : 1;
			}
			else{
				A= form->sign* form->value;
			}
			r+= (j=1);
		}
		else if( form->last_value ){
			if( form->negate ){
				A= (form->fun->value && !NaN(form->value))? 0 : 1;
			}
			else{
				A= form->sign* form->fun->value;
			}
			r+= (j=1);
		}
		else if( form->take_address ){
			if( form->take_usage /* || form->fun->is_usage */ ){
				form->fun->take_usage= True;
			}
			else{
				form->fun->take_usage= False;
			}
			  /* update the elements of this automatic array if they are not constants:
			   */
			if( form->fun->internal && form->fun->procedure && form->fun->type== _ascanf_array && !form->fun->user_internal &&
				!AlwaysUpdateAutoArrays && form->fun->procedure->list_of_constants>=0
			){
			  int n= form->fun->N;
				if( pragma_unlikely(ascanf_verbose>1) ){
					fprintf( StdErr, "#C%d: updating automatic array \"%s\", expression %s\n",
						(*level), form->fun->name, form->fun->procedure->expr
					);
				}
				form->fun->procedure->level+= 1;
				_compiled_fascanf( &n, form->fun->procedure->expr, form->fun->array, NULL, NULL, NULL, &form->fun->procedure, level );
				form->fun->procedure->level-= 1;
			}
			form->value= take_ascanf_address( form->fun );
			A= form->value;
			r+= (j=1);
		}
		else if( form->fun->function== ascanf_Variable ){
			if( form->type== _ascanf_python_object && dm_python
// 20080916: check commented-out
// 				&& (form->args || form->empty_arglist)
			){
				if( form->fun->type== _ascanf_python_object ){
				  Boolean ok= False;
				  int pn= form->alloc_argc;
				  double *args;
					if( form->args ){
						if( form->level ){
							  /* 20020816: if we're here with non-zero recursion, we shouldn't use the argvals array. This
							   \ would mess up the arguments of the calling expression form.
							   */
							if( !(args= (double*) calloc( pn, sizeof(double) )) ){
								fprintf( StdErr, "%s: argument-list allocation error (%s) (executing without arguments!); %s::%d\n",
									form->top->af->name, serror(), __FILE__, __LINE__
								);
								pn= 0;
							}
						}
						else{
							args= form->argvals;
							memset( args, 0, pn* sizeof(double) );
						}
						form->level+= 1;
						_compiled_fascanf( &pn, form->fun->name, args, NULL, NULL, NULL, &form->args, level );
					}
					else{
						form->level+= 1;
						pn= 0;
						args= NULL;
					}
					if( (*dm_python->ascanf_PythonCall)( form->fun, pn, args, &A ) ){
						form->fun->value= A;
						ok= True;
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "#CP%d \t%s[...]== %s\n",
								*level, form->fun->name, ad2str(A, d3str_format, NULL) );
						}
					}
					if( ok && form->fun->accessHandler ){
						AccessHandler( form->fun, form->fun->name, level, form, NULL, &A );
					}
					GCA();
					if( (form->level-= 1) ){
						xfree(args);
					}
					if( ok ){
						if( form->negate ){
							form->value= A= (form->fun->value && !NaN(form->fun->value))? 0 : 1;
						}
						else{
							form->value= A= form->sign* form->fun->value;
						}
						form->fun->assigns+= 1;
						r+= (j=1);
					}
				}
#if 0
				else if( form->fun->type== _ascanf_array ){
					if( af->linkedArray.dataColumn ){
						Check_linkedArray(af);
					}
				}
#endif
				else if( form->fun->type== _ascanf_novariable ){
					form->value= A= 0;
					/* if( pragma_unlikely(ascanf_verbose) ) */{
						fprintf( StdErr, "#%dC: value of deleted %s\"%s\"=0 (was %s)\n",
							(*level), (form->fun->internal)? "internal " : "", form->fun->name, ad2str( form->fun->value, d3str_format, NULL)
						);
					}
				}
			}
			else if( form->args && form->args->list_of_constants && form->alloc_argc /* < 2 */ ){
			  int pn= form->alloc_argc;
#ifdef ASCANF_FORM_ARGVALS
			    /* 20020816: it MAY be that we have to distinguish form->level!=0 and form->level==0 here too,
				 \ as in the other places where we use form->argvals or allocate new mem depending on the level!
				 \ It is slightly more complicated here, as we can jump outside the scope, and thus would
				 \ have to test-and-decrement/deallocate in multiple locations (or use a 2level goto mechanism).
				 */
			  double *args= form->argvals;
#else
			  ALLOCA( args, double, pn+1, args_len);
				memset( args, 0, (pn+1)* sizeof(double) );
#endif
				if( (pn= evaluate_constants_args_list( form, args, pn, level ))> 0 ){
					switch( form->fun->type ){
						default:
							if( (pn= evaluate_constants_list( form, args, form->alloc_argc, level, False ))<= 0 ){
								if( unused_pragma_unlikely( !(pn==-2 && form->fun->type== _ascanf_array) ) ){
									if( form->list_of_constants ){
										correct_constants_args_list( form, level, __FILE__, __LINE__ );
									}
									goto call_compiled_ascanf_function;
								}
							}
							break;
						case _ascanf_variable:
							  /* This case is sufficiently simple to not need to call an additional function */
							form->fun->value= args[0];
							form->fun->assigns+= 1;
							if( form->fun->accessHandler ){
								AccessHandler( form->fun, form->fun->name, level, form, NULL, NULL );
							}
							break;
					}
					goto cf_variable_return_value;
				}
				else{
/* 					if( form->args->list_of_constants ){	*/
/* 						correct_constants_args_list( form->args, level, __FILE__, __LINE__ );	*/
/* 					}	*/
					goto call_compiled_ascanf_function;
				}
			}
			else if( !form->args ){
				if( form->fun->type== _ascanf_variable || form->fun->type== _ascanf_array ||
					form->fun->type== _ascanf_simplestats || form->fun->type== _ascanf_simpleanglestats
					|| form->fun->type== _ascanf_python_object
				){
#ifdef VERBOSE_CONSTANTS_EVALUATION
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, "#%dC: \t%s%s",
							(*level)+1, CF_prefix(form), FUNNAME(form->fun)
						);
						if( form->fun->type== _ascanf_array ){
							fprintf( StdErr, "[%d]", form->fun->last_index );
						}
						fprintf( StdErr, "==%s\n", ad2str( form->fun->value, d3str_format, NULL) );
					}
#endif
cf_variable_return_value:;
					if( form->negate ){
						A= (form->fun->value && !NaN(form->fun->value))? 0 : 1;
					}
					else{
						A= form->sign* form->fun->value;
					}
					form->value= A;
					form->fun->reads+= 1;
					r+= (j=1);
				}
				else if( form->fun->type== _ascanf_novariable ){
					form->value= A= 0;
					/* if( pragma_unlikely(ascanf_verbose) ) */{
						fprintf( StdErr, "#%dC: value of deleted %s\"%s\"=0 (was %s)\n",
							(*level), (form->fun->internal)? "internal " : "", form->fun->name, ad2str( form->fun->value, d3str_format, NULL)
						);
					}
				}
			}
			else{
				goto call_compiled_ascanf_function;
			}
		}
		else if( form->type== _ascanf_procedure ){
			if( form->fun->type== _ascanf_procedure ){
			  Boolean ok= False;
			  double *lArgList= af_ArgList->array;
			  int lArgc= af_ArgList->N, auA= ascanf_update_ArgList;
			  Compiled_Form *gpar;
				if( form->parent && form->fun->procedure->parent ){
					  /* reparent the procedure's form so that its grandparent is the same
					   \ as the grandparent of the current form. This is as it should be.
					   */
					gpar= form->fun->procedure->parent->parent;
					form->fun->procedure->parent->parent= form->parent->parent;
				}
				  /* Arguments to the procedure (proc[exprlist]) show up as form->args.	*/
				if( form->args ){
#if ASCANF_FORM_ARGVALS == 1 || 1
				    /* 20020418: Here too! (but I'm not 100% sure about the value for pn) */
				  int pn= form->alloc_argc;
				  double *args;
					if( form->level ){
						  /* 20020816: if we're here with non-zero recursion, we shouldn't use the argvals array. This
						   \ would mess up the arguments of the calling expression form.
						   */
						if( !(args= (double*) calloc( pn, sizeof(double) )) ){
							fprintf( StdErr, "%s: argument-list allocation error (%s) (executing without arguments!); %s::%d\n",
								form->top->af->name, serror(), __FILE__, __LINE__
							);
							pn= 0;
						}
					}
					else{
						args= form->argvals;
						memset( args, 0, pn* sizeof(double) );
					}
#else
				  int pn= form->argc;
				  ALLOCA( args, double, pn+1, args_len);
					memset( args, 0, (pn+1)* sizeof(double) );
#endif
					form->level+= 1;
					_compiled_fascanf( &pn, form->fun->name, args, NULL, NULL, NULL, &form->args, level );
					SET_AF_ARGLIST( args, pn );
					ascanf_update_ArgList= False;
					if( !form->fun->procedure->fun ){
						ok= True;
					}
					else if( form->fun->dialog_procedure ){
						if( evaluate_procedure( &j, form->fun, &A, level ) ){
							form->fun->value= A;
							ok= True;
						}
					}
					else{
						form->fun->procedure->level+= 1;
						if( call_compiled_ascanf_function( i, &s, &A, &j, "compiled_fascanf", &form->fun->procedure, level) ){
							  /* 20000910: form->fun->procedure->value should also be set!!	*/
							form->fun->value= form->fun->procedure->value= A;
							ok= True;
							if( pragma_unlikely(ascanf_verbose) ){
	/* 							fprintf( StdErr, "#C%d \t%s== %s\n", *level, form->fun->name, ad2str(A, d3str_format, NULL) );	*/
								Print_ProcedureCall( StdErr, form->fun, level );
								fprintf( StdErr, "== %s\n", ad2str(A, d3str_format, NULL) );
							}
						}
						form->fun->procedure->level-= 1;
					}
					if( ok && form->fun->accessHandler ){
						AccessHandler( form->fun, form->fun->name, level, form, NULL, &A );
					}
					GCA();
#if ASCANF_FORM_ARGVALS == 1 || 1
					if( (form->level-= 1) ){
						xfree(args);
					}
#endif
				}
				else{
					form->level+= 1;
					SET_AF_ARGLIST( ascanf_ArgList, 0 );
					ascanf_update_ArgList= False;
					if( !form->fun->procedure->fun ){
						ok= True;
					}
					else if( form->fun->dialog_procedure ){
						if( evaluate_procedure( &j, form->fun, &A, level ) ){
							form->fun->value= A;
							ok= True;
						}
					}
					else{
						form->fun->procedure->level+= 1;
						if( call_compiled_ascanf_function( i, &s, &A, &j, "compiled_fascanf", &form->fun->procedure, level) ){
							if( pragma_unlikely(ascanf_verbose) ){
								fprintf( StdErr, "#C%d \t%s== %s\n", *level, form->fun->name, ad2str(A, d3str_format, NULL) );
							}
							  /* 20000910: form->fun->procedure->value should also be set!!	*/
							form->fun->value= form->fun->procedure->value= A;
							ok= True;
						}
						form->fun->procedure->level-= 1;
					}
					if( ok && form->fun->accessHandler ){
						AccessHandler( form->fun, form->fun->name, level, form, NULL, &A );
					}
					form->level-= 1;
				}
				if( ok ){
					if( form->negate ){
						form->value= A= (form->fun->procedure->value && !NaN(form->fun->procedure->value))? 0 : 1;
					}
					else{
						form->value= A= form->sign* form->fun->procedure->value;
					}
					form->fun->assigns+= 1;
					r+= (j=1);
				}
				if( form->fun->procedure->parent ){
					form->fun->procedure->parent->parent= gpar;
				}
				SET_AF_ARGLIST( lArgList, lArgc );
				ascanf_update_ArgList= auA;
			}
			else if( form->fun->type== _ascanf_novariable ){
				form->value= A= 0;
				/* if( pragma_unlikely(ascanf_verbose) ) */{
					fprintf( StdErr, "#%dC: value of deleted %s\"%s\"=0 (was %s)\n",
						(*level), (form->fun->internal)? "internal " : "", form->fun->name, ad2str( form->fun->value, d3str_format, NULL)
					);
				}
			}
		}
		else{
call_compiled_ascanf_function:;
			if( call_compiled_ascanf_function( i, &s, &A, &j, "compiled_fascanf", &form, level) ){
				r+= 1;
			}
		}
		SET_CHANGED_FLAG( ch, i, A, *a, j);
		*a++= A;
		form= form->cdr;
		i++;
		if( !reset_ascanf_index_value ){
			(*ascanf_index_value)+= 1;
		}
		if( ascanf_arg_error && *ascanf_ExitOnError> 0 ){
			fprintf( StdErr, "compiled_fascanf(\"%s\"): raising read-terminate because of error(s) encountered\n",
				s
			);
			ascanf_exit= 1;
			j= EOF;
		}

	}
/* 	(*level)--;	*/

	if( using_ps ){
		param_scratch_inUse= psiU;
	}

	if( pragma_unlikely(ascanf_verbose) && !(*level) ){
	  char hdr[128];
		sprintf( hdr, "#%dC: ", (*level) );
		fputs( hdr, StdErr );
		Print_Form( StdErr, Form, 1, True, hdr, NULL, "\n", True );
	}
	reset_ascanf_currentself_value= racv;
	reset_ascanf_index_value= raiv;

/* 	SET_AF_ARGLIST( ArgList, Argc );	*/
	if( Nset ){
		*ascanf_data0= DATA[0];
		*ascanf_data1= DATA[1];
		*ascanf_data2= DATA[2];
		*ascanf_data3= DATA[3];
		Nset= 0;
	}
	if( r< *n){
		*n= r;					/* not enough read	*/
		if( Form && *Form ){
			(*Form)->ok= EOF;
		}
		return( EOF);				/* so return EOF	*/
	}
	if( Form && *Form ){
		(*Form)->ok= r;
	}
	return( r);
}

/* read multiple floating point values, uses 'Form' instead of 's'	*/
int compiled_fascanf( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], Compiled_Form **Form)
{ int level= 0;
#ifndef USE_AA_REGISTER
  int _comp_= ascanf_SyntaxCheck;
  int avp= ascanf_verbose, acp= ascanf_comment, app= ascanf_popup;
  int AI= *Allocate_Internal;
#endif

	if( !a ){
		*n= 0;
		return( EOF);
	}

#ifndef USE_AA_REGISTER
	PAS_already_protected= False;
	if( PAS_Lazy_Protection!=2 || sigsetjmp(pa_jmp_top,1)==0 )
#endif
	{
		if( !Form || !(*Form) ){
			return( _fascanf( n, s, a, ch, data, column, NULL, &level, NULL ) );
		}
		else{
			return( _compiled_fascanf( n, s, a, ch, data, column, Form, &level ) );
		}
	}
#ifndef USE_AA_REGISTER
	else{
		*n= -1;
		ascanf_SyntaxCheck= _comp_;
		*ascanf_verbose_value= ascanf_verbose= avp, ascanf_comment= acp, ascanf_popup= app;
		*Allocate_Internal= AI;
		return(EOF);
	}
#endif
}

  /* Return the ascanf_Function entry in the function/variable tabels associated with the
   \ compiled form passed as an argument.
   */
ascanf_Function *Procedure_From_Code( void *ptr )
{ Compiled_Form *form= ptr;
	if( form ){
		if( form->af && form->af->type== _ascanf_procedure ){
			return( form->af );
		}
		else if( form->top && form->top->af && form->top->af->type== _ascanf_procedure ){
			return( form->top->af );
		}
		else{
			return( NULL );
		}
	}
	else{
		return( NULL);
	}
}

int compile_procedure_code( int *n, char *source, double *result, ascanf_Function *af, int *level )
{ Compiled_Form *parent= NULL;
  int uaA= ascanf_update_ArgList;
  ascanf_Function plf, *proc_local_Functions= NULL, **vlF= vars_local_Functions;
  int proc_locals= 0, *lF= local_Functions;
  char plfname[128]= "compile_procedure_code::proc_local_Functions";

	  /* destroy any previous code:	*/
	if( af->procedure ){
		if( af->procedure->parent && af->procedure->parent->fun== af ){
			Destroy_Form( &af->procedure->parent );
		}
		Destroy_Form( &af->procedure );
	}

	af->procedure_separator = ascanf_separator;

	if( AllowProcedureLocals ){
		memset( &plf, 0, sizeof(ascanf_Function) );
		plf.name= plfname;
		plf.name_length= strlen(plfname);
		plf.function= ascanf_DeclareVariable;
		proc_local_Functions= &plf;
		vars_local_Functions= &proc_local_Functions;
		local_Functions= &proc_locals;
	}
	else{
	 // 20090416: must be able to be switched off!!
		vars_local_Functions= NULL;
		local_Functions= NULL;
	}

	*n= 1;
		  /* 20020502: */
		ascanf_update_ArgList= False;
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "##ac%d \t(%s) $ args array fixed at %d (instead of %d) elements\n",
				(*level), af->name, af_ArgList->N, *n
			);
		}
	  /* and compile the new...	*/
	__fascanf( n, source, result, NULL, NULL, NULL, &af->procedure, level, NULL );
	Correct_Form_Top( &(af->procedure), NULL, af );
	{
	  /* add a stub form that will give the procedure code a parent, i.e.
	   \ that parent's name will be the procedure name.
	   */
		Add_Form( &parent, _ascanf_procedure, NULL, 0, NULL, af, NULL );
		  /* af->links will now be 1 too high: decrement it: */
		af->links-= 1;
		  /* Install the stub parent: */
		af->procedure->parent= parent;
	}
	ascanf_update_ArgList= uaA;

// 20090416: why would we NOT want to restore these variables ALWAYS???
// 	if( AllowProcedureLocals )
	{
		vars_local_Functions= vlF;
		local_Functions= lF;
	}

	if( *n== 1 ){
	  Sinc sinc;
		sinc.sinc.string= NULL;
		Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic );
		Sflush( &sinc );
		Sprint_string( &sinc, "", NULL, "", source );
		xfree( af->procedure->expr );
		af->procedure->expr= sinc.sinc.string;

		if( AllowProcedureLocals ){
			af->procedure->vars_local_Functions= proc_local_Functions->cdr;
			af->procedure->local_Functions= proc_locals;
		}
		else{
			af->procedure->vars_local_Functions= NULL;
			af->procedure->local_Functions= 0;
		}

		return(1);
	}
	else{
		if( proc_local_Functions->cdr ){
			fprintf( StdErr, "\n### Warning: unsuccessfull compilation of \"%s\" generated %d local variables\n",
				af->name, proc_locals
			);
		}
		return(0);
	}
}

static ascanf_Function *EP_proc= NULL;
static int *EP_level= NULL;

extern Window xtb_input_dialog_inputfield;

static xtb_hret EP_editor_h( Window win, int bval, void *info)
{ ALLOCA( errbuf, char, ASCANF_FUNCTION_BUF+ 256, errbuf_len);
	if( EP_proc && EP_level ){
		errbuf[0]= '\0';
		EditProcedure( EP_proc, EP_level, EP_proc->name, errbuf );
		xtb_ti_set( xtb_input_dialog_inputfield, ascanf_ProcedureCode(EP_proc), NULL);
		if( errbuf[0] ){
			xtb_error_box( win, errbuf, "Message while editing procedure code:" );
		}
	}
	return( XTB_HANDLED );
}

int evaluate_procedure( int *n, ascanf_Function *proc, double *args, int *level )
{ int r;
  char *tbh= TBARprogress_header;
	TBARprogress_header= proc->name;
	if( proc->dialog_procedure ){
	  Window aw= ascanf_window;
	  extern Window LocalWin_window();
		r= 0;
		if( ActiveWin ){
			ascanf_window= LocalWin_window( ActiveWin );
		}
		else if( !ascanf_window ){
			if( init_window ){
				ascanf_window= init_window;
			}
			else{
			  char *w= cgetenv( "WINDOWID");
				if( w ){
					ascanf_window= atoi(w);
				}
			}
		}
		if( proc->procedure->level> 0 ){
			ascanf_arg_error= 1;
			ascanf_emsg= "Procedure already being executed!";
		}
		else if( ascanf_window ){
		  extern xtb_hret display_ascanf_variables_h();
		  extern xtb_hret process_hist_h();
		  int mlen= (proc->procedure->expr)? strlen(proc->procedure->expr) : 0;
		  char *expr;
		  char title[128]= "#x01Enter a single expression for procedure ", *nbuf;
			if( (expr = XGstrdup( proc->procedure->expr )) ){
				strncat( title, proc->name, 127- strlen(title) );
				EP_proc= proc;
				EP_level= level;
				if( (nbuf= xtb_input_dialog( ascanf_window, expr, 2*mlen, strlen(expr),
						(proc->usage)? proc->usage : "",
						parse_codes( title ),
						False,
						"Defined Vars/Arrays", display_ascanf_variables_h,
						"History", process_hist_h,
						"Edit", EP_editor_h
					))
				){
					cleanup( expr );
					if( expr[0] ){
					  double dum= 0;
						*n= 1;
						compile_procedure_code( n, expr, &dum, proc, level );
						if( *n== 1 ){
							r= 1;
						}
					}
					if( nbuf!= expr ){
						xfree( nbuf );
					}
				}
				xfree(expr);
			}
			else{
				xtb_error_box( ascanf_window, serror(), "Could not obtain memory:" );
			}
		}
		else{
			ascanf_arg_error= 1;
			ascanf_emsg= "No window active...";
		}
		ascanf_window= aw;
	}
	else{
		r= 1;
	}
	if( r ){
		*n= 1;
		proc->procedure->level+= 1;
		r= _compiled_fascanf( n, proc->procedure->expr, args, NULL, NULL, NULL, &proc->procedure, level );
		proc->procedure->level-= 1;
		  /* 20000910: form->fun->procedure->value should be set!!	*/
		proc->procedure->value= args[0];
		if( pragma_unlikely(ascanf_verbose) ){
/* 			fprintf( StdErr, "#C%d \t%s== ", *level, proc->name );	*/
			Print_ProcedureCall( StdErr, proc, level );
			fputs( "== ", StdErr );
		}
		TBARprogress_header= tbh;
		return(r);
	}
	else{
		*n= 0;
		TBARprogress_header= tbh;
		return( 0 );
	}
}

int Print_ProcedureCall( FILE *fp, ascanf_Function *proc, int *level )
{
	fprintf( fp, "#C%d \t%s", *level, proc->name );
	if( af_ArgList->N ){
	  int i;
		fprintf( fp, "[%s", ad2str(af_ArgList->array[0], d3str_format, NULL) );
		for( i= 1; i< af_ArgList->N; i++ ){
			fprintf( fp, ",%s", ad2str(af_ArgList->array[i], d3str_format, NULL) );
		}
		fputc( ']', fp );
	}
	return(1);
}

static int EditProcedure( ascanf_Function *af, /* double *arg, */ int *level, char *expr, char *errbuf )
{ char tnam[64]= "/tmp/XG-EditPROC-XXXXXX";
  int fd= mkstemp(tnam);
  int m= 0;
  FILE *fp= NULL, *rfp;
	if( !expr ){
		expr= "";
	}
	if( af->procedure->level> 0 ){
		if( errbuf ){
			sprintf( errbuf,
				"EditProcedure() #%d (%s): %s: procedure is being executed.\n",
				(*level), (TBARprogress_header)? TBARprogress_header : "", expr
			);
		}
		else{
			fprintf( StdErr,
				"EditProcedure() #%d (%s): %s: procedure is being executed.\n",
				(*level), (TBARprogress_header)? TBARprogress_header : "", expr
			);
		}
		return(m);
	}
	if( fd!= -1 ){
		close(fd);
		fp= fopen( tnam, "w" );
	}
	if( fp ){
	  char *code= ascanf_ProcedureCode(af);
	  char *opcode= (af->dialog_procedure)? "ASKPROC" : "DEPROC-noEval";
	  char *editor= getenv( "XG_EDITOR" ), *command= NULL;
		if( strstr( code, "#xn") ){
			fprintf( fp, "%s[", opcode );
			print_string2( fp, "", ",", af->name, False );
			fputs( code, fp );
		}
		else if( strstr( code, "\n" ) ){
			fprintf( fp, "%s[", opcode );
			print_string2( fp, "", ",", af->name, False );
			print_string( fp, "", NULL, "", code );
		}
		else{
			fprintf( fp, "%s[", opcode );
			print_string2( fp, "", ",", af->name, False );
			Print_Form( fp, &af->procedure, 0, True, NULL, "\t", NULL, False );
		}
		if( af->usage ){
			  /* 20000507: preserve codes in usage strings.	*/
			print_string2( fp, ",\"", "\"", af->usage, True );
		}
		fputs( "] @", fp );
		if( af->dymod && af->dymod->name && af->dymod->path ){
			fprintf( fp, " # Module={\"%s\",%s}", af->dymod->name, af->dymod->path );
		}
		if( af->fp ){
			fprintf( fp, " # open file fd=%d", fileno(af->fp) );
		}
		if( af->cfont ){
			fprintf( fp, " # cfont: %s\"%s\"/%s[%g]",
				(af->cfont->is_alt_XFont)? "(alt.) " : "",
				af->cfont->XFont.name, af->cfont->PSFont, af->cfont->PSPointSize
			);
		}
		if( af->label ){
			fprintf( fp, " # label={%s}", af->label );
		}
		fputs( "\n\n", fp );
		if( !editor ){
			editor= "xterm -e vi";
		}
		fclose(fp);
		if( (command= concat2( command, editor, " ", tnam, NULL )) ){
		  Sinc input;
		  char *d, buf[512];
			system( command );
			xfree(command);
			input.sinc.string= NULL;
			Sinc_string_behaviour( &input, NULL, 0,0, SString_Dynamic );
			Sflush( &input );
			  /* There are not many cases where throwing away events is a good idea, but this seems to be one.
			   \ We just got back control from an external programme which in all likelihood obscured us and otherwise
			   \ caused events to be generated. We don't know if the targeted window (e.g. an input dialog as from
			   \ xtb_input_dialog) is still there, and if X11 context reads are not going to return garbage.
			   \ Hence this desperate, kludgy way of handling this = not.
			   */
			XSync(disp, True);
			if( (rfp= fopen(tnam, "r")) ){
				  /* It would be easier to be able to use IncludeFile() here,
				   \ but we can very well end up here through a call to new_param_now(),
				   \ a routine that can not be called recursively (but that would need to
				   \ be in that case). Thus, we do it with the Sinc mechanism (a simple
				   \ way to include a whole file into a single buffer), and then pass that
				   \ buffer to _fascanf().
					IncludeFile( ActiveWin, fp, tnam );
				   */
				while( !feof(rfp) && !ferror(rfp) && (d= fgets( buf, sizeof(buf)/sizeof(char), rfp)) ){
					Sputs( buf, &input );
				}
				Sflush( &input );
				fclose(rfp);
			}
			else{
				unlink(tnam);
				goto editproc_fopen_error;
			}
			if( input.sinc.string ){
			  double *arg;
			  int ape= ascanf_propagate_events;
				  /* We should not pass the arglist to the following call to _fascanf() */
				cleanup( input.sinc.string );
				m= ASCANF_MAX_ARGS;
				  /* 20030413: don't propagate events while compiling. This means that no asynchronous
				   \ redraw can be initiated that would attempt to access 'our' procedure while it
				   \ is in the process of receiving new code. That would crash things. This solution
				   \ is probably a little overkill; the no-propagation should probably be invoked in
				   \ interactive_param_now().
				   */
				ascanf_propagate_events= 0;
				if( (arg = (double*) calloc( m, sizeof(double) )) ){
					__fascanf( &m, input.sinc.string, arg, NULL, NULL, NULL, NULL, level, NULL );
					xfree(arg);
				}
				else{
					fprintf( StdErr, "EditProcedure() couldn't allocate %d doubles (%s)\n", m, serror() );
				}
				ascanf_propagate_events= ape;
				GCA();
			}
			xfree( input.sinc.string );
		}
		else{
			if( errbuf ){
				sprintf( errbuf,
					"EditProcedure() #%d (%s): %s: construct command to edit %s (%s)\n",
					(*level), (TBARprogress_header)? TBARprogress_header : "", expr,
					tnam, serror()
				);
			}
			else{
				fprintf( StdErr,
					"EditProcedure() #%d (%s): %s: construct command to edit %s (%s)\n",
					(*level), (TBARprogress_header)? TBARprogress_header : "", expr,
					tnam, serror()
				);
			}
		}
		unlink(tnam);
	}
	else{
editproc_fopen_error:;
		if( errbuf ){
			sprintf( errbuf,
				"EditProcedure() #%d (%s): %s: can't open temp file %s (%s)\n",
				(*level), (TBARprogress_header)? TBARprogress_header : "", expr,
				tnam, serror()
			);
		}
		else{
			fprintf( StdErr,
				"EditProcedure() #%d (%s): %s: can't open temp file %s (%s)\n",
				(*level), (TBARprogress_header)? TBARprogress_header : "", expr,
				tnam, serror()
			);
		}
	}
	return(m);
}

/*
 \ 20031112: a wrapper routine to allow calling ascanf callbacks from within C code.
 \ Some additional variables other than the actual arguments cannot be avoided...
 \ double ASCB_call( ASCANF_CALLBACK( (*function) ), int *success, int level, char *expr, int max_argc, int argc, ... )
 \ CAUTION: the arguments have to be doubles, even constants (the compiler won't know what to promote to...)
 */
double ASCB_call( void *Function, int *success, int level, char *expr, int max_argc, int argc, VA_DCL )
{ va_list ap;
  double result= 0;
  ASCANF_CALLBACK( (*function) );
  int rval= 0;
	function= Function;
	if( max_argc< 0 ){
		max_argc= (ASCANF_MAX_ARGS>=0)? ASCANF_MAX_ARGS : 0;
	}
	CLIP( argc, 0, max_argc );
	if( function ){
	  extern int ascanf_arguments;
	  int aaa = ascanf_arguments;
	  double *arg = NULL;
#ifdef ASCANF_ALTERNATE
	  ascanf_Callback_Frame __ascb_frame;
#endif
		ascanf_arg_error = 0;
		if( argc ){
			arg = (double*) malloc( (max_argc+1) * sizeof(double) );
			if( arg ){
				arg[0]= 0;
				va_start(ap, argc);
				for( ascanf_arguments= 0; ascanf_arguments< argc; ascanf_arguments++ ){
					arg[ascanf_arguments]= va_arg(ap, double);
					if( !arg[ascanf_arguments] ){
						argc = ascanf_arguments;
					}
				}
				va_end(ap);
			}
			else{
				fprintf( StdErr, "ASCB_call(\"%s\"): can't allocate %d arguments (%s)\n",
					expr, max_argc+1, serror()
				);
				ascanf_arg_error= 1;
				ascanf_emsg = " (allocation failure in ASCB_call()) ";
			}
		}
		else{
			ascanf_arguments = 0;
		}
		if( !ascanf_arg_error ){
// #ifdef DEBUG
			if( ascanf_verbose ){
				fprintf( StdErr, "%p", function );
				if( argc ){
				  int i;
					fprintf( StdErr, "[%s", ad2str( arg[0], d3str_format, 0 ) );
					for( i = 1 ; i < argc ; i++ ){
						fprintf( StdErr, ",%s", ad2str( arg[i], d3str_format, 0 ) );
					}
					fputs( "]\n", StdErr );
				}
			}
// #endif
#ifdef ASCANF_ALTERNATE
			__ascb_frame.args= (ascanf_arguments)? arg : NULL;
			__ascb_frame.result= &result;
			__ascb_frame.level= &level;
			__ascb_frame.self = NULL;
			__ascb_frame.compiled= NULL;
#	if defined(ASCANF_ARG_DEBUG)
			__ascb_frame.expr= expr;
#	endif
			rval= (*function)( &__ascb_frame );
#else
			rval= (*function)( (ascanf_arguments)? arg : NULL, &result, &level );
#endif
			xfree(arg);
			ascanf_arguments = aaa;
		}
	}
	if( success ){
		*success= rval;
	}
	return(result);
}

char AScanf_Compile_Options[]= "$Options: @(#)"
	" ascanf: Default max. arguments="
	STRING(AMAXARGSDEFAULT)
	";"
#ifndef ASCANF_ALTERNATE
	" <original callback argument passing>"
#endif
#ifdef ASCANF_FORM_ARGVALS
	" ASCANF_FORM_ARGVALS="
	STRING(ASCANF_FORM_ARGVALS)
#endif
#if ASCANF_AUTOVARS_UNIQUE
	" ASCANF_AUTOVARS_UNIQUE="
	STRING(ASCANF_AUTOVARS_UNIQUE)
#endif
#ifdef VERBOSE_CONSTANTS_EVALUATION
	" VERBOSE_CONSTANTS_EVALUATION"
#endif
#ifdef ASCANF_ARG_DEBUG
	" ASCANF_ARG_DEBUG"
#endif
#ifdef DEBUG
	" DEBUG"
#endif
	" $"
;

void Set_Ascanf_Special_Fun( ascanf_Function *af )
{
	if( af->function== ascanf_matherr_fnc ){
		af->special_fun= matherr_fun;
	}
	else if( af->function== ascanf_global_fnc ){
		af->special_fun= global_fun;
	}
	else if( af->function== ascanf_verbose_fnc ){
		af->special_fun= verbose_fun;
	}
	else if( af->function== ascanf_noverbose_fnc ){
		af->special_fun= no_verbose_fun;
	}
	else if( af->function== ascanf_IDict_fnc ){
		af->special_fun= IDict_fun;
	}
	else if( af->function== ascanf_comment_fnc ){
		af->special_fun= comment_fun;
	}
	else if( af->function== ascanf_popup_fnc ){
		af->special_fun= popup_fun;
	}
	else if( af->function== ascanf_if ){
		af->special_fun= ifelse_fun;
	}
	else if( af->function== ascanf_switch0 ){
		af->special_fun= switch_fun0;
	}
	else if( af->function== ascanf_switch ){
		af->special_fun= switch_fun;
	}
	else if( af->function== ascanf_dowhile ){
		af->special_fun= dowhile_fun;
	}
	else if( af->function== ascanf_whiledo ){
		af->special_fun= whiledo_fun;
	}
	else if( af->function== ascanf_DeclareVariable ){
		af->special_fun= DCL_fun;
	}
	else if( af->function== ascanf_DeclareProcedure ){
		af->special_fun= DEPROC_fun;
	}
	else if( af->function== ascanf_DeleteVariable ){
		af->special_fun= Delete_fun;
	}
	else if( af->function== ascanf_EditProcedure ){
		af->special_fun= EditPROC_fun;
	}
	else if( af->function== ascanf_for_to ){
		af->special_fun= for_to_fun;
	}
	else if( af->function== ascanf_for_toMAX ){
		af->special_fun= for_toMAX_fun;
	}
	else if( af->function== ascanf_and ){
		af->special_fun= AND_fun;
	}
	else if( af->function== ascanf_or ){
		af->special_fun= OR_fun;
	}
	else if( af->function== ascanf_Variable || af->function== ascanf_Procedure ){
		af->special_fun= not_a_function;
	}
	else if( af->function== ascanf_systemtime ){
		af->special_fun= systemtime_fun;
	}
	else if( af->function== ascanf_systemtime2 ){
		af->special_fun= systemtime_fun2;
	}
	else if( af->function == ascanf_SHelp ){
		af->special_fun = SHelp_fun;
	}
	else{
		af->special_fun= direct_fun;
	}
}

/* 20080916: routine to check whether an ascanf_Function has a valid callback, correcting for certain types: */
int ascanf_CheckFunction( ascanf_Function *af )
{ int ret= 1;
	if( !af->function ){
		switch( af->type ){
			case _ascanf_variable:
			case _ascanf_array:
			case _ascanf_simplestats:
			case _ascanf_simpleanglestats:
				/* For the time being! Information could be given about the previous and actual number of observations. */
			case _ascanf_python_object:
			case _ascanf_novariable:
				af->function= ascanf_Variable;
				break;
			case _ascanf_procedure:
				af->function= ascanf_Procedure;
			default:
				ret= 0;
				break;
		}
	}
	return( ret );
}

  /* This routine makes dynamic copies of the name and usage entries in the global tables.
   \ These are most likely entered as compile-time constants. If ever these get de-allocated,
   \ hell breaks loose (i.e. we crash). Of course, this should never happen, but we can't
   \ be sure of that.
   \ 20020409: another very important initialisation: of the special_fun field!!!
   */
int Ascanf_Initialise()
{ ascanf_Function *af= vars_ascanf_Functions;
  int j, i, N= ascanf_Functions, rVN;
  char buf[64];
  extern char *XGstrdup();

	zero_div_zero.d= 0.0/0.0;

	for( j= 0; j< 1; j++ ){
		for( i= 0; i< N; i++, af++ ){
			if( af->name ){
				af->name= XGstrdup( af->name );
				if( index( af->name, '[') || index( af->name, ']') ){
					fprintf( StdErr, "!!\n!! Warning: entry %d name \"%s\" contains invalid characters!\n", i, af->name );
				}
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
			Set_Ascanf_Special_Fun(af);
			af->car= NULL;
// 			register_VariableName(af);
			take_ascanf_address(af);
		}
		switch( j ){
#if 0
			  /* See below!! */
			case 0:
				af= vars_internal_Functions;
				N= internal_Functions;
				break;
#endif
			default:
				break;
		}
	}

	rVN = register_VariableNames(1);
	af= vars_internal_Functions;
	for( i= 0; i< internal_Functions; i++, af++ ){
		if( af->name ){
			af->name= XGstrdup( af->name );
			if( strcmp( af->name, "$ascanf.interrupt")== 0 ){
				af_interrupt= af;
				ascanf_interrupt_value= &af_interrupt->value;
				*ascanf_interrupt_value= ascanf_interrupt;
				af->accessHandler= &internal_AHandler;
			}
			else if( strcmp( af->name, "$ascanf.escape")== 0 ){
				af_escape= af;
				ascanf_escape_value= &af_escape->value;
				*ascanf_escape_value= ascanf_escape;
				af->accessHandler= &internal_AHandler;
			}
		}
		else{
			sprintf( buf, "Internal-Function-%d", i );
			af->name= XGstrdup( buf );
		}
		if( af->usage ){
			af->usage= XGstrdup( af->usage );
		}
		ascanf_CheckFunction(af);
		if( af->function!= ascanf_Variable ){
			set_NaN(af->value);
			 /* 20050104: there are not many internal functions, but the one there is (DCL?) was not being
			  \ initialised correctly!!
			  */
			Set_Ascanf_Special_Fun(af);
		}
		else{
			af->special_fun= not_a_function;
		}
// 		register_VariableName(af);
		take_ascanf_address(af);
	}
	ascanf_VarLabel->usage= NULL;
	ascanf_VarLabel->is_usage= True;
	register_VariableName(ascanf_VarLabel);
	ascanf_d3str_format->usage= strdup(d3str_format);
	ascanf_d3str_format->is_usage= True;
	ascanf_d3str_format->accessHandler= &internal_AHandler;
	register_VariableName( ascanf_d3str_format );
	ascanf_XGOutput->name= strdup(ascanf_XGOutput->name);
	ascanf_XGOutput->fp= stdout;
	ascanf_XGOutput->fp_is_pipe= False;
	ascanf_XGOutput->usage= strdup("stdout");
	register_VariableName( ascanf_XGOutput );

	af_verbose->old_value= af_verbose_old_value= take_ascanf_address( af_verbose );
	ascanf_verbose= ASCANF_TRUE(*ascanf_verbose_value);
	register_VariableName( af_verbose );

	AllowSimpleArrayOps= ASCANF_TRUE(*AllowSimpleArrayOps_value);
	af_AllowSimpleArrayOps->accessHandler= &internal_AHandler;
	register_VariableName( af_AllowSimpleArrayOps );

	AllowArrayExpansion= ASCANF_TRUE(*AllowArrayExpansion_value);
	af_AllowArrayExpansion->accessHandler= &internal_AHandler;
	register_VariableName( af_AllowArrayExpansion );

	AlwaysUpdateAutoArrays= ASCANF_TRUE(*AlwaysUpdateAutoArrays_value);
	af_AlwaysUpdateAutoArrays->accessHandler= &internal_AHandler;
	register_VariableName( af_AlwaysUpdateAutoArrays );

	AllowProcedureLocals= ASCANF_TRUE(*AllowProcedureLocals_value);
	af_AllowProcedureLocals->accessHandler= &internal_AHandler;
	register_VariableName( af_AllowProcedureLocals );

	PrintNaNCode= ASCANF_TRUE( ascanf_d2str_NaNCode->value );
	ascanf_d2str_NaNCode->accessHandler= &internal_AHandler;
	register_VariableName( ascanf_d2str_NaNCode );

	Find_Point_use_precision= ! ASCANF_TRUE(*Find_Point_exhaustive_value);
	af_Find_Point_exhaustive->accessHandler= &internal_AHandler;
	register_VariableName( af_Find_Point_exhaustive );

	af_setNumber->accessHandler= &internal_AHandler;
	register_VariableName( af_setNumber );
	af_Counter->accessHandler= &internal_AHandler;
	register_VariableName( af_Counter );

	Elapsed_Since( &AscanfCompileTimer, True );
	SS_Reset_(SS_AscanfCompileTime);

	Dprint_fp= StdErr;
	ascanf_Dprint_fp->accessHandler= &internal_AHandler;
	register_VariableName( ascanf_Dprint_fp );

	af_ArgList_address= take_ascanf_address(af_ArgList);

#ifdef linux
{ char *dev= getenv("KRAN_DEVICE");
	if( !dev ){
		dev= "/dev/urandom";
	}
	if( (rndFD= open( dev, O_RDONLY ))== -1 ){
		fprintf( StdErr, "Can't open %s (%s); kran[] won't give valid results!\n", dev, serror() );
	}
}
#endif

	signal( SIGSEGV, segv_handler );
	signal( SIGBUS, segv_handler );

	if( (AutoLoadTable= (DyModAutoLoadTables*)calloc( AutoLoads, sizeof(DyModAutoLoadTables))) ){
		for( i= 0; i< AutoLoads; i++ ){
			AutoLoadTable[i]= _AutoLoadTable[i];
			AutoLoadTable[i].functionName= strdup(_AutoLoadTable[i].functionName);
			AutoLoadTable[i].DyModName= strdup(_AutoLoadTable[i].DyModName);
		}
		Auto_LoadDyMod( AutoLoadTable, AutoLoads, NULL );
	}
	else{
		AutoLoads= 0;
	}

	register_VariableNames(rVN);

	return(0);
}
