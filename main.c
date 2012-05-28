#ifndef _MAIN_C
#define _MAIN_C

#include "config.h"
IDENTIFY( "xgraph entry module" );

#define POPUP_IDENT

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>


#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "XXseg.h"
#include "xtb/xtb.h"
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include "NaN.h"

#include "xgALLOCA.h"

#include "Elapsed.h"

#include "fdecl.h"
#include "copyright.h"

/* #include "x86_gcc_fpe.c"	*/

XSegment *Xsegs, *lowYsegs, *highYsegs;		/* Point space for X */
XXSegment *XXsegs;
#define LYsegs	lowYsegs
#define HYsegs	highYsegs
XSegment_error *XsegsE;
long XsegsSize, XXsegsSize, YsegsSize, XsegsESize;

#include "buildplatform.h"
#ifndef XGraphBuildPlatform
#	define XGraphBuildPlatform	STRING(CC)
#endif

#ifdef __GNUC__
	char XGraphBuildString[]=  "version " STRING(VERSION_MAJOR) STRING(VERSION_MINOR) STRING(VERSION_PATCHL) "; gcc" STRING(__GNUC__)
#	ifdef __GNUC_MINOR__
		"." STRING(__GNUC_MINOR__)
#	endif
#	ifdef __GNUC_PATCHLEVEL__
		"." STRING(__GNUC_PATCHLEVEL__)
#	endif
		"[" __DATE__" "__TIME__",\"" XGraphBuildPlatform "\"]";
#else
	char XGraphBuildString[]=  "version " STRING(VERSION_MAJOR) STRING(VERSION_MINOR) STRING(VERSION_PATCHL) "; cc[" __DATE__" "__TIME__",\"" XGraphBuildPlatform "\"]";
#endif

DataSet *AllSets;
psUserInfo HO_Previous_psUI;
LocalWin *ActiveWin= NULL, *LastDrawnWin= NULL, HO_PreviousWin, *HO_PreviousWin_ptr= NULL;
/* A pointer to the tail of a linked list of all our windows:	*/
LocalWindows *WindowList= NULL, *WindowListTail;
extern LocalWin StubWindow;
AttrSet _AllAttrs[MAXATTR+1], *AllAttrs;

int debugging= 0;

#ifdef linux
	clock_t _linux_clk_tck;
#endif

extern FILE *StdErr;

void *XSynchronise( Display *disp, Bool onoff )
{
	return( (void*) XSynchronize( disp, onoff ) );
}

Window LocalWin_window( LocalWin *wi )
{
	if( wi ){
		return( wi->window );
	}
	else{
		return( 0 );
	}
}

LocalWin *aWindow( LocalWin *w )
{
	if( !w ){
		w= &StubWindow;
	}
	if( !w->window ){
	  char *wi= cgetenv("WINDOWID");
		if( wi ){
			w->window= atoi(wi);
		}
		else{
			w->window= RootWindow(disp, screen);
		}
	}
	return(w);
}

/* 20040209: corrected long undetected bug which considered the mesg parameter to be dynamic
 \ memory by definition, and always xfree'ed it...
 \ Thanks Mac OS X malloc tools for detecting this!!
 */
int XG_error_box( LocalWin **wi, char *title, char *mesg, VA_DCL )
{ va_list ap;
  int r= False;
  char *msg;
	va_start(ap, mesg);
	msg= (mesg)? strdup(mesg) : NULL;
	if( wi && title ){
	  char *c;
		while( (c= va_arg(ap, char*)) ){
			msg= concat2( msg, c, NULL );
		}
	}
	va_end(ap);
	if( title && msg ){
		if( wi ){
		  LocalWin *w= aWindow(*wi);
			if( disp ){
				r= xtb_error_box( w->window, msg, title );
			}
			else{
				fprintf( StdErr, "\n\t%s:\n%s\n", title, msg );
				r= False;
			}
		}
		else{
			fprintf( StdErr, "\n\t%s:\n%s\n", title, msg );
			r= False;
		}
		xfree( msg );
	}
	return(r);
}

FILE *StdErr, *NullDevice= NULL;

/* extern double Gonio_Base_Offset, Units_per_Radian;	*/
/* #define Gonio(fun,x)	(fun(((x)+Gonio_Base_Offset)/Units_per_Radian))	*/
/* #define InvGonio(fun,x)	((fun(x)*Units_per_Radian-Gonio_Base_Offset))	*/
/* #define Sin(x) Gonio(sin,x)	*/
/* #define Cos(x) Gonio(cos,x)	*/
/* #define Tan(x) Gonio(tan,x)	*/
/* #define ArcSin(x) InvGonio(asin,x)	*/
/* #define ArcCos(x) InvGonio(acos,x)	*/
/* #define ArcTan(wi,x,y) (_atan3(wi,x,y)*Units_per_Radian-Gonio_Base_Offset)	*/

int EndianType;
char *WhatEndian(int endian)
{ static char et[16];
	switch( endian ){
		case 0:
			strcpy(et, "big");
			break;
		case 1:
			strcpy(et, "little");
			break;
		default:
			snprintf(et, sizeof(et)/sizeof(char), "weird:%d", endian );
			break;
	}
	return(et);
}

void CheckEndianness()
{ union{
	char t[2];
	short s;
  } e;
#if 0
  short one16= 1;
  unsigned char one8= (*((unsigned char*) &one16));
#endif
	e.t[0]= 'a';
	e.t[1]= 'b';
	switch( e.s ){
		case 0x6162:
			 /* big endian: most significant bit is at the start (leftmost) */
			EndianType= 0;
			break;
		case 0x6261:
			 /* little endian: most significant bit is at the end (rightmost) */
			EndianType= 1;
			break;
		default:
			fprintf( stderr, "Found weird endian type: *((short*)\"ab\")!= 0x6162 nor 0x6261 but 0x%hx\n",
				e.s
			);
			EndianType= 2;
			break;
	}
}


#ifdef DEBUG
	char *charPTR= NULL;
	char **charptrPTR= NULL;
	short *shortPTR= NULL;
	int *intPTR= NULL;
	double *doublePTR= NULL;
	long *longPTR= NULL;
	DataSet *setPTR= NULL;
	LocalWin *wiPTR= NULL;
	FILE **FILE_PPTR= NULL;
	xtb_frame *framePTR= NULL;
#endif

#if defined(HAVE_FFTW_CHECK) && defined(HAVE_FFTW) && HAVE_FFTW
#include "ascanf.h"
#include "compiled_ascanf.h"

#include <fftw.h>
#include <rfftw.h>

extern int FFTW_Initialised;
extern char *FFTW_wisdom;

extern int ascanf_arguments, ascanf_arg_error;
extern char *ascanf_emsg;

double rfftw_data[768], rfftw_output[2*(768/2+1)];

/* A test that uses the lowlevel ascanf routine to perform a rfftw (Fast Fourier Transform).
 \ This operation can be influenced (resulting in -NaN128s) by I don't yet know what that is not very cosher
 \ (stack? context?), so it may be useful as a testprobe.
 */
void testrfftw(char *fname, int linenr, int cnt)
{ double *data= rfftw_data, *output= rfftw_output, result;
  int i, level= 0;
  ascanf_Callback_Frame frame;

	for( i= 0; i< sizeof(rfftw_data)/sizeof(double); i++ ){
		data[i]= (i>=310 && i<=457)? 1 : 0;
	}
	frame.args= data;
	frame.result= &result;
	frame.level= &level;
	frame.self= NULL;
#ifdef ASCANF_FRAME_EXPRESSION
	frame.expr= "youhou ;)";
#endif

	do_rfftw( sizeof(rfftw_data)/sizeof(double), data, output, NULL );
	if( output ){
	  int nn= 0;
		for( i= 0; i< sizeof(rfftw_data)/sizeof(double); i++ ){
			if( NaN(output[i]) ){
				nn+= 1;
			}
		}
		if( nn ){
			fprintf( StdErr, "testrfftw called from %s:%d,%d; %d NaNs in rfftw result: error!\n", fname, linenr, cnt, nn );
		}
	}
}
#endif

extern int Check_Option( int (*compare_fun)(), char *arg, char *check, int len), Opt01;

/* #if defined(__GNUC__) && (defined(__MACH__) || defined(__APPLE_CC__)) && FFTW_CYCLES_PER_SEC == 1	*/
#if (defined(__MACH__) || defined(__APPLE_CC__)) && defined(MACH_ABSOLUTE_TIME_FACTOR)
#include <mach/mach_time.h>
double Mach_Absolute_Time_Factor;
#elif defined(USE_PERFORMANCECOUNTER)
double PerformanceCounter_Calibrator;
#endif

int XGmain( int argc, char *argv[] )
{ /* Don't initialise any variables here, just to be sure not to interfere with the
   \ alignment code that main() starts with.
   */
  char *pn;
  Boolean restart;
  char *command= NULL;
  void *dum;
  int i;

	pn= rindex( argv[0], '/' );
	restart= 0;
	register_FILEsDescriptor(stdin);
	register_FILEsDescriptor(stdout);
	StdErr= register_FILEsDescriptor(stderr);

	for( i= 0; i< argc; i++ ){
		if( Check_Option( strncasecmp, argv[i], "-db", 3) == 0) {
		  static int db= 1;
			debugFlag = (Opt01== -1)? db : Opt01;
			if( Opt01!= -1 && Opt01!= 1 ){
				debugLevel= Opt01;
			}
			db= !debugFlag;
		}
	}

	CheckPrefsDir( "xgraph" );

#if defined(HAVE_FFTW_CHECK) && defined(HAVE_FFTW) && HAVE_FFTW
	testrfftw(__FILE__,__LINE__,-1);
#endif

	dum= &xgalloca;

	if( strcmp( (pn)? &pn[1] : argv[0], "XGraph")== 0
		|| strcmp( (pn)? &pn[1] : argv[0], "GXraph")== 0
	){
		restart= 1;
	}
	else if( strcmp( argv[1], "-XGraph")== 0 ){
		restart= 2;
	}
	if( restart ){
	  int j;
		if( pn ){
			pn[1]= '\0';
			command= concat2( command, argv[0], "XGraph.sh", NULL );
		}
		else{
		  char us[2048];
		  int len;
			if( !(len= Whereis( "XGraph.sh", us, 1023 )) ){
				strcpy( us, "/usr/local/bin/XGraph.sh");
			}
			else{
				if( us[len-1]== '\n' ){
					us[len-1]= '\0';
				}
			}
			command= concat2( command, us, NULL );
		}
		for( j= 1, i= restart; i< argc; i++ ){
			if( strcmp( argv[i], "-XGraph" ) ){
				argv[j++]= argv[i];
			}
		}
		if( j< argc ){
			argv[j]= NULL;
		}
		if( debugFlag ){
			fprintf( StdErr, "%s: scripting through-start using execvp(%s,argv).\n",
				argv[0], command
			);
		}
		errno= 0;
		i= execvp( command, argv);
		fprintf( StdErr, "%s: execvp(%s,argv) returns %d while attempting the \"scripting\" startup (%s).\n",
			argv[0], command, i, serror()
		);
		return(-1);
	}

	CheckEndianness();
	NullDevice= register_FILEsDescriptor( fopen( "/dev/null", "r+" ) );

	  /* I have the project of providing an attribute entry '-1' that would result in invisible lines or something
	   \ of the sort. Unfortunately, one can't draw invisible lines by meddling with the dash pattern...
	   */
	AllAttrs= &(_AllAttrs)[1];

	Ascanf_Initialise();
	sprintf( d3str_format, "%%.%dg", DBL_DIG+1 );

/* 	fprintf( StdErr, "PS_PRINTING=0x%lx, PS_FINISHED=0x%lx, X_DISPLAY=0x%lx, XG_DUMPING=0x%lx\n",	*/
/* 		PS_PRINTING, PS_FINISHED, X_DISPLAY, XG_DUMPING	*/
/* 	);	*/

#if defined(HAVE_FFTW_CHECK) && defined(HAVE_FFTW) && HAVE_FFTW
	testrfftw(__FILE__,__LINE__,-1);
#endif

/* #if defined(__GNUC__) && (defined(__MACH__) || defined(__APPLE_CC__)) && FFTW_CYCLES_PER_SEC == 1	*/
#if (defined(__MACH__) || defined(__APPLE_CC__)) && defined(MACH_ABSOLUTE_TIME_FACTOR)
	{ struct mach_timebase_info timebase;

		mach_timebase_info(&timebase);
		Mach_Absolute_Time_Factor= ((double)timebase.numer / (double)timebase.denom) * 1e-9;
		if( debugFlag && Mach_Absolute_Time_Factor!= MACH_ABSOLUTE_TIME_FACTOR ){
			fprintf( StdErr, "Warning: precompiled time calibration %g different from runtime value %ge-9/%g=%g (diff=%g)!\n",
				MACH_ABSOLUTE_TIME_FACTOR,
				(double) timebase.numer, (double) timebase.denom, Mach_Absolute_Time_Factor,
				MACH_ABSOLUTE_TIME_FACTOR-Mach_Absolute_Time_Factor
			);
		}
	}
#elif defined(USE_PERFORMANCECOUNTER)
	{ long long lpFrequency;
		if( !QueryPerformanceFrequency(&lpFrequency) ){
			PerformanceCounter_Calibrator = 0;
		}
		else{
// 			PerformanceCounter_Calibrator = 1.0 / ((double) lpFrequency.QuadPart);
			PerformanceCounter_Calibrator = 1.0 / ((double) lpFrequency);
		}
	}
#endif

	return(0);
}

int main( int argc, char *argv[] )
{
#if 0
  /* From FFTW 3.0.1 : */
#if defined(__GNUC__) && defined(__i386__) && ( (__GNUC__ == 2 && __GNUC_MINOR__ >= 9) || __GNUC__ > 2 )
     /*
      * horrible hack to align the stack to a 16-byte boundary.
      *
      * We assume a gcc version >= 2.95 so that
      * -mpreferred-stack-boundary works.  Otherwise, all bets are
      * off.  However, -mpreferred-stack-boundary does not create a
      * stack alignment, but it only preserves it.  Unfortunately,
      * many versions of libc on linux call main() with the wrong
      * initial stack alignment, with the result that the code is now
      * pessimally aligned instead of having a 50% chance of being
      * correct.
      */
	{
		/*
		* Use alloca to allocate some memory on the stack.
		* This alerts gcc that something funny is going
		* on, so that it does not omit the frame pointer
		* etc.
		*/
		(void)__builtin_alloca(16);

		/*
		* Now align the stack pointer
		*/
		__asm__ __volatile__ ("andl $-16, %esp");

		if( getenv("XG_BAD_STACKALIGN") ){
			/* pessimally align the stack, in order to check whether the
			   stack re-alignment hacks in FFTW3 work
			*/
			__asm__ __volatile__ ("addl $-4, %esp");
		}
	}
#endif

#ifdef __ICC /* Intel's compiler for ia32 */
     {
		/*
		* Simply calling alloca seems to do the right thing.
		* The size of the allocated block seems to be irrelevant.
		*/
		_alloca(16);
     }
#endif
#endif

// #ifdef __CYGWIN__
	// 20100331: it seems cygwin's random functions return a very non-random and consistent 0 when not initialised.
	// in order to allow repeatable random sequences, initialise the 2 used generators with something constant:
	srand( (long) 'XGRF' );
	srand48( (unsigned int) 'XGRF' );
// #endif

#ifdef linux
	_linux_clk_tck = sysconf(_SC_CLK_TCK);
#endif

	{ int r= XGmain( argc, argv );
		return( (r)? r : xgraph( argc, argv) );
	}
}

#endif
