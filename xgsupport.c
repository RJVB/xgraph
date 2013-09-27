/* 
vi:ts=4:sw=4:
 * xgraph - A Simple Plotter for X
 *
 * David Harrison
 * University of California,  Berkeley
 * 1986, 1987, 1988, 1989
 *
 * Please see copyright.h concerning the formal reproduction rights
 * of this software.

 \\ This has become a HUGE module. Should be split up into several
 \\ (when I get into the mood....)

 */

#include "config.h"
IDENTIFY( "Main support module; contains XGraph saving routines" );

#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <math.h>
#include <string.h>
#ifndef _APOLLO_SOURCE
#	include <strings.h>
#endif
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <fcntl.h>

#include <signal.h>
#include <time.h>
#ifdef linux
#	include <sys/time.h>
#endif

#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
#	define USE_SSE2
#	include <xmmintrin.h>
#	include <emmintrin.h>
#	include "AppleVecLib.h"
#endif

#ifdef XG_DYMOD_SUPPORT
#	include "dymod.h"
#endif

#include <pwd.h>
#include <ctype.h>
#include "xgout.h"
#include "xgraph.h"
#include "xtb/xtb.h"

#include "hard_devices.h"
extern xtb_frame HO_Dialog;
extern xtb_frame SD_Dialog;

#include <X11/Xutil.h>
#include <X11/keysym.h>

#include "new_ps.h"

#include <setjmp.h>
extern jmp_buf toplevel;

#define ZOOM
#define TOOLBOX

#ifndef MAXFLOAT
#define MAXFLOAT	HUGE
#endif

#ifndef MAXPATHLEN
#	define MAXPATHLEN 1024
#endif

#define BIGINT		0xfffffff

#include "NaN.h"

#define GRIDPOWER 	10
extern int INITSIZE;

#define CONTROL_D	'\004'
#define CONTROL_C	'\003'
#define TILDE		'~'

#define BTNPAD		1
#define BTNINTER	3

#ifndef MAX
#	define MAX(a,b)	((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#	define MIN(a,b)	((a) < (b) ? (a) : (b))
#endif
  /* return b if b<a and b>0, else return a	*/
#define MINPOS(a,b)	(((b)<=0 || (a) < (b))? (a) : (b))
#define MAXNEG(a,b)	(((b)<=0 && (b) > (a))? (b) : (a))
#define ABS(x)		(((x)<0)?-(x):(x))
#define SIGN(x)		(((x)<0)?-1:1)
#define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}

#include <float.h>
#include "ascanf.h"
extern char ascanf_separator;

#include "Elapsed.h"
#include <errno.h>
#include <sys/stat.h>

#include "fdecl.h"
#include "copyright.h"

extern double zero_epsilon;

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#	include "arrayvops.h"
#elif defined(__SSE2__) || defined(__SSE3__)
#	include "arrayvops.h"
#endif

#define NORMSIZEX	600
#define NORMSIZEY	400
#define NORMASP		(((double)NORMSIZEX)/((double)NORMSIZEY))
#define MINDIM		100

extern void init_X();
#ifdef TOOLBOX
extern void do_error();
#endif
extern char *tildeExpand();

extern LocalWin StubWindow;

extern double psm_base, psm_incr, psdash_power;
extern int psm_changed, psMarkers, ps_coloured, ps_transparent, psEPS, psDSC;
extern int psSetPage;
extern double psSetPage_width, psSetPage_height;

extern double page_scale_x, page_scale_y;
extern int preserve_screen_aspect;

extern int MaxSets;
extern DataSet *AllSets;

extern int NCols, xcol, ycol, ecol, lcol, MaxCols;

extern int DrawAllSets, DetermineBoundsOnly;
extern int Determine_AvSlope, Determine_tr_curve_len;

extern int local_buf_size;
static int change_local_buf_size= False;
static int new_local_buf_size= -1;

extern Boolean changed_Allow_Fractions;
extern int Allow_Fractions;

extern LocalWin *ActiveWin;

/* For reading in the data */
extern int setNumber , fileNumber;
extern int maxSize ;

/* our stderr	*/
extern FILE *StdErr;
extern int use_pager;

extern char *ShowLegends( LocalWin *wi, int PopUp, int this_one);
extern int detach, PS_PrintComment, Argc;
extern char **Argv, *titleText2;
extern XGStringList *PSSetupIncludes;

extern XGStringList *Exit_Exprs;

extern FILE *NullDevice, **PIPE_fileptr;

int XG_preserve_filetime= False;
char *XG_preserved_filetime= NULL;

char PrintTime[256], mFileTime[256];

#include "xfree.h"

extern double *ascanf_memory;
extern double *ascanf_self_value, *ascanf_current_value;
extern double *ascanf_setNumber, *ascanf_numPoints, *ascanf_counter, *ascanf_Counter;
extern char *ascanf_emsg;
extern int reset_ascanf_currentself_value, reset_ascanf_index_value, ascanf_arg_error, ascanf_arguments, ascanf_SyntaxCheck;
extern char *TBARprogress_header;
extern ascanf_Function vars_ascanf_Functions[];
extern int ascanf_Functions;

int WM_TBAR= 20;

Boolean PIPE_error= False, isPIPE= False;
void PIPE_handler( int sig)
{  FILE *fp;
	if( PIPE_fileptr ){
	  int r= 0;
		fp= *PIPE_fileptr;
		if( isPIPE ){
			r= pclose(fp);
			isPIPE= False;
		}
		else{
			if( fp!= stderr && fp!= stdout && fp!= NullDevice && fp ){
				r= fclose(fp);
			}
		}
		if( PIPE_fileptr== &StdErr ){
/*  			*StdErr= *stderr;	*/
			StdErr= stderr;
		}
		else{
			if( !NullDevice ){
				NullDevice= register_FILEsDescriptor( fopen( "/dev/null", "w") );
			}
/*  			if( NullDevice ){	*/
/*  				**PIPE_fileptr= *NullDevice;	*/
/*  			}	*/
			*PIPE_fileptr= NullDevice;
		}
		PIPE_error= True;
		fprintf( StdErr, "Broken Pipe (%d), trying redirection to %s\n",
			r, (*PIPE_fileptr==stderr)? "stderr": "/dev/null"
		);
		fflush( StdErr );
	}
	signal( SIGPIPE, PIPE_handler);
}

extern void handle_FPE();

char *substitute( char *s, int c, int n)
{ char *S= s;
	while( s && *s ){
		if( *s== c ){
			*s= n;
		}
		s++;
	}
	return( S );
}

char *parse_codes( char *T )
{  char *c= T, *d= T;
   int is_hex;
	if( !T ){
		return( 0 );
	}
	while( *d ){
		is_hex= 0;
/* 20020910: */
/* 		if( *d== '#' && (d== T || d[-1]!= '#') && d[1]== 'x' )	*/
		if( *d== '#' && d[1]== 'x' )
		{
		 char hex_code[3]= { '\0', '\0', '\0'};
		 int code;
			switch( d[2] ){
				case 'N':
				case 'n':
					*c++= '\n';
					is_hex+= 1;
					d+= 3;
					break;
				case 'R':
				case 'r':
					*c++= '\r';
					is_hex+= 1;
					d+= 3;
					break;
				case 'T':
				case 't':
					*c++= '\t';
					is_hex+= 1;
					d+= 3;
					break;
				default:
					strncpy( hex_code, &d[2], 2 );
					if( sscanf( hex_code, "%x", &code )== 1 ){
						*c++ = (char) (code & 0x00ff);
						if( '\n'!= 0x0a && *c== 0x0a ){
							*c= '\n';
							if( debugFlag ){
								fprintf( StdErr, "parse_codes(\"%s\"): changed #x%s into '\\n' (#x%x)\n",
									T, hex_code, (int) *c
								);
								fflush( StdErr );
							}
						}
						is_hex= 1;
						d+= 4;
					}
					break;
			}
		}
		if( !is_hex ){
			if( *d== '\\' ){
				if( d[1]== 'n' && d[2]== '\n' ){
					d+= 2;
				}
				else if( d[1]== 'r' && d[2]== '\r' ){
					d+= 2;
				}
			}
			  /* 20020827: */
/* 			if( *d== '#' && d[1]== '#' ){	*/
/* 				*c++= *d++;	*/
/* 				d++;	*/
/* 			}	*/
/* 			else	*/
			{
				*c++ = *d++;
			}
		}
	}
	*c++ = *d++;
	return( T );
}

/* buffer to store comments in datafile:	*/
char *comment_buf= NULL;
int NoComment= False, comment_size= 0;

char *Add_Comment( char *comment, int left_align )
{ int len, add_newline= 0;
  char *c;
	if( !comment ){
		return( comment_buf );
	}
	if( NoComment ){
		return( (comment_buf)? comment_buf : "" );
	}
	len= strlen( parse_codes(comment) )+1;
	if( len== 1 ){
		return( comment_buf );
	}
	if( comment[len-2]!= '\n' ){
		add_newline= 1;
		len+= 1;
	}
	if( len ){
		if( left_align ){
			left_align= !xtb_is_leftalign(comment);
		}
		comment_buf= realloc( comment_buf, comment_size+ len+ ((left_align)? 2 : 1) );
		if( !comment_size ){
		  /* initial text	*/
			c= comment_buf;
		}
		else{
		  /* append	*/
			c= &comment_buf[strlen(comment_buf)];
		}
		if( left_align ){
			*c++= ' ';
		}
		while( *comment ){
		  /* copy from the right place, substituting TABs for single spaces	*/
			*c++ = (*comment== '\t')? ' ' : *comment;
			comment++;
		}
		if( add_newline ){
			*c++= '\n';
		}
		*c= '\0';
		comment_size+= len;
	}
	return( comment_buf );
}

char *add_comment( char *comment )
{
	return( Add_Comment(comment, False) );
}

/* buffer to store history of process (*DATA_PROCESS*, *TRANSFORM_..* etc):	*/
char *process_history= NULL;
int NoProcessHist = False, process_history_size= 0;
extern xtb_frame *process_pmenu;

char *add_process_hist( char *expression )
{ int len= ((expression)? strlen(expression) : 0)+1;
  char *c;
	if( !expression ){
		xfree( process_history );
		process_history_size= 0;
		GCA();
		return( NULL );
	}
	if( len== 1 || NoProcessHist ){
		GCA();
		return( process_history );
	}
	  /* remove trailing whitespace	*/
	if( *expression ){
	  int i= 0;
		c= &expression[strlen(expression)-1];
		while( isspace((unsigned char)*c) && c> expression ){
			i++;
			c--;
		}
		if( i ){
			c[1]= '\0';
		}
	}
	else{
		c = expression;
	}
	if( c[0]== '@' ){
		if( debugFlag ){
			fprintf( StdErr, "add_process_hist(): excluded statement ending in '@' (%s)\n",
				expression
			);
		}
		GCA();
		return( process_history );
	}
	{ ALLOCA( exp, char, len, explen);
		c= exp;
		while( *expression ){
		  /* copy from the right place, substituting TABs for single spaces	*/
			switch( *expression ){
				case '\t':
				case '\n':
					*c = ' ';
					break;
				default:
					*c = *expression;
					break;
			}
			c++;
			expression++;
		}
		*c= '\0';
		if( process_history ){
			if( strstr( process_history, exp) ){
				GCA();
				return( process_history );
			}
		}
		len+= 1;
		if( len ){
			process_history= realloc( process_history, process_history_size+ len+ 1 );
			if( !process_history_size ){
			  /* initial text	*/
				c= process_history;
			}
			else{
			  /* append	*/
				c= &process_history[strlen(process_history)];
			}
			{ char *d= exp;
				while( *d ){
					*c++= *d++;
				}
			}
			*c++= '\n';
			*c= '\0';
			process_history_size+= len;
		}
		GCA();
	}
	return( process_history );
}

  /* 20020323: this routine does more or less what Add_Comment does, except that
   \ it uses a user-specified Sinc (instead of the global comment_buf variable).
   \ It also doesn't call parse_codes() on the string to be added, and doesn't replace the TAB characters.
   */
char *Add_SincList( Sinc *List, char *string, int left_align )
{ int len, add_newline= 0;
	if( !List ){
		return( NULL );
	}
	if( !string || !*string ){
		return( List->sinc.string );
	}
	len= strlen( string );
	if( string[len-1]!= '\n' ){
		add_newline= 1;
		len+= 1;
	}
	if( len ){
		if( left_align ){
			left_align= !xtb_is_leftalign(string);
		}
		if( left_align ){
			Sputc( ' ', List );
		}
		Sputs( string, List );
/* 		while( *string ){	*/
/* 			Sputc( ((*string== '\t')? ' ' : *string), List );	*/
/* 			string++;	*/
/* 		}	*/
		if( add_newline ){
			Sputc( '\n', List );
		}
	}
	return( List->sinc.string );
}

char *XGrindex( const char *s, char c)
{
	return( (s)? rindex(s,c) : NULL );
}

char *strcpalloc( char **dest, int *alloclen, const char *src )
{ int len= (src)? strlen(src)+1 : 0, lendest;
	if( !dest || !alloclen ){
		return( NULL);
	}
	if( *alloclen< 0 ){
		*alloclen= lendest= (*dest)? strlen(*dest)+1 : 0;
	}
	if( !*dest ){
		*dest= (char*) calloc( len, sizeof(char) );
		*alloclen= len;
	}
	else if( len> *alloclen ){
		*dest= (char*) realloc( *dest, len* sizeof(char));
		*alloclen= len;
	}
	if( *dest ){
		strcpy( *dest, src );
	}
	return( *dest );
}

char *stralloccpy( char **dest, char *src, int minlen )
{ int dlen, slen;
	minlen+= 1;
	dlen= (*dest)? strlen(*dest)+1 : 0;
	if( dlen< minlen ){
		dlen= minlen;
	}
	if( src ){
		if( (slen= strlen(src)+1)> dlen ){
			dlen= slen;
		}
		*dest= realloc( *dest, dlen* sizeof(char) );
		strcpy( *dest, src );
	}
	else{
		*dest= realloc( *dest, dlen* sizeof(char) );
		*dest[0]= '\0';
	}
	return( *dest );
}

// 20101018: strcpy() replacement that is safe for overlapping buffers:
char *XGstrcpy( char *dest, const char *src )
{
	memmove( dest, src, sizeof(char) * (strlen(src)+1) );
	return dest;
}

// 20101018: strncpy() replacement that is safe for overlapping buffers:
char *XGstrncpy( char *dest, const char *src, size_t n )
{ int sn = strlen(src)+1;
	if( n < sn ){
		sn = n;
	}
	memmove( dest, src, n * sizeof(char) );
	return dest;
}

char *XGstrdup( const char *c )
{  int len;
   char *d;
	  /* 980616: don't return NULL for empty string passed!	*/
/* 	if( c && *c )	*/
	if( c )
	{
		len= strlen(c)+ 2;
		if( (d= calloc( len, sizeof(char) ))== NULL ){
			fprintf( StdErr, "XGstrdup(%s): can't allocate buffer of length %d (%s)\n",
				c, len, serror()
			);
			fflush( StdErr );
			exit(10);
		}
		else{
			  /* 20020706 */
			memcpy( d, c, len-1 );
		}
		return( d );
	}
	else{
		return( NULL );
	}
}

char *XGstrdup2( char *c , char *c2 )
{  int len= 0;
   char *d;
	if( c || c2 ){
		if( c ){
			len= strlen(c);
		}
		if( c2 ){
			len+= strlen(c2);
		}
		len+= 2;
		if( (d= calloc( len, sizeof(char) ))== NULL ){
			fprintf( StdErr, "XGstrdup2(%s,%s): can't allocate buffer of length %d (%s)\n",
				(c)? c : "NULL", (c2)? c2 : "NULL", len, serror()
			);
			fflush( StdErr );
			exit(10);
		}
		else{
			if( c ){
				strcpy( d, c );
				if( c2 ){
					strcat( d, c2);
				}
			}
			else{
				strcpy( d, c2 );
			}
		}
		return( d );
	}
	else{
		return( NULL );
	}
}

/* concat2(): concats a number of strings; reallocates the first, returning a pointer to the
 \ new area. Caution: a NULL-pointer is accepted as a first argument: passing a single NULL-pointer
 \ as the (1-item) arglist is therefore valid, but possibly lethal!
 */
#ifdef CONCATBLABLA
char *concat2( char *a, char *b, ...)
{}
#endif
char *concat2(char *string, VA_DCL)
{ va_list ap;
  int n= 0, tlen= 0;
  char *c= string, *buf= NULL, *first= NULL;
	if( debugFlag ){
		fprintf( StdErr, "concat2(): ");
	}
	va_start(ap, string);
	do{
		if( !n ){
			first= c;
		}
		if( c ){
			if( debugFlag && debugLevel ){
				fprintf( StdErr, "#%d[%p]", n, c );
				fflush( StdErr );
			}
			tlen+= strlen(c);
			if( debugFlag && debugLevel ){
				fprintf( StdErr, "=%d ", strlen(c) );
				fflush( StdErr );
			}
		}
		n+= 1;
	}
	while( (c= va_arg(ap, char*)) != NULL || n== 0 );
	va_end(ap);
	if( n== 0 || tlen== 0 ){
		if( debugFlag ){
			fprintf( StdErr, "%d strings, %d bytes: no action\n",
				n, tlen
			);
			fflush( StdErr );
		}
		return( NULL );
	}
	tlen+= 4;
	if( first ){
		buf= realloc( first, tlen* sizeof(char) );
	}
	else{
		buf= first= calloc( tlen, sizeof(char) );
	}
	if( buf ){
		if( debugFlag && debugLevel> 1 ){
			fprintf( StdErr, ".. \"%s\" ", buf );
			fflush( StdErr );
		}
		va_start(ap, string);
		  /* realloc() leaves the contents of the old array intact,
		   \ but it may no longer be found at the same location. Therefore
		   \ need not make a copy of it; the first argument can be discarded.
		   \ strcat'ing is done from the 2nd argument.
		   */
		while( (c= va_arg(ap, char*)) != NULL ){
			if( debugFlag && debugLevel> 1 ){
				fprintf( StdErr, "\"%s\" ", c );
				fflush( StdErr );
			}
			strcat( buf, c);
		}
		va_end(ap);
		buf[tlen-1]= '\0';
		if( debugFlag ){
			fprintf( StdErr, "%d bytes (%d strings) concatenated in buffer of length %d (%s)\n",
				strlen(buf), n, tlen, buf
			);
			fflush( StdErr );
		}
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "can't (re)allocate %d bytes to concat %d strings\n",
				tlen, n
			);
			fflush( StdErr );
		}
	}
	return( buf );
}

/* concat(): concats a number of strings	*/
#ifdef CONCATBLABLA
char *concat( char *a, char *b, ...)
{}
#endif
char *concat(char *first, VA_DCL)
{ va_list ap;
  int n= 0, tlen= 0;
  char *c= first, *buf= NULL;
	va_start(ap, first);
	do{
		tlen+= strlen(c)+ 1;
		n+= 1;
	}
	while( (c= va_arg(ap, char*)) != NULL );
	va_end(ap);
	if( n== 0 || tlen== 0 ){
		if( debugFlag ){
			fprintf( StdErr, "concat(): %d strings, %d bytes: no action\n",
				n, tlen
			);
		}
		return( NULL );
	}
	if( (buf= calloc( tlen, sizeof(char) )) ){
		c= first;
		va_start(ap, first);
		do{
			strcat( buf, c);
		}
		while( (c= va_arg(ap, char*)) != NULL );
		va_end(ap);
		if( debugFlag ){
			fprintf( StdErr, "concat(): %d bytes to concat %d strings\n",
				tlen, n
			);
		}
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "concat(): can't get %d bytes to concat %d strings\n",
				tlen, n
			);
		}
	}
	return( buf );
}

/* calls tempnam(), and appends "@" to prevent inclusion in the process_history */
char *XGtempnam( const char *dir, const char *prefix )
{
#if defined(linux) || defined(__APPLE__) || defined(__CYGWIN__)
 char *template = (dir)? concat( dir, "/", prefix, "-XXXXXX", NULL ) : concat( prefix, "-XXXXXX", NULL );
 char *name = NULL, *tnam = (template)? mktemp(template) : NULL;
	if( tnam ){
		// sadly we cannot append the '@' to the template (which must end in 6 X's): that
		// means we cannot use mkdtemp or any other variant which creates the temp. file at once
		// (OTOH, we do not necessarily always use the tempnam to create a temp file!)
		name = concat( tnam, "@", NULL );
		xfree(template);
	}
#else
 char *name= tempnam( dir, prefix );
	if( name ){
		if( (name= (char*) realloc( name, (strlen(name) + 2)* sizeof(char) )) ){
			strcat( name, "@" );
		}
		else{
			errno= ENOMEM;
		}
	}
#endif
	return( name );
}

/* Basic transformation stuff */

extern double llx, lly, llpx, llpy, llny, urx, ury; /* Bounding box of all data */
extern double real_min_x, real_max_x, real_min_y, real_max_y;
extern Window deleteWindow;



extern double scale_av_x, scale_av_y;
extern double _scale_av_x, _scale_av_y;
extern SimpleStats SS_X, SS_Y;
extern int show_stats;
extern SimpleStats SS_x, SS_y, SS_e, SS_Points;
extern SimpleStats SS__x, SS__y, SS__e;

#define loX	win_geo.bounds._loX
#define loY	win_geo.bounds._loY
#define lopX	win_geo.bounds._lopX
#define hinY	win_geo.bounds._hinY
#define lopY	win_geo.bounds._lopY
#define hiX	win_geo.bounds._hiX
#define hiY	win_geo.bounds._hiY
#define XOrgX	win_geo._XOrgX
#define XOrgY	win_geo._XOrgY
#define XOppX	win_geo._XOppX
#define XOppY	win_geo._XOppY
#define UsrOrgX	win_geo._UsrOrgX
#define UsrOrgY	win_geo._UsrOrgY
#define UsrOppX	win_geo._UsrOppX
#define UsrOppY	win_geo._UsrOppY
#define XUnitsPerPixel	win_geo._XUnitsPerPixel
#define YUnitsPerPixel	win_geo._YUnitsPerPixel

void LWG_printf( FILE *fp, char *ind, LocalWinGeo *wig)
{
	fprintf( fp, "%s_loX=%g, _loY=%g, _lopX=%g, _lopY=%g, _hinY=%g, _hiX=%g, _hiY=%g (%s)\n", ind,
		wig->bounds._loX, wig->bounds._loY, wig->bounds._lopX, wig->bounds._lopY,
		wig->bounds._hinY, wig->bounds._hiX, wig->bounds._hiY,
		(wig->user_coordinates)? "user specified" : "automatic"
	);
	fprintf( fp, "%s pure _loX=%g, _loY=%g, _lopX=%g, _lopY=%g, _hinY=%g, _hiX=%g, _hiY=%g\n", ind,
		wig->pure_bounds._loX, wig->pure_bounds._loY, wig->pure_bounds._lopX,
		wig->pure_bounds._lopY, wig->pure_bounds._hinY, wig->pure_bounds._hiX,
		wig->pure_bounds._hiY
	);
    fprintf( fp, "%s_XOrgX=%d, _XOrgY=%d\n", ind, wig->_XOrgX, wig->_XOrgY);
    fprintf( fp, "%s_XOppX=%d, _XOppY=%d\n", ind, wig->_XOppX, wig->_XOppY);
    fprintf( fp, "%s_UsrOrgX=%g, _UsrOrgY=%g\n", ind, wig->_UsrOrgX, wig->_UsrOrgY);
    fprintf( fp, "%s_UsrOppX=%g, _UsrOppY=%g\n", ind, wig->_UsrOppX, wig->_UsrOppY);
    fprintf( fp, "%s_XUnitsPerPixel=%g\n", ind, wig->_XUnitsPerPixel);
    fprintf( fp, "%s_YUnitsPerPixel=%g\n", ind, wig->_YUnitsPerPixel);
}

extern XContext win_context ;
extern XContext frame_context ;

/* Other globally set defaults */

extern Display *disp;		/* Open display            */
extern Visual *vis;			/* Standard visual         */
extern Colormap cmap;			/* Standard colormap       */
extern int use_RootWindow;
extern int screen;			/* Screen number           */
extern int search_vislist;	/* find best screen	*/
extern int depth;			/* Depth of screen         */
extern int install_flag;		/* Install colormaps       */
extern Pixel black_pixel;		/* Actual black pixel      */
extern Pixel white_pixel;		/* Actual white pixel      */
extern Pixel bgPixel;		/* Background color        */
extern Pixel normPixel;		/* Norm grid color         */
extern int bdrSize;			/* Width of border         */
extern Pixel bdrPixel;			/* Border color            */
extern Pixel zeroPixel;		/* Zero grid color         */
extern Pixel axisPixel, gridPixel;
extern Pixel highlightPixel;
extern char *blackCName;
extern char *whiteCName;
extern char *bgCName;
extern char *normCName;
extern char *bdrCName;
extern char *zeroCName;
extern char *axisCName, *gridCName;
extern char *highlightCName;
extern double zeroWidth;			/* Width of zero line      */
extern char zeroLS[MAXLS];		/* Line style spec         */
extern int zeroLSLen;			/* Length of zero LS spec  */
extern double axisWidth, gridWidth;			/* Width of axis line      */
extern double errorWidth;
extern char gridLS[MAXLS];		/* Axis line style spec    */
extern int gridLSLen;			/* Length of axis line style */
extern int XLabelLength;		/* length in characters of axis labels	*/
extern int YLabelLength;		/* length in characters of axis labels	*/
extern Pixel echoPix;			/* Echo pixel value        */
extern XGFontStruct dialogFont;
extern XGFontStruct dialog_greekFont;
extern XGFontStruct axisFont;		/* Font for axis labels    */
extern XGFontStruct legendFont;		/* Font for axis labels    */
extern XGFontStruct labelFont;		/* Font for axis labels    */
extern XGFontStruct legend_greekFont;
extern XGFontStruct label_greekFont;
extern XGFontStruct title_greekFont;
extern XGFontStruct axis_greekFont;
extern XGFontStruct titleFont;		/* Font for title labels   */
extern XGFontStruct markFont;
extern XGFontStruct cursorFont;
extern XGFontStruct fbFont, fb_greekFont;
extern int use_markFont;
extern int use_X11Font_length, use_gsTextWidth, auto_gsTextWidth, _use_gsTextWidth, used_gsTextWidth;
extern int scale_plot_area_x, scale_plot_area_y;
extern char titleText[MAXBUFSIZE+1]; 	/* Plot title              */
extern char titleText_0;
extern char XUnits[MAXBUFSIZE+1];	/* X Unit string           */
extern char YUnits[MAXBUFSIZE+1];	/* Y Unit string	   */
extern int XUnitsSet, YUnitsSet, titleTextSet;
extern int _bwFlag;
extern int MonoChrome;		/* don't use colours	*/
extern int print_immediate;	/* post printdialog before drawing window	*/
extern double *do_gsTextWidth_Batch;
extern int htickFlag, vtickFlag;			/* Don't draw full grid    */
extern int zeroFlag;			/* draw zero-lines?	*/
extern int noExpX, noExpY;	/* don't use engineering notation	*/
extern int axisFlag;
extern int bbFlag;			/* Whether to draw bb      */
extern int noLines;			/* Don't draw lines        */
extern int legend_type, no_legend, legend_always_visible;
extern double legend_ulx, legend_uly, xname_x, xname_y,
	yname_x, yname_y;
extern int legend_placed, xname_placed, yname_placed, yname_vertical, intensity_legend_placed;
extern int no_title, AllTitles, no_intensity_legend;
extern int plot_only_file;
extern int *plot_only_set, plot_only_set_len;
extern int markFlag;			/* Draw marks at points    */
extern int arrows, overwrite_marks;	/* draw lines first, then marks; or reverse	*/
extern int overwrite_AxGrid, overwrite_legend;
extern int pixelMarks;			/* Draw pixel markers      */
extern int UseRealXVal,
	UseRealYVal;
extern int logXFlag;			/* Logarithmic X axis      */
extern int logYFlag;			/* Logarithmic Y axis      */
extern char log_zero_sym_x[MAXBUFSIZE+1], log_zero_sym_y[MAXBUFSIZE+1];	/* symbol to indicate log_zero	*/
extern int lz_sym_x, lz_sym_y;	/* log_zero symbols set?	*/
extern int log_zero_x_mFlag, log_zero_y_mFlag;
extern double log_zero_x, log_zero_y;	/* substitute 0.0 for these values when using log axis	*/
extern double log10_zero_x, log10_zero_y;
extern int sqrtXFlag, sqrtYFlag;
extern double powXFlag, powYFlag;
extern double powAFlag;
extern int use_xye_info;
extern double data[ASCANF_DATA_COLUMNS];
extern int column[ASCANF_DATA_COLUMNS];	/* order of x, y, error columns in input	*/
extern int autoscale;
extern int disconnect, split_set, splits_disconnect;
extern char *split_reason;

extern double _Xscale,	_Yscale, _DYscale, MXscale, MYscale, MDYscale;
extern double _MXscale, _MYscale, _MDYscale;
extern double Xscale, Yscale, DYscale;
extern double Xscale2, Yscale2, XscaleR;

extern int barFlag;			/* Draw bar graph          */
extern int triangleFlag;	/* draws error "triangles"	*/
extern int error_regionFlag;
extern int process_bounds, transform_axes, polarFlag, polarLog, absYFlag, raw_display, show_overlap;
extern int vectorFlag;
extern int vectorType;
extern double radix, radix_offset, vectorLength, vectorPars[MAX_VECPARS];
extern char radixVal[64], radix_offsetVal[64];
extern int barBase_set, barWidth_set, barType;
extern double barBase, barWidth;	/* Base and width of bars  */
extern int use_errors;		/* use error specifications?	*/
extern int no_errors;
extern double lineWidth;			/* Width of data lines     */
extern char *geoSpec ;		/* Geometry specification  */
extern int numFiles ;		/* Number of input files   */
extern int file_splits;		/* Number of split input files (catenations of multiple files)	*/
extern char **inFileNames; 	/* File names              */
extern char UPrintFileName[];
extern char *PrintFileName;
extern int PrintingWindow;
extern double newfile_incr_width;
extern int filename_in_legend;
extern int labels_in_legend;
extern char *Odevice ;		/* Output device   	   */
extern char *Odisp ; 	/* Output disposition      */
extern char *OfileDev ;		/* Output file or device   */
extern char *Oprintcommand;
extern int debugFlag , debugLevel;		/* Whether debugging is on */
extern double Xbias_thres, Ybias_thres;

extern double Xincr_factor, Yincr_factor;
extern int XLabels, YLabels;

/* Possible user specified bounding box */
extern int x_scale_start, y_scale_start;
extern int xp_scale_start, yp_scale_start, yn_scale_start;
extern double MusrLX, MusrRX, usrLpX, usrLpY, usrHnY, MusrLY, MusrRY;
extern int use_lx, use_ly;
extern int use_max_x, use_max_y;

/* Total number of active windows */
extern int Num_Windows ;
extern Window New_win;
extern char *Prog_Name;
extern int progname;
extern char Window_Name[256];

/* switch that controls the gathering of subsequent equal x-values
 * on a per set basis when dumping a SpreadSheet. Might mess up the
 * order in a set, and take very long!
 */
extern int Sort_Sheet;
int XG_SaveBounds= 0;
char XGBoundsString[512];

extern Pixmap dotMap ;

extern int XErrHandler();	/* Handles error messages */

extern LocalWin *CopyFlags( LocalWin *dest, LocalWin *src );

extern Cursor zoomCursor, labelCursor, cutCursor, filterCursor;
extern int nbytes, maxitems;

extern int do_transform( LocalWin *wi, char *filename, double line_count, char *buffer, int *spot_ok, DataSet *this_set,
	double *xvec, double *ldxvec, double *hdxvec, double *yvec, double *ldyvec, double *hdyvec,
	double *xvec_1, double *yvec_1, int use_log_zero, int spot, double xscale, double yscale, double dyscale, int is_bounds,
	int data_nr, Boolean just_doit
);

extern int _Handle_An_Event( XEvent *theEvent, int level, int CheckFirst, char *caller);

extern int Handle_An_Event( int level, int CheckFirst, char *caller, Window win, long mask);

long caller_process= 0;
int parent= 1;
char descent[16]= "parent";

extern LocalWindows *WindowList, *WindowListTail;

extern char **Argv, *InFiles, *InFilesTStamps;
extern int Argc;

Window thePrintWindow, theSettingsWindow;
LocalWin *thePrintWin_Info, *theSettingsWin_Info;

int TrueGray= 0;

int Exitting= False;

int XG_XSync( Display *disp, Bool discard )
{ int r;
	if( discard ){
	  XEvent evt[128];
	  int i= 0, e= 0;
		while( XEventsQueued( disp, QueuedAfterFlush) ){
			XNextEvent( disp, &evt[e]);
			if( e< 127 && !evt[e].xany.send_event &&
				(e== 0 || evt[e].xany.window!= evt[e-1].xany.window) && !Exitting
			){
				switch( evt[e].type ){
					case Expose:
						if( evt[e].xexpose.count > 0 ){
							break;
						}
					case ButtonPress:
					case ButtonRelease:
					case KeyPress:
					case KeyRelease:
						if( debugFlag ){
							fprintf( StdErr, "%s: storing event #%d (#%ld, %s, s_e=%d) for window %ld for resending after flush\n",
								(e==0)? "XG_XSync()" : "          ", e, evt[e].xany.serial, event_name(evt[e].type),
								evt[e].xany.send_event, evt[e].xany.window
							);
						}
						e+= 1;
						break;
				}
			}
			else if( debugFlag && debugLevel>= 1 ){
				fprintf( StdErr, "%s: discarding event #%d (#%ld, %s, s_e=%d) for window %ld\n",
					(e==0)? "XG_XSync()" : "          ", i, evt[e].xany.serial, event_name(evt[e].type),
					evt[e].xany.send_event, evt[e].xany.window
				);
			}
			i+= 1;
		}
		r= xtb_XSync( disp, discard);
		if( debugFlag ){
			fprintf( StdErr, "XG_XSync(): discarded queued events, resending %d stored..\n", e );
		}
		for( i= 0; i< e && !Exitting; i++ ){
			evt[i].xany.send_event= 1;
			XSendEvent( disp, evt[i].xexpose.window, False,
				ExposureMask|ButtonPressMask|ButtonReleaseMask|KeyPressMask|KeyReleaseMask, &evt[i]
			);
		}
	}
	else{
		r= xtb_XSync( disp, discard);
	}
	return( r );
}

void CleanUp();
unsigned int X11_c_KeyCode= 0;

void ExitProgramme(int ret)
{ LocalWindows *WL= WindowList;
	if( !StartUp && !Exitting ){
		XSync( disp, (ret==2)? False : True );
	}
	else if( ActiveWin && ActiveWin!= &StubWindow && (ret==1 || ret==2) && !Exitting ){
	  static XKeyEvent xkey;
	  extern XEvent *RelayEvent;
	  extern int RelayMask;
		  /* This is a somewhat precarious situation. We ended up here probably while executing
		   \ code in the X11 library internals. If we want to exit nicely, we'll have to give
		   \ X11 a chance to return from that, otherwise deadlock situations or worse can occur
		   \ on some servers (XFree86...), esp. when we afterwards call XSync(), XFlush() or
		   \ similar (and this can happen behind our backs!). The solution is to relay the signal
		   \ as an X11 event that we handle from within _Handle_An_Event(), that is, from within
		   \ our own code. The drawback is that we may end up in a situation where we don't check
		   \ for X11 events (e.g. if the user disables event checking in ascanf and creates a dead
		   \ loop in his code). Therefore, we only do this once.
		   */
		  /* Tell the world that we want to exit: */
		Exitting= True;
		  /* Initialise an XKey event to represent a KeyPress of the 'c' key with the Control pressed too: */
		xkey.type= KeyPress;
		xkey.display= disp;
		xkey.window= ActiveWin->window;
		xkey.root= RootWindow( disp, screen );
		xkey.state= ControlMask;
		  /* Use a previously stored keycode representation for the c key; calling XKeysymToKeycode() here can
		   \ block some X11 servers in some situations!!
		   */
		xkey.keycode= X11_c_KeyCode;
		xkey.send_event= 1;
		  /* Make sure that we remain the handler for the signal we just received: */
		signal( ret, ExitProgramme );
		  /* And make sure that, if all goes well (= the user's code that will continue to finish doesn't reset
		   \ it) processing of potentially "dead-loopy" code is switched off by selecting raw mode:
		   */
		ActiveWin->raw_display= 1;
		  /* Now we could send the event, and return to whoever called us, with fingers, toes and whatever crossed :)
		   \ However, it turns out that even in that case XFree84 4.01 can get locked-up. Therefore, we relay the event
		   \ through a global Event pointer, which then will be inserted in the event queue by a "causally" called
		   \ piece of code....
		   */
		  /* 
			XSendEvent( disp, ActiveWin->window, False, KeyPressMask|KeyReleaseMask, (XEvent*) &xkey);
			fprintf( StdErr, "ExitProgramme(%d): sent a ^C X11 keypress event; your request should be handled!\n", ret );
		   */
		RelayEvent= (XEvent*) &xkey;
		RelayMask= KeyPressMask|KeyReleaseMask;
		fprintf( StdErr, "ExitProgramme(%d): inserted a ^C X11 keypress event in the queue; your request should be handled!\n", ret );
		return;
		  /* If ever deadlocks still occur despite all this trouble, a switch should be installed that causes us
		   \ to exit *without* being nice to the X server. (NB: it is us that get in a lock situation waiting for the X
		   \ server: not the X server itself locking up!) Chances are that the X and C libraries are capable of handling
		   \ a brute exit quite well (they do when we crash, after all...)
		   */
	}
	thePrintWindow= 0;
	thePrintWin_Info= NULL;
	Exitting= True;
	if( Exit_Exprs ){
		  /* Force a reset on the internal state of the 2 routines likely to refuse executing our commands: */
		Evaluate_ExpressionList( NULL, NULL, False, NULL );
		new_param_now( NULL, NULL, 0 );
		  /* Now try to execute. */
		Evaluate_ExpressionList( ActiveWin, &Exit_Exprs, True, "exit expressions" );
		Exit_Exprs= XGStringList_Delete(Exit_Exprs);
	}
	while( /* Num_Windows && */ WL ){
	  LocalWin *lwi= WL->wi;
		  /* 20010901: RJVB: move WL to the next element in the list now,
		   \ because DelWindow() will free the current element...
		   */
		WL= WL->next;
		if( DelWindow( lwi->window, lwi) ){
			if( ret>= 0 ){
				xfree(lwi);
			}
		}
	}
	Num_Windows= 0;
	ActiveWin= NULL;
	if( ret>= 0 ){
		CleanUp();
		exit( ret );
	}
	else{
		longjmp( toplevel, ret);
	}
}

#define INITIAL_MAXSETS	128
int MaxSets= INITIAL_MAXSETS;
extern int ux11_min_depth;

extern int DumpDHex, DumpPens, DProcFresh;

int BackingStore_argset= False;

extern int BinarySize;

char *Collect_Arguments( LocalWin *wi, char *cbuf, int Len )
{ int len= 0;
  extern int DiscardedShadows;
	if( !cbuf ){
		return( NULL );
	}
	cbuf[0]= '\0';
	if( !wi || !Len ){
		return( cbuf );
	}
	if( install_flag || depth!= DefaultDepth(disp,screen) || ux11_useDBE ){
		if( ux11_useDBE ){
			len= sprintf( &cbuf[strlen(cbuf)], "-use_XDBE1 " );
		}
		if( depth!= DefaultDepth(disp,screen) ){
			len= sprintf( &cbuf[strlen(cbuf)], "-MinBitsPPixel %d ", ux11_min_depth );
		}
		if( ux11_vis_class>= 0 ){
		  extern char *VisualClass[];
			len= sprintf( &cbuf[strlen(cbuf)], "-VisualType %s ", VisualClass[ux11_vis_class] );
		}
	}
	if( BackingStore_argset ){
	  extern int BackingStore;
		len= sprintf( &cbuf[strlen(cbuf)], "-bs%d ", BackingStore );
	}
	if( wi->data_silent_process ){
		len= sprintf( &cbuf[strlen(cbuf)], "-proc_ign_ev%d ", wi->data_silent_process );
	}
	if( progname && Prog_Name && *Prog_Name ){
		len= sprintf( &cbuf[strlen(cbuf)], "-progname %s ", Prog_Name );
	}
	if( MaxSets> INITIAL_MAXSETS ){
		len= sprintf( &cbuf[strlen(cbuf)], "-maxsets %d ", MaxSets );
	}
	if( detach> 0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-detach%s", (detach== 1)? "0 " : " " );
	}
	len= sprintf( &cbuf[strlen(cbuf)], "-separator %c ", ascanf_separator );
	if( !wi->axisFlag ){
		strcat( cbuf, "-noaxis " );
	}
	len= sprintf( &cbuf[strlen(cbuf)], "-bb%d ", (wi->bbFlag && wi->bbFlag!= -1)? 1 : 0 );
	len= sprintf( &cbuf[strlen(cbuf)], "-htk%d -vtk%d ", (wi->htickFlag>0)? 1 : 0, (wi->vtickFlag>0)? 1 : 0 );
#if 0
	if( wi->htickFlag ){
		if( wi->vtickFlag ){
			strcat( cbuf, "-tk " );
		}
		else{
			strcat( cbuf, "-htk " );
		}
	}
	else if( wi->vtickFlag ){
		strcat( cbuf, "-vtk " );
	}
#endif
	if( wi->xname_placed ){
		len= sprintf( &cbuf[strlen(cbuf)], "%s %g,%g ", (wi->xname_trans)? "-x_ul1" : "-x_ul",
			wi->xname_x, wi->xname_y );
	}
	if( wi->yname_placed ){
		len= sprintf( &cbuf[strlen(cbuf)], "%s %g,%g ", (wi->yname_trans)? "-y_ul1" : "-y_ul",
			wi->yname_x, wi->yname_y );
	}
	if( wi->yname_vertical ){
		strcat( cbuf, "-y_vert1 " );
	}
	if( wi->legend_placed ){
		len= sprintf( &cbuf[strlen(cbuf)], "%s %g,%g ", (wi->legend_trans)? "-legend_ul1" : "-legend_ul",
			wi->_legend_ulx, wi->_legend_uly );
	}
	if( wi->IntensityLegend.legend_placed ){
		len= sprintf( &cbuf[strlen(cbuf)], "%s %g,%g ", 
			(wi->IntensityLegend.legend_trans)? "-intensity_legend_ul1" : "-intensity_legend_ul",
			wi->IntensityLegend._legend_ulx, wi->IntensityLegend._legend_uly );
	}
	if( wi->use_average_error ){
		strcat( cbuf, "-average_error1 " );
	}
/* 
	if( !wi->use_errors ){
		strcat( cbuf, "-noerr " );
	}
	if( wi->triangleFlag ){
		strcat( cbuf, "-triangle " );
	}
	if( wi->error_region ){
		strcat( cbuf, "-error_region " );
	}
 */
	if( wi->no_title ){
		strcat( cbuf, "-notitle " );
	}
	if( AllTitles ){
		strcat( cbuf, "-AllTitles " );
	}
	{ extern int DrawColouredSetTitles;
		if( DrawColouredSetTitles ){
			strcat( cbuf, "-ColouredSetTitles " );
		}
	}
	if( wi->no_ulabels ){
		strcat( cbuf, "-nolabels " );
	}
	if( wi->no_legend ){
		strcat( cbuf, "-nolegend " );
	}
	if( wi->no_pens ){
		strcat( cbuf, "-nopens " );
	}
	if( wi->no_intensity_legend ){
		strcat( cbuf, "-nointensity_legend " );
	}
/* 	if( wi->legend_type!= legend_type ){	*/
/* 		len= sprintf( &cbuf[strlen(cbuf)], "%s-legendtype %d ", cbuf, wi->legend_type );	*/
/* 	}	*/
	if( wi->no_legend_box ){
		strcat( cbuf, "-nolegendbox " );
	}
	len= sprintf( &cbuf[strlen(cbuf)], "-mindlegend%d ", wi->legend_always_visible );
	if( !wi->filename_in_legend ){
		strcat( cbuf, "-fn0 " );
	}
	if( wi->labels_in_legend ){
		strcat( cbuf, "-lb1 " );
	}
	if( wi->sqrtXFlag> 0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-powx %g ", wi->powXFlag );
	}
	else{
		strcat( cbuf, "-sqrtx0 " );
	}
	if( wi->logXFlag> 0 ){
		strcat( cbuf, "-lnx ");
	}
	if( wi->sqrtYFlag> 0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-powy %g ", wi->powYFlag );
	}
	else{
		strcat( cbuf, "-sqrty0 " );
	}
	if( wi->logYFlag> 0 ){
		strcat( cbuf, "-lny ");
	}
	if( wi->absYFlag ){
		strcat( cbuf, "-absy ");
	}
	if( wi->polarFlag ){
		len= sprintf( &cbuf[strlen(cbuf)], "-polar%g ", wi->radix );
		if( wi->powAFlag!= 1.0 ){
			len= sprintf( &cbuf[strlen(cbuf)], "-powA %g ", wi->powAFlag );
		}
	}
	if( wi->vectorFlag> 0 ){
/* 		len= sprintf( &cbuf[strlen(cbuf)], "-vectors %g ", vectorLength );	*/
		switch( vectorType ){
			case 0:
			default:
				len= sprintf( &cbuf[strlen(cbuf)], "-vectorpars %d,%s ", vectorType,
					d2str( vectorLength, "%.8g", NULL )
				);
				break;
			case 1:
				len= sprintf( &cbuf[strlen(cbuf)], "-vectorpars %d,%s,%s,%s ", vectorType,
					d2str( vectorLength, "%.8g", NULL ),
					d2str( vectorPars[0], "%.8g", NULL ),
					d2str( vectorPars[1], "%.8g", NULL )
				);
				break;
		}
	}
	if( barBase_set ){
		len= sprintf( &cbuf[strlen(cbuf)], "-brb %g ", barBase );
	}
	if( barWidth_set ){
		len= sprintf( &cbuf[strlen(cbuf)], "-brw %g ", barWidth );
	}
	if( barType ){
		len= sprintf( &cbuf[strlen(cbuf)], "-brt %d ", barType );
	}
	if( wi->Xincr_factor!= 1.0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-xstep %g ", wi->Xincr_factor );
	}
	if( wi->Yincr_factor!= 1.0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-ystep %g ", wi->Yincr_factor );
	}
	if( wi->ValCat_X_incr!= 1.0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-ValCat_xstep %g ", wi->ValCat_X_incr );
	}
	if( wi->ValCat_Y_incr!= 1.0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-ValCat_ystep %g ", wi->ValCat_Y_incr );
	}
	if( wi->Xbias_thres ){
		len= sprintf( &cbuf[strlen(cbuf)], "-biasX %g ", wi->Xbias_thres );
	}
	if( wi->Ybias_thres ){
		len= sprintf( &cbuf[strlen(cbuf)], "-biasY %g ", wi->Ybias_thres );
	}
	if( wi->zeroFlag ){
		len= sprintf( &cbuf[strlen(cbuf)], "-zl%d ", wi->zeroFlag );
	}
	if( wi->Xscale!= 1.0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-Scale_X %g ", wi->Xscale );
	}
	if( wi->Yscale!= 1.0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-Scale_Y %g ", wi->Yscale );
	}
	if( !wi->process_bounds ){
		strcat( cbuf, "-process_bounds0 " );
	}
	len= sprintf( &cbuf[strlen(cbuf)], "-transform_axes%d ", (wi->transform_axes<=0)? 0 : 1 );
	if( wi->raw_display ){
		strcat( cbuf, "-raw_display1 " );
	}
	if( wi->overwrite_AxGrid ){
		strcat( cbuf, "-overwrite_AxGrid1 " );
	}
	if( wi->overwrite_legend ){
		strcat( cbuf, "-overwrite_legend1 " );
	}
	if( wi->show_overlap ){
		len= sprintf( &cbuf[strlen(cbuf)], "-show_overlap1 %s", (wi->show_overlap==2)? "raw " : "" );
	}
	if( wi->log_zero_y || wi->log_zero_x ){
		len= sprintf( &cbuf[strlen(cbuf)], "-log_zero_xy %g,%g ", wi->log_zero_x, wi->log_zero_y );
	}
	if( wi->log_zero_x_mFlag ){
		len= sprintf( &cbuf[strlen(cbuf)], "-log_zero_x_%s ", (wi->log_zero_x_mFlag< 0)? "min" : "max" );
	}
	if( wi->log_zero_y_mFlag ){
		len= sprintf( &cbuf[strlen(cbuf)], "-log_zero_y_%s ", (wi->log_zero_y_mFlag< 0)? "min" : "max" );
	}
	if( wi->lz_sym_x ){
		len= sprintf( &cbuf[strlen(cbuf)], "-log_zero_sym_x %s ", strlen(wi->log_zero_sym_x)? wi->log_zero_sym_x : "\"\"" );
	}
	if( wi->lz_sym_y ){
		len= sprintf( &cbuf[strlen(cbuf)], "-log_zero_sym_y %s ", strlen(wi->log_zero_sym_y)? wi->log_zero_sym_y : "\"\"" );
	}
/* 
	if( wi->dump_average_values ){
		strcat( cbuf, "-DumpAverage1 " );
	}
	if( wi->DumpProcessed ){
		strcat( cbuf, "-DumpProcessed1 " );
	}
 */
	if( wi->DumpBinary ){
		len= sprintf( &cbuf[strlen(cbuf)], "-DumpBinary%d ", BinarySize );
	}
	if( wi->DumpAsAscanf ){
		strcat( cbuf, "-DumpAsAscanf1 " );
	}
	if( DumpDHex ){
		strcat( cbuf, "-DumpDHex1 " );
	}
	if( use_xye_info ){
		strcat( cbuf, "-Cauto " );
	}
	if( wi->win_geo.nobb_range_X ){
		if( wi->win_geo.user_coordinates ){
			len= sprintf( &cbuf[strlen(cbuf)], " -nbb_LX %s,%s ",
				d2str( wi->win_geo.nobb_loX, "%.8g", NULL),
				d2str( wi->win_geo.nobb_hiX, "%.8g", NULL)
			);
		}
		else{
			len= sprintf( &cbuf[strlen(cbuf)], " -nbb_lx %s,%s ",
				d2str( wi->win_geo.nobb_loX, "%.8g", NULL),
				d2str( wi->win_geo.nobb_hiX, "%.8g", NULL)
			);
		}
	}
	if( wi->win_geo.nobb_range_Y ){
		if( wi->win_geo.user_coordinates ){
			len= sprintf( &cbuf[strlen(cbuf)], " -nbb_LY %s,%s ",
				d2str( wi->win_geo.nobb_loY, "%.8g", NULL),
				d2str( wi->win_geo.nobb_hiY, "%.8g", NULL)
			);
		}
		else{
			len= sprintf( &cbuf[strlen(cbuf)], " -nbb_ly %s,%s ",
				d2str( wi->win_geo.nobb_loY, "%.8g", NULL),
				d2str( wi->win_geo.nobb_hiY, "%.8g", NULL)
			);
		}
	}
	{ int len;
		if( wi->aspect ){
			len= sprintf( XGBoundsString, " -lx %s,%s -ly %s,%s -XGBounds1 ",
				d2str( wi->win_geo.aspect_base_bounds._loX, "%.8g", NULL),
				d2str( wi->win_geo.aspect_base_bounds._hiX, "%.8g", NULL),
				d2str( wi->win_geo.aspect_base_bounds._loY, "%.8g", NULL),
				d2str( wi->win_geo.aspect_base_bounds._hiY, "%.8g", NULL)
			);
		}
		else if( wi->win_geo.user_coordinates ){
			len= sprintf( XGBoundsString, " -LX %s,%s -LY %s,%s -XGBounds1 ",
				d2str( wi->loX, "%.8g", NULL),
				d2str( wi->hiX, "%.8g", NULL),
				d2str( wi->loY, "%.8g", NULL),
				d2str( wi->hiY, "%.8g", NULL)
			);
		}
		else{
			len= sprintf( XGBoundsString, " -lx %s,%s -ly %s,%s -XGBounds1 ",
				d2str( wi->loX, "%.8g", NULL),
				d2str( wi->hiX, "%.8g", NULL),
				d2str( wi->loY, "%.8g", NULL),
				d2str( wi->hiY, "%.8g", NULL)
			);
		}
		if( len > sizeof(XGBoundsString)/sizeof(char) ){
			fprintf( StdErr, "Collect_Arguments(): wrote %d bytes in %d long XGBoundsString - prepare for crash!\n",
				len, sizeof(XGBoundsString)/sizeof(char)
			);
		}
	}
	if( XG_SaveBounds ){
		len= sprintf( &cbuf[strlen(cbuf)], "%s", XGBoundsString );
	}
	if( wi->fit_xbounds> 0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-fit_xbounds%d ", wi->fit_xbounds );
	}
	if( wi->fit_ybounds> 0 ){
		strcat( cbuf, "-fit_ybounds " );
	}
	len= sprintf( &cbuf[strlen(cbuf)], "-fit_after%d ", wi->fit_after_draw );
	len= sprintf( &cbuf[strlen(cbuf)], "-exact_X%d ", wi->exact_X_axis );
	len= sprintf( &cbuf[strlen(cbuf)], "-exact_Y%d ", wi->exact_Y_axis );
	len= sprintf( &cbuf[strlen(cbuf)], "-ValCat_X%d ", wi->ValCat_X_axis );
	len= sprintf( &cbuf[strlen(cbuf)], "-ValCat_X_levels %d ", wi->ValCat_X_levels );
	len= sprintf( &cbuf[strlen(cbuf)], "-ValCat_X_grid%d ", wi->ValCat_X_grid );
	len= sprintf( &cbuf[strlen(cbuf)], "-ValCat_Y%d ", wi->ValCat_Y_axis );
	len= sprintf( &cbuf[strlen(cbuf)], "-all_ValCat_I%d ", wi->show_all_ValCat_I );
	len= sprintf( &cbuf[strlen(cbuf)], "-ValCat_I%d ", wi->ValCat_I_axis );
	len= sprintf( &cbuf[strlen(cbuf)], "-aspect%d ", wi->aspect );
	len= sprintf( &cbuf[strlen(cbuf)], "-x_symm%d ", wi->x_symmetric );
	len= sprintf( &cbuf[strlen(cbuf)], "-y_symm%d ", wi->y_symmetric );
	if( preserve_screen_aspect ){
		strcat( cbuf, "-preserve_aspect " );
	}
	if( MonoChrome== 2 ){
		strcat( cbuf, "-monochrome " );
	}
	if( DiscardedShadows> 0 ){
		len= sprintf( &cbuf[strlen(cbuf)], "-DPShadow%d ", DiscardedShadows );
	}
	if( TrueGray ){
		len= sprintf( &cbuf[strlen(cbuf)], "-TrueGray%d ", TrueGray );
	}
	{ extern void *dm_python;
	  extern int UsePythonVersion;
		if( dm_python && UsePythonVersion > 0 ){
			len= sprintf( &cbuf[strlen(cbuf)], "-python%d ", UsePythonVersion );
		}
	}
	if( len> Len ){
		fprintf( StdErr, "Collect_Arguments(): wrote %d bytes in %d long buffer - prepare for crash!\n",
			len, Len
		);
	}
	return( cbuf );
}

extern int XGStoreColours, XGIgnoreCNames;

void Restart(LocalWin *wi, FILE *showfp )
{
  char *c= cgetenv( "XGRAPHCOMMAND" ), cbuf[1152], *env= NULL, *lenv= NULL;
  int ret, set_env= 0;
  char *command= NULL, *bg= (showfp)? "" : " &";
  char fname[256];
	Collect_Arguments( wi, cbuf, 1152);
	if( cbuf[0] ){
		if( wi->dev_info.resized== 1 ){
			sprintf( cbuf, "%s -print_sized", cbuf );
		}
		if( UPrintFileName[0] ){
			env= concat( "XGRAPH_ARGUMENTS=", cbuf, " -pf ", UPrintFileName, NULL);
		}
		else if( PrintFileName ){
			env= concat( "XGRAPH_ARGUMENTS=", cbuf, " -pf ", PrintFileName, NULL);
		}
		else{
			env= concat( "XGRAPH_ARGUMENTS=", cbuf, NULL);
		}
		set_env= 1;
	}
	  /* See if there's some runtime data that we would like to show up in the
	   \ restarted graph.
	   */
	if( !showfp ){
	  UserLabel *ul= wi->ulabel;
	  FILE *fp;
	  extern pid_t getpid();
	  int idx, NumObs= 0, labels= 0, points= 0;
		sprintf( fname, "/tmp/xgraph_labels.%d", getpid() );
		fp= fopen( fname, "wb" );
		if( fp ){
			XStoreName( disp, wi->window, "Saving information on average-sets");
			XG_XSync( disp, False );
			lenv= concat( "XGRAPH_LABELS=", fname, NULL );
			  /* Check for average sets added runtime. 	*/
			for( idx= 0; idx< setNumber; idx++ ){
				if( AllSets[idx].internal_average && AllSets[idx].average_set && !wi->dump_average_values){
					  /* This routine dumps the necessary headers, and the
					   \ actual *AVERAGE* command.
					   */
					if( AllSets[idx].setName ){
						print_string( fp, "*LEGEND*", "\\n\n", "\n", AllSets[idx].setName );
					}
					if( (wi->new_file[idx]) && AllSets[idx].fileName ){
						print_string( fp, "*FILE*", "\\n\n", "\n\n", AllSets[idx].fileName );
					}
					else if( idx== 0 || strcmp( AllSets[idx].fileName, AllSets[idx-1].fileName) ){
						print_string( fp, "*file*", "\\n\n", "\n\n", AllSets[idx].fileName );
					}
					_DumpSetHeaders( fp, wi, &AllSets[idx], &points, &NumObs, True /* , 0 */ );
				}
			}
			if( ul ){
				XStoreName( disp, wi->window, "Saving UserLabels");
				XG_XSync( disp, False );
				  /* Now see if there are UserLabels to be passed on.	*/
				while( ul ){
					if( ul->rt_added ){
					  /* Save only runtime-added labels; the others will already be
					   \ read from file, and we don't want them twice!
					   */
/* 						fprintf( fp, "*ULABEL* %s %s %s %s %d %d %d %d %d %d",	*/
						fprintf( fp, "*ULABEL* %s %s %s %s set=%d transform?=%d draw?=%d lpoint=%d vertical?=%d nobox?=%d lWidth=%g type=%s",
							d2str( ul->x1, d3str_format, NULL),
							d2str( ul->y1, d3str_format, NULL),
							d2str( ul->x2, d3str_format, NULL),
							d2str( ul->y2, d3str_format, NULL),
							ul->set_link, ul->do_transform, ul->do_draw, ul->pnt_nr, ul->vertical, ul->nobox,
							ul->lineWidth, ULabelTypeNames[ul->type]
						);
						if( XGStoreColours ){
							fprintf( fp, " cname=\"%s\"",
								(ul->pixvalue< 0)? ul->pixelCName :
									(ul->pixlinked && ul->set_link>= 0 && ul->set_link< setNumber)? "linked" : "default"
							);
						}
						fprintf( fp, "\n%s\n\n",
							ul->label
						);
						labels+= 1;
					}
					ul= ul->next;
				}
			}
			fclose(fp);
			if( labels || points || NumObs ){
			  /* We really created some output. So we should use the file.	*/
				putenv( lenv );
				if( debugFlag ){
					fprintf( StdErr, "Restart(): XGRAPH_LABELS=%s\n", getenv("XGRAPH_LABELS"));
				}
			}
			else{
			  /* Empty file. No need to keep it, nor to tell our child to include it.	*/
				unlink( fname );
			}
		}
	}
	if( !c ){
	  char *exec;
		if( showfp ){
			exec= "";
		}
		else if( getenv("XGRAPH_ARGUMENTS") ){
			exec= "echo \"Old: args=$XGRAPH_ARGUMENTS\" ; exec ";
		}
		else{
			exec= "exec ";
		}
		if( UPrintFileName[0] ){
			command= concat( exec, "xgraph ", cbuf, " -pf ", UPrintFileName, " -fli0 ", InFiles, bg, NULL);
		}
		else if( PrintFileName ){
			command= concat( exec, "xgraph ", cbuf, " -pf ", PrintFileName, " -fli0 ", InFiles, bg, NULL);
		}
		else{
			command= concat( exec, "xgraph ", cbuf, " -fli0 ", InFiles, bg, NULL);
		}
		if( command ){
			if( debugFlag ){
				fprintf( StdErr, "Restart(): using current settings;\nsystem(%s)\n",
					command
				);
			}
			if( !showfp ){
				XStoreName( disp, wi->window, command);
				XG_XSync( disp, False );
				if( !(ret= system(command) ) ){
					xfree(env);
					xfree( command );
#ifdef __GNUC__
					ExitProgramme(0);
#else
					ExitProgramme(-1);
#endif
					return;
				}
				else{
					fprintf( StdErr, "Restart(): system(%s) returned %d (%s)\n",
						command, ret, serror()
					);
					xfree( env );
					xfree( command );
					if( fname[0] ){
						unlink(fname);
					}
				}
			}
			else{
				fprintf( showfp, "(%s)\n", command );
				fflush( showfp );
				xfree( env );
				xfree( command );
				return;
			}
		}
	}
	else{
	  char *exec= (showfp)? "" : "echo \"args=$XGRAPH_ARGUMENTS\" ; exec ";
		if( (command= concat( exec, c, bg, NULL)) ){
			if( debugFlag ){
				fprintf( StdErr, "Restart(): using $XGRAPHCOMMAND; system(%s)\n",
					command
				);
			}
			if( !showfp ){
				if( set_env ){
					putenv( env );
					if( debugFlag ){
						fprintf( StdErr, "Restart(): XGRAPH_ARGUMENTS=%s\n", getenv("XGRAPH_ARGUMENTS"));
					}
					set_env= 0;
				}
				if( !(ret= system(command) ) ){
					xfree( command );
#ifdef __GNUC__
					ExitProgramme(0);
#else
					ExitProgramme(-1);
#endif
					return;
				}
				else{
					fprintf( StdErr, "Restart(): system(%s) returned %d (%s)\n",
						command, ret, serror()
					);
					xfree( env );
					xfree( command );
					if( fname[0] ){
						unlink(fname);
					}
				}
			}
			else{
				fprintf( showfp, "(" );
				if( env || lenv ){
					fprintf( showfp, "env");
					if( env ){
						fprintf( showfp, " \"%s\"", env );
					}
					if( lenv ){
						fprintf( showfp, " \"%s\"", lenv );
					}
				}
				fprintf( showfp, " %s)\n", command );
				fflush( showfp );
				xfree( env );
				xfree( lenv );
				xfree( command );
				return;
			}
		}
		else{
			fprintf( StdErr, "Restart(): can't get memory for command string\n");
		}
	}
	if( !showfp ){
		fprintf( StdErr, "Restart(): restarting program with original argv,argc (default)\n" );
		if( set_env ){
			putenv( env );
			fprintf( StdErr, "\tXGRAPH_ARGUMENTS=%s\n", getenv("XGRAPH_ARGUMENTS"));
			set_env= 0;
		}
		ret= execv( Argv[0], Argv);
		fprintf( StdErr, "Restart(): execv(%s,argv) returns %d\n",
			Argv[0], ret
		);
		if( fname[0] ){
			unlink(fname);
		}
	}
	else{
	  int i;
		fprintf( showfp, "(" );
		if( env || lenv ){
			fprintf( showfp, "env");
			if( env ){
				fprintf( showfp, " \"%s\"", env );
			}
			if( lenv ){
				fprintf( showfp, " \"%s\"", lenv );
			}
		}
		fprintf( showfp, " %s)\n", Argv[0] );
		for( i= 1; i< Argc; i++ ){
			fprintf( showfp, " %s", Argv[i] );
		}
		fprintf( showfp, ")\n" );
		fflush( showfp );
	}
	xfree( env );
	xfree( lenv );
	xfree( command );
	if( fname[0] ){
		unlink(fname);
	}
}

void Restart_handler( int sig)
{
	Restart( ActiveWin, NULL );
	return;
}

void Dump_handler( int sig)
{  char errmesg[ERRBUFSIZE];
   static char *name= NULL, *wname= NULL;
   static FILE *fp= NULL;
   int n= 0;
   LocalWin *wi;
   LocalWindows *WL= WindowList;

	signal( SIGUSR1, Dump_handler );
	if( fp || name || wname ){
		fprintf( StdErr, "[Thanks! I was already dumping.. \"%s\" to \"%s\"!\n]",
			(wname)? wname : "?", (name)? name : "??"
		);
		if( fp ){
			fclose( fp );
			fp= NULL;
		}
		xfree( name );
		if( wname ){
			XFree( wname );
			wname= NULL;
		}
	}
	while( WL ){
		wi= WL->wi;
		sprintf( errmesg, ".sigdump_%d", n );
		if( wi->hard_devices[XGRAPH_DEVICE].dev_file[0] ){
			name= concat( wi->hard_devices[XGRAPH_DEVICE].dev_file, errmesg, NULL);
			fp= fopen( name, "wb");
		}
		if( !name || !fp ){
			fp= fopen( (name= concat( "XGraph.xg", errmesg, NULL)), "wb" );
		}
		XFetchName( disp, wi->window, &wname );
		if( fp ){
			*errmesg= '\0';
			_XGraphDump( wi, fp, errmesg );
			if( *errmesg ){
				fputs( errmesg, StdErr );
			}
			fprintf( StdErr, "Dump_Handler(%d): dumped window \"%s\" to \"%s\"\n",
				sig, wname, name
			);
			fprintf( fp, "*EXTRATEXT* dump generated by USR1 signal; window's titlebar was:\n%s\n",
				wname
			);
			if( TBARprogress_header ){
				fprintf( fp, "Process %d was maybe doing \"%s\"\n", getpid(), TBARprogress_header );
			}
			if( wi== ActiveWin ){
				fprintf( fp, "This window was currently active\n" );
			}
			fputs( "\n", StdErr );
			fclose( fp );
		}
		else{
			fprintf( StdErr, "Dump_handler(%d): can't open \"%s\" to dump window \"%s\" (%s)\n",
				sig, name, wname, serror()
			);
		}
		xfree( name );
		XFree( wname );
		WL= WL->next;
		n+= 1;
	}
	return;
}

extern Time_Struct Start;

int detach_notify_stdout= False;

void kill_caller_process()
{ FILE *out= (detach_notify_stdout)? stdout : StdErr;
	if( caller_process){
		Elapsed_Since( &Start, True );
		fprintf( out, "%s.%s %d: telling parent %ld to quit - u=%s s=%s t=%s\n",
			Prog_Name, descent, getpid(),
			caller_process,
			d2str( Start.Time, "%g", NULL),
			d2str( Start.sTime, "%g", NULL),
			d2str( Start.Tot_Time, "%g", NULL)
		);
		fflush( out );
		kill( caller_process, SIGUSR1);
		caller_process= 0;
		  /* re-attach stdin to the nulldevice to really detach from terminal	*/
		freopen( "/dev/null", "r+", stdin );
		  /* become a new group. This is the only way I know of to really detach
		   \ from the calling process. When detaching with a fork(), the child would
		   \ otherwise remain attached to a calling shell(script), so that even after
		   \ the parent has quit, an interrupt to that shell would terminate the child
		   \ (us). This is not what we want if we are to be detached!
		   */
		if( setsid()== -1 ){
			fprintf( StdErr, "%s.%s: error in setsid() (%s)", Prog_Name, descent, serror() );
		}
	}
}

void notify( int sig)
{ FILE *out= (detach_notify_stdout)? stdout : StdErr;
	if( parent ){
		Elapsed_Since( &Start, True );
		fprintf( out, "%s.%s: detaching... u=%s s=%s t=%s ",
			Prog_Name, descent,
			d2str( Start.Time, "%g", NULL),
			d2str( Start.sTime, "%g", NULL),
			d2str( Start.Tot_Time, "%g", NULL)
		);
		fflush( out );
		fprintf( out, ",, _exit\n");
		_exit(0);
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "%s.%s: Ignoring detach signal\n", Prog_Name, descent );
			fflush( StdErr );
		}
		signal( SIGUSR1, SIG_IGN );
	}
	return;
}

void cont_handler( int sig )
{
  LocalWindows *WL= WindowList;
  LocalWin *lwi;
	while( WL && WL->wi ){
		lwi= WL->wi;
		if( !lwi->delete_it ){
			xtb_bt_swap( lwi->settings );
			  /* 980528	*/
			lwi->event_level= 0;
			RedrawNow( lwi );
			if( lwi->delete_it!= -1 ){
				xtb_bt_swap( lwi->settings );
			}
		}
		if( WL ){
			WL= WL->next;
		}
	}
	signal( SIGCONT, cont_handler );
}

void close_pager()
{
	if( StdErr!= stderr){
	 FILE *fp= StdErr;
		StdErr= stderr;
		fprintf( StdErr, "%s.%s: closing pager\n", Prog_Name, descent );
		fflush( StdErr);
		pclose( fp);
		use_pager= 0;
	}
}

void Show_Stats(FILE *fp, char *label, SimpleStats *SSx, SimpleStats *SSy, SimpleStats *SSe, SimpleStats *SS_sy, SimpleStats *SS_SY)
{
	if( show_stats && SSx->count ){
	 int len;
	 int i;
	 extern Boolean ReadData_terpri;
#ifdef __GNUC__
	 char lbuf[strlen(label)+1];
#else
	 char lbuf[MAXBUFSIZE];
#endif

		if( ReadData_terpri ){
			fputs( "\n", StdErr);
			ReadData_terpri= False;
		}
		  /* make a SPACE-buffer with the same length as the label	*/
		for( i= 0; i< strlen(label); i++ ){
			lbuf[i]= ' ';
		}
		lbuf[i]= '\0';

		len= fprintf( fp, "%s: X<%ld>= [%s:%s] %s +- %s Y<%ld>= [%s:%s] %s +- %s E<%ld>= [%s:%s] %s +- %s\n",
			label,
			SSx->count,
			d2str( SSx->min, "%g", NULL),
			d2str( SSx->max, "%g", NULL),
			d2str( SS_Mean( SSx), "%g", NULL),
			d2str( SS_St_Dev( SSx), "%g", NULL),

			SSy->count,
			d2str( SSy->min, "%g", NULL),
			d2str( SSy->max, "%g", NULL),
			d2str( SS_Mean( SSy), "%g", NULL),
			d2str( SS_St_Dev( SSy), "%g", NULL),

			SSe->count,
			d2str( SSe->min, "%g", NULL),
			d2str( SSe->max, "%g", NULL),
			d2str( SS_Mean( SSe), "%g", NULL),
			d2str( SS_St_Dev( SSe), "%g", NULL)
		);
		if( SS_sy ){
			len+= fprintf( fp, "%s: Set-Y<%ld>= [%s:%s] %s +- %s\n",
				lbuf,
				SS_sy->count,
				d2str( SS_sy->min, "%g", NULL),
				d2str( SS_sy->max, "%g", NULL),
				d2str( SS_Mean( SS_sy), "%g", NULL),
				d2str( SS_St_Dev( SS_sy), "%g", NULL)
			);
		}
		if( SS_SY ){
			len+= fprintf( fp, "%s: All Y<%ld>= [%s:%s] %s +- %s\n",
				lbuf,
				SS_SY->count,
				d2str( SS_SY->min, "%g", NULL),
				d2str( SS_SY->max, "%g", NULL),
				d2str( SS_Mean( SS_SY), "%g", NULL),
				d2str( SS_St_Dev( SS_SY), "%g", NULL)
			);
		}
		if( len> 0 ){
#ifdef __GNUC__
		  char buf[len+32];
#else
		  char buf[MAXBUFSIZE];
#endif
		  size_t n, blen = sizeof(buf);
			n = snprintf( buf, blen, " %s: X<%ld>= [%s:%s] %s +- %s   Y<%ld>= [%s:%s] %s +- %s   E<%ld>= [%s:%s] %s +- %s\n",
				label,
				SSx->count,
				d2str( SSx->min, "%g", NULL),
				d2str( SSx->max, "%g", NULL),
				d2str( SS_Mean( SSx), "%g", NULL),
				d2str( SS_St_Dev( SSx), "%g", NULL),

				SSy->count,
				d2str( SSy->min, "%g", NULL),
				d2str( SSy->max, "%g", NULL),
				d2str( SS_Mean( SSy), "%g", NULL),
				d2str( SS_St_Dev( SSy), "%g", NULL),

				SSe->count,
				d2str( SSe->min, "%g", NULL),
				d2str( SSe->max, "%g", NULL),
				d2str( SS_Mean( SSe), "%g", NULL),
				d2str( SS_St_Dev( SSe), "%g", NULL)
			);
			if( SS_sy ){
				n += snprintf( &buf[n], blen-n, "%s: Set-Y<%ld>= [%s:%s] %s +- %s\n",
					lbuf,
					SS_sy->count,
					d2str( SS_sy->min, "%g", NULL),
					d2str( SS_sy->max, "%g", NULL),
					d2str( SS_Mean( SS_sy), "%g", NULL),
					d2str( SS_St_Dev( SS_sy), "%g", NULL)
				);
			}
			if( SS_SY ){
				n += snprintf( &buf[n], blen-n, "%s: All Y<%ld>= [%s:%s] %s +- %s\n",
					lbuf,
					SS_SY->count,
					d2str( SS_SY->min, "%g", NULL),
					d2str( SS_SY->max, "%g", NULL),
					d2str( SS_Mean( SS_SY), "%g", NULL),
					d2str( SS_St_Dev( SS_SY), "%g", NULL)
				);
			}
			add_comment(buf);
		}
		if( fp== StdErr && !use_pager && !( isatty( fileno(fp) ) && isatty( fileno(stdout) ) ) ){
			fprintf( stdout, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				label,
				d2str( SS_Mean( SSx), "%g", NULL),
				d2str( SS_St_Dev( SSx), "%g", NULL),
				d2str( SS_Mean( SSy), "%g", NULL),
				d2str( SS_St_Dev( SSy), "%g", NULL),
				d2str( SS_Mean( SSe), "%g", NULL),
				d2str( SS_St_Dev( SSe), "%g", NULL)
			);
		}
	}
}

#ifdef TOOLBOX

/*
 * Button handling functions
 */

/*ARGSUSED*/
xtb_hret del_func( Window win, int bval, xtb_data info)
/* Button window    */ /* Button value     */ /* User information */
/*
 * This routine is called when the `Close' button is pressed in
 * an xgraph window.  It causes the window to go away.
 */
{
    Window the_win = (Window) info;
    LocalWin *wi;

    if (!XFindContext(disp, the_win, win_context, (caddr_t *) &wi)) {
		bval= !wi->delete_it;
		xtb_bt_set(win, bval, (char *) 0);
		if( bval ){
			DelWindow(the_win, wi);
		}
		else{
			Boing(2);
			wi->delete_it= 0;
		}
    }
    return XTB_HANDLED;
}

int pw_centre_on_X= 0, pw_centre_on_Y= 0;

/*ARGSUSED*/
xtb_hret hcpy_func( Window win, int bval, xtb_data info)
/* Button Window    */ /* Button value     */ /* User Information */
/*
 * This routine is called when the hardcopy button is pressed
 * in an xgraph window.  It causes the output dialog to be
 * posted.
 */
{
    Window the_win = (Window) info;
    LocalWin *wi;
	int nw= Num_Windows;
	unsigned int mask= xtb_bt_event->xbutton.state;

    xtb_bt_set(win, 1, (char *) 0);
    if (!XFindContext(disp, the_win, win_context, (caddr_t *) &wi)) {
		wi->redraw= 0;
		wi->pw_placing= PW_PARENT;
		if( CheckMask(mask, ShiftMask) ){
			print_immediate= -1;
		}
		if( wi->HO_Dialog && wi->HO_Dialog->mapped && wi->HO_Dialog->parent== the_win ){
			wi->HO_Dialog->mapped= (wi->HO_Dialog->mapped< 0)? 1 : -1;
		}
		PrintWindow(the_win, wi);
    }
    if( nw== Num_Windows)
		xtb_bt_set(win, 0, (char *) 0);
    return XTB_HANDLED;
}

/*ARGSUSED*/
xtb_hret settings_func( Window win, int bval, xtb_data info)
/* Button Window    */ /* Button value     */ /* User Information */
/*
 * This routine is called when the settings button is pressed
 * in an xgraph window.  It causes the output dialog to be
 * posted.
 */
{
    Window the_win = (Window) info;
    LocalWin *wi;
	int nw= Num_Windows;

    xtb_bt_set(win, 1, (char *) 0);
    if (!XFindContext(disp, the_win, win_context, (caddr_t *) &wi)) {
		wi->redraw= 0;
		wi->pw_placing= PW_PARENT;
		if( wi->SD_Dialog && wi->SD_Dialog->mapped && wi->SD_Dialog->parent== the_win ){
			wi->SD_Dialog->mapped= (wi->SD_Dialog->mapped< 0)? 1 : -1;
		}
		DoSettings(the_win, wi);
    }
    if( nw== Num_Windows)
		xtb_bt_set(win, 0, (char *) 0);
    return XTB_HANDLED;
}

/*ARGSUSED*/
xtb_hret info_func( Window win, int bval, xtb_data info)
/* Button Window    */ /* Button value     */ /* User Information */
/*
 * This routine is called when the info button is pressed
 * in an xgraph window.  It causes the info window to popup
 */
{
    Window the_win = (Window) info;
    LocalWin *wi;
	int nw= Num_Windows;
	static int active= 0;

	if( !active ){
		xtb_bt_set(win, 1, (char *) 0);
		if (!XFindContext(disp, the_win, win_context, (caddr_t *) &wi)) {
		  int id;
		  char *sel= NULL;
		  xtb_frame *menu= NULL;
			active= 1;
			id= xtb_popup_menu( wi->window, comment_buf, "Comments &c. found in datafiles:", &sel, &menu);

			if( sel ){
				while( *sel && isspace( (unsigned char) *sel) ){
					sel++;
				}
			}
			if( sel && *sel ){
				if( debugFlag ){
					xtb_error_box( wi->window, sel, "Copied to clipboard:" );
				}
				else{
					Boing(10);
				}
				XStoreBuffer( disp, sel, strlen(sel), 0);
				  // RJVB 20081217
				xfree(sel);
			}
			xtb_popup_delete( &menu );
			active= 0;
		}
		if( nw== Num_Windows)
			xtb_bt_set(win, 0, (char *) 0);
		return XTB_HANDLED;
	}
	else{
		return XTB_NOTDEF;
	}
}

extern xtb_hret label_func( Window win, int bval, xtb_data info);

xtb_hret ssht_func( Window win, int bval, xtb_data info)
/* Button Window    */ /* Button value     */ /* User Information */
/*
 * This routine is called when the Ssht button is pressed
 * in an xgraph window.
 */
{
    Window the_win = (Window) info;
    LocalWin *wi;
	extern int X_silenced();

	if (!XFindContext(disp, the_win, win_context, (caddr_t *) &wi)) {
	  int os= wi->silenced;
		xtb_bt_set(win, (wi->silenced= !X_silenced(wi)), (char *) 0);
		wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
		if( !wi->silenced && os< 0 ){
			RedrawNow(wi);
		}
	}
	return XTB_HANDLED;
}
#endif

char *transform_description= NULL;
char *transform_x_buf= NULL, *transform_y_buf= NULL, transform_separator = '\0';
int transform_x_buf_len= 0, transform_y_buf_len= 0;
int transform_x_buf_allen= 0, transform_y_buf_allen= 0;

extern char *cleanup(char *);

int allocerr= 0;

void *_XGrealloc( void** ptr, size_t n, char *name, char *size )
{ void *mem;
	if( !(mem= ((ptr && *ptr)? realloc( *ptr, n) : calloc( n, 1))) ){
		if( name && size ){
			fprintf( StdErr, "xgraph::XGrealloc(): Error (re)allocating %s=0x%lx, size %s=%lu -> 0x%lx (%s)\n",
				name, ptr, size, (unsigned long) n, mem, serror()
			);
		}
		// 20120622: 
		if( ptr ){
			free(*ptr);
			*ptr = NULL;
		}
		allocerr+= 1;
	}
	return( mem );
}

#include <sys/mman.h>
int xgShFD = -1;
#define XGSHAREDMEMNAME	"/dev/zero"; //"XGShMem-XXXXXX"
char XgSharedMemName[64] = "";

void *_XGreallocShared( void* ptr, size_t N, size_t oldN, char *name, char *size )
{ void *mem;
  int flags = MAP_SHARED;
#ifndef MAP_ANON
	if( xgShFD < 0 ){
		if( !XgSharedMemName[0] ){
			strcpy( XgSharedMemName, XGSHAREDMEMNAME );
// 			mktemp(XgSharedMemName);
		}
 		if( (xgShFD = open( XgSharedMemName, O_RDWR )) < 0 ){
			fprintf( StdErr, "xgraph::XGreallocShared(): can't open/create descriptor for allocating %s=0x%lx, size %s=%lu -> 0x%lx (%s)\n",
				(name)? name : "<unknown>", ptr, size, (unsigned long) N, mem, serror()
			);
			return NULL;
		}
	}
#else
	flags |= MAP_ANON;
#endif
	mem = mmap( NULL, N, (PROT_READ|PROT_WRITE), flags, xgShFD, 0 );
	if( !mem ){
		if( name && size ){
			fprintf( StdErr, "xgraph::XGreallocShared(): Error (re)allocating %s=0x%lx, size %s=%lu -> 0x%lx (%s)\n",
				name, ptr, size, (unsigned long) N, mem, serror()
			);
		}
		allocerr+= 1;
	}
	else{
		// 20121228:
		madvise( mem, N, MADV_SEQUENTIAL );
		register_AllocatedMemorySize( mem, N );
		memset( mem, 0, N );
		if( ptr ){
		  size_t oN;
			// 20121228: ptr points to a block of size oldN, not N !!!
			if( get_AllocatedMemorySize( ptr, &oN ) ){
				if( oN != oldN ){
					fprintf( StdErr, "xgraph::XGreallocShared(): buffer 0x%p has size %lu but is claimed to have size %lu\n",
						   ptr, oN, oldN );
				}
			}
			// 20121228: and evidently one should not attempt to copy more into mem than it can contain...
			// we're not only handling cases where memory is reallocated LARGER than it was...!
 			memmove( mem, ptr, MIN(N,oldN) );
			munmap( ptr, oldN );
		}
	}
	return( mem );
}

void XGfreeShared(void **ptr, size_t N)
{
	if( ptr && *ptr && N ){
		munmap( *ptr, N );
		*ptr = NULL;
	}
}

double *param_scratch= NULL;
int param_scratch_len= 0, param_scratch_inUse= 0;

double *clean_param_scratch()
{
	  /* Check if we need to (re-)allocate the scratch memory, e.g. because of a call
	   \ to MaxArguments[] or to Ascanf_AllocMem(). (NB: only expansion is handled here...)
	   */
	if( !param_scratch || ASCANF_MAX_ARGS> param_scratch_len ){
	  int psl= param_scratch_len;
		if( param_scratch_inUse ){
			if( debugFlag ){
				fprintf( StdErr, "clean_param_scratch(): arena in use; remains at %d elements\n",
					param_scratch_len
				);
			}
		}
		else{
			if( !ASCANF_MAX_ARGS ){
				ASCANF_MAX_ARGS= AMAXARGSDEFAULT;
			}
			if( ASCANF_MAX_ARGS> param_scratch_len ){
				param_scratch_len= ASCANF_MAX_ARGS;
			}
			if( !(param_scratch= (double*) XGrealloc( param_scratch, param_scratch_len* sizeof(double))) ){
				fprintf( StdErr, "Can't get memory for param_scratch[%d] array (%s)\n", param_scratch_len , serror() );
				return(NULL);
			}
			else if( (ascanf_verbose && psl) || debugFlag ){
				fprintf( StdErr, "clean_param_scratch(): re-allocated arena from %d to %d elements\n",
					psl, param_scratch_len
				);
			}
		}
		fflush( StdErr );
	}
	memset( param_scratch, 0, sizeof(double)* param_scratch_len );
	return( param_scratch );
}

extern Window ascanf_window;

int fascanf_eval( int *n, char *expression, double *vals, double *data, int *column, int compile )
{ int r= 0, N= *n;
  double val= vals[0];
  char *exprend;
	if( !expression ){
		return(0);
	}
	exprend= &expression[strlen(expression)-1];
	fascanf_unparsed_remaining= expression;
	  /* 20031116: made the whiledo loop in to a dowhile loop; a single digit was *not* evaluated because
	   \ fascanf_unparsed_remaining would start out being equal to exprend.
	   */
	do{
	  char *prev= fascanf_unparsed_remaining;
	  char *end_prev= &prev[strlen(prev)];
	  Boolean print_result;
		if( strncmp( fascanf_unparsed_remaining, "? ", 2 )== 0 && fascanf_unparsed_remaining[2] ){
			fascanf_unparsed_remaining+= 2;
			print_result= True;
		}
		else{
			print_result= False;
		}
		if( compile ){
		  struct Compiled_Form *form= NULL;
		  char *c= fascanf_unparsed_remaining;
			fascanf( n, c, param_scratch, NULL, data, column, &form );
			*n= N;
			vals[0]= val;
			*ascanf_self_value= val;
			*ascanf_current_value= val;
			reset_ascanf_currentself_value= 0;
			compiled_fascanf( n, c, vals, NULL, data, column, &form );
			r+= *n;
			Destroy_Form( &form );
		}
		else{
			fascanf( n, fascanf_unparsed_remaining, vals, NULL, data, column, NULL );
			r+= *n;
		}
		if( print_result && *n ){
		  int i;
			fprintf( StdErr, "#R# %s", ad2str( vals[0], d3str_format, NULL ) );
			for( i= 1; i< *n; i++ ){
				fprintf( StdErr, ", %s", ad2str( vals[i], d3str_format, NULL ) );
			}
			fprintf( StdErr, "\n" );
		}
		if( fascanf_unparsed_remaining< expression || fascanf_unparsed_remaining>= exprend ){
			fascanf_unparsed_remaining= NULL;
		}
		while( fascanf_unparsed_remaining && *fascanf_unparsed_remaining &&
				(isspace( *fascanf_unparsed_remaining ) ||
					(*fascanf_unparsed_remaining== '@' && (!fascanf_unparsed_remaining[1] || fascanf_unparsed_remaining[1]== '\n') )
				)
		){
			fascanf_unparsed_remaining++;
		}
		if( fascanf_unparsed_remaining ){
			if( *fascanf_unparsed_remaining== '#' || fascanf_unparsed_remaining<= prev || fascanf_unparsed_remaining>=end_prev ){
				fascanf_unparsed_remaining= NULL;
			}
		}
		if( fascanf_unparsed_remaining && *fascanf_unparsed_remaining && (debugFlag || ascanf_verbose || scriptVerbose) ){
			fprintf( StdErr,
				"fascanf_eval(): stored %d results in %d-long vector; parsing next expression(list);\n"
				"                return value will be the value of this expression (and not %s)!\n"
				"                \"%s\"\n"
				, r, N, ad2str( vals[0], d3str_format, NULL), fascanf_unparsed_remaining
			);
		}
	}
	while( fascanf_unparsed_remaining>= expression && fascanf_unparsed_remaining< exprend && *fascanf_unparsed_remaining );
	fascanf_unparsed_remaining= NULL;
	return(r);
}

char *backwcomp(char *buf)
{
	if( buf ){
	  char *c= strstr(buf, " @ # label=\"");
		  /* 20030414:
		   \ earlier versions would create statements like
		   \ *EVAL* DCL[foo,bar] @ # label="boo.xg"
		   \ The current parser balks at this, wanting
		   \ *EVAL* DCL[foo,bar] # label={boo.xg} @
		   \ Therefore, send the " @" to the
		   \ end of the line and change the quotes for curly braces.
		   \ This can be done in place.
		   */
		if( c ){
		  char *d= &c[2];
		  int inlabel= 0;
			while( d && *d && !index("\r\n", *d) ){
				if( *d== '"' ){
					if( d[-1]== '=' ){
						*d= '{';
					}
					else{
						if( inlabel && (!d[1] || isspace(d[1])) ){
							*d= '}';
						}
					}
					inlabel= !inlabel;
				}
				*c++ = *d++;
			}
			*c++ = ' ';
			*c++ = '@';
			while( d && *d ){
				*c++ = *d++;
			}
		}
	}
	return(buf);
}

int new_param_now( char *ExprBuf, double *val, int N)
{ static char active= 0;
  double result= 0;
  int orgN = N;

	if( !ExprBuf && !val ){
		active= 0;
		return(0);
	}
	else{
	  int rsacsv= reset_ascanf_currentself_value, n, r= 0;
	  char *TBh= TBARprogress_header;

		/* in GNU gcc, LMAXBUFSIZE is a variable	*/

	  char *exprbuf= SubstituteOpcodes( backwcomp(ExprBuf), "*Print-File*", &PrintFileName, NULL );
	  int lexp= (exprbuf && *exprbuf)? strlen(exprbuf) : 0;

// 	  ALLOCA( membuf, char, lexp+ LMAXBUFSIZE+1, membuf_len);
// 	  ALLOCA( optbuf, char, lexp+ LMAXBUFSIZE+1, optbuf_len);
	  char *membuf, *optbuf;

	  extern int ascanf_check_int, ascanf_PopupWarn;
	  int aci= ascanf_check_int, aci_val= 10, apw= ascanf_PopupWarn;
	  Boolean compile;
	  char *Title= NULL;
	  static long prev_hash= 0, prev_len= 0;
	  long hash, len;
	  Window aw= ascanf_window;
		/* 20001110:
		 \ It *could be important to use a local copy for the column array...!
		 */
	  int column[ASCANF_DATA_COLUMNS]= {0,1, 2, 3};
	  extern int *dbF_cache;
	  int dbF= debugFlag, *dbFc= dbF_cache;

		if( !exprbuf ){
			GCA();
			return(0);
		}
		if( active ){
			if( SD_Dialog.win ){
				xtb_error_box( SD_Dialog.win, "Please, wait 'till the current *EVAL* command finishes...", "Notice" );
			}
			else{
				fputs( "Please, wait 'till the current *EVAL* command finishes...\n", StdErr );
			}
			GCA();
			return(0);
		}

		membuf = (char*) calloc( lexp+ LMAXBUFSIZE+1, sizeof(char) );
		optbuf = (char*) calloc( lexp+ LMAXBUFSIZE+1, sizeof(char) );
		if( !membuf || !optbuf ){
			xfree(membuf);
			xfree(optbuf);
			fprintf( StdErr, "new_param_now() couldn't allocate 1 or 2 buffers of size %d (%s)\n",
				lexp+ LMAXBUFSIZE+1, serror()
			);
			return(0);
		}

		if( !val ){
			val= &result;
		}

		if( ActiveWin ){
			if( ActiveWin->debugFlag== 1 ){
				debugFlag= True;
			}
			else if( ActiveWin->debugFlag== -1 ){
				debugFlag= False;
			}
		}
		dbF_cache= &dbF;

		strcpy( membuf, exprbuf );
		clean_param_scratch();

		active= 1;
		if( N== 0 ){
			N= 1;
		}
		else if( N< 0 ){
			N= param_scratch_len;
		}
		n= N;
		param_scratch[0]= *val;
		*ascanf_self_value= *val;
		*ascanf_current_value= *val;
		reset_ascanf_currentself_value= 0;
		cleanup( membuf );
		len= strlen(membuf);
		hash= ascanf_hash( membuf, NULL );
		sprintf( optbuf, "*EVAL* %s [%ld]\n", membuf, hash );
		if( ascanf_verbose ){
			fputs( optbuf, StdErr);
			fflush( StdErr );
		}
		if( SD_Dialog.win && SD_Dialog.mapped ){
			XFetchName( disp, SD_Dialog.win, &Title );
			XStoreName( disp, SD_Dialog.win, optbuf );
		}
		if( !TBARprogress_header ){
			TBARprogress_header= "*EVAL*";
		}

		  /* *EVAL* expressions are not compiled as they will typically be
		   \ evaluated only once. Unless exprs containing a "for-to" or "while" command, as long
		   \ as there is no DEPROC[] call starting the line.
		   */
		ascanf_check_int= aci_val;
		ascanf_PopupWarn= (QuietErrors)? False : True;
		if( strncmp( membuf, "ASKPROC", 7)== 0 || (strncmp( membuf, "DEPROC", 6)== 0 && !index( "*", membuf[6])) ){
			compile= False;
		}
		else{
			compile= ( (strstr( membuf, "for-to") || strstr( membuf, "while") || strstr(membuf, "Apply2Array"))
				&& !strstr( membuf, "Delete[") )? True : False;
			if( compile && debugFlag ){
				fprintf( StdErr, "*EVAL*: evaluating compiled expression containing loop-construct(s)\n");
			}
		}
		if( ActiveWin ){
			ascanf_window= ActiveWin->window;
		}
		r= fascanf_eval( &n, membuf, param_scratch, data, column, compile );
		if( !ascanf_arg_error && n ){
			if( hash!= prev_hash || len!= prev_len ){
				Add_Comment( optbuf, True );
			}
			add_process_hist( membuf );
			prev_hash= hash;
			prev_len= len;
		}
		  /* 20060424: it should be possible to change ascanf_check_int through the CheckEvent ascanf function: */
		if( ascanf_check_int== aci_val ){
			ascanf_check_int= aci;
		}
		ascanf_PopupWarn= apw;
		ascanf_window= aw;
		reset_ascanf_currentself_value= rsacsv;
		TBARprogress_header= TBh;

		if( exprbuf!= ExprBuf ){
			xfree( exprbuf );
		}
		GCA();

		if( SD_Dialog.win && SD_Dialog.mapped && Title ){
			XStoreName( disp, SD_Dialog.win, Title );
			XFree( Title );
		}
		active= 0;
		*val= param_scratch[0];
		// 20101105: we really ought to return more than just a single value!!
		{ int i;
			for( i = 1 ; i < r && i < orgN ; i++ ){
				val[i] = param_scratch[i];
			}
		}
		debugFlag= dbF;
		dbF_cache= dbFc;
		xfree(membuf);
		xfree(optbuf);
		return(r);
	}
}

int new_param_now_allwin( char *ExprBuf, double *val, int N)
{ LocalWindows *WL= WindowList;
  LocalWin *AW= ActiveWin;
  extern Window ascanf_window;
  Window aw= ascanf_window;
  int r= 0;
	if( !WL || InitWindow ){
		r= new_param_now( ExprBuf, val, N );
	}
	else{
		while( WL ){
			if( WL->wi ){
				ActiveWin= WL->wi;
				ascanf_window= ActiveWin->window;
				r+= new_param_now( ExprBuf, val, N );
				WL= WL->next;
			}
		}
		ActiveWin= AW;
		ascanf_window= aw;
	}
	return(r);
}

int interactive_param_now_xwin( Window win, char *expr, int tilen, int maxlen, char *message, double *x,
	int modal, double verbose, int AllWin
)
{ extern xtb_hret display_ascanf_variables_h();
  extern xtb_hret process_hist_h();
  int r= 0;
  extern Window ascanf_window;
  Window aw= ascanf_window;
  char title[128], *nbuf;
  Sinc sinc;
	if( AllWin ){
		strcpy( title, "#x01Enter *EVAL* expression(s) !!Applied to ALL open windows!!" );
	}
	else{
		strcpy( title, "#x01Enter *EVAL* expression(s)" );
	}
	if( *expr ){
		  /* 20020313, 20020317 */
		Sinc_string_behaviour( &sinc, strdup(expr), 0,0, SString_Dynamic );
		Sflush(&sinc);
		Srewind(&sinc);
		Sprint_string( &sinc, "", NULL, "", expr );
		strncpy( expr, sinc.sinc.string, maxlen );
		xfree( sinc.sinc.string);
	}
	if( (nbuf= xtb_input_dialog( win, expr, tilen, maxlen, message,
			parse_codes(title),
			modal,
			"Defined Vars/Arrays", display_ascanf_variables_h,
			"History", process_hist_h,
			"Edit", SimpleEdit_h
		))
	){
		cleanup( expr );
		if( expr[0] ){
			strcat( expr, "\n");
			if( verbose ){
				fprintf( StdErr, "# %s", expr );
			}
			ascanf_window= win;
			r= (AllWin)? new_param_now_allwin(expr, x, -1) : new_param_now( expr, x, -1 );
			ascanf_window= aw;
		}
		if( nbuf!= expr ){
			xfree( nbuf );
		}
	}
	return(r);
}

int interactive_param_now( LocalWin *wi, char *expr, int tilen, int maxlen, char *message, double *x, int modal, int verbose )
{ LocalWin *aw= ActiveWin;
  int r;
	if( wi ){
		ActiveWin= wi;
	}
	r= interactive_param_now_xwin( wi->window, expr, tilen, maxlen, message, x, modal, verbose, False );
	ActiveWin= aw;
	return(r);
}

int interactive_param_now_allwin( LocalWin *wi, char *expr, int tilen, int maxlen, char *message, double *x, int modal, int verbose )
{ LocalWin *aw= ActiveWin;
  int r;
	if( wi ){
		ActiveWin= wi;
	}
	r= interactive_param_now_xwin( wi->window, expr, tilen, maxlen, message, x, modal, verbose, True );
	ActiveWin= aw;
	return(r);
}

/* Given <string>, try to evaluate it as an ascanf expression, and check whether the value it returns
 \ is an ascanf pointer that points to a variable with a string (in the usage field). The take_usage
 \ argument will signal (if not NULL) whether or not the pointer itself was a stringpointer (e.g.
 \ `"this is a string") instead of just a pointer to a variable with a non-null usage string.
 \ When an ascanf string is found, a pointer to it is returned; otherwise, the string is returned
 \ without change. No duplication is done, so do not free this pointer!
 \ We don't allow the automatic creation of variables while doing this -- this also deactivates warnings
 \ about undefined variables that are bound to occur when trying to evaluate something that is not
 \ an ascanf expression.
 \ 20040512: we do allow stringvariable creation.
 */
char *ascanf_string( char *string, int *take_usage )
{ double x;
  extern int ascanf_AutoVarCreate, ascanf_AutoVarWouldCreate_msg;
  int aac= ascanf_AutoVarCreate, aawc= ascanf_AutoVarWouldCreate_msg;
	if( string ){
		ascanf_AutoVarCreate= False;
		ascanf_AutoVarWouldCreate_msg= False;
		if( new_param_now( string, &x, -1 ) ){
		  ascanf_Function *af= parse_ascanf_address( x, 0, "ascanf_string", 0, take_usage );
			if( af && af->usage ){
				string= af->usage;
			}
		}
		ascanf_AutoVarWouldCreate_msg= aawc;
		ascanf_AutoVarCreate= aac;
	}
	return( string );
}

int new_transform_x_process( LocalWin *new_info)
{ int len, n;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( new_info, "new *TRANSFORM_X*" );
	len= new_info->transform.x_len= strlen( cleanup( new_info->transform.x_process ) );
	hash= ascanf_hash( new_info->transform.x_process, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= 1;
	data[0]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &new_info->transform.C_x_process );
	new_info->transform.separator = ascanf_separator;
	fascanf( &n, new_info->transform.x_process, param_scratch, NULL, data, NULL,
		&new_info->transform.C_x_process
	);
	if( debugFlag ){
		fprintf( StdErr, "TRANSFORM_X:\n");
		Print_Form( StdErr, &new_info->transform.C_x_process, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		new_info->transform.x_len= 0;
		if( ascanf_arg_error ){
			fprintf( StdErr, "\tascanf_arg_error; TRANSFORM_X statement discarded (n=%d)\n", n );
			xfree( new_info->transform.x_process );
			new_info->transform.x_len= 0;
			Destroy_Form( &new_info->transform.C_x_process );
		}
		else if( len ){
			fprintf( StdErr, "\tTRANSFORM_X statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " *TRANSFORM_X*: ";
			add_comment( c);
			Add_Comment( new_info->transform.x_process, True );
		}
		prev_hash= hash;
		add_process_hist( new_info->transform.x_process );
	}
	TitleMessage( new_info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_transform_y_process( LocalWin *new_info)
{ int len, n;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( new_info, "new *TRANSFORM_Y*" );
	len= new_info->transform.y_len= strlen( cleanup(new_info->transform.y_process) );
	hash= ascanf_hash( new_info->transform.y_process, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= 1;
	data[0]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &new_info->transform.C_y_process );
	new_info->transform.separator = ascanf_separator;
	fascanf( &n, new_info->transform.y_process, param_scratch, NULL, data, NULL,
		&new_info->transform.C_y_process
	);
	if( debugFlag ){
		fprintf( StdErr, "TRANSFORM_Y:\n");
		Print_Form( StdErr, &new_info->transform.C_y_process, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		new_info->transform.y_len= 0;
		if( ascanf_arg_error ){
			fprintf( StdErr, "\tascanf_arg_error; TRANSFORM_Y statement discarded (n=%d)\n", n );
			xfree( new_info->transform.y_process );
			new_info->transform.y_len= 0;
			Destroy_Form( &new_info->transform.C_y_process );
		}
		else if( len ){
			fprintf( StdErr, "\tTRANSFORM_Y statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " *TRANSFORM_Y*: ";
			add_comment( c);
			Add_Comment( new_info->transform.y_process, True );
		}
		prev_hash= hash;
		add_process_hist( new_info->transform.y_process);
	}
	TitleMessage( new_info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_data_init( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DATA_INIT*" );
	len= theWin_Info->process.data_init_len= strlen( cleanup(theWin_Info->process.data_init) );
	hash= ascanf_hash( theWin_Info->process.data_init, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_data_init );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.data_init, param_scratch, NULL, data, column,
		&theWin_Info->process.C_data_init
	);
	if( debugFlag ){
		fprintf( StdErr, "New DATA_INIT:\n");
		Print_Form( StdErr, &theWin_Info->process.C_data_init, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.data_init_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DATA_INIT statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DATA_INIT*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.data_init, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.data_init );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_data_before( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DATA_BEFORE*" );
	len= theWin_Info->process.data_before_len= strlen( cleanup(theWin_Info->process.data_before) );
	hash= ascanf_hash( theWin_Info->process.data_before, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_data_before );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.data_before, param_scratch, NULL, data, column,
		&theWin_Info->process.C_data_before
	);
	if( debugFlag ){
		fprintf( StdErr, "New DATA_BEFORE:\n");
		Print_Form( StdErr, &theWin_Info->process.C_data_before, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.data_before_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DATA_BEFORE statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DATA_BEFORE*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.data_before, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.data_before );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_data_process( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DATA_PROCESS*" );
	len= theWin_Info->process.data_process_len= strlen( cleanup(theWin_Info->process.data_process) );
	hash= ascanf_hash( theWin_Info->process.data_process, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= ASCANF_DATA_COLUMNS;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_data_process );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.data_process, data, NULL, data, column,
		&theWin_Info->process.C_data_process
	);
	if( debugFlag ){
		fprintf( StdErr, "New DATA_PROCESS:\n");
		Print_Form( StdErr, &theWin_Info->process.C_data_process, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.data_process_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DATA_PROCESS statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DATA_PROCESS*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.data_process, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.data_process );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_data_after( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DATA_AFTER*" );
	len= theWin_Info->process.data_after_len= strlen( theWin_Info->process.data_after );
	hash= ascanf_hash( theWin_Info->process.data_after, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_data_after );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.data_after, param_scratch, NULL, data, column,
		&theWin_Info->process.C_data_after
	);
	if( debugFlag ){
		fprintf( StdErr, "New DATA_AFTER:\n");
		Print_Form( StdErr, &theWin_Info->process.C_data_after, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.data_after_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DATA_AFTER statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DATA_AFTER*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.data_after, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.data_after );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_data_finish( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DATA_FINISH*" );
	len= theWin_Info->process.data_finish_len= strlen( theWin_Info->process.data_finish );
	hash= ascanf_hash( theWin_Info->process.data_finish, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_data_finish );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.data_finish, param_scratch, NULL, data, column,
		&theWin_Info->process.C_data_finish
	);
	if( debugFlag ){
		fprintf( StdErr, "New DATA_FINISH:\n");
		Print_Form( StdErr, &theWin_Info->process.C_data_finish, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.data_finish_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DATA_FINISH statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DATA_FINISH*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.data_finish, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.data_finish );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_draw_before( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DRAW_BEFORE*" );
	len= theWin_Info->process.draw_before_len= strlen( cleanup(theWin_Info->process.draw_before) );
	hash= ascanf_hash( theWin_Info->process.draw_before, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_draw_before );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.draw_before, param_scratch, NULL, data, column,
		&theWin_Info->process.C_draw_before
	);
	if( debugFlag ){
		fprintf( StdErr, "New DRAW_BEFORE [%d elements]:\n", n);
		Print_Form( StdErr, &theWin_Info->process.C_draw_before, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.draw_before_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DRAW_BEFORE statement without effect (n=%d) \"%s\"\n",
				n, theWin_Info->process.draw_before
			);
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DRAW_BEFORE*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.draw_before, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.draw_before );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_draw_after( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DRAW_AFTER*" );
	len= theWin_Info->process.draw_after_len= strlen( cleanup(theWin_Info->process.draw_after) );
	hash= ascanf_hash( theWin_Info->process.draw_after, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_draw_after );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.draw_after, param_scratch, NULL, data, column,
		&theWin_Info->process.C_draw_after
	);
	if( debugFlag ){
		fprintf( StdErr, "New DRAW_AFTER [%d elements]:\n", n);
		Print_Form( StdErr, &theWin_Info->process.C_draw_after, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.draw_after_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DRAW_AFTER statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DRAW_AFTER*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.draw_after, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.draw_after );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_dump_before( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DUMP_BEFORE*" );
	len= theWin_Info->process.dump_before_len= strlen( cleanup(theWin_Info->process.dump_before) );
	hash= ascanf_hash( theWin_Info->process.dump_before, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_dump_before );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.dump_before, param_scratch, NULL, data, column,
		&theWin_Info->process.C_dump_before
	);
	if( debugFlag ){
		fprintf( StdErr, "New DUMP_BEFORE [%d elements]:\n", n);
		Print_Form( StdErr, &theWin_Info->process.C_dump_before, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.dump_before_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DUMP_BEFORE statement without effect (n=%d) \"%s\"\n",
				n, theWin_Info->process.dump_before
			);
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DUMP_BEFORE*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.dump_before, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.dump_before );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_dump_after( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *DUMP_AFTER*" );
	len= theWin_Info->process.dump_after_len= strlen( cleanup(theWin_Info->process.dump_after) );
	hash= ascanf_hash( theWin_Info->process.dump_after, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_dump_after );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.dump_after, param_scratch, NULL, data, column,
		&theWin_Info->process.C_dump_after
	);
	if( debugFlag ){
		fprintf( StdErr, "New DUMP_AFTER [%d elements]:\n", n);
		Print_Form( StdErr, &theWin_Info->process.C_dump_after, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.dump_after_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or DUMP_AFTER statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *DUMP_AFTER*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.dump_after, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.dump_after );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_enter_raw_after( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *ENTER_RAW_AFTER*" );
	len= theWin_Info->process.enter_raw_after_len= strlen( cleanup(theWin_Info->process.enter_raw_after) );
	hash= ascanf_hash( theWin_Info->process.enter_raw_after, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_enter_raw_after );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.enter_raw_after, param_scratch, NULL, data, column,
		&theWin_Info->process.C_enter_raw_after
	);
	if( debugFlag ){
		fprintf( StdErr, "New ENTER_RAW_AFTER [%d elements]:\n", n);
		Print_Form( StdErr, &theWin_Info->process.C_enter_raw_after, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.enter_raw_after_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or ENTER_RAW_AFTER statement without effect (n=%d) \"%s\"\n",
				n, theWin_Info->process.enter_raw_after
			);
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *ENTER_RAW_AFTER*: ";
			add_comment( c);
			Add_Comment( theWin_Info->process.enter_raw_after, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.enter_raw_after );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_leave_raw_after( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  static long prev_hash= 0;
  long hash;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *LEAVE_RAW_AFTER*" );
	len= theWin_Info->process.leave_raw_after_len= strlen( cleanup(theWin_Info->process.leave_raw_after) );
	hash= ascanf_hash( theWin_Info->process.leave_raw_after, NULL );
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &theWin_Info->process.C_leave_raw_after );
	theWin_Info->process.separator = ascanf_separator;
	fascanf( &n, theWin_Info->process.leave_raw_after, param_scratch, NULL, data, column,
		&theWin_Info->process.C_leave_raw_after
	);
	if( debugFlag ){
		fprintf( StdErr, "New LEAVE_RAW_AFTER [%d elements]:\n", n);
		Print_Form( StdErr, &theWin_Info->process.C_leave_raw_after, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		theWin_Info->process.leave_raw_after_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or LEAVE_RAW_AFTER statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		if( hash!= prev_hash ){
		  char c[]= " New *LEAVE_RAW_AFTER*: ";
			add_comment( c );
			Add_Comment( theWin_Info->process.leave_raw_after, True );
		}
		prev_hash= hash;
		add_process_hist( theWin_Info->process.leave_raw_after );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_set_process( LocalWin *theWin_Info, DataSet *this_set )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *SET_PROCESS*" );
	if( cleanup(this_set->process.set_process) ){
		len= this_set->process.set_process_len= strlen( this_set->process.set_process );
	}
	else{
		len= this_set->process.set_process_len= 0;
	}
	*ascanf_self_value= *ascanf_current_value= 0.0;
	  /* 20070611: n must be 4 to account for all 4 $DATA{} values... */
	n= 4;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &this_set->process.C_set_process );
	if( !len ){
		ascanf_propagate_events= ape;
		return(0);
	}
	this_set->process.separator = ascanf_separator;
	fascanf( &n, this_set->process.set_process, data, NULL, data, column,
		&this_set->process.C_set_process
	);
	{  char buf[256];
		sprintf( buf, "New *SET_PROCESS* <set %d>: ", this_set->set_nr );
		Add_Comment( buf, True );
		Add_Comment( this_set->process.set_process, True );
	}
	if( debugFlag ){
		fprintf( StdErr, "New *SET_PROCESS* <set %d>: ", this_set->set_nr );
		Print_Form( StdErr, &this_set->process.C_set_process, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		this_set->process.set_process_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or SET_PROCESS statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		add_process_hist( this_set->process.set_process );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

GenericProcess BoxFilter[BOX_FILTERS];

int new_process_BoxFilter_process( LocalWin *theWin_Info, int which )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  GenericProcess *process;
  char *msg;
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	switch( which ){
		default:
			fprintf( StdErr, "new_process_BoxFilter_process() called with invalid 'which' parameter %d\n", which );
			return(0);
			break;
		case BOX_FILTER_INIT:
			process= &BoxFilter[which];
			msg= "new *BOX_FILTER_INIT*: ";
			strcpy( process->command, "*BOX_FILTER_INIT*" );
			break;
		case BOX_FILTER_PROCESS:
			process= &BoxFilter[which];
			msg= "new *BOX_FILTER*: ";
			strcpy( process->command, "*BOX_FILTER*" );
			break;
		case BOX_FILTER_AFTER:
			process= &BoxFilter[which];
			msg= "new *BOX_FILTER_AFTER*: ";
			strcpy( process->command, "*BOX_FILTER_AFTER*" );
			break;
		case BOX_FILTER_FINISH:
			process= &BoxFilter[which];
			msg= "new *BOX_FILTER_FINISH*: ";
			strcpy( process->command, "*BOX_FILTER_FINISH*" );
			break;
		case BOX_FILTER_CLEANUP:
			process= &BoxFilter[which];
			msg= "new *BOX_FILTER_CLEANUP*: ";
			strcpy( process->command, "*BOX_FILTER_CLEANUP*" );
			break;
	}
	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, msg );
	if( cleanup(process->process) ){
		len= process->process_len= strlen( process->process );
	}
	else{
		len= process->process_len= 0;
	}
	process->separator = ascanf_separator;
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= param_scratch_len;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &process->C_process );
	if( !len ){
		ascanf_propagate_events= ape;
		return(0);
	}
	TBARprogress_header= msg;
	process->separator = ascanf_separator;
	fascanf( &n, process->process, data, NULL, data, column,
		&process->C_process
	);
	{  char buf[256];
		sprintf( buf, msg );
		Add_Comment( buf, True );
		Add_Comment( process->process, True );
	}
	if( debugFlag ){
		fprintf( StdErr, msg );
		Print_Form( StdErr, &process->C_process, 0, True, NULL, NULL, "\n", True );
		if( which== 3 && strstr( process->process, "$Counter" ) ){
			fprintf( StdErr, "##\t\tNB: *BOX_FILTER_AFTER* replaces original *BOX_FILTER_FINISH* statement!\n" );
		}
	}
	if( ascanf_arg_error || n== 0 ){
		process->process_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or BOX_FILTER statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		add_process_hist( process->process );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

int new_process_Cross_fromwin_process( LocalWin *theWin_Info )
{ int n, len;
  extern double *ascanf_self_value, *ascanf_current_value;
  extern int ascanf_arg_error, reset_ascanf_currentself_value;
  Cursor_Cross *s= &(theWin_Info->curs_cross);
  extern int ascanf_propagate_events;
  int ape= ascanf_propagate_events;

	ascanf_propagate_events= False;
	clean_param_scratch();
	TitleMessage( theWin_Info, "new *CROSS_FROMWIN_PROCESS*" );
	if( cleanup(s->fromwin_process.process) ){
		len= s->fromwin_process.process_len= strlen( s->fromwin_process.process );
	}
	else{
		len= s->fromwin_process.process_len= 0;
	}
	*ascanf_self_value= *ascanf_current_value= 0.0;
	n= 3;
	data[0]= data[1]= data[2]= data[3]= 0.0;
	reset_ascanf_currentself_value= 0;
	Destroy_Form( &s->fromwin_process.C_process );
	if( !len ){
		ascanf_propagate_events= ape;
		return(0);
	}
	s->fromwin_process.separator = ascanf_separator;
	TBARprogress_header= s->fromwin_process.command;
	s->fromwin_process.separator = ascanf_separator;
	fascanf( &n, s->fromwin_process.process, data, NULL, data, column,
		&s->fromwin_process.C_process
	);
	{  char buf[256];
		sprintf( buf, "New *CROSS_FROMWIN_PROCESS*: " );
		Add_Comment( buf, True );
		Add_Comment( s->fromwin_process.process, True );
	}
	if( debugFlag ){
		fprintf( StdErr, "New *CROSS_FROMWIN_PROCESS*:" );
		Print_Form( StdErr, &s->fromwin_process.C_process, 0, True, NULL, NULL, "\n", True );
	}
	if( ascanf_arg_error || n== 0 ){
		s->fromwin_process.process_len= 0;
		if( ascanf_arg_error || len ){
			fprintf( StdErr, "\tascanf_arg_error or CROSS_FROMWIN_PROCESS statement without effect (n=%d)\n", n );
			fflush( StdErr );
		}
	}
	else{
		add_process_hist( s->fromwin_process.process );
	}
	TitleMessage( theWin_Info, NULL );
	ascanf_propagate_events= ape;
	return(1);
}

char settings_text[16]="Settings";

/* 20010720: I made this callback mechanism interface to _Handle_An_Event() a bit tidier.
 \ Specifically, the LocalWin datapointer is now passed correctly, i.e. via an xtb_registry_info
 \ pointer of the xtb_User type. Note that this is an xtb-registered event handler: the first argument is thus
 \ the current event, and NOT a Window id as in the callback routines the user normally sees (when he/she
 \ interacted with a button or the like).
 */
/* xtb_hret xtb_LocalWin_h( XEvent *evt, LocalWin *wi )	*/
xtb_hret xtb_LocalWin_h( XEvent *evt, xtb_registry_info *info )
{ static char active= 0;
  LocalWin *wi;
	if( info && info->type== xtb_User && info->func== (void*) xtb_LocalWin_h ){
		wi= info->val;
	}
	if( wi && !active && wi->delete_it!= -1 ){
	  /* we must prevent getting caught in a recursiveness here!
	   \ Silently ignore events for the 2 "special" windows.
	   */
		if( wi!= &StubWindow && wi!= InitWindow ){
			if( wi->delete_it!= -1 ){
				active= 1;
				_Handle_An_Event( evt, wi->event_level, 0, "xtb_LocalWin_h" );
				active= 0;
			}
			else{
				fprintf( StdErr, "xtb_LocalWin_h(): event for deleted window %d:%d:%d ignored\n",
					wi->parent_number, wi->pwindow_number, wi->window_number
				);
			}
		}
		return( XTB_HANDLED );
	}
	else{
		return( XTB_NOTDEF );
	}
}

LocalWin *ConsWindow( LocalWin *wi)
{  LocalWindows *new;  
   xtb_registry_info *entry= NULL;
	if( (new= (LocalWindows*)calloc( 1, sizeof(LocalWindows))) ){
		new->wi= wi;
		wi->WindowList_Entry= new;
		new->above= new->under= None;
		if( (new->frame= (xtb_frame*) calloc(1, sizeof(xtb_frame))) &&
			(entry= (xtb_registry_info*) calloc( 1, sizeof(xtb_registry_info)))
		){
			entry->frame= new->frame;
			entry->func= (void*) xtb_LocalWin_h;
			entry->val= wi;
			entry->type= xtb_User;
			new->frame->info= (xtb_data) entry;
			if( wi->title_template ){
				new->frame->description= strdup(wi->title_template);
			}
/* 			xtb_register( new->frame, wi->window, (void*) xtb_LocalWin_h, wi );	*/
			xtb_register( new->frame, wi->window, (void*) xtb_LocalWin_h, entry );
			if( wi->XDBE_buffer ){
				xtb_register( new->frame, wi->XDBE_buffer, (void*) xtb_LocalWin_h, entry );
			}
		}
		if( WindowList ){
			new->next= WindowList;
			WindowList->prev= new;
			new->prev= NULL;
			WindowList= new;
		}
		else{
			WindowList= new;
			WindowListTail= new;
			new->next= NULL;
			new->prev= NULL;
		}
	}
	else{
	  /* This is really an error, but we continue anyway.	*/
		fprintf( StdErr, "Internal error: can't add new window to list (%s)\n", serror());
	}
	return( wi );
}

LocalWin *RemoveWindow( LocalWin *wi )
{  LocalWindows *l= WindowList, *k= NULL;
	while( l ){
		if( l->wi== wi ){
		  LocalWindows *m= l;
			if( k ){
			  /* Not the first, so point the previous at the next	*/
				if( (k->next= l->next) ){
					k->next->prev= k;
				}
			}
			else{
			  /* The first, so point WindowList to the next	*/
				if( (WindowList= l->next) ){
					WindowList->prev= NULL;
					if( WindowList->next ){
						WindowList->next->prev= WindowList;
					}
				}
			}
			  /* Advance l before freeing this item. k is *not*
			   \ advanced, since maybe the next item also qualifies
			   \ for removal.
			   */
			l= l->next;
			if( m->frame ){
			  xtb_registry_info *info= NULL;
				  /* 20010720: it is important that wi->window is still valid! */
				if( wi->XDBE_buffer ){
					xtb_unregister( wi->XDBE_buffer, &info );
				}
				xtb_unregister( wi->window, &info );
				if( info ){
					info->func= NULL;
					info->type= -1;
					xfree( info );
				}
				xfree( m->frame->description );
				xfree( m->frame );
			}
			xfree( m );
			wi->WindowList_Entry= NULL;
		}
		else{
			k= l;
			l= l->next;
		}
	}
	WindowListTail= k;
	return( wi);
}

Cursor theCursor = (Cursor) 0, noCursor= (Cursor)0;
int ButtonContrast= 65535/3;
Boolean no_buttons= False;
Boolean X_psMarkers= False;

char *YAv_SortTypes[]= { "Xsrt", "Isrt", "XIsr", "IXsr", "Nsrt"},
	*YAv_Sort_Desc[]= { "Sort X values\n", "Sort pointnumber of first occurance of X value\n",
		"Sort X values, with pointnumbers as secundary key\n",
		"Sort pointnumbers, with X values as secundary key\n",
		"Don't sort\n"
	};

int N_YAv_SortTypes= 5;

static xtb_hret YAv_SortFun( Window win, int old, int new, xtb_data info)
{ LocalWin *wi;
  Window the_win = (Window) info;
    if (!XFindContext(disp, the_win, win_context, (caddr_t *) &wi)) {
		xtb_br_get( wi->YAv_Sort_frame.win);
	}
	return( XTB_HANDLED);
}

short *_drawingOrder= NULL;
int drawingOrder_set= 0;

int realloc_WinDiscard( LocalWin *wi, int n )
{ int idx;
	if( !(wi->discardpoint= (signed char**) XGrealloc( wi->discardpoint, n* sizeof(char*))) ){
		fprintf( StdErr, "realloc_WinDiscard(): can't get discardpoint array (%s)\n", serror() );
		return( 0 );
	}
	allocerr= 0;
	for( idx= 0; idx< setNumber; idx++ ){
	  DataSet *this_set= &AllSets[idx];
		if( wi->discardpoint[idx] ){
			wi->discardpoint[idx]= (signed char*) XGrealloc( wi->discardpoint[idx],
				(this_set->allocSize+ 2)* sizeof(char)
			);
		}
	}
	for( ; idx< n; idx++ ){
		wi->discardpoint[idx]= NULL;
	}
	return(1);
}

int realloc_LocalWin_data( LocalWin *new_info, int n)
{ int idx;
	if( !(new_info->draw_set= (short*) XGrealloc( new_info->draw_set, n* sizeof(short))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get draw_set array (%s)\n", serror() );
		return( 0 );
	}
	if( !(_drawingOrder= (short*) XGrealloc( _drawingOrder, n* sizeof(short))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get drawingOrder array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->mark_set= (short*) XGrealloc( new_info->mark_set, n* sizeof(short))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get mark_set array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->numVisible= (int*) XGrealloc( new_info->numVisible, n* sizeof(int))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get numVisible array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->group= (short*) XGrealloc( new_info->group, n* sizeof(short))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get group array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->fileNumber= (short*) XGrealloc( new_info->fileNumber, n* sizeof(short))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get fileNumber array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->new_file= (short*) XGrealloc( new_info->new_file, n* sizeof(short))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get new_file array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->plot_only_set= (short*) XGrealloc( new_info->plot_only_set, n* sizeof(short))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get plot_only_set array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->legend_line= (LegendLine*) XGrealloc( new_info->legend_line, n* sizeof(LegendLine))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get legend_line array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->xcol= (int*) XGrealloc( new_info->xcol, n* sizeof(int))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get xcol array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->ycol= (int*) XGrealloc( new_info->ycol, n* sizeof(int))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get ycol array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->ecol= (int*) XGrealloc( new_info->ecol, n* sizeof(int))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get ecol array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->lcol= (int*) XGrealloc( new_info->lcol, n* sizeof(int))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get lcol array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->error_type= (int*) XGrealloc( new_info->error_type, n* sizeof(int))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get error_type array (%s)\n", serror() );
		return( 0 );
	}

	if( !(new_info->pointVisible= (signed char**) XGrealloc( new_info->pointVisible, n* sizeof(signed char*))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get pointVisible array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->curve_len= (double**) XGrealloc( new_info->curve_len, n* sizeof(double*))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get curve_len array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->error_len= (double**) XGrealloc( new_info->error_len, n* sizeof(double*))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get error_len array (%s)\n", serror() );
		return( 0 );
	}
#ifdef TR_CURVE_LEN
	if( !(new_info->tr_curve_len= (double**) XGrealloc( new_info->tr_curve_len, n* sizeof(double*))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get tr_curve_len array (%s)\n", serror() );
		return( 0 );
	}
#endif
	allocerr= 0;
	for( idx= 0; idx< setNumber; idx++ ){
	  DataSet *this_set= &AllSets[idx];
		new_info->pointVisible[idx]= (signed char*) XGrealloc( new_info->pointVisible[idx],
			(this_set->allocSize+ 2)* sizeof(signed char)
		);
		new_info->curve_len[idx]= (double*) XGrealloc( new_info->curve_len[idx],
			(this_set->allocSize+ 2)* sizeof(double)
		);
		new_info->error_len[idx]= (double*) XGrealloc( new_info->error_len[idx],
			(this_set->allocSize+ 2)* sizeof(double)
		);
#ifdef TR_CURVE_LEN
		new_info->tr_curve_len[idx]= (double*) XGrealloc( new_info->tr_curve_len[idx],
			(this_set->allocSize+ 2)* sizeof(double)
		);
#endif
	}
	for( ; idx< n; idx++ ){
		new_info->pointVisible[idx]= NULL;
		new_info->curve_len[idx]= NULL;
		new_info->error_len[idx]= NULL;
#ifdef TR_CURVE_LEN
		new_info->tr_curve_len[idx]= NULL;
#endif
	}
	if( allocerr ){
		fprintf( StdErr, "realloc_LocalWin_data(): %d allocation errors alloc'ing curvelength arrays (%s)\n", allocerr, serror() );
		return( 0 );
	}

	if( new_info->discardpoint ){
		realloc_WinDiscard( new_info, n );
	}
	else if( !(new_info->discardpoint= (signed char**) XGrealloc( new_info->discardpoint, n* sizeof(char*))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get discardpoint array (%s)\n", serror() );
		return( 0 );
	}

	if( !(new_info->set_X= (SimpleStats*) XGrealloc( new_info->set_X, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_X array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_Y= (SimpleStats*) XGrealloc( new_info->set_Y, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_Y array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_E= (SimpleStats*) XGrealloc( new_info->set_E, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_E array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_V= (SimpleStats*) XGrealloc( new_info->set_V, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_V array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_O= (SimpleAngleStats*) XGrealloc( new_info->set_O, n* sizeof(SimpleAngleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_O array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_tr_X= (SimpleStats*) XGrealloc( new_info->set_tr_X, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_tr_X array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_tr_Y= (SimpleStats*) XGrealloc( new_info->set_tr_Y, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_tr_Y array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_tr_E= (SimpleStats*) XGrealloc( new_info->set_tr_E, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_tr_E array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_tr_V= (SimpleStats*) XGrealloc( new_info->set_tr_V, n* sizeof(SimpleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_tr_V array (%s)\n", serror() );
		return( 0 );
	}
	if( !(new_info->set_tr_O= (SimpleAngleStats*) XGrealloc( new_info->set_tr_O, n* sizeof(SimpleAngleStats))) ){
		fprintf( StdErr, "realloc_LocalWin_data(): can't get set_tr_O array (%s)\n", serror() );
		return( 0 );
	}

	return(1);
}

extern int reverseFlag;

  /* Warning: one such static variable per module that can perform "smart" xtb_init calling	*/
static int xtb_UseColours= False;

extern int CursorCross;

void SelectXInput( LocalWin *wi )
{
	if( CursorCross ){
		XDefineCursor(disp, wi->window, noCursor);
		XSelectInput(disp, wi->window,
				 VisibilityChangeMask|ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask|
				 PointerMotionMask|PointerMotionHintMask|ButtonMotionMask
		);
	}
	else{
		XDefineCursor(disp, wi->window, theCursor);
		XSelectInput(disp, wi->window,
				 VisibilityChangeMask|ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask|SubstructureRedirectMask
		);
	}
}

extern Atom wm_delete_window;
int BackingStore= False;
XdbeSwapAction XG_XDBE_SwapAction= XdbeBackground;

#define __DLINE__	(double)__LINE__

Window NewWindow( char *progname, LocalWin **New_Info, double _lowX, double _lowY, double _lowpX, double _lowpY,
					double _hinY, double _upX, double _upY, double asp,
					LocalWin *parent, double xscale, double yscale, double dyscale, int add_padding
)
/*
 * Creates and maps a new window.  This includes allocating its
 * local structure and associating it with the XId for the window.
 * The aspect ratio is specified as the ratio of width over height.
 \ The boundary information stored in new_info->loX &c contains the
 \ scaling factors. This should be removed to enable scale changes
 \ in the running program.
 \ 940906 scale changes have been added using wi->[XY]scale
 \ and the global variables Xscale2 and Yscale2.
 \ To correctly store boundaries, [XY]scale2 are set to 1.0 in
 \ this function. DrawWindow() will reset them to the correct values
 \ for the window to be drawn.
 */
{ Window new_window;
  XSizeHints *sizehints, WSH;
  XSetWindowAttributes wattr;
  XWMHints *wmhints, WMH;
  XClassHint *classhints;
  int width, height;
  unsigned long wamask;
  char defSpec[120];
  static double pad;
  static double lowX, lowY;
  static double lowpX, lowpY, highnY;
  static double upX, upY;
  static int win_number= -1;
  XEvent evt;
  LocalWin *new_info;
  int pF= polarFlag, aYF= absYFlag;
  extern int User_Coordinates;
  extern int IconicStart;
  Boolean init_lwi= False;

	if( !X11_c_KeyCode ){
		X11_c_KeyCode= XKeysymToKeycode( disp, XK_c );
	}

	if( New_Info && *New_Info== &StubWindow ){
		new_info= *New_Info;
		init_lwi= True;
	}
	else if( !(new_info = (LocalWin *) calloc( 1, sizeof(LocalWin))) ){
		fprintf( StdErr, "NewWindow(): can't get info buffer (%s)\n", serror() );
		return( 0 );
	}

	Elapsed_Since(&new_info->draw_timer, True);

	if( !(new_info->process.data_init= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get data_init buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.data_init_allen= MAXBUFSIZE;
	if( !(new_info->process.data_before= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get data_before buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.data_before_allen= MAXBUFSIZE;
	if( !(new_info->process.data_process= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get data_process buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.data_process_allen= MAXBUFSIZE;
	if( !(new_info->process.data_after= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get data_after buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.data_finish_allen= MAXBUFSIZE;
	if( !(new_info->process.data_finish= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get data_finish buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.data_finish_allen= MAXBUFSIZE;
	if( !(new_info->process.draw_before= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get draw_before buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.draw_before_allen= MAXBUFSIZE;
	if( !(new_info->process.draw_after= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get draw_after buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.draw_after_allen= MAXBUFSIZE;
	if( !(new_info->process.dump_before= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get dump_before buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.dump_before_allen= MAXBUFSIZE;
	if( !(new_info->process.dump_after= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get dump_after buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.dump_after_allen= MAXBUFSIZE;
	if( !(new_info->process.enter_raw_after= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get enter_raw_after buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.enter_raw_after_allen= MAXBUFSIZE;
	if( !(new_info->process.leave_raw_after= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get leave_raw_after buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		xfree( new_info );
		return( 0 );
	}
	new_info->process.leave_raw_after_allen= MAXBUFSIZE;

	if( !(new_info->transform.x_process= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get transform.x_process buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		DelWindowTransform( &new_info->transform );
		xfree( new_info );
		return( 0 );
	}
	new_info->transform.x_allen= MAXBUFSIZE;
	if( !(new_info->transform.y_process= (char*) calloc( MAXBUFSIZE, sizeof(char))) ){
		fprintf( StdErr, "NewWindow(): can't get transform.y_process buffer (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		DelWindowTransform( &new_info->transform );
		xfree( new_info );
		return( 0 );
	}
	new_info->transform.y_allen= MAXBUFSIZE;

	if( !(new_info->hard_devices= (Hard_Devices*) calloc( hard_count, sizeof(Hard_Devices))) ){
		fprintf( StdErr, "NewWindow(): can't get hard_devices array (%s)\n", serror() );
		DelWindowProcess( &new_info->process );
		DelWindowTransform( &new_info->transform );
		xfree( new_info );
		return( 0 );
	}
	new_info->current_device= -1;

	if( !realloc_LocalWin_data( new_info, MAXSETS) ){
		fprintf( StdErr, "NewWindow(): can't get window-specific data structures (%s)\n", serror() );
		xfree( new_info->hard_devices );
		DelWindowProcess( &new_info->process );
		DelWindowTransform( &new_info->transform );
		xfree( new_info );
		return(0);
	}
	else{
	  int i;
		for( i= 0; i< MAXSETS; i++ ){
			new_info->draw_set[i]= 1;
			new_info->fileNumber[i]= -1;
			new_info->group[i]= -1;
			new_info->xcol[i]= xcol;
			new_info->ycol[i]= ycol;
			new_info->ecol[i]= ecol;
			new_info->lcol[i]= lcol;
			new_info->error_type[i]= -1;
		}
	}

	*New_Info= new_info;
	CopyFlags( new_info, parent );
	CopyFlags( NULL, parent );

	absYFlag= 0;

	lowX= _lowX;
	lowY= _lowY;
	lowpX=_lowpX;
	lowpY= _lowpY;
	highnY= _hinY;
	upX= _upX;
	upY= _upY;

    if( upX > lowX || polarFlag ){
		new_info->loX = lowX;
		new_info->hiX = upX;
    } else {
		new_info->loX = llx;
		new_info->hiX = urx;
    }
    if( upY > lowY) {
		new_info->loY = lowY;
		new_info->hiY = upY;
    } else {
		new_info->loY = lly;
		new_info->hiY = ury;
    }
	new_info->lopX= lowpX;
	new_info->lopY= lowpY;
	new_info->hinY= highnY;
	new_info->axis_stuff.XLabelLength= XLabelLength;
	new_info->axis_stuff.YLabelLength= YLabelLength;

	new_info->axis_stuff.rawX.axis= new_info->axis_stuff.X.axis= X_axis;
	new_info->axis_stuff.rawY.axis= new_info->axis_stuff.Y.axis= Y_axis;
	new_info->axis_stuff.I.axis= I_axis;
	new_info->axis_stuff.rawX.N= new_info->axis_stuff.X.N= 0;
	new_info->axis_stuff.rawY.N= new_info->axis_stuff.Y.N= 0;
	new_info->axis_stuff.I.N= 0;

	new_info->datawin.apply= 0;
	set_NaN(new_info->datawin.llX);
	new_info->datawin.llY= new_info->datawin.llX;
	new_info->datawin.urX= new_info->datawin.urY= new_info->datawin.llX;

	  /* bounds are currently untransformed, so copy them
	   \ in the pure_bounds structure
	   */
	new_info->win_geo.pure_bounds= new_info->win_geo.bounds;
	new_info->win_geo.pure_bounds.pure= 1;
	new_info->win_geo.user_coordinates= User_Coordinates;
 	new_info->win_geo.aspect_base_bounds= new_info->win_geo.bounds;
// 	set_NaN(new_info->win_geo.aspect_base_bounds._loX);
// 	set_NaN(new_info->win_geo.aspect_base_bounds._loY);
// 	set_NaN(new_info->win_geo.aspect_base_bounds._hiX);
// 	set_NaN(new_info->win_geo.aspect_base_bounds._hiY);

	lowX= new_info->loX;
	lowY= new_info->loY;
	upX= new_info->hiX;
	upY= new_info->hiY;

	new_info->Xscale= 1.0;
	new_info->_Xscale= 1.0;
	new_info->Yscale= 1.0;

	new_info->legend_always_visible= 0;

	  /* Fill with NaNs: */
	Reset_AFTFit_History( new_info );

	if( init_lwi ){
		return(1);
	}

    /* Increase the padding for aesthetics */
    if (new_info->hiX - new_info->loX == 0.0) {
	 int okL= 1, okH= 1;
		do_transform( new_info, "Xpad", __DLINE__, "NewWindow(low)", &okL, NULL, &lowX, NULL, NULL, &lowY, NULL, NULL, NULL, NULL, 0,
			-1, xscale, yscale, dyscale, 1, 0, False
		);
		do_transform( new_info, "Xpad", __DLINE__, "NewWindow(high)", &okH, NULL, &upX, NULL, NULL, &upY, NULL, NULL, NULL, NULL, 0,
			-1, xscale, yscale, dyscale, 1, 0, False
		);
		if( okL && okH ){
			pad = MAX(0.5, fabs(upX/2.0));
			upX+= pad;
			lowX-= pad;
			upX= new_info->hiX = Reform_X( new_info, upX, upY);
			lowX= new_info->loX = Reform_X( new_info, lowX, lowY);
		}
    }
    if (new_info->hiY - new_info->loY == 0) {
	 int okL= 1, okH= 1;
		do_transform( new_info, "Ypad", __DLINE__, "NewWindow(low)", &okL, NULL, &lowX, NULL, NULL, &lowY, NULL, NULL, NULL, NULL, 0,
			-1, xscale, yscale, dyscale, 1, 0, False
		);
		do_transform( new_info, "Ypad", __DLINE__, "NewWindow(high)", &okH, NULL, &upX, NULL, NULL, &upY, NULL, NULL, NULL, NULL, 0,
			-1, xscale, yscale, dyscale, 1, 0, False
		);
		if( okL && okH ){
			pad = MAX(0.5, fabs(upY/2.0));
			upY+= pad;
			lowY-= pad;
			upY= new_info->hiY = Reform_Y( new_info, upY, upX);
			lowY= new_info->loY = Reform_Y( new_info, lowY, lowX );
		}
    }

/* 	polarFlag= 0;	*/
		/* Add 10% padding to *real* (transformed) bounding box (div by 20 yields 5%) */
	if( add_padding ){
	  int okL= 1, okH= 1;
		new_info->win_geo.padding= 20.0;
		do_transform( new_info, "low", __DLINE__, "NewWindow(low)", &okL, NULL, &lowX, NULL, NULL, &lowY, NULL, NULL, NULL, NULL, 0,
			-1, xscale, yscale, dyscale, 1, 0, False
		);
		do_transform( new_info, "high", __DLINE__, "NewWindow(high)", &okH, NULL, &upX, NULL, NULL, &upY, NULL, NULL, NULL, NULL, 0,
			-1, xscale, yscale, dyscale, 1, 0, False
		);
		if( okL && okH ){
			if( !pF  && !use_lx ){
				pad = (upX - lowX) / new_info->win_geo.padding;
				lowX-= pad;
				upX+= pad;
				new_info->loX= Reform_X( new_info, lowX, lowY );
				new_info->hiX= Reform_X( new_info, upX, upY );
			}
			if( !use_ly ){
				pad = (upY - lowY) / new_info->win_geo.padding;
				lowY-= pad;
				upY+= pad;
				new_info->loY= Reform_Y( new_info, lowY, lowX ); 
				new_info->hiY= Reform_Y( new_info, upY, upX );
			}
		}
	}
	else{
		new_info->win_geo.padding= 0.0;
	}
	polarFlag= pF;
	absYFlag= aYF;

	new_info->Xscale= Xscale2;
	new_info->_Xscale= Xscale2;
	new_info->Yscale= Yscale2;

    /* Aspect ratio computation */
    if (asp < 1.0) {
		height = NORMSIZEY;
		width = ((int) (((double) height) * asp));
    } else {
		width = NORMSIZEX;
		height = ((int) (((double) width) / asp));
    }
    height = MAX(MINDIM, height);
    width = MAX(MINDIM, width);
    (void) sprintf(defSpec, "%dx%d+0+0", width, height);

    wamask = ux11_fill_wattr(&wattr, CWBackPixel, bgPixel,
			     CWBorderPixel, bdrPixel, CWColormap, cmap, UX11_END);
	if( BackingStore ){
		if( debugFlag ){
			fprintf( StdErr, "Attempting to obtain backingstore+save_under support\n" );
		}
		wattr.backing_store= Always;
		wattr.save_under= True;
	}
	else{
		if( debugFlag ){
			fprintf( StdErr, "Attempting to disable backingstore+save_under support\n" );
		}
		wattr.backing_store= NotUseful;
		wattr.save_under= False;
	}
	wamask|= CWBackingStore|CWSaveUnder;

	if( !(sizehints= XAllocSizeHints()) ){
		sizehints= &WSH;
	}

    sizehints->flags = PPosition|PSize|PMinSize|PBaseSize|PMaxSize;
    sizehints->x = sizehints->y = 0;
    sizehints->min_width= 10;
	sizehints->base_width= sizehints->width = width;
	sizehints->min_height= 10;
    sizehints->base_height= sizehints->height = height;
	sizehints->max_width= sizehints->max_height= 65535;

	if( geoSpec ){
		xtb_ParseGeometry( geoSpec, sizehints, 0, False );
	}
	if( use_RootWindow && win_number== 0 ){
		new_window= RootWindow( disp, screen );
	}
	else{
		new_window = XCreateWindow(disp, RootWindow(disp, screen),
					   sizehints->x, sizehints->y,
					   (unsigned int) sizehints->width,
					   (unsigned int) sizehints->height,
					   (unsigned int) bdrSize,
					   depth, InputOutput, vis,
					   wamask, &wattr
		);
	}

	XSetWMProtocols( disp, new_window, &wm_delete_window, 1 );

    if( new_window) {
	 char *c= XGstrdup(progname);
	 int window_number, pwindow_number, parent_number;

		new_info->pid= getpid();
		new_info->visual= vis;
		new_info->cmap= cmap;

		if( parent ){
			parent_number= parent->parent_number+ 1;
			pwindow_number= parent->window_number;
			window_number= parent->childs;
			parent->childs+=1;
		}
		else{
			parent_number= 0;
			pwindow_number= 0;
			window_number= win_number;
		}
		win_number+= 1;

		new_info->parent_number= parent_number;
		new_info->pwindow_number= pwindow_number;
		new_info->window_number= window_number;
		new_info->childs= 0;
		new_info->title_template= XGstrdup(progname);

		new_info->window= new_window;
		new_info->draw_count= 0;

		if( !(wmhints= XAllocWMHints()) ){
			wmhints= &WMH;
			new_info->wmhints= NULL;
		}
		else{
			new_info->wmhints= wmhints;
		}
		wmhints->flags = InputHint | StateHint;
		wmhints->input = True;
		if( IconicStart ){
			wmhints->initial_state = IconicState;
			new_info->mapped= 0;
		}
		else{
			wmhints->initial_state = NormalState;
			new_info->mapped= 0;
		}

		IconicStart= 0;
		if( new_window!= RootWindow( disp, screen ) ){
		  static XTextProperty wName, iName;
/* 			XSetWMHints(disp, new_window, wmhints);	*/
			if( XStringListToTextProperty( &new_info->title_template, 1, &wName)== 0 ){
			}
			if( XStringListToTextProperty( &new_info->title_template, 1, &iName)== 0 ){
			}
			if( (classhints= XAllocClassHint()) ){
			  char *c;
			  static char *class= NULL;
				if( (c= rindex(Argv[0], '/')) ){
					classhints->res_name= ( &c[1] );
				}
				else{
					classhints->res_name= ( Argv[0] );
				}
				if( !class ){
					class= strdup(Prog_Name);
					class[0]= toupper(class[0]);
					class[1]= toupper(class[1]);
				}
				classhints->res_class= class;
			}
			XSetWMProperties( disp, new_window, &wName, &iName,
				Argv, Argc, sizehints, wmhints, classhints );
			if( classhints ){
				XFree( classhints );
			}
		}

		{ char **argv= NULL;

			if( new_window!= RootWindow( disp, screen ) ){
				XSetStandardProperties( disp, new_window,
					c, c, None, argv, 0, sizehints
				);
				XSetNormalHints(disp, new_window, sizehints);
			}
			XGetNormalHints(disp, new_window, sizehints);
		}

		xfree(c);
		c= NULL;

		new_info->dev_info.area_w= sizehints->width;
		new_info->dev_info.area_h= sizehints->height;

		if (!win_context) {
			win_context = XUniqueContext();
		}
		if( !frame_context ){
			frame_context= XUniqueContext();
		}

		XSaveContext(disp, new_window, win_context, (caddr_t) new_info);

		{ char close_text[]= "Close",
		  hc_text[]= "HardCopy",
		  info_text[]= "Info",
		  label_text[]= "Label",
		  sht_text[]= "Ssht";
		  XColor zero, grid, norm, bg;
			zero.pixel= zeroPixel;
			XQueryColor( disp, cmap, &zero );
			grid.pixel= gridPixel;
			XQueryColor( disp, cmap, &grid );
			norm.pixel= normPixel;
			XQueryColor( disp, cmap, &norm );
			bg.pixel= bgPixel;
			XQueryColor( disp, cmap, &bg );
			  /* Make buttons */
			if( cursorFont.font ){
				  /* If there's enough luminance-contrast between those 2 colours, use 'm for
				   \ the buttons.
				   */
				if( xtb_UseColours && fabs( xtb_PsychoMetric_Gray(&zero) - xtb_PsychoMetric_Gray(&grid) )>= ButtonContrast ){
					if( debugFlag ){
						fprintf( StdErr, "NewWindow(): lum(zeroPixel:%u,%u,%u)==%g lum(gridPixel:%u,%u,%u)==%g\n",
							zero.red, zero.green, zero.blue,
							xtb_PsychoMetric_Gray(&zero),
							grid.red, grid.green, grid.blue,
							xtb_PsychoMetric_Gray(&grid)
						);
					}
					xtb_init(disp, screen, zeroPixel, gridPixel, cursorFont.font, NULL, False);
				}
				else if( xtb_UseColours && fabs( xtb_PsychoMetric_Gray(&norm) - xtb_PsychoMetric_Gray(&bg) )>= ButtonContrast ){
					xtb_init(disp, screen, normPixel, bgPixel, cursorFont.font, NULL, False);
				}
				else{
					xtb_init(disp, screen, black_pixel, white_pixel, cursorFont.font, NULL, False);
				}
				sprintf( close_text, "%c", XC_pirate );
				sprintf( hc_text, "%c", XC_spraycan );
				sprintf( settings_text, "%c", XC_watch );
				sprintf( info_text, "%c", XC_spider );
				sprintf( label_text, "%c", XC_pencil );
				sprintf( sht_text, "%c", XC_iron_cross );

			}

			xtb_bt_new2(new_window, close_text, XTB_CENTERED, del_func,
				   (xtb_data) new_window, &new_info->cl_frame);
			new_info->close = new_info->cl_frame.win;
			XSaveContext( disp, new_info->close, frame_context, (caddr_t) &new_info->cl_frame );

			xtb_bt_new2(new_window, hc_text, XTB_CENTERED, hcpy_func,
				   (xtb_data) new_window, &new_info->hd_frame
			);
			new_info->hardcopy = new_info->hd_frame.win;
			XSaveContext( disp, new_info->hardcopy, frame_context, (caddr_t) &new_info->hd_frame );

			xtb_bt_new2(new_window, settings_text, XTB_CENTERED, settings_func,
				   (xtb_data) new_window, &new_info->settings_frame
			);
			new_info->settings= new_info->settings_frame.win;
			XSaveContext( disp, new_info->settings, frame_context, (caddr_t) &new_info->settings_frame );
			if( cursorFont.font ){
			  /* XC_watch (shown while drawing) is wider than XC_question_arrow; therefore the window
			   \ is created with XC_watch, and then XC_q._a. is shown.
			   */
				settings_text[0]= XC_question_arrow;
				xtb_bt_set_text( new_info->settings, xtb_bt_get(new_info->settings, NULL), settings_text, (xtb_data) 0);
			}

			xtb_bt_new2(new_window, info_text, XTB_CENTERED, info_func,
				   (xtb_data) new_window, &new_info->info_frame
			);
			new_info->info= new_info->info_frame.win;
			XSaveContext( disp, new_info->info, frame_context, (caddr_t) &new_info->info_frame );

			xtb_bt_new2(new_window, label_text, XTB_CENTERED, label_func,
				   (xtb_data) new_window, &new_info->label_frame
			);
			new_info->label= new_info->label_frame.win;
			XSaveContext( disp, new_info->label, frame_context, (caddr_t) &new_info->label_frame );

			xtb_bt_new2(new_window, sht_text, XTB_CENTERED, ssht_func,
				   (xtb_data) new_window, &new_info->ssht_frame
			);
			XSaveContext( disp, new_info->ssht_frame.win, frame_context, (caddr_t) &new_info->ssht_frame );

			  /* The buttonrow with various YAveraging types is (again) textual, using dialogFont	*/
			if( xtb_UseColours && fabs( xtb_PsychoMetric_Gray(&zero) - xtb_PsychoMetric_Gray(&grid) )>= ButtonContrast ){
				xtb_init(disp, screen, zeroPixel, gridPixel, dialogFont.font, dialog_greekFont.font, True );
			}
			else if( xtb_UseColours && fabs( xtb_PsychoMetric_Gray(&norm) - xtb_PsychoMetric_Gray(&bg) )>= ButtonContrast ){
				xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, True );
			}
			else{
				xtb_init(disp, screen, black_pixel, white_pixel, dialogFont.font, dialog_greekFont.font, True );
			}

			xtb_br_new( new_window, N_YAv_SortTypes, YAv_SortTypes, 0,
				   YAv_SortFun, (xtb_data) new_window, &new_info->YAv_Sort_frame
			);
			xtb_describe( &new_info->YAv_Sort_frame, "Select type of sorting for YAveraging\n"
					"(Shift-Mod1-^ or Shift-Mod1-7)\n"
			);
			{ int i;
				for( i= 0; i< N_YAv_SortTypes; i++ ){
					xtb_describe( new_info->YAv_Sort_frame.framelist[i], YAv_Sort_Desc[i] );
					XSaveContext( disp, new_info->YAv_Sort_frame.framelist[i]->win, frame_context,
						(caddr_t) &new_info->YAv_Sort_frame
					);
				}
			}
			XSaveContext( disp, new_info->YAv_Sort_frame.win, frame_context, (caddr_t) &new_info->YAv_Sort_frame );

			XMoveWindow(disp, new_info->YAv_Sort_frame.win,
				(int) (sizehints->width- (5*BTNPAD+ new_info->YAv_Sort_frame.width+ new_info->cl_frame.width+
											new_info->hd_frame.width+ new_info->settings_frame.width+
											new_info->info_frame.width+ new_info->label_frame.width+ 
											new_info->ssht_frame.width+ 7* BTNINTER)
					),
				(int) (BTNPAD)
			);
			XMoveWindow(disp, new_info->close,
				(int) (sizehints->width- (BTNPAD+ new_info->cl_frame.width+ new_info->hd_frame.width+
											new_info->settings_frame.width+ new_info->info_frame.width+ new_info->label_frame.width+
											new_info->ssht_frame.width+ 5* BTNINTER)),
				(int) (BTNPAD)
			);
			XMoveWindow(disp, new_info->hardcopy,
				(int) (sizehints->width- (BTNPAD+ new_info->hd_frame.width+
											new_info->settings_frame.width+ new_info->info_frame.width+ new_info->label_frame.width+
											new_info->ssht_frame.width+ 4* BTNINTER)),
				(int) (BTNPAD)
			);
			XMoveWindow(disp, new_info->settings,
				(int) (sizehints->width- (BTNPAD+ new_info->settings_frame.width+ new_info->info_frame.width+ new_info->label_frame.width+
				new_info->ssht_frame.width+ 3* BTNINTER)),
				(int) (BTNPAD)
			);
			XMoveWindow(disp, new_info->info,
				(int) (sizehints->width- (BTNPAD+ new_info->info_frame.width+ new_info->label_frame.width+ 
				new_info->ssht_frame.width+ 2* BTNINTER)),
				(int) (BTNPAD)
			);
			XMoveWindow(disp, new_info->label,
				(int) (sizehints->width- (BTNPAD+ new_info->label_frame.width+ 
				new_info->ssht_frame.width+ 1* BTNINTER)),
				(int) (BTNPAD)
			);
			XMoveWindow(disp, new_info->ssht_frame.win,
				(int) (sizehints->width- (BTNPAD+ new_info->ssht_frame.width+ 0* BTNINTER)),
				(int) (BTNPAD)
			);

			xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );

			xtb_describe( &new_info->cl_frame, "Close Button\nCloses this window.\n");
			xtb_describe( &new_info->hd_frame, "Hardcopy Dialog Button\nClick to open print dialogue\n"
					"Shift-Click to print with current settings and close window\n"
			);
			xtb_describe( &new_info->settings_frame, "Settings Dialog Button\nClick to open settings dialogue");
			xtb_describe( &new_info->info_frame, "File Info Dialog Button\nClick to see comments/info in datafiles\n");
			xtb_describe( &new_info->label_frame,
				"Click to add an arrow with a text-label; first click defines \"arrow-point\"\n"
				" Press Shift to obtain a contextual label visible only when certain sets (marked, highlighted,..) are shown\n"
				" Press Mod1 (alt) to clip to the plotting area the textboxes of those labels that have the arrowpoint inside\n"
			);
			xtb_describe( &new_info->ssht_frame, "Set to silence drawing");
		}

		new_info->window= new_window;
		new_info->delete_it= 0;

		if( ux11_useDBE< 0 ){
			if( !(new_info->XDBE_buffer= XdbeAllocateBackBufferName( disp, new_window, XG_XDBE_SwapAction)) ){
				fprintf( StdErr, "NewWindow(): can't allocate the X11 DoubleBuffer buffer: feature disabled.\n" );
			}
		}

		SelectXInput( new_info );
		XDefineCursor(disp, new_window, (CursorCross)? noCursor : theCursor);

		  /* Set device info */
		Set_X(new_info, &(new_info->dev_info));
		if( !new_info->window && new_info->delete_it== -1 ){
			return( 0 );
		}
		if( X_psMarkers ){
			X_ps_Marks( new_info->dev_info.user_state, True );
		}
		else{
			X_ps_Marks( new_info->dev_info.user_state, False );
		}

		new_info->curs_cross.line[0].x1= -1;
		new_info->curs_cross.line[0].y1= -1;
		new_info->curs_cross.line[0].x2= -1;
		new_info->curs_cross.line[0].y2= -1;
		new_info->curs_cross.line[1].x1= -1;
		new_info->curs_cross.line[1].y1= -1;
		new_info->curs_cross.line[1].x2= -1;
		new_info->curs_cross.line[1].y2= -1;
		new_info->curs_cross.OldLabel[0]= '\0';

		if( new_window!= RootWindow( disp, screen) ){
			XMapRaised(disp, new_window);
			if( wmhints->initial_state!= IconicState ){
				new_info->mapped= 1;
			}
			XMapSubwindows( disp, new_window);
			if( no_buttons ){
				XUnmapSubwindows( disp, new_window);
				new_info->cl_frame.mapped= 0;
				new_info->hd_frame.mapped= 0;
				new_info->settings_frame.mapped= 0;
				new_info->info_frame.mapped= 0;
				new_info->label_frame.mapped= 0;
				new_info->ssht_frame.mapped= 0;
				new_info->YAv_Sort_frame.mapped= 0;
			}
		}

		New_win= new_window;
		new_info->redraw= -1;
		new_info->init_pass= 1;

		  /* Copy the flags once more...	*/
		CopyFlags( new_info, parent );
		new_info->transform.x_len= strlen(new_info->transform.x_process);
		new_info->transform.y_len= strlen(new_info->transform.y_process);
		new_info->process.data_init_len= strlen(new_info->process.data_init);
		new_info->process.data_before_len= strlen(new_info->process.data_before);
		new_info->process.data_process_len= strlen(new_info->process.data_process);
		new_info->process.data_after_len= strlen(new_info->process.data_after);
		new_info->process.data_finish_len= strlen(new_info->process.data_finish);
		new_info->process.draw_before_len= strlen(new_info->process.draw_before);
		new_info->process.draw_after_len= strlen(new_info->process.draw_after);
		new_info->process.dump_before_len= strlen(new_info->process.dump_before);
		new_info->process.dump_after_len= strlen(new_info->process.dump_after);
		new_info->process.enter_raw_after_len= strlen(new_info->process.enter_raw_after);
		new_info->process.leave_raw_after_len= strlen(new_info->process.leave_raw_after);

		XNextEvent( disp, &evt);
		if( debugFlag){
			fprintf( StdErr, "NewWindow: 0x%lx #%02d.%02d.%02d\n", New_win,
				new_info->parent_number, new_info->pwindow_number, new_info->window_number
			);
			fflush( StdErr);
		}

		Num_Windows+= 1;
		ConsWindow(new_info);

		Elapsed_Since(&new_info->draw_timer, True);
		SetWindowTitle( new_info, Tot_Time );

		return( new_window );
    } else {
		return (Window) 0;
    }
}

void DelWindowTransform( Transform *transform )
{
	Destroy_Form( &transform->C_x_process );
	Destroy_Form( &transform->C_y_process );
	xfree( transform->x_process );
	xfree( transform->y_process );
}

void DelWindowProcess( Process *process )
{
	Destroy_Form( &process->C_data_init );
	Destroy_Form( &process->C_data_before );
	Destroy_Form( &process->C_data_process );
	Destroy_Form( &process->C_data_after );
	Destroy_Form( &process->C_data_finish );
	Destroy_Form( &process->C_draw_before );
	Destroy_Form( &process->C_draw_after );
	Destroy_Form( &process->C_dump_before );
	Destroy_Form( &process->C_dump_after );
	Destroy_Form( &process->C_enter_raw_after );
	Destroy_Form( &process->C_leave_raw_after );
	xfree( process->data_init );
	xfree( process->data_before );
	xfree( process->data_process );
	xfree( process->data_after );
	xfree( process->data_finish );
	xfree( process->draw_before );
	xfree( process->draw_after );
	xfree( process->dump_before );
	xfree( process->dump_after );
	xfree( process->enter_raw_after );
	xfree( process->leave_raw_after );
}

int DelWindow( Window win, LocalWin *wi)
/*
 * This routine actually deletes the specified window and
 * decrements the window count.
 */
{ xtb_data info;
  int idx;
  static LocalWin *active= NULL;
  XGPen *pen_list;

	if( wi->delete_it== -1 ){
		return(0);
	}
	if( active== wi ){
		return(-1);
	}
	active= wi;

	if( ActiveWin== wi ){
		ActiveWin= NULL;
	}
	if( LastDrawnWin== wi ){
		LastDrawnWin= NULL;
	}

	  /* If there is a dialogue connected to this window, close it.	*/
	if( wi->SD_Dialog ){
	  xtb_frame *SD_Dialog= wi->SD_Dialog;
		if( Num_Windows<= 1 ){
			_CloseSD_Dialog( SD_Dialog, True );
		}
		else{
			CloseSD_Dialog( SD_Dialog );
		}
		wi->SD_Dialog= NULL;
	}
	if( wi->HO_Dialog ){
	  Window win= wi->HO_Dialog->win;
		CloseHO_Dialog( wi->HO_Dialog );
		wi->HO_Dialog= NULL;
		if( Num_Windows<= 1 ){
			XDestroyWindow(disp, win );
		}
	}
	Delete_ULabels(wi);
	if( wi->XDBE_buffer ){
		XdbeDeallocateBackBufferName( disp, wi->XDBE_buffer );
	}
	if( (pen_list= wi->pen_list) ){
	  XGPen *pen;
	  int i;
		while( pen_list ){
			pen= pen_list;
#if 1
			PenReset( pen, True );
#else
			if( pen->hlpixelCName ){
				FreeColor( &pen->hlpixelValue, &pen->hlpixelCName );
			}
			if( pen->position ){
				for( i= 0; i< pen->positions; i++ ){
					xfree( pen->position[i].text );
					xfree( pen->position[i].TextBox );
					if( pen->position[i].colour.pixvalue< 0 && pen->position[i].pixelCName ){
						FreeColor( &(pen->position[i].colour.pixelValue), &pen->position[i].pixelCName );
					}
					if( pen->position[i].flcolour.pixvalue< 0 && pen->position[i].flpixelCName ){
						FreeColor( &(pen->position[i].flcolour.pixelValue), &pen->position[i].flpixelCName );
/* 						XFreeColors( disp, cmap, &pen->position[i].flpixelValue, 1, 0);	*/
/* 						xfree( pen->position[i].flpixelCName );	*/
					}
				}
			}
			xfree( pen->position );
#endif
			pen_list= pen->next;
			xfree( pen );
		}
		wi->pen_list= NULL;
		wi->numPens= 0;
	}

	DelWindowTransform( &wi->transform );
	DelWindowProcess( &wi->process );

	if( wi!= &StubWindow ){
		wi->dev_info.xg_silent( wi->dev_info.user_state, True );

		if( win ){
			XDeleteContext(disp, win, win_context);
		}

#ifdef TOOLBOX
		{ int i;
			for( i= 0; i< N_YAv_SortTypes; i++ ){
				XDeleteContext( disp, wi->YAv_Sort_frame.framelist[i]->win, frame_context );
			}
		}
		XDeleteContext( disp, wi->YAv_Sort_frame.win, frame_context );
		XDeleteContext( disp, wi->close, frame_context );
		XDeleteContext( disp, wi->hardcopy, frame_context );
		XDeleteContext( disp, wi->settings, frame_context );
		XDeleteContext( disp, wi->info, frame_context );
		XDeleteContext( disp, wi->label, frame_context );
		XDeleteContext( disp, wi->ssht_frame.win, frame_context );
		xtb_br_del(wi->YAv_Sort_frame.win);
		wi->YAv_Sort_frame.win= 0;
		xtb_bt_del(wi->close, &info);
		wi->close= 0;
		xtb_bt_del(wi->hardcopy, &info);
		wi->hardcopy= 0;
		xtb_bt_del(wi->settings, &info);
		wi->settings= 0;
		xtb_bt_del(wi->info, &info);
		wi->info= 0;
		xtb_bt_del(wi->label, &info);
		wi->label= 0;
		xtb_bt_del(wi->ssht_frame.win, &info);
		wi->ssht_frame.win= 0;
#endif
		wi->window= (Window) 0;
		wi->delete_it= -1;
	}
	xfree( wi->title_template);
	xfree( wi->draw_set );
	xfree( wi->mark_set );
	xfree( wi->group );
	xfree( wi->fileNumber );
	xfree( wi->new_file );
	xfree( wi->plot_only_set );
	xfree( wi->legend_line );
	if( wi->hard_devices ){
		for( idx= 0; idx< hard_count; idx++ ){
			xfree( wi->hard_devices[idx].dev_file );
			xfree( wi->hard_devices[idx].dev_printer );
		}
		xfree( wi->hard_devices );
	}
	xfree( wi->xcol );
	xfree( wi->ycol );
	xfree( wi->ecol );
	xfree( wi->lcol );
	  /* 990614: MaxSets below used to be setNumber...	*/
	for( idx= 0; idx< MaxSets; idx++ ){
		if( wi->curve_len ){
			xfree( wi->curve_len[idx] );
		}
		if( wi->error_len ){
			xfree( wi->error_len[idx] );
		}
#ifdef TR_CURVE_LEN
		if( wi->tr_curve_len ){
			xfree( wi->tr_curve_len[idx] );
		}
#endif
		if( wi->set_O ){
			xfree( wi->set_O[idx].sample );
		}
		if( wi->set_tr_O ){
			xfree( wi->set_tr_O[idx].sample );
		}
		if( wi->discardpoint ){
			xfree( wi->discardpoint[idx] );
		}
	}
	xfree( wi->curve_len );
	xfree( wi->error_len );
#ifdef TR_CURVE_LEN
	xfree( wi->tr_curve_len );
#endif
	xfree( wi->discardpoint );
	xfree( wi->set_X);
	xfree( wi->set_Y);
	xfree( wi->set_E);
	xfree( wi->set_O);
	xfree( wi->set_tr_X);
	xfree( wi->set_tr_Y);
	xfree( wi->set_tr_E);
	xfree( wi->set_tr_O);

	if( wi->ValCat_X ){
		Free_ValCat( wi->ValCat_X );
	}
	if( wi->ValCat_XFont ){
		Free_CustomFont( wi->ValCat_XFont );
		xfree( wi->ValCat_XFont );
	}
	if( wi->ValCat_Y ){
		Free_ValCat( wi->ValCat_Y );
	}
	if( wi->ValCat_YFont ){
		Free_CustomFont( wi->ValCat_YFont );
		xfree( wi->ValCat_YFont );
	}
	if( wi->ValCat_I ){
		Free_ValCat( wi->ValCat_I );
	}
	if( wi->ValCat_IFont ){
		Free_CustomFont( wi->ValCat_IFont );
		xfree( wi->ValCat_IFont );
	}
	if( strlen( wi->textrel.gs_fn) ){
		unlink( wi->textrel.gs_fn );
	}

	if( wi!= &StubWindow ){
		close_X( wi->dev_info.user_state );
		xfree( wi->dev_info.user_state );
		  /* 20010720 Just put back the window id in order to completely remove <wi> from our linked WindowList *and* from 
		   \ the xtb registry! It is probably a good idea to call RemoveWindow as the last thing before destroying
		   \ the window (but then again maybe not!), but RemoveWindow() needs to know the window id in order to be able
		   \ to unregister <wi> from the xtb registry.
		   */
		wi->window= win;
		RemoveWindow( wi);
		wi->window= 0;
		if( wi->wmhints ){
			XFree( wi->wmhints );
		}
		if( win && win!= RootWindow( disp, screen) ){
			XDestroyWindow(disp, win);
			if( ascanf_window== win ){
				ascanf_window= 0;
			}
		}
		wi->mapped= 0;
		Num_Windows -= 1;
	}
	XFlush( disp );
	if( !Exitting ){
		if( Num_Windows<= 0){
/* 			CleanUp();	*/
/* 			exit(0);	*/
			ExitProgramme(0);
		}
		if( !WindowList ){
			longjmp( toplevel, 2);
		}
	}
	active= NULL;
	return(1);
}

extern int Animating;
void XGIconify( LocalWin *wi )
{ LocalWindows *WL= NULL;
  LocalWin *lwi;
  extern int handle_event_times;
  int hit= handle_event_times;
	if( hit== 0 ){
		WL= WindowList;
		lwi= WL->wi;
	}
	else{
		lwi= wi;
	}
	do{
		if( hit== 0 ){
			if( lwi->SD_Dialog && lwi->SD_Dialog->mapped> 0 ){
				if( !XIconifyWindow( disp, lwi->SD_Dialog->win, screen ) ){
					XUnmapWindow( disp, lwi->SD_Dialog->win );
				}
				lwi->SD_Dialog->mapped= 0;
			}
			if( lwi->HO_Dialog && lwi->HO_Dialog->mapped> 0 ){
				if( !XIconifyWindow( disp, lwi->HO_Dialog->win, screen ) ){
					XUnmapWindow( disp, lwi->HO_Dialog->win );
				}
				lwi->HO_Dialog->mapped= 0;
			}
		}
		if( lwi->mapped ){
			if( !XIconifyWindow( disp, lwi->window, screen) ){
				if( lwi->window!= wi->window ){
				  /* This is for braindead windowmanagers who might remove all windows
				   \ from their list of icons.
				   */
					XUnmapWindow( disp, lwi->window );
				}
			}
			lwi->mapped= 0;
			lwi->animate= False;
			lwi->animating= False;
			lwi->halt= True;
		}
		if( WL ){
			WL= WL->next;
			lwi= (WL)? WL->wi : 0;
		}
	} while( WL && lwi );
	Animating= False;
}

char *XGFetchName( LocalWin *wi)
{ static char *_Title= NULL;
	if( wi ){
		XFetchName( disp, wi->window, &_Title );
	}
	else if( _Title ){
		XFree( _Title );
		_Title= NULL;
		return( "<NULL>" );
	}
	return( _Title );
}

  /* Percentual changes in fitting. Caution: can be Inf	*/
double Fit_ChangePerc_X= 0, Fit_ChangePerc_Y= 0, Fit_ChangePerc= 0;

int Fit_XBounds( LocalWin *wi, Boolean redraw )
{ int change= 0;
  double _loX, _hiX, dlx= 0, dhx= 0;
  int pF= wi->polarFlag, old_silent, dc= wi->draw_count, fitting= wi->fitting, silenced= wi->silenced,
	txl= wi->transform.x_len, tyl= wi->transform.y_len, ta= wi->transform_axes;

	if( !wi->fitting ){
		wi->silenced= True;
		if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && !ta ){
			  /* Same applies as to process_bounds. We want to know the bounds exclusif of TRANSFORM_??
			   \ effects, so we temporarily disable them
			   */
			wi->transform.x_len= 0;
			wi->transform.y_len= 0;
			wi->transform_axes= -1;
			Fit_XBounds( wi, True );
			  /* Careful to restore: DrawWindow *may* have installed a new routine!	*/
			if( !wi->transform.x_len ){
				wi->transform.x_len= txl;
			}
			if( !wi->transform.y_len ){
				wi->transform.y_len= tyl;
			}
			  /* Store the raw values.	*/
			wi->win_geo.R_UsrOrgX= wi->win_geo.bounds._loX;
			wi->win_geo.R_UsrOppX= wi->win_geo.bounds._hiX;
			wi->win_geo.R_UsrOrgY= wi->win_geo.bounds._loY;
			wi->win_geo.R_UsrOppY= wi->win_geo.bounds._hiY;
			  /* Now do the whole thing again...	*/
		}
		wi->polarFlag= False;
		if( redraw ||
			(!wi->raw_display && /* wi->process_bounds && */
				(wi->transform.x_len || wi->process.data_process_len ||
					wi->process.draw_before_len || wi->process.draw_after_len )
			)
		){
			  /* We need to know the true bounds, not the bounds of the processed data.
			   \ If the axes are *not* subjected to processing/transformations, this is not
			   \ necessary.
			   */
			if( wi->process_bounds ){
				wi->raw_display= True;
			}
			wi->redraw= 1;
			wi->draw_count= 0;
			old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			TitleMessage( wi, "(Re)Scaling... X" );
			if( !RemoteConnection ){
				XFlush( disp);
			}
			wi->fitting= fitXAxis;
			wi->transform_axes= 1;
			DrawWindow( wi );
			wi->fitting= fitting;
			wi->dev_info.xg_silent( wi->dev_info.user_state, old_silent );
			if( wi->process_bounds ){
				wi->raw_display= False;
			}
		}
		else if( pF ){
			wi->redraw= 1;
			wi->draw_count= 0;
			old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			TitleMessage( wi, "(Re)Scaling... X" );
			if( !RemoteConnection ){
				XFlush( disp);
			}
			wi->fitting= fitXAxis;
			wi->transform_axes= 1;
			DrawWindow( wi );
			wi->fitting= fitting;
			wi->dev_info.xg_silent( wi->dev_info.user_state, old_silent );
		}
		wi->transform_axes= ta;
		wi->draw_count= dc;
	}
	_loX= (wi->logXFlag>0 && wi->SS_X.nr_min<= 0)? wi->SS_X.pos_min :
		(wi->BarDimensionsSet)? MIN(wi->SS_X.nr_min, wi->MinBarX) : wi->SS_X.nr_min;
	_hiX= (wi->BarDimensionsSet)? MAX(wi->SS_X.nr_max, wi->MaxBarX) : wi->SS_X.nr_max;
	if( wi->logXFlag > 0 ){
		if( _loX<= 0 ){
			_loX= (wi->win_geo.pure_bounds._loX> 0)?
				wi->win_geo.pure_bounds._loX : wi->win_geo.pure_bounds._lopX;
		}
		if( _hiX<= 0 ){
			_hiX= wi->win_geo.pure_bounds._hiX;
		}
	}
	if( pF ){
		if( abs(wi->fit_xbounds)== 2 ){
		  /* should adapt the radix. This may not work as expected in
		   \ all cases...
		   \ Adding 1e-14 accounts for numeric inprecision (?) on HP9000/730
		   */
		  double pB= fabs( _hiX - _loX)+ 2e-14;
			if( wi->radix!= pB ){
				change+= 1;
				wi->radix= pB;
				Gonio_Base( wi, wi->radix, wi->radix_offset );
			}
		}
	}
	else if( !wi->win_geo.user_coordinates ){
	  /* See if we want padding or not. In polar mode, no extra padding is performed
	   \ (could result in "full circle" display instead of "half circle".
	   */
	  double _loY= wi->win_geo.bounds._loY,
			_hiY= wi->win_geo.bounds._hiY;
		_loX= Trans_X( wi, _loX);
		_hiX= Trans_X( wi, _hiX);
		_loY= Trans_Y( wi, _loY, 1);
		_hiY= Trans_Y( wi, _hiY, 1);
		if( wi->win_geo.padding ){
		  double padx= (_hiX - _loX)/ wi->win_geo.padding;
			_loX-= padx;
			_hiX+= padx;
		}
		_loX= Reform_X( wi, _loX, _loY );
		_hiX= Reform_X( wi, _hiX, _hiY );
	}
	if( wi->Xscale ){
		_loX/= wi->Xscale;
		_hiX/= wi->Xscale;
	}

	if( wi->fit_after_draw ){
	  double _hiY= wi->win_geo.bounds._hiY, _loY= wi->win_geo.bounds._loY;
		AlterGeoBounds( wi, True, &_loX, &_loY, &_hiX, &_hiY );
	}

	if( wi->win_geo.bounds._loX!= _loX ){
		change+= 1;
		dlx= _loX* 100.0/ wi->win_geo.bounds._loX;
		wi->win_geo.bounds._loX= _loX;
	}
	else{
		dlx= 100;
	}
	if( wi->win_geo.bounds._hiX!= _hiX ){
		change+= 1;
		dhx= _hiX* 100.0/ wi->win_geo.bounds._hiX;
		wi->win_geo.bounds._hiX= _hiX;
	}
	else{
		dhx= 100;
	}
	if( SIGN(dlx)!= SIGN(dhx) ){
		Fit_ChangePerc_X= Fit_ChangePerc= fabs(dlx)+ fabs(dhx)- 100;
	}
	else{
		if( dlx< 0 ){
			Fit_ChangePerc_X= Fit_ChangePerc= MIN(dlx, dhx)- 100;
		}
		else{
			Fit_ChangePerc_X= Fit_ChangePerc= MAX(dlx, dhx)- 100;
		}
	}
	wi->polarFlag= pF;
	wi->silenced= silenced;
	TitleMessage( wi, NULL);
	return( change );
}

int Fit_YBounds( LocalWin *wi, Boolean redraw )
{ int change= 0;
  double _loY, _hiY, dly, dhy;
  int pF= wi->polarFlag, old_silent, dc= wi->draw_count, fitting= wi->fitting, silenced= wi->silenced,
		txl= wi->transform.x_len, tyl= wi->transform.y_len, ta= wi->transform_axes;

	if( !wi->fitting ){
		wi->silenced= True;
		if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && !ta ){
			  /* Same applies as to process_bounds. We want to know the bounds exclusif of TRANSFORM_??
			   \ effects, so we temporarily disable them
			   */
			wi->transform.x_len= 0;
			wi->transform.y_len= 0;
			wi->transform_axes= -1;
			Fit_YBounds( wi, True );
			  /* Careful to restore: DrawWindow *may* have installed a new routine!	*/
			if( !wi->transform.x_len ){
				wi->transform.x_len= txl;
			}
			if( !wi->transform.y_len ){
				wi->transform.y_len= tyl;
			}
			  /* Store the raw values.	*/
			wi->win_geo.R_UsrOrgX= wi->win_geo.bounds._loX;
			wi->win_geo.R_UsrOppX= wi->win_geo.bounds._hiX;
			wi->win_geo.R_UsrOrgY= wi->win_geo.bounds._loY;
			wi->win_geo.R_UsrOppY= wi->win_geo.bounds._hiY;
			  /* Now do the whole thing again...	*/
		}
		wi->polarFlag= False;
		if( (!wi->raw_display && /* wi->process_bounds && */
			(wi->transform.y_len || wi->process.data_process_len ||
				wi->process.draw_before_len || wi->process.draw_after_len )
			) ||
			redraw
		){
			  /* We need to know the true bounds, not the bounds of the processed data.
			   \ If the axes are *not* subjected to processing/transformations, this is not
			   \ necessary.
			   */
			if( wi->process_bounds ){
				wi->raw_display= True;
			}
			wi->redraw= 1;
			wi->draw_count= 0;
			old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			TitleMessage( wi, "(Re)Scaling... Y" );
			if( !RemoteConnection ){
				XFlush( disp);
			}
			wi->fitting= fitYAxis;
			wi->transform_axes= 1;
			DrawWindow( wi );
			wi->fitting= fitting;
			wi->dev_info.xg_silent( wi->dev_info.user_state, old_silent );
			if( wi->process_bounds ){
				wi->raw_display= False;
			}
		}
		else if( pF ){
			wi->redraw= 1;
			wi->draw_count= 0;
			old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			TitleMessage( wi, "(Re)Scaling... Y" );
			if( !RemoteConnection ){
				XFlush( disp);
			}
			wi->fitting= fitYAxis;
			wi->transform_axes= 1;
			DrawWindow( wi );
			wi->fitting= fitting;
			wi->dev_info.xg_silent( wi->dev_info.user_state, old_silent );
		}
		wi->transform_axes= ta;
		wi->draw_count= dc;
	}
	if( wi->use_errors ){
		_loY= (wi->logYFlag>0 && wi->SS_LY.nr_min<= 0)?
			MIN(wi->SS_Y.pos_min,wi->SS_LY.pos_min) : MIN( wi->SS_Y.nr_min, wi->SS_LY.nr_min);
	}
	else{
		_loY= (wi->logYFlag>0 && wi->SS_Y.nr_min<= 0)? wi->SS_Y.pos_min : wi->SS_Y.nr_min;
	}
	if( wi->use_errors ){
		_hiY= MAX( wi->SS_Y.nr_max, wi->SS_HY.nr_max);
	}
	else{
		_hiY= wi->SS_Y.nr_max;
	}
	if( wi->logYFlag > 0 ){
		if( _loY<= 0 ){
			_loY= (wi->win_geo.pure_bounds._loY> 0)?
				wi->win_geo.pure_bounds._loY : wi->win_geo.pure_bounds._lopY;
		}
		if( _hiY<= 0 ){
			_hiY= wi->win_geo.pure_bounds._hiY;
		}
	}
	  /* Check if we want padding. The Y-coordinates in polar-mode ("radius")
	   \ do allow padding, contrary to the X-coordinates ("angle").
	   */
	if( !wi->win_geo.user_coordinates ){
	  double _loX= wi->win_geo.bounds._loX,
			_hiX= wi->win_geo.bounds._hiX;
		_loY= Trans_Y( wi, _loY, 1);
		_hiY= Trans_Y( wi, _hiY, 1);
		_loX= Trans_X( wi, _loX);
		_hiX= Trans_X( wi, _hiX);
		if( wi->win_geo.padding ){
		  double pady= (_hiY - _loY)/ wi->win_geo.padding;
			_loY-= pady;
			_hiY+= pady;
		}
		_loY= Reform_Y( wi, _loY, _loX );
		_hiY= Reform_Y( wi, _hiY, _hiX );
	}
	if( wi->Yscale ){
		_loY/= wi->Yscale;
		_hiY/= wi->Yscale;
	}

	if( wi->fit_after_draw ){
	  double _hiX= wi->win_geo.bounds._hiX, _loX= wi->win_geo.bounds._loX;
		AlterGeoBounds( wi, True, &_loX, &_loY, &_hiX, &_hiY );
	}

	if( wi->win_geo.bounds._loY!= _loY ){
		change+= 1;
		dly= _loY* 100.0/ wi->win_geo.bounds._loY;
		wi->win_geo.bounds._loY= _loY;
	}
	else{
		dly= 100;
	}
	if( wi->win_geo.bounds._hiY!= _hiY ){
		change+= 1;
		dhy= _hiY* 100.0/ wi->win_geo.bounds._hiY;
		wi->win_geo.bounds._hiY= _hiY;
	}
	else{
		dhy= 100;
	}
	if( SIGN(dly)!= SIGN(dhy) ){
		Fit_ChangePerc_Y= Fit_ChangePerc= fabs(dly)+ fabs(dhy)- 100;
	}
	else{
		if( dly< 0 ){
			Fit_ChangePerc_Y= Fit_ChangePerc= MIN(dly, dhy)- 100;
		}
		else{
			Fit_ChangePerc_Y= Fit_ChangePerc= MAX(dly, dhy)- 100;
		}
	}
	wi->polarFlag= pF;
	wi->silenced= silenced;
	TitleMessage( wi, NULL);
	return( change );
}

int Fit_XYBounds( LocalWin *wi, Boolean redraw )
{ int change= 0;
  double _loX, _hiX, dlx, dhx;
  double _loY, _hiY, dly, dhy;
#ifdef DEBUG
  double _lX, _hX, _lY, _hY;
  double _wlX, _whX, _wlY, _whY;
#endif
  int pF= wi->polarFlag, old_silent, dc= wi->draw_count, fitting= wi->fitting, silenced= wi->silenced,
	txl= wi->transform.x_len, tyl= wi->transform.y_len, ta= wi->transform_axes;

	if( !wi->fitting ){
		wi->silenced= True;
	/* 	if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && !ta && (txl || tyl) )	*/
		if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && !ta )
		{
			  /* Same applies as to process_bounds. We want to know the bounds exclusif of TRANSFORM_??
			   \ effects, so we temporarily disable them
			   */
			wi->transform.x_len= 0;
			wi->transform.y_len= 0;
			wi->transform_axes= -1;
			Fit_XYBounds( wi, True );
			  /* Careful to restore: DrawWindow *may* have installed a new routine!	*/
			if( !wi->transform.x_len ){
				wi->transform.x_len= txl;
			}
			if( !wi->transform.y_len ){
				wi->transform.y_len= tyl;
			}
			  /* Store the raw values.	*/
			wi->win_geo.R_UsrOrgX= wi->win_geo.bounds._loX;
			wi->win_geo.R_UsrOppX= wi->win_geo.bounds._hiX;
			wi->win_geo.R_UsrOrgY= wi->win_geo.bounds._loY;
			wi->win_geo.R_UsrOppY= wi->win_geo.bounds._hiY;
			  /* Now do the whole thing again...	*/
		}
		wi->polarFlag= False;
		if( (!wi->raw_display && /* wi->process_bounds && */
				(wi->transform.x_len || wi->process.data_process_len || wi->transform.y_len ||
				wi->process.draw_before_len || wi->process.draw_after_len )
			) ||
			redraw
		){
			  /* We need to know the true bounds, not the bounds of the processed data.
			   \ If the axes are *not* subjected to processing/transformations, this is not
			   \ necessary.
			   */
			if( wi->process_bounds ){
				wi->raw_display= True;
			}
			wi->redraw= 1;
			wi->draw_count= 0;
			old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			TitleMessage( wi, "(Re)Scaling... XY" );
			if( !RemoteConnection ){
				XFlush( disp);
			}
			wi->fitting= fitBothAxes;
			  /* Now don't do anything special for the axes (transform_axes means
			   \ just show the results of the transformations on the axes)
			   */
			wi->transform_axes= 1;
			DrawWindow( wi );
			wi->fitting= fitting;
			wi->dev_info.xg_silent( wi->dev_info.user_state, old_silent );
			if( wi->process_bounds ){
				wi->raw_display= False;
			}
		}
		else if( pF ){
			wi->redraw= 1;
			wi->draw_count= 0;
			old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
			TitleMessage( wi, "(Re)Scaling... XY" );
			if( !RemoteConnection ){
				XFlush( disp);
			}
			wi->fitting= fitBothAxes;
			wi->transform_axes= 1;
			DrawWindow( wi );
			wi->fitting= fitting;
			wi->dev_info.xg_silent( wi->dev_info.user_state, old_silent );
		}
		wi->transform_axes= ta;
		wi->draw_count= dc;
	}
	  /* 20050509: this is where we can decide to alter nr_min and nr_max (working with local copies?!).
	   \ The interest might be to use quantiles, i.e. the boundaries of a certain percentage of the data,
	   \ which could be used to exclude a certain number of outliers.
	   */
	_loX= (wi->logXFlag>0 && wi->SS_X.nr_min<= 0)? wi->SS_X.pos_min :
		(wi->BarDimensionsSet)? MIN(wi->SS_X.nr_min, wi->MinBarX) : wi->SS_X.nr_min;
	_hiX= (wi->BarDimensionsSet)? MAX(wi->SS_X.nr_max, wi->MaxBarX) : wi->SS_X.nr_max;
	if( wi->logXFlag > 0 ){
		if( _loX<= 0 ){
			_loX= (wi->win_geo.pure_bounds._loX> 0)?
				wi->win_geo.pure_bounds._loX : wi->win_geo.pure_bounds._lopX;
		}
		if( _hiX<= 0 ){
			_hiX= wi->win_geo.pure_bounds._hiX;
		}
	}
	if( wi->use_errors ){
		_loY= (wi->logYFlag>0 && wi->SS_LY.nr_min<= 0)?
			MIN(wi->SS_Y.pos_min,wi->SS_LY.pos_min) : MIN( wi->SS_Y.nr_min, wi->SS_LY.nr_min);
	}
	else{
		_loY= (wi->logYFlag>0 && wi->SS_Y.nr_min<= 0)? wi->SS_Y.pos_min : wi->SS_Y.nr_min;
	}
	if( wi->use_errors ){
		_hiY= MAX( wi->SS_Y.nr_max, wi->SS_HY.nr_max);
	}
	else{
		_hiY= wi->SS_Y.nr_max;
	}
	if( wi->logYFlag > 0 ){
		if( _loY<= 0 ){
			_loY= (wi->win_geo.pure_bounds._loY> 0)?
				wi->win_geo.pure_bounds._loY : wi->win_geo.pure_bounds._lopY;
		}
		if( _hiY<= 0 ){
			_hiY= wi->win_geo.pure_bounds._hiY;
		}
	}
	if( pF ){
		if( abs(wi->fit_xbounds)== 2 ){
		  /* should adapt the radix. This may not work as expected in
		   \ all cases...
		   \ Adding 1e-14 accounts for numeric inprecision (?) on HP9000/730
		   */
		  double pB= fabs( _hiX - _loX)+ 2e-14;
			if( wi->radix!= pB ){
				change+= 1;
				wi->radix= pB;
				Gonio_Base( wi, wi->radix, wi->radix_offset );
			}
		}
	}
	else{
	  /* See if we want padding or not. In polar mode, no extra padding is performed
	   \ (could result in "full circle" display instead of "half circle".
	   */
		if( !wi->win_geo.user_coordinates ){
			_loX= Trans_X( wi, _loX);
			_hiX= Trans_X( wi, _hiX);
			_loY= Trans_Y( wi, _loY, 1);
			_hiY= Trans_Y( wi, _hiY, 1);
			if( wi->win_geo.padding ){
#if defined(linux) && defined(__GNUC__) && defined(i386)
			    /* 20000502 RJB: another gcc/pentium weird NaN result??!	*/
			  double padx= (_hiX - _loX);
			  double pady= (_hiY - _loY);
			  double pad= wi->win_geo.padding;
				padx/= pad;
				pady/= pad;
#else
			  double padx= (_hiX - _loX)/ wi->win_geo.padding;
			  double pady= (_hiY - _loY)/ wi->win_geo.padding;
#endif
				_loX-= padx;
				_hiX+= padx;
				_loY-= pady;
				_hiY+= pady;
			}
			_loX= Reform_X( wi, _loX, _loY );
			_hiX= Reform_X( wi, _hiX, _hiY );
			_loY= Reform_Y( wi, _loY, _loX );
			_hiY= Reform_Y( wi, _hiY, _hiX );
		}
	}
	if( wi->Xscale ){
		_loX/= wi->Xscale;
		_hiX/= wi->Xscale;
	}
	if( wi->Yscale ){
		_loY/= wi->Yscale;
		_hiY/= wi->Yscale;
	}

#ifdef DEBUG
	_lX= _loX;
	_hX= _hiX;
	_lY= _loY;
	_hY= _hiY;
	_wlX= wi->win_geo.bounds._loX;
	_whX= wi->win_geo.bounds._hiX;
	_wlY= wi->win_geo.bounds._loY;
	_whY= wi->win_geo.bounds._hiY;
#endif
	if( wi->fit_after_draw ){
		AlterGeoBounds( wi, True, &_loX, &_loY, &_hiX, &_hiY );
	}

	if( wi->win_geo.bounds._loX!= _loX ){
		change+= 1;
		dlx= _loX* 100.0/ wi->win_geo.bounds._loX;
		wi->win_geo.bounds._loX= _loX;
	}
	else{
		dlx= 100;
	}
	if( wi->win_geo.bounds._hiX!= _hiX ){
		change+= 1;
		dhx= _hiX* 100.0/ wi->win_geo.bounds._hiX;
		wi->win_geo.bounds._hiX= _hiX;
	}
	else{
		dhx= 100;
	}
	if( SIGN(dlx)!= SIGN(dhx) ){
		Fit_ChangePerc_X= fabs(dlx)+ fabs(dhx)- 100;
	}
	else{
		if( dlx< 0 ){
			Fit_ChangePerc_X= MIN(dlx, dhx)- 100;
		}
		else{
			Fit_ChangePerc_X= MAX(dlx, dhx)- 100;
		}
	}
	if( wi->win_geo.bounds._loY!= _loY ){
		change+= 1;
		dly= _loY* 100.0/ wi->win_geo.bounds._loY;
		wi->win_geo.bounds._loY= _loY;
	}
	else{
		dly= 100;
	}
	if( wi->win_geo.bounds._hiY!= _hiY ){
		change+= 1;
		dhy= _hiY* 100.0/ wi->win_geo.bounds._hiY;
		wi->win_geo.bounds._hiY= _hiY;
	}
	else{
		dhy= 100;
	}
	if( SIGN(dly)!= SIGN(dhy) ){
		Fit_ChangePerc_Y= fabs(dly)+ fabs(dhy)- 100;
	}
	else{
		if( dly< 0 ){
			Fit_ChangePerc_Y= MIN(dly, dhy)- 100;
		}
		else{
			Fit_ChangePerc_Y= MAX(dly, dhy)- 100;
		}
	}
	if( fabs(Fit_ChangePerc_X)> fabs(Fit_ChangePerc_Y) ){
		Fit_ChangePerc= Fit_ChangePerc_X;
	}
	else{
		Fit_ChangePerc= Fit_ChangePerc_Y;
	}
	wi->polarFlag= pF;
	wi->silenced= silenced;
	TitleMessage( wi, NULL);
	return( change );
}

LocalWin *StackWindows(int direction)
{  LocalWindows *l; 
   LocalWin *wi;
	if( direction== 'd' ){
	  /* Newest at bottom of "stack"	*/
		l= WindowList;
		while( l ){
			wi= l->wi;
			if( wi->delete_it!= -1 && wi->mapped ){
				XMapRaised( disp, wi->window);
				if( debugFlag ){
					fprintf( StdErr, "StackWindows(): Raising window \"%s\"\n", XGFetchName(wi) );
				}
			}
			l= l->next;
		}
	}
	else if( direction== 'u' ){
	  /* Newest at top of "stack"	*/
		l= WindowListTail;
		while( l ){
			wi= l->wi;
			if( wi->delete_it!= -1 && wi->mapped ){
				XMapRaised( disp, wi->window);
				if( debugFlag ){
					fprintf( StdErr, "StackWindows(): Raising window \"%s\"\n", XGFetchName(wi) );
				}
			}
			l= l->prev;
		}
	}
	return( wi);
}

#define RND(val)	((int) ((val) + 0.5))

LocalWin *ZoomWindow_PS_Size( LocalWin *wi, int set_aspect, double aspect, Boolean raise )
{
  int resized= 0;
  XWindowAttributes win_attr;
  XSizeHints hints;
  extern int Synchro_State;
  extern void *X_Synchro();
  int sync= Synchro_State;

	Synchro_State= 0;
	X_Synchro(wi);
	if( wi->dev_info.resized> 0 ){
		XResizeWindow(disp, wi->window,
			(win_attr.width= wi->dev_info.old_area_w),
			(win_attr.height= wi->dev_info.old_area_h)
		);
		resized= 1;
		wi->aspect_ratio= 0;
	}
	else if( set_aspect ){
	  int xrange= wi->XOppX- wi->XOrgX,
			yrange= wi->XOppY - wi->XOrgY,
			n= 0, w, h;
	  double delta_x, delta_y;
	  Boolean maxpect= True;
		if( !aspect ){
			aspect= 1.0;
		}
		wi->aspect_ratio= aspect;

		if( NaN(aspect) ){
		  /* NaN signals that we should find the same aspect as those of the axes. This
		   \ is the "window-size-adapting" homologue of the "1:1" function in the Settings
		   \ Dialog.
		   */
			aspect= fabs( (wi->win_geo.bounds._hiX - wi->win_geo.bounds._loX) /
					(wi->win_geo.bounds._hiY - wi->win_geo.bounds._loY)
				);
			maxpect= False;
		}

		if( aspect< 0 ){
			aspect*= -1;
			maxpect= False;
		}

		if( maxpect ){
			  /* This will ensure "maxpect"	*/
			wi->dev_info.area_w= XG_DisplayWidth(disp, screen, wi);
			wi->dev_info.area_h= XG_DisplayHeight(disp, screen, wi)- WM_TBAR;
		}

		w= wi->dev_info.area_w, h= wi->dev_info.area_h;
		delta_x= (aspect* yrange - xrange);
		delta_y= (xrange/ aspect - yrange);
		wi->halt= 0;
		*ascanf_escape_value= ascanf_escape= 0;
		while( (double)xrange/ (double)yrange!= aspect && n< 50 && (delta_x || delta_y) ){
		  int rd= wi->raw_display, ro= wi->raw_once, rv= wi->raw_val;
			if( fabs(delta_x)< MAXINT && fabs(delta_y)< MAXINT ){
			  int d_x= (int) delta_x, d_y= (int) delta_y;
				if( xrange>= yrange && wi->dev_info.area_w> 0 ){
					if( wi->dev_info.area_h+ d_y< XG_DisplayHeight(disp, screen, wi)-WM_TBAR ){
						wi->dev_info.area_h+= d_y;
					}
					else{
						wi->dev_info.area_w-= d_x;
					}
				}
				else if( yrange> xrange ){
					if( wi->dev_info.area_w+ d_x< XG_DisplayWidth(disp, screen, wi) ){
						wi->dev_info.area_w+= d_x;
					}
					else{
						wi->dev_info.area_h-= d_y;
					}
				}
				wi->dev_info.area_h= wi->dev_info.area_h & 0x0000ffff;
				wi->dev_info.area_w= wi->dev_info.area_w & 0x0000ffff;
				if( wi->dev_info.area_w> 0 && wi->dev_info.area_h> 0 ){
					TransformCompute( wi, False );
					if( (n % 5)== 0 ){
						SetWindowTitle( wi, 0.0);
					}
					xrange= wi->XOppX- wi->XOrgX;
					yrange= wi->XOppY - wi->XOrgY;
					delta_x= (aspect* yrange - xrange);
					delta_y= (xrange/ aspect - yrange);
				}
				else{
					delta_x= delta_y= 0;
				}
			}
			else{
				delta_x= delta_y= 0;
			}
			n+= 1;
			if( Handle_An_Event( wi->event_level, 1, "ZoomWindow_PS_Size()", wi->window, 
					/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
				)
			){
/* 				wi->event_level--;	*/
/* 				return(NULL);	*/
			}
			else if( wi->halt || ascanf_escape ){
/* 				return( NULL );	*/
			}
			wi->raw_display= rd;
			wi->raw_once= ro;
			wi->raw_val= rv;
		}
		if( delta_x== 0 && delta_y== 0 ){
			TransformCompute( wi, False );
			SetWindowTitle( wi, 0.0);
			xrange= wi->XOppX- wi->XOrgX;
			yrange= wi->XOppY - wi->XOrgY;
		}
		win_attr.width= wi->dev_info.area_w;
		wi->dev_info.old_area_w= wi->dev_info.area_w= w;
		win_attr.height= wi->dev_info.area_h;
		wi->dev_info.old_area_h= wi->dev_info.area_h= h;
		wi->dev_info.resized= 0;
		XMoveResizeWindow( disp, wi->window, 0, WM_TBAR, win_attr.width, win_attr.height );
/* 		XMoveResizeWindow( disp, wi->window, 0, 0, win_attr.width, win_attr.height );	*/
		hints.flags = USSize|USPosition;
		hints.x= 0; hints.y= WM_TBAR;
		hints.width = win_attr.width; hints.height = win_attr.height;
		XSetNormalHints(disp, wi->window, &hints);
		resized= 3;
		if( debugFlag ){
			fprintf( StdErr, "ZoomWindow_PS_Size(asp %s): %dx%d+%d+%d ; aspect= %s, %d tries\n",
				d2str( aspect, "%.15g", NULL),
				win_attr.width, win_attr.height,
				0, WM_TBAR,
				d2str( (double)xrange/ (double) yrange, "%.15g", NULL),
				n
			);
		}
	}
	else if( !preserve_screen_aspect ){
	  double hcentimeters, wcentimeters, plot_area_x, plot_area_y;
	  double DHeight= (double) XG_DisplayHeight(disp, screen, wi), DHeightMM= (double) XG_DisplayHeightMM(disp, screen);
	  extern int screen;

		{ double WM_TBARMM= WM_TBAR* DHeightMM/ DHeight;
		  int dnr= (wi->current_device>= 0)? wi->current_device : 0;
			DHeight-= WM_TBAR;
			DHeightMM-= WM_TBARMM;
			plot_area_x= (double) (wi->dev_info.area_w- wi->dev_info.bdr_pad)/ (double) wi->XOppX;
			plot_area_y= fabs((double) (wi->dev_info.area_h- wi->dev_info.bdr_pad)/ (double) (wi->dev_info.area_h- wi->XOrgY));
			wi->dev_info.resized= 0;
			hcentimeters= fabs( 10* wi->hard_devices[dnr].dev_max_height);
			wcentimeters= fabs( 10* wi->hard_devices[dnr].dev_max_width);
		}
#ifdef OLD_LANDSCAPE
		if( hcentimeters> wcentimeters ){
			win_attr.height= RND( plot_area_y* wcentimeters * XG_DisplayWidth(disp,screen, wi)/ XG_DisplayWidthMM(disp,screen) );
			win_attr.width= RND( plot_area_x* hcentimeters * DHeight/ DHeightMM );
		}
		else
#endif
		{
			win_attr.width= RND( wcentimeters * XG_DisplayWidth(disp,screen, wi)/ XG_DisplayWidthMM(disp,screen) );
			win_attr.height= RND( hcentimeters * DHeight/ DHeightMM );
		}
		XResizeWindow( disp, wi->window, win_attr.width, win_attr.height );
		hints.flags = USSize;
		hints.width = win_attr.width; hints.height = win_attr.height;
		XSetNormalHints(disp, wi->window, &hints);
		resized= 2;
		if( debugFlag ){
			fprintf( StdErr, "ZoomWindow_PS_Size(): %g x %g cm ; plot_area= %g x %g => window %dx%d==%gx%gcm\n",
				0.1* wcentimeters, 0.1* hcentimeters, plot_area_x, plot_area_y,
				win_attr.width, win_attr.height,
				0.1* win_attr.width* XG_DisplayWidthMM(disp,screen)/ XG_DisplayWidth(disp,screen, wi),
				0.1* win_attr.height* XG_DisplayHeightMM(disp,screen)/ XG_DisplayHeight(disp,screen, wi)
			);
		}
	}
	else{
		Boing(5);
	}
	if( resized ){
	  int x, y;
	  Window dummy;
/* 		XG_XSync( disp, False);	*/
		XGetWindowAttributes(disp, wi->window, &win_attr );
		  /* The manpage says XGetWindowAttributes returns the window's (x,y) relative
		   \ to the origin of its parent (i.c. the RootWindow). So why does it not (always?),
		   \ and do I need XTranslateCoordinates()?
		   \ And how do I figure out the y-coordinate to pass to the WM to get the window *including*
		   \ its frame where I want it (i.e. not the upper-left corner of the window inside the frame..)
		   */
		XTranslateCoordinates( disp, wi->window, RootWindow(disp, screen), 0, 0,
			&win_attr.x, &win_attr.y, &dummy
		);
		CLIP_EXPR( x, win_attr.x, 0, XG_DisplayWidth(disp, screen, wi) - win_attr.width );
/* 		CLIP_EXPR( y, win_attr.y, 0, XG_DisplayHeight(disp, screen, wi) - win_attr.height );	*/
		CLIP_EXPR( y, win_attr.y, WM_TBAR, XG_DisplayHeight(disp, screen, wi) - win_attr.height );
		if( win_attr.x!= x || win_attr.y!= y ){
			XMoveWindow( disp, wi->window, x, y );
			hints.flags = USPosition;
			hints.x= x; hints.y= y;
			XSetNormalHints(disp, wi->window, &hints);
			if( debugFlag ){
				fprintf( StdErr, "ZoomWindow_PS_Size(): moving window from %d,%d to %d,%d to keep it onscreen\n",
					win_attr.x, win_attr.y, x, y
				);
			}
		}
		AdaptWindowSize( wi, wi->window, win_attr.width, win_attr.height );
		if( resized== 2 ){
			wi->dev_info.resized= 1;
		}
		else if( resized== 3 ){
			wi->dev_info.resized= 2;
		}
		else{
			wi->dev_info.resized= 0;
		}
		if( raise ){
			XRaiseWindow( disp, wi->window);
		}
		wi->redraw= 1;
	}
	Synchro_State= !sync;
	X_Synchro(wi);
	return(wi);
}

int Num_Mapped()
{ LocalWindows *WL= WindowList;
  int n= 0;
	while( WL ){
		if( WL->wi && WL->wi->mapped ){
			n+= 1;
		}
		WL= WL->next;
	}
	return(n);
}

/* This routine tiles the current windows, putting the newest (direction=='d')
 \ or the oldest (direction=='u') in the lower right corner. Windows are resized
 \ such that an array as close to nxn as possible (with more windows horizontally
 \ if necessary) results.
 */
LocalWin *TileWindows(int direction, int horvert )
{  LocalWindows *l;
   LocalWin *wi= (WindowList)? WindowList->wi : NULL;
   int e, nx, ny, sx, sy, X, Y, i, j;
   XSizeHints hints;
   int NW= Num_Mapped();

	X= XG_DisplayWidth(disp, screen, wi);
	Y= XG_DisplayHeight(disp, screen, wi)/* - WM_TBAR	*/;
	if( X> Y ){
		nx= (horvert>0)? 1 : (int) ssceil( sqrt((double)NW) );
		ny= (horvert<0)? 1 : (int) ssceil( (double) NW/ (double) nx );
	}
	else{
		ny= (horvert>0)? 1 : (int) ssceil( sqrt((double)NW) );
		nx= (horvert<0)? 1 : (int) ssceil( (double) NW/ (double) ny );
	}
	sx= (int) ( (double) (X= XG_DisplayWidth(disp, screen, wi)) / (double) nx );
	sy= (int) ( (double) (XG_DisplayHeight(disp, screen, wi)/* - WM_TBAR	*/) / (double) ny );
	Y= XG_DisplayHeight(disp, screen, wi);
	if( debugFlag ){
		fprintf( StdErr, "TileWindows(): %d windows on %dx%d screen => %dx%d, size %dx%d\n",
			NW, X, Y,
			nx, ny, sx, sy
		);
	}
	if( direction== 'd' ){
		l= WindowList;
		for( i= 0; i< ny && l; ){
		  extern int ui_NumHeads;
		  int scr_x= 0, scr_y= 0;

			Y-= sy;
			X= XG_DisplayWidth( disp, screen, wi);
			for( j= 0; j< nx && l; ){
				wi= l->wi;
				if( ui_NumHeads> 1 && wi ){
					ux11_multihead_DisplayWidth( disp, screen,
						(wi->dev_info.area_x+wi->dev_info.area_w/2), (wi->dev_info.area_y+wi->dev_info.area_h/2),
						&scr_x, &scr_y, NULL );
				}
				if( wi->delete_it!= -1 && wi->mapped ){
					X-= sx;
					XMoveResizeWindow( disp, wi->window, scr_x+X, scr_y+Y, sx, sy );
					hints.flags = USSize|USPosition;
					hints.x= scr_x+X; hints.y= scr_x+Y;
					hints.width = sx; hints.height = sy;
					XSetNormalHints(disp, wi->window, &hints);
					XGetNormalHints(disp, wi->window, &hints);
/* 
					wi->dev_info.area_w= hints.width;
					wi->dev_info.area_h= hints.height;
 */
					AdaptWindowSize( wi, wi->window, sx, sy);
					XMapRaised( disp, wi->window);
					if( debugFlag ){
						fprintf( StdErr, "TileWindows(): window \"%s\" -g %dx%d+%d+%d\n",
							XGFetchName(wi),
							sx, sy, X, Y
						);
					}
					e= 0;
					wi->redraw= 1;
					while( Handle_An_Event( wi->event_level, 1, "TileWindows()", wi->window, 
							ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask
						)
					){
						e+= 1;
					}
					if( !e || wi->redraw ){
						RedrawNow( wi );
					}
					j+= 1;
				}
				l= l->next;
				if( j== nx ){
					i+= 1;
				}
			}
		}
	}
	else if( direction== 'u' ){
		l= WindowListTail;
		for( i= 0; i< ny && l; ){
		  extern int ui_NumHeads;
		  int scr_x= 0, scr_y= 0;

			Y-= sy;
			X= XG_DisplayWidth( disp, screen, wi);
			for( j= 0; j< nx && l; ){
				wi= l->wi;
				if( ui_NumHeads> 1 && wi ){
					ux11_multihead_DisplayWidth( disp, screen,
						(wi->dev_info.area_x+wi->dev_info.area_w/2), (wi->dev_info.area_y+wi->dev_info.area_h/2),
						&scr_x, &scr_y, NULL );
				}
				if( wi->delete_it!= -1 && wi->mapped ){
					X-= sx;
					XMoveResizeWindow( disp, wi->window, scr_x+X, scr_y+Y, sx, sy );
					hints.flags = USSize|USPosition;
					hints.x= scr_x+X; hints.y= scr_y+Y;
					hints.width = sx; hints.height = sy;
					XSetNormalHints(disp, wi->window, &hints);
					XGetNormalHints(disp, wi->window, &hints);
/* 
					wi->dev_info.area_w= hints.width;
					wi->dev_info.area_h= hints.height;
 */
					AdaptWindowSize( wi, wi->window, sx, sy);
					XMapRaised( disp, wi->window);
					if( debugFlag ){
						fprintf( StdErr, "TileWindows(): window \"%s\" -g %dx%d+%d+%d\n",
							XGFetchName(wi),
							sx, sy, X, Y
						);
					}
					e= 0;
					wi->redraw= 1;
					while( Handle_An_Event( wi->event_level, 1, "TileWindows()", wi->window, 
							ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask
						)
					){
						e+= 1;
					}
					if( !e || wi->redraw ){
						RedrawNow( wi );
					}
					j+= 1;
				}
				l= l->prev;
				if( j== nx ){
					i+= 1;
				}
			}
		}
	}
	return( wi);
}

/* Open a new window for every file, displaying one file per window	*/
LocalWin *Tile_Files(LocalWin *wi, Boolean rescale)
{ int new= 0, fn, grps, fxb, fyb;
  LocalWin *next, *ret;
  LocalWindows *l;

	  /* Initialise the argument-window to display the first file:	*/
	wi->plot_only_file= -1;
	cycle_plot_only_file( wi, 1 );
	files_and_groups( wi, &fn, &grps );
	  /* As long as we manage to create new windows AND we haven't ended up
	   \ at the last file:
	   */
	while( wi && wi->plot_only_file< fn-1 ){
	  Cursor curs= zoomCursor;
	  extern int HandleMouse();
		  /* Make a duplicate of the current window:	*/
		 HandleMouse(Window_Name,
					  NULL,
					  wi, &next, &curs
		);
		if( next ){
			new+= 1;
			  /* If we succeeded, display the next file in it:	*/
			cycle_plot_only_file( next, 1 );
			  /* and update the "loop-window" & its info on the number
			   \ of files and groups.
			   */
			wi= next;
			files_and_groups( wi, &fn, &grps );
		}
	}
	if( new ){
		  /* If we have created new windows, tile them..	*/
		ret= TileWindows( 'd', 0 );
		l= WindowList;
		if( rescale ){
			  /* And scale their boundaries, so that they show their contents..	*/
			while( l ){
				wi= l->wi;
				fxb= wi->fit_xbounds;
				fyb= wi->fit_ybounds;
				if( !(fxb && fyb) ){
					wi->fit_xbounds= True;
					wi->fit_ybounds= True;
					RedrawNow( wi );
					wi->fit_xbounds= fxb;
					wi->fit_ybounds= fyb;
				}
				l= l->next;
			}
		}
		return( ret );
	}
	else{
		return( wi );
	}
}

/* Open a new window for every group, displaying one file per window	*/
LocalWin *Tile_Groups(LocalWin *wi, Boolean rescale)
{ int new= 0, fn, grps, fxb, fyb;
  LocalWin *next, *ret;
  LocalWindows *l;

	  /* Initialise the argument-window to display the first group:	*/
	wi->plot_only_group= -1;
	cycle_plot_only_group( wi, 1 );
	files_and_groups( wi, &fn, &grps );
	  /* As long as we manage to create new windows AND we haven't ended up
	   \ at the last group:
	   */
	while( wi && wi->plot_only_group< grps-1 ){
	  Cursor curs= zoomCursor;
		  /* Make a duplicate of the current window:	*/
		HandleMouse(Window_Name,
					  NULL,
					  wi, &next, &curs
		);
		if( next ){
			new+= 1;
			  /* If we succeeded, display the next group in it:	*/
			cycle_plot_only_group( next, 1 );
			  /* and update the "loop-window" & its info on the number
			   \ of files and groups.
			   */
			wi= next;
			files_and_groups( wi, &fn, &grps );
		}
	}
	if( new ){
		  /* If we have created new windows, tile them..	*/
		ret= TileWindows( 'd', 0 );
		l= WindowList;
		if( rescale ){
			  /* And scale their boundaries, so that they show their contents..	*/
			while( l ){
				wi= l->wi;
				fxb= wi->fit_xbounds;
				fyb= wi->fit_ybounds;
				if( !(fxb && fyb) ){
					wi->fit_xbounds= True;
					wi->fit_ybounds= True;
					RedrawNow( wi );
					wi->fit_xbounds= fxb;
					wi->fit_ybounds= fyb;
				}
				l= l->next;
			}
		}
		return( ret );
	}
	else{
		return( wi );
	}
}

char *XLABEL_string= NULL, *YLABEL_string= NULL;
int XLABEL_stringlen= 0, YLABEL_stringlen= 0;

#ifdef __GNUC__
inline
#endif
char *Parse_XYLABEL( LocalWin *wi, char *labelstring, char **target, int *target_len )
{ char *parsed_end, *ntt= ParseTitlestringOpcodes( wi, wi->first_drawn, labelstring, &parsed_end );
	if( ntt ){
		if( ntt[0]== '`' && parsed_end[-1]== '`' ){
			if( strlen(ntt)-2> MAXBUFSIZE ){
				xtb_error_box( wi->window, "parsed X/Y label is too large to store: keeping the original!", "Error!" );
				goto use_parsed;
			}
			else{
				parsed_end[-1]= '\0';
				strcpy( labelstring, &ntt[1] );
				goto not_parsed;
			}
		}
		else{
use_parsed:;
			xfree( (*target) );
			*target= ntt;
			*target_len= strlen(ntt);
		}
	}
	else{
not_parsed:;
		if( !(*target) || *target_len< strlen(labelstring) ){
			xfree( (*target) );
			*target= strdup( labelstring );
			*target_len= strlen( (*target) );
		}
		else{
			strcpy( *target, labelstring );
		}
	}
	return( *target );
}

#ifdef __GNUC__
inline
#endif
char *raw_XLABEL(LocalWin *wi )
{
	if( ((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) ){
		return( &(wi->XUnits[0]) );
	}
	else{
		return( &(wi->tr_XUnits[0]) );
	}
}

#ifdef __GNUC__
inline
#endif
char *raw_YLABEL(LocalWin *wi )
{
	if( ((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) ){
		return( &(wi->YUnits[0]) );
	}
	else{
		return( &(wi->tr_YUnits[0]) );
	}
}

char *XLABEL(LocalWin *wi )
{
	return( Parse_XYLABEL( wi, raw_XLABEL(wi), &XLABEL_string, &XLABEL_stringlen ) );
}

char *YLABEL(LocalWin *wi )
{
	return( Parse_XYLABEL( wi, raw_YLABEL(wi), &YLABEL_string, &YLABEL_stringlen ) );
}

int RawDisplay( LocalWin *wi, int value )
{
	if( wi ){
	 int prev= wi->raw_display;
	 static char *expr= NULL, *which, active= 0;
	 struct Compiled_Form *C_expr= NULL;
		wi->raw_display= value;
		if( !active ){
			if( value && !prev && wi->process.enter_raw_after_len ){
				which= "*ENTER_RAW_AFTER*";
				expr= wi->process.enter_raw_after;
				C_expr= wi->process.C_enter_raw_after;
			}
			else if( !value && prev && wi->process.leave_raw_after_len ){
				which= "*LEAVE_RAW_AFTER*";
				expr= wi->process.leave_raw_after;
				C_expr= wi->process.C_leave_raw_after;
			}
			if( expr && C_expr ){
			  int n= param_scratch_len;
			  LocalWin *AW= ActiveWin;
			  Window aw= ascanf_window;
				active= 1;
				*ascanf_self_value= 0.0;
				*ascanf_current_value= 0.0;
				*ascanf_counter= (*ascanf_Counter)= 0.0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				*ascanf_setNumber= 0;
				*ascanf_numPoints= 0;
				ActiveWin= wi;
				ascanf_window= wi->window;
				if( ascanf_verbose ){
					fprintf( StdErr, "RawDisplay(): %s: %s", which, expr );
					fflush( StdErr );
				}
				TitleMessage( wi, which );
				TBARprogress_header= which;
				ascanf_arg_error= 0;
				compiled_fascanf( &n, expr, param_scratch, NULL, NULL, NULL, &C_expr );
				TBARprogress_header= NULL;
				ActiveWin= AW;
				ascanf_window= aw;
				active= 0;
			}
		}
		else{
			fprintf( StdErr, "RawDisplay(%d): already active: %s: %s\n", value, which, expr );
		}
		return( wi->raw_display );
	}
	else{
		return(0);
	}
}

#ifdef TOOLBOX
int PrintWindow( Window win, LocalWin *wi)
/*
 * This routine posts a dialog asking about the hardcopy
 * options desired.  If the user hits `OK',  the hard
 * copy is performed.
 */
{  char *title= "D'" ;
   char Title[512], *xsemi, *ysemi;
   ALLOCA( inside_title, char, 2*LMAXBUFSIZE+10, inside_title_len);
   Boolean remap= True;
   int ret;
	  /* If the Dialog already belonged to some window, unmap it. Unmapping (by CloseHO_Dialog)
	   \ unsets the <thePrintWindow> and <thePrintWin_Info> variables.
	   */
	if( thePrintWin_Info && thePrintWin_Info->HO_Dialog ){
		  /* This shouldn't be done when the current window owns the dialog!	*/
		if( thePrintWin_Info!= wi ){
			if( wi->pw_placing!= PW_PARENT ){
				wi->pw_placing= PW_CENTRE_ON;
				{ XWindowAttributes winInfo;
				  XSizeHints hints;
				  Window dummy;
					XGetWindowAttributes(disp, thePrintWin_Info->HO_Dialog->win, &winInfo);
					XTranslateCoordinates( disp, thePrintWin_Info->HO_Dialog->win, RootWindow(disp, screen),
							  0, 0, &hints.x, &hints.y, &dummy
					);
					hints.y-= WM_TBAR;
					pw_centre_on_X= hints.x+ thePrintWin_Info->HO_Dialog->width/2;
					pw_centre_on_Y= hints.y+ thePrintWin_Info->HO_Dialog->height/2;
				}
			}
			CloseHO_Dialog( thePrintWin_Info->HO_Dialog );
		}
	}
	  /* Close the dialog when a NULL LocalWin pointer is given (win=0 can be
	   \ the RootWindow!)
	   */
	if( !wi ){
		return(1);
	}
	if( wi->HO_Dialog && wi->HO_Dialog->mapped ){
		if( wi->HO_Dialog->parent== win ){
			XMapRaised( disp, wi->HO_Dialog->win );
			  /* If do_gsTextWidth_Batch, always call ho_dialog(). Otherwise, only when
			   \ the current window doesn't yet own the dialog.
			   */
			remap= (*do_gsTextWidth_Batch)? True : False;
		}
		else{
		  /* Should never get here.	*/
			xtb_error_box( wi->window, "Hardcopy Dialog is already attached to another window\n(Close it first)\n", "Notice" );
			return(1);
		}
	}
	thePrintWindow= win;
	thePrintWin_Info= wi;
	if( !remap ){
		return(0);
	}
	sprintf( Title, "%s%s", title, XGFetchName( wi ) );
	XGFetchName( NULL );
	Title[sizeof(Title)-1]= '\0';
	if( (xsemi= index( XLABEL(wi), ';')) )
		*xsemi= 0;
	if( (ysemi= index( YLABEL(wi), ';')) )
		*ysemi= 0;
	sprintf( inside_title, "\"%s\" vs. \"%s\"; w#%02d.%02d.%02d",
		YLABEL(wi), XLABEL(wi), wi->parent_number, wi->pwindow_number, wi->window_number
	);
	if( xsemi)
		*xsemi= ';';
	if( ysemi)
		*ysemi= ';';
	ret= ho_dialog(win, wi, Title, (char *) wi, Title, inside_title);
	xtb_XSync( disp, False );
	GCA();
	return(ret);
}

int DoSettings( Window win, LocalWin *wi)
/*
 * This routine posts a dialog asking about the settings
 * options desired.
 */
{  char *title= "S'" ;
   char Title[512], *xsemi, *ysemi;
   ALLOCA( inside_title, char, 2*LMAXBUFSIZE+10, inside_title_len);
   Boolean remap= True;
   int ret;
	  /* If the Dialog already belonged to some window, unmap it. Unmapping (by CloseHO_Dialog)
	   \ unsets the <thePrintWindow> and <thePrintWin_Info> variables.
	   */
	if( theSettingsWin_Info && theSettingsWin_Info->SD_Dialog ){
		  /* This shouldn't be done when the current window owns the dialog!	*/
		if( theSettingsWin_Info!= wi ){
			if( wi->pw_placing!= PW_PARENT ){
				wi->pw_placing= PW_CENTRE_ON;
				{ XWindowAttributes winInfo;
				  XSizeHints hints;
				  Window dummy;
					XGetWindowAttributes(disp, theSettingsWin_Info->SD_Dialog->win, &winInfo);
					XTranslateCoordinates( disp, theSettingsWin_Info->SD_Dialog->win, RootWindow(disp, screen),
							  0, 0, &hints.x, &hints.y, &dummy
					);
					hints.y-= WM_TBAR;
					pw_centre_on_X= hints.x+ theSettingsWin_Info->SD_Dialog->width/2;
					pw_centre_on_Y= hints.y+ theSettingsWin_Info->SD_Dialog->height/2;
				}
			}
			CloseSD_Dialog( theSettingsWin_Info->SD_Dialog );
		}
	}
	  /* Close the dialog when a NULL LocalWin pointer is given (win=0 can be
	   \ the RootWindow!)
	   */
	if( !wi ){
		return(1);
	}
	if( wi->SD_Dialog /* && wi->SD_Dialog->mapped */ ){
		if( wi->SD_Dialog->parent== win ){
			XMapRaised( disp, wi->SD_Dialog->win );
			remap= False;
		}
		else{
			xtb_error_box( wi->window, "Settings Dialog is already attached to another window\n(Close it first)\n", "Notice" );
			return(1);
		}
	}
	theSettingsWindow= win;
	theSettingsWin_Info= wi;
	if( !remap ){
		return(0);
	}
	sprintf( Title, "%s%s", title, XGFetchName( wi ) );
	XGFetchName( NULL );
	Title[sizeof(Title)-1]= '\0';
	if( (xsemi= index( XLABEL(wi), ';')) )
		*xsemi= 0;
	if( (ysemi= index( YLABEL(wi), ';')) )
		*ysemi= 0;
	sprintf( inside_title, "\"%s\" vs. \"%s\"; w#%02d.%02d.%02d",
		YLABEL(wi), XLABEL(wi), wi->parent_number, wi->pwindow_number, wi->window_number
	);
	if( xsemi)
		*xsemi= ';';
	if( ysemi)
		*ysemi= ';';
	ret= settings_dialog(win, wi, Window_Name, (char *) wi, Title, inside_title);
	xtb_XSync( disp, False );
	GCA();
	return( ret );
}
#endif

#define TRANX(xval) \
(((double) ((xval) - wi->XOrgX)) * wi->XUnitsPerPixel + wi->UsrOrgX)

#define TRANY(yval) \
(wi->UsrOppY - (((double) ((yval) - wi->XOrgY)) * wi->YUnitsPerPixel))

extern char *ParsedColourName;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

extern XColor GetThisColor;

/* Default line styles */
extern char *defStyle[MAXATTR] ;

/* Default color names */
extern char *defColors[MAXATTR] ;

Pixmap _XCreateBitmapFromData(Display *dis, Drawable win, char *mark_bits, unsigned int mark_w, unsigned int mark_h)
{  Pixmap Return;
   static int count= 0;

	if( debugFlag){
		if( count== 1)
			fputs( "Creating mark ", StdErr);
		fprintf( StdErr, "#%d ", count);
	}
	Return= XCreateBitmapFromData( dis, win, mark_bits, mark_w, mark_h);
/* 	Return= XCreatePixmapFromBitmapData( dis, win, mark_bits, mark_w, mark_h, normPixel, bgPixel, depth );	*/
	if( debugFlag ){
		if( Return== None){
			fputs( "(failed) ", StdErr);
		}
		else{
			fprintf( StdErr, "(0x%lx) ", Return);
		}
		fflush( StdErr);
	}
	if( count== MAXATTR && debugFlag )
		fputc( '\n', StdErr);
	count++;
	return( Return);
}

/* stat( filename, &stats);	*/

char *FontName( XGFontStruct *f)
{ static char *unknown= "???";
	if( f== &axisFont){
		return( "axisFont");
	}
	else if( f== &dialogFont){
		return( "dialogFont");
	}
	else if( f== &dialog_greekFont){
		return( "dialog_greekFont");
	}
	else if( f== &legendFont){
		return( "legendFont");
	}
	else if( f== &labelFont){
		return( "labelFont");
	}
	else if( f== &legend_greekFont){
		return( "legend_greekFont");
	}
	else if( f== &label_greekFont){
		return( "label_greekFont");
	}
	else if( f== &title_greekFont){
		return( "title_greekFont");
	}
	else if( f== &axis_greekFont){
		return( "axis_greekFont");
	}
	else if( f== &titleFont){
		return( "titleFont");
	}
	else if( f== &markFont){
		return( "markFont");
	}
	else if( f== &cursorFont){
		return( "cursorFont");
	}
	else if( f== &fbFont){
		return( "fbFont");
	}
	else if( f== &fb_greekFont){
		return( "fb_greekFont");
	}
	else if( *f->use ){
		return( f->use );
	}
	else{
		return(unknown);
	}
}

char RememberedPrintFont[256]= "";
double RememberedPrintSize= -1;

char *Platform_FileName_Adapt(char *name)
{ char *c= name;
	while( c && *c ){
		if( *c== ':' ){
			*c= '_';
		}
		c++;
	}
	return(name);
}

char *PrefsDir= NULL;

 /* Check for the presence of the preferences directory, and create it if necessary.
  \ Should be called from main(), as one of the first calls (it can exit, and does so
  \ without calling any cleanup routines)!
  */
char *CheckPrefsDir( char *name )
{ static char called= 0;
  FILE *fp;
	if( !called && name && *name ){
	  char *pd = getenv("XGPREFSDIR");
		if( pd ){
			pd = strdup(pd);
		}
		else{
#if defined(__APPLE_CC__) || defined(__MACH__)
			pd= concat( getenv("HOME"), "/Library/", name, NULL );
#else
			pd= concat( getenv("HOME"), "/.Preferences/.", name, NULL );
#endif
		}
		if( pd ){
		  struct stat stats;
			xfree( PrefsDir );
			PrefsDir= pd;
			if( stat( pd, &stats) ){
				fprintf( StdErr, "Creating new preferences directory \"%s\"\n", pd );
			}
			else{
				return PrefsDir;
			}
		}
		else{
			if( !getenv( "HOME" ) ){
				fprintf( StdErr, "You should set the HOME environmental variable to the name of your home directory!\n" );
				exit(-1);
			}
			fprintf( StdErr,
				"There was some problem in constructing the preferences directory name (%s) - will try to create it anyway\n"
				" Proceed with toes crossed!\n", serror()
			);
		}
#if defined(__APPLE_CC__) || defined(__MACH__)
		if( (fp= popen( "sh", "w")) )
#else
		if( (fp= popen( "sh", "wb")) )
#endif
		{
			fprintf( fp, "XGPREFSDIR=%s ; export XGPREFSDIR\n", PrefsDir );
// 			fprintf( fp, "cd `dirname ${XGPREFSDIR}`\n" );

// #if defined(__APPLE_CC__) || defined(__MACH__)
// 			fprintf( fp, "XGPREFSDIR=\"Library\"\n" );
// 			fprintf( fp, "if [ -d ${XGPREFSDIR}/%s ] ;then\n", name );
// #else
// 			fprintf( fp, "XGPREFSDIR=\".Preferences\"\n" );
// 			fprintf( fp, "if [ -d ${XGPREFSDIR}/.%s ] ;then\n", name );
// #endif
// 			fprintf( fp, "     exit 0\n" );
// 			fprintf( fp, "fi\n" );

			fprintf( fp, "set -x\n" );

			fprintf( fp, "if [ ! -d ${XGPREFSDIR} ] ;then\n" );
			fprintf( fp, "     mkdir -p ${XGPREFSDIR}\n" );
			fprintf( fp, "fi\n" );

			fprintf( fp, "if [ ! -d ${XGPREFSDIR}/../.xtb_default_fonts ] ;then\n" );
			fprintf( fp, "     mkdir ${XGPREFSDIR}/../.xtb_default_fonts\n" );
			fprintf( fp, "fi\n" );

			pclose( fp );

			called= 1;
		}
		else{
			fprintf( StdErr,
				"Warning: can't open connection to the shell sh to verify/establish a preferences directory! (%s)\n"
				" You're likely to run into problems (soon) hereafter!\n", serror()
			);
		}
	}
	if( !PrefsDir ){
		called= False;
	}
	return( PrefsDir );
}

char *DisplayDirectory( char *target, int len )
{ static char buf[MAXPATHLEN];
  char *home= getenv( "HOME");
	if( !PrefsDir && !CheckPrefsDir("xgraph") ){
		return(NULL);
	}
	if( !target ){
		target= buf;
		len= sizeof(buf);
	}
	if( !home ){
		if( debugFlag)
			fprintf( StdErr, "Can't find home directory (set HOME variable)\n");
		return( NULL );
	}
	if( !disp ){
		if( debugFlag ){
			fprintf( StdErr, "Error: DISPLAY not yet opened -- can't determine DisplayDirectory yet!\n" );
		}
		return( NULL );
	}
/* 	snprintf( target, len, "%s/.Preferences/.xgraph/%s", home, DisplayString(disp) );	*/
	snprintf( target, len, "%s/%s", PrefsDir, DisplayString(disp) );
#if defined(__CYGWIN__)
	Platform_FileName_Adapt( target );
#endif
	return( target );
}

int RememberedFont( Display *disp, XGFontStruct *font, char **rfont_name)
{  char name[MAXPATHLEN];
   static char font_name[256];
   char ps_data[256];
   XFontStruct *tempFont;
   FILE *fp;
   struct stat stats;

	*rfont_name= NULL;
	if( debugFlag ){
		fprintf( StdErr, "Reading remembered \"%s\": ",
			FontName( font)
		);
	}
	if( !DisplayDirectory( name, sizeof(name) ) ){
		return( 0);
	}
	sprintf( &name[strlen(name)], "/%s", FontName( font) );
#if defined(__CYGWIN__)
	Platform_FileName_Adapt( name );
#endif
	if( stat( name, &stats) ){
		if( debugFlag ){
			perror( name);
		}
		sprintf( name, "%s/../%s/%s/%s", PrefsDir, ".xtb_default_fonts", DisplayString(disp), FontName( font) );
#if defined(__CYGWIN__)
		Platform_FileName_Adapt( name );
#endif
		if( stat( name, &stats) ){
			if( debugFlag ){
				perror( name);
			}
			return(0);
		}
	}
	if( (fp= fopen( name, "r")) ){
	  int ret= 1;
		fgets( font_name, 255, fp);
		if( font_name[strlen(font_name)-1]== '\n' )
			font_name[strlen(font_name)-1]= '\0';
		font->font= NULL;
		if( (tempFont= XLoadQueryFont( disp, font_name)) ){
			font->font= tempFont;
			strncpy( font->name, font_name, 127);
			if( debugFlag){
				fprintf( StdErr, "%s : %s (%s)\n", FontName( font), font_name, name);
			}
			*rfont_name= font_name;
		}
		else{
			if( debugFlag)
				fprintf( StdErr, "Can't load font '%s' for %s\n", font_name, FontName(font) );
/* 			return(0);	*/
			ret= 0;
		}
		RememberedPrintFont[0]= '\0';
		RememberedPrintSize= -1;
		while( fgets( ps_data, 255, fp) && !feof(fp) && !ferror(fp) ){
			if( strncmp( ps_data, "*PRINT*", 7)== 0 ){
				strncpy( RememberedPrintFont, cleanup(&ps_data[7]), 255 );
				RememberedPrintFont[255]= '\0';
			}
			else if( strncmp( ps_data, "*SIZE*", 6)== 0 ){
			  double size;
				if( sscanf( cleanup(&ps_data[6]), "%lf", &size ) ){
					RememberedPrintSize= size;
				}
			}
		}
		fclose( fp);
		return( ret);
	}
	else
		perror( name);
	return(0);
}

int RememberFont( Display *disp, XGFontStruct *font, char *font_name)
{  char name[256], *home= getenv( "HOME");
   FILE *fp;

	if( debugFlag ){
		fprintf( StdErr, "Remembering '%s' for \"%s\": ",
			font_name, FontName( font)
		);
	}
	if( !DisplayDirectory( name, sizeof(name) ) ){
		return( 0);
	}
	if( mkdir( name, 0744) ){
		if( errno!= EEXIST){
			if( debugFlag){
				perror( name);
			}
			sprintf( name, "%s/../%s", PrefsDir, ".xtb_default_fonts" );
#if defined(__CYGWIN__)
			Platform_FileName_Adapt( name );
#endif
			if( mkdir( name, 0744) ){
				if( errno!= EEXIST){
					if( debugFlag){
						perror( name);
					}
					return(0);
				}
			}
		}
	}
/* 	strcat( name, "/"); strcat( name, DisplayString(disp) ); */
#if defined(__CYGWIN__)
	Platform_FileName_Adapt(name);
#endif
	if( mkdir( name, 0744) ){
		if( errno!= EEXIST){
			if( debugFlag)
				perror( name);
			return(0);
		}
	}
	strcat( name, "/"); strcat( name, FontName(font) );
	if( (fp= fopen( name, "wb")) ){
		fprintf( fp, "%s\n", font_name);
		if( RememberedPrintFont[0] ){
			fprintf( fp, "*PRINT*%s\n", RememberedPrintFont );
		}
		if( RememberedPrintSize> 0 ){
			fprintf( fp, "*SIZE*%g\n", RememberedPrintSize );
		}
		fclose( fp);
		if( debugFlag)
			fprintf( StdErr, "Remembered %s in '%s'\n", FontName(font), name);
		return( 1);
	}
	else
		perror( name);
	return(0);
}

char *def_str;
Pixel def_pixel;
int def_int;
XFontStruct *def_font;
double def_dbl;
char *ResourceTemplate= NULL;

#if defined(__CYGWIN__)
	int XG_GetDefault_Level= 1;
#else
	int XG_GetDefault_Level= 2;
#endif

  /* This routine retrieves a variable <name> from the X-resource
   \ database. If an environmental variable <name> exists, its
   \ value is used instead.
   */
char *XG_GetDefault( Display *disp, char *Prog_Name, char *name )
{ char *r= NULL;
	if( ResourceTemplate ){
		if( strncmp( ResourceTemplate, name, strlen(name) ) ){
			return(NULL);
		}
		else{
			if( (r= index( ResourceTemplate, ':')) ){
				r++;
				while( isspace(*r)){
					r++;
				}
				if( debugFlag ){
					fprintf( StdErr, "resource %s*%s:%s\n",
						Prog_Name, name, r
					);
					fflush( StdErr );
				}
				return(r);
			}
			else{
				return(NULL);
			}
		}
	}
	else if( XG_GetDefault_Level > 1 ){
		if( (r= getenv(name)) ){
			if( debugFlag ){
				fprintf( StdErr, "default %s*%s: using env.var %s=%s\n",
					Prog_Name, name, name, r
				);
				fflush( StdErr );
			}
			return(r);
		}
		else{
		  char dd[MAXPATHLEN];
		  FILE *fp;
			if( DisplayDirectory( dd, MAXPATHLEN ) ){
				sprintf( &dd[strlen(dd)], "/%s", name );
				if( (fp= fopen(dd, "r")) ){
				  static char buf[256];
					fgets( buf, 256, fp );
					buf[255]= '\0';
					if( buf[strlen(buf)-1]== '\n' ){
						buf[strlen(buf)-1]= '\0';
					}
					fclose(fp);
					if( debugFlag ){
						fprintf( StdErr, "default %s*%s: using 1st %d characters from %s=%s\n",
							Prog_Name, name, sizeof(buf), dd, buf
						);
						fflush( StdErr );
					}
					return( buf );
				}
			}
		}
	}
	if( disp && (r= XGetDefault( disp, Prog_Name, name )) ){
		if( debugFlag && debugLevel ){
			fprintf( StdErr, "default %s*%s:%s\n",
				Prog_Name, name, r
			);
			fflush( StdErr );
		}
	}
	else if( disp && debugFlag && debugLevel ){
		fprintf( StdErr, "default %s*%s not found in env or X-resources.\n",
			Prog_Name, name
		);
		fflush( StdErr );
	}
	return( r );
}

int rd_pix(char *name)
/* Result in def_pixel */
{
    if( (def_str = XG_GetDefault(disp, Prog_Name, name)) ){
		if( GetColor(def_str, &def_pixel ) ){
			return 1;
		}
		else{
			return(0);
		}
    }
	else{
		return 0;
    }
}

int rd_int(char *name)
/* Result in def_int */
{
    if( (def_str = XG_GetDefault(disp, Prog_Name, name)) ){
		if (sscanf(def_str, "%d", &def_int) == 1) {
			return 1;
		} else {
			fprintf(StdErr, "warning: could not read integer value for %s\n", name);
			return 0;
		}
    } else {
		return 0;
    }
}

int rd_str(char *name)
/* Result in def_str */
{
    if( (def_str = XG_GetDefault(disp, Prog_Name, name)) ){
		return 1;
    } else {
		return 0;
    }
}

int rd_font(char *name, char **font_name)
/* Result in def_font */
{
    if( name && (def_str = XG_GetDefault(disp, Prog_Name, name)) ){
		if( (def_font = XLoadQueryFont(disp, def_str)) ){
			*font_name= def_str;
			return 1;
		} else {
			fprintf(StdErr, "warning: could not load font for %s\n", name);
			return 0;
		}
    } else {
		return 0;
    }
}

int rd_flag(char *name)
/* Result in def_int */
{
    if( (def_str = XG_GetDefault(disp, Prog_Name, name)) ){
		def_int = (stricmp(def_str, "true") == 0) || (stricmp(def_str, "on") == 0) || (stricmp(def_str, "1") == 0);
		return 1;
    } else {
		return 0;
    }
}

int rd_dbl(char *name)
/* Result in def_dbl */
{
    if( (def_str = XG_GetDefault(disp, Prog_Name, name)) ){
		if (sscanf(def_str, "%lg", &def_dbl) == 1) {
			return 1;
		} else {
			fprintf(StdErr, "warning: could not read value of %s\n", name);
			return 0;
		}
    } else {
		return 0;
    }
}

/* The following routine reverses a colour. Black and white are reversed (whatever
 \ their colour actually is!!). Other pixel-values assume a monotonously increasing
 \ mapping from pixel value (index) to colour.
 */
Pixel ReversePixel(Pixel *pixValue)
{
    if (*pixValue == white_pixel)
      *pixValue = black_pixel;
    else if (*pixValue == black_pixel)
      *pixValue = white_pixel;
	else{
		CLIP_EXPR( *pixValue, (1 << depth)-1 - *pixValue, 0, (1 << depth)-1 );
	}
	return( *pixValue );
}

extern double win_aspect_precision;
extern double highlight_par[];
extern int highlight_mode, highlight_npars, ShiftLineEchoSegments;
extern char *GetFont( XGFontStruct *Font, char *resource_name, char *default_font, long size, int bold, int use_remembered );
double Progress_ThresholdTime= 0;
extern double Font_Width_Estimator, ps_l_offset, ps_b_offset;

int ReadDefaults()
/*
 * Reads X default values which override the hard-coded defaults
 * set up by InitSets.
 */
{  extern double overlap_legend_tune;
   extern double get_radix();
    char newname[100];
    int idx, XGDL = XG_GetDefault_Level;
	extern int ascanf_check_int, Print_Orientation;
	extern Boolean psSeg_disconnect_jumps;
	extern int SetIconName, UnmappedWindowTitle;
	extern int PAS_Lazy_Protection;

	XG_GetDefault_Level = 2;
	if( rd_int( "XG-GetDefaults" ) ){
		XG_GetDefault_Level = def_int;
	}
	else{
		XG_GetDefault_Level = XGDL;
	}

	if( rd_str( "X11-Greek-Template") ){
	  extern char *X11_greek_template;
		X11_greek_template= strdup(def_str);
	}
	if( rd_str( "PS-Greek-Font") ){
	  extern char *PS_greek_font;
		PS_greek_font= strdup(def_str);
	}

	if( !ResourceTemplate ){
		if( disp ){
			GetFont( &legendFont, "LegendFont", NULL, 4500, 1, 1);
			if( RememberedPrintFont[0] ){
				strncpy( hard_devices[PS_DEVICE].dev_legend_font, RememberedPrintFont, sizeof(hard_devices[idx].dev_legend_font)-1 );
			}
			if( RememberedPrintSize> 0 ){
				hard_devices[PS_DEVICE].dev_legend_size= RememberedPrintSize;
			}
			{ char *gf;
				 if( Update_greekFonts( 'LEGN') ){
					  RememberFont( disp, &legend_greekFont, legend_greekFont.name );
				 }
				 else if( !RememberedFont( disp, &legend_greekFont, &gf) ){
					 fprintf( stdout, "xgraph: can't get legend_greek font matching legendFont (%s)\n",
						 legendFont.name
					 );
				 }
			}
		}
		if( disp ){
			GetFont( &labelFont, "LabelFont", NULL, 5000, 1, 1);
			if( RememberedPrintFont[0] ){
				strncpy( hard_devices[PS_DEVICE].dev_label_font, RememberedPrintFont, sizeof(hard_devices[idx].dev_label_font)-1 );
			}
			if( RememberedPrintSize> 0 ){
				hard_devices[PS_DEVICE].dev_label_size= RememberedPrintSize;
			}
			{ char *gf;
				 if( Update_greekFonts( 'LABL') ){
					  RememberFont( disp, &label_greekFont, label_greekFont.name );
				 }
				 else if( !RememberedFont( disp, &label_greekFont, &gf) ){
					 fprintf( stdout, "xgraph: can't get label_greek font matching labelFont (%s)\n",
						 labelFont.name
					 );
				 }
			}
		}
		if( disp ){
			GetFont( &titleFont, "TitleFont", NULL, 4500, 1, 1);
			if( RememberedPrintFont[0] ){
				strncpy( hard_devices[PS_DEVICE].dev_title_font, RememberedPrintFont, sizeof(hard_devices[idx].dev_title_font)-1 );
			}
			if( RememberedPrintSize> 0 ){
				hard_devices[PS_DEVICE].dev_title_size= RememberedPrintSize;
			}
			{ char *gf;
				 if( Update_greekFonts( 'TITL') ){
					  RememberFont( disp, &title_greekFont, title_greekFont.name );
				 }
				 else if( !RememberedFont( disp, &title_greekFont, &gf) ){
					 fprintf( stdout, "xgraph: can't get title_greek font matching titleFont (%s)\n",
						 titleFont.name
					 );
				 }
			}
		}
		if( disp ){
			GetFont( &axisFont, "AxisFont", NULL, 4175, 0, 1);
			if( RememberedPrintFont[0] ){
				strncpy( hard_devices[PS_DEVICE].dev_axis_font, RememberedPrintFont, sizeof(hard_devices[idx].dev_axis_font)-1 );
			}
			if( RememberedPrintSize> 0 ){
				hard_devices[PS_DEVICE].dev_axis_size= RememberedPrintSize;
			}
			{ char *gf;
				 if( Update_greekFonts( 'AXIS') ){
					  RememberFont( disp, &axis_greekFont, axis_greekFont.name );
				 }
				 else if( !RememberedFont( disp, &axis_greekFont, &gf) ){
					 fprintf( stdout, "xgraph: can't get axis_greek font matching axisFont (%s)\n",
						 axisFont.name
					 );
				 }
			}
		}
		if( disp ){
			GetFont( &dialogFont, "DialogFont", NULL, 4175, -1, 1);
			{ char *gf;
				 if( Update_greekFonts( 'DIAL') ){
					  RememberFont( disp, &dialog_greekFont, dialog_greekFont.name );
				 }
				 else if( !RememberedFont( disp, &dialog_greekFont, &gf) ){
					 fprintf( stdout, "xgraph: can't get dialog_greek font matching dialogFont (%s)\n",
						 dialogFont.name
					 );
				 }
			}
		}
		if( disp ){
			GetFont( &fbFont, "FeedBackFont", NULL, 4000, -1, 1);
			{ char *gf;
				 if( Update_greekFonts( 'FDBK') ){
					  RememberFont( disp, &fb_greekFont, fb_greekFont.name );
				 }
				 else if( !RememberedFont( disp, &fb_greekFont, &gf) ){
					 fprintf( stdout, "xgraph: can't get fb_greek font matching fbFont (%s)\n",
						 fbFont.name
					 );
				 }
			}
		}
		if( disp && !GetFont( &cursorFont, "CursorFont", "cursor", 0, 0, 1) ){
			fprintf( stdout, "xgraph: can't get the standard 'cursor' font\n" );
		}
		if( disp && use_markFont ){
// 			GetFont( &markFont, "MarkFont", "spc08x08e", 3000, 0, 1);
			  // 20100401
			GetFont( &markFont, "MarkFont", "spc12x12e", 3000, 0, 1);
		}
		else{
			markFont= cursorFont;
		}
	}

	if( XG_GetDefault_Level <= 0 ){
		return(1);
	}

	sprintf( newname, "DisplayWidthMM-%d", screen );
	if( rd_int(newname) ){
	  extern int DisplayWidth_MM;
		DisplayWidth_MM= def_int;
	}
	else{
		sprintf( newname, "DisplayWidthMM" );
		if( rd_int(newname) ){
		  extern int DisplayWidth_MM;
			DisplayWidth_MM= def_int;
		}
	}
	sprintf( newname, "DisplayHeightMM-%d", screen );
	if( rd_int(newname) ){
	  extern int DisplayHeight_MM;
		DisplayHeight_MM= def_int;
	}
	else{
		sprintf( newname, "DisplayHeightMM" );
		if( rd_int(newname) ){
		  extern int DisplayHeight_MM;
			DisplayHeight_MM= def_int;
		}
	}
	sprintf( newname, "DisplayXRes-%d", screen );
	if( rd_int(newname) ){
	  extern int DisplayXRes;
		DisplayXRes= def_int;
	}
	else{
		sprintf( newname, "DisplayXRes" );
		if( rd_int(newname) ){
		  extern int DisplayXRes;
			DisplayXRes= def_int;
		}
	}
	sprintf( newname, "DisplayYRes-%d", screen );
	if( rd_int(newname) ){
	  extern int DisplayYRes;
		DisplayYRes= def_int;
	}
	else{
		sprintf( newname, "DisplayYRes" );
		if( rd_int(newname) ){
		  extern int DisplayYRes;
			DisplayYRes= def_int;
		}
	}

	if( rd_flag( "UnmappedWindowTitle") ){
		UnmappedWindowTitle= def_int;
	}
	if( rd_flag( "SetIconName") ){
		SetIconName= def_int;
	}
	if( rd_flag( "Ignore_UnmapNotify" ) ){
	  extern int Ignore_UnmapNotify;
		Ignore_UnmapNotify= def_int;
	}

	if( rd_dbl( "psLeftMarginOffset") ){
		ps_l_offset= def_dbl;
	}
	if( rd_dbl( "psBottomMarginOffset") ){
		ps_b_offset= def_dbl;
	}
	if( rd_dbl( "psFontWidthEstimator") ){
		Font_Width_Estimator= def_dbl;
	}
	if( rd_flag( "gsTextWidths" ) ){
		use_gsTextWidth= def_int;
	}
	if( rd_flag( "auto_gsTextWidths" ) ){
		auto_gsTextWidth= def_int;
	}
	if( rd_int( "Shift_Line_Echo_Segments" ) ){
		ShiftLineEchoSegments= def_int;
	}
	if( rd_dbl( "WindowAspect_Precision") ){
		win_aspect_precision= def_dbl;
	}
	if( rd_dbl( "XBiasThreshold" ) ){
		Xbias_thres= def_dbl;
	}
	if( rd_dbl( "YBiasThreshold" ) ){
		Ybias_thres= def_dbl;
	}
	if( rd_dbl( "Progress_ThresholdTime") ){
		Progress_ThresholdTime= def_dbl;
	}
	if( rd_int( "AscanfCheckEvent") ){
		ascanf_check_int= def_int;
	}
	if( rd_int( "Ascanf_Lazy_Address_Protection" ) ){
		PAS_Lazy_Protection= def_int;
	}
	if( rd_int( "MinBitsPPixel") ){
		ux11_min_depth= def_int;
	}
	if( rd_int( "VisualType") ){
		ux11_vis_class= def_int;
	}
	if( rd_flag( "BestVisual" ) ){
		search_vislist= def_int;
	}
    if (rd_pix("Background")) {
		bgPixel = def_pixel;
		StoreCName( bgCName );
    }
    if (rd_int("BorderSize")) {
		bdrSize = def_int;
    }
    if (rd_pix("Border")) {
		bdrPixel = def_pixel;
		StoreCName( bdrCName );
    }
	if( rd_flag( "BackingStore") ){
		BackingStore= def_int;
	}
	if( rd_int( "WM_Titlebar_Height") ){
		WM_TBAR= def_int;
	}
	if( rd_flag( "NoButtons" ) ){
		no_buttons= def_int;
	}
	if( rd_flag( "X_psMarkers" ) ){
		X_psMarkers= def_int;
	}
	if( rd_flag( "psSegmentDisconnectJumps") ){
		psSeg_disconnect_jumps= def_int;
	}
	if( rd_flag( "AllowFractionPrinting") ){
		Allow_Fractions= def_int;
	}
    if (rd_flag("engX")) {
		noExpX = ! def_int;
    }
    if (rd_flag("engY")) {
		noExpY = ! def_int;
    }

	if( rd_str("radix")){
		get_radix( def_str, &radix, radixVal );
	}
	else if( rd_str("polarBase")){
		get_radix( def_str, &radix, radixVal );
	}

    if (rd_str("GridStyle")) {
		gridLSLen = xtb_ProcessStyle(def_str, gridLS, MAXLS);
    }
	else{
		gridLSLen = xtb_ProcessStyle( gridLS, gridLS, MAXLS);
	}

    if (rd_pix("Foreground")) {
		normPixel = def_pixel;
		StoreCName( normCName );
    }

	if( rd_int( "MinButtonContrast") ){
		ButtonContrast= def_int;
	}

	if( rd_flag( "ReparentDialogs" ) ){
	  extern int xtb_ReparentDialogs;
		xtb_ReparentDialogs= def_int;
	}

    if (rd_pix("ZeroColor")){
		zeroPixel = def_pixel;
		StoreCName( zeroCName );
	}
    if (rd_pix("ZeroColour")){
		zeroPixel = def_pixel;
		StoreCName( zeroCName );
	}
    if (rd_pix("HighlightColor")){
		highlightPixel = def_pixel;
		StoreCName( highlightCName );
	}
    if (rd_pix("HighlightColour")){
		highlightPixel = def_pixel;
		StoreCName( highlightCName );
	}
    if (rd_pix("AxisColor")){
		axisPixel = def_pixel;
		StoreCName( axisCName );
	}
    if (rd_pix("AxisColour")){
		axisPixel = def_pixel;
		StoreCName( axisCName );
	}
    if (rd_pix("GridColor")){
		gridPixel = def_pixel;
		StoreCName( gridCName );
	}
    if (rd_pix("GridColour")){
		gridPixel = def_pixel;
		StoreCName( gridCName );
	}

    if (rd_str("ZeroStyle")) {
		zeroLSLen = xtb_ProcessStyle(def_str, zeroLS, MAXLS);
    }

    if (rd_dbl("AxisWidth")) axisWidth = def_dbl;
    if (rd_dbl("GridWidth")){
		gridWidth = def_dbl;
	}
	else{
		gridWidth= axisWidth;
	}
    if (rd_dbl("ErrorWidth")) errorWidth = def_dbl;

    if (rd_dbl("ZeroSize")) zeroWidth = def_dbl;

    if (rd_str("Device")){
		xfree( Odevice );
		Odevice = XGstrdup(def_str);
	}
    if (rd_str("Disposition")){
		xfree( Odisp );
		Odisp = XGstrdup(def_str);
	}
    if (rd_str("FileOrDev")){
		xfree( OfileDev );
		OfileDev = def_str;
	}
    if( rd_str( "PrintCommand") )
		 Oprintcommand= def_str;

    if (rd_str("Orientation")) {
		Print_Orientation= ( strncasecmp( def_str, "Portrait", 4) )? 1 : 0;
    }

	if( rd_dbl( "PSdashPower") ){
		psdash_power= def_dbl;
	}

	if( rd_dbl( "PSmarkBase") ){
		psm_base= def_dbl;
	}
	if( rd_dbl( "PSmarkIncrement") ){
		psm_incr= def_dbl- 1.0;
	}

      /* Read device specific parameters */
    for (idx = 0;  idx < hard_count;  idx++) {
		if( debugFlag){
			fprintf( StdErr, "Reading Resources for device '%s_'\n", hard_devices[idx].dev_name);
			fflush( StdErr);
		}
		if( !ResourceTemplate ){
			if( Oprintcommand){
/* 				hard_devices[idx].dev_spec= Oprintcommand;	*/
				stralloccpy( &hard_devices[idx].dev_spec, Oprintcommand, MFNAME-1);
			}
		}
		sprintf(newname, "%s_Dimension", hard_devices[idx].dev_name);
		if (rd_dbl(newname)){
			hard_devices[idx].dev_max_height = def_dbl;
			hard_devices[idx].dev_max_width = -1 ;
		}
		sprintf(newname, "%s_maxHeight", hard_devices[idx].dev_name);
		if (rd_dbl(newname))
			hard_devices[idx].dev_max_height = def_dbl;
		sprintf(newname, "%s_maxWidth", hard_devices[idx].dev_name);
		if (rd_dbl(newname))
			hard_devices[idx].dev_max_width = def_dbl;

		sprintf(newname, "%s_PrinterName", hard_devices[idx].dev_name);
		if (rd_str(newname)) {
			stralloccpy( &hard_devices[idx].dev_printer, def_str, MFNAME-1);
		}
		sprintf(newname, "%s_OutputTitleFont", hard_devices[idx].dev_name);
		if (rd_str(newname)) {
			(void) strncpy(hard_devices[idx].dev_title_font, def_str, sizeof(hard_devices[idx].dev_title_font)-1 );
		}
		sprintf(newname, "%s_OutputTitleSize", hard_devices[idx].dev_name);
		if (rd_dbl(newname))
			hard_devices[idx].dev_title_size = def_dbl;

		sprintf(newname, "%s_OutputLegendFont", hard_devices[idx].dev_name);
		if (rd_str(newname)) {
			(void) strncpy(hard_devices[idx].dev_legend_font, def_str, sizeof(hard_devices[idx].dev_legend_font)-1 );
		}
		sprintf(newname, "%s_OutputLegendSize", hard_devices[idx].dev_name);
		if (rd_dbl(newname))
			hard_devices[idx].dev_legend_size = def_dbl;

		sprintf(newname, "%s_OutputLabelFont", hard_devices[idx].dev_name);
		if (rd_str(newname)) {
			(void) strncpy(hard_devices[idx].dev_label_font, def_str, sizeof(hard_devices[idx].dev_label_font)-1 );
		}
		sprintf(newname, "%s_OutputLabelSize", hard_devices[idx].dev_name);
		if (rd_dbl(newname))
			hard_devices[idx].dev_label_size = def_dbl;

		sprintf(newname, "%s_OutputAxisFont", hard_devices[idx].dev_name);
		if (rd_str(newname)) {
			(void) strncpy(hard_devices[idx].dev_axis_font, def_str, sizeof(hard_devices[idx].dev_axis_font)-1 );
		}
		sprintf(newname, "%s_OutputAxisSize", hard_devices[idx].dev_name);
		if (rd_dbl(newname))
			hard_devices[idx].dev_axis_size = def_dbl;
    }

    if (rd_flag("Ticks")){
		htickFlag = def_int;
		vtickFlag = def_int;
	}
    if (rd_flag("ZeroLines")) zeroFlag = def_int;

	if( rd_flag( "CursorCross") ){
		CursorCross= def_int;
	}

    if (rd_flag("SmallPixels")) {
		if (def_int) {
			noLines = markFlag = 1;
			pixelMarks = 1;
		}
    }
    if (rd_flag("LargePixels")) {
		if (def_int) {
			markFlag = 1; pixelMarks = 2;
		}
    }
    if (rd_flag("Markers")) {
		if (def_int) {
			markFlag = 1;
			pixelMarks = 0;
		}
    }

	if( rd_dbl( "HighlightTune" ) ){
		highlight_par[0]= def_dbl;
	}
	if( rd_str( "HighlightPars" ) ){
	  double ul[5]= {0,0};
	  int i, n= highlight_npars;
		if( fascanf2( &n, def_str, ul, ',' )> 0 ){
			for( i= 0; i< n; i++ ){
				highlight_par[i]= ul[i];
			}
		}
	}
	if( rd_int( "HighlightMode" ) ){
		highlight_mode= def_int;
	}
	if( rd_dbl( "OverlapLegendTune") ){
		overlap_legend_tune= def_dbl;
	}
    if (rd_flag("StyleMarkers")) {
		if (def_int) {
			markFlag = 1;
			pixelMarks = 0;
		}
    }
    if (rd_flag("DrawAxes")) axisFlag = def_int;
    if (rd_flag("BoundBox")) bbFlag = def_int;
    if (rd_flag("NoLines")) noLines = def_int;
    if (rd_flag("PixelMarkers")) pixelMarks = def_int;
    if (rd_dbl("LineWidth")) lineWidth = def_dbl;

	if( rd_flag( "MindLegend") ){
		legend_always_visible= def_int;
	}
	if( rd_flag( "NoLegend") ){
		no_legend= def_int;
	}
	if( rd_int( "LegendType") ){
		CLIP_EXPR( legend_type, def_int, 0, 1);
	}

    /* Read the default line and color attributes */
    for (idx = 0;  idx < MAXATTR;  idx++) {
		(void) sprintf(newname, "%d_Style", idx);
		if (rd_str(newname)) {
			AllAttrs[idx].lineStyleLen =
			  xtb_ProcessStyle(def_str, AllAttrs[idx].lineStyle, MAXLS);
		}
		sprintf(newname, "%d_Color", idx);
		if( rd_pix(newname) && MonoChrome!= 2 ){
			AllAttrs[idx].pixelValue = def_pixel;
			StoreCName( AllAttrs[idx].pixelCName );
		}
		sprintf(newname, "%d_Colour", idx);
		if( rd_pix(newname) && MonoChrome!= 2 ){
			AllAttrs[idx].pixelValue = def_pixel;
			StoreCName( AllAttrs[idx].pixelCName );
		}
    }

    if( disp ){
		if (rd_flag("ReverseVideo")) {
			reverseFlag= def_int;
			ReallocColours(True);
		}
    }
	return(0);
}

xtb_frame *vars_pmenu= NULL, *process_pmenu= NULL;

/* An attempt at cleaning up X resources and data. "Regular" memory
 \ allocations are supposed to be cleaned up by the kernel.
 */
void CleanUp()
{ double x;
  static char called= 0;
  extern void *lib_termcap, *lib_readline;
  extern double Animation_Time, Animations;
  extern SimpleStats Animation_Windows;
  extern double XDBE_inside, XDBE_outside;
  extern unsigned int XDBE_count;
  extern char *parse_seconds( double sec, char *buf);

	if( called ){
		return;
	}

	if( Exit_Exprs ){
		  /* Force a reset on the internal state of the 2 routines likely to refuse executing our commands: */
		Evaluate_ExpressionList( NULL, NULL, False, NULL );
		new_param_now( NULL, NULL, 0 );
		  /* Now try to execute. */
		Evaluate_ExpressionList( ActiveWin, &Exit_Exprs, True, "exit expressions" );
		Exit_Exprs= XGStringList_Delete(Exit_Exprs);
	}

	if( disp ){
	  int i;
	  extern GC HandleMouseGC, ACrossGC, BCrossGC;

		called= True;

		if( AllSets ){
			for( i= 0; i< MaxSets; i++ ){
				Destroy_Form( &AllSets[i].process.C_set_process );
			}
			for( i= 0; i< MaxSets; i++ ){
				Destroy_Set( &AllSets[i], False );
			}
			xfree(AllSets);
			setNumber= 0;
		}

		if( !StartUp ){
			  /* If we're still starting up, there should be no (important) events
			   \ to discard!
			   */
			XSync( disp, True );
		}

		xtb_popup_delete( &vars_pmenu );
		xtb_popup_delete( &process_pmenu );

		if( IntensityColourFunction.XColours && IntensityColourFunction.NColours ){
			xfree( IntensityColourFunction.expression );
			Destroy_Form( &IntensityColourFunction.C_expression );
			IntensityColourFunction.name_table= XGStringList_Delete( IntensityColourFunction.name_table );
			if( IntensityColourFunction.NColours && IntensityColourFunction.XColours ){
			  ALLOCA( pixx, unsigned long, 8192, pixx_len);
			  Time_Struct timer;
				if( IntensityColourFunction.NColours>= 8192 ){
					Elapsed_Since(&timer, True);
					fprintf( StdErr, "[%d intensity colourmap entries..", IntensityColourFunction.NColours );
					fflush( StdErr );
				}
				if( pixx ){
				  int j, NC;
					if( (NC= IntensityColourFunction.NColours)>= 8192 ){
						fprintf( StdErr, "collecting.." );
						fflush( StdErr );
					}
					for( j= 0, i= 0; i< IntensityColourFunction.NColours; i++, j++ ){
						if( j== 8192 ){
							fprintf( StdErr, ".." );
							fflush( StdErr );
							XFreeColors( disp, cmap, pixx, j, 0);
							j= 0;
							NC-= 8192;
						}
						pixx[j]= IntensityColourFunction.XColours[i].pixel;
					}
					if( NC ){
						if( NC>= 8191 ){
							fprintf( StdErr, ".." );
							fflush( StdErr );
						}
						XFreeColors( disp, cmap, pixx, NC, 0);
					}
					GCA();
				}
				else{
					for( i= 0; i< IntensityColourFunction.NColours; i++ ){
						XFreeColors( disp, cmap, &IntensityColourFunction.XColours[i].pixel, 1, 0);
					}
				}
				xtb_XSync( disp, False );
				if( IntensityColourFunction.NColours>= 8192 ){
					Elapsed_Since(&timer, True);
					fprintf( StdErr, "freed (%ss)]\n", d2str( timer.Tot_Time, NULL, NULL) );
				}
			}
			xfree( IntensityColourFunction.XColours );
			xfree( IntensityColourFunction.exactRGB );
		}

		if( HandleMouseGC ){
			XFreeGC( disp, HandleMouseGC);
		}
		if( ACrossGC ){
			XFreeGC( disp, ACrossGC);
			XFreeGC( disp, BCrossGC);
		}

		xtb_close();

		XFreeCursor(disp, noCursor );
		XFreeCursor(disp, theCursor );
		XFreeCursor(disp, zoomCursor );
		XFreeCursor(disp, labelCursor );
		XFreeCursor(disp, cutCursor );
		XFreeCursor(disp, filterCursor );

		if( legend_greekFont.font!= legendFont.font ){
			UnGetFont( &legend_greekFont, "legend_GreekFont" );
		}
		if( label_greekFont.font!= labelFont.font ){
			UnGetFont( &label_greekFont, "label_GreekFont" );
		}
		if( title_greekFont.font!= titleFont.font ){
			UnGetFont( &title_greekFont, "title_GreekFont" );
		}
		if( axis_greekFont.font!= axisFont.font ){
			UnGetFont( &axis_greekFont, "axis_GreekFont" );
		}
		if( dialog_greekFont.font!= dialogFont.font ){
			UnGetFont( &dialog_greekFont, "dialog_GreekFont" );
		}
		UnGetFont( &legendFont, "LegendFont" );
		UnGetFont( &labelFont, "LabelFont" );
		UnGetFont( &titleFont, "TitleFont" );
		UnGetFont( &axisFont, "AxisFont" );
		UnGetFont( &dialogFont, "DialogFont" );
		UnGetFont( &cursorFont, "CursorFont" );
		if( use_markFont ){
			UnGetFont( &markFont, "MarkFont" );
		}
		for( i= 0; i< MAXATTR; i++ ){
			XFreePixmap( disp, AllAttrs[i].markStyle );
		}
		XFreePixmap( disp, dotMap );
		if( cmap!= DefaultColormap( disp, DefaultScreen(disp)) ){
			XFreeColormap( disp, cmap );
		}

		  /* Used to close display here - not a good idea! */

		{ extern int animations;
		  extern Time_Struct Animation_Timer;
			if( animations ){
				Elapsed_Since( &Animation_Timer, False );
				Animation_Time+= Animation_Timer.HRTot_T;
				Animations+= animations;
				animations= 0;
			}
		}
		  /* 20031021: printing of animation timing used to be below this line */
	}
	DelWindow( 0, &StubWindow );
	xfree( PrintFileName );
	gsResetTextWidths(NULL, True);

#ifdef XG_DYMOD_SUPPORT
	TBARprogress_header= "# CleanUp() final stage";
/*  	if( !Unloaded_Used_Modules )	*/
	if( DyModList )
	{
		new_param_now( NULL, NULL, 0 );
		if( debugFlag || scriptVerbose ){
			new_param_now( "for-toMAX[0,2, verbose[ IDict[Delete[$AllDefined]], Delete[$AllDefined] ] ]", &x, -1 );
		}
		else{
			new_param_now( "for-toMAX[0,2, IDict[Delete[$AllDefined]], Delete[$AllDefined] ]", &x, -1 );
		}
	}

	UnloadDyMods();

#endif
	if( lib_readline ){
	  extern FNPTR( gnu_write_history, int, (char *fn));
		if( gnu_write_history ){
		  char *fn= concat(PrefsDir, "/history", NULL);
			if( fn ){
				(*gnu_write_history)( fn );
				xfree( fn );
			}
		}
		dlclose( lib_readline );
		lib_readline= NULL;
	}
	if( lib_termcap ){
		dlclose( lib_termcap );
		lib_termcap= NULL;
	}

	if( disp ){
		if( Animation_Time> 0 || Animations> 0 ){
		  char buf[512];
			SS_Mean_(Animation_Windows);
			fprintf( StdErr, "%s.%u: %s for %s animations in an average of %s windows (%s/(anim*w);%s/(s*w))\n",
				Prog_Name, getpid(),
				parse_seconds(Animation_Time,buf),
				d2str( Animations, "%g", NULL ),
				d2str( Animation_Windows.mean, "%g", NULL),
				d2str( Animation_Time/(Animations* Animation_Windows.mean), "%g", NULL),
				d2str( Animations/(Animation_Time* Animation_Windows.mean), "%g", NULL)
			);
			Animation_Time= 0.0;
			Animations= 0.0;
		}
		if( XDBE_count ){
		  char buf[512], buf2[512];
			fprintf( StdErr,
				"%s.%u: %u frames drawn with XDBE; %ss \"inside\" the buffer swaps, %ss \"outside\"; %sHz effective.\n",
				Prog_Name, getpid(),
				XDBE_count,
				parse_seconds( XDBE_inside, buf ),
				parse_seconds( XDBE_outside, buf2), d2str( XDBE_count/(XDBE_inside+XDBE_outside), 0,0)
			);
		}
	}

	if( ascanf_verbose ){
	  extern SimpleStats SS_AscanfCompileTime;
		if( SS_AscanfCompileTime.count ){
		  char pc[]= "#xb1";
			fprintf( StdErr, "Ascanf compiler time stats: %s\n",
				SS_sprint_full( NULL, "%g", parse_codes(pc), 0, &SS_AscanfCompileTime )
			);
		}
	}

	if( xgShFD >= 0 ){
		close(xgShFD);
		if( strcmp( XgSharedMemName, "/dev/zero" ) ){
			shm_unlink(XgSharedMemName);
		}
	}

	  /* 20010719: closing the display should be the very last thing that we do!! */
	if( disp ){
		XCloseDisplay( disp );
		disp= NULL;
	}
}

extern int PS_PrintComment;
extern int Show_Progress;

extern int argerror(char *err, char *val);

double get_radix(char *arg, double *radix, char *radixVal )
{ double polarbase;
	if( !strncasecmp( arg, "PI", 2) ){
		*radix= M_PI;
		sprintf( radixVal, "PI");
	}
	else if( !strncasecmp( arg, "2PI", 3) ){
		*radix= 2* M_PI;
		sprintf( radixVal, "2PI");
	}
	else if( sscanf( arg, "%lf", &polarbase) ){
		*radix= polarbase;
		sprintf( radixVal, "%g", *radix);
	}
	return( *radix );
}

extern int ParseArgs( int argc, char *argv[]);

extern int NewSet( LocalWin *wi, DataSet **this_set, int spot );

extern int legend_setNumber;

extern int AddPoint_discard;

extern int ReadData(FILE *stream, char *the_file, int filenr);

int XG_Stripped= 0;
extern char d3str_format[16];
extern int d3str_format_changed;

int AxisValueMinDigits= 2;
char *AxisValueFormat= NULL;

typedef struct value{
	double x, y, err;
	int flag, has_error;
} Values;

typedef struct xvalue{
	double x;
	int indeks, set;
} XValues;

#define Add_Xval(Xval,Set,indeks,X)	{Xval->x=(X);Xval->set=Set;Xval->indeks=indeks;NXvalues++;XXval=Xval++;}

XValues *X_Values;

static XValues *Search_Xval_Back( XValues *XXval, long indeks, Values *Val)
{
	for( ; XXval->x!= Val->x && XXval->indeks== indeks && XXval>= X_Values ; XXval-- ){
		;
	}
	return( XXval);
}

/* 
static Values *Search_Val_Forward( _Values, indeks, set, N, Xval)
Values **_Values;
long indeks, set, N;
XValues *Xval;
{  long i;
	_Values+= indeks+ 1;
	for( i= indeks+1; i< N; i++, _Values++ ){
		if( (*_Values)[set].x== Xval->x && (*_Values)[set].flag){
			return( &(*_Values)[set] );
		}
	}
	return( NULL);
}
 */

static Values *Search_Val_Forward( Xval, xv, NXvalues, _Values, set, N)
XValues *Xval;
long xv, NXvalues;
Values **_Values;
long set, N;
{  long i, indeks= Xval->indeks;
   Values *v;
   double x= Xval->x;
	for( ++Xval, i= xv+1; i< NXvalues; i++, Xval++){
		if( (v= &_Values[indeks][set])->flag ){
			if( Xval->x== x && v->x== x &&
				_Values[indeks-1][set].flag== 0
			){
				return( v );
			}
		}
	}
	return(NULL);
}

static int Print_Val( FILE *fp, long row, Values *Val, int *Xoutput, int *output, char *tabbuf )
{ int cols= 0;
	if( ! *Xoutput){
		fprintf( fp, "%ld\t%s", row, d2str(Val->x, d3str_format, NULL) );
		*Xoutput= 1;
		cols+= 1;
	}
	if( *tabbuf){
		fputs( tabbuf, fp);
		tabbuf[0]= '\0';
	}
	fputc( '\t', fp);
	if( Val->has_error){
		fprintf( fp, "%s\t%s", 
			d2str( Val->y, d3str_format, NULL), 
			d2str( Val->err, d3str_format, NULL)
		);
		*output = 1;
		cols+= 2;
	}
	else{
		fprintf( fp, "%s", d2str( Val->y, d3str_format, NULL) );
		*output = 1;
		cols+= 1;
	}
	return( cols );
}

/* make a dump in the Cricket Graph TM ascii format	*/
static int _SpreadSheetDump( LocalWin *wi, FILE *fp, char errmsg[ERRBUFSIZE], int CricketGraph)
{  long I, i, j, NXvalues= 0, set, indeks, cols, mcols= 0;
   char *c, CricketPolar= CricketGraph && wi->polarFlag ;
   DataSet *Set= AllSets;
   long N, NN, transformed= 0;
   XValues *Xval, *XXval;
   Values **_Values, *Val;
   int _setNumber= 0, *set_nr;

	/* Determine the number of points, and try to allocate memory
	 * for them before writing anything to fp
	 */
	for( NN= N= Set->numPoints, i= 0; i< setNumber; i++, Set++){
		if( !(draw_set(wi,i)== 0 && XG_Stripped) ){
			if( Set->numPoints> N){
				N= Set->numPoints;
			}
			if( Set->numPoints > 0 ){
				NN+= Set->numPoints;
			}
			_setNumber+= 1;
		}
	}
	if( debugFlag)
		fprintf( StdErr, "_SpreadSheetDump(): allocating %ld XValues\n", NN);
	if( (set_nr= (int*) calloc( _setNumber, sizeof(int) ))== NULL ){
		sprintf( errmsg, "_SpreadSheetDump(): can't allocate %d setnr buffer\n%s", _setNumber, serror() );
		return(1);
	}
	if( (Xval= (XValues*)calloc( NN, sizeof(XValues)))== NULL){
		sprintf( errmsg, "_SpreadSheetDump(): can't allocate %ld XValues\n%s", NN, serror() );
		return(1);
	}
	X_Values= Xval;

	if( debugFlag)
		fprintf( StdErr, "_SpreadSheetDump(): allocating %d x %ld Values\n", _setNumber, N);
	if( (_Values= (Values**) calloc( N, sizeof(Values*)))== NULL){
		sprintf( errmsg, "_SpreadSheetDump(): can't allocate %ld *Values\n%s", N, serror() );
		return(1);
	}
	else{
		for( i= 0; i< N; i++){
			if( (_Values[i]= (Values*) calloc( _setNumber, sizeof(Values)))== NULL){
				sprintf( errmsg, "_SpreadSheetDump(): can't allocate %d Values, indeks %ld\n%s", _setNumber, i, serror() );
				return(1);
			}
		}
	}

	if( debugFlag)
		fprintf( StdErr, "_SpreadSheetDump(): Writing header and making spread-sheet\n");
/* 	Set= AllSets;	*/
	if( CricketGraph ){
		fputs( "* XGraph data", fp );
	}
	fprintf( fp, "\t%s", XLABEL(wi));
	for( Set= AllSets, I= 0, i= 0; i< setNumber; i++, Set++){
		if( !(draw_set( wi, i)== 0 && XG_Stripped) ){
		  int _j;
/* 			fprintf( fp, "\t%10.10s: ", YLABEL(wi) );	*/
			fputc( '\t', fp );
			{ char *c= Set->setName;
				while( *c ){
					switch( *c ){
						default:
							fputc( *c, fp );
							break;
						case '\n':
							fputs( "\\n", fp  );
							break;
						case '\r':
							fputs( "\\r", fp );
							break;
						case '\t':
							fputs( "\\t", fp );
							break;
					}
					c++;
				}
			}
			if( wi->use_errors && Set->use_error && Set->has_error && Set->numErrors && wi->error_type[Set->set_nr] && !CricketPolar ){
				  /* 20020203: Don't repeat setName for the error column... */
				switch( wi->error_type[Set->set_nr] ){
					default:
					case 1:
					case EREGION_FLAG:
					case 2:
					case 3:
						fprintf( fp, "\t(Error)" );
						break;
					case 4:
						fprintf( fp, "\t(Direction)" );
						break;
					case INTENSE_FLAG:
						fprintf( fp, "\t(Intensity)" );
						break;
					case MSIZE_FLAG:
						fprintf( fp, "\t(Size)" );
						break;
				}
			}
			for( _j= 0, j= 0; _j< Set->numPoints; _j++){
				if( (Set->plot_interval<= 0 || (!XG_Stripped || (_j % Set->plot_interval)==0)) && !DiscardedPoint( wi, Set, _j) ){
				  double x= XVAL( Set, _j), y= YVAL( Set, _j), err= ERROR( Set, _j), vv, ldy= y-err, hdy=y+err;

					if( 1|| !PrintingWindow){
						if( Handle_An_Event( wi->event_level, 1, "_SpreadSheetDump()", wi->window, 
								/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
							)
						){
							wi->event_level--;
							sprintf( errmsg, "_SpreadSheetDump(): unhandled event\n" );
							return(1);
						}
					}
					if( wi->delete_it== -1 ){
						sprintf( errmsg, "_SpreadSheetDump(): window no longer exists\n" );
						return(1);
					}
					if( wi->halt ){
						wi->redraw= 0;
						  /* wi->halt must be unset before calling this function again. This
						   \ makes it possible to retain redrawing for a while...
						   */
						sprintf( errmsg, "_SpreadSheetDump(): user break\n" );
						return(1);
					}
					  /* *DATA_PROCESS*	*/
					Set->data[0][0]= x; Set->data[0][1]= y; Set->data[0][2]= err;
					Set->data[0][3]= (Set->lcol>= 0)? VVAL(Set,_j) : 0;
					if( !Set->raw_display && !wi->vectorFlag ){
						DrawData_process( wi, Set, Set->data, _j, 1, ASCANF_DATA_COLUMNS, &x, &y, NULL, NULL,
							NULL, &ldy, NULL, &hdy, NULL, NULL, NULL, NULL
						);
					}
					x= Set->data[0][0]; y= Set->data[0][1]; err= Set->data[0][2]; vv= Set->data[0][3];

					if( CricketPolar ){
					  double *x_1= (j)? &Set->xvec[_j-1] : NULL,
							*y_1= (j)? &Set->yvec[_j-1] : NULL;
					  int ok= 1;
						  /* *TRANSFORM_?*	*/
						if( do_TRANSFORM( wi, _j, 1, 3, &x, NULL, NULL, &y, &ldy, &hdy, 0, False )> 0 ){
							transformed+= 1;
							  /* We cannot just transform the error: *TRANSFORM_?* is an axis-transformation,
							   \ and thus works on (y-err) and (y+err). Thus we determine the transformed
							   \ error as ((y+err)-(y-err))/2, which gives a bar of the right size, only
							   \ (possibly incorrectly) symmetrical around the transformed y.
							   */
							err= fabs(hdy- ldy)/ 2.0;
						}
						  /* do_transform() will do the polar transformation on the x and y only (err is not changed)	*/
						  /* We must NOT do do_TRANSFORM() again, so pass -1 to the is_bounds argument	*/
						do_transform( wi /* NULL */,
							Set->fileName, __DLINE__, "_SpreadSheetDump()", &ok, Set, &x, NULL, NULL, &y, NULL, NULL,
							x_1, y_1, 1, _j, 1.0, 1.0, 1.0, -1, 0, False
						);
						_Values[j][I].x= ((double) ((int)( x * 100.0+ 0.5)))/ 100.0;
						_Values[j][I].y= ((double) ((int)( y * 100.0+ 0.5)))/ 100.0;
					}
					else{
						  /* *TRANSFORM_?*	*/
						if( do_TRANSFORM( wi, j, 1, 3, &x, NULL, NULL, &y, &ldy, &hdy, 0, False )> 0 ){
							transformed+= 1;
							err= fabs(hdy- ldy)/ 2.0;
						}
						_Values[j][I].x= x;
						_Values[j][I].y= y;
					}
					_Values[j][I].err= err;
					_Values[j][I].flag= 1;
					_Values[j][I].has_error= wi->use_errors && Set->use_error && Set->has_error && Set->numErrors && !CricketPolar ;
					j+= 1;
				}
			}
			for( ; j< N; j++ ){
				_Values[j][I].flag= 0;
				_Values[j][I].has_error= wi->use_errors && Set->use_error && Set->has_error && Set->numErrors && !CricketPolar ;
			}
			set_nr[I]= i;
			I+= 1;
		}
		else if( debugFlag ){
			fprintf( StdErr, "\tskipping set %ld\n", i );
		}
	}
	fprintf( fp, "\n%s\n%s\n", YLABEL(wi), PrintTime );

	XXval= Xval= X_Values;
	if( debugFlag){
		fputs( "_SpreadSheetDump(): sorting XValues\n", StdErr);
	}
	for( indeks= 0; indeks< N; indeks++){
		Val= _Values[indeks];
		for( set= 0; set< _setNumber; set++, Val++ ){
			if( 1|| !PrintingWindow){
				if( Handle_An_Event( wi->event_level, 1, "_SpreadSheetDump()", wi->window, 
						/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
					)
				){
					wi->event_level--;
					sprintf( errmsg, "_SpreadSheetDump(): unhandled event\n" );
					return(1);
				}
			}
			if( wi->delete_it== -1 ){
				sprintf( errmsg, "_SpreadSheetDump(): window no longer exists\n" );
				return(1);
			}
			if( wi->halt ){
				wi->redraw= 0;
				  /* wi->halt must be unset before calling this function again. This
				   \ makes it possible to detain redrawing for a while...
				   */
				sprintf( errmsg, "_SpreadSheetDump(): user break\n" );
				return(1);
			}
			if( Val->flag ){
				if( indeks!= XXval->indeks || Xval== X_Values ){
				  /* either the first for this indeks, or the very first (indeks==0)	*/
					Add_Xval( Xval, set_nr[set], indeks, Val->x);
				}
				else if( Val->x!= XXval->x ){
				  /* need to check if not already present, e.g. because the previous
				   * x (x[-1]) differed from x[-2] which equals x.
				   * So we search backwards until we run into the same x,indeks (meaning x is present), or into a 
				   * different indeks, or the start of the list.
				   */
					XXval= Search_Xval_Back( XXval, indeks, Val);
					if( XXval->x!= Val->x || XXval->indeks!= indeks ){
					  /* we didn't find anything	*/
						Add_Xval( Xval, set_nr[set], indeks, Val->x);
					}
					else
					  /* reset XXval to point to the last added entry	*/
						XXval= &Xval[-1];
				}
			}
		}
	}

	if( debugFlag){
		fprintf( StdErr, "_SpreadSheetDump(): found %ld different XValues\n", NXvalues);
		if( debugLevel> 1){
			fprintf( StdErr, "(index,set,xval):\n");
			for( i= 0; i< NXvalues- 1; i++){
				fprintf( StdErr, "(%d,%d,%g)\t", X_Values[i].indeks, X_Values[i].set, X_Values[i].x );
			}
			fprintf( StdErr, "(%d,%d,%g)\n", X_Values[i].indeks, X_Values[i].set, X_Values[i].x );
		}
	}

/* 	fprintf( fp, "%s\n", YLABEL(wi));	*/
	NN= 0;
	c= strtok( titleText, "\t:;,");
	for( XXval= Xval= X_Values, i= 0; i< NXvalues; i++, Xval++){
	  int output, Xoutput;
	  char tabbuf[1024];
	  Values *VVal;

		indeks= Xval->indeks;
		Val= _Values[indeks];
		Xoutput= output= 0;
		tabbuf[0]= '\0';
		if( c ){
			fprintf( fp, "%s", c);
			c= strtok( NULL, "\t:;,");
			output = 1;
		}
		for( cols= 0, set= 0; set< _setNumber; set++, Val++){
			if( Val->flag && Val->x== Xval->x ){
				cols+= Print_Val( fp, NN, Val, &Xoutput, &output, tabbuf);
			}
			else if( Sort_Sheet ){
				if( (VVal= Search_Val_Forward( Xval, i, NXvalues, _Values, set, N)) ){
					cols+= Print_Val( fp, NN, VVal, &Xoutput, &output, tabbuf);
					VVal->flag= 0;
				}
			}
			else{
				if( Val->has_error)
					strcat( tabbuf, "\t \t ");
				else
					strcat( tabbuf, "\t ");
			}
		}
		if( output){
			fputc( '\n', fp);
			mcols= MAX( mcols, cols );
			NN++;
			if( debugFlag ){
				fprintf( StdErr, "row %ld, %ld columns (%ld x %ld)\n", NN, cols, NN, mcols );
			}
		}
	}

	sprintf( errmsg, "Created a%s %ld rows x %ld(%d) columns spreadsheet\n",
		(Sort_Sheet)? " sorted" : "n unsorted",
		NN, mcols, _setNumber+ 2
	);
	if( CricketPolar ){
		strcat( errmsg, "Sheet contains the (transformed) projection as shown in the graph\n");
	}
	else{
		strcat( errmsg, "Sheet contains the original (non-transformed) data\n");
	}
	if( transformed ){
		strcat( errmsg, "Y-Errors have been transformed to *TRANSFORM_Y* of ((y+err)-(y-err))/2\n");
	}
	if( debugFlag){
		fputs( errmsg, StdErr);
	}
	xfree( X_Values);
	for( i= 0; i< N; i++){
		xfree( _Values[i]);
	}
	xfree( _Values);
	xfree( set_nr );

	return( 1);
}

/* make a dump in a generic ASCII format	*/
int SpreadSheetDump( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
)
{ int r;
	if( initFile ){
		r= _SpreadSheetDump( thePrintWin_Info, fp, errmsg, 0);
	}
	else{
		return( 1 );
	}
	if( r && strlen(errmsg) )
		do_message( errmsg);
	errmsg[0]= '\0';
	return( 0);
}

/* make a dump in the Cricket Graph TM ascii format	*/
int CricketDump( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
)
{ int r;
	if( initFile ){
		r= _SpreadSheetDump( thePrintWin_Info, fp, errmsg, 1);
	}
	else{
		return( 1 );
	}
	if( r && strlen(errmsg) )
		do_message( errmsg);
	errmsg[0]= '\0';
	return( 0);
}

int count_char( char *string, char c)
{  int n= 0;
	while( string && *string ){
		if( *string== c ){
			n+= 1;
		}
		string++;
	}
	return( n );
}

/* Print a string that can contain a string - a section within '"'s. Within such a section,
 \ print non-printables (control characters) in a properly escaped fashion. <instring> points
 \ to a parental variable that is to be used as a status indicitor for the being-in-a-string.
 \ 20020118: when *instring<0, don't modify it: -1==False, -2==True
 */
int Sprint_string_string( Sinc *fp, char *header, char *trailer, char *string, int *instring )
{ int len= 0;
	  int hlen= (header)? strlen(header) : 0;
	  char *c= string;
		Sputs( header, fp);
		len+= hlen;
		while( c && *c ){
			if( *instring>= 0 && *c== '"' && (c== string || c[-1]!= '\\') ){
				*instring= !(*instring);
			}
			if( isprint(*c) ){
				Sputc( *c, fp );
				len+= 1;
			}
			else if( *instring>0 || *instring==-2 ){ switch( *c ){
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
				  char buf[8];
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
					  char buf[8];
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

int print_string_string( FILE *fp, char *header, char *trailer, char *string, int *instring )
{ Sinc sinc;
	return( Sprint_string_string( Sinc_file( &sinc, fp, 0,0), header, trailer, string, instring ) );
}

  /* wrapper for backwards compatibility:	*/
int print_string2( FILE *fp, char *header, char *trailer, char *string, int instring )
{
	return( print_string_string( fp, header, trailer, string, &instring ) );
}

int sprint_string_string( char **target, char *header, char *trailer, char *string, int *instring )
{ Sinc sinc;
  int r;
	sinc.sinc.string= NULL;
	Sinc_string_behaviour( &sinc, NULL, 0,0, SString_Dynamic);
	Sflush(&sinc);
	r= Sprint_string_string( &sinc, header, trailer, string, instring );
	Sflush(&sinc);
	*target= sinc.sinc.string;
	return(r);
}

  /* wrapper for backwards compatibility:	*/
int sprint_string2( char **target, char *header, char *trailer, char *string, int instring )
{
	return( sprint_string_string( target, header, trailer, string, &instring ) );
}

int Sprint_string( Sinc *fp, char *header, char *wrapstr, char *trailer, char *string )
{ int len= 0;
  char *last= NULL, *empty="";
	if( !string ){
		string= empty;
	}
	if( count_char( string, '\n')< 1 && !index( string, '"') ){
		len+= Sputs( header, fp );
		len+= Sputs( string, fp );
		len+= Sputs( trailer, fp );
	}
	else{
	  int hlen= (header)? strlen(header) : 0;
	  char *Buf= XGstrdup( string), *buf= Buf, *c;
	  int contained_string= False, has_string;
		Sputs( header, fp);
		len+= hlen;
		has_string= (index( string, '"'))? True : False;
		while( xtb_getline( &buf, &c ) ){
			if( c!= Buf ){
				if( contained_string ){
					Sputs( "#xn", fp);
					len+= 3;
				}
				else{
					len+= Sputs( (wrapstr)? wrapstr : "\n", fp);
				}
			}
			  /* 20020409: one of course should of course print through Sprint_string_string
			   \ (which will handle the opcodes when we're in a string) even if the current line
			   \ doesn't have quotes... The only exception is when there are no quotes in the
			   \ whole text to be printed (or when there are no more quotes in the remaining text,
			   \ but that's no trivial to obtain here).
			   */
			if( has_string /* index( c, '"' ) */ ){
				len+= Sprint_string_string( fp, "", "", c, &contained_string );
			}
			else{
				len+= Sputs( c, fp );
				contained_string= False;
			}
			last= c;
		}
		if( last ){
		  int llen= strlen(last);
			if( last[llen-1]== 'n' && last[llen-2]== '\\' ){
				if( debugFlag ){
					fprintf( StdErr, "Sprint_string(\"%s\"): last part terminates with newline sequence: forced newline to output\n",
						string
					);
				}
				Sputc( '\n', fp );
			}
		}
		len+= Sputs( trailer, fp );
		xfree( Buf );
	}
	Sflush( fp );
	return( len );
}

int print_string( FILE *fp, char *header, char *wrapstr, char *trailer, char *string )
{ Sinc sinc;
	return( Sprint_string( Sinc_file( &sinc, fp, 0,0), header, wrapstr, trailer, string) );
}

int DumpDiscarded_Points( FILE *fp, LocalWin *wi, DataSet *this_set, int startlen, int linelen, char *trailer )
{ int i, p, s= -1, len= startlen, ok= 0, n= 0;
  char pbuf[128];
	if( this_set->discardpoint ){
		for( p= 0, i= 0; i< this_set->numPoints; i++ ){
		  int plen;
			if( DiscardedPoint( wi, this_set, i)> 0 || (DiscardedPoint( wi, this_set,i)< 0 && wi->DumpProcessed) ){
				n+= 1;
				if( s== -1 ){
					s= i;
					ok= 0;
				}
				else if( i== this_set->numPoints- 1 ){
					if( s< i- 1 ){
						plen= sprintf( pbuf, " %d-%d", s, i );
					}
					else{
						plen= sprintf( pbuf, " %d", i );
					}
					s= i;
					ok= 1;
				}
				else{
					ok= 0;
				}
				if( ok ){
					if( len+ plen+1 >= linelen ){
						fputc( '\n', fp );
						len= fprintf( fp, "                             " );
					}
					len+= fprintf( fp, pbuf );
				}
				p= i;
			}
			else if( s>= 0 ){
				if( s< i- 1 ){
					plen= sprintf( pbuf, " %d-%d", s, i-1 );
				}
				else{
					plen= sprintf( pbuf, " %d", s );
				}
				if( len+ plen+1 >= linelen ){
					fputc( '\n', fp );
					len= fprintf( fp, "                             " );
				}
				len+= fprintf( fp, pbuf );
				s= -1;
			}
		}
		if( trailer ){
			fputs( trailer, fp );
		}
	}
	return(n);
}

char *today()
{ time_t timer= time(NULL);
  static char buf[256];
  strncpy( buf, asctime( localtime(&timer) ), 256 );
  buf[255]= '\0';
  return( cleanup(buf) );
}

char *arrow_types[4]= { "", "*ARROWS* B", "*ARROWS* E", "*ARROWS* BE" };

char *SetColourName( DataSet *this_set )
{ char *c;
	if( this_set->pixvalue< 0 ){
		c= this_set->pixelCName;
	}
	else if( this_set->pixvalue== this_set->set_nr % MAXATTR ){
		c= "default";
	}
	else{
		c= AllAttrs[this_set->pixvalue].pixelCName;
	}
	return(c);
}

void _DumpSetHeaders( FILE *fp, LocalWin *wi, DataSet *this_set, int *points, int *NumObs, Boolean newline /* , int empty 	*/ )
{ int err, dp= 0;
  char asep = ascanf_separator;
	if( this_set->XUnits && strcmp( this_set->XUnits, wi->XUnits) ){
		print_string( fp, "*XYlabel*", "\\n\n", "\n", this_set->XUnits );
	}
	if( this_set->YUnits && strcmp( this_set->YUnits, wi->YUnits) ){
		print_string( fp, "\t*YXlabel*", "\\n\n", "\n", this_set->YUnits );
	}
	if( this_set->process.set_process_len ){
		if( this_set->process.description ){
				if( wi->DumpProcessed ){
					fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
				}
			print_string( fp, "*SET_PROCESS_DESCRIPTION*", "\\n\n", "\n\n", this_set->process.description );
		}
		if( wi->DumpProcessed ){
			fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
		}
		if( strstr( this_set->process.set_process, "\n" )
		   && (!this_set->process.separator || this_set->process.separator == ascanf_separator)
		){ char *sp= this_set->process.set_process;
			print_string( fp, "**SET_PROCESS**\n", NULL, " @\n*!SET_PROCESS*\n", sp);
		}
		else{
			fprintf( fp, "**SET_PROCESS**\n" );
			Print_Form( fp, &this_set->process.C_set_process, 0, True, NULL, "\t", "\n", False );
			fputs( " @\n*!SET_PROCESS*\n", fp );
			if( wi->DumpProcessed ){
				fputc( '\n', fp );
			}
		}
	}
	if( this_set->displaced_x || this_set->displaced_y ){
		fprintf( fp, "*DISPLACED* %s,%s\n",
			d2str( this_set->displaced_x, d3str_format, NULL), d2str( this_set->displaced_y, d3str_format, NULL)
		);
	}
	if( this_set->error_point!= -1 ){
		fprintf( fp, "*ERROR_POINT* %d\n", this_set->error_point );
	}
	if( wi->error_type[this_set->set_nr]!= -1 ){
		fprintf( fp, "*ERROR_TYPE* %d", wi->error_type[this_set->set_nr] );
		if( this_set->ebarWidth_set ){
			fprintf( fp, ",%s", d2str( this_set->ebarWidth, "%g", NULL) );
		}
		fputc( '\n', fp );
	}
	if( (this_set->barBase_set && (!barBase_set || this_set->barBase!= barBase)) ||
		(this_set->barWidth_set && (!barWidth_set || this_set->barWidth!= barWidth)) ||
		this_set->barType!= barType
	){
		fprintf( fp, "*BARPARS* %g,%g,%d\n", this_set->barBase, this_set->barWidth, this_set->barType );
	}
	if( this_set->valueMarks ){
		fputs( "*VALUEMARKS*", fp );
		if( CheckMask( this_set->valueMarks, VMARK_ON) ){
			fputs( " ON", fp );
		}
		if( CheckMask( this_set->valueMarks, VMARK_FULL) ){
			fputs( " FULL", fp );
		}
		if( CheckMask( this_set->valueMarks, VMARK_RAW) ){
			fputs( " RAW", fp );
		}
		fputs( "\n", fp );
	}
	if( this_set->Xscale!= 1 || this_set->Yscale!= 1 || this_set->DYscale!= 1 ){
		fprintf( fp, "*SCALFACT* %g %g %g\n", this_set->Xscale, this_set->Yscale, this_set->DYscale );
	}
	fprintf( fp, "*COLUMNS* N=%d x=%d y=%d e=%d", this_set->ncols,
		wi->xcol[this_set->set_nr], wi->ycol[this_set->set_nr], wi->ecol[this_set->set_nr]
	);
	if( wi->lcol[this_set->set_nr]>= 0 ){
		fprintf( fp, " l=%d", this_set->lcol);
	}
	fprintf( fp, " n=%d", this_set->Ncol);
	if( this_set->set_link>= 0 && this_set->set_link< setNumber ){
/* 		fprintf( fp, " L2=%d", this_set->set_link-empty );	*/
		fprintf( fp, " L2=%d", AllSets[this_set->set_link].dumped_set );
	}
	fputc( '\n', fp );
	if( this_set->ColumnLabels ){
	  Sinc sink;
		Sprint_SetLabelsList( Sinc_file( &sink, fp, 0,0), this_set->ColumnLabels, "*LABELS* new\n", "\n\n" );
	}

#ifdef XGSAVE_OLD_AUTOSCRIPT
	if( this_set->noLines== 0 && this_set->markFlag== 0){
		err= 0;
	}
	else if( this_set->noLines && this_set->markFlag ){
		err= 1;
	}
	else if( this_set->noLines== 0 && this_set->markFlag ){
		err= 2;
	}
	else if( this_set->polarFlag== 0 && this_set->barFlag && this_set->noLines && this_set->markFlag ){
		err= 4;
	}
	else{
		err= -1;
	}
	fprintf( fp, "*AUTOSCRIPT* %d %d %d 8 %g %d 0 %g\n", this_set->linestyle+1, err,
		this_set->linestyle+1, this_set->lineWidth, this_set->elinestyle, this_set->elineWidth
	);
#else
	err= 0;
	if( this_set->noLines ){
		err|= 1;
	}
	if( this_set->markFlag ){
		switch( this_set->pixelMarks ){
			case 1:
				err|= (1 << 1);
				break;
			case 2:
				err|= (2 << 1);
				break;
			case 0:
				err|= (3 << 1);
				break;
		}
	}
	if( this_set->barFlag ){
		err|= (1 << 3);
	}
	if( this_set->overwrite_marks ){
		err|= (1 << 4);
	}
	if( this_set->show_legend ){
		err|= (1 << 5);
	}
	if( !this_set->use_error ){
		err|= (1 << 6);
	}
	if( wi->legend_line[this_set->set_nr].highlight ){
		err|= (1 << 7);
	}
	if( this_set->raw_display ){
		err|= (1 << 8);
	}
	if( wi->mark_set[this_set->set_nr]> 0 ){
		err|= (1 << 9);
	}
	if( !this_set->show_llines ){
		err|= (1 << 10);
	}
	if( this_set->floating ){
		err|= (1 << 11);
	}
	fprintf( fp, "*PROPERTIES* colour=%d flags=0x%x linestyle=%d mS=8 lineWidth=%g elinestyle=%d marker=%d elineWidth=%g",
		this_set->pixvalue, err,
		this_set->linestyle+1, this_set->lineWidth, this_set->elinestyle,
		(this_set->markstyle< 0 || XGStoreColours)? ABS(this_set->markstyle) : -1 * ABS(this_set->markstyle), this_set->elineWidth
	);
	if( !NaNorInf(this_set->markSize) ){
		fprintf( fp, " markSize=%s", d2str(this_set->markSize, NULL, NULL));
	}
	if( XGStoreColours ){
		fprintf( fp, " cname=\"%s\"", SetColourName(this_set) );
		if( wi->legend_line[this_set->set_nr].pixvalue< 0 ){
			fprintf( fp, " hl_cname=\"%s\"", wi->legend_line[this_set->set_nr].pixelCName);
		}
	}
	fputs( "\n", fp);
#endif

	{ int numPoints= 0, pp, j;
		if( this_set->discardpoint || wi->discardpoint
#ifdef RUNSPLIT
			|| this_set->splithere
#endif
		){
			  /* wi->discardpoint: window-based discarded points are never dumped,
			   \ that's why we need somewhat more intricate checking
			   */
			if( wi->DumpProcessed ){
				for( j= 0; j< this_set->numPoints && !(SplitHere(this_set, j) && splits_disconnect); j++ ){
					if( !DiscardedPoint( wi, this_set, j) ){
						numPoints+= 1;
					}
					else{
						dp+= 1;
					}
				}
			}
			else{
				for( j= 0; j< this_set->numPoints && !(SplitHere(this_set, j) && splits_disconnect); j++ ){
/* 					if( !(DiscardedPoint( wi, this_set, j)> 0 || (DiscardedPoint( wi, this_set,j)< 0 && wi->DumpProcessed)) )	*/
					if( DiscardedPoint( NULL, this_set, j)<= 0 &&
						!(WinDiscardedPoint(wi, this_set, j)< 0)
					)
					{
						numPoints+= 1;
					}
					else{
						dp+= 1;
					}
				}
			}
		}
		else{
			numPoints= this_set->numPoints;
		}
		pp= (XG_Stripped && this_set->plot_interval)? (numPoints+1) / this_set->plot_interval + 1 : numPoints;
		if( pp!= *points ){
			fprintf( fp, "*POINTS* %d\n", (*points= pp) );
		}
	}
	if( this_set->discardpoint && dp ){
	  int len;
		len= fprintf( fp, "*EXTRATEXT* Discarded points:" );
		DumpDiscarded_Points( fp, wi, this_set, len, LMAXBUFSIZE, "\n\n" );
	}
	if( this_set->NumObs!= *NumObs ){
		fprintf( fp, "*N* %d\n", (*NumObs= this_set->NumObs) );
	}
	  /* 20010816: use memcmp() instead of comparing elements of vectorPars 1-by-1. */
	if( this_set->vectorType!= vectorType || this_set->vectorLength!= vectorLength ||
		memcmp( this_set->vectorPars, vectorPars, MAX_VECPARS* sizeof(double) )
/* 		|| this_set->vectorPars[0]!= vectorPars[0] || this_set->vectorPars[1]!= vectorPars[1]	*/
		  /* A global vectorpars commandline specification is only done when wi->vectorFlag>0. This is
		   \ maybe not set, e.g. when the lastly drawn set is not in vector mode. Hence, we should
		   \ check for the following possibility:
		   \ 20030225: always include it if wi->vectorFlag<=0
		   */
		|| (wi->vectorFlag<= 0 /* && wi->error_type[this_set->set_nr]== 4 */ )
	){
		switch( this_set->vectorType ){
		  int i;
			case 0:
				fprintf( fp, "*VECTORPARS* %d,%s\n", this_set->vectorType,
					d2str( this_set->vectorLength, d3str_format, NULL )
				);
				break;
			case 1:
				fprintf( fp, "*VECTORPARS* %d,%s,%s,%s\n",
					this_set->vectorType,
					d2str( this_set->vectorLength, d3str_format, NULL ),
					d2str( this_set->vectorPars[0], d3str_format, NULL ),
					d2str( this_set->vectorPars[1], d3str_format, NULL )
				);
				break;
			default:
				fprintf( fp, "*VECTORPARS* %d,%s,%s",
					this_set->vectorType,
					d2str( this_set->vectorLength, d3str_format, NULL ),
					d2str( this_set->vectorPars[0], d3str_format, NULL )
				);
				for( i= 1; i< MAX_VECPARS; i++ ){
					fprintf( fp, ",%s", d2str( this_set->vectorPars[i], d3str_format,0) );
				}
				fputc( '\n', fp );
				break;
		}
		vectorType= this_set->vectorType;
		vectorLength= this_set->vectorLength;
		memcpy( vectorPars, this_set->vectorPars, MAX_VECPARS* sizeof(double));
	}
	if( this_set->plot_interval> 0 && !XG_Stripped ){
		fprintf( fp, "*INTERVAL* %d\n", this_set->plot_interval );
	}
	if( this_set->adorn_interval> 0 ){
		fprintf( fp, "*ADORN_INT* %d\n",
			(XG_Stripped && this_set->plot_interval)? this_set->adorn_interval/ this_set->plot_interval : this_set->adorn_interval
		);
	}
	switch( this_set->arrows ){
		case 1:
			fprintf( fp, "%s", arrow_types[this_set->arrows] );
			if( this_set->sarrow_orn_set ){
				fprintf( fp, " %s", d2str( this_set->sarrow_orn, NULL, NULL) );
			}
			fputs( "\n", fp );
			break;
		case 2:
			fprintf( fp, "%s", arrow_types[this_set->arrows] );
			if( this_set->earrow_orn_set ){
				fprintf( fp, " %s", d2str( this_set->earrow_orn, NULL, NULL) );
			}
			fputs( "\n", fp );
			break;
		case 3:
			fprintf( fp, "%s", arrow_types[this_set->arrows] );
			if( this_set->sarrow_orn_set ){
				fprintf( fp, " %s", d2str( this_set->sarrow_orn, NULL, NULL) );
				if( this_set->earrow_orn_set ){
					fprintf( fp, ",%s", d2str( this_set->earrow_orn, NULL, NULL) );
				}
			}
			else if( this_set->earrow_orn_set ){
				fprintf( fp, " NaN,%s", d2str( this_set->earrow_orn, NULL, NULL) );
			}
			fputs( "\n", fp );
			break;
		case 0:
			break;
		default:
			fprintf( StdErr, "Warning: set#%d->arrows==%d illegal value set to 0\n",
				this_set->set_nr, this_set->arrows
			);
			this_set->arrows= 0;
			break;
	}
	if( this_set->numAssociations && this_set->Associations ){
	  int i;
		if( this_set->numAssociations> ASCANF_MAX_ARGS ){
		  /* This should probably never happen! */
			fprintf( fp, "*EVAL* MaxArguments[%d] @\n", this_set->numAssociations );
			Ascanf_AllocMem( this_set->numAssociations );
		}
		  /* 20031013: use ad2str() instead of d2str(). */
		fprintf( fp, "*ASSOCIATE* %s", ad2str( this_set->Associations[0], d3str_format, NULL) );
		for( i= 1; i< this_set->numAssociations; i++ ){
			fprintf( fp, ",%s", ad2str( this_set->Associations[i], d3str_format, NULL) );
		}
		fputc( '\n', fp );
	}
	if( this_set->set_info ){
	  char *Buf= XGstrdup( this_set->set_info), *buf= Buf, *c;
	  int n;
		fprintf( fp, "*SET_INFO_LIST*" );
		while( xtb_getline( &buf, &c ) ){
			if( !*c ){
			  /* empty line that would terminate the list when reading. A single whitespace
			   \ will prevent this.
			   */
				fputs( " \n", fp );
			}
			else{
				fprintf( fp, "%s\n", c );
			}
		}
		if( (n= DumpDiscarded_Points( fp, wi, this_set, 0, LMAXBUFSIZE, NULL )) ){
			fprintf( fp, " discarded; %d of %d points (automatically added %s)\n", n, this_set->numPoints, today() );
		}
		fputs( "\n", fp );
		xfree( Buf );
	}
	if( this_set->average_set && !wi->dump_average_values ){
		fprintf( fp, "*AVERAGE* %s\n", this_set->average_set );
		if( newline ){
		  /* output the newline otherwise output after the last datapoint	*/
			fputs( "\n", fp );
		}
	}
}

int XGDump_PrintPars= 1, Init_XG_Dump= True, XGDump_Labels= True, XGDump_AllWindows= False;
int XG_Really_Incomplete= False;
char *XGDump_AllWindows_Filename= NULL;

extern char *ascanf_ProcedureCode();

#if defined(__MACH__) || (defined(linux) && !defined(_XOPEN_SOURCE))
char *cuserid(const char *dum)
{ struct passwd *pwd;
	if( (pwd= getpwuid( getuid())) ){
		return( pwd->pw_name );
	}
	else{
		return( "" );
	}
}
#endif

void _Dump_Arg0_Info( LocalWin *wi, FILE *fp, char **pinfo, int add_hist )
{ int i;
  char *CommandLine, *buf= NULL, bits[64];
  extern char ExecTime[256], XGraphBuildString[];
/* 	if( !fp ){	*/
/* 		fp= StdErr;	*/
/* 	}	*/
	if( (CommandLine= cgetenv( "XGRAPHCOMMAND" )) ){
		buf= concat2( buf, " ", CommandLine, "\n", NULL );
	}
	{
	  char *c= rindex( Argv[0], '/' );
		buf= concat2( buf, " ", (c)? &c[1] : Argv[0], NULL );
		for( i= 1; i< Argc; i++ ){
		  char *c= Argv[i];
		  int has_white= 0;
			while( c && *c){
				if( isspace(*c) || index("[]{}*!?()", *c) ){
					has_white+= 1;
				}
				c++;
			}
			if( has_white ){
				buf= concat2( buf, " '", Argv[i], "'", NULL );
			}
			else{
				buf= concat2( buf, " ", Argv[i], NULL );
			}
		}
	}
#if defined(linux) || defined(__CYGWIN__)
	{ char *on= NULL, *en= NULL;
		if( !isatty( fileno(stdin) ) ){
			buf= concat2( buf, " <", get_fdName( fileno(stdin) ), NULL );
		}
		if( !isatty( fileno(StdErr) ) ){
			buf= concat2( buf, " 2>", (en= get_fdName(fileno(StdErr))), NULL );
		}
		if( !isatty( fileno(stdout) ) ){
			buf= concat2( buf, " >", (on= get_fdName(fileno(stdout))), NULL );
		}
		if( add_hist ){
			add_process_hist( buf );
		}
		if( en ){
			buf= concat2( buf, "  ## sh/bash notation; escape as necessary!", NULL );
		}
	}
#else
	if( add_hist ){
		add_process_hist( buf );
	}
#endif
	sprintf( bits, " (%d bit) ", sizeof(void*) * 8 );
	if( XGraphBuildString ){
		buf= concat2( buf, "\n XGraph build", bits, XGraphBuildString, "\n", NULL );
	}
	if( ExecTime ){
		buf= concat2( buf, " Executed d.d. ", ExecTime, "\n", NULL );
	}
	if( InFilesTStamps ){
		buf= concat2( buf, " Input files: ", InFilesTStamps, "\n", NULL );
	}
	else if( InFiles ){
		buf= concat2( buf, " Input files: ", InFiles, "\n", NULL );
	}
	{ char lbuf[1024], *c= getcwd( lbuf, 1024);
		if( !c ){
			c= (char*) serror();
		}
		buf= concat2( buf, " Working directory: ", c, "\n", NULL );
	}
	{ extern char *VisualClass[];
	  char lbuf[512];
	  Visual *v= (wi && wi->visual)? wi->visual : vis;
		sprintf( lbuf, " Last opened on display %s/%d/%d (visual 0x%lx, %d planes %s) by %s\n",
			DisplayString(disp),
			(wi)? wi->dev_info.area_w : -1, (wi)? wi->dev_info.area_h : -1,
			v->visualid, depth, VisualClass[v->class], cuserid(NULL)
		);
		StringCheck( lbuf, sizeof(lbuf)/sizeof(char), __FILE__, __LINE__ );
		buf= concat2( buf, lbuf, NULL);
	}
#ifdef XG_DYMOD_SUPPORT
	{ DyModLists *libs= DyModList;
	  char *DyModTypeString( DyModTypes type );
		while( libs ){
			buf= concat2( buf, " Module \"",
				(libs->libname)? libs->libname : "", "\", loaded as \"",
				(libs->name)? libs->name : "", ",\" from \"",
				(libs->path)? libs->path : "", "\"; type ",
				(libs->typestring)? libs->typestring : DyModTypeString( libs->type),
				"; ",
				(libs->buildstring)? libs->buildstring : "(no build info)",
				(libs->auto_loaded)? " (auto-loaded)" : "",
				"\n", NULL
			);
			libs= libs->cdr;
		}
	}
#endif
	buf= concat2( buf, "\n", NULL);
	if( fp ){
		fputs( buf, fp );
	}
	if( pinfo ){
		*pinfo= buf;
	}
	else{
		xfree( buf );
	}
}

static int _XGDump_PlotSets( FILE* fp, LocalWin *wi)
{ int first, i, empty, n= 0;
	empty= 0;
	i= 0;
	  /* find first set that is drawn	*/
	while( ! draw_set(wi, i) && i< setNumber ){
		if( AllSets[i].numPoints<= 0 ){
			empty+= 1;
		}
		i+= 1;
	}
	first= i;
	  /* found the first; find the first that is not
	   \ drawn.
	   */
	while( i< setNumber && (draw_set(wi, i) || AllSets[i].numPoints<= 0) ){
/* 			if( AllSets[i].numPoints<= 0 ){	*/
/* 				empty+= 1;	*/
/* 			}	*/
		i+= 1;
	}
	  /* If the first drawn set is #0, and the last is setNumber-1,
	   \ then all sets are displayed, and we don't need to include a
	   \ selection command.
	   */
	if( first> 0 || i< setNumber ){
		i= first;
		if( i< setNumber && draw_set(wi, i) ){
		  int len;
			  /* Every empty (deleted) set is not saved, and therefore causes the setnumbers of
			   \ all its succesors to be decreased by 1.
			   */
			len= fprintf( fp, "*ARGUMENTS* -plot_only_set %d", (i++) - empty );
			n= 1;
			  /* find and write following sets that are drawn.	*/
			while( i< setNumber ){
				if( draw_set(wi, i) ){
					if( len>= LMAXBUFSIZE-16 ){
						fputs( "\n", fp);
						len= fprintf( fp, "*ARGUMENTS* -plot_only_set %d", i- empty );
					}
					else{
						len+= fprintf( fp, ",%d", i- empty );
					}
					n+= 1;
				}
				else if( AllSets[i].numPoints<= 0 ){
					empty+= 1;
				}
				i+= 1;
			}
			fputs( "\n", fp );
		}
		else if( AllSets[i].numPoints<= 0 ){
			empty+= 1;
		}
	}
	else{
		n= setNumber;
	}
	if( n== 0 ){
		fprintf( fp, "*ARGUMENTS* -plot_only_set -1\n" );
	}
	if( empty ){
		fprintf( fp, "# %d empty sets were found\n", empty );
	}
	return( n );
}

static int _XGDump_MarkedSets( FILE* fp, LocalWin *wi, char *prefix )
{ int i, empty= 0, n= 0;
	i= 0;
	  /* find first set that is marked	*/
	while( wi->mark_set[i]<= 0 && i< setNumber ){
		if( AllSets[i].numPoints<= 0 ){
			empty+= 1;
		}
		i+= 1;
	}
	if( i< setNumber && wi->mark_set[i]> 0 ){
	  int len;
		if( AllSets[i].numPoints<= 0 ){
			empty+= 1;
		}
		if( prefix ){
			fputs( prefix, fp );
		}
		len= fprintf( fp, "*ARGUMENTS* -mark_set %d", (i++) - empty );
		  /* find and write following sets that are marked.	*/
		while( i< setNumber ){
			if( AllSets[i].numPoints<= 0 ){
				empty+= 1;
			}
			if( wi->mark_set[i]> 0 ){
				if( len>= LMAXBUFSIZE-16 ){
					fputs( "\n", fp);
					if( prefix ){
						fputs( prefix, fp );
					}
					len= fprintf( fp, "*ARGUMENTS* -mark_set %d", i - empty );
				}
				else{
					len+= fprintf( fp, ",%d", i - empty );
				}
				n+= 1;
			}
			i+= 1;
		}
		fputs( "\n", fp );
	}
	if( empty ){
		fprintf( fp, "# %d empty sets were found\n", empty );
	}
	return(n);
}

void _XGDump_MoreArguments( FILE *fp, LocalWin *wi )
{ int idx;

	if( debugFlag ){
		fprintf( fp, "*EXTRATEXT* PostScript Settings:\n");
		{ char *xpos[]= { "left", "centre", "right"}, *ypos[]= { "bottom", "centre", "top"};
		  extern int showpage;
			if( wi->print_orientation ){
				fprintf( fp, "            scale %g, page pos=%s,%s%s\n",
					wi->ps_scale, xpos[wi->ps_ypos], ypos[wi->ps_xpos], (showpage)? ", do showpage" : "" );
			}
			else{
				fprintf( fp, "            scale %g, page pos=%s,%s%s\n",
					wi->ps_scale, xpos[wi->ps_xpos], ypos[wi->ps_ypos], (showpage)? ", do showpage" : "" );
			}
		}
	}
	if( XGDump_PrintPars ){
		fprintf( fp, "\n%s", (XGDump_PrintPars==2)? "\n*ARGUMENTS* -XGDump_PrintPars1\n" : "" );
	}
	fprintf( fp, "*ARGUMENTS* -geometry %dx%d", wi->dev_info.area_w, wi->dev_info.area_h );
	if( wi->dev_info.resized== 1 ){
		fprintf( fp, "\n*ARGUMENTS* -print_sized" );
	}
	if( wi->aspect_ratio!= 0 ){
		fprintf( fp, " -win_aspect %s", d2str( wi->aspect_ratio, NULL, NULL) );
	}
	fprintf( fp, " -spax%d", scale_plot_area_x );
	fprintf( fp, " -spay%d", scale_plot_area_y );
	fputc( '\n', fp );
	fprintf( fp, "*ARGUMENTS* -maxWidth %g -maxHeight %g",
		wi->hard_devices[PS_DEVICE].dev_max_width, wi->hard_devices[PS_DEVICE].dev_max_height
	);
	fputc( '\n', fp );
	fprintf( fp, "*ARGUMENTS* -hl_mode %d -hl_pars %s", highlight_mode, d2str( highlight_par[0], "%.15g", NULL) );
	for( idx= 1; idx< highlight_npars; idx++ ){
		fprintf( fp, ",%s", d2str( highlight_par[idx], "%.15g", NULL) );
	}
	if( wi->AlwaysDrawHighlighted ){
		fprintf( fp, " -hl_too%d", wi->AlwaysDrawHighlighted );
	}
	fputc( '\n', fp );
	fprintf( fp,
		"# PostScript Settings:\n"
		"*ARGUMENTS* %s -ps_scale %s -ps_xpos %d -ps_ypos %d -ps_offset %s,%s -ps_rgb%d -ps_transp%d -ps_fest %g "
		"-gs_twidth%d -gs_twidth_auto%d -ps_eps%d -ps_dsc%d -ps_setpage%d %gx%g\n",
		(wi->print_orientation)? "-Landscape" : "-Portrait",
		d2str( wi->ps_scale, "%.15g", NULL), wi->ps_xpos, wi->ps_ypos,
		d2str( wi->ps_l_offset, "%.15g", NULL), d2str( wi->ps_b_offset, "%.15g", NULL),
		ps_coloured, ps_transparent, Font_Width_Estimator, use_gsTextWidth, auto_gsTextWidth,
		psEPS, psDSC, psSetPage, psSetPage_width, psSetPage_height
	);
	if( strlen(XG_PS_NUp_buf) ){
		fprintf( fp, "\n## Current setting for PostScript NUp printing (mod-click on the AllWin button in the hardcopy dialog):\n");
 		fprintf( fp, "## '%s'\n\n", XG_PS_NUp_buf );
	}
	fprintf( fp,
		"*ARGUMENTS* -ps_tf \"%s\" -ps_tf_size %g -ps_lef \"%s\" -ps_lef_size %g\n"
		"*ARGUMENTS* -ps_laf \"%s\" -ps_laf_size %g -ps_af \"%s\" -ps_af_size %g\n",
		wi->hard_devices[PS_DEVICE].dev_title_font, wi->hard_devices[PS_DEVICE].dev_title_size,
		wi->hard_devices[PS_DEVICE].dev_legend_font, wi->hard_devices[PS_DEVICE].dev_legend_size,
		wi->hard_devices[PS_DEVICE].dev_label_font, wi->hard_devices[PS_DEVICE].dev_label_size,
		wi->hard_devices[PS_DEVICE].dev_axis_font, wi->hard_devices[PS_DEVICE].dev_axis_size
	);
	fprintf( fp,
		"*ARGUMENTS* -tf \"%s\" -lef \"%s\"\n"
		"*ARGUMENTS* -laf \"%s\" -af \"%s\"\n",
		titleFont.name, legendFont.name, labelFont.name, axisFont.name
	);
	fprintf( fp, "*ARGUMENTS* -bar_legend_dimension %s,%s,%s\n",
		d2str( wi->bar_legend_dimension_weight[0], 0,0),
		d2str( wi->bar_legend_dimension_weight[1], 0,0),
		d2str( wi->bar_legend_dimension_weight[2], 0,0)
	);
	if( psm_changed && !XGDump_PrintPars ){
		fprintf( fp, "Markersizes have been changed:\n\n*ARGUMENTS* -PSm %g,%g\n\n", psm_base, psm_incr+ 1 );
	}
	else{
		fprintf( fp, "*ARGUMENTS* -PSm %g,%g\n\n", psm_base, psm_incr+ 1 );
		fputc( '\n', fp);
	}
}

void Sprint_LabelsList( Sinc *sink, LabelsList *llist, char *header )
{ int instring= -2;
	if( header ){
		Sputs( header, sink );
	}
	while( llist ){
		Sputs( d2str( llist->column, "%g",NULL), sink );
		Sputc( ',', sink );
		  /* llist->label is a string that should be printed with escaped non-printables: pass instring==-2 */
		Sprint_string_string( sink, NULL, NULL, llist->label, &instring );
/* 		Sputs( llist->label, sink );	*/
		Sputc( '\n', sink );
		if( llist->min!= llist->max ){
			llist++;
		}
		else{
			llist= NULL;
		}
	}
}

void Sprint_SetLabelsList( Sinc *sink, LabelsList *llist, char *header, char *trailer )
{ int tablevel= 0, instring= -2;
	if( header ){
		Sputs( header, sink );
	}
	while( llist ){
		while( llist->column> tablevel ){
			Sputc( '\t', sink );
			tablevel+= 1;
		}
		  /* llist->label is a string that should be printed with escaped non-printables: pass instring==-2 */
		Sprint_string_string( sink, NULL, NULL, llist->label, &instring );
/* 		Sputs( llist->label, sink );	*/
		if( llist->min!= llist->max ){
			llist++;
		}
		else{
			llist= NULL;
		}
	}
	if( trailer ){
		Sputs( trailer, sink );
	}
}

void _XGDump_Labels( FILE *fp, LocalWin *wi )
{ Sinc sink;
	if( wi->DumpProcessed && !wi->raw_display ){
		print_string( fp, "*XLABEL*", "\\n\n", "\n", wi->tr_XUnits );
		print_string( fp, "*YLABEL*", "\\n\n", "\n", wi->tr_YUnits );
	}
	else{
		print_string( fp, "*XLABEL*", "\\n\n", "\n", wi->XUnits );
		if( strcmp( wi->XUnits, wi->tr_XUnits ) ){
			print_string( fp, "*XLABEL_TRANS*", "\\n\n", "\n", wi->tr_XUnits );
		}
		print_string( fp, "*YLABEL*", "\\n\n", "\n", wi->YUnits );
		if( strcmp( wi->YUnits, wi->tr_YUnits ) ){
			print_string( fp, "*YLABEL_TRANS*", "\\n\n", "\n", wi->tr_YUnits );
		}
	}
	Sprint_LabelsList( Sinc_file( &sink, fp, 0,0), wi->ColumnLabels, "\n*COLUMNLABELS* new\n" );
	fputc( '\n', fp );
}

void Dump_CustomFont( FILE *fp, CustomFont *cf, char *header )
{
	fflush( fp );
	if( header ){
		fputs( header, fp );
	}
	else{
		fprintf( fp, "#*CustomFont-%s* new\n", cf->PSFont );
	}
	fprintf( fp, "PS=%s\n", cf->PSFont );
	fprintf( fp, "PSSize=%g\n", cf->PSPointSize );
	fprintf( fp, "PSReEncode=%u\n", cf->PSreencode );
	fprintf( fp, "X=%s\n", cf->XFont.name );
	if( cf->alt_XFontName ){
		fprintf( fp, "X2=%s\n", cf->alt_XFontName );
	}
	fprintf( fp, "\n" );
}

static void _XGDump_ValCats( FILE *fp, LocalWin *wi )
{
	if( wi->ValCat_XFont ){
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *VAL_CAT_X_FONT* ...");
			XG_XSync( disp, False );
		}
		Dump_CustomFont( fp, wi->ValCat_XFont, "*VAL_CAT_X_FONT* new\n" );
	}
	if( wi->ValCat_X ){
	  ValCategory *vcat= wi->ValCat_X;
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *VAL_CAT_X* ...");
			XG_XSync( disp, False );
		}
		  /* Dump a *new* VAL_CAT_X statement. It could be interesting to see
		   \ what happens when the new is left out, and thus merging files will
		   \ merge categories..
		   */
		fprintf( fp, "*VAL_CAT_X* new\n" );
		while( vcat ){
			fprintf( fp, "%s,%s\n", dd2str( wi, vcat->val, NULL, NULL), vcat->category );
			if( vcat->min!= vcat->max ){
				vcat++;
			}
			else{
				vcat= NULL;
			}
		}
		fprintf( fp, "\n" );
	}
	if( wi->ValCat_YFont ){
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *VAL_CAT_Y_FONT* ...");
			XG_XSync( disp, False );
		}
		Dump_CustomFont( fp, wi->ValCat_YFont, "*VAL_CAT_Y_FONT* new\n" );
	}
	if( wi->ValCat_Y ){
	  ValCategory *vcat= wi->ValCat_Y;
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *VAL_CAT_Y* ...");
			XG_XSync( disp, False );
		}
		fprintf( fp, "*VAL_CAT_Y* new\n" );
		while( vcat ){
			fprintf( fp, "%s,%s\n", dd2str( wi, vcat->val, NULL, NULL), vcat->category );
			if( vcat->min!= vcat->max ){
				vcat++;
			}
			else{
				vcat= NULL;
			}
		}
		fprintf( fp, "\n" );
	}
	if( wi->ValCat_IFont ){
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *VAL_CAT_I_FONT* ...");
			XG_XSync( disp, False );
		}
		Dump_CustomFont( fp, wi->ValCat_IFont, "*VAL_CAT_I_FONT* new\n" );
	}
	if( wi->ValCat_I ){
	  ValCategory *vcat= wi->ValCat_I;
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *VAL_CAT_I* ...");
			XG_XSync( disp, False );
		}
		fprintf( fp, "*VAL_CAT_I* new\n" );
		while( vcat ){
			fprintf( fp, "%s,%s\n", dd2str( wi, vcat->val, NULL, NULL), vcat->category );
			if( vcat->min!= vcat->max ){
				vcat++;
			}
			else{
				vcat= NULL;
			}
		}
		fprintf( fp, "\n" );
	}
}

int _Dump_ascanf_CustomFont( FILE *fp, ascanf_Function *af )
{ int len;
	fflush(fp);
	len= fprintf( fp, "*EVAL* CustomFont[&%s%c\"%s\"%c\"%s\"%c%s%c%d",
		af->name, ascanf_separator,
		af->cfont->XFont.name, ascanf_separator, af->cfont->PSFont, ascanf_separator,
		ad2str( af->cfont->PSPointSize, d3str_format,0), ascanf_separator, af->cfont->PSreencode
	);
	if( af->cfont->alt_XFontName ){
		len+= fprintf( fp, "%c\"%s\"", ascanf_separator, af->cfont->alt_XFontName );
	}
	len+= fprintf( fp, "] @\n" );
	return(len);
}

static void _XGDump_PointerVariables( FILE *fp, ascanf_Function_type type, int *pointers )
{ /* Save potentially existing pointer variables:	*/
  int f= 0, output= False;
  extern ascanf_Function vars_ascanf_Functions[];
  extern int ascanf_Functions;
  ascanf_Function *af= vars_ascanf_Functions[f].cdr;
  int rootvar= 0, header= 0, l= 0, len= 0, n= 0, tit= 0;
	/* OnePerLine: whether we want 1 declaration per EVAL line	*/
  int OnePerLine= 1;
  char sepstr[2], sepstrq[3];
	if( !type ){
		fprintf( fp, "*EXTRATEXT* Declaring variables, pointers, etc. that are pointers (strings), or are used as such\n"
			" It can be that they should have been declared before; this will lead to unexpected behaviour\n"
			" In that case, move their declaration manually...\n"
			" Pointers to procedures are omitted at this time.\n\n"
		);
	}
	else{
		fprintf( fp, "*EXTRATEXT* Declaring variables, pointers, etc. that are pointers to procedures, or are used as such\n"
			" It can be that they should have been declared before; this will lead to unexpected behaviour\n"
			" In that case, move their declaration manually...\n\n"
		);
	}
	sepstr[0] = ascanf_separator, sepstr[1] = '\0';
	sepstrq[0] = ascanf_separator, sepstrq[1] = '"', sepstrq[2] = '\0';
	if( !af ){
		f+= 1;
		af= &vars_ascanf_Functions[f];
		rootvar= 1;
	}
	while( af ){
		if( (af->type== _ascanf_variable || af->type== _ascanf_array )
			&& !(af->name[0]== '$' && !af->dollar_variable)
			// RJVB 20081202: cease dumping variables from auto-loaded modules
			&& !(af->dymod && af->dymod->auto_loaded)
		){
		  int ok= 0, take_usage;
		  ascanf_Function *pf;
			len= 0;
			if( OnePerLine ){
				if( HO_Dialog.win && !tit ){
					XStoreName( disp, HO_Dialog.win, "Dumping ascanf pointer variables ...");
					XG_XSync( disp, False );
					tit= 1;
				}
			}
			else{
				if( !header ){
					if( HO_Dialog.win && !tit ){
						XStoreName( disp, HO_Dialog.win, "Dumping ascanf pointer variables ...");
						XG_XSync( disp, False );
					}
					len= fprintf( fp, "*EVAL* ");
					fflush(fp);
					header= 1;
				}
				else{
					len+= fputc( ascanf_separator, fp);
				}
			}
			pf= parse_ascanf_address( af->value, 0, "_XGDump_PointerVariables", 0, &take_usage);
			if( af->type== _ascanf_variable && (af->is_address || pf ) ){
				l= 0;
				if( pf ){
					if( type== 0 && pf->type== _ascanf_procedure ){
						if( OnePerLine ){
							l+= fprintf( fp, "# Delaying declaration of pointer-to-procedure: " );
						}
						ok= 0;
					}
					else if( (type== 0 || (type== pf->type ))
						  // 20090412: added case pf->is_usage
						&& (!take_usage || pf->is_usage || pf->user_internal)
					){
						if( OnePerLine && l== 0 ){
							len= fprintf( fp, "*EVAL* ");
							fflush(fp);
							header= 1;
						}
						l+= fprintf( fp, "DCL[%c", (af->is_usage)? '`' : '&' );
						l+= print_string2( fp, "", sepstr, af->name, False );
						  /* 20020517: */
						l+= print_string2( fp, "", "", ad2str(af->value, d3str_format,0), False );
						*pointers-= 1;
						ok= 1;
					}
				}
				else if( type== 0 ){
					if( OnePerLine && l== 0 ){
						len= fprintf( fp, "*EVAL* ");
						fflush(fp);
						header= 1;
					}
					l+= fprintf( fp, "DCL[%c", (af->is_usage)? '`' : '&' );
					l+= print_string2( fp, "", sepstr, af->name, False );
					l+= fprintf( fp, "%s", ad2str( af->value, d3str_format, NULL) );
					*pointers-= 1;
					ok= 1;
				}
				if( l ){
					if( af->usage && !rootvar && ok && !af->dymod ){
						  /* 20000507: preserve codes in usage strings.	*/
						l+= print_string2( fp, sepstrq, "\"]", af->usage, True );
						n= ASCANF_MAX_ARGS;
						header= 1;
					}
					else{
						fputs( "]", fp );
						l+= 1;
						n+= 1;
					}
				}
				len+= l;
				ok= 1;
			}
			if( ok && ((n>= ASCANF_MAX_ARGS && header) || len+ l>= LMAXBUFSIZE || OnePerLine) ){
				if( OnePerLine ){
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
				}
				if( len ){
					fputs( " @", fp );
					fputc( '\n', fp );
					len+= 3;
				}
				if( af->cfont ){
					len= _Dump_ascanf_CustomFont( fp, af );
				}
				n= 0;
				header= 0;
				output= True;
			}
		}
		if( !(af= af->cdr) ){
			f+= 1;
			if( f< ascanf_Functions ){
				af= &vars_ascanf_Functions[f];
				rootvar= 1;
			}
		}
		else{
			rootvar= 0;
		}
		if( len>= new_local_buf_size ){
			new_local_buf_size= len+ 10;
			change_local_buf_size= True;
		}
	}
	if( output ){
		fputc( '\n', fp );
	}
}

static void _XGDump_Process( FILE *fp, LocalWin *wi, int always )
{ char asep = ascanf_separator;
	if( HO_Dialog.win ){
		XStoreName( disp, HO_Dialog.win, "Dumping *DATA_PROCESS* ...");
		XG_XSync( disp, False );
	}

	if( wi->process.description ){
		if( wi->DumpProcessed ){
			fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
		}
		print_string( fp, "*DATA_PROCESS_DESCRIPTION*", "\\n\n", "\n\n", wi->process.description );
	}
	if( wi->process.separator ){
		ascanf_separator = wi->process.separator;
		fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
	}
	 // 20090416: separated DRAW_BEFORE and DRAW_AFTER dumping: DRAW_BEFORE is executed 1st and thus could contain
	 // declarations used by code executed subsequently.
	if( wi->process.draw_before_len || always ){

		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *DRAW_BEFORE* ...");
			XG_XSync( disp, False );
		}

		if( wi->process.draw_before_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->process.draw_before, "\n" ) ){
				print_string( fp, "**DRAW_BEFORE**\n", NULL, " @\n*!DRAW_BEFORE*\n\n", wi->process.draw_before);
			}
			else{
				fprintf( fp, "**DRAW_BEFORE**\n" );
				Print_Form( fp, &wi->process.C_draw_before, 0, True, NULL, "\t", "\n", False );
				fputs( " @\n*!DRAW_BEFORE*\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*DRAW_BEFORE*\n" );
		}
	}
	fputc( '\n', fp );
	if( wi->process.data_init_len ){
		if( wi->DumpProcessed ){
			fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
		}
		if( strstr( wi->process.data_init, "\n" ) ){
			print_string( fp, "**DATA_INIT**\n", NULL, " @\n*!DATA_INIT*\n\n", wi->process.data_init);
		}
		else{
			fprintf( fp, "**DATA_INIT**\n" );
			Print_Form( fp, &wi->process.C_data_init, 0, True, NULL, "\t", "\n", False );
			fputs( " @\n*!DATA_INIT*\n", fp );
		}
	}
	else if( always ){
		fprintf( fp, "*DATA_INIT*\n\n" );
	}
	if( wi->process.data_before_len ){
		if( wi->DumpProcessed ){
			fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
		}
		if( strstr( wi->process.data_before, "\n" ) ){
			print_string( fp, "**DATA_BEFORE**\n", NULL, " @\n*!DATA_BEFORE*\n\n", wi->process.data_before);
		}
		else{
			fprintf( fp, "**DATA_BEFORE**\n" );
			Print_Form( fp, &wi->process.C_data_before, 0, True, NULL, "\t", "\n", False );
			fputs( " @\n*!DATA_BEFORE*\n", fp );
		}
	}
	else if( always ){
		fprintf( fp, "*DATA_BEFORE*\n\n" );
	}
	if( wi->DumpProcessed ){
		fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
	}
	if( wi->process.data_process_len ){
		if( strstr( wi->process.data_process, "\n" ) ){
			print_string( fp, "**DATA_PROCESS**\n", NULL, " @\n*!DATA_PROCESS*\n\n", wi->process.data_process);
		}
		else{
			fprintf( fp, "**DATA_PROCESS**\n" );
			Print_Form( fp, &wi->process.C_data_process, 0, True, NULL, "\t", "\n", False );
			fputs( " @\n*!DATA_PROCESS*\n", fp );
		}
	}
	else if( always ){
		fprintf( fp, "*DATA_PROCESS*\n\n" );
	}
	if( wi->process.data_after_len ){
		if( wi->DumpProcessed ){
			fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
		}
		if( strstr( wi->process.data_after, "\n" ) ){
			print_string( fp, "**DATA_AFTER**\n", NULL, " @\n*!DATA_AFTER*\n\n", wi->process.data_after);
		}
		else{
			fprintf( fp, "**DATA_AFTER**\n" );
			Print_Form( fp, &wi->process.C_data_after, 0, True, NULL, "\t", "\n", False );
			fputs( " @\n*!DATA_AFTER*\n", fp );
		}
	}
	else if( always ){
		fprintf( fp, "*DATA_AFTER*\n\n" );
	}
	if( wi->process.data_finish_len ){
		if( wi->DumpProcessed ){
			fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
		}
		if( strstr( wi->process.data_finish, "\n" ) ){
			print_string( fp, "**DATA_FINISH**\n", NULL, " @\n*!DATA_FINISH*\n\n", wi->process.data_finish);
		}
		else{
			fprintf( fp, "**DATA_FINISH**\n" );
			Print_Form( fp, &wi->process.C_data_finish, 0, True, NULL, "\t", "\n", False );
			fputs( " @\n*!DATA_FINISH*\n", fp );
		}
	}
	else if( always ){
		fprintf( fp, "*DATA_FINISH*\n\n" );
	}
	fputc( '\n', fp );
	if( wi->process.draw_after_len || always ){

		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *DRAW_AFTER* ...");
			XG_XSync( disp, False );
		}

		if( wi->process.draw_after_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->process.draw_after, "\n" ) ){
				print_string( fp, "**DRAW_AFTER**\n", NULL, " @\n*!DRAW_AFTER*\n\n", wi->process.draw_after);
			}
			else{
				fprintf( fp, "**DRAW_AFTER**\n" );
				Print_Form( fp, &wi->process.C_draw_after, 0, True, NULL, "\t", "\n", False );
				fputs( " @\n*!DRAW_AFTER*\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*DRAW_AFTER*\n" );
		}
		fputc( '\n', fp );
	}
	if( (wi->process.dump_before_len || wi->process.dump_after_len) || always ){

		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *DUMP_PROCESS* ...");
			XG_XSync( disp, False );
		}

		if( wi->process.dump_before_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->process.dump_before, "\n" ) ){
				print_string( fp, "**DUMP_BEFORE**\n", NULL, " @\n*!DUMP_BEFORE*\n\n", wi->process.dump_before);
			}
			else{
				fprintf( fp, "**DUMP_BEFORE**\n" );
				Print_Form( fp, &wi->process.C_dump_before, 0, True, NULL, "\t", "\n", False );
				fputs( " @\n*!DUMP_BEFORE*\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*DUMP_BEFORE*\n" );
		}
		if( wi->process.dump_after_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->process.dump_after, "\n" ) ){
				print_string( fp, "**DUMP_AFTER**\n", NULL, " @\n*!DUMP_AFTER*\n\n", wi->process.dump_after);
			}
			else{
				fprintf( fp, "**DUMP_AFTER**\n" );
				Print_Form( fp, &wi->process.C_dump_after, 0, True, NULL, "\t", "\n", False );
				fputs( " @\n*!DUMP_AFTER*\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*DUMP_AFTER*\n" );
		}
		fputc( '\n', fp );
	}
	if( (wi->process.enter_raw_after_len || wi->process.leave_raw_after_len) || always ){

		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *RAW_AFTER* ...");
			XG_XSync( disp, False );
		}

		if( wi->process.enter_raw_after_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->process.enter_raw_after, "\n" ) ){
				print_string( fp, "**ENTER_RAW_AFTER**\n", NULL, " @\n*!ENTER_RAW_AFTER*\n\n", wi->process.enter_raw_after);
			}
			else{
				fprintf( fp, "**ENTER_RAW_AFTER**\n" );
				Print_Form( fp, &wi->process.C_enter_raw_after, 0, True, NULL, "\t", "\n", False );
				fputs( " @\n*!ENABLE_RAW_AFTER*\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*ENTER_RAW_AFTER*\n" );
		}
		if( wi->process.leave_raw_after_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->process.leave_raw_after, "\n" ) ){
				print_string( fp, "**LEAVE_RAW_AFTER**\n", NULL, " @\n*!LEAVE_RAW_AFTER*\n\n", wi->process.leave_raw_after);
			}
			else{
				fprintf( fp, "**LEAVE_RAW_AFTER**" );
				Print_Form( fp, &wi->process.C_leave_raw_after, 0, True, NULL, "\t", "\n", False );
				fputs( " @\n*!LEAVE_RAW_AFTER*\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*LEAVE_RAW_AFTER*\n" );
		}
		fputc( '\n', fp );
	}
	if( (wi->transform.x_len || wi->transform.y_len) || always ){

		if( wi->transform.separator && wi->transform.separator != ascanf_separator ){
			ascanf_separator = wi->transform.separator;
			fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
		}

		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *TRANSFORM_{X,Y}* ...");
			XG_XSync( disp, False );
		}

		if( wi->transform.description ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			print_string( fp, "*TRANSFORM_DESCRIPTION*", "\\n\n", "\n\n", wi->transform.description );
		}
		if( wi->transform.x_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->transform.x_process, "\n" ) ){
				print_string( fp, "*TRANSFORM_X*", "\\n\n", "\n\n", wi->transform.x_process );
			}
			else{
				fprintf( fp, "*TRANSFORM_X* " );
				Print_Form( fp, &wi->transform.C_x_process, 0, True, NULL, "\t", NULL, False );
				fputs( "\n\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*TRANSFORM_X*\n" );
		}
		if( wi->transform.y_len ){
			if( wi->DumpProcessed ){
				fprintf( fp, "*EXTRATEXT* Dumping processed values: processes included for completeness:\n" );
			}
			if( strstr( wi->transform.y_process, "\n" ) ){
				print_string( fp, "*TRANSFORM_Y*", "\\n\n", "\n\n", wi->transform.y_process );
			}
			else{
				fprintf( fp, "*TRANSFORM_Y* " );
				Print_Form( fp, &wi->transform.C_y_process, 0, True, NULL, "\t", NULL, False );
				fputs( "\n\n", fp );
			}
		}
		else if( always ){
			fprintf( fp, "*TRANSFORM_Y*\n" );
		}
		fputc( '\n', fp );
	}
	if( ascanf_separator != asep ){
		ascanf_separator = asep;
		fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
	}
}

  /* Dump the processes that need an active window:	*/
static void _XGDump_WindowProcs( FILE *fp, LocalWin *wi, int always )
{ char asep = ascanf_separator;
	if( XGDump_AllWindows && XGDump_AllWindows_Filename && (wi->curs_cross.fromwin_process.process_len || always) ){
		if( wi->curs_cross.fromwin_process.separator ){
			ascanf_separator = wi->curs_cross.fromwin_process.separator;
			fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
		}
		if( ! wi->curs_cross.fromwin_process.process_len ){
			fprintf( fp, "*CROSS_FROMWIN_PROCESS*\n\n" );
		}
		else if( strstr( wi->curs_cross.fromwin_process.process, "\n" ) ){
#if 1
			fprintf( fp, "**CROSS_FROMWIN_PROCESS**\n%d:%d:%d::%s\n*!CROSS_FROMWIN_PROCESS*\n\n",
				wi->curs_cross.fromwin->parent_number, wi->curs_cross.fromwin->pwindow_number, wi->curs_cross.fromwin->window_number,
				wi->curs_cross.fromwin_process.process
			);
#else
			fprintf( fp, "**CROSS_FROMWIN_PROCESS*\n%d:%d:%d::%s\n\n",
				wi->curs_cross.fromwin->parent_number, wi->curs_cross.fromwin->pwindow_number, wi->curs_cross.fromwin->window_number,
				wi->curs_cross.fromwin_process.process
			);
#endif
		}
		else{
/* 			fprintf( fp, "*CROSS_FROMWIN_PROCESS*\\n\n%d:%d:%d::",	*/
			fprintf( fp, "**CROSS_FROMWIN_PROCESS*\n%d:%d:%d::",
				wi->curs_cross.fromwin->parent_number, wi->curs_cross.fromwin->pwindow_number, wi->curs_cross.fromwin->window_number
			);
			Print_Form( fp, &wi->curs_cross.fromwin_process.C_process, 0, True, NULL, "\t", NULL, False );
			fputs( "\n\n", fp );
		}
		if( ascanf_separator != asep ){
			ascanf_separator = asep;
			fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
		}
	}
}

/* 20040919: new function, also ignoring ul->draw_it. */
void _XGDump_Linked_UserLabels( FILE *fp, LocalWin *wi, int set_nr, DataSet *this_set, Boolean ignore_draw_it, int empty, int *set_links )
{ UserLabel *ul= wi->ulabel;
  char corr[128];
	while( ul ){
		if( ul->set_link== this_set->set_nr && (ul->draw_it || ignore_draw_it) ){
			if( this_set->set_nr- empty!= set_nr ){
				sprintf( corr, " (linked2 dumped set #%d)", set_nr );
			}
			else{
				corr[0]= '\0';
			}
			fprintf( fp, "*ULABEL* %s %s %s %s set=%d transform?=%d draw?=%d lpoint=%d vertical?=%d nobox?=%d lWidth=%g type=%s",
				d2str( ul->x1, d3str_format, NULL),
				d2str( ul->y1, d3str_format, NULL),
				d2str( ul->x2, d3str_format, NULL),
				d2str( ul->y2, d3str_format, NULL),
				this_set->set_nr- empty, ul->do_transform, ul->do_draw, ul->pnt_nr, ul->vertical, ul->nobox,
				ul->lineWidth, ULabelTypeNames[ul->type]
			);
			if( XGStoreColours ){
				fprintf( fp, " cname=\"%s\"",
					(ul->pixvalue< 0)? ul->pixelCName :
						(ul->pixlinked)? "linked" : "default"
				);
			}
			fprintf( fp, "%s\n%s\n\n",
				corr, ul->label
			);
			ul->draw_it= 0;
			*set_links-= 1;
		}
		ul= ul->next;
	}
}

static void _XGDump_UserLabels( FILE *fp, LocalWin *wi, Boolean ignore_draw_it, int empty )
{ UserLabel *ul= wi->ulabel;
	while( ul ){
		if( ul->draw_it || ignore_draw_it ){
		  int ulsetl;
			if( ul->set_link< -1 ){
				ulsetl= ul->set_link;
			}
			else{
				ulsetl= (ignore_draw_it)? ul->set_link- empty : -1;
			}
/* 			fprintf( fp, "*ULABEL* %s %s %s %s %d %d %d %d %d",	*/
			fprintf( fp, "*ULABEL* %s %s %s %s set=%d transform?=%d draw?=%d lpoint=%d vertical?=%d nobox?=%d lWidth=%g type=%s",
				d2str( ul->x1, d3str_format, NULL),
				d2str( ul->y1, d3str_format, NULL),
				d2str( ul->x2, d3str_format, NULL),
				d2str( ul->y2, d3str_format, NULL),
				ulsetl, ul->do_transform, ul->do_draw, ul->pnt_nr, ul->vertical, ul->nobox,
				ul->lineWidth, ULabelTypeNames[ul->type]
			);
			if( XGStoreColours ){
				fprintf( fp, " cname=\"%s\"",
					(ul->pixvalue< 0)? ul->pixelCName :
						(ul->pixlinked)? "linked" : "default"
				);
			}
			fprintf( fp, "\n%s\n\n",
				ul->label
			);
		}
		ul= ul->next;
	}
}

/* 20020901: VERSION_LISTs can contain empty line(s) when updated manually. The easiest way to 
 \ handle this in a way that no problems arise reading back in the dump is to spread such a list
 \ over multiple VERSION_LIST statements, a new one for each (first) empty line encountered. This
 \ will remove the empty lines from the read-back-in result, but "tant pis".
 */
static void _XGDump_VersionList( FILE *fp, char *version_list, char *header )
{  char *c, *d;
	if( !version_list || !fp ){
		return;
	}
	fprintf( fp, "\n*VERSION_LIST*\n" );
	if( header ){
		fputs( header, fp );
	}
	d= c= version_list;
	while( d && c && *c ){
	 int el;
		el= 0;
		  /* check for empty line(s): */
		if( (d= strstr( c, "\n\n")) ){
			while( d[1] && d[1]== '\n' ){
			  /* Find the next non-empty line (d is incremented at least once): */
				d++;
				el+= 1;
			}
			  /* and cut before it. */
			*d= '\0';
		}
		fprintf( fp, "%s", c );
		if( !index( c, '\n' ) ){
			fputs( "\n", fp );
		}
		if( d ){
			  /* restore the newline */
			*d= '\n';
			  /* and continue after it */
			c= &d[1];
			if( *c ){
				if( el<= 1 ){
					fputc( '\n', fp );
				}
				fprintf( fp, "# Skipped %d empty line(s)\n*VERSION_LIST*\n", el );
			}
		}
	}
}

int DumpKeyParams= False;

static void _XGDump_Finish( FILE *fp, LocalWin *wi )
{
	if( Init_XG_Dump ){
	  char asep = ascanf_separator;
		if( wi->next_include_file && XG_Stripped!= 2 ){
			fprintf( fp, "#This file is to be included in the next visualisation of this file:\n%s\n", wi->next_include_file);
		}

		if( wi->next_startup_exprs ){
		  XGStringList *ex= wi->next_startup_exprs;
			fprintf( fp, "#This/ese command(s) to be executed after startup in the next visualisation of this file:\n" );
			if( wi->next_startup_exprs->separator ){
				ascanf_separator = wi->next_startup_exprs->separator;
				fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
			}
			while( ex ){
/* 				print_string( fp, "**STARTUP_EXPR*\n", NULL, "\n", ex->text );	*/
				print_string( fp, "**STARTUP_EXPR**\n", NULL, " @\n*!STARTUP_EXPR*\n", ex->text );
				ex= ex->next;
			}
			fputc( '\n', fp );
		}
		if( wi->Dump_commands ){
		  XGStringList *ex= wi->Dump_commands;
			fprintf( fp, "#This/ese command(s) to be executed after startup in the next visualisation of this file:\n" );
			if( wi->Dump_commands->separator && wi->Dump_commands->separator != ascanf_separator ){
				ascanf_separator = wi->Dump_commands->separator;
				fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
			}
			while( ex ){
				print_string( fp, "", "\\n\n", "\n", ex->text );
				ex= ex->next;
			}
			fputc( '\n', fp );
		}
		  /* Commands specific for windows dumped in DumpProcessed mode:
		   \ they must allow to undo all previous commands..
		   */
		if( wi->DumpProcessed_commands && wi->DumpProcessed ){
		  XGStringList *ex= wi->DumpProcessed_commands;
			fprintf( fp, "#This/ese command(s) to be executed after startup in the next visualisation of this file:\n" );
			if( wi->DumpProcessed_commands->separator && wi->DumpProcessed_commands->separator != ascanf_separator ){
				ascanf_separator = wi->DumpProcessed_commands->separator;
				fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
			}
			while( ex ){
				print_string( fp, "", "\\n\n", "\n", ex->text );
				ex= ex->next;
			}
			fputc( '\n', fp );
		}

		if( ascanf_separator != asep ){
			ascanf_separator = asep;
			fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
		}

		if( DumpKeyParams ){
		  int len;
			fputs( "*ARGUMENTS* -DumpKeyEVAL1\n", fp );
			fflush(fp);
			len= ListKeyParamExpressions( fp, True );
			fputc( '\n', fp );
			if( len>= new_local_buf_size ){
				new_local_buf_size= len+ 10;
				change_local_buf_size= True;
			}
		}

	}
}

#define SH_NEEDS_PATH
/* Set DIRECT_XGRAPH_SCRIPT if it is ok to put the path to the xgraph binary
 \ directly in a shell header. Since no other options can be passed in this
 \ fashion, this is not a good idea if it is to be possible to specify a
 \ minimal bit-depth-of screen (-MinBitsPPixel). The XGraph startup script
 \ scans for this.
#ifdef sgi
#	define DIRECT_XGRAPH_SCRIPT
#endif
 */

int Whereis( char *Prog_Name, char *us, int len )
{ FILE *pfp;
  char cbuf[2048];
	us[0]= '\0';
	if( Prog_Name && *Prog_Name ){
		sprintf( cbuf, "which %s </dev/null", Prog_Name );
		if( progname ){
			strcat( cbuf, " # -progname" );
		}
	}
	else{
		sprintf( cbuf, "which %s </dev/null", (Argv[0] && Argv[0][0])? Argv[0] : "xgraph" );
	}
	if( disp && HO_Dialog.win ){
		XStoreName( disp, HO_Dialog.win, cbuf );
		XG_XSync( disp, False );
	}
	PIPE_error= False;
	if( (pfp= popen( cbuf, "r")) ){
		fgets( us, len, pfp);
		pclose(pfp);
		if( strstr( us, "not found") ){
			us[0]= '\0';
		}
	}
	return( strlen(us) );
}

int Update_LMAXBUFSIZE( int update, char *errmsg )
{
	if( change_local_buf_size || (new_local_buf_size!= LMAXBUFSIZE && new_local_buf_size> 0) ){
	  int nlbs= new_local_buf_size;
		if( new_local_buf_size % 128 ){
			new_local_buf_size= (new_local_buf_size/ 128 + 1) * 128;
		}
		if( update ){
			LMAXBUFSIZE= new_local_buf_size;
			change_local_buf_size= False;
		}
		else if( errmsg ){
			sprintf( errmsg,
				"*BUFLEN* (line buffer length) needs to be at least %d bytes.\n"
				"It will be changed from %d to %d (be)for(e) the next dump.\n"
				"You may wish to dump once more to generate a correct file.\n", nlbs, LMAXBUFSIZE, new_local_buf_size
			);
			if( new_local_buf_size > 16*1024 ){
				strcat( errmsg, "(Warning: huge line-buffersize. Making a backup or saving to another file is suggested!)\n" );
			}
			fputs( errmsg, StdErr );
		}
		return(1);
	}
	return(0);
}

int DumpDHex= False, DumpPens= False, DProcFresh= True;
extern int DumpAsAscanf;

char *dd2str( LocalWin *wi, double x, const char *format, char **Buf )
{ ascanf_Function *saf;
  int take_usage= 0;
	if( wi->DumpAsAscanf && (saf= parse_ascanf_address( x, 0, "dd2str", 0, &take_usage)) && take_usage ){
		return( ad2str( x, format, Buf ) );
	}
	else{
		return( d2str( x, format, (Buf)? *Buf : NULL ) );
	}
}

void SetBinaryDumpData( BinaryField *field, int elem, double value, Extremes *extr )
{
	switch( BinarySize ){
		case sizeof(float):
			field->data4[elem]= (float) value;
			break;
		case sizeof(unsigned short):
			field->data2[elem]= (unsigned short) ( (value-extr[elem].min)/(extr[elem].max-extr[elem].min) * USHRT_MAX );
			break;
		case sizeof(unsigned char):
			field->data1[elem]= (unsigned char) ( (value-extr[elem].min)/(extr[elem].max-extr[elem].min) * UCHAR_MAX );
			break;
		default:
			fprintf( StdErr,
				"SetBinaryDumpData(): unsupported BinarySize==%d,"
				" assuming 'double'\n",
				BinarySize
			);
			  /* fall-through */
		case sizeof(double):
			field->data[elem]= value;
			break;
	}
}

/* 20031029: first version of a routine that finds and dumps the current set's per-column extremes.
 \ This is necessary in the future planned version for very (many) large datasets, in which the
 \ data are stored as unsigned shorts (necessitating a mapping of the 64K onto the real values).
 */
void DumpExtremes( LocalWin *wi, FILE *fp, DataSet *this_set, int start, Boolean *bin_active, Extremes *extr )
{ int i, j, first= True;
  unsigned long N= 0;

	if( *bin_active ){
		BinaryTerminate(fp);
		fputc( '\n', fp );
		*bin_active= False;
	}
	for( i= 0; i< this_set->ncols; i++ ){
		set_Inf( extr[i].min, 1 );
		set_Inf( extr[i].max, -1);
	}
	for( j= start; j< this_set->numPoints; j++ ){
		if( !DiscardedPoint( wi, this_set, j) ){
			if( this_set->plot_interval<= 0 || (XG_Stripped!=1 || (j % this_set->plot_interval)==0) ){
			  double v;
				for( i= 0; i< this_set->ncols; i++ ){
					if( wi->DumpProcessed ){
						if( i== this_set->xcol ){
							v= this_set->xvec[j];
						}
						else if( i== this_set->ycol ){
							v= this_set->yvec[j];
						}
						else if( i== this_set->ecol ){
							if( !wi->transform.y_len /* wi->vectorFlag	*/ ){
								v= this_set->errvec[j];
							}
							else{
								v= (this_set->hdyvec[j]- this_set->ldyvec[j])/2;
							}
						}
						else if( i== this_set->lcol ){
							if( !wi->transform.x_len /* wi->vectorFlag	*/ ){
								v= this_set->lvec[j];
							}
							else{
								v= (this_set->hdxvec[j]- this_set->ldxvec[j])/2;
							}
						}
						else{
							v= this_set->columns[i][j];
						}
					}
					else{
						v= this_set->columns[i][j];
					}
					if( Check_Ignore_NaN( *SS_Ignore_NaN, v) ){
						if( first ){
							extr[i].min= extr[i].max= v;
							first= False;
						}
						else{
							if( v< extr[i].min ){
								extr[i].min= v;
							}
							else if( v> extr[i].max ){
								extr[i].max= v;
							}
						}
					}
				}
				N+= 1;
			}
		}
	}
	if( N ){
	  int len;
		len= fprintf( fp, "*EXTREMES* %s,%s",
			dd2str( wi, extr[0].min, d3str_format, NULL), dd2str( wi, extr[0].max, d3str_format, NULL)
		);
		for( i= 1; i< this_set->ncols; i++ ){
			len+= fprintf( fp, ", %s,%s",
				dd2str( wi, extr[i].min, d3str_format, NULL), dd2str( wi, extr[i].max, d3str_format, NULL)
			);
		}
		fputc( '\n', fp ); len+= 1;
		if( len>= new_local_buf_size ){
			if( debugFlag ){
				fprintf( StdErr, "*EXTREMES* dump: extended line %d>=%d, current BUFLEN=%d\n",
					len, new_local_buf_size, LMAXBUFSIZE
				);
			}
			new_local_buf_size= len+ 10;
			change_local_buf_size= True;
		}
	}
}

int _XGDump_WhichLabels( LocalWin *wi )
{ UserLabel *ul= wi->ulabel;
  int nr;
  int set_links= 0;
	while( ul ){
		if( (nr= ul->set_link)>= 0 ){
			if( AllSets[nr].numPoints> 0 && (!XG_Stripped || draw_set(wi,nr)) ){
				set_links+= 1;
				ul->draw_it= 1;
			}
			else{
				ul->draw_it= 0;
			}
		}
		else{
			ul->draw_it= 1;
		}
		ul= ul->next;
	}
	return(set_links);
}

  /* make a dump in the XGraph ascii format	*/
int _XGraphDump( LocalWin *wi, FILE *fp, char errmsg[ERRBUFSIZE] )
{  int idx, j, i, start, pointers= 0, linkedArrays= 0;
   DataSet *this_set, *prev_set;
   char cbuf[2048], *_Title= NULL;
   mode_t mask;
   int ugi= use_greek_inf;
   int points= 0, NumObs= 0, nsets= 0, set_nr= 0, set_links= 0;
   Boolean bin_active= False, allow_short_bin= False, next_new_file= False;
   char *prev_fileName= NULL, *AllWinVar= NULL;
   int bdc, empty, PR= PS_STATE(wi)->Printing, xgs= XG_Stripped, wdp= wi->DumpProcessed;
   int AF= Allow_Fractions, dph= d2str_printhex;
   LocalWin *AW= ActiveWin;

	Allow_Fractions= False;

	if( DumpDHex ){
		d2str_printhex= True;
	}

	if( !change_local_buf_size || new_local_buf_size<= 0 ){
		new_local_buf_size= LMAXBUFSIZE;
	}

	  /* This gives us the current umask:	*/
	mask= umask(0);
	umask(mask);
	  /* now do a chmod +x of the current file:	*/
	fchmod( fileno(fp), (0777-mask) | S_IXUSR | S_IXGRP );

	  /* 20010722: set ActiveWin to the window being dumped: */
	ActiveWin= wi;
	PS_STATE(wi)->Printing= XG_DUMPING;

	if( XGDump_AllWindows ){
	  /* Dump all sets when all windows are to be dumped. It would be more elegant to
	   \ only dump the sets being drawn in any of the windows, but much more work
	   \ to implement too.
	   */
		XG_Stripped= 0;
		  /* Also, we can't dump the cooked results, because this might mess up
		   \ the data which can be window-dependent.
		   */
		wi->DumpProcessed= 0;
	}

	if( HO_Dialog.win ){
		XFetchName( disp, HO_Dialog.win, &_Title );
	}
	if( Init_XG_Dump ){
#ifdef SH_NEEDS_PATH
	 { char us[2048];
	   static char *location= NULL;
		  /* /bin/sh needs a full path specification to access XGraph,
		   \ the script needed to start xgraph by "executing" an XGraph dump
		   */
		us[0]= '\0';
		if( !location ){
			Whereis( Prog_Name, us, 1023 );
			location= XGstrdup( us );
		}
		else{
			strcpy( us, location );
		}
#	ifndef DIRECT_XGRAPH_SCRIPT
		if( us[0] ){
		  char *c;
			if( debugFlag ){
				fprintf( StdErr, "_XGraphDump(): argv[0]=%s; us=%s using path ", Argv[0], us );
			}
		  /* We expect XGraph in the same directory as ourselves (Argv[0])	*/
			if( (c= rindex( us, '/')) ){
				*c= '\0';
				if( debugFlag ){
					fprintf( StdErr, "%s\n", us );
				}
/* 				fprintf( fp, "#!/bin/sh %s/GXraph\n", us );	*/
				fprintf( fp, "#!%s/GXraph -uniq\n", us );
				*c= '/';
			}
			else{
			  char cwd[MAXPATHLEN+2];
			  /* No full path - let's assume we're started up in the current directory	*/
				getcwd(cwd, MAXPATHLEN+1 );
				if( debugFlag ){
					fprintf( StdErr, "%s\n", cwd );
				}
/* 				fprintf( fp, "#!/bin/sh %s/GXraph\n", cwd );	*/
				fprintf( fp, "#!%s/GXraph -uniq\n", cwd );
			}
		}
		else{
/* 			fprintf( fp, "#!/bin/sh /usr/local/bin/GXraph\n" );	*/
			fprintf( fp, "#!/usr/local/bin/GXraph -uniq\n" );
		}
#	else
#error Obsolete functionality!
		if( us[0] ){
			if( debugFlag ){
				fprintf( StdErr, "_XGraphDump(): argv[0]=%s; us=%s using directly as script-header ", Argv[0], us );
			}
			fprintf( fp, "#!%s\n", us );
		}
		else{
			fprintf( fp, "#!/usr/local/bin/xgraph\n" );
		}
#	endif
	 }
#else
		  /* /bin/sh is clever enough to find XGraph when it is in our path (and executable)	*/
		fprintf( fp, "#!/bin/sh GXraph\n" );
#endif
		fflush( fp );
		fprintf( fp, "# Generated: %s", PrintTime);
		if( XG_preserved_filetime ){
			fprintf( fp, "# Preserved: %s\n", XG_preserved_filetime );
		}
		else{
			fputs( "\n", fp );
		}
		fflush( fp );
		fprintf( fp, "# vim:ts=4:sw=4:nowrap:\n" );
		fflush( fp );

		{ int n, rsacsv= reset_ascanf_currentself_value;
			if( wi->process.dump_before_len ){
				clean_param_scratch();
				n= param_scratch_len;
				*ascanf_self_value= 0.0;
				*ascanf_current_value= 0.0;
				*ascanf_counter= (*ascanf_Counter)= 0.0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "_XGraphDump(): DUMP Before: %s", wi->process.dump_before);
					fflush( StdErr );
				}
				if( HO_Dialog.win ){
					XStoreName( disp, HO_Dialog.win, "Executing *DUMP_BEFORE*...");
					XG_XSync( disp, False );
				}
				TBARprogress_header= "*DUMP_BEFORE*";
				ascanf_arg_error= 0;
				compiled_fascanf( &n, wi->process.dump_before, param_scratch, NULL, data, column, &wi->process.C_dump_before );
				TBARprogress_header= NULL;
				TitleMessage( wi, NULL );
			}
			reset_ascanf_currentself_value= rsacsv;
		}
	}

	if( wi->DumpBinary && MaxCols> BinaryDump.columns ){
		if( !AllocBinaryFields( MaxCols, "_XGraphDump()" ) ){
			wi->DumpBinary= False;
		}
	}
	bdc= BinaryDump.columns;

	use_greek_inf= 0;

	if( Init_XG_Dump || !XG_Really_Incomplete ){
	  extern char *WhatEndian(int);
/* 		fprintf( fp, "*ENDIAN*%d\n", EndianType );	*/
		fprintf( fp, "*ENDIAN* %s\n", WhatEndian(EndianType) );
		fprintf( fp, "*BUFLEN* %d\n", LMAXBUFSIZE );

		  /* 20010929: dump printfile name, too. */
		if( UPrintFileName[0] || PrintFileName ){
		  char buf[1024], *c= getcwd( buf, 1024);
			if( !c ){
				c= (char*) serror();
			}
			fprintf( fp, "\n*ARGUMENTS* -pf %s\n", (UPrintFileName[0])? UPrintFileName : PrintFileName );
			fprintf( fp, "*EXTRATEXT*\n"
				"Printing file name specified through %s;\n"
				"Current working directory is %s\n\n",
					(UPrintFileName[0])? "Print Dialog" : "-pf commandline option",
					c
			);
		}

		if( XGDump_AllWindows && XGDump_AllWindows_Filename ){
			if( !(AllWinVar= XGtempnam( getenv("TMPDIR"), "XGAWD")) ){
				AllWinVar= strdup("XGAWD");
			}
			fprintf( fp,
				"\n*CONDITIONAL_EXTRATEXT* !AND[ActiveWin,Defined?[%s]]\n"
				"*SKIP_TO* Restore_All_Windows\n\n",
				AllWinVar
			);
		}

		{ Boolean describe= False;
		  int i;
			if( wi->DumpProcessed ){
				if( wi->process.description ){
					describe= True;
				}
				else if( wi->transform.description ){
					describe= True;
				}
				else{
					for( i= 0; i< setNumber && !describe; i++ ){
						if( (!XG_Stripped || draw_set(wi, i)) && AllSets[i].process.description ){
							describe= True;
						}
					}
				}
			}
			if( wi->version_list || describe ){
			  char *c;
				_XGDump_VersionList( fp, wi->version_list, NULL );
				if( describe ){
					_XGDump_VersionList( fp, wi->process.description, "Performed processing information: " );
					_XGDump_VersionList( fp, wi->process.description, "Performed transformation information: " );
					for( i= 0; i< setNumber; i++ ){
						if( (!XG_Stripped || draw_set( wi, i)) && (c= AllSets[i].process.description) ){
						  char hdr[256];
							snprintf( hdr, sizeof(hdr), "Performed set #%d processing information: ", i );
							_XGDump_VersionList( fp, AllSets[i].process.description, hdr );
						}
					}
				}
				fputs( "\n\n\n", fp );
			}
		}
	}
	if( Init_XG_Dump ){
		if( d3str_format_changed ){
			fprintf( fp, "*DPRINTF*%s\n", d3str_format );
		}
		if( AxisValueFormat ){
			fprintf( fp, "*AXVAL_DPRINTF*%s\n", AxisValueFormat );
		}
		fprintf( fp, "*AXVAL_MINDIGITS*%d\n", AxisValueMinDigits );
		fflush( fp );
	}

	if( wi->DumpProcessed && DProcFresh ){
		CheckProcessUpdate(wi, False, False, False );
	}

	if( Init_XG_Dump ){
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Collecting commandline arguments...");
			XG_XSync( disp, False );
		}

		Collect_Arguments( wi, cbuf, 2048);
		if( strlen( cbuf) ){
		  extern int ReadData_Outliers_Warn;
			  /* Lines starting with *EXTRATEXT* are ignored: all following lines up to
			   \ the first empty line are stored internally as comments (in the Info box).
			   */
			fputs( "*EXTRATEXT* Some commandline options...\n\n", fp );
			fprintf( fp, "*ARGUMENTS* %s -fli0 -warnOutliers%d", cbuf, ReadData_Outliers_Warn );
			fputc( '\n', fp);
		}
		if( !XG_SaveBounds ){
			fputs( "\n## Uncomment the following line to restore the axes bounds as they were when this file was saved:\n", fp );
			fprintf( fp, "# *ARGUMENTS*%s\n\n", XGBoundsString );
		}
		fprintf( fp, "*ARGUMENTS* -Cxye -legendtype %d -radix %g -radix_offset %g", wi->legend_type, wi->radix, wi->radix_offset );
		fprintf( fp, " -tcl%d -tas%d", Determine_tr_curve_len, Determine_AvSlope );
		  /* If the Allow_Fractions flag was changed, save it	*/
		if( changed_Allow_Fractions ){
			fprintf( fp, " -frac%d", (AF)? 1 : 0 );
		}
		fprintf( fp, " -splits_disconnect%d", splits_disconnect );
		fputs( "\n", fp );
	}

	if( HO_Dialog.win ){
		XStoreName( disp, HO_Dialog.win, "Saving plot_only collection...");
		XG_XSync( disp, False );
	}
	  /* If all sets are being saved, save info on the selection currently
	   \ being displayed.
	   */
	if( setNumber && !XG_Stripped ){
		_XGDump_PlotSets(fp, wi);
	}
	  /* Save which sets are marked	*/
	if( HO_Dialog.win ){
		XStoreName( disp, HO_Dialog.win, "Saving marked collection...");
		XG_XSync( disp, False );
	}
	if( Init_XG_Dump || !XG_Really_Incomplete ){
		if( setNumber ){
			_XGDump_MarkedSets(fp, wi, "#" );
		}
	}

	if( HO_Dialog.win ){
		XStoreName( disp, HO_Dialog.win, "Collecting additional settings...");
		XG_XSync( disp, False );
	}

	  /* We must set the gamma correction flag before any colours are allocated,
	   \ and even when Init_XG_Dump=False since it may affect how set colours are
	   \ allocated.
	   */
	if( Init_XG_Dump || !XG_Really_Incomplete ){
		fprintf( fp, "*EVAL* $AllowGammaCorrection[%g] @\n", *AllowGammaCorrection );
	}

	if( Init_XG_Dump ){
	  Boolean do_ah= False, do_proc= False;
	  char sepstr[2], sepstrq[3];

		sepstr[0] = ascanf_separator, sepstr[1] = '\0';
		sepstrq[0] = ascanf_separator, sepstrq[1] = '"', sepstrq[2] = '\0';

		fprintf( fp, "*ARGUMENTS* -bw %d -aw %g -lw %g -ew %g\n", bdrSize, axisWidth, lineWidth, errorWidth );
		fprintf( fp, "*ARGUMENTS* -gw %g -zw %g -gp %x", gridWidth, zeroWidth, gridLS[0] );
			for( i= 1; i< gridLSLen; i++ ){
				fprintf( fp, "%x", gridLS[i] );
			}
			fputs( "\n\n", fp );

		if( XGStoreColours ){
			fputs( "\n", fp );
			fputs( "*ARGUMENTS* -DumpColours", fp );
			if( reverseFlag ){
				fputs( " -rv", fp );
			}
			if( XGIgnoreCNames ){
				fputs( " -IgnoreCNames", fp );
			}
			fprintf( fp, "\n*GLOBAL_COLOURS*\n"
				"black=%s\n"
				"white=%s\n"
				"bg=%s\n"
				"fg=%s\n"
				"bdr=%s\n"
				"zero=%s\n"
				"grid=%s\n"
				"hl=%s\n\n",
					blackCName,
					whiteCName,
					bgCName,
					normCName,
					bdrCName,
					zeroCName,
					gridCName,
					highlightCName
			);
		}
		fputs( "\n", fp );

		if( PSSetupIncludes ){
		  XGStringList *li= PSSetupIncludes;
		  int n= -1;
			if( li->text ){
				fprintf( fp, "*ARGUMENTS*" );
				n+= 1;
			}
			while( li ){
				if( li->text ){
					fprintf( fp, " -ps_s_inc %s", li->text );
					n+= 1;
				}
				li= li->next;
			}
			if( n>= 0 ){
				fputc( '\n', fp );
			}
			fputc( '\n', fp );
		}

		_XGDump_MoreArguments( fp, wi );

		if( wi->show_overlap ){
		  int op= 0;
			if( wi->overlap_buf && strlen(wi->overlap_buf) ){
				op= fprintf( fp, "*EXTRATEXT* %s\n", wi->overlap_buf );
			}
			if( wi->overlap2_buf && strlen(wi->overlap2_buf) ){
				fputs( (op)? "           " : "*EXTRATEXT*", fp );
				op+= fprintf( fp, " %s\n", wi->overlap2_buf );
			}
			if( op ){
				fputc( '\n', fp );
			}
		}
		if( PS_PrintComment && strlen( comment_buf ) ){
		  char *Buf, *Buf2, *buf= XGstrdup( comment_buf ), *buf2= ShowLegends( wi, False, -1 ), *c;

			if( HO_Dialog.win ){
				XStoreName( disp, HO_Dialog.win, "Dumping Info buffer...");
				XG_XSync( disp, False );
			}

			fprintf( fp, "*EXTRATEXT* Info Buffer:\n");
			Buf= buf;
			Buf2= buf2;
			while( buf){
				while( xtb_getline( &buf, &c) ){
					if( strlen( c ) ){
						fprintf( fp, "%s\n", c );
					}
					else{
						if( c== buf || c== buf2 ){
							fprintf( fp, "\n\n*EXTRATEXT* %s Buffer:\n", (c==buf)? "Info" : "Legend" );
						}
						else{
							fprintf( fp, "\n*EXTRATEXT* %s Buffer:\n", (c==buf)? "Info" : "Legend" );
						}
					}
				}
				buf= buf2;
				buf2= NULL;
			}
			fputc( '\n', fp);
			xfree( Buf );
			xfree( Buf2 );
		}
		if( titleText2 && strlen( titleText2 ) ){
			fprintf( fp, "*ARGUMENTS* -T \"%s\"\n\n", titleText2 );
		}
		_XGDump_Labels( fp, wi );
		fputc( '\n', fp );

		if( ASCANF_MAX_ARGS && ASCANF_MAX_ARGS!= AMAXARGSDEFAULT ){
			fputs( "# Make sure to restore the actual non-default number of ascanf arguments\n", fp );
			fprintf( fp, "*EVAL* MaxArguments[%d] @\n\n", ASCANF_MAX_ARGS );
		}

#ifdef XG_DYMOD_SUPPORT
		if( !wi->DumpProcessed ){
		  DyModLists *list= DyModList;
		  int dump= 0;
			if( list ){
				if( !list->auto_loaded && !list->no_dump){
					dump+= 1;
				}
				  /* go down the list, updating the car field */
				list->car= NULL;
				while( list->cdr ){
					list->cdr->car= list;
					list= list->cdr;
					if( !list->auto_loaded ){
						dump+= 1;
					}
				}
				if( dump ){
					fprintf( fp, "*LOAD_MODULE*\n" );
					list= DyModList;
					while( list ){
						if( !list->auto_loaded && !list->no_dump){
							fprintf( fp, "%s\n", list->name );
						}
						else if( debugFlag ){
							fprintf( StdErr, "Skipping LOAD_MODULE %s because of auto_loaded=%d no_dump=%d\n",
								   list->name, list->auto_loaded, list->no_dump );
						}
						list= list->car;
					}
					fputc( '\n', fp );
				}
			}
		}
#endif

		if( wi->init_exprs ){
		  XGStringList *list= wi->init_exprs;
		  char asep = ascanf_separator;
			if( HO_Dialog.win ){
				XStoreName( disp, HO_Dialog.win, "Dumping *INIT expressions ...");
				XG_XSync( disp, False );
			}
			if( list->separator ){
				ascanf_separator = list->separator;
				fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
			}
			fprintf( fp, "*INIT_BEGIN* new\n" );
			while( list ){
				fputs( list->text, fp );
				list= list->next;
			}
			fprintf( fp, "*INIT_END*\n" );
			if( ascanf_separator != asep ){
				ascanf_separator = asep;
				fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
			}
		}

		if( ascanf_memory ){
		  int header= 0, len= 0, l= 0;
			for( idx= 0; idx< ASCANF_MAX_ARGS; idx++ ){
				if( ascanf_memory[idx] ){
					if( !header ){
						if( HO_Dialog.win ){
							XStoreName( disp, HO_Dialog.win, "Dumping MEM array ...");
							XG_XSync( disp, False );
						}

						len= fprintf( fp, "*EVAL*" );
						fflush( fp );
						header= 1;
					}
					else{
						len+= fprintf( fp, "%c", ascanf_separator );
					}
					l= fprintf( fp, "MEM[%d%c%s]", idx, ascanf_separator, ad2str( ascanf_memory[idx], d3str_format, NULL) );
					len+= l;
					if( len+ l>= LMAXBUFSIZE ){
						fputs( "\n", fp );
						header= 0;
					}
				}
			}
			if( header ){
				fputs( "\n\n", fp);
				fflush(fp);
			}
		}
		{ extern double ***ascanf_mxyz_buf;
		  extern int ascanf_mxyz_X, ascanf_mxyz_Y, ascanf_mxyz_Z;
		  int i, j, k, header= 0, len= 0, l= 0, n;
		  char msg[128];
			if( ascanf_mxyz_X && ascanf_mxyz_Y && ascanf_mxyz_Z ){
				fprintf( fp, "*EVAL* SETMXYZ[%d%c%d%c%d]\n",
					ascanf_mxyz_X, ascanf_separator, ascanf_mxyz_Y, ascanf_separator, ascanf_mxyz_Z
				);
				for( i= 0; i< ascanf_mxyz_X; i++ ){
					header= 0;
					n= 0;
					for( j= 0; j< ascanf_mxyz_Y; j++ ){
						for( k= 0; k< ascanf_mxyz_Z; k++ ){
							if( ascanf_mxyz_buf[i][j][k] ){
								if( !header ){
									sprintf( msg, "Dumping %dx%dx%d MXYZ array non-zero item [%d,%d,%d]",
										ascanf_mxyz_X, ascanf_mxyz_Y, ascanf_mxyz_Z, i, j, k
									);
									if( HO_Dialog.win ){
										XStoreName( disp, HO_Dialog.win, msg);
										XG_XSync( disp, False );
									}
									len= fprintf( fp, "*EVAL*" );
									fflush( fp );
									header= 1;
								}
								else{
									len+= fputc( ascanf_separator, fp);
								}
								l= fprintf( fp, "MXYZ[%d%c%d%c%d%c%s]",
									i, ascanf_separator, j, ascanf_separator, k, ascanf_separator,
									ad2str( ascanf_mxyz_buf[i][j][k], d3str_format, NULL )
								);
								len+= l;
								n+= 1;
								if( (n>= ASCANF_MAX_ARGS && header) || len+ l>= LMAXBUFSIZE ){
									if( l+len>= new_local_buf_size ){
										if( debugFlag ){
											fprintf( StdErr, "MXYZ dump: extended line %d>=%d, current BUFLEN=%d\n",
												l+len, new_local_buf_size, LMAXBUFSIZE
											);
										}
										new_local_buf_size= l+len+ 10;
										change_local_buf_size= True;
									}
									fputs( "\n", fp );
									fflush(fp);
									n= 0;
									header= 0;
								}
							}
						}
					}
					if( header ){
						fputs( "\n", fp);
						fflush(fp);
					}
				}
				if( header ){
					fputs( "\n\n", fp);
					fflush( fp);
				}
			}
		}
		{ extern double **ascanf_mxy_buf;
		  extern int ascanf_mxy_X, ascanf_mxy_Y;
		  int i, j, header= 0, len= 0, l= 0, n;
		  char msg[128];
			if( ascanf_mxy_X && ascanf_mxy_Y ){
				fprintf( fp, "*EVAL* SETMXY[%d%c%d]\n", ascanf_mxy_X, ascanf_separator, ascanf_mxy_Y );
				for( i= 0; i< ascanf_mxy_X; i++ ){
					header= 0;
					n= 0;
					for( j= 0; j< ascanf_mxy_Y; j++ ){
						if( ascanf_mxy_buf[i][j] ){
							if( !header ){
								sprintf( msg, "Dumping %dx%d MXY array non-zero item [%d,%d]",
									ascanf_mxy_X, ascanf_mxy_Y, i, j
								);
								if( HO_Dialog.win ){
									XStoreName( disp, HO_Dialog.win, msg);
									XG_XSync( disp, False );
								}
								len= fprintf( fp, "*EVAL*" );
								fflush( fp );
								header= 1;
							}
							else{
								len+= fputc( ascanf_separator, fp);
							}
							l= fprintf( fp, "MXY[%d%c%d%c%s]",
								i, ascanf_separator, j, ascanf_separator,
								ad2str( ascanf_mxy_buf[i][j], d3str_format, NULL )
							);
							len+= l;
							n+= 1;
							if( (n>= ASCANF_MAX_ARGS && header) || len+ l>= LMAXBUFSIZE ){
								if( l+len>= new_local_buf_size ){
									if( debugFlag ){
										fprintf( StdErr, "SETMXY dump: extended line %d>=%d, current BUFLEN=%d\n",
											l+len, new_local_buf_size, LMAXBUFSIZE
										);
									}
									new_local_buf_size= l+len+ 10;
									change_local_buf_size= True;
								}
								fputs( "\n", fp );
								fflush(fp);
								n= 0;
								header= 0;
							}
						}
					}
					if( header ){
						fputs( "\n", fp);
						fflush(fp);
					}
				}
				if( header ){
					fputs( "\n\n", fp);
					fflush( fp);
				}
			}
		}
		
		{ int f= 0;
		  ascanf_Function *af= vars_ascanf_Functions[f].cdr;
		  int rootvar= 0, header= 0, len= 0, l= 0, n= 0, tit= 0;
		    /* OnePerLine: whether we want 1 declaration per EVAL line	*/
		  int OnePerLine= 1;

		    /* 20001115: modified the code as follows. Names can be strings, that should be
			 \ printed "correctly", ie. with non-printable characters properly escaped. The
			 \ print_string2() routine does this; it is now used to print the names. Takes
			 \ a few more function calls because it is of course not as flexible as fprintf()..
			 \ The old printing statements are preserved within comments for the time being.
			 */
			if( !af ){
				f+= 1;
				af= &vars_ascanf_Functions[f];
				rootvar= 1;
			}
			while( af ){
				len= 0;
				if( af->accessHandler && !do_ah ){
					do_ah= True;
				}
				if( af->type== _ascanf_array && af->linkedArray.dataColumn ){
					Check_linkedArray(af);
				}
				if( (af->type== _ascanf_variable || af->type== _ascanf_array ||
						af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats
						|| af->type== _ascanf_python_object
					)
					&& !(af->name[0]== '$' && !af->dollar_variable)
					// RJVB 20081202: cease dumping variables from auto-loaded modules
					&& !(af->dymod && af->dymod->auto_loaded)
				){
				  int ok= 1, totlen= 0;
					if( OnePerLine ){
						if( HO_Dialog.win && !tit ){
							XStoreName( disp, HO_Dialog.win, "Dumping ascanf variables ...");
							XG_XSync( disp, False );
							tit= 1;
						}
					}
					else{
						if( !header ){
							if( HO_Dialog.win && !tit ){
								XStoreName( disp, HO_Dialog.win, "Dumping ascanf variables ...");
								XG_XSync( disp, False );
							}
							len= fprintf( fp, "*EVAL* ");
							fflush( fp );
							header= 1;
						}
						else{
							len+= fputc( ascanf_separator, fp);
						}
					}
					if( af->type== _ascanf_variable || af->type== _ascanf_python_object ){
					  ascanf_Function *pf;
					  int take_usage;
						len= 0;
						if( af->is_address ||
							( (pf= parse_ascanf_address( af->value, 0, "_XGraphDump", 0, &take_usage)) &&
								(!take_usage || pf->user_internal)
							)
						){
							pointers+= 1;
							if( OnePerLine ){
								len= fprintf( fp, "# Delaying declaration of pointer variable: " );
								goto dump_var;
							}
							ok= 0;
						}
						else{
							if( OnePerLine ){
								len+= fprintf( fp, "*EVAL* ");
								fflush( fp );
								header= 1;
							}
dump_var:;
							l= print_string2( fp, "DCL[", sepstr, af->name, False );
							l+= fprintf( fp, "%s", ad2str( af->value, d3str_format, NULL) );
							if( af->usage && !rootvar && ok && !af->dymod ){
								  /* 20000507: preserve codes in usage strings.	*/
								l+= print_string2( fp, sepstrq, "\"]", af->usage, True );
								n= ASCANF_MAX_ARGS;
								header= 1;
							}
							else{
								fputs( "]", fp );
								l+= 1;
								n+= 1;
							}
							len+= l;
						}
					}
					else if( af->type== _ascanf_array && af->linkedArray.dataColumn ){
						linkedArrays+= 1;
						if( OnePerLine ){
							len= fprintf( fp, "# Delaying initialisation of linkedArray variable:\n" );
							len+= fprintf( fp, "*EVAL* ");
							fflush( fp );
							header= 1;
						}
						l= print_string2( fp, "DCL[", sepstr, af->name, False );
						l+= fprintf( fp, "-1,0" );
						if( af->usage && !rootvar && ok && !af->dymod ){
							l+= print_string2( fp, sepstrq, "\"]", af->usage, True );
							n= ASCANF_MAX_ARGS;
							header= 1;
						}
						else{
							fputs( "]", fp );
							l+= 1;
							n+= 1;
						}
						len+= l;
					}
					else if( af->type== _ascanf_array ){
					  int i, idx= 0, ind= -1, wrapped= 0, ilen= 0, nlen= 0;
					  ALLOCA( ibuf, char, strlen(af->name)+ 64, ibuf_len);
						ilen= sprintf( ibuf, "%sDCL[%s%d]", (wrapped)? "" : sepstr, sepstr, af->N );
						  /* 20010621: why not make use of the last_index field ?! */
						ind= af->last_index;
						if( af->iarray ){
							if( OnePerLine ){
								len= fprintf( fp, "*EVAL* ");
								fflush( fp );
								header= 1;
							}
							l= fprintf( fp, "DCL[" );
							l+= (nlen= print_string2( fp, "", "", af->name, False ));
							l+= fprintf( fp, "%c%d%c%d", ascanf_separator, af->N, ascanf_separator, af->iarray[0] );
							if( ind< 0 && ((int) af->value)== af->iarray[0] ){
								ind= 0;
							}
						}
						else{
						  ascanf_Function *saf= NULL;
						  int take_usage;
							for( i= 0; i< af->N && !saf; i++ ){
								if( (saf= parse_ascanf_address(af->array[i], 0, "_XGraphDump", False, &take_usage)) &&
									!(take_usage || saf->internal)
								){
									if( !OnePerLine ){
										fputc( '\n', fp );
									}
									fprintf( StdErr,
										"Warning: the array \"%s\" may contain references"
										" to not yet defined (string)variables!\n", af->name
									);
									fprintf( fp,
										"*ECHO*\n"
										"Warning: the following array \"%s\" may contain references"
										" to not yet defined (string)variables!\n\n",
											af->name
									);
									if( !OnePerLine ){
										len= fprintf( fp, "*EVAL* ");
										fflush( fp );
										header= 1;
									}
								}
								else{
									saf= NULL;
								}
							}
							if( OnePerLine ){
								len= fprintf( fp, "*EVAL* ");
								fflush( fp );
								header= 1;
							}
							l= fprintf( fp, "DCL[" );
							l+= (nlen= print_string2( fp, "", "", af->name, False ));
							if( (saf= parse_ascanf_address(af->array[0], 0, "_XGraphDump", False, &take_usage)) ){
								if( take_usage ){
									if( saf->internal && !saf->user_internal ){
										l+= fprintf( fp, "%c%d%c`\"%s\"",
											ascanf_separator, af->N, ascanf_separator, saf->usage
										);
									}
									else{
										l+= fprintf( fp, "%c%d%c`%s",
											ascanf_separator, af->N, ascanf_separator, saf->name
										);
									}
								}
								else{
									l+= fprintf( fp, "%c%d%c&%s", ascanf_separator, af->N, ascanf_separator, saf->name );
								}
							}
							else{
								l+= fprintf( fp, "%c%d%c%s",
									ascanf_separator, af->N, ascanf_separator, ad2str( af->array[0], d3str_format,0)
								);
							}
							if( ind< 0 && af->value== af->array[0] ){
								ind= 0;
							}
						}
#ifdef ORIG_ARRAY_USAGE_DUMP
						  /* 20010701: descriptions and labels are only taken into account when a variable is first declared. Thus,
						   \ for an array definition that can span several lines, and likely at least 2 DCL[] invocations
						   \ (the last to set the last_index field), we are obliged to do a first declaration that
						   \ allocates the storage, sets the 1st element, and assigns the description and/or label.
						   \ 20040514: descriptions dumped this way are not restored when the first element is a string itself.
						   */
						if( af->usage && !rootvar && !af->dymod ){
							  /* 20000507: preserve codes in usage strings.	*/
							l+= print_string2( fp, sepstrq, "\"]", af->usage, True );
							n= ASCANF_MAX_ARGS;
							header= 1;
						}
						else
#endif
						{
							fputs( "]", fp );
							l+= 1;
							n+= 1;
						}
						if( af->label ){
							fprintf( fp, " # label={%s}", af->label );
						}
						if( af->N> 1 ){
						  /* dump values from 1 onwards */
							fputs( " @\n", fp );
							len= fprintf( fp, "*EVAL* " );
							fflush( fp );
							header= 1;
							l= print_string2( fp, "DCL[", sepstr, af->name, False );
							l+= fprintf( fp, "%d", 1 );
							ilen= 0;
							for( i= 1; i< af->N; i++ ){
								if( af->iarray ){
									l+= fprintf( fp, "%c%d", ascanf_separator, af->iarray[i] );
									wrapped= 0;
									if( ind< 0 && ((int)af->value)== af->iarray[i] ){
										ind= i;
									}
								}
								else{
								  ascanf_Function *saf;
								  int take_usage;
									if( (saf= parse_ascanf_address(af->array[i], 0, "_XGraphDump", False, &take_usage)) ){
										if( take_usage ){
											if( saf->internal && !saf->user_internal ){
												l+= fprintf( fp, "%c`\"%s\"", ascanf_separator, saf->usage );
											}
											else{
												l+= fprintf( fp, "%c`%s", ascanf_separator, saf->name );
											}
										}
										else{
											l+= fprintf( fp, "%c&%s", ascanf_separator, saf->name );
										}
									}
									else{
										l+= fprintf( fp, "%c%s",
											ascanf_separator, ad2str( af->array[i], d3str_format, NULL)
										);
									}
									wrapped= 0;
									if( ind< 0 && af->value== af->array[i] ){
										ind= i;
									}
								}
								idx+= 1;
								if( len+l+ ilen+ 2>= LMAXBUFSIZE || idx== ASCANF_MAX_ARGS-1- 2 ){
									if( l+len+ ilen+ 2>= new_local_buf_size ){
										if( debugFlag ){
											fprintf( StdErr, "%s dump [%d]: extended line %d>=%d, current BUFLEN=%d\n",
												af->name, __LINE__,
												l+len+ilen+2, new_local_buf_size, LMAXBUFSIZE
											);
										}
										new_local_buf_size= l+len+ ilen+ 2+ 10;
										change_local_buf_size= True;
									}
									fputs( "] @\n", fp );
									idx= 0;
									totlen+= l+len+ilen+2;
									if( i< af->N-1 ){
										len= fprintf( fp, "*EVAL* " );
										fflush( fp );
										header= 1;
										l= print_string2( fp, "DCL[", sepstr, af->name, False );
										  /* 20010621: print the index for the first NEXT element, so that the
										   \ remaining elements get stored correctly.
										   */
										l+= fprintf( fp, "%d", i+1 );
									}
									else{
										l= 0;
										wrapped= 1;
										header= 0;
									}
								}
							}
							if( !wrapped ){
								l+= fprintf( fp, "]" );
							}
						}
						if( ind> 0 ){
						  /* Maybe we'd better make sure that the first call-without-arguments to this
						   \ array returns the possibly expected result...
						   */
							if( wrapped && !header ){
								len= fprintf( fp, "*EVAL* " );
								fflush( fp );
								header= 1;
							}
#define LAST_INDEX_ARRAY_20031002
#ifdef LAST_INDEX_ARRAY_20031002
							  /* simple read suffices to set last_index; there should be no need to store the argument once more... */
							l+= print_string2( fp, ((wrapped)? "" : sepstr), "[", af->name, False );
							l+= fprintf( fp, "%d", ind );
#else
/* 							l+= fprintf( fp, "%sDCL[%s%s%d", (wrapped)? "" : sepstr, af->name, sepstr, ind );	*/
							{ char decl[6];
								if( wrapped ){
									strcpy( decl, "DCL[" );
								}
								else{
									strcpy( decl, ",DCL[" );
									decl[0] = ascanf_separator;
								}
								l+= print_string2( fp, decl, sepstr, af->name, False );
							}
							l+= fprintf( fp, "%d", ind );
							if( af->iarray ){
								l+= fprintf( fp, "%c%d", ascanf_separator, af->iarray[ind] );
							}
							else{
							  ascanf_Function *saf;
							  int take_usage;
								if( (saf= parse_ascanf_address(af->array[ind], 0, "_XGraphDump", False, &take_usage)) ){
									if( take_usage ){
										if( saf->internal && !saf->user_internal ){
											l+= fprintf( fp, "%c`\"%s\"", ascanf_separator, saf->usage );
										}
										else{
											l+= fprintf( fp, "%c`%s", ascanf_separator, saf->name );
										}
									}
									else{
										l+= fprintf( fp, "%c&%s", ascanf_separator, saf->name );
									}
								}
								else{
									l+= fprintf( fp, "%c%s",
										ascanf_separator, ad2str( af->array[ind], d3str_format, NULL)
									);
								}
							}
#endif
							fputs( "]", fp );
							l+= 1;
							n+= 1;
						}
#ifndef ORIG_ARRAY_USAGE_DUMP
						if( af->usage && !rootvar && !af->dymod ){
						  char fnstr[16];
							  /* 20040514: preserve codes in usage strings.	*/
							if( wrapped && !header ){
								len= fprintf( fp, "*EVAL* " );
								fflush( fp );
								header= 1;
							}
							if( wrapped ){
								strcpy( fnstr, "printf['" );
							}
							else{
								strcpy( fnstr, ",printf[`" );
								fnstr[0] = ascanf_separator;
							}
							l+= print_string2( fp, fnstr, ",", af->name, False );
							l+= print_string2( fp, "\"", "\"]", af->usage, True );
							n= ASCANF_MAX_ARGS;
							header= 1;
						}
#endif
						len+= l;
						n+= 1;
					}
					else if( af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats ){
						len= 0;
						if( OnePerLine ){
							len+= fprintf( fp, "*EVAL* ");
							fflush( fp );
							header= 1;
						}
						l= print_string2( fp, "DCL[", sepstr, af->name, False );
						if( af->N ){
							l+= fprintf( fp, "%d%s%s", af->N, sepstr, ad2str( af->value, d3str_format, NULL) );
						}
						else{
							l+= fprintf( fp, "%s", ad2str( af->value, d3str_format, NULL) );
						}
						if( af->usage && !rootvar && ok && !af->dymod ){
							  /* 20000507: preserve codes in usage strings.	*/
							l+= print_string2( fp, sepstrq, "\"]", af->usage, True );
							n= ASCANF_MAX_ARGS;
							header= 1;
						}
						else{
							fputs( "]", fp );
							l+= 1;
							n+= 1;
						}
						if( af->type== _ascanf_simplestats ){
						  char fnstr[] = ", $SS_StatsBin[&";
							fnstr[0] = ascanf_separator;
							l+= print_string2( fp, fnstr, sepstr, af->name, False );
							l+= fprintf( fp, "%d ]", (af->SS)? af->SS->exact : 0 );
						}
						else{
						  char fnstr[] = ", $SAS_StatsBin[&";
							fnstr[0] = ascanf_separator;
							l+= print_string2( fp, fnstr, sepstr, af->name, False );
							if( af->SAS ){
								l+= fprintf( fp, "%d%c%s%c%s", af->SAS->exact, ascanf_separator,
									ad2str( af->SAS->Gonio_Base, d3str_format, 0), ascanf_separator,
									ad2str( af->SAS->Gonio_Offset, d3str_format, 0)
								);
							}
							l+= fprintf( fp, " ]" );
						}
						len+= l;
					}
					if( ok && ((n>= ASCANF_MAX_ARGS && header) || len>= LMAXBUFSIZE || OnePerLine) ){
						if( len>= new_local_buf_size ){
							if( debugFlag ){
								fprintf( StdErr, "%s dump [%d]: extended line %d>=%d, current BUFLEN=%d\n",
									af->name, __LINE__,
									len, new_local_buf_size, LMAXBUFSIZE
								);
							}
							new_local_buf_size= len+ 10;
							change_local_buf_size= True;
						}
						if( OnePerLine ){
							if( af->dymod && af->dymod->name && af->dymod->path ){
								len+= fprintf( fp, " # Module={\"%s\",%s}", af->dymod->name, af->dymod->path );
							}
							if( af->fp ){
								len+= fprintf( fp, " # open file fd=%d", fileno(af->fp) );
							}
							if( af->label && af->type!= _ascanf_array ){
								len+= fprintf( fp, " # label={%s}", af->label );
							}
							if( af->cfont ){
								len+= fprintf( fp, "  # cfont: %s\"%s\"/%s[%g]",
									(af->cfont->is_alt_XFont)? "(alt.) " : "",
									af->cfont->XFont.name, af->cfont->PSFont, af->cfont->PSPointSize
								);
							}
						}
						fputs( " @", fp );
						fputc( '\n', fp );
						len+= 3;
						totlen+= len;
						if( af->cfont ){
							len= _Dump_ascanf_CustomFont( fp, af );
							if( len> totlen ){
								totlen= len;
							}
						}
						n= 0;
						header= 0;
					}
					if( !totlen ){
						totlen= len;
					}
					if( totlen>= new_local_buf_size ){
						new_local_buf_size= totlen+ 10;
						change_local_buf_size= True;
					}
				}
				else if( af->type== _ascanf_procedure && !do_proc ){
					do_proc= True;
				}
				if( !(af= af->cdr) ){
					f+= 1;
					if( f< ascanf_Functions ){
						af= &vars_ascanf_Functions[f];
						rootvar= 1;
					}
				}
				else{
					rootvar= 0;
				}
				if( OnePerLine && len>= new_local_buf_size ){
					new_local_buf_size= len+ 10;
					change_local_buf_size= True;
				}
			}
			if( header || OnePerLine ){
				fputs( "\n", fp );
				fflush( fp);
			}
			if( header || OnePerLine ){
				fputs( "\n", fp );
				fflush( fp);
			}
		}
		if( pointers> 0 ){
			_XGDump_PointerVariables( fp, 0, &pointers );
		}
		if( do_proc ){
		  extern ascanf_Function vars_ascanf_Functions[];
		  extern int ascanf_Functions;
		  int f= 0, len= 0;
		  ascanf_Function *af= vars_ascanf_Functions[f].cdr;
		  int rootvar= 0, header= 0;
			if( !af ){
				f+= 1;
				af= &vars_ascanf_Functions[f];
				rootvar= 1;
			}
			while( af ){
				len= 0;
				if( (af->type== _ascanf_procedure) && !(af->name[0]== '$' && !af->dollar_variable)){
					if( !header ){
						if( HO_Dialog.win ){
							XStoreName( disp, HO_Dialog.win, "Dumping ascanf procedures ...");
							XG_XSync( disp, False );
						}
						header= 1;
					}
					  /* 20010208: NB: a copy of this code resides in ascanfc.c, the part that handles
					   \ EditPROC[].
					   */
					if( af->type== _ascanf_procedure ){
					  char *code= ascanf_ProcedureCode(af);
					  char *opcode= (af->dialog_procedure)? "ASKPROC" : "DEPROC-noEval";
					  char *trailer= NULL;
						if( code && strstr( code, "\n" )
							&& (!af->procedure_separator || af->procedure_separator == ascanf_separator)
						){
							len+= fprintf( fp, "**EVAL**\n%s[", opcode );
							fflush( fp );
							len+= print_string2( fp, "", sepstr, af->name, False );
							len+= print_string( fp, "", NULL, "", code );
							trailer= "\n*!EVAL*\n\n";
						}
						else{
							len+= fprintf( fp, "**EVAL**\n%s[", opcode );
							fflush( fp );
							len+= print_string2( fp, "", sepstr, af->name, False );
							len+= Print_Form( fp, &af->procedure, 0, True, NULL, "\t", "\n", False );
							trailer= "\n*!EVAL*\n\n";
						}
						if( af->usage && !rootvar && !af->dymod ){
							  /* 20000507: preserve codes in usage strings.	*/
							len+= print_string2( fp, sepstrq, "\"", af->usage, True );
						}
						fputs( "]", fp );
						if( af->dymod && af->dymod->name && af->dymod->path ){
							fprintf( fp, " # Module={\"%s\",%s}", af->dymod->name, af->dymod->path );
						}
						if( af->fp ){
							fprintf( fp, " # open file fd=%d", fileno(af->fp) );
						}
						if( af->cfont ){
							fprintf( fp, "  # cfont: %s\"%s\"/%s[%g]",
								(af->cfont->is_alt_XFont)? "(alt.) " : "",
								af->cfont->XFont.name, af->cfont->PSFont, af->cfont->PSPointSize
							);
						}
						if( af->label ){
							fprintf( fp, " # label={%s}", af->label );
						}
						fputs( " @", fp );
						if( trailer ){
							fputs( trailer, fp );
							len+= strlen(trailer);
						}
						else{
							fputs( "\n\n", fp );
							len+= 5;
						}
					}
				}
				if( !(af= af->cdr) ){
					f+= 1;
					if( f< ascanf_Functions ){
						af= &vars_ascanf_Functions[f];
						rootvar= 1;
					}
				}
				else{
					rootvar= 0;
				}
				if( len>= new_local_buf_size ){
					new_local_buf_size= len+ 10;
					change_local_buf_size= True;
				}
			}
			if( header ){
				fputc( '\n', fp );
			}
		}
		if( pointers ){
			if( pointers> 0 ){
				_XGDump_PointerVariables( fp, _ascanf_procedure, &pointers );
			}
			if( pointers ){
				fprintf( StdErr, "There remain%s %d pointer variables; this is probably a bug in XGraph!\n",
					(pointers>1)? "s" : "" , pointers
				);
				fprintf( fp, "*EXTRATEXT*\nThere remain%s %d pointer variables; this is probably a bug in XGraph!\n\n",
					(pointers>1)? "s" : "" , pointers
				);
			}
		}
		if( do_ah ){
		  extern ascanf_Function vars_ascanf_Functions[], internal_AHandler;
		  extern int ascanf_Functions;
		  int f= 0, len= 0;
		  ascanf_Function *af= vars_ascanf_Functions[f].cdr;
		  int rootvar= 0, header= 0;
			if( !af ){
				f+= 1;
				af= &vars_ascanf_Functions[f];
				rootvar= 1;
			}
			while( af ){
				len= 0;
				if( af->accessHandler && af->accessHandler!= &internal_AHandler
					&& !(af->dymod && af->dymod->auto_loaded)
				){
				  ascanf_Function *aH= af->accessHandler;
				  double *aH_par= af->aH_par;
					if( !header ){
						if( HO_Dialog.win ){
							XStoreName( disp, HO_Dialog.win, "Dumping ascanf AccessHandler information ...");
							XG_XSync( disp, False );
						}
						header= 1;
					}
					len+= fprintf( fp, "*EVAL* AccessHandler[" );
					len+= print_string2( fp, "&", sepstr, af->name, False );
					len+= print_string2( fp, "&", sepstr, aH->name, False );
					fflush(fp);
					switch( aH->type ){
						case _ascanf_variable:
						case _ascanf_array:
						case _ascanf_procedure:
						case NOT_EOF:
						case NOT_EOF_OR_RETURN:
						case _ascanf_function:{
							len+= fprintf( fp, "%s%c%s%c%d%c%s",
								ad2str( aH_par[0], d3str_format, NULL ), ascanf_separator,
								ad2str( aH_par[1], d3str_format, NULL), ascanf_separator,
								af->aH_flags[2], ascanf_separator,
								ad2str( aH_par[2], d3str_format, NULL)
							);
							break;
						}
						default:
							af->accessHandler= NULL;
							break;
					}
					fprintf( fp, "%s%d%s%d] @\n", sepstr, af->aH_flags[0], sepstr, af->aH_flags[1] );
				}
				if( !(af= af->cdr) ){
					f+= 1;
					if( f< ascanf_Functions ){
						af= &vars_ascanf_Functions[f];
						rootvar= 1;
					}
				}
				else{
					rootvar= 0;
				}
				if( len>= new_local_buf_size ){
					new_local_buf_size= len+ 10;
					change_local_buf_size= True;
				}
			}
			if( header ){
				fputc( '\n', fp );
			}
		}

		if( IntensityColourFunction.NColours ){
			if( HO_Dialog.win ){
				XStoreName( disp, HO_Dialog.win, "Dumping *INTENSITY_COLOURS* ...");
				XG_XSync( disp, False );
			}

			if( IntensityColourFunction.name_table ){
			  XGStringList *tab= IntensityColourFunction.name_table;
				fprintf( fp, "*INTENSITY_COLOUR_TABLE* new\n" );
				while( tab ){
					fprintf( fp, "%s\n", tab->text );
					tab= tab->next;
				}
				fputc( '\n', fp );
			}
			if( IntensityColourFunction.expression ){
				if( strstr( IntensityColourFunction.expression, "\n" ) ){
					print_string( fp, "*INTENSITY_COLOURS*", "\\n\n", "\n",
						IntensityColourFunction.expression
					);
				}
				else{
					fprintf( fp, "*INTENSITY_COLOURS* " );
					Print_Form( fp, &IntensityColourFunction.C_expression, 0, True, NULL, "\t", NULL, False );
					fputs( "\n", fp );
				}
				if( IntensityColourFunction.use_table ){
					fprintf( fp, "*INTENSITY_COLOUR_TABLE* use\n" );
				}
			}
			if( IntensityColourFunction.range_set ){
				fprintf( fp, "*INTENSITY_RANGE* %s%c%s\n",
					d2str( IntensityColourFunction.range.min, NULL, NULL), ascanf_separator,
					d2str( IntensityColourFunction.range.max, NULL, NULL)
				);
			}
			fputs( "\n", fp );
		}
		_XGDump_ValCats( fp, wi );

		_XGDump_Process( fp, wi, False );

		{ extern char **ascanf_SS_names;
		  extern char **ascanf_SAS_names;
		  int n= 0;
			if( ascanf_SS_names ){
				for( idx= 0; idx< ASCANF_MAX_ARGS; idx++ ){
					if( ascanf_SS_names[idx] ){
						fprintf( fp, "*ASS_NAME* %d::%s\n", idx, ascanf_SS_names[idx] );
						n++;
					}
				}
			}
			if( ascanf_SAS_names ){
				for( idx= 0; idx< ASCANF_MAX_ARGS; idx++ ){
					if( ascanf_SAS_names[idx] ){
						fprintf( fp, "*ASAS_NAME* %d::%s\n", idx, ascanf_SAS_names[idx] );
						n++;
					}
				}
			}
			if( n ){
				fputc( '\n', fp );
			}
		}
		  /* Here the global PROPERTIES (with 4 arguments) should be saved.	*/
		if( Xscale!= 1 || Yscale!= 1 || DYscale!= 1 ){
			fprintf( fp, "*SCALEFACT* %g %g %g\n", Xscale, Yscale, DYscale );
		}
	}

	  /* find out which UserLabels are going to be dumped	*/
	set_links+= _XGDump_WhichLabels(wi);

	prev_set= NULL;
	empty= 0;
	  /* 20040216: determine the 'dumped set number' beforehand: we need that for correctly saving
	   \ the Linked2Set information.
	   */
	for( idx= 0; idx< setNumber; idx++ ){
		this_set= &AllSets[idx];
		if( this_set->numPoints> 0 && (!XG_Stripped || draw_set(wi, idx)) ){
			this_set->dumped_set= this_set->set_nr- empty;
		}
		else{
			empty+= 1;
			this_set->dumped_set= -empty;
		}
	}
	empty= 0;
	for( idx= 0; idx< setNumber; idx++ ){
	  char msg[128];
	  Boolean rf_dumped= False;
	  extern char *Read_File_Buf;

		this_set= &AllSets[idx];

		if( XG_Stripped== 2 && this_set->read_file && this_set->read_file_point== 0 && wi->draw_set[idx]> 0){
			if( bin_active ){
				if( bin_active== True ){
					BinaryTerminate(fp);
				}
				fputc( '\n', fp );
				bin_active= False;
			}
			fprintf( fp, "# Set %d has inclusion before start-of-data:\n%s\n\n", idx, this_set->read_file );
			rf_dumped= True;
		}

		this_set->xcol= wi->xcol[idx];
		this_set->ycol= wi->ycol[idx];
		this_set->ecol= wi->ecol[idx];
		this_set->lcol= wi->lcol[idx];

		if( this_set->numPoints> 0 ){
			if( !XG_Stripped || draw_set(wi, idx) ){
			  int set_link= this_set->set_link;

				sprintf( msg, "Dumping set #%d ...", idx );
				if( HO_Dialog.win ){
					XStoreName( disp, HO_Dialog.win, msg );
					XG_XSync( disp, False );
				}

				if( this_set->ncols> MaxCols ){
					fprintf( StdErr,
						"xgraph::_XGraphDump(): detected internal error: set[%d]->ncols==%d > MaxCols==%d (corrected)\n",
						idx, this_set->ncols, MaxCols
					);
					MaxCols= this_set->ncols;
					if( wi->DumpBinary && !AllocBinaryFields( MaxCols, "_XGraphDump()" ) ){
						wi->DumpBinary= False;
					}
				}

				if( (this_set->set_link>= 0 && (
						(XG_Stripped && !draw_set(wi,this_set->set_link))
					  /* 20031010: it is also not a wise idea to keep links when dumping processed values.
					   \ more elaborate checking could be done here, but the simplest/safest is just to
					   \ remove the link in this case...
					   */
						|| wi->DumpProcessed
					))
				){	
					if( this_set->set_link ){
						this_set->set_link*= -1;
					}
					else{
						this_set->set_link= -1;
					}
				}

				if( Init_XG_Dump || !XG_Really_Incomplete ){
					if( XG_Stripped!= 2 ){
						if( empty ){
							fprintf( fp, "\n*Set* %d,%d (#%d orig.) (#%d dumped)\n",
								this_set->set_nr- empty, setNumber- empty, this_set->set_nr, set_nr
							);
						}
						else{
							fprintf( fp, "\n*Set* %d,%d (#%d dumped)\n", this_set->set_nr, setNumber- empty, set_nr );
						}
					}
					if( this_set->titleText ){
						  /* 990617: I'd think that there is no longer need to not dump identical titles..
						   \ Especially with the new possibility of opcodes that make for dynamic titlestrings..
						   \ If bluntly dumping all titles causes problems, another heuristic will need
						   \ to be found.
						   */
						if( True ||
							(prev_set== NULL && strcmp( this_set->titleText, titleText)) ||
							(prev_set && prev_set->titleText && strcmp( this_set->titleText, prev_set->titleText)) ||
							(prev_set && !prev_set->titleText)
						){
							print_string( fp, "*TITLE*", "\\n\n", "\n", this_set->titleText );
						}
					}
					else{
					  /* This will "lock" the non-titledness of the set	*/
						fprintf( fp, "#no set-title..\n*TITLE*\n" );
					}
					if( this_set->setName ){
						print_string( fp, "*LEGEND*", "\\n\n", "\n", this_set->setName );
					}
					if( this_set->fileName ){
					  /* This must be checked for all sets!	*/
						if( (wi->new_file[idx]) || next_new_file ){
							print_string( fp, "*FILE*", "\\n\n", "\n\n", this_set->fileName );
						}
						else{
							if( idx== 0 || !prev_fileName || strcmp( this_set->fileName, prev_fileName) ){
								print_string( fp, "*file*", "\\n\n", "\n\n", this_set->fileName );
							}
						}
						prev_fileName= this_set->fileName;
						next_new_file= False;
					}
				}
				if( this_set->average_set && !wi->dump_average_values ){
					start= this_set->averagedPoints;
				}
				else{
					start= 0;
				}
				if( Init_XG_Dump || !XG_Really_Incomplete ){
					_DumpSetHeaders( fp, wi, this_set, &points, &NumObs, (start>= this_set->numPoints) /* , empty */ );
				}
				else if( this_set->ColumnLabels ){
				  Sinc sink;
					Sprint_SetLabelsList( Sinc_file( &sink, fp, 0,0), this_set->ColumnLabels, NULL, "\n" );
				}
// 20090211: we know how to handle when points!=this_set->numPoints?!
// 				allow_short_bin= ((points== this_set->numPoints) && !(this_set->read_file && this_set->read_file_point>=0));
				allow_short_bin= (!(this_set->read_file && this_set->read_file_point>=0));
				if( start< this_set->numPoints && this_set->set_link< 0 ){
				  ALLOCA( Extreme, Extremes, this_set->ncols, extr_len);
					DumpExtremes( wi, fp, this_set, start, &bin_active, Extreme );
#define MINSAVESETS	1
					if( wi->DumpProcessed ){
						  /* 20051103: MINSAVESETS used to be 3.... */
						if( this_set->has_error || this_set->ncols>= MINSAVESETS ){
							if( Init_XG_Dump || !XG_Really_Incomplete ){
								if( wi->transform.y_len /* !wi->vectorFlag */ ){
									fprintf( fp, "*ECHO* Warning: errorbars may not be as one might suspect!\n\n" );
								}
							}
							if( wi->DumpBinary ){
								  /* Dump the 1st BINARYDATA header	*/
								if( allow_short_bin ){
									fprintf( fp, "*BINARYDATA* l=%d c=%d s=%d se=-1\n",
// 										this_set->numPoints, this_set->ncols, BinarySize
										points, this_set->ncols, BinarySize
									);
									bin_active= -1;
								}
								else{
									fprintf( fp, "*BINARYDATA*\n" );
									bin_active= True;
								}
							}
							if( this_set->xcol==this_set->ycol || this_set->xcol==this_set->ecol
								|| this_set->ycol==this_set->ecol
							){
								fprintf( StdErr, "Set #%d (%s): X, Y and/or Error columns are identical: this may cause problems in the output! (%d,%d,%d)\n",
									this_set->set_nr, (this_set->setName)? this_set->setName : "?",
									this_set->xcol, this_set->ycol, this_set->ecol
								);
							}
							for( j= start; j< this_set->numPoints; j++ ){
								if( !DiscardedPoint( wi, this_set, j) ){
									if( SplitHere(this_set, j) ){
										if( bin_active ){
											  /* We must exit binary mode here. Dump the terminator, followed by
											   \ a newline, the information to be dumped in ASCII, and re-enter
											   \ binary mode. Note: BinaryTerminator.columns==0, so the 2nd fwrite()
											   \ currently doesn't dump anything..!
											   \ 990629: only need to test for regular binary mode; the short version
											   \ is enabled only when all points are dumped.
											   */
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "*SPLIT* %s\n", (SplitHere(this_set,j)> 0)? "mcut" : "tcut" );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
									if( this_set->plot_interval<= 0 || (XG_Stripped!=1 || (j % this_set->plot_interval)==0) ){
									  int i;
										if( bin_active ){
											  /* Set the columns field, and fill the data array with
											   \ the actual datapoints.
											   */
											BinaryDump.columns= this_set->ncols;
											for( i= 0; i< this_set->ncols; i++ ){
												if( i== this_set->xcol ){
													SetBinaryDumpData( &BinaryDump, i, this_set->xvec[j], Extreme );
												}
												else if( i== this_set->ycol ){
													SetBinaryDumpData( &BinaryDump, i, this_set->yvec[j], Extreme );
												}
												else if( i== this_set->ecol ){
													if( !wi->transform.y_len /* wi->vectorFlag	*/ ){
														SetBinaryDumpData( &BinaryDump, i, this_set->errvec[j], Extreme );
													}
													else{
														if( wi->error_type[this_set->set_nr]== 4 ){
															fprintf( StdErr, "_XGraphDump(): DumpProcessed in vectorMode with TRANSFORM_? not implemented!\n" );
														}
														SetBinaryDumpData( &BinaryDump, i, (this_set->hdyvec[j]- this_set->ldyvec[j])/2, Extreme );
													}
												}
												else if( i== this_set->lcol ){
													if( !wi->transform.x_len /* wi->vectorFlag	*/ ){
														SetBinaryDumpData( &BinaryDump, i, this_set->lvec[j], Extreme );
													}
													else{
														if( wi->error_type[this_set->set_nr]== 4 ){
															fprintf( StdErr, "_XGraphDump(): DumpProcessed in vectorMode with TRANSFORM_? not implemented!\n" );
														}
														SetBinaryDumpData( &BinaryDump, i, (this_set->hdxvec[j]- this_set->ldxvec[j])/2, Extreme );
													}
												}
												else{
													SetBinaryDumpData( &BinaryDump, i, this_set->columns[i][j], Extreme );
												}
											}
											  /* Dump the fields. I would have liked to do that in a single statement,
											   \ but (on a SG O2), I got strange SIGSEGV errors when I allocated the
											   \ arena as an int array of sizeof(int) [for the columns] + <cols>* sizeof(double)
											   \ bytes long, and initialised the data field as &<arena>[1] . Alignment/padding problem?
											   \ At least, this works, and it presents an enormous speedgain vs. ASCII dumps.
											   \ Might be even faster when I switch of the progress information in binary mode.. :)
											   \ (speedgain when *reading* is between 4 and 5)
											   \ 990629: addition of a "short" binary mode, in which all points & columns
											   \ are dumped, and we can *thus* omit the initial "columns column" (which is there
											   \ mainly to provide a terminator, currently).
											   */
											BinaryDumpData(fp, allow_short_bin );
										}
										else{
											for( i= 0; i< this_set->ncols; i++ ){
												if( i== this_set->xcol ){
													fprintf( fp, "%s", dd2str( wi, this_set->xvec[j], d3str_format, NULL));
												}
												else if( i== this_set->ycol ){
													fprintf( fp, "%s", dd2str( wi, this_set->yvec[j], d3str_format, NULL));
												}
												else if( i== this_set->ecol ){
													if( !wi->transform.y_len /* wi->vectorFlag */ ){
														fprintf( fp, "%s", dd2str( wi, this_set->errvec[j], d3str_format, NULL));
													}
													else{
														if( wi->error_type[this_set->set_nr]== 4 ){
															fprintf( StdErr, "_XGraphDump(): DumpProcessed in vectorMode with TRANSFORM_? not implemented!\n" );
														}
														fprintf( fp, "%s",
															dd2str( wi, (this_set->hdyvec[j]- this_set->ldyvec[j])/2, d3str_format, NULL)
														);
													}
												}
												else if( i== this_set->lcol ){
													if( !wi->transform.x_len /* wi->vectorFlag */ ){
														fprintf( fp, "%s", dd2str( wi, this_set->lvec[j], d3str_format, NULL));
													}
													else{
														if( wi->error_type[this_set->set_nr]== 4 ){
															fprintf( StdErr, "_XGraphDump(): DumpProcessed in vectorMode with TRANSFORM_? not implemented!\n" );
														}
														fprintf( fp, "%s",
															dd2str( wi, (this_set->hdxvec[j]- this_set->ldxvec[j])/2, d3str_format, NULL)
														);
													}
												}
												else{
													fprintf( fp, "%s",
														dd2str( wi, this_set->columns[i][j], d3str_format, NULL)
													);
												}
												if( i< this_set->ncols- 1 ){
													fputc( '\t', fp );
												}
											}
											if( wi->use_average_error ){
												fprintf( fp, "\t%s", d2str( this_set->av_error_r, d3str_format, NULL));
											}
											fputc( '\n', fp );
										}
									}
									if( splits_disconnect && SplitHere(this_set, j) && this_set->arrows ){
										if( bin_active ){
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "%s\n", arrow_types[this_set->arrows] );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
								}
								if( XG_Stripped== 2 && !rf_dumped && this_set->read_file && idx>= this_set->read_file_point ){
									if( bin_active ){
										BinaryTerminate(fp);
										fputc( '\n', fp );
										bin_active= False;
									}
									fprintf( fp, "# Set %d has inclusion at point %d:\n%s\n\n",
										idx, this_set->read_file_point, this_set->read_file
									);
									rf_dumped= True;
								}
							}
							if( bin_active ){
								if( bin_active== True ){
									BinaryTerminate(fp);
								}
								fputc( '\n', fp );
								bin_active= False;
							}
						}
#if 0
/* 20051103: I don't understand what this is doing here...! */
						else{
							if( wi->DumpBinary ){
								if( allow_short_bin ){
									fprintf( fp, "*BINARYDATA* l=%d c=%d s=%d se=-1\n",
// 										this_set->numPoints, 2, BinarySize
										points, 2, BinarySize
									);
									bin_active= -1;
								}
								else{
									fprintf( fp, "*BINARYDATA*\n" );
									bin_active= True;
								}
							}
							for( j= start; j< this_set->numPoints; j++ ){
								if( !DiscardedPoint( wi, this_set, j) ){
									if( SplitHere(this_set, j) ){
										if( bin_active ){
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "*SPLIT* %s\n", (SplitHere(this_set,j)> 0)? "mcut" : "tcut" );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
									if( this_set->plot_interval<= 0 || (XG_Stripped!=1 || (j % this_set->plot_interval)==0) ){
										if( bin_active ){
											BinaryDump.columns= 2;
											SetBinaryDumpData( &BinaryDump, 0, this_set->xvec[j], Extreme );
											SetBinaryDumpData( &BinaryDump, 1, this_set->yvec[j], Extreme );
											BinaryDumpData(fp, allow_short_bin );
										}
										else{
											fprintf( fp, "%s\t%s\n",
												dd2str( wi, this_set->xvec[j], d3str_format, NULL),
												dd2str( wi, this_set->yvec[j], d3str_format, NULL)
											);
										}
									}
									if( splits_disconnect && SplitHere(this_set, j) && this_set->arrows ){
										if( bin_active ){
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "%s\n", arrow_types[this_set->arrows] );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
								}
								if( XG_Stripped== 2 && !rf_dumped && this_set->read_file && idx>= this_set->read_file_point ){
									if( bin_active ){
										BinaryTerminate(fp);
										fputc( '\n', fp );
										bin_active= False;
									}
									fprintf( fp, "# Set %d has inclusion at point %d:\n%s\n\n",
										idx, this_set->read_file_point, this_set->read_file
									);
									rf_dumped= True;
								}
							}
							if( bin_active ){
								if( bin_active== True ){
									BinaryTerminate(fp);
								}
								fputc( '\n', fp );
								bin_active= False;
							}
						}
#endif
					}
					else{
						  /* 20051103 This test used to be 'this_set_ncols> 3' (i.e. > instead of >=!).
						   \ Let's just save everything, and always with the same mechanism.
						   */
						if( this_set->has_error || this_set->ncols>= MINSAVESETS ){
							if( wi->DumpBinary ){
								if( allow_short_bin ){
									fprintf( fp, "*BINARYDATA* l=%d c=%d s=%d se=-1\n",
// 										this_set->numPoints, this_set->ncols, BinarySize
										points, this_set->ncols, BinarySize
									);
									bin_active= -1;
								}
								else{
									fprintf( fp, "*BINARYDATA*\n" );
									bin_active= True;
								}
							}
							for( j= start; j< this_set->numPoints; j++ ){
								if( DiscardedPoint( NULL, this_set, j)<= 0 &&
									!(WinDiscardedPoint(wi, this_set, j)< 0)
								){
									if( SplitHere(this_set, j)> 0 ){
										if( bin_active ){
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "*SPLIT* mcut\n" );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
									if( this_set->plot_interval<= 0 || (XG_Stripped!=1 || (j % this_set->plot_interval)==0) ){
									  int i;
										if( bin_active ){
											BinaryDump.columns= this_set->ncols;
											for( i= 0; i< this_set->ncols; i++ ){
												SetBinaryDumpData( &BinaryDump, i, this_set->columns[i][j], Extreme );
											}
											BinaryDumpData(fp, allow_short_bin );
										}
										else{
											for( i= 0; i< this_set->ncols; i++ ){
												fprintf( fp, "%s",
													dd2str( wi, this_set->columns[i][j], d3str_format, NULL)
												);
												if( i< this_set->ncols- 1 ){
													fputc( '\t', fp );
												}
											}
											if( wi->use_average_error ){
												fprintf( fp, "\t%s", d2str( this_set->av_error_r, d3str_format, NULL));
											}
											fputc( '\n', fp );
										}
									}
									if( splits_disconnect && SplitHere(this_set, j) && this_set->arrows ){
										if( bin_active ){
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "%s\n", arrow_types[this_set->arrows] );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
								}
								if( XG_Stripped== 2 && !rf_dumped && this_set->read_file && idx>= this_set->read_file_point ){
									if( bin_active ){
										BinaryTerminate(fp);
										fputc( '\n', fp );
										bin_active= False;
									}
									fprintf( fp, "# Set %d has inclusion at point %d:\n%s\n\n",
										idx, this_set->read_file_point, this_set->read_file
									);
									rf_dumped= True;
								}
							}
							if( bin_active ){
								if( bin_active== True ){
									BinaryTerminate(fp);
								}
								fputc( '\n', fp );
								bin_active= False;
							}
						}
#if 0
/* 20051103: I don't understand what this is doing here...! */
						else{
							if( wi->DumpBinary ){
								if( allow_short_bin ){
									fprintf( fp, "*BINARYDATA* l=%d c=%d s=%d se=-1\n",
// 										this_set->numPoints, 2, BinarySize
										points, 2, BinarySize
									);
									bin_active= -1;
								}
								else{
									fprintf( fp, "*BINARYDATA*\n" );
									bin_active= True;
								}
							}
							for( j= start; j< this_set->numPoints; j++ ){
								if( DiscardedPoint( NULL, this_set, j)<= 0 &&
									!(WinDiscardedPoint(wi, this_set, j)< 0)
								){
									if( SplitHere(this_set, j)> 0 ){
										if( bin_active ){
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "*SPLIT* mcut\n" );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
									if( this_set->plot_interval<= 0 || (XG_Stripped!=1 || (j % this_set->plot_interval)==0) ){
										if( bin_active ){
											BinaryDump.columns= 2;
											SetBinaryDumpData( &BinaryDump, 0, this_set->xval[j], Extreme );
											SetBinaryDumpData( &BinaryDump, 1, this_set->yval[j], Extreme );
											BinaryDumpData(fp, allow_short_bin );
										}
										else{
											fprintf( fp, "%s\t%s\n",
												dd2str( wi, this_set->xval[j], d3str_format, NULL),
												dd2str( wi, this_set->yval[j], d3str_format, NULL)
											);
										}
									}
									if( splits_disconnect && SplitHere(this_set, j) && this_set->arrows ){
										if( bin_active ){
											BinaryTerminate(fp);
											fputc( '\n', fp );
											bin_active= False;
										}
										fprintf( fp, "%s\n", arrow_types[this_set->arrows] );
										if( wi->DumpBinary ){
											fprintf( fp, "*BINARYDATA*\n" );
											bin_active= True;
										}
									}
								}
								if( XG_Stripped== 2 && !rf_dumped && this_set->read_file && idx>= this_set->read_file_point ){
									if( bin_active ){
										BinaryTerminate(fp);
										fputc( '\n', fp );
										bin_active= False;
									}
									fprintf( fp, "# Set %d has inclusion at point %d:\n%s\n\n",
										idx, this_set->read_file_point, this_set->read_file
									);
									rf_dumped= True;
								}
							}
							if( bin_active ){
								if( bin_active== True ){
									BinaryTerminate(fp);
								}
								fputc( '\n', fp );
								bin_active= False;
							}
						}
#endif
					}
					fputc( '\n', fp );
				}
				else if( XG_Stripped== 2 && this_set->read_file && !rf_dumped ){
					if( bin_active ){
						BinaryTerminate(fp);
						fputc( '\n', fp );
						bin_active= False;
					}
					if( Init_XG_Dump || !XG_Really_Incomplete ){
						fprintf( fp, "# Set %d has inclusion at point %d:\n%s\n\n",
							idx, this_set->read_file_point, this_set->read_file
						);
					}
					rf_dumped= True;
				}
				  /* find links specific for this set.	*/
				if( (Init_XG_Dump || !XG_Really_Incomplete) && set_links> 0 && XGDump_Labels ){
					_XGDump_Linked_UserLabels( fp, wi, set_nr, this_set, False, empty, &set_links );
				}
				fflush( fp );
				prev_set= this_set;
				nsets+= 1;
				set_nr+= 1;

				this_set->set_link= set_link;
			}
			else{
				sprintf( msg, "Skipping set #%d ...", idx );
				if( HO_Dialog.win ){
					XStoreName( disp, HO_Dialog.win, msg );
					XG_XSync( disp, False );
				}
				if( Init_XG_Dump || !XG_Really_Incomplete ){
					if( XG_Stripped!= 2 ){
						fprintf( fp, "# %s\n", msg );
					}
				}
				prev_fileName= this_set->fileName;
				  /* This non-dumped set starts a new group, something we'd like to preserve..	*/
				if( wi->new_file[idx] ){
					next_new_file= True;
				}
				empty+= 1;
			}
		}
		else{
			sprintf( msg, "Skipping deleted set #%d ...", idx );
			if( HO_Dialog.win ){
				XStoreName( disp, HO_Dialog.win, msg );
				XG_XSync( disp, False );
			}
			if( Init_XG_Dump || !XG_Really_Incomplete ){
				if( XG_Stripped!= 2 ){
					fprintf( fp, "# %s\n", msg );
				}
			}
			  /* This non-dumped set starts a new group, something we'd like to preserve..	*/
			if( wi->new_file[idx] ){
				next_new_file= True;
			}
			empty+= 1;
		}
		if( XG_Stripped== 2 ){
			  /* check against wi->draw_set, instead of using draw_set() which returns
			   \ 0 for deleted sets
			   */
			if( this_set->read_file && !rf_dumped && wi->draw_set[this_set->set_nr]> 0 ){
				if( bin_active ){
					BinaryTerminate(fp);
					fputc( '\n', fp );
					bin_active= False;
				}
				if( Init_XG_Dump || !XG_Really_Incomplete ){
					fprintf( fp, "# Set %d has inclusion at point %d:\n%s\n\n",
						idx, this_set->read_file_point, this_set->read_file
					);
				}
				rf_dumped= True;
			}
			if( Read_File_Buf && idx== setNumber-1 && wi->draw_set[idx]> 0 ){
				if( bin_active ){
					BinaryTerminate(fp);
					fputc( '\n', fp );
					bin_active= False;
				}
				if( Init_XG_Dump || !XG_Really_Incomplete ){
					fprintf( fp, "# Trailing inclusions:\n%s\n\n", Read_File_Buf);
				}
			}
		}
	}
	if( wi->DumpBinary ){
		BinaryDump.columns= bdc;
	}
	if( Init_XG_Dump && strlen( titleText ) ){
	  /* (-t title) is shown as last	*/
		fprintf( fp, "\n*ARGUMENTS* -t \"%s\"\n\n", titleText );
	}

	if( linkedArrays ){
	  int f= 0;
	  ascanf_Function *af= vars_ascanf_Functions[f].cdr;
	  int rootvar= 0, len= 0, l= 0;
		if( !af ){
			f+= 1;
			af= &vars_ascanf_Functions[f];
			rootvar= 1;
		}
		while( af ){
			len= 0;
			if( af->type== _ascanf_array && af->linkedArray.dataColumn ){
				len= fprintf( fp, "*EVAL* ");
				l= print_string2( fp, "LinkArray2DataColumn[&", ",", af->name, False );
				l+= fprintf( fp, "%d,%d] @", af->linkedArray.set_nr, af->linkedArray.col_nr );
				len+= l;
				linkedArrays-= 1;
			}
			if( !(af= af->cdr) ){
				f+= 1;
				if( f< ascanf_Functions ){
					af= &vars_ascanf_Functions[f];
					rootvar= 1;
				}
			}
			else{
				rootvar= 0;
			}
			if( len>= new_local_buf_size ){
				new_local_buf_size= len+ 10;
				change_local_buf_size= True;
			}
		}
		fputs( "\n", fp );
		fflush( fp);
	}

	if( Init_XG_Dump || !XG_Really_Incomplete ){
		  /* Dump the remaining UserLabels	*/
		if( XG_Stripped!= 2 && XGDump_Labels ){
		  UserLabel *ul= wi->ulabel;
			if( ul ){
				if( HO_Dialog.win ){
					XStoreName( disp, HO_Dialog.win, "Dumping UserLabels...");
					XG_XSync( disp, False );
				}
			}
			_XGDump_UserLabels( fp, wi, False, empty );
		}

		fprintf( fp, "\n*EXTRATEXT* dumped %d sets of %d\n\n", nsets, MaxSets );
	}

	if( Init_XG_Dump ){

		if( DumpPens ){
			fprintf( fp, "*ARGUMENTS* -DumpPens1\n" );
			Dump_XGPens( wi, fp, True, NULL );
		}
		else if( !wi->DumpProcessed ){
			Dump_XGPens( wi, fp, False, NULL );
		}

		_Dump_Arg0_Info( wi, NullDevice, NULL, True);
		if( process_history ){
			fprintf( fp, "*PROCESS_HISTORY*%s\n\n", process_history );
		}

		_XGDump_Finish( fp, wi );

		if( XGDump_AllWindows && XGDump_AllWindows_Filename ){
		  int w= 0, N= Num_Mapped();
		    /* We'll cycle through the list in backwards fashion, because new windows
			 \ are cons'ed into it, i.e. in FILO order... We'd like to be able to
			 \ restore the window order as well.
			 */
		  LocalWindows *WL= WindowListTail;
		  LocalWin *lwi;
			if( HO_Dialog.win ){
				XStoreName( disp, HO_Dialog.win, "Dumping the other windows...");
				XG_XSync( disp, False );
			}
			fprintf( fp,
				"\n*EVAL* DCL[%s,1,\"This variable indicates that the data in this file has"
				" been read, and that the window structure can now be restored\"] @\n", AllWinVar
			);
/* 			fprintf( fp, "*SCRIPT_FILE* %s\n", XGDump_AllWindows_Filename );	*/
			fprintf( fp, "*SCRIPT_FILE* *This-File*\n" );
			fprintf( fp, "*EVAL* exit[1] @\n\n" );
			fprintf( fp, "*Restore_All_Windows*\n" );
			fprintf( fp, "*EVAL* IDict[ Delete[%s] ] @\n", AllWinVar );
			fprintf( fp, "*EVAL* IDict[ DCL[%%XGAWD_WinList,%d,0] ], %%XGAWD_WinList[0,ActiveWin] @\n", N );
			fprintf( fp, "*UPDATE_WIN*\n*EVAL* RedrawNow[1] @\n" );
			fflush( fp );
			while( WL ){
				if( (lwi= WL->wi)!= wi ){
				  int lwdp= lwi->DumpProcessed;
					w+= 1;
					lwi->DumpProcessed= 0;
					fprintf( fp, "\n*EXTRATEXT* Window %d\n\n", w );
					fprintf( fp, "*ACTIVATE* %%XGAWD_WinList[%d]\n*CLONE*\n", w- 1 );
					fprintf( fp, "*EVAL* %%XGAWD_WinList[%d,ActiveWin] @\n", w );
					fprintf( fp, "*UPDATE_FLAGS*\n" );
					Collect_Arguments( lwi, cbuf, 2048);
					if( strlen( cbuf) ){
						fprintf( fp, "*ARGUMENTS* %s", cbuf );
						fputc( '\n', fp);
					}
					fprintf( fp,
						"*ARGUMENTS* -legendtype %d -radix %g -radix_offset %g -raw_display%d\n",
						lwi->legend_type, lwi->radix, lwi->radix_offset,
						lwi->raw_display
					);
					fprintf( fp, "*EXTRATEXT*\nRemoving labels copied from parent window...\n\n"
						"*ULABEL* reset\n\n"
					);
					_XGDump_PlotSets( fp, lwi );
					_XGDump_MarkedSets( fp, lwi, NULL );
					_XGDump_MoreArguments( fp, lwi );
					_XGDump_Labels( fp, lwi );
					_XGDump_ValCats( fp, lwi );
					_XGDump_Process( fp, lwi, True );
					_XGDump_WindowProcs( fp, lwi, True );
					set_links= _XGDump_WhichLabels(lwi);
					for( empty= 0, idx= 0; idx< setNumber; idx++ ){
						if( AllSets[idx].numPoints> 0 && draw_set( lwi, idx) ){
						  /* set has already been dumped */
							if( (Init_XG_Dump || !XG_Really_Incomplete) && set_links> 0 && XGDump_Labels ){
								  /* dump this set's label(s) */
								_XGDump_Linked_UserLabels( fp, lwi, idx, &AllSets[idx], False, empty, &set_links );
							}
						}
						else if( XG_Stripped ){
							  /* 20040919: only increase the empty counter when the set has not been dumped! */
							empty+= 1;
						}
					}
					if( XG_Stripped!= 2 && XGDump_Labels ){
						  /* 20040919: dump remaining labels, do NOT ignore draw_it */
						_XGDump_UserLabels( fp, lwi, False, empty );
					}

					if( DumpPens ){
						Dump_XGPens( lwi, fp, True, NULL );
					}
					else if( !lwi->DumpProcessed ){
						Dump_XGPens( lwi, fp, False, NULL );
					}

					fprintf( fp, "*UPDATE_WIN*\n" );
					fprintf( fp, "*EVAL* redraw[1] @\n" );
					fprintf( fp, "*GEOM* %dx%d+%d+%d\n",
						lwi->dev_info.area_w, lwi->dev_info.area_h,
						lwi->dev_info.area_x, lwi->dev_info.area_y
					);

					lwi->DumpProcessed= lwdp;
				}
				else{
					fprintf( fp, "\n*EXTRATEXT* %%XGAWD_WinList[0] was here on the stack during the dumping\n\n" );
					fprintf( fp, "*ACTIVATE* %%XGAWD_WinList[0]\n" );
					fprintf( fp, "*GEOM* %dx%d+%d+%d\n",
						lwi->dev_info.area_w, lwi->dev_info.area_h,
						lwi->dev_info.area_x, lwi->dev_info.area_y
					);
				}
				WL= WL->prev;
			}
			  /* Output some more things for the "root" window:	*/
			fprintf( fp, "*ACTIVATE* %%XGAWD_WinList[0]\n" );
			fprintf( fp, "*UPDATE_FLAGS*\n" );
			Collect_Arguments( wi, cbuf, 2048);
			if( strlen( cbuf) ){
				fprintf( fp, "*ARGUMENTS* %s", cbuf );
				fputc( '\n', fp);
			}
			fprintf( fp,
				"*ARGUMENTS* -legendtype %d -radix %g -radix_offset %g -raw_display%d\n",
				wi->legend_type, wi->radix, wi->radix_offset,
				wi->raw_display
			);
			_XGDump_MoreArguments( fp, lwi );
			_XGDump_WindowProcs( fp, wi, True );
			fprintf( fp, "*UPDATE_WIN*\n" );
			fprintf( fp, "*EVAL* redraw[1] @\n\n" );

			fprintf( fp, "*EVAL* IDict[ Delete[%%XGAWD_WinList] ] @\n\n" );
		}
		else{
		  /* _XGDump_WindowProcs( fp, wi, False ) ??? */
		}

		fprintf( fp, "*EXTRATEXT* Created by command:\n");
		_Dump_Arg0_Info( wi, fp, NULL, False);

		{ int n, rsacsv= reset_ascanf_currentself_value;
			if( wi->process.dump_after_len ){
				clean_param_scratch();
				n= param_scratch_len;
				*ascanf_self_value= 0.0;
				*ascanf_current_value= 0.0;
				*ascanf_counter= (*ascanf_Counter)= 0.0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "_XGraphDump(): DUMP after: %s", wi->process.dump_after);
					fflush( StdErr );
				}
				if( HO_Dialog.win ){
					XStoreName( disp, HO_Dialog.win, "Executing *DUMP_AFTER*...");
					XG_XSync( disp, False );
				}
				TBARprogress_header= "*DUMP_AFTER*";
				ascanf_arg_error= 0;
				compiled_fascanf( &n, wi->process.dump_after, param_scratch, NULL, data, column, &wi->process.C_dump_after );
				TBARprogress_header= NULL;
				TitleMessage( wi, NULL );
			}
			reset_ascanf_currentself_value= rsacsv;
		}

	}

	if( Exit_Exprs ){
	  XGStringList *list= Exit_Exprs;
	  char asep = ascanf_separator;
		if( HO_Dialog.win ){
			XStoreName( disp, HO_Dialog.win, "Dumping *EXIT expressions ...");
			XG_XSync( disp, False );
		}
		if( list->separator ){
			ascanf_separator = list->separator;
			fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
		}
		while( list ){
			print_string( fp, "**EXIT_EXPR**\n", NULL, " @\n*!EXIT_EXPR*\n\n", list->text );
			list= list->next;
		}
		if( ascanf_separator != asep ){
			ascanf_separator = asep;
			fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
		}
		fprintf( fp, "\n" );
	}

	use_greek_inf= ugi;
	if( HO_Dialog.win ){
		if( _Title ){
			XStoreName( disp, HO_Dialog.win, _Title );
			if( !RemoteConnection ){
				XSync( disp, False );
			}
			XFree( _Title );
		}
	}
	GCA();

	  /* Clean up	*/
	XG_Stripped= xgs;
	PS_STATE(wi)->Printing= PR;
	wi->DumpProcessed= wdp;
	ActiveWin= AW;

	Allow_Fractions= AF;
	d2str_printhex= dph;

	if( Update_LMAXBUFSIZE(False, errmsg) ){
		if( Init_XG_Dump || !XG_Really_Incomplete ){
			fprintf( fp, "*POPUP*\n"
				" There was a problem with the read-buffer length for this file\n"
				" Its suggested size is %d bytes (see the *BUFLEN* command at the beginning)\n"
				" The following warning was generated when this file was saved:\nXGraph message:\n"
				"%s\n",
				new_local_buf_size, errmsg
			);
		}
		return(1);
	}

	return( 0 );
}

extern int hardcopy_init_error;

/* make a dump in the XGraph ascii format	*/
int XGraphDump( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
)
{ int r;
  PS_Printing pr= PS_STATE(thePrintWin_Info)->Printing;
	if( initFile ){
		PS_STATE(thePrintWin_Info)->Printing= XG_DUMPING;
		hardcopy_init_error= r= _XGraphDump( thePrintWin_Info, fp, errmsg);
		PS_STATE(thePrintWin_Info)->Printing= pr;
	}
	else{
		return( 1 );
	}
	if( r && strlen(errmsg) ){
		do_message( errmsg);
	}
	errmsg[0]= '\0';
	return( 0);
}

/* dump the commandline "causing" this graph window	*/
int DumpCommand( FILE *fp, int width, int height, int orient,
	char *tf, double ts, char *lef, double les, char *laf, double las, char *af, double as,
	LocalWin *wi, char errmsg[ERRBUFSIZE], int initFile
)
{
	if( initFile ){
		Restart( thePrintWin_Info, fp );
	}
	else{
		return( 1 );
	}
	return( 0);
}

int XG_continue= 1;
static void XG_continue_( int sig )
{
	XG_continue= 1;
	signal( SIGALRM, XG_continue_ );
#ifdef DEBUG
	fprintf( StdErr, "alarm - continue\n");
#endif
}

int XG_sleep_once( double time, Boolean wait_now )
{ static struct itimerval rtt, ortt;
  static Boolean stored= 0;
	if( time> 0 ){
		if( stored ){
			  /* restore the previous setting of the timer.	*/
			setitimer( ITIMER_REAL, &ortt, &rtt );
			stored= False;
		}
		rtt.it_value.tv_sec= (unsigned long) ssfloor(time);
		rtt.it_value.tv_usec= (unsigned long) ( (time- rtt.it_value.tv_sec)* 1000000 );
		rtt.it_interval.tv_sec= 0;
		rtt.it_interval.tv_usec= 0;
		signal( SIGALRM, XG_continue_ );
		XG_continue= 0;
		setitimer( ITIMER_REAL, &rtt, &ortt );
		if( wait_now ){
#ifdef DEBUG
			fprintf( StdErr, "XG_sleep_once(%g) set&waiting.. ", time );
			fflush( StdErr );
#endif
			pause();
			  /* restore the previous setting of the timer.	*/
			setitimer( ITIMER_REAL, &ortt, &rtt );
			stored= False;
		}
		else{
#ifdef DEBUG
			fprintf( StdErr, "XG_sleep_once(%g) set.. ", time );
			fflush( StdErr );
#endif
			stored= True;
		}
	}
	else{
		if( stored ){
			if( wait_now ){
#ifdef DEBUG
				fprintf( StdErr, "XG_sleep_once(%g) waiting.. ", time );
				fflush( StdErr );
#endif
				pause();
			}
			setitimer( ITIMER_REAL, &ortt, &rtt );
			stored= False;
		}
		return( 0 );
	}
	return( 1 );
}

int DumpFILES( LocalWin *wi )
{ int r= 0, no= 0, nn= 0, dd= 0, idx;
  char *fn, *fnhist= NULL, msg[1024], *msglist= NULL, *_Title= NULL;
  int IXD= Init_XG_Dump, xgs= XG_Stripped, ds, alrm_set= 0;
  FILE *fp;
	XG_Stripped= 2;
	wi->ctr_A= 0;
	if( HO_Dialog.win ){
		XFetchName( disp, HO_Dialog.win, &_Title );
	}
	for( idx= 0; idx< setNumber; idx++ ){
	  /* Store the current pattern	*/
		if( wi->draw_set[idx]> 0 ){
			wi->draw_set[idx]*= -1;
		}
		else{
			wi->draw_set[idx]= 0;
		}
	}
	XG_continue= 1;
	for( idx= 0; idx< setNumber; idx++ ){
	  int dds= wi->draw_set[idx];
		  /* Select the (only) set to be dumped	*/
		wi->draw_set[idx]= 1;
		if( idx ){
			wi->draw_set[idx-1]= ds;
		}
		ds= dds;
		fn= AllSets[idx].fileName;
		if( AllSets[idx].numPoints<= 0 ){
			r+= 1;
			dd+= 1;
			sprintf( msg, "Set #%d has been deleted\n", idx );
			msglist= concat2( msglist, " ", msg, NULL );
		}
		else if( fn ){
			if( fnhist && strstr( fnhist, fn) ){
				fp= fopen( fn, "a");
				Init_XG_Dump= False;
			}
			else{
				{ time_t timer= time(NULL);
					strncpy( PrintTime, ctime(&timer), 255 );
				}
				PrintTime[255]= '\0';
				Init_XG_Dump= True;
				if( 0 /* fnhist && XG_continue */ ){
#ifdef DEBUG
					fprintf( StdErr, "File \"%s\": ", fn );
					fflush( StdErr );
#endif
					XG_sleep_once( 1.0, False );
					alrm_set= True;
				}
				fp= fopen( fn, "wb");
			}
			if( fp ){
			  char msg2[512];
			  int rr;
			  Time_Struct timer;
				msg[0]= '\0';
				Elapsed_Since(&timer, True);
				rr= _XGraphDump( wi, fp, msg);
				fclose(fp);
				if( rr ){
					sprintf( msg2, "Error %s set #%d to \"%s\"\n", (Init_XG_Dump)? "rewriting" : "re-appending", idx, fn );
					fputs( msg2, StdErr );
				}
				else{
					sprintf( msg2, "Set #%d %s to \"%s\"\n", idx, (Init_XG_Dump)? "rewritten" : "re-appended", fn );
					fnhist= concat2( fnhist, " ", fn, " ", NULL );
				}
				msglist= concat2( msglist, " ", msg2, NULL );
				if( strlen(msg) ){
					msglist= concat2( msglist, "      (", msg, ")\n", NULL );
					xtb_error_box( wi->window, msg, "Message" );
				}
				r+= rr;
				if( HO_Dialog.win ){
					XStoreName( disp, HO_Dialog.win, msg2 );
					if( !RemoteConnection ){
						XSync( disp, False );
					}
				}
				  /* Warning! This is a sort of asynchronous waiting alg. - the idea is to
				   \ have at least 1sec. time between closing one file and (re-)opening the next
				   \ file. If the dumping takes more time, there's no need to wait - the ALARM
				   \ signal will have been delivered, XG_continue reset to True, etc. I don't
				   \ know exactly what this signalling mechanism does to the dumping - should
				   \ be transparent, but if ever strange effects are observed during the dumping
				   \ of long files, or onto SLOW media, this is one possible culprit.
				   */
				if( Elapsed_Since(&timer, False)< 1 /* !XG_continue */ ){
#ifdef DEBUG
					fprintf( StdErr, "File \"%s\": ", fn );
					fflush( StdErr );
#endif
/* 					XG_sleep_once( -1, True );	*/
					sleep(1);
					alrm_set= False;
				}
			}
			else{
				r+= 1;
				no+= 1;
				sprintf( msg, "Couldn't open \"%s\" for set #%d (%s)\n", fn, idx, serror() );
				fputs( msg, StdErr );
				msglist= concat2( msglist, " ", msg, NULL );
			}
		}
		else{
			r+= 1;
			nn+= 1;
			sprintf( msg, "Set #%d has no name\n", idx );
			fputs( msg, StdErr );
			msglist= concat2( msglist, " ", msg, NULL );
		}
	}
	if( alrm_set ){
		  /* reset the alarm mechanism to its stored state	*/
		XG_sleep_once( -1, False );
	}
	Init_XG_Dump= IXD;
	XG_Stripped= xgs;
	for( idx= 0; idx< setNumber; idx++ ){
		if( wi->draw_set[idx]< 0 ){
			wi->draw_set[idx]*= -1;
		}
		else{
			wi->draw_set[idx]= 0;
		}
	}
	xfree( fnhist );
	sprintf( msg, "Rewrote %d files (%d file error(s), %d set(s) without fileName, %d deleted)", idx- r, no, nn, dd );
	msglist= concat2( msglist, " ", msg, NULL );
	xtb_error_box( wi->window, msglist, "Result(s)" );
	xfree( msglist );
	if( HO_Dialog.win ){
		if( _Title ){
			XStoreName( disp, HO_Dialog.win, _Title );
			if( !RemoteConnection ){
				XSync( disp, False );
			}
			XFree( _Title );
		}
	}
	return( r );
}

extern char ps_comment[1024];

extern int DrawWindow(LocalWin *wi);

int titles= 0;

/* parse_code(T) and remove heading and trailing whitespace	*/
char *cleanup( char *T )
{  char *c= T;
   static int h= 0, t= 0;
	if( !T ){
		return(NULL);
	}
	else if( ! *T ){
		return(T);
	}
	T= parse_codes( T );
	h= 0;
	t= 0;
	if( debugFlag ){
		fprintf( StdErr, "cleanup(0x%lx=\"%s\") ->\n", T, T);
	}
	  /* remove heading whitespace	*/
	if( isspace(*c) ){
		while( *c && isspace(*c) ){
			c++;
			h++;
		}
// 20101018: strings shouldn't overlap for strcpy...!
 		XGstrcpy( T, c);
	}
	  /* remove trailing whitespace	*/
	if( strlen( T) ){
		c= &T[ strlen(T)-1 ];
		if( isspace(*c) ){
			while( isspace(*c) && c> T ){
				c--;
				t++;
			}
			c[1]= '\0';
		}
	}
	if( debugFlag ){
		fprintf( StdErr, "\"%s\" (h=%d,t=%d)\n", (h||t)? T : "<no change>", h, t);
		fflush( StdErr );
	}
	return(T);
}

extern double (*RoundUp_log)(LocalWin *wi, double x);

extern double gridBase, gridStep, gridJuke[101];
extern int gridNJuke, gridCurJuke, floating;
extern double floating_value;

#define ADD_GRID(val)	(gridJuke[gridNJuke++] = (*ag_func)(wi,val))

extern int is_log_zero, is_log_zero2;

int AXsprintf( char *str, char *format, double val)
{
	if( AxisValueFormat ){
		d2str( val, format, str );
		return( strlen(str) );
	}
	else{
		return( sprintf( str, format, val ) );
	}
}

static char *condlffm(int exp, char *a)
{ static char buf[256];

	if( AxisValueFormat ){
		sprintf( buf, "%s%s", AxisValueFormat, a );
	}
	else if( exp ){
		sprintf( buf, "%%.%dlf%s", AxisValueMinDigits+ 1, a);
	}
	else{
		sprintf( buf, "%%.%dlf%s", AxisValueMinDigits, a);
	}
	StringCheck( buf, sizeof(buf)/sizeof(char), __FILE__, __LINE__ );
	return( buf );
}

 /* Axis value printing routine. If AxisValueFormat is set, that format is
  \ used. Otherwise, a heuristic is used which takes AxisValueMinDigits, and
  \ decides on a format with either that many, or one more, decimals, and
  \ either a %lf (decimal) or a %le (scientific) based.
  \ In polar mode, it prints duplets.
  */
char *WriteValue( LocalWin *wi,
	char *str, double val, double val2,
	int exp, int logFlag, int sqrtFlag,
	AxisName axis, int use_real_value, double step_size, int len)
{
    int idx;
	char *lz_sym= NULL, *lz_sym2= NULL,
		format2[64]= "%.2lf\0\0\0\0\0\0\0\0\0\0",
		command[DBL_MAX_10_EXP+128];
	static char format[32]= "%.2lf\0\0\0\0\0\0\0\0\0\0";
#define CONDLFFM(a)	condlffm(exp,a)
	double (*func)(LocalWin *wi, double x, double y), (*func2)(LocalWin *wi, double x, double y),
		Val= val, Val2= val2, scale;
	int lenf2= 0, lens= 0, lenc= 0, lenz= 0;

	if( axis== X_axis){
		func= Reform_X;
		func2= Reform_Y;
/* 		RoundUp_log= nlog10X;	*/
		RoundUp_log= cus_log10X;
		scale= wi->Xscale;
	}
	else{
		func= Reform_Y;
		func2= Reform_X;
/* 		RoundUp_log= nlog10Y;	*/
		RoundUp_log= cus_log10Y;
		scale= wi->Yscale;
	}

	strcpy( format, CONDLFFM("") );

	is_log_zero= 0;
	is_log_zero2= 0;
    if( logFlag ){
	  /* use the real value	*/
			if( axis== X_axis && wi->_log_zero_x && val== wi->log10_zero_x){
				is_log_zero= 1;
				if( wi->lz_sym_x)
					lz_sym= wi->log_zero_sym_x;
				val= 0.0;
				if( wi->_log_zero_y && val2== wi->log10_zero_y ){
					is_log_zero2= 1;
					if( wi->lz_sym_y)
						lz_sym2= wi->log_zero_sym_y;
					val2= 0.0;
				}
			}
			else if( axis== Y_axis && wi->_log_zero_y && val== wi->log10_zero_y){
				is_log_zero= 1;
				if( wi->lz_sym_y)
					lz_sym= wi->log_zero_sym_y;
				val= 0.0;
				if( wi->_log_zero_x && val2== wi->log10_zero_x ){
					is_log_zero2= 1;
					if( wi->lz_sym_x)
						lz_sym2= wi->log_zero_sym_x;
					val2= 0.0;
				}
			}
			else if( use_real_value){
				val= (*func)( wi, val, val2);
			}
	}
	else if( sqrtFlag && use_real_value ){
		val= (*func)( wi, val, val2);
	}
	else if( wi->polarFlag || use_real_value ){
		val= (*func)( wi, val, val2);
	}
	val2= (*func2)( wi, val2, Val);

	if( debugFlag && wi->axisFlag ){
		fprintf( StdErr, "WriteValue(%s,",
			(axis== X_axis)? "X":"Y"
		);
		fflush( StdErr );
		fprintf( StdErr, "%s=%s,%s=%s):",
			d2str( Val, "%lf", NULL),
			d2str( val, "%lf", NULL),
			d2str( Val2, "%lf", NULL),
			d2str( val2, "%lf", NULL)
		);
		fflush( StdErr );
	}

	if( fabs(val)< zero_epsilon && step_size> zero_epsilon )
		val= 0.0;
	if( fabs(val2)< zero_epsilon && step_size> zero_epsilon )
		val2= 0.0;

	if( wi->polarFlag && axis!= X_axis ){
	  /* swap: (val,val2) is (y,x), but will be printed as (x,y)	*/
	  double v= val;
	  int lz= is_log_zero;
	  char *lzs= lz_sym;
		val= val2; val2= v;
		is_log_zero= is_log_zero2; is_log_zero2= lz;
		lz_sym= lz_sym2; lz_sym2= lzs;
	}

	  /* Construct the format string	*/
	if( val || AxisValueFormat ){
/* 	 char *a= MATHERR_MARK();	*/
	 double rounded= fabs( (*RoundUp_log)( wi, fabs(val/scale)) - exp), lzval;
	 char lzbuf[DBL_MAX_10_EXP+12], lz_match= 0;
		if( AxisValueFormat ){
			strcpy( format, AxisValueFormat );
		}
		else{
			if( fabs(val)< 1.0 && rounded>= 1 ){
				if( rounded>= 2 ){
					sprintf( format, "%%.%dle", AxisValueMinDigits );
				}
				else{
					sprintf( format, "%%.%dlf", AxisValueMinDigits+ 1);
				}
			}
			else if( rounded>= 3 ){
				sprintf( format, "%%.%dle", AxisValueMinDigits );
			}
		}

		  /* check if our representation equals the representation
		   \ of the appropriate log_zero_[xy].
		   */
		lens= AXsprintf( str, format, val );
		if( lens> len ){
			fprintf( StdErr, "WriteValue(%s),%d: wrote %d bytes in %d buffer\n",
				str, __LINE__, lens, len
			);
		}
		if( axis== X_axis && logFlag && wi->_log_zero_x ){
			lzval= wi->log10_zero_x;
			if( (lenz= AXsprintf( lzbuf, format, wi->_log_zero_x )) &&
				strcmp( lzbuf, str)== 0
			){
				lz_match= 1;
			}
		}
		else if( axis== Y_axis && logFlag && wi->_log_zero_y ){
			lzval= wi->log10_zero_y;
			if( (lenz= AXsprintf( lzbuf, format, wi->_log_zero_y )) &&
				strcmp( lzbuf, str)== 0
			){
				lz_match= 1;
			}
		}
		if( lz_match ){
			if( debugFlag && wi->axisFlag ){
				fprintf( StdErr, "[%d,%d,%s,%s==%s]", lens, lenz, format, str, lzbuf );
				fflush( StdErr );
			}
			is_log_zero= 1;
			  /* Writing the present <val> (!= log10_zero_[xy]) would equal writing log_zero_[xy]. To correctly handle
			   \ this situation (without copying code ;-), we will call ourselves again,
			   \ with val=log10_zero_[xy]. In principle, we should not return here, so
			   \ it is okay to do a simple, unconditional tailrecursion.
			   */
			return( WriteValue(wi, str, lzval, Val2, exp, logFlag, sqrtFlag, axis, use_real_value, step_size, len ) );
		}
	}
	if( !val && !AxisValueFormat ){
		if( is_log_zero){
			if( lz_sym){
				strcpy( format, lz_sym);
			}
			else{
				strcpy( format, "0*");
			}
		}
		else{
			strcpy( format, "0");
		}
	}
	  /* second part of the format string	*/
	if( wi->polarFlag ){
		if( val2 || AxisValueFormat ){
/* 		  char *a= MATHERR_MARK();	*/
		  double rounded= fabs( (*RoundUp_log)( wi, fabs(val2)) - exp);
			if( AxisValueFormat ){
				sprintf( format2, "%s", AxisValueFormat );
			}
			else{
				if( fabs(val2)< 1.0 && rounded>= 1 ){
					if( rounded>= 2 ){
						sprintf( format2, "%%.%dle", AxisValueMinDigits );
					}
					else{
						sprintf( format2, "%%.%dlf", AxisValueMinDigits+1 );
					}
				}
				else if( rounded>= 3 ){
					sprintf( format2, "%%.%dle", AxisValueMinDigits );
				}
				else{
					strcpy( format2, CONDLFFM("") );
				}
			}
		}
		else{
			if( is_log_zero2){
				if( lz_sym2){
					lenf2= sprintf( format2, "%s", lz_sym2);
				}
				else{
					strcpy( format2, "0*");
				}
			}
			else{
				strcpy( format2, "0");
			}
		}
	}
	if( exp < 0 ){
		for( idx = exp;  idx < 0;  idx++ ){
			val *= 10.0;
			val2 *= 10.0;
		}
	} else {
		for( idx = 0;  idx < exp;  idx++ ){
			val /= 10.0;
			val2 /= 10.0;
		}
	}
	if( !wi->polarFlag ){
	  double v;
/* 		lens= sprintf(str, format, val);	*/
		lens= sprintf(str, "%s", d2str( val, format, NULL) );
		if( !index(str, '/') && sscanf( str, "%lf", &v)== 1 ){
		  char b[256];
			strcpy( b, str);
			lens= sprintf(str, "%s", d2str( v, format, NULL) );
			if( debugFlag ){
				fprintf( StdErr, "WriteValue(%.15g,\"%s\")=%s -> %.15g=%s=",
					val, format, b, v, str
				);
			}
		}
		lenc= sprintf( command, "(\"%s\", %le);", format, val);
	}
	else{
		if( val || AxisValueFormat ){
			lens= sprintf( str, "(%s\\#xb0\\,%s)", d2str( val, format, NULL), d2str( val2, format2, NULL) );
			lenc= sprintf( command, "(\"(%s\\#xb0\\,%s)\", %le, %le);", format, format2, val, val2);
		}
		else{
		  double v;
			lens= sprintf( str, "(0\\#xb0\\,%s)", d2str( val2, format2, NULL) );
			if( !index(str, '/') && sscanf( str, "%lf", &v)== 1 ){
			  char b[256];
				strcpy( b, str);
/* 				lens= sprintf(str, "%s", d2str( v, format2, NULL) );	*/
				lens= sprintf(str, "%s", d2str( v, format, NULL) );
				if( debugFlag ){
					fprintf( StdErr, "WriteValue(%.15g)=%s -> %.15g=%s=",
						val2, b, v, str
					);
				}
			}
			lenc= sprintf( command, "(\"(0\\#xb0\\,%s)\", %le);", format2, val2);
		}
	}
	if( lens> len ){
		fprintf( StdErr, "WriteValue(%s),%d: wrote %d bytes in %d buffer\n",
			str, __LINE__, lens, len
		);
	}
	if( debugFlag && wi->axisFlag ){
		fprintf( StdErr, " (%d,%d,%d)", lens, lenc, lenf2 );
		fflush( StdErr );
	}
	parse_codes( str );
	if( strlen(str)> len ){
		fprintf( StdErr, "WriteValue(%s),%d: wrote %d bytes in %d buffer (after parse_codes)\n",
			str, __LINE__, strlen(str), len
		);
	}
	if( debugFlag && wi->axisFlag ){
		fprintf( StdErr, " %s = %s (-> \"%s\")\n",
			command, str, format
		);
		fflush( StdErr );
	}
	return( format );
}


#define LEFT_CODE	0x01
#define RIGHT_CODE	0x02
#define BOTTOM_CODE	0x04
#define TOP_CODE	0x08

extern char *clipcode(int code);

/* Clipping algorithm from Neumann and Sproull by Cohen and Sutherland */
#define C_CODE(xval, yval, rtn) \
rtn = 0; \
if ((xval) < wi->UsrOrgX) rtn = LEFT_CODE; \
else if ((xval) > wi->UsrOppX) rtn = RIGHT_CODE; \
if ((yval) < wi->UsrOrgY) rtn |= BOTTOM_CODE; \
else if ((yval) > wi->UsrOppY) rtn |= TOP_CODE

/* mark_inside1: is (sx1,sy1) inside the clipping window?
 \ mark_inside2: is (sx2,sy2) inside the window after clipping?
 */
extern int ClipWindow();

/* #define NUMSETS	MAXSETS	*/

#define NUMSETS	setNumber

char *ShowLegends( LocalWin *wi, int PopUp, int this_one )
/* This popsup a small errorbox containing a message and the legends
 \ of the datasets shown in the graphs.
 */
{  char *msg, place[256];
   int i, N= 0, nP= 0, tnP= 0, HN= 0, MN= 0, DN= 0, SN= 0, fn_len= 0, overlap_len, sn_len= 0;
     /* rstring returns a copy of the displayed text if PopUp is false. It should be freed after
	  \ after use!
	  */
   char *rstring= NULL;
   double YRange= (double)(wi->XOppY - wi->XOrgY);
   Boolean reset= False;
   int first= -1, last= -1;

	errno= 0;

	if( (wi->no_legend && PopUp) ){
		if( PopUp ){
			xtb_error_box( wi->window, "No legends are shown\n", "The legend box");
		}
		return(NULL);
	}
	if( !wi->legend_length ){
		reset= True;
		DrawLegend( wi, False, NULL );
	}

	overlap_len= (wi->show_overlap && wi->overlap_buf)? strlen(wi->overlap_buf) : 0;
	overlap_len+= (wi->show_overlap && wi->overlap2_buf)? strlen(wi->overlap2_buf) : 0;
	if( use_X11Font_length ){
	  static char pc[]= "May be slightly too narrow for sets displayed with marks\n"
	  			"since it accounts for the PostScript width of those marks\n-";
		msg= parse_codes(pc);
	}
	else{
		msg= "";
	}
	if( wi->legend_placed ){
		sprintf( place, "-legend_ul%s %s,%s\nAspect y/x= %s\n", (wi->legend_trans)? "1" : "",
			d2str( wi->_legend_ulx, "%g", NULL),
			d2str( wi->_legend_uly, "%g", NULL),
			(YRange)? d2str( (double)(wi->XOppX - wi->XOrgX) / YRange +0.05, "%g", NULL) : "?"
		);
	}
	else{
		place[0]= '\0';
	}
	for( i= 0; i< NUMSETS; i++ ){
	  char *f= (wi->labels_in_legend)? AllSets[i].YUnits : XGrindex( AllSets[i].fileName, '/');
		if( f ){
			fn_len+= strlen(&f[1])+ 2;
		}
		if( draw_set(wi, i ) ){
			if( first== -1 ){
				first= i;
			}
			last= i;
			if( AllSets[i].nameBuf ){
				sn_len+= strlen( AllSets[i].nameBuf );
			}
			else if( AllSets[i].setName ){
				sn_len+= strlen( AllSets[i].setName );
			}
		}
	}
#define EXTRASPACE	256
	{ char nr_sets[]= "%s\n %d sets of %d (%d points of %d), %d marked, %d highlighted\n";
	  int alloc_len= wi->legend_length+ sn_len+ fn_len+ overlap_len+ 128+
				setNumber* (156 + EXTRASPACE) + strlen(place)+ 7+ strlen(msg)+ 1057 + strlen(nr_sets)+ 10* 16,
		plen= 0;
	  char *text;
		if( (text= calloc( alloc_len, sizeof(char) )) ){
		  DataSet *this_set= NULL;
				plen= sprintf( text, "%s%s\n Xval: %s\n Yval: %s\n%s%s%s%s%s%s%s%s%s%s%s%s\n X: %s\n LY: %s\n HY: %s\n%s%s%s Pnts: %s\n"
							   " Bounding Box: %s,%s - %s,%s\n",
					msg, place,
					SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SS_Xval),
					SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SS_Yval),
					(wi->SS_E.count)? " E: " : "",
					(wi->SS_E.count)? SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SS_E) : "",
					(wi->SS_E.count)? "\n" : "",
					(wi->SAS_O.pos_count||wi->SAS_O.neg_count)? " O: " : "",
					(wi->SAS_O.pos_count||wi->SAS_O.neg_count)?
						SAS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SAS_O) : "",
					(wi->SAS_O.pos_count||wi->SAS_O.neg_count)? "\n" : "",
					(wi->SS_I.count)? " Int: " : "",
					(wi->SS_I.count)? SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SS_I) : "",
					(wi->SS_I.count)? "\n" : "",
					(wi->SAS_slope.pos_count||wi->SAS_slope.neg_count)? " Slope: " : "",
					(wi->SAS_slope.pos_count||wi->SAS_slope.neg_count)?
						SAS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SAS_slope) :
							((Determine_AvSlope)? "" : " <use -tas to determine average slope>\n"),
					(wi->SAS_slope.pos_count||wi->SAS_slope.neg_count)? "\n" : "",
					SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SS_X),
					SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SS_LY),
					SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SS_HY),
					(wi->SAS_scrslope.pos_count||wi->SAS_scrslope.neg_count)? " Slope (screen): " : "",
					(wi->SAS_scrslope.pos_count||wi->SAS_scrslope.neg_count)?
						SAS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &wi->SAS_scrslope) :
							((Determine_AvSlope)? "" : " <use -tas to determine average slope>\n"),
					(wi->SAS_scrslope.pos_count||wi->SAS_scrslope.neg_count)? "\n" : "",
					SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_Points),
					d2str( Reform_X(wi, wi->UsrOrgX, wi->UsrOrgY), "%.8g", NULL),
					d2str( Reform_Y(wi, wi->UsrOrgY, wi->UsrOrgX), "%.8g", NULL),
					d2str( Reform_X(wi, wi->UsrOppX, wi->UsrOppY), "%.8g", NULL),
					d2str( Reform_Y(wi, wi->UsrOppY, wi->UsrOppX), "%.8g", NULL)
				);
			if( wi->datawin.apply ){
				plen= sprintf( text, "%s DataWin: %s,%s - %s,%s\n", text,
					d2str( wi->datawin.llX, "%.8g", NULL),
					d2str( wi->datawin.llY, "%.8g", NULL),
					d2str( wi->datawin.urX, "%.8g", NULL),
					d2str( wi->datawin.urY, "%.8g", NULL)
				);
			}
			plen= sprintf( text, "%s\n", text );
			for( i= 0; i< NUMSETS; i++ ){
			  char *f= (wi->labels_in_legend)? AllSets[i].YUnits : XGrindex( AllSets[i].fileName, '/'),
					control_a[2]= { 0x01, 0};
				if( !wi->labels_in_legend && f ){
					f++;
				}
				tnP+= AllSets[i].numPoints;
				if( draw_set(wi, i) && (this_one< 0 || i== this_one) ){
				  char linktxt[64];
					N+= 1;
					if( wi->mark_set[i]> 0 ){
						MN+= 1;
					}
					if( wi->legend_line[i].highlight ){
						HN+= 1;
					}
					this_set= &AllSets[i];
					nP+= this_set->numPoints/ ( (this_set->plot_interval<= 0)? 1 : this_set->plot_interval );
					if( this_set->show_legend && this_set->numPoints> 0 ){
					  int j;
#ifdef RUNSPLIT
						SN= 0;
						if( this_set->splithere ){
/* 						  DataSet *this_set= &AllSets[i];	*/
							for( j= 0; j< this_set->numPoints; j++ ){
								if( SplitHere( this_set, j) ){
									SN+= 1;
								}
							}
						}
#endif
						DN= 0;
						if( this_set->discardpoint || (wi->discardpoint && wi->discardpoint[i]) ){
						  int slen;
							if( i== first && i== last ){
							  char *tnam= XGtempnam( getenv("TMPDIR"), "XGSLd");
							  FILE *fp= (tnam)? fopen( tnam, "wb") : NULL;
								if( fp ){
									slen= fprintf( fp, "Set #%d discarded points: ", this_set->set_nr );
									DN= DumpDiscarded_Points( fp, wi, this_set, slen, 100, "\n\n" );
									fclose( fp );
									if( DN && (fp= fopen( tnam, "r")) ){
									  int Len, len= strlen(text);
									  char *c= &text[len], *r;
										unlink(tnam);
										Len= len;
										do{
											r= fgets( c, (alloc_len- len)/ 2, fp );
											len= strlen(text);
											c= &text[len];
										} while( r && !feof(fp) && !ferror(fp) && len< alloc_len );
										fclose( fp );
										if( len- Len> 0 ){
											alloc_len+= len- Len;
											if( !(text= realloc( text, alloc_len ) ) ){
												xtb_error_box( wi->window, "Can't resize textbuffer to %d bytes", "Error" );
												return( NULL );
											}
										}
									}
									else{
										unlink(tnam);
									}
									if( tnam ){
										xfree(tnam);
									}
								}
								else{
									if( tnam ){
										xfree(tnam);
									}
									goto count_discarded;
								}
							}
							else{
count_discarded:;
								for( j= 0; j< this_set->numPoints; j++ ){
									if( DiscardedPoint( wi, this_set, j) ){
										DN+= 1;
									}
								}
							}
						}
						if( this_set->set_linked ){
							sprintf( linktxt, "<-#%-3d", this_set->set_link );
						}
						else{
							linktxt[0]= '\0';
						}
						plen= sprintf( text, "%s%s %s%s%s #%-3d%s %d,%d vs. %d,%d (%d/%d) \\#xa6\\%-3d%s \\#xb1\\ %s (%s)%s%s%s%s%s%s%s\n",
							text,
							(wi->legend_line[i].highlight)? control_a : "",
							(wi->plot_only_file+1== wi->fileNumber[i])? "\\#xed\\ " : "",
							(wi->numVisible[i])? "" : "-(",
							(this_set->nameBuf)? this_set->nameBuf : this_set->setName,
							i,
							linktxt,
							wi->ycol[i], wi->ecol[i], wi->xcol[i], wi->lcol[i],
							wi->numVisible[i], this_set->numPoints,
							this_set->fileNumber,
							(f)? f : "",
							d2str( this_set->av_error, NULL, NULL),
							d2str( this_set->av_error_r, NULL, NULL),
							(this_set->displaced_x)? d2str( this_set->displaced_x, " dx=%g", NULL) : "",
							(this_set->displaced_y)? d2str( this_set->displaced_y, " dy=%g", NULL) : "",
							(wi->mark_set[i]> 0)? " \\#xf0\\" : "",
							(wi->plot_only_set0== i)? " \\#xc4\\" : "",
							(wi->numVisible[i])? "" : ")-",
							(SN)? d2str( (double) SN, " %g spl", NULL) : "",
							(DN)? d2str( (double) DN, " %g dsc", NULL) : ""
						);
					}
				}
			}
			if( wi->show_overlap ){
				if( wi->overlap_buf && strlen(wi->overlap_buf) ){
					(plen= sprintf( text, "%s\n %s\n", text, wi->overlap_buf ));
				}
				if( wi->overlap2_buf && strlen(wi->overlap2_buf) ){
					plen+= sprintf( text, "%s %s\n", text, wi->overlap2_buf );
				}
			}
			plen= sprintf( text, nr_sets, text, N, setNumber, nP, tnP, MN, HN );
			if( N== 1 && this_set && this_set->first_error>=0 && this_set->last_error>=0 ){
				plen= sprintf( text, "%sfirst error #%d = %s ; last error #%d = %s\n", text,
					this_set->first_error, d2str( this_set->errvec[this_set->first_error], NULL, NULL),
					this_set->last_error, d2str( this_set->errvec[this_set->last_error], NULL, NULL)
				);
			}

			if( wi->pen_list ){
			  XGPen *Pen= wi->pen_list;
			  int ndrawn= 0, last_drawn= -1;
				while( Pen ){
					if( Pen->current_pos && (Pen->drawn || !Pen->skip) ){
						ndrawn+= 1;
						last_drawn= Pen->pen_nr;
					}
					Pen= Pen->next;
				}
				plen= sprintf( text, "%s\n%d pen(s) drawn out of %d; last #%d\n",
					text,
					ndrawn, wi->numPens, last_drawn
				);
			}

#ifdef _APOLLO_SOURCE
			plen= strlen( text );
#endif
			if( plen> alloc_len ){
				fprintf( StdErr, "ShowLegends: wrote %d bytes in a buffer of %d (%s)!!\n%s\n",
					plen, alloc_len, serror(),
					text
				);
				xtb_error_box( wi->window, "Buffer overwritten in ShowLegends\nYou should exit now!\n", "Fatal Error" );
			}
			else{
				parse_codes( text );
				if( PopUp ){
				  int id;
				  char *sel= NULL;
				  xtb_frame *menu= NULL;
					id= xtb_popup_menu( wi->window, text, "The legend box", &sel, &menu);

					if( sel ){
						while( *sel && isspace( (unsigned char) *sel) ){
							sel++;
						}
					}
					if( sel && *sel ){
						if( debugFlag ){
							xtb_error_box( wi->window, sel, "Copied to clipboard:" );
						}
						else{
							Boing(10);
						}
						XStoreBuffer( disp, sel, strlen(sel), 0);
						  // RJVB 20081217
						xfree(sel);
					}
					xtb_popup_delete( &menu );
				}
				else{
					rstring= XGstrdup( text);
				}
			}
			xfree( text );
		}
		else{
			xtb_error_box( wi->window, "No memory for displaying the legends\n", "The legend box");
		}
	}
	if( reset ){
		wi->legend_length= 0;
	}
	return(rstring);
}

int display_ascanf_statbins( LocalWin *wi )
{ int i;
  extern SimpleStats *ascanf_SS;
  extern SimpleAngleStats *ascanf_SAS;
  extern char **ascanf_SS_names;
  extern char **ascanf_SAS_names;
  char buf[4096], *name;
  Sinc List;
	if( !wi ){
		return(0);
	}
	List.sinc.string= NULL;
	Sinc_string_behaviour( &List, NULL, 0,0, SString_Dynamic );
	Sflush( &List );
	for( i= 0; ascanf_SS && i< ASCANF_MAX_ARGS; i++ ){
		if( ascanf_SS[i].count ){
			name= (ascanf_SS_names && ascanf_SS_names[i])? ascanf_SS_names[i] : "";
			sprintf( buf, " %s SS[%d]: %s\n", name, i, SS_sprint_full( NULL, d3str_format, " \\#xb1\\ ", 0, &ascanf_SS[i] ) );
			StringCheck( buf, sizeof(buf)/sizeof(char), __FILE__, __LINE__ );
/* 			add_comment( buf );	*/
			Add_SincList( &List, buf, False );
		}
	}
	for( i= 0; ascanf_SAS && i< ASCANF_MAX_ARGS; i++ ){
		if( ascanf_SAS[i].pos_count || ascanf_SAS[i].neg_count ){
			name= (ascanf_SAS_names && ascanf_SAS_names[i])? ascanf_SAS_names[i] : "";
			sprintf( buf, " %s SAS[%d]: %s\n", name, i, SAS_sprint_full( NULL, d3str_format, " \\#xb1\\ ", 0, &ascanf_SAS[i] ) );
			StringCheck( buf, sizeof(buf)/sizeof(char), __FILE__, __LINE__ );
/* 			add_comment( buf );	*/
			Add_SincList( &List, buf, False );
		}
	}
	if( List.sinc.string ){
	  int id;
	  char *sel= NULL;
	  xtb_frame *menu= NULL;
		id= xtb_popup_menu( wi->window, List.sinc.string, "Used statistics and angular statistics bins", &sel, &menu);

		if( sel ){
			while( *sel && isspace( (unsigned char) *sel) ){
				sel++;
			}
		}
		if( sel && *sel ){
			if( debugFlag ){
				xtb_error_box( wi->window, sel, "Copied to clipboard:" );
			}
			else{
				Boing(10);
			}
			XStoreBuffer( disp, sel, strlen(sel), 0);
			  // RJVB 20081217
			xfree(sel);
		}
		xtb_popup_delete( &menu );
		xfree( List.sinc.string);
	}
	else{
		xtb_error_box( wi->window, "None", "Used statistics and angular statistics bins" );
	}
	return(1);
}

int sort_ascanf_Functions( ascanf_Function *a, ascanf_Function *b )
{ char *an= a->name, *bn= b->name;
	if( (an[0]== '$' && bn[0]== '$') || (an[0]!= '$' && bn[0]!= '$') ){
	  int r;
		if( index( "%\\", an[0]) ){
			an++;
		}
		if( index( "%\\", bn[0]) ){
			bn++;
		}
		if( !(r= strcasecmp( an, bn )) ){
			r= strcmp( an, bn );
		}
		return(r);
	}
	  /* Always move names starting with a '$' towards the end of the list	*/
	else if( an[0]== '$' ){
		return( 1 );
	}
	else{
		return(-1);
	}
}

static char *addcbuf(char *cbuf, int *cbuf_len, int *cbl, char *s, int back, int buf_len, char *fname, int lineno )
{ int bl= 0;
	  /* 20030413: the StringCheck must be performed here... 	*/
	StringCheck( s, buf_len, fname, lineno );
	if( back> 0 ){
		*cbl-= back;
		cbuf[*cbl]= '\0';
	}
	if( *cbl+(bl=strlen(s))>= *cbuf_len-1 || !cbuf ){
		if( !*cbuf_len ){
			*cbuf_len= MAX(128,bl);
		}
		while( *cbl+ bl>= *cbuf_len-1 ){
			*cbuf_len*= 2;
		}
		cbuf= XGrealloc(cbuf, *cbuf_len* sizeof(char));
		*cbl= strlen(cbuf);
	}
	if( cbuf ){
		strcat( cbuf, s);
		*cbl+= bl;
	}
	return( cbuf );
}

/* Display all the currently defined variables (scalars, arrays, procedures) in a popup
 \ window, possibly showing some usage (assign/read) statistics (show_info==True). If
 \ search is pointing to a string, a variable with that name is looked for, and if found,
 \ the output normally generated for that variable is returned through the same pointer*.
 \ This memory will have to be de-allocated after use. The routine doesn't generate any
 \ output when used to search. This functionality makes the %[name] opcode in titles possible.
 */
ascanf_Function *foundVar= NULL;
int display_ascanf_variables( Window win, Boolean show_info, Boolean show_dollars, char **search )
{ char /* *cb= comment_buf, */ *space, *newline;
  int /* cs= comment_size, */ nc= NoComment, lines= 0;
  int rootvar= 0, f= 0;
  ascanf_Function *saf= NULL;
  char *buf= NULL;
  int buf_len= LMAXBUFSIZE;
  char *cbuf= NULL;
  int cbl= 0, cbuf_len= 0;
  Sinc List;
  extern ascanf_Function vars_ascanf_Functions[], vars_internal_Functions[];
  extern int ascanf_Functions, internal_Functions;
  extern double *Allocate_Internal;
  int listSize= (*Allocate_Internal)? internal_Functions : ascanf_Functions;
  ascanf_Function *funList= (*Allocate_Internal)? vars_internal_Functions : vars_ascanf_Functions;
  ascanf_Function *af= funList[f].cdr;

/* #define ADDCBUF(s,back)	cbuf=addcbuf(cbuf, &cbuf_len, &cbl, buf,back)	*/
#define ADDCBUF(s,back)	cbuf=addcbuf(cbuf, &cbuf_len, &cbl, (s),back, buf_len, __FILE__, __LINE__)
#define DAV_return(r)	xfree(buf);xfree(cbuf); cbuf_len= 0; GCA(); return(r);
#define CEXPNDBUF(size)	{ int l;\
	if( (l= (size))> buf_len ){ \
		while( buf_len<= l ){ \
			buf_len*= 2; \
		} \
		if( !(buf= (char*) realloc( buf, buf_len* sizeof(char))) ){ \
			DAV_return(0); \
		} \
	} \
}

	foundVar= NULL;

	if( !(buf= (char*) calloc( buf_len, sizeof(char))) ){
		return(0);
	}
	if( !search || !*search ){
/* 		comment_buf= NULL;	*/
/* 		comment_size= 0;	*/
		List.sinc.string= NULL;
		Sinc_string_behaviour( &List, NULL, 0,0, SString_Dynamic );
		Sflush( &List );
		NoComment= False;
		space= " ";
		newline= "\n";
	}
	else{
		newline= space= "";
	}
	if( !af ){
		f+= 1;
		af= &funList[f];
		rootvar= 1;
	}
	if( !search && af ){
	  int i, n= 0;
	  ascanf_Function *f= af;
		  /* Determine the number of elements in this tree:	*/
		while( f ){
			f= f->cdr;
			n++;
		}
		  /* We'd undoubtedly like an alphabetical listing. However, the linkedlist in which
		   \ the variables &c are stored is ordered by increasing hash-value. So. One solution
		   \ implemented here is to make a temporary, local copy in an *array*, sort that
		   \ in the required way ($ variables at the end, some ignoring of initial "special"
		   \ symbols) using qsort. Then update this array's list structure, and use the
		   \ base-address of this array to travel down its tree. It is perfectly safe to
		   \ make a copy of the original tree - *just* the original tree: all the new nodes
		   \ point into the same addresses as the (fields in the) original tree! So it is
		   \ *not* safe to meddle with fields (no new->name[0]= '?', whereas a new->name= "?"
		   \ is OK). Of course, we must de-allocate the array at the end. We use realloc to
		   \ dynamically update the array's size. And we don't (need to) take any of these
		   \ actions if there's only one element in the original tree...
		   \ PS: theoretically, it is possible to include only the "interesting" items
		   \ in the new tree... (i.e. no functions).
		   \ PPS: We perform the same sorting everytime we find a n>1 tree in the
		   \ vars_ascanf_Functions array. That means that there's no complete, global
		   \ alphabet. order. But the nice part is that procedures (which hook onto
		   \ DEPROC instead of onto DCL) are listed after the variables... ;-)
		   */
		if( n> 1 && (saf= XGrealloc( saf, n* sizeof(ascanf_Function))) ){
			f= af;
			i= 0;
			while( f ){
				saf[i]= *f;
				f= f->cdr;
				i+= 1;
			}
			qsort( saf, n, sizeof(ascanf_Function), (void*) sort_ascanf_Functions );
			saf[n-1].cdr= NULL;
			for( i= n-2; i>= 0; i-- ){
				saf[i].cdr= &saf[i+1];
			}
			af= saf;
		}
	}
	while( af ){
		if( (af->type== _ascanf_variable || af->type== _ascanf_array || af->type== _ascanf_procedure ||
				af->type== _ascanf_simplestats || af->type== _ascanf_simpleanglestats
				|| af->type== _ascanf_python_object
			) && 
			(af->name[0]!= '$' || af->dollar_variable || show_dollars)
		){
			if( af->N> 0 && (af->array || af->iarray) ){
			  int j;
				if( af->linkedArray.dataColumn ){
					Check_linkedArray(af);
				}
				if( af->iarray ){
					sprintf( buf, "%s%s== [%d]{%d", space, af->name, af->N, af->iarray[0] );
					ADDCBUF( buf, 0 );
					for( j= 1; j< af->N && j< 48; j++ ){
						sprintf( buf, ",%d", af->iarray[j] );
						ADDCBUF( buf, 0 );
					}
					if( af->N> j ){
						strcpy( buf, ",.." );
						ADDCBUF( buf, 0 );
						for( j= af->N-3; j< af->N; j++ ){
							sprintf( buf, ",%d", af->iarray[j] );
							ADDCBUF( buf, 0 );
						}
					}
				}
				else{
				  ascanf_Function *saf;
				  int take_usage;
					if( (saf= parse_ascanf_address(af->array[0], 0, "display_ascanf_variables()", False, &take_usage)) ){
						if( take_usage ){
							if( saf->internal ){
								CEXPNDBUF(strlen(af->name)+ strlen(saf->usage)+ 64);
								sprintf( buf, "%s%s== [%d]{`\"%s\"", space, af->name, af->N, saf->usage );
							}
							else{
								sprintf( buf, "%s%s== [%d]{`%s", space, af->name, af->N, saf->name );
							}
						}
						else{
							sprintf( buf, "%s%s== [%d]{&%s", space, af->name, af->N, saf->name );
						}
					}
					else{
						sprintf( buf, "%s%s== [%d]{%s", space, af->name, af->N, ad2str( af->array[0], d3str_format, NULL) );
					}
					ADDCBUF( buf, 0 );
					for( j= 1; j< af->N; j++ ){
						if( j< 48 || j>= af->N-3 ){
							if( (saf= parse_ascanf_address(af->array[j], 0, "display_ascanf_variables()", False, &take_usage)) ){
								if( take_usage ){
									if( saf->internal ){
										CEXPNDBUF(strlen(saf->usage)+ 8);
										sprintf( buf, ",`\"%s\"", saf->usage );
									}
									else{
										sprintf( buf, ",`%s", saf->name );
									}
								}
								else{
									sprintf( buf, ",&%s", saf->name );
								}
							}
							else{
								sprintf( buf, ",%s", ad2str( af->array[j], d3str_format, NULL) );
							}
							ADDCBUF( buf, 0 );
						}
						else if( j== 48 ){
							strcpy( buf, ",.." );
							ADDCBUF( buf, 0 );
						}
					}
#ifdef OLD20010110
					if( af->N> j ){
						strcpy( buf, ",.." );
						ADDCBUF( buf, 0 );
						for( j= af->N-3; j< af->N; j++ ){
							if( (saf= parse_ascanf_address(af->array[j], 0, "display_ascanf_variables()", False, &take_usage)) ){
								if( take_usage ){
									if( saf->internal ){
										CEXPNDBUF(strlen(saf->usage)+ 8);
										sprintf( buf, ",`\"%s\"", saf->usage );
									}
									else{
										sprintf( buf, ",`%s", saf->name );
									}
								}
								else{
									sprintf( buf, ",&%s", space, af->name, af->N, saf->name );
								}
							}
							else{
								sprintf( buf, ",%s", ad2str( af->array[j], d3str_format, NULL) );
							}
							ADDCBUF( buf, 0 );
						}
					}
#endif
				}
				sprintf( buf, "}[%d]== %s%s", af->last_index, ad2str( af->value, d3str_format, NULL), newline );
				ADDCBUF( buf, 0 );
			}
			else if( af->procedure ){
			  char *c= ascanf_ProcedureCode(af), *d;
			  int l;
				CEXPNDBUF( ((c)? strlen(c) : 0 + strlen(af->name)) );
				l= sprintf( buf, "%s%s== ", space, af->name );
				d= &buf[l];
				while( c && *c ){
/* 
					if( *c== '\n' ){
						A
						*d++= '\\';
						*d++= '\\';
						*d++= 'n';
					}
					else
 */
					if( *c== '\t' ){
						*d++= ' ';
					}
					else{
						*d++= *c;
					}
					c++;
				}
				*d++= '\0';
				ADDCBUF( buf, 0 );
				sprintf( buf, "== %s%s%s",
					ad2str( af->value, d3str_format, NULL),
					(af->dialog_procedure)? " (dialog)" : "",
					newline
				);
				ADDCBUF( buf, 0 );
			}
			else if( af->type== _ascanf_simplestats ){
			  SimpleStats *SS;
				if( af->N> 1 ){
					if( af->last_index>= 0 && af->last_index< af->N ){
						SS= &af->SS[af->last_index];
					}
					else{
						SS= NULL;
					}
					sprintf( buf, "%s%s[%d:%d]== ", space, af->name, af->N, af->last_index );
				}
				else{
					sprintf( buf, "%s%s== ", space, af->name );
					SS= af->SS;
				}
				ADDCBUF( buf, 0 );
				if( SS ){
					if( SS ){
						SS_sprint( buf, d3str_format, " #xb1 ", 0, SS );
					}
					else{
						strcpy( buf, " <NULL!> " );
					}
					ADDCBUF( buf, 0 );
					sprintf( buf, " [%d] == %s%s", (SS)? SS->count : -1, ad2str( af->value, d3str_format, NULL), newline );
					ADDCBUF( buf, 0 );
				}
				else{
					sprintf( buf, " == %s%s", ad2str( af->value, d3str_format, NULL), newline );
					ADDCBUF( buf, 0 );
				}
			}
			else if( af->type== _ascanf_simpleanglestats ){
			  SimpleAngleStats *SAS;
				if( af->N> 1 ){
					if( af->last_index>= 0 && af->last_index< af->N ){
						SAS= &af->SAS[af->last_index];
					}
					else{
						SAS= NULL;
					}
					sprintf( buf, "%s%s[%d:%d]== ", space, af->name, af->N, af->last_index );
				}
				else{
					sprintf( buf, "%s%s== ", space, af->name );
					SAS= af->SAS;
				}
				ADDCBUF( buf, 0 );
				if( SAS ){
					if( SAS ){
						SAS_sprint( buf, d3str_format, " #xb1 ", 0, SAS );
					}
					else{
						strcpy( buf, " <NULL!> " );
					}
					ADDCBUF( buf, 0 );
					sprintf( buf, " [%d] == %s%s",
						(SAS)? SAS->pos_count + SAS->neg_count : -1,
						ad2str( af->value, d3str_format, NULL), newline
					);
					ADDCBUF( buf, 0 );
				}
				else{
					sprintf( buf, " == %s%s", ad2str( af->value, d3str_format, NULL), newline );
					ADDCBUF( buf, 0 );
				}
			}
			else{
			  ascanf_Function *pf;
				if( af->is_address ){
					pf= parse_ascanf_address( af->value, 0, "display_ascanf_variables", (int) ascanf_verbose, NULL );
				}
				else{
					pf= NULL;
				}
				if( pf ){
					sprintf( buf, "%s%s== %c%s%s",
						space, af->name,
						(af->is_usage)? '`' : '&',
						pf->name, newline
					);
				}
				else{
					sprintf( buf, "%s%s== %s%s", space, af->name, ad2str( af->value, d3str_format, NULL), newline );
				}
				ADDCBUF( buf, 0 );
			}
			if( show_info ){
				sprintf( buf, " A=%d R=%d L=%d%s", af->assigns, af->reads, af->links, newline );
				ADDCBUF( buf, 1 );
			}
			if( !rootvar && !(search && *search) ){
				if( af->usage ){
				  char *c;
					CEXPNDBUF( strlen(af->usage)+ 16);
					sprintf( buf, "  # \"%s\"", af->usage );
					if( *(c= &buf[strlen(buf)-2])== '\n' ){
						*c= '"';
						c[1]= '\0';
					}
					else{
						strcat( buf, "\n" );
					}
					ADDCBUF( buf, 0 );
				}
				if( af->type==_ascanf_array && af->linkedArray.dataColumn ){
					CEXPNDBUF( 256 );
					sprintf( buf, "  # array pointing to set #%d, column %d", af->linkedArray.set_nr, af->linkedArray.col_nr );
					ADDCBUF( buf, 0);
				}
				if( af->dymod && af->dymod->name && af->dymod->path ){
					CEXPNDBUF( strlen(af->dymod->name)+ strlen(af->dymod->path)+ 16);
					sprintf( buf, "  # Module={\"%s\",%s}", af->dymod->name, af->dymod->path );
					ADDCBUF( buf, 0);
				}
				if( af->fp ){
					sprintf( buf, "  # open file fd=%d", fileno(af->fp) );
					ADDCBUF( buf, 0 );
				}
				else if( af->type== _ascanf_python_object && af->PyObject_Name ){
					CEXPNDBUF( strlen(af->PyObject_Name)+ 32);
					sprintf( buf, " # PyObject %p \"%s\"", af->PyObject, af->PyObject_Name );
					ADDCBUF( buf, 0 );
				}
				if( af->cfont ){
					CEXPNDBUF( strlen((af->cfont->is_alt_XFont)? "(alt.) " : "") +
						strlen(af->cfont->XFont.name)+ strlen(af->cfont->PSFont)+ 128 );
					sprintf( buf, "  # cfont: %s\"%s\"/%s[%g]",
						(af->cfont->is_alt_XFont)? "(alt.) " : "",
						af->cfont->XFont.name, af->cfont->PSFont, af->cfont->PSPointSize
					);
					ADDCBUF( buf, 0 );
				}
				if( af->label ){
				  char *c;
					CEXPNDBUF( strlen(af->label)+ 16);
					sprintf( buf, "  # label={%s}", af->label );
					if( *(c= &buf[strlen(buf)-2])== '\n' ){
						*c= '"';
						c[1]= '\0';
					}
					else{
						strcat( buf, "\n" );
					}
					ADDCBUF( buf, 0 );
				}
			}
			if( af->accessHandler && !(search && *search) ){
			  ascanf_Function *aH= af->accessHandler;
			  double *aH_par= af->aH_par;
				switch( aH->type ){
					case _ascanf_variable:
						sprintf( buf, "  # AccessHandler: var %s=%g%s",
							aH->name, aH_par[0], newline
						);
						break;
					case _ascanf_array:
						sprintf( buf, "  # AccessHandler: array %s[%g]=%g%s",
							aH->name, aH_par[0], aH_par[1], newline
						);
						break;
					case _ascanf_procedure:{
						sprintf( buf, "  # AccessHandler: call proc %s%s",
							aH->name, newline
						);
						break;
					}
					case NOT_EOF:
					case NOT_EOF_OR_RETURN:
					case _ascanf_function:{
						sprintf( buf, "  # AccessHandler: call %s[%g,%g]%s",
							aH->name, aH_par[0], aH_par[1], newline
						);
						break;
					}
					default:
						sprintf( buf, "  # AccessHandler %s: obsolete/invalid, removed%s",
							aH->name, newline
						);
						af->accessHandler= NULL;
						break;
				}
				ADDCBUF( buf, 0 );
			}
			StringCheck( cbuf, cbuf_len/sizeof(char), __FILE__, __LINE__ );
			if( search && *search ){
				if( strcmp( *search, af->name)== 0 ){
					foundVar= af;
					if( (*search= strdup(cbuf)) ){
						DAV_return(1);
					}
					else{
						DAV_return(0);
					}
				}
			}
			else{
				Add_SincList( &List, cbuf, False );
				lines+= 1;
			}
			  /* Clear the cumulative buffer:	*/
			*cbuf= '\0';
			cbl= 0;
		}
		if( !(af= af->cdr) ){
			f+= 1;
			if( f< listSize ){
				af= &funList[f];
				rootvar= 1;
				if( !search && af ){
				  int i, n= 0;
				  ascanf_Function *f= af;
					while( f ){
						f= f->cdr;
						n++;
					}
					if( n> 1 && (saf= XGrealloc( saf, n* sizeof(ascanf_Function))) ){
						f= af;
						i= 0;
						while( f ){
							saf[i]= *f;
							f= f->cdr;
							i+= 1;
						}
						  /* Sort the list, but maintain the head at the 1st position!	*/
						qsort( &saf[1], n-1, sizeof(ascanf_Function), (void*) sort_ascanf_Functions );
						saf[n-1].cdr= NULL;
						for( i= n-2; i>= 0; i-- ){
							saf[i].cdr= &saf[i+1];
						}
						af= saf;
					}
				}
			}
		}
		else{
			rootvar= 0;
		}
	}
	xfree( saf );
	if( !search ){
		if( List.sinc.string ){
		  int id;
		  char *sel= NULL;
		  xtb_frame *menu= NULL;
			id= xtb_popup_menu( win, List.sinc.string, "Defined Variables + Values", &sel, &menu);

			if( sel && sel[0]== ' ' ){
			  char *c= index( sel, '=');
				if( c && c[1]== '=' && c[2]== ' ' ){
					*c= '\0';
				}
				if( debugFlag ){
					xtb_error_box( win, &sel[1], "Copied to clipboard:" );
				}
				else{
					Boing(10);
				}
				XStoreBuffer( disp, &sel[1], strlen(&sel[1]), 0);
				  // RJVB 20081217
				xfree(sel);
			}
			xtb_popup_delete( &menu );
			xfree( List.sinc.string);
		}
		else{
			xtb_error_box( win, "None", "Defined Variables + Values" );
		}
/* 		comment_buf= cb;	*/
/* 		comment_size= cs;	*/
		NoComment= nc;
	}
	GCA();
	if( search ){
	  /* If we're here, that means that we didn't find a variable matching the search pattern.
	   \ Set the search argument to NULL to attest to that fact.
	   */
		*search= NULL;
	}
	DAV_return(1);
}

int DBG_ascanf_variables( Boolean show_info, Boolean show_dollars, char *search )
{
	return( display_ascanf_variables( StubWindow.window, show_info, show_dollars, (search)? &search : NULL ) );
}

char *pvno_found_env_var= NULL;

/* get a var=val string from an opcode like %[name]. The opcode
 \ should be a pointer to name (hence &"%[name]"[2] in the above
 \ example, and end_mark should contain the character marking
 \ the opcode's end (']'). It can be 0, in which case the whole
 \ opcode string is used. When supplied, the end of the opcode
 \ as used is returned in end_found.
 */
char *parse_varname_opcode( char *opcode, int end_mark, char **end_found, int internal_too )
{ char *searchvar= NULL, *c;
	if( end_mark ){
		c= index( opcode, end_mark );
	}
	else{
		c= &opcode[ strlen(opcode) ];
	}
	if( end_found ){
		*end_found= c;
	}
	  /* Let's find a matching end brace.	*/
	if( c ){
	  extern double *Allocate_Internal;
	  double ai= *Allocate_Internal;
	  int redo;
		*c= '\0';
		  /* extract 'name' */
		searchvar= strdup( opcode );
		pvno_found_env_var= XGstrdup(getenv(searchvar));
		*c= end_mark;
		c= searchvar;
		*Allocate_Internal= 0;
		  /* See if we have a variable by that name: a textual representatin
		   \ of its value(s) is returned in searchvar. A NULL searchvar upon
		   \ return means that no match was found - a 0 return from display_...()
		   \ indicates an allocation problem.
		   */
		do{
			redo= False;
			if( !display_ascanf_variables( 0, False, True, &searchvar ) || !searchvar ){
				if( internal_too ){
					searchvar= c;
					*Allocate_Internal= 1;
					redo= True;
					internal_too= False;
				}
				else{
					searchvar= NULL;
					xfree( c );
				}
			}
		} while( redo );
		*Allocate_Internal= ai;
	}
	return( searchvar );
}

SimpleStats overlapMO, overlapSO;

/* Base routine for calculating the figural distance as defined by Conditt &al, J. Neurophysiol 78-1 1997.	*/
double _fig_dist( LocalWin *wi, DataSet *set1, DataSet *set2, double *SumDist, int *K, int raw, int orn_handling )
{ int k, l, newmin;
  double dist, mindist;
  double curve_len1= 1, error_len1= 1;
  double curve_len2= 1, error_len2= 1;

	if( wi ){
#ifdef TR_CURVE_LEN
		if( wi->tr_curve_len && !raw ){
			curve_len1= wi->tr_curve_len[set1->set_nr][set1->numPoints];
			curve_len2= wi->tr_curve_len[set2->set_nr][set2->numPoints];
		}
		else
#endif
		if( wi->curve_len ){
			curve_len1= wi->curve_len[set1->set_nr][set1->numPoints];
			curve_len2= wi->curve_len[set2->set_nr][set2->numPoints];
		}
		if( wi->error_len ){
			error_len1= wi->error_len[set1->set_nr][set1->numPoints];
			error_len2= wi->error_len[set2->set_nr][set2->numPoints];
		}
	}
	*SumDist= 0;
	*K= 0;
	for( k= 0; k< set1->numPoints; k++ ){
		if( !DiscardedPoint( wi, set1, k) ){
		  double x1, y1, o1;
			if( raw ){
				x1= XVAL( set1, k);
				y1= YVAL( set1, k);
				o1= ERROR( set1, k);
			}
			else{
				x1= set1->xvec[k];
				y1= set1->yvec[k];
				o1= set1->errvec[k];
			}
			newmin= True;
			for( l= 0; l< set2->numPoints; l++ ){
				if( !DiscardedPoint( wi, set2, l) ){
				  double x2, y2, o2;
					if( raw ){
						x2= XVAL( set2, l);
						y2= YVAL( set2, l);
						o2= ERROR( set2, l);
					}
					else{
						x2= set2->xvec[l];
						y2= set2->yvec[l];
						o2= set2->errvec[l];
					}
					if( orn_handling== 2 ){
					  /* Calculate the difference between 2 angles: */
					  extern double radix;
					  double rdx= (wi)? wi->radix : radix;
						if( !rdx ){
							rdx= M_2PI;
						}
						dist= fabs(o2 - o1)/ rdx;
						if( NaN(dist) ){
							dist= 0;
						}
						else if( dist> 0.5 ){
							dist= fabs( dist- 1 );
						}
						mindist= (newmin)? dist : MIN(mindist, dist);
					}
					else if( orn_handling== 0 || NaN(o1) || NaN(o2) ){
						x2-= x1;
						y2-= y1;
						dist= sqrt( x2*x2 + y2*y2 );
						mindist= (newmin)? dist : MIN(mindist, dist);
					}
					else{
						x2-= x1;
						y2-= y1;
						o2-= o1;
						dist= sqrt( x2*x2 + y2*y2 + o2*o2 );
						mindist= (newmin)? dist : MIN(mindist, dist);
					}
					newmin= False;
				}
			}
			if( !newmin ){
				*K+= 1;
				*SumDist+= mindist;
			}
		}
	}
	return( *SumDist );
}

double Calculate_SetOverlap( LocalWin *wi, DataSet *set1, DataSet *set2, SimpleStats *O1, SimpleStats *O2, double *weight, int *overlap_type, int all_vectors )
{ int k, NN= MIN(set1->numPoints, set2->numPoints), doit, ovl= *overlap_type;
  double len1, len2, overlap2= -1, _overlap= 0;
  double SYmin1, SYmax1, SEmin1, SEmax1;
  double SYmin2, SYmax2, SEmin2, SEmax2;
  double Yenvelope= 0, Eenvelope= 0, hy, ly;
  Boolean Eok= False;
  int etype1= wi->error_type[set1->set_nr], etype2= wi->error_type[set2->set_nr];
  SimpleStats ssw;

	ssw.exact= 0;
	if( (etype1== 4 || etype1== INTENSE_FLAG || etype1== MSIZE_FLAG ) &&
		(etype2== 4 || etype2== INTENSE_FLAG || etype2== MSIZE_FLAG )
	){
		ovl= *overlap_type+ 2;
	}
	switch( ovl ){
		case 3:
		case 1:
			SYmin1= set1->ripeY.min;
			SYmax1= set1->ripeY.max;
			SEmin1= set1->ripeE.min;
			SEmax1= set1->ripeE.max;
			break;
		default:
			SYmin1= set1->rawY.min;
			SYmax1= set1->rawY.max;
			SEmin1= set1->rawE.min;
			SEmax1= set1->rawE.max;
			break;
	}
	if( ovl== 3 || ovl== 1 ){
		SYmin2= set2->ripeY.min;
		SYmax2= set2->ripeY.max;
		SEmin2= set2->ripeE.min;
		SEmax2= set2->ripeE.max;
	}
	else{
		SYmin2= set2->rawY.min;
		SYmax2= set2->rawY.max;
		SEmin2= set2->rawE.min;
		SEmax2= set2->rawE.max;
	}
	if( (SYmin1<= SYmax2 || SEmin1<= SEmax2) && (SYmin2<= SYmax1 || SEmin2<= SEmax1) ){
	  double Ymin= MIN( SYmin1, SYmin2), Emin= MIN( SEmin1, SEmin2 );
	  double Ymax= MAX( SYmax1, SYmax2), Emax= MAX( SEmax1, SEmax2 );
		if( (Yenvelope= fabs(Ymax- Ymin))== 0 ){
			Yenvelope= 1;
		}
		if( (Eenvelope= fabs(Emax- Emin))== 0 ){
			Eenvelope= 1;
		}
		Eok= (wi->use_errors && set1->has_error && set1->use_error && set2->has_error && set2->use_error)?
			True: False;
	}
	if( all_vectors ){
		doit= 1;
	}
	else switch( ovl ){
		case 4:
		case 3:
			if( (SYmin1<= SYmax2 || SEmin1<= SEmax2) && (SYmin2<= SYmax1 || SEmin2<= SEmax1) ){
				doit=1;
			}
			else{
				doit= 0;
			}
			break;
		case 2:
		case 1:
		default:
			doit= 1;
			break;
	}
	if( !doit ){
		return(-1);
	}

	SS_Reset(O1);
	  /* 20020428! */
	SS_Reset(O2);
#ifndef ADVANCED_STATS
	*weight= (double)(set1->NumObs + set2->NumObs);
	if( !*weight){
		*weight= 1.0;
	}
#endif
	  /* Calculate the average overlap (between the error-bars) of the current
	   \ two sets.
	   */
	for( k= 0; k< NN; k++ ){
		if( !DiscardedPoint( wi, set1, k) && !DiscardedPoint( wi, set2, k) ){
		  int etype1= wi->error_type[set1->set_nr],
			etype2= wi->error_type[set2->set_nr];
#if ADVANCED_STATS == 1
			*weight= (double)(set1->N[k] + set2->N[k]);
#elif ADVANCED_STATS == 2
			*weight= NVAL(set1, k)+ NVAL(set2, k);
#endif
#ifdef ADVANCED_STATS
			if( !*weight){
				*weight= 1.0;
			}
#endif
			if( !all_vectors ){
				if( (etype1== 4 || etype1== INTENSE_FLAG || etype1== MSIZE_FLAG) &&
					(etype2== 4 || etype2== INTENSE_FLAG || etype2== MSIZE_FLAG)
				){
					len1= 0;
					len2= 0;
				}
				else if( ovl== 2 ){
					len1= (etype1== 4 || etype1== INTENSE_FLAG || etype1== MSIZE_FLAG)?
							0 : 2* fabs( ERROR( set1, k) );
					len2= (etype2== 4 || etype2== INTENSE_FLAG || etype2== MSIZE_FLAG)?
							0 : 2* fabs( ERROR( set2, k) );
				}
				else{
/* 									len1= fabs( set1->hdyvec[k]- set1->ldyvec[k] );	*/
/* 									len2= fabs( set2->hdyvec[k]- set2->ldyvec[k] );	*/
					len1= (etype1== 4 || etype1== INTENSE_FLAG || etype1== MSIZE_FLAG)?
							0 : 2* fabs( set1->errvec[k] );
					len2= (etype2== 4 || etype2== INTENSE_FLAG || etype2== MSIZE_FLAG)?
							0 : 2* fabs( set2->errvec[k] );
				}
				if( !len1 && !len2 ){
					switch( ovl ){
						case 4:
							_overlap= ((YVAL( set1, k)== YVAL( set2, k)) +
								(ERROR(set1, k)== ERROR(set2, k))) / 2.0;
							break;
						case 3:
							_overlap= ((set1->yvec[k]== set2->yvec[k]) +
								(set1->errvec[k]== set2->errvec[k]))/ 2.0;
							break;
						case 2:
							_overlap= (YVAL( set1, k)== YVAL( set2, k))? 1.0 : 0.0;
							break;
						case 1:
							_overlap= (set1->yvec[k]== set2->yvec[k])? 1.0 : 0.0;
							break;
					}
				}
				else if( !len1 ){
					if( ovl== 2 ){
						_overlap= (YVAL( set1, k)>= (YVAL( set2, k)- ERROR( set2, k)) &&
							YVAL( set1, k)<= (YVAL( set2, k)+ ERROR( set2, k)))? 1.0 : 0.0;
					}
					else{
						_overlap= (set1->yvec[k]>= set2->yvec[k]- set2->errvec[k] &&
							set1->yvec[k]<= set2->yvec[k]+ set2->errvec[k])? 1.0 : 0.0;
					}
				}
				else if( !len2 ){
					if( ovl== 2 ){
						_overlap= (YVAL( set2, k)>= (YVAL( set1, k)- ERROR( set1, k)) &&
							YVAL( set2, k)<= (YVAL( set1, k)+ ERROR( set1, k)))? 1.0 : 0.0;
					}
					else{
						_overlap= (set2->yvec[k]>= set1->yvec[k]- set1->errvec[k] &&
							set2->yvec[k]<= set1->yvec[k]+ set1->errvec[k])? 1.0 : 0.0;
					}
				}
				else{
					if( ovl== 2 ){
						hy= MIN(( YVAL( set1, k)+ ERROR( set1, k)),( YVAL( set2, k)+ ERROR( set2, k)));
						ly= MAX(( YVAL( set1, k)- ERROR( set1, k)),( YVAL( set2, k)- ERROR( set2, k)));
					}
					else{
						hy= MIN(set1->hdyvec[k],set2->hdyvec[k]);
						ly= MAX( set1->ldyvec[k], set2->ldyvec[k]);
					}
					if( hy-ly >= 0 ){
						_overlap= ( hy - ly) / MIN( len1, len2 );
					}
					else{
					 /* There is no overlap between the two errorbars: hy-ly
					  \ now represents the distance between top resp. bottom
					  \ of the two errorbars. Since there doesn't seem to be
					  \ an intuitive normalisor for this (maximum distance?),
					  \ we set overlap to 0 in this case (which *is* logical).
					  \ Would it be a good idea to allow for a "negative" overlap,
					  \ such that it can be classified how much two sets are apart?
					  */
						_overlap= 0.0;
					}
				}
				SS_Add_Data( O1, 1, _overlap, *weight );
				SS_Add_Data_(ssw, 1, *weight, 1 );
			}
			{ int n= 0;
				switch( ovl ){
					case 4:
					case 2:
						if( Yenvelope ){
							overlap2= 1.0- fabs(YVAL(set1,k)- YVAL(set2,k))/ Yenvelope;
							n+= 1;
						}
						if( Eenvelope && Eok ){
							overlap2+= 1.0- fabs(EVAL(set1,k)- EVAL(set2,k))/ Eenvelope;
							n+= 1;
						}
						if( n ){
							SS_Add_Data( O2, 1, overlap2/ n, *weight );
							SS_Add_Data_(ssw, 1, *weight, 1 );
						}
						break;
					case 3:
					case 1:
						if( Yenvelope ){
							overlap2= 1.0- fabs(set1->yvec[k]- set2->yvec[k])/ Yenvelope;
							n+= 1;
						}
						if( Eenvelope && Eok ){
							overlap2+= 1.0- fabs(set1->errvec[k]- set2->errvec[k])/ Eenvelope;
							n+= 1;
						}
						if( n ){
							SS_Add_Data( O2, 1, overlap2/ n, *weight );
							SS_Add_Data_(ssw, 1, *weight, 1 );
						}
						break;
				}
			}
		}
	}
	*overlap_type= ovl;
	*weight= SS_Mean_(ssw);
	return( SS_Mean(O1) );
}

double overlap( LocalWin *wi )
{  int i, j, ovl= wi->show_overlap;
   double _overlap;
	set_NaN(_overlap);
	if( ovl ){
      SimpleStats O1= EmptySimpleStats, O2= EmptySimpleStats, MO2= EmptySimpleStats, SO2= EmptySimpleStats,
	  	nO1= EmptySimpleStats, nO2= EmptySimpleStats;
      DataSet *set1, *set2;
      double weight;
      int drawn_highlights= 0;
      Boolean all_vectors= True;
		SS_Reset_(overlapMO);
		SS_Reset_(overlapSO);
		for( i= 0; i< setNumber; i++ ){
			if( draw_set(wi,i) ){
				if( wi->legend_line[i].highlight ){
					drawn_highlights+= 1;
				}
				if( wi->error_type[i]!= 4 ){
					all_vectors= False;
				}
			}
		}
		if( drawn_highlights ){
			drawn_highlights-= 1;
		}
		for( i= 0; i< setNumber; i++ ){
			if( draw_set(wi, i) && (!drawn_highlights || (wi->legend_line[i].highlight)) ){
				set1= &AllSets[i];
				for( j= i+1; j< setNumber; j++ ){
					set2= &AllSets[j];
					if( draw_set(wi, j) && (!drawn_highlights || (wi->legend_line[j].highlight)) &&
						(_overlap= Calculate_SetOverlap( wi, set1, set2, &O1, &O2, &weight, &ovl, all_vectors ))>= 0
					){

						if( all_vectors ){
						  int K, L;
						  double SumDist1, SumDist2;
							  /* think of a sensible way to use NVAL() in this figural distance measure...	*/
							weight= 1;
							_fig_dist( wi, set1, set2, &SumDist1, &K, (ovl!= 1 && ovl!= 3), 0 );
							_fig_dist( wi, set2, set1, &SumDist2, &L, (ovl!= 1 && ovl!= 3), 0 );
							SS_Add_Data_( overlapMO, 1, (SumDist1 + SumDist2)/ (K + L), weight );
							_fig_dist( wi, set1, set2, &SumDist1, &K, (ovl!= 1 && ovl!= 3), 2 );
							_fig_dist( wi, set2, set1, &SumDist2, &L, (ovl!= 1 && ovl!= 3), 2 );
							SS_Add_Data_( overlapSO, 1, (SumDist1 + SumDist2)/ (K + L), weight );
							SS_Add_Data_( nO1, 1, (K+L)/2, 1 );
						}
						if( debugFlag ){
							fprintf( StdErr, "overlap() between set #%d%s and #%d%s (over %ld points): %s [%d highlighted]\n",
								i, (wi->legend_line[i].highlight)? "(hl)" : "",
								j, (wi->legend_line[j].highlight)? "(hl)" : "", O1.count,
								SS_sprint_full( NULL, "%g", "+-", 0.0, &O1),
								drawn_highlights
							);
						}
						  /* Add the average overlap and its standard deviation to the global	*/
						if( O1.count && !all_vectors ){
							SS_Add_Data_( overlapMO, 1, SS_Mean_(O1), 1.0);
							SS_Add_Data_( overlapSO, 1, SS_St_Dev_(O1), 1.0);
							SS_Add_Data_( nO1, 1, O1.count, 1 );
						}
						if( O2.count ){
							SS_Add_Data_( MO2, 1, SS_Mean_(O2), 1.0);
							SS_Add_Data_( SO2, 1, SS_St_Dev_(O2), 1.0);
							SS_Add_Data_( nO2, 1, O2.count, 1 );
						}
					}
				}
			}
		}
		if( overlapMO.count ){
			if( !wi->overlap_buf ){
				wi->overlap_buf= calloc( 256, sizeof(char) );
			}
			if( wi->overlap_buf ){
				if( all_vectors ){
					sprintf( wi->overlap_buf, "%sigural distance: (%s \\#xb1\\ %s) \\#xd0\\ (%s \\#xb1\\ %s) [%s \\#xb1\\ %s]",
						(wi->show_overlap==2)? "Raw f" : "F",
						d2str( SS_Mean_(overlapMO), "%g", NULL), d2str( SS_St_Dev_(overlapMO), "%g", NULL),
						d2str( SS_Mean_(overlapSO), "%g", NULL), d2str( SS_St_Dev_(overlapSO), "%g", NULL),
						d2str( SS_Mean_(nO1), 0,0), d2str( SS_St_Dev_(nO1), 0,0)
					);
				}
				else{
					sprintf( wi->overlap_buf, "%sverlap: (%s \\#xb1\\ %s) \\#xb1\\ (%s \\#xb1\\ %s) [%s \\#xb1\\ %s]",
						(wi->show_overlap==2)? "Raw o" : "O",
						d2str( SS_Mean_(overlapMO), "%g", NULL), d2str( SS_St_Dev_(overlapMO), "%g", NULL),
						d2str( SS_Mean_(overlapSO), "%g", NULL), d2str( SS_St_Dev_(overlapSO), "%g", NULL),
						d2str( SS_Mean_(nO1), 0,0), d2str( SS_St_Dev_(nO1), 0,0)
					);
				}
				StringCheck( wi->overlap_buf, 256, __FILE__, __LINE__ );
				parse_codes( wi->overlap_buf );
			}
		}
		else if( wi->overlap_buf ){
			wi->overlap_buf[0]= '\0';
		}
		if( MO2.count ){
			if( !wi->overlap2_buf ){
				wi->overlap2_buf= calloc( 256, sizeof(char) );
			}
			if( wi->overlap2_buf ){
				sprintf( wi->overlap2_buf, "%sloseness: (%s \\#xb1\\ %s) \\#xb1\\ (%s \\#xb1\\ %s) [%s \\#xb1\\ %s]",
					(wi->show_overlap==2)? "Raw c" : "C",
					d2str( SS_Mean_(MO2), "%g", NULL), d2str( SS_St_Dev_(MO2), "%g", NULL),
					d2str( SS_Mean_(SO2), "%g", NULL), d2str( SS_St_Dev_(SO2), "%g", NULL),
					d2str( SS_Mean_(nO2), 0,0), d2str( SS_St_Dev_(nO2), 0,0)
				);
				StringCheck( wi->overlap2_buf, 256, __FILE__, __LINE__ );
				parse_codes( wi->overlap2_buf );
			}
		}
		else if( wi->overlap2_buf ){
			wi->overlap2_buf[0]= '\0';
		}
	}
	return( _overlap );
}

int increment_height( LocalWin *wi, char *text, char letter)
{ int h= wi->dev_info.legend_height, n= 1;
  char *c= text;
	while( c && *c ){
		if( *c== letter && c[1] ){
		  /* The last '\n' is redundant: it does not have an effect	*/
			h+= wi->dev_info.legend_height;
			n+= 1;
		}
		c++;
	}
	if( debugFlag ){
		fprintf( StdErr, "DrawLegend(increment_height()): string \"%s\" is %d high (%d lines)\n",
			text, h, n
		);
	}
	return( h );
}

/* The printing width of the data-window	*/
double window_width, window_height;
double data_width, data_height;
int hardcopy_init_error= 0;

int XGbackup( char *source, char *backup, int copy )
{ int r= -1;
	if( source && *source && backup && *backup ){
		if( !copy ){
			if( (r= rename( source, backup )) ){
				if( errno== EXDEV || errno== EMLINK ){
					copy= True;
				}
			}
		}
		  /* copy may have been set to True after a failed rename(), so redo the check: */
		if( copy ){
		  ALLOCA( command, char, (10+ strlen(backup)+ strlen(source)), clen );
			if( command ){
				sprintf( command, "cp -p %s %s", source, backup );
				r= system( command );
			}
			else{
				errno= ENOMEM;
				r= -1;
			}
		}
	}
	else{
		errno= EFAULT;
	}
	return(r);
}

#include <utime.h>

int set_file_times( char *filename, time_t atime, time_t mtime )
{ struct utimbuf tb;
  int r;
	tb.actime= atime;
	tb.modtime= mtime;
	errno= 0;
	sync();
	utime( filename, &tb );
	r= errno;
	sync();
	return(r);
}

#ifdef TOOLBOX
/*ARGSUSED*/
int do_hardcopy(char *prog, void *info, int (*init_fun)(), char *dev_spec, char *file_or_dev, int append,
		double *maxheight, double *maxwidth, int orientation,
		char *ti_fam, double ti_size, char *le_fam, double le_size, char *la_fam, double la_size, char *ax_fam, double ax_size
)
/*
 * This routine resets the function pointers to those specified
 * by `init_fun' and causes a screen redisplay.  If `dev_spec'
 * is non-zero,  it will be considered a sprintf string with
 * one %s which will be filled in with `file_or_dev' and fed
 * to popen(3) to obtain a stream.  Otherwise,  `file_or_dev'
 * is considered to be a file and is opened for writing.  The
 * resulting stream is fed to the initialization routine for
 * the device.
 */
{ LocalWin *curWin = (LocalWin *) info;
  LocalWin thisWin, *AW= ActiveWin;
  FILE *out_stream;
  char ierr[ERRBUFSIZE];
  ALLOCA( buf, char, LMAXBUFSIZE+2, buf_len);
  ALLOCA( err, char, LMAXBUFSIZE+2, err_len);
    /* ${HOME} can't be larger than longest pathname	*/
  ALLOCA( tilde, char, LMAXBUFSIZE+MAXPATHLEN, tilde_len);
  char *orient;
  int i, final_w, final_h;
    /* 950704: plot_area_[xy]= 1!	*/
  double ratio, plot_area_x= 1, plot_area_y= 1;
  int adapt_width= 1, area_w, area_h, locked= 0, fd,
	  spax= scale_plot_area_x, spay= scale_plot_area_y;
  char *_Title, *fn= NULL;
  double aspect;
  int prs_scrn_asp= preserve_screen_aspect, rdr= curWin->redraw;
  FILE *afp= ascanf_XGOutput->fp;
  int afpp= ascanf_XGOutput->fp_is_pipe;
  char *affn= ascanf_XGOutput->usage;
  char *backup= NULL;
  time_t org_atime, org_mtime;
  int exists= False;
  extern int Use_HO_Previous_TC;
  extern psUserInfo HO_Previous_psUI;
#ifndef DIRECT_PIPING
  char *tempipeName= NULL;
#endif

	PIPE_error= False;
	hardcopy_init_error= 0;
	tilde[0]= '\0';
    if (dev_spec) {
		sprintf(buf, dev_spec, file_or_dev);

HARDCOPY_POPEN:;
		if( backup && *tilde ){
			if( XGbackup( tilde, backup, append ) ){
				if( strcmp( tilde, "/dev/null") ){
					fprintf( StdErr, "do_hardcopy(): Can't backup '%s' to '%s' (%s)\n", tilde, backup, serror() );
				}
				xfree( backup );
			}
		}

#ifdef DIRECT_PIPING
/* 20030222: Eureka: my Linux box causes xgraph to crash if we start writing to a command that
 \ doesn't exist. sh as called by popen() will warn, but there is no way to check against this
 \ sort of failure. We get a valid stream, and errno is not set. The kludge in PIPE_handler() does not help...
 \ Therefore, unless DIRECT_PIPE is defined, we will now circumvent this problem by dumping the
 \ output to a temp file, and then using `cat "tempfile" | whatevercommandtheusergave`, a command
 \ that will be spawned using system(). In the hope that this won't have similar desastrous effects
 \ on some as-yet-untested system...
 \ Of course, the good thing is that we can now seek on these 'new style pipes'!
 */
		signal( SIGPIPE, PIPE_handler );
		PIPE_fileptr= &out_stream;

		  /* "Load" sh to make popen respond faster	*/
		system( "sh < /dev/null > /dev/null 2>&1");

		fflush( stdin );
		fflush( stdout );
		fflush( stderr );

		errno= 0;
		out_stream = popen(buf, "wb");
		if( !out_stream || out_stream== NullDevice || PIPE_error ) {
#ifdef TOOLBOX	    
			sprintf(err, "Unable to issue command:\n  %s (%s)\n", buf, serror() );
			xtb_error_box( curWin->window, err, "Failure" );
#else
			fprintf(StdErr, "Unable to issue command:\n  %s (%s)\n", buf, serror() );
#endif
			if( backup && *tilde ){
				if( rename( backup, tilde ) ){
					fprintf( StdErr, "do_hardcopy(): Can't restore '%s' to '%s' (%s)\n", backup, tilde, serror() );
					xfree( backup );
				}
			}
			return( (errno)? errno : -1 );
		}
		isPIPE= True;
#else
		if( !(out_stream= fopen( (tempipeName= XGtempnam( getenv("TMPDIR"), "XGpipe")), "wb")) ){
#ifdef TOOLBOX	    
			sprintf(err, "Unable to open temporary pipe buffer file:\n  %s (%s)\n", tempipeName, serror() );
			xtb_error_box( curWin->window, err, "Failure" );
#else
			fprintf(StdErr, "Unable to open temporary pipe buffer file:\n  %s (%s)\n", tempipeName, serror() );
#endif
			if( backup && *tilde ){
				if( rename( backup, tilde ) ){
					fprintf( StdErr, "do_hardcopy(): Can't restore '%s' to '%s' (%s)\n", backup, tilde, serror() );
					xfree( backup );
				}
			}
			return( (errno)? errno : -1 );
		}
		  /* just be sure: this is *not* a pipe! */
		isPIPE= False;
#endif
		xfree(fn);
		fn= strdup( buf );
		  /* We are, in fact, *not* going to append! */
		append= False;
    } else {
	  struct stat st;
		tildeExpand(tilde, file_or_dev);
		if( stat( tilde, &st)== 0 ){
			exists= True;
			org_atime= st.st_atime;
			org_mtime= st.st_mtime;
			  /* 2002110x: regular files are moved to <filename>.bak before writing. This backup
			   \ is unlinked after successfull dumping of the new file, and restored otherwise.
			   */
			if( S_ISREG(st.st_mode) && (backup= (char*) calloc( strlen(tilde)+5, sizeof(char) )) ){
				strcpy( backup, tilde );
				strcat( backup, ".bak" );
			}
		}
		else if( errno!= ENOENT ){
			fprintf( StdErr, "do_hardcopy(): can't get stats on the outputfile (%s)\t(doing without)\n", serror() );
		}
		if( !strcasecmp( tilde, "stdout") ){
			out_stream= stdout;
		}
		else if( !strcasecmp( tilde, "stderr") ){
			out_stream= StdErr;
		}
		else if( strcasecmp( &tilde[strlen(tilde)-4], ".rar" )== 0 ){
			sprintf( buf, "rar a -df -ierr -m5 -ow -rr -ep -tl -tsc4 %s", tilde );
			fn= strdup( tilde );
			fn[ strlen(fn)-4 ]= '\0';
			if( !dev_spec ){
				dev_spec= "%s";
			}
			  /* naughty!! :)	*/
			goto HARDCOPY_POPEN;
		}
		else if( strcasecmp( &tilde[strlen(tilde)-4], ".bz2" )== 0 ){
			if( append ){
				sprintf( buf, "bzip2 -9 -v >> %s", tilde );
			}
			else{
				sprintf( buf, "bzip2 -9 -v > %s", tilde );
			}
			fn= strdup( tilde );
			fn[ strlen(fn)-4 ]= '\0';
			if( !dev_spec ){
			  /* 20030227: dev_spec should point to something in this case.... */
				dev_spec= "%s";
			}
			  /* naughty!! :)	*/
			goto HARDCOPY_POPEN;
		}
		else if( strcasecmp( &tilde[strlen(tilde)-3], ".gz" )== 0 ){
			if( append ){
				sprintf( buf, "gzip -9 -v >> %s", tilde );
			}
			else{
				sprintf( buf, "gzip -9 -v > %s", tilde );
			}
			fn= strdup( tilde );
			fn[ strlen(fn)-3 ]= '\0';
			  /* naughty!! :)	*/
			goto HARDCOPY_POPEN;
		}
		else{
		  int isfifo= 0;
			if( exists ){
				  /* Don't lock a fifo!	*/
				if( S_ISFIFO(st.st_mode) || strcmp(tilde, "/dev/null")== 0 ){
				  /* fopen(.., "a") will fail on a fifo.. We'll consider /dev/null a fifo, too.	*/
					isfifo= True;
					append= False;
				}
			}
			if( backup ){
				if( XGbackup( tilde, backup, append ) ){
					if( strcmp( tilde, "/dev/null") ){
						fprintf( StdErr, "do_hardcopy(): Can't backup '%s' to '%s' (%s)\n", tilde, backup, serror() );
					}
					xfree( backup );
				}
			}
			if( append ){
				out_stream = fopen(tilde, "a");
			}
			else{
/* 				out_stream = fopen(tilde, "w");	*/
				  /* No actual truncation is done before we
				   \ start writing on the file (i.e. after getting
				   \ a lock). Since fopen(tilde,"a") ensures that no
				   \ truncation is possible, we have to resort to
				   \ lower level routines.
				   \ 980316 - I have to admit I don't remember why I want this.. - to obtain a lock?
				   */
				if( (fd= open( tilde, O_WRONLY|O_CREAT, 0755))!= -1 ){
					  /* 20010107: should do more checks before attempting a chmod...	*/
					if( strcmp( tilde, "/dev/null") && fchmod( fd, S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH) ){
						sprintf(err, "Unable to chmod(755) file `%s'\n", tilde );
						xtb_error_box( curWin->window, err, "Failure" );
					}
					  /* fdopen(fd,"w") should not truncate.	*/
					out_stream= fdopen( fd, "wb");
				}
				else{
					out_stream= NULL;
				}
			}
			if( out_stream ){
				if( !isfifo ){
					if( HO_Dialog.win ){
						XFetchName( disp, HO_Dialog.win, &_Title );
						XStoreName( disp, HO_Dialog.win, "Requesting lock on outputfile");
						XG_XSync( disp, False );
					}
#if !defined(__CYGWIN__)
#ifdef __MACH__
/* 					if( flock( fileno(out_stream), LOCK_EX) )	*/
					errno= 0;
					flockfile(out_stream);
					if( errno )
#else
					if( lockf( fileno(out_stream), F_LOCK, 0) )
#endif
					{
						fprintf( StdErr, "do_hardcopy(): can't get a lock on the outputfile (%s)\t(doing without)\n", serror() );
					}
					else{
						locked= 1;
					}
#endif
					if( HO_Dialog.win ){
						if( _Title ){
							XStoreName( disp, HO_Dialog.win, _Title );
							XFree( _Title );
						}
						XG_XSync( disp, False );
					}
				}
				else if( debugFlag ){
					fprintf( StdErr, "do_hardcopy(): no locking on fifo pipes\n" );
				}
				fn= strdup(tilde);
			}
			if( !append && !isfifo && out_stream ){
				errno= 0;
				if( ftruncate( fileno(out_stream), 0) ){
					strcpy( err, "do_hardcopy(): can't truncate outputfile (" STRING(__LINE__) ")\n" );
					xtb_error_box( curWin->window, err, "Failure" );
				}
				rewind( out_stream );
			}
		}
		if (!out_stream) {
#ifdef TOOLBOX	    
			sprintf(err, "Unable to open file `%s'\n", tilde );
			xtb_error_box( curWin->window, err, "Failure");
#else
			fprintf(StdErr, "Unable to open file `%s'\n", tilde );
#endif
			if( backup ){
				if( rename( backup, tilde ) ){
					fprintf( StdErr, "do_hardcopy(): Can't restore '%s' to '%s' (%s)\n", backup, tilde, serror() );
					xfree( backup );
				}
			}
			return( (errno)? errno : -1);
		}
    }
	  /* We make a local copy of the window to-be-printed. This ensures we can change
	   \ most things without having to reset them. thisWin.dev_info.user_state should be
	   \ initialised by (*init_fun).
	   */
	memset( &thisWin, 0, sizeof(thisWin) );
    thisWin = *curWin;
	  /* the local window doesn't need double buffering!! */
	thisWin.XDBE_buffer= (XdbeBackBuffer) 0;
	thisWin.dev_info.user_state= NULL;
	thisWin.IntensityLegend.legend_needed= 0;
	thisWin.IntensityLegend.legend_width= 0;
	thisWin.silenced= 0;
	  /* 20001216:
	   \ These are bound to be in need of change, so set initialise them to allow the best callibration:
	   */
	thisWin.redrawn= 0;
	thisWin.XUnitsPerPixel= thisWin.YUnitsPerPixel= 0;
	  /* 20010719: there are things that we can't blindly copy and meddle with. The AxisValues stores
	   \ are among them (although I don't understand why... I thought these routines were clean!). Thus,
	   \ we need to initialise them (the copies!) to virgin state, and cleanup before returning from
	   \ this function.
	   */
	thisWin.axis_stuff.rawX.axis= thisWin.axis_stuff.X.axis= X_axis;
	thisWin.axis_stuff.rawY.axis= thisWin.axis_stuff.Y.axis= Y_axis;
	thisWin.axis_stuff.I.axis= I_axis;
	thisWin.axis_stuff.rawX.N= thisWin.axis_stuff.X.N= 0;
	thisWin.axis_stuff.rawY.N= thisWin.axis_stuff.Y.N= 0;
	thisWin.axis_stuff.I.N= 0;
	thisWin.axis_stuff.rawX.array= thisWin.axis_stuff.X.array= NULL;
	thisWin.axis_stuff.rawY.array= thisWin.axis_stuff.Y.array= NULL;
	thisWin.axis_stuff.I.array= NULL;
	thisWin.axis_stuff.rawX.labelled= thisWin.axis_stuff.X.labelled= NULL;
	thisWin.axis_stuff.rawY.labelled= thisWin.axis_stuff.Y.labelled= NULL;
	thisWin.axis_stuff.I.labelled= NULL;

	for( i= 0; i< setNumber; i++ ){
		  /* Update the last_processed_wi setting for those sets that were last drawn in the
		   \ to-be-printed window;
		   */
		if( AllSets[i].last_processed_wi== curWin ){
			AllSets[i].last_processed_wi= &thisWin;
		}
	}

	  /* page_scale_[xy]: only diagnostic existence-reasons!	*/
	page_scale_x= (double) (area_w= thisWin.dev_info.area_w);
	page_scale_y= (double) (area_h= thisWin.dev_info.area_h);
	    /* Important: aspect must be the on-screen aspect ratio, thus XOxxx should all equal those of curWin	*/
	aspect= ((double) curWin->XOppX - curWin->XOrgX)/((double) curWin->XOppY - curWin->XOrgY);
	if( NaNorInf(aspect) || aspect== 0 ){
		fprintf( StdErr, "do_hardcopy(): invalid screen aspectratio=%s; ignoring aspect-preserve request\n",
			d2str( aspect, NULL, NULL)
		);
		aspect= 0;
		prs_scrn_asp= False;
		if( *maxwidth< 0 ){
			sprintf( err, "             : maxWidth=%s < 0 not valid in this case... sorry!\n", d2str( *maxwidth, 0,0) );
			xtb_error_box( curWin->window, err, "Failure" );
			return(-1);
		}
	}

	  /* 20010722: I added initialisation for <fn> in case of output to a device/pipe. It is possible
	   \ that XGDump_AllWindows_Filename should NOT be initialised in this case.
	   */
	XGDump_AllWindows_Filename= fn;
	ascanf_XGOutput->fp= out_stream;
	ascanf_XGOutput->usage= XGstrdup(fn);
    if( dev_spec && out_stream && isPIPE && out_stream!= NullDevice ) {
		ascanf_XGOutput->fp_is_pipe= True;
	}
	else{
		ascanf_XGOutput->fp_is_pipe= False;
	}
	  /* 20010722: point ActiveWin to this local window: */
	ActiveWin= &thisWin;


	if( (*maxwidth< 0 || prs_scrn_asp) && aspect ){
		prs_scrn_asp= 1;
		  /* Some guess:	*/
		ratio = ((double) thisWin.dev_info.area_w) / ((double) thisWin.dev_info.area_h);
#ifdef OLD_LANDSCAPE
		if( area_w > area_h){
			final_w = RND(*maxheight * 10000.0);
			final_h = RND(*maxheight/ratio * 10000.0);
			orient= "landscape";
		} else {
			final_w = RND(*maxheight * ratio * 10000.0);
			final_h = RND(*maxheight * 10000.0);
			orient= "portrait";
		}
#else
		orient= (orientation)? "landscape" : "portrait";
		final_w = RND(*maxheight * ratio * 10000.0);
		final_h = RND(*maxheight * 10000.0);
#endif
Preserve_Screen_Aspect:;
		PrintingWindow= 1;
		  /* We now have an initial guess that gives the desired aspect-ratio
		   \ for the entire plot-area. We try now to obtain the ratio for the
		   \ area where the data is plotted (i.e. excluding titles, axes etc)
		   */
		(*init_fun)(out_stream, final_w, final_h, orientation, ti_fam, ti_size,
				le_fam, le_size, la_fam, la_size, ax_fam, ax_size, &thisWin, ierr, 0
		);
		  /* Seed the X,YUnitsPerPixel fields with sensible initial values based on the resolution proportion:	*/
		thisWin.XUnitsPerPixel= curWin->XUnitsPerPixel* ((double) curWin->dev_info.area_w/ (double) thisWin.dev_info.area_w);
		thisWin.YUnitsPerPixel= curWin->YUnitsPerPixel* ((double) curWin->dev_info.area_h/ (double) thisWin.dev_info.area_h);
		if( thisWin.dev_info.user_state ){
			thisWin.dev_info.xg_silent( thisWin.dev_info.user_state, False ),
			TransformCompute( &thisWin, True );
			thisWin.redrawn+= 1;
			{ int n= 0;
			  double xrange= (double) thisWin.XOppX- (double) thisWin.XOrgX,
					yrange= (double) thisWin.XOppY - (double) thisWin.XOrgY;
			  double delta_x= (aspect* yrange - xrange),
					delta_y= xrange/ aspect- yrange, scale;
				if( adapt_width ){
					while( xrange/ yrange!= aspect && n< 50 && delta_x ){
					  double aw= (double) thisWin.dev_info.area_w, incr;
						scale= final_w/ aw;
						incr= delta_x* scale;
#ifdef linux
						if( debugFlag ){
							fprintf( StdErr, "n=%d, aspect=%g, range=%g,%g:%g, delta=%g,%g, scale=%d/%g=%g final_w=>%d+%g\n",
								n, aspect, xrange, yrange, xrange/yrange, delta_x, delta_y,
								final_w, aw, scale, final_w, incr
							);
						}
#endif
						final_w= (int) (final_w+ incr);
						if( final_w> 0 ){
							(*init_fun)(out_stream, final_w, final_h, orientation, ti_fam, ti_size,
									le_fam, le_size, la_fam, la_size, ax_fam, ax_size, &thisWin, ierr, 0
							);
							TransformCompute( &thisWin, True );
							thisWin.redrawn+= 1;
							xrange= thisWin.XOppX- thisWin.XOrgX;
							yrange= thisWin.XOppY - thisWin.XOrgY;
							delta_x= (aspect* yrange - xrange);
						}
						else{
							delta_x= 0;
						}
						n+= 1;
					}
				}
				else{
					while( (double)xrange/ (double)yrange!= aspect && n< 50 && delta_y ){
						scale= (double)final_h / (double) thisWin.dev_info.area_h;
						final_h+= delta_y* scale;
						if( final_h> 0 ){
							(*init_fun)(out_stream, final_w, final_h, orientation, ti_fam, ti_size,
									le_fam, le_size, la_fam, la_size, ax_fam, ax_size, &thisWin, ierr, 0
							);
							TransformCompute( &thisWin, True );
							thisWin.redrawn+= 1;
							xrange= thisWin.XOppX- thisWin.XOrgX;
							yrange= thisWin.XOppY - thisWin.XOrgY;
							delta_y= (xrange- aspect* yrange)/ aspect;
						}
						else{
							delta_y= 0;
						}
						n+= 1;
					}
				}
				if( (adapt_width && delta_x== 0) || delta_y== 0 ){
					(*init_fun)(out_stream, final_w, final_h, orientation, ti_fam, ti_size,
							le_fam, le_size, la_fam, la_size, ax_fam, ax_size, &thisWin, ierr, 0
					);
					TransformCompute( &thisWin, True );
					thisWin.redrawn+= 1;
					xrange= thisWin.XOppX- thisWin.XOrgX;
					yrange= thisWin.XOppY - thisWin.XOrgY;
				}
				aspect= (double)xrange/ (double) yrange;
			}
			fprintf( stdout, "do_hardcopy(): plotsize (W*H) is %g x %g cm, aspect=%.15g\n",
				(*maxwidth= final_w/10000.0), *maxheight, aspect
			);
			fflush( stdout );
			sprintf( ps_comment, "Geometry (W*H) %g x %g cm, aspect=%.15g\n",
				*maxwidth, *maxheight, aspect
			);
		}
		else{
			fprintf( StdErr, "do_hardcopy(): auto-sizing not available for this format\n" );
			fflush( StdErr );
		}
	}
	else{
#ifdef OLD_LANDSCAPE
		  /* *maxheight and *maxwidth are the height and width of the
		   \ hardopy on (standing=portrait) paper. Therefore the plot
		   \ is rotated if height>width (although the PS printer may 
		   \ change it again based on the real dimensions of the figure.
		   */
		if( plot_area_y* *maxheight > plot_area_x* *maxwidth ){
			final_w = RND(*maxheight * 10000.0);
			final_h = RND(*maxwidth * 10000.0);
			orient= "landscape";
			ratio= -1.0;
		} else {
			final_w = RND(*maxheight * 10000.0);
			final_h = RND(*maxwidth * 10000.0);
			orient= "portrait";
			ratio= 1.0;
		}
#else
		orient= (orientation)? "landscape" : "portrait";
		final_h = RND(*maxheight * 10000.0);
		final_w = RND(*maxwidth * 10000.0);
#endif
	}
    ierr[0] = '\0';

	(*init_fun)(out_stream, final_w, final_h, orientation, ti_fam, ti_size,
			le_fam, le_size, la_fam, la_size, ax_fam, ax_size, &thisWin, ierr, 0
	);
	if( !thisWin.XUnitsPerPixel || !thisWin.redrawn ){
		thisWin.XUnitsPerPixel= curWin->XUnitsPerPixel* ((double) curWin->dev_info.area_w/ (double) thisWin.dev_info.area_w);
	}
	if( !thisWin.YUnitsPerPixel || !thisWin.redrawn ){
		thisWin.YUnitsPerPixel= curWin->YUnitsPerPixel* ((double) curWin->dev_info.area_h/ (double) thisWin.dev_info.area_h);
	}
	if( spax && spay ){
	  /* Get the output dimensions. We need those to rescale the plot area.	*/
		page_scale_x= (double) area_w / (double) thisWin.dev_info.area_w;
		page_scale_y= (double) area_h / (double) thisWin.dev_info.area_h;
		if( thisWin.dev_info.user_state ){
		  double sx= (double) final_w/ (double) thisWin.dev_info.area_w;
		  double sy= (double) final_h/ (double) thisWin.dev_info.area_h;
			TransformCompute( &thisWin, True );
			thisWin.redrawn+= 1;
			plot_area_x= (double) (thisWin.dev_info.area_w- 0* thisWin.dev_info.bdr_pad)-
				(double) (thisWin.XOppX - thisWin.XOrgX);
			plot_area_y= fabs((double) (thisWin.dev_info.area_h- 0* thisWin.dev_info.bdr_pad)-
				fabs(thisWin.XOppY- thisWin.XOrgY));
			final_w= RND( final_w + plot_area_x* sx );
			final_h= RND( final_h + plot_area_y* sy );
		}
	}
	else if( spax ){
	  /* Get the output dimensions. We need those to rescale the plot area.	*/
		page_scale_x= (double) area_w / (double) thisWin.dev_info.area_w;
		page_scale_y= (double) area_h / (double) thisWin.dev_info.area_h;
		if( thisWin.dev_info.user_state ){
		  double sx= (double) final_w/ (double) thisWin.dev_info.area_w;
			TransformCompute( &thisWin, True );
			thisWin.redrawn+= 1;
			plot_area_x= (double) (thisWin.dev_info.area_w- 0* thisWin.dev_info.bdr_pad)-
				(double) (thisWin.XOppX - thisWin.XOrgX);
			final_w= RND( final_w + plot_area_x* sx );
			if( prs_scrn_asp ){
				spax= 0;
				adapt_width= 0;
				goto Preserve_Screen_Aspect;
			}
		}
	}
	else if( spay ){
	  /* Get the output dimensions. We need those to rescale the plot area.	*/
		page_scale_x= (double) area_w / (double) thisWin.dev_info.area_w;
		page_scale_y= (double) area_h / (double) thisWin.dev_info.area_h;
		if( thisWin.dev_info.user_state ){
		  double sy= (double) final_h/ (double) thisWin.dev_info.area_h;
			TransformCompute( &thisWin, True );
			thisWin.redrawn+= 1;
			plot_area_y= fabs((double) (thisWin.dev_info.area_h- 0* thisWin.dev_info.bdr_pad)-
				fabs(thisWin.XOppY- thisWin.XOrgY));
			final_h= RND( final_h + plot_area_y* sy );
			if( prs_scrn_asp ){
				spay= 0;
				adapt_width= 1;
				goto Preserve_Screen_Aspect;
			}
		}
	}

	if( thisWin.legend_type== 1 && !thisWin.no_legend ){
	  int i;
		for( i= 0; i< 3; i++ ){
			TransformCompute( &thisWin, True );
			thisWin.redrawn+= 1;
		}
	}

	{ time_t timer= time(NULL);
		strncpy( PrintTime, ctime(&timer), 255 );
		if( XG_preserve_filetime && exists ){
			strncpy( mFileTime, ctime(&org_mtime), 255 );
			XG_preserved_filetime= mFileTime;
		}
		else{
			XG_preserved_filetime= NULL;
		}
	}
	PrintTime[255]= '\0';
	mFileTime[255]= '\0';
    if( out_stream && (*init_fun)(out_stream, final_w, final_h, orientation, ti_fam, ti_size,
		    le_fam, le_size, la_fam, la_size, ax_fam, ax_size, &thisWin, ierr, 1)
	){
	  long cap;
	  extern int ps_page_nr;
	  int ppn= ps_page_nr, no_atend= False;
		page_scale_x= (double) area_w / (double) thisWin.dev_info.area_w;
		page_scale_y= (double) area_h / (double) thisWin.dev_info.area_h;
		if( debugFlag){
			fprintf( StdErr,
"do_hardcopy(): maxwidth=%g maxheight=%g => ratio=%g final_w=%ge4 final_h=%ge4 %s orientation\n\tarea_w,h=%d,%d page_scale=%g,%g\n",
				*maxwidth, *maxheight, ratio,
				final_w/10000.0, final_h/10000.0, orient,
				thisWin.dev_info.area_w, thisWin.dev_info.area_h,
				page_scale_x, page_scale_y
			);
			fflush(StdErr);
		}

		thisWin.clipped= False;
		thisWin.silenced= 0;
		thisWin.dev_info.xg_silent( thisWin.dev_info.user_state, False );
		if( (thisWin.fit_after_draw && (thisWin.fit_xbounds || thisWin.fit_ybounds )) ){
			if( PS_STATE(&thisWin)->Printing== PS_PRINTING ){
			  FILE *nfp= fopen( "/dev/null", "wb");
				if( nfp ){
				  /* the fit_after_draw will search for the optimal fit when we give it that chance.
				   \ We give it that chance (redrawing until redraw==False). However, we don't want
				   \ that to show in the PostScript file (other formats might be added). In principle,
				   \ the psClear() function will rewind the file when a full "window" clear is performed,
				   \ thus avoiding unnecessary output and drawing elements. However, we have no way of
				   \ knowing at this place if the file we're printing to allows seeks and truncates - if
				   \ we're doing a preview, it probably won't. Or if we're printing directly to the
				   \ printer... So, to be sure, we temporarily send the output to the null device.
				   \ NB: setting silent mode doesn't help. For reasons of efficiency, TransformCompute()
				   \ doesn't do anything in that case. Maybe we should make it do its work... :)
				   */
					PS_STATE(&thisWin)->psFile= nfp;
					cap= PS_STATE(&thisWin)->clear_all_pos;
				}
			}
		}
		if( !append && PS_STATE(&thisWin)->Printing== PS_PRINTING && psEPS && psDSC ){
		  /* 20010802: when saving an EPS file, try to have an accurate boundingbox and pagecount
		   \ at the top of the file, instead of using the much simpler (atend) macros. Some programmes
		   \ really seem to want to have this information available at once...
		   */
			no_atend= True;
		}

		if( psEPS && psDSC && !no_atend ){
		  char msg[512];
			sprintf( msg,
				"Warning: DSC structuring not complete because rewinding to put BoundingBox at file start is not possible..."
			);
			fprintf( StdErr, "%s\n", msg );
			psMessage( PS_STATE(&thisWin), msg);
		}

		  /* Do as many redraws as necessary/requested	*/
		curWin->redraw= curWin->halt= 0;
		*ascanf_escape_value= thisWin.halt= ascanf_escape= 0;
		RedrawNow(&thisWin);
		  /* We must check curWin->halt too: _Handle_An_Event will in fact find that LocalWin
		   \ if ever an X11 event occurs that sets the halt field!
		   */
		while( (thisWin.redraw || curWin->redraw) && !(thisWin.halt || curWin->halt) && !ascanf_escape ){
			thisWin.dont_clear= False;
			DrawWindow( &thisWin );
		}
		if( no_atend || (thisWin.fit_after_draw && (thisWin.fit_xbounds || thisWin.fit_ybounds )) ){
		  /* Final redraw, now unsilenced...	*/
		  psUserInfo *ps= PS_STATE(&thisWin);
			thisWin.clipped= False;
			if( ps->Printing== PS_PRINTING && ps->psFile!= out_stream ){
			  /* Restore the real output stream:	*/
				if( ps->psFile ){
					fclose(ps->psFile);
				}
				ps->psFile= out_stream;
				ps->clear_all_pos= cap;
			}
			if( no_atend ){
			  /* 20010802: we now have the information that we want to put at the head of the file,
			   \ and not at the end. Thus, we'll have to do everything once more...
			   */
				  /* Properly close the file - this allows psEnd() (for example...) to determine
				   \ the dimension information we want so badly...
				   */
				if (thisWin.dev_info.xg_end) {
					thisWin.dev_info.xg_end(&thisWin);
					  /* Remember to null the user_state field! */
					thisWin.dev_info.user_state= NULL;
				}
				errno= 0;
				fflush(out_stream);
				if( ftell(out_stream) ){
				  /* If there's anything in the file, away it goes! */
					if( ftruncate( fileno(out_stream), 0) ){
						strcpy( err, "do_hardcopy(): can't truncate outputfile (" STRING(__LINE__) ")\n" );
						xtb_error_box( curWin->window, err, "Failure" );
					}
				}
				  /* Whatever we just did to data in the file, rewind the filepointer. If
				   \ we omit this here, any future output will go to the place where the
				   \ pointer points now - the end of the previous data. This would be a great
				   \ way to have fun with humongeous files that actually contain only a few Kb of
				   \ data. But that's not our current interest :)
				   */
				rewind( out_stream );
				  /* tell (init_fun) to use previous data: */
				ps_previous_dimensions= True;
				  /* And call it once more to initialise the new output */
				(*init_fun)(out_stream, final_w, final_h, orientation, ti_fam, ti_size,
						le_fam, le_size, la_fam, la_size, ax_fam, ax_size, &thisWin, ierr, 1);
				  /* Get a copy of the new userInfo data! */
				ps= PS_STATE(&thisWin);
			}
			  /* We did so many things that could have incremented the page counter: reset it
			   \ here to its value when we started.
			   \ NB: there should be device-specific callbacks/methods for this....
			   */
			ps_page_nr= ppn;
			  /* And redraw. If for whatever reason the fitting does not converge to
			   \ a stable optimum... too bad. User will have to change things, or
			   \ print to a diskfile.
			   */
			curWin->redraw= curWin->halt= 0;
			*ascanf_escape_value= thisWin.halt= ascanf_escape= 0;
			RedrawNow(&thisWin);
			if( thisWin.fit_after_draw && thisWin.redraw && !(thisWin.halt || curWin->halt) ){
				xtb_error_box( curWin->window, " Aie - fitting did not converge to a stable optimum!\n"
					" You will probably want to print to a PostScript (disk) file to assure\n"
					" acceptable printing.\n",
					"Warning"
				);
			}
			while( (thisWin.redraw || curWin->redraw) && !(thisWin.halt || curWin->halt) && !ascanf_escape ){
				thisWin.dont_clear= False;
				DrawWindow( &thisWin );
			}
		}

		if( !Use_HO_Previous_TC ){
			HO_Previous_psUI= *PS_STATE(&thisWin);
		}

		if (thisWin.dev_info.xg_end) {
			thisWin.dev_info.xg_end(&thisWin);
			thisWin.dev_info.user_state= NULL;
		}
		window_width/= 10000.0;
		window_height/= 10000.0;
		data_width/= 10000.0;
		data_height/= 10000.0;
		page_scale_x= 1.0;
		page_scale_y= 1.0;
    } else if( strlen(ierr) ) {
#ifdef TOOLBOX
		xtb_error_box( curWin->window, ierr, "Hardcopy initialisation output" );
#else
		(void) fprintf(StdErr, "%s\n", ierr);
#endif
    }
#ifdef DIRECT_PIPING
    if( dev_spec && out_stream && isPIPE && out_stream!= NullDevice ){
		(void) pclose(out_stream);
		PIPE_fileptr= NULL;
		isPIPE= False;
    }
	else
#endif
	if( out_stream && out_stream!= stdout && out_stream!= StdErr && out_stream!= NullDevice ){
		if( locked ){
#if !defined(__CYGWIN__)
#ifdef __MACH__
/* 			flock( fileno(out_stream), LOCK_UN);		*/
			funlockfile(out_stream);
#else
			lockf( fileno(out_stream), F_ULOCK, 0);
#endif
#endif
			locked= 0;
		}
		fclose(out_stream);
#ifndef DIRECT_PIPING
		if( tempipeName ){
		  char *command= NULL;
		  int R= -1;
			  /* 20041114: another advantage of using a temp.file is that the original file times can be preserved
			   \ at that level also. In other words, the file inside the archive will have the same time information
			   \ as the original, and as the archive itself.
			   */
			if( XG_preserved_filetime ){
				if( set_file_times( tempipeName, org_atime, org_mtime ) ){
					fprintf( StdErr, "Problem preserving original filetime '%s' on %s: %s\n",
						XG_preserved_filetime, tempipeName, serror()
					);
				}
			}
			if( strncmp( fn, "rar ", 4) ){
				command= concat( "cat \"", tempipeName, "\" | ", fn, NULL );
			}
			else{
			  char *nname= strdup(tilde);
				if( strlen(nname)> 4 ){
				  char *tmp= rindex(tempipeName, '/');
					if( *tmp== '/' ){
					  /* the temp filename without the path: */
						tmp++;
					}
					nname[ strlen(nname)-4 ]= '\0';
					if( strcmp( &nname[ strlen(nname)-3 ], ".xg" ) ){
						command= concat( fn, " \"", tempipeName, "\" ; rar rn -inul ", tilde, " ", tmp, " ", nname, ".xg", NULL );
					}
					else{
						command= concat( fn, " \"", tempipeName, "\" ; rar rn -inul ", tilde, " ", tmp, " ", nname, NULL );
					}
					xfree(nname);
				}
			}
			if( debugFlag ){
				fprintf( StdErr, "Temporary pipe buffer finished, now spawning \"%s\"\n", command );
			}
			else{
				fprintf( StdErr, "%s\n", command );
			}
			fflush( StdErr );
			if( command ){
				R= system( command );
			}
			if( R== -1 || (R== 127 && errno) ){
#ifdef TOOLBOX
				xtb_error_box( curWin->window, command, "An error (possibly) occurred executing the command:" );
#else
				fprintf(StdErr, "(Possible) error executing `%s`: %s\n", command, serror() );
#endif
			}
			xfree( command );
			unlink( tempipeName );
			xfree( tempipeName );
		}
#endif
    }
	PrintingWindow= 0;

	if( backup ){
		unlink( backup );
		xfree( backup );
	}
	if( XG_preserved_filetime ){
		if( set_file_times( tilde, org_atime, org_mtime ) ){
			fprintf( StdErr, "Problem preserving original filetime '%s' on %s: %s\n",
				XG_preserved_filetime, tilde,
				serror()
			);
		}
/* 		XG_preserved_filetime= NULL;	*/
	}

	if( use_gsTextWidth ){
		curWin->axis_stuff.__XLabelLength= 0;
		curWin->axis_stuff.__YLabelLength= 0;
		curWin->axis_stuff.XLabelLength= 0;
		curWin->axis_stuff.YLabelLength= 0;
		if( curWin->textrel.gs_batch ){
			curWin->textrel.gs_batch_items= thisWin.textrel.gs_batch_items;
			strcpy( curWin->textrel.gs_fn, thisWin.textrel.gs_fn );
		}
	}
	GCA();

	for( i= 0; i< setNumber; i++ ){
		  /* Restore the last_processed_wi setting for those sets that were last drawn in the
		   \ printed window;
		   */
		if( AllSets[i].last_processed_wi== &thisWin ){
			AllSets[i].last_processed_wi= curWin;
		}
	}

	xfree( thisWin.axis_stuff.rawX.array );
	xfree( thisWin.axis_stuff.X.array );
	xfree( thisWin.axis_stuff.rawY.array );
	xfree( thisWin.axis_stuff.Y.array );
	xfree( thisWin.axis_stuff.I.array );
	xfree( thisWin.axis_stuff.rawX.labelled );
	xfree( thisWin.axis_stuff.X.labelled );
	xfree( thisWin.axis_stuff.rawY.labelled );
	xfree( thisWin.axis_stuff.Y.labelled );
	xfree( thisWin.axis_stuff.I.labelled );

	if( !Use_HO_Previous_TC ){
	  extern LocalWin HO_PreviousWin;
		  /* Now, after most dynamic info in thisWin has been freed, we (safely?) can copy the remaining static
		   \ information into the global HO_PreviousWin_ptr variable.
		   */
		HO_PreviousWin_ptr= &HO_PreviousWin;
		HO_PreviousWin= thisWin;
		  /* install a static, stub copy of the user_state, so that PS_STATE() will be able to show 
		   \ the last used Printing setting.
		   */
		HO_PreviousWin_ptr->dev_info.user_state= (void*) &HO_Previous_psUI;
	}

	xfree(fn);
	XGDump_AllWindows_Filename= NULL;

	Update_LMAXBUFSIZE(True, NULL);

	ascanf_XGOutput->fp= afp;
	ascanf_XGOutput->fp_is_pipe= afpp;
	xfree( ascanf_XGOutput->usage );
	ascanf_XGOutput->usage= affn;

	curWin->redraw= rdr;
	ActiveWin= AW;

	return( hardcopy_init_error );
}
#endif

char *tildeExpand(char *out, const char *in)
/*
 * This routine expands out a file name passed in `in' and places
 * the expanded version in `out'.  It returns `out'.
 \ 20010428: if out==NULL, allocate space.
 */
{
    char username[50], *userPntr= NULL;
    struct passwd *userRecord= NULL;

    /* Skip over the white space in the initial path */
    while ((*in == ' ') || (*in == '\t')) in++;

    /* Tilde? */
    if (in[0] == TILDE) {
		/* Copy user name into 'username' */
		in++;  userPntr = &(username[0]);
		while ((*in != '\0') && (*in != '/')) {
			*(userPntr++) = *(in++);
		}
		*(userPntr) = '\0';
		/* See if we have to fill in the user name ourselves */
		if (strlen(username) == 0) {
			userRecord = getpwuid(getuid());
		} else {
			userRecord = getpwnam(username);
		}
		if (userRecord) {
			/* Found user in passwd file.  Concatenate user directory */
			userPntr= userRecord->pw_dir;
		}
		else{
			userPntr= NULL;
		}
    }

    /* Concantenate remaining portion of file name */
	if( out ){
		sprintf( out, "%s%s", (userPntr)? userPntr : "", in );
/* 		strcat(out, in);	*/
	}
	else{
		if( userPntr ){
			out= concat( userPntr, in, NULL );
		}
		else{
			out= strdup( in );
		}
	}
    return out;
}

extern int XErrorHandled;

#ifndef stricmp
int stricmp( const char *a,  const char *b)
#ifndef NO_STRCASECMP
{
	return( strcasecmp( a, b) );
}
#else
/*
 * This routine compares two strings disregarding case.
 */
{ int value;

    if ((a == (char *) 0) || (b == (char *) 0)) {
		return a - b;
    }

    for( /* nothing */; ((*a | *b) &&
	  !(value = ((isupper(*a) ? *a - 'A' + 'a' : *a) -
		     (isupper(*b) ? *b - 'A' + 'a' : *b))));
		a++, b++
	){
      /* Empty Body */;
	}

    return value;
}
#endif
#endif

char *parse_seconds(double seconds, char *buf)
{	double secs, hours;
	double hrs, mins, days;
	static char _timestring[80];
	char *s, *timestring= (buf)? buf : _timestring;

	if( NaN(seconds) ){
		strcpy( timestring, "NaN:NaN:NaN:NaN");
		return( timestring);
	}
	if( INF(seconds) ){
		strcpy( timestring, "Inf:Inf:Inf:Inf");
		return( timestring);
	}
	if( seconds< 0){
		seconds*= -1.0;
		s= "[-]";
	}
	else
		s= "";
	hours= seconds / 3600.0;
	hrs= ssfloor(hours);
	days= ssfloor(hrs / 24.0);
	mins= ssfloor( ((hours- hrs) * 60.0) );
	secs= seconds- hrs* 3600.0 - mins* 60.0;
	hrs= fmod( hrs, 24.0);
	if( fabs(hrs) >= 24 || fabs(mins)>= 60.0 || fabs(secs)>= 60.0)
		sprintf( timestring, "??:??:??:??");
	else{
		if( days== 0)
			sprintf( timestring, "%s%02d:%02d:%02.02lf", s, (int) hrs, (int) mins, secs);
		else if( days== 1)
			sprintf( timestring, "%s%g day,%02d:%02d:%02.02lf", s, days, (int) hrs, (int) mins, secs);
		else
			sprintf( timestring, "%s%g days,%02d:%02d:%02.02lf", s, days, (int) hrs, (int) mins, secs);
	}
	return( timestring);
}

char *time_stamp( FILE *fp, char *name, char *buf, int verbose, char *postfix)
{  struct stat Stat;
   struct tm *tm;
	if( !buf){
		return( NULL);
	}
	else{
		buf[0]= '\0';
	}
	  // 20080922, 20090113:
	{ char *pyfn;
		if( (pyfn= PyOpcode_Check(name)) ){
			name= pyfn;
		}
	}
	if( stat( name, &Stat) && (fp==NULL || fstat(fileno(fp), &Stat)) ){
		if( verbose>= 0 ){
			sprintf( buf, "%s: %s", name, serror() );
		}
		else{
			sprintf( buf, "%s", name );
		}
	}
	else{
		tm= localtime( &(Stat.st_mtime) );
		if( tm){
			if( verbose){
			  char *c;
				sprintf( buf, "%s: %s", name, asctime(tm) );
				if( (c= rindex( buf, '\n')) ){
					*c= '\0';
				}
			}
			else{
#ifdef __hpux
				sprintf( buf, "%02d%02d%02d%02d%02d.%02d",
					tm->tm_year, tm->tm_mon+1, tm->tm_mday,
					tm->tm_hour, tm->tm_min,
					tm->tm_sec
				);
#else
				sprintf( buf, "%02d%02d%02d%02d%02d",
					tm->tm_mon+1, tm->tm_mday,
					tm->tm_hour, tm->tm_min,
					tm->tm_year
				);
#endif
			}
		}
		else{
			sprintf( buf, "%s: %s", name, serror() );
		}
	}
	if( postfix ){
		strcat( buf, postfix );
	}
	return(buf);
}

void process_hist(Window win)
{

	if( process_history && *process_history ){
	  char *sel= NULL;
	  int id;
		id= xtb_popup_menu( win, process_history, "Process History - selected is copied to clipboard", &sel, &process_pmenu );
		if( sel ){
			if( debugFlag ){
				xtb_error_box( win, sel, "Copied to clipboard:" );
			}
			else{
				Boing(10);
			}
			XStoreBuffer( disp, sel, strlen(sel), 0);
			  // RJVB 20081217
			xfree(sel);
		}
		xtb_popup_delete( &process_pmenu );
	}
	else{
		xtb_error_box( win, "None", "Process History" );
	}
}

xtb_hret process_hist_h(Window win, int bval, xtb_data info)
{
	process_hist( win );
	xtb_bt_set( win, 0, NULL);
	return( XTB_HANDLED );
}

char *help_fnc_selected= NULL;
int help_fnc(LocalWin *wi, Boolean refresh, Boolean showMiss )
{  FILE *cfp, *rcfp;
   char buf[1024], *c, *tnam= NULL;
   int ret;
	if( (cfp= fopen( (tnam= XGtempnam( getenv("TMPDIR"), "hlpfn")), "wb")) ){
		if( debugFlag ){
			fprintf( StdErr, "help_fnc(): opened temp file \"%s\" as buffer\n",
				tnam
			);
		}
		rcfp= fopen( tnam, "r");
	}
	else if( debugFlag ){
		fprintf( StdErr, "help_fnc(): can't open temp file \"%s\" as buffer (%s)\n",
			tnam, serror()
		);
		return(0);
	}
	ret= show_ascanf_functions( cfp, "   ", 0, 1 );
	  /* Here routines that output help info to file can be put	*/
	fflush( cfp);

	if( rcfp ){
	  char *sel= NULL;
	  int nc= NoComment, id = 0;
	  Sinc List;
		if( ret || showMiss ){
			List.sinc.string= NULL;
			Sinc_string_behaviour( &List, NULL, 0,0, SString_Dynamic );
			Sflush( &List );
			NoComment= False;
			while( (c= fgets(buf, 1023, rcfp)) && !feof(rcfp) ){
				Add_SincList( &List, c, False );
			}
			if( refresh ){
				xtb_popup_delete( &vars_pmenu );
			}
			xfree(help_fnc_selected);
			id= xtb_popup_menu( wi->window, List.sinc.string,
				"Help Dialog - Name of selected function is copied to clipboard", &sel, &vars_pmenu );
		}
		fclose( rcfp);
		unlink( tnam );
		rcfp= NULL;

		if( id> 0 && sel ){
		  int i;
		  char name[128];
			if( sel[0]== 0x01 ){
				sel++;
			}
			if( sscanf( sel, " %d[%s]: ", &i, name )== 2 ){
			  char *c= index( name, ']');
				if( c && c[1]== ':' ){
					*c= '\0';
				}
				if( debugFlag ){
					sprintf( List.sinc.string, "#%d(%s): \"%s\"\n", i, name, sel );
					xtb_error_box( wi->window, List.sinc.string, "Copied to clipboard:" );
				}
				else{
					Boing(10);
				}
				XStoreBuffer( disp, name, strlen(name), 0);
				help_fnc_selected = strdup(name);
			}
			xfree( List.sinc.string);
			xfree(sel);
		}

		NoComment= nc;
	}
	else if( debugFlag ){
		fprintf( StdErr, "help_fnc(): couldn't open temp file \"%s\" for reading (%s)\n",
			tnam, serror()
		);
	}
	xfree(tnam);
	fclose( cfp);
	return( ret );
}

FilterUndo BoxFilterUndo;

int BoxFilter_Undo( LocalWin *wi )
{ int r= 0, i, j;
  int udat[3], ok1= 1, ok2= 1, ok3=1;
  static int call= 0;
  static FILE *rfp= NULL;
  Boolean redo_enable= True;
  char *tnam;
	if( BoxFilterUndo.fp ){
		if( MaxCols> BinaryDump.columns ){
			if( !AllocBinaryFields( MaxCols, "FilterPoints_Box()" ) ){
				xtb_error_box( wi->window, "Failure getting columns read-buffer - undo lost!\n", "#$!@#" );
				fclose( BoxFilterUndo.fp );
				BoxFilterUndo.fp= NULL;
				return(0);
			}
		}
		if( rfp ){
			fclose( rfp );
		}
		errno= 0;
		if( redo_enable ){
		  char cbuf[6];
			sprintf( cbuf, "XGf%02d", call );
			call= (call + 1) % 100;
			tnam= XGtempnam( getenv("TMPDIR"), cbuf );
			if( (rfp= fopen( tnam, "w+")) ){
				rewind( rfp );
				ftruncate( fileno(rfp), 0 );
				unlink( tnam );
			}
			else{
				redo_enable= False;
			}
		}
		if( !redo_enable ){
			xtb_error_box( wi->window, "No redo available at this moment\n", "Warning" );
		}
		errno= 0;
		fseek( BoxFilterUndo.fp, 0, SEEK_SET );
		while( !(feof(BoxFilterUndo.fp) || ferror(BoxFilterUndo.fp)) && ok1 ){
			ok1= (fread( udat, sizeof(int), 3, BoxFilterUndo.fp)== 3 &&
				udat[1]>= 0 && udat[1]< setNumber && udat[2]>= 0 && udat[2]< AllSets[udat[1]].numPoints
			);
			ok2= (fread( &BinaryDump.columns, sizeof(short), 1, BoxFilterUndo.fp )== 1 &&
				BinaryDump.columns>= 0
			);
			ok3= (fread( BinaryDump.data, sizeof(double), BinaryDump.columns, BoxFilterUndo.fp)== BinaryDump.columns);
			if( ok1 && ok2 && ok3 ){
			  DataSet *this_set= &AllSets[udat[1]];
				i= udat[2];
				  /* just to be sure..:	*/
				BinaryDump.columns= MIN( BinaryDump.columns, this_set->ncols );
				if( redo_enable ){
				  int rdat[3];
				  /* Use the terminator to store the redo-values!	*/
					BinaryTerminator.columns= this_set->ncols;
					memcpy( rdat, udat, sizeof(rdat) );
					rdat[0]= r;
#ifdef DEBUG
					fprintf( StdErr, "BoxFilter_Undo(%d,%s): saving #%d set=%d pnt=%d, cols=%d",
						r, tnam, udat[0], udat[1], udat[2], this_set->ncols );
#endif
					for( j= 0; j< this_set->ncols; j++ ){
#ifdef DEBUG
					fprintf( StdErr, ",%g", this_set->columns[j][i] );
#endif
						BinaryTerminator.data[j]= this_set->columns[j][i];
					}
					fwrite( rdat, sizeof(int), 3, rfp );
					fwrite( &BinaryTerminator.columns, sizeof(short), 1, rfp );
					fwrite( BinaryTerminator.data, sizeof(double), BinaryTerminator.columns, rfp );
				}
#ifdef DEBUG
				fprintf( StdErr, "\nBoxFilter_Undo(%d): restoring #%d set=%d pnt=%d, cols=%d",
					r, udat[0], udat[1], udat[2], BinaryDump.columns );
#endif
				for( j= 0; j< BinaryDump.columns; j++ ){
					this_set->columns[j][i]= BinaryDump.data[j];
#ifdef DEBUG
					fprintf( StdErr, ",%g", this_set->columns[j][i] );
#endif
				}
#ifdef DEBUG
		fputc( '\n', StdErr );
#endif
				if( j ){
					r+= 1;
				}
				RedrawSet( udat[1], 0 );
			}
			else if( !(udat[1]==-1 && udat[2]== -1 && BinaryDump.columns== 0) ){
			  char emsg[256];
				sprintf( emsg, "BoxFilter undo file garbled after %d valid entries\nSkipping remainder!", r );
				xtb_error_box( wi->window, emsg, "Warning" );
				ok1= 0;
			}
		}
		fclose( BoxFilterUndo.fp );
		if( redo_enable ){
		  int j;
			BinaryTerminator.columns= 0;
			for( j= 0; j< MaxCols; j++ ){
				BinaryTerminator.data[j]= 0;
			}
			udat[0]= udat[1]= udat[2]= -1;
			fwrite( udat, sizeof(int), 3, rfp );
			fwrite( &BinaryTerminator.columns, sizeof(short), 1, rfp );
			fwrite( BinaryTerminator.data, sizeof(double), BinaryTerminator.columns, rfp );
			BoxFilterUndo.fp= rfp;
			rfp= NULL;
		}
		else{
			BoxFilterUndo.fp= NULL;
		}
		if( r ){
			RedrawSet( -1, True );
		}
	}
	return(r);
}

char *BoxFilter_File= NULL;
int BoxFilter_Undo_Enable= True;

int FilterPoints_Box( LocalWin *wi, char *fname,
	double _loX, double _loY, double _hiX, double _hiY, DataSet *this_set, int this_point )
{ int i, idx, SN= setNumber, dN= 0, n;
  LocalWin *aw= ActiveWin;
  DEFUN( ascanf_Variable,  ( ASCB_ARGLIST ), int );
  static char afname[]= "$FilterBox";
  static char afdescr[]= "$FilterBox[4]= {lowX,lowY,highX,highY}";
  static ascanf_Function fbox= { NULL, ascanf_Variable, 4, _ascanf_array,
		NULL, 0, 0, 0, 0, 0, 0.0
	};
  static char called= 0;
  static double *box= NULL;
  char *tnam;
  Boolean undo_enable= 1;
  int udat[3];
  char rmsg[512];
  int filters= 0, first_filter= 1;
  BoxFilters filter= 0, _filter;
  GenericProcess *Filter, *_Filter;
  char ffn[MAXPATHLEN];
  Time_Struct BFtimer;

	Elapsed_Since( &BFtimer, True );

	ActiveWin= wi;
	if( !called ){
	  ascanf_Function *af= vars_ascanf_Functions[0].cdr;
		fbox.name= afname;
		fbox.hash= ascanf_hash(fbox.name, NULL);
		fbox.type= _ascanf_array;
		fbox.N= 4;
		fbox.usage= afdescr;
		if( !box ){
			fbox.array= box= (double*) calloc( 4, sizeof(double) );
		}
		fbox.sign= 1;
		if( !af ){
			vars_ascanf_Functions[0].cdr= &fbox;
			called= 1;
		}
		else{
			while( af && af->cdr ){
				af= af->cdr;
			}
			if( af && !af->cdr ){
				af->cdr= &fbox;
				called= 1;
			}
		}
	}
	if( box ){
		box[0]= _loX, box[1]= _loY, box[2]= _hiX, box[3]= _hiY;
	}
	fbox.assigns+= 1;
	clean_param_scratch();
	if( MaxCols> BinaryDump.columns ){
		if( !AllocBinaryFields( MaxCols, "FilterPoints_Box()" ) ){
			undo_enable= False;
		}
	}
	if( BoxFilterUndo.fp ){
		fclose( BoxFilterUndo.fp );
	}
	if( BoxFilter_Undo_Enable ){
		if( undo_enable ){
			tnam= XGtempnam( getenv("TMPDIR"), "XGftU");
			if( (BoxFilterUndo.fp= fopen( tnam, "w+")) ){
				rewind( BoxFilterUndo.fp );
				ftruncate( fileno(BoxFilterUndo.fp), 0 );
				unlink( tnam );
			}
			else{
				undo_enable= False;
			}
		}
		if( !undo_enable ){
			xtb_error_box( wi->window, "No undo available at this moment\n", "Warning" );
		}
	}
	else{
		undo_enable= False;
	}

	  /* 20040605: moved here */
	if( this_set ){
	  /* Apply only to this set's this point 	*/
		idx= this_set->set_nr;
		SN= idx+1;
		if( this_point>= 0 && this_point< this_set->numPoints ){
				  /* Set the bounding "rectangle" to this point's co-ordinates.	*/
			_loX= _hiX= this_set->xvec[this_point];
			_loY= _hiY= this_set->yvec[this_point];
		}
		else{
			this_point= -1;
			_loX= _hiX= this_set->xvec[0];
			_loY= _hiY= this_set->yvec[0];
			for( i= 1; i< this_set->numPoints; i++ ){
				_loX= MIN(_loX, this_set->xvec[i]);
				_hiX= MAX(_hiX, this_set->xvec[i]);
				_loY= MIN(_loY, this_set->yvec[i]);
				_hiY= MAX(_hiY, this_set->yvec[i]);
			}
		}
		sprintf( rmsg,
			" Starting BoxFilter on set #%d (%s); (%s,%s) - (%s,%s)\n",
			idx,
			((this_point<0)? "all points" : d2str((double)this_point, "point %g", NULL)),
			d2str( _loX, NULL, NULL), d2str( _loY, NULL, NULL),
			d2str( _hiX, NULL, NULL), d2str( _hiY, NULL, NULL)
		);
	}
	else{
		idx= 0;
		this_point= -1;
		if( _loX > _hiX ){
		  double temp= _hiX;
			_hiX = _loX;
			_loX = temp;
		}
		if( _loY > _hiY ){
		  double temp= _hiY;
			_hiY = _loY;
			_loY = temp;
		}
		sprintf( rmsg,
			" Starting BoxFilter on all visible sets' points within (%s,%s) - (%s,%s)\n",
			d2str( _loX, NULL, NULL), d2str( _loY, NULL, NULL),
			d2str( _hiX, NULL, NULL), d2str( _hiY, NULL, NULL)
		);
	}

	sprintf( rmsg,
		"%s"
		" Enter a filename if necessary to define a *BOX_FILTER* and apply the required configuration(s)\n"
		" (empty name to use current; Cancel to abort operation\n",
		rmsg
	);
	StringCheck( rmsg, sizeof(rmsg)/sizeof(char), __FILE__, __LINE__ );

	  /* 20031018: it is conceivable that user may wish to fine-tune $FilterBox: make it writable
	   \ while including the filter file.
	   */
	fbox.dollar_variable= 1;
	if( !BoxFilter_File ){
		if( !(BoxFilter_File= calloc( MAXPATHLEN, sizeof(char) )) ){
			fprintf( StdErr, "BoxFilter: can't allocate filename, using temporary buffer instead! (%s)\n",
				serror()
			);
			BoxFilter_File= ffn;
		}
	}
	if(
		(fname && !Include_Files( wi, fname, "[BoxFilter]" )) ||
		(!fname && !interactive_IncludeFile( wi, rmsg, BoxFilter_File ))
	){
		xtb_error_box( wi->window, "NULL operation, or cancelled.\n", "BoxFilter abort" );
		ActiveWin= aw;
		fbox.dollar_variable= 0;
		return(dN);
	}

	if( !wi->raw_display ){
	  short done= False;
	  int si= wi->silenced, dbo= DetermineBoundsOnly;
		  /* 20050509: if we're not in raw mode, some other window might have shown our visible sets
		   \ after our own last redraw -- in which case xvec/yvec are going to be outdated. That
		   \ can be annoying, or even provoke errors. Generate a redraw if such a situation is detected!
		   */
		for( i= 0; i< setNumber && !done; i++ ){
			this_set= &AllSets[i];
			if( draw_set( wi, i) ){
				if( this_set->last_processed_wi!= wi ){
					wi->silenced= True;
					DetermineBoundsOnly= True;
					xtb_bt_set( wi->ssht_frame.win, wi->silenced, (char *) 0);
					wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr, "FilterPoints_Box(): set #%d had been last drawn in another window (%p): redrawing.\n",
							i, this_set->last_processed_wi
						);
					}
					RedrawNow( wi );
					wi->silenced= si;
					DetermineBoundsOnly= dbo;
					xtb_bt_set( wi->ssht_frame.win, wi->silenced, (char *) 0);
					wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
					done= True;
				}
			}
		}
		if( done ){
			wi->redraw= 1;
		}
	}

	ActiveWin= wi;
	Filter= NULL; filter= 0;
	  /* Find the first defined filter: */
	do{
		if( BoxFilter[filter].process_len ){
			Filter= &BoxFilter[filter];
			TBARprogress_header= BoxFilter[filter].command;
		}
		else{
			filter+= 1;
		}
	}
	while( filter< BOX_FILTER_CLEANUP && !Filter );
	if( !Filter ){
		xtb_error_box( wi->window, "No box filter(s) (*BOX_FILTER*) defined!\n", "BoxFilter abort" );
		ActiveWin= aw;
		fbox.dollar_variable= 0;
		return(dN);
	}
	  /* 20031114: the do/while loop over the filters used to be inside the loop over the sets. The result was
	   \ of course that first all INIT routines were executed, than all sets' BOX_FILTER commands, and then
	   \ all sets' FINISH commands. This is counterintuitive and (esp.) different from how the corresponding 
	   \ DATA_PROCESS commands work. Thus, the loops were exchanged, necessitating _filter and _Filter which
	   \ cache the info about the first filter to be applied.
	   */
	_filter= filter;
	_Filter= Filter;
	  /* 20040605: setting of idx, SN and this_point was done here. */
	for( ; idx< SN; idx++ ){
		  /* Do the first applicable filter for this set. */
		first_filter= True;
		filter= _filter;
		Filter= _Filter;
		do{
			if( draw_set(wi, idx) ){
			  double asn= *ascanf_setNumber, anp= *ascanf_numPoints,
				  x, y, data[ASCANF_DATA_COLUMNS];
			  int NP, column[ASCANF_DATA_COLUMNS]= {0,1,2,3}, sN= 0, last_point= -1, nP;
				this_set= &AllSets[idx];
				*ascanf_setNumber= idx;
				*ascanf_numPoints= this_set->numPoints;
				if( this_point>= 0 ){
					i= this_point;
					NP= i+ 1;
				}
				else{
					i= 0;
					NP= this_set->numPoints;
				}
				if( debugFlag || scriptVerbose ){
					fprintf( StdErr, "FilterPoints_Box(): set #%d[%d..%d> <- %s\n",
						idx, i, NP, BoxFilter[filter].command
					);
				}
				for( ; i< NP; i++ ){
					if( wi->raw_display || this_set->raw_display ){
					  /* 20050509: in raw mode, we can just as well use the 'original' data, not
					   \ the values which might depend on another window having been drawn after us...
					   */
						x= this_set->xval[i];
						y= this_set->yval[i];
					}
					else{
						x= this_set->xvec[i];
						y= this_set->yvec[i];
					}
					if( ((i== this_point) || (x>= _loX && y>= _loY && x<= _hiX && y<= _hiY)) && !DiscardedPoint( wi, this_set, i) ){
						udat[0]= dN;
						udat[1]= idx;
						udat[2]= i;
						  /* If we requested/enable undo possibilties, store the original values before applying
						   \ filter #1.
						   */
						if( first_filter && undo_enable> 0 && BoxFilter_Undo_Enable ){
						  int j;
							if( this_set->ncols> MaxCols ){
								MaxCols= this_set->ncols;
								AllocBinaryFields( MaxCols, "FilterPoints_Box()" );
							}
							BinaryDump.columns= this_set->ncols;
#ifdef DEBUG
			fprintf( StdErr, "FilterPoints_Box(%d,%s): saving #%d set=%d pnt=%d, cols=%d",
				dN, tnam, udat[0], udat[1], udat[2], this_set->ncols );
#endif
							for( j= 0; j< this_set->ncols; j++ ){
#ifdef DEBUG
			fprintf( StdErr, ",%g", this_set->columns[j][i] );
#endif
								BinaryDump.data[j]= this_set->columns[j][i];
							}
							fwrite( udat, sizeof(int), 3, BoxFilterUndo.fp );
							fwrite( &BinaryDump.columns, sizeof(short), 1, BoxFilterUndo.fp );
							fwrite( BinaryDump.data, sizeof(double), BinaryDump.columns, BoxFilterUndo.fp );
#ifdef DEBUG
			fputc( '\n', StdErr );
#endif
						}
						if( filter< BOX_FILTER_FINISH && 
							(filter> BOX_FILTER_INIT || last_point< 0)
						){
							*ascanf_self_value= (double) i;
							*ascanf_current_value= (double) i;
							*ascanf_counter= (*ascanf_Counter)= i;
							reset_ascanf_currentself_value= 0;
							reset_ascanf_index_value= True;
							data[0]= XVAL(this_set, i);
							data[1]= YVAL(this_set, i);
							data[2]= ERROR(this_set, i);
							if( this_set->lcol>= 0 ){
								data[3]= VVAL(this_set, i );
							}
							if( ascanf_verbose ){
								fprintf( StdErr, "FilterPoints_Box(%s): SET #%d, pnt #%dx y e: %s",
									Filter->command, this_set->set_nr, i, Filter->process
								);
								fflush( StdErr );
							}
							ascanf_arg_error= 0;
							n= param_scratch_len;
							compiled_fascanf( &n, Filter->process, param_scratch, NULL, data, column,
								&Filter->C_process
							);
							dN+= 1;
							sN+= 1;
						}
						  /* 20040110: unsetting first_filter here as it was would disable the undo feature for all
						   \ but the first point of each affected set. Delayed to just before proceeding to the next
						   \ filter.
						first_filter= 0;
						   */
						last_point= i;
						nP+= 1;
					}
				}
				if( filter== BOX_FILTER_FINISH && last_point>= 0 ){
					*ascanf_self_value= (double) last_point;
					*ascanf_current_value= (double) last_point;
					*ascanf_counter= (*ascanf_Counter)= last_point;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					data[0]= XVAL(this_set, last_point);
					data[1]= YVAL(this_set, last_point);
					data[2]= ERROR(this_set, last_point);
					if( this_set->lcol>= 0 ){
						data[3]= VVAL(this_set, last_point );
					}
					if( ascanf_verbose ){
						fprintf( StdErr, "FilterPoints_Box(%s): SET #%d, pnt #%dx y e: %s",
							Filter->command, this_set->set_nr, last_point, Filter->process
						);
						fflush( StdErr );
					}
					ascanf_arg_error= 0;
					n= param_scratch_len;
					compiled_fascanf( &n, Filter->process, param_scratch, NULL, data, column,
						&Filter->C_process
					);
					dN+= 1;
					sN+= 1;
				}
				*ascanf_setNumber= asn;
				*ascanf_numPoints= anp;
				if( sN ){
					RedrawSet( idx, 0 );
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr, "                    \"hit\" %d points %d times\n", dN, sN );
					}
				}
			}

			first_filter= 0;
			  /* Proceed to next filter	*/
			do{
				filter+= 1;
				Filter= &BoxFilter[filter];
				TBARprogress_header= BoxFilter[filter].command;
				filters= 1;
			} while( filter< BOX_FILTER_CLEANUP && !BoxFilter[filter].process_len );
		} while( filter< BOX_FILTER_CLEANUP );
	}
	if( BoxFilter[BOX_FILTER_CLEANUP].process ){
		  /* Only compile that expression now; ensures that Delete[] commands work in it...	*/
		new_process_BoxFilter_process( wi, BOX_FILTER_CLEANUP );
		Filter= &BoxFilter[BOX_FILTER_CLEANUP];
		TBARprogress_header= BoxFilter[BOX_FILTER_CLEANUP].command;
		if( ascanf_verbose ){
			fprintf( StdErr, "FilterPoints_Box(): cleanup: %s", Filter->process);
			fflush( StdErr );
		}
		else if( debugFlag || scriptVerbose ){
			fprintf( StdErr, "FilterPoints_Box(): %s\n", TBARprogress_header );
			fflush( StdErr );
		}
		ascanf_arg_error= 0;
		n= param_scratch_len;
		compiled_fascanf( &n, Filter->process, param_scratch, NULL, NULL, NULL, &Filter->C_process);
		dN+= 1;
	}
	if( undo_enable && BoxFilter_Undo_Enable ){
	  int j;
		BinaryDump.columns= 0;
		for( j= 0; j< MaxCols; j++ ){
			BinaryDump.data[j]= 0;
		}
		udat[0]= udat[1]= udat[2]= -1;
		fwrite( udat, sizeof(int), 3, BoxFilterUndo.fp );
		fwrite( &BinaryDump.columns, sizeof(short), 1, BoxFilterUndo.fp );
		fwrite( BinaryDump.data, sizeof(double), BinaryDump.columns, BoxFilterUndo.fp );
	}

	if( dN ){
		RedrawSet( -1, True );
	}
	  /* 20031018: user needs not be able to touch $FilterBox after this, right?! Unsetting
	   \ dollar_variable will prevent the thing from being dumped, too.
	   */
	fbox.dollar_variable= 0;
	if( debugFlag || scriptVerbose ){
		Elapsed_Since( &BFtimer, False );
		fprintf( StdErr, "FilterPoints_Box(): completed in %gs\n", BFtimer.HRTot_T );
		fflush( StdErr );
	}
	return( dN );
}

extern double *curvelen_with_discarded;

int DiscardPoint( LocalWin *wi, DataSet *this_set, int pnt_nr, int dval )
{ int set_init_pass= 0;
	if( wi && this_set && pnt_nr>= 0 && pnt_nr< this_set->numPoints ){
		if( !this_set->discardpoint ){
			this_set->discardpoint= (signed char*) calloc( this_set->allocSize, sizeof(signed char));
		}
		if( this_set->discardpoint ){
		  int discard_change= 0;
			if( dval> 0 ){
				if( !wi->drawing ){
					this_set->discardpoint[pnt_nr]= dval;
					this_set->init_pass= True;
					RedrawSet( this_set->set_nr, False );
				}
				else if( this_set->discardpoint[pnt_nr]<= 0 ){
					if( !this_set->discardpoint[pnt_nr] &&
						(!*curvelen_with_discarded || (*curvelen_with_discarded && wi->init_pass))
					){
						set_init_pass= True;
					}
					this_set->discardpoint[pnt_nr]= -1;
				}
			}
			else{
				if( this_set->discardpoint[pnt_nr]< 0 ){
					this_set->discardpoint[pnt_nr]= 0;
					discard_change= 1;
				}
				else if( dval< 0 && this_set->discardpoint[pnt_nr] ){
					this_set->discardpoint[pnt_nr]= 0;
					discard_change= 1;
				}
			}
			if( discard_change ){
				if( wi->raw_display || 
					(!*curvelen_with_discarded || (*curvelen_with_discarded && wi->init_pass))
				){
					set_init_pass= True;
				}
			}
		}
		return(set_init_pass);
	}
	else{
		return(-1);
	}
}

int DiscardPoints_Box( LocalWin *wi, double _loX, double _loY, double _hiX, double _hiY, int dval )
{ int i, idx, dN= 0, sN;
  DataSet *this_set;
  Boolean use_xval, use_yval;
	if( RAW_DISPLAY(wi) ){
		use_xval= True;
		use_yval= ! wi->absYFlag;
	}
	else{
		use_xval= use_yval= False;
	}
	for( idx= 0; idx< setNumber; idx++ ){
		if( draw_set(wi, idx) ){
		  double x, y;
			this_set= &AllSets[idx];
			sN= 0;
			for( i= 0; i< this_set->numPoints; i++ ){
				x= (use_xval)? this_set->xval[i] : this_set->xvec[i];
				y= (use_yval)? this_set->yval[i] : this_set->yvec[i];
				if( x>= _loX && y>= _loY && x<= _hiX && y<= _hiY ){
					if( !this_set->discardpoint ){
						this_set->discardpoint= (signed char*) calloc( this_set->allocSize, sizeof(signed char));
					}
					this_set->discardpoint[i]= dval;
					dN+= 1;
					sN+= 1;
				}
			}
			if( sN ){
				RedrawSet( idx, 0 );
			}
		}
	}
	if( dN ){
		RedrawSet( -1, 1 );
	}
	return( dN );
}

static double Find_Point_precision[2]= {-1,-1};
int Find_Point_use_precision= 0;

void set_Find_Point_precision( double x, double y, double *ox, double *oy )
{
	if( ox ){
		*ox= Find_Point_precision[0];
	}
	if( oy ){
		*oy= Find_Point_precision[1];
	}
	Find_Point_precision[0]= x;
	Find_Point_precision[1]= y;
}

/* Find the datapoint closest to (x,y). x and y are set to the coordinates
 \ of this point; set_rtn will point to its DataSet, and Find_Point returns
 \ its index in the set.
 \ 991012: pass do_labels==2 to find just the nearest UserLabel.
 */
int Find_Point( LocalWin *wi, double *x, double *y, DataSet **set_rtn, int do_labels, UserLabel **ulabel_rtn,
	Boolean verbose, Boolean use_sqrt, Boolean use_x, Boolean use_y
)
{ int i, j, nr= -1, disp_nr= 0;
  DataSet *this_set;
  double dist, diste, mdist= -1.0, edist= -1, px, py, tx= *x, ty= *y, err;
  char buf[2*MAXAXVAL_LEN+128], pbuf[2*MAXAXVAL_LEN+128], hit= 0, *hit_msg, *msg;
  int curX, curY;
  UserLabel *_ul;
  Boolean use_xval, use_yval, errtype_ok, ndone= True;
  extern char *splitmodestring();
  char *splmode= splitmodestring(wi);
  extern GC msgGC();

#ifdef DEBUG
/* 	XUngrabPointer(disp, CurrentTime);	*/
/* 	XUngrabKeyboard( disp, CurrentTime );	*/
/* 	XG_XSync( disp, False );	*/
#endif

	if( debugFlag ){
		fprintf( StdErr, "Find_Point(%s%s,%s%s):",
			d2str( tx, "%g", NULL), (use_x)? "" : "(ignore)",
			d2str( ty, "%g", NULL), (use_y)? "" : "(ignore)"
		);
	}
	buf[0]= '\0';
	pbuf[0]= '\0';
	if( verbose ){
	  Window dum;
	  int dum2;
	  unsigned int mask_rtn;
		XQueryPointer( disp, wi->window, &dum, &dum, &dum2, &dum2,
			  &curX, &curY, &mask_rtn
		);
		mask_rtn&= ~LockMask;
	}
	if( !ulabel_rtn ){
		ulabel_rtn= &_ul;
	}
	if( wi->raw_display ){
		use_xval= True;
		use_yval= ! wi->absYFlag;
	}
	else{
		use_xval= use_yval= False;
	}
	  /* Actually, sqrt() is a monotonically increasing function so x1 > x2 => sqrt(x1) > sqrt(x2)	*/
	use_sqrt= False;
	if( do_labels== 2 ){
		i= setNumber;
	}
	else{
		i= 0;
	}

	if( !use_x && NaNorInf(tx) ){
		tx= 0;
	}
	if( !use_y && NaNorInf(ty) ){
		ty= 0;
	}


	for( ; i< setNumber && ndone; i++ ){
		if( draw_set( wi, i) ){
			this_set= &AllSets[i];
			for( j= 0; j< this_set->numPoints && ndone; j++ ){
				if( wi->delete_it ){
					XFlush(disp);
					return(-1);
				}
				if( !DiscardedPoint( wi, this_set, j) ){
					if( this_set->numPoints> 10000 ){
						if( Handle_An_Event( wi->event_level, 1, "Find_Point-" STRING(__LINE__), wi->window, 
								StructureNotifyMask|KeyPressMask|ButtonPressMask
							)
						){
							XFlush(disp);
							return(-1);
						}
					}
					if( this_set->plot_interval<= 0 || (j % this_set->plot_interval)==0 ){
					  double ddx, ddy;
					  int phit;
						px= (use_x)? ((use_xval)? this_set->xval[j] : this_set->xvec[j]) : tx;
						py= (use_y)? ((use_yval)? this_set->yval[j] : this_set->yvec[j]) : ty;
						if( NaNorInf(px) || NaNorInf(py) ){
							goto Find_Point_nextpoint;
						}
						dist= (ddx= ( px-tx)* ( px-tx )) + (ddy= (py-ty) * (py-ty));
						msg= "";
						  /* If a point has an error, it becomes a line, the errorbar. The "target" point
						   \ can be either "outside" that line, or "inside". If inside, the point's Y coordinate
						   \ is higher than the errorbar's low Y, and lower than the errorbar's high Y - the distance
						   \ is than only dependent on the X coordinates. In the other case, the point is either
						   \ closest to the low Y or to the high Y of the errorbar, respectively: the distance is
						   \ the smallest of these two distances. This is the primary minimisation criterium for
						   \ datapoints with errorsbars. In both cases, the target point's distance to
						   \ the datapoint is taken as a secundary criterium, which is minimised when the distance from
						   \ target point to the datapoint under consideration (the primary criterium) is equal to the (current)
						   \ minimal distance.
						   */
						switch( wi->error_type[this_set->set_nr] ){
							case 1:
							case 2:
							case 3:
								errtype_ok= True;
								break;
							default:
								errtype_ok= False;
								break;
						}
						if( wi->use_errors && errtype_ok && this_set->has_error && this_set->use_error
							&& this_set->error[j] && !NaNorInf(this_set->error[j]) && use_y
						){
						  /* Include the error-region of this point.	*/
						  double pyl= (use_yval)? this_set->yval[j]- this_set->error[j] : this_set->ldyvec[j],
								pyh= (use_yval)? this_set->yval[j]+ this_set->error[j] : this_set->hdyvec[j],
								distl, disth;
							  /* The secundary crit - pythagorian distance (squared)	*/
							diste= dist;
							  /* The "true" distance to this datapoint is now re-determined, based on
							   \ which of the following two cases apply
							   */
							if( (ty>= pyl && ty<= pyh) || (ty>= pyh && ty<= pyl) ){
							  /* "within" the error (in Y) of this point. The distance is now only
							   \ proportional to the X-difference ( => an error makes a line of a point);
							   */
								if( ddx< dist ){
									dist= ddx;
									msg= " (inside error)";
								}
							}
							else{
							  /* Similarly, the outsides of an errorbar "are" the point itself.	*/
								distl= ddx + ( pyl-ty ) * ( pyl-ty );
								disth= ddx + ( pyh-ty ) * ( pyh-ty );
								if( distl< dist ){
									dist= distl;
									msg= " (low error)";
								}
								else if( disth< dist ){
									dist= disth;
									msg= " (high error)";
								}
							}
						}
						else{
						  /* This datapoint doesn't have an error, so the secundary criterium is deactivated.	*/
						  /* In vector mode, we ignore the orientation for the moment.	*/
							diste= -1;
						}
						if( debugFlag && debugLevel>= 2 ){
							fprintf( StdErr, "S%d[%d] (%g,%g),d%s=%g",
								this_set->set_nr, j,
								px, py,
								(use_sqrt)? "" : "^2",
								dist
							);
							fflush( StdErr );
						}
						  /* First check if the current primary criterium is smaller than the minimal distance found	*/
						if( dist< mdist || mdist< 0 ){
							mdist= dist;
							phit= 1;
						}
						  /* If not, check whether it is equal, the datapoint has an error, and whether the secundary
						   \ criterium is smaller than the current minimal:
						   */
						else if( dist== mdist && diste>= 0 && diste< edist ){
							edist= diste;
							phit= 1;
						}
						else{
						  /* This datapoint is not closer than any found until now	*/
							phit= 0;
						}
						if( phit ){
						  int etype= wi->error_type[this_set->set_nr];
							hit= 1;
							nr= j;
							*set_rtn= this_set;
							*ulabel_rtn= NULL;
							hit_msg= msg;
							*x= px;
							*y= py;
							if( this_set->raw_display || wi->raw_display ){
								err= this_set->error[j];
							}
							else{
								switch( etype ){
									case 4:
									case INTENSE_FLAG:
									case MSIZE_FLAG:
										err= this_set->errvec[j];
										break;
									default:
										err= (this_set->hdyvec[j] - this_set->ldyvec[j])/2;
								}
							}
							if( diste>= 0 ){
							  /* We have a valid secundary criterium. If smaller than the current minimum,
							   \ update the minimum.
							   */
								if( diste< edist || edist== -1 ){
									edist= diste;
								}
							}

							if( Find_Point_use_precision ){
								if( fabs(px-tx)<= Find_Point_precision[0] && fabs(py-ty)<= Find_Point_precision[1] ){
									ndone= False;
								}
							}

							if( debugFlag ){
								if( debugLevel< 2 ){
									fprintf( StdErr, "(%s,%s),d%s=%s(e=%s)%s ",
										d2str( px, "%g", NULL), d2str( py, "%g", NULL),
										(use_sqrt)? "" : "^2",
										d2str( dist, "%g", NULL),
										d2str( edist, "%g", NULL),
										hit_msg
									);
								}
								else{
									fprintf( StdErr, "%s\n", hit_msg );
								}
								fflush( StdErr );
							}
						}
					}
				}
Find_Point_nextpoint:;
			}
			if( debugFlag && debugLevel>= 2 ){
				fputs( "\n", StdErr );
			}
			if( hit && *set_rtn ){
				if( verbose ){
				  int len;
				  char *vx, *vy, *vi, vtbuf[256];
				  ValCategory *vcat;
					if( *buf ){
						if( *pbuf ){
							XDrawString( disp, wi->window, msgGC(wi),
								curX+ 2, curY+ 2, pbuf, strlen(pbuf)
							);
						}
						XDrawString( disp, wi->window, msgGC(wi),
							curX+ 2, curY+ 2, buf, strlen(buf)
						);
						strcpy( pbuf, buf );
					}
					if( wi->ValCat_X_axis && (vcat= Find_ValCat( wi->ValCat_X, *x, NULL, NULL)) ){
						vx= vcat->vcat_str;
					}
					else{
						vx= d2str( *x, "%g", NULL);
					}
					if( wi->ValCat_Y_axis && (vcat= Find_ValCat( wi->ValCat_Y, *y, NULL, NULL)) ){
						vy= vcat->vcat_str;
					}
					else{
						vy= d2str( *y, "%g", NULL);
					}
					if( wi->error_type[(*set_rtn)->set_nr]== INTENSE_FLAG &&
						wi->ValCat_I_axis && (vcat= Find_ValCat( wi->ValCat_I, err, NULL, NULL))
					){
						vi= vcat->vcat_str;
					}
					else if( wi->error_type[(*set_rtn)->set_nr]== 4 && (*set_rtn)->lcol>=0 ){
						sprintf( vtbuf, "%s,%s", d2str(err, NULL, NULL),
							(this_set->raw_display || wi->raw_display)?
								d2str( (*set_rtn)->lval[nr], 0, 0) : d2str( (*set_rtn)->lvec[nr], 0,0)
						);
						vi= vtbuf;
					}
					else{
						vi= d2str( err, "%g", NULL);
					}
					if( *splmode ){
						len= sprintf( buf, "[%s] ", splmode );
					}
					else{
						len= 0;
					}
					len+= sprintf( &buf[len], "Set #%d, point #%d (%s,%s,%s) N=%d [displayed set #%d]%s",
						(*set_rtn)->set_nr, nr,
						vx, vy, vi,
#if ADVANCED_STATS == 1
						(*set_rtn)->N[nr],
#elif ADVANCED_STATS == 2
						NVAL( (*set_rtn), nr),
#else
						(*set_rtn)->NumObs,
#endif
						disp_nr, hit_msg
					);
					StringCheck( buf, sizeof(buf)/sizeof(char), __FILE__, __LINE__ );
					XStoreName( disp, wi->window, buf );
					if( wi->SD_Dialog ){
					  extern int data_sn_number;
						if( data_sn_number>= 0 && data_sn_number< setNumber ){
							data_sn_number= (*set_rtn)->set_nr;
							set_changeables(2,False);
						}
					}
					XFlush( disp);
				}
				hit= 0;
			}
			disp_nr+= 1;
		}
	}

	ndone= True;

	  /* If allowed, and if any labels are present, check if maybe the target point is
	   \ closer to a label point.
	   */
	if( do_labels && wi->ulabel ){
	  UserLabel *ul= wi->ulabel;
	  Boolean set_ok;
		for( i= 0; i< wi->ulabels && ndone; i++ ){
			set_ok= True;
			if( ul->set_link>= 0 && ul->set_link< setNumber ){
				if( !draw_set(wi, ul->set_link) ){
					set_ok= False;
				}
			}
			if( set_ok ){
				for( j= 1; j>= 0; j-- ){
					if( ul->type!= UL_regular ){
					  /* stop after 1 comparison! */
						j= 0;
					}
					if( j ){
						px= (use_x)? ((use_xval)? ul->x2 : ul->tx2) : tx;
						py= (use_y)? ((use_yval)? ul->y2 : ul->ty2) : ty;
					}
					else{
						px= (use_x)? ((use_xval)? ul->x1 : ul->tx1) : tx;
						py= (use_y)? ((use_yval)? ul->y1 : ul->ty1) : ty;
					}
					if( NaNorInf(px) || NaNorInf(py) ){
						goto Find_Point_nextlabel;
					}
					  /* For line-labels, take the orthogonal distance (= shortest). This *may*
					   \ make them unavoidable, but other choices may make them "invisible".
					   */
					switch( ul->type ){
						case UL_hline:
							px= tx;
							break;
						case UL_vline:
							py= ty;
							break;
					}
					dist= ( px-tx)* ( px-tx ) + ( py-ty ) * ( py-ty );
					if( dist< mdist || mdist< 0 ){
						mdist= dist;
						nr= j;
						*set_rtn= NULL;
						*ulabel_rtn= ul;
						*x= px;
						*y= py;
						err= 0;
						hit= 1;
						if( Find_Point_use_precision ){
							if( fabs(px-tx)<= Find_Point_precision[0] && fabs(py-ty)<= Find_Point_precision[1] ){
								ndone= False;
							}
						}

						if( debugFlag ){
							fprintf( StdErr, "(%s,%s),d%s=%s ",
								d2str( *x, "%g", NULL), d2str( *y, "%g", NULL),
								(use_sqrt)? "" : "^2",
								d2str( dist, "%g", NULL)
							);
							fflush( StdErr );
						}
					}
Find_Point_nextlabel:;
				}
			}
			if( hit && *ulabel_rtn ){
				if( verbose ){
				  int len;
					if( *buf ){
						if( *pbuf ){
							XDrawString( disp, wi->window, msgGC(wi),
								curX+ 2, curY+ 2, pbuf, strlen(pbuf)
							);
						}
						XDrawString( disp, wi->window, msgGC(wi),
							curX+ 2, curY+ 2, buf, strlen(buf)
						);
						strcpy( pbuf, buf );
					}
					if( *splmode ){
						len= sprintf( buf, "[%s] ", splmode );
					}
					else{
						len= 0;
					}
					sprintf( &buf[len], "ULabel #%d, point #%d (%s,%s)",
						i, j+1,
						d2str( *x, "%g", NULL), d2str( *y, "%g", NULL)
					);
					StringCheck( buf, sizeof(buf)/sizeof(char), __FILE__, __LINE__ );
					XStoreName( disp, wi->window, buf );
					if( wi->SD_Dialog ){
					  extern int data_sn_number;
						data_sn_number= setNumber+ i;
						set_changeables(2,False);
					}
					XFlush( disp);
				}
				hit= 0;
			}
			ul= ul->next;
		}
	}
	if( debugFlag ){
		fputc( '\n', StdErr);
	}
	if( *pbuf && verbose ){
		XDrawString( disp, wi->window, msgGC(wi),
			curX+ 2, curY+ 2, pbuf, strlen(pbuf)
		);
	}
	return( nr );
}

/* Show the bifurcations (ridges) which separate the regions of attraction
 \ of one point with that of another.
 */
int Show_Ridges( LocalWin *wi, DataSet *ridge_set )
{  int N= 0, spot= 0, Spot= 0, ix, iy, x, y, X= wi->XOppX - wi->XOrgX + 1, Y= 3;
   int column[ASCANF_DATA_COLUMNS]= {0,1,2,3};
   double Data[ASCANF_DATA_COLUMNS], xprec, yprec;
      /* 
	   \ 0 1 2
	   \ 3 4 5
	   \ 6 7 8
	   */
   struct point_square{
	   int done, ix, iy;
	   double sx, sy;
	   double x, y;
	   DataSet *set;
	   UserLabel *ul;
   } *mask, *m;
   double tx, ty;

	Elapsed_Since(&wi->draw_timer, True);
	  /* Allocate a buffer that is as wide as the drawing-area, and 3 lines high, which
	   \ will hold the screen (ix,iy), the corresponding "world" (sx,sy), the
	   \ (x,y) of the nearest point, and either the set or the UserLabel that point belongs
	   \ to. Also a field which indicates if the structures hold valid info, or whether
	   \ it should be "done" again. We allocate some extra space.
	   */
	if( !(mask= (struct point_square*) calloc( X * Y+ 1, sizeof(struct point_square))) ){
		fprintf( StdErr, "Show_Ridges(): can't allocate %dx%d=%d buffer (%s)",
			X, Y,
			X * Y+ 1,
			serror()
		);
		xtb_error_box( wi->window, "Show_Ridges(): can't allocate buffer", "Failure" );
		return(1);
	}
	if( ridge_set ){
		if( (spot= ridge_set->numPoints)<= 0 ){
			ridge_set->ncols= 3;
			ridge_set->xcol= 0;
			ridge_set->ycol= 1;
			ridge_set->ecol= 2;
		}
		set_Columns( ridge_set );
		legend_setNumber= setNumber;
		ridge_set->raw_display= !wi->raw_display;
		  /* Make a guess about the number of points that'll be added	*/
		INITSIZE= (int) pow( (wi->XOppX - wi->XOrgX - 2) * (wi->XOppY - wi->XOrgY - 2) / 2.0, 0.75 );
	}

	set_Find_Point_precision( wi->win_geo.R_XUnitsPerPixel/2.0, wi->win_geo.R_YUnitsPerPixel/2.0, &xprec, &yprec );

	*ascanf_escape_value= ascanf_escape= False;

	  /* A macro to access the 1D array as 2D	*/
#define MASK2D(x,y)	mask[(y)*X + (x)]
	for( iy= wi->XOrgY+1; iy< wi->XOppY; iy++ ){
		  /* The first 2 x 3 rows are calculated outside the X-loop	*/
		x= 0;
		y= 0;
		ix= wi->XOrgX;
		m= & MASK2D(x,y);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy-1) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}
		m= & MASK2D(x, y+1);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
		}
		m= & MASK2D(x, y+2);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy+1) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}

		ix+= 1;
		m= & MASK2D(x+1, y);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy-1) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}
		m= & MASK2D(x+1, y+1);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}
		m= & MASK2D(x+1,y+2);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy+1) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}

		  /* And the rest inside the X-loop	*/
		for( ix+= 1; ix<  wi->XOppX; x++, ix++ ){
			m= & MASK2D(x+2, y);
			if( !m->done ){
				tx= TRANX( (m->ix= ix) );
				ty= TRANY( (m->iy= iy-1) );
				m->sx= m->x= Reform_X( wi, tx, ty);
				m->sy= m->y= Reform_Y( wi, ty, tx);
				Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
				m->done= 1;
			}
			m= & MASK2D(x+2, y+1);
			if( !m->done ){
				tx= TRANX( (m->ix= ix) );
				ty= TRANY( (m->iy= iy) );
				m->sx= m->x= Reform_X( wi, tx, ty);
				m->sy= m->y= Reform_Y( wi, ty, tx);
				Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
				m->done= 1;
			}
			m= &MASK2D(x+2, y+2);
			if( !m->done ){
				tx= TRANX( (m->ix= ix) );
				ty= TRANY( (m->iy= iy+1) );
				m->sx= m->x= Reform_X( wi, tx, ty);
				m->sy= m->y= Reform_Y( wi, ty, tx);
				Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
				m->done= 1;
			}
			  /* A mask of 3x3 centered around the current point of interest (ix+1,iy+1)
			   \ has been evaluated. Now check if it contains valid info, and whether each
			   \ of its 8 neighbours have different info (a different point, set or UserLabel).
			   \ If so, this point is on a ridge... and it is drawn with colour 0, or the colour
			   \ of the set it belongs to. Alternatively, it could be added to some (new) dataset...
			   \ (which will of course have to be "hidden"=undrawn to prevent unwanted artefacts ;))
			   */
			m= &MASK2D(x+1, y+1);
			if( m->set || m->ul ){
			  int _x, _y;
			  Boolean hit= False;
			  struct point_square *mm;
				for( _y= y; _y< y+2 && !hit; _y++ ){
					for( _x= x; _x< x+2 && !hit; _x++ ){
						mm= &MASK2D(_x, _y);
						if( (mm->set || mm->ul) ){
							if( mm->x!= m->x || mm->y!= m->y ||
								mm->set!= m->set || mm->ul!= m->ul
							){
							  int set_nr= (m->set)? m->set->set_nr : 0;
							  int pixvalue= (m->set)? m->set->pixvalue : 0;
								wi->dev_info.xg_dot(wi->dev_info.user_state,
											m->ix, m->iy,
											P_PIXEL, 0, pixvalue, (m->set)? m->set->pixelValue : 0, set_nr, m->set);
								if( ridge_set ){
									Data[0]= m->sx;
									Data[1]= m->sy;
/* 									Data[2]= sqrt( (mm->x- m->x)*(mm->x- m->x) + (mm->y- m->y)*(mm->y- m->y) );	*/
									Data[2]= sqrt( (m->sx- m->x)*(m->sx- m->x) + (m->sy- m->y)*(m->sy- m->y) );
									AddPoint_discard= False;
									AddPoint( &ridge_set, &spot, &Spot, 3, Data, column,
										"Show_Ridge()", 0, 0, NULL, "", &wi->process
									);
#if ADVANCED_STATS == 1
									ridge_set->N[spot-1]= 1;
#elif ADVANCED_STATS == 2
									if( ridge_set->Ncol>= 0 ){
										ridge_set->columns[ridge_set->Ncol][spot-1]= 1;
									}
#endif
								}
								else{
									spot+= 1;
								}
								hit= True;
							}
						}
					}
				}
			}
			  /* Anyway, this was a (potential) bifurcation point	*/
			N+= 1;
			  /* Check for some event. This routine can take quite a while...	*/
			if( Handle_An_Event( wi->event_level, 1, "Show_Ridges-" STRING(__LINE__), wi->window, 
					StructureNotifyMask|KeyPressMask|ButtonPressMask
				) || ascanf_escape
			){
				XG_XSync( disp, True );
				if( wi->delete_it!= -1 ){
					wi->event_level--;
					wi->redraw= wi->redraw_val;
				}
				xfree( mask );
				if( ridge_set ){
					Destroy_Set( ridge_set, False );
				}
				set_Find_Point_precision( xprec, yprec, NULL, NULL );
				return(1);
			}
			if( wi->delete_it== -1 ){
				xfree( mask );
				if( ridge_set ){
					Destroy_Set( ridge_set, False );
				}
				set_Find_Point_precision( xprec, yprec, NULL, NULL );
				return(1);
			}
			if( wi->redraw || wi->halt ){
				wi->event_level--;
				xfree( mask );
				if( ridge_set ){
					Destroy_Set( ridge_set, False );
				}
				set_Find_Point_precision( xprec, yprec, NULL, NULL );
				return(1);
			}
		}
		XG_XSync( disp, False );
		  /* All points on 3 scanlines have been evaluated. Now shift
		   \ the lowest 2 lines upward (they can be used again), and
		   \ "reset" the lowest line - it will receive info on the next
		   \ scanline. Und so weit.
		   */
		for( ix= 0; ix< X; ix++ ){
			MASK2D(ix,0)= MASK2D(ix,1);
			MASK2D(ix,1)= MASK2D(ix,2);
			MASK2D(ix,2).done= False;
		}
	}
	if( ridge_set ){
		ridge_set->allocSize= ridge_set->numPoints;
		realloc_points( ridge_set, ridge_set->allocSize+ 2, False );
		if( spot> maxitems ){
			maxitems= spot;
			if( wi ){
				realloc_Xsegments();
			}
		}
		ridge_set->NumObs= 1;
		if( !ridge_set->setName ){
			ridge_set->setName= XGstrdup( "Bifurcations" );
		}
	}
	Elapsed_Since( &wi->draw_timer, True );
	{ char buf[256];
		sprintf( buf, "%d of max. %d in %s(%s)s",
			spot, N,
			d2str( wi->draw_timer.Time, "%g", NULL),
			d2str( wi->draw_timer.Tot_Time, "%g", NULL)
		);
		XStoreName( disp, wi->window, buf );
		XFlush( disp);
	}
	xfree( mask );
	set_Find_Point_precision( xprec, yprec, NULL, NULL );
	return(0);
}

/* Show the bifurcations (ridges) which separate the regions of attraction
 \ of one point with that of another.
 */
int Show_Ridges2( LocalWin *wi, DataSet *ridge_set )
{  int N= 0, spot= 0, Spot= 0, ix, iy, x, y, Y= wi->XOppY - wi->XOrgY + 1, X= 3;
   int column[ASCANF_DATA_COLUMNS]= {0,1,2,3};
   double Data[ASCANF_DATA_COLUMNS], xprec, yprec;
      /* 
	   \ 0 1 2
	   \ 3 4 5
	   \ 6 7 8
	   */
   struct point_square{
	   int done, ix, iy;
	   double sx, sy;
	   double x, y;
	   DataSet *set;
	   UserLabel *ul;
   } *mask, *m;
   double tx, ty;

	Elapsed_Since(&wi->draw_timer, True);
	  /* Allocate a buffer that is as wide as the drawing-area, and 3 lines high, which
	   \ will hold the screen (ix,iy), the corresponding "world" (sx,sy), the
	   \ (x,y) of the nearest point, and either the set or the UserLabel that point belongs
	   \ to. Also a field which indicates if the structures hold valid info, or whether
	   \ it should be "done" again. We allocate some extra space.
	   */
	if( !(mask= (struct point_square*) calloc( X * Y+ 1, sizeof(struct point_square))) ){
		fprintf( StdErr, "Show_Ridges(): can't allocate %dx%d=%d buffer (%s)",
			X, Y,
			X * Y+ 1,
			serror()
		);
		xtb_error_box( wi->window, "Show_Ridges2(): can't allocate buffer", "Failure" );
		return(1);
	}
	if( ridge_set ){
		if( (spot= ridge_set->numPoints)<= 0 ){
			ridge_set->ncols= 3;
			ridge_set->xcol= 0;
			ridge_set->ycol= 1;
			ridge_set->ecol= 2;
		}
		set_Columns( ridge_set );
		legend_setNumber= setNumber;
		ridge_set->raw_display= !wi->raw_display;
		  /* Make a guess about the number of points that'll be added	*/
		INITSIZE= (int) pow( (wi->XOppX - wi->XOrgX - 2) * (wi->XOppY - wi->XOrgY - 2) / 2.0, 0.75 );
	}

	*ascanf_escape_value= ascanf_escape= False;

	set_Find_Point_precision( wi->win_geo.R_XUnitsPerPixel/2.0, wi->win_geo.R_YUnitsPerPixel/2.0, &xprec, &yprec );

	  /* A macro to access the 1D array as 2D	*/
#define MASK2(y,x)	mask[(x)*Y + (y)]
	for( ix= wi->XOrgX+1; ix< wi->XOppX; ix++ ){
		  /* The first 2 x 3 rows are calculated outside the Y-loop	*/
		x= 0;
		y= 0;
		iy= wi->XOrgY;
		m= & MASK2(y,x);
		if( !m->done ){
			tx= TRANX( (m->ix= ix-1) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}
		m= & MASK2(y, x+1);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
		}
		m= & MASK2(y, x+2);
		if( !m->done ){
			tx= TRANX( (m->ix= ix+1) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}

		iy+= 1;
		m= & MASK2(y+1, x);
		if( !m->done ){
			tx= TRANX( (m->ix= ix-1) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}
		m= & MASK2(y+1, x+1);
		if( !m->done ){
			tx= TRANX( (m->ix= ix) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}
		m= & MASK2(y+1,x+2);
		if( !m->done ){
			tx= TRANX( (m->ix= ix+1) );
			ty= TRANY( (m->iy= iy) );
			m->sx= m->x= Reform_X( wi, tx, ty);
			m->sy= m->y= Reform_Y( wi, ty, tx);
			Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
			m->done= 1;
		}

		  /* And the rest inside the X-loop	*/
		for( iy+= 1; iy<  wi->XOppY; y++, iy++ ){
			m= & MASK2(y+2, x);
			if( !m->done ){
				tx= TRANX( (m->ix= ix-1) );
				ty= TRANY( (m->iy= iy) );
				m->sx= m->x= Reform_X( wi, tx, ty);
				m->sy= m->y= Reform_Y( wi, ty, tx);
				Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
				m->done= 1;
			}
			m= & MASK2(y+2, x+1);
			if( !m->done ){
				tx= TRANX( (m->ix= ix) );
				ty= TRANY( (m->iy= iy) );
				m->sx= m->x= Reform_X( wi, tx, ty);
				m->sy= m->y= Reform_Y( wi, ty, tx);
				Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
				m->done= 1;
			}
			m= &MASK2(y+2, x+2);
			if( !m->done ){
				tx= TRANX( (m->ix= ix+1) );
				ty= TRANY( (m->iy= iy) );
				m->sx= m->x= Reform_X( wi, tx, ty);
				m->sy= m->y= Reform_Y( wi, ty, tx);
				Find_Point( wi, &(m->x), &(m->y), &(m->set), 1, &(m->ul), False, True, True, True );
				m->done= 1;
			}
			  /* A masker of 3x3 centered around the current point of interest (ix+1,iy+1)
			   \ has been evaluated. Now check if it contains valid info, and whether each
			   \ of its 8 neighbours have different info (a different point, set or UserLabel).
			   \ If so, this point is on a ridge... and it is drawn with colour 0, or the colour
			   \ of the set it belongs to. Alternatively, it could be added to some (new) dataset...
			   \ (which will of course have to be "hidden"=undrawn to prevent unwanted artefacts ;))
			   */
			m= &MASK2(y+1, x+1);
			if( m->set || m->ul ){
			  int _x, _y;
			  Boolean hit= False;
			  struct point_square *mm;
				for( _x= x; _x< x+2 && !hit; _x++ ){
					for( _y= y; _y< y+2 && !hit; _y++ ){
						mm= &MASK2(_y, _x);
						if( (mm->set || mm->ul) ){
							if( mm->x!= m->x || mm->y!= m->y ||
								mm->set!= m->set || mm->ul!= m->ul
							){
							  int set_nr= (m->set)? m->set->set_nr : 0;
							  int pixvalue= (m->set)? m->set->pixvalue : 0;
								wi->dev_info.xg_dot(wi->dev_info.user_state,
											m->ix, m->iy,
											P_PIXEL, 0, pixvalue, (m->set)? m->set->pixelValue : 0, set_nr, m->set);
								if( ridge_set ){
									Data[0]= m->sx;
									Data[1]= m->sy;
/* 									Data[2]= sqrt( (mm->x- m->x)*(mm->x- m->x) + (mm->y- m->y)*(mm->y- m->y) );	*/
									Data[2]= sqrt( (m->sx- m->x)*(m->sx- m->x) + (m->sy- m->y)*(m->sy- m->y) );
									AddPoint_discard= False;
									AddPoint( &ridge_set, &spot, &Spot, 3, Data, column,
										"Show_Ridge2()", 0, 0, NULL, "", &wi->process
									);
#if ADVANCED_STATS == 1
									ridge_set->N[spot-1]= 1;
#elif ADVANCED_STATS == 2
									if( ridge_set->Ncol>= 0 ){
										ridge_set->columns[ridge_set->Ncol][spot-1]= 1;
									}
#endif
								}
								else{
									spot+= 1;
								}
								hit= True;
							}
						}
					}
				}
			}
			  /* Anyway, this was a (potential) bifurcation point	*/
			N+= 1;
			  /* Check for some event. This routine can take quite a while...	*/
			if( Handle_An_Event( wi->event_level, 1, "Show_Ridges-" STRING(__LINE__), wi->window, 
					StructureNotifyMask|KeyPressMask|ButtonPressMask
				) || ascanf_escape
			){
				XG_XSync( disp, True );
				if( wi->delete_it!= -1 ){
					wi->event_level--;
					wi->redraw= wi->redraw_val;
				}
				xfree( mask );
				if( ridge_set ){
					Destroy_Set( ridge_set, False );
				}
				set_Find_Point_precision( xprec, yprec, NULL, NULL );
				return(1);
			}
			if( wi->delete_it== -1 ){
				xfree( mask );
				if( ridge_set ){
					Destroy_Set( ridge_set, False );
				}
				set_Find_Point_precision( xprec, yprec, NULL, NULL );
				return(1);
			}
			if( wi->redraw || wi->halt ){
				wi->event_level--;
				xfree( mask );
				if( ridge_set ){
					Destroy_Set( ridge_set, False );
				}
				set_Find_Point_precision( xprec, yprec, NULL, NULL );
				return(1);
			}
		}
		XG_XSync( disp, False );
		  /* All points on 3 scanlines have been evaluated. Now shift
		   \ the lowest 2 lines upward (they can be used again), and
		   \ "reset" the lowest line - it will receive info on the next
		   \ scanline. Und so weit.
		   */
		for( iy= 0; iy< Y; iy++ ){
			MASK2(iy,0)= MASK2(iy,1);
			MASK2(iy,1)= MASK2(iy,2);
			MASK2(iy,2).done= False;
		}
	}
	if( ridge_set ){
		ridge_set->allocSize= ridge_set->numPoints;
		realloc_points( ridge_set, ridge_set->allocSize+ 2, False );
		if( spot> maxitems ){
			maxitems= spot;
			if( wi ){
				realloc_Xsegments();
			}
		}
		ridge_set->NumObs= 1;
		if( !ridge_set->setName ){
			ridge_set->setName= XGstrdup( "Bifurcations" );
		}
	}
	Elapsed_Since( &wi->draw_timer, True );
	{ char buf[256];
		sprintf( buf, "%d of max. %d in %s(%s)s",
			spot, N,
			d2str( wi->draw_timer.Time, "%g", NULL),
			d2str( wi->draw_timer.Tot_Time, "%g", NULL)
		);
		XStoreName( disp, wi->window, buf );
		XFlush( disp);
	}
	xfree( mask );
	set_Find_Point_precision( xprec, yprec, NULL, NULL );
	return(0);
}

/* Find the two consecutive points in *set that either "surround" the given <tx>, and
 \ calculate the interpolated y at <tx> (return 1), or if <tx> is an element of *set, return
 \ those (x,y,e) (return -1).
 */
int Interpolate( LocalWin *wi, DataSet *this_set, double tx, int *pnr, double *xp, double *yp, double *ep, double *np, Boolean use_transformed )
{ int j;
  double xs, xl, ys, yl, es, el;
  int hit= 0;
	xs= (use_transformed)? this_set->xvec[0] : XVAL(this_set,0);
	for( j= 1; j< this_set->numPoints && hit== 0; j++ ){
		if( !DiscardedPoint( wi, this_set, j) ){
			xl= (use_transformed)? this_set->xvec[j] : XVAL(this_set,j);
			if( xs == tx ){
				hit= -1;
				*pnr= j- 1;
			}
			else{
				if( xs< tx && xl > tx ){
					hit= 1;
					*pnr= j-1;
				}
				else{
					xs= xl;
				}
			}
		}
	}
	if( hit == 1 ){
		ys= (use_transformed)? this_set->yvec[*pnr] : YVAL(this_set,*pnr);
		es= (this_set->use_error)? ((use_transformed)? this_set->errvec[*pnr] : ERROR(this_set,*pnr)) : 0;
		yl= (use_transformed)? this_set->yvec[*pnr+1] : YVAL(this_set,(*pnr+1));
		el= (this_set->use_error)? ((use_transformed)? this_set->errvec[*pnr+1] : ERROR(this_set,(*pnr+1))) : 0;
		*xp= tx;
		*yp= ys + (yl - ys)* ((tx - xs)/(xl - xs));
#ifdef ADVANCED_STATS
		{ int ns, nl;
			*np= 0;
			if( (ns= NVAL( this_set, *pnr)) ){
				(*np)+= 1;
			}
			if( (nl= NVAL( this_set, *pnr+1)) ){
				(*np)+= 1;
			}
/* 			*np= (ns + nl);	*/
/* 			*ep= (es * this_set->N[*pnr] + el * this_set->N[*pnr+1])/ *np;	*/
			*ep= (es * ns + el * nl)/ *np;
			*np/= 2;
		}
#else
		  /* Assume that the error or the interpolated point is just the average of the two
		   \ surrounding errors.
		   */
		*np= this_set->NumObs;
		*ep= (es + el)/2;
#endif
		if( debugFlag && debugLevel ){
			fprintf( StdErr, "Interpolate(%s,#%d) between (%s,%s,%s) and (%s,%s,%s) => (%s,%s,%s), N=%s\n",
				d2str( tx, "%g", NULL), this_set->set_nr,
				d2str( xs, "%g", NULL),
				d2str( ys, "%g", NULL),
				d2str( es, "%g", NULL),
				d2str( xl, "%g", NULL),
				d2str( yl, "%g", NULL),
				d2str( el, "%g", NULL),
				d2str( *xp, "%g", NULL),
				d2str( *yp, "%g", NULL),
				d2str( *ep, "%g", NULL),
				d2str( *np, "%g", NULL)
			);
		}
	}
	else if( hit== -1 ){
		*xp= xs;
		*yp= (use_transformed)? this_set->yvec[*pnr] : YVAL(this_set,*pnr);
		*ep= (this_set->use_error)? ((use_transformed)? this_set->errvec[*pnr] : ERROR(this_set,*pnr)) : 0;
	}
	return( hit );
}

typedef struct YAverage{
  int idx;
  double x, y, errr;
  SimpleStats SS_Y;
} YAverage;

static int Xsort_YAverages( YAverage *a, YAverage *b)
{
	if( a->x< b->x ){
		return( -1 );
	}
	else if( a->x > b->x ){
		return( 1 );
	}
	else{
		return( 0 );
	}
}

static int Isort_YAverages( YAverage *a, YAverage *b)
{
	if( a->idx< b->idx ){
		return( -1 );
	}
	else if( a->idx > b->idx ){
		return( 1 );
	}
	else{
		return( 0 );
	}
}

static int XIsort_YAverages( YAverage *a, YAverage *b)
{
	if( a->x< b->x ){
		return( -1 );
	}
	else if( a->x > b->x ){
		return( 1 );
	}
	else{
		return( Isort_YAverages(a,b) );
	}
}

static int IXsort_YAverages( YAverage *a, YAverage *b)
{
	if( a->idx< b->idx ){
		return( -1 );
	}
	else if( a->idx > b->idx ){
		return( 1 );
	}
	else{
		return( Xsort_YAverages(a,b) );
	}
}

int Average( LocalWin *wi, int *av_sets, char *filename, int sub_div, int line_count, char *buffer, Process *proc,
	Boolean use_transformed, Boolean XYAveraging, char *YAv_Sort, Boolean add_Interpolations, Boolean ScattEllipse
)
{  SimpleStats SS_X= EmptySimpleStats, SS_Y= EmptySimpleStats, SS_OBS= EmptySimpleStats;
   int idx, i, j, maxPoints= 0, spot, Spot, ap= 0, avSet= setNumber;
   DataSet *av_set, *this_set, *other_set;
   double ni, nj, Data[4], _xval, _yval, _error, txval, tyval, terror;
   int column[ASCANF_DATA_COLUMNS]= {0,1,2,3};
   int add_X, add_Y, pt_idx, p_idx= -1, NN= 0;
   unsigned int add_f, tlen= 0, add_t, *tcount, *title;
   long YSort= *((long*) YAv_Sort);
#ifdef __GNUC__
   char label[MaxSets * 5+128];
#else
   char *label= calloc( MaxSets* 5+128, sizeof(char));
   if( !label ){
	   fprintf( StdErr, "Average(): can't get memory for labelstring (%s)\n", serror() );
	   if( wi ){
		   xtb_error_box( wi->window, "Average(): no memory for labelstring\n", "Error" );
	   }
	   return(0);
   }
#endif

	if( debugFlag ){
		fprintf( StdErr, "Average({ " );
	}
	if( ScattEllipse ){
		sprintf( label, "ScattEllipse" );
	}
	else if( !XYAveraging ){
		sprintf( label, "Y-%s", YAv_Sort );
	}
	else{
		label[0]= '\0';
	}
	if( add_Interpolations ){
		strcat( label, " I");
	}
	  /* the last set with serial setNumber will receive the data!	*/
	for( p_idx= -1, i= 0, idx= 0; idx< setNumber; idx++ ){
		if( debugFlag ){
			fprintf( StdErr, "%d ", av_sets[idx] );
		}
		if( av_sets[idx] ){
			if( label[0] ){
				sprintf( label, "%s %d", label, idx );
			}
			else{
				sprintf( label, "%d", idx );
			}
			i+= 1;
			if( AllSets[idx].titleText ){
				if( p_idx< 0 || !tlen || strcmp( AllSets[idx].titleText, AllSets[p_idx].titleText) ){
					  /* An "empty" title is specified by an empty string, not a NULL pointer. The
					   \ strlen of that is 1 (?!)
					   */
					if( AllSets[idx].titleText[0] ){
						tlen= MAX( tlen, strlen( AllSets[idx].titleText ) );
					}
				}
				p_idx= idx;
			}
			maxPoints= MAX( maxPoints, AllSets[idx].numPoints );
			  /* NN is the total of number of points in all sets that are to be averaged.	*/
			NN+= AllSets[idx].numPoints;
		}
	}
	if( i== 0 ){
		if( wi ){
			xtb_error_box( wi->window, "Must have at least 2 sets for averaging!\n", "Error" );
		}
		else{
			fprintf( StdErr, "Must have at least 2 sets for averaging!\n" );
		}
#ifndef __GNUC__
		xfree( label );
#endif
		return(0);
	}
	use_transformed= (use_transformed)? 1 : 0;
	for( idx= 0; idx< setNumber; idx++ ){
		if( AllSets[idx].average_set && !strcmp( label, AllSets[idx].average_set ) ){
		  int rd= (AllSets[idx].raw_display)? 1 : 0;
			if( !(use_transformed ^ rd) ){
				if( wi ){
					xtb_error_box( wi->window, "These sets have already been averaged!\nsetName shown above",
						AllSets[idx].setName
					);
#ifndef __GNUC__
					xfree( label );
#endif
					return(-idx);
				}
				else{
					fprintf( StdErr, "Warning: these sets have already been averaged!\n" );
				}
			}
		}
		else{
		  /* sets already averaged, but the following is inferred:
		   \ previous "raw", current request for transformed (rd=0, use_transformed=1)
		   \ previous transformed, current request for raw   (rd=1, use_transformed=0)
		   \ In both cases, rd ^ use_transformed == 1
		   \ This can of course be thwarted by changing the previous average's raw_display parameter...
		   */
		}
	}

	  /* Check if we want to average the transformed values, and the window
	   \ needs a (silenced) redraw.
	   */
	if( use_transformed && (wi->redraw || wi->raw_display) ){
	  int old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True ),
			rd= wi->raw_display;
		wi->raw_display= False;
		if( debugFlag ){
			fprintf( StdErr, "Average(transformed): window needs redraw - doing silently\n");
		}
		RedrawNow( wi );
		wi->dev_info.xg_silent( wi->dev_info.user_state, old_silent );
		wi->raw_display= rd;
	}

	av_set= &AllSets[avSet];
	if( av_set->numPoints< 0 ){
		av_set->numPoints= 0;
	}
	if( !av_set->numPoints ){
#if ADVANCED_STATS==2
		av_set->ncols= 4;
		av_set->Ncol= 3;
#else
		av_set->ncols= 3;
#endif
		av_set->xcol= 0;
		av_set->ycol= 1;
		av_set->ecol= 2;
	}
	set_Columns( av_set );
	if( debugFlag ){
		if( XYAveraging ){
			fprintf( StdErr, "}) => %d sets \"%s\", %d points, in set #%d with %d points\n",
				i, label, maxPoints,
				avSet, av_set->numPoints
			);
		}
		else{
			fprintf( StdErr, "}) => %d sets \"%s\", maximal %d points, in set #%d with %d points\n",
				i, label, NN,
				avSet, av_set->numPoints
			);
		}
	}
	add_X= !av_set->XUnits;
	add_Y= !av_set->YUnits;
	add_f= !av_set->fileName;
	add_t= !av_set->titleText && tlen;
	if( add_t ){
		tlen+= 1;
		if( !(av_set->titleText= calloc( tlen+ 1, sizeof(char))) ||
			!(title= (unsigned int*) calloc( tlen+1, sizeof(int))) ||
			!(tcount= (unsigned int*) calloc( tlen+ 1, sizeof(int)))
		){
			xfree( av_set->titleText );
			add_t= 0;
		}
		else if( debugFlag ){
			fprintf( StdErr, "Will construct \"average\" title of length %d\n", tlen- 1 );
		}
	}
	  /* determine the XUnits and YUnits of this set: all of them...	*/
	for( pt_idx= -1, p_idx= -1, idx= 0; idx< setNumber; idx++ ){
	  int marge= strlen("*XYlabel*")+ 3;
		if( av_sets[idx] ){
			if( add_X && AllSets[idx].XUnits ){
				if( av_set->XUnits ){
					if( p_idx< 0 || !AllSets[p_idx].XUnits || strcmp( AllSets[idx].XUnits, AllSets[p_idx].XUnits) ){
						  /* We check if the result doesn't become longer than the read buffer	*/
						if( strlen( av_set->XUnits)+ strlen(AllSets[idx].XUnits)+ 4< LMAXBUFSIZE- marge ){
							av_set->XUnits= concat2( av_set->XUnits, " ; ", AllSets[idx].XUnits, NULL);
							if( debugFlag && debugLevel ){
								fprintf( StdErr, "X+ \"%s\" ", AllSets[idx].XUnits );
							}
						}
					}
				}
				else{
					  /* We should check the length here also, but it doesn't seem to necessary
					   \ (how many labels will be too long... it's just because of the fast growth
					   \ when concatting that this check is necessary!
					   */
					av_set->XUnits= XGstrdup( AllSets[idx].XUnits );
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "X= \"%s\" ", AllSets[idx].XUnits );
					}
				}
			}
			if( add_Y && AllSets[idx].YUnits ){
				if( av_set->YUnits ){
					if( p_idx< 0 || !AllSets[p_idx].YUnits || strcmp( AllSets[idx].YUnits, AllSets[p_idx].YUnits) ){
						if( strlen( av_set->YUnits)+ strlen(AllSets[idx].YUnits)+ 4< LMAXBUFSIZE- marge ){
							av_set->YUnits= concat2( av_set->YUnits, " ; ", AllSets[idx].YUnits, NULL);
							if( debugFlag && debugLevel ){
								fprintf( StdErr, "Y+ \"%s\" ", AllSets[idx].YUnits );
							}
						}
					}
				}
				else{
					av_set->YUnits= XGstrdup( AllSets[idx].YUnits );
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "Y= \"%s\" ", AllSets[idx].YUnits );
					}
				}
			}
			if( add_f && AllSets[idx].fileName ){
				if( av_set->fileName ){
					if( p_idx< 0 || !AllSets[p_idx].fileName || strcmp( AllSets[idx].fileName, AllSets[p_idx].fileName) ){
						if( strlen( av_set->fileName)+ strlen(AllSets[idx].fileName)+ 4< LMAXBUFSIZE- marge ){
							av_set->fileName= concat2( av_set->fileName, ", ", AllSets[idx].fileName, NULL);
						}
					}
				}
				else{
					av_set->fileName= XGstrdup( AllSets[idx].fileName );
				}
			}
			  /* This takes care of "averaging" the titles...	*/
			if( add_t && AllSets[idx].titleText ){
			  char *s= AllSets[idx].titleText;
			  unsigned int *d= title, *n= tcount, N= 0;
				if( pt_idx< 0 || !*d || strcmp( s, AllSets[pt_idx].titleText ) ){
					while( *s && N< tlen ){
						*d++ += *s++;
						*n++ += 1;
						N+= 1;
					}
					pt_idx= idx;
				}
			}
			p_idx= idx;
		}
	}
	if( add_t ){
	  int i;
	  char *d= av_set->titleText;
	  unsigned int *s= title, *n= tcount;
		for( i= 0; i< tlen; i++, d++, s++, n++ ){
			if( *n ){
				*d = (char) (*s / *n);
			}
		}
		av_set->titleText[tlen-1]= '\0';
		xfree( title );
		xfree( tcount );
	}
	if( add_X && av_set->XUnits ){
		av_set->XUnits= realloc( av_set->XUnits, sizeof(char)* (1+ strlen(av_set->XUnits)) );
	}
	if( add_Y && av_set->YUnits ){
		av_set->YUnits= realloc( av_set->YUnits, sizeof(char)* (1+ strlen(av_set->YUnits)) );
	}
	if( add_f && av_set->fileName ){
		av_set->fileName= realloc( av_set->fileName, sizeof(char)* (1+ strlen(av_set->fileName)) );
	}
	if( (add_X || add_Y) && debugFlag ){
		fputs( "\n", StdErr );
	}
	spot= av_set->numPoints;
	Spot= av_set->numPoints;
	legend_setNumber= setNumber;
	SS_Reset_(SS_OBS);
	if( ScattEllipse ){
		if( debugFlag ){
			fprintf( StdErr, "ScattEllipse pass 1 - determination of centre of scatter" );
		}
		SS_Reset_(SS_X);
		SS_Reset_(SS_Y);
		for( idx= 0; idx< setNumber; idx++ ){
			if( av_sets[idx] ){
				this_set= &AllSets[idx];
				for( j= 0; j< this_set->numPoints; j++ ){
					if( !DiscardedPoint( wi, this_set, j) ){
#ifdef ADVANCED_STATS
						nj= NVAL( this_set, j);
						if( nj<= 0 ){
							nj= 1;
						}
#else
						nj= this_set->NumObs;
#endif
						if( use_transformed ){
							_xval= this_set->xvec[j];
							_yval= this_set->yvec[j];
							_error= (this_set->use_error && !NaN(this_set->errvec[j]))? fabs( this_set->errvec[j] ) : 0;
						}
						else{
							_xval= XVAL(this_set,j);
							_yval= YVAL(this_set,j);
							_error= (this_set->use_error && !NaN(ERROR(this_set,j)))? ERROR(this_set,j) : 0;
						}
						if( debugFlag && debugLevel== 1 ){
							fprintf( StdErr, "; point %d.%d (%s,%s)*%s/(1+%s)", idx, j,
								d2str( _xval, "%g", NULL),
								d2str( _yval, "%g", NULL),
								d2str( nj, "%g", NULL),
								d2str( _error, "%g", NULL)
							);
							fflush( StdErr );
						}
						nj/= (1+ _error);
						SS_Add_Data_( SS_X, 1, _xval, 1 );
						SS_Add_Data_( SS_Y, 1, _yval, nj );
					}
				}
			}
		}
		if( debugFlag ){
			fputc( '\n', StdErr );
		}
		fprintf( StdErr, "X scatter: %s\n", SS_sprint_full( NULL, "%g", " +- ", 0, &SS_X ) );
		fprintf( StdErr, "Y scatter: %s\n", SS_sprint_full( NULL, "%g", " +- ", 0, &SS_Y ) );
		if( SS_X.count && SS_Y.count ){
		  double mx= SS_Mean_(SS_X), my= SS_Mean_(SS_Y),
				sx= SS_St_Dev_(SS_X), sy= SS_St_Dev_(SS_Y), dx, dy, angle, angle1, angle2,
				cos_angle, sin_angle;
		  double cvM[2][2], L1, L2, Vy, Uy,
				Sx2= 0, Sxy= 0, Sy2= 0, Swy= 0, Swy2= 0;
		  int nnj= 0;
			if( sx== 0 || sy== 0 ){
				if( debugFlag ){
					fprintf( StdErr, "Sorry, won't do this for data on a horizontal or vertical line\n" );
				}
#ifndef __GNUC__
				xfree(label);
#endif
				return(0);
			}
			  /* Now do a 2D PCA to determine the axes and direction (rotation) of the ellipse.  */
			if( debugFlag ){
				fprintf( StdErr, "ScattEllipse pass 2 - PCA\n" );
			}
			for( idx= 0; idx< setNumber; idx++ ){
				if( av_sets[idx] ){
					this_set= &AllSets[idx];
					for( j= 0; j< this_set->numPoints; j++ ){
						if( !DiscardedPoint( wi, this_set, j) ){
#ifdef ADVANCED_STATS
							nj= NVAL( this_set, j);
							if( nj<= 0 ){
								nj= 1;
							}
#else
							nj= this_set->NumObs;
#endif
							if( use_transformed ){
								_xval= this_set->xvec[j];
								_yval= this_set->yvec[j];
								_error= (this_set->use_error && !NaN(this_set->errvec[j]))? fabs( this_set->errvec[j] ) : 0;
							}
							else{
								_xval= XVAL(this_set,j);
								_yval= YVAL(this_set,j);
								_error= (this_set->use_error && !NaN(ERROR(this_set,j)))? ERROR(this_set,j) : 0;
							}
							dx= _xval- mx;
							dy= _yval- my;
							nj/= (1+ _error);
							Sx2+= dx* dx;
							Sxy+= dx* dy* nj;
							Swy+= nj;
							Sy2+= dy* dy* nj* nj;
							Swy2+= nj* nj;
							nnj+= 1;
						}
					}
				}
			}
			if( Sxy< 0 ){
				Sxy= 0;
			}
			  /* The covariance matrix:	*/
			cvM[0][0]= sqrt(Sx2); cvM[0][1]= sqrt(Sxy/ (Swy/nnj));
			cvM[1][0]= cvM[0][1]; cvM[1][1]= sqrt(Sy2/ (Swy2/nnj));

			{ double a= 1,
					b= -(cvM[0][0]+ cvM[1][1]),
					c= cvM[0][0]* cvM[1][1]- cvM[0][1]* cvM[1][0],
					cudet= sqrt( b* b- 4* a* c );
				  /* Its eigenvalues:	*/
				L1= (-b + cudet)/ (2* a);
				L2= (-b - cudet)/ (2* a);
			}

			  /* The Y co-ordinates of the corresponding eigenvectors.	*/
			if( cvM[0][1]== 0 ){
				if( cvM[0][0] > cvM[1][1] ){
					angle1= 0;
					angle2= 90;
				}
				else{
					angle1= 90;
					angle2= 0;
				}
			}
			else{
				Vy= (L1- cvM[0][0]) / cvM[0][1];
				Uy= (L2- cvM[0][0]) / cvM[0][1];
#ifndef degrees
#	define degrees(a)			((a)*57.295779512)
#endif
				  /* And their angles. angle2 - angle1 should be +- 90 degrees!	*/
				angle1= degrees( atan3( 1, Vy) );
				angle2= degrees( atan3( 1, Uy) );

			}

			fprintf( StdErr, "CoVarMatrix: [ [%s,%s] [%s,%s] ]; eigenv1 %s@%s deg, eigenv2 %s@%s deg\n",
				d2str( cvM[0][0], NULL, NULL), d2str( cvM[0][1], NULL, NULL),
				d2str( cvM[1][0], NULL, NULL), d2str( cvM[1][1], NULL, NULL),
				d2str( L1, NULL, NULL), d2str( angle1, NULL, NULL),
				d2str( L2, NULL, NULL), d2str( angle2, NULL, NULL)
			);

			  /* L1 and L2 are related to the corresponding (hor or vert) standard deviation by a
			   \ factor sqrt( weight_sum - 1 ).
			   */
			angle= angle1;

			  /* For the radii of the ellipsis, we want the standard deviation along the axes specified
			   \ by the principal components..! So we have to calculate the scatter once more, by rotating
			   \ the data over -angle.
			   */
			if( debugFlag ){
				fprintf( StdErr, "ScattEllipse pass 3 - scatter in rotated points" );
			}
			if( angle ){
			  double gb= Gonio_Base_Value;
				Gonio_Base( NULL, 360.0, 0.0);
				cos_angle= Cos( -angle );
				sin_angle= Sin( -angle );
				Gonio_Base( NULL, gb, 0.0 );
				SS_Reset_(SS_X);
				SS_Reset_(SS_Y);
				for( idx= 0; idx< setNumber; idx++ ){
					if( av_sets[idx] ){
						this_set= &AllSets[idx];
						for( j= 0; j< this_set->numPoints; j++ ){
							if( !DiscardedPoint( wi, this_set, j) ){
#ifdef ADVANCED_STATS
								nj= NVAL( this_set, j);
								if( nj<= 0 ){
									nj= 1;
								}
#else
								nj= this_set->NumObs;
#endif
								if( use_transformed ){
									_xval= this_set->xvec[j];
									_yval= this_set->yvec[j];
									_error= (this_set->use_error && !NaN(this_set->errvec[j]))? fabs( this_set->errvec[j] ) : 0;
								}
								else{
									_xval= XVAL(this_set,j);
									_yval= YVAL(this_set,j);
									_error= (this_set->use_error && !NaN(ERROR(this_set,j)))? this_set->error[j] : 0;
								}
								dx= _xval- mx;
								dy= _yval- my;
								if( angle ){
								  double ddx= dx* (cos_angle)- dy* sin_angle,
									ddy= dx* (sin_angle)+ dy* cos_angle;
									_xval= mx+ ddx;
									_yval= my+ ddy;
								}
								if( debugFlag && debugLevel== 1 ){
									fprintf( StdErr, "; point %d.%d (%s,%s)*%s/(1+%s)", idx, j,
										d2str( _xval, "%g", NULL),
										d2str( _yval, "%g", NULL),
										d2str( nj, "%g", NULL),
										d2str( _error, "%g", NULL)
									);
									fflush( StdErr );
								}
								nj/= (1+ _error);
								SS_Add_Data_( SS_X, 1, _xval, 1 );
								SS_Add_Data_( SS_Y, 1, _yval, nj );
							}
						}
					}
				}
				if( debugFlag ){
					fputc( '\n', StdErr );
				}
				fprintf( StdErr, "X-rot scatter: %s\n", SS_sprint_full( NULL, "%g", " +- ", 0, &SS_X ) );
				fprintf( StdErr, "Y-rot scatter: %s\n", SS_sprint_full( NULL, "%g", " +- ", 0, &SS_Y ) );
			}
			mx= SS_Mean_(SS_X);
			my= SS_Mean_(SS_Y);
			sx= 1.5* SS_St_Dev_(SS_X);
			sy= 1.5* SS_St_Dev_(SS_Y);
			fprintf( StdErr, "Scatter of sets within ellipse at (%s,%s), radii (%s,%s), rotation=%s degrees\n",
				d2str( mx, NULL, NULL), d2str( my, NULL, NULL),
				d2str( sx, NULL, NULL), d2str( sy, NULL, NULL),
				d2str( angle1, NULL, NULL)
			);
			SS_Add_Data_( SS_OBS, 1, (SS_X.count+ SS_Y.count)/2.0, 1);
			ap= AddEllipse( &av_set, mx, my, sx, sy, 720, angle1,
				&spot, &Spot, Data, column, filename, sub_div, line_count, NULL, buffer, 
				proc
			);
#if ADVANCED_STATS == 1
			for( i= 0; i< av_set->numPoints; i++ ){
				av_set->N[i]= SS_X.count;
			}
#elif ADVANCED_STATS == 2
			if( av_set->Ncol>= 0 ){
				for( i= 0; i< av_set->numPoints; i++ ){
					av_set->columns[av_set->Ncol][i]= SS_X.count;
				}
			}
#endif
			sprintf( label, "%s (%s,%s,%s,%s,%s\\#xb0\\)\n",
				label,
				d2str( mx, "%.2lf", NULL), d2str( my, "%.2lf", NULL),
				d2str( sx, "%.2lf", NULL), d2str( sy, "%.2lf", NULL),
				d2str( angle1, "%.2lf", NULL)
			);
		}
		else{
#ifndef __GNUC__
			xfree( label );
#endif
			return( 0 );
		}
	}
	else if( XYAveraging ){
		for( i= 0; i< maxPoints; i++ ){
			SS_Reset_(SS_X);
			SS_Reset_(SS_Y);
			  /* Find the first set who has this point:	*/
			for( idx= 0; idx< setNumber; idx++ ){
				if( av_sets[idx] && i< AllSets[idx].numPoints ){
					break;
				}
			}
			this_set= &AllSets[idx];
#ifdef ADVANCED_STATS
					ni= NVAL( this_set, i);
					if( ni<= 0 ){
						ni= 1;
					}
#else
					ni= this_set->NumObs;
#endif
			  /* 20030314: ignore errors (<-0) when !use_error */
			if( use_transformed ){
				txval= this_set->xvec[i];
				tyval= this_set->yvec[i];
/* 				terror= fabs( (this_set->hdyvec[i] - this_set->ldyvec[i])/ 2 );	*/
				terror= (this_set->use_error)? ((NaN(this_set->errvec[i]))? 0 : this_set->errvec[i]) : 0;
			}
			else{
				txval= XVAL(this_set,i);
				tyval= YVAL(this_set,i);
/* 				terror= this_set->error[i];	*/
				terror= (this_set->use_error)? ((NaN(ERROR(this_set,i)))? 0 : ERROR(this_set,i)) : 0;
			}
			if( debugFlag && debugLevel== 1 ){
				fprintf( StdErr, "Point %d, first set %d (%s,%s)*%s/(1+%s)", i, idx,
					d2str( txval, "%g", NULL),
					d2str( tyval, "%g", NULL),
					d2str( ni, "%g", NULL),
					d2str( terror, "%g", NULL)
				);
				fflush( StdErr );
			}
			ni/= (1+ terror);
			  /* Add the first data of this point:	*/
			SS_Add_Data_( SS_X, 1, txval, 1 );
			SS_Add_Data_( SS_Y, 1, tyval, ni );
			  /* Find all next sets who have this point:	*/
			for( j= idx+ 1; j< setNumber; j++ ){
				if( av_sets[j] && i< AllSets[j].numPoints && !DiscardedPoint( wi, &AllSets[j], i) ){
					other_set= &AllSets[j];
#ifdef ADVANCED_STATS
					nj= NVAL( other_set, j);
					if( nj<= 0 ){
						nj= 1;
					}
#else
					nj= other_set->NumObs;
#endif
					if( use_transformed ){
						_xval= other_set->xvec[i];
						_yval= other_set->yvec[i];
/* 						_error= (other_set->use_error)? fabs( (other_set->hdyvec[i] - other_set->ldyvec[i])/ 2 ) : 0;	*/
						_error= (other_set->use_error)? ((NaN(other_set->errvec[i]))? 0 : other_set->errvec[i]) : 0;
					}
					else{
						_xval= XVAL( other_set, i);
						_yval= YVAL( other_set, i);
						_error= (other_set->use_error)? ((NaN(ERROR(other_set,i)))? 0 : ERROR( other_set, i)) : 0;
					}
					if( debugFlag && debugLevel== 1 ){
						fprintf( StdErr, "; set %d (%s,%s)*%s/(1+%s)", j,
							d2str( _xval, "%g", NULL),
							d2str( _yval, "%g", NULL),
							d2str( nj, "%g", NULL),
							d2str( _error, "%g", NULL)
						);
						fflush( StdErr );
					}
					nj/= (1+ _error);
					SS_Add_Data_( SS_X, 1, _xval, 1 );
					SS_Add_Data_( SS_Y, 1, _yval, nj );
				}
			}
			if( SS_X.count && SS_Y.count ){
			  int p= spot;
				Data[0]= SS_Mean_( SS_X );
				Data[1]= SS_Mean_( SS_Y );
				Data[2]= SS_St_Dev_( SS_Y );
				if( Data[2]< 0 ){
					Data[2]= 0;
				}
				if( debugFlag && debugLevel== 1 ){
					fprintf( StdErr, "; av.point #%d (%s,%s)\n", spot,
						d2str( Data[0], "%g", NULL),
						SS_sprint_full( NULL, "%g", " +- ", 0.0, &SS_Y)
					);
					fflush( StdErr );
				}
				AddPoint_discard= False;
				AddPoint( &av_set, &spot, &Spot, 3, Data, column,
					filename, sub_div, line_count, NULL, buffer, proc
				);
				if( spot> p ){
					ap+= 1;
				}
				SS_Add_Data_( SS_OBS, 1, SS_X.count, 1.0 );
#if ADVANCED_STATS == 1
				av_set->N[spot-1]= SS_X.count;
#elif ADVANCED_STATS == 2
				if( av_set->Ncol>= 0 ){
					av_set->columns[av_set->Ncol][spot-1]= SS_X.count;
				}
#endif
			}
		}
	}
	else{
	  YAverage *YData, *data;
	  int dd, XValues= 0;
		if( (YData= (struct YAverage*) calloc( NN, sizeof(struct YAverage)))== NULL ){
			if( wi ){
				xtb_error_box( wi->window, "Can't allocate buffer for averaging\n", "Error" );
			}
			else{
				fprintf( StdErr, "Can't allocate buffer for averaging\n" );
			}
		}
		else{
			for( idx= 0; idx< setNumber; idx++ ){
				if( av_sets[idx] ){
					this_set= &AllSets[idx];
					for( i= 0; i< this_set->numPoints; i++ ){
						if( !DiscardedPoint( wi, this_set, i) ){
							if( use_transformed ){
								txval= this_set->xvec[i];
								tyval= this_set->yvec[i];
/* 								terror= fabs( (this_set->hdyvec[i] - this_set->ldyvec[i])/ 2 );	*/
								terror= (this_set->use_error)? (NaN(this_set->errvec[i])? 0 : this_set->errvec[i]) : 0;
							}
							else{
								txval= XVAL(this_set,i);
								tyval= YVAL(this_set,i);
/* 								terror= this_set->error[i];	*/
								terror= (this_set->use_error)? (NaN(this_set->error[i])? 0 : this_set->error[i]) : 0;
							}
#ifdef ADVANCED_STATS
							ni= NVAL( this_set, i);
							if( ni<= 0 ){
								ni= 1;
							}
#else
							ni= this_set->NumObs;
#endif
							ni/= (1+ terror);
							data= YData;
							dd= 0;
							  /* See if we already have had this X-value.	*/
							while( data->SS_Y.count && data->x!= txval && dd< NN ){
								data++;
								dd+= 1;
							}
							if( dd< NN ){
								if( data->SS_Y.count== 0 ){
									data->idx= i;
									XValues+= 1;
								}
								data->x= txval;
								data->y= tyval;
								data->errr= terror;
								SS_Add_Data_( data->SS_Y, 1, tyval, ni );
							}
						}
					}
				}
			}
			if( add_Interpolations ){
			  double xp, yp, ep, np;
			  int pnr, NP= 0;
				if( debugFlag ){
					fprintf( StdErr, "Average(): interpolating.. ");
					fflush( StdErr );
				}
				data= YData;
				for( i= 0; i< XValues; i++ ){
					for( idx= 0; idx< setNumber; idx++ ){
						if( av_sets[idx] ){
							if( Interpolate( wi, &AllSets[idx], data->x, &pnr, &xp, &yp, &ep, &np, use_transformed)== 1 ){
								data->y= yp;
								data->errr= ep;
								SS_Add_Data_( data->SS_Y, 1, yp, np/(1+ep) );
								NP+= 1;
							}
						}
					}
					data++;
				}
				if( debugFlag ){
					fprintf( StdErr, "%d of %d X values\n", NP, XValues);
				}
			}
			switch( YSort ){
				case 'trsX':
				case 'Xsrt':
				  /* All Y-values have been collected. They should now be sorted
				   \ by X-value.
				   */
					if( debugFlag ){
						fprintf( StdErr, "Average(): qsorting %d unique of %d X values\n",
							XValues, NN
						);
					}
					qsort( YData, XValues, sizeof(struct YAverage), (void*) Xsort_YAverages );
					break;
				case 'trsI':
				case 'Isrt':
				  /* All Y-values have been collected. They will now be sorted
				   \ by index.
				   */
					if( debugFlag ){
						fprintf( StdErr, "Average(): qsorting %d unique of %d indices\n",
							XValues, NN
						);
					}
					qsort( YData, XValues, sizeof(struct YAverage), (void*) Isort_YAverages );
					break;
				case 'rsIX':
				case 'XIsr':
				  /* All Y-values have been collected. They should now be sorted
				   \ by X,index-value
				   */
					if( debugFlag ){
						fprintf( StdErr, "Average(): qsorting %d unique of %d X,index values\n",
							XValues, NN
						);
					}
					qsort( YData, XValues, sizeof(struct YAverage), (void*) XIsort_YAverages );
					break;
				case 'rsXI':
				case 'IXsr':
				  /* All Y-values have been collected. They will now be sorted
				   \ by index,X value.
				   */
					if( debugFlag ){
						fprintf( StdErr, "Average(): qsorting %d unique of %d index,X values\n",
							XValues, NN
						);
					}
					qsort( YData, XValues, sizeof(struct YAverage), (void*) IXsort_YAverages );
					break;
				case 'trsN':
				case 'Nsrt':
				default:
					if( debugFlag ){
						fprintf( StdErr, "Average(): found %d of maximal %d X values\n",
							XValues, NN
						);
					}
					break;
			}
			data= YData;
			dd= 0;
			  /* Now add all points	*/
			{ int N= (debugFlag)? NN : XValues;
			while( dd< N ){
				if( data->SS_Y.count ){
				  int p= spot;
					Data[0]= data->x;
					  /* This is a bit arguable. Points that are not an average of
					   \ multiple data are incorporated as the original data.
					   */
					if( data->SS_Y.count> 1 ){
						Data[1]= SS_Mean_( data->SS_Y );
						Data[2]= SS_St_Dev_( data->SS_Y );
					}
					else{
						Data[1]= data->y;
						Data[2]= data->errr;
					}
					if( Data[2]< 0 ){
						Data[2]= 0;
					}
					if( debugFlag ){
						if( debugLevel== 1 ){
							if( data->SS_Y.count> 1 ){
								fprintf( StdErr, "Y-Average #%d (x,<y>)=(%s,%s)\n", spot,
									d2str( Data[0], "%g", NULL),
									SS_sprint_full( NULL, "%g", " +- ", 0.0, &data->SS_Y)
								);
							}
							else{
								fprintf( StdErr, "Y-Average #%d original point (x,y,e)=(%s,%s,%s)\n", spot,
									d2str( Data[0], "%g", NULL),
									d2str( Data[1], "%g", NULL),
									d2str( Data[2], "%g", NULL)
								);
							}
						}
						if( dd>= XValues ){
							fprintf( StdErr, "\tpoint#%d > #%d of different X values found?!\n",
								dd, XValues
							);
						}
						fflush( StdErr );
					}
					AddPoint_discard= False;
					AddPoint( &av_set, &spot, &Spot, 3, Data, column,
						filename, sub_div, line_count, NULL, buffer, proc
					);
					if( spot> p ){
						ap+= 1;
					}
					SS_Add_Data_( SS_OBS, 1, data->SS_Y.count, 1.0 );
#if ADVANCED_STATS == 1
					av_set->N[spot-1]= data->SS_Y.count;
#elif ADVANCED_STATS == 2
					if( av_set->Ncol>= 0 ){
						av_set->columns[av_set->Ncol][spot-1]= data->SS_Y.count;
					}
#endif
				}
				data++;
				dd+= 1;
			} }
			xfree( YData );
		}
	}
	if( spot> maxitems ){
		maxitems= spot;
		if( wi ){
			realloc_Xsegments();
		}
	}
	if( !av_set->setName || strcmp(av_set->setName, filename)== 0 ){
	  char lb= '\0', *hdr= (ScattEllipse)? "" : (XYAveraging)? "XY-Average of sets " : "Y-Average of sets ";
	  char *tr= (use_transformed)? " (tr.)" : "";
	  int n= -1;
		  /* make sure the automatically generated setname can be read back when
		   \ dumped to an XGraph dump.
		   */
		if( strlen(hdr)+ strlen(label)+ strlen("*LEGEND*")+ 1>= LMAXBUFSIZE ){
			n= LMAXBUFSIZE- strlen(hdr)- strlen("*LEGEND*")- 2;
			if( n>= 0 && n<= strlen(label) ){
				lb= label[n];
				label[n]= '\0';
			}
		}
		av_set->setName= concat( hdr, label, tr, NULL );
		if( lb ){
			  /* We still need the full label, so restore it after copying:	*/
			label[n]= lb;
		}
	}
	av_set->NumObs= (int) (SS_Mean_(SS_OBS)+ 0.5);
	av_set->averagedPoints+= ap;
	if( use_transformed ){
	  /* We already used transformed values. No reason to transform (DATA_PROCESS) 'm again.	*/
		av_set->raw_display= True;
	}
	else{
		av_set->raw_display= False;
	}
	if( debugFlag ){
		fprintf( StdErr, "Averaged sets \"%s\" resulting in %d(%d total) points, NumObs=%s, setName=\"%s\", title=\"%s\"\n",
			label, ap, spot, SS_sprint_full(NULL, "%g", " +- ", 0.0, &SS_OBS ),
			av_set->setName, (av_set->titleText)? av_set->titleText : "<<None>>"
		);
	}
	av_set->average_set= XGstrdup( label );
	av_set->internal_average= False;
#ifndef __GNUC__
	xfree( label );
#endif
	return( spot );
}

int CheckProcessUpdate( LocalWin *wi, int only_drawn, int always, int show_redraw )
{ int i, r= 0;
	if( !setNumber ){
		return(0);
	}
	if( wi->checking== (int) CheckProcessUpdate ){
		return(-1);
	}
	if( always || !RAW_DISPLAY(wi) ){
	  /* 980819: make sure we get "the latest version"	*/
	  DataSet *this_set;
	  Boolean done= False;
	  extern double *disable_SET_PROCESS;
	  int si= wi->silenced,
			ut= wi->use_transformed,
			das= DrawAllSets,
			dbo= DetermineBoundsOnly,
			chk= wi->checking;
		wi->checking= (int) CheckProcessUpdate;
		if( show_redraw ){
			wi->silenced= False;
			DetermineBoundsOnly= False;
		}
		else{
			DetermineBoundsOnly= True;
			if( !always ){
				wi->silenced= True;
			}
		}
		xtb_bt_set( wi->ssht_frame.win, wi->silenced, (char *) 0);
		wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
		wi->use_transformed= False;
		if( !only_drawn ){
			DrawAllSets= True;
		}
		while( wi->redraw || wi->init_pass ){
			TitleMessage( wi, "Redrawing to get up-to-date values\n" );
			if( wi->HO_Dialog && wi->HO_Dialog->win ){
				XStoreName( disp, wi->HO_Dialog->win, "Redrawing to dump up-to-date values");
				XG_XSync( disp, False );
			}
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr,
					"Redrawing window 0x%lx %02d:%02d:%02d to get up-to-date values because wi->redraw==%d or wi->init_pass==%d\n",
					wi, wi->parent_number, wi->pwindow_number, wi->window_number,
					wi->redraw, wi->init_pass
				);
			}
			DrawAllSets= True;
			RedrawNow( wi );
			r= 1;
			TitleMessage( wi, NULL );
		}
		for( i= 0; i< setNumber && !done; i++ ){
			this_set= &AllSets[i];
			if( !only_drawn || draw_set( wi, i) ){
				if( this_set->init_pass ||
					(this_set->last_processed_wi!= wi &&
						(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len ||
							(!*disable_SET_PROCESS && this_set->process.set_process_len)
						)
					)
				){
					TitleMessage( wi, "Redrawing to get up-to-date values\n" );
					if( wi->HO_Dialog && wi->HO_Dialog->win ){
						XStoreName( disp, wi->HO_Dialog->win, "Redrawing to get up-to-date values");
						XG_XSync( disp, False );
					}
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr,
							"Redrawing window 0x%lx %02d:%02d:%02d to get up-to-date values because set #%d is not up-to-date\n",
							wi, wi->parent_number, wi->pwindow_number, wi->window_number, i
						);
					}
					if( !only_drawn ){
						DrawAllSets= True;
						RedrawNow(wi);
					}
					else{
						RedrawNow( wi );
					}
					TitleMessage( wi, NULL );
					done= True;
					r= 1;
				}
			}
		}
		wi->use_transformed= ut;
		wi->silenced= si;
		xtb_bt_set( wi->ssht_frame.win, wi->silenced, (char *) 0);
		wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
		DrawAllSets= das;
		DetermineBoundsOnly= dbo;
		wi->checking= chk;
	}
	return(r);
}
