#include "config.h"
IDENTIFY( "matherr error reporting routine, and d2str" );

/* Don't include xgraph.h in this file!	*/
#include "stddef.h"
#include "stdlib.h"
#include "stdio.h"
#include <math.h>
#include <float.h>
#include <signal.h>

#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
#	define USE_SSE2
#	include <xmmintrin.h>
#	include <emmintrin.h>
#	include "AppleVecLib.h"
#endif

#include <string.h>
#ifndef	_APOLLO_SOURCE
#	include <strings.h>
#else
	extern char *index( const char *string, const int tok);
#endif

#include "copyright.h"

#undef _IDENTIFY
#include "Macros.h"
#define streq(a,b)	!strcmp(a,b)
#define strneq(a,b,n)	!strncmp(a,b,n)

#define M_2PI     6.283185307179586231995926937088370323181152344

#define radians(a)		((a)/57.295779512)
#define onexit(code,name,cond,verb)	atexit((code))
extern FILE *StdErr;
#define cx_stderr	StdErr

extern int debugFlag;

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#	include "arrayvops.h"
#elif defined(__SSE2__) || defined(__SSE3__)
#	include "arrayvops.h"
#endif

int XGstrcmp(const char *s1, const char *s2)
{
	if( !s1 ){
		return( (s2)? -1 : 0 );
	}
	else if( !s2 ){
		return( 1 );
	}
	else{
		return( strcmp( s1, s2 ) );
	}
}

int XGstrncmp(const  char *s1, const     char *s2, size_t n)
{
	if( !s1 ){
		return( (s2)? -1 : 0 );
	}
	else if( !s2 ){
		return( 1 );
	}
	else{
		return( strncmp( s1, s2, n ) );
	}
}

int XGstrcasecmp(const char    *s1, const char     *s2)
{
	if( !s1 ){
		return( (s2)? -1 : 0 );
	}
	else if( !s2 ){
		return( 1 );
	}
	else{
		return( strcasecmp( s1, s2 ) );
	}
}

int XGstrncasecmp(const char *s1, const char *s2, size_t n)
{
	if( !s1 ){
		return( (s2)? -1 : 0 );
	}
	else if( !s2 ){
		return( 1 );
	}
	else{
		return( strncasecmp( s1, s2, n ) );
	}
}

#define strcmp XGstrcmp
#define strncmp XGstrncmp
#define strcasecmp XGstrcasecmp
#define strncasecmp XGstrncasecmp

#define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}

double dcmp( double b, double a, double prec)
{ double a_low, a_high;

	if( prec>= 0.0 ){
		a_low= (1.0-prec)* a;
		a_high= (1.0+prec)* a;
		if( a_low> a_high ){
			SWAP( a_low, a_high, double );
		}
		if( b< a_low ){
			return( b- a_low );
		}
		else if( b> a_high ){
			return( b- a_high );
		}
		else{
			return( 0.0 );
		}
	}
	else{
	  /* prec==-1 is meant to check if b equals
	   \ a allowing for the machine imprecision
	   \ DBL_EPSILON. Let's hope that a_low and
	   \ a_high are calculated correctly!
	   */
		prec*= -1.0;
		b-= a;
		if( b< - prec * DBL_EPSILON || b> prec * DBL_EPSILON ){
			return( b - prec * DBL_EPSILON );
		}
		else{
			return( 0.0 );
		}
	}
}

double dcmp2( double b, double a, double prec)
{ double a_low, a_high;
  double b_low, b_high;

	if( prec>= 0.0 ){
		a_low= (1.0-prec)* a;
		a_high= (1.0+prec)* a;
		if( a_low> a_high ){
			SWAP( a_low, a_high, double );
		}
		b_low= (1.0-prec)* b;
		b_high= (1.0+prec)* b;
		if( b_low> b_high ){
			SWAP( b_low, b_high, double );
		}
		if( b_low< a_low ){
			return( b_low- a_low );
		}
		else if( b_high> a_high ){
			return( b_high- a_high );
		}
		else{
			return( 0.0 );
		}
	}
	else{
	  /* prec==-1 is meant to check if b equals
	   \ a allowing for the machine imprecision
	   \ DBL_EPSILON. Let's hope that a_low and
	   \ a_high are calculated correctly!
	   */
		prec*= -1.0;
		b-= a;
		if( b< - prec * DBL_EPSILON || b> prec * DBL_EPSILON ){
			return( b - prec * DBL_EPSILON );
		}
		else{
			return( 0.0 );
		}
	}
}

char greek_inf[4]= { '\\', 0xa5, '\\', '\0'},
	greek_min_inf[5]= { '-', '\\', 0xa5, '\\', '\0' };

double Entier(double x)
{
	if( x>0 ){
		x = ssfloor(x);
	}
	else{
		x = ssceil(x);
	}
	return x;
}

/* Searching for possible (shorter printing) fractions does take some time, so some of
 \ the less common denominators are commented out. A negative value is a scale-factor to apply
 \ to the other, positive denominators to be tested.
 */
int _d2str_factors_ext[]=
			{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
				-3, -6, -10, -11, -33, -66, -100, -111, -333, -666, -1000, -1111, -3333, -6666, 0 },
	_d2str_factors[]=
			{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 200, 1000, 2000, 0 },
	*d2str_factors= _d2str_factors;

/* Given a real 0<=d<1, find in the array DF an element df > 0 such that 
 \ d * df * scale is integer.
 */
static int find_d2str_factor( double d, int *DF, int scale)
{  int *df= DF;
   double t;
   int n;
	while( d>= 0 && d< 1 && df && *df ){
		if( *df> 0 ){
			n= scale* *df;
			t= d* n;
			if( fabs(floor(t) - t)<= DBL_EPSILON && t>= 1 ){
				return( n );
			}
		}
		df++;
	}
	return( 0 );
}

#define D2STR_BUFS	100

int use_greek_inf= 0, Allow_Fractions= 1;
int d2str_printhex= False;
int PrintNaNCode= False;

/* The internal buffers should be long enough to print the longest doubles.
 \ Since %lf prints 1e308 as a 1 with 308 0s and then some, we must take
 \ this possibility into account... DBL_MAX_10_EXP+12=320 on most platforms.
 */
char *d2str( double d, const char *Format , char *buf )
#ifndef OLD_D2STR
{  static char internal_buf[D2STR_BUFS][DBL_MAX_10_EXP+12], *template= NULL, nan_buf[9*sizeof(long)];
   static int buf_nr= 0, temp_len= 0;
   int Sign, plen= 0, mlen, frac_len= 0;
   Boolean nan= False, ext_frac= False;
   char *format= (char *) Format, *buf_arg= buf, frac_buf[DBL_MAX_10_EXP+12], *slash;
	if( !buf ){
	 /* use one of the internal buffers, cycling through them.	*/
		buf= internal_buf[buf_nr];
		buf_nr= (buf_nr+1) % D2STR_BUFS;
		mlen= DBL_MAX_10_EXP+12;
	}
	else{
		mlen= -1;
	}


	nan_buf[0]= '\0';
	if( NaN(d) ){
		if( PrintNaNCode ){
		  unsigned long nan_val= NaN(d);
			sprintf( nan_buf, "%sNaN%lu",
				(I3Ed(d)->s.s)? "-" : "",
				(nan_val & 0x000ff000) >> 12
			);
		}
		else{
			strcpy( nan_buf, "NaN" );
		}
		nan= True;
		format= NULL;
	}
	else if( (Sign=Inf(d)) ){
		if( Sign> 0 ){
			strcpy( nan_buf, (use_greek_inf)? GREEK_INF : "Inf");
		}
		else if( Sign< 0){
			strcpy( nan_buf, (use_greek_inf)? GREEK_MIN_INF : "-Inf");
		}
		nan= True;
		format= NULL;
	}
	else if( d2str_printhex || (format && strcmp(format, "%dhex")== 0) ){
	  IEEEfp *ie= (IEEEfp*) &d;
	  long test;
		buf[0]= '\0';
		if( d>= -MAXLONG-1 && d<= MAXLONG ){
			if( (double)(test= (long) d)== d ){
				sprintf( buf, "%ld", test );
			}
		}
		  /* 20030405: */
		else if( floor(d)== d ){
			sprintf( buf, "%g", d );
		}
		if( !*buf ){
			sprintf( buf, "0x%lx:%lx", ie->l.high, ie->l.low );
		}
		return(buf);
	}

	{ char *c;
	  int len;
		if( !format ){
			format= "%g";
		}
		else if( format[0]== '%' && format[1]== '!' ){
		  char *c= &format[2], *sform;
		  int pedantic, signif, fallback= False;
			if( *c== '!' ){
				c++;
				pedantic= True;
			}
			else{
				pedantic= False;
			}
/* 			if( nan ){	*/
/* 				format= "%g";	*/
/* 			}	*/
/* 			else	*/
			if( sscanf( c, "%d", &signif) && signif> 0 ){
			  double pd= fabs(d);
			  double order;
			  int iorder;
				if( pd> 0 || pedantic ){
					if( pd ){
						order= log10(pd);
/* 						iorder= (int) ((order< 0)? ssceil(order) : ssfloor(order));	*/
						iorder= (int) ssfloor(order);
					}
					else{
						iorder= 0;
					}
					if( iorder< -3 || iorder>= signif ){
						sform= "%.*e";
						iorder= 0;
					}
					else{
						if( !pedantic && fabs(floor(d)- d)<= DBL_EPSILON ){
						  /* Ignore the precision argument:	*/
							sform= "%g";
							fallback= True;
						}
						else{
						  /* make sure we print all the requested digits
						   \ (%g won't print trailing zeroes).
						   \ Also add a small, significance-dependent value
						   \ to ensure that e.g. with '%!4g', 10.555 prints
						   \ as 10.56 (i.e. that the last 5 rounds up the
						   \ preceding digit.
						   */
							sform= "%.*f";
							if( d> 0 ){
								d+= pow( 10, iorder- signif- 0.0);
							}
							else{
								d-= pow( 10, iorder- signif- 0.0);
							}
						}
					}
				}
				else{
					iorder= 0;
					  /* sform="%2$g" should cause the 2nd argument after the format string 
					   \ to be printed with "%g", but not all machines appear to support this format.
					   */
					sform= "%g";
					fallback= True;
				}
				if( fallback ){
					sprintf( buf, sform, d );
				}
				else{
					sprintf( buf, sform, signif- iorder- 1, d );
				}
				return( buf );
			}
		}
		if( (len= strlen(format)+1) > temp_len ){
		  char *c= template;
			if( !(template= (char*) calloc( len, 1)) ){
				fprintf( stderr, "matherr::d2str(%g,\"%s\"): can't allocate buffer (%s)\n",
					d, format, serror()
				);
				fflush( stderr );
			}
			else{
				if( c ){ free( c );}
				temp_len= len;
				strcpy( template, format);
				format= template;
			}
		}
		else{
			strcpy( template, format);
			format= template;
		}
		if( format== template ){
		  /* in this case we may alter the format-string
		   \ this is done if we are printing a nan. In that
		   \ case, the floating printing specification (fps) is
		   \ substituted for a similar '%s'
		   \ a fps has the form "%[<length>.<res>][l]{fFeEgG}";
		   \ the result the form "%[<length>.<res>]s".
		   */
			do{
				c= index( format, '%');
				if( c && c[1]== '%' ){
					c+= 2;
				}
			} while( c && *c && *c!= '%' );
			if( nan && c && *c== '%' ){
			  char *d= &c[1];
				strcpy( buf, nan_buf );
				while( *d && !( index( "gGfFeE", *d) || (index( "lL", *d) && index( "gGfFeE", d[1]))) ){
					d++;
				}
				if( *d ){
					c= d;
					if( *d== 'l' || *d== 'L' ){
						d+= 2;
					}
					else{
						d++;
					}
					*c++= 's';
					while( *d ){
						*c++= *d++;
					}
					*c= '\0';
				}
				else{
					return( buf );
				}
			}
			else if( nan ){
			  /* screwed up	*/
				return( buf );
			}
		}
		else{
			nan= False;
		}
		slash= index( format, '/');
		if( !strcmp( format, "%g'%g/%g") || !slash){
		  /* we handle fractions as a/b (when -1< d <1), or
		   \ else as p'a/b, with |p|> 1, when either it has been
		   \ specifically requested, or when the format does not specify
		   \ a fraction, in which case we only print this way when it does
		   \ not take more space, and Allow_Fractions is true.
		   */
			ext_frac= True;
		}
		if( !nan ){
		  double D= ABS(d), p= (ext_frac && D> 1)? Entier(d) : 0, P= ABS(p), dd= D-P;
			  /* dd is the non-integer part of d	*/
			if( dd && (Allow_Fractions || slash) ){
			  /* Find a denominator f such that d=n/f with n integer and f>0
			   \ d2str_factors is an integer array containing the
			   \ denominators f (positive elements) and scale-factors
			   \ (negative elements) to apply to the denominators.
			   */
			  int *df, f;
			  int scale= 1;
			      /* We need the absolute value of the printable, and possibly its
				  \ integer part (when we are allowed to print as p'a/b)
				  */
				if( d2str_factors== NULL || d2str_factors== _d2str_factors || d2str_factors== _d2str_factors_ext ){
				  /* if we're using 1 of the 2 default factor-lists, select the extended one
				   \ if a fraction was explicitely requested, and the shorter one otherwise.
				   */
					d2str_factors= (slash)? _d2str_factors_ext : _d2str_factors;
				}
				df= d2str_factors;
				if( !(f= find_d2str_factor( dd, d2str_factors, scale )) ){
					  /* Nothing found, check for scale-factors. An array {1,2,3,-10}
					   \ with thus check {1,2,3,10,20,30} as possible denominators
					   */
					  /* try it with a simple factor	*/
					while( df && *df && !f ){
						if( *df< 0 ){
							scale= - *df;
							f= find_d2str_factor( dd, d2str_factors, scale);
						}
						df++;
					}
					df= d2str_factors;
					while( df && *df && !f ){
					  int *df2= d2str_factors;
						if( *df> 1 ){
							while( !f && df2 && *df2 ){
								if( *df2< 0 ){
								  double sc= pow( - *df2, *df );
									if( sc< MAXINT ){
										  /* try it with power *df of factor *df2 ...	*/
										f= find_d2str_factor( dd, d2str_factors, (int) sc );
									}
								}
								df2++;
							}
						}
						df++;
					}
				}
				if( f && f!= 1 ){
				  /* print as (f*d)/f	*/
					if( slash ){
						if( ext_frac ){
							if( p ){
								frac_len= sprintf( frac_buf, format, p, f* dd, (double) f );
							}
							else{
							  /* ext_frac is only true when format=="%g'%g/%g", so we can safely
							   \ make the following simplification:
							   */
								frac_len= sprintf( frac_buf, "%g/%g", f* d, (double) f );
							}
						}
						else{
							frac_len= sprintf( frac_buf, format, f* d, (double) f );
						}
					}
					else{
						if( p ){
							frac_len= sprintf( frac_buf, "%g'%g/%g", p, f* dd, (double) f );
						}
						else{
							frac_len= sprintf( frac_buf, "%g/%g", f* d, (double) f );
						}
					}
				}
			}
		}
		  /* Now make the output	*/
		if( !nan && strneq( format, "1/", 2) ){
		  /* Maybe one day this will be changed to
		   \ handle different cases (e.g. x/%g) where
		   \ x any number
		   */
		  /* Just see the following else if..!	*/
			if( d!= 0 ){
				plen= sprintf( buf, format, 1.0/d);
			}
			else{
				plen= sprintf( buf, &format[2], d);
			}
		}
		else if( (c= slash) ){
			if( !nan ){
				if( frac_len ){
					plen= frac_len;
					strcpy( buf, frac_buf );
				}
				else{
					if( ext_frac ){
					  /* asked for %g'%g/%g, but we didn't find a fraction. So:	*/
						plen= sprintf( buf, "%g", d);
					}
					else{
						*c= '\0';
						plen= sprintf( buf, format, d);
						*c= '/';
					}
				}
			}
			else{
				*c= '\0';
				plen= sprintf( buf, format, nan_buf, 1);
				*c= '/';
			}
		}
		else{
			if( !nan ){
				plen= sprintf( buf, format, d);
			}
			else if( format== template ){
			  /* print the nan_buf according to format. Just in case
			   \ the originale Format was something like '%g/%g', an
			   \ extra argument of 1.0 is added. Now Inf will print like
			   \ Inf/1 .
			   \ 960325: allow printing of <nan> as (possible fraction) to
			   \ prevent Inf/1.
			   */
				plen= sprintf( buf, format, nan_buf, 1.0 );
			}
			if( Allow_Fractions && (frac_len> 0 && frac_len<= plen) ){
				if( debugFlag ){
					fprintf( StdErr, "d2str(): %s == %s\n", buf, frac_buf );
				}
				strcpy( buf, frac_buf );
				plen= frac_len;
			}
		}
	}
	if( plen<= 0 && (Format && *Format) ){
		fprintf( cx_stderr, "d2str(%g,\"%s\",0x%lx)=> \"%s\": no output generated: report to sysop (will return %g)\n",
			d, (Format)? Format : "<NULL>", buf_arg, buf, d
		);
		plen= sprintf( buf, "%g", d );
		fflush( cx_stderr );
	}
	if( mlen> 0 && plen> mlen ){
		fprintf( cx_stderr, "d2str(%g): wrote %d bytes in internal buffer of %d\n",
			d, plen, mlen
		);
		fflush( cx_stderr );
	}
	return( buf );
}
#else
{  static char internal_buf[D2STR_BUFS][DBL_MAX_10_EXP+12], *template= NULL, nan_buf[9*sizeof(long)];
   static int buf_nr= 0, temp_len= 0;
   int Sign, plen= -1, mlen;
   Boolean nan= False;
   char *format= Format;
	if( !buf ){
	 /* use one of the internal buffers, cycling through them.	*/
		buf= internal_buf[buf_nr];
		buf_nr= (buf_nr+1) % D2STR_BUFS;
		mlen= DBL_MAX_10_EXP+12;
	}
	else{
		mlen= -1;
	}
	nan_buf[0]= '\0';
	if( NaN(d) ){
	  unsigned long nan_val= NaN(d);
		sprintf( nan_buf, "NaN%lu", (nan_val & 0x000ff000) >> 12);
		nan= True;
	}
	else if( (Sign=Inf(d)) ){
		if( Sign> 0 ){
			strcpy( nan_buf, (use_greek_inf)? GREEK_INF : "Inf");
		}
		else if( Sign< 0){
			strcpy( nan_buf, (use_greek_inf)? GREEK_MIN_INF : "-Inf");
		}
		nan= True;
	}
	/* else	*/{
	  char *c;
	  int len;
		if( !format ){
			format= "%g";
		}
		if( (len= strlen(format)+1) > temp_len ){
		  char *c= template;
			if( !(template= (char*) calloc( len, 1)) ){
				fprintf( stderr, "matherr::d2str(%g,\"%s\"): can't allocate buffer (%s)\n",
					d, format, serror()
				);
				fflush( stderr );
			}
			else{
				if( c ){ free( c );}
				temp_len= len;
				strcpy( template, format);
				format= template;
			}
		}
		else{
			strcpy( template, format);
			format= template;
		}
		if( format== template ){
		  /* in this case we may alter the format-string
		   \ this is done if we are printing a nan. In that
		   \ case, the floating printing specification (fps) is
		   \ substituted for a similar '%s'
		   \ a fps has the form "%[<length>.<res>][l]{fFeEgG}";
		   \ the result the form "%[<length>.<res>]s".
		   */
			do{
				c= index( format, '%');
				if( c[1]== '%' ){
					c+= 2;
				}
			} while( *c && *c!= '%' );
			if( nan && *c== '%' ){
			  char *d= &c[1];
				strcpy( buf, nan_buf );
				while( *d && !index( "gGfFeE", *d) ){
					d++;
					if( *d== 'l' || *d== 'L' && index( "gGfFeE", d[1] ) ){
						d++;
					}
				}
				if( *d ){
					c= d++;
					*c++= 's';
					while( *d ){
						*c++= *d++;
					}
					*c= '\0';
				}
				else{
					return( buf );
				}
			}
			else if( nan ){
			  /* screwed up	*/
				return( buf );
			}
			plen= strlen(buf);
		}
		else{
			nan= False;
		}
		if( !nan && strneq( format, "1/", 2) ){
		  /* Maybe one day this will be changed to
		   \ handle different cases (e.g. x/%g) where
		   \ x any number
		   */
			if( d!= 0 ){
				plen= sprintf( buf, format, 1.0/d);
			}
			else{
				plen= sprintf( buf, &format[2], d);
			}
		}
		else if( !nan && (c= index( format, '/')) ){
		  int *df= d2str_factors, f;
		  int scale= 1;
			if( !(f= find_d2str_factor( d, d2str_factors, scale )) ){
				while( df && *df && !f ){
					if( *df< 0 ){
						scale= - *df;
						f= find_d2str_factor( d, d2str_factors, scale);
					}
					df++;
				}
			}
			if( f && f!= 1 ){
			  /* print as (f*d)/f	*/
				plen= sprintf( buf, format, f* d, (double) f );
			}
			else{
				*c= '\0';
				plen= sprintf( buf, format, d);
				*c= '/';
			}
		}
		else{
			if( !nan ){
				plen= sprintf( buf, format, d);
			}
			else if( format== template ){
			  /* print the nan_buf according to format. Just in case
			   \ the originale Format was something like '%g/%g', an
			   \ extra argument of 1.0 is added. Now Inf will print like
			   \ Inf/1 .
			   */
				plen= sprintf( buf, format, nan_buf, 1.0 );
			}
		}
	}
	if( mlen> 0 ){
	  /* means we can do a check for overwriting the (internal) buffer	*/
#ifdef _APOLLO_SOURCE
		plen= strlen(buf);
#else
		if( plen== -1 || plen> mlen ){
		  /* plen not determined, or maybe wrong... (double check)	*/
			plen= strlen(buf);
		}
#endif
		if( plen> mlen ){
			fprintf( StdErr, "d2str(%g)=\"%s\": wrote %d bytes in internal buffer of %d\n",
				d, buf, plen, mlen
			);
			fflush( StdErr );
		}
	}
	return( buf );
}
#endif

int matherr_off= 0, matherr_verbose= 0;
unsigned long matherr_calls= 0, matherr_called= 0;

char *matherrMark= NULL;
static char *mMark= NULL;

char *matherr_mark( char *mark )
{  char *c= matherrMark;
	mMark= matherrMark= mark;
	return( c );
}

char *matherror[]={
	"NO EXCEPTION!"
	,"DOMAIN (argument out of range)"
	,"SING (argument singularity)"
	,"OVERFLOW (overflow range)"
	,"UNDERFLOW (underflow range)"
	,"TLOSS (total loss of significance)"
	,"PLOSS (partial loss of significance)"
};

static unsigned long pow_0_0= 0, pow_0_neg= 0, pow_neg_odd= 0, pow_neg_nodd= 0,
	pow_inf_small= 0, pow_inf= 0,
	sqrt_nan= 0, acos_dom= 0, asin_dom= 0, atan2_0_0= 0,
	exp_over= 0, pow_neg_over= 0, pow_pos_over= 0, pow_0_over= 0,
	exp_under= 0, pow_under= 0, log_= 0, log_0= 0,
	gon_mod= 0, pow_x_inf= 0, pow_x__inf= 0, sin_dom= 0, cos_dom= 0, fmod_dom= 0;

char *matherr_msg= NULL;
Boolean matherr_message= True;

void matherr_report()
{ unsigned long others;

	if( !matherr_calls)
		return;
	fprintf( cx_stderr, "\nmatherr reports %lu calls, of which:\n", matherr_calls);
	if( pow_x__inf)
		fprintf( cx_stderr, "pow(x,-Inf)          : %lu (returns 0)\n", pow_x__inf );
	if( pow_x_inf)
		fprintf( cx_stderr, "pow(x,Inf)           : %lu (returns Inf)\n", pow_x_inf );
	if( pow_0_0)
		fprintf( cx_stderr, "pow(0,0)             : %lu (returns 0)\n", pow_0_0);
	if( pow_0_neg)
		fprintf( cx_stderr, "pow(0,<0)            : %lu (returns Inf)\n", pow_0_neg );
	if( pow_neg_odd)
		fprintf( cx_stderr, "pow(<0,1/odd)        : %lu\n", pow_neg_odd );
	if( pow_neg_nodd)
		fprintf( cx_stderr, "pow(<0,1/even)       : %lu\n", pow_neg_nodd );
	if( pow_inf)
		fprintf( cx_stderr, "pow(+/-Inf,<-1 or >1): %lu (returns +/-Inf)\n", pow_inf );
	if( pow_inf_small)
		fprintf( cx_stderr, "pow(+/-Inf,<-1,1>)   : %lu (returns 0; 1 for pow(Inf,0))\n", pow_inf_small );
	if( sqrt_nan)
		fprintf( cx_stderr, "sqrt(NaN)            : %lu (returns NaN)\n", sqrt_nan );
	if( fmod_dom)
		fprintf( cx_stderr, "fmod domain          : %lu (returns 0)\n", fmod_dom);
	if( cos_dom)
		fprintf( cx_stderr, "cos domain           : %lu (returns 0)\n", cos_dom);
	if( sin_dom)
		fprintf( cx_stderr, "sin domain           : %lu (returns 0)\n", sin_dom);
	if( acos_dom)
		fprintf( cx_stderr, "acos(<-1) or acos(>1): %lu (returns +/- Pi)\n", acos_dom );
	if( asin_dom)
		fprintf( cx_stderr, "asin(<-1) or asin(>1): %lu (returns +/- Pi/2)\n", asin_dom );
	if( atan2_0_0)
		fprintf( cx_stderr, "atan2(0,0)           : %lu (returns 0)\n", atan2_0_0 );
	if( exp_over)
		fprintf( cx_stderr, "exp() overflows      : %lu (returns Inf)\n", exp_over);
	if( pow_neg_over)
		fprintf( cx_stderr, "pow(<0,too large)    : %lu (returns +/- Inf)\n", pow_neg_over);
	if( pow_pos_over)
		fprintf( cx_stderr, "pow(>0,too large)    : %lu (returns Inf)\n", pow_pos_over);
	if( pow_0_over)
		fprintf( cx_stderr, "pow(0,too large)     : %lu (returns 0)\n", pow_0_over);
	if( exp_under)
		fprintf( cx_stderr, "exp(too small)       : %lu (returns 0)\n", exp_under);
	if( pow_under)
		fprintf( cx_stderr, "pow(too small,y)     : %lu (returns 0)\n", pow_under);
	if( log_)
		fprintf( cx_stderr, "log(<0),log(NaN)     : %lu (returns -Inf/NaN)\n", log_);
	if( log_0)
		fprintf( cx_stderr, "log(0.0),log10(0.0)  : %lu (returns -Inf)\n", log_0);
	if( gon_mod)
		fprintf( cx_stderr, "gonio routine called with value too large: %lu (arguments moduloed 2PI)\n", gon_mod);
	others= pow_0_0+ pow_0_neg+ pow_neg_odd+ pow_neg_nodd+ sqrt_nan+ acos_dom+ asin_dom+ atan2_0_0+
			exp_over+ pow_neg_over+ pow_pos_over+ pow_0_over+ exp_under+ pow_under+ log_ + log_0+
			gon_mod+ pow_inf+ pow_inf_small+ pow_x_inf+ pow_x__inf+ sin_dom+ cos_dom+ fmod_dom;
	if( matherr_calls- others)
		fprintf( cx_stderr, "Others               : %lu\n\n", matherr_calls- others );
	if( matherr_msg /* && matherr_message */ ){
		if( matherr_msg[0]== '\0' && matherr_msg[1]!= '\0' )
			matherr_msg[0]= 'm';
		fprintf( cx_stderr, "Last %s message: '%s'\n", (matherr_message)? "printed" : "internal", matherr_msg);
	}
	fflush( cx_stderr);
}

/* matherr_msg will contain the message of the last action undertaken
 * by matherr. It is printed to cx_stderr when verbose==True. In
 * some more common/trivial cases, verbose is set to 0; in a few even
 * more trivial cases matherr_msg[0] to 0. This makes it possible to
 * (when matherr_msg= extra_logmsg) have the common/trivial cases logged
 * to the logfile (Addlog checks for strlen(extra_logmsg)
 * ; the even more trivial cases can be detected with extra_logmsg[1]!= 0
 * When matherr screws up (is called again while handling an error) it
 * notifies this on cx_stderr, and exits immediately.
 */

int matherr(struct exception *x)
{ int r=0, verbose= 1, args;
  static int no_msg= 0;
  static int _args= 2;
  char Arg1[80], Arg2[80], Ret[80], nan1[80], nan2[80], nanret[80];
  unsigned long *_a1, *_a2, *_aret;
  static int screwed, called= 0, active= 0, special= 0;
  static char msg[256], ret_msg[128];

	matherr_calls++;
	matherr_called++;

	if( matherr_off){
		return(0);
	}

	if( !called){
		onexit( matherr_report, "matherr_report", 0, 1);
		called= 1;
	}

	if( !matherr_msg)
		matherr_msg= msg;

	if( NaN(x->arg1)){
		d2str( x->arg1, "%le", Arg1);
		_a1= (unsigned long*) &x->arg1;
		sprintf( nan1, "[0x%lx:%lx]", _a1[0], _a1[1]);
	}
	else{
		*nan1= '\0';
		d2str( x->arg1, "%le", Arg1);
/* 
		if( Inf(x->arg1))
			sprintf( Arg1, "%cInf", (Inf(x->arg1)>0) ? '+': '-' );
		else
			sprintf( Arg1, "%le", x->arg1);
 */
	}

	if( NaN(x->arg2)){
		d2str( x->arg2, "%le", Arg2);
		_a2= (unsigned long*) &x->arg2;
		sprintf( nan2, "[0x%lx:%lx]", _a2[0], _a2[1]);
	}
	else{
		*nan2= '\0';
		d2str( x->arg2, "%le", Arg2);
/* 
		if( Inf(x->arg2))
			sprintf( Arg2, "%cInf", (Inf(x->arg2)>0) ? '+': '-' );
		else
			sprintf( Arg2, "%le", x->arg2);
 */
	}

	if( NaN(x->retval)){
		d2str( x->retval, "%le", Ret);
		_aret= (unsigned long*) &x->retval;
		sprintf( nanret, "[0x%lx:%lx]", _aret[0], _aret[1]);
	}
	else{
		*nanret= '\0';
		d2str( x->retval, "%le", Ret);
/* 
		if( Inf(x->retval))
			sprintf( Ret, "%cInf", (Inf(x->retval)>0) ? '+': '-' );
		else
			sprintf( Ret, "%le", x->retval);
 */
	}
	sprintf( ret_msg, " [will return %s]", Ret);

	if( active && /* ! */ special ){
		if( matherr_msg[0]== 0 && matherr_msg[1])
			matherr_msg[0]= ' ';
		if( _args == 2 ){
			fprintf( cx_stderr, "%s \\\n %s(%s%s,%s%s)==%s%s: %s\n\tscrewed up - returning %s\n",
				matherr_msg, x->name, Arg1, nan1, Arg2, nan2, Ret, nanret, matherror[x->type], Ret
			);
		}
		else{
			fprintf( cx_stderr, "%s =>\n %s(%s%s)==%s%s: %s\n\tscrewed up - returning %s\n",
				matherr_msg, x->name, Arg1, nan1, Ret, nanret, matherror[x->type], Ret
			);
		}
		fflush( cx_stderr);
		active-= 1;
		screwed= 1;
/* 		return( (int) x->retval);	*/
		return( 1 );
	}
	active+= 1;
	screwed= 0;
	args= 1;
	switch( x->type){
		case DOMAIN:
			if( !strcmp( x->name, "sqrt") ){
				if( NaN( x->arg1) ){
					x->retval= x->arg1;
					strcpy( ret_msg, " [will return NaN]");
					verbose= 0;
					sqrt_nan++;
				}
/* 				else if( x->arg1< 0.0)
					x->retval= sqrt(-1.0 * x->arg1);	*/
				sprintf( ret_msg, " [will return %s]", d2str( x->retval, "%.10le", NULL) );
				r= 1;
			}
			else if( !strcmp( x->name, "asin") ){
				if( NaN(x->arg1) ){
					x->retval= 1.0;
				}
				else{
					x->retval= ( (x->arg1< 0.0)?-1 : 1) * M_PI_2;
				}
				asin_dom++;
				sprintf( ret_msg, " [will return %s]", d2str( x->retval, "%.10le", NULL) );
				r= 1;
			}
			else if( !strcmp( x->name, "acos") ){
				if( NaN(x->arg1) )
					x->retval= 1.0;
				else
					x->retval= (x->arg1<= -1.0)? M_PI : 0.0;
				acos_dom++;
				sprintf( ret_msg, " [will return %s]", d2str( x->retval, "%.10le", NULL) );
				r= 1;
			}
			else if( !strcmp( x->name, "atan2") ){
				atan2_0_0++;
				x->retval= 0.0;
				strcpy( ret_msg, " [will return 0.0]");
				if( !special)
					no_msg= 1;
				else
					no_msg= 0;
				verbose= 0;
				r= 1;
				args= 2;
			}
			else if( !strcmp( x->name, "fmod") ){
				fmod_dom++;
				x->retval= 0.0;
				strcpy( ret_msg, " [will return 0.0]");
				if( !special)
					no_msg= 1;
				else
					no_msg= 0;
				verbose= 0;
				r= 1;
				args= 2;
			}
			else if( !strncmp( x->name, "log", 3) ){
				if( x->arg1< 0 ){
					strcpy( ret_msg, " [will return -Inf]");
					set_Inf( x->retval, -1 );
					if( !special ){
						no_msg= 1;
					}
					else{
						no_msg= 0;
					}
					verbose= 0;
				}
				r= 1;
				log_++;
			}
			else if( !strcmp( x->name, "pow") ){
			  int infsign;
				args= 2;
				if( (infsign= Inf(x->arg2)) ){
					set_Inf(x->retval, infsign);
					if( infsign> 0){
						pow_x_inf++;
						strcpy( ret_msg, " [will return Inf]");
					}
					else{
						pow_x__inf++;
						strcpy( ret_msg, " [will return 0]");
					}
					r= 1;
				}
				else if( (infsign= Inf(x->arg1)) ){
					if( fabs( x->arg2) < 1.0){
						pow_inf_small++;
						x->retval= (x->arg2==0)? 1.0 : 0.0;
						sprintf( ret_msg, " [will return %s]", d2str( x->retval, "%.10le", NULL) );
						r= 1;
						if( !x->arg2){
							if( !special)
								no_msg= 1;
							else
								no_msg= 0;
						}
						verbose= 0;
					}
					else{
						pow_inf++;
						set_Inf( x->retval, infsign);
						if( infsign>0 )
							strcpy( ret_msg, " [will return Inf]");
						else
							strcpy( ret_msg, " [will return -Inf]");
						r= 1;
						verbose= 0;
					}
				}
				else if( x->arg1== 0.0 ){
					if( x->arg2== 0.0){
						x->retval= 0.0;
						strcpy( ret_msg, " [will return 0]");
						if( !special)
							no_msg= 1;
						else
							no_msg= 0;
						verbose= 0;
						r= 1;
						pow_0_0++;
					}
					else if( x->arg2< 0.0){
						set_Inf( x->retval, 1);
						strcpy( ret_msg, " [will return Inf]");
						pow_0_neg++;
						r= 1;
					}
				}
				else if( x->arg1< 0.0 && x->arg2 ){
				  double dum;
				  int sp= special;
					_args= 2;
					dum= fmod( 1.0/ x->arg2, 2.0);
#ifdef OLD_POW_NEG_
					if( dum)
					{
						pow_neg_odd++;
						special= 1;
						x->retval= -1* pow( - x->arg1, x->arg2);
						special= 0;
						sprintf( ret_msg, " [will return %s]", d2str( x->retval, "%.10le", NULL) );
						r= 1;
						if( !special)
							no_msg= 1;
						else
							no_msg= 0;
						verbose= 0;
					}
					else{
						r= 0;
						sprintf( ret_msg, " [will return %s]", d2str( x->retval, "%.10le", NULL) );
						pow_neg_nodd++;
					}
#else
					if( dum) {
						pow_neg_odd++;
						verbose= 0;
					}
					else{
						pow_neg_nodd++;
						if( debugFlag ){
							special= 1;
							verbose= 1;
						}
						else{
							special= 0;
							verbose= 0;
						}
					}
/* 					special= 1;	*/
					x->retval= -1* pow( - x->arg1, x->arg2);
					special= sp;
					sprintf( ret_msg, " [will return %s]", d2str( x->retval, "%.10le", NULL) );
					r= 1;
					if( !special)
						no_msg= 1;
					else
						no_msg= 0;
#endif
				}
			}
			else if( !strcmp( x->name, "sin") ){
				sin_dom++;
				strcpy( ret_msg, " [will return 0]");
				verbose= 0;
				r= 1;
				x->retval= 0.0;
			}
			else if( !strcmp( x->name, "cos") ){
				cos_dom++;
				strcpy( ret_msg, " [will return 0]");
				verbose= 0;
				r= 1;
				x->retval= 0.0;
			}
			else{
				strcpy( ret_msg, " [will return 0]");
				x->retval= 0.0;
			}
			break;
		case OVERFLOW:
			if( !strcmp( x->name, "exp") ){
				exp_over++;
				set_Inf( x->retval, 1);
				strcpy( ret_msg, " [will return Inf]");
				r= 1;
			}
			else if( !strcmp( x->name, "pow") ){
				args= 2;
				if( x->arg1< 0 ){
					pow_neg_over++;
					_args= 1;
					if( 2.0* floor(x->arg2/ 2.0)!= x->arg2 ){
						set_Inf( x->retval, -1);
						strcpy( ret_msg, " [will return -Inf]");
					}
					else{
						set_Inf( x->retval, 1);
						strcpy( ret_msg, " [will return Inf]");
					}
				}
				else if( x->arg1> 0){
					verbose= 0;
					pow_pos_over++;
					set_Inf( x->retval, 1);
					strcpy( ret_msg, " [will return Inf]");
				}
				else{
					pow_0_over++;
					x->retval= 0.0;
					strcpy( ret_msg, " [will return 0]");
					if( !special)
						no_msg= 1;
					else
						no_msg= 0;
					verbose= 0;
				}
				r= 1;
			}
			break;
		case UNDERFLOW:
			if( !strcmp(x->name, "exp") ){
				exp_under++;
				x->retval= 0.0;
				strcpy( ret_msg, " [will return 0]");
				if( !special)
					no_msg= 1;
				else
					no_msg= 0;
				verbose= 0;
				r= 1;
			}
			if( !strcmp( x->name, "pow") ){
				args= 2;
				pow_under++;
				x->retval= 0.0;
				strcpy( ret_msg, " [will return 0]");
				if( !special)
					no_msg= 1;
				else
					no_msg= 0;
				verbose= 0;
				r= 1;
			}
			break;
		case TLOSS:	
		case PLOSS:{
		 double arg, (*fun)(), dum;
			if( !strcmp( x->name, "cos") )
				fun= cos;
			else if( !strcmp( x->name, "sin") )
				fun= sin;
			else if( !strcmp( x->name, "tan") )
				fun= tan;
			else
				fun= NULL;
			if( fun){
			  int its= 0;
				  /* warp it to modulo 2PI	*/
				if( !((arg= radians(x->arg1))>= -M_2PI && arg<= M_2PI ) ){
					_args= 2;
					arg= x->arg1;
					if( Inf(arg) ){
						arg= 0;
					}
					else{
					  int sp= special;
						while( !(arg>= -M_2PI && arg<= M_2PI) ){
							special= 1;
							if( (arg= fmod( arg, M_2PI))== arg ){
							  /* arg too big for fmod: do it like this:	*/
								_args= 1;
								arg= M_2PI* modf( arg/ M_2PI, &dum);
								its++;
							}
							its++;
						}
						special= sp;
					}
					_args= 1;
					x->retval= (*fun)(arg);
					sprintf( ret_msg,
						" [%s(%.10leRad->%.10leRad)=%.10le (%dx)]",
							x->name, x->arg1, arg, x->retval, its
					);
					x->arg1= arg;
					r= 1;
					verbose= 0;
					if( !special)
						no_msg= 1;
					else
						no_msg= 0;
					gon_mod++;
				}
			}
			break;
		}
		case SING:
			if( !strncmp( x->name, "log", 3) ){
				strcpy( ret_msg, " [will return -Inf]");
				set_Inf( x->retval, -1);
				r= 1;
				log_0++;
			}
			break;
		default:
			r= 0;
			ret_msg[0]= '\0';
			break;
	}
	if( matherr_verbose ){
		verbose= 1;
		no_msg= 0;
	}
	if( !no_msg || screwed || ( (matherr_calls-1) % 100)==0 ){
		if( args== 2){
			sprintf( matherr_msg, "matherr(#%d): %s(%s%s,%s%s)==%s%s: %s%s%s",
				matherr_calls,
				x->name, Arg1, nan1, Arg2, nan2, Ret, nanret, matherror[x->type], ret_msg,
				(screwed)? " (screwed)" : ""
			);
		}
		else{
			sprintf( matherr_msg, "matherr(#%d): %s(%s%s)==%s%s: %s%s%s",
				matherr_calls,
				x->name, Arg1, nan1, Ret, nanret, matherror[x->type], ret_msg,
				(screwed)? " (screwed)" : ""
			);
		}
		matherr_message= True;
	}
	else
		matherr_message= False;
	if( verbose && matherr_msg[0] ){
		if( matherr_verbose== -1 ){
			fputc( '<', cx_stderr );
		}
		fputs( matherr_msg, cx_stderr);
		matherr_msg[0]= '\0';
		if( mMark ){
			fprintf( cx_stderr, "\n\tmarked: %s", mMark );
			mMark= NULL;
		}
		if( matherr_verbose== -1 ){
			fputc( '>', cx_stderr );
		}
		else{
			fputc( '\n', cx_stderr);
		}
		fflush( cx_stderr);
	}
	if( active== 1 ){
		mMark= NULL;
	}
	active-= 1;
	return( r);
}

#if defined(__hp9000s800) || defined(sgi)
	void handle_FPE(int sig, int code, struct sigcontext *scp)
#elif apollo
	void handle_FPE(int sig, int code )
#else
	void handle_FPE(sig)
	int sig;
#endif
{ static long calls= 0;

	calls+= 1;
	use_greek_inf= 0;
#ifdef __hp9000s800
{
  int i, ugi= use_greek_inf;
  char *type="";
		switch(code){
			case 12:
				type= "overflow";
				break;
			case 13:
				type= "conditional";
				break;
			case 14:
				type= "assist exception";
				break;
			case 22:
				type= "assist emulation";
				break;
		}
		fprintf( StdErr, "handle_FPE(#%ld,%s)\n", calls, type);
		fprintf( StdErr, "\tContext: syscall= 0x%lx, sl_error=%u, sl_rval=0x%lx,0x%lx (%s)\n\tsl_args= 0x%lx",
			scp->sc_syscall, (unsigned int) scp->sc_error, scp->sc_rval1, scp->sc_rval2,
			d2str( *((double*) &scp->sc_rval1), "%g", NULL ),
			scp->sc_arg[0]
		);
		for( i= 1; i< NUMARGREGS; i++){
			fprintf( StdErr, ", 0x%lx", scp->sc_arg[i] );
		}
		fprintf( StdErr, "\n\tsc_args= 0x%lx", scp->sc_args[0] );
		for( i= 1; i< NUMARGREGS; i++){
			fprintf( StdErr, ", 0x%lx", scp->sc_args[i] );
		}
		fprintf( StdErr, " (%s,", d2str( *((double*) &scp->sc_args[0]), "%g", NULL ) );
		fprintf( StdErr, "%s)", d2str( *((double*) &scp->sc_args[2]), "%g", NULL ) );
		fputc( '\n', StdErr );
		use_greek_inf= ugi;

		if( cgetenv( "XG_DUMP_FPE") ){
		  /* this system does not increment the program counter
		   \ while handling a signal. If we return now, the same
		   \ trap will be generated, now resulting in an abort
		   */
			signal( SIGFPE, SIG_DFL);
			return;
		}
}
#elif sgi
{
  char *type="";
		switch(code){
			case BRK_OVERFLOW:
				type= "integer overflow";
				break;
			case BRK_DIVZERO:
				type= "division by zero";
				break;
			case BRK_MULOVF:
				type= "multiple overflow";
				break;
		}
		fprintf( StdErr, "handle_FPE(#%ld,%s)\n", calls, type);
		  /* sigcontext is a little bit too complex on this architecture, and seems
		   \ not to contain any user-relevant information...
		   */

		if( cgetenv( "XG_DUMP_FPE") ){
		  /* this system does not increment the program counter
		   \ while handling a signal. If we return now, the same
		   \ trap will be generated, now resulting in an abort
		   */
			signal( SIGFPE, SIG_DFL);
			return;
		}
}
#elif apollo
{
  char *type="";
		switch( code ){
			case FPE_INTOVF_TRAP:
				type= "integer overflow";
				break;
			case FPE_INTDIV_TRAP:
				type= "integer division by zero";
				break;
			case FPE_FLTOVF_TRAP:
				type= "floating overflow";
				break;
			case FPE_FLTDIV_TRAP:
				type= "floating/decimal division by zero";
				break;
			case FPE_FLTUND_TRAP:
				type= "floating underflow";
				break;
			case FPE_DECOVF_TRAP:
				type= "decimal overflow";
				break;
			case FPE_SUBRNG_TRAP:
				type= "subscript range";
				break;
			case FPE_FLTOVF_FAULT:
				type= "floating overflow";
				break;
			case FPE_FLTDIV_FAULT:
				type= "floating divide by zero";
				break;
			case FPE_FLTUND_FAULT:
				type= "floating underflow";
				break;
			case FPE_FLTNAN_FAULT:
				type= "signalling NAN";
				break;
			case 1179685:
				type= "divide by Inf";
				break;
			default:
				type= "?";
				break;
		}
		fprintf( StdErr, "handle_FPE(#%ld,%d==%s)\n", calls, code, type);
		if( code== 1179685 || code== 8 ){
		  /* These we ignore, since they can arise due to the acceptance
		   \ of Infs
		   */
			return;
		}
}
#else
		fprintf( StdErr, "handle_FPE(#%ld)\n", calls);
#endif
		fflush( StdErr );
		if( cgetenv( "XG_DUMP_FPE") ){
#undef abort
			abort();
		}
		else if( calls>= 10 ){
			exit(sig);
		}
#define abort() xg_abort()
}

/* Some miscellaneous functions	*/


  /* RJVB 980107: permute array a of length n and type t
   \ by swapping each element with a randomly chosen other element.
   \ RJVB 20001010: The size argument` indicates the size of a's elements: currently,
   \ these can be doubles (8 bytes: compiler will complain when sizeof(double)==sizeof(long)),
   \ shorts (2 bytes) and longs (4 bytes; same as ints on most current platforms).
   \ When quick==True, only swaps elements from the 1st half with elements from the 2nd half.
   \ This should suffise in many cases. Beware that this may not give the best shuffling for
   \ small arrays!
   */
int q_Permute( void *a, int n, int size, int quick)
{ int y, j;
  int N, offset;

	if( !a ){
		return(0);
	}
	if( quick ){
		N= n/2;
		offset= N;
		if( n== 2 ){
			  /* When n==2, we would always swap the elements on each invocation. Therefore,
			   \ we randomly determine whether or not to swap.
			   */
			if( drand48()< 0.5 ){
				return(1);
			}
		}
	}
	else{
		N= n;
		offset= 0;
	}
#if DEBUG==2
	fprintf( stderr, "q_Permute(n=%d,s=%d); 0..%d with %d..%d:", n, size, N-1, offset, n-1 );
#endif
	switch( size ){
#ifndef __x86_64__
		case sizeof(double):{
		  double *d, *dd, ddd;
			dd= d= (double *) a;
			for( j= 0; j< N; j++){
				y= (int)( N* drand48()+ offset);
#if DEBUG==2
				fprintf( stderr, " (%d/%d)", j, y );
#endif
				ddd= *d;
				*d++= dd[y];
				dd[y]= ddd;
			}
			return( j);
			break;
		}
#endif
		case sizeof(short):{
		  short *i, *ii, iii;
			ii= i= (short *) a;
			for( j= 0; j< N; j++){
				y= (int)(N* drand48()+ offset);
#if DEBUG==2
				fprintf( stderr, " (%d/%d)", j, y );
#endif
				iii= *i;
				*i++= ii[y];
				ii[y]= iii;
			}
			return( j);
			break;
		}
		case sizeof(long):{
		  long *l, *ll, lll;
			ll= l= (long *) a;
			for( j= 0; j< N; j++){
				y= (int)( N* drand48()+ offset);
#if DEBUG==2
				fprintf( stderr, " (%d/%d)", j, y );
#endif
				lll= *l;
				*l++= ll[y];
				ll[y]= lll;
			}
			return( j);
			break;
		}
		default:
			return( 0);
			break;
	}
	return( j );
}

