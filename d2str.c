/* (C) 991008 RJB 
 \ Routine(s) for printing a double to string, extending C's customary formatting.
 \ Syntax:
 \ char *d2str( double value, char *format, char *target_buffer )
 \ target_buffer may be NULL, in which case a buffer from an internal (rotating) set will be used.
 \ format may be NULL, in which case the behaviour defaults to "%g".
 \ Options for format:
 \ "!<n>": prefix to regular format selector (g,f,e,..): request a fixed <n> number of significant
 \         digits to print, e.g. (n==4): 0.001234, 0.01234, 1.234, 123.4, 1.234e4
 \ "/"   : used between two regular format selectors, e.g. "%g/%g": make d2str() look for a
 \         representation of <value> as a fraction.
 \ Global flag: Allow_Fractions. Like the "/" format opcode, but looks for a fraction, and
 \ uses this representation only when not resulting in a longer string than the one that would
 \ result from the use of the specified format.
 */

#include "NaN.h"

#ifndef serror
#	include <errno.h>
#	if !defined(linux) && !defined(__MACH__)
		extern char *sys_errlist[];
		extern int sys_nerr;
#	endif
#	ifndef __CYGWIN__
#		define serror()	((errno<0||errno>=sys_nerr)?"Invalid errno":((errno==0)?"No Error":sys_errlist[errno]))
#	else
#		define serror()	strerror(errno)
#	endif
#endif


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

int Allow_Fractions= 1;

#define streq(a,b)	!strcmp(a,b)
#define strneq(a,b,n)	!strncmp(a,b,n)

/* The internal buffers should be long enough to print the longest doubles.
 \ Since %lf prints 1e308 as a 1 with 308 0s and then some, we must take
 \ this possibility into account... DBL_MAX_10_EXP+12=320 on most platforms.
 */
char *d2str( double d, char *Format , char *buf )
{  static char internal_buf[D2STR_BUFS][DBL_MAX_10_EXP+12], *template= NULL, nan_buf[9*sizeof(long)];
   static int buf_nr= 0, temp_len= 0;
   int Sign, plen= 0, mlen, frac_len= 0;
   Boolean nan= False, ext_frac= False;
   char *format= Format, frac_buf[DBL_MAX_10_EXP+12], *slash;
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
			strcpy( nan_buf, "Inf");
		}
		else if( Sign< 0){
			strcpy( nan_buf, "-Inf");
		}
		nan= True;
	}
	/* else	*/{
	  char *c;
	  int len;
		if( !format ){
			format= "%g";
		}
		else if( format[0]== '%' && format[1]== '!' ){
		  char *c= &format[2], *sform;
		  int signif;
			if( nan ){
				format= "%g";
			}
			else if( sscanf( c, "%d", &signif) && signif> 0 ){
			  double pd= fabs(d);
			  double order;
			  int iorder;
				if( pd> 0 ){
					order= log10(pd);
/* 					iorder= (int) ((order< 0)? ceil(order) : floor(order));	*/
					iorder= (int) floor(order);
					if( iorder< -3 || iorder>= signif ){
						sform= "%.*e";
						iorder= 0;
					}
					else{
						if( fabs(floor(d)- d)<= DBL_EPSILON ){
						  /* Ignore the precision argument:	*/
							sform= "%2$g";
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
					sform= "%2$g";
				}
				sprintf( buf, sform, signif- iorder- 1, d );
				return( buf );
			}
		}
		if( (len= strlen(format)+1) > temp_len ){
		  char *c= template;
			if( !(template= (char*) calloc( len, 1)) ){
				fprintf( stderr, "d2str(%g,\"%s\"): can't allocate buffer (%s)\n",
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
				while( *d && !index( "gGfFeE", *d) ){
					d++;
					if( (*d== 'l' || *d== 'L') && index( "gGfFeE", d[1] ) ){
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
#ifdef XGRAPH
				if( debugFlag ){
					fprintf( stderr, "d2str(): %s == %s\n", buf, frac_buf );
				}
#endif
				strcpy( buf, frac_buf );
				plen= frac_len;
			}
		}
	}
	if( mlen> 0 && plen> mlen ){
		fprintf( cx_stderr, "d2str(%g): wrote %d bytes in internal buffer of %d\n",
			d, plen, mlen
		);
		fflush( cx_stderr );
	}
	else if( plen<= 0 ){
		fprintf( cx_stderr, "d2str(%g)= \"%s\": no output generated: report to sysop\n", d, buf);
		fflush( cx_stderr );
	}
	return( buf );
}
