/* ascanf.c: simple scanner for double arrays
 \ that doubles as a function interpreter.
 * (C) R. J. Bertin 1990,1991,..,1994
 * :ts=5
 * :sw=5
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "cpu.h"

#include "Macros.h"
#include "varsintr.h"

#include <signal.h>

#ifndef STR
#define STR(name)	# name
#endif
#ifndef STRING
#define STRING(name)	STR(name)
#endif

IDENTIFY( "ascanf routines");

extern FILE *StdErr;

long d2long( x)
double x;
{  double X= (double) MAXLONG;
	if( x <= X && x>= -X -1L ){
		return( (long) x);
	}
	else if( x< 0)
		return( LONG_MIN);
	else if( x> 0)
		return( MAXLONG);
}

short d2short(x)
double x;
{  double X= (double)MAXLONG;
	if( x <= X && x>= -X -1L ){
		return( (long) x);
	}
	else if( x< 0)
		return( LONG_MIN);
	else if( x> 0)
		return( MAXLONG);
}

int d2int( x)
double x;
{  double X= (double)MAXLONG;
	if( x <= X && x>= -X -1L ){
		return( (long) x);
	}
	else if( x< 0)
		return( LONG_MIN);
	else if( x> 0)
		return( MAXLONG);
}

/* find the first occurence of the character 'match' in the
 \ string arg_buf, skipping over balanced pairs of <brace-left>
 \ and <brace_right>. If match equals ' ' (a space), any whitespace
 \ is assumed to match.
 */
char *find_balanced( char *arg_buf, const char match, const char brace_left, const char brace_right)
{ int brace_level= 0;
  char *d= arg_buf;
	while( d && *d && !( ((match==' ')? isspace((unsigned char)*d) : *d== match) && brace_level== 0) ){
		if( *d== brace_left){
			brace_level++;
		}
		else if( *d== brace_right){
			brace_level--;
		}
		d++;
	}
	return( d );
}

char ascanf_separator= ',';

/* this index() function skips over a (nested) matched
 * set of square braces.
 */
char *ascanf_index( char *s, char c)
{
	if( !( s= find_balanced( s, c, '[', ']')) ){
		return(NULL);
	}
	if( (c== ' ')? isspace((unsigned char)*s) : *s== c ){
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
int reset_ascanf_index_value= True;
double ascanf_self_value, ascanf_current_value, ascanf_index_value, ascanf_memory[ASCANF_MAX_ARGS], ascanf_progn_return;
int ascanf_arguments, ascanf_arg_error, ascanf_verbose= 0;

double *ascanf_data_buf;
int *ascanf_column_buf;

static int ascanf_while_loop= 0;

DEFUN(ascanf_if, ( double args[ASCANF_MAX_ARGS], double *result), int);
DEFUN(ascanf_progn, ( double args[ASCANF_MAX_ARGS], double *result), int);
DEFUN(ascanf_verbose_fnc, ( double args[ASCANF_MAX_ARGS], double *result), int);
DEFUN(fascanf, ( int *n, char *s, double *a, char *ch, double data[3], int column[3]), int);
DEFUN( ascanf_whiledo, ( double args[ASCANF_MAX_ARGS], double *result), int );
DEFUN( ascanf_dowhile, ( double args[ASCANF_MAX_ARGS], double *result), int );
DEFUN( ascanf_print, ( double args[ASCANF_MAX_ARGS], double *result), int );
DEFUN( ascanf_Eprint, ( double args[ASCANF_MAX_ARGS], double *result), int );

int ascanf_function( Function, Index, s, ind, A, caller)
ascanf_Function *Function;
int Index;
char **s, *caller;
int ind;
double *A;
{ ALLOCA( arg_buf, char, ASCANF_FUNCTION_BUF, arg_buf_len);
  char *c= arg_buf, *d, *args= arg_buf;
  double arg[ASCANF_MAX_ARGS];
  int i= 0, n= Function->args, brace_level= 0, ok= 0, ok2= 0, verb= ascanf_verbose;
  DEFMETHOD( function, (double arguments[ASCANF_MAX_ARGS], double *result), int)= Function->function;
  char *name= Function->name;
  static int level= 0;

	level++;
	ascanf_arg_error= 0;
	if( function== ascanf_verbose_fnc ){
		ascanf_verbose= 1;
		fprintf( StdErr, "#%d:\t%s\n", level, *s );
	}
	if( *(d= &((*s)[ind]) )== '[' ){
		d++;
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
		*c= '\0';
		if( *d== ']' ){
		  /* there was an argument list: parse it	*/
			while( isspace((unsigned char)*args) )
				args++;
			if( !strlen(args) ){
			  /* empty list...	*/
				ascanf_arguments= 0;
				if( ascanf_verbose ){
					fprintf( StdErr, "#%d\t%s== ", level, name );
					fflush( StdErr );
				}
				ok= (*function)( NULL, A);
				ascanf_current_value= *A;
				ok2= 1;
			}
			else{
				if( i<= ASCANF_FUNCTION_BUF-1 ){
				  int j;
				  char *larg[3]= { NULL, NULL, NULL};
					for( arg[0]= 0.0, j= 1; j< Function->args; j++){
						arg[j]= 1.0;
					}
					if( function== ascanf_if ){
					  int N= 0;
						larg[0]= arg_buf;
						larg[1]= ascanf_index( larg[0], ascanf_separator);
						larg[2]= ascanf_index( &(larg[1][1]), ascanf_separator);
						if( larg[0] ){
							N= 1;
							n= 1;
							fascanf( &n, larg[0], &arg[0], NULL, NULL, NULL );
							if( arg[0] ){
							  /* try to evaluate a second argument	*/
								if( larg[1] ){
									n= 1;
									fascanf( &n, &(larg[1][1]), &arg[1], NULL, NULL, NULL);
									if( n ){
										N= 2;
									}
								}
							}
							else{
								set_NaN( arg[1] );
								if( larg[2] ){
								  /* try to evaluate the third argument.	*/
									n= 1;
									fascanf( &n, &(larg[2][1]), &arg[2], NULL, NULL, NULL);
									if( n ){
										N= 3;
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
/* 
					else if( function== ascanf_progn || function== ascanf_dowhile ||
							function== ascanf_print || function== ascanf_Eprint || function== ascanf_verbose_fnc
					){
						ascanf_progn_return= 0.0;
					}
 */
					do{
						if( ascanf_while_loop> 0 ){
						  /* We can come back here while evaluating the arguments of
						   \ a while function. We don't want to loop those forever...
						   */
							if( ascanf_verbose ){
								fputs( " (loop)\n", StdErr );
							}
							ascanf_while_loop= -1;
						}
						  /* ascanf_whiledo is the C while() construct. If the first
						   \ element, the test, evals to false, the rest of the arguments
						   \ are not evaluated. ascanf_dowhile tests the last argument,
						   \ which is much easier to implement.
						   */
						if( function== ascanf_whiledo ){
						  int N= n-1;
/* 							ascanf_progn_return= 0.0;	*/
							larg[0]= arg_buf;
							larg[1]= ascanf_index( larg[0], ascanf_separator);
							n= 1;
							fascanf( &n, larg[0], &arg[0], NULL, NULL, NULL );
							if( arg[0] ){
							  /* test argument evaluated true, now evaluate rest of the args	*/
								n= N;
								fascanf( &n, &larg[1][1], &arg[1], NULL, NULL, NULL );
								 /* the actual number of arguments:	*/
								n+= 1;
							}
							else{
							  /* test false: we skip the rest	*/
								n= 1;
							}
						}
						else{
							if( !(larg[0] || larg[1] || larg[2]) ){
								fascanf( &n, arg_buf, arg, NULL, NULL, NULL);
							}
						}
						ascanf_arguments= n;
						if( ascanf_verbose ){
							fprintf( StdErr, "#%d\t%s[%s", level, name, d2str( arg[0], "%g", NULL) );
							for( j= 1; j< n; j++){
								fprintf( StdErr, ",%s", d2str( arg[j], "%g", NULL) );
							}
							fprintf( StdErr, "]== ");
							fflush( StdErr );
						}
						ok= (*function)( arg, A);
						ascanf_current_value= *A;
					} while( ascanf_while_loop> 0 );
					ok2= 1;
				}
				else{
					fprintf( StdErr, "%s(\"%s\"): %s arguments list too long (%s)\n",
						caller, *s, arg_buf
					);
				}
			}
			*s= d;
		}
		else{
			fprintf( StdErr, "%s(\"%s\"): missing ']'\n", caller, *s);
		}
	}
	else{
	  /* there was no argument list	*/
		*args= '\0';
		ascanf_arguments= 0;
		if( ascanf_verbose ){
			fprintf( StdErr, "#%d\t%s== ", level, name );
			fflush( StdErr );
		}
		ok= (*function)(NULL,A);
		ascanf_current_value= *A;
		ok2= 1;
	}
	if( ok2){
		if( ascanf_verbose ){
			fprintf( StdErr, "%s%s",
				d2str( *A, "%g", NULL),
				(level== 1)? "\t  ," : "\t->"
			);
			if( ascanf_arg_error ){
				fprintf( StdErr, " (needs %d arguments)", Function->args );
			}
			fputc( '\n', StdErr );
			fflush( StdErr);
		}
	}
	else if( ascanf_arg_error ){
		fprintf( StdErr, "%s== %s needs %d arguments\n", name, d2str( *A, "%g", NULL), Function->args );
		fflush( StdErr );
	}
	if( function== ascanf_verbose_fnc ){
		ascanf_verbose= verb;
	}
	level--;
	GCA();
	return( ok);
}

int set_ascanf_memory( double d)
{  int i;
	for( i= 0; i< ASCANF_MAX_ARGS; i++ ){
		ascanf_memory[i]= d;
	}
	return( i );
}

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

int ascanf_array_manip( double args[ASCANF_MAX_ARGS], double *result, DoubleOrInt *array, int size)
{
	if( !args || ascanf_arguments== 0 ){
		ascanf_arg_error= 0;
		if( ascanf_verbose ){
			fprintf( StdErr, " (slots) " );
		}
		*result= (double) size;
		return(1);
	}
	if( (args[0]= floor( args[0] ))>= -1 && args[0]< ASCANF_MAX_ARGS ){
	  int i= (int) args[0], I= i+1;
		if( ascanf_arguments>= 2 ){
			if( i== -1 ){
				I= ASCANF_MAX_ARGS;
				i= 0;
			}
			for( ; i< I; i++ ){
				set_DOI(array,i, args[1]);
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
		ascanf_arg_error= 1;
		*result= 0;
	}
	return( 1 );
}

int ascanf_mem( double args[ASCANF_MAX_ARGS], double *result)
{ DoubleOrInt doi;
	doi.p.d= ascanf_memory;
	doi.type= _double;
	return( ascanf_array_manip( args, result, &doi, ASCANF_MAX_ARGS ) );
}

int ascanf_data( double args[ASCANF_MAX_ARGS], double *result)
{ DoubleOrInt doi;
	doi.p.d= ascanf_data_buf;
	doi.type= _double;
	return( ascanf_array_manip( args, result, &doi, 3 ) );
}

int ascanf_column( double args[ASCANF_MAX_ARGS], double *result)
{ DoubleOrInt doi;
	doi.p.i= ascanf_column_buf;
	doi.type= _int;
	return( ascanf_array_manip( args, result, &doi, 3 ) );
}

double **ascanf_mxy_buf;
int ascanf_mxy_X= 0, ascanf_mxy_Y= 0;

int free_ascanf_mxy_buf()
{ int i;
	if( ascanf_mxy_X>= 0 && ascanf_mxy_Y>= 0 && ascanf_mxy_buf ){
		for( i= 0; i< ascanf_mxy_X; i++ ){
			if( ascanf_mxy_buf[i] ){
				free( ascanf_mxy_buf[i] );
			}
		}
		free( ascanf_mxy_buf );
		ascanf_mxy_buf= NULL;
		ascanf_mxy_X= 0;
		ascanf_mxy_Y= 0;
	}
}

int ascanf_setmxy( double args[ASCANF_MAX_ARGS], double *result)
{
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ((args[0]= floor( args[0] ))>= 0 ) &&
		((args[1]= floor( args[1] ))>= 0 )
	){
	  int i, ok;
		free_ascanf_mxy_buf();
		ascanf_mxy_X= (int) args[0];
		ascanf_mxy_Y= (int) args[1];
		if( (ascanf_mxy_buf= (double**) calloc( ascanf_mxy_X, sizeof(double*))) ){
			for( ok= 1, i= 0; i< ascanf_mxy_X; i++ ){
				if( !(ascanf_mxy_buf[i]= (double*) calloc( ascanf_mxy_Y, sizeof(double))) ){
					ok= 0;
				}
			}
			if( !ok ){
				ascanf_arg_error= 1;
				free_ascanf_mxy_buf();
				*result= 0;
				return( 0 );
			}
			*result= (double) ascanf_mxy_X * ascanf_mxy_Y * sizeof(double);
		}
	}
	return( 1 );
}

int ascanf_mxy( double args[ASCANF_MAX_ARGS], double *result)
{
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ((args[0]= floor( args[0] ))>= -1 && args[0]< ascanf_mxy_X) &&
		((args[1]= floor( args[1] ))>= -1 && args[1]< ascanf_mxy_Y) &&
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
			for( ; x< X; x++ ){
				for( y= y0; y< Y; y++ ){
					ascanf_mxy_buf[x][y]= args[2];
				}
			}
			*result= args[2];
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
		ascanf_arg_error= 1;
		*result= 0;
	}
	return( 1 );
}

int ascanf_Index( double args[ASCANF_MAX_ARGS], double *result)
{
	*result= ascanf_index_value;
	return( 1);
}

int ascanf_self( double args[ASCANF_MAX_ARGS], double *result )
{
	*result= ascanf_self_value;
	return( 1 );
}

int ascanf_current( double args[ASCANF_MAX_ARGS], double *result )
{
	*result= ascanf_current_value;
	return( 1 );
}

extern double drand48();
#define drand()	drand48()

/* routine for the "ran[low,high]" ascanf syntax	*/
int ascanf_random( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		*result= drand();
		return( 1 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= drand() * (args[1]- args[0]) + args[0];
		return( 1 );
	}
}

/* routine for the "add[x,y]" ascanf syntax	*/
int ascanf_add( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= args[0];
		for( i= 1; i< ascanf_arguments; i++ ){
			*result+= args[i];
		}
		return( 1 );
	}
}

/* routine for the "sub[x,y]" ascanf syntax	*/
int ascanf_sub( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= args[0];
		for( i= 1; i< ascanf_arguments; i++ ){
			*result-= args[i];
		}
		return( 1 );
	}
}

/* routine for the "mul[x,y]" ascanf syntax	*/
int ascanf_mul( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= args[0];
		for( i= 1; i< ascanf_arguments; i++ ){
			*result*= args[i];
		}
		return( 1 );
	}
}

/* routine for the "div[x,y]" ascanf syntax	*/
int ascanf_div( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i;
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( !args[1]){
			set_Inf( *result, args[0] );
		}
		else{
			*result= args[0];
			for( i= 1; i< ascanf_arguments; i++ ){
				if( !args[i]){
					set_Inf( *result, *result );
				}
				else{
					*result/= args[i];
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
int ascanf_fac( double args[ASCANF_MAX_ARGS], double *result )
{
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

/* routine for the "atan2[x,y]" ascanf syntax	*/
int ascanf_atan2( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		*result= args[2]* ( atan2( args[0], args[1] ) / M_2PI );
		return( 1 );
	}
}

/* return arg(x,y) in radians [0,2PI]	*/
double arg( x, y)
double x, y;
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
		else
			return( 0.0);
	}
}

/* routine for the "arg[x,y]" ascanf syntax	*/
int ascanf_arg( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		if( ascanf_arguments< 3 || args[2]== 0.0 ){
			args[2]= M_2PI;
		}
		*result= args[2] * (arg( args[0], args[1] )/ M_2PI);
		return( 1 );
	}
}

/* routine for the "len[x,y]" ascanf syntax	*/
int ascanf_len( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= sqrt( args[0]*args[0] + args[1]*args[1] );
		return( 1 );
	}
}

/* routine for the "log[x,y]" ascanf syntax	*/
int ascanf_log( double args[ASCANF_MAX_ARGS], double *result )
{ double largs1;
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		if( ascanf_arguments== 1 || args[1]== 1 ){
			largs1= 1.0;
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
int ascanf_pow( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= pow( args[0], args[1] );
		return( 1 );
	}
}

/* routine for the "pi[mul_fact]" ascanf syntax	*/
int ascanf_pi( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		*result= M_PI;
		return( 1 );
	}
	else{
		  /* we don't need args[1], so we set it to 0
		   * (which prints nicer)
		   */
		args[1]= 0.0;
		*result= M_PI * args[0];
		return( 1 );
	}
}

/* routine for the "split[]" ascanf syntax	*/
int ascanf_split( double args[ASCANF_MAX_ARGS], double *result )
{  extern int split_set;
	if( !args){
		split_set= 1;
		*result= ascanf_self_value;
		return( 1 );
	}
	else{
		*result= ascanf_self_value;
		split_set= (args[0])? -1 : 1;
		return( 1 );
	}
}

#include <sys/times.h>

typedef struct time_struct{
	double TimeStamp, Tot_TimeStamp;
	double Time, Tot_Time;
}Time_Struct;

#define SECSPERDAY 86400.0
#ifdef CLK_TCK
#	define TICKS_PER_SECOND (double)(CLK_TCK)
#else	/* assume 60 ticks..	*/
#	define TICKS_PER_SECOND (60.0)
#endif
static double Then, Tot_Then;
double Tot_Start= 0.0, Tot_Time= 0.0, Used_Time= 0.0;

void Set_Timer()
{	struct tms tms;
	Tot_Then= (double)times( &tms);
	Then= (double)(tms.tms_utime+ tms.tms_cutime);
	Tot_Time= 0.0;
	Used_Time= 0.0;
	if( !Tot_Start)
		Tot_Start= Tot_Then;
}

static struct tms ET_tms;
static double ET_Now, ET_Tot_Now;
double Elapsed_Time()	/* return number of seconds since last call */
{	struct tms *tms= &ET_tms;
	double Elapsed;

	ET_Tot_Now= (double) times( tms);
	ET_Now= (double)(tms->tms_utime+ tms->tms_cutime);
	Elapsed= (ET_Now- Then)/TICKS_PER_SECOND;
		Then= ET_Now;
	Tot_Time= (ET_Tot_Now- Tot_Then)/TICKS_PER_SECOND;
		Tot_Then= ET_Tot_Now;
	return( (Used_Time= Elapsed) );
}

static struct tms ES_tms;
double Elapsed_Since( then)	/* return number of seconds since last call */
Time_Struct *then;
{	struct tms *tms= &ES_tms;
	double ES_Now, ES_Tot_Now, Elapsed;

	ES_Tot_Now= (double) times( tms);
	ES_Now= (double)(tms->tms_utime+ tms->tms_cutime);
	then->Tot_Time= Tot_Time= (ES_Tot_Now- then->Tot_TimeStamp)/TICKS_PER_SECOND;
	then->Time= Elapsed= (ES_Now- then->TimeStamp)/TICKS_PER_SECOND;
		then->TimeStamp= ES_Now;
		then->Tot_TimeStamp= ES_Tot_Now;
	return( (Used_Time= Elapsed) );
}

int ascanf_elapsed( double args[ASCANF_MAX_ARGS], double *result )
{  static Time_Struct timer;
   static char called= 0;
	Elapsed_Since( &timer );
	if( !called){
		called= 1;
		*result= 0.0;
		return(1);
	}
	if( !ascanf_arguments ){
		*result= timer.Tot_Time;
	}
	else{
		*result= timer.Time;
	}
	return( 1 );
}

int ascanf_time( double args[ASCANF_MAX_ARGS], double *result )
{  static Time_Struct timer_0;
   Time_Struct timer;
   static char called= 0;
	if( !called){
	  /* initialise the real timer	*/
		called= 1;
		Set_Timer();
		Elapsed_Since( &timer_0 );
		*result= 0.0;
		return(1);
	}
	else{
	  /* make a copy of the timer, and determine the time
	   \ since initialisation
	   */
		timer= timer_0;
		Elapsed_Since( &timer );
	}
	if( !ascanf_arguments ){
		*result= timer.Tot_Time;
	}
	else{
		*result= timer.Time;
	}
	return( 1 );
}

#ifndef degrees
#	define degrees(a)			((a)*57.295779512)
#endif
#ifndef radians
#	define radians(a)			((a)/57.295779512)
#endif

int ascanf_degrees( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		*result= degrees(M_2PI);
		return( 1 );
	}
	else{
		  /* we don't need args[1], so we set it to 0
		   * (which prints nicer)
		   */
		args[1]= 0.0;
		*result= degrees( args[0]);
		return( 1 );
	}
}

int ascanf_radians( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		*result= radians(360.0);
		return( 1 );
	}
	else{
		  /* we don't need args[1], so we set it to 0
		   * (which prints nicer)
		   */
		args[1]= 0.0;
		*result= radians( args[0]);
		return( 1 );
	}
}

int ascanf_sin( double args[ASCANF_MAX_ARGS], double *result )
{
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

int ascanf_cos( double args[ASCANF_MAX_ARGS], double *result )
{
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

int ascanf_tan( double args[ASCANF_MAX_ARGS], double *result )
{
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

int ascanf_exp( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		*result= exp(1.0);
		return( 1 );
	}
	else{
		  /* we don't need args[1], so we set it to 0
		   * (which prints nicer)
		   */
		args[1]= 0.0;
		*result= exp( args[0]);
		return( 1 );
	}
}

int ascanf_dcmp( double args[ASCANF_MAX_ARGS], double *result )
{
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
double normal_rand( double av, double stdv )
{
	static int iset=0;
	static double gset;
	double fac,r,v1,v2;

	if  (iset == 0) {
		do {
			v1= 2.0* drand()- 1.0;
			v2= 2.0* drand()- 1.0;
			r=v1*v1+v2*v2;
		} while (r >= 1.0);
		fac= stdv * sqrt(-2.0*log(r)/r);
		gset= v1*fac;
		iset=1;
		return( av + v2*fac );
	} else {
		iset=0;
		return av + gset;
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

int ascanf_misc_fun( double args[ASCANF_MAX_ARGS], double *result, int code, int argc )
{
	if( !args || ascanf_arguments< argc ){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int a0= (args[0]!= 0.0), a1= (args[1]!= 0.0);
		if( ascanf_arguments== 2 ){
			args[2]= 0.0;
		}
		switch( code ){
			case 0:
				*result= (args[0])? args[1] : args[2];
				break;
			case -1:
				*result= (double) ( args[0] == args[1] );
				break;
			case 1:
				*result= (double) ( args[0] > args[1] );
				break;
			case 2:
				*result= (double) ( args[0] < args[1] );
				break;
			case 3:
				*result= (double) ( args[0] >= args[1] );
				break;
			case 4:
				*result= (double) ( args[0] <= args[1] );
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
				*result= MIN( args[0], args[1] );
				break;
			case 12:
				*result= MAX( args[0], args[1] );
				break;
			case 13:
				*result= floor( args[0] );
				break;
			case 14:
				*result= ceil( args[0] );
				break;
			case 15:
				*result= uniform_rand( args[0], args[1] );
				break;
			case 16:
				*result= abnormal_rand( args[0], args[1] );
				break;
			case 17:
				*result= erf( args[0] );
				break;
			case 18:
				*result= erfc( args[0] );
				break;
	    		case 19:
	    			*result= normal_rand( args[0], args[1] );
	    			break;
	    		case 20:
				if( args[0]< args[1] ){
					*result=  args[1];
				}
				else if( args[0]> args[2] ){
					*result= args[2];
				}
				else{
					*result= args[0];
				}
	    			break;
		}
		return( 1 );
	}
}

int ascanf_if( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 0, 2) );
}

int ascanf_if2( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 0, 2) );
}

int ascanf_gt( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 1, 2) );
}

int ascanf_eq( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, -1, 2) );
}

int ascanf_lt( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 2, 2) );
}

int ascanf_ge( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 3, 2) );
}

int ascanf_le( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 4, 2) );
}

int ascanf_and( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 5, 2) );
}

int ascanf_or( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 6, 2) );
}

int ascanf_xor( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 7, 2) );
}

int ascanf_not( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 8, 1) );
}

int ascanf_abs( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 9, 1) );
}

int ascanf_return( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 10, 1) );
}

int ascanf_min( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 11, 2) );
}

int ascanf_max( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 12, 2) );
}

int ascanf_floor( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 13, 1) );
}

int ascanf_ceil( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 14, 1) );
}

int ascanf_uniform( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 15, 2) );
}

int ascanf_abnormal( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 16, 2) );
}

int ascanf_erf( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 17, 1) );
}

int ascanf_erfc( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 18, 1) );
}

int ascanf_normal( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 19, 2) );
}

int ascanf_clip( double args[ASCANF_MAX_ARGS], double *result)
{
	return( ascanf_misc_fun( args, result, 20, 3) );
}

int ascanf_progn( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		*result= ascanf_progn_return;
		return( 1 );
	}
}

int ascanf_whiledo( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args || ascanf_arguments< 1 || (args[0] && ascanf_arguments< 2) ){
	  /* if !args[0], this routine is called with ascanf_arguments==1,
	   \ in which case it should set ascanf_while_loop to true
	   \ (this is to avoid jumping or deep nesting in ascanf_function
	   */
		ascanf_arg_error= 1;
		ascanf_while_loop= 0;
		return( 0 );
	}
	else{
		*result= ascanf_progn_return;
		ascanf_while_loop= (args[0]!= 0);
		return( 1 );
	}
}

int ascanf_dowhile( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
		ascanf_while_loop= 0;
		return( 0 );
	}
	else{
		*result= ascanf_progn_return;
		ascanf_while_loop= (args[ascanf_arguments-1]!= 0);
		return( 1 );
	}
}

/* routine for the "print[x,y,...]" ascanf syntax	*/
int ascanf_print( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i;
		if( ascanf_arguments< 2 ){
			*result= args[0];
		}
		else{
			*result= ascanf_progn_return;
		}
		fprintf( stdout, "print[%s", d2str( args[0], "%g", NULL) );
		for( i= 1; i< ascanf_arguments; i++ ){
			fprintf( stdout, ",%s", d2str( args[i], "%g", NULL) );
		}
		fprintf( stdout, "]== %s\n", d2str( *result, "%g", NULL) );
		return( 1 );
	}
}

/* routine for the "Eprint[x,y,...]" ascanf syntax	*/
int ascanf_Eprint( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
	  int i;
		if( ascanf_arguments< 2 ){
			*result= args[0];
		}
		else{
			*result= ascanf_progn_return;
		}
		fprintf( StdErr, "print[%s", d2str( args[0], "%g", NULL) );
		for( i= 1; i< ascanf_arguments; i++ ){
			fprintf( StdErr, ",%s", d2str( args[i], "%g", NULL) );
		}
		fprintf( StdErr, "]== %s\n", d2str( *result, "%g", NULL) );
		return( 1 );
	}
}

/* routine for the "verbose[x,y,...]" ascanf syntax	*/
int ascanf_verbose_fnc( double args[ASCANF_MAX_ARGS], double *result )
{
	if( ascanf_arguments== 1 ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return( 1 );
}

/* routine for the "fmod[x,y]" ascanf syntax	*/
int ascanf_fmod( double args[ASCANF_MAX_ARGS], double *result )
{
	if( !args){
		ascanf_arg_error= 1;
		return( 0 );
	}
	else{
		ascanf_arg_error= (ascanf_arguments< 2 );
		*result= fmod( args[0], args[1] );
		return( 1 );
	}
}

ascanf_Function vars_ascanf_Functions[]= {
	{ "index", ascanf_Index, 0, NOT_EOF, "index (of element)" },
	{ "self", ascanf_self, 0, NOT_EOF, "self (initial value = parameter range variable)" },
	{ "split", ascanf_split, 1, NOT_EOF, "split: new dataset or split[1]: new dataset after current point; returns self" },
	{ "current", ascanf_current, 0, NOT_EOF, "current value" },
	{ "elapsed", ascanf_elapsed, 1, NOT_EOF, "elapsed since last call (real time) or elapsed[x] (system time)"},
	{ "time", ascanf_time, 1, NOT_EOF, "time since first call (real time) or time[x] (system time)"},
	{ "add", ascanf_add, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "add[x,y[,..]]" },
	{ "sub", ascanf_sub, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "sub[x,y[,..]]" },
	{ "mul", ascanf_mul, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "mul[x,y[,..]]" },
	{ "div", ascanf_div, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "div[x,y[,..]]" },
	{ "ran", ascanf_random, 2, NOT_EOF, "ran or ran[low,high]" },
	{ "pi", ascanf_pi, 1, NOT_EOF, "pi or pi[mulfac]" },
	{ "pow", ascanf_pow, 2, NOT_EOF_OR_RETURN, "pow[x,y]" },
	{ "fac", ascanf_fac, 2, NOT_EOF_OR_RETURN, "fac[x,dx]" },
	{ "radians", ascanf_radians, 1, NOT_EOF, "radians[x]" },
	{ "degrees", ascanf_degrees, 1, NOT_EOF, "degrees[x]" },
	{ "sin", ascanf_sin, 2, NOT_EOF_OR_RETURN, "sin[x[,base]]" },
	{ "cos", ascanf_cos, 2, NOT_EOF_OR_RETURN, "cos[x[,base]]" },
	{ "tan", ascanf_tan, 2, NOT_EOF_OR_RETURN, "tan[x[,base]]" },
	{ "atan2", ascanf_atan2, 3, NOT_EOF_OR_RETURN, "atan2[x,y[,base]] atan(y/x)" },
	{ "arg", ascanf_arg, 3, NOT_EOF_OR_RETURN, "arg[x,y[,base]] angle to (x,y) in 0..2PI" },
	{ "len", ascanf_len, 2, NOT_EOF_OR_RETURN, "len[x,y] distance to (x,y)" },
	{ "exp", ascanf_exp, 1, NOT_EOF, "exp[x]" },
	{ "erf", ascanf_erf, 1, NOT_EOF_OR_RETURN, "erf[x]" },
	{ "1-erf", ascanf_erfc, 1, NOT_EOF_OR_RETURN, "1-erf[x]" },
	{ "log", ascanf_log, 2, NOT_EOF_OR_RETURN, "log[x,y]" },
	{ "abs", ascanf_abs, 1, NOT_EOF_OR_RETURN, "abs[x]" },
	{ "floor", ascanf_floor, 1, NOT_EOF_OR_RETURN, "floor[x]" },
	{ "ceil", ascanf_ceil, 1, NOT_EOF_OR_RETURN, "ceil[x]" },
	{ "cmp", ascanf_dcmp, 3, NOT_EOF_OR_RETURN, "cmp[x,y[,precision]]" },
	{ "fmod", ascanf_fmod, 2, NOT_EOF_OR_RETURN, "fmod[x,y]" },
	{ "ifelse2", ascanf_if2, 3, NOT_EOF_OR_RETURN, "ifelse2[expr,val1,[else-val:0]] - all arguments are evaluated" },
	{ "ifelse", ascanf_if, 3, NOT_EOF_OR_RETURN, "ifelse[expr,val1,[else-val:0]] - lazy evaluation" },
	{ "=", ascanf_eq, 2, NOT_EOF_OR_RETURN, "=[x,y]" },
	{ ">=", ascanf_ge, 2, NOT_EOF_OR_RETURN, ">=[x,y]" },
	{ "<=", ascanf_le, 2, NOT_EOF_OR_RETURN, "<=[x,y]" },
	{ ">", ascanf_gt, 2, NOT_EOF_OR_RETURN, ">[x,y]" },
	{ "<", ascanf_lt, 2, NOT_EOF_OR_RETURN, "<[x,y]" },
	{ "AND", ascanf_and, 2, NOT_EOF_OR_RETURN, "AND[x,y] (boolean)" },
	{ "OR", ascanf_or, 2, NOT_EOF_OR_RETURN, "OR[x,y] (boolean)" },
	{ "XOR", ascanf_xor, 2, NOT_EOF_OR_RETURN, "XOR[x,y] (boolean)" },
	{ "NOT", ascanf_not, 1, NOT_EOF_OR_RETURN, "NOT[x] (boolean)" },
	{ "MIN", ascanf_min, 2, NOT_EOF_OR_RETURN, "MIN[x,y]" },
	{ "MAX", ascanf_max, 2, NOT_EOF_OR_RETURN, "MAX[x,y]" },
	{ "DATA", ascanf_data, 2, NOT_EOF_OR_RETURN,
		"DATA[n[,expr]]: set DATA[n] to expr, or get DATA[n]: n=0..2 direct datapoint manipulation\n"
	},
	{ "COLUMN", ascanf_column, 2, NOT_EOF_OR_RETURN,
		"COLUMN[n[,expr]]: set COLUMN[n] to expr, or get COLUMN[n]: n=0..2 direct datapoint order manipulation\n"
	},
	{ "MEM", ascanf_mem, 2, NOT_EOF_OR_RETURN,
		"MEM[n[,expr]]: set MEM[n] to expr, or get MEM[n]: " STRING(ASCANF_MAX_ARGS) " locations"\
		"\n\tSpecify n=-1 to set whole range"
	},
	{ "SETMXY", ascanf_setmxy, 2, NOT_EOF_OR_RETURN, "SETMXY[I,J]: set new dimensions of MXY buffer" },
	{ "MXY", ascanf_mxy, 3, NOT_EOF_OR_RETURN,
		"MXY[i,j[,expr]]: set MXY[i,j] to expr, or get MXY[i,j]"
	},
	{ "return", ascanf_return, 1, NOT_EOF_OR_RETURN, "return[x]" },
	{ "progn", ascanf_progn, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "progn[expr1[,..,expN]]: value set by return[x]" },
	{ "whiledo", ascanf_whiledo, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "whiledo[test_expr,expr1[,..,expN]]: value set by return[x]" },
	{ "dowhile", ascanf_dowhile, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "dowhile[expr1[,..,expN],test_expr]: value set by return[x]" },
	{ "print", ascanf_print, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "print[x[,..]]: returns first (and only) arg,\n\
\t\tvalue set by return[y]" },
	{ "Eprint", ascanf_Eprint, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "Eprint[x[,..]]: prints on stderr" },
	{ "verbose", ascanf_verbose_fnc, ASCANF_MAX_ARGS, NOT_EOF_OR_RETURN, "verbose[[x,..]]: turns on verbosity for its scope" },
	{ "uniform", ascanf_uniform, 2, NOT_EOF_OR_RETURN, "uniform[av,stdv]: random number in a uniform distribution" },
	{ "abnormal", ascanf_abnormal, 2, NOT_EOF_OR_RETURN, "abnormal[av,stdv]: random number in an abnormal distribution" },
	{ "normal", ascanf_normal, 2, NOT_EOF_OR_RETURN, "normal[av,stdv]: random number in a normal distribution" },
	{ "clip", ascanf_clip, 3, NOT_EOF_OR_RETURN, "clip[expr,min,max]" },
}, *ascanf_FunctionList= vars_ascanf_Functions;

int ascanf_Functions= sizeof(vars_ascanf_Functions)/sizeof(ascanf_Function);

long ascanf_hash( name)
char *name;
{  long hash= 0L;
#ifdef DEBUG_EXTRA
   int len= 0;
#endif
	while( *name && *name!= '[' && *name!= ',' && !isspace((unsigned char)*name) ){
		hash+= hash<<3L ^ *name++;
#ifdef DEBUG_EXTRA
		len+= 1;
#endif
	}
	return( hash);
}

int check_for_ascanf_function( int Index, char **s, double *result, int *ok, char *caller)
{  int i, len;
   ascanf_Function *af;
   long hash= ascanf_hash( *s );
   static char called= 0;
	if( !called ){
	  long seed;
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
		prev_ran= drand();
		called= 1;
	}
	if( reset_ascanf_index_value ){
		ascanf_self_value= *result;
		ascanf_index_value= (double) Index;
		ascanf_current_value= *result;
	}
	for( i= 0; i< ascanf_Functions && ascanf_FunctionList; i++){
		af= &ascanf_FunctionList[i];
		while( af && af->name && af->function ){
			if( af->name_length ){
				len= af->name_length;
			}
			else{
				len= af->name_length= strlen( af->name );
			}
			if( !af->hash ){
				af->hash= ascanf_hash( af->name );
			}
			if( hash== af->hash ){
				if( !strncmp( *s, af->name, len) ){
					*ok= ascanf_function( af, Index, s, len, result, caller);
					switch(af->type){
						case NOT_EOF:
							return( (*ok!= EOF) );
							break;
						case NOT_EOF_OR_RETURN:
							return( (*ok== EOF)? 0 : *ok );
							break;
						default:
							return( *ok);
							break;
					}
				}
			}
#ifdef DEBUG_EXTRA
			else{
				if( ascanf_verbose ){
					fprintf( StdErr, "!%s(%lx,%lx)", af->name, hash, af->name );
					fflush( StdErr );
				}
			}
#endif
			af= af->cdr;
		}
	}
	return(0);
}

int show_ascanf_functions( FILE *fp, char *prefix)
{  int i;
   ascanf_Function *af;
	if( !fp)
		return(0);
	fprintf( fp, "%s*** ascanf functions:\n", prefix);
	for( i= 0; i< ascanf_Functions && ascanf_FunctionList; i++){
		af= &ascanf_FunctionList[i];
		while( af){
			af->name_length= strlen( af->name );
			af->hash= ascanf_hash( af->name );
			fprintf( fp, "%s%s\t[%d]\t(%d,%lx)\n", prefix, (af->usage)? af->usage : af->name,
				af->args, af->name_length, af->hash
			);
			af= af->cdr;
		}
	}
	return(1);
}

int add_ascanf_functions( ascanf_Function *array, int n)
{  int i;
   ascanf_Function *af= &vars_ascanf_Functions[ascanf_Functions-1];
	for( i= n-2; i>= 0; i--){
		array[i].cdr= &array[i+1];
	}
	array[n-1].cdr= af->cdr;
	af->cdr= array;
	return( n);
}

/* ascanf(n, s, a) (ArrayScanf) functions; read a maximum of <n>
 * values from buffer 's' into array 'a'. Multiple values must be
 * separated by a comma; whitespace is ignored. <n> is updated to
 * contain the number of values actually read; this is also returned
 * unless an error occurs, in which case EOF is returned.
 * NOTE: cascanf(), dascanf() and lascanf() correctly handle mixed decimal and
 * hex values ; hex values must be preceded by '0x'
 */

/* read multiple floating point values	*/
int fascanf( n, s, a, ch, data, column)
int *n;							/* max # of elements in a	*/
char *s;					/* input string	*/
double *a;					/* target array	*/
char *ch;
double data[3];
int column[3];
{	int i= 0, r= 0, j= 1;
	char *q;
	double A;
	static int level= 0;

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}
	if( data )
		ascanf_data_buf= data;
	if( column )
		ascanf_column_buf= column;
	if( !level){
/* 		reset_ascanf_index_value= True;	*/
	}
	else{
		reset_ascanf_index_value= False;
	}
	if( reset_ascanf_index_value ){
		ascanf_index_value= -1;
	}
	level++;
	for( i= 0, q= ascanf_index( s, ascanf_separator); i< *n && j!= EOF && q && *s; a++, i++ ){
		*q= '\0';
		for( ; isspace((unsigned char)*s) && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		if( *s ){
			A= *a;
			if( !strncmp( s, "NaN", 3) ){
				set_NaN( A );
				r+= (j= 1);
			}
			else if( !strncmp( s, "-Inf", 4) ){
				set_Inf( A, -1);
				r+= (j= 1);
			}
			else if( !strncmp( s, "Inf", 3) ){
				set_Inf( A, 1);
				r+= (j= 1);
			}
			else if( check_for_ascanf_function( i, &s, &A, &j, "fascanf") ){
				r+= 1;
			}
			else if( !strncmp( s, "0x", 2) ){
			  long l;
				s+= 2;
				if( !(j= sscanf( s, "%lx", &l))!= EOF){
					r+= j;
					A= (double) l;
				}
			}
			else if( index( s, '/') ){
			  double B;
				if( (j= sscanf( s, FLOFMT"/"FLOFMT, &A, &B ))== 2){
					r+= 1;
					if( B ){
						A= A/ B;
					}
					else{
						set_Inf( A, (A)? A : 1);
					}
				}
			}
			else{
				if( (j= sscanf( s, FLOFMT, &A ))!= EOF)
					r+= j;
			}
			SET_CHANGED_FLAG( ch, i, A, *a, j);
			*a= A;
		}
		*q= ascanf_separator;
		s= q+ 1;
		q= ascanf_index( s, ascanf_separator);
	}
	for( ; isspace((unsigned char)*s) && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s){
		A= *a;
		if( !strncmp( s, "NaN", 3) ){
			set_NaN( A);
			r+= 1;
		}
		else if( !strncmp( s, "-Inf", 4) ){
			set_Inf( A, -1);
			r+= 1;
		}
		else if( !strncmp( s, "Inf", 3) ){
			set_Inf( A, 1);
			r+= 1;
		}
		else if( check_for_ascanf_function( i, &s, &A, &j, "fascanf") ){
			r+= 1;
		}
		else if( !strncmp( s, "0x", 2) ){
		  long l;
			s+= 2;
			if( !(j= sscanf( s, "%lx", &l))!= EOF){
				r+= j;
				A= (double) l;
			}
		}
		else if( index( s, '/') ){
		  double B;
			if( (j= sscanf( s, FLOFMT"/"FLOFMT, &A, &B ))== 2){
				r+= 1;
				if( B ){
					A= A/ B;
				}
				else{
					set_Inf( A, (A)? A : 1);
				}
			}
		}
		else{
			if( (j= sscanf( s, FLOFMT, &A ))!= EOF)
				r+= j;
		}
		SET_CHANGED_FLAG( ch, i, A, *a, j);
		*a++= A;
	}
	level--;
	if( r< *n){
		*n= r;					/* not enough read	*/
		return( EOF);				/* so return EOF	*/
	}
	return( r);
}

#ifdef ALL_ASCANF_VARIANTS

/* read multiple characters (bytes)	*/
int cascanf( n, s, a, ch)
int *n;
char *s;
char *a, *ch;
{	int r= 0, i= 0, j= 1;
	char *q;
	int A;
	static int level= 0;

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}
	if( !level){
		reset_ascanf_index_value= True;
	}
	else{
		reset_ascanf_index_value= False;
	}
	if( reset_ascanf_index_value ){
		ascanf_index_value= -1;
	}
	level++;
	for( i= 0, q= ascanf_index( s, ascanf_separator); i< *n && j!= EOF && q && *s; a++, i++){
		/* temporarily discard the rest of the buffer	*/
		*q= '\0';
		/* skip whitespace	*/
		for( ; *s== ' ' || *s== '\t' && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		if( *s ){
		  double B= (double) *a;
			A= (int) *a;
			if( !strncmp( s, "0x", 2) ){
				s+= 2;
				j= sscanf( s, "%x", &A);
			}
			else if( check_for_ascanf_function( i, &s, &B, &j, "cascanf") ){
				r+= 1;
				A= d2int(B);
			}
			else
				j= sscanf( s, "%d", &A);
			if( j!= EOF)
				r+= j;
			SET_CHANGED_FLAG( ch, i, A, *a, j);
			*a= (char) A;
		}
		*q= ascanf_separator;					/* restore	*/
		s= q+ 1;					/* next number ?	*/
		q= ascanf_index( s, ascanf_separator);			/* end of next number	*/
	}
	for( ; *s== ' ' || *s== '\t' && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s){
	  double B= (double) *a;
		A= (int) *a;
		if( !strncmp( s, "0x", 2) ){
			s+= 2;
			j= sscanf( s, "%x", &A);
		}
		else if( check_for_ascanf_function( i, &s, &B, &j, "cascanf") ){
			r+= 1;
			A= d2int(B);
		}
		else
			j= sscanf( s, "%d", &A);
		SET_CHANGED_FLAG( ch, i, A, *a, j);
		*a++= A;
		if( j!= EOF)
			r+= j;
	}
	level--;
	if( r< *n){
		*n= r;
		return( EOF);
	}
	return( r);
}

#ifndef MCH_AMIGA
#	define SHORT_DECI	"%hd"
#	define SHORT_HEXA	"%hx"
#else
#	define SHORT_DECI	"%d"
#	define SHORT_HEXA	"%x"
#endif

/* read multiple 16 bits integers (dec or hex)	*/
int dascanf( n, s, a, ch)
int *n;
char *s;
short *a;
char *ch;
{	int r= 0, i= 0, j= 1;
	char *q;
	short A;
	static int level= 0;

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}
	if( !level){
		reset_ascanf_index_value= True;
	}
	else{
		reset_ascanf_index_value= False;
	}
	if( reset_ascanf_index_value ){
		ascanf_index_value= -1;
	}
	level++;
	for( i= 0, q= ascanf_index( s, ascanf_separator); i< *n && j!= EOF && q && *s; a++, i++){
		*q= '\0';
		/* skip whitespace	*/
		for( ; *s== ' ' || *s== '\t' && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		if( *s ){
		  double B= (double) *a;
			A= (short) *a;
			if( !strncmp( s, "0x", 2) ){
				s+= 2;
				j= sscanf( s, SHORT_HEXA, &A);
			}
			else if( check_for_ascanf_function( i, &s, &B, &j, "dascanf") ){
				r+= 1;
				A= d2short( B);
			}
			else
				j= sscanf( s, SHORT_DECI, &A);
			if( j!= EOF)
				r+= j;
			SET_CHANGED_FLAG( ch, i, A, *a, j);
			*a= A;
		}
		*q= ascanf_separator;
		s= q+ 1;
		q= ascanf_index( s, ascanf_separator);
	}
	for( ; *s== ' ' || *s== '\t' && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s){
	  double B= (double) *a;
		A= (short) *a;
		if( !strncmp( s, "0x", 2) ){
			s+= 2;
			j= sscanf( s, SHORT_HEXA, &A );
		}
		else if( check_for_ascanf_function( i, &s, &B, &j, "dascanf") ){
			r+= 1;
			A= d2short( B);
		}
		else
			j= sscanf( s, SHORT_DECI, &A );
		if( j!= EOF)
			r+= j;
		SET_CHANGED_FLAG( ch, i, A, *a, j);
		*a++= A;
	}
	level--;
	if( r< *n){
		*n= r;
		return( EOF);
	}
	return( r);
}

/* read multiple 32 bits long integers (dec or hex)	*/
int lascanf( n, s, a, ch)
int *n;
char *s;
long *a;
char *ch;
{	int r= 0, i= 0, j= 1;
	char *q;
	long A;
	static int level= 0;

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}
	if( !level){
		reset_ascanf_index_value= True;
	}
	else{
		reset_ascanf_index_value= False;
	}
	if( reset_ascanf_index_value ){
		ascanf_index_value= -1;
	}
	level++;
	for( i= 0, q= ascanf_index( s, ascanf_separator); i< *n && j!= EOF && q && *s; a++, i++){
		*q= '\0';
		for( ; *s== ' ' || *s== '\t' && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		if( *s ){
		  double B= (double) *a;
			A= (long) *a;
			if( !strncmp( s, "0x", 2) ){
				s+= 2;
				j= sscanf( s, "%lx", &A );
			}
			else if( check_for_ascanf_function( i, &s, &B, &j, "lascanf") ){
				r+= 1;
				A= (long) B;
			}
			else
				j= sscanf( s, "%ld", &A );
			if( j!= EOF)
				r+= j;
			SET_CHANGED_FLAG( ch, i, A, *a, j);
			*a= A;
		}
		*q= ascanf_separator;
		s= q+ 1;
		q= ascanf_index( s, ascanf_separator);
	}
	for( ; *s== ' ' || *s== '\t' && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s){
	  double B= (double) *a;
		A= (long) *a;
		if( !strncmp( s, "0x", 2) ){
			s+= 2;
			j= sscanf( s, "%lx", &A );
		}
		else if( check_for_ascanf_function( i, &s, &B, &j, "lascanf") ){
			r+= 1;
			A= (long) B;
		}
		else
			j= sscanf( s, "%ld", &A );
		if( j!= EOF)
			r+= j;
		SET_CHANGED_FLAG( ch, i, A, *a, j);
		*a++= A;
	}
	level--;
	if( r< *n){
		*n= r;
		return( EOF);
	}
	return( r);
}

/* read multiple 32 bits long hexadecimal integers */
int xascanf( n, s, a, ch)
int *n;
char *s;
long *a;
char *ch;
{	int r= 0, i= 0, j= 1;
	char *q;
	long A;
	static int level= 0;

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}
	if( !level){
		reset_ascanf_index_value= True;
	}
	else{
		reset_ascanf_index_value= False;
	}
	if( reset_ascanf_index_value ){
		ascanf_index_value= -1;
	}
	level++;
	for( i= 0, q= ascanf_index( s, ascanf_separator); i< *n && j!= EOF && q && *s; a++, i++){
		*q= '\0';
		for( ; *s== ' ' || *s== '\t' && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		if( *s ){
		  double B= (double) *a;
			A= (long) *a;
			if( check_for_ascanf_function( i, &s, &B, &j, "xascanf") ){
				r+= 1;
				A= (long) B;
			}
			else if( (j= sscanf( s, "0x%lx", &A ))!= EOF)
				r+= j;
			SET_CHANGED_FLAG( ch, i, A, *a, j);
			*a= A;
		}
		*q= ascanf_separator;
		s= q+ 1;
		q= ascanf_index( s, ascanf_separator);
	}
	for( ; *s== ' ' || *s== '\t' && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s){
	  double B= (double) *a;
		A= (long) *a;
		if( check_for_ascanf_function( i, &s, &B, &j, "xascanf") ){
			r+= 1;
			A= (long) B;
		}
		else if( ( j= sscanf( s, "0x%lx", &A ))!= EOF)
			r+= j;
		SET_CHANGED_FLAG( ch, i, A, *a, j);
		*a++= A;
	}
	level--;
	if( r< *n){
		*n= r;
		return( EOF);
	}
	return( r);
}

/* read multiple floating point values: floats	*/
/* since this function is younger than its brother that
 \ reads doubles, and since that brother was named
 \ fascanf(), this one had to be called hfascanf().
 \ After all, if a double(float) is called a float (f),
 \ than a float - being half a double - is a halffloat (hf)
 */
int hfascanf( n, s, a, ch)
int *n;							/* max # of elements in a	*/
char *s;					/* input string	*/
float *a;					/* target array	*/
char *ch;
{	int i= 0, r= 0, j= 1;
	char *q;
	double A;
	static int level= 0;

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}
	if( !level){
		reset_ascanf_index_value= True;
	}
	else{
		reset_ascanf_index_value= False;
	}
	if( reset_ascanf_index_value ){
		ascanf_index_value= -1;
	}
	level++;
	for( i= 0, q= ascanf_index( s, ascanf_separator); i< *n && j!= EOF && q && *s; a++, i++ ){
		*q= '\0';
		for( ; *s== ' ' || *s== '\t' && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		if( *s ){
			A= (double) *a;
			if( !strncmp( s, "NaN", 3) ){
				set_NaN( A );
				r+= (j= 1);
			}
			else if( !strncmp( s, "-Inf", 4) ){
				set_Inf( A, -1);
				r+= (j= 1);
			}
			else if( !strncmp( s, "Inf", 3) ){
				set_Inf( A, 1);
				r+= (j= 1);
			}
			else if( check_for_ascanf_function( i, &s, &A, &j, "fascanf") ){
				r+= 1;
			}
			else if( !strncmp( s, "0x", 2) ){
			  long l;
				s+= 2;
				if( !(j= sscanf( s, "%lx", &l))!= EOF){
					r+= j;
					A= (double) l;
				}
			}
			else if( index( s, '/') ){
			  double B;
				if( (j= sscanf( s, FLOFMT"/"FLOFMT, &A, &B ))== 2){
					r+= 1;
					if( B ){
						A= A/ B;
					}
					else{
						set_Inf( A, (A)? A : 1);
					}
				}
			}
			else{
				if( (j= sscanf( s, FLOFMT, &A ))!= EOF)
					r+= j;
			}
			SET_CHANGED_FLAG( ch, i, A, (double) *a, j);
			*a= (float) A;
		}
		*q= ascanf_separator;
		s= q+ 1;
		q= ascanf_index( s, ascanf_separator);
	}
	for( ; *s== ' ' || *s== '\t' && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s){
		A= (double) *a;
		if( !strncmp( s, "NaN", 3) ){
			set_NaN( A);
			r+= 1;
		}
		else if( !strncmp( s, "-Inf", 4) ){
			set_Inf( A, -1);
			r+= 1;
		}
		else if( !strncmp( s, "Inf", 3) ){
			set_Inf( A, 1);
			r+= 1;
		}
		else if( check_for_ascanf_function( i, &s, &A, &j, "fascanf") ){
			r+= 1;
		}
		else if( !strncmp( s, "0x", 2) ){
		  long l;
			s+= 2;
			if( !(j= sscanf( s, "%lx", &l))!= EOF){
				r+= j;
				A= (double) l;
			}
		}
		else if( index( s, '/') ){
		  double B;
			if( (j= sscanf( s, FLOFMT"/"FLOFMT, &A, &B ))== 2){
				r+= 1;
				if( B ){
					A= A/ B;
				}
				else{
					set_Inf( A, (A)? A : 1);
				}
			}
		}
		else{
			if( (j= sscanf( s, FLOFMT, &A ))!= EOF)
				r+= j;
		}
		SET_CHANGED_FLAG( ch, i, A, (double) *a, j);
		*a++= (float) A;
	}
	level--;
	if( r< *n){
		*n= r;					/* not enough read	*/
		return( EOF);				/* so return EOF	*/
	}
	return( r);
}

/* read multiple values in an array of pointers to a basic type	*/
int ascanf( type, n, s, a, ch)
ObjectTypes type;
int *n;
char *s;
char **a;
char *ch;
{	int N, i= 0, r= 0, j= 1;
	char *q;
	VariableType A;
	VariableSet B;

	if( !a || !s || !*s){
		*n= 0;
		return( EOF);
	}
	for( i= 0, q= ascanf_index( s, ascanf_separator); i< *n && j!= EOF && q && *s; a++, i++ ){
		*q= '\0';
		for( ; *s== ' ' || *s== '\t' && *s; s++);
		RESET_CHANGED_FLAG( ch, i);
		if( *s && *a ){
			N= 1;
			A.ptr= (void*) *a;
			memcpy( &B, *a, sizeof(VariableSet) );
			switch( type){
				case _char:
					r+= (j= cascanf( &N, s, A.c, (ch)? &ch[i] : NULL ));
					SET_CHANGED_FLAG( ch, i, *A.c, B.c, j);
					break;
				case _short:
					r+= (j= dascanf( &N, s, A.s, (ch)? &ch[i] : NULL ));
					SET_CHANGED_FLAG( ch, i, *A.s, B.s, j);
					break;
				case _int:
					if( sizeof(int)== 2)
						r+= (j= dascanf( &N, s, (short*) A.i, (ch)? &ch[i] : NULL ));
					else
						r+= (j= lascanf( &N, s, (long*) A.i, (ch)? &ch[i] : NULL ));
					SET_CHANGED_FLAG( ch, i, *A.i, B.i, j);
					break;
				case _long:
					r+= (j= lascanf( &N, s, A.l, (ch)? &ch[i] : NULL ));
					SET_CHANGED_FLAG( ch, i, *A.l, B.l, j);
					break;
				case _float:
					r+= (j= hfascanf( &N, s, A.f, (ch)? &ch[i] : NULL ));
					SET_CHANGED_FLAG( ch, i, *A.f, B.f, j);
					break;
				case _double:
					r+= (j= fascanf( &N, s, A.d, (ch)? &ch[i] : NULL ));
					SET_CHANGED_FLAG( ch, i, *A.d, B.d, j);
					break;
				default:
					if( type< MaxObjectType) 
						fprintf( StdErr, "ascanf::ascanf(): illegal type %s\n",
							ObjectTypeNames[type]
						);
					else
						fprintf( StdErr, "ascanf::ascanf(): illegal type %d\n",
							type 
						);
					*n= 0;
					reset_ascanf_index_value= True;
					return(0);
						
			}
		}
		*q= ascanf_separator;
		s= q+ 1;
		q= ascanf_index( s, ascanf_separator);
	}
	for( ; *s== ' ' || *s== '\t' && *s; s++);
	RESET_CHANGED_FLAG( ch, i);
	if( !q && i< *n && *s && *a){
		N= 1;
		A.ptr= (void*) *a;
		memcpy( &B, *a, sizeof(VariableSet) );
		switch( type){
			case _char:
				r+= (j= cascanf( &N, s, A.c, (ch)? &ch[i] : NULL ));
				SET_CHANGED_FLAG( ch, i, *A.c, B.c, j);
				break;
			case _short:
				r+= (j= dascanf( &N, s, A.s, (ch)? &ch[i] : NULL ));
				SET_CHANGED_FLAG( ch, i, *A.s, B.s, j);
				break;
			case _int:
				if( sizeof(int)== 2)
					r+= (j= dascanf( &N, s, (short*) A.i, (ch)? &ch[i] : NULL ));
				else
					r+= (j= lascanf( &N, s, (long*) A.i, (ch)? &ch[i] : NULL ));
				SET_CHANGED_FLAG( ch, i, *A.i, B.i, j);
				break;
			case _long:
				r+= (j= lascanf( &N, s, A.l, (ch)? &ch[i] : NULL ));
				SET_CHANGED_FLAG( ch, i, *A.l, B.l, j);
				break;
			case _float:
				r+= (j= hfascanf( &N, s, A.f, (ch)? &ch[i] : NULL ));
				SET_CHANGED_FLAG( ch, i, *A.f, B.f, j);
				break;
			case _double:
				r+= (j= fascanf( &N, s, A.d, (ch)? &ch[i] : NULL ));
				SET_CHANGED_FLAG( ch, i, *A.d, B.d, j);
				break;
			default:
				if( type< MaxObjectType) 
					fprintf( StdErr, "ascanf::ascanf(): illegal type %s\n",
						ObjectTypeNames[type]
					);
				else
					fprintf( StdErr, "ascanf::ascanf(): illegal type %d\n",
						type 
					);
				*n= 0;
				reset_ascanf_index_value= True;
				return(0);
					
		}
	}
	if( r< *n){
		*n= r;					/* not enough read	*/
		return( EOF);				/* so return EOF	*/
	}
	return( r);
}

#endif	/* ALL_ASCANF_VARIANTS	*/
