#include <stdio.h>
#include <math.h>

#include "Macros.h"
#include "ascanf.h"

#include <fftw.h>
#include <rfftw.h>

int FFTW_Initialised= False;
char *FFTW_wisdom= NULL;

#define StdErr	stderr

double *ascanf_verbose= 1;
int ascanf_arguments, ascanf_arg_error;
char *ascanf_emsg= NULL;

#undef xfree
#define xfree(v)	if(v){free(v);}

int ascanf_InitFFTW()
{ char *home= getenv( "HOME" ), *c, wise_file[512];
  FILE *fp;
  char *w;

	if( FFTW_Initialised ){
		return(1);
	}
	wise_file[511]= '\0';
	if( (c= getenv( "XG_WISDOM")) ){
		strncpy( wise_file, c, 511 );
	}
	else{
		sprintf( wise_file, "%s/.Preferences/.xgraph/fft_wisdom", home );
	}
	if( (fp= fopen( wise_file, "r" )) ){
#if defined(HAVE_FFTW) && HAVE_FFTW
		if( fftw_import_wisdom_from_file(fp)!= FFTW_SUCCESS ){
			fprintf( StdErr, "ascanf_InitRFFTW(): read error on wisdom file %s: %s\n", wise_file, serror() );
			fclose(fp);
			return(0);
		}
#else
		fprintf( StdErr, "ascanf_InitRFFTW(): you need to compile with -DHAVE_FFTW; fftw routines will not compute\n" );
#endif
		fclose(fp);
#if defined(HAVE_FFTW) && HAVE_FFTW
		if( FFTW_wisdom ){
			xfree( FFTW_wisdom );
		}
		w= fftw_export_wisdom_to_string();
		FFTW_wisdom= strdup(w);
		fftw_free(w);
#endif
	}
	else if( errno!= ENOENT ){
		fprintf( StdErr, "ascanf_InitRFFTW(): read error on wisdom file %s: %s\n", wise_file, serror() );
		return(0);
	}
	FFTW_Initialised= True;
	return(1);
}

int ascanf_CloseFFTW()
{ char *home= getenv( "HOME" ), *c, wise_file[512];
  FILE *fp;
  char *w= NULL;
  int complete;

	if( !FFTW_Initialised ){
		return(1);
	}
	wise_file[511]= '\0';
	if( (c= getenv( "XG_WISDOM")) ){
		strncpy( wise_file, c, 511 );
	}
	else{
		sprintf( wise_file, "%s/.Preferences/.xgraph/fft_wisdom", home );
	}
#if defined(HAVE_FFTW) && HAVE_FFTW
	complete= 0;
	if( (w= fftw_export_wisdom_to_string()) && FFTW_wisdom ){
		if( !strcmp( w, FFTW_wisdom) ){
			fftw_free(w);
			if( complete ){
				fftw_forget_wisdom();
				xfree( FFTW_wisdom );
				FFTW_Initialised= False;
			}
			return(1);
		}
	}
	if( (fp= fopen( wise_file, "w" )) ){
		if( *ascanf_verbose ){
			fprintf( StdErr, "(stored increased wisdom in %s)== ", wise_file );
		}
		fftw_export_wisdom_to_file(fp);
		fclose(fp);
	}
	else{
		fprintf( StdErr, "ascanf_CloseRFFTW(): write error on wisdom file %s: %s\n", wise_file, serror() );
		return(0);
	}
	xfree( FFTW_wisdom );
	if( complete ){
		fftw_forget_wisdom();
		FFTW_Initialised= False;
	}
	else{
		FFTW_wisdom= strdup(w);
	}
#else
	fprintf( StdErr, "ascanf_CloseRFFTW(): no FFTW, no wisdom to store...\n" );
#endif
	return(1);
}

#if defined(HAVE_FFTW) && HAVE_FFTW
#if defined(DEBUG) && defined(sgi)
#	include <fft.h>
#endif
#endif

#define USE_RFFTWND

#define USE_WISDOM

int ascanf_rfftw( ASCB_ARGLIST, int N, double *input, double **output, double **power )
{ ASCB_FRAME
  int i;
#if defined(HAVE_FFTW) && HAVE_FFTW
  rfftwnd_plan p;
#endif
#if DEBUG==2
  double *sgcoeff= NULL;
#endif
	*result= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
	  int N= 0, NN= 0;
		NN= 2* (N/ 2+ 1);
		if( !input || !output ){
			ascanf_emsg= "(empty input or output)";
			ascanf_arg_error= 1;
			return(0);
		}
		xfree( *output );
		if( !(*output= (double*) calloc( NN, sizeof(double))) ){
			ascanf_emsg= " (can't get memory for output) ";
			ascanf_arg_error= 1;
		}
		if( power ){
			xfree( *power );
			if( !(*power= (double*) calloc( NN, sizeof(double))) ){
				ascanf_emsg= " (can't get memory for powerspectrum array) ";
				ascanf_arg_error= 1;
			}
			else{
				memset( *power, 0, sizeof(double)* NN );
			}
		}
		if( ascanf_arg_error || !N || !*output || (power && !*power) ){
			return(0);
		}

#if defined(HAVE_FFTW) && HAVE_FFTW
#ifdef USE_WISDOM
		if( FFTW_Initialised ){
			p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE|FFTW_USE_WISDOM );
		}
		else{
			p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_USE_WISDOM );
		}
#else
		if( FFTW_Initialised ){
			p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE );
		}
		else{
			p= rfftw2d_create_plan( 1, N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE );
		}
#endif

		{ fftw_real *in= (fftw_real*) input;
		  fftw_complex *out= (fftw_complex*) *output;
			rfftwnd_one_real_to_complex( p, in, out );
		}

		  /* phase can also be calculated: phase= arctan( imag/real )	*/
		if( power ){
		  double *pa= *power, scale= N*N;
		  fftw_complex *out= (fftw_complex*) *output;
			for( i= 0; i<= N/ 2; i++ ){
				pa[i]= (out[i].re*out[i].re + out[i].im*out[i].im)/ scale;
			}
		}

		rfftwnd_destroy_plan(p);
#endif

		*result= NN;
		return(1);
	}
}

main()
{ double data[768], *output= NULL, result;
  int i, level= 0;
  ascanf_Callback_Frame frame;

	for( i= 0; i< sizeof(data)/sizeof(double); i++ ){
		data[i]= (i>=310 && i<=457)? 1 : 0;
	}
	frame.args= data;
	frame.result= &result;
	frame.level= &level;
	frame.self= NULL;
#ifdef DEBUG
	frame.expr= "youhou ;)";
#endif

	ascanf_InitFFTW();
	ascanf_rfftw( &frame, sizeof(data)/sizeof(double), data, &output, NULL );
	if( output ){
	  int nn= 0;
		for( i= 0; i< sizeof(data)/sizeof(double); i++ ){
			if( NaN(output[i]) ){
				nn+= 1;
			}
		}
		if( nn ){
			fprintf( StdErr, "%d NaNs in rfftw result: error!\n", nn );
		}
		free( output );
	}
	ascanf_CloseFFTW();
	exit(0);
}
