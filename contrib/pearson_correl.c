#include "config.h"
IDENTIFY( "Pearson correlation ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
#include "new_ps.h"
#include "xtb/xtb.h"

#include "NaN.h"

#include "xgraph.h"
#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
#include "ascanfc-table.h"
#include "compiled_ascanf.h"

#define SIGN(x)		(((x)<0)?-1:1)
#ifndef MIN
#	define MIN(a, b)	((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#	define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif

#include "simanneal.h"

#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;
	double (*normal_rand_ptr)( int i, double av, double stdv );
	int *DrawAllSets_ptr;
	double *radix_ptr;
	int (*fascanf_ptr)( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], struct Compiled_Form **form );
	int (*Destroy_Form_ptr)( struct Compiled_Form **form );

#	define DrawAllSets	(*DrawAllSets_ptr)
#	define normal_rand	(*normal_rand_ptr)
#	define fascanf	(*fascanf_ptr)
#	define Destroy_Form	(*Destroy_Form_ptr)

void (*continuous_simanneal_ptr)(double **p, double y[], int ndim, double pb[], double *yb, double ftol, double (*funk) (double []) , int *iter, double T);
void (*sran1_ptr)( long seed );
#define continuous_simanneal	(*continuous_simanneal_ptr)
#define sran1	(sran1_ptr)

extern double *pc_normalise_orientation, *pc_normalise_spatial, *pc_absolute_orientation, *pc_pointwise;

extern double *pc_transform_pearcorr, *pc_transform_pearcorr_param;

static void print_settings(FILE *fp);

/* Pearson's correlation index between a series Xi and Yi:
 \  Sum( (Xi- Xmean)(Yi- Ymean) ) / ( sqrt( Sum( (Xi-Xmean)**2 ) ) * sqrt( Sum( (Yi-Ymean)**2 ) ) )
 \ The upper part can be simplified as
 \  Sum(Xi*Yi)- Sum(Xi)*Sum(Yi)/N
 \ The lower part as
 \  sqrt( Sum(Xi**2) - Sum(Xi)**2/N ) * sqrt( Sum(Yi**2) - Sum(Yi)**2/N )
 \ Thus, one needs a single pass only, calculating:
 \  Sum(Xi*Yi), Sum(Xi), Sum(Yi), Sum(Xi**2) and Sum(Yi**2)
 */

static double *pc_noise_percentage, *pc_formal;

#define drand()	drand48()

/* Lowlevel routine. Does not do any error checking! */
double drand_in( double min, double max)
{ double rng= (max-min);

	if( rng ){
		return( drand()* rng+ min );
	}
	else{
		return( min );
	}
}

double nrand( double av, double stdv )
{ double r= normal_rand( ASCANF_MAX_ARGS-1, av, stdv );
	if( r== 0 && ascanf_arg_error ){
		r= av+ (drand()- 0.5)* stdv;
		ascanf_arg_error= 0;
	}
	return(r);
}

static double *pc_xcol, *pc_ycol, *pc_ecol;
static double *pc_transformation;
double *pc_transformation_parameters= NULL, _pc_transformation_parameters[2]= {1, 0};

#define TRANSFORM if( pt ){ \
	  double dx, dy, de; \
		switch( pt ){ \
			case 1:{ \
				dx= dy= de= pc_transformation_parameters[0]* k + pc_transformation_parameters[1]; \
				break; \
			} \
			case 2:{ \
				dx= dy= de= pc_transformation_parameters[0]* sin( k* M_2PI/(N-1) ) + pc_transformation_parameters[1]; \
				break; \
			} \
			case 3:{ \
				dx= nrand( pc_transformation_parameters[1], pc_transformation_parameters[0] ); \
				dy= nrand( pc_transformation_parameters[1], pc_transformation_parameters[0] ); \
				de= nrand( pc_transformation_parameters[1], pc_transformation_parameters[0] ); \
				break; \
			} \
		} \
		x1+= dx, y1+= dy, o1+= de; \
		x2+= dx, y2+= dy, o2+= de; \
	} \

/* A remedy against a constant "dimension" in one of the sets - constant X, Y or E/O - can be quite simply
 \ to add a similar non-constant value to both sets. That is, adding a straight, non-horizontal line to
 \ to straight lines won't influence the correlation between them. Similarly, adding e.g. a single period
 \ of a sine function (with a period equal to the number of points in the sets!) won't influence the result
 \ as long as the amplitude is small with respect to the data range.
 \ These 2 options can be hardcoded, like the transformations in fig_dist.so . If necessary, one can also
 \ provide hooks to pass 3 ascanf procedures that will be evaluated for X, Y and/or E/O, with all the
 \ usual variables correctly initialised (both are transparent extensions that do not require a modification
 \ of the calling convention).
 */

static double pearson_correl_formal( LocalWin *wi, DataSet *set1, DataSet *set2,
	double *spatial, double *angular, int *K, int raw1, int raw2 )
{ int k;
  double mX1= 0, mX2= 0;
  double mY1= 0, mY2= 0;
  double mO1= 0, mO2= 0;
  double sX12= 0, sX1_2= 0, sX2_2= 0;
  double sY12= 0, sY1_2= 0, sY2_2= 0;
  double sO12= 0, sO1_2= 0, sO2_2= 0;
  double d1, d2, cx, cy, co;
  int N= MIN(set1->numPoints, set2->numPoints), na1= False, na2= False, pt= *pc_transformation;
  double pnc= *pc_noise_percentage/100.0;

	*K= 0;

	for( k= 0; k< N; k++ ){
		if( !DiscardedPoint( wi, set1, k) && !DiscardedPoint( wi, set2, k) ){
		  double x1, y1, o1;
		  double x2, y2, o2;
			if( raw1 ){
				x1= XVAL( set1, k);
				y1= YVAL( set1, k);
				o1= ERROR( set1, k);
			}
			else{
				x1= set1->xvec[k];
				y1= set1->yvec[k];
				o1= set1->errvec[k];
			}
			if( raw2 ){
				x2= XVAL( set2, k);
				y2= YVAL( set2, k);
				o2= ERROR( set2, k);
			}
			else{
				x2= set2->xvec[k];
				y2= set2->yvec[k];
				o2= set2->errvec[k];
			}
			TRANSFORM;
			mX1+= x1, mX2+= x2;
			mY1+= y1, mY2+= y2;
			mO1+= o1, mO2+= o2;
			*K+= 1;
		}
	}
	mX1/= *K, mX2/= *K;
	mY1/= *K, mY2/= *K;
	mO1/= *K, mO2/= *K;
	*K= 0;
	for( k= 0; k< N; k++ ){
		if( !DiscardedPoint( wi, set1, k) && !DiscardedPoint( wi, set2, k) ){
		  double x1, y1, o1;
		  double x2, y2, o2;
			if( raw1 ){
				x1= XVAL( set1, k);
				y1= YVAL( set1, k);
				o1= ERROR( set1, k);
			}
			else{
				x1= set1->xvec[k];
				y1= set1->yvec[k];
				o1= set1->errvec[k];
			}
			if( raw2 ){
				x2= XVAL( set2, k);
				y2= YVAL( set2, k);
				o2= ERROR( set2, k);
			}
			else{
				x2= set2->xvec[k];
				y2= set2->yvec[k];
				o2= set2->errvec[k];
			}
			TRANSFORM;
			if( pnc ){
/* 				if( !na1 && (k== N-1 || drand()< 1.0/N) )	*/
				{
/* 					x1+= (drand() - 0.5)* *pc_noise_percentage* x1/ 100.0;	*/
/* 					y1+= (drand() - 0.5)* *pc_noise_percentage* y1/ 100.0;	*/
/* 					o1+= (drand() - 0.5)* *pc_noise_percentage* o1/ 100.0;	*/
					x1= nrand( x1, pnc* x1 );
					y1= nrand( y1, pnc* y1 );
					o1= nrand( o1, pnc* o1 );
					na1= True;
				}
/* 				if( !na2 && (k== N-1 || drand()< 1.0/N) )	*/
				{
/* 					x2+= (drand() - 0.5)* *pc_noise_percentage* x2/ 100.0;	*/
/* 					y2+= (drand() - 0.5)* *pc_noise_percentage* y2/ 100.0;	*/
/* 					o2+= (drand() - 0.5)* *pc_noise_percentage* o2/ 100.0;	*/
					x2= nrand( x2, pnc* x2 );
					y2= nrand( y2, pnc* y2 );
					o2= nrand( o2, pnc* o2 );
					na2= True;
				}
			}
			d1= x1-mX1, d2= x2-mX2;
			sX12+= d1*d2;
			sX1_2+= d1*d1, sX2_2+= d2*d2;
			d1= y1-mY1, d2= y2-mY2;
			sY12+= d1*d2;
			sY1_2+= d1*d1, sY2_2+= d2*d2;
			d1= o1-mO1, d2= o2-mO2;
			sO12+= d1*d2;
			sO1_2+= d1*d1, sO2_2+= d2*d2;
			*K+= 1;
		}
	}
	cx= (*pc_xcol)? sX12/ ( sqrt(sX1_2) * sqrt(sX2_2) ) : 1;
	cy= (*pc_ycol)? sY12/ ( sqrt(sY1_2) * sqrt(sY2_2) ) : 1;
	co= (*pc_ecol)? sO12/ ( sqrt(sO1_2) * sqrt(sO2_2) ) : 1;

	  /* XXX The simple product is not the best of ideas: when both factors are <0, the resulting correlation should
	   \ not (by miracle :)) become positive! Better is maybe cx*cy* (cx+cy)/2, or a product that is negative
	   \ when 1 or both of the factors are. In the former case, the *pc_xcol/ycol/ecol tests should be adapted.
	   */
	*spatial= (*pc_xcol* cx + *pc_ycol* cy)/ (*pc_xcol + *pc_ycol);
	*angular= co;

	if( ascanf_verbose>= 2 ){
		fprintf( StdErr, "Pearson correlation (formal):\n" );
		print_settings(StdErr);
		if( ascanf_verbose> 2 ){
			if( *pc_xcol) fprintf( StdErr,
				"\tmean=%g,%g: cx= sX12 / ( sqrt(sX1_2) * sqrt(sX2_2) ) =\n\t\t"
				"%g, %g, %g = %s\n",
				mX1, mX2,
				sX12, sX1_2, sX2_2, ad2str( cx, d3str_format, NULL)
			);
			if( *pc_ycol) fprintf( StdErr,
				"\tmean=%g,%g: cy= sY12 / ( sqrt(sY1_2) * sqrt(sY2_2) ) =\n\t\t"
				"%g, %g, %g = %s\n",
				mY1, mY2,
				sY12, sY1_2, sY2_2, ad2str( cy, d3str_format, NULL)
			);
			if( *pc_ecol) fprintf( StdErr,
				"\tmean=%g,%g: co= sO12 / ( sqrt(sO1_2) * sqrt(sO2_2) ) =\n\t\t"
				"%g, %g, %g = %s\n",
				mO1, mO2,
				sO12, sO1_2, sO2_2, ad2str( co, d3str_format, NULL)
			);
		}
		else{
			fprintf( StdErr, "\tcx=%g, cy=%g, co=%g\n", cx, cy, co );
		}
	}
	return( (*pc_xcol* cx + *pc_ycol* cy + *pc_ecol* co)/ (*pc_xcol + *pc_ycol + *pc_ecol) );
}

static double pearson_correl( LocalWin *wi, DataSet *set1, DataSet *set2,
	double *spatial, double *angular, int *K, int raw1, int raw2 )
{ int k;
  double sX1i= 0, sX2i= 0, sX12i= 0, sX1_2i= 0, sX2_2i= 0;
  double sY1i= 0, sY2i= 0, sY12i= 0, sY1_2i= 0, sY2_2i= 0;
  double sO1i= 0, sO2i= 0, sO12i= 0, sO1_2i= 0, sO2_2i= 0;
  double cx, cy, co;
  int N= MIN(set1->numPoints, set2->numPoints), na1= False, na2= False, pt= *pc_transformation;
  double pnc= *pc_noise_percentage/100.0;

	*K= 0;

	for( k= 0; k< N; k++ ){
		if( !DiscardedPoint( wi, set1, k) && !DiscardedPoint( wi, set2, k) ){
		  double x1, y1, o1;
		  double x2, y2, o2;
			if( raw1 ){
				x1= XVAL( set1, k);
				y1= YVAL( set1, k);
				o1= ERROR( set1, k);
			}
			else{
				x1= set1->xvec[k];
				y1= set1->yvec[k];
				o1= set1->errvec[k];
			}
			if( raw2 ){
				x2= XVAL( set2, k);
				y2= YVAL( set2, k);
				o2= ERROR( set2, k);
			}
			else{
				x2= set2->xvec[k];
				y2= set2->yvec[k];
				o2= set2->errvec[k];
			}
			TRANSFORM;
			if( pnc ){
/* 				if( !na1 && (k== N-1 || drand()< 1.0/N) )	*/
				{
/* 					x1+= (drand() - 0.5)* *pc_noise_percentage* x1/ 100.0;	*/
/* 					y1+= (drand() - 0.5)* *pc_noise_percentage* y1/ 100.0;	*/
/* 					o1+= (drand() - 0.5)* *pc_noise_percentage* o1/ 100.0;	*/
					x1= nrand( x1, pnc* x1 );
					y1= nrand( y1, pnc* y1 );
					o1= nrand( o1, pnc* o1 );
					na1= True;
				}
/* 				if( !na2 && (k== N-1 || drand()< 1.0/N) )	*/
				{
/* 					x2+= (drand() - 0.5)* *pc_noise_percentage* x2/ 100.0;	*/
/* 					y2+= (drand() - 0.5)* *pc_noise_percentage* y2/ 100.0;	*/
/* 					o2+= (drand() - 0.5)* *pc_noise_percentage* o2/ 100.0;	*/
					x2= nrand( x2, pnc* x2 );
					y2= nrand( y2, pnc* y2 );
					o2= nrand( o2, pnc* o2 );
					na2= True;
				}
			}
			sX1i+= x1, sX2i+= x2;
			sX1_2i+= x1*x1, sX2_2i+= x2*x2;
			sX12i+= x1* x2;
			sY1i+= y1, sY2i+= y2;
			sY1_2i+= y1*y1, sY2_2i+= y2*y2;
			sY12i+= y1* y2;
			sO1i+= o1, sO2i+= o2;
			sO1_2i+= o1*o1, sO2_2i+= o2*o2;
			sO12i+= o1* o2;
			*K+= 1;
		}
	}
	cx= (*pc_xcol)? (sX12i- sX1i*sX2i/ *K) / ( sqrt(sX1_2i - sX1i*sX1i/ *K) * sqrt(sX2_2i - sX2i*sX2i/ *K) ) : 1;
	cy= (*pc_ycol)? (sY12i- sY1i*sY2i/ *K) / ( sqrt(sY1_2i - sY1i*sY1i/ *K) * sqrt(sY2_2i - sY2i*sY2i/ *K) ) : 1;
	co= (*pc_ecol)? (sO12i- sO1i*sO2i/ *K) / ( sqrt(sO1_2i - sO1i*sO1i/ *K) * sqrt(sO2_2i - sO2i*sO2i/ *K) ) : 1;

	  /* XXX The simple product is not the best of ideas: when both factors are <0, the resulting correlation should
	   \ not (by miracle :)) become positive! Better is maybe cx*cy* (cx+cy)/2, or a product that is negative
	   \ when 1 or both of the factors are. In the former case, the *pc_xcol/ycol/ecol tests should be adapted.
	   */
	*spatial= (*pc_xcol* cx + *pc_ycol* cy)/ (*pc_xcol + *pc_ycol);
	*angular= co;

	if( ascanf_verbose>= 2 ){
		fprintf( StdErr, "Pearson correlation:\n" );
		print_settings(StdErr);
		if( ascanf_verbose> 2 ){
			if( *pc_xcol) fprintf( StdErr,
				"\tcx= (sX12i- sX1i*sX2i/ *K) / ( sqrt(sX1_2i - sX1i**2/ *K) * sqrt(sX2_2i - sX2i**2/ *K) ) =\n\t\t"
				"%g, %g, %g, %d, %g, %g, %d, %g, %g, %d = %s\n",
				sX12i, sX1i, sX2i, *K, sX1_2i, sX1i, *K, sX2_2i, sX2i, *K, ad2str( cx, d3str_format, NULL)
			);
			if( *pc_ycol) fprintf( StdErr,
				"\tcy= (sY12i- sY1i*sY2i/ *K) / ( sqrt(sY1_2i - sY1i**2/ *K) * sqrt(sY2_2i - sY2i**2/ *K) ) =\n\t\t"
				"%g, %g, %g, %d, %g, %g, %d, %g, %g, %d = %s\n",
				sY12i, sY1i, sY2i, *K, sY1_2i, sY1i, *K, sY2_2i, sY2i, *K, ad2str( cy, d3str_format, NULL)
			);
			if( *pc_ecol ) fprintf( StdErr,
				"\tco= (sO12i- sO1i*sO2i/ *K) / ( sqrt(sO1_2i - sO1i**2/ *K) * sqrt(sO2_2i - sO2i**2/ *K) ) =\n\t\t"
				"%g, %g, %g, %d, %g, %g, %d, %g, %g, %d = %s\n",
				sO12i, sO1i, sO2i, *K, sO1_2i, sO1i, *K, sO2_2i, sO2i, *K, ad2str( co, d3str_format, NULL)
			);
		}
		else{
			fprintf( StdErr, "\tcx=%g, cy=%g, co=%g\n", cx, cy, co );
		}
	}
	return( (*pc_xcol* cx + *pc_ycol* cy + *pc_ecol* co)/ (*pc_xcol + *pc_ycol + *pc_ecol) );
}

double pearson_correlation( LocalWin *wi, int idx1, int idx2, double *spatial, double *orientation, int raw )
{ int N= 0;
  double ret= (*pc_formal)? pearson_correl_formal( wi, &AllSets[idx1], &AllSets[idx2], spatial, orientation, &N, raw, raw ) :
	  pearson_correl( wi, &AllSets[idx1], &AllSets[idx2], spatial, orientation, &N, raw, raw );
	if( ! *pc_ecol ){
		set_NaN(*orientation);
	}
	return( ret );
}

double pearson_correlation_shiftrot( LocalWin *wi, int idx1, int idx2,
	int scale, double Scale,
	int shift, double shiftX, double shiftY, int rotate, double rotX, double rotY, double angle, double Radix,
	double *spatial, double *orientation, int raw )
{ extern int DrawAllSets;
  int all_vectors, N, i, ds, das, hls;
  double ret;
  double ang_sin, ang_cos;
  DataSet *set2= &AllSets[idx2];
  ALLOCA( txvec, double, set2->numPoints, txvec_len );
  ALLOCA( tyvec, double, set2->numPoints, tyvec_len );
  ALLOCA( terrvec, double, set2->numPoints, terrvec_len );
  double *sxvec= set2->xvec, *syvec= set2->yvec, *serrvec= set2->errvec;
  double rdx= (wi)? wi->radix : *radix_ptr;

	if( wi ){
		if( wi->error_type[idx1]== 4 && wi->error_type[idx2]== 4 ){
			all_vectors= True;
		}
		else{
			all_vectors= False;
		}
	}
	else{
		if( AllSets[idx1].error_type== 4 && AllSets[idx2].error_type== 4 ){
			all_vectors= True;
		}
		else{
			all_vectors= False;
		}
	}
	if( !Radix ){
		Radix= M_2PI;
	}
	if( !shift ){
		shiftX= shiftY= 0;
	}
	if( !rotate ){
		rotX= rotY= angle= 0;
		ang_cos= 1, ang_sin= 0;
	}
	else{
		  /* sine and cosine of the rotation angle: */
		ang_sin= sin( M_2PI * angle/ Radix );
		ang_cos= cos( M_2PI * angle/ Radix );
	}
	if( !scale ){
		Scale= 1;
	}
	for( i= 0; i< set2->numPoints; i++ ){
	  double x, y, o;
		  /* Determine scaled co-ordinates, shifted to put rotation centre in origin: */
		if( raw ){
			x= Scale* (XVAL( set2, i))- rotX;
			y= Scale* (YVAL( set2, i))- rotY;
			o= ERROR( set2, i);
		}
		else{
			x= Scale* (set2->xvec[i])- rotX;
			y= Scale* (set2->yvec[i])- rotY;
			o= set2->errvec[i];
		}
		  /* Determine transformed co-ordinates: rotated, and shifted as requested
		   \ (note that we shift over (rotX,rotY) to cancel the earlier shift)
		   */
		txvec[i]= rotX+ shiftX+ (x* ang_cos- y* ang_sin);
		tyvec[i]= rotY+ shiftY+ (y* ang_cos+ x* ang_sin);
		  /* Also rotate the error column if it represents an orientation: */
		terrvec[i]= (all_vectors)? o+ angle : o;
	}
	  /* Store the temporary transformed co-ordinates in the set's transformed buffer: */
	set2->xvec= txvec;
	set2->yvec= tyvec;
	set2->errvec= terrvec;
	if( wi ){
		ds= wi->draw_set[idx2];
		wi->draw_set[idx2]= 0;
		hls= wi->legend_line[idx2].highlight;
		wi->legend_line[idx2].highlight= 0;
		wi->radix= Radix;
	}
	else{
		*radix_ptr= Radix;
	}
	das= DrawAllSets;
	DrawAllSets= 0;

	if( ascanf_verbose ){
		print_settings( StdErr );
		fprintf( StdErr,
			"\nSet %d vs. %d; %s mode; weights x,y,e=%g,%g,%g, do scale,shift,rotate=%d,%d,%d;\n"
			" shift (%g,%g) rotate (%g,%g,%g) radix %g; scale %g\n",
			idx1, idx2, (raw)? "raw" : "cooked",
			*pc_xcol, *pc_ycol, *pc_ecol,
			scale, shift, rotate,
			shiftX, shiftY, rotX, rotY, angle, Radix, Scale
		);
	}

	  /* Calculate the component(s) of the Pearson correlation: */
	ret= (*pc_formal)? pearson_correl_formal( wi, &AllSets[idx1], set2, spatial, orientation, &N, raw, 0 ) :
		pearson_correl( wi, &AllSets[idx1], set2, spatial, orientation, &N, raw, 0 );
	if( !*pc_ecol ){
		set_NaN(*orientation);
	}
	  /* restore the set's own transformed buffers */
	set2->xvec= sxvec;
	set2->yvec= syvec;
	set2->errvec= serrvec;
	if( wi ){
		wi->draw_set[idx2]= ds;
		wi->legend_line[idx2].highlight= hls;
		wi->radix= rdx;
	}
	else{
		*radix_ptr= rdx;
	}
	DrawAllSets= das;
	GCA();
	return(ret);
}

/* 20010713: From this point onwards, the routines do very largely the same things as their counterparts in
 \ contrib/fig_dist.c ! However, since I want to keep fig_dist.so and pearson_correlation.so independent,
 \ I have preferred to host these double implementations, whereas a specialised library "minimise.so" would
 \ have been possible...
 */

int ascanf_pearson_correl( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx1, idx2, raw;
  ascanf_Function *taf1= NULL, *taf2= NULL;
  double spatial, orientation;
	if( ascanf_arguments>= 3 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			idx1= (int) args[0];
			if( AllSets[idx1].numPoints<= 0 ){
				ascanf_emsg= " (set 1 is empty and/or deleted) ";
				ascanf_arg_error= True;
				idx1= -1;
			}
		}
		else{
			ascanf_emsg= " (setnumber 1 out of range) ";
			ascanf_arg_error= True;
			idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< setNumber || ascanf_SyntaxCheck) ){
			idx2= (int) args[1];
			if( AllSets[idx2].numPoints<= 0 ){
				ascanf_emsg= " (set 2 is empty and/or deleted) ";
				ascanf_arg_error= True;
				idx2= -1;
			}
		}
		else{
			ascanf_emsg= " (setnumber 2 out of range) ";
			ascanf_arg_error= True;
			idx2= -1;
		}
		if( idx1>= 0 && idx2>= 0 && AllSets[idx1].numPoints!= AllSets[idx2].numPoints ){
			ascanf_emsg= " (sets have different numbers of points!) ";
			ascanf_arg_error= True;
		}
		raw= (args[2])? True : False;
		if( ascanf_arguments> 3 ){
			taf1= parse_ascanf_address( args[3], _ascanf_variable, "ascanf_pearson_correl", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 4 ){
			taf2= parse_ascanf_address( args[4], _ascanf_variable, "ascanf_pearson_correl", (int) ascanf_verbose, NULL );
		}
		if( !ascanf_arg_error && idx1>= 0 && idx2>= 0 ){
			*result= pearson_correlation( (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin : NULL, idx1, idx2,
				&spatial, &orientation, raw
			);
			if( taf1 ){
				taf1->value= spatial;
			}
			if( taf2 ){
				taf2->value= orientation;
			}
		}
		else{
			set_NaN(*result);
			if( taf1 ){
				set_NaN(taf1->value);
			}
			if( taf2 ){
				set_NaN(taf2->value);
			}
		}
	}
	else{
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

int ascanf_pearson_correl_shiftrot( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx1, idx2, raw;
  ascanf_Function *taf1= NULL, *taf2= NULL;
  double spatial, orientation, shiftX, shiftY, rotX, rotY, angle, rdx, Scale;
  int shift, rotate;
	if( ascanf_arguments>= 10 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			idx1= (int) args[0];
			if( AllSets[idx1].numPoints<= 0 ){
				ascanf_emsg= " (set 1 is empty and/or deleted) ";
				ascanf_arg_error= True;
				idx1= -1;
			}
		}
		else{
			ascanf_emsg= " (setnumber 1 out of range) ";
			ascanf_arg_error= True;
			idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< setNumber || ascanf_SyntaxCheck) ){
			idx2= (int) args[1];
			if( AllSets[idx2].numPoints<= 0 ){
				ascanf_emsg= " (set 2 is empty and/or deleted) ";
				ascanf_arg_error= True;
				idx2= -1;
			}
		}
		else{
			ascanf_emsg= " (setnumber 2 out of range) ";
			ascanf_arg_error= True;
			idx2= -1;
		}
		if( idx1>= 0 && idx2>= 0 && AllSets[idx1].numPoints!= AllSets[idx2].numPoints ){
			ascanf_emsg= " (sets have different numbers of points!) ";
			ascanf_arg_error= True;
		}
		raw= (args[2])? True : False;
		Scale= args[3];
		shiftX= args[4];
		shiftY= args[5];
		rotX= args[6];
		rotY= args[7];
		angle= args[8];
		rdx= args[9];
		if( ascanf_arguments> 10 ){
			taf1= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_pearson_correl_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 11 ){
			taf2= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_pearson_correl_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( !ascanf_arg_error && idx1>= 0 && idx2>= 0 ){
		  int scale= (NaN(Scale))? False : True;
			shift= (NaN(shiftX) || NaN(shiftY))? False : True;
			rotate= (NaN(rotX) || NaN(rotY) || NaN(angle))? False : True;
			*result= pearson_correlation_shiftrot( (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin : NULL, idx1, idx2,
				scale, Scale, shift, shiftX, shiftY, rotate, rotX, rotY, angle, rdx,
				&spatial, &orientation, raw
			);
			if( taf1 ){
				taf1->value= spatial;
			}
			if( taf2 ){
				taf2->value= orientation;
			}
		}
		else{
			set_NaN(*result);
			if( taf1 ){
				set_NaN(taf1->value);
			}
			if( taf2 ){
				set_NaN(taf2->value);
			}
		}
	}
	else{
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

static int ascanf_setshiftrot( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx1, pnt_nr;
  double shiftX, shiftY, rotX, rotY, angle, rdx, Scale;
  ascanf_Function *rx, *ry, *ro;
	if( ascanf_arguments>= 12 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			idx1= (int) args[0];
			if( AllSets[idx1].numPoints<= 0 ){
				ascanf_emsg= " (set is empty and/or deleted) ";
				ascanf_arg_error= True;
				idx1= -1;
			}
		}
		else{
			ascanf_emsg= " (setnumber out of range) ";
			ascanf_arg_error= True;
			idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< AllSets[idx1].numPoints || ascanf_SyntaxCheck) ){
			pnt_nr= (int) args[1];
		}
		else{
			ascanf_emsg= " (point index out of range) ";
			ascanf_arg_error= True;
			pnt_nr= -1;
		}
		Scale= args[2];
		shiftX= args[3];
		shiftY= args[4];
		rotX= args[5];
		rotY= args[6];
		angle= args[7];
		rdx= args[8];
		rx= parse_ascanf_address( args[9], _ascanf_variable, "ascanf_setshiftrot", (int) ascanf_verbose, NULL );
		ry= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_setshiftrot", (int) ascanf_verbose, NULL );
		ro= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_setshiftrot", (int) ascanf_verbose, NULL );
		if( rx && ry && ro && idx1>= 0 && pnt_nr>= 0 && !ascanf_arg_error ){
		  DataSet *set= &AllSets[idx1];
		  double ang_sin, ang_cos;
		  double x, y, o;

			if( !rdx ){
				rdx= M_2PI;
			}
			if( NaN(Scale) ){
				Scale= 1;
			}
			if( (NaN(shiftX) || NaN(shiftY)) ){
				shiftX= shiftY= 0;
			}
			if( (NaN(rotX) || NaN(rotY) || NaN(angle)) ){
				rotX= rotY= angle= 0;
				ang_cos= 1, ang_sin= 0;
			}
			else{
				ang_sin= sin( M_2PI * angle/ rdx );
				ang_cos= cos( M_2PI * angle/ rdx );
			}
			x= Scale* XVAL( set, pnt_nr)- rotX;
			y= Scale* YVAL( set, pnt_nr)- rotY;
			o= ERROR( set, pnt_nr);
			rx->value= rotX+ shiftX+ (x* ang_cos- y* ang_sin);
			ry->value= rotY+ shiftY+ (y* ang_cos+ x* ang_sin);
			if( ActiveWin && ActiveWin!= StubWindow_ptr ){
				ro->value= (ActiveWin->error_type[idx1]== 4)? o+ angle : o;
			}
			else if( set->error_type== 4 ){
				ro->value= o+ angle;
			}
			else{
				ro->value= 0;
			}
			*result= 1;
		}
	}
	else{
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

static LocalWin *mfg_wi;
static double mfg_spatial, mfg_orientation, mfg_radix, mfg_min, mfg_max;
static int mfg_idx1, mfg_idx2, mfg_raw;
static double fdsr_last_args[8];
static int mfg_shift, mfg_rotate, mfg_scale;
static double mfg_Scale, mfg_shiftX, mfg_shiftY, mfg_rotX, mfg_rotY, mfg_angle;

static double *pc_bound_rotation_centre, *pc_use_provided;
static int mfg_brc, mfg_upr;

double pearcorrshiftrot( double args[7] )
{ static double called= 0;
	fdsr_last_args[0]= ++called;
	memcpy( &fdsr_last_args[1], args, 6*sizeof(double) );
	if( !mfg_shift ){
		args[1]= mfg_shiftX;
		args[2]= mfg_shiftY;
	}
	if( mfg_rotate<= 0 ){
		if( !mfg_rotate ){
			args[3]= mfg_rotX;
			args[4]= mfg_rotY;
		}
		args[5]= mfg_angle;
	}
	if( !mfg_scale ){
		args[6]= mfg_Scale;
	}
	else if( args[6]< 0 ){
		args[6]*= -1;
	}
	if( mfg_brc ){
		if( args[3]< mfg_min ){
			args[3]= mfg_min;
		}
		else if( args[3]> mfg_max ){
			args[3]= mfg_max;
		}
		if( args[4]< mfg_min ){
			args[4]= mfg_min;
		}
		else if( args[4]> mfg_max ){
			args[4]= mfg_max;
		}
	}
	return( (fdsr_last_args[7]= 1- pearson_correlation_shiftrot( mfg_wi, mfg_idx1, mfg_idx2,
		True, args[6], True, args[1], args[2], True, args[3], args[4], args[5], mfg_radix,
		&mfg_spatial, &mfg_orientation, mfg_raw
	)) );
}

double pearcorrrot( double args[5] )
{ static double called= 0;
	fdsr_last_args[0]= ++called;
	memcpy( &fdsr_last_args[1], args, 4* sizeof(double) );
	if( mfg_rotate<= 0 ){
		if( !mfg_rotate ){
			args[1]= mfg_rotX;
			args[2]= mfg_rotY;
		}
		args[3]= mfg_angle;
	}
	if( !mfg_scale ){
		args[4]= mfg_Scale;
	}
	else if( args[4]< 0 ){
		args[4]*= -1;
	}
	if( mfg_brc ){
		if( args[1]< mfg_min ){
			args[1]= mfg_min;
		}
		else if( args[1]> mfg_max ){
			args[1]= mfg_max;
		}
		if( args[2]< mfg_min ){
			args[2]= mfg_min;
		}
		else if( args[2]> mfg_max ){
			args[2]= mfg_max;
		}
	}
	return( (fdsr_last_args[7]= 1- pearson_correlation_shiftrot( mfg_wi, mfg_idx1, mfg_idx2,
		True, args[4], True, mfg_shiftX, mfg_shiftY, True, args[1], args[2], args[3], mfg_radix,
		&mfg_spatial, &mfg_orientation, mfg_raw
	)) );
}

static double initialise_best( double P[7], double best[7], double (*minimise)(double *) )
{ int i;
	if( mfg_shift ){
		for( i= 1; i<= 6; i++ ){
			P[i]= best[i];
		}
	}
	else{
		P[1]= best[1], P[2]= best[2], P[3]= best[3]; P[4]= best[4];
	}
	return( (*minimise)( P ) );
}

static void pc_initial_simplex( double **P, double *bestP, double *fdP, double (*minimise)(double *),
	double shiftX, double shiftY, double rotX, double rotY, double angle, double escale,
	double minX, double minY, double maxX, double maxY, double Dscale)
{
	  /* Initialise the initial simplex. This is a tricky affaire, since it seems to determine the
	   \ space that continuous_simanneal() will search for a minimum. Thus, we take as the 1st
	   \ and 2nd elements the initial guess and the raw configuration. The remaining elements are
	   \ initialised such that the probably full search space is spanned: that's why we need minX, maxY, etc.
	   \ For the angle, the full range from 0 to mfg_radix is spanned; for the scale, values in
	   \ the appropriate range are used.
	   \ When not shifting, we use a subset of the full dimensioned simplex; otherwise, the "inactive"
	   \ dimensions are just ignored by the pearcorrshiftrot(), and we discard any non-neutral values
	   \ that continuous_simanneal() might have concocted.
	   */
	if( ascanf_verbose == 2 ){
	  /* (never happens?!) */
		fprintf( StdErr, "\n Initial guess parameters: shiftX=%g, shiftY=%g, rotX=%g, rotY=%g, angle=%g scale=%g\n",
			shiftX, shiftY, rotX, rotY, angle, escale
		);
		if( *pc_bound_rotation_centre ){
			fprintf( StdErr, " Rotation centre bounds: (%g,%g)\n",
				mfg_min, mfg_max
			);
		}
	}
	if( mfg_shift ){
		  /* P1 of the simplex is the initially-guessed shiftrot'ed state: 	*/
		P[1][1]= shiftX, P[1][2]= shiftY;
		P[1][3]= rotX, P[1][4]= rotY, P[1][5]= 0, P[1][6]= escale;
		fdP[1]= (*minimise)( P[1] );
		  /* P2 of the simplex is the non-shiftrot'ed state. Upon the 1st call, this
		   \ will be an ensemble of all 0s.
		   */
		fdP[2]= initialise_best( P[2], bestP, minimise );

		P[3][1]= minX, P[3][2]= minY;
		P[3][3]= minX, P[3][4]= minY, P[3][5]= mfg_radix* 0.2, P[3][6]= 0.2* Dscale;
		fdP[3]= (*minimise)( P[3] );
		P[4][1]= minX, P[4][2]= minY;
		P[4][3]= maxX, P[4][4]= maxY, P[4][5]= -mfg_radix* 0.4, P[4][6]= 0.4* Dscale;
		fdP[4]= (*minimise)( P[4] );
		P[5][1]= maxX, P[5][2]= maxY;
		P[5][3]= maxX, P[5][4]= maxY, P[5][5]= mfg_radix* 0.6, P[5][6]= 0.6* Dscale;
		fdP[5]= (*minimise)( P[5] );
		P[6][1]= maxX, P[6][2]= maxY;
		P[6][3]= minX, P[6][4]= minY, P[6][5]= -mfg_radix* 0.8, P[6][6]= 0.8* Dscale;
		fdP[6]= (*minimise)( P[6] );
		{ double xx= (maxX+minX)/2, yy= (maxY+minY)/2;
			P[7][1]= xx, P[7][2]= yy;
			P[7][3]= xx, P[7][4]= yy, P[7][5]= mfg_radix* 0.5, P[7][6]= 0.5* Dscale;
			fdP[7]= (*minimise)( P[7] );
		}
	}
	else{
		  /* P1 of the simplex is the initially-guessed shiftrot'ed state: 	*/
		P[1][1]= rotX, P[1][2]= rotY, P[1][3]= 0, P[1][4]= escale;
		fdP[1]= (*minimise)( P[1] );
		fdP[2]= initialise_best( P[2], bestP, minimise );

		P[3][1]= minX, P[3][2]= minY, P[3][3]= mfg_radix* 0.2, P[3][4]= 0.2* Dscale;
		fdP[3]= (*minimise)( P[3] );
		P[4][1]= maxX, P[4][2]= maxY, P[4][3]= -mfg_radix* 0.4, P[4][4]= 0.4* Dscale;
		fdP[4]= (*minimise)( P[4] );
		P[5][1]= maxX, P[5][2]= maxY, P[5][3]= mfg_radix* 0.6, P[5][4]= 0.6* Dscale;
		fdP[5]= (*minimise)( P[5] );
		P[6][1]= minX, P[6][2]= minY, P[6][3]= -mfg_radix* 0.8, P[6][4]= 0.8* Dscale;
		fdP[6]= (*minimise)( P[6] );
		{ double xx= (maxX+minX)/2, yy= (maxY+minY)/2;
			P[7][1]= xx, P[7][2]= yy, P[7][3]= mfg_radix* 0.5, P[7][4]= 0.5* Dscale;
			fdP[7]= (*minimise)( P[7] );
		}
	}
	if( ascanf_verbose == 2 ){
		fprintf( StdErr,
			" initial simplex: (%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)\n"
			" initial simplex values: %g,%g,%g,%g,%g,%g,%g\n"
			" initial best parameters: %g,%g,%g,%g,%g,%g\n"
			, P[1][1], P[1][2], P[1][3], P[1][4], P[1][5], P[1][6],
			P[2][1], P[2][2], P[2][3], P[2][4], P[2][5], P[2][6],
			P[3][1], P[3][2], P[3][3], P[3][4], P[3][5], P[3][6],
			P[4][1], P[4][2], P[4][3], P[4][4], P[4][5], P[4][6],
			P[5][1], P[5][2], P[5][3], P[5][4], P[5][5], P[5][6],
			P[6][1], P[6][2], P[6][3], P[6][4], P[6][5], P[6][6],
			P[7][1], P[7][2], P[7][3], P[7][4], P[7][5], P[7][6],
			fdP[1], fdP[2], fdP[3], fdP[4], fdP[5], fdP[6], fdP[7],
			bestP[1], bestP[2], bestP[3], bestP[4], bestP[5], bestP[6]
		);
	}
}

/* 20010618: allow fixed non-zero settings for !shift, !scale and/or !rotate... 	*/

/* NB: we maximise the correlation by MINIMISING 1-correlation...! */

int ascanf_maximal_pearsoncorrel( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double ftol;
  ascanf_Function *Rscale, *RshiftX, *RshiftY, *RrotX, *RrotY, *Rangle, *Iter, *taf1= NULL, *taf2= NULL;
  int restarts;
	if( ascanf_arguments>= 17 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			mfg_idx1= (int) args[0];
			if( AllSets[mfg_idx1].numPoints<= 0 ){
				ascanf_emsg= " (set 1 is empty and/or deleted) ";
				ascanf_arg_error= True;
				mfg_idx1= -1;
			}
		}
		else{
			ascanf_emsg= " (setnumber 1 out of range) ";
			ascanf_arg_error= True;
			mfg_idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< setNumber || ascanf_SyntaxCheck) ){
			mfg_idx2= (int) args[1];
			if( AllSets[mfg_idx2].numPoints<= 0 ){
				ascanf_emsg= " (set 2 is empty and/or deleted) ";
				ascanf_arg_error= True;
				mfg_idx2= -1;
			}
		}
		else{
			ascanf_emsg= " (setnumber 2 out of range) ";
			ascanf_arg_error= True;
			mfg_idx2= -1;
		}
		if( mfg_idx1>= 0 && mfg_idx2>= 0 && AllSets[mfg_idx1].numPoints!= AllSets[mfg_idx2].numPoints ){
			ascanf_emsg= " (sets have different numbers of points!) ";
			ascanf_arg_error= True;
		}
		mfg_raw= (args[2])? True : False;
		mfg_scale= (args[3])? True : False;
		mfg_shift= (args[4])? True : False;
		CLIP_EXPR_CAST( int, mfg_rotate, double, args[5], -MAXINT, MAXINT );
		mfg_radix= args[6];
		Iter= parse_ascanf_address( args[7], _ascanf_variable, "ascanf_maximal_pearsoncorrel", (int) ascanf_verbose, NULL );
		ftol= args[8];
		CLIP_EXPR_CAST( int, restarts, double, args[9], 0, MAXINT );
		if( (Rscale= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_maximal_pearsoncorrel", (int) ascanf_verbose, NULL )) ){
			mfg_Scale= Rscale->value;
		}
		if( (RshiftX= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_maximal_pearsoncorrel", (int) ascanf_verbose, NULL )) ){
			mfg_shiftX= RshiftX->value;
		}
		if( (RshiftY= parse_ascanf_address( args[12], _ascanf_variable, "ascanf_maximal_pearsoncorrel", (int) ascanf_verbose, NULL )) ){
			mfg_shiftY= RshiftY->value;
		}
		if( (RrotX= parse_ascanf_address( args[13], _ascanf_variable, "ascanf_maximal_pearsoncorrel", (int) ascanf_verbose, NULL )) ){
			mfg_rotX= RrotX->value;
		}
		if( (RrotY= parse_ascanf_address( args[14], _ascanf_variable, "ascanf_maximal_pearsoncorrel", (int) ascanf_verbose, NULL )) ){
			mfg_rotY= RrotY->value;
		}
		if( (Rangle= parse_ascanf_address( args[15], _ascanf_variable, "ascanf_maximal_pearsoncorrel", (int) ascanf_verbose, NULL )) ){
			mfg_angle= Rangle->value;
		}
		if( ascanf_arguments> 16 ){
			taf1= parse_ascanf_address( args[16], _ascanf_variable, "ascanf_pearson_correl_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 17 ){
			taf2= parse_ascanf_address( args[17], _ascanf_variable, "ascanf_pearson_correl_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( mfg_idx1>= 0 && mfg_idx2>= 0 && !ascanf_arg_error && Iter && !ascanf_SyntaxCheck ){
		  DataSet *set1= &AllSets[mfg_idx1];
		  DataSet *set2= &AllSets[mfg_idx2];
		  int iter= Iter->value, i, ndim= 6, r;
		  double x, y, bestfdP, shiftX, shiftY, rotX, rotY, angle, gravX1, gravY1, gravX2, gravY2;
		  double minX, maxX, minY, maxY, escale, Dscale;
		  ALLOCA( fdP, double, ndim+2, fdPlen );
		  ALLOCA( bestP, double, ndim+1, bestPlen );
		  ALLOCA( P, double*, ndim+2, Plen );
		  ALLOCA( P1, double, ndim+1, P1len );
		  ALLOCA( P2, double, ndim+1, P2len );
		  ALLOCA( P3, double, ndim+1, P3len );
		  ALLOCA( P4, double, ndim+1, P4len );
		  ALLOCA( P5, double, ndim+1, P5len );
		  ALLOCA( P6, double, ndim+1, P6len );
		  ALLOCA( P7, double, ndim+1, P7len );
		  double (*minimise)( double*)= pearcorrshiftrot, cl1= 0, cl2= 0;

			mfg_brc= (int) *pc_bound_rotation_centre;

			if( !(mfg_upr= (int) *pc_use_provided) ){
			  /* Set the user-provided parameters to neutral values since we're supposed to ignore them. */
				mfg_Scale= 1;
				mfg_shiftX= mfg_shiftY= mfg_rotX= mfg_rotY= mfg_angle= 0;
			}
			if( mfg_Scale< 0 ){
				mfg_Scale*= -1;
			}

			if( ActiveWin && ActiveWin!= StubWindow_ptr ){
				mfg_wi= ActiveWin;
			}
			else{
				mfg_wi= NULL;
			}

			if( !mfg_upr && mfg_rotate< 0 ){
				mfg_rotate= False;
			}

/* 			if( mfg_rotate ){	*/
				if( !mfg_radix ){
					mfg_radix= M_2PI;
				}
/* 			}	*/
/* 			else{	*/
/* 				mfg_radix= 0;	*/
/* 			}	*/

			for( i= 0; i< ndim; i++ ){
				bestP[i]= 0;
			}
			  /* initial scale should be 1!! */
			bestP[6]= 1;

			  /* Do nothing if the 2 sets are equal: */
			if( mfg_idx1== mfg_idx2 ){
				bestfdP= 1;
				iter= 0;
				goto minimised;
			}

			if( !mfg_shift ){
				if( !mfg_rotate && !mfg_scale ){
					  /* there's actually nothing to minimise!! */
					bestfdP= pearson_correlation( mfg_wi, mfg_idx1, mfg_idx2,
						&mfg_spatial, &mfg_orientation, mfg_raw
					);
					iter= 0;
					goto minimised;
				}
				else{
					  /* use a different subroutine since we skip the 1st two dimensions (shiftX,shiftY) */
					minimise= pearcorrrot;
					ndim= 4;
				}
			}
			  /* when not rotating we can ignore the last few dimensions. this works even though
			   \ the minimising function will try to vary them...
			   */
/* 			if( !mfg_rotate ){	*/
/* 				minimise= pearcorrshift;	*/
/* 				ndim= 3;	*/
/* 			}	*/

			if( P ){
			  /* initialise the simplex memory structure: */
				P[0]= NULL;
				P[1]= P1, P[2]= P2, P[3]= P3, P[4]= P4, P[5]= P5, P[6]= P6, P[7]= P7;
				for( i= 1; i< ndim+2; i++ ){
					if( !P[i] ){
						ascanf_emsg= " (memory allocation problem) ";
						goto bail_out;
					}
				}
			}
			else{
				ascanf_emsg= " (memory allocation problem) ";
				goto bail_out;
			}

			  /* Determine the unscaled sets' gravity points and X,Y ranges: */
			x= y= 0;
			for( i= 0; i< set1->numPoints; i++ ){
			  double x1, y1, px, py;
				if( !DiscardedPoint(mfg_wi, set1, i ) ){
					if( mfg_raw ){
						x1= XVAL( set1, i);
						y1= YVAL( set1, i);
					}
					else{
						x1= set1->xvec[i];
						y1= set1->yvec[i];
					}
					if( i ){
						minX= MIN( minX, x1 );
						maxX= MAX( maxX, x1 );
						minY= MIN( minY, y1 );
						maxY= MAX( maxY, y1 );
					}
					else{
						minX= maxX= x1;
						minY= maxY= y1;
					}
					x+= x1;
					y+= y1;
					if( i ){
						cl1+= sqrt( (XVAL(set1,i) - XVAL(set1,i-1))*(XVAL(set1,i) - XVAL(set1,i-1)) +
							(YVAL(set1,i) - YVAL(set1,i-1))*(YVAL(set1,i) - YVAL(set1,i-1)) );
						  /* Don't update curve_len since that one depends on curvelen_with_discarded, but
						   \ our purpose here not...
						   */
/* 						if( mfg_wi && set1->last_processed_wi!= mfg_wi ){	*/
/* 							mfg_wi->curve_len[mfg_idx1][i]= cl1;	*/
/* 						}	*/
					}
					px= x1;
					py= y1;
				}
			}
			if( mfg_wi && set1->last_processed_wi!= mfg_wi ){
				mfg_wi->curve_len[mfg_idx1][set1->numPoints]= cl1;
			}
			gravX1= x/ set1->numPoints;
			gravY1= y/ set1->numPoints;
			x= y= 0;
			for( i= 0; i< set2->numPoints; i++ ){
			  double x2, y2, px, py;
				if( !DiscardedPoint( mfg_wi, set2, i ) ){
					if( mfg_raw ){
						x2= XVAL( set2, i);
						y2= YVAL( set2, i);
					}
					else{
						x2= set2->xvec[i];
						y2= set2->yvec[i];
					}
					if( i ){
						mfg_min= MIN( mfg_min, x2 );
						mfg_max= MAX( mfg_max, x2 );
						mfg_min= MIN( mfg_min, y2 );
						mfg_max= MAX( mfg_max, y2 );
					}
					else{
						mfg_min= MIN( x2, y2 );
						mfg_max= MAX( x2, y2 );
					}
					minX= MIN( minX, x2 );
					maxX= MAX( maxX, x2 );
					minY= MIN( minY, y2 );
					maxY= MAX( maxY, y2 );
					x+= x2;
					y+= y2;
					if( i ){
						cl2+= sqrt( (XVAL(set2,i) - XVAL(set2,i-1))*(XVAL(set2,i) - XVAL(set2,i-1)) +
							(YVAL(set2,i) - YVAL(set2,i-1))*(YVAL(set2,i) - YVAL(set2,i-1)) );
/* 						if( mfg_wi && set2->last_processed_wi!= mfg_wi ){	*/
/* 							mfg_wi->curve_len[mfg_idx2][i]= cl2;	*/
/* 						}	*/
					}
					px= x2;
					py= y2;
				}
			}
			if( mfg_wi && set2->last_processed_wi!= mfg_wi ){
				mfg_wi->curve_len[mfg_idx2][set2->numPoints]= cl2;
			}
			gravX2= x/ set2->numPoints;
			gravY2= y/ set2->numPoints;

			  /* try to determine an initial guess for the scale factor. This will be such
			   \ that the two sets have equal curve length. If no window is active, we just
			   \ take 1 as a guess (but we could of course calculate the curve lengths here...)
			   */
			if( mfg_scale ){
				if( cl2 ){
					escale= cl1/ cl2;
					Dscale= 2* escale;
				}
				else{
					mfg_scale= False;
				}
			}
			if( !mfg_scale ){
				if( mfg_upr ){
					escale= Dscale= mfg_Scale;
				}
				else{
					escale= 1;
					Dscale= 1;
				}
			}

			  /* The initial centre of rotation will be the to-be-rotated set's gravity point: */
			rotX= gravX2, rotY= gravY2, angle= 0;
			  /* The initial shift will be such that set2's gravity point will coincide with that of set1: */
			shiftX= gravX1- gravX2, shiftY= gravY1- gravY2;

			mfg_spatial= mfg_orientation= -1;

			pc_initial_simplex( P, bestP, fdP, minimise, shiftX, shiftY, rotX, rotY, 0, escale,
				minX, minY, maxX, maxY, Dscale
			);


			  /* Initialise the best function value ever with something large. This means a safe margin larger than
			   \ the Pearson correlation at the "raw" configuration. The minimising function will in fact use this value
			   \ as a reference: if the initial value is lower than the real (global) minimum, no action occurs!
			   */
			bestfdP= 10* fdP[2];
			if( ascanf_verbose ){
				fprintf( StdErr,
					"\nRef: set %d, Adapt: set %d; %s mode; weights x,y,e=%g,%g,%g, do scale,shift,rotate=%d,%d,%d; %s rotation centre\n"
					" starting with best value 10*<unshifted>=%s; correlation at gravity point (shift %g,%g,%g,%g,%g) %s\n",
					mfg_idx1, mfg_idx2, (mfg_raw)? "raw" : "cooked",
					*pc_xcol, *pc_ycol, *pc_ecol,
					mfg_scale, mfg_shift, mfg_rotate,
					(*pc_bound_rotation_centre)? "bounded" : "unbounded",
					d2str( bestfdP,0,0),
					shiftX, shiftY, rotX, rotY, angle,
					d2str( fdP[2],0,0)
				);
				if( mfg_upr ){
					if( !mfg_scale ){
						fprintf( StdErr, "\tImposed scale: %g\n", mfg_Scale );
					}
					if( !mfg_shift ){
						fprintf( StdErr, "\tImposed shift: (%g,%g)\n", mfg_shiftX, mfg_shiftY );
					}
					if( !mfg_rotate ){
						fprintf( StdErr, "\tImposed rotation: (%g,%g) over %g\n", mfg_rotX, mfg_rotY, mfg_angle );
					}
					else if( mfg_rotate< 0 ){
						fprintf( StdErr, "\tImposed rotation: over %g\n", mfg_angle );
					}
				}
			}

			if( restarts ){
				continuous_simanneal( P, fdP, ndim, bestP, &bestfdP, (ftol)? 2* ftol : 1e-6, minimise, &iter, 0 );
			}
			else{
				continuous_simanneal( P, fdP, ndim, bestP, &bestfdP, ftol, minimise, &iter, 0 );
			}
			if( ascanf_verbose ){
				fprintf( StdErr,
					" maximal Pearson correlation (t=0): min@(%g,%g,%g,%g,%g,%g) == (%g,%g)=%g in %d iterations\n"
					" calls=%g, fdsr_last_args={%g,%g,%g,%g,%g,%g}=%g\n"
					" simplex: (%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)\n"
					, bestP[6], bestP[1], bestP[2], bestP[3], bestP[4], bestP[5],
					mfg_spatial, mfg_orientation,
					bestfdP, iter,
					fdsr_last_args[0],
					fdsr_last_args[1], fdsr_last_args[2], fdsr_last_args[3], fdsr_last_args[4], fdsr_last_args[5],
						fdsr_last_args[6], fdsr_last_args[7]
					, P[1][1], P[1][2], P[1][3], P[1][4], P[1][5], P[1][6],
					P[2][1], P[2][2], P[2][3], P[2][4], P[2][5], P[2][6],
					P[3][1], P[3][2], P[3][3], P[3][4], P[3][5], P[3][6],
					P[4][1], P[4][2], P[4][3], P[4][4], P[4][5], P[4][6],
					P[5][1], P[5][2], P[5][3], P[5][4], P[5][5], P[5][6],
					P[6][1], P[6][2], P[6][3], P[6][4], P[6][5], P[6][6],
					P[7][1], P[7][2], P[7][3], P[7][4], P[7][5], P[7][6]
				);
			}
			for( r= 1; r<= restarts; r++ ){
			  int iter2= Iter->value;
			  double best= bestfdP;
#define PROBE_FRACT	0.95
				pc_initial_simplex( P, bestP, fdP, minimise, shiftX, shiftY, rotX, rotY, 0, escale,
					minX, minY, maxX, maxY, Dscale
				);
				bestfdP= bestfdP* PROBE_FRACT;
				continuous_simanneal( P, fdP, ndim, bestP, &bestfdP, ftol, minimise, &iter2, 0 );
				if( bestfdP== PROBE_FRACT* best ){
				  ALLOCA( lP, double, ndim+1, lPlen );
					if( ascanf_verbose ){
						fprintf( StdErr,
							"\n Probe best value found did not improve: re-determining it at the best parameters\n"
						);
					}
					bestfdP= initialise_best( lP, bestP, minimise );
				}
				if( ascanf_verbose ){
					fprintf( StdErr,
						"\n R%d maximal Pearson correlation (t=0): min@(%g,%g,%g,%g,%g,%g) == (%g,%g)=%g in %d iterations\n"
						" calls=%g, fdsr_last_args={%g,%g,%g,%g,%g,%g}=%g\n"
						" simplex: (%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g,%g)\n"
						, r, bestP[6], bestP[1], bestP[2], bestP[3], bestP[4], bestP[5],
						mfg_spatial, mfg_orientation,
						bestfdP, iter2,
						fdsr_last_args[0],
						fdsr_last_args[1], fdsr_last_args[2], fdsr_last_args[3], fdsr_last_args[4], fdsr_last_args[5],
							fdsr_last_args[6], fdsr_last_args[7]
						, P[1][1], P[1][2], P[1][3], P[1][4], P[1][5], P[1][6],
						P[2][1], P[2][2], P[2][3], P[2][4], P[2][5], P[2][6],
						P[3][1], P[3][2], P[3][3], P[3][4], P[3][5], P[3][6],
						P[4][1], P[4][2], P[4][3], P[4][4], P[4][5], P[4][6],
						P[5][1], P[5][2], P[5][3], P[5][4], P[5][5], P[5][6],
						P[6][1], P[6][2], P[6][3], P[6][4], P[6][5], P[6][6],
						P[7][1], P[7][2], P[7][3], P[7][4], P[7][5], P[7][6]
					);
				}
				iter+= iter2;
			}

			if( !mfg_shift ){
				  /* We used a subset of the full dimensionality: now put the values where we expect them later on: */
				bestP[6]= bestP[4];
				bestP[5]= bestP[3];
				bestP[4]= bestP[2];
				bestP[3]= bestP[1];
				if( !mfg_upr ){
					bestP[1]= bestP[2]= 0;
				}
				else{
					bestP[1]= mfg_shiftX;
					bestP[2]= mfg_shiftY;
				}
			}
			if( !mfg_rotate && !mfg_upr ){
				bestP[3]= bestP[4]= bestP[5]= 0;
			}
minimised:;
			Iter->value= iter;
			if( RshiftX ){
				RshiftX->value= bestP[1];
			}
			if( RshiftY ){
				RshiftY->value= bestP[2];
			}
			if( RrotX ){
				RrotX->value= bestP[3];
			}
			if( RrotY ){
				RrotY->value= bestP[4];
			}
			if( Rangle ){
				Rangle->value= bestP[5];
			}
			if( Rscale ){
				Rscale->value= (mfg_scale || mfg_upr)? fabs( bestP[6] ): 1;
			}
			*result= bestfdP;
			if( taf1 ){
				taf1->value= mfg_spatial;
			}
			if( taf2 ){
				taf2->value= mfg_orientation;
			}
			GCA();
		}
		else{
			set_NaN(*result);
			set_NaN(Iter->value);
			if( RshiftX ){
				set_NaN(RshiftX->value);
			}
			if( RshiftY ){
				set_NaN(RshiftY->value);
			}
			if( RrotX ){
				set_NaN(RrotX->value);
			}
			if( RrotY ){
				set_NaN(RrotY->value);
			}
			if( Rangle ){
				set_NaN(Rangle->value);
			}
			if( Rscale ){
				set_NaN(Rscale->value);
			}
			if( taf1 ){
				set_NaN(taf1->value);
			}
			if( taf2 ){
				set_NaN(taf2->value);
			}
		}
	}
	else{
bail_out:;
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

static ascanf_Function pearson_correl_Function[] = {
	{ "$pearson-correlation-xcol", NULL, 2, _ascanf_variable,
		"$pearson-correlation-xcol: weight to apply to the sets' xcol in the calculation\n"
		" of the correlation.\n", 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$pearson-correlation-ycol", NULL, 2, _ascanf_variable,
		"$pearson-correlation-ycol: weight to apply to the sets' ycol in the calculation\n"
		" of the correlation.\n", 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$pearson-correlation-ecol", NULL, 2, _ascanf_variable,
		"$pearson-correlation-ecol: weight to apply to the sets' ecol in the calculation\n"
		" of the correlation.\n", 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$pearson-correlation-bound-rotation-centre", NULL, 2, _ascanf_variable,
		"$pearson-correlation-bound-rotation-centre: determines if the (rotX,rotY) centre of rotation is bounded\n"
		" to be within the adjustable set's X and Y range during pearson-correlation maximisation.\n"
		" If not set, (rotX,rotY) can vary freely, which means the set can be shifted even when the <shift>\n"
		" argument to maximal-pearson-correlation[] is False. When bounded, such shifts can still occur, but the\n"
		" rotation centre can only vary \"within\" the set.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$pearson-correlation-use-provided-values", NULL, 2, _ascanf_variable,
		"$pearson-correlation-use-provided-values: by default, maximal-pearson-correlation[] ignores the values\n"
		" pointed to by the return pointers (&Rscale, &shiftX, etc), using those pointers only to return\n"
		" the parameter values for which it found a maximal correlation. When this flag is set, however,\n"
		" it will use the values pointed to for those parameters for which the corresponding maxisimation flag\n"
		" is False. Thus, when <rotate> is false, a maximal correlation will be found for the values passed in\n"
		" (&RrotX,&RrotY,&Rangle), etc. Special case: rotate==-1; in this case, only &Rangle is used, finding a\n"
		" maximal correlation at that particular rotation angle.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$pearson-correlation-transformation", NULL, 2, _ascanf_variable,
		"$pearson-correlation-transformation: select a transformation to be applied to the data before doing\n"
		" the correlation. The transformations supplied can be a remedy against a constant 'dimension' in one\n"
		" of the sets - constant X, Y or E/O. The simple trick is to add a similar non-constant value to both sets.\n"
		" That is, adding a straight, non-horizontal line to straight lines won't influence the correlation between\n"
		" them. Similarly, adding e.g. a single period of a sine function (with a period equal to the number of\n"
		" points in the sets!) won't influence the result as long as the amplitude is small with respect to the\n"
		" data range. The sine transformation has the property of not altering the average. Finally, noise can be\n"
		" added, which has the property of (theoretically) not altering the average, and not carrying signal itself.\n"
		" A constant bias can be added with $pearson-correlation-transformation-parameters[1].\n"
		" 0: no transformation\n"
		" 1: add a linear component, mul[$pearson-correlation-transformation-parameters[0],$Counter]\n"
		" 2: add a sine component, mul[$pearson-correlation-transformation-parameters[0],sin[$Counter,sub[$numPoints,1]]]\n"
		" 3: add Gaussion noise, with average 0 and standard deviation $pearson-correlation-transformation-parameters[0].\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$pearson-correlation-transformation-parameters", NULL, 2, _ascanf_array,
		"$pearson-correlation-transformation-parameters[2]: the parameters to the transformation selected\n"
		" by $pearson-correlation-transformation: (usually) a gain,constant-bias value-pair.\n"
		, 1, 0, 0, 0, 0, 0, 0.0, &_pc_transformation_parameters[0], NULL, NULL, NULL, NULL, 0, 0, 
		sizeof(pc_transformation_parameters)/sizeof(double)
	},
	{ "$pearson-correlation-noise-percentage", NULL, 2, _ascanf_variable,
		"$pearson-correlation-noise-percentage: add the specified percentage of Gaussian random noise to the\n"
		" data (x, y, e) for which the correlation is calculated. This is intended to protect against sets\n"
		" with (at least 1) constant dimension.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$pearson-correlation-formal-form", NULL, 2, _ascanf_variable,
		"$pearson-correlation-formal-form: use the formal form of the formula, not the simplified version\n"
		" that requires only a single pass over the data. The formal form can have advantages when a dimension\n"
		" in one of the sets has no variation at all (= all points have the same value) due to very small\n"
		" (roundoff) errors that may arise.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "pearson-correlation", ascanf_pearson_correl, 5, NOT_EOF_OR_RETURN,
		"pearson-correlation[set1,set2,raw[,&spatial,&orientation]]\n"
		" A routine that calculates the Pearson correlation between 2 sets.\n"
		" The correlation is calculated for both the sets' current X, Y and error/orientation columns,\n"
		" as a weighted linear average. The weights are specified with the $pearson-correlation-?col\n"
		" variables.\n"
	},
	{ "pearson-correlation-shiftrot", ascanf_pearson_correl_shiftrot, 12, NOT_EOF_OR_RETURN,
		"pearson-correlation-shiftrot[set1, set2, raw, scale, shiftX, shiftY, rotX, rotY, angle, radix[,&spatial,&orientation]]\n"
		" This calculates the Pearson correlation after a rotation over <angle> (base <radix>) around (rotX,rotY) and a\n"
		" subsequent shift over (shiftX,shiftY) of the 2nd set. The set is first scaled over <scale>.\n"
		" When set2 is in vector mode (error column contains orientations), the orientations are rotated over <angle>,\n"
		" otherwise, the error column is not modified. The other arguments are as described under pearson-correlation[].\n"
		" When either of the scale/shift/rotation parameters is a NaN, the corresponding operation(s) is/are not performed.\n"
	},
	{ "Set-shiftrot", ascanf_setshiftrot, 12, NOT_EOF,
		"Set-shiftrot[set,pnt_nr,scale,shiftX,shiftY,rotX,rotY,angle,radix, &rX, &rY, &rE]\n"
		" Return a set's co-ordinates (X,Y,E) at point <pnt_nr> after a rotation of <angle> (base <radix>)\n"
		" around (rotX,rotY), and a shift over (shiftX,shiftY). The transformed values are stored in\n"
		" &rX, &rY, &rE. If the set's error column represent orientations (vector mode), rE=E+angle;\n"
		" otherwise, rE=E.\n"
	},
	{ "maximal-pearson-correlation", ascanf_maximal_pearsoncorrel, 18, NOT_EOF_OR_RETURN,
		"maximal-pearson-correlation[set1,set2,raw,scale,shift,rotate,radix,&iterations,fractol,restarts,&Rscale,&RshiftX,&RshiftY,&RrotX,&RrotY,&Rangle[,&spat,&orn]]\n"
		" Find the maximal Pearson correlation by adjusting the shift and rotation parameters as described for\n"
		" pearson-correlation-shiftrot[]. Iterations contains the number of iterations to run, and returns the\n"
		" number actually performed (over all restarts); fractol is the fractional tolerance for the solution; the other pointers\n"
		" contain the final shift&rot parameters that result in the maximal correlation.\n"
		" Shifting is only done when <shift>, rotation only when <rotate>; when both are False, no maximisation takes place...\n"
		" See also $pearson-correlation-use-provided-values.\n"
		" NB: when shift=False, rotX and rotY are still subject to change (cf. $pearson-correlation-bound-rotation-centre)!\n"
	},
};

static double *pc_xcol= &pearson_correl_Function[0].value;
static double *pc_ycol= &pearson_correl_Function[1].value;
static double *pc_ecol= &pearson_correl_Function[2].value;
static double *pc_bound_rotation_centre= &pearson_correl_Function[3].value;
static double *pc_use_provided= &pearson_correl_Function[4].value;
static double *pc_transformation= &pearson_correl_Function[5].value;
static double *pc_noise_percentage= &pearson_correl_Function[7].value;
static double *pc_formal= &pearson_correl_Function[8].value;

static int pearson_correl_Functions= sizeof(pearson_correl_Function)/sizeof(ascanf_Function);

static void print_settings(FILE *fp)
{
	fprintf( fp, "Current pearson-correlation settings: weights x,y,e=%g,%g,%g bRctr=%g upr=%g noise=%g, formal=%g, transform=%g,%g\n",
		*pc_xcol, *pc_ycol, *pc_ecol, *pc_bound_rotation_centre, *pc_use_provided,
		*pc_noise_percentage, *pc_formal,
		*pc_transformation, pc_transformation_parameters[0]
	);
}

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= pearson_correl_Function;
  static char called= 0, ptp= -1;
  int i, warn= True;
  char buf[64];

	for( i= 0; i< pearson_correl_Functions; i++, af++ ){
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
			if( Check_Doubles_Ascanf( af, label, warn ) ){
				warn= False;
			}
			if( af->array== _pc_transformation_parameters ){
				ptp= i;
			}
		}
		if( ptp== i ){
			af->array= pc_transformation_parameters;
			af->N= sizeof(_pc_transformation_parameters)/sizeof(double);
		}
		af->dymod= new;
	}
	called+= 1;
}

static int initialised= False;

static Compiled_Form *ran_PM_hook= NULL;

DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;

	if( !DMBase ){
	  DyModLists *simanneal= NULL;
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
		  /* 20051214: load the simanneal.so library, and get the necessary symbols from it: */
		Auto_LoadDyMod_LastPtr(NULL, -1, "sran-PM", &simanneal );
		if( simanneal->size != sizeof(DyModLists) ){
			fprintf( StdErr, "Error: received simanneal DyMod handle of unexpected size\n" );
			fprintf( stderr, "DyMod API version mismatch: either this module or XGraph is newer than the other...\n" );
			fprintf( stderr, "Now returning DM_Error to our caller!\n" );
			return( DM_Error );
		}

		XGRAPH_VARIABLE( radix_ptr, "radix");
		XGRAPH_VARIABLE( DrawAllSets_ptr, "DrawAllSets");
		XGRAPH_FUNCTION( normal_rand_ptr, "normal_rand");
		XGRAPH_FUNCTION( fascanf_ptr, "fascanf" );
		XGRAPH_FUNCTION( Destroy_Form_ptr, "Destroy_Form" );
		if( simanneal ){
			DYMOD_FUNCTION(simanneal, continuous_simanneal_ptr, "continuous_simanneal");
			DYMOD_FUNCTION(simanneal, sran1_ptr, "sran1");
		}
		else{
			fprintf( StdErr, "Error loading a simanneal dymod, can't obtain address for continuous_simanneal &c\n" );
			return( DM_Error );
		}
		  /* now "fix" the function we want, so no attempt will be made to auto-unload the (auto-loaded) library containing it: */
		{ int n= 1;
		  char expr[256];
		  double dummy;
			snprintf( expr, sizeof(expr), "IDict[ verbose[ ran-PM[0%c1]%c \"" __FILE__ "\"] ]", ascanf_separator, ascanf_separator );
			fascanf( &n, expr, &dummy, NULL, NULL, NULL, &ran_PM_hook );
			if( !ran_PM_hook ){
				fprintf( StdErr,
					"%s::initDyMod(): failed to 'lock' the needed simanneal.so functions: proceed with toes crossed!\n",
					__FILE__
				);
			}
		}
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );

	if( !(pc_transformation_parameters= (double*) calloc( 1, sizeof(_pc_transformation_parameters))) ){
		pc_transformation_parameters= _pc_transformation_parameters;
	}
	else{
		memcpy( pc_transformation_parameters, _pc_transformation_parameters, sizeof(_pc_transformation_parameters) );
	}
	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( pearson_correl_Function, pearson_correl_Functions, "pearson_correlation::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-pearson-correlation" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A library with routines for determining the Pearson correlation between 2 sets,\n"
		" and (in the end) the maximal Pearson correlation by shifting and rotating the 2nd\n"
		" set with respect to the 1st.\n"
		" This library contains a certain number of routines in common with the fig_dist.so library,\n"
		" but is independent of it.\n"
		" The simanneal.so library must be loaded first!\n"
	);

	sran1( (long) time(NULL) );

	called= True;

	return( DM_Ascanf );
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
	fprintf( SE, "%s::closeDyMod(%d): Closing %s loaded from %s, call %d", __FILE__,
		force, target->name, target->path, ++called
	);
	if( target->loaded4 ){
		fprintf( SE, "; auto-loaded because of \"%s\"", target->loaded4 );
	}
	if( initialised ){
	  int r= remove_ascanf_functions( pearson_correl_Function, pearson_correl_Functions, force );
		if( force || r== pearson_correl_Functions ){
			for( i= 0; i< pearson_correl_Functions; i++ ){
				pearson_correl_Function[i].dymod= NULL;
			}
			initialised= False;
			xfree( target->libname );
			xfree( target->buildstring );
			xfree( target->description );
			ret= target->type= DM_Unloaded;
			if( ran_PM_hook ){
				Destroy_Form( &ran_PM_hook );
			}
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
			fprintf( SE, " -- refused: variables are in use (remove_ascanf_functions() removed %d out of %d)\n",
				r, pearson_correl_Functions
			);
		}
	}
	return(ret);
}

