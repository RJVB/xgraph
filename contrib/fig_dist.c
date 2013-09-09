/* fig_dist.c (C) 2001,2002 R.J.V. Bertin.
 \ Ascanf routines for calculating and minimising the figural distance between 2 sets.
 \ Based on the definition by Conditt et al., J. Neurophysiol. 78-1 1997.
 \ Version: 20020502
 */

#include "config.h"
IDENTIFY( "Figural distance ascanf library module" );

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
   \ (Or, we might use fdecl_stubs.h)
   */
#include "xgout.h"
#include "xgraph.h"
#include "new_ps.h"
#include "xtb/xtb.h"

#include "SS.h"

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
	int *DrawAllSets_ptr;
	double *radix_ptr;
	SimpleAngleStats* (*SAS_Add_Data_angsincos_ptr)( SimpleAngleStats *a, long count,
		double sum, double sinval, double cosval,
		double weight);
	int (*_ascanf_SAS_StatsBin_ptr)( ascanf_Function *af, int exact );
	double (*conv_angle__ptr)( double phi, double base);
	int (*fascanf_ptr)( int *n, char *s, double *a, char *ch, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS], struct Compiled_Form **form );
	int (*Destroy_Form_ptr)( struct Compiled_Form **form );

#	define DrawAllSets	(*DrawAllSets_ptr)
#	define _ascanf_SAS_StatsBin	(*_ascanf_SAS_StatsBin_ptr)
#	define SAS_Add_Data_angsincos	(*SAS_Add_Data_angsincos_ptr)
#	define conv_angle_	(*conv_angle__ptr)
#	define fascanf	(*fascanf_ptr)
#	define Destroy_Form	(*Destroy_Form_ptr)

void (*continuous_simanneal_ptr)(double **p, double y[], int ndim, double pb[], double *yb, double ftol, double (*funk) (double []) , int *iter, double T);
void (*sran1_ptr)( long seed );
#define continuous_simanneal	(*continuous_simanneal_ptr)
#define sran1	(sran1_ptr)

static double *fd_normalise_orientation, *fd_consistent_norm_orn, *fd_normalise_spatial, *fd_absolute_orientation, *fd_pointwise;

static double *fd_transform_figdist, *fd_transform_figdist_param;
static SimpleAngleStats *fd_SAS_figdist_dirn;

static void print_settings(FILE *fp);

/* Base routine for calculating the figural distance as defined by Conditt &al, J. Neurophysiol 78-1 1997.
 \ Calculates the basic figural distance from set1 to set2, returning the sum-of-distances in *SumDist
 \ and the number of samples in *K. The raw1 and raw2 arguments indicate which values to use for set1
 \ and set2 respectively; raw or transformed (cooked). orn_handling indicates what to do with the 3rd
 \ column (error, orientation, intensity, etc.):
 \ 0: ignore
 \ 1: use as a 3rd dimension Cartesian co-ordinate; distance= sqrt(x**2 + y**2 + o**2)
 \ 2: use as orientation (angle), and calculate the difference subject to $figural-distance-absolute-orientation
 */
static void oneway_fig_dist( LocalWin *wi, DataSet *set1, DataSet *set2,
	double *X1, double *Y1, double *O1, int nP1, double *X2, double *Y2, double *O2, int nP2,
	double *SumDist, int *K, int raw1, int raw2, int orn_handling, int *closest, int dirn )
{ int k, l, l0, L, newmin= True, nosquare= False;
  double dist, mindist, dx, dy, o1, o2;
  double rdx= (wi)? wi->radix : *radix_ptr;
  Boolean fdao= (*fd_absolute_orientation)? True : False;
  Boolean fdpw= (*fd_pointwise)? True : False;

	*SumDist= 0;
	*K= 0;

	if( *fd_normalise_orientation && *fd_normalise_spatial ){
		*fd_normalise_spatial= 0;
	}
	if( !rdx ){
		rdx= M_2PI;
	}
	else if( rdx< 0 ){
		rdx*= -1;
	}
	if( fd_SAS_figdist_dirn ){
		if( !fd_SAS_figdist_dirn->Gonio_Base ){
			fd_SAS_figdist_dirn->Gonio_Base= rdx;
			fd_SAS_figdist_dirn->Gonio_Offset= 0;
		}
		if( !dirn ){
			SAS_Reset(fd_SAS_figdist_dirn);
		}
	}

	if( set1 ){
		nP1= set1->numPoints;
	}
	if( set2 ){
		nP2= set2->numPoints;
	}

	set_NaN(o1);
	set_NaN(o2);

	for( k= 0; k< nP1; k++ ){
		if( !DiscardedPoint( wi, set1, k) ){
		  double x1, y1;
			if( set1 ){
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
			}
			else{
				x1= X1[k];
				y1= Y1[k];
				if( O1 ){
					o1= O1[k];
				}
			}
			newmin= True;
			if( fdpw ){
			  /* Only determine the distance to point l=k, i.e. the paired
			   \ sample on the other curve.
			   \ 20030330: L should be equal/inferior to set2->numPoints...!
			   */
				l0= k, L= MIN(k+1, nP2);
			}
			else{
			  /* Determine the distances to all points on the other curve,
			   \ and find the smallest one.
			   */
				l0= 0, L= nP2;
			}
			for( l= l0; l< L; l++ ){
				if( !DiscardedPoint( wi, set2, l) ){
				  double x2, y2;
					if( set2 ){
						if( raw2 ){
							x2= XVAL( set2, l);
							y2= YVAL( set2, l);
							o2= ERROR( set2, l);
						}
						else{
							x2= set2->xvec[l];
							y2= set2->yvec[l];
							o2= set2->errvec[l];
						}
					}
					else{
						x2= X2[l];
						y2= Y2[l];
						if( O2 ){
							o2= O2[l];
						}
					}
					if( orn_handling== 2 ){
					  /* Calculate the difference between 2 angles: */
						if( NaN(o1) || NaN(o2) ){
							dist= 0;
							if( newmin ){
								mindist= dist;
							}
							else if( (dist)< (mindist) ){
								mindist= dist;
							}
						}
						else if( fdao ){
							dist= fabs(o2 - o1)/ rdx;
							if( NaN(dist) ){
								dist= 0;
							}
							else if( dist> 0.5 ){
								dist= fabs( dist- 1 );
							}
							if( newmin ){
								mindist= dist;
							}
							else if( (dist)< (mindist) ){
								mindist= dist;
							}
						}
						else{
							dist= (conv_angle_(o2,rdx) - conv_angle_(o1,rdx))/ rdx;
							if( NaN(dist) ){
								dist= 0;
								mindist= (newmin)? dist : MIN(mindist, dist);
							}
							else if( dist> 0.5 ){
								dist= ( dist- 1 );
							}
							else if( dist<= -0.5 ){
								dist= ( dist+ 1 );
							}
							if( newmin ){
								mindist= dist;
							}
							else if( fabs(dist)< fabs(mindist) ){
								mindist= dist;
							}
						}
						nosquare= False;
					}
					else if( orn_handling== 0 || NaN(o1) || NaN(o2) ){
#if DEBUG == 2
					  double x3= x2-x1, y3= y2-y1;
#define x2 x3
#define y2 y3
#else
						x2-= x1;
						y2-= y1;
#endif
						if( *fd_transform_figdist!= 1 || *fd_transform_figdist_param!= 2 ){
							dist= sqrt( x2*x2 + y2*y2 );
							nosquare= False;
						}
						else{
							dist= x2*x2 + y2*y2;
							nosquare= True;
						}
						if( (newmin || dist< mindist) ){
							dx= x2;
							dy= y2;
							if( closest ){
								closest[k]= l;
#if DEBUG == 2
#undef x2
#undef y2
								if( set1 && set2 )
								fprintf( StdErr, "oneway_fig_dist(%d->%d): point %d::%d at %g from %d::%d ( sqrt( (%g-%g)=%g**2 + (%g-%g)=%g**2) = sqrt(%g+%g) )\n",
									set1->set_nr, set2->set_nr,
									set1->set_nr, k, dist, set2->set_nr, l,
									x2, x1, x2-x1, y2, y1, y2-y1, pow(x2-x1,2), pow(y2-y1,2)
								);
#endif
							}
							mindist= dist;
						}
					}
					else{
						x2-= x1;
						y2-= y1;
						o2-= o1;
						if( *fd_transform_figdist!= 1 || *fd_transform_figdist_param!= 2 ){
							dist= sqrt( x2*x2 + y2*y2 + o2*o2 );
							nosquare= False;
						}
						else{
							dist= x2*x2 + y2*y2 + o2*o2;
							nosquare= True;
						}
						if( closest && (newmin || dist< mindist) ){
							closest[k]= l;
						}
						mindist= (newmin)? dist : MIN(mindist, dist);
					}
					newmin= False;
				}
			}
			if( !newmin ){
			  int sgn= SIGN(mindist);
			  double md= (mindist< 0)? -mindist : mindist;
				*K+= 1;
				  /* 20010703: implementation of additional transformation (log and power)
				   \ of the current found minimal distance. NOTE: this supposes that the
				   \ transformation is not inversive, i.e. that the smallest distance remains
				   \ the smallest after transformation.
				   */
				switch( (int) *fd_transform_figdist ){
					case 1:
						if( !nosquare && *fd_transform_figdist_param!= 1 ){
							if( *fd_transform_figdist_param< 0 ){
								mindist= sgn* pow( md, - *fd_transform_figdist_param );
							}
							else{
								mindist= sgn* pow( md, *fd_transform_figdist_param );
							}
						}
						break;
					case 2:
						if( mindist ){
							if( *fd_transform_figdist_param== 10 ){
								mindist= sgn* log10( md );
							}
							else if( *fd_transform_figdist_param<= 1 ){
								mindist= sgn* log( md );
							}
							else{
								mindist= sgn* log(md)/ log(*fd_transform_figdist_param);
							}
						}
						break;
				}
				(*SumDist)+= mindist;
				if( fd_SAS_figdist_dirn && orn_handling==0 ){
				  double ang;
					if( dirn ){
						dx*= -1, dy*= -1;
					}
					ang= atan2( dy, dx )*
						fd_SAS_figdist_dirn->Units_per_Radian+ fd_SAS_figdist_dirn->Gonio_Offset;
					SAS_Add_Data_angsincos( fd_SAS_figdist_dirn, 1, ang, dy, dx, 1.0 );
				}
			}
		}
	}
	if( orn_handling== 2 ){
		if( !fdao && *SumDist< 0 ){
			*SumDist*= -1;
		}
		if( !(*fd_normalise_orientation || *fd_consistent_norm_orn) ){
		  /* If we were looking at orientations, and they should not be normalised, convert
		   \ the found sum-of-distances to radians:
		   \ 20020502: do this only when not fd_consistent_norm_orn; see the description of
		   \ $figural-distance-constistent-orientation-normalisation.
		   */
			(*SumDist)*= M_2PI;
		}
	}
}

static double *fd_orientation;

double fig_dist( LocalWin *wi, DataSet *set1, DataSet *set2,
	double *X1, double *Y1, double *O1, int nP1, double *X2, double *Y2, double *O2, int nP2,
	int raw1, int raw2, int all_vectors, double *spatial, double *orientation, int *closest12, int *closest21 )
{ int K, L;
  double SumDist1, SumDist2;
	oneway_fig_dist( wi, set1, set2, X1, Y1, O1, nP1, X2, Y2, O2, nP2, &SumDist1, &K, raw1, raw2, 0, closest12, 0 );
	if( !*fd_pointwise ){
		oneway_fig_dist( wi, set2, set1, X1, Y1, O1, nP1, X2, Y2, O2, nP2, &SumDist2, &L, raw2, raw1, 0, closest21, 1 );
	}
	else{
		L= 0, SumDist2= 0;
	}
	*spatial= (SumDist1 + SumDist2)/ (K + L);
	if( all_vectors && *fd_orientation ){
	  SimpleAngleStats *SASfdd= fd_SAS_figdist_dirn;
		fd_SAS_figdist_dirn= NULL;
		oneway_fig_dist( wi, set1, set2, X1, Y1, O1, nP1, X2, Y2, O2, nP2, &SumDist1, &K, raw1, raw2, 2, NULL, 0 );
		if( *fd_pointwise ){
			L= 0, SumDist2= 0;
		}
		else{
			oneway_fig_dist( wi, set2, set1, X1, Y1, O1, nP1, X2, Y2, O2, nP2, &SumDist2, &L, raw2, raw1, 2, NULL, 1 );
		}
		*orientation= (SumDist1 + SumDist2)/ (K + L);
		switch( (int) *fd_normalise_orientation ){
			case 2:
				*orientation*= (1+ *spatial)* 2;
				break;
			default:
			case 1:
				if( *fd_consistent_norm_orn ){
					if( *spatial ){
						*orientation *= 2* *spatial;
					}
				}
				else{
					*orientation *= (*spatial)? 2* *spatial : M_2PI;
				}
				break;
			case 0:
				if( *fd_normalise_spatial && *orientation ){
					*spatial *= *orientation;
				}
				break;
		}
		fd_SAS_figdist_dirn= SASfdd;
		return( sqrt(*spatial * *spatial + *orientation * *orientation) );
	}
	else{
		  /* 20020501: we set *orientation to a NaN if not used!? */
/* 		*orientation= -1;	*/
		set_NaN(*orientation);
		return( *spatial );
	}
}

double figural_distance( LocalWin *wi, int idx1, int idx2, double *spatial, double *orientation, int raw )
{ int all_vectors;
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
	return( fig_dist( wi, &AllSets[idx1], &AllSets[idx2],
		NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, raw, raw, all_vectors, spatial, orientation, NULL, NULL ) );
}

double figural_distance_shiftrot( LocalWin *wi, int idx1, int idx2,
	int scale, double Scale,
	int shift, double shiftX, double shiftY, int rotate, double rotX, double rotY, double angle, double Radix,
	double *spatial, double *orientation, int raw, int *closest12, int *closest21 )
{ int all_vectors;
  extern int DrawAllSets;
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
	{ int i, ds, das, hls;
	  double ret;
	  double ang_sin, ang_cos;
	  DataSet *set2= &AllSets[idx2];
	  ALLOCA( txvec, double, set2->numPoints, txvec_len );
	  ALLOCA( tyvec, double, set2->numPoints, tyvec_len );
	  ALLOCA( terrvec, double, set2->numPoints, terrvec_len );
	  double *sxvec= set2->xvec, *syvec= set2->yvec, *serrvec= set2->errvec;
	  double rdx= (wi)? wi->radix : *radix_ptr;

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
				"\nSet %d vs. %d; %s mode; %s, %s%s %s %s orientation, do scale,shift,rotate=%d,%d,%d;\n"
				" shift (%g,%g) rotate (%g,%g,%g) radix %g; scale %g; ",
				idx1, idx2, (raw)? "raw" : "cooked",
				(*fd_pointwise)? "point-wise" : "Conditt",
				(*fd_normalise_spatial)? "normalised spatial, " : "",
				(*fd_orientation)? "using" : "ignoring",
				(*fd_absolute_orientation)? "absolute" : "signed",
				(*fd_normalise_orientation)? "normalised" : "radians",
				scale, shift, rotate,
				shiftX, shiftY, rotX, rotY, angle, Radix, Scale
			);
		}

		  /* Calculate the component(s) of the figural distance: */
		ret= fig_dist( wi, &AllSets[idx1], set2, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0,
			raw, 0, all_vectors, spatial, orientation, closest12, closest21 );
		if( ascanf_verbose ){
			fprintf( StdErr, "spatial=%g orientation=%g combined=%g\n",
				*spatial, *orientation, ret
			);
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
}

int ascanf_fig_dist( ASCB_ARGLIST )
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
		raw= (args[2])? True : False;
		if( ascanf_arguments> 3 ){
			taf1= parse_ascanf_address( args[3], _ascanf_variable, "ascanf_fig_dist", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 4 ){
			taf2= parse_ascanf_address( args[4], _ascanf_variable, "ascanf_fig_dist", (int) ascanf_verbose, NULL );
		}
		if( !ascanf_arg_error && idx1>= 0 && idx2>= 0 ){
			*result= figural_distance( (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin : NULL, idx1, idx2,
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
			*result= -1;
		}
	}
	else{
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

int ascanf_fig_dist_shiftrot( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx1, idx2, raw, *closest12= NULL, *closest21= NULL;
  ascanf_Function *taf1= NULL, *taf2= NULL, *taf3= NULL, *taf4= NULL;
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
		raw= (args[2])? True : False;
		Scale= args[3];
		shiftX= args[4];
		shiftY= args[5];
		rotX= args[6];
		rotY= args[7];
		angle= args[8];
		rdx= args[9];
		if( ascanf_arguments> 10 ){
			taf1= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_fig_dist_shiftrot", ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 11 ){
			taf2= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_fig_dist_shiftrot", ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 12 ){
			if( !(taf3= parse_ascanf_address( args[12], _ascanf_array, "ascanf_fig_dist_shiftrot", ascanf_verbose, NULL)) ||
				!taf3->iarray
			){
				fprintf( StdErr, " (ignored 12th argument (%s->%%closest12) that is not a pointer to an integer array!) ",
					ad2str( args[12], d3str_format, NULL)
				);
				fflush( StdErr );
			}
			else if( !ascanf_arg_error ){
				Resize_ascanf_Array( taf3, AllSets[idx1].numPoints, NULL );
				closest12= taf3->iarray;
			}
		}
		if( ascanf_arguments> 13 ){
			if( !(taf4= parse_ascanf_address( args[13], _ascanf_array, "ascanf_fig_dist_shiftrot", ascanf_verbose, NULL)) ||
				!taf4->iarray
			){
				fprintf( StdErr, " (ignored 13th argument (%s->%%closest21) that is not a pointer to an integer array!) ",
					ad2str( args[13], d3str_format, NULL)
				);
				fflush( StdErr );
			}
			else if( !ascanf_arg_error ){
				Resize_ascanf_Array( taf4, AllSets[idx2].numPoints, NULL );
				closest21= taf4->iarray;
			}
		}
		if( !ascanf_arg_error && idx1>= 0 && idx2>= 0 ){
		  int scale= (NaN(Scale))? False : True;
			shift= (NaN(shiftX) || NaN(shiftY))? False : True;
			rotate= (NaN(rotX) || NaN(rotY) || NaN(angle))? False : True;
			*result= figural_distance_shiftrot( (ActiveWin && ActiveWin!= StubWindow_ptr)? ActiveWin : NULL, idx1, idx2,
				scale, Scale, shift, shiftX, shiftY, rotate, rotX, rotY, angle, rdx,
				&spatial, &orientation, raw, closest12, closest21
			);
			if( taf1 ){
				taf1->value= spatial;
			}
			if( taf2 ){
				taf2->value= orientation;
			}
		}
		else{
			*result= -1;
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
		ascanf_arg_error= 0;
		ascanf_emsg= NULL;
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
			fprintf( StdErr, " (set number %s out of range  (max %d)) ", 
				ad2str( args[0], d3str_format, 0), setNumber
			);
			ascanf_arg_error= True;
			idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< AllSets[idx1].numPoints || ascanf_SyntaxCheck) ){
			pnt_nr= (int) args[1];
		}
		else{
			ascanf_emsg= " (point index out of range) ";
			fprintf( StdErr, " (invalid point %s for set %d has %d points) ", 
				ad2str( args[1], d3str_format, 0), idx1, AllSets[idx1].numPoints
			);
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

static double scale_bounds[2];
static double *fd_bound_rotation_centre, *fd_use_provided;
static int mfg_brc, mfg_bsc, mfg_upr;

double figdistshiftrot( double args[7] )
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
	if( mfg_bsc ){
		if( args[6]< scale_bounds[0] ){
			args[6]= scale_bounds[0];
		}
		else if( args[6]> scale_bounds[1] ){
			args[6]= scale_bounds[1];
		}
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
	return( (fdsr_last_args[7]= figural_distance_shiftrot( mfg_wi, mfg_idx1, mfg_idx2,
		True, args[6], True, args[1], args[2], True, args[3], args[4], args[5], mfg_radix,
		&mfg_spatial, &mfg_orientation, mfg_raw, NULL, NULL
	)) );
}

double figdistrot( double args[5] )
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
	if( mfg_bsc ){
		if( args[4]< scale_bounds[0] ){
			args[4]= scale_bounds[0];
		}
		else if( args[4]> scale_bounds[1] ){
			args[4]= scale_bounds[1];
		}
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
	return( (fdsr_last_args[7]= figural_distance_shiftrot( mfg_wi, mfg_idx1, mfg_idx2,
		True, args[4], True, mfg_shiftX, mfg_shiftY, True, args[1], args[2], args[3], mfg_radix,
		&mfg_spatial, &mfg_orientation, mfg_raw, NULL, NULL
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

static void fd_initial_simplex( double **P, double *bestP, double *fdP, double (*minimise)(double *),
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
	   \ dimensions are just ignored by the figdistshiftrot(), and we discard any non-neutral values
	   \ that continuous_simanneal() might have concocted.
	   */
	if( ascanf_verbose == 2 ){
	  /* (never happens?!) */
		fprintf( StdErr, "\n Initial guess parameters: shiftX=%g, shiftY=%g, rotX=%g, rotY=%g, angle=%g scale=%g\n",
			shiftX, shiftY, rotX, rotY, angle, escale
		);
		if( mfg_bsc ){
			fprintf( StdErr, " Scale bounds: (%s,%s)\n",
				d2str(scale_bounds[0],0,0), d2str(scale_bounds[1],0,0)
			);
		}
		if( *fd_bound_rotation_centre ){
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

int ascanf_minimal_figdist( ASCB_ARGLIST )
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
		mfg_raw= (args[2])? True : False;
		mfg_scale= (args[3])? True : False;
		mfg_shift= (args[4])? True : False;
		CLIP_EXPR_CAST( int, mfg_rotate, double, args[5], -MAXINT, MAXINT );
		mfg_radix= args[6];
		Iter= parse_ascanf_address( args[7], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		ftol= args[8];
		CLIP_EXPR_CAST( int, restarts, double, args[9], 0, MAXINT );
		if( (Rscale= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL )) ){
			mfg_Scale= Rscale->value;
		}
		if( (RshiftX= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL )) ){
			mfg_shiftX= RshiftX->value;
		}
		if( (RshiftY= parse_ascanf_address( args[12], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL )) ){
			mfg_shiftY= RshiftY->value;
		}
		if( (RrotX= parse_ascanf_address( args[13], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL )) ){
			mfg_rotX= RrotX->value;
		}
		if( (RrotY= parse_ascanf_address( args[14], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL )) ){
			mfg_rotY= RrotY->value;
		}
		if( (Rangle= parse_ascanf_address( args[15], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL )) ){
			mfg_angle= Rangle->value;
		}
		if( ascanf_arguments> 16 ){
			taf1= parse_ascanf_address( args[16], _ascanf_variable, "ascanf_fig_dist_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 17 ){
			taf2= parse_ascanf_address( args[17], _ascanf_variable, "ascanf_fig_dist_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( mfg_idx1>= 0 && mfg_idx2>= 0 && !ascanf_arg_error && Iter && !ascanf_SyntaxCheck ){
		  DataSet *set1= &AllSets[mfg_idx1];
		  DataSet *set2= &AllSets[mfg_idx2];
		  int iter= Iter->value, i, ndim= 6, r;
		  double x, y, bestfdP, shiftX, shiftY, rotX, rotY, angle, gravX1, gravY1, gravX2, gravY2;
		  double minX, maxX, minY, maxY, escale, Dscale;
		  ALLOCA( fdP, double, ndim+2, fdPlen );
		  ALLOCA( bestP, double, ndim+1, bestPlen );
		  ALLOCA( P1, double, ndim+1, P1len );
		  ALLOCA( P2, double, ndim+1, P2len );
		  ALLOCA( P3, double, ndim+1, P3len );
		  ALLOCA( P4, double, ndim+1, P4len );
		  ALLOCA( P5, double, ndim+1, P5len );
		  ALLOCA( P6, double, ndim+1, P6len );
		  ALLOCA( P7, double, ndim+1, P7len );
		  ALLOCA( P, double*, ndim+2, Plen );
		  double (*minimise)( double*)= figdistshiftrot, cl1= 0, cl2= 0;

			mfg_brc= (int) *fd_bound_rotation_centre;
			mfg_bsc= (!Inf(scale_bounds[0]) || !Inf(scale_bounds[1]))? 1 : 0;

			if( !(mfg_upr= (int) *fd_use_provided) ){
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
					bestfdP= figural_distance( mfg_wi, mfg_idx1, mfg_idx2,
						&mfg_spatial, &mfg_orientation, mfg_raw
					);
					iter= 0;
					goto minimised;
				}
				else{
					  /* use a different subroutine since we skip the 1st two dimensions (shiftX,shiftY) */
					minimise= figdistrot;
					ndim= 4;
				}
			}
			  /* when not rotating we can ignore the last few dimensions. this works even though
			   \ the minimising function will try to vary them...
			   */
/* 			if( !mfg_rotate ){	*/
/* 				minimise= figdistshift;	*/
/* 				ndim= 3;	*/
/* 			}	*/

			  /* simple check whether we got the memory. On a number of systems, ALLOCA (through alloca) won't
			   \ fail (= we'll die at once when it does). On others, we'll just suppose that P will be NULL too
			   \ when one of the preceding allocations failed...
			   */
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

			fd_initial_simplex( P, bestP, fdP, minimise, shiftX, shiftY, rotX, rotY, 0, escale,
				minX, minY, maxX, maxY, Dscale
			);


			  /* Initialise the best function value ever with something large. This means a safe margin larger than
			   \ the figural distance at the "raw" configuration. The minimising function will in fact use this value
			   \ as a reference: if the initial value is lower than the real (global) minimum, no action occurs!
			   */
			bestfdP= 10* fdP[2];
			if( ascanf_verbose ){
				fprintf( StdErr,
					"\nRef: set %d, Adapt: set %d; %s mode; %s, %s%s %s %s orientation, do scale,shift,rotate=%d%s,%d,%d; %s rotation centre\n"
					" starting with best value 10*<unshifted>=%s; distance at gravity point (shift %g,%g,%g,%g,%g) %s\n",
					mfg_idx1, mfg_idx2, (mfg_raw)? "raw" : "cooked",
					(*fd_pointwise)? "point-wise" : "Conditt",
					(*fd_normalise_spatial)? "normalised spatial, " : "",
					(*fd_orientation)? "using" : "ignoring",
					(*fd_absolute_orientation)? "absolute" : "signed",
					(*fd_normalise_orientation)? "normalised" : "radians",
					mfg_scale,
					(mfg_bsc)? "(bounded)" : "",
					mfg_shift, mfg_rotate,
					(*fd_bound_rotation_centre)? "bounded" : "unbounded",
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
					" minimal figural distance (t=0): min@(%g,%g,%g,%g,%g,%g) == (%g,%g)=%g in %d iterations\n"
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
				fd_initial_simplex( P, bestP, fdP, minimise, shiftX, shiftY, rotX, rotY, 0, escale,
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
						"\n R%d minimal figural distance (t=0): min@(%g,%g,%g,%g,%g,%g) == (%g,%g)=%g in %d iterations\n"
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
		}
	}
	else{
bail_out:;
		ascanf_arg_error= True;
	}
	GCA();
	return( !ascanf_arg_error );
}

int ascanf_turtle_dist( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *reference= NULL, *test= NULL;
  double K;
	ascanf_arg_error= 0;
	if( ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	if( !(reference= parse_ascanf_address( args[0], _ascanf_array, "ascanf_turtle_dist", (int) ascanf_verbose, NULL )) ){
		ascanf_emsg= " (1st argument should point to an array) ";
		ascanf_arg_error= True;
	}
	if( !(test= parse_ascanf_address( args[1], _ascanf_array, "ascanf_turtle_dist", (int) ascanf_verbose, NULL ))
		|| test->N!= reference->N
	){
		ascanf_emsg= " (2nd argument should point to an array of the same dimension as the 1st) ";
		ascanf_arg_error= True;
	}
	if( ascanf_arguments> 2 ){
		K= args[2];
	}
	else{
		K= 1;
	}
	if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
	  int i;
	  double dist= 0, len= 0;
#define ARRVAL(a,i)	((a->iarray)? a->iarray[i] : a->array[i])

		for( i= 0; i< reference->N; i++ ){
			dist+= ARRVAL(test,i) - ARRVAL(reference,i);
			len+= ARRVAL(reference,i);
		}
		if( (len< 0 && K> 0) || (len>= 0 && K< 0) ){
			K*= -1;
		}
		*result= (dist) / (len + K);
	}
	else{
		set_NaN(*result);
	}
	return( !ascanf_arg_error );
}

int ascanf_fig_dist_arrays( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int raw;
  ascanf_Function *taf1= NULL, *taf2= NULL, *X1, *Y1, *O1= NULL, *X2, *Y2, *O2= NULL;
  double spatial, orientation;
	if( ascanf_arguments>= 7 ){
		if( !(X1= parse_ascanf_address( args[0], _ascanf_array, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL)) ||
			!X1->array
		){
			ascanf_emsg= " (X1 argument must be a double array) ";
			ascanf_arg_error= 1;
		}
		if( !(Y1= parse_ascanf_address( args[1], _ascanf_array, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL)) ||
			!Y1->array || Y1->N!= X1->N
		){
			ascanf_emsg= " (Y1 argument must be a double array of X1's dimension) ";
			ascanf_arg_error= 1;
		}
		if( (O1= parse_ascanf_address( args[2], _ascanf_array, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL)) &&
			(!O1->array || O1->N!= X1->N)
		){
			ascanf_emsg= " (O1 argument must be a double array of X1's dimension, or 0) ";
			ascanf_arg_error= 1;
		}
		if( !(X2= parse_ascanf_address( args[3], _ascanf_array, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL)) ||
			!X2->array
		){
			ascanf_emsg= " (X2 argument must be a double array) ";
			ascanf_arg_error= 1;
		}
		if( !(Y2= parse_ascanf_address( args[4], _ascanf_array, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL)) ||
			!Y2->array || Y2->N!= X2->N
		){
			ascanf_emsg= " (Y2 argument must be a double array of X2's dimension) ";
			ascanf_arg_error= 1;
		}
		if( (O2= parse_ascanf_address( args[5], _ascanf_array, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL)) &&
			(!O2->array || O2->N!= X2->N)
		){
			ascanf_emsg= " (O2 argument must be a double array of X2's dimension, or 0) ";
			ascanf_arg_error= 1;
		}
		raw= (args[6])? True : False;
		if( ascanf_arguments> 7 ){
			taf1= parse_ascanf_address( args[7], _ascanf_variable, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 8 ){
			taf2= parse_ascanf_address( args[8], _ascanf_variable, "ascanf_fig_dist_arrays", (int) ascanf_verbose, NULL );
		}
		if( !ascanf_arg_error ){
			*result= fig_dist( NULL, NULL, NULL,
				X1->array, Y1->array, (O1)? O1->array : NULL, X1->N,
				X2->array, Y2->array, (O2)? O2->array : NULL, X2->N,
				raw, raw, (O1 && O2)? True : False, &spatial, &orientation, NULL, NULL
			);
			if( taf1 ){
				taf1->value= spatial;
			}
			if( taf2 ){
				taf2->value= orientation;
			}
		}
		else{
			*result= -1;
		}
	}
	else{
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

static ascanf_Function fig_dist_Function[] = {
	{ "$figural-distance-point-wise", NULL, 2, _ascanf_variable,
		"$figural-point-wise: If set, the figural distance (spatial and angular) is calculated as the average\n"
		" distance between corresponding (paired) points in the 2 sets. Otherwise, the figural distance is\n"
		" calculated as defined by Conditt & al: for each point, find the point in the other set for which the\n"
		" distance is smallest. Then, calculate the average minimal distance over all points in set1 to those in set2,\n"
		" idem for set2 vs. set1, and define the figural distance as the average of those 2 average min. distances.\n"
		" Obviously, when both sets have the same number of points, both methods should and do give the same result\n"
		" when the 2 sets have a reasonable overlap (i.e. when the nearest point in set A is not the same for all\n"
		" points in set B). The point-wise method is faster, and easier to minimise.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$figural-distance-orientation", NULL, 2, _ascanf_variable,
		"$figural-distance-orientation: whether or not to include the orientation (set's ecol) in the calculation\n"
		" of the figural distance.\n", 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$figural-distance-normalise-spatial", NULL, 2, _ascanf_variable,
		"$figural-distance-normalise-spatial: normalise the calculated spatial figural distance\n"
		" to the calculated angular distance by multiplying it with the angular distance.\n"
		" This is an approximation of a complementary option to $figural-distance-normalise-orientation,\n"
		" with which it is mutually exclusive (overridden by it).\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$figural-distance-normalise-orientation", NULL, 2, _ascanf_variable,
		"$figural-distance-normalise-orientation: normally, orientations (error columns...) are included\n"
		" in the figural-distance computation after a conversion from the current radix to radix \\R\\\n"
		" (NB: the current radix is set via the Settings dialog, and not via the radix of the rotation\n"
		" to be applied to the adjustable set!). When this variable is set to 1, orientations are normalised\n"
		" with respect to the current spatial figural distance, such that an average orientation difference\n"
		" of PI (180 degrees) is equalised to the current spatial distance. If that spatial distance is 0,\n"
		" the orientation difference is expressed in radix \\R\\ (see $figural-distance-consistent-orientation-normalisation).\n"
		" For a value of 2, normalisation is performed with respect to 2*(1+spatial) [instead of 2*spatial].\n"
		" This ensures a smooth transition around spatial approx. 0\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$figural-distance-consistent-orientation-normalisation", NULL, 2, _ascanf_variable,
		"$figural-distance-consistent-orientation-normalisation: determines the radix \\R\\ of the\n"
		" angular figural distance. Originally, this was 2\\p\\, i.e. radians, but this caused an\n"
		" inconsistency when normalisation was off, or 1 with zero spatial distance. Therefore, the\n"
		" radix is now 1. Set this variable to 0 to revert to the old behaviour.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$figural-distance-absolute-orientation", NULL, 2, _ascanf_variable,
		"$figural-distance-absolute-orientation: normally, difference in orientation (error columns...)\n"
		" between 2 points is the absolute difference (like the spatial difference) (variable set to True).\n"
		" When unset (False), signed orientation differences are taken into account, such that the minimal\n"
		" distance found between any 2 points can be negative. This has the potential advantage that the\n"
		" average minimal (=figural) distance can be 0 if the orientations have random variations around an\n"
		" identical mean.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$figural-distance-bound-rotation-centre", NULL, 2, _ascanf_variable,
		"$figural-distance-bound-rotation-centre: determines if the (rotX,rotY) centre of rotation is bounded\n"
		" to be within the adjustable set's X and Y range during figural-distance minimisation.\n"
		" If not set, (rotX,rotY) can vary freely, which means the set can be shifted even when the <shift>\n"
		" argument to minimal-figural-distance[] is False. When bounded, such shifts can still occur, but the\n"
		" rotation centre can only vary \"within\" the set.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$figural-distance-scale-bounds", NULL, 2, _ascanf_array,
		"$figural-distance-scale-bounds: determines if the scale is bounded\n"
		" by the specified minimal and maximal values. ([-Inf,Inf]: not bounded...)\n"
		, 1, 0, 0, 0, 0, 0, 0.0, &scale_bounds[0], NULL, NULL, NULL, NULL, 0, 0, 2
	},
	{ "$figural-distance-use-provided-values", NULL, 2, _ascanf_variable,
		"$figural-distance-use-provided-values: by default, minimal-figural-distance[] ignores the values\n"
		" pointed to by the return pointers (&Rscale, &shiftX, etc), using those pointers only to return\n"
		" the parameter values for which it found a minimal figural distance. When this flag is set, however,\n"
		" it will use the values pointed to for those parameters for which the corresponding minisimation flag\n"
		" is False. Thus, when <rotate> is false, a minimal distance will be found for the values passed in\n"
		" (&RrotX,&RrotY,&Rangle), etc. Special case: rotate==-1; in this case, only &Rangle is used, finding a\n"
		" minimal distance at that particular rotation angle.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$figural-distance-transformation", NULL, 2, _ascanf_variable,
		"$figural-distance-transformation: specifies a transformation to be applied to each individual\n"
		" (minimal or point-to-point) distance used for calculating the figural distance. Current possibilities are:\n"
		"   0: none\n"
		"   1: power transformation\n"
		"   2: logarithm (with log(0)== 0...!)\n"
		" In all cases, sign-preserving transformations are used; i.e. sqrt(-2)= -sqrt(2), etc.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$figural-distance-transformation-parameters", NULL, 2, _ascanf_variable,
		"$figural-distance-transformation-parameters: currently, a single parameter that applies to the\n"
		" selected transformation:\n"
		" * the power for power transformations (2 for square, 0.5 for square-root: positive values only!)\n"
		" * the logarithm's base for log transforms (value <=1 means natural, e-based, logarithm).\n"
		, 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$figural-distance-spatial-direction", NULL, 2, _ascanf_variable,
		"$figural-distance-spatial-direction: the average direction between the selected closest points.\n"
		" Gives the direction from a point of set1 to the point of set2.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "figural-distance", ascanf_fig_dist, 5, NOT_EOF_OR_RETURN,
		"figural-distance[set1,set2,raw[,&spatial,&orientation]]\n"
		" A routine that calculates the figural distance between 2 sets. For 2 sets in vector mode\n"
		" (errors are orientations), the value returned is the average of the spatial figural distance\n"
		" (the average of the distances between all pairs of points), and the angular figural distance\n"
		" (the average of the orientation differences). The 2 components are returned in &spatial and\n"
		" &orientation if these pointers are provided. When one or both of the sets have another setting\n"
		" for the error column, only the spatial figural distance is determined.\n"
		" After Conditt &al, J. Neurophysiol 78-1 1997.\n"
	},
	{ "figural-distance-shiftrot", ascanf_fig_dist_shiftrot, 14, NOT_EOF_OR_RETURN,
		"figural-distance-shiftrot[set1, set2, raw, scale, shiftX, shiftY, rotX, rotY, angle, radix\n"
		"                          [,&spatial,&orientation,&%closest12,&%closest21]]\n"
		" This calculates the figural distance after a rotation over <angle> (base <radix>) around (rotX,rotY) and a\n"
		" subsequent shift over (shiftX,shiftY) of the 2nd set. The set is first scaled over <scale>.\n"
		" When set2 is in vector mode (error column contains orientations), the orientations are rotated over <angle>,\n"
		" otherwise, the error column is not modified. The other arguments are as described under figural-distance[].\n"
		" When either of the scale/shift/rotation parameters is a NaN, the corresponding operation(s) is/are not performed.\n"
		" The optional <%closest12> <%closest21> integer array pointers will return the pairs of closest points found\n"
		" (uniquely based on the spatial figural distance).\n"
	},
	{ "Set-shiftrot", ascanf_setshiftrot, 12, NOT_EOF,
		"Set-shiftrot[set,pnt_nr,scale,shiftX,shiftY,rotX,rotY,angle,radix, &rX, &rY, &rE]\n"
		" Return a set's co-ordinates (X,Y,E) at point <pnt_nr> after a rotation of <angle> (base <radix>)\n"
		" around (rotX,rotY), and a shift over (shiftX,shiftY). The transformed values are stored in\n"
		" &rX, &rY, &rE. If the set's error column represent orientations (vector mode), rE=E+angle;\n"
		" otherwise, rE=E.\n"
	},
	{ "minimal-figural-distance", ascanf_minimal_figdist, 18, NOT_EOF_OR_RETURN,
		"minimal-figural-distance[set1,set2,raw,scale,shift,rotate,radix,&iterations,fractol,restarts,&Rscale,&RshiftX,&RshiftY,&RrotX,&RrotY,&Rangle[,&spat,&orn]]\n"
		" Find the minimal figural distance by adjusting the shift and rotation parameters as described for\n"
		" figural-distance-shiftrot[]. Iterations contains the number of iterations to run, and returns the\n"
		" number actually performed (over all restarts); fractol is the fractional tolerance for the solution; the other pointers\n"
		" contain the final shift&rot parameters that result in the minimal distance.\n"
		" Shifting is only done when <shift>, rotation only when <rotate>; when both are False, no minimisation takes place...\n"
		" See also $figural-distance-use-provided-values.\n"
		" NB: when shift=False, rotX and rotY are still subject to change (cf. $figural-distance-bound-rotation-centre)!\n"
	},
	{ "turtle-distance", ascanf_turtle_dist, 3, NOT_EOF_OR_RETURN,
		"turtle-distance[&reference,&test[,K]]: calculates\n"
		"     sum(test[i]-reference[i])/(sum(reference[i]) + K)\n"
		" where K is some constant (default 1) that determines the resolution and prevents zero division.\n"
		" When sum(reference[i]) and K are of opposite sign, K is multiplied by -1.\n"
	},
	{ "figural-distance-arrays", ascanf_fig_dist_arrays, 9, NOT_EOF_OR_RETURN,
		"figural-distance-arrays[&X1,&Y1,&O1,&X2,&Y2,&O2,raw[,&spatial,&orientation]]\n"
		" A routine that calculates the figural distance between 2 pairs (triplets) of arrays. If O1 and O2 are not 0,\n"
		" (orientations), the value returned is the average of the spatial figural distance\n"
		" (the average of the distances between all pairs of points), and the angular figural distance\n"
		" (the average of the orientation differences). The 2 components are returned in &spatial and\n"
		" &orientation if these pointers are provided. Otherwise, only the spatial figural distance is determined.\n"
		" This is the equivalent of figural-distance[].\n"
		" After Conditt &al, J. Neurophysiol 78-1 1997.\n"
	},
};

static double *fd_pointwise= &fig_dist_Function[0].value;
static double *fd_orientation= &fig_dist_Function[1].value;
static double *fd_normalise_spatial= &fig_dist_Function[2].value;
static double *fd_normalise_orientation= &fig_dist_Function[3].value;
static double *fd_consistent_norm_orn= &fig_dist_Function[4].value;
static double *fd_absolute_orientation= &fig_dist_Function[5].value;
static double *fd_bound_rotation_centre= &fig_dist_Function[6].value;
static double *fd_use_provided= &fig_dist_Function[8].value;
static double *fd_transform_figdist= &fig_dist_Function[9].value;
static double *fd_transform_figdist_param= &fig_dist_Function[10].value;
static ascanf_Function *fd_figdist_dirn= &fig_dist_Function[11];

static int fig_dist_Functions= sizeof(fig_dist_Function)/sizeof(ascanf_Function);

static void print_settings(FILE *fp)
{
	fprintf( fp, "Current figural-distance settings: pw=%g orn=%g nS=%g nO=%g absO=%g bRctr=%g upv=%g transf=%g,%g\n",
		*fd_pointwise, *fd_orientation, *fd_normalise_spatial, *fd_normalise_orientation,
		*fd_absolute_orientation, *fd_bound_rotation_centre, *fd_use_provided,
		*fd_transform_figdist, *fd_transform_figdist_param
	);
}

static unsigned int tsa_calls= 0;
static double tsa( double args[3] )
{ double x= args[1]- 10, y= args[2]- 2;
	tsa_calls+= 1;
	return( 1+ ( x*x + y*y ) );
}

int test_simanneal( int sa )
{ double fdP[4], bestP[3], bestfdP, ftol= DBL_EPSILON;
  int i, iter= 1000;
  ALLOCA( P, double*, 4, Plen );
  ALLOCA( P1, double, 3, P1len );
  ALLOCA( P2, double, 3, P2len );
  ALLOCA( P3, double, 3, P3len );
  double T= 0;
  Time_Struct timer;

	if( P ){
		P[0]= NULL;
		P[1]= P1, P[2]= P2, P[3]= P3;
	}
	else{
		return(0);
	}

	  /* The starting simplex must cover the range within which the minimum is to be found?! */
	P[1][1]= P[1][2]= 20;
	fdP[1]= tsa( P[1] );
	P[2][1]= 15, P[2][2]= 3;
	fdP[2]= tsa( P[2] );
	P[3][1]= -20, P[3][2]= -20;
	fdP[3]= tsa( P[3] );
	memset( bestP, 0, sizeof(bestP) );
	bestfdP= 1e9;
	iter= 1000;
	Elapsed_Since( &timer, True );
	if( sa ){
	  double temp=20;
		do{
			iter= 1000;
			tsa_calls= 0;
			Elapsed_Since( &timer, False );
			continuous_simanneal( P, fdP, 2, bestP, &bestfdP, ftol, tsa, &iter, temp );
			Elapsed_Since( &timer, False );
			T+= timer.HRTot_T;
			fprintf( StdErr,
				"test simanneal T=%g, 1+(x-10)**2+(y-2)**2; min@(%g,%g) == %g in %d iterations, %u calls\n"
				" simplex: (%g,%g)(%g,%g)(%g,%g)\n",
				temp,
				bestP[1], bestP[2], bestfdP, iter, tsa_calls,
				P[1][1], P[1][2], P[2][1], P[2][2], P[3][1], P[3][2]
			);
			temp/= 10.0;
		} while( iter< 0 && temp> 2e-8 );
	}
	else{
		tsa_calls= 0;
		iter= 1000;
		continuous_simanneal( P, fdP, 2, bestP, &bestfdP, ftol, tsa, &iter, 0 );
		Elapsed_Since( &timer, False );
		T+= timer.HRTot_T;
		fprintf( StdErr,
			"test simanneal t=0,ftol=%g, 1+(x-10)**2+(y-2)**2; min@(%g,%g) == %g in %d iterations, %u calls\n"
			" simplex: (%g,%g)(%g,%g)(%g,%g)\n",
			ftol, bestP[1], bestP[2], bestfdP, iter, tsa_calls,
			P[1][1], P[1][2], P[2][1], P[2][2], P[3][1], P[3][2]
		);
	}
	fprintf( StdErr, "Elapsed time: %gs\n", T );
	GCA();
	return(iter);
}

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= fig_dist_Function;
  static char called= 0;
  int i, warn= True;
  char buf[64];

	set_Inf(scale_bounds[0], -1 );
	set_Inf(scale_bounds[1], 1 );
	for( i= 0; i< fig_dist_Functions; i++, af++ ){
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
		}
		af->dymod= new;
	}
	if( fd_figdist_dirn && _ascanf_SAS_StatsBin( fd_figdist_dirn, 1 ) ){
		fd_SAS_figdist_dirn= fd_figdist_dirn->SAS;
	}
	else{
		fd_SAS_figdist_dirn= NULL;
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
		XGRAPH_FUNCTION( _ascanf_SAS_StatsBin_ptr, "_ascanf_SAS_StatsBin");
		XGRAPH_FUNCTION( SAS_Add_Data_angsincos_ptr, "SAS_Add_Data_angsincos")
		XGRAPH_FUNCTION( conv_angle__ptr, "conv_angle_");
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

	if( !initialised ){
		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( fig_dist_Function, fig_dist_Functions, "fig_dist::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-figural-distance" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A library with routines for determining the figural distance between 2 sets,\n"
		" and (in the end) the minimal figural distance by shifting and rotating the 2nd\n"
		" set with respect to the 1st.\n"
	);

	sran1( (long) time(NULL) );

	if( debugFlag ){
		test_simanneal(True);
		test_simanneal(False);
	}

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
	  int r= remove_ascanf_functions( fig_dist_Function, fig_dist_Functions, force );
		if( force || r== fig_dist_Functions ){
			for( i= 0; i< fig_dist_Functions; i++ ){
				fig_dist_Function[i].dymod= NULL;
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
				r, fig_dist_Functions
			);
		}
	}
	return(ret);
}
