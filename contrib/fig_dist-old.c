#ifndef XG_DYMOD_SUPPORT
#	define XG_DYMOD_SUPPORT
#endif

#include <stdio.h>
#include <stdlib.h>
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

#include <errno.h>
#ifndef linux
	extern char *sys_errlist[];
	extern int sys_nerr;
#endif

#include "xgraph.h"
#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"

#ifndef MIN
#	define MIN(a, b)	((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#	define MAX(a, b)	((a) > (b) ? (a) : (b))
#endif

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc).
   */
extern int ascanf_Variable();

  /* Get the dynamic module definitions:	*/
#include "dymod.h"

#include "simanneal.h"

extern double *fd_normalise_orientation;

/* Base routine for calculating the figural distance as defined by Conditt &al, J. Neurophysiol 78-1 1997.	*/
static double fig_dist( LocalWin *wi, DataSet *set1, DataSet *set2, double *SumDist, int *K, int raw1, int raw2, int orn_handling )
{ int k, l;
  double dist, mindist;

	*SumDist= 0;
	*K= 0;
	for( k= 0; k< set1->numPoints; k++ ){
		if( !DiscardedPoint( wi, set1, k) ){
		  double x1, y1, o1;
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
			mindist= -1;
			for( l= 0; l< set2->numPoints; l++ ){
				if( !DiscardedPoint( wi, set2, l) ){
				  double x2, y2, o2;
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
					if( orn_handling== 2 ){
					  /* Calculate the difference between 2 angles: */
					  extern double radix;
					  double rdx= (wi)? wi->radix : radix;
						dist= fabs(o2 - o1)/ ( (rdx)? rdx : M_2PI );
/* 						dist= fabs(o2 - o1);	*/
						if( NaN(dist) ){
							dist= 0;
						}
						else if( dist> 0.5 ){
							dist= fabs( dist- 1 );
						}
					}
					else if( orn_handling== 0 || NaN(o1) || NaN(o2) ){
						x2-= x1;
						y2-= y1;
						dist= sqrt( x2*x2 + y2*y2 );
					}
					else{
						x2-= x1;
						y2-= y1;
						o2-= o1;
						dist= sqrt( x2*x2 + y2*y2 + o2*o2 );
					}
					mindist= (mindist== -1)? dist : MIN(mindist, dist);
				}
			}
			if( mindist>= 0 ){
				*K+= 1;
				*SumDist+= mindist;
			}
		}
	}
	if( orn_handling== 2 && !*fd_normalise_orientation ){
	  /* If we were looking at orientations, and they should not be normalised, convert
	   \ the found sum-of-distances to radians:
	   */
		(*SumDist)*= M_2PI;
	}
}

extern double *fd_orientation;

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
	{ int K, L;
	  double SumDist1, SumDist2;
		fig_dist( wi, &AllSets[idx1], &AllSets[idx2], &SumDist1, &K, raw, raw, 0 );
		fig_dist( wi, &AllSets[idx2], &AllSets[idx1], &SumDist2, &L, raw, raw, 0 );
		*spatial= (SumDist1 + SumDist2)/ (K + L);
		if( all_vectors && *fd_orientation ){
			fig_dist( wi, &AllSets[idx1], &AllSets[idx2], &SumDist1, &K, raw, raw, 2 );
			fig_dist( wi, &AllSets[idx2], &AllSets[idx1], &SumDist2, &L, raw, raw, 2 );
			*orientation= (SumDist1 + SumDist2)/ (K + L);
			if( *fd_normalise_orientation ){
				*orientation *= (*spatial)? 2* *spatial : M_2PI;
			}
/* 			return( (*spatial + *orientation) / 2 );	*/
			return( sqrt(*spatial * *spatial + *orientation * *orientation) );
		}
		else{
			*orientation= -1;
			return( *spatial );
		}
	}
}

double figural_distance_shiftrot( LocalWin *wi, int idx1, int idx2,
	int scale, double Scale,
	int shift, double shiftX, double shiftY, int rotate, double rotX, double rotY, double angle, double radix,
	double *spatial, double *orientation, int raw )
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
	{ int K, L, i;
	  double SumDist1, SumDist2, ret;
	  double ang_sin, ang_cos;
	  DataSet *set2= &AllSets[idx2];
	  ALLOCA( txvec, double, set2->numPoints, txvec_len );
	  ALLOCA( tyvec, double, set2->numPoints, tyvec_len );
	  ALLOCA( terrvec, double, set2->numPoints, terrvec_len );
	  double *sxvec= set2->xvec, *syvec= set2->yvec, *serrvec= set2->errvec;

		if( !radix ){
			radix= M_2PI;
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
			ang_sin= sin( M_2PI * angle/ radix );
			ang_cos= cos( M_2PI * angle/ radix );
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

		  /* Calculate the component(s) of the figural distance: */
		fig_dist( wi, &AllSets[idx1], set2, &SumDist1, &K, raw, 0, 0 );
		fig_dist( wi, set2, &AllSets[idx1], &SumDist2, &L, 0, raw, 0 );
		*spatial= (SumDist1 + SumDist2)/ (K + L);
		if( all_vectors && *fd_orientation ){
			fig_dist( wi, &AllSets[idx1], set2, &SumDist1, &K, raw, 0, 2 );
			fig_dist( wi, set2, &AllSets[idx1], &SumDist2, &L, 0, raw, 2 );
			*orientation= (SumDist1 + SumDist2)/ (K + L);
			if( *fd_normalise_orientation ){
				*orientation *= (*spatial)? 2* *spatial : M_2PI;
			}
			  /* the composite figural distance: */
			ret= ( sqrt(*spatial * *spatial + *orientation * *orientation) );
		}
		else{
			*orientation= -1;
			ret= ( *spatial );
		}
		  /* restore the set's own transformed buffers */
		set2->xvec= sxvec;
		set2->yvec= syvec;
		set2->errvec= serrvec;
		return(ret);
	}
}

extern LocalWin *ActiveWin, StubWindow;

int ascanf_fig_dist( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx1, idx2, raw;
  ascanf_Function *taf1= NULL, *taf2= NULL;
  double spatial, orientation;
	if( ascanf_arguments>= 3 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			idx1= (int) args[0];
		}
		else{
			ascanf_emsg= " (setnumber 1 out of range) ";
			ascanf_arg_error= True;
			idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< setNumber || ascanf_SyntaxCheck) ){
			idx2= (int) args[1];
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
			*result= figural_distance( (ActiveWin && ActiveWin!= &StubWindow)? ActiveWin : NULL, idx1, idx2,
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
  int idx1, idx2, raw;
  ascanf_Function *taf1= NULL, *taf2= NULL;
  double spatial, orientation, shiftX, shiftY, rotX, rotY, angle, radix, Scale;
  int shift, rotate;
	if( ascanf_arguments>= 10 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			idx1= (int) args[0];
		}
		else{
			ascanf_emsg= " (setnumber 1 out of range) ";
			ascanf_arg_error= True;
			idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< setNumber || ascanf_SyntaxCheck) ){
			idx2= (int) args[1];
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
		radix= args[9];
		if( ascanf_arguments> 10 ){
			taf1= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_fig_dist_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 11 ){
			taf2= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_fig_dist_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( !ascanf_arg_error && idx1>= 0 && idx2>= 0 ){
		  int scale= (NaN(Scale))? False : True;
			shift= (NaN(shiftX) || NaN(shiftY))? False : True;
			rotate= (NaN(rotX) || NaN(rotY) || NaN(angle))? False : True;
			*result= figural_distance_shiftrot( (ActiveWin && ActiveWin!= &StubWindow)? ActiveWin : NULL, idx1, idx2,
				scale, Scale, shift, shiftX, shiftY, rotate, rotX, rotY, angle, radix,
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

int ascanf_setshiftrot( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx1, pnt_nr;
  double shiftX, shiftY, rotX, rotY, angle, radix, Scale;
  ascanf_Function *rx, *ry, *ro;
	if( ascanf_arguments>= 12 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			idx1= (int) args[0];
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
		radix= args[8];
		rx= parse_ascanf_address( args[9], _ascanf_variable, "ascanf_setshiftrot", (int) ascanf_verbose, NULL );
		ry= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_setshiftrot", (int) ascanf_verbose, NULL );
		ro= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_setshiftrot", (int) ascanf_verbose, NULL );
		if( rx && ry && ro && idx1>= 0 && pnt_nr>= 0 && !ascanf_arg_error ){
		  DataSet *set= &AllSets[idx1];
		  double ang_sin, ang_cos;
		  double x, y, o;

			if( !radix ){
				radix= M_2PI;
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
				ang_sin= sin( M_2PI * angle/ radix );
				ang_cos= cos( M_2PI * angle/ radix );
			}
			x= Scale* XVAL( set, pnt_nr)- rotX;
			y= Scale* YVAL( set, pnt_nr)- rotY;
			o= ERROR( set, pnt_nr);
			rx->value= rotX+ shiftX+ (x* ang_cos- y* ang_sin);
			ry->value= rotY+ shiftY+ (y* ang_cos+ x* ang_sin);
			if( ActiveWin && ActiveWin!= &StubWindow ){
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
static double mfg_spatial, mfg_orientation, mfg_radix, mfg_minX, mfg_maxX, mfg_minY, mfg_maxY;
static int mfg_idx1, mfg_idx2, mfg_raw;
static double fdsr_last_args[8];
static int mfg_shift, mfg_rotate, mfg_scale;

extern double *fd_bound_rotation_centre;

double figdistshiftrot( double args[7] )
{ static double called= 0;
	fdsr_last_args[0]= ++called;
	memcpy( &fdsr_last_args[1], args, 6*sizeof(double) );
	if( *fd_bound_rotation_centre ){
		if( args[3]< mfg_minX ){
			args[3]= mfg_minX;
		}
		else if( args[3]> mfg_maxX ){
			args[3]= mfg_maxX;
		}
		if( args[4]< mfg_minY ){
			args[4]= mfg_minY;
		}
		else if( args[4]> mfg_maxY ){
			args[4]= mfg_maxY;
		}
	}
	return( (fdsr_last_args[7]= figural_distance_shiftrot( mfg_wi, mfg_idx1, mfg_idx2,
		mfg_scale, args[6], mfg_shift, args[1], args[2], mfg_rotate, args[3], args[4], args[5], mfg_radix,
		&mfg_spatial, &mfg_orientation, mfg_raw
	)) );
}

double figdistshift( double args[4] )
{ static double called= 0;
	fdsr_last_args[0]= ++called;
	memcpy( &fdsr_last_args[1], args, 3* sizeof(double) );
	return( (fdsr_last_args[7]= figural_distance_shiftrot( mfg_wi, mfg_idx1, mfg_idx2,
		mfg_scale, args[3], True, args[1], args[2], False, 0, 0, 0, mfg_radix,
		&mfg_spatial, &mfg_orientation, mfg_raw
	)) );
}

double figdistrot( double args[5] )
{ static double called= 0;
	fdsr_last_args[0]= ++called;
	memcpy( &fdsr_last_args[1], args, 4* sizeof(double) );
	if( *fd_bound_rotation_centre ){
		if( args[1]< mfg_minX ){
			args[1]= mfg_minX;
		}
		else if( args[1]> mfg_maxX ){
			args[1]= mfg_maxX;
		}
		if( args[2]< mfg_minY ){
			args[2]= mfg_minY;
		}
		else if( args[2]> mfg_maxY ){
			args[2]= mfg_maxY;
		}
	}
	return( (fdsr_last_args[7]= figural_distance_shiftrot( mfg_wi, mfg_idx1, mfg_idx2,
		mfg_scale, args[4], False, 0, 0, True, args[1], args[2], args[3], mfg_radix,
		&mfg_spatial, &mfg_orientation, mfg_raw
	)) );
}

int ascanf_minimal_figdist( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  double ftol;
  ascanf_Function *Rscale, *RshiftX, *RshiftY, *RrotX, *RrotY, *Rangle, *Iter, *taf1= NULL, *taf2= NULL;
	if( ascanf_arguments>= 16 ){
		if( args[0]>= 0 && (args[0]< setNumber || ascanf_SyntaxCheck) ){
			mfg_idx1= (int) args[0];
		}
		else{
			ascanf_emsg= " (setnumber 1 out of range) ";
			ascanf_arg_error= True;
			mfg_idx1= -1;
		}
		if( args[1]>= 0 && (args[1]< setNumber || ascanf_SyntaxCheck) ){
			mfg_idx2= (int) args[1];
		}
		else{
			ascanf_emsg= " (setnumber 2 out of range) ";
			ascanf_arg_error= True;
			mfg_idx2= -1;
		}
		mfg_raw= (args[2])? True : False;
		mfg_scale= (args[3])? True : False;
		mfg_shift= (args[4])? True : False;
		mfg_rotate= (args[5])? True : False;
		mfg_radix= args[6];
		Iter= parse_ascanf_address( args[7], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		ftol= args[8];
		Rscale= parse_ascanf_address( args[9], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		RshiftX= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		RshiftY= parse_ascanf_address( args[11], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		RrotX= parse_ascanf_address( args[12], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		RrotY= parse_ascanf_address( args[13], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		Rangle= parse_ascanf_address( args[14], _ascanf_variable, "ascanf_minimal_figdist", (int) ascanf_verbose, NULL );
		if( ascanf_arguments> 15 ){
			taf1= parse_ascanf_address( args[15], _ascanf_variable, "ascanf_fig_dist_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 16 ){
			taf2= parse_ascanf_address( args[16], _ascanf_variable, "ascanf_fig_dist_shiftrot", (int) ascanf_verbose, NULL );
		}
		if( mfg_idx1>= 0 && mfg_idx2>= 0 && !ascanf_arg_error && Iter && !ascanf_SyntaxCheck ){
		  DataSet *set1= &AllSets[mfg_idx1];
		  DataSet *set2= &AllSets[mfg_idx2];
		  double x, y, fdP[8], bestP[7], bestfdP, shiftX, shiftY, rotX, rotY, angle, gravX1, gravY1, gravX2, gravY2;
		  double minX, maxX, minY, maxY, escale, Dscale;
		  int iter= Iter->value, i, ndim= 6;
		  ALLOCA( P, double*, ndim+2, Plen );
		  ALLOCA( P1, double, ndim+1, P1len );
		  ALLOCA( P2, double, ndim+1, P2len );
		  ALLOCA( P3, double, ndim+1, P3len );
		  ALLOCA( P4, double, ndim+1, P4len );
		  ALLOCA( P5, double, ndim+1, P5len );
		  ALLOCA( P6, double, ndim+1, P6len );
		  ALLOCA( P7, double, ndim+1, P7len );
		  double (*minimise)( double*)= figdistshiftrot;

			if( ActiveWin && ActiveWin!= &StubWindow ){
				mfg_wi= ActiveWin;
			}
			else{
				mfg_wi= NULL;
			}

			if( mfg_rotate ){
				if( !mfg_radix ){
					mfg_radix= M_2PI;
				}
			}
			else{
				mfg_radix= 0;
			}

			memset( bestP, 0, sizeof(bestP) );

			  /* Do nothing if the 2 sets are equal: */
			if( mfg_idx1== mfg_idx2 ){
				bestP[6]= 1;
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
					bestP[6]= 1;
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
				if( mfg_raw ){
					x+= XVAL( set1, i);
					y+= YVAL( set1, i);
				}
				else{
					x+= set1->xvec[i];
					y+= set1->yvec[i];
				}
				if( i ){
					minX= MIN( minX, x );
					maxX= MAX( maxX, x );
					minY= MIN( minY, y );
					maxY= MAX( maxY, y );
				}
				else{
					minX= maxX= x;
					minY= maxY= y;
				}
			}
			gravX1= x/ set1->numPoints;
			gravY1= y/ set1->numPoints;
			x= y= 0;
			for( i= 0; i< set2->numPoints; i++ ){
				if( mfg_raw ){
					x+= XVAL( set2, i);
					y+= YVAL( set2, i);
				}
				else{
					x+= set2->xvec[i];
					y+= set2->yvec[i];
				}
				if( i ){
					mfg_minX= MIN( mfg_minX, x );
					mfg_maxX= MAX( mfg_maxX, x );
					mfg_minY= MIN( mfg_minY, y );
					mfg_maxY= MAX( mfg_maxY, y );
				}
				else{
					mfg_minX= mfg_maxX= x;
					mfg_minY= mfg_maxY= y;
				}
				minX= MIN( minX, x );
				maxX= MAX( maxX, x );
				minY= MIN( minY, y );
				maxY= MAX( maxY, y );
			}
			gravX2= x/ set2->numPoints;
			gravY2= y/ set2->numPoints;

			  /* try to determine an initial guess for the scale factor. This will be such
			   \ that the two sets have equal curve length. If no window is active, we just
			   \ take 1 as a guess (but we could of course calculate the curve lengths here...)
			   */
			if( mfg_scale ){
				if( ActiveWin && ActiveWin!= &StubWindow ){
					if( ActiveWin->curve_len[mfg_idx2][set2->numPoints] ){
						escale= ActiveWin->curve_len[mfg_idx1][set1->numPoints]/
								ActiveWin->curve_len[mfg_idx2][set2->numPoints];
						Dscale= 2* escale;
					}
					else{
						mfg_scale= False;
					}
				}
				else{
					escale= 1;
					Dscale= 1;
				}
			}
			if( !mfg_scale ){
				ndim-= 1;
				escale= 1;
				Dscale= 1;
			}

			  /* The initial centre of rotation will be the to-be-rotated set's gravity point: */
			rotX= gravX2, rotY= gravY2, angle= 0;
			  /* The initial shift will be such that set2's gravity point will coincide with that of set1: */
			shiftX= gravX1- gravX2, shiftY= gravY1- gravY2;

			mfg_spatial= mfg_orientation= -1;

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
			if( mfg_shift ){
				  /* P1 of the simplex is the initially-guessed shiftrot'ed state: 	*/
				P[1][1]= shiftX, P[1][2]= shiftY;
				P[1][3]= rotX, P[1][4]= rotY, P[1][5]= 0, P[1][6]= escale;
				fdP[1]= (*minimise)( P[1] );
				  /* P2 of the simplex is the non-shiftrot'ed state: 	*/
				P[2][1]= P[2][2]= P[2][3]= P[2][4]= P[2][5]= 0; P[2][6]= 1;
				fdP[2]= (*minimise)( P[2] );

				P[3][1]= minX, P[3][2]= minY;
				P[3][3]= minX, P[3][4]= minY, P[3][5]= mfg_radix* 0.2, P[3][6]= 0.2* Dscale;
				fdP[3]= (*minimise)( P[3] );
				P[4][1]= minX, P[4][2]= minY;
				P[4][3]= maxX, P[4][4]= maxY, P[4][5]= mfg_radix* 0.4, P[4][6]= 0.4* Dscale;
				fdP[4]= (*minimise)( P[4] );
				P[5][1]= maxX, P[5][2]= maxY;
				P[5][3]= maxX, P[5][4]= maxY, P[5][5]= mfg_radix* 0.6, P[5][6]= 0.6* Dscale;
				fdP[5]= (*minimise)( P[5] );
				P[6][1]= maxX, P[6][2]= maxY;
				P[6][3]= minX, P[6][4]= minY, P[6][5]= mfg_radix* 0.8, P[6][6]= 0.8* Dscale;
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
				  /* P2 of the simplex is the non-shiftrot'ed state: 	*/
				P[2][1]= P[2][2]= P[2][3]= 0; P[2][4]= 1;
				fdP[2]= (*minimise)( P[2] );

				P[3][1]= minX, P[3][2]= minY, P[3][3]= mfg_radix* 0.2, P[3][4]= 0.2* Dscale;
				fdP[3]= (*minimise)( P[3] );
				P[4][1]= maxX, P[4][2]= maxY, P[4][3]= mfg_radix* 0.4, P[4][4]= 0.4* Dscale;
				fdP[4]= (*minimise)( P[4] );
				P[5][1]= maxX, P[5][2]= maxY, P[5][3]= mfg_radix* 0.6, P[5][4]= 0.6* Dscale;
				fdP[5]= (*minimise)( P[5] );
				P[6][1]= minX, P[6][2]= minY, P[6][3]= mfg_radix* 0.8, P[6][4]= 0.8* Dscale;
				fdP[6]= (*minimise)( P[6] );
				{ double xx= (maxX+minX)/2, yy= (maxY+minY)/2;
					P[7][1]= xx, P[7][2]= yy, P[7][3]= mfg_radix* 0.5, P[7][4]= 0.5* Dscale;
					fdP[7]= (*minimise)( P[7] );
				}
			}

			  /* Initialise the best function value ever with something large. This means a safe margin larger than
			   \ the figural distance at the "raw" configuration. The minimising function will in fact use this value
			   \ as a reference: if the initial value is lower than the real (global) minimum, no action occurs!
			   */
			bestfdP= 10* fdP[2];
			if( ascanf_verbose ){
				fprintf( StdErr,
					" (starting with best value 10*<unshifted>=%s; distance at gravity point (shift %g,%g,%g,%g,%g) %s) ",
					d2str( bestfdP,0,0),
					shiftX, shiftY, rotX, rotY, angle,
					d2str( fdP[2],0,0)
				);
			}

			continuous_simanneal( P, fdP, ndim, bestP, &bestfdP, ftol, minimise, &iter, 0 );
			if( ascanf_verbose ){
				fprintf( StdErr,
					"\n minimal figural distance (t=0): min@(%g,%g,%g,%g,%g,%g) == (%g,%g)=%g in %d iterations\n"
					" calls=%g, fdsr_last_args={%g,%g,%g,%g,%g,%g}=%g\n"
/* 					" simplex: (%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g)(%g,%g,%g,%g,%g)\n"	*/
					, bestP[6], bestP[1], bestP[2], bestP[3], bestP[4], bestP[5],
					mfg_spatial, mfg_orientation,
					bestfdP, iter,
					fdsr_last_args[0],
					fdsr_last_args[1], fdsr_last_args[2], fdsr_last_args[3], fdsr_last_args[4], fdsr_last_args[5],
						fdsr_last_args[6], fdsr_last_args[7]
/* 					, P[1][1], P[1][2], P[1][3], P[1][4], P[1][5],	*/
/* 					P[2][1], P[2][2], P[2][3], P[2][4], P[2][5],	*/
/* 					P[3][1], P[3][2], P[3][3], P[3][4], P[3][5],	*/
/* 					P[4][1], P[4][2], P[4][3], P[4][4], P[4][5],	*/
/* 					P[5][1], P[5][2], P[5][3], P[5][4], P[5][5],	*/
/* 					P[6][1], P[6][2], P[6][3], P[6][4], P[6][5]	*/
				);
			}

			if( !mfg_shift ){
				  /* We used a subset of the full dimensionality: now put the values where we expect them later on: */
				bestP[6]= bestP[4];
				bestP[5]= bestP[3];
				bestP[4]= bestP[2];
				bestP[3]= bestP[1];
				bestP[1]= bestP[2]= 0;
			}
			if( !mfg_rotate ){
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
				Rscale->value= (mfg_scale)? bestP[6] : 1;
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
	return( !ascanf_arg_error );
}

static ascanf_Function fig_dist_Function[] = {
	{ "$figural-distance-orientation", ascanf_Variable, 2, _ascanf_variable,
		"$figural-distance-orientation: whether or not to include the orientation (set's ecol) in the calculation\n"
		" of the figural distance.\n", 1, 0, 0, 0, 0, 0, 1.0
	},
	{ "$figural-distance-normalise-orientation", ascanf_Variable, 2, _ascanf_variable,
		"$figural-distance-normalise-orientation: normally, orientations (error columns...) are included\n"
		" in the figural-distance computation after a conversion from the current radix to radians\n"
		" (NB: the current radix is set via the Settings dialog, and not via the radix of the rotation\n"
		" to be applied to the adjustable set!). When this variable is set, orientations are normalised\n"
		" with respect to the current spatial figural distance, such that an average orientation difference\n"
		" of PI (180 degrees) is equalised to the current spatial distance. If that spatial distance is 0,\n"
		" the orientation difference is expressed in radians.\n"
		, 1, 0, 0, 0, 0, 0, 0.0
	},
	{ "$figural-distance-bound-rotation-centre", ascanf_Variable, 2, _ascanf_variable,
		"$figural-distance-bound-rotation-centre: determines if the (rotX,rotY) centre of rotation is bounded\n"
		" to be within the adjustable set's X and Y range during figural-distance minimisation.\n"
		" If not set, (rotX,rotY) can vary freely, which means the set can be shifted even when the <shift>\n"
		" argument to minimal-figural-distance[] is False. When bounded, such shifts can still occur, but the\n"
		" rotation centre can only vary \"within\" the set.\n"
		, 1, 0, 0, 0, 0, 0, 1.0
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
	{ "figural-distance-shiftrot", ascanf_fig_dist_shiftrot, 12, NOT_EOF_OR_RETURN,
		"figural-distance-shiftrot[set1, set2, raw, scale, shiftX, shiftY, rotX, rotY, angle, radix[,&spatial,&orientation]]\n"
		" This calculates the figural distance after a rotation over <angle> (base <radix>) around (rotX,rotY) and a\n"
		" subsequent shift over (shiftX,shiftY) of the 2nd set. The set is first scaled over <scale>.\n"
		" When set2 is in vector mode (error column contains orientations), the orientations are rotated over <angle>,\n"
		" otherwise, the error column is not modified. The other arguments are as described under figural-distance[].\n"
		" When either of the scale/shift/rotation parameters is a NaN, the corresponding operation(s) is/are not performed.\n"
	},
	{ "Set-shiftrot", ascanf_setshiftrot, 12, NOT_EOF,
		"Set-shiftrot[set,pnt_nr,scale,shiftX,shiftY,rotX,rotY,angle,radix, &rX, &rY, &rE]\n"
		" Return a set's co-ordinates (X,Y,E) at point <pnt_nr> after a rotation of <angle> (base <radix>)\n"
		" around (rotX,rotY), and a shift over (shiftX,shiftY). The transformed values are stored in\n"
		" &rX, &rY, &rE. If the set's error column represent orientations (vector mode), rE=E+angle;\n"
		" otherwise, rE=E.\n"
	},
	{ "minimal-figural-distance", ascanf_minimal_figdist, 17, NOT_EOF_OR_RETURN,
		"minimal-figural-distance[set1,set2,raw,scale,shift,rotate,radix,&iterations,fractol,&Rscale,&RshiftX,&RshiftY,&RrotX,&RrotY,&Rangle[,&spat,&orn]]\n"
		" Find the minimal figural distance by adjusting the shift and rotation parameters as described for\n"
		" figural-distance-shiftrot[]. Iterations contains the number of iterations to run, and returns the\n"
		" number actually performed; fractol is the fractional tolerance for the solution; the other pointers\n"
		" contain the final shift&rot parameters that result in the minimal distance.\n"
		" Shifting is only done when <shift>, rotation only when <rotate>; when both are False, no minimisation takes place...\n"
		" NB: when shift=False, rotX and rotY are still subject to change (cf. $figural-distance-bound-rotation-centre)!\n"
	},
};

static double *fd_orientation= &fig_dist_Function[0].value;
static double *fd_normalise_orientation= &fig_dist_Function[1].value;
static double *fd_bound_rotation_centre= &fig_dist_Function[2].value;

static int fig_dist_Functions= sizeof(fig_dist_Function)/sizeof(ascanf_Function);

static double tsa( double args[3] )
{ double x= args[1]- 10, y= args[2]- 2;
	return( 1+ ( x*x + y*y ) );
}

int test_simanneal( int sa )
{ double fdP[4], bestP[3], bestfdP, ftol= DBL_EPSILON;
  int i, iter= 1000;
  ALLOCA( P, double*, 4, Plen );
  ALLOCA( P1, double, 3, P1len );
  ALLOCA( P2, double, 3, P2len );
  ALLOCA( P3, double, 3, P3len );

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
	if( sa ){
		continuous_simanneal( P, fdP, 2, bestP, &bestfdP, ftol, tsa, &iter, 20 );
		fprintf( StdErr,
			"test simanneal t=20, 1+(x+1)**2+(y-2)**2; min@(%g,%g) == %g in %d iterations\n"
			" simplex: (%g,%g)(%g,%g)(%g,%g)\n",
			bestP[1], bestP[2], bestfdP, iter,
			P[1][1], P[1][2], P[2][1], P[2][2], P[3][1], P[3][2]
		);
		iter= 1000;
		continuous_simanneal( P, fdP, 2, bestP, &bestfdP, ftol, tsa, &iter, 2 );
		fprintf( StdErr,
			"test simanneal t=2, 1+(x-10)**2+(y-2)**2; min@(%g,%g) == %g in %d iterations\n"
			" simplex: (%g,%g)(%g,%g)(%g,%g)\n",
			bestP[1], bestP[2], bestfdP, iter,
			P[1][1], P[1][2], P[2][1], P[2][2], P[3][1], P[3][2]
		);
		iter= 1000;
		continuous_simanneal( P, fdP, 2, bestP, &bestfdP, ftol, tsa, &iter, 0.1 );
		fprintf( StdErr,
			"test simanneal t=0.1, 1+(x+1)**2+(y-2)**2; min@(%g,%g) == %g in %d iterations\n"
			" simplex: (%g,%g)(%g,%g)(%g,%g)\n",
			bestP[1], bestP[2], bestfdP, iter,
			P[1][1], P[1][2], P[2][1], P[2][2], P[3][1], P[3][2]
		);
	}
	else{
		iter= 1000;
		continuous_simanneal( P, fdP, 2, bestP, &bestfdP, ftol, tsa, &iter, 0 );
		fprintf( StdErr,
			"test simanneal t=0, 1+(x+1)**2+(y-2)**2; min@(%g,%g) == %g in %d iterations\n"
			" simplex: (%g,%g)(%g,%g)(%g,%g)\n",
			bestP[1], bestP[2], bestfdP, iter,
			P[1][1], P[1][2], P[2][1], P[2][2], P[3][1], P[3][2]
		);
	}
	return(iter);
}

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= fig_dist_Function;
  static char called= 0;
  int i;
  char buf[64];
  extern char *XGstrdup();

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
			if( af->function!= ascanf_Variable ){
				set_NaN(af->value);
			}
			if( label ){
				af->label= XGstrdup( label );
			}
		}
		af->dymod= new;
	}
	called+= 1;
}

static int initialised= False;

DyModTypes initDyMod( DyModLists *new )
{ static int called= 0;
	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, new->name, new->path, XG_IDENTIFY(), ++called );
	if( !initialised ){
		af_initialise( new, new->name );
		add_ascanf_functions( fig_dist_Function, fig_dist_Functions, "fig_dist::initDyMod()" );
		initialised= True;
	}
	new->libHook= NULL;
	new->libname= XGstrdup( "DM-figural-distance" );
	new->buildstring= XGstrdup(XG_IDENTIFY());
	new->description= XGstrdup(
		" A library with routines for determining the figural distance between 2 sets,\n"
		" and (in the end) the minimal figural distance by shifting and rotating the 2nd\n"
		" set with respect to the 1st.\n"
	);

	sran1( (long) time(NULL) );
	test_simanneal(True);
	test_simanneal(False);

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
		if( force || r> 0 ){
			for( i= 0; i< fig_dist_Functions; i++ ){
				fig_dist_Function[i].dymod= NULL;
			}
			initialised= False;
			xfree( target->libname );
			xfree( target->buildstring );
			xfree( target->description );
			ret= target->type= DM_Unloaded;
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
			fprintf( SE, " -- refused: variables are in use (remove_ascanf_functions() returns %d)\n", r );
		}
	}
	return(ret);
}

/* _init() and _fini() are called at first initialisation, and final unloading respectively. This works under linux, and
 \ maybe solaris - not under Irix 6.3. It also requires that the -nostdlib flag is passed to gcc.
 */
int _init()
{ static int called= 0;
	fprintf( StdErr, "%s::_init(): call #%d\n", __FILE__, ++called );
}

int _fini()
{ static int called= 0;
	fprintf( StdErr, "%s::_fini(): call #%d\n", __FILE__, ++called );
}

