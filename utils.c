#include "config.h"
IDENTIFY( "Utils ascanf library module" );

#ifndef XG_DYMOD_SUPPORT
#error You need to define XG_DYMOD_SUPPORT in config.h - and compile xgraph with it!!!
#endif


#include <stdio.h>
#include <stdlib.h>

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

#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
  /* If we want to be able to access the "expression" field in the callback argument, we need compiled_ascanf.h .
   \ If we don't include it, we will just get 'theExpr= NULL' ... (this applies to when -DDEBUG)
   */
#include "compiled_ascanf.h"
#include "ascanfc-table.h"

  /* For us to be able to access the calling programme's internal variables, the calling programme should have
   \ had at least 1 object file compiled with the -rdynamic flag (gcc 2.95.2, linux). Under irix 6.3, this
   \ is the default for the compiler and/or the OS (gcc).
   */

#include <float.h>

#include "dymod_interface.h"
static DyMod_Interface DMBaseMem, *DMBase= NULL;

	ascanf_Function* (*Create_Internal_ascanfString_ptr)( char *string, int *level );
	UserLabel* (*Add_UserLabel_ptr)( LocalWin *wi, char *labeltext, double x1, double y1, double x2, double y2,
		int point_label, DataSet *point_label_set, int point_l_nr, double point_l_x, double point_l_y,
		ULabelTypes type,
		int allow_name_trans, unsigned int mask_rtn_pressed, unsigned int mask_rtn_released,
		int noDialog
	);
	ULabelTypes (*Parse_ULabelType_ptr)( char ULtype[2] );
	char* (*ULabel_pixelCName_ptr)(UserLabel*, int* );
	char **ULabelTypeNames_ptr;
	double (*atan3_ptr)( double x, double y);
	void (*FreeColor_ptr)( Pixel*, char** );
	int (*GetColor_ptr)( char*, Pixel* );
	double (*SS_Skew_ptr)( SimpleStats * );
	char **ParsedColourName_ptr;
	AttrSet **AllAttrs_ptr;

#	define Create_Internal_ascanfString	(*Create_Internal_ascanfString_ptr)
#	define Add_UserLabel	(*Add_UserLabel_ptr)
#	define Parse_ULabelType	(*Parse_ULabelType_ptr)
#	define ULabelTypeNames	(ULabelTypeNames_ptr)
#	define atan3			(*atan3_ptr)
#	define ULabel_pixelCName	(*ULabel_pixelCName_ptr)
#	define FreeColor	(*FreeColor_ptr)
#	define GetColor	(*GetColor_ptr)
#	define SS_Skew	(*SS_Skew_ptr)
#	define ParsedColourName	*(ParsedColourName_ptr)
#	define AllAttrs	(*AllAttrs_ptr)

#include "LineCircle.c"

int ascanf_LineCircleClip ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *centre, *first, *second, *af_line0= NULL, *af_line1= NULL;
	if( ascanf_arguments< 6 ){
		ascanf_arg_error= 1;
	}
	else{
		if( (af_line0= parse_ascanf_address( args[0], _ascanf_array, "ascanf_LineCircleClip", (int) ascanf_verbose, NULL )) &&
			af_line0->N!= 2
		){
			ascanf_emsg= " (1st argument must be a scalar [slope] or a double array[2])== ";
			ascanf_arg_error= 1;
		}
		if( (af_line1= parse_ascanf_address( args[1], _ascanf_array, "ascanf_LineCircleClip", (int) ascanf_verbose, NULL )) &&
			af_line1->N!= 2
		){
			ascanf_emsg= " (2nd argument must be a scalar [intercept] or a double array[2])== ";
			ascanf_arg_error= 1;
		}
		if( (centre= parse_ascanf_address( args[2], _ascanf_array, "ascanf_LineCircleClip", (int) ascanf_verbose, NULL )) &&
			centre->N!= 2
		){
			ascanf_emsg= " (3rd argument must a double array[2])== ";
			ascanf_arg_error= 1;
		}
		if( (first= parse_ascanf_address( args[4], _ascanf_array, "ascanf_LineCircleClip", (int) ascanf_verbose, NULL )) &&
			first->N!= 2
		){
			ascanf_emsg= " (5th argument must a double array[2])== ";
			ascanf_arg_error= 1;
		}
		if( (second= parse_ascanf_address( args[5], _ascanf_array, "ascanf_LineCircleClip", (int) ascanf_verbose, NULL )) &&
			second->N!= 2
		){
			ascanf_emsg= " (6th argument must a double array[2])== ";
			ascanf_arg_error= 1;
		}
	}
	if( !ascanf_arg_error && !ascanf_SyntaxCheck ){
		if( af_line0 && af_line1 ){
			*result= clip_line_by_circle( af_line0->array, af_line1->array, centre->array, args[3], first->array, second->array );
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " ((%s,%s)-(%s,%s) in (x-%s)^2+(y-%s)^2==%s : (%s,%s)-(%s,%s))== ",
					ad2str( af_line0->array[0], d3str_format,0), ad2str( af_line0->array[1], d3str_format,0),
					ad2str( af_line1->array[0], d3str_format,0), ad2str( af_line1->array[1], d3str_format,0),
					ad2str( centre->array[0], d3str_format,0), ad2str( centre->array[1], d3str_format,0),
					ad2str( args[3], d3str_format, 0),
					ad2str( first->array[0], d3str_format,0), ad2str( first->array[1], d3str_format,0),
					ad2str( second->array[0], d3str_format,0), ad2str( second->array[1], d3str_format,0)
				);
			}
		}
		else{
			*result= clip_line_by_circle2( args[0], args[1], centre->array, args[3], first->array, second->array );
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (Y=%sX+%s in (X-%s)^2+(Y-%s)^2==%s : (%s,%s)-(%s,%s))== ",
					ad2str( args[0], d3str_format,0), ad2str( args[1], d3str_format,0),
					ad2str( centre->array[0], d3str_format,0), ad2str( centre->array[1], d3str_format,0),
					ad2str( args[3], d3str_format, 0),
					ad2str( first->array[0], d3str_format,0), ad2str( first->array[1], d3str_format,0),
					ad2str( second->array[0], d3str_format,0), ad2str( second->array[1], d3str_format,0)
				);
			}
		}
	}
	else{
		*result= 0;
	}
	return( !ascanf_arg_error );
}

int ascanf_FitCircle2Triangle ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *rx= NULL, *ry= NULL, *angle1= NULL, *angle2= NULL, *topangle= NULL;
  double x1, y1, x0, y0, x2, y2, x= 0, y= 0;
	set_NaN(*result);
	if( args && ascanf_arguments>= 6 ){
	  int swap;
	  double radix= M_2PI, sign;
		x1= args[0], y1= args[1];
		x0= args[2], y0= args[3];
		x2= args[4], y2= args[5];
		if( ascanf_arguments> 6 && ASCANF_TRUE(args[6]) ){
			rx= parse_ascanf_address( args[6], _ascanf_variable, "ascanf_FitCircle2Triangle", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 7 && ASCANF_TRUE(args[7]) ){
			ry= parse_ascanf_address( args[7], _ascanf_variable, "ascanf_FitCircle2Triangle", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 8 && ASCANF_TRUE(args[8]) ){
			angle1= parse_ascanf_address( args[8], _ascanf_variable, "ascanf_FitCircle2Triangle", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 9 && ASCANF_TRUE(args[9]) ){
			angle2= parse_ascanf_address( args[9], _ascanf_variable, "ascanf_FitCircle2Triangle", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 10 && ASCANF_TRUE(args[10]) ){
			topangle= parse_ascanf_address( args[10], _ascanf_variable, "ascanf_FitCircle2Triangle", (int) ascanf_verbose, NULL );
		}
		if( ascanf_arguments> 11 && ASCANF_TRUE(args[11]) ){
			radix= args[11];
		}

		  /* Make the triangle top (middle point) the origin: */
		x1-= x0, x2-= x0;
		y1-= y0, y2-= y0;
		if( angle1 ){
			angle1->value= radix * atan3( x1, y1 ) / M_2PI;
			if( angle1->accessHandler ){
				AccessHandler( angle1, "FitCircle2Triangle", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
		}
		if( angle2 ){
			angle2->value= radix * atan3( x2, y2 ) / M_2PI;
			if( angle2->accessHandler ){
				AccessHandler( angle2, "FitCircle2Triangle", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
		}
		if( topangle ){
			topangle->value= (x2*x1) + (y2*y1);
			if( topangle->accessHandler ){
				AccessHandler( topangle, "FitCircle2Triangle", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
		}
		  /* The direction of rotation is determined from the vectorproduct of (P1-P2) * (P3-P1);
		   \ if the sign of the Z component of the resulting vector is positive, than the rotation
		   \ is counterclockwise.
		   \ 20041216: this has to be done before the potential x/y swapping explained below
		   \ (or the swapping has to be taken into account otherwise: better do things here).
		   */
		if( x2*y1 - x1*y2 < 0 ){
		  /* clockwise rotation: */
			sign= -1;
		}
		else{
		  /* counterclockwise rotation: */
			sign= 1;
		}

		  /* This implementation solves the set of equations
		   \ 0 == x1(x-x1) + y1(y-y1) == x2(x-x2) + y2(y-y2)
		   \ where (x,y) is the intersection of the normals to 2 segments in (x1,y1) and (x2,y2) respectively.
		   \ The segment (0,0) - (x,y) gives the diameter of the circle passing through (x1,y1), (0,0) and (x2,y2).
		   \ This system is solved by first determining the expression for x, and substituting that into the expr.
		   \ for y (and thus calculates y and with that x). The case y2==0 (y2==y0 in the original co-ordinates)
		   \ is handled 'automatically', but x1==0 needs to be handled separately; the easiest way seems to
		   \ just swap the x and y co-ordinates.
		   */
		if( x1 == 0 ){
			SWAP(x1, y1, double );
			SWAP(x2, y2, double );
			swap= True;
		}
		else{
			swap= False;
		}
		  /* Do the actual calculations, but not if we have a tie: */
		if( !(x1== x2 && y1== y2) ){
			y= ( x2*x2 + y2*y2 - x2 * (x1*x1 +y1*y1)/x1 ) / ( y2 - x2*y1/x1 );
			x= ( y1 * (y1-y) ) / x1 + x1;
		}
		else{
			x= x1;
			y= y1;
		}
		  /* We're more interested in the centre and the radius, so divide (x,y) by 2
		   \ now that (x0,y0) == (0,0).
		   */
		x/= 2, y/= 2;

		  /* We return the radius: */
		*result= sign * sqrt( x*x + y*y );
		  /* We may need to swap the calculated (x,y): */
		if( swap ){
			SWAP(x, y, double );
		}
		  /* Finally, translate back to (x0,y0): */
		x+= x0, y+= y0;
		if( rx ){
			rx->value= x;
			if( rx->accessHandler ){
				AccessHandler( rx, "FitCircle2Triangle", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
		}
		if( ry ){
			ry->value= y;
			if( ry->accessHandler ){
				AccessHandler( ry, "FitCircle2Triangle", level, ASCB_COMPILED, AH_EXPR, NULL   );
			}
		}
	}
	else{
		ascanf_arg_error= 1;
	}
	return( !ascanf_arg_error );
}

int ascanf_Monotone ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *array, *idx= NULL;
  int really_monotone= False;
	set_NaN(*result);
	if( ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else{
	  int i= 1, ok= True;
	  int dirn;
		if( (array= parse_ascanf_address( args[0], _ascanf_array, "ascanf_Monotone", (int) ascanf_verbose, NULL )) ){
		  int N= array->N-1;
			if( ascanf_arguments> 1 ){
				really_monotone= ASCANF_TRUE( args[1] );
			}
			if( ascanf_arguments> 2 ){
				idx= parse_ascanf_address( args[2], _ascanf_variable, "ascanf_Monotone", (int) ascanf_verbose, NULL );
			}
			if( array->iarray ){
			  int prev= array->iarray[0], cur= array->iarray[1];
				if( cur> prev ){
					dirn= 1;
				}
				else if( cur< prev ){
					dirn= -1;
				}
				else{
					if( really_monotone ){
						ok= False;
					}
					dirn= 0;
				}
				if( ok ){
					if( really_monotone ){
						switch( dirn ){
							case 0:
								while( ok && i< N ){
									if( cur!= prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->iarray[++i];
									}
								}
								break;
							case 1:
								while( ok && i< N ){
									if( cur<= prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->iarray[++i];
									}
								}
								break;
							case -1:
								while( ok && i< N ){
									if( cur>= prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->iarray[++i];
									}
								}
								break;
						}
					}
					else{
						switch( dirn ){
							case 1:
								while( ok && i< N ){
									if( cur< prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->iarray[++i];
									}
								}
								break;
							case -1:
								while( ok && i< N ){
									if( cur> prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->iarray[++i];
									}
								}
								break;
						}
					}
				}
			}
			else{
			  double prev= array->array[0], cur= array->array[1];
				if( cur> prev ){
					dirn= 1;
				}
				else if( cur< prev ){
					dirn= -1;
				}
				else{
					if( really_monotone ){
						ok= False;
					}
					dirn= 0;
				}
				if( ok ){
					if( really_monotone ){
						switch( dirn ){
							case 0:
								while( ok && i< N ){
									if( cur!= prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->array[++i];
									}
								}
								break;
							case 1:
								while( ok && i< N ){
									if( cur<= prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->array[++i];
									}
								}
								break;
							case -1:
								while( ok && i< N ){
									if( cur>= prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->array[++i];
									}
								}
								break;
						}
					}
					else{
						switch( dirn ){
							case 1:
								while( ok && i< N ){
									if( cur< prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->array[++i];
									}
								}
								break;
							case -1:
								while( ok && i< N ){
									if( cur> prev ){
										ok= False;
									}
									else{
										prev= cur;
										cur= array->array[++i];
									}
								}
								break;
						}
					}
				}
			}
			if( ok ){
				*result= (dirn)? dirn : 1;
			}
			else{
				*result= 0;
				if( idx ){
					idx->value= i;
					if( idx->accessHandler ){
						AccessHandler( idx, "MonotoneArray", level, ASCB_COMPILED, AH_EXPR, NULL   );
					}
				}
			}
		}
		else{
			ascanf_emsg= " (invalid array argument) ";
			ascanf_arg_error= True;
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_RemoveTies ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *source, *uniq= NULL, *s2u= NULL, *u2s= NULL;
	set_NaN(*result);
	if( ascanf_arguments< 2 ){
		ascanf_arg_error= 1;
	}
	else{
	  int i= 1, j, N;
		if( !(source= parse_ascanf_address( args[0], _ascanf_array, "ascanf_RemoveTies", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid source argument) ";
			ascanf_arg_error= True;
		}
		if( !(uniq= parse_ascanf_address( args[1], _ascanf_array, "ascanf_RemoveTies", (int) ascanf_verbose, NULL )) 
			|| (source->iarray && !uniq->iarray)
			|| (source->array && !uniq->array)
		){
			ascanf_emsg= " (invalid uniq argument, not an array or not matching the source) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 2 && ASCANF_TRUE(args[2]) ){
			if( !(s2u= parse_ascanf_address( args[2], _ascanf_array, "ascanf_RemoveTies", (int) ascanf_verbose, NULL )) ){
				ascanf_emsg= " (invalid s2u argument ignored) ";
			}
		}
		if( ascanf_arguments> 3 && ASCANF_TRUE(args[3]) ){
			if( !(u2s= parse_ascanf_address( args[3], _ascanf_array, "ascanf_RemoveTies", (int) ascanf_verbose, NULL )) ){
				ascanf_emsg= " (invalid u2s argument ignored) ";
			}
		}
		if( !ascanf_arg_error ){
			N= (source->N)? 1 : 0;
			if( source->iarray ){
			  int p= source->iarray[0];
				for( i= 1; i< source->N; i++ ){
					if( source->iarray[i]!= p ){
						N+= 1;
						p= source->iarray[i];
					}
				}
			}
			else{
			  double p= source->array[0];
				for( i= 1; i< source->N; i++ ){
					if( source->array[i]!= p ){
						N+= 1;
						p= source->array[i];
					}
				}
			}
			Resize_ascanf_Array( uniq, N, NULL );
			if( s2u ){
				Resize_ascanf_Array( s2u, source->N, NULL );
			}
			if( u2s ){
				Resize_ascanf_Array( u2s, N, NULL );
			}
			if( source->iarray ){
			  int p= source->iarray[0];
				  /* 20070713: initialise i to 0 too! */
				i= j= 0;
				uniq->iarray[j]= p;
				if( s2u ){
					ASCANF_ARRAY_ELEM_SET(s2u, i, j);
				}
				if( u2s ){
					ASCANF_ARRAY_ELEM_SET(u2s, 0, 0);
				}
				for( i= 1; i< source->N && j< uniq->N-1; i++ ){
					if( source->iarray[i]!= p ){
						j+= 1;
						uniq->iarray[j]= (p= source->iarray[i]);
						if( u2s ){
							ASCANF_ARRAY_ELEM_SET(u2s, j, i);
						}
					}
					if( s2u ){
						ASCANF_ARRAY_ELEM_SET(s2u, i, j);
					}
				}
			}
			else{
			  double p= source->array[0];
				i= j= 0;
				uniq->array[j]= p;
				if( s2u ){
					ASCANF_ARRAY_ELEM_SET(s2u, i, j);
				}
				if( u2s ){
					ASCANF_ARRAY_ELEM_SET(u2s, 0, 0);
				}
				for( i= 1; i< source->N && j< uniq->N-1; i++ ){
					if( source->array[i]!= p ){
						j+= 1;
						uniq->array[j]= (p= source->array[i]);
						if( u2s ){
							ASCANF_ARRAY_ELEM_SET(u2s, j, i);
						}
					}
					if( s2u ){
						ASCANF_ARRAY_ELEM_SET(s2u, i, j);
					}
				}
			}
			if( uniq->accessHandler ){
				AccessHandler( uniq, "ascanf_RemoveTies", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			if( s2u && s2u->accessHandler ){
				AccessHandler( s2u, "ascanf_RemoveTies", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			if( u2s && u2s->accessHandler ){
				AccessHandler( u2s, "ascanf_RemoveTies", level, ASCB_COMPILED, AH_EXPR, NULL );
			}
			*result= N;
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_RemoveTrend ( ASCB_ARGLIST )
{ ASCB_FRAME
  ascanf_Function *source, *dest= NULL;
  int start, end= -1, ignore_NaN_av= True;
  double startTrend, endTrend;

	set_NaN(*result);
	if( ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
	}
	else{
	  int i;
		if( !(source= parse_ascanf_address( args[0], _ascanf_array, "ascanf_RemoveTrend", (int) ascanf_verbose, NULL )) ){
			ascanf_emsg= " (invalid source argument) ";
			ascanf_arg_error= True;
		}
		if( args[1]>= 0 && args[1]< source->N ){
			start= (int) args[1];
		}
		else{
			ascanf_emsg= " (invalid start index) ";
			ascanf_arg_error= True;
		}
		startTrend= args[2];
		if( ascanf_arguments> 3 ){
			if( args[3]>= -1 && args[3]< source->N ){
				end= (int) args[3];
			}
			else{
				ascanf_emsg= " (invalid end index) ";
				ascanf_arg_error= True;
			}
		}
		if( end== -1 ){
			end= source->N-1;
		}
		if( ascanf_arguments> 4 ){
			endTrend= args[4];
		}
		else{
			endTrend= startTrend;
		}
		if( ascanf_arguments> 5 && ASCANF_TRUE(args[5])
			&& !(dest= parse_ascanf_address( args[5], _ascanf_array, "ascanf_RemoveTrend", (int) ascanf_verbose, NULL ))
		){
			ascanf_emsg= " (invalid dest array, ignored) ";
		}
		if( dest== source ){
			dest= NULL;
		}
		if( ascanf_arguments> 6 && ASCANF_TRUE(args[6]) ){
			ignore_NaN_av= False;
		}
		if( !ascanf_arg_error ){
		  int N;
		  SimpleStats ss;
			if( dest ){
				Resize_ascanf_Array( dest, source->N, NULL );
				N= MIN(source->N, dest->N);
			}
			else{
				N= source->N;
				dest= source;
			}
			if( isNaN(startTrend) || isNaN(endTrend) ){
			  // calculate average over the selected range:
				SS_Init_(ss);
				if( start!= end ){
					for( i= start; i< N && i<= end; i++ ){
					  double val= ASCANF_ARRAY_ELEM(source, i);
						if( !NaNorInf(val) || ignore_NaN_av ){
							SS_Add_Data_( ss, 1, val, 1 );
						}
					}
				}
				else{
					start= 0; end= N-1;
					for( i= 0; i< N; i++ ){
					  double val= ASCANF_ARRAY_ELEM(source, i);
						if( !NaNorInf(val) || ignore_NaN_av ){
							SS_Add_Data_( ss, 1, val, 1 );
						}
					}
				}
				SS_Mean_(ss);
				if( !NaNorInf(ss.mean) ){
				  // RJVB 20090112: don't make the data unusable...
					if( ascanf_verbose ){
						fprintf( StdErr, " (replacing startTrend=%s and/or endTrend=%s with average %s (stdev %s, skew %s) over range %d-%d; slope=",
							ad2str(startTrend, d3str_format, NULL), ad2str(endTrend, d3str_format, NULL),
							ad2str(ss.mean, d3str_format, NULL),
							ad2str( SS_St_Dev_(ss), d3str_format, NULL),
							ad2str( SS_Skew_(ss), d3str_format, NULL),
							start, end
						);
					}
					if( isNaN(startTrend) ){
						startTrend= ss.mean;
					}
					if( isNaN(endTrend) ){
						endTrend= ss.mean;
					}
					if( ascanf_verbose ){
						if( start != end ){
							fprintf( StdErr, "%g)== ", (endTrend - startTrend) / (end - start) );
						}
						else{
							fprintf( StdErr, "0)== " );
						}
					}
				}
				else{
					if( ascanf_verbose ){
						fprintf( StdErr, " (average is %s over range %d-%d: ignored!)== ",
							ad2str(ss.mean, d3str_format, NULL), start, end
						);
					}
				}
			}

			if( !isNaN(startTrend) && !isNaN(endTrend) ){
				if( start!= end ){
				  double slope= (endTrend - startTrend) / (end - start);
					for( i= start; i< N && i<= end; i++ ){
					  double y= (i - start) * slope + startTrend;
						ASCANF_ARRAY_ELEM_SET(dest, i, (ASCANF_ARRAY_ELEM(source, i) - y) );
					}
				}
				else{
					for( i= 0; i< N; i++ ){
						ASCANF_ARRAY_ELEM_SET(dest, i, (ASCANF_ARRAY_ELEM(source, i) - startTrend) );
					}
				}
			}
			*result= i;
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_raise ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	*result= 0;
	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
	}
	else if( !ascanf_SyntaxCheck ){
		if( disp ){
			XFlush( disp );
		}
		raise( (int) args[0] );
	}
	return( !ascanf_arg_error );
}

static UserLabel *_GetULabelNr( int nr)
{  UserLabel *ret= NULL;
	if( nr>= 0 && nr< ActiveWin->ulabels ){
	  UserLabel *ul= ActiveWin->ulabel;
		while( ul && nr>= 0 ){
			nr-= 1;
			ret= ul;
			ul= ul->next;
		}
	}
	return( ret );
}

int ascanf_ULabelStuff( LocalWin *wi, ASCB_ARGLIST, int update )
{ ASCB_FRAME
  UserLabel *ul= NULL;
  ascanf_Function *idx_rtn= NULL, *labeltext= NULL, *labelcoords= NULL, *labellink= NULL, *labeltype= NULL;
  int take_usage, unlink_label= False, noDialog;
	*result= 0;
	if( update ){
		idx_rtn= parse_ascanf_address( args[0], _ascanf_variable, "ascanf_ULabelStuff", (int) ascanf_verbose, NULL );
	}
	if( idx_rtn || (update && args[0]== -1) ){
	  DataSet *point_label_set= NULL;
	  ULabelTypes type= UL_regular;
		ascanf_arg_error= 0;
		if( ascanf_arguments>= 2 ){
			if( !(labeltext= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_getULabel",
					(int) ascanf_verbose, &take_usage )) || !labeltext->usage
			){
				ascanf_emsg= " (2nd argument must point to a valid non-null string) ";
				ascanf_arg_error= 1;
			}
			if( !(labelcoords=
				parse_ascanf_address( args[2], _ascanf_array, "ascanf_getULabel", (int) ascanf_verbose, NULL ) )
				|| labelcoords->N< 4
			){
				ascanf_emsg= " (3rd argument must point to an array with at least 4 elements) ";
				ascanf_arg_error= 1;
			}
			if( ascanf_arguments> 3 ){
				if( args[3]< -1 || args[3]> setNumber ){
					ascanf_emsg= " (4th argument must be a valid setNumber) ";
					ascanf_arg_error= 1;
				}
				else if( args[3]>= 0 ){
					point_label_set= &AllSets[ (int) args[3] ];
				}
				else{
					unlink_label= True;
				}
			}
			if( ascanf_arguments> 4 && args[4] ){
				if( (labeltype=
					parse_ascanf_address( args[4], _ascanf_variable, "ascanf_getULabel", (int) ascanf_verbose, NULL ) )
					&& (!update || labeltype->usage)
				){
					type= Parse_ULabelType( labeltype->usage );
				}
				else{
					ascanf_emsg= " (5th argument if present must point to a string variable) ";
					ascanf_arg_error= 1;
				}
			}
		}
		else{
			ascanf_arg_error= 1;
		}
		if( !ascanf_arg_error ){
		  int i;
			noDialog= (ascanf_arguments> 6)? ASCANF_TRUE(args[6]) : False;
			ul= Add_UserLabel( wi, labeltext->usage,
				ASCANF_ARRAY_ELEM(labelcoords, 0), ASCANF_ARRAY_ELEM(labelcoords, 1),
				ASCANF_ARRAY_ELEM(labelcoords, 2), ASCANF_ARRAY_ELEM(labelcoords, 3),
				0, point_label_set, -1, 0, 0, type, (wi->raw_display)? True : False,
				0, 0, noDialog
			);
			if( ul ){
				if( unlink_label ){
					ul->set_link= -1;
				}
				wi->redraw= True;
				labeltext= Create_Internal_ascanfString( (ul->labelbuf)? ul->labelbuf : ul->label, level );
				*result= take_ascanf_address(labeltext);
				i= wi->ulabels;
				if( idx_rtn ){
					while( i>= 0 && ul ){
						if( _GetULabelNr(i)== ul ){
							ul= NULL;
							args[0]= i;
							idx_rtn->value= i;
						}
						else{
							i-= 1;
						}
					}
				}
			}
		}
	}
	else if( args[0]>= 0 && args[0]< wi->ulabels ){
		ul= _GetULabelNr( (int) args[0] );
		if( !ul ){
			fprintf( stderr, " (ascanf_GetULabel(%d): unexpectedly couldn't get that label (%s)) ",
				(int) args[0], serror()
			);
			ascanf_arg_error= 1;
			*result= 0;
			return(0);
		}
		if( ascanf_arguments> 1 ){
			labeltext=
				parse_ascanf_address( args[1], _ascanf_variable, "ascanf_getULabel",
					(int) ascanf_verbose, &take_usage );
		}
		if( labeltext ){
			if( update ){
				if( labeltext->usage ){
					strncpy( ul->label, labeltext->usage, sizeof(ul->label)-1 );
				}
				else{
					ul->label[0]= '\0';
				}
				wi->redraw= True;
			}
			else{
				xfree(labeltext->usage);
				labeltext->usage= XGstrdup( (ul->labelbuf)? ul->labelbuf : ul->label );
				if( labeltext->accessHandler ){
					AccessHandler( labeltext, "ascanf_ULabelStuff", level, ASCB_COMPILED, AH_EXPR, NULL );
				}
			}
			*result= args[1];
		}
		else{
			if( update && ASCANF_TRUE(args[1]) ){
				ul->label[0]= '\0';
				wi->redraw= True;
				*result= args[0];
			}
			else{
				labeltext= Create_Internal_ascanfString( (ul->labelbuf)? ul->labelbuf : ul->label, level );
				*result= take_ascanf_address(labeltext);
			}
		}
		if( ascanf_arguments> 2 && args[2] ){
			if( (labelcoords=
				parse_ascanf_address( args[2], _ascanf_array, "ascanf_getULabel", (int) ascanf_verbose, NULL ) )
			){
				if( update ){
					if( labelcoords->N>= 4 ){
						ul->x1= ASCANF_ARRAY_ELEM(labelcoords, 0);
						ul->y1= ASCANF_ARRAY_ELEM(labelcoords, 1);
						ul->x2= ASCANF_ARRAY_ELEM(labelcoords, 2);
						ul->y2= ASCANF_ARRAY_ELEM(labelcoords, 3);
						wi->redraw= True;
					}
					else{
						ascanf_emsg= " (3rd argument must be at least 4 elements long) ";
						ascanf_arg_error= 1;
					}
				}
				else{
					Resize_ascanf_Array( labelcoords, 4, NULL );
					ASCANF_ARRAY_ELEM_SET( labelcoords, 0, ul->x1 );
					ASCANF_ARRAY_ELEM_SET( labelcoords, 1, ul->y1 );
					ASCANF_ARRAY_ELEM_SET( labelcoords, 2, ul->x2 );
					ASCANF_ARRAY_ELEM_SET( labelcoords, 3, ul->y2 );
					if( labelcoords->accessHandler ){
						AccessHandler( labelcoords, "ascanf_ULabelStuff", level, ASCB_COMPILED, AH_EXPR, NULL );
					}
				}
			}
			else{
				ascanf_emsg= " (3rd argument must point to an array, if present) ";
				ascanf_arg_error= 1;
			}
		}
		if( ascanf_arguments> 3 ){
			if( update ){
				if( args[3]== -1 ){
					if( ul->set_link>= 0 ){
						wi->redraw= True;
						ul->set_link= -1;
					}
				}
				else if( args[3]>= 0 && args[3]< setNumber ){
					ul->set_link= args[3];
					wi->redraw= True;
				}
				else{
					ascanf_emsg= " (4th argument must be a valid setnumber) ";
				}
			}
			else if( args[3] ){
				if( (labellink=
					parse_ascanf_address( args[3], _ascanf_variable, "ascanf_getULabel", (int) ascanf_verbose, NULL ) )
				){
					labellink->value= ul->set_link;
					if( labellink->accessHandler ){
						AccessHandler( labellink, "ascanf_ULabelStuff", level, ASCB_COMPILED, AH_EXPR, NULL );
					}
				}
				else{
					ascanf_emsg= " (4th argument must point to a scalar, if present) ";
					ascanf_arg_error= 1;
				}
			}
		}
		if( ascanf_arguments> 4 ){
			if( args[4] ){
				if( (labeltype=
					parse_ascanf_address( args[4], _ascanf_variable, "ascanf_getULabel", (int) ascanf_verbose, NULL ) )
					&& (!update || labeltype->usage)
				){
					if( update ){
						ul->type= Parse_ULabelType( labeltype->usage );
						wi->redraw= True;
					}
					else{
						xfree(labeltype->usage);
						if( ul->type>= UL_regular && ul->type< UL_types ){
							labeltype->usage= XGstrdup( ULabelTypeNames[ul->type] );
						}
						else{
							ul->type= UL_regular;
							labeltype->usage= XGstrdup("RL");
						}
						if( labeltype->accessHandler ){
							AccessHandler( labeltype, "ascanf_ULabelStuff", level, ASCB_COMPILED, AH_EXPR, NULL );
						}
					}
				}
				else{
					ascanf_emsg= " (5th argument if present must point to a string variable) ";
					ascanf_arg_error= 1;
				}
			}
		}
	}
	// 20080912: cases below used to be included in the else{} just finished above making it impossible to specify
	// a colour for a new auto-numbered label
	if( (update && ascanf_arguments> 5) || ascanf_arguments> 6 ){
	  ascanf_Function *clinked= NULL, *cspec= NULL;
		if( update ){
			if( (cspec=
				parse_ascanf_address( args[5], _ascanf_variable, "ascanf_setULabel", (int) ascanf_verbose, NULL ) )
			){
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)
			  Pixel tp;
				if( strcasecmp( cspec->usage, "default")== 0 ||
					(strncasecmp( cspec->usage, "default", 7)== 0 && isspace(cspec->usage[7]))
				){
					xfree( ul->pixelCName );
					ul->pixvalue= 0;
					ul->pixlinked= 0;
					ul->pixelValue= AllAttrs[ul->pixvalue].pixelValue;
				}
				else if( strcasecmp( cspec->usage, "linked")== 0 ||
					(strncasecmp( cspec->usage, "linked", 6)== 0 && isspace(cspec->usage[6]))
				){
					if( ul->set_link>= 0 && ul->set_link< setNumber ){
						xfree( ul->pixelCName );
						ul->pixvalue= 0;
						ul->pixlinked= 1;
					}
					else{
						ascanf_emsg= "Label isn't linked to a valid set";
						ascanf_arg_error= 1;
					}
				}
				else if( GetColor( cspec->usage, &tp) ){
					if( ul->pixvalue< 0 ){
						FreeColor( &ul->pixelValue, &ul->pixelCName );
					}
					StoreCName( ul->pixelCName );
					ul->pixelValue= tp;
					ul->pixvalue= -1;
					ul->pixlinked= 0;
				}
			}
		}
		else{
			if( (!ASCANF_TRUE(args[5]) || (clinked=
				parse_ascanf_address( args[5], _ascanf_variable, "ascanf_getULabel", (int) ascanf_verbose, NULL ) ) )
				&& (cspec=
					parse_ascanf_address( args[6], _ascanf_variable, "ascanf_getULabel", (int) ascanf_verbose, NULL ) )
			){
			  int type;
				xfree( cspec->usage );
				cspec->usage= XGstrdup( ULabel_pixelCName( ul, &type ) );
				if( cspec->accessHandler ){
					AccessHandler( cspec, "ascanf_ULabelStuff", level, ASCB_COMPILED, AH_EXPR, NULL );
				}
				if( clinked ){
					clinked->value= (type==-1)? True : False;
					if( clinked->accessHandler ){
						AccessHandler( clinked, "ascanf_ULabelStuff", level, ASCB_COMPILED, AH_EXPR, NULL );
					}
				}
			}
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_GetULabel( ASCB_ARGLIST )
{ ASCB_FRAME
	if( !ActiveWin && !StubWindow_ptr /* || ActiveWin == StubWindow_ptr */ ){
		*result= -1;
	}
	else{
	  LocalWin *AW= ActiveWin;
		if( !ActiveWin ){
			ActiveWin= StubWindow_ptr;
		}
		if( !args || !ascanf_arguments ){
			*result= ActiveWin->ulabels;
		}
		else if( ascanf_arguments>= 1 ){
			ascanf_ULabelStuff( ActiveWin, ASCB_ARGUMENTS, False );
		}
		else{
			ascanf_arg_error= 1;
		}
		ActiveWin= AW;
	}
	return( !ascanf_arg_error );
}

int ascanf_SetULabel( ASCB_ARGLIST )
{ ASCB_FRAME
	if( !ActiveWin && !StubWindow_ptr /* || ActiveWin == StubWindow_ptr */ ){
		*result= -1;
	}
	else{
	  LocalWin *AW= ActiveWin;
		if( !ActiveWin ){
			ActiveWin= StubWindow_ptr;
		}
		if( !args || !ascanf_arguments ){
			*result= ActiveWin->ulabels;
		}
		else if( ascanf_arguments> 1 ){
			ascanf_ULabelStuff( ActiveWin, ASCB_ARGUMENTS, True );
		}
		else{
			ascanf_arg_error= 1;
		}
		ActiveWin= AW;
	}
	return( !ascanf_arg_error );
}

int ascanf_SubsetArray ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *source, *target= NULL;
  int start, end;
  static ascanf_Function AF= {NULL};
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 2 ){
		ascanf_emsg= " (need at least 2 arguments)";
		ascanf_arg_error= 1;
	}
	else{
		if( !(source= parse_ascanf_address( args[0], _ascanf_array, "ascanf_SubsetArray", (int) ascanf_verbose, NULL ))
			|| source->sourceArray
		){
			ascanf_emsg= " (1st argument must be a non-subsetted array)== ";
			ascanf_arg_error= 1;
		}
		if( !ascanf_arg_error ){
			if( args[1]>= 0 && args[1]< source->N ){
				start= (int) args[1];
			}
			else{
				ascanf_emsg= " (2nd argument not a valid subscript for specified array)== ";
				ascanf_arg_error= 1;
			}
			if( ascanf_arguments> 2 ){
				if( args[2]>= 0 ){
					end= MIN( (int) args[2], source->N-1);
				}
				else{
					ascanf_emsg= " (3rd argument must be positive!)== ";
					ascanf_arg_error= 1;
				}
			}
			else{
				end= source->N-1;
			}
		}
		if( ascanf_arguments> 3 && args[3] ){
			if( !(target= parse_ascanf_address( args[3], _ascanf_array, "ascanf_SubsetArray", (int) ascanf_verbose, NULL )) ||
				  /* target variables must be internal, to prevent dumping them when the source has been deallocated! */
				!target->internal
			){
				fprintf( StdErr, " (warning: ignoring non-internal/non-array 4th argument \"%s\")== ", ad2str(args[3], d3str_format, NULL) );
				target= NULL;
			}
		}
	}
	*result= 0;
	if( source && !ascanf_arg_error ){
	  char subset[256];
		if( !target ){
			target= &AF;
			if( AF.name ){
			  double oa= target->own_address;
				xfree( AF.name );
				xfree(target->usage);
				memset( target, 0, sizeof(ascanf_Function) );
				target->own_address= oa;
			}
			else{
				memset( target, 0, sizeof(ascanf_Function) );
			}
			target->type= _ascanf_array;
			target->is_address= target->take_address= True;
			target->is_usage= target->take_usage= False;
			target->internal= True;
			snprintf( subset, sizeof(subset)/sizeof(char), "[%d..%d]", start, end );
			target->name= concat( source->name, subset, NULL );
			if( !target->own_address ){
				take_ascanf_address(target);
			}
		}
		else if( !target->sourceArray ){
			xfree(target->array);
			xfree(target->iarray);
			snprintf( subset, sizeof(subset)/sizeof(char), "[%d..%d]", start, end );
			xfree(target->usage);
			target->usage= concat( source->name, subset, NULL );
		}
		if( source->iarray ){
			target->iarray= &source->iarray[start];
			target->array= NULL;
		}
		else{
			target->iarray= NULL;
			target->array= &source->array[start];
		}
		target->N= end- start + 1;
		target->last_index= 0;
		target->sourceArray= source;
		*result= target->own_address;
	}
	return(!ascanf_arg_error);
}

typedef struct HistogramEntry{
	double value;
	unsigned long N;
} HistogramEntry;

static int qs_ints( int *a, int *b )
{
	return( *a - *b );
}

static int qs_doubles( double *a, double *b )
{
	if( *a < *b ){
		return(-1);
	}
	else if( *a > *b ){
		return(1);
	}
	else{
		return(0);
	}
}

int ascanf_fmadd ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	ascanf_arg_error= 0;
	if( !args || ascanf_arguments< 3 ){
		ascanf_emsg= " (need at least 3 arguments)";
		ascanf_arg_error= 1;
		set_NaN(*result);
	}
	else{
		*result= (args[0] * args[1]) + args[2];
	}
	return( !ascanf_arg_error );
}

static ascanf_Function utils_Function[] = {
	{ "Clip-Line-by-Circle", ascanf_LineCircleClip, 6, NOT_EOF,
		"Clip-Line-by-Circle[slope,intercept,&centre,radius,&first_ret,&secnd_ret] or\n"
		" Clip-Line-by-Circle[&line0,&line1,&centre,radius,&first_ret,&secnd_ret]: clip a linesegment by a circle\n"
		" and return the intersections. <line0>, <line1> (2 points on the line), <centre>, <first_ret> and\n"
		" <secnd_ret> are all pointers to 2-element arrays. If the specified line is not completely outside\n"
		" the circle, the functions returns the number of intersecting points and the intersections (x,y) in\n"
		" first_ret and secnd_ret. When called with slope,intercept, a line0 and line1 point pair is calculated\n"
		" at centre[0]-radius*2 and centre[0]+radius*2. If slope==Inf, a line X=intercept is assumed, with\n"
		" Y=centre[1] +- radius*2.\n"
	},
	{ "FitCircle2Triangle", ascanf_FitCircle2Triangle, 12, NOT_EOF,
		"FitCircle2Triangle[x1,y1, x2,y2, x3,y3[, &rx[, &ry[, &angle1[, &angle2[, &topangle[,radix]]]]]] ]: \n"
		" returns the radius and optionally the centre of the\n"
		" circle that passes through the three given points that define a triangle. (x2,y2) is taken to be the\n"
		" \"top\" of the triangle. When the &topangle argument is given, <topangle> will return 0 for an orthogonal top, \n"
		" <0 for an obtuse topangle and >0 for a sharp topangle. <angle1> and <angle2> can be used to obtain the\n"
		" \"world\" angles of the first and second segments forming the triangle.\n"
	},
	{ "MonotoneArray", ascanf_Monotone, 3, NOT_EOF_OR_RETURN,
		"MonotoneArray[&array[,true-monotone[,&idx]]]: determines whether the given array is increasing (returns 1) or\n"
		" or decreasing (returns -1). If <true-monotone> is False (the default), ties are allowed, otherwise\n"
		" only purely increasing and purely decreasing arrays will result in non-zero return values.\n"
		" If <idx> points to a variable, it will contain the index of the first 'refused' element, if any.\n"
	},
	{ "RemoveTies", ascanf_RemoveTies, 4, NOT_EOF_OR_RETURN,
		"RemoveTies[&source,&uniq[,&s2u]]: takes a source array, and returns it without ties in the <uniq> array.\n"
		" If <s2u> is a pointer to an array, this array will be set to source's size and contain for each of source's\n"
		" elements the index of that value in the unique array (source to uniq mapping).\n"
		" If <u2s> is a pointer to a valid array, this array will contain the uniq-to-source mapping, that is,\n"
		" it will contain the indices of the unique values in <source>\n"
		" Returns the number of unique values.\n"
		" The source and uniq arrays must be of the same type (double or integer)\n"
	},
	{ "RemoveTrend", ascanf_RemoveTrend, 7, NOT_EOF_OR_RETURN,
		"RemoveTrend[&source,start,startTrend[,end,endTrend[,&dest[,excludeNaNs]]]: takes a source array, and removes a trend from it.\n"
		" The trend is a straight line from (start,startTrend) to (end,endTrend) that is subtracted from <source> over the\n"
		" interval (start,end). When start==end and/or startTrend==endTrend, a continuous value is subtracted, over the\n"
		" full length if start==end. Specify end==-1 as an alternative to end==length(source)-1. When <dest> is missing\n"
		" or not an array, the operation is done 'in-place'.\n"
		" If either startTrend or endTrend is NaN, it is replaced be average of <source> over [start,end].\n"
		" If excludeNaNs is False (default), no action is taken when the average over the indicated range is Inf or NaN;\n"
		" otherwise, the average is calculated excluding any Inf or NaN samples inside the range.\n"
	},
	{ "GetULabel", ascanf_GetULabel, 7, NOT_EOF_OR_RETURN,
		"GetULabel[i[,&labeltext[,&coords[,&linked2[,&type[,&clinked?,&cspec]]]]]]: retrieves the active window's UserLabel number <i>.\n"
		" When <labeltext> points to a scalar, it will receive the label's text, and GetULabel will return a\n"
		" point to that variable (otherwise, a newly created internal-dict variable will be returned).\n"
		" The optional <coords> array returns the label's two sets of co-ordinates (the labelled point and\n"
		" the co-ordinates of the text itself); the set linked to or the label's type can be returned in <linked2> and <type>.\n"
		" For the type specification (a 2 element string), see the *ULABEL* description (or the UserLabel creation dialog).\n"
		" Upon error, 0 is returned.\n"
		" Called without arguments, this function returns the number of labels (or -1 upon error)\n"
	},
	  /* Add optional &clinked?,&cspec pair to GetULabel, &cspec argument to SetULabel. */
	{ "SetULabel", ascanf_SetULabel, 7, NOT_EOF_OR_RETURN,
		"SetULabel[i[,&labeltext[,&coords[,&linked2[,&type[,&cspec[,noDialog?]]]]]]]: modifies or creates a UserLabel in the active window.\n"
		" When <i> identifies an existing label, it is modified; when <i> is -1 or points to a scalar, a new\n"
		" label is created (in the latter case, the label number is returned in i). Otherwise the calling convention\n"
		" is identical to GetULabel.\n"
		" NB: new label creation is handled through the same dialog that also handles interactive label creation,\n"
		" initialised with the specified values.\n"
	},
	{ "raise-signal", ascanf_raise, 1, NOT_EOF,
		"raise-signal[sig]: raise a (numeric!) signal. Canonical/symbolic names ought to be defined in constants.so!\n"
		" NB: given its dangerous nature, this function is not auto-loading.\n"
	},
	{ "SubsetArray", ascanf_SubsetArray, 4, NOT_EOF_OR_RETURN,
		"SubsetArray[&source, start[,end[,&target]]]: return an array which is a 'linked subset' of the source array.\n"
		" This allows access to and modification of a sub-range of the source array, without copying the values twice.\n"
		" <start> and the optional <end> denote the subsetting bounds (upper bound defaults to source->N-1). If no target\n"
		" is specified, a pointer is returned to an internal variable, which will be named to reflect the source name. If\n"
		" target is specified, it must be a pointer to an existing *internal* array. Its own array memory is destroyed, and\n"
		" replaced with a pointer to the source array. In all cases, care must be taken not to attempt to use the target array\n"
		" after the source has been deleted (or reduced such that the subset has become invalid): there is no checking on this\n"
		" whatsoever, and thus crashes are not impossible.\n"
	},
	{ "fmadd", ascanf_fmadd, 3, NOT_EOF,
		"fmadd[x,y,z]: calculate (x*y) + z.\n"
	},
};
static int utils_Functions= sizeof(utils_Function)/sizeof(ascanf_Function);

static void af_initialise( DyModLists *new, char *label )
{ ascanf_Function *af= utils_Function;
  static char called= 0;
  int i;
  char buf[64];

	for( i= 0; i< utils_Functions; i++, af++ ){
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
			Check_Doubles_Ascanf( af, label, True );
		}
		af->dymod= new;
	}
	called+= 1;
}

static int initialised= False;

DyModTypes initDyMod( INIT_DYMOD_ARGUMENTS )
{ static int called= 0;

	if( !DMBase ){
		DMBaseMem.sizeof_DyMod_Interface= sizeof(DyMod_Interface);
		DMBaseMem.sizeof_ascanf_Function= sizeof(ascanf_Function);
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
		XGRAPH_FUNCTION(Create_Internal_ascanfString_ptr, "Create_Internal_ascanfString");
		XGRAPH_FUNCTION(Add_UserLabel_ptr, "Add_UserLabel");
		XGRAPH_FUNCTION(Parse_ULabelType_ptr, "Parse_ULabelType");
		XGRAPH_VARIABLE( ULabelTypeNames_ptr, "ULabelTypeNames" );
		XGRAPH_FUNCTION(atan3_ptr, "atan3");
		XGRAPH_FUNCTION(ULabel_pixelCName_ptr, "ULabel_pixelCName" );
		XGRAPH_FUNCTION(FreeColor_ptr, "FreeColor" );
		XGRAPH_FUNCTION(GetColor_ptr, "GetColor" );
		XGRAPH_FUNCTION(SS_Skew_ptr, "SS_Skew" );
		XGRAPH_VARIABLE(ParsedColourName_ptr, "ParsedColourName" );
		XGRAPH_VARIABLE(AllAttrs_ptr, "AllAttrs" );
	}

	fprintf( StdErr, "%s::initDyMod(): Initialising %s loaded from %s (build: %s), call %d\n", __FILE__, theDyMod->name, theDyMod->path, XG_IDENTIFY(), ++called );
	if( !initialised ){

		af_initialise( theDyMod, theDyMod->name );
		add_ascanf_functions( utils_Function, utils_Functions, "utils::initDyMod()" );
		initialised= True;
	}
	theDyMod->libHook= NULL;
	theDyMod->libname= XGstrdup( "DM-utils" );
	theDyMod->buildstring= XGstrdup(XG_IDENTIFY());
	theDyMod->description= XGstrdup(
		" A dynamic module (library) that provides\n"
		" various useful functions.\n"
	);

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
	if( initialised ){
	  int r;
		fprintf( StdErr, "%s::closeDyMod(%d): Closing %s loaded from %s, call %d", __FILE__,
			force, target->name, target->path, ++called
		);
		if( target->loaded4 ){
			fprintf( StdErr, "; auto-loaded because of \"%s\"", target->loaded4 );
		}
		r= remove_ascanf_functions( utils_Function, utils_Functions, force );
		if( force || r== utils_Functions ){
			for( i= 0; i< utils_Functions; i++ ){
				utils_Function[i].dymod= NULL;
			}
			initialised= False;
			xfree( target->libname );
			xfree( target->buildstring );
			xfree( target->description );
			ret= target->type= DM_Unloaded;
			if( r<= 0 || ascanf_emsg ){
				fprintf( StdErr, " -- warning: variables are in use (remove_ascanf_functions() returns %d,\"%s\")",
					r, (ascanf_emsg)? ascanf_emsg : "??"
				);
				Unloaded_Used_Modules+= 1;
				if( force ){
					ret= target->type= DM_FUnloaded;
				}
			}
			fputc( '\n', StdErr );
		}
		else{
			fprintf( StdErr, " -- refused: variables are in use (remove_ascanf_functions() returns %d out of %d)\n",
				r, utils_Functions
			);
		}
	}
	return(ret);
}
