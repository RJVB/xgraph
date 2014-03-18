#ifndef XG_DYMOD_SUPPORT
#	define XG_DYMOD_SUPPORT
#endif

#include "config.h"
IDENTIFY( "ascanf interface to XGraph" );

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <math.h>

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

#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
#	define USE_SSE2
#	include <xmmintrin.h>
#	include <emmintrin.h>
#	include "AppleVecLib.h"
#endif

#include "fdecl.h"

  /* get the ascanf definitions:	*/
#include "ascanf.h"
#include "compiled_ascanf.h"

  /* Definitions, externals, etc:	*/

#include "XXseg.h"
extern XSegment *Xsegs, *lowYsegs, *highYsegs;		/* Point space for X */
extern XXSegment *XXsegs;
#define LYsegs	lowYsegs
#define HYsegs	highYsegs
extern XSegment_error *XsegsE;
extern long XsegsSize, XXsegsSize, YsegsSize, XsegsESize;

extern int use_ps_LStyle, ps_LStyle;

#define __DLINE__	(double)__LINE__

extern int maxitems;

extern char *xgraph_NameBuf;
extern int xgraph_NameBufLen;

extern LocalWin *ActiveWin, StubWindow;

extern int XGStoreColours;

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun/sse_mathfun.h"
#	include "arrayvops.h"
#elif defined(__SSE2__) || defined(__SSE3__)
#	include "arrayvops.h"
#endif

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
#define R_UsrOrgX	win_geo.R_UsrOrgX
#define R_UsrOrgY	win_geo.R_UsrOrgY
#define R_UsrOppX	win_geo.R_UsrOppX
#define R_UsrOppY	win_geo.R_UsrOppY
#define XUnitsPerPixel	win_geo._XUnitsPerPixel
#define YUnitsPerPixel	win_geo._YUnitsPerPixel
#define R_XUnitsPerPixel	win_geo.R_XUnitsPerPixel
#define R_YUnitsPerPixel	win_geo.R_YUnitsPerPixel

#define TRANX(xval) \
(((double) ((xval) - wi->XOrgX)) * wi->XUnitsPerPixel + wi->UsrOrgX)

#define TRANY(yval) \
(wi->UsrOppY - (((double) ((yval) - wi->XOrgY)) * wi->YUnitsPerPixel))



extern char *ParsedColourName;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

extern int print_immediate, Animating, AddPoint_discard, Determine_tr_curve_len, Determine_AvSlope;

extern Window thePrintWindow, theSettingsWindow;
extern LocalWin *thePrintWin_Info;

extern Window ascanf_window;

extern double *ascanf_setNumber, *ascanf_Counter, *ascanf_numPoints;

int ascanf_raw_display ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin ){
		*result= (double) ActiveWin->raw_display;
		if( ascanf_arguments && !ascanf_SyntaxCheck ){
			if( (RawDisplay(ActiveWin, (int)args[0])!= (int) *result)
				&& (ascanf_arguments<= 1 || !ASCANF_TRUE(args[1]))
			){
				ActiveWin->redraw= 1;
			}
		}
	}
	else{
		*result= -1;
	}
	return(1);
}

int ascanf_Silenced ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow && !ActiveWin->delete_it ){
	  LocalWin *lwi= ActiveWin;
		if( ascanf_arguments && !ascanf_SyntaxCheck ){
		  int os= lwi->silenced;
			lwi->silenced= (int) args[0];
			xtb_bt_set( lwi->ssht_frame.win, lwi->silenced, (char *) 0);
			lwi->dev_info.xg_silent( lwi->dev_info.user_state, lwi->silenced );
			if( (!lwi->silenced && os< 0)
				&& (ascanf_arguments<= 1 || !ASCANF_TRUE(args[1]))
			){
				lwi->redraw= 1;
			}
		}
		*result= (double) X_silenced( lwi);
	}
	else{
		*result= -1;
	}
	return(1);
}

int ascanf_Fitting ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && !ActiveWin->delete_it ){
		*result= (ActiveWin->fit_xbounds> 0 || ActiveWin->fit_ybounds> 0)? (double) ActiveWin->fitting : -1;
	}
	else{
		*result= -1;
	}
	return(1);
}

int ascanf_DumpProcessed ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow && !ActiveWin->delete_it ){
		switch( PS_STATE(ActiveWin)->Printing ){
			case XG_DUMPING:
				  /* XGraph dump allows choice between dumping of processed and raw values:	*/
				*result= (double) ActiveWin->DumpProcessed;
				break;
			case PS_PRINTING:
				  /* All other printing types (probably) dump processed values:	*/
				*result= 1;
				break;
			default:
				*result= -1;
				break;
		}
	}
	else{
		*result= -1;
	}
	return(1);
}

int ascanf_EventLevel ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && !ActiveWin->delete_it ){
		*result= (double) ActiveWin->event_level;
	}
	else{
		*result= -1;
	}
	return(1);
}

void _ascanf_RedrawNow( int unsilence, int all, int update )
{ char active= False;
	if( !active && ((ActiveWin && ActiveWin!= &StubWindow) || all) ){
	  LocalWin *awi= ActiveWin, *wi;
	  LocalWindows *WL= WindowList;
		active= True;
		if( all ){
			wi= WL->wi;
		}
		else{
			wi= ActiveWin;
		}
		do{
		  int pim= print_immediate;
		  int os= wi->silenced;
			if( wi->mapped || wi== ActiveWin ){
				if( os && unsilence ){
					wi->silenced= False;
					xtb_bt_set( wi->ssht_frame.win, wi->silenced, (char *) 0);
					wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
				}
				print_immediate= 0;
				  // 20081208
				if( update ){
					wi->animate= 0;
					Animating= False;
					do{
						RedrawNow(wi);
					} while( wi->redraw && !(wi->animate || Animating) );
				}
				else{
				 XEvent dum;
				  /* 20050110: a manual redraw cancels all pending Expose events that would
				   \ cause (subsequent) redraws. (NB: events generated after this point are not
				   \ affected, of course.)
				   */
					while( XCheckWindowEvent(disp, wi->window, ExposureMask|VisibilityChangeMask, &dum) );
					RedrawNow( wi );
				}
				print_immediate= pim;
				if( wi->silenced!= os ){
					wi->silenced= os;
					xtb_bt_set( wi->ssht_frame.win, wi->silenced, (char *) 0);
					wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
				}
			}
			if( all && WL->next ){
				WL= WL->next;
				wi= WL->wi;
			}
			else{
				wi= NULL;
			}
		} while( wi );
		if( update ){
			Handle_An_Events( -1, 1, "_ascanf_RedrawNow", 0, 0 );
		}
		ActiveWin= awi;
		active= False;
	}
}

int ascanf_RedrawNow ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
  int unsilence= True;
  int all;
	if( ascanf_arguments && args ){
		*result= args[0];
		unsilence= ASCANF_TRUE(args[0]);
	}
	else{
		*result= ascanf_progn_return;
	}
	if( ascanf_arguments> 1 && ASCANF_TRUE(args[1]) ){
		all= True;
	}
	else{
		all= False;
	}
	if( !ascanf_SyntaxCheck ){
		_ascanf_RedrawNow( unsilence, all, 0 );
	}
	if( ascanf_arguments && args ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return(1);
}

int ascanf_DrawTime ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && !ascanf_SyntaxCheck ){
		Elapsed_Since( &ActiveWin->draw_timer, False );
		if( ascanf_arguments ){
			*result= ActiveWin->draw_timer.Time;
		}
		else{
			*result= ActiveWin->draw_timer.HRTot_T;
		}
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_ActiveWin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern LocalWin StubWindow, *InitWindow;

	if( ActiveWin ){
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(window 0x%lx %02d:%02d:%02d x11ID=%lu)== ",
				ActiveWin, ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number,
				ActiveWin->window
			);
		}
	}
	*result= (ActiveWin== &StubWindow)? -1 : (double) ((unsigned long) ActiveWin);
	if( args && ascanf_arguments ){
	  LocalWin *lwi1= NULL, *lwi2= NULL;
	  long d;
		CLIP_EXPR_CAST( long, d, double, args[0], -LONG_MAX, LONG_MAX );
		lwi1= (LocalWin*) (d);
		if( d== -1 ){
			ActiveWin= (InitWindow)? InitWindow : &StubWindow;
		}
		else if( lwi1 ){
		  LocalWindows *WL= WindowList;
			CLIP_EXPR_CAST( unsigned long, d, double, args[0]+1, -LONG_MAX, LONG_MAX );
			lwi2= (LocalWin*) (d);
			while( WL && WL->wi!= lwi1 ){
				WL= WL->next;
			}
			if( !WL && lwi2 ){
				WL= WindowList;
				while( WL && WL->wi!= lwi2 ){
					WL= WL->next;
				}
			}
			if( WL ){
				ActiveWin= WL->wi;
			}
		}
		else{
			ActiveWin= NULL;
		}
		if( ActiveWin ){
			if( WindowList ){
			  /* Only allow event checking when there is a WindowList - otherwise ascanf_check_event() might think
			   \ we're actually quitting, while we may be starting up!!
			   */
				ascanf_window= ActiveWin->window;
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "(new active window 0x%lx %02d:%02d:%02d x11ID=%lu)== ",
					ActiveWin, ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number,
					ActiveWin->window
				);
			}
		}
	}
	return(1);
}

int ascanf_ActiveWinPrinting ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		switch( PS_STATE(ActiveWin)->Printing ){
			default:
			case X_DISPLAY:
				*result= 0;
				break;
			case PS_PRINTING:
			case PS_FINISHED:
				*result= 1;
				break;
			case XG_DUMPING:
				*result= 2;
				break;
		}
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_ActiveWinTWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		*result= (double) (ActiveWin->dev_info.area_w);
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_ActiveWinTHeight ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		*result= (double) (ActiveWin->dev_info.area_h);
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_ActiveWinXMin ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		*result= (double) (ActiveWin->loX);
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_ActiveWinXMax ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		*result= (double) (ActiveWin->hiX);
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_ActiveWinYMin ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		*result= (double) (ActiveWin->loY);
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_ActiveWinYMax ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		*result= (double) (ActiveWin->hiY);
	}
	else{
		*result= 0;
	}
	return(1);
}

int new_tr_LABEL( char *d, char *s )
{ int n= 0;
	while( s && *s && n< MAXBUFSIZE ){
		if( *s== '%' ){
		  char *ins= NULL;
			switch( s[1] ){
				case '%':
					*d++= *s++;
					s++;
					n+= 1;
					break;
				case 'X':
					ins= ActiveWin->XUnits;
					s+= 2;
					break;
				case 'Y':
					ins= ActiveWin->YUnits;
					s+= 2;
					break;
				default:
					*d++= *s++;
					*d++= *s++;
					n+= 2;
					break;
			}
			while( ins && *ins && n< MAXBUFSIZE ){
				*d++= *ins++;
				n+= 1;
			}
		}
		else{
			*d++= *s++;
			n+= 1;
		}
	}
	*d= '\0';
	return( n );
}

int ascanf_ActiveWinXLabel ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	*result= 0;
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( ascanf_arguments ){
		  ascanf_Function *ret= NULL, *new;
		  int take_usage;
			if( args[0]== 0 || (ret= parse_ascanf_address( args[0], _ascanf_variable, "ascanf_ActiveWinXLabel",
					(int) ascanf_verbose, &take_usage ))
			){
				if( !ascanf_SyntaxCheck ){
					if( ret ){
						xfree( ret->usage );
						ret->usage= strdup( ActiveWin->tr_XUnits );
						ret->is_usage= True;
						ret->take_usage= True;
					}
					*result= args[0];
					if( ascanf_arguments> 1 ){
						if( (new= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_ActiveWinXLabel",
							(int) ascanf_verbose, &take_usage )) && take_usage && new->usage
						){
							new_tr_LABEL( ActiveWin->tr_XUnits, new->usage );
						}
						else{
							ascanf_emsg= " (2nd argument should be a valid stringpointer) ";
							ascanf_arg_error= True;
						}
					}
				}
			}
			else{
				ascanf_emsg= " (1st argument should be NULL or a (string)pointer) ";
				ascanf_arg_error= True;
			}
		}
	}
	return(1);
}

int ascanf_ActiveWinYLabel ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	*result= 0;
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( ascanf_arguments ){
		  ascanf_Function *ret= NULL, *new;
		  int take_usage;
			if( args[0]== 0 || (ret= parse_ascanf_address( args[0], _ascanf_variable, "ascanf_ActiveWinYLabel",
					(int) ascanf_verbose, &take_usage ))
			){
				if( !ascanf_SyntaxCheck ){
					if( ret ){
						xfree( ret->usage );
						ret->usage= strdup( ActiveWin->tr_YUnits );
						ret->is_usage= True;
						ret->take_usage= True;
					}
					*result= args[0];
					if( ascanf_arguments> 1 ){
						if( (new= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_ActiveWinYLabel",
							(int) ascanf_verbose, &take_usage )) && take_usage && new->usage
						){
/* 							strncpy( ActiveWin->tr_YUnits, new->usage, MAXBUFSIZE-1 );	*/
							new_tr_LABEL( ActiveWin->tr_YUnits, new->usage );
						}
						else{
							ascanf_emsg= " (2nd argument should be a valid stringpointer) ";
							ascanf_arg_error= True;
						}
					}
				}
			}
			else{
				ascanf_emsg= " (1st argument should be NULL or a (string)pointer) ";
				ascanf_arg_error= True;
			}
		}
	}
	return(1);
}

int ascanf_QueryPointer ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  extern double ascanf_PointerPos[2];
	if( ActiveWin && ActiveWin!= &StubWindow ){
	  LocalWin *wi= ActiveWin;
	  Window dum;
	  int dum2, curX, curY;
	  unsigned int mask_rtn;

		if( XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,
			  &curX, &curY, &mask_rtn
			)
		){
			ascanf_PointerPos[0]= Reform_X( wi, TRANX(curX), TRANY(curY) );
			ascanf_PointerPos[1]= Reform_Y( wi, TRANY(curY), TRANX(curX) );
			*result= 1;
		}
		else{
			*result= 0;
		}
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_setFitOnce ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments ){
		if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		  extern int FitOnce;
			if( ASCANF_TRUE(args[0]) ){
				*result= ActiveWin->FitOnce= FitOnce= 1;
			}
			else{
				*result= ActiveWin->FitOnce= FitOnce= 0;
			}
		}
	}
	else{
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= (ActiveWin->FitOnce || ActiveWin->fit_xbounds || ActiveWin->fit_ybounds);
		}
		else{
			ascanf_arg_error= 1;
			*result= 0;
		}
	}
	return(1);
}

extern int DiscardedShadows;

int ascanf_DiscardedShadows ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments ){
		if( !ascanf_SyntaxCheck ){
			if( args[0] ){
				*result= DiscardedShadows= SIGN(args[0]);
			}
			else{
				*result= DiscardedShadows= 0;
			}
		}
	}
	else{
		*result= DiscardedShadows;
	}
	return(1);
}

int ascanf_setQuick ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ascanf_arguments ){
		if( ActiveWin && !ascanf_SyntaxCheck ){
			if( args[0] ){
				*result= ActiveWin->use_transformed= 1;
			}
			else{
				*result= ActiveWin->use_transformed= 0;
			}
		}
	}
	else{
		if( ActiveWin ){
			*result= ActiveWin->use_transformed;
		}
		else{
			ascanf_arg_error= 1;
			*result= 0;
		}
	}
	return(1);
}

int ascanf_setredraw ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		ActiveWin->animate= 1;
		Animating= True;
	}
	if( ascanf_arguments ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return(1);
}

extern xtb_frame HO_Dialog;

#ifdef OLD_ASCANF_PRINT
int ascanf_Print ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ActiveWin && ActiveWin!= &StubWindow && !ActiveWin->HO_Dialog && !ascanf_SyntaxCheck ){
	  int pim= print_immediate;
	  LocalWin *wi= ActiveWin;
		print_immediate= 1;
		RedrawNow( ActiveWin );
		print_immediate= pim;
		ActiveWin= wi;
	}
	if( ascanf_arguments ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return(1);
}
#else
int ascanf_Print ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
  static char active= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !active && !ascanf_SyntaxCheck ){
	  int pim= print_immediate;
	  LocalWin *wi= ActiveWin;
		print_immediate= -2;
		active= 1;
		if( HO_Dialog.mapped ){
			CloseHO_Dialog( &HO_Dialog );
		}
		thePrintWindow= 0;
		thePrintWin_Info= NULL;
		wi->HO_Dialog= NULL;
		PrintWindow( wi->window, wi );
		print_immediate= pim;
		ActiveWin= wi;
		active= 0;
	}
	if( ascanf_arguments ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return(1);
}

#endif

int ascanf_PrintOK ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
  static char active= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !active && !ascanf_SyntaxCheck ){
	  int pim= print_immediate;
	  LocalWin *wi= ActiveWin;
		print_immediate= -1;
		active= 1;
/* 		RedrawNow( ActiveWin );	*/
		if( HO_Dialog.mapped ){
			CloseHO_Dialog( &HO_Dialog );
		}
		thePrintWindow= 0;
		thePrintWin_Info= NULL;
		wi->HO_Dialog= NULL;
		PrintWindow( wi->window, wi );
		print_immediate= pim;
		ActiveWin= wi;
		active= 0;
	}
	if( ascanf_arguments ){
		*result= args[0];
	}
	else{
		*result= ascanf_progn_return;
	}
	return(1);
}

int ascanf_SetCycleSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		*result= ActiveWin->plot_only_set0;
		if( ascanf_arguments && args[0]>= 0 && args[0]< setNumber ){
			ActiveWin->plot_only_set0= (int) args[0];
			cycle_plot_only_set( ActiveWin, 0 );
			ActiveWin->animate= 1;
			Animating= True;
		}
		if( ascanf_arguments> 1 && args[1] ){
			XG_XSync( disp, False );
		}
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_CycleSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ActiveWin && ActiveWin!= &StubWindow && args[0] && !ascanf_SyntaxCheck ){
		cycle_plot_only_set( ActiveWin, (int) args[0] );
		ActiveWin->animate= 1;
		Animating= True;
		if( ascanf_arguments> 1 && args[1] ){
			XG_XSync( disp, False );
		}
	}
	*result= ascanf_progn_return;
	return(1);
}

int ascanf_CycleDrawnSets ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ActiveWin && ActiveWin!= &StubWindow && args[0] && !ascanf_SyntaxCheck ){
		cycle_drawn_sets( ActiveWin, (int) args[0] );
		ActiveWin->animate= 1;
		Animating= True;
		if( ascanf_arguments> 1 && args[1] ){
			XG_XSync( disp, False );
		}
	}
	*result= ascanf_progn_return;
	return(1);
}

int ascanf_newGroup ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	*result= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		if( args ){
			if( args[0]< -1 || args[0]>= setNumber ){
				ascanf_emsg= " (invalid set) ";
				ascanf_arg_error= 1;
			}
			else{
			  int n, N;
			  int nf= (ascanf_arguments> 1 && ASCANF_TRUE(args[1]))? True : False;
				if( args[0]< 0 ){
					n= 0;
					N= setNumber;
				}
				else{
					n= (int) args[0];
					N= n+1;
				}
				for( ; n< N; n++ ){
					if( ascanf_arguments> 1 ){
						ActiveWin->new_file[n]= nf;
					}
					*result= ActiveWin->new_file[n];
				}
			}
			files_and_groups( ActiveWin, NULL, NULL);
			if( ascanf_arguments> 1 ){
				ActiveWin->redraw= True;
			}
		}
		else{
			ascanf_arg_error= 1;
		}
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_Group ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	*result= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		files_and_groups( ActiveWin, NULL, NULL);
		if( args ){
			if( args[0]< 0 || args[0]>= setNumber ){
				ascanf_emsg= " (invalid set) ";
				ascanf_arg_error= 1;
			}
			else{
				*result= ActiveWin->group[ (int) args[0] ];
			}
		}
		else{
			*result= ActiveWin->numGroups;
		}
	}
	else{
		if( args ){
			*result= setNumber;
		}
	}
	return(1);
}

int ascanf_GroupSets ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int snr, i, grp, n=0;
	*result= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		files_and_groups( ActiveWin, NULL, NULL);
		if( args ){
			if( args[0]< 0 || args[0]>= setNumber ){
				ascanf_emsg= " (invalid set) ";
				ascanf_arg_error= 1;
			}
			else{
				snr= (int) args[0];
			}
		}
		else{
			snr= *ascanf_setNumber;
		}
		if( !ascanf_arg_error ){
			grp= ActiveWin->group[snr];
			for( i= 0; i< setNumber; i++ ){
				if( ActiveWin->group[i]== grp ){
					n+= 1;
				}
			}
			*result= n;
		}
	}
	else{
		if( args ){
			*result= 0;
		}
	}
	return(1);
}

int ascanf_CycleGroup ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ActiveWin && ActiveWin!= &StubWindow && args[0] && !ascanf_SyntaxCheck ){
		cycle_plot_only_group( ActiveWin, (int) args[0] );
		ActiveWin->animate= 1;
		Animating= True;
		if( ascanf_arguments> 1 && args[1] ){
			XG_XSync( disp, False );
		}
	}
	*result= ascanf_progn_return;
	return(1);
}

int ascanf_File ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	*result= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		files_and_groups( ActiveWin, NULL, NULL);
		if( args ){
			if( args[0]< 0 || args[0]>= setNumber ){
				ascanf_emsg= " (invalid set) ";
				ascanf_arg_error= 1;
			}
			else{
				if( ascanf_arguments> 1 && args[1]>= 0 && args[1]< MAXINT ){
					ActiveWin->fileNumber[ (int) args[0] ]= (int) args[1];
				}
				*result= ActiveWin->fileNumber[ (int) args[0] ];
			}
		}
		else{
			*result= ActiveWin->numFiles;
		}
	}
	else{
		if( args ){
			*result= setNumber;
		}
	}
	return(1);
}

int ascanf_FileSets ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int snr, i, frp, n=0;
	*result= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		files_and_groups( ActiveWin, NULL, NULL);
		if( args ){
			if( args[0]< 0 || args[0]>= setNumber ){
				ascanf_emsg= " (invalid set) ";
				ascanf_arg_error= 1;
			}
			else{
				snr= (int) args[0];
			}
		}
		else{
			snr= *ascanf_setNumber;
		}
		if( !ascanf_arg_error ){
			frp= ActiveWin->fileNumber[snr];
			for( i= 0; i< setNumber; i++ ){
				if( ActiveWin->fileNumber[i]== frp ){
					n+= 1;
				}
			}
			*result= n;
		}
	}
	else{
		if( args ){
			*result= 0;
		}
	}
	return(1);
}

int ascanf_CycleFile ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		return(0);
	}
	if( ActiveWin && ActiveWin!= &StubWindow && args[0] && !ascanf_SyntaxCheck ){
		cycle_plot_only_file( ActiveWin, (int) args[0] );
		ActiveWin->animate= 1;
		Animating= True;
		if( ascanf_arguments> 1 && args[1] ){
			XG_XSync( disp, False );
		}
	}
	*result= ascanf_progn_return;
	return(1);
}

int ascanf_Reverse ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		SwapSets( ActiveWin );
		ActiveWin->animate= 1;
		Animating= True;
		if( ascanf_arguments && args[0] ){
			XG_XSync( disp, False );
		}
	}
	*result= ascanf_progn_return;
	return(1);
}

int ascanf_All ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern double ascanf_progn_return;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
		ShowAllSets( ActiveWin );
		ActiveWin->animate= 1;
		Animating= True;
		if( ascanf_arguments && args[0] ){
			XG_XSync( disp, False );
		}
	}
	*result= ascanf_progn_return;
	return(1);
}

int ascanf_Marked ( ASCB_ARGLIST )
{ ASCB_FRAME_RESULT
  int i, N= 0;
	if( ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
	  LocalWin *wi= ActiveWin;
		for( i= 0; i< setNumber; i++ ){
			N+= (ActiveWin->draw_set[i]= (ActiveWin->mark_set[i]> 0)? 1 : 0);
		}
		ActiveWin->ctr_A= 0;
		ActiveWin->printed= 0;
		ActiveWin->redraw= 1;
		if( !wi->delete_it ){
			ActiveWin= wi;
		}
	}
	*result= N;
	return(1);
}

int ascanf_FirstDrawn ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	*result= -1;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= (ActiveWin->first_drawn== idx);
		}
	}
	else if( ActiveWin ){
		*result= ActiveWin->first_drawn;
	}
	return(1);
}

int ascanf_LastDrawn ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	*result= -1;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= (ActiveWin->last_drawn== idx);
		}
	}
	else if( ActiveWin ){
		*result= ActiveWin->last_drawn;
	}
	return(1);
}

int ascanf_DrawSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx, val, min;
	if( ascanf_arguments> 1 ){
		min= -1;
	}
	else{
		min= 0;
	}
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= min && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				val= (int) args[1];
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, "(set #%d, draw=%d in window 0x%lx %02d:%02d:%02d)== ",
						idx, val,
						ActiveWin, ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
					);
				}
				if( idx== -1 ){
					if( val== -1 ){
						r= -1;
						SwapSets(ActiveWin);
					}
					else{
						r= !val;
						for( idx= 0; idx< setNumber; idx++ ){
							if( ActiveWin->draw_set[idx]!= val ){
								ActiveWin->draw_set[idx]= val;
								ActiveWin->redraw= 1;
							}
						}
					}
				}
				else{
					r= ActiveWin->draw_set[idx];
					ActiveWin->draw_set[idx]= val;
					if( r!= val ){
						ActiveWin->redraw= 1;
					}
				}
				*result= r;
			}
			else{
				*result= ActiveWin->draw_set[idx];
			}
		}
		else{
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				val= (int) args[1];
				if( pragma_unlikely(ascanf_verbose) && ActiveWin ){
					fprintf( StdErr, "(set #%d, draw=%d in global set information)== ",
						idx, val,
						ActiveWin, ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
					);
				}
				if( idx< 0 ){
					r= !val;
					for( idx= 0; idx< setNumber; idx++ ){
						AllSets[idx].draw_set= val;
					}
				}
				else{
					r= AllSets[idx].draw_set;
					AllSets[idx].draw_set= val;
				}
				*result= r;
			}
			else{
				*result= AllSets[idx].draw_set;
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_MarkSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx, val;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				val= (int) args[1];
				r= ActiveWin->mark_set[idx];
				ActiveWin->mark_set[idx]= val;
				*result= r;
			}
			else{
				*result= ActiveWin->mark_set[idx];
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

extern double *curvelen_with_discarded;

int ascanf_DiscardedPoint ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx, pnt_nr, val;
	if( ascanf_arguments>= 2 &&
		(args[0]>= 0 && args[0]< setNumber) &&
		(args[1]>= 0 && args[1]< AllSets[(idx=(int)args[0])].numPoints)
	){ if( ActiveWin && ActiveWin!= &StubWindow ){
		pnt_nr= (int) args[1];
		if( ascanf_arguments> 2 && !ascanf_SyntaxCheck ){
		  DataSet *this_set= &AllSets[idx];
		  int change= 0;
			val= (ASCANF_TRUE(args[2]))? SIGN(args[2]) : 0;
			r= DiscardedPoint( ActiveWin, this_set, pnt_nr );
			if( val> 0 ){
				if( !this_set->discardpoint ){
					this_set->discardpoint= (signed char*) calloc( this_set->allocSize, sizeof(signed char));
				}
				if( this_set->discardpoint ){
					if( this_set->discardpoint[pnt_nr]<= 0 ){
						this_set->discardpoint[pnt_nr]= 1;
						if( !this_set->discardpoint[pnt_nr] &&
							(!*curvelen_with_discarded || (ActiveWin && *curvelen_with_discarded && ActiveWin->init_pass))
						){
							this_set->init_pass= True;
						}
					}
				}
			}
			else if( this_set->discardpoint ){
				if( this_set->discardpoint[pnt_nr]< 0 ){
					this_set->discardpoint[pnt_nr]= 0;
					change= 1;
				}
				else if( val< 0 && this_set->discardpoint[pnt_nr] ){
					this_set->discardpoint[pnt_nr]= 0;
					change= 1;
				}
			}
			if( change ){
				if( (ActiveWin && ActiveWin->raw_display) || 
					(!*curvelen_with_discarded || (ActiveWin && *curvelen_with_discarded && ActiveWin->init_pass))
				){
					this_set->init_pass= True;
				}
			}
			if( pnt_nr== *ascanf_Counter && idx== *ascanf_setNumber ){
				AddPoint_discard= val;
			}
			*result= r;
		}
		else{
			*result= DiscardedPoint( ActiveWin, &AllSets[idx], pnt_nr );
		}
	} }
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

  /* ProcessSet[<setnr>[,pnt_nr]]	*/
int ascanf_ProcessSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int n=0, idx, pnt_nr= -1;
	if( ascanf_arguments>= 1 && (args[0]>= -1 && args[0]< setNumber) ){
		idx= (int) args[0];
		if( ascanf_arguments> 1 ){
			if( args[1]< -1 || (idx>=0 && args[1]>= AllSets[idx].numPoints) ){
				ascanf_arg_error= 1;
				ascanf_emsg= "(invalid point number)";
			}
			else{
				pnt_nr= (int) args[1];
			}
		}
		if( ActiveWin && !ascanf_SyntaxCheck && (idx< 0 || !AllSets[idx].raw_display) && ActiveWin!= &StubWindow ){
		  int i, j, N, P;
		  LocalWin *wi= ActiveWin;
		  static char active= 0;
			if( active ){
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (ProcessSet called by itself - ignored!) " );
					*result= 0;
					return(0);
				}
			}
			active= 1;
			if( idx< 0 ){
				i= 0;
				N= setNumber;
			}
			else{
				i= idx;
				N= idx+1;
			}
			Draw_Process( wi, True );
			for( ; i< N; i++ ){
			  DataSet *this_set= &AllSets[i];
			  double sx1, sy1, sx2, sy2, sx3, sy3, sx4, sy4, sx5, sy5, sx6, sy6;
			  int ncoords;
				this_set->last_processed_wi= NULL;
				if( pnt_nr< 0 ){
					j= 0;
					P= this_set->numPoints;
				}
				else if( pnt_nr< this_set->numPoints ){
					j= pnt_nr;
					P= pnt_nr+ 1;
				}
				else{
					P= 0;
					j= 1;
					fprintf( StdErr, " (invalid point number %d for set %d - skipping set) ", pnt_nr, i );
					if( !pragma_unlikely(ascanf_verbose) ){
						fputc( '\n', StdErr );
					}
				}
				if( wi->vectorFlag && this_set->vectorType== 2 && this_set->lcol>= 0 ){
					ncoords= 4;
				}
				else{
					ncoords= 3;
				}
				CurveLen( wi, i, P, True, False, NULL );
				*ascanf_setNumber= i;
				for( ; j< P; j++ ){
					*ascanf_numPoints= this_set->numPoints;
					this_set->data[0][0]= sx1 = XVAL( this_set, j);
					this_set->data[0][1]= sy1 = YVAL( this_set, j);
					this_set->data[0][2]= ERROR( this_set, j);
					if( this_set->lcol>= 0 ){
						this_set->data[0][3]= VVAL( this_set, j );
					}
					DrawData_process( wi, this_set, this_set->data, j, 1, ncoords,
						&sx1, &sy1, &sx2, &sy2, &sx3, &sy3, &sx4, &sy4, &sx5, &sy5, &sx6, &sy6
					);
					this_set->xvec[j]= sx1;
					this_set->yvec[j]= sy1;
					this_set->errvec[j]= this_set->data[0][2];
					this_set->lvec[j]= this_set->data[0][3];
					n+= 1;
				}
#ifdef TR_CURVE_LEN
				if( Determine_tr_curve_len ){
					tr_CurveLen( wi, i, P, True, False, NULL );
				}
#endif
				this_set->last_processed_wi= wi;
			}
			Draw_Process( wi, False );
			*result= n;
			active= 0;
			ActiveWin= wi;
		}
		else{
			*result= 0;
		}
	}
	else{
		ascanf_arg_error= 1;
		ascanf_emsg= "(invalid setnumber)";
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_BoxFilter ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL;
  int idx= -1, pnt_nr= -1;
	set_NaN(*result);
	ascanf_arg_error= 0;
	if( args && ascanf_arguments>= 5 ){
	  int take_usage;
	  char *fname;
	  DataSet *this_set= NULL;
		if( (af= parse_ascanf_address( args[0], _ascanf_variable, "ascanf_BoxFilter", (int) ascanf_verbose, &take_usage)) ){
			fname= af->usage;
		}
		else{
			if( args[0] ){
				fprintf( StdErr, " (warning: ignoring non-string 1st argument %s) ", ad2str(args[0], d3str_format, NULL) );
			}
			fname= NULL;
		}
		if( ascanf_arguments> 5 ){
			if( args[5]>= 0 && args[5]< setNumber ){
				idx= (int) args[5];
				this_set= &AllSets[idx];
				if( ascanf_arguments> 6 ){
					if( args[6]>= 0 && args[6]< this_set->numPoints ){
						pnt_nr= (int) args[6];
					}
					else{
						fprintf( StdErr,
							" (warning: ignoring invalid pointnumber %s for set %s) ",
							ad2str(args[6], d3str_format, NULL), ad2str(args[5], d3str_format, NULL)
						);
					}
				}
			}
			else{
				fprintf( StdErr, " (warning: ignoring invalid setnumber %s) ", ad2str(args[5], d3str_format, NULL) );
			}
		}
		if( ActiveWin && !ascanf_SyntaxCheck && !ascanf_arg_error ){
			new_param_now( NULL, NULL, 0 );
			*result= FilterPoints_Box( ActiveWin, fname, args[1], args[2], args[3], args[4], this_set, pnt_nr );
		}
	}
	else{
		ascanf_arg_error= True;
	}
	return( !ascanf_arg_error );
}

int ascanf_Markers ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( args && ascanf_arguments>= 1 ){
	  int snr;
		*result= 0;
		if( args[0]>= 0 && args[0]< setNumber ){
			snr= (int) args[0];
			if( ActiveWin && ActiveWin!= &StubWindow ){
			  DataSet *this_set= &AllSets[snr];
				*result= this_set->markFlag;
				if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
					if( args[1]>= 0 && args[1]< ERROR_TYPES ){
						*result= this_set->markFlag= (args[1])? 1 : 0;
						if( ascanf_arguments> 2 ){
						  double t;
							CLIP_EXPR( t, args[2], 0, 2 );
							this_set->pixelMarks= (int) t;
						}
						if( ascanf_arguments> 3 ){
							this_set->markSize= args[3];
						}
					}
				}
			}
		}
		else{
			ascanf_emsg= "(invalid setnumber)";
			ascanf_arg_error= 1;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= True;
		*result= 0;
		return(0);
	}
}

int ascanf_error_type ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( args && ascanf_arguments>= 1 ){
	  int snr, type= -1, fit= 0;
		*result= -1;
		if( args[0]>= 0 && args[0]< setNumber ){
		  extern int get_error_type(), set_error_type();
			if( ActiveWin && ActiveWin!= &StubWindow ){
				snr= (int) args[0];
				*result= get_error_type( ActiveWin, snr );
				if( ascanf_arguments> 1 ){
					if( args[1]>= 0 && args[1]< ERROR_TYPES ){
						type= (int) args[1];
						if( ascanf_arguments> 2 ){
							fit= (args[2])? 1 : 0;
						}
						if( !ascanf_SyntaxCheck ){
							set_error_type( ActiveWin, snr, &type, !fit );
							*result= type;
						}
					}
					else{
						ascanf_emsg= "(invalid error type)";
						ascanf_arg_error= 1;
						*result= -1;
					}
				}
			}
		}
		else{
			ascanf_emsg= "(invalid setnumber)";
			ascanf_arg_error= 1;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= True;
		*result= 0;
		return(0);
	}
}

int ascanf_HighlightSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx, val;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				val= (int) args[1];
				r= ActiveWin->legend_line[idx].highlight;
				ActiveWin->legend_line[idx].highlight= val;
				*result= r;
			}
			else{
				*result= ActiveWin->legend_line[idx].highlight;
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_AdornInt ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
			r= AllSets[idx].adorn_interval;
			AllSets[idx].adorn_interval= (int) args[1];
			if( ActiveWin && 
				(r!= (int) args[1] && (ascanf_arguments<= 2 || !ASCANF_TRUE(args[2])))
				&& ActiveWin!= &StubWindow
			){
				ActiveWin->redraw= 1;
			}
			*result= r;
		}
		else{
			*result= AllSets[idx].adorn_interval;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_PlotInt ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= -1 && idx< setNumber)
	){
		if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
		  int N;
			if( idx== -1 ){
				idx= 0;
				N= setNumber;
			}
			else{
				N= idx+1;
			}
			for( ; idx< N; idx++ ){
				r= AllSets[idx].plot_interval;
				AllSets[idx].plot_interval= (int) args[1];
				if( ActiveWin &&
					(r!= (int) args[1] && (ascanf_arguments<= 2 || !ASCANF_TRUE(args[2])))
				){
					ActiveWin->redraw= 1;
				}
				*result= r;
			}
		}
		else if( idx>= 0 ){
			*result= AllSets[idx].plot_interval;
		}
		else{
			*result= 0;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int Adorned_Points( DataSet *this_set )
{ int adorned;
	if( (this_set->adorn_interval && (this_set->plot_interval % this_set->adorn_interval)== 0) ||
		(this_set->plot_interval && (this_set->adorn_interval % this_set->plot_interval)== 0)
	){
	  int n= MAX( this_set->plot_interval, this_set->adorn_interval );
		if( this_set->numPoints % n ){
			adorned= this_set->numPoints/n + 1;
		}
		else{
			adorned= this_set->numPoints/n;
		}
	}
	else{
	  int n= MAX(1,this_set->plot_interval)* MAX(1,this_set->adorn_interval);
		if( this_set->numPoints % n ){
			adorned= this_set->numPoints/n + 1;
		}
		else{
			adorned= this_set->numPoints/n;
		}
	}
	return( adorned );
}

int ascanf_RawSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx, val;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( !ascanf_SyntaxCheck ){
		if( idx>= 0 && idx< setNumber ){
			if( ascanf_arguments> 1 ){
				val= ASCANF_TRUE( args[1] );
				r= AllSets[idx].raw_display;
				AllSets[idx].raw_display= val;
				*result= r;
			}
			else{
				*result= AllSets[idx].raw_display;
			}
		}
		else{
			ascanf_arg_error= 1;
			*result= 0;
			return(1);
		}
	}
	return(1);
}

int ascanf_FloatSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx, val;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
			val= (int) args[1];
			r= AllSets[idx].floating;
			AllSets[idx].floating= val;
			*result= r;
		}
		else{
			*result= AllSets[idx].floating;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_RedrawSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, immediate= 0;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= -1 && idx< setNumber)
	){
		if( !ascanf_SyntaxCheck ){
			if( ascanf_arguments> 1 ){
				immediate= (int) args[1];
			}
			*result= RedrawSet( idx, immediate );
		}
		else{
			*result= 0;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SkipSetOnce ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int r, idx, val;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
			val= (int) args[1];
			r= AllSets[idx].skipOnce;
			AllSets[idx].skipOnce= val;
			*result= r;
		}
		else{
			*result= AllSets[idx].skipOnce;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_ShiftSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( args && ascanf_arguments>= 2 ){
	  int snr, dirn= 0, extreme= 0, redraw=1;
		*result= -1;
		if( args[0]>= 0 && args[0]< setNumber ){
			snr= (int) args[0];
			if( args[1]< 0 || args[1]> 0 ){
				dirn= (args[1]< 0)? -1 : 1;
				if( ascanf_arguments> 2 ){
					extreme= (args[2])? True : False;
				}
				if( ascanf_arguments> 3 ){
					redraw= (args[3])? True : False;
				}
				if( !ascanf_SyntaxCheck ){
					*result= ShiftDataSet( snr, dirn, extreme, redraw );
				}
			}
			else{
				ascanf_arg_error= 1;
				ascanf_emsg= "(direction must be <0 or >0)";
			}
		}
		else{
			ascanf_emsg= "(invalid setnumber)";
			ascanf_arg_error= 1;
		}
		return( 1 );
	}
	else{
		ascanf_arg_error= True;
		*result= 0;
		return(0);
	}
}

int ascanf_LinkedSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	*result= 0;
	if( args && ascanf_arguments>= 1 ){
		if( args[0]>= 0 && args[0]< setNumber ){
		  DataSet *this_set= &AllSets[(int) args[0]];
			if( this_set->set_link>= 0 && this_set->set_link< setNumber ){
				if( ascanf_arguments>= 2 ){
				  ascanf_Function *af;
					if(
						(af= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_LinkedSet", (int) ascanf_verbose, NULL))
					){
						af->value= this_set->set_link;
					}
					else if( ascanf_arguments>= 3 && ASCANF_TRUE(args[2]) &&
						args[1]>=0 && args[1]< setNumber
					){
						LinkSet2( this_set, (int) args[1] );
					}
				}
				*result= 1;
			}
			else if( ascanf_arguments>= 3 && ASCANF_TRUE(args[2]) &&
				args[1]>=0 && args[1]< setNumber
			){
				LinkSet2( this_set, (int) args[1] );
				*result= 1;
			}
		}
		else{
			ascanf_arg_error= True;
			ascanf_emsg= "(invalid setnumber)";
		}
		return(1);
	}
	else{
		ascanf_arg_error= True;
		return(0);
	}
}

int ascanf_CheckAssociations_AND ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
	  DataSet *set= &AllSets[idx];
		if( ascanf_arguments== 1 ){
			*result= set->numAssociations;
		}
		else if( set->numAssociations && !ascanf_SyntaxCheck ){
		  int i, j;
		  double a;
			*result= 1;
			for( i= 1; i< ascanf_arguments && *result; i++ ){
			  int hit= 0;
				a= args[i];
				for( j= 0; j< set->numAssociations && !hit; j++ ){
					if( set->Associations[j]!= a ){
						hit= 1;
					}
				}
				if( hit ){
					*result= 0;
				}
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " {%s", ad2str( set->Associations[0], NULL, NULL) );
				for( i= 1; i< set->numAssociations; i++ ){
					fprintf( StdErr, ",%s", ad2str( set->Associations[i], NULL, NULL) );
				}
				if( *result ){
					fprintf( StdErr, "}[%d]==%g== %g == ", j-1, set->Associations[j-1], a );
				}
				else{
					fprintf( StdErr, "}[%d]==%g!= %g == ", j-1, set->Associations[j-1], a );
				}
			}
		}
		else{
			*result= 0;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_CheckAssociations_OR ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
	  DataSet *set= &AllSets[idx];
		if( ascanf_arguments== 1 ){
			*result= set->numAssociations;
		}
		else if( set->numAssociations && !ascanf_SyntaxCheck ){
		  int i, j;
		  double a;
			*result= 0;
			for( i= 1; i< ascanf_arguments && !*result; i++ ){
				a= args[i];
				for( j= 0; j< set->numAssociations && !*result; j++ ){
					if( set->Associations[j]== a ){
						*result= 1;
					}
				}
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " {%s", ad2str( set->Associations[0], NULL, NULL) );
				for( i= 1; i< set->numAssociations; i++ ){
					fprintf( StdErr, ",%s", ad2str( set->Associations[i], NULL, NULL) );
				}
				if( *result ){
					fprintf( StdErr, "}[%d]==%g== %g == ", j-1, set->Associations[j-1], a );
				}
				else{
					fprintf( StdErr, "}[%d]==%g!= %g == ", j-1, set->Associations[j-1], a );
				}
			}
		}
		else{
			*result= 0;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_CheckAssociation_OR ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, ass;
	if( ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	if( (args[0]< 0 || args[0]>= setNumber) ){
		ascanf_emsg= " (set# out of range) ";
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	if( args[1]>= 0 && args[1]< AllSets[(idx= (int)args[0])].numAssociations ){
	  DataSet *set= &AllSets[idx];
		ass= (int) args[1];
		if( set->numAssociations && !ascanf_SyntaxCheck ){
		  int i, j;
		  double ref= set->Associations[ass], a;
			*result= 0;
			for( j= 2; j< ascanf_arguments && !*result; j++ ){
				a= args[j];
				if( ref== a ){
					*result= 1;
				}
			}
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " {%s", ad2str( set->Associations[0], NULL, NULL) );
				for( i= 1; i< set->numAssociations; i++ ){
					fprintf( StdErr, ",%s", ad2str( set->Associations[i], NULL, NULL) );
				}
				if( *result ){
					fprintf( StdErr, "}[%d]==%g== arg[%d]=%g == ", ass, ref, j-1, a );
				}
				else{
					fprintf( StdErr, "}[%d]==%g!= arg[%d]=%g == ", ass, ref, j-1, a );
				}
			}
		}
		else{
			*result= 0;
		}
	}
	else{
		ascanf_emsg= " (assc# out of range) ";
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_getAssociation ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, aI;
	set_NaN( *result );
	if( ascanf_arguments>= 1 ){
		if( (idx= (int)args[0])>= 0 && idx< setNumber ){
		  DataSet *set= &AllSets[idx];
			if( ascanf_arguments== 1 ){
				*result= set->numAssociations;
			}
			else{
			  ascanf_Function *af;
			  int i;
				if( (af= parse_ascanf_address(args[1], _ascanf_array, "ascanf_getAssociation", ascanf_verbose, NULL)) ){
					Resize_ascanf_Array( af, set->numAssociations, result );
					if( af->N!= set->numAssociations ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (can't resize %s (%s)) ", af->name, serror() );
							fflush( StdErr );
							ascanf_arg_error= True;
						}
					}
					else{
						for( i= 0; i< set->numAssociations; i++ ){
							ASCANF_ARRAY_ELEM_SET(af, i, set->Associations[i]);
						}
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " {%s", ad2str( set->Associations[0], NULL, NULL) );
							for( i= 1; i< set->numAssociations; i++ ){
								fprintf( StdErr, ",%s", ad2str( set->Associations[i], NULL, NULL) );
							}
							fprintf( StdErr, "}== " );
						}
						*result= set->numAssociations;
					}
				}
				else if( (aI= (int)args[1])>= 0 && aI< set->numAssociations ){
					*result= set->Associations[aI];
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " {%s", ad2str( set->Associations[0], NULL, NULL) );
						for( i= 1; i< set->numAssociations; i++ ){
							fprintf( StdErr, ",%s", ad2str( set->Associations[i], NULL, NULL) );
						}
						fprintf( StdErr, "}[%d]== ", aI );
					}
				}
				else{
					if( !ascanf_SyntaxCheck || aI< 0 ){
						ascanf_arg_error= 1;
						ascanf_emsg= "(assc# out of range)";
					}
					else{
						*result= 0;
					}
				}
			}
		}
		else{
			if( !ascanf_SyntaxCheck || idx< 0 ){
				ascanf_arg_error= 1;
				ascanf_emsg= "(set# out of range)";
				set_NaN( *result );
			}
			else{
				*result= 0;
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(0);
	}
	return(1);
}

int ascanf_Associate ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, ass, n;
	if( ascanf_arguments>= 2 ){
		if( (idx= (int)args[0])>= 0 && idx< setNumber &&
			(ass= (int) args[1])>= -1
		){
		  DataSet *this_set= &AllSets[idx];
			if( (n= ascanf_arguments- 2)> 0 ){
			  int new_alloc;
				if( ass== -1 ){
					ass= this_set->numAssociations;
				}
				new_alloc= (ass+ n)- this_set->allocAssociations;
				if( ascanf_SyntaxCheck ){
					*result= this_set->numAssociations;
				}
				else if( new_alloc<= 0 || (this_set->Associations= (double*) realloc( this_set->Associations,
						sizeof(double)* (this_set->allocAssociations+ new_alloc))
					)
				){
				  int i;
					if( new_alloc> 0 ){
						if( (this_set->allocAssociations+= new_alloc)> ASCANF_MAX_ARGS ){
							Ascanf_AllocMem( this_set->allocAssociations );
						}
						for( i= this_set->numAssociations; i< this_set->allocAssociations; i++ ){
							set_NaN(this_set->Associations[i]);
						}
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (%d new associations) ", new_alloc );
						}
					}
					for( i= 0; i< n; i++ ){
						this_set->Associations[i+ ass]= args[i+2];
					}
					if( i+ ass>= this_set->numAssociations ){
						this_set->numAssociations= i+ ass;
					}
					if( pragma_unlikely(ascanf_verbose) ){
					  int i;
						fprintf( StdErr, " {%s", ad2str( AllSets[idx].Associations[0], NULL, NULL) );
						for( i= 1; i< AllSets[idx].numAssociations; i++ ){
							fprintf( StdErr, ",%s", ad2str( AllSets[idx].Associations[i], NULL, NULL) );
						}
						fprintf( StdErr, "}[-1]== " );
					}
				}
				else{
					fprintf( StdErr, "ascanf_Associate(%d): can't (re)allocate memory for %d associations (%d new) (%s)\n",
						idx,
						this_set->allocAssociations+ new_alloc, new_alloc, serror()
					);
				}
			}
			else if( !ascanf_SyntaxCheck ){
				if( ass== 0 ){
					xfree( this_set->Associations );
					this_set->numAssociations= this_set->allocAssociations= 0;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " removed associations for set #%d== ",
							this_set->set_nr
						);
					}
				}
				else if( ass> 0 ){
				  int i;
					this_set->allocAssociations= ass;
					this_set->Associations= (double*) realloc( this_set->Associations,
						sizeof(double)* this_set->allocAssociations
					);
					for( i= this_set->numAssociations; i< this_set->allocAssociations; i++ ){
						this_set->Associations[i]= 0;
					}
					this_set->numAssociations= ass;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " number of associations for set #%d set to %d== ",
							this_set->set_nr, this_set->numAssociations
						);
					}
				}
			}
			*result= this_set->numAssociations;
		}
		else{
			ascanf_arg_error= 1;
			*result= 0;
			ascanf_emsg= "(set# out of range)";
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
	}
	return(1);
}

int ascanf_ValCat_any( double *args, double *result, ValCategory *ValCat, char *descr, char *caller )
{ int exact= 1;
	if( ascanf_arguments>= 1 ){
	  double x= args[0];
	  ValCategory *vcat, *mincat, *maxcat;
	  ascanf_Function *af;
	  int take_usage= 0;
		*result= 0;
		if( ascanf_arguments> 1 ){
			exact= (args[1])? 1 : 0;
		}
		vcat= Find_ValCat( ValCat, x, &mincat, &maxcat);
		if( vcat && vcat->category){
			if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, " (%s value %s = \"%s\") ", descr, ad2str(vcat->val, NULL, NULL), vcat->category );
			}
			*result= 1;
		}
		else{
			if( (mincat || maxcat) && !exact ){
			  /* Use the closest neighbouring category instead	*/
				if( !mincat ){
					mincat= maxcat;
				}
				else if( !maxcat ){
					maxcat= mincat;
				}
				if( x- mincat->val< maxcat->val- x ){
					vcat= mincat;
				}
				else{
					vcat= maxcat;
				}
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (close %s value %s = \"%s\") ", descr, ad2str(vcat->val, NULL, NULL), vcat->category );
				}
				*result= 1;
			}
		}
		if( ascanf_arguments> 2 &&
			(af= parse_ascanf_address( args[2], _ascanf_variable, caller, (int) ascanf_verbose, &take_usage ))
		){
			af->value= (vcat)? vcat->val : x;
			if( take_usage ){
				xfree( af->usage );
				af->usage= strdup( (vcat)? vcat->category : ad2str(x, NULL, NULL) );
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
	}
	return(1);
}

int ascanf_ValCat_X ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !ActiveWin || ascanf_SyntaxCheck || ActiveWin== &StubWindow ){
		return(1);
	}
	else{
		return( ascanf_ValCat_any( args, result, ActiveWin->ValCat_X, "X", "ascanf_ValCat_X" ) );
	}
}

int ascanf_ValCat_Y ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !ActiveWin || ascanf_SyntaxCheck || ActiveWin== &StubWindow ){
		return(1);
	}
	else{
		return( ascanf_ValCat_any( args, result, ActiveWin->ValCat_Y, "Y", "ascanf_ValCat_Y" ) );
	}
}

int ascanf_ValCat_I ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !ActiveWin || ascanf_SyntaxCheck || ActiveWin== &StubWindow ){
		return(1);
	}
	else{
		return( ascanf_ValCat_any( args, result, ActiveWin->ValCat_I, "I", "ascanf_ValCat_I" ) );
	}
}

int ascanf_radix ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( ascanf_arguments && ActiveWin->radix!= args[0] && !ascanf_SyntaxCheck ){
			*result= ActiveWin->radix= args[0];
			Gonio_Base( ActiveWin, ActiveWin->radix, ActiveWin->radix_offset );
			if( ascanf_arguments> 1 && args[1] ){
				ActiveWin->redraw= 1;
				XG_XSync( disp, False );
			}
		}
		else{
			*result= ActiveWin->radix;
		}
	}
	return(1);
}

int ascanf_radix_offset ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( ascanf_arguments && ActiveWin->radix_offset!= args[0] && !ascanf_SyntaxCheck ){
			*result= ActiveWin->radix_offset= args[0];
			Gonio_Base( ActiveWin, ActiveWin->radix, ActiveWin->radix_offset );
			if( ascanf_arguments> 1 && args[1] ){
				ActiveWin->redraw= 1;
				XG_XSync( disp, False );
			}
		}
		else{
			*result= ActiveWin->radix_offset;
		}
	}
	return(1);
}

int ascanf_xmin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_X[(int)args[0]].min;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}


int ascanf_xmax ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_X[(int)args[0]].max;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

int ascanf_ymin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_Y[(int)args[0]].min;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

int ascanf_ymax ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_Y[(int)args[0]].max;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

int ascanf_errmin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= (ActiveWin->vectorFlag)? ActiveWin->set_O[(int)args[0]].min : ActiveWin->set_E[(int)args[0]].min;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

int ascanf_errmax ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= (ActiveWin->vectorFlag)? ActiveWin->set_O[(int)args[0]].max : ActiveWin->set_E[(int)args[0]].max;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

int ascanf_tr_xmin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_tr_X[(int)args[0]].min;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}


int ascanf_tr_xmax ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_tr_X[(int)args[0]].max;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

int ascanf_tr_ymin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_tr_Y[(int)args[0]].min;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}


int ascanf_tr_ymax ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= ActiveWin->set_tr_Y[(int)args[0]].max;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

int ascanf_tr_errmin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= (ActiveWin->vectorFlag)? ActiveWin->set_tr_O[(int)args[0]].min : ActiveWin->set_tr_E[(int)args[0]].min;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}


int ascanf_tr_errmax ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( !args){
			ascanf_arg_error= 1;
			return( 1 );
		}
		else if( ascanf_arguments>= 1 ){
			if( args[0]>= 0 && args[0]< MaxSets ){
				*result= (ActiveWin->vectorFlag)? ActiveWin->set_tr_O[(int)args[0]].max : ActiveWin->set_tr_E[(int)args[0]].max;
			}
			else{
				*result= 0;
			}
			return( 1 );
		}
		else{
			ascanf_arg_error= 1;
			return( 1 );
		}
	}
	else{
		*result= 0;
		return(1);
	}
}

/* 20030404: determine the quadrant a given point (x,y) is in, and
 \ also whether it is on one of the axes.
 */
#ifdef __GNUC__
inline
#endif
int Quadrant( double x, double y, int *on_axis )
{ int R, ax;
	if( !on_axis ){
		on_axis= &ax;
	}
	*on_axis= 0;
	if( x== 0 && y== 0 ){
		R= 0;
	}
	else if( y>= 0 && x> 0 ){
		R= 1;
		if( y== 0 ){
			*on_axis= 1;
		}
	}
	else if( x<= 0 && y> 0 ){
		R= 2;
		if( x== 0 ){
			*on_axis= 2;
		}
	}
	else if( y<= 0 && x< 0 ){
		R= 3;
		if( y== 0 ){
			*on_axis= 3;
		}
	}
	else if( x>= 0 && y< 0 ){
		R= 4;
		if( x== 0 ){
			*on_axis= 4;
		}
	}
	return(R);
}

/* 20030404: rotate a point (*x,*y) over the angle defined by (s,c), updating *x and *y */
#ifdef __GNUC__
inline
#endif
void rotate_sincos( double *x, double*y, double s, double c )
{ double ry= *y * c + *x * s, 
		rx= *x * c - *y * s;
	*x= rx, *y= ry;
}

/* 20030404: rotate a point (*x,*y) over <angle>, updating *x and *y */
#ifdef __GNUC__
inline
#endif
void rotate_angle( double *x, double*y, double angle )
{ double s, c;
	SinCos( angle, &s, &c );
	rotate_sincos( x, y, s, c );
}

double _CurveLen(
	LocalWin *wi, int idx, DataSet *set, double *X, double *Y, int Points, int End, int pnr,
	double *clen, int update, int Signed, double *lengths )
{ double R;
  int urgent;
	if( wi && set && update && X== set->xvec && set->processing_wi== wi && wi->processing_set== set ){
		urgent= True;
	}
	if( (update && (Signed || !set || (set && set->last_processed_wi!= wi))) || urgent ){
	  int i, disc, pQ= 0;
	  double cl= 0, x, y, px, py, pgx, pgy;
	  static double gravX, gravY;
		px= X[0], py= Y[0];
		clen[0]= 0;
		R= 0;
		if( lengths ){
			lengths[0]= 0;
		}
		if( Signed== 3 && End== Points ){
		  double sx= 0, sy= 0;
		  unsigned long N= 0;
			for( i= 0; i< End; i++ ){
				if( !(disc= DiscardedPoint(wi, set, i)) ||
					(disc< 0 && *curvelen_with_discarded== 1) ||
					(disc> 0 && *curvelen_with_discarded== 2)
				){
					sx+= X[i], sy+= Y[i];
					N+= 1;
				}
			}
			gravX= sx/N, gravY= sy/N;
			pgx= px- gravX, pgy= py- gravY;
		}
		for( i= 1; i< End; i++ ){
			if( !(disc= DiscardedPoint(wi, set, i)) ||
				(disc< 0 && *curvelen_with_discarded== 1) ||
				(disc> 0 && *curvelen_with_discarded== 2)
			){
			  double dx= (x= X[i])-px, dy= (y= Y[i])-py;
			  double len= sqrt( dx*dx + dy*dy );
				if( !NaN(len) ){
					cl+= len;
					if( Signed ){
						if( (Signed==1 && dx<0) || (Signed==-1 && dy<0) ){
							R-= len;
							if( lengths ){
								lengths[i]= -len;
							}
						}
						else if( Signed== 2 ){
						  double dorn= atan2( (y*px - x*py), (x*px + y*py) );
							if( dorn< 0 ){
								len*= -1;
							}
							R+= len;
							if( lengths ){
								lengths[i]= len;
							}
						}
						else if( Signed== 3 ){
						  double rx= ((x- gravX)- pgx), ry= ((y- gravY)- pgy);
						  double slen= sqrt( px*px + py*py );
						  int Q, axis;
							  /* rotate over -1 times the angle (=back) to the positive X-axis */
							rotate_sincos( &rx, &ry, -py/slen, px/slen );
							if( pragma_unlikely(ascanf_verbose) ){
								Q= Quadrant(rx,ry,&axis);
								fprintf( StdErr, "(%d->%d (%g,%g)->(%g,%g))-[%g,%g],rot=%g => rx,ry=(%g,%g), pQ=%d, Q=%d.%d",
									i-1, i,
									X[i-1], Y[i-1], X[i], Y[i], 
									gravX, gravY, Atan2(-py, px),
									rx, ry,
									pQ, Q, axis
								);
							}
							if( !(Q= Quadrant( rx, ry, &axis )) ){
								Q= pQ;
							}
							else if( axis== 1 ){
								if( !(Q= pQ) ){
									Q= 4;
								}
							}
							else if( axis== 3 ){
								if( !(Q= pQ) ){
									Q= 2;
								}
							}
							if( pragma_unlikely(ascanf_verbose) ){
								fprintf( StdErr, "=>Q%d\n", Q );
							}
							switch( Q ){
								case 0:
									fprintf( StdErr,
										"CurveLen(#%d[%d]): singularity: tie or point (%s,%s)=>(%s,%s) is/in centroid %s,%s\n",
										(set)? set->set_nr : -1, i, d2str(x,0,0), d2str(y,0,0),
										d2str(rx,0,0), d2str(ry,0,0),
										d2str(gravX,0,0), d2str(gravY,0,0)
									);
								case 3:
								case 4:
									len*= -1;
									break;
							}
							R+= len;
							if( lengths ){
								lengths[i]= len;
							}
							pQ= Q;
							pgx= x- gravX, pgy= y- gravY;
						}
						else{
							R+= len;
							if( lengths ){
								lengths[i]= len;
							}
						}
						  /* And make sure the curve_len (cl) var equals R...! */
						cl= R;
					}
					else if( lengths ){
						lengths[i]= len;
					}
					clen[i]= cl;
				}
				px= x, py= y;
			}
		}
		if( End== Points ){
			clen[i]= cl;
		}
		if( !Signed ){
			R= clen[pnr];
		}
	}
	else if( clen ){
		R= clen[pnr];
	}
	return( R );
}

  /* Interface to curve_len for modules that don't know LocalWin.
   \ !! Assumes valid arguments!!
   \ 20030329: only calculates upto the requested point...
   \ 20030403: can do a "signed" calculation: segment length will be negative if dx<0 and/or dy<0
   \           when update==True.
   \ 20030404: more solutions to determine segment length (= rotation direction); see the expl. in
   \           SHelp[curve_len] .
   */
double CurveLen( LocalWin *wi, int idx, int pnr, int update, int Signed, double *lengths )
{
#if 1
  DataSet *this_set= &AllSets[idx];
	return( _CurveLen( wi, idx, this_set, this_set->xval, this_set->yval, this_set->numPoints, MIN(pnr+1, this_set->numPoints),
		pnr, wi->curve_len[idx], update, Signed, lengths )
	);
#else
  double R;
	if( update && (Signed || AllSets[idx].last_processed_wi!= wi) ){
	  int i, disc, end= MIN(pnr+1, AllSets[idx].numPoints), pQ= 0;
	  double cl= 0, x, y, px, py, pgx, pgy;
	  static double gravX, gravY;
	  DataSet *set= &AllSets[idx];
		px= XVAL(set,0), py= YVAL(set,0);
		wi->curve_len[idx][0]= 0;
		R= 0;
		if( lengths ){
			lengths[0]= 0;
		}
		if( Signed== 3 && end== AllSets[idx].numPoints ){
		  double sx= 0, sy= 0;
		  unsigned long N= 0;
			for( i= 0; i< end; i++ ){
				if( !(disc= DiscardedPoint(wi, set, i)) ||
					(disc< 0 && *curvelen_with_discarded== 1) ||
					(disc> 0 && *curvelen_with_discarded== 2)
				){
					sx+= XVAL(set,i), sy+= YVAL(set,i);
					N+= 1;
				}
			}
			gravX= sx/N, gravY= sy/N;
			pgx= px- gravX, pgy= py- gravY;
		}
		for( i= 1; i< end; i++ ){
			if( !(disc= DiscardedPoint(wi, set, i)) ||
				(disc< 0 && *curvelen_with_discarded== 1) ||
				(disc> 0 && *curvelen_with_discarded== 2)
			){
			  double dx= (x= XVAL(set,i))-px, dy= (y= YVAL(set,i))-py;
			  double len= sqrt( dx*dx + dy*dy );
				cl+= len;
				if( Signed ){
					if( (Signed==1 && dx<0) || (Signed==-1 && dy<0) ){
						R-= len;
						if( lengths ){
							lengths[i]= -len;
						}
					}
					else if( Signed== 2 ){
					  double dorn= atan2( (y*px - x*py), (x*px + y*py) );
						if( dorn< 0 ){
							len*= -1;
						}
						R+= len;
						if( lengths ){
							lengths[i]= len;
						}
					}
					else if( Signed== 3 ){
					  double rx= ((x- gravX)- pgx), ry= ((y- gravY)- pgy);
					  double slen= sqrt( px*px + py*py );
					  int Q, axis;
						  /* rotate over -1 times the angle (=back) to the positive X-axis */
						rotate_sincos( &rx, &ry, -py/slen, px/slen );
						if( pragma_unlikely(ascanf_verbose) ){
							Q= Quadrant(rx,ry,&axis);
							fprintf( StdErr, "(%d->%d (%g,%g)->(%g,%g))-[%g,%g],rot=%g => rx,ry=(%g,%g), pQ=%d, Q=%d.%d",
								i-1, i,
								XVAL(set,i-1), YVAL(set,i-1), XVAL(set,i), YVAL(set,i), 
								gravX, gravY, Atan2(-py, px),
								rx, ry,
								pQ, Q, axis
							);
						}
						if( !(Q= Quadrant( rx, ry, &axis )) ){
							Q= pQ;
						}
						else if( axis== 1 ){
							if( !(Q= pQ) ){
								Q= 4;
							}
						}
						else if( axis== 3 ){
							if( !(Q= pQ) ){
								Q= 2;
							}
						}
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "=>Q%d\n", Q );
						}
						switch( Q ){
							case 0:
								fprintf( StdErr,
									"CurveLen(#%d[%d]): singularity: tie or point (%s,%s)=>(%s,%s) is/in centroid %s,%s\n",
									(set)? set->set_nr : -1, i, d2str(x,0,0), d2str(y,0,0),
									d2str(rx,0,0), d2str(ry,0,0),
									d2str(gravX,0,0), d2str(gravY,0,0)
								);
							case 3:
							case 4:
								len*= -1;
								break;
						}
						R+= len;
						if( lengths ){
							lengths[i]= len;
						}
						pQ= Q;
						pgx= x- gravX, pgy= y- gravY;
					}
					else{
						R+= len;
						if( lengths ){
							lengths[i]= len;
						}
					}
					  /* And make sure the curve_len (cl) var equals R...! */
					cl= R;
				}
				else if( lengths ){
					lengths[i]= len;
				}
				wi->curve_len[idx][i]= cl;
				px= x, py= y;
			}
		}
		if( end== AllSets[idx].numPoints ){
			wi->curve_len[idx][i]= cl;
		}
		if( !Signed ){
			R= wi->curve_len[idx][pnr];
		}
	}
	else{
		R= wi->curve_len[idx][pnr];
	}
	return( R );
#endif
}

#ifdef TR_CURVE_LEN
  /* Interface to tr_curve_len for modules that don't know LocalWin.
   \ !! Assumes valid arguments!!
   \ 20030329: only calculates upto the requested point...
   */
double tr_CurveLen( LocalWin *wi, int idx, int pnr, int update, int Signed, double *lengths )
{
#if 1
  DataSet *this_set= &AllSets[idx];
	return( _CurveLen( wi, idx, this_set, this_set->xvec, this_set->yvec, this_set->numPoints, MIN(pnr+1, this_set->numPoints),
		pnr, wi->tr_curve_len[idx], update, Signed, lengths )
	);
#else
 double R;
	if( update && (Signed || AllSets[idx].last_processed_wi!= wi) ){
	  int i, disc, end= MIN(pnr+1, AllSets[idx].numPoints), pQ;
	  double cl= 0, x, y, px, py, pgx, pgy;
	  static double gravX, gravY;
	  DataSet *set= &AllSets[idx];
		px= set->xvec[0], py= set->yvec[0];
		wi->tr_curve_len[idx][0]= 0;
		R= 0;
		if( lengths ){
			lengths[0]= 0;
		}
		if( Signed== 3 && end== AllSets[idx].numPoints ){
		  double sx= 0, sy= 0;
		  unsigned long N= 0;
			for( i= 0; i< end; i++ ){
				if( !(disc= DiscardedPoint(wi, set, i)) ||
					(disc< 0 && *curvelen_with_discarded== 1) ||
					(disc> 0 && *curvelen_with_discarded== 2)
				){
					sx+= set->xvec[i], sy+= set->yvec[i];
					N+= 1;
				}
			}
			gravX= sx/N, gravY= sy/N;
			pgx= px- gravX, pgy= py- gravY;
		}
		for( i= 1; i< end; i++ ){
			if( !(disc= DiscardedPoint(wi, set, i)) ||
				(disc< 0 && *curvelen_with_discarded== 1) ||
				(disc> 0 && *curvelen_with_discarded== 2)
			){
			  double dx= (x= set->xvec[i])-px, dy= (y= set->yvec[i])-py;
			  double len= sqrt( dx*dx + dy*dy );
				cl+= len;
				if( Signed ){
					if( (Signed==1 && dx<0) || (Signed==-1 && dy<0) ){
						R-= len;
						if( lengths ){
							lengths[i]= -len;
						}
					}
					else if( Signed== 2 ){
					  double dorn= atan2( (x*px + y*py), (y*px - x*py) );
						if( dorn< 0 ){
							len*= -1;
						}
						R+= len;
						if( lengths ){
							lengths[i]= len;
						}
					}
					else if( Signed== 3 ){
					  double rx= ((x- gravX)- pgx), ry= ((y- gravY)- pgy);
					  double slen= sqrt( px*px + py*py );
					  int Q, axis;
						  /* rotate over -1 times the angle (=back) to the positive X-axis */
						rotate_sincos( &rx, &ry, -py/slen, px/slen );
						if( pragma_unlikely(ascanf_verbose) ){
							Q= Quadrant(rx,ry,&axis);
							fprintf( StdErr, "(%d->%d (%g,%g)->(%g,%g))-[%g,%g],rot=%g => rx,ry=(%g,%g), pQ=%d, Q=%d.%d",
								i-1, i,
								set->xvec[i-1], set->yvec[i-1], set->xvec[i], set->yvec[i], 
								gravX, gravY, Atan2(-py, px),
								rx, ry,
								pQ, Q, axis
							);
						}
						if( !(Q= Quadrant( rx, ry, &axis )) ){
							Q= pQ;
						}
						else if( axis== 1 ){
							if( !(Q= pQ) ){
								Q= 4;
							}
						}
						else if( axis== 3 ){
							if( !(Q= pQ) ){
								Q= 2;
							}
						}
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, "=>Q%d\n", Q );
						}
						switch( Q ){
							case 0:
								fprintf( StdErr,
									"tr_CurveLen(#%d[%d]): singularity: tie or point (%s,%s)=>(%s,%s) is/in centroid %s,%s\n",
									set->set_nr, i, d2str(x,0,0), d2str(y,0,0),
									d2str(rx,0,0), d2str(ry,0,0),
									d2str(gravX,0,0), d2str(gravY,0,0)
								);
							case 3:
							case 4:
								len*= -1;
								break;
						}
						R+= len;
						if( lengths ){
							lengths[i]= len;
						}
						pQ= Q;
						pgx= x- gravX, pgy= y- gravY;
					}
					else{
						R+= len;
						if( lengths ){
							lengths[i]= len;
						}
					}
					cl= R;
				}
				else if( lengths ){
					lengths[i]= len;
				}
				wi->tr_curve_len[idx][i]= cl;
				px= x, py= y;
			}
		}
		if( end== AllSets[idx].numPoints ){
			wi->curve_len[idx][i]= cl;
		}
		if( !Signed ){
			R= wi->tr_curve_len[idx][pnr];
		}
	}
	else{
		R= wi->tr_curve_len[idx][pnr];
	}
	return( R );
#endif
}
#endif

  /* Interface to error_len for modules that don't know LocalWin.
   \ !! Assumes valid arguments!!
   */
double ErrorLen( LocalWin *wi, int idx, int pnr, int update )
{
	if( update && AllSets[idx].last_processed_wi!= wi ){
	  int i, disc, end= MIN(pnr+1, AllSets[idx].numPoints);
	  double el= 0, e, pe;
	  DataSet *set= &AllSets[idx];
		pe= ERROR(set,0);
		wi->error_len[idx][0]= 0;
		for( i= 1; i< end; i++ ){
			if( !(disc= DiscardedPoint(wi, set, i)) ||
				(disc< 0 && *curvelen_with_discarded== 1) ||
				(disc> 0 && *curvelen_with_discarded== 2)
			){
				e= ERROR(set,i);
				el+= fabs( e - pe );
				wi->error_len[idx][i]= el;
				pe= e;
			}
		}
		wi->error_len[idx][i]= el;
	}
	return( wi->error_len[idx][pnr] );
}

int ascanf_curve_len ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( ActiveWin && idx>= 0 && idx< setNumber && ActiveWin!= &StubWindow ){
		if( ascanf_arguments> 1 ){
		  int pnr= (int) args[1], N;
		  int Signed= (ascanf_arguments> 2)? (int) args[2] : False;
		  ascanf_Function *af= NULL, *laf= NULL;
			if( pnr== -1 ){
				pnr= AllSets[idx].numPoints-1;
			}
			N= MIN( pnr+1, AllSets[idx].numPoints );
			if( ascanf_arguments> 3 ){
				if( (af= parse_ascanf_address(args[3], _ascanf_array, "ascanf_curve_len", ascanf_verbose, NULL)) ){
					Resize_ascanf_Array( af, N, result );
					if( af->N!= N ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (can't resize %s (%s)) ", af->name, serror() );
							fflush( StdErr );
						}
						af= NULL;
					}
				}
			}
			if( ascanf_arguments> 4 ){
				if( (!(laf= parse_ascanf_address(args[4], _ascanf_array, "ascanf_curve_len", False, NULL)) || !laf->array) ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (ignoring non-double-array 5th argument) " );
						fflush( StdErr );
					}
				}
				else{
					Resize_ascanf_Array( laf, N, result );
					if( laf->N!= N ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (can't resize %s (%s)) ", laf->name, serror() );
							fflush( StdErr );
						}
						laf= NULL;
					}
				}
			}
			if( pnr>= 0 && pnr< AllSets[idx].numPoints ){
				if( af || laf ){
				  int i;
					CurveLen( ActiveWin, idx, N, True, Signed, (laf)? laf->array : NULL );
					if( af ){
						for( i= 0; i< N; i++ ){
							if( af->iarray ){
								af->value= af->iarray[i]= CurveLen( ActiveWin, idx, i, False, Signed, NULL );
							}
							else{
								af->value= af->array[i]= CurveLen( ActiveWin, idx, i, False, Signed, NULL );
							}
						}
						af->last_index= N-1;
					}
					if( laf ){
						laf->value= laf->array[ (laf->last_index= N-1) ];
					}
					*result= CurveLen( ActiveWin, idx, pnr, False, Signed, NULL);
				}
				else{
					*result= CurveLen( ActiveWin, idx, pnr, True, Signed, NULL);
				}
			}
			else{
				*result= 0;
				fprintf( StdErr, "curve_len[%d]: i=%d not in [0,%d>\n", idx, pnr, AllSets[idx].numPoints );
			}
		}
		else if( AllSets[idx].numPoints> 0 ){
			*result= CurveLen( ActiveWin, idx, AllSets[idx].numPoints, True, False, NULL);
		}
	}
	else{
		*result= 0;
		if( ActiveWin && ActiveWin!= &StubWindow ){
			fprintf( StdErr, "curve_len[%d]: invalid setNumber\n", idx );
		}
	}
	return( 1 );
}

int ascanf_error_len ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( ActiveWin && idx>= 0 && idx< setNumber && ActiveWin!= &StubWindow ){
		if( ascanf_arguments> 1 ){
		  int pnr= (int) args[1];
			if( pnr>= 0 && pnr< AllSets[idx].numPoints ){
				*result= ActiveWin->error_len[idx][pnr];
			}
			else{
				*result= 0;
				fprintf( StdErr, "error_len[%d]: i=%d not in [0,%d>\n", idx, pnr, AllSets[idx].numPoints );
			}
		}
		else if( AllSets[idx].numPoints> 0 ){
			*result= ActiveWin->error_len[idx][AllSets[idx].numPoints];
		}
	}
	else{
		*result= 0;
		if( ActiveWin && ActiveWin!= &StubWindow ){
			fprintf( StdErr, "error_len[%d]: invalid setNumber\n", idx );
		}
	}
	return( 1 );
}

int ascanf_tr_curve_len ( ASCB_ARGLIST )
{
#ifdef TR_CURVE_LEN
 ASCB_FRAME_SHORT
 int idx;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( pragma_unlikely(ascanf_verbose) ){
		if( !Determine_tr_curve_len ){
			fprintf( StdErr, " (tr_curve_len option has not been activated with -tcl) " );
		}
	}
	if( ActiveWin && idx>= 0 && idx< setNumber && ActiveWin!= &StubWindow ){
		if( ascanf_arguments> 1 ){
		  int pnr= (int) args[1], N;
		  int Signed= (ascanf_arguments> 2)? (int) args[2] : False;
		  ascanf_Function *af= NULL, *laf= NULL;
			if( pnr== -1 ){
				pnr= AllSets[idx].numPoints-1;
			}
			N= MIN( pnr+1, AllSets[idx].numPoints );
			if( ascanf_arguments> 3 ){
				if( (af= parse_ascanf_address(args[3], _ascanf_array, "ascanf_tr_curve_len", ascanf_verbose, NULL)) ){
					Resize_ascanf_Array( af, N, result );
					if( af->N!= N ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (can't resize %s (%s)) ", af->name, serror() );
							fflush( StdErr );
						}
						af= NULL;
					}
				}
			}
			if( ascanf_arguments> 4 ){
				if( !(laf= parse_ascanf_address(args[4], _ascanf_array, "ascanf_tr_curve_len", False, NULL)) || !laf->array ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (ignoring non-double-array 5th argument) " );
						fflush( StdErr );
					}
				}
				else{
					Resize_ascanf_Array( laf, N, result );
					if( laf->N!= N ){
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (can't resize %s (%s)) ", laf->name, serror() );
							fflush( StdErr );
						}
						laf= NULL;
					}
				}
			}
			if( pnr>= 0 && pnr< AllSets[idx].numPoints ){
				if( af || laf ){
				  int i;
					tr_CurveLen( ActiveWin, idx, N, True, Signed, (laf)? laf->array : NULL );
					if( af ){
						for( i= 0; i< N; i++ ){
							if( af->iarray ){
								af->value= af->iarray[i]= tr_CurveLen( ActiveWin, idx, i, False, Signed, NULL );
							}
							else{
								af->value= af->array[i]= tr_CurveLen( ActiveWin, idx, i, False, Signed, NULL );
							}
						}
						af->last_index= N-1;
					}
					if( laf ){
						laf->value= laf->array[ (laf->last_index= N-1) ];
					}
					*result= tr_CurveLen( ActiveWin, idx, pnr, False, Signed, NULL);
				}
				else{
					*result= tr_CurveLen( ActiveWin, idx, pnr, True, Signed, NULL );
				}
			}
			else{
				*result= 0;
			}
		}
		else if( AllSets[idx].numPoints> 0 ){
/* 			*result= ActiveWin->tr_curve_len[idx][AllSets[idx].numPoints];	*/
			*result= tr_CurveLen( ActiveWin, idx, AllSets[idx].numPoints, True, False, NULL );
		}
	}
	else{
		*result= 0;
	}
	return( 1 );
#else
 ASCB_FRAME_RESULT
	fprintf( StdErr, "tr_curve_len: not available - compile with -DTR_CURVE_LEN!\n" );
	*result= 0;
	return(1);
#endif
}

int ascanf_curve_len_arrays ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	ascanf_arg_error= 0;
	if( ascanf_arguments> 1 ){
	  int pnr= -1, N;
	  int Signed= False;
	  ascanf_Function *xvals, *yvals, *af= NULL, *laf= NULL;
		if( !(xvals= parse_ascanf_address(args[0], _ascanf_array, "ascanf_curve_len_arrays", ascanf_verbose, NULL))
			|| !(yvals= parse_ascanf_address(args[1], _ascanf_array, "ascanf_curve_len_arrays", ascanf_verbose, NULL))
		){
			ascanf_arg_error= True;
			goto bail_out;
		}
		else if( xvals->N != yvals->N || !xvals->array || !yvals->array ){
			ascanf_emsg= " (co-ordinate arrays must be of doubles and same size)== ";
			ascanf_arg_error= 1;
			goto bail_out;
		}
		if( ascanf_arguments> 2 ){
			pnr= (int) args[2];
		}
		if( ascanf_arguments> 3 ){
			Signed= (int) args[2];
		}
		if( pnr== -1 ){
			pnr= xvals->N- 1;
		}
		N= MIN( pnr+1, xvals->N );
		if( ascanf_arguments> 4 ){
			if( (af= parse_ascanf_address(args[4], _ascanf_array, "ascanf_curve_len_arrays", ascanf_verbose, NULL)) ){
				Resize_ascanf_Array( af, N, result );
				if( af->N!= N ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (can't resize %s (%s)) ", af->name, serror() );
						fflush( StdErr );
					}
					af= NULL;
				}
			}
		}
		if( ascanf_arguments> 5 ){
			if( (!(laf= parse_ascanf_address(args[5], _ascanf_array, "ascanf_curve_len_arrays", False, NULL)) || !laf->array) ){
				if( pragma_unlikely(ascanf_verbose) ){
					fprintf( StdErr, " (ignoring non-double-array 6th argument) " );
					fflush( StdErr );
				}
			}
			else{
				Resize_ascanf_Array( laf, N, result );
				if( laf->N!= N ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (can't resize %s (%s)) ", laf->name, serror() );
						fflush( StdErr );
					}
					laf= NULL;
				}
			}
		}
		if( pnr>= 0 && pnr< xvals->N ){
		  double *clen;
#ifdef DEBUG
			clen= (double*) calloc( N+1, sizeof(double) );
#else
			clen= (double*) malloc( (N+1) * sizeof(double) );
#endif
			if( !clen ){
				goto bail_out;
			}
			if( af || laf ){
			  int i;
				_CurveLen( NULL, 0, NULL, xvals->array, yvals->array, xvals->N, N, N, clen, True, Signed,
					(laf)? laf->array : NULL );
				if( af ){
					for( i= 0; i< N; i++ ){
						if( af->iarray ){
							af->value= af->iarray[i]= clen[i]; /* CurveLen( ActiveWin, idx, i, False, Signed, NULL ); */
						}
						else{
							af->value= af->array[i]= clen[i]; /* CurveLen( ActiveWin, idx, i, False, Signed, NULL ); */
						}
					}
					af->last_index= N-1;
				}
				if( laf ){
					laf->value= laf->array[ (laf->last_index= N-1) ];
				}
				*result= clen[pnr]; /* CurveLen( ActiveWin, idx, pnr, False, Signed, NULL); */
			}
			else{
				*result= _CurveLen( NULL, 0, NULL, xvals->array, yvals->array, xvals->N, N, pnr, clen, True, Signed,
					(laf)? laf->array : NULL );
			}
			xfree(clen);
		}
		else{
			*result= 0;
			fprintf( StdErr, "curve_len_arrays[%s,%s]: i=%d not in [0,%d>\n",
				xvals->name, yvals->name, pnr, xvals->N );
		}
	}
bail_out:;
	return( !ascanf_arg_error );
}

int ascanf_NumPoints ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args){
	  int i;
/* 		*result= (double) maxitems;	*/
		*result= 0;
		  /* 20030929: I don't think user's ascanf code will be very interested in maxitems, but rather in this: */
		for( i= 0; i< setNumber; i++ ){
			if( AllSets[i].numPoints> *result ){
				*result= AllSets[i].numPoints;
			}
		}
		return( 1 );
	}
	else{
		if( args[0]>= 0 && args[0]< MaxSets ){
		  int idx= (int) args[0];
			if( ascanf_arguments> 1 ){
				if( args[1]>= 0 && args[1]<= MAXINT ){
					if( args[1]!= AllSets[idx].numPoints && !ascanf_SyntaxCheck ){
					  int np= AllSets[idx].numPoints, i, j;
						if( (AllSets[idx].numPoints= (int) args[1]) ){
							realloc_points( &AllSets[idx], (int) args[1], False );
						}
						else{
							  /* 20010901: RJVB: set the skipOnce flag for this set when we set
							   \ numPoints to 0. The *DATA_...* processing chain for the current
							   \ point ($Counter) will be finished, but everything afterwards
							   \ in the data-processing loop (in DrawData()) should not be done
							   \ (especially not setting array elements array[this_set->numPoints-1] ...
							   */
							AllSets[idx].skipOnce= True;
						}
						for( i= np; i< AllSets[idx].numPoints; i++ ){
							for( j= 0; j< AllSets[idx].ncols; j++ ){
								set_NaN( AllSets[idx].columns[j][i] );
							}
							if( ActiveWin && ActiveWin!= &StubWindow ){
								if( ActiveWin->curve_len ){
									ActiveWin->curve_len[idx][i]= ActiveWin->curve_len[idx][np];
								}
								if( ActiveWin->error_len ){
									ActiveWin->error_len[idx][i]= ActiveWin->error_len[idx][np];
								}
							}
						}
						if( AllSets[idx].numPoints> maxitems ){
							maxitems= AllSets[idx].numPoints;
							realloc_Xsegments();
						}
						if( !ActiveWin || !ActiveWin->drawing ){
							RedrawSet( idx, False );
						}
					}
				}
			}
			*result= (double) AllSets[idx].numPoints;
		}
		else{
			if( !ascanf_SyntaxCheck ){
				ascanf_emsg= " (invalid setNumber)== ";
				ascanf_arg_error= True;
			}
			*result= 0;
		}
		return( 1 );
	}
}

int ascanf_pointVisible ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
  int pnr, new_set= False;
  unsigned char new;
	if( args && ascanf_arguments>= 2 ){
		idx= (int) args[0];
		pnr= (int) args[1];
	}
	else{
		idx= -1;
		ascanf_arg_error= 1;
	}
	if( ascanf_SyntaxCheck ){
		return( !ascanf_arg_error );
	}
	if( ActiveWin && idx>= 0 && idx< setNumber && ActiveWin!= &StubWindow ){
	  int N;
	  ascanf_Function *af= NULL;
		if( pnr< 0 ){
			pnr= AllSets[idx].numPoints-1;
		}
		N= AllSets[idx].numPoints;
		if( ascanf_arguments> 2 ){
			if( (af= parse_ascanf_address(args[2], _ascanf_array, "ascanf_pointVisible", ascanf_verbose, NULL)) ){
				Resize_ascanf_Array( af, N, result );
				if( af->N!= N ){
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (can't resize %s (%s)) ", af->name, serror() );
						fflush( StdErr );
					}
					af= NULL;
				}
			}
			else{
				new_set= True;
				new= (unsigned char) ASCANF_TRUE(args[2]);
			}
		}
		if( pnr>= 0 && pnr< AllSets[idx].numPoints ){
		  int i;
			*result= ActiveWin->pointVisible[idx][pnr];
			if( pragma_unlikely(ascanf_verbose) && !*result ){
			  DataSet *this_set= &AllSets[idx];
			  /* try to give feedback why a point is not visible. I myself tend to overlook some reasons, at times */
				if( DiscardedPoint( ActiveWin, this_set, pnr) && !DiscardedShadows ){
					fprintf( StdErr, " (point is discarded) " );
				}
				if( (this_set->plot_interval> 0 && (pnr % this_set->plot_interval)!= 0) ){
					fprintf( StdErr, " (point not drawn due to plot_interval==%d) ", this_set->plot_interval );
				}
			}
			if( af ){
				for( i= 0; i< N; i++ ){
					if( af->iarray ){
						af->value= af->iarray[i]= ActiveWin->pointVisible[idx][i];
					}
					else{
						af->value= af->array[i]= ActiveWin->pointVisible[idx][i];
					}
				}
				af->last_index= N-1;
			}
			else if( new_set ){
				ActiveWin->pointVisible[idx][pnr]= new;
			}
		}
		else{
			*result= 0;
			fprintf( StdErr, "pointVisible[%d]: i=%d not in [0,%d>\n", idx, pnr, AllSets[idx].numPoints );
		}
	}
	else{
		*result= 0;
		if( ActiveWin && ActiveWin!= &StubWindow ){
			fprintf( StdErr, "pointVisible[%d,%d]: invalid setNumber or missing point number\n", idx, pnr );
		}
	}
	return( !ascanf_arg_error );
}

static int findLabelColumn( int set_nr, double colnr, int partial, int withcase, int oldval, char *caller )
{ int ret= -1, take_usage;
  ascanf_Function *col;
	if( !(col= parse_ascanf_address(colnr, _ascanf_variable, caller, (int) ascanf_verbose, &take_usage))
		|| !take_usage
	){
		if( col ){
			ascanf_arg_error= 1;
			ascanf_emsg= " (column specification should be a number or a string!) ";
		}
		CLIP_EXPR( ret, (int) colnr, 0, AllSets[set_nr].ncols-1);
	}
	else{
		if( col->usage ){
			if( AllSets[set_nr].ColumnLabels ){
				ret= Find_LabelsListColumn( AllSets[set_nr].ColumnLabels, col->usage,
					ASCANF_TRUE(partial), ASCANF_TRUE(withcase) );
				if( ascanf_verbose && ret>= 0 ){
					fprintf( StdErr, " (found \"%s\" in set #%d labels, column=%d)", col->usage, set_nr, ret );
				}
			}
			if( ret< 0 && ActiveWin && ActiveWin->ColumnLabels ){
				ret= Find_LabelsListColumn( ActiveWin->ColumnLabels, col->usage,
					ASCANF_TRUE(partial), ASCANF_TRUE(withcase) );
				if( ascanf_verbose && ret>= 0 ){
					fprintf( StdErr, " (found \"%s\" in the window labels, column=%d)", col->usage, ret );
				}
			}
		}
	}
	if( ret< 0 ){
		return(oldval);
	}
	else{
		return(ret);
	}
}

int ascanf_LabelledColumn ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	set_NaN(*result);
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( idx>= 0 && idx< setNumber ){
		if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
			{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
					withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
				*result= findLabelColumn( idx, args[1], partial, withcase, -1, "ascanf_LabelledColumn" );
			}
		}
	}
	return( 1 );
}

int ascanf_xcol ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( idx>= 0 && idx< setNumber ){
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
						withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
					ActiveWin->xcol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->xcol[idx], "ascanf_xcol" );
				}
				if( AllSets[idx].xcol!= ActiveWin->xcol[idx] ){
					AllSets[idx].xcol= ActiveWin->xcol[idx];
					AllSets[idx].init_pass= True;
				}
			}
			*result= (double) ActiveWin->xcol[idx];
		}
		else{
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				for( idx= 0; idx< setNumber; idx++ ){
					{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
							withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
						ActiveWin->xcol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->xcol[idx], "ascanf_xcol" );
					}
					if( AllSets[idx].xcol!= ActiveWin->xcol[idx] ){
						AllSets[idx].xcol= ActiveWin->xcol[idx];
						AllSets[idx].init_pass= True;
					}
				}
				*result= args[1];
			}
			else{
				*result= 0;
			}
		}
	}
	else{
		*result= -1;
		if( idx>= 0 && idx< setNumber && AllSets ){
			*result= (double) AllSets[idx].xcol;
		}
	}
	return( 1 );
}

int ascanf_ycol ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( idx>= 0 && idx< setNumber ){
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
						withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
					ActiveWin->ycol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->ycol[idx], "ascanf_ycol" );
				}
				if( AllSets[idx].ycol!= ActiveWin->ycol[idx] ){
					AllSets[idx].ycol= ActiveWin->ycol[idx];
					AllSets[idx].init_pass= True;
				}
			}
			*result= (double) ActiveWin->ycol[idx];
		}
		else{
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				for( idx= 0; idx< setNumber; idx++ ){
					{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
							withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
						ActiveWin->ycol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->ycol[idx], "ascanf_ycol" );
					}
					if( AllSets[idx].ycol!= ActiveWin->ycol[idx] ){
						AllSets[idx].ycol= ActiveWin->ycol[idx];
						AllSets[idx].init_pass= True;
					}
				}
				*result= args[1];
			}
			else{
				*result= 0;
			}
		}
	}
	else{
		*result= -1;
		if( idx>= 0 && idx< setNumber && AllSets ){
			*result= (double) AllSets[idx].ycol;
		}
	}
	return( 1 );
}

int ascanf_ecol ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( idx>= 0 && idx< setNumber ){
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
						withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
					ActiveWin->ecol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->ecol[idx], "ascanf_ecol" );
				}
				if( AllSets[idx].ecol!= ActiveWin->ecol[idx] ){
					AllSets[idx].ecol= ActiveWin->ecol[idx];
					AllSets[idx].init_pass= True;
				}
			}
			*result= (double) ActiveWin->ecol[idx];
		}
		else{
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				for( idx= 0; idx< setNumber; idx++ ){
					{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
							withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
						ActiveWin->ecol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->ecol[idx], "ascanf_ecol" );
					}
					if( AllSets[idx].ecol!= ActiveWin->ecol[idx] ){
						AllSets[idx].ecol= ActiveWin->ecol[idx];
						AllSets[idx].init_pass= True;
					}
				}
				*result= args[1];
			}
			else{
				*result= 0;
			}
		}
	}
	else{
		*result= -1;
		if( idx>= 0 && idx< setNumber && AllSets ){
			*result= (double) AllSets[idx].ecol;
		}
	}
	return( 1 );
}

int ascanf_lcol ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( idx>= 0 && idx< setNumber ){
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
						withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
					ActiveWin->lcol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->lcol[idx], "ascanf_lcol" );
				}
				if( AllSets[idx].lcol!= ActiveWin->lcol[idx] ){
					AllSets[idx].lcol= ActiveWin->lcol[idx];
					AllSets[idx].init_pass= True;
				}
			}
			*result= (double) ActiveWin->lcol[idx];
		}
		else{
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				for( idx= 0; idx< setNumber; idx++ ){
					{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
							withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
						ActiveWin->lcol[idx]= findLabelColumn( idx, args[1], partial, withcase, ActiveWin->lcol[idx], "ascanf_lcol" );
					}
					if( AllSets[idx].lcol!= ActiveWin->lcol[idx] ){
						AllSets[idx].lcol= ActiveWin->lcol[idx];
						AllSets[idx].init_pass= True;
					}
				}
				*result= args[1];
			}
			else{
				*result= 0;
			}
		}
	}
	else{
		*result= -1;
		if( idx>= 0 && idx< setNumber && AllSets ){
			*result= (double) AllSets[idx].lcol;
		}
	}
	return( 1 );
}

int ascanf_Ncol ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, nc;
	if( args && ascanf_arguments ){
		idx= (int) args[0];
	}
	else{
		idx= (int) *ascanf_setNumber;
	}
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( idx>= 0 && idx< setNumber ){
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
						withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
					nc= findLabelColumn( idx, args[1], partial, withcase, nc, "ascanf_Ncol" );
				}
				if( AllSets[idx].Ncol!= nc ){
					AllSets[idx].Ncol= nc;
  /* Even if we don't support "advanced" stats, keep the Ncol field updated:	*/
#if ADVANCED_STATS == 2
					AllSets[idx].init_pass= True;
#endif
				}
			}
#if ADVANCED_STATS == 2
			*result= (double) AllSets[idx].Ncol;
#else
			*result= (double) AllSets[idx].NumObs;
#endif
		}
		else{
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
				for( idx= 0; idx< setNumber; idx++ ){
					{ int partial= (ascanf_arguments>=3)? ASCANF_TRUE(args[2]) : 0,
							withcase= (ascanf_arguments>=4)? ASCANF_TRUE(args[3]) : 0;
						nc= findLabelColumn( idx, args[1], partial, withcase, nc, "ascanf_Ncol" );
					}
					if( AllSets[idx].Ncol!= nc ){
						AllSets[idx].Ncol= nc;
#if ADVANCED_STATS == 2
						AllSets[idx].init_pass= True;
#endif
					}
				}
				*result= args[1];
			}
			else{
				*result= 0;
			}
		}
	}
	else{
		*result= -1;
		if( idx>= 0 && idx< setNumber && AllSets ){
			*result= (double) AllSets[idx].Ncol;
		}
	}
	return( 1 );
}

int ascanf_CursorCross ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  extern int CursorCross, CursorCross_Labeled;
	if( ascanf_arguments> 0 && !ascanf_SyntaxCheck ){
	  LocalWindows *WL= WindowList;
		CursorCross= (args[0])? True : False;
		CLIP_EXPR( CursorCross_Labeled, (int) args[0], 0, 3 );
		while( WL ){
			SelectXInput( WL->wi );
			WL= WL->next;
		}
	}
	*result= (CursorCross)? CursorCross_Labeled : 0;
	return( 1 );
}

/* accessing linestyle, lineWidth, elinestyle, elineWidth, Colour[set,`af], markStyle, markSize	*/

int ascanf_SetLineStyle ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, val;
	if( ascanf_arguments>= 1 &&
		args[0]>= 0 && args[0]< setNumber
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			idx= (int)args[0];
			*result= AllSets[idx].linestyle;
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				CLIP_EXPR_CAST( int, val, double, args[1], 0, MAXINT );
				AllSets[idx].linestyle= val;
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetLineWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= AllSets[idx].lineWidth;
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				AllSets[idx].lineWidth= args[1];
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetELineStyle ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, val;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= AllSets[idx].elinestyle;
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				CLIP_EXPR_CAST( int, val, double, args[1], 0, MAXINT );
				AllSets[idx].elinestyle= val;
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetELineWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= AllSets[idx].elineWidth;
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				AllSets[idx].elineWidth= args[1];
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetBarWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( AllSets[idx].barWidth_set ){
				*result= AllSets[idx].barWidth;
			}
			else{
				set_NaN( *result );
			}
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				AllSets[idx].barWidth= args[1];
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetEBarWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( AllSets[idx].ebarWidth_set ){
				*result= AllSets[idx].ebarWidth;
			}
			else{
				set_NaN( *result );
			}
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				AllSets[idx].ebarWidth= args[1];
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetCBarWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( !AllSets[idx].current_bW_set && draw_set(ActiveWin, idx) ){
				AllSets[idx].init_pass= True;
				ActiveWin->animate= True;
				Animating= True;
			}
			*result= AllSets[idx].current_barWidth;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetCEBarWidth ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( !AllSets[idx].current_ebW_set && draw_set(ActiveWin, idx) ){
				AllSets[idx].init_pass= True;
				ActiveWin->animate= True;
				Animating= True;
			}
			*result= AllSets[idx].current_ebarWidth;
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetMarkStyle ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx, val;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= ABS(AllSets[idx].markstyle);
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				CLIP_EXPR_CAST( int, val, double, args[1], 1, MAXINT );
				AllSets[idx].markstyle= -val;
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetMarkSize ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
	if( ascanf_arguments>= 1 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			*result= AllSets[idx].markSize;
			if( ascanf_arguments> 1 && !ascanf_SyntaxCheck ){
				AllSets[idx].markSize= args[1];
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetColour ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
  ascanf_Function *raf;
  static ascanf_Function AF= {NULL};
  static char *AFname= "SetColour-Static-StringPointer";

	if( ascanf_arguments>= 2 &&
		((idx= (int)args[0])>= 0 && idx< setNumber)
	){
		if( !(raf= parse_ascanf_address(args[1], _ascanf_variable, "ascanf_SetColour", False, NULL)) ){
			raf= &AF;
			if( raf->name ){
				xfree(raf->usage);
				memset( raf, 0, sizeof(ascanf_Function) );
			}
			else{
				raf->usage= NULL;
			}
			raf->type= _ascanf_variable;
			raf->name= AFname;
		}
		raf->is_address= raf->take_address= True;
		raf->is_usage= raf->take_usage= True;
		raf->internal= True;
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( AllSets[idx].pixvalue< 0 ){
				  /* Store the colourname in the provided stringpointer, and return args[1],
				   \ which still points to it.
				   */
				xfree( raf->usage );
				raf->usage= XGstrdup( AllSets[idx].pixelCName );
				raf->is_usage= True;
				*result= (raf==&AF)? take_ascanf_address(&AF) : args[1];
			}
			else{
				*result= AllSets[idx].pixvalue;
			}
			if( ascanf_arguments> 2 && !ascanf_SyntaxCheck ){
			  ascanf_Function *naf;
			  char *cname= NULL;
				if( (naf= ascanf_ParseColour( args[2], &cname, True, "ascanf_SetColour")) ){
					if( cname ){
					  Pixel tp;
						if( GetColor( cname, &tp ) ){
							if( AllSets[idx].pixvalue< 0 ){
								FreeColor( &AllSets[idx].pixelValue, &AllSets[idx].pixelCName );
							}
							AllSets[idx].pixelValue= tp;
							AllSets[idx].pixvalue= -1;
							StoreCName( AllSets[idx].pixelCName );
							if( naf && naf->type== _ascanf_array && naf->N== 3 ){
								StoreCName( naf->usage );
							}
							XGStoreColours = 1;
						}
					}
					else{
						return(1);
					}
				}
				else{
				  int v;
					if( args[2]< 0 ){
						ascanf_emsg= " (colour number should be positive) ";
						ascanf_arg_error= True;
						return(1);
					}
					CLIP_EXPR_CAST( int, v, double, args[2], 0, MAXINT );
					AllSets[idx].pixvalue= v;
					XGStoreColours = 1;
				}
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		ascanf_emsg= " (too few arguments or invalid setnumber) ";
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_SetHLColour ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int idx;
  ascanf_Function *raf;
	if( ascanf_arguments>= 2 &&
		((idx= (int)args[0])>= 0 && idx< setNumber) &&
		(raf= parse_ascanf_address(args[1], _ascanf_variable, "ascanf_SetHLColour", False, NULL))
	){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *wi= ActiveWin;
			  /* Store the colourname in the provided stringpointer, and return args[1],
			   \ which still points to it.
			   */
			xfree( raf->usage );
			if( wi->legend_line[idx].pixvalue< 0 ){
				raf->usage= XGstrdup( wi->legend_line[idx].pixelCName );
			}
			else{
				raf->usage= XGstrdup( highlightCName );
			}
			raf->is_usage= True;
			*result= args[1];
			if( ascanf_arguments> 2 && !ascanf_SyntaxCheck ){
			  ascanf_Function *naf;
			  char *cname= NULL;
/* 			  int take_usage;	*/
/* 				if( (naf= parse_ascanf_address(args[2], _ascanf_variable, "ascanf_SetHLColour", False, &take_usage)) && take_usage ){	*/
/* 					cname= naf->usage;	*/
/* 				}	*/
				if( (naf= ascanf_ParseColour( args[2], &cname, True, "ascanf_SetHLColour")) ){
				  /* nothing	*/
				}
				else{
				  int v;
					if( args[2]>= 0 ){
						CLIP_EXPR_CAST( int, v, double, args[2], 0, MAXINT );
						  /* Set highlight colours are either determined per set, or the global
						   \ highlight colour is used. Thus, if we ask for one of the attribute
						   \ colours, that colour's name must be retrieved.
						   */
						cname= AllAttrs[ v % MAXATTR ].pixelCName;
					}
				}
				if( cname ){
				  Pixel tp;
					if( GetColor( cname, &tp ) ){
						FreeColor( &wi->legend_line[idx].pixelValue, &wi->legend_line[idx].pixelCName );
						wi->legend_line[idx].pixelValue= tp;
						wi->legend_line[idx].pixvalue= -1;
						StoreCName( wi->legend_line[idx].pixelCName );
						if( naf && naf->type== _ascanf_array && naf->N== 3 ){
							StoreCName( naf->usage );
						}
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (new set #%d highlight colour %s) ", idx, wi->legend_line[idx].pixelCName );
						}
						XGStoreColours = 1;
					}
				}
				else{
					FreeColor( &wi->legend_line[idx].pixelValue, &wi->legend_line[idx].pixelCName );
					wi->legend_line[idx].pixelValue= 0;
					wi->legend_line[idx].pixvalue= 0;
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (will use global highlight colour %s) ", highlightCName );
					}
					XGStoreColours = 1;
				}
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		ascanf_emsg= " (too few arguments or invalid setnumber) ";
		*result= 0;
		return(1);
	}
	return(1);
}

int ascanf_FitBounds ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int x, y= False;
	if( args && ascanf_arguments>= 1 ){
		x= (args[0])? True: False;
		if( ascanf_arguments> 1 ){
			y= (args[1])? True: False;
		}
		if( !ascanf_SyntaxCheck && ActiveWin && !ActiveWin->fitting && ActiveWin!= &StubWindow ){
		  int ut= ActiveWin->use_transformed, fx= ActiveWin->fit_xbounds,
		  		fy= ActiveWin->fit_ybounds;
			ActiveWin->fitting= fitDo;
			ActiveWin->use_transformed= True;
			ActiveWin->fit_xbounds= False;
			ActiveWin->fit_ybounds= False;
			if( x ){
				if( y ){
					*result= Fit_XYBounds( ActiveWin, False );
				}
				else{
					*result= Fit_XBounds( ActiveWin, False );
				}
			}
			else if( y ){
				*result= Fit_YBounds( ActiveWin, False );
			}
			else{
				*result= 0;
			}
			ActiveWin->fit_xbounds= fx;
			ActiveWin->fit_ybounds= fy;
			ActiveWin->fitting= False;
			ActiveWin->use_transformed= ut;
		}
		else{
			*result= 0;
		}
	}
	else{
		*result= 0;
		ascanf_arg_error= 1;
	}
	return(1);
}

int ascanf_ActiveWinAxisValues ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *values= NULL, *labelled= NULL;
  int which, raw;

	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= True;
		return(0);
	}

	CLIP_EXPR_CAST( int, which, double, args[0], -MAXINT, MAXINT );
	if( which< 0 || which> 4 ){
		ascanf_emsg= " (the <which> argument should be between 0 and 4)== ";
		ascanf_arg_error= 1;
	}
	CLIP_EXPR_CAST( int, raw, double, args[1], -MAXINT, MAXINT );
	if( !(values= parse_ascanf_address( args[2], _ascanf_array, "ascanf_ActiveWinAxisValues", (int) ascanf_verbose, NULL)) ||
		values->name[0]== '%'
	){
		ascanf_emsg= " (3rd argument should point to an array of doubles)== ";
		ascanf_arg_error= 1;
	}
	if( ascanf_arguments> 3 &&
		!(labelled= parse_ascanf_address( args[3], _ascanf_array, "ascanf_ActiveWinAxisValues", (int) ascanf_verbose, NULL))
	){
		ascanf_emsg= " (4th argument should point to an array)== ";
	}

	if( !ascanf_arg_error && ActiveWin && ActiveWin!= &StubWindow && !ascanf_SyntaxCheck ){
	  AxisValues *av= NULL;
		switch( which ){
			case 0:
				av= &ActiveWin->axis_stuff.X;
				break;
			case 1:
				av= &ActiveWin->axis_stuff.Y;
				break;
			case 2:
				av= &ActiveWin->axis_stuff.I;
				break;
			case 3:
				break;
			case 4:
				break;
		}
		if( av && av->array ){
		  int i, n= (av->last_index)? av->last_index : 1;
			Resize_ascanf_Array( values, n, result );
			if( labelled ){
				Resize_ascanf_Array( labelled, n, result );
			}
			for( i= 0; i< av->last_index; i++ ){
				if( i< values->N ){
					values->array[i]= av->array[i];
				}
				if( labelled && i< labelled->N ){
					if( labelled->iarray ){
						labelled->iarray[i]= (int) av->labelled[i];
					}
					else{
						labelled->array[i]= (double) av->labelled[i];
					}
				}
			}
			if( i== 0 ){
				set_NaN(values->array[0]);
				*result= 0;
			}
		}
		else{
			Resize_ascanf_Array( values, 1, result );
			if( labelled ){
				Resize_ascanf_Array( labelled, 1, result );
			}
			set_NaN(values->array[0]);
			*result= 0;
		}
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_DataWin ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( args && ascanf_arguments>= 1 ){
		if( !ascanf_SyntaxCheck && ActiveWin && !ActiveWin->fitting && ActiveWin!= &StubWindow ){
			if( ActiveWin->datawin.apply ){
			  /* make sure that repeating apply==True won't modify old_user_coordinates: */
				ActiveWin->win_geo.user_coordinates= ActiveWin->datawin.old_user_coordinates;
			}
			if( (*result= ActiveWin->datawin.apply= (args[0])? True : False) ){
				ActiveWin->datawin.old_user_coordinates= ActiveWin->win_geo.user_coordinates;
				ActiveWin->win_geo.user_coordinates= (args[0]<0)? False : True;
			}
			else{
				if( ActiveWin->datawin.old_user_coordinates< 0 ){
					ActiveWin->win_geo.user_coordinates= False;
				}
				else if( ActiveWin->datawin.old_user_coordinates> 0 ){
					ActiveWin->win_geo.user_coordinates= True;
				}
			}
			if( ascanf_arguments>= 3 ){
				if( args[1] ){
					ActiveWin->datawin.llX= args[2];
				}
			}
			if( ascanf_arguments>= 5 ){
				if( args[3] ){
					ActiveWin->datawin.llY= args[4];
				}
			}
			if( ascanf_arguments>= 7 ){
				if( args[5] ){
					ActiveWin->datawin.urX= args[6];
				}
			}
			if( ascanf_arguments>= 9 ){
				if( args[7] ){
					ActiveWin->datawin.urY= args[8];
				}
			}
		}
		else{
			*result= 0;
		}
	}
	else{
		*result= 0;
		ascanf_arg_error= 1;
	}
	return(1);
}

int ascanf_DataWinScroll ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( args && ascanf_arguments>= 2 ){
		if( !ascanf_SyntaxCheck && ActiveWin && !ActiveWin->fitting && ActiveWin!= &StubWindow ){
			if( !NaN(args[0]) ){
				ActiveWin->datawin.llX+= args[0];
				ActiveWin->datawin.urX+= args[0];
			}
			if( !NaN(args[1]) ){
				ActiveWin->datawin.llY+= args[1];
				ActiveWin->datawin.urY+= args[1];
			}
			if( ascanf_arguments> 2 ){
				*result= ActiveWin->datawin.apply= (args[2])? True : False;
			}
		}
		else{
			*result= 0;
		}
	}
	else{
		*result= 0;
		ascanf_arg_error= 1;
	}
	return(1);
}

int ascanf_titleText ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL, *naf= NULL;
  int take_usage= 0;
  static ascanf_Function AF= {NULL};
  static char *AFname= "titleText-Static-StringPointer";
  int idx= 0;
	af= &AF;
	if( af->name ){
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
	}
	else{
		af->usage= NULL;
	}
	af->type= _ascanf_variable;
	af->name= AFname;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	af->internal= True;
	ascanf_arg_error= False;
	if( args ){
		CLIP_EXPR_CAST( int, idx, double, args[0], 0, 1 );
		if( ascanf_arguments> 1 ){
		  ascanf_Function *f= parse_ascanf_address( args[1], _ascanf_variable, "ascanf_titleText",
		  			(int) ascanf_verbose, &take_usage );
			if( f ){
				af= f;
			}
		}
		if( ascanf_arguments> 2 ){
			if( (naf= parse_ascanf_address( args[2], _ascanf_variable, "ascanf_titleText", (int) ascanf_verbose, &take_usage )) &&
				!take_usage
			){
				ascanf_emsg= " (new titleText argument must be a stringpointer) ";
				ascanf_arg_error= True;
				naf= NULL;
			}
		}
	}
	*result= 0;
	if( af && !ascanf_arg_error ){
	  extern char *titleText2;
		if( !ascanf_SyntaxCheck ){
			if( naf ){
				if( idx ){
					xfree( titleText2 );
					titleText2= (naf->usage)? strdup(naf->usage) : strdup("");
				}
				else{
				  extern int titleTextSet;
					if( naf->usage ){
						strncpy( titleText, naf->usage, MAXBUFSIZE );
						titleTextSet= True;
					}
					else{
						titleText[0]= '\0';
						titleTextSet= False;
					}
				}
			}
			xfree( af->usage );
			af->usage= XGstrdup( (idx)? titleText2 : titleText);
			*result= take_ascanf_address( af );
		}
		else{
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= take_ascanf_address( af );
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_SetOverlap ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	ascanf_arg_error= 0;
	set_NaN(*result);
	if( !args || ascanf_arguments< 5 ){
		ascanf_arg_error= 1;
	}
	else if( args[0]>= 0 && args[0]< setNumber && args[1]>= 0 && args[1]< setNumber ){
	  int idx1= (int) args[0], idx2= (int) args[1];
	  SimpleStats *O1, *O2;
	  ascanf_Function *aO1, *aO2, *afw= NULL;
	  extern SimpleStats *ascanf_SS;
		if( !(aO1= parse_ascanf_address( args[3], _ascanf_simplestats, "ascanf_SetOverlap", (int) ascanf_verbose, NULL )) ){
			if( args[3]< 0 || args[3]> ASCANF_MAX_ARGS ){
				ascanf_emsg= " (4th argument must point to an existing $SS_StatsBin variable, or be a valid StatsBin index!) ";
				ascanf_arg_error= 1;
			}
			else if( Check_ascanf_SS() ){
				O1= &ascanf_SS[ (int) args[3] ];
			}
		}
		else{
			O1= aO1->SS;
		}
		if( !(aO2= parse_ascanf_address( args[4], _ascanf_simplestats, "ascanf_SetOverlap", (int) ascanf_verbose, NULL )) ){
			if( args[4]< 0 || args[4]> ASCANF_MAX_ARGS ){
				ascanf_emsg= " (5th argument must point to an existing $SS_StatsBin variable, or be a valid StatsBin index!) ";
				ascanf_arg_error= 1;
			}
			else if( Check_ascanf_SS() ){
				O2= &ascanf_SS[ (int) args[4] ];
			}
		}
		else{
			O2= aO2->SS;
		}
		if( ascanf_arguments> 5 ){
			afw= parse_ascanf_address( args[5], _ascanf_variable, "ascanf_SetOverlap", (int) ascanf_verbose, NULL );
		}
		if( ActiveWin && !ascanf_SyntaxCheck && !ascanf_arg_error ){
		  double overlap, weight;
		  int ovl= (args[2])? 2 : 1;
			  /* Calculate the overlap, regardless of whether both sets are in vector mode! */
			overlap= Calculate_SetOverlap( ActiveWin, &AllSets[idx1], &AllSets[idx2], O1, O2, &weight, &ovl, False );
			if( overlap>= 0 ){
				*result= overlap;
				if( afw ){
					afw->value= weight;
				}
			}
		}
	}
	else{
		ascanf_emsg= " (setnumber argument(s) out of bounds) ";
		ascanf_arg_error= 1;
	}
	return(!ascanf_arg_error);
}

double ascanf_WaitForEvent_h( int type, char *message, char *caller )
{ double ret= 0;
	if( ActiveWin && ActiveWin!= &StubWindow ){
		if( pragma_unlikely(ascanf_verbose) ){
			fprintf( StdErr, "(window 0x%lx %02d:%02d:%02d)",
				ActiveWin, ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
			);
		}
		ret= WaitForEvent( ActiveWin, type, message, caller );
	}
	return( ret );
}

int ascanf_WaitEvent ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *atype=NULL, *msg=NULL;
  int take_usage, type= 0;

	if( ascanf_arguments ){
		if( (atype= parse_ascanf_address( args[0], 0, "ascanf_WaitEvent", (int) ascanf_verbose, &take_usage )) && take_usage ){
			if( atype->usage ){
				type= -1;
				if( strncasecmp( atype->usage, "key", 3)== 0 ){
					type= KeyPress;
				}
			}
			if( type== -1 ){
				ascanf_emsg= " (unknown or invalid event specification ignored) ";
				ascanf_arg_error= 1;
				type= 0;
			}
		}
		else{
			ascanf_emsg= " (event specification must be a string) ";
			ascanf_arg_error= 1;
		}
		if( ascanf_arguments> 1 ){
			if( !(msg= parse_ascanf_address( args[1], 0, "ascanf_WaitEvent", (int) ascanf_verbose, &take_usage )) || !take_usage ){
				ascanf_emsg= " (second argument must be a message-string) ";
				msg= NULL;
			}
		}
	}
	if( !ascanf_SyntaxCheck ){
		*result= ascanf_WaitForEvent_h( type, (msg)? msg->usage : NULL, "ascanf_WaitEvent()" );
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_Arrays2ValCat ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *cats, *vals= NULL;
  int take_usage, which= 0;

	*result= -1;
	if( ascanf_arguments< 2 ){
		ascanf_arg_error= True;
		return(0);
	}
	ascanf_arg_error= False;
	if( args[0]< 0 || args[0]> 2 ){
		ascanf_emsg= " (1st argument selects the axis and must be 0, 1 or 2) ";
		ascanf_arg_error= 1;
	}
	else{
		which= (int) args[0];
	}
	if( !(cats= parse_ascanf_address( args[1], _ascanf_array, "ascanf_Arrays2ValCat", ascanf_verbose, NULL)) ||
		cats->iarray
	){
		ascanf_emsg= " (2nd argument must be an array with the category labels!) ";
		ascanf_arg_error= True;
	}
	if( ascanf_arguments> 2 ){
		if( !(vals= parse_ascanf_address( args[2], _ascanf_array, "ascanf_Arrays2ValCat", ascanf_verbose, NULL)) ){
			ascanf_emsg= " (optional 3rd argument must be an array with the category values!) ";
		}
	}
	if( ascanf_arg_error ){
		return(0);
	}
	if( ActiveWin && !ascanf_SyntaxCheck && ActiveWin!= &StubWindow ){
	  ascanf_Function *string;
	  ValCategory **vcat;
	  int i, N, vn= 0;
		switch( which ){
			case 0:
				vcat= &ActiveWin->ValCat_X;
				break;
			case 1:
				vcat= &ActiveWin->ValCat_Y;
				break;
			case 2:
				vcat= &ActiveWin->ValCat_I;
				break;
		}
		*vcat= Free_ValCat(*vcat);
		if( vals ){
			N= MIN( cats->N, vals->N);
			for( i= 0; i< N; i++ ){
				if( (string= parse_ascanf_address( cats->array[i], 0, "ascanf_Arrays2ValCat", ascanf_verbose, &take_usage)) &&
					take_usage
				){
					*vcat= Add_ValCat( *vcat, &vn, (vals->iarray)? vals->iarray[i] : vals->array[i],
						(string->usage)? string->usage : ""
					);
				}
			}
		}
		else{
			for( i= 0; i< cats->N; i++ ){
				if( (string= parse_ascanf_address( cats->array[i], 0, "ascanf_Arrays2ValCat", ascanf_verbose, &take_usage)) &&
					take_usage
				){
					*vcat= Add_ValCat( *vcat, &vn, cats->array[i], (string->usage)? string->usage : "" );
				}
			}
		}
		*result= vn;
	}
	else{
		*result= 0;
	}
	return(1);
}

int ascanf_eval ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af;
  int r= 0, take_usage;
	set_NaN(*result);
	if( args && ascanf_arguments ){
		if( (af= parse_ascanf_address( args[0], 0, "ascanf_eval", (int) ascanf_verbose, &take_usage )) && take_usage ){
		  int argc= ascanf_arguments, compile;
			if( ascanf_arguments>= 2 ){
				compile= (ASCANF_TRUE(args[1]))? True : False;
				ascanf_arguments-= 2;
			}
			else{
				compile= True;
				ascanf_arguments= 0;
			}
			{ int n= ascanf_arguments;
/* 			  double *lArgList= af_ArgList->array;	*/
			  int /* lArgc= af_ArgList->N, */ auA= ascanf_update_ArgList /*, *level= __ascb_frame->level */;
			  char *fur= fascanf_unparsed_remaining;
#ifdef ASCB_FRAME_EXPRESSION
/* 			  char *expr= __ascb_frame->expr;	*/
#else
/* 			  char lexpr[128]= "expr[<procedure>,...]";	*/
/* 			  char *expr= (AH_EXPR)? AH_EXPR : lexpr;	*/
#endif
#if DEBUG == 2
			  double *largs;
#else
			  ALLOCA( largs, double, argc, largs_len);
#endif
#if DEBUG == 2
				largs= (double*) calloc( argc, sizeof(double) );
#endif
				memset( largs, 0, sizeof(double)* argc );
				if( ascanf_arguments ){
					memcpy( largs, &args[2], ascanf_arguments* sizeof(double) );
					SET_AF_ARGLIST( largs, ascanf_arguments );
				}
				else{
					SET_AF_ARGLIST( ascanf_ArgList, 0 );
				}
				if( !n ){
					n= 1;
				}
				if( fascanf_eval( &n, af->usage, largs, NULL, NULL, compile ) ){
					*result= largs[0];
				}
				fascanf_unparsed_remaining= fur;
				ascanf_update_ArgList= auA;
				GCA();
#if DEBUG == 2
				xfree( largs );
#endif
			}
			ascanf_arguments= argc;
		}
		else{
			ascanf_emsg= " (1st argument must be a stringpointer) ";
			ascanf_arg_error= 1;
		}
	}
	else{
		ascanf_arg_error= 1;
		r= 0;
	}
	return !ascanf_arg_error;
}

int ascanf_SetProcess ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  int take_usage;
  ascanf_Function *raf;
  static ascanf_Function AF= {NULL};
  static char *AFname= "SetProcess-Static-StringPointer";
	raf= &AF;
	if( raf->name ){
		xfree(raf->usage);
		memset( raf, 0, sizeof(ascanf_Function) );
	}
	else{
		raf->usage= NULL;
	}
	raf->type= _ascanf_variable;
	raf->name= AFname;
	raf->is_address= raf->take_address= True;
	raf->is_usage= raf->take_usage= True;
	raf->internal= True;

	if( args && ascanf_arguments>= 1 ){
		if( args[0]>= 0 && args[0]< setNumber ){
		  ascanf_Function *af;
		  int set_nr= (int) args[0];
		  DataSet *this_set= &AllSets[set_nr];
		  char *c= this_set->process.set_process ;
			ascanf_arg_error= False;
			xfree( raf->usage );
			raf->usage= XGstrdup( (c)? c : "" );
			raf->value= (c)? True : False;
			*result= take_ascanf_address(raf);
			if( ascanf_arguments> 1 &&
				(af= parse_ascanf_address( args[1], 0, "ascanf_SetProcess", (int) ascanf_verbose, &take_usage )) && take_usage &&
				!ascanf_SyntaxCheck
			){
				if( af->usage && *af->usage ){
					if( XGstrcmp(af->usage, this_set->process.set_process) ){
						xfree( this_set->process.set_process );
						this_set->process.set_process= XGstrdup(af->usage);
						xfree( this_set->process.description );
						if( ascanf_arguments> 2 &&
							(af= parse_ascanf_address( args[2], 0, "ascanf_SetProcess", (int) ascanf_verbose, &take_usage ))
								&& take_usage
						){
							this_set->process.description= XGstrdup(af->usage);
						}
						new_process_set_process( ActiveWin, this_set );
						if( pragma_unlikely(ascanf_verbose) ){
							fprintf( StdErr, " (installed new set %d *SET_PROCESS* %s \"%s\") ",
								set_nr, this_set->process.set_process,
								(af->usage)? af->usage : "<no description>"
							);
						}
					}
				}
				else{
					xfree( this_set->process.set_process );
					new_process_set_process( ActiveWin, this_set );
					if( pragma_unlikely(ascanf_verbose) ){
						fprintf( StdErr, " (removed set %d *SET_PROCESS*) ", set_nr);
					}
				}
			}
			else if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "(process argument 1==%s is not a stringpointer) ", ad2str(args[1], NULL, NULL) );
			}
		}
		else{
			ascanf_emsg= " (invalid setnumber) ";
			ascanf_arg_error= True;
		}
	}
	else{
		ascanf_arg_error= True;
		*result= 0;
	}
	return(!ascanf_arg_error);
}

int ascanf_ParseArguments ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af;
  int i= 0, take_usage;
	if( args && !ascanf_SyntaxCheck ){
		*result= 0;
		if( ActiveWin && ActiveWin!= &StubWindow && ascanf_arguments ){
			CopyFlags( NULL, ActiveWin );
		}
		for( i= 0; i< ascanf_arguments; i++ ){
			if( (af= parse_ascanf_address( args[i], 0, "ascanf_ParseArguments", (int) ascanf_verbose, &take_usage )) &&
				take_usage
			){
				if( af->usage && *af->usage ){
					ParseArgsString2( af->usage, setNumber );
					*result+= 1;
				}
			}
			else if( pragma_unlikely(ascanf_verbose) ){
				fprintf( StdErr, "(argument %d==%s is not a stringpointer) ", i, ad2str(args[i], NULL, NULL) );
			}
		}
		if( ActiveWin && ActiveWin!= &StubWindow && *result ){
		  LocalWin *lwi= ActiveWin;
			CopyFlags( lwi, NULL );
			UpdateWindowSettings( lwi, False, True );
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
		}
	}
	if( !i ){
		ascanf_arg_error= True;
	}
	return(!ascanf_arg_error);
}

static char *setColumnLabelsString( DataSet *set, int column, char *newstr, int new, int nCI, int *ColumnInclude )
{ char *ret= NULL;
  LabelsList *List;
	if( newstr ){
		if( new ){
			set->ColumnLabels= Free_LabelsList(set->ColumnLabels);
		}
		if( *newstr ){
		  char *e= &newstr[strlen(newstr)-1];
			List= set->ColumnLabels;
			if( *e== '\n' ){
				*e= '\0';
			}
			else{
				e= NULL;
			}
			if( column< 0 ){
				set->ColumnLabels= Parse_SetLabelsList( List, newstr, '\t', nCI, ColumnInclude );
			}
			else{
				set->ColumnLabels= Add_LabelsList( List, NULL, column, newstr );
			}
			if( e ){
				*e= '\n';
			}
			RedrawSet(set->set_nr, False);
		}
		List= set->ColumnLabels;
	}
	else{
		List= (set->ColumnLabels)? set->ColumnLabels : 
			((ActiveWin)? ActiveWin->ColumnLabels : NULL);
	}
	if( List ){
	  Sinc sink;
		sink.sinc.string= NULL;
		Sinc_string_behaviour( &sink, NULL, 0,0, SString_Dynamic );
		Sflush( &sink );
		if( column< 0 ){
			Sprint_SetLabelsList( &sink, List, "", "" );
		}
		else{
			Sputs( Find_LabelsListLabel( List, column ), &sink );
		}
		ret= sink.sinc.string;
	}
	return(ret);
}

char *ColumnLabelsString( DataSet *set, int column, char *newstr, int new, int nCI, int *ColumnInclude )
{ char *ret = setColumnLabelsString( set, column, newstr, new, nCI, ColumnInclude );
	if( newstr  && set->links ){
	  int i;
	  DataSet *that_set;
		for( i = 0 ; i< setNumber ; i++ ){
			if( (that_set= &AllSets[i])->set_link== set->set_nr ){
				setColumnLabelsString( that_set, column, newstr, new, nCI, NULL );
			}
		}
	}
	return ret;
}

int ascanf_SetColumnLabels ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  ascanf_Function *af= NULL, *naf= NULL;
  int take_usage= 0;
  static ascanf_Function AF= {NULL};
  static char *AFname= "SetColumnLabels-Static-StringPointer";
  int idx= (int) *ascanf_setNumber, new= False, column= -1;
	af= &AF;
	if( af->name ){
		xfree(af->usage);
		memset( af, 0, sizeof(ascanf_Function) );
	}
	else{
		af->usage= NULL;
	}
	af->type= _ascanf_variable;
	af->name= AFname;
	af->is_address= af->take_address= True;
	af->is_usage= af->take_usage= True;
	af->internal= True;
	ascanf_arg_error= False;
	if( args ){
		if( args[0]>= 0 && ((AllSets && args[0]<= setNumber) || ascanf_SyntaxCheck) ){
			idx= (int) args[0];
		}
		else{
			ascanf_emsg= " (setnumber out of range) ";
			ascanf_arg_error= True;
		}
		if( ascanf_arguments> 1 ){
			column= (int) args[1];
		}
		if( ascanf_arguments> 2 ){
			if( !(af= parse_ascanf_address( args[2], _ascanf_variable, "ascanf_SetColumnLabels", (int) ascanf_verbose, &take_usage )) ){
				af= &AF;
			}
		}
		if( ascanf_arguments> 3 ){
			if( (naf= parse_ascanf_address( args[3], _ascanf_variable, "ascanf_SetColumnLabels", (int) ascanf_verbose, &take_usage )) &&
				!take_usage
			){
				ascanf_emsg= " (new ColumnLabels argument must be a stringpointer) ";
				ascanf_arg_error= True;
				naf= NULL;
			}
		}
		if( ascanf_arguments> 4 ){
			new= (args[4])? True : False;
		}
	}
	*result= 0;
	if( af && !ascanf_arg_error ){
		if( !ascanf_SyntaxCheck && AllSets && idx< setNumber ){
		  char *list;
			if( naf ){
				list= ColumnLabelsString( &AllSets[idx], column, (naf->usage)? naf->usage : "", new, 0,NULL );
			}
			else{
				list= ColumnLabelsString( &AllSets[idx], column, NULL, new, 0,NULL );
			}
			xfree( af->usage );
			af->usage= list;
			*result= take_ascanf_address( af );
		}
		else{
			  /* When compiling/ checking syntax, we *must* return a safe pointer of the correct type.
			   \ Otherwise, printf[] might complain. NB: this means that the SyntaxCheck field must be
			   \ set in this function's entry in the function table!
			   */
			*result= take_ascanf_address( af );
		}
	}
	return( !ascanf_arg_error );
}

int ascanf_bgColour ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
	if( ascanf_arguments>= 1 ){
	  ascanf_Function *raf;
		raf= parse_ascanf_address(args[0], _ascanf_variable, "ascanf_bgColour", False, NULL);
		*result= args[0];
		if( ActiveWin && ActiveWin!= &StubWindow ){
			if( raf ){
				  /* Store the colourname in the provided stringpointer, and return args[1],
				   \ which still points to it.
				   */
				xfree( raf->usage );
				raf->usage= XGstrdup( bgCName );
				raf->is_usage= True;
			}
			if( ascanf_arguments>= 2 && !ascanf_SyntaxCheck ){
			  ascanf_Function *naf;
			  char *cname= NULL;
				if( (naf= ascanf_ParseColour( args[1], &cname, True, "ascanf_bgColour")) ){
					if( cname ){
					  Pixel tp;
						if( GetColor( cname, &tp ) && bgPixel!= tp ){
						  LocalWindows *WL= WindowList;
						  extern GC ACrossGC, BCrossGC;
							FreeColor( &bgPixel, &bgCName );
							bgPixel = tp;
							StoreCName( bgCName );
/* 							ReallocColours( True );	*/
							xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );
							RecolourCursors();
							ChangeCrossGC( &ACrossGC );
							ChangeCrossGC( &BCrossGC );
							while( WL ){
							  LocalWin *lwi= WL->wi;
								XSetWindowBackground( disp, lwi->window, bgPixel );
								WL= WL->next;
							}
							if( naf && naf->type== _ascanf_array && naf->N== 3 ){
								StoreCName( naf->usage );
							}
							XGStoreColours = 1;
						}
					}
					else{
						return(1);
					}
				}
				else{
					ascanf_emsg= " (invalid colour specification) ";
					ascanf_arg_error= True;
				}
			}
		}
	}
	else{
		ascanf_arg_error= 1;
		*result= 0;
	}
	return(1);
}

int ascanf_AddDataPoints ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  DataSet *this_set;
  int spot, Spot, N= ascanf_arguments- 1;
  int column[ASCANF_DATA_COLUMNS]= {0,1,2,3};
  extern struct Process ReadData_proc;
	if( !args || ascanf_arguments< 3 ){
		ascanf_arg_error= 1;
		return(1);
	}
	if( args[0]< 0 || (StartUp && args[0]> setNumber) || (!StartUp && args[0]> setNumber) ){
		ascanf_arg_error= 1;
		ascanf_emsg= "(set# out of range)";
		return(1);
	}
	this_set= &AllSets[(int) args[0]];
	spot= Spot= this_set->numPoints;
	if( N> this_set->ncols ){
		fprintf( StdErr, " (N=%d values truncated to %d) ", N, this_set->ncols );
		N= this_set->ncols;
	}
	AddPoint( &this_set, &spot, &Spot, N, &args[1], column, __FILE__, 0, __LINE__, NULL, "AddDataPoints[]", &ReadData_proc );
	if( StartUp ){
		maxitems+= N;
	}
	else if( ActiveWin && this_set->numPoints> maxitems ){
		maxitems= this_set->numPoints;
		realloc_Xsegments();
	}
	*result= spot;
	return(1);
}

/* NewSet[[points[,columns[,link2]]]]	*/
int ascanf_NewSet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT
  DataSet *this_set;
  int idx= 0, set_link= -1;
  double points= -1, ncols= -1;
  extern DataSet *find_NewSet();
#ifdef NO_FILE_SCALING
  extern double Xscale,		/* scale factors for x,y,dy	*/
		Yscale,
		DYscale;		/* reset for each new set and file read-in	*/
#endif
	if( ascanf_arguments> 0 ){
		points= ssfloor( args[0] );
	}
	if( ascanf_arguments> 1 ){
		ncols= ssfloor( args[1] );
	}
	if( ascanf_arguments> 2 ){
		set_link= ssfloor( args[2] );
	}
	if( !ascanf_SyntaxCheck && (this_set= find_NewSet( ActiveWin, &idx )) ){
		if( set_link>= 0 ){
			LinkSet2( this_set, set_link );
		}
		else{
			if( points>= 0 ){
				this_set->numPoints= points;
			}
			if( ncols>= 3 ){
				this_set->ncols= ncols;
			}
			if( this_set->numPoints ){
				realloc_points( this_set, this_set->numPoints, False );
			}
		}
		if( StartUp ){
			maxitems+= this_set->numPoints;
		}
		else if( ActiveWin && this_set->numPoints> maxitems ){
			maxitems= this_set->numPoints;
			realloc_Xsegments();
		}
		  /* 20040301 */
		{ LocalWindows *WL= WindowList;
			while( WL ){
				if( WL->wi->fileNumber ){
					WL->wi->fileNumber[ this_set->set_nr ]= this_set->fileNumber;
				}
				WL= WL->next;
			}
		}
		this_set->draw_set= 0;
#ifdef NO_FILE_SCALING
		this_set->Xscale= Xscale;
		this_set->Yscale= Yscale;
		this_set->DYscale= DYscale;
#else
		this_set->Xscale= 1;
		this_set->Yscale= 1;
		this_set->DYscale= 1;
#endif
		*result= idx;
	}
	else{
		*result= -1;
	}
	return(1);
}

int ascanf_DestroySet ( ASCB_ARGLIST )
{ ASCB_FRAME_SHORT

	if( !args || ascanf_arguments< 1 ){
		ascanf_arg_error= 1;
		*result= 0;
		return(1);
	}
	if( args[0]< 0 || args[0]> setNumber ){
		ascanf_arg_error= 1;
		ascanf_emsg= "(set# out of range)";
		*result= 0;
		return(1);
	}
	if( AllSets[(int)args[0]].numPoints> 0 ){
		Destroy_Set( &AllSets[(int) args[0]], False );
		CleanUp_Sets();
		*result= 1;
	}
	else{
		*result= 0;
	}
	return(1);
}

