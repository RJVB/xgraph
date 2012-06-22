/* 
vim:ts=4:sw=4:
 * xgraph - A Simple Plotter for X
 *
 * David Harrison
 * University of California,  Berkeley
 * 1986, 1987, 1988, 1989
 *
 * Please see copyright.h concerning the formal reproduction rights
 * of this software.

 \ 961125: this module is more or less involved with input-handling: ReadData c.s. & ParseArgs c.s.

 */

#include "config.h"
IDENTIFY( "Main input-related routines" );

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

#ifdef XG_DYMOD_SUPPORT
#	include "dymod.h"
#endif

#include <pwd.h>
#include <ctype.h>
#include "xgout.h"
#include "xgraph.h"
#include "xtb/xtb.h"

#include "hard_devices.h"
extern xtb_frame SD_Dialog, HO_Dialog;

#include <X11/Xutil.h>
#include <X11/keysym.h>

#include "new_ps.h"

#include "Python/PythonInterface.h"

#ifdef linux
#	include <asm/poll.h>
#elif !defined(__MACH__)
#	include <poll.h>
#endif

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

#ifndef MININT
#	define MININT (1 << (8*sizeof(int)-1))
#endif
#ifndef MAXINT
#	define MAXINT (-1 ^ (MININT))
#endif

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

#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif

#include <float.h>
#include "XXseg.h"
#include "ascanf.h"

#include "fdecl.h"

#include "copyright.h"

extern double zero_epsilon;

extern double Gonio_Base_Value, Gonio_Base_Value_2, Gonio_Base_Value_4;
extern double Units_per_Radian, Gonio_Base_Offset;

extern double Gonio_Base( LocalWin *wi, double base, double offset);

#define Gonio(fun,x)	(fun(((x)+Gonio_Base_Offset)/Units_per_Radian))
#define InvGonio(fun,x)	((fun(x)*Units_per_Radian-Gonio_Base_Offset))
#define Sin(x) Gonio(sin,x)
#define Cos(x) Gonio(cos,x)
#define Tan(x) Gonio(tan,x)
#define ArcSin(x) InvGonio(asin,x)
#define ArcCos(x) InvGonio(acos,x)
#define ArcTan(wi,x,y) (_atan3(wi,x,y)*Units_per_Radian-Gonio_Base_Offset)

#define nsqrt(x)	(x<0.0 ? 0.0 : sqrt(x))
#define sqr(x)		((x)*(x))

#define ISCOLOR		(wi->dev_info.dev_flags & D_COLOR)

#define MAX_LINESTYLE			MAXATTR /* ((bwFlag)?(MAXATTR-1):(MAXSETS/MAXATTR)-1) 	*/
#define COLMARK(set) ((set) / MAXATTR)

#define BWMARK(set) \
((set) % MAXATTR)

#define NORMSIZEX	600
#define NORMSIZEY	400
#define NORMASP		(((double)NORMSIZEX)/((double)NORMSIZEY))
#define MINDIM		100

extern int XGTextWidth();

extern void init_X();
#ifdef TOOLBOX
extern void do_error();
#endif
extern char *tildeExpand();

extern double psm_base, psm_incr, psdash_power;
extern int psm_changed, psMarkers;

extern double page_scale_x, page_scale_y;
extern int preserve_screen_aspect;
extern int XG_preserve_filetime;

extern int MaxSets;
extern int NCols, xcol, ycol, ecol, lcol, Ncol, MaxCols, error_type;
extern DataSet *AllSets;

extern int local_buf_size;

extern double Progress_ThresholdTime;

extern XSegment *Xsegs, *lowYsegs, *highYsegs;		/* Point space for X */
extern XXSegment *XXsegs;
#define LYsegs	lowYsegs
#define HYsegs	highYsegs
extern XSegment_error *XsegsE;
extern long XsegsSize, XXsegsSize, YsegsSize, XsegsESize;

extern LocalWin *ActiveWin;

/* For reading in the data */
extern int setNumber , fileNumber;
extern int maxSize ;

extern int legend_type;

/* our stderr	*/
extern FILE *StdErr;
extern int use_pager;

extern char *ShowLegends( LocalWin *wi, int PopUp, int this_one);
extern int detach, PS_PrintComment, Argc;
extern char **Argv, *titleText2;

extern FILE *NullDevice, **PIPE_fileptr;

extern double *ascanf_memory;
extern double *ascanf_self_value, *ascanf_current_value;
extern int reset_ascanf_currentself_value, reset_ascanf_index_value, ascanf_arg_error;

/* Basic transformation stuff */

extern double llx, lly, llpx, llpy, llny, urx, ury; /* Bounding box of all data */
extern double real_min_x, real_max_x, real_min_y, real_max_y;

extern char *XGstrdup(const char *), *parse_codes(char *), *cleanup(char *);

extern double scale_av_x, scale_av_y;
extern double _scale_av_x, _scale_av_y;
extern SimpleStats SS_X, SS_Y, SS_SY;
extern int show_stats;
extern SimpleStats SS_x, SS_y, SS_e, SS_sy;
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
#define R_UsrOrgX	win_geo.R_UsrOrgX
#define R_UsrOrgY	win_geo.R_UsrOrgY
#define R_UsrOppX	win_geo.R_UsrOppX
#define R_UsrOppY	win_geo.R_UsrOppY
#define XUnitsPerPixel	win_geo._XUnitsPerPixel
#define YUnitsPerPixel	win_geo._YUnitsPerPixel
#define R_XUnitsPerPixel	win_geo.R_XUnitsPerPixel
#define R_YUnitsPerPixel	win_geo.R_YUnitsPerPixel

extern int dump_average_values, DumpProcessed;

extern double _Xscale,		/* global scale factors for x,y,dy	*/
	_Yscale,
	_DYscale,		/* reset for each new file read-in	*/
	MXscale,
	MYscale,
	MDYscale;
extern double
	_MXscale,
	_MYscale,
	_MDYscale;
extern double Xscale,		/* scale factors for x,y,dy	*/
	Yscale,
	DYscale;		/* reset for each new set and file read-in	*/
extern double Xscale2, Yscale2,
	   XscaleR;
extern int linestyle, elinestyle, pixvalue, markstyle;
extern int SameAttrs, ResetAttrs;

extern double *ascanf_memory;
extern double *ascanf_self_value, *ascanf_current_value;
extern int ascanf_exit, reset_ascanf_currentself_value,
	reset_ascanf_index_value, ascanf_arg_error;
extern double *ascanf_setNumber, *ascanf_TotalSets, *ascanf_numPoints, *ascanf_Counter, *ascanf_counter;

extern int autoscale, disconnect, split_set;
extern char *split_reason;
int ignore_splits= False, splits_disconnect= True;

extern int x_scale_start, y_scale_start;
extern int xp_scale_start, yp_scale_start, yn_scale_start;
extern LocalWin startup_wi;
extern double MusrLX, MusrRX, usrLpX, usrLpY, usrHnY, MusrLY, MusrRY;
extern int use_lx, use_ly;
extern int use_max_x, use_max_y;
extern int User_Coordinates;

extern int zeroFlag, triangleFlag, polarFlag, polarLog, NumObs;
extern int vectorType;
extern double vectorLength, vectorPars[MAX_VECPARS];
extern double radix, radix_offset;
extern char radixVal[64], radix_offsetVal[64];
extern int vectorFlag;
extern int FitOnce;
extern int NewProcess_Rescales;

extern int ReadData_commands;
extern Boolean ReadData_Outliers_Warn;
extern Process ReadData_proc;
extern char *transform_x_buf;
extern char *transform_y_buf;
extern int transform_x_buf_len, transform_y_buf_len;
extern int transform_x_buf_allen, transform_y_buf_allen;

extern int file_splits, plot_only_file, filename_in_legend, labels_in_legend, _process_bounds;
extern double data[ASCANF_DATA_COLUMNS];
extern int XUnitsSet, YUnitsSet, titleTextSet, column[ASCANF_DATA_COLUMNS], use_errors, no_errors, Show_Progress;
extern double newfile_incr_width;

extern LocalWin StubWindow, *primary_info, *check_wi(LocalWin **wi, char *caller);

extern int PS_PrintComment;
extern Window ascanf_window;

extern void PIPE_handler( int sig);
extern void handle_FPE();

extern int absYFlag, use_average_error, error_regionFlag, use_markFont, overwrite_legend,
	overwrite_AxGrid, overwrite_marks, arrows, process_bounds;
extern char *Prog_Name;
extern double Xbias_thres, Ybias_thres, legend_ulx, legend_uly, intensity_legend_ulx, intensity_legend_uly;

extern int UseRealXVal,
	UseRealYVal;
extern int logXFlag;			/* Logarithmic X axis      */
extern int logYFlag;			/* Logarithmic Y axis      */
extern int exact_X_axis, exact_Y_axis, ValCat_X_axis, ValCat_X_levels, ValCat_Y_axis, show_all_ValCat_I, ValCat_I_axis;
extern int ValCat_X_grid;
extern char log_zero_sym_x[MAXBUFSIZE+1], log_zero_sym_y[MAXBUFSIZE+1];	/* symbol to indicate log_zero	*/
extern int lz_sym_x, lz_sym_y;	/* log_zero symbols set?	*/
extern int log_zero_x_mFlag, log_zero_y_mFlag;
extern double log_zero_x, log_zero_y;	/* substitute 0.0 for these values when using log axis	*/
extern double _log_zero_x, _log_zero_y;	/* transformed log_zero_[xy]	*/
extern double log10_zero_x, log10_zero_y;
extern int sqrtXFlag, sqrtYFlag;
extern double powXFlag, powYFlag;
extern double powAFlag;

extern int raw_display, show_overlap, scale_plot_area_x, scale_plot_area_y, transform_axes, XG_SaveBounds;
extern double Xincr_factor, Yincr_factor, ValCat_X_incr, ValCat_Y_incr;
extern int xname_placed, yname_placed, yname_vertical, yname_trans, xname_trans, legend_placed, legend_trans;
extern int intensity_legend_placed, no_intensity_legend, intensity_legend_trans;
extern double xname_x, xname_y, yname_x, yname_y;
extern int progname, IconicStart, use_RootWindow, Sort_Sheet, size_window_print, print_immediate, settings_immediate;
extern double *do_gsTextWidth_Batch;
int lFX= 0, lFY= 0;
extern int FitX, FitY;
extern int Aspect, XSymmetric, YSymmetric;

extern int *plot_only_set, plot_only_set_len, *highlight_set, *mark_set, highlight_set_len, mark_set_len,
	no_title, AllTitles, no_legend, no_legend_box, legend_always_visible, Print_Orientation;
int mark_sets;
extern char *PrintFileName;
extern char UPrintFileName[];
extern int noExpY, noExpX, axisFlag, barBase_set, barWidth_set, barType, barType_set, debugging, debugLevel;
extern char *InFiles, *InFilesTStamps;

int WarnNewSet= False;

extern int maxitems;
extern int use_xye_info;
extern char *AddFile, *ScriptFile;
extern LocalWin *ScriptFileWin;

#include "xfree.h"

int NewSet( LocalWin *wi, DataSet **this_set, int spot )
{ LocalWindows *WL;
	if (spot > maxSize)
		maxSize = spot;
#ifdef NO_SETSREALLOC
    if (setNumber >= MAXSETS- 1) {
	  char msg[512];
		sprintf( msg, "Too many data sets - use -maxsets <x> with x>%d\n", MaxSets);
		fprintf( StdErr, msg );
		if( wi ){
			xtb_error_box( wi->window, msg, "NewSet" );
		}
		  /* set setNumber to its maximal value, instead of letting it
		   \ stick at the one-but-maximal value...
		   */
		setNumber= MAXSETS;
		return -1;
    } else
#else
    if( setNumber >= MAXSETS- 1){
	  /* The true index of this set, independent of whether or not the set_nr
	   \ field is still the same as the index.
	   */
	  int set_nr= (*this_set) - AllSets;
		if( !realloc_sets( wi, MaxSets, MaxSets* 2, "NewSet" ) ){
			return( -1 );
		}
		else{
			  /* We *must* reset the current *this_set: there is no guarantee
			   \ whatsoever that an array keeps it base-address across a reallocation.
			   */
			*this_set= &AllSets[set_nr];
			fprintf( StdErr, "\t(consider using -maxsets <n>)\n" );
		}
	}
#endif
	{
		if( (*this_set)->numPoints> 0 || ((*this_set)->set_link>= 0 && !(*this_set)->set_linked) ){
		  char buf[32], *name;
			  /* before we show stats (and they are added to the Info box!), check
			   \ if anything should be appended to this set's name.
			   */
			if( (*this_set)->appendName ){
			  char *c;
				c= (*this_set)->setName;
				if( c ){
					name= XGstrdup2( c, (*this_set)->appendName );
				}
				else{
					name= XGstrdup( (*this_set)->appendName );
				}
			}
			else if( (*this_set)->setName ){
				name= XGstrdup( (*this_set)->setName );
			}
			else{
				name= d2str( (double) setNumber, "Set #%g", buf);
			}
			Show_Stats( StdErr,
				(strlen(name)<= 30)? name : d2str((double)setNumber, "Set #%g", buf),
				&SS_x, &SS_y, &SS_e, &SS_sy, NULL
			);
			SS_Reset_(SS_x);
			SS_Reset_(SS_y);
			SS_Reset_(SS_e);
			SS_Reset_(SS_sy);
			if( name!= buf ){
				xfree(name);
			}

			if( debugFlag){
				fprintf( StdErr, "Set #%d: %d points (allocsize %d)\n",
					setNumber, (*this_set)->numPoints, (*this_set)->allocSize
				);
				fprintf( StdErr, "\tMax. scale factors: %g %g %g\n", _MXscale, _MYscale, _MDYscale );
				fprintf( StdErr, "\tMax. scale factors based on data*scale: %g %g %g\n", MXscale, MYscale, MDYscale );
				fprintf( StdErr, "\tAbsolute vertical set bounds: (%g,%g) - (%g,%g)\n",
					(*this_set)->lowest_y.x* (*this_set)->Xscale,
					(*this_set)->lowest_y.y* (*this_set)->Yscale - (*this_set)->lowest_y.errr* (*this_set)->DYscale,
					(*this_set)->highest_y.x* (*this_set)->Xscale,
					(*this_set)->highest_y.y* (*this_set)->Yscale + (*this_set)->highest_y.errr* (*this_set)->DYscale
				);
			}
			if( (*this_set)->set_link< 0 ){
				  /* Get rid of redundant memory, with a reserve of one point	*/
				realloc_points( *this_set, (*this_set)->numPoints+ 2, False );
				  /* Assume that the next set will contain as many points as the current one	*/
				INITSIZE= (*this_set)->numPoints+ 1;
			}

			setNumber++;
			WL= WindowList;
			while( WL ){
				WL->wi->sets_reordered= True;
				WL= WL->next;
			}


			AllSets[setNumber].NumObs= (*this_set)->NumObs;
			AllSets[setNumber].vectorLength= (*this_set)->vectorLength;
			AllSets[setNumber].vectorType= (*this_set)->vectorType;
			memcpy( AllSets[setNumber].vectorPars, (*this_set)->vectorPars, MAX_VECPARS* sizeof(double));
			AllSets[setNumber].set_nr= setNumber;
			if( SameAttrs ){
				linestyle= AllSets[setNumber].linestyle = (*this_set)->linestyle;
				AllSets[setNumber].elinestyle = (*this_set)->elinestyle;
				pixvalue= AllSets[setNumber].pixvalue = (*this_set)->pixvalue;
				xfree( AllSets[setNumber].pixelCName );
				AllSets[setNumber].pixelCName= XGstrdup((*this_set)->pixelCName );
				  /* 20010222: If we want to enforce identical markerstyles on all sets
				   \ that persist in a 2nd gen. read, we'll have to use the negative
				   \ values!
				   */
				(*this_set)->markstyle= -ABS( (*this_set)->markstyle );
				markstyle= AllSets[setNumber].markstyle = (*this_set)->markstyle;
			}
			else{
				linestyle= AllSets[setNumber].linestyle = ((*this_set)->linestyle+ 1) % MAX_LINESTYLE;
				elinestyle= AllSets[setNumber].elinestyle = ((*this_set)->elinestyle+ 1) % MAX_LINESTYLE;
				pixvalue= AllSets[setNumber].pixvalue = ((*this_set)->pixvalue+ 1) % MAXATTR;
				xfree( AllSets[setNumber].pixelCName );
				AllSets[setNumber].pixelCName= XGstrdup( AllAttrs[pixvalue].pixelCName );
				markstyle= AllSets[setNumber].markstyle = ABS( (*this_set)->markstyle )+ 1;
			}
			AllSets[setNumber].error_type= error_type;
			  /* 20030314: */
			AllSets[setNumber].use_error= use_errors;
			AllSets[setNumber].set_link= -1;
			AllSets[setNumber].links= 0;

			*this_set= &AllSets[setNumber];
			(*this_set)->init_pass= True;
			*ascanf_setNumber= setNumber;
			*ascanf_TotalSets= setNumber;
			*ascanf_numPoints= (*this_set)->numPoints;
		}

		return maxSize;
    }
}

DataSet *find_NewSet( LocalWin *wi, int *idx )
{ int i, sn= setNumber;
  DataSet *set= NULL;
	if( idx && *idx>= 0 ){
		i= *idx;
	}
	else{
		i= 0;
	}
	while( (AllSets[i].numPoints> 0 || AllSets[i].set_link>= 0) && i< MaxSets ){
		i++;
	}
	if( i== MaxSets && AllSets[i-1].numPoints> 0 ){
		setNumber-= 1;
		set= &AllSets[setNumber];
		if( set->numPoints<= 0 ){
			set->numPoints= 0;
			i= setNumber;
			setNumber= sn;
			if( ascanf_verbose ){
				fprintf( StdErr, " (last set %d was empty/deleted)==", i );
			}
		}
		else{
			if( NewSet( wi, &set, 0 )== -1 ){
				i= -1;
				set= NULL;
			}
			else{
				if( ascanf_verbose ){
					fprintf( StdErr, " (created new sets)==" );
				}
			}
		}
		if( idx ){
			*idx= i;
		}
	}
	else if( AllSets[i].numPoints<= 0 ){
		set= &AllSets[i];
		if( idx ){
			*idx= i;
		}
		set->numPoints= 0;
		if( i>= setNumber ){
			setNumber= i+ 1;
		}
		if( ascanf_verbose ){
			fprintf( StdErr, " (found empty/deleted set %d)==", i );
		}
	}
	return( set );
}

unsigned long mem_alloced= 0L;

extern int allocerr;

void *XGcalloc( size_t n, size_t s)
{ void *mem= calloc( n, s);
	if( !mem ){
		allocerr+= 1;
	}
	return( mem );
}

extern short* _drawingOrder;
extern int drawingOrder_set;

extern int realloc_LocalWin_data( LocalWin*, int);

/* expand the set-related fields in *all* currently available windows	*/
int realloc_LocalWinList_data( LocalWin *wi, int start, int new )
{ LocalWindows *WL= WindowList;
    /* reallocate the "input" window's datastructures first:	*/
  LocalWin *WI= &StubWindow;
	do{
		errno= 0;
		if( !realloc_LocalWin_data( WI, new) ){
			if( WI!= &StubWindow ){
				return( 0 );
			}
			else{
				fprintf( StdErr, "realloc_LocalWinList_data(): error re-allocating startup-window - "
					"proceeding with caution (%s)\n",
					serror()
				);
			}
		}
		else{
		  int i;
			for( i= start; i< new; i++ ){
				WI->draw_set[i]= 1;
				WI->mark_set[i]= 0;
				WI->numVisible[i]= 0;
				WI->fileNumber[i]= -1;
				WI->new_file[i]= 0;
				WI->group[i]= -1;
				WI->xcol[i]= xcol;
				WI->ycol[i]= ycol;
				WI->ecol[i]= ecol;
				WI->lcol[i]= lcol;
				WI->error_type[i]= -1;
				WI->curve_len[i]= WI->error_len[i]= NULL;
#ifdef TR_CURVE_LEN
				WI->tr_curve_len[i]= NULL;
#endif
				if( WI->discardpoint ){
					WI->discardpoint[i]= NULL;
				}
				SS_Reset_(WI->set_X[i] );
				SS_Reset_(WI->set_Y[i] );
				SS_Reset_(WI->set_E[i] );
				SS_Reset_(WI->set_V[i] );
				WI->set_O[i].Gonio_Base= WI->radix;
				WI->set_O[i].Gonio_Offset= WI->radix_offset;
				WI->set_O[i].Nvalues= 0;
				WI->set_O[i].exact= 0;
				WI->set_O[i].sample= NULL;
				SAS_Reset_(WI->set_O[i] );
				SS_Reset_(WI->set_tr_X[i] );
				SS_Reset_(WI->set_tr_Y[i] );
				SS_Reset_(WI->set_tr_E[i] );
				SS_Reset_(WI->set_tr_V[i] );
				WI->set_tr_O[i].Gonio_Base= WI->radix;
				WI->set_tr_O[i].Gonio_Offset= WI->radix_offset;
				WI->set_tr_O[i].Nvalues= 0;
				WI->set_tr_O[i].exact= 0;
				WI->set_tr_O[i].sample= NULL;
				SAS_Reset_(WI->set_tr_O[i] );
				_drawingOrder[i]= i;
			}
		}
		if( WL ){
			WI= WL->wi;
			WL= WL->next;
		}
		else{
			WI= NULL;
		}
	} while( WI );
	return( 1 );
}

extern int MaxnumFiles;

/* Reallocate inFileNames	*/
int realloc_FileNames( LocalWin *wi, int offset, int new, char *caller )
{ extern char **inFileNames;
	if( (inFileNames= realloc( inFileNames, sizeof(char*)* new)) ){
		if( new> offset ){
			memset( &inFileNames[offset], 0, sizeof(char*)* (new- offset) );
		}
		MaxnumFiles= new;
		return( MaxnumFiles );
	}
	else{
	  char msg[512];
		sprintf( msg, "%s: Can't realloc inFileNames space (%s)\n", caller, serror() );
		fprintf( StdErr, msg );
		if( wi ){
			xtb_error_box( wi->window, msg, "realloc_FileNames()" );
		}
		MaxnumFiles= 0;
		return(0);
	}
}

/* Reallocate sets, expanding the number of sets	*/
int realloc_sets( LocalWin *wi, int offset, int new, char *caller )
{ int fok;
	if( new> MaxnumFiles ){
		fok= realloc_FileNames( wi, MaxnumFiles, new, "realloc_sets()" );
	}
	if( (AllSets= realloc( AllSets, sizeof(DataSet)* new)) &&
		fok &&
		realloc_LocalWinList_data( wi, offset, new)
	){
		if( new> offset ){
			memset( &AllSets[offset], 0, sizeof(DataSet)* (new- offset) );
		}
		Initialise_Sets( offset, new );
		fprintf( StdErr, "%s(%d) expanded DataSet space from %d to %d\n",
			caller, setNumber, MaxSets, new
		);
		MaxSets= new;
		return( MaxSets );
	}
	else{
	  char msg[512];
		sprintf( msg, "%s: problem reallocating dataset space (%s) - use -maxsets <x> with x>%d\n", caller, serror(), MaxSets);
		fprintf( StdErr, msg );
		if( wi ){
			xtb_error_box( wi->window, msg, "realloc_sets()" );
		}
		  /* set setNumber to its maximal value, instead of letting it
		   \ stick at the one-but-maximal value...
		   */
		setNumber= MAXSETS;
		return 0;
	}
}

double **XGrealloc_2d_doubles( double **cur_columns, int ncols, int nlines, int cur_ncols, int cur_nlines, char *caller )
{ double **columns= NULL;
  int i;
	if( !cur_columns ){
		if( (columns= (double**) XGreallocShared( columns, ncols* sizeof(double*), 0 )) ){
			if( debugFlag ){
				fprintf( StdErr, "%s: allocating %d columns X %d entries\n", caller, ncols, nlines );
			}
			for( i= 0; i< ncols; i++ ){
				columns[i]= NULL;
				if( !(columns[i]= (double*) XGreallocShared( columns[i], nlines* sizeof(double), 0 )) ){
					if( debugFlag ){
						fprintf( StdErr, "%s,n=%d: can't allocate %d entries for column %d (%s)\n",
							caller, ncols, nlines, i, serror()
						);
					}
					break;
				}
			}
		}
	}
	else{
		if( (columns= (double**) XGreallocShared( cur_columns, ncols* sizeof(double*), cur_ncols * sizeof(double*) )) ){
			if( debugFlag ){
				fprintf( StdErr, "%s: re-allocate from %d to %d columns X %d entries\n",
					caller, cur_ncols, ncols, nlines
				);
			}
			for( i= 0; i< ncols; i++ ){
			  size_t cl;
				if( i>= cur_ncols ){
					columns[i]= NULL;
					cl = 0;
				}
				else{
					cl = cur_nlines;
				}
				if( !(columns[i]= (double*) XGreallocShared( columns[i], nlines* sizeof(double), cl * sizeof(double) )) ){
					if( debugFlag ){
						fprintf( StdErr, "%s,n=%d: can't re-allocate %d entries for column %d (%s)\n",
							caller, ncols, nlines, i, serror()
						);
					}
					break;
				}
			}
		}
	}
	return( columns );
}

void XGfree_2d_doubles( double ***Columns, int ncols, int nlines )
{ int c;
  double **columns;
  	if( Columns && *Columns ){
		columns = *Columns;
		for( c = 0 ; c < ncols ; c++ ){
			XGfreeShared(columns[c], nlines * sizeof(double) );
		}
		XGfreeShared(columns, ncols * sizeof(double*) );
		*Columns = NULL;
	}
}

/* Allocate or re-allocate a set's columns, that is, the 2D array where the datapoints are stored.
 \ As of 20001101, the assignment of the new arena to this_set->columns is performed in this routine!
 */
double **realloc_columns( DataSet *this_set, int ncols )
{ int i;
	if( this_set->set_link>= 0 ){
		if( debugFlag || scriptVerbose ){
			fprintf( StdErr, "realloc_columns(#%d): requested realloc-points for linked-to set #%d - no action taken!!\n",
				this_set->set_nr, this_set->set_link );
		}
		return( this_set->columns );
	}
	if( !this_set->allocSize ){
		if( debugFlag ){
			fprintf( StdErr, "realloc_columns(#%d): no number of entries specified yet - no action taken\n", this_set->set_nr );
		}
		return( NULL );
	}
	{ char us[128];
		sprintf( us, "realloc_columns(#%d)", this_set->set_nr );
		this_set->columns= XGrealloc_2d_doubles( this_set->columns, ncols, this_set->allocSize, this_set->ncols, this_set->allocatedSize, us );
	}
	this_set->ncols= ncols;
	this_set->allocatedSize = this_set->allocSize;
	if( this_set->links ){
	  DataSet *that_set;
		for( i= 0; i< setNumber; i++ ){
			if( (that_set= &AllSets[i])->set_link== this_set->set_nr ){
				that_set->ncols= this_set->ncols;
				that_set->columns= this_set->columns;
			}
		}
	}
	Check_Sets_LinkedArrays( this_set );
	return( this_set->columns );
}

void realloc_points( DataSet *this_set, int allocSize, int force )
{ LocalWindows *WL= WindowList;
	allocerr= 0;
	if( this_set->allocSize== 0 ){
	  /* realloc _may_ do a malloc() when passed a NULL pointer, but
	   \ we want calloc()...
	   */
		this_set->allocSize = allocSize;
		this_set->discardpoint= NULL;
		this_set->xvec= NULL;
		this_set->xvec = (double *) XGrealloc( this_set->xvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->yvec= NULL;
		this_set->yvec = (double *) XGrealloc( this_set->yvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->lvec= NULL;
		this_set->lvec = (double *) XGrealloc( this_set->lvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->ldxvec= NULL;
		this_set->ldxvec = (double *) XGrealloc( this_set->ldxvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->hdxvec= NULL;
		this_set->hdxvec = (double *) XGrealloc( this_set->hdxvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->ldyvec= NULL;
		this_set->ldyvec = (double *) XGrealloc( this_set->ldyvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->hdyvec= NULL;
		this_set->hdyvec = (double *) XGrealloc( this_set->hdyvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->errvec= NULL;
		this_set->errvec = (double *) XGrealloc( this_set->errvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->columns= realloc_columns( this_set, (this_set->ncols> 2)? this_set->ncols : NCols);
		this_set->mem_alloced= this_set->ncols* ( sizeof(double**) + this_set->allocSize* sizeof(double)) +
			this_set->allocSize* ( 7* sizeof(double)+ 1* sizeof(int));
#if ADVANCED_STATS == 1
		this_set->N= NULL;
		this_set->N = (int *) XGrealloc((char *) this_set->N,
			  (unsigned) (this_set->allocSize * sizeof(int)));
		this_set->mem_alloced+= this_set->allocSize* ( sizeof(double)+ 1* sizeof(int));
#endif
#ifdef RUNSPLIT
		this_set->splithere= NULL;
#endif
		mem_alloced+= this_set->mem_alloced;
		while( WL ){
			WL->wi->pointVisible[this_set->set_nr]= (signed char*) XGrealloc( WL->wi->pointVisible[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(signed char)
			);
			mem_alloced+= sizeof(signed char)* this_set->allocSize;
			WL->wi->curve_len[this_set->set_nr]= (double*) XGrealloc( WL->wi->curve_len[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(double)
			);
			mem_alloced+= sizeof(double)* this_set->allocSize;
			WL->wi->error_len[this_set->set_nr]= (double*) XGrealloc( WL->wi->error_len[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(double)
			);
			mem_alloced+= sizeof(double)* this_set->allocSize;
#ifdef TR_CURVE_LEN
			WL->wi->tr_curve_len[this_set->set_nr]= (double*) XGrealloc( WL->wi->tr_curve_len[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(double)
			);
			mem_alloced+= sizeof(double)* this_set->allocSize;
#endif
			if( WL->wi->discardpoint && WL->wi->discardpoint[this_set->set_nr] ){
				WL->wi->discardpoint[this_set->set_nr]= (char*) XGrealloc( WL->wi->discardpoint[this_set->set_nr],
					(this_set->allocSize+ 2)* sizeof(char)
				);
				mem_alloced+= sizeof(char)* this_set->allocSize;
			}
			WL= WL->next;
		}
	}
	else if( this_set->allocSize!= allocSize || force ){
	  int oas= this_set->allocSize;
	  /* realloc handles both increasing and decreasing of allocated
	   \ memory, with preservation of the contents of the lesser
	   \ of the two (old,new) blocks. We don't take any action if
	   \ the requested sizes are equal (realloc() should neither, but
	   \ that we can't be sure of).
	   */
		mem_alloced-= this_set->mem_alloced- 2* sizeof(double)* this_set->allocSize;
		this_set->allocSize = allocSize;
		if( this_set->discardpoint ){
			this_set->discardpoint = (signed char *)
			  XGrealloc( this_set->discardpoint,
				  (unsigned) (this_set->allocSize *
					  sizeof(signed char)));
			if( allocSize> oas ){
				memset( &this_set->discardpoint[oas], 0, (allocSize- oas)* sizeof(signed char) );
			}
			mem_alloced+= this_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
		}
		this_set->xvec = (double *) XGrealloc( this_set->xvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->yvec = (double *) XGrealloc( this_set->yvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->lvec = (double *) XGrealloc( this_set->lvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->ldxvec = (double *) XGrealloc( this_set->ldxvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->hdxvec = (double *) XGrealloc( this_set->hdxvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->ldyvec = (double *) XGrealloc( this_set->ldyvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->hdyvec = (double *) XGrealloc( this_set->hdyvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		this_set->errvec = (double *) XGrealloc( this_set->errvec,
			  (unsigned) (this_set->allocSize * sizeof(double)));
		if( this_set->set_link< 0 ){
			this_set->columns= realloc_columns( this_set, (this_set->ncols> 2)? this_set->ncols : 3 );
			this_set->mem_alloced= this_set->ncols* ( sizeof(double**) + this_set->allocSize* sizeof(double)) +
				this_set->allocSize* ( 7* sizeof(double)+ 1* sizeof(int));
		}
		else{
			this_set->mem_alloced= this_set->allocSize* ( 7* sizeof(double)+ 1* sizeof(int));
		}
#if ADVANCED_STATS == 1
		this_set->N = (int *) XGrealloc((char *) this_set->N,
			  (unsigned) (this_set->allocSize * sizeof(int)));
		this_set->mem_alloced+= this_set->allocSize* ( sizeof(double)+ 1* sizeof(int));
#endif
#ifdef RUNSPLIT
		if( this_set->splithere ){
			this_set->splithere = (signed char *) XGrealloc( this_set->splithere,
				  (unsigned) (this_set->allocSize * sizeof(signed char)));
			if( allocSize> oas ){
				memset( &this_set->splithere[oas], 0, (allocSize- oas)* sizeof(signed char) );
			}
			this_set->mem_alloced+= this_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
		}
#endif
		mem_alloced+= this_set->mem_alloced;
		while( WL ){
			WL->wi->pointVisible[this_set->set_nr]= (double*) XGrealloc( WL->wi->pointVisible[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(signed char)
			);
			mem_alloced+= sizeof(signed char)* this_set->allocSize;
			WL->wi->curve_len[this_set->set_nr]= (double*) XGrealloc( WL->wi->curve_len[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(double)
			);
			mem_alloced+= sizeof(double)* this_set->allocSize;
			WL->wi->error_len[this_set->set_nr]= (double*) XGrealloc( WL->wi->error_len[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(double)
			);
			mem_alloced+= sizeof(double)* this_set->allocSize;
#ifdef TR_CURVE_LEN
			WL->wi->tr_curve_len[this_set->set_nr]= (double*) XGrealloc( WL->wi->tr_curve_len[this_set->set_nr],
				(this_set->allocSize+ 2)* sizeof(double)
			);
			mem_alloced+= sizeof(double)* this_set->allocSize;
#endif
			if( WL->wi->discardpoint && WL->wi->discardpoint[this_set->set_nr] ){
				WL->wi->discardpoint[this_set->set_nr]= (char*) XGrealloc( WL->wi->discardpoint[this_set->set_nr],
					(this_set->allocSize+ 2)* sizeof(char)
				);
				mem_alloced+= sizeof(char)* this_set->allocSize;
			}
			WL= WL->next;
		}
	}
	if( allocerr ){
		fprintf( StdErr, "realloc_points(): %d failure(s) allocating %dx%d points for set %d (%s)\n",
			allocerr, allocSize, this_set->ncols, this_set->set_nr, serror()
		);
		xg_abort();
	}
	if( this_set->links ){
	  int i;
	  DataSet *that_set;
		for( i= 0; i<= setNumber; i++ ){
			if( (that_set= &AllSets[i])->set_link== this_set->set_nr && that_set->numPoints>= 0 ){
				that_set->ncols= this_set->ncols;
				that_set->columns= this_set->columns;
				  /* 20031012: .... forgot this?! .... */
				that_set->numPoints= this_set->numPoints;
				realloc_points( that_set, this_set->allocSize, force );
			}
		}
	}
}

void realloc_Xsegments()
{ int mi= (maxitems)? maxitems : 1;
	allocerr= 0;

	Xsegs = (XSegment *) XGrealloc( Xsegs, (unsigned)mi * sizeof(XSegment));
	XXsegs = (XXSegment *) XGrealloc( XXsegs, (unsigned)mi * sizeof(XXSegment));
	lowYsegs = (XSegment *) XGrealloc( lowYsegs, (unsigned)(mi+ 8) * sizeof(XSegment));
	highYsegs = (XSegment *) XGrealloc( highYsegs, (unsigned)(mi+ 8) * sizeof(XSegment));
	XsegsE = (XSegment_error *) XGrealloc( XsegsE, (unsigned)mi * sizeof(XSegment_error));

	XsegsSize= (long) mi * sizeof(XSegment);
	XXsegsSize= (long) mi * sizeof(XXSegment);
	YsegsSize= (long) (mi+8) * sizeof(XSegment);
	XsegsESize= (long) mi * sizeof(XSegment_error);

	if( allocerr ){
		fprintf( StdErr, "realloc_Xsegments(): %d allocation failure(s) (%s)\n", allocerr, serror() );
		xg_abort();
	}
}

void Destroy_Set( DataSet *this_set, int relinking )
{ int nr, i;
  LocalWindows *WL= WindowList;
		if( !this_set ){
			return;
		}
		nr= this_set->set_nr;
		if( !relinking && this_set->process.set_process_len ){
			xfree( this_set->process.set_process );
			this_set->process.set_process_len= 0;
			Destroy_Form( &this_set->process.C_set_process );
		}
		this_set->allocSize = 0;
		this_set->numPoints = -1;
		this_set->draw_set= 0;
		xfree( this_set->xvec );
		xfree( this_set->yvec );
		xfree( this_set->lvec );
		xfree( this_set->ldxvec );
		xfree( this_set->hdxvec );
		xfree( this_set->ldyvec );
		xfree( this_set->hdyvec );
		xfree( this_set->errvec );
		if( this_set->set_link< 0 ){
			XGfree_2d_doubles( &this_set->columns, this_set->ncols, this_set->allocatedSize );
			if( this_set->links ){
			  DataSet *that_set;
				for( i= 0; i< setNumber && this_set->links>= 0; i++ ){
					  /* 20031107: (that_set..)->set_link was ..->set_nr !! */
					if( (that_set= &AllSets[i])->set_link== this_set->set_nr && this_set!= that_set ){
						Destroy_Set( that_set, False );
						this_set->links-= 1;
					}
				}
			}
		}
		else if( this_set->set_link>= 0 && this_set->set_link< setNumber ){
			AllSets[this_set->set_link].links-= 1;
			this_set->columns= NULL;
			this_set->set_link= -1;
			this_set->set_linked= False;
		}
		this_set->ncols = 0;
#if ADVANCED_STATS == 1
		xfree( this_set->N );
#endif
		if( !relinking ){
			xfree( this_set->read_file );
			xfree( this_set->Associations );
			this_set->numAssociations= 0;
			this_set->allocAssociations= 0;
			xfree_setitem( this_set->setName, this_set );
			xfree_setitem( this_set->appendName, this_set );
			xfree_setitem( this_set->fileName, this_set );
			xfree_setitem( this_set->XUnits, this_set );
			xfree_setitem( this_set->YUnits, this_set );
			xfree_setitem( this_set->titleText, this_set );
		}
		xfree_setitem( this_set->average_set, this_set );
		Check_Sets_LinkedArrays( this_set );
		this_set->propsSet= False;
		while( WL ){
		  LocalWin *lwi= WL->wi;
			if( !relinking ){
				if( lwi->plot_only_group+ 1== lwi->group[nr] ){
					lwi->plot_only_group-= 1;
				}
				lwi->fileNumber[nr]= -1;
				if( lwi->plot_only_file+ 1== lwi->fileNumber[nr] ){
					lwi->plot_only_file-= 1;
				}
				lwi->group[nr]= -1;
				lwi->fileNumber[nr]= -1;
				lwi->new_file[nr]= 0;
				lwi->draw_set[nr]= 0;
				lwi->mark_set[nr]= 0;
				lwi->numVisible[nr]= 0;
				lwi->legend_line[nr].highlight= 0;
				lwi->xcol[nr]= -1;
				lwi->ycol[nr]= -1;
				lwi->ecol[nr]= -1;
				lwi->lcol[nr]= -1;
				lwi->error_type[nr]= 0;
				lwi->set_O[nr].Gonio_Base= lwi->radix;
				lwi->set_O[nr].Gonio_Offset= lwi->radix_offset;
				lwi->set_O[nr].exact= 0;
				lwi->set_tr_O[nr].Gonio_Base= lwi->radix;
				lwi->set_tr_O[nr].Gonio_Offset= lwi->radix_offset;
				lwi->set_tr_O[nr].exact= 0;
			}
			xfree( lwi->curve_len[nr] );
			xfree( lwi->error_len[nr] );
#ifdef TR_CURVE_LEN
			xfree( lwi->tr_curve_len[nr] );
#endif
			if( lwi->discardpoint ){
				xfree( lwi->discardpoint[nr] );
			}
			SS_Reset_( lwi->set_X[nr] );
			SS_Reset_( lwi->set_Y[nr] );
			SS_Reset_( lwi->set_E[nr] );
			lwi->set_O[nr].Nvalues= 0;
			xfree( lwi->set_O[nr].sample);
			SAS_Reset_( lwi->set_O[nr] );
			SS_Reset_( lwi->set_tr_X[nr] );
			SS_Reset_( lwi->set_tr_Y[nr] );
			SS_Reset_( lwi->set_tr_E[nr] );
			lwi->set_tr_O[nr].Nvalues= 0;
			xfree( lwi->set_tr_O[nr].sample);
			SAS_Reset_( lwi->set_tr_O[nr] );
			WL= WL->next;
		}
}
	
int CleanUp_Sets()
{  int i, n= 0;
	if( setNumber>= MaxSets-1 ){
		i= MaxSets-1;
	}
	else{
		i= setNumber- 1;
	}
	while( i>= 0 && AllSets[i].numPoints<= 0 ){
#ifdef DEBUG
		fprintf( StdErr, "CleanUp_Sets(): #%d->numPoints=%d", i, AllSets[i].numPoints );
#endif
		if( AllSets[i].numPoints< 0 ){
			AllSets[i].numPoints= 0;
			setNumber-= 1;
			n+= 1;
#ifdef DEBUG
			fprintf( StdErr, ", set to #%d->numPoints=%d, setNumber now %d\n", i, AllSets[i].numPoints, setNumber );
#endif
		}
		i--;
	}
#ifdef DEBUG
	if( n ){
		fprintf( StdErr, "CleanUp_Sets(): %d replaced to free pool\n", n );
	}
#endif
	return(n);
}

int SwapDataSet( int set_a, int set_b, int do_redraw )
{ DataSet *A= NULL, *B= NULL;
  UserLabel *ul;
  LocalWin *lwi;
  LocalWindows *WL;
  char msg[512];
	if( set_a< 0 || set_a>= setNumber || set_b< 0 || set_b>= setNumber ||
		set_a== set_b
	){
		return(0);
	}
	A= &AllSets[set_a];
	B= &AllSets[set_b];
	  /* If these sets are linked to, the linking sets must be updated	*/
	if( A->links ){
	  DataSet *that_set;
	  int i;
		for( i= 0; i< setNumber; i++ ){
			if( (that_set= &AllSets[i])->set_link== A->set_nr ){
				that_set->set_link= B->set_nr;
			}
		}
	}
	if( B->links ){
	  DataSet *that_set;
	  int i;
		for( i= 0; i< setNumber; i++ ){
			if( (that_set= &AllSets[i])->set_link== B->set_nr ){
				that_set->set_link= A->set_nr;
			}
		}
	}
	  /* Now, we can swap the set_nr fields, so that after swapping the sets,
	   \ *they* have not changed. That is, the set_nr fields in an array of DataSet
	   \ should increase monotonically before and after the swap.
	   */
	SWAP( A->set_nr, B->set_nr, int );
	  /* Now, swap the sets. This is a pure exchange of memory, so no reallocation
	   \ needs to be done, and no distinction will have to be made between "static"
	   \ data and allocated data (or so I hope).
	   */
	SWAP( *A, *B, DataSet );
	  /* Now, we will have to go through all windows, to exchange the information that
	   \ they store...
	   */
	sprintf( msg, "Swapping set-information %d with %d", set_a, set_b );
	WL= WindowList;
	while( WL ){
		lwi= WL->wi;

		if( debugFlag ){
			TitleMessage( lwi, msg );
		}

		lwi->sets_reordered= True;

		SWAP( lwi->draw_set[set_a], lwi->draw_set[set_b], short );
		SWAP( lwi->mark_set[set_a], lwi->mark_set[set_b], short );

		  /* I don't know whether it is a good idea to swap this info too;
		   \ just do it for the time being.
		   */
		SWAP( lwi->group[set_a], lwi->group[set_b], short );
		SWAP( lwi->fileNumber[set_a], lwi->fileNumber[set_b], short );
		SWAP( lwi->new_file[set_a], lwi->new_file[set_b], short );

		if( lwi->plot_only_set0== set_a ){
			lwi->plot_only_set0= set_b;
		}
		else if( lwi->plot_only_set0== set_b ){
			lwi->plot_only_set0= set_a;
		}
		  /* Something may need to be done with lwi->plot_only_set[], but not just
		   \ simply swapping the elements!!
		   */
		SWAP( lwi->numVisible[set_a], lwi->numVisible[set_b], int );
		SWAP( lwi->legend_line[set_a], lwi->legend_line[set_b], LegendLine );
		SWAP( lwi->xcol[set_a], lwi->xcol[set_b], int );
		SWAP( lwi->ycol[set_a], lwi->ycol[set_b], int );
		SWAP( lwi->ecol[set_a], lwi->ecol[set_b], int );
		SWAP( lwi->lcol[set_a], lwi->lcol[set_b], int );
		SWAP( lwi->error_type[set_a], lwi->error_type[set_b], int );
		  /* Caution: wi->discardpoint is expected to be of type char**, with each
		   \ element allocated independently (an array of char*).
		   */
		if( lwi->discardpoint ){
			SWAP( lwi->discardpoint[set_a], lwi->discardpoint[set_b], char* );
		}
		  /* Same applies to curve_len, error_len, etc.
		   \ They must be interchanged because each set has an array in these fields,
		   \ the length of which depends on the number of points in the set...
		   */
		if( lwi->pointVisible ){
		  /* 20050212: we shouldn't have forgotten this one! */
			SWAP( lwi->pointVisible[set_a], lwi->pointVisible[set_b], signed char* );
		}
		if( lwi->curve_len ){
			SWAP( lwi->curve_len[set_a], lwi->curve_len[set_b], double* );
		}
		if( lwi->error_len ){
			SWAP( lwi->error_len[set_a], lwi->error_len[set_b], double* );
		}
#ifdef TR_CURVE_LEN
		if( lwi->tr_curve_len ){
			SWAP( lwi->tr_curve_len[set_a], lwi->tr_curve_len[set_b], double* );
		}
#endif
		  /* Also swap the statistics bins. They can be in exact mode, in which case
		   \ there are arrays allocated within them.
		   */
		SWAP( lwi->set_X[set_a], lwi->set_X[set_b], SimpleStats );
		SWAP( lwi->set_Y[set_a], lwi->set_Y[set_b], SimpleStats );
		SWAP( lwi->set_E[set_a], lwi->set_E[set_b], SimpleStats );
		SWAP( lwi->set_V[set_a], lwi->set_V[set_b], SimpleStats );
		SWAP( lwi->set_O[set_a], lwi->set_O[set_b], SimpleAngleStats );
		SWAP( lwi->set_tr_X[set_a], lwi->set_tr_X[set_b], SimpleStats );
		SWAP( lwi->set_tr_Y[set_a], lwi->set_tr_Y[set_b], SimpleStats );
		SWAP( lwi->set_tr_E[set_a], lwi->set_tr_E[set_b], SimpleStats );
		SWAP( lwi->set_tr_V[set_a], lwi->set_tr_V[set_b], SimpleStats );
		SWAP( lwi->set_tr_O[set_a], lwi->set_tr_O[set_b], SimpleAngleStats );

		ul= lwi->ulabel;
		while( ul ){
			if( ul->set_link== set_a ){
				ul->set_link= set_b;
			}
			else if( ul->set_link== set_b ){
				ul->set_link= set_b;
			}
			ul= ul->next;
		}

		if( debugFlag ){
			TitleMessage( lwi, NULL );
		}

		WL= WL->next;
	}
	  /* Mark the sets to be redrawn in all windows. Do the redraw
	   \ only for the 2nd set... :)
	   */
	RedrawSet( set_a, False );
	RedrawSet( set_b, do_redraw );
	return(2);
}

int ShiftDataSet( int dsn, int dirn, int to_extreme, int do_redraw )
{ int new= -1;
	if( dsn>= 0 && dsn< setNumber ){
	  char msg[128];
		if( to_extreme ){
			if( dirn< 0 && dsn> 0 ){
				new= dsn- 1;
				do{
					sprintf( msg, "Swapping set %d with %d", dsn, new );
					if( SD_Dialog.mapped ){
						XStoreName(disp, SD_Dialog.win, msg );
						if( !RemoteConnection ){
							XFlush( disp );
						}
					}
					SwapDataSet( new, dsn, (new)? False : do_redraw );
					new-= 1;
					dsn-= 1;
				} while( new>= 0 );
				new= 0;
			}
			else if( dirn> 0 && dsn< setNumber- 1){
				new= dsn+ 1;
				do{
					sprintf( msg, "Swapping set %d with %d", dsn, new );
					if( SD_Dialog.mapped ){
						XStoreName(disp, SD_Dialog.win, msg );
						if( !RemoteConnection ){
							XFlush( disp );
						}
					}
					SwapDataSet( new, dsn, (new== setNumber-1)? do_redraw : False );
					new+= 1;
					dsn+= 1;
				} while( new< setNumber );
				new= setNumber- 1;
			}
			else{
				Boing( -75);
			}
		}
		else{
			if( dirn< 0&& dsn> 0 ){
				new= dsn- 1;
			}
			else if( dirn> 0 && dsn< setNumber- 1){
				new= dsn+ 1;
			}
			else{
				Boing( -75);
			}
			if( new!= -1 ){
				sprintf( msg, "Swapping set %d with %d", dsn, new );
				if( SD_Dialog.mapped ){
					XStoreName(disp, SD_Dialog.win, msg );
					if( !RemoteConnection ){
						XFlush( disp );
					}
				}
				SwapDataSet( new, dsn, do_redraw );
			}
		}
	}
	return(new);
}

  /* Shift all sets that are marked as drawn in their draw_set field.
   \ This field is normally only used during startup to determine which
   \ sets to draw initially; in further normal use, the window's draw_set
   \ field is used instead. This frees up the set->draw_set field for use
   \ by this function. Initialise these fields in the desired way (to reflect
   \ the target window's drawn sets, or highlighted sets, or ...), and call
   \ ShiftDataSets_Drawn() with the same arguments as those used for 
   \ ShiftDataSet().
   */
int ShiftDataSets_Drawn( LocalWin *lwi, int dirn, int extreme, int do_redraw )
{ int extr= False, i, n= 0, first_set, last_set= -1;
	for( i= 0; i< setNumber; i++ ){
		if( AllSets[i].draw_set ){
			n+= 1;
		}
	}
	if( n> 1 && extreme ){
		extr= extreme;
		extreme= False;
	}
	if( dirn< 0 ){
		do{
			first_set= -1;
			for( i= 0; i< setNumber; i++ ){
				if( AllSets[i].draw_set ){
					last_set= ShiftDataSet( i, dirn, extreme, False );
					if( first_set== -1 ){
						first_set= last_set;
					}
				}
			}
		} while( extr && (first_set> 0) );
	}
	else if( dirn> 0 ){
		do{
			first_set= -1;
			for( i= setNumber- 1; i>= 0; i-- ){
				if( AllSets[i].draw_set ){
					last_set= ShiftDataSet( i, dirn, extreme, False );
					if( first_set== -1 ){
						first_set= last_set;
					}
				}
			}
		} while( extr && (last_set>= 0 && first_set< setNumber- 1) );
	}
	if( last_set>= 0 ){
		RedrawSet( last_set, do_redraw );
		return( n );
	}
	else{
		return( -n );
	}
}

int LinkSet2( DataSet *this_set, int set_link )
{
	if( set_link== this_set->set_nr ){
		return(-1);
	}
	if( set_link>= 0 && set_link< setNumber ){
	  DataSet *that_set= &AllSets[set_link];
		if( this_set->columns && this_set->set_link< 0 ){
			Destroy_Set( this_set, True );
		}
		else if( this_set->set_link>= 0 && this_set->set_link< setNumber ){
			AllSets[this_set->set_link].links-= 1;
		}
		this_set->set_link= set_link;
		this_set->set_linked= True;
		this_set->ncols= that_set->ncols;
		this_set->columns= that_set->columns;
		this_set->numPoints= that_set->numPoints;
		this_set->has_error= that_set->has_error;
		this_set->numErrors= that_set->numErrors;
		  /* 20040301: */
		this_set->fileNumber= that_set->fileNumber;
		realloc_points( this_set, that_set->allocSize, True );
		that_set->links+= 1;
	}
	else if( set_link< 0 ){
		  /* 20050109: maybe do a full destroy here. */
		Destroy_Set( this_set, True );
	}
	else{
		this_set->set_link= set_link;
		this_set->set_linked= False;
	}
	CleanUp_Sets();
	return(this_set->set_link);
}

int legend_setNumber= -1;
int AddPoint_discard;

int Raw_NewSets= 0, CorrectLinks= True;

extern int StartUp, MacStartUp;
extern double *param_scratch;
extern int param_scratch_len;

/* 20010824: procedure partly adapted for lcol implementation. */
int AddPoint( DataSet **This_Set, int *spot, int *Spot, int numcoords, double *Data, int column[ASCANF_DATA_COLUMNS],
	char *filename, int sub_div, int line_count, FileLinePos *flpos, char *buffer, Process *proc
)
{  double yerror;
   double xvec, ldxvec, hdxvec, yvec, ldyvec, hdyvec;
   static double real_minY, real_maxY;
   int i, spot_ok, set_link= -1;
   DataSet *this_set= *This_Set, *that_set;
   static double delta_x, previous_x;
   static double *data= NULL;
   static int DataLen= 0;

       /* Set name of set to file name if appropriate */
    if( !this_set->setName ){
		if( filename ){
			this_set->setName = XGstrdup(filename);
		}
		else{
		  char setname[64];
			sprintf( setname, "Set %d", this_set->set_nr );
			this_set->setName = XGstrdup(setname);
		}
    }
	if( !this_set->fileName || (strcmp( this_set->fileName, filename) && this_set->numPoints<=0) ){
		xfree( this_set->fileName );
		this_set->fileName = XGstrdup(filename);
	}

	if( !data || !DataLen ){
		data= (double*) calloc( this_set->ncols, sizeof(double) );
		DataLen= this_set->ncols;
	}
	else if( this_set->ncols> DataLen ){
		data= (double*) realloc( (char*) data, this_set->ncols* sizeof(double) );
		DataLen= this_set->ncols;
	}

	data[0]= Data[column[0]];
	data[1]= Data[column[1]];
/* 	data[2]= (numcoords> column[2])? Data[column[2]] : 0;	*/
	data[2]= Data[column[2]];
	if( this_set->ncols> 3 && column[3]>= 0 && numcoords> column[3] ){
		data[3]= Data[column[3]];
	}
	  /* copy the rest of the columns	*/
	{ int i;
		for( i= 4; i< this_set->ncols; i++ ){
			data[i]= Data[i];
		}
	}

	  /* 20020918: */
	if( *spot< 0 ){
		*spot= 0;
	}
	if( *Spot< 0 ){
		*Spot= 0;
	}

	  /* 941212: dataprocessing used to take place before the call to
	   \ AddPoint. Now it is done in AddPoint(), so it takes places
	   \ before all other business to ensure the same functionality.
	   \ Advantage: all data read or generated can be processed...
	   */
	if( proc->data_process && proc->data_process_now ){
	  int n, ok, rsacsv= reset_ascanf_currentself_value;
		clean_param_scratch();
		{
/* 
#ifdef __GNUC__
		  char legend[proc->data_init_len+proc->data_before_len+proc->data_after_len+proc->data_finish_len+proc->data_process_len+128];
#else
		  char legend[4*(LMAXBUFSIZE+1)+128];
#endif
 */
			if( legend_setNumber!= setNumber ){
			  /* if this set doesn't have a name yet,
			   \ baptise it
			   */
			  char *legend;
			  int legend_len = proc->data_init_len + proc->data_before_len + proc->data_after_len
			  				+ proc->data_finish_len + proc->data_process_len + 128;
				xfree( this_set->setName );
				if( (legend = (char*) malloc( legend_len * sizeof(char) )) ){
					sprintf( legend, "DATA processing:\n%s%s %s%sx y e=%s%s%s %s%s",
						(proc->data_init)? "Init=" : "",
						(proc->data_init)? proc->data_init : "",
						(proc->data_before)? "Before=" : "",
						(proc->data_before)? proc->data_before : "",
						proc->data_process,
						(proc->data_after)? "After=" : "",
						(proc->data_after)? proc->data_after : "",
						(proc->data_finish)? "Finish=" : "",
						(proc->data_finish)? proc->data_finish : ""
					);
					this_set->setName= legend;
				}
				else{
					this_set->setName = XGstrdup("untitled");
				}
				this_set->draw_set= 1;
				this_set->show_legend= 1;
				this_set->show_llines= 1;
				legend_setNumber= setNumber;
			}
		}
		ok= 1;
		{ char change[ASCANF_DATA_COLUMNS];
		  double _data[ASCANF_DATA_COLUMNS];
		  int ncols;
			_data[0]= data[this_set->xcol];
			_data[1]= data[this_set->ycol];
			_data[2]= data[this_set->ecol];
			if( this_set->lcol>= 0 ){
				_data[3]= data[this_set->lcol];
				ncols= 4;
			}
			else{
				ncols= 3;
			}
			if( proc->data_init && *spot== 0 && AddPoint_discard<= 0 ){
				n= param_scratch_len;
				*ascanf_self_value= (double) *spot;
				*ascanf_current_value= (double) *spot;
				*ascanf_counter= *spot;
				*ascanf_Counter= *Spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "DATA Init: %s", proc->data_init);
					fflush( StdErr );
				}
				compiled_fascanf( &n, proc->data_init, param_scratch, NULL, _data, column, &proc->C_data_init );
				if( ascanf_arg_error || !n ){
					ok= 0;
				}
			}
			if( ok && proc->data_before && AddPoint_discard<= 0 ){
				n= param_scratch_len;
				*ascanf_self_value= (double) *spot;
				*ascanf_current_value= (double) *spot;
				*ascanf_counter= *spot;
				*ascanf_Counter= *Spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "DATA Before: %s", proc->data_before);
					fflush( StdErr );
				}
/* 				fascanf( &n, proc->data_before, param_scratch, NULL, _data, column, NULL );	*/
				compiled_fascanf( &n, proc->data_before, param_scratch, NULL, _data, column, &proc->C_data_before );
				if( ascanf_arg_error || !n ){
					ok= 0;
				}
			}
			if( ok && AddPoint_discard<= 0 ){
				n= ncols;
				*ascanf_self_value= (double) *spot;
				*ascanf_current_value= (double) *spot;
				*ascanf_counter= *spot;
				(*ascanf_Counter)= *Spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "DATA x y e: %s", proc->data_process );
					fflush( StdErr );
				}
/* 				fascanf( &n, proc->data_process, _data, change, _data, column, NULL );	*/
				compiled_fascanf( &n, proc->data_process, _data, change, _data, column, &proc->C_data_process );
				if( !ascanf_arg_error && n &&
					(change[0]== 'N' || change[0]== 'R') &&
					(change[1]== 'N' || change[1]== 'R')
				){
					ok= 1;
				}
				else{
					ok= 0;
				}
			}
			if( ok && proc->data_after && AddPoint_discard<= 0 ){
				n= param_scratch_len;
				*ascanf_self_value= (double) *spot;
				*ascanf_current_value= (double) *spot;
				*ascanf_counter= *spot;
				(*ascanf_Counter)= *Spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "DATA After: %s", proc->data_after );
					fflush( StdErr );
				}
/* 				fascanf( &n, proc->data_after, param_scratch, NULL, _data, column, NULL );	*/
				compiled_fascanf( &n, proc->data_after, param_scratch, NULL, _data, column, &proc->C_data_after );
				if( ascanf_arg_error || !n ){
					ok= 0;
				}
			}
			  /* 990912: either this one is executed always, or never...	*/
			if( ok && proc->data_finish && *spot== this_set->numPoints-1 && AddPoint_discard<= 0 ){
				n= param_scratch_len;
				*ascanf_self_value= (double) *spot;
				*ascanf_current_value= (double) *spot;
				*ascanf_counter= *spot;
				(*ascanf_Counter)= *Spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "DATA Finish: %s", proc->data_finish );
					fflush( StdErr );
				}
				compiled_fascanf( &n, proc->data_finish, param_scratch, NULL, _data, column, &proc->C_data_finish );
				if( ascanf_arg_error || !n ){
					ok= 0;
				}
			}
			data[this_set->xcol]= _data[0];
			data[this_set->ycol]= _data[1];
			data[this_set->ecol]= _data[2];
			if( this_set->lcol>= 0 ){
				data[this_set->lcol]= _data[3];
			}
		}
		reset_ascanf_currentself_value= rsacsv;
		fflush( StdErr );
		if( !ok ){
			fprintf( StdErr, "\"%s\": error\n", buffer );
			fflush( StdErr );
		}
	}
	if( AddPoint_discard> 0 ){
		(*Spot)++;
		return(0);
	}

	if( (*This_Set)->set_link>= 0 ){
		if( (*This_Set)->set_link>= setNumber ){
			fprintf( StdErr, "AddPoint(#%d,%d): set points to future set #%d; points added now get lost!\n",
				(*This_Set)->set_nr, *spot, (*This_Set)->set_link
			);
			return(0);
		}
		else{
			this_set= &AllSets[ (set_link= (*This_Set)->set_link) ];
/* 			*spot+= this_set->numPoints;	*/
/* 			*Spot+= this_set->numPoints;	*/
		}
	}
	that_set= this_set;
	if( split_set> 0 || *spot!= 0 ){
	  double diff= data[this_set->xcol] - previous_x;
		  /* Now we can determine a new delta_x, and
		   \ if it changes sign, create a new dataset
		   \ if so permitted.
		   */
		if( set_link>= 0 ){
			delta_x= diff;
		}
		else if( (split_set> 0 && *spot && splits_disconnect) || (disconnect && SIGN(diff) != SIGN(delta_x)) ){
		  DataSet *new_set, *os= *This_Set;
		  int sn= setNumber;

			if( NewSet( NULL, This_Set, *spot )!= -1 ){
				new_set= *This_Set;
				if( setNumber> sn || os!= new_set ){
					if( WarnNewSet ){
						if( flpos ){
							line_count= UpdateLineCount( flpos->stream, flpos, line_count );
						}
						fprintf( StdErr, "#---- Starting new set #%d; file \"%s\", line %d.%d (AddPoint())\n",
							setNumber, filename, line_count, sub_div
						);
					}
				}
				*spot= 0;
				*Spot= 0;
				xfree( new_set->setName );
				new_set->setName= XGstrdup( this_set->setName);
				legend_setNumber= setNumber;
				new_set->appendName= XGstrdup( parse_codes(split_reason) );
				new_set->fileName= XGstrdup( this_set->fileName );
				new_set->fileNumber= this_set->fileNumber;
				new_set->titleText= XGstrdup( this_set->titleText );
				new_set->show_legend= this_set->show_legend;
				new_set->show_llines= this_set->show_llines;
				new_set->draw_set= this_set->draw_set;
				new_set->has_error= this_set->has_error;
				new_set->use_error= this_set->use_error;
				new_set->lineWidth= this_set->lineWidth;
				new_set->elineWidth= this_set->elineWidth;
				new_set->markFlag= this_set->markFlag;
				new_set->noLines= this_set->noLines;
				new_set->floating= False;
				new_set->pixelMarks= this_set->pixelMarks;
				new_set->barFlag= this_set->barFlag;
				new_set->polarFlag= this_set->polarFlag;
				new_set->arrows= this_set->arrows;
				new_set->overwrite_marks= this_set->overwrite_marks;
				new_set->radix= this_set->radix;
				new_set->radix_offset= this_set->radix_offset;
				new_set->Xscale= this_set->Xscale;
				new_set->Yscale= this_set->Yscale;
				new_set->DYscale= this_set->DYscale;
				new_set->NumObs= this_set->NumObs;
				new_set->vectorLength= this_set->vectorLength;
				new_set->vectorType= this_set->vectorType;
				memcpy( new_set->vectorPars, this_set->vectorPars, MAX_VECPARS* sizeof(double));
				new_set->ncols= this_set->ncols;
				new_set->xcol= this_set->xcol;
				new_set->ycol= this_set->ycol;
				new_set->ecol= this_set->ecol;
				new_set->lcol= this_set->lcol;
				new_set->Ncol= this_set->Ncol;
				new_set->error_type= this_set->error_type;
				if( that_set== this_set ){
				  /* No reference to a linked set	*/
					that_set= *This_Set;
				}
				this_set= *This_Set;
				if( debugFlag ){
					if( flpos ){
						line_count= UpdateLineCount( flpos->stream, flpos, line_count );
					}
					fprintf( StdErr,
						"AddPoint: disconnected set #%d from previous (%s)\n"
						"\tfile \"%s\" (%s), line %d.%d: \"%s\"\n",
						setNumber, (split_reason)? split_reason : "",
						filename, this_set->fileName, 
						sub_div, line_count, buffer
					);
					fflush( StdErr );
				}
				xfree( split_reason );
				split_set= 0;
			}
		}
		else if( split_set> 0 && !*spot && splits_disconnect ){
		  /* we don't really split an as-yet empty set.
		   \ we do however update the appendName field
		   \ (really appending...)
		   */
			parse_codes(split_reason);
			if( this_set->appendName ){
				this_set->appendName= concat2( this_set->appendName, split_reason, NULL );
			}
			else{
				this_set->appendName= XGstrdup( split_reason );
			}
			xfree( split_reason );
			split_set= 0;
		}
		else if( split_set> 0 && !splits_disconnect ){
		  /* later, after (potential) call to realloc_points!!!
		   \ This case here only to prevent update of delta_x.
		   */
		}
		else{
			delta_x= diff;
		}
	}
	else if( *spot== 1 ){
	  /* first time we can determine delta_x	*/
		delta_x= data[this_set->xcol] - previous_x;
	}

	previous_x= data[this_set->xcol];

			if (*spot >= this_set->allocSize) {
			  /* Time to make the arrays bigger (or initialize them) */
				if (this_set->allocSize == 0) {
					realloc_points( this_set, INITSIZE, False );
				} else {
					realloc_points( this_set, this_set->allocSize* 2, False );
				}
			}

	if( split_set> 0 && !splits_disconnect ){
	  /* Just record the split-command, interpreted as a "lift pen" command. Don't care for
	   \ the value of *spot here.
	   */
		if( !this_set->splithere ){
		  /* We just allocate as much space as needed here. If ever points are added to this set
		   \ (by loading additional files), realloc_points will take care of increasing the arena.
		   */
			this_set->splithere= (signed char*) calloc( this_set->allocSize, sizeof(signed char));
			this_set->mem_alloced+= this_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
			mem_alloced+= this_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
		}
		if( this_set->splithere ){
		  /* if we got the memory, swap the split-indicator at this point.	*/
			this_set->splithere[*spot]= ! this_set->splithere[*spot];
		}
		xfree( split_reason );
		split_set= 0;
	}

	if( split_set< 0 ){
	  /* split after the point currently being added	*/
		split_set= 1;
	}

			spot_ok= 1;

			_MXscale= MAX( MXscale, Xscale);
			_MYscale= MAX( MYscale, Yscale);
			_MDYscale= MAX( MDYscale, DYscale);

#ifdef NO_FILE_SCALING
			this_set->Xscale= Xscale;
			this_set->Yscale= Yscale;
			this_set->DYscale= DYscale;
#	define Xscale	1.0
#	define Yscale	1.0
#	define DYscale	1.0
#else
			this_set->Xscale= 1;
			this_set->Yscale= 1;
			this_set->DYscale= 1;
#endif

			for( i= 0; i< this_set->ncols; i++ ){
				this_set->columns[i][*spot]= data[i];
			}

			if( set_link>= 0 ){
				  /* For This_Set, columns points to another set's columns...	*/
				(*This_Set)->columns= this_set->columns;
				this_set= (*This_Set);
			}

			  /* Pfff.... the read-time scalefactors are applied only to the specified
			   \ X, Y and Error columns. I don't see another option...
			   */
			i= this_set->xcol;
				this_set->xvec[*spot]= (this_set->columns[i][*spot]*= Xscale);
				SS_Add_Data_( SS_x, 1, data[i], 1.0);
				SS_Add_Data_( SS__x, 1, data[i], 1.0);
			i= this_set->ycol;
				this_set->yvec[*spot]= (this_set->columns[i][*spot]*= Yscale);
				SS_Add_Data_( SS_y, 1, data[i], 1.0);
				SS_Add_Data_( SS__y, 1, data[i], 1.0);
			if( vectorFlag ){
				this_set->ldxvec[*spot]= this_set->xvec[*spot];
				if( !NaNorInf(data[this_set->ecol]) ){
					this_set->hdxvec[*spot]= this_set->xvec[*spot]+ this_set->vectorLength* Cos(data[this_set->ecol]);
				}
				else{
					this_set->hdxvec[*spot]= this_set->xvec[*spot];
				}
				yerror= data[this_set->ecol];
			}
			else{
				this_set->ldxvec[*spot]= this_set->hdxvec[*spot]= this_set->xvec[*spot];
				yerror= data[this_set->ecol]* DYscale;
			}
			SS_Add_Data_( SS_e, 1, (data[this_set->ecol]), 1.0);
			SS_Add_Data_( SS__e, 1, (data[this_set->ecol]), 1.0);
			if( numcoords>= 3 /* && this_set->has_error */ ){
				if( vectorFlag ){
					this_set->ldyvec[*spot]= this_set->yvec[*spot];
					if( !NaNorInf(yerror) ){
						this_set->hdyvec[*spot]= this_set->yvec[*spot] + this_set->vectorLength* Sin(yerror);
					}
					else{
						this_set->hdyvec[*spot]= this_set->yvec[*spot];
					}
				}
				else{
					this_set->ldyvec[*spot]= this_set->yvec[*spot] - yerror;
					this_set->hdyvec[*spot]= this_set->yvec[*spot] + yerror;
				}
				this_set->has_error= 1;
				this_set->numErrors++;
			}
			else{
/* 				yerror = 0.0;	*/
				set_NaN(yerror);
				this_set->ldyvec[*spot]= this_set->hdyvec[*spot]= this_set->yvec[*spot];
/* 				this_set->has_error= 0;	*/
			}
			if( this_set->ecol!= this_set->ycol && this_set->ecol!= this_set->xcol 
				&& this_set->ecol!= this_set->lcol && this_set->ecol!= this_set->Ncol
			){
			  /* 20051103: Do NOT modify the columns structure when ecol points to another of the pre-defined columns!! */
				this_set->columns[this_set->ecol][*spot]= yerror;
			}
			this_set->data[0][2]= yerror;

#if ADVANCED_STATS == 1
			  /* Initially, we just assume all points are based on this_set->NumObs observations:	*/
			this_set->N[*spot]= this_set->NumObs;
#endif

			  /* This calculates the set-average and std.dev over the Y&E values	*/
			if( this_set->NumObs> 0 ){
			  double sum= data[this_set->ycol]* this_set->NumObs;
			  double sum_sqr= sum* sum/this_set->NumObs + (data[this_set->ecol]* data[this_set->ecol])* (this_set->NumObs-1);
				SS_sy.min= (SS_sy.count)? MIN( SS_sy.min, this_set->ldyvec[*spot]) : this_set->ldyvec[*spot];
				SS_sy.max= (SS_sy.count)? MAX( SS_sy.max, this_set->hdyvec[*spot]) : this_set->hdyvec[*spot];
				SS_sy.count+= (long) this_set->NumObs;
				SS_sy.weight_sum+= (double) this_set->NumObs;
				SS_sy.sum+= sum;
				SS_sy.sum_sqr+= sum_sqr;

				  /* And the overall average	*/
				SS_SY.min= (SS_SY.count)? MIN( SS_SY.min, this_set->ldyvec[*spot]) : this_set->ldyvec[*spot];
				SS_SY.max= (SS_SY.count)? MAX( SS_SY.max, this_set->hdyvec[*spot]) : this_set->hdyvec[*spot];
				SS_SY.count+= (long) this_set->NumObs;
				SS_SY.weight_sum+= (double) this_set->NumObs;
				SS_SY.sum+= sum;
				SS_SY.sum_sqr+= sum_sqr;
			}

#undef Xscale
#undef Yscale
#undef DYscale

			this_set->data[0][0]= xvec= this_set->xvec[*spot];
			ldxvec= this_set->ldxvec[*spot];
			hdxvec= this_set->hdxvec[*spot];
			this_set->data[0][1]= yvec= this_set->yvec[*spot];
			ldyvec= this_set->ldyvec[*spot];
			hdyvec= this_set->hdyvec[*spot];

			if( *spot== 0 ){
				real_minY= yvec - yerror;
				real_maxY= yvec + yerror;
				this_set->lowest_y.x= xvec;
				this_set->lowest_y.y= yvec;
				this_set->lowest_y.errr= yerror;
				this_set->highest_y.x= xvec;
				this_set->highest_y.y= yvec;
				this_set->highest_y.errr= yerror;
			}

#ifndef NO_FILE_SCALING
#	define Xscale	1.0
#	define Yscale	1.0
#	define DYscale	1.0
#endif

			if( autoscale ){
			  double minX= xvec, minpX= xvec, minY= yvec, minpY= yvec, maxnY= yvec;
			  double maxX= xvec, maxY= yvec;
			  double
				real_ldxvec= xvec* Xscale,
				real_hdxvec= xvec* Xscale,
				real_ldyvec= yvec* Yscale - DYscale * yerror,
				real_hdyvec= yvec* Yscale + DYscale * yerror;
			  double minx= MIN(real_ldxvec, real_hdxvec),
					maxx= MAX(real_ldxvec, real_hdxvec),
				    miny= MIN(real_ldyvec, real_hdyvec),
					maxy= MAX(real_ldyvec, real_hdyvec);

				if( NaNorInf(real_ldxvec) || NaNorInf(real_hdxvec) ||
					NaNorInf(real_ldyvec) || NaNorInf(real_hdyvec) ||
					this_set->floating
				){
					  /* I hate using break or continue...
					   \ always mess'm up, and never know
					   \ where I'll end up (except of course
					   \ for break in a switch body)
					   */
					goto skip_autoscale;
				}
				  /* For the moment it's too tedious to make the scale-factors
				   \ changeable from the running programme. Therefore we can 
				   \ use the scaled values (real_...vec) to determine the overall
				   \ bounds over all datasets. Individual bounds (unscaled!) are
				   \ (voidly) determined in the vertical: this_set->lowest_y and 
				   \ this_set->highest_y. One day this may lead to changeable
				   \ scaling!
				   */
				   /* 20001009: much of the code around this point is probably no longer necessary.
				    \ I should clean it out some day...
					*/
				minX= MIN( minX, real_ldxvec);
				minX= MIN( minX, real_hdxvec);
				minY= MIN( minY, real_ldyvec);
				minY= MIN( minY, real_hdyvec);

				maxnY= MAXNEG( maxnY, real_ldyvec);
				maxnY= MAXNEG( maxnY, real_hdyvec);

				minpX= MINPOS( minpX, real_ldxvec);
				minpX= MINPOS( minpX, real_hdxvec);
				minpY= MINPOS( minpY, real_ldyvec);
				minpY= MINPOS( minpY, real_hdyvec);

				maxX= MAX( maxX, real_ldxvec);
				maxX= MAX( maxX, real_hdxvec);
				maxY= MAX( maxY, real_ldyvec);
				maxY= MAX( maxY, real_hdyvec);

				if( yvec- yerror< real_minY ){
					real_minY= yvec- yerror;
					this_set->lowest_y.x= xvec;
					this_set->lowest_y.y= yvec;
					this_set->lowest_y.errr= yerror;
				}
				else if( yvec+ yerror> real_maxY ){
					real_maxY= yvec+ yerror;
					this_set->highest_y.x= xvec;
					this_set->highest_y.y= yvec;
					this_set->highest_y.errr= yerror;
				}

				if( minpX > 0 ){
					if( xp_scale_start ){
						llpx= minpX;
						xp_scale_start= 0;
					}
					else{
						if( minpX< llpx )
							llpx= minpX;
					}
				}
				if( x_scale_start ){
					llx= minX;
					urx= maxX;
					x_scale_start= 0;
					real_min_x= minx;
					real_max_x= maxx;
				}
				else{
					if( minX < llx )
					   llx  = minX;
					if( maxX > urx )
					   urx = maxX;
					if( minx< real_min_x ){
						real_min_x= minx;
						MXscale= Xscale;
					}
					if( maxx> real_max_x ){
						real_max_x= maxx;
						MXscale= Xscale;
					}
				}

				  /* maximum negative Y	or 0 */
				if( maxnY<= 0 ){
					if( yn_scale_start ){
						llny= maxnY;
						yn_scale_start= 0;
					}
					else{
						if( maxnY> llny )
							llny= maxnY;
					}
				}
				  /* minimum positive Y	*/
				if( minpY> 0 ){
					if( yp_scale_start ){
						llpy= minpY;
						yp_scale_start= 0;
					}
					else{
						if( minpY< llpy )
							llpy= minpY;
					}
				}
				if( y_scale_start ){
					lly= minY;
					ury= maxY;
					y_scale_start= 0;
					real_min_y= miny;
					real_max_y= maxy;
				}
				else{
					if( minY < lly)
					   lly = minY;
					if( maxY > ury)
					   ury = maxY;
					if( miny< real_min_y ){
						real_min_y= miny;
						MYscale= Yscale;
						MDYscale= DYscale;
					}
					if( maxy> real_max_y ){
						real_max_y= maxy;
						MYscale= Yscale;
						MDYscale= DYscale;
					}
				}

				if( !use_lx ){
					startup_wi.R_UsrOrgX= llx;
					usrLpX= llpx;
					startup_wi.R_UsrOppX= urx;
				}
				if( !use_ly ){
					startup_wi.R_UsrOrgY= lly;
					usrLpY= llpy;
					usrHnY= llny;
					startup_wi.R_UsrOppY= ury;
				}
			}
skip_autoscale:;
			SS_Add_Data_( SS_X, 1, xvec, 1.0);
			SS_Add_Data_( SS_Y, 1, yvec, 1.0);

#undef Xscale
#undef Yscale
#undef DYscale

			spot_ok= 1;
			{  char number[64];
			   double lcount;
				sprintf( number, "%d.%d", sub_div, line_count );
				sscanf( number, "%lf", &lcount );
				if( *spot ){
					do_transform( NULL,
						filename, lcount, buffer, &spot_ok, this_set, &xvec, &ldxvec, &hdxvec, &yvec, &ldyvec, &hdyvec,
						&this_set->xvec[*spot-1], &this_set->yvec[*spot-1], 1, *spot, 1.0, 1.0, 1.0, 0, 0, 0
					);
				}
				else{
					do_transform( NULL,
						filename, lcount, buffer, &spot_ok, this_set, &xvec, &ldxvec, &hdxvec, &yvec, &ldyvec, &hdyvec,
						NULL, NULL, 1, *spot, 1.0, 1.0, 1.0, 0, 0, 0
					);
				}
			}

			(*spot)++;
			(*Spot)++;
			that_set->numPoints += 1;
			  /* the following line only has a "real" effect when this_set!= that_set, i.e. when set_link>=0	*/
			this_set->numPoints= that_set->numPoints;
			this_set->has_error= that_set->has_error;
			this_set->numErrors= that_set->numErrors;
			*ascanf_numPoints= this_set->numPoints;
			if( !StartUp ){
				that_set->points_added+= 1;
			}

			if( Raw_NewSets ){
				that_set->raw_display= True;
			}
	GCA();
	return(1);
}

double param_dt= 1.0;

int line_count;

Boolean ReadData_terpri= False;
char *ReadData_thisCurrentFileName= NULL;
FILE *ReadData_thisCurrentFP= NULL;
int ReadData_IgnoreVERSION= 0;

int AddEllipse( DataSet **this_set, double x, double y, double rx, double ry, int points, double skew,
	int *spot, int *Spot, double data[ASCANF_DATA_COLUMNS], int column[ASCANF_DATA_COLUMNS],
	char *filename, int sub_div, int line_count, FileLinePos *flpos, char *buffer, 
	Process *proc
)
{ double a, cos_sk, sin_sk;
   extern double Units_per_Radian;
   int i;
#ifdef DEBUG
   double ax[4][3];
   int axn= -1;
#endif
	  /* if this set doesn't have a name yet,
	   \ baptise it
	   */
	if( legend_setNumber!= setNumber ){
	  char legend[128];
		sprintf( legend, "*ELLIPSE* %g %g %g %g %d %g",
			x, y, rx, ry, points, skew
		);
		xfree( (*this_set)->setName );
		(*this_set)->setName= XGstrdup( legend );
		(*this_set)->draw_set= 1;
		(*this_set)->show_legend= 1;
		(*this_set)->show_llines= 1;
		legend_setNumber= setNumber;
	}
	if( (*this_set)->allocSize< points+ 2 ){
		realloc_points( (*this_set), (*this_set)->numPoints+ points+ 2, False );
	}
	Gonio_Base( NULL, 360.0, 0.0 );
	cos_sk= cos( (skew+ Gonio_Base_Offset)/ Units_per_Radian );
	sin_sk= sin( (skew+ Gonio_Base_Offset)/ Units_per_Radian );
	for( i= 0; i<= points; i++ ){
	  double dx, dy;
		a= 2* M_PI* ((double)i/ (double)points);
		data[0]= x+ (dx= rx* cos(a));
		data[1]= y+ (dy= ry* sin(a));
		data[3]= data[2]= 0.0;
		if( skew ){
		  double ddx= dx* (cos_sk)- dy* sin_sk,
			ddy= dx* (sin_sk)+ dy* cos_sk;
			data[0]= x+ ddx;
			data[1]= y+ ddy;
		}
#ifdef DEBUG
		if( i== 0 ){
			axn+= 1;
			ax[axn][0]= data[0];
			ax[axn][1]= data[1];
			ax[axn][2]= data[2];
		}
		else if( i== points / 4 ){
			axn+= 1;
			ax[axn][0]= data[0];
			ax[axn][1]= data[1];
			ax[axn][2]= data[2];
		}
		else if( i== points / 2 ){
			axn+= 1;
			ax[axn][0]= data[0];
			ax[axn][1]= data[1];
			ax[axn][2]= data[2];
		}
		else if( i== 3* points / 4 ){
			axn+= 1;
			ax[axn][0]= data[0];
			ax[axn][1]= data[1];
			ax[axn][2]= data[2];
		}
#endif
		AddPoint_discard= False;
		AddPoint( this_set, spot, Spot, 3, data, column, filename, sub_div, line_count, flpos, buffer, proc );
	}
#ifdef DEBUG
	if( axn>= 2 ){
		AddPoint_discard= False;
		AddPoint( this_set, spot, Spot, 3, ax[0], column, filename, sub_div, line_count, flpos, buffer, proc );
		AddPoint_discard= False;
		AddPoint( this_set, spot, Spot, 3, ax[2], column, filename, sub_div, line_count, flpos, buffer, proc );
		AddPoint_discard= False;
		AddPoint( this_set, spot, Spot, 3, ax[1], column, filename, sub_div, line_count, flpos, buffer, proc );
		if( axn>= 3 ){
			AddPoint_discard= False;
			AddPoint( this_set, spot, Spot, 3, ax[3], column, filename, sub_div, line_count, flpos, buffer, proc );
		}
		AddPoint_discard= False;
		AddPoint( this_set, spot, Spot, 3, ax[2], column, filename, sub_div, line_count, flpos, buffer, proc );
	}
#endif
	return( i );
}

extern Boolean PIPE_error;

char data_separator= ' ';

int AllocBinaryFields( int nc, char *caller )
{ int ret= 1;
	if( !(BinaryDump.data= (double*) realloc( BinaryDump.data, nc* sizeof(double)))
		|| !(BinaryDump.data4= (float*) realloc( BinaryDump.data4, nc* sizeof(float)))
		|| !(BinaryDump.data2= (unsigned short*) realloc( BinaryDump.data2, nc* sizeof(unsigned short)))
		|| !(BinaryDump.data1= (unsigned char*) realloc( BinaryDump.data1, nc* sizeof(unsigned char)))
	){
		fprintf( StdErr, "%s: can't allocate binary dumpbuffer (%s)\n", caller, serror() );
		fflush( StdErr );
		ret= 0;
	}
	else{
		BinaryFieldSize= nc* sizeof(double)+ sizeof(int);
		BinaryDump.columns= nc;
	}
	if( !(BinaryTerminator.data= (double*) realloc( BinaryTerminator.data, nc* sizeof(double)))
		|| !(BinaryTerminator.data4= (float*) realloc( BinaryTerminator.data4, nc* sizeof(float)))
		|| !(BinaryTerminator.data2= (unsigned short*) realloc( BinaryTerminator.data2, nc* sizeof(unsigned short)))
		|| !(BinaryTerminator.data1= (unsigned char*) realloc( BinaryTerminator.data1, nc* sizeof(unsigned char)))
	){
		fprintf( StdErr, "%s: can't allocate binary dumpbuffer (%s)\n", caller, serror() );
		ret= 0;
	}
	else{
	  int i;
		BinaryTerminator.columns= 0;
		for( i= 0; i< nc; i++ ){
			BinaryTerminator.data[i]= 0;
			BinaryTerminator.data4[i]= 0;
			BinaryTerminator.data2[i]= 0;
			BinaryTerminator.data1[i]= 0;
		}
	}
	return(ret);
}

extern int BinarySize;

void BinaryTerminate(FILE *fp)
{
	fwrite( &BinaryTerminator.columns, sizeof(short), 1, fp );
	if( BinarySize== sizeof(double) ){
		fwrite( BinaryTerminator.data, sizeof(double), BinaryTerminator.columns, fp );
	}
	else if( BinarySize== sizeof(float) ){
		fwrite( BinaryTerminator.data4, sizeof(float), BinaryTerminator.columns, fp );
	}
	else if( BinarySize== sizeof(unsigned short) ){
		fwrite( BinaryTerminator.data2, sizeof(unsigned short), BinaryTerminator.columns, fp );
	}
	else if( BinarySize== sizeof(unsigned char) ){
		fwrite( BinaryTerminator.data1, sizeof(unsigned char), BinaryTerminator.columns, fp );
	}
}

int BinaryDumpData(FILE *fp, int allow_short_bin)
{ int ret;
	if( !allow_short_bin ){
		fwrite( &BinaryDump.columns, sizeof(short), 1, fp );
	}
	if( BinarySize== sizeof(double) ){
		ret= fwrite( BinaryDump.data, sizeof(double), BinaryDump.columns, fp );
	}
	else if( BinarySize== sizeof(float) ){
		ret= fwrite( BinaryDump.data4, sizeof(float), BinaryDump.columns, fp );
	}
	else if( BinarySize== sizeof(unsigned short) ){
		ret= fwrite( BinaryDump.data2, sizeof(unsigned short), BinaryDump.columns, fp );
	}
	else if( BinarySize== sizeof(unsigned char) ){
		ret= fwrite( BinaryDump.data1, sizeof(unsigned char), BinaryDump.columns, fp );
	}
	return(ret);
}

int DumpFile= False;
int DumpIncluded= True;
int DumpPretty= True;
extern int DumpBinary, DumpDHex, DumpPens;

extern int ReadPipe;
extern char *ReadPipe_name;
extern FILE *ReadPipe_fp;

char *next_include_file= NULL, *version_list= NULL;

char *Read_File_Buf= NULL;

extern int new_param_now( char *expression, double *selfvalue, int N );

void set_Columns( DataSet *this_set)
{ LocalWindows *WL= WindowList;
	while( WL ){
	  LocalWin *lwi= WL->wi;
		if( lwi->xcol ){
			lwi->xcol[this_set->set_nr]= this_set->xcol;
			lwi->ycol[this_set->set_nr]= this_set->ycol;
			lwi->ecol[this_set->set_nr]= this_set->ecol;
			lwi->lcol[this_set->set_nr]= this_set->lcol;
		}
		WL= WL->next;
	}
}

XGStringList *Init_Exprs= NULL, **the_init_exprs= NULL, *Startup_Exprs= NULL, *DumpProcessed_commands= NULL, *Dump_commands= NULL;
XGStringList *nextGEN_Startup_Exprs= NULL, *Exit_Exprs= NULL;
int Allow_InitExprs= True;

extern char d3str_format[16];
extern int d3str_format_changed;

int No_IncludeFiles= False;

extern int GetColor( char *name, Pixel *pix);
extern char *ParsedColourName;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

extern int XGStoreColours, XGIgnoreCNames;

ColourFunction IntensityColourFunction;
extern char *TBARprogress_header;
extern int ascanf_propagate_events;

int Free_Intensity_Colours();
int Intensity_Colours( char *exp );

extern int SwapEndian;

int scriptVerbose= 0;

#if defined(__GNUC__) && __GNUC__ < 3
inline
#endif
long long hash64( char *name, unsigned int *hash_len)
{  long long hash= 0L;
   unsigned int len= 0;

	while( *name ){
		hash+= hash<<3L ^ *name++;
		len+= 1;
	}
	if( hash_len ){
		*hash_len= len;
	}
	return( hash);
}

XGStringList *XGStringList_Delete( XGStringList *list )
{ XGStringList *last= list;
	while( last ){
		list= last;
		last= list->next;
		xfree( list->text );
		xfree( list );
	}
	return( list );
}

XGStringList *XGStringList_AddItem( XGStringList *list, char *text )
{ XGStringList *new, *last;
	if( (new= (XGStringList*) calloc( 1, sizeof(XGStringList) )) ){
		new->text= XGstrdup(text);
		new->hash= hash64( new->text, NULL );
		new->next= NULL;
		if( (last= list) ){
			if( last->last ){
				last= last->last;
			}
			else{
				while( last->next ){
					last= last->next;
				}
			}
			if( last ){
				last->next= new;
			}
			list->last= new;
		}
		else{
			list= new;
			list->last= list;
		}
	}
	return( list );
}

XGStringList *XGStringList_Pop( XGStringList *list )
{
	if( list ){
	  XGStringList *pop= list;
		if( (list= list->next) ){
			list->last= pop->last;
		}
		xfree(pop->text);
		xfree(pop);
	}
	return(list);
}

XGStringList *XGStringList_PopLast( XGStringList *list )
{
	if( list ){
	  XGStringList *pop= list, *prev= list;
		while( pop->next ){
			prev= pop;
			pop= pop->next;
		}
		if( pop!= list ){
			list->last= prev;
			prev->next= NULL;
			xfree(pop->text);
			xfree(pop);
		}
		else{
			xfree(list->text);
			xfree(list);
		}
	}
	return(list);
}

XGStringList *XGStringList_FindItem( XGStringList *list, char *text, int *item )
{ int nr= 0;
	if( list && text ){
	  long long hash= hash64(text, NULL);
		while( list && (list->hash!= hash || strcmp(list->text, text)) ){
			list= list->next;
			nr+= 1;
		}
	}
	if( list && *item ){
		*item= nr;
	}
	return(list);
}

LabelsList *Find_LabelsList( LabelsList *llist, int column )
{ Boolean found= False;
	if( !llist ){
		return(NULL);
	}
	if( llist->column== column ){
		return(llist);
	}
	while( llist && !found ){
		if( column== llist->column ){
			found= True;
		}
		else{
			if( !(column>= llist->min && column<= llist->max) ){
				llist= NULL;
			}
			else if( llist->min!= llist->max ){
				llist++;
			}
			else{
			  /* We reached the end of the array, or <column> is outside the
			   \ range of the rest of the array
			   */
				llist= NULL;
			}
		}
	}
	if( llist && (found || llist->column== column) ){
		return(llist);
	}
	else{
		return(NULL);
	}
}

char *Find_LabelsListLabel( LabelsList *llist, int column )
{ LabelsList *ll= Find_LabelsList( llist, column);
	if( ll ){
		return( ll->label );
	}
	else{
		return(NULL);
	}
}

int Find_LabelsListColumn( LabelsList *llist, char *label, int partial, int withcase )
{ int col= 0;
  LabelsList *ll= Find_LabelsList(llist,col);
	while( ll && label ){
		if( ll->label ){
			if( partial ){
				if( withcase ){
					if( strstr(ll->label, label ) ){
						label= NULL;
					}
				}
				else{
					if( strcasestr(ll->label, label) ){
						label= NULL;
					}
				}
			}
			else{
				if( withcase ){
					if( strcmp(ll->label, label)== 0 ){
						label= NULL;
					}
				}
				else{
					if( strcasecmp(ll->label, label)== 0 ){
						label= NULL;
					}
				}
			}
		}
		if( label ){
			col+= 1;
			ll= Find_LabelsList(llist,col);
		}
		else{
			ll= NULL;
		}
	}
	return( (label)? -1 : col );
}

/* a qsort routine to increment-sort a ValCategory array.	*/
static int sort_LList( LabelsList *a, LabelsList *b )
{
	if( a->column< b->column ){
		return(-1);
	}
	else if( a->column> b->column ){
		return(1);
	}
	else{
		return(0);
	}
}

/* Add a LabelsList item to an array that may or may not already exist. current_N contains
 \ a pointer to the lists current length, column & label define the new entry. The routine
 \ does a qsort to always keep the list in increasing order and afterwards determines for
 \ each element the range of the values of the current and the following elements. This will
 \ permit fast searching, and also marks the end of the list (min=max).
 */
LabelsList *Add_LabelsList( LabelsList *current_LList, int *current_N, int column, char *label )
{ LabelsList *llist;
  int lN;
	if( !current_N ){
		current_N= &lN;
		lN= -1;
	}
	  /* 20030929: don't oblige user to know the length at each and every invocation. */
	if( *current_N< 0 ){
		*current_N= LabelsList_N(current_LList);
	}
	if( !(llist= Find_LabelsList( current_LList, column )) ){
	  int i;
	  SimpleStats SS;
		  /* Re-allocate the array: the new item will be at the end.	*/
		if( !(current_LList= (LabelsList*) XGrealloc( current_LList, (*current_N+1)* sizeof(LabelsList) )) ){
			fprintf( StdErr, "Add_LabelsList(N=%d, %d, \"%s\"): can't add item (%s)\n",
				*current_N, column, label, serror()
			);
			return( NULL );
		}
		current_LList[*current_N].column= column;
		current_LList[*current_N].label= XGstrdup(label);
		  /* Now sort the array	*/
		qsort( current_LList, *current_N+1, sizeof(LabelsList), (void*) sort_LList );
		  /* Initialise a SimpleStats bin, and update the min and
		   \ max fields working backwards
		   */
		SS_Init_(SS);
		for( i= *current_N; i>= 0; i-- ){
			SS_Add_Data_(SS, 1, current_LList[i].column, 1.0 );
			current_LList[i].min= SS.min;
			current_LList[i].max= SS.max;
		}
		if( debugFlag ){
			fprintf( StdErr, "Add_LabelsList(N=%d, %d, \"%s\"): new item, current range is %s-%s\n",
				*current_N, column, label,
				d2str( current_LList[0].min, NULL, NULL), d2str( current_LList[0].max, NULL, NULL)
			);
		}
		*current_N+= 1;
	}
	else{
	  /* This is just a change of label - i.e. text changes, no new entry. No
	   \ changes are necessary to the rest of the array.
	   */
		xfree(llist->label);
		llist->label= XGstrdup(label);
	}
	return( current_LList );
}

/* Return the length of a LabelsList array	*/
int LabelsList_N( LabelsList *llist )
{ unsigned short N= 0;
	while( llist ){
		if( llist->min!= llist->max ){
			llist++;
		}
		else{
			llist= NULL;
		}
		N++;
	}
	return( N );
}

LabelsList *Copy_LabelsList( LabelsList *dest, LabelsList *llist )
{ int N= 0;
	while( llist ){
		if( (dest= Add_LabelsList( dest, &N, llist->column, llist->label )) ){
			if( llist->min!= llist->max ){
				llist++;
			}
			else{
				llist= NULL;
			}
		}
		else{
			llist= NULL;
		}
	}
	return( dest );
}

/* Free a LabelsList array	*/
LabelsList *Free_LabelsList( LabelsList *llist )
{ LabelsList *vc= llist;
	while( llist ){
		xfree( llist->label );
		if( llist->min!= llist->max ){
			llist++;
		}
		else{
			llist= NULL;
		}
	}
	xfree( vc );
	return( NULL );
}

/* Updates a LabelsList according to the information in <labels>, which consists of labels
 \ separated by <separator>.
 */
LabelsList *Parse_SetLabelsList( LabelsList *llist, char *labels, int separator, int nCI, int *ColumnInclude )
{ char *c, C;
  int N= LabelsList_N( llist );
  int tablevel= N, excluded= 0;
	if( !labels || !*labels){
		return( llist );
	}
	parse_codes( labels );
	if( debugFlag ){
		fprintf( StdErr, "Parse_SetLabelsList( \"%s\", sep='%c'):\n", labels, separator );
	}
	while( *labels && *labels== ((separator== ' ' || separator== '\t')? '\t' : separator) ){
		labels++;
		if( debugFlag ){
			fprintf( StdErr, "\t%d: empty\n", tablevel );
		}
		tablevel+= 1;
	}
	if( !labels ){
		return( llist );
	}
	do{
		if( (c= index( &labels[1], (separator== ' ' || separator== '\t')? '\t' : separator )) ){
			C= *c;
			*c= '\0';
		}
		if( *labels ){
			if( *labels== ((separator== ' ' || separator== '\t')? '\t' : separator) && labels[1] ){
				labels++;
			}
			if( *labels!= ((separator== ' ' || separator== '\t')? '\t' : separator) ){
				if( !ColumnInclude || tablevel>= nCI || ColumnInclude[tablevel] ){
					if( !(llist= Add_LabelsList( llist, &N, tablevel-excluded, labels )) ){
						labels= NULL;
						if( debugFlag ){
							fprintf( StdErr, "\tcouldn't add label \"%s\" for column %d: %s\n", labels, tablevel, serror() );
						}
					}
					else if( debugFlag ){
						fprintf( StdErr, "\t%d,\"%s\"\n", tablevel, labels );
					}
				}
				else{
					excluded+= 1;
				}
			}
			else if( debugFlag ){
				fprintf( StdErr, "\t%d: empty\n", tablevel );
			}
			tablevel+= 1;
		}
		if( c ){
			*c= C;
		}
		labels= c;
	} while( labels && *labels );
	return( llist );
}

/* returns instring (possibly in a new memory area) with variable-name opcodes parsed:
 \ <opcode_start>name<opcode_terminator> is changed by a "name=value(s)" string if
 \ no_cook is True, or by a "value(s)" string if no_cook is False. In the latter case,
 \ this results in acceptable input for e.g. the DCL ascanf function; it can be used
 \ to parse "DCL[foo,${foo}]" into "DCL[foo,5,0,1,2,3,4]" if foo is an array of 5 elements
 \ {0,1,2,3,4}.
 \ Used to parse setName and *ASK_EVAL* default commands.
 \ When no parsing is done, the initial, unchanged instring is returned.
 */
char *String_ParseVarNames( char *instring, char *opcode_start, int opcode_terminator, int include_varname, int no_cook, char *caller )
{ char *searchvar, *sb;
  extern char *pvno_found_env_var;
  extern ascanf_Function *foundVar;
	sb= instring;
	while( sb && (searchvar= strstr( sb, opcode_start)) && searchvar[2] ){
	  char *rest, *found= parse_varname_opcode( &searchvar[2], opcode_terminator, &rest, True );
	  char *val;
		if( !found && pvno_found_env_var ){
			val= found= pvno_found_env_var;
			foundVar= NULL;
		}
		else if( found ){
			if( foundVar ){
			  ascanf_Function *af;
			  int take_usage;
				if( (af= parse_ascanf_address( foundVar->value, 0, "String_ParseVarNames",
						(int) ascanf_verbose, &take_usage )) && take_usage && af->usage
				){
				  Sinc input;
				  int instring= 1;
					xfree( found );
					input.sinc.string= NULL;
					Sinc_string_behaviour( &input, NULL, 0,0, SString_Dynamic );
					Sflush( &input );
					Sputs( foundVar->name, &input );
					Sputs( "==\"", &input );
					Sprint_string_string( &input, NULL, "\"", af->usage, &instring );
					found= input.sinc.string;
				}
			}
			if( (val= strstr( found, "==" )) ){
				val+= 2;
				while( isspace(*val) ){
					val++;
				}
				if( *val== '[' ){
				  char *c;
				  extern char *strrstr();
#ifdef CSPV_FIND_LASTITEM_VALUE
				  /* an array printout..	*/
					if( (val= strrstr( val, "==")) ){
						while( isspace(*val) ){
							val++;
						}
					}
					else{
						val= "0";
					}
#else
					if( no_cook ){
						if( (val= strrstr( val, "}[")) ){
							val++;
							*val= '\0';
						}
						val= found;
					}
					else if( (c= strstr( val, "]{")) ){
						val++;
						c[0]= ','; c[1]= ' ';
						if( (c= strstr( c, "}[" )) ){
							*c= '\0';
						}
					}
					else{
						val= "0";
					}
#endif
				}
				else{
				  /* Cut off trailing value information.	*/
				  char *c= strstr( val, "==" );
					if( c ){
						*c= '\0';
					}
				}
			}
			else{
				XG_error_box( &ActiveWin, "Warning", "Ignoring opcode for non-conforming variable ", searchvar, NULL );
				val= "0";
			}
		}
		else{
		  char buf[2]= {0,0};
			buf[0]= opcode_terminator;
			XG_error_box( &ActiveWin, "Warning", "Incomplete/invalid varname opcode ", opcode_start, "..", buf, NULL );
			val= "0";
		}
		if( val && *val ){
		  char *db= instring;
			if( rest && *rest ){
				  /* Put a nullbyte for the debug feedback below	*/
				*rest= '\0';
				rest++;
			}
			if( debugFlag || ascanf_verbose ){
				fprintf( StdErr, "%s: %s}==%s\n", caller, searchvar, val );
			}
			  /* Cut out the opcode from the default expression. The remainder after the
			   \ opcode is in <rest>
			   */
			*searchvar= '\0';
			if( no_cook ){
				if( include_varname ){
					instring= concat( db, found, rest, NULL );
				}
				else{
					instring= concat( found, rest, NULL );
				}
			}
			else{
				if( include_varname ){
					instring= concat( db, val, rest, NULL );
				}
				else{
					instring= concat( val, rest, NULL );
				}
			}
			sb= strstr( instring, rest );
			xfree( db );
		}
		xfree( found );
	}
	return( instring );
}

extern char ascanf_separator;

int change_stdfile( char *newfile, FILE *stdfile )
{ char *_dev_tty=ttyname(fileno(stdin));
  FILE *fp, *fpo;
  char errbuf[512], *mode= "a";
	if( !_dev_tty ){
		_dev_tty= "/dev/tty";
	}
	if( !newfile ){
		newfile= _dev_tty;
	}
	else{
	  char *c= strstr( newfile, "::" );
		if( c && strncasecmp( newfile, "new", 3)== 0 ){
			mode= "w";
			newfile= &c[2];
		}
	}
	if( !newfile || !strlen(newfile) ){
		newfile= _dev_tty;
	}
	{ char *c= newfile;
	  char *stdfname= "??";
		if( stdfile== stdout ){
			stdfname= "stdout";
		}
		else if( stdfile== StdErr ){
			stdfname= "StdErr";
		}
		else if( stdfile== stderr ){
			stdfname= "stderr";
		}
		if( (fp= fopen( c, mode)) ){
			fclose(fp);
			if( !(fpo= freopen( c, mode, stdfile)) ){
				sprintf( errbuf, "xgraph::change_stdfile(\"%s\"): error: can't change %s (%s) - reopening %s\n",
					c, stdfname, serror(), _dev_tty
				);
				XG_error_box( &ActiveWin, "Warning", errbuf, NULL );
				fpo= freopen( _dev_tty, "a", stdfile);
				c= _dev_tty;
			}
		}
		if( !fpo ){
			sprintf( errbuf, "xgraph::change_stdfile(\"%s\"): error: can't change %s (%s)\n",
				c, stdfname, serror()
			);
			XG_error_box( &ActiveWin, "Error", errbuf, NULL );
			return( 0);
		}
		if( c== _dev_tty){
			return( 0);
		}
		else{
			return( 1);
		}
	}
}

extern char *key_param_now[256+8*12], key_param_separator;

int ListKeyParamExpressions( FILE *fp, int newline )
{ int i, n= 0, l, maxlen= 0;
  char asep = ascanf_separator;
	for( i= 0; i< 8*(256+12); i++ ){
	 char *expr= key_param_now[i];
		if( expr ){
		 int pp= (strstr( expr, "\n"))? True : False;
		 char *ppat= (pp)? "*" : "", *pat= (pp)? "*\n" : " ";
			if( key_param_separator && key_param_separator != ascanf_separator ){
				ascanf_separator = key_param_separator;
				fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
			}
			l= 0;
			if( i< 256 && isprint(i) ){
				fprintf( fp, "# Key '%c' (0x%02x):\n", i, i );
				l= fprintf( fp, "%s*KEY_EVAL*%s%c::", ppat, pat, i );
				n+= 1;
			}
			else{
			  char *mnam[]= { "   ", "c  ", "m  ", "cm  ", "s  ", "sc ", "sm ", "smc" };
				if( i> 8* 256 ){
				  int n= (i-8*256) % 12;
				  int mask= (i-8*256) / 12;
					fprintf( fp, "# fnkey %d (0x%02x):\n", i, i );
					l= fprintf( fp, "%s*KEY_EVAL*%s%sF%d::", ppat, pat, mnam[mask], n+1 );
					n+= 1;
				}
				else{
				  int n= i % 256;
				  int mask= i/ 256;
					if( i>= 256 ){
						fprintf( fp, "# key %d (0x%02x):\n", i, i );
						l= fprintf( fp, "%s*KEY_EVAL*%s%s%c::", ppat, pat, mnam[mask], n );
					}
					else{
						if( iscntrl(i) ){
							mask= 1;
							n= i- 1+ 'A';
							fprintf( fp, "# key ^%c (0x%02x):\n", n, i );
							l= fprintf( fp, "%s*KEY_EVAL*%s%s%c::", ppat, pat, mnam[mask], n );
						}
						else{
							fprintf( fp, "# key 0x%02x:\n", i );
							l= fprintf( fp, "%s*KEY_EVAL*%s0x%02x::", ppat, pat, i );
						}
					}
					n+= 1;
				}
			}
			if( strstr( expr, "\n" ) ){
				l+= fprintf( fp, "%s", expr);
				if( expr[strlen(expr)-1]!= '\n' ){
					fputc( '\n', StdErr );
					l+= 1;
				}
			}
			else{
				l+= print_string( fp, "", "\\n\n", "\n", expr );
			}
// 			fputs( " @", fp );
			if( pat[0] == '*' ){
				fputs( "*!KEY_EVAL*\n", fp );
			}
			if( newline ){
				fputc( '\n', fp );
			}
			maxlen= MAX( maxlen, l );
		}
	}
	if( ascanf_separator != asep ){
		ascanf_separator = asep;
		fprintf( fp, "*ARGUMENTS* -separator %c\n", ascanf_separator );
	}
	return(maxlen);
}

extern char *process_history;
extern xtb_hret process_hist_h();

LocalWindows *Find_WindowListEntry_String( char *expr, LocalWin **wi1, LocalWin **wi2 )
{ LocalWin *lwi1= NULL, *lwi2= NULL;
  LocalWindows *WL= WindowList;
  int pnum, pwinnum, winnum;
	if( strcmp( expr, "*SELF*" ) == 0 || strcmp( expr, "*ACTIVE*" ) == 0 ){
		while( WL && WL->wi != ActiveWin ){
			WL= WL->next;
		}
		if( WL ){
			lwi1= WL->wi;
		}
	}
	else if( sscanf( expr, "%d:%d:%d", &pnum, &pwinnum, &winnum)== 3 ){
		while( WL &&
			!(WL->wi->parent_number== pnum && WL->wi->pwindow_number== pwinnum && WL->wi->window_number== winnum)
		){
			WL= WL->next;
		}
		if( WL ){
			lwi1= WL->wi;
		}
	}
	else{
		ReadData_proc.param_range[0]= 0;
		new_param_now( expr, &ReadData_proc.param_range[0], 1 );
		lwi1= (LocalWin*) ((unsigned long) ReadData_proc.param_range[0]);
		lwi2= (LocalWin*) (((unsigned long) ReadData_proc.param_range[0])+ 1);
		if( lwi1 ){
			while( WL && WL->wi!= lwi1 ){
				WL= WL->next;
			}
			if( !WL && lwi2 ){
				WL= WindowList;
				while( WL && WL->wi!= lwi2 ){
					WL= WL->next;
				}
			}
		}
	}
	if( WL && (debugFlag || scriptVerbose) ){
		fprintf( StdErr, "Find_WindowListEntry_String(\"%s\"): found window 0x%lx %02d:%02d:%02d\n", 
			expr,
			WL->wi, WL->wi->parent_number, WL->wi->pwindow_number, WL->wi->window_number
		);
	}
	if( wi1 ){
		*wi1= lwi1;
	}
	if( wi2 ){
		*wi2= lwi2;
	}
	return( WL );
}

double WaitForEvent( LocalWin *wi, int type, char *msg, char *caller )
{ int cont, e= 0;
  XEvent evt;
  Time_Struct timer;
  char *tmsg= NULL;
	Elapsed_Since( &timer, True );
	e+= Handle_An_Events( wi->event_level, 1, caller, wi->window, 0);
	wi->redraw= 0;
	switch( type ){
		case KeyPress:
			if( msg ){
				/* Caution: only de-allocate tmsg below when msg!= NULL!! */
				tmsg= concat( "Waiting for a key press [", msg, "]...", NULL );
			}
			else{
				tmsg= "Waiting for a key press...";
			}
			TitleMessage( wi, tmsg );
			break;
		default:
			if( msg ){
				/* Caution: only de-allocate tmsg below when msg!= NULL!! */
				tmsg= concat( "Waiting for an event [", msg, "]...", NULL );
			}
			else{
				tmsg= "Waiting for an event...";
			}
			TitleMessage( wi, tmsg );
			e+= Handle_An_Events( wi->event_level, 1, caller, wi->window, 0);
			XSync( disp, True );
			break;
	}
	if( msg ){
		xfree( tmsg );
	}
	do{
		XNextEvent( disp, &evt );
		cont= 0;
		if( evt.xany.window!= wi->window || evt.type== ClientMessage ){
			_Handle_An_Event( &evt, wi->event_level+1, 0, caller );
			cont= 1;
		}
		else if( type ){
			if( evt.xany.type!= type ){
				cont= 1;
			}
		}
	} while( cont );
	TitleMessage( wi, NULL );
	_Handle_An_Event( &evt, wi->event_level, 0, "WaitForEvent()-finished" );
	Elapsed_Since( &timer, False );
	return( timer.HRTot_T );
}

char *strrstr(const char *a, const char *b)
{ unsigned int lena, lenb, l=0;
  int success= 0;
  char *c;
	
	lena=strlen(a);
	lenb=strlen(b);
	l = lena - lenb;
	c = &a[l];
	do{
		while( l > 0 && *c!= *b){
			c--;
			l -= 1;
		}
		if( c== a && *c!= *b ){
			return(NULL);
		}
		if( !(success= !strncmp( c, b, lenb)) ){
			if( l > 0 ){
				c--;
				l -= 1;
			}
		}
	} while( !success && l > 0 );
	if( !success ){
		return(NULL);
	}
	else{
		return(c);
	}
}

/* SubstituteOpcodes( char *source, char *opcode1, char *subst1, ..., 0)= parsed_string
 \ scan <source> for occurrences of opcode1, opcode2, ..., and replace all by
 \ subs1, subst2, ... The result is returned; if no changes were made, this is a pointer
 \ to the <source> string, else, a pointer to dynamically allocated memory. Hence:
 	if( (result= SubstituteOpcodes( source, op1, sub1, op2, sub2, 0))!= source ){
		xfree(result);
	}
  \ 20010113
  */
char *SubstituteOpcodes(char *source, VA_DCL)
{ va_list ap;
  int n= 0, tlen= 0;
  char *src, *orig, *opcode, *subst, *PF= NULL, *buf= NULL, *c= NULL;
	va_start(ap, source);
	orig= src= source;
	if( src ){
		while( (opcode= va_arg(ap, char*)) && (subst= va_arg(ap, char*)) ){
			if( subst== (char*) &PrintFileName ){
				if( ActiveWin && ActiveWin!= &StubWindow && ActiveWin->current_device>= 0 ){
				  char *c;
					PF= strdup(ActiveWin->hard_devices[ActiveWin->current_device].dev_file);
					if( PF ){
						if( (c= strrstr(PF, ".ps")) ||
							(c= strrstr(PF, ".xg")) ||
							(c= strrstr(PF, ".ss")) ||
							(c= strrstr(PF, ".cg")) ||
							(c= strrstr(PF, ".sh")) ||
							(c= strrstr(PF, ".idraw")) ||
							(c= strrstr(PF, ".hpgl"))
						){
							*c= '\0';
						}
						subst= PF;
					}
				}
				if( !PF ){
					if( UPrintFileName[0] ){
						subst= UPrintFileName;
					}
					else if( PrintFileName ){
						subst= PrintFileName;
					}
					else{
						subst= "UnTitled";
					}
				}
			}
			if( (c= strstr(src, opcode)) ){
				n= 1;
				while( (c= strstr( &c[1], opcode)) ){
					n+= 1;
				}
				if( debugFlag ){
					fprintf( StdErr, "SubstituteOpcodes(\"%s\"): replacing %dx \"%s\" by \"%s\"\n",
						src, n, opcode, subst
					);
				}
				tlen= strlen(src) + n* strlen(subst)+ 1;
				if( (buf= XGrealloc( buf, tlen* sizeof(char))) ){
				  char *d= buf, *c= src;
				  int clen= strlen(opcode), slen= strlen(subst);
					while( *c ){
						if( strncmp(c, opcode, clen)== 0 ){
							strcat( d, subst);
							d+= slen;
							c+= clen;
						}
						else{
							*d++= *c++;
						}
						*d= '\0';
					}
					StringCheck( buf, tlen, __FILE__, __LINE__);
					if( src!= orig ){
						xfree(src);
					}
					src= strdup(buf);
				}
			}
			xfree(PF);
		}
	}
	va_end(ap);
	if( src!= orig ){
		xfree(src);
	}

	return( (buf)? buf : src );
}

extern char *comment_buf;
extern int comment_size, NoComment;
int NextGen= 0;

int next_gen_include_file( char *command, char *fname )
{
	while( *fname && isspace((unsigned char)*fname) ){
		fname++;
	}
	if( ActiveWin ){
		if( *fname ){
			ActiveWin->next_include_file= concat2( ActiveWin->next_include_file, "*", command, "\n", NULL);
			if( debugFlag ){
				fprintf( StdErr, "2nd generation include file(s): \"%s\"\n", ActiveWin->next_include_file );
			}
		}
		else{
			xfree( ActiveWin->next_include_file );
			if( debugFlag ){
				fprintf( StdErr, "2nd generation include file(s) removed\n" );
			}
		}
	}
	else{
		if( *fname ){
			next_include_file= concat2( next_include_file, "*", command, "\n", NULL);
			if( debugFlag ){
				fprintf( StdErr, "2nd generation include file(s): \"%s\"\n", next_include_file );
			}
		}
		else{
			xfree( ActiveWin->next_include_file );
			if( debugFlag ){
				fprintf( StdErr, "2nd generation include file(s) removed\n" );
			}
		}
	}
	return(1);
}

/* char *ReadData_buffer= NULL; int *ukl;	*/


int IncludeFile( LocalWin *rwi, FILE *strm, char *Fname, int event, char *skip_to )
{ extern double _Xscale, _Yscale, _DYscale;
  extern double Xscale, Yscale, DYscale;
  extern int maxitems, fileNumber, use_lx, use_ly;
  int RDc= ReadData_commands, n, idx, ok= 1, old_max= maxitems;
  int fitx, fity,
	FX, FY;
  int redraw, wid;
  extern char *skip_to_label;
  char *stl= skip_to_label;
  FILE *close_again= NULL;

/* 	if( !rwi ){	*/
/* 		return(0);	*/
/* 	}	*/
	if( rwi ){
		fitx= rwi->fit_xbounds;
		fity= rwi->fit_ybounds;
		redraw= rwi->redraw;
	}

	_Xscale= _Yscale= _DYscale= 1.0;
	Xscale= Yscale= DYscale= 1.0;
	use_lx= 0;
	use_ly= 0;
	ActiveWin= rwi;
	if( rwi ){
		CopyFlags( NULL, rwi );
		wid= rwi->window;
	}
	else{
		wid= DefaultRootWindow( disp );
	}
	  /* FX and FY should be copies (via global vars) of rwi->fit_xbounds and ~fit_ybounds:	*/
	lFX= FX= FitX; lFY= FY= FitY;
	if( strm ){
		add_process_hist( Fname );
	}
	else if( Fname ){
	  /* 20040127	*/
		strm= close_again= fopen(Fname, "r");
	}
	skip_to_label= skip_to;
	ReadData_commands= 0;
	if( (n = ReadData(strm, Fname, fileNumber)) < 0 ){
		xtb_error_box( wid, "data formatting error\n", Fname );
		if( rwi ){
			rwi->redraw= redraw;
		}
		ok= 0;
	}
	else if( n> maxitems ){
	  DataSet *cur_set= &AllSets[setNumber], *os= cur_set;
	  int sn= setNumber;
		maxitems= n;
		NewSet( NULL, &cur_set, 0);
		if( WarnNewSet && (setNumber> sn || os!= cur_set) ){
			fprintf( StdErr, "#---- Starting new set #%d; file \"%s\" (end) (IncludeFile())\n",
				setNumber, Fname
			);
		}
		  /* eliminate trailing empty datasets	*/
		for( idx= setNumber-1; idx>= 0; idx-- ){
			if( AllSets[idx].numPoints<= 0 ){
				setNumber-= 1;
			}
		}
	}
	  /* 20040127	*/
	if( close_again ){
		fclose( close_again );
		strm= close_again= NULL;
	}
	ReadData_commands= RDc;
	skip_to_label= stl;
	ascanf_exit= 0;
	if( maxitems> old_max ){
		realloc_Xsegments();
		if( rwi ){
			rwi->fit_xbounds= 1;
			rwi->fit_ybounds= 1;
		}
		ok= 2;
	}
	else{
		maxitems= old_max;
	}
	if( AddFile ){
	 char buf[1024];
		sprintf( buf, "*ADD_FILE* %s : sorry, doesn't work after startup ...\n", AddFile );
		xtb_error_box( wid, buf, "Notice" );
		xfree( AddFile );
	}
	if( ok ){
		  /* Ensure that 'a la fin', we set the autoscaling to the setting
		   \ specified in the includefile, if any. Otherwise, reset it to
		   \ the value it had before including. This could probably be done
		   \ a little more elegantly, but these flags are touched at soo many
		   \ levels... (lFX and lFY are set *only* by ParseArgs)
		   */
		if( lFX!= FX ){
			fitx= lFX;
		}
		if( lFY!= FY ){
			fity= lFY;
		}
		if( rwi ){
			CopyFlags( rwi, NULL);
			UpdateWindowSettings( rwi, False, True );
		}
		for( idx= 0; idx< setNumber; idx++ ){
		  char *c;
			if( AllSets[idx].appendName ){
				c= AllSets[idx].setName;
				if( c ){
					AllSets[idx].setName= XGstrdup2( c, AllSets[idx].appendName );
					xfree(c);
					xfree( AllSets[idx].appendName );
				}
				else{
					AllSets[idx].setName= AllSets[idx].appendName;
				}
				AllSets[idx].setName= String_ParseVarNames( AllSets[idx].setName, "%[", ']', True, True, "IncludeFile()" );
				AllSets[idx].appendName= NULL;
			}
		}
		set_changeables(2,False);
		if( ScriptFile ){
			if( ScriptFileWin ){
				ReadScriptFile( ScriptFileWin );
				ScriptFileWin->redraw= redraw;
			}
		}
		if( ok== 2 && !fitx && !fity ){
			xtb_error_box( wid, "Fitting window to new boundaries!", "Read File Message");
		}
  /* calling DrawWindow() at this point when already drawing can lead to crashes in some cases since
   \ it doesn't contain a robust check against nested redraws (except for those
   \ caused by X events). Therefore, we prefer to generate a redraw by sending an Expose event
   \ to the window. Nesting in this case just leads to strange drawing results, but no crashes.
		SD_redraw_fun( 0, 1, NULL );
   */
		if( !rwi ){
			return( n );
		}
		if( event ){
			ExposeEvent( rwi );
		}
	}
	if( rwi ){
		rwi->fit_xbounds= fitx;
		rwi->fit_ybounds= fity;
	}
	set_changeables(2,False);
	return(n);
}

int Include_Files( LocalWin *wi, char *fn, char *caller )
{ char Fname[2*MAXPATHLEN];
  int r= 0;
  LocalWin *aw= ActiveWin;
  extern char *tildeExpand();
  FILE *strm;
  char End, *end= NULL, *pyfn;
	while( *fn && isspace((unsigned char)*fn) ){
		fn++;
	}
	if( ActiveWin && ActiveWin->drawing ){
		fprintf( StdErr, "IncludeFiles(\"%s\",\"%s\"): a window is currently drawing: installing the file(s) as *SCRIPT_FILE*\n",
			fn, caller
		);
		ScriptFile= XGstrdup(fn);
		ScriptFileWin= wi;
		return(0);
	}
	  // 20101019:
	if( wi ){
		CopyFlags(NULL, wi);
	}
	do{
	  int is_pipe= 0;
		ActiveWin= wi;
		{ char *c= (fn[0]== '|')? &fn[1] : fn;
			if( fn[0]== '|' ){
				is_pipe= True;
			}
			end= NULL;
			while( !end && *c ){
				if( fn[0]== '|' ){
					if( *c== '|' && (c== fn || c[-1]!= '\\') ){
						end= c;
					}
				}
				else{
					if( isspace((unsigned char) *c ) && (c== fn || c[-1]!= '\\') ){
						end= c;
					}
				}
				c++;
			}
			if( end ){
				End= *end;
				*end= '\0';
			}
			else if( is_pipe ){
				fprintf( StdErr, "### Warning: \"pipe command\" \"%s\" called from \"%s\" not properly terminated by trailing '|'!\n",
					&fn[1], caller
				);
			}
		}
		if( strcmp( fn, "*CLIPBOARD*")== 0 ){
		 int nbytes= 0;
		  char *clipboard= XFetchBuffer( disp, &nbytes, 0);
			if( clipboard && nbytes ){
			  char *tnam= XGtempnam( getenv("TMPDIR"), "XGCLP");
				if( tnam && (strm= fopen( tnam, "wb")) ){
					fwrite( clipboard, sizeof(char), nbytes, strm );
					fputs( "\n", strm );
					fclose( strm );
					strm= fopen( tildeExpand(Fname, tnam), "r" );
					unlink(tnam);
				}
				else{
					xtb_error_box( wi->window, "Can't create or open temporary *CLIPBOARD* file\n", "Failure" );
				}
				if( tnam ){
					xfree(tnam);
				}
				  /* 20040922 */
				XFree(clipboard);
			}
			else{
				xtb_error_box( wi->window, "There is nothing on the clipboard...\n", "Bummer" );
				strm= NULL;
			}
		}
		else if( (pyfn= PyOpcode_Check(fn)) ){
		  extern DM_Python_Interface *dm_python;
			if( dm_python && dm_python->Import_Python_File ){
				if( (*dm_python->isAvailable)() ){
					(*dm_python->Import_Python_File)( pyfn, NULL, 0, False );
				}
				else{
					xtb_error_box( wi->window, "Python is not available to read the specified file - do you have a Python console open?\n", pyfn );
				}
			}
			else{
				xtb_error_box( wi->window, "failure to initialise python, not reading the specified file\n", pyfn );
			}
			goto include_next_file;
		}
		else if( fn[0]== '|' ){
		  extern Boolean PIPE_error;
			tildeExpand( Fname, &fn[1] );
			PIPE_error= False;
			strm= popen( Fname, "r");
			is_pipe= 1;
		}
		else{
			strm = fopen( tildeExpand( Fname, fn), "r");
			is_pipe= 0;
		}
		if( !strm){
			xtb_error_box( wi->window, "Can't open file\n", Fname );
		}
		else{
			IdentifyStreamFormat( Fname, &strm, &is_pipe );
			r+= IncludeFile( wi, strm, Fname, False, NULL );
			if( is_pipe ){
				pclose(strm);
			}
			else{
				fclose( strm);
			}
		}
include_next_file:;
		if( end ){
			*end= End;
			fn= ++end;
			while( *fn && isspace((unsigned char)*fn) ){
				fn++;
			}
		}
		else{
			fn= NULL;
		}
	} while( fn && *fn );
	ActiveWin= aw;
	return(r);
}

int interactive_IncludeFile(LocalWin *wi, char *msg, char *ret_fn)
{ char fname[MAXPATHLEN], *Msg= NULL, *nbuf;
  int r= 0;
  extern char *parse_codes( char*);
  char pc[]= "#x01Request";
	if( !msg ){
		msg= "Please enter a filename (or pipe),\n or multiple filenames separated by whitespace\n or *CLIPBOARD* (guess what for):";
	}
	if( ret_fn ){
	  char *c= ascanf_string(ret_fn, NULL);
		strncpy( fname, ret_fn, MAXPATHLEN-1 );
		if( c!= ret_fn ){
			if( (Msg= concat( msg, "\n", ret_fn, "==\"", c, "\"\n", NULL )) ){
				msg= Msg;
			}
		}
	}
	else{
		fname[0]= '\0';
	}
	if( wi && (nbuf= xtb_input_dialog( wi->window, fname, 80, MAXPATHLEN,
			msg, parse_codes( pc),
			False,
			NULL, NULL, "Edit", SimpleEdit_h, " ... ", SimpleFileDialog_h
		))
	){
	  LocalWin *aw= ActiveWin;
		errno= 0;
		if( ret_fn ){
			strncpy( ret_fn, fname, MAXPATHLEN-1 );
		}
		if( !*fname ){
			return(-1);
		}
		{ char *c= ascanf_string(fname,NULL);
			if( c!= fname ){
				strncpy( fname, c, MAXPATHLEN-1 );
			}
		}
		 /* 20040605: store Include_Files()'s return value in r!!!?	*/
		r= Include_Files( wi, fname, "[IncludeFile dialog]" );
		if( !r ){
			r= -1;
		}
		else{
			ExposeEvent(wi);
		}
		ActiveWin= aw;
		if( nbuf!= fname ){
			xfree( nbuf );
		}
	}
	xfree( Msg );
	return(r);
}

int ReadScriptFile( LocalWin *wi )
{ char *fname= ScriptFile;
  char *SF= ScriptFile;
  LocalWin *aw= ActiveWin;
  int r= 0;
  extern int raw_display_init;
	if( ScriptFile ){
		errno= 0;
		Handle_An_Events( 1, 1, "waiting to do *SCRIPT_FILE*", 0, 0);
		{ LocalWindows *WL= WindowList;
		  int n= 0;
			  /* forcedly redraw all windows that haven't been redrawn before continuing	*/
			while( WL ){
				if( WL->wi && !WL->wi->redrawn ){
					RedrawNow( WL->wi );
					n+= 1;
				}
				WL= WL->next;
			}
			if( n ){
				Handle_An_Events( 1, 1, "ReadScriptFile-"STRING(__LINE__), 0, 0);
			}
		}
		ScriptFile= NULL;
		if( (raw_display_init< 0 || raw_display_init== 1) && wi->raw_display ){
			wi->raw_display= False;
			fprintf( StdErr, "ReadScriptFile(\"%s\"): unsetting raw_display because raw_display_init==%d\n",
				fname, raw_display_init
			);
		}
		r= Include_Files(wi, fname, "*SCRIPT_FILE*" );
		xfree( SF );
		if( r ){
			Handle_An_Events( 1, 1, "ReadScriptFile-"STRING(__LINE__), 0, 0);
			if( raw_display_init== 2 && wi->raw_display ){
				wi->raw_display= False;
				fprintf( StdErr, "IncludeFile(): unsetting raw_display after including \"%s\" because raw_display_init==%d\n",
					fname, raw_display_init
				);
			}
			ExposeEvent( wi );
		}
		ActiveWin= aw;
	}
	return( r );
}

int From_Tarchive= 0;
char Tarchive[MAXPATHLEN]= "";

/* rar p -ierr archive files	*/
/* zip -p archive files	*/

int IdentifyStreamFormat( char *fname, FILE **strm, int *is_pipe )
{ int r= 0;
	errno= 0;
	if( strm && *strm ){
	  struct stat sb;

		  /* 20040922: Mac OS X needs explicit tests: the fseek( ftell()) test below doesn't fire on fifos etc. */
		if( fstat( fileno(*strm), &sb) ){
			fprintf( StdErr, "IdentifyStreamFormat(): can't fstat \"%s\": bailing out (%s)\n",
				fname, serror()
			);
			return(0);
		}

		if( !( S_ISREG(sb.st_mode)
#ifdef S_ISLNK
				|| S_ISLNK(sb.st_mode)
#endif
			)
		){
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "IdentifyStreamFormat(): \"%s\" is not likely to be seekable: won't even try.\n"
					" Make sure your data is in proper and uncompressed (or whatever) form!\n",
					fname
				);
			}
			return(0);
		}

		if( fseek( *strm, ftell(*strm), SEEK_SET ) || errno== EBADF ){
			errno= EBADF;
			if( debugFlag || scriptVerbose  ){
				fprintf( StdErr, "IdentifyStreamFormat(\"%s\"): not seekable: won't try to identify and read 'as is'.\n",
					fname
				);
			}
		}
	}
	if( strm && is_pipe && (*strm!= stdin || From_Tarchive) && !*is_pipe && errno!= EBADF ){
	  union m{
		  char mstr[3];
		  unsigned short magic;
	  } magic;
	  char command[MAXPATHLEN];
	  FILE *tarfp= NULL, *s= *strm;
	  int compressed= 0;
/* from /etc/magic:
0	short		0x1f8b		gzip compressed data
0	string		BZh			bzip2 compressed data
0	short		0x1f9d		compress(1) output
0	short	0xff1f	compact(1)ed file
0	short	0x1f1e	packed data
0	string		PK\003\004	Zip archive data
0	string		Rar!		RAR archive data
*/
		if( From_Tarchive> 0 && (tarfp= fopen(Tarchive,"r")) ){
			*strm= tarfp;
		}
		if( read( fileno(*strm), &magic, sizeof(magic) )== sizeof(magic) ){
			switch( magic.magic ){
				case 0x8b1f:
				case 0x9d1f:
				case 0x1e1f:
				case 0x1f8b:
				case 0x1f9d:
				case 0x1f1e:
					sprintf( command, "gzip -dv < %s", fname );
					compressed= 1;
					break;
				case 0x1fff:
				case 0xff1f:
					sprintf( command, "uncompact < %s", fname );
					compressed= 2;
					break;
				default:
					command[0]= '\0';
					break;
			}
			if( !*command && strncmp( magic.mstr, "BZh", 3)== 0 ){
				sprintf( command, "bzip2 -d < %s", fname );
				compressed= 3;
			}
			if( tarfp && !*command ){
			  char pat[5];
				sprintf( pat, "PK%c%c", 3, 4 );
				if( strncmp( magic.mstr, pat, 4)== 0 ){
					compressed= 4;
				}
				else if( strncmp( magic.mstr, "Rar!", 4)== 0 ){
					compressed= 5;
				}
			}
			if( !*command && strncmp( magic.mstr, "Rar!", 4)== 0 ){
				sprintf( command, "unrar p -ierr %s", fname );
				compressed= 6;
			}
			PIPE_error= False;
			if( command[0] && !tarfp ){
			  FILE *pfp= popen( command, "r");
				if( pfp ){
					if( *strm && *strm!= stdin ){
						fclose( *strm );
					}
					*strm= pfp;
					*is_pipe= 2;
					r= 1;
				}
				else{
					fprintf( StdErr, "\"%s\" is compressed, but couldn't execute command \"%s\" (%s)\n",
						fname, command, serror()
					);
					r= EOF;
				}
			}
		}
		if( tarfp ){
			*strm= s;
			fclose( tarfp );
			if( compressed== 2 ){
				fprintf( StdErr, "\"%s\" is a tar archive compressed with an unsupported compressor -- can't extract %s\n",
					Tarchive, fname
				);
				r= EOF;
			}
			else{
			  char *tuncr;
				command[0]= '\0';
				switch( compressed ){
					case 1: tuncr= "z"; break;
					case 3: tuncr= "y"; break;
					case 4:
						tuncr= NULL;
						snprintf( command, sizeof(command)/sizeof(char),
							"unzip -p %s %s", Tarchive, fname
						);
						break;
					case 5:
						tuncr= NULL;
						snprintf( command, sizeof(command)/sizeof(char),
							"unrar p -ierr %s %s", Tarchive, fname
						);
						break;
					default: tuncr= ""; break;
				}
				if( tuncr ){
					snprintf( command, sizeof(command)/sizeof(char),
						"tar -x%s%sOf %s %s",
							(debugFlag)? "vv" : "", tuncr, Tarchive, fname
					);
				}
				if( debugFlag ){
					fprintf( StdErr, "Extracting file with '%s'\n", command );
				}
				if( (s= popen( command, "r")) ){
					if( *strm && *strm!= stdin ){
						fclose( *strm );
					}
					*strm= s;
					*is_pipe= 2;
					r= 1;
				}
				else{
					fprintf( StdErr, "Couldn't execute command \"%s\" (%s)\n",
						command, serror()
					);
					r= EOF;
				}
			}
		}
		if( !*is_pipe ){
			if( *strm && *strm!= stdin ){
				errno= 0;
				rewind( *strm );
				if( errno ){
					fprintf( StdErr, "IdentifyStreamFormat(): failed to rewind the input file \"%\": expect read errors! (%s)\n",
						fname, serror()
					);
				}
			}
		}
	}
	return(r);
}

int ParseInputString( LocalWin *wi, char *input )
{ char *tnam= XGtempnam( getenv("TMPDIR"), "XGPIS");
  FILE *strm;
	if( tnam && (strm= fopen( tnam, "wb")) ){
	  char *c= input;
		while( *c ){
			if( *c== '\\' && c[1]== 'n' ){
				fputc( '\n', strm );
				c+= 2;
			}
			else{
				fputc( *c++, strm );
			}
		}
		fputc( '\n', strm);
		fclose( strm );
		strm= fopen( tnam, "r" );
		unlink( tnam );
		IncludeFile( wi, strm, tnam, True, NULL );
		fclose( strm );
		  /* 20040922 */
		xfree(tnam);
		return(1);
	}
	else{
		if( tnam ){
			xtb_error_box( wi->window, "Can't open temporary file\n", tnam );
			xfree(tnam);
		}
		else{
			xtb_error_box( wi->window, "Can't open temporary file\n", "<NULL!!>" );
		}
		return(0);
	}
}

int interactive_parse_string( LocalWin *wi, char *expr, int tilen, int maxlen, char *message, int modal, double verbose )
{ extern xtb_hret display_ascanf_variables_h();
  int r= 0;
  extern Window ascanf_window;
  Window aw= ascanf_window;
  char *nbuf;
  char pc[]= "#x01Enter any valid XGraph input:";
	if( !ActiveWin ){
		ActiveWin= wi;
	}
	if( (nbuf= xtb_input_dialog( wi->window, expr, tilen, maxlen, message,
			parse_codes(pc),
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
			ascanf_window= wi->window;
			r= ParseInputString( wi, expr );
			ascanf_window= aw;
		}
		if( nbuf!= expr ){
			xfree( nbuf );
		}
	}
	return(r);
}

int Opt01;

int ArgsParsed= 0, ArgError= False;

int argerror( char *err, char *val)
{  char *PAGER= getenv("PAGER");
   FILE *fp;
   int i;
   extern int fod_num;
   extern char *fodstrs[];
	if( !PAGER){
		PAGER= "more";
	}
	PIPE_error= False;
#if defined(__APPLE__) || defined(__MACH__)
	if( !isatty(fileno(stdin)) || use_pager || !(fp= popen( PAGER, "w")) )
#else
	if( !isatty(fileno(stdin)) || use_pager || !(fp= popen( PAGER, "wb")) )
#endif
	{
		fp= StdErr;
	}
	signal( SIGPIPE, PIPE_handler );
	PIPE_fileptr= &StdErr;
	if( err && val){
		fprintf( fp, "Error: %s: %s", val, err);
		if( errno ){
			fprintf( fp, " (%s)", serror() );
		}
		fputs( "\n\n", fp );
	}
	if( !ArgsParsed ){
		fputs( "usage : xgraph [-XGraph] [-nodetach] [files] [-bd color] [-bg color] [-fg color] [-zg color]\n", fp);
		fputs( "         [-df best|dialog_font] [-af best|axis_font] [-lef best|legend_font] [-laf best|label_font] [-tf best|title_font]\n", fp);
		fputs( "         [-bw bdr_width] [-eng] [-absy] [-mf[01]]\n", fp);
		fputs( "         [-detach] [-<digit> set_name] [-t|T title] [-x unitname] [-y unitname]\n", fp);
		fputs( "         [-lx|-LX x1,x2] [-ly|-LY y1,y2] [-nosort] [-auto] [-bar] [-brb base] [-brw width]\n", fp);
		fputs( "         [-bb] [-db] [-polar[radix]] [-radix <radix>] [-radix_offset <offset>] [-ln{x,y}] [-sqrt{x,y}] [-nl] [-m] [-M] [-p] [-P] [-rv] [-noerr]\n", fp);
		fputs( "         [-log_zero_{x,y} X] [-log_zero_x_sym_{x,y} sym] [-tk] [-zl] [{-aw,-lw} linewidth]\n", fp);
		fputs( "         [-display host:display.screen] [=geospec] [-notitle] [-nolegend]\n", fp);
		fputs( "         [-Cxye|-Cauto] [-PSm base,incr] [-colour] [-monochrome] [-print] [-help] [-pf filename]\n", fp);
		fputs( "         [-scale_av_{x,y} a] [-stats] [-zero epsilon] [-fli] [-fn] [-spa] [-triangle]\n", fp);
		fputs( "         [-ml{x,y} c1,c2] [-progname <name>] [-pow{x,y}[01] <val>] [-pow{A,S} <val>] [-plot_only_file <#>]\n", fp);
		fputs( "         [-legend_ul <x>,<y>] [-{x,y}step <step>] [-maxHeight|-maxWidth <dim>] [-noreal_{x,y}_val]\n", fp);
		fputs( "         [-PrintInfo] [-bias{X,Y} <val>] [-disconnect] [-{x,y}_ul <x>,<y>] [-progress] [-process_bounds] [-plot_only_set <list>]\n", fp);
		fputs( "         [-print_as <type>] [-print_to <dev>] [-transform_axes] [-overwrite_marks] [-overwrite_legend] [-XGBounds] [-raw_display] [-error_region]\n", fp);
		fputs( "         [-show_overlap] [-average_error] [-PN <expr>] [-DumpAverage] [-SameAttr] [-nosplit] [-preserve_aspect] [-separator <c>]\n", fp);
		fputs( "         [-python <python file>] [-preserve_ftime] [-IO_Import <module> <file>] [-registry]\n", fp);
	 
		fprintf( fp, "-absy\ttake absolute value of Y values before any other transformation [%d]\n", absYFlag );
		fprintf( fp, "-aspect\tchanges bounds to have the same range on both axes [%d]\n", Aspect );
		fputs(	"-auto\tforce auto scaling (ignore indications in files)\n", fp);
		fprintf( fp, "-average_error\ttoggle displaying of average error [%d]\n", use_average_error );
		fprintf( fp, "-aw w\tset width of axis lines [%g]\n", axisWidth);
		fprintf( fp,	"-bar\tDraw bar graph with base -brb, width -brw and type -brt [%g,%g,%d]\n", barBase, barWidth, barType);
		fprintf( fp, "-bias{X,Y}\tallow X/Y axis translation over the average if axis-step is smaller than [%g,%g]\n",
			Xbias_thres, Ybias_thres
		);
		fputs(	"-bb[01]\tNo Box / Box around data\n", fp);
		fprintf( fp, "-Cauto\tderive order of x,y,error columns from *Cxye*<x>,<y>,<e> statement in file(s) [%d]\n",
			use_xye_info
		);
		fputs(	"-Cxye\torder of x, y, error columns in input\n", fp);
		fputs(	"-db\tTurns on debugging mode\n", fp);
		fputs(	"-detach\tDetach after initial communication with X-Server\n", fp);
		fputs(  "-disconnect\tCreate a new set when the X-co-ordinate increase reverses direction\n", fp);
		fprintf( fp, "-DumpAverage[01]\tDump the average values calculated (by *AVERAGE*) in XGraph dumps, instead of the commando [%d]\n",
			dump_average_values
		);
		fprintf( fp, "-DumpBinary[01]\tDump the binary data instead of the ASCII values [%d]\n",
			DumpBinary
		);
		fprintf( fp, "-DumpIncluded[01]\tWhen dumping read data to terminal, dump included files also [%d]\n",
			DumpIncluded
		);
		fprintf( fp, "-DumpProcessed[01]\tDump the processed values calculated (by *DATA_PROCESS*) in XGraph dumps, instead of the raw values [%d]\n",
			DumpProcessed
		);
		fprintf( fp, "-DumpRead[01]\tDump the data read to the terminal, and quit while all files have been read[%d]\n"
				"When -DumpBinary is given, data are output in binary, else in ASCII\n",
			DumpFile
		);
		fputs(	"-eng[XY]<01>\tEngineering notation on X or Y axis\n", fp);
		fprintf( fp, "-error_region\tConnect all lower and higher ends of the errorbars, creating an error region [%d]\n",
			error_regionFlag
		);
		fprintf( fp, "-ew w\tset width of errorbar lines [%g]\n", errorWidth);
		fputs(  "-fascanf_functions\tgenerate overview of fascanf functions useful for the *PARAM_FUNCTIONS* command\n", fp);
		fprintf( fp, "-fascanf_verbose\tShow fascanf's function parsing progress (very verbose!) [%d]\n", (ascanf_verbose)? 1 : 0 );
		fprintf( fp, "-fit_xbounds[012]\tAlways scale the X-axis to the window - flag 2 also fits the radix to range in polar mode [%d]\n", FitX );
		fprintf( fp, "-fit_ybounds[01]\tAlways scale the Y-axis to the window [%d]\n", FitY );
		fprintf( fp, "-fli[01]\tIncrement lineWidth for each new file [%g]\n", newfile_incr_width );
		fprintf( fp, "-fn[01]\tshow filename to the right of the legend [%d]\n", filename_in_legend );
		fprintf( fp, "-gg colour\tset colour of grid (not tick!) lines [%s]\n", gridCName);
		fprintf( fp, "-gp pat\tset pattern of tick/grid lines [%x", gridLS[0] );
			for( i= 1; i< gridLSLen; i++ ){
				fprintf( fp, "%x", gridLS[i] );
			}
			fprintf( fp, "]\n" );
		fprintf( fp, "-gw w\tset width of tick/grid lines [%g]\n", gridWidth);
		fputs(	"-help[1]\tPrint help. -help1 also display the manpage if correcly installed\n", fp);
		fprintf( fp, "-IO_Import <module> <file>\timport <file> using IO_Import library <module>\n" );
		fprintf( fp, "-lb[01]\tshow label instead of filename to the right of the legend [%d]\n", labels_in_legend );
		fprintf( fp, "-legend_ul <x>,<y>\tplace the upper left corner of the legend box at co-ordinate (x,y) [%g,%g]\n\tCan also be done by Shift-clicking with the 1st mousebutton\n",
			legend_ulx, legend_uly
		);
		fprintf( fp, "-ln{x,y}[01]\tLinear / Logarithmic scale for X or Y axis [%d,%d]\n", logXFlag, logYFlag );
		fprintf( fp, "-log_zero_{x,y} X\tSubsitute X for 0 in input on a log x/y axis [%g,%g]\n", _log_zero_x, _log_zero_y);
		fprintf( fp, "-log_zero_{x,y}_{min,max}\tSubsitute the min or max for 0 in input on a log x/y axis [%d,%d]\n", log_zero_x_mFlag, log_zero_y_mFlag);
		fprintf( fp, "-log_zero_sym_{x,y} S\tShow symbol S at log_zero location [%s,%s]\n", log_zero_sym_x, log_zero_sym_y);
		fputs(       "                     \t\t(default: 0*)\n", fp);
		fprintf( fp, "-lw w\tset width of lines [%g]\n", lineWidth);
		fprintf( fp,	"-lx x1,x2\tSet x axis to interval x1,x2 [%g,%g]\n", startup_wi.R_UsrOrgX, startup_wi.R_UsrOppX );
		fprintf( fp,	"-ly y1,y2\tSet y axis to interval y1,y2 [%g,%g]\n", startup_wi.R_UsrOrgY, startup_wi.R_UsrOppY );
		fprintf( fp,    "-LX , -LY\tidem; no padding (hard interval)\n");
		fputs(	"-m -M\tMark points distinctively (M varies with color)\n", fp);
#ifdef __GNUC__
		fprintf( fp, "-maxlbuf\tspecify maximum length of (data) lines [%d]\n", LMAXBUFSIZE );
#else
		fprintf( fp, "-maxlbuf\tspecify maximum length of (data) lines [%d]\n", LMAXBUFSIZE );
#endif
		fprintf( fp, "-maxsets\tspecify maximum number of sets [%d]\n", MaxSets );
		fprintf( fp, "-mf[01]\tUse markFont or builtin (%d) bitmaps for markers [%d]\n", MAXATTR, use_markFont );
		fprintf( fp,	"-mlx x1,x2\tLimit x axis to interval x1,x2 [%g,%g]\n", MusrLX, MusrRX );
		fprintf( fp,	"-mly y1,y2\tLimit y axis to interval y1,y2 [%g,%g]\n", MusrLY, MusrLY );
		fputs(	"-nl\tDon't draw lines (scatter plot)\n", fp);
		fputs(  "-noaxis\tDon't draw axes\n", fp);
		fputs(  "-nodetach\tDisable the -detach option. For debugging...\n", fp );
		fputs(	"-noerr\tDon't draw errorbars. Can be changed in settings dialog,\n", fp);
		fputs(	"      \tso scaling accounts for values in error column\n", fp);
		fprintf( fp, "-noreal_{x,y}_val\tShow transformed values along axes, instead of the real values [%d,%d]\n",
			!UseRealXVal, !UseRealYVal
		);
		fputs(  "-nosort\tSpreadSheet save will not sort on XValue\n", fp);
		fprintf( fp, "-nosplit[01]\tIgnore *SPLIT* commands in input [%d]\n", ignore_splits );
		fprintf( fp, "-overwrite_legend\ttoggle overwriting of legend over data or vs. [%d]\n", overwrite_legend );
		fprintf( fp, "-overwrite_marks\ttoggle overwriting of marks over data or vs. [%d]\n", overwrite_marks );
		fputs(	"-p -P\tMark points with dot (P means big dot)\n", fp);
		fputs( "-PN <expr>\tevaluate <expr> as an *EVAL* (was: *PARAM_NOW*) command\n", fp);
		fprintf( fp, "-preserve_aspect\tPreserve the on-screen aspect-ratio in the PS output [%d]\n",
			preserve_screen_aspect
		);
		fprintf( fp, "-preserve_ftime\tPreserve the output file's time stamps [%d]\n",
			XG_preserve_filetime
		);
		fputs(  "-print\tImmediately post hardcopy dialog; don't draw all data first\n", fp);
		fputs(	     "    \t\talso activated by zooming with 2nd mousebutton\n", fp);
		fprintf( fp, "-print_as <type>\tspecify type of print: <%s", hard_devices[0].dev_name );
			for( i= 1; i< hard_count; i++ ){
				fprintf( fp, "|%s", hard_devices[i].dev_name );
			}
			fprintf( fp, "> [%s]\n", Odevice );
		fprintf( fp, "-print_to <dev>\tspecify disposition of print: <\"%s\"", fodstrs[0] );
			for( i= 1; i< fod_num; i++ ){
				fprintf( fp, "|\"%s\"", fodstrs[i] );
			}
			fprintf( fp, "> [%s]\n", Odisp );
		fputs(  "-printOK\tImmediately do hardcopy; don't draw all data first\n", fp);
		fprintf( fp, "-process_bounds\tToggle *DATA_PROCESS* processing of axes-bounds[%d]\n", process_bounds );
		fprintf( fp, "-progress\tToggle progress indication of PARAM_FUNCTIONS (can speed up 3x!) [%d]\n", Show_Progress );
		fprintf( fp, "-PSm\tset base and increment size for PS markers [%g,%g]\n", psm_base, psm_incr+1);
		fprintf( fp, "-maxWidth|-maxHeight\thardcopy output dimensions [%g x %g]\n",
				hard_devices[PS_DEVICE].dev_max_width, hard_devices[PS_DEVICE].dev_max_height
		);
		fprintf( fp, "-page\tAll debug output to $PAGER [%s]\n", (getenv("PAGER"))? getenv("PAGER") : "" );
		fputs(	"-pf\tFilename for printfile (extension is added)\n", fp);
		fputs(	"-plot_only_file <#>\tInitially plot only sets in file number <#>\n", fp);
		fputs(	"-plot_only_set <list>\tInitially plot only sets given in <list>\n", fp);
		fprintf( fp,	"-polar[base]\tDraw data in polar plot, with base (e.g. 360, or PI) [%g]\n", radix);
		fputs(	"-print\tpost a hardcopy dialog without drawing the data\n", fp);
		fputs(	"-printOK\tpost a hardcopy dialog without drawing the data, print and exit\n", fp);
		fprintf( fp, "-PrintInfo\tprint (PostScript only) the comments in the datafiles on a separate page [%d]\n", PS_PrintComment);
		fprintf( fp, "-progname <name>\tset name of program to <name>: this is used to find X defaults [%s]\n", Prog_Name);
		fprintf( fp, "-powA <val>\tPower value for sin(x) and cos(x) in polar mode [%g]\n", powAFlag );
		fprintf( fp, "-pow{x,y}[01] <val>\tPower value for \"sqrt\" scale for X or Y axis [%g,%g]\n", powXFlag, powYFlag );
		fprintf( fp, "-radix <base>\tSpecify the radix, or base, for gonio (e.g. 360, or PI) [%g]\n", radix);
		fprintf( fp, "-radix_offset <offset>\tSpecify the offset, for gonio (e.g. 90 for 0deg=upwards when radix==360) [%g]\n", radix_offset);
		fprintf( fp, "-raw_display\tDisplay raw data, without runtime transformations [%d]\n", raw_display );
		{ int rVN = register_VariableNames(1);
			fprintf( fp, "-registry[0]\tmaintain a registry of ascanf object names [%d]\n", rVN );
			register_VariableNames(rVN);
		}
		fprintf( fp, "-ResetAttr\tAutomatic style-designation, reset for each new file[%d]\n", ResetAttrs );
		fputs(	"-rv\tReverse video on black and white displays\n", fp);
		fprintf( fp, "-SameAttr\tNo automatic linestyle-designation: new style is inherited from the current set's style [%d]\n", SameAttrs );
		fprintf( fp,  "-scale_av_{x,y} X\tset x,y range to average of x,y +/- X times the st.deviation [%g,%g]\n",
			_scale_av_x, _scale_av_y
		);
		fprintf( fp, "-separator <c>|0xnn\tuse character 'c' (or char with code 0xnn) as the separator between data-columns [%c=0x%x]\n",
			data_separator, data_separator
		);
		fprintf( fp, "-show_overlap [raw]\ttoggle the display of the average overlap in the legend box [%d]\n", show_overlap );
		fprintf( fp,	"-spa{x,y,}[01]\tmaxWidth and maxHeight apply to plot-regio or complete plot [%d,%d]\n",
			scale_plot_area_x, scale_plot_area_y
		);
		fprintf( fp, "-splits_disconnect[01]\t whether a *SPLIT* command disconnects to start a new dataset [%d]\n", splits_disconnect );
		fprintf( fp,	"-sqrt{x,y}[01]\tLinear / Square-root scale for X or Y axis [%d,%d]\n", sqrtXFlag, sqrtYFlag );
		fprintf( fp,	"-stats\tprint some statistics about each *FILE* [%d]\n", show_stats);
		fprintf( fp,	"-sV\tgive feedback on successfull execution of \"scripting\" commands [%d]\n", scriptVerbose );
		fprintf( fp,	"-SwapEndian[01]\tswap little to big endian (or reverse) for binary input [%d]\n", SwapEndian );
		fputs( "-t|-T <title>\tset title (-t) or specify additional title (-T)\n", fp );
		fprintf( fp,    "-transform_axes\tAxes numbers show result of TRANSFORM_[XY] processing [%d]\n", transform_axes );
		fputs(	"-tk[01]\tGrid / tick marks\n", fp);
		fputs(	"-triangle\tDraw error \"triangles\"\n", fp);
		fprintf( fp,	"-use_XDBE[0|1]\tuse the X11 DoubleBuffer Extension [%d]\n", ux11_useDBE );
		fprintf( fp, "-vectors <length>\tselects error-column=vector, and specifies the length of the orientation-vector in following sets [%d,%s]\n",
			vectorFlag, d2str( vectorLength, NULL, NULL)
		);
		fprintf( fp, "-vectorpars <type>,<length>[,pars]\tselects error-column=vector, and specifies the length of the orientation-vector in following sets [%d,%d,%s]\n",
			vectorFlag, vectorType, d2str( vectorLength, NULL, NULL)
		);
		fprintf( fp, "-wns[0|1]\tWarn if a new set is started, giving information on where it occurs [%d].\n", WarnNewSet );
		fprintf( fp, "-XGBounds\ttoggle including definition of bounding-box in an XGraph dump [%d]\n", XG_SaveBounds );
		fprintf( fp, "-XGraph\tOnly accepted as first argument. This has the effect that xgraph immediately relaunches\n"
		             "       \titself through the XGraph.sh script (expected in the same directory).\n"
		);
		fprintf( fp, "-{x,y}step\tControls labelling of {x,y} axis [%g,%g]\n", Xincr_factor, Yincr_factor );
		fprintf( fp, "-{x,y}_ul <x>,<y>\tplace the upper left corner of the {x,y} unitname at co-ordinate (x,y) [%g,%g] [%g,%g]\n\tCan also be done by Shift-clicking with the {3rd,2nd} mousebutton\n",
			xname_x, xname_y,
			yname_x, yname_y
		);
		fprintf( fp, "-zero eps\ttreat eps as zero if range is larger [%.15g]\n", zero_epsilon);
		fputs(	"-zl[01]\tdraw lines x=0 and y=0\n", fp);
		fputs(	"Command line arguments override settings in inputfiles\n", fp);
		fputs( "\tIn (almost) all strings, a '\\' switches back and forth between greekFont and the normal font;\n", fp );
		fputs( "\t   #xUV is substituted for a character with hexcode 0xUV .\n", fp );
		if( !val){
		  extern char XGJoinBytes1[], XGJoinBytes2[];
			fputs( "\nFormat options in inputfile:\n", fp);
			fputs( "\t*BUFLEN* <len>\tset maxlbuf to <len> : must come before any data\n", fp);
			fputs( "\t*TITLE*\t<title>\n", fp);
			fputs( "\t*XLABEL* (*XYLABEL*)\t<{y- or} x-axis label>\n", fp);
			fputs( "\t*YLABEL* (*YXLABEL*)\t<{x- or} y-axis label>\n", fp);
			fputs( "\t\t*[XY][YX]LABEL* variants are column-sensitive, and are stored locally per set\n", fp );
			fputs( "\t*FILE*\t<filename>\n", fp);
			fputs( "\t*LEGEND*\t<set_name>\n", fp);
			fputs( "\t*LEGTXT* <set_name_lines>\tappends to the setname from the following lines upto the 1st blank\n", fp);
			fputs( "\t*XAXISDAT*\t<MajTic>(unsupp) <SubDiv>(unsupp) <axmin> <axmax> <axlog> <dum> <more>\n", fp);
			fputs( "\t          \t\t<more> lines to skip\n", fp);
			fputs( "\t*YAXISDAT*\t<MajTic>(unsupp) <SubDiv>(unsupp) <axmin> <axmax> <axlog> <dum> <more>\n", fp);
			fputs( "\t          \t\t<more> lines to skip\n", fp);
			fputs( "\t*PROPERTIES*\t1 <errorFlags> <AxSpec> <margin>(unsupp)\t(global settings)\n", fp);
			fputs( "\t            \t               0: ticks, nobox; ", fp);
				fputs( "1: grid, nobox (default); ", fp);
				fputs( "2: ticks, box; ", fp);
				fputs( "3: grid, box; ", fp);
				fputs( "11,12,13: idem, with -zl flag set\n", fp);
			fputs( "\t*PROPERTIES*\t<colour>(unsupp) <type> <linestyle> <marksize>(unsupp) <linewidth> <errorlinestyle> <marker> <errorlinewidth>\n", fp);
			fputs( "\t            \tObsolete <type>s:  ^ 0: line; ", fp);
				fputs( "1: scatter (-nl -m); ", fp);
				fputs( "2: scatter,line (-m); ", fp);
				fputs( "4: bar (-bar -nl -m)\n", fp);
			fprintf( fp,
				   "\t            \tNew <type> bitmask 100000<mrk><srd>-<hl><ne><SL><OM><b>XX<-nl>\n"
				   "\t            \t   XX: 01=-p 10=-P 11=-m\n"
				   "\t            \t   mrk: set marked\n"
				   "\t            \t   srd: set raw_display\n"
				   "\t            \t   hl: set highlighted\n"
				   "\t            \t   ne: set doesn't show errors\n"
				   "\t            \t   SL: set shows in LegendBox\n"
				   "\t            \t   OM: set's markers overwrite\n"
				   "\t            \t   b: set is bar graph\n"
				   "\t            \t   -nl: set is dot graph\n"
			);
			fputs( "\t            \t<linestyle>                ^1: solid line\n", fp);
			fputs( "\t            \t<marker>: number >= 1. Defaults are saved as <= -1 (which have no effect)\n", fp);
			fputs( "\t*SCALEFACT*\t<file-wide XScale> <file-wide YScale> <file-wide ErrorScale>\n", fp);
			fputs( "\t*SCALFACT*\t<set-spec. XScale> <set-spec. YScale> <set-spec. ErrorScale>\n", fp);
			fputs( "\t*EXTRATEXT*\tstarts Multiplot private section until next empty line\n", fp);
			fputs( "\t*ELLIPSE* x y rx ry points\tgenerates an ellipse\n", fp);
			fputs( "\t*PARAM_RANGE* min,max[,points]\tspecify range for the 'self' function in the next *PARAM_FUNCTIONS* call\n", fp);
			fputs( "\t*ASK_EVAL*message[::default expressions]\tAsk for expressions to be evaluated immediately\n", fp);
			fputs( "\t*EVAL* expressions\tevaluated when encountered\n", fp);
			fputs( "\t*PARAM_BEFORE* expressions\tevaluated before the *PARAM_FUNCTIONS* call\n", fp);
			fputs( "\t*PARAM_AFTER* expressions\tevaluated after the *PARAM_FUNCTIONS* call\n", fp);
			fputs( "\t*PARAM_FUNCTIONS* exp1,exp2,exp3\tdetermine x,y,error according to exp1,exp2,exp3\n", fp);
				fputs( "\t\twhich are expressions understood by fascanf: numbers or functions.\n", fp);
				if( err && strcasecmp( err, "-fascanf_functions" )== 0 ){
					show_ascanf_functions( fp, "\t\t\t", 1, 1);
				}
				else{
					fputs( "\t\tFor an overview of these functions, give option -fascanf_functions\n", fp);
				}
			fputs( "\tData processing functions: data in DATA[0]..DATA[2] X,Y,E columnnumber in COLUMN[0]..COLUMN[2]\n", fp);
			fputs( "\t*DATA_BEFORE* expressions\tevaluated before the *DATA_PROCESS* call\n", fp);
			fputs( "\t*DATA_PROCESS* exp1,exp2,exp3\tdetermine x,y,error according to exp1,exp2,exp3\n", fp);
			fputs( "\t*DATA_AFTER* expressions\tevaluated after the *DATA_PROCESS* call\n", fp);
			fputs( "\t*TRANSFORM_X* exp\tspecify a transformation function of DATA[0] for the x-axis\n", fp);
			fputs( "\t*TRANSFORM_Y* exp\tspecify a transformation function of DATA[0] for the y-axis\n", fp);
			fputs( "\t*ERROR_POINT* nr\tspecify a single point that will show its error-flag\n", fp );
			fputs( "\t*ARROWS* B|E|BE [orn[,orn]]\tadd start, end or start&end arrow\n", fp);
			fputs( "\t*POINTS* nr\tnumber of points per set until further notice\n", fp );
			fputs( "\t*ARGUMENTS* <commandline options>\n", fp);
			fputs( "\t*READ_FILE*<name> or *READ_FILE*<| command>\tinsert data from file or pipe\n", fp);
			fputs( "\t*SPLIT* (reason)\tcreate a new set at this point, for (reason), which is appended\n\
	\t               \tafter the current legend\n", fp
			);
			fprintf( fp, "\t*N*<exp>\tspecifies the number of observations made for each datapoint in following sets [%d]\n",
				NumObs
			);
			fprintf( fp, "\t*VECTORLENGTH*<exp>\tselects error-column=vector, and specifies the length of the orientation-vector in following sets [%s]\n",
				d2str( vectorLength, NULL, NULL)
			);
			fprintf( fp, "\t*AVERAGE* s1 s2 ...\tCalculates the average of the X and Y,E in sets s1, s2 ...\n" );
			fprintf( fp, "\t*ULABEL* x1 y1 x2 y2 set transform? draw?\tDefines a UserLabel. The arrow-side is given by (x1,y1),\n"
						 "\t                                   \tthe label-side by (x2,y2). The label is read from the lines\n"
						 "\t                                   \tupto the first empty line following the *ULABEL* command\n"
			);
			fprintf( fp, "\n\n\tInput lines can be split over multiple lines for several commands when either the command is\n"
				"\tfollowed by \\n or preceded by a *; in both cases nothing may follow it on that same line. The concerned\n"
				"\tcommands are (most of) those for which the 1st and 2nd letter are contained in the strings below:\n"
				"\t1st: \"%s\" 2nd: \"%s\"\n",
				XGJoinBytes1, XGJoinBytes2
			);
		}
	}
	if( fp!= StdErr){
		pclose( fp);
	}
	if( Opt01== 1 && isatty(fileno(stdin)) ){
		system( "man xgraph" );
	}
	if( ArgsParsed ){
		ArgError= True;
		return( (err && val) );
	}
	else{
		exit( (err && val) );
	}
}

char *option_history= NULL;
int option_history_size= 0;
xtb_frame *option_pmenu;

char *add_option_hist( char *expression )
{ int len= ((expression)? strlen(expression) : 0)+1;
  char *c;
  char *exp;
	if( !expression ){
		xfree( option_history );
		option_history_size= 0;
		return( NULL );
	}
	if( len== 1 ){
		return( option_history );
	}
	  /* remove trailing whitespace	*/
	{ int i= 0;
		c= &expression[strlen(expression)-1];
		while( isspace((unsigned char)*c) && c> expression ){
			i++;
			c--;
		}
		if( i ){
			c[1]= '\0';
		}
	}
	exp = (char*) malloc( len * sizeof(char) );
	if( !exp ){
		return option_history;
	}
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
	if( option_history ){
		if( strstr( option_history, exp) ){
			xfree(exp);
			return( option_history );
		}
	}
	len+= 1;
	if( len ){
		option_history= realloc( option_history, option_history_size+ len+ 2 );
		if( !option_history_size ){
		  /* initial text	*/
			c= option_history;
		}
		else{
		  /* append	*/
			c= &option_history[strlen(option_history)];
		}
		{ char *d= exp;
			  /* prepend a whitespace to separate and to left-align in the popup menu	*/
			*c++= ' ';
			while( *d ){
				*c++= *d++;
			}
		}
		*c++= '\n';
		*c= '\0';
		option_history_size+= len;
		xtb_popup_delete( &option_pmenu );
	}
	xfree(exp);
	return( option_history );
}

void option_hist(LocalWin *wi)
{

	if( option_history && *option_history ){
	  char *sel= NULL;
	  int id;
		id= xtb_popup_menu( wi->window, option_history, "Options History - selected is copied to clipboard", &sel, &option_pmenu );
		while( sel && *sel && isspace((unsigned char) *sel) ){
			sel++;
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
	}
	else{
		xtb_error_box( wi->window, "None", "Options History" );
	}
}


#ifdef _AUX_SOURCE
	extern int strncasecmp();
	extern int strcasecmp();
#endif
int Check_Option( int (*compare_fun)(), char *arg, char *check, int len)
{  int ret, ret2= 1, alen, clen;
	if( !arg || !check ){
		Opt01= 0;
		return(0);
	}
	alen= strlen(arg);
	clen= strlen(check);
	if( len<= 0 ){
		len= strlen( check);
	}
	if( compare_fun== strncmp ){
		ret= strncmp( arg, check, len );
	}
	else if( compare_fun== strncasecmp ){
		ret= strncasecmp( arg, check, len);
	}
	else if( compare_fun== strcmp ){
		if( (ret= strcmp( arg, check )) ){
			if( debugFlag ){
				ret2= strncmp( arg, check, len);
			}
		}
	}
	else if( compare_fun== strcasecmp ){
		if( (ret= strcasecmp( arg, check )) ){
			if( debugFlag ){
				ret2= strncasecmp( arg, check, len);
			}
		}
	}
	if( ret== 0 ){
		if( isdigit( (unsigned)arg[len]) ){
			Opt01= (int) (arg[len] - '0');
		}
		else if( arg[len]== '-' && isdigit( (unsigned)arg[len+1]) ){
			Opt01= - (int) (arg[len+1] - '0');
		}
		else if( clen< alen ){
			if( isdigit( (unsigned)arg[clen]) ){
				Opt01= (int) (arg[clen] - '0');
			}
			else if( arg[clen]== '-' && isdigit( (unsigned)arg[clen+1]) ){
				Opt01= - (int) (arg[clen+1] - '0');
			}
		}
		else if( isdigit( (unsigned)arg[alen-1]) ){
			Opt01= (int) (arg[alen-1] - '0');
			if( arg[alen-2]== '-' ){
				Opt01*= -1;
			}
		}
		else{
			Opt01= -1;
		}
		if( debugFlag ){
			fprintf( StdErr, "Check_Option(\"%s\"): found \"%s\" with switch %d\n",
				arg, check, Opt01
			);
		}
		add_option_hist( arg );
	}
	else if( debugFlag && ret2== 0 ){
		fprintf( StdErr, "Check_Option(\"%s\"): matches argument \"%s\" over %d bytes\n",
			arg, check, len
		);
	}
	return( ret );
}

Boolean changed_Allow_Fractions= False;

int ps_xpos= 1, ps_ypos= 1;
double ps_scale= 100.0;
extern double ps_l_offset, ps_b_offset;

int XGStoreColours= 0, XGIgnoreCNames= 0;
int reverseFlag= 0;
extern Pixel highlightPixel, echoPix;

int GetColorDefault( char *value, Pixel *tempPixel, char *def_res_name, char *def_col_name, Pixel *def_pix )
{
	if( strcasecmp( value, "default") ){
		return( GetColor( value, tempPixel ) );
	}
	else{
		*tempPixel= -1;
		if( def_res_name ){
		 extern int rd_pix();
		 extern Pixel def_pixel;
			if( rd_pix( def_res_name ) ){
				*tempPixel= def_pixel;
			}
		}
		if( *tempPixel== -1 && def_col_name ){
			if( !GetColor( def_col_name, tempPixel ) ){
				*tempPixel= -1;
			}
		}
		if( *tempPixel== -1 && def_pix ){
			*tempPixel= *def_pix;
		}
		return( (*tempPixel!= -1) );
	}
}

extern double win_aspect, highlight_par[];
extern int highlight_npars, highlight_mode;
extern double Font_Width_Estimator;
extern int TrueGray;


extern int AlwaysDrawHighlighted;
extern GC ACrossGC, BCrossGC;

void Add_InFile( char *fname )
{ char *tsbuf;
  /* Should be an input file */
	if( numFiles>= MaxnumFiles ){
		if( !realloc_FileNames( NULL, MaxnumFiles, MaxnumFiles+ MaxnumFiles/2, "Add_InFile()") ){
			CleanUp();
			exit(-1);
		}
	}
	inFileNames[numFiles] = XGstrdup(fname);
	InFiles= concat2( InFiles, "'", fname, "' ", NULL);
	if( (tsbuf = (char*) malloc( (strlen(fname)+256) * sizeof(char) )) ){
		InFilesTStamps= concat2( InFilesTStamps, "'", time_stamp( NULL, fname, tsbuf, True, NULL), "' ", NULL);
		xfree(tsbuf);
	}
	else{
		InFilesTStamps= concat2( InFilesTStamps, "'", "(timestamp unavailable)", "' ", NULL);
	}
	numFiles++;
}

int ParseArgs( int argc, char *argv[])
/*
 * This routine parses the argument list for xgraph.  There are too
 * many to mention here so I won't.
 \ 950703 (RJB): sic!!
 */
{ int idx, set, r= 0;
  Pixel tempPixel;
  char *font_name;

	idx = 1;
	ArgError= False;
	while (idx < argc && !ArgError ){

		if( debugFlag ){
			fprintf( StdErr, "ParseArgs(%d,%s)\n", idx, argv[idx] );
		}

		if( !argv[idx] || !argv[idx][0] ){
			argerror( "Empty or missing argument!", argv[idx] );
		}
		if( argv[idx][0]== '-' && argv[idx][1]== '-' && argv[idx][2] ){
		  /* hack in some support for linux-style-like arguments (--arg)	*/
			strcpy( argv[idx], &argv[idx][1] );
		}

		if( Check_Option( strncasecmp, argv[idx], "-help", 5)== 0 ){
		    argerror( NULL, NULL);
		}
		else if( strncasecmp( argv[idx], "-separator", 10)== 0 ){
		  char c;
			if( idx+1 >= argc ){
				argerror( "missing separator", argv[idx]);
			}
			c = argv[idx+1][0];
			if( c > 0 && c<= 127 && !isdigit(c) && !isalpha(c) && !index( "[]{}\"`&@$*", c) ){
			  extern char ascanf_separator;
				ascanf_separator= c;
			}
			else{
				argerror( "invalid separator value", argv[idx+1] );
			}
			idx+= 2;
		}
		else if( strcasecmp( argv[idx], "-fascanf_functions")== 0 ){
			Opt01= 0;
			argerror( argv[idx], NULL);
		}

		else if( Check_Option( strncasecmp, argv[idx], "-psn_", -1)== 0
			&& isdigit(argv[idx][5]) && argv[idx][6]=='_'
		){
			if( debugFlag ){
				fprintf( StdErr, "ignoring Mac OS X window-manager argument \"%s\"\n", argv[idx] );
			}
			MacStartUp= True;
			idx += 1;
		}

		else if( argv[idx][0] == '-' && argv[idx][1] ){
			  /* Check to see if its a data set name */
			  /* 20000407: I never use this option. Deactivated it.	*/
			if( False && sscanf(argv[idx], "-%d", &set) == 1 ){
			/* The next string is a set name */
				if (idx+1 >= argc){
					argerror("missing set name", argv[idx]);
				}
				else{
					xfree( AllSets[set].setName );
					AllSets[set].setName = XGstrdup(argv[idx+1]);
					idx += 2;
				}
			}
			else{
				  /* Some non-dataset option */
				if( Check_Option( strcmp, argv[idx], "-PSm", -1)==0 ){
					if (idx+1 >= argc){
						argerror( "missing base,incr", argv[idx]);
					}
					else{
						psm_incr+= 1.0;
						sscanf( argv[idx+1], "%lf,%lf", &psm_base, &psm_incr);
						if( debugFlag){
							fprintf( StdErr, "PSm base,incr= %g,%g\n", psm_base, psm_incr);
						}
						psm_incr-= 1.0;
						psm_changed+= 1;
						idx += 2;
					}
				}
				else if( Check_Option( strncmp, argv[idx], "-readline", 9)==0 ){
					Conditional_Toggle(Use_ReadLine);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-IPN", 4)==0 ){
				  extern int read_params_now;
					Conditional_Toggle(read_params_now);
					idx+= 1;
				}
				else if( Check_Option( strcmp, argv[idx], "-PN", -1)==0 ){
				  double val= 0.0;
					if (idx+1 >= argc){
						argerror( "missing expression", argv[idx]);
					}
					else{
						new_param_now( argv[idx+1], &val, -1 );
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-python", 3)==0 ){
				  extern DM_Python_Interface *dm_python;
				  extern int Init_Python();
					if (idx+1 >= argc){
						argerror( "missing expression", argv[idx]);
					}
					else{
						if( Init_Python() && dm_python->Import_Python_File ){
							(*dm_python->Import_Python_File)( argv[idx+1], NULL, 0, False );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-IO_Import", 7)==0 ){
					if (idx+1 >= argc){
						argerror( "missing IO_Import module", argv[idx]);
					}
					else if (idx+2 >= argc){
						argerror( "missing data filename", argv[idx]);
					}
					else{
						if( debugFlag ){
							fprintf( StdErr, "Immediate import from \"%s\" using IO_Import module \"%s\" into set #%d\n",
								argv[idx+2], argv[idx+1], setNumber
							);
						}
						IOImport_Data( argv[idx+1], argv[idx+2] );
						idx += 3;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-x_ul", 5)==0 ){
				  double ul[2]= {0,0};
				  int r, n= 2;
					if (idx+1 >= argc){
						argerror( "missing x,y", argv[idx]);
					}
					else{
// 						r = fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						r = fascanf2( &n, argv[idx+1], ul, ',' );
						if( r == 2 ){
							xname_placed= 1;
						}
						else if( n== 1 ){
							xname_placed= 2;
						}
						else if( n== 0 ){
							xname_placed= 3;
						}
						xname_x= ul[0];
						xname_y= ul[1];
						if( Opt01== 1 ){
							xname_trans= 1;
						}
						if( debugFlag){
							if( xname_placed ){
								fprintf( StdErr, "xname upperleft= %s,%s\n",
									d2str( xname_x, "%g", NULL), d2str( xname_y, "%g", NULL)
								);
							}
							else{
								fprintf( StdErr, "xname upperleft: missing co-ordinate(s)\n");
							}
							fflush( StdErr );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-y_ul", 5)==0 ){
				  double ul[2]= {0,0};
				  int r, n= 2;
					if (idx+1 >= argc)
						argerror( "missing x,y", argv[idx]);
					else{
// 						r = fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						r = fascanf2( &n, argv[idx+1], ul, ',' );
						if( r == 2 ){
							yname_placed= 1;
						}
						else if( n== 1 ){
							yname_placed= 2;
						}
						else if( n== 0 ){
							yname_placed= 3;
						}
						yname_x= ul[0];
						yname_y= ul[1];
						if( Opt01== 1 ){
							yname_trans= 1;
						}
						if( debugFlag){
							if( yname_placed ){
								fprintf( StdErr, "yname upperleft= %s,%s\n",
									d2str( yname_x, "%g", NULL), d2str( yname_y, "%g", NULL)
								);
							}
							else{
								fprintf( StdErr, "yname upperleft: missing co-ordinate(s)\n");
							}
							fflush( StdErr );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-y_vert", 7)==0 ){
					yname_vertical= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-legend_ul", 10)==0 ||
					Check_Option( strncasecmp, argv[idx], "-legend_placed", 14)== 0
				){
				  double ul[2]= {0,0};
				  int n= 2;
					if (idx+1 >= argc){
						argerror( "missing x,y", argv[idx]);
					}
					else{
						if( strncasecmp( argv[idx+1], "not", 3)== 0 || strncasecmp( argv[idx+1], "off", 3)== 0 ){
							legend_placed= 0;
						}
// 						else if( fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL)== 2 )
						else if( fascanf2( &n, argv[idx+1], ul, ',' )== 2 )
						{
							legend_placed= 1;
						}
						else if( n== 1 ){
							legend_placed= 2;
						}
						else if( n== 0 ){
							legend_placed= 3;
						}
						if( legend_placed ){
							legend_ulx= ul[0];
							legend_uly= ul[1];
							if( Opt01== 1 ){
								legend_trans= 1;
							}
						}
						if( debugFlag){
							if( legend_placed ){
								fprintf( StdErr, "Legend upperleft= %s,%s\n",
									d2str( legend_ulx, "%g", NULL), d2str( legend_uly, "%g", NULL)
								);
							}
							fflush( StdErr );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-intensity_legend_ul", 10)==0 ||
					Check_Option( strncasecmp, argv[idx], "-intensity_legend_placed", 14)== 0
				){
				  double ul[2]= {0,0};
				  int n= 2;
					if (idx+1 >= argc)
						argerror( "missing x,y", argv[idx]);
					else{
// 						if( fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL)== 2 )
						if( fascanf2( &n, argv[idx+1], ul, ',' )== 2 )
						{
							intensity_legend_placed= 1;
						}
						else if( n== 1 ){
							intensity_legend_placed= 2;
						}
						else if( n== 0 ){
							intensity_legend_placed= 3;
						}
						intensity_legend_ulx= ul[0];
						intensity_legend_uly= ul[1];
						if( Opt01== 1 ){
							intensity_legend_trans= 1;
						}
						if( debugFlag){
							if( legend_placed ){
								fprintf( StdErr, "Intensity legend upperleft= %s,%s\n",
									d2str( intensity_legend_ulx, "%g", NULL), d2str( intensity_legend_uly, "%g", NULL)
								);
							}
							else{
								fprintf( StdErr, "Intensity legend upperleft: missing co-ordinate(s)\n");
							}
							fflush( StdErr );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-progname", 6)== 0 ){
					if (idx+1 >= argc || argv[idx+1][0]== 0 )
						argerror( "missing or invalid programname", argv[idx]);
					else{
						xfree( Prog_Name );
						Prog_Name= XGstrdup( argv[idx+1] );
						progname= 1;
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-Ignore_UnmapNotify", -1)== 0){
				  extern int Ignore_UnmapNotify;
					Ignore_UnmapNotify= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-iconic", 5)== 0){
					IconicStart= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-root", 5)== 0){
					use_RootWindow= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-XGDump_PrintPars", 17)== 0 ){
				  extern int XGDump_PrintPars;
					Conditional_Toggle( XGDump_PrintPars );
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-XGBounds", 9)== 0 ){
					Conditional_Toggle( XG_SaveBounds );
					idx++;
				}
				else if( Check_Option( strncmp, argv[idx], "-NoIncludes", 11)== 0 ){
					Conditional_Toggle( No_IncludeFiles );
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-nosort", 6)== 0){
					Sort_Sheet= 0;
					if( debugFlag)
						fprintf( StdErr, "SpreadSheet will not be sorted on XValue\n");
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-nosplit", 8)== 0){
					ignore_splits= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-splits_disconnect", 18)== 0){
					splits_disconnect= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-page", 5)== 0){
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-PrintInfo", 10)== 0){
					Conditional_Toggle(PS_PrintComment);
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-preserve_aspect", 13)== 0){
					preserve_screen_aspect= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-preserve_ftime", 15)== 0 ){
					Conditional_Toggle(XG_preserve_filetime);
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-print_as", 9)== 0){
					if (idx+1 >= argc){
						argerror( "missing specification", argv[idx]);
					}
					else{
						Odevice= XGstrdup( argv[idx+1] );
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-print_to", 9)== 0){
					if (idx+1 >= argc){
						argerror( "missing specification", argv[idx]);
					}
					else{
						Odisp= XGstrdup( argv[idx+1] );
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-print_sized", 10)== 0){
					size_window_print= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_xpos", 8)== 0){
					if (idx+1 >= argc){
						argerror( "missing specification", argv[idx]);
					}
					else{
						CLIP_EXPR( ps_xpos, atoi( argv[idx+1] ), 0, 2);
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_ypos", 8)== 0){
					if (idx+1 >= argc){
						argerror( "missing specification", argv[idx]);
					}
					else{
						CLIP_EXPR( ps_ypos, atoi( argv[idx+1] ), 0, 2);
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_rgb", 7)== 0){
				  extern int ps_coloured;
					ps_coloured= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_transp", 10)== 0){
				  extern int ps_transparent;
					ps_transparent= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-gs_twidth_batch", 16)== 0){
					*do_gsTextWidth_Batch= (Opt01)? 1 : 0;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-gs_twidth", 10)== 0){
				  extern int use_gsTextWidth;
					use_gsTextWidth= (Opt01)? 1 : 0;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-gs_twidth_auto", 15)== 0){
				  extern int auto_gsTextWidth;
					auto_gsTextWidth= (Opt01)? 1 : 0;
					idx+= 1;
				}
				else if( Check_Option( strcmp, argv[idx], "-ps_fest", -1)==0 ){
				  double x;
					if (idx+1 >= argc)
						argerror( "missing base,incr", argv[idx]);
					else{
						if( sscanf( argv[idx+1], "%lf", &x)== 1 ){
							Font_Width_Estimator= x;
							set_HO_printit_win();
							if( debugFlag){
								fprintf( StdErr, "Font_Width_Estimator=%g\n", x);
							}
						}
						idx += 2;
					}
				}
				else if( Check_Option( strcmp, argv[idx], "-ps_offset", -1)==0 ){
				  double x, y;
					if (idx+1 >= argc)
						argerror( "missing base,incr", argv[idx]);
					else{
						if( sscanf( argv[idx+1], "%lf,%lf", &x, &y)== 2 ){
							ps_l_offset= x;
							ps_b_offset= y;
							if( ActiveWin ){
								ActiveWin->ps_l_offset= x;
								ActiveWin->ps_b_offset= y;
							}
							if( debugFlag){
								fprintf( StdErr, "PS left, bottom offset= %g,%g\n", x, y);
							}
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_scale", 8)== 0){
					if (idx+1 >= argc){
						argerror( "missing specification", argv[idx]);
					}
					else{
						if( (ps_scale= atof( argv[idx+1] ))< 0 ){
							ps_scale= 0;
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_s_inc", 9)== 0){
				  extern XGStringList *PSSetupIncludes;
					if (idx+1 >= argc){
						argerror( "missing filename", argv[idx]);
					}
					else{
					  char exp[MAXCHBUF*2];
						PSSetupIncludes= XGStringList_AddItem( PSSetupIncludes, tildeExpand( exp, argv[idx+1] ) );
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_eps", 7)== 0){
				  extern int psEPS;
					psEPS= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_dsc", 7)== 0){
				  extern int psDSC;
					psDSC= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ps_setpage", 11)== 0){
				  extern int psSetPage;
				  extern double psSetPage_width, psSetPage_height;
				  double w, h;
					psSetPage= Opt01;
					if( idx+1< argc ){
						if( sscanf( argv[idx+1], "%lfx%lf", &w, &h)== 2
							|| sscanf( argv[idx+1], "%lfX%lf", &w, &h)== 2
						){
							psSetPage_width= w;
							psSetPage_height= h;
							idx+= 1;
						}
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-print", 6)== 0){
					if( Check_Option( strncasecmp, argv[idx], "-printOK", 8) == 0)
						print_immediate= -1;
					else
						print_immediate= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-really_incomplete", 18)== 0){
				  extern int XG_Really_Incomplete;
					Conditional_Toggle(XG_Really_Incomplete);
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-settings", 9)== 0){
					settings_immediate= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-monochrome", 5)== 0){
					MonoChrome= 2;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-TrueGray", -1)== 0){
					TrueGray= Opt01;
					idx++;
				}
				else if( Check_Option( strcmp, argv[idx], "-detach0", -1)==0 ){
					if( detach>= 0 )
						detach= 1;
					idx++;
				}
				else if( Check_Option( strcmp, argv[idx], "-detach", -1)==0 ){
					if( detach>= 0 )
						detach= 2;
					idx++;
				}
				else if( Check_Option( strncmp, argv[idx], "-tellPID", -1)==0 ){
				  extern int detach_notify_stdout;
					detach_notify_stdout= (Opt01)? True : False;
					idx++;
				}
				else if( Check_Option( strcmp, argv[idx], "-nodetach", -1)==0 ){
					detach= -1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-stats", 6)== 0 ){
					show_stats= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-Scale_X", -1)== 0){
				  double x;
					if (idx+1 >= argc)
						Xscale2= 1.0;
					else if( sscanf(argv[idx+1], "%lf", &x) )
						Xscale2= x;
					if( debugFlag)
						fprintf( StdErr, "Xscale2=%g\n", Xscale2);
					idx+= 2;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-Scale_Y", -1)== 0){
				  double x;
					if (idx+1 >= argc)
						Yscale2= 1.0;
					else if( sscanf(argv[idx+1], "%lf", &x) )
						Yscale2= x;
					if( debugFlag)
						fprintf( StdErr, "Yscale2=%g\n", Yscale2);
					idx+= 2;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-scale_av_x", -1)== 0){
				  double x;
					if (idx+1 >= argc)
						scale_av_x= _scale_av_x;
					else if( sscanf(argv[idx+1], "%lf", &x) )
						_scale_av_x= scale_av_x= x;
					if( debugFlag)
						fprintf( StdErr, "scale_av_x=%g\n", scale_av_x);
					idx+= 2;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-scale_av_y", -1)== 0){
				  double x;
					if (idx+1 >= argc)
						scale_av_y= _scale_av_y;
					else if( sscanf(argv[idx+1], "%lf", &x) )
						_scale_av_y= scale_av_y= x;
					if( debugFlag)
						fprintf( StdErr, "scale_av_y=%g\n", scale_av_y);
					idx+= 2;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-noreal_x_val", -1)== 0){
					UseRealXVal= 0;
					idx+= 1;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-noreal_y_val", -1)== 0){
					UseRealYVal= 0;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-vectorpars", 11)== 0 ||
					Check_Option( strncasecmp, argv[idx], "-vanepars", 9)== 0
				){
/* 				  double pars[2+MAX_VECPARS];	*/
/* 				  int n= 2+MAX_VECPARS;	*/
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
						if( Parse_vectorPars( argv[idx+1], NULL, True, NULL, argv[idx]) ){
							if( debugFlag ){
								fprintf( StdErr, "vector type=%d vectorLength= %g type1_pars=%g,%g\n",
									vectorType, vectorLength,
									vectorPars[0], vectorPars[1]
								);
							}
							if( Opt01 ){
								vectorFlag= 1;
								error_type= 4;
								FitOnce= True;
							}
							else{
								vectorFlag= 0;
								error_type= -1;
								FitOnce= False;
							}
							if( ActiveWin && ActiveWin->error_type && ActiveWin->error_type[setNumber]== -1 ){
								ActiveWin->error_type[setNumber]= error_type;
							}
							else if( AllSets && AllSets[setNumber].numPoints<= 0 ){
								AllSets[setNumber].error_type= error_type;
							}
						}
						else{
							argerror( "need at least 2 numbers", argv[idx] );
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-vectors", 8)== 0 ||
					Check_Option( strncasecmp, argv[idx], "-vanes", 6)== 0){
				  double x;
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							vectorLength= x;
							if( Opt01 ){
								vectorFlag= 1;
								error_type= 4;
								FitOnce= True;
							}
							else{
								vectorFlag= 0;
								error_type= -1;
								FitOnce= False;
							}
							if( ActiveWin && ActiveWin->error_type && ActiveWin->error_type[setNumber]== -1 ){
								ActiveWin->error_type[setNumber]= error_type;
							}
							else if( AllSets && AllSets[setNumber].numPoints<= 0 ){
								AllSets[setNumber].error_type= error_type;
							}
						}
						if( debugFlag)
							fprintf( StdErr, "vectorLength=%g\n", vectorLength);
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-xstep", -1)== 0){
				  double x;
				  int n= 1;
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
// 						fascanf( &n, argv[idx+1], &x, NULL, NULL, NULL, NULL );
						fascanf2( &n, argv[idx+1], &x, ',' );
						if( n== 1 ){
							ValCat_X_incr= Xincr_factor= x;
						}
						if( debugFlag){
							fprintf( StdErr, "xstep=%g\n", Xincr_factor);
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ystep", -1)== 0){
				  double x;
				  int n= 1;
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
// 						fascanf( &n, argv[idx+1], &x, NULL, NULL, NULL, NULL );
						fascanf2( &n, argv[idx+1], &x, ',' );
						if( n== 1 ){
							ValCat_Y_incr= Yincr_factor= x;
						}
						if( debugFlag){
							fprintf( StdErr, "ystep=%g\n", Yincr_factor);
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-highlight_set", -1)== 0){
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
					  int i,j;
#ifdef NO_SETSREALLOC
					  int n= MAXSETS;
#else
					  int n= MAXSETS* 4;
#endif
#ifdef __GNUC__
					  double list[n];
#else
					  double *list= (double*) calloc( n, sizeof(double));
						if( !list ){
							argerror( "no memory for -highlight_set buffer", argv[idx] );
						}
#endif
						i= n;
// 						fascanf( &i, argv[idx+1], list, NULL, NULL, NULL, NULL );
						fascanf2( &i, argv[idx+1], list, ',' );
						if( i> 0 ){
							if( (highlight_set= (int*) realloc( (char*) highlight_set, (i+1)* sizeof(int) )) ){
								for( j= 0, highlight_set_len= 0; j< i; j++ ){
									if( list[j]>= 0
#ifdef NO_SETSREALLOC
										&& list[j]< MAXSETS
#endif
									){
										highlight_set[highlight_set_len]= (int) list[j];
										highlight_set_len+= 1;
									}
								}
							}
							else{
								argerror( "can't reallocate highlight_set buffer", argv[idx] );
							}
						}
						else{
							highlight_set_len= -1;
						}
#ifndef __GNUC__
						xfree( list );
#endif
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-mark_set", -1)== 0){
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
					  int i,j;
#ifdef NO_SETSREALLOC
					  int n= MAXSETS;
#else
					  int n= MAXSETS* 4;
#endif
#ifdef __GNUC__
					  double list[n];
#else
					  double *list= (double*) calloc( n, sizeof(double));
						if( !list ){
							argerror( "no memory for -mark_set buffer", argv[idx] );
						}
#endif
						i= n;
// 						fascanf( &i, argv[idx+1], list, NULL, NULL, NULL, NULL );
						fascanf2( &i, argv[idx+1], list, ',' );
						if( i> 0 ){
							if( (mark_set= (int*) realloc( (char*) mark_set, (i+1)* sizeof(int) )) ){
								for( j= 0, mark_set_len= 0; j< i; j++ ){
									if( list[j]>= 0
#ifdef NO_SETSREALLOC
										&& list[j]< MAXSETS
#endif
									){
										mark_set[mark_set_len]= (int) list[j];
										mark_set_len+= 1;
									}
								}
								mark_sets= i+1;
							}
							else{
								argerror( "can't reallocate mark_set buffer", argv[idx] );
								mark_sets= 0;
							}
						}
						else{
							mark_set_len= -1;
						}
#ifndef __GNUC__
						xfree( list );
#endif
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-plot_only_set", -1)== 0){
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
					  int i,j;
#ifdef NO_SETSREALLOC
					  int n= MAXSETS;
#else
					  int n= MAXSETS* 4;
#endif
#ifdef __GNUC__
					  double list[n];
#else
					  double *list= (double*) calloc( n, sizeof(double));
						if( !list ){
							argerror( "no memory for -plot_only_set buffer", argv[idx] );
						}
#endif
	/* 					sscanf(argv[idx+1], "%d", &plot_only_set);	*/
						i= n;
// 						fascanf( &i, argv[idx+1], list, NULL, NULL, NULL, NULL );
						fascanf2( &i, argv[idx+1], list, ',' );
						if( i> 0 ){
							if( (plot_only_set= (int*) realloc( (char*) plot_only_set, (i+1)* sizeof(int) )) ){
								for( j= 0, plot_only_set_len= 0; j< i; j++ ){
									  /* If only always invalid set-numbers are given, the list will exist,
									   \ but be empty. This means that no sets will be drawn.
									   */
									if( list[j]>= 0
#ifdef NO_SETSREALLOC
										&& list[j]< MAXSETS
#endif
									){
										plot_only_set[plot_only_set_len]= (int) list[j];
										plot_only_set_len+= 1;
									}
								}
							}
							else{
								argerror( "can't reallocate plot_only_set buffer", argv[idx] );
							}
						}
						else{
							plot_only_set_len= -1;
						}
#ifndef __GNUC__
						xfree( list );
#endif
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-plot_only_file", -1)== 0){
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						sscanf(argv[idx+1], "%d", &plot_only_file);
						if( plot_only_file< 0 ){
							plot_only_file= 0;
						}
						if( debugFlag)
							fprintf( StdErr, "plot_only_file=%d\n", plot_only_file );
						idx+= 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-nocomments", -1)== 0) {
					if( Opt01 ){
						NoComment= 1;
					}
					else{
						NoComment= 0;
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-notitle", -1)== 0) {
					if( Opt01 ){
						no_title= 1;
					}
					else{
						no_title= 0;
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-AllTitles", -1)== 0) {
					if( Opt01 ){
						AllTitles= 1;
					}
					else{
						AllTitles= 0;
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ColouredSetTitles", -1)== 0) {
				  extern int DrawColouredSetTitles;
					if( Opt01 ){
						DrawColouredSetTitles= 1;
					}
					else{
						DrawColouredSetTitles= 0;
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-colour", 5)== 0 ){
					MonoChrome= 0;
					idx++;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-nolegendbox", -1)== 0) {
					if( Opt01 ){
						no_legend_box= 1;
					}
					else{
						no_legend_box= 0;
					}
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-nolegend", -1)== 0) {
					if( Opt01 ){
						no_legend= 1;
					}
					else{
						no_legend= 0;
					}
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-nopens", -1)== 0) {
				  extern int no_pens;
					if( Opt01 ){
						no_pens= 1;
					}
					else{
						no_pens= 0;
					}
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-nointensity_legend", -1)== 0) {
					if( Opt01 ){
						no_intensity_legend= 1;
					}
					else{
						no_intensity_legend= 0;
					}
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-nolabels", -1)== 0) {
				  extern int no_ulabels;
					if( Opt01 ){
						no_ulabels= 1;
					}
					else{
						no_ulabels= 0;
					}
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-mindlegend", -1)== 0) {
					if( Opt01 ){
						legend_always_visible= 1;
					}
					else{
						legend_always_visible= 0;
					}
					idx+= 1;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-legendtype", -1)== 0) {
				  int x;
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						if( (x= atoi(argv[idx+1]))>= 0 && x<= 1){
							legend_type= x;
						}
						idx+= 2;
					}
				}
/* #ifdef __GNUC__	*/
				else if (Check_Option( strcasecmp,argv[idx], "-maxlbuf", -1)== 0) {
				  int x;
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						if( (x= atoi(argv[idx+1]))> LMAXBUFSIZE ){
							LMAXBUFSIZE= x;
							if( debugFlag){
								fprintf( StdErr, "maximum (data) linelength set to %d\n", LMAXBUFSIZE );
							}
						}
						idx+= 2;
					}
				}
/* #endif	*/
				else if (Check_Option( strcasecmp,argv[idx], "-maxsets", -1)== 0) {
				  int x;
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						if( (x= atoi(argv[idx+1]))> 0 ){
							if( x> MaxSets ){
								if( !realloc_sets( NULL, MaxSets, x, "ParseArgs") ){
									CleanUp();
									exit( -1 );
								}
							}
						}
						if( debugFlag)
							fprintf( StdErr, "MaxSets=%d\n", MaxSets);
						idx+= 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-separator", 4)== 0) {
				  char *sep;
					if (idx+1 >= argc){
						argerror("missing character", argv[idx]);
					}
					else{
						sep= argv[idx+1];
						if( sep[0]== '0' && (sep[1]== 'x' || sep[1]== 'X') ){
						  int x= ' ';
							if( sscanf( sep, "0x%x", &x ) && x> 0 && x< 256 ){
								data_separator= x;
							}
							else{
								fprintf( StdErr, "%s %s=0x%x: conversion or range error\n",
									argv[idx], sep, x
								);
							}
						}
						else{
							data_separator= sep[0];
						}
						if( debugFlag){
							fprintf( StdErr, "column separator='%c' (0x%x)\n", data_separator, data_separator );
						}
						idx+= 2;
					}
				}

					/* these have been handled already. Syntax is
					 * correct, else we would not have ended up here.
					 \ Caveat: new font options must be added here too!
					 */
				else if (Check_Option( strcasecmp,argv[idx], "-tf", -1) == 0) {
					  /* Title Font */
					if( idx+1 >= argc)
						argerror("missing font", argv[idx]);
					else{
						font_name= argv[idx+1];
						New_XGFont( 'TITL', font_name );
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-df", -1) == 0) {
					  /* Dialog Font */
					if( idx+1 >= argc)
						argerror("missing font", argv[idx]);
					else{
						font_name= argv[idx+1];
						if( New_XGFont( 'DIAL', font_name ) ){
						  LocalWindows *WL= WindowList;
						  LocalWin *wi;
						  int i;
						  xtb_frame *f;
							ReallocColours(False);
							  /* Update the fonts used in the dialogs. Note that this does not update
							   \ button widths!!
							   */
							if( HO_Dialog.win && HO_Dialog.framelist ){
								f= *(HO_Dialog.framelist);
								for( i= 0; i< HO_Dialog.frames; i++ ){
									if( f[i].chfnt ){
										(*f[i].chfnt)( f[i].win, dialogFont.font, dialog_greekFont.font);
									}
								}
							}
							if( SD_Dialog.win && SD_Dialog.framelist ){
								f= *(SD_Dialog.framelist);
								for( i= 0; i< SD_Dialog.frames; i++ ){
									if( f[i].chfnt ){
										(*f[i].chfnt)( f[i].win, dialogFont.font, dialog_greekFont.font);
									}
								}
							}
							ChangeCrossGC( &ACrossGC );
							ChangeCrossGC( &BCrossGC );
							while( WL ){
								wi= WL->wi;
/* 
								for( i= 0; i< N_YAv_SortTypes; i++ ){
									XDeleteContext( disp, wi->YAv_Sort_frame.framelist[i]->win, frame_context );
									xfree( wi->YAv_Sort_frame->framelist[i].description );
								}
								XDeleteContext( disp, wi->YAv_Sort_frame.win, frame_context );
								xfree( wi->YAv_Sort_frame.description);
								xtb_br_del(wi->YAv_Sort_frame.win);
								xtb_br_new( wi->window, N_YAv_SortTypes, YAv_SortTypes, 0,
									   YAv_SortFun, (xtb_data) wi->window, &wi->YAv_Sort_frame
								);
								xtb_describe( &wi->YAv_Sort_frame, "Select type of sorting for YAveraging\n"
										"(Shift-Mod1-^ or Shift-Mod1-7)\n"
								);
								{ int i;
									for( i= 0; i< N_YAv_SortTypes; i++ ){
										xtb_describe( wi->YAv_Sort_frame.framelist[i], YAv_Sort_Desc[i] );
										XSaveContext( disp, wi->YAv_Sort_frame.framelist[i]->win, frame_context,
											(caddr_t) &wi->YAv_Sort_frame
										);
									}
								}
								XSaveContext( disp, wi->YAv_Sort_frame.win, frame_context, (caddr_t) &wi->YAv_Sort_frame );
 */
								(*wi->YAv_Sort_frame.chfnt)( wi->YAv_Sort_frame.win, dialogFont.font, dialog_greekFont.font );
								WL= WL->next;
							}
						}
					}
					idx+= 2;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-af", -1) == 0) {
					if( idx+1 >= argc)
						argerror("missing font", argv[idx]);
					else{
						font_name= argv[idx+1];
						New_XGFont( 'AXIS', font_name );
						idx+= 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-mf", 3) == 0) {
					if( Opt01== 0 )
						use_markFont = 0;
					else
						use_markFont = 1;
					idx++;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-lf", -1) == 0 ||
					Check_Option( strcasecmp,argv[idx], "-lef", -1)== 0
				) {
					if( idx+1 >= argc)
						argerror("missing font", argv[idx]);
					else{
						font_name= argv[idx+1];
						New_XGFont( 'LEGN', font_name );
						idx+= 2;
					}
				}
				else if ( Check_Option( strcasecmp,argv[idx], "-laf", -1)== 0 ) {
					if( idx+1 >= argc)
						argerror("missing font", argv[idx]);
					else{
						font_name= argv[idx+1];
						New_XGFont( 'LABL', font_name );
						idx+= 2;
					}
				}

				else if( Check_Option( strcasecmp, argv[idx], "-maxHeight", -1)== 0 ){
				  int i;
				  double x;
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									ActiveWin->hard_devices[i].dev_max_height = x;
								}
								else{
									hard_devices[i].dev_max_height = x;
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-maxWidth", -1)== 0 ){
				  int i;
				  double x;
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									ActiveWin->hard_devices[i].dev_max_width = x;
								}
								else{
									hard_devices[i].dev_max_width = x;
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_tf", -1)== 0 ){
				  int i;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing fontname", argv[idx]);
					else{
						if( argv[idx+1][0] ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									strncpy(ActiveWin->hard_devices[i].dev_title_font, argv[idx+1],
										sizeof(ActiveWin->hard_devices[i].dev_title_font)-1 );
								}
								else{
									strncpy(hard_devices[i].dev_title_font, argv[idx+1],
										sizeof(hard_devices[i].dev_title_font)-1 );
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_tf_size", -1)== 0 ){
				  int i;
				  double x;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									ActiveWin->hard_devices[i].dev_title_size= x;
								}
								else{
									hard_devices[i].dev_title_size= x;
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_lef", -1)== 0 ){
				  int i;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing fontname", argv[idx]);
					else{
						if( argv[idx+1][0] ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									strncpy(ActiveWin->hard_devices[i].dev_legend_font, argv[idx+1],
										sizeof(ActiveWin->hard_devices[i].dev_legend_font)-1 );
								}
								else{
									strncpy(hard_devices[i].dev_legend_font, argv[idx+1],
										sizeof(hard_devices[i].dev_legend_font)-1 );
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_lef_size", -1)== 0 ){
				  int i;
				  double x;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									ActiveWin->hard_devices[i].dev_legend_size= x;
								}
								else{
									hard_devices[i].dev_legend_size= x;
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_laf", -1)== 0 ){
				  int i;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing fontname", argv[idx]);
					else{
						if( argv[idx+1][0] ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									strncpy(ActiveWin->hard_devices[i].dev_label_font, argv[idx+1],
										sizeof(ActiveWin->hard_devices[i].dev_label_font)-1 );
								}
								else{
									strncpy(hard_devices[i].dev_label_font, argv[idx+1],
										sizeof(hard_devices[i].dev_label_font)-1 );
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_laf_size", -1)== 0 ){
				  int i;
				  double x;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									ActiveWin->hard_devices[i].dev_label_size= x;
								}
								else{
									hard_devices[i].dev_label_size= x;
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_af", -1)== 0 ){
				  int i;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing fontname", argv[idx]);
					else{
						if( argv[idx+1][0] ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									strncpy(ActiveWin->hard_devices[i].dev_axis_font, argv[idx+1],
										sizeof(ActiveWin->hard_devices[i].dev_axis_font)-1 );
								}
								else{
									strncpy(hard_devices[i].dev_axis_font, argv[idx+1],
										sizeof(hard_devices[i].dev_axis_font)-1 );
								}
							}
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ps_af_size", -1)== 0 ){
				  int i;
				  double x;
					if (idx+1 >= argc || !argv[idx+1][0] )
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							for( i= 0; i< hard_count; i++ ){
								if( ActiveWin ){
									ActiveWin->hard_devices[i].dev_axis_size= x;
								}
								else{
									hard_devices[i].dev_axis_size= x;
								}
							}
						}
						idx+= 2;
					}
				}

				else if( Check_Option( strncmp, argv[idx], "-DPShadow", 9)== 0 ){
				  extern int DiscardedShadows;
					DiscardedShadows= Opt01;
					idx++;
				}
				else if( Check_Option( strncmp, argv[idx], "-Cross", 6)== 0 ){
				  extern int CursorCross, CursorCross_Labeled;
				  LocalWindows *WL= WindowList;
					CursorCross= Opt01;
					if( Opt01> 1 ){
						CursorCross_Labeled= Opt01-1;
					}
					else{
						CursorCross_Labeled= False;
					}
					while( WL ){
						SelectXInput( WL->wi );
						WL= WL->next;
					}
					idx++;
				}
				else if( Check_Option( strncmp, argv[idx], "-Cauto", 6)== 0 ){
					use_xye_info= Opt01;
					idx++;
				}
				else if( Check_Option( strncmp, argv[idx], "-Columns", 8)== 0 ){
#	define NCOLUMNS	5
				  int n= NCOLUMNS;
				  double ul[NCOLUMNS+1]= {3, 0, 1, 2, -1};
					if( idx+1 >= argc ||
// 						(fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL)< 0 && n< 4)
						(fascanf2( &n, argv[idx+1], ul, ',' )< 0 && n< 4)
					){
						argerror("missing co-ordinate(s)", argv[idx]);
					}
					else{
						CLIP_EXPR( NCols, (int) ul[0], 3, (int) ul[0]);
						CLIP_EXPR( xcol, (int) ul[1], 0, NCols-1 );
						CLIP_EXPR( ycol, (int) ul[2], 0, NCols-1 );
						CLIP_EXPR( ecol, (int) ul[3], 0, NCols-1 );
						CLIP_EXPR( lcol, (int) ul[3], -1, NCols-1 );
						if( n> 4 ){
							CLIP_EXPR( Ncol, (int) ul[4], ul[4], NCols- 1 );
						}
						MaxCols= MAX( MaxCols, NCols );
						if( NCols> MaxCols ){
							MaxCols= NCols;
							if( BinaryDump.data ){
								AllocBinaryFields( MaxCols, "ReadData()" );
							}
						}
						idx+= 2;
					}
#undef NCOLUMNS
				}
				else if( Check_Option( strncmp, argv[idx], "-C", 2)==0 ){
				  char *o= &argv[idx][2];
				  int O= 0, i;
					if( strlen(o)== 3){
						for( i= 0; i< 3; i++){
							O*= 10;
							switch( o[i]){
								case 'x':
									O+= 1;
									column[0]= i;
									break;
								case 'y':
									O+= 2;
									column[1]= i;
									break;
								case 'e':
									O+= 3;
									column[2]= i;
									break;
							}
						}
					}
					if( O== 123 || O== 132 || O== 213 || O== 231 || O== 321 || O== 312 ){
						if( debugFlag){
							fprintf( StdErr, "Input columns: x=%d y=%d e=%d\n",
								column[0], column[1], column[2]
							);
						}
						use_xye_info= 0;
					}
					else{
					  char msg[80];
						sprintf( msg, "%s (%d)", argv[idx], O);
						argerror( "invalid -C option", msg);
					}
					idx++;
				}
				else if( !Check_Option( strncasecmp, argv[idx], "-fli", 4) ){
					if( !strcasecmp( argv[idx], "-fli0") )
						newfile_incr_width= 0;
					else{
						sscanf( argv[idx], "-fli%lf", &newfile_incr_width );
					}
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-spax", 5)== 0 ||
					Check_Option( strncasecmp, argv[idx], "-spay", 5)== 0 ||
					Check_Option( strncasecmp, argv[idx], "-spa", 4)== 0
				){
					if( index( "xX", argv[idx][4]) ){
						scale_plot_area_x= Opt01;
					}
					else if( index( "yY", argv[idx][4] ) ){
						scale_plot_area_y= Opt01;
					}
					else{
						scale_plot_area_x= scale_plot_area_y= Opt01;
					}
					idx++;
				}
				else if( !Check_Option( strncasecmp, argv[idx], "-lb", 3) ){
					if( ActiveWin && ActiveWin!= &StubWindow ){
						ActiveWin->labels_in_legend= Opt01;
					}
					else if( !StartUp ){
						labels_in_legend= Opt01;
					}
					else if( !strcasecmp( argv[idx], "-lb0") )
						labels_in_legend= -1;
					else
						labels_in_legend= 1;
					idx++;
				}
				else if( !Check_Option( strncasecmp, argv[idx], "-fn", 3) ){
					if( ActiveWin && ActiveWin!= &StubWindow ){
						ActiveWin->filename_in_legend= Opt01;
					}
					else if( !StartUp ){
						filename_in_legend= Opt01;
					}
					else if( !strcasecmp( argv[idx], "-fn0") )
						filename_in_legend= -1;
					else
						filename_in_legend= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-Landscape", 4)== 0){
					Print_Orientation= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-Portrait", 4)== 0){
					Print_Orientation= 0;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpColours", -1)== 0){
					XGStoreColours= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpAverage", 12)== 0){
					dump_average_values= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpProcessed", 14)== 0){
					DumpProcessed= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpDHex", 9)== 0){
					DumpDHex= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpPens", 9)== 0){
					DumpPens= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpAsAscanf", 13)== 0){
				  extern int DumpAsAscanf;
					DumpAsAscanf= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpBinary", 11)== 0){
					DumpBinary= Opt01;
					if( strcasecmp( argv[idx], "-DumpBinary" ) == 0 ){
					  // 20101028: this would result in Opt01==1 and hence a dump using bytes ...
					  // not what we'd hope for as default behaviour!!
						BinarySize = sizeof(double);
						DumpBinary = True;
					}
					else if( DumpBinary== sizeof(float) || DumpBinary== sizeof(unsigned char) || DumpBinary== sizeof(unsigned short) ){
						BinarySize= DumpBinary;
						DumpBinary= True;
					}
					else{
						BinarySize= sizeof(double);
						DumpBinary= True;
					}
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpRead", 9)== 0){
					DumpFile= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpIncluded", 13)== 0){
					DumpIncluded= Opt01;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-DumpKeyEVAL", 12)== 0 ||
					Check_Option( strncasecmp, argv[idx], "-DumpKeyParam", 13)== 0
				){
				  extern int DumpKeyParams;
					DumpKeyParams= Opt01;
					idx++;
				}
				else if( !strcasecmp( argv[idx], "-bf") ){
				  extern char *BoxFilter_File;
					if (idx+1 >= argc){
						argerror( "missing BoxFilter file name", argv[idx]);
					}
					else{
						if( !BoxFilter_File ){
							BoxFilter_File= calloc(MAXPATHLEN, sizeof(char));
						}
						if( BoxFilter_File ){
							strncpy( BoxFilter_File, argv[idx+1], MAXPATHLEN-1 );
						}
						else{
							fprintf( StdErr, "Warning: ignoring BoxFilter filename prespecification \"%s\" (%s)\n",
								argv[idx+1], serror()
							);
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-BoxFilterUndo", -1)== 0 ){
				  extern int BoxFilter_Undo_Enable;
					BoxFilter_Undo_Enable= (Opt01)? True : False;
					idx+= 1;
				}
				else if( !strcasecmp( argv[idx], "-pf") ){
				  extern char *substitute();
					if (idx+1 >= argc)
						argerror( "missing PrintFile name", argv[idx]);
					else{
/* 						xfree( PrintFileName );	*/
						  /* 20050211: since we cannot currently modify an existing PrintFileName at runtime,
						   \ at least provide a means to override one specified in a file by the command line!
						   */
						if( !PrintFileName ){
							PrintFileName= substitute( XGstrdup( argv[idx+1] ), ' ', '_' );
						}
						idx += 2;
					}
				}
				else if( !strcasecmp( argv[idx], "-dir") ){
				  char exp[MAXCHBUF*2];
					if (idx+1 >= argc)
						argerror( "missing directory name", argv[idx]);
					else{
						if( chdir( tildeExpand( exp, argv[idx+1]) ) ){
							fprintf( StdErr, "Can't chdir to %s (%s)\n", argv[idx+1], serror() );
						}
						idx += 2;
					}
				}
				else if( !strcasecmp(argv[idx], "-x") || !Check_Option( strncasecmp,argv[idx], "-x ", 3) ) {
					/* Units for X axis */
					if( argv[idx][2]== ' ' && argv[idx][3]!= '\0' ){
						strcpy( (ActiveWin && ActiveWin!= &StubWindow)? ActiveWin->XUnits : XUnits, &argv[idx][3]);
						idx += 1;
					}
					else{
						if (idx+1 >= argc)
						  argerror("missing axis name", argv[idx]);
						else{
							(void) strcpy((ActiveWin && ActiveWin!= &StubWindow)? ActiveWin->XUnits : XUnits, argv[idx+1]);
							idx += 2;
						}
					}
					XUnitsSet= 1;
				}
				else if( !strcasecmp(argv[idx], "-y") || !Check_Option( strncasecmp,argv[idx], "-y ", 3) ) {
					/* Units for Y axis */
					if( argv[idx][2]== ' ' && argv[idx][3]!= '\0' ){
						strcpy( (ActiveWin && ActiveWin!= &StubWindow)? ActiveWin->YUnits : YUnits, &argv[idx][3]);
						idx += 1;
					}
					else{
						if (idx+1 >= argc)
						  argerror("missing axis name", argv[idx]);
						else{
							(void) strcpy((ActiveWin && ActiveWin!= &StubWindow)? ActiveWin->YUnits : YUnits, argv[idx+1]);
							idx += 2;
						}
					}
					YUnitsSet= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-zl", 3) == 0) {
					if( ActiveWin && ActiveWin!= &StubWindow ){
						ActiveWin->zeroFlag= Opt01;
					}
					else if( !StartUp ){
						zeroFlag= Opt01;
					}
					else if( Opt01== 0 )
						zeroFlag = -1;
					else
						zeroFlag = 2;
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-tcl", 4) == 0) {
				  extern int Determine_tr_curve_len;
					Determine_tr_curve_len= Opt01;
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-tas", 4) == 0) {
				  extern int Determine_AvSlope;
					Determine_AvSlope= Opt01;
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-tk", 3) == 0) {
					/* Draw tick marks instead of full grid */
					if( ActiveWin && ActiveWin!= &StubWindow ){
						ActiveWin->htickFlag= Opt01;
						ActiveWin->vtickFlag= Opt01;
					}
					else if( !StartUp ){
						htickFlag= Opt01;
						vtickFlag= Opt01;
					}
					else if( Opt01== 0 ){
						htickFlag = -1;
						vtickFlag = -1;
					}
					else{
						htickFlag = (Opt01< 0)? 2 : 1;
						vtickFlag = (Opt01< 0)? 2 : 1;
					}
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-htk", 3) == 0) {
				  /* Draw tick marks instead of horizontal gridlines */
					if( ActiveWin && ActiveWin!= &StubWindow ){
						ActiveWin->htickFlag= Opt01;
					}
					else if( !StartUp ){
						htickFlag= Opt01;
					}
					else if( Opt01== 0 )
						htickFlag = -1;
					else
						htickFlag = 2;
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-vtk", 3) == 0) {
				  /* Draw tick marks instead of vertical gridlines */
					if( ActiveWin && ActiveWin!= &StubWindow ){
						ActiveWin->vtickFlag= Opt01;
					}
					else if( !StartUp ){
						vtickFlag= Opt01;
					}
					else if( Opt01== 0 )
						vtickFlag = -1;
					else
						vtickFlag = 2;
					idx++;
				}
				else if( !strcmp(argv[idx], "-T") || !Check_Option( strncmp,argv[idx], "-T ", 3) ) {
				  char *tit;
					  /* additional Title of plot */
					if( argv[idx][2]== ' ' && argv[idx][3]!= '\0' ){
						tit= &argv[idx][3];
						idx += 1;
					}
					else{
						if( idx+1 >= argc){
						  argerror("missing plot title", argv[idx]);
						}
						else{
							tit= argv[idx+1];
							idx += 2;
						}
					}
					xfree( titleText2 );
					titleText2= XGstrdup( tit );
				}
				else if( !strcmp(argv[idx], "-t") || !Check_Option( strncmp,argv[idx], "-t ", 3) ) {
					/* Title of plot */
					if( argv[idx][2]== ' ' && argv[idx][3]!= '\0' ){
						strcpy( titleText, &argv[idx][3]);
						idx += 1;
					}
					else{
						if( idx+1 >= argc)
						  argerror("missing plot title", argv[idx]);
						else{
							(void) strcpy(titleText, argv[idx+1]);
							idx += 2;
						}
					}
					titleTextSet= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-IgnoreSetCName", 15) == 0 ||
					Check_Option( strncmp, argv[idx], "-IgnoreCNames", 13) == 0
				){
					XGIgnoreCNames= Opt01;
					idx++;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-fg", -1) == 0) {
				  /* Foreground color */
					if (idx+1 >= argc)
						argerror("missing color", argv[idx]);
					else{
						if( GetColorDefault(argv[idx+1], &tempPixel, "Foreground", "black", NULL ) ){
							FreeColor( &normPixel, &normCName );
							normPixel = tempPixel;
							StoreCName( normCName );
							ReallocColours( True );
							if( MonoChrome ){
								for( set = 0;  set < MAXATTR;  set++) {
									AllAttrs[set].pixelValue= normPixel;
									xfree( AllAttrs[set].pixelCName );
									AllAttrs[set].pixelCName= XGstrdup( normCName );
								}
							}
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-bg", -1) == 0) {
					/* Background color */
					if (idx+1 >= argc)
						argerror("missing color", argv[idx]);
					else{
						if( GetColorDefault(argv[idx+1], &tempPixel, "Background", "LightGray", NULL ) ){
							if( bgPixel!= tempPixel){
							  LocalWindows *WL= WindowList;
								FreeColor( &bgPixel, &bgCName );
								bgPixel = tempPixel;
								StoreCName( bgCName );
								ReallocColours( True );
								if( disp ){
	/* 								RecolourCursors();	*/
									ChangeCrossGC( &ACrossGC );
									ChangeCrossGC( &BCrossGC );
									while( WL ){
									  LocalWin *lwi= WL->wi;
										XSetWindowBackground( disp, lwi->window, bgPixel );
										WL= WL->next;
									}
								}
							}
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-bd", -1) == 0) {
					/* Border color */
					if (idx+1 >= argc)
						argerror("missing color", argv[idx]);
					else{
						if( GetColorDefault(argv[idx+1], &tempPixel, "Border", "black", NULL ) ){
							FreeColor( &bdrPixel, &bdrCName );
							bdrPixel = tempPixel;
							StoreCName( bdrCName );
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-bw", -1) == 0) {
					/* Border width */
					if (idx+1 >= argc)
						argerror("missing border size", argv[idx]);
					else{
						bdrSize = atoi(argv[idx+1]);
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-hc", -1) == 0) {
				  extern Pixel xtb_light_pix;
					/* Highlight grid color */
					if (idx+1 >= argc)
						argerror("missing color", argv[idx]);
					else{
						if( GetColorDefault(argv[idx+1], &tempPixel, "HighlightColor", NULL, &xtb_light_pix) ){
							FreeColor( &highlightPixel, &highlightCName );
							highlightPixel = tempPixel;
							StoreCName( highlightCName );
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-hl_pars", -1) == 0) {
				  double ul[5]= {0,0};
				  int i, n= highlight_npars;
					if (idx+1 >= argc)
						argerror( "missing parameter(s)", argv[idx]);
					else{
// 						if( fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL)> 0 )
						if( fascanf2( &n, argv[idx+1], ul, ',' )> 0 )
						{
							for( i= 0; i< n; i++ ){
								highlight_par[i]= ul[i];
							}
						}
						idx+= 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-hl_mode", -1) == 0) {
				  double ul[1]= {0};
				  int n= 1;
					if (idx+1 >= argc)
						argerror( "missing mode selector", argv[idx]);
					else{
// 						if( fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL)> 0 )
						if( fascanf2( &n, argv[idx+1], ul, ',' )> 0 )
						{
							highlight_mode= (int)ul[0];
						}
						idx+= 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-hl_too", 7) == 0) {
					Conditional_Toggle(AlwaysDrawHighlighted);
					idx+= 1;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-zw", -1) == 0) {
				  extern double zeroWidth;
					/* Set the zero width */
					if (idx+1 >= argc)
						argerror("missing line width", argv[idx]);
					else{
						sscanf( argv[idx+1], "%lf", &zeroWidth );
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-zg", -1) == 0) {
					/* Zero grid color */
					if (idx+1 >= argc)
						argerror("missing color", argv[idx]);
					else{
						if( GetColorDefault(argv[idx+1], &tempPixel, "ZeroColor", "Red", NULL ) ){
							FreeColor( &zeroPixel, &zeroCName );
							zeroPixel = tempPixel;
							StoreCName( zeroCName );
						}
						if( disp ){
							ChangeCrossGC( &ACrossGC );
							ChangeCrossGC( &BCrossGC );
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-ag", -1) == 0) {
					if (idx+1 >= argc)
						argerror("missing color", argv[idx]);
					else{
						if( GetColorDefault(argv[idx+1], &tempPixel, "AxisColor", "Black", NULL) ){
							FreeColor( &axisPixel, &axisCName );
							axisPixel = tempPixel;
							StoreCName( axisCName );
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-gg", -1) == 0) {
					  /* Gray color */
					if (idx+1 >= argc)
						argerror("missing color", argv[idx]);
					else{
						if( GetColorDefault(argv[idx+1], &tempPixel, "GridColor", "Gray50", NULL) ){
							FreeColor( &gridPixel, &gridCName );
							gridPixel = tempPixel;
							StoreCName( gridCName );
						}
						if( disp ){
							ChangeCrossGC( &ACrossGC );
							ChangeCrossGC( &BCrossGC );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp,argv[idx], "-bs", -1) == 0 ){
				  extern int BackingStore, BackingStore_argset;
				  LocalWindows *WL;
				  XSetWindowAttributes wattr, gattr;
				  unsigned long wamask;
				  extern LocalWin *InitWindow;
					BackingStore= (Opt01)? True : False;
					BackingStore_argset= True;
					WL= WindowList;
					if( BackingStore ){
						wattr.backing_store= Always;
						wattr.save_under= True;
					}
					else{
						wattr.backing_store= NotUseful;
						wattr.save_under= False;
					}
					wamask= CWBackingStore|CWSaveUnder;
					if( debugFlag ){
						fprintf( StdErr, "Backing store turned %s; updating all open windows\n",
							(BackingStore)? "on" : "off"
						);
					}
					while( WL ){
						if( WL->wi!= InitWindow ){
							if( debugFlag ){
								XGetWindowAttributes( disp, WL->wi->window, &gattr );
								fprintf( StdErr, "Window 0x%lx backing_store was %d; save_under was %d\n",
									WL->wi->window, gattr.backing_store, gattr.save_under );
							}
							XChangeWindowAttributes( disp, WL->wi->window, wamask, &wattr );
						}
						WL= WL->next;
					}
					idx+= 1;
				}
				else if( Check_Option( strcasecmp,argv[idx], "-rv", -1) == 0 ||
						Check_Option( strcasecmp,argv[idx], "-reverse", -1) == 0
				) {
					/* Reverse video option */
					reverseFlag= !reverseFlag;
					ReallocColours(True);
/* 
					ReversePixel(&bgPixel);
					ReversePixel(&bdrPixel);
					ReversePixel(&zeroPixel);
					ReversePixel(&axisPixel);
					ReversePixel(&gridPixel);
					ReversePixel(&normPixel);
					ReversePixel(&echoPix);
					ReversePixel(&highlightPixel);
					for (set = 0;  set < MAXATTR;  set++) {
						ReversePixel(&(AllAttrs[set].pixelValue));
					}
 */
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-eng", 4) == 0) {
					/* Draw tick marks instead of full grid */
					if( Opt01== 0 )
						noExpY= noExpX= 1;
					else
						noExpY= noExpX= 0;
					idx++;
				}
				else if (Check_Option( strncmp,argv[idx], "-bias", 5) == 0) {
				  double bias;
					if (idx+1 >= argc){
						argerror("missing value", argv[idx]);
					}
					else{
						sscanf(argv[idx+1], "%lf", &bias );
						switch( argv[idx][5] ){
							case 'X':
								Xbias_thres= bias;
								break;
							case 'Y':
								Ybias_thres= bias;
								break;
						}
						idx+= 2;
					}
				}
				else if (Check_Option( strncmp,argv[idx], "-engX", 5) == 0) {
					/* Draw tick marks instead of full grid */
					if( Opt01== 0 )
						noExpX= 1;
					else
						noExpX= 0;
					idx++;
				}
				else if (Check_Option( strncmp,argv[idx], "-engY", 5) == 0) {
					/* Draw tick marks instead of full grid */
					if( Opt01== 0 )
						noExpY= 1;
					else
						noExpY= 0;
					idx++;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-noaxis", 7) == 0) {
					axisFlag= 0;
					idx++;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-mlx", -1) == 0) {
					  /* Limit the X co-ordinates */
					if (idx+1 >= argc){
						argerror("missing co-ordinate(s)", argv[idx]);
					}
					else{
						sscanf(argv[idx+1], "%lf,%lf", &MusrLX, &MusrRX);
						use_max_x= 1;
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-mly", -1) == 0) {
					  /* Limit the Y co-ordinates */
					if (idx+1 >= argc){
						argerror("missing co-ordinate(s)", argv[idx]);
					}
					else{
						sscanf(argv[idx+1], "%lf,%lf", &MusrLY, &MusrRY);
						use_max_y= 1;
						idx += 2;
					}
				}
				else if( Check_Option( strncasecmp,argv[idx], "-exact_X", 8) == 0) {
					Conditional_Toggle( exact_X_axis );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-exact_Y", 8) == 0) {
					Conditional_Toggle( exact_Y_axis );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ValCat_X_grid", 14) == 0) {
					Conditional_Toggle( ValCat_X_grid );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ValCat_X_levels", 16) == 0) {
					if (idx+1 >= argc){
						argerror("missing co-ordinate(s)", argv[idx]);
					}
					else{
						ValCat_X_levels= atoi(argv[idx+1]);
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-ValCat_xstep", -1)== 0){
				  double x;
				  int n= 1;
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
// 						fascanf( &n, argv[idx+1], &x, NULL, NULL, NULL, NULL );
						fascanf2( &n, argv[idx+1], &x, ',' );
						if( n== 1 ){
							ValCat_X_incr= x;
						}
						if( debugFlag){
							fprintf( StdErr, "ValCat_xstep=%g\n", ValCat_X_incr);
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-ValCat_ystep", -1)== 0){
				  double x;
				  int n= 1;
					if (idx+1 >= argc){
						argerror("missing number", argv[idx]);
					}
					else{
// 						fascanf( &n, argv[idx+1], &x, NULL, NULL, NULL, NULL );
						fascanf2( &n, argv[idx+1], &x, ',' );
						if( n== 1 ){
							ValCat_Y_incr= x;
						}
						if( debugFlag){
							fprintf( StdErr, "ValCat_ystep=%g\n", ValCat_Y_incr);
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ValCat_X", -1) == 0) {
					Conditional_Toggle( ValCat_X_axis );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ValCat_Y", -1) == 0) {
					Conditional_Toggle( ValCat_Y_axis );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-all_ValCat_I", 8) == 0) {
					Conditional_Toggle( show_all_ValCat_I );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ValCat_I", -1) == 0) {
					Conditional_Toggle( ValCat_I_axis );
					idx+= 1;
				}
				else if( Check_Option( strcasecmp,argv[idx], "-lx", -1) == 0) {
				  int n= 2;
				  double ul[2]= {0,0};
					/* Limit the X co-ordinates */
					if (idx+1 >= argc)
						argerror("missing co-ordinate(s)", argv[idx]);
					else{
// 						fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						fascanf2( &n, argv[idx+1], ul, ',' );
						startup_wi.R_UsrOrgX= ul[0];
						startup_wi.R_UsrOppX= ul[1];
						if( startup_wi.R_UsrOrgX > startup_wi.R_UsrOppX ){
						  double c= startup_wi.R_UsrOppX;
							startup_wi.R_UsrOppX= startup_wi.R_UsrOrgX;
							startup_wi.R_UsrOrgX= c;
						}
						use_lx= 1;
						if( Check_Option( strcmp, argv[idx], "-LX" , -1)== 0 ){
							User_Coordinates= 1;
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-ly", -1) == 0) {
				  int n= 2;
				  double ul[2]= {0,0};
					/* Limit the Y co-ordinates */
					if (idx+1 >= argc)
						argerror("missing co-ordinate(s)", argv[idx]);
					else{
// 						fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						fascanf2( &n, argv[idx+1], ul, ',' );
						startup_wi.R_UsrOrgY= ul[0];
						startup_wi.R_UsrOppY= ul[1];
						if( startup_wi.R_UsrOrgY > startup_wi.R_UsrOppY ){
						  double c= startup_wi.R_UsrOppY;
							startup_wi.R_UsrOppY= startup_wi.R_UsrOrgY;
							startup_wi.R_UsrOrgY= c;
						}
						use_ly= 1;
						if( Check_Option( strcmp, argv[idx], "-LY" , -1)== 0 ){
							User_Coordinates= 1;
						}
						idx += 2;
					}
				}
				else if( Check_Option( strcasecmp,argv[idx], "-bbox", -1) == 0
					|| Check_Option( strcasecmp,argv[idx], "-boundingbox", -1) == 0
				){
				  int n= 4;
				  double ul[4]= {0,0,0,0};
					/* Limit the lower-left and upper-right corners */
					if (idx+1 >= argc)
						argerror("missing bounding-box co-ordinates", argv[idx]);
					else{
// 						fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						fascanf2( &n, argv[idx+1], ul, ',' );
						if( n== 4 ){
							startup_wi.R_UsrOrgX= ul[0];
							startup_wi.R_UsrOrgY= ul[1];
							startup_wi.R_UsrOppX= ul[2];
							startup_wi.R_UsrOppY= ul[3];
							use_lx= use_ly= 1;
							if( Check_Option( strcmp, argv[idx], "-BBox" , -1)== 0
								|| Check_Option( strcmp,argv[idx], "-BoundingBox", -1) == 0
							){
								User_Coordinates= 1;
							}
						}
						else{
							argerror( "need 4 bounding-box do-ordinates!", argv[idx] );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strcasecmp,argv[idx], "-lleft", -1) == 0) {
				  int n= 2;
				  double ul[2]= {0,0};
					/* Limit the lower-left */
					if (idx+1 >= argc)
						argerror("missing co-ordinate(s)", argv[idx]);
					else{
// 						fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						fascanf2( &n, argv[idx+1], ul, ',' );
						if( n== 2 ){
							startup_wi.R_UsrOrgX= ul[0];
							startup_wi.R_UsrOrgY= ul[1];
							use_lx= use_ly= 1;
							if( Check_Option( strcmp, argv[idx], "-LLeft" , -1)== 0 ){
								User_Coordinates= 1;
							}
						}
						else{
							argerror( "need 2 lower-left co-ordinates" , argv[idx] );
						}
						idx += 2;
					}
				}
				else if( Check_Option( strcasecmp,argv[idx], "-uright", -1) == 0) {
				  int n= 2;
				  double ul[2]= {0,0};
					/* Limit the upper-right */
					if (idx+1 >= argc)
						argerror("missing co-ordinate(s)", argv[idx]);
					else{
// 						fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						fascanf2( &n, argv[idx+1], ul, ',' );
						if( n == 2 ){
							startup_wi.R_UsrOppX= ul[0];
							startup_wi.R_UsrOppY= ul[1];
							if( startup_wi.R_UsrOrgX > startup_wi.R_UsrOppX ){
							  double c= startup_wi.R_UsrOppX;
								startup_wi.R_UsrOppX= startup_wi.R_UsrOrgX;
								startup_wi.R_UsrOrgX= c;
							}
							if( startup_wi.R_UsrOrgY > startup_wi.R_UsrOppY ){
							  double c= startup_wi.R_UsrOppY;
								startup_wi.R_UsrOppY= startup_wi.R_UsrOrgY;
								startup_wi.R_UsrOrgY= c;
							}
							use_lx= use_ly= 1;
							if( Check_Option( strcmp, argv[idx], "-URight" , -1)== 0 ){
								User_Coordinates= 1;
							}
						}
						else{
							argerror( "need 2 upper-right co-ordinates" , argv[idx] );
						}
						idx += 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-bb", 3) == 0) {
					  /* Draw bounding box around graph region */
					if( ActiveWin && ActiveWin!= &StubWindow ){
						 /* 20040921: update bbFlag global, to ensure that CopyFlags doesn't undo the setting of AW->bbFlag! */
						bbFlag= ActiveWin->bbFlag= (Opt01)? True : False;
					}
					else if( !StartUp ){
						bbFlag= Opt01;
					}
					else if( Opt01== 0)
						bbFlag = -1;
					else
						bbFlag = (Opt01< 0)? 2 : 1;
					idx++;
				}
				else if( Check_Option( strcasecmp,argv[idx], "-nbb_lx", -1) == 0) {
				  int n= 2;
				  double ul[2]= {0,0};
					if (idx+1 >= argc)
						argerror("missing co-ordinate(s)", argv[idx]);
					else{
// 						fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						fascanf2( &n, argv[idx+1], ul, ',' );
						startup_wi.win_geo.nobb_loX= ul[0];
						startup_wi.win_geo.nobb_hiX= ul[1];
						startup_wi.win_geo.nobb_range_X= 1;
						if( startup_wi.win_geo.nobb_loX > startup_wi.win_geo.nobb_hiX ){
							SWAP( startup_wi.win_geo.nobb_loX, startup_wi.win_geo.nobb_hiX, double );
						}
						if( Check_Option( strcmp, argv[idx], "-nbb_LX" , -1)== 0 ){
							User_Coordinates= 1;
						}
						idx += 2;
					}
				}
				else if( Check_Option( strcasecmp,argv[idx], "-nbb_ly", -1) == 0) {
				  int n= 2;
				  double ul[2]= {0,0};
					if (idx+1 >= argc)
						argerror("missing co-ordinate(s)", argv[idx]);
					else{
// 						fascanf( &n, argv[idx+1], ul, NULL, NULL, NULL, NULL);
						fascanf2( &n, argv[idx+1], ul, ',' );
						startup_wi.win_geo.nobb_loY= ul[0];
						startup_wi.win_geo.nobb_hiY= ul[1];
						startup_wi.win_geo.nobb_range_Y= 1;
						if( startup_wi.win_geo.nobb_loY > startup_wi.win_geo.nobb_hiY ){
							SWAP( startup_wi.win_geo.nobb_loY, startup_wi.win_geo.nobb_hiY, double );
						}
						if( Check_Option( strcmp, argv[idx], "-nbb_LY" , -1)== 0 ){
							User_Coordinates= 1;
						}
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-lw", -1) == 0) {
					/* Set the line width */
					if (idx+1 >= argc)
						argerror("missing line width", argv[idx]);
					else{
						sscanf( argv[idx+1], "%lf", &lineWidth );
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-aw", -1) == 0) {
					/* Set the axis width */
					if (idx+1 >= argc)
						argerror("missing line width", argv[idx]);
					else{
						sscanf( argv[idx+1], "%lf", &axisWidth );
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-gw", -1) == 0) {
					/* Set the grid width */
					if (idx+1 >= argc)
						argerror("missing line width", argv[idx]);
					else{
						sscanf( argv[idx+1], "%lf", &gridWidth );
						idx += 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-gp", 3) == 0) {
				  /* Set the grid style */
					if( Opt01 ){
						if (idx+1 >= argc)
							argerror("missing line style PATTERN", argv[idx]);
						else{
							gridLSLen= xtb_ProcessStyle( argv[idx+1], gridLS, MAXLS);
							idx++;
						}
					}
					else{
						gridLS[0]= 1;
						gridLSLen= 0;
					}
					idx += 1;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-ew", -1) == 0) {
					/* Set the errorbar width */
					if (idx+1 >= argc)
						argerror("missing line width", argv[idx]);
					else{
						sscanf( argv[idx+1], "%lf", &errorWidth );
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-nl", -1) == 0) {
					noLines = (Opt01< 0)? 2 : 1;
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-ResetAttr", 8) == 0) {
					ResetAttrs = 1;
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-SameAttr", 7) == 0) {
					SameAttrs = 1;
					idx++;
				}
				else if (Check_Option( strncmp,argv[idx], "-m", 2) == 0) {
					/* 
					 * Mark each point with an individual marker 
					 * varying with linestyle
					 */
					if( Opt01 ){
						markFlag = (Opt01< 0)? 2 : 1;  pixelMarks = 0;
					}
					else{
						markFlag= 0;
					}
					idx++;
				}
				else if (Check_Option( strcmp,argv[idx], "-M", -1) == 0) {
					/*
					 * Mark each point with an individual marker
					 * varying with color.
					 */
					if( Opt01 ){
						markFlag = (Opt01< 0)? 2 : 1; pixelMarks = 0;
					}
					else{
						markFlag= 0;
					}
					idx++;
				}
				else if (Check_Option( strcmp,argv[idx], "-p", -1) == 0) {
					/* Draw small pixel sized markers */
					if( Opt01 ){
						noLines =  pixelMarks = 1;
						markFlag= (Opt01< 0)? 2 : 1;
					}
					else{
						markFlag= 0;
					}
					idx++;
				}
				else if (Check_Option( strcmp,argv[idx], "-P", -1) == 0) {
					/* Draw large pixel sized markers */
					if( Opt01 ){
						markFlag = (Opt01< 0)? 2 : 1; pixelMarks = 2;
					}
					else{
						markFlag= 0;
					}
					idx++;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ln", 3) == 0 && (argv[idx][3]== 'x' || argv[idx][3]== 'y') ) {
					if (Check_Option( strncasecmp,argv[idx], "-lnxy", 5) == 0) {
						  /* The axes are logarithmic */
						if( Opt01== 0)
							logYFlag= logXFlag = -1;
						else
							logYFlag= logXFlag = 1+ atoi(&argv[idx][5]);
					}
					else if (Check_Option( strncasecmp,argv[idx], "-lnx", 4) == 0) {
						  /* The X axis is logarithmic */
						if( Opt01== 0)
							logXFlag = -1;
						else
							logXFlag = 1+ atoi(&argv[idx][4]);
					}
					else if (Check_Option( strncasecmp,argv[idx], "-lny", 4) == 0 ) {
						  /* The Y axis is logarithmic */
						if( Opt01== 0)
							logYFlag = -1;
						else
							logYFlag = 1+ atoi(&argv[idx][4]);
					}
/* 
					if( sqrtXFlag> 0 && logXFlag> 0)
						sqrtXFlag= -1;
					if( sqrtYFlag> 0 && logYFlag> 0)
						sqrtYFlag= -1;
 */
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-log_zero_x_min", 15)== 0 ){
					log_zero_x_mFlag= -1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-log_zero_x_max", 15)== 0 ){
					log_zero_x_mFlag= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-log_zero_y_min", 15)== 0 ){
					log_zero_y_mFlag= -1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-log_zero_y_max", 15)== 0 ){
					log_zero_y_mFlag= 1;
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-log_zero_", 10)== 0 ){
					if( idx+ 1 >= argc)
						argerror( "missing value", argv[idx]);
					else{
						if( argv[idx][10]== 'x' ){
							if( argv[idx][11]== 'y'){
								if( sscanf( argv[idx+1], "%lf,%lf", &log_zero_x, &log_zero_y)== 1){
									log_zero_y= log_zero_x;
								}
								log_zero_x_mFlag= 0;
								log_zero_y_mFlag= 0;
							}
							else{
								sscanf( argv[idx+1], "%lf", &log_zero_x);
								log_zero_x_mFlag= 0;
							}
						}
						else if( argv[idx][10]== 'y' ){
							sscanf( argv[idx+1], "%lf", &log_zero_y);
							log_zero_y_mFlag= 0;
						}
						else if( !strcasecmp( argv[idx], "-log_zero_sym_x") ){
							lz_sym_x= 1;
							strcpy( log_zero_sym_x, argv[idx+1]);
						}
						else if( !strcasecmp( argv[idx], "-log_zero_sym_y") ){
							lz_sym_y= 1;
							strcpy( log_zero_sym_y, argv[idx+1]);
						}
						else
							argerror( "invalid -log_zero_ option", &argv[idx][10]);
						if( log_zero_x< 0)
							log_zero_x= 0.0;
						if( log_zero_y< 0)
							log_zero_y= 0.0;
						if( debugFlag){
							fprintf( StdErr, "log_zero_x= %g %s\n", log_zero_x, (lz_sym_x)? log_zero_sym_x : "" );
							fprintf( StdErr, "log_zero_y= %g %s\n", log_zero_y, (lz_sym_y)? log_zero_sym_y : "" );
						}
						_log_zero_x= log_zero_x;
						_log_zero_y= log_zero_y;
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp,argv[idx], "-pow", 4) == 0 && (argv[idx][4]== 'x' || argv[idx][4]== 'y') ) {
					if (Check_Option( strncasecmp,argv[idx], "-powxy", 6) == 0) {
						if( Opt01== 0){
							sqrtYFlag= sqrtXFlag = -1;
						}
						else{
							sqrtYFlag= sqrtXFlag = 2;
						}
						if( idx+ 1 >= argc){
							if( Opt01 ){
								argerror( "missing value", argv[idx]);
							}
						}
						else{
							if( argv[idx][5]== 'y'){
								if( sscanf( argv[idx+1], "%lf,%lf", &powXFlag, &powYFlag)== 1)
									powYFlag= powXFlag;
							}
							else{
								sscanf( argv[idx+1], "%lf", &powXFlag );
							}
						}
					}
					else if (Check_Option( strncasecmp,argv[idx], "-powx", 5) == 0) {
						if( Opt01== 0){
							sqrtXFlag = -1;
						}
						else{
							sqrtXFlag = 2;
						}
						if( idx+ 1 >= argc){
							if( Opt01 ){
								argerror( "missing value", argv[idx]);
							}
						}
						else{
							sscanf( argv[idx+1], "%lf", &powXFlag );
						}
					}
					else if (Check_Option( strncasecmp,argv[idx], "-powy", 5) == 0) {
						if( Opt01== 0)
							sqrtYFlag = -1;
						else
							sqrtYFlag = 2;
						if( idx+ 1 >= argc){
							if( Opt01 ){
								argerror( "missing value", argv[idx]);
							}
						}
						else{
							sscanf( argv[idx+1], "%lf", &powYFlag );
						}
					}
					if( sqrtXFlag> 0 && logXFlag> 0)
						logXFlag= -1;
					if( sqrtYFlag> 0 && logYFlag> 0)
						logYFlag= -1;
					idx+= 2;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-powA", 5) == 0 ){
					if( idx+ 1 >= argc){
						argerror( "missing value", argv[idx]);
					}
					else{
						sscanf( argv[idx+1], "%lf", &powAFlag );
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp,argv[idx], "-sqrt", 5) == 0 && (argv[idx][5]== 'x' || argv[idx][5]== 'y') ) {
					if (Check_Option( strncasecmp,argv[idx], "-sqrtxy", 7) == 0) {
						if( Opt01== 0)
							sqrtYFlag= sqrtXFlag = -1;
						else
							sqrtYFlag= sqrtXFlag = 2;
					}
					else if (Check_Option( strncasecmp,argv[idx], "-sqrtx", 6) == 0) {
						if( Opt01== 0)
							sqrtXFlag = -1;
						else
							sqrtXFlag = 2;
					}
					else if (Check_Option( strncasecmp,argv[idx], "-sqrty", 6) == 0) {
						if( Opt01== 0)
							sqrtYFlag = -1;
						else
							sqrtYFlag = 2;
					}
					if( sqrtXFlag> 0 && logXFlag> 0)
						logXFlag= -1;
					if( sqrtYFlag> 0 && logYFlag> 0)
						logYFlag= -1;
					idx++;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-disconnect", -1) == 0){
					disconnect= 1;
					idx++;
				}
				else if( Check_Option( strcasecmp, argv[idx], "-auto", -1) == 0){
					autoscale= 2;
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-absy", 5) == 0) {
					absYFlag= 1;
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-radix_offset", 10) == 0) {
				  /* This is used to just specify the radix, which formerly was called polarBase	*/
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						get_radix( argv[idx+1], &radix_offset, radix_offsetVal );
						if( debugFlag){
							fprintf( StdErr, "radix_offset: %lf\n", radix_offset);
						}
						Gonio_Base( NULL, radix, radix_offset );
						idx+= 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-radix", 6) == 0) {
				  /* This is used to just specify the radix, which formerly was called polarBase	*/
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						get_radix( argv[idx+1], &radix, radixVal );
						if( debugFlag){
							fprintf( StdErr, "radix: %lf\n", radix);
						}
						Gonio_Base( NULL, radix, radix_offset );
						idx+= 2;
					}
				}
				else if (Check_Option( strncasecmp,argv[idx], "-polar", 6) == 0) {
					  /* Draw polar graph */
					polarFlag = 2;
					barFlag= 0;
					if( argv[idx][6]){
						get_radix( &argv[idx][6], &radix, radixVal);
						Gonio_Base( NULL, radix, radix_offset );
					}
					idx++;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-triangle", 9 ) == 0) {
					  /* Draw bar graph */
					if( Opt01== 0 ){
						triangleFlag= 0;
						error_type= -1;
					}
					else{
						triangleFlag = 2;
						error_type= 2;
					}
					if( ActiveWin && ActiveWin->error_type && ActiveWin->error_type[setNumber]== -1 ){
						ActiveWin->error_type[setNumber]= error_type;
					}
					else if( AllSets && AllSets[setNumber].numPoints<= 0 ){
						AllSets[setNumber].error_type= error_type;
					}
					idx++;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-error_region", 13 ) == 0) {
					  /* Draw bar graph */
					if( Opt01== 0 ){
						error_regionFlag= 0;
						error_type= -1;
					}
					else{
						error_regionFlag = 2;
						error_type= 3;
					}
					if( ActiveWin && ActiveWin->error_type && ActiveWin->error_type[setNumber]== -1 ){
						ActiveWin->error_type[setNumber]= error_type;
					}
					else if( AllSets && AllSets[setNumber].numPoints<= 0 ){
						AllSets[setNumber].error_type= error_type;
					}
					idx++;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-bar", -1) == 0) {
					/* Draw bar graph */
					barFlag = 2;
					idx++;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-brw", -1) == 0) {
					/* Set width of bar */
					if (idx+1 >= argc)
						argerror("missing width", argv[idx]);
					else{
						(void) sscanf(argv[idx+1], "%lf", &barWidth);
						barWidth_set= 1;
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-brb", -1) == 0) {
					if (idx+1 >= argc)
						argerror("missing base", argv[idx]);
					else{
						(void) sscanf(argv[idx+1], "%lf", &barBase);
						barBase_set= 1;
						idx += 2;
					}
				}
				else if (Check_Option( strcasecmp,argv[idx], "-brt", -1) == 0) {
					if (idx+1 >= argc)
						argerror("missing type", argv[idx]);
					else{
						CLIP_EXPR(barType, atoi(argv[idx+1]), 0, BARTYPES-1 );
						barType_set= 1;
						idx += 2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-bar_legend_dimension", 21)== 0 ){
					if (idx+1 >= argc)
						argerror("missing dimension weight specifications", argv[idx]);
					else{
					  double bw[3];
					  int n= 3, r;
					  extern double bar_legend_dimension_weight[3];
						memcpy( bw, bar_legend_dimension_weight, sizeof(bw) );
// 						r= fascanf( &n, argv[idx+1], bw, NULL, NULL, NULL, NULL);
						r= fascanf2( &n, argv[idx+1], bw, ',' );
						if( r>= 1 ){
							memcpy( bar_legend_dimension_weight, bw, sizeof(bw) );
						}
						idx+= 2;
					}
				}
				else if( strcmp( argv[idx], "-debugging")==0 ){
					debugging= 1;
					idx+= 1;
				}
				else if (Check_Option( strncmp,argv[idx], "-sV", 3) == 0){
					Conditional_Toggle( scriptVerbose );
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-proc_ign_ev", 12) == 0) {
				  extern int data_silent_process;
					Conditional_Toggle( data_silent_process );
					idx+= 1;
				}
				else if (Check_Option( strncasecmp,argv[idx], "-synchro", 8) == 0) {
				  extern int HardSynchro;
					Conditional_Toggle( HardSynchro );
					Conditional_Toggle( Synchro_State );
					Synchro_State= !Synchro_State;
					X_Synchro(NULL);
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-db_expr_error", 14) == 0) {
				  extern int Redo_Error_Expression;
					Conditional_Toggle( Redo_Error_Expression );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-warnOutliers", -1) == 0 ){
					ReadData_Outliers_Warn= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-wns", 4) == 0 ){
					WarnNewSet= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-UnmappedWindowTitle", -1) == 0) {
				  extern int UnmappedWindowTitle;
					UnmappedWindowTitle= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-SetIconName", -1) == 0) {
				  extern int SetIconName;
					SetIconName= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-ReparentDialogs", -1) == 0) {
				  extern int xtb_ReparentDialogs;
					xtb_ReparentDialogs= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-RemoteConnection", 14) == 0) {
				  static int db= 1;
					RemoteConnection = (Opt01== -1)? db : Opt01;
					db= !RemoteConnection;
					if( debugFlag && RemoteConnection ){
						fprintf( StdErr, "Using RemoteConnection mode\n" );
					}
					fflush( StdErr);
					idx++;
				}
				else if( Check_Option( strncasecmp,argv[idx], "-db", 3) == 0) {
				  static int db= 1;
				  int l;
					debugFlag = (Opt01== -1)? db : Opt01;
					db= !debugFlag;
					if( idx< argc- 1 && (sscanf(argv[idx+1], "%d", &l)== 1) ){
						debugLevel= l;
						idx+= 1;
					}
					if( debugFlag ){
						fprintf( StdErr, "DebugLevel is %d\n", debugLevel);
					}
					else{
						fprintf( StdErr, "Debug off (level %d)\n", debugLevel);
					}
					fflush( StdErr);
					if( !RemoteConnection ){
						Synchro_State= 0;
						X_Synchro(NULL);
						if( !RemoteConnection ){
							XFlush(disp);
						}
					}
					idx++;
				}
				else if (Check_Option( strcasecmp,argv[idx], "-display", -1) == 0) {
					/* Harmless display specification */
					idx += 2;
				}
				else if( strncasecmp( argv[idx], "-VisualType", 11)== 0 ){
				  int x;
				  extern char *VisualClass[];
				  extern int VisualClasses;
					if( idx+1 >= argc || argv[idx+1][0]== 0 ){
						argerror( "missing or invalid type", argv[idx]);
					}
					else{
						x= 0;
						while( x< VisualClasses && strncasecmp( argv[idx+1], VisualClass[x], strlen(VisualClass[x])) ){
							x+= 1;
						}
						if( x>= VisualClasses ){
							argerror( "Visualtype must be one of: StaticGray GrayScale StaticColor PseudoColor TrueColor DirectColor",
								argv[idx]
							);
						}
						else{
							ux11_vis_class= x;
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-use_XDBE", -1)== 0 ){
				  int x= (Opt01)? True : False;
					  /* Have to do a more complicated test here (than just x!=ux11_useDBE) since ux11_useDBE is set to -1
					   \ when an X11 Visual has been found that supports DBE.
					   */
					if( (x && !ux11_useDBE) || (!x && ux11_useDBE) ){
						ux11_useDBE= x;
#ifdef RUNTIME_VISUAL_CHANGE
						XG_choose_visual();
						ReallocColours(True);
						fprintf( StdErr, "%s: warning: New visual chosen.\n", argv[idx] );
#else
						if( !install_flag ){
							install_flag= 1;
						}
						if( ActiveWin ){
							xtb_error_box( ActiveWin->window, "New value set, but runtime visual changes are not supported yet",
								"-use_XDBE Notice"
							);
						}
						else{
							fprintf( StdErr, "%s: New value set, but runtime visual changes are not supported yet\n",
								argv[idx]
							);
						}
#endif
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-MinBitsPPixel", -1)== 0 ){
				  int x;
					if( idx+1 >= argc || argv[idx+1][0]== 0 )
						argerror( "missing or invalid depth", argv[idx]);
					else{
						if( (x= atoi( argv[idx+1]))> 0 && x<= 24 && x!= ux11_min_depth ){
#ifdef RUNTIME_VISUAL_CHANGE
							ux11_min_depth= x;
							XG_choose_visual();
							ReallocColours(True);
							fprintf( StdErr, "%s: warning: New visual chosen.\n", argv[idx] );
#else
							if( !install_flag ){
								install_flag= 1;
							}
							if( ActiveWin ){
								xtb_error_box( ActiveWin->window, "New value set, but runtime visual changes are not supported yet",
									"-MinBitsPPixel Notice"
								);
							}
							else{
								fprintf( StdErr, "%s %s: New value set, but runtime visual changes are not supported yet\n",
									argv[idx], argv[idx+1]
								);
							}
							ux11_min_depth= x;
#endif
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-average_error", 14) == 0){
					Conditional_Toggle( use_average_error);	
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-noerr", -1) == 0){
					if( Opt01 ){
						use_errors= 0;	
						no_errors= 1;
					}
					else{
						use_errors= 1;	
						no_errors= 0;
					}
					idx++;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-geometry", 4)== 0){
					if( idx+1 >= argc)
						argerror( "missing specification", argv[idx]);
					else{
						  /* Its a geometry specification */
						geoSpec = XGstrdup(argv[idx+1]);
						if( ActiveWin ){
						  XSizeHints hints;
							xtb_ParseGeometry( geoSpec, &hints, ActiveWin->window, True );
							geoSpec[0]= '\0';
						}
						idx+=2;
					}
				}
				else if( Check_Option( strcasecmp, argv[idx], "-zero" , -1)== 0 ){
				  double x;
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) )
							zero_epsilon= x;
						if( debugFlag)
							fprintf( StdErr, "zero_epsilon=%.15g\n", zero_epsilon);
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-overwrite_legend" , 17)== 0 ){
					Conditional_Toggle( overwrite_legend);
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-overwrite_AxGrid" , 17)== 0 ){
					Conditional_Toggle( overwrite_AxGrid);
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-start_arrow" , 10)== 0 ){
					if( Opt01 ){
						arrows|= 1;
					}
					else{
						arrows&= ~1;
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-end_arrow" , 8)== 0 ){
					if( Opt01 ){
						arrows|= 2;
					}
					else{
						arrows&= ~2;
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-overwrite_marks" , 16)== 0 ){
					Conditional_Toggle(overwrite_marks);
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-raw_display_init" , 17)== 0 ||
					Check_Option( strncasecmp, argv[idx], "-raw_init_display" , 17)== 0
				){
				  extern int raw_display_init;
					raw_display= True;
					raw_display_init= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-raw_display" , 12)== 0 ){
					Conditional_Toggle(raw_display);
					if( ActiveWin && ActiveWin!= &StubWindow ){
						RawDisplay( ActiveWin, raw_display );
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-process_bounds" , 15)== 0 ){
					Conditional_Toggle( process_bounds);
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-transform_axes" , 15)== 0 ){
					Conditional_Toggle( transform_axes);
					if( !transform_axes && (polarFlag> 0 || logXFlag> 0 || logYFlag> 0 || sqrtXFlag> 0 || sqrtYFlag> 0) ){
						fprintf( StdErr, "Warning: possibly illegal combination of -polar, -sqrt? or -ln? and -transform_axes\n" );
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-show_overlap" , 13)== 0 ){
					Conditional_Toggle( show_overlap);
					if( show_overlap && idx+1< argc ){
						if( !strcmp(argv[idx+1], "raw") ){
							show_overlap= 2;
							idx+= 1;
						}
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-progress" , 9)== 0 ){
					Conditional_Toggle( Show_Progress);
					idx+= 1;
				}
				else if(
					Check_Option( strncasecmp, argv[idx], "-fascanf_verbose" , 16)== 0
					|| Check_Option( strncasecmp, argv[idx], "-ascanf_verbose" , 15)== 0
				){
					ascanf_verbose= Opt01;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-compile_verbose" , 16)== 0){
				  extern double *ascanf_compile_verbose;
					if( ascanf_compile_verbose ){
						*ascanf_compile_verbose= (Opt01)? 1 : 0;
					}
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-NewProcess_Rescales" , 20)== 0 ){
					Conditional_Toggle( NewProcess_Rescales );
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-fit_xybounds", 8)== 0 ){
					if( Opt01> 2 ){
					  /* backw. comp.	*/
						Opt01= -1;
					}
					lFX= FitX= abs(Opt01);
					lFY= FitY= abs(Opt01);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-fit_xbounds", 7)== 0 ){
					if( Opt01> 2 ){
					  /* backw. comp.	*/
						Opt01= -1;
					}
					lFX= FitX= abs(Opt01);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-fit_ybounds", 7)== 0 ){
					if( Opt01> 2 ){
					  /* backw. comp.	*/
						Opt01= -1;
					}
					lFY= FitY= abs(Opt01);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-fit_after", 5)== 0 ){
				  extern int fit_after_draw;
					if( Opt01> 2 ){
					  /* backw. comp.	*/
						Opt01= -1;
					}
					fit_after_draw= abs(Opt01);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-win_aspect", 7)== 0 ){
				  double x;
					if (idx+1 >= argc)
						argerror("missing number", argv[idx]);
					else{
						if( sscanf(argv[idx+1], "%lf", &x) ){
							win_aspect= x;
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strncmp, argv[idx], "-x_symmetric", 7)== 0 ){
					XSymmetric= ABS(Opt01);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-y_symmetric", 7)== 0 ){
					YSymmetric= ABS(Opt01);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-aspect", 7)== 0 ){
					Aspect= ABS(Opt01);
					idx+= 1;
				}
				else if( Check_Option( strncmp, argv[idx], "-fractions", 5)== 0 ){
				  extern int Allow_Fractions;
					Allow_Fractions= ABS(Opt01);
					changed_Allow_Fractions= True;
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-pipe", 5)== 0 ){
					if (idx+1 >= argc){
						argerror( "missing specification", argv[idx]);
					}
					else if( ReadPipe_fp && ReadPipe_name ){
						if( strcasecmp( argv[idx+1], "close" )== 0 ){
							ReadPipe= False;
						}
						else{
							fprintf( StdErr, "-pipe %s: ignored because already reading from another pipe (%s)\n",
								argv[idx+1], ReadPipe_name
							);
						}
					}
					else{
						ReadPipe= True;
						ReadPipe_name= XGstrdup( argv[idx+1] );
					}
					idx+= 2;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-skip_to", 8)== 0 ){
				  extern char *skip_to_label;
					if (idx+1 >= argc){
						argerror( "missing label", argv[idx]);
					}
					skip_to_label= strdup(argv[idx+1]);
					idx+= 2;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-script", 7)== 0 ||
					Check_Option( strcasecmp, argv[idx], "-f", 0)== 0
				){
					if (idx+1 >= argc){
						argerror( "missing specification", argv[idx]);
					}
					else{
						if( No_IncludeFiles ){
							fprintf( StdErr, "ParseArgs(%s): \"%s\" ignored because of -NoIncludes\n",
								argv[idx], argv[idx+1]
							);
							fflush( StdErr );
						}
						else{
							if( argv[idx][2]== 0 ){
							  /* 20010601: multiple -f options can be given */
								if( ScriptFile ){
									ScriptFile= concat2( ScriptFile, "\t", argv[idx+1], NULL );
								}
								else{
									ScriptFile= XGstrdup( argv[idx+1] );
								}
							}
							else{
								xfree( ScriptFile );
								ScriptFile= XGstrdup( argv[idx+1] );
							}
							ScriptFileWin= ActiveWin;
						}
						idx+= 2;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-SwapEndian", 11)== 0 ){
					Conditional_Toggle( SwapEndian );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-remove_inputfiles", 18)== 0 ){
				  extern int RemoveInputFiles;
					Conditional_Toggle( RemoveInputFiles );
					idx+= 1;
				}
				else if( Check_Option( strncasecmp, argv[idx], "-2ndGEN", 7)== 0 ){
					Conditional_Toggle( NextGen );
					idx+= 1;
				}
				else if( strcmp( argv[idx], "-XGraph")== 0 ){
					argerror( "option should be specified as the firstmost.", argv[idx]);
				}
				else if( Check_Option( strncmp, argv[idx], "-xrm", 4)== 0 ){
					if (idx+1 >= argc){
						argerror( "missing resource string", argv[idx]);
					}
					else{
					  extern char *ResourceTemplate;
					  char *rt= ResourceTemplate;
						ResourceTemplate= argv[idx+1];
						if( strncmp( ResourceTemplate, Prog_Name, strlen(Prog_Name))== 0 ){
							ResourceTemplate+= strlen(Prog_Name);
						}
						else if( index( ".*", *ResourceTemplate) ){
						  /* nothing */
						}
						else{
							ResourceTemplate= NULL;
						}
						if( ResourceTemplate ){
							if( index( ".*", *ResourceTemplate) ){
								ResourceTemplate++;
								while( isspace( *ResourceTemplate ) ){
									ResourceTemplate++;
								}
							}
							else{
								ResourceTemplate= NULL;
							}
						}
						if( ResourceTemplate ){
							ReadDefaults();
						}
						else{
							fprintf( StdErr, "ParseArgs(): -xrm \"%s\" does not specify a resource for this programme \"%s\"\n",
								argv[idx+1], Prog_Name
							);
						}
						ResourceTemplate= rt;
						idx+= 2;
					}
				}
				else if( Check_Option( strncmp, argv[idx], "-tar", 4)== 0 ){
					if( Opt01> 0 ){
						if (idx+1 >= argc){
							argerror( "missing tar archive name", argv[idx]);
						}
						From_Tarchive= Opt01;
						strncpy( Tarchive, argv[idx+1], sizeof(Tarchive)/sizeof(char) );
						idx+= 2;
					}
					else{
						Tarchive[0]= '\0';
						From_Tarchive= 0;
						idx+= 1;
					}
				}
				else if( Check_Option( strncasecmp, argv[idx], "-registry", 9)== 0 ){
				    // get the current setting:
				  int rVN = register_VariableNames(1);
					Conditional_Toggle( rVN );
					 // select the desired setting:
					register_VariableNames(rVN);
					idx+= 1;
				}
				else {
					argerror("unknown option", argv[idx]);
				}
			}
		}
		else if (argv[idx][0] == '=') {
			  /* Its a geometry specification */
			geoSpec = XGstrdup( &(argv[idx][1]) );
			if( ActiveWin ){
			  XSizeHints hints;
				xtb_ParseGeometry( geoSpec, &hints, ActiveWin->window, True );
				geoSpec[0]= '\0';
			}
			idx++;
		}
		else {
			Add_InFile( argv[idx] );
			idx++;
		}

		if( ArgError ){
			r+= 1;
		}
    }
	return( r );
}

extern char *cleanup( char *T);

int ParseArgsString( char *string )
{ int argc= 1, first= 0, quote= 0;
  char **argv, *ArgBuf= XGstrdup(string), *argbuf= ArgBuf;
/* We define our own Isspace() (isspace). The ctype version will
 \ accept (some) extended ascii characters as whitespace on some
 \ machines. One example: \\#xa5\\ (after parse_codes) will yield
 \ two argumentlist-entries "\\", with the result that we cannot
 \ specify an infinity symbol (in greekFont).
 \ NOTE: I have no idea why this happens. On my A/UX mac, where this
 \ happens, the ctype implementation is functionally identical to the HP,
 \ where it does not. Compiler bug??
 \ 950915 NO. It just shows these macros are old. Indexing an array with
 \ a macro-argument without casting is DANGEROUS. In this case, e.g. char c=·
 \ (0xb7) will expand to __ctype[0xb7], or (as most chars are signed) __ctype[-73].
 \ I doubt if the array is defined there... The solution used elsewhere is
 \ to first cast the argument to an unsigned char.
*/
#define Isspace(c)	((c)==' '||(c)=='	'||(c)=='\n'||(c)=='\r')
	if( !ArgBuf ){
		fprintf( StdErr, "ParseArgsString(%s): can't get temp. buffer for parsing (%s)\n", (string)? string : "<NULL>", serror() );
		return(0);
	}
	while( *argbuf && Isspace( *argbuf ) ){
		argbuf++;
		first+= 1;
	}
	cleanup( argbuf );
	  /* find all arguments	*/
	if( *argbuf ){
		argc+= 1;
	}
	while( *argbuf ){
		if( *argbuf== '"' ){
			quote= !quote;
		}
		if( !quote && Isspace( *argbuf ) ){
			while( *argbuf && Isspace( *argbuf ) ){
				argbuf++;
			}
			if( *argbuf ){
				argc+= 1;
			}
		}
		else{
			argbuf++;
		}
	}
	argbuf= &ArgBuf[first];
	quote= 0;
	if( (argv= (char**) calloc( argc, sizeof(char*) ) ) ){
		argv[0]= Prog_Name;
		argc= 1;
		if( *argbuf== '"' ){
			quote= 1;
			argbuf++;
		}
		argv[argc++]= argbuf;
		while( *argbuf ){
			if( (!quote && Isspace( *argbuf )) || (quote && *argbuf== '"') ){
				if( *argbuf== '"' && quote ){
					quote= 0;
				}
				*argbuf= '\0';
				argbuf++;
				while( *argbuf && Isspace( *argbuf ) ){
					argbuf++;
				}
				if( *argbuf== '"' ){
					quote= !quote;
					argbuf++;
				}
				if( *argbuf ){
					argv[argc++]= argbuf;
				}
			}
			else{
				if( *argbuf== '"' ){
					quote= !quote;
				}
				argbuf++;
			}
		}
		if( debugFlag ){
			fprintf( StdErr, "ParseArgsString(%s): %d options; ",
				ArgBuf, argc
			);
			for( first= 1; first< argc; first++ ){
				fprintf( StdErr, "%s ", argv[first] );
			}
			fputc( '\n', StdErr );
		}
		ParseArgs( argc, argv);
		xfree( argv );
		xfree( ArgBuf );
		return( argc );
	}
	else{
		fprintf( StdErr, "ParseArgsString(%s): can't get buffer for %d options\n", (string)? string : "<NULL>", argc);
		return(0);
	}
}

int ParseArgsString2( char *optbuf, int set_nr )
{ int i, tg= TrueGray;
  int r= ParseArgsString( optbuf );
  DataSet *this_set;
	if( r> 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			Set_X( ActiveWin, &(ActiveWin->dev_info) );
		}
		if( tg!= TrueGray ){
			ReallocColours(True);
		}
		for( i= set_nr; i< MAXSETS; i++){
			this_set= &AllSets[i];
			this_set->lineWidth= lineWidth;
			this_set->elineWidth= errorWidth;
			this_set->markFlag= markFlag;
			this_set->noLines= noLines;
/* 
			if( this_set->pixvalue>= 0 && this_set->pixvalue< MAXATTR ){
				this_set->pixelValue= AllAttrs[this_set->pixvalue].pixelValue;
				xfree( this_set->pixelCName );
				this_set->pixelCName= XGstrdup( AllAttrs[this_set->pixvalue].pixelCName );
			}
 */
			this_set->pixelMarks= pixelMarks;
			this_set->barFlag= barFlag;
			if( barBase_set ){
				this_set->barBase= barBase;
				this_set->barBase_set= barBase_set;
			}
			if( barWidth_set ){
				this_set->barWidth_set= barWidth_set;
				this_set->barWidth= barWidth;
			}
			if( barType_set ){
				this_set->barType= barType;
			}
			this_set->polarFlag= polarFlag;
			this_set->radix= radix;
			this_set->radix_offset= radix_offset;
			this_set->vectorLength= vectorLength;
			this_set->vectorType= vectorType;
			memcpy( this_set->vectorPars, vectorPars, MAX_VECPARS* sizeof(double));
			  /* 20020826:.... */
			this_set->arrows = arrows;
			this_set->overwrite_marks = overwrite_marks;
			if( !this_set->ncols ){
				this_set->ncols= NCols;
			}
			this_set->xcol= xcol;
			this_set->ycol= ycol;
			this_set->ecol= ecol;
			this_set->lcol= lcol;
			this_set->Ncol= Ncol;
		}
		if( ActiveWin && ActiveWin!= &StubWindow ){
			TransformCompute( ActiveWin, False );
		}
	}
	return(r);
}

int ParseArgsString3( char *argbuf, int set_nr )
{ int DF= DumpFile, DB= DumpBinary, DI= DumpIncluded, ret;
	ret = ParseArgsString2( argbuf, set_nr );

	  /* The following 3 which have to do with Dumping to the terminal
	   \ should not be allowed to (be) changed in-file, only by a real 
	   \ commandline argument.
	   */
	DumpFile= DF;
	if( DF ){
		DumpBinary= DB;
		DumpIncluded= DI;
	}
	return( ret );
}

int Evaluate_ExpressionList( LocalWin *wi, XGStringList **Exprs, int dealloc, char *descr )
{ static int active= 0;
  FILE *strm;
	if( !Exprs && !wi ){
		active= 0;
		return(0);
	}
	else{
	  char *tnam= XGtempnam( getenv("TMPDIR"), "XGEXL");
	  XGStringList *exprs= *Exprs;
	  int r= 0;
	  char asep = ascanf_separator;
		if( active ){
			return(-1);
		}
		if( tnam && (strm= fopen( tnam, "wb")) ){
			active= 1;
			// 20101113: we honour just the list's 1st separator setting:
			if( exprs->separator ){
				ascanf_separator = exprs->separator;
			}
			while( exprs ){
			  XGStringList *cur= exprs;
			  char *c= cur->text;
				if( debugFlag ){
					fprintf( strm, "#%s\n", c );
				}
				while( *c ){
					if( *c== '\\' && c[1]== 'n' ){
						fputs( "\\n\n", strm );
						c+= 3;
					}
					else{
						fputc( *c++, strm );
					}
				}
				if( cur->text[strlen(cur->text)-1]!= '\n' ){
					fputs( "\n", strm );
				}
				if( debugFlag ){
					fprintf( StdErr, "{%s} \"%s\"\n", descr, cur->text );
					fflush( StdErr );
				}
				exprs= cur->next;
				if( dealloc ){
					xfree( cur->text );
					xfree( cur );
				}
			}
			if( dealloc ){
				*Exprs= exprs;
			}
			fclose( strm );
			if( (strm= fopen( tnam, "r" )) ){
			  char *comm= concat( "Evaluating file with ", descr, "\n", NULL );
			  LocalWin *aw= ActiveWin;
				  /* Make sure the file disappears (completely) when closed the next time	*/
				unlink( tnam );
				if( wi ){
				  extern double *ascanf_ActiveWinWidth, *ascanf_ActiveWinHeight;
					ascanf_window= wi->window;
					*ascanf_ActiveWinWidth= wi->XOppX - wi->XOrgX;
					*ascanf_ActiveWinHeight= wi->XOppY - wi->XOrgY;
					ActiveWin= wi;
					  // 20101019:
					CopyFlags(NULL, wi);
				}
				add_comment( comm );
				r= IncludeFile( wi, strm, tnam, True, NULL );
				fclose( strm );
				xfree( comm );
				comm= concat( "Finished with file with ", descr, "\n", NULL );
				add_comment( comm );
				xfree( comm );
				ascanf_separator = asep;
				if( wi && wi->delete_it== -1 ){
					wi->redraw= 1;
					wi->printed= 0;
					ActiveWin= aw;
					wi->event_level--;
					wi->drawing= 0;
					return(-2);
				}
				ActiveWin= aw;
			}
			else{
				XG_error_box( &wi, "Can't re-open (read) temporary file\n", tnam, "(", descr, ")\n", NULL );
			}
			xfree(tnam);
			active= 0;
			ascanf_separator = asep;
		}
		else{
			if( tnam ){
				XG_error_box( &wi, "Can't open (write) temporary file\n", tnam, "(", descr, ")\n", NULL );
				xfree(tnam);
			}
			else{
				XG_error_box( &wi, "Can't open (write) temporary file\n", "<NULL!!>", "(", descr, ")\n", NULL );
			}
		}
		return(r);
	}
}

void Duplicate_Visible_Sets(LocalWin *wi, int new_win )
{  char errmesg[ERRBUFSIZE];
   FILE *fp;
   char *f_o_d= NULL, *geo= ( wi->dev_info.resized== 1 )?  " -print_sized" : "";
   char *self= (Argv[0])? Argv[0] : "xgraph";
   XWindowAttributes win_attr;
   Window dummy;

	if( new_win /* CheckMask( buttonmask, ControlMask) */ ){
		XGetWindowAttributes(disp, wi->window, &win_attr);
		XTranslateCoordinates( disp, wi->window, win_attr.root, 0, 0,
			&win_attr.x, &win_attr.y, &dummy
		);
		win_attr.y-= WM_TBAR;
		sprintf( errmesg, " -geometry %dx%d+%d+%d ", win_attr.width, win_attr.height,
			win_attr.x- win_attr.border_width, win_attr.y
		);

		if( UPrintFileName[0] ){
			f_o_d= concat( "env XGRAPH_ARGUMENTS= ", self, errmesg, " -detach -pf ", UPrintFileName, geo, NULL);
		}
		else if( PrintFileName ){
			f_o_d= concat( "env XGRAPH_ARGUMENTS= ", self, errmesg, " -detach -pf ", PrintFileName, geo, NULL);
		}
		else{
			f_o_d= concat( "env XGRAPH_ARGUMENTS= ", self, errmesg, " -detach ", geo, NULL);
		}
		fp= popen( f_o_d, "wb");
	}
	else /* if( CheckMask( buttonmask, Mod1Mask) ) */{
		f_o_d= XGtempnam( getenv("TMPDIR"), "xgrph");
		if( f_o_d ){
			fp= fopen( f_o_d, "wb");
		}
		else{
			fp = NULL;
		}
	}
	if( debugFlag ){
		fprintf( StdErr, "%s\n", (f_o_d)? f_o_d : "<NULL!!>" );
		fflush( StdErr );
	}
	if( fp ){
	  extern int XGDump_PrintPars, XG_Stripped, XGDump_Labels, Init_XG_Dump;
	  int xdpp= XGDump_PrintPars, DP= wi->DumpProcessed, DA= wi->dump_average_values,
		  XGS= XG_Stripped, DL= XGDump_Labels, IXD= Init_XG_Dump;
		errmesg[0]= '\0';
		  /* Make sure we dump (all!) the data we want, and in the
		   \ way we want (need..) it. So: dump the raw values and
		   \ the transforming routines, but don't dump the averaging commands
		   \ but the data instead. This ressembles as closely as possible what
		   \ would happen with an in-process copy of the window.
		   */
		CopyFlags( NULL, wi );
		wi->DumpProcessed= (new_win /* CheckMask( buttonmask, ControlMask) */)? 0 : 1;
		wi->dump_average_values= 1;
		XG_Stripped= (new_win /* CheckMask( buttonmask, ControlMask) */)? 0 : 1;
		if( new_win /* CheckMask( buttonmask, ControlMask) */ ){
			if( !XGDump_PrintPars ){
				XGDump_PrintPars= 1;
			}
		}
		else{
			XGDump_PrintPars= 0;
			Init_XG_Dump= False;
		}
		XGDump_Labels= (new_win /* CheckMask( buttonmask, ControlMask) */)? 1 : 0;
		_XGraphDump( wi, fp, errmesg );
		wi->DumpProcessed= DP;
		wi->dump_average_values= DA;
		XG_Stripped= XGS;
		XGDump_Labels= DL;
		XGDump_PrintPars= xdpp;
		Init_XG_Dump= IXD;
		if( errmesg[0] ){
			xtb_error_box( wi->window, errmesg, "Error" );
		}
		if( new_win /* CheckMask( buttonmask, ControlMask) */ ){
			pclose( fp );
		}
		else{
			fclose( fp );
			if( (fp= fopen( f_o_d, "r") ) ){
			  extern int Raw_NewSets, CorrectLinks;
			  int rns= Raw_NewSets, CL= CorrectLinks;
				Raw_NewSets= True;
				CorrectLinks= False;
				IncludeFile( wi, fp, f_o_d, True, NULL );
				Raw_NewSets= rns;
				CorrectLinks= CL;
				fclose( fp );
				unlink( f_o_d );
			}
		}
	}
	else{
		sprintf( errmesg, "Can't open stream to \"%s\" (%s)\n", f_o_d, serror() );
		xtb_error_box( wi->window, errmesg, "Error" );
	}
	if( f_o_d ){
		xfree( f_o_d );
	}
}

void XG_SimpleConsole()
{ extern int read_params_now;
  static char active= 0;
	if( active ){
		fprintf( StdErr, "Console already active!\n" );
		return;
	}
	if( isatty(fileno(stdin)) ){
	  char *buffer;
	  int lnr= 0, pnnr= 0, cont= True, ae= ascanf_exit;
	  double val= 0;
	  LocalWin *AW= ActiveWin;
		if( !(buffer = (char*) calloc( LMAXBUFSIZE+2, sizeof(char) )) ){
			fprintf( StdErr, "No memory for %d byte console input buffer (%s)\n", LMAXBUFSIZE+2, serror() );
			return;
		}
		buffer[0]= '\0';
		ActiveWin= &StubWindow;
		ascanf_exit= False;
		active= 1;
		while( cont && !feof(stdin) && !ferror(stdin) && !ascanf_exit ){
			if( read_params_now== 2 ){
				pnnr+= interactive_param_now_xwin( StubWindow.window, buffer, -2,
					LMAXBUFSIZE, "Type exit to exit", &val, True, 1, False );
				if( buffer[0] ){
					lnr+= 1;
				}
			}
			else{
				if( Use_ReadLine ){
					if( ReadLine( buffer, LMAXBUFSIZE, "# ", &lnr, XGStartJoin, XGEndJoin ) ){
						if( DumpFile ){
							fputs( buffer, stdout );
						}
						pnnr+= new_param_now( buffer, &val, -1 );
						lnr+= 1;
					}
					else{
						cont= False;
					}
				}
				else{
					fputs( "# ", StdErr ); fflush( StdErr );
					if( ReadString( buffer, LMAXBUFSIZE, stdin, &lnr, XGStartJoin, XGEndJoin ) ){
						if( DumpFile ){
							fputs( buffer, stdout );
						}
						pnnr+= new_param_now( buffer, &val, -1 );
						lnr+= 1;
					}
				}
			}
		}
		if( pnnr ){
		  extern char d3str_format[16];
			fprintf( StdErr, "%d lines read, %d non-void expressions, final result is %s\n",
				lnr, pnnr, d2str( val, d3str_format, NULL)
			);
		}
		ActiveWin= AW;
		ascanf_exit= ae;
		xfree(buffer);
		active= 0;
	}
	else{
		fprintf( StdErr, "Not an interactive terminal, no console fonctionality.\n" );
	}
}
