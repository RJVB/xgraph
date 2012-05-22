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

 \\ This has become a HUGE module. Should be split up into several
 \\ (when I get into the mood....)
 \\ Hurray - we now have xgsupport.c (i.e. two HUge files...)
 */

#include "config.h"
IDENTIFY( "Main module: main event loop and drawing routines" );

#ifdef linux
#	define _XOPEN_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <sys/param.h>
#include <math.h>
#include <string.h>
#ifndef _APOLLO_SOURCE
#	include <strings.h>
#endif
#ifdef _AUX_SOURCE
	extern int strncasecmp();
#endif

#ifdef _AUX_SOURCE
#	include <sys/types.h>
#endif

#if defined(__APPLE__) && (defined(i386) || defined(__i386__) || defined(x86_64) || defined(__x86_64__))
#	define USE_SSE2
#	include <xmmintrin.h>
#	include <emmintrin.h>
#	include "AppleVecLib.h"
#endif

#include <signal.h>

#include <pwd.h>
#include <ctype.h>
#include "xgout.h"
#include "xgraph.h"
#include "xtb/xtb.h"
#include "hard_devices.h"
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/Xatom.h>

#include "new_ps.h"
extern char ps_comment[1024];

#ifdef XG_DYMOD_SUPPORT
#	include "dymod.h"
#endif
#include "Python/PythonInterface.h"

#include <setjmp.h>
jmp_buf toplevel, stop_wait_event;
int wait_evtimer= 0;

#define ZOOM
#define TOOLBOX

#ifndef MAXFLOAT
#define MAXFLOAT	HUGE
#endif

#define BIGINT		0xfffffff

#if defined(__GNUC__) && defined(DEBUG)
	void *malloc_address= (void*) malloc;
#endif

#include "NaN.h"

#define GRIDPOWER 	10

int INITSIZE = 128;

#define CONTROL_D	'\004'
#define CONTROL_C	'\003'
#define TILDE		'~'

#define BTNPAD		1
#define BTNINTER	3

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
#define SIGN(x)		(((x)<0)?-1:(x>0)?1:0)
#define ZERO_THRES	-18.0	/* used to be: (1.0e-7); gcc generated wrong float constant in DrawGridAndAxis ** RJB **	*/
#ifndef _AUX_SOURCE
/* #	define zero_thres 1e-18	*/
#	define zero_thres DBL_MIN
#endif

#include <float.h>

#include "ascanf.h"
extern char ascanf_separator;

#include "XXseg.h"
#include "Elapsed.h"
#include "XG_noCursor"
#include <errno.h>
#include <sys/stat.h>
#include <X11/cursorfont.h>

#include "fdecl.h"
#include "copyright.h"

double zero_epsilon= DBL_EPSILON* 1.5;

extern int matherr_off, TrueGray;

#if defined(__SSE4_1__) || defined(__SSE4_2__)
#	define USE_SSE4
#	define SSE_MATHFUN_WITH_CODE
#	include "sse_mathfun.h"
#	include "arrayvops.h"
#elif defined(__SSE2__) || defined(__SSE3__)
#	include "arrayvops.h"
#endif

#define nsqrt(x)	(x<0.0 ? 0.0 : sqrt(x))
#define sqr(x)		((x)*(x))

#define ISCOLOR		(wi->dev_info.dev_flags & D_COLOR)

#define DEFAULT_PIXVALUE(set) 	((set) % MAXATTR)
#define DEFAULT_LINESTYLE(set) ((set)%MAXATTR)
#define MAX_LINESTYLE			MAXATTR /* ((bwFlag)?(MAXATTR-1):(MAXSETS/MAXATTR)-1) 	*/
#define LINESTYLE(set)	(AllSets[set].linestyle)
#define ELINESTYLE(set)	((AllSets[set].elinestyle>=0)?AllSets[set].elinestyle:LINESTYLE(set))
// #define LINEWIDTH(set)	(AllSets[set].lineWidth)
#define ELINEWIDTH(wi,set)	((AllSets[set].elineWidth>=0)?AllSets[set].elineWidth:LINEWIDTH(wi,set))
#define DEFAULT_MARKSTYLE(set) ((set)+1)
#define PIXVALUE(set)	AllSets[set].pixvalue, AllSets[set].pixelValue
#define MARKSTYLE(set)	(ABS(AllSets[set].markstyle)-1)

#define COLMARK(set) ((set) / MAXATTR)

extern char *ParsedColourName;
extern int GetColor( char *name, Pixel *pix);
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

inline int LINEWIDTH(LocalWin *wi, int set)
{ int r;
	if( !wi ){
		r= (int) ( ((AllSets[set].lineWidth>=0)? AllSets[set].lineWidth : -AllSets[set].lineWidth) + 0.5 );
	}
	else{
		if( AllSets[set].lineWidth>= 0 ){
			r= (int) (AllSets[set].lineWidth + 0.5);
		}
		else{
			  // RJVB 20081210: sometimes we'd like to specify a "real-world" width ...
			  // be aware that XDrawSegments (used by seg_X()) gives those a horrible look!
			if( wi->aspect ){
				r= SCREENXDIM(wi, -AllSets[set].lineWidth);
			}
			else{
				r= (int) ((SCREENXDIM(wi, -AllSets[set].lineWidth) + SCREENYDIM(wi, -AllSets[set].lineWidth)) / 2.0 + 0.5);
			}
		}
	}
	return(r);
}

#define BWMARK(set) \
((set) % MAXATTR)

#define NORMSIZEX	600
#define NORMSIZEY	400
#define NORMASP		(((double)NORMSIZEX)/((double)NORMSIZEY))
#define MINDIM		100

extern int XGTextWidth();
extern void ExitProgramme(), Restart(), Restart_handler(), Dump_handler();

extern void init_X();
#ifdef TOOLBOX
extern void do_error();
#endif

extern double psm_base, psm_incr, psdash_power;
extern int psm_changed, psMarkers, internal_psMarkers;
extern double Xdpi;

extern double dcmp(), dcmp2();

double page_scale_x= 1.0, page_scale_y= 1.0;
int preserve_screen_aspect= 0;
double win_aspect= 0.0;

extern int MaxSets;
int NCols= 3, MaxCols= 3;
int xcol= 0, ycol= 1, ecol= 2, lcol= -1, Ncol= -1, error_type= 1;
extern DataSet *AllSets;

char *Error_TypeNames[ERROR_TYPES]= { "No", "Bar", "Triangle", "Region", "Vector", "Intensity", "MarkerSize", "Box" };


extern int ascanf_SplitHere;
extern Window ascanf_window;
extern double *ascanf_ActiveWinWidth, *ascanf_ActiveWinHeight;
extern double *ascanf_memory;
extern double *ascanf_self_value, *ascanf_current_value, ascanf_log_zero_x, ascanf_log_zero_y;
extern int ascanf_exit, reset_ascanf_currentself_value,
	reset_ascanf_index_value, ascanf_arg_error, ascanf_arguments;
extern double *ascanf_Counter, *ascanf_counter, *ascanf_numPoints, *ascanf_setNumber, *ascanf_TotalSets;
extern char *TBARprogress_header;

int ReadData_commands;
Process ReadData_proc;
extern char *transform_description, transform_separator;
extern char *transform_x_buf;
extern char *transform_y_buf;
extern int transform_x_buf_len, transform_y_buf_len;
extern int transform_x_buf_allen, transform_y_buf_allen;
extern double *ReDo_DRAW_BEFORE, *Really_DRAW_AFTER;

int Determine_AvSlope= False, Determine_tr_curve_len= False;

int read_params_now= False;

#ifdef linux
#	include <asm/poll.h>
#elif !defined(__MACH__)
#	include <poll.h>
#endif

int StartUp= 1, MacStartUp= 0, ReadPipe= False;
FILE *ReadPipe_fp= NULL;
char *ReadPipe_name= NULL;

extern int TransformCompute( LocalWin *wi, Boolean warn_about_size );

extern double *disable_SET_PROCESS, *SET_PROCESS_last, *DataWin_before_Processing;

int local_buf_size= MAXBUFSIZE;

typedef struct ShiftUndoBuf{
	DataSet *set;
	UserLabel *ul;
	int idx, whole_set, sets;
	double x, y;
} ShiftUndoBuf;
ShiftUndoBuf ShiftUndo;
LocalWin *LabelClipUndo= NULL;

typedef struct SplitUndoBuf{
	DataSet *set;
	int idx;
	unsigned char split;
} SplitUndoBuf;
SplitUndoBuf SplitUndo;

typedef struct DiscardUndoBuf{
	DataSet *set;
	int idx;
	unsigned char split;
	UserLabel *head, ul;
	double lX, lY, hX, hY;
	LocalWin *wi;
} DiscardUndoBuf;
DiscardUndoBuf DiscardUndo;

extern XSegment *Xsegs, *lowYsegs, *highYsegs;		/* Point space for X */
extern XXSegment *XXsegs;
#define LYsegs	lowYsegs
#define HYsegs	highYsegs
extern XSegment_error *XsegsE;
extern long XsegsSize, XXsegsSize, YsegsSize, XsegsESize;

/* For reading in the data */
int setNumber = 0, fileNumber= 0;
char *AddFile= NULL, *ScriptFile= NULL;
LocalWin *ScriptFileWin= NULL;
int maxSize = 0;

/* our stderr	*/
extern FILE *StdErr;
int use_pager= 0;

FILE **PIPE_fileptr= NULL;

extern void PIPE_handler( int sig);
extern void handle_FPE();

extern char *parse_codes( char *T );

/* buffer to store comments in datafile:	*/
extern char *comment_buf;
extern int comment_size, NoComment;

extern char *add_comment( char *comment ), *time_stamp( FILE *fp, char *filename, char *buff, int verbose, char *pf), *today();

extern char *XGstrdup2( char *c , char *c2 );

/* Basic transformation stuff */

double llx, lly, llpx, llpy, llny, urx, ury; /* Bounding box of all data */
double real_min_x, real_max_x, real_min_y, real_max_y;

Window deleteWindow= (Window) 0;

#if !(defined(DEBUG) && defined(DEBUGSUPPORT))
#else
#undef SS_Add_Data_
extern void SS_Add_Data(SimpleStats *a, const int cnt, const double sm, const double wght);
#	define SS_Add_Data_(a,cnt,sm,wght)	SS_Add_Data(&(a), (int) cnt, (double) sm, (double) wght)
#endif

double scale_av_x= -1.0, scale_av_y= -1.0;
double _scale_av_x= 3.0, _scale_av_y= 3.0;
SimpleStats SS_X, SS_Y, SS_SY, SS_Points;
int show_stats= 0;
SimpleStats SS_x, SS_y, SS_e, SS_sy;
SimpleStats SS__x, SS__y, SS__e;

SimpleStats SS_mX, SS_mY, SS_mE, SS_mI, SS_mLY, SS_mHY, SS_mMO, SS_mSO, SS_mPoints;
SimpleAngleStats SAS_mO;

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
#define WinOrgX	win_geo._WinOrgX
#define WinOrgY	win_geo._WinOrgY
#define WinOppX	win_geo._WinOppX
#define WinOppY	win_geo._WinOppY
#define R_UsrOrgX	win_geo.R_UsrOrgX
#define R_UsrOrgY	win_geo.R_UsrOrgY
#define R_UsrOppX	win_geo.R_UsrOppX
#define R_UsrOppY	win_geo.R_UsrOppY
#define XUnitsPerPixel	win_geo._XUnitsPerPixel
#define YUnitsPerPixel	win_geo._YUnitsPerPixel
#define R_XUnitsPerPixel	win_geo.R_XUnitsPerPixel
#define R_YUnitsPerPixel	win_geo.R_YUnitsPerPixel

LocalWin startup_wi;

int dump_average_values= 0, DumpProcessed= 0;
int DumpBinary= False;
int DumpAsAscanf= False;

XContext win_context = (XContext) 0;
XContext frame_context = (XContext) 0;

/* Other globally set defaults */

Display *disp=0;		/* Open display            */
Atom wm_delete_window;
Visual *vis;			/* Standard visual         */
Colormap cmap;			/* Standard colormap       */
int use_RootWindow, IconicStart= 0;
int screen;			/* Screen number           */
int search_vislist= 0;	/* find best screen	*/
int depth;			/* Depth of screen         */
int install_flag;		/* Install colormaps       */
Pixel black_pixel;		/* Actual black pixel      */
char *blackCName= NULL;
Pixel white_pixel;		/* Actual white pixel      */
char *whiteCName= NULL;
Pixel bgPixel= -1;		/* Background color        */
char *bgCName= NULL;
int bdrSize;			/* Width of border         */
Pixel bdrPixel;			/* Border color            */
char *bdrCName= NULL;
Pixel zeroPixel= -1;		/* Zero grid color         */
char *zeroCName= NULL;
Pixel axisPixel= -1;	/* Used for the axes	*/
char *axisCName= NULL;
Pixel gridPixel= -1;	/* Used a.o. for the grid, and the legendbox shadow	*/
char *gridCName= NULL;
Pixel highlightPixel= -1;
char *highlightCName= NULL;
double zeroWidth;			/* Width of zero line      */
char zeroLS[MAXLS];		/* Line style spec         */
int zeroLSLen= 0;			/* Length of zero LS spec  */
Pixel normPixel= -1, textPixel= -1;		/* Normal color         */
int use_textPixel= 0;
char *normCName= NULL;
double axisWidth, gridWidth;			/* Width of axis line      */
double errorWidth;			/* Width of errorbar line		*/
char gridLS[MAXLS]= "11";		/* Axis line style spec    */
int gridLSLen= 2;			/* Length of axis line style */
int XLabelLength= 10;		/* length in characters of axis labels	*/
int YLabelLength= 10;		/* length in characters of axis labels	*/

#define XLabelLength	wi->axis_stuff.XLabelLength
#define YLabelLength	wi->axis_stuff.YLabelLength

Pixel echoPix;			/* Echo pixel value        */
XGFontStruct dialogFont= { NULL, "dialogFont" };
XGFontStruct dialog_greekFont= {NULL, "dialog_greekFont"};
XGFontStruct axisFont= { NULL, "axisFont" };		/* Font for axis labels    */
XGFontStruct legendFont= { NULL, "legendFont"};
XGFontStruct labelFont= { NULL, "labelFont" };
XGFontStruct legend_greekFont= {NULL, "legend_greekFont"};
XGFontStruct label_greekFont= {NULL, "label_greekFont"};
XGFontStruct title_greekFont= {NULL, "title_greekFont"};
XGFontStruct axis_greekFont= {NULL, "axis_greekFont"};
XGFontStruct titleFont= {NULL, "titleFont"};		/* Font for title labels   */
XGFontStruct markFont= {NULL, "markFont"};
XGFontStruct cursorFont= {NULL, "cursorFont"};
XGFontStruct fbFont= {NULL, "fbFont"}, fb_greekFont= {NULL, "fb_greekFont"};
int use_markFont;
#ifdef DEBUG
int use_gsTextWidth= 1;
#else
int use_gsTextWidth= 1;
#endif
int auto_gsTextWidth= 1;
int use_X11Font_length= 1, _use_gsTextWidth, used_gsTextWidth, prev_used_gsTextWidth;
int scale_plot_area_x= 0, scale_plot_area_y= 0;
char titleText[MAXBUFSIZE+1]; 	/* Plot title              */
char *titleText2= NULL;
char titleText_0;
char XUnits[MAXBUFSIZE+1]= INIT_XUNITS;	/* X Unit string           */
char YUnits[MAXBUFSIZE+1]= INIT_YUNITS;	/* Y Unit string	   */
char tr_XUnits[MAXBUFSIZE+1];	/* X Unit string           */
char tr_YUnits[MAXBUFSIZE+1];	/* Y Unit string	   */
int XUnitsSet= 0, YUnitsSet= 0, titleTextSet= 0;
int _bwFlag;
int MonoChrome= 0;		/* don't use colours	*/
int print_immediate= 0;	/* post printdialog before drawing window	*/
extern double *do_gsTextWidth_Batch;
int settings_immediate= 0;
int size_window_print= 0;
int FitX= 0, FitY= 0, fit_after_draw= 0;
double fit_after_precision= 0.005;
int FitOnce= False;
int NewProcess_Rescales= False;
int Aspect= 0, XSymmetric= 0, YSymmetric= 0;
int htickFlag, vtickFlag;
int zeroFlag;			/* draw zero-lines?	*/
int noExpX= 0, noExpY= 0;	/* don't use engineering notation	*/
int axisFlag= 1;
int bbFlag;			/* Whether to draw bb      */
int noLines;			/* Don't draw lines        */
int SameAttrs= 0, ResetAttrs= 0;
int no_pens= 0;
int legend_type= 0, no_legend= 0, no_legend_box= 0, no_intensity_legend, legend_always_visible= 1;
int no_ulabels= 0;
double legend_ulx, legend_uly, xname_x, xname_y,
	yname_x, yname_y, intensity_legend_ulx, intensity_legend_uly;
int legend_placed= 0, xname_placed= 0, yname_placed= 0, yname_vertical= 0, intensity_legend_placed= 0;
int legend_trans= 0, xname_trans= 0, yname_trans= 0, intensity_legend_trans= 0;
int no_title= 0, AllTitles= False, DrawColouredSetTitles= False;
int AlwaysDrawHighlighted= False;
int plot_only_file= 0;
int *plot_only_set= NULL, plot_only_set_len= -1;
int *highlight_set= NULL, highlight_set_len= -1;
int *mark_set= NULL, mark_set_len= -1;
int markFlag;			/* Draw marks at points    */
int arrows= 0;
int overwrite_marks= 1;	/* draw lines first, then marks; or reverse	*/
int overwrite_legend= 0, overwrite_AxGrid= 0;
int pixelMarks;			/* Draw pixel markers      */
int UseRealXVal= 1,
	UseRealYVal= 1;
int logXFlag;			/* Logarithmic X axis      */
int logYFlag;			/* Logarithmic Y axis      */
char log_zero_sym_x[MAXBUFSIZE+1], log_zero_sym_y[MAXBUFSIZE+1];	/* symbol to indicate log_zero	*/
int lz_sym_x= 0, lz_sym_y= 0;	/* log_zero symbols set?	*/
int log_zero_x_mFlag= 0, log_zero_y_mFlag= 0;
double log_zero_x= 0.0, log_zero_y= 0.0;	/* substitute 0.0 for these values when using log axis	*/
double _log_zero_x= 0.0, _log_zero_y= 0.0;	/* transformed log_zero_[xy]	*/
double log10_zero_x= 0.0, log10_zero_y= 0.0;
int sqrtXFlag, sqrtYFlag;
double powXFlag= 0.5, powYFlag= 0.5;
double powAFlag= 1.0;
int use_xye_info= 0;
double data[ASCANF_DATA_COLUMNS];
int column[ASCANF_DATA_COLUMNS]= {0,1,2,3};	/* order of x, y, error columns in input	*/
int autoscale= 1;
int disconnect= 0, split_set= 0;
char *split_reason= NULL;

double _Xscale= 1.0,		/* global scale factors for x,y,dy	*/
	_Yscale= 1.0,
	_DYscale= 1.0,		/* reset for each new file read-in	*/
	MXscale= 1.0,
	MYscale= 1.0,
	MDYscale= 1.0;
double
	_MXscale= 1.0,
	_MYscale= 1.0,
	_MDYscale= 1.0;
double Xscale= 1.0,		/* scale factors for x,y,dy	*/
	Yscale= 1.0,
	DYscale= 1.0;		/* reset for each new set and file read-in	*/
double Xscale2= 1.0, Yscale2= 1.0,
	   XscaleR= 1.0;

int barFlag;			/* Draw bar graph          */
int triangleFlag= 0;	/* draws error "triangles"	*/
int error_regionFlag= 0;
int process_bounds= 1, _process_bounds= 1;
int transform_axes= 1;
int raw_display= 0, raw_display_init= 0;
int show_overlap= 0;
int Redo_Error_Expression= False;
extern double overlap(LocalWin*);
int polarFlag, polarLog= 0, absYFlag= 0, vectorFlag= 0;
int vectorType= 0;
double radix= 2* M_PI, radix_offset= 0, vectorLength= 1, vectorPars[MAX_VECPARS]= { 1.0/3.0, 5};
char radixVal[64]= "2PI", radix_offsetVal[64]= "0";
int ebarWidth_set= 0;
int barBase_set= 0, barWidth_set= 0, barType= 0, barType_set= 0;
double barBase, barWidth;	/* Base and width of bars  */
int use_errors= 1;		/* use error specifications?	*/
int use_average_error= 0;
int no_errors= 0;
double lineWidth;			/* Width of data lines     */
int linestyle, elinestyle, pixvalue, markstyle;
char *geoSpec;		/* Geometry specification  */
int MaxnumFiles= 0, numFiles = 0;		/* Number of input files   */
extern int IOImportedFiles;
int RemoveInputFiles= 0;
int file_splits= 0;		/* Number of split input files (catenations of multiple files)	*/
char **inFileNames= NULL, *InFiles= NULL, *InFilesTStamps; 	/* File names              */
extern char UPrintFileName[];
char *PrintFileName= NULL;
int PrintingWindow= 0;
int Print_Orientation= 1;	/* 1=landscape	*/
double newfile_incr_width= 0;
int filename_in_legend= 0, labels_in_legend= 0;
char *Odevice = NULL;		/* Output device   	   */
char *Odisp = NULL; 	/* Output disposition      */
char *OfileDev = "";		/* Output file or device   */
char *Oprintcommand= NULL;
int debugFlag = 0, debugLevel= 0, *dbF_cache= NULL;		/* Whether debugging is on */
double Xbias_thres=0.005, Ybias_thres= 0.005;

int QuietErrors= False, quiet_error_count= 100;
int SetIconName= True, UnmappedWindowTitle= True;

double Xincr_factor= 1.0, Yincr_factor= 1.0;
double ValCat_X_incr= 1.0, ValCat_Y_incr= 1.0;
int XLabels= 0, YLabels= 0;

/* Possible user specified bounding box */
int x_scale_start= 1, y_scale_start= 1;
int xp_scale_start= 1, yp_scale_start= 1, yn_scale_start= 1;
double MusrLX, MusrRX, usrLpX= 0.0, usrLpY= 0.0, usrHnY= 0.0, MusrLY, MusrRY;
int use_lx= 0, use_ly= 0;
int use_max_x= 0, use_max_y= 0;
int User_Coordinates= 0;

extern int Synchro_State;
extern void *X_Synchro();

/* Total number of active windows */
int Num_Windows = 0;
Window New_win;
char *Prog_Name= NULL;
int progname= 0;
char Window_Name[256];

/* switch that controls the gathering of subsequent equal x-values
 * on a per set basis when dumping a SpreadSheet. Might mess up the
 * order in a set, and take very long!
 */
int Sort_Sheet= 1;
extern int XG_SaveBounds;

#include "xfree.h"

void _xfree(void *x, char *file, int lineno )
{
	if( x ){
#ifdef DEBUG
	if( debugLevel ){
	  int i;
	  char *c= x;
		fprintf(StdErr,"%s,line %d: Freeing 0x%lx { ", file, lineno, x);
		for( i= 0; i< sizeof(int); i++ ){
			fprintf( StdErr, "%x ", c[i] );
		}
		fprintf( StdErr, "} \"" );
		for( i= 0; i< sizeof(int); i++ ){
			fprintf( StdErr, "%c", c[i] );
		}
		fputs( "\"\n", StdErr );
	}
#endif
		free(x);
// 		x=NULL;
	}
}

void _xfree_setitem(void *x, char *file, int lineno, DataSet *this_set )
{ DataSet *alt_set;
  int i, hits= 0;
	if( x ){
		  /* See if other sets share this set's buffer that is about to be deleted.	*/
		for( i= 0; i< setNumber; i++ ){
			alt_set= &AllSets[i];
			if( alt_set!= this_set ){
				if( (char*)x== alt_set->setName ){
					alt_set->setName= strdup(x);
					hits+= 1;
				}
				if( (char*)x== alt_set->appendName ){
					alt_set->appendName= strdup(x);
					hits+= 1;
				}
				if( (char*)x== alt_set->fileName ){
					alt_set->fileName= strdup(x);
					hits+= 1;
				}
				if( (char*)x== alt_set->XUnits ){
					alt_set->XUnits= strdup(x);
					hits+= 1;
				}
				if( (char*)x== alt_set->YUnits ){
					alt_set->YUnits= strdup(x);
					hits+= 1;
				}
				if( (char*)x== alt_set->titleText ){
					alt_set->titleText= strdup(x);
					hits+= 1;
				}
				if( (char*)x== alt_set->average_set ){
					alt_set->average_set= strdup(x);
					hits+= 1;
				}
			}
		}
		_xfree( x, file, lineno);
#ifdef DEBUG
		if( hits ){
			fprintf( StdErr, "\treallocated %d links to this buffer\n", hits );
		}
#endif
	}
}

/*
 * Marker bitmaps
 */

#include "dot.11"

#include "mark1.12"
#include "mark2.12"
#include "mark3.12"
#include "mark4.12"
#include "mark5.12"
#include "mark6.12"
#include "mark7.12"
#include "mark8.12"
#include "mark9.12"
#include "mark10.12"
#include "mark11.12"
#include "mark12.12"
#include "mark13.12"
#include "mark14.12"
#include "mark15.12"
#include "mark16.12"

/* Sizes exported for marker drawing */
unsigned int dot_w = dot_width;
unsigned int dot_h = dot_height;
unsigned int mark_w = mark1_width;
unsigned int mark_h = mark1_height;
int mark_cx = mark1_x_hot;
/* Contrary to what lint says,  these are used in xgX.c */
int mark_cy = mark1_y_hot;

Pixmap dotMap = (Pixmap) 0;

extern int XErrHandler();	/* Handles error messages */

char *_strcpy( char *d, char *s)
{
	if( d && s && *s ){
		return( strcpy( d, s) );
	}
	return( NULL );
}

extern int ps_xpos, ps_ypos;
extern double ps_scale, ps_l_offset, ps_b_offset;

extern char *next_include_file;
extern XGStringList *nextGEN_Startup_Exprs;
int exact_X_axis= True, exact_Y_axis= True, show_all_ValCat_I= False;
int ValCat_X_axis= False, ValCat_X_levels= 1, ValCat_Y_axis= False, ValCat_I_axis= False;
int ValCat_X_grid= 0;
CustomFont *VCat_XFont, *VCat_YFont, *VCat_IFont;
ValCategory *VCat_X= NULL, *VCat_Y= NULL, *VCat_I= NULL;

LabelsList *ColumnLabels= NULL;

  /* weights on width,height	*/
double bar_legend_dimension_weight[3]= {3, 1.5, 1};

int data_silent_process= False;

/* Update local or global settings. These are read into the global variables,
 \ upon opening a new window, local copies are bound to that window based
 \ either on the global variables (first window, some local stubs), or based
 \ on the parentwindow.
 */
LocalWin *CopyFlags( LocalWin *dest, LocalWin *src )
{ int i;
  char asep = ascanf_separator;
	if( src ){
		if( dest ){
		  /* CopyFlags( dest, src): copy from src to dest	*/
			dest->no_ulabels= src->no_ulabels;
			dest->no_legend= src->no_legend;
			dest->no_intensity_legend= src->no_intensity_legend;
			dest->legend_type= src->legend_type;
			dest->legend_always_visible= src->legend_always_visible;
			dest->no_legend_box= src->no_legend_box;
			dest->no_title= src->no_title;
			dest->no_pens= src->no_pens;
			dest->filename_in_legend= src->filename_in_legend;
			dest->labels_in_legend= src->labels_in_legend;
			dest->axisFlag= src->axisFlag;
			dest->bbFlag= src->bbFlag;
			dest->htickFlag= src->htickFlag;
			dest->vtickFlag= src->vtickFlag;
			dest->zeroFlag= src->zeroFlag;
			dest->use_errors= src->use_errors;
			dest->use_average_error= src->use_average_error;
			dest->triangleFlag= src->triangleFlag;
			dest->error_region= src->error_region;
			dest->polarFlag= src->polarFlag;
			dest->logXFlag= src->logXFlag;
			dest->logYFlag= src->logYFlag;
			dest->sqrtXFlag= src->sqrtXFlag;
			dest->sqrtYFlag= src->sqrtYFlag;
			dest->powXFlag= src->powXFlag;
			dest->powYFlag= src->powYFlag;
			dest->powAFlag= src->powAFlag;
			dest->absYFlag= src->absYFlag;
			dest->exact_X_axis= src->exact_X_axis;
			dest->exact_Y_axis= src->exact_Y_axis;
			dest->ValCat_X_axis= src->ValCat_X_axis;
			dest->ValCat_X_levels= src->ValCat_X_levels;
			dest->ValCat_X_grid= src->ValCat_X_grid;
			dest->ValCat_Y_axis= src->ValCat_Y_axis;
			dest->ValCat_I_axis= src->ValCat_I_axis;
			if( src->ValCat_XFont ){
				dest->ValCat_XFont= Init_CustomFont(
					src->ValCat_XFont->XFont.name, src->ValCat_XFont->alt_XFontName,
					src->ValCat_XFont->PSFont, src->ValCat_XFont->PSPointSize, src->ValCat_XFont->PSreencode
				);
			}
			if( src->ValCat_X ){
			  ValCategory *vcat= src->ValCat_X;
			  int N= 0;
				dest->ValCat_X= Free_ValCat( dest->ValCat_X );
				while( vcat ){
					if( (dest->ValCat_X= Add_ValCat( dest->ValCat_X, &N, vcat->val, vcat->category )) ){
						if( vcat->min!= vcat->max ){
							vcat++;
						}
						else{
							vcat= NULL;
						}
					}
					else{
						vcat= NULL;
					}
				}
			}
			if( src->ValCat_YFont ){
				dest->ValCat_YFont= Init_CustomFont(
					src->ValCat_YFont->XFont.name, src->ValCat_YFont->alt_XFontName,
					src->ValCat_YFont->PSFont, src->ValCat_YFont->PSPointSize, src->ValCat_YFont->PSreencode
				);
			}
			if( src->ValCat_Y ){
			  ValCategory *vcat= src->ValCat_Y;
			  int N= 0;
				dest->ValCat_Y= Free_ValCat( dest->ValCat_Y );
				while( vcat ){
					if( (dest->ValCat_Y= Add_ValCat( dest->ValCat_Y, &N, vcat->val, vcat->category )) ){
						if( vcat->min!= vcat->max ){
							vcat++;
						}
						else{
							vcat= NULL;
						}
					}
					else{
						vcat= NULL;
					}
				}
			}
			if( src->ValCat_IFont ){
				dest->ValCat_IFont= Init_CustomFont(
					src->ValCat_IFont->XFont.name, src->ValCat_IFont->alt_XFontName,
					src->ValCat_IFont->PSFont, src->ValCat_IFont->PSPointSize, src->ValCat_IFont->PSreencode
				);
			}
			if( src->ValCat_I ){
			  ValCategory *vcat= src->ValCat_I;
			  int N= 0;
				dest->ValCat_I= Free_ValCat( dest->ValCat_I );
				while( vcat ){
					if( (dest->ValCat_I= Add_ValCat( dest->ValCat_I, &N, vcat->val, vcat->category )) ){
						if( vcat->min!= vcat->max ){
							vcat++;
						}
						else{
							vcat= NULL;
						}
					}
					else{
						vcat= NULL;
					}
				}
			}
			if( src->ColumnLabels ){
				dest->ColumnLabels= Free_LabelsList( dest->ColumnLabels );
				dest->ColumnLabels= Copy_LabelsList( dest->ColumnLabels, src->ColumnLabels );
			}
			dest->radix= src->radix;
			dest->radix_offset= src->radix_offset;
			dest->vectorFlag= src->vectorFlag;
			dest->Xbias_thres= src->Xbias_thres;
			dest->Ybias_thres= src->Ybias_thres;
			dest->_legend_ulx= src->_legend_ulx;
			dest->_legend_uly= src->_legend_uly;
			dest->xname_x= src->xname_x;
			dest->xname_y= src->xname_y;
			dest->yname_x= src->yname_x;
			dest->yname_y= src->yname_y;
			dest->legend_placed= src->legend_placed;
			dest->xname_placed= src->xname_placed;
			dest->yname_placed= src->yname_placed;
			dest->yname_vertical= src->yname_vertical;
			dest->legend_trans= src->legend_trans;
			dest->xname_trans= src->xname_trans;
			dest->yname_trans= src->yname_trans;
			   /* These are a number of fields that are actually only
			    \ interesting to copy from window to window.
				*/
			dest->IntensityLegend= src->IntensityLegend;
			strcpy( dest->XUnits, src->XUnits);
			strcpy( dest->YUnits, src->YUnits);
			strcpy( dest->tr_XUnits, src->tr_XUnits);
			strcpy( dest->tr_YUnits, src->tr_YUnits);
			dest->plot_only_group= src->plot_only_group;
			dest->plot_only_file= src->plot_only_file;
			dest->plot_only_set0= src->plot_only_set0;
			dest->plot_only_set_len= src->plot_only_set_len;
			dest->ctr_A= src->ctr_A;
			dest->AlwaysDrawHighlighted= src->AlwaysDrawHighlighted;
			for( i= 0; i< MaxSets; i++ ){
				dest->plot_only_set[i]= src->plot_only_set[i];
				dest->mark_set[i]= src->mark_set[i];
				dest->draw_set[i]= src->draw_set[i];
				dest->new_file[i]= src->new_file[i];
				dest->fileNumber[i]= src->fileNumber[i];
				dest->numVisible[i]= src->numVisible[i];
				dest->legend_line[i].low_y= src->legend_line[i].low_y;
				dest->legend_line[i].high_y= src->legend_line[i].high_y;
				dest->legend_line[i].highlight= src->legend_line[i].highlight;
				if( (dest->legend_line[i].pixvalue= src->legend_line[i].pixvalue)< 0 ){
					xfree( dest->legend_line[i].pixelCName );
					dest->legend_line[i].pixelCName= XGstrdup( src->legend_line[i].pixelCName );
					dest->legend_line[i].pixelValue= src->legend_line[i].pixelValue;
				}
				if( dest->xcol && src->xcol ){
					dest->xcol[i]= src->xcol[i];
					dest->ycol[i]= src->ycol[i];
					dest->ecol[i]= src->ecol[i];
					dest->lcol[i]= src->lcol[i];
				}
				dest->error_type[i]= src->error_type[i];
			}

			if( src->transform.separator ){
				ascanf_separator = src->transform.separator;
			}
			xfree( dest->transform.description );
			dest->transform.description= XGstrdup( src->transform.description );
			if( (dest->transform.x_len= src->transform.x_len) ){
				strcpalloc( &dest->transform.x_process, &dest->transform.x_allen, src->transform.x_process);
				new_transform_x_process( dest );
			}
			if( (dest->transform.y_len= src->transform.y_len) ){
				strcpalloc( &dest->transform.y_process, &dest->transform.y_allen, src->transform.y_process);
				new_transform_y_process( dest );
			}

			if( src->process.separator ){
				ascanf_separator = src->process.separator;
			}
			else{
				ascanf_separator = asep;
			}
			xfree( dest->process.description );
			dest->process.description= XGstrdup( src->process.description );
			if( (dest->process.data_init_len= src->process.data_init_len) ){
				strcpalloc( &dest->process.data_init, &dest->process.data_init_allen, src->process.data_init);
				new_process_data_init( dest );
			}
			if( (dest->process.data_before_len= src->process.data_before_len) ){
				strcpalloc( &dest->process.data_before, &dest->process.data_before_allen, src->process.data_before);
				new_process_data_before( dest );
			}
			if( (dest->process.data_process_len= src->process.data_process_len) ){
				strcpalloc( &dest->process.data_process, &dest->process.data_process_allen, src->process.data_process);
				new_process_data_process( dest );
			}
			if( (dest->process.data_after_len= src->process.data_after_len) ){
				strcpalloc( &dest->process.data_after, &dest->process.data_after_allen, src->process.data_after);
				new_process_data_after( dest );
			}
			if( (dest->process.data_finish_len= src->process.data_finish_len) ){
				strcpalloc( &dest->process.data_finish, &dest->process.data_finish_allen, src->process.data_finish);
				new_process_data_finish( dest );
			}
			if( (dest->process.draw_before_len= src->process.draw_before_len) ){
				strcpalloc( &dest->process.draw_before, &dest->process.draw_before_allen, src->process.draw_before);
				new_process_draw_before( dest );
			}
			if( (dest->process.draw_after_len= src->process.draw_after_len) ){
				strcpalloc( &dest->process.draw_after, &dest->process.draw_after_allen, src->process.draw_after);
				new_process_draw_after( dest );
			}
			if( (dest->process.dump_before_len= src->process.dump_before_len) ){
				strcpalloc( &dest->process.dump_before, &dest->process.dump_before_allen, src->process.dump_before);
				new_process_dump_before( dest );
			}
			if( (dest->process.dump_after_len= src->process.dump_after_len) ){
				strcpalloc( &dest->process.dump_after, &dest->process.dump_after_allen, src->process.dump_after);
				new_process_dump_after( dest );
			}
			if( (dest->process.enter_raw_after_len= src->process.enter_raw_after_len) ){
				strcpalloc( &dest->process.enter_raw_after, &dest->process.enter_raw_after_allen, src->process.enter_raw_after);
				new_process_enter_raw_after( dest );
			}
			if( (dest->process.leave_raw_after_len= src->process.leave_raw_after_len) ){
				strcpalloc( &dest->process.leave_raw_after, &dest->process.leave_raw_after_allen, src->process.leave_raw_after);
				new_process_leave_raw_after( dest );
			}

			if( src->curs_cross.fromwin_process.separator ){
				ascanf_separator = src->curs_cross.fromwin_process.separator;
			}
			else{
				ascanf_separator = asep;
			}
			if( (dest->curs_cross.fromwin_process.process_len= src->curs_cross.fromwin_process.process_len) ){
			  LocalWin *aw= ActiveWin;
				dest->curs_cross.fromwin= ActiveWin= src->curs_cross.fromwin;
				ascanf_window= ActiveWin->window;
				strcpalloc( &dest->curs_cross.fromwin_process.process, &dest->curs_cross.fromwin_process.process_allen, src->curs_cross.fromwin_process.process);
				new_process_Cross_fromwin_process( dest );
				ActiveWin= aw;
			}

			ascanf_separator = asep;

			dest->process_bounds= src->process_bounds;
			RawDisplay( dest, src->raw_display );
			dest->transform_axes= src->transform_axes;
			dest->show_overlap= src->show_overlap;
			dest->overwrite_legend= src->overwrite_legend;
			dest->overwrite_AxGrid= src->overwrite_AxGrid;
			dest->Xincr_factor= src->Xincr_factor;
			dest->Yincr_factor= src->Yincr_factor;
			dest->ValCat_X_incr= src->ValCat_X_incr;
			dest->ValCat_Y_incr= src->ValCat_Y_incr;
			dest->print_orientation= src->print_orientation;
			dest->ps_xpos= src->ps_xpos;
			dest->ps_ypos= src->ps_ypos;
			dest->ps_scale= src->ps_scale;
			dest->ps_l_offset= src->ps_l_offset;
			dest->ps_b_offset= src->ps_b_offset;
			dest->dump_average_values= src->dump_average_values;
			dest->DumpProcessed= src->DumpProcessed;
			dest->DumpBinary= src->DumpBinary;
			dest->DumpAsAscanf= src->DumpAsAscanf;
			strcpy( dest->log_zero_sym_x, src->log_zero_sym_x );
			strcpy( dest->log_zero_sym_y, src->log_zero_sym_y );
			dest->lz_sym_x= src->lz_sym_x;
			dest->lz_sym_y= src->lz_sym_y;
			dest->log10_zero_x= src->log10_zero_x;
			dest->log10_zero_y= src->log10_zero_y;
			dest->log_zero_x= src->log_zero_x;
			dest->log_zero_y= src->log_zero_y;
			dest->log_zero_x_mFlag= src->log_zero_x_mFlag;
			dest->log_zero_y_mFlag= src->log_zero_y_mFlag;
			dest->fit_xbounds= src->fit_xbounds;
			dest->fit_ybounds= src->fit_ybounds;
			dest->fit_after_draw= src->fit_after_draw;
			dest->fit_after_precision= src->fit_after_precision;
			dest->aspect= ABS(src->aspect);
			dest->x_symmetric= ABS(src->x_symmetric);
			dest->y_symmetric= ABS(src->y_symmetric);
			dest->aspect_ratio= src->aspect_ratio;
			if( dest->hard_devices && src->hard_devices ){
				Copy_Hard_Devices( dest->hard_devices, src->hard_devices );
			}
			if( src->version_list ){
				xfree( dest->version_list );
				dest->version_list= strdup( src->version_list );
			}
			if( src->next_include_file ){
				xfree( dest->next_include_file );
				dest->next_include_file= strdup( src->next_include_file );
			}
			if( src->next_startup_exprs ){
			  XGStringList *new, *last= NULL, *ex;
				while( dest->next_startup_exprs ){
					ex= dest->next_startup_exprs;
					xfree( ex->text );
					dest->next_startup_exprs= ex->next;
					xfree( ex );
				}
				ex= src->next_startup_exprs;
				while( ex ){
					if( (new= (XGStringList*) calloc( 1, sizeof(XGStringList) )) ){
						new->text= XGstrdup(ex->text);
						new->separator = ex->separator;
						new->next= NULL;
						if( last ){
							last->next= new;
							last= new;
						}
						else{
							last= dest->next_startup_exprs= new;
						}
					}
					ex= ex->next;
				}
			}
			if( src->init_exprs ){
			  XGStringList *new, *last= NULL, *ex;
				while( dest->init_exprs ){
					ex= dest->init_exprs;
					xfree( ex->text );
					dest->init_exprs= ex->next;
					xfree( ex );
				}
				ex= src->init_exprs;
				while( ex ){
					if( (new= (XGStringList*) calloc( 1, sizeof(XGStringList) )) ){
						new->text= XGstrdup(ex->text);
						new->separator = ex->separator;
						new->next= NULL;
						if( last ){
							last->next= new;
							last= new;
						}
						else{
							last= dest->init_exprs= new;
						}
					}
					ex= ex->next;
				}
				dest->new_init_exprs= True;
			}
			memcpy( dest->bar_legend_dimension_weight, src->bar_legend_dimension_weight, sizeof(bar_legend_dimension_weight) );
			dest->data_silent_process= src->data_silent_process;
		}
		else{
		  /* CopyFlags( NULL, src): copy from src to globals	*/
			no_ulabels= src->no_ulabels;
			no_legend= src->no_legend;
			no_intensity_legend= src->no_intensity_legend;
			legend_always_visible= src->legend_always_visible;
			no_legend_box= src->no_legend_box;
			no_title= src->no_title;
			no_pens= src->no_pens;
			filename_in_legend= src->filename_in_legend;
			labels_in_legend= src->labels_in_legend;
			axisFlag= src->axisFlag;
			bbFlag= src->bbFlag;
			htickFlag= src->htickFlag;
			vtickFlag= src->vtickFlag;
			zeroFlag= src->zeroFlag;
			use_errors= src->use_errors;
			triangleFlag= src->triangleFlag;
			error_regionFlag= src->error_region;
			polarFlag= src->polarFlag;
			logXFlag= src->logXFlag;
			logYFlag= src->logYFlag;
			sqrtXFlag= src->sqrtXFlag;
			sqrtYFlag= src->sqrtYFlag;
			powXFlag= src->powXFlag;
			powYFlag= src->powYFlag;
			powAFlag= src->powAFlag;
			absYFlag= src->absYFlag;
			exact_X_axis= src->exact_X_axis;
			exact_Y_axis= src->exact_Y_axis;
			ValCat_X_axis= src->ValCat_X_axis;
			ValCat_X_levels= src->ValCat_X_levels;
			ValCat_X_grid= src->ValCat_X_grid;
			ValCat_Y_axis= src->ValCat_Y_axis;
			show_all_ValCat_I= src->show_all_ValCat_I;
			ValCat_I_axis= src->ValCat_I_axis;
			radix= src->radix;
			radix_offset= src->radix_offset;
			vectorFlag= src->vectorFlag;
			Xbias_thres= src->Xbias_thres;
			Ybias_thres= src->Ybias_thres;
			legend_ulx= src->_legend_ulx;
			legend_uly= src->_legend_uly;
			xname_x= src->xname_x;
			xname_y= src->xname_y;
			yname_x= src->yname_x;
			yname_y= src->yname_y;
			legend_placed= src->legend_placed;
			xname_placed= src->xname_placed;
			yname_placed= src->yname_placed;
			yname_vertical= src->yname_vertical;
			legend_trans= src->legend_trans;
			xname_trans= src->xname_trans;
			yname_trans= src->yname_trans;
				  /* 980828	*/
				plot_only_set_len= src->plot_only_set_len;
				Xscale2= src->Xscale;
				Xscale2= src->_Xscale;
				Yscale2= src->Yscale;
				for( i= 0; src->plot_only_set && plot_only_set && i< MaxSets && i< plot_only_set_len; i++ ){
					plot_only_set[i]= src->plot_only_set[i];
				}
				for( i= 0; i< MaxSets; i++ ){
					if( src->xcol ){
						if( src->xcol[i]>= 0 ){
							AllSets[i].xcol= src->xcol[i];
						}
						if( src->ycol[i]>= 0 ){
							AllSets[i].ycol= src->ycol[i];
						}
						if( src->ecol[i]>= 0 ){
							AllSets[i].ecol= src->ecol[i];
						}
						if( src->lcol[i]>= 0 ){
							AllSets[i].lcol= src->lcol[i];
						}
					}
				}
			Xincr_factor= src->Xincr_factor;
			Yincr_factor= src->Yincr_factor;
			ValCat_X_incr= src->ValCat_X_incr;
			ValCat_Y_incr= src->ValCat_Y_incr;
			  /* added 951029	*/
			process_bounds= src->process_bounds;
			raw_display= src->raw_display;
			transform_axes= src->transform_axes;
			show_overlap= src->show_overlap;
			overwrite_legend= src->overwrite_legend;
			overwrite_AxGrid= src->overwrite_AxGrid;
				  /* 980828	*/
				_strcpy( XUnits, src->XUnits);
				_strcpy( YUnits, src->YUnits);
				_strcpy( tr_XUnits, src->tr_XUnits);
				_strcpy( tr_YUnits, src->tr_YUnits);
				Print_Orientation= src->print_orientation;
				ps_xpos= src->ps_xpos;
				ps_ypos= src->ps_ypos;
				ps_scale= src->ps_scale;
				ps_l_offset= src->ps_l_offset;
				ps_b_offset= src->ps_b_offset;
				dump_average_values= src->dump_average_values;
				DumpProcessed= src->DumpProcessed;
				DumpBinary= src->DumpBinary;
				DumpAsAscanf= src->DumpAsAscanf;
				strcpy( log_zero_sym_x, src->log_zero_sym_x );
				strcpy( log_zero_sym_y, src->log_zero_sym_y );
				lz_sym_x= src->lz_sym_x;
				lz_sym_y= src->lz_sym_y;
				log10_zero_x= src->log10_zero_x;
				log10_zero_y= src->log10_zero_y;
				log_zero_x= src->log_zero_x;
				log_zero_y= src->log_zero_y;
				log_zero_x_mFlag= src->log_zero_x_mFlag;
				log_zero_y_mFlag= src->log_zero_y_mFlag;
				FitX= src->fit_xbounds;
				FitY= src->fit_ybounds;
				fit_after_draw= src->fit_after_draw;
				fit_after_precision= src->fit_after_precision;
				FitOnce= src->FitOnce;
				Aspect= ABS(src->aspect);
				XSymmetric= ABS(src->x_symmetric);
				YSymmetric= ABS(src->y_symmetric);
				win_aspect= src->aspect_ratio;
				AlwaysDrawHighlighted= src->AlwaysDrawHighlighted;
				memcpy( bar_legend_dimension_weight, src->bar_legend_dimension_weight, sizeof(bar_legend_dimension_weight) );
				data_silent_process= src->data_silent_process;
		}
	}
	else if( dest ){
	  /* CopyFlags( dest, NULL): copy from globals to dest
	   \ If dest is an uninitialised, local temporary LocalWin, be
	   \ sure to memset(0) it.
	   */
		dest->no_ulabels= no_ulabels;
		dest->no_legend= no_legend;
		dest->no_intensity_legend= no_intensity_legend;
		dest->legend_always_visible= legend_always_visible;
		dest->legend_type= legend_type;
		dest->no_legend_box= no_legend_box;
		dest->no_title= no_title;
		dest->no_pens= no_pens;
		dest->filename_in_legend= filename_in_legend;
		dest->labels_in_legend= labels_in_legend;
		dest->axisFlag= axisFlag;
		dest->bbFlag= bbFlag;
		dest->htickFlag= htickFlag;
		dest->vtickFlag= vtickFlag;
		dest->zeroFlag= zeroFlag;
		dest->use_errors= use_errors;
		dest->triangleFlag= triangleFlag;
		dest->error_region= error_regionFlag;
		dest->polarFlag= polarFlag;
		dest->logXFlag= logXFlag;
		dest->logYFlag= logYFlag;
		dest->sqrtXFlag= sqrtXFlag;
		dest->sqrtYFlag= sqrtYFlag;
		dest->powXFlag= powXFlag;
		dest->powYFlag= powYFlag;
		dest->powAFlag= powAFlag;
		dest->absYFlag= absYFlag;
		dest->exact_X_axis= exact_X_axis;
		dest->exact_Y_axis= exact_Y_axis;
		dest->ValCat_X_axis= ValCat_X_axis;
		dest->ValCat_X_levels= ValCat_X_levels;
		dest->ValCat_X_grid= ValCat_X_grid;
		dest->ValCat_Y_axis= ValCat_Y_axis;
		dest->show_all_ValCat_I= show_all_ValCat_I;
		dest->ValCat_I_axis= ValCat_I_axis;
		dest->radix= radix;
		dest->radix_offset= radix_offset;
		dest->vectorFlag= vectorFlag;
		dest->Xbias_thres= Xbias_thres;
		dest->Ybias_thres= Ybias_thres;
		dest->_legend_ulx= legend_ulx;
		dest->_legend_uly= legend_uly;
		dest->xname_x= xname_x;
		dest->xname_y= xname_y;
		dest->yname_x= yname_x;
		dest->yname_y= yname_y;
		dest->legend_placed= legend_placed;
		dest->xname_placed= xname_placed;
		dest->yname_placed= yname_placed;
		dest->yname_vertical= yname_vertical;
		dest->legend_trans= legend_trans;
		dest->xname_trans= xname_trans;
		dest->yname_trans= yname_trans;

		dest->plot_only_set_len= plot_only_set_len;
		dest->Xscale= ABS(Xscale2);
		dest->_Xscale= Xscale2;
		dest->Yscale= Yscale2;
		for( i= 0; dest->plot_only_set && plot_only_set && i< MaxSets && i< plot_only_set_len; i++ ){
			dest->plot_only_set[i]= plot_only_set[i];
		}
		for( i= 0; i< MaxSets; i++ ){
			if( dest->xcol ){
				dest->xcol[i]= AllSets[i].xcol;
				dest->ycol[i]= AllSets[i].ycol;
				dest->ecol[i]= AllSets[i].ecol;
				dest->lcol[i]= AllSets[i].lcol;
			}
		}
		for( i= 0; dest->error_type && i< MaxSets; i++ ){
			  /* The only logical default:	*/
			  /* 990615: don't touch window error_types anymore..	*/
			if( 1 || AllSets[i].error_type!= -1 ){
/* 				dest->error_type[i]= AllSets[i].error_type;	*/
			}
			else{
				if( vectorFlag ){
					dest->error_type[i]= 4;
				}
				else if( error_regionFlag ){
					dest->error_type[i]= 3;
				}
				else if( triangleFlag ){
					dest->error_type[i]= 2;
				}
				else if( !no_errors ){
					dest->error_type[i]= 1;
				}
				else{
					dest->error_type[i]= 0;
				}
			}
		}
		dest->Xincr_factor= Xincr_factor;
		dest->Yincr_factor= Yincr_factor;
		dest->ValCat_X_incr= ValCat_X_incr;
		dest->ValCat_Y_incr= ValCat_Y_incr;
			/* added 951029	*/
		dest->process_bounds= process_bounds;
		RawDisplay( dest, raw_display );
		dest->transform_axes= transform_axes;
		dest->show_overlap= show_overlap;
		dest->overwrite_legend= overwrite_legend;
		dest->overwrite_AxGrid= overwrite_AxGrid;
		for( i= 0; dest->numVisible && i< MaxSets; i++ ){
			  /* The only logical default:	*/
			dest->numVisible[i]= AllSets[i].numPoints;
		}
		_strcpy( dest->XUnits, XUnits);
		_strcpy( dest->YUnits, YUnits);
/* 		_strcpy( dest->tr_XUnits, (*tr_XUnits)? tr_XUnits : XUnits );	*/
/* 		_strcpy( dest->tr_YUnits, (*tr_YUnits)? tr_YUnits : YUnits );	*/
		_strcpy( dest->tr_XUnits, tr_XUnits );
		_strcpy( dest->tr_YUnits, tr_YUnits );
		dest->print_orientation= Print_Orientation;
		dest->ps_xpos= ps_xpos;
		dest->ps_ypos= ps_ypos;
		dest->ps_scale= ps_scale;
		dest->ps_l_offset= ps_l_offset;
		dest->ps_b_offset= ps_b_offset;
		dest->dump_average_values= dump_average_values;
		dest->DumpProcessed= DumpProcessed;
		dest->DumpBinary= DumpBinary;
		dest->DumpAsAscanf= DumpAsAscanf;
		strcpy( dest->log_zero_sym_x, log_zero_sym_x );
		strcpy( dest->log_zero_sym_y, log_zero_sym_y );
		dest->lz_sym_x= lz_sym_x;
		dest->lz_sym_y= lz_sym_y;
		dest->log10_zero_x= log10_zero_x;
		dest->log10_zero_y= log10_zero_y;
		dest->log_zero_x= log_zero_x;
		dest->log_zero_y= log_zero_y;
		dest->log_zero_x_mFlag= log_zero_x_mFlag;
		dest->log_zero_y_mFlag= log_zero_y_mFlag;
		dest->fit_xbounds= FitX;
		dest->fit_ybounds= FitY;
		dest->fit_after_draw= fit_after_draw;
		dest->fit_after_precision= fit_after_precision;
		dest->FitOnce= FitOnce;
		if( dest->aspect && !Aspect ){
			dest->win_geo.bounds = dest->win_geo.aspect_base_bounds;
		}
		dest->aspect= ABS(Aspect);
		dest->x_symmetric= ABS(XSymmetric);
		dest->y_symmetric= ABS(YSymmetric);
		dest->aspect_ratio= win_aspect;
		dest->AlwaysDrawHighlighted= AlwaysDrawHighlighted;
		if( dest->hard_devices && (!dest->hard_devices[0].dev_name || !dest->hard_devices[0].dev_name[0]) ){
			Copy_Hard_Devices( dest->hard_devices, &hard_devices[0] );
		}
		memcpy( dest->bar_legend_dimension_weight, bar_legend_dimension_weight, sizeof(bar_legend_dimension_weight) );
		dest->data_silent_process= data_silent_process;
	}
	return( dest );
}

LocalWin *check_wi( LocalWin **wi, char *caller )
{ static LocalWin lwi;
  static char called= 0, *wid= NULL;
	if( !called ){
		wid= cgetenv("WINDOWID");
		called= 1;
	}
	if( !wi || !*wi ){
		if( debugLevel>= 1 ){
			fprintf( StdErr, "check_wi(NULL,%s)\n", caller );
		}
		memset( &lwi, 0, sizeof(lwi) );
		if( wid ){
			lwi.window= atoi( wid);
		}
		else{
			lwi.window= RootWindow(disp, screen);
		}
		return( (*wi= CopyFlags( &lwi, NULL) ) );
	}
	else{
		return( *wi );
	}
}

Cursor zoomCursor, labelCursor, cutCursor, filterCursor;
int nbytes, maxitems, NumObs= 2;

extern double Gonio_Base_Value, Gonio_Base_Value_2, Gonio_Base_Value_4;
extern double Units_per_Radian, Gonio_Base_Offset;

#define __DLINE__	(double)__LINE__

extern double Gonio_Base( LocalWin *wi, double base, double offset);

#define Gonio(fun,x)	(fun(((x)+Gonio_Base_Offset)/Units_per_Radian))
#define InvGonio(fun,x)	((fun(x)*Units_per_Radian-Gonio_Base_Offset))
#define Sin(x) Gonio(sin,x)
#define Cos(x) Gonio(cos,x)
#define Tan(x) Gonio(tan,x)
#define ArcSin(x) InvGonio(asin,x)
#define ArcCos(x) InvGonio(acos,x)
#define ArcTan(wi,x,y) (_atan3(wi,x,y)*Units_per_Radian-Gonio_Base_Offset)

extern char *matherr_mark();

extern double cus_pow();

#define MATHERR_MARK()	matherr_mark(__FILE__ ":" STRING(__LINE__))

#ifndef degrees
#	define degrees(a)			((a)*57.295779512)
#endif
#ifndef radians
#	define radians(a)			((a)/57.295779512)
#endif


extern double atan3( double y, double x);
/* return arg(x,y) in [0,M_2PI]	*/
extern double _atan3( LocalWin *wi, double x, double y);

int _int_MAX( double a, double b)
{
	a+= 0.5;
	b+= 0.5;
	return( MAX( (int)a, (int)b ) );
}

/* determines if the X-bounds, interpreted as
 \ an angle in a polar plot, fall within the
 \ bounds set by l and h
 \ Angles are supposed to run from 0 to radix ([0,1])
 \ 981120: radix_offset shouldn't change any to this!
 */
int INSIDE(LocalWin *wi, double _loX, double _hiX, double l, double h)
{  double lx= fmod(_loX, wi->radix)/ wi->radix,
		hx= fmod(_hiX, wi->radix)/ wi->radix;
	if( !lx && _loX!= 0 ){
		lx= 1;
	}
	else if( lx< 0 ){
		lx= 1+ lx;
	}
	if( !hx && _hiX!= 0 ){
		hx= 1;
	}
	else if( hx< 0 ){
		hx= 1+ hx;
	}
	if( h> l ){
		return( lx>=l && lx<=h && hx>=l && hx<=h && hx>= lx );
	}
	else{
	  /* e.g. INSIDE(wi, 0.75, 0.25):
	   \ lx must be in [0.75,1] and hx within [0,0.25]
	   */
		return( (lx>=l && lx<= 1) && (hx>=0 && hx<=h) && hx<= lx );
	}
}

/* #define WINSIDE(wi,l,h)	INSIDE(wi,(wi)->loX,(wi)->hiX,(l),(h))	*/
int WINSIDE( LocalWin *wi, double l, double h)
{  int inside= INSIDE( wi, wi->loX, wi->hiX, l, h);
	if( inside ){
		wi->win_geo.low_angle= l* wi->radix;
		wi->win_geo.high_angle= h* wi->radix;
	}
	return( inside );
}

#ifndef MININT
#	define MININT (1 << (8*sizeof(int)-1))
#endif
#ifndef MAXINT
#	define MAXINT (-1 ^ (MININT))
#endif

#ifdef REAL_SCREENXY
  /* This should be the real code for SCREENX and SCREENY. It always returns
   \ a valid integer. It breaks on Inf (DBL_MAX) however
   */
int SCREENX(LocalWin *ws, double userX)
{  double screenX;
   int ret;
	if( !ws ){
		return(0);
	}
	CLIP_EXPR( ret, ws->XOrgX+ (screenX= (userX - ws->UsrOrgX)/ws->XUnitsPerPixel + 0.5), MININT, MAXINT);
	return( ret );
}

int SCREENY(LocalWin *ws, double userY)
{  double screenY;
   int ret;
	if( !ws ){
		return(0);
	}
	CLIP_EXPR( ret, ws->XOppY- (screenY= (userY - ws->UsrOrgY)/ws->YUnitsPerPixel + 0.5), MININT, MAXINT );
	return( ret );
}
#else
  /* Make do with a faulty SCREENX and SCREENY. SCREENX returns a valid integer + ws->XOrgX ; SCREENY
   \ ws->XOppY - a valid integer .
   */
int SCREENX(LocalWin *ws, double userX)
{  double screenX;
	if( !ws ){
		return(0);
	}
	if( ws->XUnitsPerPixel ){
		CLIP_EXPR( screenX, (userX - ws->UsrOrgX)/ws->XUnitsPerPixel + 0.5, MININT, MAXINT);
	}
	else{
	  /* division by 0 is not a good idea if casting to int is performed afterwards..
	   \ especially not if one gets 0/0 (which doesn't result in +- Inf!!), but a NaN.
	   \ (In principal, +- Inf should correctly result in MAXINT or MININT by the above
	   \ formula).
	   */
		screenX= MAXINT;
	}
 	return( (int) (screenX) + ws->XOrgX );
}

int SCREENY(LocalWin *ws, double userY)
{  double screenY;
	if( !ws ){
		return(0);
	}
	if( ws->YUnitsPerPixel ){
		CLIP_EXPR( screenY, (userY - ws->UsrOrgY)/ws->YUnitsPerPixel + 0.5, MININT, MAXINT );
	}
	else{
		screenY= MAXINT;
	}
 	return( ws->XOppY - (int)(screenY) );
}
#endif

int SCREENXDIM(LocalWin *ws, double userX)
{  double screenX;
	if( !ws ){
		return(0);
	}
	if( ws->XUnitsPerPixel ){
		CLIP_EXPR( screenX, (userX)/ws->XUnitsPerPixel + 0.5, MININT, MAXINT);
	}
	else{
		screenX= MAXINT;
	}
	return( (int) screenX );
}

int SCREENYDIM(LocalWin *ws, double userY)
{  double screenY;
	if( !ws ){
		return(0);
	}
	if( ws->YUnitsPerPixel ){
		CLIP_EXPR( screenY, (userY)/ws->YUnitsPerPixel + 0.5, MININT, MAXINT );
	}
	else{
		screenY= MAXINT;
	}
	return( (int) screenY );
}

double cus_sqrt( double x)
{  double y, sqrt();
	if( x< 0){
		y= -1.0 * sqrt( -1.0 * x );
		if( debugFlag && debugLevel> 1 )
			fprintf( StdErr, "cus_sqrt(%g)=%g\n", x, y);
	}
	else
		y= sqrt(x);
	return( y);
}

double cus_pow( double x, double p)
{  double y, pow();
	if( x< 0){
		MATHERR_MARK();
		y= -1.0 * pow( -1.0 * x, p );
	}
	else if( DBL_EPSILON> ABS(x) ){
		y= 0.0;
	}
	else{
		MATHERR_MARK();
		y= pow(x,p);
	}
	if( debugFlag && debugLevel> 1 ){
		fprintf( StdErr, "cus_pow(%g,%g)=%g\n", x, p, y);
		fflush( StdErr );
	}
	return( y);
}

/* Return x to the power 1/powXFlag	*/
double cus_powX( LocalWin *wi, double x)
{
	check_wi( &wi, "cus_powX" );
	if( wi->sqrtXFlag> 0 && !wi->powXFlag ){
		wi->powXFlag= 0.5;
	}
	return( cus_pow( x* wi->Xscale, 1/wi->powXFlag) );
}

double cus_powY( LocalWin *wi, double x)
{
	check_wi( &wi, "cus_powY" );
	if( wi->sqrtYFlag> 0 && !wi->powYFlag ){
		wi->powYFlag= 0.5;
	}
	return( cus_pow( x* wi->Yscale, 1/wi->powYFlag) );
}

#ifdef i386
double myLog10(double x)
{ double y;

	asm("fldlg2     \n\t"	//Push log10(2) onto fp stack
		"fldl %[x]  \n\t"	//Push x onto fp stack
		"fyl2x      \n\t"	//Compute s(1)*log2(s(0)) = log10(2)*log2(x).  Result in s(0)
		:     "=t" (y)		//gcc stuff: output is to be "y" and will be on top of fp stack (st(0))
		: [x] "m"  (x)		//gcc stuff: input is to be "x", is in memory, and to be called "x" in asm
	);
	return y;
}
#endif

double wilog10( LocalWin *wi, double x )
{
	return( log10(x) );
}

double cus_log10(double x)
{
// 	return( (x<0)? - log10( -x ) : log10(x) );
	if( x < 0 ){
		return - log10( -x );
	}
	else{
		return log10(x);
	}
}

/* take the 10log from argument x. If the sqrtFlag is set,
 \ the powXFlag'th power of the result is returned
 */
double cus_log10X(LocalWin *wi, double x)
{
	check_wi( &wi, "cus_log10X" );
	if( wi->sqrtXFlag> 0 && !wi->powXFlag ){
		wi->powXFlag= 0.5;
	}
	if( wi->sqrtXFlag && wi->sqrtXFlag!= -1)
		return( cus_pow( cus_log10(x* wi->Xscale), wi->powXFlag ) );
	else
		return( cus_log10(x* wi->Xscale) );
}

double cus_log10Y( LocalWin *wi, double y)
{
	check_wi( &wi, "cus_log10Y" );
	if( wi->sqrtYFlag> 0 && !wi->powYFlag ){
		wi->powYFlag= 0.5;
	}
	if( wi->sqrtYFlag && wi->sqrtYFlag!= -1)
		return( cus_pow( cus_log10(y* wi->Yscale), wi->powYFlag ) );
	else
		return( cus_log10(y* wi->Yscale) );
}

/* To get around an inaccurate log */
double nlog10X( LocalWin *wi, double x)
{  double xs= wi->Xscale, answer;
	check_wi( &wi, "nlog10X" );
	wi->Xscale= 1.0;
	answer= (x == 0.0 ? 0.0 : cus_log10X(wi, x) + 1e-15);
	wi->Xscale= xs;
	return( answer );
}

double nlog10Y( LocalWin *wi, double x)
{  double ys= wi->Yscale, answer;
	check_wi( &wi, "nlog10Y" );
	wi->Yscale= 1.0;
	answer= (x == 0.0 ? 0.0 : cus_log10Y(wi, x) + 1e-15);
	wi->Yscale= ys;
	return( answer );
}

double cus_sqr( double x)
{
	return( (x<0)? -1.0 * x * x : x * x );
}

/* Return 10 to the power (x to the power 1/powXFlag) or
 \ 10 to the power x (if sqrtXFlag is not set)
 */
double cus_pow_y_pow_xX( LocalWin *wi,  double y, double x)
{
	check_wi( &wi, "cus_pow_y_pow_xX" );
	if( wi->sqrtXFlag> 0 && !wi->powXFlag ){
		wi->powXFlag= 0.5;
	}
	if( wi->sqrtXFlag && wi->sqrtXFlag!= -1)
		return( pow( y, cus_powX(wi, x) ) );
	else
		return( pow(y, x) );
}

double cus_pow_y_pow_xY( LocalWin *wi, double y, double x)
{
	check_wi( &wi, "cus_pow_y_pow_xY" );
	if( wi->sqrtYFlag> 0 && !wi->powYFlag ){
		wi->powYFlag= 0.5;
	}
	if( wi->sqrtYFlag && wi->sqrtYFlag!= -1){
		return( pow( y, cus_powY(wi, x) ) );
	}
	else
		return( pow(y, x) );
}

/* Return the untransformed x belonging to the transformed
 \ coordinate pair (x,y)
 */
double Reform_X( LocalWin *wi, double x, double y)
{ static double value;
  double xs;
	check_wi( &wi, "Reform_X" );
	xs= wi->_Xscale;
	wi->Xscale= wi->_Xscale= 1;
	if( wi->polarFlag ){
		if( Gonio_Base_Value!= wi->radix || Gonio_Base_Offset!= wi->radix_offset ){
			Gonio_Base( wi, wi->radix, wi->radix_offset );
		}
		x= ArcTan(wi, x,y)/ XscaleR;
	}
	if( wi->sqrtXFlag> 0 && !wi->powXFlag ){
		wi->powXFlag= 0.5;
	}
	if( wi->logXFlag && wi->logXFlag!= -1 ){
		value=( cus_pow_y_pow_xX( wi, 10.0, x) );
	}
	else if( wi->sqrtXFlag> 0 && wi->sqrtYFlag!= -1 ){
		value=( cus_powX( wi, x ) );
	}
	else{
		value=( x);
	}
	wi->_Xscale= xs;
	wi->Xscale= ABS(xs);
	return( value );
}

double Reform_Y( LocalWin *wi, double y, double x)
{ static double value;
  double ys;
	check_wi( &wi, "Reform_Y" );
	ys= wi->Yscale;
	wi->Yscale= 1;
	if( wi->polarFlag ){
		if( Gonio_Base_Value!= wi->radix || Gonio_Base_Offset!= wi->radix_offset ){
			Gonio_Base( wi, wi->radix, wi->radix_offset );
		}
		if( wi->powAFlag!= 1 ){
		  double angle= ArcTan(wi, x,y), c, s;
			  /* This takes care of the power
			   \ of the gonio terms
			   */
			MATHERR_MARK();
			c= cus_pow( Cos(angle), wi->powAFlag- 1);
			MATHERR_MARK();
			s= cus_pow( Sin(angle), wi->powAFlag- 1);
			if( c ){
				x/= c;
			}
			if( s ){
				y/= s;
			}
		}
		  /* Use Pythagoras	*/
		y= sqrt( x*x + y*y );
	}
	if( wi->sqrtYFlag> 0 && !wi->powYFlag ){
		wi->powYFlag= 0.5;
	}
	if( wi->logYFlag && wi->logYFlag!= -1 ){
		value=( cus_pow_y_pow_xY( wi, 10.0, y ) );
	}
	else if( wi->sqrtYFlag && wi->sqrtYFlag!= -1 ){
		value=( cus_powY( wi, y ) );
	}
	else{
		value=( y);
	}
	wi->Yscale= ys;
	return( value );
}

double X_Value(LocalWin *wi, double x)
{ int pF= wi->polarFlag;
  double X;
	check_wi( &wi, "X_Value" );
	wi->polarFlag= 0;
	X= Reform_X(wi, x,0);
	wi->polarFlag= pF;
	return(X);
}

double Y_Value(LocalWin *wi, double y)
{ int pF= wi->polarFlag;
  double Y;
	check_wi( &wi, "Y_Value" );
	wi->polarFlag= 0;
	Y= Reform_Y(wi, y,0);
	wi->polarFlag= pF;
	return(Y);
}

double Trans_X( LocalWin *wi, double x)
{
	check_wi( &wi, "Trans_X" );
	if( wi->logXFlag && wi->logXFlag!= -1 ){
		if( x== 0 && wi->lopX> wi->_log_zero_x && wi->_log_zero_x ){
			return( wi->log10_zero_x );
		}
		else{
			return( cus_log10X( wi, x) );
		}
	}
	else if( wi->sqrtXFlag && wi->sqrtXFlag!= -1){
		if( !wi->powXFlag ){
			wi->powXFlag= 0.5;
		}
		return( cus_pow( x* wi->Xscale, wi->powXFlag ) );
	}
	else{
		return( x* wi->Xscale );
	}
}

double Trans_Y( LocalWin *wi, double y, int is_bounds)
{
	check_wi( &wi, "Trans_Y" );
	if( wi->absYFlag && !is_bounds ){
		y= ABS(y);
	}
	if( wi->logYFlag && wi->logYFlag!= -1 ){
		if( y== 0 && wi->lopY> wi->_log_zero_y && wi->_log_zero_y ){
			return( wi->log10_zero_y );
		}
		else{
			return( cus_log10Y( wi, y) );
		}
	}
	else if( wi->sqrtYFlag && wi->sqrtYFlag!= -1){
		if( !wi->powYFlag ){
			wi->powYFlag= 0.5;
		}
		return( cus_pow( y* wi->Yscale, wi->powYFlag ) );
	}
	else{
		return( y* wi->Yscale );
	}
}

/* The reverse of Reform_X(wi, x, y)	*/
double Trans_XY( LocalWin *wi, double *x, double *y, int is_bounds)
{ double X;
	*x= Trans_X( wi, *x);
	*y= Trans_Y( wi, *y, is_bounds);
	if( wi->polarFlag ){
		if( Gonio_Base_Value!= wi->radix || Gonio_Base_Offset!= wi->radix_offset ){
			Gonio_Base( wi, wi->radix, wi->radix_offset );
		}
		X= *x * XscaleR;
		*y= ABS( *y);
		*x= cus_pow( Cos(X), wi->powAFlag ) * *y;
		*y*= cus_pow( Sin(X), wi->powAFlag );
	}
	return( *x );
}

/* The reverse of Reform_Y(wi, y, x)	*/
double Trans_YX( LocalWin *wi, double *y, double *x, int is_bounds)
{ double X;
	*x= Trans_X( wi, *x);
	*y= Trans_Y( wi, *y, is_bounds);
	if( wi->polarFlag ){
		if( Gonio_Base_Value!= wi->radix || Gonio_Base_Offset!= wi->radix_offset ){
			Gonio_Base( wi, wi->radix, wi->radix_offset );
		}
		X= *x * XscaleR;
		*y= ABS( *y);
		*x= cus_pow( Cos(X), wi->powAFlag ) * *y;
		*y*= cus_pow( Sin(X), wi->powAFlag );
	}
	return( *y );
}

char **Argv;
int Argc;

char event_read_buf[64]= "";
int event_read_buf_cleared= False;
extern double *ascanf_ReadBufVal;

char *LocalWinRepr( LocalWin *wi, char *buf )
{ static char lbuf[16];
	if( !buf ){
		buf = lbuf;
	}
	if( wi ){
		sprintf( buf, "%02d.%02d.%02d",
			wi->parent_number, wi->pwindow_number, wi->window_number
		);
	}
	else{
		sprintf( buf, "<NULL>" );
	}
	return( buf );
}


char *splitmodestring(LocalWin *wi)
{ char *c;
	if( wi->cutAction & _deleteAction ){
		c= "Delete mode";
	}
	else if( wi->cutAction & _spliceAction ){
		c= "Splice/Cut/Split mode";
	}
	else{
		c= "";
	}
	return( c );
}

extern int XG_XSync( Display *, Bool );

void SetWindowTitle( LocalWin *wi, double time )
{
	if( !wi ){
		fprintf( StdErr, "SetWindowTitle(%s) called with NULL LocalWindow pointer!\n",
			d2str( time, 0,0)
		);
		return;
	}
	if( !wi->window ){
		return;
	}
	else{
	  char *pf= (UPrintFileName[0])? UPrintFileName : (PrintFileName)? PrintFileName : "";
	  char *splmode= splitmodestring(wi);
	  char *form, rb[72]= "", snr[128];
#ifdef STATIC_TITLE
	  char window_name[1024];
	  int window_name_len= sizeof(window_name);
#else
	  _ALLOCA( window_name, char, strlen(wi->title_template)+strlen(pf)+strlen(XLABEL(wi))+strlen(YLABEL(wi))+128+ strlen(splmode)+3+sizeof(snr),
		  window_name_len);
#endif
	  double YRange= (double)(wi->XOppY - wi->XOrgY);
	  double aspect= fabs( (wi->win_geo.bounds._hiX - wi->win_geo.bounds._loX) /
						(wi->win_geo.bounds._hiY - wi->win_geo.bounds._loY)
					);
	  int len;

		if( !wi->window ){
			return;
		}

		if( wi->draw_count> 0 ){
			time/= wi->draw_count;
		}
		switch( wi->dev_info.resized ){
			case 0:
			default:
				form= "%.2g";
				break;
			case 1:
				form= "%.2g+";
				break;
			case 2:
				form= "%.2g*";
				break;
		}
		if( NaN(wi->aspect_ratio) ){
			form= "%.2g**";
		}
		else if( wi->aspect_ratio ){
			form= "%.2g*";
		}
		if( event_read_buf[0] ){
			sprintf( rb, " \"%s\"", event_read_buf);
		}
#ifdef STATIC_TITLE
		len= sprintf( window_name, wi->title_template, (TrueGray)? "XGg" : "XG",
			((!wi->raw_display && wi->use_transformed))? "Q " :
				((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val))? "R " : "",
			wi->parent_number, wi->pwindow_number, wi->window_number,
			d2str( aspect, NULL, NULL),
			(YRange)? d2str( (double)(wi->XOppX - wi->XOrgX) / YRange, form, NULL) : "?",
			d2str( time, "%.3g", NULL), wi->draw_count, (long) getpid(), rb
		);
#else
		{ char *x= XLABEL(wi), *y= YLABEL(wi), *X= x, *Y= y, XX= 0, YY= 0;
		  char modbuf[16];
			  /* Temporarily truncate the [XY]label at the first ';'	*/
			  /* But skip leading whitespace and ';'s	*/
			while( *X && (isspace(*X) || *X== ';') ){
				X++;
			}
			while( *X && *X!= ';' ){
				X++;
			}
			if( *X ){
				  /* Store the original char to put it back later	*/
				XX= *X;
				*X= '\0';
			}
			while( *Y && (isspace(*Y) || *Y== ';') ){
				Y++;
			}
			while( *Y && *Y!= ';' ){
				Y++;
			}
			if( *Y ){
				YY= *Y;
				*Y= '\0';
			}
			modbuf[0]= ((!wi->raw_display && wi->use_transformed))? 'Q' :
				RAW_DISPLAY(wi)? 'R' : '\0',
			modbuf[1]= ' ';
			modbuf[2]= '\0';
			if( wi->silenced ){
				if( modbuf[0] ){
					modbuf[1]= ' ';
					modbuf[2]= 'S';
					modbuf[3]= ' ';
					modbuf[4]= '\0';
				}
				else{
					modbuf[0]= 'S';
					modbuf[1]= ' ';
					modbuf[2]= '\0';
				}
			}
			if( *splmode ){
				len= sprintf( window_name, "[%s] ", splmode );
			}
			else{
				len= 0;
			}
			if( wi->num2Draw== 1 ){
				sprintf( snr, " #%d ", wi->first_drawn );
			}
			else{
				sprintf( snr, " %d-%d;%d/%d/%d ", wi->first_drawn, wi->last_drawn, wi->numDrawn, wi->num2Draw, setNumber );
			}
			len+= sprintf( &window_name[len], wi->title_template, (TrueGray)? "XGg" : "XG",
				modbuf,
				wi->parent_number, wi->pwindow_number, wi->window_number,
				d2str( aspect, "%.2g", NULL),
				(YRange)? d2str( (double)(wi->XOppX - wi->XOrgX) / YRange, form, NULL) : "?",
				pf,
				y, x, snr,
				d2str( time, "%.3g", NULL), wi->draw_count, (long) getpid(), rb
			);
			if( !*X && XX ){
				*X= XX;
			}
			if( !*Y && YY ){
				*Y= YY;
			}
		}
#endif
	/* 	wi->draw_count= -1;	*/
		if( len>= window_name_len ){
			fprintf( StdErr, "SetWindowTitle(): wrote %d bytes in string of %d: prepare for crash\n",
				len, window_name_len
			);
		}
		if( wi->mapped || UnmappedWindowTitle ){
			XStoreName(disp, wi->window, parse_codes( window_name ) );
		}
		if( SetIconName ){
			XSetIconName(disp, wi->window, window_name );
		}
		if( debugFlag ){
			fprintf( StdErr, "WindowTitle(\"%s\")\n", window_name );
		}
		if( wi->SD_Dialog || wi->HO_Dialog ){
		  ALLOCA( wn, char, window_name_len+8, wn_len);
			if( wi->SD_Dialog ){
				sprintf( wn, "S'%s", window_name );
				XStoreName(disp, wi->SD_Dialog->win, wn );
				if( SetIconName ){
					XSetIconName(disp, wi->SD_Dialog->win, wn );
				}
			}
			if( wi->HO_Dialog ){
				sprintf( wn, "D'%s", window_name );
				XStoreName(disp, wi->HO_Dialog->win, wn );
				if( SetIconName ){
					XSetIconName(disp, wi->HO_Dialog->win, wn );
				}
			}
		}
		if( wi->mapped ){
			XG_XSync( disp, False );
		}
		GCA();
	}
}

int noTitleMessages= 0;

void TitleMessage( LocalWin *wi, char *msg )
{ static char *name= NULL;
	if( disp && wi && wi->window && !noTitleMessages ){
		if(
			debugFlag
#ifdef DEBUG
			|| getenv("TITLEMESSAGES")
#endif
		){
			fprintf( StdErr, "TitleMessage(0x%lx,%s,%s)\n", wi, msg, name );
			fflush( StdErr );
		}
		if( !wi->animating ){
			if( msg ){
				if( wi->mapped || UnmappedWindowTitle){
					if( !name ){
						if( !RemoteConnection ){
							XFlush( disp );
						}
						if( !XFetchName( disp, wi->window, &name ) ){
							name= NULL;
						}
					}
					XStoreName( disp, wi->window, msg );
					if( wi->mapped && !RemoteConnection ){
						XFlush( disp );
					}
				}
				strncpy( ps_comment, msg, 1023);
				if( debugFlag ){
					fprintf( StdErr, "TitleMessage(\"%s\")\n", msg );
				}
			}
			else if( name ){
				if( wi->mapped || UnmappedWindowTitle){
					XStoreName( disp, wi->window, name );
					if( wi->mapped && !RemoteConnection ){
						XFlush( disp );
					}
				}
				if( debugFlag ){
					fprintf( StdErr, "TitleMessage() \"%s\"\n", name );
				}
				strncpy( ps_comment, name, 1023);
				XFree( name );
				name= NULL;
			}
		}
	}
}

void AdaptWindowSize( LocalWin *wi, Window win, int w, int h )
{ XWindowAttributes win_attr;
  Boolean changed= False;
	XGetWindowAttributes(disp, win, &win_attr);
	if( !wi->animating ){
	   Window dummy;
		XTranslateCoordinates( disp, win, win_attr.root, 0, 0,
			&win_attr.x, &win_attr.y, &dummy
		);
		wi->dev_info.area_x= win_attr.x;
		wi->dev_info.area_y= (win_attr.y-= WM_TBAR);
	}
	if( w && h ){
		win_attr.width= w;
		win_attr.height= h;
	}
	if( wi->dev_info.area_w != win_attr.width ){
		if( wi->dev_info.resized!= -1 ){
			wi->dev_info.old_area_w= wi->dev_info.area_w;
		}
		wi->dev_info.area_w = win_attr.width;
		changed= True;
	}
	if( wi->dev_info.area_h != win_attr.height ){
		if( wi->dev_info.resized!= -1 ){
			wi->dev_info.old_area_h= wi->dev_info.area_h;
		}
		wi->dev_info.area_h = win_attr.height;
		changed= True;
	}
	if( changed ){
		if( !wi->redraw ){
			wi->redraw= 1;
		}
		wi->dev_info.resized= 0;
	}
	XMoveWindow(disp, wi->YAv_Sort_frame.win,
		(int) (win_attr.width- (5*BTNPAD+ wi->YAv_Sort_frame.width+ wi->cl_frame.width+ wi->hd_frame.width+
									wi->settings_frame.width+ wi->info_frame.width+ wi->label_frame.width+ wi->ssht_frame.width+
									7* BTNINTER)),
		(int) (BTNPAD)
	);
	XMoveWindow(disp, wi->close,
		(int) (win_attr.width- (BTNPAD+ wi->cl_frame.width+ wi->hd_frame.width+
									wi->settings_frame.width+ wi->info_frame.width+ wi->label_frame.width+ wi->ssht_frame.width+
									5* BTNINTER)),
		(int) (BTNPAD)
	);
	XMoveWindow(disp, wi->hardcopy,
		(int) (win_attr.width- (BTNPAD+ wi->hd_frame.width+
									wi->settings_frame.width+ wi->info_frame.width+ wi->label_frame.width+ wi->ssht_frame.width+
									4* BTNINTER)),
		(int) (BTNPAD)
	);
	XMoveWindow(disp, wi->settings,
		(int) (win_attr.width- (BTNPAD+ wi->settings_frame.width+ wi->info_frame.width+ wi->label_frame.width+ wi->ssht_frame.width+
								3* BTNINTER)),
		(int) (BTNPAD)
	);
	XMoveWindow(disp, wi->info,
		(int) (win_attr.width- (BTNPAD+ wi->info_frame.width+ wi->label_frame.width+ wi->ssht_frame.width+
								2* BTNINTER)),
		(int) (BTNPAD)
	);
	XMoveWindow(disp, wi->label,
		(int) (win_attr.width- (BTNPAD+ wi->label_frame.width+ wi->ssht_frame.width+
								1* BTNINTER)),
		(int) (BTNPAD)
	);
	XMoveWindow(disp, wi->ssht_frame.win,
		(int) (win_attr.width- (BTNPAD+ wi->ssht_frame.width+
								0* BTNINTER)),
		(int) (BTNPAD)
	);
	if( wi->label_IFrame.mapped ){
		XMoveWindow(disp, wi->label_IFrame.win,
			(int) (win_attr.width- (BTNPAD+ wi->label_IFrame.width+ 1* BTNINTER))/ 2,
			(int) (win_attr.height- (BTNPAD+ wi->label_IFrame.height+ 1* BTNINTER))/ 2
		);
	}
	SetXClip( wi, 0, 0, wi->dev_info.area_w, wi->dev_info.area_h );
	wi->clipped= 0;
}

LocalWin StubWindow, *primary_info, *StubWindow_ptr= &StubWindow, *InitWindow= NULL;
Window init_window;
extern LocalWin *ActiveWin;

extern LocalWindows *WindowList, *WindowListTail;

int DrawAllSets= False;

int draw_set( LocalWin *wi, int idx)
{ int r= 0;
	DRAW_SET( wi, wi->draw_set, idx, r );
	return( r );
}

int _DiscardedPoint( LocalWin *wi, DataSet *set, int idx)
{ int dis_set= DiscardedPoint(NULL, set, idx), win_set;
	if( wi && wi->discardpoint && wi->discardpoint[set->set_nr] ){
		win_set= (int) wi->discardpoint[set->set_nr][idx];
		if( dis_set<= 0 && win_set< 0 ){
			return( -1 );
		}
		else if( dis_set< 0 && win_set== 0 ){
			return( dis_set );
		}
		else{
			return( dis_set || win_set );
		}
	}
	else{
		return( dis_set );
	}
}

extern short *_drawingOrder;
extern int drawingOrder_set;

/* Begin for a support of changing the order in which sets are to be drawn	*/
int drawingOrder( LocalWin *wi, int idx)
{
	if( idx>= setNumber ){
		return(idx);
	}
	else{
		if( !drawingOrder_set ){
		  int i;
			for( i= 0; i< setNumber; i++ ){
				_drawingOrder[i]= i;
			}
		}
		return( _drawingOrder[idx] );
	}
}
/* #define DO(n)	drawingOrder(wi,(n))	*/
#define DO(n)	(n)

extern int data_sn_number;
Boolean set_sn_nr= False, dialog_redraw= False;

int cycle_plot_only_set( LocalWin *wi, int sets)
{ int i;
	wi->redrawn= 0;
	wi->plot_only_set0= (wi->plot_only_set0+ sets ) % setNumber;
	if( wi->plot_only_set0< 0 ){
		wi->plot_only_set0= setNumber + wi->plot_only_set0;
	}
	for( i= 0; i< setNumber; i++ ){
		wi->draw_set[i]= 0;
	}
	wi->draw_set[ wi->plot_only_set0 ]= 1;
	if( data_sn_number>= 0 && data_sn_number< setNumber ){
		data_sn_number= wi->plot_only_set0;
	}
	wi->ctr_A= 0;
	if( DiscardUndo.set== (DataSet*) -1 && DiscardUndo.wi ){
		DiscardUndo.set= NULL;
		DiscardUndo.wi= NULL;
	}
	return( 1 );
}

void cycle_drawn_sets( LocalWin *wi, int sets )
{
  int i, j, first, next;
	wi->redrawn= 0;
	for( i= 0; i< ABS(sets); i++ ){
		if( wi->ctr_A ){
		  /* This command erases the information stored when ^A is pressed	*/
			for( j= 0; j< setNumber; j++ ){
				if( wi->draw_set[j]< 0 ){
					wi->draw_set[j]= 0;
				}
			}
			wi->ctr_A= 0;
		}
		if( sets< 0 ){
			for( j= 0, first= wi->draw_set[j]; j< setNumber-1; j++ ){
				next= 1;
				while( j+next< setNumber && AllSets[j+next].numPoints<= 0 ){
					next+= 1;
				}
				wi->draw_set[j]= wi->draw_set[j+next];
			}
			  /* Wrap: when draw_set[j] is set to first, a set "shifted off"
			   \ at the low end, reappears at the high end (i.e. displayed set #0
			   \ becomes displayed set #setNumber-1). Setting it to -first
			   \ only stores the info, without drawing the set. After shifting, we
			   \ will then check whether sets are drawn, and multiply by -1 when
			   \ nothing is visible.
			   */
			wi->draw_set[j]= - first;
		}
		else{
			for( j= setNumber-1, first= wi->draw_set[j]; j> 0; j-- ){
				next= 1;
				while( j-next>= 0 && AllSets[j-next].numPoints<= 0 ){
					next+= 1;
				}
				wi->draw_set[j]= wi->draw_set[j-next];
			}
			wi->draw_set[j]= - first;
		}
		for( j= 0, first= 0; j< setNumber && !first ; j++ ){
			if( wi->draw_set[j]> 0 ){
				first= 1;
				if( data_sn_number>= 0 && data_sn_number< setNumber ){
					data_sn_number= j;
				}
			}
		}
		if( !first ){
			for( j= 0; j< setNumber; j++ ){
				wi->draw_set[j]*= -1;
			}
		}
	}
	if( debugFlag ){
		fprintf( StdErr, "Cycled sets: ");
		for( j= 0; j< setNumber; j++ ){
			fprintf( StdErr, "%d ", wi->draw_set[j] );
		}
		fputc( '\n', StdErr );
	}
	wi->ctr_A= 0;
	if( DiscardUndo.set== (DataSet*) -1 && DiscardUndo.wi ){
		DiscardUndo.set= NULL;
		DiscardUndo.wi= NULL;
	}
}

int cycle_highlight_sets( LocalWin *wi, int sets )
{
  int i, j, first, next;
	wi->redrawn= 0;
	for( i= 0; i< ABS(sets); i++ ){
		if( sets< 0 ){
			for( j= 0, first= wi->legend_line[j].highlight; j< setNumber-1; j++ ){
				next= 1;
				while( j+next< setNumber && AllSets[j+next].numPoints<= 0 ){
					next+= 1;
				}
				wi->legend_line[j].highlight= wi->legend_line[j+next].highlight;
			}
			  /* Wrap: when legend_line[j].highlight is set to first, a set "shifted off"
			   \ at the low end, reappears at the high end (i.e. displayed set #0
			   \ becomes displayed set #setNumber-1). Setting it to -first
			   \ only stores the info, without drawing the set. After shifting, we
			   \ will then check whether sets are drawn, and multiply by -1 when
			   \ nothing is visible.
			   */
			wi->legend_line[j].highlight= - first;
		}
		else{
			for( j= setNumber-1, first= wi->legend_line[j].highlight; j> 0; j-- ){
				next= 1;
				while( j-next>= 0 && AllSets[j-next].numPoints<= 0 ){
					next+= 1;
				}
				wi->legend_line[j].highlight= wi->legend_line[j-next].highlight;
			}
			wi->legend_line[j].highlight= - first;
		}
		for( j= 0, first= 0; j< setNumber && !first ; j++ ){
			if( wi->legend_line[j].highlight> 0 ){
				first= 1;
				if( data_sn_number>= 0 && data_sn_number< setNumber ){
					data_sn_number= j;
				}
			}
		}
		if( !first ){
			for( j= 0; j< setNumber; j++ ){
				if( (wi->legend_line[j].highlight*= -1) ){
					first= 1;
				}
			}
		}
	}
	if( debugFlag ){
		fprintf( StdErr, "Cycled sets: ");
		for( j= 0; j< setNumber; j++ ){
			fprintf( StdErr, "%d ", wi->legend_line[j].highlight );
		}
		fputc( '\n', StdErr );
	}
	  /* The return-code of this function tells whether any sets are actually highlighted	*/
	return( (first)? True : False );
}

void files_and_groups( LocalWin *wi, int *fn, int *grps )
{ int i;
	wi->numFiles= 1;
	wi->numGroups= 1;
	  /* Determine the true number of files present (altered by
	   \ taking averages (Shift-Mod1-a)).
	   */
	wi->group[0]= 0;
	for( i= 1; i< setNumber; i++ ){
		if( wi->fileNumber[i]!= wi->fileNumber[i-1] ){
			wi->numFiles+= 1;
		}
		if( wi->new_file[i] ){
			wi->group[i]= wi->group[i-1] + 1;
			wi->numGroups+= 1;
		}
		else{
			wi->group[i]= wi->group[i-1];
		}
	}
	if( fn ){
		*fn= wi->numFiles;
	}
	if( grps ){
		*grps= wi->numGroups;
	}
	if( debugFlag ){
		fprintf( StdErr, "files_and_groups(): %d files; %d groups\n", (int) wi->numFiles, (int) wi->numGroups );
	}
}

void cycle_plot_only_group( LocalWin *wi, int groups )
{ int i, fn, grps, first= 0;
	wi->redrawn= 0;
	files_and_groups( wi, &fn, &grps );
	wi->plot_only_group= (wi->plot_only_group+ groups) % grps;
	if( wi->plot_only_group< 0 ){
		wi->plot_only_group= grps + wi->plot_only_group;
	}
	if( debugFlag ){
		fprintf( StdErr, "cycle_plot_only_group(%d): plotting group %d of %d\n", groups, wi->plot_only_group+ 1, grps );
	}
	for( i= 0; i< setNumber; i++ ){
		if( wi->group[i]== wi->plot_only_group && AllSets[i].numPoints> 0 ){
			wi->draw_set[i]= 1;
			if( !first ){
				first= 1;
				if( data_sn_number>= 0 && data_sn_number< setNumber ){
					data_sn_number= i;
				}
			}
		}
		else{
			wi->draw_set[i]= 0;
		}
	}
	wi->ctr_A= 0;
	if( DiscardUndo.set== (DataSet*) -1 && DiscardUndo.wi ){
		DiscardUndo.set= NULL;
		DiscardUndo.wi= NULL;
	}
}

void cycle_plot_only_file( LocalWin *wi, int files )
{ int i, fn, grps, first= 0;
	wi->redrawn= 0;
	files_and_groups( wi, &fn, &grps );
	wi->plot_only_file= (wi->plot_only_file+ files) % fn;
	if( wi->plot_only_file< 0 ){
		wi->plot_only_file= fn + wi->plot_only_file;
	}
	if( debugFlag ){
		fprintf( StdErr, "cycle_plot_only_file(): plotting file %d\n", wi->plot_only_file+ 1 );
	}
	for( i= 0; i< setNumber; i++ ){
		if( wi->fileNumber[i]== -1 ){
			wi->fileNumber[i]= (i)? wi->fileNumber[i-1] : 0;
		}
		if( wi->fileNumber[i]== wi->plot_only_file+1 && AllSets[i].numPoints> 0 ){
			wi->draw_set[i]= 1;
			if( !first ){
				first= 1;
				if( data_sn_number>= 0 && data_sn_number< setNumber ){
					data_sn_number= i;
				}
			}
		}
		else{
			wi->draw_set[i]= 0;
		}
	}
	wi->ctr_A= 0;
	if( DiscardUndo.set== (DataSet*) -1 && DiscardUndo.wi ){
		DiscardUndo.set= NULL;
		DiscardUndo.wi= NULL;
	}
}

void SwapSets( LocalWin *lwi )
{ int i;
	lwi->redrawn= 0;
	for( i= 0; i< setNumber; i++ ){
		if( AllSets[i].numPoints> 0 ){
			lwi->draw_set[i]= !lwi->draw_set[i];
		}
	}
	lwi->ctr_A= 0;
	lwi->redraw= 1;
	lwi->halt= 0;
	lwi->printed= 0;
	lwi->draw_count= 0;
	if( DiscardUndo.set== (DataSet*) -1 && DiscardUndo.wi ){
		DiscardUndo.set= NULL;
		DiscardUndo.wi= NULL;
	}
}

void ShowAllSets( LocalWin *lwi )
{ int i;
	lwi->redrawn= 0;
	for( i= 0; i< setNumber; i++ ){
		lwi->draw_set[i]= (lwi->draw_set[i]<0)? 0 : (lwi->draw_set[i]==0)? -1 : 1;
	}
	lwi->ctr_A= ! lwi->ctr_A;
	lwi->redraw= 1;
	lwi->halt= 0;
	lwi->printed= 0;
	lwi->draw_count= 0;
	if( DiscardUndo.set== (DataSet*) -1 && DiscardUndo.wi ){
		DiscardUndo.set= NULL;
		DiscardUndo.wi= NULL;
	}
}

double Animation_Time= 0.0, Animations= 0.0;
SimpleStats Animation_Windows;
int Animating= False;

/* XDBE timing variables are only used when XDBE_TIME_STATS is defined! */
double XDBE_inside= 0, XDBE_outside= 0;
unsigned int XDBE_count= 0, XDBE_validtimer= False;

extern LocalWin *theSettingsWin_Info, *thePrintWin_Info;

GC msgGC(LocalWin *wi)
{ static GC msgGC= 0;
	if( msgGC == (GC) 0 ){
		unsigned long gcmask;
		XGCValues gcvals;

		gcmask = ux11_fill_gcvals(&gcvals, GCForeground, zeroPixel ^ bgPixel,
					  GCFunction, GXxor, GCFont, fbFont.font->fid,
					  UX11_END);
		msgGC = XCreateGC(disp, wi->window, gcmask, &gcvals);
	}
	return( msgGC );
}

char *Get_YAverageSorting(LocalWin *wi)
{ extern char *YAv_SortTypes[];
  int srt= xtb_br_get(wi->YAv_Sort_frame.win);
	if( srt< 0 ){
		return( "Nsrt" );
	}
	if( debugFlag ){
		fprintf( StdErr, "Get_YAverageSorting(#%d)== \"%s\"\n",
			srt, YAv_SortTypes[srt]
		);
	}
	return( YAv_SortTypes[srt] );
}

extern Cursor theCursor, noCursor;

void Add_mStats( LocalWin *wi )
{ extern SimpleStats overlapMO, overlapSO;
	SS_Add_Data_( SS_mX, 1, SS_Mean_(wi->SS_Xval), 1.0 );
	SS_Add_Data_( SS_mY, 1, SS_Mean_(wi->SS_Yval), 1.0 );
	if( wi->SAS_O.pos_count || wi->SAS_O.neg_count ){
		SAS_Add_Data_( SAS_mO, 1, SAS_Mean_(wi->SAS_O), 1.0, (int) *SAS_converts_angle );
	}
	if( wi->SS_E.count ){
		SS_Add_Data_( SS_mE, 1, SS_Mean_(wi->SS_E), 1.0 );
	}
	if( wi->SS_I.count ){
		SS_Add_Data_( SS_mI, 1, SS_Mean_(wi->SS_I), 1.0 );
	}
	SS_Add_Data_( SS_mLY, 1, SS_Mean_(wi->SS_LY), 1.0 );
	SS_Add_Data_( SS_mHY, 1, SS_Mean_(wi->SS_HY), 1.0 );
	if( wi->show_overlap ){
		overlap(wi);
		SS_Add_Data_( SS_mMO, 1, SS_Mean_(overlapMO), 1.0 );
		SS_Add_Data_( SS_mSO, 1, SS_Mean_(overlapSO), 1.0 );
	}
	SS_Add_Data_( SS_mPoints, 1, SS_Mean_(SS_Points), 1.0 );
}

#define __PS_STATE(wi) ((psUserInfo*)wi->dev_info.user_state)

psUserInfo *PS_STATE(LocalWin *wi )
{  static psUserInfo lpUI;
	if( wi->dev_info.user_state ){
		return( __PS_STATE(wi) );
	}
	else{
		lpUI.Printing= X_DISPLAY;
		return( &lpUI );
	}
}

/* #ifdef __GNUC__	*/
/* __inline__	*/
/* #endif	*/
/* extern void DBE_Swap( LocalWin *wi, XdbeSwapInfo *XDBE_info, int finish );	*/

int RedrawAgain( LocalWin *wi )
{
	if( wi ){
		if( wi->redraw== -2 ){
			return(0);
		}
		wi->redraw= 1;
		wi->halt= 0;
		  /* This routine is not to be called from within the Settings Dialog handlers
		   \ as it resets the flag indicating a redraw generated by one of those..
		   */
		dialog_redraw= False;
		if( wi->raw_once< 0 ){
			wi->raw_display= wi->raw_val;
		}
		wi->raw_once= 0;
		if( PS_STATE(wi)->Printing!= PS_PRINTING){
			AdaptWindowSize( wi, wi->window, 0, 0 );
		}
		files_and_groups( wi, NULL, NULL );
	}
	return( DrawWindow( wi ) );
}

int RedrawNow( LocalWin *wi )
{ char *faur= fascanf_unparsed_remaining;
  int r;
	if( wi->delete_it== -1 || !setNumber ){
		return(0);
	}
	if( wi ){
		  /* Don't redraw a window whose redraw has been previously delayed. This
		   \ delayed redraw should be performed by the delaying call to DrawWindow(),
		   \ or else the toplevel loop will do it.
		   \ Do not neither redraw a window that is being animated. The toplevel loop
		   \ will redraw it. Thus, the animation will be less disturbed. Also, it can
		   \ happen that allowing the redraw here will interfere with processing being
		   \ performed during the animation (if the ongoing redraw does not immediately
		   \ abort after the currently requested redraw - e.g. $CurrentSet will have
		   \ a different value then). Get it?
		   */
		if( wi->redraw== -2 || (Animating && wi->animating) ){
			return(0);
		}
		else if( wi->redrawn== -3 ){
			if( xtb_error_box( wi->window,
					" At least 3 successive blocking errors occured while attempting\n"
					" to redraw this window the last time. Would you like to try again?",
					"Note:"
				)> 0
			){
				wi->redrawn= 0;
			}
			else{
				wi->redraw= 0;
				wi->halt= 1;
				*ascanf_escape_value= ascanf_escape= 1;
				XSync( disp, True );
				return(0);
			}
		}
		wi->draw_count= 0;
		if( !RAW_DISPLAY(wi) ){
			Check_Process_Dependencies( wi );
		}
		  /* 20000904: may need this!	*/
		wi->dont_clear= False;
#ifdef XDBE_TIME_STATS
		if( wi->raw_display && !wi->animate ){
			XDBE_validtimer= False;
		}
#endif
	}
	r= RedrawAgain( wi );
	fascanf_unparsed_remaining= faur;
#if defined(__GNUC__) && defined(i386)
	if( wi->pen_list && wi->no_pens== -1 ){
		wi->no_pens= 0;
		wi->redraw= 1;
	}
#endif
	return( r );
}

Status ExposeEvent( LocalWin *rwi )
{ XEvent evt;
	evt.type= Expose;
	evt.xexpose.display= disp;
	evt.xexpose.x= 0;
	evt.xexpose.y= 0;
	evt.xexpose.width= rwi->dev_info.area_w;
	evt.xexpose.height= rwi->dev_info.area_h;
	evt.xexpose.window= rwi->window;
	evt.xexpose.count= 0;
	rwi->redraw= 1;
	rwi->halt= 0;
	rwi->draw_count= 0;
	return( XSendEvent( disp, rwi->window, 0, ExposureMask, &evt) );
}

int RedrawSet( int set_nr, Boolean doit )
{ LocalWindows *WL= WindowList;
  LocalWin *wi;
  int n= 0;
	while( WL ){
		wi= WL->wi;
		if( set_nr>= 0 ){
			if( draw_set( wi, set_nr) ){
				wi->redraw+= 1;
				n+= 1;
			}
		}
		WL= WL->next;
	}
	if( doit ){
	  WL= WindowList;
		n= 0;
		while( WL ){
			wi= WL->wi;
			if( wi->redraw  && !wi->silenced ){
				{ XEvent dum;
				  /* 20050110: a manual redraw cancels all pending Expose events that would
				   \ cause (subsequent) redraws. (NB: events generated after this point are not
				   \ affected, of course.)
				   */
					XG_XSync( disp, False );
					while( XCheckWindowEvent(disp, wi->window, ExposureMask|VisibilityChangeMask, &dum) );
				}
				RedrawNow( wi );
				n+= 1;
			}
			WL= WL->next;
		}
	}
	return(n);
}

extern char *ShowLegends(LocalWin *, int, int);
extern char *XG_GetString( LocalWin *wi, char *text, int buflen, Boolean do_events);

extern int X_silenced();

int handle_event_times= 1;

GC ACrossGC= 0, BCrossGC;
int CursorCross= 0, CursorCross_Labeled= 0;

void ChangeCrossGC( GC *CrossGC)
{ XGCValues gcvals;
  unsigned long gcmask;

	if( *CrossGC ){
	  XColor fg_color, cc_color;
		gcvals.plane_mask= AllPlanes;
		gcvals.font = fbFont.font->fid;
		if( CrossGC== &ACrossGC ){
			gcvals.foreground = zeroPixel ^ bgPixel;
		}
		else{
/* 			gcvals.foreground = highlightPixel ^ bgPixel;	*/
			gcvals.foreground = gridPixel ^ bgPixel;
		}
		gcmask = GCPlaneMask | GCFont|GCForeground;
		XChangeGC( disp, *CrossGC, gcmask, &gcvals);
		fg_color.pixel = normPixel;
		XQueryColor(disp, cmap, &fg_color);
		cc_color.pixel= zeroPixel;
		XQueryColor(disp, cmap, &cc_color);
		XRecolorCursor( disp, noCursor, &cc_color, &fg_color);
	}
}

#define TRANX(xval) \
(((double) ((xval) - wi->XOrgX)) * wi->XUnitsPerPixel + wi->UsrOrgX)

#define TRANY(yval) \
(wi->UsrOppY - (((double) ((yval) - wi->XOrgY)) * wi->YUnitsPerPixel))

static Window CC_focus;

void DrawCCross( LocalWin *lwi, Boolean erase, int curX, int curY, char *label )
{ Cursor_Cross *s;
  int revert, Label_len, allow_label= 0;

	if( !CC_focus ){
		XGetInputFocus( disp, &CC_focus, &revert);
	}
	s= &lwi->curs_cross;
	if( erase ){
		if( s->OldLabel[0] && s->had_focus ){
			XDrawString( disp, lwi->window, s->gc,
				s->label_x, s->label_y, s->OldLabel, s->OldLabel_len
			);
			s->OldLabel[0]= '\0';
		}
		if( s->line[0].x1>= 0 && s->line[0].y1>= 0 && s->line[0].x2>= 0 && s->line[0].y2>= 0 ){
			XDrawSegments(disp, lwi->window, s->gc, &s->line[0], 1);
		}
		if( s->line[1].x1>= 0 && s->line[1].y1>= 0 && s->line[1].x2>= 0 && s->line[1].y2>= 0 ){
			XDrawSegments(disp, lwi->window, s->gc, &s->line[1], 1);
		}
	}
	if( CC_focus== lwi->window ){
		s->gc= ACrossGC;
		s->had_focus= True;
	}
	else{
		s->gc= BCrossGC;
		s->had_focus= (CursorCross_Labeled> 1)? True : False;
	}
	CC_focus= 0;
	if( curX>= 0 && curX< lwi->dev_info.area_w ){
		s->line[0].x1= s->line[0].x2= curX;
		s->line[0].y1= 0;
		s->line[0].y2= lwi->dev_info.area_h;
		XDrawSegments(disp, lwi->window, s->gc, &s->line[0], 1);
		allow_label+= 1;
	}
	else{
		s->line[0].x1= s->line[0].y1= s->line[0].x2= s->line[0].y2= -1;
	}
	if( curY>= 0 && curY< lwi->dev_info.area_h ){
		s->line[1].y1= s->line[1].y2= curY;
		s->line[1].x1= 0;
		s->line[1].x2= lwi->dev_info.area_w;
		XDrawSegments(disp, lwi->window, s->gc, &s->line[1], 1);
		allow_label+= 1;
	}
	else{
		s->line[1].x1= s->line[1].y1= s->line[1].x2= s->line[1].y2= -1;
	}
	if( s->had_focus && label && allow_label== 2 ){
	  int dir, ascent, descent, max_ascent, width, height, tx= curX+ 3, ty= curY- 3, just;
	  XCharStruct bb;
		CLIP_EXPR( Label_len, strlen(label), 0, sizeof(s->OldLabel) );

		XTextExtents( fbFont.font, label, Label_len, &dir, &ascent, &descent, &bb);
		max_ascent= bb.ascent;
		width= bb.rbearing - bb.lbearing;
		height= bb.ascent + bb.descent;

		if( tx+ width> lwi->dev_info.area_w ){
			tx= curX- 3;
			if( ty- height< 0 ){
				ty= curY+ 3;
				just= T_UPPERRIGHT;
			}
			else{
				just= T_LOWERRIGHT;
			}
		}
		else{
			if( ty- height< 0 ){
				ty= curY+ 3;
				just= T_UPPERLEFT;
			}
			else{
				just= T_LOWERLEFT;
			}
		}

		  /* justification code; (almost) straight copy from text_X(). Should be a
		   \ single routine in xgX.c ...
		   */
		switch (just) {
				case T_CENTER:
					tx-= (width/2);
					ty-= (height/2);
					break;
				case T_LEFT:
					ty-= (height/2);
					break;
				case T_VERTICAL:
				case T_UPPERLEFT:
					break;
				case T_TOP:
					tx-= (width/2);
					break;
				case T_UPPERRIGHT:
					tx-= width;
					break;
				case T_RIGHT:
					tx-= width;
					ty-= (height/2);
					break;
				case T_LOWERRIGHT:
					tx-= width;
					ty-= height;
					break;
				case T_BOTTOM:
					tx-= (width/2);
					ty-= height;
					break;
				case T_LOWERLEFT:
					ty-= height;
					break;
		}
		XDrawString( disp, lwi->window, s->gc,
			(s->label_x= tx), (s->label_y= ty+ max_ascent), label, Label_len
		);

		strncpy( s->OldLabel, label, (s->OldLabel_len= Label_len) );
		s->OldLabel[ sizeof(s->OldLabel)- 1 ]= '\0';
	}
	else{
		s->label_x= s->label_y= -1;
	}
}

void DrawCCrosses( LocalWin *wi, XEvent *evt, double cx, double cy, int curX, int curY, char *label, char *caller )
{ LocalWindows *WL= WindowList;
  LocalWin *lwi, *aw= ActiveWin;
  Cursor_Cross *s;
  Boolean do_next;

	if( debugFlag ){
		fprintf( StdErr, "MotionEvent (%s #%d) for window %d.%d.%d from %s\n",
			event_name(evt->type), evt->xany.serial,
			wi->parent_number, wi->pwindow_number, wi->window_number,
			caller
		);
	}

	if( wi->mapped ){
		  /* 20041214: If we jump into the while-loop below, we must make sure that the first window in
		   \ the WindowList is handled also! That is, don't update WL the first time around.
		   */
		do_next= False;
		lwi= wi;
		goto do_wi_now;
	}
	else{
		do_next= True;
	}

	while( WL ){
	  double x, y;
	  int sx, sy;
	  int ok;
		lwi= WL->wi;
		if( lwi->mapped && lwi!= wi ){
do_wi_now:;
			s= &lwi->curs_cross;
			if( debugFlag ){
				{ int revert;
					XGetInputFocus( disp, &CC_focus, &revert);
				}
				fprintf( StdErr, "MotionEvent (lwi %d.%d.%d%s): old coords (%d,%d);",
					lwi->parent_number, lwi->pwindow_number, lwi->window_number,
					((CC_focus== lwi->window)? " [inside]" : " [outside]"),
					s->line[0].x1, s->line[1].y1
				);
				fflush( StdErr );
			}
			x= cx;
			y= cy;
			ok= 1;
			if( lwi== wi ){
				sx= curX;
				sy= curY;
			}
			// 20110506: allow a 'local' (= 'self') process!
			/* else */ {
				if( s->fromwin== wi && s->fromwin_process.C_process ){
				  double result[2];
				  int n= 2;
					ActiveWin= wi;
					result[0]= data[0]= x;
					result[1]= data[1]= y;
					*ascanf_self_value= -1;
					*ascanf_current_value= -1;
					*ascanf_counter= (*ascanf_Counter)= -1;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					if( ascanf_verbose ){
						fprintf( StdErr, "DrawCCrosses(): DATA x y: %s", s->fromwin_process.process );
						fflush( StdErr );
					}
					ascanf_arg_error= 0;
					TBARprogress_header= s->fromwin_process.command;
					compiled_fascanf( &n, s->fromwin_process.process, result, NULL, data, column, &s->fromwin_process.C_process );
					if( n>= 1 ){
						x= result[0];
					}
					if( n>= 2 ){
						y= result[1];
					}
					ActiveWin= aw;
				}
				if( x != cx || y != cy ){
					do_transform( lwi, "CursorPos", __DLINE__, "DrawCCrosses()", &ok, NULL,
						&x, NULL, NULL, &y,
						NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0,
						0, 0, False
					);
					if( !ok ){
						x= cx;
						y= cy;
						Trans_XY( lwi, &x, &y, 0 );
					}
					sx= SCREENX( lwi, x );
					sy= SCREENY( lwi, y );
				}
			}
			if( CursorCross_Labeled && (!label || *label=='\0') ){
			  char coords[256], formX[16]= "%.4g", formY[16]= "%.4g";
			  double OnePixX, OnePixY, lcx, lcy;;
				CC_focus= wi->window;
				if( CursorCross_Labeled> 1 ){
				  int revert;
					XGetInputFocus( disp, &CC_focus, &revert);
				}
				OnePixX= Reform_X( wi, wi->XUnitsPerPixel, wi->YUnitsPerPixel );
				OnePixY= Reform_Y( wi, wi->YUnitsPerPixel, wi->XUnitsPerPixel );
				if( CC_focus== lwi->window ){
					lcx= cx, lcy= cy;
				}
				else{
					lcx= Reform_X(lwi, x,y), lcy= Reform_Y(lwi, y,x);
				}
				if( wi->dev_info.area_w> 1024 ){
					formX[2]= '5';
					if( OnePixY< 0.1 || OnePixY> 1 ){
						formY[2]= '5';
					}
				}
				else if( OnePixX> 0.4 && (lcx>=1000 && lcx<10000) ){
					formX[2]= '5';
				}
				if( CC_focus== lwi->window ){
					sprintf( coords, "%s,%s", d2str( cx, formX, NULL), d2str( cy, formY, NULL) );
				}
				else{
				  /* 20040921: show the co-ordinates in the actual window. Take care to do the reverse transform necessary
				   \ to account for log/sqrt axis/es.
				   */
					sprintf( coords, "%s,%s", d2str( Reform_X(lwi, x,y), formX, NULL), d2str( Reform_Y(lwi, y,x), formY, NULL) );
				}
				DrawCCross( lwi, True, sx, sy, coords );
			}
			else{
				DrawCCross( lwi, True, sx, sy, label );
			}
			if( !RemoteConnection ){
				  /* 20041214: doesn't have a lot of effect on the multiple-window ghosting phenomenon: */
				XSync( disp, False );
			}

			if( debugFlag ){
				fprintf( StdErr, " new coords (%s,%s)=>(%s,%s[%d])=>(%d,%d) -> (%d,%d)\n",
					d2str( cx, NULL, NULL), d2str( cy, NULL, NULL),
					d2str( x, NULL, NULL), d2str( y, NULL, NULL), ok,
					s->line[0].x1, s->line[1].y1,
					s->line[0].x2, s->line[1].y2
				);
				fflush( StdErr );
			}
		}
#ifdef DEBUG
		else if( debugFlag ){
			fprintf( StdErr, "DrawCCrosses(): skipping lwi==wi which has been done first.\n" );
		}
#endif

		if( do_next ){
			WL= WL->next;
		}
		else{
			do_next= True;
		}
	}
	ActiveWin= aw;
}

xtb_hret label_func( Window win, int bval, xtb_data info)
/* Button Window    */ /* Button value     */ /* User Information */
/*
 * This routine is called when the label button is pressed
 * in an xgraph window.
 */
{ Window the_win = (Window) info;
  LocalWin *wi;
  extern LocalWin *LabelClipUndo;

	if (!XFindContext(disp, the_win, win_context, (caddr_t *) &wi)) {
		if( CheckMask( xtb_modifier_state, Mod1Mask) ){
		  UserLabel *ul= wi->ulabel;
		  int clipped= 0;
			xtb_bt_set(win, !wi->add_label, (char *) 0);
			while( ul ){
				if( ul->tx1>= wi->loX && ul->tx1<= wi->hiX && ul->ty1>= wi->loY && ul->ty1<= wi->hiY ){
				  int changes= 0;
				  double old2[2];
					old2[0]= ul->x2;
					old2[1]= ul->y2;
					if( ul->tx2< wi->loX ){
						ul->x2= wi->loX;
						changes+= 1;
					}
					if( ul->tx2> wi->hiX ){
						ul->x2= wi->hiX;
						changes+= 1;
					}
					if( ul->ty2< wi->loY ){
						ul->y2= wi->loY;
						changes+= 1;
					}
					if( ul->ty2> wi->hiY ){
						ul->y2= wi->hiY;
						changes+= 1;
					}
					if( changes ){
						ul->do_transform= 0;
						wi->redraw= 1;
						if( !ul->old2 ){
							ul->old2= (double*) calloc( 2, sizeof(double) );
						}
						if( ul->old2 ){
							memcpy( ul->old2, old2, sizeof(old2) );
							clipped+= 1;
						}
					}
				}
				ul= ul->next;
			}
			if( clipped ){
				LabelClipUndo= wi;
				ShiftUndo.set= NULL;
				ShiftUndo.ul= NULL;
				SplitUndo.set= NULL;
				DiscardUndo.set= NULL;
				DiscardUndo.ul.label[0]= '\0';
				if( wi->PlaceUndo.valid ){
					wi->PlaceUndo.valid= False;
				}
				if( BoxFilterUndo.fp ){
					fclose( BoxFilterUndo.fp );
					BoxFilterUndo.fp= NULL;
				}
			}
			if( wi->redraw ){
				RedrawNow( wi );
			}
		}
		else{
			wi->add_label= !wi->add_label;
			if( wi->add_label && CheckMask(xtb_modifier_state,ShiftMask) ){
				wi->add_label= -1;
			}
		}
		xtb_bt_set(win, wi->add_label, (char *) 0);
	}
	return XTB_HANDLED;
}

char *key_param_now[8*(256+12)], key_param_separator = '\0';

int animations= 0, animation_windows= 0;
Time_Struct Animation_Timer, XDBE_Timer;

double LastActionDetails[5];

XEvent *RelayEvent;
int RelayMask;
int Ignore_UnmapNotify= False;

int _Handle_An_Event( XEvent *theEvent, int level, int CheckFirst, char *caller)
{ LocalWin *wi = NULL;
  xtb_frame *frame= NULL, *framelist= NULL;
  int frames= 0, f_context_t, w_context_t, xtb_return, nbytes, idx, flushed= -1;
  char keys[MAXKEYS];
  KeySym keysymbuffer[MAXKEYS];
  static unsigned long calls= 0;
  int return_code= 0;
  Boolean handle_expose_now= False;
  int dbF= debugFlag, *dbFc= dbF_cache;
  extern int Allow_DrawCursorCrosses;
  int gADCC= Allow_DrawCursorCrosses;

		if( RelayEvent ){
			XSendEvent( disp, RelayEvent->xany.window, False, RelayMask, (XEvent*) RelayEvent );
			fprintf( StdErr, "_Handle_An_Event(): sent a relayed event!\n" );
			RelayEvent= NULL;
			RelayMask= 0;
		}
		  /* Ignore any events when we're exitting, UNLESS this is a KeyPress that was sent through
		   \ XSendEvent(). Because that is probably a signal that we received and that ExitProgramme()
		   \ relayed as an X11 event!
		   */
		if( Exitting ){
			if( !(theEvent->type== KeyPress && theEvent->xany.send_event) ){
				return(1);
			}
			else{
				fprintf( StdErr, "_Handle_An_Event(): received a KeyPress event sent through XSendEvent() -- a signal() relayed by ExitProgramme()?!\n" );
				debugFlag= True;
			}
		}

		  /* 20041217:
		   \ We need double setting/caching of Allow_DrawCursorCrosses ([de]activates DrawCursorCrosses() which can be called
		   \ from e.g. xtb_dispatch(). Basically, this variable should be True whenever DrawCursorCrosses() is called by
		   \ some child of ours, EXCEPT when called by (a child of) xtb_dispatch() invoked somewhat below, where
		   \ we set Allow_DrawCursorCrosses to False. This makes it possible to call DrawCCrosses() ourselves, avoiding it
		   \ also being called through those xtb_dispatch() invocations, while also calling it (through DrawCursorCrosses())
		   \ when e.g. an input dialog is open.
		   \ Complex stuff.... too complex.
		   */
		Allow_DrawCursorCrosses= True;

		if( event_read_buf[0] ){
			if( sscanf( event_read_buf, "%d", &handle_event_times )< 1 ){
				handle_event_times= 1;
			}
			if( sscanf( event_read_buf, "%lf", ascanf_ReadBufVal )< 1 ){
				set_NaN(*ascanf_ReadBufVal);
			}
		}

		w_context_t= XFindContext(theEvent->xany.display,
				 theEvent->xany.window,
				 win_context, (caddr_t *) &wi
		);
		f_context_t= XFindContext(theEvent->xany.display,
				 theEvent->xany.window,
				 frame_context, (caddr_t *) &frame
		);
		if( wi && !w_context_t ){
			if( wi->debugFlag== 1 ){
				debugFlag= True;
			}
			else if( wi->debugFlag== -1 ){
				debugFlag= False;
			}
		}
		dbF_cache= &dbF;
		if( debugFlag && debugLevel== -2 ){
			if( wi && !w_context_t ){
				fprintf( StdErr, "_Handle_An_Event(%s,%s): evt.xany.window=0x%lx wi->window=0x%lx\n",
					event_name(theEvent->type), caller,
					theEvent->xany.window, wi->window
				);
			}
			else if( w_context_t!= XCNOENT ){
				fprintf( StdErr, "_Handle_An_Event(%s,%s): window context error #%d\n",
					event_name(theEvent->type), caller,
					w_context_t
				);
			}
			if( !f_context_t ){
				fprintf( StdErr, "_Handle_An_Event(%s,%s): evt.xany.window=0x%lx frame->parent=0x%lx\n",
					event_name(theEvent->type), caller,
					theEvent->xany.window, frame->parent
				);
			}
			else if( f_context_t!= XCNOENT ){
				fprintf( StdErr, "_Handle_An_Event(%s,%s): frame context error #%d\n",
					event_name(theEvent->type), caller,
					f_context_t
				);
			}
			fflush( StdErr );
		}
		if( wi && !w_context_t && f_context_t ){
			if( wi->pid!= getpid() ){
				if( debugFlag ){
				  char *name;
					XFetchName( disp, theEvent->xany.window, &name );
					fprintf( StdErr, "_Handle_An_Event(%s,%s): window 0x%lx (%s) doesn't belong to us\n",
						event_name(theEvent->type), caller, theEvent->xany.window, name
					);
					fflush( StdErr );
					XFree( name );
				}
				debugFlag= dbF;
				dbF_cache= dbFc;
				Allow_DrawCursorCrosses= gADCC;
				return(0);
			}
			frames= (wi->label_IFrame.win)? 8 : 7;
			if( (unsigned long) &wi->hd_frame - (unsigned long) &wi->cl_frame == sizeof(xtb_frame) ){
				framelist= & wi->cl_frame;
			}
			else{
			  /* Padding makes it impossible to take the address of the 1st frame as a pointer
			   \ to an array of (8) frames..
			   */
			  static xtb_frame ff[8];
				ff[0]= wi->cl_frame;
				ff[1]= wi->hd_frame;
				ff[2]= wi->settings_frame;
				ff[3]= wi->info_frame;
				ff[4]= wi->label_frame;
				ff[5]= wi->ssht_frame;
				if( frames== 8 ){
					ff[6]= wi->label_IFrame;
					ff[7]= wi->YAv_Sort_frame;
				}
				else{
					ff[6]= wi->YAv_Sort_frame;
				}
				framelist= &ff[0];
			}
		}
		if( !f_context_t ){
			frames= 1;
			framelist= frame;
		}
		if( debugFlag && debugLevel== -2 && wi && frame ){
			fprintf( StdErr, "_Handle_An_Event(\"%s\"): wi=0x%lx frame=0x%lx (%s), framelist=0x%lx[%d] (%s)\n",
				caller, wi,
				frame, (frame && frame->description)? frame->description : "??",
				framelist, frames, (framelist && framelist->description)? framelist->description : "??"
			);
			fflush( StdErr );
		}
		xtb_return= XTB_NOTDEF;
		if( w_context_t ){
		  int handled= 0, Level= 1;
		  int ADCC= Allow_DrawCursorCrosses;
		  /* This means the event doesn't refer to anything in the win_context. Check if
		   \ we have a Dialog open, and pass the event to the respective handlers.
		   */
			Allow_DrawCursorCrosses= False;
			sprintf( ps_comment, "_Handle_An_Event(%d,\"%s\") %s #%ld (flushed %d)",
				level, caller, event_name(theEvent->type), theEvent->xany.serial, flushed
			);
			if( theSettingsWin_Info ){
				if( Handle_SD_Event( theEvent, &handled, (xtb_hret*) &xtb_return, &Level, 0) &&
					theSettingsWin_Info
				){
					CloseSD_Dialog( theSettingsWin_Info->SD_Dialog );
				}
			}
			if( !handled && xtb_return!= XTB_HANDLED && xtb_return!= XTB_STOP && thePrintWin_Info ){
				if( Handle_HO_Event( theEvent, &handled, (xtb_hret*) &xtb_return, &Level, 0) &&
					thePrintWin_Info
				){
					CloseHO_Dialog( thePrintWin_Info->HO_Dialog );
				}
			}
			  /* If still not handled, it might be an event for one of the frames in <framelist>	*/
			if( !handled && xtb_return!= XTB_HANDLED && xtb_return!= XTB_STOP ){
				xtb_return= xtb_dispatch(theEvent->xany.display, (!f_context_t)? frame->parent : theEvent->xany.window,
							frames, framelist, theEvent
				);
			}
			  /* Nothing found */
			debugFlag= dbF;
			dbF_cache= dbFc;
			Allow_DrawCursorCrosses= ADCC;
			Allow_DrawCursorCrosses= gADCC;
			return(0);
		}
		else{
		  int ADCC= Allow_DrawCursorCrosses;
			  /* Even if a context was found, it might be an event for one of the frames in <framelist>	*/
			Allow_DrawCursorCrosses= False;
			xtb_return= xtb_dispatch(theEvent->xany.display, (!f_context_t)? frame->parent : theEvent->xany.window,
						frames, framelist, theEvent
			);
			Allow_DrawCursorCrosses= ADCC;
			if( xtb_return!= XTB_NOTDEF || xtb_return== XTB_HANDLED || xtb_return== XTB_STOP ){
				debugFlag= dbF;
				dbF_cache= dbFc;
				Allow_DrawCursorCrosses= gADCC;
				return(0);
			}
		}
		if( !wi ){
			if( debugFlag ){
			  char *name;
				XFetchName( disp, theEvent->xany.window, &name );
				fprintf( StdErr, "_Handle_An_Event(%s,%s): can't find LocalWin for window 0x%lx (%s)\n",
					event_name(theEvent->type), caller, theEvent->xany.window, name
				);
				fflush( StdErr );
				XFree( name );
			}
			debugFlag= dbF;
			dbF_cache= dbFc;
			Allow_DrawCursorCrosses= gADCC;
			return(0);
		}
		if( debugFlag){
			fprintf( StdErr, "_Handle_An_Event(%d,\"%s\") %s #%lu %02d:%02d:%02d (flushed %d)%s\n",
				level, caller, event_name(theEvent->type), theEvent->xany.serial,
				wi->parent_number, wi->pwindow_number, wi->window_number,
				flushed,
				(wi->silenced)? " silenced, thus only key & button events handled!" : ""
			);
			fflush( StdErr );
		}

		if( size_window_print ){
			wi->dev_info.resized= 0;
			ZoomWindow_PS_Size( wi, 0, 0, 1 );
			size_window_print= 0;
		}
		switch (theEvent->type ){
			case ClientMessage:{
				if( debugFlag ){
					fprintf( StdErr, "_Handle_An_Event(\"%s\") client message 0x%ld (%s), format=%d, window=%ld (=?%ld)\n",
						caller, theEvent->xclient.message_type, XGetAtomName(disp, theEvent->xclient.message_type),
						theEvent->xclient.format, theEvent->xclient.window, wi->window
					);
				}
				if( wi->window== theEvent->xclient.window && theEvent->xclient.data.l[0]== wm_delete_window &&
					strcmp( XGetAtomName(disp, theEvent->xclient.message_type), "WM_PROTOCOLS")== 0
				){
					if( Num_Windows== 1 ){
						if( xtb_error_box( wi->window,
							"\001This is the only/last window\n"
							"Use ^C or the Quit button to exit\n",
							"Notification"
							)> 0
						){
							goto exit_the_programme;
						}
					}
					else{
					  /* Delete this window */
						DelWindow(theEvent->xkey.window, wi);
						return_code= 1;
					}
				}
				break;
			}
			case MotionNotify: if( CursorCross ){
			  Window dum;
			  int dum2, curX, curY;
			  unsigned int mask_rtn;
			  double cx, cy;

				XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,
					  &curX, &curY, &mask_rtn
				);

				mask_rtn= xtb_Mod2toMod1( mask_rtn );
				cx= Reform_X( wi, TRANX(curX), TRANY(curY) );
				cy= Reform_Y( wi, TRANY(curY), TRANX(curX) );
#if DEBUG > 2
				fprintf( StdErr, "%g\t%g\n",
					Reform_X( wi, wi->XUnitsPerPixel, wi->YUnitsPerPixel ),
					Reform_Y( wi, wi->YUnitsPerPixel, wi->XUnitsPerPixel )
				);
#endif
				DrawCCrosses( wi, theEvent, cx, cy, curX, curY, NULL, "_Handle_An_Event()" );
				break;
			}
			case CirculateNotify:
				if( debugFlag){
					fprintf( StdErr, "CirculateNotify: ");
					if( theEvent->xcirculate.send_event ){
						fprintf( StdErr, " sendevent");
					}
					fprintf( StdErr,
						" \"%s\" window=%u,%u (#%02d.%02d.%02d) place=%s\n",
					   caller,
					   theEvent->xcirculate.event, theEvent->xcirculate.window,
					   wi->parent_number, wi->pwindow_number, wi->window_number,
					   (theEvent->xcirculate.place== PlaceOnTop)? "on top" : "on bottom"
					);
					fflush( StdErr );
				}
				break;
			case ConfigureNotify:{

				if( debugFlag){
				 XSizeHints sh;
				 long sr;
					XGetWMSizeHints( disp, wi->window, &sh, &sr, XA_WM_SIZE_HINTS );
					fprintf( StdErr, "ConfigureNotify: ");
					if( theEvent->xconfigure.send_event ){
						fprintf( StdErr, " sendevent");
					}
					fprintf( StdErr,
						" \"%s\" win=0x%x (#%02d.%02d.%02d) x=%d y=%d width=%d height=%d (WM +%d+%d) above=%ld\n",
					   caller,
					   theEvent->xconfigure.window,
					   wi->parent_number, wi->pwindow_number, wi->window_number,
					   theEvent->xconfigure.x, theEvent->xconfigure.y,
					   theEvent->xconfigure.width, theEvent->xconfigure.height,
					   sh.x, sh.y,
					   theEvent->xconfigure.above
					);
					fflush( StdErr );
				}
				if( wi->WindowList_Entry ){
					wi->WindowList_Entry->above= theEvent->xconfigure.above;
				}
				else{
					xtb_error_box( wi->window, "Loose window!", "Oops!!" );
				}
				if( !wi->silenced ){
					AdaptWindowSize( wi, theEvent->xany.window, 0, 0 );
					if( wi->redraw && theEvent->xconfigure.window== wi->window ){
					  /* discard other events (at least as many ExposeEvents as subwindows...), and redraw window
						xtb_XSync( disp, True );
					   */
						handle_expose_now= True;
						if( !wi->redrawn && wi->mapped ){
							  /* reinitialise some axis-related memory variables. It is hoped that
							   \ this helps in overcoming the sometimes incorrect display of Y-axis numbers
							   \ (partly outside the window) by resizing the window.
							   */
							wi->axis_stuff.__XLabelLength= 0;
							wi->axis_stuff.__YLabelLength= 0;
							XLabelLength= 0;
							YLabelLength= 0;
							goto handle_expose;
							return_code= 1;
						}
					}
				}
				break;
			}
			case UnmapNotify:
				if( theEvent->xany.window== wi->window && !Ignore_UnmapNotify ){
					XGIconify( wi );
				}
				break;
			case MapNotify:
				if( theEvent->xany.window== wi->window ){
				  LocalWindows *WL= NULL;
				  LocalWin *lwi;
				  int hit= handle_event_times;
					if( hit== 0 ){
						WL= WindowList;
						lwi= WL->wi;
						event_read_buf[0]= '\0';
						set_NaN( *ascanf_ReadBufVal );
					}
					else{
						lwi= wi;
					}
					do{
						if( !lwi->mapped ){
							XMapRaised( disp, lwi->window );
							lwi->mapped= 1;
							lwi->redrawn= 0;
							if( !UnmappedWindowTitle ){
								SetWindowTitle( lwi, lwi->draw_timer.HRTot_T );
							}
							if( !lwi->silenced ){
/* 								RedrawNow( lwi );	*/
/* 								return_code= 1;	*/
							}
						}
						if( hit== 0 ){
							if( lwi->SD_Dialog && !lwi->SD_Dialog->mapped ){
								XMapRaised( disp, lwi->SD_Dialog->win );
								lwi->SD_Dialog->mapped= 1;
							}
							if( lwi->HO_Dialog && !lwi->HO_Dialog->mapped ){
								XMapRaised( disp, lwi->HO_Dialog->win );
								lwi->HO_Dialog->mapped= 1;
							}
						}
						XRaiseWindow( disp, lwi->window );
						XG_XSync( disp, False );
						if( WL ){
							WL= WL->next;
							lwi= (WL)? WL->wi : 0;
						}
					} while( WL && lwi );
				}
				break;
			case VisibilityNotify:
				if( debugFlag){
					fprintf( StdErr, "VisibilityNotify: ");
					if( theEvent->xvisibility.send_event ){
						fprintf( StdErr, " sendevent");
					}
					fprintf( StdErr, " \"%s\" win=0x%x (0x%lx=#%02d.%02d.%02d) state=%d\n",
					   caller,
					   theEvent->xvisibility.window, wi->window,
					   wi->parent_number, wi->pwindow_number, wi->window_number,
					   theEvent->xvisibility.state
					);
					fflush( StdErr );
				}
				if( theEvent->xvisibility.window== wi->window && !wi->silenced ){
					switch( theEvent->xvisibility.state ){
						case VisibilityPartiallyObscured:
							wi->redraw= True;
						case VisibilityFullyObscured:
							  /* Don't change anything in this case.	*/
							break;
						default:
							wi->redraw= True;
							break;
					}
				}
				break;
			case Expose:{
				if( debugFlag){
					fprintf( StdErr, "expose: ");
					if( theEvent->xexpose.send_event ){
						fprintf( StdErr, " sendevent");
					}
					fprintf( StdErr, " \"%s\" win=0x%x (#%02d.%02d.%02d) x=%d y=%d width=%d height=%d count=%d\n",
					   caller,
					   theEvent->xexpose.window,
					   wi->parent_number, wi->pwindow_number, wi->window_number,
					   theEvent->xexpose.x, theEvent->xexpose.y,
					   theEvent->xexpose.width, theEvent->xexpose.height,
					   theEvent->xexpose.count
					);
					fflush( StdErr );
				}
				handle_expose_now= (theEvent->xexpose.count<= 0)? True : False;
				if( level> 1 ){
					if( !wi->redraw )
						wi->redraw= 1;
				}
				else if( theEvent->xexpose.count <= 0 || wi->redraw ){
handle_expose:;
					if( theEvent->xany.window== wi->window ){
						if( !wi->redraw && wi->redrawn ){
							if( !wi->clipped ){
								SetXClip( wi, theEvent->xexpose.x, theEvent->xexpose.y,
										   theEvent->xexpose.width, theEvent->xexpose.height
								);
							}
							else{
								AddXClip( wi, theEvent->xexpose.x, theEvent->xexpose.y,
										   theEvent->xexpose.width, theEvent->xexpose.height
								);
							}
							wi->clipped= 1;
						}
						else{
							wi->clipped= 0;
						}
						init_X(wi->dev_info.user_state);
						if( !wi->silenced && handle_expose_now ){
						  XEvent dum;
							if( wi->redraw ){
							  /* 20050110: remove all pending events that would cause another redraw of this window:  */
								XG_XSync( disp, False );
								while( XCheckWindowEvent(disp, theEvent->xexpose.window, ExposureMask|VisibilityChangeMask, &dum) );
							}
							if( debugFlag ){
								fprintf( StdErr, "handle_expose: redrawing %swindow, #%d\n",
									(wi->clipped)? "clipped " : "", wi->redrawn+1
								);
							}
							RedrawNow( wi );
							if( wi->delete_it!= -1 ){
								SetXClip( wi, 0, 0, wi->dev_info.area_w, wi->dev_info.area_h );
								wi->clipped= 0;
								if( !XEventsQueued( disp, QueuedAfterFlush) ){
									while( wi->redraw && wi->redraw!= -2 && !wi->silenced ){
										DrawWindow( wi);
									}
								}
							}
							return_code= 1;
						}
					}
				}
				else if( !wi->redraw ){
				  /* make sure it is redrawn completely at least once	*/
#ifndef SMART_REFRESH
					wi->redraw= 1;
#else
					if( !wi->raw_display || theEvent->xexpose.count> 2 ){
						wi->redraw= 1;
					}
					else if( !wi->clipped ){
						SetXClip( wi, theEvent->xexpose.x, theEvent->xexpose.y,
								   theEvent->xexpose.width, theEvent->xexpose.height
						);
					}
					else{
						AddXClip( wi, theEvent->xexpose.x, theEvent->xexpose.y,
								   theEvent->xexpose.width, theEvent->xexpose.height
						);
					}
					wi->clipped= 1;
#endif
				}
				if( wi->redraw ){
					wi->dont_clear= False;
				}
				break;
			}
			case KeyPress:{
				nbytes = XLookupString( &theEvent->xkey, keys, MAXKEYS,
						   (KeySym *) 0, (XComposeStatus *) 0);
				{  unsigned int buttonmask= xtb_Mod2toMod1(theEvent->xbutton.state);
				   int nocontrol= 0, fnkey= 0;
				   Boolean hit= True, ModMem;

					handle_event_times= 1;
					keysymbuffer[0]= XLookupKeysym( (XKeyPressedEvent*) &theEvent->xkey, 0);
					  /* we handle only one keypress here
					   \ but we check event_read_buf to see if a number was
					   \ entered before
					   */
					if( sscanf( event_read_buf, "%d", &handle_event_times )< 1 ){
						handle_event_times= 1;
					}
					else if( handle_event_times< 0 ){
					  /* Entering the number as negative does not require the modifier
					   \ keys to be held during all <handle_event_times> handle_event_times the event is processed.
					   */
						handle_event_times*= -1;
						ModMem= True;
					}
					else{
						ModMem= False;
					}
					if( sscanf( event_read_buf, "%lf", ascanf_ReadBufVal )< 1 ){
						set_NaN(*ascanf_ReadBufVal);
					}
					{ Boolean fn= True;
					  int mask= 0;
						switch( keysymbuffer[0] ){
							case XK_F1:
								fnkey= 0;
								break;
							case XK_F2:
								fnkey= 1;
								break;
							case XK_F3:
								fnkey= 2;
								break;
							case XK_F4:
								fnkey= 3;
								break;
							case XK_F5:
								fnkey= 4;
								break;
							case XK_F6:
								fnkey= 5;
								break;
							case XK_F7:
								fnkey= 6;
								break;
							case XK_F8:
								fnkey= 7;
								break;
							case XK_F9:
								fnkey= 8;
								break;
							case XK_F10:
								fnkey= 9;
								break;
							case XK_F11:
								fnkey= 10;
								break;
							case XK_F12:
								fnkey= 11;
								break;
							default:
								fn= False;
								break;
						}
						if( CheckMask(buttonmask, ControlMask) ){
							mask|= 1;
						}
						if( CheckMask(buttonmask, Mod1Mask) ){
							mask|= 2;
						}
						if( CheckMask(buttonmask, ShiftMask) ){
							mask|= 4;
						}
						if( fn ){
							fnkey= 8*256+ mask* 12+ fnkey;
							if( !nbytes ){
								nbytes= 1;
								keys[0]= '\0';
							}
						}
						else if( nbytes ){
						  KeySym ksb= keysymbuffer[0];
							switch( mask ){
								case 0:
									fnkey= keys[0];
									break;
								case 1:{
								  char c= toupper(keys[0]);
									if( c>='A' && c<= 'Z' ){
										fnkey= c- 'A'+ 1;
									}
									else if( ksb< 256 && isprint(ksb) ){
									  char d= toupper(ksb);
										if( d>='A' && d<= 'Z' ){
											fnkey= d- 'A'+ 1;
										}
										else{
											fnkey= mask* 256+ d;
										}
									}
									else{
										fnkey= keys[0];
									}
									break;
								}
								case 4:
									fnkey= toupper( keys[0] );
									break;
								default:
									if( iscntrl(keys[0]) ){
									  char c= toupper( keys[0] );
										if( c>='A' && c<= 'Z' ){
											fnkey= c- 'A'+ 1;
										}
										else if( ksb< 256 && isprint(ksb) ){
											fnkey= mask* 256+ toupper( ksb );
										}
									}
									else{
										fnkey= mask* 256+ toupper( keys[0] );
									}
									break;
							}
						}
					}
					if( debugFlag ){
						fprintf( StdErr,
							"_Handle_An_Event(): \"%s\" event_read_buf=\"%s\": "
							"%d handle_event_times %s; mask=%s; keys[%d]=\"%s\" (keycode %u); fnkey=%d\n",
							caller,
							event_read_buf, handle_event_times, XKeysymToString(keysymbuffer[0]), xtb_modifiers_string(buttonmask),
							nbytes, keys, theEvent->xkey.keycode, fnkey
						);
						fflush( StdErr );
					}
					wi->halt= 0;
					  /* loop <handle_event_times> handle_event_times, or when ^R has been hit (which resets event_read_buf and hence <handle_event_times>),
					   \ AND a key has been pressed.
					   */
/* 					for( idx = 0;  (keys[0]== 0x12 || keys[0]== 'r' || keys[0]== 'R' || idx < handle_event_times) && hit;  idx++)	*/
					for( idx = 0;  (handle_event_times== 0 || idx < handle_event_times) && hit && !wi->halt;  idx++)
					{
					  int i, flush;
					  char buf[256];
					  Window dum;
					  int dum2, curX, curY;
					  unsigned int mask_rtn;
						if( handle_event_times> 1 ){
							XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,
								  &curX, &curY, &mask_rtn
							);
							mask_rtn= xtb_Mod2toMod1( mask_rtn );
							mask_rtn&= ~LockMask;
							sprintf( buf, "\"%s\"(%c) %d", XKeysymToString(keysymbuffer[0]), keys[0], idx );
							XDrawString( disp, wi->window, msgGC(wi),
								curX+ 2, curY+ 2, buf, strlen(buf)
							);
							XG_XSync( disp, False );
							if( idx && !ModMem ){
								buttonmask= mask_rtn;
							}
						}
						flush= 0;
						if( keysymbuffer[0]== XK_Right || keysymbuffer[0]== XK_Left ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								if( CheckMask(buttonmask, ShiftMask) ){
								  /* we lwill cycle the collection of drawn sets	*/
									cycle_drawn_sets( lwi, (keysymbuffer[0]== XK_Right)? 1 : -1 );
									if( CheckMask( buttonmask, ControlMask) ){
										Add_mStats( lwi );
										nocontrol= True;
										Boing(10);
									}
								}
								else if( CheckMask(buttonmask, Mod1Mask) ){
								  /* we lwill cycle the collection of highlighted sets	*/
									if( !cycle_highlight_sets( lwi, (keysymbuffer[0]== XK_Right)? 1 : -1 ) ){
									  /* This means no sets were highlighted. Highlight the first or last displayed	*/
									  int first= -1, last= -1, i;
										for( i= 0; i< setNumber; i++ ){
											if( draw_set( lwi, i) ){
												if( first< 0 ){
													first= i;
												}
												last= i;
											}
										}
										if( keysymbuffer[0]== XK_Right && first>= 0 ){
											lwi->legend_line[first].highlight= True;
										}
										else if( keysymbuffer[0]== XK_Left && last>= 0 ){
											lwi->legend_line[last].highlight= True;
										}
									}
									if( CheckMask( buttonmask, ControlMask) ){
										Add_mStats( lwi );
										nocontrol= True;
										Boing(10);
									}
								}
								else{
								  /* cycle the displayed set only	*/
									cycle_plot_only_set( lwi, (keysymbuffer[0]==XK_Right)? 1 : -1 );
								}
								lwi->printed= 0;
								RedrawNow( lwi );
								flush= 1;
								if( lwi->SD_Dialog ){
									set_changeables(2,False);
								}
								return_code= 1;
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keysymbuffer[0]== XK_Up || keysymbuffer[0]== XK_Down ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								if( CheckMask( buttonmask, ShiftMask ) ){
									cycle_plot_only_group( lwi, (keysymbuffer[0]== XK_Up)? 1 : -1 );
								}
								else{
									cycle_plot_only_file( lwi, (keysymbuffer[0]== XK_Up)? 1 : -1 );
								}
								lwi->printed= 0;
								RedrawNow( lwi );
								return_code= 1;
								flush= 1;
								if( lwi->SD_Dialog ){
									set_changeables(2,False);
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keysymbuffer[0]== XK_Return || keysymbuffer[0]== XK_KP_Enter ){
						  int plot_only_set0, plot_only_group, plot_only_file;
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( CheckMask(buttonmask, ShiftMask) || CheckMask( buttonmask, Mod1Mask) ){
							   /* set startingpoint	*/
							  int i;
								for( i= 0; i< setNumber && !draw_set(wi, i); i++ ){
								}
								if( draw_set( wi, i ) ){
									plot_only_set0= i;
									plot_only_group= wi->group[i];
									plot_only_file= wi->fileNumber[i];
									Boing(5);
									return_code= 1;
									flush= 1;
								}
								if( handle_event_times== 0 ){
									WL= WindowList;
									lwi= WL->wi;
								}
								else{
									lwi= wi;
								}
								do{
									lwi->plot_only_set0= plot_only_set0;
									lwi->plot_only_group= plot_only_group;
									lwi->plot_only_file= plot_only_file;
									if( CheckMask( buttonmask, Mod1Mask) ){
									  /* set group & files	*/
										  /* cycle_plot_only_file() is a little bizarre...	*/
										cycle_plot_only_file( lwi, -1 );
										cycle_plot_only_group( lwi, 0 );
										lwi->printed= 0;
										RedrawNow( lwi );
									}
									if( lwi->SD_Dialog ){
										set_changeables(2,False);
									}
									if( WL ){
										WL= WL->next;
										lwi= (WL)? WL->wi : 0;
									}
								} while( WL && lwi );
							}
							if( return_code && keys[0]== '\n' ){
								keys[0]= '\0';
							}
						}
						else if( keysymbuffer[0]== 'a' && CheckMask(buttonmask, Mod1Mask) ){
						  double aspect= 0;
						  char *c= event_read_buf;
						  int n;
							while( *c && !index( "0123456789+-.=", *c) ){
								c++;
							}
							if( c[0]== '=' && !c[1] ){
								set_NaN(aspect);
								n= 1;
							}
							else{
								if( (n= sscanf( c, "%lf", &aspect ))< 1 ){
									aspect= 1;
								}
							}

							if( (n== 1 && aspect) || (n< 1 && !wi->aspect_ratio) ){
								ZoomWindow_PS_Size( wi, 1, aspect, 1);
							}
							else{
								wi->dev_info.resized= 0;
								wi->aspect_ratio= 0;
								wi->redraw= 1;
								wi->halt= 0;
								wi->draw_count= 0;
							}

							event_read_buf[0]= '\0';
							set_NaN( *ascanf_ReadBufVal );
							handle_event_times= 1;
							if( keys[0]== 'a' ){
								keys[0]= '\0';
							}
							return_code= 1;
						}
						else if( keysymbuffer[0]== 'n' && CheckMask(buttonmask, Mod1Mask) ){
							Restart( wi, StdErr );
						}
						else if( keysymbuffer[0]== 't' && CheckMask(buttonmask, Mod1Mask) ){
							if( keys[0]== 't' ){
								keys[0]= '\0';
							}
							if( CheckMask( buttonmask, ShiftMask) ){
								process_hist(wi->window);
							}
							else{
							  static char *expr= NULL;
							  int expr_len= 0;
								if( !expr || LMAXBUFSIZE+1> expr_len ){
									expr_len= LMAXBUFSIZE+1;
									expr= (char*) XGrealloc( expr, expr_len* sizeof(char));
								}
								if( expr ){
									interactive_parse_string( wi, expr, -2, LMAXBUFSIZE, "", False, 0 );
								}
								else{
									xtb_error_box( wi->window, "Can't get expression buffer", "Error" );
									expr_len= 0;
								}
							}
							return_code= 1;
						}
						else if( keysymbuffer[0] == 't' ){
							return_code= 1;
							if( CheckMask( buttonmask, ShiftMask) ){
								TileWindows( 'd', CheckMask( buttonmask, ControlMask) );
							}
							else{
								TileWindows( 'u', CheckMask( buttonmask, ControlMask) );
							}
							if( keys[0]== 't' || keys[0]== 'T' || keys[0]== 0x14 ){
								keys[0]= '\0';
							}
						}
						else if( keysymbuffer[0]== 'q' && CheckMask(buttonmask, Mod1Mask) ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								lwi->use_transformed= ! lwi->use_transformed;
								xtb_bt_swap( lwi->settings );
								lwi->printed= 0;
								RedrawNow( lwi );
								return_code= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							if( keys[0]== 'q' ){
								keys[0]= '\0';
							}
						}
						else if( keysymbuffer[0]== 'r' && CheckMask(buttonmask, Mod1Mask) ){
						  int r, sn= setNumber, err;
						  DataSet *ridge_set= NULL;
						  static int fn= -1;
						  Time_Struct timer;
						  XdbeBackBuffer dbebuf= wi->XDBE_buffer;
							xtb_bt_swap( wi->settings );
							if( CheckMask( buttonmask, ShiftMask ) ){
								ridge_set= &AllSets[setNumber-1];
								setNumber-= 1;
								if( ridge_set->numPoints> 0 ){
									if( NewSet( wi, &ridge_set, 0 )== -1 ){
										ridge_set= NULL;
									}
								}
								else{
									setNumber= sn;
								}
							}
							if( ridge_set ){
								if( ridge_set->setName ){
									xfree( ridge_set->setName );
									ridge_set->setName= NULL;
								}
								xfree( ridge_set->XUnits );
								xfree( ridge_set->YUnits );
								xfree( ridge_set->fileName );
								xfree( ridge_set->titleText );
								wi->draw_set[ridge_set->set_nr]= 0;
							}

							  /* If we're adding a set, make sure user sees what the
							   \ result is based on. Else, leave the window as it is -
							   \ he might have left a "disabled" set around for comparison
							   */
							if( ridge_set ){
								if( wi->redraw ){
									RedrawNow( wi );
								}
							}
							else if( wi->XDBE_buffer ){
								  /* Temporarily set the doublebuffer to NULL, and call Set_X() to ensure
								   \ all output goes to wi->window, not to wi->XDBE_buffer (in that case,
								   \ we wouldn't see a thing...!
								   */
								wi->XDBE_buffer= (XdbeBackBuffer) 0;
								Set_X( wi, &(wi->dev_info) );
							}
							Elapsed_Since( &timer, True );
							if( handle_event_times== 2 ){
								err= Show_Ridges2( wi, ridge_set );
							}
							else{
								err= Show_Ridges( wi, ridge_set );
							}
							Elapsed_Since( &timer, False );
							if( err ){
								Destroy_Set( ridge_set, False );
								ridge_set= NULL;
							}
							if( ridge_set ){
							  LocalWindows *WL= WindowList;
							  ALLOCA( name, char, LMAXBUFSIZE+2, name_len);
							  char *nbuf;
							  char pc[]= "#x01Request";
								if( CheckMask( buttonmask, ControlMask) ){
									strncpy( name, ridge_set->setName, LMAXBUFSIZE);
									name[LMAXBUFSIZE]= '\0';
									if( (nbuf= xtb_input_dialog( wi->window, name, 80, LMAXBUFSIZE,
											"Please enter name for new ridges set:", parse_codes(pc),
											False,
											NULL, NULL, NULL, NULL, NULL, NULL
										))
									){
										xfree( ridge_set->setName );
										ridge_set->setName= XGstrdup( name );
										if( nbuf!= name ){
											xfree( nbuf );
										}
									}
									nocontrol= 1;
								}
								ridge_set->show_legend= 1;
								ridge_set->show_llines= 1;
								ridge_set->draw_set= 1;
								ridge_set->error_type= 0;
								ridge_set->has_error= 1;
								ridge_set->use_error= 0;
								ridge_set->markFlag= 1;
								ridge_set->pixelMarks= 1;
								ridge_set->barFlag= 0;
								ridge_set->overwrite_marks= 1;
								ridge_set->noLines= 1;
								ridge_set->floating= False;
								ridge_set->elineWidth= (ridge_set->lineWidth/2.0);
								if( !ridge_set->fileName ){
									ridge_set->fileName= XGstrdup( "Bifurcations");
								}
								  /* Update the relevant information on this set in the LocalWin
								   \ structures.
								   */
								wi->draw_set[setNumber]= 1;
								wi->redraw= 1;
								if( fn!= fileNumber+ 1 ){
									fn= fileNumber+ 1;
								}
								while( WL ){
								  LocalWin *lwi= WL->wi;
									lwi->numVisible[setNumber]= r;
									lwi->fileNumber[setNumber]= ridge_set->fileNumber= fn;
									if( lwi->fileNumber[setNumber]!= lwi->fileNumber[setNumber-1] ){
										lwi->new_file[setNumber]= ridge_set->new_file= 1;
									}
									else{
										lwi->new_file[setNumber]= ridge_set->new_file= 0;
									}
									lwi->error_type[setNumber]= 0;
									if( draw_set( lwi, setNumber) ){
										lwi->halt= 0;
										lwi->redraw= 1;
									}
									WL= WL->next;
								}
								setNumber= ridge_set->set_nr+ 1;
							}
							else{
								setNumber= sn;
								if( (wi->XDBE_buffer= dbebuf) ){
								  /* Now restore the backbuffer..!*/
									Set_X( wi, &(wi->dev_info) );
								}
							}
/* 							SetWindowTitle( wi, timer.Tot_Time );	*/
							SetWindowTitle( wi, timer.HRTot_T );
							return_code= 1;
						}
						else if( keysymbuffer[0] == 'r' && CheckMask(buttonmask, ControlMask) ){
							  /* "^R"	*/
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
						  int i, I= (handle_event_times> 1)? handle_event_times : 1;
						  int ss;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							event_read_buf[0]= '\0';
							set_NaN( *ascanf_ReadBufVal );
							handle_event_times= 1;
							do{
								nocontrol= 1;
								for( i= 0; i< I && lwi->delete_it!= -1; i++ ){
									xtb_bt_swap( lwi->settings );
									  /* 980528	*/
									lwi->event_level= 0;
									if( I> 1 ){
										lwi->animate= 1;
									}
									lwi->clipped= 0;
									if( CheckMask( buttonmask, ShiftMask) ){
										ss= lwi->data_sync_draw;
										lwi->data_sync_draw= True;
									}
									{ XEvent dum;
									  /* 20050110: a manual redraw cancels all pending Expose events that would
									   \ cause (subsequent) redraws. (NB: events generated after this point are not
									   \ affected, of course.)
									   */
										XG_XSync( disp, False );
										while( XCheckWindowEvent(disp, lwi->window, ExposureMask|VisibilityChangeMask, &dum) );
									}
									RedrawNow( lwi );
									if( lwi->delete_it!= -1 ){
										xtb_bt_swap( lwi->settings );
										if( CheckMask( buttonmask, ShiftMask) ){
											lwi->data_sync_draw= ss;
										}
									}
								}
								return_code= 1;
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							if( keys[0]== 0x12 || keys[0]== 'r' || keys[0]== 'R' ){
								keys[0]= '\0';
							}
							return_code= 1;
						}
						else if( keysymbuffer[0]== 's' && CheckMask(buttonmask, Mod1Mask) ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
							  int ut= lwi->use_transformed;
							  int fa= lwi->fit_after_draw;
								if( tolower(event_read_buf[0])== 'x' ){
									lwi->FitOnce= 'x';
									memmove( event_read_buf, &event_read_buf[1], sizeof(event_read_buf)-1 );
								}
								else if( tolower(event_read_buf[0])== 'y' ){
									lwi->FitOnce= 'y';
									memmove( event_read_buf, &event_read_buf[1], sizeof(event_read_buf)-1 );
								}
								else{
									lwi->FitOnce= 1;
								}
								if( CheckMask(buttonmask, ShiftMask) ){
									lwi->use_transformed= 0;
								}
								if( CheckMask(buttonmask, ControlMask) ){
									lwi->fit_after_draw= !lwi->fit_after_draw;
								}
								RedrawNow(lwi);
								lwi->use_transformed= ut;
								lwi->fit_after_draw= fa;
/* 								SetWindowTitle( lwi, lwi->draw_timer.Tot_Time );	*/
								SetWindowTitle( lwi, lwi->draw_timer.HRTot_T );
								return_code= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							if( keys[0]== 's' || keys[0]== 'S' ){
								keys[0]= '\0';
							}
							else if( CheckMask(buttonmask, ControlMask) && keys[0]== 0x13 ){
								keys[0]= '\0';
							}
							return_code= 1;
						}
						else if( keysymbuffer[0]== 'c' && CheckMask(buttonmask, Mod1Mask) ){
							SS_Reset_(SS_mX);
							SS_Reset_(SS_mY);
							SS_Reset_(SS_mE);
							SS_Reset_(SS_mI);
							SAS_mO.Gonio_Base= radix;
							SAS_mO.Gonio_Offset= radix_offset;
							SAS_mO.exact= 0;
							SAS_Reset_(SAS_mO);
							SS_Reset_(SS_mLY);
							SS_Reset_(SS_mHY);
							SS_Reset_(SS_mMO);
							SS_Reset_(SS_mSO);
							SS_Reset_(SS_mPoints);
							if( keys[0]== 'c' ){
								keys[0]= '\0';
							}
							Boing(10);
							return_code= 1;
						}
						else if( keysymbuffer[0]== 'i' && CheckMask(buttonmask, Mod1Mask) ){
							Add_mStats( wi );
							if( keys[0]== 'i' ){
								keys[0]= '\0';
							}
							Boing(10);
							return_code= 1;
						}
						else if( (keysymbuffer[0]== 'd' || keysymbuffer[idx]== 'D') &&
								CheckMask(buttonmask, (Mod1Mask|ShiftMask|ControlMask))
						){
							for( i= 0; i< setNumber; i++ ){
								if( wi->draw_set[i] ){
									RedrawSet( i, 0 );
									Destroy_Set( &AllSets[i], False );
									wi->draw_set[i]= 0;
								}
							}
							CleanUp_Sets();
							XG_XSync( disp, True );
							if( wi->SD_Dialog ){
								if( !update_SD_size() ){
									format_SD_Dialog( wi->SD_Dialog, 0 );
								}
							}
							wi->redraw= 1;
							wi->plot_only_set0= -1;
							wi->plot_only_file= -1;
							handle_event_times= 1;
							keys[0]= '\0';
							xtb_error_box( wi->window, "Data of displayed sets were deleted\n"
								"Empty sets before (a) non-empty set(s) still take slots\n"
								"Showing all previously hidden data\n", "Notice"
							);
							return_code= 1;
							goto reverse_drawn_sets;
						}
						else if( keysymbuffer[0] == 'd' && CheckMask(buttonmask, ShiftMask) ){
							/* Handle creating a new window */
#ifdef ZOOM
						  pid_t cpid= 0;
							if( CheckMask( buttonmask, ControlMask) || CheckMask( buttonmask, Mod1Mask) ){
								  /* I'd like to do this, with fork(), but that causes problems with the X connections?!
								   \ Probably something like having the same windows open in 2 applications...
								   */
								  /* So we do it like this...	*/
								Duplicate_Visible_Sets(wi, (CheckMask(buttonmask,ControlMask))? True : False );
								cpid= -1;
							}
							else{
								cpid= 0;
							}
							if( cpid== 0 ){
							  Cursor curs= zoomCursor;
								HandleMouse(Window_Name,
											  NULL,
											  wi, NULL, &curs
								);
							}
							if( cpid> 0 && wi->redraw ){
								RedrawNow( wi );
							}
							return_code= 1;
							if( keys[0]== 'D' || keys[0]== CONTROL_D ){
								keys[0]= '\0';
							}
#endif
						}
						else if( keysymbuffer[0] == '4' && CheckMask(buttonmask, ShiftMask|ControlMask) ){
							display_ascanf_statbins( wi );
							return_code= 1;
							if( keys[0]== '$' || keys[0]== '4' ){
								keys[0]= '\0';
							}
						}
						else if( keysymbuffer[ 0 ] == 'm' && CheckMask(buttonmask, Mod1Mask) ){
						  /* unmark displayed sets.	*/
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->settings );
								for( i= 0; i< setNumber; i++ ){
									lwi->mark_set[i]-= (lwi->draw_set[i])? 1 : 0;
								}
								xtb_bt_swap( lwi->settings );
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							nocontrol= 1;
							Boing(1);
							if( keys[0]== 'm' ){
								keys[0]= '\0';
							}
							handle_event_times= 1;
						}
						else if( keysymbuffer[ 0 ] == 'h' && CheckMask(buttonmask, Mod1Mask) ){
						  /* unhighlight displayed sets.	*/
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->settings );
								for( i= 0; i< setNumber; i++ ){
									if( lwi->draw_set[i] ){
										lwi->legend_line[i].highlight= 0;
										lwi->redraw= 1;
									}
								}
								xtb_bt_swap( lwi->settings );
								nocontrol= 1;
								if( lwi->redraw ){
									lwi->halt= 0;
									lwi->draw_count= 0;
									RedrawNow( lwi );
								}
								if( lwi->SD_Dialog ){
									set_changeables(2,False);
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							if( keys[0]== 'h' ){
								keys[0]= '\0';
							}
							handle_event_times= 1;
						}
						else if( keysymbuffer[ 0 ] == 'z' && CheckMask(buttonmask, Mod1Mask) ){
							if( ShiftUndo.set ){
							  double x= ShiftUndo.x, y= ShiftUndo.y;
								if( ShiftUndo.whole_set ){
								  int idx, i, np= 0;
								  DataSet *this_set= ShiftUndo.set;
									for( idx= 0; idx< ShiftUndo.sets; idx++ ){
										if( ShiftUndo.sets> 1 ){
											this_set= ((DataSet**)ShiftUndo.set)[idx];
										}
										  /* Undo the previous wholeset shift
										   \ Interpret the x,y fields as amounts of shift (dx,dy)
										   */
										for( i= 0; i< this_set->numPoints; i++ ){
											this_set->columns[this_set->xcol][i]-= x;
											this_set->columns[this_set->ycol][i]-= y;
										}
										this_set->displaced_x-= x;
										this_set->displaced_y-= y;
										RedrawSet( this_set->set_nr, 0 );
										np+= i;
									}
									RedrawSet( this_set->set_nr, 1 );
									  /* Make sure that the next time we redo the shift
									   \ by negating the x,y amounts of shift.
									   */
									ShiftUndo.x*= -1;
									ShiftUndo.y*= -1;
									LastActionDetails[0]= 1.1;
									LastActionDetails[1]= -x;
									LastActionDetails[2]= -y;
									LastActionDetails[3]= np;
									LastActionDetails[4]= ShiftUndo.sets;
								}
								else{
									ShiftUndo.x= ShiftUndo.set->columns[ShiftUndo.set->xcol][ShiftUndo.idx];
									ShiftUndo.y= ShiftUndo.set->columns[ShiftUndo.set->ycol][ShiftUndo.idx];
									ShiftUndo.set->columns[ShiftUndo.set->xcol][ShiftUndo.idx]= x;
									ShiftUndo.set->columns[ShiftUndo.set->ycol][ShiftUndo.idx]= y;
									RedrawSet( ShiftUndo.set->set_nr, 1 );
									LastActionDetails[0]= 1.1;
									LastActionDetails[1]= x- ShiftUndo.x;
									LastActionDetails[2]= y- ShiftUndo.y;
									LastActionDetails[3]= 1;
									LastActionDetails[4]= 0;
								}
							}
							else if( ShiftUndo.ul ){
							  double x= ShiftUndo.x, y= ShiftUndo.y;
								if( ShiftUndo.idx== 0 ){
									ShiftUndo.x= ShiftUndo.ul->x1;
									ShiftUndo.y= ShiftUndo.ul->y1;
									ShiftUndo.ul->x1= x;
									ShiftUndo.ul->y1= y;
								}
								else{
								  UserLabel *ul= ShiftUndo.ul;
									ShiftUndo.x= ul->x2;
									ShiftUndo.y= ul->y2;
									if( ul->x1== ul->x2 && ul->y1== ul->y2 ){
										ul->x1= x;
										ul->y1= y;
									}
									ul->x2= x;
									ul->y2= y;
								}
								wi->redraw= 1;
								wi->halt= 0;
								wi->draw_count= 0;
							}
							else if( LabelClipUndo== wi ){
							  UserLabel *ul= wi->ulabel;
							  double old2[2];
								while( ul ){
									if( ul->old2 ){
										old2[0]= ul->x2;
										old2[1]= ul->y2;
										ul->x2= ul->old2[0];
										ul->y2= ul->old2[1];
										memcpy( ul->old2, old2, sizeof(old2) );
										wi->redraw= 1;
										wi->halt= 0;
										wi->draw_count= 0;
									}
									ul= ul->next;
								}
							}
							else if( SplitUndo.set ){
							  unsigned char split= SplitUndo.split;
								if( CheckMask( buttonmask, ShiftMask) ){
								  /* Undo all splits.	*/
								  int idx;
									for( idx= 0; idx< SplitUndo.set->numPoints; idx++ ){
										SplitUndo.set->splithere[idx]= 0;
									}
									SplitUndo.split= 0;
								}
								else{
									SplitUndo.split= SplitHere(SplitUndo.set, SplitUndo.idx);
									SplitUndo.set->splithere[SplitUndo.idx]= split;
								}
								RedrawSet( SplitUndo.set->set_nr, 1 );
							}
							else if( DiscardUndo.set || DiscardUndo.ul.label[0] ){
								if( DiscardUndo.set ){
								  unsigned char Discard= DiscardUndo.split;
									if( DiscardUndo.set== (DataSet*) -1 && DiscardUndo.wi ){
										if( DiscardPoints_Box( DiscardUndo.wi,
											DiscardUndo.lX, DiscardUndo.lY,
											DiscardUndo.hX, DiscardUndo.hY, Discard)
										){
											DiscardUndo.split= !Discard;
/* 											wi->redraw= 1;	*/
										}
										else{
											DiscardUndo.set= NULL;
											DiscardUndo.wi= NULL;
										}
									}
									else{
										if( CheckMask(buttonmask, ShiftMask) ){
										  /* Undo all deletions.	*/
										  int idx;
											for( idx= 0; idx< DiscardUndo.set->numPoints; idx++ ){
												DiscardUndo.set->discardpoint[idx]= 0;
											}
											if( wi->discardpoint && wi->discardpoint[DiscardUndo.set->set_nr] ){
												for( idx= 0; idx< DiscardUndo.set->numPoints; idx++ ){
													wi->discardpoint[DiscardUndo.set->set_nr][idx]= 0;
												}
											}
											DiscardUndo.split= 0;
										}
										else{
											  /* 990614: I don't see (yet) why I should include window-specific
											   \ discards in the undo function. Those are intended to discard
											   \ points outside the visible region.
											   */
											DiscardUndo.split= DiscardedPoint( NULL, DiscardUndo.set, DiscardUndo.idx);
											DiscardUndo.set->discardpoint[DiscardUndo.idx]= Discard;
										}
										RedrawSet( DiscardUndo.set->set_nr, 1 );
									}
								}
								else if( DiscardUndo.ul.label[0] ){
								  UserLabel *h= wi->ulabel, *ul= (UserLabel*) calloc( 1, sizeof(UserLabel) );
									*ul= DiscardUndo.ul;
									while( h && h!= DiscardUndo.head ){
										h= h->next;
									}
									if( h ){
										ul->next= h->next;
										h->next= ul;
									}
									else{
										ul->next= wi->ulabel;
										wi->ulabel= ul;
									}
									DiscardUndo.ul.label[0]= '\0';
									DiscardUndo.head= NULL;
									wi->redraw= 1;
									wi->halt= 0;
									wi->draw_count= 0;
									wi->ulabels+= 1;
								}
							}
							else if( wi->PlaceUndo.valid ){
							  PlaceUndoBuf prev= wi->PlaceUndo;
								switch( wi->PlaceUndo.which ){
									case YPlaced:
										wi->PlaceUndo.x= wi->yname_x;
										wi->PlaceUndo.y= wi->yname_y;
										wi->PlaceUndo.placed= wi->yname_placed;
										wi->PlaceUndo.trans= wi->yname_trans;
										wi->yname_x= prev.x;
										wi->yname_y= prev.y;
										wi->yname_placed= prev.placed;
										wi->yname_trans= prev.trans;
										break;
									case XPlaced:
										wi->PlaceUndo.x= wi->xname_x;
										wi->PlaceUndo.y= wi->xname_y;
										wi->PlaceUndo.placed= wi->xname_placed;
										wi->PlaceUndo.trans= wi->xname_trans;
										wi->xname_x= prev.x;
										wi->xname_y= prev.y;
										wi->xname_placed= prev.placed;
										wi->xname_trans= prev.trans;
										break;
									case LegendPlaced:
										wi->PlaceUndo.x= wi->_legend_ulx;
										wi->PlaceUndo.y= wi->_legend_uly;
										wi->PlaceUndo.sx= wi->legend_ulx;
										wi->PlaceUndo.sy= wi->legend_uly;
										wi->PlaceUndo.placed= wi->legend_placed;
										wi->PlaceUndo.trans= wi->legend_trans;
										wi->_legend_ulx= prev.x;
										wi->_legend_uly= prev.y;
										wi->legend_ulx= prev.sx;
										wi->legend_uly= prev.sy;
										wi->legend_placed= prev.placed;
										wi->legend_trans= prev.trans;
										break;
									case IntensityLegendPlaced:
										wi->PlaceUndo.x= wi->IntensityLegend._legend_ulx;
										wi->PlaceUndo.y= wi->IntensityLegend._legend_uly;
										wi->PlaceUndo.sx= wi->IntensityLegend.legend_ulx;
										wi->PlaceUndo.sy= wi->IntensityLegend.legend_uly;
										wi->PlaceUndo.placed= wi->IntensityLegend.legend_placed;
										wi->PlaceUndo.trans= wi->IntensityLegend.legend_trans;
										wi->IntensityLegend._legend_ulx= prev.x;
										wi->IntensityLegend._legend_uly= prev.y;
										wi->IntensityLegend.legend_ulx= prev.sx;
										wi->IntensityLegend.legend_uly= prev.sy;
										wi->IntensityLegend.legend_placed= prev.placed;
										wi->IntensityLegend.legend_trans= prev.trans;
										break;
								}
								wi->redraw= 1;
								wi->halt= 0;
								wi->draw_count= 0;
							}
							else if( BoxFilterUndo.fp ){
								if( BoxFilter_Undo( wi ) ){
									wi->halt= 0;
									wi->draw_count= 0;
								}
							}
							else{
								Boing(5);
							}
							if( keys[0]== 'z' ){
								keys[0]= '\0';
							}
						}
						else if( index( "oO4567890", keysymbuffer[ 0 ]) && CheckMask(buttonmask, (ShiftMask|Mod1Mask))){
						  int sn= setNumber;
						  DataSet *av_set= &AllSets[setNumber-1], *first= NULL, *last= NULL;
						  int r, i, use_transformed, XYAveraging, add_Interpol, ScattEllipse;
						  char *YAv_Xsort;
						  static int fn= -1;
#ifdef __GNUC__
						  int _draw_set[MaxSets];
#else
						  int *_draw_set= calloc( MaxSets, sizeof(int) );
						  if( !_draw_set ){
							  xtb_error_box( wi->window, "Averaging: can't get memory for sets buffer\n", "Error" );
						  }
						  else{
#endif
							use_transformed= index( "579Oo", keysymbuffer[0])? True : False;
							XYAveraging= index( "45", keysymbuffer[0])? True : False;
							YAv_Xsort= index( "6789", keysymbuffer[0])? Get_YAverageSorting(wi) : "Nsrt";
							add_Interpol= index( "89", keysymbuffer[0])? True : False;
							if( index( "oO0", keysymbuffer[0]) ){
								ScattEllipse= True;
							}
							else{
								ScattEllipse= False;
							}
							xtb_bt_swap( wi->settings );
							setNumber-= 1;
							if( av_set->numPoints> 0 ){
								if( NewSet( wi, &av_set, 0 )== -1 ){
									av_set= NULL;
								}
							}
							if( av_set ){
								  /* if necessary, make space for the setName, which is generated
								   \ by Average().
								   */
								if( av_set->setName ){
									xfree( av_set->setName );
									av_set->setName= NULL;
								}
								xfree( av_set->XUnits );
								xfree( av_set->YUnits );
								xfree( av_set->fileName );
								xfree( av_set->titleText );
								for( i= 0; i< setNumber; i++ ){
									  /* Determine which sets are to be averaged (the displayed ones)	*/
									if( (_draw_set[i]= draw_set(wi, i)) ){
										if( !first ){
											  /* The first of the displayed sets serves as a template for
											   \ some of the properties of the average set.
											   */
											first= &AllSets[i];
										}
										last= &AllSets[i];
									}
								}
								if( (r= Average( wi, _draw_set, "window", 0, 0, "", &wi->process,
												use_transformed, XYAveraging, YAv_Xsort, add_Interpol, ScattEllipse
											)
									)> 0
								){
								  LocalWindows *WL= WindowList;
								  ALLOCA( name, char, LMAXBUFSIZE+2, name_len);
								  char *nbuf;
								  char pc[]= "#x01Request";
									if( CheckMask( buttonmask, ControlMask) ){
										strncpy( name, av_set->setName, LMAXBUFSIZE);
										name[LMAXBUFSIZE]= '\0';
										if( (nbuf= xtb_input_dialog( wi->window, name, 80, LMAXBUFSIZE,
												"Please enter name for new Average set:", parse_codes(pc),
												False,
												NULL, NULL, NULL, NULL, NULL, NULL
											))
										){
											xfree( av_set->setName );
											av_set->setName= XGstrdup( name );
											if( nbuf!= name ){
												xfree( nbuf );
											}
										}
										nocontrol= 1;
									}
									av_set->show_legend= 1;
									av_set->show_llines= 1;
									av_set->draw_set= 1;
									av_set->has_error= av_set->use_error= 1;
									av_set->error_type= wi->error_type[first->set_nr];
									av_set->markFlag= first->markFlag;
									av_set->pixelMarks= first->pixelMarks;
									av_set->barFlag= first->barFlag;
									av_set->overwrite_marks= first->overwrite_marks;
									av_set->markstyle= first->markstyle;
									if( newfile_incr_width ){
										if( (av_set->lineWidth= last->lineWidth+newfile_incr_width)> lineWidth ){
											lineWidth= av_set->lineWidth;
										}
									}
									else{
										av_set->lineWidth= last->lineWidth;
									}
									av_set->elineWidth= av_set->lineWidth/2.0;
									if( !av_set->fileName ){
										av_set->fileName= XGstrdup( "Internal Av.");
									}
									av_set->internal_average= 1;
									realloc_points( av_set, av_set->numPoints+ 2, False );
									  /* Update the relevant information on this set in the LocalWin
									   \ structures.
									   */
									wi->draw_set[setNumber]= 1;
									wi->redraw= 1;
									if( fn!= fileNumber+ 1 ){
										fn= fileNumber+ 1;
									}
									while( WL ){
									  LocalWin *lwi= WL->wi;
										lwi->numVisible[setNumber]= r;
										lwi->fileNumber[setNumber]= av_set->fileNumber= fn;
										if( av_set->error_type!= -1 ){
											lwi->error_type[setNumber]= av_set->error_type;
										}
										if( lwi->fileNumber[setNumber]!= lwi->fileNumber[setNumber-1] ){
											lwi->new_file[setNumber]= av_set->new_file= 1;
										}
										else{
											lwi->new_file[setNumber]= av_set->new_file= 0;
										}
										if( draw_set( lwi, setNumber) ){
											lwi->halt= 0;
											lwi->redraw= 1;
										}
										WL= WL->next;
									}
									setNumber= av_set->set_nr+ 1;
									return_code= 1;
								}
								else{
									if( r< 0 ){
									  /* we found a set that has already been averaged - make it visible.
									   \ In that case, Average() returns the set's index handle_event_times -1 ...
									   */
										wi->draw_set[-r]= 1;
										wi->halt= 0;
										wi->redraw= 1;
										return_code= 1;
									}
									setNumber= sn;
								}
							}
							else{
								setNumber= sn;
								hit= 0;
							}
							xtb_bt_swap( wi->settings );
#ifndef __GNUC__
							xfree( _draw_set );
						  }
#endif
							keys[0]= '\0';
						}
						else if( index( "fb", keysymbuffer[0] ) && CheckMask(buttonmask, Mod1Mask) ){
							if( CheckMask(buttonmask, ShiftMask) ){
								  /* 20031027: select cross cursor. It is not reall doable to store
								   \ and reset the cursor state when we finish filter mode (other than the symbol used)
								   \ so we don't. (There are 2 ways to exit: by de-activating (= here), or by
								   \ finishing the filtering process...; cross-cursor mode is global, filtering
								   \ mode is window-specific...)
								   */
								if( !CursorCross ){
								  LocalWindows *WL= WindowList;
									CursorCross= True;
									CursorCross_Labeled= True;
									while( WL ){
										SelectXInput( WL->wi );
										WL= WL->next;
									}
								}
								if( (wi->filtering= !wi->filtering) ){
									XDefineCursor(disp, wi->window, filterCursor);
								}
								else{
									XDefineCursor(disp, wi->window, (CursorCross)? noCursor : theCursor);
								}
							}
							else if( keysymbuffer[0]== 'f' ){
							  static char ifn[MAXPATHLEN];
/* 								interactive_IncludeFile(wi, "Please enter a filename to parse:" , ifn);	*/
								if( interactive_IncludeFile(wi, NULL, ifn) ){
									return_code= 1;
								}
							}
							if( keys[0]== 'f' || keys[0]== 'F' || keys[0]== 'B' || keys[0]== 'b' ){
								keys[0]= '\0';
							}
						}
						else if( index( "fF", keysymbuffer[0] ) && CheckMask(buttonmask, ShiftMask) ){
							return_code= 1;
							Tile_Files( wi, CheckMask(buttonmask, ControlMask) );
							if( index( "fF", keys[0]) ){
								keys[0]= '\0';
							}
						}
						else if( index( "gG", keysymbuffer[0] ) && CheckMask(buttonmask, ShiftMask) ){
							return_code= 1;
							Tile_Groups( wi, CheckMask(buttonmask, ControlMask) );
							if( index( "gG", keys[0]) ){
								keys[0]= '\0';
							}
						}
						else if( index( "01=", keysymbuffer[0]) && CheckMask(buttonmask,Mod1Mask) ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								if( keysymbuffer[0]== '=' ){
									lwi->legend_always_visible= !lwi->legend_always_visible;
								}
								else{
									lwi->legend_type= keysymbuffer[0]- '0';
								}
								lwi->printed= 0;
								RedrawNow( lwi );
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							return_code= 1;
							if( keys[0]== '0' || keys[0]== '1' ){
								keys[0]= '\0';
							}
						}
						else if( keysymbuffer[0] == 'l' && CheckMask(buttonmask, Mod1Mask) ){
							if( (wi->add_label= !wi->add_label) ){
								XDefineCursor(disp, wi->window, labelCursor);
							}
							else{
								XDefineCursor(disp, wi->window, (CursorCross)? noCursor : theCursor);
							}
							if( CheckMask(buttonmask, ShiftMask) ){
							  /* Means we'll check whether either all marked or all highlighted
							   \ sets are drawn (in that order).
							   */
								wi->add_label= -1;
							}
							xtb_bt_set( wi->label, wi->add_label, (char *) 0);
							return_code= 1;
							if( keys[0]== 'l' ){
								keys[0]= '\0';
							}
						}
#ifdef RUNSPLIT
						else if( keysymbuffer[0] == 'x' && CheckMask(buttonmask, Mod1Mask) ){
							if( wi->cutAction^= _spliceAction ){
								XDefineCursor(disp, wi->window, cutCursor);
							}
							else{
								XDefineCursor(disp, wi->window, (CursorCross)? noCursor : theCursor);
							}
/* 							SetWindowTitle( wi, wi->draw_timer.Tot_Time );	*/
							SetWindowTitle( wi, wi->draw_timer.HRTot_T );
							return_code= 1;
							if( keys[0]== 'x' ){
								keys[0]= '\0';
							}
						}
#endif
						else if( keysymbuffer[0] == 'd' && CheckExclMask(buttonmask, Mod1Mask) ){
							if( wi->cutAction^= _deleteAction ){
								XDefineCursor(disp, wi->window, cutCursor);
							}
							else{
								XDefineCursor(disp, wi->window, (CursorCross)? noCursor : theCursor);
							}
/* 							SetWindowTitle( wi, wi->draw_timer.Tot_Time );	*/
							SetWindowTitle( wi, wi->draw_timer.HRTot_T );
							return_code= 1;
							if( keys[0]== 'd' ){
								keys[0]= '\0';
							}
						}
						else if( keysymbuffer[0] == 'p' &&
							( CheckExclMask(buttonmask, Mod1Mask) ||
							  CheckExclMask(buttonmask, Mod1Mask|ShiftMask))
						){
						  static char *expr= NULL;
						  int expr_len= 0;
						  static double x= 0;
							if( !expr || LMAXBUFSIZE+1> expr_len ){
								expr_len= LMAXBUFSIZE+1;
								expr= (char*) XGrealloc( expr, expr_len* sizeof(char));
							}
							if( expr ){
								if( CheckMask(buttonmask, ShiftMask) ){
									interactive_param_now_allwin( wi, expr, -2, LMAXBUFSIZE, "", &x, False, 0 );
								}
								else{
									interactive_param_now( wi, expr, -2, LMAXBUFSIZE, "", &x, False, 0 );
								}
							}
							else{
								xtb_error_box( wi->window, "Can't get expression buffer", "Error" );
								expr_len= 0;
							}
							return_code= 1;
							if( keys[0]== 'p' || keys[0]== 'P' ){
								keys[0]= '\0';
							}
						}
						else if( keysymbuffer[0]== XK_Escape ){
						  LocalWindows *WL= WindowList;
							handle_event_times= 1;
							event_read_buf[0]= '\0';
							set_NaN( *ascanf_ReadBufVal );
							while( WL ){
/* 								SetWindowTitle( WL->wi, WL->wi->draw_timer.Tot_Time );	*/
								SetWindowTitle( WL->wi, WL->wi->draw_timer.HRTot_T );
								if( WL ){
									WL= WL->next;
								}
							}
							if( keys[0]== 0x1b ){
								keys[0]= '\0';
							}
							return_code= 1;
						}
						else if( keysymbuffer[0]== XK_Delete ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								if( CheckMask( buttonmask, Mod1Mask) ){
								  int s, p;
									if( lwi->discard_invisible && lwi->discardpoint && !CheckMask( buttonmask, ShiftMask) ){
										for( s= 0; s< setNumber; s++ ){
											if( draw_set(lwi, s) && lwi->discardpoint[s] ){
												for( p= 0; p< AllSets[s].numPoints; p++ ){
													if( (int) lwi->discardpoint[s][p]== -1 ){
														lwi->discardpoint[s][p]= 0;
													}
												}
											}
										}
									}
									lwi->discard_invisible= 0;
									RedrawNow( lwi );
								}
								else if( !lwi->discard_invisible ){
								  int old_silent;
									lwi->discard_invisible= 1;
									lwi->redraw= 1;
									lwi->draw_count= 0;
									old_silent= lwi->dev_info.xg_silent( lwi->dev_info.user_state, True );
									TitleMessage( lwi, "Discarding invisible points..." );
									if( !RemoteConnection ){
										XFlush( disp);
									}
									  /* 20000429: why is this a DrawWindow() and not a RedrawNow() call?	*/
									DrawWindow( lwi );
									lwi->dev_info.xg_silent( lwi->dev_info.user_state, wi->silenced || old_silent );
									TitleMessage( lwi, NULL );
									lwi->FitOnce= True;
									RedrawNow( lwi );
								}
								if( WL ){
									WL= WL->next;
									lwi= WL->wi;
								}
							} while( WL && lwi );
							return_code= 1;
						}
						else if( keysymbuffer[ 0 ] == XK_BackSpace && CheckMask(buttonmask, Mod1Mask) ){
							handle_event_times= 0;
							XGIconify( wi );
							break;
						}
						else if( keysymbuffer[0]== 'v' && CheckMask(buttonmask, ShiftMask|ControlMask) ){
						  char *clipboard= NULL;
						  int nbytes= 0;
							if( (clipboard= XFetchBuffer( disp, &nbytes, 0)) && nbytes ){
							  int id;
							  char *sel= NULL;
							  xtb_frame *menu= NULL;
								id= xtb_popup_menu( wi->window, clipboard, "Current clipboard (X cutbuffer 0) contents:",
									&sel, &menu
								);
								xtb_popup_delete( &menu );
							}
							if( keys[0]== 'v' ){
								keys[0]= '\0';
							}
							return_code= 1;
						}
						else{
							hit= False;
						}

						  /* erase the message by writing it once more	*/
						if( handle_event_times> 1 ){
							XDrawString( disp, wi->window, msgGC(wi),
								curX+ 2, curY+ 2, buf, strlen(buf)
							);
						}
						Handle_An_Event( wi->event_level+1, 1, "_Handle_An_Event(KeyPress)-" STRING(__LINE__), wi->window,
								/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
						);
						if( wi->delete_it || wi->halt ){
							  /* 20000427: sync made false	*/
							xtb_XSync( disp, False );
							hit= False;
						}
						if( flush ){
							  /* 20000427: sync made false	*/
							XG_XSync( disp, False );
						}
						if( hit && handle_event_times== 0 ){
						  /* event not handled here, and handle_event_times==0, so we must prevent an
						   \ endless loop
						   */
							break;
						}
					}
					if( handle_event_times> 1 && debugFlag ){
						fprintf( StdErr, "_Handle_An_Event(): event_read_buf=\"%s\": %d x %s, done %d handle_event_times; mask=%s; keys[%d]=\"%s\", keycode=%u\n",
							event_read_buf, handle_event_times, (keysymbuffer[0])? XKeysymToString(keysymbuffer[0]) : "", idx,
							xtb_modifiers_string(buttonmask), nbytes, keys, theEvent->xkey.keycode
						);
						fflush( StdErr );
					}
					for( idx = 0;  idx < nbytes;  idx++){
					  Boolean store= False, showTit= False;
						hit= True;
						if( ! keys[idx] ){
						  /* well.... nothing...	*/
							hit= False;
						}
						else if( keys[idx] == CONTROL_D ){
							if( Num_Windows== 1 ){
								if( xtb_error_box( wi->window,
									"\001This is the only/last window\n"
									"Use ^C or the Quit button to exit\n",
									"Notification"
									)> 0
								){
									goto exit_the_programme;
								}
							}
							else{
							  /* Delete this window */
								DelWindow(theEvent->xkey.window, wi);
								nocontrol= 1;
								return_code= 1;
							}
						} else if( keys[idx] == CONTROL_C ){
							  /* Exit program */
exit_the_programme:;
							XLowerWindow( disp, wi->window );
							if( animations ){
								Elapsed_Since( &Animation_Timer, False );
								Animation_Time+= Animation_Timer.HRTot_T;
								Animations+= animations;
								animations= 0;
								SetWindowTitle( wi, Animation_Time );
							}
							{ LocalWindows *WL = WindowList;
								while( WL ){
									WL->wi->raw_display = 1;
									WL= WL->next;
								}
							}
							Exitting= True;
							if( !RemoteConnection ){
// 								XFlush( disp );
								XSync( disp, True );
							}
							raise( SIGINT );
#ifndef DEBUG
#ifdef __GNUC__
							  /* Hmmm. Cleaning up before exit can cause a core-dump?!
							   \ So we don't clean up. (Compiled with debug-info, and in
							   \ gdb this doesn't happen => compiler bug?)
							   \ 980920: resolved..?
							   \ 20020513: of course, after the raise() above, we should normally not end up here!
							   */
							ExitProgramme(-1);
#else
							  /* If this crashes also, change -1 to 0	*/
							ExitProgramme(-1);
#endif
#endif
							nocontrol= 1;
							debugFlag= dbF;
							dbF_cache= dbFc;
							Allow_DrawCursorCrosses= gADCC;
							return(1);
						}
						else if( keys[idx] == 0x0e ){
						  /* ^N	*/
							nocontrol= 1;
							Restart(wi, NULL );
						}
						else if( keys[idx] == 0x09 ){
						  /* ^I = TAB	*/
							wi->logXFlag= 0;
							wi->logYFlag= 0;
							wi->sqrtXFlag= 0;
							wi->sqrtYFlag= 0;
							wi->polarFlag= 0;
							wi->raw_display= 1;
							wi->legend_placed= 0;
							wi->xname_placed= 0;
							wi->yname_placed= 0;
							wi->IntensityLegend.legend_placed= 0;
							wi->printed= 0;
							nocontrol= 1;
							RedrawNow( wi );
							return_code= 1;
						}
						else if( keys[idx] == '?' ){
							help_fnc(wi, CheckMask(buttonmask, Mod1Mask), True );
						}
						else if( keys[idx] == 'I' ){
						  _ALLOCA( msg, char, 16+256*10, msg_len);
						  char *sl= NULL, *cmsg= NULL;
						  int plen;
							errno= 0;
							msg[0]= '\0';
							if( SS_mX.count || SS_mPoints.count ){
								plen= sprintf( msg, "User Collected Statistics:\n X: %s\n Y: %s\n%s%s%s%s%s%s%s%s%s LY: %s\n HY: %s\n Pnts: %s\n",
									SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_mX),
									SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_mY),
									(SS_mE.count)? " E: " : "",
									(SS_mE.count)? SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_mE) : "",
									(SS_mE.count)? "\n" : "",
									(SAS_mO.pos_count||SAS_mO.neg_count)? " O: " : "",
									(SAS_mO.pos_count||SAS_mO.neg_count)?
										SAS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SAS_mO) : "",
									(SAS_mO.pos_count||SAS_mO.neg_count)? "\n" : "",
									(SS_mI.count)? " Int: " : "",
									(SS_mI.count)? SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_mI) : "",
									(SS_mI.count)? "\n" : "",
									SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_mLY),
									SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_mHY),
									SS_sprint_full( NULL, "%.8g", " \\#xb1\\ ", 0.0, &SS_mPoints)
								);
							}
							if( wi->show_overlap ){
								plen= sprintf( msg, "%s %sverlap: (%s \\#xb1\\ %s) \\#xb1\\ (%s \\#xb1\\ %s)\n\n",
										msg,
										(wi->show_overlap==2)? "Raw o" : "O",
										d2str( SS_Mean_(SS_mMO), "%g", NULL), d2str( SS_St_Dev_(SS_mMO), "%g", NULL),
										d2str( SS_Mean_(SS_mSO), "%g", NULL), d2str( SS_St_Dev_(SS_mSO), "%g", NULL)
								);
							}
							else{
								plen= sprintf( msg, "%s\n", msg );
							}
							if( plen> msg_len/sizeof(char) ){
								fprintf( StdErr, "%s::%d: wrote %d bytes in a buffer of %d!!\n",
									__FILE__, __LINE__,
									plen, msg_len
								);
								fflush( StdErr );
								xtb_error_box( wi->window, "Buffer overwritten\nYou should exit now!\n", "Fatal Error" );
							}
							cmsg= concat( parse_codes( msg ), (sl= ShowLegends( wi, False, -1)), NULL );
							{
							  int id;
							  char *sel= NULL;
							  xtb_frame *menu= NULL;
								id= xtb_popup_menu( wi->window, cmsg, "Drawn Sets Information", &sel, &menu);

								if( sel && sel[0] ){
									if( debugFlag ){
										xtb_error_box( wi->window, sel, "Copied to clipboard:" );
									}
									else{
										Boing(10);
									}
									if( strstr( sel, "Bounding Box:") ){
									  char *c;
									  char *format= concat( d3str_format, ",", NULL);
										if( !format ){
											format= "%g,";
										}
										c= concat( "-bbox ",
											d2str( wi->win_geo.bounds._loX, format, NULL),
											d2str( wi->win_geo.bounds._loY, format, NULL),
											d2str( wi->win_geo.bounds._hiX, format, NULL),
											d2str( wi->win_geo.bounds._hiY, d3str_format, NULL),
											NULL
										);
										if( c ){
											xfree(sel);
											sel = c;
										}
									}
									XStoreBuffer( disp, sel, strlen(sel), 0);
									 // RJVB 20081217
									xfree(sel);
								}
								xtb_popup_delete( &menu );
							}
							xfree( cmsg );
							xfree( sl);
						}
						else if( keys[idx] == 'i' ){
						  extern xtb_hret info_func();
							info_func( wi->info, 0, (xtb_data) wi->window );
							if( wi->redraw){
								RedrawNow( wi );
								return_code= 1;
							}
						} else if( keys[idx] == 'l' ){
							StackWindows( 'u');
							return_code= 1;
						} else if( keys[idx] == 'L' ){
							return_code= 1;
							StackWindows( 'd');
						}
						else if( keys[idx] == 'f' ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								if( lwi->filename_in_legend ){
									if( lwi->labels_in_legend ){
										lwi->labels_in_legend= lwi->filename_in_legend= 0;
									}
									else{
										lwi->labels_in_legend= 1;
									}
								}
								else{
									lwi->filename_in_legend= 1;
								}
								RedrawNow( lwi );
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							return_code= 1;
						} else if( keys[idx] == 'p' ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								X_ps_Marks( wi->dev_info.user_state, -1 );
								RedrawNow( lwi );
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							return_code= 1;
						}
						else if( keys[idx] == 'o' || keys[idx]== 'O' ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								if( lwi->show_overlap ){
									lwi->show_overlap= 0;
								}
								else{
									lwi->show_overlap= (keys[idx]== 'O')? 2 : 1;
								}
								overlap(lwi);
								if( wi->overlap_buf && strlen( lwi->overlap_buf) ){
									RedrawNow( lwi );
									return_code= 1;
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 'S' ){
							xtb_bt_swap( wi->settings );
							wi->pw_placing= PW_MOUSE;
							DoSettings(theEvent->xany.window, wi);
							if( wi->redraw && !wi->silenced ){
								RedrawNow( wi );
								return_code= 1;
							}
							if( wi->delete_it!= -1 ){
								xtb_bt_swap( wi->settings );
							}
						}
						else if( keys[idx] == 'P' ){
							xtb_bt_swap( wi->hardcopy );
							wi->pw_placing= PW_MOUSE;
							PrintWindow(theEvent->xany.window, wi);
							if( wi->redraw){
								RedrawNow( wi );
								return_code= 1;
							}
							if( wi->delete_it!= -1 ){
								xtb_bt_swap( wi->hardcopy );
							}
						}
						else if( keys[idx] == 'g' ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
						  /* Toggle the presence of the buttons. Hide if mapped
						   \ by unmapping them; redraw events (one for each button..)
						   \ are discarded; global redraw is initiated in the end.
						   \ Unhide by "XMapRaised"ing.
						   */
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
							  int flush;
								flush= 0;
								if( lwi->YAv_Sort_frame.mapped ){
									XUnmapWindow( disp, lwi->YAv_Sort_frame.win );
									lwi->YAv_Sort_frame.mapped= 0;
									XUnmapWindow( disp, lwi->cl_frame.win );
									lwi->cl_frame.mapped= 0;
									XUnmapWindow( disp, lwi->hd_frame.win );
									lwi->hd_frame.mapped= 0;
									XUnmapWindow( disp, lwi->settings_frame.win );
									lwi->settings_frame.mapped= 0;
									XUnmapWindow( disp, lwi->info_frame.win );
									lwi->info_frame.mapped= 0;
									XUnmapWindow( disp, lwi->label_frame.win );
									lwi->label_frame.mapped= 0;
									XUnmapWindow( disp, lwi->ssht_frame.win );
									lwi->ssht_frame.mapped= 0;
									flush= 1;
									lwi->redraw= 1;
								}
								else{
									XMapRaised( disp, lwi->YAv_Sort_frame.win);
									lwi->YAv_Sort_frame.mapped= 1;
									XMapRaised( disp, lwi->cl_frame.win);
									lwi->cl_frame.mapped= 1;
									XMapRaised( disp, lwi->hd_frame.win);
									lwi->hd_frame.mapped= 1;
									XMapRaised( disp, lwi->settings_frame.win);
									lwi->settings_frame.mapped= 1;
									XMapRaised( disp, lwi->info_frame.win);
									lwi->info_frame.mapped= 1;
									XMapRaised( disp, lwi->label_frame.win);
									lwi->label_frame.mapped= 1;
									XMapRaised( disp, lwi->ssht_frame.win);
									lwi->ssht_frame.mapped= 1;
									xtb_bt_set( wi->ssht_frame.win, X_silenced(lwi), NULL );
								}
								if( flush ){
								  XEvent dum;
								  int n= 0;
								  /* 20050110: a manual redraw cancels all pending Expose events that would
								   \ cause (subsequent) redraws. (NB: events generated after this point are not
								   \ affected, of course.)
								   */
									XG_XSync( disp, False );
									while( XCheckWindowEvent(disp, lwi->window, ExposureMask|VisibilityChangeMask, &dum) ){
										n+= 1;
									}
								}
								if( lwi->redraw ){
									lwi->dont_clear= True;
									RedrawNow( lwi );
									return_code= 1;
								}
								flush= 0;
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 'R' ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
							  int ut= lwi->use_transformed;
								RawDisplay( lwi, !lwi->raw_display );
/* 								lwi->raw_display= ! lwi->raw_display;	*/
								xtb_bt_swap( lwi->settings );
								lwi->printed= 0;
								lwi->use_transformed= 0;
								RedrawNow( lwi );
								lwi->use_transformed= ut;
/* 								SetWindowTitle( lwi, lwi->draw_timer.Tot_Time );	*/
								SetWindowTitle( lwi, lwi->draw_timer.HRTot_T );
								return_code= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 0x10 ){
							  /* "^P"	*/
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->hardcopy );
								lwi->halt= 0;
								ZoomWindow_PS_Size( lwi, 0, 0, 1 );
								return_code= 1;
								nocontrol= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->hardcopy );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 0x01 ){
						  int i;
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							for( i= 0; i< setNumber; i++ ){
								AllSets[i].draw_set= (AllSets[i].draw_set<0)? 0 : (AllSets[i].draw_set==0)? -1 : 1;
							}
							do{
								xtb_bt_swap( lwi->settings );
								ShowAllSets( lwi );
								RedrawNow( lwi );
								return_code= 1;
								nocontrol= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 0x13 ){
							  /* "^S"	*/
						  int i;
						  LocalWindows *WL;
						  LocalWin *lwi;
reverse_drawn_sets:;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								WL= NULL;
								lwi= wi;
							}
							for( i= 0; i< setNumber; i++ ){
								if( AllSets[i].numPoints> 0 ){
									AllSets[i].draw_set= !AllSets[i].draw_set;
								}
							}
							do{
								xtb_bt_swap( lwi->settings );
								SwapSets( lwi );
								RedrawNow( lwi );
								return_code= 1;
								nocontrol= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 'C' || keys[idx]== '<' ){
						  int i;
						  LocalWindows *WL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								WL= NULL;
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->settings );
								if( keys[idx]== 'C' ){
									for( i= 0; i< setNumber; i++ ){
										lwi->mark_set[i]= 0;
									}
								}
								lwi->plot_only_set0= -1;
								lwi->plot_only_file= -1;
								lwi->plot_only_group= -1;
								xtb_bt_swap( lwi->settings );
								Boing(10);
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 'm' ){
						  int i;
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->settings );
								for( i= 0; i< setNumber; i++ ){
									lwi->mark_set[i]+= (lwi->draw_set[i])? 1 : 0;
								}
								xtb_bt_swap( lwi->settings );
								if( lwi->SD_Dialog ){
									set_changeables(2,False);
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							Boing(1);
						}
						else if( keys[idx] == 'M' ){
						  int i;
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->settings );
								for( i= 0; i< setNumber; i++ ){
									lwi->draw_set[i]= (lwi->mark_set[i]> 0)? 1 : 0;
								}
								lwi->ctr_A= 0;
#ifdef DEBUG
								for( i= 0; i< setNumber; i++ ){
									fprintf( StdErr, "%d ", lwi->mark_set[i] );
								}
								fputc( '\n', StdErr );
								fflush( StdErr );
#endif
								if( setNumber ){
									i= 0;
									  /* find first set that is drawn	*/
									while( ! draw_set( lwi, i) && i< setNumber ){
										i+= 1;
									}
									if( i< setNumber && draw_set(lwi, i) ){
										  /* write it to stdout lwith an appropriate header	*/
										fprintf( stdout, "-plot_only_set %d", i++ );
										  /* find and write follolwing sets that are drawn.	*/
										while( i< setNumber ){
											if( draw_set(lwi, i) ){
												fprintf( stdout, ",%d", i );
											}
											i+= 1;
										}
										fputc( '\n', stdout );
									}
								}
								lwi->printed= 0;
								RedrawNow( lwi );
								return_code= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == 'h' ){
						  int i;
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->settings );
								for( i= 0; i< setNumber; i++ ){
									if( (lwi->legend_line[i].highlight+= (lwi->draw_set[i])? 1 : 0) ){
										if( AllSets[i].s_bounds_set ){
											AllSets[i].s_bounds_set= -1;
										}
									}
								}
								RedrawNow( lwi );
								xtb_bt_swap( lwi->settings );
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx] == '*' ){
						  LocalWindows *WL;
						  LocalWin *lwi;
						  int i, ok= 0;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								WL= NULL;
								lwi= wi;
							}
							do{
								if( (lwi->AlwaysDrawHighlighted= !lwi->AlwaysDrawHighlighted) ){
									for( i= 0; !ok && i< setNumber; i++ ){
										ok= lwi->legend_line[i].highlight;
									}
									if( ok ){
										lwi->printed= 0;
										RedrawNow( lwi );
									}
								}
								  /* 20050114 */
								else{
									lwi->printed= 0;
									RedrawNow( lwi );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							return_code= 1;
						}
						else if( keys[idx] == 'H' ){
						  int i;
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								xtb_bt_swap( lwi->settings );
								for( i= 0; i< setNumber; i++ ){
									lwi->draw_set[i]= (lwi->legend_line[i].highlight> 0)? 1 : 0;
								}
								lwi->ctr_A= 0;
								if( setNumber ){
									i= 0;
									  /* find first set that is drawn	*/
									while( ! draw_set( lwi, i) && i< setNumber ){
										i+= 1;
									}
									if( i< setNumber && draw_set(lwi, i) ){
										  /* write it to stdout lwith an appropriate header	*/
										fprintf( stdout, "-plot_only_set %d", i++ );
										  /* find and write follolwing sets that are drawn.	*/
										while( i< setNumber ){
											if( draw_set(lwi, i) ){
												fprintf( stdout, ",%d", i );
											}
											i+= 1;
										}
										fputc( '\n', stdout );
									}
								}
								lwi->printed= 0;
								RedrawNow( lwi );
								return_code= 1;
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
								}
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
						}
						else if( keys[idx]== ' ' ){
						  LocalWindows *WL= NULL;
						  LocalWin *lwi;
							if( handle_event_times== 0 ){
								WL= WindowList;
								lwi= WL->wi;
							}
							else{
								lwi= wi;
							}
							do{
								lwi->halt= 1;
								if( WL ){
									WL= WL->next;
									lwi= (WL)? WL->wi : 0;
								}
							} while( WL && lwi );
							*ascanf_escape_value= ascanf_escape= 1;
							return_code= 1;
						}
						else if( keys[idx]== 0x18 ){
						  /* ^X	*/
						  extern int ascanf_interrupt;
						  extern double *ascanf_interrupt_value;
							ascanf_escape= ascanf_interrupt= True;
							*ascanf_escape_value= *ascanf_interrupt_value= 1;
							return_code= 1;
						}
						else if( keys[idx]== 0x1f ){
						  /* ^_ == ^/	*/
							display_ascanf_variables( wi->window, CheckMask(buttonmask, Mod1Mask), True, NULL );
							return_code= 1;
						}
						else{
							store= True;
						}
						if( store ){
							if( strlen(event_read_buf)< sizeof(event_read_buf)- 1 ){
							  int n;
								strncat( event_read_buf, &keys[idx], 1);
								if( sscanf( event_read_buf, "%d", &n ) ){
									handle_event_times= n;
									if( handle_event_times== 1 && event_read_buf[0]!= '1' ){
									  /* Smart reset.. :)	*/
										event_read_buf[0]= '\0';
									}
								}
								if( sscanf( event_read_buf, "%lf", ascanf_ReadBufVal )< 1 ){
									set_NaN(*ascanf_ReadBufVal);
								}
								showTit= True;
							}
							hit= False;
						}
						if( !fnkey ){
							fnkey= keys[idx];
						}
						if( fnkey>= 0 && fnkey< 8*(256+12) && key_param_now[fnkey] ){
						  char *expr= key_param_now[fnkey];
						  double val= 0;
						  LocalWin *aw= ActiveWin;
						  char asep = ascanf_separator;
							ActiveWin= wi;
							if( key_param_separator ){
								ascanf_separator = key_param_separator;
							}
							new_param_now( expr, &val, -1 );
							ascanf_separator = asep;
							ActiveWin= aw;
						}
						fnkey= 0;
						if( event_read_buf_cleared ){
							event_read_buf_cleared= False;
							store= False;
							showTit= True;
						}
						  /* 20010606: storage of key[] into event_read_buf took place here */
						if( showTit ){
						  LocalWindows *WL= WindowList;
							while( WL ){
/* 								SetWindowTitle( WL->wi, WL->wi->draw_timer.Tot_Time );	*/
								SetWindowTitle( WL->wi, WL->wi->draw_timer.HRTot_T );
								if( WL ){
									WL= WL->next;
								}
							}
						}
					}
					if( !Exitting && wi->redraw ){
						if( !wi->animating ){
							event_read_buf[0]= '\0';
							set_NaN( *ascanf_ReadBufVal );
						}
						if( !wi->silenced ){
							RedrawNow( wi );
							  /* 20010625: don't propagate a True return code that will interrupt ongoing activity
							   \ elsewhere when the event's target window is silenced!
							   */
							return_code= 1;
							if( !nocontrol && CheckMask(buttonmask, ControlMask) ){
								ShowLegends( wi, True, -1 );
							}
						}
					}
					if( !Exitting && wi->SD_Dialog ){
						set_changeables(2,False);
					}
				}
				break;
			}
			case ButtonPress:{
			  Cursor curs= (CursorCross)? noCursor : theCursor;
				/* Handle creating a new window, or adding a label */
#ifdef ZOOM
				if( wi->add_label ){
					curs= labelCursor;
				}
				else if( wi->cutAction ){
					curs= cutCursor;
				}
				else if( wi->filtering ){
					curs= filterCursor;
				}
				else{
					curs= zoomCursor;
				}
				HandleMouse(Window_Name,
							    /* Wow. Forgot the '&' under gcc, and it works! It
								 \ correctly infers that I want a pointer to the xbutton struct,
								 \ not the thing itself!! Wish every compiler was so smart...
								 */
							  &(theEvent->xbutton),
							  wi, NULL, &curs
				);
				  /* 990318: restore cursor, but not when the desired action is not complete	*/
				if( wi->add_label ){
					curs= labelCursor;
				}
				else if( wi->cutAction ){
					curs= cutCursor;
				}
				else if( curs== zoomCursor ){
					curs= (CursorCross)? noCursor : theCursor;
				}
				XDefineCursor( disp, wi->window, curs );
#ifdef TOOLBOX
				if( wi->redraw ){
					RedrawNow( wi );
					if( wi->SD_Dialog ){
						set_changeables(2,False);
					}
					return_code= 1;
				}
#endif
#endif
				break;
			}
			default:
				if( debugFlag){
					fprintf(StdErr, "Unhandled event type: %s\n", event_name(theEvent->type) );
					fflush( StdErr );
				}
				break;
		}
		if( size_window_print ){
			wi->dev_info.resized= 0;
			ZoomWindow_PS_Size( wi, 0, 0, 1 );
			size_window_print= 0;
		}
		calls+= 1;
		if( Exitting || !WindowList ){
		  /* If the WindowList has become empty, we can safely assume
		   \ we'd better quit.
		   */
			longjmp( toplevel, 1);
		}
		GCA();
		debugFlag= dbF;
		dbF_cache= dbFc;
		Allow_DrawCursorCrosses= gADCC;
		return(return_code);
}

int Handle_MaybeLockedWindows(Bool flush)
{ static LocalWindows *WL= NULL;
  int n= 0;

	if( WL ){
		return(0);
	}

	  /* forcedly redraw all windows that haven't been redrawn before continuing	*/
	  /* 20000428, RJVB: This block handles the events related to window-birth. (Probably)
	   \ depending on window manager and X server, a newly created window receives a large
	   \ number of events related to its appearance that should not each generate a redraw.
	   \ Thus, for each newly created window, we handle those events, and than do a full
	   \ redraw.
	   \ Another possibility is that, despite all precautions, windows remain "locked" in
	   \ a wait for another window to be redrawn: wi->redraw==-2.
	   */
	WL= WindowList;
	while( WL ){
	  LocalWin *wi= WL->wi;
		WL= WL->next;
		if( wi && wi->draw_errors< MIN( quiet_error_count, 5) && (!wi->redrawn || (wi->delayed && wi->redraw!= -2) ) ){
		  int si= wi->silenced, m= 0;
			if( !wi->redrawn ){
				  /* Flush some events related to the window's creation - that is,
				   \ handle them, without doing the redraw that otherwise might be
				   \ associated. To this effect, we set wi->silenced to True.
				   */
				wi->silenced= 1;
				m+= Handle_An_Events( 1, 1, "main-premapping-"STRING(__LINE__), wi->window, StructureNotifyMask|VisibilityChangeMask );
				m+= Handle_An_Events( 1, 1, "main-premapping-"STRING(__LINE__), wi->window, ExposureMask );
				  /* Now do the first redraw for this window:	*/
				if( debugFlag ){
					fprintf( StdErr,
						"Window %02d:%02d:%02d: %d creation/remapping events handled,"
						" redrawing %sfor the first time after (re)map\n",
						wi->parent_number, wi->pwindow_number, wi->window_number, m,
						(si)? "(silenced) " : ""
					);
				}
				  /* 20000505: call TransformCompute() for the 1st time, just to make sure that
				   \ that 1st time does not occur with wi->silenced set (which would cause things
				   \ like legend_placed to get lost.
				   */
				wi->silenced= 0;
				TransformCompute( wi, True );
				wi->silenced= si;
			}
			wi->redraw= 1;
			RedrawNow( wi );
			wi->delayed= False;
			n+= 1;
		}
	}
	if( n && flush ){
		Handle_An_Events( 1, 1, "main-"STRING(__LINE__), 0, 0);
	}
	return(n);
}

int Handle_An_Event( int level, int CheckFirst, char *caller, Window win, long mask)
{ XEvent theEvent;
  int queued, ret;

		if( ReadPipe ){
			queued= XEventsQueued( disp, QueuedAfterReading );
		}
		else{
			queued= XEventsQueued( disp, QueuedAlready);
/* 			queued= XEventsQueued( disp, QueuedAfterFlush);	*/
		}
		if( CheckFirst ){
			if( !queued ){
				return(0);
			}
		}

		if( !win ){
			if( !queued ){
				Handle_MaybeLockedWindows(False);
			}
			if( queued || !Animating ){
				XNextEvent( disp, &theEvent);
				ret= _Handle_An_Event( &theEvent, level, CheckFirst, caller);
			}
		}
		else{
		  /* see if there is an event for win matching the mask
		   \ if not, return False
		   \ else handle the event
		   */
			if( !XCheckWindowEvent( disp, win, mask, &theEvent) ){
				return( 0 );
			}
			ret= _Handle_An_Event( &theEvent, level, CheckFirst, caller);
		}
		return( ret );
}

int Handle_An_Events( int level, int CheckFirst, char *caller, Window win, long mask)
{ XEvent theEvent;
  int queued, n=0;

		queued= XEventsQueued( disp, QueuedAfterFlush);
		if( CheckFirst ){
			if( !queued ){
				return(n);
			}
		}

		do{
			if( !win ){
				if( !queued ){
					Handle_MaybeLockedWindows(False);
				}
				if( queued || !Animating ){
					XNextEvent( disp, &theEvent);
					_Handle_An_Event( &theEvent, level, CheckFirst, caller);
				}
			}
			else{
			  /* see if there is an event for win matching the mask
			   \ if not, return the number of events processed,
			   \ else handle the event and continue checking.
			   */
				if( !XCheckWindowEvent( disp, win, mask, &theEvent) ){
					return( n );
				}
				_Handle_An_Event( &theEvent, level, CheckFirst, caller);
			}
			queued= XEventsQueued( disp, QueuedAfterFlush);
			n+= 1;
		} while( queued );
		return( n );
}


extern long caller_process;
extern int parent;
extern char descent[16];

extern void kill_caller_process();

extern void notify( int sig);
extern void cont_handler( int sig);

extern void close_pager();

int detach= 0;
extern int debugging;

Time_Struct Start;

int UpdateWindowSettings( LocalWin *wi , Boolean is_primary, Boolean dealloc )
{ int i;
  extern char *version_list;
/* 	wi->plot_only_file= -1;	*/
/* 	wi->plot_only_group= -1;	*/
	  /* An empty but existing plot_only_set means draw no set at all.	*/
	if( plot_only_set && plot_only_set_len>= 0 ){
		wi->plot_only_set_len= plot_only_set_len;
		for( i= 0; i< MAXSETS; i++ ){
			wi->draw_set[i]= 0;
			AllSets[i].draw_set= 0;
		}
		wi->plot_only_set0= -1;
		if( debugFlag){
			fprintf( StdErr, "-plot_only_set " );
		}
		for( i= 0; i< plot_only_set_len; i++ ){
			if( wi->plot_only_set0== -1 ){
				wi->plot_only_set0= plot_only_set[i];
			}
			wi->draw_set[ plot_only_set[i] ]= 1;
			AllSets[ plot_only_set[i] ].draw_set= 1;
			wi->plot_only_set[i]= plot_only_set[i];
			if( debugFlag ){
				fprintf( StdErr, "%d,", plot_only_set[i] );
			}
		}
		if( debugFlag){
			fputc( '\n', StdErr );
			fflush( StdErr );
		}
		if( dealloc ){
			xfree( plot_only_set );
			plot_only_set_len= -1;
		}
	}
	else if( is_primary ){
		for( i= 0; i< MAXSETS; i++ ){
			wi->draw_set[i]= AllSets[i].draw_set;
		}
	}
	  /* Copy set-based error_type when it has been specified	*/
	for( i= 0; i< MAXSETS; i++ ){
		if( is_primary ){
			if( AllSets[i].error_type!= -1 ){
				wi->error_type[i]= AllSets[i].error_type;
			}
		}
		  /* Copy set-specific highlighting colour. This might eventually make
		   \ the use of highlight_set[_len] (cf. below) obsolete.
		   */
		if( AllSets[i].hl_info.pixvalue< 0 ){
			wi->legend_line[i].pixvalue= AllSets[i].hl_info.pixvalue;
			wi->legend_line[i].pixelValue= AllSets[i].hl_info.pixelValue;
			wi->legend_line[i].pixelCName= AllSets[i].hl_info.pixelCName;
			AllSets[i].hl_info.pixelCName= NULL;
			AllSets[i].hl_info.pixelValue= 0;
			AllSets[i].hl_info.pixvalue= 0;
		}
	}
	if( mark_set && mark_set_len>= 0 ){
		for( i= 0; i< MAXSETS; i++ ){
			wi->mark_set[i]= 0;
		}
		if( debugFlag){
			fprintf( StdErr, "-mark_set " );
		}
		for( i= 0; i< mark_set_len; i++ ){
			wi->mark_set[ mark_set[i] ]= 1;
			if( debugFlag ){
				fprintf( StdErr, "%d,", mark_set[i] );
			}
		}
		if( debugFlag){
			fputc( '\n', StdErr );
			fflush( StdErr );
		}
		if( dealloc ){
			xfree( mark_set );
			mark_set_len= -1;
		}
	}
	if( highlight_set && highlight_set_len>= 0 ){
		for( i= 0; i< MAXSETS; i++ ){
			  /* We might use the presence of a hl_info per-set struct now.	*/
			wi->legend_line[i].highlight= 0;
		}
		if( debugFlag){
			fprintf( StdErr, "-highlight_set " );
		}
		for( i= 0; i< highlight_set_len; i++ ){
			wi->legend_line[ highlight_set[i] ].highlight= 1;
			if( debugFlag ){
				fprintf( StdErr, "%d,", highlight_set[i] );
			}
		}
		if( debugFlag){
			fputc( '\n', StdErr );
			fflush( StdErr );
		}
		if( dealloc ){
			xfree( highlight_set );
			highlight_set_len= -1;
		}
	}

	if( is_primary ){
		for( i= 0; i< setNumber; i++ ){
			if( AllSets[i].fileNumber== -1 ){
				AllSets[i].fileNumber= (i)? AllSets[i-1].fileNumber : 0;
			}
			wi->fileNumber[i]= AllSets[i].fileNumber;
			wi->new_file[i]= AllSets[i].new_file;
			if( AllSets[i].highlight ){
				wi->legend_line[i].highlight= 1;
				  /* highlighting is supposed to be a window-specific setting!	*/
				AllSets[i].highlight= 0;
			}
			wi->xcol[i]= AllSets[i].xcol;
			wi->ycol[i]= AllSets[i].ycol;
			wi->ecol[i]= AllSets[i].ecol;
			wi->lcol[i]= AllSets[i].lcol;
		}
	}
	if( wi->polarFlag && !use_lx && WINSIDE( wi, 0.75, 0.25) ){
		wi->loX= (wi->radix+ wi->radix_offset)/ 2.0;
		wi->hiX= (wi->radix+ wi->radix_offset)/ 4.0;
	}
	if( log_zero_x_mFlag ){
		wi->log_zero_x= (log_zero_x_mFlag< 0)? wi->loX : wi->hiX;
	}
	if( log_zero_y_mFlag ){
		wi->log_zero_y= (log_zero_y_mFlag< 0)? wi->loY : wi->hiY;
	}
	wi->use_average_error= use_average_error;
	wi->redraw= -1;
	wi->printed= 0;
	if( version_list && dealloc ){
		wi->version_list= concat2( wi->version_list, version_list, NULL );
		xfree( version_list );
	}
	if( next_include_file && dealloc ){
		xfree( wi->next_include_file );
		wi->next_include_file= next_include_file;
		next_include_file= NULL;
	}
	if( dealloc ){
		if( nextGEN_Startup_Exprs ){
			wi->next_startup_exprs= XGStringList_Delete( wi->next_startup_exprs );
			wi->next_startup_exprs= nextGEN_Startup_Exprs;
			nextGEN_Startup_Exprs= NULL;
		}
		if( Init_Exprs ){
			wi->init_exprs= XGStringList_Delete( wi->init_exprs );
			wi->init_exprs= Init_Exprs;
			wi->new_init_exprs= True;
			Init_Exprs= NULL;
		}
		if( Dump_commands ){
			wi->Dump_commands= XGStringList_Delete( wi->Dump_commands );
			wi->Dump_commands= Dump_commands;
			Dump_commands= NULL;
		}
		if( DumpProcessed_commands ){
			wi->DumpProcessed_commands= XGStringList_Delete( wi->DumpProcessed_commands );
			wi->DumpProcessed_commands= DumpProcessed_commands;
			DumpProcessed_commands= NULL;
		}
	}
	if( startup_wi.win_geo.nobb_range_X ){
		wi->win_geo.nobb_range_X= startup_wi.win_geo.nobb_range_X;
		wi->win_geo.nobb_loX= startup_wi.win_geo.nobb_loX;
		wi->win_geo.nobb_hiX= startup_wi.win_geo.nobb_hiX;
		startup_wi.win_geo.nobb_range_X= 0;
	}
	if( startup_wi.win_geo.nobb_range_Y ){
		wi->win_geo.nobb_range_Y= startup_wi.win_geo.nobb_range_Y;
		wi->win_geo.nobb_loY= startup_wi.win_geo.nobb_loY;
		wi->win_geo.nobb_hiY= startup_wi.win_geo.nobb_hiY;
		startup_wi.win_geo.nobb_range_Y= 0;
	}
	if( VCat_X ){
		Free_ValCat( wi->ValCat_X );
		wi->ValCat_X= VCat_X;
		VCat_X= NULL;
	}
	if( VCat_XFont ){
		Free_CustomFont( wi->ValCat_XFont );
		xfree( wi->ValCat_XFont );
		wi->ValCat_XFont= VCat_XFont;
		VCat_XFont= NULL;
	}
	if( VCat_Y ){
		Free_ValCat( wi->ValCat_Y );
		wi->ValCat_Y= VCat_Y;
		VCat_Y= NULL;
	}
	if( VCat_YFont ){
		Free_CustomFont( wi->ValCat_YFont );
		xfree( wi->ValCat_YFont );
		wi->ValCat_YFont= VCat_YFont;
		VCat_YFont= NULL;
	}
	if( VCat_I ){
		Free_ValCat( wi->ValCat_I );
		wi->ValCat_I= VCat_I;
		VCat_I= NULL;
	}
	if( VCat_IFont ){
		Free_CustomFont( wi->ValCat_IFont );
		xfree( wi->ValCat_IFont );
		wi->ValCat_IFont= VCat_IFont;
		VCat_IFont= NULL;
	}

	if( ColumnLabels ){
		Free_LabelsList( wi->ColumnLabels );
		wi->ColumnLabels= ColumnLabels;
		ColumnLabels= NULL;
	}
	return(1);
}

int is_pipe;

extern Window NewWindow( char *progname, LocalWin **New_Info, double _lowX, double _lowY, double _lowpX, double _lowpY,
					double _hinY, double _upX, double _upY, double asp,
					LocalWin *parent, double xscale, double yscale, double dyscale, int add_padding
);

extern Boolean PIPE_error;
extern int DumpFile, ArgsParsed;


char ExecTime[257];

static int wakeup= 1;
static void X_sleep_wakeup( int sig )
{
	wakeup= 1;
	signal( SIGALRM, X_sleep_wakeup );
}

int ReadInitFile(char *sFname)
{ FILE *rfp;
  extern int NoProcessHist;
  int n= -1, nc= NoComment, nph= NoProcessHist, ap= ArgsParsed;
	NoComment= True;
	NoProcessHist= True;
	ArgsParsed= True;
	if( sFname && (rfp= fopen( sFname, "r" )) ){
		n= ReadData( rfp, sFname, -1 );
		if( n> 0 ){
			maxitems= n;
		}
		ascanf_exit= 0;
		fclose( rfp );
	}
	NoComment= nc;
	NoProcessHist= nph;
	ArgsParsed= ap;
	return(n);
}

int xgraph( int argc, char *argv[] )
  /* This used to be the old main. Reads the default and other initial settings, takes care
   \ of parsing the commandline, reads the files found on the commandline, and fires up the
   \ graphics, entering the main event-loop.
   */
{ Window primary;
  FILE *strm;
  XColor fg_color, bg_color;
  char *labels_file= NULL, *printfile= NULL;
  int exit_level= 0, idx, commandlength=0;
  int added_files= 0;
  extern double *ascanf_ScreenWidth, *ascanf_ScreenHeight;
  int animate;
  extern int Check_Option(), Opt01;
  double AGC;

	Elapsed_Since( &Start, True );
	Elapsed_Since( &Start, True );
	  /* This is to link in matherr.c containing the matherr routine	*/
	matherr_off= 0;

	Argv= argv;
	Argc= argc;
	{ time_t timer= time(NULL);
		strcpy( ExecTime, ctime(&timer) );
	}
	  /* Wipe out the newline..	*/
	ExecTime[strlen(ExecTime)-1]= '\0';
    Prog_Name = rindex( argv[0], '/');

	if( !Prog_Name){
		Prog_Name= argv[0];
	}
	else{
		Prog_Name++;
	}
	Prog_Name= XGstrdup( Prog_Name );
	progname= 0;

	{ char *cb= NULL;
#if defined(__MACH__) || (defined(linux) && !defined(_XOPEN_SOURCE))
	  extern char *cuserid(char*);
	  char *uid = cuserid(NULL);
#elif defined(linux) && defined(_XOPEN_SOURCE)
#	ifdef L_cuserid
	  char UID[L_cuserid+2];
#	else
	  extern char *cuserid(char*);
	  char UID[16];
#	endif
	  char *uid = cuserid(UID);
#else
	  char *uid = "<someone>";
#endif
		cb= concat2( cb, "At ", ExecTime, ", for ", uid, ": ", Prog_Name, NULL);
		for( idx= 1; idx< argc; idx++ ){
			cb= concat2( cb, " ", argv[idx], NULL);
		}
		if( cb ){
			add_comment( cb );
		}
		xfree( cb );
	}

	memset( &startup_wi, 0, sizeof(LocalWin) );

	  /* Initial parsing of arguments, finding those which must be handled before
	   \ any other initialisations.
	   */
	for( idx= 0; idx< argc; idx++){
		commandlength+= strlen( argv[idx] )+ 1;
		if( Check_Option( strncasecmp,argv[idx], "-RemoteConnection", 14) == 0) {
			RemoteConnection = Opt01;
			if( debugFlag && RemoteConnection ){
				fprintf( StdErr, "Using RemoteConnection mode\n" );
			}
			fflush( StdErr);
		}
		else if( Check_Option( strncasecmp, argv[idx], "-psn_", -1)== 0
			&& isdigit(argv[idx][5]) && argv[idx][6]=='_'
		){
		  char *c= getenv("DISPLAY");
			if( debugFlag ){
				fprintf( StdErr, "ignoring Mac OS X window-manager argument \"%s\"\n", argv[idx] );
			}
			MacStartUp= True;
			if( !c ){
				if( debugFlag ){
					fprintf( StdErr, "NB: DISPLAY variable not defined, setting it to \":0.0\" just in case.\n" );
				}
				setenv( "DISPLAY", ":0.0" );
			}
		}
		else if (Check_Option( strncasecmp,argv[idx], "-db", 3) == 0) {
		  int l;
			/* Debug on */
			fprintf( StdErr, "%s - maximal linelength is %d\n", argv[0], LMAXBUFSIZE);
			debugFlag= Opt01;
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
		}
		else if( Check_Option( strncasecmp,argv[idx], "-wns", 4) == 0 ){
		  extern int WarnNewSet;
			WarnNewSet= Opt01;
		}
		else if( strcmp( argv[idx], "-debugging")==0 ){
			debugging= 1;
		}
		else if( strcmp( argv[idx], "-nodetach")==0 ){
			detach= -1;
		}
		else if( strcmp( argv[idx], "-detach")==0 ){
			detach= 2;
		}
		else if( strncasecmp( argv[idx], "-page", 5)== 0){
			if( !isatty(fileno(stdin)) || !caller_process){
			  char *PAGER=getenv("PAGER");
				use_pager= 1;
					if( !PAGER)
						PAGER="more";
					PIPE_error= False;
#if defined(__APPLE__) || defined(__MACH__)
					if( !(StdErr= register_FILEsDescriptor( popen(PAGER, "w")) ) )
#else
					if( !(StdErr= register_FILEsDescriptor( popen(PAGER, "wb")) ) )
#endif
					{
						fprintf( stderr, "%s.%s: can't open pipe to \"%s\"\n", Prog_Name, descent, PAGER);
						StdErr= register_FILEsDescriptor(stderr);
						use_pager= 0;
					}
					else{
						signal( SIGPIPE, PIPE_handler );
						PIPE_fileptr= &StdErr;
						atexit( close_pager );
					}
			}
			else{
				fprintf( stderr, "%s.%s: not attached to a tty\n", Prog_Name, descent);
			}
		}
		else if( strncasecmp( argv[idx], "-progname", 6)== 0 ){
		  /* This allows the user to use a different X-resource database.	*/
			if( idx+1 >= argc || argv[idx+1][0]== 0 )
				argerror( "missing or invalid programname", argv[idx]);
			xfree( Prog_Name );
			Prog_Name= XGstrdup( argv[idx+1] );
			commandlength+= strlen(Prog_Name);
			progname= 1;
			idx += 1;
		}
		else if( strncasecmp( argv[idx], "-VisualType", 11)== 0 ){
		  int x;
		  extern char *VisualClass[];
		  extern int VisualClasses;
			if( idx+1 >= argc || argv[idx+1][0]== 0 ){
				argerror( "missing or invalid type", argv[idx]);
			}
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
			idx+= 1;
		}
		else if( Check_Option( strncasecmp, argv[idx], "-use_XDBE", -1)== 0 ){
			ux11_useDBE= (Opt01)? True : False;
			idx+= 1;
		}
		else if( strncasecmp( argv[idx], "-MinBitsPPixel", 14)== 0 ){
		  int x;
			if( idx+1 >= argc || argv[idx+1][0]== 0 )
				argerror( "missing or invalid depth", argv[idx]);
			if( (x= atoi( argv[idx+1]))> 0 && x<= 24 ){
				ux11_min_depth= x;
			}
			idx+= 1;
		}
		else if( strncasecmp( argv[idx], "-separator", 10)== 0 ){
		  char c;
			if( idx+1 >= argc ){
				argerror( "missing separator", argv[idx]);
			}
			c = argv[idx+1][0];
			if( c > 0 && c<= 127 && !isdigit(c) && !isalpha(c) && !index( "[]{}\"`&@$*", c) ){
				ascanf_separator= c;
			}
			else{
				argerror( "invalid separator value", argv[idx+1] );
			}
			idx+= 1;
		}
	}
	if( use_pager && detach> 0 ){
		fprintf( StdErr, "Detaching disabled when using pager\n" );
		detach= -1;
	}
	  /* Original versions of detach. OLD_DETACH spawns another, slightly different
	   \ xgraph process. The newer version forks a child, and has the child doing
	   \ all the work, with the parent waiting in the foreground to quit. Accounting
	   \ doesn't work (except for total-time), and errors (core-dumps) are not reported.
	   \ 970324: OLD_DETACH code deleted.
	   */
	if( detach== 1 && !debugging ){
	  pid_t cpid= fork();
		if( cpid== -1 ){
			fprintf( StdErr, "Error - couldn't fork/detach (%s)\n", serror() );
		}
		else{
			if( cpid ){
				if( debugFlag ){
					fprintf( StdErr, "Parent %ld pause()s, waiting for child %ld\n",
						(long) getpid(), (long) cpid
					);
					fflush( StdErr );
				}
				parent= 1;
				signal( SIGUSR1, notify );
				  /* pause() until our real child tells us to die. If before that
				   \ time we receive a signal, we both die (if lethal)
				   */
				pause();
				fprintf( StdErr, ".. _exit\n");
				_exit(0);
			}
			else{
				parent= 0;
				signal( SIGUSR1, SIG_IGN );
				Elapsed_Since( &Start, True );
				strcpy( descent, "child" );
				caller_process= (long) getppid();
				if( debugFlag ){
					fprintf( StdErr, "Child %ld will detach from %ld\n", (long) getpid(), caller_process );
					fflush( StdErr );
				}
				atexit( kill_caller_process);
			}
		}
	}

      /* Open up new display
	   \ If we allow finding the best display (search_vislist is True),
	   \ don't search if user specified a display.
	   */
	if( !disp ){
	  int search;
		disp = ux11_open_display(argc, argv, &search );
		if( search_vislist ){
			search_vislist= !search;
		}
	}
    XSetErrorHandler(XErrHandler);
    XSetIOErrorHandler(XFatalHandler);

	wm_delete_window= XInternAtom( disp, "WM_DELETE_WINDOW", False );

	if( rd_flag( "MonoChrome") )
		MonoChrome= 1;

	if( VendorRelease(disp)== 4 && !strcmp( ServerVendor(disp), "Hewlett-Packard Company") ){
	  /* cannot draw the mark-bitmaps, so we just put a font in its place	*/
		use_markFont= 1;
	}
	else{
		use_markFont= 0;
	}

	*ascanf_ScreenWidth= DisplayWidth(disp, screen);
	*ascanf_ScreenHeight= DisplayHeight(disp, screen);

	  /* Another initial scan for arguments that modify the
	   * initialisation behaviour.
	   */
    for( idx= 1; idx< argc; idx++){
	  char *font_name;
		if( strncasecmp( argv[idx], "-colour", 5)== 0 && strcasecmp(argv[idx], "-ColouredSetTitles") )
			MonoChrome= 0;
		else if( strncasecmp( argv[idx], "-monochrome", 5)== 0 )
			MonoChrome= 2;
		else if( strcasecmp(argv[idx], "-maxlbuf") == 0 ){
		  /* Allow another maximum linelength	*/
		  int x;
			if( idx+1 >= argc)
				argerror("missing number", argv[idx]);
			if( (x= atoi(argv[idx+1]))> LMAXBUFSIZE ){
				LMAXBUFSIZE= x;
				if( debugFlag){
					fprintf( StdErr, "maximum (data) linelength set to %d\n", LMAXBUFSIZE );
				}
			}
			idx+= 1;
		}
		else if( strcasecmp(argv[idx], "-maxsets") == 0 ){
		  /* Specify the maximum number of datasets	*/
		  int x;
			if( idx+1 >= argc)
				argerror("missing number", argv[idx]);
			if( (x= atoi(argv[idx+1]))> 0 ){
				MaxSets= x;
			}
			if( debugFlag)
				fprintf( StdErr, "MaxSets=%d\n", MaxSets);
			idx+= 1;
		}
		else if( strncasecmp(argv[idx], "-mf", 3) == 0 ){
			if( argv[idx][3]== '0' )
				use_markFont = 0;
			else
				use_markFont = 1;
			idx++;
		}
		  /* Font handling:	*/
		else if( strcasecmp(argv[idx], "-tf") == 0 ){
			  /* Title Font */
			if( idx+1 >= argc)
				argerror("missing font", argv[idx]);
			font_name= argv[idx+1];
			New_XGFont( 'TITL', font_name );
			idx += 1;
		}
		else if( strcasecmp(argv[idx], "-lf") == 0 || strcasecmp(argv[idx], "-lef")== 0 ){
			/* Legend Font */
			if( idx+1 >= argc)
				argerror("missing font", argv[idx]);
			font_name= argv[idx+1];
			New_XGFont( 'LEGN', font_name );
			idx += 1;
		}
		else if( strcasecmp(argv[idx], "-laf") == 0 ){
			/* Label Font */
			if( idx+1 >= argc)
				argerror("missing font", argv[idx]);
			font_name= argv[idx+1];
			New_XGFont( 'LABL', font_name );
			idx += 1;
		}
		else if( strcasecmp(argv[idx], "-df") == 0 ){
			/* Label Font */
			if( idx+1 >= argc)
				argerror("missing font", argv[idx]);
			font_name= argv[idx+1];
			New_XGFont( 'DIAL', font_name );
			idx += 1;
		}
		else if( strcasecmp(argv[idx], "-af") == 0 ){
			/* Label Font */
			if( idx+1 >= argc)
				argerror("missing font", argv[idx]);
			font_name= argv[idx+1];
			New_XGFont( 'AXIS', font_name );
			idx += 1;
		}
		else if( Check_Option( strncasecmp,argv[idx], "-synchro", 8) == 0) {
		  extern int HardSynchro, Synchro_State;
			Conditional_Toggle( HardSynchro );
			Conditional_Toggle( Synchro_State );
			Synchro_State= !Synchro_State;
			X_Synchro(NULL);
			idx+= 1;
		}
    }

      /* Set up hard-wired defaults and allocate spaces */
    InitSets();
	Init_Hard_Devices();
    fg_color.pixel = normPixel;
    XQueryColor(disp, cmap, &fg_color);
    bg_color.pixel = bgPixel;
    XQueryColor(disp, cmap, &bg_color);

	{ XColor cc_color;
	  Pixmap nocurs= XCreateBitmapFromData( disp, RootWindow(disp,screen),
						  XG_noCursor_bits, XG_noCursor_width, XG_noCursor_height
					);
		cc_color.pixel= zeroPixel;
		XQueryColor(disp, cmap, &cc_color);
		noCursor= XCreatePixmapCursor( disp, nocurs, nocurs, &cc_color, &fg_color, 0, 0);
		XFreePixmap( disp, nocurs );
	}
	theCursor = XCreateFontCursor(disp, XC_tcross );
    zoomCursor = XCreateFontCursor(disp, XC_sizing);
    labelCursor = XCreateFontCursor(disp, XC_pencil);
    cutCursor = XCreateFontCursor(disp, XC_cross);
	filterCursor= XCreateFontCursor(disp, XC_dotbox );

	// 20100401: hardcode a number of default settings I've been using for as long as I remember:
	{ extern int DisplayHeight_MM, DisplayWidth_MM;
	  extern double Progress_ThresholdTime;
	  extern int WM_TBAR;
	  extern Boolean no_buttons, X_psMarkers;
		axisWidth = 2;
		Odevice = XGstrdup("XGraph");
		Odisp = XGstrdup("Append File");
		DisplayHeight_MM = 216;
		DisplayWidth_MM = 285;
		errorWidth = 1;
		if( GetColor( "Yellow", &highlightPixel ) ){
			StoreCName( highlightCName );
		}
		else{
			highlightPixel = -1;
		}
		lineWidth = 2;
		no_buttons = 1;
		psdash_power = 0.65;
		psm_incr = 1.5;
		Oprintcommand = XGstrdup("%s");
		Progress_ThresholdTime = 0.25;
		WM_TBAR = 19;
		X_psMarkers = True;
		Ybias_thres = -1;
		if( GetColor( "Red", &zeroPixel ) ){
			StoreCName( zeroCName );
		}
		else{
			zeroPixel = -1;
		}
		noExpX = noExpY = 1;
		get_radix( "360", &radix, radixVal );
		ps_b_offset = ps_l_offset = 0;
		 // Landscape.
		Print_Orientation = 1;
	}

      /* Read X defaults and override hard-coded defaults */
    ReadDefaults();

	  /* Initialise a local, dummy LocalWin (just for the settings):	*/
	memset( &StubWindow, 0, sizeof(StubWindow) );
	{ LocalWin *ww= NULL;
		check_wi( &ww, "xgraph" );
		StubWindow.window= ww->window;
		  /* Allocate storage areas:	*/
		ww= &StubWindow;
		NewWindow( Window_Name, &ww, 0, 0, 0, 0, 1, 1, 1, NORMASP, NULL, 1.0, 1.0, 1.0, 1);
	}
	CopyFlags( &StubWindow, NULL );
	StubWindow._log_zero_x= StubWindow.log_zero_x;
	StubWindow._log_zero_y= StubWindow.log_zero_y;
	if( _log_zero_x)
		StubWindow.log10_zero_x= cus_log10X( &StubWindow, _log_zero_x);
	if( _log_zero_y)
		StubWindow.log10_zero_y= cus_log10Y( &StubWindow, _log_zero_y);

#ifdef TOOLBOX
    xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );
#endif

	  /* Parse the initialisation file if it exists:	*/
	{ char *sFname;
		sFname= concat( PrefsDir, "/xg_init.xg", NULL );
		if( sFname ){
			ReadInitFile(sFname);
			xfree(sFname);
		}
		ReadInitFile("./xg_init.xg");
	}

      /* Parse the argument list */
    ParseArgs(argc, argv);
	ArgsParsed= True;
      /* Update the data sets */
    for( idx = 0;  idx < MAXSETS;  idx++ ){
		AllSets[idx].lineWidth = lineWidth;
		AllSets[idx].elineWidth = errorWidth;
		AllSets[idx].markFlag = markFlag;
		if( barBase_set ){
			AllSets[idx].barBase_set= barBase_set;
			AllSets[idx].barBase= barBase;
		}
		if( barWidth_set ){
			AllSets[idx].barWidth_set= barWidth_set;
			AllSets[idx].barWidth= barWidth;
		}
		if( barType_set ){
			AllSets[idx].barType= barType;
		}
		AllSets[idx].arrows = arrows;
		AllSets[idx].overwrite_marks = overwrite_marks;
		AllSets[idx].noLines = noLines;
		AllSets[idx].floating= False;
		if( !AllSets[idx].ncols ){
			AllSets[idx].ncols= NCols;
		}
		AllSets[idx].xcol= xcol;
		AllSets[idx].ycol= ycol;
		AllSets[idx].ecol= ecol;
		AllSets[idx].lcol= lcol;
		AllSets[idx].Ncol= Ncol;
    }

	  /* Do some re-initialisations:	*/

#ifdef TOOLBOX
    xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );
#endif

	  /* Re-initialise the local, dummy LocalWin (just for the settings):	*/
	CopyFlags( &StubWindow, NULL );
	StubWindow._log_zero_x= StubWindow.log_zero_x;
	StubWindow._log_zero_y= StubWindow.log_zero_y;
	if( _log_zero_x)
		StubWindow.log10_zero_x= cus_log10X( &StubWindow, _log_zero_x);
	if( _log_zero_y)
		StubWindow.log10_zero_y= cus_log10Y( &StubWindow, _log_zero_y);

	  /* check some settings	*/
	if( polarFlag)
		barFlag= 0;

      /* Read the data into the data sets */
	x_scale_start= 1;
	y_scale_start= 1;
	xp_scale_start= 1;
	yp_scale_start= 1;
	yn_scale_start= 1;
	llpx= -1; llpy= -1;
	llny= 1;
	SS_Reset_( SS_X);
	SS_Reset_( SS_Y);
	SS_Reset_( SS_SY);
	SS_Reset_( SS_x);
	SS_Reset_( SS_y);
	SS_Reset_( SS_e);
	SS_Reset_( SS_sy);
	SS_Reset_( SS__x);
	SS_Reset_( SS__y);
	SS_Reset_( SS__e);

	if( !cgetenv( "XG_DUMP_FPE") ){
		signal( SIGFPE, handle_FPE);
	}
	signal( SIGHUP, ExitProgramme);
	signal( SIGINT, ExitProgramme);
	signal( SIGUSR2, Restart_handler);

	{ char *c= getenv( "XGRAPH_LABELS" );
		if( c ){
			labels_file= XGstrdup(c);
			putenv( "XGRAPH_LABELS=" );
		}
	}

	  /* 20010720: the joys of C and X11 combined :) All of a sudden, I get X errors when freeing the allocated
	   \ cursors and pixmaps from CleanUp() when I haven't opened a LocalWin window... I don't think I modified
	   \ anything that could possibly have had that result, but it happens. It doesn't happen when I open an
	   \ initial window, to be discarded as soon as the user caused one to be opened. I don't understand that
	   \ neither, but I'll just have to live with it. We open it iconified, though. Let's just say that it
	   \ gives something somebody could try to interact with :)
	   */
	if(1){
	  int is= IconicStart;
	  char *gs= geoSpec, igs[]= "32x32+0+0";
	  extern char XGraphBuildString[];
	  char *title= concat( "XGraph -- ", XGraphBuildString, NULL );
		IconicStart= True;
		geoSpec= igs;
		NewWindow( title, &InitWindow,
			startup_wi.R_UsrOrgX, startup_wi.R_UsrOrgY, usrLpX, usrLpY, usrHnY,
			startup_wi.R_UsrOppX, startup_wi.R_UsrOppY, NORMASP, NULL, 1.0, 1.0, 1.0, 1
		);
		XSync( disp, False );
		init_window= InitWindow->window;
		while( Handle_An_Event( InitWindow->event_level, 1, "xgraph- Initial Window Mapping", 0, 0) ){ }
		IconicStart= is;
		geoSpec= gs;
		xfree(title);
	}

	if( numFiles || (!ReadPipe && !IOImportedFiles) ){
		if( !numFiles ){
		  extern char data_separator;

			if( MacStartUp ){
				fprintf( StdErr, "xgraph: GUI startup without input files!\n" );
				goto all_files_read;
			}

			fprintf( StdErr, "xgraph: waiting for data; 2 or 3 columns" );
			if( Use_ReadLine && isatty(fileno(stdin)) ){
				fputs( " using GNU readline", StdErr );
			}
			if( isspace( data_separator ) ){
				fputs( " separated by whitespace\n", StdErr );
			}
			else{
				fprintf( StdErr, " separated by '%c'\n", data_separator );
			}
			fflush( StdErr );
			if( !MaxnumFiles ){
				MaxnumFiles= 1;
			}
			if( !inFileNames ){
				realloc_FileNames( &StubWindow, 0, MaxnumFiles, "xgraph()" );
			}
			inFileNames[0]= XGstrdup("stdin");
		}
		idx= 0;
		do{
		  char *fname, *pyfn;
		  int addfile= 0, is_labels= 0;
		  extern int From_Tarchive;
			is_pipe= 0;
			if( !AddFile ){
				fname= inFileNames[idx];
			}
			else{
			  /* The previous file specified a file that should be read immediately
			   \ after that file is read, processed & closed. This allows for
			   \ recursive inclusion independent of the maximal number of open files.
			   \ we make a local copy of addfile, and xfree addfile, which sets it to null.
			   */
				fname= XGstrdup(AddFile);
				if( AddFile== labels_file ){
					is_labels= 1;
				}
				xfree(AddFile);
				addfile= 1;
				if( debugFlag ){
					fprintf( stderr, "xgraph: reading file \"%s\"\n", fname );
				}
			}
			if( fname[0]== '|' ){
			  /* a filename starting with a '|' is taken to be a pipe-specification.	*/
				fname++;
				PIPE_error= False;
				strm= popen( fname, "r" );
				is_pipe= 1;
			}
			else if( (pyfn= PyOpcode_Check( fname)) ){
			  extern DM_Python_Interface *dm_python;
				if( dm_python && dm_python->Import_Python_File ){
					if( (*dm_python->isAvailable)() ){
						(*dm_python->Import_Python_File)( pyfn, NULL, 0, False );
					}
					else{
						fprintf( StdErr, "Python is not available to read \"%s\" - do you have a Python console open?\n", pyfn );
					}
				}
				else{
					fprintf( StdErr, "xgraph: failure to initialise python, not reading '%s'\n", pyfn );
					fflush( StdErr );
				}
				goto read_next_file;
			}
			else{
				if( (addfile || numFiles) && !From_Tarchive ){
					if( strcmp(fname, "-")== 0 ){
					  extern char data_separator;
					  char *tn;
						fprintf( StdErr, "xgraph: waiting for data; 2 or 3 columns separated by '%c'\n", data_separator );
						fflush( StdErr );
						if( fname== inFileNames[idx] ){
							xfree( inFileNames[idx] );
							inFileNames[idx]= XGstrdup( "stdin" );
							fname= inFileNames[idx];
						}
						strm= stdin;
						if( feof(strm) && (tn= ttyname(fileno(strm))) ){
							strm= freopen( tn, "r", strm );
						}
					}
					else{
						strm= fopen( fname, "r" );
					}
				}
				else{
					strm= stdin;
				}
			}
			if( !strm ){
				fprintf(stderr, "warning:  cannot open file `%s' (%s)\n",
					   fname, serror()
				);
			}
			else if( feof(strm) || ferror(strm) ){
				fprintf(stderr, "warning:  eof or error on file `%s' (%s)\n",
					   fname, serror()
				);
			}
			else{
			  DataSet *cur_set;
			  int n;
				IdentifyStreamFormat( fname, &strm, &is_pipe );
				_Xscale= _Yscale= _DYscale= 1.0;
				Xscale= Yscale= DYscale= 1.0;
				ReadData_commands= 0;
#ifdef DEBUG
	{ extern FILE **FILE_PPTR;
				FILE_PPTR= &strm;
	}
#endif
				if( (n = ReadData(strm, fname, idx )) <= 0 ){
					ascanf_exit= 0;
					if( ReadData_commands<= 0 ){
						fprintf(StdErr, "No numerical data in \"%s\"\n", fname );
					}
					else{
						fprintf( StdErr, "Only (%d) command(s) in \"%s\"\n",
							ReadData_commands, fname
						);
						if( From_Tarchive> 0 ){
							From_Tarchive-= 1;
						}
					}
				}
				else{
				  /* 950622: ReadData used to initialise a new set by calling
				   \ NewSet() in its return statement. This is no longer the case, in order to
				   \ allow for "true" inclusion with the *READ_FILE* command (also
				   \ strange behaviour resulted when recursively including the current file).
				   \ Therefore, we do it here (we still want a new set for each
				   \ file specified on the command line).
				   */
				  int sa= SameAttrs;
					maxitems= n;
					ascanf_exit= 0;
					cur_set= &AllSets[setNumber];
					if( ResetAttrs && cur_set->numPoints> 0 ){
					  /* This is necessary to prevent the incrementing of linestyle, lineWidth etc.
					   \ Normally, ReadData() ensures that this does not happen, except when
					   \ a set is ended by EOF (and thus, cur_set->numPoints > 0).
					   */
						SameAttrs= True;
					}
					if( NewSet( NULL, &cur_set, 0)== -1 ){
						numFiles= idx;
						xfree( AddFile );
					}
					SameAttrs= sa;
					if( From_Tarchive> 0 ){
						From_Tarchive-= 1;
					}
				}
				  /* 990319: dynamic filenames...?!
				if( strlen(fname)<= MFNAME ){
					printfile= fname;
				}
				else if( !(printfile= rindex( fname, '/')) || strlen(printfile)> MFNAME ){
					printfile= &fname[ strlen(fname)- MFNAME ];
				}
				   */
				printfile= fname;
			}
			if( strm ){
				if( is_pipe ){
				  /* pipes should be closed in a different manner, and the filename should
				   \ be restored if the pipe has been opened on user-request.
				   */
					pclose(strm);
					if( is_pipe== 1 ){
						fname--;
					}
					else if( is_pipe== 2 && RemoveInputFiles ){
						if( debugFlag ){
							fprintf( StdErr, "Removing inputfile \"%s\" as requested\n", fname );
						}
						unlink(fname);
					}
					is_pipe= 0;
				}
				else if( strm!= stdin ){
					fclose(strm);
					if( RemoveInputFiles ){
						if( debugFlag ){
							fprintf( StdErr, "Removing inputfile \"%s\" as requested\n", fname );
						}
						unlink(fname);
					}
				}
			}
read_next_file:;
			if( fname== inFileNames[idx] ){
			  /* This is the "normal" case, processing files specified on the commandline.
			   \ We should thus increment the loop-counter.
			   \ In the next loop, we either read an AddFile, or the next specified file.
			   */
				idx++;
			}
			else{
			  /* We just read an AddFile, which means we only have to free fname.	*/
				if( is_labels ){
					unlink( fname );
					labels_file= NULL;
				}
				xfree(fname);
				added_files+= 1;
			}
			if( idx>= numFiles && !AddFile ){
			  /* There remains nothing to be done? Then to really finish things off,
			   \ we read the labels_file (if specified) as an AddFile.
			   */
				AddFile= labels_file;
			}
		} while( idx< numFiles || AddFile );
	}
all_files_read:;

	numFiles+= added_files;

	if( read_params_now ){
		XG_SimpleConsole();
	}
	read_params_now= False;

	if( maxitems== 0 && setNumber && maxSize ){
	  /* This can happen if there's a read error in the first file, but
	   \ still some data was successfully read.
	   */
		maxitems= maxSize;
	}
	if( maxitems /* && !setNumber */ && AllSets[setNumber].numPoints> 0 ){
		setNumber+= 1;
	}

	if( ReadPipe ){
	  char *sFname= NULL;
		fprintf( StdErr, "xgraph: reading data from \"%s\"\n", ReadPipe_name );
		if( ReadPipe_name[0]== '|' ){
			sFname= tildeExpand( sFname, &ReadPipe_name[1] );
			ReadPipe_fp= popen( sFname, "r");
		}
		else if( strcmp( ReadPipe_name, "-")== 0 ){
			ReadPipe_fp= stdin;
			sFname= XGstrdup("stdin");
		}
		else{
			ReadPipe_fp= fopen( (sFname= tildeExpand( sFname, ReadPipe_name)), "r");
		}
		if( ReadPipe_fp ){
			xfree( ReadPipe_name );
			ReadPipe_name= sFname;
		}
		else{
			fprintf( StdErr, "xgraph: can't open file or pipe \"%s\" (%s) to continuously read from (%s)\n",
				ReadPipe_name, sFname, serror()
			);
			ReadPipe= False;
			xfree( ReadPipe_name );
			xfree( sFname );
		}
	}

	if( (maxitems<= 0 || setNumber<= 0) && !ReadPipe ){
		fprintf( StdErr, "%s: maxitems=%d, setNumber=%d; no data found\n",
			Prog_Name, maxitems, setNumber
		);
/* 		CleanUp();	*/
/* 		exit(0);	*/
		ExitProgramme(0);
	}

	  /* Check if any arguments are specified through the environment	*/
	{ char *c= getenv( "XGRAPH_ARGUMENTS" );
		if( c && *c ){
			if( DumpFile ){
				fprintf( stdout, "*ARGUMENTS*%s\n", c );
			}
			else{
				c= XGstrdup(c);
				ParseArgsString(c);
				xfree(c);
			}
			putenv( "XGRAPH_ARGUMENTS=" );
		}
	}

	if( plot_only_file> numFiles+ file_splits ){
		fprintf( StdErr, "%s: no sets drawn of %d files (%d disk-files/streams) since -plot_only_file %d\n",
			Prog_Name, numFiles+file_splits, numFiles, plot_only_file
		);
		fflush( StdErr );
	}

	  /* Check if things must be appended to the setName	*/
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
			AllSets[idx].setName= String_ParseVarNames( AllSets[idx].setName, "%[", ']', True, True, "xgraph()" );
			AllSets[idx].appendName= NULL;
		}
	}
	  /* eliminate trailing empty datasets	*/
	for( idx= setNumber-1; idx>= 0; idx-- ){
		if( AllSets[idx].numPoints<= 0 ){
			setNumber-= 1;
		}
		else{
			idx= -1;
		}
	}

	if( MAXSETS> 128 ){
		fprintf( StdErr, "Used %d sets out of %d\n", setNumber, MAXSETS );
	}

	Show_Stats( StdErr, AllSets[setNumber-1].fileName, &SS__x, &SS__y, &SS__e, NULL, &SS_SY);

	  /* This is a slightly different detach algorithm: Here the parent reads in
	   \ all data, and is thus blessed with all timing and errors due to datareading
	   */
	if( detach== 2 && !debugging ){
	  pid_t cpid;
		  /* We might get the USR1 signal from the child - setup to ignore it	*/
/* 		signal( SIGUSR1, SIG_IGN );	*/
		signal( SIGUSR1, notify );
		if( (cpid= fork())== -1 ){
			fprintf( StdErr, "Error - couldn't fork/detach (%s)\n", serror() );
		}
		else{
			if( cpid ){
				parent= 1;
				pause();
				if( debugFlag ){
					fprintf( StdErr, "Parent %ld exits; leaves child %ld\n",
						(long) getpid(), (long) cpid
					);
					fflush( StdErr );
				}
				fprintf( StdErr, ".. _exit\n");
				_exit(0);
			}
			else{
				parent= 0;
				signal( SIGUSR1, SIG_IGN );
				Elapsed_Since( &Start, True );
				strcpy( descent, "child" );
				caller_process= (long) getppid();
				if( debugFlag ){
					fprintf( StdErr, "Child %ld will detach from %ld\n", (long) getpid(), caller_process );
					fflush( StdErr );
				}
			}
		}
	}
	  /* Now we are ready to do some local/internal work -
	   * if requested, signal the parent process to exit
	   * (detach), so that e.g. another xgraph process can
	   * be started.
	   * Since we also read our data, it is safe to exit
	   * even if our caller will remove the inputfiles.
	   */
	if( caller_process)
	{
		kill_caller_process();
	}

	if( exit_level || (maxitems== 0 && !ReadPipe) || DumpFile ){
		xfree( comment_buf );
/* 		CleanUp();	*/
/* 		exit( exit_level);	*/
		ExitProgramme(exit_level);
	}

	if( printfile){
		if( printfile[0]== '/' ){
			XGstrcpy( printfile, &printfile[1] );
		}
		xfree( hard_devices[PS_DEVICE].dev_file );
		hard_devices[PS_DEVICE].dev_file= concat2( hard_devices[PS_DEVICE].dev_file, printfile, ".ps", NULL);
		xfree( hard_devices[SPREADSHEET_DEVICE].dev_file );
		hard_devices[SPREADSHEET_DEVICE].dev_file= concat2( hard_devices[SPREADSHEET_DEVICE].dev_file, printfile, ".csv", NULL);
		xfree( hard_devices[CRICKET_DEVICE].dev_file );
		hard_devices[CRICKET_DEVICE].dev_file= concat2( hard_devices[CRICKET_DEVICE].dev_file, printfile, ".cg", NULL);
#ifdef HPGL_DUMP
		xfree( hard_devices[HPGL_DEVICE].dev_file );
		hard_devices[HPGL_DEVICE].dev_file= concat2( hard_devices[HPGL_DEVICE].dev_file, printfile, ".hpgl", NULL);
#endif
#ifdef IDRAW_DUMP
 		xfree( hard_devices[IDRAW_DEVICE].dev_file );
		hard_devices[IDRAW_DEVICE].dev_file= concat2( hard_devices[IDRAW_DEVICE].dev_file, printfile, ".idraw", NULL);	*/
#endif
		xfree( hard_devices[XGRAPH_DEVICE].dev_file );
		hard_devices[XGRAPH_DEVICE].dev_file= concat2( hard_devices[XGRAPH_DEVICE].dev_file, printfile , ".xg", NULL);
		xfree( hard_devices[COMMAND_DEVICE].dev_file );
		hard_devices[COMMAND_DEVICE].dev_file= concat2( hard_devices[COMMAND_DEVICE].dev_file, printfile , ".sh", NULL);
	}
	if( PrintFileName){
		xfree( hard_devices[PS_DEVICE].dev_file );
		hard_devices[PS_DEVICE].dev_file= concat2( hard_devices[PS_DEVICE].dev_file, PrintFileName, ".ps", NULL);
		xfree( hard_devices[SPREADSHEET_DEVICE].dev_file );
		hard_devices[SPREADSHEET_DEVICE].dev_file= concat2( hard_devices[SPREADSHEET_DEVICE].dev_file, PrintFileName , ".csv", NULL);
		xfree( hard_devices[CRICKET_DEVICE].dev_file );
		hard_devices[CRICKET_DEVICE].dev_file= concat2( hard_devices[CRICKET_DEVICE].dev_file, PrintFileName, ".cg", NULL);
#ifdef HPGL_DUMP
		xfree( hard_devices[HPGL_DEVICE].dev_file );
		hard_devices[HPGL_DEVICE].dev_file= concat2( hard_devices[HPGL_DEVICE].dev_file, PrintFileName, ".hpgl", NULL);
#endif
#ifdef IDRAW_DUMP
 		xfree( hard_devices[IDRAW_DEVICE].dev_file );
		hard_devices[IDRAW_DEVICE].dev_file= concat2( hard_devices[IDRAW_DEVICE].dev_file, PrintFileName , ".idraw", NULL);	*/
#endif
		xfree( hard_devices[XGRAPH_DEVICE].dev_file );
		hard_devices[XGRAPH_DEVICE].dev_file= concat2( hard_devices[XGRAPH_DEVICE].dev_file, PrintFileName , ".xg", NULL);
		xfree( hard_devices[COMMAND_DEVICE].dev_file );
		hard_devices[COMMAND_DEVICE].dev_file= concat2( hard_devices[COMMAND_DEVICE].dev_file, PrintFileName , ".sh", NULL);
	}

	if( !strlen( titleText) ){
	  /* Some default title	*/
		strcpy(titleText, "X Graph");
	}
	else if( !titleTextSet ){
	  /* this means titleText is a copy of some (probably the last)
	   \ set's titleText.
	   */
		titleText_0= titleText[0];
		titleText[0]= '\0';
	}

	if( debugFlag){
		fprintf( StdErr, "Data Bounding Box: %g %g (%g,%g(%g)) %g %g\n", llx, lly, llpx, llpy, llny, urx, ury);
	}

	signal( SIGUSR1, Dump_handler);

	realloc_Xsegments();

	if( scale_av_x> 0.0){
		llx= SS_Mean_( SS_X)- scale_av_x* SS_St_Dev_(SS_X);
		urx= SS_Mean_( SS_X)+ scale_av_x* SS_St_Dev_(SS_X);
		fprintf( StdErr, "X values: %s +/- %s (*%g)\n",
			d2str( SS_Mean_( SS_X), "%g", NULL),
			d2str( SS_St_Dev_( SS_X), "%g", NULL),
			scale_av_x
		);
	}
	if( scale_av_y> 0.0){
		lly= SS_Mean_( SS_Y)- scale_av_y* SS_St_Dev_(SS_Y);
		ury= SS_Mean_( SS_Y)+ scale_av_y* SS_St_Dev_(SS_Y);
		fprintf( StdErr, "Y values: %s +/- %s (*%g)\n",
			d2str( SS_Mean_( SS_Y), "%g", NULL),
			d2str( SS_St_Dev_( SS_Y), "%g", NULL),
			scale_av_y
		);
	}

      /* Nasty hack here for bar graphs */
    if( barFlag> 0 ){
	  double _barWidth= barWidth;
		if( barWidth<= 0){
			_barWidth= (urx-llx)/((double)maxSize+1);
		}
		llx -= _barWidth;
		urx += _barWidth;
    }

	CopyFlags( &StubWindow, NULL );

	if( debugFlag ){
		fprintf( StdErr, "Pre Bounding Box x=%g,%g,%g y=%g,%g,%g,%g\n",
			startup_wi.R_UsrOrgX, usrLpX, startup_wi.R_UsrOppX, startup_wi.R_UsrOrgY, usrLpY, usrHnY, startup_wi.R_UsrOppY
		);
		if( use_max_x ){
			fprintf( StdErr, "Pre Bounding X %g %g\n", MusrLX, MusrRX);
		}
		if( use_max_x ){
			fprintf( StdErr, "Pre Bounding Y %g %g\n", MusrLY, MusrRY);
		}
	}

	  /* Check the data-limits against the user-supplied limits.
	   \ The bounding box of the data is clipped against the
	   \ box defined by the limits
	   */
	if( use_max_x ){
		if( llx< MusrLX ){
			llx= MusrLX;
		}
		else if( llx> MusrRX ){
			llx= MusrRX;
		}
		if( urx< MusrLX ){
			urx= MusrLX;
		}
		else if( urx> MusrRX ){
			urx= MusrRX;
		}
	}
	if( use_max_y ){
		if( lly< MusrLY ){
			lly= MusrLY;
		}
		else if( lly> MusrRY ){
			lly= MusrRY;
		}
		if( ury< MusrLY ){
			ury= MusrLY;
		}
		else if( ury> MusrRY ){
			ury= MusrRY;
		}
	}

	  /* If the user didn't supply any bounds, set the bounds
	   \ to what is found in the data. This ensures that NewWindow
	   \ as called below always receives a valid box
	   */
	if( !use_lx ){
		startup_wi.R_UsrOrgX= llx;
		startup_wi.R_UsrOppX= urx;
	}
	else{
		FitOnce= False;
	}
	if( !use_ly ){
		startup_wi.R_UsrOrgY= lly;
		startup_wi.R_UsrOppY= ury;
	}
	else{
		FitOnce= False;
	}
	usrLpX= llpx;
	usrLpY= llpy;
	usrHnY= llny;

	if( debugFlag ){
		fprintf( StdErr, "User BoundingBox x=%g,%g,%g y=%g,%g,%g,%g\n",
			startup_wi.R_UsrOrgX, usrLpX, startup_wi.R_UsrOppX, startup_wi.R_UsrOrgY, usrLpY, usrHnY, startup_wi.R_UsrOppY
		);
		fprintf( StdErr, "BoundingBox x=%g,%g y=%g,%g\n", llx, urx, lly, ury);
		if( use_max_x ){
			fprintf( StdErr, "Bounding X %g %g\n", MusrLX, MusrRX);
		}
		if( use_max_x ){
			fprintf( StdErr, "Bounding Y %g %g\n", MusrLY, MusrRY);
		}
	}

    /* Create initial window */
#ifdef TOOLBOX
    xtb_init(disp, screen, normPixel, bgPixel, dialogFont.font, dialog_greekFont.font, False );
#endif
	  /* Standard window-name prefix: XGraph id, window serial, and aspect ratio of the graph	*/
	sprintf( Window_Name, "%%s %%s%%02d.%%02d.%%02d x/y=%%s:%%s " );
#ifdef STATIC_TITLE
	  /* Filename:	*/
	if( PrintFileName){
		strncat( Window_Name, " ", 256- strlen(Window_Name) );
		strncat( Window_Name, PrintFileName, 256- strlen(Window_Name) );
	}
	else if( printfile){
		strncat( Window_Name, " ", 256- strlen(Window_Name) );
		strncat( Window_Name, printfile, 256- strlen(Window_Name) );
	}
	  /* Y vs. X	*/
	{ char *c;
		strncat( Window_Name, " - ", 256- strlen(Window_Name) );
		if( (c= index( YUnits, ';')) ){
			*c= '\0';
		}
		strncat( Window_Name, YUnits, 256- strlen(Window_Name) );
		if( c){
			*c= ';';
		}
		strncat( Window_Name, " vs. ", 256- strlen(Window_Name) );
		if( (c= index( XUnits, ';')) ){
			*c= '\0';
		}
		strncat( Window_Name, XUnits, 256- strlen(Window_Name) );
		if( c){
			*c= ';';
		}
	}
#else
	strncat( Window_Name, "%s %s vs. %s%s", 256- strlen(Window_Name) );
#endif

	  /* highlightPixel can be initialised only now, since it may depend on xtb_light_pix	*/
	if( highlightPixel== -1){
	  extern Pixel xtb_light_pix;
		highlightPixel = xtb_light_pix;
		if( debugFlag){
			fprintf( StdErr, "highlightPixel (*HighlightColor) set to xtb_light_pix\n");
			fflush( StdErr);
		}
	}

	  /* draw-timing info:	*/
	strncat( Window_Name, " [%ss;%ld] $=%ld%s", 256- strlen(Window_Name) );
    primary= NewWindow(Window_Name, &primary_info,
		startup_wi.R_UsrOrgX, startup_wi.R_UsrOrgY, usrLpX, usrLpY, usrHnY,
		startup_wi.R_UsrOppX, startup_wi.R_UsrOppY, NORMASP, NULL, 1.0, 1.0, 1.0, 1
	);
    if( !primary ){
		(void) fprintf(StdErr, "Main window would not open\n");
		xfree( comment_buf );
/* 		CleanUp();	*/
/* 		exit(1);	*/
		ExitProgramme(1);
    }
	else{

		if( InitWindow ){
			DelWindow( InitWindow->window, InitWindow );
			InitWindow= NULL;
			init_window= 0;
		}

		SS_Reset_(SS_mX);
		SS_Reset_(SS_mY);
		SS_Reset_(SS_mE);
		SS_Reset_(SS_mI);
		SAS_mO.Gonio_Base= radix;
		SAS_mO.Gonio_Offset= radix_offset;
		SAS_mO.Nvalues= 0;
		SAS_mO.sample= NULL;
		SAS_mO.exact= 0;
		SAS_Reset_(SAS_mO);
		SS_Reset_(SS_mLY);
		SS_Reset_(SS_mHY);
		SS_Reset_(SS_mMO);
		SS_Reset_(SS_mSO);
		SS_Reset_(SS_mPoints);
		UpdateWindowSettings( primary_info, True, True );
		  /* If necessary, initialise the new window's user defined labels	*/
		Copy_ULabels( primary_info, &StubWindow );
		use_lx= 0;
		use_ly= 0;
	}

	XUnits[0]= '\0';
	YUnits[0]= '\0';
	tr_XUnits[0]= '\0';
	tr_YUnits[0]= '\0';

	RecolourCursors();

	  /* Tell d2str it can now safely use a "graphic" representation
	   \ of +- Inf using the Symbol font.
	   */
	use_greek_inf= 1;

	StartUp= False;

	signal( SIGCONT, cont_handler );

	AGC= *AllowGammaCorrection;

	if( Xsegs && XsegsE && XXsegs){
	  Time_Struct evtimer;
	  double wait_time;

		setjmp( toplevel );

		if( ACrossGC == (GC) 0 ){
			unsigned long gcmask;
			XGCValues gcvals;

			gcmask = ux11_fill_gcvals(&gcvals, GCForeground, zeroPixel ^ bgPixel,
						  GCFunction, GXxor, GCFont, fbFont.font->fid,
						  UX11_END);
			ACrossGC = XCreateGC(disp, primary_info->window, gcmask, &gcvals);
			gcmask = ux11_fill_gcvals(&gcvals, GCForeground,
/* 							highlightPixel ^ bgPixel,	*/
							gridPixel ^ bgPixel,
						  GCFunction, GXxor, GCFont, dialogFont.font->fid,
						  UX11_END);
			BCrossGC = XCreateGC(disp, primary_info->window, gcmask, &gcvals);
		}

		while( (setNumber> 0 || ReadPipe) && Num_Windows > 0 ){
		  int check_first= ReadPipe;

			if( RelayEvent ){
				XSendEvent( disp, RelayEvent->xany.window, False, RelayMask, (XEvent*) RelayEvent );
				fprintf( StdErr, "toplevel: sent a relayed event!\n" );
				RelayEvent= NULL;
				RelayMask= 0;
			}

			if( setNumber && Handle_MaybeLockedWindows(True) ){
				check_first= 1;
			}
			  /* Some actions generate a plenitude of events and server-internal actions
			   \ like the opening of the settings dialog. We must be careful not to disturb
			   \ such chains of actions, esp. if xtb_register() is involved: failure can
			   \ lead to corrupted data-structures (at least on my IRIX 6.3 X R6 server), which
			   \ in most cases ultimately lead to a crash. Therefore, the "wait_evtimer" mechanism
			   \ allows to spend some time doing other things (like event-handling) after opening
			   \ the settings dialog, e.g., before doing another complex thing, like reading a
			   \ script file. The amount of time waited depends on whether or not we're in synchronous
			   \ mode. In any event, we continously check against the availability of events and
			   \ the time passed: when more than half the waiting time has passed, and there are no
			   \ more events to process, we "busy-wait" the rest of the time, checking all the while
			   \ as events may come in bursts (given the distributed nature of X).
			   */
			if( wait_evtimer ){
			  int N= 0;
			  double ltime= 0;
				do{
					Elapsed_Since( &evtimer, False );
					if( evtimer.Tot_Time>= wait_time ){
						wait_evtimer= False;
					}
					else if( debugFlag && debugLevel ){
						fprintf( StdErr, "xgraph: %g<%g seconds passed since last time-requiring complex event\n",
							evtimer.Tot_Time, wait_time
						);
					}
/* 					setjmp( stop_wait_event );	*/
					if( !wait_evtimer ){
					  LocalWindows *WL= WindowList;
					  LocalWin *wi;
					  int n= 0;
						  /* Great, we can continue to run unrestrained. First, redraw all windows
						   \ having queued a redraw. (likely there is only one window..)
						   */
						while( WL && (wi= WL->wi) ){
							if( wi->redraw || !wi->redrawn ){
								RedrawNow( wi );
								n+= 1;
							}
							WL= WL->next;
						}
						if( debugFlag || ltime> 0.5 ){
							fprintf( StdErr,
								"xgraph: %g seconds passed since last time-requiring complex event "
								"- handled %d queued redraws (%d busywait loops in %g seconds)\n",
								evtimer.Tot_Time, n, N, evtimer.Tot_Time- ltime
							);
						}
					}
					if( !XEventsQueued( disp, QueuedAfterFlush) ){
						if( !N ){
							ltime= evtimer.Tot_Time;
						}
						N+= 1;
					}
				  /* Loop (busy-waiting!!) when there are no events queued. This prevents getting
				   \ blocked.
				   */
				} while( !XEventsQueued( disp, QueuedAfterFlush) && wait_evtimer &&
					evtimer.Tot_Time>= wait_time/2.0 && evtimer.Tot_Time< wait_time
				);
				if( N>= 1 ){
				  /* We probably depleted the event queue. So prevent getting stuck:	*/
					check_first= 1;
				}
			}

			if( *AllowGammaCorrection!= AGC ){
				AGC= *AllowGammaCorrection;
				ReallocColours( True );
			}

			  /* Check if there are (still) windows
			   \ that want a redraw. This implements animation. It is only performed in
			   \ the toplevel event-loop, and then only if no events are queued. On the
			   \ other hand, we don't block-wait (XNextEvent) if the Animating flag is set
			   \ and there are no events in the queue.
			   */
			if( Animating ){
			  int queued= XEventsQueued( disp, QueuedAfterFlush);
			  LocalWindows *WL= WindowList;
			  extern double *ascanf_SyncedAnimation;
				if( (animate= !Exitting) ){
					Animating= False;
				}
				WL= WindowList;
				Elapsed_Since( &Animation_Timer, True );
				while( WL && animate && !queued && !Exitting ){
					animate= 0;
					animation_windows= 0;
					do{
					  LocalWin *lwi= WL->wi;
						if( !lwi->silenced ){
							if( lwi->animate && !lwi->halt ){
								xtb_bt_swap( lwi->settings );
								xtb_bt_swap( lwi->close );
								lwi->printed= 0;
								  /* We start animating the window at this point:	*/
								lwi->animating= True;
								RedrawAgain( lwi );
								if( lwi->delete_it!= -1 ){
									xtb_bt_swap( lwi->settings );
									xtb_bt_swap( lwi->close );
									if( lwi->animate ){
										animate+= lwi->animate;
									}
									else{
										lwi->animating= False;
									}
									if( !lwi->animate && lwi->draw_count> 0 ){
										Elapsed_Since(&lwi->draw_timer, True);
/* 										SetWindowTitle( lwi, lwi->draw_timer.Tot_Time );	*/
										SetWindowTitle( lwi, lwi->draw_timer.HRTot_T );
									}
								}
								animations+= 1;
								animation_windows+= 1;
							}
							else{
								lwi->animating= False;
							}
						}
						WL= WL->next;
					} while( WL && WL!= WindowList && WindowList && !Exitting );
					WL= WindowList;
					queued= XEventsQueued( disp, QueuedAfterFlush);
					if( animation_windows ){
						SS_Add_Data_( Animation_Windows, 1, animation_windows, 1.0);
					}
				}
				if( *ascanf_SyncedAnimation && !Exitting ){
					XSync( disp, False );
				}
				if( animations ){
					Elapsed_Since( &Animation_Timer, False );
					Animation_Time+= Animation_Timer.HRTot_T;
					Animations+= animations;
					animations= 0;
				}
			}
			if( !Handle_An_Event(1, check_first, "main", 0, 0 ) ){
				if( ReadPipe ){
				  struct itimerval rtt, ortt;
					rtt.it_value.tv_sec= 0;
					  /* 60Hz sleep/wakeup cycle in absence of X events: */
					rtt.it_value.tv_usec= 1000000 / 60;
					rtt.it_interval.tv_sec= 0;
					rtt.it_interval.tv_usec= 0;
					signal( SIGALRM, X_sleep_wakeup );
					wakeup= 0;
					setitimer( ITIMER_REAL, &rtt, &ortt );
					pause();
					  /* restore the previous setting of the timer.	*/
					setitimer( ITIMER_REAL, &ortt, &rtt );
				}
			}
			check_first= ReadPipe;

			if( settings_immediate ){
				settings_immediate= 0;
				primary_info->pw_placing= PW_MOUSE;
				DoSettings( primary_info->window, primary_info );
				Elapsed_Since( &evtimer, True );
				Elapsed_Since( &evtimer, True );
				  /* This is an action that really needs to start the wait_timer.	*/
				wait_evtimer= True;
				wait_time= (Synchro_State)? 3.0 : 1.0;
			}

			if( !wait_evtimer ){
				if( ScriptFile ){
					ReadScriptFile( (ScriptFileWin)? ScriptFileWin : primary_info );
				}

				if( ReadPipe ){
#ifndef __MACH__
				  struct pollfd check[1];
				  int r;
					check[0].fd= fileno(ReadPipe_fp);
					check[0].events= POLLIN|POLLPRI|POLLRDNORM|POLLRDBAND;
					check[0].revents= 0;
					if( (r= poll( check, 1, 100 ))> 0 && !CheckORMask(check[0].revents, POLLHUP|POLLERR|POLLNVAL) )
#endif
					{
						IncludeFile( primary_info, ReadPipe_fp, ReadPipe_name, True, NULL );
						if( !ReadPipe ){
							if( ReadPipe_name[0]== '|' ){
								pclose(ReadPipe_fp);
							}
							else if( ReadPipe_fp!= stdin ){
								fclose(ReadPipe_fp);
							}
							ReadPipe_fp= NULL;
							xfree(ReadPipe_name);
						}
					}
#if defined(DEBUG) && !defined(__MACH__)
					else if( r<= 0 ){
						fprintf( StdErr, "Poll \"%s\" returns %d, revents=%d: %s\n",
							ReadPipe_name, r, check[0].revents, serror()
						);
					}
#endif
				}
			}

		}
		xfree( Xsegs );
		xfree( XXsegs );
		xfree( XsegsE );
	}

	for( idx= 0; idx< MaxSets; idx++ ){
		Destroy_Form( &AllSets[idx].process.C_set_process );
	}

	for( idx= 0; idx< MaxnumFiles; idx++ ){
		xfree( inFileNames[idx] );
	}
    xfree( inFileNames);
	xfree( InFiles );
	xfree( InFilesTStamps );
	xfree( comment_buf );
    ExitProgramme(0);
    xfree( AllSets);
	return(0);
}

/*
 * Button handling functions
 */

extern Window thePrintWindow, theSettingsWindow;

extern char *XGFetchName( LocalWin *wi);

#ifdef ZOOM

#define DRAWBOX(c) \
if( startX < curX ){ \
   boxEcho.x = startX; \
   boxEcho.width = curX - startX; \
} else { \
   boxEcho.x = curX; \
   boxEcho.width = startX - curX; \
} \
if( startY < curY ){ \
   boxEcho.y = startY; \
   boxEcho.height = curY - startY; \
} else { \
   boxEcho.y = curY; \
   boxEcho.height = startY - curY; \
} \
XDrawRectangles(disp, win, HandleMouseGC, &boxEcho, 1);\
if(c && CursorCross ){\
	DrawCCross( wi, True, curX, curY, box_coords );\
}

#ifdef DEBUG_MEASURE
#	define DRAWLINE(c) \
   lineEcho.x1 = startX; \
   lineEcho.x2 = curX; \
   lineEcho.y1 = startY; \
   lineEcho.y2 = curY; \
XDrawSegments(disp, win, HandleMouseGC, &lineEcho, 1); \
if( measure==1 && diag && diag!= prev_diag){\
	_arc_X( disp, win, HandleMouseGC, startX, startY, (int)(radx+ 0.5), (int)(rady+ 0.5), 0.0, 360.0);\
		sprintf( box_coords, "[%g,%g,%g]", radx* wi->XUnitsPerPixel, rady* wi->YUnitsPerPixel, diag);\
		XDrawString( disp, wi->window, HandleMouseGC,\
			curX+ 5, curY+ 5, box_coords, strlen(box_coords)\
		);\
}\
if(c && CursorCross ){\
	DrawCCross( wi, True, curX, curY, box_coords );\
}
#else
#	define DRAWLINE(c) \
   lineEcho.x1 = startX; \
   lineEcho.x2 = curX; \
   lineEcho.y1 = startY; \
   lineEcho.y2 = curY; \
XDrawSegments(disp, win, HandleMouseGC, &lineEcho, 1); \
if( measure==1 && diag && diag!= prev_diag){\
	_arc_X( disp, win, HandleMouseGC, startX, startY, (int)(radx+ 0.5), (int)(rady+ 0.5), 0.0, 360.0);\
}\
if(c && CursorCross ){\
	DrawCCross( wi, True, curX, curY, box_coords );\
}
#endif

#define DRAWECHO(c) if( (measure && !apply2box) || wi->add_label){ DRAWLINE(c); } else { DRAWBOX(c); }

#endif

#ifdef ZOOM

#define	TAB			'\t'
#define	BACKSPACE	0010
#define DELETE		0177
#define	CONTROL_P	0x1b	/* actually ESC	*/
#define CONTROL_U	0025
#define CONTROL_W	0027
#define CONTROL_X	0030

static int XG_df_accepted= 0;

static xtb_hret XG_df_fun(win, ch, text, val)
Window win;			/* Widget window   */
int ch;				/* Typed character */
char *text;			/* Copy of text    */
xtb_data val;			/* User info       */
/*
 * This is the handler function for the text widget for
 * specifing the file or device name.  It supports simple
 * line editing operations.
 */
{
  char Text[MAXCHBUF];
  int accept;
  extern char *word_sep;

    if( (ch == BACKSPACE) || (ch == DELETE) ){
		if( !( (ch== DELETE)? xtb_ti_dch_right(win) : xtb_ti_dch(win)) )
			Boing( 5);
		return( XG_df_fun( win, 0, text, val) );
    }
	else if( (ch == CONTROL_U) || (ch == CONTROL_X) ){
		(void) xtb_ti_set(win, "", (xtb_data) 0);
		return( XG_df_fun( win, 0, text, val) );
    }
	else if( ch== CONTROL_W){
	  char *str;
		if( *text)
			str= &text[ strlen(text)-1 ];
		else{
			Boing( 5);
			return( XG_df_fun( win, 0, text, val) );
		}
		if( index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
			}
			return( XG_df_fun( win, 0, text, val) );
		}
		while( *str && !index( word_sep, *str) ){
			if( !xtb_ti_dch(win) ){
				Boing( 5);
				return( XG_df_fun( win, 0, text, val) );
			}
			str--;
		}
	}
	else if( ch && ch!= XK_Meta_L && ch!= XK_Meta_R && ch!= XK_Alt_L && ch!= XK_Alt_R &&
		ch!= XK_Super_L && ch!= XK_Super_R && ch!= XK_Hyper_L && ch!= XK_Hyper_R &&
		ch!= XK_Down && ch!= XK_Up
	){
	  /* Insert if printable - ascii dependent */
		if( !xtb_ti_ins(win, ch) ){
			Boing( 5);
		}
    }
	xtb_ti_get( win, Text, (xtb_data) NULL );
	if( (accept= (ch== XK_Down || ch== XK_Up || ch== 0x12 )) ){
		XG_df_accepted= 1;
	}
    return XTB_HANDLED;
}

char *XG_GetString( LocalWin *wi, char *text, int maxlen, Boolean do_events )
{ ALLOCA( msg, char, maxlen+32, msg_len);

	XG_df_accepted= 0;
	if( !wi->label_IFrame.win ){
	  Pixel xbp= xtb_back_pix, xnp= xtb_norm_pix, xlp= xtb_light_pix, xmp= xtb_middle_pix;

		xtb_back_pix= xtb_white_pix;
		xtb_norm_pix= xtb_black_pix;
		xtb_light_pix= xtb_Lgray_pix;
		xtb_middle_pix= xtb_Mgray_pix;
		xtb_ti_new( wi->window, text, 50, maxlen,
			XG_df_fun, (xtb_data) 0, &wi->label_IFrame
		);
		xtb_back_pix= xbp;
		xtb_norm_pix= xnp;
		xtb_light_pix= xlp;
		xtb_middle_pix= xmp;
	}
	else if( wi->label_IFrame.win ){
		XMapRaised( disp, wi->label_IFrame.win );
		xtb_ti_set( wi->label_IFrame.win, text, NULL);
	}
	if( wi->label_IFrame.win ){
	  XWindowAttributes win_attr;
	  XEvent evt;
		XGetWindowAttributes(disp, wi->window, &win_attr);
		XMoveWindow(disp, wi->label_IFrame.win,
			(int) (win_attr.width- (BTNPAD+ wi->label_IFrame.width+ 1* BTNINTER))/ 2,
			(int) (win_attr.height- (BTNPAD+ wi->label_IFrame.height+ 1* BTNINTER))/ 2
		);
		XSetInputFocus( disp, wi->label_IFrame.win, RevertToParent, CurrentTime);
		while( XEventsQueued( disp, QueuedAfterFlush)> 0){
			XNextEvent( disp, &evt );
			xtb_dispatch( disp, wi->window, 1, &wi->label_IFrame, &evt );
		}
		{
		  int xtb_return;
			do {
			  extern xtb_frame SD_Dialog, HO_Dialog, sd_af[], ho_af[];
			  extern int ho_last_f, sd_last_f;
				XNextEvent(disp, &evt);
				xtb_return= xtb_dispatch( disp, wi->window, 1, &wi->label_IFrame, &evt );
				if( xtb_return!= XTB_HANDLED && xtb_return!= XTB_STOP ){
					xtb_return= xtb_dispatch(disp,SD_Dialog.win, sd_last_f, sd_af, &evt);
					if( xtb_return!= XTB_HANDLED && xtb_return!= XTB_STOP && HO_Dialog.mapped> 0 ){
						xtb_return= xtb_dispatch(disp, HO_Dialog.win, ho_last_f, ho_af, &evt);
					}
					xtb_ti_get( wi->label_IFrame.win, text, NULL);
					sprintf( msg, "XG_GetString(\"%s\")", text );
					if( do_events && xtb_return!= XTB_HANDLED && xtb_return!= XTB_STOP ){
						_Handle_An_Event( &evt, 1, 0, msg );
					}
				}
			} while( xtb_return != XTB_STOP && !XG_df_accepted );
		}
		xtb_ti_get( wi->label_IFrame.win, text, NULL);
		XUnmapWindow( disp, wi->label_IFrame.win );
		wi->label_IFrame.mapped= 0;
		return( text );
	}
	else{
		return( NULL );
	}
}

#define GET_SHIFTNEIGHBOURS(set,idx,slx,sly,srx,sry)	{\
	if(idx==0){\
		slx=set->xvec[idx];\
		sly=set->yvec[idx];\
	}\
	else{\
		slx=set->xvec[idx-1];\
		sly=set->yvec[idx-1];\
	}\
	if(idx==set->numPoints-1){\
		srx=set->xvec[idx];\
		sry=set->yvec[idx];\
	}\
	else{\
		srx=set->xvec[idx+1];\
		sry=set->yvec[idx+1];\
	}\
}

GC HandleMouseGC = (GC) 0;

extern unsigned long mem_alloced;

void check_marked_hlt( LocalWin *wi, Boolean *all_marked, Boolean *all_hlt, Boolean *none_marked, Boolean *none_hlt )
{  int idx, marked= 0, hlt= 0, md= 0, hd= 0, drawn= 0;
	*all_marked= True;
	*all_hlt= True;
	*none_marked= False;
	*none_hlt= False;
	for( idx = 0; idx < setNumber;  idx++ ){
		if( !draw_set( wi, idx ) ){
			if( wi->mark_set[idx]> 0 ){
				*all_marked= False;
				marked+= 1;
			}
			if( wi->legend_line[idx].highlight> 0 ){
				*all_hlt= False;
				hlt+= 1;
			}
		}
		else{
			drawn+= 1;
			if( wi->mark_set[idx]<= 0 ){
				*none_marked= True;
			}
			else{
				marked+= 1;
				md+= 1;
			}
			if( wi->legend_line[idx].highlight<= 0 ){
				*none_hlt= True;
			}
			else{
				hlt+= 1;
				hd+= 1;
			}
		}
	}
	if( marked>= setNumber ){
		*all_marked= True;
	}
	else if( marked== 0 ){
		*all_marked= False;
		*none_marked= False;
	}
	if( hlt>= setNumber ){
		*all_hlt= True;
	}
	else if( hlt== 0 ){
		*all_hlt= False;
		*none_hlt= False;
	}
	if( drawn< setNumber ){
		if( *all_marked || md ){
			*none_marked= False;
		}
		if( *all_hlt || hd ){
			*none_hlt= False;
		}
	}
}

UserLabel *update_LinkedLabel( LocalWin *wi, UserLabel *new, DataSet *point_label_set, int point_l_nr,
	Boolean short_label
)
{ char *buf, buf2[2*MAXAXVAL_LEN+256];
  char *vx, *vy;
  ValCategory *vcat;

	new->set_link= point_label_set->set_nr;
	new->pnt_nr= point_l_nr;

	sprintf( buf2, "#%d: ", point_l_nr );
	if( new->label[0]!= '\0' && !strstr( new->label, buf2) ){
	  /* this is a point-linked label with static text.	*/
		return( new );
	}

	new->short_flag= short_label;
	new->x1= point_label_set->xvec[point_l_nr];
	new->y1= point_label_set->yvec[point_l_nr];
	if( new->nobox ){
		new->x2= new->x1;
		new->y2= new->y1;
	}
	new->eval= point_label_set->errvec[point_l_nr];
	if( wi->ValCat_X_axis && (vcat= Find_ValCat( wi->ValCat_X, new->x1, NULL, NULL)) ){
		vx= vcat->vcat_str;
	}
	else{
		vx= d2str( new->x1, "%g", NULL);
	}
	if( wi->ValCat_Y_axis && (vcat= Find_ValCat( wi->ValCat_Y, new->y1, NULL, NULL)) ){
		vy= vcat->vcat_str;
	}
	else{
		vy= d2str( new->y1, "%g", NULL);
	}
	sprintf( buf2, "#%d: (%s,%s", point_l_nr, vx, vy);
	if( point_label_set->columns[point_label_set->ecol][point_l_nr] ){
		  /* 990710: doesn't make much of a difference...
		if( wi->raw_display || point_label_set->raw_display ){
			sprintf( buf2, "%s %s %s)", buf2, (wi->vectorFlag)? "\\#xd0\\" : "\\#xb1\\",
				d2str( point_label_set->columns[point_label_set->ecol][point_l_nr], "%g", NULL)
			);
		}
		else
		   */
		{
			sprintf( buf2, "%s %s %s)", buf2, (wi->vectorFlag)? "\\#xd0\\" : "\\#xb1\\",
				d2str( point_label_set->errvec[point_l_nr], "%g", NULL)
			);
		}
	}
	else{
		strcat( buf2, ")" );
	}
#if ADVANCED_STATS == 1
	sprintf( buf2, "%s, N=%d", buf2, point_label_set->N[point_l_nr] );
#elif ADVANCED_STATS == 2
	sprintf( buf2, "%s, N=%s", buf2, d2str( NVAL( point_label_set, point_l_nr), "%g", NULL) );
#endif
	if( short_label ){
		strncpy( new->label, buf2, MAXBUFSIZE);
	}
	else{
		if( (buf= concat( point_label_set->setName, ":\n", buf2, NULL)) ){
			strncpy( new->label, buf, MAXBUFSIZE);
			xfree(buf);
		}
	}
	return( new );
}

int ShiftLineEchoSegments= 20;

/* Handles the various functions of the mouse(buttons). I'd like to add
 \ a panning option sometime. And maybe rewrite this code... can probably
 \ be cleaned up considerably.
 */
int HandleMouse( char *progname, XButtonPressedEvent *evt, LocalWin *wi, LocalWin **New_Info, Cursor *cur)
{ Window win, new_win;
  LocalWin *new_info= NULL;
  Window root_rtn, child_rtn;
  XEvent theEvent;
  int dX, dY, startX, startY, curX, curY, newX, newY, stopFlag, numwin= 0;
  int root_x, root_y;
  int i, add_padding= 1, cloning= 0;
/*   int user_coordinates= 2;	*/
    /* 20020930: */
  int user_coordinates= wi->win_geo.user_coordinates;
  unsigned int mask_rtn_pressed= 0, mask_rtn_released= 0, capslock;
  double _loX, _loY, _lopX, _lopY, _hinY, _hiX, _hiY, xprec, yprec;
  char *name= NULL, box_coords[256];
  int raw_display, set_raw_display= 0, allow_name_trans= (wi->raw_display)? True : False;
  XRectangle boxEcho;
  XSegment lineEcho, shiftEcho[4];
  int shiftEchos= 0, lineEchos= 0, dlE;
  ALLOCA( lineEchoSeg, XSegment, ShiftLineEchoSegments, lineEchoSeg_len);
  int measure= 0, point_label= 0, point_l_nr= -1, sx_min, sy_min, sx_max, sy_max;
  static DataSet **displ_set= NULL;
  int displ_sets= 1;
  DataSet *point_label_set= NULL;
  UserLabel *ULabel= NULL;
  double point_l_x, point_l_y, slx, sly, srx, sry, clikX, clikY;
  double radx, rady, diag, prev_diag;
  Boolean handled= False, del_upto= False, del_from= False, PGrabbed = False,
	KGrabbed= False, apply2box= False, displacing= False, CCdrawn= False;
  static const int buttonMask[] = { 0, Button1Mask, Button2Mask, Button3Mask, Button4Mask, Button5Mask };
  static const char *bnames[]= { "", "Button1", "Button2", "Button3", "Button4", "Button5" };

	raw_display= wi->raw_display;

	if( !setNumber ){
		return(0);
	}

	if( !evt ){
	  /* we are called to duplicate window <wi>	*/
		_loX= wi->loX;
		_loY= wi->loY;
		_lopX= wi->lopX;
		_lopY= wi->lopY;
		_hinY= wi->hinY;
		_hiX= wi->hiX;
		_hiY= wi->hiY;
		add_padding= 0;
		stopFlag= 1;
		user_coordinates= wi->win_geo.user_coordinates;
		cloning= 1;
		  /* This is said not to be good programming practice. It is so easy, however.... :))	*/
		goto clone_window;
	}

	if( wi->silenced ){
		GCA();
		return(0);
	}

    win = evt->window;

	set_Find_Point_precision( wi->R_XUnitsPerPixel/2.0, wi->R_YUnitsPerPixel/2.0, &xprec, &yprec );

	if( debugFlag && debugLevel==-2 ){
	  int bstate=evt->state, bbutton= evt->button;
		XQueryPointer(disp, win, &root_rtn, &child_rtn, &root_x, &root_y,
				    &curX, &curY, &mask_rtn_pressed
		);
		fprintf( StdErr, "HandleMouse event: %s == 0x%lx (%s)", event_name(((XEvent*)evt)->xany.type),
			mask_rtn_pressed, xtb_modifiers_string(mask_rtn_pressed) );
		fprintf( StdErr, " state=0x%lx (%s) button=%d (%s)",
			bstate, xtb_modifiers_string(bstate),
			bbutton, (bbutton>=0 && bbutton< sizeof(bnames)/sizeof(char*))? bnames[bbutton] : "?!"
		);
		fprintf( StdErr, " cur=[%d,%d]/[%d,%d] root=[%d,%d]/[%d,%d]\n",
			   curX, curY, evt->x, evt->y,
			   root_x, root_y, evt->x_root, evt->y_root
		);
		// 20120508: weirdness. XQueryPointer() does not return mask_rtn_pressed with all modifiers set properly
		// in all cases. On Mac OS X 10.6 (and possibly 10.5) it does not register Mod1Mask when Button1 is held.
		// ShiftMask also seems to be missing together with Mod1Mask.
		// Fortunately this information *is* present in evt->state at this time.
		if( CheckMask(evt->state, Mod1Mask) && !CheckMask(mask_rtn_pressed, Mod1Mask) ){
			mask_rtn_pressed |= evt->state;
		}
	}
	curX = evt->x, curY = evt->y;
	mask_rtn_pressed= xtb_Mod2toMod1( evt->state );
	  /* We don't use the LockMask, so let's unset it...	*/
	capslock= CheckMask( mask_rtn_pressed, LockMask);
	mask_rtn_pressed&= ~LockMask;
	// splice in the Button mask
	mask_rtn_pressed |= buttonMask[evt->button];

	  /* 20050114: hardcoded mouse scroll wheel support on the button4/5 'z' axis.
	   \ These *buttonpress* events do not show up in mask_rtn_pressed obtained via
	   \ XQueryPointer (because called after receiving the mouse event, and it is not
	   \ a 'held' button?). They do show up in ButtonReleased events, though. Anyway,
	   \ update the mask, which we use below to process the event, so that we do not
	   \ take spurious action.
	   \ NB: the code below associates an empty mask_rtn_pressed to a Button1 press!
	   */
	switch( evt->button ){
		case Button4:
			if( CheckMask(mask_rtn_pressed, ControlMask) ){
				XWarpPointer( disp, None, None, 0,0,0,0, 1, 0 );
			}
			else{
				XWarpPointer( disp, None, None, 0,0,0,0, 0, -1 );
			}
			GCA();
			set_Find_Point_precision( xprec, yprec, NULL, NULL );
			return(0);
			break;
		case Button5:
			if( CheckMask(mask_rtn_pressed, ControlMask) ){
				XWarpPointer( disp, None, None, 0,0,0,0, -1, 0 );
			}
			else{
				XWarpPointer( disp, None, None, 0,0,0,0, 0, 1 );
			}
			GCA();
			set_Find_Point_precision( xprec, yprec, NULL, NULL );
			return(0);
			break;
	}

	CheckProcessUpdate(wi, True, True, True );
#if 0
	if( !wi->raw_display ){
	  DataSet *this_set;
	  Boolean done= False;
		for( i= 0; i< setNumber && !done; i++ ){
			this_set= &AllSets[i];
			if( draw_set( wi, i) ){
				if( this_set->last_processed_wi!= wi &&
					(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len ||
						(!*disable_SET_PROCESS && this_set->process.set_process_len)
					)
				){
					RedrawNow( wi );
					done= True;
				}
			}
		}
	}
#endif
#pragma mark ***HM Pre interactive loop
	if( CheckMask(mask_rtn_pressed, ControlMask) || CheckMask(mask_rtn_pressed,Mod1Mask) ){
	  Boolean done= False;
		  /* 980902: Changed this block to add deleting functionality to the middle and right
		   \ mousebuttons (i.e. when wi->cutAction&_deleteAction). This should not interfere with the
		   \ other functionalities of those buttons, but this routine is getting so complex
		   \ from a conditional/combinatorial point of view... (huh.. AI = brittle?? :)).
		   \ There is an almost litteral repetition of this block somewhat below, after
		   \ the first next XQueryPointer call.
		   */
				  /* 20010524: initialise point_l_[xy] here, instead of under the 20000131 comment below:	*/
				startX = curX;  startY = curY;
				point_l_x= Reform_X( wi, TRANX(startX), TRANY(startY) );
				point_l_y= Reform_Y( wi, TRANY(startY), TRANX(startX) );
		if( CheckMask(mask_rtn_pressed,Mod1Mask) ){
			if( CheckMask(mask_rtn_pressed, Button1Mask) ){
			  /* This case handles displacing datapoints.	*/
				  /* 20000131: the lines upto Find_Point were within the non-raw_mode special block.
				   \ Moved them above so that we can have a single place to initiate set-group displacements.
				   */
				point_l_nr= Find_Point( wi, &point_l_x, &point_l_y, &point_label_set, 1, &ULabel, True, True, True, True );
				if( !wi->raw_display && (wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len) ){
					if( !(wi->cutAction || wi->filtering) ){
						if( wi->ulabel && point_l_nr>= 0 && ULabel ){
						  /* We clicked closest to a ULabel: no need to do a raw redraw...	*/
							set_raw_display= False;
						}
						else{
							set_raw_display= True;
							point_l_nr= -1;
						}
					}
				}
				measure= 2;
				if( !wi->cutAction && !wi->filtering ){
					*cur= (CursorCross)? noCursor : theCursor;
				}
				done= True;
				  /* 20050114: */
				displacing= (wi->filtering)? False : True;
				if( point_label_set ){
					xfree( displ_set );
					if( capslock && CheckMask( mask_rtn_pressed, ShiftMask) ){
					  int i, gr= wi->group[point_label_set->set_nr];
						files_and_groups( wi, NULL, NULL );
						displ_sets= 0;
						  /* Find out how many sets belong to this group:	*/
						for( i= 0; i< setNumber; i++ ){
							if( wi->group[i]== gr && draw_set(wi, i) ){
								displ_sets+= 1;
							}
						}
						  /* Store references to the sets, and retrieve their global bounding box:	*/
						if( displ_sets> 1 && (displ_set= (DataSet**) calloc( displ_sets, sizeof(DataSet*))) ){
						  int j= 0;
							for( i= 0; i< setNumber && j< displ_sets; i++ ){
								if( wi->group[i]== gr && draw_set(wi, i) ){
									displ_set[j++]= &AllSets[i];
								}
							}
						}
						else{
							displ_sets= 1;
						}
					}
				}
			}
			if( wi->cutAction & _deleteAction ){
				if( CheckMask(mask_rtn_pressed, Button2Mask) ){
					measure= 2;
					del_upto= True;
					done= True;
				}
				else if( CheckMask(mask_rtn_pressed, Button3Mask) ){
					measure= 2;
					del_from= True;
					done= True;
				}
			}
			if( CheckMask(mask_rtn_pressed, Button3Mask) ){
				done= True;
				point_label= 1;
				if( wi->filtering ){
				  /* PointFilter	*/
					apply2box= False;
					measure= 2;
				}
				if( CheckMask( mask_rtn_pressed, ControlMask) ){
					wi->add_label= 1;
				}
			}
			else if( wi->filtering ){
			  /* BoxFilter: the distinction with a PointFilter can be made
			   \ by checking apply2box.
			   */
				measure= 2;
				apply2box= True;
				done= True;
			}
		}
		if( !done ){
			if( CheckMask(mask_rtn_pressed, Button2Mask) ){
				measure= 1;
				*cur= (CursorCross)? noCursor : theCursor;
			}
			else if( CheckMask(mask_rtn_pressed, Button3Mask) ){
				point_label= 1;
				wi->add_label= 1;
			}
		}
	}

	if( measure || point_label ){
		if( wi->redraw &&
			(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
		){ int fad= wi->fit_after_draw;
			wi->fit_after_draw= False;
			RedrawNow( wi );
			wi->fit_after_draw= fad;
		}
	}

#ifdef DEBUG
	if( !debugging )
#endif
	{
#pragma mark ***HM GrabPointer
		if( XGrabPointer(disp, win, True,
				 (unsigned int) (ButtonPressMask|ButtonReleaseMask|
						 PointerMotionMask|PointerMotionHintMask|
						 Button1MotionMask|Button2MotionMask|Button3MotionMask|
						 Button4MotionMask|Button5MotionMask),
				 GrabModeAsync, GrabModeAsync,
				 win, *cur, CurrentTime
			) != GrabSuccess
		){
			Boing( 0);
			GCA();
			set_Find_Point_precision( xprec, yprec, NULL, NULL );
			return 0;
		}
		PGrabbed = True;
		if( XGrabKeyboard(disp, win, True,
				 GrabModeAsync, GrabModeAsync, CurrentTime) == GrabSuccess
		){
			KGrabbed= True;
		}
	}

	if( HandleMouseGC == (GC) 0 ){
	  unsigned long gcmask;
	  XGCValues gcvals;
	  int echoLSLen;
	  char echoLS[MAXLS];

		echoLSLen= xtb_ProcessStyle( "22", echoLS, MAXLS);
		gcmask = ux11_fill_gcvals(&gcvals, GCForeground, zeroPixel ^ bgPixel,
					  GCFunction, GXxor, GCFont, fbFont.font->fid, GCLineStyle, LineSolid,
					  UX11_END);
		HandleMouseGC = XCreateGC(disp, win, gcmask, &gcvals);
		if( echoLSLen> 0 ){
			XSetDashes(disp, HandleMouseGC, 1, echoLS, echoLSLen);
			XSetLineAttributes( disp, HandleMouseGC, 0, LineOnOffDash, CapButt, JoinRound );
		}
		else{
			XSetLineAttributes( disp, HandleMouseGC, 0, LineSolid, CapButt, JoinRound );
		}
	}
	XFetchName( disp, wi->window, &name );
	startX = curX, startY = curY;

	if( point_label || measure== 2 ){
	  /* Find the closest datapoint, and warp the pointer thereto	*/
	  double clkX, clkY;
		clikX= clkX= Reform_X( wi, TRANX(startX), TRANY(startY) );
		clikY= clkY= Reform_Y( wi, TRANY(startY), TRANX(startX) );
		Trans_XY( wi, &clikX, &clikY, 0);
#ifdef DEBUG_POINTLABEL
		fprintf( StdErr, "Point_label: scr (%d,%d) -> (%s,%s) [%s,%s,%d,%d] =>", startX, startY,
			d2str( point_l_x, "%g", NULL), d2str( point_l_y, "%g", NULL),
			d2str( clikX, "%g", NULL), d2str( clikY, "%g", NULL),
			SCREENX(wi,clikX), SCREENY(wi, clikY)
		);
#endif
		XFillArc( disp, win, HandleMouseGC, SCREENX(wi,clikX)- 3, SCREENY(wi,clikY)-3,
			6, 6, 0, 270* 64
		);
		if( measure== 2 ){
			if( !(((wi->cutAction & _deleteAction) && CheckMask(mask_rtn_pressed, ShiftMask|Button1Mask))
				 || (wi->filtering && apply2box))
			){
			  /* Not deleting, and not applying a BoxFilter: find the nearest point.	*/
				if( point_l_nr< 0 ){
					point_l_x= clkX;
					point_l_y= clkY;
					point_l_nr= Find_Point( wi, &point_l_x, &point_l_y, &point_label_set, 1, &ULabel, True, True, True, True );
				}
			}
			else{
				  /* Holding down the Shiftkey with Button1 in Delete mode selects "delete within a box" mode	*/
				apply2box= True;
			}
		}
		else{
			if( point_l_nr< 0 ){
				point_l_x= clkX;
				point_l_y= clkY;
				point_l_nr= Find_Point( wi, &point_l_x, &point_l_y, &point_label_set, 0, NULL, True, True, True, True );
			}
		}
		if( point_l_nr>= 0 ){
		  double x= point_l_x, y= point_l_y;
		  int old_silent, asp= wi->aspect, fx= wi->fit_xbounds, fy= wi->fit_ybounds, fo= wi->FitOnce, si= wi->silenced,
		  	fad= wi->fit_after_draw;

			if( !RAW_DISPLAY(wi) ){
				if( set_raw_display &&
					(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
				){
					if( CheckMask(mask_rtn_pressed, ShiftMask) && ULabel ){
						old_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
						wi->aspect= 0;
						wi->fit_xbounds= 0;
						wi->fit_ybounds= 0;
						wi->FitOnce= 0;
						wi->silenced= 1;
					}

					wi->raw_display= 1;
					wi->raw_once= 0;
					wi->fit_after_draw= False;
					RedrawNow( wi );
					wi->fit_after_draw= fad;
					if( CheckMask(mask_rtn_pressed, ShiftMask) && ULabel ){
						wi->dev_info.xg_silent( wi->dev_info.user_state, (wi->silenced)? True : old_silent );
						wi->aspect= asp;
						wi->fit_xbounds= fx;
						wi->fit_ybounds= fy;
						wi->FitOnce= fo;
						wi->silenced= si;
					}

					if( point_label_set ){
						x= point_l_x= XVAL(point_label_set, point_l_nr);
						y= point_l_y= YVAL(point_label_set, point_l_nr);
					}
					else if( ULabel ){
						if( point_l_nr== 0 ){
							x= point_l_x= ULabel->x1;
							y= point_l_y= ULabel->y1;
						}
						else{
							x= point_l_x= ULabel->x2;
							y= point_l_y= ULabel->y2;
						}
					}
				}
				else{
				  /* don't do anything..	*/
				}
			}
			else if( point_label_set ){
			  /* 20000131: these lines ensure that even in raw_display mode,
			   \ the pointer is warped to the nearest point, e.g. when displacing
			   \ a (group of) set(s). Note that this may not always be as welcome..
			   */
				x= point_l_x= XVAL(point_label_set, point_l_nr);
				y= point_l_y= YVAL(point_label_set, point_l_nr);
			}
			if( !apply2box ){
				if( debugFlag && !wi->raw_display ){
					fprintf( StdErr, "Clicked at (%d,%d), nearest to Set#%d-(%s,%s) ",
						startX, startY,
						(point_label_set)? point_label_set->set_nr : -1,
						d2str( x, NULL, NULL), d2str( y, NULL, NULL)
					);
				}
				Trans_XY( wi, &x, &y, 0);
				if( debugFlag && !wi->raw_display ){
					fprintf( StdErr, "[%s,%s] ", d2str(x, NULL, NULL), d2str(y, NULL, NULL) );
				}
				startX= SCREENX( wi, x );
				startY= SCREENY( wi, y );
				if( debugFlag && !wi->raw_display ){
					fprintf( StdErr, "at (%d,%d)\n", startX, startY );
				}
				XWarpPointer( disp, None, wi->window, 0, 0, 0, 0, startX, startY );
			}
			if( CursorCross ){
				DrawCCrosses( wi, (XEvent*) evt, point_l_x, point_l_y, startX, startY, NULL, "HandleMouse()" );
				  // 20080627: avoid re-drawing the cursor-cross after doing it here, to avoid showing different
				  // co-ordinates at the cross and in the titlebar.
				CCdrawn= True;
			}
			if( measure== 2
#ifdef RUNSPLIT
				&& !wi->cutAction && !wi->filtering
#endif
			){
				if( point_label_set ){
					if( CheckMask( mask_rtn_pressed, ShiftMask ) ){
						sx_min= point_label_set->sx_min;
						sy_min= point_label_set->sy_min;
						sx_max= point_label_set->sx_max;
						sy_max= point_label_set->sy_max;
						if( displ_set ){
							for( i= 0; i< displ_sets; i++ ){
								sx_min= MIN( sx_min, displ_set[i]->sx_min );
								sy_min= MIN( sy_min, displ_set[i]->sy_min );
								sx_max= MAX( sx_max, displ_set[i]->sx_max );
								sy_max= MAX( sy_max, displ_set[i]->sy_max );
							}
						}
						slx= sx_min;
						sly= sy_min;
						srx= sx_max;
						sry= sy_max;
						shiftEcho[0].x1= shiftEcho[3].x2= slx;
						shiftEcho[0].y1= shiftEcho[3].y2= sly;
						shiftEcho[0].x2= shiftEcho[1].x1= slx;
						shiftEcho[0].y2= shiftEcho[1].y1= sry;
						shiftEcho[1].x2= shiftEcho[2].x1= srx;
						shiftEcho[1].y2= shiftEcho[2].y1= sry;
						shiftEcho[2].x2= shiftEcho[3].x1= srx;
						shiftEcho[2].y2= shiftEcho[3].y1= sly;
						shiftEchos= 4;
						dlE= point_label_set->numPoints/ (ShiftLineEchoSegments+ 1);
						if( dlE< 1 ){
							dlE= 1;
						}
						lineEchos= 0;
						for( i= 0; i< point_label_set->numPoints && lineEchos< ShiftLineEchoSegments; i++ ){
							if( i % dlE== 0 ){
							  int x= SCREENX(wi, point_label_set->xvec[i]);
							  int y= SCREENY(wi, point_label_set->yvec[i]);
								lineEchoSeg[lineEchos].x2= lineEchoSeg[lineEchos].x1= x;
								lineEchoSeg[lineEchos].y2= lineEchoSeg[lineEchos].y1= y;
								if( lineEchos ){
									lineEchoSeg[lineEchos-1].x2= x;
									lineEchoSeg[lineEchos-1].y2= y;
								}
								Xsegs[lineEchos]= lineEchoSeg[lineEchos];
								lineEchos+= 1;
							}
						}
					}
					else{
						GET_SHIFTNEIGHBOURS( point_label_set, point_l_nr, slx, sly, srx, sry);
						Trans_XY( wi, &slx, &sly, 0);
						Trans_XY( wi, &srx, &sry, 0);
						shiftEcho[0].x1= SCREENX( wi, slx);
						shiftEcho[0].y1= SCREENY( wi, sly);
						shiftEcho[1].x1= shiftEcho[0].x2= startX;
						shiftEcho[1].y1= shiftEcho[0].y2= startY;
						shiftEcho[1].x2= SCREENX( wi, srx);
						shiftEcho[1].y2= SCREENY( wi, sry);
						shiftEchos= 2;
					}
					XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
					if( lineEchos ){
						XDrawSegments( disp, win, HandleMouseGC, Xsegs, lineEchos );
					}
				}
				else if( ULabel ){
					switch( ULabel->type ){
						case UL_regular:
						default:
							if( point_l_nr== 0 ){
								slx= (wi->raw_display)? ULabel->x2 : ULabel->tx2;
								sly= (wi->raw_display)? ULabel->y2 : ULabel->ty2;
								srx= slx;
								sry= sly;
							}
							else{
								slx= (wi->raw_display)? ULabel->x1 : ULabel->tx1;
								sly= (wi->raw_display)? ULabel->y1 : ULabel->ty1;
								srx= (wi->raw_display)? ULabel->x2 : ULabel->tx2;
								sry= (wi->raw_display)? ULabel->y2 : ULabel->ty2;
							}
							shiftEchos= 1;
							break;
						case UL_hline:
							slx= wi->loX;
							srx= wi->hiX;
							sly= sry= (wi->raw_display)? ULabel->y1 : ULabel->ty1;
							shiftEchos= 2;
							break;
						case UL_vline:
							slx= srx= (wi->raw_display)? ULabel->x1 : ULabel->tx1;
							sly= wi->loY;
							sry= wi->hiY;
							shiftEchos= 2;
							break;
					}
					if( point_l_nr== 0 || (slx!=srx || sly!= sry) ){
						Trans_XY( wi, &slx, &sly, 0);
						if( shiftEchos== 1 ){
							shiftEcho[0].x1= SCREENX( wi, slx);
							shiftEcho[0].y1= SCREENY( wi, sly);
							shiftEcho[0].x2= startX;
							shiftEcho[0].y2= startY;
						}
						else{
							Trans_XY( wi, &srx, &sry, 0);
							shiftEcho[0].x1= SCREENX( wi, slx);
							shiftEcho[0].y1= SCREENY( wi, sly);
							shiftEcho[1].x1= shiftEcho[0].x2= startX;
							shiftEcho[1].y1= shiftEcho[0].y2= startY;
							shiftEcho[1].x2= SCREENX( wi, srx);
							shiftEcho[1].y2= SCREENY( wi, sry);
						}
						XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
					}
					if( point_l_nr== 1 ){
					  XPoint *xp= (XPoint*) ULabel->box;
						XDrawLines( disp, win, HandleMouseGC, xp, 10, CoordModeOrigin );
					}
				}
			}
		}
		else if( !apply2box ){
			if( PGrabbed ){
				XUngrabPointer(disp, CurrentTime);
			}
			if( KGrabbed ){
				XUngrabKeyboard( disp, CurrentTime );
			}
			XG_XSync( disp, False );
			xtb_error_box( wi->window, "No visible point close to your entry", "Warning" );
			GCA();
			set_Find_Point_precision( xprec, yprec, NULL, NULL );
			return( 0 );
		}
	}
	_loX= Reform_X( wi, TRANX(startX), TRANY(startY) );
	_loY= Reform_Y( wi, TRANY(startY), TRANX(startX) );
	// 20120508: redundant?
#if 0
	XQueryPointer(disp, win, &root_rtn, &child_rtn, &root_x, &root_y,
		  &curX, &curY, &mask_rtn_pressed
	);
	mask_rtn_pressed= xtb_Mod2toMod1( mask_rtn_pressed );
	capslock= CheckMask( mask_rtn_pressed, LockMask);
	mask_rtn_pressed&= ~LockMask;
#endif
#pragma mark ***HM Pre interactive loop mask check
	if( CheckMask(mask_rtn_pressed, ControlMask) || CheckMask(mask_rtn_pressed,Mod1Mask) ){
	  Boolean done= False;
		if( CheckMask(mask_rtn_pressed,Mod1Mask) ){
			if( CheckMask(mask_rtn_pressed, Button1Mask)){
				if( !wi->filtering && !wi->raw_display
					&& (wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
				){
					  /* 20000407: (maybe) do a raw redraw. Store the previous state of the raw_display flag,
					   \ and record the fact that is a once-only,to-be-restored event. Probably should be
					   \ done at the other raw redraw places in this routine also.
					   \ 20050521: don't when wi->filtering...
					   */
					wi->raw_val= wi->raw_display;
					wi->raw_once= -1;
					wi->raw_display= True;
					wi->redraw= 1;
				}
				measure= 2;
				done= True;
			}
			if( wi->cutAction & _deleteAction ){
				if( CheckMask(mask_rtn_pressed, ShiftMask|Button1Mask) ){
					  /* Holding down the Shiftkey with Button1 in Delete mode selects "delete within a box" mode	*/
					apply2box= True;
				}
				if( CheckMask(mask_rtn_pressed, Button2Mask) ){
					measure= 2;
					del_upto= True;
					done= True;
				}
				else if( CheckMask(mask_rtn_pressed, Button3Mask) ){
					measure= 2;
					del_from= True;
					done= True;
				}
			}
			if( CheckMask(mask_rtn_pressed, Button3Mask) ){
				done= True;
				point_label= 1;
				if( wi->filtering ){
					apply2box= False;
					measure= 2;
				}
				if( CheckMask( mask_rtn_pressed, ControlMask) ){
					wi->add_label= 1;
				}
			}
			else if( wi->filtering ){
				done= True;
				measure= 2;
				apply2box= True;
			}
		}
		if( !done ){
			if( CheckMask(mask_rtn_pressed, Button2Mask)){
				measure= 1;
			}
			else if( CheckMask(mask_rtn_pressed, Button3Mask) ){
				point_label= 1;
				wi->add_label= 1;
#ifdef DEBUG_POINTLABEL
				fprintf( StdErr, "(%s,%s), scr (%d,%d) -> (%s,%s); current (%d,%d)\n",
					d2str( point_l_x, "%g", NULL), d2str( point_l_y, "%g", NULL),
					startX, startY,
					d2str( _loX, "%g", NULL), d2str( _loY, "%g", NULL),
					curX, curY
				);
#endif
			}
			else{
				if( curY>= wi->title_uly && curY<= wi->title_lry && wi->graph_titles && !wi->no_title ){
					if( PGrabbed ){
						XUngrabPointer(disp, CurrentTime);
					}
					if( KGrabbed ){
						XUngrabKeyboard( disp, CurrentTime );
					}
					XG_XSync( disp, False );
					xtb_error_box( wi->window, wi->graph_titles, "Current Titles:");
					GCA();
					set_Find_Point_precision( xprec, yprec, NULL, NULL );
					return( 0 );
				}
				else if( curX>= wi->legend_ulx && curX<= wi->legend_frx &&
					curY>= wi->legend_uly- wi->dev_info.bdr_pad && curY<= wi->legend_lry &&
					!wi->no_legend
				){
					if( !wi->no_legend){
					  int this_one= -1;
						if( PGrabbed ){
							XUngrabPointer(disp, CurrentTime);
						}
						if( KGrabbed ){
							XUngrabKeyboard( disp, CurrentTime );
						}
						XG_XSync( disp, False );
						if( CheckMask(mask_rtn_pressed, ShiftMask) ){
							for( i= 0; i< setNumber; i++ ){
								if( draw_set(wi,i) && ((curY>= wi->legend_line[i].low_y && curY<= wi->legend_line[i].high_y) ||
									(curY<= wi->legend_line[i].low_y && curY>= wi->legend_line[i].high_y))
								){
									this_one= i;
								}
							}
						}
						ShowLegends( wi, True, this_one );
						GCA();
						set_Find_Point_precision( xprec, yprec, NULL, NULL );
						return( 0 );
					}
				}
				else if( wi->ulabel ){
				  UserLabel *ul= wi->ulabel;
				  int n= 1;
				  Boolean all_marked, all_hlt, none_marked, none_hlt;
					check_marked_hlt( wi, &all_marked, &all_hlt, &none_marked, &none_hlt );
					while( ul ){
						if( ul->set_link>= 0 && AllSets[ul->set_link].numPoints<= 0 ){
							ul->label[0]= '\0';
						}
						if( strlen( ul->label) &&
							( ul->set_link== -1 ||
								(ul->set_link== -2 && all_marked) ||
								(ul->set_link== -3 && all_hlt) ||
								(ul->set_link== -4 && none_marked) ||
								(ul->set_link== -5 && none_hlt) ||
								(ul->set_link>= 0 && draw_set(wi, ul->set_link) )
							) &&
							ul->draw_it &&
							curX>= ul->box[0].x1 && curX<= ul->box[2].x1 &&
							curY>= ul->box[0].y1 && curY<= ul->box[2].y1
						){
						  char buf[MAXBUFSIZE*2];
						  char tit[256], control_a[2]= { 0x01, 0};

							if( PGrabbed ){
								XUngrabPointer(disp, CurrentTime);
							}
							if( KGrabbed ){
								XUngrabKeyboard( disp, CurrentTime );
							}
							XG_XSync( disp, False );

							sprintf( tit, "%sULabel %d of %d = %d", control_a, n, wi->ulabels, setNumber+ n- 1 );

							sprintf( buf, " \"%s\"\n", ul->label );
							if( ul->set_link>= 0 ){
								sprintf( buf, "%sLinked to #%d\n", buf, ul->set_link );
							}
							else if( ul->set_link== -2 ){
								sprintf( buf, "%sLinked to ALL marked sets\n", buf );
							}
							else if( ul->set_link== -3 ){
								sprintf( buf, "%sLinked to ALL highlighted sets\n", buf );
							}
							else if( ul->set_link== -4 ){
								sprintf( buf, "%svisible when NONE of the marked sets are drawn, or all sets\n", buf );
							}
							else if( ul->set_link== -5 ){
								sprintf( buf, "%svisible when NONE of the highlighted sets are drawn, or all sets\n", buf );
							}
							sprintf( buf, "%sraw: (%s,%s)", buf, d2str( ul->x2, NULL, NULL), d2str( ul->y2, NULL, NULL) );
							if( ul->tx1!= ul->tx2 || ul->ty1!= ul->ty2 ){
								sprintf( buf, "%s->(%s,%s)\n", buf, d2str( ul->x1, NULL, NULL), d2str( ul->y1, NULL, NULL) );
							}
							if( !wi->raw_display && ul->do_transform &&
								(ul->x1!= ul->tx1 || ul->x2!= ul->tx2 || ul->y1!= ul->ty1 || ul->y2!= ul->ty2)
							){
								sprintf( buf, "%strans: (%s,%s)", buf, d2str( ul->tx2, NULL, NULL), d2str( ul->ty2, NULL, NULL) );
								if( ul->tx1!= ul->tx2 || ul->ty1!= ul->ty2 ){
									sprintf( buf, "%s->(%s,%s)\n", buf, d2str( ul->tx1, NULL, NULL), d2str( ul->ty1, NULL, NULL) );
								}
							}
							StringCheck( buf, MAXBUFSIZE*2, __FILE__, __LINE__ );
							data_sn_number= setNumber+ n- 1;
	/* set_sn_nr will cause the SettingsDialog's settings to be reset to those of
	 \ the first drawn/shown set. This can have unwanted side-effects (it is not always
	 \ correctly reset [screen-dependent?] => potentially changing the not-the-intended
	 \ settings.
							set_sn_nr= True;
	 */
							if( wi->SD_Dialog ){
								set_changeables(2,False);
							}
							xtb_error_box( wi->window, buf, tit );
							GCA();
							set_Find_Point_precision( xprec, yprec, NULL, NULL );
							return( 0 );
						}
						ul= ul->next;
						n+= 1;
					}
				}
			}
		}
	}
	else if( !CheckMask(mask_rtn_pressed, ShiftMask) ){
		if( curX>= wi->legend_ulx && curX<= wi->legend_frx &&
			curY>= wi->legend_uly- wi->dev_info.bdr_pad && curY<= wi->legend_lry
		){
		  int i;
			if( !wi->no_legend){
			  LocalWindows *WL= NULL;
			  LocalWin *lwi;
			  int do_redraw= 0;
				if( PGrabbed ){
					XUngrabPointer(disp, CurrentTime);
				}
				if( KGrabbed ){
					XUngrabKeyboard( disp, CurrentTime );
				}
				XG_XSync( disp, False );
				  /* code to determine set to be highlighted	*/

				if( debugFlag ){
					fprintf( StdErr, "Pointer curY==%d within", curY);
				}
				for( i= 0; i< setNumber; i++ ){
					if( draw_set(wi,i) && ((curY>= wi->legend_line[i].low_y && curY<= wi->legend_line[i].high_y) ||
						(curY<= wi->legend_line[i].low_y && curY>= wi->legend_line[i].high_y))
					){
						if( debugFlag ){
							fprintf( StdErr, " >legend_line.low_y,high_y==%d,%d of Set #%d<",
								wi->legend_line[i].low_y, wi->legend_line[i].high_y, i
							);
						}
						do_redraw+= 1;
						if( handle_event_times== 0 ){
							WL= WindowList;
							lwi= WL->wi;
						}
						else{
							lwi= wi;
						}
						do{
							if( (lwi->legend_line[i].highlight= !lwi->legend_line[i].highlight) ){
							  /* adding a highlight does not require clearing the window first	*/
								if( AllSets[i].s_bounds_set ){
									AllSets[i].s_bounds_set= -1;
								}
								RedrawSet( i, False );
							}
							if( WL ){
								WL= WL->next;
								lwi= (WL)? WL->wi : 0;
							}
						} while( WL && lwi );
					}
					else if( debugFlag && draw_set(wi,i) && !do_redraw ){
						fprintf( StdErr, " [legend_line.low_y,high_y==%d,%d of Set #%d]",
							wi->legend_line[i].low_y, wi->legend_line[i].high_y, i
						);
					}
				}
				if( debugFlag ){
					fprintf( StdErr, "%s\n", (do_redraw)? " OK" : " [nothing]" );
				}
				if( handle_event_times== 0 ){
					WL= WindowList;
					lwi= WL->wi;
				}
				else{
					lwi= wi;
				}
				do{
					if( do_redraw ){
					  /* adding a highlight does not require clearing the window first	*/
						lwi->dont_clear= 1;
					}
					RedrawNow( lwi );
					if( WL ){
						WL= WL->next;
						lwi= (WL)? WL->wi : 0;
					}
				} while( WL && lwi );
				GCA();
				set_Find_Point_precision( xprec, yprec, NULL, NULL );
				return( 0 );
			}
		}
	}

	if( curX!= startX || curY!= startY ){
		_hiX= Reform_X( wi, TRANX(curX), TRANY(curY) );
		_hiY= Reform_Y( wi, TRANY(curY), TRANX(curX) );
		sprintf( box_coords, "(%.4g,%.4g) - (%.4g,%.4g)", _loX, _loY, _hiX, _hiY );
	}
	else{
		_hiX= _loX;
		_hiY= _loY;
		sprintf( box_coords, "(%.4g,%.4g)", _loX, _loY);
	}
	StringCheck( box_coords, sizeof(box_coords)/sizeof(char), __FILE__, __LINE__ );
	if( !(point_label || measure== 2) || apply2box ){
		XStoreName( disp, wi->window, box_coords );
	}

	prev_diag= 0.0;
	if( measure==1 && wi->logXFlag<= 0 && wi->logYFlag<= 0 && wi->sqrtXFlag<= 0 && wi->sqrtYFlag<= 0 ){
	  double dx= (_hiX - _loX), dy= (_hiY - _loY);
		  /* the radii of an ellipse containing all points at a certain distance.
		   \ only valid without transformations. Also wi->Xscale!= wi->Yscale is a problem:
		   \ when Xscale=1,Yscale=2 radx,rady=(400,400) should be (200,400) under 90deg, and (400,800) under 0 or 180deg;
		   \ when Xscale=2,Yscale=1 radx,rady=(200,200) should be (400,200) under 90deg, and (200,100) under 0 or 180deg;
		   \ This still requires some thinking....
		   */
		rady= sqrt( dx* dx+ dy* dy);
		radx= rady/ wi->XUnitsPerPixel;
		rady= rady/ wi->YUnitsPerPixel;
		if( radx> wi->dev_info.area_w || rady> wi->dev_info.area_h ){
			radx= 0;
			rady= 0;
		}
		diag= radx* radx + rady* rady;
	}
	else{
		radx= rady= 0;
		diag= 0;
	}
      /* Draw first box or line */
	  /* The argument to DRAWECHO controls whether the cursorcross is drawn. This is
	   \ currently done by calling DrawCCross() with the erase flag set. This could be
	   \ polished: there is no need to undraw the cross when the window gets a redraw
	   \ before DRAWECHO is called (in which case the undraw becomes an unundraw...).
	   */
	DRAWECHO( !CCdrawn );
	stopFlag = 0;
	if( measure && measure!= 2 ){
		sprintf( box_coords, "(%s,%s)",
			d2str( _loX, "%.4g", NULL),
			d2str( _loY, "%.4g", NULL)
		);
		StringCheck( box_coords, sizeof(box_coords)/sizeof(char), __FILE__, __LINE__ );
		XDrawString( disp, wi->window, HandleMouseGC,
			startX+ 2, startY+ 2, box_coords, strlen(box_coords)
		);
	}
	while( !stopFlag ){
#pragma mark ***HM interactive loop
/* 		XAllowEvents( disp, AsyncKeyboard, CurrentTime );	*/
		XNextEvent(disp, &theEvent);
		switch( theEvent.xany.type ){
			case MotionNotify:{
#pragma mark ***HMil MotionNotify
			  Boolean redraw_box;
				  /* 990326: checks for apply2box and wi->cutAction. This to
				   \ have a rubber box visible when deleting points within a screensection,
				   \ and to preserve Find_Point's message when deleting (from) a specific point
				   */
				// 20120508: XQueryPointer appears to be required here. If I take the information from
				// theEvent, I only get a single MotionNotify, which is obviously not what we'd want.
				XQueryPointer(disp, win, &root_rtn, &child_rtn, &root_x, &root_y,
						    &newX, &newY, &mask_rtn_pressed
				);
// 				mask_rtn_pressed= xtb_Mod2toMod1( mask_rtn_pressed );
// 				capslock= CheckMask( mask_rtn_pressed, LockMask);
// 				mask_rtn_pressed&= ~LockMask;
				if( debugFlag && debugLevel<=-1 ){
					fprintf( StdErr, "MotionNotify event: %s == 0x%lx (%s)",
						event_name(theEvent.xany.type),
						mask_rtn_pressed, xtb_modifiers_string(mask_rtn_pressed)
					);
					fprintf( StdErr, " state=0x%lx (%s) at [%d,%d]",
						theEvent.xmotion.state, xtb_modifiers_string(theEvent.xmotion.state),
						newX, newY
					);
					fprintf( StdErr, " /[%d,%d] root=[%d,%d]/[%d,%d]\n",
						   theEvent.xmotion.x, theEvent.xmotion.y,
						   root_x, root_y, theEvent.xmotion.x_root, theEvent.xmotion.y_root
					);
				}
				newX = theEvent.xmotion.x, newY = theEvent.xmotion.y;
				mask_rtn_pressed= xtb_Mod2toMod1( theEvent.xmotion.state );
				capslock= CheckMask( mask_rtn_pressed, LockMask);
				mask_rtn_pressed&= ~LockMask;
//					if( !debugFlag || debugLevel>-1 ){
//						fprintf( StdErr, "MotionNotify event: %s == 0x%lx (%s)",
//							event_name(theEvent.xany.type),
//							mask_rtn_pressed, xtb_modifiers_string(mask_rtn_pressed)
//						);
//						fprintf( StdErr, " state=0x%lx (%s) at [%d,%d]",
//							theEvent.xmotion.state, xtb_modifiers_string(theEvent.xmotion.state),
//							newX, newY
//						);
//						fprintf( StdErr, " root=[%d,%d]\n",
//							   theEvent.xmotion.x_root, theEvent.xmotion.y_root
//						);
//					}
				
				if( displacing && CheckMask(mask_rtn_pressed, ControlMask|Button1Mask|Mod1Mask) ){
				  int dx= abs( newX- startX), dy= abs( newY- startY);
					if( dx> dy ){
						newY= startY;
					}
					else if( dy> dx ){
						newX= startX;
					}
				}

				if( !CheckMask( mask_rtn_pressed, ShiftMask ) ||
					(CheckMask( mask_rtn_pressed, ShiftMask) && CheckMask( mask_rtn_pressed, Mod1Mask) &&
						CheckMask( mask_rtn_pressed, Button1Mask)
					) ||
					(CheckMask( mask_rtn_pressed, ShiftMask) && CheckMask( mask_rtn_pressed, ControlMask) &&
						CheckMask( mask_rtn_pressed, Button3Mask)
					) ||
					apply2box
				){
					  /* Undraw the old box */
					DRAWECHO(0);
					  /* Draw the new one */
					prev_diag= diag;
					dX= newX - curX;
					dY= newY - curY;
					curX = newX;  curY = newY;
					if( measure==1 && wi->logXFlag<= 0 && wi->logYFlag<= 0 && wi->sqrtXFlag<= 0 && wi->sqrtYFlag<= 0 ){
					  double dx, dy;
						_hiX= Reform_X( wi, TRANX(curX), TRANY(curY) );
						_hiY= Reform_Y( wi, TRANY(curY), TRANX(curX) );
						dx= (_hiX - _loX); dy= (_hiY - _loY);
						rady= sqrt( dx* dx+ dy* dy);
						radx= rady/ wi->XUnitsPerPixel;
						rady= rady/ wi->YUnitsPerPixel;
						if( radx> wi->dev_info.area_w || rady> wi->dev_info.area_h ){
							radx= 0;
							rady= 0;
							diag= 0;
						}
						else{
							diag= radx* radx + rady* rady;
						}
					}
					else{
						radx= rady= 0;
						diag= 0;
					}
					redraw_box= True;
				}
				else{
					  /* Shift-Mod1-Button3 (highlight line) warps pointer to nearest point on every
					   \ mouse movement.
					   */
					if( point_label &&
						CheckMask(mask_rtn_pressed, Mod1Mask) && !CheckMask(mask_rtn_pressed, ControlMask) &&
						(abs(curX-newX)> 0 || abs(curY-newY)> 0)
					){
					  double _point_l_x, _point_l_y;
					  DataSet *_point_label_set;
					  int _point_l_nr;
					  /* Find the closest datapoint, and warp the pointer thereto	*/
						_point_l_x= Reform_X( wi, TRANX(newX), TRANY(newY) );
						_point_l_y= Reform_Y( wi, TRANY(newY), TRANX(newX) );
						if( (_point_l_nr= Find_Point( wi, &_point_l_x, &_point_l_y, &_point_label_set, 0, NULL, True, True, True, True ))>= 0 ){
						  double x= point_l_x, y= point_l_y;
							Trans_XY( wi, &x, &y, 0);
							newX= SCREENX( wi, x );
							newY= SCREENY( wi, y );
							XWarpPointer( disp, None, wi->window, 0, 0, 0, 0, newX, newY );
							if( CursorCross ){
								DrawCCrosses( wi, (XEvent*) evt, point_l_x, point_l_y, newX, newY, NULL, "HandleMouse()" );
								  // 20080627: avoid re-drawing the cursor-cross after doing it here, to avoid showing different
								  // co-ordinates at the cross and in the titlebar.
								CCdrawn= True;
							}
							  /* Found a new point. Update the record.	*/
							point_l_x= _point_l_x;
							point_l_y= _point_l_y;
							point_label_set= _point_label_set;
							point_l_nr= _point_l_nr;
						}
					}
					curX = newX;  curY = newY;
					redraw_box= False;
				}
				if( curX!= startX || curY!= startY ){
					_hiX= Reform_X( wi, TRANX(curX), TRANY(curY) );
					_hiY= Reform_Y( wi, TRANY(curY), TRANX(curX) );
					if( measure && !wi->cutAction && !(wi->filtering && apply2box) ){
/* 					  double dx= _hiX - _loX, dy= _hiY - _loY;	*/
					  double dx= (curX!=startX)? _hiX - point_l_x: 0,
					  		dy= (curY!=startY)? _hiY - point_l_y : 0;
						sprintf( box_coords, "(%s,%s): d=%s[%s,%s] %sx,%sx %sdeg",
							d2str( _hiX, "%.4g", NULL),
							d2str( _hiY, "%.4g", NULL),
							d2str( sqrt( dx*dx + dy*dy ), "%.4g", NULL),
							d2str( dx, "%.4g", NULL),
							d2str( dy, "%.4g", NULL),
/* 							d2str( _hiX / _loX, "%.4g", NULL), d2str( _hiY / _loY, "%.4g", NULL),	*/
							d2str( _hiX / point_l_x, "%.4g", NULL), d2str( _hiY / point_l_y, "%.4g", NULL),
							d2str( degrees(atan3( dx, dy)), "%.4g", NULL)
						);
						StringCheck( box_coords, sizeof(box_coords)/sizeof(char), __FILE__, __LINE__ );
						if( measure== 2 && point_l_nr>= 0
#ifdef RUNSPLIT
							&& !wi->cutAction && !wi->filtering
#endif
						){
							if( point_label_set ){
							  int dx, dy;
								XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
								if( lineEchos ){
									XDrawSegments(disp, win, HandleMouseGC, Xsegs, lineEchos);
								}
								if( CheckMask( mask_rtn_pressed, ShiftMask) ){
									dx= curX- startX;
									dy= curY- startY;
									slx= dx+ sx_min;
									sly= dy+ sy_min;
									srx= dx+ sx_max;
									sry= dy+ sy_max;
									shiftEcho[0].x1= shiftEcho[3].x2= slx;
									shiftEcho[0].y1= shiftEcho[3].y2= sly;
									shiftEcho[0].x2= shiftEcho[1].x1= slx;
									shiftEcho[0].y2= shiftEcho[1].y1= sry;
									shiftEcho[1].x2= shiftEcho[2].x1= srx;
									shiftEcho[1].y2= shiftEcho[2].y1= sry;
									shiftEcho[2].x2= shiftEcho[3].x1= srx;
									shiftEcho[2].y2= shiftEcho[3].y1= sly;
									shiftEchos= 4;
									for( i= 0; i< lineEchos; i++ ){
										Xsegs[i].x1= lineEchoSeg[i].x1+ dx;
										Xsegs[i].y1= lineEchoSeg[i].y1+ dy;
										Xsegs[i].x2= lineEchoSeg[i].x2+ dx;
										Xsegs[i].y2= lineEchoSeg[i].y2+ dy;
									}
								}
								else{
									GET_SHIFTNEIGHBOURS( point_label_set, point_l_nr, slx, sly, srx, sry);
									Trans_XY( wi, &slx, &sly, 0);
									Trans_XY( wi, &srx, &sry, 0);
									shiftEcho[0].x1= SCREENX( wi, slx);
									shiftEcho[0].y1= SCREENY( wi, sly);
									shiftEcho[1].x1= shiftEcho[0].x2= curX;
									shiftEcho[1].y1= shiftEcho[0].y2= curY;
									shiftEcho[1].x2= SCREENX( wi, srx);
									shiftEcho[1].y2= SCREENY( wi, sry);
									shiftEchos= 2;
								}
								XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
								if( lineEchos ){
									XDrawSegments(disp, win, HandleMouseGC, Xsegs, lineEchos);
								}
							}
							else if( ULabel ){
								XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
								switch( ULabel->type ){
									case UL_regular:
									default:
										if( point_l_nr== 0 ){
											slx= (wi->raw_display)? ULabel->x2 : ULabel->tx2;
											sly= (wi->raw_display)? ULabel->y2 : ULabel->ty2;
											srx= slx;
											sry= sly;
										}
										else{
											slx= (wi->raw_display)? ULabel->x1 : ULabel->tx1;
											sly= (wi->raw_display)? ULabel->y1 : ULabel->ty1;
											srx= (wi->raw_display)? ULabel->x2 : ULabel->tx2;
											sry= (wi->raw_display)? ULabel->y2 : ULabel->ty2;
										}
										shiftEchos= 1;
										break;
									case UL_hline:
										slx= wi->loX;
										srx= wi->hiX;
										sly= sry= (wi->raw_display)? ULabel->y1 : ULabel->ty1;
										shiftEchos= 2;
										break;
									case UL_vline:
										slx= srx= (wi->raw_display)? ULabel->x1 : ULabel->tx1;
										sly= wi->loY;
										sry= wi->hiY;
										shiftEchos= 2;
										break;
								}
								if( point_l_nr== 0 || (slx!=srx || sly!= sry) ){
									Trans_XY( wi, &slx, &sly, 0);
									if( shiftEchos== 1 ){
										shiftEcho[0].x1= SCREENX( wi, slx);
										shiftEcho[0].y1= SCREENY( wi, sly);
										shiftEcho[0].x2= curX;
										shiftEcho[0].y2= curY;
									}
									else{
										Trans_XY( wi, &srx, &sry, 0);
										shiftEcho[0].x1= SCREENX( wi, slx);
										shiftEcho[0].y1= SCREENY( wi, sly);
										shiftEcho[1].x1= shiftEcho[0].x2= curX;
										shiftEcho[1].y1= shiftEcho[0].y2= curY;
										shiftEcho[1].x2= SCREENX( wi, srx);
										shiftEcho[1].y2= SCREENY( wi, sry);
									}
									XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
								}
								if( point_l_nr== 1 ){
								  int i;
								  XPoint *xp= (XPoint*) ULabel->box;
									XDrawLines( disp, win, HandleMouseGC, xp, 10, CoordModeOrigin );
									for( i= 0; i< 10; i++ ){
										xp[i].x+= dX;
										xp[i].y+= dY;
									}
									XDrawLines( disp, win, HandleMouseGC, xp, 10, CoordModeOrigin );
								}
							}
						}
					}
					else if( !wi->cutAction || apply2box ){
						sprintf( box_coords, "(%.4g,%.4g) - (%.4g,%.4g)", _loX, _loY, _hiX, _hiY );
						StringCheck( box_coords, sizeof(box_coords)/sizeof(char), __FILE__, __LINE__ );
					}
					  /* else Find_Point() updates the window's titlebar.	*/
					if( !point_label && (!wi->cutAction || apply2box) ){
						XStoreName( disp, wi->window, box_coords );
					}
					XG_XSync( disp, False );
				}
				else if( !point_label && !wi->cutAction ){
					sprintf( box_coords, "(%.4g,%.4g)", _loX, _loY);
					StringCheck( box_coords, sizeof(box_coords)/sizeof(char), __FILE__, __LINE__ );
					XStoreName( disp, wi->window, box_coords );
					XG_XSync( disp, False );
				}
				if( redraw_box ){
					DRAWECHO( !CCdrawn );
				}
				break;
			}
			case ButtonRelease:{
#pragma mark ***HMil ButtonRelease
			  int bstate=theEvent.xbutton.state, bbutton= theEvent.xbutton.button;
				  /* 20050114: ignore button events associated with mouse scroll wheel handling: */
				if( bbutton== 4 || bbutton== 5 ){
					break;
				}
				if( !CheckMask(mask_rtn_pressed, ShiftMask) ){
					DRAWECHO(0);
				}
				if( debugFlag && debugLevel==-2 ){
					XQueryPointer(disp, win, &root_rtn, &child_rtn, &root_x, &root_y,
						  &newX, &newY, &mask_rtn_released
					);
					fprintf( StdErr, "ButtonRelease event: %s == 0x%lx (%s)",
						event_name(theEvent.xany.type),
						mask_rtn_released, xtb_modifiers_string(mask_rtn_released)
					);
					fprintf( StdErr, " state=0x%lx (%s) button=%d (%s)",
						bstate, xtb_modifiers_string(bstate),
						bbutton, (bbutton>=0 && bbutton< sizeof(bnames)/sizeof(char*))? bnames[bbutton] : "?!"
					);
					fprintf( StdErr, " cur=[%d,%d]/[%d,%d] root=[%d,%d]/[%d,%d]\n",
						   newX, newY, theEvent.xbutton.x, theEvent.xbutton.y,
						   root_x, root_y, theEvent.xbutton.x_root, theEvent.xbutton.y_root
					);
				}
				newX = theEvent.xbutton.x, newY = theEvent.xbutton.y;
				mask_rtn_released= xtb_Mod2toMod1( theEvent.xbutton.state );
				capslock= CheckMask( mask_rtn_released, LockMask);
				mask_rtn_released&= ~LockMask;
				mask_rtn_released |= buttonMask[theEvent.xbutton.button];

				if( displacing && CheckMask(mask_rtn_pressed, ControlMask|Button1Mask|Mod1Mask) ){
				  int dx= abs( newX- startX), dy= abs( newY- startY);
					if( dx> dy ){
						newY= startY;
					}
					else if( dy> dx ){
						newX= startX;
					}
				}

				if( PGrabbed ){
					XUngrabPointer(disp, CurrentTime);
				}
				if( KGrabbed ){
					XUngrabKeyboard( disp, CurrentTime );
				}
#ifdef DEBUG
				if( !RemoteConnection ){
					XFlush( disp );
				}
#endif
				if( point_label || measure ){
					if( measure== 2 ){
#ifdef RUNSPLIT
						if( !wi->cutAction )
#endif
						{
							if( point_label_set ){
								XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
								if( lineEchos ){
									XDrawSegments(disp, win, HandleMouseGC, Xsegs, lineEchos);
								}
							}
							else if( ULabel ){
								XDrawSegments(disp, win, HandleMouseGC, shiftEcho, shiftEchos);
							}
						}
					}
					else{
						XFillArc( disp, win, HandleMouseGC, SCREENX(wi,clikX)- 3, SCREENY(wi,clikY)-3,
							6, 6, 0, 270* 64
						);
					}
				}
				XG_XSync( disp, False );
				if( measure==1 && CheckMask(mask_rtn_released, ControlMask) ){
				  char result[256];
					sprintf( result, "(%s,%s) - %s",
						d2str( _loX, "%.4g", NULL),
						d2str( _loY, "%.4g", NULL),
						box_coords
					);
					StringCheck( result, sizeof(result)/sizeof(char), __FILE__, __LINE__ );
					sprintf( box_coords, "(%s,%s)",
						d2str( _loX, "%.4g", NULL),
						d2str( _loY, "%.4g", NULL)
					);
					StringCheck( box_coords, sizeof(box_coords)/sizeof(char), __FILE__, __LINE__ );
					XDrawString( disp, wi->window, HandleMouseGC,
						startX+ 2, startY+ 2, box_coords, strlen(box_coords)
					);
					xtb_error_box( wi->window, result, "Result" );
				}
				if( measure== 2  && (point_l_nr>= 0 || apply2box) ){
					if( !CheckMask(mask_rtn_released, Mod1Mask) ){
						handled= True;
						goto delete_aborted;
					}
#ifdef RUNSPLIT
					if( (wi->cutAction & _spliceAction) && point_label_set ){
						if( !point_label_set->splithere ){
						  /* We just allocate as much space as needed here. If ever points are added to this set
						   \ (by loading additional files), realloc_points will take care of increasing the arena.
						   */
							point_label_set->splithere= (signed char*) calloc( point_label_set->allocSize, sizeof(signed char));
							point_label_set->mem_alloced+= point_label_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
							mem_alloced+= point_label_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
						}
						if( point_label_set->splithere ){
						  /* if we got the memory, swap the split-indicator at this point.	*/
							SplitUndo.idx= point_l_nr;
							if( (SplitUndo.set= point_label_set) ){
								SplitUndo.split= SplitHere(point_label_set, point_l_nr);
							}
							point_label_set->splithere[point_l_nr]= ! point_label_set->splithere[point_l_nr];
						}
						ShiftUndo.set= NULL;
						ShiftUndo.ul= NULL;
						LabelClipUndo= NULL;
						DiscardUndo.set= NULL;
						DiscardUndo.ul.label[0]= '\0';
						wi->PlaceUndo.valid= False;
						if( BoxFilterUndo.fp ){
							fclose( BoxFilterUndo.fp );
							BoxFilterUndo.fp= NULL;
						}
						wi->cutAction&= ~_spliceAction;
						*cur= (CursorCross)? noCursor : theCursor;
						RedrawSet( point_label_set->set_nr, 1 );
						handled= True;
					}
					else
#endif
					if( (wi->cutAction & _deleteAction) || wi->filtering ){
					  int i, I;
					  Boolean filt= False;
						if( apply2box ){
							if( _loX > _hiX ){
								double temp;

								temp = _hiX;
								_hiX = _loX;
								_loX = temp;
							}
							if( _loY > _hiY ){
								double temp;

								temp = _hiY;
								_hiY = _loY;
								_loY = temp;
							}
							if( wi->filtering ){
							  /* BoxFilter:	*/
								if( FilterPoints_Box( wi, NULL, _loX, _loY, _hiX, _hiY, NULL, -1 ) ){
									ShiftUndo.set= NULL;
									ShiftUndo.ul= NULL;
									LabelClipUndo= NULL;
									SplitUndo.set= NULL;
									DiscardUndo.set= NULL;
									DiscardUndo.ul.label[0]= 0;
									wi->PlaceUndo.valid= 0;
									filt= True;
								}
								wi->filtering= 0;
								*cur= (CursorCross)? noCursor : theCursor;
							}
							else{
								if( debugFlag ){
									sprintf( box_coords, "Del in (%s,%s) - (%s,%s)",
										d2str( _loX, "%.4g", NULL), d2str( _loY, "%.4g", NULL),
										d2str( _hiX, "%.4g", NULL), d2str( _hiY, "%.4g", NULL)
									);
									StringCheck( box_coords, sizeof(box_coords)/sizeof(char), __FILE__, __LINE__ );
									xtb_error_box( wi->window, box_coords, "Warning:" );
								}
								if( DiscardPoints_Box( wi, _loX, _loY, _hiX, _hiY, 1 ) ){
									DiscardUndo.set= (DataSet*) -1;
									DiscardUndo.idx= -1;
									DiscardUndo.split= 0;
									DiscardUndo.wi= wi;
									DiscardUndo.lX= _loX;
									DiscardUndo.lY= _loY;
									DiscardUndo.hX= _hiX;
									DiscardUndo.hY= _hiY;
									wi->raw_display= !raw_display;
								}
							}
							handled= True;
						}
						else if( wi->filtering ){
						  /* PointFilter...: the bounding "rectangle" will be set
						   \ by FilterPoints_Box() itself.
						   */
							if( FilterPoints_Box( wi, NULL, 0, 0, 0, 0, point_label_set, point_l_nr ) ){
								ShiftUndo.set= NULL;
								ShiftUndo.ul= NULL;
								LabelClipUndo= NULL;
								SplitUndo.set= NULL;
								DiscardUndo.set= NULL;
								DiscardUndo.ul.label[0]= 0;
								wi->PlaceUndo.valid= 0;
								filt= True;
							}
							wi->filtering= 0;
							point_label= 0;
							*cur= (CursorCross)? noCursor : theCursor;
							handled= True;
						}
						else if( point_label_set ){
							if( !point_label_set->discardpoint ){
							  /* We just allocate as much space as needed here. If ever points are added to this set
							   \ (by loading additional files), realloc_points will take care of increasing the arena.
							   */
								point_label_set->discardpoint= (signed char*) calloc( point_label_set->allocSize, sizeof(signed char));
							}
							if( del_upto ){
								i= 0;
								I= point_l_nr+ 1;
							}
							else if( del_from ){
								i= point_l_nr;
								I= point_label_set->numPoints;
							}
							else{
								i= point_l_nr;
								I= point_l_nr+ 1;
							}
							if( point_label_set->discardpoint ){
								  /* Save the old state first..	*/
								DiscardUndo.idx= point_l_nr;
								if( (DiscardUndo.set= point_label_set) ){
									  /* 990614: I don't see (yet) why I should include window-specific
									   \ discards in the undo function. Those are intended to discard
									   \ points outside the visible region.
									   */
									DiscardUndo.split= DiscardedPoint( NULL, point_label_set, point_l_nr);
								}
								if( CheckMask(mask_rtn_released, ShiftMask) ){
								  /* if we got the memory, swap the discard-indicator(s) at this/these point(s).	*/
									for( ; i< I; i++ ){
										point_label_set->discardpoint[i]= ! point_label_set->discardpoint[i];
									}
								}
								else{
								  /* if we got the memory, delete this/these point(s).	*/
									for( ; i< I; i++ ){
										point_label_set->discardpoint[i]= 1;
									}
								}
								point_label_set->init_pass= True;
							}
							DiscardUndo.ul.label[0]= '\0';
							RedrawSet( point_label_set->set_nr, 1 );
							handled= True;
						}
						else if( ULabel ){
							DiscardUndo.set= NULL;
							DiscardUndo.ul= *ULabel;
							  /* Find the label's "parent" label (its "car")	*/
							DiscardUndo.head= wi->ulabel;
							if( DiscardUndo.head!= ULabel ){
								while( DiscardUndo.head && DiscardUndo.head->next!= ULabel ){
									DiscardUndo.head= DiscardUndo.head->next;
								}
							}
							  /* Make it disappear at the next redraw:	*/
							ULabel->label[0]= '\0';
							wi->redraw= 1;
							wi->raw_display= !raw_display;
							handled= True;
						}
						ShiftUndo.set= NULL;
						ShiftUndo.ul= NULL;
						LabelClipUndo= NULL;
						SplitUndo.set= NULL;
						wi->PlaceUndo.valid= False;
						if( !filt && BoxFilterUndo.fp ){
						  /* Don't delete this undo if we just created it!	*/
							fclose( BoxFilterUndo.fp );
							BoxFilterUndo.fp= NULL;
						}
						wi->cutAction&= ~_deleteAction;
						*cur= (CursorCross)? noCursor : theCursor;
					}
					else{
					  double *ulx, *uly, dx, dy;
					  Boolean setX= False, setY= False;
						  /* See which co-ordinate to change. No need to touch the other(s),
						   \ since redetermining the co-ordinate from its screen-representation
						   \ may entail errors. The undo buffer(s) are always updated!
						   */
						if( curX!= startX ){
							_hiX= Reform_X( wi, TRANX(curX), TRANY(curY) )/ wi->Xscale;
							setX= True;
						}
						if( curY!= startY ){
							_hiY= Reform_Y( wi, TRANY(curY), TRANX(curX) )/ wi->Yscale;
							setY= True;
						}
						if( ULabel ){
							ulx= (point_l_nr== 0)? &(ULabel->x1) : &(ULabel->x2);
							uly= (point_l_nr== 0)? &(ULabel->y1) : &(ULabel->y2);
						}
#ifdef DEBUG_SHIFTING
						if( point_label_set ){
							if( displ_sets> 1 ){
								fprintf( StdErr, "moving group #%d of set #%d[%d] (%d)",
									wi->group[point_label_set->set_nr],
									point_label_set->set_nr, point_l_nr, displ_sets
								);
							}
							else{
								fprintf( StdErr, "moving set #%d[%d]",
									point_label_set->set_nr, point_l_nr
								);
							}
							fprintf( StdErr, " from (%s,%s) to (%s,%s)\n",
								d2str( XVAL( point_label_set, point_l_nr), "%g", NULL),
								d2str( YVAL( point_label_set, point_l_nr), "%g", NULL),
								d2str( _hiX, (setX)? "%g" : "SAME", NULL), d2str( _hiY, (setY)? "%g" : "SAME", NULL)
							);
						}
						else if( ULabel ){
							fprintf( StdErr, "moving ULabel \"%s\"[%d] from (%s,%s) to (%s,%s)\n",
								ULabel->label, point_l_nr,
								d2str( *ulx, "%g", NULL),
								d2str( *uly, "%g", NULL),
								d2str( _hiX, (setX)? "%g" : "SAME", NULL), d2str( _hiY, (setY)? "%g" : "SAME", NULL)
							);
						}
#endif
						ShiftUndo.idx= point_l_nr;
						if( point_label_set || displ_set ){
						  /* Actually, when displ_set, point_label_set too..	*/
							ShiftUndo.whole_set= CheckMask( mask_rtn_released, ShiftMask);
						}
						if( (ShiftUndo.set= point_label_set) ){
							ShiftUndo.sets= displ_sets;
							if( !ShiftUndo.whole_set ){
								ShiftUndo.x= point_label_set->columns[point_label_set->xcol][point_l_nr];
								ShiftUndo.y= point_label_set->columns[point_label_set->ycol][point_l_nr];
								if( setX ){
									point_label_set->columns[point_label_set->xcol][point_l_nr]= _hiX;
								}
								if( setY ){
									point_label_set->columns[point_label_set->ycol][point_l_nr]= _hiY;
								}
								LastActionDetails[0]= 1;
								LastActionDetails[1]= point_label_set->columns[point_label_set->xcol][point_l_nr]- ShiftUndo.x;
								LastActionDetails[2]= point_label_set->columns[point_label_set->ycol][point_l_nr]- ShiftUndo.y;
								LastActionDetails[3]= 1;
								LastActionDetails[4]= 0;
							}
							else{
							  /* Holding down the Shiftkey should move the whole set by the
							   \ specified amounts. The x,y fields of the ShiftUndo struct
							   \ are thus interpreted as delta-x, delta-y.
							   */
							  int idx, i, np= 0;
							  DataSet *this_set= point_label_set;
								if( displ_set ){
									ShiftUndo.set= (DataSet*) displ_set;
								}
								ShiftUndo.x= dx= (setX)? _hiX- this_set->columns[this_set->xcol][point_l_nr] : 0;
								ShiftUndo.y= dy= (setY)? _hiY- this_set->columns[this_set->ycol][point_l_nr] : 0;
								for( idx= 0; idx< displ_sets; idx++ ){
									if( displ_set ){
										this_set= displ_set[idx];
									}
									if( setX ){
										this_set->displaced_x+= dx;
										for( i= 0; i< this_set->numPoints; i++ ){
											this_set->columns[this_set->xcol][i]+= dx;
										}
									}
									if( setY ){
										this_set->displaced_y+= dy;
										for( i= 0; i< this_set->numPoints; i++ ){
											this_set->columns[this_set->ycol][i]+= dy;
										}
									}
									RedrawSet( this_set->set_nr, 0 );
									if( setX || setY ){
										np+= i;
									}
								}
								RedrawSet( point_label_set->set_nr, 1 );
								LastActionDetails[0]= 1;
								LastActionDetails[1]= ShiftUndo.x;
								LastActionDetails[2]= ShiftUndo.y;
								LastActionDetails[3]= np;
								LastActionDetails[4]= idx;
							}
							ShiftUndo.ul= NULL;
						}
						else if( (ShiftUndo.ul= ULabel) ){
							ShiftUndo.x= *ulx;
							ShiftUndo.y= *uly;
							ShiftUndo.whole_set= 0;
							if( point_l_nr && *ulx == ULabel->x1 && *uly== ULabel->y1 ){
								if( setX ){
									ULabel->x1= _hiX;
								}
								if( setY ){
									ULabel->y1= _hiY;
								}
							}
							if( setX ){
								*ulx= _hiX;
							}
							if( setY ){
								*uly= _hiY;
							}
						}
						  /* To ensure that the next lines generate a redraw ;-)	*/
						wi->raw_display= !raw_display;
						LabelClipUndo= NULL;
						SplitUndo.set= NULL;
						DiscardUndo.set= NULL;
						DiscardUndo.ul.label[0]= '\0';
						if( wi->PlaceUndo.valid ){
							wi->PlaceUndo.valid= False;
						}
						if( BoxFilterUndo.fp ){
							fclose( BoxFilterUndo.fp );
							BoxFilterUndo.fp= NULL;
						}
					}
				}
				if( wi->raw_display != raw_display ){
delete_aborted:;
					wi->raw_display= raw_display;
					RedrawNow( wi );
				}
				XG_XSync( disp, False );
				stopFlag = 1;
				  /* LockMask already removed: remove other unwanted:	*/
				mask_rtn_released&= ~(Mod2Mask|Mod3Mask|Mod4Mask|Mod5Mask);
				if( CheckExclMask(mask_rtn_released, ShiftMask) )
				{
				  int okX= 1;
				  char *Name= titleText;
				  double x= Reform_X( wi, TRANX(curX), TRANY(curY) )/ wi->Xscale,
					y= Reform_Y( wi, TRANY(curY), TRANX(curX) )/ wi->Yscale;
					ShiftUndo.set= NULL;
					ShiftUndo.ul= NULL;
					LabelClipUndo= NULL;
					wi->PlaceUndo.valid= True;
					SplitUndo.set= NULL;
					DiscardUndo.set= NULL;
					DiscardUndo.ul.label[0]= '\0';
					if( BoxFilterUndo.fp ){
						fclose( BoxFilterUndo.fp );
						BoxFilterUndo.fp= NULL;
					}
					switch( theEvent.xbutton.button ){
						case Button2:
							wi->PlaceUndo.which= YPlaced;
							wi->PlaceUndo.x= wi->yname_x;
							wi->PlaceUndo.y= wi->yname_y;
							wi->PlaceUndo.placed= wi->yname_placed;
							wi->PlaceUndo.trans= wi->yname_trans;
							yname_x= wi->yname_x= x;
							yname_y= wi->yname_y= y;
							wi->yname_placed= 1;
							wi->yname_trans= (allow_name_trans ||
									!(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
							);
							fprintf( stdout, "\"%s\" -y_ul%s %g,%g\n",
								(name)? name : Name, (wi->yname_trans)? "1" : "", yname_x, yname_y
							);
							break;
						case Button3:
							wi->PlaceUndo.which= XPlaced;
							wi->PlaceUndo.x= wi->xname_x;
							wi->PlaceUndo.y= wi->xname_y;
							wi->PlaceUndo.placed= wi->xname_placed;
							wi->PlaceUndo.trans= wi->xname_trans;
							xname_x= wi->xname_x= x;
							xname_y= wi->xname_y= y;
							wi->xname_placed= 1;
							wi->xname_trans= (allow_name_trans ||
									!(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
							);
							fprintf( stdout, "\"%s\" -x_ul%s %g,%g\n",
								(name)? name : Name, (wi->xname_trans)? "1" : "", xname_x, xname_y
							);
							break;
						case Button1:
						default:
							if( capslock ){
								wi->PlaceUndo.which= IntensityLegendPlaced;
								wi->PlaceUndo.x= wi->IntensityLegend._legend_ulx;
								wi->PlaceUndo.y= wi->IntensityLegend._legend_uly;
								wi->PlaceUndo.sx= wi->IntensityLegend.legend_ulx;
								wi->PlaceUndo.sy= wi->IntensityLegend.legend_uly;
								wi->PlaceUndo.placed= wi->IntensityLegend.legend_placed;
								wi->PlaceUndo.trans= wi->IntensityLegend.legend_trans;
								legend_ulx= wi->IntensityLegend._legend_ulx= x;
								legend_uly= wi->IntensityLegend._legend_uly= y;
								do_transform( wi, "IntensityLegend.legend_ul", __DLINE__,
									"HandleMouse(IntensityLegend.legend_ul)", &okX, NULL,
									&legend_ulx, NULL, NULL, &legend_uly,
									NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, (wi->raw_display)? 0 : -1, 0, False
								);
								wi->IntensityLegend.legend_ulx= SCREENX( wi, legend_ulx );
								wi->IntensityLegend.legend_uly= SCREENY( wi, legend_uly );
								wi->IntensityLegend.legend_placed= 1;
								wi->IntensityLegend.legend_trans= (allow_name_trans ||
										!(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
								);
								fprintf( stdout, "\"%s\" -intensity_legend_ul%s %g,%g\n",
									(name)? name : Name, (wi->IntensityLegend.legend_trans)? "1" : "",
									wi->IntensityLegend._legend_ulx, wi->IntensityLegend._legend_uly
								);
							}
							else{
								wi->PlaceUndo.which= LegendPlaced;
								wi->PlaceUndo.x= wi->_legend_ulx;
								wi->PlaceUndo.y= wi->_legend_uly;
								wi->PlaceUndo.sx= wi->legend_ulx;
								wi->PlaceUndo.sy= wi->legend_uly;
								wi->PlaceUndo.placed= wi->legend_placed;
								wi->PlaceUndo.trans= wi->legend_trans;
								legend_ulx= wi->_legend_ulx= x;
								legend_uly= wi->_legend_uly= y;
								do_transform( wi, "legend_ul", __DLINE__, "HandleMouse(legend_ul)", &okX, NULL,
									&legend_ulx, NULL, NULL, &legend_uly,
									NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, (wi->raw_display)? 0 : -1, 0, False
								);
								wi->legend_ulx= SCREENX( wi, legend_ulx );
								wi->legend_uly= SCREENY( wi, legend_uly );
								wi->legend_placed= 1;
								wi->legend_trans= (allow_name_trans ||
										!(wi->transform.x_len || wi->transform.y_len || wi->process.data_process_len)
								);
								fprintf( stdout, "\"%s\" -legend_ul%s %g,%g\n",
									(name)? name : Name, (wi->legend_trans)? "1" : "", wi->_legend_ulx, wi->_legend_uly
								);
							}
							break;
					}
					if( !wi->redraw ){
						wi->redraw= 1;
					}
					wi->printed= 0;
					fflush( stdout );
				}
				  /* 20020930: check for measure<=1 instead of measure==1 to re-enable zoom-print. */
/* 				else if( !(CheckMask(mask_rtn_pressed, Button2Mask) || measure==1 || handled) )	*/
				else if( !(CheckMask(mask_rtn_pressed, Button2Mask)) || measure<=1 || handled )
				{
					  /* Figure out relative bounding box */
					_loX= Reform_X( wi, TRANX(startX), TRANY(startY) )/ wi->Xscale;
					_hiX= Reform_X( wi, TRANX(curX), TRANY(curY) )/ wi->Xscale;
					_loY= Reform_Y( wi, TRANY(startY) , TRANX(startX) )/ wi->Yscale;
					_hiY= Reform_Y( wi, TRANY(curY), TRANX(curX) )/ wi->Yscale;
					_lopX= _loX;
					_lopY= _loY;
					_hinY= - _lopY;
					if( wi->add_label ){
						Add_UserLabel( wi, NULL, _loX, _loY, _hiX, _hiY,
							point_label, point_label_set, point_l_nr, point_l_x, point_l_y, UL_regular, allow_name_trans,
							mask_rtn_pressed, mask_rtn_released, False
						);
						wi->add_label= 0;
						*cur= (CursorCross)? noCursor : theCursor;
						xtb_bt_set( wi->label_frame.win, wi->add_label, NULL);
					}
					  /* 990710: the following if(){} used to be under the
					   \ if( wi->add_label) { if( ControlMask ){ update_LinkedLabel }} block?!
					   \ 990714: And very well it should be.. Even outside the
					   \ wi->add_label block! This block
					   \ highlights a "selected" set, the ControlMask block
					   \ creates a linked label...
					   */
					else if( point_label && !del_from ){
						if( point_l_nr>= 0 && CheckMask( mask_rtn_pressed, Button3Mask) ){
							if( CheckMask( mask_rtn_released, Mod1Mask) ){
							  LocalWindows *WL= NULL;
							  LocalWin *lwi;
								if( handle_event_times== 0 ){
									WL= WindowList;
									lwi= WL->wi;
								}
								else{
									lwi= wi;
								}
								do{
									if( (lwi->legend_line[point_label_set->set_nr].highlight=
										!lwi->legend_line[point_label_set->set_nr].highlight)
									){
										lwi->dont_clear= 1;
										if( AllSets[point_label_set->set_nr].s_bounds_set ){
											AllSets[point_label_set->set_nr].s_bounds_set= -1;
										}
									}
									RedrawNow( lwi );
									if( WL ){
										WL= WL->next;
										lwi= (WL)? WL->wi : 0;
									}
								} while( WL && lwi );
							}
						}
					}
/*
					else if( measure== 2 ){
					  Already done..
					}
*/
					  /* 20030220: don't do the following for measure!=0 instead of measure!=2 ! */
					else if( !measure && startX-curX!= 0 && startY-curY!= 0 ){
						wi->raw_display= 1;
						if( !raw_display && debugFlag ){
							fprintf( StdErr, "HandleMouse(): raw_display flag temporarily set to True\n");
							fflush( StdErr);
						}
						if( _loX > _hiX && !(polarFlag && INSIDE(wi,_loX, _hiX, 0.75, 0.25)) ){
							double temp;

							temp = _hiX;
							_hiX = _loX;
							_loX = temp;
						}
						if( _loY > _hiY ){
							double temp;

							temp = _hiY;
							_hiY = _loY;
							_loY = temp;
						}
						if( CheckMask(mask_rtn_pressed, Button3Mask) ){
							wi->loX= _loX;
							wi->loY= _loY;
							wi->lopX= _lopX;
							wi->lopY= _lopY;
							wi->hinY= _hinY;
							wi->hiX= _hiX;
							wi->hiY= _hiY;
							wi->win_geo.user_coordinates= user_coordinates;
							wi->fit_xbounds= False;
							wi->fit_ybounds= 0;
							wi->FitOnce= 0;
							wi->aspect= 0;
							wi->redraw= True;
						}
						else{
clone_window:;
							new_win = NewWindow(progname, &new_info, _loX, _loY, _lopX, _lopY, _hinY, _hiX, _hiY,
										NORMASP, wi, 1.0, 1.0, 1.0, add_padding
							);
						}
						wi->raw_display= raw_display;
						if( new_win && new_info ){
							  /* 20000407: why not do the following?	*/
							if( cloning ){
								CopyFlags( new_info, wi );
							}
							if( ActiveWin== wi ){
								ActiveWin= new_info;
							}
							numwin = 1;
							new_info->raw_display= raw_display;
							new_info->ctr_A= wi->ctr_A;
							for( i= 0; i< MaxSets; i++ ){
								new_info->draw_set[i]= wi->draw_set[i];
								new_info->fileNumber[i]= wi->fileNumber[i];
							}
							new_info->Xscale= wi->Xscale;
							new_info->Yscale= wi->Yscale;
							new_info->redraw= -1;
							new_info->win_geo.user_coordinates= user_coordinates;
#ifdef STRICT_PADDING_COPYING
							if( wi->user_coordinates && wi->logXFlag && wi->logXFlag!= -1 && wi->_log_zero_x> 0 &&
								wi->logYFlag && wi->logYFlag!= -1 && wi->_log_zero_y> 0
							){
							  /* In these cases the current window will add padding in TransformCompute
							   \ so the new window should mimic that...
							   */
								new_info->win_geo.padding= wi->win_geo.padding;
							}
#else
							  /* We want the new window to behave exactly as the current window
							   \ might behave... NewWindow() didn't add padding, but if other functions
							   \ want to - go ahead.
							   */
							new_info->win_geo.padding= wi->win_geo.padding;
#endif
							if( !cloning ){
								new_info->legend_placed= 0;
								new_info->xname_placed= 0;
								new_info->yname_placed= 0;
								new_info->IntensityLegend.legend_placed= 0;
								  /* If user zoomed, we really want to see that area,
								   \ and not necessarily all displayed data... So, no
								   \ adaptive scaling!
								   */
								new_info->fit_xbounds= 0;
								new_info->fit_ybounds= 0;
								new_info->FitOnce= 0;
								new_info->aspect= 0;
							}
							else{
							  /* Do things exclusively for window clones:
							   \ copy the geometry
							   */
							  XWindowAttributes win_attr;
							  Window dummy;
								XGetWindowAttributes(disp, wi->window, &win_attr);
								XTranslateCoordinates( disp, wi->window, win_attr.root, 0, 0,
									&win_attr.x, &win_attr.y, &dummy
								);
								win_attr.y-= WM_TBAR;
								XMoveWindow( disp, new_info->window, win_attr.x- win_attr.border_width, win_attr.y );
								XResizeWindow( disp, new_info->window, win_attr.width, win_attr.height );
							}
							  /* 981210: UserLabels are also copied to zoomed windows.. why not, after all?!	*/
							Copy_ULabels( new_info, wi );
							if( wi->pen_list ){
							    /* 20031112: a should-be much faster way to copy pens to the new window. The
								 \ various flags being (re)set should not be needed.
								 */
								Copy_XGPens( wi, new_info );
							}
							if( ! new_info->raw_display && debugFlag && evt &&
								(new_info->transform.x_len || new_info->transform.y_len ||
								 new_info->process.data_process_len
								)
							){
								xtb_error_box( new_info->window, "May need to set raw_display for correct zooming\n"
										"(key 'R' or with Settings Dialogue)\n"
										"Ignoring special options!\n",
										"Warning"
								);
							}
							if( evt && theEvent.xbutton.button== Button2){
								print_immediate= 2;
							}
							else if( CheckMask(mask_rtn_pressed, ControlMask) ){
							  /* Cannot just DelWindow(wi->window,wi) here: if <wi> was
							   \ the only window around (NumWindows=1), DelWindow will exit
							   \ the program.
							   */
								wi->delete_it= 1;
								wi->redraw= 1;
							}
#if defined(__GNUC__) && defined(i386)
							if( new_info->pen_list || new_info->no_pens== -1 ){
							  int np = new_info->no_pens;
 								new_info->no_pens= 0;
								  /* 20031113:
								   \ drawing that new window here alleviates a weird phenomenon due to the new Copy_XGPens()
								   \ routine, and that permanently messes up the scaling. (It looks like another fl.point
								   \ alignment problem as it only occurs on x86, and not on MIPS/SGI with the same compiler.)
								   */
 								RedrawNow(new_info);
								new_info->no_pens = np;
							}
#endif
						}
					}
				}
				break;
			}
			case KeyPress:
				break;
			case KeyRelease:{
#pragma mark ***HMil KeyRelease
			  KeySym ks= XLookupKeysym( (XKeyEvent*) &(theEvent.xkey), 0);
				if( debugFlag ){
					fprintf( StdErr, "KeyRelease event: %s == %s, mask=%s\n",
						   event_name(theEvent.xany.type), XKeysymToString(ks),
						   xtb_modifiers_string(theEvent.xkey.state)
					);
				}
				if( ks== XK_Escape ){
					if( !CheckMask(mask_rtn_pressed, ShiftMask) ){
						DRAWECHO(0);
					}
					if( PGrabbed ){
						XUngrabPointer(disp, CurrentTime);
					}
					if( KGrabbed ){
						XUngrabKeyboard( disp, CurrentTime );
					}
					wi->redraw= 1;
					stopFlag= True;
				}
				break;
			}
			  /* 20050114: */
			case ButtonPress:{
#pragma mark ***HMil ButtonPress
				if( theEvent.xbutton.button>= 4 ){
#ifdef DEBUG
					if( debugFlag && debugLevel==-2 ){
						XQueryPointer(disp, win, &root_rtn, &child_rtn, &root_x, &root_y,
							  &newX, &newY, &mask_rtn_released
						);
						fprintf( StdErr, "ButtonRelease event: %s == 0x%lx (%s)",
							event_name(theEvent.xany.type),
							mask_rtn_released, xtb_modifiers_string(mask_rtn_released)
						);
						fprintf( StdErr, " state=0x%lx (%s) button=%d (%s) at %d,%d\n",
							theEvent.xbutton.state, xtb_modifiers_string(theEvent.xbutton.state),
							theEvent.xbutton.button,
							(theEvent.xbutton.button>=0 && theEvent.xbutton.button< sizeof(bnames)/sizeof(char*))?
									bnames[theEvent.xbutton.button] : "?!",
							newX, newY
						);
					}
					newX = theEvent.xbutton.x, newY = theEvent.xbutton.y;
					mask_rtn_released= xtb_Mod2toMod1( theEvent.xbutton.state );
					capslock= CheckMask( mask_rtn_released, LockMask);
					mask_rtn_released&= ~LockMask;
					mask_rtn_pressed |= buttonMask[theEvent.xbutton.button];
#endif
					switch( theEvent.xbutton.button ){
						case Button4:
							if( CheckMask(theEvent.xbutton.state, ControlMask) ){
								XWarpPointer( disp, None, None, 0,0,0,0, +1, 0 );
							}
							else{
								XWarpPointer( disp, None, None, 0,0,0,0, 0, -1 );
							}
							break;
						case Button5:
							if( CheckMask(theEvent.xbutton.state, ControlMask) ){
								XWarpPointer( disp, None, None, 0,0,0,0, -1, 0 );
							}
							else{
								XWarpPointer( disp, None, None, 0,0,0,0, 0, 1 );
							}
							break;
					}
					break;
				}
				else{
				  /* fall through to default case: */
				}
			}
			default:
				if( debugFlag ){
					fprintf( StdErr, "Unhandled event: %s\n", event_name(theEvent.xany.type) );
				}
				numwin= 0;
				break;
		}
    }
	if( name ){
		XStoreName( disp, wi->window, name );
		XFree( name );
		name= NULL;
	}
	if( New_Info ){
		*New_Info= new_info;
	}
	GCA();
	set_Find_Point_precision( xprec, yprec, NULL, NULL );
    return numwin;
}
#endif

/* Default line styles */
char *defStyle[MAXATTR] = {
    "11", "44", "1142", "31", "88", "113111", "2231", "224",
    "55", "66", "2253", "42", "99", "224222", "3342", "335"
  };

/* Default color names */
char *defColors[MAXATTR] = {
    "red", "indianred", "blue", "green",
    "cyan", "orchid", "orange", "violet",
    "red", "indianred", "blue", "green",
    "cyan", "orchid", "orange", "violet"
  };

extern Pixmap _XCreateBitmapFromData(Display *dis, Drawable win, char *mark_bits, unsigned int mark_w, unsigned int mark_h);

/* stat( filename, &stats);	*/

extern char *FontName(XGFontStruct *f);

extern int RememberedFont( Display *disp, XGFontStruct *font, char **rfont_name);

extern int RememberFont( Display *disp, XGFontStruct *font, char *font_name);

void Initialise_Sets( int start, int end )
{  int idx;
/*    char setname[64];	*/
    for( idx = start;  idx < end;  idx++ ){
/* 		sprintf( setname, "Set %d", idx );	*/
		AllSets[idx].set_nr= idx;
		AllSets[idx].setName = NULL;
		AllSets[idx].fileName = NULL;
		AllSets[idx].XUnits= NULL;
		AllSets[idx].YUnits= NULL;
		AllSets[idx].fileNumber = -1;
		AllSets[idx].titleText = NULL;
		AllSets[idx].new_file = 0;
		AllSets[idx].numPoints = 0;
		  /* by default, we support errorbars, but we don't have them yet	*/
		AllSets[idx].has_error = 0;
		AllSets[idx].use_error = 1;
		AllSets[idx].lineWidth = lineWidth;
		AllSets[idx].elineWidth = errorWidth;
		AllSets[idx].linestyle = DEFAULT_LINESTYLE(idx);
		AllSets[idx].elinestyle = DEFAULT_LINESTYLE(idx);
		AllSets[idx].pixvalue = DEFAULT_PIXVALUE(idx);
		AllSets[idx].pixelValue= AllAttrs[AllSets[idx].pixvalue].pixelValue;
		AllSets[idx].pixelCName= XGstrdup( AllAttrs[AllSets[idx].pixvalue].pixelCName );
		AllSets[idx].markstyle = DEFAULT_MARKSTYLE(idx);
		AllSets[idx].markFlag = markFlag;
		set_NaN( AllSets[idx].markSize );
		if( barBase_set ){
			AllSets[idx].barBase= barBase;
			AllSets[idx].barBase_set= barBase_set;
		}
		if( barWidth_set ){
			AllSets[idx].barWidth_set= barWidth_set;
			AllSets[idx].barWidth= barWidth;
		}
		AllSets[idx].barType= barType;
		AllSets[idx].arrows = arrows;
		AllSets[idx].vectorLength = vectorLength;
		AllSets[idx].vectorType = vectorType;
		memcpy( AllSets[idx].vectorPars, vectorPars, MAX_VECPARS* sizeof(double));
		AllSets[idx].overwrite_marks = overwrite_marks;
		AllSets[idx].noLines = noLines;
		AllSets[idx].floating= False;
		AllSets[idx].allocSize = 0;
		AllSets[idx].error_point = -1;
		AllSets[idx].xvec = AllSets[idx].yvec =
			AllSets[idx].ldxvec= AllSets[idx].hdxvec=
			AllSets[idx].ldyvec= AllSets[idx].hdyvec= (double *) 0;
		AllSets[idx].plot_interval= 0;
		AllSets[idx].adorn_interval= 0;

		AllSets[idx].Xscale= 1;
		AllSets[idx].Yscale= 1;
		AllSets[idx].DYscale= 1;

		if( !AllSets[idx].ncols ){
			AllSets[idx].ncols= NCols;
		}
		AllSets[idx].xcol= xcol;
		AllSets[idx].ycol= ycol;
		AllSets[idx].ecol= ecol;
		AllSets[idx].lcol= lcol;
		AllSets[idx].Ncol= Ncol;
		AllSets[idx].error_type= -1;

		AllSets[idx].allocAssociations= 0;
		AllSets[idx].numAssociations= 0;
		AllSets[idx].Associations= NULL;

		AllSets[idx].set_link= -1;
		AllSets[idx].set_linked= False;
		AllSets[idx].links= 0;

		AllSets[idx].init_pass= True;

    }
}

int BinaryFieldSize= 0, BinarySize= sizeof(double);
BinaryField BinaryDump, BinaryTerminator;

void InitSets()
/*
 * Initializes the data sets with default information.
 */
{
    int idx;
    Window temp_win;

	if( !(AllSets= calloc( MaxSets, sizeof(DataSet) )) ){
		fprintf( StdErr, "xgraph - can't allocate DataSets (%s)\n", serror() );
		exit(-10);
	}
	if( !(inFileNames= calloc( MaxSets, sizeof(char*) )) ){
		fprintf( StdErr, "xgraph - can't allocate filename buffers (%s)\n", serror() );
		exit(-10);
	}
	MaxnumFiles= MaxSets;

	BinaryDump.data= NULL;
	BinaryTerminator.data= NULL;

	XG_choose_visual();
    GetColor("black", &black_pixel );
	StoreCName( blackCName );
    GetColor("white", &white_pixel );
	StoreCName( whiteCName );

	temp_win= RootWindow( disp, screen);

    _bwFlag = (depth < 4);

    bdrSize = 2;
    bdrPixel = black_pixel;
	xfree( bdrCName); bdrCName= XGstrdup(blackCName);
    htickFlag = 0;
    vtickFlag = 0;
	zeroFlag= 0;
    markFlag = 0;
    pixelMarks = 0;
	axisFlag= 1;
    bbFlag = 0;
    noLines = 0;
    sqrtXFlag= logXFlag = 0;
    sqrtYFlag= logYFlag = 0;
	triangleFlag = 0;
	error_regionFlag= 0;
	polarFlag= 0;
    barFlag = 0;
	barBase_set= 0;
    barBase = 0.0;
	barWidth_set= 0;
    barWidth = -1.0;		/* Holder */
	barType= 0;
    lineWidth = 1;
	linestyle= 0;
	elinestyle= 0;
	pixvalue= 0;
	markstyle= 1;
    echoPix = BIGINT;
    /* Set the user bounding box */
	x_scale_start= 1;
	y_scale_start= 1;
	xp_scale_start= 1;
	yp_scale_start= 1;
	yn_scale_start= 1;
	llpx= -1; llpy= -1;
	llny= 1;
    /* Depends critically on whether the display has color */
    if( MonoChrome ){
		/* Its black and white */
		if( bgPixel== -1 && GetColor(whiteCName, &bgPixel) ){
			StoreCName( bgCName );
			if( debugFlag){
				fprintf( StdErr, "bgPixel (*Background) set to \"%s\"\n", bgCName );
				fflush( StdErr);
			}
		}
		if( zeroPixel== -1 && GetColor( blackCName, &zeroPixel) ){
			StoreCName( zeroCName );
			if( debugFlag){
				fprintf( StdErr, "zeroPixel (*ZeroColor) set to \"%s\"\n", zeroCName );
				fflush( StdErr);
			}
		}
		if( axisPixel== -1 && GetColor( blackCName, &axisPixel) ){
			StoreCName( axisCName );
			if( debugFlag){
				fprintf( StdErr, "axisPixel (*AxisColor) set to \"%s\"\n", axisCName );
				fflush( StdErr);
			}
		}
		if( gridPixel== -1 && GetColor( blackCName, &gridPixel) ){
			StoreCName( gridCName );
			if( debugFlag){
				fprintf( StdErr, "gridPixel (*GridColor) set to \"%s\"\n", gridCName );
				fflush( StdErr);
			}
		}
		zeroWidth = 3;
		if( normPixel== -1 && GetColor( blackCName, &normPixel) ){
			StoreCName( normCName );
			if( debugFlag){
				fprintf( StdErr, "normPixel (*Foreground) set to black_pixel (%s)\n", normCName );
				fflush( StdErr);
			}
		}
		  /* Initialize set defaults */
		for( idx = 0;  idx < MAXATTR;  idx++ ){
			/* Needs work! */
			AllAttrs[idx].lineStyleLen =
			  xtb_ProcessStyle(defStyle[idx], AllAttrs[idx].lineStyle, MAXLS);
			if( GetColor( normCName, &AllAttrs[idx].pixelValue) ){
				StoreCName( AllAttrs[idx].pixelCName );
			}
		}
    } else {
		/* Its color */
		if( bgPixel== -1){
			if( GetColor("LightGray", &bgPixel) ){
				StoreCName( bgCName );
				if( debugFlag){
					fprintf( StdErr, "bgPixel (*Background) set to LigthGray\n");
					fflush( StdErr);
				}
			}
		}
		if( zeroPixel== -1){
			if( GetColor( "Red", &zeroPixel) ){
				StoreCName( zeroCName );
				if( debugFlag){
					fprintf( StdErr, "zeroPixel (*ZeroColor) set to Red\n");
					fflush( StdErr);
				}
			}
		}
		if( axisPixel== -1 && GetColor( blackCName, &axisPixel) ){
			StoreCName( axisCName );
			if( debugFlag){
				fprintf( StdErr, "axisPixel (*AxisColor) set to \"%s\"\n", axisCName );
				fflush( StdErr);
			}
		}
		if( gridPixel== -1){
			if( GetColor( "Gray50", &gridPixel) ){
				StoreCName( gridCName );
				if( debugFlag){
					fprintf( StdErr, "gridPixel (*GridColor) set to Gray50\n");
					fflush( StdErr);
				}
			}
		}
		zeroWidth = 1;
		if( normPixel== -1){
			normPixel = black_pixel;
			xfree( normCName ); normCName= XGstrdup(blackCName);
			if( debugFlag){
				fprintf( StdErr, "normPixel (*Foreground) set to black_pixel\n");
				fflush( StdErr);
			}
		}
		/* Initalize attribute colors defaults */
/* 		AllAttrs[0].lineStyle[0] = '\0';	*/
/* 		AllAttrs[0].lineStyleLen = 0;	*/
		GetColor(defColors[0], &(AllAttrs[0].pixelValue) );
		StoreCName( AllAttrs[0].pixelCName );
		for( idx = 1;  idx < MAXATTR;  idx++ ){
			AllAttrs[idx].lineStyleLen =
			  xtb_ProcessStyle(defStyle[idx-1], AllAttrs[idx].lineStyle, MAXLS);
			GetColor(defColors[idx], &(AllAttrs[idx].pixelValue) );
			StoreCName( AllAttrs[idx].pixelCName );
		}
    }
      /* Initialize the data sets */
	Initialise_Sets( 0, MAXSETS );
      /* Store bitmaps for dots and markers */
    dotMap = _XCreateBitmapFromData(disp, temp_win, dot_bits, dot_w, dot_h);

    AllAttrs[0].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark1_bits, mark_w, mark_h);
    AllAttrs[1].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark2_bits, mark_w, mark_h);
    AllAttrs[2].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark3_bits, mark_w, mark_h);
    AllAttrs[3].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark4_bits, mark_w, mark_h);
    AllAttrs[4].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark5_bits, mark_w, mark_h);
    AllAttrs[5].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark6_bits, mark_w, mark_h);
    AllAttrs[6].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark7_bits, mark_w, mark_h);
    AllAttrs[7].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark8_bits, mark_w, mark_h);
    AllAttrs[8].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark9_bits, mark_w, mark_h);
    AllAttrs[9].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark10_bits, mark_w, mark_h);
    AllAttrs[10].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark11_bits, mark_w, mark_h);
    AllAttrs[11].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark12_bits, mark_w, mark_h);
    AllAttrs[12].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark13_bits, mark_w, mark_h);
    AllAttrs[13].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark14_bits, mark_w, mark_h);
    AllAttrs[14].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark15_bits, mark_w, mark_h);
    AllAttrs[15].markStyle = _XCreateBitmapFromData(disp, temp_win,
						  mark16_bits, mark_w, mark_h);
}

extern char *def_str;
extern Pixel def_pixel;
extern int def_int;
extern XFontStruct *def_font;
extern double def_dbl;

extern int rd_pix(char *name);

extern int rd_int(char *name);

extern int rd_str(char *name);

extern int rd_font(char *name, char **font_name);

extern int rd_flag(char *name);

extern int rd_dbl(char *name);

extern char *GetFont( XGFontStruct *Font, char *resource_name, char *default_font, long size, int bold, int use_remembered );

extern int ReadDefaults();

extern int PS_PrintComment;
int Show_Progress= 1;

extern int AddPoint_discard;

void reset_Scalers(char *msg)
{
	  /* set scalers to global scalers	*/
	if( debugFlag){
		fprintf( StdErr, "Resetting set specific scale factors (%s)\n", msg);
		fprintf( StdErr, "File wide scale factors: %g %g %g\n", _Xscale, _Yscale, _DYscale);
		fprintf( StdErr, "Set specific scale factors were: %g %g %g\n", Xscale, Yscale, DYscale);
		fprintf( StdErr, "Max. scale factors: %g %g %g\n", _MXscale, _MYscale, _MDYscale );
		fprintf( StdErr, "Max. scale factors based on data*scale: %g %g %g\n", MXscale, MYscale, MDYscale );
	}
	Xscale= _Xscale;
	Yscale= _Yscale;
	DYscale= _DYscale;
}

double TRANSFORMED_x, TRANSFORMED_y, TRANSFORMED_ldy, TRANSFORMED_hdy;

/* reset_ascanf_currentself_value is set to False (0) in almost every case before calling
 \ fascanf() or fascanf2(). This is because ascanf_current and ascanf_self are controlled
 \ by the calling program (us), and not by the fascanf() routines.
 \ reset_ascanf_index_value should actually be controlled by fascanf(), but is set to True
 \ just in case.
 */

int do_TRANSFORM( LocalWin *lwi, int spot, int nrx, int nry, double *xvec, double *ldxvec, double *hdxvec,
	double *yvec, double *ldyvec, double *hdyvec, int is_bounds, int just_doit
)
{ int ret= 0, ugi= use_greek_inf;
   use_greek_inf= 0;
#ifdef DEBUG
{ int ok= 1;
	if( !ok ){
		return(0);
	}
}
#endif
	if( !lwi ){
		fprintf( StdErr, "do_TRANSFORM(lwi=NULL!, spot=%d, nrx=%d,nry=%d, xvec=%s, yvec=%s,..., is_bounds= %d\n",
			spot, nrx, nry, d2str( *xvec, "%g", NULL), d2str( *yvec, "%g", NULL), is_bounds
		);
		fflush( StdErr);
		use_greek_inf= ugi;
		return(0);
	}
	if( lwi->absYFlag && *yvec< 0 && !NaNorInf(*yvec) && !is_bounds ){
	  /* translate over 2*yvec to get absolute value	*/
	  double dy= -2* *yvec;
		*yvec+= dy;
		*ldyvec+= dy;
		*hdyvec+= dy;
	}

	if( just_doit || !( lwi->raw_display ||
			  /* _process_bounds is set to False (0) when a *TRANSFORM_?* is encountered in
			   \ the datafile. It ensures that the bounds of the data are determined correctly (raw).
			   \ DrawWindow sets it to True before any actual drawing takes place.
			   */
			(is_bounds && (_process_bounds== 0 || lwi->process_bounds==0)) ||
			(is_bounds && lwi->transform_axes<= 0 ) ||
			  /* is_bounds==-1 means we're doing a placement of the legend,x-label or y-label box	*/
			is_bounds== -1
		)
	){
		if( nrx> 3 ){
			nrx= 3;
		}
		if( nry> 3 ){
			nry= 3;
		}
		if( lwi->transform.x_len ){
		  int i, n, rsacsv= reset_ascanf_currentself_value;
		  char change[1], *name[3]= { "x", "lx", "hx"};
		  double val[4];
		  int aae= 0;
			val[0]= *xvec;
			for( i= 0; i< nrx; i++ ){
				n= 1;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "do_TRANSFORM(): TRANSFORM_X %s: %s", name[i], lwi->transform.x_process );
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*TRANSFORM_X*";
				compiled_fascanf( &n, lwi->transform.x_process, val, change, val, NULL, &lwi->transform.C_x_process );
				aae+= ascanf_arg_error;
				if( /* ascanf_arg_error || */ !n ){
					fprintf( StdErr, "do_TRANSFORM(#%d): TRANSFORM_X %s %s: error (err=%d,n=%d)\n",
						spot, name[i], d2str( val[0], "%g", NULL),
						ascanf_arg_error, n
					);
					fflush( StdErr );
				}
				/* else */{
					switch( i ){
						case 0:
							*xvec= val[0];
							if( ldxvec ){
								val[0]= *ldxvec;
							}
							else if( hdxvec ){
								val[0]= *hdxvec;
								i= 1;
							}
							break;
						case 1:
							if( ldxvec ){
								*ldxvec= val[0];
								if( hdxvec ){
									val[0]= *hdxvec;
								}
							}
							break;
						case 2:
							if( hdxvec ){
								*hdxvec= val[0];
							}
							break;
					}
					ret+= 1;
				}
			}
			reset_ascanf_currentself_value= rsacsv;
		}
		if( lwi->transform.y_len ){
		  int i, n, rsacsv= reset_ascanf_currentself_value;
		  double val[4];
		  char change[1], *name[3]= { "y", "ly", "hy"};
		  int aae= 0;
			val[0]= *yvec;
			for( i= 0; i< nry; i++ ){
				n= 1;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "do_TRANSFORM(): TRANSFORM_Y %s: %s", name[i], lwi->transform.y_process );
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*TRANSFORM_Y*";
				compiled_fascanf( &n, lwi->transform.y_process, val, change, val, NULL, &lwi->transform.C_y_process );
				aae+= ascanf_arg_error;
				if( /* ascanf_arg_error || */ !n ){
					fprintf( StdErr, "do_TRANSFORM(#%d): TRANSFORM_Y %s %s: error (err=%d,n=%d)\n",
						spot, name[i], d2str( val[0], "%g", NULL),
						ascanf_arg_error, n
					);
					fflush( StdErr );
				}
				/* else */{
				  /* substitute values.	*/
					switch( i ){
						case 0:
							*yvec= val[0];
							if( ldyvec ){
								val[0]= *ldyvec;
							}
							break;
						case 1:
							if( ldyvec ){
								*ldyvec= val[0];
								if( hdyvec ){
									val[0]= *hdyvec;
								}
							}
							break;
						case 2:
							if( hdyvec ){
								*hdyvec= val[0];
							}
							break;
					}
					ret+= 1;
				}
			}
			reset_ascanf_currentself_value= rsacsv;
		}
		use_greek_inf= ugi;
		return( ret? ret : -1 );
	}
	TBARprogress_header= NULL;
	use_greek_inf= ugi;
	return( 0 );
}

/* Perform axis-transformations on a datapoint. For axis-bounds, the transformations specified
 \ by *TRANSFORM_?* are also performed (is_bounds=1). If is_bounds==0, whether or not this
 \ transformation is performed is determined by do_TRANSFORM(). For is_bounds==-1, it is not done.
 \ This routine also substitutes zero for log_zero_? in logarithmic mode, and does the polar
 \ transformation (always as the last step!)
 */
int do_transform( LocalWin *wi, char *filename, double line_count, char *buffer, int *spot_ok, DataSet *this_set,
	double *xvec, double *ldxvec, double *hdxvec, double *yvec, double *ldyvec, double *hdyvec,
	double *xvec_1, double *yvec_1, int use_log_zero, int spot, double xscale, double yscale, double dyscale, int is_bounds,
	int data_nr, Boolean just_doit
)
{
	double (*funcx)(), (*funcy)();
	double Ldxvec=1.0, Hdxvec=1.0, Ldyvec=1.0, Hdyvec=1.0, y_sign, ldy_sign, hdy_sign;
	int verbose= (debugFlag || strncmp( buffer, "Draw", 4) );
	LocalWin LWI, *lwi;

#ifdef DEBUG
	{ int ok= 1;
		if( !ok ){
			return(0);
		}
	}
#endif
		if( !*spot_ok ){
			return(0);
		}

	/* Settings and flags must be accessed through lwi
	 \ in this routine!!
	 */
		if( !wi ){
			lwi= &LWI;
			memset( lwi, 0, sizeof(LocalWin));
			CopyFlags( lwi, NULL );
			lwi->raw_display= 0;
		}
		else{
			lwi= (LocalWin*) wi;
		}

		if( !ldxvec ){
			ldxvec= &Ldxvec;
			*ldxvec= *xvec;
		}
		if( !hdxvec ){
			hdxvec= &Hdxvec;
			*hdxvec= *xvec;
		}
		if( !ldyvec ){
			ldyvec= &Ldyvec;
			*ldyvec= *yvec;
		}
		if( !hdyvec ){
			hdyvec= &Hdyvec;
			*hdyvec= *yvec;
		}

		funcx= Trans_X;
		funcy= Trans_Y;

		*spot_ok= 1;

		if( this_set ){
			xscale= this_set->Xscale;
			yscale= this_set->Yscale;
			dyscale= this_set->DYscale;
		}
		if( debugFlag && debugLevel== -1 ){
			fprintf( StdErr, "do_transform(0x%lx,%s,%g,%s,0x%lx=%d,%s,\n",
				wi, (filename)?filename:"NULL", line_count,
				(buffer)?buffer:"NULL", spot_ok, *spot_ok, (this_set)?this_set->setName:"NULL"
			);
			fprintf( StdErr, "\t(%g,%g,%g),(%g,%g,%g)\t{[%d,%d^%g]|[%d,%d^%g]},\n\t%g,%g,%d,%d\n) * (%g,%g,%g) =>",
				*xvec, *ldxvec, *hdxvec, *yvec, *ldyvec, *hdyvec,
				lwi->logXFlag, lwi->sqrtXFlag, lwi->powXFlag, lwi->logYFlag, lwi->sqrtYFlag, lwi->powYFlag,
				(xvec_1)? *xvec_1 : 0, (yvec_1)? *yvec_1 : 0, use_log_zero, spot,
				xscale, yscale, dyscale
			);
			fflush( StdErr);
		}

		*xvec*= xscale;
		*yvec*= yscale;

		if( this_set ){
/* 950619: this happens in DrawData(): data[][2] contains the transformed error. Many transformations
 \ however will cause the y-value yvec to drift off-center from the (correct) errorbar. In those
 \ cases the following computation is incorrect! (It is still done for the cases where yvec is substituted
 \ for log_zero_y: there we have no better choice).
			*ldyvec= *yvec - this_set->data[data_nr][2]* dyscale;
			*hdyvec= *yvec + this_set->data[data_nr][2]* dyscale;
 */
		}
		else{
			*ldyvec*= dyscale;
			*hdyvec*= dyscale;
		}
		if( NaNorInf(*ldyvec) ){
			*ldyvec= 0.0;
		}
		if( NaNorInf(*hdyvec) ){
			*hdyvec= 0.0;
		}

		if( lwi->absYFlag && *yvec< 0 && !NaNorInf(*yvec) && !is_bounds ){
		  /* translate over 2*yvec to get absolute value	*/
		  double dy= -2* *yvec;
			*yvec+= dy;
			*ldyvec+= dy;
			*hdyvec+= dy;
		}

		if( lwi->vectorFlag || (wi && this_set && wi->error_type[this_set->set_nr]== MSIZE_FLAG) ){
		  /* Shouldn't do anything here (? who knows what will turn up :)))	*/
		}
		else if( !lwi->polarFlag ){
			*ldxvec= *xvec;
			*hdxvec= *xvec;
			ldy_sign= y_sign= hdy_sign= 1.0;
		}
		else{
		  /* y is actually a length, so we should allow
		   \ logarithms of negative values by making them
		   \ positive at first, and restoring the sign
		   \ in the end
		   */
			ldy_sign= (*ldyvec< 0)? -1.0 : 1.0;
			y_sign= (*yvec< 0)? -1.0 : 1.0;
			hdy_sign= (*hdyvec< 0)? -1.0 : 1.0;
			*ldyvec*= ldy_sign;
			*yvec*= y_sign;
			*hdyvec*= hdy_sign;
		}

		do_TRANSFORM( lwi, spot, 1, 3, xvec, NULL, NULL, yvec, ldyvec, hdyvec, is_bounds, just_doit );
		TRANSFORMED_x= *xvec;
		TRANSFORMED_y= *yvec;
		TRANSFORMED_ldy= *ldyvec;
		TRANSFORMED_hdy= *hdyvec;

		if( !NaNorInf(*xvec) ){
			if( ((lwi->sqrtXFlag && lwi->sqrtXFlag!=-1)) || (lwi->logXFlag && lwi->logXFlag!= -1) /* && !lwi->polarFlag	*/ ){
			  int log_done;
			  /* if sqrt mode uses cus_sqrt, no checks for negatives are necessary	*/
				  /* substitute 0 for a non-positive value if requested	*/
				if( lwi->logXFlag> 0 && lwi->_log_zero_x && use_log_zero && *xvec== 0 && is_bounds<= 0 ){
					*xvec= lwi->log10_zero_x;
					*ldxvec= *xvec;
					*hdxvec= *xvec;
					log_done= 1;
				}
				else{
					log_done= 0;
				}
				if( lwi->logXFlag> 0 && !log_done && *xvec<= 0.0 && is_bounds> 0 ){
					if( *xvec< 0 && lwi->logXFlag== 3 ){
						*xvec= (*funcx)( lwi, *xvec);
						log_done= 1;
					}
					else if( lwi->lopX> lwi->_log_zero_x && lwi->_log_zero_x && use_log_zero ){
						*xvec= lwi->log10_zero_x;
						*ldxvec= *xvec;
						*hdxvec= *xvec;
						log_done= 1;
					}
					else if( lwi->lopX> 0.0 ){
						*xvec= lwi->lopX;
						*ldxvec= *xvec;
						*hdxvec= *xvec;
					}
					else{
						*spot_ok= 0;
					}
				}
				if( lwi->logXFlag> 0 && *xvec <= 0.0 && !log_done ){
					if( !(lwi->logXFlag== 3 && *xvec< 0.0) ){
						if( verbose ){
							fprintf(StdErr, "file: `%s', line: %g (%s)\n",
								   filename, line_count, buffer
							);
							fprintf(StdErr, "X=%g value in sqrt/logarithmic mode\n", *xvec);
						}
						*spot_ok= 0;
						if( xvec_1 ){
							*ldyvec= *hdyvec= *yvec= *yvec_1;
							*xvec= *xvec_1;
						}
					}
				}
				if( *spot_ok ){
					if( !log_done)
						*xvec = (*funcx)( lwi, *xvec);
					if( lwi->logXFlag> 0 || lwi->sqrtXFlag> 0 ){
						if( lwi->logXFlag> 0 && lwi->_log_zero_x && use_log_zero && *ldxvec== 0.0 )
							*ldxvec= lwi->log10_zero_x;
						else if( (*ldxvec< 0.0 && lwi->logXFlag== 3) || *ldxvec> 0.0 || lwi->logXFlag<= 0 )
							*ldxvec= (*funcx)( lwi, *ldxvec);
						else
							*ldxvec= *xvec;
						if( lwi->logXFlag> 0 && lwi->_log_zero_x && use_log_zero && *hdxvec== 0.0 )
							*hdxvec= lwi->log10_zero_x;
						else if( (*hdxvec< 0.0 && lwi->logXFlag== 3) || *hdxvec> 0.0 || lwi->logXFlag<= 0 )
							*hdxvec= (*funcx)( lwi, *hdxvec);
						else
							*hdxvec= *xvec;
					}
				}
			}
			else{
				*xvec= Trans_X( lwi, *xvec );
				*ldxvec= Trans_X( lwi, *ldxvec );
				*hdxvec= Trans_X( lwi, *hdxvec );
			}
		}
		else{
			if( lwi->polarFlag ){
			  /* x=Inf in polarmode? Make that x=lwi->radix (e.g. 360 degrees)	*/
				*xvec= lwi->radix+ lwi->radix_offset;
			}
			else if( lwi->logXFlag && lwi->logXFlag!= -1 && Inf(*xvec)== -1 ){
				if( verbose ){
					fprintf( StdErr, "X= -Inf in log mode\n");
					fflush( StdErr);
				}
				*spot_ok= 0;
				if( xvec_1 ){
					*ldyvec= *hdyvec= *yvec= *yvec_1;
					*xvec= *xvec_1;
				}
			}
		}
		if( !lwi->polarFlag && !lwi->vectorFlag && (!wi || !this_set || wi->error_type[this_set->set_nr]!= MSIZE_FLAG) ){
			*ldxvec= *xvec;
			*hdxvec= *xvec;
		}

		if( !NaNorInf(*yvec) ){
			if( ((lwi->sqrtYFlag && lwi->sqrtYFlag!=-1)) || (lwi->logYFlag && lwi->logYFlag!= -1) ){
			  int log_done;
				if( lwi->logYFlag> 0 && lwi->_log_zero_y && use_log_zero && *yvec== 0 && is_bounds<= 0 ){
					*yvec= lwi->log10_zero_y;
					if( this_set ){
						*ldyvec= lwi->_log_zero_y - this_set->data[data_nr][2] * dyscale;
						*hdyvec= lwi->_log_zero_y + this_set->data[data_nr][2] * dyscale;
					}
					log_done= 1;
				}
				else{
					log_done= 0;
				}
				if( lwi->logYFlag> 0 && !log_done && *yvec<= 0.0 && is_bounds> 0 ){
					if( *yvec< 0 && lwi->logYFlag== 3 ){
						*yvec= (*funcy)(lwi, *yvec, is_bounds);
						log_done= 1;
					}
					else if( lwi->lopY> lwi->_log_zero_y && lwi->_log_zero_y && use_log_zero ){
						*yvec= lwi->log10_zero_y;
						if( this_set ){
							*ldyvec= lwi->_log_zero_y - this_set->data[data_nr][2] * dyscale;
							*hdyvec= lwi->_log_zero_y + this_set->data[data_nr][2] * dyscale;
						}
						log_done= 1;
					}
					else if( lwi->lopY> 0.0 ){
						*yvec= lwi->lopY;
						if( this_set ){
							*ldyvec= *yvec - this_set->data[data_nr][2] * dyscale;
							*hdyvec= *yvec + this_set->data[data_nr][2] * dyscale;
						}
					}
					else{
						*spot_ok= 0;
					}
				}
				if( (lwi->logYFlag> 0 && *yvec <= 0.0) && !log_done ){
					if( !(lwi->logYFlag== 3 && *yvec< 0.0) ){
						if( verbose ){
							fprintf(StdErr, "file: `%s', line: %g (%s)\n",
								   filename, line_count, buffer
							);
							fprintf(StdErr, "Y=%g value in sqrt/logarithmic mode\n", *yvec);
						}
						*spot_ok= 0;
						if( xvec_1 ){
							*ldyvec= *hdyvec= *yvec= *yvec_1;
							*xvec= *xvec_1;
						}
					}
				}
				if( *spot_ok ){
					if( !log_done)
						*yvec = (*funcy)(lwi, *yvec, is_bounds);
					if( lwi->logYFlag> 0 || lwi->sqrtYFlag> 0 ){
						if( lwi->logYFlag> 0 && lwi->_log_zero_y && use_log_zero && *ldyvec== 0.0 )
							*ldyvec= lwi->log10_zero_y;
						else if( (*ldyvec< 0.0 && lwi->logYFlag== 3) || *ldyvec> 0.0 || lwi->logYFlag<= 0 )
							*ldyvec= (*funcy)( lwi, *ldyvec, is_bounds);
						else{
/* 							*ldyvec= *yvec;	*/
							  /* set it to -Inf, the only sensible answer to log(0) . This will cause
							   \ the errorbar to be plotted downwards out of view; an error-region will
							   \ just be "invisible" there. Setting *ldyvec=*yvec would cause a lack of
							   \ the downward errorbar (no problem), but would also cause the error-region
							   \ to suggest an asymmetric error.
							   */
							set_Inf( *ldyvec, -1 );
						}
						if( lwi->logYFlag> 0 && lwi->_log_zero_y && use_log_zero && *hdyvec== 0.0 )
							*hdyvec= lwi->log10_zero_y;
						else if( (*hdyvec< 0.0 && lwi->logYFlag== 3) || *hdyvec> 0.0 || lwi->logYFlag<= 0 )
							*hdyvec= (*funcy)( lwi, *hdyvec, is_bounds);
						else{
/* 							*hdyvec= *yvec;	*/
							  /* This probably never happens - I think it will cause desirable behaviour.	*/
							set_Inf( *hdyvec, -1 );
						}
					}
				}
			}
			else{
				*yvec= Trans_Y( lwi, *yvec, is_bounds );
				if( lwi->absYFlag ){
					if( *ldyvec< 0 && *hdyvec > 0 ){
						*ldyvec= 0.0;
					}
					else{
						*ldyvec= Trans_Y( lwi, *ldyvec, is_bounds );
					}
					*hdyvec= Trans_Y( lwi, *hdyvec, is_bounds );
				}
				else{
					*ldyvec= Trans_Y( lwi, *ldyvec, is_bounds );
					*hdyvec= Trans_Y( lwi, *hdyvec, is_bounds );
#define SWAP(a,b,type)	{type c= (a); (a)= (b); (b)= c;}
					if( !lwi->vectorFlag ){
					  /* In "vector" mode, really don't make sure the low y is smaller
					   \ than the high y...
					   */
						if( *ldyvec> *hdyvec ){
							SWAP( *ldyvec, *hdyvec, double );
						}
					}
				}
			}
		}
		else if( lwi->logYFlag && lwi->logYFlag!= -1 && Inf(*yvec)== -1 ){
			if( verbose ){
				fprintf( StdErr, "Y= -Inf in log mode\n");
				fflush( StdErr);
			}
			*spot_ok= 0;
			if( xvec_1 ){
				*ldyvec= *hdyvec= *yvec= *yvec_1;
				*xvec= *xvec_1;
			}
		}

		if( lwi->polarFlag ){
		  double x, y;
		  double sinx= cus_pow( Sin(*xvec * XscaleR), lwi->powAFlag);
		  double cosx= cus_pow( Cos(*xvec * XscaleR), lwi->powAFlag);
			if( this_set && this_set->barFlag> 0 ){
				this_set->barFlag*= -1;
			}
			*ldyvec*= ldy_sign;
			*yvec*= y_sign;
			*hdyvec*= hdy_sign;

			x= cosx * *ldyvec;
			y= sinx * *ldyvec;
			*ldxvec= x;
			*ldyvec= y;

			x= cosx * *hdyvec;
			y= sinx * *hdyvec;
			*hdxvec= x;
			*hdyvec= y;

			x= cosx * *yvec;
			y= sinx * *yvec;
			*xvec= x;
			*yvec= y;

		}
		else if( this_set && this_set->barFlag< 0 ){
			this_set->barFlag*= -1;
		}

		if( debugFlag && debugLevel== -1 ){
			fprintf( StdErr, " (%g,%g,%g),(%g,%g,%g),%d\n",
				*xvec, *ldxvec, *hdxvec, *yvec, *ldyvec, *hdyvec,
				*spot_ok
			);
			fflush( StdErr);
		}
		return( *spot_ok );
}

typedef struct value{
	double x, y, err;
	int flag, has_error;
} Values;

typedef struct xvalue{
	double x;
	int indeks, set;
} XValues;

extern XValues *X_Values;

extern XValues *Search_Xval_Back( XValues *XXval, long indeks, Values *Val);

extern Values *Search_Val_Forward( XValues *Xval, long xv, long NXvalues, Values **_Values, long set, long N);

/* make a dump in the Cricket Graph TM ascii format	*/
extern int _SpreadSheetDump( LocalWin *wi, FILE *fp, char errmsg[ERRBUFSIZE], int CricketGraph);

extern void DrawData( LocalWin *win, Boolean bounds_only );

extern char *ascanf_emsg;

int DiscardedShadows= 0;


extern double *param_scratch;
extern int param_scratch_len;

int Draw_Process(LocalWin *wi, int before )
{
#ifdef DEBUG
	if( !wi ){
		fprintf( StdErr, "Draw_Process() called with NULL argument\n" );
		return(0);
	}
#endif
	if( !RAW_DISPLAY(wi) ){
	  int n, rsacsv= reset_ascanf_currentself_value;
			clean_param_scratch();
			if( before ){
				if( wi->process.draw_before_len ){
					n= param_scratch_len;
					*ascanf_self_value= 0.0;
					*ascanf_current_value= 0.0;
					*ascanf_counter= (*ascanf_Counter)= 0.0;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					*ascanf_setNumber= 0;
					*ascanf_numPoints= 0;
					if( ascanf_verbose ){
						fprintf( StdErr, "DrawWindow(): DRAW Before: %s", wi->process.draw_before);
						fflush( StdErr );
					}
					TitleMessage( wi, "*DRAW_BEFORE*" );
					TBARprogress_header= "*DRAW_BEFORE*";
					ascanf_arg_error= 0;
					compiled_fascanf( &n, wi->process.draw_before, param_scratch, NULL, data, column, &wi->process.C_draw_before );
					TBARprogress_header= NULL;
				}
			}
			else{
				if( wi->process.draw_after_len ){
					n= param_scratch_len;
					*ascanf_self_value= 0.0;
					*ascanf_current_value= 0.0;
					*ascanf_counter= (*ascanf_Counter)= 0.0;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					*ascanf_numPoints= 0;
					if( ascanf_verbose ){
						fprintf( StdErr, "DrawData(): DRAW After: %s", wi->process.draw_after );
						fflush( StdErr );
					}
					TitleMessage( wi, "*DRAW_AFTER*" );
					TBARprogress_header= "*DRAW_AFTER*";
					ascanf_arg_error= 0;
					compiled_fascanf( &n, wi->process.draw_after, param_scratch, NULL, data, column, &wi->process.C_draw_after );
					TBARprogress_header= NULL;
				}
			}
		reset_ascanf_currentself_value= rsacsv;
		TitleMessage( wi, NULL );
	}
	return(1);
}

LocalWin *ChangeWindow= NULL;

double PointTotal, win_aspect_precision= 0.01;

int DrawWindow_Update_Procs= 0;

extern ColourFunction IntensityColourFunction;
int DrawIntensityLegend( LocalWin*, Boolean);

/* Change the co-ordinates of the axes as passed in the last 4 arguments
 \ according to various settings for the window passed in the first argument.
 \ Updating the window's settings in the win_geo.bounds structure must be
 \ done by the caller. When absflag is True, AlterGeoBounds() considers the
 \ the different settings as 0/1 booleans, and not as <=0/>0 booleans. This
 \ is to make it work during fitting. NB: absFlag is thus not synonym for wi->absYFlag!!
 \ 20010726: This routine can be extended to take a user-defined data-window into
 \ account. Doing that after (optional) autoscaling means extra work, but will
 \ probably be the most versatile and transparent option.
 */
void AlterGeoBounds( LocalWin *wi, int absflag, double *_loX, double *_loY, double *_hiX, double *_hiY )
{
	if( wi->aspect> 0 || (absflag && wi->aspect) ){
	 int i;
		  // 20081215: this is a rather huge hack to try to address the issue that bounds don't stay fixed in 1:1 mode.
		if( wi->win_geo.bounds._loX == *_loX && !NaN(wi->win_geo.aspect_base_bounds._loX) ){
			*_loX = wi->win_geo.aspect_base_bounds._loX;
		}
		else{
			wi->win_geo.aspect_base_bounds._loX = *_loX;
		}
		if( wi->win_geo.bounds._loY == *_loY && !NaN(wi->win_geo.aspect_base_bounds._loY) ){
			*_loY = wi->win_geo.aspect_base_bounds._loY;
		}
		else{
			wi->win_geo.aspect_base_bounds._loY = *_loY;
		}
		if( wi->win_geo.bounds._hiX == *_hiX && !NaN(wi->win_geo.aspect_base_bounds._hiX) ){
			*_hiX = wi->win_geo.aspect_base_bounds._hiX;
		}
		else{
			wi->win_geo.aspect_base_bounds._hiX = *_hiX;
		}
		if( wi->win_geo.bounds._hiY == *_hiY && !NaN(wi->win_geo.aspect_base_bounds._hiY) ){
			*_hiY = wi->win_geo.aspect_base_bounds._hiY;
		}
		else{
			wi->win_geo.aspect_base_bounds._hiY = *_hiY;
		}
		for( i= 0; i< 1; i++ ){
		  double dx= *_hiX - *_loX, dy= *_hiY - *_loY;
		  double aspect= (dy)? fabs( dx/ dy ) : 1.0;
		  double xrange, yrange, raspect;

			TransformCompute(wi, True);
			xrange= wi->XOppX- wi->XOrgX;
			yrange= wi->XOppY - wi->XOrgY;

			  /* compensate for the actual sizes of the window	*/
			raspect= (xrange && yrange)? xrange/ yrange : 1.0;
			aspect/= raspect;
			if( (aspect> 1.001 || aspect< 0.999) ){
			  double diff= ( aspect- 1)/ 2;
				if( dy== 0 ){
				  double d= fabs(dx)/2;
					*_loY-= d;
					*_hiY+= d;
				}
				else if( dx== 0 ){
				  double d= fabs(dy)/2;
					*_loX-= d;
					*_hiX+= d;
				}
				else if( diff> 0 ){
					*_loY-= dy* diff;
					*_hiY+= dy* diff;
				}
				else{
					diff= ( 1/aspect- 1)/2;
					*_loX-= dx* diff;
					*_hiX+= dx* diff;
				}
			}
		}
	}
	else{
	  double hX= ABS(*_hiX), lX= ABS(*_loX), hY= ABS(*_hiY), lY= ABS(*_loY);
// 		if( NaN(wi->win_geo.aspect_base_bounds._loX) ){
			wi->win_geo.aspect_base_bounds._loX = *_loX;
// 		}
// 		if( NaN(wi->win_geo.aspect_base_bounds._loY) ){
			wi->win_geo.aspect_base_bounds._loY = *_loY;
// 		}
// 		if( NaN(wi->win_geo.aspect_base_bounds._hiX) ){
			wi->win_geo.aspect_base_bounds._hiX = *_hiX;
// 		}
// 		if( NaN(wi->win_geo.aspect_base_bounds._hiY) ){
			wi->win_geo.aspect_base_bounds._hiY = *_hiY;
// 		}
		if( wi->x_symmetric> 0 || (absflag && wi->x_symmetric) ){
			if( lX> hX ){
				*_loX= -lX;
				*_hiX= lX;
			}
			else{
				*_loX= -hX;
				*_hiX= hX;
			}
		}
		if( (wi->y_symmetric> 0 || (absflag && wi->y_symmetric)) ){
			if( lY> hY ){
				*_loY= -lY;
				*_hiY= lY;
			}
			else{
				*_loY= -hY;
				*_hiY= hY;
			}
		}
	}
	if( wi->datawin.apply ){
		if( !NaN(wi->datawin.llX) && *_loX< wi->datawin.llX ){
			*_loX= wi->datawin.llX;
		}
		if( !NaN(wi->datawin.urX) && *_hiX> wi->datawin.urX ){
			*_hiX= wi->datawin.urX;
		}
		if( !NaN(wi->datawin.llY) && *_loY< wi->datawin.llY ){
			*_loY= wi->datawin.llY;
		}
		if( !NaN(wi->datawin.urY) && *_hiY> wi->datawin.urY ){
			*_hiY= wi->datawin.urY;
		}
	}
}

extern double Fit_ChangePerc_X, Fit_ChangePerc_Y, Fit_ChangePerc;

void Reset_AFTFit_History( LocalWin *wi )
{ int i;
	if( wi && wi->aftfit_history ){
	  FitHistory *h= wi->aftfit_history;
		for( i= 0; i< AFTFITHIST; i++ ){
			set_NaN(h->x);
			set_NaN(h->y);
			h++;
		}
	}
}

int Fit_After_Draw( LocalWin *wi, char *old_Wname )
{
	if( wi->fit_after_draw && !wi->fitting && (wi->fit_xbounds> 0 || wi->fit_ybounds> 0) ){
	  int rd= wi->redraw;
	  int asp= wi->aspect, xsym= wi->x_symmetric, ysym= wi->y_symmetric;
	  int fx= wi->fit_xbounds;
	  int fy= wi->fit_ybounds, ut= wi->use_transformed, fi= wi->fitting;
	  int changes= 0, dont= False;
		  /* determine the current data's bounds by calling DrawData
		   \ with the bounds_only flag set.
		   */
		wi->redraw= 0;
		wi->aspect= -ABS(wi->aspect);
		wi->x_symmetric= -ABS(wi->x_symmetric);
		wi->y_symmetric= -ABS(wi->y_symmetric);
		TitleMessage( wi, "(Re)Scaling after draw..." );
		  /* Fit_XBounds() maybe calls us again, so we should prevent endless loops.
		   \ Also, when we are re-called to establish the X-bounds, we do not have to
		   \ do the Y-bounds then!
		   */
		wi->fit_xbounds*= -1;
		wi->fit_ybounds*= -1;
		wi->use_transformed= True;
		wi->fitting= True;
		if( wi->fit_xbounds && wi->fit_ybounds ){
			changes= Fit_XYBounds( wi, False );
			if( wi->fit_after_precision>= 0 && fabs(Fit_ChangePerc)<= wi->fit_after_precision ){
				dont= True;
			}
		}
		else if( wi->fit_xbounds ){
			changes= Fit_XBounds( wi, False );
			if( wi->fit_after_precision>= 0 && fabs(Fit_ChangePerc_X)<= wi->fit_after_precision ){
				dont= True;
			}
		}
		else if( wi->fit_ybounds ){
			changes= Fit_YBounds( wi, False );
			if( wi->fit_after_precision>= 0 && fabs(Fit_ChangePerc_Y)<= wi->fit_after_precision ){
				dont= True;
			}
		}
		wi->use_transformed= ut;
		wi->fitting= fi;
		wi->fit_xbounds= fx;
		wi->fit_ybounds= fy;
		if( changes ){
		  char msg[256];
		  FitHistory *now, *prev;
		  int r= wi->redrawn-1, i, j, n= 0;
		  double dx= 0, dy= 0;
			  /* 20010801: maintain a history of percentual changes: */
			if( r< 0 ){
				r= 0;
			}
			now= &wi->aftfit_history[ r % AFTFITHIST ];
			now->x= Fit_ChangePerc_X;
			now->y= Fit_ChangePerc_Y;

			if( debugFlag && debugLevel ){
				fprintf( StdErr, "Change(%s,%s,r=%d);", d2str( Fit_ChangePerc_X,0,0), d2str( Fit_ChangePerc_Y,0,0), r % AFTFITHIST );
			}

			  /* Calculate the average change in these changes over the last
			   \ AFTFITHIST redraws:
			   */
			for( i= 1; i< AFTFITHIST && i<= r; i++ ){
				j= (r- i) % AFTFITHIST;
				prev= &wi->aftfit_history[j];
				if( !NaN(now->x) && !NaN(now->y) ){
					dx+= now->x- prev->x;
					dy+= now->y- prev->y;
					n+= 1;
					if( debugFlag && debugLevel ){
						fprintf( StdErr, " (%d,%s,%s)", n, d2str( dx,0,0), d2str( dy,0,0) );
					}
					now= prev;
				}
			}
			if( n ){
				dx/= n, dy/= n;
				if( debugFlag && debugLevel ){
					fputc( '\n', StdErr );
				}
			}
			sprintf( msg,
				"%d changes (%s%%,%s%%) to after-draw fit (%s%%,%s%% over last %d): %s.",
				changes,
				d2str( Fit_ChangePerc_X, NULL, NULL),
				d2str( Fit_ChangePerc_Y, NULL, NULL),
				d2str( dx, NULL, NULL), d2str( dy, NULL, NULL), n,
				(rd)? "redraw was pending already" : (dont)? "within precision, no redraw" : "scheduling redraw"
			);
			StringCheck( msg, sizeof(msg)/sizeof(char), __FILE__, __LINE__ );
			if( wi->SD_Dialog ){
				XStoreName( disp, wi->SD_Dialog->win, msg );
				if( !RemoteConnection ){
					XSync( disp, False );
				}
			}
			else{
				TitleMessage( wi, msg );
			}
			if( debugFlag ){
				fprintf( StdErr, "DrawWindow(): %s\n", msg );
			}
			if( !dont ){
				if( n>= AFTFITHIST-1 ){
				  double thres= - fabs(wi->fit_after_precision);
				  double average= 0;
				  int na= 0;
				  /* 20010801: there's one less interval than there are entries... :)	*/
					if( Fit_ChangePerc_X ){
						average+= dx;
						na+= 1;
					}
					if( Fit_ChangePerc_Y ){
						average+= dy;
						na+= 1;
					}
					if( na ){
						average/= na;
					}
					if( na && average> thres ){
						sprintf( msg,
							"After-draw fit (%s%%,%s%%)=%s%% over last %d redraws: fit_after looping aborted.",
							d2str( dx, NULL, NULL), d2str( dy, NULL, NULL), d2str( average, 0,0), n
						);
						StringCheck( msg, sizeof(msg)/sizeof(char), __FILE__, __LINE__ );
						fprintf( StdErr, "DrawWindow(): %s\n", msg );
						dont= True;

							/* Should reset these data after succesfull fit!! */
					}
				}
			}
			else{
				  /* Reset fitting history	*/
				Reset_AFTFit_History( wi );
				if( debugFlag && debugLevel ){
					fputs( "Found acceptable fit: reset after-fit history\n", StdErr );
				}
			}
		}
		else{
			  /* Reset fitting history	*/
			Reset_AFTFit_History( wi );
			if( debugFlag && debugLevel ){
				fputs( "Reset after-fit history\n", StdErr );
			}
		}
		if( old_Wname ){
			TitleMessage( wi,old_Wname);
		}
		if( (changes && !dont) || rd> 0 ){
			wi->redraw= True;
			wi->animate= 1;
			Animating= True;
			if( wi->SD_Dialog ){
				set_changeables(2,False);
			}
		}
		else{
			wi->redraw= rd;
		}
		wi->aspect= asp;
		wi->x_symmetric= xsym;
		wi->y_symmetric= ysym;
		return( dont );
	}
	return(0);
}

#ifdef __GNUC__
inline
#endif
void DBE_Swap( LocalWin *wi, XdbeSwapInfo *XDBE_info, int finish )
{
#ifdef XDBE_TIME_STATS
  extern double Delta_Tot_T;
#endif
	if( !wi->silenced ){
		  /* The initialisation of the swapping (as far as the server needs this)
		   \ should (probably!) be done at the start of a redraw, even if the first
		   \ thing done is autoscaling. Because this is the only way we can catch
		   \ the processing insided the swaps -- and that processing can generate
		   \ graphics output!
		   \ However, we do not take any action when the window is silenced (otherwise
		   \ it turns out that all output is lost!). While (in practice) this means that
		   \ e.g. pen output during processing should end up between the swaps, it also
		   \ means that the fitting procedure is effectively forced to take place outside
		   \ the swaps (since we silence the window for that). Clearly this implementation
		   \ can be improved!
		   */
		if( wi->XDBE_buffer && !(wi->XDBE_init || finish) ){
		  extern XdbeSwapAction XG_XDBE_SwapAction;	// XdbeBackground by default
			XDBE_info->swap_window= wi->window;
			XDBE_info->swap_action= XG_XDBE_SwapAction;
			XdbeBeginIdiom( disp );
			wi->XDBE_init= True;
#ifdef XDBE_TIME_STATS
			if( XDBE_validtimer ){
				Elapsed_Since( &XDBE_Timer, False );
				XDBE_outside+= Delta_Tot_T;
			}
			else{
				Elapsed_Since( &XDBE_Timer, True );
			}
#endif
		}
		if( !wi->fitting ){
			if( finish && wi->XDBE_buffer && wi->XDBE_init ){
				XdbeEndIdiom( disp );
				XdbeSwapBuffers( disp, XDBE_info, 1 );
				wi->XDBE_init= False;
#ifdef XDBE_TIME_STATS
				Elapsed_Since( &XDBE_Timer, False );
				if( XDBE_validtimer ){
					XDBE_inside+= Delta_Tot_T;
					XDBE_count+= 1;
				}
				XDBE_validtimer= True;
#endif
			}
			if( CursorCross && finish && !wi->animate ){
			  Window dum;
			  int dum2, sx, sy;
			  unsigned int mask_rtn;
				XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,
					  &sx, &sy, &mask_rtn
				);
				mask_rtn= xtb_Mod2toMod1( mask_rtn );
				DrawCCross( wi, False, sx, sy, NULL );
			}
		}
	}
}

int DrawWindow( LocalWin *wi)
/*
 * Draws the data in the window.  Does not clear the window.
 * The data is scaled so that all of the data will fit.
 * Grid lines are drawn at the nearest power of 10 in engineering
 * notation..  Draws axis numbers along bottom and left hand edges.
 * Centers title at top of window.
 \ 990806: added a bunch of checks for wi->fitting, to disable things
 \ when called by Fit_??Bounds().
 */
{   static int printing= 0;
	char *old_Wname= NULL;
	Boolean dont_clear= False /* , same_win */;
	int s_bounds_set= 0, ret_code= 1, PrepareAll= 0;
	int pim= print_immediate;
	static Boolean check_other_redraw= False;
	Time_Struct draw_timer= wi->draw_timer;
	static int level= 0;
	char msgbuf[512];
	LocalWin *AW= ActiveWin;
	XdbeSwapInfo _XDBE_info, *XDBE_info= &_XDBE_info;
	extern double *ascanf_IgnoreQuick, *ascanf_SetsReordered;
	int dbF= debugFlag, *dbFc= dbF_cache;

/* 	testrfftw(__FILE__,__LINE__,-1);	*/

	if( !wi ){
		fprintf( StdErr, "DrawWindow() called with NULL argument\n" );
		return(0);
	}
	else if( wi== &StubWindow ){
		fprintf( StdErr, "DrawWindow() called with the stub window pointer!!\n" );
		return(0);
	}

	if( wi->redraw== -2 ){
		return(0);
	}
	if( wi->debugFlag== 1 ){
		debugFlag= True;
	}
	else if( wi->debugFlag== -1 ){
		debugFlag= False;
	}
	dbF_cache= &dbF;

	  /* Check the global flag indicating we should re-do eventual calculations.  */
	if( *ascanf_IgnoreQuick && wi->use_transformed ){
	  int ut= wi->use_transformed, r;
		wi->use_transformed= 0;
		*ascanf_IgnoreQuick= 0;
		r= DrawWindow( wi );
		wi->use_transformed= ut;
/* 		SetWindowTitle( wi, wi->draw_timer.Tot_Time );	*/
		SetWindowTitle( wi, wi->draw_timer.HRTot_T );
		debugFlag= dbF;
		dbF_cache= dbFc;
		return(r);
	}

	if( !wi->draw_count ){
		Elapsed_Since( &draw_timer, True );
		if( !X_silenced(wi) ){
			Elapsed_Since(&wi->draw_timer, True);
		}
	}

	  /* First, check if this window ought to be deleted.	*/
	if( wi->delete_it ){
		DelWindow( wi->window, wi);
		debugFlag= dbF;
		dbF_cache= dbFc;
		if( !wi->delete_it ){
			deleteWindow= (Window) 0;
			return( 0);
		}
		else{
			wi->redraw= 0;
			return(0);
		}
	}
	if( !wi->window){
		wi->redraw= 0;
		DelWindow( wi->window, wi );
		debugFlag= dbF;
		dbF_cache= dbFc;
		return(0);
	}

	  /* Now, check some other conditions:	*/
	if( !wi->animate ){
		XFetchName( disp, wi->window, &old_Wname );
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/
	if( setNumber<= 0 ){
		if( !(ReadPipe && ReadPipe_fp) && wi!= InitWindow ){
			fprintf( StdErr, "DrawWindow(0x%lx,\"%s\"): no more sets to display - you should exit!\n",
				wi, (old_Wname)? old_Wname : "?" );
		}
		wi->redraw= 0;
		if( old_Wname ){
			XFree( old_Wname );
		}
		debugFlag= dbF;
		dbF_cache= dbFc;
		return(0);
	}
	if( !wi->dev_info.user_state ){
	  /* This should never happen!	*/
		fprintf( StdErr, "DrawWindow(0x%lx,\"%s\") wi->dev_info.user_state is NULL\n", wi, (old_Wname)? old_Wname : "??" );
		wi->redraw= 0;
		if( old_Wname ){
			XFree( old_Wname );
		}
		debugFlag= dbF;
		dbF_cache= dbFc;
		return(0);
	}

	  /* When (potentially! I don't want to write an elaborate check right now) drawing with transformations,
	   \ prevent concurrent redraws of windows that could cause inferences because of the non-entrantness of
	   \ the processing code. To do this, we mark a window requesting to be redrawn, and return immediately.
	   \ At the end of the toplevel call to DrawWindow (ourselves), we check the other windows for any such
	   \ marked windows, and redraw them. For this to work properly, we should never interfere with this
	   \ "redraw later" (: wi->redraw== -2) flag!
	   */
	  /* 990719: usually, the LocalWins are the same for 2 identical windows. However, do_hardcopy makes
	   \ a local copy, so in this case we have to verify the other elements. Probably, just checking
	   \ for identical X11 window IDs should be enough.
			same_win= ( ActiveWin== wi ||
				(ActiveWin->parent_number== wi->parent_number && ActiveWin->pwindow_number== wi->pwindow_number &&
					ActiveWin->window_number== wi->window_number && ActiveWin->window== wi->window)
			);
	   */
	if( (ActiveWin && ActiveWin->drawing) &&
		ActiveWin!= wi &&
		(!wi->raw_display || !ActiveWin->raw_display) &&
		PS_STATE(wi)->Printing== X_DISPLAY
	){
		check_other_redraw= True;
		if( debugFlag ){
			fprintf( StdErr, "DrawWindow(\"%02d.%02d.%02d\"): delaying redraw because already drawing %02d.%02d.%02d\n",
				wi->parent_number, wi->pwindow_number, wi->window_number,
				ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
			);
		}
		sprintf( msgbuf, "Waiting for window %02d.%02d.%02d",
			ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
		);
		XStoreName( disp, wi->window, msgbuf );
		wi->redraw= -2;
		wi->delayed= True;
		if( old_Wname ){
			XFree( old_Wname );
		}
		debugFlag= dbF;
		dbF_cache= dbFc;
		return(0);
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/


	if( PS_STATE(wi)->Printing== X_DISPLAY && !wi->drawing && !wi->fitting ){
	  /* If we're not printing, and noone else is drawing, and we are in the toplevl
	   \ invocation for the current window, it is probably save to update the LMAXBUFSIZE
	   \ variable, when this is required.
	   */
		if( Update_LMAXBUFSIZE(True, NULL) /* && debugFlag */ ){
			fprintf( StdErr, "DrawWindow(\"%02d.%02d.%02d\"): performed pending *BUFLEN* update\n",
				wi->parent_number, wi->pwindow_number, wi->window_number
			);
		}
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	  /* This shouldn't be necessary, but we do it to be sure	*/
	CopyFlags( NULL, wi );
/* testrfftw(__FILE__,__LINE__,-1);	*/

	if( debugFlag){
		fprintf( StdErr, "DrawWindow(0x%lx): drawing window 0x%lx \"%s\"\n",
			(unsigned long) wi, (unsigned long)wi->window, (old_Wname)? old_Wname : "?"
		);
		fflush( StdErr);
	}

	if( wi->halt ){
		Elapsed_Since( &draw_timer, True );
/* 		SetWindowTitle(wi, Tot_Time);	*/
		SetWindowTitle(wi, draw_timer.HRTot_T );
		if( old_Wname ){
			XFree( old_Wname );
		}
		debugFlag= dbF;
		dbF_cache= dbFc;
		return(0);
	}

	if( PS_STATE(wi)->Printing== PS_PRINTING ){
		gsResetTextWidths( wi, False );
	}

	ActiveWin= wi;
	ascanf_window= wi->window;
	*ascanf_ActiveWinWidth= wi->XOppX - wi->XOrgX;
	*ascanf_ActiveWinHeight= wi->XOppY - wi->XOrgY;

	*ascanf_SetsReordered= wi->sets_reordered;

	wi->BarDimensionsSet= 0;
	wi->eBarDimensionsSet= 0;

	wi->curs_cross.line[0].x1= -1;
	wi->curs_cross.line[0].y1= -1;
	wi->curs_cross.line[0].x2= -1;
	wi->curs_cross.line[0].y2= -1;
	wi->curs_cross.line[1].x1= -1;
	wi->curs_cross.line[1].y1= -1;
	wi->curs_cross.line[1].x2= -1;
	wi->curs_cross.line[1].y2= -1;
	wi->curs_cross.OldLabel[0]= '\0';

	wi->event_level++;
	wi->drawing= 1;

	  /* 20000502: 	*/
	if( wi->plot_only_set0== -1 ){
		wi->plot_only_set0= 0;
	}

	if( debugFlag && wi->event_level> 1 ){
	  char *name;
		XFetchName( disp, wi->window, &name );
		fprintf( StdErr, "DrawWindow(0x%lx=\"%x\") drawing with event_level==%d\n",
			wi, name, wi->event_level
		);
		XFree( name );
	}

	PointTotal= 0;

	if( !wi->animate ){
		parse_codes( wi->YUnits );
		parse_codes( wi->tr_YUnits );
		parse_codes( wi->XUnits );
		parse_codes( wi->tr_XUnits );
	}

	wi->num2Draw= 0;
	{ int i, mi= maxitems;
	  Boolean init_pass= False;
	  int iln= wi->IntensityLegend.legend_needed;
		wi->IntensityLegend.legend_needed= False;
		wi->first_drawn= -1;
		for( i= 0; i< setNumber; i++ ){
		  DataSet *this_set= &AllSets[i];
			if( this_set->set_link>= 0 && !this_set->set_linked ){
				if( LinkSet2( &AllSets[i], this_set->set_link )>= 0 ){
					this_set->init_pass= True;
				}
			}
			if( wi->xcol[i]< 0 ){
				wi->xcol[i]= this_set->xcol;
			}
			else{
				this_set->xcol= wi->xcol[i];
			}
			if( wi->ycol[i]< 0 ){
				wi->ycol[i]= this_set->ycol;
			}
			else{
				this_set->ycol= wi->ycol[i];
			}
			if( wi->ecol[i]< 0 ){
				wi->ecol[i]= this_set->ecol;
			}
			else{
				this_set->ecol= wi->ecol[i];
			}
			if( wi->lcol[i]< 0 ){
				wi->lcol[i]= this_set->lcol;
			}
			else{
				this_set->lcol= wi->lcol[i];
			}
			if( this_set->init_pass ){
				init_pass= True;
				PointTotal+= this_set->numPoints;
			}
			else if( wi->init_pass ){
				PointTotal+= this_set->numPoints;
			}
			if( !draw_set( wi, i ) ){
				PrepareAll+= 1;
				if( this_set->numPoints>= 0 ){
					  /* 20031021: unset visibility flags on not-drawn sets. Drawn sets will have
					   \ these flags set during the drawing.
					   */
					memset( wi->pointVisible[i], 0, this_set->numPoints * sizeof(signed char) );
				}
			}
			else{
				if( wi->first_drawn< 0 ){
					wi->first_drawn= i;
				}
				wi->last_drawn= i;
				wi->num2Draw+= 1;
				if( wi->error_type[i]== INTENSE_FLAG || (this_set->barType==4 && this_set->barFlag>0) ){
					if( !IntensityColourFunction.XColours ){
						Default_Intensity_Colours();
					}
					else if( IntensityColourFunction.XColours && !IntensityColourFunction.NColours ){
						if( !Intensity_Colours( IntensityColourFunction.expression ) ){
							xtb_error_box( wi->window,
								(IntensityColourFunction.expression)? IntensityColourFunction.expression :
									"<default colours>",
								"No intensity colours were allocated: unexpected behaviour!"
							);
						}
					}

					wi->IntensityLegend.legend_needed= -1;
				}
			}
			if( this_set->numPoints> mi ){
				mi= this_set->numPoints;
			}
		}
		if( mi> maxitems ){
			maxitems= mi;
			realloc_Xsegments();
		}
		if( init_pass ){
		  int raw= wi->raw_display, rd= wi->redraw;
		  char *msg = strdup( "Determining bounds & lengths..." );
			TitleMessage( wi, msg );
			if( wi->init_pass ){
			  /* 981111: let's see if we can speed up things by setting raw_display
			   \ to a temporary true:
			   */
				wi->raw_display= True;
			}
			wi->redraw= 0;
			do{
			  /* Repeat until at least the window's init_pass field is False (necessary when
			   \ e.g. first opening a window causes events that make DrawData return before
			   \ finishing.
			   */
				DrawData( wi, True );
				if( wi->init_pass ){
					msg = concat2( msg, ".", NULL );
					TitleMessage( wi, msg );
				}
			} while( wi->init_pass );
			xfree(msg);
			wi->raw_display= raw;
			wi->redraw= rd;
			TitleMessage( wi, old_Wname );
		}

/* 		if( wi->AlwaysDrawHighlighted ){	*/
/* 			for( i= 0; i< setNumber; i++ ){	*/
/* 				if( wi->legend_line[i].highlight ){	*/
/* 					wi->draw_set[i]= 1;	*/
/* 				}	*/
/* 			}	*/
/* 		}	*/

		if( !wi->no_intensity_legend ){
			if( wi->IntensityLegend.legend_needed && iln ){
			  /* restore cached value.	*/
				wi->IntensityLegend.legend_needed= iln;
			}
		}
		else{
				wi->IntensityLegend.legend_needed= False;
		}
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	  /* animation-flag must be reset explicitly!	*/
	if( !wi->fitting ){
		if( !wi->animate ){
			wi->animating= False;
		}
		wi->animate= 0;
	}

	*ascanf_TotalSets= setNumber;

	if( Startup_Exprs && !wait_evtimer && !settings_immediate ){
		if( Evaluate_ExpressionList( wi, &Startup_Exprs, True, "startup expressions" )== -2 ){
			ActiveWin= AW;
			if( old_Wname ){
				XFree( old_Wname );
			}
			debugFlag= dbF;
			dbF_cache= dbFc;
			return(0);
		}
	}

/* testrfftw(__FILE__,__LINE__,-1);	*/

	  /* Check if a new *TRANSFORM_?* or *DATA_????* has been specified.	*/
	if( transform_description ){
		xfree( wi->transform.description );
		wi->transform.description= transform_description;
		transform_description= NULL;
	}
	if( transform_x_buf && transform_x_buf_len ){
	  char asep = ascanf_separator;
		strcpalloc( &wi->transform.x_process, &wi->transform.x_allen, transform_x_buf );
		transform_x_buf[0]= '\0';
		transform_x_buf_len= 0;
		ascanf_separator = transform_separator;
		new_transform_x_process( wi);
		ascanf_separator = asep;
		if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
			wi->FitOnce= 1;
		}
	}
	if( transform_y_buf && transform_y_buf_len ){
	  char asep = ascanf_separator;
		strcpalloc( &wi->transform.y_process, &wi->transform.y_allen, transform_y_buf );
		transform_y_buf[0]= '\0';
		transform_y_buf_len= 0;
		ascanf_separator = transform_separator;
		new_transform_y_process( wi);
		ascanf_separator = asep;
		if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
			wi->FitOnce= 1;
		}
	}
	if( wi->transform.x_len== 0 && wi->transform.y_len== 0 ){
		if( wi->transform_axes== 0 ){
		  /* values of -1 indicates no TRANSFORM_X and no TRANSFORM_Y exists	*/
			wi->transform_axes= -1;
		}
	}
	else{
		if( wi->transform_axes<= 0 ){
			if( (wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val) ){
			  /* values of -1 indicates also that TRANSFORM_X and TRANSFORM_Y are not evaluated	*/
				wi->transform_axes= -1;
			}
			else{
				wi->transform_axes= 0;
			}
		}
	}
	if( !wi->transform_axes && !wi->raw_display ){
		if( (wi->logXFlag> 0 || wi->sqrtXFlag> 0 || wi->polarFlag> 0) && strlen( wi->transform.x_process ) ){
			xtb_error_box( wi->window, "DrawWindow(): set \"trax\" or unset either \"polar\", \"logX\" or \"powX\"\n", "Attention" );
			DoSettings(wi->window, wi);
			wi->redraw= 1;
			wi->printed= 0;
			ActiveWin= AW;
			wi->event_level--;
			wi->drawing= 0;
			if( old_Wname ){
				XFree( old_Wname );
			}
			debugFlag= dbF;
			dbF_cache= dbFc;
			return(0);
		}
		else if( (wi->logYFlag> 0 || wi->sqrtYFlag> 0 || wi->polarFlag> 0) && strlen( wi->transform.y_process ) ){
			xtb_error_box( wi->window, "DrawWindow(): set \"trax\" or unset either \"polar\", \"logY\" or \"powY\"\n", "Attention" );
			DoSettings(wi->window, wi);
			wi->redraw= 1;
			wi->printed= 0;
			ActiveWin= AW;
			wi->event_level--;
			wi->drawing= 0;
			if( old_Wname ){
				XFree( old_Wname );
			}
			debugFlag= dbF;
			dbF_cache= dbFc;
			return(0);
		}
	}

	wi->process.separator = ReadData_proc.separator;
	if( ReadData_proc.description ){
		xfree( wi->process.description );
		wi->process.description= ReadData_proc.description;
		ReadData_proc.description= NULL;
	}
	if( !ReadData_proc.data_process_now ){
		if( ReadData_proc.data_init_len ){
			strcpalloc( &wi->process.data_init, &wi->process.data_init_allen, ReadData_proc.data_init );
			ReadData_proc.data_init_len= 0;
			new_process_data_init( wi );
			wi->printed= 0;
			if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
				wi->FitOnce= 1;
			}
		}
		if( ReadData_proc.data_before_len ){
			strcpalloc( &wi->process.data_before, &wi->process.data_before_allen, ReadData_proc.data_before );
			ReadData_proc.data_before_len= 0;
			new_process_data_before( wi );
			wi->printed= 0;
			if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
				wi->FitOnce= 1;
			}
		}
		if( ReadData_proc.data_process_len ){
			strcpalloc( &wi->process.data_process, &wi->process.data_process_allen, ReadData_proc.data_process );
			ReadData_proc.data_process_len= 0;
			new_process_data_process( wi );
			wi->printed= 0;
			if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
				wi->FitOnce= 1;
			}
		}
		if( ReadData_proc.data_after_len ){
			strcpalloc( &wi->process.data_after, &wi->process.data_after_allen, ReadData_proc.data_after );
			ReadData_proc.data_after_len= 0;
			new_process_data_after( wi );
			wi->printed= 0;
			if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
				wi->FitOnce= 1;
			}
		}
		if( ReadData_proc.data_finish_len ){
			strcpalloc( &wi->process.data_finish, &wi->process.data_finish_allen, ReadData_proc.data_finish );
			ReadData_proc.data_finish_len= 0;
			new_process_data_finish( wi );
			wi->printed= 0;
			if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
				wi->FitOnce= 1;
			}
		}
	}

	if( ReadData_proc.draw_before_len ){
		strcpalloc( &wi->process.draw_before, &wi->process.draw_before_allen, ReadData_proc.draw_before );
		ReadData_proc.draw_before_len= 0;
		new_process_draw_before( wi );
		wi->printed= 0;
		if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
			wi->FitOnce= 1;
		}
	}
	if( ReadData_proc.draw_after_len ){
		strcpalloc( &wi->process.draw_after, &wi->process.draw_after_allen, ReadData_proc.draw_after );
		ReadData_proc.draw_after_len= 0;
		new_process_draw_after( wi );
		wi->printed= 0;
		if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) && NewProcess_Rescales ){
			wi->FitOnce= 1;
		}
	}

	if( ReadData_proc.dump_before_len ){
		strcpalloc( &wi->process.dump_before, &wi->process.dump_before_allen, ReadData_proc.dump_before );
		ReadData_proc.dump_before_len= 0;
		new_process_dump_before( wi );
		wi->printed= 0;
	}
	if( ReadData_proc.dump_after_len ){
		strcpalloc( &wi->process.dump_after, &wi->process.dump_after_allen, ReadData_proc.dump_after );
		ReadData_proc.dump_after_len= 0;
		new_process_dump_after( wi );
		wi->printed= 0;
	}

	if( ReadData_proc.enter_raw_after_len ){
		strcpalloc( &wi->process.enter_raw_after, &wi->process.enter_raw_after_allen, ReadData_proc.enter_raw_after );
		ReadData_proc.enter_raw_after_len= 0;
		new_process_enter_raw_after( wi );
		wi->printed= 0;
	}
	if( ReadData_proc.leave_raw_after_len ){
		strcpalloc( &wi->process.leave_raw_after, &wi->process.leave_raw_after_allen, ReadData_proc.leave_raw_after );
		ReadData_proc.leave_raw_after_len= 0;
		new_process_leave_raw_after( wi );
		wi->printed= 0;
	}

	  /* Move this to after the installation of all window-specific processing expressions! */
	if( wi->init_exprs && wi->new_init_exprs && !wait_evtimer && !settings_immediate ){
	  extern int ReadData_IgnoreVERSION;
	  int riv= ReadData_IgnoreVERSION;
		TitleMessage( wi, "Evaluating INIT expressions" );
		  /* It wouldn't be really useful to evaluate the VERSION, VERSION_LIST and SET_INFO_LIST commands
		   \ once more. Just parse them, but don't do anything with them.
		   */
		ReadData_IgnoreVERSION= True;
		if( Evaluate_ExpressionList( wi, &wi->init_exprs, False, "INIT expressions, \"window\" evaluation" )== -2 ){
			wi->new_init_exprs= False;
			ReadData_IgnoreVERSION= riv;
			if( old_Wname ){
				XFree( old_Wname );
			}
			ActiveWin= AW;
			debugFlag= dbF;
			dbF_cache= dbFc;
			return(0);
		}
		ReadData_IgnoreVERSION= riv;
		wi->new_init_exprs= False;
	}

	if( DrawWindow_Update_Procs ){
		wi->halt= 1;
		wi->redraw= 1;
		if( old_Wname ){
			XFree( old_Wname );
		}
		debugFlag= dbF;
		dbF_cache= dbFc;
		return( 1 );
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	level+= 1;

	  /* We're now going to some things that might potentially call us recursively
	   \ and thus mess with settings that we'd not want to change. So store their
	   \ setting (at function initialisation), and set them to 0 here, to reset
	   \ them just before they're actually needed.
	   */
	print_immediate= 0;

	if( wi->raw_once> 0 || (!wi->raw_display && wi->use_transformed) ){
		wi->raw_val= wi->raw_display;
		wi->raw_display= True;
		wi->raw_once= -1;
		if( debugFlag ){
			fprintf( StdErr, "Pre-rescale raw_once\n" );
		}
	}

	if( !(wi->raw_once== -1 && wi->raw_display) || wi->use_transformed
	  /* 980925: I don't know whether the following condition is necessary/constructive	*/
		|| wi->process_bounds
	){
		SS_Reset_( wi->SS_X );
		SS_Reset_( wi->SS_Y );
		SS_Reset_( wi->SS_Xval );
		SS_Reset_( wi->SS_Yval );
		SS_Reset_( wi->SS_E );
		  /* 991213: this one is not necessary for autoscaling. But it is
		   \ for determining the plotting region if the int.legend has not
		   \ been placed... If placed, it is probably safe to throw away the
		   \ gathered intensity statistics
		   */
		if( !wi->IntensityLegend.legend_needed || wi->IntensityLegend.legend_placed ||
			(wi->FitOnce || wi->fit_xbounds> 0 || wi->fit_ybounds> 0)
		){
			SS_Reset_( wi->SS_I );
		}
		wi->SAS_O.Gonio_Base= wi->radix;
		wi->SAS_O.Gonio_Offset= wi->radix_offset;
		wi->SAS_O.exact= 0;
		SAS_Reset_(wi->SAS_O );
		SS_Reset_( wi->SS_LY );
		SS_Reset_( wi->SS_HY );
		SS_Reset_( SS_Points );
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	{ int fo_ax;
		if( wi->FitOnce> 0 && wi->fit_xbounds> 0 && wi->fit_ybounds> 0 ){
			wi->FitOnce= 0;
		}
		if( wi->FitOnce> 0 ){
		  /* This is to prevent endless loops..	*/
			fo_ax= wi->FitOnce;
			wi->FitOnce= -1;
		}
		else if( wi->FitOnce== -1 ){
			wi->FitOnce= -2;
		}
		if( (wi->FitOnce== -1 || wi->fit_xbounds> 0 || wi->fit_ybounds> 0) &&
			!wi->fitting && (!wi->fit_after_draw || wi->FitOnce==-1)
		){
		  int rd= wi->redraw;
		  int asp= wi->aspect, xsym= wi->x_symmetric, ysym= wi->y_symmetric;
		  int fx= wi->fit_xbounds;
		  int fy= wi->fit_ybounds;
		  int silent;
			  /* determine the current data's bounds by calling DrawData
			   \ with the bounds_only flag set.
			   */
			wi->redraw= 0;
			wi->aspect= -ABS( wi->aspect );
			wi->x_symmetric= -ABS( wi->x_symmetric );
			wi->y_symmetric= -ABS( wi->y_symmetric );
			if( wi->FitOnce== -1 ){
				TitleMessage( wi, "Incidental rescaling..." );
			}
			else{
				TitleMessage( wi, "(Re)Scaling..." );
			}
			if( ( !wi->raw_display &&
					(wi->transform.x_len || wi->process.data_process_len || wi->transform.y_len ||
						wi->process.draw_before_len || wi->process.draw_after_len )
				) || wi->polarFlag
			){
				/* silent redraw handled by the Fit_.Bounds() functions.	*/
			}
			else{
			  int fi= wi->fitting;
				silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
				wi->fitting= True;
				DrawData( wi, True );
				if( wi->delete_it== -1 || wi->dev_info.user_state== NULL ){
					wi->aspect= asp;
					wi->x_symmetric= xsym;
					wi->y_symmetric= ysym;
					ret_code= 0; goto DrawWindow_return;
				}
				wi->fitting= fi;
				wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced || silent );
			}
			if( wi->FitOnce== -1 && (fo_ax== 'x' || fo_ax== 'y') ){
				wi->FitOnce= -2;
			}
			  /* Fit_XBounds() maybe calls us again, so we should prevent endless loops.
			   \ Also, when we are re-called to establish the X-bounds, we do not have to
			   \ do the Y-bounds then!
			   */
			wi->fit_xbounds*= -1;
			wi->fit_ybounds*= -1;
			if( wi->FitOnce== -1 || (wi->fit_xbounds && wi->fit_ybounds) ){
				Fit_XYBounds( wi, False );
				if( wi->raw_once== 0 || (!wi->raw_display && wi->use_transformed) ){
					wi->raw_once= 1;
				}
			}
			else if( wi->fit_xbounds || fo_ax== 'x' ){
				Fit_XBounds( wi, False );
				if( wi->raw_once== 0 || (!wi->raw_display && wi->use_transformed) ){
					wi->raw_once= 1;
				}
			}
			else if( wi->fit_ybounds || fo_ax== 'y' ){
				Fit_YBounds( wi, False );
				if( wi->raw_once== 0 || (!wi->raw_display && wi->use_transformed) ){
					wi->raw_once= 1;
				}
			}
			wi->fit_xbounds= fx;
			wi->fit_ybounds= fy;
			wi->redraw= rd;
			wi->aspect= asp;
			wi->x_symmetric= xsym;
			wi->y_symmetric= ysym;
			TitleMessage( wi,old_Wname);
		}
		if( wi->FitOnce ){
			wi->FitOnce= False;
		}
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	if( wi->raw_once< 0 ){
		wi->raw_display= wi->raw_val;
		wi->raw_once= 0;
		if( debugFlag ){
			fprintf( StdErr, "Pre-rescale raw_once desactivated\n" );
		}
	}
	else if( wi->raw_once> 0 || (!wi->raw_display && wi->use_transformed) ){
		wi->raw_val= wi->raw_display;
		wi->raw_display= True;
		wi->raw_once= -1;
		if( debugFlag ){
			fprintf( StdErr, "Post-rescale raw_once\n" );
		}
	}

	{ double _hiX= wi->win_geo.bounds._hiX, _loX= wi->win_geo.bounds._loX,
			_hiY= wi->win_geo.bounds._hiY, _loY= wi->win_geo.bounds._loY;
		AlterGeoBounds( wi, False, &_loX, &_loY, &_hiX, &_hiY );
		if( wi->win_geo.bounds._loX!= _loX ){
			wi->win_geo.bounds._loX= _loX;
		}
		if( wi->win_geo.bounds._hiX!= _hiX ){
			wi->win_geo.bounds._hiX= _hiX;
		}
		if( wi->win_geo.bounds._loY!= _loY ){
			wi->win_geo.bounds._loY= _loY;
		}
		if( wi->win_geo.bounds._hiY!= _hiY ){
			wi->win_geo.bounds._hiY= _hiY;
		}
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	print_immediate= pim;

	  /* Draw_Process( wi, 1) was here. Moved it to after the TransformCompute().	*/

	if( wi->delete_it== -1 || wi->dev_info.user_state== NULL ){
		ActiveWin= AW;
		level-= 1;
		wi->event_level--;
		wi->drawing= 0;
		if( old_Wname ){
			XFree( old_Wname );
		}
		debugFlag= dbF;
		dbF_cache= dbFc;
		return(0);
	}

/* 	if( !wi->animate && PS_STATE(wi)->Printing!= PS_PRINTING )	*/
	if( PS_STATE(wi)->Printing== X_DISPLAY )
	{
		if( !wi->animate ){
			Set_X( wi, &(wi->dev_info) );
		}
		DBE_Swap( wi, XDBE_info, 0 );
	}

	if( !wi->clipped ){
		SetXClip( wi, 0, 0, wi->dev_info.area_w, wi->dev_info.area_h );
	}
	if( !wi->dont_clear ){
		if( wi->clipped ){
		  extern XRectangle XGClipRegion[];
		  extern int ClipCounter;
		  int i;
			for( i= 0; i< ClipCounter; i++ ){
				wi->dev_info.xg_clear(wi->dev_info.user_state,
					(int) XGClipRegion[i].x, (int) XGClipRegion[i].y,
					(int) XGClipRegion[i].width, (int) XGClipRegion[i].height, 0, 0
				);
			}
		}
		else{
			wi->dev_info.xg_clear(wi->dev_info.user_state, 0, 0,
				wi->dev_info.area_w, wi->dev_info.area_h, 0, 0
			);
		}
	}
	else{
		if( !X_silenced(wi) ){
			wi->dont_clear= False;
			dont_clear= True;
		}
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	if( !wi->animate ){
		xtb_br_redraw( wi->YAv_Sort_frame.win);
		xtb_bt_redraw( wi->close);
		xtb_bt_redraw( wi->hardcopy);
		xtb_bt_redraw( wi->settings);
		xtb_bt_redraw( wi->info);
		xtb_bt_redraw( wi->label);
		xtb_bt_set( wi->ssht_frame.win, wi->silenced, NULL );
	}

/* testrfftw(__FILE__,__LINE__,-1);	*/

	_process_bounds= 1;

	if( wi->xname_placed ){
	 int okX= 1;
		wi->tr_xname_x= wi->xname_x;
		wi->tr_xname_y= wi->xname_y;
		do_transform( wi, "xname_x,y", __DLINE__, "DrawWindow(xname_x,y)", &okX, NULL,
			&wi->tr_xname_x, NULL, NULL, &wi->tr_xname_y,
			NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, (wi->xname_trans)? 0 : -1, 0,
			(wi->xname_trans && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)))
		);
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/
	if( wi->yname_placed ){
	 int okX= 1;
		wi->tr_yname_x= wi->yname_x;
		wi->tr_yname_y= wi->yname_y;
		do_transform( wi, "yname_x,y", __DLINE__, "DrawWindow(yname_x,y)", &okX, NULL,
			&wi->tr_yname_x, NULL, NULL, &wi->tr_yname_y,
			NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, (wi->yname_trans)? 0 : -1, 0,
			(wi->yname_trans && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)))
		);
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	  /* Call to Set_X() was here: */
/* testrfftw(__FILE__,__LINE__,-1);	*/

	if( PS_STATE(wi)->Printing== PS_PRINTING ){
		gsResetTextWidths( wi, False );
	}

      /* Figure out the transformation constants */
    if( TransformCompute(wi, True) ){
	  extern char settings_text[16];
	  Boolean aspect_check= False;

/* testrfftw(__FILE__,__LINE__,-1);	*/
		  /* 20010712: Draw_Before not subject to QuickMode, hence the test against !raw_display && use_transformed
		   \ below commented out.
		   */
		if( *ReDo_DRAW_BEFORE || !( (wi->raw_once== -1 && wi->raw_display) /* || (!wi->raw_display && wi->use_transformed) */ ) ){
			Draw_Process( wi, 1 );
/* testrfftw(__FILE__,__LINE__,-1);	*/
		}
		else{
			if( debugFlag ){
				fprintf( StdErr, "Skipping transformations, using results of previous redraw/rescale\n" );
			}
		}
		if( wi->delete_it== -1 || wi->dev_info.user_state== NULL ){
			ActiveWin= AW;
			level-= 1;
			wi->event_level--;
			wi->drawing= 0;
			if( old_Wname ){
				XFree( old_Wname );
			}
			DBE_Swap( wi, XDBE_info, 1 );
			debugFlag= dbF;
			dbF_cache= dbFc;
			return(0);
		}

		if( cursorFont.font && !wi->animate && wi->settings_frame.mapped ){
		  char watch[2]= { XC_watch, '\0' };
			xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
		}

		Xscale2= wi->Xscale;
		Yscale2= wi->Yscale;

		wi->redraw_val= wi->redraw;
		wi->redraw= 0;
		wi->legend_length= 0;
		do{
/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( Handle_An_Event( wi->event_level, 1, "DrawWindow", wi->window,
					/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
				)
			){
				  /* 20000427: changed to False	*/
				XG_XSync( disp, False );
				ret_code= 0; goto DrawWindow_return;
			}
			if( wi->delete_it== -1 ){
				ret_code= 0; goto DrawWindow_return;
			}
			if( wi->redraw){
				if( PS_STATE(wi)->Printing== PS_PRINTING ){
					gsResetTextWidths( wi, False );
				}

/* testrfftw(__FILE__,__LINE__,-1);	*/
				if( !TransformCompute(wi, True)){
/* testrfftw(__FILE__,__LINE__,-1);	*/
					ret_code= 0; goto DrawWindow_return;
				}
/* testrfftw(__FILE__,__LINE__,-1);	*/
				if( !wi->dont_clear ){
					if( wi->clipped ){
					  extern XRectangle XGClipRegion[];
					  extern int ClipCounter;
					  int i;
						for( i= 0; i< ClipCounter; i++ ){
							wi->dev_info.xg_clear(wi->dev_info.user_state,
								(int) XGClipRegion[i].x, (int) XGClipRegion[i].y,
								(int) XGClipRegion[i].width, (int) XGClipRegion[i].height, 0, 0
							);
						}
					}
					else{
						wi->dev_info.xg_clear(wi->dev_info.user_state, 0, 0,
							wi->dev_info.area_w, wi->dev_info.area_h, 0, 0
						);
					}
/* testrfftw(__FILE__,__LINE__,-1);	*/
				}
				else{
					if( !X_silenced(wi) ){
						wi->dont_clear= False;
						dont_clear= True;
					}
				}
				wi->redraw_val= wi->redraw;
				wi->redraw= 0;
			}
			if( wi->delete_it== -1 ){
				ret_code= 0; goto DrawWindow_return;
			}

/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( !wi->fitting ){
				  /* Draw the axis unit labels,  grid lines,  and grid labels */
				  /* This routine fine-tunes transformation constants: changes
				   * are reflected in wi->redraw
				   */
				if( wi->overwrite_AxGrid ){
					if( wi->axisFlag ){
						strcpy( ps_comment, "axis, grid, labels, establishing area only");
						DrawGridAndAxis(wi, False);
					}
				}
				else{
					strcpy( ps_comment, "axis, grid, labels");
					  /* The first time drawing a window, axis_stuff.DoIt==0. Thus, we
					   \ pass the special value -1 as the doIt argument to DGAX(). In all
					   \ other cases, the DoIt field can be passed without problem.
					   */
					DrawGridAndAxis(wi, (wi->axis_stuff.DoIt)? wi->axis_stuff.DoIt : -1 );
				}
/* testrfftw(__FILE__,__LINE__,-1);	*/
			}
			if( wi->delete_it== -1 ){
				ret_code= 0; goto DrawWindow_return;
			}

			if( !wi->redraw && !wi->fitting ){
				if( !wi->no_title){
					  /* Draw the title */
					strncpy( ps_comment, "title", sizeof(ps_comment)/sizeof(char) );
					DrawTitle(wi, 1);
				}
				else{
					if( titleText && *titleText ){
						strncpy( ps_comment, titleText, sizeof(ps_comment)/sizeof(char) );
					}
					DrawTitle(wi, -1);
				}
/* testrfftw(__FILE__,__LINE__,-1);	*/
				if( !wi->overwrite_legend ){
					  /* Draw the legend */
					strcpy( ps_comment, "legend &/ labels");
					DrawLegend(wi, True, NULL);
				}
/* testrfftw(__FILE__,__LINE__,-1);	*/
			}
			if( wi->delete_it== -1 ){
				ret_code= 0; goto DrawWindow_return;
			}
			if( wi->redraw ){
				strcpy( ps_comment, "DrawGridAndAxis() decided better fit to paper is possible - trying again");
				if( debugFlag ){
					fprintf( StdErr, "%s\n", ps_comment );
				}
			}

			if( PS_STATE(wi)->Printing!= PS_PRINTING && !aspect_check && (wi->aspect_ratio || NaN(wi->aspect_ratio)) &&
				!X_silenced(wi) &&
				wi->fit_xbounds>= 0 && wi->fit_ybounds>= 0 && !wi->fitting &&
				wi->event_level< 2
			){
			  int xrange= wi->XOppX- wi->XOrgX,
					yrange= wi->XOppY - wi->XOrgY;
			  double caspect;
				if( NaN(wi->aspect_ratio) ){
					caspect= fabs( (wi->win_geo.bounds._hiX - wi->win_geo.bounds._loX) /
									(wi->win_geo.bounds._hiY - wi->win_geo.bounds._loY)
								);
				}
				else{
					caspect= fabs(wi->aspect_ratio);
				}
/* testrfftw(__FILE__,__LINE__,-1);	*/
				if( dcmp( (double)xrange/ (double)yrange, caspect, win_aspect_precision) ){
					if( wi->dev_info.resized> 0 ){
						wi->dev_info.resized= 0;
					}
					ZoomWindow_PS_Size( wi, 1, wi->aspect_ratio, 0 );
/* testrfftw(__FILE__,__LINE__,-1);	*/
					aspect_check= True;
				}
			}
			if( wi->delete_it== -1 ){
				ret_code= 0; goto DrawWindow_return;
			}

		} while( wi->redraw && wi->redraw!= -2 );
		if( wi->silenced ){
			wi->silenced= -1;
		}

/* testrfftw(__FILE__,__LINE__,-1);	*/
		set_HO_printit_win();
/* testrfftw(__FILE__,__LINE__,-1);	*/

#ifdef TOOLBOX
		if( settings_immediate && !wi->fitting ){
			settings_immediate= 0;
			wi->pw_placing= PW_MOUSE;
			DoSettings( wi->window, wi);
		}
		if( print_immediate && !wi->fitting ){
			if( wi->draw_count== 0 && (DumpProcessed || wi->DumpProcessed) ){
			  /* Make sure that the processes are evaluated..	*/
			  int pim2= print_immediate;
			  int nl= wi->no_legend, nt= wi->no_title;
				print_immediate= 0;
				if( debugFlag){
					fprintf( StdErr, "DrawWindow(0x%lx): printing window 0x%lx - evaluating processes\n",
						(unsigned long)wi, (unsigned long)wi->window
					);
					fflush( StdErr);
				}
				if( PrepareAll ){
					wi->no_legend= 1;
					wi->no_title= 1;
					ShowAllSets( wi );
				}
				DrawWindow( wi );
				if( wi->delete_it== -1 ){
					ret_code= 0; goto DrawWindow_return;
				}
				if( PrepareAll ){
					wi->no_legend= nl;
					wi->no_title= nt;
					ShowAllSets( wi );
				}
				print_immediate= pim2;
			}
			if( !printing){
			  Window win;
			  LocalWin *lwin= wi;

				printing= 1;
				win= wi->window;
				if( debugFlag){
					fprintf( StdErr, "DrawWindow(0x%lx): printing window 0x%lx\n",
						(unsigned long)lwin, (unsigned long)win
					);
					fflush( StdErr);
				}
				lwin->pw_placing= PW_PARENT;
				PrintWindow( win, lwin);
				printing= 0;
			}
			else{
				/* Draw the data sets themselves */
				strcpy( ps_comment, "the data");
				DrawData(wi, False);
#ifdef WHY
				if( !wi->redraw /* && !wi->no_legend */ ){
					  /* Draw the intensity legend */
					strcpy( ps_comment, "Intensity legend &/ labels");
					DrawIntensityLegend(wi, True);
				}
#endif
				if( wi->overwrite_AxGrid ){
					if( !wi->redraw ){
					  /* Draw the axes and grid/ticks */
						strcpy( ps_comment, "axis, grid, labels, now drawing");
						DrawGridAndAxis(wi, True);
					}
				}
				if( !wi->redraw /* && !wi->no_legend */ && !wi->fitting ){
					  /* Draw the intensity legend */
					strcpy( ps_comment, "Intensity legend &/ labels");
					DrawIntensityLegend(wi, True);
				}
				if( wi->overwrite_legend ){
					if( !wi->redraw /* && !wi->no_legend */ ){
						  /* Draw the legend */
						strcpy( ps_comment, "legend &/ labels");
						DrawLegend(wi, True, NULL);
					}
				}
			}
			  /* Don't touch print_immediate - set its backup-value pim to
			   \ its (new) value instead..
			   */
			pim= print_immediate;
			wi->redraw= 0;
			wi->redrawn+= 1;
			ps_comment[0]= '\0';
			ret_code= 0; goto DrawWindow_return;
		}
#endif

		if( wi->delete_it== -1 ){
			ret_code= 0; goto DrawWindow_return;
		}

/* testrfftw(__FILE__,__LINE__,-1);	*/
		if( !PrintingWindow){
			if( Handle_An_Event( wi->event_level, 1, "DrawWindow", wi->window,
					/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
				)
			){
				  /* 20000427: changed to False	*/
				XG_XSync( disp, False );
				ret_code= 0; goto DrawWindow_return;
			}
			if( wi->delete_it ){
				ret_code= 0; goto DrawWindow_return;
			}
			if( wi->redraw ){
				ret_code= 0; goto DrawWindow_return;
			}
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/

		if( !wi->clipped && dont_clear ){
		  int i;
		  DataSet *this_set;
			  /* See if there is a selection of sets that needs (exclusif) redrawing. This is done
			   \ by setting a (multiple) clipmask that contains these sets (and possibly some others..)
			   */
			for( i= 0; i< setNumber; i++ ){
				this_set= &AllSets[i];
				if( this_set->s_bounds_set== -1 && this_set->s_bounds_wi== wi ){
				  /* This is one..	*/
					if( !X_silenced( wi ) ){
					  int margeX= MAX(10, X_ps_MarkSize_X((double) i));
					  int margeY= MAX(10, X_ps_MarkSize_Y((double) i));
						  /* Only perform the clipping for non-silent redraws!	*/
						if( !s_bounds_set ){
							SetXClip( wi, this_set->sx_min- margeX, this_set->sy_min- margeY,
								this_set->sx_max- this_set->sx_min+ margeX, this_set->sy_max- this_set->sy_min+ margeY
							);
						}
						else{
							AddXClip( wi, this_set->sx_min- margeX, this_set->sy_min- margeY,
								this_set->sx_max- this_set->sx_min+ margeX, this_set->sy_max- this_set->sy_min+ margeY
							);
						}
						if( debugFlag ){
							fprintf( StdErr, "Set #%d drawing clipped to (%d,%d) - (%d,%d) with marge %d,%d\n",
								i,
								this_set->sx_min, this_set->sy_min,
								this_set->sx_max, this_set->sy_max, margeX, margeY
							);
						}
						s_bounds_set+= 1;
					}
					else{
					  /* If we're drawing in silenced mode, make sure the s_bounds_flag
					   \ gets restored to -1 (DrawData does that)
					   */
						this_set->s_bounds_set= -2;
					}
				}
			}
			  /* If all are to be redrawn, restore default situation	*/
			if( s_bounds_set== setNumber ){
				for( i= 0; i< setNumber; i++ ){
					this_set->s_bounds_set= -1;
				}
				SetXClip( wi, 0, 0, wi->dev_info.area_w, wi->dev_info.area_h );
				s_bounds_set= 0;
			}
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/

		  /* Draw the data sets themselves */
		strcpy( ps_comment, "the data");
		DrawData(wi, False);
/* testrfftw(__FILE__,__LINE__,-1);	*/

		if( wi->delete_it!= -1 ){
			if( s_bounds_set ){
			  /* Now we must revert to "no" clipping	*/
				SetXClip( wi, 0, 0, wi->dev_info.area_w, wi->dev_info.area_h );
				s_bounds_set= 0;
			}
/* testrfftw(__FILE__,__LINE__,-1);	*/

			if( wi->overwrite_AxGrid && !wi->fitting ){
				if( !wi->redraw ){
				  /* Draw the axes and grid/ticks */
					strcpy( ps_comment, "axis, grid, labels, now drawing");
					DrawGridAndAxis(wi, True);
				}
			}
/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( !wi->redraw /* && !wi->no_legend */ && !wi->fitting ){
				  /* Draw the intensity legend */
				strcpy( ps_comment, "Intensity legend &/ labels");
				DrawIntensityLegend(wi, True);
			}
/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( wi->overwrite_legend && !wi->fitting ){
				if( !wi->redraw /* && !wi->no_legend */ ){
					  /* Draw the legend */
					strcpy( ps_comment, "legend &/ labels");
					DrawLegend(wi, True, NULL);
				}
			}
/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( !*Really_DRAW_AFTER ){
				Draw_Process( wi, 0 );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			}
			if( wi->delete_it== -1 ){
				ret_code= 0; goto DrawWindow_return;
			}
			wi->draw_count+= 1;
			  /* Timing measure, SetWindowTitle used to be here	*/
			wi->redraw_val= 0;
			wi->redrawn+= 1;
			  /* This information is now stored in wi->legend_placed:	*/
			// 20101019: but resetting the variables here may have more annoying side-effects than
			// letting the respective settings be "sticky"...
#if 0
			legend_placed= 0;
			intensity_legend_placed= 0;
			xname_placed= 0;
			yname_placed= 0;
			yname_vertical= 0;
#else
			legend_placed= wi->legend_placed;
			intensity_legend_placed= wi->IntensityLegend.legend_placed;
			xname_placed= wi->xname_placed;
			yname_placed= wi->yname_placed;
			yname_vertical= wi->yname_vertical;
#endif
			if( cursorFont.font && !wi->animate && wi->settings_frame.mapped ){
			  /* Restore the normal Settings-dialogue button text	*/
				xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), settings_text, (xtb_data) 0);
			}

			if( *Really_DRAW_AFTER && !wi->fitting ){
				Draw_Process( wi, 0 );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			}
		}

		  /* 20040430: Fit_After_Draw() used to be called just below, outside the test
		   \ for successful return from TransformCompute...!
		   */
		Fit_After_Draw(wi, old_Wname );

		  /* 20040430: this is a reasonable place to assume we finished drawing without errors... */
		wi->draw_errors= 0;

    }
	else{
		wi->redraw= 0;
		if( wi->silenced ){
			wi->silenced= -1;
		}
	}

	  /* This code used to be somewhat before:	*/
	if( (!wi->animate && !wi->fitting) || wi->halt ){
			if( X_silenced(wi) ){
				Elapsed_Since( &draw_timer, True );
			}
			else{
				Elapsed_Since(&wi->draw_timer, True);
				  /* A little visual effect that shows we're done	*/
				xtb_bt_swap( wi->close );
				xtb_bt_swap( wi->hardcopy );
				xtb_bt_swap( wi->settings );
				xtb_bt_swap( wi->info );
				xtb_bt_swap( wi->label );
				xtb_bt_swap( wi->close );
				xtb_bt_swap( wi->hardcopy );
				xtb_bt_swap( wi->settings );
				xtb_bt_swap( wi->info );
				xtb_bt_swap( wi->label );
/* 				if( CursorCross && !wi->fitting ){	*/
/* 				  Window dum;	*/
/* 				  int dum2, sx, sy;	*/
/* 				  unsigned int mask_rtn;	*/
/* 					XQueryPointer(disp, wi->window, &dum, &dum, &dum2, &dum2,	*/
/* 						  &sx, &sy, &mask_rtn	*/
/* 					);	*/
/* 					DrawCCross( wi, False, sx, sy, NULL );	*/
/* 				}	*/
			}
			SetWindowTitle( wi, Tot_Time );
			if( old_Wname ){
				XFree( old_Wname );
			}
			XFetchName( disp, wi->window, &old_Wname );
			if( debugFlag ){
				if( X_silenced(wi) ){
					fprintf( StdErr, "DrawWindow(): silenced redraw took %g seconds\n", Tot_Time );
				}
				else{
					fprintf( StdErr, "DrawWindow(): took %g seconds\n", Tot_Time );
				}
				fflush( StdErr );
			}
	}

DrawWindow_return:;
	ps_comment[0]= '\0';
	if( wi->FitOnce ){
		wi->FitOnce= False;
	}
	if( wi->raw_once< 0 ){
		wi->raw_display= wi->raw_val;
		wi->raw_once= 0;
		if( debugFlag ){
			fprintf( StdErr, "Post-rescale raw_once desactivated\n" );
		}
	}

	DBE_Swap( wi, XDBE_info, 1 );

	if( wi->silenced ){
		wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced );
	}
	if( !wi->animate ){
		xtb_bt_set( wi->ssht_frame.win, X_silenced(wi), NULL );
	}

	level-= 1;

	print_immediate= pim;

	if( RemoteConnection ){
		XSync( disp, False );
	}

	if( ChangeWindow ){
	  Cursor curs= (CursorCross)? noCursor : theCursor;
	  /* This window has to be "reincarnated" for some reason. E.g. after
	   \ a bunch of sets has been deleted, this can be necessary because GCs
	   \ have become corrupt. Probably this means that something is freed that
	   \ shouldn't have been.
	   */
		HandleMouse(Window_Name,
					  NULL,
					  ChangeWindow, NULL, &curs
		);
		DelWindow( ChangeWindow->window, ChangeWindow );
		ChangeWindow= NULL;
	}

	LastDrawnWin = wi;
	ActiveWin= AW;

	if( s_bounds_set ){
	  /* Now we must revert to "no" clipping	*/
		SetXClip( wi, 0, 0, wi->dev_info.area_w, wi->dev_info.area_h );
		s_bounds_set= 0;
	}
	if( wi->delete_it!= -1 ){
		wi->init_pass= 0;
		wi->event_level--;
	}

	wi->drawing= 0;
	if( !wi->fitting ){
		wi->sets_reordered= False;
		*ascanf_SetsReordered= wi->sets_reordered;
		if( !wi->animate || wi->halt ){
			CompactAxisValues( wi, &wi->axis_stuff.X );
			CompactAxisValues( wi, &wi->axis_stuff.Y );
			CompactAxisValues( wi, &wi->axis_stuff.I );
		}
	}

	if( check_other_redraw && !wi->fitting ){
		if( level== 0 ){
		  /* We're at the toplevel, now check for and perform delayed redraws	*/
		  LocalWindows *WL= WindowList;
		  LocalWin *lwi;
			check_other_redraw= False;
			if( debugFlag ){
				fprintf( StdErr, "DrawWindow(%02d.%02d.%02d): doing delayed redraws.. ",
					wi->parent_number, wi->pwindow_number, wi->window_number
				);
				fflush( StdErr );
			}
			while( WL ){
				lwi= WL->wi;
				if( lwi->redraw== -2 ){
					if( debugFlag ){
						fprintf( StdErr, "%02d.%02d.%02d .. ",
							lwi->parent_number, lwi->pwindow_number, lwi->window_number
						);
						fflush( StdErr );
					}
					lwi->redraw= 1;
					lwi->clipped= 0;
					{ LocalWindows *WWL= WindowList;
					  LocalWin *llwi;
						sprintf( msgbuf, "Waiting for window %02d.%02d.%02d",
							lwi->parent_number, lwi->pwindow_number, lwi->window_number
						);
						while( WWL ){
							llwi= WWL->wi;
							if( llwi->redraw== -2 ){
								XStoreName( disp, llwi->window, msgbuf );
								llwi->delayed= True;
							}
							WWL= WWL->next;
						}
					}
					RedrawNow( lwi );
					lwi->delayed= False;
				}
				WL= WL->next;
			}
			if( debugFlag ){
				fputc( '\n', StdErr );
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "DrawWindow(%02d.%02d.%02d)-%d there are delayed redraws..\n",
				level,
				wi->parent_number, wi->pwindow_number, wi->window_number
			);
		}
	}

	if( old_Wname ){
		XFree( old_Wname );
	}

/* 	testrfftw(__FILE__,__LINE__,-1);	*/

	debugFlag= dbF;
	dbF_cache= dbFc;

	if( RemoteConnection ){
		XSync( disp, False );
	}

	return(ret_code);
}


extern int titles;

double overlap_legend_tune= -0.5, highlight_par[]= {0.75, 1};
int highlight_mode= 0, highlight_npars= sizeof(highlight_par)/sizeof(highlight_par[0]);

/* parse_code(T) and remove heading and trailing whitespace	*/
extern char *cleanup( char *T );

/* 20001230: function that scans a string (part of the Title text) for opcodes, and parses
 \ them. Only returns non-NULL when some action has been taken; a non-NULL result (string)
 \ must be de-allocated later. The end of the parsed original string is returned in *parsed_end.
 */
char *ParseTitlestringOpcodes( LocalWin *wi, int idx, char *title, char **parsed_end )
{ char *opcode, *stt= title;
  char xcol[16]= "", ycol[16]= "", ecol[16]= "", lcol[16]= "", sbuf[16]= "", rbuf[5]= "",
		*fn= NULL, *sn= NULL, pbuf[16]= "", vbuf[16]= "", *cx= NULL, *cy= NULL, *ce= NULL, *cee= NULL, *cl= NULL, *cN= NULL;
  char opcodes[]= "XYEFNPRSVC[";
  int Nopcodes= 0;
	if( parsed_end ){
		*parsed_end= NULL;
	}
	if( stt && *stt && (opcode= index( stt, '%')) ){
		do{
			if( index( opcodes, opcode[1] ) ){
				Nopcodes+= 1;
			}
			stt= opcode+1;
			opcode= (*stt)? index( stt, '%' ) : NULL;
		 /* Remove the check for Nopcodes in the while() statement below to count all occurrences.
		  \ Now, we just content ourselves with the presence of a single opcode to do the big loop
		  \ below.
		  */
		} while( Nopcodes== 0 && opcode && stt && *stt );
	}
	stt= title;
	if( Nopcodes ){
	  int n= strlen(stt);
	  char *searchvar= NULL;
		if( wi && idx>= 0 && idx< setNumber ){
		  DataSet *set= &AllSets[idx];
			n+= sprintf( xcol, "%d", wi->xcol[idx] );
			n+= sprintf( ycol, "%d", wi->ycol[idx] );
			n+= sprintf( ecol, "%d", wi->ecol[idx] );
			n+= sprintf( lcol, "%d", wi->lcol[idx] );
			n+= sprintf( pbuf, "%d", set->numPoints );
			n+= sprintf( sbuf, "%d", set->set_nr );
			n+= (set->setName)? strlen((sn= set->setName)) : 0;
			n+= (set->fileName)? strlen((fn= set->fileName)) : 0;
			n+= sprintf( vbuf, "%d", wi->numVisible[idx] );
			if( !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val) ||
					set->raw_display) &&
				((set->process.set_process_len && !*disable_SET_PROCESS) || wi->process.data_process_len )
			){
			  /* TRANSFORM_[XY] are not considered transformations of a set..	*/
				n+= sprintf( rbuf, "ripe");
			}
			else{
				n+= sprintf( rbuf, "raw");
			}
			if( (cx= Find_LabelsListLabel( (set->ColumnLabels)? set->ColumnLabels : wi->ColumnLabels, wi->xcol[idx])) ){
				n+= strlen(cx);
			}
			if( (cy= Find_LabelsListLabel( (set->ColumnLabels)? set->ColumnLabels : wi->ColumnLabels, wi->ycol[idx])) ){
				n+= strlen(cy);
			}
			if( (ce= Find_LabelsListLabel( (set->ColumnLabels)? set->ColumnLabels : wi->ColumnLabels, wi->ecol[idx])) ){
				n+= strlen(ce);
				if( set->has_error && set->use_error && wi->error_type[set->set_nr] ){
					if( (cee= (char*) calloc( strlen(ce)+ 16, sizeof(char))) ){
						switch( wi->error_type[set->set_nr] ){
							case 1:
							case 2:
							case 3:
							case EREGION_FLAG:
								n+= sprintf( cee, " #xb1 %s", ce );
								break;
							default:
								n+= sprintf( cee, ", %s", ce );
								break;
						}
					}
				}
			}
			if( (cl= Find_LabelsListLabel( (set->ColumnLabels)? set->ColumnLabels : wi->ColumnLabels, wi->lcol[idx])) ){
				n+= strlen(cl);
			}
			if( (cN= Find_LabelsListLabel( (set->ColumnLabels)? set->ColumnLabels : wi->ColumnLabels, set->Ncol)) ){
				n+= strlen(cN);
			}
		}
		if( (stt= calloc( n+1, sizeof(char))) ){
		  char *s= title, *d= stt, *v;
		  extern char *pvno_found_env_var;
			while( *s ){
				if( *s== '%' ){
					opcode= s;
					s++;
					switch( *s ){
						case 'X':
							v= xcol;
							break;
						case 'Y':
							v= ycol;
							break;
						case 'E':
							v= ecol;
							break;
						case 'L':
							v= lcol;
							break;
						case 'F':
							v= fn;
							break;
						case 'N':
							if( idx>= 0 && idx< setNumber && title== AllSets[idx].setName ){
							  /* disable the %N opcode while parsing setName strings... */
								v= "%N";
							}
							else{
								v= sn;
							}
							break;
						case 'P':
							v= pbuf;
							break;
						case 'R':
							v= rbuf;
							break;
						case 'S':
							v= sbuf;
							break;
						case 'V':
							v= vbuf;
							break;
						case 'C':
							switch( s[1] ){
								case 'X':
									v= cx;
									s++;
									break;
								case 'Y':
									v= cy;
									s++;
									break;
								case 'E':
									v= ce;
									s++;
									break;
								case 'e':
									v= cee;
									s++;
									break;
								case 'L':
								case 'V':
									v= cl;
									s++;
									break;
								case 'N':
									v= cN;
									s++;
									break;
								default:
									v= "%C";
									break;
							}
							break;
						case '[':
							opcode= s;
							if( !(searchvar= parse_varname_opcode( &opcode[1], ']', NULL, True )) ){
/* 								searchvar= pvno_found_env_var;	*/
								xfree( pvno_found_env_var );
							}
							if( searchvar ){
							  /* We found a variable, and it is now printed in searchvar..	*/
							  char *new;
							  int i, here= ((unsigned long) d - (unsigned long) stt);
								n+= strlen( searchvar );
								  /* Reallocate the target title buffer	*/
								if( (new= calloc( n+1, sizeof(char))) ){
									d= new;
									  /* Copy up to where we were	*/
									for( i= 0; i< here; i++ ){
										*d++= stt[i];
									}
									xfree( stt );
									stt= new;
								}
								else{
								  /* Failure: just forget about it.	*/
									n-= strlen(searchvar);
									xfree( searchvar );
								}
								v= searchvar;
								break;
							}
							else{
							  /* No break; fall through to default case. */
							}
						default:
						  /* invalid opcode: just copy it..	*/
							v= NULL;
							*d++ = '%';
							*d++ = *s;
							break;
					}
					while( v && *v ){
						*d++ = *v++;
					}
					if( searchvar && *s== '[' ){
					  /* This was the %[name] opcode. Skip over it.	*/
						s= index( s, ']' );
						  /* And de-allocate the memory it occupies	*/
						xfree( searchvar );
					}
					s++;
				}
				else{
					*d++ = *s++;
				}
			}
			*d= '\0';
			  /* 20040911: */
			parse_codes(stt);
			if( parsed_end ){
				*parsed_end= d;
			}
		}
	}
	else{
	  /* Return NULL if we didn't (need to) do anything!	*/
		stt= NULL;
	}
	  /* free any memory explicitely allocated: */
	xfree(cee);
	return( stt );
}

int DrawTitle( LocalWin *wi, int draw)
/*
 * This routine draws the title of the graph centered in
 * the window.  It is spaced down from the top by an amount
 * specified by the constant PADDING.  The font need not be
 * fixed width.  The routine returns the height of the
 * title in pixels. When the <draw> argument is False, only the
 \ total length of all titles to be drawn is determined (titles are
 \ drawn only once). When True, titles are drawn, and copied in the
 \ wi->graph_titles buffer, which is used when the user Control-Clicks
 \ in the title region.
 */
{ int y= wi->dev_info.axis_pad, i, j= 0, cnr, idx, graph_titles_length= wi->graph_titles_length;
  char *prev_title= "\0";
  char *stt= NULL;
  Boolean free_stt= False, free_prev_title= False;
#ifdef DEBUG
	if( !wi ){
		fprintf( StdErr, "DrawTitle() called with NULL argument\n" );
		return(0);
	}
#endif
	if( wi->delete_it== -1 ){
		return(0);
	}
	TitleMessage( wi, "Title..." );
	if( !draw ){
	  /* determine number of titles to be drawn	*/
		titles= 0;
		wi->graph_titles_length= 0;
	}
	else if( wi->legend_always_visible ){
		  /* ystart= axis_pad+ wi->XOrgY- wi->XOrgY[default]:
		   \ compensate for rescaling to accomodate legendbox depassing
		   \ the data-region.
		   */
		y= wi->dev_info.axis_pad+ wi->XOrgY-
			(2*wi->dev_info.bdr_pad+ titles* wi->dev_info.title_height+ wi->dev_info.label_height/2);
		if( !wi->yname_placed && !wi->yname_vertical ){
			y-= wi->dev_info.label_height+ wi->dev_info.bdr_pad;
		}
		  /* In case of 100% vertical overlap of title and legendbox,
		   \ move the title upwards.
		   */
		if( wi->legend_uly<= y && wi->legend_lry>= y+ titles* wi->dev_info.title_height ){
			y= wi->legend_uly- wi->dev_info.axis_pad;
		}
	}
	wi->title_uly= wi->title_lry= -1;
	if( wi->graph_titles ){
		wi->graph_titles[0]= '\0';
	}
    if( (stt= titleText2) && strlen(stt) ){
		if( draw ){
			if( wi->title_uly== -1 ){
				wi->title_uly= y;
			}
			if( draw> 0 ){
			  char *ntt, *parsed_end= NULL;
				if( (ntt= ParseTitlestringOpcodes( wi, wi->first_drawn, titleText2, &parsed_end )) ){
					stt= ntt;
					free_stt= True;
					if( stt[0]== '`' && parsed_end[-1]== '`' ){
						parsed_end[-1]= '\0';
						xfree( titleText2 );
						titleText2= strdup( &stt[1] );
						xfree( stt );
						stt= titleText2;
						free_stt= False;
					}
				}
				xg_text_highlight= False;
				wi->dev_info.xg_text(wi->dev_info.user_state,
/* 						 wi->dev_info.area_w/2, y,	*/
						 wi->XOrgX + (wi->XOppX - wi->XOrgX)/2, y,
						 stt,
						 T_TOP, T_TITLE, NULL
				);
				if( debugFlag ){
					fprintf( stderr, "DrawTitle(%s) (-t title)\n", titleText2 );
				}
			}
			y+= wi->dev_info.title_height;
			wi->title_lry= y;
			if( wi->graph_titles ){
				strncat( wi->graph_titles, titleText2, wi->graph_titles_length- strlen(wi->graph_titles)- 1 );
				strcat( wi->graph_titles, "\n");
				if( debugFlag ){
					fprintf( stderr, "drawtitle(): added (-t title) - length is %d of %d\n",
						strlen( wi->graph_titles), wi->graph_titles_length
					);
				}
			}
		}
		else{
			titles+= 1;
			cleanup( titleText2 );
			wi->graph_titles_length+= strlen(titleText2) + 2;
			if( debugFlag ){
				fprintf( stderr, "drawtitle(%s): (-t title) length %d - total length %d\n",
					titleText2,
					strlen(titleText2), wi->graph_titles_length
				);
			}
		}
		prev_title= stt;
		free_prev_title= free_stt;
		stt= NULL;
	}
	free_stt= False;
    for( cnr= i= 0; cnr< setNumber && !(draw && i>= titles); cnr++ ){
	  Boolean different;
		idx= DO(cnr);
		if( free_stt ){
			xfree( stt );
		}
		stt= AllSets[idx].titleText;
		free_stt= False;
		if( stt ){
			  /* Compare previous and current title now, before (maybe) freeing
			   \ a parsed title of the previous set.
			   */
			different= (AllTitles || DrawColouredSetTitles)? True : strcmp( stt, prev_title );
			if( !draw && AllSets[idx].titleChanged ){
				cleanup( stt );
			}
			if( draw_set(wi, idx) ){
			  char *ntt, *parsed_end= NULL;
				j+= 1;
				if( (ntt= ParseTitlestringOpcodes( wi, idx, AllSets[idx].titleText, &parsed_end )) ){
					stt= ntt;
					free_stt= True;
					if( stt[0]== '`' && parsed_end[-1]== '`' ){
						parsed_end[-1]= '\0';
						xfree( AllSets[idx].titleText );
						AllSets[idx].titleText= strdup( &stt[1] );
						xfree( stt );
						stt= AllSets[idx].titleText;
						free_stt= False;
					}
				}
			}
			if( draw_set(wi, idx) &&
				different &&
				  /* 990416: a hacked-in support for not having titles
				   \ at all sets... still, by default, adding a title to
				   \ one set will set that title on all following sets without.
				   \ this should probably change...
				   */
				strcmp( stt, " ") &&
				stt[0]
			){
				if( draw ){
				  Pixel color0= AllAttrs[0].pixelValue;
				  char *tt= stt;
				  int n, hidx= idx, hl= (wi->legend_line[idx].highlight)? 1 : 0,
						x= wi->XOrgX + (wi->XOppX - wi->XOrgX)/2;
					  /* 20020324: I do not see quickly what this loop is supposed to do, and how it
					   \ might have to be modified to support AllTitles drawing... Let's try to
					   \ just not execute it when AllTitles==True.
					   */
					if( !AllTitles ){
						for( n= idx+1; n< setNumber && !strcmp( tt, AllSets[n].titleText); n++ ){
							if( wi->legend_line[n].highlight ){
								if( !hl ){
									hidx= n;
								}
								hl+= 1;
							}
						}
						n-= idx+1;
					}
					else{
						n= 0;
					}
					if( wi->title_uly== -1 ){
						wi->title_uly= y;
					}

					if( draw> 0 ){
						if( hl && highlight_par[1] ){
							textPixel= (AllSets[hidx].pixvalue< 0)? AllSets[hidx].pixelValue :
								AllAttrs[AllSets[hidx].pixvalue].pixelValue;
							use_textPixel= 1;
							AllAttrs[0].pixelValue=
								(wi->legend_line[hidx].pixvalue< 0)? wi->legend_line[hidx].pixelValue : highlightPixel;
							xg_text_highlight= True;
							xg_text_highlight_colour= 0;
						}
						else if( DrawColouredSetTitles ){
							textPixel= (AllSets[hidx].pixvalue< 0)? AllSets[hidx].pixelValue :
								AllAttrs[AllSets[hidx].pixvalue].pixelValue;
							use_textPixel= 1;
						}
						wi->dev_info.xg_text(wi->dev_info.user_state,
/* 								 wi->dev_info.area_w/2, y,	*/
								 x, y,
								 tt,
								 T_TOP, T_TITLE, NULL
						);
						AllAttrs[0].pixelValue= color0;
						  /* we currently take use_textPixel as a once-only switch, so we don't
						   \ save/restore its state!
						   */
						use_textPixel= 0;
					}

					y+= wi->dev_info.title_height;
					wi->title_lry= y;
					if( debugFlag ){
						fprintf( StdErr, "DrawTitle(%s)\n", tt );
						fflush( StdErr );
					}
					if( wi->graph_titles ){
						strncat( wi->graph_titles, tt,
							wi->graph_titles_length- strlen(wi->graph_titles)- 1
						);
						strcat( wi->graph_titles, "\n");
						if( debugFlag ){
							fprintf( StdErr, "DrawTitle(): added title #%d, set #%d - length is %d of %d\n",
								i+1, idx,
								strlen( wi->graph_titles), wi->graph_titles_length
							);
						}
					}
				}
				else{
					  /* 990706: since we now allow empty strings for empty titles, we must no longer
					   \ exclude those from the title-count!
					   */
					if( 1 /* strlen(stt) */ ){
						titles+= 1;
						wi->graph_titles_length+= strlen(stt) + 2;
						if( debugFlag ){
							fprintf( StdErr, "DrawTitle(%s): title #%d, set #%d length %d - total length %d\n",
								stt, titles, idx,
								strlen(stt), wi->graph_titles_length
							);
						}
					}
				}
				i+= 1;
				if( free_prev_title ){
					xfree( prev_title );
				}
				prev_title= stt;
				free_prev_title= free_stt;
			}
			if( AllSets[idx].titleChanged && !AllTitles && !strcmp( titleText, stt ) ){
			  /* This title might have been drawn (or maybe not)
			   \ so disable the global title
			   */
				if( titleText[0] ){
					titleText_0= titleText[0];
				}
				titleText[0]= '\0';
			}
			AllSets[idx].titleChanged= 0;
			if( prev_title== stt ){
				  /* Make sure we're not going to xfree(stt) in the next loop! */
				stt= NULL;
			}
		}
		else if( !draw && idx ){
		  /* 950710:
		   \ NULL pointer as titleText. Point titleText to the previous
		   \ set's titleText. Thus all sets following a title-setting
		   \ up to the next titlesetting inherit the same titleText. Note
		   \ that we do not (longer) increment the titles counter, since
		   \ this new title is a copy of one already considered.
		   */
		  /* 990429: disabled this feature
			if( (AllSets[idx].titleText= XGstrdup(AllSets[idx-1].titleText)) ){
				AllSets[idx].titleChanged= 1;
				i+= 1;
			}
		     instead "clearing" the prev_title feature:
		   */
			i+= 1;
			if( free_prev_title ){
				xfree( prev_title );
			}
			prev_title= "";
			free_prev_title= False;
		}
	}
	  /* 20020507: changed j into j==0 to implement the remark below ! */
	if( j==0 && i== 0 && titleText[0]== '\0' && titleText[1]!= '\0' && titleTextSet ){
	  /* no titles drawn, restore global title	*/
	      /* 20020507: and this is also how titleText2 behaves! */
		titleText[0]= titleText_0;
	}
	  /* RJB 980518 */
	else if( titleText[0] && !titleTextSet ){
		titleText_0= titleText[0];
		titleText[0]= '\0';
	}
    if( strlen(titleText) ){
		free_stt= False;
		stt= titleText;
		if( draw ){
			if( wi->title_uly== -1 ){
				wi->title_uly= y;
			}
			if( draw> 0 ){
			  char *ntt, *parsed_end= NULL;
				if( (ntt= ParseTitlestringOpcodes( wi, wi->last_drawn, titleText, &parsed_end )) ){
					stt= ntt;
					free_stt= True;
					if( stt[0]== '`' && parsed_end[-1]== '`' ){
						parsed_end[-1]= '\0';
						strncpy( titleText, &stt[1], MAXBUFSIZE );
						xfree( stt );
						stt= titleText;
						free_stt= False;
					}
				}
				wi->dev_info.xg_text(wi->dev_info.user_state,
/* 						 wi->dev_info.area_w/2, y,	*/
						 wi->XOrgX + (wi->XOppX - wi->XOrgX)/2, y,
						 stt,
						 T_TOP, T_TITLE, NULL
				);
			}
			y+= wi->dev_info.title_height;
			wi->title_lry= y;
			if( debugFlag ){
				fprintf( StdErr, "DrawTitle(%s) (-t title)\n", titleText );
			}
			if( wi->graph_titles ){
				strncat( wi->graph_titles, titleText, wi->graph_titles_length- strlen(wi->graph_titles)- 1 );
				strcat( wi->graph_titles, "\n");
				if( debugFlag ){
					fprintf( StdErr, "DrawTitle(): added (-t title) - length is %d of %d\n",
						strlen( wi->graph_titles), wi->graph_titles_length
					);
				}
			}
		}
		else{
			titles+= 1;
			cleanup( titleText );
			wi->graph_titles_length+= strlen(titleText) + 2;
			if( debugFlag ){
				fprintf( StdErr, "DrawTitle(%s): (-t title) length %d - total length %d\n",
					titleText,
					strlen(titleText), wi->graph_titles_length
				);
			}
		}
	}
	if( !draw ){
		if( debugFlag ){
			fprintf( StdErr, "DrawTitle(): %d titles (total length: %d)\n", titles, wi->graph_titles_length );
			fflush( StdErr );
		}
		  /* Some extra buffer space	*/
		wi->graph_titles_length+= 8* setNumber + 16;
		if( !wi->graph_titles || wi->graph_titles_length> graph_titles_length ){
			if( debugFlag ){
				fprintf( StdErr, "DrawTitle(): discarding old wi->graph_titles[%d] 0x%lx: allocating new of size %d\n",
					graph_titles_length, wi->graph_titles, wi->graph_titles_length
				);
				fflush( StdErr );
			}
			xfree( wi->graph_titles );
			if( wi->graph_titles_length ){
				wi->graph_titles= calloc( wi->graph_titles_length, sizeof(char) );
			}
			else{
				wi->graph_titles= NULL;
			}
		}
		else if( graph_titles_length> wi->graph_titles_length ){
		  /* the already allocated size was larger than the newly acquired size. No need to update
		   \ the length indicator, as that would cause an unnecessary reallocation the next time the full
		   \ length is required.
		   */
			wi->graph_titles_length= graph_titles_length;
		}
	}
	if( free_prev_title ){
		xfree(prev_title);
	}
	if( free_stt ){
		xfree( stt );
	}
	TitleMessage( wi, NULL);
	return(1);
}

double (*RoundUp_log)(LocalWin *wi, double x);

#define Check_Inf(x)	if( Inf(x)== -1 ){ \
	x= -DBL_MAX; \
} \
else if( Inf(x)== 1 ){ \
	x= DBL_MAX; \
}

char *xgraph_NameBuf= NULL;
int xgraph_NameBufLen= 0;

  /* If set, we use the geometric information stored in HO_PreviousWin_ptr. Presently, this only
   \ applies to [XY]O[rp][gp][XY].
   */
int Use_HO_Previous_TC= False;

/* Shrink the plotting area such that a rectangle at realworl co-ordinates (ulX,ulY) and
 \ with screen-co-ordinates (ulx,uly,lrx,lry) will fit in the window.
 */
int CShrinkArea( LocalWin *wi, /* double _loX, double _loY, double _hiX, double _hiY,	*/
	double ulX, double ulY, int ulx, int uly, int lrx, int lry,
	int uhptc /* a local copy of Use_HO_Previous_TC, we can be different from the global flag */
)
{ int new, redo= 0;
  double ref, R;
	if( !uhptc ){
	  double _loX= wi->UsrOrgX, _loY= wi->UsrOrgY, _hiX= wi->UsrOppX, _hiY= wi->UsrOppY;
		if( lrx> (ref= wi->dev_info.area_w - 2* wi->dev_info.bdr_pad) ){
			R= _hiX- _loX;
			new= (R* (ref- (lrx- ulx)- wi->XOrgX))/ (ulX- wi->UsrOrgX)+ wi->XOrgX;
			if( new> wi->XOrgX && new< ref ){
				wi->XOppX= new;
				redo= True;
			}
		}
		if( ulx< wi->dev_info.bdr_pad ){
			new= wi->XOrgX+ (wi->dev_info.bdr_pad- ulx);
			if( new> 0 && new< wi->XOppX ){
				wi->XOrgX= new;
				redo= True;
			}
		}
		if( lry> (ref= wi->dev_info.area_h - wi->dev_info.bdr_pad) ){
			R= _hiY- _loY;
			new= ((ref- (lry- uly))* R- (ulY- wi->UsrOrgY)* wi->XOrgY) /
				(R- (ulY- wi->UsrOrgY));
			if( new> wi->XOrgY && new< ref ){
				wi->XOppY= new;
				redo= True;
			}
		}
		if( uly< (2* wi->dev_info.bdr_pad+ wi->dev_info.label_height/2) ){
			new= wi->XOrgY+ ((2* wi->dev_info.bdr_pad+ wi->dev_info.label_height/2)- uly);
			if( new> 0 && new< wi->XOppY ){
				wi->XOrgY= new;
				redo= True;
			}
		}
	}
	return( redo );
}

double HL_WIDTH(double w)
{
	switch( highlight_mode ){
		case 0:
		default:
			return(5* pow(MAX(1,(w)), highlight_par[0]));
			break;
		case 1:
			return( (w+ highlight_par[0]) );
			break;
	}
}

void MarkerSizes( LocalWin *wi, int idx, int ps_mark_scale, int *mw, int *mh )
{
	if( AllSets[idx].markFlag && !NaNorInf(AllSets[idx].markSize) ){
		if( AllSets[idx].markSize< 0 ){
			*mw= SCREENXDIM( wi, -AllSets[idx].markSize );
			*mh= SCREENYDIM( wi, -AllSets[idx].markSize );
		}
		else{
/* 			*mw= *mh= AllSets[idx].markSize* ps_mark_scale;	*/
			*mw= *mh= AllSets[idx].markSize* wi->dev_info.mark_size_factor;
		}
	}
	else{
/* 		*mw= *mh= ps_mark_scale * (psm_base + (int)(idx/internal_psMarkers)* psm_incr);	*/
		*mw= *mh= (psm_base + (int)(idx/internal_psMarkers)* psm_incr)* wi->dev_info.mark_size_factor;
	}
}

int TransformCompute( LocalWin *wi, Boolean warn_about_size )
/*
 * This routine figures out how to draw the axis labels and grid lines.
 * Both linear and logarithmic axes are supported.  Axis labels are
 * drawn in engineering notation.  The power of the axes are labeled
 * in the normal axis labeling spots.  The routine also figures
 * out the necessary transformation information for the display
 * of the points (it touches _XOrgX, _XOrgY, _UsrOrgX, _UsrOrgY, and
 * UnitsPerPixel).
 */
{
#ifdef REDUNDANT
    double bbCenX, bbCenY, bbHalfWidth, bbHalfHeight;
#endif
	double _loX, _loY, _hiX, _hiY;
	extern int XFontWidth();
    int polarType, okX, okY, idx, maxFontWidth= XFontWidth(legendFont.font), maxName, maxFile, overlap_size, legendWidth;
	double maxlineWidth= 0;
	int legend_lrx, legend_frx, l_placed = legend_placed;
	double markerWidth= 0, first_markerHeight= 0;
    ALLOCA( err, char, LMAXBUFSIZE+2, err_len);
	static char active= 0, XSw= 0, YSw= 0;
	int redo= 0, looped= 0, showlines= 0;
	int axis_width, axis_height;
	int asn= (int) *ascanf_setNumber;
	int uhptc, no_xname= False, no_yname= False;

#ifdef DEBUG
	if( !wi ){
		fprintf( StdErr, "TransformCompute() called with NULL argument\n" );
		return(0);
	}
#endif

	if( wi->silenced && !wi->init_pass ){
		return(1);
	}

/* testrfftw(__FILE__,__LINE__,-1);	*/
	axis_width= wi->dev_info.axis_width;
	if( wi->ValCat_YFont && wi->ValCat_Y && wi->ValCat_Y_axis ){
	  /* axis_width used to determine offset of Y-axis => depends on width of Y-axis font.	*/
		axis_width= MAX( axis_width, (*wi->dev_info.xg_CustomFont_width)( wi->ValCat_YFont ) );
	}
	axis_height= wi->dev_info.axis_height;
	if( wi->ValCat_XFont && wi->ValCat_X && wi->ValCat_X_axis ){
	  /* axis_height used to determine vertical position of X-axis => depends on X-axis font.	*/
		axis_height= MAX( axis_height, (*wi->dev_info.xg_CustomFont_height)( wi->ValCat_XFont ) );
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	  /* temporarily set ascanf_setNumber to -1: set-specific processing shouldn't
	   \ affect bounds or log_zero_ or...
	   */
	*ascanf_setNumber= -1;
	*ascanf_numPoints= 0;

	XSw= 0;
	YSw= 0;
	TitleMessage( wi, "Figuring out geometry and things" );
/* testrfftw(__FILE__,__LINE__,-1);	*/
/* 	if( !active && !wi->transform_axes && (wi->transform.x_len || wi->transform.y_len) &&	*/
	if( !active && !wi->transform_axes &&
		!((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val))
	){
	 /* Determine the bounding box excluding only the TRANSFORM_[XY] operations.
	  \ This is done in the easiest way: calling ourselves again, with a flag set
	  \ indicating that boundary values without TRANSFORM_[XY] should be used.
	  \ The static char active ensures we don't land an endless loop here...
	  \ NOTE: this does not work when logFlag or powFlag is set for an axis - i.o.w.
	  \ a log/sqrt axis with a TRANSFORM_? cannot show its "true numbers" (log/sqrt
	  \ transformation should be last, but won't be for the ranges).
	  \ 990409: This means we temporarily store R_U..O..[XY] in the win_geo.bounds
	  \ structure. It is the responsability of the Fit_..Bounds routines to ensure
	  \ that those R_U... values are correct (all but TRANSFORM_.. processings).
	  \ We call ourselves with transform_axes=1, because the bounds we pass are
	  \ TRANSFORM_.. "raw", and thus the axes can show these values!
	  */
	  char *xsw= "", *ysw= "";
		active= 1;
		wi->transform_axes= 1;
		_loX= wi->loX;
		_loY= wi->loY;
		_hiX= wi->hiX;
		_hiY= wi->hiY;
		wi->loX= wi->R_UsrOrgX;
		wi->loY= wi->R_UsrOrgY;
		wi->hiX= wi->R_UsrOppX;
		wi->hiY= wi->R_UsrOppY;
/* testrfftw(__FILE__,__LINE__,-1);	*/
		TransformCompute( wi, warn_about_size );
/* testrfftw(__FILE__,__LINE__,-1);	*/
		  /* We now have determined everything for the case without TRANSFORM_.. processing;
		   \ Restore the values
		   */
		wi->transform_axes= 0;
		wi->loX= _loX;
		wi->loY= _loY;
		wi->hiX= _hiX;
		wi->hiY= _hiY;
		active= 0;
		  /* Determine if TRANSFORM_.. swap the ranges:	*/
		{
		  double xl= wi->R_UsrOrgX, xh= wi->R_UsrOppX, lx= xl, hx= xl, y= wi->R_UsrOrgY;
			do_TRANSFORM( wi, 0, 1, 1, &xl, NULL, NULL, &y, &lx, &hx, 1, False );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			y= wi->R_UsrOppY;
			do_TRANSFORM( wi, 0, 1, 1, &xh, NULL, NULL, &y, &lx, &hx, 1, False );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( xl> xh ){
			  /* Swap the X bounds	*/
				y= wi->R_UsrOppX;
				wi->R_UsrOppX= wi->R_UsrOrgX;
				wi->R_UsrOrgX= y;
				wi->R_XUnitsPerPixel*= -1;
				xsw= " (X swapped)";
				XSw= 1;
			}
			else{
				XSw= 0;
			}
		}
		{
		  double x= wi->R_UsrOrgX, yl= wi->R_UsrOrgY, yh= wi->R_UsrOppY, ly= yl, hy= yl;
			do_TRANSFORM( wi, 0, 1, 1, &x, NULL, NULL, &yl, &ly, &hy, 1, False );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			x= wi->R_UsrOppX;
			do_TRANSFORM( wi, 0, 1, 1, &x, NULL, NULL, &yh, &ly, &hy, 1, False );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( yl> yh ){
			  /* Swap the Y bounds	*/
				x= wi->R_UsrOppY;
				wi->R_UsrOppY= wi->R_UsrOrgY;
				wi->R_UsrOrgY= x;
				wi->R_YUnitsPerPixel*= -1;
				ysw= " (Y swapped)";
				YSw= 1;
			}
			else{
				YSw= 0;
			}
		}
		if( debugFlag ){
			fprintf( StdErr, "TransformCompute(): boundingbox excluding only TRANSFORM_[XY]: (%g,%g) - (%g,%g)%s%s\n",
				wi->R_UsrOrgX, wi->R_UsrOrgY, wi->R_UsrOppX, wi->R_UsrOppY,
				xsw, ysw
			);
			fflush( StdErr );
		}
	}
	if( wi->polarFlag ){
		wi->logXFlag= 0;
		wi->sqrtXFlag= 0;
		if( wi->_Xscale< 0 ){
			wi->Xscale= ABS(wi->Xscale);
			XscaleR= wi->radix / Trans_X( wi, wi->radix );
		}
		else{
			XscaleR= 1.0;
		}
		if( debugFlag ){
			fprintf( StdErr, "TransformCompute(): Xscale=%g XscaleR= %g\n", wi->Xscale, XscaleR );
			fflush( StdErr );
		}
		Gonio_Base( wi, wi->radix, wi->radix_offset );
		if( WINSIDE( wi, 0, 0.25) ){
			wi->win_geo.polar_axis= X_axis;
			polarType= 1;
			_loX= wi->radix/ 4.0;
			  /* 981120: or maybe this?? But I think the offset shouldn't
			   \ change the range(s), and is/should be accounted for at
			   \ the final stage..!
			_loX= (wi->radix+wi->radix_offset)/ 4.0;
			   */
			_hiX= 0;
		}
		else if( WINSIDE( wi, 0.25, 0.5) ){
			wi->win_geo.polar_axis= Y_axis;
			polarType= 2;
			_loX= wi->radix/ 2.0;
			_hiX= wi->radix/ 4.0;
		}
		else if( WINSIDE( wi, 0, 0.5) ){
			wi->win_geo.polar_axis= X_axis;
			polarType= 3;
			_loX= wi->radix/ 2.0;
			_hiX= wi->radix/ 4.0;
		}
		else if( WINSIDE( wi, 0.5, 0.75 ) ){
			wi->win_geo.polar_axis= X_axis;
			polarType= 5;
			_loX= wi->radix/ 2.0;
			_hiX= wi->radix* 0.25;
		}
		else if( WINSIDE( wi, 0.25, 0.75) ){
			wi->win_geo.polar_axis= Y_axis;
			polarType= 4;
			_loX= wi->radix/ 2.0;
			_hiX= wi->radix/ 4.0;
		}
		else if( WINSIDE( wi, 0.75, 1) ){
			wi->win_geo.polar_axis= Y_axis;
			polarType= 7;
			_loX= wi->radix* 0.25;
			_hiX= 0;
		}
		else if( WINSIDE( wi, 0.5, 1) ){
			wi->win_geo.polar_axis= X_axis;
			polarType= 6;
			_loX= wi->radix/ 2.0;
			_hiX= wi->radix* 0.25;
		}
		else if( WINSIDE( wi, 0.75, 0.25) ){
			wi->win_geo.polar_axis= Y_axis;
			polarType= 8;
			_loX= wi->radix/ 4.0;
			_hiX= 0.0;
		}
		else{
			wi->win_geo.polar_axis= X_axis;
			wi->win_geo.low_angle= 0;
			wi->win_geo.high_angle= wi->radix;
			polarType= 9;
			_loX= (wi->radix/ 2.0);
			_hiX= (wi->radix/ 4.0);
		}
		_loY= MAX( ABS(wi->loY), ABS(wi->hiY));
		_hiY= _loY;
	}
	else{
/* testrfftw(__FILE__,__LINE__,-1);	*/
		polarType= 0;
		_loX= wi->loX;
		_hiY= wi->hiY;
		if( wi->absYFlag && wi->hinY< 0 && !wi->fit_ybounds ){
			if( wi->win_geo.bounds._loY> 0 ){
				_loY= MIN( fabs(wi->win_geo.bounds._hinY), wi->win_geo.bounds._loY );
			}
			else{
				_loY= fabs( wi->win_geo.bounds._hinY );
			}
			CLIP_EXPR( _loY, _loY - 0.1* fabs( _hiY - _loY), 0.0, MAXFLOAT );
		}
		else{
			_loY= wi->win_geo.bounds._loY;
		}
		_hiX= wi->hiX;
/* testrfftw(__FILE__,__LINE__,-1);	*/
	}
	  /* Yscale is adapted to actual range when
	   \ first window drawn first time in polar mode with logarithmic radius
	   \ wi->Yscale==0
	   */
	if( (wi->polarFlag && wi->logYFlag && (wi->redraw== -1 || wi->redraw_val==-1)
			&& wi->parent_number== 0 && wi->pwindow_number== 0 && wi->window_number== 0
		) || wi->Yscale== 0.0
	){
	  double _loY;
		if( wi->absYFlag && wi->win_geo.pure_bounds._hinY< 0 && !wi->fit_ybounds ){
			if( wi->win_geo.pure_bounds._loY> 0 ){
				_loY= MIN( fabs(wi->win_geo.pure_bounds._hinY), wi->win_geo.pure_bounds._loY );
			}
			else{
				_loY= (wi->win_geo.pure_bounds._hinY)? fabs( wi->win_geo.pure_bounds._hinY ) : wi->win_geo.pure_bounds._lopY;
			}
/* 			CLIP_EXPR( _loY, _loY - 0.1* fabs( _hiY - _loY), 0.0, MAXFLOAT );	*/
		}
		else{
			_loY= (wi->win_geo.pure_bounds._loY> 0 || wi->fit_ybounds )? wi->win_geo.pure_bounds._loY : wi->win_geo.pure_bounds._lopY;
		}
		if( _loY ){
		  double RoundUp(LocalWin*, double);
			RoundUp_log= wilog10;
			wi->Yscale= RoundUp(wi, 1/ _loY);
			if( (wi->polarFlag && wi->logYFlag && wi->redraw== -1 && wi->parent_number== 0
				&& wi->pwindow_number== 0 && wi->window_number== 0 ) &&
				fabs(_loY)> 1.0
			){
			  /* This feature is meant to rescale the data so that logarithmic Y<1-values do not
			   \ mess up polar representations ( log10(x<1) < 0 => suggests wrong angle).
			   \ If Yscale<1 then there are no Y<1 values, making it unnecessary to adapt Yscale
			   */
				wi->Yscale= 1.0;
				_loY= 1.0;
			}
			if( debugFlag ){
				fprintf( StdErr, "TransformCompute(): hinY=%g loY=%g lopY=%g => Yscale set to RoundUp(1/%g=%g)=%g\n",
					wi->win_geo.pure_bounds._hinY, wi->win_geo.pure_bounds._loY, wi->win_geo.pure_bounds._lopY,
					_loY, 1/_loY, wi->Yscale
				);
				fflush( StdErr );
			}
		}
		else{
			wi->Yscale= 1.0;
			if( debugFlag ){
				fprintf( StdErr, "TransformCompute(): hinY=%g loY=%g lopY=%g => Yscale set to %g\n",
					wi->win_geo.pure_bounds._hinY, wi->win_geo.pure_bounds._loY, wi->win_geo.pure_bounds._lopY,
					wi->Yscale
				);
				fflush( StdErr );
			}
		}
	}

    /*
     * First,  we figure out the origin in the X window.  Above
     * the space we have the title and the Y axis unit label.
     * To the left of the space we have the Y axis grid labels.
	 \ We do NOT do all of that when instructed to use the information from a previously drawn window!
	 \ (a previously drawn window about which we have valid size & printing info, and only if we're
	 \  sending to the same printing channel ourselves!)
     */

	if( Use_HO_Previous_TC && HO_PreviousWin_ptr && HO_PreviousWin_ptr->dev_info.user_state
		&& PS_STATE(wi)->Printing!= X_DISPLAY
		&& PS_STATE(wi)->Printing== PS_STATE(HO_PreviousWin_ptr)->Printing
	){
		wi->XOrgX= HO_PreviousWin_ptr->XOrgX;
		wi->XOrgY= HO_PreviousWin_ptr->XOrgY;
		wi->XOppX= HO_PreviousWin_ptr->XOppX;
		wi->XOppY= HO_PreviousWin_ptr->XOppY;
		uhptc= True;
	}
	else{
		uhptc= False;

		  /* 20040921: don't allocate space for empty, unplaced labels: */
		{ char *c= XLABEL(wi);
			while( isspace(*c) ){
				c++;
			}
			no_xname= (strlen(c)< 1)? True : False;
		}
		{ char *c= YLABEL(wi);
			while( isspace(*c) ){
				c++;
			}
			no_yname= (strlen(c)< 1)? True : False;
		}
		wi->XOrgX = wi->dev_info.bdr_pad + (YLabelLength *
				((wi->textrel.used_gsTextWidth>0 || (wi->textrel.used_gsTextWidth< 0 && wi->textrel.prev_used_gsTextWidth> 0))?
					1 : axis_width)
			) + wi->dev_info.bdr_pad;

		   /* remove leading and trailing spaces from title	*/
		cleanup( titleText );

		DrawTitle( wi, 0 );
		 if( !wi->no_title && titles ){
			 wi->XOrgY = 2* wi->dev_info.bdr_pad + titles* wi->dev_info.title_height+ wi->dev_info.label_height/2;
		 }
		 else{
			 if( debugFlag ){
				fprintf( StdErr, "TransformCompute(): no_title (%d) or strlen(titleText)==0 (%d)\n",
					wi->no_title, strlen( titleText)
				);
				fflush( StdErr);
			 }
			 wi->XOrgY = 2* wi->dev_info.bdr_pad+ wi->dev_info.label_height/2;
		}
		if( !wi->yname_placed && !no_yname ){
			wi->XOrgY+= wi->dev_info.label_height + wi->dev_info.bdr_pad;
		}
	}

    /*
     * Now we find the lower right corner..  Below the space we
     * have the X axis grid labels.  There also we
     * have the X axis unit label and the legend.  We assume the
     * worst case size for the unit label.
	 \ 990802: "seed" with some sensible values, to be able to do world->screen
	 \ co-ordinate conversions.
	 \ 20001215: do that only when the window has just been born?! For hardcopies, do_hardcopy()
	 \ should initialise at least X,YUnitsPerPixel based on the resolution proportions, so
	 \ that DrawLegend2() can give a correct idea about the dimensions of the legend it will draw...
     */

	if( !wi->redrawn ){
		if( wi->ValCat_X_levels== 0 ){
			wi->ValCat_X_levels= 1;
		}
		if( !uhptc ){
			wi->XOppX= wi->dev_info.area_w - wi->dev_info.bdr_pad;
			wi->XOppY = wi->dev_info.area_h - 2*wi->dev_info.bdr_pad - axis_height;
			if( wi->ValCat_X_axis && wi->axisFlag ){
				wi->XOppY-= (abs(wi->ValCat_X_levels)- 1)* axis_height;
			}
		}
		wi->UsrOrgX = _loX;
		wi->UsrOppX = _hiX;
		wi->UsrOrgY = _loY;
		wi->UsrOppY = _hiY;
	}
	if( !wi->XUnitsPerPixel ){
		wi->XUnitsPerPixel = (_hiX - _loX)/((double) (wi->XOppX - wi->XOrgX));
	}
	if( !wi->YUnitsPerPixel ){
		wi->YUnitsPerPixel = (_hiY - _loY)/((double) (wi->XOppY - wi->XOrgY));
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/

	  /* Seeded...	*/

	overlap(wi);
/* testrfftw(__FILE__,__LINE__,-1);	*/

    maxName = 0;
	maxFile = 0;
	maxlineWidth= 0;
	if( !wi->no_legend && wi->legend_type== 1 ){
	  LegendDimensions dim;
		  /* 20001215: DrawLegend2() (legend_type==1) is the more complex of the 2 legend
		   \ drawing routines; it does "WYSIWIG" display of bar plots, etc. It has become
		   \ too complicated to guess here what the exact dimensions it will require are.
		   \ Therefore, just call it with a dimension structure to fill in everything we
		   \ need and/or just like to know.
		   */
		DrawLegend2( wi, False, &dim );
		first_markerHeight= dim.first_markerHeight;
		markerWidth= dim.markerWidth;
		maxFile= dim.maxFile;
		maxName= dim.maxName;
		overlap_size= dim.overlap_size;
		maxFontWidth= dim.maxFontWidth;
	}
	else{
	  int mf= 0;
	  double lw= 0, _mw= -1, mW= markerWidth;
	  int mw, mh;
	  int ps_mark_scale;
	  char *name= NULL;

		showlines= 0;
		if( PS_STATE(wi)->Printing== PS_PRINTING){
		  /* in new_ps.c:
		   \ PS_MARK * ui->baseWidth = PS_MARK * rd(VDPI/POINTS_PER_INCH*BASE_WIDTH)
		   */
			ps_mark_scale= PS_MARK * PS_STATE(wi)->baseWidth;
		}
		else{
/* 			ps_mark_scale= 1;	*/
			ps_mark_scale= PS_MARK* BASE_WIDTH* (Xdpi/POINTS_PER_INCH)/ 0.9432624113475178; /* 1;	*/
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
		  /* To find out what is drawn and what not:	*/
		if( wi->filename_in_legend ){
			DrawLegend( wi, False, NULL );
/* testrfftw(__FILE__,__LINE__,-1);	*/
		}
/* 		for( idx = 0;  wi->no_legend== 0 && idx < MAXSETS;  idx++ )	*/
		for( idx = 0;  wi->no_legend== 0 && idx < setNumber;  idx++ )
		{
			if( AllSets[idx].numPoints > 0 ){
				int tempSize;
				char *n= parse_codes( AllSets[idx].setName );
				char *fn= (wi->labels_in_legend)? AllSets[idx].YUnits : AllSets[idx].fileName;

				if( wi->labels_in_legend && !fn ){
					if( idx ){
						fn= AllSets[idx].YUnits= XGstrdup( AllSets[idx-1].YUnits );
					}
				}
				if( wi->labels_in_legend && !AllSets[idx].XUnits ){
					if( idx ){
						AllSets[idx].XUnits= XGstrdup( AllSets[idx-1].XUnits );
					}
				}
				if( !fn ){
					fn= "";
				}
				if( n && strcmp( n, "*NOLEGEND*")!= 0 && draw_set(wi, idx) && AllSets[idx].show_legend ){
				  char *line;
					tempSize= 0;
					if( !xgraph_NameBuf || strlen(n)> xgraph_NameBufLen ){
						xfree( xgraph_NameBuf );
						xgraph_NameBuf= XGstrdup(n);
						xgraph_NameBufLen= strlen(xgraph_NameBuf);
					}
					else{
						strcpy( xgraph_NameBuf, n );
					}
					name= xgraph_NameBuf;
					while( name && xtb_getline( &name, &line ) ){
					 int len;
						if( xtb_has_greek( line) ){
							maxFontWidth= MAX(maxFontWidth, XFontWidth(legend_greekFont.font) );
						}
						if( !use_X11Font_length ){
							len = strlen(line);
						}
						else{
							len= XGTextWidth( wi, line, T_LEGEND, NULL );
						}
						tempSize= MAX( tempSize, len);
					}
					if( tempSize > maxName){
						maxName = tempSize;
					}
					mf+= AllSets[idx].markFlag;
					  /* Actually half the width/height:	*/
					MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
					  /* 20001213: quick hack to avoid to wide legends	*/
					if( wi->legend_type== 1 && !wi->no_legend && !(wi->legend_placed || legend_placed) &&
						AllSets[idx].markFlag && AllSets[idx].markSize< 0 &&
						wi->XOppX+ 2* wi->dev_info.bdr_pad+ LEG2_LINE_LENGTH(wi,mw)> wi->dev_info.area_w
					){
					  double ms= AllSets[idx].markSize;
						set_NaN(AllSets[idx].markSize);
						MarkerSizes( wi, idx, ps_mark_scale, &mw, &mh );
						AllSets[idx].markSize= ms;
					}
					if( !first_markerHeight ){
						first_markerHeight= mw;
					}
/* 					if( (AllSets[idx].markFlag && !AllSets[idx].pixelMarks) && mw> _mw )	*/
					if( mw> _mw )
					{
						markerWidth= mW+ mw;
						_mw= mw;
					}
					if( (!AllSets[idx].noLines || AllSets[idx].barFlag> 0) && AllSets[idx].show_llines ){
						showlines+= 1;
						lw= LINEWIDTH(wi, idx);
						if( AllSets[idx].barFlag> 0 ){
							lw= 5;
						}
						if( wi->legend_line[idx].highlight || AllSets[idx].barFlag> 0 ){
							lw= HL_WIDTH(lw);
						}
					}
					if( lw> maxlineWidth ){
						maxlineWidth= lw;
					}
				}
				if( wi->filename_in_legend && AllSets[idx].filename_shown ){
					if( !use_X11Font_length ){
						tempSize = strlen(fn);
					}
					else{
						tempSize= XGTextWidth( wi, fn, T_LEGEND, NULL );
					}
					if( tempSize > maxFile
/* 							&& (idx== setNumber- 1 || wi->new_file[idx+1] )	*/
					){
						maxFile = tempSize;
					}
				}
			}
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
		if( wi->show_overlap && (wi->overlap_buf || wi->overlap2_buf) ){
			if( !use_X11Font_length ){
				overlap_size = strlen( wi->overlap_buf );
				overlap_size = MAX( overlap_size, strlen( wi->overlap2_buf ) );
			}
			else{
				overlap_size= XGTextWidth( wi, wi->overlap_buf, T_LEGEND, NULL );
				overlap_size= MAX( overlap_size, XGTextWidth( wi, wi->overlap2_buf, T_LEGEND, NULL ));
			}
		}
		else{
			overlap_size= 0;
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
		if( !mf)
			markFlag= 0;

/* testrfftw(__FILE__,__LINE__,-1);	*/
		overlap_size+= (overlap_legend_tune + SIGN(overlap_legend_tune))* wi->dev_info.bdr_pad;
    }
	markerWidth= (double)((int)(markerWidth* 2 + 0.5));
	if( !use_X11Font_length ){
		maxFile= wi->dev_info.bdr_pad+ maxFile* wi->dev_info.legend_width;
		maxName= wi->dev_info.bdr_pad+ maxName* wi->dev_info.legend_width;
		overlap_size= wi->dev_info.bdr_pad+ overlap_size* wi->dev_info.legend_width;
	}
	  /* 20001215: do the following scaling only if not outputting to X11 screen (then the width
	   \ estimate will be correct)
	  */
	else if( wi->legend_type!= 1 && PS_STATE(wi)->Printing!= X_DISPLAY ){
	  /* scale the screen-based lenght estimate with the ratio of PS length estimate over
	   \ default screen length estimate (XFontWidth(); used when !use_X11Font_length)
	   */
		if( wi->textrel.used_gsTextWidth<= 0 ){
			maxFile= (int)((maxFile)* ((double)wi->dev_info.legend_width/ maxFontWidth )) + wi->dev_info.bdr_pad;
			maxName= (int)((maxName)* ((double)wi->dev_info.legend_width/ maxFontWidth )) + wi->dev_info.bdr_pad;
			overlap_size= (int)((overlap_size)* ((double)wi->dev_info.legend_width/ maxFontWidth )) + wi->dev_info.bdr_pad;
			if( debugFlag ){
				fprintf( StdErr, "TransformCompute(): legend_width=%d maxFontWidth=%d markerWidth=%g maxFile=%d "
						"maxName=%d overlap_size=%d overlap_tune=%g\n",
					wi->dev_info.legend_width, maxFontWidth, markerWidth,
					maxFile, maxName, overlap_size, (overlap_legend_tune + SIGN(overlap_legend_tune))* wi->dev_info.bdr_pad
				);
				fflush( StdErr );
			}
		}
	}
	if( wi->filename_in_legend<= 0 ){
		maxFile= 0;
		wi->filename_in_legend= 0;
	}

	{ Boolean dl2= (wi->legend_type==1)? True : False;
		  /* Worst case size of the legend, incorporating border padding for framing box: */
		  /* 20001215: use LEG2_LINE_LENGTH here too!	*/

		if( dl2 ){
#ifdef DEBUG
			if( showlines ){
				legend_lrx= (int) (LEG2_LINE_LENGTH(wi,markerWidth) + maxName + 0* 2.25* wi->dev_info.bdr_pad+ 0.5);
			}
			else{
				legend_lrx= (int) (markerWidth + maxName + 1.25* wi->dev_info.bdr_pad+ 0.5);
			}
#endif
			  /* 20001215: but actually, we already have that value, calculated by DrawLegend2():	*/
			legend_lrx= wi->legend_lrx;
		}
		else{
			legend_lrx= (int) markerWidth + maxName + wi->dev_info.bdr_pad;
		}
		  /* stage one in calculation of right-hand x-coordinates of legend: the legend widths	*/
		if( overlap_size> (markerWidth+ maxFile+ maxName) ){
			  /* The righthand-end of the marker-lines in the legend, and also the
			   \ X-coordinate of the lines combining sets.
			   */
			  /* The right edge of the legendbox is determined by the length of the overlap-string.	*/
			legend_frx= (int) overlap_size;
		}
		else{
			  /* The right edge of the legendbox is determined by the combined lengths
			   \ of the maximum legendstring and filename-string.
			   */
			legend_frx= maxFile + legend_lrx;
		}
		if( dl2 ){
			legend_frx= wi->legend_frx;
		}
	}
	  /* Just to be sure:	*/
	if( wi->legend_type!= 1 && legend_lrx> legend_frx ){
		legend_frx= legend_lrx + wi->dev_info.bdr_pad;
	}
	  /* The true legendWidth (?); legend_frx is only later "transformed" into
	   \ the absolute far-right coordinate, by adding the left coordinate to it.
	   */
	legendWidth= legend_frx+ wi->dev_info.bdr_pad;
/* testrfftw(__FILE__,__LINE__,-1);	*/

	switch( legend_placed ){
		case 2:
			legend_uly= wi->win_geo.pure_bounds._hiY;
			break;
		case 3:
			legend_ulx= wi->win_geo.pure_bounds._hiX;
			break;
	}
	switch( wi->legend_placed ){
		case 2:
			wi->_legend_uly= wi->win_geo.pure_bounds._hiY;
			break;
		case 3:
			wi->_legend_ulx= wi->win_geo.pure_bounds._hiX;
			break;
	}

	  /* X label is now printed UNDER the Xaxis, so we don't need
	   * to compensate for it anymore
	   \ 941006: yes we do! We don't want it to extend under the
	   \ requested bounding box! So dev_info.xname_vshift must
	   \ be incorporated in the calculation of XOppY
	   */
	if( !uhptc ){
		if( !(legend_placed || wi->legend_placed) && !wi->no_legend ){
			wi->XOppX = wi->dev_info.area_w - 3* wi->dev_info.bdr_pad - legendWidth;
		}
		else{
			wi->XOppX = wi->dev_info.area_w - wi->dev_info.bdr_pad;
		}
		if( markFlag && !wi->no_legend ){
			wi->XOppX-= mark_w/ 2;
		}
	}
	  /* calculate the lower boundary of the plot-region. Below come
	   \ the axis numbers and the Xlabel, all spaced with dev_info.bdr_pad.
	   \ xname_vshift is the distance XOppY - bottom_of_Xlabel; this contains
	   \ all size info of axis numbers and XLabel!
	   */
	if( wi->ValCat_X_levels== 0 ){
		wi->ValCat_X_levels= 1;
	}
	{ int fnt_height= wi->dev_info.axis_height;
		if( PS_STATE(wi)->Printing== PS_PRINTING){
			if( wi->ValCat_IFont && wi->ValCat_I && wi->ValCat_I_axis ){
				fnt_height= MAX( fnt_height, CustomFont_psHeight( wi->ValCat_IFont ) );
			}
		}
		else{
			if( wi->ValCat_IFont && wi->ValCat_I && wi->ValCat_I_axis ){
				fnt_height= MAX( fnt_height, CustomFont_height_X( wi->ValCat_IFont ) );
			}
		}
		wi->IntensityLegend.legend_height= fnt_height+ wi->dev_info.tick_len+ wi->dev_info.bdr_pad+ 2;
	}

	if( !uhptc ){
		if( !wi->xname_placed && !no_xname ){
			wi->XOppY = wi->dev_info.area_h - wi->dev_info.bdr_pad - wi->dev_info.xname_vshift;
		}
		else{
			wi->XOppY = wi->dev_info.area_h - 2*wi->dev_info.bdr_pad - axis_height;
			if( wi->ValCat_X_axis && wi->axisFlag ){
				wi->XOppY-= (abs(wi->ValCat_X_levels)- 1)* axis_height;
			}
		}
		if( wi->IntensityLegend.legend_needed && !wi->IntensityLegend.legend_placed ){
			wi->XOppY-= wi->IntensityLegend.legend_height;
		}

		if( (wi->XOrgX >= wi->XOppX) || (wi->XOrgY >= wi->XOppY) ){
			TitleMessage( wi, NULL );
			sprintf( err, "Drawing area too small (error #%ld)\n(%d,%d)-(%d,%d)\n%s",
				wi->draw_errors,
				wi->XOrgX, wi->XOrgY, wi->XOppX, wi->XOppY,
				((wi->init_pass)? "Window's first time redraw\n" :
					((wi->draw_errors>= 5)? "Will open Settings Dialog!\n" : ""))
			);
#ifdef TOOLBOX
			if( warn_about_size && !wi->init_pass ){
				if( QuietErrors && wi->draw_errors< quiet_error_count ){
					TitleMessage( wi, err );
				}
				else{
					if( Handle_An_Event( wi->event_level, 1, "TransformCompute()", wi->window,
							/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
						)
					){
						XG_XSync( disp, False );
						if( wi->delete_it== -1 ){
							goto bail;
						}
					}
					XG_error_box( &wi, "Warning", err, NULL );
					if( wi->draw_errors>= 5 ){
						xtb_bt_swap( wi->settings );
						wi->pw_placing= PW_MOUSE;
						wi->redrawn= -3;
						DoSettings(wi->window, wi);
						*ascanf_escape_value= ascanf_escape= 1;
					}
				}
			}
#else
			(void) fprintf(StdErr, "%s\n", err );
#endif
			wi->draw_errors+= 1;
			AdaptWindowSize( wi, wi->window, 0, 0 );
			goto bail;
		}
	}

    /*
     * We now have a bounding box for the drawing region.
     * Figure out the units per pixel using the data set bounding box.
     */

	  /* put the right log_zero value in the (global) log_zero_{x,y} variables,
	   \ which are then subjected to the current processing functions.
	   */
	if( !wi->log_zero_x_mFlag ){
	  /* User-specified value	*/
		wi->_log_zero_x= log_zero_x= wi->log_zero_x;
	}
	else{
	  /* value related to the current bounds.	*/
		wi->_log_zero_x= log_zero_x= (wi->log_zero_x_mFlag< 0)? wi->loX : wi->hiX;
	}
	if( !wi->log_zero_y_mFlag ){
		wi->_log_zero_y= log_zero_y= wi->log_zero_y;
	}
	else{
		wi->_log_zero_y= log_zero_y= (wi->log_zero_y_mFlag< 0)? wi->loY : wi->hiY;
	}

	ascanf_log_zero_x= log_zero_x;
	ascanf_log_zero_y= log_zero_y;

	  /* determine the (processed/transformed) substitution value for 0 in log mode:	*/
	if( (logXFlag> 0 && log_zero_x) || (logYFlag> 0 && log_zero_y) ){
	   DataSet *this_set= &AllSets[0];
	   double sx1, sy1, sx3, sy3, sx4, sy4;
		this_set->data[0][0]= sx1 = log_zero_x;
		this_set->data[0][1]= sy1 = log_zero_y;
		this_set->data[0][2]= 0.0;
		this_set->data[0][3]= 0.0;
		sx4= sx3= sx1;
		sy4= sy3 = sy1;

		if( _process_bounds && !this_set->raw_display ){
			DrawData_process( wi, this_set, this_set->data, 0, 1, 3, &sx1, &sy1, NULL, NULL,
				&sx3, &sy3, &sx4, &sy4, NULL, NULL, NULL, NULL
			);
		}
		do_TRANSFORM( wi, 0, 1, 1, &sx1, NULL, NULL, &sy1, &sy3, &sy4, 1, False );
		if( logXFlag && log_zero_x && sx1> 0 ){
			wi->log10_zero_x= cus_log10X(wi, sx1);
			wi->_log_zero_x= sx1;
		}
		else{
			wi->_log_zero_x= 0.0;
		}
		if( logYFlag && log_zero_y && sy1> 0 ){
			wi->log10_zero_y= cus_log10Y(wi, sy1);
			wi->_log_zero_y= sy1;
		}
		else{
			wi->_log_zero_y= 0.0;
		}
	}

	okX= 1;
	okY= 1;
	  /* If on-the-fly-processing is requested, we need to subject the boundaries
	   \ to this processing also!
	   */
	if( _process_bounds && wi->process_bounds && !wi->raw_display && wi->process.data_process_len ){
	  int i, spot= 0, n, ok, rsacsv= reset_ascanf_currentself_value;
	  double data[2][ASCANF_DATA_COLUMNS];
/* testrfftw(__FILE__,__LINE__,-1);	*/
		clean_param_scratch();
/* testrfftw(__FILE__,__LINE__,-1);	*/
		ok= 1;
		data[0][0]= _loX;
		data[0][1]= _loY;
		data[0][3]= data[0][2]= 0.0;
		data[1][0]= _hiX;
		data[1][1]= _hiY;
		data[1][3]= data[1][2]= 0.0;
		for( i= 0; i< 2; i++, spot++ ){
		  char change[ASCANF_DATA_COLUMNS];
		  int aae= 0;
			  /* 990912: not processing a set, but let's just do the DATA_INIT command(s)...: 	*/
			if( wi->process.data_init_len && spot== 0 ){
				n= param_scratch_len;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "TransformCompute(): DATA Init: %s", wi->process.data_init);
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*DATA_INIT*";
				compiled_fascanf( &n, wi->process.data_init, param_scratch, NULL, data[i], column, &wi->process.C_data_init );
				aae+= ascanf_arg_error;
				if( /* ascanf_arg_error || */ !n ){
					ok= 0;
				}
			}
			if( wi->process.data_before_len ){
				n= param_scratch_len;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "TransformCompute(): DATA Before: %s", wi->process.data_before);
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*DATA_BEFORE*";
				compiled_fascanf( &n, wi->process.data_before, param_scratch, NULL, data[i], column, &wi->process.C_data_before );
				aae+= ascanf_arg_error;
				if( /* ascanf_arg_error || */ !n ){
					ok= 0;
				}
			}
			if( ok ){
				n= ASCANF_DATA_COLUMNS;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "TransformCompute(): DATA x y e: %s", wi->process.data_process );
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*DATA_PROCESS*";
				compiled_fascanf( &n, wi->process.data_process, data[i], change, data[i], column, &wi->process.C_data_process );
				aae+= ascanf_arg_error;
				if( /* !ascanf_arg_error && */ n &&
					(change[0]== 'N' || change[0]== 'R') &&
					(change[1]== 'N' || change[1]== 'R')
				){
					ok= 1;
				}
				else{
					ok= 0;
				}
			}
			if( ok && wi->process.data_after_len ){
				n= param_scratch_len;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "TransformCompute(): DATA After: %s", wi->process.data_after );
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*DATA_AFTER*";
				compiled_fascanf( &n, wi->process.data_after, param_scratch, NULL, data[i], column, &wi->process.C_data_after );
				aae+= ascanf_arg_error;
				if( /* ascanf_arg_error || */ !n ){
					ok= 0;
				}
			}
			if( aae || !ok ){
				fprintf( StdErr, "TransformCompute(): %s,%s: error\n",
					d2str( data[i][0], "%g", NULL), d2str( data[i][1], "%g", NULL)
				);
				fflush( StdErr );
			}
			/* else */{
			  /* substitute values.	*/
				switch( i ){
					case 0:
						_loX = data[0][0];
						_loY = data[0][1];
						break;
					case 1:
						_hiX = data[1][0];
						_hiY = data[1][1];
						break;
				}
			}
			TBARprogress_header= NULL;
			ok= 1;
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
		reset_ascanf_currentself_value= rsacsv;
	}
	do_transform( wi, "low", __DLINE__, "TransformCompute(low)", &okX, NULL, &_loX, NULL, NULL, &_loY, NULL, NULL, NULL, NULL, 1,
		-1, 1.0, 1.0, 1.0, 1, 0, 0
	);
/* testrfftw(__FILE__,__LINE__,-1);	*/
	do_transform( wi, "high", __DLINE__, "TransformCompute(high)", &okY, NULL, &_hiX, NULL, NULL, &_hiY, NULL, NULL, NULL, NULL, 1,
		-1, 1.0, 1.0, 1.0, 1, 0, 0
	);
/* testrfftw(__FILE__,__LINE__,-1);	*/
	if( !okX ){
		Boing(10);
		if( !wi->init_pass && !wi->textrel.gs_batch ){
			xtb_error_box( wi->window, "TransformCompute(): invalid low bound(s) for axis settings\n", "Error" );
		}
		else{
			fprintf( StdErr, "TransformCompute(): error: invalid low bound(s) for axis settings\n" );
		}
		wi->draw_errors+= 1;
		TitleMessage( wi, NULL );
		goto bail;
	}
	else if( !okY ){
		Boing(10);
		if( !wi->init_pass && !wi->textrel.gs_batch ){
			xtb_error_box( wi->window, "TransformCompute(): invalid high bound(s) for axis settings\n", "Error" );
		}
		else{
			fprintf( StdErr, "TransformCompute(): error: invalid high bound(s) for axis settings\n" );
		}
		wi->draw_errors+= 1;
		TitleMessage( wi, NULL );
		goto bail;
	}

	switch( polarType ){
		case 1:
			_loX= -0.1 * _hiX;
			_hiY= _loY;
			_loY= -0.1 * _hiY;
			break;
		case 2:
			_hiX= -0.1 * _loX;
			_loY= -0.1 * _hiY;
			break;
		case 3:
			_hiX= -_loX;
			_loY= -0.1 * _hiY;
			break;
		case 4:
			_hiX= -0.1 * _loX;
			_loY= -_hiY;
			break;
		case 5:
			_hiX= -0.1 * _loX;
			_loY= - _hiY;
			_hiY= -0.1 * _loY;
			break;
		case 6:
			_hiX= -_loX;
			_loY= - _hiY;
			_hiY= -0.1 * _loY;
			break;
		case 7:
			_loX= -0.1 * _hiX;
			_loY*= -1;
			_hiY= -0.1 * _loY;
			break;
		case 8:
			_loX= -0.1 * _hiX;
			_hiY= _loY;
			_loY= -_hiY;
			break;
		case 9:
			_hiX= - _loX;
			_loY= - _hiY;
			break;
	}

	wi->axis_stuff.log_zero_x_spot= LOG_ZERO_INSIDE;
	wi->axis_stuff.log_zero_y_spot= LOG_ZERO_INSIDE;
	if( !wi->win_geo.user_coordinates ){
	 /* If the user didn't specify the coordinates by zooming, or in the settings dialogue,
	  \ we allow adaptation of ranges to current log_zero_x and/or
	  \ log_zero_y. If so, the possible 10% padding is undone, and
	  \ then redone after the changing of the "real" boundaries.
	  */
	  double padx= 0.0, pady= 0.0;
	  int done= 0;
/* testrfftw(__FILE__,__LINE__,-1);	*/
		if( wi->win_geo.padding ){
			padx= (_hiX - _loX)/ wi->win_geo.padding;
			pady= (_hiY - _loY)/ wi->win_geo.padding;
		}
		if( debugFlag ){
			fprintf( StdErr, "TransformCompute(): checking bounds (%g:%g)-(%g,%g)",
				Reform_X( wi, _loX, _loY), Reform_Y( wi, _loY, _loX),
				Reform_X( wi, _hiX, _hiY), Reform_Y( wi, _hiY, _hiX)
			);
		}
		if( wi->logXFlag && wi->logXFlag!= -1){
			if( wi->_log_zero_x> 0 ){
				if( wi->log10_zero_x<= _loX ){
					wi->axis_stuff.log_zero_x_spot= LOG_ZERO_LOW;
				}
				else if( wi->log10_zero_x>= _hiX ){
					wi->axis_stuff.log_zero_x_spot= LOG_ZERO_HIGH;
				}
				if( wi->log10_zero_x< _loX ){
					_loX= wi->log10_zero_x;
					_hiX-= padx;
				}
				else if( wi->log10_zero_x> _hiX ){
					_hiX= wi->log10_zero_x;
					_loX+= padx;
				}
				if( wi->win_geo.padding ){
					padx= (_hiX - _loX)/ wi->win_geo.padding;
					_loX-= padx;
					_hiX+= padx;
				}
				done= 1;
			}
		}
		if( wi->logYFlag && wi->logYFlag!= -1){
			if( wi->_log_zero_y> 0 ){
				if( wi->log10_zero_y<= _loY ){
					wi->axis_stuff.log_zero_y_spot= LOG_ZERO_LOW;
				}
				else if( wi->log10_zero_y>= _hiY ){
					wi->axis_stuff.log_zero_y_spot= LOG_ZERO_HIGH;
				}
				if( wi->log10_zero_y< _loY ){
					_loY= wi->log10_zero_y;
					_hiY-= pady;
				}
				else if( wi->log10_zero_y> _hiY ){
					_hiY= wi->log10_zero_y;
					_loY+= pady;
				}
				if( wi->win_geo.padding ){
					pady= (_hiY - _loY)/ wi->win_geo.padding;
					_loY-= pady;
					_hiY+= pady;
				}
				done= 1;
			}
		}
		if( debugFlag ){
			if( done ){
				fprintf( StdErr, "; increased to (%g:%g)-(%g,%g) to include log_zero_x,y (%g,%g)\n",
					Reform_X( wi, _loX, _loY), Reform_Y( wi, _loY, _loX),
					Reform_X( wi, _hiX, _hiY), Reform_Y( wi, _hiY, _hiX),
					X_Value( wi, wi->log10_zero_x), Y_Value( wi, wi->log10_zero_y)
				);
				fflush( StdErr );
			}
			else{
				fputc( '\n', StdErr );
				fflush( StdErr );
			}
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
	}
	if( _hiX == _loX ){
	 double pad= MAX(0.5, fabs(_hiX/2.0));
		_hiX+= pad;
		_loX-= pad;
	}
	if( _hiY == _loY ){
	 double pad= MAX(0.5, fabs(_hiY/2.0));
		_hiY+= pad;
		_loY-= pad;
	}

/* testrfftw(__FILE__,__LINE__,-1);	*/
	Check_Inf( _loX );
	Check_Inf( _hiX );
	Check_Inf( _loY );
	Check_Inf( _hiY );
/* testrfftw(__FILE__,__LINE__,-1);	*/

	do{
	  double Xrange= (double) wi->XOppX- (double) wi->XOrgX,
		  Yrange= (double) wi->XOppY- (double) wi->XOrgY;
		redo= False;

		wi->XUnitsPerPixel = (_hiX - _loX)/Xrange;
		wi->YUnitsPerPixel = (_hiY - _loY)/Yrange;

		if( NaNorInf(wi->XUnitsPerPixel) || wi->XUnitsPerPixel== 0 ){
			if( debugFlag || wi->fitting || wi->init_pass ){
				fprintf( StdErr,
					"X units-per-pixel is invalid:\n choose a different transformation\n or (a different collection of) sets\n"
					" or (un) set either transform_axes, process_bounds, or both\n"
				);
				fflush( StdErr );
				fprintf( StdErr, "%s%s%s%s%s%s%s=%s\n",
					d2str( wi->XUnitsPerPixel, "Xupp=%g", NULL),
					"; world=[", d2str( _loX, NULL, NULL), d2str( _hiX, ",%g]", NULL),
					"screen=[", d2str( (double) wi->XOrgX, NULL, NULL), d2str( (double) wi->XOppX, ",%g]", NULL),
					d2str( Xrange, NULL, NULL)
				);
				fflush( StdErr );
				  /* 20000505: we *can* get here with wi->fitting not set... but even if debugFlag
				   \ or wi->init_pass, we should return(0) (methinks).
				   */
				if( !wi->fitting ){
					TitleMessage( wi, NULL );
					goto bail;
				}
			}
			if( !wi->fitting && !wi->textrel.gs_batch ){
				XG_error_box( &wi,
					"X units-per-pixel is invalid:\n choose a different transformation\n or (a different collection of) sets\n"
					" or (un) set either transform_axes, process_bounds, or both\n",
					d2str( wi->XUnitsPerPixel, "Xupp=%g", NULL),
					"; world=[", d2str( _loX, NULL, NULL), d2str( _hiX, ",%g]", NULL),
					"screen=[", d2str( (double) wi->XOrgX, NULL, NULL), d2str( (double) wi->XOppX, ",%g]", NULL), "=",
					d2str( Xrange, NULL, NULL),
					NULL
				);
				wi->draw_errors+= 1;
				TitleMessage( wi, NULL );
				goto bail;
			}
			else{
						  /* When reversing this if/else, also reverse the if( !wi->fitting ){} several lines above!	*/
						fprintf( StdErr, "If scaling behaviour here is irritating, try reversing this if/else block (line %s::%d)\n",
							__FILE__, __LINE__
						);

				wi->XUnitsPerPixel= 1;
			}
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
		if( NaNorInf(wi->YUnitsPerPixel) || wi->YUnitsPerPixel== 0 ){
/*
			xtb_error_box( wi->window, "Y scale is invalid:\n choose a different transformation\n"
				" or (a different collection of) sets\n",
				d2str( wi->YUnitsPerPixel, "Error: scale=%g", NULL)
			);
 */
			if( debugFlag || wi->fitting || wi->init_pass ){
				fprintf( StdErr,
					"Y units-per-pixel is invalid:\n choose a different transformation\n or (a different collection of) sets\n"
					" or (un) set either transform_axes, process_bounds, or both\n"
				);
				fprintf( StdErr,
					"%s%s%s%s%s%s%s=%s\n",
					d2str( wi->YUnitsPerPixel, "Yupp=%g", NULL),
					"; world=[", d2str( _loY, NULL, NULL), d2str( _hiY, ",%g]", NULL),
					"; screen=[", d2str( (double) wi->XOrgY, NULL, NULL), d2str( (double) wi->XOppY, ",%g]", NULL),
					d2str( Yrange, NULL, NULL)
				);
				fflush( StdErr );
				if( !wi->fitting ){
					TitleMessage( wi, NULL );
					goto bail;
				}
			}
			  /* 20000505: above test was for !wi->fitting. Either that was correct, or the similar block
			   \ above for X units-per-pixel was incorrect, and should return(0) for wi->fitting...
			   */
			if( !wi->fitting && !wi->textrel.gs_batch ){
				XG_error_box( &wi,
					"Y units-per-pixel is invalid:\n choose a different transformation\n or (a different collection of) sets\n"
					" or (un) set either transform_axes, process_bounds, or both\n",
					d2str( wi->YUnitsPerPixel, "Yupp=%g", NULL),
					"; world=[", d2str( _loY, NULL, NULL), d2str( _hiY, ",%g]", NULL),
					"; screen=[", d2str( (double) wi->XOrgY, NULL, NULL), d2str( (double) wi->XOppY, ",%g]", NULL), "=",
					d2str( Yrange, NULL, NULL),
					NULL
				);
				wi->draw_errors+= 1;
				TitleMessage( wi, NULL );
				goto bail;
			}
			else{
						  /* When reversing this if/else, also reverse the if( !wi->fitting ){} several lines above!	*/
						fprintf( StdErr, "If scaling behaviour here is irritating, try reversing this if/else block (line %s::%d)\n",
							__FILE__, __LINE__
						);

				wi->YUnitsPerPixel= 1;
			}
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/

		wi->UsrOrgX = _loX;
		wi->UsrOppX = _hiX;
		wi->UsrOrgY = _loY;
		wi->UsrOppY = _hiY;
		if( wi->transform_axes ){
		  /* The R_UsrO..[XY] fields should always be useable by the DrawGridAndAxis()
		   \ routine for drawing the axes.
		   \ 990411:
		   \ transform_axes can be 1 (transform the axes; an X or Y TRANSFORM exists),
		   \ or -1 (no such transformation, or raw_display mode)
		   */
			wi->R_UsrOrgX= wi->UsrOrgX;
			wi->R_UsrOppX= wi->UsrOppX;
			wi->R_UsrOrgY= wi->UsrOrgY;
			wi->R_UsrOppY= wi->UsrOppY;
			wi->R_XUnitsPerPixel= wi->XUnitsPerPixel;
			wi->R_YUnitsPerPixel= wi->YUnitsPerPixel;
		}
		else{
		  double y;
			if( XSw ){
				wi->XUnitsPerPixel*= -1;
				y= wi->UsrOppX;
				wi->UsrOppX= wi->UsrOrgX;
				wi->UsrOrgX= y;
			}
			if( YSw ){
				wi->YUnitsPerPixel*= -1;
				y= wi->UsrOppY;
				wi->UsrOppY= wi->UsrOrgY;
				wi->UsrOrgY= y;
			}
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/

		if( debugFlag ){
			fprintf( StdErr, "TransformCompute(): boundingbox: (%g,%g) - (%g,%g)\n",
				wi->UsrOrgX, wi->UsrOrgY, wi->UsrOppX, wi->UsrOppY
			);
			fflush( StdErr );
		}
		  /* Determine the upper left coordinate of the legend box	*/
		if( legend_placed ){
		  /* put global coordinates in LocalWin	*/
			wi->_legend_ulx= legend_ulx;
			wi->_legend_uly= legend_uly;
			wi->legend_placed= legend_placed;
			wi->legend_trans= legend_trans;
		}
		if( xname_placed ){
			wi->xname_x= xname_x;
			wi->xname_y= xname_y;
			wi->xname_placed= 1;
			wi->xname_trans= xname_trans;
			xname_placed= 0;
			xname_trans= 0;
		}
		if( yname_placed ){
			wi->yname_x= yname_x;
			wi->yname_y= yname_y;
			wi->yname_placed= 1;
			wi->yname_trans= yname_trans;
			yname_placed= 0;
			yname_trans= 0;
		}
		if( yname_vertical ){
			wi->yname_vertical= True;
			yname_vertical= False;
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
		if( wi->legend_placed ){
		  /* Process LocalWin coordinates	*/
			okX=1;
			legend_ulx= wi->_legend_ulx;
			legend_uly= wi->_legend_uly;
/* testrfftw(__FILE__,__LINE__,-1);	*/
			do_transform( wi, "legend_ul", __DLINE__, "TransformCompute(legend_ul)", &okX, NULL, &legend_ulx, NULL, NULL, &legend_uly,
				NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, (wi->legend_trans)? 0 : -1, 0,
				(wi->legend_trans && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)))
			);
/* testrfftw(__FILE__,__LINE__,-1);	*/
			wi->legend_uly = SCREENY(wi,legend_uly);
			if( wi->legend_placed== 3 && !looped ){
				wi->legend_ulx= SCREENX(wi,legend_ulx) - legend_lrx;
				legend_lrx= SCREENX(wi,legend_ulx) - wi->legend_ulx;
				wi->_legend_ulx= Reform_X( wi, TRANX(wi->legend_ulx), TRANY(wi->legend_uly) );
				wi->_legend_uly= Reform_Y( wi, TRANY(wi->legend_uly), TRANX(wi->legend_ulx) );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			}
			else{
				wi->legend_ulx = SCREENX(wi,legend_ulx);
			}
			if( (wi->legend_placed== 2 || wi->legend_placed== 3) && (legend_placed || debugFlag) ){
			  FILE *fp= (debugFlag)? StdErr : stdout;
				fprintf( fp, "\"%s\": -legend_ul%s %g,%g\n", XGFetchName( wi),
					(wi->legend_trans)? "1" : "", wi->_legend_ulx, wi->_legend_uly );
				XGFetchName( NULL );
				fflush( fp );
			}
			legend_placed= 0;
/* testrfftw(__FILE__,__LINE__,-1);	*/
		}
		else{
		  /* Use default: place it at the right-upper corner
		   \ outside of the graph
		   */
			wi->legend_ulx = wi->XOppX + 2* wi->dev_info.bdr_pad;
			wi->legend_uly = wi->XOrgY- wi->dev_info.bdr_pad/2;
		}
		if( wi->legend_type!= 1 ){
			wi->legend_uly+= (int)(first_markerHeight + 0.5);
		}
		else if( !wi->no_legend ){
			  /* 20010112: the following is necessary in some cases, notably when
			   \ -fit_after and 1:1 are both activated. In that case, DrawLegend2()
			   \ gets called by us in the last auto-rescale/aspect-correct that will
			   \ get accepted, and thus not be followed by a final redraw.
			   */
			if( debugFlag ){
				fprintf( StdErr,
					"TransformCompute(): correcting DrawLegend2(0x%lx) legend_line.low_y,high_y for final legend_uly==%d\n",
					wi, wi->legend_uly
				);
			}
			for( idx= 0; idx< setNumber; idx++ ){
				if( wi->legend_line[idx].low_y>= 0 ){
					wi->legend_line[idx].low_y+= wi->legend_uly;
				}
				if( wi->legend_line[idx].high_y>= 0 ){
					wi->legend_line[idx].high_y+= wi->legend_uly;
				}
			}
		}
		  /* add the left-hand x-coordinates to the legend widths to get
		   \ the right-hand x-coordinates
		   */
		wi->legend_lrx= legend_lrx+ wi->legend_ulx;
		wi->legend_frx= legend_frx+ wi->legend_ulx;

		if( intensity_legend_placed ){
		  /* put global coordinates in LocalWin	*/
			wi->IntensityLegend._legend_ulx= intensity_legend_ulx;
			wi->IntensityLegend._legend_uly= intensity_legend_uly;
			wi->IntensityLegend.legend_placed= intensity_legend_placed;
			wi->IntensityLegend.legend_trans= intensity_legend_trans;
		}
		if( wi->IntensityLegend.legend_placed ){
		  /* Process LocalWin coordinates	*/
		  double ilegend_ulx= wi->IntensityLegend._legend_ulx,
			ilegend_uly= wi->IntensityLegend._legend_uly;
			okX=1;
/* testrfftw(__FILE__,__LINE__,-1);	*/
			do_transform( wi, "intensity_legend_ul", __DLINE__, "TransformCompute(IntensityLegend.legend_ul)",
				&okX, NULL, &ilegend_ulx, NULL, NULL, &ilegend_uly,
				NULL, NULL, NULL, NULL, 1, -1, 1.0, 1.0, 1.0, (wi->IntensityLegend.legend_trans)? 0 : -1, 0,
				(wi->IntensityLegend.legend_trans && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)))
			);
			wi->IntensityLegend.legend_uly = SCREENY(wi,ilegend_uly);
			wi->IntensityLegend.legend_ulx = SCREENX(wi,ilegend_ulx);
/* testrfftw(__FILE__,__LINE__,-1);	*/
			intensity_legend_placed= 0;
		}
		else{
		  /* Use default: just inside at bottom. TEMPORARY 	*/
/* testrfftw(__FILE__,__LINE__,-1);	*/
			if( !wi->no_intensity_legend && (wi->IntensityLegend.legend_needed< 0 || wi->IntensityLegend.legend_width<=0) ){
				DrawIntensityLegend( wi, False );
/* testrfftw(__FILE__,__LINE__,-1);	*/
			}
/* 			wi->IntensityLegend.legend_ulx = wi->XOrgX+ wi->dev_info.bdr_pad;	*/
			wi->IntensityLegend.legend_ulx= (wi->XOrgX+ wi->XOppX - wi->IntensityLegend.legend_width)/ 2;
/* 			wi->IntensityLegend.legend_uly = wi->XOppY- wi->dev_info.tick_len- wi->dev_info.bdr_pad- wi->IntensityLegend.legend_height;	*/
			wi->IntensityLegend.legend_uly= wi->dev_info.area_h- wi->IntensityLegend.legend_height;
			if( !wi->xname_placed && !no_xname ){
				wi->IntensityLegend.legend_uly-= wi->dev_info.label_height* 1.25;
			}
		}

		  /* Verify that the legend is "in view" by adapting wi->XO[rp][gp][XY] where necessary	*/
		if( wi->legend_always_visible &&
			( ((legend_placed || wi->legend_placed) && !wi->no_legend) ||
			  ((intensity_legend_placed || wi->IntensityLegend.legend_placed) && !wi->no_intensity_legend) ||
				wi->yname_placed || wi->xname_placed
			)
/* 			&& !looped	*/
		){
			if( wi->legend_placed && !wi->no_legend && looped< 1 ){
			  int frx;
				DrawLegend( wi, False, NULL );
				frx= wi->legend_frx+ (((int)(axisWidth+ zeroWidth+0.5))/2)* wi->dev_info.var_width_factor;
				redo= CShrinkArea( wi, /* _loX, _loY, _hiX, _hiY, 	*/
					wi->_legend_ulx, wi->_legend_uly, wi->legend_ulx, wi->legend_uly, frx, wi->legend_lry,
					uhptc
				);
				looped= 1;
			}
			if( !wi->no_intensity_legend && wi->IntensityLegend.legend_needed && wi->IntensityLegend.legend_placed && looped< 1 ){
				DrawIntensityLegend( wi, False );
			}
			if( !wi->no_intensity_legend && wi->IntensityLegend.legend_needed> 0 && wi->IntensityLegend.legend_placed && looped< 1 ){
			  int frx;
				frx= wi->IntensityLegend.legend_frx+ (((int)(axisWidth+ zeroWidth+0.5))/2)* wi->dev_info.var_width_factor;
				redo= CShrinkArea( wi, /* _loX, _loY, _hiX, _hiY, 	*/
					wi->IntensityLegend._legend_ulx, wi->IntensityLegend._legend_uly,
					wi->IntensityLegend.legend_ulx, wi->IntensityLegend.legend_uly, frx, wi->IntensityLegend.legend_lry,
					uhptc
				);
				looped= 1;
			}
			if( wi->xname_placed && !redo && looped< 2 ){
			  int len= XGTextWidth( wi, XLABEL(wi), T_LABEL, NULL ), ulx, uly, frx, fry;
			  int maxFontWidth= XFontWidth(labelFont.font);
				if( !_use_gsTextWidth ){
					if( xtb_has_greek( wi->XUnits) ){
						maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
					}
					len= (int)( len* ((double)wi->dev_info.label_width/ maxFontWidth ));
				}
				ulx= SCREENX( wi, wi->tr_xname_x);
				uly= SCREENY( wi, wi->tr_xname_y);
				frx= ulx+ len;
				fry= uly+ wi->dev_info.label_height;
				redo= CShrinkArea( wi, wi->tr_xname_x, wi->tr_xname_y, ulx, uly, frx, fry, uhptc );
				looped= 2;
			}
			if( wi->yname_placed && !redo && looped< 3 ){
			  int len= XGTextWidth( wi, YLABEL(wi), T_LABEL, NULL ), ulx, uly, frx, fry;
			  double yy;
			  int maxFontWidth= XFontWidth(labelFont.font);
				ulx= SCREENX( wi, wi->tr_yname_x);
				uly= SCREENY( wi, wi->tr_yname_y);
				if( !_use_gsTextWidth ){
					if( xtb_has_greek( wi->XUnits) ){
						maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
					}
					len= (int)( len* ((double)wi->dev_info.label_width/ maxFontWidth ));
				}
				if( wi->yname_vertical ){
					frx= ulx+ wi->dev_info.label_height;
					fry= uly;
					uly= fry- len;
					  /* The y co-ordinate of the real topleft point	*/
					yy= Reform_Y( wi, TRANY(uly), TRANX(ulx) );
					redo= CShrinkArea( wi, wi->tr_yname_x, yy, ulx, uly, frx, fry, uhptc );
				}
				else{
					frx= ulx+ len;
					fry= uly+ wi->dev_info.label_height;
					redo= CShrinkArea( wi, wi->tr_yname_x, wi->tr_yname_y, ulx, uly, frx, fry, uhptc );
				}
				looped= 3;
			}
		}
		if( wi->pen_list && !wi->no_pens ){
			if( looped< 4 && Outside_PenText( wi, &redo ) ){
				looped= 4;
			}
		}
		if( debugFlag && redo ){
			fprintf( StdErr, "TransformCompute(): redoing plot-region boundaries calculations...\n" );
			fflush( StdErr );
		}
/* testrfftw(__FILE__,__LINE__,-1);	*/
	}
	while( redo );
	  /* User co-ordinates of the window's corners.
	   \ Used to be calculated for each point in ClipWindow()...
	   \ Is not 100% transformation etc. proof...
	   */
	wi->WinOrgX= Reform_X( wi, TRANX(0), TRANY(wi->dev_info.area_h) );
	wi->WinOrgY= Reform_Y( wi, TRANY(wi->dev_info.area_h), TRANX(0) );
	wi->WinOppX= Reform_X( wi, TRANX(wi->dev_info.area_w), TRANY(0) );
	wi->WinOppY= Reform_Y( wi, TRANY(0), TRANX(wi->dev_info.area_w) );


	  /* Preen empty UserLabels	*/
	{ UserLabel *ul, *pul= NULL;
	  int ulabels= 0;
		while( wi->ulabel && !strlen(wi->ulabel->label) ){
			xfree( wi->ulabel->pixelCName );
			xfree( wi->ulabel->old2 );
			wi->ulabel= wi->ulabel->next;
			wi->ulabels-= 1;
		}
		ul= wi->ulabel;
		while( ul ){
			if( ul->set_link>= 0 && AllSets[ul->set_link].numPoints<= 0 ){
				ul->label[0]= '\0';
			}
			if( !strlen( ul->label) ){
				if( ShiftUndo.ul== ul ){
					ShiftUndo.ul= NULL;
				}
				xfree( ul->pixelCName );
				xfree( ul->old2 );
				if( pul ){
					pul->next= ul->next;
					xfree(ul);
					ul= pul->next;
				}
				else{
					wi->ulabel= ul->next;
					xfree( ul);
					ul= wi->ulabel;
				}
				wi->ulabels-= 1;
			}
			else{
				ulabels+= 1;
			}
			pul= ul;
			if( ul ){
				if( ul->set_link>= setNumber ){
					ul->set_link= setNumber- 1;
				}
				ul= ul->next;
			}
		}
		wi->ulabels= ulabels;
	}
/* testrfftw(__FILE__,__LINE__,-1);	*/
    /*
     * Everything is defined so we can now use the SCREENX and SCREENY
     * transformations.
     */
bail:
	TitleMessage( wi, NULL );
/* testrfftw(__FILE__,__LINE__,-1);	*/
	GCA();
/* testrfftw(__FILE__,__LINE__,-1);	*/
	*ascanf_setNumber= asn;
    legend_placed = l_placed;
    return 1;
}

extern int ps_transparent;

/* Find a category for the input value <val>, in the array <vcat>.
 \ The routine also tries to determine the categories with the associated
 \ values that closest surround <val> if <val> itself is not in the array;
 \ those category structures are returned in <low> and <high> if these are
 \ non-null pointers.
 */
ValCategory *Find_ValCat( ValCategory *vcat, double val, ValCategory **low, ValCategory **high )
{ Boolean found= False, jumped= False;
  ValCategory *mincat= NULL, *maxcat= NULL;
  unsigned int N;
	if( !vcat || !vcat->N ){
		return(NULL);
	}
	if( vcat->val== val ){
		return(vcat);
	}
	N= vcat->N-1;
	if( N && vcat[1].idx== vcat->idx+1 ){
	  double range;
	  /* A simple check to the conformity of the array */
		if( vcat->idx && vcat[-1].idx== vcat->idx-1 ){
		  /* We can retrieve the 1st element of the list: */
			vcat= &vcat[-vcat->idx];
		}
		if( !vcat->idx && (range=vcat->max- vcat->min) && N> 4 ){
		  int idx= (int) ((val- vcat->min)* N/ range + 0.5);
			  /* See if we can guess which element will have (or be closest to) this item: */
			if( idx> 0 && idx< vcat->N ){
				vcat= &vcat[idx];
				jumped= True;
			}
		}
	}
	if( vcat->val< val ){
		mincat= vcat;
	}
	else{
		maxcat= vcat;
	}
	  /* We always want to find neighbours, even if <val> is outside
	   \ the region of the first element. So this test is done after
	   \ possibly updating mincat and/or maxcat.
	   */
	while( vcat && !found ){
		if( val== vcat->val ){
			found= True;
		}
		else{
			if( vcat->idx== 0 ){
				  /* If we got here, then we "descended" to element 0. We can switch off the "jumped"
				   \ mode.
				   */
				jumped= False;
			}
			if( vcat->val< val ){
				if( !mincat ){
					if( vcat->val< val ){
						mincat= vcat;
					}
				}
				else if( val- vcat->val< val- mincat->val ){
				  /* This vcat->val is closer to val than maxcat->val:
				   \ update maxcat.
				   */
					mincat= vcat;
				}
			}
			else{
				if( !maxcat ){
					if( vcat->val> val ){
						maxcat= vcat;
					}
				}
				else if( vcat->val- val< maxcat->val- val ){
					maxcat= vcat;
				}
			}
			if( !(val>= vcat->min && val<= vcat->max) ){
				if( val> vcat->min ){
				  /* by definition, vcat->max will be smaller than val..
				   \ if user requested a low neighbour, find the element
				   \ with the maximum value.
				   */
					if( low ){
						while( vcat->min!= vcat->max ){
							vcat++;
						}
						mincat= vcat;
					}
					maxcat= NULL;
				}
				else{
					mincat= NULL;
				}
				if( !jumped ){
					  /* Finished. If jumped, than we may have jumped to the wrong spot! */
					vcat= NULL;
				}
			}
			if( jumped ){
			  /* Now determine if, and in which direction, we proceed. If vcat->val > val, we have
			   \ to descend, but only when the element below is >=val (otherwise, we'll end up oscillating).
			   \ We only do that when we're not in element 0... Same reasoning for ascending.
			   */
				if( vcat->val> val ){
					if( vcat->idx && vcat[-1].val >= val ){
						vcat--;
					}
					else{
						vcat= NULL;
					}
				}
				else if( vcat->idx< N && vcat[1].val<= val ){
					vcat++;
				}
				else{
					vcat= NULL;
				}
			}
			else if( vcat ){
				if( vcat->min!= vcat->max /* && (val>= vcat->min && val<= vcat->max) */ ){
					vcat++;
				}
				else{
				  /* We reached the end of the array, or <val> is outside the
				   \ range of the rest of the array
				   */
					vcat= NULL;
				}
			}
		}
	}
	if( low ){
		*low= mincat;
	}
	if( high ){
		*high= maxcat;
	}
	if( vcat && (found || vcat->val== val) ){
		return(vcat);
	}
	else{
		return(NULL);
	}
}

/* a qsort routine to increment-sort a ValCategory array.	*/
static int sort_ValCat( ValCategory *a, ValCategory *b )
{
	if( a->val< b->val ){
		return(-1);
	}
	else if( a->val> b->val ){
		return(1);
	}
	else{
		return(0);
	}
}

/* Add a ValCategory item to an array that may or may not already exist. current_N contains
 \ a pointer to the lists current length, value & category define the new entry. The routine
 \ does a qsort to always keep the list in increasing order and afterwards determines for
 \ each element the range of the values of the current and the following elements. This will
 \ permit fast searching, and also marks the end of the list (min=max).
 */
ValCategory *Add_ValCat( ValCategory *VCat, int *current_N, double value, char *category )
{ ValCategory *vcat;
	if( !current_N || *current_N< 0 ){
		return(NULL);
	}
	if( !(vcat= Find_ValCat( VCat, value, NULL, NULL )) ){
	  int i, N= *current_N+1;
	  SimpleStats SS;
		  /* Re-allocate the array: the new item will be at the end.	*/
		if( !(VCat= (ValCategory*) XGrealloc( VCat, N* sizeof(ValCategory) )) ){
			fprintf( StdErr, "Add_ValCat(N=%d, %s, \"%s\"): can't add item (%s)\n",
				*current_N, d2str( value, NULL, NULL), category, serror()
			);
			return( NULL );
		}
		VCat[*current_N].val= value;
		VCat[*current_N].category= XGstrdup(category);
		  /* an internal representation of the category, for use by Find_Point and update_LinkedLabel	*/
		{ char *vcs= concat( "\"", category, "\"=", d2str( value, NULL, NULL), NULL );
			strncpy( VCat[*current_N].vcat_str, vcs, MAXAXVAL_LEN );
			VCat[*current_N].vcat_str[MAXAXVAL_LEN-1]= '\0';
			xfree( vcs );
		}
		  /* Now sort the array	*/
		qsort( VCat, N, sizeof(ValCategory), (void*) sort_ValCat );
		  /* Initialise a SimpleStats bin, and update the min and
		   \ max fields working backwards
		   */
		SS_Init_(SS);
		for( i= *current_N; i>= 0; i-- ){
			SS_Add_Data_(SS, 1, VCat[i].val, 1.0 );
			VCat[i].min= SS.min;
			VCat[i].max= SS.max;
			VCat[i].idx= i;
			VCat[i].N= N;
		}
		if( debugFlag ){
			fprintf( StdErr, "Add_ValCat(N=%d, %s, \"%s\"): new item, current range is %s-%s\n",
				*current_N, d2str( value, NULL, NULL), category,
				d2str( VCat[0].min, NULL, NULL), d2str( VCat[0].max, NULL, NULL)
			);
		}
		*current_N+= 1;
	}
	else{
	  /* This is just a change of category - i.e. text changes, no new entry. No
	   \ changes are necessary to the rest of the array.
	   */
		xfree(vcat->category);
		vcat->category= XGstrdup(category);
	}
	return( VCat );
}

/* Return the length of a ValCategory array	*/
int ValCat_N( ValCategory *vcat )
{ int N= 0;
  ValCategory *vc= vcat;
	while( vcat ){
		vcat->idx= N;
		if( vcat->min!= vcat->max ){
			vcat++;
		}
		else{
			vcat= NULL;
		}
		N++;
	}
	if( vc ){
		vc->N= N;
	}
	return( N );
}

/* Free a ValCategory array	*/
ValCategory *Free_ValCat( ValCategory *vcat )
{ ValCategory *vc= vcat;
	while( vcat ){
#if __GNUC__ == 4 && __GNUC_MINOR__==0 && (defined(__APPLE_CC_) || defined(__MACH__)) && !defined(DEBUG)
		 /* 20050606: I don't know what this kludge is necessary for. I suspect a compiler issue.
		  \ without the print statement below, I get warnings about freeing twice, and they appear to
		  \ happen when we're called with vcat==NULL. I can trace only this much, as compiling this file
		  \ with -g makes the issue go away (a real Heisenbug) (the calling file ReadData.c *can* be compiled
		  \ with -g, and that shows calling with NULL).
		  */
		fprintf( NullDevice, "Free_ValCat(%p::%p->%p=\"%s\")\n", vc, vcat, vcat->category, vcat->category );
#endif
		xfree( vcat->category );
		vcat->idx= 0;
		vcat->N= 0;
		if( vcat->min!= vcat->max ){
			vcat++;
		}
		else{
			vcat= NULL;
		}
	}
	xfree( vc );
	return( NULL );
}

extern char *WriteValue( LocalWin *wi, char *str, double val, double val2, int exp, int logFlag, int sqrtFlag,
	AxisName axis, int use_real_value, double step_size, int len);

#if defined(i386) && defined(__GNUC__)
/* It has happened to me that the pow() function crashed on me, probably a matter of the wrong combination
 \ of aligment problems and compiler flags.
 */
#	ifdef DEBUG
double XGpow( double x, double y )
{
	if( debugFlag ){
	  double p;
		fprintf( StdErr, "pow(%g,%g)= ", x, y ); fflush( StdErr );
		p= pow(x, y);
		fprintf( StdErr, "%g\n", p ); fflush( StdErr );
		return(p);
	}
	else{
		return( pow(x,y) );
	}
}
#	else
double XGpow( double x, double y )
{
	return( pow( x, y ) );
}
#	endif

#else
#define XGpow	pow
#endif

int AddAxisValue( LocalWin *wi, AxisValues *av, double val )
{ char an[AxisNames]= "XYI";
	if( !av->array || av->last_index >= av->N ){
	  int old= av->N;
		av->N= (av->N)? 2* av->N : 8;
		if( debugFlag && debugLevel ){
			fprintf( StdErr, " [expanding from %d to %d] ",
				old, av->N
			);
			fflush(StdErr);
		}
		if( !(av->array= (double*) realloc( av->array, av->N* sizeof(double))) ||
			!(av->labelled= (char*) realloc( av->labelled, av->N* sizeof(char)))
		){
			fprintf( StdErr, "AddAxisValue(%c,%d,%s): can't expand to %d items (%s)\n",
				an[av->axis], av->last_index, d2str(val,0,0), av->N, serror()
			);
			xfree( av->array );
			xfree( av->labelled );
			av->N= 0;
			return( -1 );
		}
	}
	if( debugFlag && debugLevel ){
		fprintf( StdErr, " [adding %c value %d==%s of %d] ",
			an[av->axis], av->last_index, d2str(val,0,0), av->N
		);
		fflush(StdErr);
	}
	if( !av->last_index || val!= av->array[av->last_index-1] ){
		av->labelled[av->last_index]= 0;
		av->array[av->last_index++]= val;
	}
	return( av->last_index );
}

int AxisValueCurrentLabelled( LocalWin *wi, AxisValues *av, int label )
{
	return( (av->last_index)? (av->labelled[av->last_index-1]= label) : 0 );
}

int CompactAxisValues( LocalWin *wi, AxisValues *av )
{ char an[AxisNames]= "XYI";
	if( av->last_index && av->last_index< av->N ){
	  int old= av->N;
		av->N= av->last_index+ 1;
		if( debugFlag && debugLevel ){
			fprintf( StdErr, "CompactAxisValue(%c): compacting from %d to %d items\n",
				an[av->axis], old, av->N
			);
			fflush(StdErr);
		}
		if( !(av->array= (double*) realloc( av->array, av->N* sizeof(double))) ||
			!(av->labelled= (char*) realloc( av->labelled, av->N* sizeof(char)))
		){
			fprintf( StdErr, "CompactAxisValue(%c,%d): can't compact to %d items (%s)\n",
				an[av->axis], av->last_index, av->N, serror()
			);
			xfree( av->array );
			xfree( av->labelled );
			av->N= 0;
			return( -1 );
		}
	}
	return( av->N );
}

int DrawGridAndAxis( LocalWin *wi, int doIt )
/*
 * This routine draws grid line labels in engineering notation,
 * the grid lines themselves,  and unit labels on the axes.
 */
{ int j, xvcN= 0, expX= 0, expY= 0;		/* Engineering powers */
    int startX;
    int Yspot, Xspot, lspot, hspot, X_lspot= -1, Y_lspot= -1, X_hspot= -1, Y_hspot= -1,
		last_spot, polar_dot_x, polar_dot_y, polar_width, polar_height;
	double _R_UsrOrgX, _R_UsrOrgY, _R_UsrOppX, _R_UsrOppY;
    char power[32], value[MAXAXVAL_LEN], last_value[MAXAXVAL_LEN];
	ValCategory *vcat, *mincat, *maxcat;
	ALLOCA( final, char, LMAXBUFSIZE+10, final_len);
    double Xincr, Yincr, Xstart, Ystart, Yindex, Xindex, larger,
		XRange, YRange, Xbias, Ybias, val, Mval;
#ifdef _AUX_SOURCE
		/* this is for gcc: ** RJB **	*/
	double zero_thres= XGpow( 10.0, ZERO_THRES)
#endif
    XSegment segs[2];
	int axxmin, axxmax, axymin, axymax, x_margin;
    double gridscale, numscale, initGrid(LocalWin*, double, double, int, int, AxisName, int), stepGrid(double);
	double T_Yindex;
	int LogXFlag= polarLog || (wi->logXFlag && wi->logXFlag!= -1),
		LogYFlag= (wi->logYFlag && wi->logYFlag!= -1);
	int SqrtXFlag= polarLog || (wi->sqrtXFlag && wi->sqrtXFlag!= -1),
		SqrtYFlag= (wi->sqrtYFlag && wi->sqrtYFlag!= -1);
	int _XLabelLength, _YLabelLength;
	Pixel color1= AllAttrs[1].pixelValue;
	int LSLen1= AllAttrs[1].lineStyleLen;
	char LS1[MAXLS];
	int lcolour= 0, lstyle= 0, ltype= L_AXIS, prev_silent;
	Boolean axis_ok;
	extern int is_log_zero, is_log_zero2;
	extern double gridBase, gridStep;
	CustomFont *vcatfont;
	int X_axis_width, _X_a_w, X_axis_height;
	double __X_a_w;
	int Y_axis_width, _Y_a_w, Y_axis_height;
	double Xincr_factor= (wi->ValCat_X_axis)? wi->ValCat_X_incr : wi->Xincr_factor,
		Xincr_LabelLength;
	double Yincr_factor= (wi->ValCat_Y_axis)? wi->ValCat_Y_incr : wi->Yincr_factor;
	int valcat_vgrid= ValCat_X_grid, last_width= 0;
	_ALLOCA( last_right, int, (abs(wi->ValCat_X_levels)+ 1), last_right_size);

#define __XLabelLength	wi->axis_stuff.__XLabelLength
#define __YLabelLength	wi->axis_stuff.__YLabelLength
#define DoIt	wi->axis_stuff.DoIt
#define _polarFlag	wi->axis_stuff._polarFlag

#ifdef DEBUG
	if( !wi ){
		fprintf( StdErr, "DrawGridAndAxis() called with NULL argument\n" );
		return(0);
	}
#endif
	  /* AllAttrs[1] is used to temporarily pass gridPixel to the drawing routines	*/
	memcpy( LS1, AllAttrs[1].lineStyle, MAXLS* sizeof(char) );

	DoIt= (doIt>= 0)? doIt : 0;

	TitleMessage( wi, "Grids & Axes" );

	if( !DoIt ){
	  /* Silence the output	*/
		prev_silent= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
	}
	else{
	  /* Don't change the output, but retrieve the silenced state	*/
		prev_silent= X_silenced(wi);
	}

	Y_axis_width= X_axis_width= wi->dev_info.axis_width;
	Y_axis_height= X_axis_height= wi->dev_info.axis_height;
	if( wi->ValCat_XFont && wi->ValCat_X && wi->ValCat_X_axis ){
		X_axis_width= MAX( X_axis_width, (*wi->dev_info.xg_CustomFont_width)( wi->ValCat_XFont ) );
		X_axis_height= MAX( X_axis_height, (*wi->dev_info.xg_CustomFont_height)( wi->ValCat_XFont ) );
	}
	if( wi->ValCat_YFont && wi->ValCat_Y && wi->ValCat_Y_axis ){
		Y_axis_width= MAX( Y_axis_width, (*wi->dev_info.xg_CustomFont_width)( wi->ValCat_YFont ) );
		Y_axis_height= MAX( Y_axis_height, (*wi->dev_info.xg_CustomFont_height)( wi->ValCat_YFont ) );
	}
	if( wi->textrel.used_gsTextWidth ){
		  /* Cache the calculated width values, and set them to 1	*/
		_X_a_w= X_axis_width;
		__X_a_w= (double) X_axis_width/ (double) XFontWidth(axisFont.font);
		_Y_a_w= Y_axis_width;
		X_axis_width= Y_axis_width= 1;
	}
	else{
		_X_a_w= _Y_a_w= 1; __X_a_w= 1;
	}

	_R_UsrOrgX= wi->R_UsrOrgX;
	_R_UsrOppX= wi->R_UsrOppX;
	if( wi->win_geo.nobb_range_X ){
	  int okl= 1, okh= 1;
	  double _loX= wi->win_geo.nobb_loX,
			_hiX= wi->win_geo.nobb_hiX,
			_loY= wi->R_UsrOrgY;
		okl= do_TRANSFORM( wi, 0, 1, 1, &_loX, NULL, NULL, &_loY, NULL, NULL, 1, True );
		_loY= wi->R_UsrOrgY;
		okh= do_TRANSFORM( wi, 0, 1, 1, &_hiX, NULL, NULL, &_loY, NULL, NULL, 1, True );
		if( !okl ){
			Boing(10);
			XG_error_box( &wi, "warning", "DrawGridAndAxis(nobb_loX): ignored invalid low bound for non-border axis settings:\n",
				d2str(wi->win_geo.nobb_loX, 0,0), ",", d2str(wi->R_UsrOrgY, 0,0), "=>",
				d2str(_loX,0,0), ",", d2str(_loY,0,0),
				NULL
			);
			TitleMessage( wi, NULL );
			X_lspot= X_hspot= -1;
		}
		else if( !okh ){
			Boing(10);
			XG_error_box( &wi, "warning", "DrawGridAndAxis(nobb_hiX): ignored invalid high bound for non-border axis settings:\n",
				d2str(wi->win_geo.nobb_hiX, 0,0), ",", d2str(wi->R_UsrOrgY, 0,0), "=>",
				d2str(_loX,0,0), ",", d2str(_loY,0,0),
				NULL
			);
			TitleMessage( wi, NULL );
			X_lspot= X_hspot= -1;
		}
		else{
		  int x;
			if( (x= SCREENX( wi, _loX))>= 0 ){
				X_lspot= x;
				_R_UsrOrgX= MIN( wi->win_geo.nobb_loX, wi->R_UsrOrgX);
			}
			else{
				XG_error_box( &wi, "warning",
					"DrawGridAndAxis(): ignoring invalid non-border Xaxis low setting\n (" __FILE__ STRING(__LINE__) "):\n",
					d2str(wi->win_geo.nobb_loX, 0,0), ",", d2str(wi->R_UsrOrgY, 0,0), "=>",
					d2str(_loX,0,0), "; screen Y=", d2str(x,0,0),
					NULL
				);
			}
			if( (x= SCREENX( wi, _hiX))>= 0 ){
				X_hspot= x;
				_R_UsrOppX= MAX( wi->win_geo.nobb_hiX, wi->R_UsrOppX);
			}
			else{
				XG_error_box( &wi, "warning",
					"DrawGridAndAxis(): ignoring invalid non-border Xaxis high setting\n (" __FILE__ STRING(__LINE__) "):\n",
					d2str(wi->win_geo.nobb_hiX, 0,0), ",", d2str(wi->R_UsrOrgY, 0,0), "=>",
					d2str(_hiX,0,0), "; screen Y=", d2str(x,0,0),
					NULL
				);
			}
		}
	}
	_R_UsrOrgY= wi->R_UsrOrgY;
	_R_UsrOppY= wi->R_UsrOppY;
	if( wi->win_geo.nobb_range_Y ){
	  int okl= 1, okh= 1;
	  double _loY= wi->win_geo.nobb_loY,
			_hiY= wi->win_geo.nobb_hiY,
			_loX= wi->R_UsrOrgX;
		okl= do_TRANSFORM( wi, 0, 1, 1, &_loX, NULL, NULL, &_loY, NULL, NULL, 1, True );
		_loX= wi->R_UsrOrgX;
		okh= do_TRANSFORM( wi, 0, 1, 1, &_loX, NULL, NULL, &_hiY, NULL, NULL, 1, True );
		if( !okl ){
			Boing(10);
			XG_error_box( &wi, "warning", "DrawGridAndAxis(nobb_loY): ignored invalid low bound for non-border axis settings:\n",
				d2str(wi->R_UsrOrgX, 0,0), ",", d2str(wi->win_geo.nobb_loY, 0,0), "=>",
				d2str(_loX,0,0), ",", d2str(_loY,0,0),
				NULL
			);
			TitleMessage( wi, NULL );
			Y_lspot= Y_hspot= -1;
		}
		else if( !okh ){
			Boing(10);
			XG_error_box( &wi, "warning", "DrawGridAndAxis(nobb_hiY): ignored invalid high bound for non-border axis settings:\n",
				d2str(wi->R_UsrOrgX, 0,0), ",", d2str(wi->win_geo.nobb_hiY, 0,0), "=>",
				d2str(_loX,0,0), ",", d2str(_hiY,0,0),
				NULL
			);
			TitleMessage( wi, NULL );
			Y_lspot= Y_hspot= -1;
		}
		else{
		  int x;
			if( (x= SCREENY( wi, _loY))>= 0 ){
				Y_hspot= x;
				_R_UsrOppY= MAX( wi->win_geo.nobb_hiY, wi->R_UsrOppY);
			}
			else{
				XG_error_box( &wi, "warning",
					"DrawGridAndAxis(): ignoring invalid non-border Yaxis low setting\n (" __FILE__ STRING(__LINE__) "):\n",
					d2str(wi->R_UsrOrgX, 0,0), ",", d2str(wi->win_geo.nobb_loY, 0,0), "=>",
					d2str(_loY,0,0), "; screen Y=", d2str(x,0,0),
					NULL
				);
			}
			if( (x= SCREENY( wi, _hiY))>= 0 ){
				Y_lspot= x;
				_R_UsrOrgY= MIN( wi->win_geo.nobb_loY, wi->R_UsrOrgY);
			}
			else{
				XG_error_box( &wi, "warning",
					"DrawGridAndAxis(): ignoring invalid non-border Yaxis high setting\n (" __FILE__ STRING(__LINE__) "):\n",
					d2str(wi->R_UsrOrgX, 0,0), ",", d2str(wi->win_geo.nobb_hiY, 0,0), "=>",
					d2str(_hiY,0,0), "; screen Y=", d2str(x,0,0),
					NULL
				);
			}
		}
	}

      /*
       * Grid display powers are computed by taking the log of
       * the largest numbers and rounding down to the nearest
       * multiple of 3. (Engineering notation)
       */
	if( !noExpX){
		if( LogXFlag || SqrtXFlag ){
		  double orgx,	oppx;
/*
		  	if( LogXFlag){
			   	orgx= cus_pow_y_pow_xX(wi, 10.0, _R_UsrOrgY);
				oppx= cus_pow_y_pow_xX(wi, 10.0, _R_UsrOppX);
			}
			else{
			   	orgx= cus_sqr( _R_UsrOrgY);
				oppx= cus_sqr( _R_UsrOppX);
			}
 */
			orgx= Reform_X( wi, _R_UsrOrgX, _R_UsrOrgY );
			oppx= Reform_X( wi, _R_UsrOppX, _R_UsrOppY );
			if( orgx > oppx ){
				larger = orgx;
			} else {
				larger = oppx;
			}
			expX = ((int) ssfloor(cus_log10X(wi,fabs(larger/ wi->Xscale))/3.0)) * 3;
		} else {
			if( fabs(_R_UsrOrgX) > fabs(_R_UsrOppX) ){
				larger = fabs(_R_UsrOrgX);
			} else {
				larger = fabs(_R_UsrOppX);
			}
			expX = ((int) ssfloor(cus_log10X(wi,fabs(larger/ wi->Xscale))/3.0)) * 3;
		}
    }
	if( !noExpY){
		if( LogYFlag || SqrtYFlag){
		  double orgy, oppy;
/*
			if( LogYFlag){
				orgy= cus_pow_y_pow_xY(wi, 10.0, _R_UsrOrgY);
				oppy= cus_pow_y_pow_xY(wi, 10.0, _R_UsrOppY);
			}
			else{
				orgy= cus_sqr( _R_UsrOrgY);
				oppy= cus_sqr( _R_UsrOppY);
			}
 */
			orgy= Reform_Y( wi, _R_UsrOrgY, _R_UsrOrgX );
			oppy= Reform_Y( wi, _R_UsrOppY, _R_UsrOppX );
			if( orgy > oppy ){
				larger = orgy;
			} else {
				larger = oppy;
			}
			expY = ((int) ssfloor(cus_log10Y(wi,(larger/ wi->Yscale))/3.0)) * 3;
		} else {
			if( fabs(_R_UsrOrgY) > fabs(_R_UsrOppY) ){
				larger = fabs(_R_UsrOrgY);
			} else {
				larger = fabs(_R_UsrOppY);
			}
			expY = ((int) ssfloor(cus_log10Y(wi,(larger/ wi->Yscale))/3.0)) * 3;
		}
    }

	  /* Now,  the grid lines or tick marks
	   \ 990709: (finally) the drawing of gridlines/ticks is incorporated in the same loop which
	   \ draws the gridlabels. This has the huge advantage of drawing the grid at the same place
	   \ as the labels, even if the label "decides" to be positioned at another place than decided
	   \ by initGrid (if exact_?_axis is true, or when ValCat_?_axis is true). The old loops have not
	   \ become obsolete: they determine a.o. YLabels and XLabels, and some other constants used in
	   \ the updated 2nd loops. The old, displaced grid/tick code is preserved within #ifdefs.
	   */
	if( NaNorInf(Yincr_factor) || ((int) Yincr_factor != Yincr_factor) ){
		gridscale= Yincr_factor;
	}
	else{
		gridscale= 1.0;
	}
	Yincr = (wi->dev_info.axis_pad + Y_axis_height)* gridscale * wi->R_YUnitsPerPixel;
	Ystart = initGrid(wi, _R_UsrOrgY, Yincr, LogYFlag, SqrtYFlag , Y_axis, 0 );
	Xindex= gridBase+ gridStep;
	if( Xindex- gridBase> 0 ){
		axis_ok= True;
	}
	else{
		axis_ok= False;
	}
	Yspot= wi->XOrgY;
	axymin= -1;
	axymax= -1;
	for( YLabels= 0, Yindex = Ystart;
		axis_ok &&
		(	(Yincr> 0 && (val=Y_Value(wi,Yindex)) < (Mval=Y_Value(wi,_R_UsrOppY))) ||
			(Yincr< 0 && val > Mval)
		);
		YLabels++, Yindex = stepGrid(1.0)
	){
	 double T_Yindex= Yindex;
/* 		if( !wi->transform_axes && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) )	*/
		if( !wi->transform_axes )
		{
		  /* TRANSFORM_[XY] has not yet been applied to the boundaries - i.e. the labels shown
		   \ at the axes are based on the non-TRANSFORM_[XY]'ed ranges. They must however be
		   \ shown at the correct location, so the necessary transforms must be done now.
		   */
		  double x= _R_UsrOrgX, ly= T_Yindex, hy= T_Yindex;
			do_TRANSFORM( wi, 0, 1, 1, &x, NULL, NULL, &T_Yindex, &ly, &hy, 1, True );
		}
		if( YLabels== 0 ){
			Ystart= T_Yindex;
		}
		Yspot = SCREENY(wi, T_Yindex);
		if( Yspot>= wi->XOrgY && Yspot<= wi->XOppY &&
			(( wi->axis_stuff.log_zero_y_spot== LOG_ZERO_LOW && Yindex>= wi->log10_zero_y) ||
				(wi->axis_stuff.log_zero_y_spot== LOG_ZERO_HIGH && Yindex<= wi->log10_zero_y) ||
				wi->axis_stuff.log_zero_y_spot== LOG_ZERO_INSIDE
			)
		){
			YRange= T_Yindex- Ystart;
		}
	}

	  /* X grid/ticks	*/
	if( NaNorInf(Xincr_factor) || ((int) Xincr_factor != Xincr_factor) ){
		gridscale= Xincr_factor;
	}
	else{
		gridscale= 1.0;
	}
	  /* divide XLabelLength by the cached X_axis_width. This sort-of ensures identical behaviour
	   \ whether or not use_gsTextWidth.
	   */
	if( wi->textrel.used_gsTextWidth && PS_STATE(wi)->Printing== PS_PRINTING ){
	  double xil;
		xil= (XLabelLength/XFontWidth(axisFont.font)+1.5)* _X_a_w;
		if( XLabelLength ){
			Xincr_LabelLength= xil/ __X_a_w/ ((double) XLabelLength/(double) _X_a_w);
		}
		else{
			Xincr_LabelLength= (double) XLabelLength/ (double) _X_a_w;
		}
	}
	else{
		Xincr_LabelLength= (double) XLabelLength/ (double) _X_a_w;
	}
	Xincr = (wi->dev_info.axis_pad + (Xincr_LabelLength * X_axis_width))* gridscale * wi->R_XUnitsPerPixel;
	Xstart = initGrid(wi, _R_UsrOrgX, Xincr, LogXFlag, SqrtXFlag, X_axis, 0 );
	Xindex= gridBase+ gridStep;
	if( Xindex- gridBase> 0 ){
		axis_ok= True;
	}
	else{
		axis_ok= False;
	}
	Xspot= wi->XOrgX;
	axxmin= -1;
	axxmax= -1;
	if( wi->axisFlag ){
		x_margin= wi->dev_info.tick_len+ 2* wi->dev_info.bdr_pad;
	}
	for( XLabels= 0, Xindex = Xstart;
		axis_ok &&
		(	(Xincr> 0 && (val=X_Value(wi,Xindex)) < (Mval=X_Value(wi,_R_UsrOppX))) ||
			(Xincr< 0 && val > Mval)
		) ;
		XLabels++, Xindex = stepGrid(1.0)
	){
	  double T_Xindex= Xindex;
		if( !wi->transform_axes ){
		  double y= _R_UsrOrgY, ly= y, hy= y;
			do_TRANSFORM( wi, 0, 1, 1, &T_Xindex, NULL, NULL, &y, &ly, &hy, 1, True );
		}
		if( XLabels== 0 ){
			Xstart= T_Xindex;
		}
		Xspot = SCREENX(wi, T_Xindex);
		if( Xspot>= wi->XOrgX+ x_margin && Xspot<= wi->XOppX &&
			(( wi->axis_stuff.log_zero_x_spot== LOG_ZERO_LOW && Xindex>= wi->log10_zero_x) ||
				(wi->axis_stuff.log_zero_x_spot== LOG_ZERO_HIGH && Xindex<= wi->log10_zero_x) ||
				wi->axis_stuff.log_zero_x_spot== LOG_ZERO_INSIDE
			)
		){
			XRange= T_Xindex- Xstart;
		}
	}

	if( --XLabels<= 0)
		XLabels= 1;
	if( --YLabels<= 0)
		YLabels= 1;

	if( debugFlag){
		fprintf( StdErr, "Ranges: X: %g/%d Y: %g/%d\n",
			XRange, XLabels, YRange, YLabels
		);
		fflush( StdErr);
	}

	if( wi->polarFlag){
		polar_dot_x= SCREENX( wi, 0);
		polar_dot_y= SCREENY( wi, 0);
		polar_width= MAX( abs(wi->XOppX - polar_dot_x), abs( wi->XOrgX - polar_dot_x) );
		polar_height= MAX( abs(wi->XOppY - polar_dot_y), abs( wi->XOrgY - polar_dot_y) );
	}
/* 			(double)(wi->XOppX - wi->XOrgX) / (double)(wi->XOppY - wi->XOrgY)	*/

	  /*
	   * the grid line labels (axis numbers)
	   */

	if( wi->lz_sym_y ){
		parse_codes( wi->log_zero_sym_y );
	}
	if( wi->lz_sym_x ){
		parse_codes( wi->log_zero_sym_x );
	}

	  /* Now, let us draw the Y axis, and ticks or the horizontal gridlines	*/
	if( wi->htickFlag<= 0 && !wi->polarFlag ){
	  /* gridlines in non-polar mode are drawn in a special colour/style	*/
		AllAttrs[1].pixelValue= gridPixel;
		AllAttrs[1].lineStyleLen= gridLSLen;
		memcpy( AllAttrs[1].lineStyle, gridLS, gridLSLen* sizeof(char));
		lstyle= 1;
		lcolour= 1;
		ltype= L_VAR;
	}

	  /* The average Y being shown. If the Yincr is smaller than what will be shown (currently %.3lf or %.2le -
	   \ giving Yincr>0.01 is shown), we will subtract the average Y (bias) from each number before
	   \ printing it. For this reason, Yincr is first determined based upon the real YUnitsPerPixel,
	   \ and not upon R_YunitsPerPixel which excludes (only) TRANSFORM_[XY].
	   */
	Yincr = (wi->dev_info.axis_pad + Y_axis_height) * wi->YUnitsPerPixel;
	Ybias= (ABS(Yincr)* Yincr_factor < wi->Ybias_thres )? ( wi->win_geo.bounds._hiY + wi->win_geo.bounds._loY )/ 2.0 : 0;
	if( debugFlag && Ybias ){
		fprintf( StdErr, "DrawGridAndAxis(): Y=[%g,%g], dY=%g => bias=%g\n",
			wi->win_geo.bounds._loY, wi->win_geo.bounds._hiY, Yincr* Yincr_factor , Ybias
		);
	}

	if( (int) Yincr_factor != Yincr_factor ){
		gridscale= Yincr_factor;
		numscale= 1.0;
	}
	else{
		gridscale= 1.0;
		numscale= Yincr_factor;
	}
	Yincr = (wi->dev_info.axis_pad + Y_axis_height)* gridscale * wi->R_YUnitsPerPixel;
	Ystart = initGrid(wi, _R_UsrOrgY, Yincr, LogYFlag, SqrtYFlag, Y_axis, 0);
	Xindex= gridBase+ gridStep;
	if( Xindex- gridBase> 0 ){
		axis_ok= True;
	}
	else{
		axis_ok= False;
	}
	last_value[0]= '\0';

	_YLabelLength= 0;
	if( !wi->axisFlag || (wi->bbFlag && wi->bbFlag!= -1) ){
		lspot= wi->XOrgY;
		hspot= wi->XOppY;
	}
	else{
		if( Y_lspot< 0 || Y_hspot< 0 ){
			Y_lspot= lspot= wi->XOrgY+ 2* wi->dev_info.bdr_pad;
			Y_hspot= hspot= MAX( 0, wi->XOppY- 2* wi->dev_info.bdr_pad);
			wi->win_geo.nobb_loY= Reform_Y( wi, TRANY(hspot), TRANX(wi->XOrgX) );
			wi->win_geo.nobb_hiY= Reform_Y( wi, TRANY(lspot), TRANX(wi->XOrgX) );
		}
		else{
			lspot= Y_lspot;
			hspot= Y_hspot;
		}
	}
	Yspot= wi->XOrgY;

	wi->axis_stuff.rawY.last_index= 0;
	wi->axis_stuff.Y.last_index= 0;

	  /* 20010723: moved Mval computation out of loop; val must be calculation for both
	   \ signs of Yincr...! Also, made bound (Mval) inclusive.
	   */
	Mval=Y_Value(wi,_R_UsrOppY);
	for( j= 0, Yindex = Ystart, last_spot= SCREENY(wi, Yindex)-1;
		axis_ok && wi->axisFlag &&
		(	(Yincr> 0 && (val=Y_Value(wi,Yindex)) <= Mval) ||
/* 			(Yincr< 0 && val > Mval)	*/
			(Yincr< 0 && (val=Y_Value(wi,Yindex)) >= Mval)
		) ;
		Yindex = stepGrid(1.0), j++
	){
	  int zero_tick, draw_line= 1;
		T_Yindex= Yindex;
		if( (zero_tick= (ABS(T_Yindex) < zero_thres) && (wi->logYFlag== 0 || wi->logYFlag== -1)) ){
			T_Yindex= 0;
		}
		if( debugFlag ){
			fprintf( StdErr, "[%d,%s", j, d2str( T_Yindex, "%g", NULL) );
			fflush( StdErr );
		}
		if( zero_tick || fmod( (double)j, numscale )== 0 || (logYFlag && wi->_log_zero_y && Yindex== wi->log10_zero_y) ){
#ifdef DEBUG
			if( !j || abs(last_spot- Yspot)> Y_axis_height || is_log_zero || is_log_zero2 ){
				fputc( '\0', stdout);
			}
#endif
			if( !wi->transform_axes ){
			  double x= _R_UsrOrgX, ly= T_Yindex, hy= T_Yindex;
				do_TRANSFORM( wi, 0, 1, 1, &x, NULL, NULL, &T_Yindex, &ly, &hy, 1, True );
			}
#ifdef DEBUG
			if( !j || abs(last_spot- Yspot)> Y_axis_height || is_log_zero || is_log_zero2 ){
				fputc( '\0', stdout);
			}
#endif
			if( YLabels== 0 ){
				Ystart= T_Yindex;
			}
			vcatfont= NULL;
			if( !wi->polarFlag ){
			  /* Make sure we put the labels at as exactly the right spot as we can..	*/
				if( wi->ValCat_Y_axis ){
				  double y= Yindex- Ybias;
					  /* ValCat requested: see if we have a category matching the current axis "tickvalue".
					   \ We will have to do the reform to get at the real value. Or a WriteValue to
					   \ get at the "exact" value - i.e. the rounded off that would be printed
					   \ if not for the ValCat_Y_axis flag.
					   */
					WriteValue(wi, value, y, TRANX(wi->XOrgX), expY,
						LogYFlag, SqrtYFlag, Y_axis, UseRealYVal, YRange/YLabels,
						sizeof(value)
					);
					{ int n= 1;
						fascanf2( &n, value, &y, ',' );
					}
					if( wi->exact_Y_axis ){
						vcat= Find_ValCat( wi->ValCat_Y, y, NULL, NULL );
						mincat= maxcat= NULL;
					}
					else{
						vcat= Find_ValCat( wi->ValCat_Y, y, &mincat, &maxcat);
					}
					if( vcat && vcat->category ){
					  /* Found a matching category, use it instead of the value (Yindex-Ybias)	*/
						strncpy( value, vcat->category, sizeof(value)-1 );
						vcatfont= wi->ValCat_YFont;
					}
					else{
						if( (mincat || maxcat) ){
						  /* Use the closest neighbouring category instead	*/
							if( !mincat ){
								mincat= maxcat;
							}
							else if( !maxcat ){
								maxcat= mincat;
							}
							if( y- mincat->val< maxcat->val- y ){
								vcat= mincat;
							}
							else{
							  /* By default, we will use the upper neighbour in case they're both
							   \ equally distant.
							   */
								vcat= maxcat;
							}
							  /* Retrieve the corresponding tick-value/position:	*/
							y= Trans_Y( wi, vcat->val, True);
							if( debugFlag ){
								fprintf( StdErr, " [%s at Y=%g not found: using to \"%s\" at Y=%s] ",
									d2str(y, NULL, NULL), T_Yindex, vcat->category, d2str(y, NULL, NULL) );
								fflush( StdErr );
							}
							strncpy( value, vcat->category, sizeof(value)-1 );
							T_Yindex= y;
							vcatfont= wi->ValCat_YFont;
						}
						else{
							value[0]= '\0';
							draw_line= 0;
							if( debugFlag ){
								fprintf( StdErr, " [value %s not found in Y Value<>Categories] ", d2str(y, NULL,NULL) );
								fflush( StdErr );
							}
						}
					}
					  /* There is no reason why a ValCat label should not equal the previous...	*/
					last_value[0]= '\0';
				}
				else{
					WriteValue(wi, value, Yindex- Ybias, TRANX(wi->XOrgX), expY,
						LogYFlag, SqrtYFlag, Y_axis, UseRealYVal, YRange/YLabels,
						sizeof(value)
					);
					if( wi->exact_Y_axis &&
						!( wi->logYFlag>0 && strcmp( value, (wi->lz_sym_y)? wi->log_zero_sym_y : "0*")==0) &&
						(!wi->transform.y_len || RAW_DISPLAY(wi) || wi->transform_axes>0)
					){
					  double y;
					  int ys;
						  /* Thus, reconvert the rounded representation of T_Yindex to a double	*/
						{ int n= 1;
							fascanf2( &n, value, &y, ',' );
						}
						y= Trans_Y( wi, y/ wi->Yscale, True);
						if( (ys= SCREENY( wi, y ))>= lspot && ys<= hspot ){
							if( debugFlag ){
								fprintf( StdErr, " [\"%s\" at Y=%s corrected to Y=%s] ",
									value, d2str(T_Yindex, NULL, NULL), d2str(y, NULL, NULL)
								);
								fflush( StdErr );
							}
							T_Yindex= y;
						}
					}
				}
			}
			Yspot = SCREENY(wi, T_Yindex);
			if( Yspot>= lspot && Yspot<= hspot &&
				(( wi->axis_stuff.log_zero_y_spot== LOG_ZERO_LOW && Yindex>= wi->log10_zero_y) ||
					(wi->axis_stuff.log_zero_y_spot== LOG_ZERO_HIGH && Yindex<= wi->log10_zero_y) ||
					wi->axis_stuff.log_zero_y_spot== LOG_ZERO_INSIDE
				)
			){

					AddAxisValue( wi, &wi->axis_stuff.Y, T_Yindex );

					  /* Draw the grid line or tick mark */
					zero_tick= (ABS(T_Yindex) < zero_thres) && (wi->logYFlag== 0 || wi->logYFlag== -1);
					if( (wi->htickFlag> 0 && (!(zero_tick && wi->zeroFlag) || wi->ValCat_Y_axis) ) ||
						(wi->polarFlag /* && wi->win_geo.polar_axis== Y_axis */) ||
						(wi->polarFlag && zero_tick)
					){
						segs[0].x1 = wi->XOrgX;
						segs[0].x2 = wi->XOrgX + wi->dev_info.tick_len;
						segs[1].x1 = wi->XOppX - wi->dev_info.tick_len;
						segs[1].x2 = wi->XOppX;
						segs[0].y1 = segs[0].y2 = segs[1].y1 = segs[1].y2 = Yspot;
					} else if( !wi->polarFlag ){
						segs[0].x1 = wi->XOrgX;  segs[0].x2 = wi->XOppX;
						segs[0].y1 = segs[0].y2 = Yspot;
					}
					else{
						draw_line= 0;
					}
					  /* zero_thres below used to be ZERO_THRES (float constant). Somehow under A/UX
					   * gcc -O generated a float constant here that choked the GNU assembler
					   ** RJB **
					   */
					if( wi->axisFlag && DoIt && draw_line ){
						  /* Draw <zeroColour> ticks at zero when not drawing a grid, or draw a full
						   \ line when zeroFlag=True. This code seems a little too complex, but I am
						   \ not in the mood to simplify it right now.
						   \ Well - the more complex code has been deactivated by the if( 0...
						   */
						if( 0 && (zero_tick && (wi->htickFlag> 0 || wi->zeroFlag)) && !wi->ValCat_Y_axis ){
							if( !wi->polarFlag && (wi->htickFlag<= 0 || wi->zeroFlag) ){
							 XSegment line= segs[0];
								if( wi->htickFlag> 0 ){
									line.x2= segs[1].x2;
								}
								wi->dev_info.xg_seg(wi->dev_info.user_state,
										 1, &line, gridWidth+zeroWidth,
										 L_ZERO, 0, 0, 0, NULL
								);
							}
							else{
								wi->dev_info.xg_seg(wi->dev_info.user_state,
										 1, segs, gridWidth+zeroWidth,
										 L_ZERO, 0, 0, 0, NULL
								);
								if( wi->bbFlag && wi->bbFlag!= -1 ){
									if( ((wi->htickFlag && wi->htickFlag!= -1) || wi->polarFlag) ){
										wi->dev_info.xg_seg(wi->dev_info.user_state,
												 1, &(segs[1]), gridWidth+zeroWidth,
												 L_ZERO, 0, 0, 0, NULL
										);
									}
								}
							}
						} else {
						  double w; int lt, ls, lc, zt;
							if( zero_tick && !wi->ValCat_Y_axis && (wi->htickFlag> 0 || wi->zeroFlag) ){
								w= gridWidth+ zeroWidth;
								lt= L_ZERO;
								ls= 0;
								lc= 0;
								zt= wi->zeroFlag;
							}
							else{
								w= gridWidth;
								lt= ltype;
								ls= lstyle;
								lc= lcolour;
								zt= False;
							}
							wi->dev_info.xg_seg(wi->dev_info.user_state,
									 1, segs, w, lt, ls, lc, 0, NULL
							);
							if( wi->bbFlag && wi->bbFlag!= -1 && !zt ){
								if( (wi->htickFlag && wi->htickFlag!= -1) || wi->polarFlag ){
									wi->dev_info.xg_seg(wi->dev_info.user_state,
											 1, &(segs[1]), w, lt, ls, lc, 0, NULL
									);
								}
							}
						}
					}
					if( YLabels ){
						if( axymin== -1 ){
							axymin= Yspot;
						}
						if( axymax== -1 ){
							axymax= Yspot;
						}
						if( Yspot< axymin ){
							axymin= Yspot;
						}
						else if( Yspot> axymax ){
							axymax= Yspot;
						}
					}
					else{
						axymin= axymax= Yspot;
					}

				if( debugFlag ){
					fprintf( StdErr, "@%s,%d]", d2str(T_Yindex, "%g", NULL), Yspot );
					fflush( StdErr );
				}
				if( wi->polarFlag ){
					WriteValue(wi, value, Yindex- Ybias, 0.0,
						expY, LogYFlag, SqrtYFlag, Y_axis, UseRealYVal, YRange/YLabels,
						sizeof(value)
					);
				}
				  /* We check for overlap by ensuring that the pixel-distance between the last and
				   \ current spot is at least equal to the height of the axis font. The first label
				   \ is exempted from this check (last_spot may not be correct)
				   */
				if( !j || abs(last_spot- Yspot)> Y_axis_height || is_log_zero || is_log_zero2 ){
					  /* Write the axis label */
					if( wi->polarFlag ){
					  /* just like the non-polar case, the axis shows labels that really should be
					   \ alongside the X=0 axis.
					   */
						/* WriteValue done	*/
						if( wi->axisFlag && DoIt && wi->win_geo.polar_axis== Y_axis && wi->dev_info.xg_arc && !wi->htickFlag &&
							strcmp( value, last_value)
						){
						  int rad_x= SCREENXDIM( wi, ABS(T_Yindex) ),
							  rad_y= SCREENYDIM( wi, ABS(T_Yindex));
							if( rad_x<= polar_width && rad_y<= polar_height ){
							  /* Draw a gridlike set of circles in the appropriate colour/style	*/
								wi->dev_info.xg_arc( wi->dev_info.user_state,
									polar_dot_x, polar_dot_y, rad_x, rad_y,
									wi->win_geo.low_angle, wi->win_geo.high_angle,
									0, L_POLAR, 1, 1, 0
								);
							}
							if( debugFlag ){
								fprintf( StdErr, "DrawGridAndAxis(): Y-arc(%d,%d,%d,%d,%g,%g) clipped to width=%d height=%d\n",
									polar_dot_x, polar_dot_y, rad_x, rad_y,
									wi->win_geo.low_angle, wi->win_geo.high_angle,
									polar_width, polar_height
								);
								fflush( StdErr );
							}
						}
					}
					else{
						/* WriteValue done	*/
					}
					if( !use_X11Font_length ){
						_YLabelLength= MAX( strlen(value), _YLabelLength);
					}
					else{
					  int temp= XGTextWidth(wi, value, T_AXIS, vcatfont );
						if( use_X11Font_length && vcatfont && !_use_gsTextWidth ){
						  /* to correct for the division done below..	*/
							if( !_use_gsTextWidth ){
								temp*= XFontWidth(axisFont.font)/ XFontWidth( vcatfont->XFont.font);
							}
							else{
								temp*= XFontWidth(axisFont.font);
							}
						}
						_YLabelLength= MAX( temp, _YLabelLength);
					}
			/* 		value[YLabelLength]= '\0';	*/
					if( wi->axisFlag && DoIt && strcmp( value, last_value) ){
						AxisValueCurrentLabelled( wi, &wi->axis_stuff.Y, 1 );
						wi->dev_info.xg_text(wi->dev_info.user_state,
									 wi->XOrgX- wi->dev_info.bdr_pad,
									 Yspot, value, T_RIGHT, T_AXIS, vcatfont
						);
						  /* extend the tick outside of the axis, to show which tick belongs
						   \ to this label.
						   */
						if( draw_line ){
							segs[0].x1 = wi->XOrgX;
							segs[0].x2 = wi->XOrgX - wi->dev_info.tick_len/3;
							segs[0].y1 = segs[0].y2 = Yspot;
							wi->dev_info.xg_seg(wi->dev_info.user_state,
									 1, segs, gridWidth,
									 L_AXIS, 0, 0, 0, NULL
							);
						}
					}
					strcpy( last_value, value);
					last_spot= Yspot;
				}
				else if( debugFlag ){
					fprintf( StdErr, "\"%s\"@%d skipped (%d<%d)",
						value, Yspot, abs(last_spot-Yspot), Y_axis_height
					);
				}
			}
		}
		else if( debugFlag ){
			fputc( '\n', StdErr );
		}
		  /* for later use:	*/
		T_Yindex= Yindex;
	}

	  /* valcat_vgrid is set to ValCat_X_grid when we're actually drawing (and not just determining
	   \ the layout). This grid will be drawn before the other ticks, texts etc. along the X axis
	   \ (i.e. in the background). Afterwards, valcat_vgrid will be reset to False, and we'll jump
	   \ back to the DrawXAxis label.
	   */
	valcat_vgrid= (DoIt)? wi->ValCat_X_grid : False;

DrawXAxis:;
	  /* Now, let us draw the X axis, and ticks or the horizontal gridlines	*/
	if( (wi->vtickFlag<= 0 && !wi->polarFlag) || (wi->ValCat_X_axis && valcat_vgrid) ){
	  /* gridlines in non-polar mode are drawn in a special colour/style	*/
	  /* The vertical gridlines of the ValCat_X category are drawn in the same style.	*/
		AllAttrs[1].pixelValue= gridPixel;
		AllAttrs[1].lineStyleLen= gridLSLen;
		memcpy( AllAttrs[1].lineStyle, gridLS, gridLSLen* sizeof(char));
		lstyle= 1;
		lcolour= 1;
		ltype= L_VAR;
	}
	else{
	  /* We (may) need to restore the default values:	*/
		AllAttrs[1].pixelValue= color1;
		AllAttrs[1].lineStyleLen= LSLen1;
		memcpy( AllAttrs[1].lineStyle, LS1, LSLen1* sizeof(char));
		lcolour= 0;
		lstyle= 0;
		ltype= L_AXIS;
	}

	if( wi->textrel.used_gsTextWidth && PS_STATE(wi)->Printing== PS_PRINTING ){
	  double xil;
		xil= (XLabelLength/XFontWidth(axisFont.font)+1.5)* _X_a_w;
		if( XLabelLength ){
			Xincr_LabelLength= xil/ __X_a_w/ ((double) XLabelLength/(double) _X_a_w);
		}
		else{
			Xincr_LabelLength= (double) XLabelLength/ (double) _X_a_w;
		}
	}
	else{
		Xincr_LabelLength= (double) XLabelLength/ (double) _X_a_w;
	}
	Xincr = (wi->dev_info.axis_pad + (Xincr_LabelLength * X_axis_width)) * wi->XUnitsPerPixel;
	Xbias= (ABS(Xincr)* Xincr_factor < wi->Xbias_thres )? ( wi->win_geo.bounds._hiX + wi->win_geo.bounds._loX )/ 2.0 : 0;
	if( debugFlag && Xbias ){
		fprintf( StdErr, "DrawGridAndAxis(): X=[%g,%g], dX=%g => bias=%g\n",
			wi->win_geo.bounds._loX, wi->win_geo.bounds._hiX, Xincr* Xincr_factor , Xbias
		);
	}

	if( (int) Xincr_factor != Xincr_factor ){
		gridscale= Xincr_factor;
		numscale= 1.0;
	}
	else{
		gridscale= 1.0;
		numscale= Xincr_factor;
	}
	Xincr = (wi->dev_info.axis_pad + (Xincr_LabelLength * X_axis_width))* gridscale * wi->R_XUnitsPerPixel;
	Xstart = initGrid(wi, _R_UsrOrgX, Xincr, LogXFlag, SqrtXFlag , X_axis, 0 );
	Xindex= gridBase+ gridStep;
	if( Xindex- gridBase> 0 ){
		axis_ok= True;
	}
	else{
		axis_ok= False;
	}
	last_value[0]= '\0';

	_XLabelLength= 0;
	Xspot= wi->XOrgX;
	xvcN= 0;
	for( j= 0; j< last_right_size/sizeof(int); j++ ){
		last_right[j]= 0;
	}
	last_width= 0;
	if( !wi->axisFlag || (wi->bbFlag && wi->bbFlag!= -1) ){
		lspot= wi->XOrgX+ x_margin;
		hspot= wi->XOppX;
	}
	else{
		if( X_lspot< 0 || X_hspot< 0 ){
			X_lspot= lspot= wi->XOrgX+ 2* wi->dev_info.bdr_pad;
			X_hspot= hspot= MAX( 0, wi->XOppX- 2* wi->dev_info.bdr_pad);
			wi->win_geo.nobb_loX= Reform_X( wi, TRANX(lspot), TRANY(wi->XOppY) );
			wi->win_geo.nobb_hiX= Reform_X( wi, TRANX(hspot), TRANY(wi->XOppY) );
		}
		else{
			lspot= X_lspot;
			hspot= X_hspot;
		}
	}

	wi->axis_stuff.rawX.last_index= 0;
	wi->axis_stuff.X.last_index= 0;

	  /* 20010723: see 20010723 comments at start of Y-axis drawing loop. */
	Mval=X_Value(wi,_R_UsrOppX);
	for( j= 0, Xindex = Xstart, last_spot= SCREENX(wi, Xindex)-1;
		axis_ok && wi->axisFlag &&
		(	(Xincr> 0 && (val=X_Value(wi,Xindex)) <= Mval) ||
			(Xincr< 0 && (val=X_Value(wi,Xindex)) >= Mval)
		) ;
		Xindex = stepGrid(1.0), j++
	){
	  int zero_tick;
		if( (zero_tick= (ABS(Xindex) < zero_thres) && !polarLog && (wi->logXFlag== 0 || wi->logXFlag== -1)) ){
			Xindex= 0;
		}
		if( zero_tick || fmod( (double)j, numscale )== 0 || (logXFlag && wi->_log_zero_x && Xindex== wi->log10_zero_x) ){
		  int zero_tick, draw_line= 1;
		  double T_Xindex= Xindex;
/* 			if( !wi->transform_axes && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) )	*/
			if( !wi->transform_axes )
			{
			  double y= _R_UsrOrgY, ly= y, hy= y;
				do_TRANSFORM( wi, 0, 1, 1, &T_Xindex, NULL, NULL, &y, &ly, &hy, 1, True );
			}
			if( XLabels== 0 ){
				Xstart= T_Xindex;
			}
			vcatfont= NULL;
			if( wi->polarFlag ){
				WriteValue(wi, value, Xindex- Xbias, 0.0,
					expX, LogXFlag, SqrtXFlag, X_axis, UseRealXVal, XRange/ XLabels,
					sizeof(value)
				);
			}
			else{
				if( wi->ValCat_X_axis ){
				  double x= Xindex- Xbias;
					WriteValue(wi, value, x, TRANY(wi->XOppY), expX,
						LogXFlag, SqrtXFlag, X_axis, UseRealXVal, XRange/ XLabels,
						sizeof(value)
					);
					{ int n= 1;
						fascanf2( &n, value, &x, ',' );
					}
					if( wi->exact_X_axis ){
						vcat= Find_ValCat( wi->ValCat_X, x, NULL, NULL );
						mincat= maxcat= NULL;
					}
					else{
						vcat= Find_ValCat( wi->ValCat_X, x, &mincat, &maxcat);
					}
					if( vcat && vcat->category){
						strncpy( value, vcat->category, sizeof(value)-1 );
						vcatfont= wi->ValCat_XFont;
					}
					else{
						if( (mincat || maxcat) ){
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
							x= Trans_X( wi, vcat->val);
							if( debugFlag ){
								fprintf( StdErr, " [%s at X=%g not found: using to \"%s\" at X=%s] ",
									d2str(x, NULL, NULL), T_Xindex, vcat->category, d2str(x, NULL, NULL) );
								fflush( StdErr );
							}
							strncpy( value, vcat->category, sizeof(value)-1 );
							vcatfont= wi->ValCat_XFont;
							T_Xindex= x;
						}
						else{
							if( debugFlag ){
								fprintf( StdErr, " [value %s not found in X Value<>Categories] ", d2str(x, NULL,NULL) );
								fflush( StdErr );
							}
							value[0]= '\0';
							draw_line= 0;
						}
					}
					  /* There is no reason why a ValCat label should not equal the previous...	*/
					last_value[0]= '\0';
				}
				else{
					WriteValue(wi, value, Xindex- Xbias, TRANY(wi->XOppY), expX,
						LogXFlag, SqrtXFlag, X_axis, UseRealXVal, XRange/ XLabels,
						sizeof(value)
					);
					if( wi->exact_X_axis &&
						!( wi->logXFlag>0 && strcmp( value, (wi->lz_sym_x)? wi->log_zero_sym_x : "0*")==0) &&
						(!wi->transform.x_len || RAW_DISPLAY(wi) || wi->transform_axes>0)
					){
					  double x;
					  int xs;
						{ int n= 1;
							fascanf2( &n, value, &x, ',' );
						}
						x= Trans_X( wi, x/ wi->Xscale);
						if( (xs= SCREENX( wi, x))>= lspot && xs<= hspot ){
							if( debugFlag ){
								fprintf( StdErr, " [\"%s\" at X=%s corrected to X=%s] ",
									value, d2str(T_Xindex, NULL, NULL), d2str(x, NULL, NULL)
								);
								fflush( StdErr );
							}
							T_Xindex= x;
						}
					}
				}
			}
			Xspot = SCREENX(wi, T_Xindex);
			if( Xspot>= lspot && Xspot<= hspot  &&
				(( wi->axis_stuff.log_zero_x_spot== LOG_ZERO_LOW && Xindex>= wi->log10_zero_x) ||
					(wi->axis_stuff.log_zero_x_spot== LOG_ZERO_HIGH && Xindex<= wi->log10_zero_x) ||
					wi->axis_stuff.log_zero_x_spot== LOG_ZERO_INSIDE
				)
			){
				AddAxisValue( wi, &wi->axis_stuff.X, T_Xindex );

				  /* Draw the grid line or tick marks */
				zero_tick= (ABS(T_Xindex) < zero_thres) && !polarLog && (wi->logXFlag== 0 || wi->logXFlag== -1);
				if( (wi->vtickFlag> 0 && (!(zero_tick && wi->zeroFlag) || wi->ValCat_X_axis) ) ||
					(wi->polarFlag /* && wi->win_geo.polar_axis== X_axis */) ||
					(wi->polarFlag && zero_tick)
				){
					segs[0].x1 = segs[0].x2 = segs[1].x1 = segs[1].x2 = Xspot;
					segs[0].y1 = wi->XOppY;
					segs[0].y2 = wi->XOppY - wi->dev_info.tick_len;
					segs[1].y1 = wi->XOrgY;
					segs[1].y2 = wi->XOrgY + wi->dev_info.tick_len;
				} else if( !wi->polarFlag ){
					segs[0].x1 = segs[0].x2 = Xspot;
					segs[0].y1 = wi->XOrgY; segs[0].y2 = wi->XOppY;
				}
				else{
					draw_line= 0;
				}
				  /* Draw the ticks or vertical grid when it's time	*/
				if( wi->axisFlag && DoIt && draw_line && !(wi->ValCat_X_axis && valcat_vgrid) ){
					if( 0 && (zero_tick && (wi->vtickFlag> 0 || wi->zeroFlag)) && !wi->ValCat_X_axis ){
						if( !wi->polarFlag && (wi->vtickFlag<= 0 || wi->zeroFlag) ){
						 XSegment line= segs[1];
							if( wi->vtickFlag> 0 ){
								line.y2= segs[0].y1;
							}
							wi->dev_info.xg_seg(wi->dev_info.user_state,
									 1, &line, gridWidth+zeroWidth, L_ZERO, 0, 0, 0, NULL
							);
						}
						else{
							wi->dev_info.xg_seg(wi->dev_info.user_state,
									 1, segs, gridWidth+zeroWidth, L_ZERO, 0, 0, 0, NULL
							);
							if( wi->bbFlag && wi->bbFlag!= -1 ){
								if( /* !wi->zeroFlag && */ ((wi->vtickFlag && wi->vtickFlag!= -1) || wi->polarFlag) ){
									wi->dev_info.xg_seg(wi->dev_info.user_state,
											 1, &(segs[1]), gridWidth+zeroWidth, L_ZERO, 0, 0, 0, NULL
									);

								}
							}
						}
					} else {
					  double w; int lt, ls, lc, zt;
						if( zero_tick && !wi->ValCat_X_axis && (wi->vtickFlag> 0 || wi->zeroFlag) ){
							w= gridWidth+ zeroWidth;
							lt= L_ZERO;
							ls= 0;
							lc= 0;
							zt= wi->zeroFlag;
						}
						else{
							w= gridWidth;
							lt= ltype;
							ls= lstyle;
							lc= lcolour;
							zt= False;
						}
						wi->dev_info.xg_seg(wi->dev_info.user_state,
								 1, segs, w, lt, ls, lc, 0, NULL
						);
						if( wi->bbFlag && wi->bbFlag!= -1 && !zt ){
							if( (wi->vtickFlag && wi->vtickFlag!= -1) || wi->polarFlag ){
								wi->dev_info.xg_seg(wi->dev_info.user_state,
										 1, &(segs[1]), w, lt, ls, lc, 0, NULL
								);
							}
						}
					}
				}
				if( XLabels ){
					if( axxmin== -1 ){
						axxmin= Xspot;
					}
					if( axxmax== -1 ){
						axxmax= Xspot;
					}
					if( Xspot< axxmin ){
						axxmin= Xspot;
					}
					else if( Xspot> axxmax ){
						axxmax= Xspot;
					}
				}
				else{
					axxmin= axxmax= Xspot;
				}
				  /* First check against 100% overlap or some other things	*/
				if( !(!j || last_spot!= Xspot || is_log_zero || is_log_zero2 ) ){
				  /* Textual overlap with the previous label: skip (but we did draw
				   \ the tick/gridline).
				   */
					continue;
				}
				  /* Write the axis label */
				if( wi->polarFlag ){
					/* WriteValue done	*/
					if( wi->axisFlag && DoIt && wi->win_geo.polar_axis== X_axis && wi->dev_info.xg_arc &&
						!(wi->htickFlag || wi->vtickFlag) && strcmp( value, last_value)
					){
					  int rad_x= SCREENXDIM( wi, ABS(T_Xindex) ),
						  rad_y= SCREENYDIM( wi, ABS(T_Xindex));
						if( rad_x<= polar_width && rad_y<= polar_height ){
							wi->dev_info.xg_arc( wi->dev_info.user_state,
								polar_dot_x, polar_dot_y, rad_x, rad_y,
								wi->win_geo.low_angle, wi->win_geo.high_angle,
								0, L_POLAR, 1, 1, 0
							);
						}
						if( debugFlag ){
							fprintf( StdErr, "DrawGridAndAxis(): X-arc(%d,%d,%d,%d,%g,%g) clipped to width=%d height=%d\n",
								polar_dot_x, polar_dot_y, rad_x, rad_y,
								wi->win_geo.low_angle, wi->win_geo.high_angle,
								polar_width, polar_height
							);
							fflush( StdErr );
						}
					}
				}
				else{
					/* WriteValue done	*/
				}
				if( !use_X11Font_length ){
					last_width= strlen(value);
					_XLabelLength= MAX( last_width, _XLabelLength);
				}
				else{
					last_width= XGTextWidth(wi, value, T_AXIS, vcatfont );
					if( use_X11Font_length && vcatfont && !_use_gsTextWidth ){
					  /* to correct for the division done below..	*/
						if( !_use_gsTextWidth ){
							last_width*= XFontWidth(axisFont.font)/ XFontWidth( vcatfont->XFont.font);
						}
						else{
							last_width*= XFontWidth(axisFont.font);
						}
					}
					_XLabelLength= MAX( last_width, _XLabelLength);
				}
				if( wi->axisFlag && DoIt && strcmp( value, last_value) ){
				  int just, Yloc, right, left, level= 0, lr;
					  /* 20010105: last_width here was _XLabelLength	*/
					last_width+= wi->dev_info.bdr_pad/4;
					if( Xspot+ last_width/ 2> wi->dev_info.area_w ){
						just= T_UPPERRIGHT;
						right= Xspot;
						left= Xspot- last_width;
					}
					else if( Xspot- last_width/ 2< 0 ){
						just= T_UPPERLEFT;
						right= Xspot+ last_width;
						left= Xspot;
					}
					else{
						just= T_TOP;
						right= Xspot+ last_width/2;
						left= Xspot- last_width/2;
					}
					if( wi->ValCat_X_axis && wi->axisFlag ){
						if( wi->ValCat_X_levels< 0 ){
						  int vxl= - wi->ValCat_X_levels;
							Yloc= wi->XOppY + 2*wi->dev_info.bdr_pad+
								X_axis_height* ((vxl- 1)- (level= (xvcN % vxl)));
						}
						else{
							Yloc= wi->XOppY + 2*wi->dev_info.bdr_pad+ X_axis_height* (level= (xvcN % wi->ValCat_X_levels));
						}
					}
					else{
						Yloc= wi->XOppY + 2*wi->dev_info.bdr_pad;
					}
					if( left<= (lr= last_right[level]) ){
					  /* Textual overlap with the previous label: skip (but we did draw
					   \ the tick/gridline).
					   */
						continue;
					}
					last_right[level]= right;
					  /* It seems this label would be drawn. Do some final checks, and if
					   \ so requested, draw a small vertical gridline that will easy the perceptual
					   \ linking between a ValCat category label and the X-axis position it belongs
					   \ to.
					   */
					if( valcat_vgrid && wi->axisFlag && wi->ValCat_X_axis && *value && strcmp( value, last_value) ){
						segs[0].x1 = segs[0].x2 = Xspot;
						segs[0].y1 = wi->XOppY;
						segs[0].y2 = wi->XOppY + 2*wi->dev_info.bdr_pad+ X_axis_height* (xvcN % wi->ValCat_X_levels);
						wi->dev_info.xg_seg(wi->dev_info.user_state,
								 1, segs, gridWidth, ltype, lstyle, lcolour, 0, NULL
						);
						  /* Now do some house-keeping, and skip to the next grid position	*/
						xvcN+= 1;
						strcpy( last_value, value );
						continue;
					}

					AxisValueCurrentLabelled( wi, &wi->axis_stuff.X, 1 );

					wi->dev_info.xg_text(wi->dev_info.user_state,
								 Xspot, Yloc,
								 value, just, T_AXIS, vcatfont
					);
					xvcN+= 1;
					if( draw_line ){
						segs[0].x1 = segs[0].x2 = Xspot;
						segs[0].y1 = wi->XOppY;
						segs[0].y2 = wi->XOppY + wi->dev_info.tick_len/2;
						wi->dev_info.xg_seg(wi->dev_info.user_state,
								 1, segs, gridWidth, L_AXIS, 0, 0, 0, NULL
						);
					}
				}
				strcpy( value, last_value );
				last_spot= Xspot;
			}
			else if( debugFlag ){
				fprintf( StdErr, "\"%s\"@%d skipped (%d<%d)",
					value, Xspot, abs(last_spot-Xspot), X_axis_width
				);
			}
		}
	}
	if( wi->ValCat_X_axis && valcat_vgrid ){
	  /* So we just drew that ValCat vertical grid. Time to do the rest
	   \ of the X axis. Unset valcat_vgrid, and skip back to where it all
	   \ starts.
	   */
		valcat_vgrid= False;
		goto DrawXAxis;
	}

	if( wi->polarFlag> 0 && wi->axisFlag && DoIt && !(wi->htickFlag || wi->vtickFlag) ){
		  /* Draw some radial lines	*/
		Xstart= initGrid(wi, wi->win_geo.low_angle,
			XGpow(wi->win_geo.high_angle - wi->win_geo.low_angle, 2)/(wi->radix * 8),
			LogXFlag, SqrtXFlag , X_axis, 1
		);
		Xindex= gridBase+ gridStep;
		if( Xindex- gridBase> 0 ){
			axis_ok= True;
		}
		else{
			axis_ok= False;
		}
		for( j= 0, Xindex = Xstart, last_spot= SCREENX(wi, Xindex)-1;
			axis_ok &&
			(	(Xincr> 0 && (val=X_Value(wi,Xindex)) <= (Mval=X_Value(wi,wi->win_geo.high_angle))) ||
				(Xincr< 0 && val >= Mval)
			) ;
			Xindex = stepGrid(1.0), j++
		){
		  double T_Xindex= Xindex;
/* 			if( !wi->transform_axes && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) )	*/
			if( !wi->transform_axes )
			{
			  double y= _R_UsrOrgY, ly= y, hy= y;
				do_TRANSFORM( wi, 0, 1, 1, &T_Xindex, NULL, NULL, &y, &ly, &hy, 1, True );
			}
			Gonio_Base( wi, wi->radix, wi->radix_offset );
			{ double x2= Yindex* Cos(T_Xindex), y2= Yindex* Sin(T_Xindex);
			  double x1= 0, y1= 0;
			  int mark_inside1, mark_inside2, clipcode1, clipcode2;
			  XSegment line;
				if( ClipWindow( wi, NULL, False, &x1, &y1, &x2, &y2, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
/* 					if( !wi->transform_axes && !((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val)) )	*/
					if( !wi->transform_axes )
					{
					  double ly= y2, hy= y2;
						do_TRANSFORM( wi, 0, 1, 1, &x2, NULL, NULL, &y2, &ly, &hy, 1, True );
					}
					line.x1= polar_dot_x;
					line.y1= polar_dot_y;
					line.x2= SCREENX(wi, x2);
					line.y2= SCREENY(wi, y2);
					wi->dev_info.xg_seg( wi->dev_info.user_state, 1, &line, gridWidth, L_POLAR, 1, 1, 0, NULL );
				}
			}
		}
	}

	AllAttrs[1].pixelValue= color1;
	AllAttrs[1].lineStyleLen= LSLen1;
	memcpy( AllAttrs[1].lineStyle, LS1, LSLen1* sizeof(char));

	  /* Check to see if he wants a bounding box */
	if( /* wi->axisFlag && */ DoIt ){
	  XSegment bb[5];
		if( wi->bbFlag && wi->bbFlag!= -1 ){
			/* Draw bounding box */
			bb[0].x1 = bb[0].x2 = bb[1].x1 = bb[3].x2 = wi->XOrgX;
			bb[0].y1 = bb[2].y2 = bb[3].y1 = bb[3].y2 = wi->XOrgY;
			bb[1].x2 = bb[2].x1 = bb[2].x2 = bb[3].x1 = wi->XOppX;
			bb[0].y2 = bb[1].y1 = bb[1].y2 = bb[2].y1 = wi->XOppY;
			  /* 5th point to "close" the box, rounding the corner to
			   \ have a correct joint there.
			   */
			bb[4].x1= bb[3].x2; bb[4].y1= bb[3].y2;
			bb[4].x2= bb[4].x1; bb[4].y2= bb[4].y1+ wi->dev_info.bdr_pad;
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						 5, bb, axisWidth, L_AXIS, 0, -1, axisPixel, NULL);
		}
		else if( wi->axisFlag ){
		  int xmi, xma, ymi, yma;
		  int aw= (int) axisWidth* wi->dev_info.var_width_factor/ 2;
			xmi= axxmin- wi->dev_info.bdr_pad;
			if( X_lspot< xmi ){
				xmi= X_lspot;
			}
			xma= axxmax+ wi->dev_info.bdr_pad;
			if( X_hspot> xma ){
				xma= X_hspot;
			}
			  /* Determine the co-ordinates of the line. These should be
			   \ sharp, so we correct for the line's width (halfwidth stored in aw)
			   */
			bb[1].x1= bb[1].x2= bb[0].x1= xmi- aw;
			bb[0].y1= wi->XOppY;
			bb[2].x1= bb[2].x2= bb[0].x2= xma+ aw;
			bb[0].y2= wi->XOppY;
			bb[2].y1= bb[1].y1= wi->XOppY- aw /* - wi->dev_info.tick_len/ 2	*/;
			bb[2].y2= bb[1].y2= wi->XOppY+ wi->dev_info.tick_len/ 2;
			  /* First draw the horizontal line, the proper axis: */
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						 1, &bb[0], axisWidth, L_AXIS, 0, -1, axisPixel, NULL);
			  /* Then draw the two little stops: */
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						 1, &bb[1], axisWidth, L_AXIS, 0, -1, axisPixel, NULL);
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						 1, &bb[2], axisWidth, L_AXIS, 0, -1, axisPixel, NULL);
			bb[0].x1= wi->XOrgX;
/* 			ymi= MIN(axymin- wi->dev_info.bdr_pad, wi->XOppY- wi->dev_info.tick_len-  wi->dev_info.bdr_pad );	*/
/* 			yma= MAX(axymax+ wi->dev_info.bdr_pad, wi->XOrgY+ wi->dev_info.tick_len+  wi->dev_info.bdr_pad );	*/
			ymi= axymin- wi->dev_info.bdr_pad;
			if( Y_lspot< ymi ){
				ymi= Y_lspot;
			}
			yma= axymax+ wi->dev_info.bdr_pad;
			if( Y_hspot> yma ){
				yma= Y_hspot;
			}
			bb[1].y1= bb[1].y2= bb[0].y1= ymi- aw;
			bb[0].x2= wi->XOrgX;
			bb[2].y1= bb[2].y2= bb[0].y2= yma+ aw;
			bb[2].x1= bb[1].x1= wi->XOrgX- wi->dev_info.tick_len/ 2;
			bb[2].x2= bb[1].x2= wi->XOrgX+ aw /* + wi->dev_info.tick_len/ 2	*/;
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						 1, &bb[0], axisWidth, L_AXIS, 0, -1, axisPixel, NULL);
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						 1, &bb[1], axisWidth, L_AXIS, 0, -1, axisPixel, NULL);
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						 1, &bb[2], axisWidth, L_AXIS, 0, -1, axisPixel, NULL);
		}
	}

      /*
       * With the powers computed,  we can draw the axis labels.
	   \ This used to be the first thing. Now, it is (one of) the
	   \ last. This is because floating unitnames have a small rectangle
	   \ erased around them before being displayed (we don't want lines
	   \ through them; grid numbers are typically more redundant too).
       */
	if( wi->absYFlag ){
		sprintf( final, "%s|%s|", (Ybias)? "(" : "", YLABEL(wi) );
	}
	else{
		sprintf( final, "%s%s", (Ybias)? "(" : "", YLABEL(wi) );
	}
	if( wi->Yscale!= 1.0 ){
		sprintf( value, " \\%c\\ %g", (strstr(label_greekFont.name, "symbol"))? 180 : 'x' , wi->Yscale);
		strcat( final, value );
	}
	if( wi->yname_placed ){
		Xspot= SCREENX(wi, wi->tr_yname_x);
		Yspot= SCREENY(wi, wi->tr_yname_y);
	}
	else{
		Xspot= wi->XOrgX- wi->dev_info.bdr_pad- (YLabelLength* Y_axis_width);
		 if( wi->legend_always_visible ){
			 Yspot= wi->XOrgY- wi->dev_info.bdr_pad- wi->dev_info.label_height;
		 }
		 else{
			if( !wi->no_title && titles ){
				Yspot = wi->dev_info.bdr_pad * 2 + titles* wi->dev_info.title_height;
			}
			else{
				Yspot = wi->dev_info.bdr_pad * 2 ;
			}
		 }
	}
	if( Ybias< 0 ){
		sprintf( value, " + %g)", ABS(Ybias) );
		strcat( final, value );
	}
	else if( Ybias> 0 ){
		sprintf( value, " - %g)", Ybias );
		strcat( final, value );
	}
	if( ! *final ){
		/* do nothing	*/
	}
    else if( expY != 0 ){
	  int width, maxFontWidth= XFontWidth(labelFont.font);
		sprintf( value, " \\%c\\ 10", (strstr(label_greekFont.name, "symbol"))? 184 : 247 );
		strcat( final, value );
		sprintf(power, "%d", expY);
		if( !use_X11Font_length ){
			width= strlen(final)* wi->dev_info.label_width;
		}
		else{
			width= XGTextWidth(wi, final, T_LABEL, NULL);
			if( !_use_gsTextWidth ){
				if( xtb_has_greek( final) ){
					maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
				}
				width= (int)( width* ((double)wi->dev_info.label_width/ maxFontWidth ));
			}
		}
		  /* Xspot,Yspot represent the upperleft corner of the Ylabel, disregarding
		   \ the exponent. Due to the "clever" printing fashion (the mantissa centre-right-just.;
		   \ the exponent bottom-left-just. - BREAKS ON (POSTSCRIPT) PROPORTIONAL FONTS!), we
		   \ must translate this coordinate.
		   */
		Xspot+= width;
		Yspot += wi->dev_info.label_height/2;
		if( DoIt ){
			if( wi->yname_placed ){
			  int pwidth;
				if( ! *power ){
					pwidth= 0;
				}
				else if( !use_X11Font_length ){
					pwidth= strlen(power)* wi->dev_info.label_width;
				}
				else{
					pwidth= XGTextWidth(wi, power, T_LABEL, NULL);
					if( !_use_gsTextWidth ){
						if( xtb_has_greek( power) ){
							maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
						}
						pwidth= (int)( pwidth* ((double)wi->dev_info.label_width/ maxFontWidth ));
					}
				}
			     /* The clearing box assumes that the power is 10 characters of maxFontWidth wide.
				  \ Would be easy to determine its (exact) width (like of the final string above),
				  \ but that's too much asked for now.
				  */
				if( !(/* PS_STATE(wi)->Printing== PS_PRINTING && */ ps_transparent) ){
					if( wi->yname_vertical ){
						wi->dev_info.xg_clear( wi->dev_info.user_state,
							Xspot- width- wi->dev_info.bdr_pad/2, Yspot- wi->dev_info.label_height- wi->dev_info.bdr_pad/2,
							2* wi->dev_info.label_height+ wi->dev_info.bdr_pad,
							-(width+ 2* wi->dev_info.bdr_pad+ pwidth), 0, 0
						);
					}
					else{
						wi->dev_info.xg_clear( wi->dev_info.user_state,
							Xspot- width- wi->dev_info.bdr_pad/2, Yspot- wi->dev_info.label_height- wi->dev_info.bdr_pad/2,
							width+ 2* wi->dev_info.bdr_pad+ 10* maxFontWidth, 2* wi->dev_info.label_height+ wi->dev_info.bdr_pad, 0, 0
						);
					}
				}
			}
			wi->dev_info.xg_text(wi->dev_info.user_state,
						 Xspot, Yspot, final, T_RIGHT, T_LABEL, NULL);
			wi->dev_info.xg_text(wi->dev_info.user_state,
						 Xspot, Yspot, power, T_LOWERLEFT, T_LABEL, NULL);
		}
    } else {
		if( wi->polarFlag)
			strcat( final, "(radius)");
		if( wi->logYFlag){
			if( wi->polarFlag || !UseRealYVal )
				strcat( final, "(log)");
		}
		if( !wi->yname_placed ){
			if( wi->sqrtYFlag){
				if( wi->polarFlag || !UseRealYVal ){
					sprintf( final, "%s (^%g)", final, wi->powYFlag );
				}
			}
		}
		if( DoIt ){
			if( wi->yname_placed ){
			  int width, maxFontWidth= XFontWidth(labelFont.font);
				if( !use_X11Font_length ){
					width= strlen(final)* wi->dev_info.label_width;
				}
				else{
					width= XGTextWidth(wi, final, T_LABEL, NULL);
					if( !_use_gsTextWidth ){
						if( xtb_has_greek( final) ){
							maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
						}
						width= (int)( width* ((double)wi->dev_info.label_width/ maxFontWidth ));
					}
				}
				if( !(/* PS_STATE(wi)->Printing== PS_PRINTING && */ ps_transparent) ){
					if( wi->yname_vertical ){
						wi->dev_info.xg_clear( wi->dev_info.user_state, Xspot- wi->dev_info.bdr_pad/2, Yspot- wi->dev_info.bdr_pad/2,
							wi->dev_info.label_height+ wi->dev_info.bdr_pad,
							-(width+ 2* wi->dev_info.bdr_pad), 0, 0
						);
					}
					else{
						wi->dev_info.xg_clear( wi->dev_info.user_state, Xspot- wi->dev_info.bdr_pad/2, Yspot- wi->dev_info.bdr_pad/2,
							width+ 2* wi->dev_info.bdr_pad, wi->dev_info.label_height+ wi->dev_info.bdr_pad, 0, 0
						);
					}
				}
			}
			wi->dev_info.xg_text(wi->dev_info.user_state,
						 Xspot, Yspot, final,
						 (wi->yname_vertical)? T_VERTICAL : T_UPPERLEFT, T_LABEL, NULL);
		}
    }

	/* This controls placing of XLABEL	*/
/*     startX = wi->dev_info.area_w - wi->dev_info.bdr_pad;	*/
	startX = wi->XOppX - wi->dev_info.bdr_pad;
	sprintf( final, "%s%s", (Xbias)? "(" : "", XLABEL(wi) );
	if( wi->Xscale!= 1.0 ){
		sprintf( value, " \\%c\\ %g", (strstr(label_greekFont.name, "symbol"))? 180 : 'x', wi->Xscale);
		strcat( final, value );
	}
	if( !wi->xname_placed ){
		Xspot= startX;
		Yspot= wi->XOppY+ wi->dev_info.xname_vshift;
		  /* 20040129: should test for no_intensity_legend */
		if( !wi->no_intensity_legend && wi->IntensityLegend.legend_needed && !wi->IntensityLegend.legend_placed ){
			Yspot+= wi->IntensityLegend.legend_height;
		}
	}
	else{
		Xspot= SCREENX(wi, wi->tr_xname_x);
		Yspot= SCREENY(wi, wi->tr_xname_y);
	}
	if( Xbias< 0 ){
		sprintf( value, " - %g)", ABS(Xbias) );
		strcat( final, value );
	}
	else if( Xbias> 0 ){
		sprintf( value, " + %g)", Xbias );
		strcat( final, value );
	}
	if( ! *final ){
		/* Do nothing	*/
	}
	else if( expX != 0 ){
	  int width, maxFontWidth= XFontWidth(labelFont.font);
		sprintf(power, "%d", expX);
/* 		sprintf( value, " \\%c\\ 10", (strstr(label_greekFont.name, "symbol"))? 180 : 'x');	*/
		sprintf( value, " \\%c\\ 10", (strstr(label_greekFont.name, "symbol"))? 184 : 247 );
		strcat( final, value );
		if( !use_X11Font_length ){
			width= strlen(final)* wi->dev_info.label_width;
		}
		else{
			width= XGTextWidth(wi, final, T_LABEL, NULL);
			if( !_use_gsTextWidth ){
				if( xtb_has_greek( final) ){
					maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
				}
				width= (int)( width* ((double)wi->dev_info.label_width/ maxFontWidth ));
			}
		}
		if( DoIt ){
			if( wi->xname_placed ){
			  int pwidth;
				if( ! *power ){
					pwidth= 0;
				}
				else if( !use_X11Font_length ){
					pwidth= strlen(power)* wi->dev_info.label_width;
				}
				else{
					pwidth= XGTextWidth(wi, power, T_LABEL, NULL);
					if( !_use_gsTextWidth ){
						if( xtb_has_greek( power) ){
							maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
						}
						pwidth= (int)( pwidth* ((double)wi->dev_info.label_width/ maxFontWidth ));
					}
				}
				Xspot += width;
				Yspot += wi->dev_info.label_height/ 2;
				if( !(/* PS_STATE(wi)->Printing== PS_PRINTING && */ ps_transparent) ){
					wi->dev_info.xg_clear( wi->dev_info.user_state,
						Xspot- width- wi->dev_info.bdr_pad/2, Yspot- wi->dev_info.label_height- wi->dev_info.bdr_pad/2,
						width+ pwidth+ 2* wi->dev_info.bdr_pad, 2* wi->dev_info.label_height+ wi->dev_info.bdr_pad, 0, 0
					);
				}
			}
			else{
				Yspot -= wi->dev_info.label_height/2;
			}
			wi->dev_info.xg_text(wi->dev_info.user_state,
						 Xspot, Yspot,
						 power, T_LOWERLEFT, T_LABEL, NULL
			);
			wi->dev_info.xg_text(wi->dev_info.user_state,
						 Xspot, Yspot,
						 final, T_RIGHT, T_LABEL, NULL
			);
		}
    } else {
		if( wi->polarFlag){
			if( wi->powAFlag== 1.0 ){
				strcat( final, "(angle)");
			}
			else{
				sprintf( value, "(angle^%g)", wi->powAFlag );
				strcat( final, value );
			}
		}
		if( wi->logXFlag && !UseRealXVal ){
			strcat( final, "(log)");
		}
		if( wi->sqrtXFlag> 0 && !UseRealXVal ){
			sprintf( final, "%s (^%g)", final, wi->powXFlag );
		}
		if( wi->xname_placed ){
		  int width, maxFontWidth= XFontWidth(labelFont.font);
			if( !use_X11Font_length ){
				width= strlen(final)* wi->dev_info.label_width;
			}
			else{
				width= XGTextWidth(wi, final, T_LABEL, NULL);
				if( !_use_gsTextWidth ){
					if( xtb_has_greek( final) ){
						maxFontWidth= MAX( maxFontWidth, XFontWidth(label_greekFont.font) );
					}
					width= (int)( width* ((double)wi->dev_info.label_width/ maxFontWidth ));
				}
			}
			if( !(/* PS_STATE(wi)->Printing== PS_PRINTING && */ ps_transparent) ){
				wi->dev_info.xg_clear( wi->dev_info.user_state, Xspot- wi->dev_info.bdr_pad/2, Yspot- wi->dev_info.bdr_pad/2,
					width+ 2* wi->dev_info.bdr_pad, wi->dev_info.label_height+ wi->dev_info.bdr_pad, 0, 0
				);
			}
		}
		wi->dev_info.xg_text(wi->dev_info.user_state,
					 Xspot, Yspot,
					 final, (wi->xname_placed)? T_UPPERLEFT : T_LOWERRIGHT, T_LABEL, NULL
		);
    }

	if( use_X11Font_length && wi->textrel.used_gsTextWidth<= 0 ){
	  int w= XFontWidth(axisFont.font);
		_XLabelLength= (int)( (double) _XLabelLength / w + 1.5 );
		_YLabelLength= (int)( (double) _YLabelLength / w + 1.5 );
	}

	if( debugFlag ){
		fprintf( StdErr, "Current XLabelLength: %d / New: %d / Previous New: %d\n",
			XLabelLength, _XLabelLength, __XLabelLength
		);
		fprintf( StdErr, "Current YLabelLength: %d / New: %d / Previous New: %d\n",
			YLabelLength, _YLabelLength, __YLabelLength
		);
	}
	if( (__XLabelLength!= _XLabelLength && XLabelLength != _XLabelLength) ||
		(__XLabelLength== _XLabelLength && !XLabelLength) ||
		  /* 981016	*/
		(XLabelLength== _XLabelLength && __XLabelLength!= _XLabelLength) ||
		_polarFlag!= wi->polarFlag
	){
		if( debugFlag){
			fprintf( StdErr, "XLabelLength: %d -> %d\n", XLabelLength, _XLabelLength);
			fflush( StdErr);
		}
		__XLabelLength= XLabelLength;
		XLabelLength= _XLabelLength;
		wi->redraw= wi->redraw_val;
	}
	if( (__YLabelLength!= _YLabelLength && YLabelLength != _YLabelLength) ||
		(__YLabelLength== _YLabelLength && !YLabelLength) ||
		  /* 981016	*/
		(YLabelLength== _YLabelLength && __YLabelLength!= _YLabelLength) ||
		_polarFlag!= wi->polarFlag
	){
		if( debugFlag){
			fprintf( StdErr, "YLabelLength: %d -> %d\n", YLabelLength, _YLabelLength);
			fflush( StdErr);
		}
		__YLabelLength= YLabelLength;
		YLabelLength= _YLabelLength;
		wi->redraw= wi->redraw_val;
	}

	_polarFlag= wi->polarFlag;

	wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced || prev_silent );
	if( DoIt ){
		DoIt= 0;
		wi->redraw= 0;
		if( debugFlag ){
			fprintf( StdErr, "drawn\n");
			fflush( StdErr );
		}
	}
	else{
		if( wi->redraw_val!= PS_PRINTING && (wi->redraw || PS_STATE(wi)->Printing== PS_PRINTING) ){
		  char *msg="";
			DoIt= 0;
			if( PS_STATE(wi)->Printing== PS_PRINTING ){
				wi->redraw= PS_PRINTING;
				msg= "for PostScript ?buglet?";
			}
			sprintf( ps_comment, "DrawGridAndAxis(): trying again..%s", msg);
			if( debugFlag ){
				fprintf( StdErr, "%s\n", ps_comment);
				fflush( StdErr );
			}
		}
		else{
			sprintf( ps_comment, "DrawGridAndAxis(): eureka ... ");
			if( debugFlag ){
				fprintf( StdErr, ps_comment );
				fflush( StdErr);
			}
			DoIt= doIt;
			if( Handle_An_Event( wi->event_level, 1, "DrawGridAndAxis()", wi->window,
					/* ExposureMask| */StructureNotifyMask|KeyPressMask|ButtonPressMask
				)
			){
				XG_XSync( disp, False );
				if( wi->delete_it== -1 ){
					return(0);
				}
			}
			else{
			  /* We call ourselves, taking care to pass DoIt (and not doIt passed by our caller),
			   \ so that the initialisation DoIt= doIt at the start of this routine does not
			   \ change anything.. If, however, doIt< 0, then we (probably) have a virgin window's
			   \ first redraw. That means that DoIt==0, and should get set to True..
			   \ We store the value to pass in doIt, and check whether it is True. If it is not,
			   \ there is no point in calling ourselves, as we shouldn't draw anyway, and it would
			   \ catch us in a dead loop.
			   */
				doIt= (doIt< 0)? True : DoIt;
				if( doIt ){
					DrawGridAndAxis(wi, doIt );
				}
			}
		}
	}
	TitleMessage( wi, NULL );
	GCA();
	return(0);
#undef __XLabelLength
#undef __YLabelLength
#undef DoIt
#undef _polarFlag
}

double gridBase, gridStep, gridJuke[101];
int gridNJuke, gridCurJuke, floating= 0;
double floating_value= 0.0;

#define ADD_GRID(val)	(gridJuke[gridNJuke++] = (*ag_func)(wi,val))

int round_polar= 0;

double initGrid( LocalWin *wi, double low, double step, int logFlag, int sqrtFlag, AxisName axis, int polar )
{
    double ratio, x;
    double RoundUp(LocalWin*, double), stepGrid(double);
	double (*ag_func)(), powFlag;

/* 	ag_func= log10;	*/
/* 	func= pow;	*/
	if( axis== X_axis){
		RoundUp_log= nlog10X;
/* 		ag_func= cus_log10X;	*/
		ag_func= nlog10X;
		powFlag= wi->powXFlag;
	}
	else{
		RoundUp_log= nlog10Y;
/* 		ag_func= cus_log10Y;	*/
		ag_func= nlog10Y;
		powFlag= wi->powYFlag;
	}

	if( debugFlag ){
		fprintf( StdErr, "initGrid(low=%g,step=%g,%d,%d,%c)=",
			low, step, logFlag, sqrtFlag, (axis== X_axis)? 'X' : 'Y'
		);
		fflush( StdErr );
	}
    gridNJuke = gridCurJuke = 0;
    gridJuke[gridNJuke++] = 0.0;

    if( logFlag ){
/*
		if( sqrtFlag)
			ratio= cus_sqr(step);
		else
 */
			ratio = pow(10.0, step);
		gridBase = ssfloor(low);
		{ int sign= SIGN(step);
			step= ABS(step);
			gridStep = sign* ssceil( step );
		}
		if( ratio <= 3.0 ){
			if( ratio > 2.0 ){
				ADD_GRID(3.0);
			}
			else if( ratio > 1.333 ){
				ADD_GRID(2.0);
				ADD_GRID(5.0);
			}
			else if( ratio > 1.25 ){
				ADD_GRID(1.5);
				ADD_GRID(2.0);
				ADD_GRID(3.0);
				ADD_GRID(5.0);
				ADD_GRID(7.0);
			} else {
				for( x = 1.0; x < 10.0 && (x+.5)/(x+.4) >= ratio; x += .5 ){
					ADD_GRID(x + .1);
					ADD_GRID(x + .2);
					ADD_GRID(x + .3);
					ADD_GRID(x + .4);
					ADD_GRID(x + .5);
				}
				if( ssfloor(x) != x)
					ADD_GRID(x += .5);
				for( ; x < 10.0 && (x+1.0)/(x+.5) >= ratio; x += 1.0 ){
					ADD_GRID(x + .5);
					ADD_GRID(x + 1.0);
				}
				for( ; x < 10.0 && (x+1.0)/x >= ratio; x += 1.0 ){
					ADD_GRID(x + 1.0);
				}
				if( x == 7.0 ){
					gridNJuke--;
					x = 6.0;
				}
				if( x < 7.0 ){
					ADD_GRID(x + 2.0);
				}
				if( x == 10.0)
					gridNJuke--;
			}
			x = low - gridBase;
			for( gridCurJuke = -1; x >= gridJuke[gridCurJuke+1]; gridCurJuke++){
				;
			}
		}
    }
	else {
	  int rp= round_polar;
		if( sqrtFlag ){
			if( ABS(step)> 1.0 )
				step= cus_pow(step, 1/powFlag);
		}
		round_polar= polar;
		gridStep = RoundUp(wi, step);
		round_polar= rp;
		gridBase = ssfloor(low / gridStep) * gridStep;
	}
	if( axis== X_axis && wi->logXFlag && wi->_log_zero_x){
		floating= 1;
		floating_value= wi->log10_zero_x;
	}
	else if( axis== Y_axis && wi->logYFlag && wi->_log_zero_y){
		floating= 1;
		floating_value= wi->log10_zero_y;
	}
	if( debugFlag ){
		fprintf( StdErr, " ; base=%g step=%g\n",
			gridBase, gridStep
		);
		fflush( StdErr );
	}
	return( (polar)? low : stepGrid(1.0) );
}

double stepGrid( double factor)
{  double x;
    if( ++gridCurJuke >= gridNJuke ){
		gridCurJuke = 0;
		gridBase += factor* gridStep;
    }
    x= gridBase + gridJuke[gridCurJuke];
	if( floating && x > floating_value){
		gridCurJuke--;
		x= floating_value;
		floating= 0;
	}
	if( debugFlag && ActiveWin && ActiveWin->axisFlag ){
		fprintf( StdErr, "(stepGrid(%g)=%g)\t", factor, x);
		fflush( StdErr );
	}
    return(x);
}

double RoundUp( LocalWin *wi, double val )
/*
 * This routine rounds up (or down for negatives) the abs of the given number such that
 * it is some power of ten times either 1, 2, or 5 (4 for polar).  It is
 * used to find increments for grid lines.
 \ It uses the functionpointer RoundUp_log to use either
 \ a log10(x) or, if sqrtFlag, a sqrt(log10(x)) way of
 \ calculating the exponent. This works.
 \ 0 is rounded to 0
 */
{
    int exponent, idx, sign= SIGN(val);
	double val1= wi->radix/ 4.0, val2= 0.5* val1;

	if( !val ){
		return( 0.0);
	}
	if( NaN(val) || INF(val) ){
		return( val );
	}
	val= ABS(val);
	errno = 0;
	{ double e = ssfloor( (*RoundUp_log)(wi, val));
		if( NaN(e) || INF(e) ){
		  // 20101016: sadly, this can happen on my new MBP 13" ... :(
			fprintf( StdErr, "RoundUp(); RoundUp_log(%s)=%s should not happen (log10 returns %s)!",
				d2str( val, "%g", NULL), d2str( e, "%g", NULL ),
				d2str( log10(val), "%g", NULL )
			);
			if( errno ){
				fprintf( StdErr, " (%s)", serror() );
			}
			fputs( "\n", StdErr );
			return val;
		}
		else{
		    exponent = (int) e;
		}
	}
	if( debugFlag ){
		fprintf( StdErr, "RoundUp(%s): exp=%d",
			d2str( val, "%g", NULL), exponent
		);
	}
    if( exponent < 0 ){
		for( idx = exponent;  idx < 0; idx++ ){
			val *= 10.0;
			val1*= 10.0;
			val2*= 10.0;
		}
    } else {
		for( idx = 0;  idx < exponent; idx++ ){
			val /= 10.0;
			val1/= 10.0;
			val2/= 10.0;
		}
    }
	if( debugFlag ){
		fprintf( StdErr, " -> %s",
			d2str( val, "%g", NULL)
		);
	}
	if( round_polar ){
		if( val > val2 )
			val = val1;
		else if( val > val2/3 )
			val = val2;
		else if( val > val2/9.0 )
			val = val2/4.5;
		else
			val = val2/9;
	}
	else{
		if( val > 5.0)
			val = 10.0;
		else if( val > 2 )
			val = 5.0;
		else if( val > 1 )
			val = 2.0;
		else
			val = 1.0;
	}
	if( debugFlag ){
		fprintf( StdErr, "%s -> %s",
			(round_polar)? " (polar)" : "",
			d2str( val, "%g", NULL)
		);
	}
    if( exponent < 0 ){
		for( idx = exponent;  idx < 0;  idx++ ){
			val /= 10.0;
		}
    } else {
		for( idx = 0;  idx < exponent;  idx++ ){
			val *= 10.0;
		}
    }
	if( debugFlag ){
		fprintf( StdErr, " -> %s\n",
			d2str( val, "%g", NULL)
		);
	}
	val*= sign;
    return val;
}

int is_log_zero= 0, is_log_zero2= 0;

/* 20031008: modified/additional clipcodes: HOR_CODE and VER_CODE so
 \ that easy checking for horizontal and vertical outliers is possible.
 */
typedef enum ClipCodes { LEFT_CODE=0x01, RIGHT_CODE=0x02, BOTTOM_CODE=0x04, TOP_CODE=0x08, HOR_CODE=0x10, VER_CODE=0x20 } ClipCodes;

char *clipcode(int code)
{  static char cc[4][8];
   static int bnr= 0;
/* 	memset( cc[bnr], 0, sizeof(cc[bnr]) );	*/
	cc[bnr][0]= '\0';
	if( (code & LEFT_CODE) )
		strcat( cc[bnr], "L");
	if( (code & RIGHT_CODE) )
		strcat( cc[bnr], "R");
	if( (code & BOTTOM_CODE) )
		strcat( cc[bnr], "B");
	if( (code & TOP_CODE) )
		strcat( cc[bnr], "T");
	bnr= (bnr+1) % 4;
	return(cc[bnr]);
}

/* Clipping algorithm from Neumann and Sproull by Cohen and Sutherland */
#define C_CODE(xval, yval, rtn ){\
	if( (xval) < cminx){ (rtn) = LEFT_CODE|HOR_CODE;} \
	else if( (xval) > cmaxx){ (rtn) = RIGHT_CODE|HOR_CODE; }\
	else (rtn) = 0; \
	if( (yval) < cminy){ (rtn) |= BOTTOM_CODE|VER_CODE; }\
	else if( (yval) > cmaxy){ (rtn) |= TOP_CODE|VER_CODE; }\
}

/* mark_inside1: is (sx1,sy1) inside the clipping window?
 \ mark_inside2: is (sx2,sy2) inside the window after clipping?
 */
int ClipWindow( LocalWin *wi, DataSet *this_set, int floating,
	double *sx1, double *sy1, double *sx2, double *sy2, int *mark_inside1, int *mark_inside2, int *clipcode1, int *clipcode2
)
{  double tx, ty;
   int cd, wx1, wx2, wy1, wy2;
   int cs1, cs2, count= 0, nan1= 0, nan2= 0, cont= True, floating_bars= False;
   ClipCodes code1, code2;
   double cminx, cminy, cmaxx, cmaxy;

#if DEBUG==2
{ int ok= 1;
	if( !ok ){
		return(0);
	}
}
#endif
	if( !clipcode1 ){
		clipcode1= &cs1;
	}
	if( !clipcode2 ){
		clipcode2= &cs2;
	}
	if( NaNorInf(*sx1) ){
		nan1= 1;
		*sx1= (Inf(*sx1)== -1)? -DBL_MAX : DBL_MAX;
	}
	if( NaNorInf(*sy1) ){
		nan1= 1;
		*sy1= (Inf(*sy1)== -1)? -DBL_MAX : DBL_MAX;
	}
	if( NaNorInf(*sx2) ){
		nan2= 2;
		*sx2= (Inf(*sx2)== -1)? -DBL_MAX : DBL_MAX;
	}
	if( NaNorInf(*sy2) ){
		nan2= 2;
		*sy2= (Inf(*sy2)== -1)? -DBL_MAX : DBL_MAX;
	}
	if( nan1 || nan2 ){
	 /* the culprits have been changed. Pass another time through ClipWindow
	  \ to clip the remainder of the points
	  */
		ClipWindow( wi, this_set, floating, sx1, sy1, sx2, sy2, mark_inside1, mark_inside2, clipcode1, clipcode2);
		if( nan1 ) *mark_inside1= 0;
		if( nan2 ) *mark_inside2= 0;
		  /* return 1 if one of the two points is within the plotting window.
		   \ If not (both outside), we should return 0. Not that this is different
		   \ from the regular returnvalue, which is 1 if BOTH points are withing
		   \ the plotting window.
		   */
		return( !(*clipcode1) || !(*clipcode2) );
	}

	if( (this_set && this_set->floating) || floating ){
		cminx= wi->WinOrgX;
		cminy= wi->WinOrgY;
		cmaxx= wi->WinOppX;
		cmaxy= wi->WinOppY;
		if( this_set && this_set->barFlag ){
			floating_bars= True;
		}
	}
	else{
		cminx= wi->UsrOrgX;
		cminy= wi->UsrOrgY;
		cmaxx= wi->UsrOppX;
		cmaxy= wi->UsrOppY;
	}

	C_CODE(*sx1, *sy1,code1);
	C_CODE(*sx2, *sy2, code2);
	*mark_inside1 = (code1 == 0)? 1 : 0;
	*mark_inside2 = (code2 == 0)? 1 : 0;
	while( (code1 || code2) && !(nan1 || nan2) && cont ){
		if( debugFlag && debugLevel>= 1){
			fprintf( StdErr, "ClipWindow(%d): ", count);
			fflush( StdErr);
		}
		if( code1 & code2 ){
			  /* 20040305: basically accept everything 'floating' and not a dataset: */
			if( !this_set && floating ){
				cont= False;
			} else
			  /* 20031008 */
			if( !((code1 & HOR_CODE) && (code2 & HOR_CODE)) && floating_bars ){
				cont= False;
			}
			else{
				if( debugFlag && debugLevel>= 1){
					fprintf( StdErr, "line (%g,%g)-(%g,%g) outside (%s,%s)\n",
						*sx1, *sy1, *sx2, *sy2, clipcode(code1),clipcode(code2)
					);
					fflush( StdErr);
				}
				goto skip;
			}
		}
		/* RJB 920330: check for horizontal and vertical lines:
		 * these can be clipped in one iteration.
		 * The trivial case of a point needs no clipping at all.
		 */
		wx1= SCREENX(wi, *sx1);
		wy1= SCREENY(wi, *sy1);
		wx2= SCREENX(wi, *sx2);
		wy2= SCREENY(wi, *sy2);
		if( wx1 == wx2 ){
		  /* vertical line	*/
			if( wy1 == wy2 && !floating_bars ){
			  /* one point: just tell if it is inside	*/
				if( debugFlag && debugLevel>= 1){
					fprintf( StdErr, "just one point on the device");
					fflush( StdErr);
				}
				return( *mark_inside1);
			}
			else{
			  /* vertical line: clip it if possible	*/
/* 				if( !( (code1 & LEFT_CODE)) && !( (code1 & RIGHT_CODE)) )	*/
				if( !(code1 & HOR_CODE) )
				{
					if( (code1 & BOTTOM_CODE))
						*sy1= cminy;
					else if( (code1 & TOP_CODE))
						*sy1= cmaxy;
					if( (code2 & BOTTOM_CODE))
						*sy2= cminy;
					else if( (code2 & TOP_CODE))
						*sy2= cmaxy;
					if( debugFlag && debugLevel>= 1){
						fprintf( StdErr, "vertical line (%g,%g)-(%g,%g) (%s)\n",
							*sx1, *sy1, *sx2, *sy2, clipcode(code1)
						);
						fflush( StdErr);
					}
					return(1);
				}
				else{
				  /* x1 (horizontal) out of bounds	*/
					if( debugFlag && debugLevel>= 1){
						fprintf( StdErr, "vertical line (%g,%g)-(%g,%g) outside (%s)\n",
							*sx1, *sy1, *sx2, *sy2, clipcode(code1)
						);
						fflush( StdErr);
					}
					return(0);
				}
			}
		}
		else if( wy1 == wy2 ){
		  /* horizontal LINE (point case already checked)	*/
			  /* 20031007 */
			if(
/* 				(!((code1 & TOP_CODE)) && !((code1 & BOTTOM_CODE)))	*/
				!(code1 & VER_CODE)
					|| floating_bars
			){
			  /* y1 inside	*/
				if( (code1 & LEFT_CODE))
					*sx1= cminx;
				else if( (code1 & RIGHT_CODE))
					*sx1= cmaxx;
				if( (code2 & LEFT_CODE))
					*sx2= cminx;
				else if( (code2 & RIGHT_CODE))
					*sx2= cmaxx;
				if( floating_bars ){
				  /* 20031007: check if the line is outside vertical boundaries... */
					if( (code1 & BOTTOM_CODE))
						*sy2= *sy1= cminy;
					else if( (code1 & TOP_CODE))
						*sy2= *sy1= cmaxy;
				}
				if( debugFlag && debugLevel>= 1){
					fprintf( StdErr, "horizontal line (%g,%g)-(%g,%g) (%s)\n",
						*sx1, *sy1, *sx2, *sy2, clipcode(code1)
					);
					fflush( StdErr);
				}
				return(1);
			}
			else{
			  /* y1 (vertical) out of bounds	*/
				if( debugFlag && debugLevel>= 1){
					fprintf( StdErr, "horizontal line (%g,%g)-(%g,%g) outside (%s)\n",
						*sx1, *sy1, *sx2, *sy2, clipcode(code1)
					);
					fflush( StdErr);
				}
				return(0);
			}
		}
		cd = (code1)? code1 : code2;
		if( (cd & LEFT_CODE) ){	/* Crosses left edge */
			ty = *sy1 + (*sy2 - *sy1) * ((cminx - *sx1) / (*sx2 - *sx1));
			tx = cminx;
		}
		else if( (cd & RIGHT_CODE) ){ /* Crosses right edge */
			ty = *sy1 + (*sy2 - *sy1) * ((cmaxx - *sx1) / (*sx2 - *sx1));
			tx = cmaxx;
		}
		  /* 20031007: 'floating_bars' need special care: don't touch sx. */
		else if( (cd & BOTTOM_CODE) ){ /* Crosses bottom edge */
			tx = (floating_bars)? *sx1 : *sx1 + (*sx2 - *sx1) * ((cminy - *sy1) / (*sy2 - *sy1));
			ty = cminy;
		}
		else if( (cd & TOP_CODE) ){ /* Crosses top edge */
			tx = (floating_bars)? *sx1 : *sx1 + (*sx2 - *sx1) * ((cmaxy - *sy1) / (*sy2 - *sy1));
			ty = cmaxy;
		}
		if( NaNorInf(tx) || NaNorInf(ty) ){
			code1= 0xff;
			code2= 0xff;
			if( debugFlag && debugLevel>= 1 ){
				if(cd==code1){
					fprintf( StdErr, "(%s,%s)(%s) clipped to (%s,%s) - skipping\n",
						d2str( *sx1, "%g", NULL), d2str( *sy1, "%g", NULL),
						clipcode(cd),
						d2str( tx, "%g", NULL), d2str( ty, "%g", NULL)
					);
				}
				else{
					fprintf( StdErr, "(%s,%s)(%s) clipped to (%s,%s) - skipping\n",
						d2str( *sx2, "%g", NULL), d2str( *sy2, "%g", NULL),
						clipcode(cd),
						d2str( tx, "%g", NULL), d2str( ty, "%g", NULL)
					);
				}
				fflush( StdErr );
			}
			goto skip;
		}
		if( cd == code1 ){
			*sx1 = tx;  *sy1 = ty;
			C_CODE(*sx1, *sy1, code1);
		}
		else {
			*sx2 = tx;  *sy2 = ty;
			C_CODE(*sx2, *sy2, code2);
		}
		if( debugFlag && debugLevel>= 1){
			fprintf( StdErr, "line (%g,%g)-(%g,%g) (%s)\n",
				*sx1, *sy1, *sx2, *sy2, clipcode(cd)
			);
			fflush( StdErr);
		}
		count++;
	}
skip:;
	if( clipcode1 )
		*clipcode1= (int) code1;
	if( clipcode2 )
		*clipcode2= (int) code2;
	return( !code1 && !code2);
}

void make_vector( LocalWin *wi, DataSet *this_set, double X, double Y, double *sx3, double *sy3, double *sx4, double *sy4,
	double orn, double vlength
)
{
	Gonio_Base( wi, wi->radix, wi->radix_offset );
	switch( this_set->vectorType ){
		case 0:
		default:
			  /* default, traditional behaviour: take the length from this_set->vectorLength,
			   \ then fall through to the code handling the type 0 and 2 vector generation:
			   */
			vlength= this_set->vectorLength;
		case 2:
			  /* The first point is actually the datapoint itself..	*/
			*sx3 = X;
			*sy3 = Y;
			if( this_set->use_error && !NaNorInf(orn) && !NaNorInf(vlength) ){
			  double c= vlength* Cos( orn),
					s= vlength* Sin( orn);
				*sx4 = X+ c;
				*sy4 = Y+ s;
#if defined(i386) && defined(__GNUC__)
				if( NaNorInf(*sx4) || NaNorInf(*sy4) ){
					fprintf( StdErr, "make_vector(%g,%g,%s), len=%g, trign=(%s,%s): (%s,%s)-(%s,%s)\n",
						X, Y, d2str(orn,0,0), vlength, d2str(c,0,0), d2str(s,0,0),
						d2str(*sx3,0,0), d2str(*sy3,0,0), d2str(*sx4,0,0), d2str(*sy4,0,0)
					);
				}
#endif
			}
			else{
				*sx4 = X;
				*sy4 = Y;
			}

			break;
		case 1:
			vlength= this_set->vectorLength;
		case 3:
		case 4:
			if( this_set->use_error && !NaNorInf(orn) && !NaNorInf(vlength) ){
			  double c= vlength* Cos( orn),
					s= vlength* Sin( orn),
					hind_frac= 1- this_set->vectorPars[0];
				*sx3 = X- hind_frac* c;
				*sy3 = Y- hind_frac* s;
				*sx4 = X+ this_set->vectorPars[0]* c;
				*sy4 = Y+ this_set->vectorPars[0]* s;
			}
			else{
				*sx3= *sx4 = X;
				*sy3= *sy4 = Y;
			}

			break;
	}
}

void make_sized_marker( LocalWin *wi, DataSet *this_set, double X, double Y, double *sx3, double *sy3, double *sx4, double *sy4,
	double size
)
{
	if( !NaNorInf(size) && this_set->use_error ){
	  /* size is interpreted as the marker's radius	*/
		if( !NaNorInf(this_set->markSize) ){
			size*= this_set->markSize;
		}
		*sx3 = X- size;
		*sy3 = Y- size;
		*sx4 = X+ size;
		*sy4 = Y+ size;
	}
	else{
		*sx3= *sx4 = X;
		*sy3= *sy4 = Y;
	}
}

int DrawData_process(LocalWin *wi, DataSet *this_set, double data[2][ASCANF_DATA_COLUMNS], int subindex,
	int nr, int ncoords,
	double *sx1, double *sy1,
	double *sx2, double *sy2,
	double *sx3, double *sy3, double *sx4, double *sy4,
	double *sx5, double *sy5, double *sx6, double *sy6
)
{ int ok= 0;
  static int start= 0, redoing= False;
	if( !wi->raw_display ){
	  int i, spot= subindex, n, rsacsv= reset_ascanf_currentself_value;
	  int column[ASCANF_DATA_COLUMNS]= {0,1,2,3}, exprerror[6]= {0,0,0,0,0,0};
		clean_param_scratch();
		ok= 1;
		for( i= start; i< nr; i++, spot++ ){
		  char change[4];
		  int aae= 0;
			ok= 1;
			if( wi->process.data_init_len && spot== 0 ){
				n= param_scratch_len;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "DrawData(): DATA Init: %s", wi->process.data_init);
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*DATA_INIT*";
				compiled_fascanf( &n, wi->process.data_init, param_scratch, NULL, data[i], column, &wi->process.C_data_init );
				aae+= ascanf_arg_error;
				if( /* ascanf_arg_error || */ !n ){
					ok= 0;
				}
				if( ascanf_arg_error || !ok ){
					exprerror[0]+= 1;
				}
				if( this_set->numPoints< 0 ){
					return(0);
				}
			}
			if( (wi->process.data_process_len || (!*disable_SET_PROCESS && this_set->process.set_process_len)) ){
				if( ok && wi->process.data_before_len ){
					n= param_scratch_len;
					*ascanf_self_value= (double) spot;
					*ascanf_current_value= (double) spot;
					*ascanf_counter= (*ascanf_Counter)= spot;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					if( ascanf_verbose ){
						fprintf( StdErr, "DrawData(): DATA Before: %s", wi->process.data_before);
						fflush( StdErr );
					}
					ascanf_arg_error= 0;
					TBARprogress_header= "*DATA_BEFORE*";
					compiled_fascanf( &n, wi->process.data_before, param_scratch, NULL, data[i], column, &wi->process.C_data_before );
					aae+= ascanf_arg_error;
					if( /* ascanf_arg_error || */ !n ){
						ok= 0;
					}
					if( ascanf_arg_error || !ok ){
						exprerror[1]+= 1;
					}
					if( this_set->numPoints< 0 ){
						return(0);
					}
				}
#define DO_SETPROCESS	n= ASCANF_DATA_COLUMNS;		\
					*ascanf_self_value= (double) spot;		\
					*ascanf_current_value= (double) spot;		\
					*ascanf_counter= (*ascanf_Counter)= spot;		\
					reset_ascanf_currentself_value= 0;		\
					reset_ascanf_index_value= True;		\
					if( ascanf_verbose ){		\
						fprintf( StdErr, "DrawData(): SET x y e: %s", this_set->process.set_process );		\
						fflush( StdErr );		\
					}		\
					ascanf_arg_error= 0;		\
					TBARprogress_header= "*SET_PROCESS*";		\
					compiled_fascanf( &n, this_set->process.set_process, data[i], change, data[i], column,		\
						&this_set->process.C_set_process		\
					);		\
					aae+= ascanf_arg_error;		\
					if( /* !ascanf_arg_error && */ n &&		\
						(change[0]== 'N' || change[0]== 'R') &&		\
						(change[1]== 'N' || change[1]== 'R')		\
					){		\
						ok= 1;		\
					}		\
					else{		\
						ok= 0;		\
					}

				if( ok && !*disable_SET_PROCESS && *SET_PROCESS_last< 0 && this_set->process.set_process_len ){
					DO_SETPROCESS;
					if( ascanf_arg_error || !ok ){
						exprerror[3]+= 1;
					}
					if( this_set->numPoints< 0 ){
						return(0);
					}
				}

				if( ok && wi->process.data_process_len ){
					n= ncoords;
					*ascanf_self_value= (double) spot;
					*ascanf_current_value= (double) spot;
					*ascanf_counter= (*ascanf_Counter)= spot;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					if( ascanf_verbose ){
						fprintf( StdErr, "DrawData(): DATA x y e: %s", wi->process.data_process );
						fflush( StdErr );
					}
					ascanf_arg_error= 0;
					TBARprogress_header= "*DATA_PROCESS*";
					compiled_fascanf( &n, wi->process.data_process, data[i], change, data[i], column, &wi->process.C_data_process );
					aae+= ascanf_arg_error;
					if( /* !ascanf_arg_error && */ n &&
						(change[0]== 'N' || change[0]== 'R') &&
						(change[1]== 'N' || change[1]== 'R')
					){
						ok= 1;
					}
					else{
						ok= 0;
					}
					if( ascanf_arg_error || !ok ){
						exprerror[2]+= 1;
					}
					if( this_set->numPoints< 0 ){
						return(0);
					}
				}
				if( ok && !*disable_SET_PROCESS && !*SET_PROCESS_last && this_set->process.set_process_len ){
					DO_SETPROCESS;
					if( ascanf_arg_error || !ok ){
						exprerror[3]+= 1;
					}
					if( this_set->numPoints< 0 ){
						return(0);
					}
				}
				if( ok && wi->process.data_after_len ){
					n= param_scratch_len;
					*ascanf_self_value= (double) spot;
					*ascanf_current_value= (double) spot;
					*ascanf_counter= (*ascanf_Counter)= spot;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					if( ascanf_verbose ){
						fprintf( StdErr, "DrawData(): DATA After: %s", wi->process.data_after );
						fflush( StdErr );
					}
					ascanf_arg_error= 0;
					TBARprogress_header= "*DATA_AFTER*";
					compiled_fascanf( &n, wi->process.data_after, param_scratch, NULL, data[i], column, &wi->process.C_data_after );
					aae+= ascanf_arg_error;
					if( /* ascanf_arg_error || */ !n ){
						ok= 0;
					}
					if( ascanf_arg_error || !ok ){
						exprerror[4]+= 1;
					}
					if( this_set->numPoints< 0 ){
						return(0);
					}
				}
				if( ok && !*disable_SET_PROCESS && *SET_PROCESS_last>0 && this_set->process.set_process_len ){
					DO_SETPROCESS;
					if( ascanf_arg_error || !ok ){
						exprerror[3]+= 1;
					}
				}
			}
			if( ok && wi->process.data_finish_len &&
					  /* 20010828: the latter condition probably only occurs when an earlier expression
					   \ set numPoints to 0...
					   */
					(spot== this_set->numPoints-1 || (spot== 0 && this_set->numPoints== 0))
			){
				n= param_scratch_len;
				*ascanf_self_value= (double) spot;
				*ascanf_current_value= (double) spot;
				*ascanf_counter= (*ascanf_Counter)= spot;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				if( ascanf_verbose ){
					fprintf( StdErr, "DrawData(): DATA Finish: %s", wi->process.data_finish );
					fflush( StdErr );
				}
				ascanf_arg_error= 0;
				TBARprogress_header= "*DATA_FINISH*";
				compiled_fascanf( &n, wi->process.data_finish, param_scratch, NULL, data[i], column, &wi->process.C_data_finish );
				aae+= ascanf_arg_error;
				if( /* ascanf_arg_error || */ !n ){
					ok= 0;
				}
				if( ascanf_arg_error || !ok ){
					exprerror[5]+= 1;
				}
				if( this_set->numPoints< 0 ){
					return(0);
				}
			}
			TBARprogress_header= NULL;
			if( aae || !ok ){
			  int err;
			  double avb= ascanf_verbose;
				fprintf( StdErr, "DrawData(#%d[%d]): %s,%s,%s: error", this_set->set_nr,
					subindex, d2str( data[i][0], "%g", NULL), d2str( data[i][1], "%g", NULL), d2str( data[i][2], "%g", NULL)
				);
				if( ascanf_emsg ){
					fprintf( StdErr, " (%s)", ascanf_emsg );
				}
				fputc( '\n', StdErr );
				if( !redoing ){
					for( err= 0; err< sizeof(exprerror)/sizeof(int); err++ ){
					  char *exprname[6]= { "*DATA_INIT*", "*DATA_BEFORE*", "*DATA_PROCESS*", "*SET_PROCESS*",
							"*DATA_AFTER*", "*DATA_FINISH*" };
						if( exprerror[err] ){
							fprintf( StdErr, "\tError occurred in %s:\n", exprname[err] );
							switch( err ){
								case 0:
									Print_Form( StdErr, &(wi->process.C_data_init), 1, True, "\t", NULL, "\n", True );
									break;
								case 1:
									Print_Form( StdErr, &(wi->process.C_data_before), 1, True, "\t", NULL, "\n", True );
									break;
								case 2:
									Print_Form( StdErr, &(wi->process.C_data_process), 1, True, "\t", NULL, "\n", True );
									break;
								case 3:
									Print_Form( StdErr, &this_set->process.C_set_process, 1, True, "\t", NULL, "\n", True );
									break;
								case 4:
									Print_Form( StdErr, &(wi->process.C_data_after), 1, True, "\t", NULL, "\n", True );
									break;
								case 5:
									Print_Form( StdErr, &(wi->process.C_data_finish), 1, True, "\t", NULL, "\n", True );
									break;
							}
						}
					}
				}
				fflush( StdErr );
				if( Redo_Error_Expression && !ascanf_verbose ){
					ascanf_verbose= True;
					start= i;
					redoing= True;
					DrawData_process( wi, this_set, data, subindex, start+1, ncoords,
						sx1, sy1, sx2, sy2,
						sx3, sy3, sx4, sy4,
						sx5, sy5, sx6, sy6
					);
					ascanf_verbose= avb;
					start= 0;
					redoing= False;
				}
			}
			/* else */{
			  /* substitute values.	*/
				switch( i ){
					case 0:
						*sx1 = data[0][0];
						*sy1 = data[0][1];
						if( (wi->vectorFlag || wi->error_type[this_set->set_nr]== MSIZE_FLAG) && !wi->use_average_error ){
						  /* determine the 2 points of the vector (vector) flag that will embellish this
						   \ point. The orientation is given by what is otherwise interpreted as the error
						   \ The length is taken from the set-specific vectorLength.
						   */
							if( wi->error_type[this_set->set_nr]== MSIZE_FLAG ){
								make_sized_marker( wi, this_set, *sx2, *sy2, sx3, sy3, sx4, sy4, data[0][2] );
							}
							else{
								make_vector( wi, this_set, *sx1, *sy1, sx3, sy3, sx4, sy4, data[0][2], data[0][3] );
							}
						}
						else{
							*sy3 = *sy1- data[0][2];
							*sy4 = *sy1+ data[0][2];
						}
						break;
					case 1:
						*sx2 = data[1][0];
						*sy2 = data[1][1];
						if( (wi->vectorFlag || wi->error_type[this_set->set_nr]== MSIZE_FLAG) && !wi->use_average_error ){
						  /* determine the 2 points of the vector (vector) flag that will embellish this
						   \ point. The orientation is given by what is otherwise interpreted as the error
						   \ The length is taken from the set-specific vectorLength.
						   */
							if( wi->error_type[this_set->set_nr]== MSIZE_FLAG ){
								make_sized_marker( wi, this_set, *sx2, *sy2, sx5, sy5, sx6, sy6, data[1][2] );
							}
							else{
								make_vector( wi, this_set, *sx2, *sy2, sx5, sy5, sx6, sy6, data[1][2], data[1][3] );
							}
						}
						else{
							*sy5 = *sy2- data[1][2];
							*sy6 = *sy2+ data[1][2];
						}
						break;
				}
			}
		}
		reset_ascanf_currentself_value= rsacsv;
	}
	return(ok);
}

#ifdef DEBUG
	LocalWinGeo *theWin_Geo;
#endif

/* 961223
 \ Indexing of the ..segs. arrays appears to be buggy. A weird SIGSEGV resulted when a window was
 \ (partially) zoomed (excluding some sets), and afterwards an extra file was added through the READ_FILE
 \ utility in the Settings Dialog. Happened when realloc'ing the ..segs. arrays. After a lengthy posse, I
 \ found that the index X_idx-1 sometimes became -1, causing overwriting of (probably) headers of XsegsE[].
 \ Indexing now goes either through these routines, which implement a boundary-check (in DEBUG mode), or
 \ through macros with similar names, that expand to the bare-bones indexing (in "normal" mode). Be warned
 \ that I may not have found all occurences...!
 */
#ifdef DEBUG
XSegment *SegsNR( XSegment *s, int i)
{ int size;
  static XSegment dum;
	if( s== Xsegs ){
		size= XsegsSize;
	}
	else{
		size= YsegsSize;
	}
	size/= sizeof(XSegment);
	if( i< 0 || i>= size ){
		fprintf( StdErr, "SegsNR(0x%lx,%d): invalid index ([0,%d>)\n",
			s, i, size
		);
		return( &dum );
	}
	else{
		return( &s[i] );
	}
}

XXSegment *XSegsNR( XXSegment *s, int i)
{ int size;
  static XXSegment dum;
	if( s== XXsegs ){
		size= XXsegsSize;
	}
	size/= sizeof(XXSegment);
	if( i< 0 || i>= size ){
		fprintf( StdErr, "XSegsNR(0x%lx,%d): invalid index ([0,%d>)\n",
			s, i, size
		);
		return( &dum );
	}
	else{
		return( &s[i] );
	}
}

XSegment_error *SegsENR( XSegment_error *s, int i)
{ int size= XsegsESize/sizeof(XSegment_error);
  static XSegment_error dum;
	if( i< 0 || i>= size ){
		fprintf( StdErr, "SegsENR(0x%lx,%d): invalid index ([0,%d>)\n",
			s, i, size
		);
		return( &dum );
	}
	else{
		return( &s[i] );
	}
}
#else
#	define SegsNR(s,i)	(&(s[i]))
#	define XSegsNR(s,i)	(&(s[i]))
#	define SegsENR(s,i)	(&(s[i]))
#endif

#define RESET_WIN_STATE(wi)	{\
		if( (wi)->data_silent_process && !(wi)->osilent_val ){ \
			(wi)->osilent_val= (wi)->dev_info.xg_silent( (wi)->dev_info.user_state, (wi)->silenced || (wi)->osilent_val ); \
		} \
		if( (wi)->data_sync_draw ){ \
			if( (wi)->osync_val== 0 && Synchro_State ){ \
				(wi)->osync_val= Synchro_State; \
				X_Synchro((wi)); \
			} \
		} \
		else{ \
			if( (wi)->osync_val== 1 && !Synchro_State ){ \
				(wi)->osync_val= Synchro_State; \
				X_Synchro((wi)); \
			} \
		} \
}

/* 20000427: changed the discard in XG_XSync() in CHECK_EVENT() to False	*/
#define CHECK_EVENT()	{\
	if( Handle_An_Event( wi->event_level, 1, "DrawData-" STRING(__LINE__), wi->window, \
			StructureNotifyMask|KeyPressMask|ButtonPressMask\
		)\
	){\
		XG_XSync( disp, False );\
		if( wi->delete_it!= -1 ){\
			wi->event_level--;\
		}\
		RESET_WIN_STATE(wi); \
		return;\
	}\
	if( wi->delete_it== -1 || this_set->numPoints<= 0 ){\
		RESET_WIN_STATE(wi); \
		return;\
	}\
	if( wi->redraw ){\
		wi->event_level--;\
		RESET_WIN_STATE(wi);\
		return;\
	}\
}

/* 20000427: changed the discard in XG_XSync() in CHECK_EVENT2() to False	*/
#define CHECK_EVENT2() if( !PrintingWindow){ \
	if( Handle_An_Event( wi->event_level, 1, "DrawData-" STRING(__LINE__), wi->window,  \
			StructureNotifyMask|KeyPressMask|ButtonPressMask \
		) \
	){ \
		XG_XSync( disp, False ); \
		if( wi->delete_it!= -1 ){ \
			wi->event_level--; \
			wi->redraw= wi->redraw_val; \
		} \
		if( bounds_clip ){ \
			this_set->s_bounds_set= -1; \
		} \
		psSeg_disconnect_jumps= psSdj; \
		RESET_WIN_STATE(wi); \
		goto DrawData_return; \
	} \
	if( wi->delete_it== -1 ){ \
		if( bounds_clip ){ \
			this_set->s_bounds_set= -1; \
		} \
		psSeg_disconnect_jumps= psSdj; \
		RESET_WIN_STATE(wi); \
		goto DrawData_return; \
	} \
	if( wi->redraw ){ \
		wi->event_level--; \
		if( bounds_clip ){ \
			this_set->s_bounds_set= -1; \
		} \
		psSeg_disconnect_jumps= psSdj; \
		RESET_WIN_STATE(wi); \
		goto DrawData_return; \
	} \
}


/* LStyle is the type of line (L_AXIS, L_VAR,.. see xgout.h), not
 \ to be confused with the line-appearance, which is also referred to
 \ as "linestyle" (5th and 6th argument to xg_seg, respectively. The
 \ (use_)ps_LStyle allow a different behaviour for PostScript output.
 */
extern int use_ps_LStyle, ps_LStyle;

void HighlightSegment( LocalWin *wi, int idx, int nsegs, XSegment *segs, double width, int LStyle)
{ Pixel color0= AllAttrs[0].pixelValue;
  int upls= use_ps_LStyle;
  DataSet *set;
	if( idx>= 0 && idx< setNumber ){
		AllAttrs[0].pixelValue=
			(wi->legend_line[idx].pixvalue< 0)? wi->legend_line[idx].pixelValue : highlightPixel;
		set = &AllSets[idx];
	}
	else{
		AllAttrs[0].pixelValue= highlightPixel;
		set = NULL;
	}
	use_ps_LStyle= 0;
	wi->dev_info.xg_seg(wi->dev_info.user_state,
				nsegs, segs,
				width, LStyle,
				0, 0, 0, set
	);
	AllAttrs[0].pixelValue= color0;
	use_ps_LStyle= upls;
}

void CollectPointStats( LocalWin *wi, DataSet *this_set, int pnt_nr, double sx1, double sy1, double sx3, double sy3, double sx4, double sy4 )
{ double ignNaN= (*SS_Ignore_NaN==2)? 2 : 1;
	  /* 20040419: honour the SS_Ignore_NaN flag in a blunt way: */
	  /* 20050114: We don't draw NaN points. There is thus no reason to include them in statistics that are going to be used
	   \ to determine the axis scaling...! What if all points have a NaN co-ordinate?
	   */
	if( ASCANF_TRUE(ignNaN) &&
		( NaN(sx1) || NaN(sy1) )
	){
		return;
	}
	else if( ignNaN== 2 &&
		( INF(sx1) || INF(sy1) )
	){
		return;
	}

	// 20080630: when not fitting both axes, take into account only points that are within range
	// on the unfitted axis. This can probably be done simply using sx1,sy1 without taking into account
	// error bars, marker size, etc.
	if( wi->fitting== fitXAxis || (wi->fitting==fitDo && wi->fit_xbounds && !wi->fit_ybounds) ){
		if( sy1< wi->win_geo.bounds._loY || sy1> wi->win_geo.bounds._hiY ){
			return;
		}
	}
	else if( wi->fitting== fitYAxis || (wi->fitting==fitDo && wi->fit_ybounds && !wi->fit_xbounds) ){
		if( sx1< wi->win_geo.bounds._loX || sx1> wi->win_geo.bounds._hiX ){
			return;
		}
	}

	SS_Add_Data_( wi->SS_Xval, 1, sx1, 1.0);
	SS_Add_Data_( wi->SS_X, 1, sx1, 1.0);
	SS_Add_Data_( wi->SS_Yval, 1, sy1, 1.0);
	SS_Add_Data_( wi->SS_Y, 1, sy1, 1.0);
	if( this_set ){
	  int etype= wi->error_type[this_set->set_nr];
		if( wi->vectorFlag || etype== MSIZE_FLAG || etype== INTENSE_FLAG ){
		  /* The 2 edges of the vectors are interpreted as so many extra points in these sets!	*/
		  double v;
			if( etype== INTENSE_FLAG || etype== MSIZE_FLAG ){
			  double msx, msy;
			  double mS= this_set->markSize;
				if( etype== MSIZE_FLAG ){
				  double ms= this_set->markSize;
					this_set->markSize= -fabs( this_set->errvec[pnt_nr] * ((NaNorInf(ms))? 1 : ms) );
				}
				if( this_set->markSize< 0 ){
					msx= msy= -this_set->markSize;
				}
				else{
				  double ms;
					if( NaNorInf(this_set->markSize) ){
						ms= ( wi->dev_info.mark_size_factor*( (int)(this_set->set_nr/internal_psMarkers)* psm_incr+ psm_base) );
					}
					else{
						ms= this_set->markSize* wi->dev_info.mark_size_factor;
					}
					msx= ms* wi->XUnitsPerPixel;
					msy= ms* wi->YUnitsPerPixel;
				}
				if( Check_Ignore_NaN(ignNaN, (v= sx1-msx)) ){
						SS_Add_Data_( wi->SS_X, 1, v, 1.0);
				}
				if( Check_Ignore_NaN(ignNaN, (v= sx1+msx)) ){
						SS_Add_Data_( wi->SS_X, 1, v, 1.0);
				}
				if( Check_Ignore_NaN(ignNaN, (v= sy1-msy)) ){
						SS_Add_Data_( wi->SS_Y, 1, v, 1.0);
				}
				if( Check_Ignore_NaN(ignNaN, (v= sy1+msy)) ){
						SS_Add_Data_( wi->SS_Y, 1, v, 1.0);
				}
				this_set->markSize= mS;
			}
			else{
				if( Check_Ignore_NaN(ignNaN, sx3) ) SS_Add_Data_( wi->SS_X, 1, sx3, 1.0);
				if( Check_Ignore_NaN(ignNaN, sy3) ) SS_Add_Data_( wi->SS_Y, 1, sy3, 1.0);
				if( Check_Ignore_NaN(ignNaN, sx4) ) SS_Add_Data_( wi->SS_X, 1, sx4, 1.0);
				if( Check_Ignore_NaN(ignNaN, sy4) ) SS_Add_Data_( wi->SS_Y, 1, sy4, 1.0);
			}
			if( this_set->use_error && wi->error_type[this_set->set_nr]!= INTENSE_FLAG ){
				if( Check_Ignore_NaN( ignNaN, (v=MIN(sy3,sy4)) ) ){
					SS_Add_Data_( wi->SS_LY, 1, v, 1.0);
				}
				if( Check_Ignore_NaN( ignNaN, (v=MAX(sy3,sy4)) ) ){
					SS_Add_Data_( wi->SS_HY, 1, v, 1.0);
				}
			}
			else{
				SS_Add_Data_( wi->SS_LY, 1, sy1, 1.0);
				SS_Add_Data_( wi->SS_HY, 1, sy1, 1.0);
			}
		}
		else{
			if( this_set->use_error && wi->error_type[this_set->set_nr]!= INTENSE_FLAG ){
				if( Check_Ignore_NaN(ignNaN, sy3) ) SS_Add_Data_( wi->SS_LY, 1, sy3, 1.0);
				if( Check_Ignore_NaN(ignNaN, sy4) ) SS_Add_Data_( wi->SS_HY, 1, sy4, 1.0);
			}
			else{
				SS_Add_Data_( wi->SS_LY, 1, sy1, 1.0);
				SS_Add_Data_( wi->SS_HY, 1, sy1, 1.0);
			}
		}
	}
	else{
		SS_Add_Data_( wi->SS_LY, 1, sy1, 1.0);
		SS_Add_Data_( wi->SS_HY, 1, sy1, 1.0);
	}
}

/* extern void CollectPenPosStats( LocalWin *wi, XGPenPosition *pos );	*/

XSegment *make_arrow_point1( LocalWin *wi, DataSet *this_set, double xp, double yp, double ang, double alen, double aspect )
{ static XSegment arr[2];
  static double prev_alen= 1;
	if( !NaNorInf(ang) ){
	  double x1, y1, x3, y3, c, s, r_aspect;
		if( NaNorInf(alen) ){
		  double p;
			if( Inf(alen)== -1 && prev_alen> 0 ){
				p= -prev_alen;
			}
			else{
				p= prev_alen;
			}
#ifndef linux
			if( debugFlag )
#endif
			{
				fprintf( StdErr, "make_arrow_point1(): requested alen=%s; using cached previous value %s\n",
					d2str( alen, NULL, NULL), d2str( p, NULL, NULL)
				);
			}
			alen= p;
		}
		else{
			prev_alen= alen;
		}
		x3= x1= xp- alen;
		y1= yp+ alen/3.0;
		y3= yp- alen/3.0;
		if( aspect> 1 ){
			r_aspect= 1.0/ aspect;
			aspect= 1;
		}
		else{
			r_aspect= 1;
		}
		c= Cos( (-ang) )* r_aspect;
		s= Sin( (-ang) )* aspect;
		arr[0].x1= (short)( xp+ (x1-xp)* c- (y1-yp)* s+ 0.5);
		arr[0].y1= (short)( yp+ (x1-xp)* s+ (y1-yp)* c+ 0.5);
		arr[0].x2= (short)( xp+ 0.5);
		arr[0].y2= (short)( yp+ 0.5);
		arr[1].x1= arr[0].x2;
		arr[1].y1= arr[0].y2;
#define test_linux_pgcc
#define linux_pgcc_bug_yell
#if defined(linux) && !defined(test_linux_pgcc)
{
		  /* RJB 20000212: gcc version pgcc-2.91.66 19990314 (egcs-1.1.2 release), Linux Mandrake 6.1:
		   \ Weird. There are floating point calculation errors that occur in -O compiled code,
		   \ resulting in nan values. This seems to arise in some cases when evaluating an
		   \ expression of the type (a-b)*c or (a-b)/c. Thus, in the code below, I had
		   \ to change (x3-xp)*c by x3*c-xp*c to make it work. Note that similar expressions
		   \ (x3-xp)*s, (x1-xp)*c, etc. did *not* require this, at least not that I have yet
		   \ noticed...
		   \ Something similar happens in Elapsed_Since.
		   \ Some examples:
				make_arrow_point1(x=314,y=292,ang=203.649,alen=11,aspect=1:0.987132) => (nan,nan)
				make_arrow_point1(x=327,y=298,ang=62.0977,alen=11,aspect=1:0.987296) => (nan,nan)
				make_arrow_point1(x=350,y=241,ang=330.218,alen=11,aspect=1:0.987296) => (nan,nan)
				make_arrow_point1(x=468,y=330,ang=275.032,alen=11,aspect=0.987296:1) => (nan,nan)
				make_arrow_point1(x=526,y=330,ang=275.031,alen=11,aspect=0.987296:1) => (nan,nan)
				make_arrow_point1(x=80,y=326,ang=230.576,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=80,y=326,ang=230.576,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=287,y=341,ang=335.927,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=367,y=294,ang=215.105,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=296,y=263,ang=326.435,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=445,y=195,ang=95.1724,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=388,y=315,ang=18.1069,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=445,y=195,ang=95.1724,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=388,y=315,ang=18.1069,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=406,y=74,ang=22.0371,alen=11,aspect=0.987132:1) => (nan,nan)
				make_arrow_point1(x=378,y=74,ang=37.7699,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=419,y=74,ang=13.8921,alen=11,aspect=1:1) => (nan,nan)
				make_arrow_point1(x=124,y=147,ang=213.825,alen=11,aspect=1:0.987132) => (nan,nan)
				make_arrow_point1(x=446,y=199,ang=271.654,alen=11,aspect=0.987132:1) => (nan,nan)
				make_arrow_point1(x=337,y=341,ang=348.996,alen=11,aspect=1:0.987132) => (nan,nan)
				make_arrow_point1(x=80,y=326,ang=230.576,alen=11,aspect=1:1) => (nan,nan)
			\
		   */
#ifdef linux_pgcc_bug_yell
  double x2, y2;
		x2= xp+ x3*c-xp* c- (y3-yp)* s;
		y2= yp+ (x3-xp)* s+ (y3-yp)* c;
		if( NaNorInf(x2) || NaNorInf(y2) ){
			fprintf( StdErr, "make_arrow_point1(x=%s,y=%s,ang=%s,alen=%s,aspect=%s:%s) => (%s,%s)\n",
				d2str( xp, NULL, NULL), d2str( yp, NULL, NULL), d2str( ang, NULL, NULL),
				d2str( alen, NULL, NULL), d2str( aspect, NULL, NULL), d2str( r_aspect, NULL, NULL),
				d2str( x2, NULL, NULL), d2str( y2, NULL, NULL)
			);
		}
#endif
/* 		arr[1].x2= (short)( x2+ 0.5);	*/
/* 		arr[1].y2= (short)( y2+ 0.5);	*/
		arr[1].x2= (short)( xp+ x3*c-xp*c- (y3-yp)*s+ 0.5);
		arr[1].y2= (short)( yp+ x3*s-xp*s+ (y3-yp)*c+ 0.5);
/* 		fprintf( StdErr, "x2=xp+ (x3-xp)*c- (y3-yp)*s= %g+ (%g-%g)*%g- (%g-%g)*%g=%g\n",	*/
/* 			xp, x3, xp, c, y3, yp, s, x2	*/
/* 		);	*/
/* 		fprintf( StdErr, "Arrow-head tip@(%g,%g),%gdeg,len=%g,aspect=%g: (%hd,%hd)-(%hd,%hd),(%hd,%hd)-(%hd,%hd)\n",	*/
/* 			xp, yp, ang, alen, aspect,	*/
/* 			arr[0].x1, arr[0].y1, arr[0].x2, arr[0].y2,	*/
/* 			arr[1].x1, arr[1].y1, arr[1].x2, arr[1].y2	*/
/* 		);	*/
}
#else
		arr[1].x2= (short)( xp+ (x3-xp)* c- (y3-yp)* s+ 0.5);
		arr[1].y2= (short)( yp+ (x3-xp)* s+ (y3-yp)* c+ 0.5);
/* 		fprintf( StdErr, "Arrow-head tip@(%g,%g),%gdeg,len=%g,aspect=%g: (%hd,%hd)-(%hd,%hd),(%hd,%hd)-(%hd,%hd)\n",	*/
/* 			xp, yp, ang, alen, aspect,	*/
/* 			arr[0].x1, arr[0].y1, arr[0].x2, arr[0].y2,	*/
/* 			arr[1].x1, arr[1].y1, arr[1].x2, arr[1].y2	*/
/* 		);	*/
#endif
		return( arr );
	}
	return( NULL );
}


void Check_Columns(DataSet *this_set)
{ int i, I;
  LocalWindows *WL;
	if( this_set ){
		i= this_set->set_nr;
		I= i+ 1;
	}
	else{
		i= 0;
		I= setNumber;
	}
	for( ; i< I; i++ ){
	  LocalWin *lwi;
		this_set= &AllSets[i];
		if( this_set->ncols> MaxCols ){
			MaxCols= this_set->ncols;
		}
		WL= WindowList;
		while( WL && WL->wi ){
			lwi= WL->wi;
			CLIP( lwi->xcol[i], 0, this_set->ncols- 1 );
			CLIP( lwi->ycol[i], 0, this_set->ncols- 1 );
			CLIP( lwi->ecol[i], 0, this_set->ncols- 1 );
			CLIP( lwi->lcol[i], -1, this_set->ncols- 1 );
			WL= WL->next;
		}
	}
}

extern XRectangle *rect_xywh(int x, int y, int width, int height );
extern XRectangle *rect_diag2xywh(int x1, int y1, int x2, int y2 );
extern XRectangle *rect_xsegs2xywh(int ns, XSegment *segs );

/* Determine this_set's intensitycolour for <value>, returning the
 \ to-be-restored colour in *colorx, and how to restore it in *respix
 */
RGB *xg_IntRGB= NULL;
void Retrieve_IntensityColour( LocalWin *wi, DataSet *this_set, double value,
	double minIntense, double maxIntense, double scaleIntense, Pixel *colorx, int *respix
)
{
	if( !IntensityColourFunction.XColours ){
		Default_Intensity_Colours();
	}
	if( IntensityColourFunction.XColours ){
	  double c= (value- minIntense)* scaleIntense;
	  int ci;
		if( IntensityColourFunction.range_set && minIntense== maxIntense ){
			CLIP_EXPR( c, value+ 0.5, 0, IntensityColourFunction.NColours-1 );
			ci= (int) c;
		}
		else if( !NaNorInf(c) ){
			CLIP_EXPR( ci, (int) (c+ 0.5), 0, IntensityColourFunction.NColours-1 );
		}
		else{
			ci= IntensityColourFunction.NColours-1;
		}
		if( this_set->pixvalue< 0 ){
			*colorx= this_set->pixelValue;
			this_set->pixelValue= IntensityColourFunction.XColours[ci].pixel;
		}
		else{
			*colorx= AllAttrs[this_set->pixvalue].pixelValue;
			AllAttrs[this_set->pixvalue].pixelValue= IntensityColourFunction.XColours[ci].pixel;
		}
		IntensityColourFunction.last_read= ci;
		if( IntensityColourFunction.exactRGB ){
			xg_IntRGB= &(IntensityColourFunction.exactRGB[ci]);
		}
		*respix= True;
	}
}

void Draw_Bar( LocalWin *wi, XRectangle *rec, XSegment *line, double barPixels, int barType, int LStyle,
	DataSet *this_set, int set_nr, int pnt_nr, int lwidth, int lstyle, int olwidth, int olstyle,
	double minIntense, double maxIntense, double scaleIntense, Pixel colorx, int respix
)
{
	switch( barType ){
		case 4:
		case 2:
		case 1:{
			if( wi->legend_line[set_nr].highlight ){
			  XSegment hs[5];
				hs[0].x1= rec->x;
				hs[0].y1= rec->y;
				hs[0].x2= rec->x+ rec->width;
				hs[0].y2= hs[0].y1;
				hs[1].x1= hs[0].x2;
				hs[1].y1= hs[0].y2;
				hs[1].x2= hs[1].x1;
				hs[1].y2= rec->y+ rec->height;
				hs[2].x1= hs[1].x2;
				hs[2].y1= hs[1].y2;
				hs[2].x2= hs[0].x1;
				hs[2].y2= hs[2].y1;
				hs[3].x1= hs[2].x2;
				hs[3].y1= hs[2].y2;
				hs[4].x1= hs[3].x2= hs[3].x1;
				hs[4].y1= hs[3].y2= hs[0].y1;
				hs[4].x2= hs[4].x1+ 1;
				hs[4].y2= hs[4].y1;
				HighlightSegment( wi, set_nr, 5, hs, HL_WIDTH(olwidth), LStyle );
			}
			if( barType== 4 ){
				if( pnt_nr>= 0 ){
					Retrieve_IntensityColour( wi, this_set, this_set->errvec[pnt_nr],
						minIntense, maxIntense, scaleIntense, &colorx, &respix
					);
					  /* 20031109: the line below causes the bar outlines to have the intensity colour,
					   \ which is *not* the intention.
					psThisRGB= xg_IntRGB;
					   */
				}
				  /* Filling is with the intensitycolour, which is now stored in whatever
				   \ used to contain the set's line (etc.) colour. This latter colour
				   \ is "backedup" in colorx...
				   */
				wi->dev_info.xg_rect( wi->dev_info.user_state,
					rec,
					lwidth, LStyle, lstyle, -1, colorx,
					1, PIXVALUE(set_nr),
					this_set
				);
				if( respix ){
					if( this_set->pixvalue< 0 ){
						this_set->pixelValue= colorx;
					}
					else{
						AllAttrs[this_set->pixvalue].pixelValue= colorx;
					}
				}
			}
			else{
				wi->dev_info.xg_rect( wi->dev_info.user_state,
					rec,
					lwidth, LStyle, lstyle, PIXVALUE(set_nr),
					(barType== 2)? 1 : 0,
					-1, (wi->legend_line[set_nr].pixvalue< 0)? wi->legend_line[set_nr].pixelValue : highlightPixel,
					this_set
				);
			}
			break;
		}
		case 0:
		case 3:{
		  double bP;
		  int olw;
			if( wi->legend_line[set_nr].highlight ){
			  double w= olwidth;
				olw= (w= (HL_WIDTH(w) /* - w */));
				w+= barPixels;
				HighlightSegment( wi, set_nr, 1, line, w, LStyle );
			}
			else{
				olw= olwidth;
			}
			  /* Compensate the solid line's width for the width of the outline;
			   \ the outline is drawn with rounded edges!
			   */
			if( olw> 2 ){
				bP= barPixels- olw+ 2;
			}
			else{
				bP= barPixels;
			}
			if( olw> 1 ){
				olw-= 1;
			}
			if( line->y1> line->y2 ){
				line->y1-= olw;
				line->y2+= olw;
			}
			else if( line->y1< line->y2 ){
				line->y1+= olw;
				line->y2-= olw;
			}
			else{
				if( line->x1> line->x2 ){
					line->x1-= olw;
					line->x2+= olw;
				}
				else if( line->x1< line->x2 ){
					line->x1+= olw;
					line->x2-= olw;
				}
			}
			{ int lpixval, fpixval;
			  Pixel lpixel, fpixel;
				if( barType== 0 ){
					fpixval= lpixval= this_set->pixvalue;
					fpixel= lpixel= this_set->pixelValue;
				}
				else{
					fpixval= -1;
					fpixel= (wi->legend_line[set_nr].pixvalue< 0)? wi->legend_line[set_nr].pixelValue : highlightPixel,
					lpixval= this_set->pixvalue;
					lpixel= this_set->pixelValue;
				}
				wi->dev_info.xg_seg(wi->dev_info.user_state,
							1, line, bP, LStyle,
							lstyle, fpixval, fpixel, this_set);
				  /* Draw a border around. Not necessary for solid bars, but we
				   \ do it anyway to get identical behaviour.
				   */
				wi->dev_info.xg_rect( wi->dev_info.user_state,
					rec,
					olwidth, LStyle, olstyle, lpixval, lpixel,
					0, -1, 0, this_set
				);
			}
			break;
		}
		case 5:
		case 6:{
		  XSegment hook[2];
		  double bP, hl_width= HL_WIDTH(lwidth);
		  int olw;
			olw= lwidth;
			  /* 20020318: Compensate the solid line's width for the width of the outline;
			   \ the outline is drawn with rounded edges! This compensation is (of course...)
			   \ slightly different than for "regular" bars.
			   */
			if( olw> 2 ){
				bP= (barPixels- olw+ 2)* wi->dev_info.var_width_factor;
			}
			else{
				bP= barPixels* wi->dev_info.var_width_factor;
			}
			olw*= wi->dev_info.var_width_factor/2;
			if( line->y1> line->y2 ){
				line->y2+= olw;
			}
			else if( line->y1< line->y2 ){
				line->y2-= olw;
			}
			else{
				if( line->x1> line->x2 ){
					line->x1-= olw;
					line->x2+= olw;
				}
				else if( line->x1< line->x2 ){
					line->x1+= olw;
					line->x2-= olw;
				}
			}
			hook[1].x1= hook[0].x2= hook[0].x1= (this_set->barType== 5)? line->x1- bP/2 : line->x1+ bP/2;
			hook[0].y1= line->y1;
			hook[1].y2= hook[1].y1= hook[0].y2= line->y2;
			hook[1].x2= line->x1;
			if( wi->legend_line[set_nr].highlight ){
				HighlightSegment( wi, set_nr, 2, hook, hl_width, LStyle);
			}
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						2, hook,
						lwidth, LStyle,
						lstyle, this_set->pixvalue, this_set->pixelValue, this_set);
			break;
		}
	}
}

void Draw_valueMark( LocalWin *wi, DataSet *this_set, int pnt_nr,
	short sx, short sy, int colour, Pixel pixval
)
{ Pixel colour0= AllAttrs[0].pixelValue;
  char *text= NULL, x[128]= "", y[128], e[128], *join= "";
  double X, rY, Y, E;
	if( DiscardedPoint( wi, this_set, pnt_nr ) ){
		return;
	}
	if( colour< 0 ){
		AllAttrs[0].pixelValue= pixval;
	}
	else{
		AllAttrs[0].pixelValue= AllAttrs[colour % MAXATTR].pixelValue;
	}
	if( CheckMask( this_set->valueMarks, VMARK_RAW) ){
		X= XVAL( this_set, pnt_nr );
		Y= YVAL( this_set, pnt_nr );
		E= ERROR( this_set, pnt_nr );
	}
	else{
		X= this_set->xvec[pnt_nr];
		Y= this_set->yvec[pnt_nr];
		E= this_set->error[pnt_nr];
	}
	rY= this_set->yvec[pnt_nr];
	if( this_set->barFlag> 0 ){
		WriteValue( wi, y, Y, X, 0, 0, 0, Y_axis, 0, 0, 127 );
		text= concat2( text, y, NULL );
	}
	else{
		WriteValue( wi, x, X, Y, 0, 0, 0, X_axis, 0, 0, 127 );
		WriteValue( wi, y, Y, X, 0, 0, 0, Y_axis, 0, 0, 127 );
		if( wi->polarFlag ){
			text= concat2( text, x, NULL );
		}
		else{
			text= concat2( text, "(", x, ",", y, NULL );
		}
	}
	if( CheckMask( this_set->valueMarks, VMARK_FULL) && this_set->use_error ){
		WriteValue( wi, e, E, X, 0, 0, 0, Y_axis, 0, 0, 127 );
		switch( wi->error_type[ this_set->set_nr ] ){
			case 4:{
			  char pc[]= " \\#xd0\\ ";
				join= parse_codes(pc);
				break;
			}
			case INTENSE_FLAG:
			case MSIZE_FLAG:
				join= " , ";
				break;
			default:{
			  char pc[]= " \\#xb1\\ ";
				join= parse_codes( pc );
				break;
			}
		}
		text= concat2( text, join, e, (this_set->barFlag)? 0 : ")", NULL );
	}
	else{
		text= concat2( text, (this_set->barFlag)? 0 : ")", NULL );
	}
	if( text ){
		if( rY< 0 ){
			wi->dev_info.xg_text( wi->dev_info.user_state,
				sx, sy+ wi->dev_info.bdr_pad,
				text,
				T_TOP, T_AXIS, NULL
			);
		}
		else{
			wi->dev_info.xg_text( wi->dev_info.user_state,
				sx, sy- wi->dev_info.bdr_pad,
				text,
				T_BOTTOM, T_AXIS, NULL
			);
		}
	}
	xfree( text );

	AllAttrs[0].pixelValue= colour0;
}

void Draw_ErrorBar( LocalWin *wi, DataSet* this_set, int set_nr, int pX_set_nr, int X_set_nr, int pnt_nr, int first,
	double ebarPixels, int LStyle,
	double aspect
)
{ XSegment ebar[3], line[2];	/* hor.l, hor.h	*/
  double elinepixs = ELINEWIDTH(wi,set_nr);
  double hl_bwidth;
  int estyle=ELINESTYLE(set_nr),
  	pnr= (first)? SegsENR(XsegsE,X_set_nr-1)->pnt_nr1 : SegsENR(XsegsE,X_set_nr-1)->pnt_nr2,
	ok;

	if( wi->polarFlag && wi->error_type[set_nr]!= 1 ){
		return;
	}
	  /* 20030630: *do* accept elinepixs==0!!! */
	if( elinepixs < 0 ){
		elinepixs = 1;
	}
	hl_bwidth= HL_WIDTH(elinepixs)/ 2+ 1;
	ps_LStyle= L_AXIS;
/* 					use_ps_LStyle= 1;	*/
	if( wi->triangleFlag ){
	  int triangle_width;
		if( this_set->ebarWidth_set ){
			triangle_width= ebarPixels* wi->dev_info.var_width_factor;
		}
		else if( PS_STATE(wi)->Printing== PS_PRINTING){
			triangle_width= (int) (2* ps_MarkSize_X( (struct userInfo*) wi->dev_info.user_state, this_set->set_nr)+ elinepixs- 1);
		}
		else{
			triangle_width= (int) (2* X_ps_MarkSize_X(this_set->set_nr)+ elinepixs- 1);
		}
		if( !this_set->current_ebW_set ){
			this_set->current_ebarWidth= triangle_width* (wi->XUnitsPerPixel)/ wi->Xscale;
			this_set->current_ebW_set= True;
		}
		if( first ){
			ebar[0].x1 = SegsENR(XsegsE,X_set_nr-1)->X1;
			ebar[0].y1 = SegsENR(XsegsE,X_set_nr-1)->Y1;
		}
		else{
			ebar[0].x1 = SegsENR(XsegsE,X_set_nr-1)->X2;
			ebar[0].y1 = SegsENR(XsegsE,X_set_nr-1)->Y2;
		}
		  /* Triangular errorthings are "squashed" against the border of the drawing region.
		   \ That's a simple, and not entirely correct way of clipping.
		   */
		ok= 0;
		if( first ){
			if( SegsENR(XsegsE,X_set_nr-1)->Y1 != SegsENR(XsegsE,X_set_nr-1)->y1l ){
				ebar[0].x2 = SegsENR(XsegsE,X_set_nr-1)->x1- triangle_width / 2;
				ebar[0].y2 = SegsENR(XsegsE,X_set_nr-1)->y1l;
				ebar[1].x2 = SegsENR(XsegsE,X_set_nr-1)->x1+ triangle_width / 2;
				ok= 1;
			}
		}
		else if( SegsENR(XsegsE,X_set_nr-1)->Y2 != SegsENR(XsegsE,X_set_nr-1)->y2l ){
			ebar[0].x2 = SegsENR(XsegsE,X_set_nr-1)->x2- triangle_width / 2;
			ebar[0].y2 = SegsENR(XsegsE,X_set_nr-1)->y2l;
			ebar[1].x2 = SegsENR(XsegsE,X_set_nr-1)->x2+ triangle_width / 2;
			ok= 1;
		}
		if( ok ){
			ebar[1].x1 = ebar[0].x2;
			ebar[1].y1 = ebar[0].y2;
			ebar[1].y2 = ebar[1].y1;
			ebar[2].x1 = ebar[1].x2;
			ebar[2].y1 = ebar[1].y2;
			ebar[2].x2 = ebar[0].x1;
			ebar[2].y2 = ebar[0].y1;
			  /* If error_point==-5, the first and last error are drawn highlighted if the rest of the set
			   \ is not highlighted, or non-highlighted if the rest of the set is. The current pointnumber
			   \ is retrieved from XsegsE. The decision whether or not to highlight is implemented using
			   \ a "simulated" Boolean XOR (which doesn't officially exist in C): a binary XOR operating on
			   \ true Boolean (i.e. 1 or 0) values.
			   */
			if( (wi->legend_line[set_nr].highlight!=0) ^
				(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
			){
				HighlightSegment( wi, set_nr, 3, ebar, hl_bwidth, LStyle );
			}
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						3, ebar, elinepixs, LStyle,
						estyle, PIXVALUE(set_nr), this_set
			);
		}
		ok= 0;
		if( first ){
			if( SegsENR(XsegsE,X_set_nr-1)->Y1 != SegsENR(XsegsE,X_set_nr-1)->y1h ){
				ebar[0].x2 = SegsENR(XsegsE,X_set_nr-1)->x1- triangle_width / 2;
				ebar[0].y2 = SegsENR(XsegsE,X_set_nr-1)->y1h;
				ebar[1].x2 = SegsENR(XsegsE,X_set_nr-1)->x1+ triangle_width / 2;
				ok= 1;
			}
		}
		else if( SegsENR(XsegsE,X_set_nr-1)->Y2 != SegsENR(XsegsE,X_set_nr-1)->y2h ){
			ebar[0].x2 = SegsENR(XsegsE,X_set_nr-1)->x2- triangle_width / 2;
			ebar[0].y2 = SegsENR(XsegsE,X_set_nr-1)->y2h;
			ebar[1].x2 = SegsENR(XsegsE,X_set_nr-1)->x2+ triangle_width / 2;
			ok= 1;
		}
		if( ok ){
			ebar[1].x1 = ebar[0].x2;
			ebar[1].y1 = ebar[0].y2;
			ebar[1].y2 = ebar[1].y1;
			ebar[2].x1 = ebar[1].x2;
			ebar[2].y1 = ebar[1].y2;
			ebar[2].x2 = ebar[0].x1;
			ebar[2].y2 = ebar[0].y1;
			if( (wi->legend_line[set_nr].highlight!=0) ^
				(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
			){
				HighlightSegment( wi, set_nr, 3, ebar, hl_bwidth, LStyle );
			}
			wi->dev_info.xg_seg(wi->dev_info.user_state,
						3, ebar, elinepixs, LStyle,
						estyle, PIXVALUE(set_nr), this_set
			);
		}
	}
	else if( !first || (pX_set_nr!= X_set_nr) ){
	  double w;
		if( this_set->ebarWidth_set ){
			w= ebarPixels* wi->dev_info.var_width_factor;
		}
		else{
			w= wi->dev_info.errortick_len + elinepixs- 1;
		}
		if( !this_set->current_ebW_set ){
			this_set->current_ebarWidth= w* (wi->XUnitsPerPixel)/ wi->Xscale;
			this_set->current_ebW_set= True;
		}
		if( first ){
			ebar[0].x1 = SegsENR(XsegsE,X_set_nr-1)->x1;
			ebar[0].x2 = SegsENR(XsegsE,X_set_nr-1)->x1h;
			ebar[0].y1 = SegsENR(XsegsE,X_set_nr-1)->y1l;
			ebar[0].y2 = SegsENR(XsegsE,X_set_nr-1)->y1h;
		}
		else{
			ebar[0].x1 = SegsENR(XsegsE,X_set_nr-1)->x2;
			ebar[0].x2 = SegsENR(XsegsE,X_set_nr-1)->x2h;
			ebar[0].y1 = SegsENR(XsegsE,X_set_nr-1)->y2l;
			ebar[0].y2 = SegsENR(XsegsE,X_set_nr-1)->y2h;
		}
		if( wi->error_type[set_nr]== 7 ){
		  XRectangle erec;
		  int barType= (this_set->barType== 5 || this_set->barType== 6)? 1 : this_set->barType;
		  int elw= ELINEWIDTH(wi,set_nr);
			if( first ){
				erec= *rect_xywh(
					(int) (SegsENR(XsegsE,X_set_nr-1)->x1- w/2+ 0.5),
					SegsENR(XsegsE,X_set_nr-1)->y1l,
					(int) w,
					SegsENR(XsegsE,X_set_nr-1)->y1h- SegsENR(XsegsE,X_set_nr-1)->y1l
				);
			}
			else{
				erec= *rect_xywh(
					(int) (SegsENR(XsegsE,X_set_nr-1)->x2- w/2+ 0.5),
					SegsENR(XsegsE,X_set_nr-1)->y2l,
					(int) w,
					SegsENR(XsegsE,X_set_nr-1)->y2h- SegsENR(XsegsE,X_set_nr-1)->y2l
				);
			}
			Draw_Bar( wi, &erec, ebar, (this_set->ebarWidth_set)? ebarPixels : w, barType, LStyle,
				this_set, set_nr,pnt_nr,
				elw, ELINESTYLE(set_nr), elw, 0,
				0, 0, 0, 0, 0
			);
			return;
		}
		if( (wi->legend_line[set_nr].highlight!=0) ^
			(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
		){
			HighlightSegment( wi, set_nr, 1, &ebar[0], hl_bwidth, LStyle );
		}
		wi->dev_info.xg_seg(wi->dev_info.user_state,
					1, &ebar[0], elinepixs, LStyle,
					estyle, PIXVALUE(set_nr), this_set
		);
		if( !wi->polarFlag ){
			if( !wi->vectorFlag ){
				double etermPixels= MAX( elinepixs/2, 1),
					hl_btwidth= HL_WIDTH(etermPixels)/ 2+ 1;
				if( first ){
					line[0].x1 = SegsENR(XsegsE,X_set_nr-1)->x1- w / 2;
					line[0].x2 = line[0].x1 + w;
					line[1].x1= line[0].x1;
					line[1].x2= line[1].x1 + w;
					line[0].y1 = line[0].y2= ebar[0].y1;
					line[1].y1 = line[1].y2= ebar[0].y2;
					  /* Errorbars are correctly clipped. That is, they do not get the terminator
					   \ when they are clipped (i.e. when the terminator is outside the drawing region.
					   */
					if( SegsENR(XsegsE,X_set_nr-1)->e_mark_inside1 &&
						SegsENR(XsegsE,X_set_nr-1)->Y1 != SegsENR(XsegsE,X_set_nr-1)->y1l
					){
						if( (wi->legend_line[set_nr].highlight!=0) ^
							(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
						){
							HighlightSegment( wi, set_nr, 1, &line[0], hl_btwidth, LStyle );
						}
						wi->dev_info.xg_seg(wi->dev_info.user_state,
									1, &line[0], etermPixels, LStyle,
									0, PIXVALUE(set_nr), this_set
						);
					}
					if( SegsENR(XsegsE,X_set_nr-1)->e_mark_inside2 &&
						SegsENR(XsegsE,X_set_nr-1)->Y1 != SegsENR(XsegsE,X_set_nr-1)->y1h
					){
						if( (wi->legend_line[set_nr].highlight!=0) ^
							(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
						){
							HighlightSegment( wi, set_nr, 1, &line[1], hl_btwidth, LStyle );
						}
						wi->dev_info.xg_seg(wi->dev_info.user_state,
									1, &line[1], etermPixels, LStyle,
									0, PIXVALUE(set_nr), this_set
						);
					}
				}
				else{
					line[0].x1 = SegsENR(XsegsE,X_set_nr-1)->x2- w / 2;
					line[0].x2 = line[0].x1 + w;
					line[1].x1= line[0].x1;
					line[1].x2= line[1].x1 + w;
					line[0].y1 = line[0].y2= ebar[0].y1;
					line[1].y1 = line[1].y2= ebar[0].y2;
					if( SegsENR(XsegsE,X_set_nr-1)->e_mark_inside1 &&
						SegsENR(XsegsE,X_set_nr-1)->Y2 != SegsENR(XsegsE,X_set_nr-1)->y2l
					){
						if( (wi->legend_line[set_nr].highlight!=0) ^
							(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
						){
							HighlightSegment( wi, set_nr, 1, &line[0], hl_btwidth, LStyle );
						}
						wi->dev_info.xg_seg(wi->dev_info.user_state,
									1, &line[0], etermPixels, LStyle,
									0, PIXVALUE(set_nr), this_set
						);
					}
					if( SegsENR(XsegsE,X_set_nr-1)->e_mark_inside2 &&
						SegsENR(XsegsE,X_set_nr-1)->Y2 != SegsENR(XsegsE,X_set_nr-1)->y2h
					){
						if( (wi->legend_line[set_nr].highlight!=0) ^
							(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
						){
							HighlightSegment( wi, set_nr, 1, &line[1], hl_btwidth, LStyle );
						}
						wi->dev_info.xg_seg(wi->dev_info.user_state,
									1, &line[1], etermPixels, LStyle,
									0, PIXVALUE(set_nr), this_set
						);
					}
				}
			}
			else{
			  double ang= this_set->errvec[pnt_nr], vlength= this_set->lvec[pnt_nr];
				switch( this_set->vectorType ){
					case 1:{
					  double x1, y1, x2, y2;
					  double alen;
					  XSegment *arr;
						vlength= this_set->vectorLength;
					case 3:
					case 4:
						if( first ){
							x1 = SegsENR(XsegsE,X_set_nr-1)->x1;
							y1 = SegsENR(XsegsE,X_set_nr-1)->y1l;
							x2 = SegsENR(XsegsE,X_set_nr-1)->x1h;
							y2 = SegsENR(XsegsE,X_set_nr-1)->y1h;
						}
						else{
							x1 = SegsENR(XsegsE,X_set_nr-1)->x2;
							y1 = SegsENR(XsegsE,X_set_nr-1)->y2l;
							x2 = SegsENR(XsegsE,X_set_nr-1)->x2h;
							y2 = SegsENR(XsegsE,X_set_nr-1)->y2h;
						}
						if( this_set->vectorPars[1] ){
							alen= sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) )/ this_set->vectorPars[1];
						}
						else{
							alen= 0;
						}
						  /* This is a quick hack to implement arrow heads that depend on
						   \ this_set->vectorLength while the arrow's length depends on lval/lvec ($DATA{3}).
						   \ Divide by lval or lvec (i.e. vlength), and multiply by this_set->vectorLength.
						   \ Don't do anything for vlength==0: there won't be an arrow anyway.
						   */
						if( this_set->vectorType== 4 ){
							if( vlength ){
								alen*= ABS(this_set->vectorLength/vlength);
							}
						}
						  /* 20020424: here we must account for vectorType=3,4! */
						if( vlength< 0 ){
						  /* The thing is to point in the other direction. The co-ordinates x2,y2
						   \ are already correct (i.e. that point is to receive the arrowhead), only
						   \ the orientation must be inverted.
						   */
							ang+= wi->radix/ 2;
						}
#if defined(linux) && defined(__GNUC__)
/* 						make_arrow_point1( wi, this_set, x2, y2, ang, alen, aspect );	*/
#endif
						if( (arr= make_arrow_point1( wi, this_set, x2, y2, ang, alen, aspect )) ){
							if( (wi->legend_line[set_nr].highlight!=0) ^
								(this_set->error_point==-5 && (pnr==this_set->first_error || pnr== this_set->last_error))
							){
								HighlightSegment( wi, set_nr, 2, arr, hl_bwidth, LStyle );
							}
							wi->dev_info.xg_seg(wi->dev_info.user_state,
										2, arr, elinepixs, LStyle,
										estyle, PIXVALUE(set_nr), this_set
							);
						}
						break;
					}
				}
			}
		}
	}
}

double WindowAspect( LocalWin *wi )
{ double aspect;
	if( (wi->win_geo.bounds._hiY- wi->win_geo.bounds._loY) &&
		(wi->XOppX- wi->XOrgX)
	){
		aspect= ((wi->win_geo.bounds._hiX - wi->win_geo.bounds._loX) /
							(wi->win_geo.bounds._hiY - wi->win_geo.bounds._loY)) *
					(((double)wi->XOppY- wi->XOrgY)/ ((double)wi->XOppX- wi->XOrgX));
#if DEBUG==2
	fprintf( StdErr, "XOppY etc. aligment=%d; (double)XOppY alignment=%d, expr. alignment=%d\n",
		__alignof__(wi->XOppY), __alignof__( (double)wi->XOppY), __alignof__( ((double)wi->XOppY- wi->XOrgY) )
	);
#endif
#if defined(i386) && defined(__GNUC__)
		if( NaNorInf(aspect) ){
		  double xrange, yrange;
			  /* 20001128: Whoops, another bug in gcc, again the problem with aligning. It seems that an integer
			   \ calculation/expression that gets promoted to double, within a double expression, does not get
			   \ properly aligned; the result of the above statement is a -NaN128. If this happens (and happen it
			   \ does!), we recalculate the expr. using 2 local variables (...)
			   */
			if( debugFlag ){
				fprintf( StdErr, "DrawData(): FPE aspect= ((%g-%g)/(%g-%g))* ((double)%d-%d)/((double)%d-%d)=%s(%g)\n",
					wi->win_geo.bounds._hiX, wi->win_geo.bounds._loX,
					wi->win_geo.bounds._hiY, wi->win_geo.bounds._loY,
					wi->XOppY, wi->XOrgY, wi->XOppX, wi->XOrgX,
					d2str(aspect,0,0), aspect
				);
			}
			aspect= (wi->win_geo.bounds._hiX - wi->win_geo.bounds._loX) / (wi->win_geo.bounds._hiY - wi->win_geo.bounds._loY);
			if( debugFlag ){
				fprintf( StdErr, "\t data aspect=%s;", d2str(aspect,0,0) );
			}
			if( NaNorInf(aspect) ){
				if( !debugFlag ){
					fprintf( StdErr, "DrawData(): window's aspect incorrectly evaluates to NaN;" );
					fprintf( StdErr, " aspect= ((%g-%g)/(%g-%g))=%s(%g)\n",
						wi->win_geo.bounds._hiX, wi->win_geo.bounds._loX,
						wi->win_geo.bounds._hiY, wi->win_geo.bounds._loY,
						d2str(aspect,0,0), aspect
					);
				}
				fprintf( StdErr, "\t change the flags to gcc to ensure better floating point alignment, and/or notify the gcc team!\n" );
			}
			else{
				yrange= (double) wi->XOppY- (double) wi->XOrgY;
				xrange= (double) wi->XOppX- (double) wi->XOrgX;
				aspect*= yrange/xrange;
				if( debugFlag ){
					fprintf( StdErr, "screen aspect= %g/%g=%s; total=%s\n",
						yrange, xrange, d2str(yrange/xrange, 0,0),
						d2str( aspect, 0,0)
					);
				}
			}
		}
#endif
	}
	else{
		aspect= 0;
	}
	return( aspect );
}

int DetermineBoundsOnly= False;

void DrawData(LocalWin *wi, Boolean bounds_only)
/*
 * This routine draws the data sets themselves using the macros
 * for translating coordinates.
 */
{ double sx1_1, sx1, sy1, sx2, sx3, sx4, sx5, sx6, sy2, sy3, sy4, sy5, sy6, PointsDone= 0,
		pbarPixels, epbarPixels, ebarPixels;
/* 	double *ldxvec, *ldyvec, *hdxvec, *hdyvec;	*/
    int cnr, set_nr, pnt_nr, pnt_nr_1, have_bars= 0, ncoords;
    int mark_inside1, mark_inside2;
	int clipcode1, clipcode2;
    int pX_set_nr, X_set_nr, X_seg_nr, X_seg_nr_diff, LY_set_nr, HY_set_nr, LStyle;
	int ok= 1, ok2= 1, connect_errrgn, connect_id, w_use_errors= wi->use_errors;
	int adorned, discarded_point, use_lcol;
    XSegment *ptr;
	DataSet *this_set;
	static LocalWinGeo win_geo;
	char printed= 0;
	SimpleStats SS_er= EmptySimpleStats, SS_er_r= EmptySimpleStats;
	SimpleStats set_X= EmptySimpleStats, set_Y= EmptySimpleStats, set_E= EmptySimpleStats,
		set_tr_X= EmptySimpleStats, set_tr_Y= EmptySimpleStats, set_tr_E= EmptySimpleStats;
	SimpleAngleStats set_O, set_tr_O;
	SimpleAngleStats SAS_er, SAS_er_r;
	double curve_len, error_len;
#ifdef TR_CURVE_LEN
	double tr_curve_len;
#endif
	double hl_width /* , hl_bwidth */;
	Boolean sn_set= False, bounds_clip= False;
	extern Boolean psSeg_disconnect_jumps;
	Boolean psSdj= psSeg_disconnect_jumps, adorn, set_init_pass= False, reinit_pass= False,
		initRYE;
	extern double conv_angle_();

	double aspect;

	Pixel dshadowPixel= zeroPixel ^ normPixel;
	static char active= 0;

	double minIntense, maxIntense, scaleIntense;

#ifdef DEBUG
	if( !wi ){
		fprintf( StdErr, "DrawData() called with NULL argument\n" );
		return;
	}
#endif
	if( wi->delete_it== -1 ){
		return;
	}

	if( DetermineBoundsOnly ){
		bounds_only= True;
	}

	if( wi->data_silent_process && !X_silenced(wi) ){
		wi->osilent_val= wi->dev_info.xg_silent( wi->dev_info.user_state, True );
	}

	aspect= WindowAspect(wi);

	if( !(wi->raw_once== -1 && wi->raw_display) || wi->use_transformed
	  /* 980925: I don't know whether the following condition is necessary/constructive	*/
		|| wi->process_bounds
	){
		  /* 20040129: do when necessary: */
		if( wi->SS_I.count /* wi->IntensityLegend.legend_needed && !wi->IntensityLegend.legend_placed */ ){
			SS_Reset_( wi->SS_I );
		}
	}

	SAS_er.Gonio_Base= wi->radix;
	SAS_er.Gonio_Offset= wi->radix_offset;
	SAS_er.Nvalues= 0;
	SAS_er.sample= NULL;
	SAS_er.exact= 0;
		SAS_Reset_(SAS_er);
	SAS_er_r.Gonio_Base= wi->radix;
	SAS_er_r.Gonio_Offset= wi->radix_offset;
	SAS_er_r.Nvalues= 0;
	SAS_er_r.sample= NULL;
	SAS_er_r.exact= 0;
		SAS_Reset_(SAS_er_r);
	set_O.Gonio_Base= wi->radix;
	set_O.Gonio_Offset= wi->radix_offset;
	set_O.Nvalues= 0;
	set_O.sample= NULL;
	set_O.exact= 0;
		SAS_Reset_(set_O);
	set_tr_O.Gonio_Base= wi->radix;
	set_tr_O.Gonio_Offset= wi->radix_offset;
	set_tr_O.Nvalues= 0;
	set_tr_O.sample= NULL;
	set_tr_O.exact= 0;
		SAS_Reset_(set_tr_O);

	if( wi->polarFlag && !bounds_only ){
		sx2= sy2= sx1= sy1= 0.0;
		if( ClipWindow( wi, NULL, False, &sx1, &sy1, &sx2, &sy2, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
			if( mark_inside1)
				wi->dev_info.xg_dot(wi->dev_info.user_state,
							SCREENX(wi, sx1), SCREENY(wi, sy1),
							P_DOT, 0, 0, 0, 0, NULL);
		}
	}
	wi->event_level++;
#ifdef DEBUG
	theWin_Geo= &wi->win_geo;
#endif
	ascanf_window= wi->window;
	*ascanf_ActiveWinWidth= wi->XOppX - wi->XOrgX;
	*ascanf_ActiveWinHeight= wi->XOppY - wi->XOrgY;
	Gonio_Base( wi, wi->radix, wi->radix_offset );

	if( wi->use_transformed ){
	  /* We don't do the transformations (unless one of the sets has its init_pass field set),
	   \ but we do want to have the possibility for run-time scaling of the axes. Thus, we
	   \ collect the statistics necessary for this.
	   */
		TitleMessage( wi, "Getting statistics on previously transformed values");
		for( set_nr = 0;  set_nr < setNumber;  set_nr++ ){
		  int etype= wi->error_type[set_nr];
			this_set= &AllSets[set_nr];
			initRYE= True;
			if( /* AllSets[set_nr].init_pass || */ wi->init_pass ){
				TitleMessage( wi, NULL );
				goto do_transforms;
			}
			if( draw_set(wi, set_nr) && !this_set->skipOnce ){
				if( this_set->init_pass ){
					TitleMessage( wi, NULL );
					goto do_transforms;
				}
				for( pnt_nr = 0;  pnt_nr < this_set->numPoints;  pnt_nr++ ){
					if( !DiscardedPoint( wi, this_set, pnt_nr ) ){
						if( wi->datawin.apply ){
						  int pok= True;
							if( !NaN(wi->datawin.llX) && this_set->xvec[pnt_nr]< wi->datawin.llX ){
								pok= False;
							}
							else if( !NaN(wi->datawin.urX) && this_set->xvec[pnt_nr]> wi->datawin.urX ){
								pok= False;
							}
							else if( !NaN(wi->datawin.llY) && this_set->yvec[pnt_nr]< wi->datawin.llY ){
								pok= False;
							}
							else if( !NaN(wi->datawin.urY) && this_set->yvec[pnt_nr]> wi->datawin.urY ){
								pok= False;
							}
							if( !pok ){
								goto stat_next_point;
							}
						}
						if( !this_set->floating ){
							CollectPointStats( wi, this_set, pnt_nr, this_set->xvec[pnt_nr], this_set->yvec[pnt_nr],
								this_set->ldxvec[pnt_nr], this_set->ldyvec[pnt_nr],
								this_set->hdxvec[pnt_nr], this_set->hdyvec[pnt_nr]
							);
							if( etype== 4 ){
								SAS_Add_Data_( wi->SAS_O, 1, this_set->errvec[pnt_nr], 1.0, (int) *SAS_converts_angle );
							}
							else if( etype!= INTENSE_FLAG ){
								SS_Add_Data_( wi->SS_E, 1, this_set->errvec[pnt_nr], 1.0 );
							}
						}
						  /* 20040129: update SS_I even if we don't want the IntensityLegend! */
						if( etype== INTENSE_FLAG /* wi->IntensityLegend.legend_needed */ ){
							SS_Add_Data_( wi->SS_I, 1, this_set->errvec[pnt_nr], 1.0 );
						}
						if( initRYE ){
							this_set->rawY.min= this_set->rawY.max= YVAL(this_set, pnt_nr);
							this_set->rawE.min= this_set->rawE.max= EVAL(this_set, pnt_nr);
							this_set->ripeY.min= this_set->ripeY.max= this_set->yvec[pnt_nr];
							this_set->ripeE.min= this_set->ripeE.max= this_set->errvec[pnt_nr];
							initRYE= False;
						}
						else{
							this_set->rawY.min= MIN( this_set->rawY.min, YVAL(this_set, pnt_nr));
							this_set->rawY.max= MAX( this_set->rawY.max, YVAL(this_set, pnt_nr));
							this_set->rawE.min= MIN( this_set->rawE.min, EVAL(this_set, pnt_nr));
							this_set->rawE.max= MAX( this_set->rawE.max, EVAL(this_set, pnt_nr));
							this_set->ripeY.min= MIN( this_set->ripeY.min, this_set->yvec[pnt_nr]);
							this_set->ripeY.max= MAX( this_set->ripeY.max, this_set->yvec[pnt_nr]);
							this_set->ripeE.min= MIN( this_set->ripeE.min, this_set->errvec[pnt_nr]);
							this_set->ripeE.max= MAX( this_set->ripeE.max, this_set->errvec[pnt_nr]);
						}
					}
stat_next_point:;
				}
				if( this_set->barFlag> 0 ){
					have_bars+= 1;
				}
			}
			else{
				if( debugFlag ){
					fprintf( StdErr, "DrawData(%d) skipping transformation of set #%d (%d points)\n",
						__LINE__, set_nr, this_set->numPoints
					);
				}
			}
		}
		TitleMessage( wi, NULL );
	}

	if( (wi->raw_once== -1 && wi->raw_display) || (!wi->raw_display && wi->use_transformed) ){
		if( debugFlag ){
			fprintf( StdErr, "Skipping transformations, using results of previous redraw/rescale\n" );
		}
		goto draw_points;
	}

do_transforms:;
	if( !bounds_only ){
		TitleMessage( wi, "Doing transformations..." );
	}
	wi->use_errors= 0;
    for( cnr = 0;  cnr < setNumber;  cnr++ ){
		  /* Unset skipOnce. In principle, we start treating a to-be-drawn set until this flag
		   \ is set to True.
		   */
		AllSets[cnr].skipOnce= False;
	}
    for( cnr = 0;  cnr < setNumber;  cnr++ ){
	  Boolean set_ok= True;
	  int etype;
		set_nr= DO( cnr );
		etype= wi->error_type[set_nr];
		this_set= &AllSets[set_nr];
		if( !this_set->setName || this_set->numPoints<= 0 || strncmp( this_set->setName, "*NOPLOT*",8)== 0 ){
			this_set->draw_set= 0;
			wi->draw_set[set_nr]= 0;
			set_ok= False;
		}
		if( !draw_set(wi, set_nr) && !AllSets[set_nr].init_pass && !wi->init_pass ){
			if( debugFlag ){
				fprintf( StdErr, "DrawData(%d) skipping transformation of set #%d (%d points)\n",
					__LINE__, set_nr, this_set->numPoints
				);
			}
		}
		else if( set_ok ){
		  LocalWin *pw= this_set->processing_wi;

			wi->processing_set= this_set;

			if( etype== -1 ){
				if( error_type== -1 ){
					error_type= !no_errors;
				}
				etype= wi->error_type[set_nr]= error_type;
			}
			  /* Make an educated guess about the number of adorned points in this set:	*/
			adorned= Adorned_Points( this_set );
			switch( etype ){
				case 0:
				case INTENSE_FLAG:
				case MSIZE_FLAG:
					w_use_errors= False;
					wi->triangleFlag= False;
					wi->error_region= False;
					wi->vectorFlag= False;
					break;
				case 1:
				case EREGION_FLAG:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= False;
					wi->error_region= False;
					wi->vectorFlag= False;
					break;
				case 2:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= True;
					wi->error_region= False;
					wi->vectorFlag= False;
					break;
				case 3:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= False;
					wi->error_region= True;
					wi->vectorFlag= False;
					break;
				case 4:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= False;
					wi->error_region= False;
					wi->vectorFlag= True;
					break;
			}
			if( wi->vectorFlag && this_set->vectorType>= 2 && this_set->lcol>= 0 ){
				ncoords= 4;
				use_lcol= True;
			}
			else{
				ncoords= 3;
				use_lcol= False;
			}
			if( debugFlag && debugLevel ){
				fprintf( StdErr, "DrawData(%d) transforming set #%d (%d points)\n", __LINE__, set_nr, this_set->numPoints );
			}
			if( this_set->numPoints== 1){
			  int k;
			  /* fake two points	*/
			  /*
				XVAL( this_set, 1)= XVAL( this_set, 0);
				YVAL( this_set, 1)= YVAL( this_set, 0);
				ERROR( this_set, 1)= ERROR( this_set, 0);
			   */
				for( k= 0; k< this_set->ncols; k++ ){
					this_set->columns[k][1]= this_set->columns[k][0];
				}
				this_set->ldyvec[1]= this_set->ldyvec[0];
				this_set->hdyvec[1]= this_set->hdyvec[0];
				if( (this_set->numPoints= 2)> maxitems ){
					maxitems= this_set->numPoints;
					realloc_Xsegments();
				}
				this_set->show_llines= 0;
			}

			if( !wi->raw_display && (this_set->init_pass || wi->init_pass) ){
				if( this_set->numPoints>= 500 ){
				  char buf[256];
				  double perc = PointsDone/ PointTotal* 100;
					if( ((int) perc) % 10 == 0 ){
						sprintf( buf, "%sInitialising %sset #%d (%d points) >=%g%%",
							(active)? "Re-" : "",
							(wi->init_pass)? "window: " : "", set_nr, this_set->numPoints,
							perc
						);
						XStoreName( disp, wi->window, buf );
						if( !RemoteConnection ){
							XSync( disp, False );
						}
					}
				}
				PointsDone+= this_set->numPoints;
			}

			SS_Add_Data_( SS_Points, 1, this_set->numPoints, 1.0 );

			  /* 950619
			   \ Do all runtime transformations (except *TRANSFORM_[XY]*) first. Saves doing them twice for all points!
			   */
			if( !this_set->raw_display ){
				*ascanf_setNumber= set_nr;
				*ascanf_numPoints= this_set->numPoints;
			}

			SS_Reset_(SS_er);
			SS_Reset_(SS_er_r);
			SAS_er.Gonio_Base= radix;
			SAS_er.Gonio_Offset= radix_offset;
			SAS_Reset_(SAS_er);
			SAS_er_r.Gonio_Base= radix;
			SAS_er_r.Gonio_Offset= radix_offset;
			SAS_Reset_(SAS_er_r);

			if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
			  char watch[2]= { XC_heart+1, '\0' };
				xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
			}

			this_set->first_error= -1;
			this_set->last_error= -1;

			curve_len= 0;
			error_len= 0;
			wi->curve_len[set_nr][0]= 0;
			wi->error_len[set_nr][0]= 0;
#ifdef TR_CURVE_LEN
			tr_curve_len= 0;
			wi->tr_curve_len[set_nr][0]= 0;
#endif
			wi->SAS_slope.Gonio_Base= wi->radix;
			wi->SAS_slope.Gonio_Offset= wi->radix_offset;
			SAS_Reset_( wi->SAS_slope );
			wi->SAS_scrslope.Gonio_Base= wi->radix;
			wi->SAS_scrslope.Gonio_Offset= wi->radix_offset;
			SAS_Reset_( wi->SAS_scrslope );

			if( wi->init_pass || this_set->init_pass ){
				for( pnt_nr= 1; pnt_nr< this_set->numPoints; pnt_nr++ ){
					wi->curve_len[set_nr][pnt_nr]= 0;
					wi->error_len[set_nr][pnt_nr]= 0;
#ifdef TR_CURVE_LEN
					wi->tr_curve_len[set_nr][pnt_nr]= 0;
#endif
				}
			}
			SS_Reset_( set_X);
			SS_Reset_( set_Y);
			SS_Reset_( set_E);
			set_O.Gonio_Base= wi->radix;
			set_O.Gonio_Offset= wi->radix_offset;
			SAS_Reset_( set_O);
			SS_Reset_( set_tr_X);
			SS_Reset_( set_tr_Y);
			SS_Reset_( set_tr_E);
			set_tr_O.Gonio_Base= wi->radix;
			set_tr_O.Gonio_Offset= wi->radix_offset;
			SAS_Reset_( set_tr_O);
			initRYE= True;

			this_set->processing_wi= wi;

			for( pnt_nr = 0;  pnt_nr < this_set->numPoints && !this_set->skipOnce;  pnt_nr++ ){
/* 			  int discard_change= 0;	*/
				  /* 981109: added check for curvelen_with_discarded - don't know exactly
				   \ what side-effects that will have
				   */
				if( DiscardedPoint( wi, this_set,pnt_nr)<= 0 || (*curvelen_with_discarded== 2) || DiscardedShadows ){
					CHECK_EVENT();
					if( wi->halt ){
						wi->redraw= 0;
						  /* wi->halt must be unset before calling this function again. This
						   \ makes it possible to detain redrawing for a while...
						   */
						psSeg_disconnect_jumps= psSdj;
						wi->event_level--;
						goto DrawData_return;
					}

					  /* retrieve data of current point. This should not be done from one
					   \ of the ...vec fields, since that would result in iterative
					   \ transformations/processing!
					   */
					this_set->data[0][0]= sx1 = XVAL( this_set, pnt_nr);
					this_set->data[0][1]= sy1 = YVAL( this_set, pnt_nr);
					this_set->data[0][2]= ERROR( this_set, pnt_nr);
					if( use_lcol ){
						this_set->data[0][3]= VVAL( this_set, pnt_nr );
					}

					  /* Here, we can add a check to see if the current point falls within some user-defined
					   \ window - X or Y or both. If not, skip the point because the scaling will be such that
					   \ only this window will be visible inside the plotting region. If necessary, we can decide
					   \ to include the point (e.g. whtn fit_once) as long as the scaling is not influenced by
					   \ such a decision.
					   \ 20010729: investigate whether we can use the visible[] array for storing if a point
					   \ is within the datawin or not.
					   */
					if( wi->datawin.apply && *DataWin_before_Processing ){
					  int pok= True;
						if( !NaN(wi->datawin.llX) && sx1< wi->datawin.llX ){
							pok= False;
						}
						else if( !NaN(wi->datawin.urX) && sx1> wi->datawin.urX ){
							pok= False;
						}
						else if( !NaN(wi->datawin.llY) && sy1< wi->datawin.llY ){
							pok= False;
						}
						else if( !NaN(wi->datawin.urY) && sy1> wi->datawin.urY ){
							pok= False;
						}
						if( !pok ){
							goto clen_next_point;
						}
					}

					if( wi->vectorFlag ){
					  /* determine the 2 points of the vector (vector) flag that will embellish this
					   \ point. The orientation is given by what is otherwise interpreted as the error
					   \ The length is taken from the set-specific vectorLength.
					   */
						  /* vectors are determined *after* (or by) DrawData_process()	*/
						if( this_set->use_error && !NaNorInf( this_set->data[0][2] ) && !DiscardedPoint( wi, this_set, pnt_nr) ){
							SAS_Add_Data_( SAS_er_r, 1, this_set->data[0][2], 1.0, (int) *SAS_converts_angle);
						}
					}
					else{
						if( etype!= MSIZE_FLAG ){
							  /* x errorbars are not (yet) supported	*/
							sx3 = sx1;
							sy3 = sy1- this_set->data[0][2];
							sx4 = sx1;
							sy4 = sy1+ this_set->data[0][2];
						}
						if( !DiscardedPoint( wi, this_set, pnt_nr) ){
							SS_Add_Data_( SS_er_r, 1, this_set->data[0][2], 1.0);
						}
					}

					if( pnt_nr< this_set->numPoints- 1 ){
						this_set->data[1][0]= sx2 = XVAL( this_set, pnt_nr+1);
						this_set->data[1][1]= sy2 = YVAL( this_set, pnt_nr+1);
						this_set->data[1][2]= ERROR( this_set, pnt_nr+1);
						if( use_lcol ){
							this_set->data[1][3]= VVAL( this_set, pnt_nr+1 );
						}
						if( wi->vectorFlag ){
						}
						else{
							sx5 = sx2;
							sy5 = sy2- this_set->data[1][2];
							sx6 = sx2;
							sy6 = sy2+ this_set->data[1][2];
						}

						  /* Even though the point is "treated" when discarded by a
						   \ runtime transformation, we don't include it in the computation
						   \ of the curve length. The sole difference between "mouse-deleted"
						   \ and "transformation-deleted" points is to be the fact that the latter
						   \ are to be undeletable by undoing the transformation (e.g. raw display).
						   \ 981109 : Inclusion of a segment in the curve length depends on a settable
						   \ variable *curvelen_with_discarded ("$curve_len-with-discarded"). Might be
						   \ that we'll actually also want to treat the averaging in this manner!
						   */
						{ double cl= sqrt( (sx2 - sx1)*(sx2 - sx1) + (sy2 - sy1)*(sy2 - sy1) ), el;
						  int disc;
							el= fabs( ERROR( this_set, pnt_nr+1) - ERROR( this_set, pnt_nr) );
							if( wi->vectorFlag ){
							  /* the length of each orientation "segment" is the absolute difference in angle,
							   \ corrected for jumps of <radix> (360) degrees due to the circularity of orientation.
							   \ Note: summing (from 0) over a similarly corrected signed difference would
							   \ yield the orientation without the jumps, and decreasing, say, to -720degrees for
							   \ a 2x rightward turn.
							   */
								if( el>= wi->radix/ 2.0 ){
									el= fabs( conv_angle_( el, wi->radix ) );
								}
							}
							else{
							  /* error_len based on error differences .. probably not very useful!	*/
							}
							if( !(disc= DiscardedPoint( wi, this_set, pnt_nr)) ){
								SS_Add_Data_( set_X, 1, sx1, 1.0 );
								SS_Add_Data_( set_Y, 1, sy1, 1.0 );
								if( wi->vectorFlag ){
									SAS_Add_Data_( set_O, 1, ERROR(this_set, pnt_nr), 1.0, (int) *SAS_converts_angle );
								}
								else{
									SS_Add_Data_( set_E, 1, ERROR(this_set, pnt_nr), 1.0 );
								}
								if( !NaN(cl) ){
									curve_len+= cl;
								}
								wi->curve_len[set_nr][pnt_nr+1]= curve_len;
								if( !NaN(el) ){
									error_len+= el;
								}
								wi->error_len[set_nr][pnt_nr+1]= error_len;
							}
							else if( disc< 0 && *curvelen_with_discarded== 1 ){
								if( !NaN(cl) ){
									curve_len+= cl;
								}
								wi->curve_len[set_nr][pnt_nr+1]= curve_len;
								if( !NaN(el) ){
									error_len+= el;
								}
								wi->error_len[set_nr][pnt_nr+1]= error_len;
							}
							else if( disc> 0 && *curvelen_with_discarded== 2 ){
								if( !NaN(cl) ){
									curve_len+= cl;
								}
								wi->curve_len[set_nr][pnt_nr+1]= curve_len;
								if( !NaN(el) ){
									error_len+= el;
								}
								wi->error_len[set_nr][pnt_nr+1]= error_len;
							}
						}
					}
					else{
						this_set->data[1][0]= sx2 = sx1;
						this_set->data[1][1]= sy2 = sy1;
						this_set->data[1][2]= this_set->data[0][2];
						this_set->data[1][3]= this_set->data[0][3];
						if( wi->vectorFlag ){
						}
						else{
							sx5 = sx2;
							sy5 = sy2- this_set->data[1][2];
							sx6 = sx2;
							sy6 = sy2+ this_set->data[1][2];
						}
					}

					AddPoint_discard= False;
					ascanf_SplitHere= 0;
					{  int do_vectors= 0;
						if( !this_set->raw_display ){
							  /* DrawData_process() doesn't *depend* on the sx and sy values passed. It
							   \ may only change them when it changes this_set->data
							   */
							if( !DrawData_process( wi, this_set, this_set->data, pnt_nr, 1, ncoords,
								&sx1, &sy1, &sx2, &sy2, &sx3, &sy3, &sx4, &sy4, &sx5, &sy5, &sx6, &sy6
								)
							){
								do_vectors= 1;
							}
						}
						else{
							do_vectors= 1;
						}
						if( wi->datawin.apply && !*DataWin_before_Processing ){
						  int pok= True;
							if( !NaN(wi->datawin.llX) && sx1< wi->datawin.llX ){
								pok= False;
							}
							else if( !NaN(wi->datawin.urX) && sx1> wi->datawin.urX ){
								pok= False;
							}
							else if( !NaN(wi->datawin.llY) && sy1< wi->datawin.llY ){
								pok= False;
							}
							else if( !NaN(wi->datawin.urY) && sy1> wi->datawin.urY ){
								pok= False;
							}
							if( !pok ){
								goto clen_next_point;
							}
						}
						if( this_set->numPoints< 0 ){
							  /* This set was deleted!	*/
							goto transform_next_set;
						}
						if( !wi->use_average_error && do_vectors ){
						  double X= XVAL(this_set, pnt_nr), Y= YVAL(this_set, pnt_nr);
							if( wi->vectorFlag ){
							  /* determine the 2 points of the vector (vector) flag that will embellish this
							   \ point. The orientation is given by what is otherwise interpreted as the error
							   \ The length is taken from the set-specific vectorLength.
							   */
								make_vector( wi, this_set, X, Y, &sx3, &sy3, &sx4, &sy4, this_set->data[0][2], this_set->data[0][3] );
								if( pnt_nr< this_set->numPoints-1 ){
								  double X1= XVAL(this_set, pnt_nr+1), Y1= YVAL(this_set, pnt_nr+1);
									make_vector( wi, this_set, X1, Y1, &sx5, &sy5, &sx6, &sy6,
										this_set->data[1][2], this_set->data[1][3] );
								}
							}
							else if( etype== MSIZE_FLAG ){
								make_sized_marker( wi, this_set, X, Y, &sx3, &sy3, &sx4, &sy4, this_set->data[0][2] );
								if( pnt_nr< this_set->numPoints-1 ){
								  double X1= XVAL(this_set, pnt_nr+1), Y1= YVAL(this_set, pnt_nr+1);
									make_sized_marker( wi, this_set, X1, Y1, &sx5, &sy5, &sx6, &sy6, this_set->data[1][2] );
								}
							}
						}
					}
					if( DiscardPoint( wi, this_set, pnt_nr, AddPoint_discard )> 0 ){
						set_init_pass= True;
					}
					  /* 20000505: Don't make NaNs when DiscardedShadows!	*/
					if( AddPoint_discard> 0 && !DiscardedShadows ){
						set_NaN(sx1);
						set_NaN(sy1);
						set_NaN(this_set->errvec[pnt_nr]);
					}
#ifdef RUNSPLIT
					if( ascanf_SplitHere ){
					  /* drawing a window => must be a runtime call. We
					   \ can not directly cut sets, just (un)set a bit
					   \ that will cause a *SPLIT* command to be inserted in XGraph output
					   */
						if( !this_set->splithere ){
							this_set->splithere= (signed char*) calloc( this_set->allocSize, sizeof(signed char));
							this_set->mem_alloced+= this_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
							mem_alloced+= this_set->allocSize* ( sizeof(signed char)+ 1* sizeof(int));
						}
						if( this_set->splithere ){
							  /* runtime, transformation-specified setting. Don't mess with settings
							   \ requested in another fashion
							   */
							if( this_set->splithere[pnt_nr]<= 0 ){
								this_set->splithere[pnt_nr]= -1;
							}
						}
					}
					else if( this_set->splithere ){
						if( this_set->splithere[pnt_nr]< 0 ){
							this_set->splithere[pnt_nr]= 0;
						}
					}
#endif
					if( this_set->DYscale && !(wi->vectorFlag || etype== MSIZE_FLAG) ){
						sy3= sy1- this_set->DYscale* this_set->data[0][2];
						sy4= sy1+ this_set->DYscale* this_set->data[0][2];
					}

					  /* Now store the transformed values in the location where they are expected:
					   \ that is, in the ...vec fields.
					   */
					this_set->xvec[pnt_nr]= sx1;
					this_set->yvec[pnt_nr]= sy1;
					this_set->ldxvec[pnt_nr]= sx3;
					this_set->ldyvec[pnt_nr]= sy3;
					this_set->hdxvec[pnt_nr]= sx4;
					this_set->hdyvec[pnt_nr]= sy4;
					  /* 980511: shouldn't the next statement be here...?!	*/
					this_set->errvec[pnt_nr]= this_set->data[0][2];
					this_set->lvec[pnt_nr]= this_set->data[0][3];
					if( AddPoint_discard<= 0 && this_set->use_error && !DiscardedPoint( wi, this_set, pnt_nr) ){
						if( etype== 4 ){
							if( !NaNorInf(this_set->data[0][2]) ){
								SAS_Add_Data_( SAS_er, 1, this_set->data[0][2], 1.0, (int) *SAS_converts_angle);
								  /* Calculation of the average error.	*/
								if( !this_set->floating ){
									SAS_Add_Data_( wi->SAS_O, 1, this_set->data[0][2], 1.0, (int) *SAS_converts_angle);
								}
							}
						}
						else{
							SS_Add_Data_( SS_er, 1, this_set->data[0][2], 1.0);
							  /* Calculation of the average error.	*/
						}
						if( !this_set->floating && etype!= INTENSE_FLAG && etype!= 4 ){
							SS_Add_Data_( wi->SS_E, 1, this_set->data[0][2], 1.0);
						}
						  /* 20040129: update SS_I even if we don't want the IntensityLegend! */
						if( etype== INTENSE_FLAG /* wi->IntensityLegend.legend_needed */ ){
							SS_Add_Data_( wi->SS_I, 1, this_set->data[0][2], 1.0);
						}

						if( this_set->data[0][2] || wi->vectorFlag ){
							if( !NaN( this_set->data[0][2] ) ){
								if( this_set->first_error== -1 ){
									this_set->first_error= pnt_nr;
								}
								this_set->last_error= pnt_nr;
							}
						}
					}
				}
clen_next_point:;
			}

			if( this_set->skipOnce ){
				goto transform_next_set;
			}

			wi->curve_len[set_nr][this_set->numPoints-1]= curve_len;
			wi->curve_len[set_nr][this_set->numPoints]= curve_len;
			wi->error_len[set_nr][this_set->numPoints-1]= error_len;
			wi->error_len[set_nr][this_set->numPoints]= error_len;
			SS_Copy( &wi->set_X[set_nr], &set_X);
			SS_Copy( &wi->set_Y[set_nr], &set_Y);
			SS_Copy( &wi->set_E[set_nr], &set_E);
			SAS_Copy( &wi->set_O[set_nr], &set_O);

			if( set_init_pass ){
				if( !this_set->init_pass ){
					PointTotal+= this_set->numPoints;
				}
				this_set->init_pass= True;
				reinit_pass= True;
			}
			if( this_set->init_pass || wi->init_pass ){
				this_set->init_pass= set_init_pass;
				if( !draw_set( wi, set_nr ) ){
					goto transform_next_set;
				}
			}

			if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
			  char watch[2]= { XC_target+1, '\0' };
				xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
			}

			if( wi->vectorFlag ){
				this_set->av_error= SAS_Mean_(SAS_er);
				this_set->av_error_r= SAS_Mean_(SAS_er_r);
			}
			else{
				this_set->av_error= SS_Mean_(SS_er);
				this_set->av_error_r= SS_Mean_(SS_er_r);
			}
			  /* Average error is calculated: do the TRANSFORMATION	*/
			for( pnt_nr = 0;  pnt_nr < this_set->numPoints;  pnt_nr++ ){
				if( (!DiscardedPoint( wi, this_set,pnt_nr) || DiscardedShadows) &&
					(this_set->plot_interval<= 0 || (pnt_nr % this_set->plot_interval)==0)
				){
					  /* attention: there is a duplicate definition of adorn below!	*/
					adorn= (pnt_nr== 0 || pnt_nr== this_set->numPoints-1 ||
						this_set->adorn_interval<= 0 || (pnt_nr % this_set->adorn_interval)== 0 ||
						(this_set->error_point< -1 && (pnt_nr== this_set->first_error || pnt_nr== this_set->last_error))
					) && !DiscardedPoint( wi, this_set, pnt_nr);
					CHECK_EVENT();
					if( wi->halt ){
						wi->redraw= 0;
						  /* wi->halt must be unset before calling this function again. This
						   \ makes it possible to detain redrawing for a while...
						   */
						psSeg_disconnect_jumps= psSdj;
						wi->event_level--;
						goto DrawData_return;
					}
					sx1= this_set->xvec[pnt_nr];
					sx3= this_set->ldxvec[pnt_nr];
					sx4= this_set->hdxvec[pnt_nr];
					sy1= this_set->yvec[pnt_nr];
					if( wi->datawin.apply && *DataWin_before_Processing ){
					  int pok= True;
						if( !NaN(wi->datawin.llX) && sx1< wi->datawin.llX ){
							pok= False;
						}
						else if( !NaN(wi->datawin.urX) && sx1> wi->datawin.urX ){
							pok= False;
						}
						else if( !NaN(wi->datawin.llY) && sy1< wi->datawin.llY ){
							pok= False;
						}
						else if( !NaN(wi->datawin.urY) && sy1> wi->datawin.urY ){
							pok= False;
						}
						if( !pok ){
							goto transform_next_point;
						}
					}
					if( adorn || wi->error_region ){
					  int error_point= (this_set->error_point< this_set->numPoints)? this_set->error_point : this_set->numPoints-1;
						if( error_point== -1 || pnt_nr== error_point || error_point== -5 ||
							((error_point== -4 || error_point== -2) && pnt_nr== this_set->first_error) ||
							((error_point== -4 || error_point== -3) && pnt_nr== this_set->last_error)
						){
							if( !wi->vectorFlag && etype!= MSIZE_FLAG ){
								if( wi->use_average_error ){
									if( this_set->DYscale ){
										sy3= sy1- this_set->DYscale* this_set->av_error;
										sy4= sy1+ this_set->DYscale* this_set->av_error;
									}
									else{
										sy3= sy1- this_set->av_error;
										sy4= sy1+ this_set->av_error;
									}
								}
								else{
									sy3= this_set->ldyvec[pnt_nr];
									sy4= this_set->hdyvec[pnt_nr];
								}
							}
							else{
								if( wi->use_average_error && this_set->use_error ){
								  /* determine the 2 points of the vector (vector) flag that will embellish this
								   \ point. The orientation is given by what is otherwise interpreted as the error
								   \ The length is taken from the set-specific vectorLength.
								   */
									if( wi->vectorFlag ){
										make_vector( wi, this_set, sx1, sy1, &sx3, &sy3, &sx4, &sy4, this_set->av_error, this_set->data[0][3] );
									}
									else{
										make_sized_marker( wi, this_set, sx1, sy1, &sx3, &sy3, &sx4, &sy4, this_set->av_error );
									}
								}
								else{
								  /* These should already be determined:	*/
									sy3= this_set->ldyvec[pnt_nr];
									sy4= this_set->hdyvec[pnt_nr];
								}
							}
						}
						else{
						  /* Such a pity... this point will not show its error!	*/
							sx3= sx4= sx1;
							sy3= sy1;
							sy4= sy1;
						}
					}
					else{
						sx3= sx4= sx1;
						sy3= sy4= sy1;
					}
					do_TRANSFORM( wi, pnt_nr, 3, 3, &sx1, &sx3, &sx4, &sy1, &sy3, &sy4, 0, False );
					  /* Now store the transformed values in the location where they are expected:
					   \ that is, in the ...vec fields.
					   */
					this_set->xvec[pnt_nr]= sx1;
					this_set->yvec[pnt_nr]= sy1;
					this_set->ldxvec[pnt_nr]= sx3;
					this_set->ldyvec[pnt_nr]= sy3;
					this_set->hdxvec[pnt_nr]= sx4;
					this_set->hdyvec[pnt_nr]= sy4;
					if( !wi->vectorFlag && !etype== INTENSE_FLAG && etype!= MSIZE_FLAG ){
					  /* errvec is the _length_ of the transformed errorbar. Top and bottom
					   \ of this bar are most likely no longer at the right position, but
					   \ rather centered around the y-value.
					   */
						this_set->errvec[pnt_nr]= /* fabs */( sy4- sy3) / 2.0;
					}
					if( wi->datawin.apply && !*DataWin_before_Processing ){
					  int pok= True;
						if( !NaN(wi->datawin.llX) && sx1< wi->datawin.llX ){
							pok= False;
						}
						else if( !NaN(wi->datawin.urX) && sx1> wi->datawin.urX ){
							pok= False;
						}
						else if( !NaN(wi->datawin.llY) && sy1< wi->datawin.llY ){
							pok= False;
						}
						else if( !NaN(wi->datawin.urY) && sy1> wi->datawin.urY ){
							pok= False;
						}
						if( !pok ){
							goto transform_next_point;
						}
					}
					if( !DiscardedPoint( wi, this_set, pnt_nr ) ){
						if( !this_set->floating ){
							CollectPointStats( wi, this_set, pnt_nr, sx1, sy1, sx3, sy3, sx4, sy4 );
						}
						if( initRYE ){
							this_set->rawY.min= this_set->rawY.max= YVAL(this_set, pnt_nr);
							this_set->rawE.min= this_set->rawE.max= EVAL(this_set, pnt_nr);
							this_set->ripeY.min= this_set->ripeY.max= this_set->yvec[pnt_nr];
							this_set->ripeE.min= this_set->ripeE.max= this_set->errvec[pnt_nr];
							initRYE= False;
						}
						else{
							this_set->rawY.min= MIN( this_set->rawY.min, YVAL(this_set, pnt_nr));
							this_set->rawY.max= MAX( this_set->rawY.max, YVAL(this_set, pnt_nr));
							this_set->rawE.min= MIN( this_set->rawE.min, EVAL(this_set, pnt_nr));
							this_set->rawE.max= MAX( this_set->rawE.max, EVAL(this_set, pnt_nr));

							this_set->ripeY.min= MIN( this_set->ripeY.min, this_set->yvec[pnt_nr]);
							this_set->ripeY.max= MAX( this_set->ripeY.max, this_set->yvec[pnt_nr]);
							this_set->ripeE.min= MIN( this_set->ripeE.min, this_set->errvec[pnt_nr]);
							this_set->ripeE.max= MAX( this_set->ripeE.max, this_set->errvec[pnt_nr]);
						}
					}
				}
transform_next_point:;
			}
			if( this_set->barFlag> 0 ){
				if( (barBase_set || this_set->barBase_set) ){
					sx1 = 0;
					sy3= sy2= sy1 = this_set->barBase;
					do_TRANSFORM( wi, 0, 1, 1, &sx1, NULL, NULL, &sy1, &sy2, &sy3, 0, False );
					  /* We'd probably like to see the bars' base in view too..!	*/
					if( !this_set->floating ){
						SS_Add_Data_( wi->SS_Y, 1, sy1, 1.0);
					}
					this_set->barBaseY= sy1;
				}
				have_bars+= 1;
			}
			this_set->last_processed_wi= wi;
			this_set->processing_wi= pw;
		}
transform_next_set:;
	}
	wi->init_pass= False;
	wi->processing_set= False;
	if( !active ){
		if( reinit_pass ){
			if( debugFlag ){
				fprintf( StdErr, "DrawData(): once more since some sets still have init_pass flag set\n" );
			}
			active= 1;
			  /* We redo the DRAW_BEFORE if requested (there can be necessary resets in that statement),
			   \ but not the DRAW_AFTER, since we haven't finished drawing.
			   */
			if( *ReDo_DRAW_BEFORE ){
				Draw_Process( wi, 1 );
			}
			DrawData( wi, bounds_only );
			active= 0;
		}
	}
draw_points:;

	if( wi->delete_it== -1 ){
		goto DrawData_return;
	}

	  /* The statistics of the pen-co-ordinates can now be determined.	*/
	if( wi->use_transformed || wi->fitting ){
		if( wi->pen_list && !wi->no_pens ){
		  XGPen *Pen= wi->pen_list;
			while( Pen ){
				if( !Pen->floating && (Pen->drawn || !PENSKIP(wi,Pen)) ){
				  XGPenPosition *pos= Pen->position;
					for( pnt_nr= 0; pnt_nr< Pen->positions && pnt_nr< Pen->current_pos; pnt_nr++, pos++ ){
						CollectPenPosStats( wi, pos );
					}
					Pen->drawn= 0;
				}
				Pen= Pen->next;
			}
		}
	}

	if( wi->data_silent_process && !wi->osilent_val ){
		wi->osilent_val= wi->dev_info.xg_silent( wi->dev_info.user_state, wi->silenced || wi->osilent_val );
	}

	  /* 20000319: the following block was before the draw_points label,
	   \ causing doubled redraw times in QuickMode when lots of points.
	   */
	if( bounds_only || (wi->fitting && !have_bars) ){
		psSeg_disconnect_jumps= psSdj;
		wi->event_level--;
		goto DrawData_return;
	}

	if( !wi->animating ){

		xtb_bt_set( wi->ssht_frame.win, X_silenced(wi), NULL );

		TitleMessage( wi, NULL );
		if( X_silenced( wi ) ){
			TitleMessage( wi, "Silently drawing..." );
		}
		else{
			TitleMessage( wi, "Drawing..." );
		}
	}

	wi->osync_val= Synchro_State;
	if( wi->data_sync_draw ){
		if( !Synchro_State ){
			X_Synchro(wi);
		}
	}
	else if( Synchro_State ){
		X_Synchro(wi);
	}

	if( wi->pen_list && !wi->no_pens && !wi->fitting ){
	  XGPen *Pen= wi->pen_list;
		while( Pen ){
			if( !Pen->overwrite_pen && !PENSKIP(wi,Pen) && Pen->before_set<0 && Pen->after_set< 0 ){
				DrawPen( wi, Pen );
			}
			Pen= Pen->next;
		}
	}

	wi->use_errors= 0;
	wi->numDrawn= 0;
    for( cnr = 0;  cnr < setNumber;  cnr++){
	  int etype;
		set_nr= DO(cnr);
		etype= wi->error_type[set_nr];
		this_set= &AllSets[set_nr];
		if( this_set->numPoints<= 0 ){
			this_set->draw_set= 0;
			wi->draw_set[set_nr]= 0;
		}

		if( wi->pen_list && !wi->no_pens && !wi->fitting ){
		  XGPen *Pen= wi->pen_list;
			while( Pen ){
				if( !PENSKIP(wi,Pen) && Pen->before_set== set_nr ){
					DrawPen( wi, Pen );
				}
				Pen= Pen->next;
			}
		}

		if( !draw_set(wi, set_nr) || this_set->skipOnce ){
			if( debugFlag ){
				fprintf( StdErr, "DrawData(%d) skipping set #%d (%d points)\n", __LINE__, set_nr, this_set->numPoints );
			}
		}
		else{

			this_set->current_bW_set= False;
			this_set->current_ebW_set= False;

			  /* 20000405: construct a pixelvalue that (hopefully) not equal
			   \ to the set's colour, nor to the zeroPix, nor to fore and background:
			   */
			dshadowPixel= zeroPixel;
			if( this_set->pixvalue< 0 ){
				if( dshadowPixel!= this_set->pixelValue ){
					dshadowPixel= dshadowPixel ^ this_set->pixelValue;
				}
			}
			else{
				if( dshadowPixel!= AllAttrs[this_set->pixvalue].pixelValue ){
					dshadowPixel= dshadowPixel ^ AllAttrs[this_set->pixvalue].pixelValue;
				}
			}
			if( dshadowPixel!= bgPixel ){
				dshadowPixel= dshadowPixel ^ bgPixel;
			}
			if( dshadowPixel!= normPixel ){
				dshadowPixel= dshadowPixel ^ normPixel;
			}

			adorned= Adorned_Points( this_set );
			if( etype== -1 ){
				if( error_type== -1 ){
					error_type= !no_errors;
				}
				etype= wi->error_type[set_nr]= error_type;
			}
			switch( etype ){
				case 0:
				case INTENSE_FLAG:
				case MSIZE_FLAG:
					w_use_errors= False;
					wi->triangleFlag= False;
					wi->error_region= False;
					wi->vectorFlag= False;
					break;
				case 1:
				case EREGION_FLAG:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= False;
					wi->error_region= False;
					wi->vectorFlag= False;
					break;
				case 2:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= True;
					wi->error_region= False;
					wi->vectorFlag= False;
					break;
				case 3:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= False;
					wi->error_region= True;
					wi->vectorFlag= False;
					break;
				case 4:
					wi->use_errors+= (w_use_errors= True);
					wi->triangleFlag= False;
					wi->error_region= False;
					wi->vectorFlag= True;
					break;
			}

			if( debugFlag && debugLevel ){
				fprintf( StdErr, "DrawData(%d) drawing set #%d (%d points)\n", __LINE__, set_nr, this_set->numPoints );
			}

			if( !sn_set && (set_sn_nr || (data_sn_number>= 0 && data_sn_number< setNumber)) && !dialog_redraw ){
			  /* If not already set here, and if it currently is set to a valid set number,
			   \ set data_sn_number to the number of the 1st set drawn. This will ensure that
			   \ the Settings Dialog displays that set. Don't change data_sn_number when we're
			   \ called through the settings dialog.
			   */
				data_sn_number= set_nr;
				sn_set= True;
				set_sn_nr= False;
			}

			if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
			  char watch[2]= { XC_heart+ 1, '\0' };
				xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
			}

			  /* Width of the highlighting background:
			   \ function of the set's lineWidth, but not increasing linearly.
			   \ Highlighting of error-regions is slightly narrower.
			   */
/* 			hl_width= HL_WIDTH(AllSets[set_nr].lineWidth);	*/
			hl_width= HL_WIDTH( LINEWIDTH(wi,set_nr) );

			if( PS_STATE(wi)->Printing== PS_PRINTING){
			  char psSet[512];
				sprintf( psSet, "Set#%d (%d): %s", set_nr, this_set->numPoints, PS_STATE(wi)->JobName);
				{ char *msg= concat( psSet, (wi->no_title && this_set->titleText)? this_set->titleText:"",
						(wi->no_legend && this_set->setName)? this_set->setName:"",
						(this_set->process.description)? this_set->process.description:"", NULL );
/* 					if( wi->no_title && this_set->titleText ){	*/
/* 						strncpy( ps_comment, this_set->titleText, sizeof(ps_comment)/sizeof(char)-1 );	*/
/* 					}	*/
/* 					psMessage( PS_STATE(wi), psSet);	*/
/* 					if( wi->no_legend && this_set->setName ){	*/
/* 						strncpy( ps_comment, this_set->setName, sizeof(ps_comment)/sizeof(char)-1 );	*/
/* 					}	*/
/* 					psMessage( PS_STATE(wi), psSet);	*/
/* 					if( this_set->process.description ){	*/
/* 						strncpy( ps_comment, this_set->process.description, sizeof(ps_comment)/sizeof(char)-1 );	*/
/* 					}	*/
					psMessage( PS_STATE(wi), msg);
					xfree( msg );
				}
			}

			noLines= this_set->noLines;
			barFlag= (this_set->barFlag> 0);
			markFlag= this_set->markFlag;
			pixelMarks= this_set->pixelMarks;
			LStyle= L_VAR;

#	ifndef DEBUG
			  /* I don't think this is actually necessary. Might come in handy when debugging.
			   \ 20000405: It is necessary when drawing shadows of discarded points, just to prevent
			   \ other "ghosts" from being drawn. For the time being, set Xsegs to all (-1,-1).
			   \ This of course depends on the fact that x1,y1,.. are shorts and not unsigned shorts,
			   \ and on the exact behaviour of the different drawing routines. Crucial: a segment
			   \ consists of 2 points (x,y) and becomes a single dot when those two points are the
			   \ same. As long as the drawing code below (not the individual lowlevel methods!) ensure
			   \ that not-to-be-drawn datapoints are stored as such dots, off the visible canvas, setting
			   \ to -1 will probably prevent the ghosts described above. If not, for whatever reason,
			   \ another segment counter, specific to Xsegs (say, X_seg_nr) will be necessary.
			   \ 20000407: I the more elegant solution, using X_seg_nr seems to work. It was not trivial
			   \ to handle all possible deletes. If we consider X_set_nr and X_seg_nr as ranges (from 0
			   \ to this_set->numPoints maximally), then X_set_nr (XsegsE) always represents the full range of
			   \ (somehow - even as shadows) visible points, whereas X_seg_nr (Xsegs) represents the
			   \ points that may define the polygon used to draw the set's line. The big problem was
			   \ getting all the overwrite marks to be drawn, and in the right place. See the comments
			   \ at that place (search for 6x and then _diff).
			   \ One can go back to the "invisible canvas" solution above by making X_seg_nr increment
			   \ unconditionally, like X_set_nr (search for X_seg_nr++; one instance).
			   \ Another option would be to use a custom replacement for XSegment (and XDrawSegments() [called
			   \ in seg_X()]; this may be tricky since all of its side-effects and "guarantees"), by
			   \ including a "no_display" field.
			  */
			if( DiscardedShadows )
#	endif
			{
				memset( Xsegs, 0, XsegsSize );
				memset( XXsegs, 0, XXsegsSize );
				memset( LYsegs, 0, YsegsSize );
				memset( HYsegs, 0, YsegsSize );
				memset( XsegsE, 0, XsegsESize );
			}
			pX_set_nr= -1;
			X_set_nr = 0;
			X_seg_nr = 0;
			LY_set_nr = 0;
			HY_set_nr = 0;

			if( !this_set->raw_display ){
				*ascanf_setNumber= set_nr;
				*ascanf_numPoints= this_set->numPoints;
			}

			wi->numVisible[set_nr]= 0;

			if( this_set->s_bounds_set == -2 ){
			  /* This means that the clipping request is to be conserved across
			   \ a silent redraw.
			   */
				bounds_clip= True;
			}
			  /* Make sure the set's drawing screen-coordinates boundaries are updated
			   \ for this window.
			   */
			this_set->s_bounds_set= 0;
			this_set->s_bounds_wi= wi;

			sx1_1= this_set->xvec[0];
			pbarPixels= 0;
			epbarPixels= 0;

			if( wi->use_transformed ){
				curve_len= 0;
				error_len= 0;
				wi->curve_len[set_nr][0]= 0;
				wi->error_len[set_nr][0]= 0;
			}
#ifdef TR_CURVE_LEN
			tr_curve_len= 0;
#endif

			if( IntensityColourFunction.NColours> 1 ){
				if( IntensityColourFunction.range_set ){
					minIntense= IntensityColourFunction.range.min;
					maxIntense= IntensityColourFunction.range.max;
				}
				else{
					minIntense= wi->SS_I.min;
					maxIntense= wi->SS_I.max;
				}
				scaleIntense= (IntensityColourFunction.NColours- 1)/ (maxIntense- minIntense);
			}

/* 			ldxvec= this_set->ldxvec;	*/
/* 			ldyvec= this_set->ldyvec;	*/
/* 			hdxvec= this_set->hdxvec;	*/
/* 			hdyvec= this_set->hdyvec;	*/

/* 			if( this_set->pixvalue< 0 ){	*/
/* 			}	*/
/* 			else{	*/
/* 			}	*/

			  /* 20031201: new field use_first to indicate to overwrite_marks (etc) loops whether to use SegsNR(,)->x1,y1
			   \ instead of x2,y2. Should only be the case on the first datapoint, and each first following a 'gap'.
			   */
			SegsENR(XsegsE, X_set_nr)->use_first= True;
			for( pnt_nr = 0;  pnt_nr < this_set->numPoints-1;  pnt_nr++ ){
#ifdef TR_CURVE_LEN
			  double cl;
#endif
			  Boolean pok;

				sx1 = this_set->xvec[pnt_nr];
				sy1 = this_set->yvec[pnt_nr];
				sx2 = this_set->xvec[pnt_nr+1];
				sy2 = this_set->yvec[pnt_nr+1];
				if( (pok= !( NaN(sx1) || NaN(sx2) || NaN(sy1) || NaN(sy2) )) ){
#ifdef TR_CURVE_LEN
					if( Determine_tr_curve_len ){
						cl= sqrt( (sx2 - sx1)*(sx2 - sx1) + (sy2 - sy1)*(sy2 - sy1) );
					}
					else{
						cl= 0;
					}
#endif
				}

				if( wi->use_transformed ){
					if( pnt_nr< this_set->numPoints- 1 ){
					  int disc;
					  double clr, x1, y1, x2, y2;
					  double el= fabs( ERROR( this_set, pnt_nr+1) - ERROR( this_set, pnt_nr) );
/* 					  double el= fabs( this_set->errvec[pnt_nr+1] - this_set->errvec[pnt_nr] );	*/

						  /* 20010614: *why* should curve_len and error_len represent transformed values
						   \ in Quick Mode?!
						   */
						x1= XVAL( this_set, pnt_nr ), y1= YVAL( this_set, pnt_nr );
						x2= XVAL( this_set, pnt_nr+1 ), y2= YVAL( this_set, pnt_nr+1 );
						clr= sqrt( (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) );
						if( wi->vectorFlag ){
							if( el>= wi->radix/ 2.0 ){
								el= fabs( conv_angle_( el, wi->radix ) );
							}
						}
						else{
						  /* error_len based on error differences .. probably not very useful!	*/
						}
						if( !(disc= DiscardedPoint( wi, this_set, pnt_nr)) ){
							if( !NaN(clr) ){
								curve_len+= clr;
							}
							wi->curve_len[set_nr][pnt_nr+1]= curve_len;
							if( !NaN(el) ){
								error_len+= el;
							}
							wi->error_len[set_nr][pnt_nr+1]= error_len;
						}
						else if( disc< 0 && *curvelen_with_discarded== 1 ){
							if( !NaN(clr) ){
								curve_len+= clr;
							}
							wi->curve_len[set_nr][pnt_nr+1]= curve_len;
							if( !NaN(el) ){
								error_len+= el;
							}
							wi->error_len[set_nr][pnt_nr+1]= error_len;
						}
						else if( disc> 0 && *curvelen_with_discarded== 2 ){
							if( !NaN(clr) ){
								curve_len+= clr;
							}
							wi->curve_len[set_nr][pnt_nr+1]= curve_len;
							if( !NaN(el) ){
								error_len+= el;
							}
							wi->error_len[set_nr][pnt_nr+1]= error_len;
						}
					}
				}

				if( !(discarded_point= DiscardedPoint( wi, this_set, pnt_nr)) ){
					if( pok ){
						SS_Add_Data_( set_tr_X, 1, sx1, 1.0 );
						SS_Add_Data_( set_tr_Y, 1, sy1, 1.0 );
						if( wi->vectorFlag ){
							SAS_Add_Data_( set_tr_O, 1, this_set->errvec[pnt_nr], 1.0, (int) *SAS_converts_angle );
						}
						else{
							SS_Add_Data_( set_tr_E, 1, this_set->errvec[pnt_nr], 1.0 );
						}
#ifdef TR_CURVE_LEN
						if( Determine_tr_curve_len ){
							tr_curve_len+= cl;
							wi->tr_curve_len[set_nr][pnt_nr+1]= tr_curve_len;
						}
#endif
						if( Determine_AvSlope ){
							switch( Determine_AvSlope ){
								case 1:
								default:
									SAS_Add_Data_( wi->SAS_slope, 1, ArcTan( wi, (sx2-sx1), (sy2-sy1)), 1, 1 );
									break;
								case 2:
									SAS_Add_Data_( wi->SAS_slope, 1,
										fabs( conv_angle_( ArcTan( wi, (sx2-sx1), (sy2-sy1)), wi->radix)), 1, 0 );
									break;
							}
						}
					}
				}
#ifdef TR_CURVE_LEN
				else if( pok && DiscardedPoint( NULL, this_set, pnt_nr)< 0 && *curvelen_with_discarded== 1 ){
					tr_curve_len+= cl;
					wi->tr_curve_len[set_nr][pnt_nr+1]= tr_curve_len;
				}
				else if( pok && DiscardedPoint( NULL, this_set, pnt_nr)> 0 && *curvelen_with_discarded== 2 ){
					tr_curve_len+= cl;
					wi->tr_curve_len[set_nr][pnt_nr+1]= tr_curve_len;
				}
#endif

				  /* if this point is to be skipped, "jump" to the next point	*/
				if( (discarded_point /* && !DiscardedShadows */) ||
					(this_set->plot_interval> 0 && (pnt_nr % this_set->plot_interval)!= 0)
				){
					  /* 20031201 */
					SegsENR(XsegsE, X_set_nr)->use_first= True;
					  /* 20041002: */
					if( !X_seg_nr ){
						SegsENR(XsegsE, pnt_nr+1)->use_first= True;
					}
					  /* 20041002: discarded points shown as shadows should not be skipped, but
					   \ they do have to alter the use_first information as 'regular' discarded point
					   \ (the original test caused that operation to be skipped, leading to potential
					   \ Xsegs addressing errors)
					   */
					if( !(discarded_point && DiscardedShadows) ){
						goto next_point;
					}
				}
				adorn= (pnt_nr== 0 || pnt_nr== this_set->numPoints-1 ||
					this_set->adorn_interval<= 0 || (pnt_nr % this_set->adorn_interval)== 0 ||
					(this_set->error_point< -1 && (pnt_nr== this_set->first_error || pnt_nr== this_set->last_error))
				) && !discarded_point;

				CHECK_EVENT();
				if( wi->halt ){
					wi->redraw= 0;
					  /* wi->halt must be unset before calling this function again. This
					   \ makes it possible to detain redrawing for a while...
					   */
					if( bounds_clip ){
						this_set->s_bounds_set= -1;
					}
					psSeg_disconnect_jumps= psSdj;
					wi->event_level--;
					RESET_WIN_STATE(wi);
					goto DrawData_return;
				}
				if( !printed && debugFlag ){
					fputs( "Drawing in new win_geo:\n", StdErr);
					LWG_printf( StdErr, "\t", &wi->win_geo);
					fflush( StdErr);
					printed= 1;
				}
				  /* Put segment in (sx1,sy1) (sx2,sy2) */
				this_set->data[0][0]= sx1 = this_set->xvec[pnt_nr];
				this_set->data[0][1]= sy1 = this_set->yvec[pnt_nr];
				this_set->data[0][2]= this_set->errvec[pnt_nr];
				this_set->data[0][3]= this_set->lvec[pnt_nr];
				if( NaN(sx1) || NaN(sy1) ){
				  /* current (first) point is a NaN - skip to the next point */
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "DrawData() point[%d,%d]=(%s,%s,%s): skipping\n",
							set_nr, pnt_nr,
							d2str(sx1,"%g", NULL), d2str( sy1, "%g", NULL),
							d2str(this_set->errvec[pnt_nr], "%g", NULL)
						);
					}
					  /* 20031201 */
					SegsENR(XsegsE, X_set_nr)->use_first= True;
					  /* 20041002: */
					if( !X_seg_nr ){
						SegsENR(XsegsE, pnt_nr+1)->use_first= True;
					}
					goto next_point;
				}

				if( discarded_point ){
					pnt_nr_1= pnt_nr;
				}
				else{
					pnt_nr_1= pnt_nr+ 1;
					while( pnt_nr_1< this_set->numPoints-1 &&
						(DiscardedPoint( wi, this_set,pnt_nr_1) || (this_set->plot_interval> 0 && (pnt_nr_1 % this_set->plot_interval)!= 0) )
					){
						pnt_nr_1+= 1;
					}
				}
				this_set->data[1][0]= sx2 = this_set->xvec[pnt_nr_1];
				this_set->data[1][1]= sy2 = this_set->yvec[pnt_nr_1];
				this_set->data[1][2]= this_set->errvec[pnt_nr_1];
				this_set->data[1][3]= this_set->lvec[pnt_nr_1];

				  /* 20001128: Here (as elsewhere), [lh]d[xy]vec was referenced directly through
				   \ the this_set pointer. Under Linux with gcc 2.95.2 on a PIII, this caused
				   \ problems when the 2 -fschedule-insns optimisations are turned on. These optims.
				   \ do improve speed (somewhat), so I wanted to keep them. The effect was that for
				   \ pnt_nr==0, sy4 would become NaN, *unless* the outcommented printf statement was
				   \ present. This would seem a bug to me, just like the problems with the aspect
				   \ calculation above. It does not occur (for the moment, at least...) when using
				   \ a local set of pointers that reference the corresponding this_set fields.
				   \ 20001129: I discovered another option, well hidden in the docs. Until now, I used
				   \ the -ffloat-store option to get true IEEE (754) conformance. It is also possible
				   \ to do that with the _FPU_SETCW to set the FPU to double (instead of extended)
				   \ precision. This also removed this particular phenomenon...
				   */
				sx3 = this_set->ldxvec[pnt_nr];
				sy3 = this_set->ldyvec[pnt_nr];
				sx4 = this_set->hdxvec[pnt_nr];
				sy4 = this_set->hdyvec[pnt_nr];
				ok= 1;

/* 				if( pnt_nr== 0 ){	*/
/* 					fprintf( StdErr, "ebar: (%g,%g)-(%g,%g) should be (%g,%g)-(%g,%g)\n",	*/
/* 						sx3, sy3, sx4, sy4, this_set->ldxvec[pnt_nr], this_set->ldyvec[pnt_nr],	*/
/* 						this_set->hdxvec[pnt_nr], this_set->hdyvec[pnt_nr]	*/
/* 					);	*/
/* 				}	*/

				sx5 = this_set->ldxvec[pnt_nr_1];
				sy5 = this_set->ldyvec[pnt_nr_1];
				sx6 = this_set->hdxvec[pnt_nr_1];
				sy6 = this_set->hdyvec[pnt_nr_1];

				ok2= 1;
				  /* 950619: runtime transformations are done before - pass -1 (True, really don't do do_TRANSFORM())
				   \ to do_transform's is_bounds argument
				   */
				if( pnt_nr ){
					do_transform( wi,
						this_set->fileName, __DLINE__, "DrawData()", &ok, this_set, &sx1, &sx3, &sx4, &sy1, &sy3, &sy4,
						&this_set->xvec[pnt_nr-1], &this_set->yvec[pnt_nr-1], 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
					);
					if( !DiscardedPoint( wi, this_set,pnt_nr_1) ){
						do_transform( wi,
							this_set->fileName, __DLINE__, "DrawData()", &ok2, this_set, &sx2, &sx5, &sx6, &sy2, &sy5, &sy6,
							&this_set->xvec[pnt_nr-1], &this_set->yvec[pnt_nr-1], 1, pnt_nr_1, 1.0, 1.0, 1.0, -1, 1, 0
						);
					}
				}
				else{
					do_transform( wi /* NULL */,
						this_set->fileName, __DLINE__, "DrawData()", &ok, this_set, &sx1, &sx3, &sx4, &sy1, &sy3, &sy4,
						NULL, NULL, 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
					);
					if( !DiscardedPoint( wi, this_set,pnt_nr_1) ){
						do_transform( wi /* NULL */,
							this_set->fileName, __DLINE__, "DrawData()", &ok2, this_set, &sx2, &sx5, &sx6, &sy2, &sy5, &sy6,
							NULL, NULL, 1, pnt_nr_1, 1.0, 1.0, 1.0, -1, 1, 0
						);
					}
				}
				if( !ok || NaN(sx1) || NaN(sy1) ){
				  /* first point couldn't be transformed or is a NaN - skip to the next
				   \ (second) point
				   */
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "DrawData() point[%d,%d]=(%s,%s): skipping\n",
							set_nr, pnt_nr,
							d2str(sx1,"%g", NULL), d2str( sy1, "%g", NULL)
						);
					}
					  /* 20031201 */
					SegsENR(XsegsE, X_set_nr)->use_first= True;
					  /* 20041002: */
					if( !X_seg_nr ){
						SegsENR(XsegsE, pnt_nr+1)->use_first= True;
					}
					goto next_point;
				}
#define NaNCompare(a,b)	if(NaN(a)){a=b;}
				NaNCompare( sx3, sx1);
				NaNCompare( sy3, sy1);
				NaNCompare( sx4, sx1);
				NaNCompare( sy4, sy1);
				if( !ok2 || NaN(sx2) || NaN(sy2) || DiscardedPoint( wi, this_set, pnt_nr_1) ){
				  /* second point couldn't be transformed or is a NaN - make it equal to
				   \ the (ok) first point
				   */
					if( debugFlag && debugLevel ){
						fprintf( StdErr, "DrawData() point[%d,%d]=(%s,%s): using previous point\n",
							set_nr, pnt_nr_1,
							d2str(sx2,"%g", NULL), d2str( sy2, "%g", NULL)
						);
					}
					sx2 = sx1;
					sy2 = sy1;
					sx5 = sx3;
					sy5 = sy3;
					sx6 = sx4;
					sy6 = sy4;
				}

				if( !(this_set->has_error && (w_use_errors && this_set->use_error) ) ){
					sy3= sy4= sy5= sy6= sx3= sx4= sx5= sx6= 0.0;
				}

				if( debugLevel>= 1){
					fprintf( StdErr, "%d,%d\t%9g,%9g - %9g,%9g + %9g,%9g : ok=%d\n",
						pnt_nr, X_set_nr, sx1, sy1, sx3, sy3, sx4, sy4, ok
					);
					fflush( StdErr);
				}
				  /* ok and ok2 fields indicate whether errors are OK to be shown.	*/
				SegsENR(XsegsE,X_set_nr)->ok2 = SegsENR(XsegsE,X_set_nr)->ok = 0;
				SegsENR(XsegsE,X_set_nr)->mark_inside1= mark_inside1;
				  /* The rx/ry fields of the XsegsE array contain the unclipped coordinates.	*/
				SegsENR(XsegsE,X_set_nr)->rx1= sx1;
				SegsENR(XsegsE,X_set_nr)->ry1= sy1;
				SegsENR(XsegsE,X_set_nr)->rx2= sx2;
				SegsENR(XsegsE,X_set_nr)->ry2= sy2;

				connect_errrgn= 0;
				  /* Now clip to current window boundary */
				if( ClipWindow( wi, this_set, False, &sx1, &sy1, &sx2, &sy2, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
				  /* both points were inside or could be clipped into the plotting window	*/
				  int e_mark_inside1, e_mark_inside2;	/* local variants for error-bars	*/
					if( !discarded_point ){
						SegsNR(Xsegs,X_seg_nr)->x1 = SCREENX(wi, sx1);
						SegsNR(Xsegs,X_seg_nr)->y1 = SCREENY(wi, sy1);
					}
					else{
					  /* 20000405: If we're here, than that means that we want to draw shadows
					   \ of the deleted points. These must be drawn in a "shadowy" colour, so
					   \ care must be taken not to store them in the Xsegs list which will be
					   \ used to draw the linesegments. Because this would redraw the same points
					   \ in the set's colour... Since we do need the screen co-ordinates, we
					   \ store them in XXsegs->s[xy]1.
					   \ This is a fast workaround in code that I had not "visited" for a long time.
					   \ It is likely that it solves the problem (points *were* stored in Xsegs), but
					   \ I have not yet any idea of possible side-effects.
					   */
						XSegsNR(XXsegs,X_set_nr)->sx1 = SCREENX( wi, sx1);
						XSegsNR(XXsegs,X_set_nr)->sy1 = SCREENY( wi, sy1);
					}
					XSegsNR(XXsegs,X_set_nr)->x1 = sx1;
					XSegsNR(XXsegs,X_set_nr)->y1 = sy1;
					SegsENR(XsegsE, X_set_nr)->pnt_nr1= pnt_nr;
					if( clipcode2 || SplitHere(this_set, pnt_nr_1) ){
					  /* Second point not OK or split off; cut the list	*/
						if( !discarded_point ){
							SegsNR(Xsegs,X_seg_nr)->x2 = SegsNR(Xsegs,X_seg_nr)->x1;
							SegsNR(Xsegs,X_seg_nr)->y2 = SegsNR(Xsegs,X_seg_nr)->y1;
							  /* 20031201 */
							SegsENR(XsegsE, X_set_nr)->use_first= True;
							  /* 20041002: */
							if( !X_seg_nr ){
								SegsENR(XsegsE, pnt_nr+1)->use_first= True;
							}
						}
						else{
							XSegsNR(XXsegs,X_set_nr)->sx2 = SCREENX( wi, sx2);
							XSegsNR(XXsegs,X_set_nr)->sy2 = SCREENY( wi, sy2);
						}
						XSegsNR(XXsegs,X_set_nr)->x2 = XSegsNR(XXsegs,X_set_nr)->x1;
						XSegsNR(XXsegs,X_set_nr)->y2 = XSegsNR(XXsegs,X_set_nr)->y1;
						SegsENR(XsegsE, X_set_nr)->pnt_nr2= pnt_nr;
					}
					else{
					  /* Second point OK too, add segment to list */
						if( !discarded_point ){
							SegsNR(Xsegs,X_seg_nr)->x2= SCREENX(wi, sx2);
							SegsNR(Xsegs,X_seg_nr)->y2 = SCREENY(wi, sy2);
							if( Determine_AvSlope ){
								switch( Determine_AvSlope ){
									case 1:
									default:
										SAS_Add_Data_( wi->SAS_scrslope, 1,
											ArcTan( wi, (SegsNR(Xsegs,X_seg_nr)->x2-SegsNR(Xsegs,X_seg_nr)->x1),
												(SegsNR(Xsegs,X_seg_nr)->y2-SegsNR(Xsegs,X_seg_nr)->y1)), 1, 1 );
										break;
									case 2:
										SAS_Add_Data_( wi->SAS_scrslope, 1,
											fabs( conv_angle_( ArcTan( wi, (SegsNR(Xsegs,X_seg_nr)->x2-SegsNR(Xsegs,X_seg_nr)->x1),
													(SegsNR(Xsegs,X_seg_nr)->y2-SegsNR(Xsegs,X_seg_nr)->y1)), wi->radix)), 1, 0 );
										break;
								}
							}
						}
						else{
							XSegsNR(XXsegs,X_set_nr)->sx2 = SCREENX( wi, sx2);
							XSegsNR(XXsegs,X_set_nr)->sy2 = SCREENY( wi, sy2);
						}
						XSegsNR(XXsegs,X_set_nr)->x2= sx2;
						XSegsNR(XXsegs,X_set_nr)->y2 = sy2;
						SegsENR(XsegsE, X_set_nr)->pnt_nr2= pnt_nr+ 1;
					}
					if( this_set->has_error && (w_use_errors && this_set->use_error) && !wi->error_region &&
						!discarded_point
					){
						if( mark_inside1 &&
							ClipWindow( wi, this_set, False, &sx3, &sy3, &sx4, &sy4, &e_mark_inside1, &e_mark_inside2, NULL, NULL )
						){
							SegsENR(XsegsE,X_set_nr)->X1 = SCREENX(wi, sx1);
							SegsENR(XsegsE,X_set_nr)->Y1 = SCREENY(wi, sy1);
							SegsENR(XsegsE,X_set_nr)->x1 = SCREENX(wi, sx3);
							SegsENR(XsegsE,X_set_nr)->y1l= SCREENY(wi, sy3);
							SegsENR(XsegsE,X_set_nr)->e_mark_inside1= e_mark_inside1;
							SegsENR(XsegsE,X_set_nr)->x1h= SCREENX(wi, sx4);
							SegsENR(XsegsE,X_set_nr)->y1h= SCREENY(wi, sy4);
							SegsENR(XsegsE,X_set_nr)->e_mark_inside2= e_mark_inside2;
							if( !(SegsENR(XsegsE,X_set_nr)->x1== SegsENR(XsegsE,X_set_nr)->x1h &&
									SegsENR(XsegsE,X_set_nr)->y1l== SegsENR(XsegsE,X_set_nr)->y1h)
							){
								SegsENR(XsegsE,X_set_nr)->ok= 1;
							}
						}
						if( mark_inside2 &&
							ClipWindow( wi, this_set, False, &sx5, &sy5, &sx6, &sy6, &e_mark_inside1, &e_mark_inside2, NULL, NULL )
						){
							SegsENR(XsegsE,X_set_nr)->X2 = SCREENX(wi, sx2);
							SegsENR(XsegsE,X_set_nr)->Y2 = SCREENY(wi, sy2);
							SegsENR(XsegsE,X_set_nr)->x2 = SCREENX(wi, sx5);
							SegsENR(XsegsE,X_set_nr)->y2l= SCREENY(wi, sy5);
							SegsENR(XsegsE,X_set_nr)->x2h= SCREENX(wi, sx6);
							SegsENR(XsegsE,X_set_nr)->y2h= SCREENY(wi, sy6);
							if( !(SegsENR(XsegsE,X_set_nr)->x2== SegsENR(XsegsE,X_set_nr)->x2h &&
									SegsENR(XsegsE,X_set_nr)->y2l== SegsENR(XsegsE,X_set_nr)->y2h)
							){
								SegsENR(XsegsE,X_set_nr)->ok2= 1;
							}
						}
					}
					if( markFlag && mark_inside1 ){
						SegsENR(XsegsE,X_set_nr)->mark_inside1= mark_inside1;
					}
					if( debugFlag && debugLevel>= 2){
						fprintf( StdErr, "%d:\t(%6d,%6d) -(%6d,%6d) +(%6d,%6d) %d,%d m=%d,%d c=%s,%s\n",
							set_nr, (int) SegsENR(XsegsE,X_set_nr)->X1, (int) SegsENR(XsegsE,X_set_nr)->Y1,
							(int) SegsENR(XsegsE,X_set_nr)->x1, (int) SegsENR(XsegsE,X_set_nr)->y1l, (int) SegsENR(XsegsE,X_set_nr)->x1h, (int) SegsENR(XsegsE,X_set_nr)->y1h,
							(int) SegsENR(XsegsE,X_set_nr)->ok, (int) SegsENR(XsegsE,X_set_nr)->ok2,
							mark_inside1, mark_inside2,
							clipcode( clipcode1), clipcode( clipcode2)
						);
						fflush( StdErr);
					}
					pX_set_nr= X_set_nr;

					if( !discarded_point ){
						if( !X_seg_nr ){
							X_seg_nr_diff= X_set_nr;
						}
						X_seg_nr++;
					}
					X_set_nr++;

				}
				else if( debugFlag && debugLevel>= 2){
					fprintf( StdErr, "%d:\t(%g,%g)->(%g,%g) m=%d,%d c=%s,%s\n",
						set_nr, sx1, sy1, sx2, sy2,
						mark_inside1, mark_inside2,
						clipcode( clipcode1), clipcode( clipcode2)
					);
					fflush( StdErr);
				}

				if( mark_inside1 ){
					wi->numVisible[set_nr]+= 1;
					wi->pointVisible[set_nr][pnt_nr]= 1;
					connect_errrgn= 1;
					connect_id= X_set_nr-1;
				}
				else{
					wi->pointVisible[set_nr][pnt_nr]= 0;
					if( wi->discard_invisible && !wi->fitting ){
						if( !wi->discardpoint || !wi->discardpoint[set_nr] ){
							realloc_WinDiscard( wi, MaxSets);
						}
						if( !wi->discardpoint[set_nr] ){
							wi->discardpoint[set_nr]= (signed char*) calloc( this_set->allocSize, sizeof(signed char));
						}
						if( wi->discardpoint && wi->discardpoint[set_nr] ){
							wi->discardpoint[set_nr][pnt_nr]= -1;
						}
					}
				}

				  /* Draw markers if requested and they are in drawing region */
				if( ((adorn /* && markFlag */) || discarded_point) && mark_inside1 ){
					if( discarded_point ){
						SegsENR(XsegsE,X_set_nr-1)->mark1= False;
						/* if( markFlag ) */{
							wi->dev_info.xg_dot(wi->dev_info.user_state,
										XSegsNR(XXsegs,X_set_nr-1)->sx1, XSegsNR(XXsegs,X_set_nr-1)->sy1,
										P_PIXEL, 0, -1, dshadowPixel, set_nr, this_set);
						}
						goto next_point;
					}
					else{
						SegsENR(XsegsE,X_set_nr-1)->mark1= True;
					}
					if( markFlag ){
						if( /* noLines || */ !this_set->overwrite_marks ){
						  Pixel colorx;
						  int respix= False;
						  double ms= this_set->markSize;
							if( etype== INTENSE_FLAG && IntensityColourFunction.XColours ){
								Retrieve_IntensityColour( wi, this_set, this_set->errvec[pnt_nr],
									minIntense, maxIntense, scaleIntense, &colorx, &respix
								);
								psThisRGB= xg_IntRGB;
							}
							if( etype== MSIZE_FLAG ){
								this_set->markSize= -fabs( this_set->errvec[pnt_nr] * ((NaNorInf(ms))? 1 : ms) );
							}
							switch( pixelMarks ){
								case 2:
									wi->dev_info.xg_dot(wi->dev_info.user_state,
												SegsNR(Xsegs,X_seg_nr-1)->x1, SegsNR(Xsegs,X_seg_nr-1)->y1,
	/* 											P_DOT, 0, set_nr % MAXATTR, 0,  set_nr, this_set);	*/
												P_DOT, 0, PIXVALUE(set_nr),  set_nr, this_set);
									break;
								case 1:
									wi->dev_info.xg_dot(wi->dev_info.user_state,
												SegsNR(Xsegs,X_seg_nr-1)->x1, SegsNR(Xsegs,X_seg_nr-1)->y1,
												P_PIXEL, 0, PIXVALUE(set_nr), set_nr, this_set);
									break;
								case 0:
									/* Distinctive markers */
									wi->dev_info.xg_dot(wi->dev_info.user_state,
											SegsNR(Xsegs,X_seg_nr-1)->x1, SegsNR(Xsegs,X_seg_nr-1)->y1,
											P_MARK, MARKSTYLE(set_nr),
											PIXVALUE(set_nr), set_nr, this_set);
									break;
							}
							if( respix ){
								if( this_set->pixvalue< 0 ){
									this_set->pixelValue= colorx;
								}
								else{
									AllAttrs[this_set->pixvalue].pixelValue= colorx;
								}
							}
							this_set->markSize= ms;
						}
					}
					if( CheckMask(this_set->valueMarks, VMARK_ON) && !this_set->overwrite_marks ){
						Draw_valueMark( wi, this_set, pnt_nr,
							SegsNR(Xsegs,X_seg_nr-1)->x1, SegsNR(Xsegs,X_seg_nr-1)->y1,
							PIXVALUE( set_nr )
						);
					}
				}

				if( this_set->has_error && (w_use_errors && this_set->use_error) ){ if( wi->error_region ){
				  /* Error region requested: build the 2 XSegment lists of low and high Y points
				   \ This is handled separately from the (x,y) point itself, since the current node in
				   \ the error region's outline must be shown (if visible) even if the (x,y) point is not
				   \ visible. The error region lines are connected to the first and last data (x,y) points
				   \ visible.
				   */
					if( LYsegs && ClipWindow( wi, this_set, False, &sx3, &sy3, &sx5, &sy5, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
					  /* both points were inside or could be clipped into the plotting window	*/
						if( LY_set_nr== 0 && mark_inside1 && connect_errrgn && connect_id== X_set_nr-1 ){
							SegsNR(LYsegs,0)->x1= SegsNR(Xsegs,X_seg_nr-1)->x1;
							SegsNR(LYsegs,0)->y1= SegsNR(Xsegs,X_seg_nr-1)->y1;
							LY_set_nr= 1;
						}
						SegsNR(LYsegs,LY_set_nr)->x1 = SCREENX(wi, sx3);
						SegsNR(LYsegs,LY_set_nr)->y1 = SCREENY(wi, sy3);
						if( LY_set_nr== 1 ){
							SegsNR(LYsegs,0)->x2= SegsNR(LYsegs,1)->x1;
							SegsNR(LYsegs,0)->y2= SegsNR(LYsegs,1)->y1;
						}
						if( clipcode2 || SplitHere(this_set, pnt_nr_1) ){
						  /* Second point not OK, cut the list	*/
/* 951222: used to be SegsNR(Xsegs,X_set_nr)->?1	*/
							SegsNR(LYsegs,LY_set_nr)->x2 = SegsNR(LYsegs,LY_set_nr)->x1;
							SegsNR(LYsegs,LY_set_nr)->y2 = SegsNR(LYsegs,LY_set_nr)->y1;
						}
						else{
						  /* Second point OK too, add segment to list */
							SegsNR(LYsegs,LY_set_nr)->x2= SCREENX(wi, sx5);
							SegsNR(LYsegs,LY_set_nr)->y2 = SCREENY(wi, sy5);
						}
						LY_set_nr++;
					}
					if( HYsegs && ClipWindow( wi, this_set, False, &sx4, &sy4, &sx6, &sy6, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
					  /* both points were inside or could be clipped into the plotting window	*/
						if( HY_set_nr== 0 && mark_inside1 && connect_errrgn && connect_id== X_set_nr-1 ){
							SegsNR(HYsegs,0)->x1= SegsNR(Xsegs,X_seg_nr-1)->x1;
							SegsNR(HYsegs,0)->y1= SegsNR(Xsegs,X_seg_nr-1)->y1;
							HY_set_nr= 1;
						}
						SegsNR(HYsegs,HY_set_nr)->x1 = SCREENX(wi, sx4);
						SegsNR(HYsegs,HY_set_nr)->y1 = SCREENY(wi, sy4);
						if( HY_set_nr== 1 ){
							SegsNR(HYsegs,0)->x2= SegsNR(HYsegs,1)->x1;
							SegsNR(HYsegs,0)->y2= SegsNR(HYsegs,1)->y1;
						}
						if( clipcode2 || SplitHere(this_set, pnt_nr_1) ){
						  /* Second point not OK, cut the list	*/
/* 951222: used to be SegsNR(Xsegs,X_set_nr)->?1	*/
							SegsNR(HYsegs,HY_set_nr)->x2 = SegsNR(HYsegs,HY_set_nr)->x1;
							SegsNR(HYsegs,HY_set_nr)->y2 = SegsNR(HYsegs,HY_set_nr)->y1;
						}
						else{
						  /* Second point OK too, add segment to list */
							SegsNR(HYsegs,HY_set_nr)->x2= SCREENX(wi, sx6);
							SegsNR(HYsegs,HY_set_nr)->y2 = SCREENY(wi, sy6);
						}
						HY_set_nr++;
					}
				} else if( (etype<= 2 || etype== EREGION_FLAG) && this_set->ebarWidth_set ){
					if( this_set->ebarWidth<= 0.0){
					  double w2= fabs(sx2-sx1);
					  double w1= (pnt_nr)? fabs(sx1 - (sx1_1+ epbarPixels/2.0))* 2 : w2;
						epbarPixels= MIN(w1, w2);
						if( this_set->adorn_interval && (this_set->plot_interval % this_set->adorn_interval) ){
							epbarPixels*= this_set->adorn_interval;
						}
						if( !wi->eBarDimensionsSet ){
							wi->eMinBarX= sx1- epbarPixels/ 2;
							wi->eMaxBarX= sx1+ epbarPixels/ 2;
							wi->eBarDimensionsSet= 1;
						}
						else{
							wi->eMinBarX= MIN( sx1- epbarPixels/ 2, wi->eMinBarX );
							wi->eMaxBarX= MAX( sx1+ epbarPixels/ 2, wi->eMaxBarX );
						}
						ebarPixels= ( ( epbarPixels / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) + this_set->ebarWidth);
						this_set->current_ebarWidth= ebarPixels* (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)/ wi->Xscale;
						this_set->current_ebW_set= True;
						if( debugFlag && debugLevel> 2 ){
							fprintf( StdErr, "this_set->ebarWidth=%g/%g+ %g = %g pixels\n",
								epbarPixels, wi->XUnitsPerPixel* wi->dev_info.var_width_factor, this_set->ebarWidth, ebarPixels
							);
						}
					}
					else{
					  double bcx= this_set->xvec[pnt_nr],
							blx= bcx- this_set->ebarWidth/ 2,
					  		brx= blx+ this_set->ebarWidth,
							bty= this_set->yvec[pnt_nr];
					  int bok= 1, vf= wi->vectorFlag;
						if( !wi->eBarDimensionsSet ){
							wi->eMinBarX= blx;
							wi->eMaxBarX= brx;
							wi->eBarDimensionsSet= 1;
						}
						else{
							wi->eMinBarX= MIN( blx, wi->eMinBarX );
							wi->eMaxBarX= MAX( brx, wi->eMaxBarX );
						}
						wi->vectorFlag= 1;
						do_transform( wi /* NULL */,
							this_set->fileName, __DLINE__, "DrawData()", &bok, NULL, &bcx, &blx, &brx, &bty, NULL, NULL,
							NULL, NULL, 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
						);
						wi->vectorFlag= vf;
/* 						ebarPixels = ((this_set->ebarWidth / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) );	*/
						ebarPixels = (( fabs(brx- blx) / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) );
						  /* Set the current width to the requested value. Deviations from that value are the result
						   \ of transformations, and should not be taken into account. Whatever the current width
						   \ is used for afterwards, it will be subject to the same transformations!
						   */
						this_set->current_ebarWidth= this_set->ebarWidth/ wi->Xscale;
						this_set->current_ebW_set= True;
					}
					if( ebarPixels <= 0){
						ebarPixels = 1;
					}
				} }

				  /* Draw bar elements if requested */
				if( X_set_nr && barFlag ){
				  double barPixels;
				  int baseSpot;
				  XSegment line;
				  double lw= LINEWIDTH(wi, set_nr)* wi->dev_info.var_width_factor, bp;
				  XRectangle rec;

					if( !barBase_set && !this_set->barBase_set ){
						barBase= wi->win_geo._UsrOrgY;
						baseSpot = SCREENY(wi, barBase);
					}
					else{
						baseSpot = SCREENY(wi, this_set->barBaseY);
					}
					if( this_set->barWidth<= 0.0){
					  double w2= fabs(sx2-sx1);
					  double w1= (pnt_nr)? fabs(sx1 - (sx1_1+ pbarPixels/2.0))* 2 : w2;
						pbarPixels= MIN(w1, w2);
						if( !wi->BarDimensionsSet ){
							wi->MinBarX= sx1- pbarPixels/ 2;
							wi->MaxBarX= sx1+ pbarPixels/ 2;
							wi->BarDimensionsSet= 1;
						}
						else{
							wi->MinBarX= MIN( sx1- pbarPixels/ 2, wi->MinBarX );
							wi->MaxBarX= MAX( sx1+ pbarPixels/ 2, wi->MaxBarX );
						}
						barPixels= ( ( pbarPixels / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) + this_set->barWidth);
						this_set->current_barWidth= barPixels* (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)/ wi->Xscale;
						this_set->current_bW_set= True;
						if( debugFlag && debugLevel> 2 ){
							fprintf( StdErr, "this_set->barWidth=%g/%g+ %g = %g pixels\n",
								pbarPixels, wi->XUnitsPerPixel* wi->dev_info.var_width_factor, this_set->barWidth, barPixels
							);
						}
						line.x1 = line.x2 = SegsNR(Xsegs,X_seg_nr-1)->x1;
					}
					else{
					  double bcx= this_set->xvec[pnt_nr],
							blx= bcx- this_set->barWidth/ 2,
					  		brx= blx+ this_set->barWidth,
							bty= this_set->yvec[pnt_nr];
					  int bok= 1, vf= wi->vectorFlag;
						if( !wi->BarDimensionsSet ){
							wi->MinBarX= blx;
							wi->MaxBarX= brx;
							wi->BarDimensionsSet= 1;
						}
						else{
							wi->MinBarX= MIN( blx, wi->MinBarX );
							wi->MaxBarX= MAX( brx, wi->MaxBarX );
						}
						  /* temporarily set vectorFlag to prevent blx and brx to be set to bcx:	*/
						wi->vectorFlag= 1;
						  /* Determine the left (blx) and right (brx) boundaries of the bar rectangle:	*/
						do_transform( wi /* NULL */,
							this_set->fileName, __DLINE__, "DrawData()", &bok, NULL, &bcx, &blx, &brx, &bty, NULL, NULL,
							NULL, NULL, 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
						);
						wi->vectorFlag= vf;
						barPixels = (( fabs(brx- blx) / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) );
						this_set->current_barWidth= this_set->barWidth/ wi->Xscale;
						this_set->current_bW_set= True;
						  /* The bar may be asymmetric around its "official" centre, e.g. because of a logXFlag. Thus,
						   \ we calculate a new centre that is in the middle between the transformed left and right
						   \ boundaries.
						   */
						line.x1 = line.x2 = SCREENX( wi, (blx+ brx)/2 );
					}
					if( barPixels <= 0){
						barPixels = 1;
					}
					line.y1 = baseSpot;  line.y2 = SegsNR(Xsegs,X_seg_nr-1)->y1;
					  /* For a barplot, we must compensate for the linewidth in order
					   \ to get the rect in the right space/dimensions.
					   */
					if( lw> 1 ){
						  /* But I don't think I'll have to compensate for 1 pixel wide lines.	*/
						bp= barPixels* wi->dev_info.var_width_factor- lw;
					}
					else{
						lw= 0;
						bp= barPixels* wi->dev_info.var_width_factor;
					}
					rec= *rect_xywh( line.x1-bp/2,MIN(line.y1,line.y2)+ lw/2,
							bp, MAX(line.y1,line.y2)-MIN(line.y1,line.y2)- lw
					);
/* 					switch( this_set->barType ){	*/
/* 						default:	*/
						{ int lw= LINEWIDTH(wi,set_nr);
							Draw_Bar( wi, &rec, &line, barPixels, this_set->barType, LStyle, this_set, set_nr, pnt_nr,
								lw, LINESTYLE(set_nr), lw, 0,
								minIntense, maxIntense, scaleIntense, 0, 0
							);
/* 							break;	*/
						}
/* 					}	*/
				}
				if( adorn && X_set_nr && SegsENR(XsegsE,X_set_nr-1)->ok){
					Draw_ErrorBar( wi, this_set, set_nr, pX_set_nr, X_set_nr, pnt_nr, True,
						ebarPixels, LStyle, aspect
					);
				}
				sx1_1= sx1;
next_point:;
				pX_set_nr= X_set_nr;
			}
			if( wi->use_transformed ){
				wi->curve_len[set_nr][this_set->numPoints-1]= curve_len;
				wi->curve_len[set_nr][this_set->numPoints]= curve_len;
				wi->error_len[set_nr][this_set->numPoints-1]= error_len;
				wi->error_len[set_nr][this_set->numPoints]= error_len;
			}
#ifdef TR_CURVE_LEN
			wi->tr_curve_len[set_nr][this_set->numPoints-1]= tr_curve_len;
			wi->tr_curve_len[set_nr][this_set->numPoints]= tr_curve_len;
#endif
			SS_Copy( &wi->set_tr_X[set_nr], &set_tr_X);
			SS_Copy( &wi->set_tr_Y[set_nr], &set_tr_Y);
			SS_Copy( &wi->set_tr_E[set_nr], &set_tr_E);
			SAS_Copy( &wi->set_tr_O[set_nr], &set_tr_O);

			  /* pnt_nr == numPoints-1	*/
			this_set->data[0][0]= sx1 = this_set->xvec[pnt_nr];
			this_set->data[0][1]= sy1 = this_set->yvec[pnt_nr];
			this_set->data[0][2]= this_set->errvec[pnt_nr];
			this_set->data[0][3]= this_set->lvec[pnt_nr];

			if( NaN(sx1) || NaN(sy1) ){
			  /* current (last) point is a NaN - skip to the next point */
				if( debugFlag && debugLevel ){
					fprintf( StdErr, "DrawData() point[%d,%d]=(%s,%s,%s): skipping\n",
						set_nr, pnt_nr,
						d2str(sx1,"%g", NULL), d2str( sy1, "%g", NULL),
						d2str(this_set->errvec[pnt_nr], "%g", NULL)
					);
				}
				goto last_drawn;
			}

			adorn= (discarded_point= DiscardedPoint( wi, this_set, pnt_nr))? False : True;
			sx3 = this_set->ldxvec[pnt_nr];
			sy3 = this_set->ldyvec[pnt_nr];
			sx4 = this_set->hdxvec[pnt_nr];
			sy4 = this_set->hdyvec[pnt_nr];

			if( pnt_nr && !discarded_point && !DiscardedPoint( wi, this_set, pnt_nr-1) ){
				ok= 1;
				  /* there is a previous point	*/
				this_set->data[1][0]= sx2 = this_set->xvec[pnt_nr-1];
				this_set->data[1][1]= sy2 = this_set->yvec[pnt_nr-1];
				this_set->data[1][2]= this_set->errvec[pnt_nr-1];
				this_set->data[1][3]= this_set->lvec[pnt_nr-1];

				sx5 = this_set->ldxvec[pnt_nr-1];
				sy5 = this_set->ldyvec[pnt_nr-1];
				sx6 = this_set->hdxvec[pnt_nr-1];
				sy6 = this_set->hdyvec[pnt_nr-1];

				do_transform( wi /* NULL */,
					this_set->fileName, __DLINE__, "DrawData()", &ok, this_set, &sx1, &sx3, &sx4, &sy1, &sy3, &sy4,
					&this_set->xvec[pnt_nr-1], &this_set->yvec[pnt_nr-1], 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
				);

				do_transform( wi /* NULL */,
					this_set->fileName, __DLINE__, "DrawData()", &ok, this_set, &sx2, &sx5, &sx6, &sy2, &sy5, &sy6,
					&this_set->xvec[pnt_nr-2], &this_set->yvec[pnt_nr-2], 1, pnt_nr-1, 1.0, 1.0, 1.0, -1, 1, 0
				);
			}
			else{
				this_set->data[1][0]= sx2= sx1; this_set->data[1][1]= sy2= sy1;
				this_set->data[1][2]= this_set->data[0][2];
				this_set->data[1][3]= this_set->data[0][3];
				sx5= sx3; sy5= sy3;
				sx6= sx4; sy6= sy4;
				ok= 1;
				do_transform( wi /* NULL */,
					this_set->fileName, __DLINE__, "DrawData()", &ok, this_set, &sx1, &sx3, &sx4, &sy1, &sy3, &sy4,
					NULL, NULL, 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
				);
			}

			if( !ok || NaN(sx1) || NaN(sy1) ){
			  /* last point couldn't be transformed or is a NaN - skip to the next
			   \ (second) point
			   */
				if( debugFlag && debugLevel ){
					fprintf( StdErr, "DrawData() point[%d,%d]=(%s,%s): skipping\n",
						set_nr, pnt_nr,
						d2str(sx1,"%g", NULL), d2str( sy1, "%g", NULL)
					);
				}
				goto last_drawn;
			}
			NaNCompare( sx3, sx1);
			NaNCompare( sy3, sy1);
			NaNCompare( sx4, sx1);
			NaNCompare( sy4, sy1);
			if( !ok2 || NaN(sx2) || NaN(sy2) ){
			  /* second point couldn't be transformed or is a NaN - make it equal to
			   \ the (ok) first point
			   */
				if( debugFlag && debugLevel ){
					fprintf( StdErr, "DrawData() point[%d,%d]=(%s,%s): using previous point\n",
						set_nr, pnt_nr+ 1,
						d2str(sx2,"%g", NULL), d2str( sy2, "%g", NULL)
					);
				}
				sx2 = sx1;
				sy2 = sy1;
				sx5 = sx3;
				sy5 = sy3;
				sx6 = sx4;
				sy6 = sy4;
			}

			if( X_set_nr> 0 ){
				SegsENR(XsegsE,X_set_nr-1)->rx1= sx2;
				SegsENR(XsegsE,X_set_nr-1)->ry1= sy2;
				SegsENR(XsegsE,X_set_nr-1)->rx2= sx1;
				SegsENR(XsegsE,X_set_nr-1)->ry2= sy1;
			}

			  /* Handle last segment/marker
			   \ Not that (sx1,sy1) contains the *second* coordinate pair of the last
			   \ segment. (sx2,sy2) contains the first, which is the same if only one
			   \ point is to be drawn.
			   */
			connect_errrgn= 0;
			if( X_set_nr> 0 ){
				if( ok && ClipWindow( wi, this_set, False, &sx2, &sy2, &sx1, &sy1, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
				  int e_mark_inside1, e_mark_inside2;	/* local variants for error-bars	*/
				  int xx, yy;
					  /* Add segment to list */
					if( !discarded_point ){
						xx= SegsNR(Xsegs,X_seg_nr-1)->x2 = SCREENX(wi, sx1);
						yy= SegsNR(Xsegs,X_seg_nr-1)->y2 = SCREENY(wi, sy1);
						if( /* discarded_point || */ DiscardedPoint( wi, this_set,pnt_nr-1) ){
							SegsNR(Xsegs,X_seg_nr-1)->x1 = xx;
							SegsNR(Xsegs,X_seg_nr-1)->y1 = yy;
						}
					}
					else{
					  /* 20000405: If we're here, than that means that we want to draw shadows. See above	*/
						XSegsNR(XXsegs,X_set_nr-1)->sx2 = SCREENX(wi,sx1);
						XSegsNR(XXsegs,X_set_nr-1)->sy2 = SCREENY(wi,sy1);
					}
					XSegsNR(XXsegs,X_set_nr-1)->x2 = sx1;
					XSegsNR(XXsegs,X_set_nr-1)->y2 = sy1;
					SegsENR(XsegsE, X_set_nr)->pnt_nr2= pnt_nr;
					  /* 20001010: following 2 lines added.
					   \ The 2nd is (probably) what is really necessary to correctly identify the set's points
					   \ referenced by the last XsegsE element (used a.o. to overwrite draw the last's point valueMark
					   \ even when the forelast point is deleted).
					   */
					SegsENR(XsegsE, X_set_nr)->pnt_nr1= pnt_nr;
					SegsENR(XsegsE, X_set_nr-1)->pnt_nr2= pnt_nr;
					  /* This one has been counted already!	*/
/*		 			wi->numVisible[set_nr]+= 1;	*/
					if( adorn && this_set->has_error && (w_use_errors && this_set->use_error) && !wi->error_region ){
						if( mark_inside1 && ClipWindow( wi, this_set, False, &sx3, &sy3, &sx4, &sy4, &e_mark_inside1, &e_mark_inside2, NULL, NULL ) ){
							SegsENR(XsegsE,X_set_nr-1)->X2 = SCREENX(wi, sx1);
							SegsENR(XsegsE,X_set_nr-1)->Y2 = SCREENY(wi, sy1);
							SegsENR(XsegsE,X_set_nr-1)->x2 = SCREENX(wi, sx3);
							SegsENR(XsegsE,X_set_nr-1)->y2l= SCREENY(wi, sy3);
							  /* actually, we should have an e_mark_inside[12] for y2[lh]. Let's see
							   \ if this also works.. errorbar drawing takes place immediately
							   \ after this
							   */
							SegsENR(XsegsE,X_set_nr)->e_mark_inside1= e_mark_inside1;
							SegsENR(XsegsE,X_set_nr-1)->x2h= SCREENX(wi, sx4);
							SegsENR(XsegsE,X_set_nr-1)->y2h= SCREENY(wi, sy4);
							SegsENR(XsegsE,X_set_nr)->e_mark_inside2= e_mark_inside2;
							if( !(SegsENR(XsegsE,X_set_nr-1)->x2== SegsENR(XsegsE,X_set_nr-1)->x2h &&
									SegsENR(XsegsE,X_set_nr-1)->y2l== SegsENR(XsegsE,X_set_nr-1)->y2h)
							){
								SegsENR(XsegsE,X_set_nr-1)->ok= 1;
							}
						}
					}
					else{
						SegsENR(XsegsE,X_set_nr-1)->ok= 0;
					}
				}
				else{
					SegsENR(XsegsE,X_set_nr-1)->ok= 0;
				}
			}

			if( ok && mark_inside2 && X_set_nr> 0 && (AllSets[set_nr].numPoints > 0) ){
				if( !discarded_point ){
					wi->numVisible[set_nr]+= 1;
					wi->pointVisible[set_nr][pnt_nr]= 1;
				}
				connect_errrgn= 1;
				connect_id= X_set_nr-1;
			}
			else if( wi->discard_invisible && !wi->fitting ){
				if( !wi->discardpoint || !wi->discardpoint[set_nr] ){
					realloc_WinDiscard( wi, MaxSets);
				}
				if( !wi->discardpoint[set_nr] ){
					wi->discardpoint[set_nr]= (signed char*) calloc( this_set->allocSize, sizeof(signed char));
				}
				if( wi->discardpoint && wi->discardpoint[set_nr] ){
					wi->discardpoint[set_nr][pnt_nr]= -1;
				}
			}

			  /* mark_inside1 and mark_inside2 are the same because of the
			   \ previous call to ClipWindow(), but their value should be stored
			   \ in SegsENR(XsegsE,X_set_nr-1)->mark_inside2 (if at all), since for the last
			   \ point/mark the second coordinate set is being used (x2,y2)
			   \ 950323: mark_inside 1&2 no longer the same (ClipWindow used to be called
			   \ with sx1,sy1,sx1,sy2).
			   */
			if( ((adorn /* && markFlag */) || (discarded_point && DiscardedShadows)) && ok &&
				mark_inside2 && X_set_nr> 0 && (AllSets[set_nr].numPoints > 0)
			){
				if( discarded_point ){
					SegsENR(XsegsE,X_set_nr-1)->mark2= False;
					SegsENR(XsegsE,X_set_nr-1)->mark_inside2= False;
					/* if( markFlag ) */{
						wi->dev_info.xg_dot(wi->dev_info.user_state,
									XSegsNR(XXsegs,X_set_nr-1)->sx2, XSegsNR(XXsegs,X_set_nr-1)->sy2,
									P_PIXEL, 0, -1, dshadowPixel, set_nr, this_set);
					}
					goto last_drawn;
				}
				else{
					SegsENR(XsegsE,X_set_nr-1)->mark_inside2= mark_inside2;
					SegsENR(XsegsE,X_set_nr-1)->mark2= True;
				}
				if( markFlag ){
					 /* last marker is OK: possibly requested error-region should be "closed"
					  \ on this marker.
					  */
					if( /* noLines ||	*/ !this_set->overwrite_marks ){
					  Pixel colorx;
					  int respix= False;
					  double ms= this_set->markSize;
						if( etype== INTENSE_FLAG && IntensityColourFunction.XColours ){
							Retrieve_IntensityColour( wi, this_set, this_set->errvec[pnt_nr],
								minIntense, maxIntense, scaleIntense, &colorx, &respix
							);
							psThisRGB= xg_IntRGB;
						}
						if( etype== MSIZE_FLAG ){
							this_set->markSize= -fabs( this_set->errvec[pnt_nr] * ((NaNorInf(ms))? 1 : ms) );
						}
						switch( pixelMarks ){
							case 2:
								wi->dev_info.xg_dot(wi->dev_info.user_state,
											SegsNR(Xsegs,X_seg_nr-1)->x2, SegsNR(Xsegs,X_seg_nr-1)->y2,
	/* 										P_DOT, 0, set_nr % MAXATTR, 0, set_nr, this_set);	*/
											P_DOT, 0, PIXVALUE(set_nr), set_nr, this_set);
								break;
							case 1:
								wi->dev_info.xg_dot(wi->dev_info.user_state,
											SegsNR(Xsegs,X_seg_nr-1)->x2, SegsNR(Xsegs,X_seg_nr-1)->y2,
											P_PIXEL, 0, PIXVALUE(set_nr), set_nr, this_set);
								break;
							case 0:
								/* Distinctive markers */
								wi->dev_info.xg_dot(wi->dev_info.user_state,
										SegsNR(Xsegs,X_seg_nr-1)->x2, SegsNR(Xsegs,X_seg_nr-1)->y2,
										P_MARK, MARKSTYLE(set_nr),
										PIXVALUE(set_nr), set_nr, this_set);
								break;
						}
						if( respix ){
							if( this_set->pixvalue< 0 ){
								this_set->pixelValue= colorx;
							}
							else{
								AllAttrs[this_set->pixvalue].pixelValue= colorx;
							}
						}
						this_set->markSize= ms;
					}
					if( CheckMask(this_set->valueMarks, VMARK_ON) && !this_set->overwrite_marks ){
						Draw_valueMark( wi, this_set, pnt_nr,
							SegsNR(Xsegs,X_seg_nr-1)->x2, SegsNR(Xsegs,X_seg_nr-1)->y2,
							PIXVALUE( set_nr )
						);
					}
				}
			}

			  /* Last segment of possibly requested error region	*/
			if( ok && this_set->has_error && (w_use_errors && this_set->use_error) ){ if( wi->error_region ){
				if( LYsegs && ClipWindow( wi, this_set, False, &sx5, &sy5, &sx3, &sy3, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
					  /* Add segment to list */
					SegsNR(LYsegs,LY_set_nr)->x1= SegsNR(LYsegs,LY_set_nr-1)->x2 = SCREENX(wi, sx3);
					SegsNR(LYsegs,LY_set_nr)->y1= SegsNR(LYsegs,LY_set_nr-1)->y2 = SCREENY(wi, sy3);
					if( mark_inside2 && connect_errrgn && connect_id== X_set_nr- 1 ){
					  /* "Close the error region by connecting it to the last drawn point.  */
						SegsNR(LYsegs,LY_set_nr)->x2= SegsNR(Xsegs,X_seg_nr-1)->x2;
						SegsNR(LYsegs,LY_set_nr)->y2= SegsNR(Xsegs,X_seg_nr-1)->y2;
					}
					else{
					  /* The last point was not OK (clipped), so the error region will remain open	*/
						SegsNR(LYsegs,LY_set_nr)->x2= SegsNR(LYsegs,LY_set_nr)->x1;
						SegsNR(LYsegs,LY_set_nr)->y2= SegsNR(LYsegs,LY_set_nr)->y1;
					}
					LY_set_nr+= 1;
				}
				if( HYsegs && ClipWindow( wi, this_set, False, &sx6, &sy6, &sx4, &sy4, &mark_inside1, &mark_inside2, &clipcode1, &clipcode2 ) ){
					  /* Add segment to list */
					SegsNR(HYsegs,HY_set_nr)->x1= SegsNR(HYsegs,HY_set_nr-1)->x2 = SCREENX(wi, sx4);
					SegsNR(HYsegs,HY_set_nr)->y1= SegsNR(HYsegs,HY_set_nr-1)->y2 = SCREENY(wi, sy4);
					if( connect_errrgn && connect_id== X_set_nr- 1 ){
						SegsNR(HYsegs,HY_set_nr)->x2= SegsNR(Xsegs,X_seg_nr-1)->x2;
						SegsNR(HYsegs,HY_set_nr)->y2= SegsNR(Xsegs,X_seg_nr-1)->y2;
					}
					else{
						SegsNR(HYsegs,HY_set_nr)->x2= SegsNR(HYsegs,HY_set_nr)->x1;
						SegsNR(HYsegs,HY_set_nr)->y2= SegsNR(HYsegs,HY_set_nr)->y1;
					}
					HY_set_nr+= 1;
				}
			} else if( (etype<= 2 || etype== EREGION_FLAG) && this_set->ebarWidth_set ){
				if( this_set->ebarWidth<= 0.0){
				  double w2= fabs(sx2-sx1);
				  double w1= (pnt_nr)? fabs(sx1 - (sx1_1+ epbarPixels/2.0))* 2 : w2;
					epbarPixels= MIN(w1, w2);
					if( this_set->adorn_interval && (this_set->plot_interval % this_set->adorn_interval) ){
						epbarPixels*= this_set->adorn_interval;
					}
					if( !wi->eBarDimensionsSet ){
						wi->eMinBarX= sx1- epbarPixels/ 2;
						wi->eMaxBarX= sx1+ epbarPixels/ 2;
						wi->eBarDimensionsSet= 1;
					}
					else{
						wi->eMinBarX= MIN( sx1- epbarPixels/ 2, wi->eMinBarX );
						wi->eMaxBarX= MAX( sx1+ epbarPixels/ 2, wi->eMaxBarX );
					}
					ebarPixels= ( ( epbarPixels / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) + this_set->ebarWidth);
					this_set->current_ebarWidth= ebarPixels* (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)/ wi->Xscale;
					this_set->current_ebW_set= True;
					if( debugFlag && debugLevel> 2 ){
						fprintf( StdErr, "this_set->ebarWidth=%g/%g+ %g = %g pixels\n",
							epbarPixels, wi->XUnitsPerPixel* wi->dev_info.var_width_factor, this_set->ebarWidth, ebarPixels
						);
					}
				}
				else{
				  double bcx= this_set->xvec[pnt_nr],
						blx= bcx- this_set->ebarWidth/ 2,
						brx= blx+ this_set->ebarWidth,
						bty= this_set->yvec[pnt_nr];
				  int bok= 1, vf= wi->vectorFlag;
					if( !wi->eBarDimensionsSet ){
						wi->eMinBarX= blx;
						wi->eMaxBarX= brx;
						wi->eBarDimensionsSet= 1;
					}
					else{
						wi->eMinBarX= MIN( blx, wi->eMinBarX );
						wi->eMaxBarX= MAX( brx, wi->eMaxBarX );
					}
					wi->vectorFlag= 1;
					do_transform( wi /* NULL */,
						this_set->fileName, __DLINE__, "DrawData()", &bok, NULL, &bcx, &blx, &brx, &bty, NULL, NULL,
						NULL, NULL, 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
					);
					wi->vectorFlag= vf;
/* 						ebarPixels = ((this_set->ebarWidth / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) );	*/
					ebarPixels = (( fabs(brx- blx) / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) );
					this_set->current_ebarWidth= this_set->ebarWidth/ wi->Xscale;
					this_set->current_ebW_set= True;
				}
				if( ebarPixels <= 0){
					ebarPixels = 1;
				}
			} }

			  /* Handle last bar */
			if( ok && X_set_nr> 0 && AllSets[set_nr].numPoints > 0 && !discarded_point){
				if( barFlag ){
				  double barPixels;
				  int baseSpot;
				  XSegment line;
				  double lw= LINEWIDTH(wi,set_nr)* wi->dev_info.var_width_factor, bp;
				  XRectangle rec;

					if( !barBase_set && !this_set->barBase_set ){
						barBase= wi->win_geo._UsrOrgY;
						baseSpot = SCREENY(wi, barBase);
					}
					else{
						baseSpot = SCREENY(wi, this_set->barBaseY);
					}
					if( this_set->barWidth<= 0.0){
					  double w1= fabs( sx1- sx2 );
					  double w2= (pnt_nr)? fabs(sx1 - (sx1_1+ pbarPixels/2.0))* 2 : w1;
/*	 					barPixels= ( (( wi->win_geo._UsrOppX - wi->win_geo._UsrOrgX)/ (double)maxSize) /	*/
						  /* for the last bar we may take the largest of the two intervals..	*/
						pbarPixels= MAX(w1, w2);
						if( !wi->BarDimensionsSet ){
							wi->MinBarX= sx1- pbarPixels/ 2;
							wi->MaxBarX= sx1+ pbarPixels/ 2;
							wi->BarDimensionsSet= 1;
						}
						else{
							wi->MinBarX= MIN( sx1- pbarPixels/ 2, wi->MinBarX );
							wi->MaxBarX= MAX( sx1+ pbarPixels/ 2, wi->MaxBarX );
						}
						barPixels= ( ( pbarPixels / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) + this_set->barWidth);
						this_set->current_barWidth= barPixels* (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)/ wi->Xscale;
						this_set->current_bW_set= True;
						if( debugFlag && debugLevel> 1 ){
							fprintf( StdErr, "this_set->barWidth=%g, %g pixels\t", pbarPixels, barPixels );
						}
						line.x1 = line.x2 = SegsNR(Xsegs,X_seg_nr-1)->x2;
					}
					else{
					  double bcx= this_set->xvec[pnt_nr],
							blx= bcx- this_set->barWidth/ 2,
					  		brx= blx+ this_set->barWidth,
							bty= this_set->yvec[pnt_nr];
					  int bok= 1, vf= wi->vectorFlag;
						if( !wi->BarDimensionsSet ){
							wi->MinBarX= blx;
							wi->MaxBarX= brx;
							wi->BarDimensionsSet= 1;
						}
						else{
							wi->MinBarX= MIN( blx, wi->MinBarX );
							wi->MaxBarX= MAX( brx, wi->MaxBarX );
						}
						wi->vectorFlag= 1;
						do_transform( wi /* NULL */,
							this_set->fileName, __DLINE__, "DrawData()", &bok, NULL, &bcx, &blx, &brx, &bty, NULL, NULL,
							NULL, NULL, 1, pnt_nr, 1.0, 1.0, 1.0, -1, 0, 0
						);
						wi->vectorFlag= vf;
						barPixels = (( fabs(brx- blx) / (wi->XUnitsPerPixel* wi->dev_info.var_width_factor)) );
						this_set->current_barWidth= this_set->barWidth/ wi->Xscale;
						this_set->current_bW_set= True;
						line.x1 = line.x2 = SCREENX( wi, (blx+ brx)/2 );
					}
					if( barPixels <= 0)
						barPixels = 1;
					line.y1 = baseSpot;  line.y2 = SegsNR(Xsegs,X_seg_nr-1)->y2;
					  /* For a barplot, we must compensate for the linewidth in order
					   \ to get the rect in the right space/dimensions.
					   */
					if( lw> 1 ){
						  /* But I don't think I'll have to compensate for 1 pixel wide lines.	*/
						bp= barPixels* wi->dev_info.var_width_factor- lw;
					}
					else{
						lw= 0;
						bp= barPixels* wi->dev_info.var_width_factor;
					}
					rec= *rect_xywh( line.x1-bp/2,MIN(line.y1,line.y2)+ lw/2,
							bp, MAX(line.y1,line.y2)-MIN(line.y1,line.y2)- lw
					);
					switch( this_set->barType ){
#ifdef HOOK_NOT_BAR
						case 5:
						case 6:{
						  XSegment hook[2];
							hook[1].x1= hook[0].x2= hook[0].x1= (this_set->barType== 5)? line.x1- bp/2 : line.x1+ bp/2;
							hook[0].y1= line.y1;
							hook[1].y2= hook[1].y1= hook[0].y2= line.y2;
							hook[1].x2= line.x1;
							if( wi->legend_line[set_nr].highlight ){
								HighlightSegment( wi, set_nr, 2, hook, hl_width, LStyle);
							}
							wi->dev_info.xg_seg(wi->dev_info.user_state,
										2, hook,
										LINEWIDTH(wi,set_nr), LStyle,
										LINESTYLE(set_nr), PIXVALUE(set_nr), this_set);
							break;
						}
#endif
						default:
						{ int lw= LINEWIDTH(wi,set_nr);
							Draw_Bar( wi, &rec, &line, barPixels, this_set->barType, LStyle, this_set, set_nr, pnt_nr,
								lw, LINESTYLE(set_nr), lw, 0,
								minIntense, maxIntense, scaleIntense, 0, 0
							);
							break;
						}
					}
				}
				if( adorn && SegsENR(XsegsE,X_set_nr-1)->ok2 ){
					Draw_ErrorBar( wi, this_set, set_nr, pX_set_nr, X_set_nr, pnt_nr, False,
						ebarPixels, LStyle, aspect
					);
				}
			}

last_drawn:;

			if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
			  char watch[2]= { XC_target+1, '\0' };
				xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
			}

			if( AllSets[set_nr].plot_interval> 0 ){
				psSeg_disconnect_jumps= False;
			}
			else{
				  /* 20050212: somehow, psSeg_disconnect_jumps doesn't get reset after once drawing a
				   \ set with plot_interval>1. Do it here manually...
				   */
				psSeg_disconnect_jumps= True;
			}

			  /* Draw the error-regions if so requested	*/
			  /* Low side	*/
			if( AllSets[set_nr].numPoints > 0 && (LY_set_nr > 0) ){
			  double rhl_width= HL_WIDTH(AllSets[set_nr].elineWidth);
				ptr = LYsegs;
				while( LY_set_nr > wi->dev_info.max_segs ){
					CHECK_EVENT2();

					if( wi->legend_line[set_nr].highlight ){
						HighlightSegment( wi, set_nr, wi->dev_info.max_segs, ptr, rhl_width/2+ 1, LStyle);
					}
					wi->dev_info.xg_seg(wi->dev_info.user_state,
								wi->dev_info.max_segs, ptr,
								ELINEWIDTH(wi,set_nr), LStyle,
								ELINESTYLE(set_nr), PIXVALUE(set_nr), this_set
					);
					ptr += wi->dev_info.max_segs;
					LY_set_nr -= wi->dev_info.max_segs;
				}
				if( wi->legend_line[set_nr].highlight ){
					HighlightSegment( wi, set_nr, LY_set_nr, ptr, rhl_width/2+ 1, LStyle);
				}
				wi->dev_info.xg_seg(wi->dev_info.user_state,
						 LY_set_nr, ptr,
						 ELINEWIDTH(wi,set_nr), LStyle,
						 ELINESTYLE(set_nr), PIXVALUE(set_nr), this_set
				);
			}
			if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
			  char watch[2]= { XC_heart+ 1, '\0' };
				xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
			}

			  /* High side	*/
			if( AllSets[set_nr].numPoints > 0 && (HY_set_nr > 0) ){
			  double rhl_width= HL_WIDTH(AllSets[set_nr].elineWidth);
				ptr = HYsegs;
				while( HY_set_nr > wi->dev_info.max_segs ){
					CHECK_EVENT2();

					if( wi->legend_line[set_nr].highlight ){
						HighlightSegment( wi, set_nr, wi->dev_info.max_segs, ptr, rhl_width/2+ 1, LStyle);
					}
					wi->dev_info.xg_seg(wi->dev_info.user_state,
								wi->dev_info.max_segs, ptr,
								ELINEWIDTH(wi,set_nr), LStyle,
								ELINESTYLE(set_nr), PIXVALUE(set_nr), this_set
					);
					ptr += wi->dev_info.max_segs;
					HY_set_nr -= wi->dev_info.max_segs;
				}
				if( wi->legend_line[set_nr].highlight ){
					HighlightSegment( wi, set_nr, HY_set_nr, ptr, rhl_width/2+ 1, LStyle);
				}
				wi->dev_info.xg_seg(wi->dev_info.user_state,
						 HY_set_nr, ptr,
						 ELINEWIDTH(wi,set_nr), LStyle,
						 ELINESTYLE(set_nr), PIXVALUE(set_nr), this_set
				);
			}
			  /* Lastly, draw the "data-segments", and the markers
			   \ if these are to be overwritten.
			   */
			  /* 20000407: 6x X_set_nr -> X_seg_nr below	*/
			if( AllSets[set_nr].numPoints > 0 /* && (! noLines) */ && (X_seg_nr > 0) ){
			  int drawn_points= X_seg_nr;
				if( drawn_points> maxitems ){
					fprintf( StdErr, "Internal error: more points drawn (%d) than allocated (%d)\n",
						drawn_points, maxitems
					);
					drawn_points= maxitems;
				}
				ptr = Xsegs;
				if( !noLines ){
					if( debugFlag && debugLevel>= 1){
						fprintf( StdErr, "Drawing %d segments in groups of %d ", X_set_nr, wi->dev_info.max_segs);
						fflush( StdErr);
					}
					if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
					  char watch[2]= { XC_target+1, '\0' };
						xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
					}

					while( X_seg_nr > wi->dev_info.max_segs ){
						CHECK_EVENT2();

						if( wi->legend_line[set_nr].highlight ){
							HighlightSegment( wi, set_nr, wi->dev_info.max_segs, ptr, hl_width, LStyle);
						}
						wi->dev_info.xg_seg(wi->dev_info.user_state,
									wi->dev_info.max_segs, ptr,
									LINEWIDTH(wi,set_nr), LStyle,
									LINESTYLE(set_nr), PIXVALUE(set_nr), this_set);
						ptr += wi->dev_info.max_segs;
						X_seg_nr -= wi->dev_info.max_segs;
						if( debugFlag && debugLevel>= 1){
							fputc( '.', StdErr);
							fflush( StdErr);
						}
					}
					if( wi->legend_line[set_nr].highlight ){
						HighlightSegment( wi, set_nr, X_seg_nr, ptr, hl_width, LStyle);
					}
					wi->dev_info.xg_seg(wi->dev_info.user_state,
							 X_seg_nr, ptr,
							 LINEWIDTH(wi,set_nr), LStyle,
							 LINESTYLE(set_nr), PIXVALUE(set_nr), this_set);

					  /* 20000505: the code for calculating&drawing the start and end-arrows
					   \ can move to a single routine that will also do the arrows for the XGPens.
					   \ Take care about drawn_points!
					   */
					if( AllSets[set_nr].arrows & 0x01 ){
					  int idx= 0;
					  XSegment *arr;
					  double x2, y2, x2_1, y2_1,
							X2, Y2, X2_1, Y2_1,
							ang;
						  /* Find the first segment with non-zero length of this trace.
						   \ If no such segment is found, we won't draw an arrowhead, since
						   \ there is actually no trace (just a point).
						   */
						do{
							x2= SegsNR(Xsegs,idx)->x1;
							y2= SegsNR(Xsegs,idx)->y1;
							x2_1= SegsNR(Xsegs,idx)->x2;
							y2_1= SegsNR(Xsegs,idx)->y2;
							X2= XSegsNR(XXsegs,idx)->x1;
							Y2= XSegsNR(XXsegs,idx)->y1;
							X2_1= XSegsNR(XXsegs,idx)->x2;
							Y2_1= XSegsNR(XXsegs,idx)->y2;
							idx++;
						}
						while( !AllSets[set_nr].sarrow_orn_set && X2== X2_1 && Y2== Y2_1 && idx< drawn_points );
						if( AllSets[set_nr].sarrow_orn_set || X2!= X2_1 || Y2!= Y2_1 ){
							  /* Determine the 2 other points of the arrowhead, assuming the
							   \ trace's last segment was horizontal. Screen-coordinates are used for this,
							   \ to prevent having to do all the possible transformations/clippings again:
							   */
							if( AllSets[set_nr].sarrow_orn_set ){
								ang= AllSets[set_nr].sarrow_orn;
							}
							else{
								  /* Now determine the orientation of the trace's last segment, and
								   \ rotate the arrowhead around its point (x2,y2) to put it in the
								   \ same orientation. For this we make use of world-coordinates/orientation,
								   \ to ensure that a given user-specified orientation will work regardless
								   \ of e.g. the window-aspect-ratio.
								   */
								AllSets[set_nr].sarrow_orn= ang= degrees(atan3( X2_1- X2, Y2_1- Y2 ));
							}
#if defined(linux) && defined(__GNUC__)
/* 							make_arrow_point1( wi, this_set, x2, y2, ang, -wi->dev_info.tick_len, aspect );	*/
#endif
							if( (arr= make_arrow_point1( wi, this_set, x2, y2, ang, -wi->dev_info.tick_len, aspect )) ){
								if( wi->legend_line[set_nr].highlight ){
									HighlightSegment( wi, set_nr, 2, arr, hl_width, LStyle);
								}
								wi->dev_info.xg_seg(wi->dev_info.user_state,
											2, arr,
											LINEWIDTH(wi,set_nr), LStyle,
											LINESTYLE(set_nr), PIXVALUE(set_nr), this_set);
							}
						}
					}
					if( AllSets[set_nr].arrows & 0x02 ){
					  int idx= drawn_points-1;
					  XSegment *arr;
					  double x2, y2, x2_1, y2_1,
							X2, Y2, X2_1, Y2_1,
							ang;
						do{
							x2= SegsNR(Xsegs,idx)->x2;
							y2= SegsNR(Xsegs,idx)->y2;
							x2_1= SegsNR(Xsegs,idx)->x1;
							y2_1= SegsNR(Xsegs,idx)->y1;
							X2= XSegsNR(XXsegs,idx)->x2;
							Y2= XSegsNR(XXsegs,idx)->y2;
							X2_1= XSegsNR(XXsegs,idx)->x1;
							Y2_1= XSegsNR(XXsegs,idx)->y1;
							idx--;
						}
						while( !AllSets[set_nr].earrow_orn_set && X2== X2_1 && Y2== Y2_1 && idx>= 0 );
						if( AllSets[set_nr].earrow_orn_set || X2!= X2_1 || Y2!= Y2_1 ){
							if( AllSets[set_nr].earrow_orn_set ){
								ang= AllSets[set_nr].earrow_orn;
							}
							else{
								AllSets[set_nr].earrow_orn= ang= degrees( atan3( X2- X2_1, Y2- Y2_1 ) );
							}
#if defined(linux) && defined(__GNUC__)
/* 							make_arrow_point1( wi, this_set, x2, y2, ang, wi->dev_info.tick_len, aspect );	*/
#endif
							if( (arr= make_arrow_point1( wi, this_set, x2, y2, ang, wi->dev_info.tick_len, aspect )) ){
								if( wi->legend_line[set_nr].highlight ){
									HighlightSegment( wi, set_nr, 2, arr, hl_width, LStyle);
								}
								wi->dev_info.xg_seg(wi->dev_info.user_state,
											2, arr,
											LINEWIDTH(wi,set_nr), LStyle,
											LINESTYLE(set_nr), PIXVALUE(set_nr), this_set);
							}
						}
					}

					if( debugFlag && debugLevel>= 1){
						fputs( ".\n", StdErr);
						fflush( StdErr);
					}
				}
				if( markFlag && AllSets[set_nr].overwrite_marks ){
					if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
					  char watch[2]= { XC_heart+ 1, '\0' };
						xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
					}

					  /* 20000407: the X_seg_nr range can be a subset of the X_set_nr range. When <X_seg_nr_diff> points are
					   \ deleted from the start of the set, X_seg_nr=0 will correspond to X_set_nr=X_seg_nr_diff (rather,
					   \ X_seg_nr_diff will be set to this potential difference, or 0 if there is none). When points
					   \ are deleted in the middle, X_set_nr increases for these points whereas X_seg_nr doesn't. That is,
					   \ the deleted points are stored in XsegsE referenced by X_set_nr, but not in Xsegs referenced by
					   \ X_seg_nr. This means X_seg_nr must not be increased for deleted points..
					   */
					for( X_seg_nr= 0, X_set_nr= X_seg_nr_diff; X_seg_nr< drawn_points && X_set_nr< maxitems; X_set_nr++ ){
						  /* Check if this point was marked inside, and if
						   \ not one of the "real" coordinates is Inf/NaN
						   \ (otherwise, a line from Inf to inside gets a
						   \ mark on the bounding box - somehow.)
						   */
						if( SegsENR(XsegsE,X_set_nr)->mark1 &&
							!(NaNorInf(SegsENR(XsegsE,X_set_nr)->rx1) || NaNorInf(SegsENR(XsegsE,X_set_nr)->ry1))
						){
						  Pixel colorx;
						  int respix= False;
						  double ms= this_set->markSize;
						  int x, y;
							if( etype== INTENSE_FLAG && IntensityColourFunction.XColours ){
								Retrieve_IntensityColour( wi, this_set, this_set->errvec[SegsENR(XsegsE,X_set_nr)->pnt_nr1],
									minIntense, maxIntense, scaleIntense, &colorx, &respix
								);
								psThisRGB= xg_IntRGB;
							}
							if( etype== MSIZE_FLAG ){
								this_set->markSize= -fabs( this_set->errvec[SegsENR(XsegsE,X_set_nr)->pnt_nr1] *
									((NaNorInf(ms))? 1 : ms) );
							}
							  /* 20041002: re-disabled the X_seg_nr check */
							if( /* !X_seg_nr || */ SegsENR(XsegsE, X_set_nr)->use_first ){
								x= SegsNR(Xsegs,X_seg_nr)->x1;
								y= SegsNR(Xsegs,X_seg_nr)->y1;
							}
							else{
								x= SegsNR(Xsegs,X_seg_nr-1)->x2;
								y= SegsNR(Xsegs,X_seg_nr-1)->y2;
							}
							switch( pixelMarks ){
								case 2:
									wi->dev_info.xg_dot(wi->dev_info.user_state,
												x, y,
												P_DOT, 0, set_nr % MAXATTR, 0, set_nr, this_set);
									break;
								case 1:
									wi->dev_info.xg_dot(wi->dev_info.user_state,
												x, y,
												P_PIXEL, 0, PIXVALUE(set_nr), set_nr, this_set);
									break;
								case 0:
									/* Distinctive markers */
									wi->dev_info.xg_dot(wi->dev_info.user_state,
											x, y,
											P_MARK, MARKSTYLE(set_nr),
											PIXVALUE(set_nr), set_nr, this_set);
									break;
							}
							if( respix ){
								if( this_set->pixvalue< 0 ){
									this_set->pixelValue= colorx;
								}
								else{
									AllAttrs[this_set->pixvalue].pixelValue= colorx;
								}
							}
							this_set->markSize= ms;
						}
						  /* Logically, (though not to me it was at first!), a segment can have
						   \ two markers..... and there I was, wondering why the last point didn't
						   \ receive its overwrite-mark ;-)
						   */
						if( SegsENR(XsegsE,X_set_nr)->mark2 &&
							!(NaNorInf(SegsENR(XsegsE,X_set_nr)->rx2) || NaNorInf(SegsENR(XsegsE,X_set_nr)->ry2))
						){
						  Pixel colorx;
						  int respix= False;
						  double ms= this_set->markSize;
							if( etype== INTENSE_FLAG && IntensityColourFunction.XColours ){
								Retrieve_IntensityColour( wi, this_set, this_set->errvec[SegsENR(XsegsE,X_set_nr)->pnt_nr2],
									minIntense, maxIntense, scaleIntense, &colorx, &respix
								);
								psThisRGB= xg_IntRGB;
							}
							if( etype== MSIZE_FLAG ){
								this_set->markSize= -fabs( this_set->errvec[SegsENR(XsegsE,X_set_nr)->pnt_nr2] *
									((NaNorInf(ms))? 1 : ms) );
							}
							switch( pixelMarks ){
								case 2:
									wi->dev_info.xg_dot(wi->dev_info.user_state,
												SegsNR(Xsegs,X_seg_nr)->x2, SegsNR(Xsegs,X_seg_nr)->y2,
												P_DOT, 0, set_nr % MAXATTR, 0, set_nr, this_set);
									break;
								case 1:
									wi->dev_info.xg_dot(wi->dev_info.user_state,
												SegsNR(Xsegs,X_seg_nr)->x2, SegsNR(Xsegs,X_seg_nr)->y2,
												P_PIXEL, 0, PIXVALUE(set_nr), set_nr, this_set);
									break;
								case 0:
									/* Distinctive markers */
									wi->dev_info.xg_dot(wi->dev_info.user_state,
											SegsNR(Xsegs,X_seg_nr)->x2, SegsNR(Xsegs,X_seg_nr)->y2,
											P_MARK, MARKSTYLE(set_nr),
											PIXVALUE(set_nr), set_nr, this_set);
									break;
							}
							if( respix ){
								if( this_set->pixvalue< 0 ){
									this_set->pixelValue= colorx;
								}
								else{
									AllAttrs[this_set->pixvalue].pixelValue= colorx;
								}
							}
							this_set->markSize= ms;
						}
/* 						SegsENR(XsegsE,X_set_nr)->mark1= False;	*/
/* 						SegsENR(XsegsE,X_set_nr)->mark2= False;	*/
						if( !DiscardedPoint( wi, this_set, SegsENR(XsegsE,X_set_nr)->pnt_nr1 ) ){
							X_seg_nr+= 1;
						}
					}
				}

				if( CheckMask(this_set->valueMarks, VMARK_ON) && AllSets[set_nr].overwrite_marks ){
					if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
					  char watch[2]= { XC_heart+ 1, '\0' };
						xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
					}

					for( X_seg_nr= 0, X_set_nr= X_seg_nr_diff; X_seg_nr< drawn_points && X_set_nr< maxitems; X_set_nr++ ){
						  /* Check if this point was marked inside, and if
						   \ not one of the "real" coordinates is Inf/NaN
						   \ (otherwise, a line from Inf to inside gets a
						   \ mark on the bounding box - somehow.)
						   */
						if( SegsENR(XsegsE,X_set_nr)->mark1 &&
							!(NaNorInf(SegsENR(XsegsE,X_set_nr)->rx1) || NaNorInf(SegsENR(XsegsE,X_set_nr)->ry1))
						){
						  int x, y;
							  /* 20041002: re-disabled the X_seg_nr check */
							if( /* !X_seg_nr || */ SegsENR(XsegsE, X_set_nr)->use_first ){
								x= SegsNR(Xsegs,X_seg_nr)->x1;
								y= SegsNR(Xsegs,X_seg_nr)->y1;
							}
							else{
								x= SegsNR(Xsegs,X_seg_nr-1)->x2;
								y= SegsNR(Xsegs,X_seg_nr-1)->y2;
							}
								Draw_valueMark( wi, this_set, SegsENR(XsegsE,X_set_nr)->pnt_nr1,
									x, y, PIXVALUE( set_nr )
								);
						}
						if( SegsENR(XsegsE,X_set_nr)->mark2 &&
							!(NaNorInf(SegsENR(XsegsE,X_set_nr)->rx2) || NaNorInf(SegsENR(XsegsE,X_set_nr)->ry2))
						){
								Draw_valueMark( wi, this_set, SegsENR(XsegsE,X_set_nr)->pnt_nr2,
									SegsNR(Xsegs,X_seg_nr)->x2, SegsNR(Xsegs,X_seg_nr)->y2,
									PIXVALUE( set_nr )
								);
						}
/* 						SegsENR(XsegsE,X_set_nr)->mark1= False;	*/
/* 						SegsENR(XsegsE,X_set_nr)->mark2= False;	*/
						if( !DiscardedPoint( wi, this_set, SegsENR(XsegsE,X_set_nr)->pnt_nr1 ) ){
							X_seg_nr+= 1;
						}
					}
				}
				  /* Clear the mark1 and mark2 fields	*/
				if( AllSets[set_nr].overwrite_marks ){
					if( !wi->animate && wi->settings_frame.mapped && cursorFont.font ){
					  char watch[2]= { XC_heart+ 1, '\0' };
						xtb_bt_set_text( wi->settings, xtb_bt_get(wi->settings, NULL), watch, (xtb_data) 0);
					}

					for( X_seg_nr= 0, X_set_nr= X_seg_nr_diff; X_seg_nr< drawn_points && X_set_nr< maxitems; X_set_nr++ ){
						SegsENR(XsegsE,X_set_nr)->mark1= False;
						SegsENR(XsegsE,X_set_nr)->mark2= False;
						if( !DiscardedPoint( wi, this_set, SegsENR(XsegsE,X_set_nr)->pnt_nr1 ) ){
							X_seg_nr+= 1;
						}
					}
				}

			}
			if( debugLevel>= 1){
				fprintf( StdErr, "Next set: %d\n", DO(cnr+1) );
				fflush( StdErr);
			}
			if( bounds_clip ){
				this_set->s_bounds_set= -1;
			}

			wi->numDrawn+= 1;
		}

		if( wi->pen_list && !wi->no_pens && !wi->fitting ){
		  XGPen *Pen= wi->pen_list;
			while( Pen ){
				if( !PENSKIP(wi,Pen) && Pen->after_set== set_nr ){
					DrawPen( wi, Pen );
				}
				Pen= Pen->next;
			}
		}
    }

	if( wi->pen_list && !wi->no_pens && !wi->fitting ){
	  XGPen *Pen= wi->pen_list;
		while( Pen ){
			if( Pen->overwrite_pen && !PENSKIP(wi,Pen) && Pen->before_set<0 && Pen->after_set< 0 ){
				DrawPen( wi, Pen );
			}
			Pen= Pen->next;
		}
	}

	win_geo= wi->win_geo;
	wi->event_level--;
	RESET_WIN_STATE(wi);
DrawData_return:;
	xfree( set_O.sample);
	xfree( set_tr_O.sample);
	xfree( SAS_er.sample);
	xfree( SAS_er_r.sample);
	return;
}

/* #define NUMSETS	MAXSETS	*/

#define NUMSETS	setNumber

extern int StringCheck(char *, int, char *, int);
#define STRINGCHECK(s,max)	StringCheck(s,max,__FILE__,__LINE__)


#define RND(val)	((int) ((val) + 0.5))

#ifdef USE_TILDEEXPAND

extern char *_tildeExpand(char *out, char *in);
#endif

/* 20000509: Check this window to see if it has any processes defined that depend on variables
 \ that were (inadvertently..?) deleted. As of this writing, recompiling these expressions
 \ (processes) will automatically re-define those variables. Redefining a variable with the
 \ same name etc. currently re-uses the same ascanf_Function structure (these don't get
 \ deleted to simplify matters..), so that in theory, other expressions referring to these
 \ variables (directly or via pointers) remain valid.
 */
int Check_Process_Dependencies( LocalWin *wi )
{ int n= 0, m, M= 0, i;
  char asep = ascanf_separator;
	if( !wi || wi->delete_it== -1 ){
		return(0);
	}
	if( wi->transform.separator ){
		ascanf_separator = wi->transform.separator;
	}
	if( !transform_x_buf && !transform_x_buf_len ){
		if( (m= Check_Form_Dependencies( &wi->transform.C_x_process )) ){
			n+= 1;
			M+= m;
			new_transform_x_process( wi);
		}
	}
	if( !transform_y_buf && !transform_y_buf_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->transform.C_y_process )) ){
			n+= 1;
			M+= m;
			new_transform_y_process( wi);
		}
	}
	if( wi->process.separator ){
		ascanf_separator = wi->process.separator;
	}
	else{
		ascanf_separator = asep;
	}
	if( !ReadData_proc.data_init_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_data_init)) ){
			n+= 1;
			M+= m;
			new_process_data_init( wi );
		}
	}
	if( ReadData_proc.data_before_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_data_before)) ){
			n+= 1;
			M+= m;
			new_process_data_before( wi );
		}
	}
	if( ReadData_proc.data_process_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_data_process)) ){
			n+= 1;
			M+= m;
			new_process_data_process( wi );
		}
	}
	if( ReadData_proc.data_after_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_data_after)) ){
			n+= 1;
			M+= m;
			new_process_data_after( wi );
		}
	}
	if( ReadData_proc.data_finish_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_data_finish)) ){
			n+= 1;
			M+= m;
			new_process_data_finish( wi );
		}
	}

	if( !ReadData_proc.draw_before_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_draw_before)) ){
			n+= 1;
			M+= m;
			new_process_draw_before( wi );
		}
	}
	if( !ReadData_proc.draw_after_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_draw_after)) ){
			n+= 1;
			M+= m;
			new_process_draw_after( wi );
		}
	}

	if( !ReadData_proc.dump_before_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_dump_before)) ){
			n+= 1;
			M+= m;
			new_process_dump_before( wi );
		}
	}
	if( !ReadData_proc.dump_after_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_dump_after)) ){
			n+= 1;
			M+= m;
			new_process_dump_after( wi );
		}
	}

	if( !ReadData_proc.enter_raw_after_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_enter_raw_after)) ){
			n+= 1;
			M+= m;
			new_process_enter_raw_after( wi );
		}
	}
	if( !ReadData_proc.leave_raw_after_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->process.C_leave_raw_after)) ){
			n+= 1;
			M+= m;
			new_process_leave_raw_after( wi );
		}
	}
	if( wi->curs_cross.fromwin_process.separator ){
		ascanf_separator = wi->curs_cross.fromwin_process.separator;
	}
	else{
		ascanf_separator = asep;
	}
	if( wi->curs_cross.fromwin_process.process_len && wi->delete_it!= -1 ){
		if( (m= Check_Form_Dependencies( &wi->curs_cross.fromwin_process.C_process)) ){
		  LocalWin *aw= ActiveWin;
			n+= 1;
			M+= m;
			ActiveWin= wi->curs_cross.fromwin;
			ascanf_window= ActiveWin->window;
			new_process_Cross_fromwin_process( wi );
			ActiveWin= aw;
		}
	}

	for( i= 0; i< setNumber && wi->delete_it!= -1; i++ ){
		if( draw_set(wi,i) && (m= Check_Form_Dependencies( &(AllSets[i].process.C_set_process) )) ){
			if( AllSets[i].process.separator ){
				ascanf_separator = AllSets[i].process.separator;
			}
			else{
				ascanf_separator = asep;
			}
			n+= 1;
			M+= m;
			new_process_set_process( wi, &AllSets[i] );
		}
	}
	if( debugFlag ){
		fprintf( StdErr,
			"Check_Process_Dependencies(): %d processes depended on %d deleted variables%s.\n",
			n, M,
			(n>0)? ": recompiled them to redefine the variables" : ""
		);
	}
	ascanf_separator = asep;
	return(n);
}

#define RESET_SETUNDO(undobuf,Set)	{if((undobuf).set==(Set)) (undobuf).set=NULL;}

void Reset_SetUndo( DataSet *set )
{
	RESET_SETUNDO(ShiftUndo,set);
	RESET_SETUNDO(SplitUndo,set);
	RESET_SETUNDO(DiscardUndo,set);
}
