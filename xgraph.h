#ifndef _XGRAPH_H
#define _XGRAPH_H

/*
 * Globally accessible information from xgraph
 */

#ifdef __STDC__
#	include <stddef.h>
#	include <stdlib.h>
#	if !defined(__MACH__) && !defined(__CYGWIN__)
#		include <values.h>
#	endif
#endif

#define MAXKEYS		50
#define MAXATTR 	16
#define MAXSETS		MaxSets
#define	ERROR_TYPES	8

#define MAXBUFSIZE 	512

#ifndef STR
#	define STR(name)	# name
#endif

#ifndef ALLOCA
#	pragma GCC dependency "xgALLOCA.h"
#	include "xgALLOCA.h"
#endif

#pragma GCC dependency "SS.h"
#include "SS.h"

#define MAXLS		11

#define STRDUP(xx)	(strcpy(malloc((unsigned) (strlen(xx)+1)), (xx)))

#define xg_abort()	{fputs("abort\n",stderr);exit(-10);}
#ifndef DEBUG
#	define abort()	xg_abort()
#endif

extern int MaxSets;
#pragma GCC dependency "DataSet.h"
#include "DataSet.h"
extern DataSet *AllSets;

typedef struct attr_set {
    char lineStyle[MAXLS];
    int lineStyleLen;
    Pixel pixelValue;
	char *pixelCName;
    Pixmap markStyle;
} AttrSet;    

extern AttrSet _AllAttrs[MAXATTR+1], *AllAttrs;

#define BARTYPES 7

#pragma GCC dependency "Elapsed.h"
#include "Elapsed.h"

#pragma GCC dependency "xtb/xtb.h"
#include "xtb/xtb.h"

/* struct Compiled_Form;	*/

typedef struct XGStringList{
	char *text, separator;
	long long hash;
	struct XGStringList *last, *next;
} XGStringList;

typedef struct LabelsList{
	unsigned short column, shown;
	unsigned short min, max;
	char *label;
} LabelsList;

typedef struct GenericProcess{
	char *process;
	  /* current length of the process string, and its allocation length. The allocation
	   \ length is only important when editing via the settings dialog is (to be) possible.
	   */
	int process_len, process_allen;
	struct Compiled_Form *C_process;
	char *description,
	  /* command: the "*...*" command/opcode this process is referred to in dumps. */
		command[64], separator;
} GenericProcess;

typedef struct Process{
	char *data_init, *data_before, *data_process, *data_after, *data_finish;
	int data_process_now,
		data_init_len, data_before_len, data_process_len, data_after_len, data_finish_len,
		data_init_printed, data_before_printed, data_after_printed, data_finish_printed;
	struct Compiled_Form *C_data_init, *C_data_before, *C_data_process, *C_data_after, *C_data_finish;
	char *param_before, *param_functions, *param_after;
	int param_before_len, param_functions_len, param_after_len,
		param_before_printed, param_after_printed;
	struct Compiled_Form *C_param_before, *C_param_functions, *C_param_after;
	double param_range[3];
	char *draw_before, *draw_after;
	int draw_before_len, draw_after_len,
		draw_before_printed, draw_after_printed;
	struct Compiled_Form *C_draw_before, *C_draw_after;
	char *dump_before, *dump_after;
	int dump_before_len, dump_after_len,
		dump_before_printed, dump_after_printed;
	struct Compiled_Form *C_dump_before, *C_dump_after;
	char *enter_raw_after, *leave_raw_after;
	int enter_raw_after_len, leave_raw_after_len,
		enter_raw_after_printed, leave_raw_after_printed;
	struct Compiled_Form *C_enter_raw_after, *C_leave_raw_after;
	  /* allocation sizes:	*/
	int param_before_allen, param_functions_allen, param_after_allen,
		data_init_allen, data_before_allen, data_process_allen, data_after_allen, data_finish_allen,
		draw_before_allen, draw_after_allen, dump_before_allen, dump_after_allen,
		enter_raw_after_allen, leave_raw_after_allen;
	char *description, separator;
} Process;

typedef struct Transform{
	char *x_process, *y_process;
	int x_len, y_len,
		x_allen, y_allen;
	struct Compiled_Form *C_x_process, *C_y_process;
	char *description, separator;
} Transform;

typedef struct RGB{
	unsigned short red, green, blue;
} RGB;

typedef struct ColourFunction{
	char *expression;
	struct Compiled_Form *C_expression;
	XGStringList *name_table;
	int NColours, range_set, use_table, last_read;
	XColor *XColours;
	RGB *exactRGB;
	Range range;
	char *description;
} ColourFunction;
extern ColourFunction IntensityColourFunction;
#define INTENSE_FLAG	5
#define MSIZE_FLAG		6
#define EREGION_FLAG	7

typedef enum { UL_regular=0, UL_hline, UL_vline, UL_types } ULabelTypes;

extern char *ULabelTypeNames[UL_types+1];

typedef struct UserLabel{
	ULabelTypes type;
	double x1, y1, x2, y2;
	double eval;
	double tx1, ty1, tx2, ty2;
	  /* This contains all window-coordinates:	*/
	XSegment line;
	char label[MAXBUFSIZE+1], *labelbuf;
	short set_link, pnt_nr, nobox, short_flag, right_aligned, do_transform, vertical, draw_it, do_draw, free_buf;
	XSegment box[5];
	int rt_added, pixvalue, pixlinked;
	double *old2, lineWidth;
	Pixel pixelValue;
	char *pixelCName;
	struct UserLabel *next;
} UserLabel;

#define	MAXAXVAL_LEN	128

/* structure for category axes (instead of just plain numbers). Currently (990708),
 \ only a value->category is defined where a given value is associated with a give
 \ textual label (the category). This could evolve into different other types of
 \ mappings, which may or may not use the same or different structures...
 \ The structure of ValCategory contains the value, the range of values of the
 \ elements following this one (to implement a fast search; the last element is
 \ designated by min==max==val), and a pointer to the label (of which MAXAXVAL_LEN
 \ chars are used maximum).
 */
typedef struct ValCategory{
	double val, min, max;
	char *category, vcat_str[MAXAXVAL_LEN];
	unsigned int idx, N, print_len;
} ValCategory;

extern ValCategory *VCat_X, *VCat_Y, *VCat_I;
DEFUN( *Find_ValCat, (ValCategory *vcat, double value, ValCategory **low, ValCategory **high), ValCategory);
DEFUN( *Add_ValCat, (ValCategory *current_VCat, int *current_N, double value, char *category), ValCategory);
DEFUN( *Free_ValCat, (ValCategory *vcat), ValCategory);
DEFUN( ValCat_N, (ValCategory *vcat), int);

typedef struct LegendInfo{
	double _legend_ulx, _legend_uly;			/* coordinates of the legend-box	*/
	int legend_ulx, legend_uly, legend_lry,
		legend_lrx, legend_frx, legend_width,
		legend_height, legend_type;
	int legend_placed, legend_always_visible;	/* flag for user-placed legend-box	*/
	int legend_trans, legend_needed;			/* do_transform these coordinates?	*/
} LegendInfo;

typedef struct Cursor_Cross{
	XSegment line[2];
	GC gc;
	int had_focus;
	short label_x, label_y;
	char OldLabel[256];
	int OldLabel_len;
	GenericProcess fromwin_process;
	struct LocalWin *fromwin;
} Cursor_Cross;

#pragma GCC dependency "XGPen.h"
#include "XGPen.h"

/* A structure that stores several fields having to do with text-related
 \ operations.
 */
typedef struct TextRelated{
	  /* Used for the use_gsTextWidth option:	*/
	int used_gsTextWidth, prev_used_gsTextWidth;
	unsigned long NgsTW;
	double SgsTW;
	int gs_batch, gs_batch_items;
	char gs_fn[128];
} TextRelated;

#define RAW_DISPLAY(wi)	((wi->raw_display && wi->raw_once>= 0) || (wi->raw_once< 0 && wi->raw_val))

#pragma GCC dependency "LocalWin.h"
#include "LocalWin.h"

typedef struct LegendDimensions{
	double first_markerHeight, markerWidth;
	int maxFontWidth, maxName, maxFile, overlap_size;
} LegendDimensions;

#define LEG2_LINE_LENGTH(wi,mw)	8*(mw)+ (wi)->dev_info.bdr_pad

typedef struct FilterUndo{
	FILE *fp;
} FilterUndo;
extern FilterUndo BoxFilterUndo;

typedef enum BoxFilters {
	BOX_FILTER_INIT=0, BOX_FILTER_PROCESS, BOX_FILTER_AFTER, BOX_FILTER_FINISH, BOX_FILTER_CLEANUP, BOX_FILTERS
} BoxFilters;

extern XGStringList *Init_Exprs, *Startup_Exprs, *Dump_commands, *DumpProcessed_commands;
extern XGStringList *XGStringList_Delete( XGStringList *list), *XGStringList_AddItem( XGStringList *list, char *text);

#define PW_PARENT		0
#define PW_MOUSE		1
#define PW_CENTRE_ON	2
extern int pw_centre_on_X, pw_centre_on_Y;

#if defined(__APPLE_CC__) || defined(__MACH__)
#	define maxSize	XGmaxSize
#endif

extern int setNumber, maxSize;

/* Globally accessible values */
extern Display *disp;			/* Open display            */
extern Visual *vis;			/* Standard visual         */
extern Colormap cmap;			/* Standard colormap       */
extern int screen;			/* Screen number           */
extern int depth;			/* Depth of screen         */
extern int install_flag;		/* Install colormaps       */
extern Pixel black_pixel;		/* Actual black pixel      */
extern Pixel white_pixel;		/* Actual white pixel      */
extern Pixel bgPixel;			/* Background color        */
extern int bdrSize;			/* Width of border         */
extern Pixel bdrPixel;			/* Border color            */
extern Pixel zeroPixel;			/* Zero grid color         */
extern Pixel axisPixel, gridPixel;
extern Pixel highlightPixel;
extern double zeroWidth;			/* Width of zero line      */
extern char zeroLS[MAXLS];		/* Line style spec         */
extern int zeroLSLen;			/* Length of zero LS spec  */
extern Pixel normPixel, textPixel;			/* Norm grid color         */
extern int use_textPixel;
extern char *blackCName;
extern char *whiteCName;
extern char *bgCName;
extern char *normCName;
extern char *bdrCName;
extern char *zeroCName;
extern char *axisCName, *gridCName;
extern char *highlightCName;
extern double axisWidth, gridWidth;			/* Width of axis line      */
extern double errorWidth;
extern char gridLS[MAXLS];		/* grid line style spec    */
extern int gridLSLen;			/* Length of grid line style */
extern Pixel echoPix;			/* Echo pixel value        */
extern XGFontStruct dialogFont;
extern XGFontStruct dialog_greekFont;
extern XGFontStruct axisFont;		/* Font for axis labels    */
extern XGFontStruct legendFont;
extern XGFontStruct labelFont;
extern XGFontStruct legend_greekFont;
extern XGFontStruct label_greekFont;
extern XGFontStruct title_greekFont;
extern XGFontStruct axis_greekFont;
extern XGFontStruct titleFont;		/* Font for title          */
extern XGFontStruct cursorFont;
extern XGFontStruct fbFont, fb_greekFont;
extern XGFontStruct markFont;
extern char titleText[MAXBUFSIZE+1]; 	/* Plot title              */
extern char XUnits[MAXBUFSIZE+1];		/* X Unit string           */
extern char YUnits[MAXBUFSIZE+1];		/* Y Unit string	   */
extern char tr_XUnits[MAXBUFSIZE+1];		/* X Unit string           */
extern char tr_YUnits[MAXBUFSIZE+1];		/* Y Unit string	   */

#define bwFlag		(_bwFlag || MonoChrome)
extern int _bwFlag;			/* Black and white flag    */
extern int MonoChrome;

extern int htickFlag, vtickFlag;	/* Don't draw grid    */
extern int bbFlag;			/* Whether to draw bb      */
extern int noLines;			/* Don't draw lines        */
extern int markFlag;			/* Draw marks at points    */
extern int pixelMarks;			/* Draw pixel markers      */
extern int logXFlag;			/* Logarithmic X axis      */
extern int logYFlag;			/* Logarithmic Y axis      */
extern int barFlag;			/* Draw bar graph          */
extern double barBase, barWidth;	/* Base and width of bars  */
extern double lineWidth;			/* Width of data lines     */
extern char *geoSpec;			/* Geometry specification  */
extern int numFiles;			/* Number of input files   */
extern char **inFileNames;	 	/* File names              */
extern char *Odevice;			/* Output device   	   */
extern char *Odisp; 			/* Output disposition      */
extern char *OfileDev;			/* Output file or device   */
extern double Odim;			/* Output dimension        */
extern char *Otfam;			/* Output title family     */
extern double Otsize;			/* Output title size       */
extern char *Oafam;			/* Output axis family      */
extern double Oasize;			/* Output axis size        */
extern int debugFlag,			/* Whether debugging is on */
	debugLevel;
extern int QuietErrors, quiet_error_count;
extern int RemoteConnection;

extern unsigned int dot_w, dot_h;	/* Size of a dot marker    */
extern unsigned int mark_w, mark_h;	/* Size of a style marker  */
extern int mark_cx, mark_cy;	/* Center of style marker  */

extern Pixmap dotMap;		/* Large dot bitmap        */

extern int do_hardcopy();	/* Carries out hardcopy    */
extern int ho_dialog();		/* Hardcopy dialog         */
extern int settings_dialog();
extern void Set_X();		/* Initializes X device    */

extern char *XLABEL(LocalWin *wi);
extern char *YLABEL(LocalWin *wi);

extern char *strcpalloc( char **dest, int *destlen, const char *src );

/* To make lint happy */
#ifndef __STDC__
#	if !defined(_HPUX_SOURCE) && !defined(_APOLLO_SOURCE)
		extern char *sprintf();
		extern char *strcpy();
		extern char *strcat();
		extern char *malloc(), *calloc();
		extern char *realloc();
#	endif
#endif
extern void exit();
extern void free();

#define INIT_XUNITS	"X\0\b\n\r\t"
#define INIT_YUNITS	"Y\0\b\n\r\t"

  /* estimate of the WM's titlebar height	*/
extern int WM_TBAR;

extern Boolean xg_text_highlight;
extern int xg_text_highlight_colour;

  /* Versions that won't crash on NULLpointers.
   \ return -1 when !s1 &&  s2
   \         1 when  s1 && !s2
   \         0 when !s1 && !s2  (!!)
   \         else call the corresponding libc routine
   */
extern int XGstrcmp     (const char *s1, const char *s2);
extern int XGstrncmp (const  char *s1, const     char *s2, size_t n);
extern int XGstrcasecmp (const char    *s1, const char     *s2);
extern int XGstrncasecmp (const char *s1, const char *s2, size_t n);

#define GETCOLOR_USETHIS	((void*)-1)
extern XColor GetThisColor;
extern double *AllowGammaCorrection;
extern int GetColor( char *name, Pixel *pix);
extern void FreeColor( Pixel *pix, char **pixCName );

#ifdef strcmp
#	undef strcmp
#endif
#define strcmp XGstrcmp
#ifdef strncmp
#	undef strncmp
#endif
#define strncmp XGstrncmp
#define strcasecmp XGstrcasecmp
#define strncasecmp XGstrncasecmp

#ifndef strdup
	extern char *strdup();
#endif

extern char *Error_TypeNames[ERROR_TYPES];
extern char *d2str( double, const char*, char*), *ad2str( double, const char*, char**);

#ifndef CLIP
#	define CLIP(var,low,high)	if((var)<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif
#ifndef CLIP_EXPR
#	define CLIP_EXPR(var,expr,low,high)	if(((var)=(expr))<(low)){\
		(var)=(low);\
	}else if((var)>(high)){\
		(var)=(high);}
#endif
#ifndef CLIP_EXPR_CAST
/* A safe casting/clipping macro.	*/
#	define CLIP_EXPR_CAST(ttype,var,stype,expr,low,high)	{stype clip_expr_cast_lvalue=(expr); if(clip_expr_cast_lvalue<(low)){\
		(var)=(ttype)(low);\
	}else if(clip_expr_cast_lvalue>(high)){\
		(var)=(ttype)(high);\
	}else{\
		(var)=(ttype)clip_expr_cast_lvalue;}}
#endif

#ifndef Conditional_Toggle
#	define Conditional_Toggle(option)	if( Opt01== -1 ){\
		option= ! option;\
	}\
	else{\
		option= Opt01;\
	}
#endif

extern int EndianType, SwapEndian;

#endif
