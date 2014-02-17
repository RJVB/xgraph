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

 \ 961125: xgInput this module is more or less involved with input-handling: ReadData c.s. & ParseArgs c.s.
 \ 20020506: xgInput became too crowded; created ReadData.c that contains the actual reading/parsing routines.
 \ The idea is that ReadData may at some point be "displaced" by some other routine that supports a more
 \ flexible syntax (see ReadData.h), whereas the old ReadData will have to remain around to support all the
 \ old files.
 \ Therefore, this file contains only a small selection of the code, and no variable definitions -- or at least
 \ as little as possible.

 */

#include "config.h"
IDENTIFY( "ReadData() c.s." );

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
#	if defined(__MACH__) || defined(__APPLE_CC__)
#		include <sys/file.h>
#	endif
#	include <fcntl.h>
#	include <libgen.h>

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

#include "ReadData.h"

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
extern int ignore_splits, splits_disconnect;

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
extern Process ReadData_proc;
extern char *transform_x_buf, transform_separator;
extern char *transform_y_buf;
extern int transform_x_buf_len, transform_y_buf_len;
extern int transform_x_buf_allen, transform_y_buf_allen;

extern int file_splits, plot_only_file, filename_in_legend, labels_in_legend, _process_bounds;
extern double data[ASCANF_DATA_COLUMNS];
extern int XUnitsSet, YUnitsSet, titleTextSet, column[ASCANF_DATA_COLUMNS], use_errors, no_errors, Show_Progress;
extern double newfile_incr_width;

extern LocalWin StubWindow, *InitWindow, *primary_info, *check_wi(LocalWin **wi, char *caller);
extern int SetIconName, UnmappedWindowTitle;

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
extern int lFX, lFY;
extern int FitX, FitY;
extern int Aspect, XSymmetric, YSymmetric;

extern int *plot_only_set, plot_only_set_len, *highlight_set, *mark_set, highlight_set_len, mark_set_len,
	no_title, AllTitles, no_legend, no_legend_box, legend_always_visible, Print_Orientation;
extern int mark_sets;
extern char *PrintFileName;
extern char UPrintFileName[];
extern int noExpY, noExpX, axisFlag, barBase_set, barWidth_set, barType, barType_set, debugging, debugLevel;
extern char *InFiles;

extern int WarnNewSet;

extern int maxitems;
extern int use_xye_info;
extern char *AddFile, *ScriptFile;
extern LocalWin *ScriptFileWin;

#include "xfree.h"

extern XGStringList *Init_Exprs, **the_init_exprs, *Startup_Exprs, *DumpProcessed_commands, *Dump_commands, *Exit_Exprs;
extern XGStringList *nextGEN_Startup_Exprs;
extern int Allow_InitExprs;

extern char d3str_format[16];
extern int d3str_format_changed;

extern int No_IncludeFiles;

extern int GetColor( char *name, Pixel *pix);
extern char *ParsedColourName;
#define StoreCName(name)	xfree(name);name=XGstrdup(ParsedColourName)

extern int XGStoreColours, XGIgnoreCNames;

extern ColourFunction IntensityColourFunction;
extern char *TBARprogress_header;
extern int ascanf_propagate_events;

extern int Free_Intensity_Colours();
extern int Intensity_Colours( char *exp );

extern int legend_setNumber;
extern int AddPoint_discard;

extern int Raw_NewSets, CorrectLinks;

extern int StartUp;
extern double *param_scratch;
extern int param_scratch_len;

extern int DumpFile;
extern int DumpIncluded;
extern int DumpPretty;
extern int DumpBinary, DumpDHex, DumpPens;

extern int is_pipe;
extern int ReadPipe;
extern char *ReadPipe_name;
extern FILE *ReadPipe_fp;

extern char *next_include_file, *version_list;

extern char *Read_File_Buf;

extern double param_dt;

extern int line_count;

extern Boolean ReadData_terpri;
extern char *ReadData_thisCurrentFileName;
extern FILE *ReadData_thisCurrentFP;
extern int ReadData_IgnoreVERSION;

extern char *process_history;
extern xtb_hret process_hist_h();

extern unsigned long mem_alloced;

extern int allocerr;

extern Boolean PIPE_error;

extern char data_separator;


Boolean ReadData_Outliers_Warn= 1;


int SwapEndian= False;

int16_t *SwapEndian_int16( int16_t *data, int N)
{ int i;
  unsigned char *dbyt;
	for( i= 0; i< N && data; i++ ){
		dbyt= (unsigned char *) &data[i];
		SWAP( dbyt[0], dbyt[1], unsigned char );
	}
	return( data );
}

int *SwapEndian_int( int *data, int N)
{ int i, j;
  unsigned char *dbyt;
	for( i= 0; i< N && data; i++ ){
		dbyt= (unsigned char *) &data[i];
		for( j= 0; j< sizeof(int)/2; j++ ){
			SWAP( dbyt[j], dbyt[sizeof(int)- 1- j], unsigned char );
		}
	}
	return( data );
}

int32_t *SwapEndian_int32( int32_t *data, int N)
{ int i, j;
  unsigned char *dbyt;
	for( i= 0; i< N && data; i++ ){
		dbyt= (unsigned char *) &data[i];
		for( j= 0; j< sizeof(int32_t)/2; j++ ){
			SWAP( dbyt[j], dbyt[sizeof(int32_t)- 1- j], unsigned char );
		}
	}
	return( data );
}

float *SwapEndian_float( float *data, int N)
{ int i, j;
  unsigned char *dbyt;
	for( i= 0; i< N && data; i++ ){
		dbyt= (unsigned char *) &data[i];
		for( j= 0; j< sizeof(float)/2; j++ ){
			SWAP( dbyt[j], dbyt[sizeof(float)- 1- j], unsigned char );
		}
	}
	return( data );
}

double *SwapEndian_double( double *data, int N)
{ int i, j;
  unsigned char *dbyt;
	for( i= 0; i< N && data; i++ ){
		dbyt= (unsigned char *) &data[i];
		for( j= 0; j< sizeof(double)/2; j++ ){
			SWAP( dbyt[j], dbyt[sizeof(double)- 1- j], unsigned char );
		}
	}
	return( data );
}

void *SwapEndian_data( void *data, int N, size_t datasize)
{ int i, j;
  unsigned char *Data, *dbyt;
	for( i= 0; i< N && data; i++ ){
		Data= (unsigned char*) data;
		dbyt= (unsigned char *) Data + i*datasize;
		for( j= 0; j< datasize/2; j++ ){
			SWAP( dbyt[j], dbyt[datasize- 1- j], unsigned char );
		}
	}
	return( data );
}

extern int scriptVerbose;

extern char ascanf_separator;

char *ReadData_proc_Append( char *src, char *append, int *needs_free )
{ char *c, *dest= NULL;
	if( src ){
		c= &src[strlen(src)-1];
		while( isspace(*c) ){
			c--;
		}
		if( *c== '@' ){
			c--;
		}
		while( isspace(*c) ){
			c--;
		}
		c[1]= '\0';
		if( *c== ascanf_separator ){
			dest= concat2( dest, src, append, NULL );
		}
		else{
		  char sep[2];
// 			dest= concat2( dest, src, ",", append, NULL );
			// 20101113: honour the ascanf_separator setting!!
			sep[0] = ascanf_separator, sep[1] = '\0';
			dest= concat2( dest, src, sep, append, NULL );
		}
		if( !dest ){
			fprintf( StdErr, "ReadData_proc_Append(): can't append \"%s\" to \"%s\": %s\n",
				append, src, serror()
			);
			*needs_free= False;
		}
		else{
			*needs_free= True;
		}
	}
	return( (dest)? dest : append );
}

int new_BoxFilter_Parser( char *optbuf, int offset, int filter )
{ char *membuf= &optbuf[offset];
  extern GenericProcess BoxFilter[BOX_FILTERS];
  GenericProcess *Filter= &BoxFilter[filter];
	xfree( Filter->process );
	Filter->process= XGstrdup( cleanup(membuf) );
	if( filter!= BOX_FILTER_CLEANUP ){
		new_process_BoxFilter_process( (ActiveWin)? ActiveWin : &StubWindow, filter );
	}
	return(1);
}

/* subroutine checking for and handling
 \ *DATA_INIT*
 \ *DATA_BEFORE*
 \ *DATA_PROCESS*
 \ *DATA_AFTER*
 \ *DATA_FINISH*
 \ *SET_PROCESS*
 \ 990717
 \ Shortens ReadData(), and needed on gcc 2.7.2 (Irix) to be able to compile with debug
 \ (otherwise an assembler branch-out-of-ranch).
 */
static int Check_Data_Process_etc( DataSet *this_set, char *Optbuf, double data[ASCANF_DATA_COLUMNS], Boolean *DF_bin_active, Boolean *LineDumped, char *the_file, char *ReadData_level )
{ int r= 0;
  char *optbuf= Optbuf, *c;
	if( optbuf[0]== '*' && (c= index(&optbuf[1],'*')) ){
		optbuf= SubstituteOpcodes( Optbuf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
			"*This-FileDir*", dirname(the_file),
			"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level, NULL );
	}
	if( strncmp( optbuf, "*DATA_INIT*", 11)== 0 ){
	  char *membuf= &optbuf[11], *MB;
	  int do_free= False;
		if( membuf[0]== '+' ){
			MB= membuf= ReadData_proc_Append( ReadData_proc.data_init, &membuf[1], &do_free );
		}
		xfree( ReadData_proc.data_init );
		Destroy_Form( &ReadData_proc.C_data_init );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			  /* cleanup() has removed the trailing '\n' that we
			   \ really want in this case!
			   */
			strcat( membuf, "\n");
			ReadData_proc.data_init= XGstrdup( membuf );
			ReadData_proc.data_init_len= strlen( ReadData_proc.data_init );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.data_init_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DATA_INIT* 1st compile";
				fascanf( &n, ReadData_proc.data_init, param_scratch, NULL, data, column, &ReadData_proc.C_data_init );
				TBARprogress_header= TBh;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DATA_INIT statement has error(s): discarded!\n" );
					xfree( ReadData_proc.data_init );
					ReadData_proc.data_init_len= 0;
					Destroy_Form( &ReadData_proc.C_data_init );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DATA_INIT: ");
						Print_Form( StdErr, &ReadData_proc.C_data_init, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DATA_INIT*");
						Print_Form( stdout, &ReadData_proc.C_data_init, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
				ascanf_verbose= av;
			}
		}
		else{
			ReadData_proc.data_init= NULL;
			ReadData_proc.data_init_len= 0;
			if( ActiveWin && ActiveWin->process.data_init[0] ){
				ActiveWin->process.data_init[0]= '\0';
				new_process_data_init( ActiveWin );
			}
		}
		if( do_free ){
			xfree( MB );
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DATA_BEFORE*", 13)== 0 ){
	  char *membuf= &optbuf[13];
		xfree( ReadData_proc.data_before );
		Destroy_Form( &ReadData_proc.C_data_before );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			  /* cleanup() has removed the trailing '\n' that we
			   \ really want in this case!
			   */
			strcat( membuf, "\n");
			ReadData_proc.data_before= XGstrdup( membuf );
			ReadData_proc.data_before_len= strlen( ReadData_proc.data_before );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.data_before_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DATA_BEFORE* 1st compile";
				fascanf( &n, ReadData_proc.data_before, param_scratch, NULL, data, column, &ReadData_proc.C_data_before );
				TBARprogress_header= TBh;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DATA_BEFORE statement has error(s): discarded!\n" );
					xfree( ReadData_proc.data_before );
					ReadData_proc.data_before_len= 0;
					Destroy_Form( &ReadData_proc.C_data_before );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DATA_BEFORE: ");
						Print_Form( StdErr, &ReadData_proc.C_data_before, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DATA_BEFORE*");
						Print_Form( stdout, &ReadData_proc.C_data_before, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
				ascanf_verbose= av;
			}
		}
		else{
			ReadData_proc.data_before= NULL;
			ReadData_proc.data_before_len= 0;
			if( ActiveWin && ActiveWin->process.data_before[0] ){
				ActiveWin->process.data_before[0]= '\0';
				new_process_data_before( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncasecmp( optbuf, "*DATA_PROCESS*", 14)== 0 ){
	  char *membuf= &optbuf[14];
		xfree( ReadData_proc.data_process );
		if( !strncmp( optbuf, "*data_process*", 14) ){
			ReadData_proc.data_process_now= 1;
		}
		else{
			ReadData_proc.data_process_now= 0;
		}
		ReadData_proc.data_process_len= 0;
		ReadData_proc.data_process= NULL;
		Destroy_Form( &ReadData_proc.C_data_process );
		if( strlen( cleanup(membuf) ) ){
		  char change[ASCANF_DATA_COLUMNS];
		  int n=ASCANF_DATA_COLUMNS, av= (ascanf_verbose)? 1 : 0;
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
#ifdef CHECK_DATA_PROCESS
			data[0]= data[1]= data[2]= data[3]= 0.0;
			fascanf( &n, membuf, data, change, data, column, NULL );
			if( !ascanf_arg_error && n &&
				(change[0]== 'N' || change[0]== 'R') &&
				(change[1]== 'N' || change[1]== 'R')
			)
#endif
			{
				ReadData_proc.data_process= XGstrdup( membuf );
				ReadData_proc.data_process_len= strlen( ReadData_proc.data_process );
/* 				Add_Comment( optbuf, True );	*/
				/* Compile the expression	*/{
					if( debugFlag ){
						ascanf_verbose= 1;
					}
					n= ASCANF_DATA_COLUMNS;
					*ascanf_self_value= *ascanf_current_value= 0.0;
					*ascanf_counter= 0;
					(*ascanf_Counter)= 0;
					data[0]= data[1]= data[2]= data[3]= 0.0;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					TBARprogress_header= "*DATA_PROCESS* 1st compile";
					fascanf( &n, ReadData_proc.data_process, data, change, data, column, &ReadData_proc.C_data_process );
					TBARprogress_header= TBh;
					if( ascanf_arg_error ){
						fprintf( StdErr, "DATA_PROCESS statement has error(s): discarded!\n" );
						xfree( ReadData_proc.data_process );
						ReadData_proc.data_process_len= 0;
						Destroy_Form( &ReadData_proc.C_data_process );
					}
					else{
						if( debugFlag ){
							fprintf( StdErr, "Compiled DATA_PROCESS (%s): ",
								(ReadData_proc.data_process_now)? "processing while reading" : "processing while drawing"
							);
							Print_Form( StdErr, &ReadData_proc.C_data_process, 0, True, NULL, NULL, "\n", True );
						}
						if( DumpFile ){
							if( *DF_bin_active ){
								BinaryTerminate(stdout);
								fputc( '\n', stdout);
								*DF_bin_active= False;
							}
							if( !DumpPretty ){
								fputs( "*EXTRATEXT* ", stdout );
							}
							else{
								*LineDumped= True;
							}
							fprintf( stdout, "*DATA_PROCESS*");
							Print_Form( stdout, &ReadData_proc.C_data_process, 0, True, NULL, "\t", NULL, True );
/* 							if( !DumpPretty ){	*/
								fputc( '\n', stdout );
/* 							}	*/
						}
					}
					ascanf_verbose= av;
				}
			}
		}
		else{
			if( ActiveWin && ActiveWin->process.data_process[0] ){
				ActiveWin->process.data_process[0]= '\0';
				new_process_data_process( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DATA_AFTER*", 12)== 0 ){
	  char *membuf= &optbuf[12];
		xfree( ReadData_proc.data_after );
		Destroy_Form( &ReadData_proc.C_data_after );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
			ReadData_proc.data_after= XGstrdup( membuf );
			ReadData_proc.data_after_len= strlen( ReadData_proc.data_after );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.data_after_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DATA_AFTER* 1st compile";
				fascanf( &n, ReadData_proc.data_after, param_scratch, NULL, data, column, &ReadData_proc.C_data_after );
				TBARprogress_header= TBh;
				ascanf_verbose= av;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DATA_AFTER statement has error(s): discarded!\n" );
					xfree( ReadData_proc.data_after );
					ReadData_proc.data_after_len= 0;
					Destroy_Form( &ReadData_proc.C_data_after );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DATA_AFTER: ");
						Print_Form( StdErr, &ReadData_proc.C_data_after, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DATA_AFTER*");
						Print_Form( stdout, &ReadData_proc.C_data_after, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
			}
		}
		else{
			ReadData_proc.data_after= NULL;
			ReadData_proc.data_after_len= 0;
			if( ActiveWin && ActiveWin->process.data_after[0] ){
				ActiveWin->process.data_after[0]= '\0';
				new_process_data_after( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DATA_FINISH*", 13)== 0 ){
	  char *membuf= &optbuf[13];
		xfree( ReadData_proc.data_finish );
		Destroy_Form( &ReadData_proc.C_data_finish );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
			ReadData_proc.data_finish= XGstrdup( membuf );
			ReadData_proc.data_finish_len= strlen( ReadData_proc.data_finish );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.data_finish_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DATA_FINISH* 1st compile";
				fascanf( &n, ReadData_proc.data_finish, param_scratch, NULL, data, column, &ReadData_proc.C_data_finish );
				TBARprogress_header= TBh;
				ascanf_verbose= av;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DATA_FINISH statement has error(s): discarded!\n" );
					xfree( ReadData_proc.data_finish );
					ReadData_proc.data_finish_len= 0;
					Destroy_Form( &ReadData_proc.C_data_finish );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DATA_FINISH: ");
						Print_Form( StdErr, &ReadData_proc.C_data_finish, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DATA_FINISH*");
						Print_Form( stdout, &ReadData_proc.C_data_finish, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
			}
		}
		else{
			ReadData_proc.data_finish= NULL;
			ReadData_proc.data_finish_len= 0;
			if( ActiveWin && ActiveWin->process.data_finish[0] ){
				ActiveWin->process.data_finish[0]= '\0';
				new_process_data_finish( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncasecmp( optbuf, "*DATA_PROCESS_DESCRIPTION*", 26)== 0 ){
	  char *membuf= cleanup( &optbuf[26] );
		xfree( ReadData_proc.description );
		if( membuf && membuf[0] ){
			ReadData_proc.description= XGstrdup( membuf );
			if( debugFlag ){
				fputs( membuf, StdErr );
			}
		}
		else if( ActiveWin ){
			xfree( ActiveWin->process.description );
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncasecmp( optbuf, "*SET_PROCESS*", 13)== 0 ){
	  char *membuf= &optbuf[13];
		xfree( this_set->process.set_process );
		this_set->process.set_process_len= 0;
		this_set->process.set_process= NULL;
		Destroy_Form( &this_set->process.C_set_process );
		if( strlen( cleanup(membuf) ) ){
		  char change[ASCANF_DATA_COLUMNS];
		  int n=ASCANF_DATA_COLUMNS, av= (ascanf_verbose)? 1 : 0;
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
#ifdef CHECK_DATA_PROCESS
			data[0]= data[1]= data[2]= data[3]= 0.0;
			fascanf( &n, membuf, data, change, data, column, NULL );
			if( !ascanf_arg_error && n &&
				(change[0]== 'N' || change[0]== 'R') &&
				(change[1]== 'N' || change[1]== 'R')
			)
#endif
			{
				this_set->process.set_process= XGstrdup( membuf );
				this_set->process.set_process_len= strlen( this_set->process.set_process );
/* 				Add_Comment( optbuf, True );	*/
				/* Compile the expression	*/{
					if( debugFlag ){
						ascanf_verbose= 1;
					}
					n= ASCANF_DATA_COLUMNS;
					*ascanf_self_value= *ascanf_current_value= 0.0;
					*ascanf_counter= 0;
					(*ascanf_Counter)= 0;
					data[0]= data[1]= data[2]= data[3]= 0.0;
					reset_ascanf_currentself_value= 0;
					reset_ascanf_index_value= True;
					TBARprogress_header= "*SET_PROCESS* 1st compile";
					fascanf( &n, this_set->process.set_process, data, change, data, column, &this_set->process.C_set_process );
					TBARprogress_header= TBh;
					if( ascanf_arg_error ){
						fprintf( StdErr, "SET_PROCESS statement has error(s): discarded!\n" );
						xfree( this_set->process.set_process );
						this_set->process.set_process_len= 0;
						Destroy_Form( &this_set->process.C_set_process );
					}
					else{
						if( debugFlag ){
							fprintf( StdErr, "Compiled SET_PROCESS: " );
							Print_Form( StdErr, &this_set->process.C_set_process, 0, True, NULL, NULL, "\n", True );
						}
						if( DumpFile ){
							if( *DF_bin_active ){
								BinaryTerminate(stdout);
								fputc( '\n', stdout);
								*DF_bin_active= False;
							}
							if( !DumpPretty ){
								fputs( "*EXTRATEXT* ", stdout );
							}
							else{
								*LineDumped= True;
							}
							fprintf( stdout, "*SET_PROCESS*");
							Print_Form( stdout, &this_set->process.C_set_process, 0, True, NULL, "\t", NULL, True );
/* 							if( !DumpPretty ){	*/
								fputc( '\n', stdout );
/* 							}	*/
						}
					}
					ascanf_verbose= av;
				}
			}
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncasecmp( optbuf, "*SET_PROCESS_DESCRIPTION*", 25)== 0 ){
	  char *membuf= cleanup( &optbuf[25] );
		xfree( this_set->process.description );
		if( membuf && membuf[0] ){
			this_set->process.description= XGstrdup( membuf );
			if( debugFlag ){
				fputs( membuf, StdErr );
			}
		}
		else if( ActiveWin ){
			fprintf( StdErr, "Warning: entering an empty *SET_PROCESS_DESCRIPTION* has no effect!\n" );
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DRAW_BEFORE*", 13)== 0 ){
	  char *membuf= &optbuf[13];
		xfree( ReadData_proc.draw_before );
		Destroy_Form( &ReadData_proc.C_draw_before );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			  /* cleanup() has removed the trailing '\n' that we
			   \ really want in this case!
			   */
			strcat( membuf, "\n");
			ReadData_proc.draw_before= XGstrdup( membuf );
			ReadData_proc.draw_before_len= strlen( ReadData_proc.draw_before );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.draw_before_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DRAW_BEFORE* 1st compile";
				fascanf( &n, ReadData_proc.draw_before, param_scratch, NULL, data, column, &ReadData_proc.C_draw_before );
				TBARprogress_header= TBh;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DRAW_BEFORE statement has error(s): discarded!\n" );
					xfree( ReadData_proc.draw_before );
					ReadData_proc.draw_before_len= 0;
					Destroy_Form( &ReadData_proc.C_draw_before );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DRAW_BEFORE [%d elements]: ", n);
						Print_Form( StdErr, &ReadData_proc.C_draw_before, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DRAW_BEFORE*");
						Print_Form( stdout, &ReadData_proc.C_draw_before, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
				ascanf_verbose= av;
			}
		}
		else{
			ReadData_proc.draw_before= NULL;
			ReadData_proc.draw_before_len= 0;
			if( ActiveWin && ActiveWin->process.draw_before[0] ){
				ActiveWin->process.draw_before[0]= '\0';
				new_process_draw_before( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r+= 1;
	}
	else if( strncmp( optbuf, "*DRAW_AFTER*", 12)== 0 ){
	  char *membuf= &optbuf[12];
		xfree( ReadData_proc.draw_after );
		Destroy_Form( &ReadData_proc.C_draw_after );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
			ReadData_proc.draw_after= XGstrdup( membuf );
			ReadData_proc.draw_after_len= strlen( ReadData_proc.draw_after );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.draw_after_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DRAW_AFTER* 1st compile";
				fascanf( &n, ReadData_proc.draw_after, param_scratch, NULL, data, column, &ReadData_proc.C_draw_after );
				TBARprogress_header= TBh;
				ascanf_verbose= av;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DRAW_AFTER statement has error(s): discarded!\n" );
					xfree( ReadData_proc.draw_after );
					ReadData_proc.draw_after_len= 0;
					Destroy_Form( &ReadData_proc.C_draw_after );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DRAW_AFTER [%d elements]: ", n);
						Print_Form( StdErr, &ReadData_proc.C_draw_after, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DRAW_AFTER*");
						Print_Form( stdout, &ReadData_proc.C_draw_after, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
			}
		}
		else{
			ReadData_proc.draw_after= NULL;
			ReadData_proc.draw_after_len= 0;
			if( ActiveWin && ActiveWin->process.draw_after[0] ){
				ActiveWin->process.draw_after[0]= '\0';
				new_process_draw_after( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r+= 1;
	}
	else if( strncmp( optbuf, "*DUMP_BEFORE*", 13)== 0 ){
	  char *membuf= &optbuf[13];
		xfree( ReadData_proc.dump_before );
		Destroy_Form( &ReadData_proc.C_dump_before );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			  /* cleanup() has removed the trailing '\n' that we
			   \ really want in this case!
			   */
			strcat( membuf, "\n");
			ReadData_proc.dump_before= XGstrdup( membuf );
			ReadData_proc.dump_before_len= strlen( ReadData_proc.dump_before );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.dump_before_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DUMP_BEFORE* 1st compile";
				fascanf( &n, ReadData_proc.dump_before, param_scratch, NULL, data, column, &ReadData_proc.C_dump_before );
				TBARprogress_header= TBh;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DUMP_BEFORE statement has error(s): discarded!\n" );
					xfree( ReadData_proc.dump_before );
					ReadData_proc.dump_before_len= 0;
					Destroy_Form( &ReadData_proc.C_dump_before );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DUMP_BEFORE [%d elements]: ", n);
						Print_Form( StdErr, &ReadData_proc.C_dump_before, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DUMP_BEFORE*");
						Print_Form( stdout, &ReadData_proc.C_dump_before, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
				ascanf_verbose= av;
			}
		}
		else{
			ReadData_proc.dump_before= NULL;
			ReadData_proc.dump_before_len= 0;
			if( ActiveWin && ActiveWin->process.dump_before[0] ){
				ActiveWin->process.dump_before[0]= '\0';
				new_process_dump_before( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r+= 1;
	}
	else if( strncmp( optbuf, "*DUMP_AFTER*", 12)== 0 ){
	  char *membuf= &optbuf[12];
		xfree( ReadData_proc.dump_after );
		Destroy_Form( &ReadData_proc.C_dump_after );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
			ReadData_proc.dump_after= XGstrdup( membuf );
			ReadData_proc.dump_after_len= strlen( ReadData_proc.dump_after );
/* 			Add_Comment( optbuf, True );	*/
			ReadData_proc.dump_after_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*DUMP_AFTER* 1st compile";
				fascanf( &n, ReadData_proc.dump_after, param_scratch, NULL, data, column, &ReadData_proc.C_dump_after );
				TBARprogress_header= TBh;
				ascanf_verbose= av;
				if( ascanf_arg_error ){
					fprintf( StdErr, "DUMP_AFTER statement has error(s): discarded!\n" );
					xfree( ReadData_proc.dump_after );
					ReadData_proc.dump_after_len= 0;
					Destroy_Form( &ReadData_proc.C_dump_after );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled DUMP_AFTER [%d elements]: ", n);
						Print_Form( StdErr, &ReadData_proc.C_dump_after, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*DUMP_AFTER*");
						Print_Form( stdout, &ReadData_proc.C_dump_after, 0, True, NULL, "\t", NULL, True );
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				}
			}
		}
		else{
			ReadData_proc.dump_after= NULL;
			ReadData_proc.dump_after_len= 0;
			if( ActiveWin && ActiveWin->process.dump_after[0] ){
				ActiveWin->process.dump_after[0]= '\0';
				new_process_dump_after( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r+= 1;
	}
	else if( strncmp( optbuf, "*ENTER_RAW_AFTER*", 17)== 0 ){
	  char *membuf= &optbuf[17];
		xfree( ReadData_proc.enter_raw_after );
		Destroy_Form( &ReadData_proc.C_enter_raw_after );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			  /* cleanup() has removed the trailing '\n' that we
			   \ really want in this case!
			   */
			strcat( membuf, "\n");
			ReadData_proc.enter_raw_after= XGstrdup( membuf );
			ReadData_proc.enter_raw_after_len= strlen( ReadData_proc.enter_raw_after );
			ReadData_proc.enter_raw_after_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*ENTER_RAW_BEFORE* 1st compile";
				fascanf( &n, ReadData_proc.enter_raw_after, param_scratch, NULL, data, column, &ReadData_proc.C_enter_raw_after );
				TBARprogress_header= TBh;
				if( ascanf_arg_error ){
					fprintf( StdErr, "ENTER_RAW_BEFORE statement has error(s): discarded!\n" );
					xfree( ReadData_proc.enter_raw_after );
					ReadData_proc.enter_raw_after_len= 0;
					Destroy_Form( &ReadData_proc.C_enter_raw_after );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled ENTER_RAW_BEFORE [%d elements]: ", n);
						Print_Form( StdErr, &ReadData_proc.C_enter_raw_after, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*ENTER_RAW_BEFORE*");
						Print_Form( stdout, &ReadData_proc.C_enter_raw_after, 0, True, NULL, "\t", NULL, True );
						fputc( '\n', stdout );
					}
				}
				ascanf_verbose= av;
			}
		}
		else{
			ReadData_proc.enter_raw_after= NULL;
			ReadData_proc.enter_raw_after_len= 0;
			if( ActiveWin && ActiveWin->process.enter_raw_after[0] ){
				ActiveWin->process.enter_raw_after[0]= '\0';
				new_process_enter_raw_after( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r+= 1;
	}
	else if( strncmp( optbuf, "*LEAVE_RAW_AFTER*", 17)== 0 ){
	  char *membuf= &optbuf[17];
		xfree( ReadData_proc.leave_raw_after );
		Destroy_Form( &ReadData_proc.C_leave_raw_after );
		if( strlen( cleanup(membuf) ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
			ReadData_proc.leave_raw_after= XGstrdup( membuf );
			ReadData_proc.leave_raw_after_len= strlen( ReadData_proc.leave_raw_after );
			ReadData_proc.leave_raw_after_printed= 1;
			/* Compile the expression	*/{
			  int av= (ascanf_verbose)? 1 : 0, n= param_scratch_len;
				if( debugFlag ){
					ascanf_verbose= 1;
				}
				*ascanf_self_value= *ascanf_current_value= 0.0;
				*ascanf_counter= 0;
				(*ascanf_Counter)= 0;
				reset_ascanf_currentself_value= 0;
				reset_ascanf_index_value= True;
				TBARprogress_header= "*LEAVE_RAW_AFTER* 1st compile";
				fascanf( &n, ReadData_proc.leave_raw_after, param_scratch, NULL, data, column, &ReadData_proc.C_leave_raw_after );
				TBARprogress_header= TBh;
				ascanf_verbose= av;
				if( ascanf_arg_error ){
					fprintf( StdErr, "LEAVE_RAW_AFTER statement has error(s): discarded!\n" );
					xfree( ReadData_proc.leave_raw_after );
					ReadData_proc.leave_raw_after_len= 0;
					Destroy_Form( &ReadData_proc.C_leave_raw_after );
				}
				else{
					if( debugFlag ){
						fprintf( StdErr, "Compiled LEAVE_RAW_AFTER [%d elements]: ", n);
						Print_Form( StdErr, &ReadData_proc.C_leave_raw_after, 0, True, NULL, NULL, "\n", True );
					}
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							*LineDumped= True;
						}
						fprintf( stdout, "*LEAVE_RAW_AFTER*");
						Print_Form( stdout, &ReadData_proc.C_leave_raw_after, 0, True, NULL, "\t", NULL, True );
						fputc( '\n', stdout );
					}
				}
			}
		}
		else{
			ReadData_proc.leave_raw_after= NULL;
			ReadData_proc.leave_raw_after_len= 0;
			if( ActiveWin && ActiveWin->process.leave_raw_after[0] ){
				ActiveWin->process.leave_raw_after[0]= '\0';
				new_process_leave_raw_after( ActiveWin );
			}
		}
		ReadData_commands+= 1;
		r+= 1;
	}
	else if( strncasecmp( optbuf, "*BOX_FILTER_INIT*", 17)== 0 ){
	  int n= new_BoxFilter_Parser( optbuf, 17, BOX_FILTER_INIT );
		ReadData_commands+= n;
		r+= n;
	}
	else if( strncasecmp( optbuf, "*BOX_FILTER*", 12)== 0 ){
	  int n= new_BoxFilter_Parser( optbuf, 12, BOX_FILTER_PROCESS );
		ReadData_commands+= n;
		r+= n;
	}
	else if( strncasecmp( optbuf, "*BOX_FILTER_AFTER*", 18)== 0 ){
	  int n= new_BoxFilter_Parser( optbuf, 18, BOX_FILTER_AFTER );
		ReadData_commands+= n;
		r+= n;
	}
	else if( strncasecmp( optbuf, "*BOX_FILTER_FINISH*", 19)== 0 ){
	  int n= new_BoxFilter_Parser( optbuf, 19, BOX_FILTER_FINISH );
		ReadData_commands+= n;
		r+= n;
	}
	else if( strncasecmp( optbuf, "*BOX_FILTER_CLEANUP*", 20)== 0 ){
	  int n= new_BoxFilter_Parser( optbuf, 20, BOX_FILTER_CLEANUP );
		ReadData_commands+= n;
		r+= n;
	}
	if( r ){
		ReadData_proc.separator = ascanf_separator;
	}
	if( optbuf!= Optbuf ){
		xfree(optbuf);
	}
	return(r);
}

int IsLabel( char *optbuf )
{ char *c= &optbuf[1];
  Boolean label= True;
	while( *c && label && *c!= '*' ){
		if( *c== ' ' ){
			label= False;
		}
		c++;
	}
	return( label );
}

extern char *key_param_now[256+8*12], key_param_separator;

char *xgiReadString2( char *buffer, int length, char *pbuf, char *defbuf )
{ int lnr= 0;
	if( Use_ReadLine ){
	  char *prompt;
		if( defbuf ){
			prompt= concat( "\a", pbuf, " (", defbuf, "): ", NULL );
		}
		else{
			prompt= concat( "\a", pbuf, ": ", NULL );
		}
		ReadLine( buffer, LMAXBUFSIZE, prompt, &lnr, XGStartJoin, XGEndJoin );
		xfree( prompt );
	}
	else{
		if( defbuf ){
			fprintf( StdErr, "\a%s (%s): ", pbuf, defbuf );
		}
		else{
			fprintf( StdErr, "\a%s: ", pbuf );
		}
		fflush( StdErr );
		ReadString( buffer, LMAXBUFSIZE, stdin, &lnr, XGStartJoin, XGEndJoin );
	}
	return( buffer );
}

static int Interactive_Commands( DataSet *this_set, char *optbuf, char *buffer, char *filename, int sub_div, double data[ASCANF_DATA_COLUMNS], Boolean *DF_bin_active, Boolean *LineDumped, char *ReadData_level )
{ int r= 0;
	if( strncmp( optbuf, "*ACTIVATE*", 10)== 0 ){
	  char *membuf= cleanup( &optbuf[10] );
		if( strlen( membuf ) ){
		  LocalWin *lwi1= NULL, *lwi2= NULL;
		  LocalWindows *WL= WindowList;
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
				if( DumpFile ){
					if( *DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						*DF_bin_active= False;
					}
					if( DumpPretty ){
						*LineDumped= True;
						print_string( stdout, "*ACTIVATE*", "\\n\n", "\n", membuf );
					}
				}
			TBARprogress_header= "*ACTIVATE*";
			if( (WL= Find_WindowListEntry_String( membuf, &lwi1, &lwi2 )) ){
				ActiveWin= WL->wi;
				ascanf_window= ActiveWin->window;
			}
			else{
/* 				if( ActiveWin ){	*/
/* 					ascanf_window= ActiveWin->window;	*/
/* 				}	*/
				ActiveWin= NULL;
				ascanf_window= 0;
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"==%s: no window 0x%lx or 0x%lx belongs to xgraph\n",
					filename, sub_div, line_count, buffer, d2str( ReadData_proc.param_range[0], NULL, NULL),
					lwi1, lwi2
				);
			}
			TBARprogress_header= TBh;
		}
		else if( debugFlag ){
			if( ActiveWin ){
				ascanf_window= ActiveWin->window;
			}
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\": missing window id\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*GEOM*", 6)== 0 ){
	  char *geoSpec= cleanup( &optbuf[6] );
		if( ActiveWin && geoSpec && *geoSpec ){
		  LocalWin *wi= ActiveWin;
		  int geo_mask, gr= 0, an= 4;
		  int e= 0, h= wi->halt, ss= Synchro_State;
		  XWindowAttributes win_attr;
		  XSizeHints sh;
		  Window dummy;
		  char def_geom[512], ageoSpec[512];
		  double ageom[4];
		  char *TBh= TBARprogress_header;
			memset( &win_attr, 0, sizeof(XWindowAttributes));
			memset( &sh, 0, sizeof(XSizeHints));

			Synchro_State= 0; X_Synchro( wi );

			XGetWindowAttributes(disp, wi->window, &win_attr);
			XTranslateCoordinates( disp, wi->window, win_attr.root, 0, 0,
				&win_attr.x, &win_attr.y, &dummy
			);
/* 				win_attr.y-= WM_TBAR;	*/
			  /* 20060926 */
			{ long dum; XGetWMNormalHints( disp, wi->window, &sh, &dum ); }
#if 0
			sh.flags = USPosition|USSize|PPosition|PSize|PMinSize|PBaseSize|PMaxSize;
			sh.x= win_attr.x;
			sh.y= win_attr.y;
			sh.base_width= sh.width= win_attr.width;
			sh.base_height= sh.height= win_attr.height;
			sh.min_width= sh.min_height= 10;
			sh.max_width= DisplayWidth(disp, screen)* 2;
			sh.max_height= DisplayHeight(disp, screen)* 2;
#endif
			sh.width_inc= sh.height_inc= 1;
			sh.win_gravity= win_attr.win_gravity;
			sprintf( def_geom, "%dx%d+%d+%d", sh.width, sh.height, sh.x, sh.y );
			param_scratch[0]= sh.width;
			param_scratch[1]= sh.height;
			param_scratch[2]= sh.x;
			param_scratch[3]= sh.y;
			TBARprogress_header= "*GEOM*";
			if( (an= new_param_now( geoSpec, ageom, 4 ))>= 2 ){
			  int i;
			  char xsign= '+', ysign= '+';
				for( i= 0; i< an; i++ ){
					CLIP_EXPR( ageom[i], param_scratch[i], MININT, MAXINT );
				}
				if( an== 4 ){
					if( param_scratch[2]< 0 ){
						 xsign= '-';
					}
					if( param_scratch[3]< 0 ){
						 ysign= '-';
					}
					sprintf( ageoSpec, "%dx%d%c%d%c%d",
						 (int) ageom[0], (int) ageom[1], xsign, (int) ageom[2], ysign, (int) ageom[3]
					);
					geoSpec= ageoSpec;
				}
				else{
					sprintf( ageoSpec, "%dx%d", (int) ageom[0], (int) ageom[1] );
					geoSpec= ageoSpec;
				}
			}

#define XWMGEOMETRY_WORKS

#ifdef XWMGEOMETRY_WORKS
			  /* It has actually worked, but somewhere a (probable) change in the code had this
			   \ effect of effectively ruining the effectiveness of XWMGeometry()...
			   \ I don't see though where that could have happened!.
			   \ 990202: oh well...
			   \ 20060926: XWMGeometry() remains flaky. On Darwin 10.4 it adds min_width,min_height to width,height, when passing sh as obtained above?!
			   */
			{ int x, y;
			  unsigned int width, height;
				geo_mask= XParseGeometry(geoSpec, &x, &y, &width, &height);
				geo_mask= XWMGeometry(disp, screen,
					  geoSpec, def_geom, bdrSize, &sh, &win_attr.x, &win_attr.y, &win_attr.width, &win_attr.height, &gr);
				if( win_attr.width== width+ sh.min_width ){
					win_attr.width= width;
				}
				if( win_attr.height== height+ sh.min_height ){
					win_attr.height= height;
				}
			}
#else
			geo_mask= XParseGeometry(geoSpec, &win_attr.x, &win_attr.y, &win_attr.width, &win_attr.height);
#endif
			if( (geo_mask & XValue) || (geo_mask & YValue) || (geo_mask & WidthValue) || (geo_mask & HeightValue) ){
			  XWindowAttributes winfo;
			  int check= 2, count= 0;
				if( (geo_mask & XNegative) ){
				}
				if( (geo_mask & YNegative) ){
				}
				XMoveResizeWindow( disp, wi->window, win_attr.x, win_attr.y, win_attr.width, win_attr.height );
				while( check && count< 5 ){
					XGetWindowAttributes( disp, wi->window, &winfo);
					if( win_attr.width!= winfo.width ){
						if( debugFlag || abs(win_attr.width- winfo.width)> 1000 ){
							fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\" => %s:"
											" requested window width %d, got %d; correcting\n",
								filename, sub_div, line_count, buffer, geoSpec,
								win_attr.width, winfo.width
							);
						}
					}
					else{
						check-= 1;
					}
					if( win_attr.height!= winfo.height ){
						if( debugFlag || abs(win_attr.height- winfo.height)> 1000 ){
							fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\" => %s:"
											" requested window height %d, got %d; correcting\n",
								filename, sub_div, line_count, buffer, geoSpec,
								win_attr.height, winfo.height
							);
						}
					}
					else{
						check-= 1;
					}
					if( check ){
						XSync( disp, True );
						XResizeWindow( disp, wi->window, win_attr.width, win_attr.height );
						check= 2;
						count+= 1;
					}
				}
				  /* RJVB 20060926 */
				sh.x= win_attr.x;
				sh.y= win_attr.y;
				sh.base_width= sh.width;
				sh.base_height= sh.base_height;
				sh.width= win_attr.width;
				sh.height= win_attr.height;
				XSetWMNormalHints( disp, wi->window, &sh );

				AdaptWindowSize( wi, wi->window, win_attr.width, win_attr.height);
				wi->halt= 0;
				while( Handle_An_Event( wi->event_level, 1, "ReadData-GEOM", wi->window,
						ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask
					)
				){
					e+= 1;
					wi->halt= 0;
				}
				if( wi->redraw ){
					RedrawNow( wi );
				}
				wi->halt= h;
				ActiveWin= wi;
				if( debugFlag || scriptVerbose ){
					fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\" => %s\t: window 0x%lx %02d:%02d:%02d geom %dx%d+%d+%d (%d events)\n",
						filename, sub_div, line_count, buffer, geoSpec, ActiveWin,
						ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number,
						win_attr.width, win_attr.height, win_attr.x, win_attr.y, e
					);
				}
			}
			Synchro_State= !ss; X_Synchro( wi );
			TBARprogress_header= TBh;
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: missing geom spec (WxH+X+Y) or no window active to be cloned\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*TILE*", 6)== 0 ){
	  char *argbuf= cleanup( &optbuf[6] );
	  int direction, horvert= False;
		while( isspace(*argbuf) && *argbuf ){
			argbuf++;
		}
		if( index( "uUdD", *argbuf ) ){
			direction= tolower( *argbuf );
			while( !isspace(*argbuf) && *argbuf ){
				argbuf++;
			}
			while( isspace(*argbuf) && *argbuf ){
				argbuf++;
			}
			if( *argbuf ){
				if( index( "vV", *argbuf ) ){
					horvert= 1;
				}
				else if( index( "hH", *argbuf ) ){
					horvert= -1;
				}
			}
			TileWindows( direction, horvert );
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*QUIET*", 7)== 0 ){
	  char *argbuf= cleanup( &optbuf[7] );
		while( isspace(*argbuf) && *argbuf ){
			argbuf++;
		}
		if( *argbuf ){
		  int count= -1;
		  char *TBh= TBARprogress_header;
			ReadData_proc.param_range[0]= -1;
			TBARprogress_header= "*QUIET*";
			new_param_now( argbuf, &ReadData_proc.param_range[0], 1 );
			if( (count= (int) ReadData_proc.param_range[0])> 0 ){
				QuietErrors= True;
				quiet_error_count= count;
			}
			else{
				QuietErrors= False;
			}
			TBARprogress_header= TBh;
		}
		else{
			QuietErrors= True;
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*CLONE*", 7)== 0 ){
	  extern char Window_Name[256];
	  extern Cursor zoomCursor;
	  int e= 0;
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *wi= ActiveWin;
		  Cursor curs= zoomCursor;
			HandleMouse(Window_Name,
						  NULL,
						  wi, NULL, &curs
			);
			e+= Handle_An_Events( wi->event_level, 1, "ReadData-CLONE-parent", wi->window,
					ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask
			);
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: window 0x%lx clone of 0x%lx %02d:%02d:%02d cloned (%d events)\n",
					filename, sub_div, line_count, buffer, ActiveWin, wi,
					wi->parent_number, wi->pwindow_number, wi->window_number, e
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active to be cloned\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*ICONIFY*", 9)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *wi= ActiveWin;
		  int e= 0, h= wi->halt, ss= Synchro_State;
		  xtb_frame *frame= xtb_lookup_frame( wi->window );
#ifdef DEBUG
		  int dF= debugFlag, dL= debugLevel;
			debugFlag= 1, debugLevel= -2;
#endif
			Synchro_State= 0; X_Synchro( wi );
			XIconifyWindow( disp, wi->window, screen );
			wi->mapped= 0;
			if( frame ){
				frame->mapped= 0;
			}
			wi->halt= 0;
			XG_sleep_once( 0.1, True );
			while( Handle_An_Event( wi->event_level, 1, "ReadData-ICONIFY", 0*wi->window,
					ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask
				)
			){
				e+= 1;
				wi->halt= 0;
				ActiveWin= wi;
			}
			XSync( disp, True );
#ifdef DEBUG
			debugFlag= dF, debugLevel= dL;
#endif
			wi->halt= h;
			ActiveWin= wi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: window 0x%lx %02d:%02d:%02d iconified"
				" (%d events) (%s)\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number, e,
					(frame)? "state change registered in FRAME" : "FRAME not found -- state change not registered"
				);
			}
			Synchro_State= !ss; X_Synchro( wi );
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active to be iconified\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DEICONIFY*", 11)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *wi= ActiveWin;
		  int e= 0, h= wi->halt, ss= Synchro_State;
		  xtb_frame *frame= xtb_lookup_frame( wi->window );
			Synchro_State= 0; X_Synchro( wi );
			XMapRaised( disp, wi->window );
			wi->mapped= 1;
			if( frame ){
				frame->mapped= 1;
			}
			RedrawNow( wi );
			wi->halt= 0;
			XG_sleep_once( 0.1, True );
#ifdef DEBUG
		{  int dF= debugFlag, dL= debugLevel;
			debugFlag= 1, debugLevel= -2;
#endif
			XSync( disp, False );
			while( Handle_An_Event( wi->event_level, 1, "ReadData-DEICONIFY", 0*wi->window,
					ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask
				)
			){
				e+= 1;
				wi->halt= 0;
				ActiveWin= wi;
			}
			XSync( disp, False );
#ifdef DEBUG
			debugFlag= dF, debugLevel= dL;
		}
#endif
			wi->halt= h;
			ActiveWin= wi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: window 0x%lx %02d:%02d:%02d de-iconified"
				" (%d events) (%s)\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number, e,
					(frame)? "state change registered in FRAME" : "FRAME not found -- state change not registered"
				);
			}
			Synchro_State= !ss; X_Synchro( wi );
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active to be de-iconified\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*FLUSH_EVENTS*", 14)== 0 ){
	  char *discard= cleanup( &optbuf[14] );
	  int e= 0;
	  LocalWin *wi= ActiveWin;
		if( strncasecmp( discard, "discard", 7)== 0 ){
			XSync( disp, True );
		}
		else{
			XSync( disp, False );
		}
		if( wi ){
			while( Handle_An_Event( wi->event_level, 1, "ReadData-FLUSH_EVENTS", 0,
					ExposureMask|StructureNotifyMask|KeyPressMask|ButtonPressMask
				)
			){
				e+= 1;
				wi->halt= 0;
				ActiveWin= wi;
			}
		}
		XSync( disp, False );
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*WAIT_EVENT*", 12)== 0 ){
	  char *event= cleanup( &optbuf[12] ), *msg= strstr( event, "::");
	  int type= 0;
		if( strncasecmp( event, "key", 3)== 0 ){
			type= KeyPress;
		}
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *wi= ActiveWin;
			if( msg ){
				msg+= 2;
			}
			WaitForEvent( ActiveWin, type, msg, "ReadData-WAIT_EVENT" );
			ActiveWin= wi;
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active to wait for event\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*UPDATE_FLAGS*", 14)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
			CopyFlags( NULL, ActiveWin );
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: flags updated from window 0x%lx %02d:%02d:%02d\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*UPDATE_WIN*", 12)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
			CopyFlags( lwi, NULL );
			UpdateWindowSettings( lwi, False, True );
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: updated window 0x%lx %02d:%02d:%02d from flags\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*UPDATE_PROCS*", 14)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int r;
		  extern int DrawWindow_Update_Procs;
		  int os= lwi->silenced;
			lwi->silenced= True;
			xtb_bt_set( lwi->ssht_frame.win, lwi->silenced, (char *) 0);
			lwi->dev_info.xg_silent( lwi->dev_info.user_state, lwi->silenced );
			DrawWindow_Update_Procs= 1;
			r= RedrawNow( lwi );
			DrawWindow_Update_Procs= 0;
			lwi->silenced= os;
			xtb_bt_set( lwi->ssht_frame.win, lwi->silenced, (char *) 0);
			lwi->dev_info.xg_silent( lwi->dev_info.user_state, lwi->silenced );
			lwi->halt= 0;
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: window 0x%lx %02d:%02d:%02d: %s\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number,
					(r)? " procedures updated" : " window waiting or erroneous procedure"
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*MARK_DRAWN*", 12)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int i;
			xtb_bt_swap( lwi->settings );
			for( i= 0; i< setNumber; i++ ){
				lwi->mark_set[i]+= (lwi->draw_set[i])? 1 : 0;
			}
			xtb_bt_swap( lwi->settings );
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: marked drawn sets in window 0x%lx %02d:%02d:%02d\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*UNMARK_DRAWN*", 14)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int i;
			xtb_bt_swap( lwi->settings );
			for( i= 0; i< setNumber; i++ ){
				lwi->mark_set[i]-= (lwi->draw_set[i])? 1 : 0;
			}
			xtb_bt_swap( lwi->settings );
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: unmarked drawn sets in window 0x%lx %02d:%02d:%02d\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DRAW_MARKED*", 13)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int i;
			xtb_bt_swap( lwi->settings );
			for( i= 0; i< setNumber; i++ ){
				lwi->draw_set[i]= (lwi->mark_set[i])? 1 : 0;
			}
			xtb_bt_swap( lwi->settings );
			RedrawNow( lwi );
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: drawn marked sets in window 0x%lx %02d:%02d:%02d\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*SHIFT_DRAWN*", 13)== 0 ){
	  char *opcode= cleanup( &optbuf[13] );
	  int i, dirn= 0, extreme= False;
		if( strncasecmp( opcode, "START", 5)== 0 ){
			dirn= -1;
			extreme= True;
		}
		else if( strncasecmp( opcode, "END", 3)== 0 ){
			dirn= 1;
			extreme= True;
		}
		if( strncasecmp( opcode, "LEFT", 4)== 0 ){
			dirn= -1;
		}
		else if( strncasecmp( opcode, "RIGHT", 5)== 0 ){
			dirn= 1;
		}
		else if( debugFlag ){
			fprintf( StdErr,
				"ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: invalid opcode; should be one of START, LEFT, RIGHT, END\n",
				filename, sub_div, line_count, buffer
			);
		}
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int n;
			for( i= 0; i< setNumber; i++ ){
				AllSets[i].draw_set= draw_set( lwi, i );
			}
			n= ShiftDataSets_Drawn( lwi, dirn, extreme, True );
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr,
					"ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: shifted %d drawn sets %s%s in"
					" window 0x%lx %02d:%02d:%02d",
					filename, sub_div, line_count, buffer, n,
					(extreme)? "completely " : "", (dirn>0)? "to the right" : "to the left",
					ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*RAW_DISPLAY*", 13)== 0 ){
	  char *membuf= cleanup( &optbuf[13] );
	  int raw;
		if( strlen( membuf ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
				if( DumpFile ){
					if( *DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						*DF_bin_active= False;
					}
					if( DumpPretty ){
						*LineDumped= True;
						print_string( stdout, "*RAW_DISPLAY*", "\\n\n", "\n", membuf );
					}
				}
			ReadData_proc.param_range[0]= 0;
			TBARprogress_header= "*RAW_DISPLAY*";
			new_param_now( membuf, &ReadData_proc.param_range[0], 1 );
			raw= (int) ReadData_proc.param_range[0];
			if( ActiveWin && ActiveWin!= &StubWindow ){
			  LocalWin *lwi= ActiveWin;
				RawDisplay( lwi, (raw)? True : False );
				RedrawNow( lwi );
				if( lwi->SD_Dialog ){
					set_changeables(2,False);
				}
				ActiveWin= lwi;
				if( debugFlag || scriptVerbose ){
					fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: %s in window 0x%lx %02d:%02d:%02d\n",
						filename, sub_div, line_count, buffer,
						(ActiveWin->raw_display)? "raw mode" : "processing mode",
						ActiveWin,
						ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
					);
				}
			}
			else if( debugFlag ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
					filename, sub_div, line_count, buffer
				);
			}
			TBARprogress_header= TBh;
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*TITLEMSGS*", 11)== 0 ){
	  char *membuf= cleanup( &optbuf[11] );
	  extern int noTitleMessages;
		if( strlen( membuf ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
				if( DumpFile ){
					if( *DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						*DF_bin_active= False;
					}
					if( DumpPretty ){
						*LineDumped= True;
						print_string( stdout, "*TITLEMSGS*", "\\n\n", "\n", membuf );
					}
				}
			ReadData_proc.param_range[0]= 0;
			TBARprogress_header= "*TITLEMSGS*";
			new_param_now( membuf, &ReadData_proc.param_range[0], 1 );
			noTitleMessages= ! ((int) ReadData_proc.param_range[0]);
			TBARprogress_header= TBh;
		}
		else{
			noTitleMessages= 0;
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*SILENCE*", 9)== 0 ){
	  char *membuf= cleanup( &optbuf[9] );
	  int silence, all;
	  LocalWin *awi= ActiveWin, *lwi;
	  LocalWindows *WL= WindowList;
		if( strncasecmp(membuf, "all", 3)== 0 && isspace(membuf[3]) ){
			all= True;
			membuf= cleanup(&membuf[3]);
			lwi= (WL)? WL->wi : NULL;
		}
		else{
			all= False;
			lwi= ActiveWin;
		}
		if( strlen( membuf ) ){
		  char *TBh= TBARprogress_header;
			strcat( membuf, "\n");
				if( DumpFile ){
					if( *DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						*DF_bin_active= False;
					}
					if( DumpPretty ){
						*LineDumped= True;
						if( all ){
							print_string( stdout, "*SILENCE* all ", "\\n\n", "\n", membuf );
						}
						else{
							print_string( stdout, "*SILENCE*", "\\n\n", "\n", membuf );
						}
					}
				}
			ReadData_proc.param_range[0]= 0;
			TBARprogress_header= "*SILENCE*";
			new_param_now( membuf, &ReadData_proc.param_range[0], 1 );
			silence= (int) ReadData_proc.param_range[0];
			if( (lwi && lwi!= &StubWindow) || all ){
				while( lwi ){
				  int os= lwi->silenced;
					lwi->silenced= (silence)? True : False;
					xtb_bt_set( lwi->ssht_frame.win, lwi->silenced, (char *) 0);
					lwi->dev_info.xg_silent( lwi->dev_info.user_state, lwi->silenced );
					if( !lwi->silenced && os< 0 ){
						RedrawNow( lwi );
						if( lwi->SD_Dialog ){
							set_changeables(2,False);
						}
					}
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: %s in window 0x%lx %02d:%02d:%02d\n",
							filename, sub_div, line_count, buffer,
							(lwi->silenced)? "silent mode" : "display active",
							lwi,
							lwi->parent_number, lwi->pwindow_number, lwi->window_number
						);
					}
					if( WL->next ){
						WL= WL->next;
						lwi= WL->wi;
					}
					else{
						lwi= NULL;
					}
				}
				ActiveWin= awi;
			}
			else if( debugFlag ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
					filename, sub_div, line_count, buffer
				);
			}
			TBARprogress_header= TBh;
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*PROCESS_RESULTS_DRAWN*", 23)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int i, j, sets= 0;
		  DataSet *this_set;
		  char mbuf[64];
			xtb_bt_swap( lwi->settings );
			CheckProcessUpdate(lwi, True, False, False );
#if 0
			if( !lwi->raw_display ){
			  Boolean done= False;
			  extern double *disable_SET_PROCESS;
				while( lwi->redraw){
					RedrawNow( lwi );
				}
				for( i= 0; i< setNumber && !done; i++ ){
					if( draw_set( lwi, i) ){
						this_set= &AllSets[i];
						if( this_set->last_processed_wi!= lwi &&
							(lwi->transform.x_len || lwi->transform.y_len || lwi->process.data_process_len ||
								(!*disable_SET_PROCESS && this_set->process.set_process_len)
							)
						){
							RedrawNow( lwi );
							done= True;
						}
					}
				}
			}
#endif
			for( i= 0; i< setNumber; i++ ){
			  int n;
				  /* 20040608: don't touch "raw sets" */
				if( draw_set( lwi, i) && !AllSets[i].raw_display ){
					sprintf( mbuf, "PROCESS_RESULTS #%d", i );
					XStoreName( disp, lwi->window, mbuf );
					if( !RemoteConnection ){
						XSync( disp, False );
					}
					this_set= &AllSets[i];
					  /* This converts a linked set into a "static" set: */
					if( this_set->set_link>= 0 ){
					  double **lcolumns= this_set->columns;
					  int k;
						this_set->columns= NULL;
						this_set->set_link= -1;
						this_set->columns= realloc_columns( this_set, this_set->ncols );
						for( k= 0; k< this_set->ncols; k++ ){
							memcpy( this_set->columns[k], lcolumns[k], this_set->numPoints* sizeof(double) );
						}
					}
					for( n= 0, j= 0; j< this_set->numPoints; j++ ){
						if( !DiscardedPoint( lwi, this_set, j) ){
						  int k;
							for( k= 0; k< this_set->ncols; k++ ){
								if( k== this_set->xcol ){
									this_set->columns[k][n]= this_set->xvec[j];
								}
								else if( k== this_set->ycol ){
									this_set->columns[k][n]= this_set->yvec[j];
								}
								else if( k== this_set->ecol ){
									if( !lwi->transform.y_len ){
										this_set->columns[k][n]= this_set->errvec[j];
									}
									else{
										this_set->columns[k][n]= (this_set->hdyvec[j]- this_set->ldyvec[j])/2;
									}
								}
								else if( k== this_set->lcol ){
									if( !lwi->transform.x_len ){
										this_set->columns[k][n]= this_set->lvec[j];
									}
									else{
										this_set->columns[k][n]= (this_set->hdxvec[j]- this_set->ldxvec[j])/2;
									}
								}
								else{
									this_set->columns[k][n]= this_set->columns[k][j];
								}
							}
							if( this_set->splithere ){
								this_set->splithere[n]= this_set->splithere[j];
							}
							n++;
						}
					}
					sets+= 1;
					if( n<= this_set->numPoints ){
						if( n< this_set->numPoints ){
							this_set->numPoints= n;
							realloc_points( this_set, n+ 2, False );
						}
						this_set->points_added+= 1;
						for( j= 0; j< n; j++ ){
							this_set->discardpoint[j]= 0;
						}
						  /* 20030226: hah! *PROCESS_RESULTS_DRAWN* is supposed to eliminate discarded points,
						   \ so it should reset the discardpoint arrays!!!
						   */
						if( lwi && lwi->discardpoint && lwi->discardpoint[this_set->set_nr] ){
							for( j= 0; j< n; j++ ){
								lwi->discardpoint[this_set->set_nr][j]= 0;
							}
						}
						Reset_SetUndo(this_set);
						lwi->redraw= True;
					}
					RedrawSet(i, False);
				}
			}
			if( sets== setNumber ){
				strcpy( lwi->XUnits, lwi->tr_XUnits );
				strcpy( lwi->YUnits, lwi->tr_YUnits );
			}
			xtb_bt_swap( lwi->settings );
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr,
					"ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: "
						"replaced raw for processed values for %d drawn sets in window 0x%lx %02d:%02d:%02d\n",
					filename, sub_div, line_count, buffer, sets, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag || scriptVerbose ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DUPLICATE_RESULTS_DRAWN*", 25)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
			xtb_bt_swap( lwi->settings );
			CheckProcessUpdate(lwi, True, False, False );
			Duplicate_Visible_Sets( lwi, False );
			xtb_bt_swap( lwi->settings );
			if( lwi->SD_Dialog ){
				set_changeables(2,False);
			}
			ActiveWin= lwi;
			if( debugFlag || scriptVerbose ){
				fprintf( StdErr,
					"ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: "
						"duplicated the processed drawn sets in window 0x%lx %02d:%02d:%02d\n",
					filename, sub_div, line_count, buffer, ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
			}
		}
		else if( debugFlag || scriptVerbose ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DRAW_SET*", 10)== 0 ||
		strncmp( optbuf, "*MARK_SET*", 10)== 0 ||
		strncmp( optbuf, "*PROCESS_SET*", 13)== 0
	){
	  char *membuf= cleanup( (optbuf[1]== 'P')? &optbuf[13] : &optbuf[10] );
		if( ActiveWin && ActiveWin!= &StubWindow && strlen( membuf ) ){
		  LocalWin *lwi= ActiveWin;
		  int i, set_nr, hit= 0;
		  char *pat;
		  Boolean rexp= False, proc= False, restore;
		  short *setprop= (optbuf[1]== 'D')? lwi->draw_set : lwi->mark_set;
		  ALLOCA( SetProp, short, setNumber, setprop_len);

			if( optbuf[1]== 'P' ){
				setprop= lwi->draw_set;
				if( strncmp( membuf, "RESTORE", 7)== 0 ){
					  /* Save the current draw_set information.	*/
					for( i= 0; i< setNumber; i++ ){
						SetProp[i]= setprop[i];
					}
					membuf+= 7;
					restore= True;
				}
				else{
					restore= False;
				}
				proc= True;
			}

			XStoreName( disp, lwi->window, optbuf );
			if( !RemoteConnection ){
				XSync( disp, False );
			}
			if( strncmp( membuf, "TITLE", 5)== 0 ){
				pat= &membuf[5];
				while( isspace( (unsigned char) *pat ) ){
					pat++;
				}
				if( strncmp( pat, "RE^", 3)== 0 ){
					rexp= True;
					pat+= 2;
				}
				if( pat && *pat ){
					if( rexp ){
						re_comp( pat );
					}
					if( setprop== lwi->draw_set ){
						for( i= 0; i< setNumber; i++ ){
							setprop[i]= 0;
						}
					}
					if( rexp ){
						for( i= 0; i< setNumber; i++ ){
							if( AllSets[i].titleText && re_exec( AllSets[i].titleText)> 0 ){
								hit+= 1;
								setprop[i]= 1;
							}
						}
					}
					else{
						for( i= 0; i< setNumber; i++ ){
							if( AllSets[i].titleText && strstr( AllSets[i].titleText, pat ) ){
								hit+= 1;
								setprop[i]= 1;
							}
						}
					}
				}
			}
			else if( strncmp( membuf, "LEGEND", 6)== 0 ){
				pat= &membuf[6];
				while( isspace( (unsigned char) *pat ) ){
					pat++;
				}
				if( strncmp( pat, "RE^", 3)== 0 ){
					rexp= True;
					pat+= 2;
				}
				if( pat && *pat ){
					if( rexp ){
						re_comp( pat );
					}
					if( setprop== lwi->draw_set ){
						for( i= 0; i< setNumber; i++ ){
							setprop[i]= 0;
						}
					}
					if( rexp ){
						for( i= 0; i< setNumber; i++ ){
							if( AllSets[i].setName && re_exec( AllSets[i].setName)> 0 ){
								hit+= 1;
								setprop[i]= 1;
							}
						}
					}
					else{
						for( i= 0; i< setNumber; i++ ){
							if( AllSets[i].setName && strstr( AllSets[i].setName, pat ) ){
								hit+= 1;
								setprop[i]= 1;
							}
						}
					}
				}
			}
			else if( strncmp( membuf, "EVAL", 4)== 0 ){
			  GenericProcess expr;
			  int n= 1;
			  char *TBh= TBARprogress_header;
				pat= &membuf[4];
				expr.process= XGstrdup( cleanup(pat) );
				expr.C_process= NULL;
				expr.separator = ascanf_separator;
				clean_param_scratch();
				TBARprogress_header= "*DRAW_SET*/*MARK_SET*/*PROCESS_SET* EVAL";
				strncpy( expr.command, TBARprogress_header, sizeof(expr.command)-1 );
				fascanf( &n, expr.process, param_scratch, NULL, data, column, &expr.C_process );
				if( expr.C_process ){
				  double asn= *ascanf_setNumber, anp= *ascanf_numPoints;
					if( setprop== lwi->draw_set ){
						for( i= 0; i< setNumber; i++ ){
							setprop[i]= 0;
						}
					}
					for( i= 0; i< setNumber; i++ ){
						*ascanf_setNumber= i;
						*ascanf_numPoints= AllSets[i].numPoints;
						ascanf_arg_error= 0;
						n= 1;
						data[0]= data[1]= data[2]= data[3]= 0;
						compiled_fascanf( &n, expr.process, param_scratch, NULL, data, column, &expr.C_process );
						if( n && param_scratch[0] && !NaN(param_scratch[0]) ){
							hit+= 1;
							setprop[i]= 1;
						}
					}
					*ascanf_setNumber= asn;
					*ascanf_numPoints= anp;
				}
				Destroy_Form( &expr.C_process );
				xfree( expr.process );
				TBARprogress_header= TBh;
			}
			else if( strncmp( membuf, "FILE", 4)== 0 ){
				pat= &membuf[4];
				while( isspace( (unsigned char) *pat ) ){
					pat++;
				}
				if( strncmp( pat, "RE^", 3)== 0 ){
					rexp= True;
					pat+= 2;
				}
				if( pat && *pat ){
					if( rexp ){
						re_comp( pat );
					}
					if( setprop== lwi->draw_set ){
						for( i= 0; i< setNumber; i++ ){
							setprop[i]= 0;
						}
					}
					if( rexp ){
						for( i= 0; i< setNumber; i++ ){
							if( AllSets[i].fileName && re_exec( AllSets[i].fileName)> 0 ){
								hit+= 1;
								setprop[i]= 1;
							}
						}
					}
					else{
						for( i= 0; i< setNumber; i++ ){
							if( AllSets[i].fileName && strstr( AllSets[i].fileName, pat ) ){
								hit+= 1;
								setprop[i]= 1;
							}
						}
					}
				}
			}
			else if( strncmp( membuf, "ALL", 3)== 0 ){
				for( i= 0; i< setNumber; i++ ){
					setprop[i]= 1;
					hit+= 1;
				}
			}
			else if( strncmp( membuf, "REVERSE", 7)== 0 ){
				for( i= 0; i< setNumber; i++ ){
					setprop[i]= !setprop[i];
					hit+= 1;
				}
			}
			else if( strncmp( membuf, "ASSOC", 5)== 0 ){
			  int N= -1;
			  char *TBh= TBARprogress_header;
				strcat( membuf, "\n");
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( DumpPretty ){
							*LineDumped= True;
							print_string( stdout, "*DRAW_SET*", "\\n\n", "\n", membuf );
						}
					}
				pat= &membuf[5];
				while( isspace( (unsigned char) *pat ) ){
					pat++;
				}
				clean_param_scratch();
				TBARprogress_header= "*DRAW_SET*/*MARK_SET*/*PROCESS_SET* ASSOC";
				N= new_param_now( pat, param_scratch, N );
				if( N> 0 ){
				  DataSet *set;
					ReadData_proc.param_range[0] = param_scratch[0];
					if( setprop== lwi->draw_set ){
						for( i= 0; i< setNumber; i++ ){
							setprop[i]= 0;
						}
					}
					for( i= 0; i< setNumber; i++ ){
						set= &AllSets[i];
						if( set->numAssociations>= N ){
						  int j, ok= 1;
							for( j= 0; ok && j< MIN(N, set->numAssociations); j++ ){
								if( set->Associations[j]!= param_scratch[j] ){
									ok= 0;
								}
							}
							if( ok ){
								setprop[i]= 1;
								hit+= 1;
							}
						}
					}
				}
				else{
					fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\": missing or invalid argument(s) assoc[%d]={",
						filename, sub_div, line_count, buffer, N
					);
					fprintf( StdErr, "%s", d2str( param_scratch[0], NULL, NULL) );
					for( i= 1; i< N; i++ ){
						fprintf( StdErr, ",%s", d2str( param_scratch[i], NULL, NULL) );
					}
					fputs( "}\n", StdErr );
				}
				TBARprogress_header= TBh;
			}
			else{
			  char *TBh= TBARprogress_header;
				strcat( membuf, "\n");
					if( DumpFile ){
						if( *DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							*DF_bin_active= False;
						}
						if( DumpPretty ){
							*LineDumped= True;
							print_string( stdout, "*DRAW_SET*", "\\n\n", "\n", membuf );
						}
					}
				ReadData_proc.param_range[0]= 0;
				TBARprogress_header= "*DRAW_SET*/*MARK_SET*";
				new_param_now( membuf, &ReadData_proc.param_range[0], 1 );
				set_nr= (int) ReadData_proc.param_range[0];
				if( setprop== lwi->draw_set ){
					for( i= 0; i< setNumber; i++ ){
						setprop[i]= 0;
					}
				}
				if( set_nr>= 0 && set_nr< setNumber ){
					setprop[set_nr]= 1;
					hit= 1;
				}
				else{
					fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\": missing or invalid argument(s) or invalid specified set (%d)\n",
						filename, sub_div, line_count, buffer, set_nr
					);
				}
				TBARprogress_header= TBh;
			}
			if( hit ){
				if( setprop== lwi->draw_set ){
					if( proc ){
					  extern int DetermineBoundsOnly;
					  int dbo= DetermineBoundsOnly, ut= lwi->use_transformed;
						DetermineBoundsOnly= True;
						  /* The goal is to process. So temporarily disable quick-mode.	*/
						lwi->use_transformed= False;
						RedrawNow( lwi );
						lwi->use_transformed= ut;
						DetermineBoundsOnly= dbo;
						if( restore ){
							  /* Restore the current draw_set information.	*/
							for( i= 0; i< setNumber; i++ ){
								setprop[i]= SetProp[i];
							}
						}
					}
					else{
						lwi->FitOnce= True;
						RedrawNow( lwi );
					}
				}
				if( lwi->SD_Dialog ){
					set_changeables(2,False);
				}
				ActiveWin= lwi;
				if( debugFlag || scriptVerbose ){
					fprintf( StdErr,
						"ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: %d sets matched&%s in window 0x%lx %02d:%02d:%02d\n",
						filename, sub_div, line_count, buffer,
						hit,
						(setprop== lwi->draw_set)?
							(proc)? "processed" : "drawn" :
							"marked",
						ActiveWin,
						ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
					);
				}
			}
			else{
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: not found in window 0x%lx %02d:%02d:%02d\n",
					filename, sub_div, line_count, buffer,
					ActiveWin,
					ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
				);
				fflush( StdErr );
			}
		}
		else if( debugFlag ){
			if( ActiveWin && ActiveWin!= &StubWindow ){
				ascanf_window= ActiveWin->window;
			}
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\": missing argument(s)\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DRAWN_TITLES*", 14)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int i;
			for( i= 0; i< setNumber; i++ ){
				if( lwi->draw_set[i] && AllSets[i].titleText ){
					fprintf( StdErr, "%d::%s\n", i, AllSets[i].titleText );
				}
			}
			ActiveWin= lwi;
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*DRAWN_LEGENDS*", 15)== 0 ){
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  int i;
			for( i= 0; i< setNumber; i++ ){
				if( lwi->draw_set[i] && AllSets[i].setName ){
					fprintf( StdErr, "%d::%s\n", i, AllSets[i].setName );
				}
			}
			ActiveWin= lwi;
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*STDOUT*", 8)== 0
		|| strncmp( optbuf, "*STDERR*", 8)== 0
	){
	  char *membuf= cleanup( &optbuf[8] );
	  int redir= change_stdfile( membuf, (optbuf[4]== 'O')? stdout : StdErr );
		if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: %s\n",
				filename, sub_div, line_count, buffer, (redir)? " output redirected to requested file" : " output to terminal"
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*KEY_EVAL*", 10)== 0 || strncmp( optbuf, "*KEY_PARAM_NOW*", 15)== 0 ){
	  char *argbuf= cleanup( &optbuf[ (optbuf[5]== 'E')? 10 : 15] );
	  int key= -1;
		if( argbuf[0] ){
		  char *pbuf= SubstituteOpcodes( argbuf, "*This-File*", filename, "*This-FileName*", basename(filename),
		  	"*This-FileDir*", dirname(filename),
		  	"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level, NULL );
		  char *defbuf= strstr( pbuf, "::");
				if( DumpFile ){
					if( *DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						*DF_bin_active= False;
					}
					if( !DumpPretty ){
						fputs( "*EXTRATEXT* ", stdout );
					}
					else{
						*LineDumped= True;
					}
					print_string( stdout, "*KEY_EVAL*", "\\n\n", "\n", argbuf);
/* 					if( !DumpPretty ){	*/
						fputc( '\n', stdout );
/* 					}	*/
				}
			if( defbuf ){
				if( pbuf[0]== '0' && tolower(pbuf[1])== 'x' ){
					if( sscanf( pbuf, "0x%x::", &key)!= 1 ){
						key= -1;
					}
				}
				else if( pbuf[1]== ':' && pbuf[2]== ':' ){
					key= defbuf[-1];
				}
				else{
				  int mask= 0, n= 0;
				  char *c= pbuf;
					while( c< defbuf && *c && n< 3 ){
						switch( *c ){
							case 'c':
							case 'C':
								mask|= 1 << 0;
								break;
							case 'm':
							case 'M':
								mask|= 1 << 1;
								break;
							case 's':
							case 'S':
								mask|= 1 << 2;
								break;
							default:
								break;
						}
						c++;
						n+= 1;
					}
					*defbuf= '\0';
					if( sscanf( c, "F%d", &n)== 1 || sscanf( c, "f%d", &n)== 1 ){
						if( n>= 1 && n<= 12 ){
							n-= 1;
							key= 8*256+ mask* 12+ n;
						}
					}
					else if( *c> 0 ){
						switch( mask ){
							case 0:
								key= *c;
								break;
							case 1:
								if( (*c= toupper(*c))>= 'A' ){
									key= *c- 'A'+ 1;
								}
								else{
									key= mask* 256+ *c;
								}
								break;
							case 4:
								key= toupper(*c);
								break;
							default:
								key= mask* 256+ toupper(*c);
								break;
						}
					}
					*defbuf= ':';
				}
			}
			if( key>= 0 && key< 8*(256+12) ){
				xfree( key_param_now[key] );
				if( defbuf[2] ){
					key_param_separator = ascanf_separator;
// 					key_param_now[key]= concat2( key_param_now[key], cleanup(&defbuf[2]), "\n", NULL);
					// 20101115: why append a newline??
					key_param_now[key]= concat2( key_param_now[key], cleanup(&defbuf[2]), NULL);
				}
			}
			GCA();
			if( pbuf!= argbuf ){
				xfree(pbuf);
			}
		}
		else{
			ListKeyParamExpressions( stdout, False );
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncmp( optbuf, "*ASK_EVAL*", 10)== 0 || strncmp( optbuf, "*ASK_PARAM_NOW*", 15)== 0 ){
	  char *argbuf= cleanup( &optbuf[ (optbuf[5]== 'E')? 10 : 15] );
	  int modal;
		if( strncasecmp( argbuf, "modal::", 7)== 0 ){
			argbuf+= 7;
			modal= True;
		}
		else{
			modal= False;
		}
		if( argbuf[0] ){
		  char *pbuf= SubstituteOpcodes( argbuf, "*This-File*", filename, "*This-FileName*", basename(filename),
		  	"*This-FileDir*", dirname(filename),
		  	"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level, NULL );
		  char *defbuf= strstr( pbuf, "::"), *wid;
		  ALLOCA( name, char, LMAXBUFSIZE+2, name_len);
				if( DumpFile ){
					if( *DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						*DF_bin_active= False;
					}
					if( !DumpPretty ){
						fputs( "*EXTRATEXT* ", stdout );
					}
					else{
						*LineDumped= True;
					}
					print_string( stdout, "*ASK_EVAL*", "\\n\n", "\n", argbuf);
/* 					if( !DumpPretty ){	*/
						fputc( '\n', stdout );
/* 					}	*/
				}
			if( defbuf && defbuf[2] ){
				*defbuf= '\0';
				defbuf= String_ParseVarNames( XGstrdup( &defbuf[2] ), "${", '}', True, False, "*ASK_EVAL*" );
			}
			else{
				defbuf= NULL;
			}
			if( ActiveWin || InitWindow || (wid= cgetenv("WINDOWID")) ){
			  int tilen= -1;
			  LocalWin *wi= (ActiveWin)? ActiveWin : InitWindow;
				if( defbuf ){
					strncpy( name, defbuf, LMAXBUFSIZE);
					tilen= 1.5* strlen(name);
				}
				else{
					name[0]= '\0';
				}
				name[LMAXBUFSIZE]= '\0';
				if( wi ){
					TitleMessage( wi, pbuf );
					interactive_param_now( wi, name, tilen, LMAXBUFSIZE, pbuf, &ReadData_proc.param_range[0], modal, 0 );
					TitleMessage( wi, NULL );
				}
				else if( wid ){
				  Window win= atoi( wid);
					interactive_param_now_xwin( win, name, tilen, LMAXBUFSIZE, pbuf, &ReadData_proc.param_range[0], modal, 0, 0 );
				}
			}
			else{
				fflush( stdout );
				fflush( StdErr );
				xgiReadString2( name, LMAXBUFSIZE, pbuf, defbuf );
				cleanup( name );
				if( (!name[0] || name[0]== '\n') && defbuf ){
					strncpy( name, defbuf, LMAXBUFSIZE);
					name[LMAXBUFSIZE]= '\0';
				}
				if( name[0] ){
					strcat( name, "\n");
					new_param_now( name, param_scratch, -1 );
					ReadData_proc.param_range[0] = param_scratch[0];
				}
			}
			xfree( defbuf );
			if( pbuf!= argbuf ){
				xfree(pbuf);
			}
			GCA();
		}
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncasecmp( optbuf, "*CONSOLE*", 9)== 0 ){
		XG_SimpleConsole();
		ReadData_commands+= 1;
		r= 1;
	}
	else if( strncasecmp( optbuf, "*CROSS_FROMWIN_PROCESS*", 23)== 0 ){
	  char *membuf= &optbuf[23];
		while( isspace(*membuf) ){
			membuf++;
		}
		if( ActiveWin && ActiveWin!= &StubWindow ){
		  LocalWin *lwi= ActiveWin;
		  Cursor_Cross *s= &(lwi->curs_cross);
		  char *winbuf= strstr( membuf, "::");
		  LocalWindows *WL= WindowList;
		  char *TBh= TBARprogress_header;
			if( winbuf ){
				winbuf[0]= '\0';
				TBARprogress_header= "*CROSS_FROMWIN_PROCESS*";
				if( (WL= Find_WindowListEntry_String( membuf, NULL, NULL )) ){
					TBARprogress_header= "*CROSS_FROMWIN_PROCESS*";
					  /* That (lwi) window's curs_cross.fromwin must be set to the window reference
					   \ that we just parsed:
					   */
					lwi->curs_cross.fromwin= ActiveWin= WL->wi;
					ascanf_window= ActiveWin->window;
					xfree( s->fromwin_process.process );
					s->fromwin_process.separator = ascanf_separator;
					s->fromwin_process.process= XGstrdup( cleanup( &winbuf[2] ) );
					strncpy( s->fromwin_process.command, TBARprogress_header, sizeof(s->fromwin_process.command)-1 );
					new_process_Cross_fromwin_process( lwi );
					ActiveWin= lwi;
				}
				else{
					if( ActiveWin ){
						ascanf_window= ActiveWin->window;
					}
					fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"==%s: no such window belongs to xgraph\n",
						filename, sub_div, line_count, buffer, d2str( ReadData_proc.param_range[0], NULL, NULL)
					);
				}
			}
			else if( *membuf ){
				fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: format is <cursor-win>::<expression[2]>\n",
					filename, sub_div, line_count, buffer
				);
			}
			else{
				lwi->curs_cross.fromwin= NULL;
				xfree( s->fromwin_process.process );
				new_process_Cross_fromwin_process( lwi );
				ActiveWin= lwi;
			}
			TBARprogress_header= TBh;
		}
		else if( debugFlag ){
			fprintf( StdErr, "ReadData([Interactive_Commands()]%s,%d,%d) \"%s\"\t: no window active\n",
				filename, sub_div, line_count, buffer
			);
		}
		ReadData_commands+= 1;
		r= 1;
	}
	return(r);
}

#define RESETATTRS()	\
	if( ResetAttrs ){ \
		if( debugFlag ){ \
			fprintf( StdErr, \
				"ReadData(%s,l%d): line %d.%d: lw,ls,els,pv,ms=%g,%d,%d,%d,%d - resetting to %g,%d,%d,%d,%d\n", \
				filename, __LINE__, sub_div, line_count, \
				lineWidth, linestyle, elinestyle, pixvalue, markstyle, \
				lw, els, ls, pv, ms \
			); \
		} \
		lineWidth= lw; \
		linestyle= ls; \
		elinestyle= els; \
		pixvalue= pv; \
		markstyle= ms; \
		incr_width= -newfile_incr_width; \
		_incr_width= 0; \
	}

extern char *comment_buf;
extern int comment_size, NoComment;
extern int NextGen;

char *ReadData_buffer= NULL; int *ukl;

/* #define ALLOW_SOLITARY_COMMANDS	*/

  /* Two routines that determine when we go into "joining mode", and when to exit.
   \ We start the mode after a certain number of *COMMAND*s, when the command is
   \ directly followed by a "\\n" sequence followed by a newline. And when this
   \ does not occur inside a string.
   \ <buf> contains the currently read line, <first> the first non-whitespace char,
   \ dlen is buf's length, and instring signals whether or not we're in a string.
   */
char XGJoinBytes1[]= "dDPASKECBI", XGJoinBytes2[]= "AaRUSEVTOFLXY";
XGStringList *EndTags= NULL;

int XGStartJoin( char *buf, char *first, int dlen, int instring )
{ char *c1, *c2;
  int r= (!instring && *buf== '*' && dlen>=4 &&
		(c1= index( XGJoinBytes1, buf[1] )) && (c2= index( XGJoinBytes2, buf[2])) && buf[dlen-4]== '*' && buf[dlen-3]=='\\'
	);
#ifdef ALLOW_SOLITARY_COMMANDS
	if( !r && !instring && c1 && c2 && buf[dlen-2]== '*' && buf[dlen-1]== '\n' &&
		strncmp(buf, "*ECHO*", 6) &&
		strncmp(buf, "*POPUP*", 7) &&
		strncmp(buf, "*DEICON", 7)
	){
		r= 1;
		if( debugFlag || scriptVerbose ){
			fprintf( StdErr, "! Line-joining mode activated by solitary command %s", buf );
		}
	}
#else
	  /* Checks for **tag*\n format : */
	if( !r && !instring && *buf== '*' && buf[1]== '*' ){
		buf++;
		dlen-= 1;
		if( (c1= index( XGJoinBytes1, buf[1] )) && (c2= index( XGJoinBytes2, buf[2])) &&
			  /* 20020506: for the time being, we still require that the **COMMAND* version
			   \ be solitary on its own line...
			   */
			buf[dlen-2]== '*' && buf[dlen-1]== '\n' &&
			  /* exceptions: */
			strncmp(buf, "*ECHO*", 6) &&
			strncmp(buf, "*POPUP*", 7) &&
			strncmp(buf, "*DEICON", 7)
		){
			r= 1;
			if( buf[dlen-3]== '*' ){
				if( strncmp( buf, "*IF*", 4) &&
					strncmp( buf, "*ELSE*", 6) &&
					strncmp( buf, "*ENDIF*", 7)
				){
				  __ALLOCA( endtag, char, dlen+3, etlen );
				  char *c= &buf[1], *d= &endtag[2];
					endtag[0]= '*';
					endtag[1]= '!';
					while( *c && *c!= '*' ){
						*d++= *c++;
					}
					*d++= '*';
					*d++= '\n';
					*d= '\0';
					StringCheck( endtag, etlen, __FILE__, __LINE__);
					EndTags= XGStringList_AddItem( EndTags, endtag );
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr, "! Line-joining mode activated by command *%s - reading until %s", buf, endtag );
					}
					GCA();
					  /* 20040224: ATTN: temporary. The "**tag**" ... "*!tag*" syntax is currently transparent
					   \ to the user code (i.e. should not be seen!)
					   */
					buf[dlen-2]= ' ';
					  /* Return 2 here: this signals that we should not switch out of joining mode on emtpy lines */
					r= 2;
				}
				else{
					buf[dlen-2]= ' ';
				}
			}
			else if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "! Line-joining mode activated by command *%s", buf );
			}
		}
	}
#endif
	return( r );
}

int XGEndJoin( char *buf, char *first, int dlen, int instring )
{
	if( !instring && EndTags && EndTags->last ){
	  int r;
		  /* We exit only when we find the appropriate tag. */
		if( strcmp( EndTags->last->text, buf )== 0 ){
			  /* 20040224: remove the ending tag from the output!! */
			buf[0]= '\n';
			buf[1]= '\0';
			EndTags= XGStringList_PopLast( EndTags );
			r= 1;
		}
		else{
			r= 0;
		}
		return(r);
	}
	else{
	  /* We exit on a blank line not in a string.  */
		return( !instring && (*first== '\0' || *first== '\n') );
	}
}

int ReadData_BufChanged= False;
char *ReadData_AllocBuffer( char *buffer, int length, char *the_file )
{ char *b= NULL;
	if( !buffer ){
		if( !(b= ( calloc( 2*length, sizeof(char) ) )) ){
			fprintf( StdErr, "ReadData(%s): can't allocate %d character inputbuffer... (%s)\n",
				the_file, LMAXBUFSIZE+2, serror()
			);
		}
	}
	else{
		if( !(b= ( realloc( buffer, 2*length* sizeof(char) ) )) ){
			fprintf( StdErr, "ReadData(%s): can't re-allocate %d character inputbuffer... (%s)\n",
				the_file, LMAXBUFSIZE+2, serror()
			);
		}
		ReadData_BufChanged= True;
	}
	return( b );
}

typedef struct IF_Frames{
	char *expr;
	int alternative, level;
	struct IF_Frames *next;
} IF_Frames;

IF_Frames *IF_Frame_Delete( IF_Frames *list, int All )
{ IF_Frames *last= list;
	if( All ){
		while( last ){
			list= last;
			last= list->next;
			xfree( list->expr );
			xfree( list );
		}
	}
	else{
		last= list->next;
		xfree( list->expr );
		xfree( list );
		list= last;
	}
	return( list );
}

IF_Frames *IF_Frame_AddItem( IF_Frames *list, char *expr, int level )
{ IF_Frames *new;
	if( (new= (IF_Frames*) calloc( 1, sizeof(IF_Frames) )) ){
		new->expr= XGstrdup(expr);
		new->next= NULL;
		new->level= level;
		if( (/* last= */ list) ){
			new->next= list;
			list= new;
/* 			}	*/
/* 			if( last ){	*/
/* 				last->next= new;	*/
/* 			}	*/
		}
		else{
			list= new;
		}
	}
	return( list );
}

/* A brute force method to correctly update the current linenumber in the open file. 	*/
int UpdateLineCount( FILE *fp, FileLinePos *flpos, int line_count )
{ long fpos= ftell(fp), read= 0;
  char c;

	if( flpos->last_line_count== -1 || fp== ReadPipe_fp || fp== stdin ){
		  /* not a seekable file */
		return(line_count);
	}
	  /* Only do something when the current position is not equal to the last position
	   \ where we verified the line_count.
	   */
	if( flpos->last_fpos!= fpos ){
	  long N;
#ifdef DEBUG
	  long nfpos;
#endif
		if( fpos< flpos->last_fpos ){
			  /* File has been rewound. Re-determine the line_count.	*/
			fseek( fp, 0, SEEK_SET );
			flpos->last_line_count= 0;
			flpos->last_fpos= 0;
			N= fpos;
		}
		else{
			  /* Seek back to where we were last time, and determine how many characters (N) should
			   \ be read.
			   */
			fseek( fp, flpos->last_fpos, SEEK_SET );
			N= fpos- flpos->last_fpos;
		}
		line_count= flpos->last_line_count;
		  /* Get a first character	*/
		c= fgetc( fp );
		read+= 1;
		if( c== '\n' ){
			line_count+= 1;
		}
		while( !feof(fp) && !ferror(fp) && read< N ){
			  /* read up to the current file position (the one the file was at when we entered this routine),
			   \ incrementing line_count for each newline encountered. Normally, one would test c!= EOF. However,
			   \ since XGraph files can be binary, it is fully possible to encounter this situation before
			   \ the actual EOF is reached.
			   */
			c= fgetc( fp );
			read+= 1;
			if( c== '\n' ){
				line_count+= 1;
			}
		}
		  /* store the value and position we've arrived at for the next call. */
		flpos->last_line_count= line_count;
		flpos->last_fpos+= read;
#ifdef DEBUG
		nfpos= ftell(fp);
		if( fpos!= nfpos ){
			fprintf( StdErr, "UpDateLineCount(): returning at a different position than entered (%ld!=%ld) - corrected\n",
				nfpos, fpos
			);
		}
#endif
		fseek( fp, fpos, SEEK_SET );
	}
	return( line_count );
}

#define RD_RETURN(val)	ReadData_level-= 1; IF_Frame= IF_Frame_Delete(IF_Frame,True);\
	xfree(BUFFER); TBARprogress_header= NULL; ascanf_exit= 0;\
	xfree(extremes);\
	return(val)

char *xgiReadString( char *buffer, int len, FILE *fp, int *line,
	DEFMETHOD( StartJoining, (char *buf, char *first_nspace, int len, int instring), int ),
	DEFMETHOD( EndJoining, (char *buf, char *first_nspace, int len, int instring), int )
)
{
	if( fp== stdin && Use_ReadLine && isatty(fileno(stdin)) ){
		return( ReadLine( buffer, len, "#xgraph-data> ", line, StartJoining, EndJoining ) );
	}
	else{
		return( ReadString( buffer, len, fp, line, StartJoining, EndJoining ) );
	}
}

#ifdef DEBUG
/* extern size_t fread( void *ptr, size_t size, size_t nmemb, FILE *stream);	*/

size_t XGfread( void *ptr, size_t size, size_t nmemb, FILE *stream)
{
	return( fread( ptr, size, nmemb, stream ) );
}
#define fread	XGfread
#endif

#ifdef basename
#	undef basename

	char *XGbasename( char *path )
	{ static char basenameBuffer[MAXPATHLEN+1];
		strncpy( basenameBuffer, path, MAXPATHLEN );
		basenameBuffer[MAXPATHLEN] = '\0';
		return basename(basenameBuffer);
	}
#	define basename(p)	XGbasename(p)
#endif // basename
#ifdef dirname
#	undef dirname

	char *XGdirname( char *path )
	{ static char dirnameBuffer[MAXPATHLEN+1];
		strncpy( dirnameBuffer, path, MAXPATHLEN );
		dirnameBuffer[MAXPATHLEN] = '\0';
		return dirname(dirnameBuffer);
	}
#	define dirname(p)	XGdirname(p)
#endif // dirname

char *skip_to_label= NULL;

/* A little stub for calling a DM_IO module's import handler. Mostly to be able
 \ to set a meaningful breakpoint before the module is loaded, while debugging.
 */
int dm_io_import( DM_IO_Handler *dm_io, FILE *rfp, char *sFname, int filenr,
	int setNumber, DataSet *this_set, ReadData_States *state )
{
	return( (*dm_io->import)( rfp, sFname, filenr, setNumber, this_set, state ) );
}

typedef struct ReadData_Pars{
	char *filename;
	FILE *stream;
	int filenr, swap_endian;
} ReadData_Pars;

int IOImportedFiles= 0;

int doIO_Import( const char *libname, char *fname, ReadData_Pars *currentFile, ReadData_States *state, DataSet *this_set )
{ char sFname[MAXPATHLEN];
  int l_is_pipe= 0, S_E= SwapEndian, filenr, ret= 0;
  FILE *rfp= NULL;
  DyModLists *dm_ioLib;
  DM_IO_Handler *dm_io;

	if( (dm_ioLib= LoadDyMod( (char*) libname, RTLD_LAZY|RTLD_GLOBAL, True, False ))
		&& dm_ioLib->type== DM_IO && (dm_io= dm_ioLib->libHook) && dm_io->type== DM_IO && dm_io->import
	){
		if( fname && *fname ){
			if( fname[0]== '|' ){
				fname++;
				PIPE_error= False;
				if( (rfp= popen( fname, "r" )) ){
					add_process_hist( &fname[-1] );
				}
				strncpy( sFname, fname, MAXPATHLEN-1 );
				l_is_pipe= 1;
			}
			else{
				tildeExpand( sFname, fname );
				if( (rfp= fopen( sFname, "r" )) ){
					add_process_hist( fname );
					IdentifyStreamFormat( sFname, &rfp, &l_is_pipe );
				}
			}
			if( currentFile ){
				filenr= currentFile->filenr;
			}
			else{
				filenr= -1;
			}
		}
		else if( currentFile ){
			rfp= currentFile->stream;
			filenr= currentFile->filenr;
			strncpy( sFname, currentFile->filename, MAXPATHLEN-1 );
		}
		if( rfp ){
		  int (*__dm_io_import)( DM_IO_Handler *dm_io, FILE *rfp, char *sFname, int filenr,
			  int setNumber, DataSet *this_set, ReadData_States *state );
			if( currentFile ){
				SwapEndian= SwapEndian || currentFile->swap_endian;
			}
			  /* Kludge to be sure dm_io_import will NOT be called as an line function,
			   \ so that we can breakpoint it even when this file is not compiled with debugging
			   \ info....
			   */
			__dm_io_import= dm_io_import;
			(*__dm_io_import)( dm_io, rfp, sFname, filenr, setNumber, this_set, state );
			if( this_set->numPoints> maxitems ){
				maxitems= this_set->numPoints;
			}
			if( this_set->ncols> MaxCols ){
				MaxCols= this_set->ncols;
				if( BinaryFieldSize ){
					AllocBinaryFields( MaxCols, "ReadData()" );
				}
			}
			SwapEndian= S_E;
			IOImportedFiles+= 1;
			if( !currentFile || rfp!= currentFile->stream ){
				if( l_is_pipe ){
					pclose( rfp );
					if( l_is_pipe!= 2 ){
						--fname;
					}
				}
				else{
					fclose( rfp );
				}
			}
			ret= 0;
		}
		else{
			ret= 2;
		}
	}
	else{
		ret= 1;
	}
	return( ret );
}

int ReadData(FILE *stream, char *the_file, int filenr)
/*
 * Reads in the data sets from the supplied stream.  If the format
 * is correct,  it returns the current maximum number of points across
 * all data sets.  If there is an error,  it returns -1.
 */
{ char *BUFFER= NULL;
  char *optbuf= BUFFER, *buffer= BUFFER, *rsb, *IWname= NULL;
  double _incr_width= 0;
  int spot = 0, Spot= 0, sub_div= 0, numcoords, optcol;
  DataSet *this_set= &AllSets[setNumber];
  double data[ASCANF_DATA_COLUMNS];
  static int called= 0, prev_filenr= 0, restarted= False, s_e;
  static double incr_width= -1;
  int splittable_file= -1;
  char *filename;
  int param_ok;
  double lw= lineWidth;
  int ls= linestyle, els= elinestyle, pv= pixvalue, ms= markstyle;
  int fsize= 0, perc= 0, first_set= setNumber, total_sets= -1, set_progress= 0, show_perc_now= False;
  Time_Struct progress_timer;
  long fpos= 0;
  Boolean xterm= False, LineDumped= False, DF_bin_active= False, read_data= False, set_set= False,
  	swap_endian= False, s_e_set= False, seekable= True;
  int spec_set, real_set;
  static char *read_file_hist= NULL;
  char *Read_File_Hist= NULL;
  char *read_file_buf= NULL;
  int read_file_point= 0;
  int Skip_new_line= 0, Skipped_new_line= 0, endianness, endnmesg_shown= 0, unknown_label= False;
  int rd_bufsize= 0;
  IF_Frames *IF_Frame= NULL;
  int IF_level= 0;
  FileLinePos flpos;
  int doRead= True;
  XGStringList **init_exprs= NULL;
  int *new_init_exprs= NULL;
  double *extremes= NULL;
  int extremesN= 0;
  char ReadData_level_str[16];
  static int ReadData_level= -1;

	if( !(BUFFER= ReadData_AllocBuffer( BUFFER, LMAXBUFSIZE+2, the_file)) ){
		return(0);
	}

#define ReadData_RETURN(x)	{ ReadData_level-= 1; RESETATTRS(); IF_Frame= IF_Frame_Delete( IF_Frame, True ); return(x); }

	ReadData_level+= 1;
	snprintf( ReadData_level_str, sizeof(ReadData_level_str)/sizeof(char), "%d", ReadData_level );

	optbuf= BUFFER;
	buffer= BUFFER;
	rd_bufsize= LMAXBUFSIZE+2;
	if( restarted ){
	  /* Set here those things that must be changed, reset or whatever when we
	   \ restart (e.g. when we encounter a *BUFLEN* statement).
	   */
		swap_endian= s_e;
		restarted= False;
	}
	else{
	  /* And here those things that must be changed, reset or whatever when we
	   \ are a fresh invocation.
	   */
	}

	ReadData_buffer= BUFFER; ukl= &unknown_label;

	clean_param_scratch();

	line_count = 0;
	flpos.last_fpos= 0;
	flpos.last_line_count= 0;
	errno= 0;
	if( (ReadPipe && ReadPipe_fp) || is_pipe || stream== stdin ){
		  seekable= False;
	}
	else{
	  struct stat sb;
		  /* 20040922: Mac OS X needs explicit tests: the fseek( ftell()) test below doesn't fire on fifos etc. */
		if( fstat( fileno(stream), &sb) ){
			fprintf( StdErr, "ReadData(): can't fstat \"%s\"(%s)\n", the_file, serror() );
			seekable= False;
		}
		else{
			fsize= sb.st_size;

			if( !( S_ISREG(sb.st_mode)
#ifdef S_ISLNK
					|| S_ISLNK(sb.st_mode)
#endif
				)
			){
				if( debugFlag || scriptVerbose ){
					fprintf( StdErr, "ReadData(): \"%s\" is not likely to be seekable: won't even try.\n"
						" Make sure your data is in proper and uncompressed (or whatever) form!\n",
						the_file
					);
				}
				seekable= False;
			}
		}
		if( seekable && (fseek( stream, ftell(stream), SEEK_SET ) || errno== EBADF) ){
			if( debugFlag || scriptVerbose || WarnNewSet ){
				fprintf( StdErr, "ReadData(): warning: can't seek on \"%s\" (%s): accurate line numbers are not guaranteed!\n",
					the_file, serror()
				);
			}
			seekable= False;
		}
	}
	if( !seekable ){
		flpos.last_fpos= -1;
		flpos.last_line_count= -1;
	}
	flpos.stream= stream;

	split_set= 0;

	Elapsed_Since( &progress_timer, True );
	if( filenr>= 0 ){
		if( !called ){
		  int i;
			for( i= 0; i< MAXSETS; i++){
				this_set= &AllSets[i];
				this_set->set_nr= i;
				this_set->lineWidth= lineWidth;
				this_set->elineWidth= errorWidth;
				this_set->linestyle= linestyle;
				this_set->pixvalue= pixvalue;
				this_set->pixelValue= AllAttrs[pixvalue % MAXATTR].pixelValue;
				xfree( this_set->pixelCName );
				this_set->pixelCName= XGstrdup( AllAttrs[pixvalue % MAXATTR].pixelCName );
/* 				this_set->markstyle= markstyle;	*/
				this_set->has_error= 0;
				this_set->use_error= use_errors;
				this_set->numErrors= 0;
				this_set->markFlag= markFlag;
				this_set->noLines= noLines;
				this_set->floating= False;
				this_set->pixelMarks= pixelMarks;
				this_set->barFlag= barFlag;
				this_set->polarFlag= polarFlag;
				this_set->radix= radix;
				this_set->radix_offset= radix_offset;
				this_set->vectorLength= vectorLength;
				this_set->vectorType= vectorType;
				memcpy( this_set->vectorPars, vectorPars, MAX_VECPARS* sizeof(double));
				this_set->draw_set= 1;
				this_set->show_legend= 1;
				this_set->show_llines= 1;
			}
			memset( &ReadData_proc, 0, sizeof(Process) );
		}
		called+= 1;
	}

	if( debugFlag ){
		fprintf( StdErr, "lw,ls,ew,els,pv,ms=%g,%d,%g,%d,%d,%d\n",
			lineWidth, linestyle, errorWidth, elinestyle, pixvalue, markstyle
		);
	}

      /* Eliminate over-zealous set increments */
    if (setNumber > 0) {
		if( AllSets[setNumber-1].numPoints <= 0 && AllSets[setNumber-1].set_link< 0 ){
		  int r;
			setNumber--;
			r= ReadData(stream, the_file, filenr );
			RESETATTRS();
			RD_RETURN(r);
		}
    }

	if( !the_file ){
		filename= XGstrdup("stdin");
	}
	else{
		filename= XGstrdup( the_file );
	}
	{ /* struct stat stats; */
	  char *c= cgetenv( "TERM" );
		if( !strncasecmp( c, "xterm", 5 ) || !strncasecmp( c, "cygwin", 6 ) ){
			xterm= True;
		}
#if 0
		if( !fstat( fileno(stream), &stats) ){
		 /* 20040922: done while checking for seekable */
			fsize= stats.st_size;
		}
#endif
	}

	this_set= &AllSets[setNumber];

	add_comment( today() );

	sprintf( buffer, "\\#xa6\\ ");
	time_stamp( stream, filename, &buffer[strlen(buffer)], 1, "\n"  );
	strcat( buffer, ":\n");
	StringCheck( buffer, LMAXBUFSIZE+1, __FILE__, __LINE__);
	add_comment( buffer );
	buffer[0]= '\0';

	if( !this_set->propsSet ){
		this_set->fileNumber= filenr+ 1+ file_splits;
		this_set->new_file= 1;
		this_set->lineWidth= lineWidth;
		this_set->linestyle= linestyle;
		this_set->elineWidth= errorWidth;
		this_set->elinestyle= elinestyle;
		this_set->pixvalue= pixvalue;
		this_set->pixelValue= AllAttrs[pixvalue % MAXATTR].pixelValue;
		xfree( this_set->pixelCName );
		this_set->pixelCName= XGstrdup( AllAttrs[pixvalue % MAXATTR].pixelCName );
		this_set->markstyle= markstyle;
		this_set->NumObs= NumObs;
		this_set->vectorLength= vectorLength;
		this_set->vectorType= vectorType;
		memcpy( this_set->vectorPars, vectorPars, MAX_VECPARS* sizeof(double));
	}

	if( debugFlag ){
		fprintf( StdErr, "ReadData(): reading file #%d \"%s\"(%d), set #%d, width %g, incr_width=%g, maxlbuf=%d\n",
			filenr, this_set->fileName, (int) this_set->new_file, setNumber, this_set->lineWidth,
			incr_width, LMAXBUFSIZE
		);
	}

	ascanf_exit= 0;
	if( ActiveWin ){
		ActiveWin->halt= 0;
	}

	if( InitWindow ){
		if( !XFetchName( disp, InitWindow->window, &IWname ) ){
			IWname= NULL;
		}
	}

    while( setNumber < MAXSETS && !ascanf_exit && doRead ){
	  extern LocalWin *theSettingsWin_Info;
	  Boolean do_perc= seekable && !is_pipe && stream!= stdin && !(ReadPipe && ReadPipe_fp);

		if( ReadData_BufChanged || rd_bufsize!= LMAXBUFSIZE+2 ){
			  /* 20010711: Why would I unset it, and then set it in the ReadData_AllocBuffer() procedure?! */
/* 			ReadData_BufChanged= False;	*/
			if( !(BUFFER= ReadData_AllocBuffer( BUFFER, LMAXBUFSIZE+2, the_file)) ){
				ReadData_RETURN(0);
			}
			ReadData_BufChanged= False;
			  /* Got the memory; update the necessary variables!!	*/
			rd_bufsize= LMAXBUFSIZE+2;
			optbuf= BUFFER + (optbuf-buffer);
			buffer= BUFFER;
		}

		if( do_perc ){
			fpos= ftell(stream);
		}
		if( fsize && do_perc ){
		  double p= ((double) fpos/ ((double) fsize) * 100);
			if( p>= perc || show_perc_now ){
				Elapsed_Since( &progress_timer, False );
				if( progress_timer.Tot_Time>= Progress_ThresholdTime || perc== 0 ){
				  char buf[MAXPATHLEN+256];
					line_count= UpdateLineCount( stream, &flpos, line_count );
					sprintf( buf, "%s: %s%% (%ld of %d): line %d.%d",
						filename, d2str(p, "%g", NULL), fpos, fsize, sub_div+ file_splits, line_count
					);
					if( theSettingsWin_Info ){
						XStoreName( disp, theSettingsWin_Info->SD_Dialog->win, buf );
						xtb_XSync( disp, False );
					}
					else{
						if( InitWindow ){
							XStoreName( disp, InitWindow->window, buf );
							if( SetIconName ){
								XSetIconName( disp, InitWindow->window, buf );
							}
							/* if( InitWindow->mapped && !RemoteConnection ) */{
								XFlush( disp );
							}
						}
						if( xterm ){
							fprintf( stderr, "%c]0;%s%c", 0x1b, buf, 0x07 );
							fflush( stderr );
						}
						else{
#if 0
							if( !ReadData_terpri ){
								fprintf( StdErr, "%s: %2.0f%%", filename, p );
							}
							else{
								fprintf( StdErr, " %2.0f%%", p );
							}
							ReadData_terpri= True;
#else
							fprintf( StdErr, "\r%s", buf );
							ReadData_terpri= (fpos>=fsize)? True : False;
#endif
							fflush( StdErr );
						}
					}
					Elapsed_Since( &progress_timer, True );
				}
				perc+= 10;
			}
		}
		else{
		  int snr= this_set->set_nr- first_set;
			Elapsed_Since( &progress_timer, False );
			if( line_count== 0 || line_count>= perc ||
				(set_set && total_sets> 0 && snr> set_progress) || show_perc_now
			){
				if( progress_timer.Tot_Time>= Progress_ThresholdTime || perc== 0 ){
				  char buf[512];
					line_count= UpdateLineCount( stream, &flpos, line_count );
					if( set_set && total_sets> 0 ){
						sprintf( buf, "%s: line %d.%d, set #%d of %d; %s",
							(is_pipe)? "(pipe)" : the_file, sub_div+ file_splits, line_count, snr, total_sets, filename
						);
					}
					else{
						sprintf( buf, "%s: line %d.%d, set #%d; %s",
							(is_pipe)? "(pipe)" : the_file, sub_div+ file_splits, line_count, snr, filename
						);
					}
					if( theSettingsWin_Info ){
						XStoreName( disp, theSettingsWin_Info->SD_Dialog->win, buf );
						xtb_XSync( disp, False );
					}
					else{
						if( InitWindow ){
							XStoreName( disp, InitWindow->window, buf );
							if( SetIconName ){
								XSetIconName( disp, InitWindow->window, buf );
							}
							/* if( InitWindow->mapped && !RemoteConnection ) */{
								XFlush( disp );
							}
						}
						if( xterm ){
							fprintf( stderr, "%c]0;%s%c", 0x1b, buf, 0x07 );
							fflush( stderr );
						}
					}
					Elapsed_Since( &progress_timer, True );
				}
				perc+= 1000;
				set_progress= snr;
			}
		}
		show_perc_now= False;

		this_set= &AllSets[setNumber];
		*ascanf_setNumber= setNumber;
		*ascanf_TotalSets= setNumber;
		*ascanf_numPoints= this_set->numPoints;
		if( !this_set->fileName || strcmp( this_set->fileName, filename ) ){
			xfree( this_set->fileName );
			this_set->fileName= XGstrdup(filename);
		}
		fileNumber= this_set->fileNumber= filenr+ 1+ file_splits;
		if( plot_only_file && filenr+1+file_splits != plot_only_file ){
			this_set->draw_set= 0;
			if( debugFlag && this_set->numPoints<= 0 ){
				fprintf( StdErr, "ReadData(): set #%d in file #%d NOT shown initially\n",
					setNumber, filenr+1
				);
			}
		}

		if( skip_to_label && !the_init_exprs && strncmp( buffer, "*INIT_BEGIN*", 12) ){
			strncpy( buffer, "*SKIP_TO*", LMAXBUFSIZE-1 );
			strncat( buffer, skip_to_label, LMAXBUFSIZE- strlen(buffer)-1 );
			xfree( skip_to_label );
			goto skip_to_label;
		}

		if( !Skip_new_line ){
			optbuf= buffer;
			if( !(rsb= xgiReadString(buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin ))){
				  /* 990715: ReadString decrements line_count and resets
				   \ it if a line is read. I don't remember why, but I
				   \ think that at EOF, it is safe to reset line_count
				   \ (because ReadString hasn't), and otherwise we'd get
				   \ errormessages if only one line has been read before.
				   */
				if( feof(stream) ){
					line_count+= 1;
				}
				break;
			}
			LineDumped= False;
			Skipped_new_line= False;
		}
		else{
			Skipped_new_line= True;
			Skip_new_line= False;
		}

#ifdef XG_DYMOD_SUPPORT
		if( line_count== 0
#ifdef __CYGWIN__
			&& strlen(buffer)>= 3
#else
			&& strlen(buffer)>= 4
#endif
		){
		  uint32_t mask, rawmask;
			rawmask = mask= *((unsigned long*)buffer);
			if( EndianType== 1 ){
				SwapEndian_int32( (int32_t*)&mask, 1);
			}
			if( (
				strncmp( buffer, "\177ELF", 4)== 0
				  /* Mach-O object; see if it dlopens... */
				|| mask== 0xfeedface
				|| rawmask== 0xfeedface
				|| mask== 0xfeedfacf
				|| rawmask== 0xfeedfacf
				  /* Mach-O universal binary OR Java bytecode; see if it dlopens... */
				|| mask== 0xcafebabe
				|| rawmask== 0xcafebabe
#ifdef __CYGWIN__
				|| mask == 0x4d5a9000
#endif
			)){
#ifdef __CYGWIN__
			  int n= 150* sizeof(char)+ sizeof(short)+ 1, ok;
#else
			  int n= 16* sizeof(char)+ sizeof(short)+ 1, ok;
#endif
			  ALLOCA( lbuf, char, n, lbuf_len);
			  long pos= ftell(stream);
				rewind( stream );
				ok= fread( lbuf, sizeof(char), n, stream );
#ifdef __CYGWIN__DEBUG
				fprintf( stderr, "lbuf[128]=%c%c%d%d (short) lbuf[150]=%hd\n",
							lbuf[128], lbuf[129], lbuf[130], lbuf[131],
								(*((short*)&lbuf[150]))
				);
#endif
				if( ok &&
						   // shared or OpenBSD ELF object:
						( (*( (short*) &lbuf[16])== 3 || lbuf[7]== 0x12)
							|| mask==0xfeedface || rawmask==0xfeedface
							|| mask==0xfeedfacf || rawmask==0xfeedfacf
							|| mask==0xcafebabe || rawmask==0xcafebabe
#ifdef __CYGWIN__
							|| ( lbuf[128]=='P' && lbuf[129]=='E' && lbuf[130]=='\0' && lbuf[131]=='\0'
								&& ((*((short*)&lbuf[150])) & 0x2000) > 0
							)
#endif
						)
				){
				  int sv= scriptVerbose;
					fprintf( StdErr, "You're loading a shared library: will load it as a dynamic module!\n" );
					scriptVerbose= True;
					if( strcasestr( the_file, "Python" ) ){
						LoadDyMod( the_file, RTLD_NOW|RTLD_GLOBAL,  False, False );
					}
					else{
						LoadDyMod( the_file, RTLD_LAZY|RTLD_GLOBAL,  False, False );
					}
					scriptVerbose= sv;
					ReadData_RETURN(maxSize);
				}
				else{
					fseek( stream, pos, SEEK_SET );
				}
				GCA();
			}
		}
#endif

		optbuf= buffer;
		if( *optbuf && optbuf[ strlen(optbuf)-1]!= '\n'){
		  /* line too long	*/
		  int n= 0, c;
			if( (c=strlen(optbuf))< LMAXBUFSIZE ){
				optbuf[c-1]= '\n';
				optbuf[c]= '\0';
			}
			else{
				if( DumpFile && strncmp( optbuf, "*BINARYDATA*", 12) ){
					if( DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						DF_bin_active= False;
					}
					fputs( optbuf, stdout );
					fputc( '\n', stdout );
					LineDumped= True;
				}
			}
			do{
				c= getc(stream);
				n++;
			}
			while( c!= EOF && c!= '\n');
			fprintf( StdErr, "%s: line %d.%d too long (%d chars skipped from \"%s\"; set buffer to at least %d with *BUFLEN* or -maxlbuf)\n",
				filename,
				sub_div+ file_splits, line_count+ 1, n-1, optbuf, LMAXBUFSIZE+ n
			);
			fflush( StdErr );
		}
		line_count++;

		optcol= 0;

		  /* find out in what column an option entry is	*/
		while( *optbuf== '\t' && *optbuf!= '*' && *optbuf!= '\n' && *optbuf){
			optcol++;
			  /* 930930 was *optbuf++	*/
			optbuf++;
		}

		  /* 20020506: this is to support the switch-to-joining-mode extra *: */
		if( *optbuf== '*' && optbuf[1]== '*' ){
			optbuf++;
		}

		if( the_init_exprs && strncmp( optbuf, "*INIT_END*", 10) ){
		  XGStringList *new;
			if( (new= (XGStringList*) calloc( 1, sizeof(XGStringList) )) ){
				new->text= XGstrdup(buffer);
				new->separator = ascanf_separator;
				new->next= NULL;
				*the_init_exprs= new;
				the_init_exprs= &new->next;
				ReadData_commands+= 1;
				goto next_line;
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr,
					"ReadData(%s,%d,%d) \"%s\"\t: can't get memory to store INIT expression (%s)-- evaluating it now!\n",
					filename, sub_div, line_count, buffer, serror()
				);
			}
		}

		  /* 20000418: a heuristic to the parsing of an unknown label field. A field
		   \ can be followed by a line with another label, an empty line, or a line starting with
		   \ a number - all of which mean an attempt should be made to parse that line.
		   \ In all other cases, there is a big change that line following the unknown label
		   \ belongs to it, in one way or another. Notable exception: data that is specified
		   \ with an fascanf() function instead of a number... To alert the user to what is
		   \ going on, each line is printed, with a message whether it is accepted or not.
		   \ (Now only to get the line numbers correct...!)
		   */
		if( unknown_label ){
		  Boolean accept;
			if( !optbuf[0] || (optbuf[0]== '*' && IsLabel(optbuf)) || optbuf[0]== '\n' || index( "0123456789-.+", optbuf[0]) ){
				accept= True;
			}
			else{
				accept= False;
			}
			line_count= UpdateLineCount( stream, &flpos, line_count );
			fprintf( StdErr,
				"file: `%s', line: %d.%d: line %s after unknown label (%s)\n",
					filename, sub_div, line_count,
					(accept)? "accepted" : "rejected",
					buffer
			);
			if( accept ){
				unknown_label= False;
			}
			else{
				goto next_line;
			}
		}

		if( *optbuf!= '*'){
			optbuf= buffer;
			optcol= 0;
		}
		else if( debugFlag){
			fprintf( StdErr, "Option field in column %d\n", optcol);
		}

		  /* 990715: allow comments to start non left-flushed, and
		   \ accept lines with just whitespace as empty lines.
		   */
		{ char *first= buffer;
			while( *first && isspace((unsigned char) *first) && *first!= '\n' ){
				first++;
			}
			if( *first == '#' ){
				add_comment( buffer );
				goto next_line;
			}
			if( *first == '\n' ){
			  /* end of dataset: find next one skipping over
			   * empty lines
			   */
			  int c;
			  int sn= setNumber;
			  DataSet *ts= this_set;
				  /* 990715: kludge to simulate a really empty line...	*/
				buffer[0]= '\n';
				c= getc( stream);
				while( c!= EOF && c== '\n'){
					line_count++;
					c= getc(stream);
				}
				if( c!= EOF && c!= '\n' ){
					ungetc( c, stream);
				}
				else{
					line_count++;
				}
				  /* Empty line - increment data set
				   * if not the first one (allow empty header lines)
				   */
				if( NewSet( NULL, &this_set, spot ) == -1 ){
					if( ReadData_terpri ){
						fputs( "\n", StdErr );
						ReadData_terpri= False;
					}
					if( (debugFlag || scriptVerbose) && Read_File_Hist ){
						fprintf( StdErr, "Included files: %s\n", Read_File_Hist );
					}
					xfree( Read_File_Hist );
					RESETATTRS();
					RD_RETURN( -1 );
				}
				else if( setNumber> sn || this_set!= ts ){
					if( WarnNewSet ){
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf( StdErr, "#---- Starting new set #%d; file \"%s\", line %d.%d (ReadData())\n",
							setNumber, filename, line_count, sub_div
						);
					}
				}
				spot = 0;
				Spot = 0;
			}
		}

		if( read_file_buf ){
			if( this_set->read_file ){
				this_set->read_file= concat2( this_set->read_file, "\n\n", read_file_buf, NULL);
			}
			else{
				this_set->read_file= XGstrdup( read_file_buf );
			}
			read_file_buf= NULL;
			this_set->read_file_point= Spot;
		}

		if (buffer[0] == '\n') {
		  /* already handled once - test repeated because of insertion of the
		   \ above check for read_file_buf
		   \ 990715: since we need to pass that test (and I don't like replicating these
		   \ kind of tests for the sake of code-maintenance), we repeate this test,
		   \ which needn't take any action (at least..). To make it visually evident
		   \ that there ain't going to be done anything else for this (empty) line,
		   \ jump to the next_line: part. Note that this shouldn't make any difference
		   \ for the compiler/compiled code: what follows until that label is only
		   \ 'else if's (the tests of neither of which should actually be performed!).
		   */
			goto next_line;
		}
		else if (buffer[0] == '"') {
		  int len= strlen(buffer)-1;
			if( len> 0)
				buffer[len] = '\0';
			if( strcmp( this_set->setName, &buffer[1]) ){
				xfree( this_set->setName );
				this_set->setName = XGstrdup(&(buffer[1]));
			}
		}

		  /* 20051013: try to speed up things by cleverly detecting whether the current line has an opcode/command in it... */
		else if( *optbuf && *optbuf != '*' ){
			if( debugFlag && debugLevel ){
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr,
					"ReadData(): %s: line %d.%d (%c%c%c%c): this looks like it must be (ascii) data...\n"
					, filename,
					sub_div, line_count, optbuf[0], optbuf[1], optbuf[2], optbuf[3]
				);
			}
			goto ascii_ReadData;
		}

		else if( strncmp( optbuf, "*ENDIAN*", 8)== 0){
		  char *e= &optbuf[8];
			while( *e && isspace(*e) ){
				e++;
			}
			if( strncasecmp(e, "big", 3)== 0 ){
				 /* big endian: most significant bit is at the start (leftmost) */
				endianness= 0;
				goto handle_endian_cmd;
			}
			else if( strncasecmp(e, "little", 6)== 0 ){
				 /* little endian: most significant bit is at the end (rightmost) */
				endianness= 1;
				goto handle_endian_cmd;
			}
			else if( isdigit( (unsigned char)*e ) ){
				endianness= atoi(e);
handle_endian_cmd:;
				if( endianness!= EndianType ){
					if( debugFlag ){
						fprintf( StdErr,
							"ReadData(): %s: line %d.%d (%s): file claims to have been saved on a machine with a different byte-order\n"
							"\tWill proceed with bytes reversed (and fingers crossed)\n",
							filename,
							sub_div, line_count, optbuf
						);
					}
					swap_endian= True;
				}
				else{
					if( debugFlag && swap_endian ){
						fprintf( StdErr,
							"ReadData(): %s: line %d.%d (%s): file now claims to have been saved on a machine with the same byte-order\n"
							"\tWill proceed without touching byte-ordering (and fingers crossed)\n",
							filename,
							sub_div, line_count, optbuf
						);
					}
					swap_endian= False;
				}
				s_e_set= True;
			}
			else{
				fprintf( StdErr,
					"ReadData(): %s: line %d.%d (%s): unknown/invalid ENDIAN specification.\n"
					"\tWill proceed without (touching) byte-ordering (and with eyes crossed)\n",
					filename,
					sub_div, line_count, optbuf
				);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*BUFLEN*", 8)== 0){
		  int redo= 1;
		  char *len= parse_codes(&optbuf[8]);
		  double l;
			if( strlen( len) ){
			  int n= 1;
			  char *TBh= TBARprogress_header;
				TBARprogress_header= "*BUFLEN*";
				if( fascanf2( &n, len, &l, ',')!= EOF && n== 1 ){
					if( l> LMAXBUFSIZE ){
						LMAXBUFSIZE= (int) l;
						if( LMAXBUFSIZE> 262144 ){
							line_count= UpdateLineCount( stream, &flpos, line_count );
							fprintf( StdErr, "ReadData(): %s: line %d.%d (%s): *BUFLEN* %d "
									"may be too long and cause problems on some platforms!\n",
									filename, sub_div, line_count, optbuf,
									LMAXBUFSIZE
							);
						}
#ifdef OLD_BUFLENCHANGE
						if( spot== 0 ){
							redo= 1;
						}
						else{
							redo= 0;
						}
#endif
					}
				}
				else{
					line_count= UpdateLineCount( stream, &flpos, line_count );
					fprintf( StdErr, "ReadData(): %s: line %d.%d (%s): unable to parse *BUFLEN* statement\n",
							filename, sub_div, line_count, optbuf
					);
				}
				TBARprogress_header= TBh;
			}
			if( debugFlag ){
				fprintf( StdErr, "*BUFLEN* %d%s\n",
					LMAXBUFSIZE,
#ifdef OLD_BUFLENCHANGE
					(redo)? " - rereading file" : ""
#else
					" - reallocating buffer(s)"
#endif
				);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
				LineDumped= True;
			}
			if( redo ){
#ifdef OLD_BUFLENCHANGE
			  int r;
				if( ReadData_terpri ){
					fputs( "\n", StdErr );
					ReadData_terpri= False;
				}
				if( (debugFlag || scriptVerbose) && Read_File_Hist ){
					fprintf( StdErr, "Included files: %s\n", Read_File_Hist );
				}
				xfree( Read_File_Hist );

				if( read_file_buf ){
					if( this_set->read_file ){
						this_set->read_file= concat2( this_set->read_file, "\n\n", read_file_buf, NULL);
					}
					else{
						this_set->read_file= XGstrdup( read_file_buf );
					}
					read_file_buf= NULL;
					this_set->read_file_point= Spot;
				}

				  /* Ensure that sticky variables will stick. Pass on the desired value
				   \ to a new local variable through a series of static variables that
				   \ only serve this purpose.
				   */
				restarted= True;
				s_e= swap_endian;
				r= ReadData( stream, the_file, filenr );
				RESETATTRS();
				RD_RETURN(r);
#else
				if( !(BUFFER= ReadData_AllocBuffer( BUFFER, LMAXBUFSIZE+2, the_file)) ){
					ReadData_RETURN(0);
				}
				  /* Got the memory; update the necessary variables!!	*/
				rd_bufsize= LMAXBUFSIZE+1;
				optbuf= BUFFER + (optbuf-buffer);
				buffer= BUFFER;
#endif
			}
		}
		else if( strncmp( optbuf, "*DPRINTF*", 9)== 0 ){
		  char *buf= cleanup( &optbuf[9] );
		  char dbuf[256];
			if( !buf || !*buf ){
				sprintf( dbuf, "%%.%dg", DBL_DIG+1 );
				buf= dbuf;
			}
			if( buf && *buf ){
			  extern ascanf_Function *ascanf_d3str_format;
				if( strcmp( d3str_format, buf) ){
					d3str_format_changed+= 1;
				}
				strncpy( d3str_format, buf, 15 );
				d3str_format[15]= '\0';
				xfree(ascanf_d3str_format->usage);
				ascanf_d3str_format->usage= strdup(d3str_format);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*AXVAL_DPRINTF*", 15)== 0 ){
		  char *buf= cleanup( &optbuf[15] );
		  extern char *AxisValueFormat;
			if( buf && *buf ){
				if( XGstrcmp( AxisValueFormat, buf) ){
					xfree( AxisValueFormat );
					AxisValueFormat= strdup(buf);
				}
			}
			else{
				xfree( AxisValueFormat );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*AXVAL_MINDIGITS*", 17)== 0 ){
		  char *buf= cleanup( &optbuf[17] );
		  extern int AxisValueMinDigits;
		  int x= atoi(buf);
			if( x>= 0 ){
				AxisValueMinDigits= x;
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( &optbuf[2], "AXISDAT*", 8)== 0){
		  double axmin, axmax;
		  int Mtic, Subdiv, axlog, num, lines= -1, dum;
		  int len= strlen(buffer)-1;
		  LocalWin *wi= (ActiveWin)? ActiveWin : &startup_wi;
		  int *ascale= &autoscale;
			if( len>= 0)
				buffer[len]= '\0';
			if( strncmp( optbuf, "*X", 2)== 0 ){
				if( ActiveWin ){
					ascale= &wi->fit_xbounds;
				}
				axmin= llx; axmax= urx;
				axlog= logXFlag;
				num= sscanf( &optbuf[10], "%d %d %lf %lf %d %d %d",
					&Mtic, &Subdiv, &axmin, &axmax, &axlog, &dum, &lines
				);
				if( num>= 5 && logXFlag!= -1 && logXFlag!= 2 ){
					if( (logXFlag= axlog)> 0){
/* 						sqrtXFlag= -1;	*/
					}
				}
				if( *ascale!= 2){
					if( num>= 4 && (use_lx== 0 || use_lx== 2 || ActiveWin) ){
						if( wi== ActiveWin ){
							wi->loX= axmin;
							wi->hiX= axmax;
						}
						else{
							wi->R_UsrOrgX= axmin;
							wi->R_UsrOppX= axmax;
							use_lx= 1;
						}
						*ascale= 0;
					}
					if( logXFlag && logXFlag!= -1 ){
						if( wi->R_UsrOrgX==0.0 || wi->R_UsrOppX== 0.0 || (logXFlag!= 3 && (wi->R_UsrOrgX<= 0.0 || wi->R_UsrOppX<= 0.0)) ){
							logXFlag= 0;
							if( debugFlag || scriptVerbose )
								fprintf( StdErr, "x-axis (%g,%g) out of range for log-axis\n", llx, wi->R_UsrOppX );
						}
					}
				}
				if( !(num>= 7))
					lines= -1;
				if( debugFlag || scriptVerbose  ){
					fprintf( StdErr,
						"x-axis redef: [%g:%g], *ascale= %d, log=%d, skipping %d lines\n",
						axmin, axmax, *ascale, axlog, lines
					);
					if( *ascale== 2)
						fputs( "\trange spec overridden by -auto\n", StdErr);
					if( logXFlag== -1 || logXFlag== 2)
						fputs( "\tlog spec overridden by -lnx\n", StdErr);
				}
				ReadData_commands+= 1;
			}
			if( strncmp( optbuf, "*Y", 2)== 0 ){
				if( ActiveWin ){
					ascale= &wi->fit_ybounds;
				}
				axmin= lly; axmax= ury;
				axlog= logYFlag;
				num= sscanf( &optbuf[10], "%d %d %lf %lf %d %d %d",
					&Mtic, &Subdiv, &axmin, &axmax, &axlog, &dum, &lines
				);
				if( num>= 5 && logYFlag!= -1 && logYFlag!= 2 ){
					if( (logYFlag= axlog)> 0){
/* 						sqrtYFlag= -1;	*/
					}
				}
				if( *ascale!= 2){
					if( num>= 4 && (use_ly== 0 || use_ly== 2 || ActiveWin) ){
						if( wi== ActiveWin ){
							wi->loY= axmin;
							wi->hiY= axmax;
						}
						else{
							wi->R_UsrOrgY= axmin;
							wi->R_UsrOppY= axmax;
							use_ly= 2;
						}
						*ascale= 0;
					}
					if( logYFlag && logYFlag!= -1 ){
						if( wi->R_UsrOrgY== 0.0 || wi->R_UsrOppY== 0.0 || (logYFlag!= 3 && (wi->R_UsrOrgY<= 0.0 || wi->R_UsrOppY<= 0.0)) ){
							logYFlag= 0;
							if( debugFlag || scriptVerbose )
								fprintf( StdErr, "y-axis (%g,%g) out of range for log-axis\n", wi->R_UsrOrgY, wi->R_UsrOppY );
						}
					}
				}
				if( !(num>= 7))
					lines= -1;
				if( debugFlag  || scriptVerbose ){
					fprintf( StdErr, "y-axis redef: [%g:%g], *ascale= %d, log=%d, skipping %d lines\n",
						axmin, axmax, *ascale, axlog, lines
					);
					if( *ascale== 2)
						fputs( "\trange spec overridden by -auto\n", StdErr);
					if( logYFlag== -1 || logYFlag== 2)
						fputs( "\tlog spec overridden by -lny\n", StdErr);
				}
				ReadData_commands+= 1;
			}
			/* now skip the rest of the axis definition	*/
			for( ; lines> 0; lines--, line_count++ )
				xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
			line_count= UpdateLineCount( stream, &flpos, line_count );
		}
		else if( strncmp( optbuf, "*NOPLOT*", 8)== 0){
		  int len= strlen(optbuf)-1;
			if( optbuf[len]== '\n' ){
				optbuf[len] = '\0';
			}
			if( strcmp( this_set->setName, &optbuf[8]) ){
				xfree( this_set->setName );
				this_set->setName= XGstrdup( &optbuf[8] );
			}
			this_set->draw_set= 0;
			if( debugFlag){
				fprintf( StdErr, "Set #%d won't be plotted\n", setNumber);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*NOLEGEND*", 10)== 0){
		  int len= strlen(optbuf)-1;
			if( optbuf[len]== '\n' ){
				optbuf[len] = '\0';
			}
			if( strcmp( this_set->setName, &optbuf[10]) ){
				xfree( this_set->setName );
				this_set->setName= XGstrdup( &optbuf[10] );
			}
			this_set->show_legend= 0;
			legend_setNumber= setNumber;
			if( debugFlag){
				fprintf( StdErr, "No Legend for set #%d\n", setNumber);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*LEGEND*", 8)== 0){
		  /* added by RJB to give some Multiplot compatibility	*/
		  int len= strlen(optbuf)-1;
		  char *buf, *pbuf;
			if( optbuf[len]== '\n' ){
				optbuf[len] = '\0';
			}
			buf= parse_codes( &optbuf[8] );
			pbuf= SubstituteOpcodes( buf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
				"*This-FileDir*", dirname(filename),
				"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
			if( strcmp( this_set->setName, pbuf) ){
				xfree( this_set->setName );
#ifdef NO_RUNTIME_LEGEND_PARSING
				this_set->setName= String_ParseVarNames( XGstrdup(pbuf), "%[", ']', True, True, "*LEGEND*" );
#else
				this_set->setName= XGstrdup( pbuf );
#endif
			}
			this_set->draw_set= 1;
			this_set->show_legend= 1;
			this_set->show_llines= 1;
			legend_setNumber= setNumber;
			if( debugFlag){
				fprintf( StdErr, "Legend set %d: %s\n", setNumber, this_set->setName);
			}
			ReadData_commands+= 1;
			if( pbuf!= buf ){
				xfree(pbuf);
			}
		}
		else if( strncmp( optbuf, "*LEGTXT*", 8)== 0){
		  char *name= NULL, *buf= parse_codes(&optbuf[8]), *pbuf;
		  Boolean append= False;
			if( *buf== '+' ){
				buf++;
				append= True;
			}
			pbuf= SubstituteOpcodes( buf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
				"*This-FileDir*", dirname(filename),
				"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
			if( legend_setNumber== setNumber && append ){
				name= concat( this_set->setName, "\n", pbuf, NULL );
			}
			else{
				name= XGstrdup( pbuf );
			}
			if( pbuf!= buf ){
				xfree(pbuf);
			}
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			while( name && buf ){
				while( buf && optbuf[0]!= '\n' && !feof(stream) ){
					buf= parse_codes( xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin ) );
					if( buf && DumpFile ){
						fputs( buffer, stdout );
					}
					if( buf && buffer[0] && buffer[0]!= '\n' ){
						pbuf= SubstituteOpcodes( buffer, "*This-File*", the_file, "*This-FileName*", basename(the_file),
							"*This-FileDir*", dirname(the_file),
							"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
						name= concat2( name, pbuf, NULL );
						if( pbuf!= buffer ){
							xfree(pbuf);
						}
					}
					else{
						buf= NULL;
					}
					line_count++;
				}
			}
			LineDumped= True;
			xfree( this_set->setName );
#ifdef NO_RUNTIME_LEGEND_PARSING
			this_set->setName= String_ParseVarNames( name, "%[", ']', True, True, "*LEGTXT*" );
#else
			this_set->setName= name;
#endif
			this_set->draw_set= 1;
			this_set->show_legend= 1;
			this_set->show_llines= 1;
			legend_setNumber= setNumber;
			if( debugFlag){
				fprintf( StdErr, "Legend set %d: %s\n", setNumber, this_set->setName);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*CLEAR_COMMENTS*", 16)== 0){
			xfree( comment_buf );
			comment_size= 0;
			if( debugFlag){
				fprintf( StdErr, "Comments cleared\n" );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PROCESS_HISTORY*", 17)== 0){
		  char *buf= &optbuf[17], *pbuf;
			if( *buf== '\0' || *buf== '\n' ){
				add_process_hist( NULL );
			}
			else{
				pbuf= SubstituteOpcodes( buf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
					"*This-FileDir*", dirname(the_file),
					"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
				add_process_hist( pbuf );
				if( pbuf!= buf ){
					xfree( pbuf );
				}
			}
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			while( buf && *buf ){
				while( buf && optbuf[0]!= '\n' && !feof(stream) ){
					buf= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
					if( buf && DumpFile ){
						fputs( buffer, stdout );
					}
					if( buf && buffer[0] && buffer[0]!= '\n' ){
						pbuf= SubstituteOpcodes( buf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
							"*This-FileDir*", dirname(the_file),
							"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
						add_process_hist( pbuf );
						if( pbuf!= buf ){
							xfree( pbuf );
						}
					}
					else{
						buf= NULL;
					}
					line_count++;
				}
			}
			LineDumped= True;
			if( debugFlag ){
				if( process_history ){
					fprintf( StdErr, "Process history: %s\n", process_history );
				}
				else{
					fprintf( StdErr, "Process history: EMPTY\n" );
				}
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*SPLIT*", 7)== 0 ){
			if( !ignore_splits ){
				split_set= 1;
				if( splits_disconnect ){
				  int len= strlen(optbuf)-1;
					if( optbuf[len] == '\n' ){
						optbuf[len]= '\0';
					}
					split_reason= XGstrdup( &optbuf[7] );
				}
				else{
				  /* We ignore the reason.	*/
					if( debugFlag ){
						fprintf( StdErr, "%s, set #%d, pnt %d, pen lift for reason %s",
							this_set->fileName, setNumber, spot, &optbuf[7]
						);
					}
				}
			}
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*VERSION*", 9)== 0 ){
		  char *vl= NULL, *info= cleanup( &optbuf[9] ), *tinfo;
		  char lbuf[64], tsbuf[1280];
			if( !ReadData_IgnoreVERSION ){
				line_count= UpdateLineCount( stream, &flpos, line_count );
				sprintf( lbuf, "%d", line_count );
				time_stamp( stream, (the_file)? the_file : "stdin", tsbuf, True, NULL );
				{ struct tm *tm;
				  time_t timer= time(NULL);
				  char *c;
					tm= localtime( &timer );
					tinfo= cleanup( asctime(tm) );
					if( (c= rindex( tinfo, '\n')) ){
						*c= '\0';
					}
				}
				vl= concat2( vl, "* Inputfile \"", tsbuf,
					"\"; calls itself \"", filename, "\" around line ", lbuf,
					" at ", tinfo, NULL
				);
				if( info ){
					vl= concat2( vl, "; info string: \"", info, "\"\n", NULL);
				}
				else{
					vl= concat2( vl, "\n", NULL);
				}
				Add_Comment( vl, True );
				version_list= concat2( version_list, vl, NULL );
				xfree( vl );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*VERSION_LIST*", 14)== 0){
		  char *buf= parse_codes( &optbuf[14] );
			if( !ReadData_IgnoreVERSION ){
				version_list= concat2( version_list, "<", (the_file)? the_file : "stdin", ">\n", NULL );
				if( *buf ){
					if( buf[1] && buf[ strlen(buf)-1 ]== '\n' ){
						buf[ strlen(buf)-1 ]= '\0';
					}
					if( strcmp( buf, "*PINFO*")== 0 ){
					  char *pinfo= NULL;
						_Dump_Arg0_Info( ActiveWin, NULL, &pinfo, False );
						version_list= concat2( version_list, pinfo, NULL );
						xfree( pinfo );
					}
					else if( *buf && (!version_list || !strstr( version_list, buf )) ){
						version_list= concat2( version_list, buf, "\n", NULL );
					}
					Add_Comment( buf, True );
				}
			}
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fprintf( stdout, "%s", buffer );
			}
			buf[0]= '*';
			  /* 20020620: if was while?? */
			if( buf && *buf ){
			  int ilines= 0, el= (optbuf[0]== '\n')? True : False;
				while( buf && !el && !feof(stream) ){
					buf= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
					  /* 20020901: determine empty-line-ness from the unparsed buffer! */
					el= (optbuf[0]== '\n')? True : False;
					if( buf ){
						if( DumpFile ){
							fputs( buffer, stdout );
						}
						if( buf[0]!= '\n' && !ReadData_IgnoreVERSION ){
							buf= parse_codes(buf);
							if( *buf ){
								if( buf[1] && buf[ strlen(buf)-1 ]== '\n' ){
									buf[ strlen(buf)-1 ]= '\0';
								}
								if( strcmp( buf, "*PINFO*")== 0 ){
								  char *pinfo= NULL;
									_Dump_Arg0_Info( ActiveWin, NULL, &pinfo, False );
									version_list= concat2( version_list, pinfo, NULL );
									xfree( pinfo );
								}
								else if( *buf && (!version_list || !strstr( version_list, buf) || buf[0]== '\n') ){
									version_list= concat2( version_list, buf, "\n", NULL );
								}
								Add_Comment( buf, True );
								ilines+= 1;
							}
						}
						else{
							buf= NULL;
						}
					}
					line_count++;
				}
				if( ilines ){
					Skip_new_line= 1;
					line_count-= 1;
				}
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*SET_INFO_LIST*", 15)== 0){
		  char *buf= parse_codes( &optbuf[15] );
		  int ilines= 0;
			if( *buf ){
				if( buf[ strlen(buf)-1 ]== '\n' ){
					buf[ strlen(buf)-1 ]= '\0';
				}
				if( *buf && !ReadData_IgnoreVERSION ){
					this_set->set_info= concat2( this_set->set_info, buf, "\n", NULL );
					ilines+= 1;
				}
			}
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fprintf( stdout, "%s\n", buffer );
			}
			buf[0]= '*';
			  /* 20020620: if was while?? */
			if( buf && *buf ){
			  int el= (optbuf[0]== '\n')? True : False;
				while( buf && !el && !feof(stream) ){
					buf= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
					el= (optbuf[0]== '\n')? True : False;
					if( buf ){
						if( DumpFile ){
							fputs( buffer, stdout );
						}
						if( buf[0]!= '\n' && !ReadData_IgnoreVERSION ){
/* 							buf= cleanup(buf);	*/
							buf= parse_codes(buf);
							if( *buf ){
								if( buf[1] && buf[ strlen(buf)-1 ]== '\n' ){
									buf[ strlen(buf)-1 ]= '\0';
								}
								if( *buf ){
									this_set->set_info= concat2( this_set->set_info, buf, "\n", NULL );
								}
								ilines+= 1;
							}
						}
						else{
							buf= NULL;
						}
					}
					line_count++;
				}
			}
			if( ilines> 1 ){
				Skip_new_line= 1;
				line_count-= 1;
			}
			if( !ilines && !ReadData_IgnoreVERSION ){
				xfree( this_set->set_info );
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*FILE*", 6)== 0 ){
		  int len= strlen(optbuf)-1;
		  int new_file= (strncmp( optbuf, "*FILE*", 6)== 0);
		  char *name= &optbuf[6], *pname0, *pname;
			if( len> 0){
			   optbuf[len] = '\0';
			}
			sub_div+= 1;
			line_count= 0;

			while( *name && isspace((unsigned char)*name ) ){
				name++;
			}
/* 			if( strcmp( name, "%F")== 0 && the_file ){	*/
/* 				name= filename;	*/
/* 			}	*/

			pname0 = String_ParseVarNames( XGstrdup(name), "%[", ']', False, False, "*FILE*" );
			pname= SubstituteOpcodes( pname0, "*This-File*", the_file, "%F", filename, "*This-FileName*", basename(the_file),
				"*This-FileDir*", dirname(the_file),
				"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );

			Show_Stats( StdErr, filename, &SS__x, &SS__y, &SS__e, NULL, NULL);
			SS_Reset_(SS__x);
			SS_Reset_(SS__y);
			SS_Reset_(SS__e);

			if( !this_set->fileName && *pname ){
				this_set->fileName= XGstrdup( pname );
				xfree( filename );
				filename= XGstrdup( pname );
				this_set->new_file= new_file;
				if( new_file ){
					splittable_file+= 1;
				}
				if( StartUp ){
					if( filename_in_legend!= -1 ){
						filename_in_legend= 1;
					}
				}
			}
			else /* if( strcmp( this_set->fileName, pname ) )	*/
			{
				if( !this_set->fileName || strcmp( this_set->fileName, pname) ){
					if( new_file && this_set->fileName ){
					  /* a *FILE* statement with a name behind specifies a new file;
					   \ without, it depends on whether a *TITLE* statement was given
					   \ earlier. If not, only a new group is created.
					   */
						splittable_file+= 1;
					}
					xfree( this_set->fileName );
					this_set->fileName= XGstrdup( pname );
				}
				if( strcmp( filename, pname) ){
					xfree( filename );
					filename= XGstrdup( pname );
				}
				this_set->new_file= new_file;
				if( StartUp ){
					if( filename_in_legend!= -1 ){
						filename_in_legend= 1;
					}
				}
			}
			if( this_set->new_file && newfile_incr_width ){
				  /* 990723: <1> replaced below by newfile_incr_width	*/
				if( incr_width> 0 ){
					lineWidth+= newfile_incr_width;
					this_set->lineWidth+= newfile_incr_width;
				}
				_incr_width= newfile_incr_width;
			}
			if( debugFlag && filename_in_legend ){
				fprintf( StdErr, "New Graph file: %s (set #%d (%d)), width %g, incr_width=%g\n",
					this_set->fileName, setNumber, this_set->new_file, this_set->lineWidth,
					incr_width
				);
			}
			if( this_set->new_file && splittable_file> 0 ){
			  /* this file actually consists of multiple files	*/
				if( filenr==  prev_filenr ){
					file_splits+= 1;
					if( debugFlag ){
						fprintf( StdErr, "ReadData(\"%s\",%d): split file (%d total)\n",
							this_set->fileName, filenr, file_splits
						);
					}
				}
				splittable_file= 0;
				sub_div= 0;
				perc= 0;
			}
			if( pname!= pname0 ){
				xfree(pname);
			}
			if( pname0 != name ){
				xfree(pname0);
			}
			  /* add the *FILE* line to the comments, if possible
			   \ appending the timestamp of the file
			   */
			{ int nb= strlen(optbuf);
				name= &optbuf[6];
				time_stamp( NULL, this_set->fileName, name, -1, "\n" );
				optbuf[strlen(optbuf)-1]= '\0';
				add_comment( optbuf );
				  /* remove the timestamp again from the buffer in case we're dumping to the terminal	*/
				optbuf[nb]= '\0';
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*TITLE*", 7)== 0){
		  int len= strlen(optbuf)-1;
		  static char title[MAXBUFSIZE+1];
		  char *buf, *pbuf;
			if( optbuf[len]== '\n' ){
				optbuf[len] = '\0';
			}
			buf= parse_codes( &optbuf[7] );
			pbuf= SubstituteOpcodes( buf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
				"*This-FileDir*", dirname(the_file),
				"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
			  /* 990617: I think we can safely accept multiple identical titlestrings now.
			   \ We have to, since they can contain opcodes with set-specific interpretation..
			   */
			if( True || strncmp( title, pbuf, MAXBUFSIZE) ){
				strncpy( title, pbuf, MAXBUFSIZE);
				if( !titleTextSet){
					strncpy( titleText, pbuf, MAXBUFSIZE);
					this_set->titleText= XGstrdup( pbuf );
					this_set->titleChanged= 1;
					titleText[MAXBUFSIZE-1]= '\0';
				}
				else if( debugFlag){
				  /* if -t title has been given, only update filename and new_file flag	*/
					fputs( "\t*TITLE* overridden by -t\n", StdErr);
				}
				if( !this_set->fileName || (this_set->fileName && strcmp( this_set->fileName, filename)) ){
					xfree( this_set->fileName );
					this_set->fileName= XGstrdup(filename);
				}
/* 				this_set->new_file= 1;	*/
				  /* this file may consist of multiple files:	*/
/* 				splittable_file+= 1;	*/
				if( debugFlag){
					fprintf( StdErr, "(New) Graph title: %s (set #%d (%d)), width %g\n",
						titleText, setNumber, this_set->new_file, this_set->lineWidth
					);
				}
			}
			if( pbuf!= buf ){
				xfree(pbuf);
			}
			reset_Scalers(optbuf);
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*XLABEL*", 8)== 0){
		  int len= strlen(optbuf)-1, *UnitsSet= &XUnitsSet;
		  char xy= 'X', *Units= XUnits, *tr_Units= tr_XUnits, *Init_Units= INIT_XUNITS;
		  int global= strncmp( &optbuf[2], "label", 5 );
			if( len> 0)
			   optbuf[len] = '\0';
			if( !*UnitsSet || (ActiveWin && ActiveWin!= &StubWindow) ){
				if( optcol== column[2] ){
					if( use_errors ){
					  /* append +/- <errname> . Instead of '+/-', the hexadecimal
					   \ value of the "plusminus" character in the Adobe Symbol font
					   \ is inserted. This is parsed later-on by parse_codes()
					   */
						if( global ){
							if( vectorFlag ){
								strncat( Units, " \\#xd0\\ ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								strncat( Units, " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
							}
							strncat( Units, &optbuf[8], MAXBUFSIZE- strlen(Units) );
						}
						if( vectorFlag ){
							this_set->XUnits= concat2( this_set->XUnits, " \\#xd0\\ ", &optbuf[8], NULL );
						}
						else{
							this_set->XUnits= concat2( this_set->XUnits, " \\#xb1\\ ", &optbuf[8], NULL );
						}
					}
				}
				else if( optbuf[8]== '+'){
					if( global ){
						if( strlen(Units) ){
							if( memcmp(Units,Init_Units,6) ){
								strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								*Units= '\0';
							}
						}
						strncat( Units, &optbuf[9], MAXBUFSIZE- strlen(Units) );
					}
					this_set->XUnits= concat2( this_set->XUnits, &optbuf[9], NULL );
				}
				else{
					if( global ){
						Units[0]= '\0';
						strncpy( Units, &optbuf[8], MAXBUFSIZE);
						if( ActiveWin ){
							strcpy( ActiveWin->XUnits, Units );
						}
					}
/* 					this_set->XUnits= concat2( this_set->XUnits, &optbuf[8], NULL );	*/
					xfree( this_set->XUnits );
					this_set->XUnits= XGstrdup( &optbuf[8] );
				}
				parse_codes( this_set->XUnits );
				Units[MAXBUFSIZE-1]= '\0';
				if( global )
					strcpy( tr_Units, Units );
				if( debugFlag)
					fprintf( StdErr, "%cUnit name: \"%s\"%s\n", xy, Units, (global)? " for all sets" : "" );
			}
			else if( debugFlag)
				fputs( "\tXLABEL overridden by -x\n", StdErr);
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*XLABEL_TRANS*", 14)== 0){
		  int len= strlen(optbuf)-1, *UnitsSet= &XUnitsSet;
		  char xy= 'X', *Units= tr_XUnits, *Init_Units= INIT_XUNITS;
		  int global= strncmp( &optbuf[2], "label", 5 );
			if( len> 0)
			   optbuf[len] = '\0';
#if 0
			{
				if( optcol== column[2] ){
					if( use_errors ){
					  /* append +/- <errname> . Instead of '+/-', the hexadecimal
					   \ value of the "plusminus" character in the Adobe Symbol font
					   \ is inserted. This is parsed later-on by parse_codes()
					   */
						if( vectorFlag ){
							strncat( Units, " \\#xd0\\ ", MAXBUFSIZE- strlen(Units) );
						}
						else{
							strncat( Units, " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
						}
						strncat( Units, &optbuf[14], MAXBUFSIZE- strlen(Units) );
					}
				}
				else if( optbuf[8]== '+'){
					if( strlen(Units) ){
						if( memcmp(Units,Init_Units,6) ){
							strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
						}
						else{
							*Units= '\0';
						}
					}
					strncat( Units, &optbuf[15], MAXBUFSIZE- strlen(Units) );
				}
				else{
					strncpy( Units, &optbuf[14], MAXBUFSIZE);
				}
				Units[MAXBUFSIZE-1]= '\0';
				if( debugFlag)
					fprintf( StdErr, "%cUnit(transformed) name: \"%s\"\n", xy, Units );
			}
#endif
			if( !*UnitsSet || (ActiveWin && ActiveWin!= &StubWindow) || !optbuf[14] ){
				if( optcol== column[2] ){
					if( use_errors ){
					  /* append +/- <errname> . Instead of '+/-', the hexadecimal
					   \ value of the "plusminus" character in the Adobe Symbol font
					   \ is inserted. This is parsed later-on by parse_codes()
					   */
						if( global ){
							if( vectorFlag ){
								strncat( Units, " \\#xd0\\ ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								strncat( Units, " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
							}
							strncat( Units, &optbuf[14], MAXBUFSIZE- strlen(Units) );
						}
					}
				}
				else if( optbuf[8]== '+'){
					if( global ){
						if( strlen(Units) ){
							if( memcmp(Units,Init_Units,6) ){
								strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								*Units= '\0';
							}
						}
						strncat( Units, &optbuf[15], MAXBUFSIZE- strlen(Units) );
					}
				}
				else{
					if( global ){
						Units[0]= '\0';
						strncpy( Units, &optbuf[14], MAXBUFSIZE);
						if( strcmp( Units, "*XLABEL*")==0 ){
							strcpy( Units, XUnits );
						}
						if( ActiveWin ){
							strcpy( ActiveWin->tr_XUnits, Units );
						}
					}
				}
				Units[MAXBUFSIZE-1]= '\0';
				if( debugFlag)
					fprintf( StdErr, "%cUnit name: \"%s\"%s\n", xy, Units, (global)? " for all sets" : "" );
			}
			else if( debugFlag)
				fputs( "\tXLABEL overridden by -x\n", StdErr);
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*YLABEL*", 8)== 0){
		  int len= strlen(optbuf)-1, *UnitsSet= &YUnitsSet;
		  char xy= 'Y', *Units= YUnits, *tr_Units= tr_YUnits, *Init_Units= INIT_YUNITS;
		  int global= strncmp( &optbuf[2], "label", 5 );
			if( len> 0)
			   optbuf[len] = '\0';
			if( !*UnitsSet || (ActiveWin && ActiveWin!= &StubWindow) ){
				if( optcol== column[2] ){
					if( use_errors ){
						if( global ){
							if( vectorFlag ){
								strncat( Units, " \\#xd0\\ ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								strncat( Units, " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
							}
							if( optbuf[8]== '+' ){
								strncat( Units, &optbuf[9], MAXBUFSIZE- strlen(Units) );
							}
							else{
								strncat( Units, &optbuf[8], MAXBUFSIZE- strlen(Units) );
							}
						}
						if( vectorFlag ){
							this_set->YUnits= concat2( this_set->YUnits, " \\#xd0\\ ", (optbuf[8]== '+')? &optbuf[9] : &optbuf[8], NULL );
						}
						else{
							this_set->YUnits= concat2( this_set->YUnits, " \\#xb1\\ ", (optbuf[8]== '+')? &optbuf[9] : &optbuf[8], NULL );
						}
					}
				}
				else if( optbuf[8]== '+'){
					if( global ){
						if( strlen(Units) ){
							if( memcmp(Units,Init_Units,6) ){
								strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								*Units= '\0';
							}
						}
						strncat( Units, &optbuf[9], MAXBUFSIZE- strlen(Units) );
					}
					this_set->YUnits= concat2( this_set->YUnits, &optbuf[9], NULL );
				}
				else{
					if( global ){
						Units[0]= '\0';
						strncpy( Units, &optbuf[8], MAXBUFSIZE);
						if( ActiveWin ){
							strcpy( ActiveWin->YUnits, Units );
						}
					}
					xfree( this_set->YUnits );
					this_set->YUnits= XGstrdup( &optbuf[8] );
				}
				Units[MAXBUFSIZE-1]= '\0';
				if( global )
					strcpy( tr_Units, Units );
				if( debugFlag)
					fprintf( StdErr, "%cUnit name: \"%s\"%s\n", xy, Units, (global)? " for all sets" : "" );
				if( xy== 'Y')
					reset_Scalers(optbuf);
			}
			else if( debugFlag)
				fputs( "\tYLABEL overridden by -y\n", StdErr);
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*YLABEL_TRANS*", 14)== 0){
		  int len= strlen(optbuf)-1, *UnitsSet= &XUnitsSet;
		  char xy= 'Y', *Units= tr_YUnits, *Init_Units= INIT_YUNITS;
		  int global= strncmp( &optbuf[2], "label", 5 );
			if( len> 0)
			   optbuf[len] = '\0';
#if 0
			{
				if( optcol== column[2] ){
					if( use_errors ){
					  /* append +/- <errname> . Instead of '+/-', the hexadecimal
					   \ value of the "plusminus" character in the Adobe Symbol font
					   \ is inserted. This is parsed later-on by parse_codes()
					   */
						if( vectorFlag ){
							strncat( Units, " \\#xd0\\ ", MAXBUFSIZE- strlen(Units) );
						}
						else{
							strncat( Units, " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
						}
						strncat( Units, &optbuf[14], MAXBUFSIZE- strlen(Units) );
					}
				}
				else if( optbuf[8]== '+'){
					if( strlen(Units) ){
						if( memcmp(Units,Init_Units,6) ){
							strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
						}
						else{
							*Units= '\0';
						}
					}
					strncat( Units, &optbuf[15], MAXBUFSIZE- strlen(Units) );
				}
				else{
					strncpy( Units, &optbuf[14], MAXBUFSIZE);
				}
				Units[MAXBUFSIZE-1]= '\0';
				if( debugFlag)
					fprintf( StdErr, "%cUnit(transformed) name: \"%s\"\n", xy, Units );
			}
#endif
			if( !*UnitsSet || (ActiveWin && ActiveWin!= &StubWindow) || !optbuf[14] ){
				if( optcol== column[2] ){
					if( use_errors ){
						if( global ){
							if( vectorFlag ){
								strncat( Units, " \\#xd0\\ ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								strncat( Units, " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
							}
							if( optbuf[8]== '+' ){
								strncat( Units, &optbuf[15], MAXBUFSIZE- strlen(Units) );
							}
							else{
								strncat( Units, &optbuf[14], MAXBUFSIZE- strlen(Units) );
							}
						}
					}
				}
				else if( optbuf[8]== '+'){
					if( global ){
						if( strlen(Units) ){
							if( memcmp(Units,Init_Units,6) ){
								strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								*Units= '\0';
							}
						}
						strncat( Units, &optbuf[15], MAXBUFSIZE- strlen(Units) );
					}
				}
				else{
					if( global ){
						Units[0]= '\0';
						strncpy( Units, &optbuf[14], MAXBUFSIZE);
						if( strcmp( Units, "*YLABEL*")==0 ){
							strcpy( Units, YUnits );
						}
						if( ActiveWin ){
							strcpy( ActiveWin->tr_YUnits, Units );
						}
					}
				}
				Units[MAXBUFSIZE-1]= '\0';
				if( debugFlag)
					fprintf( StdErr, "%cUnit name: \"%s\"%s\n", xy, Units, (global)? " for all sets" : "" );
				if( xy== 'Y')
					reset_Scalers(optbuf);
			}
			else if( debugFlag)
				fputs( "\tYLABEL overridden by -y\n", StdErr);
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*XYLABEL*", 9)== 0){
		  int len= strlen(optbuf)-1, *UnitsSet= &XUnitsSet;
		  char xy= 'X', *Units= XUnits, *tr_Units= tr_XUnits, *Init_Units= INIT_XUNITS;
		  int global= strncmp( &optbuf[3], "label", 5 );
			if( len> 0)
			   optbuf[len] = '\0';
			if( column[0]!= 0 || column[1]!= 1 || column[2]!= 2 ){
				if( optcol!= column[0]){
					UnitsSet= &YUnitsSet;
					Units= YUnits;
					tr_Units= tr_YUnits;
					Init_Units= INIT_YUNITS;
					xy= 'Y';
					if( debugFlag)
						fprintf( StdErr, "*XYLABEL* in column %d ==> *YLABEL*\n", optcol);
					reset_Scalers(optbuf);
				}
			}
			if( !*UnitsSet || (ActiveWin && ActiveWin!= &StubWindow) ){
				if( optcol== column[2] ){
					if( use_errors ){
						if( global ){
							strncat( Units, (vectorFlag)? " \\#xd0\\ " : " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
							strncat( Units, &optbuf[9], MAXBUFSIZE- strlen(Units) );
						}
						if( Units== YUnits ){
							this_set->YUnits= concat2( this_set->YUnits, (vectorFlag)? " \\#xd0\\ " : " \\#xb1\\ ", &optbuf[9], NULL );
						}
						else{
							this_set->XUnits= concat2( this_set->XUnits, (vectorFlag)? " \\#xd0\\ " : " \\#xb1\\ ", &optbuf[9], NULL );
						}
						if( debugFlag ){
							fprintf( StdErr, "label command in column %d => concat \"%s\" to %cUnits\n",
								optcol, &optbuf[9], xy
							);
						}
					}
				}
				else if( optbuf[9]== '+'){
					if( global ){
						if( strlen(Units) ){
							if( memcmp(Units,Init_Units,6) ){
								strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								*Units= '\0';
							}
						}
						strncat( Units, &optbuf[10], MAXBUFSIZE- strlen(Units) );
					}
					if( Units== YUnits ){
						this_set->YUnits= concat2( this_set->YUnits, &optbuf[10], NULL );
					}
					else{
						this_set->XUnits= concat2( this_set->XUnits, &optbuf[10], NULL );
					}
					if( debugFlag ){
						fprintf( StdErr, "concat \"%s\" to %cUnits\n",
							&optbuf[10], xy
						);
					}
				}
				else{
					if( global ){
						strncpy( Units, &optbuf[9], MAXBUFSIZE);
					}
					if( Units== YUnits ){
						xfree( this_set->YUnits );
						this_set->YUnits= XGstrdup( &optbuf[9] );
					}
					else{
						xfree( this_set->XUnits );
						this_set->XUnits= XGstrdup( &optbuf[9] );
					}
				}
				parse_codes( this_set->XUnits );
				parse_codes( this_set->YUnits );
				Units[MAXBUFSIZE-1]= '\0';
				if( global )
					strcpy( tr_Units, Units );
				if( debugFlag)
					fprintf( StdErr, "%cUnit name: \"%s\"%s\n", xy, Units, (global)? " for all sets" : "" );
			}
			else if( debugFlag)
				fputs( "\tXYLABEL overridden by -x\n", StdErr);
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*YXLABEL*", 9)== 0){
		  int len= strlen(optbuf)-1, *UnitsSet= &YUnitsSet;
		  char xy= 'Y', *Units= YUnits, *tr_Units= tr_YUnits, *Init_Units= INIT_YUNITS;
		  int global= strncmp( &optbuf[3], "label", 5 );
			if( len> 0)
			   optbuf[len] = '\0';
/* 			if( column[0]!= 0 || column[1]!= 1 || column[2]!= 2 ){	*/
				if( optcol== column[0]){
					UnitsSet= &XUnitsSet;
					Units= XUnits;
					tr_Units= tr_XUnits;
					Init_Units= INIT_XUNITS;
					xy= 'X';
					if( debugFlag)
						fprintf( StdErr, "*YXLABEL* in column %d ==> *XLABEL*\n", optcol);
				}
/* 			}	*/
			if( !*UnitsSet || (ActiveWin && ActiveWin!= &StubWindow) ){
				if( optcol== column[2] ){
					if( use_errors ){
					  char *lbl= (optbuf[9]== '+')? &optbuf[10] : &optbuf[9];
						if( global ){
							strncat( Units, (vectorFlag)? " \\#xd0\\ " : " \\#xb1\\ ", MAXBUFSIZE- strlen(Units) );
							strncat( Units, lbl, MAXBUFSIZE- strlen(Units) );
						}
						if( Units== YUnits ){
							this_set->YUnits= concat2( this_set->YUnits, (vectorFlag)? " \\#xd0\\ " : " \\#xb1\\ ", lbl, NULL );
						}
						else{
							this_set->XUnits= concat2( this_set->XUnits, (vectorFlag)? " \\#xd0\\ " : " \\#xb1\\ ", lbl, NULL );
						}
					}
				}
				else if( optbuf[9]== '+'){
					if( global ){
						if( strlen(Units) ){
							if( memcmp(Units,Init_Units,6) ){
								strncat( Units, " ; ", MAXBUFSIZE- strlen(Units) );
							}
							else{
								*Units= '\0';
							}
						}
						strncat( Units, &optbuf[10], MAXBUFSIZE- strlen(Units) );
					}
					if( Units== YUnits ){
						this_set->YUnits= concat2( this_set->YUnits, &optbuf[10], NULL );
					}
					else{
						this_set->XUnits= concat2( this_set->XUnits, &optbuf[10], NULL );
					}
				}
				else{
					if( global ){
						strncpy( Units, &optbuf[9], MAXBUFSIZE);
					}
					if( Units== YUnits ){
						xfree( this_set->YUnits );
						this_set->YUnits= XGstrdup( &optbuf[9] );
					}
					else{
						xfree( this_set->XUnits );
						this_set->XUnits= XGstrdup( &optbuf[9] );
					}
				}
				parse_codes( this_set->XUnits );
				parse_codes( this_set->YUnits );
				Units[MAXBUFSIZE-1]= '\0';
				if( global )
					strcpy( tr_Units, Units );
				if( debugFlag)
					fprintf( StdErr, "%cUnit name: \"%s\"%s\n", xy, Units, (global)? " for all sets" : "" );
				if( xy== 'Y')
					reset_Scalers(optbuf);
			}
			else if( debugFlag)
				fputs( "\tYXLABEL overridden by -y\n", StdErr);
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PROPERTIES*", 12)== 0 ||
			strncmp( optbuf, "*AUTOSCRIPT*", 12)== 0
		){
		  int mrks, dum, err, axspec, num, estyle;
		  double width, ewidth, mSize;
		  int len= strlen(optbuf)-1, new_format= 0;
		  char *form= &optbuf[12];
			if( len>= 0)
				optbuf[len]= '\0';
			num= sscanf( &optbuf[12], "%d %d %d %d %lf %d %d %lf", &dum, &err, &axspec, &dum, &width, &estyle,
				&mrks, &ewidth
			);
			  /* PROPERTIES can have four (the relevant one) or more arguments	*/
			if( num== 4 ){
			  int tf= 0, bf= 0, zf= 0;
				use_errors= (no_errors)? 0 : err;
				switch( axspec ){
					case 11:
						zf= 1;
					case 1:
						tf= 0;
						bf= 0;
						break;
					case 12:
						zf= 1;
					case 2:
						tf= 1;
						bf= 1;
						break;
					case 13:
						zf= 1;
					case 3:
						tf= 0;
						bf= 1;
						break;
				}
				if( zeroFlag!= -1 && zeroFlag!= 2 && !polarFlag )
					zeroFlag= zf;
				if( htickFlag!= -1 && htickFlag!= 2 && !polarFlag )
					htickFlag= tf;
				if( vtickFlag!= -1 && vtickFlag!= 2 && !polarFlag )
					vtickFlag= tf;
				if( bbFlag!= -1 && bbFlag!= 2){
					bbFlag= bf;
				}
				if( debugFlag){
					fprintf( StdErr, "*PROPERTIES* Set #%d: use_error=%d: ticks=%d: box=%d: zeroLines=%d\n",
						setNumber, err, tf, bf, zf
					);
					if( no_errors )
						fputs( "\terrorflags overridden by -noerr\n", StdErr);
					if( zeroFlag== -1 || zeroFlag== 2)
						fputs( "\tzeroLines overridden by -zl\n", StdErr);
					if( htickFlag== -1 || htickFlag== 2)
						fputs( "\thticks overridden by -tk\n", StdErr);
					if( vtickFlag== -1 || vtickFlag== 2)
						fputs( "\tvticks overridden by -tk\n", StdErr);
					if( bbFlag== -1 || bbFlag== 2)
						fputs( "\tbox overridden by -bb\n", StdErr);
				}
				if( !XGIgnoreCNames ){
					goto properties_colours;
				}
			}
			else if( num!= 8 ){
				  /* The extended template must match must more exactly - leading whitespace
				   \ causes a mismatch...
				   */
				form= &optbuf[12];
				while( isspace((unsigned char) *form) ){
				  /* So we remove any..!	*/
					form++;
				}
				if( (num= sscanf( form, "colour=%d flags=0x%x linestyle=%d mS=%d lineWidth=%lf "
						"elinestyle=%d marker=%d elineWidth=%lf markSize=%lf",
							&dum, &err, &axspec, &dum, &width, &estyle, &mrks, &ewidth, &mSize
						))< 8
				){
					  /* No match. Fall back upon the default, older version.	*/
					num= sscanf( &optbuf[12],
						"%d 0x%x %d %d %lf %d %d %lf",
						&dum, &err, &axspec, &dum, &width, &estyle, &mrks, &ewidth
					);
				}
				else{
				  /* 20020203: when we needed to fall back, we're *NOT* in the new format! */
					new_format= 1;
				}
				if( num< 8 ){
					if( !XGIgnoreCNames ){
						form= &optbuf[12];
						goto properties_colours;
					}
				}
			}
			if( num>= 8){
				switch( err){
					case 0:		/* default	*/
						if( this_set->noLines!= 2)
							this_set->noLines= 0;
						else if( debugFlag)
							fputs( "\tLineplot overridden by -nl\n", StdErr);
						if( this_set->markFlag!= 2)
							this_set->markFlag= 0;
						else if( debugFlag)
							fputs( "\tLineplot overridden by -m, -M, -p, or -P\n", StdErr);
						break;
					case 1:		/* -nl -m	*/
						if( !this_set->noLines)
							this_set->noLines= 1;
						if( this_set->markFlag!= 2){
							this_set->markFlag= 1;
							this_set->pixelMarks= 0;
						}
						else if( debugFlag)
							fputs( "\tmarks overridden by -m (on anyway)\n", StdErr);
						break;
					case 2:		/* -m	*/
						if( this_set->markFlag!= 2){
							this_set->markFlag= 1;
							this_set->pixelMarks= 0;
						}
						else if( debugFlag)
							fputs( "\tmarks overridden by -m (on anyway)\n", StdErr);
						if( this_set->noLines!= 2)
							this_set->noLines= 0;
						else if( debugFlag)
							fputs( "\tlines overridden by -nl\n", StdErr);
						break;
					case 4:		/* -nl -bar -m	*/
						if( !this_set->polarFlag){
							if( !this_set->barFlag)
								this_set->barFlag= 1;
							if( !this_set->noLines)
								this_set->noLines= 1;
							if( !this_set->markFlag)
								this_set->markFlag= 1;
						}
						else if( debugFlag){
						  /* for the time being, polar plots can only be requested on the
						   * command-line
						   */
							fputs( "\tbarplot overridden by -polar\n", StdErr);
						}
						break;
					default:{
					  int mask= err;
						if( new_format || CheckMask(mask, (1<<31)) ){
						  int m= (mask & 6) >> 1;
							  /* 20010620: command line settings no longer override the settings specified
							   \ with the newer style PROPERTIES commands!
							   */
							if( CheckMask(mask,1) && !this_set->noLines ){
								this_set->noLines= 1;
							}
							else if( !CheckMask(mask,1) /* && this_set->noLines!= 2 */ ){
								this_set->noLines= 0;
							}
							if( m && this_set->markFlag!= 2 ){
								this_set->markFlag= 1;
								switch( m ){
									case 1:
										  /* -p	*/
										this_set->pixelMarks= 1;
										break;
									case 2:
										  /* -P	*/
										this_set->pixelMarks= 2;
										break;
									case 3:
										  /* -m	*/
										this_set->pixelMarks= 0;
										break;
								}
							}
							else if( !m /* && this_set->markFlag!= 2 */ ){
								this_set->markFlag= 0;
							}
							if( CheckMask(mask,8) && !this_set->barFlag && !this_set->polarFlag ){
								this_set->barFlag= 1;
							}
							else if( !CheckMask(mask,8) /* && this_set->barFlag!= 2 */ ){
								this_set->barFlag= 0;
							}
							if( CheckMask(mask,16) && !this_set->overwrite_marks ){
								this_set->overwrite_marks= 1;
							}
							else if( !CheckMask(mask,16) ){
								this_set->overwrite_marks= 0;
							}
							if( CheckMask(mask,32) && !this_set->show_legend ){
								this_set->show_legend= 1;
							}
							else if( !CheckMask(mask,32) ){
								this_set->show_legend= 0;
							}
							if( CheckMask(mask,(1<<6)) && this_set->use_error ){
								this_set->use_error= 0;
							}
							else if( !CheckMask(mask,(1<<6)) ){
								this_set->use_error= 1;
							}
							if( CheckMask(mask,(1<<7)) ){
								this_set->highlight= 1;
								if( ActiveWin && ActiveWin!= &StubWindow ){
									ActiveWin->legend_line[this_set->set_nr].highlight= this_set->highlight;
								}
							}
							else if( !CheckMask(mask,(1<<7)) ){
								this_set->highlight= 0;
								if( ActiveWin && ActiveWin!= &StubWindow ){
									ActiveWin->legend_line[this_set->set_nr].highlight= this_set->highlight;
								}
							}
							if( CheckMask(mask,(1<<8)) ){
								this_set->raw_display= 1;
							}
							else if( !CheckMask(mask,(1<<8)) ){
								this_set->raw_display= 0;
							}
							if( CheckMask(mask,(1<<9)) ){
								if( ActiveWin && ActiveWin!= &StubWindow ){
									ActiveWin->mark_set[this_set->set_nr]= 1;
								}
								else{
								  int i;
									if( !mark_set || mark_set_len>= mark_sets ){
										if( !mark_set ){
											mark_set_len= 0;
											mark_sets= MaxSets;
										}
										else{
											mark_sets*= 2;
										}
										mark_set= (int*) realloc( (char*) mark_set, (mark_sets+ 1)* sizeof(int) );
									}
									i= mark_set_len;
									if( mark_set ){
										mark_set[i]= this_set->set_nr;
										mark_set_len+= 1;
									}
									else{
										line_count= UpdateLineCount( stream, &flpos, line_count );
										fprintf( StdErr, "ReadData(): %s: line %d.%d (%s): can't reallocate mark_set buffer (%s)\n",
											filename,
											sub_div, line_count, optbuf,
											serror()
										);
										mark_set_len= -1;
										mark_sets= 0;
									}
								}
							}
							else if( !CheckMask(mask,(1<<9)) ){
								  /* We unset a mark only when we have a window (at startup, no
								   \ markers are set..)
								   */
								if( ActiveWin && ActiveWin!= &StubWindow ){
									ActiveWin->mark_set[this_set->set_nr]= 0;
								}
							}
							if( CheckMask(mask,(1<<10)) ){
								this_set->show_llines= 0;
							}
							else if( !CheckMask(mask,(1<<10)) ){
								this_set->show_llines= 1;
							}
							if( CheckMask(mask,(1<<11)) ){
								this_set->floating= 1;
							}
							else if( !CheckMask(mask,(1<<11)) ){
								this_set->floating= 0;
							}
						}
						else if( debugFlag)
							fprintf( StdErr, "\tunknown plottype specification 0x%x\n", err);
						break;
					}
				}
				{ int style= axspec-1;
				  double iw= _incr_width + incr_width;
				  char *cdef;
					if( width> 0){
						this_set->lineWidth= ((iw>0)? iw : 0.0) + width;
					}
					else{
						this_set->lineWidth= width;
					}
					if( ewidth>= 0 ){
						this_set->elineWidth= ewidth;
					}
					else{
						this_set->elineWidth= -1;
					}
					if( style>= 0 && style<= MAX_LINESTYLE ){
						this_set->linestyle= style;
						this_set->elinestyle= (estyle< 0)? -1 : estyle;
					}
					else if( style< 0 ){
						if( setNumber ){
							this_set->linestyle= (this_set[-1].linestyle + 1 ) % MAX_LINESTYLE;
							this_set->elinestyle= (estyle< 0)? -1 : estyle;
						}
						else{
							this_set->linestyle= 0;
							this_set->elinestyle= (estyle< 0)? -1 : estyle;
						}
					}
					else if( debugFlag){
						if( style>= 0 ){
							fprintf( StdErr, "\tinvalid linestyle %d (max. %d)\n",
								style, MAX_LINESTYLE
							);
						}
						else{
							fprintf( StdErr, "\tdefault linestyle %d\n",
								this_set->linestyle
							);
						}
					}
					if( mrks> 0 ){
						if( debugFlag ){
							fprintf( StdErr, "markstyle %d\n", mrks );
						}
						markstyle= this_set->markstyle= -1 * mrks;
					}
					else{
						markstyle= this_set->markstyle;
					}
					if( num>= 9 ){
						if( debugFlag ){
							fprintf( StdErr, "markSize %g\n", mSize );
						}
						this_set->markSize= mSize;
					}
					if( !XGIgnoreCNames ){
					  int has_cname, has_hl_cname;
properties_colours:;
						has_cname= 0, has_hl_cname= 0;
						if( (cdef= strstr( form, " cname=\"" )) ){
						  char *cname= &cdef[8];
							if( (cdef= index( cname, '"') ) ){
								*cdef= '\0';
							}
							XGStoreColours= 1;
							if( *cname ){
								has_cname= True;
								if( strcasecmp( cname, "default") ){
								  Pixel tempPixel;
									if( GetColor( cname, &tempPixel ) ){
										if( this_set->pixvalue< 0 ){
											FreeColor( &this_set->pixelValue, &this_set->pixelCName );
										}
										this_set->pixvalue= -1;
										this_set->pixelValue= tempPixel;
										StoreCName( this_set->pixelCName );
									}
									else{
										line_count= UpdateLineCount( stream, &flpos, line_count );
										fprintf(StdErr, "file: `%s', line: %d.%d (%s): invalid colour specification\n",
											filename, sub_div, line_count, buffer
										);
									}
								}
								else if( this_set->pixvalue< 0 ){
									xfree( this_set->pixelCName );
									this_set->pixvalue= (setNumber % MAXATTR);
								}
							}
							if( cdef ){
								*cdef= '"';
							}
						}
						if( (cdef= strstr( form, " hl_cname=\"" )) ){
						  char *cname= &cdef[11];
							if( (cdef= index( cname, '"') ) ){
								*cdef= '\0';
							}
							XGStoreColours= 1;
							if( *cname ){
								has_hl_cname= True;
								if( strcasecmp( cname, "default") ){
								  Pixel tempPixel;
									if( GetColor( cname, &tempPixel ) ){
										if( this_set->hl_info.pixvalue< 0 ){
											FreeColor( &this_set->hl_info.pixelValue, &this_set->hl_info.pixelCName );
										}
										this_set->hl_info.pixvalue= -1;
										this_set->hl_info.pixelValue= tempPixel;
										StoreCName( this_set->hl_info.pixelCName );
									}
									else{
										line_count= UpdateLineCount( stream, &flpos, line_count );
										fprintf(StdErr, "file: `%s', line: %d.%d (%s): invalid colour specification\n",
											filename, sub_div, line_count, buffer
										);
									}
								}
								else if( this_set->hl_info.pixvalue< 0 ){
									xfree( this_set->hl_info.pixelCName );
									this_set->hl_info.pixvalue= 0;
								}
							}
							if( cdef ){
								*cdef= '"';
							}
						}
						if( !has_cname && !has_hl_cname && num< 8 && num!= 4 ){
							goto invalid_properties_format;
						}
					}
				}
				this_set->propsSet= True;
				if( debugFlag){
				  char *marks[]= {"-m", "-p", "-P"};
					fprintf( StdErr,
						"*PROPERTIES* %sSet #%d: nolines=%d marks=%d[%s],%d bar=%d lineWidth=%g linestyle=%d incr_width=%g+%g"
							" elinestyle=%d elineWidth=%g",
						(new_format)? "(new format) " : "",
						setNumber, this_set->noLines, this_set->markFlag, marks[this_set->pixelMarks], this_set->markstyle,
						this_set->barFlag, this_set->lineWidth, this_set->linestyle,
						incr_width, _incr_width, this_set->elinestyle, this_set->elineWidth
					);
					if( ActiveWin && ActiveWin!= &StubWindow ){
						fprintf( StdErr, " marked=%d", ActiveWin->mark_set[this_set->set_nr] );
					}
					fputc( '\n', StdErr );
				}
				markFlag= this_set->markFlag;
				noLines= this_set->noLines;
				pixelMarks= this_set->pixelMarks;
				barFlag= this_set->barFlag;
/*
				polarFlag= this_set->polarFlag;
				radix= this_set->radix;
				radix_offset= this_set->radix_offset;
 */
			}
			else if( num!= 4 ){
invalid_properties_format:;
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "%s: line %d.%d (%s): unsupported format (skipped)\n", filename, sub_div, line_count, optbuf);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*SCALEFACT*", 11)== 0){
		  int len= strlen(optbuf)-1;
			if( len>= 0)
				optbuf[len]= '\0';
			_Xscale= _Yscale= _DYscale= 1.0;
			sscanf( &optbuf[11], "%lf %lf %lf", &_Xscale, &_Yscale, &_DYscale);
			Xscale= _Xscale;
			Yscale= _Yscale;
			DYscale= _DYscale;
			if( debugFlag)
				fprintf( StdErr, "File wide scale factors: %g %g %g\n", Xscale, Yscale, DYscale);
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*SCALFACT*", 10)== 0){
		  int len= strlen(optbuf)-1;
			if( len>= 0)
				optbuf[len]= '\0';
			sscanf( &optbuf[10], "%lf %lf %lf", &this_set->Xscale, &this_set->Yscale, &this_set->DYscale);
			if( debugFlag)
				fprintf( StdErr, "Set specific scale factors: %g %g %g\n", this_set->Xscale, this_set->Yscale, this_set->DYscale);
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*COLUMNS*", 9)== 0){
		  int len= strlen(optbuf)-1, Nc, xc, yc, ec, vc= -1;
		  int nr1= 0, nr2= 0, nr3= 0, nc, set_link= -1;
		  char *c= &optbuf[9];
			if( len>= 0)
				optbuf[len]= '\0';
			while( *c && isspace(*c) ){
				c++;
			}
			line_count= UpdateLineCount( stream, &flpos, line_count );
			if( this_set->set_link>= 0 ){
				fprintf( StdErr, "%s: line %d.%d (%s): modifying or removing set linkage (was to set %d)\n",
					filename, sub_div, line_count, optbuf, this_set->set_link
				);
			}
			if( this_set->numPoints && this_set->set_link< 0 ){
				fprintf( StdErr, "%s: line %d.%d (%s): ignoring *COLUMNS* specification for non-empty set\n",
					filename, sub_div, line_count, optbuf
				);
			}
			else if(
				(nr1= sscanf( c, "N=%d x=%d y=%d e=%d l=%d n=%d L2=%d\n", &Nc, &xc, &yc, &ec, &vc, &nc, &set_link))>= 6
				|| (nr2= sscanf( c, "N=%d x=%d y=%d e=%d n=%d L2=%d\n", &Nc, &xc, &yc, &ec, &nc, &set_link))>= 5
				|| (nr3= sscanf( c, "N=%d x=%d y=%d e=%d L2=%d\n", &Nc, &xc, &yc, &ec, &set_link))>= 4
			){
				CLIP_EXPR( Nc, Nc, 3, Nc );
				CLIP_EXPR( this_set->xcol, xc, 0, Nc- 1);
				CLIP_EXPR( this_set->ycol, yc, 0, Nc- 1);
				CLIP_EXPR( this_set->ecol, ec, -(Nc-1), Nc- 1);
				if( nr1> 4 ){
					CLIP_EXPR( this_set->lcol, vc, -1, Nc- 1);
				}
				if( nr1> 5 || nr2> 4 ){
					CLIP_EXPR( this_set->Ncol, nc, nc, Nc- 1);
				}
				if( ec< 0 ){
					 this_set->use_error= 0;
					 this_set->ecol*= -1;
				}
				if( (nr3> 4 || nr2> 5 || nr1> 6) && set_link>= 0 ){
					if( first_set && CorrectLinks && set_link+first_set!= this_set->set_nr ){
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "ReadData(), %s, line %d.%d: set_link %d tentatively corrected to %d\n",
								filename,
								sub_div, line_count, set_link, set_link+ first_set
							);
						}
						set_link+= first_set;
					}
					if( set_link!= this_set->set_nr ){
						if( set_link< setNumber ){
							LinkSet2( this_set, set_link );
							Nc= this_set->ncols;
							spot= AllSets[set_link].numPoints;
							Spot= AllSets[set_link].numPoints;
						}
						else{
							fprintf( StdErr, "%s: line %d.%d (%s): set #%d links in the future to a not-yet-existing set %d\n",
								filename, sub_div, line_count, optbuf, this_set->set_nr, set_link
							);
							LinkSet2( this_set, set_link );
						}
					}
					else{
						fprintf( StdErr, "%s: line %d.%d (%s): ignoring link-to-self attempt for set #%d\n",
							filename, sub_div, line_count, optbuf, this_set->set_nr
						);
					}
				}
				else{
					  /* Something that *may* happen: a previously linked set is converted into a non-linked set.	*/
					  /* RJVB 20081126: corrected erroneous?? nr1>5 check to nr2>5 below */
					if( (nr3> 4 || nr2> 5 || nr1> 6) && this_set->set_link>= 0 ){
						  /* Actually, 1 point left!	*/
						this_set->numPoints= 1;
						  /* De-allocate most of the allocated things:	*/
						realloc_points( this_set, 1, False );
						  /* Make sure to forget the columns pointer we were using!	*/
						this_set->columns= NULL;
					}
					set_link= this_set->set_link= -1;
				}
				if( debugFlag){
					fprintf( StdErr, "*COLUMNS* : N=%d x=%d y=%d e=%d l=%d", Nc,
						this_set->xcol, this_set->ycol, this_set->ecol, this_set->lcol
					);
#if ADVANCED_STATS == 2
					fprintf( StdErr, " n=%d", this_set->Ncol );
#endif
					fputc( '\n', StdErr );
				}
				if( set_link< 0 ){
					if( Nc> this_set->ncols && this_set->allocSize ){
						allocerr= 0;
						if( !(this_set->columns= realloc_columns( this_set, Nc )) ){
							line_count= UpdateLineCount( stream, &flpos, line_count );
							fprintf( StdErr,
								"%s: line %d.%d (%s): %d allocation failure(s) (%s)\n", filename, sub_div, line_count, optbuf,
								allocerr, serror()
							);
							xg_abort();
						}
					}
					else{
						this_set->ncols= Nc;
					}
				}
				if( Nc> MaxCols ){
					MaxCols= Nc;
					if( BinaryFieldSize ){
						AllocBinaryFields( Nc, "ReadData()" );
					}
				}
				if( WindowList ){
					set_Columns(this_set);
				}

				xfree(extremes);
				extremesN= 0;
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "%s: line %d.%d (%s): unsupported format (skipped)\n", filename, sub_div, line_count, optbuf);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*ASSOCIATE*", 11)== 0){
		  char *membuf= cleanup( &optbuf[11] );
			if( membuf ){
			  int n= param_scratch_len;
			  char *TBh= TBARprogress_header;
				clean_param_scratch();
				strcat( membuf, "\n");
				TBARprogress_header= "*ASSOCIATE*";
				fascanf( &n, membuf, param_scratch, NULL, NULL, NULL, NULL );
				TBARprogress_header= TBh;
				if( n ){
					if( (this_set->Associations= (double*) realloc( this_set->Associations,
							sizeof(double)* (this_set->allocAssociations+ n))
						)
					){
					  int i;
						for( i= 0; i< n; i++ ){
							this_set->Associations[i+ this_set->numAssociations]= param_scratch[i];
						}
						if( (this_set->allocAssociations+= n)> ASCANF_MAX_ARGS ){
							Ascanf_AllocMem( this_set->allocAssociations );
						}
						this_set->numAssociations+= n;
					}
					else{
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf( StdErr, "%s: line %d.%d (%s): can't (re)allocate memory for %d associations (%d new) (%s)\n",
							filename, sub_div, line_count, optbuf,
							this_set->allocAssociations+ n, n, serror()
						);
					}
				}
				else{
					xfree( this_set->Associations );
					this_set->numAssociations= this_set->allocAssociations= 0;
					if( debugFlag ){
						fprintf( StdErr, "%s: line %d.%d (%s): removed associations for set #%d\n",
							filename, sub_div, line_count, optbuf,
							this_set->set_nr
						);
					}
				}
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						if( DumpDHex ){
							fputs( "# ", stdout );
						}
						print_string( stdout, "*ASSOCIATE*", "\\n\n", "\n", membuf);
						if( DumpDHex ){
						  int i, ddh= d2str_printhex;
							d2str_printhex= False;
							fprintf( stdout, "*ASSOCIATE* %s", d2str( this_set->Associations[0], d3str_format, NULL) );
							for( i= 1; i< this_set->numAssociations; i++ ){
								fprintf( stdout, ",%s", d2str( this_set->Associations[i], d3str_format, NULL) );
							}
							fputc( '\n', stdout );
							d2str_printhex= ddh;
						}
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*ELSE*", 6)== 0 || strncmp( optbuf, "*ELIF*", 6)== 0 ){
			if( !IF_Frame || IF_Frame->alternative ){
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "%s: line %d.%d (%s): stray *ELSE*/*ELIF* command ignored",
					filename, sub_div, line_count, optbuf
				);
				if( IF_Frame ){
					fprintf( StdErr, " (active *IF* statement %s already had an *ELSE*/*ELIF*)", IF_Frame->expr );
				}
				fputc( '\n', StdErr );
			}
			else{
				if( IF_level== IF_Frame->level ){
					if( debugFlag || scriptVerbose ){
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf( StdErr, "%s: line %d.%d (%s): *ELSE*/*ELIF* command level %d (belongs to %s?!): skipping until next *ENDIF* or EOF\n",
							filename, sub_div, line_count, optbuf, IF_level, IF_Frame->expr
						);
					}
					IF_Frame->alternative= True;
					if( optbuf[3]== 'I' ){
					  /* An *ELIF* statement is really an ELSE followed by an IF...! */
						IF_level+= 1;
					}
					goto EXTRATEXT;
				}
			}
		}
		else if( strncmp( optbuf, "*ENDIF*", 7)== 0 ){
			if( !IF_Frame ){
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "%s: line %d.%d (%s): stray *ENDIF* command ignored",
					filename, sub_div, line_count, optbuf
				);
				fputc( '\n', StdErr );
			}
			else{
				if( IF_level== IF_Frame->level ){
					if( debugFlag || scriptVerbose ){
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf( StdErr, "%s: line %d.%d (%s): *ENDIF* command level %d (belongs to %s?!)\n",
							filename, sub_div, line_count, optbuf, IF_level, IF_Frame->expr
						);
					}
					IF_Frame= IF_Frame_Delete( IF_Frame, False );
				}
				IF_level-= 1;
			}
		}
		else if( strncmp( optbuf, "*IF*", 4)== 0 || strncmp( optbuf, "*CONDITIONAL_EXTRATEXT*", 23)== 0 ){
		  char *membuf, *opcode;
		  int is_if;
		  double result;
IF_statement:;
			if( optbuf[1]== 'I' ){
				membuf= cleanup( &optbuf[4] );
				opcode= "*IF*";
				is_if= True;
			}
			else if( optbuf[1]== 'E' ){
				membuf= cleanup( &optbuf[6] );
				opcode= "*ELIF*";
				is_if= True;
			}
			else{
				membuf= cleanup( &optbuf[23] );
				opcode= "*CONDITIONAL_EXTRATEXT*";
				is_if= False;
			}
			if( strlen( membuf ) ){
			  char *pbuf;
			  char *TBh= TBARprogress_header;
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( DumpPretty ){
							LineDumped= True;
							print_string( stdout, opcode, "\\n\n", "\n\n", membuf );
						}
					}
				TBARprogress_header= opcode;
				pbuf = SubstituteOpcodes( membuf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
					"*This-FileDir*", dirname(the_file),
					"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str,
#if defined(__MACH__) || defined(__APPLE_CC__)
					"*Darwin*", "1", "*MacOSX*", "1", "*MACH*", "1",
					"*Cygwin*", "0",
					"*linux*", "0",
#elif defined(__CYGWIN__)
					"*Darwin*", "0", "*MacOSX*", "0", "*MACH*", "0",
					"*Cygwin*", "1",
					"*linux*", "0",
#elif defined(linux)
					"*Darwin*", "0", "*MacOSX*", "0", "*MACH*", "0",
					"*Cygwin*", "0",
					"*linux*", "1",
#endif
					NULL
				);
				new_param_now( pbuf, &ReadData_proc.param_range[0], 1 );
				result= ReadData_proc.param_range[0];
				if( is_if ){
				  char *IF_command= concat( opcode, " ", pbuf, NULL );
					IF_Frame= IF_Frame_AddItem( IF_Frame, IF_command, ++IF_level );
					IF_Frame->alternative= False;
					xfree( IF_command );
					if( result && !NaN(result) ){
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "%s\t level %d: evals TRUE (%s): evaluating until next *ELSE* or *ENDIF* or EOF\n",
								buffer, IF_level, ad2str( result, d3str_format, 0)
							);
						}
					}
					else{
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "%s\t level %d: evals FALSE (%s): skipping until next *ELSE* or *ENDIF* or EOF\n",
								buffer, IF_level, ad2str( result, d3str_format, 0)
							);
						}
						goto EXTRATEXT;
					}
				}
				else{
					strcat( pbuf, "\n");
					if( result && !NaN(result) ){
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "%s\t: evals TRUE: skipping until next empty line\n", buffer );
						}
						goto EXTRATEXT;
					}
					if( debugFlag || scriptVerbose ){
						fprintf( StdErr, "%s\t: evals FALSE: evaluating following lines\n", buffer );
					}
				}
				if( pbuf!= membuf ){
					xfree(pbuf);
				}
				TBARprogress_header= TBh;
			}
			else if( debugFlag || scriptVerbose ){
				fprintf( StdErr, "%s\t: missing test-expression: evaluating following lines\n", buffer );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*EXTRATEXT*", 11)== 0){
		  char *c;
			if( optbuf[11] ){
				add_comment( &optbuf[11] );
			}
EXTRATEXT:;
			c= optbuf;
			if( DumpFile && !LineDumped ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			  /* ignore	*/;
			while( c && (IF_Frame || optbuf[0]!= '\n') && !feof(stream) ){
/* 				c= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );	*/
				c= fgets( buffer, LMAXBUFSIZE, stream );
				if( c ){
					line_count++;
					if( DumpFile ){
						fputs( buffer, stdout );
					}
					if( IF_Frame ){
					  char *buf= buffer;
						while( isspace( (unsigned char) *buf ) ){
							buf++;
						}
						if( strncmp( buf, "*IF*", 4)== 0 ){
							IF_level+= 1;
						}
						else if( strncmp( buf, "*ELSE*", 6)== 0 || strncmp( buf, "*ELIF*", 6)== 0 ){
							if( IF_level== IF_Frame->level ){
								if( !IF_Frame->alternative ){
									IF_Frame->alternative= True;
									if( debugFlag || scriptVerbose ){
										fprintf( StdErr, "%s: line %d.%d (%s): *ELSE*/*ELIF* command level %d (belongs to %s?!): turning back on evaluation.\n",
											filename, sub_div,
											(line_count= UpdateLineCount( stream, &flpos, line_count)), optbuf, IF_level, IF_Frame->expr
										);
									}
									c= NULL;
									if( buf[3]== 'I' ){
										optbuf= buf;
										goto IF_statement;
									}
								}
								else{
									fprintf( StdErr, "%s: line %d.%d (%s): stray *ELSE*/*ELIF* command level %d (belongs to %s?!) ignored\n",
										filename, sub_div,
										(line_count= UpdateLineCount( stream, &flpos, line_count)), optbuf, IF_level, IF_Frame->expr
									);
								}
							}
						}
						else if( strncmp( buf, "*ENDIF*", 7)== 0 ){
							if( IF_level== IF_Frame->level ){
								if( debugFlag || scriptVerbose ){
									fprintf( StdErr, "%s: line %d.%d (%s): *ENDIF* command level %d (belongs to %s?!): turning back on evaluation.\n",
										filename, sub_div,
										(line_count= UpdateLineCount( stream, &flpos, line_count)), optbuf, IF_level, IF_Frame->expr
									);
								}
								c= NULL;
								IF_Frame= IF_Frame_Delete( IF_Frame, False );
							}
							IF_level-= 1;
						}
					}
					else{
						Add_Comment( buffer, True );
					}
				}
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*ECHO*", 6)== 0 || strncmp( optbuf, "*POPUP*", 7)== 0 ){
		  char *c= &optbuf[6] /* , *cb= comment_buf */;
		  int /* cs= comment_size, */ nc= NoComment;
		  Boolean popup= False, store_dp_descr= False;
		  FILE *fp= StdErr;
		  Sinc List;
			if( *c== '*' && c[-1]== 'P' ){
				c++;
/* 				comment_buf= NULL;	*/
/* 				comment_size= 0;	*/
				List.sinc.string= NULL;
				Sinc_string_behaviour( &List, NULL, 0,0, SString_Dynamic );
				Sflush( &List );
				NoComment= False;
				popup= True;
			}
			else{
			  char *d= strcasestr( c, "stdout" );
				if( d ){
					fp= stdout;
					c= d+ 6;
				}
				else if( (d= strcasestr(c, "verbose")) ){
					if( !debugFlag && !scriptVerbose ){
						fp= NullDevice;
					}
					c= d+ 7;
				}
			}
			if( DumpFile ){
				fputs( optbuf, stdout );
				if( optbuf[strlen(optbuf)-1]!= '\n' ){
					fputc( '\n', stdout );
				}
			}
/* 			line_count-= 1;	*/
			do
			{
				if( c ){
					if( strncasecmp( c, "*DATA_PROCESS_DESCRIPTION*", 26)== 0 ){
						store_dp_descr= True;
					}
					else if( strncmp( c, "*FINFO*", 7)== 0 ){
					  char buf[MAXPATHLEN*2+16];
					  int l= sprintf( buf, "Current file: " );
					  char *fn= (ReadData_thisCurrentFileName)? ReadData_thisCurrentFileName : the_file;
						time_stamp( (ReadData_thisCurrentFP)? ReadData_thisCurrentFP : stream,
							(fn)? fn : "stdin", &buf[l], True, "\n"
						);
						fputs( buf, fp );
						if( popup ){
							Add_SincList( &List, buf, True );
						}
						if( store_dp_descr ){
							ReadData_proc.description= concat2( ReadData_proc.description, buf, NULL );
						}
					}
					else if( strncmp( c, "*PINFO*", 7)== 0 ){
					  char *pinfo= NULL;
						fputs( "Current xgraph info:\n", fp );
						_Dump_Arg0_Info( ActiveWin, fp, &pinfo, False );
						if( popup && pinfo ){
							Add_SincList( &List, "Current xgraph info:\n", True );
							Add_SincList( &List, pinfo, True );
						}
						xfree( pinfo );
					}
					else{
					  char *pbuf= SubstituteOpcodes( c, "*This-File*", the_file, "*This-FileName*", basename(the_file),
					  	"*This-FileDir*", dirname(the_file),
					  	"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL);
						fputs( pbuf, fp );
						if( popup ){
							Add_SincList( &List, pbuf, True );
						}
						if( store_dp_descr ){
							ReadData_proc.description= concat2( ReadData_proc.description, pbuf, NULL );
						}
						if( pbuf!= c ){
							xfree(pbuf);
						}
					}
					line_count++;
				}
				if( buffer[0]!= '\n' ){
					c= fgets( buffer, LMAXBUFSIZE, stream );
					c= buffer;
					if( DumpFile ){
						fputs( c, stdout );
					}
				}
				else{
					c= NULL;
				}
			}
			while( c && !feof(stream) );
			if( popup ){
				XG_error_box( &ActiveWin, "*POPUP*", List.sinc.string, NULL );
				xfree( List.sinc.string );
/* 				comment_buf= cb;	*/
/* 				comment_size= cs;	*/
				NoComment= nc;
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*VAL_CAT_X_FONT*", 16)== 0 || strncmp( optbuf, "*VAL_CAT_Y_FONT*", 16)== 0 ||
			strncmp( optbuf, "*VAL_CAT_I_FONT*", 16)== 0
		){
		  AxisName ax= (optbuf[9]== 'X')? X_axis : Y_axis;
		  char *xfn= NULL, *axfn= NULL, *psfn= NULL, an= optbuf[9];
		  double pssize= 0;
		  int psreencode= 1;
		  CustomFont *cf= NULL;
		  int ok= 1, pspt= 0;
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			while( optbuf[0]!= '\n' && !feof(stream) && ok ){
			  char *inp= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
				if( DumpFile ){
					fputs( buffer, stdout );
				}
				if( inp ){
					if( strncasecmp( inp, "X=", 2)== 0 ){
						xfn= strdup(cleanup(&inp[2]));
					}
					else if( strncasecmp( inp, "X2=", 3)== 0 ){
						axfn= strdup(cleanup(&inp[3]));
					}
					else if( strncasecmp( inp, "PS=", 3)== 0 ){
						psfn= strdup(cleanup(&inp[3]));
					}
					else if( strncasecmp( inp, "PSSize=", 7)== 0 ){
					  int n= 1;
						if( fascanf2( &n, cleanup(&inp[7]), &pssize, ',')!= EOF && n== 1){
							pspt= 1;
						}
					}
					else if( strncasecmp( inp, "PSReEncode=", 11)== 0 ){
						psreencode= atoi( cleanup( &inp[11] ) );
					}
				}
				line_count++;
			}
			if( ok && xfn && psfn && pspt ){
			  extern CustomFont *Init_CustomFont();
				if( !(cf= Init_CustomFont( xfn, axfn, psfn, pssize, psreencode )) ){
					fprintf( StdErr, "*ValCat_%c_Font*: can't initialise CustomFont..\n", an );
				}
				xfree( xfn);
				xfree( axfn);
				xfree( psfn );
			}
			if( cf ){
			  extern CustomFont *VCat_XFont, *VCat_YFont, *VCat_IFont;
				if( ax== X_axis ){
					Free_CustomFont( VCat_XFont );
					xfree( VCat_XFont );
					VCat_XFont= cf;
				}
				else if( an== 'I' ){
					Free_CustomFont( VCat_IFont );
					xfree( VCat_IFont );
					VCat_IFont= cf;
				}
				else{
					Free_CustomFont( VCat_YFont );
					xfree( VCat_YFont );
					VCat_YFont= cf;
				}
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*VAL_CAT_X*", 11)== 0 || strncmp( optbuf, "*VAL_CAT_Y*", 11)== 0 ||
			strncmp( optbuf, "*VAL_CAT_I*", 11)== 0
		){
		  char *c= cleanup(&optbuf[11]);
		  AxisName ax= (optbuf[9]== 'X')? X_axis : ((optbuf[9]== 'I')? I_axis : Y_axis);
		  ValCategory *vcat;
		  int ok= 1, N;
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			Add_Comment( buffer, True );
			if( ActiveWin && ActiveWin!= &StubWindow ){
				vcat= (ax== X_axis)? ActiveWin->ValCat_X : ((ax== I_axis)? ActiveWin->ValCat_I : ActiveWin->ValCat_Y);
			}
			else{
				vcat= (ax== X_axis)? VCat_X : ((ax== I_axis)? VCat_I : VCat_Y);
			}
			if( strncmp( c, "new", 3)== 0 ){
				if( debugFlag ){
					fprintf( StdErr, "Creating new %c Value<>Category\n",
						optbuf[9]
					);
				}
				vcat= Free_ValCat( vcat );
				vcat= NULL;
				N= 0;
			}
			else{
				N= ValCat_N( vcat );
			}
			while( c && optbuf[0]!= '\n' && !feof(stream) && ok ){
			  char *val_cat;
			  double val;
			  int n= 1;
				val_cat= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
				if( DumpFile ){
					fputs( buffer, stdout );
				}
				if( val_cat && (c= index( val_cat, ',')) ){
					if( fascanf2( &n, val_cat, &val, ',')!= EOF && n== 1){
					  char *e;
						  /* Remove trailing newline, if any	*/
						if( *(e= &c[strlen(c)-1])== '\n' ){
							*e= '\0';
						}
						if( !(vcat= Add_ValCat( vcat, &N, val, parse_codes(&c[1]) )) ){
							ok= 0;
						}
						else{
							Add_Comment( buffer, True );
						}
					}
				}
				line_count++;
			}
			if( ActiveWin && ActiveWin!= &StubWindow ){
				if( ax== X_axis ){
					ActiveWin->ValCat_X= vcat;
				}
				else if( ax== I_axis ){
					ActiveWin->ValCat_I= vcat;
				}
				else{
					ActiveWin->ValCat_Y= vcat;
				}
			}
			else{
				if( ax== X_axis ){
					VCat_X= vcat;
				}
				else if( ax== I_axis ){
					VCat_I= vcat;
				}
				else{
					VCat_Y= vcat;
				}
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*COLUMNLABELS*", 14)== 0 ){
		  char *c= cleanup(&optbuf[14]);
		  LabelsList *llist;
		  unsigned short ok= 1;
		  int N;
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			Add_Comment( buffer, True );
			if( ActiveWin && ActiveWin!= &StubWindow ){
				llist= ActiveWin->ColumnLabels;
			}
			else{
				llist= ColumnLabels;
			}
			if( strncmp( c, "new", 3)== 0 ){
				if( debugFlag ){
					fprintf( StdErr, "Creating new %c LabelsList\n",
						optbuf[1]
					);
				}
				llist= Free_LabelsList( llist );
				llist= NULL;
				N= 0;
			}
			else{
				N= LabelsList_N( llist );
			}
			while( c && optbuf[0]!= '\n' && !feof(stream) && ok ){
			  char *label;
			  double val;
			  int n= 1;
				label= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
				if( DumpFile ){
					fputs( buffer, stdout );
				}
				if( label && (c= index( label, ',')) ){
					if( fascanf2( &n, label, &val, ',')!= EOF && n== 1){
					  char *e;
						  /* Remove trailing newline, if any	*/
						if( *(e= &c[strlen(c)-1])== '\n' ){
							*e= '\0';
						}
						if( !(llist= Add_LabelsList( llist, &N, (int) val, parse_codes(&c[1]) )) ){
							ok= 0;
						}
						else{
							Add_Comment( buffer, True );
						}
					}
				}
				line_count++;
			}
			if( ActiveWin && ActiveWin!= &StubWindow ){
				ActiveWin->ColumnLabels= llist;
			}
			else{
				ColumnLabels= llist;
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*LABELS*", 8)== 0 ){
		  char *c;
		  LabelsList *llist;
		  unsigned short ok= 1;
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( optbuf, stdout );
				if( optbuf[strlen(optbuf)-1]!= '\n' ){
					fputs( "\n", stdout );
				}
			}
			c= cleanup(&optbuf[8]);
			Add_Comment( buffer, True );
			llist= this_set->ColumnLabels;
			if( strncmp( c, "new", 3)== 0 ){
				if( debugFlag ){
					fprintf( StdErr, "Creating new %c LabelsList\n",
						optbuf[1]
					);
				}
				llist= Free_LabelsList( llist );
				llist= NULL;
			}
			while( !feof(stream) && ok ){
			  char *labels, *e;
				c= labels= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
				if( DumpFile ){
					fputs( buffer, stdout );
				}
				if( c && *(e= &c[strlen(c)-1])== '\n' ){
					*e= '\0';
				}
				if( labels && *labels ){
					llist= Parse_SetLabelsList( llist, labels, data_separator, 0, NULL );
				}
				else{
					ok= False;
				}
				line_count++;
			}
			this_set->ColumnLabels= llist;
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*AVERAGE*", 9)== 0 ){
		  int i, n= MaxSets;
		  char *buf= &optbuf[9];
		  Boolean XYAveraging= True, add_Interpol= False, ScattEllipse= False;
		  char YAv_Xsort[5]= "Nsrt";
#ifdef __GNUC__
		  double vals[MaxSets];
		  int av_set[MaxSets];
#else
		  double *vals= calloc( MaxSets, sizeof(double) );
		  int *av_set= calloc( MaxSets, sizeof(int) );
		  if( !vals || !av_set ){
			  fprintf( StdErr, "\"%s\": cannot allocate scratch memory (%s)\n", serror() );
			  xfree( vals );
			  xfree( av_set );
		  }
		  else{
#endif
			while( isspace((unsigned char)*buf) ){
				buf++;
			}
			if( strncmp( buf, "ScattEllipse", 12)== 0 ){
				ScattEllipse= True;
				buf+= 12;
			}
			while( (isspace((unsigned char)*buf) || *buf== 'Y') && *buf ){
				if( buf[0]== 'Y' ){
					XYAveraging= False;
					if( buf[1]== '-' ){
						strncpy( YAv_Xsort, &buf[2], 4);
						YAv_Xsort[4]= '\0';
						buf+= 5;
					}
				}
				buf++;
			}
			while( (isspace((unsigned char)*buf) || *buf== 'I') && *buf ){
				if( buf[0]== 'I' ){
					add_Interpol= True;
				}
				buf++;
			}
			for( i= 0; i< MaxSets; i++ ){
				vals[i]= 0;
				av_set[i]= 0;
			}
			if( fascanf2( &n, buf, vals, ' ')!= EOF ){
				  /* determine which sets must be drawn	*/
				for( i= 0; i< n; i++ ){
				  int s= (int) vals[i];
					if( s< 0 || s>= setNumber ){
						fprintf( StdErr, "\"%s\": invalid set# %d\n", buffer, s );
					}
					else{
						av_set[s]= 1;
					}
				}
				if( this_set->setName && legend_setNumber!= setNumber ){
					xfree( this_set->setName );
					this_set->setName= NULL;
				}
				{ int s= spot;
					line_count= UpdateLineCount( stream, &flpos, line_count );
					spot= Average( NULL, av_set, filename, sub_div, line_count, buffer, &ReadData_proc,
						False, XYAveraging, YAv_Xsort, add_Interpol, ScattEllipse
					);
					if( spot>= 0 ){
						Spot+= (spot- s);
					}
					else{
						spot= s;
					}
				}
			}
			else{
				fprintf( StdErr, "\"%s\": need numbers of sets to average...\n", optbuf );
			}
			ReadData_commands+= 1;
#ifndef __GNUC__
			xfree( vals );
			xfree( av_set );
		  }
#endif
		}
		else if( strncmp( optbuf, "*ULABEL*", 8)== 0 ){
#define N_UL_PARS	10
		  double vals[N_UL_PARS]= {0,0,0,0,0,0,1, -1, 0, 0};
		  double lWidth;
		  int n= sizeof(vals)/sizeof(double), pixvalue= 0, pixlinked= 0;
		  char *buf= &optbuf[8], *cdef= NULL, *pixelCName= NULL, ULtype[3];
		  Pixel tempPixel;
			while( isspace((unsigned char)*buf) && *buf ){
				buf++;
			}
			  /* Now parse any present colourname; this messes up the inputbuffer...	*/
			if( !XGIgnoreCNames && (cdef= strstr( buf, "cname=\"" )) ){
				*cdef= '\0';
			}
			ULtype[2]= ULtype[1]= ULtype[0]= '\0';
			if( strncasecmp( buf, "reset", 5)== 0 ){
				Delete_ULabels( (ActiveWin)? ActiveWin : &StubWindow );
				n= -1;
			}
			else{
				n= sscanf( buf,
					"%lf %lf %lf %lf set=%lf transform?=%lf draw?=%lf lpoint=%lf vertical?=%lf nobox?=%lf lWidth=%lf type=%c%c",
					&vals[0], &vals[1], &vals[2], &vals[3], &vals[4],
					&vals[5], &vals[6], &vals[7], &vals[8], &vals[9], &lWidth,
					&ULtype[0], &ULtype[1]
				);
			}
			if( n> 0 && (n< N_UL_PARS ) ){
				n= N_UL_PARS;
				if( fascanf2( &n, buf, vals, ' ')== EOF || n< 5 ){
					n= 0;
				}
			}
			if( n>= 5 ){
			  UserLabel *new= (UserLabel*) calloc( sizeof(UserLabel), 1);
			  LocalWin *wi= ActiveWin;
				if( !wi ){
					wi= &StubWindow;
				}
				check_wi( &wi, "ReadData()");
				if( new ){
					  /* Dump the inputbuffer if requested and before we start messing around with it	*/
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						fputs( buffer, stdout );
					}
					  /* Now parse any present colourname; this messes up the inputbuffer...	*/
					if( !XGIgnoreCNames && cdef ){
					  char *cname= &cdef[7];
						  /* Put back in the 'c' that we removed above:	*/
						*cdef= 'c';
						if( (cdef= index( cname, '"') ) ){
							*cdef= '\0';
						}
						XGStoreColours= 1;
						if( *cname ){
							if( strcasecmp( cname, "default")== 0 ||
								(strncasecmp( cname, "default", 7)== 0 && isspace(cname[7]))
							){
								pixvalue= 0;
								pixlinked= 0;
							}
							else if( strcasecmp( cname, "linked")== 0 ||
								(strncasecmp( cname, "linked", 6)== 0 && isspace(cname[6]))
							){
								pixvalue= 0;
								pixlinked= 1;
							}
							else if( GetColor( cname, &tempPixel ) ){
								pixvalue= -1;
								StoreCName( pixelCName );
							}
							else{
								line_count= UpdateLineCount( stream, &flpos, line_count );
								fprintf(StdErr, "file: `%s', line: %d.%d (%s): invalid colour specification\n",
									filename, sub_div, line_count, buffer
								);
							}
						}
						  /* cut at colour definition to avoid interference with following parsing	*/
						*cname= '\0';
					}
					new->x1= vals[0];
					new->y1= vals[1];
					new->x2= vals[2];
					new->y2= vals[3];
					new->set_link= (int) vals[4];
					new->pixvalue= pixvalue;
					new->pixlinked= pixlinked;
					new->pixelValue= tempPixel;
					new->pixelCName= pixelCName;
					  /* If we're reading a file as not-the-first, or when we're including it,
					   \ a set's serial number is actually this_set->set_nr. Linked sets are dumped
					   \ just after the set they're linked to, so we can do the following test to see
					   \ whether we're reading linked-label data (only after having read actual data, or
					   \ after a *Set* statement - we can also be reading just label data!):
					   */
					if( (read_data || set_set) && new->set_link>= 0 ){
						if( CorrectLinks ){
							if( new->set_link== spec_set ){
								if( spec_set!= real_set ){
									if( debugFlag || scriptVerbose ){
										line_count= UpdateLineCount( stream, &flpos, line_count );
										fprintf( StdErr,
											"ReadData(), %s, line %d.%d: label's set_link %d corrected to %d "
											"(real setnr of last *Set* command)\n",
											filename,
											sub_div, line_count, new->set_link, real_set
										);
									}
								}
								new->set_link= real_set;
							}
							else if( (new->set_link+ first_set== this_set->set_nr) ){
								if( debugFlag || scriptVerbose ){
									line_count= UpdateLineCount( stream, &flpos, line_count );
									fprintf( StdErr, "ReadData(), %s, line %d.%d: label's set_link %d tentatively corrected to %d\n",
										filename,
										sub_div, line_count, new->set_link, this_set->set_nr
									);
								}
								new->set_link= this_set->set_nr;
							}
							else if( (new->set_link+ first_set== this_set->set_nr- 1) ){
								if( debugFlag || scriptVerbose ){
									line_count= UpdateLineCount( stream, &flpos, line_count );
									fprintf( StdErr, "ReadData(), %s, line %d.%d: label's set_link %d corrected to previous set %d\n",
										filename,
										sub_div, line_count, new->set_link, this_set->set_nr- 1
									);
								}
								new->set_link= this_set->set_nr- 1;
							}
						}
					}
					new->do_transform= (int) vals[5];
					new->do_draw= (int) vals[6];
					new->pnt_nr= (int) vals[7];
					new->vertical= (int) vals[8];
					if( n> 9 ){
						new->nobox= (int) vals[9];
					}
					else{
						if( new->x1!= new->x2 || new->y1!= new->y2 ){
							new->nobox= 0;
						}
						else{
							new->nobox= 1;
						}
					}
					if( n>= N_UL_PARS ){
						new->lineWidth= lWidth;
						new->type= Parse_ULabelType( ULtype );
					}
					else{
						new->lineWidth= axisWidth;
						new->type= UL_regular;
					}
					while( buf && optbuf[0]!= '\n' && !feof(stream) && strlen(new->label)< MAXBUFSIZE ){
						buf= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
						if( buf && DumpFile ){
							fputs( buffer, stdout );
						}
						if( buf && buffer[0] && buffer[0]!= '\n' ){
						  char *c;
						  Boolean append_newline;
							if( new->set_link>= 0 && new->set_link< MaxSets ){
							  DataSet *lset= &AllSets[new->set_link];
								if( strcmp( buffer, "*TITLE*\n")== 0 && lset->titleText ){
									c= lset->titleText;
									append_newline= True;
								}
								else if( strcmp( buffer, "*LEGEND*\n")== 0 && lset->setName ){
									c= lset->setName;
									append_newline= True;
								}
								else if( strcmp( buffer, "*FILE*\n")== 0 && lset->fileName ){
									c= lset->fileName;
									append_newline= True;
								}
								else{
									c= buffer;
									append_newline= False;
								}
							}
							else{
								c= buffer;
								append_newline= False;
							}
							strncat( new->label, c, MAXBUFSIZE- strlen(new->label) );
							if( append_newline ){
								strncat( new->label, "\n", MAXBUFSIZE- strlen(new->label) );
							}
						}
						line_count++;
					}
					  /* 20040922: */
					{ char *c= &(new->label[ strlen(new->label)-1 ]);
						if( *c== '\n' ){
							*c= '\0';
						}
					}
#if 0
					if( !wi->ulabel ){
						wi->ulabel= new;
					}
					else{
						new->next= wi->ulabel;
						wi->ulabel= new;
					}
#else
					wi->ulabel= Install_ULabels( wi->ulabel, new, False );
#endif
					wi->ulabels+= 1;
					LineDumped= True;
				}
				else{
					fprintf( StdErr, "\"%s\": can't allocate new UserLabel structure (%s)\n",
						optbuf, serror()
					);
					xfree( pixelCName );
				}
				ReadData_commands+= 1;
			}
			else if( n>=0 ){
				fprintf( StdErr, "\"%s\": need 4 coordinates + a set to link to\n", optbuf );
				xfree( pixelCName );
			}
		}
		else if( strncmp( optbuf, "*ELLIPSE*", 9)== 0 ){
		  double x, y, rx, ry, skew= 0, vals[6];
		  int points;
		  int n= 6;
#ifdef ELLIPSE_SSCANF
			if( sscanf( &optbuf[9], "%lf %lf %lf %lf %d %lf",
					&x, &y, &rx, &ry, &points, &skew
				)>= 5
#else
		  char *buf= &optbuf[9];
			while( isspace((unsigned char)*buf) && *buf ){
				buf++;
			}
			if( fascanf2( &n, buf, vals, ' ')!= EOF && n>= 5
#endif
			){
#ifndef ELLIPSE_SSCANF
				x= vals[0];
				y= vals[1];
				rx= vals[2];
				ry= vals[3];
				points= (int) vals[4];
				if( n== 6 ){
					skew= vals[5];
				}
#endif
				if( !this_set->numPoints ){
					this_set->ncols= 3;
					this_set->xcol= 0;
					this_set->ycol= 1;
					this_set->ecol= 2;
					this_set->lcol= -1;
					this_set->Ncol= -1;
				}
				set_Columns( this_set );
				line_count= UpdateLineCount( stream, &flpos, line_count );
				AddEllipse( &this_set, x, y, rx, ry, points, skew,
					&spot, &Spot, data, column, filename, sub_div, line_count, &flpos, buffer, &ReadData_proc
				);
			}
			else{
				fprintf( StdErr, "\"%s\": need 5 or 6 numbers (x y rx ry points [skew])\n", optbuf);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*INTERVAL*", 10)== 0 ){
		  double x= 0;
		  int n= 1;
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*INTERVAL*";
			if( fascanf( &n, parse_codes(&optbuf[10]), &x, NULL, NULL, NULL, NULL)== 1 ){
				this_set->plot_interval= (int) fabs(x);
				if( debugFlag ){
					fprintf( StdErr, "ReadData(), %s: line %d.%d plot_interval= %d\n",
						filename,
						sub_div, line_count, this_set->plot_interval
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename,
					sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*ADORN_INT*", 11)== 0 ){
		  double x= 0;
		  int n= 1;
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*ADORN_INT*";
			if( fascanf( &n, parse_codes(&optbuf[11]), &x, NULL, NULL, NULL, NULL)== 1 ){
				this_set->adorn_interval= (int) fabs(x);
				if( debugFlag ){
					fprintf( StdErr, "ReadData(), %s: line %d.%d adorn_interval= %d\n",
						filename,
						sub_div, line_count, this_set->adorn_interval
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename,
					sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncasecmp( optbuf, "*Set*", 5)== 0 ){
		  double x[2]= {0,0};
		  int n= 2;
			fascanf( &n, parse_codes(&optbuf[5]), x, NULL, NULL, NULL, NULL);
			if( n>= 1 ){
				if( (this_set->set_nr- first_set!= (int) x[0]) ){
					line_count= UpdateLineCount( stream, &flpos, line_count );
					fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): specified set#%d != current set %d-%d=%d -"
						"file may have been changed\n",
						filename,
						sub_div, line_count, optbuf, (int)x[0], this_set->set_nr, first_set, this_set->set_nr- first_set
					);
					fflush( StdErr );
				}
				  /* 990107: for the moment, always set the flag indicating we just had a *Set* statement,
				   \ even if there has been a change in the numbering.
				   */
				set_set= True;
				spec_set= (int) x[0];
				real_set= this_set->set_nr;
				if( n> 1 ){
					total_sets= (int) x[1];
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename,
					sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
		}
		else if( strncmp( optbuf, "*N*", 3)== 0 ){
		  double x= 0;
		  int n= 1;
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*N*";
			if( fascanf( &n, parse_codes(&optbuf[3]), &x, NULL, NULL, NULL, NULL)== 1 ){
				this_set->NumObs= NumObs= (int) fabs(x);
				if( debugFlag ){
					fprintf( StdErr, "ReadData(), %s: line %d.%d NumObs= %d\n",
						filename,
						sub_div, line_count, this_set->NumObs
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename,
					sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*VECTORLENGTH*", 14)== 0 || strncmp( optbuf, "*VANELENGTH*", 12)== 0 ){
		  double x= 0;
		  int n= 1;
		  char *ls= &optbuf[ (optbuf[2]== 'E')? 14 : 12 ];
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*VECTORLENGTH*";
			if( fascanf( &n, parse_codes(ls), &x, NULL, NULL, NULL, NULL)== 1 ){
				this_set->vectorType= 0;
				this_set->vectorLength= vectorLength= x;
				if( debugFlag ){
					fprintf( StdErr, "ReadData(), %s: line %d.%d vectorLength= %g\n",
						filename,
						sub_div, line_count, this_set->vectorLength
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename,
					sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*VECTORPARS*", 12)== 0 || strncmp( optbuf, "*VANEPARS*", 10)== 0 ){
/* 		  double pars[2+MAX_VECPARS];	*/
/* 		  int n= 2+MAX_VECPARS;	*/
		  char *ls= &optbuf[ (optbuf[2]== 'E')? 12 : 10 ];
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*VECTORPARS*";
			if( Parse_vectorPars( parse_codes(ls), this_set, True, NULL, "*VECTORPARS*") ){
				if( debugFlag ){
					fprintf( StdErr, "ReadData(), %s: line %d.%d type=%d vectorLength= %g type1_pars=%g,%g\n",
						filename,
						sub_div, line_count, this_set->vectorType, this_set->vectorLength,
						this_set->vectorPars[0], this_set->vectorPars[1]
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need at least 2 numbers\n",
					filename,
					sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*PARAM_RANGE*", 13)== 0 ){
		  int n= 3;
		  char change[3];
		  char *TBh= TBARprogress_header;
			param_ok= 0;
			TBARprogress_header= "*PARAM_RANGE*";
			fascanf( &n, parse_codes(&optbuf[13]), ReadData_proc.param_range, change, data, column, NULL);
			TBARprogress_header= TBh;
			switch( n ){
				case 0:
				case 1:
					param_ok= 0;
					break;
				case 2:
					if( index( "NR", change[0]) && index( "NR", change[1]) ){
						param_ok= 1;
						ReadData_proc.param_range[2]= floor( fabs( ReadData_proc.param_range[1] - ReadData_proc.param_range[0] ) );
					}
					else{
						param_ok= 0;
					}
					break;
				case 3:
					param_ok= 1;
					ReadData_proc.param_range[2]= floor( fabs( ReadData_proc.param_range[2] ));
					break;
			}
			if( !param_ok ){
				fprintf( StdErr, "\"%s\": need at least min,max\n", optbuf);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PARAM_BEFORE*", 14)== 0 ){
		  char *membuf= &optbuf[14];
			xfree( ReadData_proc.param_before );
			if( strlen( cleanup(membuf) ) ){
				  /* cleanup() has removed the trailing '\n' that we
				   \ really want in this case!
				   */
				strcat( membuf, "\n");
				ReadData_proc.param_before= XGstrdup( membuf );
				ReadData_proc.param_before_len= strlen( ReadData_proc.param_before );
				ReadData_proc.param_before_printed= 0;
			}
			else{
				ReadData_proc.param_before= NULL;
				ReadData_proc.param_before_len= 0;
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PARAM_AFTER*", 13)== 0 ){
		  char *membuf= &optbuf[13];
			xfree( ReadData_proc.param_after );
			if( strlen( cleanup(membuf) ) ){
				strcat( membuf, "\n");
				ReadData_proc.param_after= XGstrdup( membuf );
				ReadData_proc.param_after_len= strlen( ReadData_proc.param_after );
				ReadData_proc.param_after_printed= 0;
			}
			else{
				ReadData_proc.param_after= NULL;
				ReadData_proc.param_after_len= 0;
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*EVAL*", 6)== 0 || strncmp( optbuf, "*PARAM_NOW*", 11)== 0
			|| strncmp(optbuf, "*EVAL_ALL*", 10)== 0
		){
		  char *membuf= &optbuf[ (optbuf[1]== 'E')? 6 : 11];
		  int AllWin= False;
			  /* 20040630: */
			if( membuf && strncmp( membuf, "ALL*", 4 )== 0 ){
				membuf+= 4;
				AllWin= True;
			}
			membuf= cleanup(membuf);
			if( strlen( membuf ) ){
			  char *pbuf;
			  char *TBh= TBARprogress_header;
				strcat( membuf, "\n");
				pbuf= SubstituteOpcodes( membuf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
					"*This-FileDir*", dirname(the_file),
					"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL);
				TBARprogress_header= "*EVAL*";
				if( AllWin ){
					new_param_now_allwin( pbuf, &ReadData_proc.param_range[0], -1 );
				}
				else{
					new_param_now( pbuf, &ReadData_proc.param_range[0], -1 );
				}
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, "*EVAL*", "\\n\n", "\n", membuf);
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				if( pbuf!= membuf ){
					xfree(pbuf);
				}
				TBARprogress_header= TBh;
			}
			ReadData_commands+= 1;
		}
#ifdef XG_DYMOD_SUPPORT
		else if( strncmp( optbuf, "*PYTHON*", 8)== 0
			|| strncmp( optbuf, "*PYTHON2*", 9)== 0
			|| strncmp( optbuf, "*PYTHON_FILE*", 13)== 0
			|| strncmp( optbuf, "*PYTHON_SHELL*", 14)== 0
		){
		  char *defbuf= NULL;
		  extern DM_Python_Interface *dm_python;
		  extern int Init_Python();
		  int ispy2k = False;
			if( optbuf[7] == '*' ){
				// PYTHON
				defbuf= cleanup(&optbuf[8]);
			}
			else if( optbuf[8] == '*' ){
				// PYTHON2
				defbuf= cleanup(&optbuf[9]);
				ispy2k = True;
			}
			else if( optbuf[12] == '*' ){
				// PYTHON_FILE
				defbuf = cleanup(&optbuf[13]);
			}
			else{
				// PYTHON_SHELL
				defbuf = cleanup(&optbuf[14]);
			}
			if( Init_Python()
				&& dm_python->Run_Python_Expr && dm_python->Import_Python_File && dm_python->Python_SysArgv
			){
				if( *defbuf || optbuf[13]=='*' ){
				  char *pbuf, *argv[3]= {the_file, the_file, NULL};
				  int argc= 2;
				  char *tmpName, pidnr[128], phash[128];
				  FILE *fp;
				  char *TBh= TBARprogress_header;
					TBARprogress_header= "*PYTHON*";
					pbuf= SubstituteOpcodes( defbuf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
						"*This-FileDir*", dirname(the_file),
						"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL);
					if( optbuf[7]== '*' || optbuf[8] == '*' ){
						strcat( pbuf, "\n");
						  /* This was an attempt to make the sourcecode in *PYTHON* statements available to interactive
						   \ interpreters. However, for this to work, one should apparently have the .py file available
						   \ that the code was read from... :(
						   */
						snprintf( pidnr, sizeof(pidnr)/sizeof(char)-1, "%d", getpid() );
						snprintf( phash, sizeof(phash)/sizeof(char)-1, "%lx", ascanf_hash(pbuf,NULL) );
						if( (tmpName= concat( "/tmp/", basename(the_file), ".", pidnr, "-", phash, ".py", NULL))
							&& (fp= fopen(tmpName, "w"))
						){
						  FileLinePos pflpos;
						  int psize, pline_count;
							line_count= UpdateLineCount( stream, &flpos, line_count );
							fprintf( fp, "## Temporary code exported from *PYTHON* at \"%s\":%d ::\n"
								"%s\n",
								the_file, line_count,
								pbuf
							);
							if( (debugFlag || scriptVerbose) ){
								if( !fseek( fp, 0, SEEK_END ) ){
									pflpos.last_fpos= 0;
									pflpos.last_line_count= 0;
									pline_count= 0;
									pline_count= UpdateLineCount( fp, &pflpos, pline_count );
								}
								psize= ftell(fp);
							}
							fclose(fp);
/* 							(*dm_python->Import_Python_File)( tmpName, the_file, 1, ispy2k );	*/
							(*dm_python->Import_Python_File)( tmpName, NULL, 1, ispy2k );
							if( (debugFlag || scriptVerbose) ){
								fprintf( StdErr, "Executed *PYTHON* \"%s\" at \"%s\":%d: %ld bytes for %d lines\n",
									tmpName, the_file, line_count,
									psize, pline_count
								);
							}
						}
						else
						{
							(*dm_python->Python_SysArgv)( argc, argv );
							(*dm_python->Run_Python_Expr)( pbuf );
							argv[0]= "";
							(*dm_python->Python_SysArgv)( 0, argv );
							if( (debugFlag || scriptVerbose) ){
								line_count= UpdateLineCount( stream, &flpos, line_count );
								fprintf( StdErr, "Executed *PYTHON* at \"%s\":%d\n",
									the_file, line_count
								);
							}
							fflush(StdErr);
							fflush(stdout);
						}
					}
					else if( optbuf[12] == '*' ){
						(*dm_python->Import_Python_File)( pbuf, NULL, 0, False );
						if( (debugFlag || scriptVerbose) ){
							line_count= UpdateLineCount( stream, &flpos, line_count );
							fprintf( StdErr, "Imported *PYTHON_FILE* \"%s\" at \"%s\":%d\n",
								pbuf, the_file, line_count
							);
						}
					}
					else{
					  double *arg, Arg;
					  int ret, res;
						if( pbuf && new_param_now( pbuf, &ReadData_proc.param_range[0], 1 ) ){
							Arg= ReadData_proc.param_range[0];
							arg = &Arg;
						}
						else{
							arg = NULL;
						}
						if( (debugFlag || scriptVerbose) ){
							line_count= UpdateLineCount( stream, &flpos, line_count );
							fprintf( StdErr, "Entering *PYTHON_SHELL* %s at \"%s\":%d\n",
								(arg)? ad2str(*arg, d3str_format, NULL) : "", the_file, line_count
							);
						}
						ret = (*dm_python->open_PythonShell)( arg, &res );
						if( (debugFlag || scriptVerbose) ){
							fprintf( StdErr, "Exit from *PYTHON_SHELL* with return values %d:%d\n",
								ret, res
							);
						}
					}
					if( pbuf!= defbuf ){
						xfree(pbuf);
					}
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, "*PYTHON*", "\\n\n", "\n", defbuf);
						fputc( '\n', stdout );
					}
					TBARprogress_header= TBh;
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s): line %d.%d can't load Python library 'Python' (%s) or not a proper Python library\n",
					the_file, sub_div, line_count, serror()
				);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
		}
#endif
#ifdef TWO_2ndGEN_READ_FILEBLOCKS
		else if( strncasecmp( optbuf, "*2ndGEN_READ_FILE*", 18)== 0 ){
		  char *command= &optbuf[8], *pbuf;
			pbuf= SubstituteOpcodes( command, "*This-File*", the_file, "*This-FileName*", basename(the_file),
				"*This-FileDir*", dirname(filename),
				"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
			if( ActiveWin && ActiveWin!= &StubWindow ){
				ActiveWin->next_include_file= concat2( ActiveWin->next_include_file, "*", pbuf, "\n", NULL);
				if( debugFlag ){
					fprintf( StdErr, "2nd generation include file(s): \"%s\"\n", ActiveWin->next_include_file );
				}
			}
			else{
				next_include_file= concat2( next_include_file, "*", pbuf, "\n", NULL);
				if( debugFlag ){
					fprintf( StdErr, "2nd generation include file(s): \"%s\"\n", next_include_file );
				}
			}
			if( pbuf != command ){
				xfree(pbuf);
			}
			add_comment( buffer );
		}
#endif
		else if( strncmp( optbuf, "*INIT_BEGIN*", 12)== 0 ){
		  char *membuf;
		  XGStringList **last;
		  Boolean append;
init_begin:;
			if( Allow_InitExprs ){
				membuf= cleanup( &optbuf[12] );
				append= strncmp( membuf, "new", 3 );
				if( ActiveWin && ActiveWin!= &StubWindow ){
					last= &ActiveWin->init_exprs;
					new_init_exprs= &ActiveWin->new_init_exprs;
				}
				else{
					last= &Init_Exprs;
					new_init_exprs= NULL;
				}
				init_exprs= last;
				if( *last ){
					if( append ){
						  /* find last element: */
						while( (*last)->next ){
							(*last)= (*last)->next;
						}
						  /* the next line will be stored in the last element's next field: */
						last= &( (*last)->next );
					}
					else{
						*last= XGStringList_Delete( *last );
					}
				}
				the_init_exprs= last;
				Allow_InitExprs= False;
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s,%d,%d) \"%s\"\t: INIT expressions not currently allowed!\n",
					filename, sub_div, line_count, buffer
				);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*INIT_END*", 10)== 0 ){
			if( init_exprs ){
			  XGStringList *list= *init_exprs;
			  int lines= 0, evln= 0;
			  FILE *tCF= ReadData_thisCurrentFP;
			  char *tCFN= ReadData_thisCurrentFileName;
				while( list ){
					list= list->next;
					lines+= 1;
					ReadData_commands+= 1;
				}
				the_init_exprs= NULL;
				if( new_init_exprs ){
					*new_init_exprs= True;
					new_init_exprs= NULL;
				}
				  /* If ReadData() is not yet configured to print the time_stamp of another file then it is
				   \ actually reading when encountering an *FINFO* command, configure it to print the name
				   \ of the file we're currently reading. That way, *FINFO* will show this file, and not
				   \ the stats of the temp. file that will be generated to read the *INIT expressions from.
				   \ (A user may typically place an *FINFO* command in the *INIT* section! NB: the 2nd time
				   \ that the commands are executed, he/she *will* get the stats of the temp file.)
				   */
				if( !ReadData_thisCurrentFileName ){
					ReadData_thisCurrentFileName= the_file;
				}
				if( !ReadData_thisCurrentFP ){
					ReadData_thisCurrentFP= stream;
				}
				evln= Evaluate_ExpressionList( ActiveWin, init_exprs, False, "INIT expressions, first time" );
				ReadData_thisCurrentFileName= tCFN;
				ReadData_thisCurrentFP= tCF;
				if( debugFlag || scriptVerbose || skip_to_label ){
					line_count= UpdateLineCount( stream, &flpos, line_count );
					fprintf( StdErr, "ReadData(%s,%d,%d) (%s): processed %d INIT expression lines, returned %d.\n",
						filename, sub_div, line_count, optbuf,
						lines, evln
					);
					if( skip_to_label ){
						fprintf( StdErr, "\tnow continuing suspended -skip_to/*SKIP_TO* %s command!\n",
							skip_to_label
						);
					}
				}
				init_exprs= NULL;
				Allow_InitExprs= True;
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s,%d,%d) \"%s\"\t: *INIT_END* command without preceding *INIT_BEGIN* in the same file!\n",
					filename, sub_div, line_count, buffer
				);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*STARTUP_EXPR*", 14)== 0 || strncmp( optbuf, "*2ndGEN_STARTUP_EXPR*", 21)== 0 ){
		  char *membuf;
		  XGStringList *new, *last= Startup_Exprs;
		  Boolean this_GEN= (strncmp( optbuf, "*2ndGEN_", 8)!= 0);
			  /* 20020619: do not alter the buffer in any way (no more cleanup()) ! */
			membuf= &optbuf[ (this_GEN)? 14 : 21 ];
			if( NextGen ){
				this_GEN= False;
			}
			if( *membuf ){
				if( (new= (XGStringList*) malloc( sizeof(XGStringList) )) ){
					new->text= XGstrdup(membuf);
					new->separator = ascanf_separator;
					new->last = new->next= NULL;
					if( this_GEN ){
						if( (last= Startup_Exprs) ){
							if( last->last && last->last->next == NULL ){
								last = last->last;
							}
							else{
								while( last->next ){
									last= last->next;
								}
							}
							if( last ){
								last->next= new;
							}
						}
						else{
							Startup_Exprs= new;
						}
						  // 20101105: maintain a pointer to the last element. This can make an incredible
						  // difference on performance when constructing a large list!
						Startup_Exprs->last = new;
					}
					else{
						if( ActiveWin && ActiveWin!= &StubWindow ){
							if( (last= ActiveWin->next_startup_exprs ) ){
								if( last->last && last->last->next == NULL ){
									last = last->last;
								}
								else{
									while( last->next ){
										last= last->next;
									}
								}
								if( last ){
									last->next= new;
								}
							}
							else{
								ActiveWin->next_startup_exprs = new;
							}
							ActiveWin->next_startup_exprs->last = new;
						}
						else{
							if( (last= nextGEN_Startup_Exprs) ){
								if( last->last && last->last->next == NULL ){
									last = last->last;
								}
								else{
									while( last->next ){
										last= last->next;
									}
								}
								if( last ){
									last->next= new;
								}
							}
							else{
								nextGEN_Startup_Exprs= new;
							}
							nextGEN_Startup_Exprs->last= new;
						}
					}
				}
				strcat( membuf, "\n");
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, (this_GEN)? "*STARTUP_EXPR*" : "*2ndGEN_STARTUP_EXPR*", "\\n\n", "\n", membuf);
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
			}
			else{
				if( this_GEN ){
					if( Startup_Exprs ){
						Startup_Exprs= XGStringList_Delete( Startup_Exprs );
						if( debugFlag ){
							fprintf( StdErr, "(Global) STARTUP_EXPRs deleted\n" );
						}
					}
				}
				else{
					if( ActiveWin && ActiveWin!= &StubWindow ){
						if( ActiveWin->next_startup_exprs ){
							ActiveWin->next_startup_exprs= XGStringList_Delete( ActiveWin->next_startup_exprs );
							if( debugFlag ){
								fprintf( StdErr, "Window %02d:%02d:%02d 2ndGEN_STARTUP_EXPRs deleted\n",
									ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
								);
							}
						}
					}
					else{
						if( nextGEN_Startup_Exprs ){
							nextGEN_Startup_Exprs= XGStringList_Delete( nextGEN_Startup_Exprs );
							if( debugFlag ){
								fprintf( StdErr, "2ndGEN_STARTUP_EXPRs deleted\n" );
							}
						}
					}
				}
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*EXIT_EXPR*", 11)== 0 ){
		  char *membuf;
		  XGStringList *new, *last= Exit_Exprs;
			membuf= &optbuf[ 11 ];
			if( strlen( membuf ) ){
				if( (new= (XGStringList*) calloc( 1, sizeof(XGStringList) )) ){
					new->text= XGstrdup(membuf);
					new->next= NULL;
					if( (last= Exit_Exprs) ){
						while( last->next ){
							last= last->next;
						}
						if( last ){
							last->next= new;
						}
					}
					else{
						Exit_Exprs= new;
					}
				}
				strcat( membuf, "\n");
				if( DumpFile ){
					if( DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						DF_bin_active= False;
					}
					if( !DumpPretty ){
						fputs( "*EXTRATEXT* ", stdout );
					}
					else{
						LineDumped= True;
					}
					print_string( stdout, "*EXIT_EXPR*", "\\n\n", "\n", membuf);
					fputc( '\n', stdout );
				}
			}
			else{
				if( Exit_Exprs ){
					Exit_Exprs= XGStringList_Delete( Exit_Exprs );
					if( debugFlag ){
						fprintf( StdErr, "(Global) EXIT_EXPRs deleted\n" );
					}
				}
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*DUMP_COMMANDS*", 15)== 0 ){
		  char *membuf= cleanup( &optbuf[15] );
			if( strlen( membuf ) ){
				if( ActiveWin && ActiveWin!= &StubWindow ){
					ActiveWin->Dump_commands= XGStringList_AddItem( ActiveWin->Dump_commands, membuf );
				}
				else{
					Dump_commands= XGStringList_AddItem( Dump_commands, membuf );
				}
				strcat( membuf, "\n");
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, "*DUMP_COMMANDS*", "\\n\n", "\n", membuf);
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
			}
			else{
				if( ActiveWin && ActiveWin!= &StubWindow ){
					if( ActiveWin->Dump_commands ){
						ActiveWin->Dump_commands= XGStringList_Delete( ActiveWin->Dump_commands );
						if( debugFlag ){
							fprintf( StdErr, "Window %02d:%02d:%02d DUMP_COMMANDS deleted\n",
								ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
							);
						}
					}
				}
				else{
					if( Dump_commands ){
						Dump_commands= XGStringList_Delete( Dump_commands );
						if( debugFlag ){
							fprintf( StdErr, "DUMP_COMMANDS deleted\n" );
						}
					}
				}
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*DUMPPROCESSED_SPECIFIC*", 24)== 0 ){
		  char *membuf= cleanup( &optbuf[24] );
			if( strlen( membuf ) ){
			  char *pbuf= SubstituteOpcodes( membuf, "*This-File*", the_file, "*This-FileName*", basename(the_file),
			  	"*This-FileDir*", dirname(the_file),
			  	"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
				if( ActiveWin && ActiveWin!= &StubWindow ){
					ActiveWin->DumpProcessed_commands= XGStringList_AddItem( ActiveWin->DumpProcessed_commands, pbuf );
				}
				else{
					DumpProcessed_commands= XGStringList_AddItem( DumpProcessed_commands, pbuf );
				}
				strcat( membuf, "\n");
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, "*DUMPPROCESSED_SPECIFIC*", "\\n\n", "\n", membuf);
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
				if( pbuf!= membuf ){
					xfree( pbuf );
				}
			}
			else{
				if( ActiveWin && ActiveWin!= &StubWindow ){
					if( ActiveWin->DumpProcessed_commands ){
						ActiveWin->DumpProcessed_commands= XGStringList_Delete( ActiveWin->DumpProcessed_commands );
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "Window %02d:%02d:%02d DUMPPROCESSED_SPECIFICs deleted\n",
								ActiveWin->parent_number, ActiveWin->pwindow_number, ActiveWin->window_number
							);
						}
					}
				}
				else{
					if( DumpProcessed_commands ){
						DumpProcessed_commands= XGStringList_Delete( DumpProcessed_commands );
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "DUMPPROCESSED_SPECIFICs deleted\n" );
						}
					}
				}
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PARAM_FUNCTIONS*", 17)== 0 ){
		  double t;
		  int ok= 0, n;
		  char change[ASCANF_DATA_COLUMNS], label[16];
		  int show_progress= Show_Progress && isatty(fileno(StdErr));
		  char *progress[]= { "\b|", "\b/", "\b-", "\b\\", "\b|", "\b/", "\b-", "\b\\", "\b*"};
		  int perc= 0;
		  extern int reset_ascanf_currentself_value, ascanf_arg_error;
		  Time_Struct prtimer1;
			if( param_ok && optbuf ){
			  char *TBh= TBARprogress_header;

				memset( change, VAR_UNCHANGED, sizeof(change) );
				TBARprogress_header= "*PARAM_FUNCTIONS*";
				ReadData_proc.param_functions= cleanup(&optbuf[17]);
				strncpy( label, ReadData_proc.param_functions, 15 );
				label[15]= '\0';
				strcat( ReadData_proc.param_functions, "\n");
				ReadData_proc.param_functions_len= strlen( ReadData_proc.param_functions);

#ifdef TESTY
				  /* test	*/
				if( ascanf_verbose ){
					fprintf( StdErr, "test x y e: %s", ReadData_proc.param_functions );
					fflush( StdErr );
				}
				n= ASCANF_DATA_COLUMNS;
				data[0]= data[1]= data[2]= data[3]= ReadData_proc.param_range[0];
				fascanf( &n, ReadData_proc.param_functions, data, change, data, column, NULL );

				  /* We require that at least the X and Y
				   \ values are determined (calculated or re-assigned)
				   */
				if( !ascanf_arg_error && (n>=2 && n<=ASCANF_DATA_COLUMNS) &&
					(change[0]== 'N' || change[0]== 'R') &&
					(change[1]== 'N' || change[1]== 'R')
				)
#else
				   /* There is actually no reason to require changes to at least X or Y.
				    \ Rather, it should be possible to program systems that will remain
					\ stationnary for some time. In any case, a FULL evaluation should be
					\ made, including the param_before and param_after statements.
					*/
#endif
				{
				  int rsacsv= reset_ascanf_currentself_value;
					  /* if this set doesn't have a name yet,
					   \ baptise it
					   */
					{
/*
#ifdef __GNUC__
					  char legend[ReadData_proc.param_before_len+ReadData_proc.param_after_len+ReadData_proc.param_functions_len+128];
#else
					  char legend[3*(LMAXBUFSIZE+1)+128];
#endif
 */
					  _ALLOCA( legend, char,
						(ReadData_proc.param_before_len+ReadData_proc.param_after_len+ReadData_proc.param_functions_len+128+
						 ((legend_setNumber== setNumber && this_set->setName)? strlen(this_set->setName) : 0) ),
						legend_len
					  );
						if( legend_setNumber!= setNumber ){
							sprintf( legend, "%s%sx y e=%s%s%sRange=%s..%s : %s",
								(ReadData_proc.param_before_len)? "Before=" : "",
								(ReadData_proc.param_before_len)? ReadData_proc.param_before : "",
								ReadData_proc.param_functions,
								(ReadData_proc.param_after_len)? "After=" : "",
								(ReadData_proc.param_after_len)? ReadData_proc.param_after : "",
								d2str( ReadData_proc.param_range[0], 0,0),
								d2str( ReadData_proc.param_range[1], 0,0),
								d2str( ReadData_proc.param_range[2], 0,0)
							);
					  /* subsitute all ',' for newlines to put the different
					   \ expressions on different lines. Only necessary if
					   \ fascanf() doesn't accept formatted strings (with embedded newlines)
							c= legend;
							while( (c= ascanf_index( c, ascanf_separator ) ) ){
								*c++= '\n';
							}
					   */
							xfree( this_set->setName );
							this_set->setName= XGstrdup( legend );
							this_set->draw_set= 1;
							this_set->show_legend= 1;
							this_set->show_llines= 1;
							legend_setNumber= setNumber;
						}
						else{
						  int n;
							n = snprintf( legend, legend_len, "\n%s:\n", this_set->setName);
							n += snprintf( &legend[n], legend_len - n, " *PARAM_RANGE*%s,%s,%s\n",
								d2str( ReadData_proc.param_range[0], "%g", NULL),
								d2str( ReadData_proc.param_range[1], "%g", NULL),
								d2str( ReadData_proc.param_range[2], "%g", NULL)
							);
							if( !ReadData_proc.param_before_printed ){
								n += snprintf( &legend[n], legend_len - n, "%s%s",
									(ReadData_proc.param_before_len)? " *PARAM_BEFORE*" : "",
									(ReadData_proc.param_before_len)? ReadData_proc.param_before : ""
								);
								ReadData_proc.param_before_printed= 1;
							}
							if( !ReadData_proc.param_after_printed ){
								n += snprintf( &legend[n], legend_len - n, "%s%s",
									(ReadData_proc.param_after_len)? " *PARAM_AFTER*" : "",
									(ReadData_proc.param_after_len)? ReadData_proc.param_after : ""
								);
								ReadData_proc.param_after_printed= 1;
							}
							n += snprintf( &legend[n], legend_len - n,
								" *PARAM_FUNCTIONS*%s", legend, ReadData_proc.param_functions );
							StringCheck( legend, legend_len, __FILE__, __LINE__);
					  /* subsitute all ',' for newlines to put the different
					   \ expressions on different lines
							c= legend;
							while( (c= ascanf_index( c, ascanf_separator ) ) ){
								*c++= '\n';
							}
					   */
							Add_Comment( legend, True );
						}
					}
					param_dt= (ReadData_proc.param_range[1] - ReadData_proc.param_range[0])/ ReadData_proc.param_range[2];
					if( !ascanf_verbose && show_progress ){
						fputs( "*", StdErr );
						fflush( StdErr );
					}
					/* Compile the expressions	*/{
					  int av= (ascanf_verbose)? 1 : 0;
						if( debugFlag ){
							ascanf_verbose= 1;
						}
						n= param_scratch_len;
						*ascanf_self_value= *ascanf_current_value= ReadData_proc.param_range[0];
						reset_ascanf_currentself_value= 0;
						reset_ascanf_index_value= True;
						*ascanf_counter= 0;
						(*ascanf_Counter)= 0;
						fascanf( &n, ReadData_proc.param_before, param_scratch, NULL, data, column, &ReadData_proc.C_param_before );
						if( debugFlag ){
							fprintf( StdErr, "Compiled PARAM_BEFORE: ");
							Print_Form( StdErr, &ReadData_proc.C_param_before, 0, True, NULL, NULL, "\n", True );
						}
						if( DumpFile ){
							if( DF_bin_active ){
								BinaryTerminate(stdout);
								fputc( '\n', stdout);
								DF_bin_active= False;
							}
							if( !DumpPretty ){
								fputs( "*EXTRATEXT* ", stdout );
							}
							else{
								LineDumped= True;
							}
							fprintf( stdout, "*PARAM_BEFORE*");
							Print_Form( stdout, &ReadData_proc.C_param_before, 0, True, NULL, "\t", NULL, True );
/* 							if( !DumpPretty ){	*/
								fputc( '\n', stdout );
/* 							}	*/
						}
						n= ASCANF_DATA_COLUMNS;
						*ascanf_self_value= *ascanf_current_value= ReadData_proc.param_range[0];
						reset_ascanf_currentself_value= 1;
						reset_ascanf_index_value= True;
						fascanf( &n, ReadData_proc.param_functions, data, change, data, column, &ReadData_proc.C_param_functions );
						if( debugFlag ){
							fprintf( StdErr, "Compiled PARAM_FUNCTIONS: ");
							Print_Form( StdErr, &ReadData_proc.C_param_functions, 0, True, NULL, NULL, "\n", True );
						}
						if( DumpFile ){
							if( !DumpPretty ){
								fputs( "*EXTRATEXT* ", stdout );
							}
							else{
								LineDumped= True;
							}
							fprintf( stdout, "*PARAM_FUNCTIONS*");
							Print_Form( stdout, &ReadData_proc.C_param_functions, 0, True, NULL, "\t", NULL, True );
/* 							if( !DumpPretty ){	*/
								fputc( '\n', stdout );
/* 							}	*/
						}
						n= param_scratch_len;
						*ascanf_self_value= *ascanf_current_value= ReadData_proc.param_range[0];
						reset_ascanf_currentself_value= 0;
						reset_ascanf_index_value= True;
						fascanf( &n, ReadData_proc.param_after, param_scratch, NULL, data, column, &ReadData_proc.C_param_after );
						ascanf_verbose= av;
						if( debugFlag ){
							fprintf( StdErr, "Compiled PARAM_AFTER: ");
							Print_Form( StdErr, &ReadData_proc.C_param_after, 0, True, NULL, NULL, "\n", True );
						}
						if( DumpFile ){
							if( !DumpPretty ){
								fputs( "*EXTRATEXT* ", stdout );
							}
							else{
								LineDumped= True;
							}
							fprintf( stdout, "*PARAM_AFTER*");
							Print_Form( stdout, &ReadData_proc.C_param_after, 0, True, NULL, "\t", NULL, True );
/* 							if( !DumpPretty ){	*/
								fputc( '\n', stdout );
/* 							}	*/
						}
					}
					ok= !ascanf_arg_error;

					  /* if this is a virgin set, pre-allocate enough points for reasons
					   \ of memory efficiency.
					   */
					if( this_set->allocSize< ReadData_proc.param_range[2]+2 ){
						this_set->numPoints+= (int) ReadData_proc.param_range[2]+ 2;
						realloc_points( this_set, this_set->numPoints+ (int) ReadData_proc.param_range[2]+ 2, False );
					}
					if( !this_set->numPoints ){
						this_set->ncols= 3;
						this_set->xcol= 0;
						this_set->ycol= 1;
						this_set->ecol= 2;
						this_set->lcol= -1;
						this_set->Ncol= -1;
					}
					set_Columns( this_set );

					Elapsed_Since( &prtimer1, True );
/* 					Elapsed_Since( &prtimer2, True );	*/
					for( *ascanf_counter=0, (*ascanf_Counter)= 0, t= ReadData_proc.param_range[0];
							ok && t<= ReadData_proc.param_range[1] && !ascanf_exit;
							(*ascanf_counter)++, (*ascanf_Counter)++, t+= param_dt
					){
						AddPoint_discard= False;
						if( ReadData_proc.param_range[1] ){
						  double p= ( t/ (ReadData_proc.param_range[1]) * 100);
							if( p>= perc ){
								Elapsed_Since( &prtimer1, False );
								if( prtimer1.Tot_Time>= Progress_ThresholdTime || !*ascanf_counter ){
									if( theSettingsWin_Info ){
									  char buf[256];
										sprintf( buf, "*PF* %s: %s%% (%g of %g): %s line %d.%d",
											label, d2str(p, "%g", NULL), t, ReadData_proc.param_range[1], filename, sub_div+ file_splits, line_count
										);
										XStoreName( disp, theSettingsWin_Info->SD_Dialog->win, buf );
										xtb_XSync( disp, False );
									}
									else if( xterm ){
										fprintf( stderr, "%c]0;*PF* %s: %s%% (%g of %g): %s line %d.%d%c", 0x1b,
											label, d2str(p, "%g", NULL), t, ReadData_proc.param_range[1], filename,
											sub_div+ file_splits, line_count,
											0x07
										);
										fflush( stderr );
									}
									else if( !show_progress ){
										if( !ReadData_terpri ){
											fprintf( StdErr, "*PF* %s: %d%%", label, perc );
										}
										else{
											fprintf( StdErr, " %d%%", perc );
										}
										ReadData_terpri= True;
										fflush( StdErr );
									}
									Elapsed_Since( &prtimer1, True );
								}
								perc+= 10;
							}
						}
						if( ReadData_proc.param_before_len ){
/* in PARAM_BEFORE, the 'self' function evaluates to 't' (as in PARAM_FUNCTIONS).
 \ Since the memory slots should be able to accumulate data, ascanf_self_value and
 \ ascanf_current_value are set here directly, instead of through set_ascanf_memory(),
 \ which effectively resets all memory slots.
							n= set_ascanf_memory( t );
 */
							n= param_scratch_len;
							*ascanf_self_value= t;
							*ascanf_current_value= t;
							reset_ascanf_currentself_value= 0;
							reset_ascanf_index_value= True;
							if( ascanf_verbose ){
								fprintf( StdErr, "Before: %s", ReadData_proc.param_before);
								fflush( StdErr );
							}
							compiled_fascanf( &n, ReadData_proc.param_before, param_scratch, NULL, data, column, &ReadData_proc.C_param_before );
							if( ascanf_arg_error || !n ){
								ok= 0;
							}
						}
						if( ok ){
							n= ASCANF_DATA_COLUMNS;
							data[0]= data[1]= data[2]= data[3]= t;
							reset_ascanf_currentself_value= 1;
							reset_ascanf_index_value= True;
/* 							Elapsed_Since( &prtimer2, False );	*/
							if( ascanf_verbose ){
								fprintf( StdErr, "x y e: %s", ReadData_proc.param_functions );
								fflush( StdErr );
							}
							else if( show_progress /* && (prtimer2.Tot_Time>= Progress_ThresholdTime || *ascanf_counter==0) */ ){
							  static int i= 0;
								fputs( progress[i], StdErr );
								i= (i+1) % 9;
/* 								Elapsed_Since( &prtimer2, True );	*/
							}
							compiled_fascanf( &n, ReadData_proc.param_functions, data, change, data, column, &ReadData_proc.C_param_functions );
							if( !ascanf_arg_error && (n==2 || n==3) &&
								(change[0]== 'N' || change[0]== 'R') &&
								(change[1]== 'N' || change[1]== 'R')
							){
								/* AddPoint() call used to be here	*/
							}
							else{
								ok= 0;
							}
						}
						if( ok && ReadData_proc.param_after_len ){
							n= param_scratch_len;
							*ascanf_self_value= t;
							*ascanf_current_value= t;
							reset_ascanf_currentself_value= 0;
							reset_ascanf_index_value= True;
							if( ascanf_verbose ){
								fprintf( StdErr, "After: %s", ReadData_proc.param_after );
								fflush( StdErr );
							}
							compiled_fascanf( &n, ReadData_proc.param_after, param_scratch, NULL, data, column, &ReadData_proc.C_param_after );
							if( ascanf_arg_error || !n ){
								ok= 0;
							}
						}
						if( ok && AddPoint_discard<= 0 ){
							AddPoint( &this_set, &spot, &Spot, 3, data, column, filename, sub_div, line_count, &flpos, buffer, &ReadData_proc );
						}
					}
					reset_ascanf_currentself_value= rsacsv;
					Destroy_Form( &ReadData_proc.C_param_before);
					Destroy_Form( &ReadData_proc.C_param_after);
					Destroy_Form( &ReadData_proc.C_param_functions);
				}
#ifdef TESTY
				else{
					fprintf( StdErr, "procedure error or too little (%d) values given\n", n);
				}
#endif
				TBARprogress_header= TBh;
			}
			fflush( StdErr );
			if( !ok ){
				fprintf( StdErr, "\"%s\": error\n", optbuf);
				fflush( StdErr );
			}
			ReadData_proc.param_functions_len= 0;
			ReadData_commands+= 1;
		}
		else if( Check_Data_Process_etc( this_set, optbuf, data, &DF_bin_active, &LineDumped, the_file, ReadData_level_str ) ){
		  /* handled	*/
		}
		else if( strncmp( optbuf, "*TRANSFORM_X*", 13)== 0 ){
		  char *membuf= &optbuf[13];
			if( membuf[0] ){
				strcpalloc( &transform_x_buf, &transform_x_buf_allen, membuf );
				transform_x_buf_len= strlen( transform_x_buf );
				transform_separator = ascanf_separator;
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, "*TRANSFORM_X*", "\\n\n", "\n", membuf);
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
			}
			else if( ActiveWin && ActiveWin!= &StubWindow && ActiveWin->transform.x_process[0] ){
				ActiveWin->transform.x_process[0]= '\0';
				new_transform_x_process( ActiveWin );
			}
			_process_bounds= 0;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*TRANSFORM_Y*", 13)== 0 ){
		  char *membuf= &optbuf[13];
			if( membuf[0] ){
				strcpalloc( &transform_y_buf, &transform_y_buf_allen, membuf );
				transform_y_buf_len= strlen( transform_y_buf );
				transform_separator = ascanf_separator;
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, "*TRANSFORM_Y*", "\\n\n", "\n", membuf);
/* 						if( !DumpPretty ){	*/
							fputc( '\n', stdout );
/* 						}	*/
					}
			}
			else if( ActiveWin && ActiveWin!= &StubWindow && ActiveWin->transform.y_process[0] ){
				ActiveWin->transform.y_process[0]= '\0';
				new_transform_y_process( ActiveWin );
			}
			_process_bounds= 0;
			ReadData_commands+= 1;
		}
		else if( strncasecmp( optbuf, "*TRANSFORM_DESCRIPTION*", 23)== 0 ){
		  char *membuf= cleanup( &optbuf[23] );
		  extern char *transform_description;
			xfree( transform_description );
			if( membuf && membuf[0] ){
				transform_description= XGstrdup( membuf );
				if( debugFlag ){
					fputs( membuf, StdErr );
				}
			}
			else if( ActiveWin && ActiveWin!= &StubWindow ){
				xfree( ActiveWin->transform.description );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*ASS_NAME*", 10)== 0 ){
		  char *argbuf= cleanup( &optbuf[10] );
		  Boolean ok= False;
			if( argbuf[0] ){
			  char *defbuf= strstr( argbuf, "::");
			  int N;
			  extern char **ascanf_SS_names;
				if( defbuf ){
				  int len;
				  char *db= defbuf;
					defbuf[0]= '\0';
					defbuf+= 2;
					len= strlen(defbuf);
					if( defbuf[len-1]== '\n' ){
						defbuf[len-1]= '\0';
					}
					if( (N= atoi(argbuf))>= 0 ){
						if( N+1>= ASCANF_MAX_ARGS || !ascanf_SS_names ){
							Ascanf_AllocMem( MAX(N+1, ASCANF_MAX_ARGS) );
							fprintf( StdErr,
								"ReadData(%s:%s,%d.%d;*ASS_NAME*%d::%s): expanded (or tried to) ASCANF_MAX_ARGS to %d\n",
								the_file, filename, sub_div, line_count,
								N, defbuf,
								ASCANF_MAX_ARGS
							);
							fflush( StdErr );
						}
						if( ascanf_SS_names ){
							xfree( ascanf_SS_names[N] );
							ascanf_SS_names[N]= XGstrdup(defbuf);
							if( debugFlag ){
								fprintf( StdErr, "Ascanf SS[%d] name: \"%s\"\n", N, defbuf );
							}
							ok= True;
						}
					}
					*db= ':';
				}
			}
			if( debugFlag && !ok ){
				fprintf( StdErr, "ReadData(%s:%s,%d.%d): *ASS_NAME* index::string (%s)\n",
					the_file, filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
		}
		else if( strncmp( optbuf, "*ASAS_NAME*", 11)== 0 ){
		  char *argbuf= cleanup( &optbuf[11] );
		  Boolean ok= False;
			if( argbuf[0] ){
			  char *defbuf= strstr( argbuf, "::");
			  int N;
			  extern char **ascanf_SAS_names;
				if( defbuf ){
				  int len;
				  char *db= defbuf;
					defbuf[0]= '\0';
					defbuf+= 2;
					len= strlen(defbuf);
					if( defbuf[len-1]== '\n' ){
						defbuf[len-1]= '\0';
					}
					if( (N= atoi(argbuf))>= 0 ){
						if( N+1>= ASCANF_MAX_ARGS || !ascanf_SAS_names ){
							Ascanf_AllocMem( MAX(N+1, ASCANF_MAX_ARGS) );
							fprintf( StdErr,
								"ReadData(%s:%s,%d.%d;*ASAS_NAME*%d::%s): expanded (or tried to) ASCANF_MAX_ARGS to %d\n",
								the_file, filename, sub_div, line_count,
								N, defbuf,
								ASCANF_MAX_ARGS
							);
							fflush( StdErr );
						}
						if( ascanf_SAS_names ){
							xfree( ascanf_SAS_names[N] );
							ascanf_SAS_names[N]= XGstrdup(defbuf);
							if( debugFlag ){
								fprintf( StdErr, "Ascanf SAS[%d] name: \"%s\"\n", N, defbuf );
							}
							ok= True;
						}
					}
					*db= ':';
				}
			}
			if( debugFlag && !ok ){
				fprintf( StdErr, "ReadData(%s:%s,%d.%d): *ASAS_NAME* index::string (%s)\n",
					the_file, filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
		}
		else if( strncasecmp( optbuf, "*2ndGEN_READ_FILE*", 18)== 0 ){
		  char *command= &optbuf[8], *pbuf, *fname;
			pbuf= SubstituteOpcodes( command, "*This-File*", the_file, "*This-FileName*", basename(the_file),
				"*This-FileDir*", dirname(filename),
				"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL );
			fname = &pbuf[10];
			next_gen_include_file( pbuf, fname );
			if(  pbuf != command ){
				xfree(pbuf);
			}
			Add_Comment( buffer, True );
		}
		else if( strncasecmp( optbuf, "*READ_FILE*", 11)== 0 || strncasecmp( optbuf, "*ASK_READ_FILE*", 15)== 0){
		  char *fname= (optbuf[1]== 'A' || optbuf[1]== 'a')? NULL : &optbuf[11], *rf, *fname2, *FName2=NULL;
		  char sFname[MAXPATHLEN];
		  ALLOCA( name, char, LMAXBUFSIZE+2, name_len);
		  int l_is_pipe= 0;
		  FILE *rfp;
		  Boolean ok= True;
			if( !fname ){
			  char *argbuf= cleanup( &optbuf[15] );
			  Boolean fok= False;
				if( argbuf[0] ){
				  char *defbuf= strstr( argbuf, "::"), *wid;
						if( DumpFile ){
							if( DF_bin_active ){
								BinaryTerminate(stdout);
								fputc( '\n', stdout);
								DF_bin_active= False;
							}
							if( !DumpPretty ){
								fputs( "*EXTRATEXT* ", stdout );
							}
							else{
								LineDumped= True;
							}
							print_string( stdout, optbuf, "\\n\n", "\n", argbuf);
/* 							if( !DumpPretty ){	*/
								fputc( '\n', stdout );
/* 							}	*/
						}
					if( defbuf && defbuf[2] ){
						  /* Cut the specified string here:	*/
						*defbuf= '\0';
						  /* retrieve the part after the double colons	*/
						{ char *c= ascanf_string( &defbuf[2], NULL);
							if( c!= &defbuf[2] ){
								defbuf[0]= '\n';
								defbuf[1]= '>';
							}
							defbuf= XGstrdup( c );
						}
					}
					else{
						defbuf= NULL;
					}
					wid= cgetenv("WINDOWID");
					do{
					  Boolean escaped= False;
						if( disp ){
						  int tilen= -1;
						  Window win;
						  char *nbuf;
							if( defbuf ){
								strncpy( name, defbuf, LMAXBUFSIZE);
								tilen= 1.5* strlen(name);
							}
							else{
								name[0]= '\0';
							}
							name[LMAXBUFSIZE]= '\0';
							if( ActiveWin ){
								win= ActiveWin->window;
							}
							else if( InitWindow ){
								win= InitWindow->window;
							}
							else if( wid ){
								win= atoi(wid);
							}
							else{
								win= DefaultRootWindow(disp);
							}
							TitleMessage( ActiveWin, argbuf );
							if( (nbuf= xtb_input_dialog( win, name, tilen, LMAXBUFSIZE, argbuf,
									"Enter filename", False,
									"History", process_hist_h, "Edit", SimpleEdit_h, " ... ", SimpleFileDialog_h
								))
							){
								  /* Make a local copy into fname. This prevents that changes
								   \ to name will be visible via fname even when the dialog
								   \ was cancelled after a first filename spec was rejected.
								   */
								xfree( fname );
								fname= XGstrdup( cleanup(name) );
								if( nbuf!= name ){
									xfree( nbuf );
								}
							}
							else{
								escaped= True;
							}
							TitleMessage( ActiveWin, NULL );
						}
						else{
							fflush( stdout );
							fflush( StdErr );
							xgiReadString2( name, LMAXBUFSIZE, argbuf, defbuf );
							cleanup( name );
							if( (!name[0] || name[0]== '\n') && defbuf ){
								strncpy( name, defbuf, LMAXBUFSIZE);
								name[LMAXBUFSIZE]= '\0';
							}
							  /* We don't have the problem of cancelled dialogs in this case, but to be
							   \ consistent, we still make a local copy.
							   */
							xfree( fname );
							fname= XGstrdup(name);
						}
						if( !escaped ){
						  FILE *fp;
							if( fname ){
								if( (fp= fopen(fname, "r")) ){
									fok= True;
									fclose(fp);
								}
								else{
									fprintf( StdErr, "\"%s\": '%s'\n", fname, serror() );
								}
							}
						}
						else{
							fok= True;
							if( fname ){
								fprintf( StdErr,
									"Warning: dialog cancelled, selected previously entered filename"
									" \"%s\" without further verification!\n",
									fname
								);
							}
						}
					} while( !fok );
					xfree( defbuf );
					if( fname ){
						  /* Store the local copy back into the (more) global buffer, and
						   \ free the local copy.
						   */
						strncpy( name, fname, LMAXBUFSIZE );
						xfree( fname );
						fname= name;
					}
					GCA();
				}
				rf= &optbuf[5];
			}
			else{
				rf= &optbuf[1];
			}
			if( fname ){
				// start code moved 20101025
				while( *fname && isspace((unsigned char)*fname) ){
					fname++;
				}
				  /* 20040119: also remove trailing whitespace. No reason not to...! */
				{ char *c; int l= strlen(fname);
					if( l ){
						c= &fname[ l-1 ];
						if( isspace(*c) ){
							while( isspace(*c) && c> fname ){
								c--;
							}
							c[1]= '\0';
						}
					}
				}
				if( fname[ strlen(fname)-1 ]== '\n' ){
					fname[ strlen(fname)-1 ]= '\0';
				}
				fname2= SubstituteOpcodes( fname, "*This-File*", the_file, "*This-FileName*", basename(the_file),
					"*This-FileDir*", dirname(filename),
					"*Print-File*", &PrintFileName, "*Read-Level*", ReadData_level_str, NULL
				);
				if( fname2 == fname ){
				  char *fn= fname;
					if( (fname= ascanf_string(fname, NULL))!= fn ){
						if( debugFlag || scriptVerbose ){
							fprintf( StdErr, "%s: %s refers to ascanf string %s\n",
								optbuf, fn, fname
							);
						}
						strncpy( name, fname, LMAXBUFSIZE );
						fname2 = fname= name;
					}
				}
				else{
					FName2 = fname2;
				}
				// end code moved 20101025
				if( NextGen ){
					next_gen_include_file( optbuf, fname2 );
				}
				else{
					// code moved 20101025
					if( No_IncludeFiles ){
						fprintf( StdErr, "ReadData(%s): \"%s\" ignored because of -NoIncludes\n",
							the_file, optbuf
						);
						fflush( StdErr );
						ok= False;
					}
					else if( strncmp( rf, "Read_File*", 10 )== 0 ){
						if( Read_File_Hist && strstr( Read_File_Hist, fname2 ) ){
							fprintf( StdErr, "ReadData(%s): *Read_File* skips %s that has already been included into this file\n",
								the_file, fname2
							);
							fflush( StdErr );
							ok= False;
						}
					}
					else if( strncmp( rf, "read_file*", 10 )== 0 ){
						if( read_file_hist && strstr( read_file_hist, fname2 ) ){
							fprintf( StdErr, "ReadData(%s): *read_file* skips %s that has already been included\n",
								the_file, fname2
							);
							fflush( StdErr );
							ok= False;
						}
					}
					if( ok ){

						if( Spot> 0 ){
							if( this_set->read_file ){
								this_set->read_file= concat2( this_set->read_file, "\n\n", optbuf, NULL);
							}
							else{
								this_set->read_file= XGstrdup( optbuf );
							}
							this_set->read_file_point= this_set->numPoints;
						}
						else{
							if( read_file_buf ){
								read_file_buf= concat2( read_file_buf, "\n\n", optbuf, NULL);
							}
							else{
								read_file_buf= XGstrdup( optbuf );
							}
							read_file_point= 0;
						}

						if( fname2[0]== '|' ){
							fname2++;
							system( "sh < /dev/null > /dev/null 2>&1");
							PIPE_error= False;
							if( (rfp= popen( fname2, "r" )) ){
								add_process_hist( &fname2[-1] );
							}
							strncpy( sFname, fname2, MAXPATHLEN-1 );
							l_is_pipe= 1;
						}
						else{
							tildeExpand( sFname, fname2 );
							if( (rfp= fopen( sFname, "r" )) ){
								add_process_hist( fname2 );
								IdentifyStreamFormat( sFname, &rfp, &l_is_pipe );
							}
						}
						if( rfp ){
						  int n;
						  int DF= DumpFile, DB= DumpBinary, DI= DumpIncluded, DH= DumpDHex;
							if( !DumpIncluded ){
								DumpFile= False;
							}
							else if( DumpFile ){
								if( DF_bin_active ){
									BinaryTerminate(stdout);
									fputc( '\n', stdout);
									DF_bin_active= False;
								}
								fputc( '#', stdout );
								fputs( buffer, stdout );
								if( buffer[strlen(buffer)-1]!= '\n' ){
									fputc( '\n', stdout );
								}
								LineDumped= True;
							}
							add_comment( buffer );
	/* 						n= ReadData( rfp, sFname, ++filenr );	*/
							  /* 20061115: why increment filenr permanently?? */
							n= ReadData( rfp, sFname, filenr+1 );
							if( n> 0 ){
								maxitems= n;
							}
							ascanf_exit= 0;
							DumpFile= DF;
							if( DF ){
								DumpBinary= DB;
								DumpDHex= DH;
								DumpIncluded= DI;
							}
							if( l_is_pipe ){
								pclose( rfp );
								if( l_is_pipe!= 2 ){
									--fname2;
								}
								Read_File_Hist= concat2( Read_File_Hist, " ", fname2, " ", NULL );
								read_file_hist= concat2( read_file_hist, " ", fname2, " ", NULL );
							}
							else{
								fclose( rfp );
								Read_File_Hist= concat2( Read_File_Hist, " ", fname2, " ", NULL );
								read_file_hist= concat2( read_file_hist, " ", fname2, " ", NULL );
							}
							if( ReadData_BufChanged || rd_bufsize!= LMAXBUFSIZE+2 ){
								ReadData_BufChanged= False;
								if( !(BUFFER= ReadData_AllocBuffer( BUFFER, LMAXBUFSIZE+2, the_file)) ){
									ReadData_RETURN(0);
								}
								  /* Got the memory; update the necessary variables!!	*/
								rd_bufsize= LMAXBUFSIZE+2;
								optbuf= BUFFER + (optbuf-buffer);
								buffer= BUFFER;
							}
							show_perc_now= True;
						}
						else{
							fprintf( StdErr, "ReadData(%s): can't open '%s' (%s)\n",
								the_file, (l_is_pipe==1)? --fname2 : fname2, serror()
							);
							fflush( StdErr );
						}
					}
				}
				xfree(FName2);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*DUMP_FILE*", 11)== 0 ){
		  char *fname= &optbuf[11], *astring= NULL;
		  char sFname[MAXPATHLEN];
		  int l_is_pipe= 0;
		  FILE *rfp;
		  Boolean ok= True, locked= False;
		  LocalWin *wi= ActiveWin;
			while( *fname && isspace((unsigned char)*fname) ){
				fname++;
			}
			if( fname[ strlen(fname)-1 ]== '\n' ){
				fname[ strlen(fname)-1 ]= '\0';
			}
			if( !wi && StartUp ){
				wi= &StubWindow;
				CopyFlags( wi, NULL );
				UpdateWindowSettings( wi, True, False );
				if( wi->filename_in_legend< 0 ){
					wi->filename_in_legend= 0;
				}
			}
			if( ok && wi ){

				{ char *c= ascanf_string( fname, NULL);
					if( c!= fname ){
						astring= fname= strdup(c);
					}
				}
				if( fname[0]== '|' ){
					fname++;
					system( "sh < /dev/null > /dev/null 2>&1");
					PIPE_error= False;
					strncpy( sFname, fname, MAXPATHLEN-1 );
					rfp= popen( sFname, "wb" );
					l_is_pipe= 1;
				}
				else{
					tildeExpand( sFname, fname );
					if( !strcasecmp( sFname, "stdout") ){
						rfp= stdout;
					}
					else if( !strcasecmp( sFname, "stderr") ){
						rfp= StdErr;
					}
					else if( strcasecmp( &sFname[strlen(sFname)-4], ".bz2" )== 0 ){
					  char buf[2*MAXPATHLEN];
						sprintf( buf, "bzip2 -9 -v > %s", sFname );
						rfp= popen( buf, "wb" );
						l_is_pipe= 1;
						fname++;
					}
					else if( strcasecmp( &sFname[strlen(sFname)-3], ".gz" ) ){
					  struct stat st;
					  int fd, isfifo= 0;
						if( stat( sFname, &st)== 0 ){
							  /* Don't lock a fifo!	*/
							isfifo= S_ISFIFO(st.st_mode);
						}
						else if( errno!= ENOENT ){
							fprintf( StdErr, "ReadData(%s): can't get stats on the outputfile (%s)\t(doing without)\n", optbuf, serror() );
						}
						  /* No actual truncation is done before we
						   \ start writing on the file (i.e. after getting
						   \ a lock). Since fopen(sFname,"a") ensures that no
						   \ truncation is possible, we have to resort to
						   \ lower level routines.
						   \ 980316 - I have to admit I don't remember why I want this.. - to obtain a lock?
						   */
						if( (fd= open( sFname, O_WRONLY|O_CREAT, 0755))!= -1 ){
							if( fchmod( fd, S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH) ){
								fprintf( StdErr, "ReadData(%s): Unable to chmod(755) file `%s' (%s)\n", optbuf, sFname, serror() );
							}
							  /* fdopen(fd,"w") should not truncate.	*/
							rfp= fdopen( fd, "wb");
						}
						else{
							rfp= NULL;
						}
						if( rfp ){
							if( !isfifo ){
#if !defined(__CYGWIN__)
#ifdef __MACH__
/* 								if( flock( fileno(rfp), LOCK_EX) )	*/
								errno= 0;
								flockfile(rfp);
								if( errno )
#else
								if( lockf( fileno(rfp), F_LOCK, 0) )
#endif
								{
									fprintf( StdErr, "ReadData(%s): can't get a lock on the outputfile (%s)\t(doing without)\n",
										optbuf, serror()
									);
								}
								else{
									locked= 1;
								}
#endif
							}
						}
						errno= 0;
						if( ftruncate( fileno(rfp), 0) ){
							fprintf( StdErr, "ReadData(%s): can't truncate outputfile (%s)\n", optbuf, serror() );
						}
					}
					else{
					  char buf[2*MAXPATHLEN];
						sprintf( buf, "gzip -9 -v > %s", sFname );
						rfp= popen( buf, "wb" );
						l_is_pipe= 1;
						fname++;
					}
				}
				if( rfp ){
				  char ebuf[ERRBUFSIZE];
					ebuf[0]= '\0';
					_XGraphDump( wi, rfp, ebuf );
					if( *ebuf ){
						fputs( ebuf, StdErr );
					}
					if( l_is_pipe ){
						pclose( rfp );
					}
					else{
						if( rfp!= StdErr && rfp!= stdout ){
							if( locked ){
#if !defined(__CYGWIN__)
#ifdef __MACH__
/* 								flock( fileno(rfp), LOCK_UN);	*/
								funlockfile(rfp);
#else
								lockf( fileno(rfp), F_ULOCK, 0);
#endif
#endif
								locked= 0;
							}
							fclose( rfp );
						}
					}
				}
				else{
					fprintf( StdErr, "ReadData(%s): can't open %s (%s)\n",
						the_file, (l_is_pipe)? --fname : fname, serror()
					);
					fflush( StdErr );
				}
			}
			xfree( astring );
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PRINT_DEV*", 11)== 0 ){
		  char *fname= &optbuf[11];
			while( *fname && isspace((unsigned char)*fname) ){
				fname++;
			}
			if( fname[ strlen(fname)-1 ]== '\n' ){
				fname[ strlen(fname)-1 ]= '\0';
			}
			if( *fname ){
				xfree( Odisp );
				Odisp= XGstrdup( fname );
				if( debugFlag ){
					fprintf( StdErr, "Printing to \"%s\"\n", Odevice );
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s:%s,%d.%d): *PRINT_DEV* needs a type specification\n",
					the_file, filename, sub_div, line_count
				);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PRINT_AS*", 10)== 0 ){
		  char *fname= &optbuf[10];
			while( *fname && isspace((unsigned char)*fname) ){
				fname++;
			}
			if( fname[ strlen(fname)-1 ]== '\n' ){
				fname[ strlen(fname)-1 ]= '\0';
			}
			if( *fname ){
				xfree( Odevice );
				Odevice= XGstrdup( fname );
				if( debugFlag ){
					fprintf( StdErr, "Printing as \"%s\"\n", Odevice );
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s:%s,%d.%d): *PRINT_AS* needs a type specification\n",
					the_file, filename, sub_div, line_count
				);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PRINT_FILE*", 12)== 0 ){
		  char *fname= &optbuf[12];
/* 		  extern char UPrintFileName[];	*/
			while( *fname && isspace((unsigned char)*fname) ){
				fname++;
			}
			if( fname[ strlen(fname)-1 ]== '\n' ){
				fname[ strlen(fname)-1 ]= '\0';
			}
			if( fname ){
				PrintFileName= XGstrdup( fname );
				if( debugFlag ){
					fprintf( StdErr, "Printing to \"%s\"\n", PrintFileName );
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s:%s,%d.%d): *PRINT_FILE* needs a filename\n",
					the_file, filename, sub_div, line_count
				);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*PRINT*", 7)== 0){
		  char *buf= &optbuf[7];
		  double opts[7];
		  int n= 7;
		  extern int XG_Stripped, Init_XG_Dump;
			while( *buf && isspace((unsigned char)*buf) ){
				buf++;
			}
			fascanf2( &n, parse_codes(buf), opts, ' ' );
			if( n>= 5 ){
				dump_average_values= (int) (opts[0]+ 0.5);
				DumpProcessed= (int) (opts[1]+ 0.5);
				XG_Stripped= (int) (opts[2]+ 0.5);
				Sort_Sheet= (int) (opts[3]+ 0.5);
				print_immediate= (opts[4])? -1 : -2;
				if( n> 5 ){
					Init_XG_Dump= (int) (opts[5]+ 0.5);
				}
				if( n> 6 ){
					*do_gsTextWidth_Batch= (opts[6])? 1 : 0;
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s:%s,%d.%d): *PRINT* needs at least 5 arguments, not %d\n",
					the_file, filename, sub_div, line_count, n
				);
				fflush( StdErr );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*ADD_FILE*", 10)== 0 ){
		  char *fname= &optbuf[10];
			while( *fname && isspace((unsigned char)*fname) ){
				fname++;
			}
			if( fname[ strlen(fname)-1 ]== '\n' ){
				fname[ strlen(fname)-1 ]= '\0';
			}
			if( No_IncludeFiles ){
				fprintf( StdErr, "ReadData(%s): \"%s\" ignored because of -NoIncludes\n",
					the_file, optbuf
				);
				fflush( StdErr );
			}
			else{
				xfree( AddFile );
				AddFile= XGstrdup( fname );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*SCRIPT_FILE*", 13)== 0 ){
		  char *fname= &optbuf[13];
			while( *fname && isspace((unsigned char)*fname) ){
				fname++;
			}
			if( fname[ strlen(fname)-1 ]== '\n' ){
				fname[ strlen(fname)-1 ]= '\0';
			}
			if( No_IncludeFiles ){
				fprintf( StdErr, "ReadData(%s): \"%s\" ignored because of -NoIncludes\n",
					the_file, optbuf
				);
				fflush( StdErr );
			}
			else{
				xfree( ScriptFile );
				if( *fname ){
					ScriptFile= XGstrdup( (strcmp(fname, "*This-File*"))? fname : the_file );
					ScriptFileWin= ActiveWin;
				}
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*POINTS*", 8)== 0 ){
		  double x= 0;
		  int n= 1, points;
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*POINTS*";
			if( fascanf( &n, parse_codes(&optbuf[8]), &x, NULL, NULL, NULL, NULL)== 1 ){
				if( (points= (int) fabs(x))> 0 ){
					INITSIZE= points;
					if( debugFlag ){
						fprintf( StdErr, "ReadData(), %s: line %d.%d INITSIZE=nr-of-points= %d\n",
							filename, sub_div, line_count, INITSIZE
						);
					}
				}
				else{
					line_count= UpdateLineCount( stream, &flpos, line_count );
					fprintf( StdErr, "ReadData(), %s: line %d.%d ignored invalid INITSIZE=nr-of-points= %d\n",
						filename, sub_div, line_count, points
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*ARROWS*", 8)== 0 ){
		  char *buf= &optbuf[8];
		  Boolean nonsense= False;
			while( *buf && isspace( *buf ) ){
				buf++;
			}
			if( *buf ){
				if( strncasecmp(buf, "BE", 2)== 0 || strncasecmp(buf, "EB", 2)== 0 ){
					this_set->arrows= 3;
				}
				else if( buf[0]== 'b' || buf[0]== 'B' ){
					this_set->arrows= 1;
				}
				else if( buf[0]== 'e' || buf[0]== 'E' ){
					this_set->arrows= 2;
				}
				else{
					this_set->arrows= 0;
				}
			}
			else{
				this_set->arrows= 0;
			}
			  /* Find possible next word, which will be the orientation(s) of the arrowhead(s)	*/
			while( *buf && !isspace( *buf ) ){
				buf++;
			}
			while( *buf && isspace( *buf ) ){
				buf++;
			}
			this_set->sarrow_orn_set= False;
			this_set->earrow_orn_set= False;
			if( *buf ){
			  double angle[2];
			  int n= 2;
				fascanf( &n, buf, angle, NULL, NULL, NULL, NULL);
				if( n== 1){
					switch( this_set->arrows ){
						case 3:
							fprintf( StdErr, "ReadData(), %s: line %d.%d applied single orn %s to start-arrow (B)\n",
								filename, sub_div, line_count, d2str( angle[0], NULL, NULL)
							);
						case 1:
							if( !NaN(angle[0]) ){
								this_set->sarrow_orn= angle[0];
								this_set->sarrow_orn_set= True;
							}
							break;
						case 2:
							if( !NaN(angle[0]) ){
								this_set->earrow_orn= angle[0];
								this_set->earrow_orn_set= True;
							}
							break;
						default:
							nonsense= True;
							break;
					}
				}
				else if( n== 2 ){
					switch( this_set->arrows ){
						case 3:
							if( !NaN(angle[0]) ){
								this_set->sarrow_orn= angle[0];
								this_set->sarrow_orn_set= True;
							}
							if( !NaN(angle[1]) ){
								this_set->earrow_orn= angle[1];
								this_set->earrow_orn_set= True;
							}
							break;
						case 1:
							if( !NaN(angle[0]) ){
								fprintf( StdErr, "ReadData(), %s: line %d.%d applied first orn %s to start-arrow (B)\n",
									filename, sub_div, line_count, d2str( angle[0], NULL, NULL)
								);
								this_set->sarrow_orn= angle[0];
								this_set->sarrow_orn_set= True;
							}
							break;
						case 2:
							if( !NaN(angle[0]) ){
								fprintf( StdErr, "ReadData(), %s: line %d.%d applied first orn %s to end-arrow (E)\n",
									filename, sub_div, line_count, d2str( angle[0], NULL, NULL)
								);
								this_set->earrow_orn= angle[0];
								this_set->earrow_orn_set= True;
							}
							break;
						default:
							nonsense= True;
							break;
					}
				}
				else{
					nonsense= True;
				}
			}
			if( nonsense ){
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d couldn't fully interprete \"%s\"\n",
					filename, sub_div, line_count, optbuf
				);
			}
		}
		else if( strncmp( optbuf, "*DISPLACED*", 11)== 0 ){
		  int n= 2;
		  double vals[2]= {0,0};
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*DISPLACED*";
			if( fascanf( &n, &optbuf[11], vals, NULL, NULL, NULL, NULL )!= EOF ){
				this_set->displaced_x= vals[0];
				this_set->displaced_y= vals[1];
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need >= 1 number\n",
					filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*BARPARS*", 9)== 0 ){
		  int n= 3;
		  double vals[3]= {0,0,0};
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*BARPARS*";
			if( fascanf( &n, &optbuf[9], vals, NULL, NULL, NULL, NULL )== 3 ){
				if( NaN(vals[0]) ){
					this_set->barBase= barBase;
					this_set->barBase_set= False;
				}
				else{
					this_set->barBase= vals[0];
					this_set->barBase_set= True;
				}
				if( NaN(vals[1]) ){
					this_set->barWidth= barWidth;
					this_set->barWidth_set= False;
				}
				else{
					this_set->barWidth= vals[1];
					this_set->barWidth_set= True;
				}
				barType_set= False;
				CLIP_EXPR(this_set->barType, (int) vals[2], 0, BARTYPES-1 );
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need 3 numbers\n",
					filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*VALUEMARKS*", 12)== 0 ){
		  char *opts= parse_codes( &optbuf[12] ), *c;
			if( (c= strcasestr( opts, "ON")) && (c== opts || isspace(c[-1])) && (c[2]== '\0' || isspace(c[2])) ){
				this_set->valueMarks|= VMARK_ON;
			}
			else{
				this_set->valueMarks&= ~VMARK_ON;
			}
			if( (c= strcasestr( opts, "FULL")) && (c== opts || isspace(c[-1])) && (c[4]== '\0' || isspace(c[4])) ){
				this_set->valueMarks|= VMARK_FULL;
			}
			else{
				this_set->valueMarks&= ~VMARK_FULL;
			}
			if( (c= strcasestr( opts, "RAW")) && (c== opts || isspace(c[-1])) && (c[3]== '\0' || isspace(c[3])) ){
				this_set->valueMarks|= VMARK_RAW;
			}
			else{
				this_set->valueMarks&= ~VMARK_RAW;
			}
			if( debugFlag && (!opts || !*opts) ){
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): valueMarks reset\n",
					filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
		}
		else if( strncmp( optbuf, "*ERROR_POINT*", 13)== 0 ){
		  double x= 0;
		  int n= 1;
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*ERROR_POINT*";
			if( fascanf( &n, parse_codes(&optbuf[13]), &x, NULL, NULL, NULL, NULL)== 1 ){
				this_set->error_point= (int) x;
				if( debugFlag ){
					fprintf( StdErr, "ReadData(), %s: line %d.%d error_point= %d\n",
						filename, sub_div, line_count, this_set->error_point
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*ERROR_TYPE*", 12)== 0 ){
		  double x[2]= {0,0};
		  int n= 2;
			fascanf( &n, parse_codes(&optbuf[12]), x, NULL, NULL, NULL, NULL);
			if( n>= 1 ){
				CLIP_EXPR( this_set->error_type, (int) x[0], -1, ERROR_TYPES-1 );
				CLIP_EXPR( error_type, (int) x[0], 0, ERROR_TYPES-1 );
				if( n> 1 && !NaN( x[1] ) ){
					this_set->ebarWidth_set= True;
					this_set->ebarWidth= x[1];
				}
				else{
					this_set->ebarWidth_set= False;
				}
				if( ActiveWin && ActiveWin!= &StubWindow && ActiveWin->error_type ){
					ActiveWin->error_type[this_set->set_nr]= error_type;
				}
				if( debugFlag ){
					fprintf( StdErr, "ReadData(), %s: line %d.%d error_type= %d (%s); width=%s\n",
						filename, sub_div, line_count, this_set->error_type,
						Error_TypeNames[(this_set->error_type< 0)? error_type : this_set->error_type],
						(this_set->ebarWidth_set)? d2str( x[1], NULL, NULL) : "default"
					);
				}
			}
			else{
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(), %s: line %d.%d (%s): need a number\n",
					filename, sub_div, line_count, optbuf
				);
				fflush( StdErr );
			}
		}
		else if( strncmp( optbuf, "*Cxye*", 6)== 0 ){
		  double col[ASCANF_DATA_COLUMNS]= {1,2,3,4};
		  int n= ASCANF_DATA_COLUMNS;
		  char *TBh= TBARprogress_header;
			TBARprogress_header= "*Cxye*";
			if( use_xye_info && fascanf( &n, parse_codes(&optbuf[6]), col, NULL, NULL, NULL, NULL) && n>= 2 ){
				if( n== 2 ){
					if( col[0]> col[1] ){
					  /* yx	*/
						column[0]= 1;
						column[1]= 0;
					}
				}
				else if( n>= 3 ){
					if( col[1]> col[2] && col[1]> col[0] ){
						column[1]= 2;
						if( col[0]<= col[2] ){
						  /* xey	*/
							column[0]= 0;
							column[2]= 1;
						}
						else{
						  /* yex	*/
							column[0]= 1;
							column[2]= 0;
						}
					}
					else if( col[1]< col[2] && col[1]< col[0] ){
						column[1]= 0;
						if( col[0]< col[2] ){
						  /* yxe	*/
							column[0]= 1;
							column[2]= 2;
						}
						else{
						  /* exy	*/
							column[0]= 2;
							column[2]= 1;
						}
					}
					else if( col[0]> col[1] && col[1]> col[2] ){
					  /* eyx	*/
						column[0]= 2;
						column[1]= 1;
						column[2]= 0;
					}
					else{
					  /* xye - default	*/
						column[0]= 0;
						column[1]= 1;
						column[2]= 2;
					}
				}
				  /* 20020415: to restore functionality of this command, we need to set the global variables
				   \ that reference the column order. This is probably acceptable since each set can override
				   \ this setting if necessary AND this only happens when the user willingly gave the -Cauto
				   \ argument.
				   */
				xcol= column[0], ycol= column[1], ecol= column[2];
				if( debugFlag || n> 3 ){
					fprintf( StdErr, "%s (n=%d): Input columns: x=%d y=%d e=%d",
						optbuf, n,
						column[0], column[1], column[2]
					);
					if( n> 3 ){
						fprintf( StdErr, " l=%g and further arguments (n=%d) are ignored for now",
							col[3], n
						);
					}
					fputc( '\n', StdErr );
				}
			}
			else if( debugFlag ){
				fprintf( StdErr, "%s: statement ignored because -Cauto has not been given or format error (n=%d<2).\n",
					optbuf, n
				);
			}
			TBARprogress_header= TBh;
		}
		else if( strncmp( optbuf, "*RESET_ATTRS*", 13)== 0 ){
			if( debugFlag ){
				fprintf( StdErr, "lw,ls,els,pv,ms=%g,%d,%d,%d,%d - resetting to %g,%d,%d,%d,%d\n",
					lineWidth, linestyle, elinestyle, pixvalue, markstyle,
					lw, ls, els, pv, ms
				);
			}
			lineWidth= this_set->lineWidth= lw;
			linestyle= this_set->linestyle= ls;
			elinestyle= this_set->elinestyle= els;
			pixvalue= this_set->pixvalue= pv;
			markstyle= this_set->markstyle= ms;
			incr_width= -newfile_incr_width;
			_incr_width= 0;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*ARGUMENTS*", 11)== 0 ){
			add_comment( optbuf );
			ReadData_commands+= 1;
			if( debugFlag ){
				fprintf( StdErr, "ReadData(%s:%s), line %d.%d: ",
					the_file, filename, sub_div, line_count
				);
			}
			ParseArgsString3( &optbuf[11], this_set->set_nr );
			  /* See the comments below under the next_line: label why it is
			   \ important to make sure to update at least this_set at this point!!!
			   */
			this_set= &AllSets[setNumber];
		}
		else if( strncmp( optbuf, "*EXTREMES*", 10)== 0 ){
			  /* 20031029: not yet implemented. */
			if( (extremes= (double*) calloc( 2*this_set->ncols, sizeof(double) )) ){
				extremesN= 2*this_set->ncols;
// 				fascanf( &extremesN, parse_codes(&optbuf[10]), extremes, NULL, NULL, NULL, NULL );
				fascanf2( &extremesN, parse_codes(&optbuf[10]), extremes, ',' );
			}
			ReadData_commands+= 1;
		}
#ifdef XG_DYMOD_SUPPORT
		else if( strncmp( optbuf, "*DM_IO_IMPORT*", 14)== 0 || strncasecmp( optbuf, "*ASK_DM_IO_IMPORT*", 18)== 0){
		  char *defbuf= (optbuf[1]== 'A' || optbuf[1]== 'a')? NULL : cleanup(&optbuf[14]);
		  char *libname= NULL, *fname= NULL, *free_fname= NULL, *ln_end= NULL;
		  char sFname[MAXPATHLEN];
		  ALLOCA( name, char, LMAXBUFSIZE+2, name_len);
			if( !defbuf ){
			  char *argbuf= cleanup( &optbuf[18] );
			  Boolean fok= False;
					if( DumpFile ){
						if( DF_bin_active ){
							BinaryTerminate(stdout);
							fputc( '\n', stdout);
							DF_bin_active= False;
						}
						if( !DumpPretty ){
							fputs( "*EXTRATEXT* ", stdout );
						}
						else{
							LineDumped= True;
						}
						print_string( stdout, optbuf, "\\n\n", "\n", argbuf);
						fputc( '\n', stdout );
					}
				if( argbuf[0] ){
				  char *charbuf= strstr( argbuf, "::"), *wid;
				  char *fbuf, *mbuf;
					if( charbuf && charbuf[2] ){
						  /* Cut the specified string here:	*/
						*charbuf= '\0';
						  /* retrieve the part after the 1st double colons: the library name	*/
						{ char *c= ascanf_string( &charbuf[2], NULL);
							if( c!= &charbuf[2] ){
								charbuf[0]= '\n';
								charbuf[1]= '>';
							}
							libname= XGstrdup( c );
						}
						if( (fbuf= strstr(libname, "::")) ){
							  /* Cut the specified string here:	*/
							*fbuf= '\0';
							if( fbuf[2] ){
							  char *c= ascanf_string( &fbuf[2], NULL);
							  /* retrieve the part after the double colons	*/
								  /* 20040216: not completely sure what I'm doing here: */
								if( c!= &fbuf[2] ){
									ln_end= fbuf;
									fbuf[0]= '\n';
									fbuf[1]= '>';
								}
								fbuf= c;
							}
						}
					}
					else{
						charbuf= NULL;
					}
					wid= cgetenv("WINDOWID");
					if( libname ){
						mbuf= concat( argbuf, "\n Will use library module `", libname, "'\n", NULL );
						do{
						  Boolean escaped= False;
							if( disp ){
							  int tilen= 15 /* -1 */;
							  Window win;
							  char *nbuf;
								if( fbuf ){
									strncpy( name, fbuf, LMAXBUFSIZE);
									tilen= 1.5* strlen(name);
								}
								else{
									name[0]= '\0';
								}
								name[LMAXBUFSIZE]= '\0';
								if( ActiveWin ){
									win= ActiveWin->window;
								}
								else if( InitWindow ){
									win= InitWindow->window;
								}
								else if( wid ){
									win= atoi(wid);
								}
								else{
									win= DefaultRootWindow(disp);
								}
								TitleMessage( ActiveWin, argbuf );
								if( (nbuf= xtb_input_dialog( win, name, tilen, LMAXBUFSIZE, mbuf,
										"Enter filename", False,
										"History", process_hist_h, "Edit", SimpleEdit_h, " ... ", SimpleFileDialog_h
									))
								){
									  /* Make a local copy into fname. This prevents that changes
									   \ to name will be visible via fname even when the dialog
									   \ was cancelled after a first filename spec was rejected.
									   */
									if( free_fname ){
										xfree( free_fname );
									}
									free_fname= fname= XGstrdup( cleanup(name) );
									if( nbuf!= name ){
										xfree( nbuf );
									}
								}
								else{
									escaped= True;
									free_fname= fname= XGstrdup(fbuf);
								}
								TitleMessage( ActiveWin, NULL );
							}
							else{
								fflush( stdout );
								fflush( StdErr );
								xgiReadString2( name, LMAXBUFSIZE, mbuf, fbuf );
								cleanup( name );
								if( (!name[0] || name[0]== '\n') && fbuf ){
									strncpy( name, fbuf, LMAXBUFSIZE);
									name[LMAXBUFSIZE]= '\0';
								}
								  /* We don't have the problem of cancelled dialogs in this case, but to be
								   \ consistent, we still make a local copy.
								   */
								xfree( fname );
								free_fname= fname= XGstrdup(name);
							}
							if( !escaped ){
							  FILE *fp;
								if( fname ){
									tildeExpand( sFname, fname );
									if( (fp= fopen(sFname, "r")) ){
										fok= True;
										fclose(fp);
									}
									else{
										fprintf( StdErr, "\"%s\": '%s'\n", fname, serror() );
									}
								}
							}
							else{
								fok= True;
								if( fname ){
									fprintf( StdErr,
										"Warning: dialog cancelled, selected previously entered filename"
										" \"%s\" without further verification!\n",
										fname
									);
								}
							}
						} while( !fok );
					}
					if( fname ){
						  /* Store the local copy back into the (more) global buffer, and
						   \ free the local copy.
						   */
						strncpy( name, fname, LMAXBUFSIZE );
						xfree( fname );
						free_fname= NULL;
						fname= cleanup(name);
					}
					GCA();
				}
			}
			else{
				libname= XGstrdup(defbuf);
				defbuf= libname;
				while( !isspace(*defbuf) ){
					defbuf++;
				}
				*defbuf= '\0';
				fname= ascanf_string( cleanup(&defbuf[1]), NULL );
				free_fname= NULL;
			}
			if( ln_end ){
				*ln_end= '\0';
			}
			{ ReadData_Pars currentFile;
			  ReadData_States state;
				currentFile.filename= filename;
				currentFile.stream= stream;
				currentFile.filenr= filenr;
				currentFile.swap_endian= swap_endian;
				state.spot= &spot;
				state.Spot= &Spot;
				state.sub_div= sub_div;
				state.line_count= line_count;
				  /* Not supposed to be changed by the handler, unless rfp==stream! */
				state.flpos= &flpos;
				state.ReadData_proc= &ReadData_proc;
				switch( doIO_Import( libname, fname, &currentFile, &state, this_set ) ){
					case 0:
						if( ReadData_BufChanged || rd_bufsize!= LMAXBUFSIZE+2 ){
							ReadData_BufChanged= False;
							if( !(BUFFER= ReadData_AllocBuffer( BUFFER, LMAXBUFSIZE+2, the_file)) ){
								ReadData_RETURN(0);
							}
							  /* Got the memory; update the necessary variables!!	*/
							rd_bufsize= LMAXBUFSIZE+2;
							optbuf= BUFFER + (optbuf-buffer);
							buffer= BUFFER;
						}
						show_perc_now= True;
						break;
					case 2:
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf( StdErr, "ReadData(%s): line %d.%d can't open '%s' (%s)\n",
							the_file, sub_div, line_count, fname, serror()
						);
						fflush( StdErr );
						break;
					default:
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf( StdErr, "ReadData(%s): line %d.%d can't load IO library '%s' (%s) or not a proper IO library\n",
							the_file, sub_div, line_count, libname, serror()
						);
						fflush( StdErr );
						break;
				}
			}
			xfree( libname );
			if( free_fname ){
				xfree(free_fname);
			}
			ReadData_commands+= 1;
		}
#endif
		else if( strncmp( optbuf, "*BINARYDATA*", 12)== 0 ){
		  short bdc, cols= 0, scols;
		  char *defbuf= &optbuf[12];
		  extern int BinarySize;
		  int ok, lines, size= BinarySize, endian= 0;
		  Boolean foreign;
		  float *fltbuf= NULL;
		  unsigned short *shrtbuf= NULL;
		  unsigned char *bytbuf= NULL;
			if( BinaryFieldSize ){
				bdc= BinaryDump.columns;
			}
			else{
				bdc= 0;
			}
			while( *defbuf && isspace(*defbuf) ){
				defbuf++;
			}
			if( (ok= sscanf( defbuf, "lines=%d columns=%hd size=%d swap-endian=%d", &lines, &cols, &size, &endian ))< 3 ){
				ok= sscanf( defbuf, "l=%d c=%hd s=%d se=%d", &lines, &cols, &size, &endian );
			}
			if( ok< 4 || endian< 0 ){
				endian= (SwapEndian || swap_endian);
				if( !s_e_set && !endian ){
					line_count= UpdateLineCount( stream, &flpos, line_count );
					fprintf( StdErr,
						"ReadData(%s): line %d.%d Warning: starting to read *BINARYDATA* without knowing the endianness!\n",
							filename, sub_div, line_count
					);
				}
				if( endian && ! endnmesg_shown ){
					fprintf( StdErr, "ReadData(): BINARYDATA will reverse byte-order in file\n" );
					endnmesg_shown= True;
				}
			}
			if( ok>= 3 ){
				ok= (lines>= 0 && cols> 0
						&& (size== sizeof(unsigned char) || size== sizeof(unsigned short)
							|| size== sizeof(float) || size==sizeof(double)))? True : False;
				foreign= True;
			}
			else{
			  /* Check for data	*/
				ok= fread( &cols, sizeof(cols), 1, stream );
				foreign= False;
				if( (SwapEndian || swap_endian) && ok ){
					SwapEndian_int16( &cols, 1 );
				}
			}
			if( size== sizeof(float) ){
				if( !(fltbuf= (float*) calloc( cols, sizeof(float))) ){
					fprintf( StdErr, "ReadData(): BINARYDATA float read: can't get read N=%d buffer (%s)\n",
						cols, serror()
					);
					ok= 0;
				}
			}
			else if( size== sizeof(unsigned char) ){
				if( !(bytbuf= (unsigned char*) calloc( cols, sizeof(unsigned char))) ){
					fprintf( StdErr, "ReadData(): BINARYDATA byte read: can't get read N=%d buffer (%s)\n",
						cols, serror()
					);
					ok= 0;
				}
			}
			else if( size== sizeof(unsigned short) ){
				if( !(shrtbuf= (unsigned short*) calloc( cols, sizeof(unsigned short))) ){
					fprintf( StdErr, "ReadData(): BINARYDATA short read: can't get read N=%d buffer (%s)\n",
						cols, serror()
					);
					ok= 0;
				}
			}
			else if( size!= sizeof(double) ){
				line_count= UpdateLineCount( stream, &flpos, line_count );
				fprintf( StdErr, "ReadData(%s): line %d.%d: unsupported indicated size==%d;\n"
					"\t sizeof(byte)==%d, sizeof(short)==%d sizeof(float)==%d, sizeof(double)==%d;\n"
					"\t will attempt to read doubles from the file...\n",
					filename, sub_div, line_count, size,
					sizeof(unsigned char), sizeof(unsigned short), sizeof(float), sizeof(double)
				);
			}
			  /* 20000224: augmented the minimum # of columns to 3. I think this should be so...	*/
			scols= MAX( 3, cols );
			if( ok && cols ){
			  int fieldnr= 0, fline= 0;
			  Extremes *Extreme=NULL;

				if( cols> bdc || cols> MaxCols ){
					if( cols> MaxCols ){
						MaxCols= cols;
					}
					AllocBinaryFields( cols, "ReadData()" );
				}
				else{
					BinaryDump.columns= cols;
					memset( BinaryDump.data, 0, sizeof(double)* BinaryDump.columns );
					memset( BinaryDump.data4, 0, sizeof(float)* BinaryDump.columns );
					memset( BinaryDump.data2, 0, sizeof(unsigned short)* BinaryDump.columns );
					memset( BinaryDump.data1, 0, sizeof(unsigned char)* BinaryDump.columns );
				}
				if( scols!= this_set->ncols && this_set->allocSize ){
					allocerr= 0;
					if( !(this_set->columns= realloc_columns( this_set, scols )) ){
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf( StdErr, "%s: line %d.%d (%s): %d allocation failure(s) (%s)\n", filename, sub_div, line_count, optbuf,
							allocerr, serror()
						);
						xg_abort();
					}
				}
				else{
					this_set->ncols= scols;
				}
				errno= 0;
				if( extremes && extremesN==cols*2 ){
					Extreme= (Extremes*) extremes;
				}
				if( foreign ){
				  int i;
					if( fline>= lines ){
						ok= False;
					}
					else if( size== sizeof(double) ){
						ok= (fread( BinaryDump.data, sizeof(double), cols, stream)== cols);
						if( endian ){
							SwapEndian_double( BinaryDump.data, cols );
						}
						fline+= 1;
					}
					else if( size== sizeof(float) ){
					  int n;
						if( (ok= (n=fread( fltbuf, sizeof(float), cols, stream))== cols) ){
							for( i= 0; i< cols; i++ ){
								if( endian ){
									SwapEndian_float( &fltbuf[i], 1 );
								}
								BinaryDump.data4[i]= fltbuf[i];
								BinaryDump.data[i]= (double) fltbuf[i];
							}
							fline+= 1;
						}
						else{
							fprintf( StdErr, "ReadData(): %d floats read error at 'line' %d; only %d read (%s)\n",
								cols, fline, n, serror()
							);
						}
					}
					else if( size== sizeof(unsigned short) ){
					  int n;
						if( (ok= (n=fread( shrtbuf, sizeof(unsigned short), cols, stream))== cols) ){
							for( i= 0; i< cols; i++ ){
								if( endian ){
									SwapEndian_int16( (int16_t*) &shrtbuf[i], 1 );
								}
								BinaryDump.data2[i]= shrtbuf[i];
								if( Extreme ){
									BinaryDump.data[i]=
										Extreme[i].min + (shrtbuf[i] * (Extreme[i].max - Extreme[i].min) / USHRT_MAX);
								}
								else{
									BinaryDump.data[i]= (double) shrtbuf[i];
								}
							}
							fline+= 1;
						}
						else{
							fprintf( StdErr, "ReadData(): %d shorts read error at 'line' %d; only %d read (%s)\n",
								cols, fline, n, serror()
							);
						}
					}
					else if( size== sizeof(unsigned char) ){
					  int n;
						if( (ok= (n=fread( bytbuf, sizeof(unsigned char), cols, stream))== cols) ){
							for( i= 0; i< cols; i++ ){
								BinaryDump.data1[i]= bytbuf[i];
								if( Extreme ){
									BinaryDump.data[i]=
										Extreme[i].min + (bytbuf[i] * (Extreme[i].max - Extreme[i].min) / UCHAR_MAX);
								}
								else{
									BinaryDump.data[i]= (double) bytbuf[i];
								}
							}
							fline+= 1;
						}
						else{
							fprintf( StdErr, "ReadData(): %d shorts read error at 'line' %d; only %d read (%s)\n",
								cols, fline, n, serror()
							);
						}
					}
				}
				else{
				  int i;
				  /* Read the data	*/
					if( size== sizeof(double) ){
						ok= (fread( BinaryDump.data, sizeof(double), cols, stream)== cols);
						if( (SwapEndian || swap_endian) && ok ){
							SwapEndian_double( BinaryDump.data, cols );
						}
					}
					else if( size== sizeof(float) ){
						ok= (fread( fltbuf, sizeof(float), cols, stream)== cols);
						if( ok ){
							for( i= 0; i< cols; i++ ){
								if( (SwapEndian || swap_endian) ){
									SwapEndian_float( &fltbuf[i], 1 );
								}
								BinaryDump.data4[i]= fltbuf[i];
								BinaryDump.data[i]= (double) fltbuf[i];
							}
						}
					}
					else if( size== sizeof(unsigned short) ){
						if( (ok= (fread( shrtbuf, sizeof(unsigned short), cols, stream))== cols) ){
							for( i= 0; i< cols; i++ ){
								if( (SwapEndian || swap_endian) ){
									SwapEndian_int16( (int16_t*) &shrtbuf[i], 1 );
								}
								BinaryDump.data2[i]= shrtbuf[i];
								if( Extreme ){
									BinaryDump.data[i]=
										Extreme[i].min + (shrtbuf[i] * (Extreme[i].max - Extreme[i].min) / USHRT_MAX);
								}
								else{
									BinaryDump.data[i]= (double) shrtbuf[i];
								}
							}
						}
					}
					else if( size== sizeof(unsigned char) ){
						if( (ok= (fread( bytbuf, sizeof(unsigned char), cols, stream))== cols) ){
							for( i= 0; i< cols; i++ ){
								BinaryDump.data1[i]= bytbuf[i];
								if( Extreme ){
									BinaryDump.data[i]=
										Extreme[i].min + (bytbuf[i] * (Extreme[i].max - Extreme[i].min) / UCHAR_MAX);
								}
								else{
									BinaryDump.data[i]= (double) bytbuf[i];
								}
							}
						}
					}
				}
				if( ok ){

					  /* 20040919: some check against the *EXTREMES* data. If not passed,
					   \ chances are that the binary data has been copied manually from a file from a different platform,
					   \ without taking care to set the correct endianness info. User could then be warned about that.
					   \ Be careful to cast (back) to the proper datatype, as BinaryDump.data and extremes are double*, but
					   \ the data can have been promoted from floats (which could trip the test spuriously).
					   \ (NB: this happened to me, so it is not so far-fetched a situation!)
					   */
					   /* 20050423: add support here for 'sparse' (empty) columns. This would involve reading the effective
					    \ number of columns, but storing it in the appropriate this_set->column. Correction: READ the effective
						\ number into BinaryDump.data, and then introduce the 'gaps' (NaNs) into that array.
						\ Also look into making the code less redundant (i.e. integrate 1st line handling into the big loop)!
						*/
					if( extremes ){
					  int i, j, N= MIN(extremesN/2, scols), pb= 0, lastCol= -1;
					  double lastVal;
					  SimpleStats SSlow, SShigh;
						SS_Init_(SSlow);
						SS_Init_(SShigh);
						if( size== sizeof(float) ){
							for( j= i= 0; i< N; i++ ){
							  int j_1 = j+1;
								if( (float) BinaryDump.data[i]< (float) extremes[j] ||
									(float) BinaryDump.data[i]> (float) extremes[j_1]
								){
									SS_Add_Data_( SSlow, 1, ((float) BinaryDump.data[i] - (float) extremes[j]), 1.0 );
									SS_Add_Data_( SShigh, 1, ((float) BinaryDump.data[i] - (float) extremes[j_1]), 1.0 );
									pb+= 1;
									lastCol= i;
									lastVal = (float) BinaryDump.data[i];
								}
								j+= 2;
							}
						}
						else{
							for( j= i= 0; i< N; i++ ){
							  int j_1 = j+1;
								if( BinaryDump.data[i]< extremes[j] ||
									BinaryDump.data[i]> extremes[j_1]
								){
									SS_Add_Data_( SSlow, 1, (BinaryDump.data[i] - extremes[j]), 1.0 );
									SS_Add_Data_( SShigh, 1, (BinaryDump.data[i] - extremes[j_1]), 1.0 );
									pb+= 1;
									lastCol= i;
									lastVal = BinaryDump.data[i];
								}
								j+= 2;
							}
						}
						if( pb && ReadData_Outliers_Warn ){
							fprintf( StdErr,
								"\nWarning: first row of set %d has %d value(s) (of %d/%d columns, last %d) not within the specified extremes.\n"
								" This may be because this data was copied manually from a file from another platform, without\n"
								" a proper accounting for the endianness of that, and the current platform.\n",
								this_set->set_nr, pb, N, scols, lastCol
							);
							if( lastCol != -1 ){
								fprintf( StdErr, "\tlast outlier: %g not in [%g,%g]\n",
									lastVal, extremes[lastCol*2], extremes[lastCol*2+1]
								);
							}
							if( SSlow.count > 1 || SShigh.count > 1 ){
								fprintf( StdErr, "\tAverage lower error: %s ; Average higher error: %s\n",
									SS_sprint( NULL, "%g", "#xb1", 0, &SSlow),
									SS_sprint( NULL, "%g", "#xb1", 0, &SShigh)
								);
							}
						}
					}

					if( scols!= this_set->ncols ){
						if( DumpFile ){
							if( DF_bin_active ){
								BinaryTerminate(stdout);
								fputc( '\n', stdout);
								DF_bin_active= False;
							}
							fprintf( stdout, "*COLUMNS* N=%d x=%d y=%d e=%d l=%d",
								scols, this_set->xcol, this_set->ycol, this_set->ecol, this_set->lcol
							);
#if ADVANCED_STATS == 2
							fprintf( StdErr, " n=%d", this_set->Ncol );
#endif
							fputc( '\n', StdErr );
						}
						this_set->ncols= scols;
					}

					{ UserLabel *ul= StubWindow.ulabel;
						StubWindow.ulabel= ul;
						CopyFlags( &StubWindow, NULL);
					}

					if( polarFlag){
						polarLog= logYFlag;
/* #define polarLog 0	*/
						Gonio_Base( &StubWindow, StubWindow.radix, StubWindow.radix_offset );
					}
					do{
						if( DumpFile && DumpBinary ){
							if( !DF_bin_active ){
								fputs( "*BINARYDATA*\n", stdout );
								DF_bin_active= True;
							}
							BinaryDumpData( stdout, False );
							LineDumped= True;
						}
						  /* a bout of code which is largely identical to the ASCII version just below.
						   \ probably (quite) some redudancy.. I'm too lazy to clean this up now.
						   */
						if( _incr_width ){
							incr_width+= _incr_width;
							if( debugFlag ){
								fprintf( StdErr, "New incr_width (+%g) : %g\n",
									_incr_width, incr_width
								);
							}
							_incr_width= 0;
						}

						data[0]= data[1]= data[2]= data[3]= 0.0;
						  /* there *might* be a different number of columns on every "line". That's
						   \ nonsense, ofcourse, and not as XGraph itself will generate things, but,
						   \ again, one never knows.
						   */
						numcoords= BinaryDump.columns;
						if( numcoords== 1 ){
						  double d= BinaryDump.data[0];
							  /* special case. This allows single columns to be interpreted as (<index>,y)
							   \ tables, independent of the Cxye and/or Cauto option(s).
							   */
							BinaryDump.data[column[0]]= (double) spot;
							BinaryDump.data[column[1]]= d;
							numcoords= 2;
						}
						if( numcoords < 1 || numcoords> this_set->ncols ){
							    /* The linenumber on this error message is of course useless...	*/
							  line_count= UpdateLineCount( stream, &flpos, line_count );
							  fprintf(StdErr, "file: `%s', line: %d.%d (%s field %d)\n",
									 filename, sub_div, line_count, buffer, fieldnr );
							  fprintf(StdErr, "Must have between 1 and %d coordinates per line (not %d)\n", this_set->ncols, numcoords);
						}
						else{
						  int i;
							  /* Set to 0 any columns not read for this field	*/
							for( i= numcoords; i< cols; i++ ){
								if( debugFlag && i== numcoords ){
									fprintf( StdErr, "ReadData(), line %d: this may be a bug: numcoords==%d < cols==%d!\n",
										__LINE__, numcoords, cols
									);
									fflush( StdErr );
								}
								BinaryDump.data[i]= 0;
							}
							if( DumpFile && !DumpBinary ){
								if( DF_bin_active ){
									BinaryTerminate(stdout);
									fputc( '\n', stdout);
									DF_bin_active= False;
								}
								fprintf( stdout, "%s", d2str( BinaryDump.data[0], d3str_format, NULL) );
								for( i= 1; i< cols; i++ ){
									fprintf( stdout, "\t%s", d2str( BinaryDump.data[i], d3str_format, NULL) );
								}
								fputc( '\n', stdout );
								LineDumped= True;
							}
							AddPoint_discard= False;
							AddPoint( &this_set, &spot, &Spot, cols, BinaryDump.data, column, filename, sub_div, line_count, &flpos, buffer, &ReadData_proc );
							if( debugFlag && AddPoint_discard> 0 ){
							  static long N= 0;
								fprintf( StdErr, "Discarded point #%ld (#%d,%g,%g,%g) (%s:line %d.%d field %d) next: #%d\n",
									N++, Spot, BinaryDump.data[0], BinaryDump.data[1], BinaryDump.data[2],
									filename, sub_div, line_count, fieldnr, spot
								);
							}
							read_data= True;
						}
						if( foreign ){
						  int i;
							if( fline>= lines ){
								ok= False;
							}
							else if( size== 8 ){
								ok= (fread( BinaryDump.data, sizeof(double), cols, stream)== cols);
								if( endian ){
									SwapEndian_double( BinaryDump.data, cols );
								}
								fline+= 1;
							}
							else if( size== 4 ){
								if( (ok= (fread( fltbuf, sizeof(float), cols, stream))== cols) ){
									for( i= 0; i< cols; i++ ){
										if( endian ){
											SwapEndian_float( &fltbuf[i], 1 );
										}
										BinaryDump.data4[i]= fltbuf[i];
										BinaryDump.data[i]= (double) fltbuf[i];
									}
									fline+= 1;
								}
							}
							else if( size== 2 ){
								if( (ok= (fread( shrtbuf, sizeof(unsigned short), cols, stream))== cols) ){
									for( i= 0; i< cols; i++ ){
										if( endian ){
											SwapEndian_int16( (int16_t*) &shrtbuf[i], 1 );
										}
										BinaryDump.data2[i]= shrtbuf[i];
										if( Extreme ){
											BinaryDump.data[i]=
												Extreme[i].min + (shrtbuf[i] * (Extreme[i].max - Extreme[i].min) / USHRT_MAX);
										}
										else{
											BinaryDump.data[i]= (double) shrtbuf[i];
										}
									}
									fline+= 1;
								}
							}
							else if( size== 1 ){
								if( (ok= (fread( bytbuf, sizeof(unsigned char), cols, stream))== cols) ){
									for( i= 0; i< cols; i++ ){
										BinaryDump.data1[i]= bytbuf[i];
										if( Extreme ){
											BinaryDump.data[i]=
												Extreme[i].min + (bytbuf[i] * (Extreme[i].max - Extreme[i].min) / UCHAR_MAX);
										}
										else{
											BinaryDump.data[i]= (double) bytbuf[i];
										}
									}
									fline+= 1;
								}
							}
						}
						else{
						  /* Check for the next field (presence of data). 	*/
							if( !fread( &BinaryDump.columns, sizeof(short), 1, stream) ){
								BinaryDump.columns= 0;
								ok= False;
							}
							else{
							  int i;
								if( (SwapEndian || swap_endian) ){
									SwapEndian_int16( &BinaryDump.columns, 1 );
								}
								if( BinaryDump.columns< 0 ){
									if( !endian ){
										line_count= UpdateLineCount( stream, &flpos, line_count );
										fprintf(StdErr, "file: `%s', line: %d.%d (%s field %d)\n",
											filename, sub_div, line_count, buffer, fieldnr );
										fprintf(StdErr,
											"A negative (%d) number of columns was specified: "
											"you may want to try swapping the endianness (-SwapEndian1)\n",
											(int) BinaryDump.columns
										);
									}
									BinaryDump.columns= 0;
								}
								if( size== sizeof(double) ){
									ok= fread( BinaryDump.data, sizeof(double), (size_t) MIN(cols, BinaryDump.columns), stream);
									if( (SwapEndian || swap_endian) && ok ){
										SwapEndian_double( BinaryDump.data, cols );
									}
								}
								else if( size== sizeof(float) ){
									ok= fread( fltbuf, sizeof(float), (size_t) MIN(cols, BinaryDump.columns), stream);
									if( ok ){
										for( i= 0; i< cols; i++ ){
											if( (SwapEndian || swap_endian) ){
												SwapEndian_float( &fltbuf[i], 1 );
											}
											BinaryDump.data4[i]= fltbuf[i];
											BinaryDump.data[i]= (double) fltbuf[i];
										}
									}
								}
								else if( size== sizeof(unsigned short) ){
									ok= fread( shrtbuf, sizeof(unsigned short), (size_t) MIN(cols, BinaryDump.columns), stream);
									if( ok ){
										for( i= 0; i< cols; i++ ){
											if( (SwapEndian || swap_endian) ){
												SwapEndian_int16( (int16_t*) &shrtbuf[i], 1 );
											}
											BinaryDump.data2[i]= shrtbuf[i];
											if( Extreme ){
												BinaryDump.data[i]=
													Extreme[i].min + (shrtbuf[i] * (Extreme[i].max - Extreme[i].min) / USHRT_MAX);
											}
											else{
												BinaryDump.data[i]= (double) shrtbuf[i];
											}
										}
									}
								}
								else if( size== sizeof(unsigned char) ){
									ok= fread( bytbuf, sizeof(unsigned char), (size_t) MIN(cols, BinaryDump.columns), stream);
									if( ok ){
										for( i= 0; i< cols; i++ ){
											BinaryDump.data1[i]= bytbuf[i];
											if( Extreme ){
												BinaryDump.data[i]=
													Extreme[i].min + (bytbuf[i] * (Extreme[i].max - Extreme[i].min) / UCHAR_MAX);
											}
											else{
												BinaryDump.data[i]= (double) bytbuf[i];
											}
										}
									}
								}
							}
						}
						fieldnr+= 1;
						  /* Loop as long as there is data.
						   \ NB: if we really start expecting always the same fieldsize (the terminator is actually a 0 followed
						   \ by <cols> doubles), reverse the order of the 2 following test, first reading, and then checking
						   \ the columns field. Right now, things also work when the <data> field are just silently supposed to
						   \ be 0, but actually not read (and thus, they should not be present :))
						   \ We always read the number of columns specified by the field, but at maximum <cols>, of course...
						   */
					} while( BinaryDump.columns> 0 && ok );

					if( DumpFile && DumpBinary && DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout );
						DF_bin_active= False;
						LineDumped= True;
					}

					{ int c;
						  /* See if there's more on the same line. In principle, we expect a newline
						   \ after the BinaryTerminator, but one never knows :)
						   */
						if( (c= fgetc(stream))!= EOF && c!= '\n' ){
							ungetc( c, stream );
						}
					}
					LineDumped= True;
				}
			}
			  /* RJVB 20030825: weird. Reading from DOS files (with \r\n EOLs), we end up here and
			   \ stream is not correctly aligned with the end of the binary section (the newline terminating it)?!
			   \ So, we just attempt to align it ourselves: we read up to the next newline, which we unget when
			   \ found.
			   */
			{ int c= fgetc(stream);
				if( c!= '\n' ){
					while( (c!= '\n') && !feof(stream) && !ferror(stream) ){
						c= fgetc(stream);
					}
				}
				ungetc( c, stream );
			}
			xfree( fltbuf );
			xfree( shrtbuf );
			xfree( bytbuf );
		}
		else if( Interactive_Commands( this_set, optbuf, buffer, filename, sub_div, data, &DF_bin_active, &LineDumped, ReadData_level_str ) ){
		  /* handled	*/
		}
		else if( strncmp( optbuf, "*GLOBAL_COLOURS*", 16)== 0){
		  char *c= &optbuf[16];
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			while( c && optbuf[0]!= '\n' && !feof(stream) ){
				c= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
				if( c && DumpFile ){
					fputs( buffer, stdout );
				}
				if( c && optbuf[0]!= '\n' ){
				  Pixel tp;
					XGStoreColours= 1;
					cleanup(c);
					if( strncmp( c, "black=", 6)== 0 ){
						c+= 6;
						if( strcasecmp( c, blackCName) && GetColor( c, &tp) ){
							FreeColor( &black_pixel, &blackCName );
							black_pixel= tp;
							StoreCName( blackCName );
						}
					}
					else if( strncmp( c, "white=", 6)== 0 ){
						c+= 6;
						if( strcasecmp( c, whiteCName) && GetColor( c, &tp) ){
							FreeColor( &white_pixel, &whiteCName );
							white_pixel= tp;
							StoreCName( whiteCName );
						}
					}
					else if( strncmp( c, "bg=", 3)== 0 ){
						c+= 3;
						if( strcasecmp( c, bgCName) && GetColor( c, &tp) ){
							FreeColor( &bgPixel, &bgCName );
							bgPixel= tp;
							StoreCName( bgCName );
						}
					}
					else if( strncmp( c, "fg=", 3)== 0 ){
						c+= 3;
						if( strcasecmp( c, normCName) && GetColor( c, &tp) ){
						  int set;
						  Pixel np= normPixel;
							FreeColor( &normPixel, &normCName );
							normPixel= tp;
							StoreCName( normCName );
							for( set = 0;  set < MAXATTR;  set++) {
								if( AllAttrs[set].pixelValue== np ){
									xfree( AllAttrs[set].pixelCName );
									AllAttrs[set].pixelCName= XGstrdup( normCName );
									AllAttrs[set].pixelValue= normPixel;
								}
							}
						}
					}
					else if( strncmp( c, "bdr=", 4)== 0 ){
						c+= 4;
						if( strcasecmp( c, bdrCName) && GetColor( c, &tp) ){
							FreeColor( &bdrPixel, &bdrCName );
							bdrPixel= tp;
							StoreCName( bdrCName );
						}
					}
					else if( strncmp( c, "zero=", 5)== 0 ){
						c+= 5;
						if( strcasecmp( c, zeroCName) && GetColor( c, &tp) ){
							FreeColor( &zeroPixel, &zeroCName );
							zeroPixel= tp;
							StoreCName( zeroCName );
						}
					}
					else if( strncmp( c, "axis=", 5)== 0 ){
						c+= 5;
						if( strcasecmp( c, axisCName) && GetColor( c, &tp) ){
							FreeColor( &axisPixel, &axisCName );
							axisPixel= tp;
							StoreCName( axisCName );
						}
					}
					else if( strncmp( c, "grid=", 5)== 0 ){
						c+= 5;
						if( strcasecmp( c, gridCName) && GetColor( c, &tp) ){
							FreeColor( &gridPixel, &gridCName );
							gridPixel= tp;
							StoreCName( gridCName );
						}
					}
					else if( strncmp( c, "hl=", 3 )== 0 ){
						c+= 3;
						if( strcasecmp( c, highlightCName) && GetColor( c, &tp) ){
							FreeColor( &highlightPixel, &highlightCName );
							highlightPixel= tp;
							StoreCName( highlightCName );
						}
					}
					else{
						line_count= UpdateLineCount( stream, &flpos, line_count );
						fprintf(StdErr, "file: `%s', line: %d.%d (%s): invalid *GLOBAL_COLOUR* specification\n",
							filename, sub_div, line_count, buffer
						);
					}
				}
				line_count++;
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*INTENSITY_COLOURS*", 19)== 0){
		  char *exp= cleanup( &optbuf[19] );
			IntensityColourFunction.use_table= 0;
			if( strcmp( exp, "use") ){
				Intensity_Colours( exp );
			}
			else{
				Intensity_Colours( IntensityColourFunction.expression );
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*INTENSITY_COLOUR_TABLE*", 24)== 0){
		  char *cname= cleanup( &optbuf[24] );
			IntensityColourFunction.use_table= 1;
			if( strcmp( cname, "use") ){
				if( strcmp( cname, "new")== 0 ){
					IntensityColourFunction.name_table= XGStringList_Delete( IntensityColourFunction.name_table );
				}
				if( DumpFile ){
					if( DF_bin_active ){
						BinaryTerminate(stdout);
						fputc( '\n', stdout);
						DF_bin_active= False;
					}
					fputs( buffer, stdout );
				}
				while( cname && optbuf[0]!= '\n' && !feof(stream) ){
					cname= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
					if( DumpFile ){
						fputs( buffer, stdout );
					}
					if( cname && *cname && buffer[0] && buffer[0]!= '\n' ){
						IntensityColourFunction.name_table=
							XGStringList_AddItem( IntensityColourFunction.name_table, cleanup(cname) );
					}
					else{
						cname= NULL;
					}
					line_count++;
				}
				LineDumped= True;
				Intensity_Colours( NULL );
			}
			else{
				Intensity_Colours(NULL);
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*INTENSITY_RANGE*", 17)== 0){
		  char *exp= cleanup( &optbuf[17] );
			if( strcasecmp( exp, "auto")== 0 ){
				IntensityColourFunction.range_set= False;
			}
			else{
			  int n= 2;
			  double range[2];
			  char *TBh= TBARprogress_header;
				TBARprogress_header= "*INTENSITY_RANGE*";
				if( fascanf( &n, exp, range, NULL, NULL, NULL, NULL)>= 2 ){
					IntensityColourFunction.range_set= True;
					IntensityColourFunction.range.min= range[0];
					IntensityColourFunction.range.max= range[1];
				}
				TBARprogress_header= TBh;
			}
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*SKIP_TO*", 9)== 0){
		  char *label= NULL, *c;
		  int found, labellen;
skip_to_label:
			c= &optbuf[9];
			found= 0, labellen= 0;
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fputs( buffer, stdout );
			}
			LineDumped= True;
			while( c && *c && isspace(*c) ){
				c++;
			}
			if( c && *c ){
				label= c;
				if( *(c= &label[strlen(label)-1])== '\n' ){
					*c= '\0';
				}
				c= label;
				label= strdup(c);
				labellen= strlen(label);
			}
			if( label ){
			  extern int ReadString_Warn;
			  int n= 0, rw= ReadString_Warn, init_begin= False;
				c= buffer;
				ReadData_commands+= 1;
				ReadString_Warn= 0;
				while( c && !found && !feof(stream) ){
					c= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
					n+= 1;
					if( c ){
						if( DumpFile ){
							fputs( buffer, stdout );
						}
					}
					line_count++;

					optbuf= buffer;
					optcol= 0;
					while( *optbuf== '\t' && *optbuf!= '*' && *optbuf!= '\n' && *optbuf){
						optcol++;
						optbuf++;
					}
					if( optbuf[0]== '*' ){
						if( optbuf[labellen+1]== '*' && strncmp( &optbuf[1], label, labellen)== 0 ){
							found= 1;
						}
						else if( strncmp( &optbuf[1], "INIT_BEGIN*", 11)== 0 ){
							init_begin= True;
							found= 1;
						}
					}
				}
				ReadString_Warn= rw;
				line_count= UpdateLineCount( stream, &flpos, line_count );
				if( found ){
					if( init_begin ){
						fprintf( StdErr,
							"file: `%s', line: %d.%d (%s): encountered *INIT_BEGIN* after skipping %d lines searching for \"%s\"\n"
							"\tsuspending -skip_to/*SKIP_TO* until after the first *INIT_END* command!\n",
							filename, sub_div, line_count, buffer, n, label
						);
					}
					else{
						fprintf( StdErr, "file: `%s', line: %d.%d (%s): skipped %d lines searching for \"%s\"\n",
							filename, sub_div, line_count, buffer, n, label
						);
					}
					if( init_begin ){
						skip_to_label= label;
						label= NULL;
						goto init_begin;
					}
					else{
						  /* The label looks like an actual command, and can in fact be one. Thus,
						   \ we try to parse the currently read line.
						   */
						Skip_new_line= 1;
						line_count-= 1;
					}
					xfree( label );
				}
				else if( feof(stream) ){
					fprintf( StdErr,
						"file: `%s', line: %d.%d (%s): skipped remainder of file (%d lines) searching for missing \"%s\"\n",
						filename, sub_div, line_count, buffer, n, label
					);
				}
				  /* This won't hurt if we already freed label: */
				xfree( label );
			}
		}
#ifdef XG_DYMOD_SUPPORT
		else if( strncmp( optbuf, "*LOAD_MODULE*", 13)== 0){
		  char *buf= parse_codes( &optbuf[13] );
		  int flags= RTLD_LAZY, auto_unload= False, no_dump= False, autolist= False;
		  DyModAutoLoadTables new;
		  extern DyModAutoLoadTables *AutoLoadTable;
		  extern int AutoLoads;
			while( isspace(*buf) ){
				buf++;
			}
			if( strncasecmp( buf, "auto-load", 9)== 0 ){
				autolist= True;
				buf+= 9;
				while( isspace(*buf) ){
					buf++;
				}
			}
			if( strncasecmp( buf, "export", 6 )== 0 ){
				flags|= RTLD_GLOBAL;
				buf+= 6;
				while( isspace(*buf) ){
					buf++;
				}
			}
			if( strncasecmp( buf, "auto", 4 )== 0 ){
				auto_unload= True;
				buf+= 4;
				while( isspace(*buf) ){
					buf++;
				}
			}
			if( strncasecmp( buf, "nodump", 4 )== 0 ){
				no_dump= True;
				buf+= 6;
				while( isspace(*buf) ){
					buf++;
				}
			}
			if( *buf ){
				Add_Comment( optbuf, True );
			}
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fprintf( stdout, "%s\n", buffer );
			}
			buf[0]= '*';
			while( buf && *buf ){
				while( buf && optbuf[0]!= '\n' && !feof(stream) ){
					buf= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
					if( buf ){
						if( DumpFile ){
							fputs( buffer, stdout );
						}
						if( buf[0]!= '\n' ){
							buf= cleanup(buf);
							if( *buf ){
								if( buf[ strlen(buf)-1 ]== '\n' ){
									buf[ strlen(buf)-1 ]= '\0';
								}
								if( autolist ){
								  char *c;
								  int str= False;
									if( (c= ascanf_index(buf, ascanf_separator, &str)) ){
										*c= '\0';
										memset( &new, 0, sizeof(new) );
										new.functionName= buf;
										new.DyModName= &c[1];
										new.flags= flags;
										AutoLoadTable= Add_LoadDyMod( AutoLoadTable, &AutoLoads, &new, 1 );
									}
									else{
										fprintf( StdErr, "file: `%s', line: %d.%d (%s): invalid auto-load specification!\n",
											filename, sub_div, line_count, buffer
										);
									}
								}
								else{
									if( !LoadDyMod( buf, flags, no_dump, auto_unload ) ){
										ascanf_exit= True;
									}
								}
							}
						}
						else{
							buf= NULL;
						}
					}
					line_count++;
				}
			}
			if( autolist && debugFlag ){
			  int i;
				fprintf( StdErr, "DyMod autoload table:\n" );
				for( i= 0; i< AutoLoads; i++ ){
					fprintf( StdErr, "[%s] : \"%s\"\n", AutoLoadTable[i].functionName, AutoLoadTable[i].DyModName );
				}
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
		else if( strncmp( optbuf, "*UNLOAD_MODULE*", 15)== 0 ||
			strncmp( optbuf, "*RELOAD_MODULE*", 15)== 0
		){
		  char *buf= parse_codes( &optbuf[15] ), *c;
		  int reload= (optbuf[1]== 'R')? True : False, force= False, all= False;
		  int flags= RTLD_LAZY, no_dump= False, auto_unload= False;
			while( isspace(*buf) ){
				buf++;
			}
			if( !isspace( buf[strlen(buf)-1] ) ){
				strcat( buf, " ");
			}
			if( (c= strcasestr( buf, "force")) && isspace( c[5] ) ){
				force= True;
			}
			if( reload ){
				if( (c= strcasestr( buf, "export" )) && isspace( c[6]) ){
					flags|= RTLD_GLOBAL;
				}
				if( (c= strcasestr( buf, "auto" )) && isspace( c[4]) ){
					auto_unload= True;
				}
				if( (c= strcasestr( buf, "nodump" )) && isspace( c[6]) ){
					no_dump= True;
				}
			}
			if( (c= strcasestr( buf, "all")) && isspace( c[3]) ){
				if( reload ){
					line_count= UpdateLineCount( stream, &flpos, line_count );
					fprintf( StdErr, "file: `%s', line: %d.%d (%s): ignoring \"all\" opcode when reloading modules!\n",
						filename, sub_div, line_count, buffer
					);
				}
				else{
					all= True;
				}
			}
			if( *buf ){
				Add_Comment( buf, True );
			}
			if( DumpFile ){
				if( DF_bin_active ){
					BinaryTerminate(stdout);
					fputc( '\n', stdout);
					DF_bin_active= False;
				}
				fprintf( stdout, "%s\n", buffer );
			}
			if( all ){
			  int n;
			  int N= DyModsLoaded, i= 0;
				while( DyModList && i<= N /* DyModsLoaded */ ){
					UnloadDyMod( DyModList->name, &n, force );
					i+= 1;
				}
				if( DyModsLoaded ){
					fprintf( StdErr, "Warning: %d modules are still loaded!\n", DyModsLoaded );
				}
			}
			buf[0]= '*';
			while( buf && *buf ){
				while( buf && optbuf[0]!= '\n' && !feof(stream) ){
					buf= xgiReadString( buffer, LMAXBUFSIZE, stream, &line_count, XGStartJoin, XGEndJoin );
					if( buf ){
						if( DumpFile ){
							fputs( buffer, stdout );
						}
						if( buf[0]!= '\n' ){
							buf= parse_codes(buf);
							if( !all && *buf ){
							  int n, c;
								if( buf[ strlen(buf)-1 ]== '\n' ){
									buf[ strlen(buf)-1 ]= '\0';
								}
								n= UnloadDyMod( buf, &c, force );
								if( n== c ){
									if( reload ){
										  /* NB: we *could* reset Unloaded_Used_Modules here to the value it had
										   \ before the above call to UnloadDyMod(). However, as we're not
										   \ sure that the reload will have put all symbols back at they're original
										   \ addresses, we won't, since that would make a coredump in CleanUp
										   \ possible.
										   */
										if( !LoadDyMod( buf, flags, no_dump, auto_unload ) ){
											ascanf_exit= True;
										}
									}
								}
							}
						}
						else{
							buf= NULL;
						}
					}
					line_count++;
				}
			}
			LineDumped= True;
			ReadData_commands+= 1;
		}
#endif
		else if( Skipped_new_line ){
		  /* must remain just before the code that will interpret for data..
		   \ is a dummy that handles a label that has no other definition.
		   */
			Skipped_new_line= 0;
		}

		else if( optbuf[0]== '*' && IsLabel(optbuf) ){
			line_count= UpdateLineCount( stream, &flpos, line_count );
			fprintf( StdErr,
				"file: `%s', line: %d.%d (%s): unknown label (of more recent version?) ignored\n",
				filename, sub_div, line_count, buffer
			);
			unknown_label= True;
		}

		else {
ascii_ReadData:;
/* actual ReadData	check memory... */

			if( _incr_width ){
				incr_width+= _incr_width;
				if( debugFlag ){
					fprintf( StdErr, "New incr_width (+%g) : %g\n",
						_incr_width, incr_width
					);
				}
				_incr_width= 0;
			}


/* now the real
int ReadData(stream, filename)
 */

			{ UserLabel *ul= StubWindow.ulabel;
				StubWindow.ulabel= ul;
				CopyFlags( &StubWindow, NULL);
			}
			CopyFlags( &StubWindow, NULL);

			if( polarFlag){
				polarLog= logYFlag;
/* #define polarLog 0	*/
/* 				logYFlag=  0;	*/
				Gonio_Base( &StubWindow, StubWindow.radix, StubWindow.radix_offset );
			}

			if( rindex(buffer, '\n') )
				*(rindex(buffer, '\n'))= '\0';
			data[0]= data[1]= data[2]= data[3]= 0.0;
			numcoords= this_set->ncols;
			{ static double *DataBuf= NULL, nanval;
			  static int DataLen= 0;
				if( !DataBuf || !DataLen ){
					DataBuf= (double*) calloc( numcoords, sizeof(double) );
					DataLen= numcoords;
					set_NaN(nanval);
				}
				else if( numcoords> DataLen ){
					DataBuf= (double*) realloc( (char*) DataBuf, numcoords* sizeof(double) );
					DataLen= numcoords;
				}
				fascanf2( &numcoords, buffer, DataBuf, data_separator);
				if( numcoords== 1 ){
				  double d= DataBuf[0];
					  /* special case. This allows single columns to be interpreted as (<index>,y)
					   \ tables, independent of the Cxye and/or Cauto option(s).
					   */
					DataBuf[column[0]]= (double) spot;
					DataBuf[column[1]]= d;
					numcoords= 2;
				}
				if( numcoords < 1 || numcoords> this_set->ncols )
				{
					  line_count= UpdateLineCount( stream, &flpos, line_count );
					  fprintf(StdErr, "file: `%s', line: %d.%d (%s)\n",
							 filename, sub_div, line_count, buffer);
					  fprintf(StdErr, "Must have between 1 and %d coordinates per line (not %d)\n", this_set->ncols, numcoords);
				}
				else{
					if( numcoords< this_set->ncols ){
					  int i;
						/* 20051103: */
						  /* 20070427: column is int[ASCANF_DATA_COLUMNS] */
						for( i= numcoords; i< this_set->ncols && i< (sizeof(column)/sizeof(column[0])); i++ ){
							DataBuf[column[i]]= nanval;
						}
						if( debugFlag && (Spot<10 || debugLevel) ){
							fprintf(StdErr, "file: `%s', line: %d.%d: read only %d values of set's specified %d:\n"
							  	"\tfilling the remainder with NaNs (%s)\n"
								, filename, sub_div, line_count
								, numcoords, this_set->ncols
								, buffer
							);
						}
					}
					AddPoint_discard= False;
					AddPoint( &this_set, &spot, &Spot, numcoords, DataBuf, column, filename, sub_div, line_count, &flpos, buffer, &ReadData_proc );
					if( debugFlag && AddPoint_discard> 0 ){
					  static long N= 0;
						fprintf( StdErr, "Discarded point #%ld (#%d,%g,%g,%g) (%s:line %d.%d) next: #%d\n",
							N++, Spot, DataBuf[0], DataBuf[1], DataBuf[2], filename, sub_div, line_count, spot
						);
					}
					read_data= True;

					if( DumpFile ){
					  int i;
						if( DumpBinary ){
							  /* We use here this_set->ncols instead of numcoords, because numcoords
							   \ can get to be smaller than ncols, but never larger. When it happens
							   \ to be smaller (e.g. 5) on the first output after a BINARYDATA command, and returns
							   \ to ncols (e.g. 6) afterwards the read routines will expect 5 columns, and fail
							   \ (= crash, likely) when attempting to read the first or second field "back to"
							   \ 6 columns wide. Such are the joys of binary IO - difficult to make errorproof..
							   */
							if( this_set->ncols> ((BinaryFieldSize)? BinaryDump.columns : 0) ){
								AllocBinaryFields( this_set->ncols, "ReadData()" );
							}
							else{
								BinaryDump.columns= this_set->ncols;
							}
							if( !DF_bin_active ){
								fputs( "*BINARYDATA*\n", stdout );
								DF_bin_active= True;
							}
							for( i= 0; i< BinaryDump.columns; i++ ){
								BinaryDump.data[i]= DataBuf[i];
							}
							BinaryDumpData( stdout, False );
						}
						else{
							if( DF_bin_active ){
								BinaryTerminate(stdout);
								fputc( '\n', stdout);
								DF_bin_active= False;
							}
							fprintf( stdout, "%s", d2str( DataBuf[0], d3str_format, NULL) );
							for( i= 1; i< this_set->ncols; i++ ){
								fprintf( stdout, "\t%s", d2str( DataBuf[i], d3str_format, NULL) );
							}
							fputc( '\n', stdout );
						}
						LineDumped= True;
					}

				}
			}

		}

next_line:;
		  /* !!! 981210 Note very well that commands may have been executed that have changed (a) global variable(s),
		   \ like a -maxsets command line argument parsed with *ARGUMENTS*. If local variables are used to
		   \ access elements of such global variables, like this_set to address a set in the AllSets array,
		   \ be sure to update them, preferrably directly after the parsing!!!
		   \ I spent an afternoon today searching for why the **SDF@#$#@$ I got a crash when loading a large (400+ 2 points
		   \ sets) file. It crashed trying to xfree a setName previously allocated with XGstrdup(), making me suspect
		   \ writing out of stringbounds somewhere. So I set a debugger hack that allocated 8bytes more in XGstrdup, and then
		   \ the crash was when xfreeing inFileNames[7] which should be NULL, but contained 0x1 ... I had the additional luck
		   \ that for once a watchpoint worked, which broke at the line updating this_set->fileNumber below.
		   \ I finally found that the 3rd statement was a *ARGUMENTS* containing
		   \ -maxsets 1024, which prompted an expansion, and the result was that this_set==inFileNames (by some bizarre piece
		   \ of luck this was perfectly always the case).
		   */
		if( DumpFile && !LineDumped ){
			if( DF_bin_active ){
				BinaryTerminate(stdout);
				fputc( '\n', stdout);
				DF_bin_active= False;
			}
			if( strncasecmp( buffer, "*read_file*", 11)== 0 && DumpIncluded ){
				fputc( '#', stdout );
			}
			fputs( buffer, stdout );
			if( buffer[strlen(buffer)-1]!= '\n' ){
				fputc( '\n', stdout );
			}
		}

		fileNumber= this_set->fileNumber= filenr+ 1+ file_splits;
		if( plot_only_file && filenr+1+file_splits != plot_only_file ){
			this_set->draw_set= 0;
			if( debugFlag && this_set->numPoints<= 0 ){
				fprintf( StdErr, "ReadData(): set #%d in file #%d NOT shown initially\n",
					setNumber, filenr+1
				);
			}
		}
		prev_filenr= filenr;

		if( ActiveWin && ActiveWin->halt ){
			ascanf_exit= True;
		}

#ifndef __MACH__
		if( ReadPipe && ReadPipe_fp ){
		  /* identical check as that performed in the xgraph main loop...	*/
		  struct pollfd check[1];
		  int r;
			check[0].fd= fileno(ReadPipe_fp);
			check[0].events= POLLIN|POLLPRI|POLLRDNORM|POLLRDBAND;
			check[0].revents= 0;
			if( (r= poll( check, 1, 100 ))<= 0 || CheckORMask(check[0].revents, POLLHUP|POLLERR|POLLNVAL) ){
				if( feof(stream) || ferror(stream) ){
					doRead= False;
				}
			}
		}
#endif
    }

	RESETATTRS();

	  /* Check for read_file commandlines still waiting to be stored definetely	*/
	if( read_file_buf && this_set->numPoints ){
		if( this_set->read_file ){
			this_set->read_file= concat2( this_set->read_file, "\n\n", read_file_buf, NULL);
			xfree( read_file_buf );
		}
		else{
			this_set->read_file= read_file_buf;
			read_file_buf= NULL;
		}
		this_set->read_file_point= Spot;
	}
	do{
		if( read_file_buf ){
			if( Read_File_Buf ){
				Read_File_Buf= concat2( Read_File_Buf, "\n\n", read_file_buf, NULL);
				xfree( read_file_buf );
			}
			else{
				Read_File_Buf= read_file_buf;
				read_file_buf= NULL;
			}
		}
		if( this_set->read_file && this_set->numPoints<= 0){
			read_file_buf= this_set->read_file;
			this_set->read_file= NULL;
		}
	}
	while( read_file_buf );

	if( InitWindow && IWname ){
		XStoreName( disp, InitWindow->window, IWname );
		if( SetIconName ){
			XSetIconName( disp, InitWindow->window, IWname );
		}
		XFree(IWname);
		if( InitWindow->mapped && !RemoteConnection ){
			XFlush(disp);
		}
		IWname= NULL;
	}

    if( line_count <= 0 && sub_div== 0 && ReadData_commands<= 0 ){
		if( ReadData_terpri ){
			fputs( "\n", StdErr );
			ReadData_terpri= False;
		}
		fprintf(StdErr, "No data found\n");
		if( debugFlag && Read_File_Hist ){
			fprintf( StdErr, "Included files: %s\n", Read_File_Hist );
		}
		xfree( Read_File_Hist );
		RD_RETURN(-1);
    }
	if( spot> maxSize ){
		maxSize= spot;
	}

	  /* 950622: a new set is no longer in made after each invocation of ReadData	*/
	if( ReadData_terpri ){
		fputs( "\n", StdErr );
		ReadData_terpri= False;
	}
	GCA();
	if( (debugFlag || scriptVerbose) && Read_File_Hist ){
		fprintf( StdErr, "Included files: %s\n", Read_File_Hist );
	}
	xfree( Read_File_Hist );
	if( (debugFlag || scriptVerbose) ){
		if( !fseek( stream, 0, SEEK_END ) ){
			flpos.last_fpos= 0;
			flpos.last_line_count= 0;
			line_count= 0;
			line_count= UpdateLineCount( stream, &flpos, line_count );
		}
		fprintf( StdErr, "Finished file \"%s\": %ld bytes for %d lines\n", the_file, ftell(stream), line_count );
	}

	RD_RETURN( maxSize );
}

int IOImport_Data_Direct( const char *libname, char *fname, DataSet *this_set )
{ ReadData_States state;
  int spot, Spot, ret;
  struct FileLinePos flpos;
	// import appends to the specified set:
	Spot= spot= this_set->numPoints;
	state.spot= &spot;
	state.Spot= &Spot;
	state.sub_div= 0;
	state.line_count= 0;
	  /* Not supposed to be changed by the handler, unless rfp==stream! */
	flpos.stream= NullDevice;
	flpos.last_fpos= flpos.last_line_count= 0;
	state.flpos= &flpos;
	state.ReadData_proc= &ReadData_proc;
	ret= doIO_Import( libname, fname, NULL, &state, this_set );
	switch( ret ){
		case 2:
			fprintf( StdErr, "IOImport_Data(%s): can't open file for import using '%s' (%s)\n",
				fname, libname, serror()
			);
			fflush( StdErr );
			break;
		case 1:
			fprintf( StdErr, "IOImport_Data(%s): can't load IO library '%s' (%s) or not a proper IO library\n",
				fname, libname, serror()
			);
			fflush( StdErr );
			break;
	}
	return( ret );
}

int IOImport_Data( const char *libname, char *fname )
{ char *tmpName= NULL, pidnr[128];
  FILE *fp= NULL;
  int ret= 0;
	snprintf( pidnr, sizeof(pidnr)/sizeof(char)-1, "%d", getpid() );
	if( (tmpName= concat( "/tmp/IOImp-", basename(fname), ".", pidnr, ".xg", NULL))
	    && (fp= fopen(tmpName, "w"))
	){
		fprintf( fp, "*DM_IO_IMPORT* %s %s\n\n", libname, fname );
		fclose(fp);
		if( (fp= fopen(tmpName, "r")) ){
		  LocalWin *aw= ActiveWin;
			unlink(tmpName);
// 			IncludeFile( ActiveWin, fp, tmpName, False, NULL );
			IncludeFile( ActiveWin, fp, fname, False, NULL );
			fclose(fp);
			ret= 0;
			ActiveWin= aw;
		}
		else{
			unlink(tmpName);
			ret= 2;
		}
	}
	else{
		if( tmpName ){
			fprintf( StdErr, "Cannot create \"%s\" (%s)\n", tmpName, serror() );
		}
		ret= 1;
	}
	xfree(tmpName);
	return( ret );
}

