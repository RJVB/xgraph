#ifndef _DATA_SET_H

#include "ascanf.h"

#define ADVANCED_STATS	2
#define RUNSPLIT

typedef struct SetProcess{
	char *set_process;
	int set_process_len;
	struct Compiled_Form *C_set_process;
	char *description, separator;
} SetProcess;

typedef struct data_point{
	double x, y, errr;
} DataPoint;

typedef struct LegendLine{
	int low_y, high_y;
	int highlight, pixvalue;
	Pixel pixelValue;
	char *pixelCName;
} LegendLine;

#define VMARK_NONE	0
#define VMARK_ON	1
#define VMARK_FULL	(1<<1)
#define VMARK_RAW	(1<<2)

#define MAX_VECTYPES	5
#define MAX_VECPARS		2

#define STATICSETDATA \
	struct LocalWin *s_bounds_wi, *last_processed_wi, *processing_wi; \
	int filename_shown; \
	int set_nr, new_file, show_legend, show_llines, draw_set, fileNumber; \
	int numPoints; \
	int set_link, set_linked, links, dumped_set; \
	int highlight, raw_display, floating, points_added; \
	int linestyle, elinestyle, pixvalue, markstyle, markFlag, noLines, \
		pixelMarks, barFlag, polarFlag, overwrite_marks, arrows, \
		valueMarks, skipOnce;\
	double lineWidth, elineWidth, markSize; \
	Pixel pixelValue; \
	double sarrow_orn, earrow_orn; \
	Boolean sarrow_orn_set, earrow_orn_set; \
	double radix, radix_offset; \
	int error_point; \
	int NumObs, plot_interval, adorn_interval; \
	double Xscale, Yscale, DYscale; \
	Range rawY, rawE, ripeY, ripeE; \
	int sx_min, sy_min, sx_max, sy_max, s_bounds_set;\
	int numErrors, has_error, use_error; \
	int error_type;\
	double vectorLength, vectorPars[MAX_VECPARS];\
	int vectorType;\
	int first_error, last_error;\
	int barBase_set, barWidth_set, barType, ebarWidth_set, current_bW_set, current_ebW_set; \
	double barBase, barBaseY, barWidth, displaced_x, displaced_y, ebarWidth;\
	double current_barWidth, current_ebarWidth;\
	Boolean init_pass

  /* the UDataSet is to allow copying of one set into another, without
   \ messing with dynamically allocated information.
   \ 980819: SetProcess process used to be part of STATICSETDATA.
   */
typedef struct data_set_static{
	STATICSETDATA;
} UDataSet;

  /* The "full" DataSet definition starts with the same fields as the UDataSet
   \ structure, and adds the remaining fields that cannot be copied directly
   \ 980212 Note: as of this writing, no use is made of this possibility! I might
   \ eventually add functionality that will split an existing set into 2 runtime. Right
   \ now, this can be done, but the split will actually occur when reloading an exported
   \ XGraph dump (using *SPLIT* commands).
   */
typedef struct DataSet {

	STATICSETDATA;

	DataPoint lowest_y, highest_y;
	double data[2][ASCANF_DATA_COLUMNS];
	int internal_average, averagedPoints;

	char *setName, *appendName;		/* Name of set     */
	char *nameBuf;
	char *titleText;
	short titleChanged, free_nameBuf;
	char *XUnits, *YUnits;
	struct LabelsList *ColumnLabels;
	char *fileName;
	char *average_set;
	int allocSize;		/* Allocated size  */
	signed char *discardpoint;
	int ncols, xcol, ycol, ecol, lcol;
	double **columns, av_error_r;	/* original values        */
	double *xvec;		/* X values        */
	double *yvec;		/* Y values        */
	double *errvec, av_error;
	double *lvec;			/* length co-ordinate for vector mode, horizontal error bar, etc. */
	double *ldxvec, *hdxvec;	/* x-coordinates of error-bars (for polarplots, vector mode)	*/
	double *ldyvec;		/* low coordinate of error-bar	*/
	double *hdyvec;		/* high coordinate of error-bar	*/
#if ADVANCED_STATS==1
	int *N;			/* Number of observations for this (x,y,e)	*/
#elif ADVANCED_STATS==2
	int Ncol;
#endif
#ifdef RUNSPLIT
	signed char *splithere;
#endif
	SetProcess process;
	char *read_file;
	int read_file_point;
	unsigned long mem_alloced;
	int numAssociations, allocAssociations;
	double *Associations;
	char *pixelCName;
	LegendLine hl_info;
	char *set_info;
	// 20080711: an std::list list of ascanf LinkedArrays that refer to this set:
	void *LinkedArrays;
	// 20081126: should allow a *READ_FILE* after a *PROPERTIES* without resetting to defaults:
	short propsSet;
} DataSet;

struct LocalWin;
extern int _DiscardedPoint( struct LocalWin *wi, DataSet *set, int idx);
#define DiscardedPoint(wi,set,idx)	((wi)? _DiscardedPoint(wi,set,idx) : \
	((set && (set)->discardpoint)? (set)->discardpoint[idx] : 0))
#define WinDiscardedPoint(wi,set,idx)	((wi && wi->discardpoint && wi->discardpoint[set->set_nr])?\
	(int) wi->discardpoint[set->set_nr][idx] : 0)

#ifdef RUNSPLIT
#	define SplitHere(set,idx)	((set && (set)->splithere)? (set)->splithere[idx] : 0)
#else
#	define SplitHere(set,idx)	0
#endif

#define XVAL(set,idx)	(set)->columns[(set)->xcol][(idx)]
#define YVAL(set,idx)	(set)->columns[(set)->ycol][(idx)]
#define ERROR(set,idx)	(set)->columns[(set)->ecol][(idx)]
#define EVAL(set,idx)	ERROR(set,idx)
#define OVAL(set,idx)	ERROR(set,idx)
#define IVAL(set,idx)	ERROR(set,idx)
#define VVAL(set,idx)	(set)->columns[(set)->lcol][(idx)]
#define xval	columns[this_set->xcol]
#define yval	columns[this_set->ycol]
#define error	columns[this_set->ecol]
#define lval	columns[this_set->lcol]

#if ADVANCED_STATS == 1
#	define NVAL(set,idx)	(set)->N[(int)(idx)]
#elif ADVANCED_STATS == 2
#	define NVAL(set,idx)	(((set)->Ncol>=0)? (int)((set)->columns[(set)->Ncol][(int)(idx)]) : (set)->NumObs)
#endif

#define _DATA_SET_H
#endif
