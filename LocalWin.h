#ifndef _LOCALWIN_H
/* LocalWin definition plus a certain number of related type definitions */

typedef enum AxisName { X_axis= 1, Y_axis, I_axis, AxisNames } AxisName;
typedef enum CutAction { _spliceAction=0x01, _deleteAction=0x02, CutActions } CutAction;

typedef enum FittingAxis { fitNone=0, fitDo=True, fitXAxis, fitYAxis, fitBothAxes } FittingAxis;

typedef struct local_boundaries {
    double _loX, _lopX, _hinY, _loY, _lopY, _hiX, _hiY;	/* Local bounding box of window         */
    int pure;
} Local_Boundaries;

typedef struct LocalWinGeo {
	Local_Boundaries bounds;
	Local_Boundaries pure_bounds;
	Local_Boundaries aspect_base_bounds;
	int user_coordinates;
	  /* 2 lines to define the axes-range if no border is to be drawn (!bbFlag):	*/
	int nobb_range_X, nobb_range_Y, nobb_coordinates;
	double nobb_loX, nobb_loY, nobb_hiX, nobb_hiY;
	double padding;
	double low_angle, high_angle;
	AxisName polar_axis;
    int _XOrgX, _XOrgY;		/* Origin of bounding box on screen     */
    int _XOppX, _XOppY;		/* Other point defining bounding box    */
    double _UsrOrgX, _UsrOrgY;	/* Origin of bounding box in user space */
    double _UsrOppX, _UsrOppY;	/* Other point of bounding box          */
    double _WinOrgX, _WinOrgY;	/* Origin of window's lower-left corner in user space */
    double _WinOppX, _WinOppY;	/* Other point of window          */
    double R_UsrOrgX, R_UsrOrgY;	/* Idem, excluding only TRANSFORM_[XY]	*/
    double R_UsrOppX, R_UsrOppY;
    double _XUnitsPerPixel;	/* X Axis scale factor	*/
    double _YUnitsPerPixel;	/* Y Axis scale factor	*/
    double R_XUnitsPerPixel;	/* Idem, excluding only TRANSFORM_[XY]	*/
    double R_YUnitsPerPixel;
} LocalWinGeo;

typedef enum { XPlaced, YPlaced, LegendPlaced, IntensityLegendPlaced } PlaceTypes;

typedef struct PlaceUndoBuf{
	PlaceTypes which;
	Boolean valid;
	int placed, trans;
	double x,y;
	int sx, sy;
} PlaceUndoBuf;

typedef struct AxisValues{
	int last_index, N;
	double *array;
	char *labelled;
	AxisName axis;
} AxisValues;

typedef struct DataWindow{
	int apply, old_user_coordinates;
	double llX, urX;
	double llY, urY;
} DataWindow;

typedef struct FitHistory{
	double x, y;
} FitHistory;
#define AFTFITHIST	15

typedef struct LocalWin{
	Window window;							/* X-id of the window	*/
	XdbeBackBuffer XDBE_buffer;
	int XDBE_init;
	struct LocalWindows *WindowList_Entry;
	Visual *visual;
	Colormap cmap;
	XWMHints *wmhints;
	LocalWinGeo win_geo;					/* Geometry of the window	*/
	struct axis_stuff{
		int __XLabelLength, __YLabelLength,	/* Used by DrawGridAndAxis()	*/
			XLabelLength, YLabelLength,		/* Used by DrawGridAndAxis()	*/
			DoIt, _polarFlag;
#define LOG_ZERO_LOW	-1
#define LOG_ZERO_HIGH	1
#define LOG_ZERO_INSIDE	0
		int log_zero_x_spot, log_zero_y_spot;
/* 20060907 RJVB
 \ STUPID Apple gcc 4.0 does a #define I _Complex_I in complex.h
 \ which seems really dumb to do
 \ So we undefine it here...
 \ 20111031: need to undefine it also everywhere after including vecLib.h !
 */
#ifdef I
#	undef I
#endif
		AxisValues rawX, rawY, X, Y, I;
	} axis_stuff;
	double _legend_ulx, _legend_uly,		/* coordinates of the legend-box	*/
		xname_x, xname_y, tr_xname_x, tr_xname_y,
		yname_x, yname_y, tr_yname_x, tr_yname_y;
	int legend_ulx, legend_uly, legend_lry,
		legend_lrx, legend_frx, legend_type;
	int legend_placed, legend_always_visible,	/* flag for user-placed legend-box	*/
		xname_placed, yname_placed, yname_vertical;
	int legend_trans, xname_trans,			/* do_transform these coordinates?	*/
		yname_trans;
	PlaceUndoBuf PlaceUndo;
	char XUnits[MAXBUFSIZE+1], YUnits[MAXBUFSIZE+1];
	char tr_XUnits[MAXBUFSIZE+1], tr_YUnits[MAXBUFSIZE+1];
	LabelsList *ColumnLabels;
	int title_uly, title_lry;				/* Bounding Y-coordinates of the title-region	*/
	xgOut dev_info;		/* Device information                   */
	XRectangle XGClipRegion[16];
	int ClipCounter;
	Window close, hardcopy, settings, info, label;	/* Buttons for close, hardcopy, settings and info     */
	xtb_frame cl_frame, hd_frame,			/* Frames for idem	*/
		settings_frame, info_frame, label_frame, ssht_frame, label_IFrame, YAv_Sort_frame;
	xtb_frame *SD_Dialog, *HO_Dialog;		/* point to Dialogues for this window	*/
	struct hard_dev *hard_devices;
	int current_device;
	int pid, parent_number, pwindow_number,
		window_number, childs, halt, redrawn,
		data_sync_draw, osync_val,
		data_silent_process, osilent_val,
		dont_clear, redraw, clipped, delete_it,			/* drawing-states	*/
		pw_placing							/* How to place the hardcopy and settings dialogues	*/,
		ctr_A, add_label, mapped, silenced, delayed;
	int animate, animating, drawing, redraw_val, event_level;			/* some more states	*/
	char *title_template;					/* printf() template for the window-title	*/
	short *draw_set,							/* which sets to draw	*/
		*group,								/* grouping as indicated by new_file	*/
		*fileNumber,						/* the file-set to which a set belongs	*/
		*mark_set,
		*new_file,
		numGroups, numFiles, num2Draw, numDrawn, sets_reordered, first_drawn, last_drawn;
	int *numVisible, *xcol, *ycol, *ecol, *lcol, *error_type;
	int discard_invisible;
	signed char **discardpoint, **pointVisible;
	int graph_titles_length,				/* total length of titles	*/
		legend_length;						/* total length of text in legend-box	*/
	char *graph_titles;						/* the titles which are to be shown	*/
	double Xscale, _Xscale, Yscale,			/* user-defined scale factors	*/
		Xbias_thres, Ybias_thres,
		Xincr_factor, Yincr_factor,
		ValCat_X_incr, ValCat_Y_incr;
	short *plot_only_set;						/* for cycling through the sets	*/
	int plot_only_set_len,
		plot_only_set0,
		plot_only_group,
		plot_only_file,						/* for cycling through the file-sets	*/
		no_legend,
		no_legend_box,
		no_intensity_legend,
		no_title,
		filename_in_legend,
		labels_in_legend,
		axisFlag, bbFlag,
		htickFlag, vtickFlag,
		zeroFlag,
		use_errors,
		triangleFlag,
		error_region,
		vectorFlag,
		intense_Flag,
		polarFlag,
		absYFlag,
		logXFlag, logYFlag,
		sqrtXFlag, sqrtYFlag,
		exact_X_axis, exact_Y_axis,
		ValCat_X_axis, ValCat_X_levels, ValCat_X_grid, ValCat_I_axis, ValCat_Y_axis;
	ValCategory *ValCat_X, *ValCat_Y;
	CustomFont *ValCat_XFont, *ValCat_YFont;
	double powXFlag, powYFlag, powAFlag, radix, radix_offset;
	char log_zero_sym_x[MAXBUFSIZE], log_zero_sym_y[MAXBUFSIZE];	/* symbol to indicate log_zero	*/
	int lz_sym_x, lz_sym_y;											/* log_zero symbols set?	*/
	int log_zero_x_mFlag, log_zero_y_mFlag;
	double log10_zero_x, log10_zero_y, 
		log_zero_x, log_zero_y,										/* substitute 0.0 for these values when using log axis	*/
		_log_zero_x, _log_zero_y;									/* transformed values	*/
	int init_pass, FitOnce, fit_xbounds, fit_ybounds, fit_after_draw;
	double fit_after_precision;
	int aspect, x_symmetric, y_symmetric;
	double aspect_ratio;
	int process_bounds, raw_display, raw_once, raw_val, use_transformed, transform_axes, overwrite_AxGrid, overwrite_legend;
	int print_orientation, printed, dump_average_values, DumpProcessed, DumpBinary, DumpAsAscanf;
	int ps_xpos, ps_ypos;
	double ps_scale, ps_l_offset, ps_b_offset;
	long draw_count, draw_errors;
	struct Time_Struct draw_timer;
	Process process;
	Transform transform;
	double **curve_len, **error_len;
#ifdef TR_CURVE_LEN
	double **tr_curve_len;
#endif
	DataSet *processing_set;									/* points to the set currently processed (!= drawn!) in this wi */

	  /* Some statbins: first those for autoscaling etc.:	*/
	SimpleStats SS_X, SS_Y, SS_LY, SS_HY, SS_I,
		  /* These are the (X,Y,E) co-ordinates statbins:	*/
		SS_Xval, SS_Yval, SS_E;
	SimpleStats *set_X, *set_Y, *set_E, *set_V, *set_tr_X, *set_tr_Y, *set_tr_E, *set_tr_V;
	SimpleAngleStats SAS_O, SAS_slope, SAS_scrslope,
		*set_O, *set_tr_O;
	int use_average_error;
	int show_overlap;
	char *overlap_buf, *overlap2_buf;
	LegendLine *legend_line;
	int AlwaysDrawHighlighted;
	int ulabels, no_ulabels;
	UserLabel *ulabel;
	CutAction cutAction;
	FittingAxis fitting;
	int filtering, checking;
	char *next_include_file;
	XGStringList *init_exprs, *next_startup_exprs, *Dump_commands, *DumpProcessed_commands;
	int new_init_exprs;
	char *version_list;
	Cursor_Cross curs_cross;
	  /* For incorporating bar-graphics in auto-scaling.	*/
	int BarDimensionsSet, eBarDimensionsSet;
	double MinBarX, MaxBarX;
	double eMinBarX, eMaxBarX,
		bar_legend_dimension_weight[3];
	LegendInfo IntensityLegend;
	int show_all_ValCat_I;
	ValCategory *ValCat_I;
	CustomFont *ValCat_IFont;
	XGPen *pen_list, *current_pen;
	int numPens, no_pens;
	TextRelated textrel;
	DataWindow datawin;
	FitHistory aftfit_history[AFTFITHIST];
	int debugFlag;
} LocalWin;

typedef struct LocalWindows{
	LocalWin *wi;
	Window under, above;
	xtb_frame *frame;
	struct LocalWindows *prev, *next;
} LocalWindows;
extern LocalWindows *WindowList, *WindowListTail;


#	define _LOCALWIN_H
#endif
