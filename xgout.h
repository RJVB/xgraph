/*
 * Output Device Information
 *
 * This file contains definitions for output device interfaces
 * to the graphing program xgraph.
 */

#ifndef _XGOUT_H
#define _XGOUT_H

#ifdef __APPLE_CC__
#	undef pixel
#endif

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include "ux11/ux11.h"

#define D_COLOR   	0x01

/* Text justifications */
#define T_VERTICAL	-1
#define T_CENTER	0
#define T_LEFT		1
#define T_UPPERLEFT 2
#define T_TOP		3
#define T_UPPERRIGHT	4
#define T_RIGHT   	5
#define T_LOWERRIGHT	6
#define T_BOTTOM	7
#define T_LOWERLEFT 8

/* Text styles */
#define T_AXIS		0
#define T_LABEL   	2
#define T_TITLE   	1
#define T_LEGEND	3
#define T_MARK		4

/* Line Styles */
#define L_AXIS		0
#define L_ZERO		1
#define L_VAR		2
#define L_POLAR 3

/* Marker Styles */
#define P_PIXEL   	0
#define P_DOT		1
#define P_MARK		2

#ifndef _MACRO_H
	typedef void (*void_method)();
#endif

typedef struct XGFontStruct{
	XFontStruct *font;
	char use[64], name[128];
} XGFontStruct;

typedef struct CustomFont{
	XGFontStruct XFont, X_greekFont;
	char *alt_XFontName, *PSFont;
	double PSPointSize;
	short is_alt_XFont, PSreencode;
	unsigned long PSdrawcnt;
} CustomFont;

typedef struct xsegment2{
	short x1, x2,			/* x1 and x2 of XSegment	*/
		y1l, y1h,			/* y error in y1	of XSegment */
		y2l, y2h,			/* y error in y2	of XSegment */
		x1h, x2h;			/* when error-bar is not vertical	*/
	short ok, ok2;			/* is it a valid errorbar?	*/
	short X1, Y1, X2, Y2;
	short mark_inside1, mark_inside2;	/* concerns datapoints, not errorflags!	*/
	short e_mark_inside1, e_mark_inside2;
	short mark1, mark2,
		use_first;
	int pnt_nr1, pnt_nr2;
	double rx1, ry1, rx2, ry2;
} XSegment_error;


/* Output device information returned by initialization routine */

typedef struct xg_out {
	int dev_flags;		/* Device characteristic flags           */
	int area_x, area_y, area_w, area_h;   	/* Width and height in pixels            */
	int old_area_w, old_area_h;   	/* Width and height in pixels            */
	int resized;
	int bdr_pad;		/* Padding from border                   */
	int axis_pad;		/* Extra space around axis labels        */
	int tick_len;		/* Length of tick mark on axis           */
	int errortick_len;	/* Horizontal width (not penwidth) of errorbar/triangle */
	int legend_pad;   	/* Top of legend text to legend line     */
	int legend_width;		/* Width of big character of label font   */
	int legend_height;		/* Height of big character of label font  */
	int label_width;		/* Width of big character of label font   */
	int label_height;		/* Height of big character of label font  */
	int axis_width;   	/* Width of big character of axis font   */
	int axis_height;		/* Height of big character of axis font  */
	int xname_vshift;		/* vertical shift of xlabel text    */
	int title_width;		/* Width of big character of title font  */
	int title_height;		/* Height of big character of title font */
	int max_segs;		/* Maximum number of segments in group   */

	   /* factors applied to linewidths: */
	double polar_width_factor, axis_width_factor, zero_width_factor, var_width_factor;
	double mark_size_factor;

	void (*xg_clear)();   	/* Clears the drawing regio (for display-graphics)  */
	void (*xg_rect)(char *, XRectangle *, double, int, int, int, Pixel, int, int, Pixel, void*);		/* Draws a (filled) rectangle at a location              */
	void (*xg_text)(char *, int, int, char*, int, int, struct CustomFont *cfont);		/* Draws text at a location              */
	void (*xg_seg)(char *, int, XSegment *, double, int, int, int, Pixel, void *);		/* Draws a series of segments            */
	void (*xg_dot)(char *, int, int, int, int, int, Pixel, int, void *);		/* Draws a dot or marker at a location   */
	void (*xg_end)(void *);		/* Stops the drawing sequence            */
	void (*xg_arc)(char *, int, int, int, int, double, double, double, int, int, int, Pixel);
	void (*xg_polyg)(char *, XPoint *, int N, double, int, int, int, Pixel, int, int, Pixel, void*);
	int (*xg_silent)();   	/* Switches output on/off                */
	int (*xg_CustomFont_width)( struct CustomFont *cf );
	int (*xg_CustomFont_height)( struct CustomFont *cf );

	void *user_state;		/* User supplied data                    */
	int user_ssize;   		/* size of latter   */
} xgOut;

#define ERRBUFSIZE	2048

typedef struct Extremes{
	double min, max;
} Extremes;

typedef struct BinaryField{
	short columns;
	double *data;
	  /* a field that always points to *data and should *never* be treated as independent mem to be freed/realloced! */
	float *data4;
	unsigned short *data2;
	unsigned char *data1;
} BinaryField;

extern int BinaryFieldSize;
extern BinaryField BinaryDump, BinaryTerminator;
extern int DumpBinary;

#endif
