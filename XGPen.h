#ifndef _XGPEN_H
#define _XGPEN_H

/* 20000505: start of drawing pens implementation: definition of structures, and
 \           incorporation in the LocalWin structure of the probably necessary
 \           variables. xgraph.h does not (yet) depend on XGPen.h, and probably should
 \           never come to depend on it. It only includes it, and may only refer to pointers
 \           to datatypes defined here.
 \           The rationale behind this exercise/functionality is that it should be possible
 \           to draw custom-graphs from the (processed) data available.
 */

/* Definitions for the pens available to the user through ascanf() drawing functions.
 \ The syntax is inspired on the PostScript graphics conventions. A pen is up or down,
 \ can be put somewhere, and can be drawn to some location. Furthermore, it is possible
 \ at each location to put a marker, to change colour or other line attributes. It is
 \ also possible to draw a rect or an ellipse at a point location.
 \ Each window has its own sets of pens, allocated as needed and stored as a linked list.
 \ Each pen stores the points to be drawn, and per point whether a marker is requested.
 \ The positions are stored in an array that is expanded as necessary, and always contains
 \ at most <positions> points, and currently <current_pos> (the next, empty position) points.
 \ The points/segments are drawn either before, or after the datasets' lines, just
 \ like the datasets' markers. Pens should be drawn also in QuickMode, as that mode
 \ serves to draw the (latest) results from the currently defined processing (which
 \ can contain pen-drawing commands), without actually re-evaluating this processing.
 \ 20000831: Clever.. :) Pens are most likely driven through the very processing code
 \ that is circumvented in quickmode. Meaning, user drives the pen from, say, a *DATA_BEFORE*
 \ statement - and how would these co-ordinates be processed? Without (guaranteed) deadlocks
 \ etc? Nope: no distinction xval/xvec for XGPens...
 \ NB: it would seem nice to be able to change colour, linewidth, etc. on the fly, while
 \ drawing with a certain pen.
 */

/* The types of graphics operations that a pen can perform. Currently, these would
 \ seem enough.. :)
 */
typedef enum XGPenCommand{
	XGPenMoveTo=1, XGPenLineTo= 2,
	XGPenDrawRect= 3, XGPenFillRect= 4, XGPenDrawCRect= 5, XGPenFillCRect= 6,
	XGPenDrawEllps= 7, XGPenDrawPoly= 9, XGPenFillPoly= 10, XGPenText= 11, XGPenTextBox= 12,
} XGPenCommand;

/* Flags that indicate what items are set in a stub XGPenPosition structure
 \ passed to a convenience routine:
 */
typedef enum XGPenOperation {
	XGP_markType = (1<<0),
	XGP_position = (1<<1),
	XGP_linestyle = (1<<2),
	XGP_lineWidth = (1<<2),
	XGP_colour = (1<<3),
	XGP_flcolour = (1<<4),
	XGP_markSize = (1<<5),
	XGP_markFlag = (1<<6),
	XGP_text = (1<<7),
	XGP_textJust = (1<<8),
	XGP_textFntNr = (1<<9),
	XGP_textOutside = (1<<10),
	XGP_noClip = (1<<11)
} XGPenOperation;

typedef struct XGPenPosition{
	int OK;
	XGPenCommand command;
	double x, y, orn, w, h;
	double tx, ty;
	double *polyX, *polyY;
	int polyN_1, polyMax;
	short sx, sy;
	char *text;
	short textJust, textJust2[2], textSet, textFntNr, textFntHeight;
	  /* The attributes are repeated for every position. When
	   \ a change is requested, the current position is updated, and all subsequent
	   \ positions use these new values (as they always do).
	   \ Colour names are stored only in the current pen position; the subsequent
	   \ positions copy only the pixvalue and pixelValue. This means that these
	   \ attributes can *not* be copied blindly, but may be copied only when
	   \ pixelCName==0 or flpixelCName==0.
	   \ Copyable attributes are stored in the attr structure.
	   */
	struct attributes{
		short linestyle, text_outside;
		short markFlag, markType;
		double lineWidth, markSize;
		  /* whether to (try) to clip or just draw it as it comes: */
		short noclip;
	} attr;
	char *pixelCName, *flpixelCName;
	struct pixs{
		int pixvalue;
		Pixel pixelValue;
	} colour, flcolour;
	  /* whether a filled rect should have its outline drawn in the fill colour (= no frame...)	*/
	short not_outlined;
	  /* For boxed text, we need an additional PenPosition field that will serve to draw the box.
	   \ This field will be allocated and initialised by AddPenPosition.
	   */
	struct XGPenPosition *TextBox;
	struct ascanf_Function *cfont_var;
/* #ifdef DEBUG	*/
	char *expr, *caller;
	int level;
/* #endif	*/
#ifdef PENSHAVEARROWS
	  /* 20020813: arrows on pens are not going to be easy to implement, giving the mixed nature of
	   \ what can be in an XGPen structure. Contrary to what I thought before, the arrow flags should
	   \ be associated with a PenPosition, not with a Pen -- because one may want to draw multiple lines
	   \ with varying arrows in a single pen, without having the obligation to call PenDrawNow[] to ensure
	   \ that everything is drawn correctly. But even then, some bookkeeping will be needed to make sure
	   \ the correct begin- and end-point of the desired segment are found. For now, I think this is not
	   \ worth the effort. Any point can be marked by e.g. a circle (and a segment by e.g. a circle and a
	   \ rectangle); if necessary, this marking can be improved in Adobe Illustrator...
	   */
	int arrows;
	double sarrow_orn, earrow_orn;
#endif
	XGPenOperation operation;
} XGPenPosition;

typedef struct XGPen{
	struct LocalWin *wi;
	unsigned int pen_nr;
	XGPenPosition *position;
	unsigned int current_pos, positions, drawn, allocSize;
	// RJVB 20081202: any reason those fields were int and not short (or even uchar)?
	short highlight, highlight_text, skip, floating, set_link;
	Pixel hlpixelValue;
	char *hlpixelCName;
	int overwrite_pen, before_set, after_set;
	char *penName, *pen_info;
	struct XGPen *next;
} XGPen;

// in most cases, drawing a pen can be skipped regardless whether pen->skip or pen->set_link requires it
#define PENSKIP(wi,pen)	( (pen)->skip || ((wi) && ((pen)->set_link>=0) && !draw_set((wi),(pen)->set_link)) )
// in some cases, we want to be able to ignore pen->skip but not pen->set_link:
#define PENLINKDRAWN(wi,pen)	( ((wi) && ((pen)->set_link>=0) && draw_set((wi),(pen)->set_link)) )

#endif
