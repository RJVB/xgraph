/*
 * xtb - a mini-toolbox for X11
 *
 * David Harrison
 * University of California, Berkeley
 * 1988
 */

#ifndef _XTB_
#define _XTB_

#include "stdio.h"
#include <X11/Intrinsic.h>

#ifdef XGRAPH
#	include "../va_dcl.h"
#else

#	ifdef _GRAFTOOL_
#		include "local/va_dcl.h"
#	else
#		include "va_dcl.h"
#	endif

#	ifndef _XGRAPH_H
	typedef struct XGFontStruct{
		XFontStruct *font;
		char name[128];
	} XGFontStruct;



extern Display *t_disp;
extern int t_screen;

extern XGFontStruct dialogFont, dialog_greekFont, title_greekFont, titleFont, cursorFont;

#	endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern Pixel xtb_back_pix, xtb_white_pix,
	xtb_norm_pix, xtb_black_pix,
	xtb_light_pix, xtb_Lgray_pix,
	xtb_middle_pix, xtb_Mgray_pix;

/* Handler function return codes */
typedef enum xtb_hret_defn { XTB_NOTDEF, XTB_HANDLED, XTB_STOP } xtb_hret;

#define xg_abort()	{fputs("abort\n",stderr);exit(-10);}
#ifdef XGRAPH
#	ifndef DEBUG
#		define abort()	xg_abort()
#	endif
#endif

#ifndef CheckMask
#	define	CheckMask(var,mask)	(((var)&(mask))==(mask))
#endif
#ifndef CheckORMask
#	define	CheckORMask(var,mask)	(((var)&(mask))!=0)
#endif
#ifndef CheckExclMask
#	define	CheckExclMask(var,mask)	(((var)^(mask))==0)
#endif

/* If you have an ANSI compiler,  some checking will be done */
#ifdef __STDC__
#define DECLARE(func, rtn, args)	extern rtn func args
typedef void *xtb_data;
#else
#define DECLARE(func, rtn, args)	extern rtn func ()
typedef char *xtb_data;
#endif

typedef enum FrameType { xtb_None= 0, xtb_TO, xtb_TI, xtb_TI2, xtb_BT, xtb_BR, xtb_SR, xtb_ERR, xtb_BK, xtb_User, FrameTypes } FrameType;

/* Basic callback-specific information, shared by all types of widgets. The type
 \ (button, slider, text-input etc) is specified by the FrameType field.
 \ NB: The callback routine should be made more generic, with the type-specific
 \ arguments passed through a void* pointer. Right now, the 2 basic arguments
 \ are there, but every type adds its own arguments, usually before the val
 \ argument (user data; a xtb_data* or xtb_data ..). This is prone to cause
 \ crashes in the case that e.g. text-input callback would get called through
 \ an event for a text-output frame. Of course, this should not happen when
 \ the library is used as it should be...
 */
typedef struct xtb_registry_info{
	struct xtb_frame *frame;
	xtb_data val;		/* User info          */
	FNPTR( func, xtb_hret, (Window win, xtb_data val) );
					/* Function to call   */
	FrameType type;
} xtb_registry_info;

/* Basic return value */
typedef struct xtb_frame {
    Window win, parent;
    int x_loc, y_loc;
    unsigned int width, height, border;
	  /* mapped: is this frame's window currently mapped?
	   \ enabled: valid only for widget frames. Whether a widget can be manipulated by the used.
	   */
	int tabset, mapped, enabled;
	char *description;
	int frames;
	struct xtb_frame **framelist;
/* 	xtb_data info;	*/
	xtb_registry_info *info;
	void (*redraw)(Window win), (*destroy)(Window win,xtb_data info);
	int (*chfnt)(Window win, XFontStruct *f, XFontStruct *alt_f);
	int destroy_it;
} xtb_frame;

DECLARE(xtb_init, void, (Display *disp, int scrn,
			 unsigned long foreground,
			 unsigned long background,
			 XFontStruct *font, XFontStruct *font2, int parent_background));
   /* Initializes mini-toolbox */

#ifndef XGRAPH
#	define GETCOLOR_USETHIS	((void*)-1)
	extern XColor GetThisColor;
	extern int GetColor( char *name, Pixel *pix);
	extern void init_xtb_standalone(char *name, Display *disp, int scrn, XVisualInfo *vi, Colormap colourmap, int depth, unsigned long fg, unsigned long bg );
#endif

DECLARE( Boing, int, (int duration));

extern int xtb_XSync( Display *disp, Bool discard);

/* Process a textual representation of a linestyle:
 \ "1234" -> (char*) { 1, 2, 3, 4 }
 */
DECLARE( xtb_ProcessStyle, int, (char *style, char *buf, int maxbuf) );

/*
 * Basic event handling
 */

DECLARE(xtb_register, void, (xtb_frame *frame, Window win,
			     xtb_hret (*func)(XEvent *evt, xtb_data info, Window parent),
			     xtb_registry_info *info));
   /* Registers call-back function */

/* Change or update a window's existing registry:	*/
#define XTB_UPDATE_REG_FRAME	(1<<0)
#define XTB_UPDATE_REG_FUNC		(1<<1)
#define XTB_UPDATE_REG_INFO		(1<<2)
DECLARE( xtb_update_registry, int, ( Window win,
	xtb_frame *frame, xtb_hret (*func)(XEvent *, xtb_data, Window), xtb_registry_info *info, int mask));

DECLARE(xtb_lookup, xtb_data, (Window win));
   /* Returns data associated with window */
DECLARE(xtb_lookup_frame, xtb_frame*, (Window win));
   /* Returns frame associated with window */

extern int xtb_modifier_state, xtb_button_state, xtb_clipboard_copy;
DECLARE( xtb_modifiers_string, char*, (int mask));

DECLARE(xtb_dispatch, xtb_hret, (Display *disp, Window window, int frames, xtb_frame *framelist, XEvent *evt));
   /* Dispatches events for mini-toolbox */

DECLARE(xtb_unregister, int, (Window win, xtb_registry_info **info));
   /* Unregisters a call-back function */

/*
 * Formatting support
 */

#define MAX_BRANCH	50

typedef enum xtb_fmt_types_defn { W_TYPE, A_TYPE, NO_TYPE } xtb_fmt_types;
typedef enum xtb_fmt_dir_defn { HORIZONTAL, VERTICAL } xtb_fmt_dir;
  /* XTB_JUST adjusts the spacing of elements according to the width or height
   \ of the last branch (i.e. previous on the same level) adjusted with a _J extension, in case of
   \ horizontal respectively vertical formatting, and is identical to XTB_CENTER
   \ in the other direction.
   */
typedef enum xtb_just_defn {
    XTB_CENTER=0, XTB_LEFT, XTB_RIGHT, XTB_TOP, XTB_BOTTOM, XTB_JUST,
    XTB_CENTER_J, XTB_LEFT_J, XTB_RIGHT_J, XTB_TOP_J, XTB_BOTTOM_J, XTB_JUST_J
} xtb_just;

typedef struct xtb_fmt_widget_defn {
    xtb_fmt_types type;		/* W_TYPE */
    xtb_frame *w;
} xtb_fmt_widget;

typedef struct xtb_fmt_align_defn {
    xtb_fmt_types type;		/* A_TYPE */
    xtb_fmt_dir dir;		/* HORIZONTAL or VERTICAL */
    int padding;		/* Outside padding        */
    int interspace;		/* Internal padding       */
    xtb_just just;		/* Justification          */
    int ni;			/* Number of items */
	int width, height, Nmapped;
    union xtb_fmt *items[MAX_BRANCH]; /* Branches themselves */
} xtb_fmt_align;

typedef union xtb_fmt {
    xtb_fmt_types type;		/* W_TYPE or A_TYPE */
    xtb_fmt_widget wid;
    xtb_fmt_align align;
} xtb_fmt;

// RJVB 20110427: NE must be compatible with a pointer-to-address to avoid issues on 64bit!
#define NE	((xtb_fmt *) 0)

extern int xtb_fmt_tabset;

DECLARE(xtb_w, xtb_fmt *, (xtb_frame *w));
DECLARE(xtb_ws, xtb_fmt *, (xtb_frame *w, int tabset));
   /* Returns formatting structure for frame */
DECLARE(xtb_hort, xtb_fmt *, (xtb_just just, int padding, int interspace, VA_DCL) );
   /* Varargs routine for horizontal formatting */
DECLARE(xtb_vert, xtb_fmt *, (xtb_just just, int padding, int interspace, VA_DCL ) );
   /* Varargs routine for vertical formatting */
DECLARE(xtb_hort_cum, xtb_fmt *, (xtb_just just, int padding, int interspace, xtb_fmt *frame));
   /* Cumulative routine for horizontal formatting (end calling sequence with frame==NULL) */
DECLARE(xtb_vert_cum, xtb_fmt *, (xtb_just just, int padding, int interspace, xtb_fmt *frame));
   /* idem, vertical	*/
DECLARE(xtb_fmt_do, xtb_fmt *, (xtb_fmt *def, unsigned *w, unsigned *h));
   /* Carries out formatting */
DECLARE(xtb_mv_frames, void, (int nf, xtb_frame frames[]));
   /* Actually moves widgets */
DECLARE(xtb_mv_frame, void, (xtb_frame *frames));
   /* Actually moves 1 widget */
DECLARE(xtb_fmt_free, void, (xtb_fmt *def));
   /* Frees resources claimed by xtb_w, xtb_hort, and xtb_vert */
DECLARE(xtb_select_frames_tabset, void, (int nf, xtb_frame frames[], int tabset, int (*select)(int, int, int)));
   /* Map frames that have their tabset==tabset, or for which (*select) returns True; unmap the others */

/*
 * Command button frame
 */

extern XEvent *xtb_bt_event;
DECLARE(xtb_bt_new, void, (Window win, char *text,
			   xtb_hret (*func)(Window win, int state,
				       xtb_data val),
			   xtb_data val,
			   xtb_frame *frame));
   /* Creates new button  */
#define XTB_TOP_LEFT	0
#define XTB_TOP_RIGHT	1
#define XTB_CENTERED	2
DECLARE(xtb_bt_new2, void, (Window win, char *text, int pos,
			   xtb_hret (*func)(Window win, int state,
				       xtb_data val),
			   xtb_data val,
			   xtb_frame *frame));
   /* Creates new button  */

DECLARE(xtb_bt_get, int, (Window win, xtb_data *stuff));
   /* Returns state of button */
DECLARE(xtb_bt_chfnt, int, (Window win, XFontStruct *, XFontStruct*));
DECLARE(xtb_bt_set, int, (Window win, int val, xtb_data stuff));
   /* Sets state of button */
DECLARE(xtb_bt_set2, int, (Window win, int val, int val2, xtb_data stuff));
	/* Sets the state of the button, and its 2nd state	*/
DECLARE(xtb_bt_swap, int, (Window win));
DECLARE(xtb_bt_get_text, char *, (Window win));
DECLARE(xtb_bt_set_text, int, (Window win, int val, char *text, xtb_data stuff));
   /* Sets state and/or text of button */
DECLARE(xtb_bt_redraw, void, (Window win));
	/* (Re)draws a button	*/
DECLARE(xtb_bt_del, void, (Window win, xtb_data *info));
   /* Deletes a button */
DECLARE(xtb_find_next_named_button, xtb_frame*,
	(xtb_frame *framelist, int N, xtb_frame **current, int *currentN, xtb_frame **subcurrent, int *subcurrentN,
	char *pattern, int firstchar));

/*
 * Button row frame - built on top of buttons
 */

#define BR_XPAD   	2
#define BR_YPAD   	2
#define BR_INTER	2

DECLARE(xtb_br_new, void, (Window win, int cnt, char *lbls[], int initNr,
			   xtb_hret (*func)(Window win, int prev,
					    int current, xtb_data val),
			   xtb_data val,
			   xtb_frame *frame));
   /* Creates a new button row frame */
DECLARE(xtb_br2Dr_new, void, (Window win, int cnt, char *lbls[], int rows, int initNr,
			   xtb_hret (*func)(Window win, int prev,
					    int current, xtb_data val),
			   xtb_data val,
			   xtb_frame *frame));
   /* Creates a new 2D button row frame, with <rows> frame of (cnt+1)/rows buttons. */
DECLARE(xtb_br2Dc_new, void, (Window win, int cnt, char *lbls[], int columns, int initNr,
			   xtb_hret (*func)(Window win, int prev,
					    int current, xtb_data val),
			   xtb_data val,
			   xtb_frame *frame));
   /* Creates a new 2D button row frame, with <columns> frame of (cnt+1)/columns buttons. */
extern void xtb_br2D_new( Window win, int cnt, char *lbls[], int columns, int rows,
	FNPTR( format_fun, xtb_fmt*, (xtb_frame *br_frame, int cnt, xtb_frame **buttons, xtb_data val) ),
	int initNr,
	FNPTR( func, xtb_hret, (Window, int, int, xtb_data) ),
	xtb_data val, xtb_frame *frame);
  /* Idem, either distributing the buttons as columns x rows, or by the user-supplied format function. */

DECLARE(xtb_br_set, int, (Window win, int button));
DECLARE(xtb_br_get, int, (Window win));
   /* Returns currently selected button */
DECLARE(xtb_br_chfnt, int, (Window win, XFontStruct *, XFontStruct*));
DECLARE(xtb_br_redraw, void, (Window win));
DECLARE(xtb_br_del, void, (Window win));
   /* Deletes a button row */

/*
 * Text output (label) frames
 */

DECLARE(xtb_to_new, void, (Window win, char *text, int pos,
			   XFontStruct **ft, XFontStruct **ft2, xtb_frame *frame));
DECLARE(xtb_to_new2, void, (Window win, char *text, int maxwidth, int pos,
			   XFontStruct **ft, XFontStruct **ft2, xtb_frame *frame));
   /* Create new text output frame */
DECLARE(xtb_to_set, void, (Window win, char *text));
DECLARE(xtb_to_redraw, void, (Window win));
DECLARE(xtb_to_del, void, (Window win));

/*
 * Text input (editable text) frames
 */

#define MAXCHBUF	1024

DECLARE(xtb_ti_new, void, (Window win, char *text, int maxwidth, int maxchar,
			   xtb_hret (*func)(Window win, int ch,
					    char *textcopy, xtb_data val),
			   xtb_data val, xtb_frame *frame));
   /* Creates a new text input frame */		   

DECLARE( xtb_ti2_new, void, ( Window win, char *text, int maxwidth, int maxchar,
	FNPTR( tifunc, xtb_hret, (Window, int, char *, xtb_data) ),
	char *bttext,
	FNPTR( btfunc, xtb_hret, (Window, int, xtb_data) ),
	xtb_data val, xtb_frame *frame
));
DECLARE( xtb_ti2_ti_frame, xtb_frame*, (Window win) );

DECLARE( xtb_ti_scroll_left, void, (Window win, int places));
DECLARE( xtb_ti_scroll_right, void, (Window win, int places));
#ifdef STATIC_TI_BUFLEN
DECLARE(xtb_ti_get, void, (Window win, char text[MAXCHBUF], xtb_data val));
   /* Returns state of text input frame */
#else
DECLARE(xtb_ti_get, void, (Window win, char *text, xtb_data *val));
   /* Returns state of text input frame */
#endif
DECLARE( xtb_ti_length, int, (Window win, xtb_data *val));
DECLARE(xtb_ti_chfnt, int, (Window win, XFontStruct *, XFontStruct*));
DECLARE(xtb_ti_set, int, (Window win, char *text, xtb_data val));
   /* Sets the state of text input frame */
DECLARE(xtb_ti_ins, int, (Window win, int ch));
   /* Inserts character onto end of text input frame */
DECLARE(xtb_ti_dch, int, (Window win));
   /* Deletes character from text input point, leftwards (BACKSPACE) */
DECLARE(xtb_ti_dch_right, int, (Window win));
   /* Deletes character from text input point, rightwards (DELETE) */
DECLARE(xtb_ti_redraw, void, (Window win));
DECLARE(xtb_ti2_redraw, void, (Window win));
DECLARE(xtb_ti_del, void, (Window win, xtb_data *info));
   /* Deletes an text input frame */

/*
 * Block frame
 */

DECLARE(xtb_bk_new, void, (Window win, unsigned width, unsigned height,
			   xtb_frame *frame));
   /* Makes a new block frame */
DECLARE(xtb_bk_del, void, (Window win));
   /* Deletes a block frame */

/*
 \ Slideruler frame. (C) RJB 1996
 */
DECLARE( xtb_sr_set, void, (Window win, double value, xtb_data *val) );
DECLARE( xtb_sr_get, double, (Window win, int *position, xtb_data *val) );
DECLARE( xtb_sr_set_scale, void, (Window win, double minval, double maxval, xtb_data *val) );
DECLARE( xtb_sr_set_scale_integer, void, (Window win, double minval, double maxval, int int_flag, xtb_data *val) );
DECLARE( xtb_sr_get_scale, void, (Window win, double *minval, double *maxval, xtb_data *val ) );
DECLARE( xtb_sr_set_integer, void, (Window win, int int_flag, xtb_data *val ) );
DECLARE( xtb_sr_get_integer, int, (Window win, xtb_data *val ) );
DECLARE( xtb_sr_redraw, void, (Window win) );
DECLARE( xtb_sr_del, void, (Window win, xtb_data *info ) );
DECLARE( xtb_sr_chfnt, int, (Window win, XFontStruct *, XFontStruct*));
DECLARE( xtb_sr_new, void, (Window win, double minval, double maxval, double init_val, int size, int vert_flag,
	xtb_hret (*func)(Window win, int pos, double value, xtb_data val),
	xtb_data val, xtb_frame *frame));
DECLARE( xtb_sri_new, void, (Window win, int minval, int maxval, int init_val, int size, int vert_flag,
	xtb_hret (*func)(Window win, int pos, double value, xtb_data val),
	xtb_data val, xtb_frame *frame));

DECLARE( xtb_PsychoMetric_Gray, double, (XColor *rgb) );

DECLARE( xtb_getline, int, (char **source, char **next_line) );
DECLARE( xtb_is_leftalign, int, (char *c));

extern int xtb_box_ign_first_brelease;
DECLARE(xtb_error_box, int, (Window parent, char *mesg, char *title));
DECLARE(xtb_popup_menu, int, (Window parent, char *mesg, char *title, char **text_return, xtb_frame **frame_return));
DECLARE(xtb_popup_get_itemtext, char *, (xtb_frame *menu_frame, int id));
DECLARE(xtb_popup_delete, void, (xtb_frame **menu_frame));

extern Window xtb_input_dialog_inputfield;
extern char *xtb_input_edited_buffer;
DECLARE( xtb_input_dialog, char *, (Window parent, char *initialised_return_string, int tilen, int maxlen, char *mesg, char *title,
		int modal,
		char *help_label, xtb_hret (*help_btn)(Window,int,xtb_data),
		char *help_label2, xtb_hret (*help_btn2)(Window,int,xtb_data),
		char *help_label3, xtb_hret (*help_btn3)(Window,int,xtb_data)
	)
);
/* An experimental, semi-reentrant version of the above routine, for the moment modal-only: */
DECLARE( xtb_input_dialog_r, char *, (Window parent, char *initialised_return_string, int tilen, int maxlen, char *mesg, char *title,
		char *help_label, xtb_hret (*help_btn)(Window,int,xtb_data),
		char *help_label2, xtb_hret (*help_btn2)(Window,int,xtb_data),
		char *help_label3, xtb_hret (*help_btn3)(Window,int,xtb_data)
	)
);

DECLARE( xtb_toggle_greek, int, ( char *text, char *first ) );
DECLARE( xtb_has_greek, char *, (char *text));
DECLARE( xtb_has_backslash, char *, (char *text));
DECLARE( xtb_TextExtents, int, ( XFontStruct *font1, XFontStruct *font2, char *text, int Len, int *dir, int *ascent, int *descent, XCharStruct *bb, Boolean escape_backslash) );
DECLARE( xtb_TextWidth, int, (char *text, XFontStruct *font1, XFontStruct *font2));
DECLARE( XFontWidth, int, (XFontStruct *font) );
DECLARE( xtb_textfilter, char *, ( char *_text, XFontStruct *font, Boolean escape_backslash ));
DECLARE( xtb_DrawString, int, ( Display *t_disp, Window win, GC gc, int x, int y,
	char *text, int Len, XFontStruct *font1, XFontStruct *font2, int text_v_info, Boolean escape_backslash)
);

DECLARE( xtb_describe, char *, (xtb_frame *frame, char *desc));
DECLARE( xtb_describe_s, char *, (xtb_frame *frame, char *desc, int tabset));
DECLARE( xtb_enable, int, (Window win) );
DECLARE( xtb_disable, int, (Window win) );
DECLARE( xtb_enables, int, (Window win, VA_DCL ) );
DECLARE( xtb_disables, int, (Window win, VA_DCL ) );

DECLARE( xtb_CharArray, char**, (int *N, int terminate, VA_DCL ) );

DECLARE( xtb_ParseGeometry, int, (char *geoSpec, XSizeHints *hints, Window win, int update ) );

#ifndef XGRAPH
	extern char *def_str;
	extern Pixel def_pixel;
	extern int def_int;
	extern XFontStruct *def_font;
	extern double def_dbl;

	DECLARE( rd_flag, int, (char *name) );
#endif

#ifdef __cplusplus
}
#endif

#endif /* _XTB_ */
